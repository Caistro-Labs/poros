"""track.py makes a receipt the verifier can rule on, and the verifier rules on gradients."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

from tests.metal import nf4_tiny  # noqa: E402, F401  # puts scripts/metal on the path

# isort: split
import track  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / "poros" / "metal" / "verify.py"


def _script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "metal" / name), *args, "--commit", "abc"],
        check=False, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def _grads(cosines, gpu_over_cpu=1.0):
    return {f"t{i}": {"dot": c * gpu_over_cpu, "gpu_norm": gpu_over_cpu, "cpu_norm": 1.0}
            for i, c in enumerate(cosines)}


def _gradient_verdict(grads):
    from poros.metal import verify

    whole = verify._whole(grads)
    out = verify._gradient_failures("whole", whole, verify.GRAD_COSINE, verify.GRAD_NORM)
    for name, g in grads.items():
        out += verify._gradient_failures(name, g, verify.TENSOR_COSINE, verify.TENSOR_NORM)
    return out


def test_rounding_in_a_tensor_passes_and_a_wrong_gradient_does_not():
    # the worst correct kernels measured: 0.99722 and 2.09% on one tensor, the rest close
    assert _gradient_verdict(_grads([0.99722] + [0.9999] * 167)) == []
    worst = {"dot": 1.0209, "gpu_norm": 1.0209, "cpu_norm": 1.0}
    assert _gradient_verdict({**_grads([0.9999] * 167), "worst": worst}) == []
    # fp32 against bf16: every tensor passes on its own, and the vector does not
    assert "whole" in _gradient_verdict(_grads([0.9986] * 168))[0]
    assert "t0" in _gradient_verdict(_grads([0.966] + [0.9999] * 167))[0]
    worst = {"dot": 1.06, "gpu_norm": 1.06, "cpu_norm": 1.0}
    assert "worst" in " ".join(_gradient_verdict({**_grads([0.9999] * 167), "worst": worst}))
    # a whole gradient off by scale
    assert "whole" in _gradient_verdict(_grads([0.9999] * 168, gpu_over_cpu=1.02))[0]
    # a torn number names the whole as well as its tensor
    torn = {**_grads([0.9999] * 3), "t1": {"dot": None, "gpu_norm": 1.0, "cpu_norm": 1.0}}
    assert len(_gradient_verdict(torn)) == 2


def _run_folder(path, losses, **settings):
    path.mkdir()
    config = dict.fromkeys(track.SETTINGS, 1)
    config.update(model_name_or_path="Qwen/Qwen2.5-7B", max_seq_length=256, learning_rate=2e-5,
                  lora_dropout=0.0, store_path=str(path), **settings)
    (path / "poros_config.json").write_text(json.dumps(config))
    events = [{"event_type": "setup", "step": None, "data": {}}]
    events += [{"event_type": "step", "step": i, "data": {"loss": loss, "tokens": 5.0}}
               for i, loss in enumerate(losses)]
    (path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def test_track_writes_a_receipt_the_verifier_stands_behind(tmp_path):
    mac = _run_folder(tmp_path / "mac", [2.0] * 20)
    cuda = _run_folder(tmp_path / "cuda", [2.0] * 19 + [2.08])
    done = _script("track.py", "--mac", str(mac), "--cuda", str(cuda), "--out", str(tmp_path / "t"))
    assert done.returncode == 0, done.stdout + done.stderr
    receipt = tmp_path / "t" / "receipt.json"
    assert json.loads(receipt.read_text())["max_relative"] == pytest.approx(0.08 / 2.08)
    stored = subprocess.run([sys.executable, str(VERIFY), str(receipt)], check=False,
                            capture_output=True, text=True)
    assert stored.returncode == 0, stored.stdout


def test_track_fails_a_run_that_went_its_own_way_or_stopped_early(tmp_path):
    mac = _run_folder(tmp_path / "mac", [2.0] * 20)
    wrong = _run_folder(tmp_path / "wrong", [2.0] * 19 + [20.0])
    out = str(tmp_path / "w")
    done = _script("track.py", "--mac", str(mac), "--cuda", str(wrong), "--out", out)
    assert done.returncode == 1 and "over 0.05" in done.stdout
    short = _run_folder(tmp_path / "short", [2.0] * 3)
    out = str(tmp_path / "s")
    done = _script("track.py", "--mac", str(mac), "--cuda", str(short), "--out", out)
    assert done.returncode == 1 and "must hold 20" in done.stdout
