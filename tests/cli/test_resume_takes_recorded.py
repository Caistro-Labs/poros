"""A resume continues its own run: what its config leaves unset is what the run recorded.

A config with no lora_rank or lora_alpha, only `init_adapter:` naming a finished
run whose adapter is rank 8, alpha 24, started right (the start took the adapter's shape). Killed
after checkpoint-3, `poros train cut.yaml --resume-from latest` with the same file was refused:
"checkpoint-3 trained with lora_rank 8, and this run has 16". A resume does not start from the
adapter, so nothing took its shape, and the config's defaults were compared. Now each setting
the resume compares, and the shape a start adopts, is the checkpoint's where the config sets
none; one the config sets is compared, and refused on a real mismatch, as before.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.cli.test_run_flow_wording import MODEL, _chats
from tests.cli.test_train_runs import home  # noqa: F401  (the simulated machine)
from tests.conftest import checkout_env

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _poros(cwd: Path, *argv: str) -> subprocess.CompletedProcess:
    env = checkout_env({**os.environ, "POROS_HOME": str(cwd / "home"), "HF_HUB_OFFLINE": "1",
                        "POROS_DEV_SIMULATE": "1", "POROS_DEV_SIMULATE_STEP": "0.001",
                        "POROS_DEV_SIMULATE_LOAD": "0", "NO_COLOR": "1", "COLUMNS": "200"})
    for name in ("POROS_OUTPUT", "POROS_AGENT"):
        env.pop(name, None)
    return subprocess.run([sys.executable, "-c", "from poros.cli.main import cli; cli()", *argv],
                          capture_output=True, text=True, env=env, cwd=cwd,
                          stdin=subprocess.DEVNULL)


def _yaml(cwd: Path, **extra) -> Path:
    """cut.yaml: no rank, no alpha, only the adapter it starts from."""
    lines = [f"model_name_or_path: {MODEL}", "dataset: ./chats.jsonl",
             "init_adapter: ./src", "output_dir: ./cut", "max_steps: 8", "save_steps: 3",
             *(f"{k}: {json.dumps(v)}" for k, v in extra.items())]
    path = cwd / "cut.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def cut(tmp_path, request):
    """A finished run's adapter in ./src (rank 8, alpha 24), and a run of cut.yaml from it cut
    short after checkpoint-3: its checkpoint records what the run resolved (the simulated
    trainer writes empty checkpoints; a real one writes these settings, as the run took them)."""
    if getattr(request, "param", "recipe") == "recipe":
        _chats(tmp_path, 30)  # chats: SFT, a recipe's loop
    else:  # text rows: the classic loop
        (tmp_path / "chats.jsonl").write_text(
            "".join(json.dumps({"text": f"row {i} " * 20}) + "\n" for i in range(30)))
    src = tmp_path / "src"
    src.mkdir()
    (src / "poros_config.json").write_text(json.dumps(
        {"model_name_or_path": MODEL, "lora_rank": 8, "lora_alpha": 24,
         "lora_target_modules": TARGETS}))
    (src / "adapter_config.json").write_text(json.dumps(
        {"r": 8, "lora_alpha": 24, "base_model_name_or_path": MODEL}))
    first = _poros(tmp_path, "train", str(_yaml(tmp_path)), "--yes", "--json")
    assert first.returncode == 0, first.stdout + first.stderr
    took = json.loads((tmp_path / "cut" / "config.resolved.json").read_text())
    assert (took["lora_rank"], took["lora_alpha"]) == (8, 24)  # the start took the adapter's
    for later in ("checkpoint-6", "checkpoint-7"):  # cut short: as a kill -9 after step 3
        shutil.rmtree(tmp_path / "cut" / later, ignore_errors=True)
    kept = tmp_path / "cut" / "checkpoint-3"
    assert kept.is_dir()
    (kept / "poros_config.json").write_text(json.dumps(
        {k: v for k, v in took.items() if not k.startswith("_")}))
    (tmp_path / "cut" / "config.resolved.json").unlink()
    return tmp_path


@pytest.mark.parametrize("cut", ["recipe", "classic"], indirect=True)
@pytest.mark.parametrize("which", ["latest", "named"])
@pytest.mark.parametrize("source", ["kept", "deleted"])
def test_the_same_config_resumes_its_own_run(cut, which, source, request):
    if source == "deleted":  # a resume reads its own checkpoint, not the adapter it began from
        shutil.rmtree(cut / "src")
    ckpt = "latest" if which == "latest" else str(cut / "cut" / "checkpoint-3")
    said = _poros(cut, "train", str(_yaml(cut)), "--resume-from", ckpt, "--yes", "--json")
    assert said.returncode == 0, said.stdout + said.stderr  # was E_CONFIG: "this run has 16"
    now = json.loads((cut / "cut" / "config.resolved.json").read_text())
    assert (now["lora_rank"], now["lora_alpha"], now["lora_target_modules"]) == (8, 24, TARGETS)
    assert (cut / "cut" / "checkpoint-7").is_dir()  # it trained on to its last step
    loop = request.node.callspec.params["cut"]
    assert (now.get("trained_with") == "classic") == (loop == "classic"), now.get("trained_with")


@pytest.mark.parametrize("field, value, was", [("lora_rank", 16, 8),
                                               ("gradient_accumulation_steps", 2, None),
                                               ("seed", 7, None)])
def test_a_setting_the_config_gives_is_still_compared_and_refused(cut, field, value, was):
    recorded = json.loads((cut / "cut" / "checkpoint-3" / "poros_config.json").read_text())
    was = recorded[field] if was is None else was
    assert was != value
    said = _poros(cut, "train", str(_yaml(cut, **{field: value})), "--resume-from", "latest",
                  "--yes", "--json")
    assert said.returncode == 2, said.stdout + said.stderr
    err = json.loads(said.stdout)["error"]
    assert err["code"] == "E_CONFIG"
    assert (f"checkpoint-3 trained with {field} {was!r}, and this run has {value!r}"
            in " ".join(err["message"].split()))


@pytest.mark.parametrize("field", ["gradient_accumulation_steps", "per_device_batch_size",
                                   "seed", "max_seq_length", "lora_target_modules", "lora_alpha"])
def test_each_setting_left_unset_takes_the_recorded_value(cut, field):
    """One per class: the batch shape and order (accumulation, batch, seed), the length, the
    adapter's shape (targets, alpha): a value the run recorded that is not the default."""
    ckpt = cut / "cut" / "checkpoint-3" / "poros_config.json"
    recorded = json.loads(ckpt.read_text())
    other = {"gradient_accumulation_steps": 3, "per_device_batch_size": 2, "seed": 11,
             "max_seq_length": 384, "lora_target_modules": ["q_proj", "v_proj"],
             "lora_alpha": 40}[field]
    ckpt.write_text(json.dumps({**recorded, field: other}))
    said = _poros(cut, "train", str(_yaml(cut)), "--resume-from", "latest", "--yes", "--json")
    assert said.returncode == 0, said.stdout + said.stderr
    assert json.loads((cut / "cut" / "config.resolved.json").read_text())[field] == other


def test_adopt_recorded_takes_only_what_the_config_leaves_unset(tmp_path):
    from poros.core.config import PorosConfig
    from poros.preflight.resume import adopt_recorded

    (tmp_path / "poros_config.json").write_text(json.dumps(
        {"lora_rank": 8, "lora_alpha": 24, "seed": 3, "max_seq_length": 0,
         "learning_rate": 0.5}))
    cfg = PorosConfig(model_name_or_path="m", seed=5)
    took = adopt_recorded(str(tmp_path), cfg)
    assert took == {"lora_rank": 8, "lora_alpha": 24}  # seed is the config's; 0 is no length
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.seed, cfg.max_seq_length) == (8, 24, 5, 2048)
    assert cfg.learning_rate != 0.5  # not a setting a resume compares or a start adopts
    assert adopt_recorded(None, cfg) == {} and adopt_recorded(str(tmp_path / "none"), cfg) == {}


def test_runs_continue_of_a_from_run_keeps_the_adapters_shape(home, monkeypatch,  # noqa: F811
                                                               capsys):
    """The verb form's continue: a `--from` run of a rank-8,
    alpha-24 adapter, stopped after a checkpoint, continues at that shape."""
    from poros import runs
    from tests.cli.test_run_flow_wording import _interrupt_at, _screen

    monkeypatch.delenv("POROS_OUTPUT", raising=False)
    code, _ = _screen(["train", MODEL, "--data", _chats(home, 30), "--steps", "4", "--rank", "8",
                       "lora_alpha=24", "--yes"], home, capsys)
    assert code == 0
    _interrupt_at(monkeypatch, 5)
    code, _ = _screen(["train", "--from", "last", "--data", _chats(home, 40, "more.jsonl"),
                       "--steps", "9", "save_steps=3", "--yes"], home, capsys)
    cut = runs.all_runs()
    (new,) = [r for r in cut if r.parent]
    assert code == 130 and new.state == "cancelled"
    _interrupt_at(monkeypatch, None)
    code, out = _screen(["runs", "continue", new.id], home, capsys)
    assert code == 0, out
    now = json.loads((Path(new.output_dir) / "config.resolved.json").read_text())
    assert (now["lora_rank"], now["lora_alpha"]) == (8, 24)
