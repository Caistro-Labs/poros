"""The precision kit's control arm (scripts/moe_precision/moeprec_hook.py, MOEPREC_OPTIMIZER
torch-bf16): the release engine with torch's AdamW substituted for its optimizer, stepping the
tiny MoE model's bf16 expert adapters in bf16. Its weights are a plain torch AdamW run's, bit for
bit (the rule before 1.0.0), its record says the substitution, and the kit's sampler reports the
unchanged-weight share and v against exact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()

from tests.moe.test_native_entry import _loader, _model, _trainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "moe_precision"))
import moeprec_hook as hook  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_moe_control")
    _model().save_pretrained(path)
    return path


def _keep_originals(monkeypatch):
    """The hook patches the engine's classes for its whole process; here each is put back."""
    from poros.train import expert_adamw, loop

    monkeypatch.setattr(expert_adamw, "build_optimizer", expert_adamw.build_optimizer)
    monkeypatch.setattr(loop._Loop, "_record_optimizer", loop._Loop._record_optimizer)
    monkeypatch.setattr(loop._Loop, "_initialize_optimizer", loop._Loop._initialize_optimizer)


def _weights(trainer):
    return {n: p.detach().clone() for n, p in trainer.model.named_parameters()
            if p.requires_grad}


def test_the_control_arm_is_a_plain_torch_adamw_run_bit_for_bit(checkpoint, tmp_path, monkeypatch):
    knobs = dict(max_steps=4, learning_rate=1e-3)
    _keep_originals(monkeypatch)
    sample = tmp_path / "sample.jsonl"
    hook._torch_optimizer_control()
    hook._sampling(str(sample))
    control = _trainer(checkpoint, tmp_path / "control", **knobs)
    assert type(control.optimizer) is torch.optim.AdamW
    assert all(p.dtype == torch.bfloat16 for p in control.trainable_parameters)
    ran = control.config.ran_compute["expert_optimizer"]
    assert ran == {"substituted": "torch.optim.AdamW",
                   "by": "scripts/moe_precision/moeprec_hook.py", "dtypes": ["bfloat16"]}
    assert "moeprec control: torch.optim.AdamW" in control.config.compute_reason
    control.train(_loader())
    got = _weights(control)
    control.save(str(tmp_path / "control" / "saved"))
    said = json.loads((tmp_path / "control" / "saved" / "optimizer.json").read_text())
    assert said["trainable_dtype"] == "bfloat16" and said["implementation"] is None
    assert said["rule"] != "poros.expert-adamw.v4" and "torch" in said["rule"]

    monkeypatch.undo()
    _keep_originals(monkeypatch)
    from poros.train import expert_adamw

    def plain(named, config):  # the test's own torch AdamW, not the hook's
        return torch.optim.AdamW([p for _, p in named], lr=config.learning_rate,
                                 betas=(config.adam_beta1, config.adam_beta2),
                                 eps=config.adam_epsilon, weight_decay=config.weight_decay)

    monkeypatch.setattr(expert_adamw, "build_optimizer", plain)
    reference = _trainer(checkpoint, tmp_path / "plain", **knobs)
    reference.train(_loader())
    want = _weights(reference)
    assert got.keys() == want.keys()
    for name in want:
        assert torch.equal(got[name], want[name]), name

    monkeypatch.undo()  # and the new engine's own optimizer lands elsewhere: the arms differ
    new = _trainer(checkpoint, tmp_path / "new", **knobs)
    new.train(_loader())
    assert any(not torch.equal(p, want[n]) for n, p in _weights(new).items())

    lines = [json.loads(x) for x in sample.read_text().splitlines()]
    assert [x["update"] for x in lines] == [1, 2, 3, 4]
    assert all(x["optimizer"] == "AdamW" and x["dtype"] == "torch.bfloat16" for x in lines)
    assert all(0 <= x["weight_unchanged"] <= 1 and x["v_over_exact_median"] is not None
               for x in lines)


def test_an_unknown_optimizer_arm_is_refused(monkeypatch):
    monkeypatch.setenv("MOEPREC_OPTIMIZER", "sgd")
    monkeypatch.setattr(sys, "argv", ["moeprec_hook.py", "--help"])
    with pytest.raises(SystemExit, match="knows 'torch-bf16' only"):
        hook.main()


def _arm(folder: Path, commit: str, arm: str, loss=2.0, ratio=1.0):
    folder.mkdir(parents=True)
    events = [{"event_type": "step", "step": 1, "data": {"loss": loss, "grad_norm": 0.5}}]
    events += [{"event_type": "eval", "step": s, "data": {"eval_loss": 1.0}} for s in (0, 25)]
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (folder / "sample.jsonl").write_text("".join(json.dumps(
        {"update": u, "v_over_exact_median": ratio, "weight_unchanged": 0.1,
         "optimizer": "AdamW"}) + "\n" for u in range(1, 30)))
    (folder / ".kit_commit").write_text(commit + "\n")
    (folder / ".kit_arm").write_text(arm + "\n")
    return folder


def _verdict(tmp_path, *, control_commit="1a8d7e912ab", control_loss=2.0):
    import subprocess

    runs = {k: _arm(tmp_path / k, "1a8d7e912ab", k) for k in ("new", "fp32", "fp32o")}
    ctl = _arm(tmp_path / "ctl", control_commit, "control (torch.optim.AdamW substituted)",
               loss=control_loss, ratio=40.0)
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "moe_precision" / "compare.py"), "--new",
         str(runs["new"]), "--fp32", str(runs["fp32"]), "--fp32-other", str(runs["fp32o"]),
         "--control", str(ctl), "--commit", "1a8d7e912"], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"})
    return done.returncode, json.loads(done.stdout)


def test_the_verdict_needs_every_arm_on_the_release_commit_and_the_control_on_its_engine(
        tmp_path):
    code, said = _verdict(tmp_path / "a")
    assert code == 0 and said["check_4_release_commit"] and said["check_5_control_same_engine"]
    assert said["control"]["v_over_exact_last"] == 40.0  # reported, not judged
    code, said = _verdict(tmp_path / "b", control_commit="8dbba085a00")
    assert code == 1 and not said["check_4_release_commit"]
    assert said["commits"]["control"].startswith("8dbba085a")
    code, said = _verdict(tmp_path / "c", control_loss=2.5)
    assert code == 1 and not said["check_5_control_same_engine"]
