"""Held-out rows on the engine's own loop (``PorosTrainer.train(..., eval_dataset=...)``), on the
tiny CPU engine: scored before the first step, every ``eval_steps``, at each checkpoint and after
the last, as a recipe's run scores them; and a run that scores trains exactly the bytes of one
that does not, through a resume too."""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import pytest

STATE = ("adapter_model.safetensors", "adapter_weights.pt", "optimizer_state.pt", "rng_state.pt",
         "lr_scheduler_state.pt")


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _trainer(torch, outdir, **cfg):
    """With dropout on: every draw from the RNG moves what the next step trains."""
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    torch.manual_seed(0)
    model, _ = _tiny_peft_qwen2(torch, dropout=0.1)
    tr = _make_trainer(torch, model, str(outdir), max_steps=7)
    tr.config.lora_dropout = 0.1
    for k, v in {"learning_rate": 1e-2, "save_steps": 2, "eval_steps": 0,
                 "save_total_limit": 10, **cfg}.items():
        setattr(tr.config, k, v)
    return tr


def _rows(torch, n=32, seed=0):
    from tests.test_train_loop_cpu import _loader

    return _loader(torch, 256, n=n, seed=seed)


def _held(torch):
    """Held-out rows through a loader with no generator of its own: starting it draws from the
    CPU generator, the draw an eval pass must not leave behind."""
    rows = _rows(torch, n=5, seed=9).dataset
    return torch.utils.data.DataLoader(rows, batch_size=2, shuffle=False)


def _events(outdir) -> list:
    return [json.loads(x) for x in (Path(outdir) / "events.jsonl").read_text().splitlines()]


def _trained(tr, outdir) -> dict:
    """What a run trained, by digest: each step's loss and gradient norm, the adapter, each
    checkpoint's state files (its score aside), and the generator it ends with."""
    torch = pytest.importorskip("torch")
    steps = [(e["step"], e["data"]["loss"]) for e in _events(outdir)
             if e["event_type"] == "step"]
    params = b"".join(p.detach().numpy().tobytes() for _, p in sorted(
        tr.model.named_parameters()) if p.requires_grad)
    files = {f"{d.name}/{f}": hashlib.sha256((d / f).read_bytes()).hexdigest()
             for d in sorted(Path(outdir).glob("checkpoint-*")) for f in STATE
             if (d / f).exists()}
    return {"steps": steps, "grad_norms": list(tr.history.grad_norms),
            "adapter": hashlib.sha256(params).hexdigest(), "files": files,
            "rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def _scores(outdir) -> dict:
    return {d.name: json.loads((d / "eval.json").read_text())
            for d in sorted(Path(outdir).glob("checkpoint-*")) if (d / "eval.json").exists()}


def _evals(outdir) -> list:
    return [e for e in _events(outdir) if e["event_type"] == "eval"]


def test_a_run_scored_at_each_end_and_checkpoint_trains_the_bytes_of_one_that_is_not(ml,
                                                                                    tmp_path):
    torch = ml
    off = _trainer(torch, tmp_path / "off")
    bare = off.train(_rows(torch), steps=7)
    want = _trained(off, tmp_path / "off")
    on = _trainer(torch, tmp_path / "on")
    out = on.train(_rows(torch), steps=7, eval_dataset=_held(torch))

    assert _trained(on, tmp_path / "on") == want
    assert {f.split("/")[0] for f in want["files"]} == {"checkpoint-2", "checkpoint-4",
                                                       "checkpoint-6"}
    evals = _evals(tmp_path / "on")
    assert [e["step"] for e in evals] == [0, 2, 4, 6]  # before, each save, after: once a step
    assert all(e["data"]["rows"] == 5 for e in evals)
    scored = _scores(tmp_path / "on")
    assert list(scored) == ["checkpoint-2", "checkpoint-4", "checkpoint-6"]
    assert set(scored["checkpoint-2"]) == {"eval_loss", "eval_perplexity", "rows"}
    assert [scored[f"checkpoint-{s}"]["eval_loss"] for s in (2, 4, 6)] == [
        e["data"]["eval_loss"] for e in evals[1:]]
    assert out["eval_loss_start"] == evals[0]["data"]["eval_loss"]
    assert out["eval_loss"] == evals[-1]["data"]["eval_loss"]
    assert out["eval_loss"] < out["eval_loss_start"]
    # a run with no held-out rows scores nothing and says nothing of them
    assert not _evals(tmp_path / "off") and not _scores(tmp_path / "off")
    assert "eval_loss" not in bare and "eval_loss_start" not in bare
    assert bare["eval_history"] == []


def test_a_save_on_an_eval_step_scores_once(ml, tmp_path):
    torch = ml
    tr = _trainer(torch, tmp_path / "a", eval_steps=2)
    tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    assert [e["step"] for e in _evals(tmp_path / "a")] == [0, 2, 4, 6]
    tr = _trainer(torch, tmp_path / "b", eval_steps=1, save_steps=3)
    tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    assert [e["step"] for e in _evals(tmp_path / "b")] == [0, 1, 2, 3, 4, 5, 6]
    assert list(_scores(tmp_path / "b")) == ["checkpoint-3", "checkpoint-6"]


def test_turning_checkpoint_scores_off_keeps_eval_steps(ml, tmp_path):
    torch = ml
    tr = _trainer(torch, tmp_path, eval_checkpoints=False, eval_steps=3)
    tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    assert [e["step"] for e in _evals(tmp_path)] == [0, 3, 6]
    assert list(_scores(tmp_path)) == ["checkpoint-6"]  # 3 was no save, 2 and 4 no score


def test_a_resume_through_a_scored_checkpoint_is_bitwise(ml, tmp_path):
    """Resumed from the save_steps checkpoint of a step that also scores (4), and from a stop:
    each run equals the one never stopped, and neither scores the start again. Before, the loop
    saved that checkpoint before its eval, and the eval's draw from the CPU generator (a loader
    with no generator of its own) was in the run that went on but not in the checkpoint."""
    from poros.core.errors import PorosCancelledError

    torch = ml
    whole = _trainer(torch, tmp_path / "whole")
    whole.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    want = _trained(whole, tmp_path / "whole")

    part = _trainer(torch, tmp_path / "part")
    with pytest.raises(PorosCancelledError) as stopped:
        part.train(_rows(torch), steps=7, eval_dataset=_held(torch),
                   should_stop=lambda: part.global_step == 5)
    assert stopped.value.step == 5
    for name in ("checkpoint-4", "checkpoint-5"):  # the scored save, then the stop's
        resumed = _trainer(torch, tmp_path / "part")
        out = resumed.train(_rows(torch), steps=7, eval_dataset=_held(torch),
                            resume_from=str(tmp_path / "part" / name))
        said = _trained(resumed, tmp_path / "part")
        assert said["adapter"] == want["adapter"] and said["rng"] == want["rng"]
        assert {f: h for f, h in said["files"].items()
                if not f.startswith("checkpoint-5/")} == want["files"]
        trained = [loss for _, loss in want["steps"]]
        assert out["loss_history"] == trained[int(name.split("-")[1]) + 1:]
        assert [e["step"] for e in resumed.history.evals] == [6]  # the start: not again
    assert list(_scores(tmp_path / "part")) == ["checkpoint-2", "checkpoint-4", "checkpoint-6"]


def test_the_isolation_is_what_keeps_the_bytes(ml, tmp_path, monkeypatch):
    """An eval pass that drew from the generators would move the next steps' dropout: with them
    kept, it does not; without, it does (the held-out loader draws as it starts)."""
    from poros.train import loop

    torch = ml
    off = _trainer(torch, tmp_path / "off", save_steps=0)
    off.train(_rows(torch), steps=7)
    want = [loss for _, loss in _trained(off, tmp_path / "off")["steps"]]
    kept = _trainer(torch, tmp_path / "kept", save_steps=0, eval_steps=2)
    assert kept.train(_rows(torch), steps=7, eval_dataset=_held(torch))["loss_history"] == want
    monkeypatch.setattr(loop, "kept_rng", lambda device: contextlib.nullcontext())
    moved = _trainer(torch, tmp_path / "moved", save_steps=0, eval_steps=2)
    assert moved.train(_rows(torch), steps=7, eval_dataset=_held(torch))["loss_history"] != want


def test_evaluate_leaves_the_generators_as_they_were(ml, tmp_path):
    torch = ml
    tr = _trainer(torch, tmp_path)
    before = torch.get_rng_state()
    said = tr.evaluate(_held(torch))
    assert torch.equal(torch.get_rng_state(), before)
    assert set(said) == {"eval_loss", "eval_perplexity"}
