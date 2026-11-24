"""A held-out score at every checkpoint, on the tiny CPU engine: the run it scores trains exactly
the bytes it would unscored, its checkpoints keep the score, and DPO holds out pairs as SFT holds
out rows."""

from __future__ import annotations

import contextlib
import hashlib
import json

import pytest

from poros import data as pdata
from tests.test_run_api import Tok, _events

STATE = ("adapter_model.safetensors", "adapter_weights.pt", "optimizer_state.pt", "rng_state.pt",
         "lr_scheduler_state.pt")


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _run(torch, outdir, **cfg):
    """With dropout on: every draw from the RNG moves what the next step trains."""
    from poros.train.run import Run
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    torch.manual_seed(0)
    model, _ = _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj", "gate_proj"), dropout=0.1)
    tr = _make_trainer(torch, model, str(outdir), max_steps=7)
    tr.config.lora_dropout = 0.1
    for k, v in {"learning_rate": 1e-2, "save_steps": 2, "eval_steps": 0, **cfg}.items():
        setattr(tr.config, k, v)
    return Run.from_trainer(tr, Tok())


def _rows(n=60):
    rows = [{"messages": [{"role": "user", "content": f"say {i}"},
                          {"role": "assistant", "content": f"{i} {i} {i}"}]} for i in range(n)]
    return pdata.chats(rows, Tok(), batch_size=2).holdout(0.05)


def _trained(run) -> dict:
    """What a run trained, by digest: each step's loss and gradient norm, the adapter, and each
    checkpoint's state files (the score aside, and the config, which names its folder)."""
    torch = pytest.importorskip("torch")
    steps = [(e["step"], e["data"]["loss"]) for e in _events(run) if e["event_type"] == "step"]
    params = b"".join(p.detach().numpy().tobytes() for _, p in sorted(
        run._engine.model.named_parameters()) if p.requires_grad)
    files = {f"{d.name}/{f}": hashlib.sha256((d / f).read_bytes()).hexdigest()
             for d in sorted(run.dir.glob("checkpoint-*")) for f in STATE if (d / f).exists()}
    rng = torch.get_rng_state()
    return {"steps": steps, "grad_norms": list(run._engine.history.grad_norms),
            "adapter": hashlib.sha256(params).hexdigest(), "files": files,
            "rng": hashlib.sha256(rng.numpy().tobytes()).hexdigest()}


def _scores(run) -> dict:
    return {d.name: json.loads((d / "eval.json").read_text())
            for d in sorted(run.dir.glob("checkpoint-*")) if (d / "eval.json").exists()}


def test_a_run_scored_at_every_checkpoint_trains_the_bytes_of_one_that_is_not(ml, tmp_path):
    from poros.recipes import sft

    torch = ml
    data, held = _rows()
    off = _run(torch, tmp_path / "off", eval_checkpoints=False)
    sft.main(off, data, steps=7, eval_data=held)
    a = _trained(off)
    on = _run(torch, tmp_path / "on")
    sft.main(on, data, steps=7, eval_data=held)
    b = _trained(on)
    bare = _run(torch, tmp_path / "bare", eval_checkpoints=False)
    sft.main(bare, data, steps=7)  # no held-out rows at all

    assert b == a == _trained(bare)
    assert set(b["files"]) == {f"checkpoint-{s}/{f}" for s in (2, 4, 6)
                                           for f in ("adapter_model.safetensors",
                                                     "optimizer_state.pt", "rng_state.pt",
                                                     "lr_scheduler_state.pt")}
    # its checkpoints keep their scores; without, only the finished run's is kept
    scored = _scores(on)
    assert list(scored) == ["checkpoint-2", "checkpoint-4", "checkpoint-6"]
    assert list(_scores(off)) == ["checkpoint-6"] and _scores(off)["checkpoint-6"] == scored[
        "checkpoint-6"]
    assert set(scored["checkpoint-2"]) == {"eval_loss", "eval_perplexity", "rows"}
    assert scored["checkpoint-2"]["rows"] == held.rows == 3
    evals = [(e["step"], e["data"]["eval_loss"]) for e in _events(on) if e["event_type"] == "eval"]
    assert [s for s, _ in evals] == [0, 2, 4, 6]  # before, each save, after: once a step
    assert [v for _, v in evals[1:]] == [scored[f"checkpoint-{s}"]["eval_loss"]
                                         for s in (2, 4, 6)]


def test_a_save_on_an_eval_step_scores_once(ml, tmp_path):
    from poros.recipes import sft

    data, held = _rows()
    run = _run(ml, tmp_path, eval_steps=2)
    sft.main(run, data, steps=7, eval_data=held)
    steps = [e["step"] for e in _events(run) if e["event_type"] == "eval"]
    assert steps == [0, 2, 4, 6]
    run = _run(ml, tmp_path / "between", eval_steps=1, save_steps=3)
    sft.main(run, data, steps=7, eval_data=held)
    assert [e["step"] for e in _events(run) if e["event_type"] == "eval"] == [0, 1, 2, 3, 4, 5, 6]
    assert list(_scores(run)) == ["checkpoint-3", "checkpoint-6"]


def test_a_resume_through_a_scored_checkpoint_is_bitwise(ml, tmp_path):
    from poros.core.errors import PorosCancelledError
    from poros.recipes import sft

    torch = ml
    data, held = _rows()
    whole = _run(torch, tmp_path / "whole", eval_checkpoints=False)
    sft.main(whole, data, steps=7, eval_data=held)
    want = _trained(whole)

    part = _run(torch, tmp_path / "part")
    with pytest.raises(PorosCancelledError) as stopped:
        sft.main(part, data, steps=7, eval_data=held, stop=lambda: part.step == 5)
    resumed = _run(torch, tmp_path / "part")
    resumed.load(tmp_path / "part" / "checkpoint-4")  # the scored one, not the stop's
    assert stopped.value.step == 4 and resumed.step == 5
    sft.main(resumed, data, steps=7, eval_data=held)
    said = _trained(resumed)
    assert {k: said[k] for k in ("steps", "adapter", "files", "rng")} == {
        k: want[k] for k in ("steps", "adapter", "files", "rng")}
    assert list(_scores(resumed)) == ["checkpoint-2", "checkpoint-4", "checkpoint-6"]


def test_the_isolation_is_what_keeps_the_bytes(ml, tmp_path, monkeypatch):
    """An eval that drew from the RNG would move the next steps' dropout: with its generators
    kept, it does not; without, it does."""
    from poros.recipes import sft
    from poros.train import run as run_mod

    torch = ml
    data, held = _rows()
    off = _run(torch, tmp_path / "off", eval_checkpoints=False, save_steps=0)
    sft.main(off, data, steps=7, eval_data=held)

    def drawing(run):
        forward = run._engine._forward_eval_logits

        def draws(batch):
            torch.rand(1)
            return forward(batch)

        run._engine._forward_eval_logits = draws
        return run

    want = _trained(off)["steps"]
    kept = drawing(_run(torch, tmp_path / "kept", save_steps=2))
    sft.main(kept, data, steps=7, eval_data=held)
    assert _trained(kept)["steps"] == want
    monkeypatch.setattr(run_mod, "_kept_rng", lambda device: contextlib.nullcontext())
    moved = drawing(_run(torch, tmp_path / "moved", save_steps=2))
    sft.main(moved, data, steps=7, eval_data=held)
    assert _trained(moved)["steps"] != want


def test_turning_it_off_keeps_eval_steps(ml, tmp_path):
    from poros.recipes import sft

    data, held = _rows()
    run = _run(ml, tmp_path, eval_checkpoints=False, eval_steps=3)
    sft.main(run, data, steps=7, eval_data=held)
    assert [e["step"] for e in _events(run) if e["event_type"] == "eval"] == [0, 3, 6]
    assert list(_scores(run)) == ["checkpoint-6"]  # 3 was no save, 2 and 4 no eval


def _pairs(n=60):
    rows = [{"prompt": f"pick {i}", "chosen": f"yes {i}", "rejected": f"no no {i}"}
            for i in range(n)]
    return pdata.pairs(rows, Tok(), batch_size=2).holdout(0.05)


def test_dpo_scores_held_out_pairs_at_each_end_and_checkpoint(ml, tmp_path):
    import math

    from poros.recipes import dpo

    data, held = _pairs()
    assert (data.rows, held.rows) == (57, 3) and held.pairs
    run = _run(ml, tmp_path)
    dpo.main(run, data, steps=7, beta=0.5, eval_data=held)
    evals = [e for e in _events(run) if e["event_type"] == "eval"]
    assert [e["step"] for e in evals] == [0, 2, 4, 6]
    first, last = evals[0]["data"], evals[-1]["data"]
    assert math.isclose(first["eval_loss"], math.log(2), rel_tol=1e-4)  # the policy is its ref
    assert set(first) == {"eval_loss", "prefers", "rows", "held"} and first["rows"] == 3
    assert first["held"] == held.fingerprint  # the rows it scored, named
    assert 0 <= first["prefers"] <= 3 and last["eval_loss"] < first["eval_loss"]
    assert list(_scores(run)) == ["checkpoint-2", "checkpoint-4", "checkpoint-6"]
    assert (run.dir / "reference" / f"{held.fingerprint}.pt").exists()


def test_a_chat_runs_held_out_pass_reads_whether_its_answers_end(ml, tmp_path):
    """The same pass, the same loss: ends only reads the logits it already has."""
    from poros.ends import Ends, answer_ends

    data, held = _rows()
    run = _run(ml, tmp_path)
    plain = run.evaluate(held)
    at = answer_ends(held, Tok(), 1)
    assert all(len(a) == 1 for a in at)
    read = run.evaluate(held, ends=Ends(1, at))
    assert read["eval_loss"] == plain["eval_loss"]
    assert read["answer_rows"] == 3 and 0 <= read["ends"] <= 3
