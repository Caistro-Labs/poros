"""What the screens say of a run on the engine's own loop (text rows), which holds no rows out:
the train card, its step count, the /eval card and the simulator say none are, and how to score
some. A recipe's run of chats still holds its 5% out, as before."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import poros.plan.configs
from poros.cli import sheets
from poros.cli.context import Context
from poros.cli.schemas import Figure, PlanGpu
from poros.cli.ui.sheet import Sheet
from poros.cli.words.grammar import parse


def _text(path: Path, n: int) -> Path:
    path.write_text("".join(json.dumps({"text": f"row {i} of plain text"}) + "\n"
                            for i in range(n)))
    return path


def _chats(path: Path, n: int) -> Path:
    path.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"hi {i}"}, {"role": "assistant", "content": "yo"}]}) + "\n"
        for i in range(n)))
    return path


@pytest.fixture
def card(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="offline in tests"))
    _text(tmp_path / "notes.jsonl", 1240)
    _chats(tmp_path / "chats.jsonl", 1240)
    _text(tmp_path / "held.jsonl", 30)
    gpu = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=None, source="spec")
    ctx = Context(cwd=tmp_path, fixed_gpu=gpu, fixed_cached=frozenset({"Qwen/Qwen2.5-7B-Instruct"}),
                  fixed_interactive=True)

    def make(data, *more):
        return Sheet(sheets.train_form(parse(["train", "Qwen2.5-7B-Instruct", "--data", data,
                                              *more]), ctx))

    make.ctx = ctx
    return make


def _row(sheet, key):
    return next(f for f in sheet.view.fields if f.key == key)


def test_the_card_of_text_rows_holds_none_out_and_says_how_to_score(card):
    sheet = card("notes.jsonl", "--steps", "40")
    held = _row(sheet, "eval")
    assert held.shown == "none" and "rows held out" not in held.source
    assert held.source == ("this loop trains every row: give a file of held-out rows, or "
                           "--recipe sft")
    picks = [o.value for o in held.options()]  # files: no share of its rows to pick
    assert "held.jsonl" in picks and all(k.endswith(".jsonl") for k in picks)
    assert _row(sheet, "save_steps=").source == "nothing held out to score"
    # a recipe's run of chats: as before
    chats = card("chats.jsonl", "--steps", "40")
    assert _row(chats, "eval").shown == "62 chats"
    assert "scored: loss" in _row(chats, "save_steps=").source


def test_an_epoch_of_text_rows_is_every_row(card):
    """The engine's own loop trains every row: an epoch counts them all, none held out."""
    from poros.cli.sheets.common import _resolved

    a, error, _ = _resolved("train", {"subject": "Qwen2.5-7B-Instruct",
                                      "--data": "notes.jsonl"}, card.ctx)
    assert a is not None, error
    assert a.held == 0 and a.per_epoch == -(-1240 // a.batch)
    assert a.provenance["steps"].detail == "one epoch of 1,240 rows"
    a, error, _ = _resolved("train", {"subject": "Qwen2.5-7B-Instruct", "--data": "chats.jsonl"},
                            card.ctx)
    assert a.held == 62 and "62 held out" in a.provenance["steps"].detail  # a recipe's, as before
    a, _, _ = _resolved("train", {"subject": "Qwen2.5-7B-Instruct", "--data": "notes.jsonl",
                                  "--eval": "held.jsonl"}, card.ctx)
    assert a.held == 30  # the file it scores


def test_the_eval_card_never_calls_a_text_runs_rows_held_out(tmp_path, monkeypatch):
    from poros.cli.sheets import export as card

    data = _text(tmp_path / "notes.jsonl", 60)
    folder = tmp_path / "run"
    folder.mkdir()
    said = {"trained_with": "classic", "data_shape": None, "dataset": str(data),
            "eval_split": 0.05, "eval_rows": 0}
    (folder / "config.resolved.json").write_text(json.dumps(said))
    monkeypatch.setattr(card, "run_held", lambda r: {"eval_rows": None})
    assert card._held_of(SimpleNamespace(output_dir=str(folder))) == ("", None)
    # a recipe's run of the same rows held 3 out, and says so
    (folder / "config.resolved.json").write_text(json.dumps(
        {**said, "trained_with": "sft", "data_shape": "text"}))
    assert card._held_of(SimpleNamespace(output_dir=str(folder))) == ("", 3)


def _simulated(tmp_path, monkeypatch, **cfg):
    from poros.cli.ui import simulate
    from poros.core.config import PorosConfig

    monkeypatch.setenv("POROS_DEV_SIMULATE_STEP", "0")
    out = tmp_path / "sim"
    trainer = simulate.SimTrainer(PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B",
                                              output_dir=str(out), save_steps=4, **cfg))
    metrics = trainer.train(cfg.get("dataset"), steps=10, eval_dataset=cfg.get("eval_dataset"))
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    return metrics, [e for e in events if e["event_type"] == "eval"]


def test_the_simulator_scores_a_text_run_only_where_the_loop_does(tmp_path, monkeypatch):
    data = _text(tmp_path / "notes.jsonl", 60)
    metrics, evals = _simulated(tmp_path, monkeypatch, dataset=str(data))
    assert evals == [] and "eval_loss" not in metrics  # nothing held out: nothing scored
    held = _text(tmp_path / "held.jsonl", 7)
    metrics, evals = _simulated(tmp_path / "b", monkeypatch, dataset=str(data),
                                eval_dataset=str(held))
    assert evals and {e["data"]["rows"] for e in evals} == {7}
    assert all(set(e["data"]) == {"eval_loss", "rows", "simulated"} for e in evals)
    assert "eval_loss_start" in metrics and "eval_loss" in metrics
