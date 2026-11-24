"""The simulator scores and saves where the real loops do: the steps it reports a held-out score
and a checkpoint at equal those of the engine's own loop and of a recipe's run on the tiny CPU
engine, for the same settings, with and without held-out rows, through a resume. It calls the
same rules (poros.train.evals.scores_at, saves_at), so it cannot drift from them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_classic_evals import _held, _rows, _trainer

STEPS = 7
SETTINGS = [dict(save_steps=2, eval_steps=3), dict(save_steps=3, eval_steps=0),
            dict(save_steps=2, eval_steps=3, eval_checkpoints=False),
            dict(save_steps=0, eval_steps=2)]


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


class Seen:
    """What a run reports: the steps it scored at and saved at, and the scores."""

    def __init__(self):
        self.evals, self.saves = [], []

    def on_eval(self, step, said):
        self.evals.append(step)

    def on_checkpoint(self, step, path):
        self.saves.append(step)

    def __getattr__(self, name):
        return lambda *a, **k: None


def _stop(at):
    asked = []
    return (lambda: asked.append(1) or len(asked) == at + 1) if at is not None else None


def _sim(folder: Path, data: str, recipe: str, settings: dict, eval_dataset=None, stop=None,
         resume=None, eval_split=0.05):
    from poros.cli.ui import simulate
    from poros.core.config import PorosConfig

    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", output_dir=str(folder),
                      dataset=data, eval_dataset=eval_dataset, eval_split=eval_split,
                      save_total_limit=10, **settings)
    tr = simulate.SimTrainer(cfg)
    tr.recipe, tr.reporter, seen = recipe, Seen(), None
    seen = tr.reporter
    try:
        out = tr.train(data, steps=STEPS, resume_from=resume, should_stop=_stop(stop),
                       eval_dataset=eval_dataset)
    except Exception as exc:  # the stop
        assert type(exc).__name__ == "PorosCancelledError", exc
        out = None
    return seen, out


def _real_loop(torch, folder: Path, settings: dict, held: bool, stop=None, resume=None):
    tr = _trainer(torch, folder, **settings)
    tr.reporter = Seen()
    try:
        out = tr.train(_rows(torch), steps=STEPS, eval_dataset=_held(torch) if held else None,
                       resume_from=resume, should_stop=_stop(stop))
    except Exception as exc:
        assert type(exc).__name__ == "PorosCancelledError", exc
        out = None
    return tr.reporter, out


def _text(path: Path, n: int) -> str:
    path.write_text("".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(n)))
    return str(path)


@pytest.fixture
def mode(ml, monkeypatch):
    monkeypatch.setenv("POROS_DEV_SIMULATE_STEP", "0")
    return ml


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("held", [True, False])
def test_the_engines_loop_and_the_simulator_score_and_save_at_the_same_steps(mode, tmp_path,
                                                                             settings, held):
    torch = mode
    data = _text(tmp_path / "notes.jsonl", 60)
    file = _text(tmp_path / "held.jsonl", 5) if held else None
    real, real_out = _real_loop(torch, tmp_path / "real", settings, held)
    sim, sim_out = _sim(tmp_path / "sim", data, "classic", settings, eval_dataset=file)
    assert sim.evals == real.evals and sim.saves == real.saves
    assert ("eval_loss_start" in sim_out) == ("eval_loss_start" in real_out) == held
    if held:
        assert real.evals[0] == 0 and real.evals[-1] == STEPS - 1

    # stopped after update 3, then resumed: the same steps again, and the start carried
    real, _ = _real_loop(torch, tmp_path / "r2", settings, held, stop=3)
    sim, _ = _sim(tmp_path / "s2", data, "classic", settings, eval_dataset=file, stop=3)
    assert sim.evals == real.evals and sim.saves == real.saves
    real, real_out = _real_loop(torch, tmp_path / "r2", settings, held,
                                resume=str(tmp_path / "r2" / "checkpoint-3"))
    sim, sim_out = _sim(tmp_path / "s2", data, "classic", settings, eval_dataset=file,
                        resume=str(tmp_path / "s2" / "checkpoint-3"))
    assert sim.evals == real.evals and sim.saves == real.saves
    assert ("eval_loss_start" in sim_out) == ("eval_loss_start" in real_out) == held
    if held:
        assert sim_out["eval_loss_start"] == 2.91  # its own start, carried, not a later score


def _real_recipe(torch, folder: Path, settings: dict, held, stop=None, resume=None):
    from poros.cli.train_recipe import _evals, _start, with_eval
    from poros.core.errors import PorosCancelledError
    from poros.recipes import sft
    from poros.train.run import Run
    from tests.test_checkpoint_evals import Tok
    from tests.test_checkpoint_evals import _rows as _chats

    tr = _trainer(torch, folder, **settings)
    tr.reporter = Seen()
    run = Run.from_trainer(tr, Tok())
    if resume:
        run.load(resume)
    data, rows = _chats()
    asked = _stop(stop)
    try:
        out = sft.main(run, data, steps=STEPS, eval_data=rows if held else None, stop=asked)
    except PorosCancelledError:
        return tr.reporter, None
    return tr.reporter, with_eval(out, _evals(tr), _start(tr))


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("held", [True, False])
def test_a_recipes_run_and_the_simulator_score_and_save_at_the_same_steps(mode, tmp_path,
                                                                          settings, held):
    torch = mode
    chats = tmp_path / "chats.jsonl"
    chats.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"say {i}"}, {"role": "assistant", "content": "ok"}]}) + "\n"
        for i in range(60)))  # 60 rows: 3 held out, as the real run's holdout
    split = 0.05 if held else 0.0
    real, real_out = _real_recipe(torch, tmp_path / "real", settings, held)
    sim, sim_out = _sim(tmp_path / "sim", str(chats), "sft", settings, eval_split=split)
    assert sim.evals == real.evals and sim.saves == real.saves
    assert ("eval_loss_start" in sim_out) == ("eval_loss_start" in real_out) == held

    real, _ = _real_recipe(torch, tmp_path / "r2", settings, held, stop=3)
    sim, _ = _sim(tmp_path / "s2", str(chats), "sft", settings, eval_split=split, stop=3)
    assert sim.evals == real.evals and sim.saves == real.saves
    real, real_out = _real_recipe(torch, tmp_path / "r2", settings, held,
                                  resume=str(tmp_path / "r2" / "checkpoint-3"))
    sim, sim_out = _sim(tmp_path / "s2", str(chats), "sft", settings, eval_split=split,
                        resume=str(tmp_path / "s2" / "checkpoint-3"))
    assert sim.evals == real.evals and sim.saves == real.saves
    assert ("eval_loss_start" in sim_out) == ("eval_loss_start" in real_out) == held
