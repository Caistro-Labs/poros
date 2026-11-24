"""A resumed run's held-out start is the run's own first score, the model it started from, carried
in its checkpoints: resumed once or twice, with its events file gone or not, it says the start the
uninterrupted run says, on the engine's own loop and on a recipe's run. A checkpoint that keeps no
start (written before checkpoints kept it) gives the last score alone, never a start invented
from a score on the way."""

from __future__ import annotations

import pytest

from tests.test_classic_evals import _held, _rows, _trainer


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _stopped_at(torch, outdir, at, resume=None):
    """The engine's loop stopped after update ``at`` (from ``resume`` if given)."""
    from poros.core.errors import PorosCancelledError

    tr = _trainer(torch, outdir, eval_steps=2)
    with pytest.raises(PorosCancelledError):
        tr.train(_rows(torch), steps=7, eval_dataset=_held(torch), resume_from=resume,
                 should_stop=lambda: tr.global_step == at)


def _finish(torch, outdir, resume):
    tr = _trainer(torch, outdir, eval_steps=2)
    return tr.train(_rows(torch), steps=7, eval_dataset=_held(torch), resume_from=str(resume))


def test_the_engines_loop_resumed_once_and_twice_keeps_its_start(ml, tmp_path):
    from poros.cli.verbs import runs_report

    torch = ml
    whole = _trainer(torch, tmp_path / "whole", eval_steps=2).train(
        _rows(torch), steps=7, eval_dataset=_held(torch))

    once = tmp_path / "once"
    _stopped_at(torch, once, 3)
    out = _finish(torch, once, once / "checkpoint-3")
    assert out["eval_loss_start"] == whole["eval_loss_start"]
    assert out["eval_loss"] == whole["eval_loss"]
    held = runs_report.held_out(str(once))
    assert (held["eval_loss_start"], held["eval_loss"]) == (whole["eval_loss_start"],
                                                           whole["eval_loss"])

    twice = tmp_path / "twice"
    _stopped_at(torch, twice, 1)
    _stopped_at(torch, twice, 4, resume=str(twice / "checkpoint-1"))
    (twice / "events.jsonl").unlink()  # the checkpoint carries it, not the events
    out = _finish(torch, twice, twice / "checkpoint-4")
    assert out["eval_loss_start"] == whole["eval_loss_start"]
    assert out["eval_loss"] == whole["eval_loss"]


def test_a_checkpoint_that_keeps_no_start_gives_the_last_score_alone(ml, tmp_path):
    from poros.train.evals import START_FILE

    torch = ml
    _stopped_at(torch, tmp_path, 3)
    (tmp_path / "checkpoint-3" / START_FILE).unlink()  # as a checkpoint from before this
    out = _finish(torch, tmp_path, tmp_path / "checkpoint-3")
    assert "eval_loss_start" not in out and "eval_loss" in out


def _recipe(torch, outdir, **cfg):
    from poros.train.run import Run
    from tests.test_checkpoint_evals import Tok

    tr = _trainer(torch, outdir, **cfg)
    return Run.from_trainer(tr, Tok())


def _shown(run, metrics):
    from poros.cli.train_recipe import _evals, _start, with_eval

    return with_eval(metrics, _evals(run._engine), _start(run._engine))


def test_a_recipes_run_resumed_once_and_twice_keeps_its_start(ml, tmp_path):
    from poros.cli.verbs import runs_report
    from poros.core.errors import PorosCancelledError
    from poros.recipes import sft
    from tests.test_checkpoint_evals import _rows as _chats

    torch = ml
    data, held = _chats()
    whole = _recipe(torch, tmp_path / "whole")
    want = _shown(whole, sft.main(whole, data, steps=7, eval_data=held))

    folder = tmp_path / "part"
    part = _recipe(torch, folder)
    with pytest.raises(PorosCancelledError):
        sft.main(part, data, steps=7, eval_data=held, stop=lambda: part.step == 2)
    part = _recipe(torch, folder)
    part.load(folder / "checkpoint-1")
    with pytest.raises(PorosCancelledError):
        sft.main(part, data, steps=7, eval_data=held, stop=lambda: part.step == 5)
    (folder / "events.jsonl").rename(folder / "events.old")  # the checkpoint carries it
    part = _recipe(torch, folder)
    part.load(folder / "checkpoint-4")
    said = _shown(part, sft.main(part, data, steps=7, eval_data=held))
    assert said["eval_loss_start"] == want["eval_loss_start"]
    assert said["eval_loss"] == want["eval_loss"]
    found = runs_report.held_out(str(tmp_path / "whole"))
    assert found["eval_loss_start"] == want["eval_loss_start"]


def test_the_end_card_says_no_from_without_a_start(ml):
    from poros.cli.ui import screens
    from poros.cli.ui.style import resolve
    from poros.cli.ui.text import render_line

    def card(start):
        def result(losses):
            return screens.TrainResult(
                steps=7, wall_s=3.0, final_loss=1.0, first_loss=2.0, losses=losses,
                peak_vram_gb=0.0, card_gb=0.0, peak_ram_gb=None, ram_total_gb=None,
                output_dir="out", receipt="r", tokens_per_sec=1.0, eval_start=start,
                eval_end=0.9, offset=4, evals=[(5, 1.3), (7, 0.9)])

        drawn = result([2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.0])  # the card, with its curve
        said = []  # its rows (one loss: no curve to draw), then the card
        for lines in (screens.train_done(result([1.0]), 90),
                      screens.train_done(drawn, 90)):
            said.append(" ".join(" ".join(render_line(x, resolve(depth="plain"))
                                          for x in lines).split()))
        return said

    rows, drawn = card(1.5)
    assert "eval loss 0.9000 from 1.5000 on held-out rows" in rows and "Learned well" in drawn
    rows, drawn = card(None)  # the first score here (1.3, step 5) was taken on the way
    assert "eval loss 0.9000 on held-out rows" in rows and "from 1.3" not in rows
    assert not any(w in drawn for w in ("Learned well", "Barely moved", "loss rose"))
