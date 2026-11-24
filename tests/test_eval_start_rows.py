"""A resumed run keeps its start only if it is scored on the rows the start was scored on: the
start is kept with what names those rows (their count, and their tokens' fingerprint as the run
read them). Other rows (another or an edited held-out file) drop it, and the run says why; a start
that names no rows (a checkpoint from before starts did) cannot be checked, and is dropped with
nothing to say. The resume itself goes on: the held-out rows train nothing. Both loops."""

from __future__ import annotations

import json

import pytest

from tests.test_classic_evals import _held, _rows, _trainer


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _other(torch):
    """Other held-out rows: the same count, other tokens (an edited file)."""
    from tests.test_train_loop_cpu import _loader

    rows = _loader(torch, 256, n=5, seed=31).dataset
    return torch.utils.data.DataLoader(rows, batch_size=2, shuffle=False)


def _stop_at_3(torch, folder):
    from poros.core.errors import PorosCancelledError

    tr = _trainer(torch, folder, eval_steps=2)
    with pytest.raises(PorosCancelledError):
        tr.train(_rows(torch), steps=7, eval_dataset=_held(torch),
                 should_stop=lambda: tr.global_step == 3)


def _resume(torch, folder, held):
    tr = _trainer(torch, folder, eval_steps=2)
    return tr.train(_rows(torch), steps=7, eval_dataset=held,
                    resume_from=str(folder / "checkpoint-3"))


def test_the_engines_loop_keeps_its_start_only_for_the_same_rows(ml, tmp_path):
    from poros.train.evals import ROWS_CHANGED, START_FILE

    torch = ml
    whole = _trainer(torch, tmp_path / "whole", eval_steps=2).train(
        _rows(torch), steps=7, eval_dataset=_held(torch))

    same = tmp_path / "same"
    _stop_at_3(torch, same)
    said = json.loads((same / "checkpoint-3" / START_FILE).read_text())
    assert said["rows"] == 5 and isinstance(said["held"], str)
    out = _resume(torch, same, _held(torch))
    assert out["eval_loss_start"] == whole["eval_loss_start"] and "eval_start_note" not in out

    other = tmp_path / "other"
    _stop_at_3(torch, other)
    out = _resume(torch, other, _other(torch))
    assert "eval_loss_start" not in out and out["eval_start_note"] == ROWS_CHANGED
    assert "eval_loss" in out and len(out["loss_history"]) == 3  # the resume went on
    assert not (other / "checkpoint-6" / START_FILE).exists()  # nor is it carried on

    old = tmp_path / "old"  # a checkpoint as the candidate wrote it: a start naming no rows
    _stop_at_3(torch, old)
    path = old / "checkpoint-3" / START_FILE
    said = json.loads(path.read_text())
    del said["held"]
    path.write_text(json.dumps(said))
    out = _resume(torch, old, _held(torch))
    assert "eval_loss_start" not in out and "eval_start_note" not in out


def _recipe(torch, folder):
    from poros.train.run import Run
    from tests.test_checkpoint_evals import Tok

    return Run.from_trainer(_trainer(torch, folder), Tok())


def _shown(run, metrics):
    from poros.cli.train_recipe import _changed, _evals, _start, with_eval

    return with_eval(metrics, _evals(run._engine), _start(run._engine), _changed(run._engine))


def test_a_recipes_run_keeps_its_start_only_for_the_same_rows(ml, tmp_path):
    from poros import data as pdata
    from poros.core.errors import PorosCancelledError
    from poros.recipes import sft
    from poros.train.evals import ROWS_CHANGED, START_FILE
    from tests.test_checkpoint_evals import Tok
    from tests.test_checkpoint_evals import _rows as _chats

    torch = ml
    data, held = _chats()
    other = pdata.chats([{"messages": [{"role": "user", "content": f"ask {i}"},
                                       {"role": "assistant", "content": f"{i} and {i}"}]}
                         for i in range(3)], Tok(), batch_size=2)  # an edited file
    whole = _recipe(torch, tmp_path / "whole")
    want = _shown(whole, sft.main(whole, data, steps=7, eval_data=held))

    for name, rows, start in (("same", held, True), ("other", other, False),
                              ("old", held, None)):
        folder = tmp_path / name
        part = _recipe(torch, folder)
        with pytest.raises(PorosCancelledError):
            sft.main(part, data, steps=7, eval_data=held, stop=lambda: part.step == 4)
        if start is None:  # the candidate's checkpoint: a start that names no rows
            path = folder / "checkpoint-3" / START_FILE
            said = json.loads(path.read_text())
            del said["held"]
            path.write_text(json.dumps(said))
        part = _recipe(torch, folder)
        part.load(folder / "checkpoint-3")
        out = _shown(part, sft.main(part, data, steps=7, eval_data=rows))
        if start:
            assert out["eval_loss_start"] == want["eval_loss_start"]
        else:
            assert "eval_loss_start" not in out, name
        assert out.get("eval_start_note") == (ROWS_CHANGED if start is False else None)
