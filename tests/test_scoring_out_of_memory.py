"""Running out of GPU memory while scoring held-out rows is said as a training step's is (E_OOM),
with what to change, on the engine's own loop, on a recipe's run and in `poros eval`'s pass: the
batch and the length, never a shorter held-out file. The error is forced with a stub."""

from __future__ import annotations

import pytest

from tests.test_classic_evals import _held, _rows, _trainer


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _out_of_memory(torch, tr, monkeypatch):
    released = []

    def full(batch, **kw):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.48 GiB")

    monkeypatch.setattr(tr, "_forward_eval_logits", full)
    real = tr._release_retained_forward_blocks
    monkeypatch.setattr(tr, "_release_retained_forward_blocks",
                        lambda: released.append(1) or real())
    return released


def _said(err) -> str:
    return f"{err} {err.info.remediation}"


def test_the_engines_loop_says_it_ran_out_while_scoring(ml, tmp_path, monkeypatch):
    from poros.core.errors import PorosOOMError

    torch = ml
    tr = _trainer(torch, tmp_path)
    released = _out_of_memory(torch, tr, monkeypatch)
    with pytest.raises(PorosOOMError) as caught:
        tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    said = _said(caught.value)
    assert caught.value.code == "E_OOM"
    assert "while scoring the held-out rows" in said
    assert "here 2 rows of up to 16 tokens" in said  # the held-out loader's batch
    assert "a lower per_device_batch_size or max_seq_length" in said
    assert "A shorter held-out file does not help" in said
    assert released  # what the pass held is let go, as training's handler does
    assert isinstance(caught.value.__cause__, torch.cuda.OutOfMemoryError)


def test_a_recipes_run_says_it_the_same_way(ml, tmp_path, monkeypatch):
    from poros.core.errors import PorosOOMError
    from poros.recipes import sft
    from poros.train.run import Run
    from tests.test_checkpoint_evals import Tok
    from tests.test_checkpoint_evals import _rows as _chats

    torch = ml
    tr = _trainer(torch, tmp_path)
    tr.config.per_device_batch_size = 1
    _out_of_memory(torch, tr, monkeypatch)
    data, held = _chats()
    with pytest.raises(PorosOOMError) as caught:
        sft.main(Run.from_trainer(tr, Tok()), data, steps=7, eval_data=held)
    said = _said(caught.value)
    assert "while scoring the held-out rows" in said
    assert "here 1 row of up to" in said and "A shorter held-out file" in said
    assert caught.value.info.fix.get("max_seq_length")  # one row: the length is what to lower


def test_an_error_that_is_not_out_of_memory_is_left_as_it_is(ml, tmp_path, monkeypatch):
    torch = ml
    tr = _trainer(torch, tmp_path)

    def broken(batch, **kw):
        raise ValueError("a bug, not memory")

    monkeypatch.setattr(tr, "_forward_eval_logits", broken)
    with pytest.raises(ValueError, match="a bug, not memory"):
        tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))


def test_each_caller_is_told_what_it_can_change(ml, tmp_path, monkeypatch):
    """In a run a resume keeps both settings; poros eval <run> takes the run's and no other;
    poros eval <config> (and evaluate()) reads the config's."""
    from poros.cli.verbs import evaluate_pass as ep
    from poros.core.errors import PorosOOMError

    torch = ml
    tr = _trainer(torch, tmp_path)
    _out_of_memory(torch, tr, monkeypatch)
    with pytest.raises(PorosOOMError) as caught:
        tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    said = _said(caught.value)
    assert "A resume keeps per_device_batch_size and max_seq_length" in said
    assert "start the run again with a lower" in said

    with pytest.raises(PorosOOMError) as caught:  # poros eval <config>, Python's evaluate()
        tr.evaluate(_held(torch))
    said = _said(caught.value)
    assert "These are the config's per_device_batch_size and max_seq_length" in said

    rows = ep.LoopRows("held.jsonl", "eval_file", loader=_held(torch), rows=5, total=5)
    with pytest.raises(PorosOOMError) as caught:  # poros eval <run>, a run on the engine's loop
        ep.score_loop(tr, rows)
    said = _said(caught.value)
    assert "takes no other" in said and "--rows and a shorter --data file do not help" in said
    assert "Lower" not in said and not caught.value.info.fix  # nothing it could set


def test_poros_evals_recipe_pass_says_the_runs_record_sets_both(ml, tmp_path, monkeypatch):
    from poros import data as pdata
    from poros.cli.verbs import evaluate_pass as ep
    from poros.core.errors import PorosOOMError
    from tests.test_checkpoint_evals import Tok

    torch = ml
    tr = _trainer(torch, tmp_path)
    _out_of_memory(torch, tr, monkeypatch)
    rows = [{"text": f"row {i} of text"} for i in range(4)]
    data = pdata.text(rows, Tok(), batch_size=2)
    with pytest.raises(PorosOOMError) as caught:
        ep.score(tr, ep.Rows(data, "text", "held_out", 4, "x.jsonl"), None)
    said = _said(caught.value)
    assert "takes no other" in said and not caught.value.info.fix
