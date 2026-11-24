"""An eval pass on the fused-head path (CUDA, the engine's own loop) never holds the whole
``[tokens, vocab]`` logits: it scores a few rows of tokens at a time, as training's head does. Its
loss is the whole logits' mean over the same positions, the same log-softmax and the same
``nll_loss`` reduction; the head's matrix product on fewer rows may round the last bit of a logit
otherwise (the CPU's does for some row counts), so it equals the whole-logits loss to fp32
rounding, and every score of such a run takes it the same way. Nothing else changes: off that path
an eval pass is as it was, bit for bit, and so is everything a recipe's run scores (tiny CPU
engine; the path forced on, as CUDA's fused compute would take it)."""

from __future__ import annotations

import pytest

from tests.test_classic_evals import _held, _rows, _trainer


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _batch(torch):
    ids = torch.randint(0, 256, (3, 16), generator=torch.Generator().manual_seed(4))
    mask = torch.ones_like(ids)
    mask[1, 11:] = 0  # a shorter row: padding, not scored
    mask[2, 5:] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@pytest.mark.parametrize("rows", [1, 2, 3, 7, 8, 15, 64])
def test_the_chunked_loss_is_the_whole_logits_loss(ml, tmp_path, monkeypatch, rows):
    from poros.train import loop

    torch = ml
    tr = _trainer(torch, tmp_path)
    tr.model.eval()
    batch = _batch(torch)
    hidden, labels = tr._forward_eval_logits(batch, hidden_only=True)
    with torch.no_grad():
        whole = tr._compute_loss(tr._logits_from_hidden(hidden), labels)
        each_whole = loop._row_losses(tr._logits_from_hidden(hidden), labels)
        monkeypatch.setattr(loop, "_EVAL_ROWS", rows)
        chunked, each = tr._eval_loss_in_chunks(hidden, labels, rows=True)
    assert chunked.item() == pytest.approx(whole.item(), rel=1e-6, abs=0)
    assert each == pytest.approx(each_whole, rel=1e-6, abs=0)


def test_a_run_scored_in_chunks_trains_the_same_bytes_and_holds_a_chunk_at_a_time(
        ml, tmp_path, monkeypatch):
    """The whole run: its scores those of the whole logits to fp32 rounding, its training the
    same bytes, and no chunk of logits more than half the batch's tokens."""
    from tests.test_classic_evals import _trained

    torch = ml
    whole = _trainer(torch, tmp_path / "whole", eval_steps=2)
    a = whole.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    shapes = []
    real = type(whole)._logits_from_hidden

    def seen(self, hidden, normed=False):
        if normed:  # the eval pass's chunks
            shapes.append(tuple(hidden.shape))
        return real(self, hidden, normed=normed)

    monkeypatch.setattr(type(whole), "_logits_from_hidden", seen)
    chunked = _trainer(torch, tmp_path / "chunked", eval_steps=2)
    # the eval pass on the fused path; training stays on its own (it runs with grad)
    monkeypatch.setattr(chunked.head, "fused_dense_applies",
                        lambda hidden, labels: not torch.is_grad_enabled())
    b = chunked.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    assert [e["eval_loss"] for e in b["eval_history"]] == pytest.approx(
        [e["eval_loss"] for e in a["eval_history"]], rel=1e-6, abs=0)
    assert _trained(whole, tmp_path / "whole")["adapter"] == _trained(
        chunked, tmp_path / "chunked")["adapter"]
    # 2 rows of 16 tokens a batch: chunks of at most half the batch's 32 tokens, never all
    assert shapes and all(len(s) == 2 and s[0] <= 16 for s in shapes)


def test_off_the_fused_path_an_eval_pass_is_the_whole_logits_loss_bit_for_bit(ml, tmp_path):
    """The CPU, a Mac, a soft-capped head, a MoE model: the pass scores the whole logits as it
    did, so its numbers are as they were."""
    torch = ml
    tr = _trainer(torch, tmp_path)
    batch = _batch(torch)
    loader = [batch]
    said, rows = tr._eval_pass(loader)
    tr.model.eval()
    with torch.no_grad():
        logits, labels = tr._forward_eval_logits(batch)
        want = float(tr._compute_loss(logits, labels).item())
    assert said["eval_loss"] == want and rows == 3
