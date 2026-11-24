"""The chunked head's cross-entropy in one Triton kernel (CUDA): per row of bf16 logits, the
log-sum-exp (an online max and sum over the vocabulary, fp32), the target's loss term, and the
logits' grad ``(softmax - onehot)`` written in the logits' dtype -- two reads of the row and one
write, where the torch ops (upcast, logsumexp, gather, sub, exp, index, mul, cast) make about
eight passes over a [rows, vocab] fp32 copy. One program per row, one fixed block over the
vocabulary: a row's values do not depend on the others. Imported only on CUDA (head.head_loss)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CE_BLOCK, CE_WARPS = 4096, 8


@triton.jit
def _ce_rows(logits, target, valid, grad, loss, V, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits + row * V
    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), dtype=tl.float32)
    for v0 in range(0, V, BLOCK):
        cols = v0 + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(s)
    t = tl.load(target + row)
    ok = tl.load(valid + row) != 0  # a masked row is 0 whatever its logits (inf or NaN too)
    picked = tl.load(base + t).to(tl.float32)
    tl.store(loss + row, tl.where(ok, lse - picked, 0.0))
    gbase = grad + row * V
    for v0 in range(0, V, BLOCK):
        cols = v0 + tl.arange(0, BLOCK)
        mc = cols < V
        x = tl.load(base + cols, mask=mc, other=0.0).to(tl.float32)
        p = tl.exp(x - lse) - tl.where(cols == t, 1.0, 0.0)
        tl.store(gbase + cols, tl.where(ok, p, 0.0).to(grad.dtype.element_ty), mask=mc)


def cross_entropy_rows(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                       grad: torch.Tensor) -> torch.Tensor:
    """Per row of ``logits`` [R, V]: the loss term ``lse - logits[target]`` (zero where not
    ``valid``), returned [R] fp32; ``grad`` [R, V] (the logits' dtype) gets ``(softmax - onehot)``
    (zero rows where not ``valid``)."""
    rows, vocab = logits.shape
    loss = torch.empty(rows, dtype=torch.float32, device=logits.device)
    _ce_rows[(rows,)](logits, target, valid, grad, loss, vocab, BLOCK=CE_BLOCK,
                      enable_fp_fusion=True, num_warps=CE_WARPS)
    return loss
