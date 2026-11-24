"""ExpertAdamW's bf16 update as one Triton kernel on CUDA: each element read once and written once,
with no temporaries.

It computes the rule written out in poros/train/expert_adamw.py, operation by operation: plain
fp32 operations with no fused multiply-add (``enable_fp_fusion=False``: Triton emits mul.rn and
add.rn, which ptxas does not contract), the correctly rounded square root and division
(``sqrt_rn``, ``div_rn``), F (a value at most 2**-126 in magnitude read and stored as +0) after
every load and every operation, casts to bf16 to nearest even, and the three keyed draws and the
stochastic rounding in integers. So it stores the reference's bits, which the optimizer checks
on the card before it uses the kernel (``_probe_kernel``), and the GPU tests and the device
self-test assert over inputs that exercise every operation.
"""

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl

TILE = 1024


@triton.jit
def _bits(coef, j, x, draw: tl.constexpr, ok):
    """One draw's 16 keyed bits: (a * x + b) mod 2**36, its top 16 bits (``_keyed_bits``)."""
    a = tl.load(coef + 6 * j + 2 * draw, mask=ok, other=0)
    b = tl.load(coef + 6 * j + 2 * draw + 1, mask=ok, other=0)
    return (((a * x + b) & 0xFFFFFFFFF) >> 20).to(tl.int32)


@triton.jit
def _round(x32, bits):
    """``_stochastic_round_bf16``: in int32 on the magnitude, as int16 bits of the bf16."""
    ix = x32.to(tl.int32, bitcast=True)
    out = (tl.minimum(ix & 0x7FFFFFFF, 0x7F800000) + bits) >> 16
    out = out - tl.where(ix < 0, 0x8000, 0)
    out = tl.where(x32 != x32, 0x7FC0, out)
    return out.to(tl.int16)


@triton.jit
def _F(x):
    """The rule's F: +0 where |x| <= 2**-126 (a NaN stays a NaN)."""
    return tl.where(tl.abs(x) <= 1.1754943508222875e-38, 0.0, x)


@triton.jit
def _expert_adamw_bf16(hi, lo, g, m, v, coef, n, w1, b2, w2, decay, bc2s, eps, neg_step,
                       TILE: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE).to(tl.int64)
    ok = offs < n
    j = offs >> 21  # the bit function's block, and the offset in it
    x = offs & 0x1FFFFF
    g32 = _F(tl.load(g + offs, mask=ok, other=0.0).to(tl.float32))
    m32 = _F(tl.load(m + offs, mask=ok, other=0.0).to(tl.float32))
    v32 = _F(tl.load(v + offs, mask=ok, other=0.0).to(tl.float32))
    h32 = _F(tl.load(hi + offs, mask=ok, other=0.0).to(tl.float32))
    l32 = _F(tl.load(lo + offs, mask=ok, other=0.0).to(tl.float32))
    m1 = _F(m32 + _F(w1 * _F(g32 - m32)))
    v1 = _F(_F(v32 * b2) + _F(_F(w2 * g32) * g32))
    w = _F(_F(h32 + l32) * decay)
    d = _F(_F(tl.math.div_rn(_F(tl.math.sqrt_rn(v1)), bc2s)) + eps)
    w = _F(w + _F(tl.math.div_rn(_F(neg_step * m1), d)))
    h = w.to(tl.bfloat16, fp_downcast_rounding="rtne")  # to nearest: what the forward reads
    hb = tl.where(w != w, 0x7FC0, h.to(tl.int16, bitcast=True)).to(tl.int16)
    tl.store(hi + offs, hb.to(tl.bfloat16, bitcast=True), mask=ok)
    hf = hb.to(tl.bfloat16, bitcast=True).to(tl.float32)
    r = _round(_F(w - hf), _bits(coef, j, x, 2, ok))
    # no remainder where hi is not finite. Stored as int16, said here: Triton 3.8 types a where of
    # an int16 tensor and the literal 0 as int16 already (checked in its interpreter), and the
    # cast and the assert hold it there whatever a later version promotes the literal to
    r = tl.where((hf - hf) == 0.0, r, 0).to(tl.int16)
    tl.static_assert(r.dtype == tl.int16)
    tl.static_assert(hb.dtype == tl.int16)
    tl.store(lo + offs, r.to(tl.bfloat16, bitcast=True), mask=ok)
    tl.store(m + offs, _round(m1, _bits(coef, j, x, 1, ok)).to(tl.bfloat16, bitcast=True),
             mask=ok)
    tl.store(v + offs, _round(v1, _bits(coef, j, x, 0, ok)).to(tl.bfloat16, bitcast=True),
             mask=ok)


def step(hi: torch.Tensor, lo: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
         coef: torch.Tensor, *, w1: float, b2: float, w2: float, decay: float, bc2s: float,
         eps: float, neg_step: float, tile: int = TILE) -> None:
    """One update of contiguous bf16 tensors of one parameter, in place. ``coef``: int64
    ``[a_v, b_v, a_m, b_m, a_r, b_r]`` per 2**21-element block (``_coefficients``)."""
    n = hi.numel()
    # launched on the tensors' own card, whichever is current: Triton launches on the current
    # device and its stream, and a model on cuda:1 under a current cuda:0 would be read from
    # the wrong context (CPU tensors: Triton's interpreter, in the tests)
    on = torch.cuda.device(hi.device) if hi.device.type == "cuda" else contextlib.nullcontext()
    with on:
        _expert_adamw_bf16[(triton.cdiv(n, tile),)](
            hi, lo, g, m, v, coef, n, w1, b2, w2, decay, bc2s, eps, neg_step, TILE=tile,
            enable_fp_fusion=False)
