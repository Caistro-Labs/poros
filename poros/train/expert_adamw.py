"""AdamW for bf16 adapters: every stored value an unbiased rounding of its fp32 value.

A MoE model's expert LoRA adapters are bf16 parameters: the fused expert kernels read them in
bf16 (``poros.moe.fused.block_reason``), and at Qwen3-30B-A3B's 629M expert-adapter parameters
fp32 copies would not fit the small machines Poros trains MoE on. torch's AdamW on bf16
parameters loses an update smaller than half a bf16 step of the weight (at lr 2e-5 most of them),
and its second moment, which moves by 0.1 percent of the gap a step at beta2 0.999, cannot move
at bf16's resolution, so it only ratchets up. ``scripts/expert_adam_precision.py`` measures both
(docs/validation/int/cpu/expert_adam_precision/).

``ExpertAdamW`` computes torch's AdamW step in fp32 and keeps one rule for what it stores: the
forward reads the nearest bf16 weight, and every value the optimizer stores is an unbiased keyed
rounding of its fp32 value. So, per element:

* the bf16 parameter (``hi``) is the fp32 weight w rounded to nearest: what every kernel and the
  forward read;
* ``state["remainder"]`` is w - hi (exact in fp32) rounded stochastically to bf16. Its magnitude
  is at most half of hi's bf16 step, a power of two and so itself a bf16 value, and it is a
  multiple of w's own fp32 step, so hi + remainder is exact in fp32, and its expectation is w
  wherever w - hi is not read as zero (F, below): for |w| under 2**-102 the smallest remainders
  are, and under 2**-117 every one is, so there the stored weight is w rounded to nearest.
  No update is too small to count: rounding to nearest would drop, every time, any change below
  about 2**-17 of the weight (weight decay at lr 2e-5 on an idle expert, a cosine schedule's
  tail); rounded stochastically it lands in expectation;
* ``exp_avg`` (m) and ``exp_avg_sq`` (v) are their fp32 values rounded stochastically to bf16:
  rounding to nearest freezes v (beta2 0.999 moves it by 0.1 percent of the gap a step, under
  bf16's resolution) and leaves m short of a steady gradient by up to half a bf16 step over
  1 - beta1 (3.9 percent at beta1 0.9, 39 percent at 0.99).

Each step uses the unrounded fp32 m and v of that step. Rounding stochastically needs, per stored
value, 16 random bits uniform over [0, 2**16) and fresh at each update: then the rounding of each
element is unbiased, and its errors from update to update are independent, so they average out
rather than drift. Three independent draws an element an update (v, m, the remainder: each its
own blake2b key), so one value's rounding is not correlated with another's. Within a block of
2**21 elements a draw is pairwise independent (``_keyed_bits``): the rounding errors of two
elements are uncorrelated, so a sum over elements (a loss, a norm) sees their variances add, as
independent errors would; nothing more is asked of them, since no expectation of the update
depends on a joint distribution of more than one element's rounding.

fp32 parameters among bf16 ones (an Inkling MoE run's attention adapters, native fp32 adapters
beside its bf16 expert adapters; a dense run whose trainable tensors are all fp32 keeps torch's
AdamW, ``build_optimizer``) get torch's AdamW math with fp32 state, updated in place, and no
remainder.

The random bits are a pure function of (run seed, update, the parameter's name, which of the
three values, the element's flat index): blake2b keys each (update, parameter, value, block) as
``KeyedRng`` keys a layer, and a multiply-add of the element index runs on the parameter's
device with no product reaching 2**63, so CPU, CUDA and MPS draw the same bits. No torch
generator is read or moved, so dropout masks and data order do not change. The step is
elementwise, so its result does not depend on the chunking, the order parameters are visited in,
or whether the run was stopped and resumed.

The update is one function of its inputs, written out operation by operation below ("the rule"):
plain IEEE fp32 operations, no fused multiply-add, subnormals read and stored as zero. Its
reference (``_reference_bf16``) computes it in torch's basic operations, with the square root and
division corrected to the correctly rounded result, so its bits do not depend on the CPU build or
the device. On CUDA (Triton) and a Mac (Metal) a fused kernel computes the same function, one
kernel a tensor (poros/train/expert_adamw_cuda.py, poros/metal/expert_adamw_shader.py). Which
runs is chosen once, when the optimizer is built (``_implementation``): the kernel only where it
stores the reference's bits over inputs that exercise every operation (``_probe_kernel``), else
the reference (on a CUDA card in fp64, or on the CPU for a Mac's GPU, which has no fp64).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)

from poros.core.checkpoints import (  # noqa: E402  the rules, torch-free: a resume reads them
    MAX_LR,
    OLD_STATE_FIX,
    RULE,
    RULE_FILE,
    TORCH_RULE,
    old_state_message,
)

# What the trainer uses. The bit and rounding functions below are private: they take what the
# optimizer gives them and check nothing an element (a check there would cost a pass a call).
__all__ = ["ExpertAdamW", "build_optimizer", "rule_of", "trainable_dtype", "write_rule"]

# Elements the reference (the CPU's; a CUDA card's without Triton; a Mac's whose kernel is not
# used) works on at once. Its temporaries, measured on the CPU for one step (VmHWM above the
# stored state, glibc returning large blocks at once): about 100 bytes an element of a chunk, as
# fp32, fp64 and integer passes: 47.6 MiB for a 2**21-element tensor at chunks of 2**19 (rule
# v2's eager step held 46 MiB at 2**21, v1's 109 MiB). The fused kernels on CUDA and a Mac hold
# none: 48 bytes a block of 2**21 elements for the coefficients. The chunking changes no bit
# (``_keyed_bits`` keys by index).
CHUNK = 1 << 19


# ---- keyed random bits --------------------------------------------------------------------

BLOCK = 1 << 21  # the bit function's own blocks of a tensor's elements (not the step's chunks)
_WIDTH = 36  # bits of the multiply-add: w >= u + l - 1 = 21 + 16 - 1
_KEEP = (1 << _WIDTH) - 1
DRAW_V, DRAW_M, DRAW_REMAINDER = 0, 1, 2  # the three values an element stores, each its own bits
MAX_STEP = 1 << 62  # a saved step count at most this: update_key packs it as a signed int64
MAX_ELEMENTS = 1 << 32  # a tensor's elements the keys index (refused at construction above)
SEEDS = (-(2**63), 2**63)  # the run seeds update_key packs (as KeyedRng.key does): [low, high)


def update_key(seed: int, update: int, name: str) -> int:
    """The 64-bit key of one parameter's bits at one update (blake2b, as ``KeyedRng.key``).
    ``seed`` within SEEDS, ``update`` from 1.

    Two (parameter, update) pairs share a key with probability about N**2 / 2**65 over a run's
    N = parameters * updates keys (the birthday bound of a 64-bit digest): 0.17% for 25,000
    expert adapter tensors over 10,000 updates (N = 2.5e8), 2.7% at N = 1e9. A shared key costs
    only an unlucky shared draw: those two tensors at those two updates round with the same
    random bits. Each rounding is still unbiased on its own (its bits are uniform whatever the
    other's); only those two rounding errors are correlated, once, in a run of billions of
    independent ones. No result depends on keys being distinct."""
    if not SEEDS[0] <= int(seed) < SEEDS[1] or int(update) < 0:
        raise ValueError(f"update_key: a seed of {seed} and update {update}: the seed must be "
                         "a signed 64-bit integer and the update at least 0")
    data = struct.pack("<2q", int(seed), int(update)) + name.encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _block_coefficients(key: int, block: int, draw: int) -> tuple[int, int]:
    """(a, b), each uniform over [0, 2**36), for one draw (DRAW_V, DRAW_M or DRAW_REMAINDER) of
    one block of a tensor's elements at one update: blake2b of the parameter's update key, the
    block's index and the draw."""
    h = hashlib.blake2b(struct.pack("<3Q", key & (2**64 - 1), int(block), int(draw)),
                        digest_size=16)
    digest = h.digest()
    return (int.from_bytes(digest[:8], "little") & _KEEP,
            int.from_bytes(digest[8:], "little") & _KEEP)


def _keyed_bits(key: int, start: int, n: int, draw: int, device: Any = "cpu") -> torch.Tensor:
    """16 random bits (int32 in [0, 65536)) for elements ``start .. start + n`` of the tensor
    keyed ``key``, for one draw: element i's bits depend on (key, draw, i) alone.

    Element i lies in block j = i // 2**21 at offset x = i % 2**21, and its bits are
    ``((a * x + b) mod 2**36) >> 20`` with (a, b) = ``_block_coefficients(key, j, draw)``:
    Dietzfelbinger's multiply-add-shift family, strongly universal for x below 2**21, 16 output
    bits and a and b uniform over [0, 2**36) (w = 36 >= u + l - 1). For any one element its bits
    are uniform over [0, 2**16), so its stochastic rounding is unbiased; any two elements of a
    block are independent; blocks, draws, parameters and updates have their own keys.
    a * x + b stays below 2**58: no int64 lane overflows on any device. Four integer operations
    an element. Private: the optimizer gives it 0 <= start, start + n <= MAX_ELEMENTS."""
    parts = []
    i = start
    while i < start + n:
        block, x0 = divmod(i, BLOCK)
        x1 = min(BLOCK, x0 + start + n - i)
        a, b = _block_coefficients(key, block, draw)
        x = torch.arange(x0, x1, dtype=torch.int64, device=device)
        parts.append(x.mul_(a).add_(b).bitwise_and_(_KEEP).bitwise_right_shift_(_WIDTH - 16)
                     .to(torch.int32))
        i += x1 - x0
    if not parts:  # no elements: no bits
        return torch.empty(0, dtype=torch.int32, device=device)
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def _stochastic_round_bf16(x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """fp32 ``x`` to bf16, up in magnitude with probability (its 16 dropped bits) / 2**16, so
    E[result] = x. Integer work only, so every device gives the same bits for every input: adds
    ``bits`` under the kept bits of the magnitude and truncates. An infinity keeps its own bits;
    every NaN becomes the one quiet NaN (0x7fc0), where a cast would give a NaN whose sign
    depends on the device. A finite value that would round past bf16's largest becomes inf, as
    rounding to nearest would.

    int32 throughout: the magnitude is held at most 0x7f800000 (an infinity; a NaN is restored
    after), so adding 16 bits stays below 2**31. Private: ``bits`` is ``_keyed_bits``'s."""
    x = x.contiguous()
    i = x.view(torch.int32)
    out = (i & 0x7FFFFFFF).clamp_max_(0x7F800000)
    out.add_(bits.reshape(x.shape).to(torch.int32)).bitwise_right_shift_(16)
    out.add_(i < 0, alpha=-0x8000)  # the sign bit, as int16's: magnitude - 2**15
    out.masked_fill_(torch.isnan(x), 0x7FC0)
    return out.to(torch.int16).view(torch.bfloat16)


def _coefficients(key: int, n: int, device: Any = "cpu") -> torch.Tensor:
    """int64 ``[a_v, b_v, a_m, b_m, a_r, b_r]`` for every block of an n-element tensor, in block
    order: the three draws' ``_block_coefficients``, for the fused kernels."""
    out = [c for j in range(-(-n // BLOCK))
           for draw in (DRAW_V, DRAW_M, DRAW_REMAINDER) for c in _block_coefficients(key, j, draw)]
    # a blocking copy: the CPU tensor is a temporary, and a non-blocking copy could read it after
    # it is freed
    return torch.tensor(out, dtype=torch.int64).to(device)


# ---- the rule: one function of its inputs, defined operation by operation ----------------------
#
# Per element, with every scalar first rounded to fp32 (c1 = 1 - beta1, b2, c2 = 1 - beta2,
# dec = 1 - lr * weight_decay, bc = sqrt(1 - beta2**t), eps, ns = -lr / (1 - beta1**t)), every
# operation below one IEEE fp32 operation correctly rounded to nearest even, and no fused
# multiply-add anywhere. F(x) is x, except +0 where |x| <= 2**-126 (a subnormal, a zero of either
# sign, or 2**-126 itself, FLT_MIN, the smallest normal fp32): it is applied to every input and
# to every result, so a device that flushes subnormals to zero (CUDA's ftz, a Mac's GPU) or
# reads them as zero computes the same bits. FLT_MIN itself goes too because a product whose
# exact value lies in (2**-126 - 2**-150, 2**-126) rounds up to 2**-126 under IEEE, where a
# flush-to-zero that judges tininess before rounding returns 0 (x86 does, for about one such
# product in twelve): with <=, either judgement gives 0. A sum, difference or quotient of
# normal fp32 values never lands strictly in that window (its exact value is a multiple of a
# step at least 2**-149 that the window does not contain), so only products meet it.
#
#   g, m, v, h, l = F(the five stored bf16 values, as fp32)
#   m1 = F(m + F(c1 * F(g - m)))                      torch's lerp, unfused
#   v1 = F(F(v * b2) + F(F(c2 * g) * g))              mul_ then addcmul, unfused
#   w  = F(F(h + l) * dec)                            the held weight, decayed
#   d  = F(F(F(sqrt(v1)) / bc) + eps)
#   w  = F(w + F(F(ns * m1) / d))
#   h' = w rounded to the nearest bf16 (even on a tie); 0x7fc0 where w is NaN
#   l' = 0 where h' is not finite, else S(F(w - h'), the remainder's bits)
#   m' = S(m1, m's bits), v' = S(v1, v's bits)
#
# S is ``_stochastic_round_bf16`` with ``_keyed_bits``' draw. Why no fused multiply-add: both
# kernels can be held to plain operations (Triton's enable_fp_fusion=False emits mul.rn and
# add.rn, which ptxas does not contract; Metal's fp contract(off)), and the reference can then
# be written in torch's basic operations, which are single IEEE operations on every CPU build,
# where a fused multiply-add is only computed exactly by some builds (torch's lerp_ and addcmul_
# fuse in the AVX2 and AVX-512 kernels and round twice in the default one).
#
# Where F parts the rule from fp32 AdamW, and by how much. Every value F zeroes is at most 2**-126,
# and every term it can drop from the weight's update is such a value divided by d >= eps (or
# multiplied by the step before that division). With eps >= 2**-39 / sqrt(1 - beta2) (refused
# below it) and lr <= 1 (refused above it, MAX_LR), a dropped term moves a weight by at most
# 2**-126 / eps <= 2**-87, or lr / (1 - beta1**t) * 2**-87 where a flushed g or m1 would have
# been scaled up by the step: under 2**-77 per update at beta1 <= 0.999 (more only at a beta1
# closer to 1, early in a run). So in each update the rule and fp32 AdamW agree to within a
# rounding (2**-24 of it) of every weight above 2**-53 in magnitude, which is every weight a run
# trains; below that the update's absolute difference is still under 2**-77. Not
# covered, by design: weights and moments within a bf16 step of 2**-126 (1.2e-38), where the
# stochastic rounding stores 2**-126 itself and the next step's F reads it as zero (a bias of at
# most 1.2e-38 there, and nowhere else). An lr above 1 is refused rather than covered: with
# eps at its floor, lr near 2**116 turns a flushed gradient of 2**-126 into a step of 2**29.

# MAX_LR (poros.core.checkpoints, imported above): the largest lr taken, for the reason above

FLT_MIN = 2.0 ** -126  # the smallest normal fp32: it and anything smaller in magnitude is 0
FLT_MAX = 3.4028234663852886e38  # the largest finite fp32, (2 - 2**-23) * 2**127


def _f32(x: float) -> float:
    """A Python float rounded to the nearest fp32 (even on a tie), then F, as every other value
    of the rule: the rule's scalars. One that rounds past fp32's largest is refused in words
    (``_scalar_refusal`` refuses it, and a nonzero one F would zero, before any update)."""
    try:
        r = struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        r = math.copysign(math.inf, x)
    if math.isinf(r) and not math.isinf(x):
        raise ValueError(f"ExpertAdamW: a scalar of the update is {x!r}, past fp32's largest")
    return 0.0 if abs(r) <= FLT_MIN else r


def _scalar_refusal(lr: float, b1: float, b2: float, eps: float, wd: float) -> str | None:
    """Why these settings give a scalar of the rule that is not an fp32 normal or zero (one that
    overflows fp32, or a nonzero one F would read as zero, which a GPU that flushes subnormals
    reads as zero and a CPU does not), or None. 1 - beta1, beta2, 1 - beta2, 1 - lr * weight_decay
    and eps are the same at every update; sqrt(1 - beta2**t) lies in [sqrt(1 - beta2), 1], and the
    step lr / (1 - beta1**t) falls from lr / (1 - beta1) at the first update to lr, both ends
    checked (rounding to fp32 keeps the order). lr * weight_decay alone is no scalar: a tiny one
    leaves 1 - lr * weight_decay at 1."""
    with_t = (("1 - beta1", 1 - b1), ("beta2", b2), ("1 - beta2", 1 - b2),
              ("1 - lr * weight_decay", 1 - lr * wd), ("eps", eps),
              ("sqrt(1 - beta2)", math.sqrt(1 - b2)),
              ("lr / (1 - beta1) (the first update's step)", lr / (1 - b1)),
              ("lr (the step it falls to)", lr))
    for what, x in with_t:
        try:
            r = struct.unpack("<f", struct.pack("<f", x))[0]
        except OverflowError:
            r = math.inf
        if math.isinf(r) or math.isnan(r):
            return (f"{what} is {x!r}, past fp32's largest (3.4e38): the update would be "
                    "infinite")
        # judged on the exact value: a nonzero one that fp32 rounds to 0 (below 2**-150) is
        # refused as one that lands at or under 2**-126, never zeroed unsaid
        if x != 0 and abs(r) <= FLT_MIN:
            return (f"{what} is {x!r}, nonzero but at most 2**-126 (1.18e-38) as fp32, which "
                    "the rule reads as zero (as a GPU that flushes subnormals does): set it to "
                    "0 or above 2**-126")
    return None


def _scalars(lr: float, b1: float, b2: float, eps: float, wd: float, t: int) -> dict:
    """The rule's fp32 scalars of one update (``_f32``). ``b2 ** t`` and ``b1 ** t`` are Python's
    power, the C library's pow, which is not correctly rounded everywhere: two machines' libraries
    can differ in the last bit of a scalar. The rule's bits are one function of its inputs on one
    machine (the kernel and the reference read these same scalars, computed once on the CPU), so
    the kernels' check against the reference holds; a resume on another machine continues with
    that machine's scalars, as it does with its torch."""
    return {"w1": _f32(1 - b1), "b2": _f32(b2), "w2": _f32(1 - b2), "decay": _f32(1 - lr * wd),
            "bc2s": _f32(math.sqrt(1 - b2 ** t)), "eps": _f32(eps),
            "neg_step": _f32(-lr / (1 - b1 ** t))}


def _flush(x: torch.Tensor) -> torch.Tensor:
    """F: +0 where |x| <= 2**-126 (a NaN stays a NaN)."""
    return torch.where(x.abs() <= FLT_MIN, torch.zeros((), dtype=x.dtype, device=x.device), x)


def _correct(q: torch.Tensor, too_small, too_large) -> torch.Tensor:
    """q (fp32, > 0) moved one fp32 step up where ``too_small(midpoint above)`` or down where
    ``too_large(midpoint below)``: each midpoint exact in fp64 (25 significant bits)."""
    up = torch.nextafter(q, torch.full_like(q, math.inf))
    down = torch.nextafter(q, torch.zeros_like(q))
    q64 = q.double()
    q = torch.where(too_small((q64 + up.double()) * 0.5), up, q)
    return torch.where(too_large((q64 + down.double()) * 0.5), down, q)


def _sqrt(x: torch.Tensor) -> torch.Tensor:
    """The correctly rounded fp32 square root, whatever the build's sqrt: a candidate within an
    fp32 step (fp64's root, rounded), corrected by comparing the square of each neighbouring
    midpoint with x, exactly in fp64 (a 25-bit midpoint squares to at most 50 bits). No midpoint
    squares to an fp32 value (its square is odd with 49 or more significant bits), so a tie
    cannot occur. x >= 0."""
    s = torch.sqrt(x.double()).float()
    x64 = x.double()
    fixed = _correct(s, lambda mid: mid * mid < x64, lambda mid: mid * mid > x64)
    return torch.where(torch.isfinite(x) & (x > 0), fixed, s)


def _div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The correctly rounded fp32 quotient, whatever the build's division: a candidate within an
    fp32 step (fp64's quotient, rounded), its magnitude corrected by comparing each neighbouring
    midpoint times |b| with |a|, exactly in fp64 (25 + 24 bits). No quotient of two fp32 values
    is an fp32 midpoint (that midpoint times b would need more significant bits than a has), so a
    tie cannot occur. b > 0."""
    q = (a.double() / b.double()).float()
    aa, bb = a.double().abs(), b.double().abs()
    mag = _correct(q.abs(), lambda mid: mid * bb < aa, lambda mid: mid * bb > aa)
    fixed = torch.where(a < 0, -mag, mag)
    q = torch.where(torch.isfinite(q) & (q != 0) & torch.isfinite(a), fixed, q)
    # Past the largest: fp64's quotient could round up onto 2**128 - 2**103, the midpoint
    # between FLT_MAX and 2**128, which fp32 then rounds to inf (to even), where the exact
    # quotient lies below it and rounds to FLT_MAX. Compared exactly (25 + 24 bits): FLT_MAX
    # where the midpoint times |b| exceeds |a|. (No pair of fp32 values meets it on this side:
    # tests/moe/test_expert_adamw_hardening.py searches every divisor's significand; the
    # comparison makes it so by construction.) The other end needs nothing: F reads every
    # result at most 2**-126 as zero, so a subnormal quotient's rounding does not matter.
    over = torch.isinf(q) & torch.isfinite(a) & torch.isfinite(b)
    if bool(over.any()):
        below = (2.0 ** 128 - 2.0 ** 103) * bb > aa
        flt_max = torch.where(a < 0, -FLT_MAX, FLT_MAX).float()
        q = torch.where(over & below, flt_max, q)
    return q


def _reference_bf16(h16, l16, g16, m16, v16, key: int, start: int, s: dict, out) -> None:
    """The rule, in torch's basic operations, on n elements of one parameter (``start``: the
    first one's index in it, for the bits); ``out``: (h, l, m, v) bf16 views written in place.
    Its bits do not depend on the CPU build or the device: +, - and * are single IEEE
    operations in every build, and the square root and division are corrected to the correctly
    rounded result (``_sqrt``, ``_div``). Needs fp64 on its device (not a Mac's GPU)."""
    dev, n = h16.device, h16.numel()
    g, m, v = _flush(g16.float()), _flush(m16.float()), _flush(v16.float())
    h, lo = _flush(h16.float()), _flush(l16.float())
    m1 = _flush(m + _flush(s["w1"] * _flush(g - m)))
    v1 = _flush(_flush(v * s["b2"]) + _flush(_flush(s["w2"] * g) * g))
    w = _flush(_flush(h + lo) * s["decay"])
    bc = torch.full_like(v1, s["bc2s"])
    d = _flush(_flush(_div(_flush(_sqrt(v1)), bc)) + s["eps"])
    w = _flush(w + _flush(_div(_flush(s["neg_step"] * m1), d)))
    hi = w.bfloat16()
    hi = torch.where(torch.isnan(w), torch.tensor(0x7FC0, dtype=torch.int16, device=dev)
                     .view(torch.bfloat16), hi)
    r = _flush(w - hi.float())
    rem = _stochastic_round_bf16(r, _keyed_bits(key, start, n, DRAW_REMAINDER, dev))
    rem = torch.where(torch.isfinite(hi), rem, torch.zeros((), dtype=rem.dtype, device=dev))
    oh, ol, om, ov = out
    om.copy_(_stochastic_round_bf16(m1, _keyed_bits(key, start, n, DRAW_M, dev)))
    ov.copy_(_stochastic_round_bf16(v1, _keyed_bits(key, start, n, DRAW_V, dev)))
    oh.copy_(hi)
    ol.copy_(rem)


def _cuda_step(hi, lo, g, m, v, coef, s) -> None:
    from poros.train import expert_adamw_cuda

    expert_adamw_cuda.step(hi, lo, g, m, v, coef, **s)


def _mps_step(hi, lo, g, m, v, coef, s) -> None:
    from poros.metal import expert_adamw_shader

    scalars = torch.tensor([s["w1"], s["b2"], s["w2"], s["decay"], s["bc2s"], s["eps"],
                            s["neg_step"]], dtype=torch.float32).to(hi.device)
    expert_adamw_shader.library().expert_adamw_bf16(hi, lo, g, m, v, coef, scalars,
                                                    hi.numel(), threads=[hi.numel()])


_KERNEL = {"cuda": _cuda_step, "mps": _mps_step}
KERNELS = True  # False: every device runs the reference (a check's control, never a run's)


def _kernel_unavailable(kind: str) -> str | None:
    """Why a device type has no fused kernel to try, or None."""
    if not KERNELS or kind not in _KERNEL:
        return "no fused kernel for this device" if KERNELS else "kernels turned off"
    if kind == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError:
            return "Triton is not installed"
        return None
    if not hasattr(torch.mps, "compile_shader"):
        return "this torch cannot compile a Metal shader"
    if os.environ.get("PYTORCH_MPS_FAST_MATH", "").strip() not in ("", "0"):
        return "PYTORCH_MPS_FAST_MATH is set, and fast math cannot be held to the rule"
    return None


# ---- the probe: inputs that exercise every operation, and the comparison of two runs -----------


def _probe_inputs(n: int, seed: int) -> dict:
    """bf16 (h, l, g, m, v) for n elements: magnitudes over many exponents, both signs, zeros,
    subnormals, 2**-126 and values around it, the largest finite values, infinities, a NaN, a
    weight that rounds to inf; the rest random. ``_plant_edges`` adds the elements that need
    the special settings of ``PROBE_SETTINGS`` to land on an edge."""
    gen = torch.Generator().manual_seed(seed)

    def spread(scale_lo: int, scale_hi: int) -> torch.Tensor:
        mant = torch.rand(n, generator=gen) + 0.5
        exp = torch.randint(scale_lo, scale_hi, (n,), generator=gen).float()
        sign = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
        return (sign * mant * torch.exp2(exp)).bfloat16()

    h = spread(-20, 2)
    lo = (h.float() * torch.exp2(-torch.randint(9, 30, (n,), generator=gen).float())
          * (torch.rand(n, generator=gen) - 0.5)).bfloat16()
    g = spread(-40, -2)
    m = spread(-40, -4)
    v = spread(-80, -8).abs()
    special = [0.0, -0.0, 2.0 ** -126, 2.0 ** -127, -(2.0 ** -130), 2.0 ** -133, 1e-40, -1e-39,
               3.3895313892515355e38, -3.3895313892515355e38, 1.0, -1.0, 1.0078125, 0.5,
               2.0 ** -125, 2.0 ** -126 * 1.0078125]
    k = len(special)
    stride = max(1, n // (8 * k))
    for i, x in enumerate(special):  # each special value in each input, at spread positions
        for t, name in enumerate((h, lo, g, m, v)):
            j = (i * 5 + t) * stride % n
            name[j] = abs(x) if name is v else x
    v[n // 3] = float("inf")
    g[n // 2] = float("nan")
    h[n // 5], lo[n // 5] = 3.3895313892515355e38, 2.0 ** 119  # rounds to inf: no remainder
    out = {"h": h, "l": lo, "g": g, "m": m, "v": v}
    _plant_edges(out)
    return out


# The probe's settings (lr, beta1, beta2, eps, weight decay, steps). The first is an ordinary
# run's. The second puts three products strictly inside (2**-126 - 2**-150, 2**-126), the window
# F's boundary is about: each scalar c below is the fp32 just under 1 / (1 + k/128), and
# c * (1 + k/128) * 2**-126 (the planted operand times it) falls short of 2**-126 by 0.0625,
# 0.53 and 0.13 of 2**-150, so IEEE rounds it up to 2**-126 (found by exact search over k;
# tests/moe/test_expert_adamw_kernel.py checks it). 1 - beta1 = C1, beta2 = C2 and
# 1 - lr * weight_decay = C3 are each exact. The third (lr = 0) leaves the planted weights where
# they are, so their rounding to bf16 meets an exact tie.
EDGE_C1, EDGE_C2, EDGE_C3 = 0.9922480583190918, 0.9696969389915466, 0.9343065619468689
EDGE_K = (1, 4, 9)  # the planted operands: (1 + k/128) * 2**-126, one per scalar
PROBE_SETTINGS = (
    (2e-4, 0.9, 0.999, 1e-8, 0.01, 3),
    (2.0 ** -10, 1 - EDGE_C1, EDGE_C2, 1e-8, (1 - EDGE_C3) * 2.0 ** 10, 1),
    (0.0, 0.9, 0.999, 1e-8, 0.01, 1),
)
EDGE = 7  # the first planted element's index; each takes a run of ten


def _plant_edges(t: dict) -> None:
    """The elements ``PROBE_SETTINGS`` needs: each input zero but the one that matters.
    EDGE + 0: m1 = (1 - beta1) * g, g = (1 + 1/128) * 2**-126; EDGE + 1: v * beta2, v = (1 +
    4/128) * 2**-126; EDGE + 2: (h + l) * decay, h = (1 + 9/128) * 2**-126, and its negative at
    EDGE + 3 (each in F's window under setting 2); EDGE + 4..7: h + l on a bf16 tie, rounding
    down to even (1 + 2**-8 over 1.0) and up to even (1 + 3 * 2**-8 over 1.0078125), and their
    negatives (setting 3); EDGE + 8: 2**-126 itself in every input."""
    for k in t:
        t[k][EDGE:EDGE + 10] = 0.0
    k1, k2, k3 = ((1 + k / 128) * 2.0 ** -126 for k in EDGE_K)
    t["g"][EDGE] = k1
    t["v"][EDGE + 1] = k2
    t["h"][EDGE + 2], t["h"][EDGE + 3] = k3, -k3
    ties = ((1.0, 2.0 ** -8), (1.0078125, 2.0 ** -8), (-1.0, -(2.0 ** -8)),
            (-1.0078125, -(2.0 ** -8)))
    for i, (hi, lo) in enumerate(ties):
        t["h"][EDGE + 4 + i], t["l"][EDGE + 4 + i] = hi, lo
    for k in t:
        t[k][EDGE + 8] = 2.0 ** -126


def _run_steps(kind: str, device: Any, inputs: dict, steps: int, key0: int, s: dict) -> dict:
    """``steps`` updates of the probe inputs (state fed back, the grad kept) by the kernel
    (``kind`` "kernel") or the reference, on ``device``; the stored bf16 tensors, on the CPU."""
    t = {k: x.clone().to(device) for k, x in inputs.items()}
    n = t["h"].numel()
    for step in range(steps):
        key = key0 + step
        if kind == "kernel":
            coef = _coefficients(key, n, device)
            _KERNEL[torch.device(device).type](t["h"], t["l"], t["g"], t["m"], t["v"], coef, s)
        else:
            for i in range(0, n, CHUNK):  # a chunk at a time, as the optimizer runs it
                c = {k: x[i:i + CHUNK] for k, x in t.items()}
                _reference_bf16(c["h"], c["l"], c["g"], c["m"], c["v"], key, i, s,
                                (c["h"], c["l"], c["m"], c["v"]))
    return {k: x.cpu() for k, x in t.items()}


def _first_difference(got: dict, want: dict, inputs: dict) -> str | None:
    """None where every stored tensor is equal bit for bit, else the first element that is not,
    with its inputs and both outputs."""
    for k in ("h", "l", "m", "v"):
        a, b = got[k].view(torch.int16), want[k].view(torch.int16)
        bad = (a != b).nonzero().flatten()
        if bad.numel():
            i = int(bad[0])
            ins = ", ".join(f"{x}={float(inputs[x][i])!r}" for x in ("h", "l", "g", "m", "v"))
            outs = ", ".join(f"{x}: {float(got[x][i])!r} against {float(want[x][i])!r}"
                             for x in ("h", "l", "m", "v"))
            return (f"{int(bad.numel())} elements of {k} differ; the first, element {i}: "
                    f"inputs {ins}; outputs (kernel against reference) {outs}")
    return None


KERNEL_DIFFERS = "the kernel does not store the reference's bits"
_PROBED: dict = {}  # a device's verdict, kept for the process: only a definite one


def _free_memory(dev: torch.device) -> None:
    """What the process holds and no longer uses, let go: Python's garbage, and the device's
    cached blocks."""
    import gc

    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()


def _probe_kernel(device: Any) -> tuple[bool, str]:
    """Whether the device's fused kernel stores the reference's bits over the probe
    (``_probe_differences``): (ok, in words). Once a device per process.

    A probe that raises (out of memory, a driver's hiccup) is no verdict on the kernel: it runs
    again once, after the process's unused memory is freed, and if it raises again the run stops
    in words. It is never taken as a reason to step on the slower reference without being asked,
    and never kept: the next optimizer in the process probes afresh. Only a definite answer is
    kept: the kernel's bits are the reference's, or they are not (the kernel then is not used,
    loudly: ``_implementation``, ``ExpertAdamW._measure_fallbacks``)."""
    dev = torch.device(device)
    name = str(dev)
    if name in _PROBED:
        return _PROBED[name]
    why = _kernel_unavailable(dev.type)
    if why is not None:
        _PROBED[name] = (False, why)
        return _PROBED[name]
    try:
        diff = _probe_differences(dev)
    except Exception as first:  # noqa: BLE001 - said, then tried once more
        logger.warning("ExpertAdamW's check of its kernel on %s raised %s: %s; freeing memory "
                       "and checking once more", dev, type(first).__name__, first)
        _free_memory(dev)
        try:
            diff = _probe_differences(dev)
        except Exception as exc:
            raise RuntimeError(
                f"ExpertAdamW could not check its fused kernel on {dev}: the check raised "
                f"{type(first).__name__} ({first}), and again after the process's unused memory "
                f"was freed: {type(exc).__name__}: {exc}. The run stops rather than step the "
                "adapters on the slower reference unasked. Free the device's memory (another "
                "process may hold it) and start the run again") from exc
    _PROBED[name] = (diff is None, "the kernel stores the reference's bits" if diff is None
                     else f"{KERNEL_DIFFERS}: {diff}")
    return _PROBED[name]


def _probe_differences(device: Any) -> str | None:
    """The kernel against the reference over the probe: the first difference in words, or None.
    Each setting of ``PROBE_SETTINGS`` at an element count that is a multiple of 16 and at one
    that is not (Triton compiles each its own variant: vector loads for the first; the bits'
    block arithmetic is the same code in both), the ordinary setting across a block boundary at
    the count real tensors have, and small tensors for the planted edges. Its CPU side, measured:
    about 100 MiB above the start and 1.4 s on 4 threads (it held 509 MiB unchunked)."""
    key0 = update_key(0, 1, "the kernel's probe")
    for i, (lr, b1, b2, eps, wd, steps) in enumerate(PROBE_SETTINGS):
        s = _scalars(lr, b1, b2, eps, wd, 7)
        for n in ((BLOCK + 1040, 65536 + 1037) if i == 0 else (4096, 4099)):
            inputs = _probe_inputs(n, i)
            got = _run_steps("kernel", device, inputs, steps, key0 + 10 * i, s)
            want = _run_steps("reference", "cpu", inputs, steps, key0 + 10 * i, s)
            diff = _first_difference(got, want, inputs)
            if diff is not None:
                return f"setting {i}, {n} elements: {diff}"
    return None


def _reference_seconds(device: Any) -> float:
    """Seconds the reference takes for one chunk (CHUNK elements, one update) on ``device``,
    timed now: the faster of two runs after a first."""
    inputs = _probe_inputs(CHUNK, 5)
    s = _scalars(2e-4, 0.9, 0.999, 1e-8, 0.01, 7)
    dev = torch.device(device)
    if dev.type == "cuda":  # that card, not the current one
        def sync() -> None:
            torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        sync = torch.mps.synchronize
    else:
        def sync() -> None:
            return None
    best = math.inf
    for i in range(3):
        sync()
        start = time.perf_counter()
        _run_steps("reference", device, inputs, 1, i, s)
        sync()
        if i:
            best = min(best, time.perf_counter() - start)
    return best


def _implementation(device: Any) -> dict:
    """What updates the bf16 parameters on ``device``, chosen once: the fused kernel where it
    exists and stores the reference's bits, else the reference, on the device where it has fp64
    (CUDA), on the CPU where it does not (a Mac's GPU). Every choice gives the same bits."""
    dev = torch.device(device)
    if dev.type == "cpu":
        return {"device": "cpu", "path": "reference", "why": "the CPU runs the reference"}
    ok, why = _probe_kernel(dev)
    if ok:
        return {"device": dev.type, "path": "kernel", "why": why}
    path = "reference on the CPU" if dev.type == "mps" else "reference"
    logger.warning("ExpertAdamW on %s runs the %s, slower than its kernel: %s", dev, path, why)
    # differs: the probe ran the kernel and found other bits (not an absent Triton or fast math)
    return {"device": dev.type, "path": path, "why": why,
            "differs": why.startswith(KERNEL_DIFFERS)}


def _refuse(named: list, *, lr: float, betas: Any, eps: float, weight_decay: float, seed: int,
            chunk: int) -> None:
    """What ExpertAdamW refuses at construction, before any state exists, in words."""
    names = [n for n, _ in named]
    if len(set(names)) != len(names):
        twice = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"ExpertAdamW: the name {twice[0]!r} is given to more than one "
                         "parameter; the names key the random bits, so each must be its own")
    if len({id(p) for _, p in named}) != len(named):
        raise ValueError("ExpertAdamW: a parameter is given twice")
    for name, p in named:
        if p.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"ExpertAdamW: {name} is {p.dtype}; it updates bf16 and fp32")
        if not p.is_contiguous():
            raise ValueError(f"ExpertAdamW: {name} is not contiguous; the step works on a "
                             "parameter's elements as one flat run")
        if p.numel() >= MAX_ELEMENTS:
            raise ValueError(f"ExpertAdamW: {name} has {p.numel()} elements; its keys index at "
                             "most 2**32 - 1")
    _check_settings(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if not (isinstance(seed, int) and not isinstance(seed, bool) and SEEDS[0] <= seed < SEEDS[1]):
        raise ValueError(f"ExpertAdamW: seed {seed!r}; the keys take a signed 64-bit integer "
                         "seed, as the run's other keyed states do (KeyedRng)")
    if chunk < 1:
        raise ValueError(f"ExpertAdamW: chunk {chunk!r} must be at least 1")


def _check_settings(*, lr: Any, betas: Any, eps: Any, weight_decay: Any) -> None:
    """lr, betas, eps and weight decay as construction takes them, else ValueError in words: at
    construction, and on the settings a saved state brings (``load_state_dict``)."""
    try:
        b1, b2 = (float(b) for b in betas)
        lr, eps, weight_decay = float(lr), float(eps), float(weight_decay)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ExpertAdamW: lr {lr!r}, betas {betas!r}, eps {eps!r} and "
                         f"weight_decay {weight_decay!r} must be numbers ({exc})") from None
    if math.isfinite(lr) and lr > MAX_LR:
        raise ValueError(
            f"ExpertAdamW: lr = {lr!r} is above 1, the largest it takes: there a gradient or a "
            "moment its rule reads as zero (at most 2**-126) would have moved a weight by more "
            "than a rounding in fp32 AdamW (an lr above 1 is far past any run's)")
    for what, value, ok in (("lr", lr, lr >= 0), ("eps", eps, eps >= 0),
                            ("weight_decay", weight_decay, weight_decay >= 0),
                            ("betas[0]", b1, 0 <= b1 < 1), ("betas[1]", b2, 0 <= b2 < 1)):
        if not (math.isfinite(float(value)) and ok):
            raise ValueError(f"ExpertAdamW: {what} = {value!r}; lr, eps and weight_decay must "
                             "be finite and at least 0, and each beta in [0, 1)")
    from poros.core.checkpoints import min_adam_epsilon

    if eps < min_adam_epsilon(b2):
        raise ValueError(
            f"ExpertAdamW: eps = {eps!r} is under {min_adam_epsilon(b2):.3g}, the least at which "
            f"its rule (which reads values at most 2**-126 as zero) and fp32 AdamW agree to within "
            f"a rounding at beta2 {b2:g} (2**-39 / sqrt(1 - beta2)); below it a gradient whose "
            "square it reads as zero can take an enormous step")
    said = _scalar_refusal(lr, b1, b2, eps, weight_decay)
    if said is not None:
        raise ValueError(f"ExpertAdamW: {said}")


STATE_FIX = ("Resume from a checkpoint this run wrote, unchanged, or start a new run from its "
             "adapter (--from <checkpoint>).")


def _check_saved(name: str, p: torch.Tensor, saved: dict) -> None:
    """One parameter's saved state against the parameter, before anything is restored: every
    tensor the rule keeps (m, v, and a bf16 parameter's remainder), of its shape and dtype, and
    a whole step count at least 0. Refused in words, else a wrong shape would reach a kernel
    that reads it as a flat run of the parameter's elements."""
    from poros.core.errors import PorosConfigError

    want = ("exp_avg", "exp_avg_sq") + (("remainder",) if p.dtype == torch.bfloat16 else ())
    for k in want:
        value = saved.get(k)
        if not torch.is_tensor(value):
            raise PorosConfigError(f"This optimizer state has no {k} tensor for {name}.",
                                   remediation=STATE_FIX)
        if value.shape != p.shape or value.dtype != p.dtype:
            raise PorosConfigError(
                f"This optimizer state's {k} for {name} is {value.dtype} of shape "
                f"{tuple(value.shape)} ({value.numel()} elements), and {name} is {p.dtype} of "
                f"shape {tuple(p.shape)} ({p.numel()} elements): one is not the other's state.",
                remediation=STATE_FIX)
    step = saved.get("step")
    if not (torch.is_tensor(step) and step.dtype == torch.int64 and step.numel() == 1
            and 0 <= int(step) <= MAX_STEP):
        raise PorosConfigError(
            f"This optimizer state's step count for {name} is {step!r}; this optimizer saves "
            f"it as one int64 tensor from 0 to 2**62 (the keys pack the update as a signed "
            "64-bit integer, and 2**62 leaves room for any run's remaining updates).",
            remediation=STATE_FIX)


def _own(x: torch.Tensor) -> torch.Tensor:
    """x's values in a contiguous tensor of their own, copied once (``contiguous`` copies a
    tensor that is not contiguous, and returns one that is, offset in a larger buffer, as it is:
    that one is cloned)."""
    return x.contiguous() if not x.is_contiguous() else x.clone()


def _check_grad(name: str, p: torch.Tensor) -> None:
    """A gradient of the parameter's dtype, shape and device, refused in words before it reaches
    a kernel (one reads it as the parameter's elements, flat)."""
    g = p.grad
    if g.dtype != p.dtype or g.shape != p.shape or g.device != p.device:
        raise RuntimeError(
            f"ExpertAdamW: the gradient of {name} is {g.dtype} of shape {tuple(g.shape)} on "
            f"{g.device}, and {name} is {p.dtype} of shape {tuple(p.shape)} on {p.device}: "
            "a gradient must match its parameter in all three")


# ---- the optimizer ------------------------------------------------------------------------


class ExpertAdamW(torch.optim.Optimizer):
    """torch's AdamW (decoupled weight decay, bias-corrected), for bf16 parameters without the
    losses of bf16 arithmetic. ``named_params``: (name, parameter) pairs, named as the model
    names them (the names key the random bits); ``seed``: the run's."""

    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]], *, lr: float,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01, seed: int = 0, chunk: int = CHUNK) -> None:
        named = list(named_params)
        try:  # read twice below: a generator would be empty the second time
            betas = tuple(betas)
        except TypeError:  # not a pair at all (betas=0.9): refused as every bad setting is
            raise ValueError(f"ExpertAdamW: betas {betas!r} must be a pair of numbers, each in "
                             "[0, 1)") from None
        _refuse(named, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, seed=seed,
                chunk=chunk)
        self._built = False
        super().__init__([{"params": [p for _, p in named], "names": [n for n, _ in named]}],
                         {"lr": lr, "betas": tuple(betas), "eps": eps,
                          "weight_decay": weight_decay, "rule": RULE, "seed": int(seed),
                          "block": BLOCK})
        self._built = True
        self._names = {id(p): name for name, p in named}
        self.seed, self.chunk = int(seed), int(chunk)
        # chosen once, for every device that holds a bf16 parameter (``_implementation``)
        devices = dict.fromkeys(str(p.device) for _, p in named if p.dtype == torch.bfloat16)
        self.implementation = {d: _implementation(torch.device(d)) for d in devices}
        self.notice = self._measure_fallbacks(named)

    def _measure_fallbacks(self, named: list) -> str | None:
        """Where a GPU's bf16 parameters are stepped by the reference, not the kernel: its cost
        measured here and now (one chunk timed, scaled to the parameters' elements), recorded
        in ``implementation`` and said in words. None where every GPU runs its kernel."""
        said = []
        for device, impl in self.implementation.items():
            if impl["device"] == "cpu" or impl["path"] == "kernel":
                continue
            n = sum(p.numel() for _, p in named
                    if p.dtype == torch.bfloat16 and str(p.device) == device)
            where = "cpu" if impl["path"] == "reference on the CPU" else device
            seconds = _reference_seconds(where) * n / CHUNK
            impl["elements"], impl["seconds_per_update"] = n, float(f"{seconds:.3g}")
            said.append(
                f"ExpertAdamW steps the {n:,} bf16 trainable elements on {device} with its "
                f"{impl['path']}, not its fused kernel: {impl['why']}. Measured on this machine "
                f"now: about {seconds:.1f} s per update for the optimizer's step ("
                + ("the run stores the rule's bits, which the kernel would not have" if
                   impl.get("differs") else
                   "the same bits as the kernel would store; only the time differs") + ")")
        return "; ".join(said) or None

    def add_param_group(self, param_group: dict) -> None:
        """Only the constructor's own group: a group added later would have no names, and the
        names key the random bits."""
        if getattr(self, "_built", False):
            raise ValueError("ExpertAdamW takes its parameters, named, at construction only: "
                             "build a new one with every (name, parameter) pair instead of "
                             "adding a group")
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict: dict) -> None:
        """Refuses, before anything is restored, a state this rule did not write (an older
        Poros's), or one written for other parameters: other names or another order (torch
        restores by position), another seed or another block size of the bits."""
        from poros.core.errors import PorosConfigError

        # the caller's dict is left as it was: its containers copied, every tensor moved into the
        # copy (on a Mac the caller's tensors are the file, mmap'd: holding them costs no memory)
        state_dict = {**state_dict,
                      "state": {i: dict(saved) if isinstance(saved, dict) else saved
                                for i, saved in (state_dict.get("state") or {}).items()},
                      "param_groups": [dict(g) for g in state_dict.get("param_groups") or []]}
        groups = state_dict["param_groups"]
        if not groups or any(g.get("rule") != RULE for g in groups):
            other = {g.get("rule") for g in groups} - {None, RULE}
            if other:
                raise PorosConfigError(
                    f"This optimizer state was written by the update rule {other.pop()}, and "
                    f"this Poros updates with {RULE}: one is not the other's state.",
                    remediation=OLD_STATE_FIX)
            # this optimizer steps bf16 tensors: the older state was bf16 ones' too
            raise PorosConfigError(old_state_message("this optimizer state", True),
                                   remediation=OLD_STATE_FIX)
        ours = [n for g in self.param_groups for n in g["names"]]
        theirs = [n for g in groups for n in (g.get("names") or [])]
        if theirs != ours:
            first = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b),
                         min(len(ours), len(theirs)))
            raise PorosConfigError(
                f"This optimizer state is for {len(theirs)} parameters and this optimizer has "
                f"{len(ours)}; the first that differs is number {first}: "
                f"{theirs[first] if first < len(theirs) else 'none'} in the state, "
                f"{ours[first] if first < len(ours) else 'none'} here. A state is restored by "
                "position, so it would land on other parameters.",
                remediation="Build the optimizer over the same named parameters, in the same "
                "order, as the run that wrote the state.")
        for key, here in (("seed", self.seed), ("block", BLOCK)):
            for g in groups:
                if g.get(key) != here:
                    raise PorosConfigError(
                        f"This optimizer state was written with {key} {g.get(key)!r}, and this "
                        f"optimizer has {here!r}: its keyed rounding would continue on other "
                        "bits.", remediation=f"Resume with the {key} the state was written with.")
        for g in groups:  # torch restores the saved settings: each as construction takes it
            try:
                _check_settings(lr=g.get("lr"), betas=g.get("betas"), eps=g.get("eps"),
                                weight_decay=g.get("weight_decay"))
            except ValueError as exc:
                raise PorosConfigError(f"This optimizer state's settings are refused: {exc}.",
                                       remediation=STATE_FIX) from None
        params = [p for g in self.param_groups for p in g["params"]]
        # torch restores state through the saved groups' "params" lists (saved number i goes to
        # the parameter at position i of those lists), not through the state's own keys: each
        # list must be exactly this optimizer's positions, in order, as torch numbers them when
        # it saves, and the state must hold exactly those positions (this optimizer writes an
        # entry for every parameter at its first step), so every check below is of the
        # parameter the entry lands on
        if len(groups) != len(self.param_groups):
            raise PorosConfigError(
                f"This optimizer state has {len(groups)} parameter groups and this optimizer "
                f"has {len(self.param_groups)}: the state is damaged or another optimizer's.",
                remediation=STATE_FIX)
        offset = 0
        for number, (here, g) in enumerate(zip(self.param_groups, groups)):
            want = list(range(offset, offset + len(here["params"])))
            got = g.get("params")
            if not isinstance(got, list) or got != want:
                shown = got if not isinstance(got, list) or len(got) <= 8 else got[:8] + ["..."]
                raise PorosConfigError(
                    f"This optimizer state's group {number} restores its entries onto parameters "
                    f"{shown!r}, and this optimizer's parameters there are numbered "
                    f"{offset} to {offset + len(want) - 1}, in order: its moments would land on "
                    "other parameters than the ones they were written for.",
                    remediation=STATE_FIX)
            offset += len(want)
        for index, saved in (state_dict.get("state") or {}).items():
            if (not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < len(params)):
                raise PorosConfigError(
                    f"This optimizer state holds an entry for parameter {index!r}, and this "
                    f"optimizer's parameters are numbered 0 to {len(params) - 1}: the state "
                    "is damaged or another optimizer's.", remediation=STATE_FIX)
        entries = state_dict["state"]
        # an entirely empty state is an optimizer's that has not stepped (poros.Run.save("start")):
        # loading it leaves this one un-stepped. Entries for some parameters and not others stay
        # refused; a checkpoint that says updates were taken and holds no state at all is the
        # checkpoint loader's to refuse (it knows the step: poros.train.checkpointing)
        missing = [i for i in range(len(params))
                   if not isinstance(entries.get(i), dict) or not entries[i]] if entries else []
        if missing:
            raise PorosConfigError(
                f"This optimizer state has no entry for {len(missing)} of its {len(params)} "
                f"parameters, the first {self._names[id(params[missing[0]])]} (number "
                f"{missing[0]}): it would restart their moments and step count unsaid.",
                remediation=STATE_FIX)
        for index, saved in state_dict["state"].items():
            half = params[index].dtype == torch.bfloat16
            if half != ("remainder" in saved):
                raise PorosConfigError(
                    f"This optimizer state was written for {'fp32' if half else 'bf16'} "
                    f"adapters, and {self._names[id(params[index])]} is "
                    f"{'bf16' if half else 'fp32'}: one is not the other's state (a bf16 "
                    "adapter's carries its weight's remainder, an fp32 one's none).",
                    remediation="Resume with the torch_dtype the state was written at, or start "
                    "a new run from its adapter (--from <checkpoint>).")
            _check_saved(self._names[id(params[index])], params[index], saved)
        # each tensor to its parameter's device one at a time, in the copy (a 16 GB Mac's
        # continuation peaked 0.7 GB above a whole run, the file mmap'd)
        # A tensor already where it belongs is copied, so the step's in-place updates never
        # reach the caller's (torch keeps such a tensor as it is): on a GPU that is the state
        # held twice until the caller lets its own go, at the resume, before any activation.
        for index, saved in state_dict["state"].items():
            for k, value in list(saved.items()):
                if torch.is_tensor(value):
                    where = "cpu" if k == "step" else params[index].device
                    moved = value.to(where)
                    if moved.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                        moved = moved.clone()
                    saved[k] = moved
        super().load_state_dict(state_dict)  # the groups' settings, as torch restores them
        # and the state by this optimizer's own positions, checked above, not torch's mapping
        self.state = defaultdict(dict, {params[i]: dict(saved)
                                        for i, saved in state_dict["state"].items()})
        for state in self.state.values():  # the counter lives on the CPU, as an int64
            state["step"] = state["step"].to("cpu", torch.int64)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:  # a scheduler sets lr between steps: checked here,
            try:  # every setting as construction checks it, before anything is updated
                _check_settings(lr=group["lr"], betas=group["betas"], eps=group["eps"],
                                weight_decay=group["weight_decay"])
            except ValueError as exc:
                said = str(exc).removeprefix("ExpertAdamW: ")
                raise ValueError(f"ExpertAdamW: at this step, {said}. Nothing was "
                                 "updated") from None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, group)
                elif not self.state[p]:
                    # a parameter with no gradient yet still gets its entry (zeros, its count at
                    # 0: what its first update starts from in any case), so a saved state holds
                    # every parameter and a load can refuse one that does not
                    self._new_state(p)
        return loss

    def _new_state(self, p: torch.Tensor) -> None:
        """A parameter's state before its first update: zeros, and its count at 0, an int64 on
        the CPU whatever the default dtype and device (a bf16 counter would stop at 256, and one
        on the GPU would make .item() wait on it every step)."""
        state = self.state[p]
        state["step"] = torch.zeros((), dtype=torch.int64, device="cpu")
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if p.dtype == torch.bfloat16:
            state["remainder"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _update(self, p: torch.Tensor, group: dict) -> None:
        if p.grad.is_sparse:
            raise RuntimeError("ExpertAdamW does not take sparse gradients")
        name = self._names[id(p)]
        _check_grad(name, p)
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        state = self.state[p]
        half = p.dtype == torch.bfloat16
        self._same_device(name, p, state, half)
        if not state:
            self._new_state(p)
        # what a kernel reads as p.numel() flat elements, checked here too, whatever set it
        # (p.data replaced, a state edited after a load): refused before anything is updated
        if not p.is_contiguous():
            raise RuntimeError(f"ExpertAdamW: {name} is no longer contiguous; the step works on "
                               "a parameter's elements as one flat run")
        for k in ("exp_avg", "exp_avg_sq", "remainder") if half else ("exp_avg", "exp_avg_sq"):
            if state[k].shape != p.shape or state[k].dtype != p.dtype:
                raise RuntimeError(
                    f"ExpertAdamW: {name}'s {k} is {state[k].dtype} of shape "
                    f"{tuple(state[k].shape)}, and {name} is {p.dtype} of shape "
                    f"{tuple(p.shape)}: one is not the other's state")
        state["step"] += 1
        t = int(state["step"].item())
        if p.numel() == 0:  # nothing to update (its count moves, as every parameter's does)
            return
        if not half:
            self._update_fp32(p, state, lr, b1, b2, eps, wd, t)
            return
        key = update_key(self.seed, t, self._names[id(p)])
        s = _scalars(lr, b1, b2, eps, wd, t)
        g = p.grad
        if not (g.is_contiguous() and g.storage_offset() == 0):
            g = _own(g)  # a view of a larger buffer: its own copy, same values
        for k in ("exp_avg", "exp_avg_sq", "remainder"):
            if not (state[k].is_contiguous() and state[k].storage_offset() == 0):
                state[k] = _own(state[k])  # once: then it stays its own
        m, v, lo = state["exp_avg"], state["exp_avg_sq"], state["remainder"]
        path = self.implementation[str(p.device)]["path"]
        if path == "kernel":
            hi = p if p.storage_offset() == 0 else p.detach().clone()
            _KERNEL[p.device.type](hi, lo, g, m, v, _coefficients(key, p.numel(), p.device), s)
            if hi is not p:
                p.data.copy_(hi)
            return
        on_cpu = path == "reference on the CPU"
        n = p.numel()
        views = [x.view(-1) for x in (p.detach(), lo, g, m, v)]
        for i in range(0, n, self.chunk):
            e = min(n, i + self.chunk)
            h_, l_, g_, m_, v_ = (x[i:e] for x in views)
            if on_cpu:
                out = [x.cpu() for x in (h_, l_, m_, v_)]
                _reference_bf16(out[0], out[1], g_.cpu(), out[2], out[3], key, i, s, out)
                for dst, src in zip((h_, l_, m_, v_), out):
                    dst.copy_(src)
            else:
                _reference_bf16(h_, l_, g_, m_, v_, key, i, s, (h_, l_, m_, v_))

    def _same_device(self, name: str, p: torch.Tensor, state: dict, half: bool) -> None:
        """A parameter moved to another device after the optimizer was built, or whose state is on
        another device, refused in words: the implementation was chosen and checked for the device
        it was on, and a kernel reads its state where the parameter is."""
        if half and str(p.device) not in self.implementation:
            raise RuntimeError(
                f"ExpertAdamW: {name} is on {p.device}, and the optimizer was built for its "
                f"parameters on {', '.join(self.implementation)}: build the optimizer after "
                "moving the model")
        for k in ("exp_avg", "exp_avg_sq", "remainder"):
            if k in state and state[k].device != p.device:
                raise RuntimeError(
                    f"ExpertAdamW: {name} is on {p.device} and its {k} on {state[k].device}: "
                    "move the model before building the optimizer or loading its state, not "
                    "after")

    def _update_fp32(self, p, state, lr, b1, b2, eps, wd, t) -> None:
        """An fp32 parameter in a mixed set: torch's AdamW math in torch's own ops, with m and v
        updated in place (as torch's single-tensor AdamW): an Inkling MoE run's fp32 attention
        adapters, beside its bf16 expert adapters."""
        bc1, bc2_sqrt = 1 - b1 ** t, math.sqrt(1 - b2 ** t)
        w, g = p.detach(), p.grad
        m, v = state["exp_avg"], state["exp_avg_sq"]
        m.lerp_(g, 1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        if wd != 0:
            w.mul_(1 - lr * wd)
        w.addcdiv_(m, v.sqrt().div_(bc2_sqrt).add_(eps), value=-lr / bc1)

    def weights(self) -> dict[str, torch.Tensor]:
        """Each parameter's weight as the optimizer holds it (hi + remainder, fp32), by name."""
        out = {}
        for group in self.param_groups:
            for p in group["params"]:
                lo = self.state[p].get("remainder") if p in self.state else None
                out[self._names[id(p)]] = p.detach().float() + (0 if lo is None else lo.float())
        return out


def build_optimizer(named_params: Sequence[tuple[str, nn.Parameter]], config: Any
                    ) -> torch.optim.Optimizer:
    """The trainer's optimizer for its trainable parameters. Every trainable tensor fp32 (a dense
    PEFT run, a dense native one, as lora_bias none leaves them): torch's AdamW, called exactly as
    before, so its numerics are unchanged. Any bf16 tensor among them: ``ExpertAdamW`` for all. A
    MoE model's expert adapters are bf16; so are a dense run's trainable biases where lora_bias
    trains a bf16 base's own biases and nothing casts them (a Mac's load keeps the pack's dtype;
    on CUDA PEFT's k-bit preparation casts them to fp32): torch's AdamW in bf16 would lose their
    small updates, the defect ExpertAdamW fixes, so they get it too, and the run says so."""
    named_params = list(named_params)  # read twice below: a generator would be empty the second
    params = [p for _, p in named_params]
    other = sorted({str(p.dtype) for p in params} - {"torch.float32", "torch.bfloat16"})
    if other:
        # A MoE run's fp16 is refused before its load (poros.preflight.start.moe_settings_
        # refusal). What still reaches here: a dense run whose lora_bias trains a base's own fp16
        # biases, which a Mac's load keeps in the pack's dtype; the pack, not the config, decides
        # that dtype, so it is known only once the model is loaded.
        from poros.core.errors import PorosConfigError

        first = next(n for n, p in named_params if str(p.dtype) in other)
        raise PorosConfigError(
            f"Trainable tensors in {', '.join(other)} (the first, {first}): Poros's optimizers "
            "update bfloat16 and float32 tensors only.",
            remediation="Set torch_dtype: bfloat16 (the default), or lora_bias: none so the "
            "base's own biases stay frozen.")
    if all(p.dtype == torch.float32 for p in params):
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
    try:
        opt = ExpertAdamW(named_params, lr=config.learning_rate,
                          betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
                          weight_decay=config.weight_decay, seed=config.seed)
    except ValueError as exc:
        # a dense run whose bf16 tensors bring ExpertAdamW passes no MoE preflight: its settings
        # are judged here, after the load, and refused in words as the preflight would
        from poros.core.errors import PorosConfigError

        said = str(exc).removeprefix("ExpertAdamW: ")
        setting = _setting_named(said)
        raise PorosConfigError(
            f"ExpertAdamW steps this run's bf16 trainable tensors, and it refuses its settings: "
            f"{said}.",
            remediation=(f"Change {setting} as the words above say." if setting else
                         "Change the setting the words above name.")) from None
    said = _why_not_torch(named_params)
    if said is not None:  # said at the start and in compute_reason, as a fallback's cost is
        logger.warning("%s", said)
        opt.notice = said if opt.notice is None else f"{said}; {opt.notice}"
    return opt


def _setting_named(said: str) -> str | None:
    """The config setting a refusal of ExpertAdamW's words is about, or None."""
    for word, setting in (("betas[0]", "adam_beta1"), ("1 - beta1", "adam_beta1"),
                          ("betas[1]", "adam_beta2"), ("beta2", "adam_beta2"),
                          ("eps", "adam_epsilon"), ("weight_decay", "weight_decay"),
                          ("lr", "learning_rate"), ("seed", "seed")):
        if said.startswith(word) or said.startswith(f"{word} "):
            return setting
    return None


def _why_not_torch(named_params: list) -> str | None:
    """Where bf16 trainable tensors other than LoRA factors (a dense run's biases that lora_bias
    trains) bring ExpertAdamW to a run, in words; None where every bf16 one is a LoRA factor (a
    MoE model's expert adapters)."""
    base = [n for n, p in named_params if p.dtype == torch.bfloat16 and "lora_" not in n]
    if not base:
        return None
    return (f"ExpertAdamW, not torch's AdamW, steps this run's trainable tensors: {len(base)} "
            f"of them are the base's own bf16 tensors (the first, {base[0]}), trained as "
            "lora_bias asks, and torch's AdamW in bf16 would lose their small updates. A run "
            "keeps torch's AdamW where every trainable tensor is fp32 (lora_bias: none)")


def rule_of(optimizer: Any) -> str:
    return RULE if isinstance(optimizer, ExpertAdamW) else TORCH_RULE


def trainable_dtype(optimizer: Any) -> str:
    """The dtype of the tensors an optimizer updates, by name (``mixed`` where they differ)."""
    dtypes = {str(p.dtype).removeprefix("torch.") for g in optimizer.param_groups
              for p in g["params"]}
    return dtypes.pop() if len(dtypes) == 1 else "mixed"


def _said(implementation: dict) -> str:
    return "; ".join(f"{device}: {i.get('path')}" for device, i in sorted(implementation.items()))


def other_implementation(recorded: Any, optimizer: Any) -> str | None:
    """A resume whose checkpoint's optimizer state was stepped by another implementation than
    this optimizer chose (RULE_FILE's ``implementation``: kernel or reference, by device).

    On a Mac it is refused: its kernel is checked against the reference by the probe on the
    probe's inputs, not proven equal on every input, so a run that changes implementation
    midway could continue on other bits than it would have. Returns the refusal in words. On
    CUDA (whose kernel and reference store the same bits by construction, checked the same way)
    it goes on: the change is said and recorded in the optimizer's implementation
    (``resumed_from``), which the next checkpoint's RULE_FILE carries. None where nothing
    changed, nothing was recorded, or the optimizer is not ExpertAdamW."""
    now = getattr(optimizer, "implementation", None)
    if not now or not isinstance(recorded, dict) or not recorded:
        return None
    was_set = {(i.get("device"), i.get("path")) for i in recorded.values() if isinstance(i, dict)}
    if was_set == {(i["device"], i["path"]) for i in now.values()}:
        return None
    was = _said({d: i for d, i in recorded.items() if isinstance(i, dict)})
    # a Mac on either side: a state its kernel stepped, continued elsewhere, as well as the
    # reverse; either way the run would continue on other bits than it would have
    if any(i["device"] == "mps" for i in now.values()) or any(
            isinstance(i, dict) and i.get("device") == "mps" for i in recorded.values()):
        fast = os.environ.get("PYTORCH_MPS_FAST_MATH")
        chose = "; ".join(f"{d}: its {i['path']} because {i.get('why')}"
                          for d, i in sorted(now.items()))
        return (f"its optimizer state was stepped by ExpertAdamW's {was}, and here it "
                f"steps with its {_said(now)}: a Mac's kernel is checked against the reference, "
                "not proven equal on every input, so the run would continue on other bits than "
                f"it would have. This process chose ({chose}), with PYTORCH_MPS_FAST_MATH "
                f"{'unset' if fast is None else f'set to {fast!r}'} (set, it rules the kernel "
                "out). The run can continue where it stepped with the same "
                "implementation, or start again from its adapter (--from <checkpoint>)")
    logger.warning("This resume's optimizer state was stepped by ExpertAdamW's %s, and it now "
                   "steps with its %s: both store the same bits; recorded in the next "
                   "checkpoint's %s", was, _said(now), RULE_FILE)
    for i in now.values():
        i["resumed_from"] = was
    return None


def write_rule(directory: Path, optimizer: Any, torch_dtype: Any = None) -> None:
    """Record, beside every checkpoint's optimizer state, the rule that wrote it, the dtype of the
    tensors it updates and the run's torch_dtype: what a resume checks before any model loads
    (``poros.core.checkpoints.checkpoint_record``)."""
    said = {"rule": rule_of(optimizer), "trainable_dtype": trainable_dtype(optimizer),
            "torch_dtype": torch_dtype,
            # which implementation ran, for the record: kernel or reference, and where. A resume
            # on a Mac refuses another (``other_implementation``); on CUDA it is recorded
            "implementation": getattr(optimizer, "implementation", None)}
    (Path(directory) / RULE_FILE).write_text(json.dumps(said) + "\n")
