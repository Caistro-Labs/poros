"""The Metal kernel's correctly rounded fp32 square root and division, and its integer bf16
rounding, transcribed operation by operation into numpy and checked against the correctly
rounded results on the CPU, before any Mac runs them.

The shader (poros/metal/expert_adamw_shader.py) computes ``sqrt_rn`` and ``div_rn`` from fp32
operations alone: +, -, * and fma, each one IEEE operation correctly rounded (Metal guarantees
these with fp contract off), and a first guess from ``precise::sqrt`` and ``precise::divide``,
which Metal does not promise to round correctly (within a few units in the last place). Here:

* every fp32 +, - and * is numpy's float32 operation (one IEEE operation each);
* every fma is exact: the product in fp64 (24 + 24 bits fit in 53), the sum rounded to odd in
  fp64 (TwoSum gives its exact error), then to fp32: rounding to odd at 53 bits and then to
  nearest at 24 is the correctly rounded fma (Boldo and Melquiond), checked here against
  Python's math.fma on a sample;
* every guess is the correctly rounded value moved by a random whole number of units in the last
  place in [-GUESS, GUESS] (GUESS = 4, above Metal's documented bounds of 3 for the root and 2.5
  for the division), so the result holds for any guess within that bound;
* the correct results are ``poros.train.expert_adamw._sqrt`` and ``_div`` (each the correctly
  rounded fp32 result, proven against exact rationals in tests/moe/test_expert_adamw_hardening.py),
  read through F, as the rule reads every result.

Inputs are the rule's: each already through F (+0, a normal, an infinity or a NaN). Each class
of boundary is drawn on purpose: random bit patterns over every exponent, roots and quotients
within a few units of a midpoint (the hard cases), exact squares and exact quotients, powers of
two and their neighbours, the scaling edges 2**-64 and 2**64, quotients near 2**-126 (F's edge)
and near 2**128 (the overflow), and zeros, infinities and NaNs.

    python scripts/moe_precision/metal_exact_arith.py --values 100000000 --workers 4

prints one line per check and a JSON summary; exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

F32, F64, U32, I64 = np.float32, np.float64, np.uint32, np.int64
GUESS = 4
FLT_MIN = F32(2.0 ** -126)


# ---- the operations Metal guarantees -----------------------------------------------------------


def fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """fp32 fma, correctly rounded: a * b exact in fp64, a * b + c rounded to odd in fp64, then
    to nearest in fp32."""
    p = a.astype(F64) * b.astype(F64)
    c64 = c.astype(F64)
    s = p + c64
    bp = s - p
    e = (p - (s - bp)) + (c64 - bp)  # TwoSum: s + e = p + c exactly
    even = (s.view(I64) & 1) == 0
    fix = (e != 0) & even & np.isfinite(s)
    toward = np.where(e > 0, np.inf, -np.inf)
    s = np.where(fix, np.nextafter(s, toward), s)
    with np.errstate(over="ignore"):
        return s.astype(F32)


def bits(x: np.ndarray) -> np.ndarray:
    return x.view(U32)


def from_bits(u: np.ndarray) -> np.ndarray:
    return u.astype(U32).view(F32)


def F(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) <= FLT_MIN, F32(0.0), x).astype(F32)


def _off_by(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x moved by a random whole number of units in the last place in [-GUESS, GUESS] (a finite
    nonzero x keeps its sign: no step crosses zero at these magnitudes)."""
    k = rng.integers(-GUESS, GUESS + 1, size=x.shape)
    u = bits(x).astype(I64)
    moved = from_bits((u + np.where(np.signbit(x), -k, k)).astype(U32))
    return np.where(np.isfinite(x) & (x != 0), moved, x).astype(F32)


def guess_sqrt(x, rng):
    """precise::sqrt within GUESS units (x normal and positive, scaled)."""
    return _off_by(np.sqrt(x.astype(F64)).astype(F32), rng)


def guess_div(a, b, rng):
    """precise::divide within GUESS units (a, b finite, b nonzero, the quotient normal)."""
    return _off_by((a.astype(F64) / b.astype(F64)).astype(F32), rng)


# ---- the shader's functions, operation by operation ----------------------------------------


def sqrt_rn(x: np.ndarray, rng) -> np.ndarray:
    """The shader's sqrt_rn, for x through F."""
    x = x.astype(F32)
    nan = np.isnan(x) | (x < 0)
    passthrough = (x == 0) | np.isinf(x)
    small, big = x < F32(2.0 ** -64), x >= F32(2.0 ** 64)
    xs = np.where(small, x * F32(2.0 ** 64), np.where(big, x * F32(2.0 ** -64), x)).astype(F32)
    scale = np.where(small, F32(2.0 ** -32), np.where(big, F32(2.0 ** 32), F32(1.0))).astype(F32)
    work = ~(nan | passthrough)
    xs = np.where(work, xs, F32(1.0)).astype(F32)  # the other lanes return before this
    r = guess_sqrt(xs, rng)
    half_over_r = guess_div(np.full_like(r, F32(0.5)), r, rng)
    r = fma32(fma32(-r, r, xs), half_over_r, r)
    up = from_bits(bits(r) + U32(1))
    dn = from_bits(bits(r) - U32(1))
    r = np.where(fma32(r, up, -xs) < 0, up, np.where(fma32(dn, r, -xs) >= 0, dn, r)).astype(F32)
    out = (r * scale).astype(F32)
    out = np.where(passthrough, x, out)
    return np.where(nan, F32(np.nan), out).astype(F32)


def div_rn(a: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    """The shader's div_rn, for a and b through F: F(the correctly rounded a / b)."""
    a, b = a.astype(F32), b.astype(F32)
    ua, ub = bits(a), bits(b)
    sign = (ua ^ ub) & U32(0x80000000)
    nan = np.isnan(a) | np.isnan(b) | (np.isinf(a) & np.isinf(b)) | ((a == 0) & (b == 0))
    inf = ~nan & (np.isinf(a) | (b == 0))
    zero = ~nan & ~inf & ((a == 0) | np.isinf(b))
    work = ~(nan | inf | zero)
    ua = np.where(work, ua, U32(0x3F800000))
    ub = np.where(work, ub, U32(0x3F800000))
    e = ((ua >> U32(23)) & U32(0xFF)).astype(I64) - ((ub >> U32(23)) & U32(0xFF)).astype(I64)
    ma = from_bits((ua & U32(0x007FFFFF)) | U32(0x3F800000))
    mb = from_bits((ub & U32(0x007FFFFF)) | U32(0x3F800000))
    q = guess_div(ma, mb, rng)
    inv = guess_div(np.ones_like(mb), mb, rng)
    q = fma32(fma32(-q, mb, ma), inv, q)
    r = fma32(q, mb, -ma)
    up = from_bits(bits(q) + U32(1))
    dn = from_bits(bits(q) - U32(1))
    go_up = r < -((F32(0.5) * (up - q)).astype(F32) * mb).astype(F32)
    go_dn = r > ((F32(0.5) * (q - dn)).astype(F32) * mb).astype(F32)
    q = np.where(go_up, up, np.where(go_dn, dn, q)).astype(F32)
    low = q < F32(1.0)
    q = np.where(low, q * F32(2.0), q).astype(F32)
    e = e - low
    packed = from_bits(sign | ((np.clip(e, -126, 127) + 127).astype(U32) << U32(23))
                       | (bits(q) & U32(0x007FFFFF)))
    out = np.where(e > 127, from_bits(sign | U32(0x7F800000)), packed)
    out = np.where((e < -126) | ((e == -126) & (q == F32(1.0))), F32(0.0), out)
    out = np.where(inf, from_bits(sign | U32(0x7F800000)), out)
    out = np.where(zero, F32(0.0), out)
    return np.where(nan, F32(np.nan), out).astype(F32)


def rtne_bf16(x: np.ndarray) -> np.ndarray:
    """The shader's rtne_bf16: fp32 to bf16 to nearest even in integers (int16 bits)."""
    ix = x.view(np.int32).astype(I64)
    mag = np.minimum(ix & 0x7FFFFFFF, 0x7F800000)
    out = (mag + 0x7FFF + ((mag >> 16) & 1)) >> 16
    out = np.where(ix < 0, out - 0x8000, out)
    out = np.where(np.isnan(x), 0x7FC0, out)
    return out.astype(np.int16)


# ---- the correct results --------------------------------------------------------------------


def correct_sqrt(x):
    import torch

    from poros.train import expert_adamw as ea

    t = torch.from_numpy(np.where(x >= 0, x, F32(np.nan)).astype(F32))
    return ea._flush(ea._sqrt(t)).numpy()


def correct_div(a, b):
    import torch

    from poros.train import expert_adamw as ea

    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    q = ea._div(ta, tb.abs()).numpy()  # _div takes b > 0: the sign moved onto the quotient
    q = np.where(np.signbit(b) & ~np.isnan(q), -q, q).astype(F32)
    q = np.where((b == 0) & (a != 0) & ~np.isnan(a),
                 from_bits((bits(a) ^ bits(b)) & U32(0x80000000) | U32(0x7F800000)), q)
    q = np.where((b == 0) & ((a == 0) | np.isnan(a)), F32(np.nan), q)
    q = np.where(np.isinf(b) & np.isfinite(a), F32(0.0), q)
    q = np.where(np.isinf(a) & np.isinf(b), F32(np.nan), q)
    return F(q.astype(F32))


# ---- the inputs, class by class ---------------------------------------------------------------


def _random_normals(rng, n, signed=False, lo=0x00800001, hi=0x7F7FFFFF):
    u = rng.integers(lo, hi + 1, size=n, dtype=I64).astype(U32)
    if signed:
        u |= (rng.integers(0, 2, size=n, dtype=I64).astype(U32) << U32(31))
    return from_bits(u)


def _significand(rng, n, bits_=24):
    """Random fp32 values in [1, 2) with ``bits_`` significant bits."""
    m = rng.integers(1 << (bits_ - 1), 1 << bits_, size=n, dtype=I64)
    return (m.astype(F64) / float(1 << (bits_ - 1))).astype(F32)


def _odd_inverse(b, bits_=24):
    """b**-1 modulo 2**bits_ for odd b (int64 arrays): Newton's iteration, 3 bits doubling."""
    mask = (1 << bits_) - 1
    x = b.copy()
    for _ in range(4):
        x = (x * ((2 - b * x) & mask)) & mask
    return x


def _odd_root(t, bits_):
    """An r with r * r = t modulo 2**bits_, for t = 1 modulo 8 (int64 arrays): bit by bit."""
    r = np.ones_like(t)
    for i in range(3, bits_):
        wrong = ((r * r - t) >> i) & 1  # r * r < 2**52: exact in int64
        r = r + wrong * (1 << (i - 1))
        r &= (1 << bits_) - 1
    return r


def hard_quotients(rng, k):
    """(a, b) whose quotient lies within t * 2**-48 (relatively, |t| < 2**10) of a midpoint
    between two fp32 values: N * B = A * 2**24 + t with N odd of 25 bits (the midpoint's
    significand), B odd of 24 bits, A of 24 bits, N = (t / B) modulo 2**24 + 2**24."""
    big = 3 * k + 64
    B = rng.integers(1 << 22, 1 << 23, size=big, dtype=I64) * 2 + 1
    t = rng.integers(-(1 << 9), 1 << 9, size=big, dtype=I64) * 2 + 1
    N = ((t & ((1 << 24) - 1)) * _odd_inverse(B) & ((1 << 24) - 1)) + (1 << 24)
    A = (N * B - t) >> 24
    keep = (A >= (1 << 23)) & (A < (1 << 24)) & (((N * B - t) & ((1 << 24) - 1)) == 0)
    A, B = A[keep][:k], B[keep][:k]
    e = rng.integers(-60, 61, size=A.size)
    sa = np.where(rng.integers(0, 2, size=A.size) == 1, -1.0, 1.0)
    a = (sa * A.astype(F64) * np.exp2(e - 23)).astype(F32)
    b = (B.astype(F64) * np.exp2(rng.integers(-60, 61, size=A.size) - 23)).astype(F32)
    return a, b


def hard_roots(rng, k):
    """x whose root lies within about t * 2**-48 (relatively, t < 2**12) of a midpoint between
    two fp32 values: N * N = X * 2**26 + t with N odd of 25 bits, X below 2**24, t = 1 mod 8."""
    big = 3 * k + 64
    t = rng.integers(0, 1 << 9, size=big, dtype=I64) * 8 + 1
    r = _odd_root(t, 26)
    cands = [r, (-r) & ((1 << 26) - 1), (r + (1 << 25)) & ((1 << 26) - 1),
             (-r + (1 << 25)) & ((1 << 26) - 1)]
    N = np.concatenate(cands)
    t4 = np.concatenate([t] * 4)
    keep = (N >= (1 << 24)) & (N < (1 << 25))
    N, t4 = N[keep], t4[keep]
    X = (N * N - t4) >> 26
    ok = ((N * N - t4) & ((1 << 26) - 1)) == 0
    X = X[ok][:k]
    e = rng.integers(-30, 31, size=X.size) * 2
    return (X.astype(F64) * np.exp2(e)).astype(F32)


def sqrt_inputs(rng, n):
    k = n // 5
    out = [_random_normals(rng, k)]
    # roots within a few units of a midpoint: x = (r + h)**2 rounded, then moved a few units
    e = rng.integers(-62, 63, size=k)
    r = (_significand(rng, k).astype(F64) * np.exp2(e)).astype(F32)
    up = from_bits(bits(r) + U32(1))
    mid = (r.astype(F64) + up.astype(F64)) / 2
    x = (mid * mid).astype(F32)
    out.append(F(_off_by(x, rng)))
    # exact squares
    e = rng.integers(-62, 63, size=k)
    r = (_significand(rng, k, 12).astype(F64) * np.exp2(e)).astype(F32)
    out.append(F((r.astype(F64) ** 2).astype(F32)))
    # powers of two and their neighbours, the scaling edges, F's edge, the largest
    p = np.exp2(rng.integers(-125, 128, size=k)).astype(F32)
    edges = np.array([2.0 ** -64, 2.0 ** 64, 2.0 ** -126, 2.0 ** -125, 3.4028234663852886e38,
                      1.0, 2.0, 0.0, np.inf, np.nan], dtype=F32)
    p[: edges.size] = edges
    p[edges.size: 4 * edges.size] = np.repeat(edges[:3], edges.size)
    p = F(_off_by(p, rng))
    # where a midpoint test's fma is exactly zero: r * r_up = x at r = 2**e, (1 + 2**-23) * 4**e,
    # and r_dn * r = x, (1 - 2**-24) * 4**e (exact fp32 values, kept as they are)
    four = np.exp2(2.0 * np.arange(-62, 64))
    exact = np.concatenate([(1 + 2.0 ** -23) * four, (1 - 2.0 ** -24) * four])
    p[-exact.size:] = exact.astype(F32)
    out.append(p)
    out.append(F(hard_roots(rng, k)))
    return np.concatenate(out)


def div_inputs(rng, n):
    k = n // 6
    a = [_random_normals(rng, k, signed=True)]
    b = [_random_normals(rng, k, signed=True)]
    # quotients within a few units of a midpoint: a = (q + h) * b rounded, moved a few units
    bb = (_significand(rng, k).astype(F64) * np.exp2(rng.integers(-60, 61, size=k))).astype(F32)
    q = (_significand(rng, k).astype(F64) * np.exp2(rng.integers(-60, 61, size=k))).astype(F32)
    mid = (q.astype(F64) + from_bits(bits(q) + U32(1)).astype(F64)) / 2
    a.append(_off_by((mid * bb.astype(F64)).astype(F32), rng))
    b.append(bb)
    # quotients near the overflow and near F's edge, and ratios of significands near 1
    for lo_e, hi_e in ((126, 129), (-128, -124)):
        ea_ = rng.integers(-126, 128, size=k)
        eb_ = np.clip(ea_ - rng.integers(lo_e, hi_e, size=k), -126, 127)
        ea_ = np.clip(eb_ + rng.integers(lo_e, hi_e, size=k), -126, 127)
        a.append((_significand(rng, k).astype(F64) * np.exp2(ea_)).astype(F32))
        b.append((_significand(rng, k).astype(F64) * np.exp2(eb_)).astype(F32))
    m = _significand(rng, k)
    a.append(_off_by(m, rng) * np.exp2(rng.integers(-60, 61, size=k)).astype(F32))
    b.append(m * np.exp2(rng.integers(-60, 61, size=k)).astype(F32))
    special = np.array([0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 3.4028234663852886e38,
                        2.0 ** -125, 1.5 * 2.0 ** -126], dtype=F32)
    sa, sb = np.meshgrid(special, special)
    a.append(sa.ravel())
    b.append(sb.ravel())
    ha, hb = hard_quotients(rng, k)
    a.append(ha)
    b.append(hb)
    return F(np.concatenate(a)), F(np.concatenate(b))


# ---- the checks -----------------------------------------------------------------------------


def _same(got, want):
    g, w = bits(got.astype(F32)), bits(want.astype(F32))
    return (g == w) | (np.isnan(got) & np.isnan(want))


def _first(bad, *cols):
    i = int(np.flatnonzero(~bad)[0])
    return [float(c[i]) for c in cols]


def work(task):
    global GUESS
    kind, seed, n, GUESS = task
    np.seterr(all="ignore")
    rng = np.random.default_rng(seed)
    if kind == "sqrt":
        x = sqrt_inputs(rng, n)
        got, want = sqrt_rn(x, rng), correct_sqrt(x)
        ok = _same(F(got), want)
        return kind, int(x.size), int((~ok).sum()), None if ok.all() else _first(ok, x, got, want)
    a, b = div_inputs(rng, n)
    got, want = div_rn(a, b, rng), correct_div(a, b)
    ok = _same(F(got), want)
    return kind, int(a.size), int((~ok).sum()), None if ok.all() else _first(ok, a, b, got, want)


def check_fma(rng, n=200_000) -> int:
    """fma32 against exact rationals on a sample: products of random fp32 values plus a third
    within a few units of minus their product (heavy cancellation, the hard cases). Returns the
    number of differences."""
    from fractions import Fraction

    a = _random_normals(rng, n, signed=True, lo=0x30000000, hi=0x50000000)
    b = _random_normals(rng, n, signed=True, lo=0x30000000, hi=0x50000000)
    c = -(a.astype(F64) * b.astype(F64)).astype(F32)  # cancellation: the hard cases
    c = _off_by(c, rng)
    got = fma32(a, b, c)
    bad = 0
    for i in range(0, n, 97):
        exact = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        want = F32(_round_f32(exact))
        bad += int(bits(np.array([got[i]]))[0] != bits(np.array([want]))[0])
    return bad


def _round_f32(x):
    """An exact rational to the nearest fp32, even on a tie (finite, normal or zero here)."""
    from fractions import Fraction

    if x == 0:
        return 0.0
    c = F32(float(x))
    for _ in range(3):
        up, dn = np.nextafter(c, F32(np.inf)), np.nextafter(c, F32(-np.inf))
        mu = (Fraction(float(c)) + Fraction(float(up))) / 2
        md = (Fraction(float(c)) + Fraction(float(dn))) / 2
        if x > mu or (x == mu and bits(np.array([up]))[0] % 2 == 0):
            c = up
        elif x < md or (x == md and bits(np.array([dn]))[0] % 2 == 0):
            c = dn
    return float(c)


def check_rtne(chunk=1 << 25) -> int:
    """rtne_bf16 against torch's fp32-to-bf16 cast over every fp32 bit pattern (a NaN to 0x7FC0,
    where torch keeps a NaN's payload)."""
    import torch

    bad = 0
    for start in range(0, 1 << 32, chunk):
        u = np.arange(start, start + chunk, dtype=np.uint64).astype(U32)
        x = u.view(F32)
        want = torch.from_numpy(x).bfloat16().view(torch.int16).numpy()
        want = np.where(np.isnan(x), np.int16(0x7FC0), want)
        bad += int((rtne_bf16(x) != want).sum())
    return bad


# ---- the shader's own source, compiled as C++ and run on the CPU ------------------------------

_SHIM = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
using uint = uint32_t; using ushort = uint16_t;
using std::fma; using std::isnan; using std::isinf; using std::fabs; using std::min;
template <class T, class U> static T as_type(U u) {
    static_assert(sizeof(T) == sizeof(U)); T t; std::memcpy(&t, &u, sizeof t); return t; }
static uint64_t state = 0x9E3779B97F4A7C15ull;
static int GUESS = 4;
static float off_by(float x) {  // a precise:: guess: the correct value moved up to GUESS units
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const int k = int((state >> 33) % uint64_t(2 * GUESS + 1)) - GUESS;
    if (!(std::isfinite(x)) || x == 0.0f) return x;
    return as_type<float>(uint(int(as_type<uint>(x)) + (x < 0 ? -k : k)));
}
namespace precise {
static float sqrt(float x) { return off_by(std::sqrt(x)); }
static float divide(float a, float b) { return off_by(a / b); }
}
"""

_MAIN = r"""
int main(int argc, char** argv) {
    GUESS = std::atoi(argv[1]);
    std::vector<float> a, b;
    float pair[2];
    while (std::fread(pair, sizeof(float), 2, stdin) == 2) {
        a.push_back(pair[0]);
        b.push_back(pair[1]);
    }
    for (size_t i = 0; i < a.size(); ++i) {
        float out[3] = {F(sqrt_rn(F(a[i] < 0 ? -a[i] : a[i]))), F(div_rn(F(a[i]), F(b[i]))), 0.0f};
        short h = rtne_bf16(a[i]);
        std::memcpy(&out[2], &h, sizeof h);
        std::fwrite(out, sizeof(float), 3, stdout);
    }
    return 0;
}
"""


def _minimal_env() -> dict:
    """A child's environment of named variables only, never a copy of this process's."""
    import os

    return {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}


def compiled_shader_helpers(workdir) -> str | None:
    """The shader's F, sqrt_rn, div_rn and rtne_bf16, cut from its source as they are, compiled
    as C++ (fp contract off, no fast math) around a shim for Metal's names: as_type is a bit
    copy, fma the C library's (correctly rounded), and precise::sqrt and precise::divide the
    correctly rounded result moved up to GUESS units. The path of the program, or None where no
    C++ compiler is found."""
    import shutil
    import subprocess
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from poros.metal.expert_adamw_shader import EXPERT_ADAMW_SOURCE as src

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return None
    helpers = src[src.index("static inline float F("):src.index("// One thread per element.")]
    path = Path(workdir) / "shader_helpers.cpp"
    path.write_text(_SHIM + helpers + _MAIN)
    exe = Path(workdir) / "shader_helpers"
    subprocess.run([cxx, "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math",
                    str(path), "-o", str(exe)], check=True, env=_minimal_env())
    return str(exe)


def check_compiled(exe: str, n: int, seed: int, guess: int) -> dict:
    """The compiled helpers on the same classes of inputs as the transcription: differences from
    the correct results, by function."""
    import subprocess

    rng = np.random.default_rng(seed)
    x = sqrt_inputs(rng, n)
    a, b = div_inputs(rng, n)
    m = min(x.size, a.size)
    pairs = np.stack([np.concatenate([x[:m], a[:m]]), np.concatenate([b[:m], b[:m]])], axis=1)
    out = subprocess.run([exe, str(guess)], input=pairs.astype(F32).tobytes(), env=_minimal_env(),
                         capture_output=True, check=True).stdout
    got = np.frombuffer(out, dtype=F32).reshape(-1, 3)
    first, second = pairs[:, 0], pairs[:, 1]
    sq_in = F(np.abs(first))
    want_bf16 = rtne_bf16(first)
    return {"values": int(first.size),
            "sqrt_differences": int((~_same(got[:, 0], correct_sqrt(sq_in))).sum()),
            "div_differences": int((~_same(got[:, 1], correct_div(F(first), F(second)))).sum()),
            "rtne_differences": int((np.ascontiguousarray(got[:, 2]).view(np.int32)
                                     .astype(np.int16) != want_bf16).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--values", type=int, default=100_000_000)
    ap.add_argument("--chunk", type=int, default=1 << 21)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--skip-rtne", action="store_true")
    ap.add_argument("--guess", type=int, default=GUESS,
                    help="units in the last place a precise:: guess may be off by")
    args = ap.parse_args()
    start = time.time()
    np.seterr(all="ignore")
    rng = np.random.default_rng(args.seed)
    summary = {"fma_differences": check_fma(rng)}
    print(f"fma32 against exact rationals: {summary['fma_differences']} differences", flush=True)
    half = args.values // 2
    tasks = [(kind, args.seed + 1000 * i + j, args.chunk, args.guess)
             for j, kind in enumerate(("sqrt", "div"))
             for i in range(-(-half // args.chunk))]
    totals = {"sqrt": [0, 0, None], "div": [0, 0, None]}
    with Pool(min(args.workers, 8)) as pool:
        for kind, n, bad, first in pool.imap_unordered(work, tasks):
            t = totals[kind]
            t[0] += n
            t[1] += bad
            t[2] = t[2] or first
    for kind, (n, bad, first) in totals.items():
        summary[kind] = {"values": n, "differences": bad, "first": first}
        print(f"{kind}_rn against the correctly rounded result through F: {n:,} values, "
              f"{bad} differences{'' if first is None else f'; the first {first}'}", flush=True)
    import tempfile

    with tempfile.TemporaryDirectory() as work_dir:
        exe = compiled_shader_helpers(work_dir)
        if exe is None:
            print("no C++ compiler: the shader's own source not compiled here", flush=True)
        else:
            said = check_compiled(exe, min(args.values, 20_000_000) // 2, args.seed + 7,
                                  args.guess)
            summary["compiled_source"] = said
            print(f"the shader's own sqrt_rn, div_rn and rtne_bf16, compiled as C++: {said}",
                  flush=True)
    if not args.skip_rtne:
        summary["rtne_differences"] = check_rtne()
        print(f"rtne_bf16 against torch's cast over all 2**32 patterns: "
              f"{summary['rtne_differences']} differences", flush=True)
    summary["guess_units"] = args.guess
    summary["seconds"] = round(time.time() - start, 1)
    print(json.dumps(summary))
    compiled = summary.get("compiled_source", {})
    failed = (summary["fma_differences"] or summary["sqrt"]["differences"]
              or summary["div"]["differences"] or summary.get("rtne_differences")
              or any(v for k, v in compiled.items() if k.endswith("differences")))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
