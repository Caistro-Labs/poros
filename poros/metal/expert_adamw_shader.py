"""ExpertAdamW's bf16 update as one Metal kernel on a Mac: each element read once and written once,
with no temporaries.

It computes the rule written out in poros/train/expert_adamw.py, operation by operation: plain
fp32 operations with fp contract off (no fused multiply-add of the rule's), the correctly rounded
square root and division built from fp32 operations and fma alone (``sqrt_rn``, ``div_rn``: Metal's
own are not correctly rounded), F (a value at most 2**-126 in magnitude read and stored as +0)
after every load and every operation, the weight's rounding to bf16 to nearest even in integers,
and the three keyed draws (64-bit integers that never exceed 2**58) and the stochastic rounding
in integers. The optimizer checks on the Mac that it
stores the reference's bits before it uses it (``_probe_kernel``); where it does not, or fast
math is set, the reference runs instead. Importing this module pulls no torch, and it never
changes the process's environment.
"""

from __future__ import annotations

import hashlib
from functools import cache

EXPERT_ADAMW_SOURCE = """\
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

// One draw's 16 keyed bits: (a * x + b) mod 2**36, its top 16 bits; a, b < 2**36, x < 2**21.
static inline int keyed(device const long* coef, ulong j, ulong x, uint draw) {
    const ulong ab = ulong(coef[6 * j + 2 * draw]) * x + ulong(coef[6 * j + 2 * draw + 1]);
    return int((ab & 0xFFFFFFFFFul) >> 20);
}

// Stochastic rounding to bf16 in int32 on the magnitude (the int16 bits of the bf16).
static inline short round_bf16(float x, int bits) {
    const int ix = as_type<int>(x);
    int out = (min(ix & 0x7FFFFFFF, 0x7F800000) + bits) >> 16;
    if (ix < 0) out -= 0x8000;
    if (isnan(x)) out = 0x7FC0;
    return short(out);
}

// The rule's F: +0 where |x| <= 2**-126 (a NaN stays a NaN).
static inline float F(float x) { return fabs(x) <= 0x1p-126f ? 0.0f : x; }  // 2**-126 exactly

// The rule's square root and division, correctly rounded from fp32 operations alone: Metal's
// precise::sqrt and precise::divide are within a few units in the last place, not correctly
// rounded, so each is a first guess only. One Newton step with fma brings it within one unit;
// exact fma residuals then pick the correctly rounded one of it and its two neighbours. Every
// +, -, * and fma here is one IEEE operation (fp contract off), and no operand or residual is
// a subnormal, so a GPU that flushes subnormals computes the same. Transcribed operation by
// operation and checked against the correctly rounded results on the CPU over 10**8 values with
// every guess moved up to 4 units (scripts/moe_precision/metal_exact_arith.py; it holds up to
// 1024); a Mac's probe checks it on the GPU before the optimizer uses it.
static inline float nan_() { return as_type<float>(0x7FC00000u); }

// x through F (+0, a positive normal, inf or NaN).
static inline float sqrt_rn(float x) {
    if (isnan(x) || x < 0.0f) return nan_();
    if (x == 0.0f || isinf(x)) return x;
    float scale = 1.0f;  // x into [2**-64, 2**64) by an even power of two: exact either way
    if (x < 0x1p-64f) { x *= 0x1p64f; scale = 0x1p-32f; }
    else if (x >= 0x1p64f) { x *= 0x1p-64f; scale = 0x1p32f; }
    float r = precise::sqrt(x);
    r = fma(fma(-r, r, x), precise::divide(0.5f, r), r);
    const float up = as_type<float>(as_type<uint>(r) + 1u);
    const float dn = as_type<float>(as_type<uint>(r) - 1u);
    // the midpoint (r + up) / 2 squared is r * up + h**2, h = (up - r) / 2; a nonzero r * up - x
    // is a multiple of 4 h**2, so its sign alone (h**2 > 0 breaks a zero) says which side
    if (fma(r, up, -x) < 0.0f) r = up;
    else if (fma(dn, r, -x) >= 0.0f) r = dn;
    return r * scale;
}

// F(a / b correctly rounded), for a and b through F.
static inline float div_rn(float a, float b) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    const uint sign = (ua ^ ub) & 0x80000000u;
    if (isnan(a) || isnan(b) || (isinf(a) && isinf(b)) || (a == 0.0f && b == 0.0f)) return nan_();
    if (isinf(a) || b == 0.0f) return as_type<float>(sign | 0x7F800000u);
    if (a == 0.0f || isinf(b)) return 0.0f;
    int e = int((ua >> 23) & 0xFFu) - int((ub >> 23) & 0xFFu);  // a / b = (ma / mb) * 2**e
    const float ma = as_type<float>((ua & 0x007FFFFFu) | 0x3F800000u);
    const float mb = as_type<float>((ub & 0x007FFFFFu) | 0x3F800000u);
    float q = precise::divide(ma, mb);
    q = fma(fma(-q, mb, ma), precise::divide(1.0f, mb), q);
    const float r = fma(q, mb, -ma);  // exact: q is within a unit of ma / mb
    const float up = as_type<float>(as_type<uint>(q) + 1u);
    const float dn = as_type<float>(as_type<uint>(q) - 1u);
    // past a midpoint q +- h where the residual passes -+h * mb, each exact; a quotient of two
    // 24-bit significands is never a midpoint
    if (r < -((0.5f * (up - q)) * mb)) q = up;
    else if (r > (0.5f * (q - dn)) * mb) q = dn;
    if (q < 1.0f) { q *= 2.0f; e -= 1; }
    // q * 2**e, q in [1, 2): past the largest is inf; at most 2**-126 is F's zero
    if (e > 127) return as_type<float>(sign | 0x7F800000u);
    if (e < -126 || (e == -126 && q == 1.0f)) return 0.0f;
    return as_type<float>(sign | (uint(e + 127) << 23) | (as_type<uint>(q) & 0x007FFFFFu));
}

// fp32 to bf16 to nearest even, in integers (the int16 bits): no conversion instruction's
// rounding to rely on. A NaN becomes 0x7fc0; a value past bf16's largest, inf.
static inline short rtne_bf16(float x) {
    const int ix = as_type<int>(x);
    const int mag = min(ix & 0x7FFFFFFF, 0x7F800000);
    int out = (mag + 0x7FFF + ((mag >> 16) & 1)) >> 16;
    if (ix < 0) out -= 0x8000;
    return isnan(x) ? short(0x7FC0) : short(out);
}

// One thread per element. Pure per-thread function: no atomics, no threadgroup memory.
// s: [1 - beta1, beta2, 1 - beta2, 1 - lr * weight_decay, sqrt(1 - beta2**t), eps,
//     -lr / (1 - beta1**t)], each already an fp32 value
kernel void expert_adamw_bf16(
    device bfloat*       hi    [[buffer(0)]],
    device bfloat*       lo    [[buffer(1)]],
    device const bfloat* g     [[buffer(2)]],
    device bfloat*       m     [[buffer(3)]],
    device bfloat*       v     [[buffer(4)]],
    device const long*   coef  [[buffer(5)]],   // per 2**21 elements: a_v b_v a_m b_m a_r b_r
    device const float*  s     [[buffer(6)]],
    constant long&       n     [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    const long i = long(tid);
    if (i >= n) return;
    const ulong j = ulong(i) >> 21;
    const ulong x = ulong(i) & 0x1FFFFFul;
    const float g32 = F(float(g[i]));
    const float m32 = F(float(m[i]));
    const float v32 = F(float(v[i]));
    const float h32 = F(float(hi[i]));
    const float l32 = F(float(lo[i]));
    const float m1 = F(m32 + F(s[0] * F(g32 - m32)));
    const float v1 = F(F(v32 * s[1]) + F(F(s[2] * g32) * g32));
    float w = F(F(h32 + l32) * s[3]);
    const float d = F(F(div_rn(F(sqrt_rn(v1)), s[4])) + s[5]);
    w = F(w + F(div_rn(F(s[6] * m1), d)));
    const short hb = rtne_bf16(w);
    hi[i] = as_type<bfloat>(hb);
    const float hf = as_type<float>(uint(ushort(hb)) << 16);  // exact
    lo[i] = as_type<bfloat>(isfinite(hf) ? round_bf16(F(w - hf), keyed(coef, j, x, 2))
                                         : short(0));
    m[i] = as_type<bfloat>(round_bf16(m1, keyed(coef, j, x, 1)));
    v[i] = as_type<bfloat>(round_bf16(v1, keyed(coef, j, x, 0)));
}
"""

EXPERT_ADAMW_SHA256 = hashlib.sha256(EXPERT_ADAMW_SOURCE.encode()).hexdigest()


@cache
def library():
    """The compiled library, once per process. The caller (``_kernel_unavailable``) refuses it
    where PYTORCH_MPS_FAST_MATH is set: this module never changes the environment."""
    import torch

    return torch.mps.compile_shader(EXPERT_ADAMW_SOURCE)
