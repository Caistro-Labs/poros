"""The one owned piece of Metal: the NF4 decode kernel.

One fp32 product per element and one cast to bf16, in ``poros.moe.quant.dequantize``'s
op order. Importing this module pulls no torch.
"""

from __future__ import annotations

import hashlib
import os
from functools import cache

NF4_DECODE_SOURCE = """\
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

// One thread per 64-element NF4 block. Pure per-thread function:
// no atomics, no threadgroup memory, no SIMD-group reductions.
kernel void nf4_decode_bf16(
    device bfloat*       out     [[buffer(0)]],   // scratch, storage offset 0
    device const uchar*  packed  [[buffer(1)]],   // blob[0:packed_b], high nibble first
    device const uchar*  q8      [[buffer(2)]],   // blob[pa:pa+nblocks], absmax codes
    device const float*  absmax2 [[buffer(3)]],   // blob[a2:a2+4*nabsmax2]
    device const float*  offset  [[buffer(4)]],   // blob[off:off+4]
    constant float*      code8   [[buffer(5)]],   // quant.dynamic_codebook(), 256
    constant float*      code4   [[buffer(6)]],   // quant.nf4_codebook(), 16
    constant long&       n       [[buffer(7)]],   // Python ints bind as int64
    uint blk [[thread_position_in_grid]])
{
    float a = code8[q8[blk]] * absmax2[blk >> 8];   // two statements on purpose:
    a = a + offset[0];                              // quant.py rounds twice, an FMA rounds once
    const long start = long(blk) * 64;
    const long end = min(start + 64, n);
    for (long i = start; i < end; ++i) {
        const uchar b = packed[i >> 1];
        const uint nib = (i & 1) ? (b & 0xF) : (b >> 4);
        out[i] = static_cast<bfloat>(code4[nib] * a);   // one fp32 product, one cast
    }
}
"""

NF4_DECODE_SHA256 = hashlib.sha256(NF4_DECODE_SOURCE.encode()).hexdigest()


@cache
def library():
    """The compiled library, once per process. Compiled Safe with precise functions."""
    import torch

    os.environ.pop("PYTORCH_MPS_FAST_MATH", None)
    return torch.mps.compile_shader(NF4_DECODE_SOURCE)
