"""``quant.dequantize`` stays bit-identical to its original per-element decode.

A refactor pin, like ``test_expert_compute``: the reference is the original
decode (int64 nibble index, flat per-blob scaling). Batched blobs must
match one call per row.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.moe import quant  # noqa: E402


def _reference(blob, shape, dtype):
    n = torch.Size(shape).numel()
    packed_b, nblocks, nabsmax2 = quant._section_sizes(n)
    pa = quant._align4(packed_b)
    a2 = pa + quant._align4(nblocks)
    off = a2 + 4 * nabsmax2

    def scale(vals, absmax, bs):
        full = vals.numel() - vals.numel() % bs
        head = (vals[:full].view(full // bs, bs) * absmax[: full // bs, None]).reshape(full)
        return torch.cat([head, vals[full:] * absmax[-1]]) if full < vals.numel() else head

    code8 = quant.dynamic_codebook()
    absmax2 = blob[a2 : a2 + 4 * nabsmax2].view(torch.float32)
    qabsmax = blob[pa : pa + nblocks].to(torch.int64)
    absmax = scale(code8[qabsmax], absmax2, 256) + blob[off : off + 4].view(torch.float32)
    packed = blob[:packed_b]
    idx = torch.empty(packed.numel() * 2, dtype=torch.int64)
    idx[1::2] = packed & 0xF
    idx[::2] = packed >> 4
    return scale(quant.nf4_codebook()[idx][:n], absmax, 64).reshape(shape).to(dtype)


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


@pytest.mark.parametrize("shape", [(96, 256), (7, 13), (5,), (), (64 * 256 + 3,), (64 * 300,)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dequantize_matches_reference_bitwise(shape, dtype):
    torch.manual_seed(0)
    weights = torch.randn(3, *shape) * 3
    weights[0].view(-1)[:64] = 0  # include an all-zero block
    blobs = torch.stack([quant.quantize_tensor(w.to(dtype)) for w in weights])
    rows = torch.stack([_reference(b, shape, dtype) for b in blobs])
    assert torch.equal(_bits(quant.dequantize(blobs[1], shape, dtype)), _bits(rows[1]))
    assert torch.equal(_bits(quant.dequantize(blobs, shape, dtype)), _bits(rows))
