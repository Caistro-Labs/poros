"""NF4 quantization for fused expert tensors (``poros.moe`` format layer).

Bit-compatibility target: the repo's NF4 path loads with
``BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bf16)``
(``poros.train.model._load_model``), quantizes via
``bnb.nn.Params4bit(...).to(cuda)`` (``poros.train.streamed_load``),
and dequantizes with ``bnb.functional.dequantize_4bit``
(``poros.train.streamed_load``). bitsandbytes runs those as
compiled kernels; this module reimplements the same documented format in
pure torch, op-for-op after the kernels that actually execute for those
calls (bitsandbytes 0.50.0): ``quantize_4bit`` and the dequant compute
paths after ``backends/default/ops.py`` (``quantize_4bit``,
``_dequantize_4bit_compute``, ``_dequantize_blockwise_compute`` — the
implementations bnb dispatches to for 4-bit on CPU), and the double-quant
8-bit ``quantize_blockwise`` after the compiled kernel bnb dispatches to
for it (``csrc/cpu_ops.cpp`` ``quantize_cpu_impl``: raw absmax, reciprocal
scaling, 65536-bin nearest-code LUT). Verified bit-for-bit against
bitsandbytes 0.50.0 running on CPU: packed bytes, dequantized values, and
zero-block behavior all identical. bnb's
CUDA quantize kernels (``csrc/kernels.cu``) share the same codebooks,
blocksizes, and midpoint thresholds — the fp32 midpoints computed here are
bit-identical to the threshold literals in ``dQuantizeNF4`` — but their
8-bit code selection is a binary search rather than the CPU LUT, so
cross-device byte identity of the double-quant layer is not well-defined
even between bnb's own backends; the dequant math (``code[q] * absmax`` in
fp32, one cast at the end) is identical on every backend. A slice fetched
twice dequantizes bit-identically both times (the residency invariant),
and all-zero slices stay exactly zero.

Format guarantees:

- NF4 codebook: the 16 QLoRA normal-quantile values, fp32.
- Blocksize 64; one fp32 absmax per block; values scaled by the block
  reciprocal and clamped to [-1, 1]; codes are the nearest codebook entry
  with midpoint ties to the lower index (``torch.bucketize`` on fp32
  midpoints); even elements pack into the high nibble.
- Double quant (the repo's ``bnb_4bit_use_double_quant=True``): block
  absmax values are centered by their fp32 mean (the offset) and quantized
  to 8-bit dynamic-map codes with blocksize 256; dequantization
  reconstructs ``code8[q] * absmax2 + offset`` in fp32 before the NF4
  lookup, then computes ``code4[q] * absmax`` in fp32, casting to the
  target dtype once, at the end.
- Zero preservation: an all-zero block normalizes to exact zeros and
  quantizes to code 7 (NF4's exact 0.0), so it dequantizes to exact zeros
  even when its double-quantized absmax reconstructs with error. Sparse
  slabs are never densified.

Blob layout — one quantized tensor is one flat uint8 blob::

    [0, P)              packed 4-bit codes, P = ceil(n / 2)
    [Pa, Pa + B)        8-bit dynamic-map codes of the absmax, B = ceil(n / 64)
    [A2, A2 + 4M)       second-level absmax, M = ceil(B / 256) fp32 values
    [Off, Off + 4)      double-quant offset, one fp32 value

with ``Pa = align4(P)`` and ``A2 = Pa + align4(B)``: every fp32 section
starts 4-byte aligned so it can be viewed in place out of a containing
slab, and every blob size is a multiple of 4 so concatenated blobs stay
aligned. fp32 sections use native (little-endian) byte order. A blob is
self-describing given the tensor shape; the per-expert concatenation of
blobs is owned by ``poros.moe.store.MoeExpertStore``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import torch

__all__ = [
    "DOUBLE_QUANT_BLOCKSIZE",
    "NF4_BLOCKSIZE",
    "dequantize",
    "dynamic_codebook",
    "nf4_codebook",
    "quantize_experts",
    "quantize_tensor",
    "quantized_nbytes",
]

NF4_BLOCKSIZE = 64
DOUBLE_QUANT_BLOCKSIZE = 256

_QUANTIZABLE = (torch.bfloat16, torch.float16, torch.float32)


@cache
def nf4_codebook() -> torch.Tensor:
    """The 16 NF4 code values, fp32 (bnb ``get_4bit_type("nf4")``).

    bnb divides the table by its abs max, which is exactly 1.0 for NF4, so
    the hardcoded values are the codebook.
    """
    return torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=torch.float32,
    )


@cache
def dynamic_codebook() -> torch.Tensor:
    """bnb's signed 8-bit dynamic map, fp32, sorted ascending.

    Port of ``create_dynamic_map(signed=True, max_exponent_bits=7,
    total_bits=8)`` (bitsandbytes ``functional.py``) — the codebook the
    double-quant layer uses for absmax values. Same torch ops in the same
    order, so the fp32 values are bit-identical.
    """
    data: list[float] = []
    for i in range(7):
        boundaries = torch.linspace(0.1, 1, 2**i + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        scale = 10 ** (i - 6)
        data += (scale * means).tolist()
        data += (-(scale * means)).tolist()
    data.append(0)
    data.append(1.0)
    if len(data) != 256:
        raise RuntimeError(f"dynamic map must have 256 entries, got {len(data)}")
    data.sort()
    return torch.tensor(data, dtype=torch.float32)


@cache
def _nf4_midpoints() -> torch.Tensor:
    """fp32 midpoints between adjacent NF4 codes (bnb's quantize bounds)."""
    code = nf4_codebook()
    return (code[:-1] + code[1:]) / 2


@cache
def _nf4_midpoints_on(device: torch.device) -> torch.Tensor:
    """``_nf4_midpoints`` copied to ``device`` once (the same fp32 values)."""
    return _nf4_midpoints().to(device)


@cache
def _device_codebooks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic codebook and NF4 byte table on ``device``, cached per device.

    A per-dequant ``.to(device)`` would be a blocking host-to-device copy.

    Row ``b`` of the ``[256, 2]`` byte table is ``(code[b >> 4], code[b & 0xF])``:
    a packed byte's two values, high nibble first, so one lookup decodes it.
    """
    code = nf4_codebook()
    nf4_bytes = torch.stack([code.repeat_interleave(16), code.repeat(16)], dim=1)
    return dynamic_codebook().to(device), nf4_bytes.to(device)


def _align4(n: int) -> int:
    return (n + 3) // 4 * 4


def _section_sizes(n: int) -> tuple[int, int, int]:
    """(packed-code bytes, absmax-code bytes, absmax2 count) for n elements."""
    packed = (n + 1) // 2
    nblocks = -(-n // NF4_BLOCKSIZE)
    nabsmax2 = -(-nblocks // DOUBLE_QUANT_BLOCKSIZE)
    return packed, nblocks, nabsmax2


def quantized_nbytes(shape: Sequence[int]) -> int:
    """Byte size of the flat uint8 blob for a tensor of ``shape``."""
    n = 1
    for dim in shape:
        n *= int(dim)
    if n <= 0:
        raise ValueError(f"shape must have positive element count, got {tuple(shape)}")
    packed, nblocks, nabsmax2 = _section_sizes(n)
    return _align4(packed) + _align4(nblocks) + 4 * nabsmax2 + 4


def _blockwise_scaled(w_flat: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized values in [-1, 1] and per-block fp32 absmax.

    Op-for-op with bnb's default-backend ``quantize_4bit`` (the NF4 level):
    full blocks scale by the reciprocal of the clamped absmax, a tail
    block scales by division; the stored absmax of an all-zero full block
    is exact 0.0.
    """
    n = w_flat.numel()
    rem = n % blocksize
    full = n - rem
    blocks = w_flat[:full].reshape(-1, blocksize)
    absmax = blocks.abs().max(dim=-1)[0]
    scaled = torch.clamp(blocks * (1.0 / absmax.clamp(min=1e-38)).view(-1, 1), -1, 1).reshape(-1)
    if rem:
        am = w_flat[full:].abs().max().clamp(min=1e-38)
        absmax = torch.cat([absmax, am.unsqueeze(0)])
        scaled = torch.cat([scaled, torch.clamp(w_flat[full:] / am, -1, 1)])
    return scaled, absmax


def _quantize_nf4(w_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed NF4 codes + fp32 per-block absmax (bnb default ``quantize_4bit``)."""
    scaled, absmax = _blockwise_scaled(w_flat, NF4_BLOCKSIZE)
    if scaled.numel() % 2:
        scaled = torch.nn.functional.pad(scaled, (0, 1))
    mid = _nf4_midpoints() if scaled.device.type == "cpu" else _nf4_midpoints_on(scaled.device)
    q = torch.bucketize(scaled, mid, out_int32=True).to(torch.uint8)
    return (q[::2] << 4) | q[1::2], absmax


@cache
def _dynamic_lut() -> torch.Tensor:
    """The 65536-bin nearest-code LUT of bnb's compiled quantize kernel.

    Port of ``build_quantize_lut`` (bitsandbytes 0.50.0
    ``csrc/cpu_ops.cpp``): bin ``i`` represents the normalized value
    ``-1.0 + (2.0 * i) / 65535`` in fp32 and maps to the codebook index
    whose midpoint interval contains it (``midpoints[j] = 0.5 * (c[j] +
    c[j+1])`` in fp32, ties to the lower index — ``torch.bucketize`` with
    ``right=False`` counts the midpoints strictly below the bin value,
    exactly like the kernel's advancing while-loop).
    """
    code = dynamic_codebook()
    midpoints = (code[:-1] + code[1:]) * 0.5
    bins = torch.arange(65536, dtype=torch.float32)
    vals = -1.0 + (2.0 * bins) / 65535.0
    return torch.bucketize(vals, midpoints, out_int32=True).to(torch.uint8)


def _lut_codes(blocks: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
    """8-bit codes for rows of ``blocks`` via the compiled kernel's LUT path.

    Port of ``quantize_cpu_impl`` + ``norm_to_lut_index`` (bitsandbytes
    0.50.0 ``csrc/cpu_ops.cpp``): scale by the fp32 reciprocal, clamp to
    [-1, 1], snap to the 16-bit bin
    ``trunc(((v + 1.0) * 0.5) * 65535.0 + 0.5)`` (all fp32 ops in the
    kernel's order), look the bin up in the LUT. An all-zero block emits
    code 0 for every element (the kernel's explicit guard), which keeps a
    zero absmax block self-consistent: its raw stored absmax is 0.0.
    """
    lut = _dynamic_lut()
    nonzero = absmax > 0
    inv = torch.where(nonzero, 1.0 / absmax, torch.ones_like(absmax))
    scaled = torch.clamp(blocks * inv.unsqueeze(-1), -1.0, 1.0)
    idx = (((scaled + 1.0) * 0.5) * 65535.0 + 0.5).to(torch.int32).to(torch.int64)
    codes = lut[idx]
    return torch.where(nonzero.unsqueeze(-1), codes, torch.zeros_like(codes))


def _quantize_dynamic8(a_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """8-bit dynamic-map codes + raw fp32 absmax (bnb's compiled quantize kernel).

    This is the double-quant layer; bnb's ``quantize_4bit`` runs it through
    ``quantize_blockwise``, which dispatches to the compiled
    ``quantize_cpu_impl`` kernel on CPU. Raw per-block absmax (no clamp),
    blocksize 256 over the flat array, tail block handled like a full one.
    """
    n = a_flat.numel()
    rem = n % DOUBLE_QUANT_BLOCKSIZE
    full = n - rem
    blocks = a_flat[:full].reshape(-1, DOUBLE_QUANT_BLOCKSIZE)
    absmax = blocks.abs().max(dim=-1)[0]
    q = _lut_codes(blocks, absmax).reshape(-1)
    if rem:
        tail = a_flat[full:].reshape(1, rem)
        am = tail.abs().max(dim=-1)[0]
        absmax = torch.cat([absmax, am])
        q = torch.cat([q, _lut_codes(tail, am).reshape(-1)])
    return q, absmax


def _scale_blocks_(vals: torch.Tensor, absmax: torch.Tensor, blocksize: int) -> torch.Tensor:
    """``vals *= absmax`` per block of the last dim, in fp32 (bnb ``_dequantize_*_compute``)."""
    full = vals.shape[-1] - vals.shape[-1] % blocksize
    vals[..., :full].unflatten(-1, (-1, blocksize)).mul_(absmax[..., : full // blocksize, None])
    vals[..., full:].mul_(absmax[..., -1:])
    return vals


def quantize_tensor(weight: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    """Quantize one tensor to its flat uint8 blob (see module docstring).

    Deterministic: same input bytes in, same blob bytes out (G-QUANT). ``device`` (e.g.
    ``"cuda"``) runs the NF4 level there -- the fp32 upcast, each block's absmax and reciprocal
    scale, the clamp, the code lookup and the packing: exact or exactly rounded IEEE fp32 ops
    and comparisons, so the same bytes as on the CPU (checked tensor by tensor on whole models,
    ``tests/gpu/test_quant_device.py``). The double-quant layer starts with a mean, a reduction
    whose summation order sets its rounding, so it always runs on the CPU. The blob is on the
    CPU either way; one tensor's working set is on the device at a time.
    """
    if weight.dtype not in _QUANTIZABLE:
        raise ValueError(f"NF4 quantization supports bf16/fp16/fp32, got {weight.dtype}")
    if weight.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    w_flat = weight.detach().reshape(-1)
    if device is not None and torch.device(device).type != "cpu":
        packed, absmax = _quantize_nf4(w_flat.to(device).float())
        packed, absmax = packed.cpu(), absmax.cpu()
    else:
        packed, absmax = _quantize_nf4(w_flat.float())
    offset = absmax.mean()
    qabsmax, absmax2 = _quantize_dynamic8(absmax - offset)
    packed_b, nblocks, nabsmax2 = _section_sizes(weight.numel())
    pa = _align4(packed_b)
    a2 = pa + _align4(nblocks)
    off = a2 + 4 * nabsmax2
    blob = torch.zeros(off + 4, dtype=torch.uint8)
    blob[:packed_b] = packed
    blob[pa : pa + nblocks] = qabsmax
    blob[a2 : a2 + 4 * nabsmax2] = absmax2.contiguous().view(torch.uint8)
    blob[off : off + 4] = offset.reshape(1).contiguous().view(torch.uint8)
    return blob


def quantize_experts(param: torch.Tensor) -> list[torch.Tensor]:
    """Quantize a fused ``[num_experts, ...]`` parameter.

    Returns one flat uint8 blob per expert — the per-expert "bytes" the
    store concatenates into slabs. Deterministic per expert; the fused
    layout is only iterated, never mixed across experts.
    """
    if param.dim() < 2:
        raise ValueError(
            f"expected a fused [num_experts, ...] parameter, got shape {tuple(param.shape)}"
        )
    return [quantize_tensor(param[e]) for e in range(param.shape[0])]


def dequantize(blob: torch.Tensor, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a flat uint8 blob back to ``shape``/``dtype``.

    Bit-identical to the repo's NF4 dequantization of the same logical
    weights (see module docstring for the matched semantics). Leading dims
    batch blobs of one shape: ``[..., nbytes]`` dequantizes row-wise to
    ``[..., *shape]``, elementwise identical to one call per row.
    """
    if blob.dtype != torch.uint8 or blob.dim() == 0:
        raise ValueError(f"blob must be uint8 [..., nbytes], got {blob.dtype} dim {blob.dim()}")
    if dtype not in _QUANTIZABLE:
        raise ValueError(f"NF4 dequantization supports bf16/fp16/fp32, got {dtype}")
    n = 1
    for dim in shape:
        n *= int(dim)
    if n <= 0:
        raise ValueError(f"shape must have positive element count, got {tuple(shape)}")
    packed_b, nblocks, nabsmax2 = _section_sizes(n)
    pa = _align4(packed_b)
    a2 = pa + _align4(nblocks)
    off = a2 + 4 * nabsmax2
    if blob.shape[-1] != off + 4:
        raise ValueError(
            f"blob of {blob.shape[-1]} bytes cannot hold shape {tuple(shape)} "
            f"({n} elements need {off + 4} bytes)"
        )
    code8, nf4_bytes = _device_codebooks(blob.device)
    qabsmax = blob[..., pa : pa + nblocks].int()
    absmax2 = blob[..., a2 : a2 + 4 * nabsmax2].view(torch.float32)
    offset = blob[..., off : off + 4].view(torch.float32)
    absmax = _scale_blocks_(code8[qabsmax], absmax2, DOUBLE_QUANT_BLOCKSIZE) + offset
    vals = nf4_bytes[blob[..., :packed_b].int()].flatten(-2)[..., :n]
    return _scale_blocks_(vals, absmax, NF4_BLOCKSIZE).reshape((*blob.shape[:-1], *shape)).to(dtype)
