"""``poros.metal.nf4`` on the CPU route, and a Python port of the kernel. Metal where marked."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # first: numpy comes with torch; a torch-free install has none

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from poros.metal import nf4, shaders  # noqa: E402
from poros.metal.nf4 import Nf4Linear, Nf4Matmul  # noqa: E402
from poros.moe import quant  # noqa: E402
from tests.metal.nf4_tiny import arms  # noqa: E402

# The shader this file's port was written against. If this fails, the kernel changed: port it again.
PORTED_SHADER = "e4583316b09a4112c2e9dcbad22d3604e421f0f076fb52eb0f1203429639b24a"


def _sections(blob: torch.Tensor, n: int) -> tuple:
    """The kernel's buffers, cut from a blob by the layout in ``quant``'s docstring."""
    packed_b, nblocks, nabsmax2 = quant._section_sizes(n)
    pa = quant._align4(packed_b)
    a2 = pa + quant._align4(nblocks)
    off = a2 + 4 * nabsmax2
    return blob[:packed_b], blob[pa : pa + nblocks], blob[a2:off], blob[off : off + 4]


def kernel_port(out, packed, q8, absmax2, offset, code8, code4, n, *, threads, fma=False):
    """``nf4_decode_bf16`` statement by statement, one pass of the outer loop per thread. Only the
    cast waits for the end: it is the same per-element function either way."""
    packed = packed.numpy().tobytes()
    q8 = q8.numpy()
    absmax2 = absmax2.contiguous().view(torch.float32).numpy()
    offset = offset.contiguous().view(torch.float32).numpy()
    code8, code4 = code8.numpy(), code4.numpy()
    products = np.empty(n, dtype=np.float32)
    for blk in range(threads[0]):
        if fma:  # a contracted a * b + c, which rounds once. The kernel must not do this.
            a = np.float32(
                np.float64(code8[q8[blk]]) * np.float64(absmax2[blk >> 8]) + np.float64(offset[0])
            )
        else:
            a = code8[q8[blk]] * absmax2[blk >> 8]
            a = a + offset[0]
        start = blk * 64
        end = min(start + 64, n)
        for i in range(start, end):
            b = packed[i >> 1]
            nib = (b & 0xF) if i & 1 else (b >> 4)
            products[i] = code4[nib] * a
    out.copy_(torch.from_numpy(products).to(torch.bfloat16))


def _weights(shape, kind) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(shape, generator=gen, dtype=torch.float32)
    if kind == "sparse":
        return w * (torch.rand(shape, generator=gen) > 0.9)
    if kind == "small":
        return w * 1e-30
    if kind == "zero_blocks":
        w.reshape(-1)[:128] = 0
    return w


def test_port_was_written_against_this_shader():
    assert shaders.NF4_DECODE_SHA256 == PORTED_SHADER


@pytest.mark.parametrize("kind", ["randn", "sparse", "small", "zero_blocks"])
@pytest.mark.parametrize(
    "shape",
    [(7,), (64,), (65,), (1000, 3), (256 * 64,), (256 * 64 + 1,), (257 * 64 + 63,), (130, 257)],
    ids=str,
)
def test_kernel_port_is_byte_identical_to_dequantize(shape, kind):
    w = _weights(shape, kind)
    n = w.numel()
    blob = quant.quantize_tensor(w)
    out = torch.empty(n, dtype=torch.bfloat16)
    kernel_port(out, *_sections(blob, n), quant.dynamic_codebook(), quant.nf4_codebook(), n,
                threads=[quant._section_sizes(n)[1]])
    reference = quant.dequantize(blob, (n,), torch.bfloat16)
    assert torch.equal(out.view(torch.int16), reference.view(torch.int16))


def test_port_can_see_a_contracted_multiply_add():
    """Why the kernel has two statements: an FMA rounds once. On this block it changes the bf16."""
    n = 64
    blob = torch.zeros(quant.quantized_nbytes((n,)), dtype=torch.uint8)
    packed, q8, absmax2, offset = _sections(blob, n)
    packed.fill_(0xFF)  # code 15 everywhere: 1.0
    q8.fill_(4)
    absmax2.copy_(torch.tensor([float.fromhex("0x1.5bd214p-9")]).view(torch.uint8))
    offset.copy_(torch.tensor([float.fromhex("0x1.3deb62p-9")]).view(torch.uint8))
    code8, code4 = quant.dynamic_codebook(), quant.nf4_codebook()

    reference = quant.dequantize(blob, (n,), torch.bfloat16).view(torch.int16)
    two_rounds = torch.empty(n, dtype=torch.bfloat16)
    kernel_port(two_rounds, *_sections(blob, n), code8, code4, n, threads=[1])
    one_round = torch.empty(n, dtype=torch.bfloat16)
    kernel_port(one_round, *_sections(blob, n), code8, code4, n, threads=[1], fma=True)
    assert torch.equal(two_rounds.view(torch.int16), reference)
    assert not torch.equal(one_round.view(torch.int16), reference)


@pytest.mark.parametrize("start", [0, 64, 4096], ids=lambda s: f"start{s}")
@pytest.mark.parametrize("shape", [(7, 1), (64, 64), (130, 257)], ids=str)
def test_dispatch_cuts_the_right_sections_out_of_a_slab(monkeypatch, shape, start):
    """The port stands in for the compiled kernel, so the slicing runs on Linux."""
    fake = type("Library", (), {"nf4_decode_bf16": staticmethod(kernel_port)})
    monkeypatch.setattr(shaders, "library", lambda: fake)
    monkeypatch.setattr(nf4, "_codebooks", lambda: (quant.dynamic_codebook(), quant.nf4_codebook()))
    w = _weights(shape, "randn")
    blob = quant.quantize_tensor(w)
    slab = torch.full((start + blob.numel() + 128,), 0xA5, dtype=torch.uint8)
    slab[start : start + blob.numel()] = blob
    out = torch.empty(w.numel(), dtype=torch.bfloat16)
    nf4._kernel_decode(out, slab, start, w.numel())
    reference = quant.dequantize(blob, (w.numel(),), torch.bfloat16)
    assert torch.equal(out.view(torch.int16), reference.view(torch.int16))


def _linear(w: torch.Tensor, *, bias=None, start=0, cls=Nf4Linear) -> Nf4Linear:
    """A real blob in a slab, at ``start``."""
    out_features, in_features = w.shape
    blob = quant.quantize_tensor(w)
    slab = torch.zeros(start + blob.numel(), dtype=torch.uint8)
    slab[start:] = blob
    scratch = torch.empty(w.numel(), dtype=torch.bfloat16, device=slab.device)
    lin = cls(in_features, out_features, scratch=scratch, start=start, bias=bias)
    lin.slab = slab
    return lin


def _fixture():
    gen = torch.Generator().manual_seed(3)
    w = torch.randn(6, 16, generator=gen, dtype=torch.float32)
    x = torch.randn(1, 5, 16, generator=gen, dtype=torch.float32)
    bias = torch.randn(6, generator=gen).to(torch.bfloat16)
    return w, x, bias


def _decoded(w):
    return quant.dequantize(quant.quantize_tensor(w), w.shape, torch.bfloat16)


def test_forward_is_bf16_math_between_fp32_edges():
    w, x, bias = _fixture()
    y = _linear(w, bias=bias, start=64)(x)
    expect = (F.linear(x.reshape(-1, 16).to(torch.bfloat16), _decoded(w)) + bias).float()
    assert y.dtype == torch.float32 and y.shape == (1, 5, 6)
    assert torch.equal(y.view(torch.int32), expect.reshape(1, 5, 6).view(torch.int32))


def test_the_layer_is_a_linear_with_an_empty_weight():
    w, _, bias = _fixture()
    lin = _linear(w, bias=bias)
    assert isinstance(lin, torch.nn.Linear)
    assert lin.compute_dtype == lin.weight.dtype == lin.bias.dtype == torch.bfloat16
    assert lin.weight.numel() == 0
    assert not lin.weight.requires_grad and not lin.bias.requires_grad
    assert _linear(w).bias is None


def test_matmul_saves_no_weights():
    """Poison the scratch between the passes: the VJP still gets the right matrix."""
    w, x, _ = _fixture()
    lin = _linear(w)
    x2d = x.reshape(-1, 16).to(torch.bfloat16).requires_grad_(True)
    y = Nf4Matmul.apply(x2d, lin)
    assert not any(isinstance(v, torch.Tensor) for v in vars(y.grad_fn).values())
    assert y.grad_fn.saved_tensors == ()
    lin.scratch.fill_(float("nan"))
    g = torch.ones_like(y)
    y.backward(g)
    assert torch.equal(x2d.grad.view(torch.int16), torch.mm(g, _decoded(w)).view(torch.int16))


def test_plain_is_byte_equal_though_the_scratch_is_reused():
    w, x, bias = _fixture()
    g = torch.randn(5, 6, generator=torch.Generator().manual_seed(4))
    grads = []
    for cls in (Nf4Linear, arms.PlainLinear):
        a = x.clone().requires_grad_(True)
        lin = _linear(w, bias=bias, cls=cls)
        y = lin(a)
        lin.scratch.fill_(float("nan"))
        y.reshape(5, 6).backward(g)
        grads.append(a.grad.view(torch.int32))
    assert torch.equal(*grads)


def test_a_released_layer_cannot_decode():
    w, x, _ = _fixture()
    lin = _linear(w)
    lin.slab = None
    with pytest.raises(TypeError):
        lin(x)


def test_scratch_is_written_from_offset_zero():
    w, _, _ = _fixture()
    lin = _linear(w)
    out = lin.decoded()
    assert out.storage_offset() == 0
    assert out.untyped_storage()._cdata == lin.scratch.untyped_storage()._cdata


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Metal")
@pytest.mark.parametrize("shape", [(7, 1), (64, 64), (130, 257), (896, 96)], ids=str)
def test_the_real_kernel_decodes_a_slab_on_mps(shape):
    """Numeric equality at normal values. GPU against CPU bits are a probe, not required."""
    w = _weights(shape, "randn")
    lin = _linear(w, start=64)
    lin.slab, lin.scratch = lin.slab.to("mps"), lin.scratch.to("mps")
    got = lin.decoded().cpu()
    assert torch.equal(got.float(), _decoded(w).float())
    x = torch.randn(3, shape[1], generator=torch.Generator().manual_seed(6)).to("mps")
    assert torch.isfinite(lin(x).cpu()).all()
