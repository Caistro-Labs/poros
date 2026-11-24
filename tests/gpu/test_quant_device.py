"""``quant.quantize_tensor`` on CUDA is byte-for-byte the CPU's: shapes with and without a tail
block, all-zero blocks, values at the edges of the NF4 midpoints, fp32 / fp16 / bf16 inputs, and
large and denormal magnitudes. Whole models are checked by the receipts (every expert of
Qwen3-30B-A3B, every linear of Qwen2.5-0.5B)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from poros.moe import quant  # noqa: E402


def _cases():
    g = torch.Generator().manual_seed(0)
    yield torch.randn(1536, 2048, generator=g) * 0.02
    yield torch.randn(7, 131, generator=g)  # a tail block of 37
    z = torch.randn(4, 256, generator=g)
    z[1] = 0.0  # all-zero blocks
    yield z
    mid = quant._nf4_midpoints()
    up, down = mid.nextafter(torch.ones_like(mid)), mid.nextafter(-torch.ones_like(mid))
    edge = torch.cat([mid, up, down, torch.tensor([1.0, -1.0, 0.0])])
    yield edge.repeat(64)[: 64 * 32]
    yield torch.randn(512, 64, generator=g) * 1e3  # large for fp16 too (near fp32's max the
    # double-quant offset, absmax - mean, overflows on the CPU path as well)
    yield torch.randn(512, 64, generator=g) * 1e-40  # denormal
    yield torch.randn(3, 1000, generator=g).to(torch.float16).float()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cuda_quantize_is_the_cpu_bytes(dtype):
    for i, w in enumerate(_cases()):
        w = w.to(dtype)
        assert torch.equal(quant.quantize_tensor(w), quant.quantize_tensor(w, "cuda")), (dtype, i)
