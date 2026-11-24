"""The MoE Triton kernels against the torch reference they replace (CUDA).

``poros.moe.kernels.dequantize_into`` must give ``quant.dequantize``'s values bit for bit: the
decoder only moves where the arithmetic happens, never what it is.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Triton kernels run on CUDA", allow_module_level=True)
pytest.importorskip("triton")

from poros.moe import kernels, quant  # noqa: E402
from poros.moe.contracts import ExpertLayout  # noqa: E402
from poros.moe.store import ExpertSlabArena  # noqa: E402

DEV = torch.device("cuda")


def _weight(shape, seed, zero_block=False):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * 0.02
    w[0, :3] = torch.tensor([0.5, -0.7, 1e-6])  # outliers set a block's absmax
    if zero_block:
        w.view(-1)[64:128] = 0.0  # an all-zero NF4 block decodes to exact zeros
    return w.to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(1536, 2048), (2048, 768), (96, 64), (64, 300)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dequantize_into_is_quant_dequantize(shape, dtype):
    blobs = [quant.quantize_tensor(_weight(shape, s, zero_block=s == 1)) for s in range(3)]
    slabs = torch.stack(blobs).to(DEV)
    want = quant.dequantize(slabs, shape, dtype)
    out = torch.empty((3, *shape), dtype=dtype, device=DEV)
    got = kernels.dequantize_into(out, kernels.slab_pointers(slabs), 0, shape)
    assert torch.equal(got, want)
    assert torch.equal(got[1].reshape(-1)[64:128], torch.zeros(64, dtype=dtype, device=DEV))


def test_store_decodes_resident_slabs_as_its_batched_dequantize():
    layout = ExpertLayout(num_layers=1, experts_per_layer=4, top_k=2,
                          tensor_names=("gate_up_proj", "down_proj"))
    shapes = {"gate_up_proj": (192, 128), "down_proj": (128, 96)}
    arena = ExpertSlabArena(layout, shapes)
    for e in range(4):
        arena.write((0, e), "gate_up_proj", _weight(shapes["gate_up_proj"], 10 + e))
        arena.write((0, e), "down_proj", _weight(shapes["down_proj"], 20 + e))
    store = arena.build(torch.bfloat16)
    keys = [(0, 3), (0, 1), (0, 2)]
    slabs = []
    for k in keys:  # separate device slabs, as the residency holds them
        dst = torch.empty(store.slice_nbytes(k), dtype=torch.uint8)
        store.load_slice(k, dst)
        slabs.append(dst.to(DEV))
    for name in ("gate_up_proj", "down_proj"):
        want = store.dequantize_experts(keys, name, torch.stack(slabs))
        got = store.dequantize_resident(keys, name, slabs)
        assert got is not None and torch.equal(got, want)


def test_slab_pointers_are_the_slabs_and_copied_without_a_host_sync():
    """The pointer table the decoder reads: each slab's (or row's) address, made on the host and
    copied asynchronously -- under torch's sync debug mode a host sync would raise -- then decoded
    as the stacked slabs are, bit for bit."""
    shape = (128, 96)
    rows = torch.stack([quant.quantize_tensor(_weight(shape, 30 + s)) for s in range(5)]).to(DEV)
    slabs = [rows[i].clone() for i in (3, 0, 4)]  # separate device slabs, out of order
    want = quant.dequantize(torch.stack(slabs), shape, torch.bfloat16)
    warm = torch.empty((3, *shape), dtype=torch.bfloat16, device=DEV)  # codebooks, compile
    kernels.dequantize_into(warm, kernels.slab_pointers(slabs), 0, shape)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            from_list = kernels.slab_pointers(slabs)
            from_rows = kernels.slab_pointers(rows)
            out = torch.empty((3, *shape), dtype=torch.bfloat16, device=DEV)
            got = kernels.dequantize_into(out, from_list, 0, shape)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert from_list.is_cuda and from_list.dtype == torch.int64
    assert from_list.tolist() == [s.data_ptr() for s in slabs]
    assert from_rows.tolist() == [rows[i].data_ptr() for i in range(rows.shape[0])]
    assert torch.equal(got, want)


def test_slab_pointers_where_pinning_is_refused(monkeypatch):
    """A machine that refuses to page-lock: the table crosses from pageable memory, the same
    addresses, the same decode."""
    shape = (128, 96)
    rows = torch.stack([quant.quantize_tensor(_weight(shape, 40 + s)) for s in range(4)]).to(DEV)
    slabs = [rows[i].clone() for i in (2, 0, 3)]
    want = quant.dequantize(torch.stack(slabs), shape, torch.bfloat16)
    calls = []

    def refuse(self, *args, **kwargs):
        calls.append(self.shape)
        raise RuntimeError("cannot pin: the page-locked memory limit is reached")

    monkeypatch.setattr(torch.Tensor, "pin_memory", refuse)
    from_list, from_rows = kernels.slab_pointers(slabs), kernels.slab_pointers(rows)
    assert len(calls) == 2
    assert from_list.tolist() == [s.data_ptr() for s in slabs]
    assert from_rows.tolist() == [rows[i].data_ptr() for i in range(rows.shape[0])]
    out = torch.empty((3, *shape), dtype=torch.bfloat16, device=DEV)
    assert torch.equal(kernels.dequantize_into(out, from_list, 0, shape), want)
