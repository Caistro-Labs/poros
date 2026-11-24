"""nf4_dequant_cache: config gate (CPU) and bitwise equality with stock bitsandbytes (CUDA)."""
import pytest


def test_the_cache_parses_on_any_path():
    pytest.importorskip("torch")
    from poros.core.config import PorosConfig

    # on any path (read by nothing on the sequential loop: a saved config reloads)
    for path in (None, "sequential", "efficient", "fast"):
        cfg = PorosConfig(quantization="nf4", path=path, nf4_dequant_cache_mb=512)
        assert cfg.nf4_dequant_cache_mb == 512
    assert PorosConfig().nf4_dequant_cache_mb is None  # auto


def test_another_precision_is_refused_by_name():
    pytest.importorskip("torch")
    from poros.core.config import PorosConfig
    from poros.core.errors import PorosConfigError

    for q in ("bf16", "fp16", "bf16_bitwise", "int8"):
        with pytest.raises(PorosConfigError, match=f"quantization: '{q}' is not one Poros trains"):
            PorosConfig(quantization=q)
    assert PorosConfig(quantization="nf4", path="efficient").path == "efficient"


def test_scope_restores_matmul_4bit():
    pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    from poros.core.nf4_cache import DequantCache

    orig = bnb.matmul_4bit
    cache = DequantCache(1 << 30)
    with pytest.raises(RuntimeError, match="boom"):
        with cache.scope():
            if cache.enabled:
                assert bnb.matmul_4bit is not orig
            raise RuntimeError("boom")
    assert bnb.matmul_4bit is orig
    assert cache._store == {} and cache._bytes == 0


@pytest.mark.gpu
@pytest.mark.parametrize("double_quant", [True, False])
def test_cached_matmul_bitwise_equals_stock(double_quant):
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from poros.core.nf4_cache import DequantCache

    cache = DequantCache(1 << 30)
    if not cache.enabled:
        pytest.skip(f"bitsandbytes {bnb.__version__} not mirrored")
    torch.manual_seed(0)
    lin = bnb.nn.Linear4bit(
        2048, 5632, bias=True, compute_dtype=torch.bfloat16,
        compress_statistics=double_quant, quant_type="nf4",
    )
    lin.weight = bnb.nn.Params4bit(
        torch.randn(5632, 2048) * 0.02, requires_grad=False,
        compress_statistics=double_quant, quant_type="nf4",
    )
    lin.bias = torch.nn.Parameter(torch.randn(5632) * 0.01, requires_grad=False)
    lin = lin.to(torch.bfloat16).cuda()
    # M spans bnb's fused range (small M) and its dequant + F.linear range (large M).
    for M in (1, 3, 8, 64, 129, 300, 700, 1600):
        x = torch.randn(1, M, 2048, device="cuda", dtype=torch.bfloat16)
        go = torch.randn(1, M, 5632, device="cuda", dtype=torch.bfloat16)

        def fwd_bwd():
            with torch.no_grad():
                y0 = lin(x)
            xa = x.detach().requires_grad_(True)
            y1 = lin(xa)
            (ga,) = torch.autograd.grad(y1, [xa], grad_outputs=go)
            return y0, y1.detach(), ga

        ref = fwd_bwd()
        with cache.scope():
            got = fwd_bwd()
            got2 = fwd_bwd()  # second pass: every dequantized weight is a cache hit
        for r, a, b in zip(ref, got, got2):
            assert torch.equal(r, a) and torch.equal(r, b), M
    assert cache.stats["hits"] > 0 and cache.stats["fused"] > 0
    assert bnb.matmul_4bit is not cache._matmul_4bit


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


@pytest.mark.gpu
def test_boundary_offload_roundtrip_and_pool_reuse(monkeypatch):
    torch = _cuda_or_skip()
    from poros.core.boundary_offload import BoundaryOffload

    # its own allowance: hosts differ in what the process budget grants
    monkeypatch.setenv("POROS_BOUNDARY_PIN", "1G")
    offl = BoundaryOffload(torch.device("cuda"))
    torch.manual_seed(0)
    for rnd in range(2):  # the second round must reuse the first round's buffers
        xs = [torch.randn(1, T, 1024, device="cuda", dtype=torch.bfloat16)
              for T in (512, 300, 77, 512)]
        ref = [x.clone() for x in xs]
        parked = [offl.offload(x) for x in xs]
        del xs
        got = [offl.ready(offl.fetch(p)) for p in reversed(parked)][::-1]
        if offl.stats["host_full"]:
            pytest.skip("this host refuses page-locked memory")
        for r, g in zip(ref, got):
            assert torch.equal(r, g)
        if rnd == 0:
            n0 = offl.stats["slabs"]
        offl.begin_update()
    assert offl.stats["slabs"] == n0 == 1 and offl.stats["kept_on_device"] == 0
    assert offl.stats["offloaded"] == 8


@pytest.mark.gpu
def test_boundary_offload_refused_stays_on_device(monkeypatch):
    torch = _cuda_or_skip()
    from poros.core.boundary_offload import BoundaryOffload
    from poros.sss import pinning

    monkeypatch.delenv("POROS_BOUNDARY_PIN", raising=False)
    monkeypatch.setenv("POROS_BOUNDARY_HOST_MAX", "0")  # no RAM slabs either: test_boundary_offload
    monkeypatch.setattr(pinning, "reserve", lambda n: False)
    offl = BoundaryOffload(torch.device("cuda"))
    x = torch.randn(1, 64, 256, device="cuda", dtype=torch.bfloat16)
    p = offl.offload(x)
    assert p is x and offl.stats["kept_on_device"] == 1
    assert torch.equal(offl.ready(offl.fetch(p)), x)



@pytest.mark.gpu
def test_boundary_offload_allowance_zero_stays_on_device(monkeypatch):
    torch = _cuda_or_skip()
    from poros.core.boundary_offload import BoundaryOffload

    monkeypatch.setenv("POROS_BOUNDARY_PIN", "0")
    offl = BoundaryOffload(torch.device("cuda"))
    x = torch.randn(1, 64, 256, device="cuda", dtype=torch.bfloat16)
    assert offl.offload(x) is x and offl.stats["slabs"] == 0


def test_block_major_takes_the_efficient_path_on_nf4():
    pytest.importorskip("torch")
    from poros.core.config import PorosConfig, train_path

    seq = PorosConfig(quantization="nf4", lora_dropout=0.0)  # parsed: left to the run
    assert seq.path is None and train_path(seq) == "sequential"
    bm = PorosConfig(quantization="nf4", lora_dropout=0.0)
    bm.apply_cuda_defaults()
    assert train_path(bm) == "efficient"
    assert bm.nf4_dequant_cache_mb is None  # auto-sized
    assert bm.fill_uninitialized_memory is False and bm.cuda_expandable_segments is True
    # a CUDA run takes it unless told not to: tests/core/test_block_major_default.py


@pytest.mark.gpu
def test_boundary_prealloc_locks_whole_slabs(monkeypatch):
    torch = _cuda_or_skip()
    from poros.core.boundary_offload import SLAB_BYTES, BoundaryOffload

    monkeypatch.delenv("POROS_BOUNDARY_PIN", raising=False)
    offl = BoundaryOffload(torch.device("cuda"))
    got = offl.prealloc(2 * SLAB_BYTES + 5)
    if got == 0:
        pytest.skip("this host refuses page-locked memory")
    assert got == 2 * SLAB_BYTES and offl.stats["slabs"] == 2
    x = torch.randn(1, 512, 1024, device="cuda", dtype=torch.bfloat16)
    p = offl.offload(x)
    assert p is not x and torch.equal(offl.ready(offl.fetch(p)), x)
    assert offl.stats["slabs"] == 2  # used the reserved slabs, grew none
