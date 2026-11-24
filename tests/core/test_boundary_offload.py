"""path: efficient where the host will not page-lock: pooled ordinary-RAM slabs.

The pool logic runs on the CPU with the GPU faked out (no streams; a "pinned" allocation is a
plain tensor or a cuMemHostAlloc-like failure) and 1 MiB slabs. The copies themselves are
checked on CUDA.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from poros.core import boundary_offload as bo  # noqa: E402
from poros.sss import pinning  # noqa: E402

SLAB = 1 << 20
SHAPES = [((1, 64, 1024), torch.float32), ((1, 77, 1000), torch.bfloat16),
          ((2, 33, 512), torch.float32), ((1, 512, 1024), torch.bfloat16)]  # 2 slabs: 3, then 1


class _FakeTorch:
    """``torch`` as boundary_offload sees it, minus the GPU."""

    def __init__(self, pin_ok):
        self.pin_ok, self.pinned, self.pageable = pin_ok, 0, 0
        self.cuda = SimpleNamespace(Stream=lambda device=None: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *a, pin_memory=False, **k):
        if pin_memory:
            if not self.pin_ok():
                raise RuntimeError("CUDA error: out of memory")  # cuMemHostAlloc refused
            self.pinned += 1
        else:
            self.pageable += 1
        return torch.empty(*a, **k)


@pytest.fixture
def make(monkeypatch):
    import poros.core.env

    monkeypatch.setattr(bo, "SLAB_BYTES", SLAB)
    monkeypatch.delenv("POROS_BOUNDARY_PIN", raising=False)
    monkeypatch.delenv("POROS_BOUNDARY_HOST_MAX", raising=False)

    def build(*, pin_ok=lambda: True, grant=True, free_gib=1.0):
        calls = SimpleNamespace(reserve=0, release=0, free=0)

        def reserve(n):
            calls.reserve += 1
            return grant

        def release(n):
            calls.release += 1

        def host_ram_free():
            calls.free += 1
            if free_gib == "never":
                raise AssertionError("the RAM pool was sized while pinning worked")
            return free_gib

        fake = _FakeTorch(pin_ok)
        monkeypatch.setattr(bo, "torch", fake)
        monkeypatch.setattr(pinning, "reserve", reserve)
        monkeypatch.setattr(pinning, "release", release)
        monkeypatch.setattr(poros.core.env, "host_ram_free", host_ram_free)
        return bo.BoundaryOffload(torch.device("cuda")), fake, calls

    return build


def _cycle(offl, seed):
    """One update: park every shape (as offload does, minus the streams), then read all back."""
    offl.begin_update()
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(s, generator=g).to(dt) for s, dt in SHAPES]
    parked = []
    for x in xs:
        n = x.numel() * x.element_size()
        buf = offl._buffer(n)
        if buf is None:
            parked.append(None)
            continue
        host = bo._Host(buf, x.shape, x.dtype, n)
        host.view().copy_(x)
        parked.append(host)
    for x, p in zip(xs, parked):  # read after every write: no buffer overlaps another
        if p is not None:
            got = p.view()
            assert got.dtype == x.dtype and got.shape == x.shape and got.is_contiguous()
            assert torch.equal(got.view(torch.uint8), x.view(torch.uint8))
    return parked


def test_refused_budget_parks_in_pooled_ram_without_growth(make):
    offl, fake, calls = make(grant=False)
    ptrs = None
    for rnd in range(5):
        assert all(p is not None for p in _cycle(offl, rnd))
        if ptrs is None:
            ptrs, n_alloc = [s.data_ptr() for s in offl._slabs], fake.pageable
        assert [s.data_ptr() for s in offl._slabs] == ptrs and fake.pageable == n_alloc  # reused
    assert offl.stats["pageable_slabs"] == len(ptrs) == 2 and offl.stats["slabs"] == 0
    assert fake.pinned == 0 and calls.free == 1 and not offl.stats["host_full"]


def test_pin_failure_switches_to_ram_and_stops_asking(make, caplog):
    offl, fake, calls = make(pin_ok=lambda: False)
    with caplog.at_level(logging.WARNING, logger=bo.__name__):
        for rnd in range(3):
            assert all(p is not None for p in _cycle(offl, rnd))
    assert offl.stats["host_full"] and offl.stats["pageable_slabs"] == 2
    assert calls.reserve == 1 and calls.release == 1  # the refused slab's grant went back
    msgs = [r.getMessage() for r in caplog.records if r.name == bo.__name__]
    assert len(msgs) == 1 and "page-locked ceiling" in msgs[0] and "0.50 GiB" in msgs[0]


def test_prealloc_failure_marks_the_host_full(make):
    left = iter([True, False])
    offl, fake, _ = make(pin_ok=lambda: next(left, False))
    assert offl.prealloc(4 * SLAB) == SLAB and offl.stats["host_full"]
    assert all(p is not None for p in _cycle(offl, 0))  # the rest in RAM, not on the device
    assert offl.stats["slabs"] == 1 and offl.stats["pageable_slabs"] == 1


def test_ram_cap_keeps_the_rest_on_device(make, monkeypatch):
    monkeypatch.setenv("POROS_BOUNDARY_HOST_MAX", "1M")
    offl, fake, calls = make(grant=False, free_gib="never")
    parked = _cycle(offl, 0)
    assert None not in parked[:3] and parked[3] is None
    assert offl.stats["pageable_slabs"] == 1 and calls.free == 0
    x = torch.zeros(SLAB // 4)  # one more boundary: no slab for it, it stays where it is
    assert offl.offload(x) is x and offl.stats["kept_on_device"] == 1


@pytest.mark.parametrize("free_gib, slabs", [(3 / 1024, 1), (None, 0)])
def test_default_cap_is_half_the_free_ram_measured_once(make, caplog, free_gib, slabs):
    offl, fake, calls = make(grant=False, free_gib=free_gib)
    with caplog.at_level(logging.WARNING, logger=bo.__name__):
        for rnd in range(3):
            _cycle(offl, rnd)
    assert offl.stats["pageable_slabs"] == fake.pageable == slabs  # 3 MiB free: 1; unreadable: 0
    assert calls.free == 1 and len([r for r in caplog.records if r.name == bo.__name__]) == 1


def test_host_max_zero_keeps_boundaries_on_device(make, monkeypatch):
    monkeypatch.setenv("POROS_BOUNDARY_HOST_MAX", "0")
    offl, fake, _ = make(grant=False, free_gib="never")
    assert _cycle(offl, 0) == [None] * len(SHAPES) and fake.pageable == 0
    x = torch.zeros(16)
    assert offl.offload(x) is x


def test_ram_allocation_failure_stops_asking(make, monkeypatch):
    offl, fake, _ = make(grant=False)

    def no_ram(*a, pin_memory=False, **k):
        fake.pageable += 1
        raise RuntimeError("DefaultCPUAllocator: can't allocate memory")

    monkeypatch.setattr(fake, "empty", no_ram)
    assert _cycle(offl, 0) == [None] * len(SHAPES) and _cycle(offl, 1) == [None] * len(SHAPES)
    assert fake.pageable == 1 and offl.stats["pageable_slabs"] == 0


def test_pinning_that_works_never_touches_ram(make, monkeypatch):
    offl, fake, calls = make(free_gib="never")  # the budget grants every slab
    for rnd in range(3):
        assert all(p is not None for p in _cycle(offl, rnd))
    assert offl.stats["slabs"] == fake.pinned == 2 and fake.pageable == 0 and calls.free == 0
    # an allowance that is simply used up keeps the overflow on the device, as before
    monkeypatch.setenv("POROS_BOUNDARY_PIN", "1M")
    offl, fake, calls = make(free_gib="never")
    assert _cycle(offl, 0)[3] is None and offl.stats["slabs"] == 1 and fake.pageable == 0


@pytest.mark.parametrize("grant", [True, False])
def test_slabs_reserved_before_the_weights_are_a_head_start(make, grant):
    """Past the reserved slabs the pool grows as it would without them: the pin budget, then
    RAM (a 32B's 2 reserved slabs held 50 of its 120 boundaries; 70 stayed on the GPU)."""
    offl, fake, calls = make(grant=grant, free_gib="never" if grant else 1.0)
    assert offl.prealloc(SLAB + SLAB // 2) == SLAB and not offl.stats["host_full"]  # whole slabs
    for rnd in range(3):
        assert all(p is not None for p in _cycle(offl, rnd))
    assert calls.reserve == 1  # asked once for the second slab, then reused it
    assert (offl.stats["slabs"], offl.stats["pageable_slabs"]) == ((2, 0) if grant else (1, 1))


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


@pytest.mark.gpu
@pytest.mark.parametrize("refusal", ["budget", "ceiling"])
def test_pageable_slabs_round_trip_exact_on_cuda(monkeypatch, refusal):
    _cuda_or_skip()
    monkeypatch.delenv("POROS_BOUNDARY_PIN", raising=False)
    monkeypatch.delenv("POROS_BOUNDARY_HOST_MAX", raising=False)
    if refusal == "budget":
        monkeypatch.setattr(pinning, "reserve", lambda n: False)
    else:  # the budget grants, cuMemHostAlloc refuses
        monkeypatch.setattr(pinning, "reserve", lambda n: True)
        monkeypatch.setattr(pinning, "release", lambda n: None)
        monkeypatch.setattr(bo, "torch", _FakeTorch(lambda: False))
        monkeypatch.setattr(bo.torch, "cuda", torch.cuda)
    offl = bo.BoundaryOffload(torch.device("cuda"))
    torch.manual_seed(0)
    for rnd in range(3):  # later rounds reuse the first round's slabs
        offl.begin_update()
        xs = [torch.randn(1, T, 1024, device="cuda").to(dt) for T in (512, 300, 77, 512)
              for dt in (torch.bfloat16, torch.float32)]
        ref = [x.clone() for x in xs]
        parked = [offl.offload(x) for x in xs]
        del xs
        got = [offl.ready(offl.fetch(p)) for p in reversed(parked)][::-1]
        for r, g in zip(ref, got):
            assert torch.equal(r.view(torch.uint8), g.view(torch.uint8))
        if rnd == 0:
            n0 = offl.stats["pageable_slabs"]
    assert offl.stats["slabs"] == 0 and offl.stats["kept_on_device"] == 0
    assert offl.stats["pageable_slabs"] == n0 >= 1
    assert offl.stats["host_full"] == (refusal == "ceiling")
