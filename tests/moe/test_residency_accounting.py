"""The expert residency's accounting is the device bytes it holds (CPU emulation of the bulk path).

``_bulk`` copies a run of back-to-back slabs into one buffer and gives each key a view of it; the
buffer is freed only when its last view goes. The two CUDA-only seams (``host_runs``,
``transfer_run``) are emulated on the CPU as the CUDA code builds them -- one buffer per run, a view
per key -- and then: at a capacity that is not a whole number of layers, the bytes alive never pass
the capacity and always equal ``_resident_bytes``; ``_shrink_for`` frees what it says it frees."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.moe.contracts import ExpertLayout  # noqa: E402
from poros.moe.residency import ExpertBlockManager  # noqa: E402
from poros.moe.store import ExpertSlabArena  # noqa: E402

LAYERS, EXPERTS = 6, 8


def _residency(capacity_layers, look_ahead=0):
    layout = ExpertLayout(num_layers=LAYERS, experts_per_layer=EXPERTS, top_k=1,
                          tensor_names=("gate_up_proj", "down_proj"))
    shapes = {"gate_up_proj": (8, 64), "down_proj": (64, 4)}
    arena = ExpertSlabArena(layout, shapes)
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            for name, shape in shapes.items():
                arena.write((layer, e), name, torch.full(shape, float(layer)))
    store = arena.build(torch.float32)
    per_layer = EXPERTS * store.slice_nbytes((0, 0))
    res = ExpertBlockManager(store, streamed=True, capacity_bytes=int(capacity_layers * per_layer),
                             look_ahead=look_ahead)
    src = store.source

    def host_runs(keys):  # a layer's slabs back to back in pinned RAM (RamSink's carving)
        runs: dict = {}
        for k in keys:
            runs.setdefault(k[0], []).append((k, src._slab(k)))
        return list(runs.values())

    def transfer_run(run, nbytes=None):  # _TransferEngine.transfer_run without the CUDA stream
        nbytes = nbytes or [s.numel() for _, s in run]
        buf = torch.empty(sum(nbytes), dtype=torch.uint8)
        views, start = [], 0
        for (key, slab), n in zip(run, nbytes):
            view = buf[start:start + n]
            if slab is None:  # a streaming source's run: the store reads into the view
                store.load_slice(key, view)
            else:
                view.copy_(slab)
            views.append(view)
            start += n
        return views

    res._engine.host_runs, res._engine.transfer_run = host_runs, transfer_run
    return store, res, per_layer


def _as_streaming_source(res):
    """The streaming source's seams (the SSD store's, on CUDA): no host runs; the request is one
    run whose slabs the store reads (``source_runs``), counted."""
    reads = []
    res._engine.host_runs = lambda keys: None
    res._engine.source_runs = lambda keys: (reads.append(len(keys)) or [[(k, None) for k in keys]])
    return reads


def _alive(res) -> int:
    """Device bytes the resident slabs keep alive: each distinct buffer once."""
    held = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
            for v in res._resident.values()}
    return sum(held.values())


@pytest.mark.parametrize("capacity_layers", [2.6, 3.0, 4.4])
def test_bytes_alive_are_the_accounted_bytes_and_within_the_capacity(capacity_layers):
    store, res, _ = _residency(capacity_layers)
    for _ in range(2):  # a forward and a backward sweep, as an update makes
        for order in (range(LAYERS), reversed(range(LAYERS))):
            for layer in order:
                res.ensure([(layer, e) for e in range(EXPERTS)])
                assert _alive(res) == res._resident_bytes <= res._capacity
                for e in range(EXPERTS):  # the right bytes, whatever went
                    assert torch.equal(res.resident_qbytes((layer, e)),
                                       store.source._slab((layer, e)))


@pytest.mark.parametrize("capacity_layers", [1.0, 2.6, 4.4])
def test_a_streaming_sources_layer_lands_in_one_buffer(capacity_layers):
    """The SSD store's slabs (no host runs) come as one run a request when the cache holds
    several layers: one buffer per layer's request, each key a view of it; a one-layer cache
    (path efficient) lands each slab alone. The accounting is the bytes alive, the bytes the
    store's, either way."""
    store, res, per_layer = _residency(capacity_layers)
    runs = _as_streaming_source(res)
    for _ in range(2):
        for order in (range(LAYERS), reversed(range(LAYERS))):
            for layer in order:
                res.ensure([(layer, e) for e in range(EXPERTS)])
                assert _alive(res) == res._resident_bytes <= res._capacity
                held = {res.resident_qbytes((layer, e)).untyped_storage().data_ptr()
                        for e in range(EXPERTS)}
                assert len(held) == (1 if capacity_layers >= 2 else EXPERTS)
                for e in range(EXPERTS):
                    assert torch.equal(res.resident_qbytes((layer, e)),
                                       store.source._slab((layer, e)))
    assert all(n == EXPERTS for n in runs) and bool(runs) is (capacity_layers >= 2)


@pytest.mark.parametrize("streaming", [True, False])
def test_a_look_ahead_slice_never_evicts_a_streaming_sources_run(streaming):
    """Room for 23 of 24 slabs: layers 0 and 1 whole, then layer 2 but its last expert. Its
    look-ahead slice (the expert no token picked) would need a run gone; from a streaming
    source (the SSD store, where that layer would be read again from the disk) it is not loaded
    instead. A host-RAM store keeps its behaviour: a run goes for it."""
    store, res, _ = _residency(2.9, look_ahead=1)
    if streaming:
        _as_streaming_source(res)
        res._engine.streaming = True  # the CUDA seam's flag, on the CPU
    for layer in (0, 1):
        res.ensure([(layer, e) for e in range(EXPERTS)])
    res.ensure([(2, e) for e in range(EXPERTS - 1)])
    kept = all((layer, e) in res._resident for layer in (0, 1) for e in range(EXPERTS))
    assert kept is streaming and ((2, EXPERTS - 1) in res._resident) is not streaming
    assert _alive(res) == res._resident_bytes <= res._capacity
    for e in range(EXPERTS - 1):
        assert torch.equal(res.resident_qbytes((2, e)), store.source._slab((2, e)))


def test_runs_from_a_source_only_with_its_host_cache(tmp_path):
    """A request lands in one buffer for the SSD store with its host cache on a device; without
    the cache (disk-bound: the per-slab copies hide behind the reads) each slab lands alone; a
    host-RAM store has its own runs; the CPU has no copy stream."""
    from poros.moe.residency import runs_from_source
    from poros.sss import RamSource, SsdSource, SssWriter

    w = SssWriter(tmp_path / "m.sss", [((0, 0), 4096), ((0, 1), 4096)], meta={"t": 1})
    for k in ((0, 0), (0, 1)):
        w.write(k, 0, torch.zeros(4096, dtype=torch.uint8))
    w.finish().close()
    with SsdSource(tmp_path / "m.sss", pin=False) as plain, \
            SsdSource(tmp_path / "m.sss", pin=False, cache_bytes=1 << 20) as cached:
        assert not plain.host_cache and cached.host_cache
        assert runs_from_source(cached, True) and not runs_from_source(cached, False)
        assert not runs_from_source(plain, True)
    assert not runs_from_source(RamSource({}), True) and not runs_from_source(None, True)


def test_shrink_frees_real_bytes():
    _, res, per_layer = _residency(3.0)
    for layer in range(3):
        res.ensure([(layer, e) for e in range(EXPERTS)])
    before = _alive(res)
    res._shrink_for(per_layer // 2, protected={2})  # asks for half a layer: a whole run goes
    assert _alive(res) == res._resident_bytes  # what stays is what is counted
    assert before - _alive(res) >= per_layer // 2  # and the bytes asked for really went


def test_shrink_to_floor_gives_back_everything_above_the_configured_budget():
    """Between passes the cache can go back to its configured budget (one layer): what stays is
    within it and counted, the sticky set is gone, and the bytes are the right ones after."""
    store, res, per_layer = _residency(1.0)
    res._capacity = 5 * per_layer  # as grow_expert_cache leaves it
    res.set_sticky([0, 2])
    for layer in range(LAYERS - 1):
        res.ensure([(layer, e) for e in range(EXPERTS)])
    assert res._resident_bytes > per_layer
    freed = res.shrink_to_floor()
    assert freed > 0 and res._capacity == per_layer
    assert _alive(res) == res._resident_bytes <= per_layer
    assert not res._sticky
    for layer in range(LAYERS):  # and it streams on at one layer, the same bytes
        res.ensure([(layer, e) for e in range(EXPERTS)])
        assert _alive(res) == res._resident_bytes <= res._capacity
        assert torch.equal(res.resident_qbytes((layer, 0)), store.source._slab((layer, 0)))
