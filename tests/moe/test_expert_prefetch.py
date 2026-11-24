"""The expert residency's prefetch (CPU): ``layer_resident`` follows every load and eviction, a
prefetch says whether it made its layer resident, it never evicts the layers it is told to keep, and
``prefetch_neighbor`` takes the layers ahead in order until one finds no room."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from poros.moe.contracts import ExpertLayout  # noqa: E402
from poros.moe.modules import prefetch_neighbor  # noqa: E402
from poros.moe.residency import ExpertBlockManager  # noqa: E402
from poros.moe.store import ExpertSlabArena  # noqa: E402


def _residency(layers=6, experts=2, capacity_layers=3):
    layout = ExpertLayout(num_layers=layers, experts_per_layer=experts, top_k=1,
                          tensor_names=("gate_up_proj", "down_proj"))
    shapes = {"gate_up_proj": (8, 64), "down_proj": (64, 4)}
    arena = ExpertSlabArena(layout, shapes)
    for layer in range(layers):
        for e in range(experts):
            for name, shape in shapes.items():
                arena.write((layer, e), name, torch.full(shape, float(layer)))
    store = arena.build(torch.float32)
    per_layer = experts * store.slice_nbytes((0, 0))
    res = ExpertBlockManager(store, streamed=True, capacity_bytes=capacity_layers * per_layer,
                             look_ahead=0)
    return store, res


def _keys(layer, experts=2):
    return [(layer, e) for e in range(experts)]


def test_prefetch_reports_and_keeps():
    _, res = _residency()
    for layer in range(3):
        res.ensure(_keys(layer))
    assert [res.layer_resident(layer) for layer in range(6)] == [True] * 3 + [False] * 3
    # no layer outside keep + the target's own: no room
    assert res.prefetch(_keys(3), keep=[0, 1, 2]) is False
    assert not res.layer_resident(3)
    assert res.prefetch(_keys(3), keep=[1, 2]) is True  # layer 0 makes room
    assert [res.layer_resident(layer) for layer in range(6)] == [False, True, True, True, False,
                                                                  False]


def test_prefetch_neighbor_takes_layers_ahead_until_no_room():
    store, res = _residency()
    for layer in range(3):
        res.ensure(_keys(layer))
    res.ensure(_keys(0))  # layer 0 most recently used, as after a backward pass
    block = types.SimpleNamespace(layer_idx=1, num_experts=2, store=store, residency=res)
    prefetch_neighbor(block, +1)  # layer 0 may go for layer 3; layer 4 then finds no room
    assert [res.layer_resident(layer) for layer in range(6)] == [False, True, True, True, False,
                                                                  False]


def test_sticky_layers_spread_and_leave_room_to_stream():
    from poros.sss.tune import sticky_layers

    keep = sticky_layers(48, 29)
    assert len(keep) == 26 and keep == sorted(set(keep))
    assert max(b - a for a, b in zip(keep, keep[1:])) <= 2  # never two streamed layers in a row
    assert sticky_layers(3, 2) == []  # a cache of fewer layers than it streams keeps none
    assert sticky_layers(3, 6) == [0, 1, 2]  # every layer fits


def test_eviction_keeps_sticky_layers_unless_required():
    _, res = _residency(layers=4, capacity_layers=3)
    res.set_sticky([0])
    for layer in range(4):
        res.ensure(_keys(layer))
    assert {k[0] for k in res._resident} == {0, 2, 3}  # layer 1 went, not sticky layer 0
    assert res.prefetch(_keys(1), keep=[2, 3]) is False  # a prefetch never takes a sticky layer
    res.ensure(_keys(1))  # but a required one does, after the other layers (here: none left)
    assert res.layer_resident(1) and sum(res.layer_resident(layer) for layer in range(4)) == 3


class _Streaming:
    """A streaming source over a RAM store's slabs that can hand over only ``ready_keys`` without
    a disk read; every read is recorded (the SSD store's ``ready`` seam)."""

    def __init__(self, ram, ready_keys):
        self.ram, self.ready_keys, self.reads = ram, set(ready_keys), []

    def ready(self, key):
        return key in self.ready_keys

    def read_into(self, key, dst):
        self.reads.append(key)
        self.ram.read_into(key, dst)


def test_prefetch_neighbor_from_a_streaming_source_takes_only_what_it_can_hand_over():
    """From the SSD store only slabs it holds or has landed are started ahead, never one that
    would wait on the disk; a layer it cannot hand over whole ends the walk (here layer 4, whose
    expert 1 is still on the disk: its expert 0 starts, layer 5 does not)."""
    store, res = _residency(capacity_layers=6)
    ready = [*_keys(2), *_keys(3), (4, 0), *_keys(5)]
    src = _Streaming(store.source, ready)
    store._source = src
    res.ensure(_keys(1))
    block = types.SimpleNamespace(layer_idx=1, num_experts=2, store=store, residency=res)
    prefetch_neighbor(block, +1)
    assert src.reads == [*_keys(1), *_keys(2), *_keys(3), (4, 0)]
    assert [res.layer_resident(layer) for layer in range(6)] == [False, True, True, True, False,
                                                                  False]
    for key in src.reads:  # the bytes the store holds
        assert torch.equal(res.resident_qbytes(key), store.source.ram._slab(key))


def test_prefetch_neighbor_skips_a_source_that_cannot_say_what_it_holds():
    store, res = _residency(capacity_layers=6)
    reads = []
    store._source = types.SimpleNamespace(read_into=lambda key, dst: reads.append(key))
    block = types.SimpleNamespace(layer_idx=1, num_experts=2, store=store, residency=res)
    prefetch_neighbor(block, +1)
    assert reads == []
