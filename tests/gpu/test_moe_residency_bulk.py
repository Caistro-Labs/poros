"""The expert residency's bulk transfer (CUDA): a layer's experts whose pinned host slabs lie back
to back move in one copy into one device buffer, each slab a view of it -- the same bytes, the
same accounting and eviction as one copy per expert."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the bulk transfer is the CUDA path", allow_module_level=True)

from poros.moe.contracts import ExpertLayout  # noqa: E402
from poros.moe.residency import ExpertBlockManager  # noqa: E402
from poros.moe.store import ExpertSlabArena  # noqa: E402


def _store(layers=3, experts=6):
    layout = ExpertLayout(num_layers=layers, experts_per_layer=experts, top_k=2,
                          tensor_names=("gate_up_proj", "down_proj"))
    shapes = {"gate_up_proj": (64, 32), "down_proj": (32, 32)}
    arena = ExpertSlabArena(layout, shapes)
    g = torch.Generator().manual_seed(0)
    for layer in range(layers):
        for e in range(experts):
            for name, shape in shapes.items():
                arena.write((layer, e), name, torch.randn(shape, generator=g).to(torch.bfloat16))
    return arena.build(torch.bfloat16)


def _check(res, store):
    host = dict(store.source.items())
    total = 0
    for key, slab in res._resident.items():
        assert torch.equal(res.resident_qbytes(key).cpu(), host[key])
        total += slab.numel()
    assert total == res._resident_bytes


def test_bulk_moves_the_same_bytes_and_accounts_as_per_expert():
    store = _store()
    per = store.slice_nbytes((0, 0))
    res = ExpertBlockManager(store, streamed=True, capacity_bytes=2 * 6 * per, look_ahead=0,
                             device="cuda")
    layer = [(0, e) for e in range(6)]
    res.ensure(layer)
    assert any(s.untyped_storage().nbytes() > per for s in res._resident.values())  # one buffer
    _check(res, store)
    res.prefetch([(1, e) for e in range(6)], keep=[0])
    _check(res, store)
    res.ensure([(2, e) for e in range(6)])  # evicts layer 0 for layer 2
    assert all(k[0] in (1, 2) for k in res._resident)
    _check(res, store)
    assert res.bytes_transferred == 3 * 6 * per
