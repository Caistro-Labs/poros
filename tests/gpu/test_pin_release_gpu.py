"""Page-locked memory a trainer's stores hold is given back when they go, on a real card:
two block stores and two boundary pools, one after the other in
one process under POROS_PIN_BUDGET, each pins the same bytes; dropping one returns its grant."""

from __future__ import annotations

import gc

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("page-locked memory needs CUDA", allow_module_level=True)
transformers = pytest.importorskip("transformers")

from poros.core.boundary_offload import make_boundary_offload  # noqa: E402
from poros.core.residency import BlockManager  # noqa: E402
from poros.sss import pinning  # noqa: E402


@pytest.fixture
def budget(monkeypatch):
    monkeypatch.setenv("POROS_PIN_BUDGET", "3G")
    pinning.reset()
    yield
    pinning.reset()


def _store():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512,
                                   num_hidden_layers=4, num_attention_heads=4,
                                   num_key_value_heads=2)
    model = transformers.Qwen2ForCausalLM(cfg).to("cuda", torch.bfloat16)
    bm = BlockManager(model, block_size=1)
    bm.offload_all_to_cpu()
    return model, bm


def test_two_block_stores_in_one_process_pin_the_same(budget):
    model, bm = _store()
    first = pinning._state["granted"]
    stored = [t for b in bm._cpu_weight_store.values() for t in b.values()]
    assert first > 0 and all(t.is_pinned() for t in stored)
    del model, bm, stored
    gc.collect()
    assert pinning._state["granted"] == 0
    model, bm = _store()
    assert pinning._state["granted"] == first
    assert all(t.is_pinned() for b in bm._cpu_weight_store.values() for t in b.values())


def test_a_boundary_pool_returns_its_slabs(budget):
    pool = make_boundary_offload(True, torch.device("cuda"))
    assert pool._new_slab() and pinning._state["granted"] > 0
    del pool
    gc.collect()
    assert pinning._state["granted"] == 0
