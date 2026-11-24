"""Page-locked host memory a trainer holds is given back when it goes: the block store's and
the boundary slabs' grants return to the process's budget
(POROS_PIN_BUDGET), so a second trainer in one process pins what the first did. CPU: the arena,
with the page-locking stood in for; the real pins are tests/gpu/test_pin_release_gpu.py."""

from __future__ import annotations

import gc

import pytest

torch = pytest.importorskip("torch")

from poros.sss import pinning  # noqa: E402


@pytest.fixture
def budget(monkeypatch):
    monkeypatch.setenv("POROS_PIN_BUDGET", "64M")
    monkeypatch.setattr(pinning.PinnedArena, "_new_slab",
                        lambda self, n: torch.empty(n, dtype=torch.uint8))
    pinning.reset()
    yield
    pinning.reset()


def _granted():
    return pinning._state["granted"]


def test_a_released_arena_returns_its_grant_when_its_copies_go(budget):
    arena = pinning.PinnedArena(8 << 20, release_on_free=True)
    copies = [arena.pin(torch.ones(1 << 20, dtype=torch.float32)) for _ in range(2)]
    assert _granted() > 0
    del copies, arena
    gc.collect()
    assert _granted() == 0


def test_the_block_store_arenas_release_on_free():
    import inspect

    from poros.core import residency

    src = inspect.getsource(residency.BlockManager)
    assert src.count("release_on_free=True") == 2  # offload_all_to_cpu and _offload_block


def test_close_frees_the_store_and_a_with_block_closes(monkeypatch):
    from types import SimpleNamespace

    from poros.train.trainer import PorosTrainer

    t = PorosTrainer.__new__(PorosTrainer)
    store = {"layer.0": torch.ones(4)}
    t.block_manager = SimpleNamespace(_cpu_weight_store=store)
    t._boundary_offload, t.optimizer, t._lr_scheduler, t.model = object(), object(), object(), 1
    t._peft_model, t._is_setup = object(), True
    with t as same:
        assert same is t
    assert store == {} and t.block_manager is None and t._boundary_offload is None
    assert t.model is None and t.optimizer is None and not t._is_setup
