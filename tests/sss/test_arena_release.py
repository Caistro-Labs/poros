"""An arena made with ``release_on_free`` (the offloaded LM head's) gives its grants back to the
process's pin budget once what it pinned is freed: a second trainer in one process (a notebook, a
bench driver) gets the same budget the first did. CPU: the slabs are ordinary tensors here."""

from __future__ import annotations

import gc

import pytest

from poros.sss import pinning

torch = pytest.importorskip("torch")


@pytest.fixture
def ledger(monkeypatch):
    """reserve/release counted, slabs and lone pins made of ordinary memory. Garbage an earlier
    test left (a trainer's head arena in a reference cycle, on a GPU run of the whole folder) is
    collected first, so its grants go back to the real budget and not into this count."""
    gc.collect()
    held = {"granted": 0}

    def reserve(n):
        held["granted"] += n
        return True

    def release(n):
        held["granted"] -= n

    monkeypatch.setattr(pinning, "reserve", reserve)
    monkeypatch.setattr(pinning, "release", release)
    monkeypatch.setattr(pinning.PinnedArena, "_new_slab",
                        lambda self, n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self.clone())
    return held


@pytest.mark.parametrize("on_free", [True, False])
def test_the_grants_return_when_the_copies_are_freed(ledger, on_free):
    arena = pinning.PinnedArena(5 << 20, release_on_free=on_free)
    arena.SLAB = 4 << 20  # two chunks share a slab; the third is over half of one: alone
    chunks = [arena.pin(torch.ones(1 << 18)), arena.pin(torch.ones(1 << 18)),
              arena.pin(torch.ones(3 << 20, dtype=torch.uint8))]
    assert ledger["granted"] == (4 << 20) + (4 << 20)  # the slab, and the lone one's power of 2
    assert all(c.sum() > 0 for c in chunks)
    granted = ledger["granted"]
    del arena
    gc.collect()
    assert ledger["granted"] == granted  # the chunks still hold their slab: nothing returns yet
    del chunks
    gc.collect()
    assert ledger["granted"] == (0 if on_free else granted)


def test_a_view_of_a_copy_keeps_its_grant_until_the_memory_goes(ledger):
    """The grant follows the memory: a view of a copy (or a tensor its memory moved into) keeps
    the slab's pages, and so its grant, after the copy itself is gone."""
    arena = pinning.PinnedArena(1 << 20, release_on_free=True)
    arena.SLAB = 1 << 20
    copy = arena.pin(torch.ones(1 << 16))
    view = copy[8:16].view(2, 4)
    granted = ledger["granted"]
    del arena, copy
    gc.collect()
    assert ledger["granted"] == granted and view.sum() == 8
    del view
    gc.collect()
    assert ledger["granted"] == 0


def test_the_head_arena_releases_on_free():
    import inspect

    from poros.train import head

    assert "release_on_free=True" in inspect.getsource(head.DeviceHead.pinned_chunks)
