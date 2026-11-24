"""Back-to-back pinned expert slabs are copied as one run only when they lie in one storage: two
storages can sit next to each other in virtual memory, and a run is viewed through its first
slab's storage (CPU: the slabs are stand-ins)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from poros.moe.residency import _TransferEngine  # noqa: E402


class _Storage:
    def __init__(self, base):
        self.base = base

    def data_ptr(self):
        return self.base


class _Slab:
    """A pinned uint8 slab: ``n`` bytes at ``off`` in ``storage``."""

    def __init__(self, storage, off, n):
        self.storage, self.off, self.n = storage, off, n

    def is_pinned(self):
        return True

    def data_ptr(self):
        return self.storage.base + self.off

    def numel(self):
        return self.n

    def untyped_storage(self):
        return self.storage


class _Source:
    def __init__(self, slabs):
        self.slabs = slabs

    def _slab(self, key):
        return self.slabs[key]


def test_slabs_back_to_back_in_one_storage_are_one_run():
    st = _Storage(4096)
    src = _Source({"a": _Slab(st, 0, 64), "b": _Slab(st, 64, 64), "c": _Slab(st, 256, 64)})
    runs = _TransferEngine._host_runs(None, src, ["a", "b", "c"])
    assert [[k for k, _ in run] for run in runs] == [["a", "b"], ["c"]]


def test_an_adjacent_slab_in_another_storage_starts_a_new_run():
    first, second = _Storage(4096), _Storage(4096 + 128)  # mapped right after a full chunk
    src = _Source({"a": _Slab(first, 0, 64), "b": _Slab(first, 64, 64),
                   "c": _Slab(second, 0, 64), "d": _Slab(second, 64, 64)})
    runs = _TransferEngine._host_runs(None, src, ["a", "b", "c", "d"])
    assert [[k for k, _ in run] for run in runs] == [["a", "b"], ["c", "d"]]
