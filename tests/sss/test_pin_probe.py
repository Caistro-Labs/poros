"""The pin budget's probe maps private anonymous memory where the platform has MAP_PRIVATE, and
the plain anonymous mapping where it has not (native Windows, whose mmap takes no flags)."""
from __future__ import annotations

import mmap

import pytest

torch = pytest.importorskip("torch")

from poros.sss import pinning  # noqa: E402


class _Rt:
    def __init__(self):
        self.calls = []

    def cudaHostRegister(self, ptr, n, flags):  # noqa: N802 - CUDA's name
        self.calls.append(("register", n))
        return 0

    def cudaHostUnregister(self, ptr):  # noqa: N802
        self.calls.append(("unregister",))
        return 0


@pytest.fixture
def rt(monkeypatch):
    r = _Rt()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: r)
    return r


def test_where_there_is_map_private_the_probe_maps_private_memory(rt, monkeypatch):
    seen = []
    real = mmap.mmap
    monkeypatch.setattr(mmap, "mmap", lambda *a, **k: seen.append(k) or real(*a, **k))
    assert pinning._register_ok(1 << 20) is True
    assert seen == [{"flags": mmap.MAP_PRIVATE}] and rt.calls == [("register", 1 << 20),
                                                                  ("unregister",)]


def test_without_map_private_the_probe_maps_as_windows_does(rt, monkeypatch):
    real = mmap.mmap

    def windows_mmap(fileno, length, tagname=None, access=None, offset=0):  # no flags argument
        return real(fileno, length)

    monkeypatch.delattr(mmap, "MAP_PRIVATE", raising=False)
    monkeypatch.setattr(mmap, "mmap", windows_mmap)
    assert pinning._register_ok(1 << 20) is True
    assert rt.calls == [("register", 1 << 20), ("unregister",)]
