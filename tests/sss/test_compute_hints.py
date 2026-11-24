"""The fused expert compute tells a streaming slab source (the SSD store) which experts a layer's
router picked before it makes them resident, on the source's first pass (no schedule yet), so
the reads go out together instead of one by one inside ``ensure``; a source replaying its schedule
is not told. A host-RAM source is never hinted (it has nothing to read ahead)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from poros.moe import fused  # noqa: E402
from poros.sss.source import RamSource  # noqa: E402


class _Source:
    recording = True  # no schedule yet: the first pass

    def __init__(self, calls):
        self.calls = calls

    def prefetch(self, keys):
        self.calls.append(("prefetch", list(keys)))


class _Residency:
    _look_ahead = 0

    def __init__(self, calls):
        self.calls, self._resident = calls, {(3, 1): object()}

    def ensure(self, keys):
        self.calls.append(("ensure", list(keys)))


def _block(source, calls):
    return SimpleNamespace(layer_idx=3, num_experts=8, residency=_Residency(calls),
                           store=SimpleNamespace(source=source))


def test_the_fused_compute_hints_a_streaming_source_before_ensure():
    calls: list = []
    block = _block(_Source(calls), calls)
    fused._ensure(block, SimpleNamespace(hit=[0, 1, 5]))
    # the resident expert (3, 1) is not hinted; every hit expert is ensured, after the hint
    assert calls == [("prefetch", [(3, 0), (3, 5)]), ("ensure", [(3, 0), (3, 1), (3, 5)])]


def test_a_source_replaying_its_schedule_is_not_hinted():
    """Once the store replays a schedule its reads are already in flight: no hint (its
    speculative reads cost a disk-bound pass more than they save)."""
    calls: list = []
    source = _Source(calls)
    source.recording = False
    fused._ensure(_block(source, calls), SimpleNamespace(hit=[0, 5]))
    assert calls == [("ensure", [(3, 0), (3, 5)])]


def test_a_host_ram_source_is_not_hinted(monkeypatch):
    calls: list = []
    ram = RamSource.__new__(RamSource)
    monkeypatch.setattr(RamSource, "prefetch", lambda self, keys: calls.append("prefetch"),
                        raising=False)
    block = _block(ram, calls)
    fused._ensure(block, SimpleNamespace(hit=[2]))
    assert calls == [("ensure", [(3, 2)])]
