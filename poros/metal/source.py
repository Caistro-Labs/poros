"""Where a layer's packed bytes come from. The block loop cannot tell which source it has.

``acquire`` returns the layer's bytes on the device and waits for the device's queued work, so every
source syncs once per layer visit, and that wait also drops what uncommitted command buffers hold.

``FileLayers`` is the Mac's SSD store, and reports as CUDA's does (``poros.sss.ssd.SsdSource``):
``stats`` holds the same counters and ``mark_step`` ends a pass, so the trainer records a Mac SSD
run's reads in its step events and its setup record the way it records a CUDA one's.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from poros.sss.format import SssFile, SyncFileSource


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


class RamLayers:
    """Every layer on the device, filled once. Nothing moves after that."""

    def __init__(self, path: str | os.PathLike, device: torch.device):
        self.device, self.slabs = device, {}
        with SssFile.open(path) as file:
            for entry in file.entries:
                if entry.kind == "layer":
                    slab = torch.empty(entry.nbytes, dtype=torch.uint8)
                    file.read_into(entry.key, slab, verify=True)
                    self.slabs[entry.key[0]] = slab.to(device)

    def acquire(self, layer: int) -> torch.Tensor:
        sync(self.device)
        return self.slabs[layer]

    def prefetch(self, layer: int) -> None:
        pass


class FileLayers:
    """One slot on the device, and one landing buffer a reader thread fills ahead of it.

    The reader touches CPU memory only and the main thread does the blit, so no ``torch.mps.Event``
    crosses threads. The blit blocks, so a read never overwrites a landing still being copied.
    Before it waits on the reader, ``acquire`` commits the work already queued: left queued, MPS
    starts it only at the blit, after the read, and the read and the compute add up.

    Counted as ``SsdSource`` counts (``stats``): a layer already in the slot is no read (the
    device's own cache, which a source never sees); a read the block loop asked ahead for is a
    prefetch hit, one it had to start when it asked for the layer a sync read, and one read ahead
    that the loop never asked for (an evaluation's off-schedule visit) a dropped prefetch;
    ``blocked_s`` is the time ``acquire`` waited on the reader.
    """

    def __init__(self, path: str | os.PathLike, device: torch.device):
        self.source = SyncFileSource(path)  # verifies each slab on its first read in this process
        self.path, self.device = path, device
        size = max(e.nbytes for e in self.source.file.entries if e.kind == "layer")
        self.slot = torch.empty(size, dtype=torch.uint8, device=device)
        landing = torch.empty(size, dtype=torch.uint8)
        self.landing = landing.pin_memory() if device.type == "mps" else landing
        self.reader = ThreadPoolExecutor(1, thread_name_prefix="poros-read")
        self.reading = self.in_slot = self.landed = None
        self._stats = dict(reads=0, prefetch_hits=0, sync_reads=0, prefetch_dropped=0,
                           bytes_read=0, blocked_s=0.0, steps=0, cache_hits=0, cache_bytes=0)

    def _warm(self, layer: int) -> bool:
        return layer == self.in_slot

    def prefetch(self, layer: int) -> None:
        if layer == self.landed or self._warm(layer):
            return
        if self.landed is not None:  # read ahead for a layer the loop did not ask for next
            self._stats["prefetch_dropped"] += 1
        view = self.landing[: self.source.nbytes((layer, 0))]
        self.reading = self.reader.submit(self.source.read_into, (layer, 0), view)
        self.landed = layer

    def acquire(self, layer: int) -> torch.Tensor:
        slab = self.slot[: self.source.nbytes((layer, 0))]
        if self._warm(layer):
            sync(self.device)
            return slab
        ahead = layer == self.landed
        self.prefetch(layer)
        sync(self.device)  # what is already queued runs on the device while this read lands
        t0 = time.perf_counter()
        self.reading.result()
        self._stats["blocked_s"] += time.perf_counter() - t0
        slab.copy_(self.landing[: slab.numel()])
        self.in_slot, self.landed = layer, None
        self._stats["reads"] += 1
        self._stats["bytes_read"] += slab.numel()
        self._stats["prefetch_hits" if ahead else "sync_reads"] += 1
        return slab

    def mark_step(self) -> None:
        """One forward and backward pass ended (the block loop's schedule is the manager's)."""
        self._stats["steps"] += 1

    @property
    def stats(self) -> dict:
        """``SsdSource.stats``' counters and setup, as they are here: one layer read ahead
        (``depth``), buffered reads on one reader thread into one landing (``ring_bytes``),
        nothing page-locked for CUDA, no host cache."""
        return {**self._stats, "depth": 1, "direct": False, "pinned": False, "pinned_bytes": 0,
                "chunk_bytes": self.landing.numel(), "ring_bytes": self.landing.numel(),
                "cache": False, "trace_len": 0, "recording": False, "compute_s_per_read": None,
                "fallback_reason": None}
