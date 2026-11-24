"""Block-boundary activations in pinned host RAM during NF4 block-major accumulation.

Block-major keeps every block's input hidden state for all G microsteps until the
backward reaches that block: G x num_blocks x T x hidden values (72B, bs1, G=8,
T=512: 80 x 8 x 512 x 8192 x 2 B = 5.4 GB). They are written once in the
forward and read once, in reverse block order, in the backward -- so they can
live in pinned host RAM: copied down on a side stream after the forward
consumes them, fetched back one block ahead on another side stream (not the
weight-prefetch stream, so the compute stream never waits on a weight copy to
get a boundary). Exact copies: no value changes.

Host buffers are carved from page-locked slabs (``POROS_BOUNDARY_PIN`` or the
process pin budget, ``poros.sss.pinning``), reused across updates. Once the host
refuses to page-lock one (the budget says no, or the allocation fails: a WSL2
node may lock ~1 GiB or nothing), further slabs come from ordinary RAM, pooled
the same way, up to ``POROS_BOUNDARY_HOST_MAX`` (default: half the RAM free at
that moment): the same bytes, with copies that block the host. Slabs reserved
before the weights (:meth:`BoundaryOffload.prealloc`) are a head start: past
them the pool grows the same way. Past the RAM cap, or once a
``POROS_BOUNDARY_PIN`` allowance is used up, the boundary simply stays on the
device (same values, no saving for that one): a full host never fails the run.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class _Host:
    """A boundary parked in a pooled host buffer (pinned, or ordinary RAM if pinning is refused)."""

    __slots__ = ("buf", "shape", "dtype", "nbytes")

    def __init__(self, buf: torch.Tensor, shape: torch.Size, dtype: torch.dtype,
                 nbytes: int) -> None:
        self.buf, self.shape, self.dtype, self.nbytes = buf, shape, dtype, nbytes

    def view(self) -> torch.Tensor:
        return self.buf[: self.nbytes].view(self.dtype).view(self.shape)


Parked = Union[_Host, torch.Tensor]


SLAB_BYTES = 256 << 20  # a power of two: torch's host allocator locks exactly this much
_ALIGN = 512


def _boundary_allowance() -> Optional[int]:
    """``POROS_BOUNDARY_PIN`` (e.g. ``4G``): page-locked bytes for boundaries, granted apart
    from ``POROS_PIN_BUDGET`` (which the weight store fills first). Unset: slabs ask the
    process budget like any other pin."""
    import os

    raw = os.environ.get("POROS_BOUNDARY_PIN")
    if raw is None or raw.strip() == "":
        return None
    from poros.core.config import parse_bytes

    return parse_bytes(raw)


def _pageable_cap() -> int:
    """Ordinary-RAM bytes for boundaries the host would not page-lock: ``POROS_BOUNDARY_HOST_MAX``
    (e.g. ``9.5G``; ``0`` keeps them on the device), else half the host RAM free now
    (MemAvailable within the cgroup's limits, ``poros.core.env.host_ram_free``; 0 if
    unreadable)."""
    import os

    raw = os.environ.get("POROS_BOUNDARY_HOST_MAX")
    if raw is not None and raw.strip() != "":
        from poros.core.config import parse_bytes

        return parse_bytes(raw)
    from poros.core.env import host_ram_free

    free_gib = host_ram_free()
    return int(free_gib * 2**30) // 2 if free_gib else 0


class BoundaryOffload:
    """D2H / H2D copies of boundary hidden states with stream-safe lifetimes.

    Host memory: 256 MiB pinned slabs, carved with a bump pointer that is reset
    at the start of every update (:meth:`begin_update`) -- every boundary of
    the previous update has been fetched back by then -- so locked RAM is the
    largest update's boundary bytes rounded to a slab, not a power of two each.
    Ordinary-RAM slabs (pinning refused) take the same copies on the same
    streams, so the same stream order covers them (the host never touches slab
    bytes itself); CUDA only makes those copies block the host.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.d2h = torch.cuda.Stream(device=device)
        self.h2d = torch.cuda.Stream(device=device)
        self._slabs: List[torch.Tensor] = []
        self._slab = 0      # current slab index
        self._off = 0       # bump offset in it
        self._host_full = False
        self._allowance = _boundary_allowance()
        self._pageable_max: Optional[int] = None  # sized at the first refused pin
        self.stats = {"offloaded": 0, "kept_on_device": 0, "slabs": 0, "slab_bytes": 0,
                      "bytes_down": 0, "host_full": False, "pageable_slabs": 0}

    def prealloc(self, nbytes: int) -> int:
        """Page-lock up to ``nbytes`` of whole slabs now, before anything else pins; returns
        the bytes locked. The caller owns their pin-budget grant. A head start, not a cap:
        boundaries past them take slabs as they would without it (pin budget, then RAM)."""
        while self.stats["slab_bytes"] + SLAB_BYTES <= nbytes:
            try:
                slab = torch.empty(SLAB_BYTES, dtype=torch.uint8, pin_memory=True)
            except RuntimeError:  # the host's page-locked ceiling
                self._host_full = self.stats["host_full"] = True
                break
            self._slabs.append(slab)
            self.stats["slabs"] += 1
            self.stats["slab_bytes"] += SLAB_BYTES
        return self.stats["slab_bytes"]

    def begin_update(self) -> None:
        self._slab, self._off = 0, 0


    def _new_slab(self) -> bool:
        from poros.sss import pinning

        if self._host_full:
            return self._new_pageable_slab()
        if self._allowance is not None:
            if self.stats["slab_bytes"] + SLAB_BYTES > self._allowance:
                return False
        elif not pinning.reserve(SLAB_BYTES):
            return self._new_pageable_slab()
        try:
            slab = torch.empty(SLAB_BYTES, dtype=torch.uint8, pin_memory=True)
        except RuntimeError:  # the host's page-locked ceiling: stop asking
            if self._allowance is None:
                pinning.release(SLAB_BYTES)
            self._host_full = self.stats["host_full"] = True
            return self._new_pageable_slab()
        if self._allowance is None:  # granted by the budget: given back when the slab is freed
            import weakref

            weakref.finalize(slab.untyped_storage(), pinning._freed, SLAB_BYTES)
        self._slabs.append(slab)
        self.stats["slabs"] += 1
        self.stats["slab_bytes"] += SLAB_BYTES
        return True

    def _new_pageable_slab(self) -> bool:
        """A slab of ordinary RAM where the host refused to page-lock one, pooled and reused
        like the pinned slabs (a fresh host tensor per boundary grew the heap ~12 GiB in 20
        updates of a 72B). Without it the boundaries stay on the GPU, which a 72B on 24 GB
        cannot hold. Copies to and from it block the host; the bytes are the same."""
        if self._pageable_max is None:  # the first refusal: size the pool, say so once
            self._pageable_max = _pageable_cap()
            logger.warning(
                "path efficient: page-locked memory refused (%s); block inputs go to ordinary "
                "RAM slabs instead (same bytes, copies that block the host), up to %.2f GiB "
                "(POROS_BOUNDARY_HOST_MAX, default half the free RAM); past that they stay on "
                "the GPU", "page-locked ceiling" if self._host_full else "pin budget",
                self._pageable_max / 2**30)
        if (self.stats["pageable_slabs"] + 1) * SLAB_BYTES > self._pageable_max:
            return False
        try:
            slab = torch.empty(SLAB_BYTES, dtype=torch.uint8)
        except RuntimeError:  # no RAM for it after all: stop asking
            self._pageable_max = 0
            return False
        self._slabs.append(slab)
        self.stats["pageable_slabs"] += 1
        return True

    def _buffer(self, nbytes: int) -> Optional[torch.Tensor]:
        if nbytes > SLAB_BYTES:
            return None
        while True:
            if self._slab < len(self._slabs):
                if self._off + nbytes <= SLAB_BYTES:
                    buf = self._slabs[self._slab][self._off : self._off + nbytes]
                    self._off += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
                    return buf
                self._slab, self._off = self._slab + 1, 0
                continue
            if not self._new_slab():
                return None

    def offload(self, t: torch.Tensor) -> Parked:
        """Start copying ``t`` (made on the current stream) into a pooled host slab.

        Returns the parked boundary (or ``t`` itself when no host buffer is
        granted). The caller drops its own reference to ``t``; ``record_stream``
        keeps the device block alive until the copy has run.
        """
        nbytes = t.numel() * t.element_size()
        buf = self._buffer(nbytes) if t.is_contiguous() else None
        if buf is None:
            self.stats["kept_on_device"] += 1
            return t
        host = _Host(buf, t.shape, t.dtype, nbytes)
        cur = torch.cuda.current_stream(self.device)
        self.d2h.wait_stream(cur)
        self.d2h.wait_stream(self.h2d)  # a reused buffer's previous fetch has been read
        with torch.cuda.stream(self.d2h):
            host.view().copy_(t, non_blocking=True)
        t.record_stream(self.d2h)
        self.stats["offloaded"] += 1
        self.stats["bytes_down"] += nbytes
        return host

    def fetch(self, parked: Parked) -> torch.Tensor:
        """Start copying a parked boundary back; pass the result to :meth:`ready` before use."""
        if isinstance(parked, torch.Tensor):  # never left the device
            return parked
        self.h2d.wait_stream(self.d2h)  # the host copy must have landed
        with torch.cuda.stream(self.h2d):
            t = parked.view().to(self.device, non_blocking=True)
        # The slab bytes are rewritten only next update, after d2h waits on h2d.
        return t

    def ready(self, t: torch.Tensor) -> torch.Tensor:
        """Make a fetched tensor safe to use (and free) on the current stream."""
        cur = torch.cuda.current_stream(self.device)
        cur.wait_stream(self.h2d)
        t.record_stream(cur)
        return t


def make_boundary_offload(enabled: bool,
                          device: Optional[torch.device]) -> Optional[BoundaryOffload]:
    if not enabled or device is None or torch.device(device).type != "cuda":
        return None
    return BoundaryOffload(torch.device(device))
