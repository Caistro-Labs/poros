"""``SsdSource``: a ``.sss`` file streamed from an SSD through a pinned chunk ring.

How a slab gets from disk to ``dst``:

1. **Reader.** A thread pool issues ``os.preadv`` calls (the GIL is released
   while they run) on an ``O_DIRECT`` descriptor, one per fixed-size chunk,
   into a ring of page-aligned host chunks. The ring is one anonymous
   ``mmap`` registered with CUDA (``cudaHostRegister``), so it is pinned and
   sized exactly, without the caching host allocator's power-of-two rounding.
2. **Copy.** ``read_into`` copies each chunk to ``dst[off:off+len]`` on the
   caller's current stream (``non_blocking``) and records one CUDA event; the
   chunk's ring slot is reused only after that event completes.
3. **Schedule replay.** Poros schedules are static, so the reads of each
   pass (everything up to a ``mark_step``) are recorded as the trace for the
   next one. From the second pass on every read resynchronizes a pointer
   into the trace and keeps
   the next ``depth`` keys of the trace in flight. Keys a pass does not need
   (slabs the VRAM cache already holds, experts no token routed to) are
   passed over; a far jump (a pass starting after cached layers) is taken
   once the following read confirms it. A key that is not in the trace, or
   was not prefetched, is read synchronously: slower, the same bytes.
4. **Fallback.** Where ``O_DIRECT`` is refused (tmpfs, overlay roots) the
   reader uses buffered ``preadv`` plus ``posix_fadvise(DONTNEED)`` so the
   page cache does not fill with the model, and ``direct`` is ``False``.
5. **Host cache** (``cache_bytes``, off by default). The ring grows by that
   many bytes and keeps what it read: a slot whose copy has landed keeps its
   slab until a new read needs the slot (least recently used first), so a slab
   asked for again by then -- the backward's first layers, which the forward
   read last, and the next update's first, which the backward read last -- is
   served from RAM with no disk read. A key the schedule or a hint wants next
   is claimed from the cache instead of read again. On a CUDA host the cache's
   slots are page-locked on a thread of their own from the first read on and
   join the ring piece by piece (``_pin_cache``; no read waits for them): until
   they have, a slab the ring gives up is read from disk again, the same bytes.

Bytes are never interpreted: every slab is checksum-verified against the
value recorded at write time on its first read in this process
(``verify="first"``), as ``SyncFileSource`` does, so SSD and RAM sources
hold the same bytes for every key or the read raises ``SssCorruptError``.
A slab that lands in one ring chunk (every MoE expert slab at the default
8 MiB chunk) is checksummed by the reader thread that read it, over the ring
bytes the copy then moves, so a first pass hashes on every reader thread at
once instead of on the training thread; the check, and the bytes checked,
are the same.
"""

from __future__ import annotations

import bisect
import errno
import hashlib
import logging
import mmap
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from poros.sss.format import ALIGNMENT, SssCorruptError, SssFile
from poros.sss.source import SlabKey, check_dst

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_CHUNK_BYTES", "DEFAULT_RAM_BUDGET", "SsdSource"]

DEFAULT_CHUNK_BYTES = 8 << 20
DEFAULT_RAM_BUDGET = 4 << 30
DEFAULT_DEPTH = 8
_MAX_TRACE = 1 << 22  # a run that never calls mark_step stops recording here
_CONFIRM = 3  # consecutive in-order reads that confirm a far jump in the trace
_CACHE_PIECE = 256 << 20  # the host cache is page-locked this much at a time, off-thread


_DONE: Future = Future()  # a read served from the host cache: nothing in flight
_DONE.set_result(None)


def _align(n: int) -> int:
    return -(-n // ALIGNMENT) * ALIGNMENT


def _runs(trace: list) -> list[list]:
    """The trace cut into phases: maximal runs of one layer's keys."""
    out: list[list] = []
    for k in trace:
        if out and out[-1][0][0] == k[0]:
            out[-1].append(k)
        else:
            out.append([k])
    return out


def _pread_full(fd: int, buf: memoryview, offset: int, drop_cache: bool) -> None:
    """Fill ``buf`` from ``fd`` at ``offset`` (runs in a reader thread)."""
    total, pos = len(buf), offset
    while buf:
        n = os.preadv(fd, [buf], pos)
        if n <= 0:
            raise OSError(errno.EIO, f"short read at byte {pos}")
        buf, pos = buf[n:], pos + n
    if drop_cache:
        os.posix_fadvise(fd, offset, total, os.POSIX_FADV_DONTNEED)


@dataclass
class _Read:
    """One slab's chunks in flight or landed in the ring."""

    key: SlabKey
    pos: int | None  # trace position it was prefetched for (None: on demand)
    slots: list[int]
    futures: list[Future]
    hint: int | None = None  # the prefetch() call that asked for it
    started: float = field(default_factory=time.perf_counter)
    hashed: bool = False  # its one future returns the slab's checksum (_read_hashed)
    cached: bool = False  # served by the host cache: a checked slab already in its slots


class SsdSource:
    """A ``SlabSource`` over a committed ``.sss`` file on a local SSD.

    Args:
        path: The ``.sss`` data file (its ``.idx`` sits next to it).
        chunk_bytes: Ring chunk size (rounded up to 4 KiB). Slabs smaller
            than a chunk use a chunk sized to the largest slab instead.
        ram_budget: Cap on the pinned ring (``--sss-ram``); at least two
            chunks are always allocated.
        threads: Reader threads (each holds one ``preadv`` in flight).
        depth: Keys kept in flight once the trace is recorded. ``None``
            autotunes it at the first ``mark_step`` from the measured
            compute time per read and ``bandwidth`` (or, without it, the
            bandwidth of the step's synchronous reads).
        max_depth: Upper clamp for the autotuned depth.
        bandwidth: Effective bytes/s, ``min(B_ssd, B_h2d)`` from
            ``poros.sss.probe``; only used to autotune ``depth``.
        latency_bandwidth: The probe's one-read-in-flight bandwidth (the
            per-read latency the depth hides); defaults to ``bandwidth``.
        saturate: The probe's queue depth: the least depth when reads cannot
            keep up with compute (disk-bound).
        verify: ``"first"`` (default) checksums each slab on its first read
            in this process, ``"always"`` on every read, ``"never"`` never.
        direct: ``"auto"`` tries ``O_DIRECT`` and falls back to buffered
            reads; ``True`` refuses to fall back; ``False`` is buffered.
        pin: Register the ring with CUDA (when CUDA is available) so copies
            are asynchronous.
        cache_bytes: Host RAM, beyond the ring, that keeps slabs once read
            (step 5 of the module doc); 0 (default): none, the ring alone.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        ram_budget: int = DEFAULT_RAM_BUDGET,
        threads: int = 8,
        depth: int | None = None,
        max_depth: int = 64,
        bandwidth: float | None = None,
        latency_bandwidth: float | None = None,
        saturate: int | None = None,
        verify: str = "first",
        direct: bool | str = "auto",
        pin: bool = True,
        cache_bytes: int = 0,
    ) -> None:
        if verify not in ("first", "always", "never"):
            raise ValueError(f"verify must be 'first', 'always' or 'never', got {verify!r}")
        if direct not in ("auto", True, False):
            raise ValueError(f"direct must be 'auto', True or False, got {direct!r}")
        if threads < 1 or max_depth < 1 or (depth is not None and depth < 0):
            raise ValueError("threads and max_depth must be >= 1 and depth >= 0")
        self.path = Path(path)
        self._file = self._open_index(path)
        self._entries = {e.key: e for e in self._file.entries}
        self._keys = tuple(e.key for e in self._file.entries)
        self._verify = verify
        self._verified: set[SlabKey] = set()
        self._closed = False

        largest = max((_align(self._ring_nbytes(e)) for e in self._file.entries),
                      default=ALIGNMENT)
        self.chunk_bytes = min(_align(max(int(chunk_bytes), 1)), largest)
        self._chunks_max = -(-largest // self.chunk_bytes)
        self.max_depth = int(max_depth)
        budget_slots = max(2, int(ram_budget) // self.chunk_bytes)
        # Room for the prefetch window, passed-over reads, and two layers' experts (the
        # hinted phase and the one read ahead): (4 * max_depth + 8) slabs, 264 by default.
        self.num_slots = max(2, min(budget_slots, (4 * self.max_depth + 8) * self._chunks_max))
        self._prefetchable = max(1, self.num_slots // 2)  # chunks; bigger slabs stream
        self._retain = int(cache_bytes) > 0  # slots keep their slab once its copy landed
        stream_slots = self.num_slots
        # no more than every slab, one per slot: a cache larger than the file holds nothing more
        whole = sum(self._nchunks(self._ring_nbytes(e)) for e in self._file.entries)
        cache_slots = min(max(0, int(cache_bytes)) // self.chunk_bytes, whole)
        self.num_slots += cache_slots

        self.pinned = False
        self._fd = -1
        self.direct, self.fallback_reason = self._open_data(direct)
        # private anonymous memory, faulted in by torch's threads before it is registered: a
        # shared mapping (mmap's default) is faulted in by the driver a page at a time
        private = getattr(mmap, "MAP_PRIVATE", None)
        size = self.num_slots * self.chunk_bytes
        self._mm = mmap.mmap(-1, size) if private is None else mmap.mmap(-1, size, flags=private)
        self._mv = memoryview(self._mm)
        self._ring = torch.frombuffer(self._mm, dtype=torch.uint8)
        self._cache_lock = threading.Lock()
        self._cache_new: list[int] = []  # cache slots pinned off-thread, not yet free
        self._cache_pins: list[tuple[int, int]] = []  # (address, bytes) the cache registered
        self._closing = threading.Event()
        self._cache_thread: threading.Thread | None = None
        # Registered last: nothing after it can fail and leave a registered
        # range behind whose memory is unmapped (the next pin would collide).
        # Under a page-locked ceiling the ring pins its first slots only.
        self.pinned_bytes = 0
        if pin and torch.cuda.is_available():
            from poros.sss import pinning

            want = stream_slots * self.chunk_bytes
            floor = min(want, 2 * self._chunks_max * self.chunk_bytes)
            while want >= floor and not pinning.reserve(want):
                want //= 2 * self.chunk_bytes
                want *= self.chunk_bytes
            if want >= floor and want:
                try:
                    self._ring[:want].fill_(0)
                    torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(
                        self._ring.data_ptr(), want, 0))
                    self.pinned, self.pinned_bytes = True, want
                except Exception as e:  # pragma: no cover - depends on the driver
                    pinning.release(want)
                    logger.warning("SSS ring could not be pinned (%s); copies will be "
                                   "synchronous", e)
                else:
                    if want < stream_slots * self.chunk_bytes:  # only the pinned slots are used
                        stream_slots = self.num_slots = want // self.chunk_bytes
                        cache_slots = 0  # a page-locked ceiling: no room for a cache either
                        self._retain = False
                        self._prefetchable = max(1, self.num_slots // 2)
                        logger.warning(
                            "SSS ring cut to %d MiB of pinned memory (a page-locked "
                            "ceiling on this host)", want >> 20)
            else:
                logger.warning("SSS ring left unpinned: this host's page-locked memory "
                               "budget is exhausted; copies will be synchronous")
        self._pool = ThreadPoolExecutor(max_workers=int(threads),
                                        thread_name_prefix="poros-sss-read")
        # free slots, least recently released first; a free slot may still hold a slab (_held)
        self._free: OrderedDict[int, None] = OrderedDict.fromkeys(range(stream_slots))
        # the host cache's slots, page-locked off the training thread a piece at a time from the
        # first read on (not during the load, whose page faults its registrations would stall)
        self._cache_due = (stream_slots, cache_slots) if cache_slots and self.pinned else None
        if cache_slots and not self.pinned:  # nothing to pin (no CUDA, or pinning off): free now
            self._free.update(OrderedDict.fromkeys(range(stream_slots, self.num_slots)))
        self._held: dict[SlabKey, list[int]] = {}  # key -> the free slots that hold it
        self._owner: dict[int, SlabKey] = {}  # held slot -> its key
        # copies out of the ring still running: their event, slots, and the key they hold
        self._copies: deque[tuple[torch.cuda.Event, list[int], SlabKey | None]] = deque()
        self._draining: list[_Read] = []
        self._inflight: dict[SlabKey, _Read] = {}

        # Schedule replay.
        self._depth_fixed = depth is not None
        self.depth = int(depth) if depth is not None else DEFAULT_DEPTH
        self.depth = min(self.depth, self.max_depth)
        self._bandwidth = bandwidth
        self._latency_bandwidth = latency_bandwidth
        self._saturate = saturate
        self._recording = True  # no trace yet
        self._trace: list[SlabKey] = []
        self._pass: list[SlabKey] = []  # this pass's reads: the next trace
        self._pos_of: dict[SlabKey, list[int]] = {}
        self._ptr = -1
        self._win = -1  # last trace position the window has considered
        self._horizon = 0  # how far ahead a read stays wanted (two phases, set per trace)
        self._pending: tuple[int, int] | None = None  # unconfirmed far match, its length

        # Per-step timing (for autotune) and counters.
        self._step_t0: float | None = None
        self._step_blocked = 0.0
        self._step_reads = 0
        self._step_sync_bytes = 0
        self.compute_s_per_read: float | None = None
        self._stats = dict(reads=0, prefetch_hits=0, sync_reads=0, prefetch_dropped=0,
                           bytes_read=0, blocked_s=0.0, verified=0, steps=0, hinted=0,
                           phase_ahead=0, cache_hits=0, cache_bytes=0)
        self._hints = 0
        self._hint_layer: int | None = None
        self._hint_keys: set = set()

    # -- setup -----------------------------------------------------------------

    def _open_index(self, path: str | os.PathLike):
        """The slab index: ``.entries`` (``key``, ``nbytes``, ``offset``, ``checksum``),
        ``.entry(key)`` and ``.close()``. A subclass reading another layout overrides
        this, ``_open_data``, ``_read_chunk`` and ``_close_data``."""
        return SssFile.open(path)

    def _ring_nbytes(self, e) -> int:
        """Ring bytes one read of ``e`` takes (a subclass may land more than the slab: raw
        file ranges it assembles on the copy out)."""
        return e.nbytes

    def _read_chunk(self, e, start: int, buf: memoryview) -> None:
        """Fill ``buf`` with slab ``e``'s bytes from ``start`` (runs in a reader thread;
        ``start`` and ``len(buf)`` are 4 KiB multiples, ``buf`` is page-aligned)."""
        _pread_full(self._fd, buf, e.offset + start, not self.direct)

    def _close_data(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._file.close()

    def _open_data(self, direct: bool | str) -> tuple[bool, str | None]:
        """Open the data descriptor: O_DIRECT if the filesystem allows it."""
        reason = None
        if direct and hasattr(os, "O_DIRECT"):
            try:
                fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            except OSError as e:
                reason = f"O_DIRECT open refused: {e.strerror or e}"
            else:
                try:  # some filesystems accept the flag and refuse the read
                    _pread_full(fd, memoryview(mmap.mmap(-1, ALIGNMENT)), 0, drop_cache=False)
                except OSError as e:
                    os.close(fd)
                    reason = f"O_DIRECT read refused: {e.strerror or e}"
                else:
                    self._fd = fd
                    return True, None
        elif direct:
            reason = "O_DIRECT is not available on this platform"
        if direct is True:
            raise OSError(f"{self.path}: {reason}")
        if reason is not None:
            logger.warning("SSS %s: %s; using buffered reads + posix_fadvise(DONTNEED)",
                           self.path, reason)
        self._fd = os.open(self.path, os.O_RDONLY)
        return False, reason

    # -- SlabSource ------------------------------------------------------------

    @property
    def file(self) -> SssFile:
        return self._file

    @property
    def recording(self) -> bool:
        """Whether no pass has ended yet (``mark_step``): no schedule to replay, every read is
        on demand unless a hint (``prefetch``) asked for it first."""
        return self._recording

    def ready(self, key: SlabKey) -> bool:
        """Whether ``read_into(key)`` would copy without waiting on the disk: its slab sits in
        the host cache, or a read of it has landed in the ring. Reads nothing, changes
        nothing."""
        if key in self._held:
            return True
        read = self._inflight.get(key)
        return read is not None and all(f.done() for f in read.futures)

    @property
    def host_cache(self) -> bool:
        """Whether it keeps slabs in host RAM once read (``cache_bytes``): then a read may be a
        RAM copy, and its caller's per-slab bookkeeping, not the disk, sets the pace."""
        return self._retain

    def keys(self) -> Sequence[SlabKey]:
        return self._keys

    def nbytes(self, key: SlabKey) -> int:
        return self._entry(key).nbytes

    def read_into(self, key: SlabKey, dst: torch.Tensor) -> None:
        """Copy ``key``'s bytes into ``dst`` on the current stream (see the module doc)."""
        e = self._entry(key)
        key = e.key
        check_dst(key, dst, e.nbytes)
        t0 = time.perf_counter()
        if self._step_t0 is None:
            self._step_t0 = t0
        pos = None
        if len(self._pass) < _MAX_TRACE:
            self._pass.append(key)
        if not self._recording:
            pos = self._resync(key)
        read = self._inflight.pop(key, None)
        if read is None and key in self._held:
            read = self._claim(key, None)
        if read is not None and read.cached:
            self._stats["cache_hits"] += 1
            self._stats["cache_bytes"] += e.nbytes
        elif read is not None:
            self._stats["prefetch_hits"] += 1
        else:
            self._stats["sync_reads"] += 1
            self._step_sync_bytes += e.nbytes
            n = self._nchunks(self._ring_nbytes(e))
            if n <= self._prefetchable:
                read = self._submit(key, None, self._take(n, block=True))
        if pos is not None:
            self._fill_window()
        verify = self._verify == "always" or (self._verify == "first"
                                              and key not in self._verified)
        if read is not None:
            self._consume(read, dst, verify)
        else:
            self._stream(key, dst, verify)
        if verify:
            self._verified.add(key)
            self._stats["verified"] += 1
        self._stats["reads"] += 1
        self._stats["bytes_read"] += e.nbytes
        self._step_reads += 1
        blocked = time.perf_counter() - t0
        self._step_blocked += blocked
        self._stats["blocked_s"] += blocked

    def prefetch(self, keys: Sequence[SlabKey]) -> None:
        """Start reading ``keys`` now if ring slots are free (best effort), in this order.

        The MoE block calls this with a layer's routed experts the moment the
        router has picked them (``poros.moe.modules``), so a step's new experts
        are read in parallel instead of one synchronous read each. Keys a
        previous call asked for and nobody read are dropped first when the
        ring needs room.
        """
        self._hints += 1
        keys = [self._entry(k).key for k in keys]
        # The router's pick for this layer beats the trace's guess: window reads of the
        # same layer that the router did not pick are dropped, and not issued again
        # until the pointer leaves the layer.
        # The caller's list already holds the look-ahead experts it may load.
        self._hint_layer = keys[0][0] if keys else None
        self._hint_keys = set(keys)
        for k, read in list(self._inflight.items()):
            if read.pos is not None and k[0] == self._hint_layer and k not in self._hint_keys:
                del self._inflight[k]
                self._draining.append(read)
                self._stats["prefetch_dropped"] += 1
        for key in keys:
            e = self._entry(key)
            n = self._nchunks(self._ring_nbytes(e))
            if e.key in self._inflight or n > self._prefetchable:
                continue
            if e.key in self._held:
                self._inflight[e.key] = self._claim(e.key, None, hint=self._hints)
                continue
            slots = self._take(n, block=False)
            if slots is None:
                return
            read = self._submit(e.key, None, slots)
            read.hint = self._hints
            self._inflight[e.key] = read
            self._stats["hinted"] += 1
        self._prefetch_next_phase()

    def _prefetch_next_phase(self) -> None:
        """Read the phase after this one (the trace's next run of one layer's reads) while
        this layer computes: the next router waits for this layer's compute, so without
        this the two layers' reads and compute would take turns."""
        if self._recording or not self._trace or self._hint_layer is None:
            return
        n = len(self._trace)
        p, seen = self._ptr, 0
        while seen < n and self._trace[(p + 1) % n][0] == self._hint_layer:
            p, seen = (p + 1) % n, seen + 1  # to the end of this phase
        if seen >= n:
            return
        nxt = self._trace[(p + 1) % n][0]
        while seen < n and self._trace[(p + 1) % n][0] == nxt:
            p, seen = (p + 1) % n, seen + 1
            k = self._trace[p]
            if k in self._inflight:
                continue
            if k in self._held:
                self._inflight[k] = self._claim(k, p)
                continue
            nchunks = self._nchunks(self._ring_nbytes(self._entries[k]))
            if nchunks > self._prefetchable:
                continue
            slots = self._take(nchunks, block=False)
            if slots is None:
                return
            self._inflight[k] = self._submit(k, p, slots)
            self._stats["phase_ahead"] += 1

    def release(self, key: SlabKey) -> None:
        """Drop a prefetched, unread slab and free its ring slots."""
        read = self._inflight.pop(self._entry(key).key, None)
        if read is not None:
            self._draining.append(read)

    # -- schedule --------------------------------------------------------------

    def mark_step(self) -> None:
        """One forward + backward pass ended: its reads become the schedule.

        The first call also autotunes ``depth`` (unless it was given). Every
        call replaces the trace with the pass just read, so the schedule
        follows the steady state rather than the cold first step.
        """
        now = time.perf_counter()
        if self._pass:
            first = self._recording
            self._recording = False
            self._trace, self._pass = self._pass, []
            n = len(self._trace)
            self._pos_of = {}
            for i, k in enumerate(self._trace):
                self._pos_of.setdefault(k, []).append(i)
            self._ptr = n - 1  # the next read wraps to the start of the trace
            self._win = -1
            self._horizon = 2 * max(len(r) for r in _runs(self._trace))
            self._pending = None
            for k, read in list(self._inflight.items()):  # re-anchor the open window
                if read.pos is not None:
                    positions = self._pos_of.get(k)
                    if positions:
                        read.pos = positions[0]
                    else:
                        del self._inflight[k]
                        self._draining.append(read)
            if first and self._step_t0 is not None and self._step_reads:
                compute = max(now - self._step_t0 - self._step_blocked, 0.0)
                self.compute_s_per_read = max(compute / self._step_reads, 1e-6)
                if not self._depth_fixed:
                    self.depth = self._autotune_depth()
                logger.info("SSS %s: schedule of %d reads, depth=%d", self.path.name, n,
                            self.depth)
        self._stats["steps"] += 1
        self._step_t0, self._step_blocked, self._step_reads = None, 0.0, 0
        self._step_sync_bytes = 0

    def _autotune_depth(self) -> int:
        from poros.sss.tune import prefetch_depth

        mean = sum(self._entries[k].nbytes for k in self._trace) / len(self._trace)
        bandwidth = self._bandwidth
        if not bandwidth and self._step_blocked > 0:
            bandwidth = self._step_sync_bytes / self._step_blocked
        # The router hint and the next-phase read supply the queue depth; the window only
        # covers reads out of their hinted order, so it stays short (a 32-deep window
        # starved the hints of ring slots on the 3090: 39% sync reads).
        cap = min(self.max_depth, 16,
                  max(1, self.num_slots // max(1, self._nchunks(int(mean)))))
        return prefetch_depth(mean, bandwidth or 1e9, self.compute_s_per_read,
                              latency_bandwidth=self._latency_bandwidth,
                              saturate=self._saturate, d_max=cap)

    def _next(self, key: SlabKey, after: int) -> tuple[int, int] | None:
        """``key``'s first trace position after ``after`` (cyclic) and the gap to it."""
        positions = self._pos_of.get(key)
        if not positions:
            return None
        i = bisect.bisect_right(positions, after)
        q = positions[i] if i < len(positions) else positions[0]
        return q, (q - after - 1) % len(self._trace)

    def _resync(self, key: SlabKey) -> int | None:
        """Move the trace pointer to ``key``'s next occurrence, if the read is on schedule.

        A read within ``near`` positions of the pointer advances it (the keys
        in between were not needed this pass: a VRAM cache hit or an expert
        no token routed to). A farther match -- the start of a pass after
        cached layers, or a key read early or late in its phase (the
        experts a step routes to vary) -- is accepted only once
        ``_CONFIRM`` consecutive reads follow each other in the trace; until
        then reads are off-schedule (synchronous) and the pointer stays.
        """
        near = max(8, self.depth)
        hit = self._next(key, self._ptr)
        if hit is None:
            self._pending = None
            return None
        q, gap = hit
        if gap <= near:
            self._pending = None
            return self._advance(q)
        if self._pending is not None:
            after, count = self._pending
            chained = self._next(key, after)
            if chained is not None and chained[1] <= near:
                if count + 1 >= _CONFIRM:
                    self._pending = None
                    return self._advance(chained[0])
                self._pending = (chained[0], count + 1)
                return None
        self._pending = (q, 1)
        return None

    def _advance(self, q: int) -> int:
        """Point at ``q``. Prefetches passed over stay in the ring until it needs
        the room: a key read later in its phase than the trace said still hits."""
        self._ptr = q
        return q

    def _fill_window(self) -> None:
        """Keep the next ``depth`` trace positions in flight (incremental: each call
        continues where the last one stopped while that is still ahead of the pointer)."""
        n = len(self._trace)
        span = min(self.depth, n - 1)
        ahead = (self._win - self._ptr) % n
        if self._win < 0 or ahead > span:
            self._win, ahead = self._ptr, 0  # after a jump: start at the pointer
        while ahead < span:
            p = (self._win + 1) % n
            k = self._trace[p]
            if not (k[0] == self._hint_layer and k not in self._hint_keys) \
                    and k not in self._inflight:  # skip: the router did not pick it
                nchunks = self._nchunks(self._ring_nbytes(self._entries[k]))
                if k in self._held:
                    self._inflight[k] = self._claim(k, p)
                elif nchunks <= self._prefetchable:
                    slots = self._take(nchunks, block=False)
                    if slots is None:
                        return  # ring full: retry from here on the next read
                    self._inflight[k] = self._submit(k, p, slots)
            self._win, ahead = p, ahead + 1

    # -- ring ------------------------------------------------------------------

    def _nchunks(self, nbytes: int) -> int:
        return max(1, -(-_align(nbytes) // self.chunk_bytes))

    def _submit(self, key: SlabKey, pos: int | None, slots: list[int]) -> _Read:
        e = self._entries[key]
        total = _align(self._ring_nbytes(e))
        # a slab still to verify that lands in one chunk: its reader thread takes the checksum
        hashed = (len(slots) == 1 and self._hashes_in_reader(e) and (
            self._verify == "always" or (self._verify == "first" and key not in self._verified)))
        futures = []
        for i, slot in enumerate(slots):
            start = i * self.chunk_bytes
            length = min(self.chunk_bytes, total - start)
            buf = self._mv[slot * self.chunk_bytes : slot * self.chunk_bytes + length]
            fn = self._read_hashed if hashed else self._read_chunk
            futures.append(self._pool.submit(fn, e, start, buf))
        return _Read(key, pos, slots, futures, hashed=hashed)

    def _hashes_in_reader(self, e) -> bool:
        """Whether a one-chunk read of ``e`` lands the slab's own bytes in the ring (so its
        reader can checksum them); a subclass landing other bytes says no."""
        return self._ring_nbytes(e) == e.nbytes

    def _read_hashed(self, e, start: int, buf: memoryview) -> str:
        """``_read_chunk``, then the checksum of the slab bytes it landed (in a reader thread:
        hashlib lets go of the GIL over a buffer this size)."""
        self._read_chunk(e, start, buf)
        h = hashlib.blake2b(digest_size=8)
        h.update(buf[: e.nbytes])
        return h.hexdigest()

    def _pin_cache(self, first: int, count: int) -> None:
        """The host cache's slots ``first .. first + count``, page-locked ``_CACHE_PIECE`` at a
        time on this thread (started by the first read), each piece offered to the ring as it is
        ready (``_reclaim``), so neither the load nor a read waits for it; a piece the pin budget
        refuses, or that fails to register, ends it (the cache is then smaller, the bytes the
        same)."""
        from poros.sss import pinning

        per = max(1, _CACHE_PIECE // self.chunk_bytes)
        rt = torch.cuda.cudart()
        for a in range(first, first + count, per):
            if self._closing.is_set():
                return
            n = min(per, first + count - a)
            nbytes = n * self.chunk_bytes
            piece = self._ring[a * self.chunk_bytes : (a + n) * self.chunk_bytes]
            piece.fill_(0)
            if not pinning.reserve(nbytes):
                return
            if int(rt.cudaHostRegister(piece.data_ptr(), nbytes, 0)) != 0:
                pinning.release(nbytes)
                logger.warning("SSS host cache stopped at %d MiB: a piece could not be "
                               "page-locked", (a - first) * self.chunk_bytes >> 20)
                return
            with self._cache_lock:
                self._cache_pins.append((piece.data_ptr(), nbytes))
                self._cache_new.extend(range(a, a + n))

    def _reclaim(self) -> None:
        """Return slots whose copies finished or whose dropped reads landed (and take in the
        host cache's slots pinned since)."""
        if self._cache_new:
            with self._cache_lock:
                new, self._cache_new = self._cache_new, []
            for slot in new:  # empty: taken before any slot that still holds a slab
                self._free[slot] = None
                self._free.move_to_end(slot, last=False)
        while self._copies and self._copies[0][0].query():
            _, slots, key = self._copies.popleft()
            self._put_free(slots, key)
        if self._draining:
            still = []
            for read in self._draining:
                if all(f.done() for f in read.futures):
                    self._free_dropped(read)
                else:
                    still.append(read)
            self._draining = still

    def _take(self, n: int, *, block: bool) -> list[int] | None:
        """``n`` free ring slots. Frees landed copies, then evicts prefetches the
        pointer has passed (most stale first); ``block`` also waits for copies
        and drains, then evicts the prefetches farthest ahead."""
        if n > self.num_slots:
            raise RuntimeError(f"a read of {n} chunks cannot fit a ring of {self.num_slots}")
        if self._cache_due is not None:  # the first read: the host cache starts to pin
            self._cache_thread = threading.Thread(target=self._pin_cache, args=self._cache_due,
                                                  daemon=True, name="poros-sss-pin")
            self._cache_due = None
            self._cache_thread.start()
        self._reclaim()
        if len(self._free) < n:
            stale = sorted((r for r in self._inflight.values() if self._behind(r) is not None),
                           key=self._behind, reverse=True)
            for read in stale:
                if len(self._free) >= n:
                    break
                del self._inflight[read.key]
                self._stats["prefetch_dropped"] += 1
                if all(f.done() for f in read.futures):
                    self._free_dropped(read)
                elif block:
                    self._wait_quietly(read)
                    self._free_dropped(read)
                else:
                    self._draining.append(read)
        if len(self._free) < n and block:
            while len(self._free) < n and self._copies:
                event, slots, key = self._copies.popleft()
                event.synchronize()
                self._put_free(slots, key)
            while len(self._free) < n and self._draining:
                read = self._draining.pop()
                self._wait_quietly(read)
                self._free_dropped(read)
            while len(self._free) < n and self._inflight:
                victim = max(self._inflight.values(), key=self._distance)
                del self._inflight[victim.key]
                self._stats["prefetch_dropped"] += 1
                self._wait_quietly(victim)
                self._free_dropped(victim)
        if len(self._free) < n:
            return None
        return [self._pop_free() for _ in range(n)]

    # -- the host cache (cache_bytes): free slots that still hold their slab -------------------

    def _put_free(self, slots: list[int], key: SlabKey | None = None) -> None:
        """Return ``slots`` to the free list (most recently released last); with the host cache
        on, a consumed read's slots keep holding ``key``'s slab until they are taken again."""
        for slot in slots:
            self._free[slot] = None
        if key is not None and self._retain:
            for old in self._held.pop(key, ()):
                self._owner.pop(old, None)
            self._held[key] = list(slots)
            for slot in slots:
                self._owner[slot] = key

    def _free_dropped(self, read: _Read) -> None:
        """A read dropped before it was consumed: its slots go free, still holding the slab
        where the host cache had served it (a checked slab), else holding nothing."""
        self._put_free(read.slots, read.key if read.cached else None)

    def _pop_free(self) -> int:
        """The least recently released free slot; a slab it held is held no more."""
        slot, _ = self._free.popitem(last=False)
        key = self._owner.pop(slot, None)
        if key is not None:
            for other in self._held.pop(key, ()):
                self._owner.pop(other, None)
        return slot

    def _claim(self, key: SlabKey, pos: int | None, hint: int | None = None) -> _Read:
        """``key``'s held slab, out of the free list, as a landed read (no disk read)."""
        slots = self._held.pop(key)
        for slot in slots:
            del self._free[slot]
            self._owner.pop(slot, None)
        return _Read(key, pos, slots, [_DONE] * len(slots), hint=hint, cached=True)

    def _behind(self, read: _Read) -> int | None:
        """How stale ``read`` is (higher: drop it first), or None while it may still be
        needed: inside the window's reach ahead of the pointer, or a hinted read of the
        latest ``prefetch`` call. Anything else -- passed over, or re-anchored far ahead
        by a new schedule -- is stale."""
        if read.hint is not None:
            return 1 << 40 if read.hint < self._hints else None
        if read.pos is None or not self._trace:
            return None
        n = len(self._trace)
        if 0 < (read.pos - self._ptr) % n <= max(2 * self.depth, 16, self._horizon):
            return None
        return (self._ptr - read.pos) % n or n

    def _distance(self, read: _Read) -> int:
        if read.pos is None or not self._trace:
            return 0
        return (read.pos - self._ptr) % len(self._trace)

    @staticmethod
    def _wait_quietly(read: _Read) -> None:
        for f in read.futures:
            f.cancel()
        for f in read.futures:
            if not f.cancelled():
                f.exception()  # waits; a failed speculative read is not an error

    def _release_slots(self, slots: list[int], dst: torch.Tensor,
                       key: SlabKey | None = None) -> None:
        """``slots`` are free once the copy to ``dst`` has run; ``key``: the slab they hold (a
        whole, checked read's), kept by the host cache."""
        if dst.device.type == "cuda" and slots:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(dst.device))
            self._copies.append((event, slots, key))
        else:
            self._put_free(slots, key)

    def _consume(self, read: _Read, dst: torch.Tensor, verify: bool) -> None:
        """Copy a landed (or landing) slab out of the ring into ``dst``."""
        e = self._entries[read.key]
        h = hashlib.blake2b(digest_size=8) if verify and not read.hashed else None
        digest = None
        done = 0
        try:
            for i, (slot, fut) in enumerate(zip(read.slots, read.futures, strict=True)):
                digest = fut.result()
                done = i + 1
                start = i * self.chunk_bytes
                length = min(self.chunk_bytes, e.nbytes - start)
                if length <= 0:  # alignment padding only
                    continue
                base = slot * self.chunk_bytes
                if h is not None:
                    h.update(self._mv[base : base + length])
                dst[start : start + length].copy_(self._ring[base : base + length],
                                                  non_blocking=True)
        except BaseException:
            # Slots up to ``done`` landed (copies may be queued); the rest may
            # still be written by their reads, so they drain first.
            self._release_slots(read.slots[:done], dst)
            self._draining.append(_Read(read.key, None, read.slots[done:],
                                        read.futures[done:]))
            raise
        ok = not verify or (digest if read.hashed else h.hexdigest()) == e.checksum
        self._release_slots(read.slots, dst, read.key if ok else None)
        if not ok:
            raise SssCorruptError(f"slab {e.key} in {self.path} does not match its checksum")

    def _stream(self, key: SlabKey, dst: torch.Tensor, verify: bool) -> None:
        """A slab larger than half the ring: rolling chunk reads through free slots."""
        e = self._entries[key]
        total = _align(e.nbytes)
        nchunks = self._nchunks(e.nbytes)
        h = hashlib.blake2b(digest_size=8) if verify else None
        pending: deque[tuple[int, int, Future]] = deque()
        nxt = 0
        try:
            while nxt < nchunks or pending:
                while nxt < nchunks:
                    slots = self._take(1, block=not pending)
                    if slots is None:
                        break
                    start = nxt * self.chunk_bytes
                    length = min(self.chunk_bytes, total - start)
                    base = slots[0] * self.chunk_bytes
                    fut = self._pool.submit(self._read_chunk, e, start,
                                            self._mv[base : base + length])
                    pending.append((nxt, slots[0], fut))
                    nxt += 1
                i, slot, fut = pending.popleft()
                try:
                    fut.result()
                    start = i * self.chunk_bytes
                    length = min(self.chunk_bytes, e.nbytes - start)
                    base = slot * self.chunk_bytes
                    if length > 0:
                        if h is not None:
                            h.update(self._mv[base : base + length])
                        dst[start : start + length].copy_(self._ring[base : base + length],
                                                          non_blocking=True)
                finally:
                    self._release_slots([slot], dst)
        except BaseException:
            for _, slot, fut in pending:
                self._draining.append(_Read(key, None, [slot], [fut]))
            raise
        if h is not None and h.hexdigest() != e.checksum:
            raise SssCorruptError(f"slab {e.key} in {self.path} does not match its checksum")

    # -- misc ------------------------------------------------------------------

    def _entry(self, key: SlabKey):
        return self._file.entry(key)

    @property
    def stats(self) -> dict:
        """Counters: reads, prefetch hits, synchronous reads, dropped prefetches, bytes,
        seconds the caller spent blocked in ``read_into``, reads the host cache served
        (``cache_hits``, ``cache_bytes``: no disk read), and the ring/IO setup."""
        return {**self._stats, "depth": self.depth, "direct": self.direct,
                "pinned": self.pinned,
                "pinned_bytes": self.pinned_bytes + sum(n for _, n in self._cache_pins),
                "chunk_bytes": self.chunk_bytes,
                "ring_bytes": self.num_slots * self.chunk_bytes, "cache": self._retain,
                "trace_len": len(self._trace), "recording": self._recording,
                "compute_s_per_read": self.compute_s_per_read,
                "fallback_reason": self.fallback_reason}

    def close(self, *, join: bool = True) -> None:
        """Wait for every read and copy, then release the ring and descriptors. ``join=False``
        leaves the idle reader threads to exit on their own instead of joining them."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            for read in [*self._inflight.values(), *self._draining]:
                self._wait_quietly(read)
            self._inflight.clear()
            self._draining.clear()
            for event, *_ in self._copies:
                event.synchronize()
            self._copies.clear()
            self._pool.shutdown(wait=join)
        finally:
            try:
                self._closing.set()
                # its last piece registered, or none. A finalizer never joins (the thread holds
                # this source, so a collected source's thread has already ended)
                if self._cache_thread is not None and join:
                    self._cache_thread.join()
                if self.pinned:  # before the ring's memory can be unmapped
                    from poros.sss import pinning

                    rt = torch.cuda.cudart()
                    for address, nbytes in self._cache_pins:
                        rt.cudaHostUnregister(address)
                        pinning.release(nbytes)
                    self._cache_pins.clear()
                    rt.cudaHostUnregister(self._ring.data_ptr())
                    pinning.release(self.pinned_bytes)
                    self.pinned = False
            finally:  # an __init__ that failed after _open_data (the ring's mmap) gets here too
                self._close_data()

    def __enter__(self) -> SsdSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        # No join here: the collector can run this inside threading's own bookkeeping (a new
        # thread starting holds a lock that joining a thread takes again), which deadlocked a
        # process for good. A source being collected has no read left, so its readers are idle
        # and exit once the pool is shut down.
        try:
            self.close(join=False)
        except Exception:
            pass
