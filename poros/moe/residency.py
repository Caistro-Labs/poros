"""ExpertBlockManager: expert-granular residency for MoE training.

Owns movement of quantized expert slabs between the host-side ExpertStore
(workstream B) and the compute device, under the dense engine's residency
discipline (poros/core/residency.py): a static schedule, a look-ahead
window, LRU eviction under a byte capacity, and bit-exact backward
rematerialization from immutable host slabs.

Guarantees:

- Static, router-independent schedule. Per MoE layer, slices stream in
  fixed ascending expert order; a training micro-batch saturates the route
  union, so the transfer pattern is identical every step.
- Bitwise rematerialization. Host slabs are the immutable source of truth,
  so a slice re-ensured after eviction is bit-identical (torch.equal) to
  its previous instance. Backward simply calls ensure again, in reverse
  layer order -- that loop lives in the caller; this manager is
  autograd-agnostic.
- No approximation, ever. ensure never drops, substitutes, or approximates
  a requested key; a key that cannot be made resident is a hard error,
  never a silent skip.
- Exact accounting. bytes_transferred counts every host-to-device slab
  byte, including rematerializations (feeds G-BYTES).

RNG: slab movement is a pure byte copy and consumes no RNG, so the INV-1
save/restore of poros/core/blockwise.py needs no hook here. Dropout in the
MoE path lives in the resident adapters (workstream B), which own their
RNG discipline.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch

from poros.moe.contracts import ExpertKey, ExpertResidency, ExpertStore

__all__ = ["ExpertBlockManager"]


# host_runs answers kept: a pass asks for each layer's slabs (and the subsets a layer lacked), so a
# few hundred cover a model; older ones are recomputed when asked again.
_RUNS_KEPT = 512


def runs_from_source(source, on_device_stream: bool) -> bool:
    """Whether ``source``'s slabs are copied to the device a request at a time (one buffer, one
    event): a streaming source that keeps a host cache (the SSD store with ``store_options:
    {cache: ...}``), where reads are RAM copies and per-slab bookkeeping sets the pace, on a
    device with a copy stream. Without the cache an SSD store's reads wait on the disk, which
    hides that bookkeeping, and one buffer a request would only hold more VRAM reserved
    (Qwen3-30B-A3B, path efficient: 8.88 GiB against 8.33), so each slab lands on its own. A
    host-RAM source has its own runs (``host_runs``)."""
    from poros.sss.source import RamSource

    return (on_device_stream and source is not None and not isinstance(source, RamSource)
            and bool(getattr(source, "host_cache", False)))


class _TransferEngine:
    """Host-to-device movement of one expert slab, behind one small seam.

    CPU: ``load_slice`` is synchronous, so a transfer is complete on return
    and ``wait`` is a no-op -- the stream machinery's correct degenerate
    case. CUDA: the copy runs on a dedicated side stream so look-ahead
    transfers overlap compute; one event per key orders the compute stream
    after the copy, and the slab is ``record_stream``-ed against the
    default stream, mirroring the dense engine's INV-2 cross-stream safety
    rule (poros/core/residency.py).
    """

    def __init__(self, store: ExpertStore, device: torch.device) -> None:
        self._store = store
        self._device = device
        self._events: dict[ExpertKey, torch.cuda.Event] = {}
        # host_runs per key list (the slabs never move), the most recent _RUNS_KEPT of them
        self._runs: OrderedDict[tuple, list | None] = OrderedDict()
        self._waited: tuple | None = None  # (event, stream) the last wait ordered
        self._stream: torch.cuda.Stream | None = None
        if device.type == "cuda" and torch.cuda.is_available():
            self._stream = torch.cuda.Stream(device)
        #: slabs come through a streaming source's reads, copied a request at a time
        #: (source_runs; runs_from_source says when)
        self.streaming = runs_from_source(getattr(store, "source", None), self._stream is not None)

    def transfer(self, key: ExpertKey, nbytes: int) -> torch.Tensor:
        """Start moving ``key``'s slab to the device and return the slab.

        On CPU the copy is complete on return. On CUDA it may still be in
        flight; ``wait`` must order the compute stream before first use.
        """
        if self._stream is None:
            slab = torch.empty(nbytes, dtype=torch.uint8, device=self._device)
            self._store.load_slice(key, slab)
            return slab
        with torch.cuda.stream(self._stream):
            slab = torch.empty(nbytes, dtype=torch.uint8, device=self._device)
            self._store.load_slice(key, slab)
            event = torch.cuda.Event()
            event.record(self._stream)
        self._events[key] = event
        # INV-2: wait() orders the current compute stream, so record against
        # it. Also record the default stream, which may free the allocator.
        compute = torch.cuda.current_stream(self._device)
        slab.record_stream(compute)
        default = torch.cuda.default_stream(self._device)
        if compute != default:
            slab.record_stream(default)
        return slab

    def host_runs(
        self, keys: Sequence[ExpertKey]
    ) -> list[list[tuple[ExpertKey, torch.Tensor]]] | None:
        """``keys``' host slabs (pinned RAM store) grouped into runs that lie back to back in
        host memory, in ``keys`` order; None where the store has no such slabs (other sources,
        pageable memory, CPU compute)."""
        from poros.sss.source import RamSource

        source = getattr(self._store, "source", None)
        if self._stream is None or not isinstance(source, RamSource):
            return None
        memo = (tuple(keys[:1]), tuple(keys[-1:]), len(keys))
        cached = self._runs.get(memo, False)
        if cached is not False and [k for run in cached or [] for k, _ in run] == list(keys):
            self._runs.move_to_end(memo)
            return cached
        runs = self._runs[memo] = self._host_runs(source, keys)
        self._runs.move_to_end(memo)
        while len(self._runs) > _RUNS_KEPT:
            self._runs.popitem(last=False)
        return runs

    def _host_runs(self, source, keys):
        """``keys``' pinned host slabs grouped into runs: a slab joins the run before it when it
        starts where that run ends *in the same storage*. Two storages can be adjacent in
        virtual memory (a pinned chunk mapped right after a full one); ``transfer_run`` views a
        run through its first slab's storage, so a run must never cross into another."""
        runs: list[list[tuple[ExpertKey, torch.Tensor]]] = []
        end = storage = None
        for key in keys:
            slab = source._slab(key)
            if not slab.is_pinned():
                return None
            here = slab.untyped_storage().data_ptr()
            if end is not None and slab.data_ptr() == end and here == storage:
                runs[-1].append((key, slab))
            else:
                runs.append([(key, slab)])
            end, storage = slab.data_ptr() + slab.numel(), here
        return runs

    def source_runs(self, keys: Sequence[ExpertKey]
                    ) -> list[list[tuple[ExpertKey, None]]] | None:
        """``keys`` as one run for a streaming source with a host cache (the SSD store's
        ``store_options: {cache: ...}``: its slabs come through ``read_into``, not from host
        memory), so a layer's slabs land in one device buffer with one event instead of an
        allocation, an event and two stream records each; None for a host-RAM source
        (``host_runs``), an SSD store without the cache, or the CPU's compute."""
        if not self.streaming:
            return None
        return [[(key, None) for key in keys]]

    def transfer_run(self, run: list[tuple[ExpertKey, torch.Tensor | None]],
                     nbytes: Sequence[int] | None = None) -> list[torch.Tensor]:
        """Start one copy of a back-to-back run of host slabs into one device buffer; returns
        each key's slab, a view of that buffer (the same bytes ``transfer`` would move). A
        streaming source's run (``source_runs``: no host slabs) fills each view with the
        store's own ``load_slice`` on the copy stream, in ``run``'s order."""
        if run[0][1] is None:
            return self._transfer_source_run(run, nbytes)
        total = sum(slab.numel() for _, slab in run)
        first = run[0][1]
        host = torch.empty(0, dtype=torch.uint8).set_(
            first.untyped_storage(), first.storage_offset(), (total,))
        with torch.cuda.stream(self._stream):
            buf = torch.empty(total, dtype=torch.uint8, device=self._device)
            buf.copy_(host, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
        compute = torch.cuda.current_stream(self._device)
        buf.record_stream(compute)
        default = torch.cuda.default_stream(self._device)
        if compute != default:
            buf.record_stream(default)
        views, start = [], 0
        for key, slab in run:
            views.append(buf[start:start + slab.numel()])
            self._events[key] = event
            start += slab.numel()
        return views

    def _transfer_source_run(self, run, nbytes: Sequence[int]) -> list[torch.Tensor]:
        """``transfer_run`` for a streaming source: one device buffer, each key's view filled by
        ``load_slice`` (the source's ``read_into``, which copies on the current stream), one
        event for the run."""
        with torch.cuda.stream(self._stream):
            buf = torch.empty(sum(nbytes), dtype=torch.uint8, device=self._device)
            views, start = [], 0
            for (key, _), n in zip(run, nbytes, strict=True):
                view = buf[start:start + n]
                self._store.load_slice(key, view)
                views.append(view)
                start += n
            event = torch.cuda.Event()
            event.record(self._stream)
        compute = torch.cuda.current_stream(self._device)
        buf.record_stream(compute)
        default = torch.cuda.default_stream(self._device)
        if compute != default:
            buf.record_stream(default)
        for key, _ in run:
            self._events[key] = event
        return views

    def wait(self, key: ExpertKey) -> None:
        """Order the compute stream after ``key``'s copy (no-op on CPU). A run's keys share one
        event: the stream waits on it once."""
        event = self._events.pop(key, None)
        if event is not None:
            stream = torch.cuda.current_stream(self._device)
            if self._waited != (event, stream):
                stream.wait_event(event)
                self._waited = (event, stream)

    def discard(self, key: ExpertKey) -> None:
        """Drop the pending event of an evicted slice (no-op on CPU)."""
        self._events.pop(key, None)


class ExpertBlockManager(ExpertResidency):
    """Expert-slice residency under a byte capacity, on a static schedule.

    Args:
        store: Host-side owner of the immutable quantized expert slabs.
        streamed: Stream slices under ``capacity_bytes`` with LRU eviction.
            With ``streamed=False`` every slice is ensured up front and
            nothing is ever evicted -- the parity-reference configuration
            (G-TRAIN). Same code path with eviction disabled, not a fork.
        capacity_bytes: Device-resident budget. When streaming it must hold
            at least one layer's full slice set; when resident it must hold
            every slice.
        look_ahead: How many slices beyond a requested slice to keep in
            flight per layer, in static schedule order.
        device: Compute device. CPU is fully supported (synchronous copies).

    Eviction discipline: an ensure call never evicts slices of the layers
    it touches, so within a layer's forward an ensured slice cannot be
    evicted before use -- steps are single-threaded, so no other ensure can
    interleave between a layer's ensure and its compute. Eviction candidates
    are the least recently ensured slices of *other* layers. A slice becomes
    eligible again as soon as the ensure that made it resident returns
    ("released after forward use"); backward re-ensures in reverse layer
    order and gets bit-identical slabs.
    """

    def __init__(
        self,
        store: ExpertStore,
        *,
        streamed: bool = True,
        capacity_bytes: int,
        look_ahead: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if look_ahead < 0:
            raise ValueError(f"look_ahead must be >= 0, got {look_ahead}")
        self._store = store
        self._streamed = bool(streamed)
        self._capacity = int(capacity_bytes)
        self._floor = self._capacity  # never shrink below the configured budget
        self._look_ahead = int(look_ahead)
        self._device = torch.device(device)
        self._engine = _TransferEngine(store, self._device)
        # Resident slabs in LRU order (least recently ensured first).
        self._resident: OrderedDict[ExpertKey, torch.Tensor] = OrderedDict()
        self._resident_bytes = 0
        self._layer_keys: dict[int, int] = {}  # resident keys per layer (layer_resident)
        self._layer_bytes: dict[int, int] = {}  # resident bytes per layer (prefetch's room check)
        # A bulk run's keys (_bulk): their slabs are views of one device buffer, freed only when
        # the last goes, so they are evicted together and the accounting is the bytes held.
        self._run_of: dict[ExpertKey, tuple[ExpertKey, ...]] = {}
        # layers whose slices stay once resident (set_sticky); the others cycle through the rest
        self._sticky: frozenset[int] = frozenset()
        self._bytes_transferred = 0
        if not self._streamed:
            layout = store.layout
            self.ensure(
                [
                    (layer, expert)
                    for layer in range(layout.num_layers)
                    for expert in range(layout.experts_per_layer)
                ]
            )
            if self._resident_bytes > self._capacity:
                raise ValueError(
                    "streamed=False pins every expert slice: "
                    f"{self._resident_bytes} B needed > capacity_bytes="
                    f"{self._capacity}. Raise capacity_bytes or use streamed=True."
                )

    def ensure(self, keys: Sequence[ExpertKey]) -> None:
        """Block until every requested key is resident on the compute device.

        Keys are committed in static schedule order (ascending layer, then
        ascending expert). Requested keys are guaranteed resident on return;
        each requested slice also keeps the next ``look_ahead`` slices of
        its layer in flight (best-effort under capacity). Never drops,
        substitutes, or approximates a requested key.
        """
        ordered = sorted(set(keys))
        if not ordered:
            return
        protected = {layer for layer, _ in ordered}
        self._bulk([k for k in ordered if k not in self._resident], protected, required=True)
        for key in ordered:
            self._ensure_one(key, protected, required=True)
        if self._look_ahead:
            experts_per_layer = self._store.layout.experts_per_layer
            for layer, expert in ordered:
                stop = min(expert + self._look_ahead, experts_per_layer - 1)
                for nxt in range(expert + 1, stop + 1):
                    self._ensure_one((layer, nxt), protected, required=False)

    def set_sticky(self, layers: Sequence[int]) -> None:
        """Keep these layers' slices once resident: eviction takes the other layers' first
        (a slice that is required, or an OOM, still takes any). Spread evenly over a pass, a
        streamed layer's copy runs while sticky layers compute, instead of a run of streamed
        layers waiting on the copy engine back to back. Decides when bytes move, never which."""
        self._sticky = frozenset(int(layer) for layer in layers)

    def _room(self, protected: set[int]) -> int:
        """Bytes free or held by layers outside ``protected`` and the sticky layers."""
        held = sum(n for layer, n in self._layer_bytes.items()
                   if layer not in protected and layer not in self._sticky)
        return self._capacity - self._resident_bytes + held

    def layer_resident(self, layer: int) -> bool:
        """Whether every expert slice of ``layer`` is resident (or in flight)."""
        return self._layer_keys.get(layer, 0) == self._store.layout.experts_per_layer

    def prefetch(self, keys: Sequence[ExpertKey], keep: Sequence[int] = ()) -> bool:
        """Start moving ``keys``' slabs to the device without waiting for them.

        Best effort under the budget, like look-ahead: a later ``ensure`` finds each key
        resident or in flight (its copy ordered before first use by ``resident_qbytes``).
        Never evicts a slice of ``keys``' layers or of the layers in ``keep`` (the one
        computing now). The cache decides when bytes move, never which: no value changes.
        Returns whether every key is now resident or in flight.
        """
        if not self._streamed:
            return all(k in self._resident for k in keys)
        ordered = sorted(set(keys))
        protected = {layer for layer, _ in ordered} | set(keep)
        missing = [k for k in ordered if k not in self._resident]
        if missing and sum(self._store.slice_nbytes(k) for k in missing) > self._room(protected):
            return False  # no room without evicting what must stay: nothing to try
        if self._bulk(missing, protected, required=False) and any(
                k not in self._resident for k in missing):
            return False  # no room for the run: the experts one by one would find none either
        for key in ordered:
            self._ensure_one(key, protected, required=False)
        return all(k in self._resident for k in ordered)

    def _bulk(self, missing: list[ExpertKey], protected: set[int], *, required: bool) -> bool:
        """Make ``missing`` resident in as few copies as the host layout allows: one device
        buffer and one copy per run of slabs that lie back to back in pinned host memory (a
        layer's experts usually do), each key's slab a view of it -- instead of an allocation,
        a copy and two events per expert. A streaming source with a host cache (the SSD store's)
        takes the request as one run: one buffer and one event, each slab read into its view by
        the source (``source_runs``), and one eviction pass for all of it. Bytes, keys and
        accounting are ``_ensure_one``'s; anything it cannot do here (no room, an allocation
        failure) is left to ``_ensure_one``. Returns whether the slabs are bulk-copyable."""
        if len(missing) < 2:
            return False
        runs = self._engine.host_runs(missing)
        if runs is None and self._capacity >= 2 * self._layer_nbytes():
            # a streaming source's runs only for a cache of several layers: there each slab's
            # eviction scan and stream records set the pace; a one-layer cache (path efficient)
            # has little to scan, and one buffer a request would hold more VRAM reserved for no
            # time (Qwen3-30B-A3B with the host cache: 9.13 GiB against 8.35, 7.87 s against 7.82)
            runs = self._engine.source_runs(missing)
        if runs is None:
            return False
        for run in runs:
            if len(run) < 2:
                continue
            nbytes = [self._store.slice_nbytes(key) for key, _ in run]
            if any(slab is not None and n != slab.numel() for n, (_, slab) in zip(nbytes, run)):
                continue
            total = sum(nbytes)
            if self._streamed and not self._make_room(run[0][0], total, protected, False):
                continue
            try:
                views = self._engine.transfer_run(run, nbytes)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                continue  # the per-expert path makes room or fails as before
            mates = tuple(key for key, _ in run)
            for (key, _), view, n in zip(run, views, nbytes):
                self._run_of[key] = mates
                self._resident[key] = view
                self._resident_bytes += n
                self._layer_keys[key[0]] = self._layer_keys.get(key[0], 0) + 1
                self._layer_bytes[key[0]] = self._layer_bytes.get(key[0], 0) + n
                self._bytes_transferred += n
        return True

    def _layer_nbytes(self) -> int:
        """One layer's expert slabs, in bytes (the largest layer's, for a store whose layers
        differ: ``max_layer_nbytes``)."""
        largest = getattr(self._store, "max_layer_nbytes", None)
        if largest is not None:
            return int(largest)
        return self._store.layout.experts_per_layer * self._store.slice_nbytes((0, 0))

    def resident_qbytes(self, key: ExpertKey) -> torch.Tensor:
        """The resident flat uint8 slab for ``key`` on the compute device.

        Raises ``KeyError`` if absent -- a programming error; call ``ensure``
        first. There is no silent fallback.
        """
        slab = self._resident.get(key)
        if slab is None:
            raise KeyError(
                f"expert slice {key} is not resident; call ensure() for it "
                "first (no silent fallback)"
            )
        self._engine.wait(key)
        return slab

    @property
    def bytes_transferred(self) -> int:
        """Total host-to-device slab bytes moved since construction.

        Includes backward rematerializations; feeds G-BYTES.
        """
        return self._bytes_transferred

    def _ensure_one(self, key: ExpertKey, protected: set[int], *, required: bool) -> None:
        """Make ``key`` resident. Required slices are waited on; look-ahead
        slices are best-effort under capacity and may stay in flight."""
        slab = self._resident.get(key)
        if slab is not None:
            self._resident.move_to_end(key)
            if required:
                self._engine.wait(key)
            return
        nbytes = self._store.slice_nbytes(key)
        # a streaming source's look-ahead slice never evicts a request's run (one buffer: all of a
        # layer's slabs would go, read again from the disk, for one speculative slab)
        runs = required or not self._engine.streaming
        if self._streamed and not self._make_room(key, nbytes, protected, required, runs=runs):
            return  # look-ahead is best-effort: the window just shrinks
        try:
            slab = self._engine.transfer(key, nbytes)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            # A residency grown into "unused" VRAM (sss.tune.grow_expert_cache) can meet a
            # later pass's larger transients: give slices of other layers back and shrink the
            # budget to what is resident, then retry once. The cache decides when bytes move,
            # never which, so no value changes (seen as a cuMemSetAccess OOM).
            if "out of memory" not in str(exc).lower() or not self._streamed:
                raise
            if not required:
                return
            self._shrink_for(nbytes * 8, protected)
            slab = self._engine.transfer(key, nbytes)
        self._resident[key] = slab
        self._resident_bytes += nbytes
        self._layer_keys[key[0]] = self._layer_keys.get(key[0], 0) + 1
        self._layer_bytes[key[0]] = self._layer_bytes.get(key[0], 0) + nbytes
        self._bytes_transferred += nbytes
        if required:
            self._engine.wait(key)

    def shrink_to_floor(self) -> int:
        """Give back everything above the configured budget: evict LRU slices down to it, drop
        the sticky set, and return the memory to the driver. Between passes only (no layer is
        protected). Decides when bytes move, never which. Returns the bytes freed."""
        self._sticky = frozenset()
        freed = 0
        floor = getattr(self, "_floor", 1)
        while self._resident and self._resident_bytes > floor:
            for mate in self._mates(next(iter(self._resident))):
                freed += self._evict(mate)
        self._capacity = floor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return freed

    def _shrink_for(self, nbytes: int, protected: set[int]) -> None:
        """Evict LRU slices of unprotected layers until ``nbytes`` are freed (or none are
        left), set the budget to what stays resident, and return the memory to the driver."""
        freed = 0
        while freed < nbytes:
            victim = next((k for k in self._resident if k[0] not in protected), None)
            if victim is None:
                break
            for mate in self._mates(victim):  # a bulk run's buffer goes whole: bytes really freed
                freed += self._evict(mate)
        self._capacity = max(self._resident_bytes, getattr(self, "_floor", 1))
        self.oom_shrinks = getattr(self, "oom_shrinks", 0) + 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_room(
        self, key: ExpertKey, nbytes: int, protected: set[int], required: bool, *,
        runs: bool = True,
    ) -> bool:
        """Evict LRU slices of other layers until ``key`` fits.

        Returns False when a best-effort look-ahead slice cannot fit without
        evicting a protected layer (then nothing is evicted); raises when a
        requested slice cannot -- capacity too small for a layer's working set
        is a hard error, never a silent skip. The victims are found in one pass
        over the LRU order (a protected run at its head is not rescanned per
        victim). ``runs=False``: no bulk run's slab is a victim (a best-effort slice).
        """
        need = self._resident_bytes + nbytes - self._capacity
        if need <= 0:
            return True
        victims: list[ExpertKey] = []
        chosen: set[ExpertKey] = set()
        freed = 0

        def take(k: ExpertKey) -> None:  # with its bulk run's mates: one buffer, freed whole
            nonlocal freed
            for mate in self._mates(k):
                if mate not in chosen:
                    chosen.add(mate)
                    victims.append(mate)
                    freed += self._resident[mate].numel()

        for k in self._resident:
            if freed >= need:
                break
            if k[0] in protected or k[0] in self._sticky or k in chosen or (
                    not runs and k in self._run_of):
                continue
            take(k)
        if freed < need and required and self._sticky:  # a required slice takes sticky ones too
            for k in self._resident:
                if freed >= need:
                    break
                if k[0] in protected or k in chosen:
                    continue
                take(k)
        if freed < need:
            if not required:
                return False
            raise RuntimeError(
                f"ExpertBlockManager: cannot make expert slice {key} resident: "
                f"capacity_bytes={self._capacity} is exhausted by the layers of "
                f"the current request {sorted(protected)} "
                f"({self._resident_bytes} B resident, slice needs {nbytes} B). "
                "Raise capacity_bytes: a layer's full slice set must be resident at once."
            )
        for victim in victims:
            self._evict(victim)
        return True

    def _mates(self, key: ExpertKey) -> tuple[ExpertKey, ...]:
        """``key`` and, for a bulk run's key, the run's other keys still resident."""
        run = self._run_of.get(key)
        return (key,) if run is None else tuple(k for k in run if k in self._resident)

    def _evict(self, victim: ExpertKey) -> int:
        """Drop ``victim``'s slab and its accounting; the bytes it held."""
        slab = self._resident.pop(victim)
        self._run_of.pop(victim, None)
        n = slab.numel()
        self._layer_keys[victim[0]] -= 1
        self._layer_bytes[victim[0]] -= n
        self._engine.discard(victim)
        self._resident_bytes -= n
        return n
