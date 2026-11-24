"""SSS sizing rules: prefetch depth and the VRAM cache budget (torch-free).

- Prefetch depth ``d = clamp(ceil(t_read / t_c) + 2, 3, d_max)`` with
  ``t_read = nbytes / B + 100 us``: enough reads in flight to cover one
  read's latency at the effective bandwidth ``B = min(B_ssd, B_h2d)``, plus
  two for triple buffering.
- VRAM cache: after the first step, the expert residency grows into the VRAM
  the run leaves unused, ``extra = budget - peak_reserved - longer - headroom``:
  ``budget`` the VRAM this process may hold (what was free at setup, plus
  what its own allocator held), ``peak_reserved`` the run's measured peak so
  far, ``longer`` what an update larger than the measured ones would add
  (``poros.plan.estimate.token_bytes``, an estimate that errs high), ``headroom`` the
  safety margin (``max(3 GiB, 0.2 x budget)``). The cache only decides *when*
  bytes move, never which bytes, so its size cannot change numerics.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

__all__ = ["READ_OVERHEAD_S", "VRAM_HEADROOM", "CacheFit", "grow_expert_cache",
           "grown_capacity", "prefetch_depth", "sticky_layers", "vram_budget"]

READ_OVERHEAD_S = 100e-6
VRAM_HEADROOM = 3 << 30


def prefetch_depth(nbytes: float, bandwidth: float, compute_s: float, *,
                   latency_bandwidth: float | None = None, saturate: int | None = None,
                   d_min: int = 3, d_max: int = 64) -> int:
    """Keys to keep in flight for slabs of ``nbytes`` read every ``compute_s`` seconds.

    ``t_read`` is one read's latency: at ``latency_bandwidth`` (one read in
    flight, the probe's QD1 figure) when known, since a queued read only gets
    ``bandwidth`` (measured at the probe's queue depth) with that queue full.
    When the reads cannot keep up with compute at all (``nbytes / compute_s >
    bandwidth``) the disk is the bottleneck and the depth is at least
    ``saturate`` (the probe's queue depth), so the disk sees a full queue.
    """
    if bandwidth <= 0 or compute_s <= 0:
        raise ValueError("bandwidth and compute_s must be positive")
    t_read = nbytes / (latency_bandwidth or bandwidth) + READ_OVERHEAD_S
    d = math.ceil(t_read / compute_s) + 2
    if saturate and nbytes / compute_s > bandwidth:
        d = max(d, int(saturate))
    d_max = max(int(d_max), 1)
    return max(min(d, d_max), min(d_min, d_max))


@dataclass(frozen=True)
class CacheFit:
    """How ``grow_expert_cache`` sized a MoE model's expert residency (bytes)."""

    capacity: int  # the residency's budget now
    budget: int  # the VRAM this process may hold (vram_budget; at setup, where measured)
    peak: int  # the run's reserved peak so far, which the cache sits beside
    longer: int  # what an update larger than the measured ones would add (an estimate)
    headroom: int  # left free: the safety margin
    needed: int  # every expert slab's bytes: no point caching more
    cap: int | None  # moe_expert_cache_gib in bytes; None: none (fit the GPU)

    def reason(self) -> str:
        """The sizing, in words (recorded beside the size: ran_compute)."""
        def gib(n: int) -> str:
            return f"{n / 2**30:.2f} GiB"

        fit = (f"{gib(self.budget)} this process may hold - {gib(self.peak)} reserved peak "
               + (f"- {gib(self.longer)} for longer updates " if self.longer else "")
               + f"- {gib(self.headroom)} margin")
        if self.capacity >= self.needed:
            return f"every expert ({gib(self.needed)}) is cached: {fit}"
        if self.cap is not None and self.capacity >= self.cap:
            return f"moe_expert_cache_gib caps it ({fit} leaves more)"
        return f"sized to this GPU: {fit}"


def vram_budget(device) -> int | None:
    """The VRAM this process may hold on ``device``: what is free now plus what its own allocator
    holds (other processes' memory and the CUDA context excluded), at most the per-process cap
    (``_usable_vram``). None off CUDA."""
    import torch

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    usable = _usable_vram(torch, device)
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except RuntimeError:  # pragma: no cover - a driver without the query
        return usable
    return min(int(free) + int(torch.cuda.memory_reserved(device)), usable)


def grow_expert_cache(model, device, max_bytes: int | None = None, *, budget: int | None = None,
                      longer: int = 0) -> CacheFit | None:
    """Grow a prepared MoE model's expert residency into the VRAM the run leaves unused.

    ``max_bytes`` caps the residency (moe_expert_cache_gib; None: every expert's bytes);
    ``budget`` is the VRAM this process may hold, measured at setup (None: measured now; the
    lower of the two is used, so memory another process took since counts); ``longer`` is what
    an update larger than the ones measured would add to the run's peak. The headroom left
    free is ``max(3 GiB, 0.2 x budget)``. Returns the sizing, or None when there is nothing to
    grow (no CUDA, no managed blocks). ``ExpertBlockManager`` reads its budget from
    ``_capacity`` on every ``ensure``; the cache decides when bytes move, never which, so this
    cannot change numerics. The budget never shrinks here.
    """
    import torch

    from poros.moe.modules import ManagedMoeBlock

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    store = getattr(model, "_poros_moe_store", None)
    residency = next((m.residency for m in model.modules() if isinstance(m, ManagedMoeBlock)),
                     None)
    if store is None or residency is None or not hasattr(residency, "_capacity"):
        return None
    layout = store.layout
    needed = store_nbytes(store)
    now = vram_budget(device)
    budget = now if budget is None else min(int(budget), now)
    target = needed if max_bytes is None else max(min(needed, int(max_bytes)),
                                                  residency._capacity)
    peak = int(torch.cuda.max_memory_reserved(device))
    # at least 3 GiB: a cache grown to within 3 GiB of a 22.5 GiB cap left block-major
    # transients fragmenting (one 450 s update, SSD store); 0.2 of the budget by default
    headroom = max(VRAM_HEADROOM, int(budget * _headroom_frac()))
    residency._capacity = grown_capacity(residency._capacity, budget, peak + max(int(longer), 0),
                                         target, headroom=headroom)
    set_sticky = getattr(residency, "set_sticky", None)
    if set_sticky is not None and os.environ.get("POROS_MOE_STICKY", "1") != "0":
        set_sticky(sticky_layers(layout.num_layers, residency._capacity // layer_nbytes(store)))
    return CacheFit(capacity=int(residency._capacity), budget=int(budget), peak=peak,
                    longer=max(int(longer), 0), headroom=int(headroom), needed=int(needed),
                    cap=None if max_bytes is None else int(max_bytes))


def store_nbytes(store) -> int:
    """Every expert slab's bytes: each (layer, expert) at its own size, so a store whose layers
    differ in size (Inkling 2's released experts: MXFP8 layers about twice an NVFP4 one's) is
    counted whole. A store whose slabs are all one size gives layers x experts x that size."""
    layout = store.layout
    return sum(layout.experts_per_layer * store.slice_nbytes((layer, 0))
               for layer in range(layout.num_layers))


def layer_nbytes(store) -> int:
    """The largest MoE layer's expert slabs, in bytes: the unit a cache counts its layers in, so
    the layers it keeps fit whichever they are."""
    layout = store.layout
    return max(layout.experts_per_layer * store.slice_nbytes((layer, 0))
               for layer in range(layout.num_layers))


def sticky_layers(num_layers: int, layers_in_cache: int, streaming: int = 3) -> list[int]:
    """The layers a cache of ``layers_in_cache`` layers keeps, spread evenly over the model:
    all but ``streaming`` of its layers (the layer computing and two in flight), so the layers
    it streams sit between ones it keeps and their copies overlap compute. A cache that holds
    every layer keeps them all; one too small to keep any keeps none (plain LRU)."""
    if layers_in_cache >= num_layers:
        return list(range(num_layers))
    keep = max(0, layers_in_cache - streaming)
    return [layer for layer in range(num_layers)
            if (layer + 1) * keep // num_layers > layer * keep // num_layers]


def _headroom_frac() -> float:
    """Share of the budget the grown expert residency leaves free beyond the run's measured peak
    (default 0.2). The notes behind it measured the peak before update 1's optimizer step, which
    on Qwen3-30B-A3B (LoRA r16 on every expert) makes 2.2 GiB of Adam state, a tenth of an RTX
    4090's budget: there 0.2 hit a torch expandable-segment allocator assert, and 0.3 ran stable
    (13.3 s/update, and 1.43 s on the fused compute with 3.7 GiB left free, receipt q30h_def_fast).
    The peak now includes that step (poros.train.compute), so 0.2 leaves about what the stable
    0.3 left (1.49 s at 0.3: q30i_def_fast). POROS_MOE_CACHE_HEADROOM overrides."""
    import os

    try:
        return min(max(float(os.environ.get("POROS_MOE_CACHE_HEADROOM", "0.2")), 0.0), 0.9)
    except ValueError:
        return 0.2


def _usable_vram(torch, device) -> int:
    """The device's memory this process may allocate: the total, scaled by a
    ``per_process_memory_fraction`` cap when one is set (API or
    PYTORCH_CUDA_ALLOC_CONF) -- a cache grown past the cap OOMs the next pass
    (Qwen3-30B-A3B under a 0.9375 cap, SSD store)."""
    import os
    import re

    total = int(torch.cuda.get_device_properties(device).total_memory)
    frac = None
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None) or getattr(
        torch.cuda.memory, "get_per_process_memory_fraction", None)
    if getter is not None:
        try:
            frac = float(getter(device))
        except Exception:  # pragma: no cover - API drift
            frac = None
    if frac is None:
        m = re.search(r"per_process_memory_fraction:([0-9.]+)",
                      os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        frac = float(m.group(1)) if m else None
    return int(total * frac) if frac and 0 < frac < 1 else total


def grown_capacity(capacity: int, total_vram: int, peak_reserved: int, needed: int, *,
                   headroom: int = VRAM_HEADROOM) -> int:
    """The residency budget after step 1: ``capacity`` plus unused VRAM, at most ``needed``.

    ``needed`` is the byte size of every expert slab (no point caching more);
    the budget never shrinks.
    """
    extra = int(total_vram) - int(peak_reserved) - int(headroom)
    return max(int(capacity), min(int(capacity) + max(extra, 0), int(needed)))
