"""The trainer's reporting seam.

``PorosTrainer.reporter`` is ``None`` by default and the trainer prints its
plain progress lines exactly as before. A front end (the CLI's live view)
sets a reporter: the trainer then calls it instead of printing, and the
reporter decides what the terminal shows.

The probes here (host RAM, streaming position, transfer rate) are cheap
reads of counters the runtime already keeps, safe to call from a UI thread
while training runs. Torch is never imported at module import time.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class StepReport:
    """One optimizer step. ``step`` is 1-based for display."""

    step: int
    total: int
    loss: float
    step_time: float
    tokens: int  # cumulative
    tokens_per_sec: float
    lr: float
    vram_gb: float = 0.0  # allocated now
    vram_peak_gb: float = 0.0  # max allocated since training started
    vram_reserved_gb: float = 0.0
    # the loss's own numbers this step (DPO: accuracy, margin, rewards)
    extra: Dict[str, float] = field(default_factory=dict)


class Reporter:
    """No-op base. Override what you need; every hook must be cheap and must
    never raise into the training loop (the trainer guards the calls anyway)."""

    def on_start(self, trainer: Any, total: int, start_step: int) -> None: ...

    def on_step(self, report: StepReport) -> None: ...

    def on_checkpoint(self, step: int, path: str) -> None: ...

    def on_eval(self, step: int, metrics: Dict[str, Any]) -> None: ...

    def on_interrupt(self, step: int, checkpoint: Optional[str]) -> None: ...

    def on_done(self, metrics: Dict[str, Any]) -> None: ...


# --------------------------------------------------------------------------
# host memory
# --------------------------------------------------------------------------


def _read_kb(path: str, key: str) -> Optional[int]:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_ram_gb() -> Optional[float]:
    """Resident set size of this process, GB (decimal, like the VRAM figures)."""
    kb = _read_kb("/proc/self/status", "VmRSS:")
    if kb is not None:
        return kb * 1024 / 1e9
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9  # peak, not current
    except Exception:
        return None


def host_ram_peak_gb() -> Optional[float]:
    kb = _read_kb("/proc/self/status", "VmHWM:")
    if kb is not None:
        return kb * 1024 / 1e9
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9
    except Exception:
        return None


def host_ram_total_gb() -> Optional[float]:
    kb = _read_kb("/proc/meminfo", "MemTotal:")
    if kb is not None:
        return kb * 1024 / 1e9
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


@dataclass
class StreamProbe:
    """Where the stream is and how fast it moves.

    Dense runs read the trainer's ``BlockManager`` (``active_block_id``,
    ``num_blocks``, ``h2d_bytes_total``). MoE runs sum ``bytes_transferred``
    over every expert residency manager. The rate is the byte delta over
    wall time between two ``sample`` calls -- host-to-device bytes per second
    of training, not a link benchmark.
    """

    unit: str = "block"
    count: int = 0
    _bm: Any = None
    _experts: List[Any] = field(default_factory=list)
    _last_bytes: int = 0
    _last_t: float = 0.0
    rate_gbs: float = 0.0

    @classmethod
    def attach(cls, trainer: Any) -> Optional["StreamProbe"]:
        bm = getattr(trainer, "block_manager", None)
        if bm is not None and getattr(bm, "num_blocks", 0):
            p = cls(unit="block", count=int(bm.num_blocks), _bm=bm)
            p._last_bytes, p._last_t = p._bytes(), time.monotonic()
            return p
        experts = list(_expert_managers(getattr(trainer, "model", None)))
        if experts:
            layers = len(experts)
            p = cls(unit="layer", count=layers, _experts=experts)
            p._last_bytes, p._last_t = p._bytes(), time.monotonic()
            return p
        return None

    def _bytes(self) -> int:
        if self._bm is not None:
            return int(getattr(self._bm, "h2d_bytes_total", 0) or 0)
        return sum(int(getattr(e, "bytes_transferred", 0) or 0) for e in self._experts)

    def position(self) -> Optional[int]:
        """1-based index of the block on the GPU right now (dense), else None."""
        if self._bm is None:
            return None
        bid = getattr(self._bm, "active_block_id", None)
        return None if bid is None else int(bid) + 1

    def sample(self) -> float:
        """Update and return the transfer rate (GB/s) since the last sample."""
        now = time.monotonic()
        b = self._bytes()
        dt = now - self._last_t
        if dt > 0.05:
            self.rate_gbs = max(0.0, (b - self._last_bytes) / dt / 1e9)
            self._last_bytes, self._last_t = b, now
        return self.rate_gbs


def _expert_managers(model: Any) -> Iterable[Any]:
    if model is None or not hasattr(model, "modules"):
        return []
    found = []
    try:
        for m in model.modules():
            r = getattr(m, "residency", None)
            if r is not None and hasattr(r, "bytes_transferred"):
                found.append(r)
    except Exception:
        return []
    return found
