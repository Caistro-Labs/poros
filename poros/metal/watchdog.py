"""On one pool, too little memory looks like a slow machine, not an error.

A timer thread and not a check at each step boundary: a swapping step has no upper bound, so a check
there would run least when it is needed. What would hurt the machine sets ``fired``, and the loop
checkpoints and stops at its next step boundary. If no boundary comes in time, the process exits,
and the last periodic checkpoint is what ``poros runs continue`` goes on from. It reads and never
touches a tensor, so a run it does not stop is the same run. ``memory_guard=false`` turns it off.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

from poros import metal
from poros.core.errors import EXIT_CODES, PorosDiskError, PorosOOMError
from poros.metal import machine

CRITICAL = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical
# The figures below are the 16 GB reference's, and another Mac's are the same share of its memory.
REFERENCE_MEMORY = 16 * 2**30
# Swap is the whole machine's, and macOS moves a little aside by itself. Growth of 24, 26 and 51 MB
# left a run's peak where a run without it had it. At 483 and 740 MB the peak read 10% and 23% low.
SPILL_BYTES = 250_000_000
# What stops a run. Growth up to about 750 MB did no harm here. Swap lives on the boot disk whatever
# drive the run uses, so that disk has a floor too, the same on every Mac: it keeps the disk from
# filling, and whichever stop comes first stops the run.
SWAP_STOP_BYTES = 2_000_000_000
BOOT_STOP_BYTES = 3_000_000_000
BOOT_DATA = Path("/System/Volumes/Data")  # where the boot disk's writes go; "/" is sealed
INTERVAL_S = 5.0
STEP_WAIT_S = 600.0


def _share(figure: int, memory: int) -> int:
    return figure if memory == REFERENCE_MEMORY else round(figure * memory / REFERENCE_MEMORY)


def spill_bytes(memory: int) -> int:
    """Swap growth at which a run's memory figures read low, on a Mac with this memory."""
    return _share(SPILL_BYTES, memory)


def swap_stop_bytes(memory: int) -> int:
    """Swap growth that stops a run: an eighth of memory, 2 GB on 16 GB."""
    return _share(SWAP_STOP_BYTES, memory)


def boot_stop_bytes(memory: int) -> int:
    """The boot disk's floor, 3 GB on any Mac: a larger one would stop a large Mac whose disk
    simply has less free than its share of memory, before any swap grew."""
    return BOOT_STOP_BYTES


def this_memory() -> int:
    """This Mac's memory as every plan reads it, the reference's if it cannot be read."""
    return metal.memory_bytes() or REFERENCE_MEMORY


def _boot_free() -> int:
    return shutil.disk_usage("/").free


class Watchdog:
    def __init__(self, guard: bool = True):
        self.fired = threading.Event()
        self.why = ""  # what fired it, as the start of a sentence
        self._guard = guard
        self._stopped = threading.Event()
        memory = this_memory()
        self.swap_stop, self.boot_stop = swap_stop_bytes(memory), boot_stop_bytes(memory)
        self._swap_start = self._swap_peak = machine.swap_used()
        threading.Thread(target=self._watch, name="poros-watchdog", daemon=True).start()

    @property
    def swap_growth(self) -> int:
        """Bytes swap grew by in this run. At ``spill_bytes``, its memory figures read low."""
        return self._swap_peak - self._swap_start

    def stop(self) -> None:
        self._stopped.set()

    def _over(self) -> str:
        if machine.pressure() == CRITICAL:
            return "This Mac ran out of memory"
        if self.swap_growth > self.swap_stop:
            return f"Swap grew by {self.swap_growth / 1e9:.1f} GB"
        if (free := _boot_free()) < self.boot_stop:
            return f"The boot disk, where swap lives, fell to {free / 1e9:.1f} GB free"
        return ""

    def _watch(self) -> None:
        while not self._stopped.wait(INTERVAL_S):
            self._swap_peak = max(self._swap_peak, machine.swap_used())
            self.why = self._over() if self._guard else ""
            if self.why:
                self.fired.set()
                if not self._stopped.wait(STEP_WAIT_S):
                    print(
                        f"{self.why}, and the step did not finish. {_next_steps(self.why)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(EXIT_CODES["E_OOM"])
                return


def _next_steps(why: str) -> str:
    first = "Free some space on it" if why.startswith("The boot disk") else "Close other apps"
    return (
        f"{first}, then `poros runs continue`. If it stops again, start a new run with a shorter "
        "--seq-len or a lower --rank, since a run keeps its settings. memory_guard=false turns "
        "this check off."
    )


def checkpoint_first(trainer: Any) -> None:
    """Keep the run if that is safe: not onto a boot disk already under its floor, and a disk that
    cannot take it leaves the last checkpoint standing. The stop is what the user needs to hear."""
    from poros.metal import store

    on_boot = store._same_disk(Path(trainer.config.output_dir), BOOT_DATA)
    if on_boot and _boot_free() < boot_stop_bytes(this_memory()):
        return
    try:
        trainer._save_checkpoint()
    except PorosDiskError:
        pass


def is_mps_oom(exc: BaseException) -> bool:
    """Both forms torch raises on ``mps``: the allocator's plain RuntimeError, and
    OutOfMemoryError from a failed command buffer, which arrives at the next sync."""
    import torch

    return isinstance(exc, torch.OutOfMemoryError) or "MPS backend out of memory" in str(exc)


def oom_error(trainer: Any, why: str = "This Mac ran out of memory") -> PorosOOMError:
    """The error for a Mac out of memory, or nearly. No block_size advice: halving it does not
    relieve one shared pool."""
    import torch

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._release_retained_forward_blocks()
    torch.mps.empty_cache()
    return PorosOOMError(f"{why} at step {trainer.global_step}.", remediation=_next_steps(why))
