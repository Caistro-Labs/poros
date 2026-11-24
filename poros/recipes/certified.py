"""Caistro Labs certified recipes: which built-in recipe ran, at this release, on which machine,
and what that run measured.

The facts live in ``certified.json`` beside this file. Nobody types it: ``scripts/certify/
generate.py`` writes it from the receipts committed under ``docs/validation``, and a test fails
when an entry has no receipt in the tree or says anything its receipt does not. This module only
reads that file and compares a certified machine with this one. Torch-free.

A recipe is certified on this machine only when this is the model it ran on: the same card with
at least its host RAM, or a Mac of the same class. A different card, or Mac, with at least that
memory (the same platform; on CUDA at least its VRAM and host RAM) is ``same_memory_class``: it
should fit, and its time an update will differ. The word "certified" is never said of a machine
that is not the certified one.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CERTIFIER = "Caistro Labs"
SCHEMA = "poros.certified.v1"
DATA = Path(__file__).with_name("certified.json")
# Two sizes rounded to two places can differ by their rounding: a card reads 23.99 GiB on one
# probe and 23.985 on another. This much is never a claim about memory.
SLACK_GIB = 0.01
# Host RAM is read as MemTotal, which two machines sold with the same memory read up to a few
# hundred MB apart (what the firmware, the kernel or an integrated GPU keep back). 1 GiB covers
# that and is far under the step between sizes RAM is sold in (8 GiB and up).
HOST_SLACK_GIB = 1.0

# The units receipts write memory in, as bytes. A Poros receipt's ``*_gb`` fields are decimal GB.
_BYTES = {"GB": 1e9, "GiB": 2**30, "MB": 1e6, "MiB": 2**20, "KiB": 2**10, "kB": 2**10}


def to_gib(value: Optional[float], unit: str) -> Optional[float]:
    """``value`` in ``unit`` as GiB (2^30 bytes), to three places; None stays None. The one place a
    receipt's unit is turned into what Poros shows. ``kB`` is /proc's, which means KiB."""
    if value is None:
        return None
    if unit not in _BYTES:
        raise ValueError(f"unknown memory unit {unit!r}")
    return round(float(value) * _BYTES[unit] / 2**30, 3)



def _up(gib: float) -> float:
    """A memory figure to 0.1 rounded up, as every screen shows one (never below what was
    measured: poros.cli.ui.components.ceil1)."""
    return math.ceil(round(gib * 10, 6)) / 10

@dataclass(frozen=True)
class Machine:
    """A machine as certification and matching see it."""

    platform: str  # "cuda" or "mac"
    gpu: str  # the card's or the chip's name: "RTX 4090", "Apple M4 10-core GPU 16GB"
    vram_gib: Optional[float]  # the card's memory; a Mac's whole memory
    host_ram_gib: Optional[float]  # CUDA: the host's RAM; a Mac: None (one pool, in vram_gib)
    mac_class: str = ""  # a Mac's class, ``apple-m4-10gpu-16gb``
    os: str = ""  # the OS it ran on (``macOS 27.2.1``): shown, never matched

    def label(self) -> str:
        """``RTX 4090 · 24 GiB VRAM · 59.6 GiB RAM``, or ``Apple M4 10-core GPU 16GB``."""
        if self.platform == "mac":
            return self.gpu
        bits = [self.gpu, f"{self.vram_gib:.0f} GiB VRAM" if self.vram_gib else "",
                f"{self.host_ram_gib:.1f} GiB RAM" if self.host_ram_gib else ""]
        return " · ".join(b for b in bits if b)


# The memory figures, each by its own name: they must never be mistaken for each other.
# gpu_memory_used_gib is what "will it fit" reads: the card's memory the run's processes held over
# load and training, as nvidia-smi shows it (the CUDA context included). The torch peaks are its
# caching allocator's in training, what the release's published comparisons quote (reserved).
# A Mac has one pool and one figure, memory_peak_gib.
MEMORY = ("gpu_memory_used_gib", "torch_reserved_peak_gib", "torch_allocated_peak_gib",
          "memory_peak_gib", "host_ram_peak_gib")


@dataclass(frozen=True)
class Entry:
    """One certified run: a recipe on one machine, and what the run measured there."""

    recipe: str
    machine: Machine
    settings: Dict[str, Any]  # the settings that decide fit and speed, as the run resolved them
    loop: str
    data: Optional[str]  # the data's shape as the run read it; None where its receipt says none
    ran_path: Optional[str]  # the CUDA loop the run took (efficient, fast, sequential); Mac None
    ran_offload: Optional[bool]  # CUDA: whether its tables were in host RAM; Mac None
    updates: int  # optimizer steps the run measured
    s_per_step: float  # seconds an update, by s_per_step_rule
    s_per_step_rule: str  # how: "median of updates 2..N", or "mean of all updates" (a receipt's)
    gpu_memory_used_gib: Optional[float]  # CUDA: nvidia-smi's, per process, load and training
    torch_reserved_peak_gib: Optional[float]  # CUDA: torch's reserved peak in training
    torch_allocated_peak_gib: Optional[float]  # CUDA: torch's allocated peak in training
    memory_peak_gib: Optional[float]  # a Mac's one figure: its run's peak of the one pool
    host_ram_peak_gib: Optional[float]  # CUDA: resident memory that is not file-backed
    receipt: str  # its receipt, a path in the repository
    commit: str

    def fit_gib(self) -> Optional[float]:
        """The figure "will it fit" reads: the GPU memory the run used; on a Mac its peak."""
        return self.memory_peak_gib if self.machine.platform == "mac" else self.gpu_memory_used_gib

    def on_ssd(self) -> bool:
        """Its experts stream from a drive: its time an update depends on that drive."""
        return self.settings.get("store") == "ssd"

    def figures(self, time: bool = True) -> str:
        """``12.25 s an update · 5.6 GiB of GPU memory used (nvidia-smi) · 7.4 GiB host RAM``;
        without ``time``, the memory alone (a run on another drive: its time is that drive's)."""
        bits = [f"{self.s_per_step:.2f} s an update"] if time else []
        if self.machine.platform == "mac":
            if self.memory_peak_gib is not None:
                bits.append(f"{_up(self.memory_peak_gib):.1f} GiB of memory at the peak")
        else:
            if self.gpu_memory_used_gib is not None:
                bits.append(f"{_up(self.gpu_memory_used_gib):.1f} GiB of GPU memory used "
                            "(nvidia-smi)")
            if self.host_ram_peak_gib is not None:
                bits.append(f"{_up(self.host_ram_peak_gib):.1f} GiB host RAM (no file pages)")
        return " · ".join(bits)

    def torch_figures(self) -> str:
        """``torch in training: 20.5 GiB reserved, 19.6 GiB allocated``; "" where none."""
        r, a = self.torch_reserved_peak_gib, self.torch_allocated_peak_gib
        if r is None and a is None:
            return ""
        bits = ([f"{_up(r):.1f} GiB reserved"] if r is not None else []) + (
            [f"{_up(a):.1f} GiB allocated"] if a is not None else [])
        return "torch in training: " + ", ".join(bits)

    def fields(self) -> Dict[str, Any]:
        """What an agent reads: every fact, as fields."""
        m = self.machine
        return {"machine": {"platform": m.platform, "gpu": m.gpu, "vram_gib": m.vram_gib,
                            "host_ram_gib": m.host_ram_gib, "mac_class": m.mac_class or None,
                            "os": m.os or None, "label": m.label()},
                "settings": dict(self.settings), "loop": self.loop, "data": self.data,
                "ran_path": self.ran_path, "ran_offload": self.ran_offload,
                "updates": self.updates, "s_per_step": self.s_per_step,
                "s_per_step_rule": self.s_per_step_rule,
                **{k: getattr(self, k) for k in MEMORY},
                "figures": self.figures(), "torch_figures": self.torch_figures(),
                "receipt": self.receipt, "commit": self.commit}


# What a text field of the file may hold before it reaches a screen: letters, digits, spaces and
# plain marks. No control, escape or bidi character passes (\w holds none).
TEXT = re.compile(r"[\w .,:;()/+#%·_-]{0,160}")


def _text(raw: Any) -> str:
    value = "" if raw is None else raw
    if not isinstance(value, str) or not TEXT.fullmatch(value):
        raise ValueError("text")
    return value


def _machine(raw: Dict[str, Any]) -> Machine:
    platform = raw["platform"]
    if platform not in ("cuda", "mac"):
        raise ValueError("platform")
    return Machine(platform=platform, gpu=_text(raw["gpu"]), vram_gib=_num(raw.get("vram_gib")),
                   host_ram_gib=_num(raw.get("host_ram_gib")),
                   mac_class=_text(raw.get("mac_class")), os=_text(raw.get("os")))


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if type(v) not in (int, float):
        raise ValueError("number")
    return float(v)


def _entry(key: str, raw: Dict[str, Any]) -> Entry:
    settings = dict(raw["settings"])
    for v in [*settings.values(), *(x for v in settings.values() if isinstance(v, dict)
                                    for x in v.values())]:
        if isinstance(v, str):
            _text(v)
        elif type(v) not in (int, float, bool, dict):
            raise ValueError("setting")
    offload = raw.get("ran_offload")
    if offload is not None and not isinstance(offload, bool):
        raise ValueError("ran_offload")
    return Entry(recipe=_text(key), machine=_machine(raw["machine"]), settings=settings,
                 loop=_text(raw["loop"]), data=_text(raw.get("data")) or None,
                 ran_path=_text(raw.get("ran_path")) or None, ran_offload=offload,
                 updates=int(raw["updates"]), s_per_step=float(raw["s_per_step"]),
                 s_per_step_rule=_text(raw["s_per_step_rule"]),
                 **{k: _num(raw.get(k)) for k in MEMORY},
                 receipt=_text(raw["receipt"]), commit=_text(raw["commit"]))


@dataclass(frozen=True)
class Book:
    """The shipped file, read: the release it certifies, each recipe's entries, what is pending."""

    release: str
    commit: str
    entries: Dict[str, Tuple[Entry, ...]]
    pending: Dict[str, Tuple[str, ...]]  # recipe -> receipts its sources name that are not in yet
    dropped: Tuple[str, ...] = ()  # built-ins the release ships without: no certified machine


def parse(raw: Any) -> Book:
    """The file's content as a Book; ValueError for anything not of its shape."""
    try:
        if raw.get("schema") != SCHEMA:
            raise ValueError("schema")
        entries = {k: tuple(_entry(k, e) for e in v) for k, v in raw["recipes"].items()}
        pending = {k: tuple(str(p) for p in v) for k, v in raw.get("pending", {}).items()}
        dropped = tuple(str(k) for k in raw.get("dropped", []))
        return Book(str(raw["release"]), str(raw["commit"]), entries, pending, dropped)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"not a certified-recipes file: {exc}") from None


@lru_cache(maxsize=1)
def book() -> Book:
    """The shipped certifications. A file that is missing or not of its shape certifies nothing:
    no recipe is ever said to be certified on a reading that failed."""
    try:
        return parse(json.loads(DATA.read_text()))
    except (OSError, ValueError, OverflowError, RecursionError):  # a malformed file: none
        return Book("", "", {}, {})


def entries(key: str) -> Tuple[Entry, ...]:
    return book().entries.get(key, ())


def status(key: str) -> str:
    """``certified`` (it has a certified machine), ``pending`` (its runs are not in yet), or
    ``none`` (not a recipe Caistro Labs certifies: yours, a project's, a shared one)."""
    if entries(key):
        return "certified"
    return "pending" if key in book().pending else "none"


def host_ram_total_gib() -> Optional[float]:
    """This machine's RAM, GiB: MemTotal on Linux, the physical pages elsewhere."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return to_gib(int(line.split()[1]), "kB")
    except (OSError, ValueError, IndexError):
        pass
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 3)
    except (ValueError, OSError, AttributeError):
        return None


def gpu_name(raw: str) -> str:
    """A card's name without its maker's words: ``NVIDIA GeForce RTX 4090`` -> ``RTX 4090``."""
    name = " ".join(str(raw).split())
    for word in ("NVIDIA ", "GeForce "):
        name = name.replace(word, "")
    return name.replace(" (simulated)", "").strip()


def here(ctx: Any) -> Optional[Machine]:
    """This machine, from what Poros reads of it; None where there is no GPU, or the GPU is one
    named for a plan (``--gpu``), whose host RAM nothing here knows."""
    gpu = ctx.gpu
    if gpu is None or getattr(ctx, "gpu_name", None) is not None:
        return None
    if ctx.unified:
        from poros.metal.table import class_of

        return Machine("mac", gpu.name, float(gpu.total_gib), None, class_of(gpu.name))
    return Machine("cuda", gpu_name(gpu.name), float(gpu.total_gib), host_ram_total_gib(),
                   os="WSL2" if under_wsl() else "")


def under_wsl() -> bool:
    """This Linux is WSL2 (the product's own test, poros.sss.pinning): CUDA there can spill into
    system memory, so a native run's figures do not speak for it."""
    from poros.sss.pinning import _wsl

    return _wsl()


def covers(machine: Machine, this: Machine) -> bool:
    """Whether ``this`` machine has at least ``machine``'s memory where fit is decided: the same
    platform; on CUDA at least its VRAM and host RAM; on a Mac at least its memory."""
    if machine.platform != this.platform or machine.vram_gib is None or this.vram_gib is None:
        return False
    if this.vram_gib + SLACK_GIB < machine.vram_gib:
        return False
    if machine.platform == "mac":
        return True
    return (machine.host_ram_gib is not None and this.host_ram_gib is not None
            and this.host_ram_gib + HOST_SLACK_GIB >= machine.host_ram_gib)


def short_of(found: Tuple[Entry, ...], this: Optional[Machine]) -> Optional[str]:
    """Why ``this`` machine, on a certified machine's card, is not certified: its host RAM is
    less (beyond the slack). None where no certified machine has this card, or it is not that."""
    if this is None or this.platform != "cuda" or this.host_ram_gib is None:
        return None
    for e in found:
        m = e.machine
        if (m.platform == "cuda" and m.gpu == this.gpu and m.host_ram_gib is not None
                and this.host_ram_gib + HOST_SLACK_GIB < m.host_ram_gib):
            return (f"this machine has the {m.gpu} it was certified on, but {this.host_ram_gib:.1f}"
                    f" GiB of host RAM, and the certified run's machine had {m.host_ram_gib:.1f}")
    return None


def same_model(machine: Machine, this: Machine) -> bool:
    """``this`` is the certified machine's model: the same card (a Mac: the same chip, GPU cores
    and memory), with at least its host RAM."""
    if not covers(machine, this):
        return False
    if machine.platform == "mac":
        return machine.mac_class == this.mac_class
    if "wsl" in this.os.lower() and "wsl" not in machine.os.lower():
        return False  # the card under WSL2 is not the machine a native run measured
    return machine.gpu == this.gpu


CERTIFIED, SAME_MEMORY = "certified", "same_memory_class"


def match(found: Tuple[Entry, ...], this: Optional[Machine]) -> Optional[Tuple[Entry, str]]:
    """The certified run that speaks for ``this`` machine, and how: ``certified`` where this is its
    machine's model, else ``same_memory_class`` where this has at least the memory of one (the
    largest such). None: this machine has less than every one of them, or cannot be read."""
    if this is None:
        return None
    same = [e for e in found if same_model(e.machine, this)]
    if same:
        return same[0], CERTIFIED
    ok = [e for e in found if covers(e.machine, this)]
    if not ok:
        return None
    best = max(ok, key=lambda e: (e.machine.vram_gib or 0, e.machine.host_ram_gib or 0))
    return best, SAME_MEMORY


def machines(key: str) -> List[str]:
    """Each machine a recipe is certified on, by its label, in the file's order, once each."""
    seen: List[str] = []
    for e in entries(key):
        if e.machine.label() not in seen:
            seen.append(e.machine.label())
    return seen


def cards(key: str) -> List[str]:
    """The cards a recipe is certified on, once each: ``RTX 4090``, ``Apple M4 ... 16GB``."""
    names: List[str] = []
    for e in entries(key):
        if e.machine.gpu not in names:
            names.append(e.machine.gpu)
    return names
