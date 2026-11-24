"""What `machine` reads on a Mac, and what a run reads about memory and disk.

Torch-free: versions come from package metadata and nothing imports the training stack, so this
answers in a moment and on a bare install. The readers go through libSystem, so they need no sudo.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import sys
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional

from poros import metal
from poros.cli.words.models import catalog
from poros.metal import store, table

Check = tuple[str, str, str]  # name, value, status: the rows `poros machine` prints
GB_PER_GIB = 2**30 / 1e9

_RUSAGE_INFO_V4 = 4
# struct rusage_info_v4 (<sys/resource.h>) in uint64 slots: ri_uuid takes the first two.
_RI_DISKIO_BYTESREAD, _RI_LIFETIME_MAX_PHYS_FOOTPRINT, _RI_SLOTS = 18, 30, 37


@cache
def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _rusage(slot: int) -> int:
    info = (ctypes.c_uint64 * _RI_SLOTS)()
    if _libc().proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, info) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage")
    return info[slot]


def footprint() -> int:
    """This process's lifetime max phys footprint, Metal memory included: counters can stay flat
    while RSS grows, and this does not."""
    return _rusage(_RI_LIFETIME_MAX_PHYS_FOOTPRINT)


def disk_read_bytes() -> int:
    """What this process has really read from disk. A read the page cache served does not count."""
    return _rusage(_RI_DISKIO_BYTESREAD)


def _sysctl(name: str, out: Any) -> Any:
    size = ctypes.c_size_t(ctypes.sizeof(out))
    no_new_value = (None, ctypes.c_size_t(0))
    if _libc().sysctlbyname(name.encode(), ctypes.byref(out), ctypes.byref(size), *no_new_value):
        raise OSError(ctypes.get_errno(), name)
    return out


_METAL = "/System/Library/Frameworks/Metal.framework/Metal"
_GRAPHICS = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
_OBJC = "/usr/lib/libobjc.A.dylib"


@cache
def gpu_working_set() -> Optional[int]:
    """What macOS lets this Mac's GPU hold, in bytes: the default device's
    ``recommendedMaxWorkingSetSize``, the figure ``torch.mps.recommended_max_memory()`` returns,
    asked of Metal itself so no torch loads. None off a Mac, or if Metal has no device."""
    if sys.platform != "darwin":
        return None
    try:
        objc, metal = ctypes.CDLL(_OBJC), ctypes.CDLL(_METAL)
        ctypes.CDLL(_GRAPHICS)  # a process with no window server gets a device once this is in
    except OSError:
        return None
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    metal.MTLCreateSystemDefaultDevice.argtypes = []
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    device = metal.MTLCreateSystemDefaultDevice()
    if not device or not send:
        return None
    size = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(send)
    release = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(send)
    found = size(device, objc.sel_registerName(b"recommendedMaxWorkingSetSize"))
    release(device, objc.sel_registerName(b"release"))
    return int(found) or None


def swap_used() -> int:
    # struct xsw_usage: xsu_total, xsu_avail, xsu_used, then pagesize and encrypted.
    return _sysctl("vm.swapusage", (ctypes.c_uint64 * 4)())[2]


def pressure() -> int:
    """1 normal, 2 warning, 4 critical."""
    return _sysctl("kern.memorystatus_vm_pressure_level", ctypes.c_int()).value


@dataclass(frozen=True)
class MemSample:
    """One memory reading, in bytes. None means this platform has no such figure."""

    allocated: Optional[int] = None  # tensors, as torch counts them
    driver: Optional[int] = None  # what the device's allocator holds
    footprint: Optional[int] = None  # this process's lifetime max phys footprint
    swap: Optional[int] = None  # swap in use, whole machine
    pressure: Optional[int] = None  # 1 normal, 2 warning, 4 critical


def memory_sample(device: Any) -> MemSample:
    """Read memory now. No sync: it never waits for the device."""
    allocated = driver = None
    if device.type == "mps":
        import torch

        allocated = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
    if sys.platform != "darwin":  # gate scripts run off a Mac too
        return MemSample(allocated, driver)
    return MemSample(allocated, driver, footprint(), swap_used(), pressure())


def _this_mac(here: str) -> list[Check]:
    gpu = table.this_mac()
    if gpu is None:
        return [("GPU", f"{platform.machine()}: Poros needs Apple silicon", "error")]
    memory = f"{gpu.total_gib * GB_PER_GIB:.1f} GB"
    if gpu.working_set_gib:
        memory += f", {gpu.working_set_gib * GB_PER_GIB:.1f} GB of it for the GPU"
    if gpu.working_set_gib and table.working_set(here)[1] == "estimated":
        memory += " (estimated: the GPU did not answer)"
    rows = [("GPU", gpu.name, "ok"), ("Unified memory", memory, "ok")]
    budget = metal.dev_memory_gb()
    if budget is not None:
        real = metal.real_memory_bytes() / 2**30
        rows.append(("Memory budget", f"{budget} GB of {real:.0f} GB, from "
                     f"{metal.DEV_MEMORY_ENV}: a test tool, planned and guarded as that", "warn"))
    elif metal.DEV_MEMORY_ENV in os.environ:
        rows.append(("Memory budget", f"{metal.DEV_MEMORY_ENV} ignored: it takes whole GB from 4 "
                     "to less than this Mac's own, and never a Mac a run measured", "warn"))
    if not metal.readable(here):
        return [*rows, ("This Mac", "its chip or memory could not be read", "error")]
    if metal.measured(here):
        return rows
    from poros.metal import selfcheck

    return [*rows, ("This Mac", "no run measured this Mac: its rows are derived from an M4 with "
                    "16 GB for its memory, and its speed is unknown until a run here", "info"),
            selfcheck.check_row(here)]


def _macos(here: str) -> list[Check]:
    version = metal.macos_version()
    build = metal.macos_build()
    shown = f"{platform.mac_ver()[0]} ({build})"
    if version < metal.MIN_MACOS or version in metal.REFUSED_MACOS:
        return [("macOS", f"{shown}: update macOS", "error")]
    rows = [("macOS", shown, "ok")]
    builds = (r.macos_build for e in catalog() for r in e.metal if r.device_class == here)
    measured = sorted(set(filter(None, builds)))
    if measured and build not in measured:
        then = ", ".join(measured)
        rows.append(("Times", f"measured on macOS {then}; this Mac runs {build}", "info"))
    return rows


def _stack() -> list[Check]:
    rows = []
    for name, want in metal.PINS.items():
        have = metal.installed(name)
        rows.append(
            (name, have, "ok")
            if have == want
            else (name, f"{have or 'not installed'}, needs {want}", "error")
        )
    jinja = metal.installed("jinja2")
    missing = "not installed (needed for HTML reports)"
    rows.append(("jinja2", jinja or missing, "ok" if jinja else "warn"))
    return rows


def _disks() -> list[Check]:
    """A warning only when no disk has room: one low disk among others stops nothing."""
    every = store.volumes()
    room = any(v.free > 20e9 for v in every)
    return [(f"Disk {v.label}", f"{v.free / 1e9:.1f} GB free", "ok" if room else "warn")
            for v in every]


def checks() -> tuple[list[Check], bool]:
    """(the Mac's rows, whether torch is installed). Live MPS is for `train` to check: it cannot be
    told without importing torch."""
    here = metal.device_class()
    names = {"olmoe": "OLMoE", "deepseek": "DeepSeek"}  # the rest capitalize as they are written
    trains = sorted({names.get(e.family, e.family.capitalize()) for e in catalog()
                     if table.rows(e.id, here)})
    return (
        _this_mac(here)
        + _macos(here)
        + _stack()
        + [("Models", ", ".join(trains) or "none on this Mac", "info")]
        + _disks(),
        metal.installed("torch") is not None,
    )


def memory_free_gib() -> Optional[float]:
    """What the machine can hand over without swapping, from vm_stat. For display only: whether a
    run fits never depends on it."""
    out = metal.run_tool("/usr/bin/vm_stat")
    page = re.search(r"page size of (\d+)", out)
    pages = re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)", out)
    return round(sum(map(int, pages)) * int(page.group(1)) / 2**30, 2) if page else None
