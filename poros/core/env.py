"""What Poros asks of the process it runs in: the tools on a safe PATH, whether an agent
drives it, the CPU threads torch may use, and the host RAM that is free. Torch-free."""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

SYSTEM_BIN = "/usr/bin"
_CGROUP = Path("/sys/fs/cgroup")
_PROC_CGROUP = Path("/proc/self/cgroup")


@lru_cache(maxsize=16)
def _first_outside(name: str, path: str, here: str) -> str:
    for entry in path.split(os.pathsep):
        real = os.path.realpath(entry)
        # "" and "." are the working folder, and a folder inside it is the project's too
        if not os.path.isabs(entry) or real == here or real.startswith(here + os.sep):
            continue
        full = os.path.join(entry, name)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    return os.path.join(SYSTEM_BIN, name)


def outside_tool(name: str) -> str:
    """A system tool's absolute path, found once. Run by name, a ``.`` or an empty PATH entry finds
    the working folder's copy, and a project can put one there. With none anywhere this is where it
    would be, and running it fails as a missing tool does."""
    try:
        here = os.path.realpath(os.getcwd())
    except OSError:  # a working folder that has gone holds nothing
        here = os.devnull
    return _first_outside(name, os.environ.get("PATH", os.defpath), here)


def under_agent() -> bool:
    """Whether `poros agents` started this process or one above it: it sets POROS_AGENT, and a
    child inherits that."""
    return bool(os.environ.get("POROS_AGENT"))


# What the Hub's servers send with an answer to a request that carried no token, and
# huggingface_hub logs as a warning (an X-HF-Warning header, in huggingface_hub.utils._http).
_HUB_NOTICE = "You are sending unauthenticated requests to the HF Hub"


class _NoHubNotice(logging.Filter):
    """Drops that one notice, by its words; every other record passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _HUB_NOTICE not in record.getMessage()


def quiet_hub_notice() -> None:
    """The Hub's "unauthenticated requests" notice, kept off a run's screen. Poros asks the Hub
    about a public model with no token on purpose (its owner learns nothing of whoever asked:
    ``poros.plan.configs``), so the notice follows every run, a logged-in user's too, and
    nothing they do stops it. A filter on the one logger that says it, for that message only;
    a rate limit or a refused request still raises as it did."""
    log = logging.getLogger("huggingface_hub.utils._http")
    if not any(isinstance(f, _NoHubNotice) for f in log.filters):
        log.addFilter(_NoHubNotice())


class _NoCpuKernelAdvice(logging.Filter):
    """bitsandbytes, as it imports on a CPU with AVX512-BF16, warns that its optional CPU kernel
    for 4-bit matmuls is not installed ("pip install kernels"). Poros trains NF4 on CUDA and
    runs no 4-bit matmul on the CPU, so that advice is noise wherever Poros runs; every other
    record of that logger passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith("Failed to load CPU gemm_4bit_forward")


def quiet_libraries() -> None:
    """The two library notices nothing a Poros user does can act on, kept off the screen for
    the CLI and the Python API alike: bitsandbytes' CPU-kernel advice and the Hub's
    unauthenticated-requests notice. Each by its own words on its own logger; once."""
    log = logging.getLogger("bitsandbytes.backends.cpu.ops")
    if not any(isinstance(f, _NoCpuKernelAdvice) for f in log.filters):
        log.addFilter(_NoCpuKernelAdvice())
    quiet_hub_notice()


def outside_words_out() -> bool:
    """Whether outside words stay out of a result: a csv's first line, a chat template's failure
    text, a library's error text. Under `poros agents`, and in a job, whose error an agent can read
    later. The whole of them goes to stderr or the log; a result has type names and our words."""
    return under_agent() or bool(os.environ.get("POROS_JOB"))


def safe_path() -> str:
    """PATH without a relative or empty entry, or one in the working folder. Whatever spawns a tool
    by name finds it there, a library's bare `uname` too, and a project's node_modules/.bin is a
    project's. That takes a venv in the project off PATH with the rest, which costs nothing: Poros
    spawns itself by sys.executable, a full path that holds, and nothing it spawns needs that bin.
    A home folder or above one is no project: ~/.local/bin stays, and / keeps /usr/bin."""
    try:
        here = os.path.realpath(os.getcwd())
    except OSError:  # a working folder that has gone holds nothing
        here = None
    try:
        if here and Path.home().resolve().is_relative_to(here):
            here = None
    except RuntimeError:  # no home to know of, as in a bare container: the folder is still a folder
        pass

    def fine(entry: str) -> bool:
        if not os.path.isabs(entry):
            return False
        return here is None or not Path(os.path.realpath(entry)).is_relative_to(here)

    kept = filter(fine, os.environ.get("PATH", os.defpath).split(os.pathsep))
    return os.pathsep.join(kept) or os.defpath  # an empty PATH would mean the working folder


def _cgroup_cpu_quota() -> Optional[float]:
    """The tightest CPU quota, in cores, over this process's cgroup and its
    ancestors (cgroup v2 ``cpu.max``, or the v1 CFS quota); None when unset."""
    try:
        cgroups = _PROC_CGROUP.read_text().splitlines()
    except OSError:
        return None
    quotas = []
    for line in cgroups:  # "0::/path" (v2), "3:cpu,cpuacct:/path" (v1)
        hierarchy, controllers, rel = line.split(":", 2)
        if hierarchy != "0" and "cpu" not in controllers.split(","):
            continue
        root = _CGROUP if hierarchy == "0" else _CGROUP / "cpu"
        d = root / rel.lstrip("/")
        while True:
            try:
                if hierarchy == "0":
                    quota, period = (d / "cpu.max").read_text().split()
                else:
                    quota = (d / "cpu.cfs_quota_us").read_text()
                    period = (d / "cpu.cfs_period_us").read_text()
                if quota.strip() not in ("max", "-1"):
                    quotas.append(int(quota) / int(period))
            except (OSError, ValueError, ZeroDivisionError):  # absent here
                pass
            if d == root:
                break
            d = d.parent
    return min(quotas, default=None)


def cpu_threads(torch_default: int) -> Tuple[int, str]:
    """The intra-op thread count Poros runs torch with, and where it came from.

    torch starts one thread per core it sees, but a container's cgroup CPU
    quota can be far lower (7.65 cores on a 96-core RunPod host), and the
    excess threads are throttled: 13.2 vs 5.6 s/step, same losses.
    ``POROS_NUM_THREADS`` sets the count, an explicit ``OMP_NUM_THREADS`` is
    left as torch read it, and otherwise ``torch_default`` is capped at the
    affinity mask and the floor of the CPU quota, so the pool only shrinks.
    """
    if value := os.environ.get("POROS_NUM_THREADS"):
        try:
            n = int(value)
        except ValueError:
            n = 0
        if n < 1:
            from poros.core.errors import PorosConfigError

            raise PorosConfigError(
                f"POROS_NUM_THREADS must be a positive integer, got {value!r}.",
                remediation="Unset it to size torch's thread pool from the CPU quota.",
            )
        return n, "POROS_NUM_THREADS"
    if os.environ.get("OMP_NUM_THREADS"):
        return torch_default, "OMP_NUM_THREADS"
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None and quota < cpus:
        return min(torch_default, max(1, int(quota))), f"cgroup CPU quota {quota:g}"
    return min(torch_default, cpus), f"{cpus} usable CPUs"


def configure_torch_threads() -> int:
    """Size torch's intra-op thread pool with :func:`cpu_threads`; return it."""
    import torch

    threads, source = cpu_threads(torch.get_num_threads())
    if threads != torch.get_num_threads():
        logger.info("torch threads: %d (%s)", threads, source)
        torch.set_num_threads(threads)
    return threads


# cgroup v2, then v1: (mount, limit file, usage file, memory.stat keys of
# page cache, which the kernel reclaims before it OOM-kills).
_CGROUPS = (
    ("/sys/fs/cgroup", "memory.max", "memory.current", ("active_file", "inactive_file")),
    ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes",
     ("total_active_file", "total_inactive_file")),
)


def host_ram_free() -> Optional[float]:
    """GiB of host RAM this process can still get: MemAvailable, capped by the
    room left under every memory limit of its cgroup and that cgroup's
    ancestors (limit minus usage, plus reclaimable page cache), so a 46 GiB
    container on a 1 TiB host reads 46 GiB, not 1 TiB. None off Linux and macOS."""
    if sys.platform == "darwin":
        from poros.metal.machine import memory_free_gib

        return memory_free_gib()
    try:
        meminfo = Path("/proc/meminfo").read_text().split("MemAvailable:")[1]
        room = [int(meminfo.split()[0]) * 1024]
        cgroups = Path("/proc/self/cgroup").read_text().splitlines()
    except (OSError, IndexError, ValueError):
        return None
    for line in cgroups:  # "0::/path" (v2), "4:memory:/path" (v1)
        hierarchy, controllers, rel = line.split(":", 2)
        if hierarchy != "0" and "memory" not in controllers.split(","):
            continue
        mount, limit, usage, cache = _CGROUPS[hierarchy != "0"]
        d = Path(mount + rel)
        while True:
            try:
                stat = dict(row.split() for row in (d / "memory.stat").read_text().splitlines())
                room.append(int((d / limit).read_text()) - int((d / usage).read_text())
                            + sum(int(stat.get(key, 0)) for key in cache))
            except (OSError, KeyError, ValueError):  # absent, or "max": no limit here
                pass
            if d == Path(mount):
                break
            d = d.parent
    return round(min(room) / 2**30, 2)
