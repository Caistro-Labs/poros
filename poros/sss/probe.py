"""Measure the two bandwidths an SSD store runs at (about 3 s, cached).

- ``B_ssd``: random 8 MiB ``O_DIRECT`` reads at queue depth 32 (one thread
  per outstanding read, each into its own page-aligned buffer), from an
  existing ``.sss`` file or a temporary probe file written next to where the
  store will live. Buffered reads (with the page cache dropped first) when
  the filesystem refuses ``O_DIRECT``; the result says which.
- ``B_h2d``: pinned host to device copies (needs torch and CUDA; ``None``
  otherwise).

Results are cached in ``~/.cache/poros/sss_probe.json`` per directory and
device so later runs skip the measurement. The disk probe needs no torch:
the plan stays torch-free.
"""

from __future__ import annotations

import json
import mmap
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from poros.sss.files import new_file  # not poros.sss.format: that needs torch, a plan does not

__all__ = ["ProbeResult", "cache_path", "cached", "probe", "probe_disk", "probe_h2d"]

_ALIGN = 4096
_PROBE_NAME = ".poros_sss_probe.tmp"


def cache_path() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(root) / "poros" / "sss_probe.json"


@dataclass
class ProbeResult:
    """Measured bandwidths in bytes/s; ``b_h2d`` is ``None`` without a GPU."""

    directory: str
    b_ssd: float
    direct: bool
    b_h2d: float | None = None
    device: str | None = None
    fallback_reason: str | None = None
    read_bytes: int = 8 << 20
    queue_depth: int = 32
    seconds: float = 0.0
    measured_at: float = 0.0
    cached: bool = False
    # One read in flight at a time (bytes/s): the latency the prefetch depth hides.
    b_ssd_qd1: float | None = None
    # Largest page-locked registration that succeeded when it was below what
    # was tried (a ceiling, e.g. ~1 GiB in a WSL2 container); None: none found.
    pin_ceiling: int | None = None

    @property
    def bandwidth(self) -> float:
        """Effective streaming bandwidth ``min(B_ssd, B_h2d)``."""
        return self.b_ssd if self.b_h2d is None else min(self.b_ssd, self.b_h2d)


def _open(path: Path, direct: bool) -> tuple[int, bool, str | None]:
    if direct and hasattr(os, "O_DIRECT"):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                os.preadv(fd, [memoryview(mmap.mmap(-1, _ALIGN))], 0)
                return fd, True, None
            except OSError as e:
                os.close(fd)
                reason = f"O_DIRECT read refused: {e.strerror or e}"
        except OSError as e:
            reason = f"O_DIRECT open refused: {e.strerror or e}"
    else:
        reason = "O_DIRECT not requested" if not direct else "O_DIRECT unavailable"
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    return fd, False, reason


def _write_probe_file(path: Path, nbytes: int) -> None:
    block = os.urandom(8 << 20)
    with os.fdopen(new_file(path, os.O_WRONLY), "wb") as f:
        left = nbytes
        while left > 0:
            n = min(left, len(block))
            f.write(block[:n])
            left -= n
        f.flush()
        os.fsync(f.fileno())


def probe_disk(target: str | os.PathLike, *, seconds: float = 2.0, queue_depth: int = 32,
               read_bytes: int = 8 << 20, file_bytes: int = 1 << 30,
               direct: bool = True) -> ProbeResult:
    """Random-read bandwidth of the disk holding ``target`` (a ``.sss`` file or a directory).

    A directory without a ``.sss`` file gets a temporary probe file of
    ``file_bytes``, deleted afterwards.
    """
    target = Path(target)
    directory = _directory(target)
    path = target if target.is_file() else None
    if path is None:
        existing = sorted(directory.glob("*.sss"), key=lambda p: p.stat().st_size, reverse=True)
        path = existing[0] if existing and existing[0].stat().st_size >= 2 * _ALIGN else None
    temp = None
    if path is None:
        directory.mkdir(parents=True, exist_ok=True)
        temp = path = directory / _PROBE_NAME
        _write_probe_file(path, max(int(file_bytes), 2 * _ALIGN))
    try:
        size = path.stat().st_size
        read_bytes = max(_ALIGN, min(int(read_bytes), (size // 2) // _ALIGN * _ALIGN))
        last = (size - read_bytes) // _ALIGN
        fd, is_direct, reason = _open(path, direct)
        try:

            def measure(qd: int, secs: float) -> tuple[float, float]:
                total = [0]
                lock = threading.Lock()
                deadline = time.perf_counter() + secs

                def worker(seed: int) -> None:
                    rng = random.Random(seed)
                    buf = mmap.mmap(-1, read_bytes)
                    view = memoryview(buf)  # both are freed with this frame
                    done = 0
                    while time.perf_counter() < deadline:
                        off = rng.randint(0, last) * _ALIGN
                        mv, pos = view, off
                        while mv:
                            n = os.preadv(fd, [mv], pos)
                            if n <= 0:
                                break
                            mv, pos = mv[n:], pos + n
                        done += read_bytes - len(mv)
                        if not is_direct:
                            os.posix_fadvise(fd, off, read_bytes, os.POSIX_FADV_DONTNEED)
                    with lock:
                        total[0] += done

                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=qd) as pool:
                    list(pool.map(worker, range(qd)))
                elapsed = time.perf_counter() - t0
                return total[0] / elapsed, elapsed

            b_ssd, elapsed = measure(queue_depth, seconds)
            # One read at a time: the per-read latency the prefetch depth must hide.
            b_qd1, _ = measure(1, min(0.5, seconds / 4))
        finally:
            os.close(fd)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return ProbeResult(directory=str(directory.resolve()), b_ssd=b_ssd, b_ssd_qd1=b_qd1,
                       direct=is_direct, fallback_reason=None if is_direct else reason,
                       read_bytes=read_bytes, queue_depth=queue_depth, seconds=elapsed,
                       measured_at=time.time())


def probe_h2d(nbytes: int = 256 << 20, iters: int = 5) -> tuple[float | None, str | None]:
    """Pinned host to device bandwidth (bytes/s) and the device name; ``(None, None)``
    without torch or CUDA."""
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dev.copy_(host, non_blocking=True)  # warm up
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1e3
    del host, dev
    empty = getattr(torch._C, "_accelerator_emptyHostCache", None) or getattr(
        torch._C, "_host_emptyCache", None)
    if empty is not None:  # give the pinned block back to the driver
        empty()
    return nbytes * iters / seconds, torch.cuda.get_device_name()


def _directory(target: Path) -> Path:
    """The store directory: ``target`` itself unless it names a file (existing or ``.sss``)."""
    return target.parent if target.is_file() or target.suffix == ".sss" else target


def _cache_key(directory: Path) -> str:
    st = os.stat(directory)
    return f"{directory.resolve()}|dev={st.st_dev}"


def cached(target: str | os.PathLike) -> ProbeResult | None:
    """The cached probe of ``target``'s store directory, measured by an earlier ``probe`` (an
    SSD-store run); None where there is none. Measures nothing,
    writes nothing, creates no directory: what a plan shown while a card is filled reads."""
    directory = _directory(Path(target))
    try:
        hit = json.loads(cache_path().read_text()).get(_cache_key(directory))
        return ProbeResult(**{**hit, "cached": True}) if isinstance(hit, dict) else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def probe(target: str | os.PathLike, *, use_cache: bool = True, h2d: bool = True,
          **disk_kwargs) -> ProbeResult:
    """``probe_disk`` plus ``probe_h2d``, cached per directory and device."""
    target = Path(target)
    directory = _directory(target)
    directory.mkdir(parents=True, exist_ok=True)
    key = _cache_key(directory)
    cache = cache_path()
    entries: dict = {}
    result = None
    if use_cache:
        try:
            entries = json.loads(cache.read_text())
        except (OSError, ValueError):
            entries = {}
        hit = entries.get(key)
        if isinstance(hit, dict):
            try:
                result = ProbeResult(**{**hit, "cached": True})
            except TypeError:
                result = None
    if result is None or result.b_ssd_qd1 is None:  # none yet, or cached before QD1
        result = probe_disk(target, **disk_kwargs)
    if h2d and result.b_h2d is None:  # e.g. cached on a host without a GPU
        result.b_h2d, result.device = probe_h2d()
    if use_cache:
        entries[key] = {k: v for k, v in asdict(result).items() if k != "cached"}
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(entries, indent=1))
            os.replace(tmp, cache)
        except OSError:
            pass
    return result
