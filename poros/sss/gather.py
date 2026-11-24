"""In-place slabs: each slab gathered from byte ranges of existing files, no second copy.

A ``.sss`` file holds every slab contiguously, which costs a full copy of the model on
disk. When a model's files already hold every slab byte uncompressed at computable
offsets (``torch.save`` zip entries are stored, 64-byte aligned; a
``safetensors`` file is a header plus raw tensors), a slab can instead be read in place
from those files. A small sidecar index (``GatherIndex``, JSON, kilobytes to megabytes)
says, per slab, which ranges of which files fill which bytes of the slab:

- a **segment** ``(file, offset, nbytes, slab_offset)`` copies ``nbytes`` bytes from
  ``file`` at ``offset`` to ``slab_offset``; an optional ``(src_dtype, dst_dtype)`` pair
  widens float16 / bfloat16 values to float32 on the way. That is exact and unique for
  every non-NaN value; a NaN's float32 payload depends on torch's code path (vectorized
  or not, so on where a chunk splits the range), so an index builder must not emit a
  conversion over NaNs (the checksum backs that rule);
- bytes no segment covers (alignment padding) are zero;
- every slab carries the checksum (``CHECKSUM_ALGO``) of its assembled bytes, verified on
  its first read in this process, exactly as a ``.sss`` slab is.

``GatherSource`` is ``SsdSource`` with the per-chunk read replaced: O_DIRECT reads of the
4 KiB-aligned superset of the needed ranges (neighbouring ranges of one file merged into
one read) into a per-thread page-aligned scratch buffer, then a slice copy into the ring,
planned once per chunk (``plan_gather``) and run in one native call (``_gather.c``,
compiled on first use, GIL released) or, where no C compiler is available, in Python.
The ring, prefetch window, schedule replay and router hint are ``SsdSource``'s, unchanged.

The source files are only ever opened read-only; the index must live outside their
directory (``GatherIndex.save`` refuses a path inside ``root``).
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import mmap
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from poros.sss.format import ALIGNMENT, CHECKSUM_ALGO, SssCorruptError, SssFormatError
from poros.sss.source import SlabKey
from poros.sss.ssd import DEFAULT_CHUNK_BYTES, SsdSource, _Read

logger = logging.getLogger(__name__)

__all__ = ["GatherIndex", "GatherIndexStale", "GatherSlab", "GatherSource", "Segment",
           "gather_into", "native_gather", "plan_gather"]

FORMAT = "poros.sss.gather"
VERSION = 1
MERGE_GAP = 64 << 10  # ranges of one file this close are read as one (fewer, larger reads)

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_WIDEN = {("float16", "float32"), ("bfloat16", "float32")}  # exact conversions only


class GatherIndexStale(SssFormatError):
    """The index no longer describes its source files (a size or mtime changed)."""


def _down(n: int) -> int:
    return n // ALIGNMENT * ALIGNMENT


def _up(n: int) -> int:
    return -(-n // ALIGNMENT) * ALIGNMENT


@dataclass(frozen=True)
class Segment:
    """``nbytes`` slab bytes at ``slab_offset`` from ``file`` (an index into the files)
    at ``offset``; with ``src_dtype``, the file holds ``src_dtype`` values that widen to
    ``dst_dtype`` (``nbytes`` is the widened size)."""

    file: int
    offset: int
    nbytes: int
    slab_offset: int
    src_dtype: str = ""
    dst_dtype: str = ""

    def __post_init__(self) -> None:
        if self.offset < 0 or self.nbytes <= 0 or self.slab_offset < 0:
            raise ValueError(f"bad segment {self}")
        if self.src_dtype or self.dst_dtype:
            if (self.src_dtype, self.dst_dtype) not in _WIDEN:
                raise ValueError(f"segment conversion {self.src_dtype} -> {self.dst_dtype} "
                                 f"is not one of the exact widenings {sorted(_WIDEN)}")
            if self.nbytes % self._dst_size or self.slab_offset % self._dst_size:
                raise ValueError(f"segment {self} is not whole {self.dst_dtype} values")

    @property
    def _dst_size(self) -> int:
        return _DTYPES[self.dst_dtype].itemsize

    def src_range(self, lo: int, hi: int) -> tuple[int, int]:
        """File byte range of slab bytes ``[lo, hi)`` (relative to this segment)."""
        if not self.src_dtype:
            return self.offset + lo, self.offset + hi
        d, s = self._dst_size, _DTYPES[self.src_dtype].itemsize
        if lo % d or hi % d:
            raise ValueError(f"a piece [{lo}, {hi}) of {self} splits a {self.dst_dtype} value")
        return self.offset + lo // d * s, self.offset + hi // d * s

    @property
    def src_nbytes(self) -> int:
        lo, hi = self.src_range(0, self.nbytes)
        return hi - lo

    def to_json(self) -> list:
        out = [self.file, self.offset, self.nbytes, self.slab_offset]
        return out + [self.src_dtype, self.dst_dtype] if self.src_dtype else out

    @classmethod
    def from_json(cls, d: Sequence) -> Segment:
        return cls(int(d[0]), int(d[1]), int(d[2]), int(d[3]), *(str(x) for x in d[4:6]))


@dataclass(frozen=True)
class GatherSlab:
    """One slab: its size, its checksum and its segments (sorted, non-overlapping).
    ``offset`` is always 0 (``SsdSource`` reads ``offset + start``; unused here)."""

    key: SlabKey
    nbytes: int
    checksum: str
    segments: tuple[Segment, ...]
    offset: int = 0

    def __post_init__(self) -> None:
        end = 0
        for s in self.segments:
            if s.slab_offset < end:
                raise ValueError(f"slab {self.key}: segments overlap or are unsorted at "
                                 f"byte {s.slab_offset}")
            end = s.slab_offset + s.nbytes
        if end > self.nbytes:
            raise ValueError(f"slab {self.key}: segments end at {end} > {self.nbytes} bytes")

    def to_json(self) -> dict:
        return {"key": list(self.key), "nbytes": self.nbytes, "checksum": self.checksum,
                "segs": [s.to_json() for s in self.segments]}

    @classmethod
    def from_json(cls, d: Mapping) -> GatherSlab:
        return cls((int(d["key"][0]), int(d["key"][1])), int(d["nbytes"]), str(d["checksum"]),
                   tuple(Segment.from_json(s) for s in d["segs"]))


def _file_id(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


class GatherIndex:
    """The sidecar: source files (relative to ``root``, with size and mtime), the caller's
    ``meta`` and one ``GatherSlab`` per key. ``check_files`` raises ``GatherIndexStale``
    when a source file changed since the index was built."""

    def __init__(self, root: str | os.PathLike, files: Sequence[Mapping],
                 slabs: Sequence[GatherSlab], *, meta: Mapping[str, Any] | None = None,
                 path: str | os.PathLike | None = None) -> None:
        self.root = Path(root)
        self.files = [dict(f) for f in files]
        self.meta: dict[str, Any] = json.loads(json.dumps(dict(meta or {})))
        self.entries: tuple[GatherSlab, ...] = tuple(slabs)
        self._by_key = {e.key: e for e in self.entries}
        if len(self._by_key) != len(self.entries):
            raise ValueError("a slab key appears twice")
        for e in self.entries:
            for s in e.segments:
                if not 0 <= s.file < len(self.files):
                    raise ValueError(f"slab {e.key}: segment file {s.file} out of range")
        self.path = Path(path) if path is not None else None

    @classmethod
    def build(cls, root: str | os.PathLike, files: Sequence[str],
              slabs: Sequence[GatherSlab], *, meta: Mapping[str, Any] | None = None
              ) -> GatherIndex:
        """An index over ``files`` (paths relative to ``root``), stamped with their
        current sizes and mtimes."""
        root = Path(root)
        return cls(root, [{"name": str(f), **_file_id(root / f)} for f in files], slabs,
                   meta=meta)

    def file_path(self, i: int) -> Path:
        return self.root / self.files[i]["name"]

    def check_files(self) -> None:
        for i, f in enumerate(self.files):
            p = self.file_path(i)
            try:
                now = _file_id(p)
            except OSError as e:
                raise GatherIndexStale(f"{p}: {e.strerror or e}") from e
            if now["size"] != f["size"] or now["mtime_ns"] != f["mtime_ns"]:
                raise GatherIndexStale(f"{p} changed since the index was built "
                                       f"(size/mtime {now} vs {f['size']}/{f['mtime_ns']})")

    def entry(self, key: SlabKey) -> GatherSlab:
        try:
            return self._by_key[(int(key[0]), int(key[1]))]
        except KeyError:
            raise KeyError(f"slab {key} is not in this index") from None

    def to_json(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "checksum": CHECKSUM_ALGO,
                "alignment": ALIGNMENT, "root": str(self.root), "files": self.files,
                "meta": self.meta, "slabs": [e.to_json() for e in self.entries]}

    def save(self, path: str | os.PathLike) -> Path:
        """Write the index atomically (temp + fsync + rename); never inside ``root``."""
        path = Path(path)
        if path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError(f"refusing to write {path} inside the source directory "
                             f"{self.root} (in-place sources are read-only)")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | os.PathLike, *, root: str | os.PathLike | None = None,
             check: bool = True) -> GatherIndex:
        """Read an index; ``root`` overrides the recorded one (a moved directory)."""
        path = Path(path)
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SssFormatError(f"cannot read gather index {path}: {e}") from e
        if d.get("format") != FORMAT or d.get("version") != VERSION:
            raise SssFormatError(f"{path}: format {d.get('format')!r} v{d.get('version')!r}, "
                                 f"expected {FORMAT} v{VERSION}")
        if d.get("checksum") != CHECKSUM_ALGO or d.get("alignment") != ALIGNMENT:
            raise SssFormatError(f"{path}: unsupported checksum/alignment")
        idx = cls(root if root is not None else d["root"], d["files"],
                  [GatherSlab.from_json(s) for s in d["slabs"]], meta=d["meta"], path=path)
        if check:
            idx.check_files()
        return idx

    def close(self) -> None:  # the SsdSource index protocol
        return None

    @property
    def data_nbytes(self) -> int:
        """What these slabs would occupy as a ``.sss`` copy (header + 4 KiB-aligned
        slabs): the disk an in-place source saves."""
        return ALIGNMENT + sum(_up(e.nbytes) for e in self.entries)


def _pread_at_least(fd: int, buf: memoryview, offset: int, need: int) -> None:
    """Read into ``buf`` from ``offset`` until at least ``need`` bytes landed (an aligned
    O_DIRECT read may run past the end of the file: the tail is then short)."""
    got = 0
    while got < need:
        n = os.preadv(fd, [buf[got:]], offset + got)
        if n <= 0:
            raise OSError(errno.EIO, f"short read: {got} of {need} bytes at byte {offset}")
        got += n


# -- the gather plan ----------------------------------------------------------------------
#
# A chunk read is planned once (``plan_gather``) and then executed, natively (one ctypes
# call, ``_gather.c``) or in Python; both run the same plan, so they write the same bytes.
# A plan is a tuple of batches; a batch is (reads, ops) with
#   read = (file, offset, length, need, scratch_off): an aligned read into the scratch;
#   op   = (kind, src, dst, nbytes): OP_COPY / OP_ZERO / OP_F16 / OP_BF16 (scratch -> out).
# Every read of a batch lands in the scratch before its ops run; a new batch starts when
# the scratch is full.

OP_COPY, OP_ZERO, OP_F16, OP_BF16 = 0, 1, 2, 3
_OP_OF = {"float16": OP_F16, "bfloat16": OP_BF16}

Plan = tuple[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...]


def plan_gather(slab: GatherSlab, start: int, nbytes: int, cap: int, *,
                merge_gap: int = MERGE_GAP) -> Plan:
    """The reads and copies that fill slab bytes ``[start, start + nbytes)`` through a
    ``cap``-byte scratch: 4 KiB-aligned supersets of the needed file ranges (O_DIRECT-safe
    given a page-aligned scratch), ranges of one file at most ``merge_gap`` apart sharing
    a read while it fits; bytes no segment covers are zeroed."""
    end = start + nbytes
    zeros: list[tuple[int, ...]] = []
    pieces = []  # (file, src_lo, src_hi, out_lo, out_hi, segment)
    cursor = start
    for s in slab.segments:
        lo, hi = max(s.slab_offset, start), min(s.slab_offset + s.nbytes, end)
        if lo >= hi:
            continue
        if lo > cursor:
            zeros.append((OP_ZERO, 0, cursor - start, lo - cursor))
        src_lo, src_hi = s.src_range(lo - s.slab_offset, hi - s.slab_offset)
        pieces.append((s.file, src_lo, src_hi, lo - start, hi - start, s))
        cursor = hi
    if cursor < end:
        zeros.append((OP_ZERO, 0, cursor - start, end - cursor))
    pieces.sort(key=lambda p: (p[0], p[1]))
    batches: list = []
    reads: list[tuple[int, ...]] = []
    ops: list[tuple[int, ...]] = list(zeros)
    used = 0
    i = 0
    while i < len(pieces):
        f, ext_lo = pieces[i][0], _down(pieces[i][1])
        ext_need = pieces[i][2]
        j = i + 1
        while j < len(pieces) and pieces[j][0] == f \
                and _down(pieces[j][1]) - _up(ext_need) <= merge_gap \
                and _up(max(ext_need, pieces[j][2])) - ext_lo <= cap:
            ext_need = max(ext_need, pieces[j][2])
            j += 1
        length = _up(ext_need) - ext_lo
        if length > cap:
            raise ValueError(f"a read of {length} bytes exceeds the {cap}-byte scratch")
        if used + length > cap:  # the scratch is full: run this batch first
            batches.append((tuple(reads), tuple(ops)))
            reads, ops, used = [], [], 0
        reads.append((f, ext_lo, length, ext_need - ext_lo, used))
        for _, src_lo, src_hi, out_lo, out_hi, s in pieces[i:j]:
            if not (ext_lo <= src_lo <= src_hi <= ext_need and 0 <= out_lo <= out_hi <= nbytes):
                raise AssertionError(f"slab {slab.key}: piece outside its buffers")
            kind = _OP_OF[s.src_dtype] if s.src_dtype else OP_COPY
            ops.append((kind, used + src_lo - ext_lo, out_lo, out_hi - out_lo))
        used += length
        i = j
    if reads or ops:
        batches.append((tuple(reads), tuple(ops)))
    return tuple(batches)


def _run_plan_py(plan: Plan, fds: Sequence[int], drop: Sequence[bool], scratch: torch.Tensor,
                 out: torch.Tensor) -> None:
    """Execute ``plan`` in Python: ``preadv`` + ``memmove`` / ``memset`` (one thread each,
    GIL released per call), torch for the widenings (small sections, below its parallel
    grain). Not torch copies for the bytes: called from a reader thread, a large torch copy
    starts its own OpenMP team per thread, and eight reader threads then oversubscribe the
    cores (a measured 7x slower read on a 32-core host)."""
    smv = memoryview(scratch.numpy())
    out_ptr, scr_ptr = out.data_ptr(), scratch.data_ptr()
    for reads, ops in plan:
        for f, lo, length, need, soff in reads:
            _pread_at_least(fds[f], smv[soff : soff + length], lo, need)
            if drop[f]:
                os.posix_fadvise(fds[f], lo, length, os.POSIX_FADV_DONTNEED)
        for kind, src, dst, n in ops:
            if kind == OP_COPY:
                ctypes.memmove(out_ptr + dst, scr_ptr + src, n)
            elif kind == OP_ZERO:
                ctypes.memset(out_ptr + dst, 0, n)
            else:
                st = torch.float16 if kind == OP_F16 else torch.bfloat16
                s = scratch[src : src + n // 2]
                if src % 2:  # values must be aligned to view them
                    s = s.clone()
                out[dst : dst + n].view(torch.float32).copy_(s.view(st))


# -- the native executor (``_gather.c``, compiled on first use; Python if it cannot be) ----

_NATIVE_LOCK = threading.Lock()
_NATIVE: list = []  # [] untried; [fn] loaded; [None] unavailable
_NATIVE_REASON: list[str] = []


def _cache_dir() -> Path:
    base = os.environ.get("POROS_SSS_NATIVE_DIR")
    if base:
        return Path(base)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "poros" / "sss"


def native_gather():
    """The native gather function (``ctypes``), compiling ``_gather.c`` with ``$CC`` (or
    ``cc``) into ``~/.cache/poros/sss`` on first use; ``None`` when it is disabled
    (``POROS_SSS_NATIVE=0``) or cannot be built (``native_gather_reason()`` says why)."""
    if _NATIVE:
        return _NATIVE[0]
    with _NATIVE_LOCK:
        if not _NATIVE:
            fn, reason = _load_native()
            _NATIVE.append(fn)
            if reason:
                _NATIVE_REASON.append(reason)
                if os.environ.get("POROS_SSS_NATIVE", "1") != "0":
                    logger.warning("poros.sss.gather: native reader unavailable (%s); "
                                   "using the Python reader", reason)
    return _NATIVE[0]


def native_gather_reason() -> str | None:
    return _NATIVE_REASON[0] if _NATIVE_REASON else None


def _private(path: Path, *, folder: bool) -> bool:
    """This user's and writable by no one else: all a library this process loads may sit in, or
    be (a file another user could write is code they choose, run as this user)."""
    try:
        st = os.stat(path) if folder else os.lstat(path)  # the library itself: never a link
    except OSError:
        return False
    kind = stat.S_ISDIR if folder else stat.S_ISREG
    return (kind(st.st_mode) and st.st_uid == os.getuid()
            and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def _build(cc: str, src: Path, lib_path: Path) -> str | None:
    """``src`` compiled into ``lib_path``, whole by a rename and writable by this user alone; why
    it failed, or None."""
    tmp = lib_path.with_name(f"{lib_path.name}.{os.getpid()}.tmp")
    res = subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(tmp), str(src)],
                         capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"{cc} failed: {res.stderr.strip()[-300:]}"
    os.chmod(tmp, 0o755)  # whatever the umask: a group-writable library is never loaded again
    os.replace(tmp, lib_path)
    return None


def _load_native():
    if os.environ.get("POROS_SSS_NATIVE", "1") == "0":
        return None, "disabled by POROS_SSS_NATIVE=0"
    if not sys.platform.startswith("linux"):
        return None, f"not built on {sys.platform}"
    src = Path(__file__).with_name("_gather.c")
    try:
        code = src.read_bytes()
    except OSError as e:
        return None, f"cannot read {src}: {e}"
    tag = hashlib.sha256(code + platform.machine().encode()).hexdigest()[:16]
    cache, scratch = _cache_dir(), None
    try:
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not _private(cache, folder=True):  # another user's library could wait there
            scratch = Path(tempfile.mkdtemp(prefix="poros-sss-"))  # 0700, this process's alone
            logger.warning("poros.sss.gather: %s is not this user's alone (another user can "
                           "write it); building the native reader in a private temp folder "
                           "(chmod 700 it to keep the build)", cache)
    except OSError as e:
        return None, f"cannot make {cache}: {e}"
    lib_path = (scratch or cache) / f"_gather-{tag}.so"
    try:
        if not _private(lib_path, folder=False):  # none yet, or not this user's alone: rebuilt
            cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
            if not cc:
                return None, "no C compiler (cc / gcc / $CC)"
            try:
                if failed := _build(cc, src, lib_path):
                    return None, failed
            except (OSError, subprocess.SubprocessError) as e:
                return None, f"cannot build {lib_path}: {e}"
        try:
            lib = ctypes.CDLL(str(lib_path))
            if lib.poros_sss_gather_abi() != 1:
                return None, f"{lib_path}: unexpected ABI"
            fn = lib.poros_sss_gather
        except (OSError, AttributeError) as e:
            return None, f"cannot load {lib_path}: {e}"
    finally:
        if scratch is not None:  # loaded, it stays mapped: the file is no longer needed
            shutil.rmtree(scratch, ignore_errors=True)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                   ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    return fn, None


class _NativePlan:
    """A plan encoded for ``_gather.c``: per batch, the read and op arrays (int64)."""

    __slots__ = ("batches",)

    def __init__(self, plan: Plan) -> None:
        self.batches = []
        for reads, ops in plan:
            r = (ctypes.c_int64 * max(1, 5 * len(reads)))(*(x for rd in reads for x in rd))
            o = (ctypes.c_int64 * max(1, 4 * len(ops)))(*(x for op in ops for x in op))
            self.batches.append((r, len(reads), o, len(ops)))

    def run(self, fn, fds, drop, scratch_ptr: int, out_ptr: int, what) -> None:
        for r, nr, o, no in self.batches:
            rc = fn(ctypes.addressof(fds), ctypes.addressof(drop), ctypes.addressof(r), nr,
                    ctypes.addressof(o), no, scratch_ptr, out_ptr)
            if rc:
                raise OSError(-rc, f"gathering {what}: {os.strerror(-rc)}")


def gather_into(slab: GatherSlab, start: int, out: torch.Tensor, fds: Sequence[int],
                scratch: torch.Tensor, *, drop_cache: Sequence[bool] | bool = False,
                merge_gap: int = MERGE_GAP, native: bool | None = None) -> None:
    """Fill ``out`` (uint8, CPU) with ``slab``'s bytes ``[start, start + len(out))``
    (``plan_gather`` through ``scratch``, page-aligned). ``native``: ``None`` uses the
    native executor when it is available, ``True`` requires it, ``False`` runs Python."""
    for t in (out, scratch):
        if t.dtype != torch.uint8 or t.device.type != "cpu" or not t.is_contiguous():
            raise ValueError("out and scratch must be contiguous uint8 CPU tensors")
    plan = plan_gather(slab, start, out.numel(), scratch.numel(), merge_gap=merge_gap)
    drop = list(drop_cache) if isinstance(drop_cache, Sequence) else [drop_cache] * len(fds)
    fn = native_gather() if native is not False else None
    if native and fn is None:
        raise RuntimeError(f"native gather unavailable: {native_gather_reason()}")
    if fn is None:
        _run_plan_py(plan, fds, drop, scratch, out)
        return
    c_fds = (ctypes.c_int64 * max(1, len(fds)))(*fds)
    c_drop = (ctypes.c_uint8 * max(1, len(drop)))(*map(bool, drop))
    _NativePlan(plan).run(fn, c_fds, c_drop, scratch.data_ptr(), out.data_ptr(), slab.key)


class _RawPlan:
    """One slab in raw-ring mode: the aligned reads that land its file ranges back to back
    at the start of a ring slot, the ops that assemble the slab from there, and the same
    ops as ``(ring_offset, n)`` parts in slab order (``-1``: ``n`` zero bytes)."""

    __slots__ = ("reads_plan", "ops_plan", "ring_nbytes", "parts", "max_zero",
                 "_native_reads", "_native_ops")

    def __init__(self, slab: GatherSlab, reads, ops) -> None:
        self.reads_plan = ((tuple(reads), ()),)
        self.ops_plan = (((), tuple(ops)),)
        self.ring_nbytes = max(ALIGNMENT, sum(r[2] for r in reads))
        parts, at, max_zero = [], 0, 0
        for kind, src, dst, n in sorted(ops, key=lambda o: o[2]):
            if dst != at:
                raise AssertionError(f"slab {slab.key}: assembly ops leave a hole at {at}")
            parts.append((src if kind == OP_COPY else -1, n))
            if kind == OP_ZERO:
                max_zero = max(max_zero, n)
            at += n
        if at != slab.nbytes:
            raise AssertionError(f"slab {slab.key}: assembly covers {at} of {slab.nbytes}")
        self.parts = tuple(parts)
        self.max_zero = max_zero
        self._native_reads = self._native_ops = None

    @property
    def native_reads(self) -> _NativePlan:
        if self._native_reads is None:
            self._native_reads = _NativePlan(self.reads_plan)
        return self._native_reads

    @property
    def native_ops(self) -> _NativePlan:
        if self._native_ops is None:
            self._native_ops = _NativePlan(self.ops_plan)
        return self._native_ops


def _raw_plans(entries: Sequence[GatherSlab]) -> dict | None:
    """Every slab's ``_RawPlan``, or ``None`` when one does not qualify (more than one
    batch, or a widened section: those need the host-assembled ring)."""
    plans = {}
    for e in entries:
        plan = plan_gather(e, 0, e.nbytes, 1 << 60)
        if len(plan) > 1:
            return None
        reads, ops = plan[0] if plan else ((), ())
        if any(op[0] not in (OP_COPY, OP_ZERO) for op in ops):
            return None
        plans[e.key] = _RawPlan(e, reads, ops)
    return plans


class GatherSource(SsdSource):
    """A ``SlabSource`` reading every slab in place through a ``GatherIndex``.

    Args:
        index: The ``GatherIndex`` (or the path of a saved one).
        root: Overrides the index's source directory.
        **kwargs: ``SsdSource``'s (``chunk_bytes``, ``ram_budget``, ``threads``,
            ``depth``, ``verify``, ``direct``, ``pin``, ...).

        native: ``None`` (default) gathers each chunk in one native call (``_gather.c``,
            no GIL held) when it builds, else in Python; ``True`` requires it; ``False``
            is Python.
        raw_ring: ``None`` (default) lands each slab's raw file ranges in the ring and
            assembles the slab on the copy out (on the GPU for a CUDA ``dst``: one
            ``torch.cat``), whenever every slab qualifies (one read plan, no widened
            section); ``True`` requires it; ``False`` assembles in host memory.

    Each reader thread holds one page-aligned scratch buffer of ``2 * chunk_bytes +
    8 KiB`` (unpinned) for the aligned reads; the slab is assembled into the ring from
    it, so the ring's bytes, checksum and copy to ``dst`` are exactly ``SsdSource``'s.
    Each chunk's plan (``plan_gather``) is built on its first read and cached.

    Why native: the Python reader makes ~14 GIL-releasing calls per slab (a
    ``preadv`` per read, a ``memmove`` per segment); each one hands the GIL back and
    forth with the training thread, which runs Python between kernel launches (3090,
    fixed-cap Qwen3-30B, profiled: 41.6 s/step vs 31.9 for the ``.sss`` copy; native
    36.8). Why the raw ring: assembling the slab in host memory is one extra memcpy of
    every streamed byte (~15 GB/step there, ~33 CPU-s/step on that host); landing the
    raw ranges where the ring slot is (O_DIRECT, no CPU copy) and assembling them after
    the host-to-device copy, on the GPU, leaves the host the ``.sss`` copy's work.

    In raw mode the first-read checksum is taken over the slab assembled in host memory
    (then copied to ``dst``): the bytes checked are the bytes delivered.
    """

    def __init__(self, index: GatherIndex | str | os.PathLike, *,
                 root: str | os.PathLike | None = None, native: bool | None = None,
                 raw_ring: bool | None = None, **kwargs) -> None:
        if not isinstance(index, GatherIndex):
            index = GatherIndex.load(index, root=root)
        elif root is not None:
            index = GatherIndex(root, index.files, index.entries, meta=index.meta,
                                path=index.path)
        index.check_files()
        self._index = index
        self._fds: list[int] = []
        self._direct_fds: list[bool] = []
        self._tls = threading.local()
        self._scratch_lock = threading.Lock()
        self._scratches: list[mmap.mmap] = []
        self._plans: dict = {}
        self._native_req = native
        self._native = None
        self._raw, self._raw_plans = False, {}
        if raw_ring is not False:
            plans = _raw_plans(index.entries)
            if plans is not None:
                self._raw, self._raw_plans = True, plans
                largest = max((p.ring_nbytes for p in plans.values()), default=ALIGNMENT)
                kwargs["chunk_bytes"] = max(int(kwargs.get("chunk_bytes", DEFAULT_CHUNK_BYTES)),
                                            largest)
            elif raw_ring:
                raise ValueError("raw_ring=True needs every slab to be one read plan without "
                                 "widened sections")
        self._zeros: dict = {}
        super().__init__(index.path or index.root, **kwargs)

    # -- SsdSource hooks -------------------------------------------------------

    def _open_index(self, path):
        return self._index

    @property
    def index(self) -> GatherIndex:
        return self._index

    def _open_data(self, direct: bool | str) -> tuple[bool, str | None]:
        reasons = []
        try:
            for i in range(len(self._index.files)):
                fd, ok, reason = _open_file(self._index.file_path(i), direct)
                self._fds.append(fd)
                self._direct_fds.append(ok)
                if reason and reason not in reasons:
                    reasons.append(reason)
        except BaseException:
            self._close_fds()
            raise
        self._scratch_bytes = 2 * self.chunk_bytes + 2 * ALIGNMENT
        self._drop = [not d for d in self._direct_fds]
        if self._native_req is not False:
            self._native = native_gather()
            if self._native is None and self._native_req:
                self._close_fds()
                raise RuntimeError(f"native gather unavailable: {native_gather_reason()}")
        n = max(1, len(self._fds))
        self._c_fds = (ctypes.c_int64 * n)(*self._fds)
        self._c_drop = (ctypes.c_uint8 * n)(*self._drop)
        reason = "; ".join(reasons) or None
        return all(self._direct_fds), reason

    def _hashes_in_reader(self, e: GatherSlab) -> bool:
        """Raw mode lands file ranges, not the slab: its copy out checksums the assembled slab."""
        return not self._raw and super()._hashes_in_reader(e)

    def _ring_nbytes(self, e: GatherSlab) -> int:
        return self._raw_plans[e.key].ring_nbytes if self._raw else e.nbytes

    def _read_chunk(self, e: GatherSlab, start: int, buf: memoryview) -> None:
        if self._raw:  # the raw ranges straight into the ring slot (one chunk per slab)
            rp = self._raw_plans[e.key]
            if self._native is None:
                ring = torch.frombuffer(buf, dtype=torch.uint8)
                _run_plan_py(rp.reads_plan, self._fds, self._drop, ring, ring)
                return
            slot = ctypes.c_char.from_buffer(buf)
            try:
                rp.native_reads.run(self._native, self._c_fds, self._c_drop,
                                    ctypes.addressof(slot), 0, e.key)
            finally:
                del slot
            return
        k = (e.key, start, len(buf))
        plan = self._plans.get(k)
        if plan is None:
            plan = plan_gather(e, start, len(buf), self._scratch_bytes)
            if self._native is not None:
                plan = _NativePlan(plan)
            self._plans[k] = plan
        scratch = self._scratch()
        if self._native is None:
            _run_plan_py(plan, self._fds, self._drop, scratch,
                         torch.frombuffer(buf, dtype=torch.uint8))
            return
        out = ctypes.c_char.from_buffer(buf)
        try:
            plan.run(self._native, self._c_fds, self._c_drop, self._tls.scratch_ptr,
                     ctypes.addressof(out), e.key)
        finally:
            del out

    def _assemble_host(self, rp: _RawPlan, ring: torch.Tensor, out: torch.Tensor) -> None:
        if self._native is None:
            _run_plan_py(rp.ops_plan, self._fds, self._drop, ring, out)
        else:
            rp.native_ops.run(self._native, self._c_fds, self._c_drop, ring.data_ptr(),
                              out.data_ptr(), "assembly")

    def _assemble_device(self, rp: _RawPlan, stage: torch.Tensor, dst: torch.Tensor) -> None:
        zeros = None
        parts = []
        for lo, n in rp.parts:
            if lo >= 0:
                parts.append(stage[lo : lo + n])
            else:
                if zeros is None:
                    zeros = self._zeros.get(dst.device)
                    if zeros is None or zeros.numel() < rp.max_zero:
                        size = max(rp.max_zero, ALIGNMENT)
                        zeros = self._zeros[dst.device] = torch.zeros(
                            size, dtype=torch.uint8, device=dst.device)
                parts.append(zeros[:n])
        if len(parts) == 1:
            dst.copy_(parts[0])
        else:
            torch.cat(parts, out=dst)

    def _consume(self, read: _Read, dst: torch.Tensor, verify: bool) -> None:
        if not self._raw:
            return super()._consume(read, dst, verify)
        e = self._entries[read.key]
        rp = self._raw_plans[read.key]
        try:
            read.futures[0].result()
        except BaseException:
            self._draining.append(read)
            raise
        base = read.slots[0] * self.chunk_bytes
        ring = self._ring[base : base + rp.ring_nbytes]
        try:
            if verify or dst.device.type != "cuda":
                host = dst if dst.device.type == "cpu" else torch.empty(
                    e.nbytes, dtype=torch.uint8)
                self._assemble_host(rp, ring, host)
                if verify:
                    h = hashlib.blake2b(digest_size=8)
                    h.update(memoryview(host.numpy()))
                    if h.hexdigest() != e.checksum:
                        raise SssCorruptError(f"slab {e.key} in {self.path} does not match "
                                              "its checksum")
                if host is not dst:
                    dst.copy_(host)
            else:
                stage = torch.empty(rp.ring_nbytes, dtype=torch.uint8, device=dst.device)
                stage.copy_(ring, non_blocking=True)
                self._assemble_device(rp, stage, dst)
        finally:
            self._release_slots(read.slots, dst)

    def _stream(self, key, dst: torch.Tensor, verify: bool) -> None:
        if not self._raw:
            return super()._stream(key, dst, verify)
        self._consume(self._submit(key, None, self._take(1, block=True)), dst, verify)

    def _scratch(self) -> torch.Tensor:
        t = getattr(self._tls, "scratch", None)
        if t is None:
            mm = mmap.mmap(-1, self._scratch_bytes)
            with self._scratch_lock:
                self._scratches.append(mm)
            t = self._tls.scratch = torch.frombuffer(mm, dtype=torch.uint8)
            self._tls.scratch_ptr = t.data_ptr()
        return t

    def _close_fds(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()

    def _close_data(self) -> None:
        self._close_fds()
        with self._scratch_lock:
            scratches, self._scratches = self._scratches, []
        self._tls = threading.local()
        for mm in scratches:
            try:
                mm.close()
            except BufferError:  # a tensor view still exports it; freed with the view
                pass

    @property
    def stats(self) -> dict:
        return {**super().stats, "source": "in-place", "files": len(self._index.files),
                "direct_files": sum(self._direct_fds),
                "reader": "python" if self._native is None else "native",
                "ring": "raw" if self._raw else "assembled"}


def _open_file(path: Path, direct: bool | str) -> tuple[int, bool, str | None]:
    """A read-only descriptor: O_DIRECT where the filesystem takes it (a 4 KiB probe read
    at offset 0, short at the end of a small file), else buffered."""
    reason = None
    if direct and hasattr(os, "O_DIRECT"):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError as e:
            reason = f"O_DIRECT open refused: {e.strerror or e}"
        else:
            probe = mmap.mmap(-1, ALIGNMENT)
            try:
                need = min(ALIGNMENT, os.fstat(fd).st_size)
                if need:
                    _pread_at_least(fd, memoryview(probe), 0, need)
            except OSError as e:
                os.close(fd)
                reason = f"O_DIRECT read refused: {e.strerror or e}"
            else:
                return fd, True, None
            finally:
                probe.close()
    elif direct:
        reason = "O_DIRECT is not available on this platform"
    if direct is True:
        raise OSError(f"{path}: {reason}")
    return os.open(path, os.O_RDONLY), False, reason
