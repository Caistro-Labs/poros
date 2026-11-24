"""The ``.sss`` slab file: ``model.sss`` (data) + ``model.sss.idx`` (index).

Layout of ``model.sss``:

- a 4 KiB header: magic ``PSSS``, little-endian u32 format version, u64 slab
  count, u64 data size, zero padded;
- every slab at a 4 KiB-aligned offset, zero padded to the next boundary
  (O_DIRECT-ready, NVMe page aligned).

``model.sss.idx`` is JSON: the format fields, the caller's ``meta`` (model id
and revision, quant config, layout, library versions -- the reuse key), and
one entry per slab: ``key``, ``kind``, ``offset``, ``nbytes``, optional
``views`` and the slab's checksum.

Writes are crash-safe: data and index go to ``*.tmp`` files, both are
fsynced, the data file is renamed first and the index last, so a reader that
finds an index always finds the complete data it describes. A file is reused
only while ``SssFile.matches(meta)``; any mismatch means rebuild.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from poros.sss.files import new_file  # noqa: F401 -- torch-free, for the probe; named here too
from poros.sss.source import RamSource, SlabKey, check_dst

__all__ = [
    "ALIGNMENT",
    "CHECKSUM_ALGO",
    "FORMAT_VERSION",
    "RamSink",
    "SlabEntry",
    "SlabSink",
    "SssCorruptError",
    "SssFile",
    "SssFormatError",
    "SssWriter",
    "SyncFileSource",
    "slab_checksum",
]

MAGIC = b"PSSS"
FORMAT_VERSION = 1
ALIGNMENT = 4096
HEADER_NBYTES = ALIGNMENT
CHECKSUM_ALGO = "blake2b-64"
_HEADER = struct.Struct("<4sIQQ")
_IO_CHUNK = 64 << 20
_HASH_THREADS = max(1, min(16, os.cpu_count() or 1))  # SssWriter.finish's checksum readers



class SssFormatError(ValueError):
    """The file is not a complete, readable ``.sss`` of this format version."""


class SssCorruptError(SssFormatError):
    """A slab's bytes do not match the checksum recorded when it was written."""


def _align(n: int) -> int:
    return -(-n // ALIGNMENT) * ALIGNMENT


def _as_bytes(t: torch.Tensor) -> memoryview:
    if t.device.type != "cpu" or t.dtype != torch.uint8 or not t.is_contiguous():
        raise ValueError("expected a contiguous uint8 CPU tensor")
    return memoryview(t.numpy()).cast("B")


def slab_checksum(data: torch.Tensor | bytes | memoryview) -> str:
    """The recorded checksum of a slab's bytes (``CHECKSUM_ALGO``), as hex."""
    buf = _as_bytes(data) if isinstance(data, torch.Tensor) else memoryview(data)
    h = hashlib.blake2b(digest_size=8)
    for start in range(0, len(buf), _IO_CHUNK):
        h.update(buf[start : start + _IO_CHUNK])
    return h.hexdigest()


@dataclass(frozen=True)
class SlabEntry:
    """One slab in a ``.sss`` file."""

    key: SlabKey
    nbytes: int
    offset: int = 0
    kind: str = "expert"
    checksum: str = ""
    views: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {"key": list(self.key), "kind": self.kind, "offset": self.offset,
                "nbytes": self.nbytes, "checksum": self.checksum,
                "views": [dict(v) for v in self.views]}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> SlabEntry:
        return cls(key=(int(d["key"][0]), int(d["key"][1])), nbytes=int(d["nbytes"]),
                   offset=int(d["offset"]), kind=str(d["kind"]), checksum=str(d["checksum"]),
                   views=tuple(d.get("views", ())))


class _Coverage:
    """Which byte ranges of each slab have been written (writes may not overlap)."""

    def __init__(self, sizes: Mapping[SlabKey, int]) -> None:
        self._sizes = dict(sizes)
        self._ranges: dict[SlabKey, list[tuple[int, int]]] = {k: [] for k in sizes}

    def add(self, key: SlabKey, start: int, nbytes: int) -> None:
        if key not in self._sizes:
            raise KeyError(f"slab {key} was not declared")
        end = start + nbytes
        if start < 0 or end > self._sizes[key]:
            raise ValueError(f"write [{start}, {end}) is outside slab {key} "
                             f"of {self._sizes[key]} bytes")
        for a, b in self._ranges[key]:
            if start < b and a < end:
                raise ValueError(f"write [{start}, {end}) overlaps an earlier write "
                                 f"[{a}, {b}) of slab {key}")
        self._ranges[key].append((start, end))

    def missing(self) -> list[tuple[SlabKey, int]]:
        out = []
        for key, size in self._sizes.items():
            covered = sum(b - a for a, b in self._ranges[key])
            if covered != size:
                out.append((key, size - covered))
        return out


class SlabSink(Protocol):
    """Where a builder writes slab bytes; ``finish`` returns the matching source."""

    def write(self, key: SlabKey, offset: int, data: torch.Tensor) -> None:
        """Write ``data`` (contiguous uint8, CPU) at byte ``offset`` of ``key``'s slab."""
        ...

    def finish(self):
        """Seal the slabs (every byte must have been written) and return their source."""
        ...


def _declare(slabs: Sequence[tuple[SlabKey, int]]) -> dict[SlabKey, int]:
    sizes: dict[SlabKey, int] = {}
    for key, nbytes in slabs:
        key = (int(key[0]), int(key[1]))
        if key in sizes:
            raise ValueError(f"slab {key} declared twice")
        if nbytes <= 0:
            raise ValueError(f"slab {key} must have a positive size, got {nbytes}")
        sizes[key] = int(nbytes)
    return sizes


class RamSink:
    """Pinned host slabs (today's arena rows); ``finish`` returns a ``RamSource``.

    Rows are carved from ~1 GiB pinned chunks, as ``ExpertSlabArena`` does, so
    the caching host allocator's power-of-two rounding stays bounded.
    """

    _CHUNK_BYTES = 1 << 30

    def __init__(self, slabs: Sequence[tuple[SlabKey, int]], *, pin: bool = True) -> None:
        self._sizes = _declare(slabs)
        self._cov = _Coverage(self._sizes)
        self._slabs: dict[SlabKey, torch.Tensor] = {}
        pending: list[SlabKey] = []
        pending_bytes = 0
        for key, n in self._sizes.items():
            if pending and pending_bytes + n > self._CHUNK_BYTES:
                self._carve(pending, pin)
                pending, pending_bytes = [], 0
            pending.append(key)
            pending_bytes += n
        if pending:
            self._carve(pending, pin)

    def _carve(self, keys: list[SlabKey], pin: bool) -> None:
        from poros.sss import pinning

        total = sum(self._sizes[k] for k in keys)
        chunk = None
        if pin and pinning.reserve(pinning.pinned_size(total)):
            try:
                chunk = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            except RuntimeError:  # no pinning on this platform
                pinning.release(pinning.pinned_size(total))
                chunk = None
            else:  # the store's slabs live as long as it does; their grant returns with them
                import weakref

                weakref.finalize(chunk.untyped_storage(), pinning._freed,
                                 pinning.pinned_size(total))
        if chunk is None:
            chunk = torch.empty(total, dtype=torch.uint8)
        start = 0
        for k in keys:
            self._slabs[k] = chunk[start : start + self._sizes[k]]
            start += self._sizes[k]

    def write(self, key: SlabKey, offset: int, data: torch.Tensor) -> None:
        key = (int(key[0]), int(key[1]))
        self._cov.add(key, offset, data.numel())
        self._slabs[key][offset : offset + data.numel()].copy_(data.reshape(-1))

    def row(self, key: SlabKey) -> torch.Tensor:
        """``key``'s carved row, for a reader that fills it whole (``RamSource.from_sss``)."""
        return self._slabs[(int(key[0]), int(key[1]))]

    def rows(self) -> dict[SlabKey, torch.Tensor]:
        """Every carved row, filled by the caller (no coverage check: ``finish`` has that)."""
        return dict(self._slabs)

    def finish(self) -> RamSource:
        missing = self._cov.missing()
        if missing:
            (key, n), count = missing[0], len(missing)
            raise ValueError(f"{count} slab(s) not fully written, e.g. {key} ({n} bytes short)")
        return RamSource(self._slabs)


class SssWriter:
    """Write a ``.sss`` file once; ``finish`` makes it visible atomically.

    The data file is preallocated to its final size and written with
    ``pwrite`` at each slab's aligned offset, so writes may arrive in any
    order (the MoE loader fills one tensor of every expert at a time).
    Checksums are taken from the bytes on disk after the last write.
    """

    def __init__(self, path: str | os.PathLike, slabs: Sequence[tuple[SlabKey, int]],
                 *, meta: Mapping[str, Any], kinds: Mapping[SlabKey, str] | None = None,
                 views: Mapping[SlabKey, Sequence[Mapping[str, Any]]] | None = None) -> None:
        self.path = Path(path)
        self.idx_path = Path(str(path) + ".idx")
        self._meta = json.loads(json.dumps(dict(meta)))  # must be plain JSON, copied
        self._sizes = _declare(slabs)
        self._cov = _Coverage(self._sizes)
        kinds = {tuple(k): v for k, v in (kinds or {}).items()}
        views = {tuple(k): tuple(v) for k, v in (views or {}).items()}
        offset = HEADER_NBYTES
        self._entries: list[SlabEntry] = []
        for key, n in self._sizes.items():
            self._entries.append(SlabEntry(key, n, offset, kinds.get(key, "expert"), "",
                                           views.get(key, ())))
            offset += _align(n)
        self._data_nbytes = offset
        self._offsets = {e.key: e.offset for e in self._entries}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(str(self.path) + ".tmp")
        self._idx_tmp = Path(str(self.idx_path) + ".tmp")
        self._fd = new_file(self._tmp, os.O_RDWR)
        try:
            os.ftruncate(self._fd, self._data_nbytes)  # sparse until written; padding reads 0
            header = _HEADER.pack(MAGIC, FORMAT_VERSION, len(self._entries), self._data_nbytes)
            os.pwrite(self._fd, header.ljust(HEADER_NBYTES, b"\0"), 0)
        except BaseException:
            self.abort()
            raise
        self._done = False

    def write(self, key: SlabKey, offset: int, data: torch.Tensor) -> None:
        key = (int(key[0]), int(key[1]))
        flat = data.reshape(-1)
        self._cov.add(key, offset, flat.numel())
        buf = _as_bytes(flat.contiguous())
        pos = self._offsets[key] + offset
        while buf:
            n = os.pwrite(self._fd, buf, pos)
            buf, pos = buf[n:], pos + n

    def _checksum(self, e: SlabEntry) -> str:
        """Slab ``e``'s checksum, read back from the file (one of ``finish``'s threads)."""
        h = hashlib.blake2b(digest_size=8)
        pos, left = e.offset, e.nbytes
        while left:
            chunk = os.pread(self._fd, min(left, _IO_CHUNK), pos)
            if not chunk:
                raise OSError(f"short read while checksumming slab {e.key}")
            h.update(chunk)
            pos, left = pos + len(chunk), left - len(chunk)
        return h.hexdigest()

    def finish(self) -> SyncFileSource:
        missing = self._cov.missing()
        if missing:
            (key, n), count = missing[0], len(missing)
            raise ValueError(f"{count} slab(s) not fully written, e.g. {key} ({n} bytes short)")
        os.fsync(self._fd)
        # every slab's checksum from its bytes on disk, on a pool of threads (hashlib and pread
        # let go of the GIL): the same checksums, in the same order, a fraction of the time
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=_HASH_THREADS) as pool:
            sums = list(pool.map(self._checksum, self._entries))
        entries = [SlabEntry(e.key, e.nbytes, e.offset, e.kind, h, e.views)
                   for e, h in zip(self._entries, sums, strict=True)]
        os.close(self._fd)
        self._fd = -1
        idx = {"format": "poros.sss", "version": FORMAT_VERSION, "alignment": ALIGNMENT,
               "checksum": CHECKSUM_ALGO, "data_nbytes": self._data_nbytes, "meta": self._meta,
               "slabs": [e.to_json() for e in entries]}
        with os.fdopen(new_file(self._idx_tmp, os.O_WRONLY), "w") as f:
            json.dump(idx, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self._tmp, self.path)          # data first ...
        os.replace(self._idx_tmp, self.idx_path)  # ... the index last commits the pair
        dfd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        self._done = True
        return SyncFileSource(self.path)

    def abort(self) -> None:
        """Drop the partial files; nothing visible is left behind."""
        if getattr(self, "_fd", -1) >= 0:
            os.close(self._fd)
            self._fd = -1
        for p in (self._tmp, self._idx_tmp):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> SssWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None or not self._done:
            self.abort()


class SssFile:
    """A committed ``.sss`` file: its index plus synchronous slab reads."""

    def __init__(self, path: Path, idx: Mapping[str, Any], fd: int) -> None:
        self.path = path
        self.meta: dict[str, Any] = dict(idx["meta"])
        self.entries: tuple[SlabEntry, ...] = tuple(SlabEntry.from_json(s) for s in idx["slabs"])
        self._by_key = {e.key: e for e in self.entries}
        self._fd = fd

    @classmethod
    def open(cls, path: str | os.PathLike) -> SssFile:
        path = Path(path)
        idx_path = Path(str(path) + ".idx")
        if not idx_path.is_file():  # a pipe here would hang its read
            raise SssFormatError(f"{idx_path} not found: {path} is missing or was never committed")
        try:
            idx = json.loads(idx_path.read_text())
        except (OSError, ValueError, RecursionError) as e:  # not JSON, or not text
            raise SssFormatError(f"cannot read {idx_path}: {type(e).__name__}") from e
        if not isinstance(idx, dict):
            raise SssFormatError(f"{idx_path} is not an index")
        if idx.get("format") != "poros.sss" or idx.get("version") != FORMAT_VERSION:
            raise SssFormatError(f"{idx_path}: format {idx.get('format')!r} version "
                                 f"{idx.get('version')!r}, expected poros.sss v{FORMAT_VERSION}")
        if idx.get("checksum") != CHECKSUM_ALGO or idx.get("alignment") != ALIGNMENT:
            raise SssFormatError(f"{idx_path}: unsupported checksum/alignment")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)  # a pipe opens, and is refused
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise SssFormatError(f"{path} is not a regular file")
            size = os.fstat(fd).st_size
            if size != idx["data_nbytes"]:
                raise SssFormatError(f"{path} is {size} bytes, its index says "
                                     f"{idx['data_nbytes']}")
            magic, version, count, data_nbytes = _HEADER.unpack(os.pread(fd, _HEADER.size, 0))
            if magic != MAGIC or version != FORMAT_VERSION or count != len(idx["slabs"]) \
                    or data_nbytes != size:
                raise SssFormatError(f"{path}: header does not match its index")
            return cls(path, idx, fd)
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError, struct.error) as e:
            os.close(fd)
            if isinstance(e, SssFormatError):
                raise
            raise SssFormatError(f"{idx_path} is not an index: {type(e).__name__}") from e
        except BaseException:
            os.close(fd)
            raise

    @property
    def fd(self) -> int:
        """The open data file, for a caller that reads part of a slab or sets ``F_NOCACHE``."""
        return self._fd

    def matches(self, meta: Mapping[str, Any]) -> bool:
        """True iff this file was written for exactly ``meta`` (reuse it; else rebuild)."""
        return self.meta == json.loads(json.dumps(dict(meta)))

    def entry(self, key: SlabKey) -> SlabEntry:
        try:
            return self._by_key[(int(key[0]), int(key[1]))]
        except KeyError:
            raise KeyError(f"slab {key} is not in {self.path}") from None

    def read_into(self, key: SlabKey, dst: torch.Tensor, *, verify: bool = False) -> None:
        """Read ``key``'s bytes into ``dst`` (contiguous uint8, CPU), optionally verified."""
        e = self.entry(key)
        check_dst(e.key, dst, e.nbytes)
        if dst.device.type != "cpu":
            raise ValueError("SssFile.read_into fills host memory; use SyncFileSource for CUDA")
        buf = _as_bytes(dst)
        pos, view = e.offset, buf
        while view:
            n = os.preadv(self._fd, [view], pos)
            if n <= 0:
                raise SssFormatError(f"short read of slab {e.key} in {self.path}")
            view, pos = view[n:], pos + n
        if verify and slab_checksum(buf) != e.checksum:
            raise SssCorruptError(f"slab {e.key} in {self.path} does not match its checksum")

    def verify_all(self) -> None:
        """Checksum every slab; raises ``SssCorruptError`` on the first mismatch."""
        for e in self.entries:
            self.read_into(e.key, torch.empty(e.nbytes, dtype=torch.uint8), verify=True)

    def close(self) -> None:
        if getattr(self, "_fd", -1) >= 0:  # an index refused halfway never had one
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> SssFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class SyncFileSource:
    """A ``SlabSource`` reading a ``.sss`` file synchronously (buffered ``preadv``).

    The reference SSD reader: a CPU ``dst`` is filled directly; a CUDA ``dst``
    is filled through a reusable host staging buffer and a blocking copy, so
    the staging buffer is never overwritten while a copy is in flight. Each
    slab is checksum-verified on its first read in this process (or every
    read with ``verify="always"``).
    """

    def __init__(self, path: str | os.PathLike, *, verify: str = "first") -> None:
        if verify not in ("first", "always", "never"):
            raise ValueError(f"verify must be 'first', 'always' or 'never', got {verify!r}")
        self._file = SssFile.open(path)
        self._verify = verify
        self._verified: set[SlabKey] = set()
        self._stage: torch.Tensor | None = None
        self._keys = tuple(e.key for e in self._file.entries)

    @property
    def file(self) -> SssFile:
        return self._file

    def keys(self) -> Sequence[SlabKey]:
        return self._keys

    def nbytes(self, key: SlabKey) -> int:
        return self._file.entry(key).nbytes

    def read_into(self, key: SlabKey, dst: torch.Tensor) -> None:
        e = self._file.entry(key)
        check_dst(e.key, dst, e.nbytes)
        verify = self._verify == "always" or (self._verify == "first"
                                              and e.key not in self._verified)
        if dst.device.type == "cpu":
            self._file.read_into(e.key, dst, verify=verify)
        else:
            if self._stage is None or self._stage.numel() < e.nbytes:
                stage = torch.empty(e.nbytes, dtype=torch.uint8)
                try:
                    stage = stage.pin_memory()
                except RuntimeError:
                    pass
                self._stage = stage
            host = self._stage[: e.nbytes]
            self._file.read_into(e.key, host, verify=verify)
            dst.copy_(host, non_blocking=False)
        if verify:
            self._verified.add(e.key)

    def prefetch(self, keys: Sequence[SlabKey]) -> None:  # synchronous reader
        return None

    def release(self, key: SlabKey) -> None:
        return None

    def close(self) -> None:
        self._file.close()
