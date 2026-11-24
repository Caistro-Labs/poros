"""``GatherSource``: slabs read in place from existing files through a ``GatherIndex``.

Every read, in any order, through any ring and chunk size, equals the slab assembled by
hand from the source files (misaligned offsets, ranges across 4 KiB pages and chunk
boundaries, a range ending at an unaligned end of file, float16 / bfloat16 widened to
float32, zero gaps). A corrupted source byte fails the first-read checksum; a source file
whose size or mtime changed fails at open. ``POROS_SSS_TEST_DIR`` runs the O_DIRECT
variant on a real local filesystem.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

from tests.conftest import needs_fadvise

torch = pytest.importorskip("torch")

import poros.sss.gather as gather_mod  # noqa: E402
from poros.sss import (  # noqa: E402
    GatherIndex,
    GatherIndexStale,
    GatherSlab,
    GatherSource,
    Segment,
    SlabSource,
    SssCorruptError,
    slab_checksum,
)

HALF = 35_000  # b.bin: float16 values in [0, HALF), bfloat16 values in [HALF, 2 * HALF)


def _files(root: Path, seed=0):
    """Random bytes, except b.bin, which holds finite float16 then bfloat16 values (the
    ranges conversions read: widening a NaN is not bit-unique, see ``poros.sss.gather``)."""
    g = torch.Generator().manual_seed(seed)
    sizes = {"a.bin": 300_001, "b.bin": 2 * HALF + 3, "sub/c.bin": 9_999}
    data = {}
    for name, n in sizes.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        t = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
        if name == "b.bin":
            vals = torch.randn(HALF // 2, generator=g) * 1e3
            t[:HALF] = vals.to(torch.float16).view(torch.uint8)
            t[HALF : 2 * HALF] = vals.to(torch.bfloat16).view(torch.uint8)
        (root / name).write_bytes(t.numpy().tobytes())
        data[name] = t
    return list(sizes), data


def _expected(files, data, slab: GatherSlab) -> torch.Tensor:
    out = torch.zeros(slab.nbytes, dtype=torch.uint8)
    for s in slab.segments:
        src = data[files[s.file]]
        dst = out[s.slab_offset : s.slab_offset + s.nbytes]
        if not s.src_dtype:
            dst.copy_(src[s.offset : s.offset + s.nbytes])
        else:
            dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[s.src_dtype]
            raw = src[s.offset : s.offset + s.src_nbytes].clone().view(dt)
            dst.view(torch.float32).copy_(raw.to(torch.float32))
    return out


def _slabs(files, data, nslabs=12, seed=1):
    """Random slabs: 2-7 segments each, misaligned file offsets, gaps, conversions."""
    rng = random.Random(seed)
    slabs = []
    for i in range(nslabs):
        segs, at = [], rng.choice([0, 64, 128])
        for _ in range(rng.randint(2, 7)):
            f = rng.randrange(len(files))
            size = data[files[f]].numel()
            conv = rng.random() < 0.25
            n = min(rng.choice([4, 64, 1000, 4096, 5000, 40_000]), size // 2) // 4 * 4
            if conv:
                f, n = files.index("b.bin"), min(n, HALF)
                src_n, bf16 = n // 2, rng.random() < 0.5
                off = rng.randrange(0, (HALF - src_n) // 2) * 2 + (HALF if bf16 else 0)
                segs.append(Segment(f, off, n, at, "bfloat16" if bf16 else "float16",
                                    "float32"))
            else:
                off = rng.randrange(0, size - n)
                segs.append(Segment(f, off, n, at))
            at += n + rng.choice([0, 0, 60, 4096 + 4])
            at = -(-at // 4) * 4
        tail = rng.choice([0, 0, 100, 9000])
        slab = GatherSlab((i // 4, i % 4), at + tail, "", tuple(segs))
        slabs.append(GatherSlab(slab.key, slab.nbytes,
                                slab_checksum(_expected(files, data, slab)), slab.segments))
    # a segment ending at the (unaligned) end of the last file
    last = len(files) - 1
    n = data[files[last]].numel()
    eof = GatherSlab((9, 0), 4100, "", (Segment(last, n - 4000, 4000, 64),))
    slabs.append(GatherSlab(eof.key, eof.nbytes, slab_checksum(_expected(files, data, eof)),
                            eof.segments))
    return slabs


def _read(src, key):
    dst = torch.empty(src.nbytes(key), dtype=torch.uint8)
    src.read_into(key, dst)
    return dst


@pytest.fixture()
def world(tmp_path):
    root = tmp_path / "model"
    files, data = _files(root)
    slabs = _slabs(files, data)
    index = GatherIndex.build(root, files, slabs, meta={"what": "test"})
    index_path = index.save(tmp_path / "store" / "idx.json")
    return root, files, data, slabs, index_path


def _check_all(src, files, data, slabs, rounds=2, seed=0):
    keys = [s.key for s in slabs] * rounds
    random.Random(seed).shuffle(keys)
    for key in keys:
        want = _expected(files, data, src.index.entry(key))
        assert torch.equal(_read(src, key), want), key


HAVE_NATIVE = gather_mod.native_gather() is not None
needs_native = pytest.mark.skipif(not HAVE_NATIVE, reason="no C compiler: native gather off")


@needs_fadvise
@pytest.mark.parametrize("native", [False, pytest.param(True, marks=needs_native)])
@pytest.mark.parametrize("direct", [False, "auto"])
@pytest.mark.parametrize("chunk", [4096, 64 << 10, 1 << 20])
def test_gather_matches_files_any_order(world, direct, chunk, native):
    root, files, data, slabs, index_path = world
    assert any(s.offset % 4096 for e in slabs for s in e.segments)  # misaligned offsets
    assert any(s.src_dtype for e in slabs for s in e.segments)  # widened ranges
    assert any(s.nbytes > chunk for e in slabs for s in e.segments) or chunk > 40_000
    with GatherSource(index_path, direct=direct, chunk_bytes=chunk,
                      ram_budget=8 * chunk, native=native) as src:
        assert isinstance(src, SlabSource)
        assert sorted(src.keys()) == sorted(s.key for s in slabs)
        _check_all(src, files, data, slabs)
        assert src.stats["verified"] == len(slabs)
        assert src.stats["source"] == "in-place"
        assert src.stats["reader"] == ("native" if native else "python")


def _scratch(n):
    import mmap

    return torch.frombuffer(mmap.mmap(-1, n), dtype=torch.uint8)


@pytest.mark.parametrize("native", [False, pytest.param(True, marks=needs_native)])
@pytest.mark.parametrize("cap", [7 * 4096, 9 * 4096, 64 << 10])
def test_plan_batches_through_a_small_scratch(world, native, cap):
    """A scratch smaller than a chunk's reads (but not than one read) splits the plan into
    batches; every chunk of every slab, at any start, is the slab's bytes."""
    root, files, data, slabs, index_path = world
    fds = [os.open(root / f, os.O_RDONLY) for f in files]
    try:
        scratch = _scratch(cap)
        multi = 0
        for slab in slabs:
            want = _expected(files, data, slab)
            for start in range(0, slab.nbytes, 4096 * 3):
                n = min(4096 * 5, slab.nbytes - start)
                plan = gather_mod.plan_gather(slab, start, n, cap)
                multi += len(plan) > 1
                out = torch.full((n,), 0xAB, dtype=torch.uint8)
                gather_mod.gather_into(slab, start, out, fds, scratch, native=native)
                assert torch.equal(out, want[start : start + n]), (slab.key, start)
        assert cap > 7 * 4096 or multi > 0
    finally:
        for fd in fds:
            os.close(fd)


@needs_native
@pytest.mark.parametrize("src_dtype", ["float16", "bfloat16"])
def test_native_widening_is_torchs_for_every_value(tmp_path, src_dtype):
    """Every non-NaN 16-bit pattern widens to the float32 bits torch gives it (NaNs are
    never widened: the index builder refuses them), at an even and an odd file offset."""
    dt = getattr(torch, src_dtype)
    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16)
    vals = bits.view(dt)
    vals = vals[~torch.isnan(vals.float())]
    raw = vals.contiguous().view(torch.uint8)
    path = tmp_path / "w.bin"
    path.write_bytes(b"\x07" + raw.numpy().tobytes())
    want = vals.to(torch.float32).view(torch.uint8)
    n = vals.numel() * 4
    fd = os.open(path, os.O_RDONLY)
    try:
        for off in (1, 0):  # odd: the values sit misaligned in the scratch
            if off == 0:
                path.write_bytes(raw.numpy().tobytes())
            slab = GatherSlab((0, 0), n, "", (Segment(0, off, n, 0, src_dtype, "float32"),))
            for native in (True, False):
                out = torch.zeros(n, dtype=torch.uint8)
                gather_mod.gather_into(slab, 0, out, [fd], _scratch(1 << 20), native=native)
                assert torch.equal(out, want), (off, native)
    finally:
        os.close(fd)


@needs_native
def test_native_read_error_is_an_oserror(world):
    root, files, data, slabs, index_path = world
    fd = os.open(root / files[0], os.O_RDONLY)
    os.close(fd)  # a closed descriptor: EBADF from pread
    slab = GatherSlab((0, 0), 64, "", (Segment(0, 0, 64, 0),))
    with pytest.raises(OSError):
        gather_mod.gather_into(slab, 0, torch.empty(64, dtype=torch.uint8), [fd],
                               _scratch(1 << 16), native=True)


@needs_fadvise
def test_native_off_falls_back_to_python(world, monkeypatch):
    root, files, data, slabs, index_path = world
    monkeypatch.setenv("POROS_SSS_NATIVE", "0")
    monkeypatch.setattr(gather_mod, "_NATIVE", [])
    monkeypatch.setattr(gather_mod, "_NATIVE_REASON", [])
    assert gather_mod.native_gather() is None
    assert "POROS_SSS_NATIVE=0" in gather_mod.native_gather_reason()
    with GatherSource(index_path, chunk_bytes=8192, ram_budget=1 << 20) as src:
        assert src.stats["reader"] == "python"
        _check_all(src, files, data, slabs, rounds=1)
    with pytest.raises(RuntimeError, match="native gather unavailable"):
        GatherSource(index_path, native=True)


@pytest.mark.skipif(sys.platform == "darwin", reason="the native reader is not built on darwin "
                    "(poros.sss.gather): a Mac keeps its own pack")
def test_native_builds_into_its_cache_dir(tmp_path, monkeypatch):
    if not (os.environ.get("CC") or __import__("shutil").which("cc")):
        pytest.skip("no C compiler")
    monkeypatch.setenv("POROS_SSS_NATIVE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(gather_mod, "_NATIVE", [])
    monkeypatch.setattr(gather_mod, "_NATIVE_REASON", [])
    assert gather_mod.native_gather() is not None
    assert len(list((tmp_path / "cache").glob("_gather-*.so"))) == 1
    assert (tmp_path / "cache").stat().st_mode & 0o777 == 0o700


@pytest.mark.cuda_host  # the SSD store's reader is Linux's (os.posix_fadvise): no Mac has it
@pytest.mark.parametrize("planted", ["in_a_shared_folder", "writable_by_others"])
def test_native_never_loads_a_library_another_user_could_have_written(tmp_path, monkeypatch,
                                                                      planted):
    """A cached library in a folder another user can write, or itself writable by others, is
    their code: the reader is built afresh (in a private temp folder, or over it) and what was
    planted is never loaded."""
    import hashlib
    import platform
    import shutil
    import tempfile

    if not (os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")):
        pytest.skip("no C compiler")
    cache, scratch = tmp_path / "cache", tmp_path / "tmp"
    cache.mkdir()
    scratch.mkdir()
    # _load_native's name for it; a different one fails below (a second library appears)
    code = Path(gather_mod.__file__).with_name("_gather.c").read_bytes()
    tag = hashlib.sha256(code + platform.machine().encode()).hexdigest()[:16]
    lib = cache / f"_gather-{tag}.so"
    lib.write_bytes(b"not a library: another user's code")  # loaded, CDLL would refuse it
    os.chmod(cache, 0o777 if planted == "in_a_shared_folder" else 0o700)
    os.chmod(lib, 0o644 if planted == "in_a_shared_folder" else 0o666)
    monkeypatch.setenv("POROS_SSS_NATIVE_DIR", str(cache))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    fn, reason = gather_mod._load_native()
    assert fn is not None and reason is None, reason
    assert list(cache.glob("_gather-*.so")) == [lib] and not list(scratch.iterdir())
    if planted == "in_a_shared_folder":  # left as it was, and never loaded
        assert lib.read_bytes() == b"not a library: another user's code"
    else:  # rebuilt over it, this user's alone
        assert lib.read_bytes().startswith(b"\x7fELF") and lib.stat().st_mode & 0o777 == 0o755


@needs_fadvise
def test_schedule_replay_and_hints_keep_bytes(world):
    """Prefetch, hints and the replayed schedule only change when bytes move."""
    root, files, data, slabs, index_path = world
    keys = [s.key for s in slabs]
    with GatherSource(index_path, chunk_bytes=16 << 10, ram_budget=1 << 20, depth=4) as src:
        for _ in range(3):
            src.prefetch(keys[:3])
            for key in keys:
                want = _expected(files, data, src.index.entry(key))
                assert torch.equal(_read(src, key), want)
            src.mark_step()
        assert src.stats["prefetch_hits"] > 0


@needs_fadvise
def test_corrupt_source_byte_fails_first_read(world):
    root, files, data, slabs, index_path = world
    slab = slabs[0]
    seg = slab.segments[0]
    path = root / files[seg.file]
    st = path.stat()
    raw = bytearray(path.read_bytes())
    raw[seg.offset + seg.src_nbytes // 2] ^= 0x01
    path.write_bytes(bytes(raw))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # silent corruption: same mtime
    with GatherSource(index_path) as src:
        with pytest.raises(SssCorruptError):
            _read(src, slab.key)
    with GatherSource(index_path, verify="never") as src:  # only the checksum catches it
        assert not torch.equal(_read(src, slab.key), _expected(files, data, slab))


def test_changed_source_file_is_stale(world):
    root, files, data, slabs, index_path = world
    os.utime(root / files[1], ns=(1, 1))
    with pytest.raises(GatherIndexStale):
        GatherSource(index_path)
    with pytest.raises(GatherIndexStale):
        GatherIndex.load(index_path)
    GatherIndex.load(index_path, check=False)  # still readable as JSON


def test_index_round_trip_and_never_inside_root(world):
    root, files, data, slabs, index_path = world
    idx = GatherIndex.load(index_path)
    assert idx.entries == tuple(slabs) and idx.meta == {"what": "test"}
    assert idx.data_nbytes == 4096 + sum(-(-s.nbytes // 4096) * 4096 for s in slabs)
    with pytest.raises(ValueError, match="inside the source directory"):
        idx.save(root / "idx.json")
    with pytest.raises(ValueError, match="inside the source directory"):
        idx.save(root / "sub" / "idx.json")


def test_segments_validate():
    with pytest.raises(ValueError):
        Segment(0, 0, 6, 0, "float16", "float32")  # not whole float32 values
    with pytest.raises(ValueError):
        Segment(0, 0, 8, 0, "float32", "float16")  # narrowing is not exact
    with pytest.raises(ValueError):
        GatherSlab((0, 0), 100, "", (Segment(0, 0, 64, 0), Segment(0, 0, 64, 32)))
    with pytest.raises(ValueError):
        GatherSlab((0, 0), 100, "", (Segment(0, 0, 64, 64),))


def test_o_direct_on_a_real_filesystem(tmp_path):
    d = os.environ.get("POROS_SSS_TEST_DIR")
    if not d:
        pytest.skip("set POROS_SSS_TEST_DIR to a local ext4/xfs dir to test O_DIRECT")
    root = Path(d) / f"gather_test_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        files, data = _files(root, seed=3)
        slabs = _slabs(files, data, seed=4)
        index_path = GatherIndex.build(root, files, slabs).save(tmp_path / "idx.json")
        with GatherSource(index_path, direct=True, chunk_bytes=8192, ram_budget=1 << 20) as src:
            assert src.direct and src.stats["direct_files"] == len(files)
            _check_all(src, files, data, slabs)
    finally:
        for name in ("a.bin", "b.bin", "sub/c.bin"):
            (root / name).unlink(missing_ok=True)
        (root / "sub").rmdir()
        root.rmdir()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_dst(world):
    root, files, data, slabs, index_path = world
    with GatherSource(index_path, chunk_bytes=8192, ram_budget=1 << 20) as src:
        for s in slabs:
            dst = torch.empty(s.nbytes, dtype=torch.uint8, device="cuda")
            src.read_into(s.key, dst)
            assert torch.equal(dst.cpu(), _expected(files, data, s))


# -- raw ring: the slab's file ranges land in the ring, assembled on the copy out ----------


@pytest.fixture()
def plain_world(tmp_path):
    """Slabs without widened sections (what a raw ring takes), zero gaps included."""
    root = tmp_path / "model"
    files, data = _files(root, seed=5)
    slabs = [s for s in _slabs(files, data, nslabs=24, seed=6)
             if not any(x.src_dtype for x in s.segments)]
    assert len(slabs) >= 6
    index = GatherIndex.build(root, files, slabs)
    return root, files, data, slabs, index.save(tmp_path / "store" / "idx.json")


@needs_fadvise
@pytest.mark.parametrize("native", [False, pytest.param(None, marks=needs_native)])
@pytest.mark.parametrize("raw", [None, True, False])
def test_raw_ring_matches_files(plain_world, native, raw):
    root, files, data, slabs, index_path = plain_world
    with GatherSource(index_path, chunk_bytes=4096, ram_budget=1 << 20, native=native,
                      raw_ring=raw, depth=3) as src:
        assert src.stats["ring"] == ("assembled" if raw is False else "raw")
        if raw is not False:  # one chunk holds a slab's raw ranges
            assert src.chunk_bytes >= max(src._ring_nbytes(s) for s in slabs)
        keys = [s.key for s in slabs]
        for _ in range(3):  # cold, then the replayed schedule with prefetch
            src.prefetch(keys[:2])
            _check_all(src, files, data, slabs, rounds=1)
            src.mark_step()
        assert src.stats["verified"] == len(slabs)
        assert src.stats["prefetch_hits"] > 0


@needs_fadvise
def test_raw_ring_device_assembly_is_the_slab(plain_world):
    """The GPU assembly (a ``torch.cat`` of ring views and zeros), run on CPU tensors."""
    root, files, data, slabs, index_path = plain_world
    with GatherSource(index_path, raw_ring=True) as src:
        for s in slabs:
            rp = src._raw_plans[s.key]
            # page-aligned, as the source's own ring is: its raw reads land there by O_DIRECT
            ring = _scratch(rp.ring_nbytes).fill_(0x5A)
            src._read_chunk(s, 0, memoryview(ring.numpy()))
            dst = torch.full((s.nbytes,), 0xA5, dtype=torch.uint8)
            src._assemble_device(rp, ring, dst)
            assert torch.equal(dst, _expected(files, data, s)), s.key
        assert any(p[0] < 0 for rp in src._raw_plans.values() for p in rp.parts)  # zeros


@needs_fadvise
def test_raw_ring_corrupt_byte_fails_first_read(plain_world):
    root, files, data, slabs, index_path = plain_world
    slab = slabs[0]
    seg = slab.segments[0]
    path = root / files[seg.file]
    st = path.stat()
    raw = bytearray(path.read_bytes())
    raw[seg.offset + seg.nbytes // 2] ^= 0x01
    path.write_bytes(bytes(raw))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    with GatherSource(index_path, raw_ring=True) as src:
        with pytest.raises(SssCorruptError):
            _read(src, slab.key)
        _read(src, slabs[1].key)  # the ring slot came back


def test_raw_ring_refuses_widened_sections(world):
    root, files, data, slabs, index_path = world
    with pytest.raises(ValueError, match="raw_ring=True"):
        GatherSource(index_path, raw_ring=True)
    with GatherSource(index_path) as src:  # auto: host assembly
        assert src.stats["ring"] == "assembled"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("raw", [True, False])
def test_cuda_dst_raw_ring(plain_world, raw):
    root, files, data, slabs, index_path = plain_world
    with GatherSource(index_path, raw_ring=raw, ram_budget=1 << 20, depth=3) as src:
        for _ in range(2):
            for s in slabs:
                dst = torch.full((s.nbytes,), 7, dtype=torch.uint8, device="cuda")
                src.read_into(s.key, dst)
                assert torch.equal(dst.cpu(), _expected(files, data, s)), s.key
            src.mark_step()
