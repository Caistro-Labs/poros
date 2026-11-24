"""``SsdSource`` (CPU; the CUDA copy path is a skipif test).

For the streaming reader: every read, in any
order, prefetched or not, through any ring size, is ``torch.equal`` to the
bytes written. ``POROS_SSS_TEST_DIR`` points the O_DIRECT tests at a real
local filesystem (tmpfs refuses O_DIRECT; the default tmp dir then
exercises the buffered fallback).
"""

from __future__ import annotations

import os
import random
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from poros.sss import SlabSource, SsdSource, SssCorruptError, SssFile, SssWriter  # noqa: E402
from poros.sss.probe import probe, probe_disk  # noqa: E402
from poros.sss.tune import grown_capacity, prefetch_depth  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="no posix_fadvise")

META = {"model_id": "tiny", "revision_sha": "abc"}


def _slabs(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
            for k, n in sizes.items()}


def _moe_sizes(layers=4, experts=6, nbytes=10_000):
    return {(layer, e): nbytes + 37 * e for layer in range(layers) for e in range(experts)}


def _write(dirpath: Path, slabs, name="model.sss"):
    w = SssWriter(dirpath / name, [(k, v.numel()) for k, v in slabs.items()], meta=META)
    for k in sorted(slabs, reverse=True):
        w.write(k, 0, slabs[k])
    w.finish().close()
    return dirpath / name


def _read(src, key, device="cpu"):
    dst = torch.empty(src.nbytes(key), dtype=torch.uint8, device=device)
    src.read_into(key, dst)
    return dst


def _step(layers, experts, cached_fwd=0, cached_bwd=1):
    """One MoE micro-step's reads: forward layers ascending, backward descending."""
    fwd = [(layer, e) for layer in range(cached_fwd, layers) for e in range(experts)]
    bwd = [(layer, e) for layer in reversed(range(layers - cached_bwd)) for e in range(experts)]
    return fwd + bwd


def _direct_dir():
    d = os.environ.get("POROS_SSS_TEST_DIR")
    if not d:
        pytest.skip("set POROS_SSS_TEST_DIR to a local ext4/xfs dir to test O_DIRECT")
    path = Path(d) / f"sss_test_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("direct", [False, "auto"])
def test_every_key_matches_in_any_order(tmp_path, direct):
    slabs = _slabs({**_moe_sizes(), (9, 0): 1, (9, 1): 4096, (9, 2): 300_000})
    path = _write(tmp_path, slabs)
    keys = list(slabs) * 2
    random.Random(0).shuffle(keys)
    with SsdSource(path, direct=direct, chunk_bytes=64 << 10, ram_budget=1 << 20) as src:
        assert isinstance(src, SlabSource)
        assert sorted(src.keys()) == sorted(slabs)
        for key in keys:
            assert torch.equal(_read(src, key), slabs[key]), key
        assert src.stats["sync_reads"] == len(keys)  # no trace yet: all on demand
        assert src.stats["verified"] == len(slabs)  # "first": once per slab


def test_buffered_fallback_when_odirect_is_refused(tmp_path, monkeypatch):
    slabs = _slabs(_moe_sizes(1, 3))
    path = _write(tmp_path, slabs)
    real_open = os.open

    def refuse_direct(p, flags, *args):  # what overlay roots and old tmpfs do
        if flags & getattr(os, "O_DIRECT", 0):
            raise OSError(22, "Invalid argument")
        return real_open(p, flags, *args)

    monkeypatch.setattr(os, "open", refuse_direct)
    with SsdSource(path) as src:
        assert not src.direct and "O_DIRECT" in src.fallback_reason
        for key in slabs:
            assert torch.equal(_read(src, key), slabs[key])
    with pytest.raises(OSError):
        SsdSource(path, direct=True)
    res = probe_disk(path, seconds=0.1, queue_depth=2, read_bytes=4096)
    assert not res.direct and res.b_ssd > 0


def test_odirect_reads_match():
    d = _direct_dir()
    slabs = _slabs({**_moe_sizes(2, 3), (5, 0): 3 * 4096 + 1})
    path = _write(d, slabs)
    try:
        with SsdSource(path, direct=True, chunk_bytes=8192) as src:
            assert src.direct
            for key in list(slabs) * 2:
                assert torch.equal(_read(src, key), slabs[key]), key
        res = probe_disk(path, seconds=0.3, queue_depth=4, read_bytes=8192)
        assert res.direct and res.b_ssd > 0
    finally:
        for p in d.iterdir():
            p.unlink()
        d.rmdir()


def test_schedule_replay_prefetches_the_steady_state(tmp_path):
    layers, experts = 5, 6
    slabs = _slabs(_moe_sizes(layers, experts))
    path = _write(tmp_path, slabs)
    with SsdSource(path, depth=4, chunk_bytes=16 << 10, ram_budget=4 << 20) as src:
        for key in _step(layers, experts):  # step 1: recorded, synchronous
            assert torch.equal(_read(src, key), slabs[key])
        src.mark_step()
        assert not src.stats["recording"] and src.stats["trace_len"] == len(_step(layers,
                                                                                    experts))
        hits = []
        for _ in range(4):  # steady state: the first layer stays cached in VRAM
            before = src.stats["prefetch_hits"]
            sync_before = src.stats["sync_reads"]
            for key in _step(layers, experts, cached_fwd=1):
                assert torch.equal(_read(src, key), slabs[key]), key
            src.mark_step()
            hits.append((src.stats["prefetch_hits"] - before,
                         src.stats["sync_reads"] - sync_before))
        reads = len(_step(layers, experts, cached_fwd=1))
        # the first replayed pass misses once where the cached layer begins;
        # once a pass of the steady state is the schedule, every read hits
        assert hits[0] == (reads - 1, 1) and hits[2:] == [(reads, 0)] * 2, hits
        assert src.stats["verified"] == len(slabs)


def test_router_noise_keeps_the_schedule(tmp_path):
    """Experts no token picked are passed over; extra keys read synchronously."""
    layers, experts = 4, 16
    slabs = _slabs(_moe_sizes(layers, experts, 3000))
    path = _write(tmp_path, slabs)
    rng = random.Random(3)
    with SsdSource(path, depth=6, chunk_bytes=4096, ram_budget=1 << 20) as src:
        for key in _step(layers, experts):
            _read(src, key)
        src.mark_step()
        for _ in range(5):
            hits, syncs = src.stats["prefetch_hits"], src.stats["sync_reads"]
            order = [k for k in _step(layers, experts, cached_fwd=1) if rng.random() > 0.1]
            for _ in range(3):  # a key read out of its place in the schedule
                order.insert(rng.randrange(len(order)), (rng.randrange(layers), 0))
            for key in order:
                assert torch.equal(_read(src, key), slabs[key]), key
            src.mark_step()
            hit = src.stats["prefetch_hits"] - hits
            assert hit >= 0.8 * len(order), (hit, src.stats["sync_reads"] - syncs, len(order))


def test_reads_off_the_trace_stay_exact(tmp_path):
    layers, experts = 3, 4
    slabs = _slabs(_moe_sizes(layers, experts))
    path = _write(tmp_path, slabs)
    rng = random.Random(1)
    with SsdSource(path, depth=3, chunk_bytes=16 << 10, ram_budget=128 << 10) as src:
        for key in _step(layers, experts):
            _read(src, key)
        src.mark_step()
        for _ in range(3):
            order = _step(layers, experts)
            rng.shuffle(order)  # an eval pass or a router-dependent miss pattern
            for key in order + _step(layers, experts):
                assert torch.equal(_read(src, key), slabs[key]), key
            src.mark_step()
        src.prefetch([(0, 0), (1, 1)])
        src.release((0, 0))
        assert torch.equal(_read(src, (1, 1)), slabs[(1, 1)])
        assert torch.equal(_read(src, (0, 0)), slabs[(0, 0)])


def test_slab_larger_than_the_ring_streams_through_it(tmp_path):
    slabs = _slabs({(0, 0): 1_000_003, (0, 1): 5000, (1, 0): 70_000})
    path = _write(tmp_path, slabs)
    with SsdSource(path, chunk_bytes=8192, ram_budget=5 * 8192, threads=3) as src:
        assert src.num_slots == 5
        for _ in range(2):
            for key in [(0, 0), (0, 1), (1, 0)]:
                assert torch.equal(_read(src, key), slabs[key]), key
            src.mark_step()


def test_corruption_is_detected_once_then_every_read_with_always(tmp_path):
    slabs = _slabs(_moe_sizes(1, 3))
    path = _write(tmp_path, slabs)
    with SssFile.open(path) as f:
        e = f.entry((0, 1))
    with open(path, "r+b") as raw:
        raw.seek(e.offset + 123)
        b = raw.read(1)
        raw.seek(e.offset + 123)
        raw.write(bytes([b[0] ^ 1]))
    with SsdSource(path, chunk_bytes=4096) as src:
        with pytest.raises(SssCorruptError):
            _read(src, (0, 1))
        assert torch.equal(_read(src, (0, 2)), slabs[(0, 2)])  # the ring recovered
        with pytest.raises(ValueError):
            src.read_into((0, 2), torch.empty(5, dtype=torch.uint8))
        with pytest.raises(KeyError):
            src.nbytes((7, 7))
    with SsdSource(path, verify="never") as src:
        assert not torch.equal(_read(src, (0, 1)), slabs[(0, 1)])


def test_a_one_chunk_slab_is_checksummed_by_its_reader_thread(tmp_path, monkeypatch):
    """Every slab here lands in one ring chunk: its reader thread takes the checksum, on demand
    and prefetched alike, once per process (verify="first") or on every read ("always"), and a
    flipped byte is still refused."""
    slabs = _slabs(_moe_sizes(1, 4))
    path = _write(tmp_path, slabs)
    with SssFile.open(path) as f:
        e = f.entry((0, 1))
    with open(path, "r+b") as raw:
        raw.seek(e.offset + 77)
        b = raw.read(1)
        raw.seek(e.offset + 77)
        raw.write(bytes([b[0] ^ 1]))
    on_main = []
    real = SsdSource._read_hashed

    def spy(self, entry, start, buf):
        on_main.append(threading.current_thread() is threading.main_thread())
        return real(self, entry, start, buf)

    monkeypatch.setattr(SsdSource, "_read_hashed", spy)
    with SsdSource(path) as src:  # the default 8 MiB chunk: one chunk a slab
        assert torch.equal(_read(src, (0, 0)), slabs[(0, 0)])  # on demand
        src.prefetch([(0, 1), (0, 2)])
        with pytest.raises(SssCorruptError):
            _read(src, (0, 1))  # prefetched, refused
        assert torch.equal(_read(src, (0, 2)), slabs[(0, 2)])
        hashed = len(on_main)
        assert hashed == 3 and src.stats["verified"] == 2
        assert torch.equal(_read(src, (0, 0)), slabs[(0, 0)])
        assert len(on_main) == hashed  # verified once: not hashed again
    with SsdSource(path, verify="always") as src:
        for _ in range(2):
            assert torch.equal(_read(src, (0, 3)), slabs[(0, 3)])
    assert len(on_main) == hashed + 2 and not any(on_main)
    with SsdSource(path, chunk_bytes=4096) as src:  # several chunks a slab: hashed on the copy
        with pytest.raises(SssCorruptError):
            _read(src, (0, 1))
    assert len(on_main) == hashed + 2


def _disk_reads(monkeypatch) -> list:
    """Every chunk read that went to the file (the host cache serves the others)."""
    seen, real = [], SsdSource._read_chunk

    def counted(self, e, start, buf):
        seen.append(e.key)
        return real(self, e, start, buf)

    monkeypatch.setattr(SsdSource, "_read_chunk", counted)
    return seen


def _wait_for_the_host_cache(src) -> None:
    """Wait until the host cache's slots have joined the ring. On a CUDA host they are page-locked
    on a thread of their own, which the first read starts, and join the ring piece by piece as each
    is registered (``SsdSource._pin_cache``: no read waits for it); until then reads go through
    the ring alone, the same bytes, and a slab it has to give up is read from disk again. Where
    nothing is pinned the cache's slots are free from the start: no thread, nothing to wait for."""
    thread = src._cache_thread
    if thread is not None:
        thread.join(timeout=60)
        assert not thread.is_alive(), "the host cache was still being page-locked after 60 s"


def _host_cache_run(tmp_path, monkeypatch, cache_kb, after_read=None):
    """Forward up the layers, backward down, four steps, through a ring of 5 slabs (the model is
    30) and a host cache of ``cache_kb``. ``after_read(src, n)`` runs after the n-th read; by
    default the cache is waited for after the first (the read that starts its pinning).
    Returns the stats, the disk reads, the read count and the slabs."""
    if after_read is None:
        def after_read(src, n):
            if n == 1:
                _wait_for_the_host_cache(src)
    layers, experts = 6, 5
    slabs = _slabs(_moe_sizes(layers, experts))  # 30 slabs of ~10 KB
    path = _write(tmp_path, slabs)
    disk = _disk_reads(monkeypatch)
    reads = 0
    with SsdSource(path, depth=2, max_depth=2, chunk_bytes=16 << 10, ram_budget=64 << 10,
                   cache_bytes=cache_kb << 10) as src:  # a ring of 5 slabs: the model is 30
        assert src.stats["cache"] is (cache_kb > 0) and src.chunk_bytes == 3 * 4096
        assert src.num_slots == 5 + min((cache_kb << 10) // src.chunk_bytes, len(slabs))
        for _ in range(4):
            for key in _step(layers, experts):
                assert torch.equal(_read(src, key), slabs[key]), key
                reads += 1
                after_read(src, reads)
            src.mark_step()
        st = src.stats
    served = st["cache_hits"] + st["prefetch_hits"] + st["sync_reads"]
    assert served == reads == st["reads"]
    return st, disk, reads, slabs


def _assert_the_host_cache_counts(st, disk, reads, slabs, cache_kb):
    layers, experts = 6, 5
    if cache_kb == 0:
        assert st["cache_hits"] == st["cache_bytes"] == 0 and len(disk) >= reads
    elif cache_kb == 1024:  # holds the model
        assert sorted(disk) == sorted(slabs) and st["cache_hits"] == reads - len(slabs)
        step = sum(slabs[k].numel() for k in _step(layers, experts))
        assert st["cache_bytes"] == 4 * step - sum(v.numel() for v in slabs.values())
    else:  # holds part of it: some reads from RAM, the rest from disk, never a wrong byte
        assert 0 < st["cache_hits"] < reads - len(slabs) and len(disk) == reads - st["cache_hits"]


@pytest.mark.parametrize("cache_kb", [0, 64, 1024])
def test_the_host_cache_serves_what_it_still_holds(tmp_path, monkeypatch, cache_kb):
    """Forward up the layers, backward down, four steps: every read is the bytes written. With a
    host cache, a slab it still holds is served without a disk read (the backward's first layers,
    the next forward's first ones); once it holds the whole model every slab is read from disk
    once. Without one every read goes to disk, as before. On a CUDA host the cache is waited for
    after the first read (its slots join the ring once page-locked)."""
    st, disk, reads, slabs = _host_cache_run(tmp_path, monkeypatch, cache_kb)
    _assert_the_host_cache_counts(st, disk, reads, slabs, cache_kb)


def _pin_lands_when_waited_for(monkeypatch) -> threading.Event:
    """A CUDA host faked on the CPU, for the host cache's page-locking: the ring registers at once,
    on the reading thread; the cache's pieces, on their own thread, only once something joins that
    thread (a wait for the cache, or close) -- a pin slower than every read before the wait."""
    from poros.sss import pinning

    lands = threading.Event()

    class _Runtime:
        @staticmethod
        def cudaHostRegister(ptr, nbytes, flags):
            if threading.current_thread() is not threading.main_thread():
                assert lands.wait(timeout=60), "the faked pin was never waited for"
            return 0

        @staticmethod
        def cudaHostUnregister(ptr):
            return 0

    real_take = SsdSource._take

    def take(self, n, *, block):
        started = self._cache_thread
        out = real_take(self, n, block=block)
        thread = self._cache_thread
        if thread is not None and started is None:  # the read that started the pin
            join = thread.join

            def joined(timeout=None):
                lands.set()
                return join(timeout)

            thread.join = joined
        return out

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Runtime)
    monkeypatch.setattr(torch.cuda, "check_error", lambda res: None)
    monkeypatch.setattr(pinning, "reserve", lambda nbytes: True)
    monkeypatch.setattr(pinning, "release", lambda nbytes: None)
    monkeypatch.setattr(SsdSource, "_take", take)
    return lands


@pytest.mark.parametrize("cache_kb", [64, 1024])
def test_a_host_cache_pinned_off_thread_is_counted_once_it_has_joined(tmp_path, monkeypatch,
                                                                      cache_kb):
    """On a CUDA host the cache is page-locked on its own thread, and its slots join the ring when
    that is done, not at the first read: a count that takes the cache as there from the start
    fails on a slow pin (seen on a CUDA host: a slab of a model the cache holds read from disk
    twice), the 5-slab ring having given up slabs the cache would have kept. Forced here: the pin
    lands only once the run waits for it. Waited for after the first read, the counts are those of
    a host with nothing to pin; waited for only after the tenth, exactly layer 0 (evicted by then)
    is read from disk a second time, every byte right."""
    lands = _pin_lands_when_waited_for(monkeypatch)
    st, disk, reads, slabs = _host_cache_run(tmp_path, monkeypatch, cache_kb)
    assert st["pinned"] and st["cache"]  # the faked CUDA host's route: the cache pinned off-thread
    _assert_the_host_cache_counts(st, disk, reads, slabs, cache_kb)
    if cache_kb == 1024:
        lands.clear()
        (tmp_path / "late").mkdir()

        def late(src, n):
            if n == 10:
                _wait_for_the_host_cache(src)

        st, disk, reads, slabs = _host_cache_run(tmp_path / "late", monkeypatch, cache_kb, late)
        layer0 = [k for k in slabs if k[0] == 0]
        assert sorted(disk) == sorted([*slabs, *layer0])
        assert st["cache_hits"] == reads - len(slabs) - len(layer0)


def test_ready_says_what_needs_no_disk_read(tmp_path):
    """``ready``: a slab in the host cache, or whose read has landed, can be handed over without
    waiting on the disk; one never read cannot. Asking reads nothing."""
    import time

    slabs = _slabs(_moe_sizes(1, 4))
    path = _write(tmp_path, slabs)
    with SsdSource(path, cache_bytes=1 << 20) as src:
        assert not any(src.ready(k) for k in slabs)
        assert torch.equal(_read(src, (0, 0)), slabs[(0, 0)])
        assert src.ready((0, 0))  # held by the host cache
        src.prefetch([(0, 2)])
        deadline = time.time() + 10
        while not src.ready((0, 2)) and time.time() < deadline:
            time.sleep(0.01)
        assert src.ready((0, 2)) and not src.ready((0, 1))  # landed; never asked for
        assert src.stats["reads"] == 1
        assert torch.equal(_read(src, (0, 2)), slabs[(0, 2)])


def test_a_slab_that_fails_its_checksum_is_never_cached(tmp_path):
    slabs = _slabs(_moe_sizes(1, 3))
    path = _write(tmp_path, slabs)
    with SssFile.open(path) as f:
        e = f.entry((0, 1))
    with open(path, "r+b") as raw:
        raw.seek(e.offset + 5)
        b = raw.read(1)
        raw.seek(e.offset + 5)
        raw.write(bytes([b[0] ^ 1]))
    with SsdSource(path, cache_bytes=1 << 20) as src:
        assert torch.equal(_read(src, (0, 0)), slabs[(0, 0)])
        for _ in range(2):  # refused every time: a bad slab is not kept
            with pytest.raises(SssCorruptError):
                _read(src, (0, 1))
        assert torch.equal(_read(src, (0, 0)), slabs[(0, 0)])
        assert src.stats["cache_hits"] == 1


def test_the_finalizer_never_joins_the_reader_threads(tmp_path, monkeypatch):
    """The collector can finalize a source inside threading's own bookkeeping: a thread starting
    holds a lock that joining a thread takes again. A finalizer that joined the readers there
    hung the process for good (a MoE test whose disk probe was starting its threads, on an RTX
    4090). The readers are idle by then and exit on their own once the pool is shut down."""
    src = SsdSource(_write(tmp_path, _slabs(_moe_sizes(layers=1, experts=2))))
    for key in ((0, 0), (0, 1)):
        _read(src, key)
    readers = [t for t in threading.enumerate() if t.name.startswith("poros-sss-read")]
    assert readers
    joined = []
    real = threading.Thread.join
    monkeypatch.setattr(threading.Thread, "join",
                        lambda self, timeout=None: joined.append(self) or real(self, timeout))
    src.__del__()
    monkeypatch.undo()
    assert src._closed and not [t for t in joined if t in readers]
    for t in readers:
        t.join(timeout=30)
        assert not t.is_alive()


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="lists open files by /proc")
def test_a_source_whose_ring_cannot_be_mapped_leaves_no_file_open(tmp_path, monkeypatch):
    """An __init__ that fails after the data file opened (the ring's mmap, out of memory) leaks
    no descriptor: the finalizer's close() closes the file though the rest was never made."""
    import errno
    import gc
    import mmap

    path = _write(tmp_path, _slabs(_moe_sizes(layers=1, experts=2)))

    def open_on_it():
        found = []
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}") == os.path.realpath(path):
                    found.append(fd)
            except OSError:  # the listing's own descriptor, gone
                pass
        return found

    real = mmap.mmap

    def no_ring(fileno, *a, **k):
        if fileno == -1:
            raise OSError(errno.ENOMEM, "Cannot allocate memory")
        return real(fileno, *a, **k)

    monkeypatch.setattr(mmap, "mmap", no_ring)
    try:
        SsdSource(path, direct=False)
    except OSError as e:
        assert e.errno == errno.ENOMEM
    else:
        pytest.fail("the ring was mapped")
    monkeypatch.undo()
    gc.collect()
    assert open_on_it() == []


def test_probe_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    store = tmp_path / "store"
    res = probe(store, seconds=0.2, queue_depth=4, read_bytes=64 << 10, file_bytes=1 << 20)
    assert res.b_ssd > 0 and not res.cached
    assert res.directory == str(store.resolve())  # created, probed in place
    assert not list(store.glob(".poros_sss_probe*"))  # the temp file is gone
    again = probe(store, seconds=0.2, queue_depth=4, read_bytes=64 << 10, file_bytes=1 << 20)
    assert again.cached and again.b_ssd == res.b_ssd
    assert again.bandwidth == (res.b_ssd if res.b_h2d is None else min(res.b_ssd, res.b_h2d))


def test_tuning_rules():
    # 2.5 MB experts at 1 GB/s (2.6 ms) consumed every 0.5 ms: ceil(5.3) + 2
    assert prefetch_depth(2.5e6, 1e9, 0.5e-3) == 8
    assert prefetch_depth(1, 1e12, 1.0) == 3  # never below triple buffering
    assert prefetch_depth(1e9, 1e6, 1e-3, d_max=16) == 16
    gib = 1 << 30
    assert grown_capacity(gib, 24 * gib, 10 * gib, 100 * gib) == 12 * gib
    assert grown_capacity(gib, 24 * gib, 10 * gib, 5 * gib) == 5 * gib
    assert grown_capacity(gib, 24 * gib, 23 * gib, 100 * gib) == gib  # never shrinks


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_reads_on_a_side_stream_match():
    d = Path(os.environ.get("POROS_SSS_TEST_DIR") or "/tmp")
    d = d / f"sss_cuda_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    layers, experts = 4, 8
    slabs = _slabs(_moe_sizes(layers, experts, 200_000))
    path = _write(d, slabs)
    from poros.sss import pinning

    pinning.reset()  # this host's own page-locked budget
    host_pins = pinning.status()["env"] != 0  # 0: none (WSL2 whose child probe failed, or set)
    try:
        with SsdSource(path, depth=6, chunk_bytes=64 << 10, ram_budget=8 << 20) as src:
            if host_pins:
                assert src.pinned
            else:  # the ring falls back to pageable memory, and the run's record says so
                staged = pinning.staging(ring_pinned=src.stats["pinned"])
                assert not src.pinned and staged["mode"] == "unpinned", staged
                assert staged["pinned_budget_bytes"] == 0 and staged["refusals"] > 0, staged
            side = torch.cuda.Stream()
            for step in range(4):
                order = _step(layers, experts, cached_fwd=0 if step == 0 else 1)
                outs = []
                with torch.cuda.stream(side):
                    for key in order:
                        outs.append((key, _read(src, key, "cuda")))
                torch.cuda.current_stream().wait_stream(side)
                for key, out in outs:
                    assert torch.equal(out.cpu(), slabs[key]), key
                src.mark_step()
            assert src.stats["prefetch_hits"] > 0
    finally:
        for p in d.iterdir():
            p.unlink()
        d.rmdir()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_host_cache_is_pinned_off_thread_and_serves_the_same_bytes():
    """On CUDA the host cache is page-locked a piece at a time on its own thread; its slots join
    the ring as they are ready. Every read is the bytes written, the re-reads it holds come from
    RAM, and close waits for the thread and unregisters every piece. It starts at the first
    read, not during the load."""
    import poros.sss.ssd as ssd_mod

    d = Path(os.environ.get("POROS_SSS_TEST_DIR") or "/tmp") / f"sss_cache_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    layers, experts = 4, 8
    slabs = _slabs(_moe_sizes(layers, experts, 200_000))
    path = _write(d, slabs)
    from poros.sss import pinning

    pinning.reset()
    try:
        with SsdSource(path, depth=6, chunk_bytes=256 << 10, ram_budget=2 << 20,
                       cache_bytes=8 << 20) as src:
            if not src.pinned:
                pytest.skip("this host pins nothing (the ring is pageable): no cache to pin")
            assert src._cache_thread is None and src.stats["pinned_bytes"] == src.pinned_bytes
            side = torch.cuda.Stream()
            for step in range(3):
                if step == 1:  # the first read started it
                    src._cache_thread.join(timeout=60)
                    assert src.stats["pinned_bytes"] > src.pinned_bytes  # the cache's pieces
                outs = []
                with torch.cuda.stream(side):
                    for key in _step(layers, experts):
                        outs.append((key, _read(src, key, "cuda")))
                torch.cuda.current_stream().wait_stream(side)
                for key, out in outs:
                    assert torch.equal(out.cpu(), slabs[key]), key
                src.mark_step()
            assert src.stats["cache_hits"] > 0
            pieces = list(src._cache_pins)
        assert pieces and not src._cache_pins  # every piece unregistered by close
        assert ssd_mod._CACHE_PIECE > 0
    finally:
        for p in d.iterdir():
            p.unlink()
        d.rmdir()


def test_pinned_budget_finds_a_ceiling_and_keeps_headroom(monkeypatch):
    from poros.sss import pinning

    cap = 1536 << 20  # what a cloud 3090 container allows, about
    calls = []

    def register_ok(nbytes):
        calls.append(nbytes)
        return pinning.status()["granted"] + nbytes <= cap

    monkeypatch.setattr(pinning, "_register_ok", register_ok)
    monkeypatch.setattr(pinning, "_cuda", lambda: True)
    # a host that probes by registering: under WSL2 a child probe sets the budget instead
    # (test_wsl_child_probe_*), whatever this test's host is
    monkeypatch.setattr(pinning, "_wsl", lambda: False)
    monkeypatch.delenv("POROS_PIN_BUDGET", raising=False)
    pinning.reset()
    try:
        assert pinning.reserve(256 << 20)  # verified with a 1 GiB step + headroom
        assert pinning.reserve(512 << 20)  # inside the verified GiB: no probe
        assert not pinning.reserve(2 << 30)  # past the ceiling: refused, and pinning stops
        st = pinning.status()
        assert st["capped"] and st["granted"] == 768 << 20 and st["verified"] == 768 << 20
        assert st["ceiling_below"] > cap - st["granted"]
        n = len(calls)
        assert not pinning.reserve(1 << 20) and len(calls) == n  # no more probing
        monkeypatch.setenv("POROS_PIN_BUDGET", "0")
        pinning.reset()
        assert not pinning.reserve(1)
        monkeypatch.setenv("POROS_PIN_BUDGET", "1M")
        pinning.reset()
        assert pinning.reserve(1 << 20) and not pinning.reserve(1)
    finally:
        monkeypatch.delenv("POROS_PIN_BUDGET", raising=False)
        pinning.reset()
    assert pinning.pinned_size(3 << 20) == 4 << 20 and pinning.pinned_size(1) == 1


def test_no_cuda_never_pins(monkeypatch):
    from poros.sss import pinning

    monkeypatch.setattr(pinning, "_cuda", lambda: False)
    pinning.reset()
    assert not pinning.reserve(4096) and not pinning.status()["capped"]
    pinning.reset()


def test_staging_modes_for_receipts():
    from poros.sss.pinning import staging

    off = staging({"granted": 0, "verified": 0, "capped": False, "refusals": 9, "env": 0},
                  ring_pinned=False)
    assert off["mode"] == "unpinned" and off["pinned_budget_bytes"] == 0
    assert off["budget_basis"] == "POROS_PIN_BUDGET"
    capped = staging({"granted": 5, "verified": 600, "capped": True, "refusals": 2, "env": None})
    assert capped["mode"] == "partial" and capped["pinned_budget_bytes"] == 600
    full = staging({"granted": 5, "verified": 1 << 30, "capped": False, "refusals": 0,
                    "env": None}, ring_pinned=True)
    assert full["mode"] == "pinned" and full["pinned_budget_bytes"] is None


def test_wsl_is_never_probed_in_process(monkeypatch):
    from poros.sss import pinning

    monkeypatch.delenv("POROS_PIN_BUDGET", raising=False)
    monkeypatch.setattr(pinning, "_wsl", lambda: True)
    monkeypatch.setattr(pinning, "_cuda", lambda: True)
    monkeypatch.setattr(pinning, "_register_ok", lambda n: pytest.fail("probed under WSL2"))
    monkeypatch.setattr(pinning, "_wsl_probe_result", None)
    monkeypatch.setattr(pinning, "_wsl_child_probe", lambda nbytes=0: False)
    pinning.reset()
    try:
        assert not pinning.reserve(1 << 20)
        st = pinning.staging()
        assert st["mode"] == "unpinned"
        assert st["budget_basis"].startswith("WSL2 (child probe failed")
        # and a CUDA run's receipt records it
        from poros.cli.train_cmd import _staging_receipt

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _staging_receipt() == {"host_staging": st}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _staging_receipt() == {}
    finally:
        pinning.reset()


def test_wsl_child_probe_ok_pins_from_free_ram(monkeypatch):
    from poros.sss import pinning

    monkeypatch.delenv("POROS_PIN_BUDGET", raising=False)
    monkeypatch.setattr(pinning, "_wsl", lambda: True)
    monkeypatch.setattr(pinning, "_cuda", lambda: True)
    monkeypatch.setattr(pinning, "_register_ok", lambda n: pytest.fail("probed under WSL2"))
    monkeypatch.setattr(pinning, "_wsl_probe_result", None)
    probes = []
    monkeypatch.setattr(pinning, "_wsl_child_probe", lambda nbytes=0: probes.append(1) or True)
    monkeypatch.setattr(pinning, "_ram_budget", lambda: 3 << 30)
    pinning.reset()
    try:
        assert pinning.reserve(2 << 30) and not pinning.reserve(2 << 30)
        pinning.reset()
        assert pinning.reserve(1 << 20) and len(probes) == 1  # one child probe per process
        assert pinning.staging()["budget_basis"].startswith("WSL2 (child probe pinned")
        monkeypatch.setenv("POROS_WSL_PIN_PROBE", "0")
        pinning.reset()
        assert not pinning.reserve(1)
    finally:
        monkeypatch.delenv("POROS_WSL_PIN_PROBE", raising=False)
        pinning.reset()


def test_router_hint_prefetches_a_layer_and_steers_the_window(tmp_path):
    layers, experts = 3, 16
    slabs = _slabs(_moe_sizes(layers, experts, 3000))
    path = _write(tmp_path, slabs)
    with SsdSource(path, depth=4, chunk_bytes=4096, ram_budget=1 << 20) as src:
        for key in _step(layers, experts):
            _read(src, key)
        src.mark_step()
        picked = [(1, e) for e in (0, 3, 4, 9, 15)]  # this step's routing for layer 1
        before = dict(src.stats)
        src.prefetch(picked)
        assert src.stats["hinted"] - before["hinted"] == len(picked)
        for key in picked:  # ensure() then reads them: all already in flight
            assert torch.equal(_read(src, key), slabs[key])
        assert src.stats["sync_reads"] == before["sync_reads"]
        assert src.stats["prefetch_hits"] - before["prefetch_hits"] == len(picked)
        # an unpicked expert of the hinted layer is never prefetched by the window
        assert (1, 7) not in src._inflight and (1, 8) not in src._inflight
        assert torch.equal(_read(src, (2, 0)), slabs[(2, 0)])


def test_a_hint_also_reads_the_next_phase(tmp_path):
    layers, experts = 3, 8
    slabs = _slabs(_moe_sizes(layers, experts, 3000))
    path = _write(tmp_path, slabs)
    with SsdSource(path, depth=3, chunk_bytes=4096, ram_budget=1 << 20) as src:
        for key in _step(layers, experts):
            _read(src, key)
        src.mark_step()
        src.prefetch([(0, e) for e in range(experts)])  # layer 0's phase starts
        assert all((1, e) in src._inflight for e in range(experts))  # layer 1 read ahead
        for key in _step(layers, experts):
            assert torch.equal(_read(src, key), slabs[key]), key
        assert src.stats["phase_ahead"] >= experts
