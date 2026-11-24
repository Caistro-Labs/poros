"""The slab interface, the RAM backend and the ``.sss`` file (CPU only).

Bytes: every key read from a ``.sss`` file, through
every reader, is ``torch.equal`` to the bytes that were written.
"""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from poros.sss import (  # noqa: E402
    ALIGNMENT,
    RamSink,
    RamSource,
    SlabSource,
    SssCorruptError,
    SssFile,
    SssFormatError,
    SssWriter,
    SyncFileSource,
)

META = {"model_id": "tiny", "revision_sha": "abc", "quant": {"kind": "nf4", "blocksize": 64}}


def _slabs(seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = {(0, 0): 1, (0, 1): 4095, (0, 2): 4096, (1, 0): 4097, (1, 1): 3 * 4096 + 17,
             (2, 0): 70_000}
    return {k: torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
            for k, n in sizes.items()}


def _write_out_of_order(sink, slabs):
    # tensor-major order with split writes, as the MoE loader fills its arena
    for half in (0, 1):
        for key in sorted(slabs, reverse=True):
            data = slabs[key]
            cut = data.numel() // 2
            part = data[:cut] if half == 0 else data[cut:]
            if part.numel():
                sink.write(key, 0 if half == 0 else cut, part)


def _write_file(tmp_path, slabs, meta=META):
    path = tmp_path / "model.sss"
    w = SssWriter(path, [(k, v.numel()) for k, v in slabs.items()], meta=meta)
    _write_out_of_order(w, slabs)
    return path, w.finish()


def test_ram_source_is_a_pure_byte_copy():
    slabs = _slabs()
    src = RamSource(slabs)
    assert isinstance(src, SlabSource)
    for key, data in slabs.items():
        dst = torch.empty(src.nbytes(key), dtype=torch.uint8)
        src.read_into(key, dst)
        assert torch.equal(dst, data)
    with pytest.raises(ValueError):
        src.read_into((0, 1), torch.empty(4094, dtype=torch.uint8))
    with pytest.raises(ValueError):
        src.read_into((0, 1), torch.empty(4095, dtype=torch.int8))
    with pytest.raises(KeyError):
        src.nbytes((9, 9))


def test_file_round_trip_every_reader(tmp_path):
    slabs = _slabs()
    path, src = _write_file(tmp_path, slabs)
    assert isinstance(src, SlabSource)
    assert sorted(src.keys()) == sorted(slabs)
    for key, data in slabs.items():
        dst = torch.empty(src.nbytes(key), dtype=torch.uint8)
        src.read_into(key, dst)
        assert torch.equal(dst, data), key
    ram = RamSource.from_sss(path)
    for key, data in slabs.items():
        assert torch.equal(ram.slab(key), data), key
    with SssFile.open(path) as f:
        f.verify_all()
        for e in f.entries:
            assert e.offset % ALIGNMENT == 0
        assert os.path.getsize(path) == json.loads((tmp_path / "model.sss.idx").read_text())[
            "data_nbytes"]
        with open(path, "rb") as raw:
            assert raw.read(4) == b"PSSS"


def test_from_sss_lays_slabs_back_to_back_as_a_first_load_does(tmp_path):
    """A reused file fills one carved chunk in file order (RamSink's), not a block per slab:
    the expert residency copies a run of adjacent slabs at once."""
    slabs = _slabs()
    path, _ = _write_file(tmp_path, slabs)
    ram = RamSource.from_sss(path, pin=False)
    with SssFile.open(path) as f:
        keys = [e.key for e in f.entries]
    for a, b in zip(keys, keys[1:]):
        assert ram.slab(b).data_ptr() == ram.slab(a).data_ptr() + ram.slab(a).numel(), (a, b)
    for key, data in slabs.items():
        assert torch.equal(ram.slab(key), data), key


def test_ram_sink_matches_file_bytes(tmp_path):
    slabs = _slabs(1)
    sink = RamSink([(k, v.numel()) for k, v in slabs.items()])
    _write_out_of_order(sink, slabs)
    ram = sink.finish()
    _, fsrc = _write_file(tmp_path, slabs)
    for key in slabs:
        dst = torch.empty(fsrc.nbytes(key), dtype=torch.uint8)
        fsrc.read_into(key, dst)
        assert torch.equal(ram.slab(key), dst)


def test_unfinished_write_leaves_nothing_visible(tmp_path):
    slabs = _slabs()
    path = tmp_path / "model.sss"
    with pytest.raises(RuntimeError):
        with SssWriter(path, [(k, v.numel()) for k, v in slabs.items()], meta=META) as w:
            w.write((0, 0), 0, slabs[(0, 0)])
            raise RuntimeError("crash mid-build")
    assert not path.exists() and not (tmp_path / "model.sss.idx").exists()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(SssFormatError):
        SssFile.open(path)


def test_missing_and_overlapping_writes_are_refused(tmp_path):
    slabs = _slabs()
    sizes = [(k, v.numel()) for k, v in slabs.items()]
    for sink in (RamSink(sizes), SssWriter(tmp_path / "a.sss", sizes, meta=META)):
        sink.write((2, 0), 0, slabs[(2, 0)][:100])
        with pytest.raises(ValueError, match="overlaps"):
            sink.write((2, 0), 50, slabs[(2, 0)][50:200])
        with pytest.raises(ValueError, match="outside"):
            sink.write((0, 0), 1, slabs[(0, 0)])
        with pytest.raises(ValueError, match="not fully written"):
            sink.finish()
        if isinstance(sink, SssWriter):
            sink.abort()
    with pytest.raises(ValueError, match="declared twice"):
        RamSink([((0, 0), 4), ((0, 0), 4)])


def test_corruption_is_detected(tmp_path):
    slabs = _slabs()
    path, _ = _write_file(tmp_path, slabs)
    with SssFile.open(path) as f:
        e = f.entry((1, 1))
    with open(path, "r+b") as raw:
        raw.seek(e.offset + 5000)
        b = raw.read(1)
        raw.seek(e.offset + 5000)
        raw.write(bytes([b[0] ^ 0xFF]))
    src = SyncFileSource(path)
    dst = torch.empty(e.nbytes, dtype=torch.uint8)
    with pytest.raises(SssCorruptError):
        src.read_into((1, 1), dst)
    with pytest.raises(SssCorruptError):
        RamSource.from_sss(path)
    src.read_into((0, 0), torch.empty(1, dtype=torch.uint8))  # other slabs still read


def test_truncated_or_foreign_files_are_refused(tmp_path):
    slabs = _slabs()
    path, _ = _write_file(tmp_path, slabs)
    with open(path, "r+b") as raw:
        raw.truncate(os.path.getsize(path) - ALIGNMENT)
    with pytest.raises(SssFormatError, match="bytes"):
        SssFile.open(path)
    idx = tmp_path / "model.sss.idx"
    d = json.loads(idx.read_text())
    d["version"] = 99
    idx.write_text(json.dumps(d))
    with pytest.raises(SssFormatError, match="version"):
        SssFile.open(path)


def test_reuse_key_is_the_exact_meta(tmp_path):
    slabs = _slabs()
    path, _ = _write_file(tmp_path, slabs)
    with SssFile.open(path) as f:
        assert f.matches(dict(META))
        assert not f.matches({**META, "revision_sha": "def"})
        assert not f.matches({**META, "quant": {"kind": "nf4", "blocksize": 128}})


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_file_source_into_cuda_matches_ram(tmp_path):
    slabs = _slabs()
    path, src = _write_file(tmp_path, slabs)
    ram = RamSource.from_sss(path)
    for key in slabs:
        a = torch.empty(src.nbytes(key), dtype=torch.uint8, device="cuda")
        b = torch.empty_like(a)
        src.read_into(key, a)
        ram.read_into(key, b)
        torch.cuda.synchronize()
        assert torch.equal(a, b), key
