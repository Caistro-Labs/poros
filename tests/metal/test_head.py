"""``poros.metal.head`` on CPU: what the trainer sees, chunk order, and the file rows."""

from __future__ import annotations

import gc
import os
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

import torch.nn.functional as F  # noqa: E402

from poros.metal.head import CHUNK_ROWS, ChunkedHead, EmbedRows, FileRows  # noqa: E402
from poros.metal.pack import EMBED_BLOCK, HEAD_BLOCK  # noqa: E402
from poros.sss.format import SssCorruptError, SssFile  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

PlainHead = tiny.arms.PlainHead

VOCAB = 40000
HIDDEN = 16


@pytest.fixture(scope="module")
def rows():
    return torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)


@pytest.fixture(scope="module")
def tiny_pack(tmp_path_factory):
    cfg = tiny.tiny_config()
    return cfg, tiny.packed(tmp_path_factory.mktemp("head"), cfg)[1]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _hidden(*shape, seed=0, **kwargs) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), **kwargs)


class _Recording:
    """Rows that remember what was sliced, in order."""

    def __init__(self, inner):
        self.inner, self.reads, self.shape = inner, [], inner.shape

    def __getitem__(self, index):
        self.reads.append((index.start, index.stop))
        return self.inner[index]


def test_spans_cover_the_vocabulary(rows):
    assert ChunkedHead(rows).spans == [
        (0, CHUNK_ROWS),
        (CHUNK_ROWS, 2 * CHUNK_ROWS),
        (2 * CHUNK_ROWS, VOCAB),
    ]


@pytest.mark.parametrize("module", [EmbedRows, ChunkedHead])
def test_stored_bf16_is_never_named_weight(rows, module):
    m = module(rows)
    assert m.weight.numel() == 0 and m.weight.dtype == torch.float32
    assert not m.weight.requires_grad
    assert next(m.parameters()) is m.weight  # trainer.py asks an embed for its device this way
    assert [t.numel() for t in m.state_dict().values()] == [0]


def test_embed_is_an_exact_widening(rows):
    ids = torch.tensor([[5, 39999, 5, 17, 0, 16384]])
    got = EmbedRows(rows)(ids)
    assert got.dtype == torch.float32 and not got.requires_grad
    assert torch.equal(_bits(got), _bits(rows[ids].to(torch.float32)))


def test_a_whole_table_reads_in_pieces_macos_accepts(tiny_pack, monkeypatch):
    import poros.metal.head as head

    sizes, preadv = [], os.preadv

    def spy(fd, buffers, pos):
        sizes.append(len(buffers[0]))
        return preadv(fd, buffers, pos)

    with SssFile.open(tiny_pack[1] / "model.sss") as file:
        rows = FileRows(file, (EMBED_BLOCK, 0))
        whole = rows[0 : rows.shape[0]].clone()
        monkeypatch.setattr(head, "_PIECE", 64)
        monkeypatch.setattr(head.os, "preadv", spy)
        assert torch.equal(rows[0 : rows.shape[0]], whole)
    assert max(sizes) == 64 and len(sizes) > 1


def test_forward_reads_ascending_and_backward_descending(rows):
    recording = _Recording(rows)
    head = ChunkedHead(recording)
    logits = head(_hidden(3, HIDDEN, requires_grad=True))
    assert recording.reads == head.spans
    recording.reads.clear()
    logits.sum().backward()
    assert recording.reads == head.spans[::-1]


def test_logits_are_one_gemm_per_chunk(rows):
    head = ChunkedHead(rows)
    h = _hidden(1, 4, HIDDEN)
    logits = head(h)
    assert logits.shape == (1, 4, VOCAB) and logits.dtype == torch.float32
    wide = rows.to(torch.float32)
    expect = torch.cat([F.linear(h[0], wide[a:b]) for a, b in head.spans], dim=1)
    assert torch.equal(_bits(logits[0]), _bits(expect))


def test_vjp_matches_autograd_through_cat_in_bytes(rows):
    h, g = _hidden(4, HIDDEN), _hidden(4, VOCAB, seed=1)
    a, b = h.clone().requires_grad_(True), h.clone().requires_grad_(True)
    ChunkedHead(rows)(a).backward(g)
    PlainHead(rows)(b).backward(g)
    assert torch.equal(_bits(a.grad), _bits(b.grad))


def test_vjp_slices_are_contiguous(rows, monkeypatch):
    """Rule 6: no strided view of a matrix wider than 2^15 reaches mm in Poros code."""
    seen = []
    real = torch.mm

    def recording(g, w):
        seen.append((g.is_contiguous(), g.storage_offset()))
        return real(g, w)

    monkeypatch.setattr(torch, "mm", recording)
    ChunkedHead(rows)(_hidden(4, HIDDEN, requires_grad=True)).sum().backward()
    assert seen == [(True, 0)] * 3


def test_no_chunk_is_held_between_the_passes(rows):
    recording = _Recording(rows)
    logits = ChunkedHead(recording)(_hidden(2, HIDDEN, requires_grad=True))
    del recording.inner  # a saved chunk would let the backward run without these
    with pytest.raises(AttributeError):
        logits.sum().backward()


def test_logits_die_with_their_graph_without_a_gc_pass(rows):
    """No cycle through the Function's ctx: at 72B that would hold boundaries until a GC."""
    gc.disable()
    try:
        logits = ChunkedHead(rows)(_hidden(2, HIDDEN, requires_grad=True))
        ref = weakref.ref(logits)
        logits.sum().backward()
        del logits
        assert ref() is None
    finally:
        gc.enable()


def test_hidden_stays_fp32_at_the_head(tmp_path, tiny_pack):
    """Through the real trainer, including the cast at ``_logits_from_hidden``."""
    cfg, pack_path = tiny_pack
    seen = []

    def after_setup(trainer):
        trainer.model.lm_head.register_forward_pre_hook(lambda m, args: seen.append(args[0].dtype))

    tiny.run_arm(pack_path, cfg, "loop", tmp_path, dropout=0.05, steps=1, after_setup=after_setup)
    assert seen and set(seen) == {torch.float32}


def test_bf16_hidden_is_refused_by_the_gemm(rows):
    """If bf16 were ever stored under ``weight``, the trainer's cast would land here, loudly."""
    with pytest.raises(RuntimeError):
        ChunkedHead(rows)(_hidden(2, HIDDEN).to(torch.bfloat16))


def test_a_bf16_norm_promotes_at_the_multiply(tiny_pack):
    """Layer norms are stored bf16 and hidden stays fp32. Widening them first changes no bit."""
    cfg, pack_path = tiny_pack
    norm = tiny.build_model(pack_path, cfg).model.norm
    assert norm.weight.dtype == torch.bfloat16
    hidden = _hidden(1, 8, cfg.hidden_size)
    stored = norm(hidden)
    assert stored.dtype == torch.float32
    norm.weight = torch.nn.Parameter(norm.weight.data.to(torch.float32), requires_grad=False)
    assert torch.equal(_bits(stored), _bits(norm(hidden)))


def test_file_rows_read_the_bytes_of_their_slab(tiny_pack):
    cfg, pack_path = tiny_pack
    with SssFile.open(pack_path / "model.sss") as file:
        for block in (EMBED_BLOCK, HEAD_BLOCK):
            raw = torch.empty(file.entry((block, 0)).nbytes, dtype=torch.uint8)
            file.read_into((block, 0), raw)
            whole = raw.view(torch.bfloat16).reshape(-1, cfg.hidden_size)
            rows = FileRows(file, (block, 0))
            ids = torch.tensor([0, 7, 16383, 16384, 39999])
            assert rows.shape == tuple(whole.shape)
            assert torch.equal(_bits(rows[ids]), _bits(whole[ids]))
            assert torch.equal(_bits(rows[16384:32768]), _bits(whole[16384:32768]))


def test_file_rows_verify_their_slab_once(tmp_path, tiny_pack):
    _, pack_path = tiny_pack
    copy = tmp_path / "model.sss"
    copy.write_bytes((pack_path / "model.sss").read_bytes())
    (tmp_path / "model.sss.idx").write_bytes((pack_path / "model.sss.idx").read_bytes())
    with SssFile.open(copy) as file:
        fd = os.open(copy, os.O_RDWR)
        try:
            os.pwrite(fd, b"\xff\x00\xff", file.entry((HEAD_BLOCK, 0)).offset + 100)
            with pytest.raises(SssCorruptError, match="checksum"):
                FileRows(file, (HEAD_BLOCK, 0))[0:1]
            rows = FileRows(file, (EMBED_BLOCK, 0))
            rows[0:1]
            os.pwrite(fd, b"\xff\x00\xff", file.entry((EMBED_BLOCK, 0)).offset + 100)
            rows[0:1]  # checked already, on first touch
        finally:
            os.close(fd)
