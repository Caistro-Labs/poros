"""``poros.moe.fused.Plan.build`` makes the plan the per-expert loop it replaced made, bit for bit.

The reference below is the original per-expert loop, verbatim but for taking
the tile height as an argument. Every field of the plan -- the host lists and each device tensor's
dtype, shape and values -- is compared on randomized routings and the edge cases: experts with no
rows, one token, every token to one expert, empty segments, every segment count 1..8, no tokens.
On the CPU the kernels module (Triton) is stood in for by its tile height alone; with CUDA the
plans are also built on the device. A plan built on known row counts (``counts=``, the batched
backward's, from its forward visit) is the same plan, and on the device it waits on nothing; the
counts are handed over (``poros.moe.batching._take_counts``) only for the very index tensors they
were counted from.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from poros.moe import batching  # noqa: E402
from poros.moe.fused import Plan, _plan_tables  # noqa: E402
from poros.moe.pinned import try_pin  # noqa: E402

GEMM_BM = 64  # poros.moe.kernels.GEMM_BM (the stand-in's; the real module's where Triton is)


@pytest.fixture
def bm(monkeypatch) -> int:
    """The tile height Plan.build reads: the real kernels module's, else a stand-in's."""
    try:
        from poros.moe import kernels
    except ImportError:  # no Triton: a stand-in with the tile height alone
        import poros.moe

        kernels = types.ModuleType("poros.moe.kernels")
        kernels.GEMM_BM = GEMM_BM
        monkeypatch.setitem(sys.modules, "poros.moe.kernels", kernels)
        monkeypatch.setattr(poros.moe, "kernels", kernels, raising=False)
    return kernels.GEMM_BM


def _reference(top_k_index, num_experts, segments, bm):
    """Plan.build's original per-expert loop, its fields as a dict."""
    tokens, top_k = top_k_index.shape
    flat = top_k_index.reshape(-1)
    rows = flat.numel()
    dev = flat.device
    nseg = len(segments) if segments else 1
    if segments is not None and sum(segments) != tokens:
        raise ValueError(f"segments {list(segments)} do not add up to {tokens} tokens")
    with torch.no_grad():
        order = flat.argsort(stable=True)  # expert-major, token order within (segment order)
        row_seg = None
        if nseg > 1:
            seg_of_row = torch.repeat_interleave(
                torch.arange(nseg, device=dev),
                torch.tensor(list(segments), device=dev) * top_k)
            key = flat * nseg + seg_of_row
            row_seg = seg_of_row[order]
        else:
            key = flat
        counts = torch.bincount(key, minlength=num_experts * nseg).tolist()  # one host sync
    t_r0, t_r1, t_w, t_e, g_r0, g_r1, g_e, g_seg, hit = [], [], [], [], [], [], [], [], []
    w_tile: list[int] = []
    start = 0
    for e in range(num_experts):
        per_seg = counts[e * nseg:(e + 1) * nseg]
        n = sum(per_seg)
        if not n:
            continue
        w_idx = len(hit)
        hit.append(e)
        w_tile.append(len(t_r0))
        for r0 in range(start, start + n, bm):
            t_r0.append(r0)
            t_r1.append(min(r0 + bm, start + n))
            t_w.append(w_idx)
            t_e.append(e)
        g0 = start
        for s, c in enumerate(per_seg):
            if c:
                g_r0.append(g0)
                g_r1.append(g0 + c)
                g_e.append(e)
                g_seg.append(s)
            g0 += c
        start += n
    host = torch.tensor(t_r0 + t_r1 + t_w + t_e + g_r0 + g_r1 + g_e + g_seg,
                        dtype=torch.int32)
    if dev.type == "cuda":
        host = host.pin_memory()
    nt, ng = len(t_r0), len(g_r0)
    parts = host.to(dev, non_blocking=True).split([nt, nt, nt, nt, ng, ng, ng, ng])
    return dict(tokens=tokens, top_k=top_k, rows=rows, hit=hit,
                counts=torch.tensor(counts, dtype=torch.int64),
                row_tok=(order // top_k).to(torch.int32), dest=order.to(torch.int32),
                t_r0=parts[0], t_r1=parts[1], t_w=parts[2], t_e=parts[3],
                g_r0=parts[4], g_r1=parts[5], g_e=parts[6].long(), g_seg=parts[7].long(),
                segments=nseg, row_seg=row_seg, w_tile=[*w_tile, len(t_r0)])


def _same(plan: Plan, ref: dict) -> None:
    fields = set(Plan.__dataclass_fields__)
    assert fields == set(ref)
    for name in sorted(fields):
        got, want = getattr(plan, name), ref[name]
        if isinstance(want, torch.Tensor):
            assert isinstance(got, torch.Tensor), name
            assert got.dtype == want.dtype and got.shape == want.shape, name
            assert got.device == want.device, name
            assert torch.equal(got.cpu(), want.cpu()), name
        else:
            assert type(got) is type(want) and got == want, name


def _routing(gen, tokens, top_k, num_experts, used=None):
    """[tokens, top_k] distinct experts per token (as a router's top-k), drawn from the first
    ``used`` experts (all by default)."""
    used = used or num_experts
    scores = torch.rand(tokens, used, generator=gen)
    return scores.topk(top_k, dim=-1).indices


def _split(gen, tokens, nseg):
    """``nseg`` segment sizes adding up to ``tokens`` (some may be empty)."""
    cuts = sorted(torch.randint(0, tokens + 1, (nseg - 1,), generator=gen).tolist())
    return [b - a for a, b in zip([0, *cuts], [*cuts, tokens])]


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_randomized_routings(bm, device):
    gen = torch.Generator().manual_seed(0)
    for case in range(120):
        num_experts = [1, 3, 8, 16, 128][case % 5]
        top_k = min([1, 2, 4, 8][case % 4], num_experts)
        tokens = int(torch.randint(1, 700, (1,), generator=gen))
        used = int(torch.randint(top_k, num_experts + 1, (1,), generator=gen))
        idx = _routing(gen, tokens, top_k, num_experts, used).to(device)
        nseg = 1 + case % 8
        segments = None if case % 9 == 0 else _split(gen, tokens, nseg)
        _same(Plan.build(idx, num_experts, segments=segments),
              _reference(idx, num_experts, segments, bm))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nseg", range(1, 9))
def test_every_segment_count(bm, device, nseg):
    gen = torch.Generator().manual_seed(nseg)
    tokens = 512 * nseg  # Qwen3-30B-A3B at 512 tokens a micro-batch: 128 experts, top-8
    idx = _routing(gen, tokens, 8, 128).to(device)
    segments = [512] * nseg
    _same(Plan.build(idx, 128, segments=segments), _reference(idx, 128, segments, bm))
    uneven = _split(gen, tokens, nseg)
    _same(Plan.build(idx, 128, segments=uneven), _reference(idx, 128, uneven, bm))


@pytest.mark.parametrize("device", DEVICES)
def test_edge_cases(bm, device):
    gen = torch.Generator().manual_seed(7)
    cases = [
        (_routing(gen, 1, 8, 128), 128, None),  # one token
        (_routing(gen, 1, 8, 128), 128, [1]),
        (_routing(gen, 1, 2, 4), 4, [0, 1, 0]),  # one token, empty segments around it
        (torch.full((300, 1), 5), 16, None),  # every token to one expert
        (torch.full((300, 1), 5), 16, [100, 0, 200]),
        (torch.full((bm, 1), 0), 4, [bm]),  # exactly one tile
        (torch.full((bm + 1, 1), 3), 4, [1, bm]),  # one row past a tile
        (_routing(gen, 50, 2, 128, used=3), 128, [25, 25]),  # 125 experts with no rows
        (torch.empty(0, 8, dtype=torch.long), 128, None),  # no tokens
        (torch.empty(0, 8, dtype=torch.long), 128, [0, 0]),
        (_routing(gen, 9, 4, 8), 8, [0] * 7 + [9]),  # eight segments, seven empty
    ]
    for idx, num_experts, segments in cases:
        idx = idx.to(device)
        _same(Plan.build(idx, num_experts, segments=segments),
              _reference(idx, num_experts, segments, bm))


def test_refusals(bm):
    idx = torch.tensor([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="do not add up"):
        Plan.build(idx, 4, segments=[1, 2])
    with pytest.raises(ValueError, match="at or above num_experts"):
        Plan.build(idx, 3)  # expert 3 of 3: the loop dropped its rows without a word
    with pytest.raises(ValueError, match="at or above num_experts"):
        Plan.build(idx, 2, segments=[1, 1])


def test_tables_match_the_loop_at_any_tile_height():
    """The host tables alone, at other tile heights than the kernels' (the tiling arithmetic)."""
    gen = torch.Generator().manual_seed(3)
    for bm in (1, 2, 7, 16, 64, 256):
        for nseg in (1, 2, 5, 8):
            idx = _routing(gen, 333, 4, 16, used=11)
            segments = _split(gen, 333, nseg) if nseg > 1 else None
            ref = _reference(idx, 16, segments, bm)
            hit, w_tile, host, nt, ng = _plan_tables(ref["counts"], 16, nseg, bm)
            assert hit == ref["hit"] and w_tile == ref["w_tile"]
            want = torch.cat([ref[k].to(torch.int32) for k in
                              ("t_r0", "t_r1", "t_w", "t_e", "g_r0", "g_r1", "g_e", "g_seg")])
            assert host.dtype == torch.int32 and torch.equal(host, want)
            assert (nt, ng) == (ref["t_r0"].numel(), ref["g_r0"].numel())


def _refuse_pinning(monkeypatch) -> list:
    """Every ``Tensor.pin_memory`` refused, as a machine at its pin ceiling refuses; the calls."""
    calls = []

    def refuse(self, *args, **kwargs):
        calls.append(self.shape)
        raise RuntimeError("cannot pin: the page-locked memory limit is reached")

    monkeypatch.setattr(torch.Tensor, "pin_memory", refuse)
    return calls


def test_try_pin_hands_back_the_host_tensor_where_pinning_is_refused(monkeypatch):
    t = torch.arange(6)
    pinned = object()
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: pinned)
    assert try_pin(t) is pinned  # where the machine pins: the page-locked copy
    calls = _refuse_pinning(monkeypatch)
    assert try_pin(t) is t and len(calls) == 1  # refused: the tensor itself, no error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a plan pins its tables for CUDA")
def test_plan_is_the_same_where_pinning_is_refused(bm, monkeypatch):
    """On a machine that refuses to page-lock, the plan's tables cross from pageable memory: the
    same plan (it raised before, at every MoE call)."""
    gen = torch.Generator().manual_seed(13)
    idx = _routing(gen, 300, 8, 128).cuda()
    segments = [120, 0, 180]
    ref = _reference(idx, 128, segments, bm)  # the loop pins as it did: built before the refusal
    counts = Plan.build(idx, 128, segments=segments).counts
    calls = _refuse_pinning(monkeypatch)
    _same(Plan.build(idx, 128, segments=segments), ref)
    _same(Plan.build(idx, 128, segments=segments, counts=counts), ref)
    assert len(calls) == 2  # each build asked, and went on unpinned


@pytest.mark.parametrize("device", DEVICES)
def test_known_counts_give_the_same_plan_without_a_host_sync(bm, device):
    """``Plan.build(..., counts=plan.counts)``: the same plan, field for field; on the device it
    queues everything (torch's sync debug mode raises on a host sync)."""
    gen = torch.Generator().manual_seed(11)
    for case in range(24):
        nseg = 1 + case % 8
        tokens = int(torch.randint(1, 600, (1,), generator=gen))
        idx = _routing(gen, tokens, 8, 128, used=int(torch.randint(8, 129, (1,), generator=gen)))
        idx = idx.to(device)
        segments = None if nseg == 1 else _split(gen, tokens, nseg)
        counts = Plan.build(idx, 128, segments=segments).counts
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
        try:
            again = Plan.build(idx, 128, segments=segments, counts=counts)
        finally:
            if device == "cuda":
                torch.cuda.set_sync_debug_mode("default")
        _same(again, _reference(idx, 128, segments, bm))


def test_known_counts_may_be_any_layout(bm):
    """Counts as a strided view (every other element of a larger tensor) plan as their copy."""
    gen = torch.Generator().manual_seed(5)
    idx = _routing(gen, 200, 4, 16, used=9)
    counts = Plan.build(idx, 16, segments=[120, 80]).counts
    strided = torch.stack([counts, torch.zeros_like(counts)], 1).reshape(-1)[::2]
    assert not strided.is_contiguous() and torch.equal(strided, counts)
    _same(Plan.build(idx, 16, segments=[120, 80], counts=strided),
          _reference(idx, 16, [120, 80], bm))


def test_known_counts_must_count_these_rows(bm):
    idx = torch.tensor([[0, 1], [2, 3], [1, 2]])
    counts = Plan.build(idx, 4, segments=[1, 2]).counts
    for bad in (counts[:-1], torch.cat([counts, counts[:1]]), counts.to(torch.int32),
                counts + 1, (counts - 1).clamp(min=0)):
        with pytest.raises(ValueError, match="host int64 row counts"):
            Plan.build(idx, 4, segments=[1, 2], counts=bad)
    with pytest.raises(ValueError, match="host int64 row counts"):
        Plan.build(idx, 4, segments=[1, 1, 1], counts=counts)  # three segments, not two
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="host int64 row counts"):
            Plan.build(idx.cuda(), 4, segments=[1, 2], counts=counts.cuda())


class _Block:
    _fwd_counts = None


class _Entry:
    def __init__(self, index):
        self.index = index


class _Plan:
    def __init__(self):
        self.counts = torch.tensor([1, 2, 3])


def test_backward_takes_the_forward_counts_of_its_own_routes():
    a, b = torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2, dtype=torch.long)
    entries = [_Entry(a), _Entry(b)]
    plan = _Plan()
    block = _Block()
    batching._keep_counts(block, entries, plan)
    # the backward's entries are new objects (dataclasses.replace) holding the same tensors
    assert batching._take_counts(block, [_Entry(a), _Entry(b)]) is plan.counts
    assert block._fwd_counts is None
    assert batching._take_counts(block, entries) is None  # taken once
    for other in ([_Entry(a)], [_Entry(b), _Entry(a)], [_Entry(a), _Entry(b.clone())],
                  [_Entry(a), _Entry(b), _Entry(a)]):
        batching._keep_counts(block, entries, plan)
        assert batching._take_counts(block, other) is None  # equal values are not enough
        assert block._fwd_counts is None
