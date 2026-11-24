"""Fused grouped expert compute for ``ManagedMoeBlock`` (``MoeConfig.expert_compute="fused"``).

One MoE layer call's routed experts as a handful of Triton kernels (``poros.moe.kernels``) over the
exact rows each expert receives -- no padding to the busiest expert, no per-expert launches:

* forward: ``a1 = x A_gu^T`` (rank-r, per row), then one kernel for the gate/up GEMM with its
  LoRA added in fp32 and SwiGLU in the epilogue (``h``, bf16), ``a2 = h A_d^T``, one kernel for the
  down GEMM with its LoRA and the routing weight (fp32 rows), and a fixed-order sum of each token's
  ``top_k`` rows;
* backward: one kernel recomputing g/u and taking the down projection's input grad and the
  SwiGLU backward (plus each row's routing-weight grad ``<ud, h>``, so the down output is never
  recomputed), one for the gate/up projection's input grad, the same fixed-order row sum, and the
  four adapter weight grads as per-(expert, segment) sums over rows.

The math is ``Qwen3MoeExperts.forward``'s with the adapter of ``poros.moe.adapters`` (``base +
s * B A x`` at each projection) in these precisions: bf16 frozen weights and activations into the
GEMMs, fp32 accumulation, the adapter products and the SwiGLU in fp32, ``h`` rounded to bf16 once
(the down GEMM's input), each token's weighted rows summed in fp32 and rounded once. Values
differ from the reference compute's (``poros.moe.modules._experts_compute``) by bf16 rounding,
and every Poros loop runs this same compute.

Bitwise structure (what block-major relies on): every kernel has one fixed launch configuration,
a GEMM output row depends on that row's inputs alone, and each adapter weight grad is summed per
(expert, segment) in tiles from the group's first row. So computing several micro-batches
("segments") in one call gives each segment exactly the values -- outputs, input grads,
routing-weight grads, and adapter grads -- it gets in a call of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from poros.moe.pinned import try_pin

__all__ = ["FusedExpertFunction", "Plan", "block_reason", "experts_backward", "experts_forward",
           "model_reason", "supported"]


def _silu(act) -> bool:
    """Whether ``act`` is SiLU, the activation the kernels' SwiGLU computes (``swiglu_limit``
    clamps it): poros.core.activation.is_silu, the dense fused compute's check too."""
    from poros.core.activation import is_silu

    return is_silu(act)


def block_reason(block) -> str | None:
    """None if the fused compute computes this block's experts, else why not (in words): what
    the block is, whatever the call. ``supported`` adds the call's device and dtype."""
    act = getattr(block, "act_fn", None)
    if not _silu(act):
        name = getattr(act, "__name__", None) or type(act).__name__
        return f"its experts' activation is {name}; the fused kernels compute SiLU"
    if getattr(block.store, "compute_dtype", None) != torch.bfloat16:
        return "the expert store does not decode to bf16"
    if getattr(block, "expert_group", None) is not None:
        return "the block brings its own group compute"
    adapter = block.adapter
    if adapter is not None:
        if adapter.lora_A_gate_up.dtype != torch.bfloat16:
            return "the expert adapter is not bf16"
        if torch.cuda.is_available():
            return fit_reason(_rank_pad(adapter.rank))
    return None


_FIT: dict = {}


def fit_reason(rp: int) -> str | None:
    """Why the expert kernels cannot run at padded rank ``rp`` on the current GPU, or None: each
    GEMM kernel launched once over one empty row tile, through ``kernels.launch`` (which cuts
    pipeline stages where that fits), at the rank's own tiles (``kernels.tiles``). Up to ``rp``
    128 they fit every card the fused experts run on (sm_80 and newer: ahead-of-time compiles in
    docs/validation/fused/rank_tiles/); above, a card with less shared memory than they take is
    found here. Cached per device."""
    dev = torch.cuda.current_device()
    key = (dev, rp)
    if key not in _FIT:
        from triton.runtime.errors import OutOfResources

        try:
            _probe(rp, torch.device("cuda", dev))
            torch.cuda.synchronize(dev)
            _FIT[key] = None
        except OutOfResources as exc:
            name = torch.cuda.get_device_name(dev)
            _FIT[key] = (f"LoRA rank {rp} (padded): the expert kernels need more shared memory per "
                         f"block than {name} has: {exc}")
    return _FIT[key]


def _probe(rp: int, device: torch.device) -> None:
    from poros.moe import kernels as K

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    H = NI = K.GEMM_BK
    E = rows = 1

    def z(*shape: int, dt: torch.dtype = bf) -> torch.Tensor:
        return torch.zeros(*shape, dtype=dt, device=device)

    tile = z(1, dt=i32)  # one tile: rows [0, 0) of expert 0
    t = (tile, tile, tile, tile, rows, H, NI, rp, 1.0)
    x, w = z(rows, H), z(E, 2 * NI, H)
    a, buf, wr, idx = z(rows, rp, dt=f32), z(rows, H, dt=f32), z(rows, dt=f32), z(rows, dt=i32)
    common = dict(RP=rp, BM=K.GEMM_BM, BK=K.GEMM_BK, LORA=True, enable_fp_fusion=True,
                  num_stages=K.GEMM_STAGES)
    bn, wbn, ww = K.tiles(rp)  # the rank's own tiles, as in training
    K.launch(K._gate_up_fwd, (1, K.cdiv(NI, bn)), x, idx, w, a, z(E, 2 * NI, rp),
             z(rows, NI), *t, 0.0, BN=bn, LIMIT=False, num_warps=K.GEMM_WARPS, **common)
    K.launch(K._down_fwd, (1, K.cdiv(H, wbn)), z(rows, NI), z(E, H, NI), a, z(E, H, rp),
             wr, idx, buf, *t, BN=wbn, num_warps=ww, **common)
    for drop in (False, True):
        keep = z(rows, NI, dt=torch.bool) if drop else x
        K.launch(K._mid_bwd, (1, K.cdiv(NI, bn)), x, x, idx, w, z(E, H, NI), a, a,
                 z(E, 2 * NI, rp), z(E, rp, NI), wr, z(rows, NI), z(rows, 2 * NI),
                 z(K.cdiv(NI, bn), rows, dt=f32), keep, 1.0, *t, 0.0, BN=bn,
                 LIMIT=False, DROP=drop, num_warps=K.GEMM_WARPS, **common)
        keep = z(rows, H, dt=torch.bool) if drop else x
        K.launch(K._dx_rows, (1, K.cdiv(H, wbn)), z(rows, 2 * NI), w, a, z(E, rp, H), idx,
                 buf, keep, 1.0, *t, BN=wbn, DROP=drop, num_warps=ww, **common)


def model_reason(model) -> str | None:
    """The first managed MoE layer the fused compute cannot run, as a reason; None if it runs
    them all."""
    from poros.moe.modules import ManagedMoeBlock

    for module in model.modules():
        if isinstance(module, ManagedMoeBlock):
            why = block_reason(module)
            if why is not None:
                return f"layer {module.layer_idx}: {why}"
    return None


def supported(block, hidden: torch.Tensor) -> str | None:
    """None if the fused compute runs this call, else why not (in words)."""
    if not hidden.is_cuda:
        return "not on CUDA"
    if hidden.dtype != torch.bfloat16:
        return f"activations are {hidden.dtype}, not bf16"
    return block_reason(block)


@dataclass
class Plan:
    """One call's routing as the kernels read it: rows sorted by expert, the row tiles of the GEMM
    kernels, and the (expert, segment) groups the adapter grads are summed over.

    ``segments`` lists the tokens of each segment (micro-batch), in order; None is one segment.
    """

    tokens: int
    top_k: int
    rows: int
    hit: list[int]  # experts with rows, ascending
    row_tok: torch.Tensor  # [rows] int32: sorted row -> token
    dest: torch.Tensor  # [rows] int32: sorted row -> its unsorted index token * top_k + slot
    t_r0: torch.Tensor  # GEMM row tiles (one expert each)
    t_r1: torch.Tensor
    t_w: torch.Tensor  # the tile's expert in the compact decoded weights (index into ``hit``)
    t_e: torch.Tensor  # the tile's expert id (adapter rows)
    g_r0: torch.Tensor  # (expert, segment) groups of the adapter grads
    g_r1: torch.Tensor
    g_e: torch.Tensor  # [groups] int64 on device: the group's expert id
    g_seg: torch.Tensor  # [groups] int64 on device: the group's segment
    segments: int
    row_seg: torch.Tensor | None  # [rows] the segment of each sorted row (None: one segment)
    w_tile: list[int]  # host: the first tile of each hit expert, then the tile count
    counts: torch.Tensor  # host [experts * segments] int64: rows per (expert, segment)

    @classmethod
    def build(cls, top_k_index: torch.Tensor, num_experts: int,
              segments: Sequence[int] | None = None, counts: torch.Tensor | None = None) -> Plan:
        """The plan of a call routed by ``top_k_index`` [tokens, top_k]. It waits on the GPU for
        one thing, the row counts per (expert, segment) that size the launches (on CUDA three
        host syncs in a row: ``bincount`` reads its input's min and max, then the counts come
        back); the rest is queued. ``counts``: those counts of exactly these routes and segments,
        as a plan of them found them (``Plan.counts``) -- then nothing waits on the GPU."""
        from poros.moe import kernels

        tokens, top_k = top_k_index.shape
        flat = top_k_index.reshape(-1)
        rows = flat.numel()
        dev = flat.device
        nseg = len(segments) if segments else 1
        if segments is not None and sum(segments) != tokens:
            raise ValueError(f"segments {list(segments)} do not add up to {tokens} tokens")
        if rows >= 2**31:  # the kernels index rows in int32
            raise ValueError(f"{rows} routed rows: the expert kernels take fewer than 2**31")
        if counts is not None and (counts.device.type != "cpu" or counts.dtype != torch.int64
                                   or counts.shape != (num_experts * nseg,)
                                   or int(counts.sum()) != rows):
            raise ValueError(f"counts are not {num_experts * nseg} host int64 row counts adding "
                             f"up to these routes' {rows} rows")
        with torch.no_grad():
            order = flat.argsort(stable=True)  # expert-major, token order within (segment order)
            row_seg = None
            if nseg > 1:
                seg_of_row = _segment_of_rows(segments, top_k, rows, dev)
                key = flat * nseg + seg_of_row
                row_seg = seg_of_row[order]
            else:
                key = flat
            if counts is None:
                counts = torch.bincount(key, minlength=num_experts * nseg).cpu()  # the host syncs
                if counts.numel() != num_experts * nseg:  # bincount runs to the largest key
                    raise ValueError(f"an expert id at or above num_experts={num_experts}")
        hit, w_tile, host, nt, ng = _plan_tables(counts, num_experts, nseg, kernels.GEMM_BM)
        if dev.type == "cuda":
            host = try_pin(host)  # pinned where the machine pins: the copy does not wait
        parts = host.to(dev, non_blocking=True).split([nt, nt, nt, nt, ng, ng, ng, ng])
        return cls(tokens=tokens, top_k=top_k, rows=rows, hit=hit,
                   row_tok=(order // top_k).to(torch.int32), dest=order.to(torch.int32),
                   t_r0=parts[0], t_r1=parts[1], t_w=parts[2], t_e=parts[3],
                   g_r0=parts[4], g_r1=parts[5], g_e=parts[6].long(), g_seg=parts[7].long(),
                   segments=nseg, row_seg=row_seg, w_tile=w_tile, counts=counts)

    @property
    def n_tiles(self) -> int:
        return self.t_r0.numel()


def _segment_of_rows(segments: Sequence[int], top_k: int, rows: int,
                     device: torch.device) -> torch.Tensor:
    """[rows] int64: the segment of each unsorted row (segment ``s``'s tokens own ``top_k`` rows
    each, back to back). Filled a segment's slice at a time from the host sizes, so building it
    waits on nothing (a repeat_interleave by a device tensor of the sizes reads its total back)."""
    seg = torch.empty(rows, dtype=torch.long, device=device)
    start = 0
    for s, n in enumerate(segments):
        seg[start:start + n * top_k] = s
        start += n * top_k
    return seg


def _plan_tables(per: torch.Tensor, num_experts: int, nseg: int,
                 bm: int) -> tuple[list[int], list[int], torch.Tensor, int, int]:
    """A plan's host tables from its row counts per (expert, segment) (``per``: host int64,
    expert-major): the hit experts (ascending), each hit expert's first GEMM tile then the tile
    count (``w_tile``), and one int32 tensor of the tiles' ``r0, r1, w, e`` then the groups'
    ``r0, r1, e, seg``, with the tile and group counts.

    Rows are sorted by expert, then segment (token order). Expert ``e`` holds sorted rows
    ``[start_e, start_e + n_e)``; its tiles cut them every ``bm`` rows from ``start_e`` (an expert
    with no rows has none), and its (expert, segment) groups are its segments' runs with rows, in
    segment order. Whole-array ops on the host."""
    g_end = per.cumsum(0)
    g_start = g_end - per  # each (expert, segment) run's first sorted row
    start = g_start.reshape(num_experts, nseg)[:, 0]  # each expert's first sorted row
    n = per.reshape(num_experts, nseg).sum(1)  # its rows
    tiles = (n + bm - 1) // bm  # its row tiles (none without rows)
    t_first = tiles.cumsum(0) - tiles  # its first tile
    nt = int(tiles.sum())
    t_e = torch.arange(num_experts).repeat_interleave(tiles)  # each tile's expert
    t_r0 = start[t_e] + (torch.arange(nt) - t_first[t_e]) * bm
    t_r1 = torch.minimum(t_r0 + bm, (start + n)[t_e])
    hit_mask = n > 0
    t_w = (hit_mask.cumsum(0) - 1)[t_e]  # the tile's expert among the hit ones
    g = per.nonzero().squeeze(1)  # the runs with rows, expert-major
    host = torch.cat([t_r0, t_r1, t_w, t_e, g_start[g], g_end[g], g // nseg, g % nseg])
    hit = hit_mask.nonzero().squeeze(1)
    return hit.tolist(), [*t_first[hit].tolist(), nt], host.to(torch.int32), nt, g.numel()


def _rank_pad(r: int) -> int:
    from poros.core.fused_tiles import pad

    return pad(r)


# hit experts decoded at a time: the GEMMs' bf16 weights for 32 experts of Qwen3-30B-A3B are 300 MB
EXPERT_CHUNK = 32


@dataclass
class _Tiles:
    """A range of a plan's GEMM tiles (whole experts), their weight index into the range's
    decoded experts."""

    r0: torch.Tensor
    r1: torch.Tensor
    w: torch.Tensor
    e: torch.Tensor

    @property
    def n(self) -> int:
        return self.r0.numel()


def _ensure(block, plan: Plan) -> None:
    """The hit experts resident. On a streaming source's first pass (the SSD store before it has
    a schedule: ``recording``) the source hears them first (``poros.moe.modules._prefetch_hint``),
    so their reads go out together instead of one by one inside ``ensure``. Once it replays a
    schedule it is not told: the replay already has the reads in flight, and a hint's speculative
    reads (the next phase, the look-ahead experts) cost a disk-bound pass more than they save
    (Qwen3-30B-A3B, RTX 4090, path fast: 6.9 GB an update against 6.6, 3.39 s against 3.16). A
    hint moves no bytes into the residency: no value changes."""
    keys = [(block.layer_idx, e) for e in plan.hit]
    if getattr(getattr(block.store, "source", None), "recording", False):
        from poros.moe.modules import _prefetch_hint

        _prefetch_hint(block, keys)
    block.residency.ensure(keys)


def _prefetch(block, step: int) -> None:
    """The next layers' experts (``step``: +1 forward, -1 backward) toward the device. Called once
    the layer's kernels are queued: the plan cost a host sync, so bookkeeping done before them
    would leave the GPU idle; after them it runs while they do."""
    from poros.moe.modules import prefetch_neighbor

    prefetch_neighbor(block, step)


def _all_tiles(plan: Plan) -> _Tiles:
    return _Tiles(plan.t_r0, plan.t_r1, plan.t_w, plan.t_e)


def _chunks(block, plan: Plan):
    """(tiles, decoded gate_up [n, 2I, H] loader, decoded down [n, H, I] loader) per chunk of
    ``EXPERT_CHUNK`` hit experts: one chunk's weights are alive at a time. A tile's values do
    not depend on the other tiles of its launch, so the chunking changes no value."""
    from poros.moe.store import decode_experts

    keys = [(block.layer_idx, e) for e in plan.hit]
    for c0 in range(0, len(keys), EXPERT_CHUNK):
        c1 = min(c0 + EXPERT_CHUNK, len(keys))
        t0, t1 = plan.w_tile[c0], plan.w_tile[c1]
        tiles = _Tiles(plan.t_r0[t0:t1], plan.t_r1[t0:t1], plan.t_w[t0:t1] - c0,
                       plan.t_e[t0:t1])
        chunk = keys[c0:c1]
        yield (tiles, lambda chunk=chunk: decode_experts(block, chunk, "gate_up_proj"),
               lambda chunk=chunk: decode_experts(block, chunk, "down_proj"))


def _shapes(block) -> tuple[int, int]:
    """(2I, H) of the experts' gate_up."""
    two_i, h = block.store.tensor_shapes["gate_up_proj"]
    return int(two_i), int(h)


def dropout_p(block) -> float:
    """The expert adapter's LoRA dropout probability while it drops (0.0 otherwise)."""
    ad = block.adapter
    drop = getattr(ad, "dropout", None) if ad is not None else None
    p = float(getattr(drop, "p", 0.0) or 0.0)
    return p if p and drop.training else 0.0


class Masks:
    """A call's LoRA dropout masks, in sorted-row order: ``x`` [rows, H] for the gate/up adapter's
    input, ``h`` [rows, I] for the down adapter's. A segment draws its own, ``x`` then ``h``, from
    the random state it starts in -- the live generator for a call of one segment, else the state
    each segment's own call would start in (``states``, restored afterwards) -- so a segment's
    masks are the ones its own call draws. ``after`` holds each segment's state once drawn."""

    def __init__(self, plan: Plan, p: float, h: int, i: int, device: torch.device,
                 states: Sequence | None = None) -> None:
        from poros.moe.modules import _rng_state, _set_rng_state

        self.p = p
        self.scale = 1.0 / (1.0 - p)
        self.after: list = []
        if states is None:
            if plan.segments != 1:
                raise ValueError("a call of several segments draws from their states")
            self.x = torch.empty(plan.rows, h, dtype=torch.bool, device=device).bernoulli_(1 - p)
            self.h = torch.empty(plan.rows, i, dtype=torch.bool, device=device).bernoulli_(1 - p)
            self.after.append(_rng_state(device))
            return
        self.x = torch.empty(plan.rows, h, dtype=torch.bool, device=device)
        self.h = torch.empty(plan.rows, i, dtype=torch.bool, device=device)
        cuda = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda):
            for g, state in enumerate(states):
                _set_rng_state(device, state)
                rows = ((plan.row_seg == g).nonzero().squeeze(1) if plan.row_seg is not None
                        else slice(None))
                n = plan.rows if plan.row_seg is None else rows.numel()
                self.x[rows] = torch.empty(n, h, dtype=torch.bool, device=device).bernoulli_(1 - p)
                self.h[rows] = torch.empty(n, i, dtype=torch.bool, device=device).bernoulli_(1 - p)
                self.after.append(_rng_state(device))


def _masks(block, plan: Plan, h: int, i: int, device, states) -> Masks | None:
    p = dropout_p(block)
    return Masks(plan, p, h, i, device, states) if p else None


def _lora_rows(x, plan: Plan, m, k: int, r: int, strides, rp: int, gather: bool,
               keep: torch.Tensor | None = None, pscale: float = 1.0,
               tiles: _Tiles | None = None, out: torch.Tensor | None = None):
    """[rows, rp] fp32: each row times one adapter factor (``_lora_rows``), over ``tiles`` (all
    of the plan's by default) into ``out`` (a new tensor by default)."""
    from poros.moe import kernels as K

    tiles = tiles or _all_tiles(plan)
    if out is None:
        out = torch.empty(plan.rows, rp, dtype=torch.float32, device=x.device)
    K.launch(K._lora_rows, (tiles.n,),
        x, plan.row_tok, m, out, keep if keep is not None else x, pscale, tiles.r0, tiles.r1,
        tiles.e, plan.rows, k, r, *strides, RP=rp, BM=K.GEMM_BM, BK=K.GEMM_BK, GATHER=gather,
        DROP=keep is not None, enable_fp_fusion=True, num_warps=4)
    return out


def _combine(buf: torch.Tensor, plan: Plan, h: int, dtype: torch.dtype) -> torch.Tensor:
    from poros.moe import kernels as K

    out = torch.empty(plan.tokens, h, dtype=dtype, device=buf.device)
    grid = (K.cdiv(plan.tokens, K.COMB_BT), K.cdiv(h, K.COMB_BH))
    K.launch(K._combine, grid,
        buf, out, plan.tokens, h, KTOP=plan.top_k, BT=K.COMB_BT, BH=K.COMB_BH,
        enable_fp_fusion=True, num_warps=4)
    return out


def _limit(block) -> tuple[float, bool]:
    lim = getattr(block, "swiglu_limit", None)
    return (float(lim), True) if lim is not None else (0.0, False)


def experts_forward(hidden: torch.Tensor, top_k_weights: torch.Tensor, plan: Plan,
                    block, states: Sequence | None = None) -> torch.Tensor:
    """The routed experts' output [tokens, H] (``hidden``'s dtype) for one plan; ``states``: each
    segment's starting random state (``Masks``) where the adapter drops."""
    return _forward(hidden, top_k_weights, plan, block, states)[0]


def _forward(hidden, top_k_weights, plan, block, states):
    from poros.moe import kernels as K

    _ensure(block, plan)
    x = hidden.contiguous()
    two_i, h = _shapes(block)
    i = two_i // 2
    ad = block.adapter
    lora = ad is not None
    r = ad.rank if lora else 16
    rp = _rank_pad(r)
    s = float(ad.scaling) if lora else 0.0
    limit, has_limit = _limit(block)
    order = plan.dest.long()
    w_sorted = top_k_weights.reshape(-1)[order].float()
    dummy = x
    masks = _masks(block, plan, h, i, x.device, states) if lora else None
    kx, kh, ps = (masks.x, masks.h, masks.scale) if masks else (None, None, 1.0)
    a1 = (_lora_rows(x, plan, ad.lora_A_gate_up, h, r, (r * h, h, 1), rp, True, kx, ps)
          if lora else dummy)
    hb = torch.empty(plan.rows, i, dtype=x.dtype, device=x.device)
    a2 = torch.empty(plan.rows, rp, dtype=torch.float32, device=x.device) if lora else dummy
    buf = torch.empty(plan.rows, h, dtype=torch.float32, device=x.device)
    bn, wbn, ww = K.tiles(rp)
    for tiles, gate_up, down in _chunks(block, plan):
        wgu = gate_up()
        K.launch(K._gate_up_fwd, (tiles.n, K.cdiv(i, bn)),
            x, plan.row_tok, wgu, a1, ad.lora_B_gate_up if lora else dummy, hb, tiles.r0, tiles.r1,
            tiles.w, tiles.e, plan.rows, h, i, r, s, limit, RP=rp, BM=K.GEMM_BM, BN=bn,
            BK=K.GEMM_BK, LIMIT=has_limit, LORA=lora, enable_fp_fusion=True, num_warps=K.GEMM_WARPS,
            num_stages=K.GEMM_STAGES)
        del wgu
        if lora:
            _lora_rows(hb, plan, ad.lora_A_down, i, r, (r * i, i, 1), rp, False, kh, ps,
                       tiles=tiles, out=a2)
        wd = down()
        K.launch(K._down_fwd, (tiles.n, K.cdiv(h, wbn)),
            hb, wd, a2, ad.lora_B_down if lora else dummy, w_sorted, plan.dest, buf, tiles.r0,
            tiles.r1, tiles.w, tiles.e, plan.rows, h, i, r, s, RP=rp, BM=K.GEMM_BM, BN=wbn,
            BK=K.GEMM_BK, LORA=lora, enable_fp_fusion=True, num_warps=ww,
            num_stages=K.GEMM_STAGES)
        del wd
    out = _combine(buf, plan, h, hidden.dtype)
    _prefetch(block, +1)
    return out, masks


def _wgrad(out: torch.Tensor, x, plan: Plan, w_sorted, y, n: int, r: int, s: float,
           transposed: bool, gather: bool, scale: bool, keep=None, pscale: float = 1.0) -> None:
    """One adapter weight grad of every (expert, segment) group into ``out`` [segments, experts,
    ...] (zeros elsewhere): ``[n, r]`` per expert, or ``[r, n]`` (``transposed``, an A factor)."""
    from poros.moe import kernels as K

    groups = plan.g_r0.numel()
    if not groups:
        return
    seg_stride, e_stride = out.stride(0), out.stride(1)
    o_sn, o_sj = (1, n) if transposed else (r, 1)
    g_off = plan.g_seg * seg_stride + plan.g_e * e_stride
    K.launch(K._wgrad, (groups, K.cdiv(n, K.WG_BN)),
        x, plan.row_tok, w_sorted, y, out, keep if keep is not None else x, pscale, plan.g_r0,
        plan.g_r1, g_off, n, r, s, o_sn, o_sj, RP=y.shape[1], BM=K.WG_BM, BN=K.WG_BN, GATHER=gather,
        SCALE=scale, DROP=keep is not None, enable_fp_fusion=True, num_warps=4)


def experts_backward(hidden: torch.Tensor, top_k_weights: torch.Tensor, grad_out: torch.Tensor,
                     plan: Plan, block, states: Sequence | None = None):
    """Input grad [tokens, H], routing-weight grad [tokens, top_k], and per segment the adapter
    grads ``[(dA_gate_up, dB_gate_up, dA_down, dB_down)]`` (each a full ``[experts, ...]`` tensor
    in the adapter's dtype; empty list without an adapter)."""
    from poros.moe import kernels as K

    _ensure(block, plan)
    x = hidden.contiguous()
    gy = grad_out.contiguous().to(x.dtype)
    two_i, h = _shapes(block)
    i = two_i // 2
    ad = block.adapter
    lora = ad is not None
    r = ad.rank if lora else 16
    rp = _rank_pad(r)
    s = float(ad.scaling) if lora else 0.0
    limit, has_limit = _limit(block)
    order = plan.dest.long()
    w_sorted = top_k_weights.reshape(-1)[order].float()
    dummy = x
    masks = _masks(block, plan, h, i, x.device, states) if lora else None
    kx, kh, ps = (masks.x, masks.h, masks.scale) if masks else (None, None, 1.0)
    if lora:
        a1 = _lora_rows(x, plan, ad.lora_A_gate_up, h, r, (r * h, h, 1), rp, True, kx, ps)
        c2 = _lora_rows(gy, plan, ad.lora_B_down, h, r, (h * r, 1, r), rp, True)
    else:
        a1 = c2 = dummy
    bn, wbn, ww = K.tiles(rp)
    n_parts = K.cdiv(i, bn)
    hb = torch.empty(plan.rows, i, dtype=x.dtype, device=x.device)
    dgu = torch.empty(plan.rows, two_i, dtype=x.dtype, device=x.device)
    dw_part = torch.empty(n_parts, plan.rows, dtype=torch.float32, device=x.device)
    c1 = torch.empty(plan.rows, rp, dtype=torch.float32, device=x.device) if lora else dummy
    buf = torch.empty(plan.rows, h, dtype=torch.float32, device=x.device)
    for tiles, gate_up, down in _chunks(block, plan):
        wgu, wd = gate_up(), down()
        K.launch(K._mid_bwd, (tiles.n, n_parts),
            x, gy, plan.row_tok, wgu, wd, a1, c2, ad.lora_B_gate_up if lora else dummy,
            ad.lora_A_down if lora else dummy, w_sorted, hb, dgu, dw_part,
            kh if kh is not None else dummy, ps, tiles.r0, tiles.r1, tiles.w, tiles.e, plan.rows, h,
            i, r, s, limit, RP=rp, BM=K.GEMM_BM, BN=bn, BK=K.GEMM_BK, LIMIT=has_limit,
            LORA=lora, DROP=kh is not None, enable_fp_fusion=True, num_warps=K.GEMM_WARPS,
            num_stages=K.GEMM_STAGES)
        del wd
        if lora:
            _lora_rows(dgu, plan, ad.lora_B_gate_up, two_i, r, (two_i * r, 1, r), rp, False,
                       tiles=tiles, out=c1)
        K.launch(K._dx_rows, (tiles.n, K.cdiv(h, wbn)),
            dgu, wgu, c1, ad.lora_A_gate_up if lora else dummy, plan.dest, buf,
            kx if kx is not None else dummy, ps, tiles.r0, tiles.r1, tiles.w, tiles.e, plan.rows, h,
            i, r, s, RP=rp, BM=K.GEMM_BM, BN=wbn, BK=K.GEMM_BK, LORA=lora,
            DROP=kx is not None, enable_fp_fusion=True, num_warps=ww,
            num_stages=K.GEMM_STAGES)
        del wgu
    dw_sorted = torch.empty(plan.rows, dtype=torch.float32, device=x.device)
    K.launch(K._sum_parts, (K.cdiv(plan.rows, 1024),),
        dw_part, dw_sorted, plan.rows, P=n_parts, BLOCK=1024, enable_fp_fusion=True, num_warps=4)
    dx = _combine(buf, plan, h, hidden.dtype)
    dw = torch.empty(plan.rows, dtype=torch.float32, device=x.device)
    dw[order] = dw_sorted
    dw = dw.view(plan.tokens, plan.top_k).to(top_k_weights.dtype)
    grads: list[tuple[torch.Tensor, ...]] = []
    if lora:
        a2 = _lora_rows(hb, plan, ad.lora_A_down, i, r, (r * i, i, 1), rp, False, kh, ps)
        nseg = plan.segments
        params = (ad.lora_A_gate_up, ad.lora_B_gate_up, ad.lora_A_down, ad.lora_B_down)
        g_agu, g_bgu, g_ad, g_bd = (p.new_zeros((nseg, *p.shape)) for p in params)
        _wgrad(g_agu, x, plan, w_sorted, c1, h, r, s, True, True, False, kx, ps)
        _wgrad(g_bgu, dgu, plan, w_sorted, a1, two_i, r, s, False, False, False)
        _wgrad(g_ad, hb, plan, w_sorted, c2, i, r, s, True, False, True, kh, ps)
        _wgrad(g_bd, gy, plan, w_sorted, a2, h, r, s, False, True, True)
        grads = [(g_agu[k], g_bgu[k], g_ad[k], g_bd[k]) for k in range(nseg)]
    _prefetch(block, -1)
    return dx, dw, grads


class FusedExpertFunction(torch.autograd.Function):
    """Autograd boundary of the fused compute: saves the inputs only (no weights, no activations);
    backward rebuilds the plan, recomputes what it needs and accumulates the adapter grads into
    ``param.grad`` (as ``_ManagedExpertFunction`` does)."""

    @staticmethod
    def forward(ctx, hidden, top_k_index, top_k_weights, block):
        from poros.moe.modules import _rng_state

        ctx.block = block
        ctx.save_for_backward(hidden, top_k_index, top_k_weights)
        ctx.rng = _rng_state(hidden.device) if dropout_p(block) else None  # where masks start
        precomputed = block._remat_hit
        ctx.reused = precomputed is not None  # inputs verified equal to the cached call's
        if precomputed is not None:  # verified by ManagedMoeBlock: this call's exact output
            block._remat_hit = None
            return precomputed.clone()
        if hidden.shape[0] == 0:
            return hidden.clone()
        plan = Plan.build(top_k_index, block.num_experts)
        return experts_forward(hidden, top_k_weights, plan, block)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):  # type: ignore[override]
        hidden, top_k_index, top_k_weights = ctx.saved_tensors
        block = ctx.block
        if hidden.shape[0] == 0:
            return torch.zeros_like(hidden), None, torch.zeros_like(top_k_weights), None
        pending = block._pending.popleft() if block._pending else None
        if pending is not None and ctx.reused and pending[1].shape == grad_out.shape:
            # poros.moe.batching computed this micro-batch's backward with the visit's others,
            # for this grad: checked here, or on the device once the update's passes end
            checks = block._checks
            if checks is not None:
                checks.append((grad_out == pending[1]).all())
            same = checks is not None or torch.equal(grad_out, pending[1])
            if same:
                _, _, dx, dw, seg_grads = pending
                if seg_grads:
                    accumulate_adapter_grads(block.adapter, seg_grads)
                return dx, None, dw, None
        plan = Plan.build(top_k_index, block.num_experts)
        states = [ctx.rng] if ctx.rng is not None else None  # the forward's masks, drawn again
        dx, dw, grads = experts_backward(hidden, top_k_weights, grad_out, plan, block, states)
        if grads:
            accumulate_adapter_grads(block.adapter, grads[0])
        return dx, None, dw, None


def accumulate_adapter_grads(adapter, grads: Sequence[torch.Tensor]) -> None:
    """``param.grad`` += one segment's grads, the order ``_ManagedExpertFunction`` adds them in.

    A segment's grads are views into its call's ``(segments, *shape)`` buffers
    (``experts_backward``). The first is stored as a copy when the buffer holds more than it: as
    ``param.grad`` a view keeps every segment's grads alive until the grads are cleared (9.4 GiB
    of Qwen3-30B-A3B's 1x8 peak, docs/validation/int/4090/round5_moegrad/). A copy, not zeros
    plus the grad: ``0 + -0.0`` is ``+0.0``."""
    params = (adapter.lora_A_gate_up, adapter.lora_B_gate_up, adapter.lora_A_down,
              adapter.lora_B_down)
    for param, grad in zip(params, grads, strict=True):
        if param.grad is None:
            whole = grad.untyped_storage().nbytes() == grad.numel() * grad.element_size()
            param.grad = grad if whole else grad.clone()
        else:
            param.grad.add_(grad)
