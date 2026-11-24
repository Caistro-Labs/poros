"""The fused compute's launch configurations, by padded LoRA rank (torch- and Triton-free).

``poros.core.fused.kernels`` launches the dense kernels with ``tiles(R)``, ``poros.moe.kernels``
the expert kernels with ``moe_tiles(rp)``; ``poros.plan`` reads the same tables to size what a run
holds (``dense_partial_bytes``) and which compute it takes on a card (``fused_rank_reason``).

The tiles depend on the rank alone, never on the card: a rank computes the same sums in the same
order on every GPU. Ranks up to 32 (dense) and 64 (MoE) keep the configuration every receipt
ran, so their bits are unchanged (the digests in tests/gpu pin them). An adapter tile is
[BLOCK, R] fp32, so above that the blocks shrink with R until every kernel fits the smallest
per-block shared memory of a supported card, 101,376 B on sm_86 / sm_89 / sm_120
(``poros.core.card.shared_memory_limit``).
The ahead-of-time compile of every kernel at every rank, per architecture, is
docs/validation/fused/rank_tiles/ (``poros.core.card.FUSED_FIT`` is its summary).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

Cap = Tuple[int, int]

# Ranks <= 32 (dense): the configuration every receipt ran.
FWD_BM, FWD_BN, FWD_WARPS = 64, 128, 4
# backward kernels loop over rows: two pipeline stages keep their shared memory under the 4090's
BWD_BM, BWD_BN, BWD_WARPS, BWD_STAGES = 32, 128, 8, 2
SWIGLU_BWD_BM, SWIGLU_BWD_BN, SWIGLU_BWD_WARPS, SWIGLU_BWD_STAGES = 32, 64, 4, 1
TARGET_PROGRAMS = 512


class Tiles(NamedTuple):
    """One padded LoRA rank's launch configuration of every dense kernel: (BLOCK_M, BLOCK_N,
    warps[, stages]) per kernel, and ``chunk``: whether a kernel that writes per-column-tile
    partials ([column tiles, rows, R] fp32) takes a call's rows in chunks (``row_chunk``)."""

    out_fwd: Tuple[int, int, int]  # _lora_out_fwd
    out_bwd: Tuple[int, int, int, int]  # _lora_out_bwd
    swiglu_fwd: Tuple[int, int, int]  # _swiglu_fwd
    swiglu_bwd: Tuple[int, int, int, int]  # _swiglu_bwd
    rope_fwd: Tuple[int, int]  # _lora_rope_fwd: BLOCK_M, warps (a head is the column tile)
    rope_bwd: Tuple[int, int, int]  # _lora_rope_bwd: BLOCK_M, warps, stages
    chunk: bool
    partial_elems: int = 0  # with chunk: a partial buffer's most elements a launch (row_chunk)


# The target is the smallest per-block shared memory of a supported card: 101,376 B (99 KB) on
# sm_86 / sm_89 / sm_120. At R = 64 every dense kernel takes at most 60 KB, at R = 128 at most
# 80 KB (the q/k rotary pair at head dim 128; 78 KB without it). The loops run one stage above
# 32: no supported card then needs fewer. (Unsupported pre-Ampere cards, informational only: the
# same tiles fit sm_70's 96 KB at R = 128, sm_75's 64 KB only up to R = 64.)
# Larger ranks take R = 128's tiles; where those do not fit a card,
# poros.core.fused.fit keeps PEFT's modules and says why.
_TILES_LE32 = Tiles((FWD_BM, FWD_BN, FWD_WARPS), (BWD_BM, BWD_BN, BWD_WARPS, BWD_STAGES),
                    (FWD_BM, FWD_BN, FWD_WARPS),
                    (SWIGLU_BWD_BM, SWIGLU_BWD_BN, SWIGLU_BWD_WARPS, SWIGLU_BWD_STAGES),
                    (FWD_BM, FWD_WARPS), (BWD_BM, BWD_WARPS, BWD_STAGES), False)
_TILES_64 = Tiles((64, 128, 4), (32, 128, 8, 1), (32, 64, 4), (32, 32, 4, 1),
                  (64, 4), (32, 8, 1), True, 1 << 24)
# R >= 128: the SwiGLU backward's column tile 32 and 256 MiB a partial buffer: at Qwen2.5-72B's
# widths (4,096 rows a call, RTX 4090, still at 4 warps) 2.658 s an update against 2.712 with
# column tile 16 and 2.726 with 64 MiB partials
# (docs/validation/fused/rank_tiles/rtx4090/m3_r128var.out.txt). At 4 warps the plain linear's
# forward and backward and the rotary pair spilled registers (1,184 / 788 / 56 spills): eight
# warps hold their [BLOCK, 128] fp32 tiles (0 / 0 / 22), 4-10x faster a call; the SwiGLU forward
# takes 32 x 32 tiles (every candidate, timed: docs/validation/fused/rank_tiles/rtx4090/t1_*).
_TILES_128 = Tiles((32, 64, 8), (16, 64, 8, 2), (32, 32, 4), (16, 32, 4, 1),
                   (16, 8), (16, 8, 1), True, 1 << 26)
# Above rank 32 a call's rows are taken in chunks of whole row blocks so a partial buffer holds at
# most the rank's ``partial_elems`` (64 MiB fp32 at 64, 256 MiB from 128), not growing with
# R / BLOCK_N; never more than PARTIAL_ELEMS (the cap tests lower to force many chunks). The chunks
# follow the call's shape alone.
PARTIAL_ELEMS = 1 << 26


def pad(rank: int) -> int:
    """A LoRA rank padded as the kernels take it: a power of two, at least 16."""
    return max(16, 1 << (int(rank) - 1).bit_length())


def tiles(R: int) -> Tiles:
    """The dense launch configuration of padded rank ``R``."""
    return _TILES_LE32 if R <= 32 else _TILES_64 if R <= 64 else _TILES_128


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def row_chunk(t: Tiles, M: int, n_tiles: int, R: int, bm: int) -> int:
    """Rows per launch of a kernel writing ``[n_tiles, rows, R]`` partials over ``M`` rows in
    row blocks of ``bm``: all of them up to rank 32, else whole blocks under the rank's
    ``partial_elems`` (and ``PARTIAL_ELEMS``)."""
    if not t.chunk:
        return M
    per = max(1, min(t.partial_elems, PARTIAL_ELEMS) // (n_tiles * R) // bm) * bm
    return min(M, per)


def splits(M: int, N: int, bm: int, bn: int) -> Tuple[int, int, int]:
    """(column tiles, row splits, rows per split) for a backward kernel's grid."""
    n_tiles = cdiv(N, bn)
    m_blocks = cdiv(M, bm)
    s = max(1, min(m_blocks, cdiv(TARGET_PROGRAMS, n_tiles)))
    rows = cdiv(m_blocks, s) * bm
    return n_tiles, cdiv(M, rows), rows


def dense_partial_bytes(rank: int, M: int, mid: int, hidden: int) -> int:
    """The most bytes of partials one fused call holds at ``rank`` over ``M`` rows, a model of MLP
    width ``mid`` and hidden width ``hidden``: the SwiGLU backward's (its column-tile partials of
    the gate/up A-product grads, its row-split partials of the three weight grads, and above
    rank 32 the call's [M, R] sums), the SwiGLU forward's and the plain linear backward's."""
    R = pad(rank)
    t = tiles(R)
    f = 4
    bm, bn = t.swiglu_bwd[:2]
    mc = row_chunk(t, M, cdiv(mid, bn), R, bm)
    nt, ns, _ = splits(mc, mid, bm, bn)
    sw_bwd = (2 * nt * mc * R + 3 * ns * mid * R) * f
    if t.chunk:  # the call's rows and the weight-grad sums, beside a chunk's partials
        sw_bwd += (2 * M * R + 3 * mid * R) * f
    bm, bn = t.swiglu_fwd[:2]
    mc = row_chunk(t, M, cdiv(mid, bn), R, bm)
    sw_fwd = cdiv(mid, bn) * mc * R * f + (M * R * f if t.chunk else 0)
    lin = 0
    for n in (mid, hidden):
        bm, bn = t.out_bwd[:2]
        mc = row_chunk(t, M, cdiv(n, bn), R, bm)
        nt, ns, _ = splits(mc, n, bm, bn)
        lin = max(lin, (nt * mc * R + ns * n * R) * f + ((M + n) * R * f if t.chunk else 0))
    return max(sw_bwd, sw_fwd, lin)


def dense_partial_bound(rank: int, M: int, mid: int, hidden: int) -> int:
    """The most of ``dense_partial_bytes`` over every padded rank up to ``rank``'s: a bound that
    grows with the rank (rank 32's whole-call partials can exceed rank 64's chunked ones)."""
    R, best = 16, 0
    while R <= pad(rank):
        best = max(best, dense_partial_bytes(R, M, mid, hidden))
        R *= 2
    return best


class MoeTiles(NamedTuple):
    """A padded rank's expert-kernel column tiles: ``bn`` of the gate/up forward and the mid
    backward, ``wide_bn`` / ``wide_warps`` of the down forward and the input grad."""

    bn: int
    wide_bn: int
    wide_warps: int


# Up to 64 the configuration every receipt ran. At RP 128 the [BM, RP] and [RP, BN] fp32 adapter
# tiles of the down forward and the input grad took 160 KB, past sm_86 / sm_89 / sm_120's 99 KB:
# the column tiles narrow until every expert kernel takes at most 64 KB there. A row's value is
# still its inputs' alone, in one order.
_MOE_LE64 = MoeTiles(64, 256, 8)
_MOE_128 = MoeTiles(32, 64, 4)


def moe_tiles(rp: int) -> MoeTiles:
    """The expert kernels' column tiles at padded rank ``rp``."""
    return _MOE_LE64 if rp <= 64 else _MOE_128


def rope_kernel_head_dim(head_dim: Optional[int]) -> Optional[int]:
    """``head_dim`` where the q/k rotary kernels take it (a power of two, at least 32: half of it
    one 16-wide tile or more), else None, where transformers' rotary function runs instead. One
    rule for the fused layer (poros.core.fused), its fit (``_keep_unfit_ranks``) and the plan."""
    if not isinstance(head_dim, int) or head_dim < 32 or head_dim & (head_dim - 1):
        return None
    return head_dim
