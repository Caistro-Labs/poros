"""Triton kernels for the fused LoRA-over-NF4 decoder compute (``poros.core.fused``).

Imported only on CUDA, and only when the fused compute is on. Every kernel has one fixed launch
configuration per shape (no autotuning) and no atomics: a value is the same on every run and in
every Poros path (streamed, resident, sequential, block-major). A sum split across programs is
written as per-program partials and reduced afterwards by ``torch.sum`` over the partial axis.

Shapes: ``M`` rows (tokens), ``N`` output features, ``R`` the LoRA rank padded to a power of two
>= 16 (zero rows/columns, exact). LoRA ``A`` is ``[R, K]``, ``B`` is ``[N, R]`` (PEFT's
``lora_A.weight`` / ``lora_B.weight``). The LoRA math is fp32 (IEEE dot products, no TF32); the
frozen GEMMs' inputs and outputs are the compute dtype (bf16), as in bitsandbytes.

Backward kernels run over (column tile, row split) programs, each looping over its rows: sums over
rows (the LoRA weight grads) stay in registers, and only the sums over columns (the LoRA
A-product grads, ``[M, R]``) go through partials.

Element offsets are int64 (row indices and partial-buffer indices are cast before they are
multiplied): an int32 ``row * N`` wraps past 2^31 elements, e.g. Qwen2.5-72B's MLP width
(N = 29,568) past ~72.6k tokens per micro-batch. Below that the offsets, and so the values, are
the same.
"""

from __future__ import annotations

import triton
import triton.language as tl

from poros.core.fused.launch import launch  # noqa: F401  (K.launch: fits the card)

# Each padded LoRA rank's launch configuration (torch-free, so the planner reads it too):
# ``tiles(R)``; FWD_* / BWD_* / SWIGLU_BWD_* are ranks <= 32's.
from poros.core.fused_tiles import (  # noqa: F401
    BWD_BM,
    BWD_BN,
    BWD_STAGES,
    BWD_WARPS,
    FWD_BM,
    FWD_BN,
    FWD_WARPS,
    PARTIAL_ELEMS,
    SWIGLU_BWD_BM,
    SWIGLU_BWD_BN,
    SWIGLU_BWD_STAGES,
    SWIGLU_BWD_WARPS,
    TARGET_PROGRAMS,
    Tiles,
    row_chunk,
    splits,
    tiles,
)


@triton.jit
def _lora_sum(base, at, bt, s):
    """float(base) + s * (at @ bt): a frozen output plus its scaled LoRA product, fp32."""
    return base.to(tl.float32) + tl.dot(at, bt, input_precision="ieee") * s


@triton.jit
def _swiglu(g, u):
    """(sigmoid(g), silu(g), silu(g) * u), one expression for the forward and the backward."""
    sig = 1.0 / (1.0 + tl.exp(-g))
    silu = g * sig
    return sig, silu, silu * u


@triton.jit
def _lora_out_fwd(base, a, b, out, res, M, N, s,
                  R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  RESIDUAL: tl.constexpr):
    """out[M, N] (fp32) = float(base[M, N]) + s * (a[M, R] @ b[N, R]^T), then (RESIDUAL)
    res[M, N] + that: the decoder layer's residual add, in its order."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, R)
    mm = om < M
    mn = on < N
    at = tl.load(a + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
    bt = tl.load(b + on[None, :] * R + rr[:, None], mask=mn[None, :], other=0.0)
    m2 = mm[:, None] & mn[None, :]
    off = om[:, None] * N + on[None, :]
    y = _lora_sum(tl.load(base + off, mask=m2, other=0.0), at, bt, s)
    if RESIDUAL:
        y = tl.load(res + off, mask=m2, other=0.0) + y
    tl.store(out + off, y, mask=m2)


@triton.jit
def _lora_out_bwd(g, a, b, gb, da_part, db_part, M, N, rows_per_split,
                  R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """For the grad g[M, N] (fp32) of ``_lora_out_fwd``'s output: gb = g in the compute dtype (the
    frozen GEMM's grad); per column tile, the partial g @ b ([M, R]); per row split, g^T @ a for
    this column tile ([N, R]). The caller reduces the partials and applies the LoRA scale."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, R)
    mn = on < N
    bt = tl.load(b + on[:, None] * R + rr[None, :], mask=mn[:, None], other=0.0)  # [BN, R]
    db = tl.zeros((BLOCK_N, R), dtype=tl.float32)
    start = ps * rows_per_split
    for m0 in range(start, tl.minimum(start + rows_per_split, M), BLOCK_M):
        om = (m0 + tl.arange(0, BLOCK_M)).to(tl.int64)
        mm = om < M
        m2 = mm[:, None] & mn[None, :]
        gt = tl.load(g + om[:, None] * N + on[None, :], mask=m2, other=0.0)
        tl.store(gb + om[:, None] * N + on[None, :], gt.to(gb.dtype.element_ty), mask=m2)
        tl.store(da_part + pn.to(tl.int64) * M * R + om[:, None] * R + rr[None, :],
                 tl.dot(gt, bt, input_precision="ieee"), mask=mm[:, None])
        at = tl.load(a + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
        db += tl.dot(tl.trans(gt), at, input_precision="ieee")
    tl.store(db_part + ps.to(tl.int64) * N * R + on[:, None] * R + rr[None, :], db,
             mask=mn[:, None])


@triton.jit
def _swiglu_fwd(bg, bu, ag, au, wbg, wbu, hb, hf, ad_part, wad, M, N, s,
                R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                WRITE_HF: tl.constexpr, DOWN_A: tl.constexpr):
    """SwiGLU over LoRA'd gate/up: g = float(bg) + s*(ag @ wbg^T), u likewise, h = silu(g) * u.
    Writes h in the compute dtype (hb, the frozen down GEMM's input); optionally in fp32 (hf); and,
    with DOWN_A, per column tile the partial h @ wad^T (the down LoRA's A product, [M, R])."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, R)
    mm = om < M
    mn = on < N
    m2 = mm[:, None] & mn[None, :]
    agt = tl.load(ag + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
    aut = tl.load(au + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
    wgt = tl.load(wbg + on[None, :] * R + rr[:, None], mask=mn[None, :], other=0.0)
    wut = tl.load(wbu + on[None, :] * R + rr[:, None], mask=mn[None, :], other=0.0)
    off = om[:, None] * N + on[None, :]
    g = _lora_sum(tl.load(bg + off, mask=m2, other=0.0), agt, wgt, s)
    u = _lora_sum(tl.load(bu + off, mask=m2, other=0.0), aut, wut, s)
    _, _, h = _swiglu(g, u)
    tl.store(hb + off, h.to(hb.dtype.element_ty), mask=m2)
    if WRITE_HF:
        tl.store(hf + off, h, mask=m2)
    if DOWN_A:
        wat = tl.load(wad + rr[None, :] * N + on[:, None], mask=mn[:, None], other=0.0)  # [BN, R]
        part = tl.dot(tl.where(m2, h, 0.0), wat, input_precision="ieee")  # [BM, R]
        tl.store(ad_part + pn.to(tl.int64) * M * R + om[:, None] * R + rr[None, :], part,
                 mask=mm[:, None])


@triton.jit
def _swiglu_bwd(bg, bu, ag, au, wbg, wbu, dzb, dad, wad, dropm,
                dgb, dub, dag_part, dau_part, dbg_part, dbu_part, dwad_part,
                M, N, s, pscale, rows_per_split,
                R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                HAS_DROP: tl.constexpr):
    """Backward of ``_swiglu_fwd`` and of the down LoRA's input side, per (column tile, row split).

    dh = float(dzb) + mask*pscale * (dad @ wad): the down GEMM's input grad plus its LoRA's, through
    the down dropout when HAS_DROP. g, u, h are recomputed from the saved bg/bu/ag/au. Writes dg/du
    in the compute dtype (the frozen gate/up GEMM grads); per column tile the partials dg @ wbg and
    du @ wbu ([M, R], the gate/up LoRA A-product grads); per row split dg^T @ ag, du^T @ au ([N, R],
    the gate/up LoRA B grads) and dad^T @ dropout(h) ([R, N], the down LoRA A grad)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, R)
    mn = on < N
    wgt = tl.load(wbg + on[None, :] * R + rr[:, None], mask=mn[None, :], other=0.0)  # [R, BN]
    wut = tl.load(wbu + on[None, :] * R + rr[:, None], mask=mn[None, :], other=0.0)
    wat = tl.load(wad + rr[:, None] * N + on[None, :], mask=mn[None, :], other=0.0)  # [R, BN]
    dbg = tl.zeros((BLOCK_N, R), dtype=tl.float32)
    dbu = tl.zeros((BLOCK_N, R), dtype=tl.float32)
    dwa = tl.zeros((R, BLOCK_N), dtype=tl.float32)
    start = ps * rows_per_split
    for m0 in range(start, tl.minimum(start + rows_per_split, M), BLOCK_M):
        om = (m0 + tl.arange(0, BLOCK_M)).to(tl.int64)
        mm = om < M
        m2 = mm[:, None] & mn[None, :]
        off = om[:, None] * N + on[None, :]
        agt = tl.load(ag + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
        aut = tl.load(au + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
        g = _lora_sum(tl.load(bg + off, mask=m2, other=0.0), agt, wgt, s)
        u = _lora_sum(tl.load(bu + off, mask=m2, other=0.0), aut, wut, s)
        sig, silu, h = _swiglu(g, u)
        dadt = tl.load(dad + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)  # [BM, R]
        dl = tl.dot(dadt, wat, input_precision="ieee")
        if HAS_DROP:
            keep = tl.load(dropm + off, mask=m2, other=0).to(tl.float32)
            dl = dl * keep * pscale
            hd = h * keep * pscale
        else:
            hd = h
        dh = tl.load(dzb + off, mask=m2, other=0.0).to(tl.float32) + dl
        dg = tl.where(m2, dh * u * (sig * (1.0 + g * (1.0 - sig))), 0.0)
        du = tl.where(m2, dh * silu, 0.0)
        tl.store(dgb + off, dg.to(dgb.dtype.element_ty), mask=m2)
        tl.store(dub + off, du.to(dub.dtype.element_ty), mask=m2)
        tl.store(dag_part + pn.to(tl.int64) * M * R + om[:, None] * R + rr[None, :],
                 tl.dot(dg, tl.trans(wgt), input_precision="ieee"), mask=mm[:, None])
        tl.store(dau_part + pn.to(tl.int64) * M * R + om[:, None] * R + rr[None, :],
                 tl.dot(du, tl.trans(wut), input_precision="ieee"), mask=mm[:, None])
        dbg += tl.dot(tl.trans(dg), agt, input_precision="ieee")
        dbu += tl.dot(tl.trans(du), aut, input_precision="ieee")
        dwa += tl.dot(tl.trans(dadt), tl.where(m2, hd, 0.0), input_precision="ieee")
    ps64 = ps.to(tl.int64)
    tl.store(dbg_part + ps64 * N * R + on[:, None] * R + rr[None, :], dbg, mask=mn[:, None])
    tl.store(dbu_part + ps64 * N * R + on[:, None] * R + rr[None, :], dbu, mask=mn[:, None])
    tl.store(dwad_part + ps64 * R * N + rr[:, None] * N + on[None, :], dwa, mask=mn[None, :])


@triton.jit
def _add_f32(out, x0, x1, x2, M, K,
             NT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[M, K] (fp32) += float(x0) [+ float(x1) [+ float(x2)]], added left to right."""
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    om = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    ok = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    m2 = (om < M)[:, None] & (ok < K)[None, :]
    off = om[:, None] * K + ok[None, :]
    acc = tl.load(out + off, mask=m2, other=0.0)
    acc = acc + tl.load(x0 + off, mask=m2, other=0.0).to(tl.float32)
    if NT > 1:
        acc = acc + tl.load(x1 + off, mask=m2, other=0.0).to(tl.float32)
    if NT > 2:
        acc = acc + tl.load(x2 + off, mask=m2, other=0.0).to(tl.float32)
    tl.store(out + off, acc, mask=m2)


def cdiv(a: int, b: int) -> int:
    return triton.cdiv(a, b)


@triton.jit
def _rmsnorm_fwd(x, w, y, rstd, H, eps, BLOCK: tl.constexpr):
    """One row: y = w * (x * rsqrt(mean(x^2) + eps)), fp32 (HF's RMSNorm on an fp32 input); the
    row's rsqrt kept for the backward."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < H
    xv = tl.load(x + row * H + cols, mask=m, other=0.0)
    var = tl.sum(xv * xv, axis=0) / H
    r = 1.0 / tl.sqrt(var + eps)
    wv = tl.load(w + cols, mask=m, other=0.0)
    tl.store(y + row * H + cols, wv * (xv * r), mask=m)
    tl.store(rstd + row, r)


@triton.jit
def _rmsnorm_bwd(dy, x, w, rstd, dx, H, BLOCK: tl.constexpr):
    """One row of the input grad of ``_rmsnorm_fwd`` (the weight is frozen: no weight grad)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < H
    xv = tl.load(x + row * H + cols, mask=m, other=0.0)
    dyv = tl.load(dy + row * H + cols, mask=m, other=0.0)
    wv = tl.load(w + cols, mask=m, other=0.0)
    r = tl.load(rstd + row)
    xh = xv * r
    wdy = wv * dyv
    c = tl.sum(xh * wdy, axis=0) / H
    tl.store(dx + row * H + cols, (wdy - xh * c) * r, mask=m)


@triton.jit
def _rope_rows(cos, sin, om, S, BC, HALF: tl.constexpr):
    """cos/sin [BC, S, 2*HALF] halves for token rows om (row = batch*S + position)."""
    crow = ((om // S) % BC) * S + om % S
    oh = tl.arange(0, HALF)
    base = crow[:, None] * (2 * HALF) + oh[None, :]
    return (tl.load(cos + base), tl.load(cos + base + HALF),
            tl.load(sin + base), tl.load(sin + base + HALF))


@triton.jit
def _lora_rope_fwd(base, a, b, out, cos, sin, M, N, S, BC, s,
                   R: tl.constexpr, BLOCK_M: tl.constexpr, HALF: tl.constexpr):
    """``_lora_out_fwd`` for one head's columns (program axis 1), then the rotary embedding:
    out = y * cos + rotate_half(y) * sin (transformers' apply_rotary_pos_emb), y the LoRA sum."""
    pm = tl.program_id(0)
    ph = tl.program_id(1)
    om = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    lo = ph * 2 * HALF + tl.arange(0, HALF)
    hi = lo + HALF
    rr = tl.arange(0, R)
    mm = om < M
    at = tl.load(a + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
    blo = tl.load(b + lo[None, :] * R + rr[:, None])
    bhi = tl.load(b + hi[None, :] * R + rr[:, None])
    ylo = _lora_sum(tl.load(base + om[:, None] * N + lo[None, :], mask=mm[:, None], other=0.0),
                    at, blo, s)
    yhi = _lora_sum(tl.load(base + om[:, None] * N + hi[None, :], mask=mm[:, None], other=0.0),
                    at, bhi, s)
    clo, chi, slo, shi = _rope_rows(cos, sin, tl.where(mm, om, 0), S, BC, HALF)
    tl.store(out + om[:, None] * N + lo[None, :], ylo * clo - yhi * slo, mask=mm[:, None])
    tl.store(out + om[:, None] * N + hi[None, :], yhi * chi + ylo * shi, mask=mm[:, None])


@triton.jit
def _lora_rope_bwd(g, a, b, gb, da_part, db_part, cos, sin, M, N, S, BC, rows_per_split,
                   R: tl.constexpr, BLOCK_M: tl.constexpr, HALF: tl.constexpr):
    """``_lora_out_bwd`` for one head's columns, the grad first taken back through the rotary
    embedding (its transpose: dy_lo = g_lo*cos_lo + g_hi*sin_hi, dy_hi = g_hi*cos_hi -
    g_lo*sin_lo)."""
    ph = tl.program_id(0)
    ps = tl.program_id(1)
    lo = ph * 2 * HALF + tl.arange(0, HALF)
    hi = lo + HALF
    rr = tl.arange(0, R)
    blo = tl.load(b + lo[:, None] * R + rr[None, :])  # [HALF, R]
    bhi = tl.load(b + hi[:, None] * R + rr[None, :])
    dblo = tl.zeros((HALF, R), dtype=tl.float32)
    dbhi = tl.zeros((HALF, R), dtype=tl.float32)
    start = ps * rows_per_split
    for m0 in range(start, tl.minimum(start + rows_per_split, M), BLOCK_M):
        om = (m0 + tl.arange(0, BLOCK_M)).to(tl.int64)
        mm = om < M
        glo = tl.load(g + om[:, None] * N + lo[None, :], mask=mm[:, None], other=0.0)
        ghi = tl.load(g + om[:, None] * N + hi[None, :], mask=mm[:, None], other=0.0)
        clo, chi, slo, shi = _rope_rows(cos, sin, tl.where(mm, om, 0), S, BC, HALF)
        dlo = glo * clo + ghi * shi
        dhi = ghi * chi - glo * slo
        tl.store(gb + om[:, None] * N + lo[None, :], dlo.to(gb.dtype.element_ty), mask=mm[:, None])
        tl.store(gb + om[:, None] * N + hi[None, :], dhi.to(gb.dtype.element_ty), mask=mm[:, None])
        da = tl.dot(dlo, blo, input_precision="ieee") + tl.dot(dhi, bhi, input_precision="ieee")
        tl.store(da_part + ph.to(tl.int64) * M * R + om[:, None] * R + rr[None, :], da,
                 mask=mm[:, None])
        at = tl.load(a + om[:, None] * R + rr[None, :], mask=mm[:, None], other=0.0)
        dblo += tl.dot(tl.trans(dlo), at, input_precision="ieee")
        dbhi += tl.dot(tl.trans(dhi), at, input_precision="ieee")
    tl.store(db_part + ps.to(tl.int64) * N * R + lo[:, None] * R + rr[None, :], dblo)
    tl.store(db_part + ps.to(tl.int64) * N * R + hi[:, None] * R + rr[None, :], dbhi)
