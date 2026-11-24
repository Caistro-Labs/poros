"""Triton kernels of the MoE expert path (CUDA only; imported lazily).

``dequantize_into``: one pass from a ``poros.moe.quant`` NF4 blob to the compute dtype, bit for bit
``quant.dequantize``'s value. The torch reference decodes in five passes over fp32 intermediates
(the int cast, the byte-table gather, two block scalings, the cast); this reads the packed bytes
once and writes the weights once. The fp32 arithmetic is the reference's, op for op: the
double-quant absmax is ``code8[q] * absmax2`` then ``+ offset`` (two roundings: the launch turns
off fp fusion, so no FMA contracts them), each value ``code4[nibble] * absmax``, one cast to the
compute dtype (round to nearest even, as ``Tensor.to``).

The kernel reads each expert's slab through a table of device pointers, so the resident slabs are
decoded where they are, without stacking them into one tensor first.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from poros.core.fused.launch import launch  # noqa: F401  (K.launch: fits the card)
from poros.moe import quant
from poros.moe.pinned import try_pin

DQ_BLOCKS = 32  # 64-element NF4 blocks per program
_DQ2 = tl.constexpr(quant.DOUBLE_QUANT_BLOCKSIZE)  # NF4 blocks per second-level absmax


@triton.jit
def _nf4_dequant(slab_ptrs, out, code8, code4, start, pa, a2, off, n_blocks, out_stride,
                 BLOCKS: tl.constexpr):
    """Program (block range, expert): BLOCKS 64-element blocks of one expert's tensor."""
    pid = tl.program_id(0)
    e = tl.program_id(1)
    base = tl.load(slab_ptrs + e).to(tl.pointer_type(tl.uint8)) + start
    blk = pid * BLOCKS + tl.arange(0, BLOCKS)
    mb = blk < n_blocks
    q8 = tl.load(base + pa + blk, mask=mb, other=0).to(tl.int32)
    am2 = tl.load((base + a2).to(tl.pointer_type(tl.float32)) + blk // _DQ2,
                  mask=mb, other=0.0)
    offset = tl.load((base + off).to(tl.pointer_type(tl.float32)))
    absmax = tl.load(code8 + q8, mask=mb, other=0.0) * am2
    absmax = absmax + offset
    j = tl.arange(0, 32)
    byte = tl.load(base + blk[:, None] * 32 + j[None, :], mask=mb[:, None], other=0).to(tl.int32)
    hi = tl.load(code4 + (byte >> 4))
    lo = tl.load(code4 + (byte & 15))
    vals = tl.join(hi, lo).reshape(BLOCKS, 64) * absmax[:, None]  # even elements: high nibble
    o = tl.arange(0, 64)
    tl.store(out + e.to(tl.int64) * out_stride + blk[:, None] * 64 + o[None, :],
             vals.to(out.dtype.element_ty), mask=mb[:, None])


_TABLES: dict = {}


def _tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = _TABLES.get(device)
    if t is None:
        t = _TABLES[device] = (quant.dynamic_codebook().to(device), quant.nf4_codebook().to(device))
    return t


def cdiv(a: int, b: int) -> int:
    return triton.cdiv(a, b)


def supports(shape) -> bool:
    """Whether ``dequantize_into`` decodes a tensor of this shape (whole 64-element blocks)."""
    n = 1
    for d in shape:
        n *= int(d)
    return n > 0 and n % quant.NF4_BLOCKSIZE == 0


def slab_pointers(slabs) -> torch.Tensor:
    """An int64 device tensor of the slabs' data pointers (a list of 1-D uint8 slabs on one device,
    or the rows of a 2-D uint8 tensor). Made on the host and copied from page-locked memory on
    the current stream, ahead of the decode that reads it, so the host does not wait: a copy from
    pageable memory first drains the GPU, at every decode (768 waits an update of Qwen3-30B-A3B
    at 8 micro-batches of one row)."""
    if isinstance(slabs, torch.Tensor):
        row_bytes = slabs.stride(0) * slabs.element_size()
        host = torch.arange(slabs.shape[0], dtype=torch.int64) * row_bytes + slabs.data_ptr()
        device = slabs.device
    else:
        host = torch.tensor([s.data_ptr() for s in slabs], dtype=torch.int64)
        device = slabs[0].device
    if device.type == "cuda":
        host = try_pin(host)  # pinned where the machine pins: the copy does not wait
    return host.to(device, non_blocking=True)


def dequantize_into(out: torch.Tensor, ptrs: torch.Tensor, start: int, shape) -> torch.Tensor:
    """Decode one tensor (``shape``, blob at byte ``start`` of each slab) of every slab in ``ptrs``
    into ``out`` [len(ptrs), *shape] (bf16 / fp16 / fp32), bit for bit ``quant.dequantize``."""
    n = 1
    for d in shape:
        n *= int(d)
    packed_b, nblocks, nabsmax2 = quant._section_sizes(n)
    pa = quant._align4(packed_b)
    a2 = pa + quant._align4(nblocks)
    off = a2 + 4 * nabsmax2
    code8, code4 = _tables(out.device)
    grid = (triton.cdiv(nblocks, DQ_BLOCKS), ptrs.numel())
    _nf4_dequant[grid](ptrs, out, code8, code4, start, pa, a2, off, nblocks, n,
                       BLOCKS=DQ_BLOCKS, num_warps=4, enable_fp_fusion=False)
    return out


# ---------------------------------------------------------------------------------------------
# Fused grouped expert compute (``poros.moe.fused``).
#
# A layer call's ``tokens x top_k`` rows are sorted by expert (token order within an expert).
# GEMM kernels run over row tiles: tile i covers sorted rows [t_r0[i], t_r1[i]) of one expert, whose
# decoded weights are entry t_w[i] of a compact [hit experts, ...] tensor and whose adapter rows are
# entry t_e[i] of the full [experts, ...] adapter tensors. Every launch uses one fixed configuration
# (no autotuning, no split-K, no atomics), and each output row of a GEMM kernel is computed from
# that row's inputs alone with one fixed reduction order -- so a row's value does not depend on how
# many rows, or which other rows, share its call. Sums over rows (the adapter weight grads) run per
# (expert, segment) group, a program looping over the group's rows in tiles that start at the
# group's first row: a segment's sums are the same whether it runs alone or beside others.
# Integer sizes that change from call to call are not specialized on.
# ---------------------------------------------------------------------------------------------

# one launch configuration for every call (an RTX 4090 sweep at 4,096 tokens of Qwen3-30B-A3B:
# 9.75 ms forward + backward, against 11.25 ms at 3 stages and 128-wide tiles)
GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WARPS, GEMM_STAGES = 64, 64, 64, 4, 2
# the GEMMs with one accumulator over the hidden width (down forward, input grad): wider tiles
WIDE_BN, WIDE_WARPS = 256, 8
WG_BM, WG_BN = 32, 64
COMB_BT, COMB_BH = 16, 256


# A padded rank's column tiles (torch-free, beside the dense ones): ``tiles(rp)``.
from poros.core.fused_tiles import MoeTiles as Tiles  # noqa: E402,F401
from poros.core.fused_tiles import moe_tiles as tiles  # noqa: E402,F401


@triton.jit
def _dropped(xt, keep, pscale):
    """An adapter input through its dropout: x * keep * (1 / (1 - p)) in fp32, in x's dtype."""
    return (xt.to(tl.float32) * keep.to(tl.float32) * pscale).to(xt.dtype)


@triton.jit(do_not_specialize=["n_rows"])
def _lora_rows(x, row_tok, m, out, keep, pscale, t_r0, t_r1, t_e, n_rows, K, r, m_se, m_sr,
               m_sk, RP: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
               GATHER: tl.constexpr, DROP: tl.constexpr):
    """out[i, :RP] (fp32) = X[i] @ M_e^T for the rows i of tile pid, M_e[j, k] at
    m + e*m_se + j*m_sr + k*m_sk (j < r; zero for r <= j < RP). X[i] = x[row_tok[i]] (GATHER) or
    x[i], through the dropout keep[i] (DROP; sorted rows); X rows have K elements."""
    pid = tl.program_id(0)
    r0 = tl.load(t_r0 + pid)
    r1 = tl.load(t_r1 + pid)
    e = tl.load(t_e + pid).to(tl.int64)
    rows = r0 + tl.arange(0, BM)
    mr = rows < r1
    if GATHER:
        src = tl.load(row_tok + rows, mask=mr, other=0).to(tl.int64)
    else:
        src = rows.to(tl.int64)
    jj = tl.arange(0, RP)
    acc = tl.zeros((BM, RP), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < K
        xt = tl.load(x + src[:, None] * K + kk[None, :], mask=mr[:, None] & mk[None, :],
                     other=0.0)
        if DROP:
            kt = tl.load(keep + rows[:, None].to(tl.int64) * K + kk[None, :],
                         mask=mr[:, None] & mk[None, :], other=0)
            xt = _dropped(xt, kt, pscale)
        mt = tl.load(m + e * m_se + jj[None, :] * m_sr + kk[:, None] * m_sk,
                     mask=(jj[None, :] < r) & mk[:, None], other=0.0)
        acc = tl.dot(xt, mt.to(xt.dtype), acc)
    tl.store(out + rows[:, None].to(tl.int64) * RP + jj[None, :], acc, mask=mr[:, None])


@triton.jit
def _swiglu(g, u):
    """(sigmoid(g), silu(g), silu(g) * u) in fp32."""
    sig = 1.0 / (1.0 + tl.exp(-g))
    silu = g * sig
    return sig, silu, silu * u


@triton.jit(do_not_specialize=["n_rows"])
def _gate_up_fwd(x, row_tok, w, a1, b, hb, t_r0, t_r1, t_w, t_e, n_rows, H, NI, r, s, limit,
                 RP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 LIMIT: tl.constexpr, LORA: tl.constexpr):
    """hb[i, cols] = silu(g) * u for the rows of tile pid_m, g / u the gate / up columns of
    x[row_tok[i]] @ W_e^T (+ s * a1[i] @ B_e^T, fp32); W [*, 2I, H] decoded weights, B [E, 2I,
    r]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = tl.load(t_r0 + pid_m)
    r1 = tl.load(t_r1 + pid_m)
    ew = tl.load(t_w + pid_m).to(tl.int64)
    ea = tl.load(t_e + pid_m).to(tl.int64)
    rows = r0 + tl.arange(0, BM)
    mr = rows < r1
    src = tl.load(row_tok + rows, mask=mr, other=0).to(tl.int64)
    cols = pid_n * BN + tl.arange(0, BN)
    mc = cols < NI
    wbase = w + ew * (2 * NI) * H
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < H
        xt = tl.load(x + src[:, None] * H + kk[None, :], mask=mr[:, None] & mk[None, :],
                     other=0.0)
        wg = tl.load(wbase + cols[None, :].to(tl.int64) * H + kk[:, None],
                     mask=mc[None, :] & mk[:, None], other=0.0)
        wu = tl.load(wbase + (cols[None, :] + NI).to(tl.int64) * H + kk[:, None],
                     mask=mc[None, :] & mk[:, None], other=0.0)
        accg = tl.dot(xt, wg, accg)
        accu = tl.dot(xt, wu, accu)
    if LORA:
        jj = tl.arange(0, RP)
        at = tl.load(a1 + rows[:, None].to(tl.int64) * RP + jj[None, :], mask=mr[:, None],
                     other=0.0)
        bb = b + ea * (2 * NI) * r
        bg = tl.load(bb + cols[None, :] * r + jj[:, None], mask=(jj[:, None] < r) & mc[None, :],
                     other=0.0).to(tl.float32)
        bu = tl.load(bb + (cols[None, :] + NI) * r + jj[:, None],
                     mask=(jj[:, None] < r) & mc[None, :], other=0.0).to(tl.float32)
        accg = accg + s * tl.dot(at, bg, input_precision="ieee")
        accu = accu + s * tl.dot(at, bu, input_precision="ieee")
    if LIMIT:
        accg = tl.minimum(accg, limit)
        accu = tl.minimum(tl.maximum(accu, -limit), limit)
    _, _, h = _swiglu(accg, accu)
    tl.store(hb + rows[:, None].to(tl.int64) * NI + cols[None, :], h.to(hb.dtype.element_ty),
             mask=mr[:, None] & mc[None, :])


@triton.jit(do_not_specialize=["n_rows"])
def _down_fwd(hb, w, a2, b, wr, dest, buf, t_r0, t_r1, t_w, t_e, n_rows, H, NI, r, s,
              RP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              LORA: tl.constexpr):
    """buf[dest[i], cols] (fp32) = wr[i] * (hb[i] @ Wd_e^T (+ s * a2[i] @ Bd_e^T)) for the rows of
    tile pid_m; Wd [*, H, NI] decoded weights, Bd [E, H, r]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = tl.load(t_r0 + pid_m)
    r1 = tl.load(t_r1 + pid_m)
    ew = tl.load(t_w + pid_m).to(tl.int64)
    ea = tl.load(t_e + pid_m).to(tl.int64)
    rows = r0 + tl.arange(0, BM)
    mr = rows < r1
    cols = pid_n * BN + tl.arange(0, BN)
    mc = cols < H
    wbase = w + ew * H * NI
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, NI, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < NI
        ht = tl.load(hb + rows[:, None].to(tl.int64) * NI + kk[None, :],
                     mask=mr[:, None] & mk[None, :], other=0.0)
        wt = tl.load(wbase + cols[None, :].to(tl.int64) * NI + kk[:, None],
                     mask=mc[None, :] & mk[:, None], other=0.0)
        acc = tl.dot(ht, wt, acc)
    if LORA:
        jj = tl.arange(0, RP)
        at = tl.load(a2 + rows[:, None].to(tl.int64) * RP + jj[None, :], mask=mr[:, None],
                     other=0.0)
        bt = tl.load(b + ea * H * r + cols[None, :] * r + jj[:, None],
                     mask=(jj[:, None] < r) & mc[None, :], other=0.0).to(tl.float32)
        acc = acc + s * tl.dot(at, bt, input_precision="ieee")
    wv = tl.load(wr + rows, mask=mr, other=0.0)
    d = tl.load(dest + rows, mask=mr, other=0).to(tl.int64)
    tl.store(buf + d[:, None] * H + cols[None, :], acc * wv[:, None],
             mask=mr[:, None] & mc[None, :])


@triton.jit(do_not_specialize=["T"])
def _combine(buf, out, T, H, KTOP: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr):
    """out[t, cols] = sum over k = 0..KTOP-1, in that order, of buf[t * KTOP + k, cols] (fp32)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BH + tl.arange(0, BH)
    m = (t < T)[:, None] & (c < H)[None, :]
    base = buf + (t.to(tl.int64) * KTOP)[:, None] * H + c[None, :]
    acc = tl.load(base, mask=m, other=0.0)
    for k in tl.static_range(1, KTOP):
        acc += tl.load(base + k * H, mask=m, other=0.0)
    tl.store(out + t.to(tl.int64)[:, None] * H + c[None, :], acc.to(out.dtype.element_ty), mask=m)


@triton.jit(do_not_specialize=["n_rows"])
def _mid_bwd(x, gy, row_tok, wgu, wd, a1, c2, bgu, ad, wr, hb, dgu, dw_part, keep_h, pscale,
             t_r0, t_r1, t_w, t_e, n_rows, H, NI, r, s, limit,
             RP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             LIMIT: tl.constexpr, LORA: tl.constexpr, DROP: tl.constexpr):
    """Backward through the down projection and SwiGLU, for the rows of tile pid_m and intermediate
    columns pid_n: recomputes g, u (``_gate_up_fwd``), takes ud = gy[tok] @ Wd_e (+ s * c2 @ Ad_e),
    writes hb (the forward's down input), dgu = d[g, u] (bf16) and the routing-weight grad partial
    dw_part[pid_n, i] = sum over these columns of ud * hb."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = tl.load(t_r0 + pid_m)
    r1 = tl.load(t_r1 + pid_m)
    ew = tl.load(t_w + pid_m).to(tl.int64)
    ea = tl.load(t_e + pid_m).to(tl.int64)
    rows = r0 + tl.arange(0, BM)
    mr = rows < r1
    src = tl.load(row_tok + rows, mask=mr, other=0).to(tl.int64)
    cols = pid_n * BN + tl.arange(0, BN)
    mc = cols < NI
    gbase = wgu + ew * (2 * NI) * H
    dbase = wd + ew * H * NI
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    accd = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < H
        m2 = mr[:, None] & mk[None, :]
        xt = tl.load(x + src[:, None] * H + kk[None, :], mask=m2, other=0.0)
        gt = tl.load(gy + src[:, None] * H + kk[None, :], mask=m2, other=0.0)
        wg = tl.load(gbase + cols[None, :].to(tl.int64) * H + kk[:, None],
                     mask=mc[None, :] & mk[:, None], other=0.0)
        wu = tl.load(gbase + (cols[None, :] + NI).to(tl.int64) * H + kk[:, None],
                     mask=mc[None, :] & mk[:, None], other=0.0)
        wdt = tl.load(dbase + kk[:, None].to(tl.int64) * NI + cols[None, :],
                      mask=mk[:, None] & mc[None, :], other=0.0)
        accg = tl.dot(xt, wg, accg)
        accu = tl.dot(xt, wu, accu)
        accd = tl.dot(gt, wdt, accd)
    jj = tl.arange(0, RP)
    if LORA:
        at = tl.load(a1 + rows[:, None].to(tl.int64) * RP + jj[None, :], mask=mr[:, None],
                     other=0.0)
        bb = bgu + ea * (2 * NI) * r
        bg = tl.load(bb + cols[None, :] * r + jj[:, None], mask=(jj[:, None] < r) & mc[None, :],
                     other=0.0).to(tl.float32)
        bu = tl.load(bb + (cols[None, :] + NI) * r + jj[:, None],
                     mask=(jj[:, None] < r) & mc[None, :], other=0.0).to(tl.float32)
        accg = accg + s * tl.dot(at, bg, input_precision="ieee")
        accu = accu + s * tl.dot(at, bu, input_precision="ieee")
        ct = tl.load(c2 + rows[:, None].to(tl.int64) * RP + jj[None, :], mask=mr[:, None],
                     other=0.0)
        adt = tl.load(ad + ea * r * NI + jj[:, None] * NI + cols[None, :],
                      mask=(jj[:, None] < r) & mc[None, :], other=0.0).to(tl.float32)
        lora_d = s * tl.dot(ct, adt, input_precision="ieee")
        if DROP:  # the down adapter's input is dropout(h): its grad goes back through the mask
            kt = tl.load(keep_h + rows[:, None].to(tl.int64) * NI + cols[None, :],
                         mask=mr[:, None] & mc[None, :], other=0)
            lora_d = lora_d * kt.to(tl.float32) * pscale
        accd = accd + lora_d
    g = accg
    u = accu
    if LIMIT:
        g = tl.minimum(accg, limit)
        u = tl.minimum(tl.maximum(accu, -limit), limit)
    sig, silu, h = _swiglu(g, u)
    hbv = h.to(hb.dtype.element_ty)
    m2 = mr[:, None] & mc[None, :]
    tl.store(hb + rows[:, None].to(tl.int64) * NI + cols[None, :], hbv, mask=m2)
    tl.store(dw_part + pid_n.to(tl.int64) * n_rows + rows,
             tl.sum(tl.where(m2, accd * hbv.to(tl.float32), 0.0), axis=1), mask=mr)
    wv = tl.load(wr + rows, mask=mr, other=0.0)
    dh = accd * wv[:, None]
    dg = dh * u * (sig * (1.0 + g * (1.0 - sig)))
    du = dh * silu
    if LIMIT:
        dg = tl.where(accg > limit, 0.0, dg)
        du = tl.where((accu > limit) | (accu < -limit), 0.0, du)
    drow = dgu + rows[:, None].to(tl.int64) * (2 * NI)
    tl.store(drow + cols[None, :], dg.to(dgu.dtype.element_ty), mask=m2)
    tl.store(drow + NI + cols[None, :], du.to(dgu.dtype.element_ty), mask=m2)


@triton.jit(do_not_specialize=["n_rows"])
def _dx_rows(dgu, wgu, c1, agu, dest, buf, keep_x, pscale, t_r0, t_r1, t_w, t_e, n_rows, H, NI,
             r, s, RP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             LORA: tl.constexpr, DROP: tl.constexpr):
    """buf[dest[i], cols] (fp32) = dgu[i] @ Wgu_e (+ s * c1[i] @ Agu_e): a row's input grad."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = tl.load(t_r0 + pid_m)
    r1 = tl.load(t_r1 + pid_m)
    ew = tl.load(t_w + pid_m).to(tl.int64)
    ea = tl.load(t_e + pid_m).to(tl.int64)
    rows = r0 + tl.arange(0, BM)
    mr = rows < r1
    cols = pid_n * BN + tl.arange(0, BN)
    mc = cols < H
    wbase = wgu + ew * (2 * NI) * H
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, 2 * NI, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < 2 * NI
        dt = tl.load(dgu + rows[:, None].to(tl.int64) * (2 * NI) + kk[None, :],
                     mask=mr[:, None] & mk[None, :], other=0.0)
        wt = tl.load(wbase + kk[:, None].to(tl.int64) * H + cols[None, :],
                     mask=mk[:, None] & mc[None, :], other=0.0)
        acc = tl.dot(dt, wt, acc)
    if LORA:
        jj = tl.arange(0, RP)
        ct = tl.load(c1 + rows[:, None].to(tl.int64) * RP + jj[None, :], mask=mr[:, None],
                     other=0.0)
        at = tl.load(agu + ea * r * H + jj[:, None] * H + cols[None, :],
                     mask=(jj[:, None] < r) & mc[None, :], other=0.0).to(tl.float32)
        lora_x = s * tl.dot(ct, at, input_precision="ieee")
        if DROP:  # the gate/up adapter's input is dropout(x)
            kt = tl.load(keep_x + rows[:, None].to(tl.int64) * H + cols[None, :],
                         mask=mr[:, None] & mc[None, :], other=0)
            lora_x = lora_x * kt.to(tl.float32) * pscale
        acc = acc + lora_x
    d = tl.load(dest + rows, mask=mr, other=0).to(tl.int64)
    tl.store(buf + d[:, None] * H + cols[None, :], acc, mask=mr[:, None] & mc[None, :])


@triton.jit
def _wgrad(x, row_tok, wr, y, out, keep, pscale, g_r0, g_r1, g_off, N, r, s, o_sn, o_sj,
           RP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GATHER: tl.constexpr,
           SCALE: tl.constexpr, DROP: tl.constexpr):
    """An adapter weight grad per (expert, segment) group: for the rows i of group pid_g,
    acc[n, j] = sum of X[i, n] * y[i, j] (fp32), X[i] = x[row_tok[i]] (GATHER) or x[i] (through
    the dropout keep[i], DROP), times wr[i] (SCALE), y [rows, RP] fp32; then
    out[g_off[pid_g] + n * o_sn + j * o_sj] = s * acc[n, j] in out's dtype, for j < r. The rows
    are taken in tiles of BM from the group's first row, in order."""
    pid_g = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = tl.load(g_r0 + pid_g)
    r1 = tl.load(g_r1 + pid_g)
    cols = pid_n * BN + tl.arange(0, BN)
    mc = cols < N
    jj = tl.arange(0, RP)
    acc = tl.zeros((BN, RP), dtype=tl.float32)
    for m0 in range(r0, r1, BM):
        rows = m0 + tl.arange(0, BM)
        mr = rows < r1
        if GATHER:
            src = tl.load(row_tok + rows, mask=mr, other=0).to(tl.int64)
        else:
            src = rows.to(tl.int64)
        xt = tl.load(x + src[:, None] * N + cols[None, :], mask=mr[:, None] & mc[None, :],
                     other=0.0)
        if DROP:
            kt = tl.load(keep + rows[:, None].to(tl.int64) * N + cols[None, :],
                         mask=mr[:, None] & mc[None, :], other=0)
            xt = _dropped(xt, kt, pscale)
        xt = xt.to(tl.float32)
        if SCALE:
            xt = xt * tl.load(wr + rows, mask=mr, other=0.0)[:, None]
        yt = tl.load(y + rows[:, None].to(tl.int64) * RP + jj[None, :], mask=mr[:, None], other=0.0)
        acc = tl.dot(tl.trans(xt), yt, acc, input_precision="ieee")
    base = out + tl.load(g_off + pid_g)
    tl.store(base + cols[:, None].to(tl.int64) * o_sn + jj[None, :] * o_sj,
             (s * acc).to(out.dtype.element_ty), mask=mc[:, None] & (jj[None, :] < r))


@triton.jit(do_not_specialize=["n"])
def _sum_parts(part, out, n, P: tl.constexpr, BLOCK: tl.constexpr):
    """out[i] = part[0, i] + part[1, i] + ... + part[P-1, i], in that order (fp32)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    acc = tl.load(part + i, mask=m, other=0.0)
    for p in tl.static_range(1, P):
        acc += tl.load(part + p * n + i, mask=m, other=0.0)
    tl.store(out + i, acc, mask=m)
