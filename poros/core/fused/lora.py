"""Autograd Functions of the fused compute: LoRA'd NF4 linears, and a SwiGLU MLP made of three.

The math is PEFT's LoRA over a bitsandbytes ``Linear4bit``, with the same precisions:

* the frozen GEMM runs bitsandbytes' own ``Linear4bit`` on the input cast to its compute dtype
  (bf16) -- the same call PEFT's forward makes -- and its input grad is bitsandbytes'
  ``MatMul4Bit.backward`` recipe (``grad @ dequantize_4bit(W)``) on the grad cast to that dtype;
* the adapter path stays fp32: the dropped input times ``A`` and ``B``, scaled, added to the
  frozen output in fp32; the dropouts are ``native_dropout``, the kernel ``nn.Dropout`` runs on
  CUDA, drawn in PEFT's order (each wrapped linear once, gate before up before down);
* SwiGLU is ``silu(gate) * up`` in fp32.

What changes is where values live: the fp32 adapter sum is formed inside one kernel instead of
PEFT's cast / clone / matmul / scale / add sequence, the frozen outputs are kept in bf16 for the
backward (fp32 values are rebuilt from them in-kernel), and an input shared by several linears is
cast once. Sums are therefore taken in another order than PEFT's: values differ from the stock
modules in the last bits, while every Poros path (streamed, resident, sequential, block-major)
runs these same kernels and stays bitwise-equal to the others.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from poros.core.fused_tiles import pad
from poros.core.nf4_cache import dequantized_weight

_aten = torch.ops.aten


class Lin:
    """One LoRA'd ``Linear4bit``'s parts, read off a PEFT LoRA layer for one call."""

    __slots__ = ("mod", "base", "adapter", "A", "B", "s", "p", "dropout", "r", "rp", "N", "K")

    def __init__(self, mod: Any, adapter: str) -> None:
        self.mod = mod
        self.base = mod.base_layer
        self.adapter = adapter
        self.A = mod.lora_A[adapter].weight  # [r, K]
        self.B = mod.lora_B[adapter].weight  # [N, r]
        self.s = float(mod.scaling[adapter])
        self.dropout = mod.lora_dropout[adapter]
        self.p = float(getattr(self.dropout, "p", 0.0)) if isinstance(
            self.dropout, torch.nn.Dropout) else 0.0
        self.r = int(self.A.shape[0])
        self.rp = pad(self.r)  # as the kernels take it (poros.core.fused_tiles.pad)
        self.N = int(self.B.shape[0])
        self.K = int(self.A.shape[1])

    @property
    def drops(self) -> bool:
        return self.p > 0.0 and self.dropout.training


def _pad_a(A: torch.Tensor, rp: int) -> torch.Tensor:
    """``A`` [r, K] as [rp, K], zero rows appended (exact)."""
    if A.shape[0] == rp:
        return A.contiguous()
    return torch.cat([A, A.new_zeros(rp - A.shape[0], A.shape[1])], 0)


def _pad_b(B: torch.Tensor, rp: int) -> torch.Tensor:
    """``B`` [N, r] as [N, rp], zero columns appended (exact)."""
    if B.shape[1] == rp:
        return B.contiguous()
    return torch.cat([B, B.new_zeros(B.shape[0], rp - B.shape[1])], 1).contiguous()


def _drop(x: torch.Tensor, lin: Lin) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PEFT's ``lora_dropout(x)``, and its mask for the backward. ``nn.Dropout`` in training on
    CUDA is ``native_dropout`` (the same draw); the module is still called, with that as its
    forward, so its hooks see the call as they would see PEFT's."""
    if not lin.drops:
        return x, None
    mod = lin.dropout
    masks: List[torch.Tensor] = []

    def forward(inp: torch.Tensor) -> torch.Tensor:
        out, mask = _aten.native_dropout(inp, mod.p, True)
        masks.append(mask)
        return out

    mod.forward = forward
    try:
        out = mod(x)
    finally:
        del mod.forward
    return out, masks[0]


def _undrop(g: torch.Tensor, mask: Optional[torch.Tensor], lin: Lin) -> torch.Tensor:
    if mask is None:
        return g
    return _aten.native_dropout_backward(g, mask, 1.0 / (1.0 - lin.p))


def _frozen(lin: Lin, xb: torch.Tensor) -> torch.Tensor:
    """The frozen GEMM: bitsandbytes' ``Linear4bit.forward`` on an input already in its compute
    dtype (PEFT calls it on the fp32 input, which it casts to the same bf16 first)."""
    return lin.base(xb)


def _frozen_grad(lin: Lin, gb: torch.Tensor) -> torch.Tensor:
    """``MatMul4Bit.backward``'s input grad: ``gb @ dequantize_4bit(W)`` (via the block visit's
    dequantization cache when one is active)."""
    w = lin.base.weight
    return torch.matmul(gb, dequantized_weight(w, w.quant_state, gb.dtype))


def _a_product(xd: torch.Tensor, A: torch.Tensor, rp: int) -> torch.Tensor:
    """``xd @ A^T`` [M, rp], fp32 (the padded columns are zero)."""
    return xd @ _pad_a(A, rp).t()


def _out_fwd(K: Any, yb: torch.Tensor, a: torch.Tensor, Bp: torch.Tensor, s: float,
             res: Optional[torch.Tensor] = None) -> torch.Tensor:
    """float(yb) + s * a @ Bp^T, fp32 [M, N]; with ``res`` ([M, N] fp32), res + that."""
    M, N = yb.shape
    bm, bn, warps = K.tiles(a.shape[1]).out_fwd
    out = torch.empty(M, N, dtype=torch.float32, device=yb.device)
    K.launch(K._lora_out_fwd, (K.cdiv(M, bm), K.cdiv(N, bn)),
        yb, a, Bp, out, res if res is not None else out, M, N, s, R=a.shape[1], BLOCK_M=bm,
        BLOCK_N=bn, RESIDUAL=res is not None, enable_fp_fusion=True, num_warps=warps)
    return out


def _pad_cols(t: torch.Tensor, rp: int) -> torch.Tensor:
    """``t`` [M, r] as [M, rp], zero columns appended."""
    if t.shape[1] == rp:
        return t
    return torch.cat([t, t.new_zeros(t.shape[0], rp - t.shape[1])], 1)


def _grad_rows(t: torch.Tensor, lo: int, r: int, rp: int) -> torch.Tensor:
    """``t[lo:lo + r]``, a LoRA A grad's rows of its padded-rank stack; copied when the rank is
    padded (rp > r), since as ``param.grad`` a view keeps the stack's padding rows alive until the
    grads are cleared. Unpadded, a view: it keeps the whole stack alive, which is nothing but
    live grads where every linear of the group is unpadded and has a grad."""
    rows = t[lo:lo + r]
    return rows.clone() if rp != r else rows


def _residual_2d(res: Optional[torch.Tensor], M: int) -> Optional[torch.Tensor]:
    if res is None:
        return None
    if res.dtype != torch.float32:
        raise RuntimeError("fused residual add needs an fp32 residual")
    return res.reshape(M, -1).contiguous()


def _out_bwd(K: Any, g: torch.Tensor, a: torch.Tensor, Bp: torch.Tensor, s: float,
             dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(g in ``dtype``, s * g @ B  [M, rp], s * g^T @ a  [N, rp]) for the grad g [M, N]."""
    M, N = g.shape
    rp = a.shape[1]
    t = K.tiles(rp)
    bm, bn, warps, stages = t.out_bwd
    gb = torch.empty(M, N, dtype=dtype, device=g.device)
    if not t.chunk:
        nt, ns, rows = K.splits(M, N, bm, bn)
        da = torch.empty(nt, M, rp, dtype=torch.float32, device=g.device)
        db = torch.empty(ns, N, rp, dtype=torch.float32, device=g.device)
        K.launch(K._lora_out_bwd, (nt, ns),
            g, a, Bp, gb, da, db, M, N, rows, R=rp, BLOCK_M=bm, BLOCK_N=bn,
            enable_fp_fusion=True, num_warps=warps, num_stages=stages)
        return gb, da.sum(0).mul_(s), db.sum(0).mul_(s)
    da_rows = torch.empty(M, rp, dtype=torch.float32, device=g.device)
    db_sum: Optional[torch.Tensor] = None
    step = K.row_chunk(t, M, K.cdiv(N, bn), rp, bm)
    for m0 in range(0, M, step):  # whole row blocks a launch, the partials summed in row order
        m1 = min(M, m0 + step)
        nt, ns, rows = K.splits(m1 - m0, N, bm, bn)
        da = torch.empty(nt, m1 - m0, rp, dtype=torch.float32, device=g.device)
        db = torch.empty(ns, N, rp, dtype=torch.float32, device=g.device)
        K.launch(K._lora_out_bwd, (nt, ns),
            g[m0:m1], a[m0:m1], Bp, gb[m0:m1], da, db, m1 - m0, N, rows, R=rp, BLOCK_M=bm,
            BLOCK_N=bn, enable_fp_fusion=True, num_warps=warps, num_stages=stages)
        torch.sum(da, 0, out=da_rows[m0:m1])
        db_sum = db.sum(0) if db_sum is None else db_sum.add_(db.sum(0))
        del da, db
    return gb, da_rows.mul_(s), db_sum.mul_(s)


def kernels() -> Any:
    from poros.core.fused import kernels as K

    return K


class Rope:
    """The rotary embedding a q/k group's outputs take (transformers' apply_rotary_pos_emb):
    ``cos``/``sin`` [Bc, S, D] fp32 (Bc 1 or the batch), applied to the first ``n`` outputs."""

    __slots__ = ("cos", "sin", "S", "BC", "D", "n")

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor, n: int) -> None:
        self.cos, self.sin = cos.contiguous(), sin.contiguous()
        self.BC, self.S, self.D = (int(d) for d in cos.shape)
        self.n = n


def _stacked_a(lins: List[Lin]) -> torch.Tensor:
    """The linears' LoRA A weights, each padded to its rp rows, stacked ([sum rp, K])."""
    return torch.cat([_pad_a(lin.A, lin.rp) for lin in lins], 0)


def _grad_in(K: Any, frozen: List[torch.Tensor], da: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """da @ A + sum(float(f) for f in frozen), fp32 [M, Kin] (no dropout): the LoRA input grads of
    the linears that read one input in one GEMM ([M, sum rp] x [sum rp, Kin]; cuBLAS, fp32), then
    their frozen GEMMs' bf16 input grads added in one pass."""
    out = da @ A
    M, Kin = out.shape
    f = frozen + [frozen[0]] * (3 - len(frozen))
    K.launch(K._add_f32, (K.cdiv(M, 32), K.cdiv(Kin, 256)),
        out, f[0], f[1], f[2], M, Kin, NT=len(frozen), BLOCK_M=32, BLOCK_K=256,
        enable_fp_fusion=True, num_warps=4)
    return out


class LoraLinears(torch.autograd.Function):
    """Up to three LoRA'd NF4 linears over one input (q/k/v; or a single linear).

    ``x`` [..., K]; returns one fp32 output per linear. ``res``: added to the (single) output
    (the decoder layer's residual). ``rope``: a ``Rope`` the first outputs take. ``skip`` (a
    linear's index or -1): that linear's frozen GEMM is not run and its output is zeros -- for a
    rematerialization whose output only roots the gradient (``poros.core.fused.output_unused``);
    its gradients are unchanged, since no gradient reads an output value.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, res: Optional[torch.Tensor], lins: List[Lin], skip: int,
                rope: Optional[Rope], *params: torch.Tensor):
        K = kernels()
        lead, Kin = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, Kin)
        M = x2.shape[0]
        res2 = _residual_2d(res, M)
        if res2 is not None and len(lins) != 1:
            raise RuntimeError("a fused residual add takes one linear")
        dtype = lins[0].base.compute_dtype
        xb = x2.to(dtype)
        drops = any(lin.drops for lin in lins)
        masks: List[Optional[torch.Tensor]] = []
        if drops:  # each linear drops its own copy, in order (PEFT's draws)
            xds, avals = [], []
            for lin in lins:
                xd, mask = _drop(x2, lin)
                xds.append(xd)
                masks.append(mask)
                avals.append(_a_product(xd, lin.A, lin.rp))
        else:
            xds = [x2]
            masks = [None] * len(lins)
            a_all = x2 @ _stacked_a(lins).t()
            avals, c = [], 0
            for lin in lins:
                avals.append(a_all[:, c:c + lin.rp].contiguous())
                c += lin.rp
        outs = []
        for i, lin in enumerate(lins):
            a = avals[i]
            if i == skip:
                out = x2.new_zeros(()).expand(M, lin.N)
            elif rope is not None and i < rope.n:
                out = _rope_fwd(K, _frozen(lin, xb), a, _pad_b(lin.B, lin.rp), lin.s, rope)
            else:
                out = _out_fwd(K, _frozen(lin, xb), a, _pad_b(lin.B, lin.rp), lin.s, res2)
            outs.append(out.view(*lead, lin.N))
        ctx.lins, ctx.lead, ctx.rope, ctx.drops = lins, lead, rope, drops
        ctx.has_mask = [m is not None for m in masks]
        ctx.has_res = res is not None
        ctx.n_x = len(xds)
        ctx.save_for_backward(*xds, *avals, *[m for m in masks if m is not None])
        return tuple(outs)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor):
        K = kernels()
        lins, rope = ctx.lins, ctx.rope
        n = len(lins)
        saved = ctx.saved_tensors
        xds = saved[:ctx.n_x]
        avals = saved[ctx.n_x:ctx.n_x + n]
        it = iter(saved[ctx.n_x + n:])
        masks = [next(it) if h else None for h in ctx.has_mask]
        dtype = lins[0].base.compute_dtype
        frozen: List[torch.Tensor] = []
        das: List[Optional[torch.Tensor]] = []
        dBs: List[Optional[torch.Tensor]] = []
        for i, lin in enumerate(lins):
            g = grads[i]
            if g is None:
                das.append(None)
                dBs.append(None)
                continue
            g2 = g.reshape(-1, lin.N).contiguous()
            Bp = _pad_b(lin.B, lin.rp)
            if rope is not None and i < rope.n:
                gb, da, db = _rope_bwd(K, g2, avals[i], Bp, lin.s, dtype, rope)
            else:
                gb, da, db = _out_bwd(K, g2, avals[i], Bp, lin.s, dtype)
            frozen.append(_frozen_grad(lin, gb))
            del gb
            das.append(da)
            dBs.append(db[:, :lin.r].contiguous())
        dres = grads[0] if ctx.has_res else None
        live = [i for i in range(n) if das[i] is not None]
        if not live:
            return (None, dres, None, None, None, *[None] * (2 * n))
        dAs: List[Optional[torch.Tensor]] = [None] * n
        if not ctx.drops:
            x2 = xds[0]
            da_all = torch.cat([das[i] if das[i] is not None else
                                x2.new_zeros(x2.shape[0], lins[i].rp) for i in range(n)], 1)
            dx = _grad_in(K, frozen, da_all, _stacked_a(lins))
            dA_all = da_all.t() @ x2
            c = 0
            for i, lin in enumerate(lins):
                if das[i] is not None:
                    dAs[i] = _grad_rows(dA_all, c, lin.r, lin.rp)
                c += lin.rp
        else:
            dx = None
            for i in live:
                lin, da = lins[i], das[i][:, :lins[i].r]
                dAs[i] = da.t() @ xds[i]
                dxl = _undrop(da @ lin.A, masks[i], lin)
                dx = dxl if dx is None else dx.add_(dxl)
            M, Kin = dx.shape
            f = frozen + [frozen[0]] * (3 - len(frozen))
            K.launch(K._add_f32, (K.cdiv(M, 32), K.cdiv(Kin, 256)),
                dx, f[0], f[1], f[2], M, Kin, NT=len(frozen), BLOCK_M=32, BLOCK_K=256,
                enable_fp_fusion=True, num_warps=4)
        dparams: List[Optional[torch.Tensor]] = []
        for i in range(n):
            dparams += [dAs[i], dBs[i]]
        return (dx.view(*ctx.lead, dx.shape[-1]), dres, None, None, None, *dparams)


def _rope_fwd(K: Any, yb: torch.Tensor, a: torch.Tensor, Bp: torch.Tensor, s: float,
              rope: Rope) -> torch.Tensor:
    """``_out_fwd`` for an output that takes the rotary embedding, fp32 [M, N]."""
    M, N = yb.shape
    rp = a.shape[1]
    bm, warps = K.tiles(rp).rope_fwd
    out = torch.empty(M, N, dtype=torch.float32, device=yb.device)
    K.launch(K._lora_rope_fwd, (K.cdiv(M, bm), N // rope.D),
        yb, a, Bp, out, rope.cos, rope.sin, M, N, rope.S, rope.BC, s, R=rp, BLOCK_M=bm,
        HALF=rope.D // 2, enable_fp_fusion=True, num_warps=warps)
    return out


def _rope_bwd(K: Any, g: torch.Tensor, a: torch.Tensor, Bp: torch.Tensor, s: float,
              dtype: torch.dtype, rope: Rope) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_out_bwd`` for an output that took the rotary embedding."""
    M, N = g.shape
    rp = a.shape[1]
    nt = N // rope.D
    bm, warps, stages = K.tiles(rp).rope_bwd
    _, ns, rows = K.splits(M, N, bm, rope.D)
    gb = torch.empty(M, N, dtype=dtype, device=g.device)
    da = torch.empty(nt, M, rp, dtype=torch.float32, device=g.device)
    db = torch.empty(ns, N, rp, dtype=torch.float32, device=g.device)
    K.launch(K._lora_rope_bwd, (nt, ns),
        g, a, Bp, gb, da, db, rope.cos, rope.sin, M, N, rope.S, rope.BC, rows, R=rp,
        BLOCK_M=bm, HALF=rope.D // 2, enable_fp_fusion=True, num_warps=warps,
        num_stages=stages)
    return gb, da.sum(0).mul_(s), db.sum(0).mul_(s)


class SwiGLUMLP(torch.autograd.Function):
    """``down(silu(gate(x)) * up(x))`` over three LoRA'd NF4 linears.

    ``skip_down``: the down linear's frozen GEMM is not run and the output is zeros (see
    ``LoraLinears``' ``skip``); the gradients are unchanged.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, res: Optional[torch.Tensor], gate: Lin, up: Lin,
                down: Lin, skip_down: bool, *params: torch.Tensor):
        K = kernels()
        lead, H = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, H)
        M = x2.shape[0]
        res2 = _residual_2d(res, M)
        Ni = gate.N
        dtype = gate.base.compute_dtype
        xb = x2.to(dtype)
        if gate.drops or up.drops:  # gate's draw, then up's (PEFT's order)
            xg, mg = _drop(x2, gate)
            ag = _a_product(xg, gate.A, gate.rp)
            xu, mu = _drop(x2, up)
            au = _a_product(xu, up.A, up.rp)
        else:
            xg = xu = x2
            mg = mu = None
            a_all = x2 @ _stacked_a([gate, up]).t()
            ag, au = a_all[:, :gate.rp].contiguous(), a_all[:, gate.rp:].contiguous()
        bg = _frozen(gate, xb)
        bu = _frozen(up, xb)
        del xb
        hb = torch.empty(M, Ni, dtype=dtype, device=x.device)
        rp = gate.rp
        if rp != up.rp or rp != down.rp:
            raise RuntimeError("fused SwiGLU needs one LoRA rank on gate/up/down")
        Bg, Bu, Ad = _pad_b(gate.B, rp), _pad_b(up.B, rp), _pad_a(down.A, rp)
        if down.drops:
            hf = swiglu_fwd(K, bg, bu, ag, au, Bg, Bu, Ad, gate.s, hb, None)
            hd, md = _drop(hf, down)
            ad = hd @ Ad.t()
            del hf, hd
        else:
            md = None
            ad = swiglu_fwd(K, bg, bu, ag, au, Bg, Bu, Ad, gate.s, hb, Ad)
        if skip_down:
            out = x2.new_zeros(()).expand(M, down.N)
        else:
            out = _out_fwd(K, _frozen(down, hb), ad, _pad_b(down.B, rp), down.s, res2)
        del hb
        ctx.has_res = res is not None
        ctx.lins = (gate, up, down)
        ctx.lead = lead
        ctx.same_x = xg is xu
        masks = [m for m in (mg, mu, md) if m is not None]
        ctx.has_mask = (mg is not None, mu is not None, md is not None)
        ctx.save_for_backward(xg, xu if xu is not xg else xg.new_empty(0), bg, bu, ag, au, ad,
                              *masks)
        return out.view(*lead, down.N)

    @staticmethod
    def backward(ctx: Any, gout: torch.Tensor):
        K = kernels()
        gate, up, down = ctx.lins
        xg, xu, bg, bu, ag, au, ad, *masks = ctx.saved_tensors
        if ctx.same_x:
            xu = xg
        it = iter(masks)
        mg, mu, md = (next(it) if h else None for h in ctx.has_mask)
        dtype = gate.base.compute_dtype
        rp = gate.rp
        M, Ni = bg.shape
        g2 = gout.reshape(-1, down.N).contiguous()
        gbd, dad, dbd = _out_bwd(K, g2, ad, _pad_b(down.B, rp), down.s, dtype)
        dzb = _frozen_grad(down, gbd)
        del gbd
        pscale = 1.0 / (1.0 - down.p) if md is not None else 1.0
        dgb, dub, dag, dau, dBg, dBu, dAd = swiglu_bwd(
            K, bg, bu, ag, au, _pad_b(gate.B, rp), _pad_b(up.B, rp), dzb, dad,
            _pad_a(down.A, rp), md, gate.s, pscale, dtype)
        del dzb
        dag = dag.mul_(gate.s)[:, :gate.r]
        dau = dau.mul_(up.s)[:, :up.r]
        dBg = dBg.mul_(gate.s)[:, :gate.r].contiguous()
        dBu = dBu.mul_(up.s)[:, :up.r].contiguous()
        dAd = _grad_rows(dAd, 0, down.r, rp)
        fg = _frozen_grad(gate, dgb)
        del dgb
        fu = _frozen_grad(up, dub)
        del dub
        if mg is None and mu is None:
            da_all = torch.cat([_pad_cols(dag, rp), _pad_cols(dau, rp)], 1)
            dx = _grad_in(K, [fg, fu], da_all, _stacked_a([gate, up]))
            dA_all = da_all.t() @ xg
            dAg, dAu = _grad_rows(dA_all, 0, gate.r, rp), _grad_rows(dA_all, rp, up.r, rp)
        else:
            dx = _undrop(dag @ gate.A, mg, gate)
            dx = dx.add_(_undrop(dau @ up.A, mu, up))
            H = dx.shape[1]
            K.launch(K._add_f32, (K.cdiv(M, 32), K.cdiv(H, 256)),
                dx, fg, fu, fu, M, H, NT=2, BLOCK_M=32, BLOCK_K=256, enable_fp_fusion=True,
                num_warps=4)
            dAg = dag.t() @ xg
            dAu = dau.t() @ xu
        H = dx.shape[1]
        dBd = dbd[:, :down.r].contiguous()
        return (dx.view(*ctx.lead, H), gout if ctx.has_res else None, None, None, None, None,
                dAg, dBg, dAu, dBu, dAd, dBd)


def swiglu_fwd(K: Any, bg: torch.Tensor, bu: torch.Tensor, ag: torch.Tensor, au: torch.Tensor,
               Bg: torch.Tensor, Bu: torch.Tensor, Ad: torch.Tensor, s: float, hb: torch.Tensor,
               down_a: Optional[torch.Tensor]) -> torch.Tensor:
    """The SwiGLU over LoRA'd gate/up from their frozen outputs ``bg``/``bu`` [M, N] (compute
    dtype) and A products ``ag``/``au`` [M, R] (fp32): h into ``hb`` (compute dtype), and either
    h in fp32 ([M, N], ``down_a`` None: the down linear's dropout comes next) or h @ ``down_a``^T,
    the down LoRA's A product ([M, R] fp32), summed over the column tiles' partials."""
    M, N = bg.shape
    rp = ag.shape[1]
    t = K.tiles(rp)
    bm, bn, warps = t.swiglu_fwd
    grid = (K.cdiv(M, bm), K.cdiv(N, bn))
    if down_a is None:
        hf = torch.empty(M, N, dtype=torch.float32, device=bg.device)
        K.launch(K._swiglu_fwd, grid,
            bg, bu, ag, au, Bg, Bu, hb, hf, hf, Ad, M, N, s, R=rp, BLOCK_M=bm,
            BLOCK_N=bn, WRITE_HF=True, DOWN_A=False, enable_fp_fusion=True, num_warps=warps)
        return hf
    if not t.chunk:
        part = torch.empty(grid[1], M, rp, dtype=torch.float32, device=bg.device)
        K.launch(K._swiglu_fwd, grid,
            bg, bu, ag, au, Bg, Bu, hb, hb, part, down_a, M, N, s, R=rp, BLOCK_M=bm,
            BLOCK_N=bn, WRITE_HF=False, DOWN_A=True, enable_fp_fusion=True, num_warps=warps)
        return part.sum(0)
    # above rank 32 the rows in chunks: [column tiles, chunk, R] partials (K.row_chunk)
    ad = torch.empty(M, rp, dtype=torch.float32, device=bg.device)
    step = K.row_chunk(t, M, grid[1], rp, bm)
    for m0 in range(0, M, step):
        m1 = min(M, m0 + step)
        part = torch.empty(grid[1], m1 - m0, rp, dtype=torch.float32, device=bg.device)
        K.launch(K._swiglu_fwd, (K.cdiv(m1 - m0, bm), grid[1]),
            bg[m0:m1], bu[m0:m1], ag[m0:m1], au[m0:m1], Bg, Bu, hb[m0:m1], hb[m0:m1],
            part, down_a, m1 - m0, N, s, R=rp, BLOCK_M=bm, BLOCK_N=bn, WRITE_HF=False,
            DOWN_A=True, enable_fp_fusion=True, num_warps=warps)
        torch.sum(part, 0, out=ad[m0:m1])
        del part
    return ad


def swiglu_bwd(K: Any, bg: torch.Tensor, bu: torch.Tensor, ag: torch.Tensor, au: torch.Tensor,
               Bg: torch.Tensor, Bu: torch.Tensor, dzb: torch.Tensor, dad: torch.Tensor,
               Ad: torch.Tensor, md: Optional[torch.Tensor], s: float, pscale: float,
               dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    """``swiglu_fwd``'s backward with the down linear's input side: from the down GEMM's input
    grad ``dzb`` [M, N] and the down LoRA's A-product grad ``dad`` [M, R] (through the down
    dropout's mask ``md``, scaled ``pscale``), (dg, du in ``dtype`` [M, N]; dg @ Bg and du @ Bu
    [M, R]; dg^T @ ag and du^T @ au [N, R]; dad^T @ dropout(h) [R, N]), all unscaled by ``s``."""
    M, N = bg.shape
    rp = ag.shape[1]
    t = K.tiles(rp)
    bm, bn, warps, stages = t.swiglu_bwd
    dev = bg.device
    f32 = torch.float32
    dgb = torch.empty(M, N, dtype=dtype, device=dev)
    dub = torch.empty(M, N, dtype=dtype, device=dev)
    drop = md is not None

    def launch(m0: int, m1: int) -> Tuple[torch.Tensor, ...]:
        mc = m1 - m0
        nt, ns, rows = K.splits(mc, N, bm, bn)
        parts = (torch.empty(nt, mc, rp, dtype=f32, device=dev),
                 torch.empty(nt, mc, rp, dtype=f32, device=dev),
                 torch.empty(ns, N, rp, dtype=f32, device=dev),
                 torch.empty(ns, N, rp, dtype=f32, device=dev),
                 torch.empty(ns, rp, N, dtype=f32, device=dev))
        r = slice(m0, m1)
        K.launch(K._swiglu_bwd, (nt, ns),
            bg[r], bu[r], ag[r], au[r], Bg, Bu, dzb[r], dad[r], Ad,
            md[r] if drop else dzb[r], dgb[r], dub[r], *parts, mc, N, s, pscale, rows, R=rp,
            BLOCK_M=bm, BLOCK_N=bn, HAS_DROP=drop, enable_fp_fusion=True, num_warps=warps,
            num_stages=stages)
        return parts

    if not t.chunk:
        dag, dau, dbg, dbu, dwa = launch(0, M)
        return dgb, dub, dag.sum(0), dau.sum(0), dbg.sum(0), dbu.sum(0), dwa.sum(0)
    # above rank 32 the rows in chunks (K.row_chunk); the row sums added in row order
    dag = torch.empty(M, rp, dtype=f32, device=dev)
    dau = torch.empty(M, rp, dtype=f32, device=dev)
    sums: List[torch.Tensor] = []
    step = K.row_chunk(t, M, K.cdiv(N, bn), rp, bm)
    for m0 in range(0, M, step):
        m1 = min(M, m0 + step)
        pg, pu, pbg, pbu, pwa = launch(m0, m1)
        torch.sum(pg, 0, out=dag[m0:m1])
        torch.sum(pu, 0, out=dau[m0:m1])
        part = [pbg.sum(0), pbu.sum(0), pwa.sum(0)]
        sums = part if not sums else [x.add_(y) for x, y in zip(sums, part)]
        del pg, pu, pbg, pbu, pwa, part
    return (dgb, dub, dag, dau, *sums)


class RMSNorm(torch.autograd.Function):
    """HF's RMSNorm on an fp32 input with a frozen weight: ``w * (x * rsqrt(mean(x^2) + eps))``,
    one kernel per pass (the row's mean is a tree sum instead of PyTorch's)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        K = kernels()
        H = x.shape[-1]
        x2 = x.reshape(-1, H).contiguous()
        M = x2.shape[0]
        y = torch.empty_like(x2)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        block = 1 << (H - 1).bit_length()
        K.launch(K._rmsnorm_fwd, (M,),
            x2, w, y, rstd, H, eps, BLOCK=block, enable_fp_fusion=True,
            num_warps=_norm_warps(block))
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):
        K = kernels()
        x2, w, rstd = ctx.saved_tensors
        M, H = x2.shape
        dy2 = dy.reshape(M, H).contiguous()
        dx = torch.empty_like(x2)
        block = 1 << (H - 1).bit_length()
        K.launch(K._rmsnorm_bwd, (M,),
            dy2, x2, w, rstd, dx, H, BLOCK=block, enable_fp_fusion=True,
            num_warps=_norm_warps(block))
        return dx.view(ctx.shape), None, None


def _norm_warps(block: int) -> int:
    return 8 if block >= 4096 else 4
