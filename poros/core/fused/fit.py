"""Whether the dense fused kernels fit this GPU at a LoRA rank, decided before training (CUDA).

A kernel's adapter tiles are [BLOCK, R] fp32, R the padded rank (``Lin.rp``), so its shared memory
grows with the rank; a block may take at most 99 KB on sm_86 / sm_89 / sm_120 (RTX 3090 / 4090 /
5090), 96 KB on sm_70, 64 KB on sm_75, 163 KB on sm_80, 227 KB on sm_90
(``poros.core.card.shared_memory_limit``), and past it Triton raises ``OutOfResources`` at the
launch. Each rank has its own tiles (``poros.core.fused_tiles``), the same on every card: up to 128
they fit every supported card (the smallest limit, sm_86 / sm_89 / sm_120's), as the
ahead-of-time compiles in docs/validation/fused/rank_tiles/ show; above, the tiles stay R = 128's
and a card with less shared memory than they take is found here.

``reason(rp, dtype, head_dim)`` launches each kernel the fused compute runs at that rank once, on
a few rows, through ``launch`` (which cuts pipeline stages where that fits: the same values), and
returns why they do not fit, or None. Cached per device. ``poros.core.fused.install`` keeps the
LoRA layers of a rank that does not fit on PEFT's modules, and says so.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

_CACHE: Dict[Tuple[int, int, torch.dtype, Optional[int]], Optional[str]] = {}
_ROWS = 64  # a few row tiles: the kernels' tiles and constexprs as in training, M aside


def reason(rp: int, dtype: torch.dtype, head_dim: Optional[int] = None) -> Optional[str]:
    """Why the fused kernels cannot run at padded rank ``rp`` with frozen GEMMs in ``dtype``
    (and, with ``head_dim``, the rotary q/k kernels at that head size) on the current GPU; None
    where every one runs."""
    dev = torch.cuda.current_device()
    key = (dev, rp, dtype, head_dim)
    if key not in _CACHE:
        from triton.runtime.errors import OutOfResources

        try:
            _probe(rp, dtype, head_dim, torch.device("cuda", dev))
            torch.cuda.synchronize(dev)
            _CACHE[key] = None
        except OutOfResources as exc:
            props = torch.cuda.get_device_properties(dev)
            _CACHE[key] = (f"LoRA rank {rp} (padded): the fused kernels need more shared memory "
                           f"per block than {props.name} has (sm_{props.major}{props.minor}): "
                           f"{exc}")
    return _CACHE[key]


def _probe(rp: int, dtype: torch.dtype, head_dim: Optional[int], device: torch.device) -> None:
    from poros.core.fused import kernels as K

    t = K.tiles(rp)  # the rank's own tiles, as in training
    M = _ROWS
    fbm, fbn, fw = t.out_fwd
    N = 2 * head_dim if head_dim else max(fbn, t.swiglu_fwd[1])
    f32 = torch.float32

    def z(*shape: int, dt: torch.dtype = f32) -> torch.Tensor:
        return torch.zeros(*shape, dtype=dt, device=device)

    base, a, b, out = z(M, N, dt=dtype), z(M, rp), z(N, rp), z(M, N)
    for residual in (False, True):
        K.launch(K._lora_out_fwd, (K.cdiv(M, fbm), K.cdiv(N, fbn)),
                 base, a, b, out, out, M, N, 1.0, R=rp, BLOCK_M=fbm, BLOCK_N=fbn,
                 RESIDUAL=residual, enable_fp_fusion=True, num_warps=fw)
    bm, bn, bw, bs = t.out_bwd
    nt, ns, rows = K.splits(M, N, bm, bn)
    K.launch(K._lora_out_bwd, (nt, ns),
             out, a, b, z(M, N, dt=dtype), z(nt, M, rp), z(ns, N, rp), M, N, rows, R=rp,
             BLOCK_M=bm, BLOCK_N=bn, enable_fp_fusion=True, num_warps=bw, num_stages=bs)
    bm, bn, sw = t.swiglu_fwd
    grid = (K.cdiv(M, bm), K.cdiv(N, bn))
    hb = z(M, N, dt=dtype)
    K.launch(K._swiglu_fwd, grid, base, base, a, a, b, b, hb, out, out, z(rp, N), M, N, 1.0,
             R=rp, BLOCK_M=bm, BLOCK_N=bn, WRITE_HF=True, DOWN_A=False,
             enable_fp_fusion=True, num_warps=sw)
    K.launch(K._swiglu_fwd, grid, base, base, a, a, b, b, hb, hb, z(grid[1], M, rp), z(rp, N),
             M, N, 1.0, R=rp, BLOCK_M=bm, BLOCK_N=bn, WRITE_HF=False, DOWN_A=True,
             enable_fp_fusion=True, num_warps=sw)
    bm, bn, sw, ss = t.swiglu_bwd
    nt, ns, rows = K.splits(M, N, bm, bn)
    for drop in (False, True):
        K.launch(K._swiglu_bwd, (nt, ns),
                 base, base, a, a, b, b, base, a, z(rp, N),
                 z(M, N, dt=torch.bool) if drop else base, z(M, N, dt=dtype), z(M, N, dt=dtype),
                 z(nt, M, rp), z(nt, M, rp), z(ns, N, rp), z(ns, N, rp), z(ns, rp, N),
                 M, N, 1.0, 1.0, rows, R=rp, BLOCK_M=bm, BLOCK_N=bn, HAS_DROP=drop,
                 enable_fp_fusion=True, num_warps=sw, num_stages=ss)
    if head_dim:
        cos = z(1, M, head_dim)
        rbm, rw = t.rope_fwd
        K.launch(K._lora_rope_fwd, (K.cdiv(M, rbm), N // head_dim),
                 base, a, b, out, cos, cos, M, N, M, 1, 1.0, R=rp, BLOCK_M=rbm,
                 HALF=head_dim // 2, enable_fp_fusion=True, num_warps=rw)
        rbm, rw, rs = t.rope_bwd
        nt = N // head_dim
        _, ns, rows = K.splits(M, N, rbm, head_dim)
        K.launch(K._lora_rope_bwd, (nt, ns),
                 out, a, b, z(M, N, dt=dtype), z(nt, M, rp), z(ns, N, rp), cos, cos, M, N, M,
                 1, rows, R=rp, BLOCK_M=rbm, HALF=head_dim // 2, enable_fp_fusion=True,
                 num_warps=rw, num_stages=rs)
