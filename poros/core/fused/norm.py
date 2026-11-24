"""RMSNorm in one Triton kernel each way (CUDA, frozen weight), for the fused compute.

Two input precisions, each its own kernels and rounding chain (their reduction trees and warp
counts are part of their bits):

* fp32 (the dense fused layer, ``poros.core.fused``): ``lora.RMSNorm`` over ``kernels``'
  ``_rmsnorm_fwd`` / ``_rmsnorm_bwd`` (saves ``rstd``).
* bf16 / fp16 (``poros.moe`` models), here: transformers' RMSNorm (``x.float()``, ``pow``,
  ``mean``, ``rsqrt``, two multiplies, two casts) is about eight kernel passes forward and more
  backward, at every norm of a layer (input, post-attention, and Qwen3's per-head q/k norms) in
  its forward, its rematerialization and its backward. ``install_rms_norm(model)`` routes the
  T5-style RMSNorms of a MoE model (``weight * (x * rsqrt(mean(x^2) + eps)).to(x.dtype)``)
  through ``rms_norm``: one row per program, the same math in fp32 (the normalized row rounded
  to the input dtype, then times the weight, rounded), its backward the same chain rule in fp32
  for the input only (the weight is frozen). Values differ from transformers' by fp32 rounding
  (another summation order); every Poros loop runs it.

``rms_norm_values(x, w, eps)`` takes either precision to its kernels.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

# T5-style RMSNorms (weight times the normalized input in its dtype) of the MOE_FAMILIES models
# (the dense fused layer's own: poros.core.fused.RMS_NORMS)
CLASSES = ("Qwen3MoeRMSNorm", "Qwen2MoeRMSNorm", "OlmoeRMSNorm", "MixtralRMSNorm",
           "DeepseekV2RMSNorm", "Qwen3RMSNorm", "Qwen2RMSNorm")


@triton.jit
def _fwd(x, w, y, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    xv = tl.load(x + row * D + cols, mask=m, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(xv * xv, axis=0) / D + eps)
    n = (xv * r).to(y.dtype.element_ty).to(tl.float32)
    wv = tl.load(w + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y + row * D + cols, (wv * n).to(y.dtype.element_ty), mask=m)


@triton.jit
def _bwd(dy, x, w, dx, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    xv = tl.load(x + row * D + cols, mask=m, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(xv * xv, axis=0) / D + eps)
    wv = tl.load(w + cols, mask=m, other=0.0).to(tl.float32)
    # d(normalized row in the input dtype) = dy * w, rounded as the forward's product is
    gn = (tl.load(dy + row * D + cols, mask=m, other=0.0).to(tl.float32) * wv)
    gn = gn.to(dx.dtype.element_ty).to(tl.float32)
    dr = tl.sum(gn * xv, axis=0)
    dvar = -0.5 * dr * r * r * r
    tl.store(dx + row * D + cols, (gn * r + dvar * 2.0 * xv / D).to(dx.dtype.element_ty), mask=m)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        D = x.shape[-1]
        xc = x.contiguous()
        y = torch.empty_like(xc)
        block = triton.next_power_of_2(D)
        _fwd[(xc.numel() // D,)](xc, weight, y, D, eps, BLOCK=block,
                                 enable_fp_fusion=True, num_warps=4 if D > 512 else 1)
        ctx.save_for_backward(xc, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):  # type: ignore[override]
        xc, weight = ctx.saved_tensors
        D = xc.shape[-1]
        dyc = dy.contiguous()
        dx = torch.empty_like(xc)
        block = triton.next_power_of_2(D)
        _bwd[(xc.numel() // D,)](dyc, xc, weight, dx, D, ctx.eps, BLOCK=block,
                                 enable_fp_fusion=True, num_warps=4 if D > 512 else 1)
        return dx, None, None


def rms_norm(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """``module``'s forward through the fused kernels where they compute it as specified (CUDA,
    bf16/fp16 input of the weight's dtype, frozen weight), else its own forward."""
    w = module.weight
    if (hidden_states.is_cuda and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and w.dtype == hidden_states.dtype and not w.requires_grad
            and hidden_states.shape[-1] == w.shape[-1]):
        return _RMSNorm.apply(hidden_states, w, float(module.variance_epsilon))
    return type(module).forward(module, hidden_states)


def rms_norm_values(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm of ``x`` by the frozen ``w`` in this precision's kernels: fp32 the dense layer's
    (``lora.RMSNorm``), bf16 / fp16 these."""
    if x.dtype == torch.float32:
        from poros.core.fused.lora import RMSNorm

        return RMSNorm.apply(x, w, eps)
    return _RMSNorm.apply(x, w, eps)


def install_rms_norm(model: torch.nn.Module) -> int:
    """Route ``model``'s T5-style RMSNorms through ``rms_norm``; returns how many."""
    count = 0
    for module in model.modules():
        if type(module).__name__ in CLASSES and hasattr(module, "variance_epsilon") \
                and "forward" not in vars(module):
            module.forward = rms_norm.__get__(module)
            count += 1
    return count


def uninstall_rms_norm(model: torch.nn.Module) -> int:
    """Undo ``install_rms_norm``; returns how many it restored."""
    count = 0
    for module in model.modules():
        forward = vars(module).get("forward")
        if getattr(forward, "__func__", None) is rms_norm:
            del module.forward
            count += 1
    return count
