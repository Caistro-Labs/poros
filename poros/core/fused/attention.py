"""Causal attention for the fused compute.

Poros's deterministic mode (``poros.core.determinism``) restricts scaled_dot_product_attention to
its math backend, which materializes the [S, S] scores per head in fp32 and runs the mask, the
softmax and its checks as separate passes (Qwen2.5-72B: 64 heads x 512 x 512 per row). Here:

* ``sdpa_bf16`` (where the kernel takes the call): the flash kernel on q/k/v rounded to
  bf16, fp32 accumulation and softmax, the GQA groups taken by the kernel, the output back in
  fp32. A precision change from PEFT's fp32 attention, measured: Qwen2.5-0.5B, 100 updates on
  oasst1, training loss within 0.00043 of PEFT's own modules on average (max 0.0015), against
  0.021 for the same run at another micro-batch split (other roundings and dropout draws), and
  bitwise the same on a rerun (docs/validation/fused/README.md, the Qwen2.5-0.5B row).
* ``sdpa_efficient`` otherwise: the memory-efficient kernel in fp32 (IEEE products, online
  softmax), the same ``softmax(q k^T * scale + causal) v`` in one tiled pass.

Under ``torch.use_deterministic_algorithms(True)`` both backwards are deterministic: two runs give
the same bits (tests/gpu/test_fused_compute.py). Every Poros path runs the same kernel, so the
paths stay bitwise-equal to each other.

Used for the calls they compute exactly as specified: causal, no mask (no padding), no dropout,
fp32 q/k/v, a GQA group's k/v shared by its query heads. Anything else goes to transformers' own
attention function.

``poros.moe`` models (bf16) take ``sdpa_efficient`` through transformers' attention interface:
``install_sdpa(model)`` registers ``attention_forward`` under ``"poros_sdpa"`` (with transformers'
SDPA mask) and points the model's config at it; it computes the calls ``sdpa_usable`` accepts --
causal, no mask, no dropout, CUDA, q/k/v of one dtype, as many keys as queries -- and hands any
other call to transformers' own SDPA function unchanged. Its values differ from the math
backend's by rounding (another summation order), the same in every Poros loop
(``tests/gpu/test_moe_attention.py``).
"""

from __future__ import annotations

from typing import Any

import torch

from poros.core.card import flash_usable  # noqa: F401 (whether the flash kernel takes a call)

# sdpa_bf16 where flash_usable. Not a setting: tests/gpu/test_fused_compute.py turns it off to
# check the fp32 kernel.
_BF16 = True


def usable(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Any, dropout: float) -> bool:
    """Whether this call is one these kernels compute as transformers' SDPA attention would (at
    their precisions)."""
    if mask is not None or dropout or not q.is_cuda or q.dim() != 4:
        return False
    if any(t.dtype != torch.float32 for t in (q, k, v)):
        return False
    B, H, S, D = q.shape
    if k.shape[-1] != D or v.shape[-1] != D or k.shape[2] != S or v.shape[2] != S:
        return False
    return H % k.shape[1] == 0 and k.shape[1] == v.shape[1]


def sdpa_efficient(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   scale: float | None) -> torch.Tensor:
    """Causal attention through the memory-efficient SDPA kernel, k/v repeated over each GQA
    group (transformers' repeat_kv); q [B, H, S, D], k/v [B, Hkv, S, D]; returns [B, S, H, D]
    (the layout transformers' attention functions return). ``scale`` goes to SDPA as given:
    None is SDPA's own ``D ** -0.5``, which an explicit value need not equal bit for bit."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    group = q.shape[1] // k.shape[1]
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return o.transpose(1, 2)


def sdpa_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal attention through the flash SDPA kernel in bf16 (fp32 accumulation): q/k/v rounded
    to bf16, the GQA groups taken by the kernel itself, the output back in fp32; q [B, H, S, D],
    k/v [B, Hkv, S, D]; returns [B, S, H, D] contiguous. Deterministic under
    torch.use_deterministic_algorithms(True)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), is_causal=True,
            scale=scale, enable_gqa=k.shape[1] != q.shape[1])
    return o.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)


NAME = "poros_sdpa"


def sdpa_usable(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: Any,
                dropout: float, kwargs: dict) -> bool:
    """Whether ``attention_forward`` computes this call itself (``poros.moe`` models)."""
    if attention_mask is not None or dropout or not query.is_cuda or query.dim() != 4:
        return False
    if kwargs.get("position_bias") is not None or kwargs.get("output_attentions"):
        return False
    if not (query.dtype == key.dtype == value.dtype):
        return False
    q_len, kv_len = query.shape[2], key.shape[2]
    if q_len < 2 or kv_len != q_len or value.shape[2] != kv_len:
        return False
    return query.shape[1] % key.shape[1] == 0 and key.shape[1] == value.shape[1]


def attention_forward(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                      value: torch.Tensor, attention_mask: Any, dropout: float = 0.0,
                      scaling: float | None = None, is_causal: bool | None = None,
                      **kwargs: Any) -> tuple[torch.Tensor, None]:
    """transformers' attention-function signature; returns ``[B, S, H, D]`` and no weights."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    if not causal or not sdpa_usable(query, key, value, attention_mask, dropout, kwargs):
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, is_causal=is_causal, **kwargs)
    return sdpa_efficient(query, key, value, scaling).contiguous(), None


def _configs(model: torch.nn.Module) -> list:
    configs = [model.config]
    text = getattr(model.config, "get_text_config", None)
    if text is not None and text() is not model.config:
        configs.append(text())
    return configs


def install_sdpa(model: torch.nn.Module) -> bool:
    """Route ``model``'s SDPA attention through ``attention_forward``; False (nothing changed)
    where the model does not use transformers' SDPA attention."""
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    configs = _configs(model)
    if any(getattr(c, "_attn_implementation", None) != "sdpa" for c in configs):
        return False
    AttentionInterface.register(NAME, attention_forward)
    AttentionMaskInterface.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    _set_attn(model, configs, NAME)
    return True


def uninstall_sdpa(model: torch.nn.Module) -> bool:
    """Point a model ``install_sdpa`` routed back at transformers' SDPA attention."""
    configs = _configs(model)
    if not any(getattr(c, "_attn_implementation", None) == NAME for c in configs):
        return False
    _set_attn(model, configs, "sdpa")
    return True


def _set_attn(model: torch.nn.Module, configs: list, name: str) -> None:
    setter = getattr(model, "set_attn_implementation", None)
    if setter is not None:
        setter(name)
    for c in configs:
        if getattr(c, "_attn_implementation", None) != name:
            c._attn_implementation = name
