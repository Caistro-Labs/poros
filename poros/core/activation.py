"""Which activation a module computes, for the fused kernels (dense SwiGLU and MoE experts)."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["is_silu"]


def is_silu(act: Any) -> bool:
    """Whether ``act`` is SiLU, the activation the fused SwiGLU kernels compute: ``F.silu``,
    ``nn.SiLU``, or transformers' ``SiLUActivation`` (what its ``ACT2FN`` gives for ``silu``;
    ``swish`` gives ``nn.SiLU``). Decided by the callable a module runs, never by its config's
    name for it: a module whose activation was swapped is what it runs."""
    return (act is torch.nn.functional.silu or isinstance(act, torch.nn.SiLU)
            or type(act).__name__ in ("SiLU", "SiLUActivation"))
