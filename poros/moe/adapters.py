"""Resident LoRA adapters for fused expert projections.

Mirrors the dense path's ``NativeLoraLinear`` conventions
(``poros.train.adapters.NativeLoraLinear``) — kaiming-uniform A / zero B init,
``scaling = alpha / rank``, dropout on the adapter branch, and
``base + delta`` at the projection output — lifted to the fused expert
layout: one ``[num_experts, rank, in]`` A and ``[num_experts, out, rank]``
B per fused projection (``gate_up_proj``, ``down_proj``). Adapters are
resident and trainable; the router is never adapted.
``poros.moe.export`` merges them into a plain checkpoint.

The factors are bf16, what the fused expert kernels read. Their AdamW is
``poros.train.expert_adamw.ExpertAdamW``: it keeps each weight's bf16 remainder and its moments
in its state, so a bf16 factor loses no update torch's AdamW in fp32 would make.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["ExpertLoraAdapter", "attach_expert_adapters"]


class ExpertLoraAdapter(nn.Module):
    """Trainable LoRA for one MoE layer's fused gate_up_proj and down_proj.

    Per selected expert the forward math is exactly NativeLoraLinear's
    ``base_out + update.to(base_out.dtype) * scaling``, with
    ``update = B(A(dropout(x)))``. All parameters are resident and require
    grad.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A_gate_up = nn.Parameter(
            torch.empty(num_experts, rank, hidden_size, dtype=dtype, device=device)
        )
        self.lora_B_gate_up = nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size, rank, dtype=dtype, device=device)
        )
        self.lora_A_down = nn.Parameter(
            torch.empty(num_experts, rank, intermediate_size, dtype=dtype, device=device)
        )
        self.lora_B_down = nn.Parameter(
            torch.zeros(num_experts, hidden_size, rank, dtype=dtype, device=device)
        )
        # Per-expert 2-D slices so fan_in matches NativeLoraLinear exactly.
        for e in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A_gate_up[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_down[e], a=math.sqrt(5))

    def factors(self, experts: torch.Tensor) -> list[torch.Tensor]:
        """``[A_gate_up, B_gate_up, A_down, B_down]`` rows for ``experts``, one gather each."""
        fused = (self.lora_A_gate_up, self.lora_B_gate_up, self.lora_A_down, self.lora_B_down)
        return [p[experts] for p in fused]

    def delta(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``(x @ Aᵀ @ Bᵀ) * scaling``; leading dims batch experts, as ``torch.matmul`` does."""
        update = self.dropout(x).to(a.dtype) @ a.mT @ b.mT
        return update.to(x.dtype) * self.scaling


def attach_expert_adapters(
    model: nn.Module, *, rank: int, alpha: int, dropout: float = 0.0
) -> int:
    """Attach one ``ExpertLoraAdapter`` to every managed MoE block in ``model``.

    Device follows the managed block's router (the model's compute device).
    Dtype follows the expert store's compute dtype, matching dequantized
    base weights — the same rule NativeLoraLinear uses for a floating base.

    Returns the number of adapters attached; blocks already carrying an
    adapter are left alone.
    """
    from poros.moe.modules import ManagedMoeBlock

    attached = 0
    for module in [*model.modules()]:  # a layer that routes for itself takes it as a new submodule
        if isinstance(module, ManagedMoeBlock) and module.adapter is None:
            # The router's device and dtype, or a whole layer's where the layer routes for itself.
            near = module.gate.weight if module.gate is not None else next(model.parameters())
            module.attach(
                ExpertLoraAdapter(
                    module.num_experts,
                    module.hidden_size,
                    module.intermediate_size,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    dtype=getattr(module.store, "compute_dtype", near.dtype),
                    device=near.device,
                )
            )
            attached += 1
    return attached
