"""Inkling 2 Mini's text decoder, carried by Poros (text only; no vision tower, no MTP head).

Written from the official implementation in Hugging Face transformers 5.24.0,
``models/inkling2_loop/modeling_inkling2_loop.py`` (Copyright 2026 the HuggingFace Team,
Apache License 2.0; see NOTICE). The math is that implementation's; parameter names are the
same, so a state dict moves between the two unchanged. What differs is the structure Poros
needs to stream it:

- The decoder runs as a schedule of layer *visits*: layers before the loop once, the loop
  layers ``loop_num_passes`` times with shared weights, the layers after once
  (``Inkling2LoopTextModel.forward`` of the official file). Each run of visits goes through
  ``_run_visits``, the one place the streamed path (``poros.moe.inkling2.visits``) replaces.
- The state between visits is the partial residual ``hidden`` plus the list of block states the
  block attention residuals read (a Python list of ``[batch, seq, hidden]`` tensors, stacked
  where a layer reads them), plus, in a loop pass after the first, the previous pass's
  message sources. A visit reads all of it, so its backward gives a gradient to all of it.
- Training only: no KV or convolution cache, eager attention (the official one is eager too:
  every layer adds a relative position bias and a per-token sink).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from poros.moe.inkling2.config import (
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
    text_config_of,
)

__all__ = [
    "Inkling2LoopForCausalLM",
    "Inkling2LoopTextModel",
    "VisitContext",
    "FP32_MODULES",
]

#: Modules the official model keeps in fp32 whatever the model's dtype
#: (``Inkling2LoopPreTrainedModel._keep_in_fp32_modules_strict``).
FP32_MODULES = (
    "e_score_correction_bias",
    "mlp.gate.weight",
    "post_attention_convolution",
    "post_mlp_convolution",
    "sconv.conv1d",
    "attention_input",
    "mlp_input",
    "final_input",
    "message_input",
    "message_gate_proj",
    "forget_gate_proj",
)


# -- small pieces -----------------------------------------------------------------------------


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * x).to(dtype)


class Softcap(nn.Module):
    """``tanh(x / c) * c``; with no cap (None, a variant without one) the identity."""

    def __init__(self, softcap: Optional[float]) -> None:
        super().__init__()
        self.softcap = softcap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.softcap is None:
            return x
        return torch.tanh(x / self.softcap) * self.softcap


def _residual_attention(query: torch.Tensor, stacked: torch.Tensor) -> torch.Tensor:
    """softmax(q k^T / sqrt(d)) v per position, with the normalized states as keys and values:
    query ``[..., q, d]``, stacked ``[..., n, d]`` -> ``[..., q, d]`` (fp32)."""
    scores = torch.matmul(query, stacked.transpose(-1, -2)) * stacked.shape[-1] ** -0.5
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, stacked)


class BlockAttentionResidual(nn.Module):
    """A layer's input from [block states, partial]: learned queries over the RMS-normalized
    stack (one query, or three for a global layer's q, k and v inputs)."""

    def __init__(self, config: Inkling2LoopTextConfig, num_queries: int = 1) -> None:
        super().__init__()
        shape = (config.hidden_size,) if num_queries == 1 else (num_queries, config.hidden_size)
        self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
        self.norm = UnweightedRMSNorm()

    def forward(
        self, block_states: torch.Tensor, partial: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float()
        stacked = self.norm(stacked)
        query = self.query.float().view(-1, stacked.shape[-1])
        query = query.expand(*stacked.shape[:-2], -1, -1)
        return _residual_attention(query, stacked).unbind(-2)


class MessageAttentionResidual(nn.Module):
    """A loop layer's four messages (q, k, v, mlp) from the previous pass's block states."""

    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__()
        loop = config.loop_layer_index_end - config.loop_layer_index_start
        self.query = nn.Parameter(torch.zeros(loop, 4, config.hidden_size, dtype=torch.float32))
        self.norm = UnweightedRMSNorm()

    def forward(
        self, block_states: torch.Tensor, partial: torch.Tensor, loop_layer: int
    ) -> torch.Tensor:
        stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float()
        stacked = self.norm(stacked)
        query = self.query[loop_layer].float().expand(*stacked.shape[:-2], -1, -1)
        return _residual_attention(query, stacked)


class MessageGate(nn.Module):
    """``(1 - forget) * hidden + gain * message``, gates from a rank-``loop_gate_rank`` mix."""

    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__()
        self.norm = UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.down_proj = nn.Linear(2 * config.hidden_size, config.loop_gate_rank, bias=False)
        self.forget_gate_proj = nn.Linear(
            config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32
        )
        self.message_gate_proj = nn.Linear(
            config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32
        )

    def forward(self, hidden: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        hidden, message = hidden.float(), message.float()
        mixed = torch.cat([self.norm(hidden), self.norm(message)], dim=-1)
        mixed = self.down_proj(mixed.to(self.down_proj.weight.dtype)).float()
        forget = torch.tanh(self.forget_gate_proj(mixed))
        gain = torch.tanh(self.message_gate_proj(mixed))
        return (1 - forget) * hidden + gain * message


class L2Clip(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return (x * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(dtype)


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float() * self.scale.float()
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return (x * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(dtype)


class LogScale(nn.Module):
    """``x * (1 + alpha * log(max((position + 1) / n_floor, 1)))``."""

    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__()
        self.alpha = config.log_scaling_alpha
        self.n_floor = config.log_scaling_n_floor

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        scale = 1.0 + self.alpha * torch.log(
            ((position_ids + 1).float() / self.n_floor).clamp(min=1.0)
        )
        return (x.float() * scale[..., None, None]).to(dtype)


class LogScaledL2Clip(nn.Module):
    def __init__(
        self, config: Inkling2LoopTextConfig, num_heads: int, head_dim: int, threshold: float
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))
        self.l2_clip = L2Clip(threshold)
        self.log_scale = LogScale(config)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = self.l2_clip(x.float() * self.scale.float()).to(dtype)
        return self.log_scale(x, position_ids)


class RelativePositionBias(nn.Module):
    """A per-token mix of learned bias-by-distance profiles, zero beyond ``rel_extent``."""

    def __init__(
        self, config: Inkling2LoopTextConfig, num_heads: int, rel_extent: int, log_scaling: bool
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.rel_extent = rel_extent
        self.linear = nn.Linear(config.hidden_size, num_heads * config.d_rel, bias=False)
        self.norm = (
            LogScaledL2Clip(config, num_heads, config.d_rel, config.rel_l2_clip_threshold)
            if log_scaling
            else ScaledL2Clip(num_heads, config.d_rel, config.rel_l2_clip_threshold)
        )
        self.proj = nn.Parameter(torch.empty(config.d_rel, rel_extent))

    def forward(
        self, hidden: torch.Tensor, distance: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        rel = self.linear(hidden).view(*hidden.shape[:-1], self.num_heads, -1)
        rel = self.norm(rel, position_ids)
        logits = (rel @ self.proj).transpose(1, 2)  # [batch, heads, q, extent]
        index = distance.clamp(0, self.rel_extent - 1).expand(*logits.shape[:2], -1, -1)
        bias = logits.gather(-1, index)
        return bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)


def _pad_masked(x: torch.Tensor, conv_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if conv_mask is None:
        return x
    return (x * conv_mask[:, :, None]).to(x.dtype)


class ShortConvolution(nn.Module):
    """A causal depthwise convolution over the sequence, added to its input and rescaled."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=kernel_size - 1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, conv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        residual = x
        seq = x.shape[1]
        h = _pad_masked(x, conv_mask).transpose(1, 2)
        weight = self.conv1d.weight
        h = F.conv1d(
            h.to(weight.dtype), weight, None, padding=self.kernel_size - 1, groups=h.shape[1]
        )[:, :, :seq].to(x.dtype)
        h = h.transpose(1, 2)
        return ((h + residual) / math.sqrt(self.kernel_size + 1)).to(dtype)


class DynamicShortConvolution(nn.Module):
    """A short causal convolution whose taps are static per channel plus per token, shared within
    each head (the global layers' keys and values)."""

    def __init__(self, channels: int, input_size: int, num_heads: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = nn.Parameter(torch.zeros(kernel_size, channels))
        self.weight_proj = nn.Linear(input_size, num_heads * kernel_size, bias=False)

    def forward(
        self, x: torch.Tensor, dynamic_input: torch.Tensor, conv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        batch, seq, channels = x.shape
        k = self.kernel_size
        dynamic = self.weight_proj(dynamic_input).float().view(batch, seq, self.num_heads, 1, k)
        weights = self.static_weight.float().T.view(self.num_heads, -1, k) + dynamic
        x = x.float()
        residual = x
        context = _pad_masked(x, conv_mask).transpose(1, 2)
        context = F.pad(context, (k - 1, 0))
        # windows[..., lag] is the token ``lag`` steps back
        windows = context.unfold(-1, k, 1).flip(-1).transpose(1, 2)
        windows = windows.reshape(batch, seq, self.num_heads, -1, k)
        h = (windows * weights).sum(-1).reshape(batch, seq, channels)
        return ((h + residual) / math.sqrt(2 * k + 1)).to(dtype)


class QueryProjection(nn.Module):
    def __init__(
        self, config: Inkling2LoopTextConfig, num_heads: int, head_dim: int, log_scaling: bool
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        self.norm = (
            LogScaledL2Clip(config, num_heads, head_dim, config.qk_l2_clip_threshold)
            if log_scaling
            else ScaledL2Clip(num_heads, head_dim, config.qk_l2_clip_threshold)
        )

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        states = self.linear(hidden)
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states, position_ids).transpose(1, 2)


class KeyValueProjection(nn.Module):
    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        num_heads: int,
        head_dim: int,
        is_key: bool,
        dynamic: bool,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        if dynamic:
            self.sconv = DynamicShortConvolution(
                num_heads * head_dim, config.hidden_size, num_heads, config.conv_kernel_size
            )
        else:
            self.sconv = ShortConvolution(num_heads * head_dim, config.conv_kernel_size)
        self.dynamic = dynamic
        self.norm = (
            L2Clip(config.qk_l2_clip_threshold)
            if is_key
            else UnweightedRMSNorm(eps=config.rms_norm_eps)
        )

    def forward(self, hidden: torch.Tensor, conv_mask: Optional[torch.Tensor]) -> torch.Tensor:
        states = self.linear(hidden)
        if self.dynamic:
            states = self.sconv(states, hidden, conv_mask)
        else:
            states = self.sconv(states, conv_mask)
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states).transpose(1, 2)


class GatedOutputProjection(nn.Module):
    def __init__(self, config: Inkling2LoopTextConfig, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        self.linear = nn.Linear(num_heads * head_dim, config.hidden_size, bias=False)

    def forward(self, attn: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(hidden).float())
        attn = (attn.float() * gate).to(attn.dtype)
        return self.linear(attn)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq, dim = x.shape
    return (
        x[:, :, None].expand(batch, heads, n_rep, seq, dim).reshape(batch, heads * n_rep, seq, dim)
    )


class Attention(nn.Module):
    """The sliding or global attention: relative position bias, a per-token sink, gated output."""

    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int) -> None:
        super().__init__()
        sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.is_sliding = sliding
        self.head_dim = config.swa_head_dim if sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if sliding else config.num_attention_heads
        kv_heads = config.swa_num_key_value_heads if sliding else config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // kv_heads
        self.scaling = self.head_dim**-0.5
        rel_extent = config.sliding_window_size if sliding else config.rel_extent
        self.q_proj = QueryProjection(
            config, self.num_heads, self.head_dim, log_scaling=not sliding
        )
        self.k_proj = KeyValueProjection(
            config, kv_heads, self.head_dim, is_key=True, dynamic=not sliding
        )
        self.v_proj = KeyValueProjection(
            config, kv_heads, self.head_dim, is_key=False, dynamic=not sliding
        )
        self.o_proj = GatedOutputProjection(config, self.num_heads, self.head_dim)
        self.sink_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.relative_position_bias = RelativePositionBias(
            config, self.num_heads, rel_extent, log_scaling=not sliding
        )

    def forward(
        self,
        hidden: torch.Tensor,
        key_hidden: torch.Tensor,
        value_hidden: torch.Tensor,
        ctx: "VisitContext",
    ) -> torch.Tensor:
        shape = hidden.shape[:-1]
        position_ids = ctx.position_ids
        q = self.q_proj(hidden, position_ids)
        k = _repeat_kv(self.k_proj(key_hidden, ctx.conv_mask), self.num_key_value_groups)
        v = _repeat_kv(self.v_proj(value_hidden, ctx.conv_mask), self.num_key_value_groups)
        distance = ctx.sliding_distance if self.is_sliding else ctx.full_distance
        mask = ctx.sliding_mask if self.is_sliding else ctx.full_mask
        bias = self.relative_position_bias(hidden, distance, position_ids)
        sinks = self.sink_proj(hidden).transpose(1, 2).unsqueeze(-1)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        scores = scores + bias
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        # the sink is one more logit per query with a zero value: it only absorbs probability
        probs = F.softmax(torch.cat([scores, sinks], dim=-1).float(), dim=-1)[..., :-1].to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous()
        return self.o_proj(out.reshape(*shape, -1), hidden)


class SoftcappedSwiGLU(nn.Module):
    """``softcap(gate) * sigmoid(gate) * softcap(up)``, in fp32: the sigmoid sees the raw gate."""

    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__()
        self.gate_softcap = Softcap(config.mlp_gate_softcap)
        self.up_softcap = Softcap(config.mlp_up_softcap)

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        dtype = gate.dtype
        gate, up = gate.float(), up.float()
        return (self.gate_softcap(gate) * torch.sigmoid(gate) * self.up_softcap(up)).to(dtype)


class Experts(nn.Module):
    """The routed experts as 3-D tensors, transformers' layout (gate rows, then up rows).

    ``act_takes_up``: the activation is called as ``act_fn(gate, up)``, not ``act_fn(gate) * up``
    (poros.moe.modules reads this flag)."""

    act_takes_up = True

    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = SoftcappedSwiGLU(config)

    def forward(
        self, hidden: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros_like(hidden, dtype=torch.float32)
        for expert in torch.unique(top_k_index).tolist():
            slot, token = torch.where(top_k_index.T == expert)
            x = hidden[token]
            gate, up = F.linear(x, self.gate_up_proj[expert]).chunk(2, dim=-1)
            y = F.linear(self.act_fn(gate, up), self.down_proj[expert])
            y = y * top_k_weights[token, slot, None]
            out.index_add_(0, token, y.to(out.dtype))
        return out.to(hidden.dtype)


class TopKRouter(nn.Module):
    """Sigmoid scores; top-k on scores plus a correction bias; normalized; times route_scale."""

    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, dtype=torch.float32)
        )
        self.routed_scaling_factor = config.route_scale
        if config.scoring_func != "sigmoid":
            raise ValueError(f"scoring_func {config.scoring_func!r}: only sigmoid is supported")
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden.float().reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = torch.sigmoid(logits)
        index = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        ).indices
        weights = scores.gather(1, index)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, index


class SparseMoeBlock(nn.Module):
    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__()
        self.experts = Experts(config)
        self.gate = TopKRouter(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = hidden.shape
        flat = hidden.view(-1, dim)
        _, weights, index = self.gate(flat)
        return self.experts(flat, index, weights).reshape(batch, seq, dim)


# -- decoder layers ---------------------------------------------------------------------------


@dataclass
class VisitContext:
    """What every visit of one forward shares: positions, masks and distances (no cache)."""

    position_ids: torch.Tensor
    full_mask: Optional[torch.Tensor]
    sliding_mask: Optional[torch.Tensor]
    full_distance: torch.Tensor
    sliding_distance: torch.Tensor
    conv_mask: Optional[torch.Tensor]


class DecoderLayer(nn.Module):
    """One decoder layer of any of the four official kinds (sliding or global, looped or not).

    ``forward(hidden, block_states, messages, ctx)``: ``block_states`` is ``[batch, seq, n,
    hidden]`` (n may be 0), ``messages`` ``[batch, seq, 4, hidden]`` in a loop pass after the
    first, else None. Returns the next partial residual.
    """

    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_loop_layer = (
            config.loop_layer_index_start <= layer_idx < config.loop_layer_index_end
        )
        queries = 1 if self.is_sliding else 3
        self.attention_input = (
            BlockAttentionResidual(config, num_queries=queries) if layer_idx > 0 else None
        )
        self.mlp_input = BlockAttentionResidual(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoeBlock(config)
        if self.is_sliding:
            self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.q_input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.k_input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.v_input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_convolution = ShortConvolution(
            config.hidden_size, config.conv_kernel_size
        )
        self.post_mlp_convolution = ShortConvolution(config.hidden_size, config.conv_kernel_size)
        if self.is_loop_layer:
            self.q_message_gate = MessageGate(config)
            self.k_message_gate = MessageGate(config)
            self.v_message_gate = MessageGate(config)
            self.mlp_message_gate = MessageGate(config)

    def forward(
        self,
        hidden: torch.Tensor,
        block_states: torch.Tensor,
        messages: Optional[torch.Tensor],
        ctx: VisitContext,
    ) -> torch.Tensor:
        dtype = hidden.dtype
        block_residual = hidden
        if messages is not None:
            q_msg, k_msg, v_msg, mlp_msg = messages.unbind(-2)
        if self.is_sliding:
            x = (
                hidden
                if self.attention_input is None
                else self.attention_input(block_states, hidden)[0]
            )
            if messages is None:
                q = k = v = self.input_layernorm(x)
            else:
                q = self.input_layernorm(self.q_message_gate(x, q_msg))
                k = self.input_layernorm(self.k_message_gate(x, k_msg))
                v = self.input_layernorm(self.v_message_gate(x, v_msg))
        else:
            if self.attention_input is None:
                q = k = v = hidden
            else:
                q, k, v = self.attention_input(block_states, hidden)
            if messages is not None:
                q = self.q_message_gate(q, q_msg)
                k = self.k_message_gate(k, k_msg)
                v = self.v_message_gate(v, v_msg)
            q, k, v = (
                self.q_input_layernorm(q),
                self.k_input_layernorm(k),
                self.v_input_layernorm(v),
            )
        out = self.self_attn(q.to(dtype), k.to(dtype), v.to(dtype), ctx)
        out = self.post_attention_convolution(out, ctx.conv_mask)

        residual = out = out.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = out = block_residual.float() + out
        out = self.mlp_input(block_states, out)[0]
        if messages is not None:
            out = self.mlp_message_gate(out, mlp_msg)
        out = self.mlp(self.post_attention_layernorm(out).to(dtype))
        out = self.post_mlp_convolution(out, ctx.conv_mask)
        return (residual + out).to(dtype)


# -- the model --------------------------------------------------------------------------------


class Inkling2LoopPreTrainedModel(PreTrainedModel):
    config_class = Inkling2LoopTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DecoderLayer"]
    _keep_in_fp32_modules_strict = list(FP32_MODULES)
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.normal_(0.0, std)
        elif isinstance(module, nn.Embedding):
            module.weight.normal_(0.0, std)
        elif isinstance(module, RMSNorm):
            module.weight.fill_(1.0)
        elif isinstance(module, RelativePositionBias):
            module.proj.normal_(0.0, std)
        elif isinstance(module, Experts):
            module.gate_up_proj.normal_(0.0, std)
            module.down_proj.normal_(0.0, std)
        elif isinstance(module, TopKRouter):
            module.weight.normal_(0.0, std)
            module.e_score_correction_bias.zero_()
        elif isinstance(module, (ScaledL2Clip, LogScaledL2Clip)):
            module.scale.fill_(1.0)
        elif isinstance(module, DynamicShortConvolution):
            module.static_weight.zero_()
        elif isinstance(module, (BlockAttentionResidual, MessageAttentionResidual)):
            module.query.zero_()


class NormedEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int], eps: float
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_norm = RMSNorm(embedding_dim, eps=eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(super().forward(input_ids))


def _stack(states: Sequence[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """A list of ``[batch, seq, hidden]`` block states as ``[batch, seq, n, hidden]``."""
    if not states:
        return like.new_zeros(*like.shape[:-1], 0, like.shape[-1])
    return torch.stack(list(states), dim=-2)


class Inkling2LoopTextModel(Inkling2LoopPreTrainedModel):
    def __init__(self, config: Inkling2LoopTextConfig) -> None:
        super().__init__(config)
        self.embed_tokens = NormedEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id, config.rms_norm_eps
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_input = BlockAttentionResidual(config)
        self.message_input = MessageAttentionResidual(config)
        # poros.moe.inkling2.visits sets this to stream the decoder; None runs it resident
        self.poros_visit_runner: Any = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    # -- the visit schedule (official forward, lines 1362-1440 of the 5.24 file) --------------

    def segments(self) -> List[Tuple[Tuple[int, ...], int]]:
        """The forward's runs of layers and the loop pass each belongs to, in order."""
        c = self.config
        pre = tuple(range(0, c.loop_layer_index_start))
        core = tuple(range(c.loop_layer_index_start, c.loop_layer_index_end))
        post = tuple(range(c.loop_layer_index_end, c.num_hidden_layers))
        return [(pre, 0), *((core, p) for p in range(c.loop_num_passes)), (post, 0)]

    def build_context(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> VisitContext:
        batch, seq = inputs_embeds.shape[:2]
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        if position_ids is None:
            position_ids = torch.arange(seq, device=device).unsqueeze(0)
        q = torch.arange(seq, device=device)
        distance = q[:, None] - q[None, :]
        allowed = distance >= 0
        sliding = allowed & (distance < self.config.sliding_window_size)
        pad = None
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and not bool(attention_mask.all())
        ):
            pad = attention_mask.to(device=device)

        def additive(ok: torch.Tensor) -> torch.Tensor:
            ok = ok[None, None].expand(batch, 1, seq, seq)
            if pad is not None:
                ok = ok & pad[:, None, None, :].bool()
            return torch.zeros(ok.shape, dtype=dtype, device=device).masked_fill(
                ~ok, torch.finfo(dtype).min
            )

        return VisitContext(
            position_ids=position_ids,
            full_mask=additive(allowed),
            sliding_mask=additive(sliding),
            full_distance=distance,
            sliding_distance=distance,
            conv_mask=pad,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """The final norm's output as ``last_hidden_state``, as a transformers decoder returns it
        (poros.train.forward's fused MoE head reads it so)."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        ctx = self.build_context(inputs_embeds, attention_mask, position_ids)
        hidden, blocks = self.decode(inputs_embeds, ctx)
        hidden = self.final_input(_stack(blocks, hidden), hidden)[0]
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden).to(inputs_embeds.dtype))

    def decode(
        self, hidden: torch.Tensor, ctx: VisitContext
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """The decoder's visits: returns the last partial residual and the block states."""
        blocks: List[torch.Tensor] = []
        pre_hidden: Optional[torch.Tensor] = None
        pre_blocks: List[torch.Tensor] = []
        for layers, pass_idx in self.segments():
            if not layers:
                continue
            looped = self.is_loop_run(layers)
            messages = None
            if looped and pass_idx == 0:
                pre_hidden, pre_blocks = hidden, list(blocks)
            elif looped:
                # the previous pass's block states past the one its first layer appended, and
                # its last partial residual; the pass restarts from the pre-loop states
                assert pre_hidden is not None
                messages = (blocks[len(pre_blocks) + 1 :], hidden)
                hidden, blocks = pre_hidden, list(pre_blocks)
            hidden, blocks = self.run_layers(layers, pass_idx, hidden, blocks, messages, ctx)
        return hidden, blocks

    def is_loop_run(self, layers: Sequence[int]) -> bool:
        return self.config.loop_layer_index_start <= layers[0] < self.config.loop_layer_index_end

    def run_layers(
        self,
        layers: Sequence[int],
        pass_idx: int,
        hidden: torch.Tensor,
        blocks: List[torch.Tensor],
        messages: Optional[Tuple[List[torch.Tensor], torch.Tensor]],
        ctx: VisitContext,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """One run of visits, resident or (``poros_visit_runner``) streamed."""
        runner = self.poros_visit_runner
        if runner is not None:
            return runner.run(self, layers, pass_idx, hidden, blocks, messages, ctx)
        hidden, appended = self.visit(layers, pass_idx, hidden, blocks, messages, ctx)
        return hidden, blocks + appended

    def visit(
        self,
        layers: Sequence[int],
        pass_idx: int,
        hidden: torch.Tensor,
        blocks: Sequence[torch.Tensor],
        messages: Optional[Tuple[Sequence[torch.Tensor], torch.Tensor]],
        ctx: VisitContext,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run ``layers`` (one pass) from ``hidden`` and ``blocks``; return the partial residual
        and the block states the run appended, in order (a boundary layer appends its input)."""
        blocks = list(blocks)
        appended: List[torch.Tensor] = []
        start = self.config.loop_layer_index_start
        for index in layers:
            layer = self.layers[index]
            layer.poros_visit_pass = pass_idx  # poros.core.sequential_rng keys dropout by it
            msg = None
            if messages is not None:
                msg_blocks, msg_hidden = messages
                msg = self.message_input(_stack(msg_blocks, msg_hidden), msg_hidden, index - start)
            out = layer(hidden, _stack(blocks, hidden), msg, ctx)
            if layer.is_block_boundary:
                blocks.append(hidden)
                appended.append(hidden)
            hidden = out
        return hidden, appended


class Inkling2LoopForCausalLM(Inkling2LoopPreTrainedModel):
    """The text model with its unpadded, softcapped head (embed and unembed are not tied)."""

    _tied_weights_keys: dict = {}
    #: poros.moe.prepare: this decoder streams by visits (poros.moe.inkling2.visits)
    poros_visits = True
    #: The attention linears Poros adapts by default (with the experts). The message gates,
    #: the residual queries, the sink and the relative-bias projections are not adapted.
    ATTENTION_TARGETS = (
        "self_attn.q_proj.linear",
        "self_attn.k_proj.linear",
        "self_attn.v_proj.linear",
        "self_attn.o_proj.linear",
    )

    def poros_attach_attention_adapters(
        self, rank: int, alpha: int, dropout: float, seed: int
    ) -> List[str]:
        """Wrap the attention linears in ``NativeLoraLinear`` (base frozen, adapters trainable),
        initialised from ``seed`` without moving the caller's random stream. Returns the names."""
        from poros.train.adapters import NativeLoraLinear

        names = [
            n
            for n, m in self.named_modules()
            if isinstance(m, nn.Linear) and n.endswith(self.ATTENTION_TARGETS)
        ]
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1)  # the experts' adapters take ``seed``
            for name in names:
                parent_name, _, child = name.rpartition(".")
                parent = self.get_submodule(parent_name)
                setattr(
                    parent, child, NativeLoraLinear(getattr(parent, child), rank, alpha, dropout)
                )
        return names

    def __init__(self, config: Any) -> None:
        config = (
            text_config_of(config) if not isinstance(config, Inkling2LoopTextConfig) else config
        )
        super().__init__(config)
        self.model = Inkling2LoopTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.unpadded_vocab_size, bias=False)
        self.logit_softcap = Softcap(config.final_logit_softcapping)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_decoder(self) -> nn.Module:
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        ).last_hidden_state
        logits = self.logit_softcap(self.lm_head(hidden).float())
        loss = None
        if labels is not None:
            shift = F.pad(labels, (0, 1), value=-100)[..., 1:]
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                shift.reshape(-1).to(logits.device),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class Inkling2LoopWrappedForCausalLM(Inkling2LoopForCausalLM):
    """The text model built from the released wrapper config (``model_type: inkling2_loop``):
    the Auto classes' entry for it. Its config is the wrapper's ``text_config``."""

    config_class = Inkling2LoopConfig


def cast(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """``model.to(dtype)`` with the official fp32 modules left in fp32."""
    model.to(dtype)
    for name, param in model.named_parameters():
        if any(key in name for key in FP32_MODULES):
            param.data = param.data.float()
    for name, buf in model.named_buffers():
        if any(key in name for key in FP32_MODULES):
            buf.data = buf.data.float()
    return model
