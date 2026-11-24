"""The released Inkling 2 Mini checkpoint: its tensor names and its expert formats.

The checkpoint keeps Thinking Machines' own names (``model.llm.layers.N.attn.wq_du``, ...) and
stores the routed experts quantized: NVFP4 on most layers, MXFP8 on the last few (which layers
is in its ``hf_quant_config.json``; the tensors say it too). This module

- maps each checkpoint name to the decoder's parameter name (``checkpoint_targets``), as
  transformers 5.24's conversion mapping for ``inkling2_loop_text`` does;
- decodes both expert formats exactly (``decode_nvfp4``, ``decode_mxfp8``), and the gate / up
  row interleave of the fused gate-up tensor (``deinterleave_gate_up``);
- names the two ways a run can hold the experts (``ExpertCodec``): ``released`` keeps the
  released bytes and dequantizes them for compute, so a fine-tune starts from exactly the
  released model; ``nf4`` decodes them to bf16 and quantizes that to NF4, the store every other
  family uses (a second rounding).

The formats, from transformers 5.24.0 ``core_model_loading.Inkling2LoopDequantizeExperts``
(Apache-2.0, see NOTICE):

- NVFP4: two E2M1 codes per byte, the even element in the low nibble; one float8 e4m3 scale per
  16 values along a row; one fp32 factor per expert, ``amax / outer_scale_denominator``.
- MXFP8: float8 e4m3 values; one power-of-two scale ``2 ** (s - 127)`` (an unsigned byte ``s``)
  per 32 values along a row.
"""

from __future__ import annotations

import enum
import re
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

__all__ = [
    "ExpertCodec",
    "E2M1_VALUES",
    "decode_nvfp4",
    "decode_mxfp8",
    "decode_experts",
    "deinterleave_gate_up",
    "checkpoint_targets",
    "convert_state_dict",
    "SKIPPED_PREFIXES",
]


class ExpertCodec(str, enum.Enum):
    """How a run holds the released experts."""

    RELEASED = "released"  # the released NVFP4 / MXFP8 bytes, dequantized for compute
    NF4 = "nf4"  # decoded to bf16, then NF4 as for every other family


#: The 16 E2M1 values, by code (bit 3 the sign).
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def decode_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    denominator: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """NVFP4 ``[experts, rows, cols / 2]`` bytes to ``[experts, rows, cols]`` in ``dtype``.

    ``scale``: float8 e4m3 ``[experts, rows, cols / 16]``; ``amax`` one per expert;
    ``denominator`` one (or one per expert). Computed in fp32, rounded once to ``dtype``.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"NVFP4 codes are uint8, got {packed.dtype}")
    experts, rows, half = packed.shape
    cols = 2 * half
    if tuple(scale.shape) != (experts, rows, cols // 16):
        raise ValueError(
            f"NVFP4 scale shape {tuple(scale.shape)} does not fit {tuple(packed.shape)}"
        )
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = packed.long()
    values = torch.stack((lut[codes & 0xF], lut[codes >> 4]), dim=-1).view(
        experts, rows, cols // 16, 16
    )
    values = values * scale.float()[..., None]
    factor = (amax.float().reshape(-1) / denominator.float().reshape(-1)).expand(experts)
    values = values * factor[:, None, None, None]
    return values.reshape(experts, rows, cols).to(dtype)


def decode_mxfp8(
    values: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """MXFP8 ``[experts, rows, cols]`` float8 e4m3 with ``[experts, rows, cols / 32]`` byte
    exponents to ``dtype`` (in fp32, rounded once)."""
    if values.dtype != torch.float8_e4m3fn:
        raise ValueError(f"MXFP8 values are float8_e4m3fn, got {values.dtype}")
    experts, rows, cols = values.shape
    if tuple(scale.shape) != (experts, rows, cols // 32):
        raise ValueError(
            f"MXFP8 scale shape {tuple(scale.shape)} does not fit {tuple(values.shape)}"
        )
    out = values.float().view(experts, rows, cols // 32, 32)
    out = out * torch.exp2(scale.float() - 127)[..., None]
    return out.reshape(experts, rows, cols).to(dtype)


def decode_experts(
    tensors: Mapping[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """One expert tensor from its checkpoint parts (``weight``, ``weight_scale``, and for
    NVFP4 ``weight_amax`` and ``weight_outer_scale_denominator``): the format is read from them."""
    weight = tensors["weight"]
    if "weight_amax" in tensors:
        return decode_nvfp4(
            weight,
            tensors["weight_scale"],
            tensors["weight_amax"],
            tensors["weight_outer_scale_denominator"],
            dtype,
        )
    if "weight_scale" in tensors:
        return decode_mxfp8(weight, tensors["weight_scale"], dtype)
    if weight.is_floating_point() and weight.element_size() >= 2:
        return weight.to(dtype)  # already decoded (a merged bf16 save)
    raise ValueError(f"unknown expert format: {sorted(tensors)} ({weight.dtype})")


def deinterleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """The checkpoint's fused rows alternate gate and up (``g0, u0, g1, u1, ...``); the model's
    ``gate_up_proj`` holds every gate row, then every up row."""
    experts, rows, cols = weight.shape
    return weight.view(experts, rows // 2, 2, cols).transpose(1, 2).reshape(experts, rows, cols)


#: Checkpoint tensors the text decoder does not hold.
SKIPPED_PREFIXES = ("model.visual.", "mtp.", "model.mtp.")

# Ordered renames (pattern, replacement), applied in turn, as transformers 5.24's mapping.
_RENAMES: List[Tuple[str, str]] = [
    (r"^model\.llm\.unembed\.", "lm_head."),
    (r"^model\.llm\.", "model."),
    # the loop's message gates first: their "controllers.attn." is not an attention module
    (r"controllers\.attn\.", "q_message_gate."),
    (r"controllers\.k\.", "k_message_gate."),
    (r"controllers\.v\.", "v_message_gate."),
    (r"controllers\.mlp\.", "mlp_message_gate."),
    (r"\.embed\.weight$", ".embed_tokens.weight"),
    (r"\.embed_norm\.", ".embed_tokens.embed_norm."),
    (r"mlp\.gate\.bias$", "mlp.gate.e_score_correction_bias"),
    (r"\.attn\.", ".self_attn."),
    (r"self_attn\.wq_du", "self_attn.q_proj.linear"),
    (r"self_attn\.wk_dv", "self_attn.k_proj.linear"),
    (r"self_attn\.wv_dv", "self_attn.v_proj.linear"),
    (r"self_attn\.k_sconv\.", "self_attn.k_proj.sconv."),
    (r"self_attn\.v_sconv\.", "self_attn.v_proj.sconv."),
    (r"self_attn\.rel_logits_proj\.", "self_attn.relative_position_bias."),
    (r"self_attn\.wr_du", "self_attn.relative_position_bias.linear"),
    (r"self_attn\.q_l2_clip\.", "self_attn.q_proj.norm."),
    (r"self_attn\.rel_l2_clip\.", "self_attn.relative_position_bias.norm."),
    (r"self_attn\.wo_ud", "self_attn.o_proj.linear"),
    (r"self_attn\.gate_attn", "self_attn.o_proj.gate_proj"),
    (r"self_attn\.attn_sink_proj", "self_attn.sink_proj"),
    (r"attn_sconv\.", "post_attention_convolution."),
    (r"mlp_sconv\.", "post_mlp_convolution."),
    (r"(convolution|sconv)\.weight$", r"\1.conv1d.weight"),
    (r"(convolution|sconv)\.static_w$", r"\1.static_weight"),
    (r"(convolution|sconv)\.weight_proj$", r"\1.weight_proj.weight"),
    (r"mlp_norm\.", "post_attention_layernorm."),
    (r"attn_norm\.", "input_layernorm."),
    (r"(q|k|v)_input_norm\.", r"\1_input_layernorm."),
    (r"attn_res\.mlp_queries\.(\d+)$", r"layers.\1.mlp_input.query"),
    (r"attn_res\.output_query$", "final_input.query"),
    (r"_message_gate\.down_weight$", "_message_gate.down_proj.weight"),
    (r"_message_gate\.gain_weight$", "_message_gate.message_gate_proj.weight"),
    (r"_message_gate\.forget_weight$", "_message_gate.forget_gate_proj.weight"),
]
_ATTN_QUERY = re.compile(r"^(.*)attn_res\.attn_queries\.(\d+)$")
_EXPERT = re.compile(
    r"^(model\.llm\.layers\.\d+\.mlp\.experts)\.(w13|w2)_(weight(?:_scale|_amax|_outer_scale_denominator)?)$"
)
_MESSAGE_QUERIES = (
    "model.llm.middle_cycle_cross_pass_queries",
    "model.llm.middle_cycle_cross_pass_kv_queries",
)


def rename(name: str) -> str:
    """A plain (one tensor to one parameter) checkpoint name's parameter name."""
    match = _ATTN_QUERY.match(name)
    if match:  # attn_queries.i belongs to layer i + 1 (layer 0 has no attention input)
        name = f"{match.group(1)}layers.{int(match.group(2)) + 1}.attention_input.query"
    for pattern, replacement in _RENAMES:
        name = re.sub(pattern, replacement, name)
    return name


def checkpoint_targets(names: List[str]) -> Dict[str, List[str]]:
    """Each decoder parameter and the checkpoint tensors it is made from (in order). Names the
    text decoder does not hold (``SKIPPED_PREFIXES``) are left out."""
    targets: Dict[str, List[str]] = {}
    for name in names:
        if name.startswith(SKIPPED_PREFIXES):
            continue
        expert = _EXPERT.match(name)
        if expert:
            prefix, which, _ = expert.groups()
            target = rename(prefix) + (".gate_up_proj" if which == "w13" else ".down_proj")
        elif name in _MESSAGE_QUERIES:
            target = "model.message_input.query"
        else:
            target = rename(name)
        targets.setdefault(target, []).append(name)
    return targets


def merge_message_queries(queries: torch.Tensor, kv_queries: torch.Tensor) -> torch.Tensor:
    """The checkpoint's two ``[2 * loop_layers, hidden]`` message queries (q and mlp of each
    loop layer; k and v of each) as the model's ``[loop_layers, 4, hidden]`` (q, k, v, mlp)."""
    q_mlp = queries.unflatten(0, (-1, 2))
    k_v = kv_queries.unflatten(0, (-1, 2))
    return torch.cat([q_mlp[:, :1], k_v, q_mlp[:, 1:]], dim=1)


def convert_state_dict(
    read: Callable[[str], torch.Tensor],
    names: List[str],
    *,
    unpadded_vocab_size: int,
    dtype: torch.dtype = torch.bfloat16,
    expert_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """The decoder's state dict from a released checkpoint (``read(name)`` gives one tensor).

    Experts are decoded to ``expert_dtype`` (default ``dtype``); other tensors keep their stored
    dtype (the caller casts, keeping the official fp32 modules in fp32). The head drops its
    padded rows."""
    out: Dict[str, torch.Tensor] = {}
    for target, sources in checkpoint_targets(names).items():
        if target.endswith((".gate_up_proj", ".down_proj")):
            parts = {}
            for source in sources:
                part = _EXPERT.match(source).group(3)
                parts[part] = read(source)
            weight = decode_experts(parts, expert_dtype or dtype)
            out[target] = (
                deinterleave_gate_up(weight) if target.endswith("gate_up_proj") else weight
            )
        elif target == "model.message_input.query":
            tensors = {s: read(s) for s in sources}
            out[target] = merge_message_queries(
                tensors[_MESSAGE_QUERIES[0]], tensors[_MESSAGE_QUERIES[1]]
            )
        else:
            (source,) = sources
            tensor = read(source)
            if target == "lm_head.weight":
                tensor = tensor[:unpadded_vocab_size]
            out[target] = tensor
    return out


def released_names(config) -> List[str]:
    """The text tensor names a checkpoint of this decoder config holds in the released layout,
    experts unquantized (``w13_weight``, ``w2_weight``), as a merged save writes them."""
    names = [
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
        "model.llm.unembed.weight",
        "model.llm.attn_res.output_query",
        *_MESSAGE_QUERIES,
    ]
    start, end = config.loop_layer_index_start, config.loop_layer_index_end
    for i in range(config.num_hidden_layers):
        p = f"model.llm.layers.{i}."
        sliding = config.layer_types[i] == "hybrid_sliding"
        names += [
            p + f"attn.{w}.weight"
            for w in ("wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud", "gate_attn", "attn_sink_proj")
        ]
        names += [
            p + "attn.q_l2_clip.scale",
            p + "attn.rel_l2_clip.scale",
            p + "attn.rel_logits_proj.proj",
            p + "attn_sconv.weight",
            p + "mlp_sconv.weight",
            p + "mlp_norm.weight",
            p + "mlp.gate.weight",
            p + "mlp.gate.bias",
            p + "mlp.experts.w13_weight",
            p + "mlp.experts.w2_weight",
        ]
        for kv in ("k", "v"):
            names += (
                [p + f"attn.{kv}_sconv.weight"]
                if sliding
                else [p + f"attn.{kv}_sconv.static_w", p + f"attn.{kv}_sconv.weight_proj"]
            )
        names += (
            [p + "attn_norm.weight"] if sliding else [p + f"{x}_input_norm.weight" for x in "qkv"]
        )
        if start <= i < end:
            names += [
                p + f"controllers.{c}.{w}_weight"
                for c in ("attn", "k", "v", "mlp")
                for w in ("down", "forget", "gain")
            ]
        names.append(f"model.llm.attn_res.mlp_queries.{i}")
        if i > 0:
            names.append(f"model.llm.attn_res.attn_queries.{i - 1}")
    return sorted(names)


def to_checkpoint(state: Mapping[str, torch.Tensor], config) -> Dict[str, torch.Tensor]:
    """The decoder's state dict in the released layout (experts unquantized, gate / up rows
    interleaved, the head padded to ``vocab_size`` with zero rows, the message queries split):
    the inverse of ``convert_state_dict``."""
    out: Dict[str, torch.Tensor] = {}
    targets = checkpoint_targets(released_names(config))
    for target, sources in targets.items():
        tensor = state[target]
        if target == "model.message_input.query":
            out[_MESSAGE_QUERIES[0]] = tensor[:, [0, 3]].flatten(0, 1).contiguous()
            out[_MESSAGE_QUERIES[1]] = tensor[:, 1:3].flatten(0, 1).contiguous()
            continue
        (source,) = sources
        if target.endswith(".gate_up_proj"):
            experts, rows, cols = tensor.shape
            tensor = tensor.view(experts, 2, rows // 2, cols).transpose(1, 2)
            tensor = tensor.reshape(experts, rows, cols)
        elif target == "lm_head.weight":
            pad = tensor.new_zeros(config.vocab_size - tensor.shape[0], tensor.shape[1])
            tensor = torch.cat([tensor, pad])
        out[source] = tensor.contiguous()
    return out
