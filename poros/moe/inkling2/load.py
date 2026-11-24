"""Reading the released Inkling 2 Mini checkpoint into the streamed MoE load (poros.moe.load).

``fill`` walks the shards once, as ``load_moe_streamed`` does for every other family:

- every non-expert tensor goes into the meta skeleton under its decoder name
  (poros.moe.inkling2.codec), in ``dtype``, except the modules the official implementation keeps
  in fp32 (``FP32_MODULES``), which stay fp32; the head drops its padded rows; the two message
  query tensors merge into one;
- each layer's experts go to the slab arena one expert at a time: their released bytes as they
  are (``ExpertCodec.RELEASED``, a quantized checkpoint's default: poros.moe.inkling2.released),
  or decoded to the model dtype for the arena to quantize to NF4 (``ExpertCodec.NF4``: asked
  with ``moe_expert_format: nf4`` or ``POROS_INKLING_EXPERTS=nf4``, and always for an
  unquantized checkpoint; poros.moe.inkling2.formats decides); the vision tower and the
  multi-token-prediction head are not read.

A tensor whose parts span shards waits for the rest; one never completed is an error.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn

from poros.moe.inkling2 import codec
from poros.moe.inkling2.modeling import FP32_MODULES

logger = logging.getLogger(__name__)

__all__ = ["fill", "is_inkling", "released_formats"]


def released_formats(shards: Dict[str, List[str]], ordinal: Dict[str, int]) -> Optional[List[str]]:
    """Each MoE layer's released expert format (``nvfp4`` or ``mxfp8``), by ordinal, from the
    checkpoint's tensor names; None where a layer's experts are not quantized (a merged bf16
    save), which only the NF4 route reads."""
    names = {n for ns in shards.values() for n in ns}
    formats: List[Optional[str]] = [None] * len(ordinal)
    for target, sources in codec.checkpoint_targets(sorted(names)).items():
        module, _, tensor = target.rpartition(".")
        if tensor != "gate_up_proj" or module not in ordinal:
            continue
        parts = {s.rpartition(".experts.")[2] for s in sources}
        fmt = (
            "nvfp4"
            if "w13_weight_amax" in parts
            else "mxfp8"
            if "w13_weight_scale" in parts
            else None
        )
        formats[ordinal[module]] = fmt
    return None if any(f is None for f in formats) else [str(f) for f in formats]


def is_inkling(config: Any) -> bool:
    """Whether a config (an object or config.json's dict) is Inkling 2 Mini's."""
    model_type = (
        config.get("model_type")
        if isinstance(config, dict)
        else getattr(config, "model_type", None)
    )
    return model_type in ("inkling2_loop", "inkling2_loop_text")


def _dtype_for(name: str, dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if any(key in name for key in FP32_MODULES) else dtype


def fill(
    model: nn.Module,
    checkpoint_dir: str,
    shards: Dict[str, List[str]],
    *,
    dtype: torch.dtype,
    ordinal: Dict[str, int],
    arena: Any,
    assign: Callable[[nn.Module, str, torch.Tensor, torch.dtype], None],
    locate: Callable[[nn.Module, str], Any],
    released: Optional[List[str]] = None,
) -> List[str]:
    """Load ``shards`` (shard file to tensor names) into ``model`` and its experts into
    ``arena`` (None: the experts come from a reused store and are not read). ``ordinal``: each
    experts module's MoE-layer ordinal. ``released``: each layer's format when ``arena`` keeps
    the released bytes (a ``ReleasedSlabArena``), else None (the arena takes decoded weights).
    Returns the checkpoint names left unread."""
    from safetensors import safe_open

    names = [n for ns in shards.values() for n in ns]
    targets = codec.checkpoint_targets(names)
    source_of = {s: t for t, sources in targets.items() for s in sources}
    defined = set(model.state_dict().keys())
    unpadded = int(model.config.unpadded_vocab_size)
    pending: Dict[str, Dict[str, torch.Tensor]] = {}
    left = [n for n in names if n not in source_of]
    for shard, shard_names in sorted(shards.items()):
        with safe_open(os.path.join(checkpoint_dir, shard), framework="pt", device="cpu") as f:
            for name in sorted(shard_names):
                target = source_of.get(name)
                if target is None:
                    continue
                expert = target.endswith((".gate_up_proj", ".down_proj"))
                if expert and arena is None:
                    continue  # reused from the store: never read
                parts = pending.setdefault(target, {})
                parts[name] = f.get_tensor(name).clone()
                if len(parts) < len(targets[target]):
                    continue
                del pending[target]
                if expert and released is not None:
                    _write_released(target, parts, arena, ordinal)
                    continue
                if expert:
                    _write_experts(target, parts, arena, ordinal, dtype)
                    continue
                if target not in defined:
                    left.append(name)
                    continue
                tensor = codec.convert_state_dict(
                    parts.__getitem__, list(parts), unpadded_vocab_size=unpadded
                )[target]
                module, attr = locate(model, target)
                assign(module, attr, tensor, _dtype_for(target, dtype))
    if pending:
        raise RuntimeError(f"checkpoint lacks parts of {sorted(pending)[:5]}")
    return left


def _write_experts(
    target: str,
    parts: Dict[str, torch.Tensor],
    arena: Any,
    ordinal: Dict[str, int],
    dtype: torch.dtype,
) -> None:
    module, _, tensor_name = target.rpartition(".")
    layer = ordinal.get(module)
    if layer is None:
        raise RuntimeError(f"checkpoint experts {module!r} are not a MoE layer of the model")
    decoded = codec.convert_state_dict(
        parts.__getitem__, list(parts), unpadded_vocab_size=0, expert_dtype=dtype
    )[target]
    for expert in range(decoded.shape[0]):
        arena.write((layer, expert), tensor_name, decoded[expert])


def _write_released(
    target: str, parts: Dict[str, torch.Tensor], arena: Any, ordinal: Dict[str, int]
) -> None:
    """One layer's experts into a ``ReleasedSlabArena``: the released bytes, one expert at a
    time, gate / up rows de-interleaved (rows only: every scale lies within a row)."""
    module, _, tensor_name = target.rpartition(".")
    layer = ordinal.get(module)
    if layer is None:
        raise RuntimeError(f"checkpoint experts {module!r} are not a MoE layer of the model")
    # w13_weight, w13_weight_scale, ... -> weight, weight_scale, ...
    by_part = {n.rpartition(".experts.")[2].split("_", 1)[1]: t for n, t in parts.items()}
    rows = codec.deinterleave_gate_up if tensor_name == "gate_up_proj" else (lambda t: t)
    weight = by_part["weight"]
    if "weight_amax" in by_part:  # NVFP4
        amax, den = by_part["weight_amax"], by_part["weight_outer_scale_denominator"]
        factor = (amax.float().reshape(-1) / den.float().reshape(-1)).expand(weight.shape[0])
        codes, scale = rows(weight), rows(by_part["weight_scale"])
        for e in range(weight.shape[0]):
            arena.write_parts(
                (layer, e),
                tensor_name,
                {"factor": factor[e].clone(), "scale": scale[e], "codes": codes[e]},
            )
    else:  # MXFP8
        values, scale = rows(weight), rows(by_part["weight_scale"])
        for e in range(weight.shape[0]):
            arena.write_parts((layer, e), tensor_name, {"values": values[e], "scale": scale[e]})


def released_shards(
    checkpoint_dir: str, shard_map: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[str]]:
    """The checkpoint's shards and tensor names (the text decoder's file only: the MTP head
    sits in its own ``mtp.safetensors``, outside the index)."""
    if shard_map is not None:
        return shard_map
    from poros.train.streamed_load import _shard_map

    return _shard_map(checkpoint_dir)
