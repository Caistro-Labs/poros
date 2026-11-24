"""Merged export for ``poros.moe``: the LoRA folded into the experts one at a time.

A MoE checkpoint is mostly experts. This merge walks the base checkpoint's
safetensors shards on CPU, one tensor at a time, through the reader and
writer dense export uses (``poros.train.export``), and holds at most a few
expert tensors plus the adapter. Each expert tensor becomes the weight
training saw (its NF4 round trip through ``poros.moe.quant``, with gate and
up fused as the loader fuses them) plus its LoRA delta ``scaling * B @ A``,
added in fp32 and rounded once to the checkpoint dtype, the rule dense
export follows too. Every other tensor is copied unchanged (the MoE load
keeps them unquantized), and the output keeps the base's tensor names,
shards and index, so it loads wherever the base model does.
"""

from __future__ import annotations

import os
import re
import stat
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch

from poros.core.hub_files import BASE_FILES as _BASE_FILES
from poros.moe import quant
from poros.moe.arch import MoeArch
from poros.train.export import (
    CheckpointReader,
    merge_weight,
    room_before_download,
    room_for,
    shard_bytes,
    staged_output,
    write_rewritten,
)
from poros.train.streamed_load import _resolve_checkpoint_dir

__all__ = ["export_merged", "copy_base_files"]

# One expert's tensor, as the qwen-style checkpoints this export reads name it.
_EXPERT_KEY = re.compile(r"^(.+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

# One layer's experts as the model holds them, as the newest checkpoints store them.
_FUSED_KEY = re.compile(r"^(.+)\.experts\.(gate_up_proj|down_proj)$")

_FACTORS = ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down")


def _own_file(path: Path) -> bool:
    """A regular file, or in a Hub snapshot a link into that repo's own blobs, as all of them are.
    No other link: token.txt -> ~/.cache/huggingface/token is never a file of this model's."""
    try:
        if not path.is_symlink():
            return stat.S_ISREG(path.lstat().st_mode)
        folder = path.parent.resolve()
        return (folder.parent.name == "snapshots" and path.is_file()
                and path.resolve().is_relative_to(folder.parents[1] / "blobs"))
    except (OSError, RuntimeError):
        return False


def copy_base_files(source: "str | os.PathLike", output_dir: "str | os.PathLike") -> None:
    """The base's config and tokenizer, by name and as new files, into ``output_dir``."""
    from poros.core.tokenizer import read_once

    for name in _BASE_FILES:
        path = Path(source, name)
        if _own_file(path):
            Path(output_dir, name).write_bytes(read_once(str(path)))


def _merge(
    weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float, dtype: torch.dtype
) -> torch.Tensor:
    """``weight`` as training saw it, in ``dtype``, plus ``scaling * b @ a``."""
    seen = quant.dequantize(quant.quantize_tensor(weight.to(dtype)), weight.shape, dtype)
    return merge_weight(seen, a, b, scaling, weight.dtype)


def _fused_in(names, config: Any, layers: set[str]) -> dict[str, tuple[str, tuple[str, str]]]:
    """Each fused expert tensor the model reads -> its block and its LoRA factors. A draft head may
    hold others, and a checkpoint its own prefix, so the model's skeleton names them."""
    candidates = [name for name in names if _FUSED_KEY.match(name)]
    if not candidates:
        return {}
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    from poros.train.streamed_load import _key_remapper

    with torch.random.fork_rng(devices=[]), init_empty_weights(include_buffers=False):
        remap = _key_remapper(AutoModelForCausalLM.from_config(config))
    found = {}
    for name in candidates:
        layer, proj = _FUSED_KEY.match(remap(name)).groups()
        if layer in layers:
            found[name] = layer, _FACTORS[:2] if proj == "gate_up_proj" else _FACTORS[2:]
    return found


def export_merged(
    model_name_or_path: str,
    adapter: dict[str, torch.Tensor],
    output_dir: str,
    *,
    scaling: float,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
) -> None:
    """Write the base model with ``adapter`` merged into its experts to ``output_dir``.

    ``adapter`` is the expert LoRA state ``PorosTrainer.save`` writes
    (``adapter_weights.pt``), ``scaling`` its ``lora_alpha / lora_rank``,
    and ``dtype`` the dtype it trained in. The base's config, tokenizer and
    shard index are copied alongside, and nothing else in its folder is.
    """
    from transformers import AutoConfig

    room_before_download(model_name_or_path, revision, output_dir)  # before any byte is fetched
    source = _resolve_checkpoint_dir(model_name_or_path, revision)
    if os.path.realpath(output_dir) == os.path.realpath(source):
        raise ValueError(f"the output directory {output_dir} is the base checkpoint itself")
    from poros.moe.carried import ensure_registered

    ensure_registered(source)  # a family Poros carries registers itself first
    config = AutoConfig.from_pretrained(source)
    arch = MoeArch.from_config(config)
    layers = [arch.block_module(i) for i in arch.moe_layer_indices]
    expected = {f"{layer}.adapter.{factor}" for layer in layers for factor in _FACTORS}
    if adapter.keys() != expected:
        raise ValueError(
            f"the adapter does not match {model_name_or_path}'s expert LoRA: missing "
            f"{sorted(expected - adapter.keys())[:3]}, unexpected "
            f"{sorted(adapter.keys() - expected)[:3]}"
        )
    reader = CheckpointReader(source)
    room_for(shard_bytes(reader), output_dir)  # before anything is merged or written
    experts = sum(_EXPERT_KEY.match(name) is not None for name in reader.entries)
    fused = _fused_in(reader.entries, config, set(layers))
    if (experts, len(fused)) not in (
        (3 * len(layers) * arch.layout.experts_per_layer, 0),
        (0, 2 * len(layers)),
    ):
        raise ValueError(
            f"{model_name_or_path} does not store its experts as gate_proj/up_proj/"
            f"down_proj per expert ({experts} found) or fused per layer ({len(fused)} "
            "found), the only layouts merged export reads (Mixtral's w1/w3/w2 is not supported)"
        )

    # Gate and up merge together. They sit side by side in a sorted shard, and
    # a cache miss only recomputes the pair (deterministically).
    @lru_cache(maxsize=1)
    def gate_up(layer: str, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = f"{layer}.experts.{expert}.{{}}_proj.weight"
        gate = reader.get(name.format("gate"))
        fused = torch.cat([gate, reader.get(name.format("up"))])
        a, b = (adapter[f"{layer}.adapter.{f}"][expert] for f in _FACTORS[:2])
        return _merge(fused, a, b, scaling, dtype).split(len(gate))

    def transform(name: str) -> Optional[torch.Tensor]:
        if name in fused:
            layer, proj = fused[name]
            a, b = (adapter[f"{layer}.adapter.{f}"] for f in proj)
            return torch.stack([
                _merge(weight, a[e], b[e], scaling, dtype)
                for e, weight in enumerate(reader.get(name))
            ])
        match = _EXPERT_KEY.match(name)
        if match is None:
            return None  # copied byte for byte
        layer, expert, proj = match[1].removesuffix(".experts"), int(match[2]), match[3]
        if proj == "down_proj":
            a, b = (adapter[f"{layer}.adapter.{f}"][expert] for f in _FACTORS[2:])
            return _merge(reader.get(name), a, b, scaling, dtype)
        gate, up = gate_up(layer, expert)
        return up if proj == "up_proj" else gate

    with staged_output(output_dir) as stage:
        write_rewritten(reader, str(stage), transform)
        copy_base_files(source, stage)
