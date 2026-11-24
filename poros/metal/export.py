"""Merged export on a Mac: the LoRA folded into the base one shard at a time, on the CPU.

The merge is ``poros.moe.export``'s, with its one rounding. A frozen weight becomes what training
saw, its NF4 round trip, plus ``scaling * B @ A``, added in fp32 and rounded to the checkpoint's
dtype. Every other tensor is copied. Nothing holds more than one shard and the adapter, so 16 GB is
no limit, and the output keeps the base's names, shards and index, so plain transformers loads it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poros.cli.verbs.shared import size_of
from poros.cli.words.models import by_id

MERGES_UP_TO_B = 7.0  # above this, a merged model has never been loaded back
_PREFIX = "base_model.model."


def merges(model_id: str) -> bool:
    """A catalog model of a size this release merges, or a developer's own directory."""
    entry = by_id(model_id)
    return entry is None or size_of(entry) <= MERGES_UP_TO_B


def _factors(adapter: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Base tensor name -> (lora_A, lora_B), from the names in PEFT's own file."""
    factors = {
        name.removeprefix(_PREFIX).replace(".lora_A.weight", ".weight"):
            (a, adapter[name.replace(".lora_A.", ".lora_B.")])
        for name, a in adapter.items()
        if ".lora_A." in name
    }
    if len(adapter) != 2 * len(factors):
        raise ValueError(f"the adapter holds more than LoRA factors: {sorted(adapter)[:3]}")
    return factors


def merged(config: Any, adapter_dir: Path, output_dir: Path) -> None:
    """Write the base of ``config``, a run's own, with its adapter folded in, to ``output_dir``."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    from poros.metal import experts, same_base, settle
    from poros.metal.load import fetch
    from poros.metal.pack import skeleton
    from poros.moe.export import _merge, copy_base_files, export_merged
    from poros.train.streamed_load import _key_remapper, _shard_map

    settle(config)
    source, _, revision = fetch(config, weights=True)
    same_base(config.model_revision, revision, adapter_dir, config.model_name_or_path)
    shards = _shard_map(str(source))
    from poros.moe.carried import ensure_registered

    ensure_registered(source)  # a family Poros carries registers itself first
    base = skeleton(AutoConfig.from_pretrained(source, trust_remote_code=False))
    moe = experts.is_moe(base)
    if not moe:
        # The adapter says the model's names, and a shard may say others: Gemma 3's are older.
        model_name = _key_remapper(base)
        factors = _factors(load_file(adapter_dir / "adapter_model.safetensors"))
        unknown = factors.keys() - {model_name(name) for names in shards.values() for name in names}
        if unknown:
            raise ValueError(f"the adapter does not match this base: {sorted(unknown)[:3]}")

    if output_dir.resolve() == source.resolve():
        raise ValueError(f"refusing to merge into the base model's own folder: {source}")
    from poros.train.export import room_for, staged_output

    room_for(sum((source / shard).stat().st_size for shard in shards), output_dir)
    scaling = config.lora_alpha / config.lora_rank
    if moe:
        if (adapter_dir / "adapter_config.json").is_file():  # a Mac's run: PEFT's files
            export_merged(str(source), experts.read_adapter(adapter_dir), str(output_dir),
                          scaling=scaling)
        else:  # a CUDA run's, moved here: its own tensors, merged as its own export merges them
            export_merged(str(source),
                          torch.load(adapter_dir / "adapter_weights.pt", weights_only=True),
                          str(output_dir), scaling=scaling,
                          dtype=getattr(torch, config.torch_dtype))
        return
    with staged_output(output_dir) as stage:  # a failed merge leaves the folder as it was
        for shard, names in sorted(shards.items()):
            tensors = {}
            with safe_open(source / shard, framework="pt", device="cpu") as f:
                for name in sorted(names):
                    tensors[name] = f.get_tensor(name)
                    if model_name(name) in factors:
                        a, b = factors[model_name(name)]
                        tensors[name] = _merge(tensors[name], a, b, scaling, torch.bfloat16)
            save_file(tensors, stage / shard, metadata={"format": "pt"})
        copy_base_files(source, stage)
