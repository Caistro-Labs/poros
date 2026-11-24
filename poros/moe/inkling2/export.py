"""Merged export of an Inkling 2 Mini run: the base checkpoint, in its released layout, with the
LoRA folded in.

The released experts are quantized (NVFP4, MXFP8); a merged expert is not representable in
either format, so the merged checkpoint holds its experts unquantized, in the base's tensor
names: ``w13_weight`` (gate and up rows interleaved again, as the release stores them) and
``w2_weight`` in the training dtype, with their scale tensors left out and the quantization
config dropped from config.json (``hf_quant_config.json`` is not written). transformers and the
decoder's own load both read that layout (``codec.decode_experts`` takes an unquantized weight).

Each merged weight is the weight training saw plus ``scaling * B @ A``, added in fp32 and
rounded once (``poros.train.export.merge_weight``):

- an expert: its released bytes decoded (``ExpertCodec.RELEASED``), or that decoded weight's
  NF4 round trip (``ExpertCodec.NF4``), as the run held it;
- an attention linear the run adapted (``q``, ``k``, ``v``, ``o``): its bf16 base weight.

Every other tensor is copied as it is, the vision tower and the multi-token-prediction file
included, so the output loads wherever the base does. Shards keep their names; one tensor at a
time is held (an expert layer's decoded weights, at most).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional

import torch

from poros.moe.inkling2 import codec

__all__ = ["export_merged", "ATTENTION_SOURCES"]

#: The adapted attention linears: checkpoint name suffix -> the decoder module suffix.
ATTENTION_SOURCES = {
    "attn.wq_du.weight": "self_attn.q_proj.linear",
    "attn.wk_dv.weight": "self_attn.k_proj.linear",
    "attn.wv_dv.weight": "self_attn.v_proj.linear",
    "attn.wo_ud.weight": "self_attn.o_proj.linear",
}
_EXPERT_FACTORS = {
    "gate_up_proj": ("lora_A_gate_up", "lora_B_gate_up"),
    "down_proj": ("lora_A_down", "lora_B_down"),
}
_ST_DTYPES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.int64: "I64",
    torch.int32: "I32",
}


def _seen(weight: torch.Tensor, expert_codec: str, dtype: torch.dtype) -> torch.Tensor:
    """An expert weight as the run computed with it."""
    if expert_codec == codec.ExpertCodec.NF4.value:
        from poros.moe import quant

        return quant.dequantize(quant.quantize_tensor(weight.to(dtype)), weight.shape, dtype)
    return weight.to(dtype)


class _ShardWriter:
    """A safetensors file written tensor by tensor (its header first, from the planned specs)."""

    def __init__(self, path: Path, specs: Dict[str, tuple]) -> None:
        header, offset = {}, 0
        for name, (dtype, shape) in specs.items():
            n = torch.empty((), dtype=dtype).element_size()
            for d in shape:
                n *= d
            header[name] = {
                "dtype": _ST_DTYPES[dtype],
                "shape": list(shape),
                "data_offsets": [offset, offset + n],
            }
            offset += n
        header["__metadata__"] = {"format": "pt"}
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        self._file = open(path, "wb")
        self._file.write(struct.pack("<Q", len(raw)) + raw)
        self._specs, self._order = specs, list(specs)

    def write(self, name: str, tensor: torch.Tensor) -> None:
        expected = self._order.pop(0)
        dtype, shape = self._specs[name]
        if name != expected or tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
            raise RuntimeError(
                f"merged shard: {name} {tensor.dtype} {tuple(tensor.shape)} where "
                f"{expected} {dtype} {tuple(shape)} was planned"
            )
        self._file.write(tensor.contiguous().view(torch.uint8).numpy().tobytes())

    def close(self) -> None:
        if self._order:
            raise RuntimeError(f"merged shard misses {self._order[:3]}")
        self._file.close()


def export_merged(
    model_name_or_path: str,
    adapter: Dict[str, torch.Tensor],
    output_dir: str,
    *,
    scaling: float,
    dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    expert_codec: Optional[str] = None,
) -> None:
    """Write the base with ``adapter`` (the state ``PorosTrainer.save`` writes: the experts'
    ``model.layers.N.mlp.adapter.lora_*`` and the attention's ``...linear.lora_A`` / ``lora_B``)
    merged, to ``output_dir``. ``expert_codec``: how the run held the experts, as the run
    records it (its ``moe_expert_format``); never the environment. A run that does not record it
    is refused: merging against other base weights than training saw would be silently wrong."""
    from safetensors import safe_open

    from poros.moe.export import copy_base_files
    from poros.moe.inkling2.config import text_config_of
    from poros.train.export import room_before_download, room_for, staged_output
    from poros.train.streamed_load import _resolve_checkpoint_dir, _shard_map

    if expert_codec not in {c.value for c in codec.ExpertCodec}:
        from poros.core.errors import PorosConfigError

        raise PorosConfigError(
            "This run does not record how it held Inkling 2 Mini's experts (moe_expert_format: "
            f"{expert_codec!r}), so the weights it trained against cannot be told: no merge.",
            remediation="Export the adapter, or merge a run that records moe_expert_format.",
        )
    from poros.moe.inkling2 import formats

    early = formats.checkpoint_names(model_name_or_path, revision)  # before the download
    if early:
        _record_fits(formats.checkpoint_state(early), expert_codec, model_name_or_path)
    room_before_download(model_name_or_path, revision, output_dir)
    source = _resolve_checkpoint_dir(model_name_or_path, revision)
    if os.path.realpath(output_dir) == os.path.realpath(source):
        raise ValueError(f"the output directory {output_dir} is the base checkpoint itself")
    config = json.loads(Path(source, "config.json").read_text())
    text = text_config_of(config)
    shards = _shard_map(source)
    names = sorted(n for ns in shards.values() for n in ns)
    _record_fits(formats.checkpoint_state(names), expert_codec, model_name_or_path)
    targets = codec.checkpoint_targets(names)
    experts = {t: s for t, s in targets.items() if t.endswith(tuple(_EXPERT_FACTORS))}
    expected = {
        f"{t.rpartition('.experts.')[0]}.adapter.{f}"
        for t in experts
        for f in _EXPERT_FACTORS[t.rpartition(".")[2]]
    }
    attention = {}
    for name in names:
        for suffix, module in ATTENTION_SOURCES.items():
            if name.startswith("model.llm.layers.") and name.endswith(suffix):
                layer = name.split(".")[3]
                attention[name] = f"model.layers.{layer}.{module}"
    adapted = {m for m in attention.values() if f"{m}.lora_A" in adapter}
    expected |= {f"{m}.lora_{x}" for m in adapted for x in "AB"}
    if not expected <= adapter.keys() or adapter.keys() - expected:
        raise ValueError(
            f"the adapter does not match {model_name_or_path}'s expert and attention LoRA: "
            f"missing {sorted(expected - adapter.keys())[:3]}, unexpected "
            f"{sorted(adapter.keys() - expected)[:3]}"
        )
    # Each expert target leaves its parts behind and is written under its weight's name.
    part_of = {s: t for t, ss in experts.items() for s in ss}
    weight_name = {
        t: next(s for s in ss if s.endswith(("w13_weight", "w2_weight")))
        for t, ss in experts.items()
    }
    plan: Dict[str, Dict[str, tuple]] = {}
    for shard, shard_names in sorted(shards.items()):
        specs: Dict[str, tuple] = {}
        with safe_open(os.path.join(source, shard), framework="pt") as f:
            for name in sorted(shard_names):
                target = part_of.get(name)
                if target is not None:
                    if name != weight_name[target]:
                        continue  # a scale: the merged expert is not quantized
                    rows, cols = text.moe_intermediate_size, text.hidden_size
                    shape = (
                        (text.n_routed_experts, 2 * rows, cols)
                        if "gate_up" in target
                        else (text.n_routed_experts, cols, rows)
                    )
                    specs[name] = (dtype, shape)
                    continue
                piece = f.get_slice(name)
                specs[name] = (_torch_dtype(piece.get_dtype()), tuple(piece.get_shape()))
        plan[shard] = specs
    # what the merge writes, from the shapes and dtypes it will write (a released expert grows
    # about seven times, NVFP4 to fp32), and the MTP file copied beside: refused before anything
    # is merged or written where the disk has less
    mtp = Path(source, "mtp.safetensors")
    room_for(
        sum(_nbytes(*spec) for specs in plan.values() for spec in specs.values())
        + (mtp.stat().st_size if mtp.is_file() else 0),
        output_dir,
    )

    with staged_output(output_dir) as stage:
        index: Dict[str, str] = {}
        for shard, specs in plan.items():
            writer = _ShardWriter(Path(stage, shard), specs)
            with safe_open(os.path.join(source, shard), framework="pt") as f:
                for name in specs:
                    target = part_of.get(name)
                    if target is not None:
                        tensor = _merged_experts(
                            source,
                            shards,
                            experts[target],
                            target,
                            adapter,
                            scaling,
                            dtype,
                            expert_codec,
                        )
                    elif name in attention and attention[name] in adapted:
                        module = attention[name]
                        from poros.train.export import merge_weight

                        base = f.get_tensor(name)
                        tensor = merge_weight(
                            base,
                            adapter[f"{module}.lora_A"],
                            adapter[f"{module}.lora_B"],
                            scaling,
                            base.dtype,
                        )
                    else:
                        tensor = f.get_tensor(name)
                    writer.write(name, tensor)
                    index[name] = shard
            writer.close()
        copy_base_files(source, stage)  # then this merge's own index and config over the base's
        Path(stage, "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": dict(sorted(index.items()))}, indent=2)
        )
        merged_config = {k: v for k, v in config.items() if k != "quantization_config"}
        Path(stage, "config.json").write_text(json.dumps(merged_config, indent=2))
        if mtp.is_file():  # the MTP head, outside the index: as it is, copied a chunk at a time
            shutil.copyfile(mtp, Path(stage, mtp.name))


def _record_fits(state: Optional[str], expert_codec: str, base: str) -> None:
    """The run's recorded format must be one a run could hold of this base's experts
    (``formats.decide``): released of a base that stores them unquantized could not be (a run
    from before that was decided recorded released while it held NF4), nor any format of a base
    quantized on some layers only. Refused in words, nothing written."""
    from poros.core.errors import PorosConfigError
    from poros.moe.inkling2 import formats

    try:
        formats.decide(state, expert_codec)
    except PorosConfigError:
        raise PorosConfigError(
            f"This run records Inkling 2 Mini's experts held as {expert_codec}, and its base "
            f"{base} stores them "
            + (
                "unquantized, so it has no released bytes: a run on it held NF4 (one from "
                "before this release recorded released regardless), and its record cannot say "
                "which weights it trained against: no merge."
                if state == formats.UNQUANTIZED
                else "quantized on some layers only, which no run holds: no merge."
            ),
            remediation="Export the adapter alone, or train the run again on this release "
            "and merge that.",
        ) from None


def _nbytes(dtype: torch.dtype, shape: tuple) -> int:
    n = torch.empty((), dtype=dtype).element_size()
    for d in shape:
        n *= d
    return n


def _torch_dtype(name: str) -> torch.dtype:
    inverse = {v: k for k, v in _ST_DTYPES.items()}
    if name not in inverse:
        raise ValueError(f"merged export cannot copy a {name} tensor")
    return inverse[name]


def _merged_experts(
    source: str,
    shards: Dict[str, List[str]],
    parts: List[str],
    target: str,
    adapter: Dict[str, torch.Tensor],
    scaling: float,
    dtype: torch.dtype,
    expert_codec: str,
) -> torch.Tensor:
    """One layer's merged experts in the released layout (gate / up rows interleaved)."""
    from safetensors import safe_open

    from poros.train.export import merge_weight

    tensors = {}
    for name in parts:
        shard = next(s for s, ns in shards.items() if name in ns)
        with safe_open(os.path.join(source, shard), framework="pt") as f:
            tensors[name] = f.get_tensor(name)
    weight = codec.convert_state_dict(
        tensors.__getitem__, list(tensors), unpadded_vocab_size=0, expert_dtype=dtype
    )[target]
    module, _, proj = target.rpartition(".")
    a_name, b_name = _EXPERT_FACTORS[proj]
    prefix = module.rpartition(".experts")[0]
    a, b = adapter[f"{prefix}.adapter.{a_name}"], adapter[f"{prefix}.adapter.{b_name}"]
    merged = torch.stack(
        [
            merge_weight(_seen(weight[e], expert_codec, dtype), a[e], b[e], scaling, dtype)
            for e in range(weight.shape[0])
        ]
    )
    if proj == "gate_up_proj":  # back to the release's interleaved rows
        e, rows, cols = merged.shape
        merged = merged.view(e, 2, rows // 2, cols).transpose(1, 2).reshape(e, rows, cols)
    return merged.contiguous()
