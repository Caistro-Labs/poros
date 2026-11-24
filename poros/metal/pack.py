"""Safetensors to ``.sss`` on the CPU, one tensor at a time. No user step, nothing torn.

The model says what goes where: its skeleton, built from its config with no weights, is the only
table of names. ``<store>/<slug>-<rev8>/`` holds ``model.sss``, its ``.idx`` and
``small.safetensors``:

- one slab per decoder layer, key ``(layer, 0)``: an NF4 blob per ``nn.Linear`` in the layer, each
  on a 64-byte boundary. A MoE layer has two per expert (``poros.metal.experts``), and its others
  stay plain bf16, as on CUDA;
- the input embeddings at ``(EMBED_BLOCK, 0)`` and the output embeddings at ``(HEAD_BLOCK, 0)``, raw
  bf16. A tied model has no head slab: its head reads the embed's;
- ``small.safetensors``: all else the text backbone holds, in bf16, as a bf16 load rounds it.

Nothing in a text forward reaches what lies outside the backbone and the head, a vision tower or a
draft head, so it is left behind. A tensor the model has no name for, and does not say to ignore,
is refused. A slab's ``views`` say what is in it. ``SssWriter`` commits the index last, so a killed
pack leaves no store that looks finished, and a store is reused only while its meta matches.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from poros.core.arch_registry import discover_layers
from poros.core.checkpoints import shard_name
from poros.metal import experts
from poros.moe import quant
from poros.sss.format import SssFile, SssFormatError, SssWriter

PACK_VERSION = 1
BLOB_ALIGN = 64
# Slab keys outside the layer numbering.
EMBED_BLOCK = -1
HEAD_BLOCK = -2
_ROWS_PER_WRITE = 16384  # macOS refuses a single write of 2 GiB, and 72B's embed is 2.5 GB


def pack_dir(store: str | os.PathLike, model_id: str, revision: str) -> Path:
    """``<store>/<slug>-<rev8>``."""
    return Path(store) / f"{model_id.replace('/', '--')}-{revision[:8]}"


def skeleton(cfg: Any) -> nn.Module:
    """The model of ``cfg`` with no weights, tied as it ties them. Built under bf16 as a bf16 load
    builds it, since Gemma's embedding scale is rounded that way, and the global RNG never moves:
    Qwen3.5 draws in its constructor, and a pack that is reused must not differ from a fresh one."""
    with torch.random.fork_rng(devices=[]), init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        experts.split(model)
    model.tie_weights()
    return model


class Parts(NamedTuple):
    """Where a model keeps what the Mac path holds, in the model's own names."""

    layers: str  # ``model.layers``
    matrices: list[list[str]]  # per layer, the path of each linear inside it
    embed: str  # ``model.embed_tokens.weight``
    head: Optional[str]  # ``lm_head.weight``. None: tied
    small: list[str]  # the backbone's other parameters and persistent buffers
    moe: bool  # its experts are NF4, and every other linear stays bf16

    def raw(self, matrix: str) -> bool:
        return self.moe and "experts" not in matrix.split(".")

    @property
    def backbone(self) -> str:
        return self.layers.rpartition(".")[0] + "."

    @property
    def weights(self) -> set[str]:
        """The tensor behind each matrix."""
        return {f"{self.layers}.{i}.{m}.weight" for i, ms in enumerate(self.matrices) for m in ms}


def parts_of(model: nn.Module) -> Parts:
    container, layers = discover_layers(model)
    names = {module: name for name, module in model.named_modules()}
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    parts = Parts(
        layers,
        [[n for n, m in layer.named_modules() if isinstance(m, nn.Linear)] for layer in container],
        f"{names[embed]}.weight",
        None if head.weight is embed.weight else f"{names[head]}.weight",
        [],
        experts.is_moe(model),
    )
    held = parts.weights | {parts.embed}
    return parts._replace(small=[n for n in model.state_dict()
                                 if n.startswith(parts.backbone) and n not in held])


class Stored(NamedTuple):
    """One tensor in a checkpoint."""

    shard: Path
    name: str
    shape: list[int]
    dtype: str


def _read(stored: Stored | experts.Fused | experts.Row) -> torch.Tensor:
    if isinstance(stored, experts.Fused):
        return torch.cat([_read(half) for half in stored.halves])
    if isinstance(stored, experts.Row):
        with safe_open(stored.whole.shard, framework="pt") as f:
            part = f.get_slice(stored.whole.name)[stored.index : stored.index + 1]
            return part[0].to(torch.bfloat16)
    with safe_open(stored.shard, framework="pt") as f:
        return f.get_tensor(stored.name).to(torch.bfloat16)


def _tensors(source: Path, to_model_name: Callable[[str], str]) -> dict[str, Stored]:
    """The model's name for each tensor -> where it is stored, from the shard headers alone."""
    index = source / "model.safetensors.index.json"
    names = json.loads(index.read_text())["weight_map"].values() if index.exists() else None
    tensors = {}
    for name in sorted(set(names or ["model.safetensors"])):
        with safe_open(source / shard_name(name), framework="pt") as f:
            for tensor in f.keys():
                part = f.get_slice(tensor)
                tensors[to_model_name(tensor)] = Stored(source / name, tensor, part.get_shape(),
                                                        part.get_dtype())
    return tensors


def _refuse_what_has_no_place(source: Path, tensors: dict, model: nn.Module, parts: Parts) -> None:
    state, buffers = model.state_dict(), dict(model.named_buffers())
    ignored = model._keys_to_ignore_on_load_unexpected or ()
    lost = [n for n in tensors if n not in state and not any(re.search(p, n) for p in ignored)]
    lost += [n for n, m in model.named_modules() if isinstance(m, nn.Linear)
             and n.startswith(parts.backbone) and f"{n}.weight" not in parts.weights]
    if getattr(model.get_output_embeddings(), "bias", None) is not None:
        lost.append("the head's bias")
    rows = {parts.embed, parts.head} - {None}
    placed = parts.weights | rows | set(parts.small)
    missing = {n for n in placed if n not in tensors and n not in buffers}
    if lost or missing:
        raise ValueError(f"cannot pack {source}: no place for {sorted(lost)[:3]}, "
                         f"and it lacks {sorted(missing)[:3]}")
    for name in sorted(placed & tensors.keys()):
        stored = tensors[name]
        if stored.shape != list(state[name].shape):
            raise ValueError(f"cannot pack {source}: {name} is {stored.shape}, "
                             f"and its config says {list(state[name].shape)}")
        if stored.dtype != "BF16" and not (stored.dtype == "F32" and name in parts.small):
            raise ValueError(f"cannot pack {source}: {name} is {stored.dtype}, "
                             "and a Mac store holds bf16 only")


def _layout(tensors: dict, parts: Parts) -> tuple[list, dict, dict]:
    """Slab sizes, kinds and views, from shapes alone."""
    slabs, kinds, views = [], {}, {}
    for layer, matrices in enumerate(parts.matrices):
        pos, blobs = 0, []
        for matrix in matrices:
            shape = tensors[f"{parts.layers}.{layer}.{matrix}.weight"].shape
            pos = -(-pos // BLOB_ALIGN) * BLOB_ALIGN
            raw = parts.raw(matrix)
            blobs.append({"name": matrix, "offset": pos, "shape": shape,
                          **({"raw": True} if raw else {})})
            pos += 2 * shape[0] * shape[1] if raw else quant.quantized_nbytes(shape)
        slabs.append(((layer, 0), pos))
        kinds[(layer, 0)], views[(layer, 0)] = "layer", blobs
    for block, name in ((EMBED_BLOCK, parts.embed), (HEAD_BLOCK, parts.head)):
        if name is not None:
            rows, hidden = tensors[name].shape
            slabs.append(((block, 0), rows * hidden * 2))
            kinds[(block, 0)] = "rows"
            views[(block, 0)] = [{"name": name, "offset": 0, "shape": [rows, hidden]}]
    return slabs, kinds, views


def _is_current(path: Path, meta: dict) -> bool:
    try:
        with SssFile.open(path) as f:
            return f.matches(meta)
    except (SssFormatError, OSError):
        return False


def _meta(model_id: str, revision: str) -> dict:
    return {"pack": PACK_VERSION, "model": model_id, "revision": revision,
            "torch": torch.__version__}


def finished(store: str | os.PathLike, model_id: str, revision: str) -> Optional[Path]:
    """This model's pack if it is there and current. Its shards may be long gone."""
    out = pack_dir(store, model_id, revision)
    return out if _is_current(out / "model.sss", _meta(model_id, revision)) else None


def pack(source: str | os.PathLike, store: str | os.PathLike, *, model_id: str,
         revision: str) -> Path:
    """Build the store for one model, or reuse it. Returns ``pack_dir(...)``."""
    done = finished(store, model_id, revision)
    if done:
        return done
    source = Path(source)
    from poros.moe.carried import ensure_registered

    ensure_registered(source)  # a family Poros carries registers itself first
    model = skeleton(AutoConfig.from_pretrained(source, trust_remote_code=False))
    parts = parts_of(model)
    tensors = experts.fuse(_tensors(source, experts.checkpoint_names(model)))
    _refuse_what_has_no_place(source, tensors, model, parts)
    slabs, kinds, views = _layout(tensors, parts)
    meta = _meta(model_id, revision)
    out = pack_dir(store, model_id, revision)

    def read(name: str) -> torch.Tensor:
        return _read(tensors[name])

    with SssWriter(out / "model.sss", slabs, meta=meta, kinds=kinds, views=views) as writer:
        for key, nbytes in slabs:
            if kinds[key] == "layer":
                slab = torch.zeros(nbytes, dtype=torch.uint8)
                for view in views[key]:
                    weight = read(f"{parts.layers}.{key[0]}.{view['name']}.weight")
                    blob = (weight.flatten().view(torch.uint8) if view.get("raw")
                            else quant.quantize_tensor(weight))
                    slab[view["offset"] : view["offset"] + blob.numel()] = blob
                writer.write(key, 0, slab)
            else:
                name = views[key][0]["name"]
                rows, hidden = tensors[name].shape
                with safe_open(tensors[name].shard, framework="pt") as f:
                    matrix = f.get_slice(tensors[name].name)
                    for row in range(0, rows, _ROWS_PER_WRITE):
                        part = matrix[row : row + _ROWS_PER_WRITE].view(torch.uint8)
                        writer.write(key, row * hidden * 2, part)
        # Before the commit, so an index never comes without it.
        state = model.state_dict()
        small = {n: read(n) if n in tensors else state[n] for n in sorted(parts.small)}
        save_file(small, out / "small.safetensors")
        writer.finish().close()
    return out
