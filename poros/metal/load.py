"""A model on the Mac: a skeleton with no weights, frozen linears that decode, rows read on demand.

No bitsandbytes and no kbit preparation: ``Nf4Linear`` is what PEFT wraps, and the block loop gets
the bytes from ``model.poros_source``. What ``poros.metal.pack`` left behind, a vision tower, stays
empty, so it is taken off the model before PEFT can hang an adapter on it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig

from poros.cli.words.models import by_id
from poros.core.hub_files import HUB_FILES, SMALL_FILES
from poros.metal.head import ChunkedHead, EmbedRows, FileRows
from poros.metal.nf4 import Nf4Linear
from poros.metal.pack import EMBED_BLOCK, HEAD_BLOCK, finished, pack, parts_of, skeleton
from poros.metal.source import FileLayers, RamLayers
from poros.metal.store import internal_store
from poros.moe.quant import quantized_nbytes
from poros.sss.format import SssFile, SssFormatError


def _drop_what_is_empty(module: nn.Module) -> bool:
    """Whatever still has no weights is left off the model. True if anything was."""
    dropped = False
    for name, child in module.named_children():
        params = [*child.parameters()]
        if params and all(p.is_meta for p in params):
            setattr(module, name, None)
            dropped = True
        else:
            dropped = _drop_what_is_empty(child) or dropped
    return dropped


def _check_view(view: dict, linear: Any, slab_nbytes: int) -> None:
    """A pack's sizes against the config's, as the scratch comes from the one and a kernel writes
    what the other says. A pack may be anyone's: past either bound the GPU reads or writes out."""
    n = linear.in_features * linear.out_features
    size = 2 * n if view.get("raw") else quantized_nbytes((n,))
    if view["shape"][0] * view["shape"][1] != n or not 0 <= view["offset"] <= slab_nbytes - size:
        raise SssFormatError("a pack's view does not match this model's config")


def build(pack_path: Path, cfg: Any, *, file_rows: bool = False,
          device: torch.device = torch.device("cpu")) -> nn.Module:
    """The skeleton of ``cfg``, its small tensors on ``device``, its layers waiting for their
    bytes. ``file_rows`` leaves embed and head in the file; else they are resident on the CPU."""
    file = SssFile.open(pack_path / "model.sss")
    model = skeleton(cfg)
    parts = parts_of(model)
    small = {k: v.to(device) for k, v in load_file(pack_path / "small.safetensors").items()}

    largest = max(
        v["shape"][0] * v["shape"][1] for e in file.entries if e.kind == "layer" for v in e.views
    )
    scratch = torch.empty(largest, dtype=torch.bfloat16, device=device)
    for index, layer in enumerate(model.get_submodule(parts.layers)):
        for view in file.entry((index, 0)).views:
            parent, _, name = view["name"].rpartition(".")
            owner = layer.get_submodule(parent)
            old = getattr(owner, name)
            _check_view(view, old, file.entry((index, 0)).nbytes)
            setattr(owner, name, Nf4Linear(
                old.in_features, old.out_features, scratch=scratch, start=view["offset"],
                bias=small.pop(f"{parts.layers}.{index}.{view['name']}.bias", None),
                raw=view.get("raw", False),
            ))
    for name, tensor in small.items():
        parent, _, attr = name.rpartition(".")
        owner = model.get_submodule(parent)
        if isinstance(owner, nn.Conv1d) and not parts.moe:
            tensor = tensor.to(torch.float32)  # it casts its input to its weight's dtype
        setattr(owner, attr, tensor if attr in owner._buffers
                else nn.Parameter(tensor, requires_grad=False))

    def rows(block: int) -> Any:
        found = FileRows(file, (block, 0))
        return found if file_rows else found[0 : found.shape[0]]

    embed = rows(EMBED_BLOCK)
    scale = getattr(model.get_input_embeddings(), "embed_scale", None)
    model.set_input_embeddings(EmbedRows(embed, scale=None if scale is None else float(scale)))
    model.set_output_embeddings(ChunkedHead(rows(HEAD_BLOCK) if parts.head else embed))
    if parts.moe:  # it computes in bf16, as on CUDA
        model.get_input_embeddings().to(torch.bfloat16)
        model.get_output_embeddings().to(torch.bfloat16)
    if _drop_what_is_empty(model):
        # A vision tower has a q_proj of its own, and Gemma 4's is a class PEFT refuses, so what an
        # adapter says it targets has to name the text layers for a load on the whole wrapper.
        inside = sorted({re.escape(m) for matrices in parts.matrices for m in matrices})
        model.poros_lora_targets = rf"{re.escape(parts.layers)}\.\d+\.({'|'.join(inside)})"
    return model


def fetch(config: Any, *, weights: bool = False) -> tuple[Path, str, str]:
    """(the model's files, its id, its revision), its shards too only if ``weights``: a pack needs
    none of them. A local directory's revision is its shards' sizes and times, so a rewritten
    checkpoint is packed again. A Hub id the catalog does not list is anyone's repo, fetched as
    anyone fetches it: with no token."""
    name = config.model_name_or_path
    if not Path(name).is_dir():
        anyones = {} if by_id(name) else {"token": False}  # a catalog model's call is as it was
        files = Path(snapshot_download(name, revision=config.model_revision,
                                       allow_patterns=HUB_FILES if weights else SMALL_FILES,
                                       **anyones))
        return files, name, files.name
    files = Path(name)
    shards = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files.glob("*.safetensors")]
    return files, files.name, hashlib.blake2b(repr(sorted(shards)).encode()).hexdigest()


def load(config: Any, device: torch.device) -> nn.Module:
    """Download if needed, pack if needed, then the skeleton and its source of bytes."""
    files, model_id, revision = fetch(config)
    store = config.store_path or internal_store()
    packed = finished(store, model_id, revision) or pack(
        fetch(config, weights=True)[0], store, model_id=model_id, revision=revision)
    from poros.moe.carried import ensure_registered

    ensure_registered(files)  # a family Poros carries registers itself first
    cfg = AutoConfig.from_pretrained(files, attn_implementation="eager", use_cache=False,
                                     trust_remote_code=False)
    on_disk = config.store == "ssd"
    model = build(packed, cfg, file_rows=on_disk, device=device)
    model.poros_source = (FileLayers if on_disk else RamLayers)(packed / "model.sss", device)
    return model
