"""A model's layout and what it would take, from its config alone: an empty skeleton and arithmetic.

A config may be anyone's. It builds only classes this install has, so no repo's code runs, and
``auto_map`` is ignored. Everything is on the meta device, buffers too, and counts are bounded
first, so a config cannot ask for memory. Every figure is arithmetic on shapes, labeled that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from poros.core.arch_registry import bounded
from poros.metal import experts
from poros.metal.pack import parts_of
from poros.moe import quant

GB = 1e9
# Keyed by whether a model is MoE. What a run holds of its adapter: itself, Adam's two moments and,
# dense, its grads. A dense adapter is fp32, and a MoE one computes in bf16.
_COPIES = {False: 4, True: 3}
_ADAPTER_BYTES = {False: 4, True: 2}
# The rest of a run, process and all: what the table's models come to, within about 10%.
_PROCESS_GB = {False: 1.0, True: 1.8}
# What no config has a say in: code, weights' format, and where it says it came from.
_IGNORED = ("auto_map", "custom_pipelines", "quantization_config", "_name_or_path")
_SHOWN_PATH = re.compile(r"[A-Za-z0-9_.]{1,120}")


@dataclass
class Layout:
    layers: str
    count: int
    linears: int  # a layer's, the largest
    moe: bool
    experts_per_layer: int
    experts_per_token: Optional[int]
    tied: bool
    embed_bytes: int
    head_bytes: int
    small_bytes: int
    adapter_bytes_16: int
    one_layer_bytes: int  # a layer's slab, the largest
    decoded_bytes: int  # what one layer decodes to at once, the most
    pack_bytes: int
    per_token_gb: float
    weights_bytes: int = 0  # every weight in bf16: what a release in bf16 comes to

    def floor_gb(self, seq: int, rank: int) -> float:
        adapter = _COPIES[self.moe] * self.adapter_bytes_16 * rank / 16
        return (_PROCESS_GB[self.moe] + (adapter + self.one_layer_bytes + self.decoded_bytes) / GB
                + self.per_token_gb * seq)


def known_type(raw: dict) -> bool:
    from poros.moe.carried import ensure_registered

    kind = raw.get("model_type")
    ensure_registered(raw)  # a family Poros carries (Inkling 2 Mini) registers itself first
    return isinstance(kind, str) and kind in CONFIG_MAPPING


def can_build(raw: dict) -> str:
    """Why no class of this install can build the model, or "". Remote code is never run."""
    if known_type(raw):
        return ""
    if raw.get("auto_map"):
        return "it needs its own code from its repo, and Poros never runs that"
    return "this install of transformers has no class for it"


def config_of(raw: dict, **over: Any) -> Any:
    """A class of this install's config, from ``raw`` less what no config has a say in."""
    from poros.moe.carried import ensure_registered

    bounded(raw)
    ensure_registered(raw)  # a family Poros carries (Inkling 2 Mini) registers itself first
    fields = {k: v for k, v in raw.items() if k != "model_type" and k not in _IGNORED}
    return AutoConfig.for_model(raw["model_type"], **{**fields, **over})


def empty(cfg: Any) -> nn.Module:
    """The model of ``cfg`` with nothing allocated, buffers too, and the global RNG where it was."""
    with torch.device("meta"), torch.random.fork_rng(devices=[]):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        experts.split(model)
    model.tie_weights()
    return model


def _shape(linear: nn.Linear) -> tuple[int, int]:
    return linear.out_features, linear.in_features


def _per_token(text: Any) -> Optional[int]:
    """A number or nothing: what a config says here goes straight into a result."""
    said = getattr(text, "num_experts_per_tok", None) or getattr(text, "top_k_experts", None)
    return said if isinstance(said, int) else None


def layout_of(raw: dict) -> Layout:
    """The layout in this config. ValueError if the model has no layers Poros can find."""
    cfg = config_of(raw)
    text = cfg.get_text_config()
    model = empty(cfg)
    parts = parts_of(model)
    moe, slabs, decoded, adapter, per_layer, whole = parts.moe, [], [], 0, 0, 0
    for layer, matrices in zip(model.get_submodule(parts.layers), parts.matrices):
        slab = plain = expert = 0
        for name in matrices:
            linear = layer.get_submodule(name)
            rows, columns = _shape(linear)
            whole += 2 * rows * columns
            if parts.raw(name):
                slab += 2 * rows * columns
                plain += 2 * rows * columns
            else:
                slab += quant.quantized_nbytes([rows, columns])
                expert = max(expert, 2 * rows * columns) if moe else 0
                plain += 0 if moe else 2 * rows * columns
                adapter += 16 * (rows + columns) * _ADAPTER_BYTES[moe]
        slabs.append(slab)
        decoded.append(plain + 2 * expert)
    if moe:
        per_layer = max((len(b.experts.gate_up_proj) for _, b in experts._sparse_blocks(model)),
                        default=0)
    rows = lambda m: 2 * m.weight.shape[0] * m.weight.shape[1]  # noqa: E731
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    embed_bytes, head_bytes = rows(embed), rows(head) if parts.head else 0
    state = model.state_dict()
    small = sum(2 * state[n].numel() for n in parts.small)
    token_bytes = 3 * 4 * text.vocab_size + 4 * text.hidden_size * len(slabs)
    return Layout(
        layers=parts.layers if _SHOWN_PATH.fullmatch(parts.layers) else "unknown", count=len(slabs),
        linears=max(map(len, parts.matrices)), moe=moe,
        experts_per_layer=per_layer,
        experts_per_token=_per_token(text) if moe else None,
        tied=parts.head is None, embed_bytes=embed_bytes, head_bytes=head_bytes,
        small_bytes=small, adapter_bytes_16=adapter, one_layer_bytes=max(slabs),
        decoded_bytes=max(decoded), pack_bytes=sum(slabs) + embed_bytes + head_bytes + small,
        per_token_gb=token_bytes / GB, weights_bytes=whole + embed_bytes + head_bytes + small,
    )


def first_that_fits(layout: Layout, pool_gb: float) -> Optional[dict[str, Any]]:
    """The longest rows at the highest rank whose floor the pool holds. Arithmetic, not a run."""
    for rank in (16, 8, 4, 2, 1):
        for seq in (256, 128, 64):
            if layout.floor_gb(seq, rank) <= pool_gb:
                return {"rank": rank, "seq_len": seq}
    return None
