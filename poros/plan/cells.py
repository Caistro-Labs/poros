"""The receipts the plan reads (``CELLS``, from ``cells.json``; ``MAC_CELLS``, a Mac's, never mixed
with a CUDA card's), and the shapes that let one model's receipts plan another of the same shapes
(``SHAPES``, ``shape_twin``).

The one home of the receipt set: every part of the plan reads ``cells.CELLS``, ``cells.SHAPES``
and ``cells.shape_twin`` from this module at call time, so a narrowed set (the leave-one-out
check, a test: ``monkeypatch.setattr(cells, "CELLS", ...)``) applies to the whole plan.
Torch-free.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

from poros.core.arch_registry import registry_type

_ALL_CELLS: list[dict] = json.loads(
    resources.files("poros.plan").joinpath("cells.json").read_text()
)
# A figure never crosses between CUDA and a Mac, in either direction.
CELLS = [c for c in _ALL_CELLS if not c["key"]["gpu"].startswith("Apple")
         and not c.get("only_its_machine")]
MAC_CELLS = [c for c in _ALL_CELLS if c["key"]["gpu"].startswith("Apple")]
# A CUDA certification run's cells (only_its_machine): each answers its own card and no other
# machine's plan, twin or words, so they are read only through ``here``.
CERTIFIED = [c for c in _ALL_CELLS if not c["key"]["gpu"].startswith("Apple")
             and c.get("only_its_machine")]


def here(gpu_name: Optional[str]) -> list[dict]:
    """The cells a plan on the card ``gpu_name`` reads: CELLS, and that card's own CERTIFIED."""
    from poros.core.card import _norm

    return CELLS + [c for c in CERTIFIED
                    if gpu_name is not None and _norm(c["key"]["gpu"]) == _norm(gpu_name)]


# What of a config.json decides a run's memory: the plan's receipts of one model plan another
# whose config gives the same of each (a local folder of it, a fine-tune of it).
SHAPE_KEYS = ("num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
              "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings",
              "moe_intermediate_size", "num_experts", "n_routed_experts", "num_local_experts",
              "num_experts_per_tok", "top_k_experts", "n_shared_experts",
              "shared_expert_intermediate_size", "decoder_sparse_step", "mlp_only_layers",
              "layer_types")
SHAPES = json.loads(resources.files("poros.plan").joinpath("shapes.json").read_text())


def shape_signature(model_config: Optional[dict]) -> Optional[dict]:
    """The shapes of ``model_config`` (config.json) that decide a run's memory, with what the
    registry calls the model: two models with the same signature use the same VRAM at the same
    settings. None without a model_type or a layer count."""
    if not isinstance(model_config, dict):
        return None
    text = model_config.get("text_config")
    text = text if isinstance(text, dict) else model_config
    out = {"model_type": registry_type(model_config)}
    out.update({k: text.get(k) for k in SHAPE_KEYS})
    if out["model_type"] is None or not out["num_hidden_layers"]:
        return None
    return out


def prequantized(model_config: Optional[dict]) -> Optional[str]:
    """The quantizer a checkpoint's config.json says its weights are stored in (AWQ, GPTQ, FP8,
    bitsandbytes, ...: ``quantization_config.quant_method``), or None for an unquantized one."""
    if not isinstance(model_config, dict):
        return None
    text = model_config.get("text_config")
    for c in (model_config, text if isinstance(text, dict) else {}):
        q = c.get("quantization_config")
        if isinstance(q, dict):
            return str(q.get("quant_method") or "quantized")
        if q:
            return "quantized"
    return None


def _catalog_twin(model: str) -> Optional[str]:
    """The receipt model the catalog lists ``model`` as having the shapes of (a ``same-shape``
    entry's ``via``, an instruct of a base): what plans it where its config.json cannot be read,
    so a plan says the same online and offline."""
    from poros import catalog

    low = model.lower()
    row = next((m for m in catalog.raw()["model"] if m["id"].lower() == low), None)
    via = row.get("via") if row and row.get("evidence") == "same-shape" else None
    return via if via and any(c["key"]["model"] == via for c in CELLS) else None


def shape_twin(model: str, model_config: Optional[dict]) -> Optional[str]:
    """The receipt model whose config.json has ``model_config``'s shapes (``SHAPES``), for a
    model no receipt names (a local folder, a fine-tune): its receipts plan this one. None where
    receipts name ``model`` itself, no receipt model has its shapes, or its checkpoint is stored
    pre-quantized (``prequantized``: an AWQ / GPTQ / FP8 copy loads through another quantizer
    than the bf16 checkpoints the receipts trained, so their memory is not its)."""
    if any(c["key"]["model"] == model for c in CELLS):
        return None
    if prequantized(model_config):  # its weights load through another quantizer: other memory
        return None
    sig = shape_signature(model_config)
    if sig is None:  # no config.json was read (offline, not cached): the catalog's own record
        return _catalog_twin(model)
    # only a model with receipts here plans another (never itself: a twin names its own cells)
    twins = [m for m, entry in SHAPES["models"].items() if entry["signature"] == sig
             and m != model and any(c["key"]["model"] == m for c in CELLS)]
    # several (an instruct and its base): the one with the most receipts, then by name
    return min(twins, key=lambda m: (-sum(c["key"]["model"] == m for c in CELLS), m),
               default=None)
