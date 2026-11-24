"""Every dense family as a tiny random checkpoint, written the way its real one is. Its dense
reference is ``poros.metal.tiny``'s."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Qwen2Config,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3Config,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from poros.metal.tiny import DenseHead, DenseLora, dense_reference, shrink  # noqa: F401

VOCAB = 40000  # the head has three chunks, so its sum order is a real question
SMALL_VOCAB = 2000  # for what does not ask that, since temporary space is not free
SHARED = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=64, use_cache=False,
              attn_implementation="eager")
_GEMMA3 = dict(SHARED, head_dim=8, query_pre_attn_scalar=8, sliding_window=4,
               rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                                "full_attention": {"rope_type": "default", "rope_theta": 1e6}})
_GEMMA4 = dict(SHARED, head_dim=8, global_head_dim=8, num_global_key_value_heads=2,
               hidden_size_per_layer_input=0, num_kv_shared_layers=0, sliding_window=4,
               enable_moe_block=False,
               layer_types=["sliding_attention", "full_attention"] * 2)
_QWEN3_5 = dict(SHARED, head_dim=8, linear_num_value_heads=8, linear_num_key_heads=2,
                linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3,
                layer_types=["linear_attention", "full_attention"] * 2)


def _qwen3_5_vision() -> Qwen3_5VisionConfig:
    return Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=16, num_heads=2,
                               out_hidden_size=SHARED["hidden_size"])


def _siglip() -> SiglipVisionConfig:
    return SiglipVisionConfig(hidden_size=16, intermediate_size=16, num_hidden_layers=1,
                              num_attention_heads=2, image_size=8, patch_size=4)


def _gemma4_vision() -> Gemma4VisionConfig:
    return Gemma4VisionConfig(hidden_size=16, intermediate_size=16, num_hidden_layers=1,
                              num_attention_heads=2, head_dim=8, global_head_dim=8,
                              num_key_value_heads=2)


def config(family: str, *, tied: bool = True, softcap=None, vocab: int = VOCAB):
    """One family. The wrappers keep their vision tower, as the real ones do."""
    text = dict(tie_word_embeddings=tied, vocab_size=vocab)
    if softcap is not None:
        text["final_logit_softcapping"] = softcap
    if family == "qwen2":
        return Qwen2Config(**SHARED, **text)
    if family == "qwen3":
        return Qwen3Config(**SHARED, head_dim=8, **text)
    if family == "qwen3_5_text":
        return Qwen3_5TextConfig(**_QWEN3_5, **text)
    if family == "qwen3_5":
        return Qwen3_5Config(text_config=Qwen3_5TextConfig(**_QWEN3_5, **text),
                             vision_config=_qwen3_5_vision(), attn_implementation="eager",
                             tie_word_embeddings=tied)
    if family == "gemma3_text":
        return Gemma3TextConfig(**_GEMMA3, **text)
    if family == "gemma3":
        return Gemma3Config(text_config=Gemma3TextConfig(**_GEMMA3, **text),
                            vision_config=_siglip(), attn_implementation="eager",
                            mm_tokens_per_image=4, image_token_index=vocab + 1,
                            tie_word_embeddings=tied)
    if family == "gemma4_text":
        return Gemma4TextConfig(**_GEMMA4, **text)
    if family == "gemma4":
        return Gemma4Config(text_config=Gemma4TextConfig(**_GEMMA4, **text),
                            vision_config=_gemma4_vision(), audio_config=None,
                            attn_implementation="eager", tie_word_embeddings=tied)
    if family == "qwen3_8":
        return real_config("qwen3_8_27b", vocab=vocab, tie_word_embeddings=tied)
    raise ValueError(family)


_WRAPPERS = {"qwen3_5": Qwen3_5ForConditionalGeneration, "gemma3": Gemma3ForConditionalGeneration,
             "gemma4": Gemma4ForConditionalGeneration, "qwen3_8": Qwen3_5ForConditionalGeneration}


def real_config(name: str, *, hidden: int = SHARED["hidden_size"], vocab: int = VOCAB, **text):
    """A real model's own config.json (``real_configs/``), at sizes a test can hold."""
    raw = json.loads((Path(__file__).parent / "real_configs" / f"{name}.json").read_text())
    return shrink(raw, hidden=hidden, vocab=vocab, **text)


def write_fused(path, cfg, *, dtype: torch.dtype = torch.bfloat16, seed: int = 1) -> None:
    """A wrapper's checkpoint as the newest real ones are: vision tower and all, experts stored
    fused under their fused names, in two shards, and Qwen3.6's draft head beside them."""
    torch.manual_seed(seed)
    model = AutoModelForImageTextToText.from_config(cfg)
    with torch.no_grad():
        for param in model.parameters():  # fresh init has unit norms and zero routers
            param.normal_(0.0, 0.05)
    tensors = {n: t.to(dtype).clone() for n, t in model.state_dict().items()
               if not (n == "lm_head.weight" and cfg.get_text_config().tie_word_embeddings)}
    if cfg.model_type == "qwen3_5_moe":
        tensors["mtp.fc.weight"] = tensors[next(n for n in tensors if "layernorm" in n)].clone()
        for name in [n for n in tensors if ".layers.0." in n and ".experts." in n]:
            tensors[f"mtp.layers.0.{name.split('.0.', 1)[1]}"] = tensors[name].clone()
    path.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(path)
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2],
              "model-00002-of-00002.safetensors": names[1::2]}
    for shard, inside in shards.items():
        save_file({n: tensors[n] for n in inside}, path / shard, metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {n: s for s, ns in shards.items() for n in ns}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


def write_checkpoint(path, family: str, cfg, *, seed: int = 1, **save_kwargs) -> None:
    """A bf16 checkpoint with vanilla names, saved as transformers saves this family's real one.

    Fresh init has zero biases and unit norms, so nothing could see them. Qwen3.5's real shards
    keep ``A_log`` and the gated norm in fp32, at values bf16 cannot hold. Qwen3.8's are all bf16.
    """
    torch.manual_seed(seed)
    cls = _WRAPPERS.get(family)
    model = cls(cfg) if cls else AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith(".bias"):
                tensor.normal_(0.0, 0.05)
            elif "norm" in name or "layer_scalar" in name:
                tensor.add_(0.2 * torch.randn_like(tensor))
    model.to(torch.bfloat16)
    for name, param in model.named_parameters():
        if family != "qwen3_8" and name.endswith(("A_log", "linear_attn.norm.weight")):
            param.data = (param.data.float() + 0.123456789).to(torch.float32)
    model.save_pretrained(path, **save_kwargs)
