"""Every training setting the /train card's "more" list offers, in words.

Each config field gets a plain name, the group it sits in, one line on what
it does, and the values worth picking. A field missing here still shows, by
its own name under "advanced": nothing in PorosConfig is unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Choices = Tuple[Tuple[str, str], ...]  # (value, what it means)


@dataclass(frozen=True)
class Setting:
    name: str  # the PorosConfig field
    label: str  # what the card calls it
    group: str
    about: str  # one line: what changing it does
    choices: Choices = ()  # values worth picking; any typed value still works
    words: str = ""  # other words people search for it by


GROUPS = ("learning", "LoRA", "memory & speed", "saving & checks", "data", "advanced")

YES_NO: Choices = (("true", "yes"), ("false", "no"))

SETTINGS: Tuple[Setting, ...] = (
    # -- learning ------------------------------------------------------------
    Setting("learning_rate", "learning rate", "learning", "how big each update is",
            (("5e-5", "careful"), ("1e-4", ""), ("2e-4", "SFT's"), ("5e-4", "fast, can diverge"),
             ("5e-6", "DPO's")), "lr"),
    Setting("lr_scheduler", "schedule", "learning", "how the rate falls after warmup",
            (("cosine", "eases down to zero"), ("linear", "falls straight"),
             ("constant", "stays put")), "lr scheduler decay"),
    Setting("warmup_steps", "warmup", "learning", "steps the rate ramps up over",
            (("0", "none"), ("10", ""), ("50", ""), ("100", ""))),
    Setting("gradient_accumulation_steps", "rows per step", "learning",
            "rows each update learns from; more is steadier, each step slower",
            (("4", ""), ("8", "as the recipes train"), ("16", ""), ("32", "")),
            "batch size grad accum gradient accumulation"),
    Setting("per_device_batch_size", "rows at once", "learning",
            "rows on the GPU together: faster, more memory",
            (("1", "as the recipes train"), ("2", ""), ("4", "")), "batch"),
    Setting("weight_decay", "weight decay", "learning", "pulls weights toward zero",
            (("0", "off"), ("0.01", ""), ("0.1", "strong"))),
    Setting("max_grad_norm", "clip gradients", "learning", "caps a step that would jump too far",
            (("0.5", "tighter"), ("1.0", ""), ("0", "off")), "max grad norm"),
    Setting("train_on", "learn from", "learning", "which tokens of a chat it learns",
            (("answers", "every answer"), ("last", "the last answer"),
             ("all", "every token, questions too")), "loss mask"),
    Setting("dpo_beta", "beta", "learning", "how far a pairs loop may move from the base",
            (("0.05", "further"), ("0.1", ""), ("0.2", ""), ("0.5", "stays close"))),
    Setting("seed", "seed", "learning", "the same seed trains the same run", (("42", ""),)),
    Setting("adam_beta1", "Adam beta1", "learning", "momentum of the optimizer"),
    Setting("adam_beta2", "Adam beta2", "learning", "how fast the optimizer forgets"),
    Setting("adam_epsilon", "Adam epsilon", "learning", "keeps the optimizer's steps finite"),
    # -- LoRA ----------------------------------------------------------------
    Setting("lora_alpha", "alpha", "LoRA", "how strongly the adapter speaks; 2× rank is usual",
            (("16", ""), ("32", "as the recipes train"), ("64", ""))),
    Setting("lora_dropout", "dropout", "LoRA", "drops adapter inputs to fight overfitting",
            (("0", "as the recipes train"), ("0.05", ""), ("0.1", ""))),
    Setting("lora_target_modules", "layers", "LoRA", "which projections get an adapter",
            (("all-linear", "every linear layer"),), "target modules"),
    Setting("lora_bias", "bias", "LoRA", "train the bias terms too",
            (("none", ""), ("lora_only", "the adapter's"), ("all", "every bias"))),
    Setting("init_adapter", "start from", "LoRA", "an adapter folder to keep training",
            words="resume adapter from"),
    # -- memory & speed --------------------------------------------------------
    Setting("path", "path", "memory & speed",
            "the training loop for a step's micro-batches (CUDA): same result",
            (("efficient", "the default: less VRAM than fast; block inputs in host RAM"),
             ("fast", "the fastest, more VRAM: block inputs and one block's weights on the GPU"),
             ("sequential", "one micro-batch at a time: slower, and its peak can be lower")),
            "fast efficient sequential speed vram block major loop"),
    Setting("torch_dtype", "precision", "memory & speed", "the dtype it computes in",
            (("bfloat16", "as the recipes train"), ("float16", "older GPUs"),
             ("float32", "slow, exact")),
            "dtype bf16 fp16"),
    Setting("quantization", "base weights", "memory & speed", "how the frozen base is stored",
            (("nf4", "4-bit, as the recipes train"),), "quantization qlora"),
    Setting("enable_prefetch", "prefetch", "memory & speed",
            "loads the next block while this one runs", YES_NO),
    Setting("offload_suffix_inputs_to_cpu", "offload inputs", "memory & speed",
            "keeps saved activations in RAM, not VRAM", YES_NO, "activations cpu"),
    Setting("attn_implementation", "attention", "memory & speed", "which attention kernel runs",
            (("sdpa", "PyTorch's"), ("flash_attention_2", "needs flash-attn"),
             ("eager", "plain"))),
    Setting("nf4_load_strategy", "load", "memory & speed", "how the base is read onto the GPU",
            (("auto", ""), ("block_streamed", "a block at a time"), ("gpu", "all at once"))),
    # -- saving & checks -----------------------------------------------------
    Setting("save_steps", "save every", "saving & checks", "steps between checkpoints",
            (("100", ""), ("250", ""), ("500", ""))),
    Setting("save_total_limit", "keep", "saving & checks", "checkpoints kept on disk",
            (("1", ""), ("3", ""), ("5", ""))),
    Setting("eval_steps", "check every", "saving & checks", "steps between held-out losses",
            (("50", ""), ("100", ""), ("500", "")), "eval"),
    Setting("eval_checkpoints", "check at saves", "saving & checks",
            "a held-out loss in each checkpoint, to pick the best: one pass over them a save",
            YES_NO, "eval checkpoint best validation"),
    Setting("logging_steps", "log every", "saving & checks", "steps between logged losses",
            (("1", ""), ("10", ""), ("50", ""))),
    Setting("memory_guard", "memory guard", "saving & checks",
            "stops a run before swap or a full boot disk hurts this Mac", YES_NO),
    Setting("push_to_hub", "push to Hub", "saving & checks", "upload the adapter when it ends",
            YES_NO, "huggingface upload"),
    Setting("hub_model_id", "Hub repo", "saving & checks", "where push to Hub uploads it",
            words="huggingface upload repo id"),
    # -- data ----------------------------------------------------------------
    Setting("eval_split", "held out", "data", "recipe runs: share of rows kept back to score",
            (("0", "none"), ("0.05", "5%"), ("0.1", "10%")), "eval validation"),
    Setting("eval_rows", "held-out rows", "data", "recipe runs: this many rows held out instead",
            (("0", "the share"), ("50", ""), ("100", ""), ("200", "")), "eval validation count"),
    Setting("eval_dataset", "eval data", "data", "a file of held-out rows, scored on any loop",
            words="validation"),
    Setting("model_revision", "model revision", "data", "a branch, tag or commit of the model"),
)

BY_NAME = {s.name: s for s in SETTINGS}


def setting(name: str, description: str = "") -> Setting:
    """The field in words; a field not described here, by its own name."""
    return BY_NAME.get(name) or Setting(name, name.replace("_", " "), "advanced",
                                        description or name)
