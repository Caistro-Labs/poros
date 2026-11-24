"""What a load fetches of a model's repo: named, as far as a name can be known. Whatever else a repo
holds stays on the Hub, its code above all, and the size a person agrees to is the size of these."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# A config's and a tokenizer's files, and a license, which goes with any copy of the weights. A
# merged model keeps these of its base, by these names.
BASE_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json",
              "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
              "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
              "processor_config.json", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE")
# No more than these for what a pack needs before its shards, and for a model that has its pack.
SMALL_FILES = [*BASE_FILES, "vocab.txt", "additional_chat_templates/*.jinja"]
# An index names a model's shards, and only by reading it could they be named here.
HUB_FILES = [*SMALL_FILES, "*.safetensors"]


# What of a run's poros_config.json goes with an adapter when it leaves the run: the rest is this
# machine's, its home folder and where its data is.
PORTABLE_FIELDS = ("model_name_or_path", "model_revision", "quantization", "torch_dtype",
                   "adapter_backend", "lora_rank", "lora_alpha", "lora_dropout",
                   "lora_target_modules", "lora_bias", "moe_expert_format")


def portable_manifest(output_dir: "str | Path") -> Optional[bytes]:
    """A run's poros_config.json reduced to its portable adapter fields."""
    try:
        saved = json.loads((Path(output_dir) / "poros_config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict):
        return None
    return json.dumps({k: saved[k] for k in PORTABLE_FIELDS if k in saved}, indent=2).encode()
