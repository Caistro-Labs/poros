"""Which parameters are an adapter's, by name.

Poros trains a frozen base with small adapters (PEFT LoRA or native LoRA). The check of a saved
PEFT adapter (``poros.train.adapters.validate_peft_adapter_state``) tells the adapter tensors it
must hold from the base copies it may hold beside them by name; this module is that rule.
Import-safe without torch.
"""

from __future__ import annotations

from typing import Any

# Substrings that mark a parameter as belonging to an adapter (not frozen base).
_ADAPTER_NAME_KEYWORDS: tuple[str, ...] = (
    "lora_A",
    "lora_B",
    "lora_a",
    "lora_b",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude",  # DoRA magnitude vector
    "adapter",
    "ia3_l",
    "ia3",
    "loha_",
    "lokr_",
    "vera_",
    "oft_",
    "boft_",
    "prompt_embeddings",
    "prefix",
)


def is_adapter_param(name: str, param: Any = None) -> bool:
    """Heuristic: does parameter ``name`` belong to an adapter (vs frozen base)?

    Name-based and deliberately broad: it splits adapter tensors from base copies, and is never
    the ground for a parity claim.
    """
    return any(kw in name for kw in _ADAPTER_NAME_KEYWORDS)
