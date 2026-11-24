"""The model families whose code Poros carries itself, registered with transformers on demand.

transformers at Poros's pin has no class for these (Inkling 2 Mini: ``inkling2_loop`` and its
text decoder ``inkling2_loop_text``). Anything that turns such a config into a transformers
config or model asks ``ensure_registered`` first, with the model type or the config.json (a dict
or a checkpoint folder), so the answer never depends on what happened to be imported before.
Torch-free until a carried type is asked about.
"""

from __future__ import annotations

import json
import os
from importlib import import_module
from typing import Any

__all__ = ["CARRIED", "LOOPED", "declares_visits", "ensure_registered"]

#: model_type -> the module whose ``register()`` registers its config and model classes.
CARRIED = {
    "inkling2_loop": "poros.moe.inkling2.config",
    "inkling2_loop_text": "poros.moe.inkling2.config",
}


#: The carried families whose decoder runs its layers as visits, as their config classes declare
#: (``poros_visits``): what ``declares_visits`` says where transformers cannot be imported to ask
#: them (tests/moe/test_inkling2.py holds the two to each other).
LOOPED = frozenset({"inkling2_loop", "inkling2_loop_text"})


def _model_type(source: Any) -> Any:
    if isinstance(source, str) and source in CARRIED:
        return source
    if isinstance(source, dict):
        return source.get("model_type")
    if isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
        try:
            with open(os.path.join(source, "config.json")) as f:
                return json.load(f).get("model_type")
        except (OSError, ValueError):
            return None
    return getattr(source, "model_type", None)


def ensure_registered(source: Any) -> bool:
    """Register the classes of a carried family with transformers' Auto classes if ``source`` (a
    model_type, a config.json dict, a checkpoint folder or a config) is one; idempotent. Returns
    whether it was a carried family."""
    kind = _model_type(source)
    module = CARRIED.get(kind) if isinstance(kind, str) else None  # a config may say anything
    if module is None:
        return False
    import_module(module).register()
    return True


def declares_visits(source: Any) -> bool:
    """Whether the decoder of ``source`` (a model_type, a config.json dict, a checkpoint folder
    or a config) runs its layers as visits, as its config class declares (``poros_visits``): a
    looped decoder, which a path that runs each layer once a forward (the Mac's) cannot train."""
    kind = _model_type(source)
    if not isinstance(kind, str) or kind not in CARRIED:
        return False
    try:
        module = import_module(CARRIED[kind])
    except ImportError:  # no transformers here: the families' own declaration, as recorded
        return kind in LOOPED
    return bool(getattr(module.CONFIG_CLASSES[kind], "poros_visits", False))
