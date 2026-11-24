"""The models Poros names (``poros/catalog.toml``), read once: ``raw`` (the file as a table) and
``names`` (whether it names a model). The library reads it here; the CLI's words for those models
and their entries are ``poros.cli.words.models``. Torch-free.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def raw() -> dict:
    return tomllib.loads(resources.files("poros").joinpath("catalog.toml").read_text())


def names(model_id: str) -> bool:
    """Whether the catalog names ``model_id`` (its Hub id, any case)."""
    low = model_id.lower()
    return any(m["id"].lower() == low for m in raw()["model"])
