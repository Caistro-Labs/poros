"""The machine state a sentence resolves against: GPU, cache, runs, the TTY.

Every fact is read lazily and once, so a sentence that needs none of them
costs nothing, and tests build a Context with the facts fixed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional

from poros.cli.schemas import PlanGpu


def weights_cached(model_id: str, hub: Optional[str] = None) -> bool:
    """Every weight file of ``model_id`` is in the Hugging Face cache (``hub``, if named)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    import json

    def cached(name: str):
        hit = try_to_load_from_cache(model_id, name, cache_dir=hub)
        return hit if isinstance(hit, str) else None

    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = cached(index)
        if path:
            try:
                shards = set(json.loads(Path(path).read_text())["weight_map"].values())
            except (OSError, ValueError, KeyError):
                return False
            return bool(shards) and all(cached(s) for s in shards)
    return bool(cached("model.safetensors") or cached("pytorch_model.bin"))


def home() -> Path:
    return Path(os.environ.get("POROS_HOME") or Path.home() / ".poros")


@dataclass
class Context:
    cwd: Path = field(default_factory=Path.cwd)
    gpu_name: Optional[str] = None  # --gpu NAME: plan for another card
    fixed_gpu: Optional[PlanGpu] = None  # tests
    fixed_cached: Optional[frozenset[str]] = None  # tests
    fixed_interactive: Optional[bool] = None  # tests
    sheet: bool = False  # settled on a settings sheet: no start screen after it

    @cached_property
    def gpu(self) -> Optional[PlanGpu]:
        if self.fixed_gpu is not None:
            return self.fixed_gpu
        from poros.cli.ui import simulate
        from poros.core.card import detect_gpu

        if simulate.enabled() and self.gpu_name is None:
            mac = simulate.this_mac()
            if mac is not None:  # a Mac previews as itself: its chip, its one pool
                return mac
            name, total = simulate.GPU
            return PlanGpu(name=name, total_gib=total, free_gib=None, source="spec")
        return detect_gpu(self.gpu_name)

    @cached_property
    def quiet_gpu(self) -> Optional[PlanGpu]:
        """The GPU as a reader beside a run should ask: taking none of the card, or a torch."""
        from poros.cli.ui import simulate
        from poros.core.card import detect_gpu

        if self.fixed_gpu is not None or simulate.enabled():
            return self.gpu
        return detect_gpu(quietly=True)

    @property
    def unified(self) -> bool:
        """This is a Mac's one pool. Off Darwin it asks no GPU, unless a test fixed one."""
        if sys.platform != "darwin" and self.fixed_gpu is None:
            return False
        return self.gpu is not None and self.gpu.unified

    @cached_property
    def cached(self) -> frozenset[str]:
        """Catalog ids this machine holds what a run needs of: their weights in the Hugging Face
        cache, and on a Mac in any store's, or their finished pack, as `train` reads them. A
        config.json alone does not count: Poros fetches it to read a model's shape."""
        if self.fixed_cached is not None:
            return self.fixed_cached
        from poros.cli.words.models import catalog

        if self.unified:
            from poros.metal.store import ready

            return frozenset(e.id for e in catalog() if ready(e.id))
        return frozenset(e.id for e in catalog() if weights_cached(e.id))

    @property
    def interactive(self) -> bool:
        if self.fixed_interactive is not None:
            return self.fixed_interactive
        from poros.cli.questions import interactive

        return interactive()

    @cached_property
    def host_free_gib(self) -> Optional[float]:
        if self.fixed_gpu is not None or self.gpu_name is not None:
            return None  # planning for another machine: this one's RAM says nothing
        from poros.core.env import host_ram_free

        return host_ram_free()
