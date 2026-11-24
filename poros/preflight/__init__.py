"""Can this run start: every decision taken before a model downloads, in one package.

``poros train`` (``poros.cli.train_cmd``, ``poros.cli.commands.train._plan_gate``),
``PorosTrainer.from_pretrained`` and ``poros.Run`` call these; no other module holds a copy of a
rule.

==================  ===========================================================================
``card``            what the GPU is and runs: ``poros.core.card`` (``check``: the torch build,
                    bf16 -- ``E_STACK_UNSUPPORTED`` / ``E_GPU_UNSUPPORTED``; the rank fit;
                    flash). The Mac's: ``mac.card``.
``arch``            the model's family and the architecture gate (``E_ARCH_UNSUPPORTED``)
``compute``         a set fused / expert compute that cannot run here (``E_CONFIG``)
``fit``             the path and the tables adapted to fit, then the GPU (``E_OOM_PREDICTED``)
                    and host RAM refusals
``data``            the dataset a run reads: missing, unreadable, rows that stop a run
``disk``            a download larger than the free disk (``E_DISK``)
``resume``          a checkpoint whose adapter cannot load into this run
``command``         ``poros train``'s checks: each refusal worded once, and the order it takes them
``start``           the order PorosTrainer takes them in, and its route (dense / MoE)
``mac``             the Mac's branch: what a Mac checks instead (no CUDA arithmetic)
==================  ===========================================================================

The submodules load on first use (``poros.preflight.fit`` imports the plan; the rest are light).
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = ("arch", "command", "compute", "data", "disk", "fit", "mac", "resume", "start")


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"poros.preflight.{name}")
    raise AttributeError(f"module 'poros.preflight' has no attribute {name!r}")
