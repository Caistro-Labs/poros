"""A Triton launch that fits the card's shared memory (CUDA; imported with the kernels).

A launch configuration's shared memory grows with its pipeline stages: each stage holds another
copy of a loop's loaded tiles. Cards differ in what one block may take (99 KB on sm_86 / sm_89 /
sm_120, 64 KB on sm_75, 163 KB on sm_80, 227 KB on sm_90: ``poros.core.card.shared_memory_limit``),
so a configuration measured on one can exceed another's. Triton then raises ``OutOfResources``
when it loads the compiled kernel, before anything runs. ``launch`` retries with one stage fewer
until it fits, and remembers the stages per (kernel, constexprs, device). Pipelining only decides
when a loop's next tiles are loaded, never what is summed or in which order, so a kernel's values
are the same at any stage count (tests/gpu/test_xcard_kernels.py checks it bit for bit). A
configuration that does not fit at one stage raises as Triton does; ``poros.core.fused.fit`` and
``poros.moe.fused.fit_reason`` find those ranks before training.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import triton
from triton.runtime.errors import OutOfResources

_STAGES: Dict[Tuple[Any, ...], int] = {}
DEFAULT_STAGES = 3  # Triton's own default


def _key(kernel: Any, kw: Dict[str, Any]) -> Tuple[Any, ...]:
    const = tuple(sorted((k, v) for k, v in kw.items() if k not in ("num_stages",)))
    # Triton's interpreter (TRITON_INTERPRET=1: the kernels on the CPU, as the tests run them)
    # has no GPU driver to ask
    dev = ("interpreter" if os.environ.get("TRITON_INTERPRET") == "1"
           else triton.runtime.driver.active.get_current_device())
    return (kernel, const, dev)


def launch(kernel: Any, grid: Any, *args: Any, **kw: Any) -> None:
    """``kernel[grid](*args, **kw)``, with fewer pipeline stages where the card's shared memory
    is short of ``kw["num_stages"]`` (default 3)."""
    key = _key(kernel, kw)
    stages = _STAGES.get(key, kw.pop("num_stages", DEFAULT_STAGES))
    kw.pop("num_stages", None)
    while True:
        try:
            kernel[grid](*args, num_stages=stages, **kw)
        except OutOfResources:
            if stages <= 1:
                raise
            stages -= 1
            continue
        _STAGES[key] = stages
        return


def stages_used() -> Dict[str, int]:
    """The stages each launched configuration ran at, every one launched so far (cut or not: a
    configuration at its configured stages too), for receipts and tests."""
    return {f"{getattr(k[0], '__name__', k[0])}{dict(k[1])}@{k[2]}": s
            for k, s in _STAGES.items()}
