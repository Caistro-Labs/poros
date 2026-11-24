"""Matrix expansion for Poros Bench.

Expands a BenchmarkSpec into individual cells (method x model x precision)
for the benchmark matrix.  Each cell is a dict suitable for passing to a
baseline runner.

The expansion order is model-first so that the same model can be loaded
once and reused across methods and precisions where feasible.
"""

from __future__ import annotations

from typing import Any, Dict, List

from poros.bench.specs import BenchmarkSpec


def expand_matrix(spec: BenchmarkSpec) -> List[Dict[str, Any]]:
    """Expand a BenchmarkSpec into individual benchmark cells.

    Each cell is a plain dict with keys: ``method``, ``model`` (ModelSpec),
    ``precision``, ``training``, ``lora``, ``poros``.

    Args:
        spec: The benchmark specification.

    Returns:
        List of cell configurations, ordered model -> method -> precision.
    """
    cells: List[Dict[str, Any]] = []
    for model in spec.models:
        for method in spec.methods:
            for precision in spec.precisions:
                cells.append(
                    {
                        "method": method,
                        "model": model,
                        "precision": precision,
                        "training": spec.training,
                        "lora": spec.lora,
                        "poros": spec.poros,
                    }
                )
    return cells
