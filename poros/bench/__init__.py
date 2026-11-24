"""Poros bench: reproducible benchmark harness with fair-comparison guards.

Provides the full benchmark lifecycle:
- BenchmarkSpec: declare what to run (models x methods x precisions).
- DeviceSampler: a VRAM sample, and the card's facts.
- RunArtifact: one result per (method, model, precision) cell; JSONL output.
- Baselines: poros, resident, peft, unsloth, accelerate_offload,
  deepspeed_zero3 -- each in a fresh subprocess for clean peak-VRAM
  measurement.  OOM is a result, not a crash.
- Runner: run / compare / matrix orchestration.

CPU smoke mode (tiny model, no GPU) is supported for CI / development.
"""

from poros.bench.artifacts import RunArtifact, read_artifact, write_artifact
from poros.bench.device import DeviceSampler
from poros.bench.matrix import expand_matrix
from poros.bench.runner import compare, matrix, run
from poros.bench.specs import BenchmarkSpec

__all__ = [
    "BenchmarkSpec",
    "RunArtifact",
    "write_artifact",
    "read_artifact",
    "DeviceSampler",
    "run",
    "compare",
    "matrix",
    "expand_matrix",
]
