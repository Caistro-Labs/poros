"""Benchmark baselines: wrappers for each method under one shared config.

Each baseline:
- Runs in a fresh subprocess for clean peak-VRAM measurement.
- Records OOM as a result (``status="oom"``), never as an unhandled crash.
- Returns a ``RunArtifact`` with the same schema regardless of method.

SHARED CONFIG, NOT A SHARED PROTOCOL:
  Every arm is generated from one spec, so seq_len, LoRA shape, steps,
  batch size, and accumulation are structurally single-valued. The data
  pipelines are NOT identical: baseline arms train on a fixed synthetic
  batch, while the Poros trainer arm consumes the spec's real dataset.
  VRAM comparisons are insensitive to this; s/step is indicative
  (docs/bench.md#fairness-guards).
"""

from poros.bench.baselines.accelerate_offload import run_accelerate_offload_baseline
from poros.bench.baselines.deepspeed_zero3 import run_deepspeed_zero3_baseline
from poros.bench.baselines.peft import run_peft_baseline
from poros.bench.baselines.poros import run_poros_baseline
from poros.bench.baselines.resident import run_resident_baseline
from poros.bench.baselines.unsloth import run_unsloth_baseline

# Registry mapping method names to runner callables
BASELINE_REGISTRY = {
    "poros": run_poros_baseline,
    "resident": run_resident_baseline,
    "peft": run_peft_baseline,
    "unsloth": run_unsloth_baseline,
    "accelerate_offload": run_accelerate_offload_baseline,
    "deepspeed_zero3": run_deepspeed_zero3_baseline,
}

__all__ = [
    "BASELINE_REGISTRY",
    "run_poros_baseline",
    "run_resident_baseline",
    "run_peft_baseline",
    "run_unsloth_baseline",
    "run_accelerate_offload_baseline",
    "run_deepspeed_zero3_baseline",
]
