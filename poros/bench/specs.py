"""BenchmarkSpec and sub-specs for Poros Bench.

Defines the pydantic models for benchmark configuration: hardware, models,
methods, training parameters, LoRA config, and Poros-specific settings.

The spec can be loaded from a YAML file or constructed programmatically.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictSpec(BaseModel):
    """Base for every spec model: unknown keys are rejected, not dropped.

    A typo'd field (``json:`` for ``emit_json:``) would otherwise run the
    whole benchmark with the default while the author believes their
    setting applied -- the same contract PorosConfig enforces.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Sub-specs
# ---------------------------------------------------------------------------


class HardwareSpec(_StrictSpec):
    """Hardware specification for a benchmark run."""

    gpu_name: str = "unknown"
    vram_gb: float = 0.0
    pcie_gen: Optional[int] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    gpu_uuid: Optional[str] = None  # unique per physical GPU (leaderboard key)
    # Per-process VRAM ceiling for every arm. Required on WSL2, which pages
    # past-capacity allocations to host RAM instead of raising OOM.
    vram_cap_gb: Optional[float] = Field(None, gt=0)


class ModelSpec(_StrictSpec):
    """Model specification for a benchmark run."""

    model_id: str  # HF model ID
    display_name: str  # e.g. "Qwen2.5-32B"
    param_count_b: Optional[float] = None  # billions


class TrainingSpec(_StrictSpec):
    """Training parameters for a benchmark run."""

    steps: int = Field(200, ge=1)
    seq_len: int = Field(512, ge=1)
    batch_size: int = Field(1, ge=1)
    gradient_accumulation_steps: int = Field(8, ge=1)
    # "synthetic" = deterministic, zero-download, license-free random tokens
    # (what the official parity/VRAM/throughput benchmark uses and what the
    # poros paper measured on). Token content does not affect these systems
    # metrics. Use a real HF dataset id (e.g. "OpenAssistant/oasst1") only for a
    # labelled learning/loss cross-check. "synthetic:<n>" sets the sample count.
    dataset: str = "synthetic"
    seed: int = 42
    warmup_steps: int = Field(5, ge=0)  # steps to exclude from timing (symmetric across methods)
    # Official runs repeat each cell N times and report the median + spread of
    # s/step (speed is noisy: thermal throttling, allocator state). VRAM is
    # near-deterministic; we keep the worst-case peak across passes.
    n_passes: int = Field(default=3, ge=1)

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, v: str) -> str:
        # "synthetic", "synthetic:<n>", or a HuggingFace dataset id
        # (org/name). This value is embedded into a generated trainer
        # script, so restrict it to the characters a real dataset id can
        # contain and reject anything that could break out of a literal.
        if not re.fullmatch(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?(:[0-9]+)?", v):
            raise ValueError(
                f"invalid dataset id {v!r}: expected 'synthetic', "
                "'synthetic:<n>', or a HuggingFace dataset id like 'org/name'"
            )
        return v


class LoraSpec(_StrictSpec):
    """LoRA configuration for a benchmark run."""

    rank: int = Field(16, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.05, ge=0.0, lt=1.0)
    target_modules: Optional[List[str]] = None


class AccelerateSpec(_StrictSpec):
    """Settings for the ``accelerate_offload`` method."""

    # GPU share of device_map="auto" (max_memory); None = Accelerate's default,
    # which puts a model that fits wholly on the GPU (no offload at all).
    gpu_max_memory_gb: Optional[float] = Field(None, gt=0)
    gradient_checkpointing: bool = False


class Zero3Spec(_StrictSpec):
    """Settings for the ``deepspeed_zero3`` method: where ZeRO-3 offloads."""

    offload_device: Literal["cpu", "nvme"] = "cpu"  # nvme = ZeRO-Infinity
    # Required for nvme: a real NVMe mount (O_DIRECT), not tmpfs/overlay /tmp.
    nvme_path: Optional[str] = None
    # Merged one level deep into DeepSpeed's zero_optimization block (e.g.
    # stage3_max_live_parameters); empty = DeepSpeed's defaults.
    zero_optimization: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _nvme_needs_path(self) -> "Zero3Spec":
        if self.offload_device == "nvme" and not self.nvme_path:
            raise ValueError("zero3.offload_device: nvme needs zero3.nvme_path")
        return self


class PorosSpec(_StrictSpec):
    """Poros-specific configuration for a benchmark run."""

    block_size: int = Field(4, ge=1)
    enable_prefetch: bool = True
    # Throughput cells measure the real-user fast path (flash attention + async
    # prefetch), matching how resident/unsloth are timed. Determinism (math-only
    # SDPA) is for the separate bitwise-parity gate, not the speed benchmark.
    deterministic: bool = False
    # Attention backend for the model load. None = transformers default (sdpa); an
    # architecture with no SDPA implementation needs "eager".
    attn_implementation: Optional[str] = None


class ReportingSpec(_StrictSpec):
    """Reporting configuration for a benchmark run.

    The runner reads ``output_dir`` only: it always writes each RunArtifact as JSON there (and
    ``results.jsonl``), and ``poros report <output_dir>`` renders the HTML and Markdown.
    ``html``, ``markdown`` and ``emit_json`` are not read; they stay in the model because
    ``spec_hash`` (the run ids of the published artifacts) is taken over every field.
    """

    output_dir: str = "./bench_output"
    html: bool = True
    markdown: bool = True
    emit_json: bool = True


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------


VALID_METHODS = frozenset(
    {"poros", "resident", "unsloth", "peft", "accelerate_offload", "deepspeed_zero3"}
)


class BenchmarkSpec(_StrictSpec):
    """Defines a benchmark run or matrix.

    This is the top-level configuration consumed by ``poros bench run``.
    """

    spec_id: str  # human-readable identifier
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    models: List[ModelSpec]
    methods: List[str] = Field(
        default=["poros", "resident", "unsloth", "peft", "accelerate_offload"]
    )
    precisions: List[Literal["nf4", "bf16", "fp16"]] = ["nf4"]
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    lora: LoraSpec = Field(default_factory=LoraSpec)
    poros: PorosSpec = Field(default_factory=PorosSpec)
    accelerate_offload: AccelerateSpec = Field(default_factory=AccelerateSpec)
    zero3: Zero3Spec = Field(default_factory=Zero3Spec)
    reporting: ReportingSpec = Field(default_factory=ReportingSpec)
    # Off by default, and `poros bench` refuses a YAML that asks: a config
    # file grants no remote code. Only Python building a spec can set it.
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _validate_methods(self) -> "BenchmarkSpec":
        invalid = set(self.methods) - VALID_METHODS
        if invalid:
            raise ValueError(
                f"Unknown benchmark methods: {sorted(invalid)}. "
                f"Valid methods: {sorted(VALID_METHODS)}"
            )
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def spec_hash(self) -> str:
        """Deterministic SHA-256 of the spec for run-id derivation."""
        blob = self.model_dump_json(indent=None).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
