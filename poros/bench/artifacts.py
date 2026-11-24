"""RunArtifact: the result schema for a single benchmark cell.

One RunArtifact per (method, model, precision) cell in the benchmark matrix.
This is the schema consumed by the HTML report and the JSONL event stream.

The schema is aligned with the measured data shape from
the measured benchmark results JSONL (``poros_grid_pro.jsonl``) so that
existing results can be loaded without transformation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class RunArtifact(BaseModel):
    """Result of a single benchmark cell."""

    # Identity
    method: str  # a BASELINE_REGISTRY key: "poros", "peft", "deepspeed_zero3", ...
    model_id: str  # HF model ID
    display_name: str  # e.g. "Qwen2.5-32B"
    precision: str  # "nf4" | "bf16" | "fp16"

    # Hardware
    hardware: str = ""  # GPU name
    vram_total_mb: float = 0.0  # total GPU VRAM
    cuda_version: Optional[str] = None  # for official leaderboard keying
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None  # stack fingerprint
    bitsandbytes_version: Optional[str] = None  # stack fingerprint
    gpu_uuid: Optional[str] = None  # unique per physical GPU (leaderboard key)

    # VRAM measurements (MB)
    vram_peak_mb: float = 0.0  # training-loop peak allocated (worst-case across passes)
    vram_avg_mb: float = 0.0  # mean VRAM allocated per step (post-warmup)
    vram_resident_mb: Optional[float] = None  # resident reference (peak)
    vram_unsloth_mb: Optional[float] = None  # unsloth reference (peak)
    vram_poros_mb: Optional[float] = None  # poros measurement (peak)
    vram_avg_resident_mb: Optional[float] = None  # resident reference (avg)
    vram_avg_unsloth_mb: Optional[float] = None  # unsloth reference (avg)
    vram_avg_poros_mb: Optional[float] = None  # poros measurement (avg)
    # Every arm reads these the same way (bench/baselines/_subprocess.py);
    # all *_mb fields are MiB (2**20 bytes).
    vram_load_peak_mb: Optional[float] = None  # peak allocated through setup
    vram_reserved_peak_mb: Optional[float] = None  # torch allocator reserved, training peak
    # Device-wide used, whole run: includes the CUDA context, NCCL and any
    # other process on the GPU (on WSL2, the Windows desktop).
    vram_device_peak_mb: Optional[float] = None
    vram_free_at_start_mb: Optional[float] = None  # device free when the cell began
    vram_cap_effective_mb: Optional[float] = None  # allocator cap applied, if any
    # Peak host memory in use above the start, whole run: counts pinned memory,
    # not clean page cache; the container's own where a cgroup limits memory.
    host_mem_used_peak_mb: Optional[float] = None
    method_info: Optional[Dict[str, Any]] = None  # arm-specific facts (placement, pinning)

    # Speed (headline s_per_step = median across passes for official runs)
    s_per_step: float = 0.0  # seconds per optimizer step (median)
    tok_per_sec: Optional[float] = None  # tokens per second

    # Multi-pass official-run aggregates (None on a single-pass run)
    n_passes: int = 1  # how many passes produced this cell
    s_per_step_min: Optional[float] = None
    s_per_step_max: Optional[float] = None
    s_per_step_stddev: Optional[float] = None
    pass_s_per_step: Optional[List[float]] = None  # per-pass s/step
    pass_vram_peaks_mb: Optional[List[float]] = None  # per-pass VRAM peak
    pass_vram_avgs_mb: Optional[List[float]] = None  # per-pass VRAM average

    # Parity (None = NOT measured in this cell; parity is measured by the
    # dedicated GPU parity test, not by a speed/VRAM cell).
    parity_loss_max_diff: Optional[float] = None  # max |delta| on loss
    parity_weight_max_diff: Optional[float] = None  # max |delta| on adapter weights
    bitwise_exact: bool = False  # True iff measured AND both are 0.00e+00

    # Derived comparisons
    vram_saved_vs_resident_pct: Optional[float] = None
    vram_saved_vs_unsloth_pct: Optional[float] = None
    slowdown_vs_resident_x: Optional[float] = None
    slowdown_vs_unsloth_x: Optional[float] = None

    # Status
    oom: bool = False  # True if method OOM'd
    status: Literal["ok", "oom", "error", "skipped"] = "ok"
    error_message: Optional[str] = None

    # Run metadata
    steps: int = 0
    seq_len: int = 512
    block_size: Optional[int] = None  # Poros-only
    # The loop a Poros trainer cell's updates took (PorosTrainer.ran_path): "sequential", or
    # block-major accumulation's "efficient" or "fast" path. None: not a trainer cell, or not
    # known (an error before the config resolved).
    path: Optional[Literal["sequential", "efficient", "fast"]] = None
    arch: str = ""  # architecture family
    run_id: Optional[str] = None  # SHA-256 run ID
    timestamp: Optional[str] = None  # ISO 8601

    # Per-step metrics (for loss curves / memory traces)
    step_metrics: Optional[List[Dict[str, Any]]] = None
    # Host + software provenance (driver, PCIe link, CPU, host RAM, WSL, package
    # versions, poros commit), stamped by the runner on every cell.
    env: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# JSONL writers (one row per cell, atomic file writes)
# ---------------------------------------------------------------------------


def write_artifact(artifact: RunArtifact, output_dir: str | Path) -> Path:
    """Atomically write a RunArtifact to a JSON file.

    The file is named ``{method}_{arch}_{precision}.json`` and placed in
    *output_dir*.  A temporary file is written first, then renamed for
    atomicity.

    Args:
        artifact: The run artifact to write.
        output_dir: Directory to write the artifact file.

    Returns:
        Path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # method/arch/precision come from artifact metadata and form the filename;
    # a real value is a bare token, so reject path separators / parent refs so
    # the write cannot escape output_dir.
    for part in (artifact.method, artifact.arch, artifact.precision):
        if os.sep in part or (os.altsep and os.altsep in part) or ".." in part:
            raise ValueError(f"unsafe artifact filename component: {part!r}")
    # Include a model discriminator: different sizes of one family share an
    # arch (e.g. Qwen2.5-32B and -72B are both qwen2), so a name without the
    # model would collide and silently overwrite one cell with another.
    model_slug = re.sub(
        r"[^A-Za-z0-9.]+", "-", artifact.model_id or artifact.display_name or "model"
    ).strip("-").lower()
    fname = f"{artifact.method}_{model_slug}_{artifact.precision}.json"
    path = out / fname
    tmp = path.with_suffix(".tmp")
    tmp.write_text(artifact.model_dump_json(indent=2))
    tmp.rename(path)
    return path


def read_artifact(path: str | Path) -> RunArtifact:
    """Read a RunArtifact from a JSON file.

    Args:
        path: Path to the artifact JSON file.

    Returns:
        RunArtifact parsed from the file.
    """
    return RunArtifact.model_validate_json(Path(path).read_text())


def append_artifact_jsonl(artifact: RunArtifact, jsonl_path: str | Path) -> None:
    """Append a RunArtifact as one JSONL row.

    This matches the output format of
    the measured benchmark results JSONL (``poros_grid_pro.jsonl``).

    Args:
        artifact: The run artifact to append.
        jsonl_path: Path to the JSONL file.
    """
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(artifact.model_dump_json() + "\n")


def read_artifacts_jsonl(jsonl_path: str | Path) -> List[RunArtifact]:
    """Read all RunArtifacts from a JSONL file.

    Rows that fail validation are silently skipped (logged if the caller
    wishes to inspect them).

    Args:
        jsonl_path: Path to the JSONL file.

    Returns:
        List of successfully parsed RunArtifact objects.
    """
    results: List[RunArtifact] = []
    p = Path(jsonl_path)
    if not p.exists():
        return results
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(RunArtifact.model_validate_json(line))
        except Exception:
            # Skip malformed rows -- caller may log these
            continue
    return results


def load_grid_pro_row(row: Dict[str, Any]) -> RunArtifact:
    """Convert a raw ``poros_grid_pro.jsonl`` row into a RunArtifact.

    The grid-pro format uses slightly different field names; this maps them
    onto the canonical RunArtifact schema.

    Args:
        row: A single parsed JSON row from poros_grid_pro.jsonl.

    Returns:
        A RunArtifact with fields populated from the grid-pro row.

    Raises:
        ValueError: if the row is not a grid-pro row (so callers can skip
            unrelated JSONL rather than fabricate a zero-filled artifact).
    """
    if "tag" not in row and "poros_vram_mb" not in row:
        raise ValueError("not a grid-pro row")
    precision_map = {"qlora4": "nf4", "lora16": "bf16", "int8": "int8"}
    raw_quant = row.get("quant", "qlora4")
    precision = precision_map.get(raw_quant, raw_quant)

    status = row.get("status", "ok")
    oom = status == "oom"

    return RunArtifact(
        method="poros",
        model_id=row.get("tag", ""),
        display_name=row.get("tag", ""),
        precision=precision,
        arch=row.get("arch", ""),
        hardware=row.get("hardware", ""),
        vram_total_mb=0.0,
        vram_peak_mb=row.get("poros_vram_mb", 0.0),
        vram_avg_mb=row.get("poros_vram_avg_mb", 0.0),
        vram_resident_mb=row.get("resident_vram_mb"),
        vram_unsloth_mb=row.get("unsloth_vram_mb"),
        vram_poros_mb=row.get("poros_vram_mb"),
        vram_avg_resident_mb=row.get("resident_vram_avg_mb"),
        vram_avg_unsloth_mb=row.get("unsloth_vram_avg_mb"),
        vram_avg_poros_mb=row.get("poros_vram_avg_mb"),
        # 0.0 = "not in this row" (schema requires a float); the renderer
        # dashes falsy values instead of printing a fabricated 0.000
        s_per_step=row.get("s_per_step") or 0.0,
        s_per_step_stddev=row.get("s_per_step_stddev"),
        n_passes=row.get("n_passes") or 1,
        tok_per_sec=row.get("tok_per_sec"),
        block_size=row.get("block_size"),
        parity_loss_max_diff=row.get("parity_loss_max_diff"),  # None if absent
        parity_weight_max_diff=row.get("parity_weight_max_diff"),
        bitwise_exact=row.get("bitwise_exact", False),
        vram_saved_vs_resident_pct=row.get("vram_save_vs_resident_pct"),
        vram_saved_vs_unsloth_pct=row.get("vram_save_vs_unsloth_pct"),
        slowdown_vs_resident_x=row.get("slowdown_vs_resident_x"),
        slowdown_vs_unsloth_x=row.get("slowdown_vs_unsloth_x"),
        oom=oom,
        status=status if status in ("ok", "oom", "error", "skipped") else "skipped",
        # Preserve the raw status so an unknown-but-successful legacy value
        # ("completed") is at least visible in the report's Reason column
        # instead of a silent reclassification with reason None.
        error_message=row.get("reason")
        or (
            f"original status: {status!r}"
            if status not in ("ok", "oom", "error", "skipped")
            else None
        ),
        steps=row.get("steps", 0),
        seq_len=row.get("seq_len", 512),
    )
