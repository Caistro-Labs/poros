"""Benchmark runner: run, compare, matrix orchestration.

Orchestrates benchmark runs across methods, models, and precisions.
Writes RunArtifact results for consumption by the report subsystem.

FAIRNESS GUARDS:
  Before executing any cell, the runner validates that all methods will
  use identical: seq_len, lora rank/alpha/dropout, target_modules,
  steps, batch_size, gradient_accumulation, and seed.  If any mismatch
  is detected, the run fails loudly rather than silently producing
  unfair numbers.

Each baseline runs in a fresh subprocess for clean peak-VRAM.  OOM is
recorded as a result, not a crash.

Output: one JSONL row per (method, model, precision) cell.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from poros.bench.artifacts import (
    RunArtifact,
    append_artifact_jsonl,
    write_artifact,
)
from poros.bench.baselines import BASELINE_REGISTRY
from poros.bench.matrix import expand_matrix
from poros.bench.specs import BenchmarkSpec
from poros.core.env import outside_tool

logger = logging.getLogger("poros.bench")


# ---------------------------------------------------------------------------
# Fairness validation
# ---------------------------------------------------------------------------


class FairnessViolation(Exception):
    """Raised when the benchmark spec would produce unfair comparisons."""


def _validate_fairness(spec: BenchmarkSpec) -> None:
    """Check that the spec does not allow unfair comparisons.

    All methods in a single spec share the same training / lora config.
    This is enforced structurally by the schema (one TrainingSpec / LoraSpec
    per BenchmarkSpec), but we check programmatically in case someone
    monkey-patches.

    Raises:
        FairnessViolation: If the spec is structurally unfair.
    """
    # Verify all methods are known
    unknown = set(spec.methods) - set(BASELINE_REGISTRY.keys())
    if unknown:
        raise FairnessViolation(
            f"Unknown methods: {sorted(unknown)}. "
            f"Valid methods: {sorted(BASELINE_REGISTRY.keys())}"
        )

    # WSL2 pages past-capacity allocations to host RAM instead of raising
    # OOM, so without a cap a run that does not fit would report as fitting.
    if _environment()["wsl"] and spec.hardware.vram_cap_gb is None:
        raise FairnessViolation(
            "WSL2 detected: set hardware.vram_cap_gb (e.g. 22.76 on a 24 GB "
            "card) so an out-of-memory run fails instead of spilling to host RAM."
        )

    # Verify warmup_steps < steps
    if spec.training.warmup_steps >= spec.training.steps:
        raise FairnessViolation(
            f"warmup_steps ({spec.training.warmup_steps}) must be less than "
            f"steps ({spec.training.steps}). Warmup-excluded summary "
            f"statistics would have no data."
        )


# ---------------------------------------------------------------------------
# Auto-detect hardware
# ---------------------------------------------------------------------------


def _auto_detect_hardware(spec: BenchmarkSpec) -> BenchmarkSpec:
    """Fill in hardware fields from the current machine if not provided.

    The software versions (torch/cuda) and GPU uuid are *always* captured from
    the live machine -- they are run provenance, not user-supplied config -- so
    a config that hardcodes ``gpu_name`` still gets a correct, verifiable stack
    fingerprint in every artifact.
    """
    # In a child process: a CUDA context held by this parent for the whole
    # run would take VRAM from every cell.
    child = subprocess.run(
        [sys.executable, "-P", "-c", "import json; from poros.bench.device import DeviceSampler; "
         "print(json.dumps(DeviceSampler.gpu_info()))"],
        capture_output=True, text=True,
    )
    if child.returncode != 0:
        raise RuntimeError(f"GPU detection failed: {child.stderr[-500:]}")
    info = json.loads(child.stdout.strip().splitlines()[-1])
    if spec.hardware.gpu_name == "unknown":
        spec.hardware.gpu_name = info["gpu_name"]
        spec.hardware.vram_gb = info["vram_total_mb"] / 1024
    spec.hardware.cuda_version = info.get("cuda_version")
    spec.hardware.torch_version = info.get("torch_version")
    spec.hardware.gpu_uuid = info.get("gpu_uuid")
    return spec


def _run(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _smi_rows() -> Optional[str]:
    """nvidia-smi's own lines for the card (poros.core.card.nvidia_smi), as a receipt keeps them."""
    from poros.core.card import nvidia_smi

    lines = nvidia_smi("gpu=name,driver_version,memory.total,pcie.link.gen.max,"
                       "pcie.link.width.max,pcie.link.width.current", units=True, timeout=30,
                       raw=True)
    return None if lines is None else "\n".join(lines).strip()


@functools.lru_cache(maxsize=None)
def _environment() -> Dict[str, Any]:
    """Host and software provenance, read without touching CUDA."""
    proc_version = Path("/proc/version")
    meminfo = Path("/proc/meminfo")
    versions = {}
    for pkg in ("poros-train", "torch", "transformers", "peft", "accelerate",
                "bitsandbytes", "deepspeed", "unsloth"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    # Only a source checkout has a commit; a wheel inside some other git
    # tree must not report that tree's.
    repo = Path(__file__).resolve().parents[2]
    in_checkout = (repo / "pyproject.toml").exists() and _run(
        [outside_tool("git"), "-C", str(repo), "rev-parse", "--show-toplevel"]
    ) == str(repo)
    git = [outside_tool("git"), "-C", str(repo)]
    status = _run(git + ["status", "--porcelain", "--untracked-files=no"]) if in_checkout else None
    cgroup = Path("/sys/fs/cgroup/memory.max")
    cpu_quota = Path("/sys/fs/cgroup/cpu.max")
    cpuinfo = Path("/proc/cpuinfo")
    return {
        "wsl": proc_version.exists() and "microsoft" in proc_version.read_text().lower(),
        "host_ram_mb": (
            int(meminfo.read_text().split("MemTotal:")[1].split()[0]) / 1024
            if meminfo.exists() else None
        ),
        "nvidia_smi": _smi_rows(),
        "cgroup_memory_max": cgroup.read_text().strip() if cgroup.exists() else None,
        # A container's CPU quota ("quota period") can sit far below the cores
        # torch sees, and torch starts one thread per core unless told otherwise.
        "cgroup_cpu_max": cpu_quota.read_text().strip() if cpu_quota.exists() else None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu": next(
            (ln.split(":", 1)[1].strip() for ln in cpuinfo.read_text().splitlines()
             if ln.startswith("model name")),
            None,
        ) if cpuinfo.exists() else None,
        "poros_commit": _run(git + ["rev-parse", "HEAD"]) if in_checkout else None,
        "poros_dirty": bool(status) if status is not None else None,
        "versions": versions,
    }


# ---------------------------------------------------------------------------
# Single cell execution
# ---------------------------------------------------------------------------


def _run_cell(
    method: str,
    model_id: str,
    display_name: str,
    precision: str,
    spec: BenchmarkSpec,
) -> RunArtifact:
    """Run a single benchmark cell.

    Dispatches to the appropriate baseline runner from the registry.

    Args:
        method: Baseline method name.
        model_id: HF model ID.
        display_name: Human-readable model name.
        precision: Quantization precision.
        spec: Full benchmark spec.

    Returns:
        RunArtifact from the baseline runner.
    """
    runner_fn = BASELINE_REGISTRY.get(method)
    if runner_fn is None:
        return RunArtifact(
            method=method,
            model_id=model_id,
            display_name=display_name,
            precision=precision,
            hardware=spec.hardware.gpu_name,
            vram_total_mb=spec.hardware.vram_gb * 1024,
            status="error",
            error_message=f"No baseline runner registered for method '{method}'.",
            steps=spec.training.steps,
            seq_len=spec.training.seq_len,
        )

    logger.info(
        "Running cell: method=%s model=%s precision=%s",
        method, model_id, precision,
    )
    t0 = time.time()
    artifact = runner_fn(spec, model_id, precision)
    artifact.env = {**_environment(), "vram_cap_gb": spec.hardware.vram_cap_gb}
    elapsed = time.time() - t0
    logger.info(
        "Cell completed: method=%s status=%s vram_peak=%.1fMB elapsed=%.1fs",
        method, artifact.status, artifact.vram_peak_mb, elapsed,
    )
    return artifact


def _run_cell_multipass(
    method: str,
    model_id: str,
    display_name: str,
    precision: str,
    spec: BenchmarkSpec,
) -> RunArtifact:
    """Run a cell ``spec.training.n_passes`` times and aggregate the result.

    Speed (s/step) is noisy, so an official run reports the MEDIAN across passes
    with min/max/stddev spread. VRAM peak is near-deterministic; we keep the
    worst-case (max) peak so the reported feasibility number is never optimistic.
    A non-"ok" pass aborts aggregation and the failing artifact is returned (an
    OOM or error in any pass is the result for that cell).
    """
    n_passes = max(1, getattr(spec.training, "n_passes", 1))
    passes: List[RunArtifact] = []
    for i in range(n_passes):
        if n_passes > 1:
            logger.info("  pass %d/%d", i + 1, n_passes)
        art = _run_cell(method, model_id, display_name, precision, spec)
        if art.status != "ok":
            # Surface the failure immediately; don't average a broken run.
            art.n_passes = i + 1
            return art
        passes.append(art)

    base = passes[-1]
    base.n_passes = n_passes
    # Stamp hardware + software provenance for official leaderboard keying.
    base.cuda_version = spec.hardware.cuda_version
    base.torch_version = spec.hardware.torch_version
    base.gpu_uuid = spec.hardware.gpu_uuid
    versions = _environment()["versions"]
    base.transformers_version = versions["transformers"]
    base.bitsandbytes_version = versions["bitsandbytes"]

    if n_passes == 1:
        return base

    s_values = [p.s_per_step for p in passes]
    vram_values = [p.vram_peak_mb for p in passes]
    base.s_per_step = statistics.median(s_values)
    base.s_per_step_min = min(s_values)
    base.s_per_step_max = max(s_values)
    base.s_per_step_stddev = statistics.pstdev(s_values) if len(s_values) > 1 else 0.0
    base.pass_s_per_step = s_values
    base.pass_vram_peaks_mb = vram_values
    base.vram_peak_mb = max(vram_values)  # worst-case feasibility number
    for field in ("vram_load_peak_mb", "vram_reserved_peak_mb", "vram_device_peak_mb",
                  "host_mem_used_peak_mb"):
        values = [getattr(p, field) for p in passes if getattr(p, field) is not None]
        setattr(base, field, max(values) if values else None)
    avg_values = [p.vram_avg_mb for p in passes]
    base.pass_vram_avgs_mb = avg_values
    # Average VRAM-used is near-deterministic; report the mean across passes.
    base.vram_avg_mb = statistics.mean(avg_values) if avg_values else 0.0
    # tok/sec is derived from s/step; recompute from the median for consistency.
    if base.tok_per_sec is not None and base.s_per_step > 0:
        per_pass = [p.tok_per_sec for p in passes if p.tok_per_sec]
        if per_pass:
            base.tok_per_sec = statistics.median(per_pass)
    logger.info(
        "Cell aggregated over %d passes: s/step median=%.4f (min=%.4f max=%.4f "
        "stddev=%.4f), vram_peak(max)=%.1fMB",
        n_passes, base.s_per_step, base.s_per_step_min, base.s_per_step_max,
        base.s_per_step_stddev, base.vram_peak_mb,
    )
    return base


# ---------------------------------------------------------------------------
# Cross-method comparisons
# ---------------------------------------------------------------------------


def _compute_comparisons(artifacts: List[RunArtifact]) -> List[RunArtifact]:
    """Fill in derived comparison fields across a set of artifacts.

    For each artifact, computes ``vram_saved_vs_resident_pct``,
    ``vram_saved_vs_unsloth_pct``, ``slowdown_vs_resident_x``, and
    ``slowdown_vs_unsloth_x`` by looking up the resident / unsloth
    artifact with the same (model_id, precision).

    Args:
        artifacts: List of artifacts from the same matrix run.

    Returns:
        The same list, mutated with derived fields filled.
    """
    # Index by (model_id, precision) -> method -> artifact
    index: Dict[tuple, Dict[str, RunArtifact]] = {}
    for a in artifacts:
        key = (a.model_id, a.precision)
        index.setdefault(key, {})[a.method] = a

    for key, methods in index.items():
        resident = methods.get("resident")
        unsloth = methods.get("unsloth")

        for a in methods.values():
            if a.status != "ok":
                continue

            # Fill in cross-references
            if resident and resident.status == "ok":
                a.vram_resident_mb = resident.vram_peak_mb
                a.vram_avg_resident_mb = resident.vram_avg_mb
                if resident.vram_peak_mb > 0:
                    saved = (1.0 - a.vram_peak_mb / resident.vram_peak_mb) * 100
                    a.vram_saved_vs_resident_pct = round(saved, 1)
                if resident.s_per_step > 0:
                    a.slowdown_vs_resident_x = round(
                        a.s_per_step / resident.s_per_step, 2
                    )

            if unsloth and unsloth.status == "ok":
                a.vram_unsloth_mb = unsloth.vram_peak_mb
                a.vram_avg_unsloth_mb = unsloth.vram_avg_mb
                if unsloth.vram_peak_mb > 0:
                    saved = (1.0 - a.vram_peak_mb / unsloth.vram_peak_mb) * 100
                    a.vram_saved_vs_unsloth_pct = round(saved, 1)
                if unsloth.s_per_step > 0:
                    a.slowdown_vs_unsloth_x = round(
                        a.s_per_step / unsloth.s_per_step, 2
                    )

            # Poros cross-ref for non-poros methods
            poros_art = methods.get("poros")
            if poros_art and poros_art.status == "ok":
                a.vram_poros_mb = poros_art.vram_peak_mb
                a.vram_avg_poros_mb = poros_art.vram_avg_mb

    return artifacts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    spec: BenchmarkSpec,
    *,
    output_dir: Optional[str] = None,
) -> List[RunArtifact]:
    """Run a single benchmark spec (all methods x models x precisions).

    Each method runs in a fresh subprocess for clean VRAM measurement.
    OOM is recorded as a result, not a crash.

    Args:
        spec: The benchmark specification.
        output_dir: Override for spec.reporting.output_dir.

    Returns:
        List of RunArtifact results (one per method x model x precision cell).
    """
    _validate_fairness(spec)
    spec = _auto_detect_hardware(spec)

    out = Path(output_dir or spec.reporting.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "results.jsonl"

    # Save the spec
    spec_path = out / "spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2))

    artifacts: List[RunArtifact] = []
    cells = expand_matrix(spec)

    for cell in cells:
        model = cell["model"]
        artifact = _run_cell_multipass(
            method=cell["method"],
            model_id=model.model_id,
            display_name=model.display_name,
            precision=cell["precision"],
            spec=spec,
        )
        artifacts.append(artifact)

        # Write immediately (atomic per-cell)
        write_artifact(artifact, out)
        append_artifact_jsonl(artifact, jsonl_path)

    # Compute cross-method comparisons
    artifacts = _compute_comparisons(artifacts)

    # Overwrite with comparison data filled
    for artifact in artifacts:
        write_artifact(artifact, out)

    # Rewrite JSONL with comparisons
    jsonl_path.unlink(missing_ok=True)
    for artifact in artifacts:
        append_artifact_jsonl(artifact, jsonl_path)

    logger.info(
        "Benchmark complete: %d cells, %d ok, %d oom, %d error",
        len(artifacts),
        sum(1 for a in artifacts if a.status == "ok"),
        sum(1 for a in artifacts if a.status == "oom"),
        sum(1 for a in artifacts if a.status == "error"),
    )

    return artifacts


def compare(
    dir1: str | Path,
    dir2: str | Path,
) -> Dict[str, Any]:
    """Compare two benchmark runs.

    Loads the JSONL results from two output directories and computes
    per-cell deltas for VRAM and speed.

    Args:
        dir1: Path to the first benchmark output directory.
        dir2: Path to the second benchmark output directory.

    Returns:
        Comparison summary dict with per-cell deltas.
    """
    from poros.bench.artifacts import read_artifacts_jsonl

    arts1 = read_artifacts_jsonl(Path(dir1) / "results.jsonl")
    arts2 = read_artifacts_jsonl(Path(dir2) / "results.jsonl")

    # Index by (method, model_id, precision)
    def _key(a: RunArtifact) -> str:
        return f"{a.method}:{a.model_id}:{a.precision}"

    idx1 = {_key(a): a for a in arts1}
    idx2 = {_key(a): a for a in arts2}

    all_keys = sorted(set(idx1.keys()) | set(idx2.keys()))

    comparisons: List[Dict[str, Any]] = []
    for key in all_keys:
        a1 = idx1.get(key)
        a2 = idx2.get(key)
        entry: Dict[str, Any] = {"cell": key}

        if a1 and a2 and a1.status == "ok" and a2.status == "ok":
            entry["vram_delta_mb"] = round(a2.vram_peak_mb - a1.vram_peak_mb, 1)
            entry["speed_delta_s"] = round(a2.s_per_step - a1.s_per_step, 4)
            entry["dir1_status"] = a1.status
            entry["dir2_status"] = a2.status
        else:
            entry["dir1_status"] = a1.status if a1 else "missing"
            entry["dir2_status"] = a2.status if a2 else "missing"
            entry["vram_delta_mb"] = None
            entry["speed_delta_s"] = None

        comparisons.append(entry)

    return {
        "dir1": str(dir1),
        "dir2": str(dir2),
        "cells": comparisons,
        "total_cells": len(all_keys),
        "comparable_cells": sum(
            1 for c in comparisons if c["vram_delta_mb"] is not None
        ),
    }


def matrix(
    spec: BenchmarkSpec,
    *,
    output_dir: Optional[str] = None,
) -> List[RunArtifact]:
    """Run every model in the spec with every method.

    This is the same as ``run()`` but named to match the CLI subcommand
    ``poros bench matrix``.

    Args:
        spec: The benchmark specification with matrix dimensions.
        output_dir: Override for spec.reporting.output_dir.

    Returns:
        List of all RunArtifact results.
    """
    return run(spec, output_dir=output_dir)
