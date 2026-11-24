"""Subprocess isolation for benchmark baselines.

Each baseline method runs in a fresh Python process so that:
1. Peak VRAM measurement starts from a clean allocator state.
2. One method's leaked tensors cannot pollute another's measurement.
3. OOM in one method does not kill the entire benchmark matrix.

The subprocess writes its result as a single JSON line to stdout.
The parent reads it and parses it into a ``RunArtifact``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone

from poros.bench.artifacts import RunArtifact
from poros.bench.specs import BenchmarkSpec


def _make_run_id(method: str, model_id: str, precision: str) -> str:
    """Generate a deterministic run ID from cell identity + wall-clock."""
    seed = f"{method}:{model_id}:{precision}:{time.time()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _make_oom_artifact(
    method: str,
    model_id: str,
    display_name: str,
    precision: str,
    spec: BenchmarkSpec,
    error_msg: str,
) -> RunArtifact:
    """Create a RunArtifact representing an OOM event."""
    return RunArtifact(
        method=method,
        model_id=model_id,
        display_name=display_name,
        precision=precision,
        hardware=spec.hardware.gpu_name,
        vram_total_mb=spec.hardware.vram_gb * 1024,
        vram_peak_mb=0.0,
        s_per_step=0.0,
        parity_loss_max_diff=None,
        parity_weight_max_diff=None,
        bitwise_exact=False,
        oom=True,
        status="oom",
        error_message=error_msg,
        steps=spec.training.steps,
        seq_len=spec.training.seq_len,
        arch="",
        run_id=_make_run_id(method, model_id, precision),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_error_artifact(
    method: str,
    model_id: str,
    display_name: str,
    precision: str,
    spec: BenchmarkSpec,
    error_msg: str,
) -> RunArtifact:
    """Create a RunArtifact representing a non-OOM error."""
    return RunArtifact(
        method=method,
        model_id=model_id,
        display_name=display_name,
        precision=precision,
        hardware=spec.hardware.gpu_name,
        vram_total_mb=spec.hardware.vram_gb * 1024,
        vram_peak_mb=0.0,
        s_per_step=0.0,
        parity_loss_max_diff=None,
        parity_weight_max_diff=None,
        bitwise_exact=False,
        oom=False,
        status="error",
        error_message=error_msg,
        steps=spec.training.steps,
        seq_len=spec.training.seq_len,
        arch="",
        run_id=_make_run_id(method, model_id, precision),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_in_subprocess(
    script_body: str,
    method: str,
    model_id: str,
    display_name: str,
    precision: str,
    spec: BenchmarkSpec,
    timeout_seconds: int = 7200,
) -> RunArtifact:
    """Execute *script_body* in a fresh Python subprocess.

    The script must print exactly one JSON line to stdout containing the
    fields of a ``RunArtifact``.  Stderr is captured for diagnostics.

    If the subprocess OOMs (exit code or stderr contains
    ``OutOfMemoryError`` / ``CUDA out of memory``), an OOM artifact is
    returned rather than raising.

    Args:
        script_body: Python source code to execute.
        method: Baseline method name.
        model_id: HF model ID.
        display_name: Human-readable model name.
        precision: Quantization precision.
        spec: Full benchmark spec (for metadata).
        timeout_seconds: Maximum wall-clock time for the subprocess.

    Returns:
        RunArtifact parsed from the subprocess output, or an OOM/error
        artifact on failure.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-P", "-c", script_body],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _make_error_artifact(
            method, model_id, display_name, precision, spec,
            f"Subprocess timed out after {timeout_seconds}s",
        )

    stderr = result.stderr or ""

    # The arm's own JSON comes first: it carries the measurements, and an OOM
    # message a library logged to stderr does not override them.
    stdout_lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
    if result.returncode == 0 and stdout_lines:
        try:
            return RunArtifact.model_validate(json.loads(stdout_lines[-1]))
        except Exception as exc:
            return _make_error_artifact(
                method, model_id, display_name, precision, spec,
                f"Failed to parse subprocess output: {exc}. Last line: {stdout_lines[-1][:200]}",
            )

    if result.returncode == -9:
        return _make_error_artifact(
            method, model_id, display_name, precision, spec,
            f"Killed by SIGKILL, likely host out of memory: {stderr[-500:]}",
        )
    oom_signals = ("OutOfMemoryError", "CUDA out of memory", "torch.cuda.OutOfMemoryError")
    if any(sig in stderr for sig in oom_signals):
        return _make_oom_artifact(
            method, model_id, display_name, precision, spec,
            f"OOM in {method}: {stderr[-500:]}",
        )
    if result.returncode != 0:
        return _make_error_artifact(
            method, model_id, display_name, precision, spec,
            f"Subprocess exited with code {result.returncode}: {stderr[-500:]}",
        )
    return _make_error_artifact(
        method, model_id, display_name, precision, spec,
        f"Subprocess produced no output. stderr: {stderr[-500:]}",
    )


def build_common_preamble(spec: BenchmarkSpec) -> str:
    """Generate shared Python preamble for subprocess scripts.

    Sets the spec constants and the measurement helpers every arm uses.
    """
    return textwrap.dedent(f"""\
        import json, time, sys, os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        SEED = {spec.training.seed}
        STEPS = {spec.training.steps}
        SEQ_LEN = {spec.training.seq_len}
        BATCH_SIZE = {spec.training.batch_size}
        GRAD_ACCUM = {spec.training.gradient_accumulation_steps}
        LORA_RANK = {spec.lora.rank}
        LORA_ALPHA = {spec.lora.alpha}
        LORA_DROPOUT = {spec.lora.dropout}
        WARMUP_STEPS = {spec.training.warmup_steps}
        VRAM_CAP_GB = {spec.hardware.vram_cap_gb!r}

        # Shared measurement helpers: every arm reads VRAM, time, data and host
        # memory the same way, so a difference between rows is the method's.
        import threading
        import torch

        # Disable the cuDNN SDPA backend only. On some GPUs (e.g. Blackwell
        # sm_103) cuDNN builds no valid SDPA execution plans and the attention
        # call raises "No valid execution plans built"; flash / mem-efficient /
        # math remain enabled, so every method still uses the same available
        # kernels (fair comparison). This is a no-op where cuDNN SDPA works.
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)

        _MiB = 2**20
        _M = {{}}  # measurements every arm reports, filled in as the run goes
        _PEAK = {{"device": 0, "host": 0.0}}
        _STOP = threading.Event()


        def _read_kv(path):
            with open(path) as f:
                return {{ln.split()[0].rstrip(":"): int(ln.split()[1]) for ln in f}}


        try:
            _CG_MAX = int(open("/sys/fs/cgroup/memory.max").read())
        except (OSError, ValueError):  # no cgroup v2, or "max"
            _CG_MAX = None


        def _host_used_mb():
            # Memory that page-cache reclaim cannot free. In a memory-limited
            # container (RunPod) the host's MemAvailable includes other tenants,
            # so count the container's own charge minus its clean page cache
            # (pinned host memory is charged as shmem, so it stays counted).
            if _CG_MAX:
                st = _read_kv("/sys/fs/cgroup/memory.stat")
                current = int(open("/sys/fs/cgroup/memory.current").read())
                return (current - st["file"] + st["shmem"]) / _MiB
            mi = _read_kv("/proc/meminfo")
            return (mi["MemTotal"] - mi["MemAvailable"]) / 1024


        def _mem_available_mb():
            avail = _read_kv("/proc/meminfo")["MemAvailable"] / 1024
            return min(avail, _CG_MAX / _MiB - _host_used_mb()) if _CG_MAX else avail


        try:
            _HOST_START_MB = _host_used_mb()
        except Exception:  # an unreadable host never blocks the run
            _HOST_START_MB = None


        def _poll():
            # Device-wide VRAM in use (CUDA context, NCCL, other processes) and
            # host memory in use above the start (counts pinned host memory,
            # which RSS does not; clean page cache does not count).
            while not _STOP.is_set():
                try:
                    free, total = torch.cuda.mem_get_info()
                    _PEAK["device"] = max(_PEAK["device"], total - free)
                    if _HOST_START_MB is not None:
                        _PEAK["host"] = max(_PEAK["host"], _host_used_mb() - _HOST_START_MB)
                except Exception:  # a missed sample must not end the sampling
                    pass
                _STOP.wait(0.05)


        _POLLER = threading.Thread(target=_poll, daemon=True)
        if torch.cuda.is_available():
            _free, _total = torch.cuda.mem_get_info()
            _M["vram_free_at_start_mb"] = _free / _MiB
            if VRAM_CAP_GB:
                # WSL2 pages allocations past the card to host RAM instead of
                # raising OOM. The cap bounds torch's allocator; 0.5 GiB of the
                # free memory stays outside it for the context and NCCL.
                _want = min(VRAM_CAP_GB * 2**30, _total)
                _cap = min(_want, _free - 2**29)
                if _cap < 0.8 * _want:
                    raise RuntimeError(
                        f"only {{_free / 2**30:.2f}} GiB of VRAM free at start; another "
                        "process holds the GPU, so every arm would OOM against it"
                    )
                torch.cuda.set_per_process_memory_fraction(_cap / _total)
                _M["vram_cap_effective_mb"] = _cap / _MiB
            _POLLER.start()


        def _synthetic_batch():
            # A CPU generator: model loading cannot shift the tokens, so every
            # arm trains on the same batch.
            g = torch.Generator().manual_seed(SEED)
            return torch.randint(100, 30000, (BATCH_SIZE, SEQ_LEN), generator=g).cuda()


        def _start_training():
            # Split the load peak from the training peak (vram_peak_mb), and
            # drop setup's cached blocks so reserved memory starts level.
            torch.cuda.synchronize()
            _M["vram_load_peak_mb"] = torch.cuda.max_memory_allocated() / _MiB
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


        def _now():
            torch.cuda.synchronize()
            return time.perf_counter()


        def _measurements(losses=None, oom=False, **method_info):
            _STOP.set()
            if _POLLER.is_alive():
                _POLLER.join()
            m = dict(_M)
            m["vram_reserved_peak_mb"] = torch.cuda.max_memory_reserved() / _MiB
            m["vram_device_peak_mb"] = _PEAK["device"] / _MiB
            m["host_mem_used_peak_mb"] = _PEAK["host"] if _HOST_START_MB is not None else None
            if losses:
                m["step_metrics"] = [{{"step": i, "loss": v}} for i, v in enumerate(losses)]
            if oom:  # the peak allocated in the phase that failed
                m["vram_peak_mb"] = torch.cuda.max_memory_allocated() / _MiB
            if method_info:
                m["method_info"] = method_info
            return m
    """)
