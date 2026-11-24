"""Poros baseline wrapper for benchmark comparison.

Runs Poros blockwise QLoRA training (``PorosTrainer.from_pretrained(...).train(...)``) in a
fresh subprocess and returns a RunArtifact with VRAM, speed, and parity measurements.
"""

from __future__ import annotations

import textwrap

from poros.bench.artifacts import RunArtifact
from poros.bench.baselines._subprocess import (
    build_common_preamble,
    run_in_subprocess,
)
from poros.bench.specs import BenchmarkSpec


def _build_trainer_script(spec: BenchmarkSpec, model_id: str, precision: str) -> str:
    """Build the script for a PorosTrainer cell."""
    preamble = build_common_preamble(spec)

    target_modules_repr = repr(spec.lora.target_modules)
    block_size = spec.poros.block_size
    enable_prefetch = spec.poros.enable_prefetch
    deterministic = spec.poros.deterministic
    attn_impl = spec.poros.attn_implementation

    # Off unless the spec opts in: a bench YAML is user-supplied input.
    trust_remote_code = bool(spec.trust_remote_code)

    script = preamble + textwrap.dedent(f"""\
        import torch
        from datetime import datetime, timezone

        MODEL_ID = {model_id!r}
        # Benchmarks run a model id from a user-supplied YAML, so the
        # same rule as `poros train` applies: no remote code unless
        # the spec asks for it.
        TRUST_REMOTE_CODE = {trust_remote_code!r}
        PRECISION = {precision!r}
        BLOCK_SIZE = {block_size}
        ENABLE_PREFETCH = {enable_prefetch!r}
        DETERMINISTIC = {deterministic!r}
        TARGET_MODULES = {target_modules_repr}
        ATTN_IMPL = {attn_impl!r}
        HARDWARE = {spec.hardware.gpu_name!r}
        VRAM_TOTAL_MB = {spec.hardware.vram_gb * 1024}
        # Pure-python (no torch): importable even when the training stack is not.
        from poros.core.errors import PorosOOMError

        config = None  # the cell's PorosConfig, resolved for this machine by from_pretrained
        try:
            from poros import PorosTrainer, PorosConfig
            from poros.core.config import train_path
            # NOTE: a throughput cell must NOT call enforce_determinism(). That
            # forces math-only SDPA (flash attention off) and
            # use_deterministic_algorithms(True), which serializes CUDA and
            # defeats the async prefetch overlap -- handicapping poros while the
            # resident/unsloth baselines run the normal flash+async path. This
            # cell measures the real-user training path; bitwise parity
            # (0.00e+00) is proven separately by tests/test_gpu_parity.py with
            # determinism ON. Keeping the two concerns separate is what makes
            # the slowdown number fair.

            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            config = PorosConfig(
                model_name_or_path=MODEL_ID,
                quantization=PRECISION,
                block_size=BLOCK_SIZE,
                enable_prefetch=ENABLE_PREFETCH,
                deterministic=DETERMINISTIC,
                lora_rank=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                lora_target_modules=TARGET_MODULES,
                max_seq_length=SEQ_LEN,
                per_device_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                max_steps=STEPS,
                seed=SEED,
                blockwise_arch_check="warn",
                attn_implementation=ATTN_IMPL,
                trust_remote_code=TRUST_REMOTE_CODE,
            )

            trainer = PorosTrainer.from_pretrained(
                MODEL_ID,
                config=config,
                quantization=PRECISION,
                lora_rank=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                block_size=BLOCK_SIZE,
            )

            _start_training()  # the trainer resets the peak again at step 0
            t0 = time.time()
            metrics = trainer.train(
                {spec.training.dataset!r},
                steps=STEPS,
                seq_len=SEQ_LEN,
            )
            total_time = time.time() - t0

            # Peak VRAM = the training-loop peak. trainer.train() resets peak
            # stats at step 0, so this is the sustained training footprint under
            # blockwise residency (the memory-budget figure the paper reports),
            # not the one-time NF4 load spike. The load spike is a property of
            # bitsandbytes quantizing the base on-device and is documented
            # separately (it sets the physical card you load on, ~24 GB at 32B);
            # the residency win is this training peak.
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # s/step EXCLUDING warmup steps (symmetric across methods). Use the
            # per-step wall times the trainer recorded, drop the first
            # WARMUP_STEPS, and take the median (robust to a single slow step).
            import statistics as _stats
            train_metrics = trainer.training_metrics
            step_times = list(train_metrics.get("step_time_history", []))
            timed = step_times[WARMUP_STEPS:] if len(step_times) > WARMUP_STEPS else step_times
            # Each recorded step is ONE optimizer step = GRAD_ACCUM micro-batches,
            # so the token count per step includes the GRAD_ACCUM factor (keeps
            # tok_per_sec in the same unit as the resident/unsloth baselines).
            if timed:
                s_per_step = _stats.median(timed)
                timed_total = sum(timed)
                tok_per_sec = (
                    (len(timed) * SEQ_LEN * BATCH_SIZE * GRAD_ACCUM) / timed_total
                    if timed_total > 0 else 0.0
                )
            else:
                # Fallback: no per-step history -> coarse wall-clock estimate.
                s_per_step = total_time / max(STEPS, 1)
                tok_per_sec = (
                    (STEPS * SEQ_LEN * BATCH_SIZE * GRAD_ACCUM) / total_time
                    if total_time > 0 else 0.0
                )

            # Mean VRAM allocated per step, post-warmup (the average the run
            # held) -- the apples-to-apples number for comparing against the
            # resident + unsloth baselines.
            # The trainer records decimal MB; convert to MiB like every other field.
            vram_hist = [v * 1e6 / _MiB for v in train_metrics.get("vram_alloc_history_mb", [])]
            vram_pw = vram_hist[WARMUP_STEPS:] if len(vram_hist) > WARMUP_STEPS else vram_hist
            vram_avg_mb = (sum(vram_pw) / len(vram_pw)) if vram_pw else 0.0

            # Detect arch from model config (PEFT proxies .config to the base).
            arch = ""
            try:
                cfg = getattr(trainer.model, "config", None)
                arch = getattr(cfg, "model_type", "") or ""
            except Exception:
                arch = ""

            result = {{
                "method": "poros",
                "model_id": MODEL_ID,
                "display_name": MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID,
                "precision": PRECISION,
                "hardware": HARDWARE,
                "vram_total_mb": VRAM_TOTAL_MB,
                "vram_peak_mb": peak_mb,
                "vram_avg_mb": vram_avg_mb,
                "s_per_step": s_per_step,
                "tok_per_sec": tok_per_sec,
                # Parity is NOT measured in a throughput cell -- it is measured
                # by the dedicated parity self-check / GPU test. None = not
                # measured here (never fabricate a 0.00e+00 from a speed run).
                "parity_loss_max_diff": None,
                "parity_weight_max_diff": None,
                "bitwise_exact": False,
                "oom": False,
                "status": "ok",
                "steps": STEPS,
                "seq_len": SEQ_LEN,
                "block_size": BLOCK_SIZE,
                # the loop its updates took
                "path": trainer.ran_path or train_path(trainer.config),
                "arch": arch,
                **_measurements(
                    train_metrics.get("loss_history"), torch_num_threads=torch.get_num_threads()
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }}
            print(json.dumps(result))

        except (torch.cuda.OutOfMemoryError, PorosOOMError) as e:
            result = {{
                "method": "poros",
                "model_id": MODEL_ID,
                "display_name": MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID,
                "precision": PRECISION,
                "hardware": HARDWARE,
                "vram_total_mb": VRAM_TOTAL_MB,
                "vram_peak_mb": 0.0,
                "s_per_step": 0.0,
                "parity_loss_max_diff": None,
                "parity_weight_max_diff": None,
                "bitwise_exact": False,
                "oom": True,
                "status": "oom",
                "error_message": str(e)[:500],
                "steps": STEPS,
                "seq_len": SEQ_LEN,
                "block_size": BLOCK_SIZE,
                "path": None if config is None else train_path(config),
                "arch": "",
                **_measurements(oom=True),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }}
            print(json.dumps(result))
        except Exception as e:
            result = {{
                "method": "poros",
                "model_id": MODEL_ID,
                "display_name": MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID,
                "precision": PRECISION,
                "hardware": HARDWARE,
                "vram_total_mb": VRAM_TOTAL_MB,
                "vram_peak_mb": 0.0,
                "s_per_step": 0.0,
                "parity_loss_max_diff": None,
                "parity_weight_max_diff": None,
                "bitwise_exact": False,
                "oom": False,
                "status": "error",
                "error_message": str(e)[:500],
                "steps": STEPS,
                "seq_len": SEQ_LEN,
                "block_size": BLOCK_SIZE,
                "arch": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }}
            print(json.dumps(result))
    """)
    return script


def _build_script(spec: BenchmarkSpec, model_id: str, precision: str) -> str:
    """Build the Python script that runs a Poros benchmark cell."""
    return _build_trainer_script(spec, model_id, precision)


def run_poros_baseline(
    spec: BenchmarkSpec, model_id: str, precision: str
) -> RunArtifact:
    """Run a Poros benchmark cell in a fresh subprocess.

    Args:
        spec: The benchmark specification.
        model_id: HuggingFace model ID.
        precision: Quantization precision.

    Returns:
        RunArtifact with Poros benchmark results.
    """
    display = model_id.split("/")[-1] if "/" in model_id else model_id
    script = _build_script(spec, model_id, precision)
    return run_in_subprocess(
        script, "poros", model_id, display, precision, spec
    )
