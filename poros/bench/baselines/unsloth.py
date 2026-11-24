"""Unsloth baseline wrapper for benchmark comparison.

Unsloth is run in a fresh process to ensure fair VRAM measurement
(no leftover allocations from previous methods).

FAIRNESS NOTE:
  Unsloth MUST be run in a completely fresh subprocess -- never in the
  same process that loaded Poros or any other method.  This prevents
  CUDA context fragmentation from one method inflating another's
  peak-VRAM reading.
"""

from __future__ import annotations

import textwrap

from poros.bench.artifacts import RunArtifact
from poros.bench.baselines._subprocess import (
    build_common_preamble,
    run_in_subprocess,
)
from poros.bench.specs import BenchmarkSpec


def _build_script(spec: BenchmarkSpec, model_id: str, precision: str) -> str:
    """Build the Python script for an Unsloth benchmark cell."""
    preamble = build_common_preamble(spec)
    target_modules_repr = repr(spec.lora.target_modules)

    # Map precision to Unsloth's load_in parameter
    if precision == "nf4":
        load_in = "load_in_4bit=True"
    elif precision == "bf16":
        load_in = "load_in_4bit=False"
    elif precision == "fp16":
        load_in = "load_in_4bit=False"
    else:
        load_in = "load_in_4bit=True"

    # Off unless the spec opts in: a bench YAML is user-supplied input.
    trust_remote_code = bool(spec.trust_remote_code)

    script = preamble + textwrap.dedent(f"""\
        import torch
        from datetime import datetime, timezone

        MODEL_ID = {model_id!r}
        # Same rule as the other baselines: a bench YAML is
        # user-supplied input, so no remote code unless asked.
        TRUST_REMOTE_CODE = {trust_remote_code!r}
        PRECISION = {precision!r}
        TARGET_MODULES = {target_modules_repr}
        HARDWARE = {spec.hardware.gpu_name!r}
        VRAM_TOTAL_MB = {spec.hardware.vram_gb * 1024}

        try:
            from unsloth import FastLanguageModel

            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_ID,
                max_seq_length=SEQ_LEN,
                dtype=None,  # auto-detect
                trust_remote_code=TRUST_REMOTE_CODE,
                # The same checkpoint as every other arm, quantized on load,
                # not Unsloth's pre-quantized mirror (its dynamic 4-bit keeps
                # some layers in 16-bit).
                use_exact_model_name=True,
                {load_in},
            )

            target = TARGET_MODULES
            if target is None:
                target = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=target,
                bias="none",
            )
            model.train()

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dummy_ids = _synthetic_batch()
            labels = dummy_ids.clone()

            # Some architectures (e.g. gemma3/gemma4 multimodal text stacks)
            # require token-type inputs at train time (token_type_ids,
            # mm_token_type_ids, ...); probe and supply all-text zeros.
            import re as _re
            model_inputs = {{"input_ids": dummy_ids, "labels": labels}}
            for _ in range(4):
                try:
                    with torch.no_grad():
                        model(**model_inputs)
                    break
                except Exception as probe_err:
                    _m = _re.search(r"`(\\w*token_type_ids)`", str(probe_err))
                    if _m and _m.group(1) not in model_inputs:
                        model_inputs[_m.group(1)] = torch.zeros_like(dummy_ids)
                    else:
                        raise

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=2e-5,
            )

            _start_training()

            # Time ONE OPTIMIZER STEP (GRAD_ACCUM micro-batches) per recorded
            # entry, mirroring Poros's per-optimizer-step timing so s_per_step
            # is the same unit across all methods (fair comparison).
            step_times = []
            vram_samples = []
            losses = []
            for step in range(STEPS):
                t_start = _now()
                for _ in range(GRAD_ACCUM):
                    outputs = model(**model_inputs)
                    loss = outputs.loss / GRAD_ACCUM
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step_times.append(_now() - t_start)
                losses.append(loss.item() * GRAD_ACCUM)
                vram_samples.append(torch.cuda.memory_allocated() / (1024 * 1024))

            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            post_warmup = step_times[WARMUP_STEPS:]
            import statistics as _stats
            avg_s = _stats.median(post_warmup) if post_warmup else 0.0
            tok_per_sec = (SEQ_LEN * BATCH_SIZE * GRAD_ACCUM) / avg_s if avg_s > 0 else 0.0
            vram_pw = (
                vram_samples[WARMUP_STEPS:]
                if len(vram_samples) > WARMUP_STEPS else vram_samples
            )
            vram_avg_mb = (sum(vram_pw) / len(vram_pw)) if vram_pw else 0.0

            arch = ""
            try:
                arch = model.config.model_type
            except Exception:
                pass

            result = {{
                "method": "unsloth",
                "model_id": MODEL_ID,
                "display_name": MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID,
                "precision": PRECISION,
                "hardware": HARDWARE,
                "vram_total_mb": VRAM_TOTAL_MB,
                "vram_peak_mb": peak_mb,
                "vram_avg_mb": vram_avg_mb,
                "s_per_step": avg_s,
                "tok_per_sec": tok_per_sec,
                "parity_loss_max_diff": None,
                "parity_weight_max_diff": None,
                "bitwise_exact": False,
                "oom": False,
                "status": "ok",
                "steps": STEPS,
                "seq_len": SEQ_LEN,
                "arch": arch,
                **_measurements(losses),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }}
            print(json.dumps(result))

        except torch.cuda.OutOfMemoryError as e:
            result = {{
                "method": "unsloth",
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
                "arch": "",
                **_measurements(oom=True),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }}
            print(json.dumps(result))
        except ImportError as e:
            # Unsloth not installed -- record as skipped, not error
            result = {{
                "method": "unsloth",
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
                "status": "skipped",
                "error_message": "unsloth not installed: " + str(e)[:200],
                "steps": STEPS,
                "seq_len": SEQ_LEN,
                "arch": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }}
            print(json.dumps(result))
        except Exception as e:
            result = {{
                "method": "unsloth",
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
                "arch": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }}
            print(json.dumps(result))
    """)
    return script


def run_unsloth_baseline(
    spec: BenchmarkSpec, model_id: str, precision: str
) -> RunArtifact:
    """Run an Unsloth benchmark cell in a fresh process.

    This always spawns a new subprocess to guarantee clean VRAM measurement.

    Args:
        spec: The benchmark specification.
        model_id: HuggingFace model ID.
        precision: Quantization precision.

    Returns:
        RunArtifact with Unsloth baseline results.
    """
    display = model_id.split("/")[-1] if "/" in model_id else model_id
    script = _build_script(spec, model_id, precision)
    return run_in_subprocess(
        script, "unsloth", model_id, display, precision, spec
    )
