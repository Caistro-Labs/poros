"""Accelerate device-map offload baseline wrapper for benchmark comparison.

Uses Accelerate's DISPATCH-based offload (``device_map="auto"`` plus an
``offload_folder``), which places layers across GPU/CPU/disk and moves them
per forward, then runs LoRA training. This is NOT
``accelerate.cpu_offload_with_hook`` (hook-based prefetch); the row is named
for what it runs. It is the closest existing alternative to Poros's
blockwise approach.

NOTE: Accelerate device-map offload is known to OOM at 32B+ on a 24GB card
because the offload path still materializes full activations.  OOM is
recorded as a result (``status="oom"``), not a crash.
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
    """Build the Python script for an Accelerate device-map offload cell."""
    preamble = build_common_preamble(spec)
    target_modules_repr = repr(spec.lora.target_modules)

    quant_config_block = ""
    if precision == "nf4":
        quant_config_block = textwrap.dedent("""\
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                # Without it transformers refuses any CPU/disk placement of a
                # 4-bit model; with it, offloaded modules stay unquantized.
                llm_int8_enable_fp32_cpu_offload=True,
            )
        """)
    else:
        quant_config_block = "bnb_config = None"

    dtype_str = "torch.bfloat16" if precision != "fp16" else "torch.float16"

    # Align the injected (possibly multi-line) quant block to its 12-space
    # insertion indent so the outer textwrap.dedent strips a uniform prefix;
    # otherwise this block's 0-indent lines defeat dedent and the generated
    # subprocess script fails to compile (IndentationError).
    quant_config_block = quant_config_block.replace("\n", "\n" + " " * 12).rstrip()

    # Off unless the spec opts in: a bench YAML is user-supplied input.
    trust_remote_code = bool(spec.trust_remote_code)

    script = preamble + textwrap.dedent(f"""\
        import torch
        from datetime import datetime, timezone

        MODEL_ID = {model_id!r}
        # Benchmarks run a model id from a user-supplied YAML, so
        # the same rule as `poros train` applies: no remote code
        # unless the spec asks for it.
        TRUST_REMOTE_CODE = {trust_remote_code!r}
        PRECISION = {precision!r}
        TARGET_MODULES = {target_modules_repr}
        GPU_MAX_MEMORY_GB = {spec.accelerate_offload.gpu_max_memory_gb!r}
        GRADIENT_CHECKPOINTING = {spec.accelerate_offload.gradient_checkpointing!r}
        HARDWARE = {spec.hardware.gpu_name!r}
        VRAM_TOTAL_MB = {spec.hardware.vram_gb * 1024}

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig

            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            {quant_config_block}

            # Accelerate's DISPATCH-based offload: device_map="auto" places
            # layers across GPU/CPU/disk and moves them per forward. This is
            # not accelerate.cpu_offload_with_hook (hook-based prefetch) --
            # the row is named for what it runs.
            # 80% of free host RAM; what does not fit goes to offload_folder
            # on disk instead of being OOM-killed. device_map sizes NF4 modules
            # at 4 bits, but offloaded ones load unquantized (bf16, 4x larger).
            CPU_SHARE_MB = int(_mem_available_mb() * 0.8 / (4 if PRECISION == "nf4" else 1))
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype={dtype_str},
                quantization_config=bnb_config,
                trust_remote_code=TRUST_REMOTE_CODE,
                device_map="auto",
                max_memory=(
                    {{0: f"{{GPU_MAX_MEMORY_GB}}GiB", "cpu": f"{{CPU_SHARE_MB}}MiB"}}
                    if GPU_MAX_MEMORY_GB else None
                ),
                offload_folder="./offload_tmp",
            )
            # Where device_map put each module (CPU/disk = offloaded). Without
            # hf_device_map the model landed on one device.
            device_map = getattr(model, "hf_device_map", None)
            placement = sorted({{str(d) for d in device_map.values()}}) if device_map else [
                str(next(model.parameters()).device)
            ]
            _M["method_info"] = {{"placement": placement}}
            if GRADIENT_CHECKPOINTING:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={{"use_reentrant": False}}
                )

            target = TARGET_MODULES
            if target is None:
                target = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

            lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=target,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.train()

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dummy_ids = _synthetic_batch()
            labels = dummy_ids.clone()

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
                    outputs = model(input_ids=dummy_ids, labels=labels)
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
                "method": "accelerate_offload",
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
                "method": "accelerate_offload",
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
        except Exception as e:
            result = {{
                "method": "accelerate_offload",
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


def run_accelerate_offload_baseline(
    spec: BenchmarkSpec, model_id: str, precision: str
) -> RunArtifact:
    """Run an Accelerate device-map offload benchmark cell in a fresh subprocess.

    Args:
        spec: The benchmark specification.
        model_id: HuggingFace model ID.
        precision: Quantization precision.

    Returns:
        RunArtifact with Accelerate offload baseline results.
    """
    display = model_id.split("/")[-1] if "/" in model_id else model_id
    script = _build_script(spec, model_id, precision)
    return run_in_subprocess(
        script, "accelerate_offload", model_id, display, precision, spec
    )
