"""DeepSpeed ZeRO-3 offload baseline wrapper for benchmark comparison.

Single-GPU ZeRO-3 with the frozen base offloaded to pinned CPU memory
(ZeRO-Offload) or NVMe (ZeRO-Infinity), per ``spec.zero3``. DeepSpeed
gathers each parameter onto the GPU when a module needs it and releases it
afterwards, so this is the established alternative to Poros's blockwise
residency. Gradient checkpointing is on, which is the memory-saving setting.

NF4 cells follow transformers' ZeRO-3 + QLoRA recipe (``bnb_4bit_quant_storage``
equal to the load dtype): the quantized model loads onto the GPU, then
``deepspeed.initialize`` offloads it. bf16/fp16 cells load through
``zero.Init``, so the base never sits whole on the GPU.

The optimizer is ``torch.optim.AdamW`` with its state and the fp32 adapter
master weights on the GPU; adapter bf16 copies above
``stage3_param_persistence_threshold`` are offloaded with the base.

As in every arm, ``vram_peak_mb`` is the training peak and
``vram_load_peak_mb`` the load (for NF4, the whole quantized model on the GPU
before ``initialize`` offloads it). OOM is recorded as a result
(``status="oom"``).
"""

from __future__ import annotations

import textwrap

from poros.bench.artifacts import RunArtifact
from poros.bench.baselines._subprocess import (
    build_common_preamble,
    run_in_subprocess,
)
from poros.bench.specs import BenchmarkSpec


def _ds_config(spec: BenchmarkSpec, precision: str) -> dict:
    """The DeepSpeed config for one cell; ``spec.zero3`` keys override."""
    z = spec.zero3
    zero = {
        "stage": 3,
        "offload_param": {"device": z.offload_device, "pin_memory": True},
    }
    if z.offload_device == "nvme":
        zero["offload_param"]["nvme_path"] = z.nvme_path
    # One level deep, so overriding offload_param.buffer_size keeps device.
    for key, value in z.zero_optimization.items():
        zero[key] = {**zero.get(key, {}), **value} if isinstance(value, dict) else value
    return {
        "train_micro_batch_size_per_gpu": spec.training.batch_size,
        "gradient_accumulation_steps": spec.training.gradient_accumulation_steps,
        "fp16" if precision == "fp16" else "bf16": {"enabled": True},
        "zero_optimization": zero,
        "steps_per_print": 10**9,
    }


def _build_script(spec: BenchmarkSpec, model_id: str, precision: str) -> str:
    """Build the Python script for a DeepSpeed ZeRO-3 offload cell."""
    preamble = build_common_preamble(spec)
    dtype_str = "torch.bfloat16" if precision != "fp16" else "torch.float16"

    quant_config_block = "bnb_config = None"
    if precision == "nf4":
        quant_config_block = textwrap.dedent("""\
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        """)
    # Align the injected block to its 12-space insertion indent so the outer
    # textwrap.dedent strips a uniform prefix (see peft.py).
    quant_config_block = quant_config_block.replace("\n", "\n" + " " * 12).rstrip()

    script = preamble + textwrap.dedent(f"""\
        import torch
        from datetime import datetime, timezone

        MODEL_ID = {model_id!r}
        TRUST_REMOTE_CODE = {bool(spec.trust_remote_code)!r}
        TARGET_MODULES = {spec.lora.target_modules!r}
        DS_CONFIG = {_ds_config(spec, precision)!r}
        RESULT = {{
            "method": "deepspeed_zero3",
            "model_id": MODEL_ID,
            "display_name": {model_id.split("/")[-1]!r},
            "precision": {precision!r},
            "hardware": {spec.hardware.gpu_name!r},
            "vram_total_mb": {spec.hardware.vram_gb * 1024},
            "parity_loss_max_diff": None,
            "parity_weight_max_diff": None,
            "bitwise_exact": False,
            "steps": STEPS,
            "seq_len": SEQ_LEN,
        }}


        try:
            # One process, no launcher: DeepSpeed reads the torch.distributed env.
            import socket
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                free_port = str(s.getsockname()[1])
            for k, v in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                         ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", free_port)):
                os.environ.setdefault(k, v)
            import deepspeed
            from transformers import AutoConfig, AutoModelForCausalLM
            from transformers.integrations import HfDeepSpeedConfig
            from peft import get_peft_model, LoraConfig

            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            {quant_config_block}

            offload_param = DS_CONFIG["zero_optimization"]["offload_param"]
            if offload_param["device"] == "nvme" and "buffer_size" not in offload_param:
                # An NVMe swap buffer must hold the largest tensor: the embeddings.
                cfg = AutoConfig.from_pretrained(
                    MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE
                ).get_text_config()
                offload_param["buffer_size"] = cfg.vocab_size * cfg.hidden_size

            # Must stay alive through from_pretrained: it makes the load ZeRO-3 aware.
            hf_ds_config = HfDeepSpeedConfig(DS_CONFIG)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype={dtype_str},
                quantization_config=bnb_config,
                trust_remote_code=TRUST_REMOTE_CODE,
            )
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={{"use_reentrant": False}}
            )
            model = get_peft_model(model, LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=TARGET_MODULES or [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            ))
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=2e-5,
            )
            engine, _, _, _ = deepspeed.initialize(
                model=model, optimizer=optimizer, config=DS_CONFIG,
            )
            engine.train()

            dummy_ids = _synthetic_batch()
            labels = dummy_ids.clone()
            _start_training()

            # Time ONE OPTIMIZER STEP (GRAD_ACCUM micro-batches) per recorded entry,
            # the same unit as every other arm. The engine scales the loss by
            # GRAD_ACCUM and only steps the optimizer at the accumulation boundary.
            step_times = []
            vram_samples = []
            losses = []
            for step in range(STEPS):
                t_start = _now()
                for _ in range(GRAD_ACCUM):
                    loss = engine(input_ids=dummy_ids, labels=labels).loss
                    engine.backward(loss)
                    engine.step()
                step_times.append(_now() - t_start)
                vram_samples.append(torch.cuda.memory_allocated() / _MiB)
                losses.append(loss.item())

            import statistics as _stats
            post_warmup = step_times[WARMUP_STEPS:]
            avg_s = _stats.median(post_warmup) if post_warmup else 0.0
            vram_pw = vram_samples[WARMUP_STEPS:] or vram_samples
            RESULT.update(
                vram_peak_mb=torch.cuda.max_memory_allocated() / _MiB,
                vram_avg_mb=sum(vram_pw) / len(vram_pw),
                s_per_step=avg_s,
                tok_per_sec=(SEQ_LEN * BATCH_SIZE * GRAD_ACCUM) / avg_s if avg_s > 0 else 0.0,
                oom=False,
                status="ok",
                arch=getattr(model.config, "model_type", ""),
                **_measurements(losses),
            )
        except torch.cuda.OutOfMemoryError as e:
            RESULT.update(
                oom=True, status="oom", error_message=str(e)[:500], **_measurements(oom=True),
            )
        except Exception as e:
            RESULT.update(oom=False, status="error", error_message=str(e)[:500])
        RESULT["timestamp"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(RESULT))
    """)
    return script


def run_deepspeed_zero3_baseline(
    spec: BenchmarkSpec, model_id: str, precision: str
) -> RunArtifact:
    """Run a DeepSpeed ZeRO-3 offload benchmark cell in a fresh subprocess.

    Args:
        spec: The benchmark specification.
        model_id: HuggingFace model ID.
        precision: Quantization precision.

    Returns:
        RunArtifact with DeepSpeed ZeRO-3 baseline results.
    """
    display = model_id.split("/")[-1] if "/" in model_id else model_id
    script = _build_script(spec, model_id, precision)
    return run_in_subprocess(
        script, "deepspeed_zero3", model_id, display, precision, spec
    )
