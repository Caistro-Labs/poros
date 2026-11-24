"""Absolute GPU gate for poros.moe: bitwise parity, speed, and eval CE.

Checks streamed-vs-resident bitwise parity, step speed and eval CE; default model
Qwen/Qwen3-30B-A3B-Instruct-2507.

Usage on a CUDA machine:
    python scripts/moe_absolute_gate.py --out moe_absolute.json
    python scripts/moe_absolute_gate.py --model Qwen/Qwen3-30B-A3B-Instruct-2507
    python scripts/moe_absolute_gate.py --loader streamed  # poros.moe.load
    python scripts/moe_absolute_gate.py --loader streamed --store ssd --store-path DIR
        # both arms read the experts from DIR's .sss (built on first use): streamed from the
        # SSD (poros.sss.SsdSource) vs every expert resident in VRAM
    python scripts/moe_absolute_gate.py --lora-dropout 0.05  # PorosConfig default
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

log = logging.getLogger("moe_absolute_gate")

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

TRAIN_TEXTS = [
    "Poros streams frozen NF4 expert slices during QLoRA so a Mixture-of-Experts "
    "checkpoint can train on a single GPU without dropping routed experts.",
    "Bitwise parity means the streamed run and the fully resident run produce "
    "identical losses, gradients, and adapter weights at every optimizer step.",
    "The router stays frozen. Only expert LoRA adapters receive gradients. "
    "Dense parameters either stream through the existing blockwise engine or "
    "stay pinned when the leftover model already fits.",
    "Qwen3-MoE fused experts are gate_up_proj and down_proj per layer. Stock "
    "bitsandbytes cannot quantize that 3-D layout, which is why poros.moe "
    "owns the NF4 store and the expert residency manager.",
]
EVAL_TEXTS = [
    "Held-out evaluation uses different sentences so eval CE is not a replay of "
    "the training tokens. Streamed and resident must still match exactly.",
    "A second eval span covers tool use, code, and multilingual leftovers so "
    "the CE is not a single padded line. The number is quality under the "
    "parity claim, not a public leaderboard score.",
    "If streamed and resident disagree here, the training losses already failed "
    "or the eval path diverged after the last optimizer step.",
]


def _log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)
    log.info(msg)


def _check_model_type(model_id: str, model_type: str | None) -> None:
    from poros.moe.arch import MOE_FAMILIES

    if model_type not in MOE_FAMILIES:
        raise SystemExit(
            f"{model_id} has model_type={model_type!r}; prepare_moe accepts "
            f"{sorted(MOE_FAMILIES)}. Qwen3.8-27B is dense. Use {DEFAULT_MODEL}."
        )


def _tokenize(tokenizer, texts: list[str], seq_len: int, device: str):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        add_special_tokens=True,
    )
    return encoded["input_ids"].to(device)


def _ssd_options(store_path: str) -> dict[str, Any]:
    """The SSD source's options as ``PorosTrainer`` sets them (poros.train.trainer._setup_moe)."""
    from poros.sss.probe import probe

    found = probe(store_path)
    return {"bandwidth": found.bandwidth, "latency_bandwidth": found.b_ssd_qd1,
            "threads": max(8, found.queue_depth)}


def _load_model(model_id: str, loader: str, store: str = "ram", store_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    from poros.core.arch_registry import registry_type

    _log(f"loading {model_id} on CPU bf16 ({loader} loader, store {store})")
    started = time.perf_counter()
    if loader == "streamed":
        from poros.moe.load import load_moe_streamed

        ssd = {"store": "ssd", "store_path": store_path, "ssd_options": _ssd_options(store_path)
               } if store == "ssd" else {}
        model = load_moe_streamed(model_id, torch.bfloat16, **ssd)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    _check_model_type(model_id, registry_type(model.config))
    _log(f"loaded in {time.perf_counter() - started:.1f}s")
    return model


def _gpu_identity() -> dict[str, Any]:
    import subprocess

    import torch
    import transformers

    props = torch.cuda.get_device_properties(0)
    smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,"
            "pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": int(props.total_memory),
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _peak_vram() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated())


def _reset_peak() -> None:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _padded_rows(tokenizer, count: int, seq_len: int, device: str) -> dict[str, Any]:
    """A real padded batch: oasst1 rows of different lengths, with a mask, pads out of the loss."""
    from datasets import load_dataset

    texts: list[str] = []
    for row in load_dataset("OpenAssistant/oasst1", split="train"):
        text = row.get("text") or ""
        if (
            row.get("lang") == "en"
            and 100 < len(text)
            and not any(len(text) == len(t) for t in texts)
        ):
            texts.append(text)
            if len(texts) == count:
                break
    if len(texts) < count:
        raise SystemExit(f"only {len(texts)} usable oasst1 rows")
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_len
    )
    mask = encoded["attention_mask"]
    if bool(mask.all()):
        raise SystemExit("the batch has no padding, so it would not test the padded path")
    labels = encoded["input_ids"].clone()
    labels[mask == 0] = -100
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": mask.to(device),
        "labels": labels.to(device),
    }


def _train(model, input_ids, *, steps: int, lr: float, attention_mask=None, labels=None):
    import torch

    from poros.moe.gates import _train_adapter_steps

    torch.cuda.synchronize()
    started = time.perf_counter()
    trace = _train_adapter_steps(
        model, input_ids, steps=steps, lr=lr, attention_mask=attention_mask, labels=labels
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return trace, elapsed / steps


def _eval_ce_tensor(model, input_ids):
    import torch

    model.eval()
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss.detach().clone()
    model.train()
    return loss


def run_gate(
    *,
    model_id: str,
    steps: int,
    seq_len: int,
    lr: float,
    seed: int,
    device: str,
    loader: str = "hf",
    lora_dropout: float = 0.0,
    padded_rows: int = 0,
    store: str = "ram",
    store_path: str | None = None,
) -> dict[str, Any]:
    import gc
    import resource

    import torch
    from transformers import AutoConfig, AutoTokenizer

    from poros.core.arch_registry import registry_type
    from poros.core.determinism import enforce_determinism
    from poros.moe.arch import MoeArch
    from poros.moe.prepare import MoeConfig, default_backends, prepare_moe, streaming_config

    if device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("moe_absolute_gate requires CUDA")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    enforce_determinism()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    _log(f"config {model_id}")
    from poros.moe.inkling2.config import register

    register()  # Inkling 2 Mini: Poros's own config and decoder classes (no-op for the rest)
    config = AutoConfig.from_pretrained(model_id)
    model_type = registry_type(config)
    _check_model_type(model_id, model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    capacity = streaming_config(config).capacity_bytes
    text = getattr(config, "text_config", None) or config
    train_ids = _tokenize(tokenizer, TRAIN_TEXTS, seq_len, device)
    padded = _padded_rows(tokenizer, padded_rows, seq_len, device) if padded_rows else None
    eval_ids = _tokenize(tokenizer, EVAL_TEXTS, seq_len, device)
    backends = default_backends()
    arms: dict[str, dict[str, Any]] = {}
    gpu = _gpu_identity()
    _log(
        f"gpu={gpu['name']} sm={gpu['compute_capability']} "
        f"vram={gpu['total_memory_bytes']} one_layer={capacity}"
    )

    for label, streamed in (("streamed", True), ("resident", False)):
        gc.collect()
        _reset_peak()
        torch.manual_seed(seed)  # both arms draw the same dropout masks
        model = _load_model(model_id, loader, store, store_path)
        if hasattr(getattr(model, "_poros_moe_store", None), "max_layer_nbytes"):
            # a store whose layers differ in size (Inkling 2's released experts): its own budget
            capacity = streaming_config(model.config, store=model._poros_moe_store).capacity_bytes
        if not arms:  # the process high-water mark is this load's own peak only here
            load_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        moe_cfg = MoeConfig(
            streamed=streamed,
            capacity_bytes=capacity if streamed else 0,
            look_ahead=1 if streamed else 0,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=lora_dropout,
            seed=seed,
            device=device,
        )
        _log(f"prepare_moe {label} streamed={streamed}")
        prepared = time.perf_counter()
        report = prepare_moe(model, moe_cfg, backends=backends)
        _log(
            f"prepare_moe {label} done in {time.perf_counter() - prepared:.1f}s "
            f"dense_streamed={report.dense_streamed} "
            f"experts={report.experts_quantized} blocks={report.blocks_installed}"
        )
        _log(f"train {label} steps={steps} seq_len={seq_len}")
        model.train()  # as PorosTrainer does; from_pretrained leaves the base in eval mode
        if padded is None:
            trace, sec_per_step = _train(model, train_ids, steps=steps, lr=lr)
        else:
            trace, sec_per_step = _train(
                model,
                padded["input_ids"],
                steps=steps,
                lr=lr,
                attention_mask=padded["attention_mask"],
                labels=padded["labels"],
            )
        eval_loss = _eval_ce_tensor(model, eval_ids).detach().cpu()
        peak = _peak_vram()
        _log(
            f"{label} sec/step={sec_per_step:.4f} peak_vram={peak} "
            f"eval_ce={float(eval_loss):.6f} last_loss={float(trace.losses[-1]):.6f}"
        )
        arms[label] = {
            "report": {
                "guarantee": report.guarantee,
                "streamed": report.streamed,
                "dense_streamed": report.dense_streamed,
                "experts_quantized": report.experts_quantized,
                "blocks_installed": report.blocks_installed,
                "warnings": list(report.warnings),
            },
            "losses": [float(x) for x in trace.losses],
            "loss_tensors": [t.detach().cpu() for t in trace.losses],
            "grad_tensors": [t.detach().cpu() for t in trace.grad_norms],
            "adapter_tensors": {k: v.detach().cpu() for k, v in trace.final_adapters.items()},
            "eval_loss_tensor": eval_loss,
            "eval_loss": float(eval_loss),
            "sec_per_step": sec_per_step,
            "peak_vram_bytes": peak,
            "capacity_bytes": capacity if streamed else 0,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    s, r = arms["streamed"], arms["resident"]
    loss_equal = all(torch.equal(a, b) for a, b in zip(s["loss_tensors"], r["loss_tensors"]))
    grad_equal = all(torch.equal(a, b) for a, b in zip(s["grad_tensors"], r["grad_tensors"]))
    adapter_names = sorted(s["adapter_tensors"])
    adapter_equal = adapter_names == sorted(r["adapter_tensors"]) and all(
        torch.equal(s["adapter_tensors"][n], r["adapter_tensors"][n]) for n in adapter_names
    )
    eval_equal = torch.equal(s["eval_loss_tensor"], r["eval_loss_tensor"])
    max_abs = max(float((a - b).abs().max()) for a, b in zip(s["loss_tensors"], r["loss_tensors"]))
    parity_ok = bool(loss_equal and grad_equal and adapter_equal and max_abs == 0.0)
    quality_ok = bool(eval_equal)

    def _arm_public(arm: dict[str, Any]) -> dict[str, Any]:
        skip = {"loss_tensors", "grad_tensors", "adapter_tensors", "eval_loss_tensor"}
        return {k: v for k, v in arm.items() if k not in skip}

    receipt = {
        "schema": "poros.moe.absolute_gate.v1",
        "model_id": model_id,
        "model_type": str(model_type),
        "loader": loader,
        "store": store,
        "load_host_rss_high_water_bytes": load_rss,
        "arch": {
            "hidden_size": int(text.hidden_size),
            # One expert's width (intermediate_size on olmoe/mixtral); key kept for the schema.
            "moe_intermediate_size": MoeArch.from_config(config).intermediate_size,
            "num_hidden_layers": int(text.num_hidden_layers),
            "num_experts": int(text.num_experts),
            "num_experts_per_tok": MoeArch.from_config(config).layout.top_k,
        },
        "rejected_today": {
            "Qwen/Qwen3.8-27B": "dense Gated DeltaNet + FFN, not qwen3_moe",
            "Qwen/Qwen3.8-2.4T-A95B": "MoE but 2.4T, not a 48GB card",
        },
        "steps": steps,
        "seq_len": seq_len,
        "lr": lr,
        "lora_dropout": lora_dropout,
        "seed": seed,
        "gpu": gpu,
        "one_layer_capacity_bytes": capacity,
        "parity": {
            "losses_bitwise_equal": bool(loss_equal),
            "grad_norms_bitwise_equal": bool(grad_equal),
            "adapter_weights_bitwise_equal": bool(adapter_equal),
            "max_abs_loss_diff": max_abs,
            "passed": parity_ok,
        },
        "speed": {
            "streamed_sec_per_step": s["sec_per_step"],
            "resident_sec_per_step": r["sec_per_step"],
            "streamed_over_resident": (
                s["sec_per_step"] / r["sec_per_step"] if r["sec_per_step"] else None
            ),
            "streamed_peak_vram_bytes": s["peak_vram_bytes"],
            "resident_peak_vram_bytes": r["peak_vram_bytes"],
        },
        "quality": {
            "eval_ce_streamed": s["eval_loss"],
            "eval_ce_resident": r["eval_loss"],
            "eval_ce_bitwise_equal": bool(eval_equal),
            "passed": quality_ok,
        },
        "streamed": _arm_public(s),
        "resident": _arm_public(r),
        "passed": bool(parity_ok and quality_ok),
    }
    headline = {
        "parity": receipt["parity"]["passed"],
        "quality": receipt["quality"]["passed"],
        "max_abs_loss_diff": max_abs,
        "streamed_sec_per_step": s["sec_per_step"],
        "resident_sec_per_step": r["sec_per_step"],
    }
    verdict = "PASS" if receipt["passed"] else "FAIL"
    _log(f"ABSOLUTE GATE {verdict} {json.dumps(headline)}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loader", choices=("hf", "streamed"), default="hf")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--store", choices=("ram", "ssd"), default="ram",
                        help="ssd: both arms read the experts from --store-path's .sss "
                        "(--loader streamed)")
    parser.add_argument("--store-path", default=None)
    parser.add_argument(
        "--padded-rows",
        type=int,
        default=0,
        help="train on this many real oasst1 rows, padded, with an attention mask",
    )
    parser.add_argument("--out", default="moe_absolute.json")
    args = parser.parse_args()
    if args.store == "ssd" and (args.loader != "streamed" or not args.store_path):
        parser.error("--store ssd needs --loader streamed and --store-path")
    receipt = run_gate(
        model_id=args.model,
        steps=args.steps,
        seq_len=args.seq_len,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        loader=args.loader,
        lora_dropout=args.lora_dropout,
        padded_rows=args.padded_rows,
        store=args.store,
        store_path=args.store_path,
    )
    if args.padded_rows:
        receipt["train_data"] = (
            f"{args.padded_rows} oasst1 rows, padded, attention mask, pads out of the loss"
        )
    text = json.dumps(receipt, indent=2, default=str)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    print(text)


if __name__ == "__main__":
    main()
