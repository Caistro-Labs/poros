"""Paper-protocol quality run for poros.moe: synthetic tokens, or a HF dataset (--dataset).

It measures the held-out loss of a MoE fine-tune on synthetic tokens:
`_SyntheticTokenDataset` (token ids in [100, 30000), seed 42),
seq 512, batch 1, grad-accum 8, AdamW, cosine schedule, dropout 0, held-out
eval CE on a reserved synthetic split. With --dataset (e.g.
OpenAssistant/oasst1) it trains on the Poros trainer's HF loader instead
and evaluates on the first --eval-samples rows of the shuffled train split.
Any family in poros.moe.arch.MOE_FAMILIES runs.

Canonical paper cells are 200 steps. Default here is 50, matching the
published 5090/blog synthetic pass. 200 remains available via --steps.
``--dataset`` swaps the synthetic tokens for a real HF dataset (the paper's
real-data cross-check is ``OpenAssistant/oasst1``) through the trainer's own
loader; the first eval batches of its shuffled train split are held out from
training. ``--checkpoint-dir`` saves each arm every ``--checkpoint-every``
steps; rerunning the same command on the same code resumes bitwise where a
killed run stopped.

    python scripts/moe_synthetic_quality.py --out moe_syn_quality.json
    python scripts/moe_synthetic_quality.py --dataset OpenAssistant/oasst1 --steps 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("moe_synthetic_quality")

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)
    log.info(msg)


def _load_model(model_id: str):
    from poros.moe.load import load_moe_streamed

    # The trainer's loader: experts quantize as they load, so host RAM stays
    # well under the bf16 checkpoint (a 31 GB DeepSeek-V2-Lite on a 31 GiB host).
    _log(f"loading {model_id} on CPU bf16, experts NF4")
    started = time.perf_counter()
    model = load_moe_streamed(model_id)
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


def _cosine_lambda(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = float(step - warmup) / max(1, max_steps - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _batch_to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def _mute_route(model):
    blocks = [m for m in model.modules() if getattr(m, "on_route", None) is not None]
    observers = [m.on_route for m in blocks]
    for block in blocks:
        block.on_route = None
    return blocks, observers


def _restore_route(blocks, observers) -> None:
    for block, observer in zip(blocks, observers, strict=True):
        block.on_route = observer


def _forward_loss(model, batch):
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["labels"],
    )
    return out.loss


def _eval_curve(model, loader, device: str) -> dict[str, Any]:
    import torch

    model.eval()
    losses: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            losses.append(_forward_loss(model, batch).detach())
    model.train()
    stacked = torch.stack(losses)
    mean = stacked.float().mean()
    return {
        "mean": float(mean),
        "mean_tensor": mean.detach().cpu(),
        "per_batch": [float(x) for x in losses],
        "per_batch_tensors": [t.detach().cpu() for t in losses],
        "ppl": math.exp(float(mean)) if math.isfinite(float(mean)) else float("inf"),
    }


def _adam_tensors(optimizer) -> dict[str, Any]:
    blobs: dict[str, Any] = {}
    for i, group in enumerate(optimizer.param_groups):
        for j, param in enumerate(group["params"]):
            state = optimizer.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    blobs[f"g{i}.p{j}.{key}"] = state[key].detach().cpu().clone()
    return blobs


def _run_key(protocol: list[Any]) -> str:
    """Hash of the protocol and the code it runs, so a checkpoint never resumes other code."""
    import hashlib

    import poros

    digest = hashlib.sha256(json.dumps(protocol).encode())
    for path in [Path(__file__), *sorted(Path(poros.__file__).parent.rglob("*.py"))]:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _save(obj: Any, path: Path, on_save: Callable[[], None] | None) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a kill mid-save keeps the previous checkpoint
    if on_save is not None:
        on_save()


def _train_arm(
    *,
    model,
    train_loader,
    eval_loader,
    steps: int,
    ga: int,
    lr: float,
    warmup: int,
    weight_decay: float,
    max_grad_norm: float,
    device: str,
    eval_every: int,
    checkpoint: Path | None = None,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("no trainable parameters after prepare_moe")
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    scheduler = LambdaLR(
        optimizer, lambda step: _cosine_lambda(step, warmup, steps)
    )

    batches = list(train_loader)  # unshuffled: micro-batch m is batches[m % len]
    adapters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    # Everything a resumed run needs to continue bitwise as if never stopped.
    state: dict[str, Any] = {
        "step": 0,
        "events": [],
        "train_loss_tensors": [],
        "grad_tensors": [],
        "eval_points": [],
        "eval_tensors": {},
        "train_seconds": 0.0,
        "peak_vram_bytes": 0,
        "resumed_at_steps": [],
    }
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True)  # tensors, numbers, lists
        with torch.no_grad():
            for name, param in adapters.items():
                param.copy_(saved["adapters"][name])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        state = saved["state"]
        del saved  # drop the host copy of the adapters before training
        state["resumed_at_steps"].append(state["step"])
        _log(f"resumed from {checkpoint} at step {state['step']}")
    events = state["events"]
    train_loss_tensors = state["train_loss_tensors"]
    grad_tensors = state["grad_tensors"]
    eval_points = state["eval_points"]
    eval_tensors = state["eval_tensors"]

    if not eval_tensors:
        eval0 = _eval_curve(model, eval_loader, device)
        eval_points.append({"step": 0, "eval_ce": eval0["mean"], "eval_ppl": eval0["ppl"]})
        eval_tensors[0] = eval0
        _log(f"eval step=0 ce={eval0['mean']:.6f} ppl={eval0['ppl']:.4f}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(state["step"] + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for micro in range((step - 1) * ga, step * ga):
            batch = _batch_to_device(batches[micro % len(batches)], device)
            loss = _forward_loss(model, batch) / ga
            micro_losses.append(loss.detach() * ga)
            blocks, observers = _mute_route(model)
            try:
                loss.backward()
            finally:
                _restore_route(blocks, observers)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        with torch.no_grad():
            sq = torch.stack([p.grad.pow(2).sum() for p in params if p.grad is not None])
            grad_norm = torch.sqrt(sq.sum())
        optimizer.step()
        scheduler.step()
        step_loss = torch.stack(micro_losses).mean()
        train_loss_tensors.append(step_loss.detach().cpu())
        grad_tensors.append(grad_norm.detach().cpu())
        rec = {
            "step": step,
            "loss": float(step_loss),
            "grad_norm": float(grad_norm),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        events.append(rec)
        if step == 1 or step == steps or step % 5 == 0:
            _log(
                f"step {step}/{steps} loss={rec['loss']:.6f} "
                f"grad={rec['grad_norm']:.4f} lr={rec['lr']:.2e}"
            )
        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            ev = _eval_curve(model, eval_loader, device)
            eval_points.append(
                {"step": step, "eval_ce": ev["mean"], "eval_ppl": ev["ppl"]}
            )
            eval_tensors[step] = ev
            _log(f"eval step={step} ce={ev['mean']:.6f} ppl={ev['ppl']:.4f}")
        due = checkpoint_every > 0 and step % checkpoint_every == 0 and step < steps
        if checkpoint is not None and due:  # the last step is saved as the finished arm
            torch.cuda.synchronize()
            state["train_seconds"] += time.perf_counter() - t0  # checkpoint I/O isn't train time
            state["peak_vram_bytes"] = max(state["peak_vram_bytes"], _peak_vram())
            state["step"] = step
            _save(
                {
                    "adapters": {n: p.detach().cpu() for n, p in adapters.items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "state": state,
                },
                checkpoint,
                on_checkpoint,
            )
            _log(f"checkpoint step={step} -> {checkpoint}")
            t0 = time.perf_counter()

    torch.cuda.synchronize()
    elapsed = state["train_seconds"] + time.perf_counter() - t0
    return {
        "events": events,
        "eval_points": eval_points,
        "train_loss_tensors": train_loss_tensors,
        "grad_tensors": grad_tensors,
        "eval_tensors": eval_tensors,
        "adapter_tensors": {n: p.detach().cpu().clone() for n, p in adapters.items()},
        "adam_tensors": _adam_tensors(optimizer),
        "sec_per_step": elapsed / steps,
        "train_seconds": elapsed,
        "loss_first": events[0]["loss"],
        "loss_last": events[-1]["loss"],
        "peak_vram_bytes": max(state["peak_vram_bytes"], _peak_vram()),
        "resumed_at_steps": state["resumed_at_steps"],
    }


def run_quality(
    *,
    model_id: str,
    steps: int,
    seq_len: int,
    batch_size: int,
    ga: int,
    lr: float,
    warmup: int,
    seed: int,
    train_samples: int,
    eval_samples: int,
    eval_every: int,
    device: str,
    dataset: str = "synthetic",
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 25,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    import gc
    import shutil
    from itertools import islice

    import torch
    from transformers import AutoConfig

    from poros.core.arch_registry import registry_type
    from poros.core.determinism import enforce_determinism
    from poros.moe.arch import MOE_FAMILIES
    from poros.moe.prepare import MoeConfig, default_backends, prepare_moe, streaming_config
    from poros.train.datasets import _build_synthetic_dataloader, _load_hf_dataset_as_dataloader

    if device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("moe_synthetic_quality requires CUDA")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    enforce_determinism()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = AutoConfig.from_pretrained(model_id)
    if registry_type(config) not in MOE_FAMILIES:
        raise SystemExit(
            f"{model_id} has model_type={registry_type(config)!r}; "
            f"prepare_moe accepts {sorted(MOE_FAMILIES)}"
        )
    capacity = streaming_config(config).capacity_bytes
    gpu = _gpu_identity()
    _log(
        f"gpu={gpu['name']} dataset={dataset} "
        f"seq={seq_len} bs={batch_size} ga={ga} steps={steps} seed={seed}"
    )

    if dataset == "synthetic":
        train_loader = _build_synthetic_dataloader(
            n_samples=train_samples,
            max_seq_length=seq_len,
            batch_size=batch_size,
            seed=seed,
        )
        eval_loader = _build_synthetic_dataloader(
            n_samples=eval_samples,
            max_seq_length=seq_len,
            batch_size=batch_size,
            seed=seed + 1,
        )
    else:
        loader = _load_hf_dataset_as_dataloader(
            dataset_name=dataset,
            model_name_or_path=model_id,
            max_seq_length=seq_len,
            batch_size=batch_size,
            seed=seed,
        )
        # Materialized so both arms replay the same batches: a shuffling
        # loader's generator would advance between the arms' epochs. One
        # epoch draws without replacement, so eval rows never train.
        n_eval, rest = divmod(eval_samples, batch_size)
        batches = list(islice(loader, n_eval + steps * ga))
        if rest or not n_eval or len(batches) < n_eval + steps * ga:
            raise SystemExit(
                f"{dataset}: need eval_samples as a positive multiple of batch_size and "
                f"{n_eval + steps * ga} batches in one epoch, got {len(batches)}"
            )
        eval_loader, train_loader = batches[:n_eval], batches[n_eval:]
    backends = default_backends()
    arms: dict[str, dict[str, Any]] = {}
    run_dir = None
    if checkpoint_dir is not None:
        protocol = [model_id, dataset, steps, seq_len, batch_size, ga, lr, warmup, seed]
        protocol += [train_samples, eval_samples, eval_every, gpu["name"], gpu["torch"]]
        protocol += [gpu["transformers"]]
        run_dir = Path(checkpoint_dir) / _run_key(protocol)
        run_dir.mkdir(parents=True, exist_ok=True)
        _log(f"checkpoints in {run_dir} every {checkpoint_every} steps")

    for label, streamed in (("streamed", True), ("resident", False)):
        done = run_dir / f"{label}.done.pt" if run_dir is not None else None
        if done is not None and done.exists():
            arms[label] = torch.load(done, weights_only=True)
            _log(f"{label} arm already finished; loaded {done}")
            continue
        gc.collect()
        _reset_peak()
        model = _load_model(model_id)
        moe_cfg = MoeConfig(
            streamed=streamed,
            capacity_bytes=capacity if streamed else 0,
            look_ahead=1 if streamed else 0,
            lora_rank=16,
            lora_alpha=32,
            seed=seed,
            device=device,
        )
        _log(f"prepare_moe {label}")
        prepared = time.perf_counter()
        report = prepare_moe(model, moe_cfg, backends=backends)
        _log(
            f"prepare_moe {label} {time.perf_counter() - prepared:.1f}s "
            f"dense_streamed={report.dense_streamed}"
        )
        arm = _train_arm(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            steps=steps,
            ga=ga,
            lr=lr,
            warmup=warmup,
            weight_decay=0.01,
            max_grad_norm=1.0,
            device=device,
            eval_every=eval_every,
            checkpoint=run_dir / f"{label}.pt" if run_dir is not None else None,
            checkpoint_every=checkpoint_every,
            on_checkpoint=on_checkpoint,
        )
        arm["report"] = {
            "guarantee": report.guarantee,
            "streamed": report.streamed,
            "dense_streamed": report.dense_streamed,
            "experts_quantized": report.experts_quantized,
            "blocks_installed": report.blocks_installed,
        }
        arms[label] = arm
        if done is not None:
            _save(arm, done, on_checkpoint)
            (run_dir / f"{label}.pt").unlink(missing_ok=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    s, r = arms["streamed"], arms["resident"]
    loss_equal = all(
        torch.equal(a, b) for a, b in zip(s["train_loss_tensors"], r["train_loss_tensors"])
    )
    grad_equal = all(
        torch.equal(a, b) for a, b in zip(s["grad_tensors"], r["grad_tensors"])
    )
    adapter_names = sorted(s["adapter_tensors"])
    adapter_equal = adapter_names == sorted(r["adapter_tensors"]) and all(
        torch.equal(s["adapter_tensors"][n], r["adapter_tensors"][n]) for n in adapter_names
    )
    adam_names = sorted(s["adam_tensors"])
    adam_equal = adam_names == sorted(r["adam_tensors"]) and all(
        torch.equal(s["adam_tensors"][n], r["adam_tensors"][n]) for n in adam_names
    )
    eval_steps = sorted(s["eval_tensors"])
    eval_equal = eval_steps == sorted(r["eval_tensors"]) and all(
        torch.equal(
            s["eval_tensors"][k]["mean_tensor"], r["eval_tensors"][k]["mean_tensor"]
        )
        and all(
            torch.equal(a, b)
            for a, b in zip(
                s["eval_tensors"][k]["per_batch_tensors"],
                r["eval_tensors"][k]["per_batch_tensors"],
            )
        )
        for k in eval_steps
    )
    max_abs = max(
        float((a - b).abs().max())
        for a, b in zip(s["train_loss_tensors"], r["train_loss_tensors"])
    )
    parity_ok = bool(
        loss_equal
        and grad_equal
        and adapter_equal
        and adam_equal
        and eval_equal
        and max_abs == 0.0
    )

    def _public(arm: dict[str, Any]) -> dict[str, Any]:
        return {
            "report": arm["report"],
            "events": arm["events"],
            "eval_points": arm["eval_points"],
            "sec_per_step": arm["sec_per_step"],
            "train_seconds": arm["train_seconds"],
            "loss_first": arm["loss_first"],
            "loss_last": arm["loss_last"],
            "peak_vram_bytes": arm["peak_vram_bytes"],
            "resumed_at_steps": arm["resumed_at_steps"],
        }

    receipt = {
        "schema": "poros.moe.synthetic_quality.v1",
        "model_id": model_id,
        "protocol": {
            "dataset": f"synthetic:{train_samples}" if dataset == "synthetic" else dataset,
            "eval_dataset": (
                f"synthetic:{eval_samples}"
                if dataset == "synthetic"
                else f"{dataset}: first {eval_samples} rows of the shuffled train split"
            ),
            "eval_seed": seed + 1 if dataset == "synthetic" else None,
            "steps": steps,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "gradient_accumulation_steps": ga,
            "tokens_per_step": batch_size * ga * seq_len,
            "lr": lr,
            "warmup_steps": warmup,
            "lr_scheduler": "cosine",
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "seed": seed,
            "eval_every": eval_every,
            "note": (
                "Same loader as the poros paper synthetic cells "
                "(_SyntheticTokenDataset, vocab [100, 30000)). "
                "Default 50 steps matches the 5090/blog synthetic pass; "
                "canonical table cells are 200."
                if dataset == "synthetic"
                else "poros trainer HF loader (train split, one row per text field, "
                "padded to seq_len; tokens_per_step counts padding)."
            ),
        },
        "gpu": gpu,
        "parity": {
            "losses_bitwise_equal": bool(loss_equal),
            "grad_norms_bitwise_equal": bool(grad_equal),
            "adapter_weights_bitwise_equal": bool(adapter_equal),
            "adam_state_bitwise_equal": bool(adam_equal),
            "eval_ce_bitwise_equal": bool(eval_equal),
            "max_abs_loss_diff": max_abs,
            "passed": parity_ok,
        },
        "quality": {
            "streamed_loss_first": s["loss_first"],
            "streamed_loss_last": s["loss_last"],
            "resident_loss_first": r["loss_first"],
            "resident_loss_last": r["loss_last"],
            "streamed_eval": s["eval_points"],
            "resident_eval": r["eval_points"],
            "passed": bool(eval_equal),
        },
        "speed": {
            "streamed_sec_per_step": s["sec_per_step"],
            "resident_sec_per_step": r["sec_per_step"],
            "streamed_peak_vram_bytes": s["peak_vram_bytes"],
            "resident_peak_vram_bytes": r["peak_vram_bytes"],
        },
        "streamed": _public(s),
        "resident": _public(r),
        "passed": parity_ok,
    }
    if run_dir is not None:  # keep the receipt next to the run; drop the arms once it passed
        (run_dir.parent / f"{run_dir.name}.json").write_text(json.dumps(receipt, default=str))
        if receipt["passed"]:
            shutil.rmtree(run_dir)
        if on_checkpoint is not None:
            on_checkpoint()
    verdict = "PASS" if receipt["passed"] else "FAIL"
    _log(
        f"SYNTHETIC QUALITY {verdict} "
        f"loss {s['loss_first']:.6f}->{s['loss_last']:.6f} "
        f"max_abs={max_abs} eval_equal={eval_equal}"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="synthetic", help='HF id or "synthetic"')
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ga", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=512, help="synthetic only")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="moe_synthetic_quality.json")
    parser.add_argument("--checkpoint-dir", help="resume a killed run from here")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    receipt = run_quality(
        model_id=args.model,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        ga=args.ga,
        lr=args.lr,
        warmup=args.warmup,
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        eval_every=args.eval_every,
        device=args.device,
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )
    text = json.dumps(receipt, indent=2, default=str)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    print(text)


if __name__ == "__main__":
    main()
