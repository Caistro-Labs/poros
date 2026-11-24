"""Resident-vs-blockwise parity gate under different PyTorch execution modes.

The trainer docstring claims the blockwise==resident equivalence is
*structural* -- it should hold regardless of SDPA backend, TF32, autocast, or
torch.compile, because both paths run identical layer math and differ only in
weight residency. This sweep measures that claim instead of asserting it.

For each mode the gate runs ``PorosTrainer._forward_selfcheck`` (blockwise,
load/compute/release) against ``_forward_selfcheck_resident`` (all blocks
resident) **inside the same mode context**, and compares with ``torch.equal``.
One JSON record per mode is appended to the output file.

Each mode runs in a fresh subprocess so global state (determinism flags, TF32,
compile caches) cannot leak across modes.

Usage:
    python scripts/parity_modes_sweep.py --all --out modes_sweep.jsonl
    python scripts/parity_modes_sweep.py --mode sdpa_flash  # single, in-proc
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time

MODEL = os.environ.get("POROS_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B")

MODES = [
    "baseline_deterministic",
    "sdpa_default_fast",
    "sdpa_flash",
    "sdpa_efficient",
    "sdpa_math",
    "tf32_on",
    "amp_bf16_autocast",
    "compile",
]


def _mode_context(mode: str, torch):
    """Return a context manager applying the mode to forward passes."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if mode == "sdpa_flash":
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    if mode == "sdpa_efficient":
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    if mode == "sdpa_math":
        return sdpa_kernel([SDPBackend.MATH])
    if mode == "amp_bf16_autocast":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_mode(mode: str, model_id: str = MODEL, steps: int = 3) -> dict:
    import torch

    from poros.train.trainer import PorosTrainer

    record: dict = {
        "mode": mode,
        "model": model_id,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    deterministic = mode == "baseline_deterministic"
    if mode == "tf32_on":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    trainer = PorosTrainer.from_pretrained(
        model_id,
        quantization="nf4",
        block_size=4,
        deterministic=deterministic,
        blockwise_arch_check="error",
    )
    assert trainer.block_manager is not None

    vocab = getattr(getattr(trainer.model, "config", None), "vocab_size", 32000)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, 64), device="cuda:0")

    resident_fwd = trainer._forward_selfcheck_resident
    blockwise_fwd = trainer._forward_selfcheck
    if mode == "compile":
        # Compile the transformer layers (what a user would do); the streaming
        # driver itself mutates param storage between layer calls, which is
        # data movement, not graph math, so it stays eager.
        bm = trainer.block_manager
        for bid in range(bm.num_blocks):
            for layer in bm.get_block_modules(bid):
                layer.compile()

    try:
        with torch.no_grad(), _mode_context(mode, torch):
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            resident = resident_fwd(trainer.model, input_ids)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            blockwise = blockwise_fwd(trainer.model, input_ids)
            # repeat blockwise to measure run-to-run reproducibility
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            blockwise2 = blockwise_fwd(trainer.model, input_ids)
    except RuntimeError as exc:
        if "No available kernel" in str(exc):
            # the forced SDPA backend cannot serve this call (e.g. flash
            # does not accept an explicit attention mask) -- an upstream
            # kernel constraint, not a Poros parity result
            record["skipped"] = f"kernel unavailable: {exc}"
            record["seconds"] = round(time.time() - t0, 1)
            return record
        raise
    record["gate_equal"] = bool(torch.equal(resident, blockwise))
    record["max_abs_diff"] = float((resident - blockwise).abs().max().item())
    record["rerun_equal"] = bool(torch.equal(blockwise, blockwise2))
    record["resident_loss"] = float(resident.item())
    record["blockwise_loss"] = float(blockwise.item())

    # short blockwise training-loss sanity in this mode
    losses = []
    with _mode_context(mode, torch):
        for _ in range(steps):
            loss = blockwise_fwd(trainer.model, input_ids)
            losses.append(float(loss.item()))
    record["train_losses_finite"] = all(math.isfinite(x) for x in losses)
    record["seconds"] = round(time.time() - t0, 1)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="modes_sweep.jsonl")
    args = ap.parse_args()

    if args.all:
        for mode in MODES:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--mode",
                    mode,
                    "--model",
                    args.model,
                    "--out",
                    args.out,
                ],
            )
            status = "ok" if proc.returncode == 0 else "FAILED"
            print(f"[sweep] {mode}: {status}", flush=True)
            if proc.returncode != 0:
                with open(args.out, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "mode": mode,
                                "model": args.model,
                                "error": f"subprocess exit {proc.returncode}",
                            }
                        )
                        + "\n"
                    )
        return

    record = run_mode(args.mode, args.model)
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
