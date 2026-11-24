"""Resident-vs-blockwise TRAINING parity gate.

Runs one (or more) full optimizer updates on the NF4 base twice: once with every block held
resident on GPU (plain autograd, no streaming, no rematerialization) and once through the
blockwise streaming path -- from identical initial state, and compares with ``torch.equal``:

* per-microstep losses
* adapter gradients immediately before the optimizer step
* adapter weights after the final step
* Adam optimizer state (exp_avg / exp_avg_sq / step)

Each arm runs in its own subprocess (identical seeds) so global RNG, CUDA
context, and determinism flags cannot leak between arms; the driver then
compares the saved tensor dumps bitwise.

Usage (on a CUDA box):
    python scripts/training_parity.py --g 1 --out parity_g1.json
    # gradient accumulation, and a long sequence:
    python scripts/training_parity.py --model Qwen/Qwen2.5-7B --block-size 4 --g 4 \
        --steps 200 --lora-dropout 0.0 --out parity_nf4_7b_g4.json
    python scripts/training_parity.py --model Qwen/Qwen2.5-32B --block-size 4 --g 1 \
        --steps 20 --seq-len 4096 --lora-dropout 0.0 --out parity_nf4_32b_seq4k.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from poros.validate.training_gate import SEED, build_trainer, compare, ran_strategy  # noqa: F401
from poros.validate.training_gate import make_batches as _make_batches  # noqa: F401
from poros.validate.training_gate import run_arm as _run_arm

MODEL = os.environ.get("POROS_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


def run_arm(
    arm: str,
    g: int,
    steps: int,
    seq_len: int,
    model_id: str,
    out_path: str,
    lora_dropout: float = 0.05,
    block_size: int = 2,
    load_strategy: str = "gpu",
) -> None:
    """One arm in this process, saved for the driver to compare. The arm itself lives in
    poros.validate.training_gate, where `poros prove <model> --unvalidated` runs it too."""
    import torch

    trainer = build_trainer(
        model_id, g=g, steps=steps, lora_dropout=lora_dropout,
        block_size=block_size, load_strategy=load_strategy,
    )
    dump = _run_arm(trainer, arm, g=g, steps=steps, seq_len=seq_len, block_major=False)
    torch.save(dump, out_path)
    print(f"[{arm}] saved {out_path} ({dump['seconds']:.1f}s)")


def _compare_files(resident_path: str, blockwise_path: str) -> dict:
    import torch

    return compare(torch.load(resident_path, weights_only=False),
                   torch.load(blockwise_path, weights_only=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["resident", "blockwise"])
    ap.add_argument("--g", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--quantization", choices=["nf4"], default="nf4",
                    help="The base's storage: nf4, the only one.")
    ap.add_argument(
        "--load-strategy", choices=["gpu", "block_streamed"], default="gpu",
        help="Run the gate with the block-streamed NF4 load "
        "(both arms load identically; parity must still be 0.00e+00).",
    )
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dump")
    ap.add_argument("--out", default="training_parity.json")
    args = ap.parse_args()

    if args.arm:
        run_arm(
            args.arm, args.g, args.steps, args.seq_len,
            args.model, args.dump,
            lora_dropout=args.lora_dropout,
            block_size=args.block_size,
            load_strategy=args.load_strategy,
        )
        return

    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for arm in ("resident", "blockwise"):
            paths[arm] = os.path.join(td, f"{arm}.pt")
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--arm", arm, "--g", str(args.g), "--steps", str(args.steps),
                "--seq-len", str(args.seq_len), "--model", args.model,
                "--dump", paths[arm],
                "--lora-dropout", str(args.lora_dropout),
                "--block-size", str(args.block_size),
                "--load-strategy", args.load_strategy,
            ]
            subprocess.run(cmd, check=True)
        record = {
            "model": args.model,
            "quantization": args.quantization,
            "load_strategy": ran_strategy(args.load_strategy),
            "block_size": args.block_size,
            "g": args.g,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "block_major": False,
            "lora_dropout": args.lora_dropout,
            **_compare_files(paths["resident"], paths["blockwise"]),
        }
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
