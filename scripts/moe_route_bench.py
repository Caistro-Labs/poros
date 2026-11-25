"""Interleaved A/B benchmark of the MoE routing path on real CPU or CUDA tensors.

From the candidate checkout:
    git worktree add --detach ../poros-base 0f1f9ab
    python -m scripts.moe_route_bench --baseline ../poros-base --device cuda

The baseline is its checkout's original fused.py, loaded alongside the candidate so both
see the same tensors, thread count and allocator. CUDA times synchronize at both ends.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from poros.moe import fused
from poros.moe.modules import _rng_state


def load_baseline(root):
    path = root / "poros/moe/fused.py"
    spec = importlib.util.spec_from_file_location("poros_route_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure(fn, iterations, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter_ns() - t) / iterations / 1000


def compare(old, new, device, rounds, iterations):
    for f in (old, new):
        measure(f, 3, device)
    samples = [[], []]
    for r in range(rounds):
        for arm in ((0, 1) if r % 2 == 0 else (1, 0)):
            samples[arm].append(measure((old, new)[arm], iterations, device))
    medians = [statistics.median(s) for s in samples]
    ratios = [a / b for a, b in zip(*samples)]
    return {"baseline_us": medians[0], "candidate_us": medians[1],
            "paired_speedup_median": statistics.median(ratios),
            "paired_speedup_iqr": np.percentile(ratios, [25, 75]).tolist(),
            "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--with-masks", action="store_true",
                        help="also time generating both full-size Qwen expert dropout masks")
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.threads) < 1:
        parser.error("rounds, iterations and threads must be positive")
    torch.set_num_threads(args.threads)
    baseline = load_baseline(args.baseline)
    d = torch.device(args.device)
    result = {"platform": platform.platform(), "torch": str(torch.__version__),
              "device": torch.cuda.get_device_name(d) if d.type == "cuda" else "cpu",
              "threads": torch.get_num_threads(), "rounds": args.rounds,
              "iterations": args.iterations, "seed": 2026, "shapes": {}}
    gen = torch.Generator().manual_seed(2026)
    for name, lengths, active in [("one", [512], 128), ("eight", [512] * 8, 128),
                                  ("ragged", [0, 511, 512, 70, 1, 601, 340, 19], 16)]:
        idx = torch.rand(sum(lengths), active, generator=gen).topk(8).indices.to(d)
        plan = fused.Plan.build(idx, 128, lengths)
        ref = baseline.Plan.build(idx, 128, lengths)
        for key in vars(ref):
            a, b = getattr(plan, key), getattr(ref, key)
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b, key
        cases = {
            "host_tables": (lambda: baseline._plan_tables(plan.counts, 128, len(lengths), 64),
                            lambda: fused._plan_tables(plan.counts, 128, len(lengths), 64)),
            "build": (lambda: baseline.Plan.build(idx, 128, lengths),
                      lambda: fused.Plan.build(idx, 128, lengths)),
            "build_known_counts": (lambda: baseline.Plan.build(idx, 128, lengths, plan.counts),
                                   lambda: fused.Plan.build(idx, 128, lengths, plan.counts)),
        }
        if len(lengths) > 1:
            cases["dropout_row_indices"] = (
                lambda: [(ref.row_seg == g).nonzero().flatten() for g in range(ref.segments)],
                lambda: fused._dropout_rows(plan))
            assert all(torch.equal(a, b) for a, b in zip(
                *(f() for f in cases["dropout_row_indices"]), strict=True))
            if args.with_masks:
                states = [_rng_state(d) for _ in lengths]
                cases["draw_both_masks"] = (
                    lambda: baseline.Masks(ref, 0.1, 2048, 768, d, states),
                    lambda: fused.Masks(plan, 0.1, 2048, 768, d, states))
        result["shapes"][name] = {label: compare(
            *pair, d, args.rounds, 1 if label == "draw_both_masks" else args.iterations)
            for label, pair in cases.items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
