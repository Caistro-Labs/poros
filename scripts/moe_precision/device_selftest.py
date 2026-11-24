"""One-file self-test: does this machine's GPU store ExpertAdamW's rule, bit for bit?

ExpertAdamW (poros/train/expert_adamw.py) defines its bf16 update as one function, operation by
operation, and runs it as a fused kernel on CUDA (Triton) and on a Mac (Metal), or as its
reference. Run it from a checkout of the commit under test, with that checkout's Python:

    python scripts/moe_precision/device_selftest.py            # mps where there is one, else cuda
    python scripts/moe_precision/device_selftest.py --device cuda
    python scripts/moe_precision/device_selftest.py --time     # and one step's time (about 1 GB)

What it checks, each line "ok" or "FAIL":
  1. the keyed bits of all three draws on the device equal the CPU's
  2. the stochastic rounding on the device equals the CPU's
  3. the fused kernel's whole update (weight, remainder, m, v) equals the reference's, bit for
     bit: the optimizer's own probe (products in F's window at 2**-126, a weight on a bf16 tie),
     then three AdamW settings over three updates fed back, each at an element count that is a
     multiple of 16 and at one that is not (Triton compiles each its own variant), over inputs
     that exercise every operation (many exponents, both signs, zeros, subnormals, 2**-126, the
     largest finite values, a NaN, infinities, a weight that rounds to inf, the last partial
     block); a FAIL names the first differing element, its inputs and both outputs
  4. the kernel run twice gives the same bits
  5. the optimizer, built here, chooses the kernel (its own check at construction passed)
Pass: exit code 0 and every check line "ok"; the last line is one JSON object for the record.
Timing (--time) is printed separately and decides nothing. Takes under a minute.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's poros

import torch  # noqa: E402

from poros.train import expert_adamw as ea  # noqa: E402

SETTINGS = [(2e-4, (0.9, 0.999), 1e-8, 0.01), (2e-5, (0.99, 0.95), 1e-6, 0.1),
            (1e-3, (0.0, 0.5), 0.0, 0.0)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--time", action="store_true",
                    help="also time one step over 128M elements against torch's AdamW")
    args = ap.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else None)
    if device is None:
        print("no mps or cuda device here: nothing to compare the CPU with")
        return 2
    results: dict = {"device": device, "torch": torch.__version__, "rule": ea.RULE,
                     "machine": platform.platform(), "checks": {}}

    def check(name: str, ok: bool, detail: str = "") -> None:
        results["checks"][name] = bool(ok)
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    # 1. the bits of every draw, over keys with every high bit pattern and indices past 2**24
    for key in (0, 1, 2 ** 32 - 1, 2 ** 63 - 25, ea.update_key(42, 7, "a.adapter.lora_A_down")):
        for start, n in ((0, 1 << 16), ((1 << 24) + 3, 4099), ((1 << 31) - 17, 64)):
            for draw in (ea.DRAW_V, ea.DRAW_M, ea.DRAW_REMAINDER):
                want = ea._keyed_bits(key, start, n, draw)
                got = ea._keyed_bits(key, start, n, draw, device).cpu()
                same = torch.equal(got, want)
                check(f"bits key={key} start={start} draw={draw}", same,
                      "" if same else f"{int((got != want).sum())} of {n} differ")

    # 2. the rounding, given the same bits, over many exponents and both signs
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1 << 18, generator=g) * torch.logspace(-30, 3, 1 << 18)
    x[:4] = torch.tensor([0.0, -0.0, float("inf"), float("nan")])
    bits = ea._keyed_bits(9, 0, x.numel(), ea.DRAW_V)
    want = ea._stochastic_round_bf16(x, bits)
    got = ea._stochastic_round_bf16(x.to(device), bits.to(device)).cpu()
    same = torch.equal(got.view(torch.int16), want.view(torch.int16))
    check("stochastic rounding", same,
          "" if same else f"{int((got.view(torch.int16) != want.view(torch.int16)).sum())} differ")

    # 3, 4. the kernel's whole update against the reference, and against itself
    why = ea._kernel_unavailable(torch.device(device).type)
    results["kernel_unavailable"] = why
    if why is not None:
        check("the fused kernel can run here", False, why)
    else:
        diff = ea._probe_differences(device)
        check("fused kernel, the optimizer's own probe (F's edge, a bf16 tie, both Triton "
              "variants)", diff is None, diff or "every setting, bit for bit")
        for n in (2 * ea.BLOCK + 1040, 2 * ea.BLOCK + 1037):  # a multiple of 16, and not
            inputs = ea._probe_inputs(n, 11)
            for i, (lr, betas, eps, wd) in enumerate(SETTINGS):
                s = ea._scalars(lr, betas[0], betas[1], eps, wd, 4)
                key = ea.update_key(3, 4 + i, "model.layers.2.mlp.adapter.lora_B_gate_up")
                want = ea._run_steps("reference", "cpu", inputs, 3, key, s)
                got = ea._run_steps("kernel", device, inputs, 3, key, s)
                diff = ea._first_difference(got, want, inputs)
                check(f"fused kernel, whole update, {n} elements, lr {lr:g} betas {betas} "
                      f"eps {eps:g} wd {wd:g}", diff is None,
                      diff or "weight, remainder, m and v bit for bit")
                if i == 0:
                    again = ea._run_steps("kernel", device, inputs, 3, key, s)
                    diff = ea._first_difference(again, got, inputs)
                    check(f"fused kernel, run twice, {n} elements", diff is None,
                          diff or "the same bits")

    # 5. what the optimizer chooses here
    p = torch.nn.Parameter(torch.zeros(4096, dtype=torch.bfloat16, device=device))
    built = ea.ExpertAdamW([("w", p)], lr=1e-3)
    chosen = built.implementation[str(p.device)]
    results["implementation"] = chosen
    check("the optimizer chooses the fused kernel", chosen["path"] == "kernel",
          built.notice or chosen["why"])

    if args.time:
        sync = getattr(torch, torch.device(device).type).synchronize
        params = [torch.nn.Parameter(torch.zeros(1 << 21, dtype=torch.bfloat16, device=device))
                  for _ in range(64)]
        for q in params:
            q.grad = torch.full_like(q, 1e-4)
        timed = {}
        for name, opt in (("expert_adamw", ea.ExpertAdamW([(f"p{i}", q) for i, q in
                                                            enumerate(params)], lr=2e-4)),
                          ("torch_adamw_bf16", torch.optim.AdamW(params, lr=2e-4))):
            for _ in range(2):
                opt.step()
            sync()
            t0 = time.perf_counter()
            for _ in range(10):
                opt.step()
            sync()
            timed[name] = (time.perf_counter() - t0) / 10
            opt.state.clear()
        results["step_seconds"] = timed
        print(f"time: one step over 128M bf16 elements: ExpertAdamW {timed['expert_adamw']:.4f} "
              f"s, torch AdamW (bf16) {timed['torch_adamw_bf16']:.4f} s", flush=True)
    ok = all(results["checks"].values())
    results["pass"] = ok
    print(json.dumps(results))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
