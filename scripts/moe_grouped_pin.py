"""Pin, verify and benchmark poros.moe's grouped expert backend on one wide MoE layer.

    python scripts/moe_grouped_pin.py record
    python scripts/moe_grouped_pin.py verify --procs 3 --out receipt.json
    python scripts/moe_grouped_pin.py bench --out bench.json

The layer: hidden 4096, 288 experts of 2048, top-8, rank-16 expert LoRA, NF4
store, all experts resident (``nf4_block`` below: transformers' Glm5NextTextMoE router and
shared expert, the widest expert layout the backend was pinned on). ``verify``
runs each process with empty Triton / inductor caches (plus one warm) and
compares the sha256 of the expert output, the token and routing-weight
grads and every LoRA grad. ``bench`` times a Poros step (block forward
without grad, rematerialization with grad, backward) for the reference and
grouped computes, with and without the remat output reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

TOKENS = (2048, 4096)

# -- the layer: one real-width MoE layer on poros.moe (CUDA) -----------------------------------

H, INTER, EXPERTS, TOP_K = 4096, 2048, 288, 8


def text_config():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    return Glm5NextTextConfig(
        hidden_size=H,
        moe_intermediate_size=INTER,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=1,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        mlp_layer_types=["sparse"],
        pad_token_id=0,
        vocab_size=1024,
    )


def hf_block(cfg):
    """The HF sparse block with real router + shared expert, experts on the meta device."""
    import torch
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    with torch.device("meta"):
        block = Glm5NextTextMoE(cfg)
    block.gate.to_empty(device="cuda")
    block.shared_experts.to_empty(device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for p in [*block.gate.parameters(), *block.shared_experts.parameters()]:
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 0.02)
        block.gate.e_score_correction_bias = torch.zeros(EXPERTS, device="cuda")
    block.gate.to(torch.float32)
    block.shared_experts.to(torch.bfloat16)
    return block


def nf4_block(cfg, expert_batch: int = 8):
    import torch

    from poros.moe import quant
    from poros.moe.contracts import ExpertLayout
    from poros.moe.modules import ManagedMoeBlock
    from poros.moe.prepare import default_backends
    from poros.moe.store import ExpertSlabArena

    shapes = {"gate_up_proj": (2 * INTER, H), "down_proj": (H, INTER)}
    layout = ExpertLayout(
        num_layers=1,
        experts_per_layer=EXPERTS,
        top_k=TOP_K,
        tensor_names=("gate_up_proj", "down_proj"),
    )
    arena = ExpertSlabArena(layout, shapes)
    gen = torch.Generator().manual_seed(1)
    blobs = {
        n: quant.quantize_tensor(torch.randn(*s, generator=gen) * 0.02) for n, s in shapes.items()
    }
    for e in range(EXPERTS):  # one quantized expert's bytes for all: same compute, no 14 GB build
        for n, blob in blobs.items():
            start, end = arena._offsets[n]
            arena._slabs[(0, e)][start:end] = blob
    arena._unwritten.clear()
    store = arena.build(torch.bfloat16)
    residency = default_backends().build_residency(store, False, 0, 0, "cuda")
    block = ManagedMoeBlock(hf_block(cfg), 0, store, residency)
    block.expert_batch = expert_batch
    return block



def _layer(expert_compute: str, expert_batch: int, reuse: bool = False):
    import torch

    from poros.moe.adapters import ExpertLoraAdapter

    block = nf4_block(text_config(), expert_batch=expert_batch)
    torch.manual_seed(2)
    block.adapter = ExpertLoraAdapter(EXPERTS, H, INTER, rank=16, alpha=32, device="cuda")
    with torch.no_grad():
        for p in block.adapter.parameters():
            p.normal_(0, 0.01, generator=torch.Generator(device="cuda").manual_seed(3))
    block.residency.ensure([(0, e) for e in range(EXPERTS)])
    block.expert_compute = expert_compute
    block.reuse_remat_output = reuse
    return block


def _x(tokens):
    import torch

    gen = torch.Generator(device="cuda").manual_seed(10 + tokens)
    return (torch.randn(1, tokens, 4096, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)


def _sha(t) -> str:
    import torch

    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def cmd_record(args) -> None:
    from poros.moe import grouped

    pin = {"env": grouped.environment(), "note": args.note}
    path = args.out or str(grouped.default_pin_path(pin["env"]))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(pin, fh, indent=1, sort_keys=True)
    print("pinned ->", path)


def cmd_run(args) -> None:
    import torch

    from poros.core.determinism import enforce_determinism
    from poros.moe import grouped

    enforce_determinism()
    grouped.activate()
    block = _layer("grouped", args.expert_batch)
    result = {}
    for tokens in TOKENS:
        x = _x(tokens).requires_grad_(True)
        torch.manual_seed(5)
        out = block(x)
        up = torch.randn(
            out.shape,
            generator=torch.Generator(device="cuda").manual_seed(6),
            device="cuda",
            dtype=out.dtype,
        )
        for p in block.adapter.parameters():
            p.grad = None
        out.backward(up)
        torch.cuda.synchronize()
        hashes = {"out": _sha(out), "dx": _sha(x.grad)}
        hashes.update({n: _sha(p.grad) for n, p in block.adapter.named_parameters()})
        result[str(tokens)] = hashes
    with open(args.save, "w") as fh:
        json.dump(result, fh, indent=1)


def _spawn(save, dirs, expert_batch):
    env = {**os.environ, "TRITON_CACHE_DIR": dirs[0], "TORCHINDUCTOR_CACHE_DIR": dirs[1]}
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "run",
        "--save",
        save,
        "--expert-batch",
        str(expert_batch),
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    with open(save) as fh:
        out = json.load(fh)
    return out, round(time.time() - t0, 1)


def cmd_verify(args) -> None:
    from poros.moe import grouped

    runs, wall = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        dirs = None
        for i in range(args.procs):
            dirs = (tempfile.mkdtemp(dir=tmp), tempfile.mkdtemp(dir=tmp))
            runs[f"cold_{i}"], wall[f"cold_{i}"] = _spawn(
                os.path.join(tmp, f"p{i}.json"), dirs, args.expert_batch
            )
        runs["warm"], wall["warm"] = _spawn(os.path.join(tmp, "w.json"), dirs, args.expert_batch)
    ref = runs["cold_0"]
    equal = all(r == ref for r in runs.values())
    receipt = {
        "gate": "moe_grouped_pin verify (grouped expert backend)",
        "env": grouped.environment(),
        "tokens": list(TOKENS),
        "expert_batch": args.expert_batch,
        "processes": list(runs),
        "byte_identical": equal,
        "tensors_per_length": len(ref[str(TOKENS[0])]),
        "wall_s": wall,
        "hashes": runs,
    }
    with open(args.out, "w") as fh:
        json.dump(receipt, fh, indent=1)
    print(json.dumps({k: receipt[k] for k in ("byte_identical", "tensors_per_length", "wall_s")}))
    if not equal:
        sys.exit(1)


def cmd_bench(args) -> None:
    import torch

    from poros.core.determinism import enforce_determinism
    from poros.moe import grouped

    enforce_determinism()
    grouped.activate()
    rows = []
    variants = [
        ("reference", 8, False),
        ("reference", 8, True),
        ("grouped", 8, False),
        ("grouped", 8, True),
        ("grouped", 32, True),
    ]
    for compute, eb, reuse in variants:
        block = _layer(compute, eb, reuse)
        for tokens in args.tokens:
            x = _x(tokens)

            def step(block=block, x=x):
                torch.manual_seed(5)
                with torch.no_grad():
                    block(x)
                torch.manual_seed(5)
                xi = x.detach().requires_grad_(True)
                out = block(xi)
                out.backward(torch.ones_like(out))

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                step()
                times = []
                for _ in range(args.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                row = {
                    "compute": compute,
                    "expert_batch": eb,
                    "remat_reuse": reuse,
                    "tokens": tokens,
                    "step_ms": round(statistics.median(times) * 1000, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "remat_hits": block.remat_hits,
                }
            except torch.OutOfMemoryError:
                row = {
                    "compute": compute,
                    "expert_batch": eb,
                    "remat_reuse": reuse,
                    "tokens": tokens,
                    "step_ms": "OOM",
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del block
        torch.cuda.empty_cache()
    with open(args.out, "w") as fh:
        json.dump(
            {
                "bench": "one real-width MoE layer (NF4, 288x2048, top-8, LoRA r16): "
                "no-grad forward + remat forward + backward, median",
                "reps": args.reps,
                "env": grouped.environment(),
                "rows": rows,
            },
            fh,
            indent=1,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument(
        "--note",
        default="grouped expert backend: torch.nn.functional.grouped_mm + chunked backward",
    )
    r.add_argument("--out")
    r = sub.add_parser("run")
    r.add_argument("--save", required=True)
    r.add_argument("--expert-batch", type=int, default=32)
    r = sub.add_parser("verify")
    r.add_argument("--procs", type=int, default=3)
    r.add_argument("--expert-batch", type=int, default=32)
    r.add_argument("--out", required=True)
    r = sub.add_parser("bench")
    r.add_argument("--tokens", type=int, nargs="+", default=[2048, 4096, 8192])
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--out", required=True)
    args = p.parse_args()
    {"record": cmd_record, "run": cmd_run, "verify": cmd_verify, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
