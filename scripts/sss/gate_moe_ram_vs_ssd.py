"""G-SSS-MoE: RAM-store vs SSD-store MoE training must be bitwise equal.

Every arm trains the same model through ``PorosTrainer`` (the user path:
streamed load, ``prepare_moe``, dense streaming on CUDA, schedule replay,
capacity growth) on the deterministic ``synthetic`` data, and records:

- every step's loss (exact, as ``float.hex``);
- a blake2b digest of every micro-batch's logits bytes (lm_head output);
- a blake2b digest of every step's adapter gradient bytes, before clipping;
- the final adapter weights (``<arm>_adapter.safetensors``);
- s/step, peak VRAM, peak host RSS and the SSD source's counters.

``compare`` then requires equal loss lists, equal digest lists and
``torch.equal`` final adapter weights between the RAM arm and each SSD arm
(over the steps both ran), and writes the receipt. Byte-digest equality is
stricter than ``torch.equal`` (it also separates -0.0 from 0.0).

Arms: ``ram`` (the reference); ``ssd`` (builds the ``.sss`` on its first run
and reuses it after; the VRAM cache grows after step 1); ``ssd-fixedcap``
(reuses the file, cache kept at the receipt size: a second capacity value).

Usage (one arm per process; wrap GPU jobs in flock on shared boxes)::

    python scripts/sss/gate_moe_ram_vs_ssd.py truncate --model DIR --layers 6 --dest DIR6
    python scripts/sss/gate_moe_ram_vs_ssd.py arm --model DIR --arm ram --out OUT --steps 200
    python scripts/sss/gate_moe_ram_vs_ssd.py arm --model DIR --arm ssd --store-path S --out OUT
    python scripts/sss/gate_moe_ram_vs_ssd.py arm --model DIR --arm ssd-fixedcap --store-path S \\
        --out OUT
    python scripts/sss/gate_moe_ram_vs_ssd.py compare --out OUT --receipt receipt.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# This checkout's poros, not whatever an editable install points at.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


def truncate(args: argparse.Namespace) -> None:
    """A K-layer view of a checkpoint: new config + index, shards symlinked (no copy)."""
    src, dest = Path(args.model).resolve(), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    k = int(args.layers)
    config["num_hidden_layers"] = k
    for key in ("layer_types", "mlp_only_layers"):
        if isinstance(config.get(key), list) and key == "layer_types":
            config[key] = config[key][:k]
        elif isinstance(config.get(key), list):
            config[key] = [i for i in config[key] if i < k]
    if "max_window_layers" in config:
        config["max_window_layers"] = min(config["max_window_layers"], k)
    (dest / "config.json").write_text(json.dumps(config, indent=1))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    keep = {}
    for name, shard in index["weight_map"].items():
        if ".layers." in name and int(name.split(".layers.")[1].split(".")[0]) >= k:
            continue
        keep[name] = shard
    (dest / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": keep}, indent=1))
    for shard in sorted(set(keep.values())):
        link = dest / shard
        if not link.exists():
            link.symlink_to(src / shard)
    for extra in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt"):
        if (src / extra).exists() and not (dest / extra).exists():
            (dest / extra).symlink_to(src / extra)
    _stamp(f"truncated {src} to {k} layers at {dest}: {len(keep)} tensors, "
           f"{len(set(keep.values()))} shards linked")


class _DirectRun:
    """The trainer's surface for families ``PorosTrainer`` does not route (mixtral):
    ``load_moe_streamed`` + ``prepare_moe`` with the streaming config, AdamW on the expert
    adapters, deterministic synthetic ids."""

    def __init__(self, args: argparse.Namespace, ssd: bool) -> None:
        import torch

        from poros.moe.prepare import prepare_moe, streaming_config

        self.args, self.ssd = args, ssd
        self._sss_probe = None
        ssd_options = None
        if ssd:
            from poros.sss.probe import probe

            self._sss_probe = probe(args.store_path)
            ssd_options = {"ram_budget": _bytes(args.sss_ram or "4G"),
                           "bandwidth": self._sss_probe.bandwidth,
                           "latency_bandwidth": self._sss_probe.b_ssd_qd1,
                           "threads": max(8, self._sss_probe.queue_depth)}
            if os.environ.get("POROS_SSS_DEPTH"):
                ssd_options["depth"] = int(os.environ["POROS_SSS_DEPTH"])
        torch.manual_seed(42)
        from poros.moe.load import load_moe_streamed

        model = load_moe_streamed(args.model, torch.bfloat16,
                                  store="ssd" if ssd else "ram",
                                  store_path=args.store_path if ssd else None,
                                  ssd_options=ssd_options)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: dry runs
        cfg = streaming_config(model.config, lora_rank=16, lora_alpha=32, lora_dropout=0.05,
                               seed=42, device=self.device)
        if args.expert_batch:  # fewer experts dequantized at once (small GPUs)
            cfg = dataclasses.replace(cfg, expert_batch=args.expert_batch)
        self.report = prepare_moe(model, cfg)
        from poros.moe.modules import ManagedMoeBlock

        store = next(m.store for m in model.modules() if isinstance(m, ManagedMoeBlock))
        model._poros_moe_store = store
        self.model = model
        self._sss_source = store.source if hasattr(store.source, "mark_step") else None
        self.history.losses: list[float] = []
        self.history.step_times: list[float] = []
        self.config = SimpleNamespace(seed=42, lora_dropout=0.05,
                                      sss_ram=_bytes(args.sss_ram or "4G"))
        params = [p for _, p in sorted(model.named_parameters()) if p.requires_grad]
        self._opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)

    def _adapter_optimizer_step(self) -> None:
        self._opt.step()
        self._opt.zero_grad(set_to_none=True)

    def train(self, _dataset, *, steps: int, seq_len: int) -> dict:
        import torch

        import poros.sss.tune as tune

        cfg = self.model.config
        vocab = getattr(getattr(cfg, "text_config", cfg), "vocab_size")
        self.model.train()
        for step in range(steps):
            t0 = time.perf_counter()
            gen = torch.Generator().manual_seed(1000 + step)
            ids = torch.randint(1, vocab, (1, seq_len), generator=gen).to(self.device)
            loss = self.model(input_ids=ids, labels=ids).loss
            loss.backward()
            if self._sss_source is not None:
                self._sss_source.mark_step()
                if step == 0:
                    tune.grow_expert_cache(self.model, self.device)
            self._adapter_optimizer_step()
            self.history.losses.append(float(loss.detach()))
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.history.step_times.append(time.perf_counter() - t0)
            if step % self.args.logging_steps == 0:
                print(f"step {step:>4}/{steps}  loss {self.history.losses[-1]:.4f}  "
                      f"{self.history.step_times[-1]:.2f} s/step", flush=True)
        return self._compute_metrics()

    def _compute_metrics(self) -> dict:
        import torch

        cuda = self.device == "cuda"
        return {"peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0,
                "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if cuda else 0,
                "moe_guarantee": self.report.guarantee}


def _bytes(v) -> int:
    from poros.core.config import parse_bytes

    return parse_bytes(v) if isinstance(v, str) else int(v)


class _StopTraining(Exception):
    """Ends an arm after ``--stop-after`` steps of a ``--steps`` schedule."""


def _digest_tensor(h, t) -> None:
    import torch

    flat = t.detach().contiguous().reshape(-1)
    h.update(flat.view(torch.uint8).cpu().numpy().tobytes())


def run_arm(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import save_file

    import poros.sss.tune as tune
    from poros.train.trainer import PorosTrainer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ssd = args.arm.startswith("ssd")
    label = args.label or args.arm
    if ssd != label.startswith("ssd") or (not ssd and label != "ram"):
        raise SystemExit(f"label {label!r} must be 'ram' for the RAM arm and start with "
                         "'ssd' for an SSD arm (compare pairs them by name)")
    if args.arm == "ssd-fixedcap":
        tune.grown_capacity = lambda capacity, *a, **k: int(capacity)
    sss_path = None
    if ssd:
        sss_path = list(Path(args.store_path).glob("*.moe.sss")) if args.store_path else []
    reused_file = bool(sss_path)
    t0 = time.perf_counter()
    _stamp(f"arm {label}: loading {args.model} (store={'ssd' if ssd else 'ram'}, "
           f".sss present={reused_file})")
    if args.entry == "direct":
        trainer = _DirectRun(args, ssd)
    else:
        trainer = PorosTrainer.from_pretrained(
            args.model,
            store="ssd" if ssd else "ram",
            store_path=args.store_path if ssd else None,
            output_dir=str(out / f"run_{label}"),
            max_steps=args.steps,
            max_seq_length=args.seq_len,
            gradient_accumulation_steps=args.grad_accum,
            per_device_batch_size=1,
            save_steps=0,
            eval_steps=0,
            logging_steps=args.logging_steps,
            dataset="synthetic",
            progress=True,
            **({"sss_ram": args.sss_ram} if args.sss_ram else {}),
        )
    load_s = time.perf_counter() - t0
    _stamp(f"loaded in {load_s:.1f} s; rss {_rss_gib():.2f} GiB")
    model = trainer.model
    logits_digests: list[str] = []

    def on_logits(_module, _inp, output):
        h = hashlib.blake2b(digest_size=16)
        _digest_tensor(h, output)
        logits_digests.append(h.hexdigest())

    model.get_output_embeddings().register_forward_hook(on_logits)
    named = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
    grad_digests: list[str] = []
    step_fn = trainer._adapter_optimizer_step

    stop = args.stop_after or args.steps

    source = trainer._sss_source
    sss_steps: list[dict] = []
    read_log: list | None = [] if (args.read_log and source is not None) else None
    if read_log is not None:
        inner_read, inner_mark = source.read_into, source.mark_step

        def logged_read(key, dst):
            hit = tuple(key) in source._inflight
            inner_read(key, dst)
            read_log.append([key[0], key[1], int(hit)])

        def logged_mark():
            read_log.append("MARK")
            inner_mark()

        source.read_into, source.mark_step = logged_read, logged_mark

    def hashed_step():
        if source is not None:
            st = source.stats
            sss_steps.append({k: st[k] for k in ("reads", "prefetch_hits", "sync_reads",
                                                 "prefetch_dropped", "bytes_read",
                                                 "blocked_s", "depth")})
        if len(grad_digests) >= stop:  # same schedule as a full run, fewer steps
            raise _StopTraining
        h = hashlib.blake2b(digest_size=16)
        for name, p in named:
            h.update(name.encode())
            if p.grad is None:
                h.update(b"<none>")
            else:
                _digest_tensor(h, p.grad)
        grad_digests.append(h.hexdigest())
        step_fn()

    trainer._adapter_optimizer_step = hashed_step
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        metrics = trainer.train("synthetic", steps=args.steps, seq_len=args.seq_len)
    except _StopTraining:
        metrics = trainer._compute_metrics()
        del logits_digests[stop * args.grad_accum:]
    adapter = {n: p.detach().to("cpu").contiguous() for n, p in named}
    save_file(adapter, str(out / f"{label}_adapter.safetensors"))
    h = hashlib.blake2b(digest_size=16)
    for n, t in sorted(adapter.items()):
        h.update(n.encode())
        _digest_tensor(h, t)
    record = {
        "arm": label,
        "kind": args.arm,
        "model": args.model,
        "store": "ssd" if ssd else "ram",
        "sss_file_reused": reused_file if ssd else None,
        "config": {"steps": args.steps, "stop_after": args.stop_after,
                   "expert_batch": args.expert_batch, "entry": args.entry,
                   "seq_len": args.seq_len,
                   "grad_accum": args.grad_accum, "dataset": "synthetic",
                   "seed": trainer.config.seed, "lora_dropout": trainer.config.lora_dropout,
                   "sss_ram": trainer.config.sss_ram if ssd else None},
        "losses_hex": [float(x).hex() for x in trainer.history.losses],
        "losses": list(trainer.history.losses),
        "logits_blake2b": logits_digests,
        "grads_blake2b": grad_digests,
        "adapter_blake2b": h.hexdigest(),
        "adapter_tensors": len(adapter),
        "adapter_params": sum(t.numel() for t in adapter.values()),
        "load_s": load_s,
        "s_per_step": list(trainer.history.step_times),
        "peak_vram_allocated_gb": metrics.get("peak_vram_allocated_gb"),
        "peak_vram_reserved_gb": metrics.get("peak_vram_reserved_gb"),
        "peak_host_rss_gib": _rss_gib(),
        "sss_stats": None if source is None else source.stats,
        "sss_steps": sss_steps,
        "speed": _speed(trainer.history.step_times, sss_steps,
                        getattr(trainer._sss_probe, "b_ssd", None)),
        "sss_probe": None if trainer._sss_probe is None else vars(trainer._sss_probe),
        "pinning": _pinning_status(),
        "store_plugin": _store_plugin(trainer),
        "staging": _staging(_pinning_status(), source),
        "env": _env(),
    }
    (out / f"{label}.json").write_text(json.dumps(record, indent=1, default=str))
    if read_log is not None:
        Path(args.read_log).write_text(json.dumps(read_log))
    _stamp(f"arm {label} done: {len(trainer.history.losses)} steps, "
           f"median s/step "
           f"{_median(trainer.history.step_times[1:] or trainer.history.step_times):.2f}, "
           f"rss {_rss_gib():.2f} GiB")


def _store_plugin(trainer) -> dict | None:
    """``{name, version}`` of the expert-store plugin the run used (poros.stores), or None."""
    model = getattr(trainer, "model", None)
    name = getattr(model, "_poros_store_plugin", None)
    if name is None:
        return None
    from importlib.metadata import PackageNotFoundError, version

    from poros.stores import available_stores

    dist = available_stores().get(name)
    try:
        ver = version(dist) if dist else None
    except PackageNotFoundError:
        ver = None
    return {"name": name, "distribution": dist, "version": ver}


def _pinning_status() -> dict:
    from poros.sss.pinning import status

    return status()


def _staging(pinning_status: dict, source) -> dict:
    from poros.sss.pinning import staging

    return staging(pinning_status, ring_pinned=None if source is None
                   else bool(source.stats.get("pinned")))


def _speed(step_times: list, sss_steps: list, b_ssd: float | None) -> dict:
    """Steady-state numbers (step 0, the cold schedule-recording step, left out)."""
    out = {"s_per_step_median": _median(step_times[1:] or step_times)}
    if len(sss_steps) >= 3:
        first, last = sss_steps[1], sss_steps[-1]
        n = len(sss_steps) - 2  # steps 1 .. N-2 completed between the two snapshots
        reads = last["reads"] - first["reads"]
        gb = (last["bytes_read"] - first["bytes_read"]) / n / 1e9
        out.update(gb_per_step=gb, reads_per_step=reads / n,
                   sync_read_frac=(last["sync_reads"] - first["sync_reads"]) / max(reads, 1),
                   dropped_per_step=(last["prefetch_dropped"] - first["prefetch_dropped"]) / n,
                   blocked_s_per_step=(last["blocked_s"] - first["blocked_s"]) / n)
        if b_ssd:
            floor = gb * 1e9 / b_ssd
            out.update(disk_floor_s=floor, frac_of_disk_floor=floor / out["s_per_step_median"])
    return out


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _env() -> dict:
    import torch
    import transformers

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent).stdout.strip()
    except OSError:
        sha = None
    commit_file = Path(__file__).resolve().parents[2] / "COMMIT"
    if not sha and commit_file.exists():  # a git-archive checkout
        sha = commit_file.read_text().strip()
    return {"host": platform.node(), "python": sys.version.split()[0],
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "commit": sha,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def compare(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import load_file

    out = Path(args.out)
    ref = json.loads((out / "ram.json").read_text())
    ref_adapter = load_file(str(out / "ram_adapter.safetensors"))
    results = {}
    ok = True
    for path in sorted(out.glob("ssd*.json")):
        arm = json.loads(path.read_text())
        n = min(len(arm["losses_hex"]), len(ref["losses_hex"]))
        if arm["config"]["steps"] != ref["config"]["steps"]:
            raise SystemExit(f"{arm['arm']} ran a {arm['config']['steps']}-step schedule, "
                             f"ram a {ref['config']['steps']}-step one: not comparable")
        full = len(arm["losses_hex"]) == len(ref["losses_hex"])
        m = min(len(arm["logits_blake2b"]), len(ref["logits_blake2b"]))
        res = {
            "steps_compared": n,
            "losses_equal": arm["losses_hex"][:n] == ref["losses_hex"][:n],
            "logits_equal": arm["logits_blake2b"][:m] == ref["logits_blake2b"][:m],
            "logits_compared": m,
            "grads_equal": arm["grads_blake2b"][:n] == ref["grads_blake2b"][:n],
            "first_loss_mismatch": next((i for i in range(n)
                                         if arm["losses_hex"][i] != ref["losses_hex"][i]), None),
        }
        if full:
            adapter = load_file(str(out / f"{arm['arm']}_adapter.safetensors"))
            res["adapter_torch_equal"] = (adapter.keys() == ref_adapter.keys() and all(
                torch.equal(adapter[k], ref_adapter[k]) for k in ref_adapter))
        else:
            res["adapter_torch_equal"] = None  # a shorter run: its weights are earlier
        res["pass"] = all(v for k, v in res.items()
                          if k.endswith("_equal") and v is not None)
        ok &= res["pass"]
        results[arm["arm"]] = res
    staging = {}
    for p in sorted(out.glob("*.json")):
        if p.stem in ("ram", *results):
            arm = json.loads(p.read_text())
            if "staging" not in arm and arm.get("pinning") is not None:  # older arm records
                arm["staging"] = _staging(arm["pinning"], None if not arm.get("sss_stats") else
                                          type("S", (), {"stats": arm["sss_stats"]})())
            staging[p.stem] = arm.get("staging")
    receipt = {
        "schema": "poros.sss.gate_moe.v1",
        "staging": staging,
        "gate": "G-SSS-MoE: RAM-store vs SSD-store training, torch.equal / byte-equal",
        "pass": ok and bool(results),
        "results": results,
        "arms": {p.stem: {k: v for k, v in json.loads(p.read_text()).items()
                          if k not in ("logits_blake2b", "grads_blake2b", "losses_hex")}
                 for p in sorted(out.glob("*.json")) if p.stem in ("ram", *results)},
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=1, default=str))
    _stamp(f"G-SSS-MoE {'PASS' if receipt['pass'] else 'FAIL'}: "
           + json.dumps({k: v["pass"] for k, v in results.items()}))
    if not receipt["pass"]:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truncate")
    t.add_argument("--model", required=True)
    t.add_argument("--layers", type=int, required=True)
    t.add_argument("--dest", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--model", required=True)
    a.add_argument("--arm", choices=("ram", "ssd", "ssd-fixedcap"), required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--store-path")
    a.add_argument("--steps", type=int, default=200)
    a.add_argument("--stop-after", type=int, default=None,
                   help="stop after this many steps of the --steps schedule (the LR "
                   "schedule depends on --steps, so a shorter arm must keep it)")
    a.add_argument("--seq-len", type=int, default=512)
    a.add_argument("--grad-accum", type=int, default=1)
    a.add_argument("--logging-steps", type=int, default=10)
    a.add_argument("--sss-ram", default=None)
    a.add_argument("--label", default=None,
                   help="output name (default: the arm); SSD labels must start with 'ssd'")
    a.add_argument("--read-log", default=None, help="write every SSD read key here (JSON)")
    a.add_argument("--expert-batch", type=int, default=0,
                   help="direct entry: MoeConfig.expert_batch (0: no cap)")
    a.add_argument("--entry", choices=("trainer", "direct"), default="trainer",
                   help="trainer: PorosTrainer (validated families); direct: load_moe_streamed "
                   "+ prepare_moe + AdamW (mixtral)")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--receipt", required=True)
    args = ap.parse_args()
    if args.cmd == "arm" and args.arm.startswith("ssd") and not args.store_path:
        ap.error("ssd arms need --store-path")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    {"truncate": truncate, "arm": run_arm, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
