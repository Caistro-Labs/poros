"""`poros prove` on a Mac: the same training from RAM and from SSD, byte for byte.

Both arms are the product's block loop from one seed, each in its own process, and
``poros.metal.verify`` makes the comparison: losses, logits, gradients, adapter and optimizer. A
model too large for RAM runs from SSD twice. The parent never imports torch, so it holds no memory
the arms could use.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from poros import metal, runs
from poros.cli.verbs import mac
from poros.core.errors import EXIT_CODES, PorosError
from poros.core.refusal import Refusal
from poros.metal import store, table
from poros.metal.verify import RECEIPT_SCHEMA, arm_from_transcript, digest_chain, write_verdict

CHILD = [sys.executable, "-P", "-m", "poros.metal.prove"]
_LABELS = {"ram-loop": "RAM", "ssd": "SSD", "ssd.2": "SSD again"}
_PINNED = ("chip", "gpu_cores", "macos_build", "shader_sha", "torch", "transformers")


def _tree_hash() -> str:
    root, h = Path(__file__).parents[1], hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _disk_identity(store_path: str) -> str:
    """A digest of the store's disk, or ``none`` off a Mac: its identity never reaches a file."""
    if sys.platform != "darwin":
        return "none"
    path = Path(store_path).resolve()
    while not os.path.ismount(path):
        path = path.parent
    info = metal.run_tool("/usr/sbin/diskutil", "info", str(path))
    # Machine identity (an SSD's serial, link, filesystem) is hashed before it reaches any file.
    parts = re.findall(
        r"(?:Device / Media Name|Protocol|File System Personality|Volume UUID):\s+(.+)", info
    )
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def pin(store_path: str, commit: str, wheel: str) -> dict:
    """What a comparison is pinned to. Arms with different pins refuse to compare."""
    import peft
    import torch
    import transformers

    from poros.metal import shaders

    chip, cores, _ = metal.device_class().rsplit("-", 2)
    return {
        "chip": chip,
        "gpu_cores": cores.removesuffix("gpu"),
        "macos_build": metal.macos_build() or "not a Mac",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hash": torch.version.git_version,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "shader_sha256": shaders.NF4_DECODE_SHA256,
        "route": "linear",
        "ssd_sha256": _disk_identity(store_path),
        "poros_commit": commit,
        "engine_tree_hash": _tree_hash(),
        "wheel_hash": wheel,
    }


def _record(trainer, writer) -> None:
    """Each step's loss, logits and gradients, digested before the optimizer sees them."""
    from poros.metal.capture import capture

    params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    forward, step = trainer._forward_blockwise_adapter_model, trainer._adapter_optimizer_step
    seen, count = {}, itertools.count()

    def on_forward(batch):
        loss = forward(batch)
        seen["loss"] = capture(loss)
        return loss

    def on_step():
        writer.microstep(next(count), grads={n: p.grad for n, p in params.items()}, **seen)
        step()

    trainer._get_lm_head().register_forward_hook(
        lambda _module, _args, output: seen.update(logits=capture(output))
    )
    trainer._forward_blockwise_adapter_model, trainer._adapter_optimizer_step = on_forward, on_step


def _run_arm(a) -> dict:
    """One arm, in this process: its transcript on disk, and what the parent shows of it."""
    from poros.cli.ui.components import version
    from poros.core.config import PorosConfig
    from poros.metal import machine
    from poros.metal.capture import TranscriptWriter
    from poros.train.trainer import PorosTrainer

    out = Path(a.out)
    config = PorosConfig(
        model_name_or_path=a.model,
        store="ssd" if a.arm.startswith("ssd") else "ram",
        store_path=a.store_path,
        max_seq_length=a.seq,
        gradient_accumulation_steps=1,
        max_steps=a.steps,
        save_steps=0,
        eval_steps=0,
        output_dir=str(out / a.arm),
        progress=False,
    )
    trainer = PorosTrainer.from_pretrained(a.model, config=config)
    params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    writer = TranscriptWriter(
        out / f"{a.arm}.jsonl", arm=a.arm, gate="prove", model=a.model, seq=a.seq, g=1,
        steps=a.steps, lora_dropout=config.lora_dropout, layers=0, expected_records=a.steps,
        pin=pin(a.store_path, os.environ.get("POROS_COMMIT") or version(), "none"),
    )
    _record(trainer, writer)
    metrics = trainer.train("synthetic", steps=a.steps, seq_len=a.seq)
    writer.final(a.steps, adapter=params, adam=trainer.optimizer.state_dict(),
                 scheduler=trainer._lr_scheduler.state_dict())
    writer.close()
    footprint = machine.memory_sample(trainer.device).footprint
    return {
        "lora_dropout": config.lora_dropout,
        "layers": trainer.block_manager.num_layers,
        "witness": {"checked": trainer._witness.checked, "mismatches": trainer._witness.mismatches},
        "peak_gib": footprint / 2**30 if footprint else None,
        "s_per_step": metrics["avg_step_time_ms"] / 1000,
        "final_loss": metrics["final_loss"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--model", "--arm", "--store-path", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    a = ap.parse_args(argv)
    code = 0
    try:
        report = _run_arm(a)
    except PorosError as exc:
        code = EXIT_CODES[exc.code]
        report = {"code": exc.code, "message": exc.info.message, "fix": exc.info.remediation}
    (Path(a.out) / f"{a.arm}.json").write_text(json.dumps(report) + "\n")
    return code


def store_for(model_id: str) -> str:
    """The store this model is in or goes to. `prove` asks no question for it: with several disks
    to choose from it takes the one with the most room, and `train` finds the model there."""
    return mac.mac_store(model_id, None) or str(store.qualifying(model_id)[0].store)


def _run_child(a, name: str, store_path: str, work: Path) -> dict:
    """One arm in its own process, shown as an activity. A refusal in it is a refusal here."""
    from poros.cli.ui import components as c
    from poros.cli.ui import term

    with term.get().activity(f"Training from {_LABELS[name]}") as act:
        done = subprocess.run(
            [*CHILD, "--model", a.model, "--arm", name, "--store-path", store_path,
             "--steps", str(a.steps), "--seq", str(a.seq_len), "--out", str(work)],
            check=False, capture_output=True, text=True,
        )
        act.done([c.event("Trained", _LABELS[name], c.duration(act.elapsed))])
    path = work / f"{name}.json"
    if done.returncode and not path.is_file():
        tail = done.stderr.strip().splitlines()[-1:] or [f"exit {done.returncode}"]
        raise Refusal("E_INTERNAL", "A run did not finish", tail[0])
    report = json.loads(path.read_text())
    if done.returncode:
        raise Refusal(report["code"], "A run did not finish", report["message"],
                      fix=report["fix"])
    return report


def run(a, ctx: Any):
    from poros.cli.sheets.listings import largest_certified
    from poros.cli.ui import components as c
    from poros.cli.verbs.prove import Arm, MacProveOutput

    stamp = datetime.now().strftime("%m%d-%H%M%S")
    work = runs.home() / "prove" / stamp
    work.mkdir(parents=True)
    here = metal.device_class()
    row = table.pick(a.model, here)
    method = row.method or ("ssd-twice" if row.tier == "ssd" else "ram-vs-ssd")
    names = ("ssd", "ssd.2") if method == "ssd-twice" else ("ram-loop", "ssd")
    store_path = store_for(a.model)
    reports = [_run_child(a, name, store_path, work) for name in names]

    entries = [arm_from_transcript(work / f"{name}.jsonl") for name in names]
    witness = {
        "checked": sum(r["witness"]["checked"] for r in reports),
        "expected": a.steps * sum(r["layers"] for r in reports),
        "mismatches": sum(r["witness"]["mismatches"] for r in reports),
    }
    receipt = work / "receipt.json"
    pinned = entries[0]["pin"]
    receipt.write_text(json.dumps({
        "schema": RECEIPT_SCHEMA, "gate": "prove", "model": a.model, "seq": a.seq_len, "g": 1,
        "steps": a.steps, "first_microstep": 0, "expected_records": a.steps,
        "lora_dropout": reports[0]["lora_dropout"], "pin": pinned, "arms": entries,
        "witness": witness, "pass": None,
    }))
    differs = write_verdict(receipt, work)
    env = {
        "poros": c.version(),
        "gpu": table.name_of(here),
        "shader_sha": pinned["shader_sha256"],
        **{k: pinned[k] for k in ("python", "torch", "transformers", "chip", "gpu_cores",
                                  "macos_build")},
    }
    digest = None if differs else digest_chain(sorted(entries[0]["digests"].values()))
    against, reproduced, pinned_differs = None, None, {}
    if a.receipt:
        before = json.loads(Path(a.receipt).read_text())
        against, reproduced = a.receipt, digest is not None and digest == before.get("digest")
        pinned_differs = {
            k: [before["env"].get(k), env[k]] for k in _PINNED if before["env"].get(k) != env[k]
        }

    def arm(report: dict) -> Arm:
        return Arm(block_size=1, peak_vram_gib=report["peak_gib"],
                   s_per_step=round(report["s_per_step"], 4), final_loss=report["final_loss"])

    path = runs.home() / "receipts" / f"prove-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = MacProveOutput(
        model=a.model, steps=a.steps, seq_len=a.seq_len, simulated=False,
        resident=arm(reports[0]), streamed=arm(reports[1]), identical=not differs,
        compared=["losses", "logits", "gradients", "adapter", "optimizer", "scheduler"],
        differs=differs, receipt=str(path), env=env, against=against, reproduced=reproduced,
        pinned_differs=pinned_differs, largest_certified=largest_certified(ctx),
        method=method, witness=witness, machine=metal.device_class(),
    )
    runs.write_json(path, {**out.model_dump(mode="json", by_alias=True), "digest": digest})
    out.exit_code = 0 if out.identical and out.reproduced is not False else 1
    return out


def view(r: Any, w: int) -> list:
    """The two runs side by side, then the result. No word of the machinery behind it."""
    from poros.cli.ui import components as c
    from poros.cli.ui.text import S, ln
    from poros.cli.views.parts import _w, rail_text
    from poros.cli.views.run_screens import _short_path
    from poros.cli.words.models import by_id

    ww, lw = _w(w), 12
    twice = r.method == "ssd-twice"
    if r.identical:
        said = "two runs from SSD, the same" if twice else "RAM or SSD, the same"
        head = ln(S("● ", "tx"), S("Proved", "hi"), S(f"  {said}", "tx"))
    else:
        head = c.error_event("Not identical", "the two runs differ", ww)[0]
    entry = by_id(r.model)
    out = [head, c.rail(S(entry.short if entry else r.model, "hi"),
                        S(f"  ·  {r.steps} steps · length {r.seq_len} · one seed, two runs", "dm"))]
    out.append(c.rail())
    labels = ("memory, 1st", "memory, 2nd") if twice else ("memory, RAM", "memory, SSD")
    most = max(r.resident.peak_vram_gib or 0, r.streamed.peak_vram_gib or 0) or None
    for label, a in zip(labels, (r.resident, r.streamed)):
        gib = a.peak_vram_gib
        out.append(c.rail(S(label.ljust(lw), "dm"),
                          *c.chips(gib / most if gib and most else None, 16, "a"),
                          S(f"{gib:.1f} GiB".rjust(10) if gib else "—".rjust(10), "tx"),
                          S(f"  {a.s_per_step:.2f} s/step", "dm")))
    out.append(c.rail())
    if r.identical:
        shown = " · ".join(r.compared)
        out.append(c.rail(S("result".ljust(lw), "dm"), S("identical", "a"),
                          S(f"  {shown}, bit for bit", "dm")))
    else:
        out.append(c.rail(S("result".ljust(lw), "dm"), S("differs", "e"),
                          S("  the receipt says where", "tx")))
    if r.against:
        same = "identical to the receipt it re-ran" if r.reproduced else "not the receipt it re-ran"
        out.append(c.rail(S("again".ljust(lw), "dm"), S(same, "tx" if r.reproduced else "e")))
        for key, (then, now) in r.pinned_differs.items():
            out += rail_text(f"{key}: {then} then, {now} here", w)
    if r.largest_certified:  # a certified recipe's own, never a plan's
        out.append(c.rail(S("certified".ljust(lw), "dm"), S(r.largest_certified, "hi"),
                          S("  the largest model certified on this Mac; poros recipes lists all",
                            "dm")))
    out += c.rail_path("receipt", _short_path(r.receipt), ww, lw=lw)
    out.append([])
    return out + c.continue_with("recipes" if r.identical else "machine", ww)


if __name__ == "__main__":
    sys.exit(main())
