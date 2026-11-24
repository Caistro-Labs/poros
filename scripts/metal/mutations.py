#!/usr/bin/env python3
"""The `mutations` gate: each fault with its predicted catch and predicted miss.

    mutations.py --model Qwen/Qwen2.5-0.5B --seq 256 --out DIR [--only bit-middle,bit-last]
    mutations.py --model tiny --out DIR

A gate that cannot fail proves nothing, so each fault is put in at its own site, in its own process,
and the real comparators judge it. ``tiny`` is a small random Qwen2 on whatever device this machine
has. A misprediction is a bug to investigate, not a stop: it is reported, the rest go on, and the
script exits 0 unless it could not run. Off a Mac, what only Metal or the OS's disk counters can do
is skipped or marked as simulated, and says so. Writes <out>/mutations.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import arms
import torch

from poros.core.config import PorosConfig
from poros.metal.capture import TranscriptWriter
from poros.metal.source import RamLayers
from poros.metal.verify import Transcript, disk_failures, first_difference
from poros.metal.witness import RematWitness
from poros.sss.format import SyncFileSource
from poros.train.trainer import PorosTrainer

# (what should catch it, what should miss it).
PREDICTED = {
    "bit-middle": (("streaming", "witness"), ("repeat",)),
    "bit-last": (("streaming",), ("witness, user mode",)),
    "rng-off": (("direct", "witness, after the first update"),
                ("witness, before the first update", "streaming")),
    "negative-zero": (("byte comparator",), ("torch.equal",)),
    "stopped-short": (("declared counts",), ()),
    "secret-ram": (("disk counters",), ("digests",)),
    "nan": (("nonfinite count",), ("digests",)),
    "mps-rng-removed": (("resume refuses",), ()),
}
FLIP_BYTE, FLIP_BIT = 100, 0x80  # in the first blob's packed nibbles, for any model
NEEDS_METAL = "needs Metal"


# -- the faults, each in the arm's own process ---------------------------------------------------


def _flip_once(pick) -> None:
    """One bit in one read of one layer. Not the first read: that one verifies its slab's checksum,
    and the flip is a bad read past it."""
    read_into, reads = SyncFileSource.read_into, Counter()

    def flipping(self, key, dst):
        read_into(self, key, dst)
        layers = sum(e.kind == "layer" for e in self.file.entries)
        if key[0] == pick(layers):
            reads[key] += 1
            if reads[key] == 2:
                dst[FLIP_BYTE] ^= FLIP_BIT

    SyncFileSource.read_into = flipping


def _no_witness() -> None:
    RematWitness.check = lambda self, step, loss: None


def _user_mode() -> None:
    """The product's own source: a warm hit at the turn, so both passes read the same bytes."""
    arms._read_from_disk = lambda trainer, kind: {}


def _adapter_starts_at(value: float) -> None:
    """The first adapter element, just before digesting, in the digest only."""
    final = TranscriptWriter.final

    def edited(self, microsteps, *, adapter, adam, scheduler):
        name = next(iter(adapter))
        copy = adapter[name].detach().clone()
        copy[(0,) * copy.dim()] = value
        final(self, microsteps, adapter={**adapter, name: copy}, adam=adam, scheduler=scheduler)

    TranscriptWriter.final = edited


def _one_step_short() -> None:
    train = PorosTrainer.train
    PorosTrainer.train = lambda self, data, *, steps: train(self, data, steps=steps - 1)


def _serve_from_memory() -> None:
    read_from_disk = arms._read_from_disk

    def secretly(trainer, kind):
        disk = read_from_disk(trainer, kind)
        manager = trainer.block_manager
        manager.source = RamLayers(manager.source.path, trainer.device)
        return disk

    arms._read_from_disk = secretly


def _nan_in_captured_logits() -> None:
    """In what is digested and counted, so the run goes on: its own guard would stop a real NaN."""
    on_logits = arms.Recorder._on_logits

    def poisoned(self, module, args, output):
        bad = output.detach().clone()
        bad[..., 0, 0] = float("nan")
        on_logits(self, module, args, bad)

    arms.Recorder._on_logits = poisoned


def _middle(layers: int) -> int:
    return layers // 2


def _last(layers: int) -> int:
    return layers - 1


SETUPS = {
    "flip-middle": lambda: _flip_once(_middle),
    "flip-middle-unwatched": lambda: (_flip_once(_middle), _no_witness()),
    "flip-last-user": lambda: (_flip_once(_last), _user_mode()),
    "zero": lambda: _adapter_starts_at(0.0),
    "negative-zero": lambda: _adapter_starts_at(-0.0),
    "short": _one_step_short,
    "secret-ram": _serve_from_memory,
    "nan": _nan_in_captured_logits,
}


# -- the judges ----------------------------------------------------------------------------------


class Bench:
    """Runs arms once each, and reads what the real comparators say of them."""

    def __init__(self, a, argv):
        self.a, self.argv, self.out = a, argv, Path(a.out)
        self.codes = {}

    def arm(self, name: str, setup: str = "") -> int:
        if name not in self.codes:
            self.codes[name] = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *self.argv, "--arm", name,
                 "--fault", setup, "--child"],
                check=False, capture_output=True,
            ).returncode
        return self.codes[name]

    def transcript(self, name: str) -> Transcript:
        return Transcript(self.out / f"{name}.jsonl")

    def report(self, name: str) -> dict:
        return json.loads((self.out / f"{name}.json").read_text())

    def differ(self, one: str, other: str) -> bool:
        """A transcript cut short differs from anything: the verifier refuses to compare it."""
        a, b = self.transcript(one), self.transcript(other)
        return bool(a.problems or b.problems) or first_difference(a, b) is not None


def _bit_middle(b: Bench) -> dict:
    code = b.arm("ssd.flip-middle", "flip-middle")
    b.arm("ram-loop")
    b.arm("ssd.flip-1", "flip-middle-unwatched")
    b.arm("ssd.flip-2", "flip-middle-unwatched")
    return {
        "streaming": b.differ("ram-loop", "ssd.flip-1"),
        "witness": code == 4 and b.report("ssd.flip-middle")["witness"]["mismatches"] > 0,
        "repeat": b.differ("ssd.flip-1", "ssd.flip-2"),
    }


def _bit_last(b: Bench) -> dict:
    b.arm("ram-loop")
    code = b.arm("ssd.flip-last", "flip-last-user")
    return {
        "streaming": b.differ("ram-loop", "ssd.flip-last"),
        "witness, user mode": code == 4 or b.report("ssd.flip-last")["witness"]["mismatches"] > 0,
    }


def _rng_off(b: Bench) -> dict:
    b.arm("mono")
    b.arm("ram-loop")
    b.arm("ram-loop.rng-off")
    b.arm("ssd.rng-off")
    code = b.arm("ram-loop.rng-off-witnessed")
    stopped_at = (b.report("ram-loop.rng-off-witnessed").get("witness") or {}).get("step")
    return {
        "direct": b.differ("mono", "ram-loop.rng-off"),
        "witness, after the first update": code == 4 and stopped_at > 1,
        "witness, before the first update": stopped_at == 1,
        "streaming": b.differ("ram-loop.rng-off", "ssd.rng-off"),
        "stopped at step": stopped_at,  # later than 2 under warmup: its first update has lr 0
    }


def _negative_zero(b: Bench) -> dict:
    b.arm("ram-loop.zero", "zero")
    b.arm("ram-loop.negative-zero", "negative-zero")
    return {
        "byte comparator": b.differ("ram-loop.zero", "ram-loop.negative-zero"),
        "torch.equal": not torch.equal(torch.tensor(0.0), torch.tensor(-0.0)),
    }


def _stopped_short(b: Bench) -> dict:
    b.arm("ram-loop.short", "short")
    problems = b.transcript("ram-loop.short").problems
    return {"declared counts": any("declared" in p for p in problems)}


def _secret_ram(b: Bench) -> dict:
    """Off a Mac there is no counter or page cache to read. What each arm would have shown goes in
    instead, so this says the verifier reads them, and nothing about the OS's."""
    b.arm("ram-loop")
    b.arm("ssd")
    b.arm("ssd.secret", "secret-ram")
    disks = {name: b.report(name)["disk"] for name in ("ssd", "ssd.secret")}
    if not arms.MAC:
        for disk, read in zip(disks.values(), (disks["ssd"]["predicted_read_bytes"], 0)):
            disk.update(preflight_resident_fraction=0.0, read_bytes=read)
    honest, secret = (disk_failures({"name": n, "disk": d}) for n, d in disks.items())
    return {
        "disk counters": bool(secret) and not honest,
        "digests": b.differ("ram-loop", "ssd.secret"),
        "disks": disks,
        **({} if arms.MAC else {"note": "counters simulated: there are none off a Mac"}),
    }


def _nan(b: Bench) -> dict:
    b.arm("ram-loop.nan", "nan")
    b.arm("ssd.nan", "nan")
    return {
        "nonfinite count": b.transcript("ram-loop.nan").nonfinite()[0] > 0,
        "digests": b.differ("ram-loop.nan", "ssd.nan"),
    }


def _mps_rng_removed(b: Bench) -> dict:
    if not arms.MAC:
        return {"skipped": NEEDS_METAL}
    b.arm("ram-loop.resume", "resume-without-mps-rng")
    return {"resume refuses": b.report("ram-loop.resume").get("code") == "E_CONFIG",
            "seen": b.report("ram-loop.resume")}


JUDGES = {
    "bit-middle": _bit_middle,
    "bit-last": _bit_last,
    "rng-off": _rng_off,
    "negative-zero": _negative_zero,
    "stopped-short": _stopped_short,
    "secret-ram": _secret_ram,
    "nan": _nan,
    "mps-rng-removed": _mps_rng_removed,
}


def _resume_without_mps_rng(a) -> None:
    """A real checkpoint, with its MPS generator state taken out, resumed in a new trainer."""
    out, report = Path(a.out), {}
    config = PorosConfig(
        model_name_or_path=a.model, store="ram", max_seq_length=a.seq, max_steps=2,
        save_steps=0, eval_steps=0, output_dir=str(out / a.arm), progress=False,
    )
    trainer = PorosTrainer.from_pretrained(a.model, config=config)
    vocab = trainer.model.config.get_text_config().vocab_size
    trainer.train(arms.token_stream(vocab, a.seq, 2), steps=2)
    checkpoint = Path(trainer._save_checkpoint())
    state = torch.load(checkpoint / "rng_state.pt", weights_only=True)
    del state["mps_rng"]
    torch.save(state, checkpoint / "rng_state.pt")
    try:
        trainer.load_checkpoint(str(checkpoint))
        report = {"code": None}
    except Exception as exc:  # noqa: BLE001 - which error it is, is the finding
        report = {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)[:300]}
    (out / f"{a.arm}.json").write_text(json.dumps(report))


def _verdict(fault: str, seen: dict) -> dict:
    catch, miss = PREDICTED[fault]
    if "skipped" in seen:
        return {"fault": fault, "skipped": seen["skipped"]}
    wrong = [f"{g} was to catch it and did not" for g in catch if not seen[g]]
    wrong += [f"{g} was to miss it and caught it" for g in miss if seen[g]]
    return {"fault": fault, "catch": catch, "miss": miss, "seen": seen, "misprediction": wrong}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = arms.parser()
    next(action for action in ap._actions if action.dest == "arm").required = False
    ap.set_defaults(gate="streaming")
    ap.add_argument("--fault", default="")
    ap.add_argument("--only", default=",".join(PREDICTED))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.child:
        if a.fault == "resume-without-mps-rng":
            _resume_without_mps_rng(a)
            return 0
        if a.fault:
            SETUPS[a.fault]()
        arms.run_arm(a)
        return 0
    if a.model == "tiny":
        checkpoint = out / "tiny-qwen2"
        if not checkpoint.exists():
            arms.write_checkpoint(checkpoint, arms.tiny_config())
        argv = [*argv, "--model", str(checkpoint), "--seq", "16"]
        a = ap.parse_args(argv)
    bench = Bench(a, argv)
    results = []
    for fault in a.only.split(","):
        results.append(_verdict(fault, JUDGES[fault](bench)))
        (out / "mutations.json").write_text(json.dumps(results, indent=1) + "\n")
    wrong = [m for r in results for m in r.get("misprediction", ())]
    print(json.dumps({"device": "mps" if arms.MAC else "cpu", "faults": results,
                      "mispredictions": wrong}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
