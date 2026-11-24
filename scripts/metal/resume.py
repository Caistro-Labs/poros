#!/usr/bin/env python3
"""The `resume` gate: a run killed with -9 and a run stopped, each continued,
against one never interrupted, by digest, all through the real CLI in fresh homes.

    resume.py --model Qwen/Qwen2.5-0.5B --save-steps 10 --out DIR [--data notes.jsonl]

checkpoint-K holds K + 1 steps, so each run trains K + 2, and its last step shows any Adam,
scheduler or RNG state a resume lost. Rows of text or ``synthetic`` train through the classic loop,
and a file of chats through ``poros.Run`` and the SFT recipe, which the report names. Writes
<out>/resume.json, and exits 1 if an arm differs or did not end up interrupted. Below the gate's
stated length the report says it is a probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from poros.core.checkpoints import list_checkpoints
from poros.metal.verify import short_of
from poros.runs import _progress

POROS = [sys.executable, "-c", "from poros.cli.main import cli; cli()"]
# What a checkpoint holds of the run, compared byte for byte: its weights and state, and its
# held-out scores (eval.json, eval_start.json: the same rows and weights give the same bytes).
# Not optimizer.json, whose implementation field may hold measured seconds.
KEPT = {"adapter_model.safetensors", "optimizer_state.pt", "lr_scheduler_state.pt", "rng_state.pt",
        "eval.json", "eval_start.json"}
STOPPED = 7  # E_JOB_CANCELLED: a stop request honoured
POLL_S = 0.05


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while piece := f.read(1 << 24):
            h.update(piece)
    return h.hexdigest()


class Arm:
    """One run in its own home, so the newest run there is this one."""

    def __init__(self, a, name: str):
        self.a, self.name = a, name
        self.home = Path(a.out) / name
        self.home.mkdir(parents=True)
        self.env = {**os.environ, "POROS_HOME": str(self.home)}
        self.log = open(self.home / "poros.log", "ab")

    def _argv(self, *words: str) -> list[str]:
        return [*POROS, *words, "--json"]

    def start(self) -> subprocess.Popen:
        return subprocess.Popen(
            self._argv("train", self.a.model, "--data", self.a.data,
                       "--steps", str(self.a.save_steps + 2), "--yes",
                       f"save_steps={self.a.save_steps}"),
            env=self.env, stdout=self.log, stderr=self.log, start_new_session=True,
        )

    def poros(self, *words: str) -> int:
        return subprocess.run(self._argv(*words), env=self.env, stdout=self.log, stderr=self.log,
                              check=False).returncode

    @property
    def folder(self) -> Path | None:
        return next(iter(sorted((self.home / "runs").glob("*"))), None)

    def wait_until(self, proc: subprocess.Popen, ready) -> bool:
        """Whether ``ready`` came true while the run was still going."""
        while proc.poll() is None:
            if self.folder and ready(self.folder):
                return True
            time.sleep(POLL_S)
        return False

    def digests(self) -> dict:
        """What a finished run keeps, each file by digest, and the loss at each step."""
        folder = self.folder
        files = {
            str(path.relative_to(folder)): _sha256(path)
            for directory in (folder, *sorted(folder.glob("checkpoint-*")))
            for path in directory.iterdir()
            if path.name in KEPT
        }
        losses, recipe = {}, None
        for line in (folder / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event.get("event_type") == "step":
                losses[str(event["step"])] = event["data"]["loss"]
            elif event.get("event_type") == "setup":
                recipe = event["data"].get("recipe", recipe)
        return {"files": files, "losses": losses, "recipe": recipe}


def _uninterrupted(a) -> dict:
    arm = Arm(a, "uninterrupted")
    code = arm.start().wait()
    return {"exit": code, "interrupted": False, "continued": None, **arm.digests()}


def _killed(a) -> dict:
    arm = Arm(a, "kill-9")
    proc = arm.start()
    interrupted = arm.wait_until(proc, lambda folder: list_checkpoints(folder))
    if interrupted:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    return {"exit": proc.returncode, "interrupted": interrupted,
            "continued": arm.poros("runs", "continue", "last"), **arm.digests()}


def _stopped(a) -> dict:
    arm = Arm(a, "stop")
    proc = arm.start()
    halfway = (a.save_steps + 2) // 2
    interrupted = arm.wait_until(proc, lambda folder: (_progress(str(folder))[0] or 0) >= halfway)
    if interrupted:
        arm.poros("runs", "stop", "last")
    proc.wait()
    return {"exit": proc.returncode, "interrupted": interrupted and proc.returncode == STOPPED,
            "continued": arm.poros("runs", "continue", "last"), **arm.digests()}


def compare(base: dict, arm: dict) -> list[str]:
    """Why this arm is not the uninterrupted run: nothing is compared by anything but bytes."""
    out = [] if arm["interrupted"] else ["never interrupted"]
    if arm["continued"] != 0:
        out.append(f"runs continue exited {arm['continued']}")
    if "adapter_model.safetensors" not in arm["files"]:
        out.append("no final adapter")
    shared = base["files"].keys() & arm["files"].keys()
    if not any("checkpoint-" in name for name in shared):
        out.append("no checkpoint in common to compare")
    out += [f"{name} differs" for name in sorted(shared)
            if base["files"][name] != arm["files"][name]]
    out += [f"loss at step {s} differs" for s in base["losses"]
            if base["losses"][s] != arm["losses"].get(s)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = _uninterrupted(a)
    arms = {"kill-9": _killed(a), "stop": _stopped(a)}
    report = {
        "model": a.model,
        "save_steps": a.save_steps,
        "steps": a.save_steps + 2,
        "data": a.data,
        "recipe": base["recipe"],
        "probe": short_of({"gate": "resume", "model": a.model, "steps": a.save_steps + 2}),
        "failures": {
            "uninterrupted": [] if base["exit"] == 0 else [f"exited {base['exit']}"],
            **{name: compare(base, arm) for name, arm in arms.items()},
        },
        "uninterrupted": base,
        **arms,
    }
    (Path(a.out) / "resume.json").write_text(json.dumps(report, indent=1) + "\n")
    shown = ("model", "steps", "recipe", "probe", "failures")
    print(json.dumps({k: report[k] for k in shown}, indent=1))
    return 1 if any(report["failures"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
