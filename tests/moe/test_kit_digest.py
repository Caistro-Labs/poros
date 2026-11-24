"""The kit's digest.py (scripts/moe_precision): ``same`` cannot run on fewer than two runs or on
a run missing its adapter or optimizer state (ERROR, exit 2, never a PASS of empty hashes), and
``stopped`` passes only where a resume marker below the last update names the stop's checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
DIGEST = ROOT / "scripts" / "moe_precision" / "digest.py"


def _run(*args):
    out = subprocess.run([sys.executable, str(DIGEST), *map(str, args)], capture_output=True,
                         text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    return out.returncode, json.loads(out.stdout)


def _a_run(folder: Path, *, adapter=True, optimizer=True, events=()):
    ck = folder / "checkpoint-4"
    ck.mkdir(parents=True)
    if adapter:
        torch.save({"a.lora_A": torch.ones(2, 2)}, ck / "adapter_weights.pt")
    if optimizer:
        torch.save({"state": {0: {"step": torch.tensor(4)}}, "param_groups": []},
                   ck / "optimizer_state.pt")
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return folder


def test_same_on_one_run_or_a_missing_file_cannot_run(tmp_path):
    a = _a_run(tmp_path / "a")
    code, said = _run("same", a)
    assert code == 2 and said["status"] == "error" and "two runs or more" in said["error"]
    code, said = _run("same", a, _a_run(tmp_path / "b"))
    assert code == 0 and said["pass"]
    for name, kw, what in (("c", {"adapter": False}, "no adapter file"),
                           ("d", {"optimizer": False}, "no optimizer_state.pt")):
        code, said = _run("same", a, _a_run(tmp_path / name, **kw))
        assert code == 2 and said["status"] == "error" and what in said["error"], said


def _step(s):
    return {"event_type": "step", "step": s, "data": {"loss": 1.0, "grad_norm": 1.0}}


def test_stopped_needs_a_resume_below_the_last_update_at_the_stop_s_checkpoint(tmp_path):
    whole = [_step(s) for s in range(1, 17)]
    code, said = _run("stopped", _a_run(tmp_path / "never", events=whole))
    assert code == 1 and not said["pass"] and "the stop never landed" in said["why"]
    resumed = ([_step(s) for s in range(1, 10)]
               + [{"event_type": "resume", "step": 9,
                   "data": {"checkpoint": str(tmp_path / "r" / "checkpoint-8")}}]
               + [_step(s) for s in range(9, 17)])
    code, said = _run("stopped", _a_run(tmp_path / "r", events=resumed))
    assert code == 0 and said["pass"], said
    at_end = whole + [{"event_type": "resume", "step": 16,
                       "data": {"checkpoint": str(tmp_path / "e" / "checkpoint-15")}}]
    assert _run("stopped", _a_run(tmp_path / "e", events=at_end))[0] == 1
    # digest counts a step a resume took again once
    code, said = _run(_a_run(tmp_path / "d", events=resumed))
    assert code == 0 and len(said["losses"]) == 16
