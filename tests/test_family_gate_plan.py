"""What the family gate does for each row, compared with a recorded plan (``FG_PLAN=1``).

``tests/data/family_gate_plan.txt`` holds, row by row, every stage and run the gate does for the
families it covered when the plan was recorded (before the gate learned the looped-decoder
stage). A change to scripts/family_gate.sh may add stages for a new kind of model; for every
recorded row the plan must stay the same, line for line.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "family_gate.sh"
GOLDEN = Path(__file__).parent / "data" / "family_gate_plan.txt"


def plan(tmp_path: Path, **env) -> str:
    done = subprocess.run(
        ["bash", str(GATE), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "FG_PLAN": "1",
            "POROS_PYTHON": "true",
            "POROS_GPU_LOCK": str(tmp_path / "lock"),
            "FG_MODELS": str(tmp_path / "models"),
            **env,
        },
    )
    text = done.stdout.replace(str(tmp_path / "models"), "MODELS").replace(str(tmp_path), "TMP")
    text = text.replace(str(ROOT), "TREE")
    return re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", "DATE", text)


def sections(text: str) -> dict[str, str]:
    """Each row's lines, by its key (the ``--- key (...)`` header starts a row)."""
    out: dict[str, list[str]] = {}
    key = None
    for line in text.splitlines():
        m = re.match(r"^--- (\S+) \(", line)
        if m:
            key = m[1]
        if key is not None and not line.startswith(("end ", "FAILS=")):
            out.setdefault(key, []).append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def test_every_recorded_row_runs_as_recorded(tmp_path):
    want = sections(GOLDEN.read_text())
    got = sections(plan(tmp_path))
    assert want, "the recorded plan is empty"
    for key, lines in want.items():
        assert got.get(key) == lines, f"the gate's plan for {key} changed"


def test_a_looped_decoder_s_paths_are_its_refusals_and_the_sequential_run(tmp_path):
    """A row whose decoder declares visits (here: planned as one) has no block-major training
    run: efficient and fast each run to be refused, path unset runs twice (sequential, with its
    reason), resume runs on the sequential path, and the streamed-vs-resident gate still runs."""
    rows = sections(plan(tmp_path, FG_PLAN_VISITS="inkling2_loop", FG_FAMILIES="inkling2_loop"))
    lines = rows["inkling2_loop"].splitlines()
    runs = [line for line in lines if line.startswith("PLAN run ")]
    names = [line.split()[2] for line in runs]
    assert names == [
        "inkling2_loop_eff",
        "inkling2_loop_fast",
        "inkling2_loop_seq",
        "inkling2_loop_rep",
        "inkling2_loop_resA",
    ]
    assert "path=efficient" in runs[0] and "path=fast" in runs[1]
    assert not any("path=" in r for r in runs[2:])  # unset: the run decides, and says why
    assert any(
        line.startswith("PLAN expect inkling2_loop_seq ran=sequential path_reason~")
        for line in lines
    )
    assert any("inkling2_loop:refused:eff" in line for line in lines)
    assert any("inkling2_loop:refused:fast" in line for line in lines)
    assert any("moe_absolute_gate.py" in line for line in lines)  # streamed vs resident


def test_every_recorded_row_is_still_a_row(tmp_path):
    """The table only grew: every recorded row is still planned (the new one beside them)."""
    want = sections(GOLDEN.read_text())
    got = sections(plan(tmp_path))
    assert set(want) <= set(got) and "inkling2_loop" in got


def test_refused_reads_a_refusal_and_nothing_else(tmp_path):
    import json
    import sys

    helper = ROOT / "scripts" / "family_gate" / "refused.py"
    words = "reruns its middle layers"
    err = {
        "schema": "poros.error.v1",
        "code": "E_CONFIG",
        "exit_code": 2,
        "message": f"path: fast trains block-major, which cannot ... with a decoder that {words}",
    }
    log = tmp_path / "log.txt"
    log.write_text("starting\n" + json.dumps(err) + "\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def verdict(rc, text=words):
        return subprocess.run(
            [sys.executable, str(helper), str(log), str(run_dir), str(rc), text],
            capture_output=True,
        ).returncode

    assert verdict(2) == 0
    assert verdict(0) == 1  # it ran
    assert verdict(2, "another reason") == 1  # refused, but not for this
    (run_dir / "events.jsonl").write_text(json.dumps({"event_type": "step"}) + "\n")
    assert verdict(2) == 1  # it trained


OOM = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"


def _moe_gate(tmp_path, key, **env):
    """The row's moe-gate verdict line, planned (no GPU: FG_PLAN_GPU_* say how its GPU step
    ends)."""
    rows = sections(plan(tmp_path, FG_FAMILIES=key, **env))
    return [line for line in rows[key].splitlines() if f"{key}:moe-gate" in line]


def test_a_looped_row_fails_without_streamed_against_resident(tmp_path):
    """A looped decoder's paths compare sequential with sequential only, so its row needs
    moe-gate (streamed against resident) to PASS on the hardware: where the resident arm runs
    out of memory with FG_RESIDENT_EVIDENCE set, it is a FAIL, not the SKIP another MoE row
    takes; and where moe-gate is not run at all, the row FAILS too."""
    oom = dict(FG_RESIDENT_EVIDENCE="elsewhere", FG_PLAN_GPU_RC="1", FG_PLAN_GPU_SAYS=OOM)
    looped = _moe_gate(tmp_path, "inkling2_loop", FG_PLAN_VISITS="inkling2_loop", **oom)
    assert (
        len(looped) == 1
        and looped[0].startswith("FAIL")
        and "needs streamed==resident" in looped[0]
    )
    other = _moe_gate(tmp_path, "olmoe", **oom)
    assert len(other) == 1 and other[0].startswith("SKIP")  # unchanged for every other MoE row
    unrun = _moe_gate(tmp_path, "inkling2_loop", FG_PLAN_VISITS="inkling2_loop", FG_CHECKS="paths")
    assert len(unrun) == 1 and unrun[0].startswith("FAIL") and "not run" in unrun[0]
