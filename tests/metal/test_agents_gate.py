"""agents_gate.py and its verifier, with no weights: the rule on a receipt, and its faults, then the
gate's read section against the real `poros agents` in the simulator."""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from poros.metal import verify
from tests.conftest import needs_ml

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "metal" / "agents_gate.py"
SHA = "ab" * 32
GOOD = {
    "schema": verify.RECEIPT_SCHEMA, "gate": "agents", "model": "Qwen/Qwen2.5-0.5B", "seq": 256,
    "g": 8, "steps": 24, "first_microstep": 0, "expected_records": 192, "lora_dropout": 0.0,
    "pin": {**dict.fromkeys(verify.PIN_KEYS, "x"), "macos_build": "not a Mac",
            "ssd_sha256": "none", "poros_commit": "abc"},
    "checks": {name: {"pass": True, "why": [], "seconds": 1.0, "calls": []}
               for name in verify.AGENT_CHECKS},
    "stdout": {"calls": 9, "invalid": []},
    "offered": [{"from": "fit", "command": "poros agents fit m", "problem": ""}],
    "adapters": dict.fromkeys(verify.AGENT_ADAPTERS, SHA),
    "big": {},
    "leaks": [],
    "pass": None,
}


def _rule(tmp_path, change=None):
    receipt = copy.deepcopy(GOOD)
    if change:
        change(receipt)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    return verify.write_verdict(path, tmp_path), path


def test_a_clean_receipt_passes_and_stays_sealed(tmp_path):
    assert _rule(tmp_path) == ([], tmp_path / "receipt.json")
    stored = json.loads((tmp_path / "receipt.json").read_text())
    assert stored["pass"] is True and verify.check_stored(stored) == []


FAULTS = {
    "a call that prints more than its object": (
        lambda r: r["stdout"].update(invalid=["machine: 'Loading checkpoint shards'"]),
        "not one object"),
    "no call read at all": (lambda r: r.update(stdout={}), "stdout"),
    "a stale command": (
        lambda r: r["offered"].append({"command": "poros agents fit", "problem": "no modle"}),
        "an offered command fails"),
    "nothing offered": (lambda r: r.update(offered=[]), "no command was offered"),
    "adapter bytes that differ": (
        lambda r: r["adapters"].update(person="cd" * 32), "the adapters differ"),
    "an adapter never compared": (lambda r: r["adapters"].pop("waited"), "no adapter from waited"),
    "a check that failed": (
        lambda r: r["checks"]["life.more"].update({"pass": False, "why": ["it refused"]}),
        "it refused"),
    "a check that never ran": (lambda r: r["checks"].pop("read.data"), "read.data did not run"),
    "a token in a reply": (lambda r: r["leaks"].append("run list: a token"), "a reply leaked"),
    "a run too short to stop": (lambda r: r.update(steps=4, expected_records=32), "a probe"),
    "a Mac that left a family out": (
        lambda r: (r["pin"].update(macos_build="25A"), r["big"].update(
            {"x": {"family": "dense"}, "y": {"family": "gemma"}})), "['moe']"),
}


@pytest.mark.parametrize("fault", FAULTS)
def test_a_fault_fails_the_receipt(tmp_path, fault):
    change, said = FAULTS[fault]
    failures, path = _rule(tmp_path, change)
    assert any(said in f for f in failures), failures
    assert json.loads(path.read_text())["pass"] is False


def test_a_mac_with_every_family_passes(tmp_path):
    failures, _ = _rule(tmp_path, lambda r: (r["pin"].update(macos_build="25A"), r["big"].update(
        {"a": {"family": "dense"}, "b": {"family": "gemma"}, "c": {"family": "moe"}})))
    assert failures == []


# ── the script ──────────────────────────────────────────────────────────


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(GATE.parent))
    return __import__("agents_gate")


def test_every_reply_is_judged_for_what_it_offers_and_leaks(gate, tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    g = gate.Gate(NS(out=str(tmp_path), env=[], store=None, poros="poros"))

    def judged(text):
        g.judge(gate.Reply(("t",), 0, text, 0.1, 1.0), "t")

    judged('{"schema": "poros.x.v1", "next": ["poros agents fit m"], "hint": "words",'
           ' "questions": [{"choices": [{"command": "poros agents run list"}]}]}')
    assert sorted(g.offered) == ["poros agents fit m", "poros agents run list"]
    assert g.leaks == [] and g.invalid == []
    judged('{"schema": "poros.x.v1", "x": "\\u001b[31mred"}')
    judged('{"schema": "poros.x.v1", "x": "hf_' + "a" * 30 + '"}')
    assert [leak.split(": ")[1] for leak in g.leaks] == ["ANSI codes", "a token"]
    judged("Loading weights")
    judged('{"schema": "poros.x.v1"}\n{"schema": "poros.x.v1"}')
    assert len(g.invalid) == 2
    judged(json.dumps({"schema": "poros.x.v1", "log": f"{g.home}/l", "output": f"{g.home}/o"}))
    assert g.home_paths == {"poros.x.v1 log"}
    # Which disk: the internal store (a home path) is one of the answers, and may be the default,
    # where the internal disk has room; a home path in any other question is still one too many
    asked = {"id": "store_path", "prompt": "Which disk?", "default": f"{g.home}/store",
             "choices": [{"value": f"{g.home}/store", "label": "internal"},
                         {"value": "/Volumes/X/poros", "label": "X"}]}
    judged(json.dumps({"schema": "poros.needs_input.v1", "questions": [asked]}))
    assert g.home_paths == {"poros.x.v1 log"}
    other = {"id": "model", "default": f"{g.home}/m", "choices": [{"value": f"{g.home}/m"}]}
    judged(json.dumps({"schema": "poros.needs_input.v1", "questions": [other]}))
    assert g.home_paths == {"poros.x.v1 log", "poros.needs_input.v1 default",
                            "poros.needs_input.v1 value"}


def _simulated(tmp_path, poros, *more):
    """The gate's read section, against a real `poros agents` in the simulator."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, str(GATE), "--model", "Qwen/Qwen2.5-0.5B",
         "--data", str(tmp_path / "c.jsonl"), "--out", str(tmp_path / "out"), "--poros", poros,
         "--only", "read,offered", "--commit", "abc",
         "--env", "POROS_DEV_SIMULATE=1", f"PYTHONPATH={ROOT}", *more],
        env=env, capture_output=True, text=True, timeout=300)


@pytest.fixture
def chats(tmp_path):
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    (tmp_path / "c.jsonl").write_text((json.dumps(row) + "\n") * 40)


@needs_ml
def test_the_read_and_offered_sections_hold_against_the_real_cli(gate, tmp_path, chats):
    # -P, or `python -m` would run the module the gate plants in its folder, as it would anywhere
    done = _simulated(tmp_path, f"{sys.executable} -P -m poros.cli.main")
    assert done.returncode == 1, done.stdout + done.stderr  # a development run cannot pass
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    for name in ("read.version", "read.models_fit", "read.data", "read.errors",
                 "security.outside_words", "offered.runs", "words.paths", "security.planted"):
        assert receipt["checks"][name]["pass"], (name, receipt["checks"][name]["why"])
    assert receipt["stdout"]["invalid"] == [] and receipt["leaks"] == []
    assert receipt["offered"] and receipt["pass"] is False


def test_a_cli_that_prints_to_stdout_is_caught(gate, tmp_path, chats):
    noisy = tmp_path / "noisy-poros"
    noisy.write_text('#!/bin/sh\necho "Loading weights"\n'
                     f'exec {sys.executable} -P -m poros.cli.main "$@"\n')
    noisy.chmod(noisy.stat().st_mode | stat.S_IEXEC)
    done = _simulated(tmp_path, str(noisy))
    assert done.returncode == 1
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert any("Loading weights" in line for line in receipt["stdout"]["invalid"])
    assert not receipt["checks"]["read.version"]["pass"]
