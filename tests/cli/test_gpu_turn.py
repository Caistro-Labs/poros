"""Merged export and eval share the lock jobs queue on, and a tool call never waits behind it."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import typer

import poros
from poros import jobs, runs
from poros.cli import app, contract, questions
from poros.cli import root as cli_root
from poros.cli.context import Context
from poros.cli.schemas import PlanGpu
from poros.cli.verbs import evaluate_pass, shared
from poros.core.refusal import Refusal
from poros.jobs import runner
from poros.preflight import arch as preflight_arch
from tests.conftest import hub_holding_configs, needs_ml

CARD = PlanGpu(name="RTX", total_gib=8.0, free_gib=8.0, source="nvidia-smi")


@pytest.fixture
def home(tmp_path, monkeypatch):
    # the finished run's base is downloaded: a merged export of a base not here, offline, is
    # refused for that before its turn is asked for (shared.download_confirm)
    hub_holding_configs(monkeypatch, tmp_path / "hub", "Qwen/Qwen2.5-0.5B", weights=True)
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.delenv("POROS_DEV_SIMULATE", raising=False)
    monkeypatch.setattr("poros.core.card.detect_gpu", lambda *a, **k: CARD)
    monkeypatch.setattr(jobs, "gpu_uuid", lambda: "GPU-x")
    (tmp_path / "home").mkdir()
    yield tmp_path
    contract.reset()


def _say(argv, home, capsys):
    code = app.dispatch(argv, Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False))
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    return code, json.loads(lines[-1])


def _turn_taken(home: Path):
    """The lock, held by someone else: a job's runner, say."""
    held = open(home / "home" / "gpu-GPU-x.lock", "a")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return held


def _finished() -> runs.Run:
    rid = runs.new_id()
    d = runs.create(rid, {"model": "Qwen/Qwen2.5-0.5B", "data": "synthetic", "steps": 3})
    runs.update(d, state="succeeded")
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"x")
    (d / "poros_config.json").write_text(json.dumps(
        {"model_name_or_path": "Qwen/Qwen2.5-0.5B", "dataset": "synthetic"}))
    return next(r for r in runs.all_runs() if r.id == rid)


def _training() -> str:
    time.sleep(1.1)  # newest first, by the second
    rid = runs.new_id()
    runs.update(runs.create(rid, {"model": "m", "data": "d"}), state="running", pid=os.getpid())
    return rid


def _never(*a, **k):
    raise AssertionError("loaded a model beside a run")


@pytest.mark.parametrize("argv", [["eval", "{run}"],
                                  pytest.param(["export", "{run}", "--format", "merged"],
                                               marks=needs_ml)],  # a merge's deps come first
                         ids=["eval", "export"])
def test_beside_a_run_a_call_is_refused_at_once_and_says_how_to_wait(home, capsys, monkeypatch,
                                                                     argv):
    monkeypatch.setattr(evaluate_pass, "settings", _never)
    # before its turn no torch is imported (750 MB, for hours); only whether it is installed
    monkeypatch.setattr(cli_root, "_require_torch",
                        lambda *a, import_it=True, **k: _never() if import_it else None)
    done = _finished()
    live = _training()
    held = _turn_taken(home)
    begun = time.monotonic()
    code, out = _say([a.format(run=done.id) for a in argv], home, capsys)
    assert time.monotonic() - begun < 5  # a tool call: it never hangs for hours
    assert code == 2 and out["error"]["code"] == "E_USAGE", out
    here = "this Mac" if sys.platform == "darwin" else "the GPU"  # a Mac's one pool
    assert f"run {live} is using {here}" in out["error"]["message"]
    assert out["error"]["hint"] == f"poros runs {live} --wait 30"
    held.close()


def test_an_eval_holds_the_card_while_it_scores_and_lets_go(home, capsys, monkeypatch):
    def scores(*a, **k):
        assert jobs.hold_machine() is None
        raise Refusal("E_USAGE", "Scored", "held the card")

    monkeypatch.setattr(evaluate_pass, "settings", scores)
    code, out = _say(["eval", _finished().id], home, capsys)
    assert out["error"]["message"] == "held the card", out
    assert jobs.hold_machine() is not None


@pytest.mark.cuda_host(reason="a Mac's merged export first checks the base against the revision "
                               "its run recorded (poros.metal.same_base); this run recorded none")
def test_a_merged_export_holds_the_card_until_it_is_written(home, capsys, monkeypatch):
    torch = pytest.importorskip("torch")
    r = _finished()
    torch.save({}, Path(r.output_dir) / "adapter_weights.pt")
    seen = []
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda name: ("olmoe", ""))
    monkeypatch.setattr("poros.moe.export.export_merged",
                        lambda *a, **k: seen.append(jobs.hold_machine() is None))
    code, out = _say(["export", r.id, "--format", "merged"], home, capsys)
    assert code == 0 and out["format"] == "merged", out
    assert seen == [True] and jobs.hold_machine() is not None


def test_an_adapter_copy_takes_no_turn(home, capsys):
    held = _turn_taken(home)
    code, out = _say(["export", _finished().id, "--format", "adapter"], home, capsys)
    assert code == 0 and out["format"] == "adapter", out
    held.close()


def test_in_a_terminal_an_export_waits_its_turn_and_an_eval_does_not(home, monkeypatch):
    monkeypatch.setattr(questions, "interactive", lambda: True)
    monkeypatch.setattr(contract, "json_mode", lambda: False)
    held = _turn_taken(home)
    with pytest.raises(typer.Exit):
        shared.hold_gpu("eval")
    threading.Timer(0.6, held.close).start()
    begun = time.monotonic()
    assert shared.hold_gpu("a merged export", queue=True) is not None
    assert time.monotonic() - begun >= 0.5


def test_a_job_queues_behind_them(home):
    """The runner's own wait, on the file they hold: same name, same lock."""
    job = home / "home" / "jobs" / "20261104-000000-aaaa"
    job.mkdir(parents=True)
    ours = shared.hold_gpu("eval")
    threading.Timer(1.5, (job / "stop").touch).start()
    assert runner._wait_for_gpu(job, "GPU-x", "t") is None  # cancelled, never its turn
    ours.close()
    (job / "stop").unlink()
    assert runner._wait_for_gpu(job, "GPU-x", "t") is not None


def _raises(kind):
    def raiser():
        raise kind("x")

    return raiser


@pytest.mark.parametrize("gone", [lambda m: m.setattr("poros.core.card.detect_gpu", lambda: None),
                                  lambda m: m.setenv("POROS_DEV_SIMULATE", "1"),
                                  lambda m: m.setattr(jobs, "hold_machine", _raises(OSError)),
                                  lambda m: m.setattr(jobs, "hold_machine", _raises(ImportError))],
                         ids=["no GPU", "simulated", "no home for it", "no flock"])
def test_nothing_to_hold_stops_nothing(home, monkeypatch, gone):
    gone(monkeypatch)
    assert shared.hold_gpu("eval") is None


@pytest.mark.parametrize("argv", [pytest.param(["export", "{dir}"], marks=needs_ml),
                                  pytest.param(["export", "{run}", "--format", "merged"],
                                               marks=needs_ml),
                                  ["eval", "{run}"]],
                         ids=["export a folder", "export a run", "eval"])
def test_beside_a_run_a_real_process_takes_no_torch(home, argv):
    r = _finished()
    (home / "sitecustomize.py").write_text(
        "import poros.jobs, poros.core.card\nfrom poros.cli.schemas import PlanGpu\n"
        f"poros.core.card.detect_gpu = lambda *a, **k: PlanGpu(**{CARD.model_dump()!r})\n"
        'poros.jobs.gpu_uuid = lambda: "GPU-x"\n')
    env = {**os.environ, "PYTHONPATH": f"{home}{os.pathsep}{Path(poros.__file__).parent.parent}"}
    held = _turn_taken(home)
    done = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "poros.cli.main",
         *[a.format(run=r.id, dir=r.output_dir) for a in argv], "--json"],
        env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    held.close()
    assert done.returncode == 2 and json.loads(done.stdout)["error"]["code"] == "E_USAGE"
    assert not any(line.split("|")[-1].strip() == "torch" for line in done.stderr.splitlines())
