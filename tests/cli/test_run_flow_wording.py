"""What a person reads as a run stops, saves, resumes and ends.

One convention on screen: a step number is the steps done, counted from 1 ("after 13 steps",
"step 13/100"). On disk nothing changes: a checkpoint's folder and an event's ``step`` name the
last update's index from 0, so checkpoint-12 holds 13 steps, and where a folder's name stands
beside a count the line shows both. On the simulated trainer (no GPU).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pytest

from poros import runs
from poros.cli import app
from poros.cli.context import Context
from tests.cli.test_train_runs import home, say  # noqa: F401  (the simulated machine)

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _chats(home, rows: int, name: str = "chats.jsonl") -> str:  # noqa: F811
    (home / name).write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"What is {i} + {i}?"},
        {"role": "assistant", "content": f"It is {2 * i}."}]}) + "\n" for i in range(rows)))
    return name


def _screen(argv, home, capsys) -> tuple[int, str]:  # noqa: F811
    """(exit code, everything the command wrote), in process."""
    ctx = Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False)
    old = os.getcwd()
    os.chdir(home)
    try:
        code = app.dispatch(argv, ctx)
    finally:
        os.chdir(old)
    return code, capsys.readouterr().out


def _interrupt_at(monkeypatch, step: Optional[int]) -> None:
    """Ctrl-C as the simulated trainer's update ``step`` (its index from 0) is under way; None:
    no ctrl-c any more."""
    from poros.cli.ui import simulate

    real = getattr(simulate.SimTrainer._event, "real", simulate.SimTrainer._event)

    def event(self, i, *a, **k):
        if i == step:
            raise KeyboardInterrupt
        return real(self, i, *a, **k)

    event.real = real
    monkeypatch.setattr(simulate.SimTrainer, "_event", event)


# ── ctrl-c: what was saved, read from the disk, and the command that resumes it ─────────────────


def test_ctrl_c_names_the_checkpoint_it_saved_and_the_command_that_resumes(home, capsys,  # noqa: F811
                                                                           monkeypatch):
    """It said "no checkpoint yet · set save_steps..." two seconds after "Saved checkpoint-10"."""
    _interrupt_at(monkeypatch, 6)
    code, out = _screen(["train", MODEL, "--data", _chats(home, 60), "--yes", "--steps", "20",
                         "save_steps=4"], home, capsys)
    (run,) = runs.all_runs()
    assert code == 130 and run.state == "cancelled"
    assert "Stopped after 6 steps" in out
    assert "checkpoint-4" in out and "it holds 5 steps: a resume starts there" in out
    assert "the 1 step after it trains again" in out
    assert f"poros runs continue {run.id}" in out
    assert "no checkpoint" not in out
    # the command it names does resume it, from that checkpoint
    _interrupt_at(monkeypatch, None)
    code, obj = say(["runs", "continue", run.id], home, capsys)
    assert code == 0 and obj["runs"][0]["state"] == "succeeded"
    assert obj["runs"][0]["step"] == 20


def test_ctrl_c_before_any_checkpoint_says_so_and_offers_what_works(home, capsys,  # noqa: F811
                                                                    monkeypatch):
    _interrupt_at(monkeypatch, 3)
    code, out = _screen(["train", MODEL, "--data", _chats(home, 60), "--yes", "--steps", "20"],
                        home, capsys)
    (run,) = runs.all_runs()
    assert code == 130 and run.state == "cancelled"
    assert "Stopped after 3 steps" in out
    assert "no checkpoint was saved: ctrl-c saves none" in out
    assert "runs continue" not in out  # there is nothing to continue from


def test_a_config_runs_ctrl_c_names_its_own_resume_command(home, capsys, monkeypatch):  # noqa: F811
    from typer.testing import CliRunner

    from poros.cli import main

    _interrupt_at(monkeypatch, 7)
    cfg = home / "run.yaml"
    cfg.write_text(f"model_name_or_path: {MODEL}\ndataset: {home / _chats(home, 60)}\n"
                   f"output_dir: {home / 'out'}\nmax_steps: 20\nsave_steps: 5\n")
    monkeypatch.delenv("POROS_OUTPUT")
    res = CliRunner().invoke(main.app, ["train", str(cfg)])
    capsys.readouterr()
    assert res.exit_code == 130, res.output
    assert "Stopped after 7 steps" in res.output
    assert "checkpoint-5" in res.output and "it holds 6 steps" in res.output
    assert "--resume-from latest" in res.output and "no checkpoint" not in res.output


# ── one convention for a step on screen ─────────────────────────────────────────────────────────


def test_a_line_that_names_a_folder_beside_a_count_shows_both():
    from poros.cli.ui.train_view import checkpoint_said, interrupt_lines, steps_said

    def text(lines):
        return "\n".join("".join(t for t, _ in line) for line in lines)

    # a trainer names a folder by its last update's index: it holds one more step than its name
    assert checkpoint_said("/r/checkpoint-12", 13) == "checkpoint-12 (13 steps)"
    assert checkpoint_said("/r/checkpoint-13", 13) == "checkpoint-13"  # a name that is its count
    assert (steps_said(1), steps_said(1200)) == ("1 step", "1,200 steps")
    stopped = text(interrupt_lines(13, "/r/checkpoint-12", "run.yaml", 100,
                                   "poros runs continue abcd"))
    assert "Stopped after 13 steps" in stopped and "at step" not in stopped
    assert "/r/checkpoint-12" in stopped and "poros runs continue abcd" in stopped
    whole = text(interrupt_lines(13, "/r/checkpoint-12", "run.yaml", 100, held=13))
    assert "it holds all 13 steps" in whole  # checkpoint-12 beside "13 steps": both, said
    behind = text(interrupt_lines(13, "/r/checkpoint-9", "run.yaml", 100, held=10))
    assert "it holds 10 steps: a resume starts there; the 3 steps after it train again" in behind
    one = text(interrupt_lines(11, "/r/checkpoint-9", "run.yaml", 100, held=10))
    assert "the 1 step after it trains again" in one


def test_a_stop_request_and_a_resume_count_the_steps_done(home, capsys, monkeypatch):  # noqa: F811
    from poros.cli.ui import simulate

    stop_at = {"n": 0}
    real = simulate.SimTrainer._event

    def event(self, i, *a, **k):
        if i == 4 and not stop_at["n"]:  # the fifth update: a stop request lands
            stop_at["n"] = 1
            (runs.root() / runs.all_runs()[0].id / "stop").touch()
        return real(self, i, *a, **k)

    monkeypatch.setattr(simulate.SimTrainer, "_event", event)
    code, out = _screen(["train", MODEL, "--data", _chats(home, 60), "--yes", "--steps", "12"],
                        home, capsys)
    (run,) = runs.all_runs()
    assert code == 7 and run.state == "cancelled" and run.step == 5
    assert "Stopped after 5 steps" in out and "checkpoint-4" in out
    assert "it holds all 5 steps" in out
    code, out = _screen(["runs", "continue", run.id], home, capsys)
    assert code == 0 and "Resumed after step 5/12" in out
    assert "step 6/12" in out and "Resumed at" not in out


def test_the_cancelled_error_counts_steps_and_keeps_the_index():
    from poros.core.errors import PorosCancelledError

    exc = PorosCancelledError(12, "/r/checkpoint-12")
    assert exc.step == 12  # the last update's index, as the folder is named
    assert exc.info.message == "cancelled after 13 steps; /r/checkpoint-12 holds the run so far."
    assert "after 1 step;" in PorosCancelledError(0, "/r/checkpoint-0").info.message


def test_the_python_progress_line_is_said_at_the_first_step_then_every_logging_steps():
    """The API printed `step    0/20` where the CLI printed `step 1/100`."""
    from types import SimpleNamespace

    pytest.importorskip("torch")  # the trainer's progress lives beside torch
    from poros.train.progress import _Progress

    def said(first: int, total: int, every: int, known: bool = True) -> list[int]:
        me = SimpleNamespace(config=SimpleNamespace(logging_steps=every), global_step=0,
                             _total_optimizer_steps=total if known else 0)
        out = []
        for me.global_step in range(first, total):
            if _Progress._logs_step(me, first):
                out.append(me.global_step + 1)  # the count its line shows
        return out

    assert said(0, 25, 10, known=False) == [1, 10, 20]  # a length it does not know: as before
    assert said(13, 40, 10) == [14, 20, 30, 40]  # a resumed run: its first step, then the tens
    assert said(0, 3, 1) == [1, 2, 3]
    # the last step is always said: a resumed loop of 9 at logging_steps 2 showed 7/9 and 8/9
    # and never 9/9, and a run of 25 at 10 ended on "step 20/25"
    assert said(6, 9, 2) == [7, 8, 9]
    assert said(0, 25, 10) == [1, 10, 20, 25]


# ── a resume of a run that has no step left ─────────────────────────────────────────────────────


def _config_run(home, capsys, *argv, json_mode=False, **fields):  # noqa: F811
    from typer.testing import CliRunner

    from poros.cli import main

    said = {"model_name_or_path": MODEL, "dataset": str(home / _chats(home, 60)),
            "output_dir": str(home / "out"), "max_steps": 8, "save_steps": 4, **fields}
    cfg = home / "run.yaml"
    cfg.write_text("".join(f"{k}: {json.dumps(v)}\n" for k, v in said.items()))
    res = CliRunner().invoke(main.app, ["train", str(cfg), *argv,
                                        *(["--json"] if json_mode else [])])
    capsys.readouterr()
    return res


def test_resuming_a_finished_run_says_it_is_finished_and_prints_no_loss(home, capsys,  # noqa: F811
                                                                        monkeypatch):
    """It printed "Trained 40 steps ... final loss nan", exit 0."""
    monkeypatch.delenv("POROS_OUTPUT")
    assert _config_run(home, capsys).exit_code == 0
    receipt = (home / "out" / "receipt.json").read_text()
    res = _config_run(home, capsys, "--resume-from", "latest")
    assert res.exit_code == 0, res.output  # a retry loop's last pass ends here
    assert "Already finished 8 of 8 steps" in res.output
    assert "--steps 16" in res.output and "--resume-from latest" in res.output
    assert "nan" not in res.output.lower() and "Trained" not in res.output
    assert (home / "out" / "receipt.json").read_text() == receipt  # the run's own is kept
    # the command it names trains it further
    res = _config_run(home, capsys, "--steps", "16", "--resume-from", "latest")
    assert res.exit_code == 0 and "Resumed after step 8/16" in res.output


def test_a_finished_run_with_its_adapter_saved_is_said_before_anything_loads(home, capsys,  # noqa: F811
                                                                             monkeypatch):
    from poros.cli.ui import simulate

    assert _config_run(home, capsys, json_mode=True).exit_code == 0
    (home / "out" / "adapter_model.safetensors").write_bytes(b"x")  # as a GPU run leaves one

    def loaded(*a, **k):
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(simulate.SimTrainer, "from_pretrained", loaded)
    res = _config_run(home, capsys, "--resume-from", "latest", json_mode=True)
    assert res.exit_code == 0, res.output
    line = next(x for x in res.output.splitlines() if x.startswith('{"schema"'))
    assert json.loads(line)["metrics"] == {"total_steps": 0, "steps_done": 8,
                                           "already_finished": True}
    # fewer steps than it holds is finished too; more steps trains on
    res = _config_run(home, capsys, "--steps", "6", "--resume-from", "latest", json_mode=True)
    assert res.exit_code == 0 and '"already_finished":true' in res.output


# ── the wait while the data is read for the run's record ───────────────────────────────────────


def test_the_read_of_the_data_shows_how_far_it_is_past_a_second(home, capsys,  # noqa: F811
                                                               monkeypatch):
    """A resume hashes its data file to compare it, and a fresh run to record it: a large file
    was half a minute of silence. One activity line, shown past a second, with the read's
    progress; the read made here is the one the trainer finds done."""
    from poros.cli.ui import term
    from poros.preflight import command, resume

    seen = []

    class Line:
        elapsed = 0.0

        def __init__(self, label, delay):
            self.label, self.delay = label, delay
            seen.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def done(self, lines):
            pass

    monkeypatch.setattr(term.Term, "activity",
                        lambda self, label, **kw: Line(label, kw.get("delay", 0.0)))
    monkeypatch.setattr(resume, "trained_on",
                        lambda dataset, progress=None: progress(32 << 20, 64 << 20))
    assert _config_run(home, capsys, json_mode=True).exit_code == 0
    fresh = next(x for x in seen if x.label.startswith("Reading the data"))
    assert (fresh.label, fresh.delay) == ("Reading the data for the run's record  50%", 1.0)

    def compared(resume_from, cfg, progress=None):
        progress(1 << 20, 4 << 20)
        return None

    seen.clear()
    monkeypatch.setattr(command, "resume_refusal", compared)
    res = _config_run(home, capsys, "--steps", "12", "--resume-from", "latest", json_mode=True)
    assert res.exit_code == 0, res.output
    again = next(x for x in seen if x.label.startswith("Checking the data"))
    assert (again.label, again.delay) == ("Checking the data matches the checkpoint  25%", 1.0)
