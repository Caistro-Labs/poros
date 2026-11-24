"""`runs <id> --live`: a run's live state for agents, read from its events.jsonl."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from poros import runs
from poros.cli.commands import jobs as cli_commands_jobs
from poros.cli.verbs import runs_live


def _run(home, state="running", steps=4, t0=None):
    folder = home / "runs" / "1027-1400-aaaa"
    folder.mkdir(parents=True)
    (folder / "action.json").write_text(json.dumps(
        {"schema": "poros.action.train.v1", "model": "Qwen/Qwen2.5-7B-Instruct", "steps": 10}))
    (folder / "status.json").write_text(json.dumps({
        "schema": "poros.run.status.v1", "id": folder.name, "state": state, "pid": os.getpid(),
        "created_at": "2026-10-27T18:00:00+00:00", "started_at": None, "finished_at": None,
        "output_dir": str(folder), "error": None}))
    t0 = t0 or time.time() - 60
    _steps(folder, range(steps), t0)
    return folder


def _steps(folder, steps, t0):
    with open(folder / "events.jsonl", "a") as f:
        for i in steps:
            f.write(json.dumps({"event_type": "step", "step": i, "timestamp": t0 + i,
                                "data": {"loss": 3.0 - i / 10, "s_per_step": 1.0,
                                         "tokens": 1000 * (i + 1),
                                         "vram_allocated_mb": 6000.0}}) + "\n")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path))
    return tmp_path


def _one():
    return next(r for r in runs.all_runs() if r.id == "1027-1400-aaaa")


def test_a_finished_run_reads_as_it_ended(home):
    _run(home, state="cancelled", t0=1000.0)
    out = runs_live.snapshot(_one())
    assert out.schema_ == "poros.run_live.v1" and out.state == "cancelled"
    assert (out.step, out.loss_first, out.tokens_per_s) == (4, 3.0, 1000.0)
    assert out.elapsed_s == 3.0 and out.eta_s is None  # first step to last; no eta once over
    assert out.card[0].endswith("4/10  stopped") and "eta" not in out.card[0]
    assert out.frame is None


def test_the_card_is_two_narrow_rows_of_plain_text(home):
    _run(home)
    out = runs_live.snapshot(_one(), card_gib=24.0)
    # the model whole: an Instruct run's card never names the base
    assert out.card == ["▛▜ Qwen2.5-7B-Instruct  ━━━━━━──────────  4/10  eta 6.0s",
                        "   loss 3.00 █▆▃▁ 2.70 · peak 5.6/24.0 GiB"]


def test_the_frame_is_the_run_card_with_its_line_unbroken(home):
    _run(home, steps=10)
    out = runs_live.snapshot(_one(), frame=True, card_gib=24.0)
    assert out.frame[:2] == ["▛▜ Step 10 of 10 · 100%", "   Qwen2.5-7B-Instruct"]
    axis = next(i for i, r in enumerate(out.frame) if r.split() == ["1", "5", "10"])
    curve = [r[8:] for r in out.frame[3:axis]]
    assert set("".join(curve)) <= set(" ─╮╰╯╭│○")  # a line in plain text: no fill to fade
    assert all(any(col < len(r) and r[col] != " " for r in curve) for col in range(40))
    # one row a step: like for like, no arrow; the held-out rows give the verdict. Ten steps: a
    # tenth is one step
    assert [r.strip() for r in out.frame[axis + 2:axis + 4]] == [
        "train loss 3.0000 to 2.1000", "its first step and its last"]
    assert max(len(r) for r in out.frame + out.card) <= 56  # a phone's chat column


def test_wait_returns_once_the_run_moves_on(home):
    folder = _run(home)
    later = threading.Timer(0.3, _steps, (folder, range(4, 7), time.time()))
    later.start()
    out = runs_live.snapshot(_one(), wait=3)
    later.join()
    assert out.step == 7 and out.state == "running"


def test_wait_returns_when_the_run_ends(home):
    folder = _run(home)

    def end():
        _steps(folder, [4], time.time())
        status = json.loads((folder / "status.json").read_text())
        (folder / "status.json").write_text(json.dumps({**status, "state": "succeeded"}))

    threading.Timer(0.3, end).start()
    out = runs_live.snapshot(_one(), wait=100)
    assert out.state == "succeeded" and out.step == 5


def test_detach_names_the_watch_command_on_stderr(monkeypatch, capsys):
    """A shell agent's next step after --detach: the command that gives it the card."""
    from types import SimpleNamespace

    from poros.cli import contract, train_cmd
    from poros.cli.verbs import train_run

    job = SimpleNamespace(job_id="20261027-2100-ab12")
    monkeypatch.setattr(cli_commands_jobs, "_jobs", lambda: SimpleNamespace(
        submit=lambda *a, **k: job, gpu_uuid=lambda: None))
    monkeypatch.setattr(train_cmd, "run", lambda cfg, *, submit, **k: submit())
    monkeypatch.setattr(train_run, "config_dict", lambda a, out: {})
    monkeypatch.setattr(contract, "json_mode", lambda: True)
    a = SimpleNamespace(model="m", force=False, disk=None, name="n", unvalidated=None,
                        command=lambda: "poros train m")
    assert train_run._submit(a, cfg=None) is job
    out = capsys.readouterr()
    assert out.out == ""
    assert "poros runs 20261027-2100-ab12 --live --wait 30" in out.err


def test_a_recipes_tokens_are_each_steps_own(home):
    """The classic loop counts tokens so far; a recipe's event says the step's."""
    folder = _run(home, steps=0)
    rows = [{"event_type": "setup", "step": None, "data": {"recipe": "sft"}},
            *({"event_type": "step", "step": i, "timestamp": 1000.0 + i,
               "data": {"loss": 2.0, "s_per_step": 2.0, "tokens": tokens}}
              for i, tokens in enumerate([600.0, 400.0]))]
    (folder / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # the run's mean: 1,000 tokens in 4 s, never a step's own (200) and never -100
    assert runs_live.snapshot(_one()).tokens_per_s == 250.0


def test_the_held_out_loss_is_in_the_live_state(home):
    folder = _run(home, state="succeeded", steps=2)
    write = lambda loss: (folder / "events.jsonl").open("a").write(  # noqa: E731
        json.dumps({"event_type": "eval", "data": {"eval_loss": loss}}) + "\n")
    assert runs_live.snapshot(_one()).eval_loss_start is None
    write(2.5)
    begun = runs_live.snapshot(_one())
    assert (begun.eval_loss_start, begun.eval_loss) == (2.5, None)
    write(2.25)
    ended = runs_live.snapshot(_one())
    assert (ended.eval_loss_start, ended.eval_loss) == (2.5, 2.25)


def test_a_failed_run_says_why_in_the_live_state(home):
    folder = _run(home, state="failed")
    status = json.loads((folder / "status.json").read_text())
    status["error"] = {"code": "E_OOM", "message": "out of memory"}
    (folder / "status.json").write_text(json.dumps(status))
    assert runs_live.snapshot(_one()).error["code"] == "E_OOM"


def test_watching_a_run_never_imports_torch(home):
    """Each poll of `runs` would hold a torch and a process's worth of the card under the run."""
    _run(home)
    env = {**os.environ, "POROS_HOME": str(home)}
    env.pop("POROS_DEV_SIMULATE", None)
    done = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "poros.cli.main", "runs", "1027-1400-aaaa",
         "--live", "--json"], env=env, capture_output=True, text=True, timeout=120)
    assert json.loads(done.stdout)["schema"] == "poros.run_live.v1"
    assert not any(line.split("|")[-1].strip() == "torch" for line in done.stderr.splitlines())


def _ended(home, losses, gap_after=None):
    """A finished run of these losses, one second a step, and an hour's pause after step
    ``gap_after``: a stop, a wait in the queue, and a continue."""
    folder = _run(home, state="succeeded", steps=0)
    (folder / "action.json").write_text(json.dumps(
        {"schema": "poros.action.train.v1", "model": "Qwen/Qwen2.5-7B-Instruct",
         "steps": len(losses)}))
    with open(folder / "events.jsonl", "w") as f:
        for i, loss in enumerate(losses):
            at = 1000.0 + i + (3600.0 if gap_after is not None and i > gap_after else 0.0)
            f.write(json.dumps({"event_type": "step", "step": i, "timestamp": at,
                                "data": {"loss": loss, "s_per_step": 1.0}}) + "\n")
    return folder


def test_the_frames_title_gives_the_training_time_not_the_pause(home):
    _ended(home, [3.0 - i / 10 for i in range(10)], gap_after=4)
    out = runs_live.snapshot(_one(), frame=True)
    assert out.frame[0] == "▛▜ Trained · 10 steps · 10s"  # ten steps of a second, the hour out
    assert out.elapsed_s > 3600  # the wall time stays in its own field


def test_a_spike_on_the_frames_axis_says_what_it_is(home):
    losses = [3.0 - i / 50 for i in range(40)]
    losses[20] += 1.0
    _ended(home, losses)
    out = runs_live.snapshot(_one(), frame=True, card_gib=24.0)
    axis = next(r for r in out.frame if r.strip().startswith("1 "))
    assert "▲ 21" in axis
    assert "   ▲ a loss spike at that step" in out.frame
    assert max(len(r) for r in out.frame) <= 56


def test_a_frame_with_no_spike_says_nothing_of_one(home):
    _ended(home, [3.0 - i / 50 for i in range(40)])
    assert not any("▲" in r for r in runs_live.snapshot(_one(), frame=True).frame)


def test_a_wait_returns_inside_an_agent_shells_two_minutes():
    """Common agent shells stop a call at 120 s: a wait, and a stop, return well before it."""
    assert runs_live.WAIT_MAX_S <= 100


def test_learned_says_whether_answers_end_and_how_held_out_pairs_choose():
    from poros.cli.verbs import runs_report

    chats = {"eval_loss_start": 2.0, "eval_loss": 1.0, "eval_rows": 40, "eval_ends": 30,
             "eval_answer_rows": 40, "eval_prefers_start": None, "eval_prefers": None}
    assert runs_report.learned(chats) == (
        "held-out loss 2.000 to 1.000 on 40 rows, 50% lower; ends its answers 30 of 40")
    pairs = {"eval_loss_start": 0.693, "eval_loss": 0.5, "eval_rows": 4, "eval_ends": None,
             "eval_answer_rows": None, "eval_prefers_start": 1, "eval_prefers": 3}
    assert runs_report.learned(pairs, (5, 0.8), True, (1, 80)) == (
        "held-out DPO loss 0.693 to 0.500 on 4 pairs, 28% lower; few pairs, so this is rough; "
        "chosen over rejected on 3 of 4 held-out pairs, from 1; 79 of 80 pairs could count: 1 is "
        "the same on both sides as it trains them")


def test_the_frame_says_whether_a_finished_chat_runs_answers_end():
    from poros.cli.ui import agent_text
    from poros.cli.ui.screens import TrainState
    from poros.cli.verbs.runs_follow import _apply

    st = TrainState(model="Qwen/Qwen2.5-0.5B-Instruct", output_dir="x", run="r", total=20)
    for i in range(20):
        _apply(st, {"event_type": "step", "step": i, "data": {"loss": 3.0 - i / 10}})
    for step, loss, ends in ((0, 2.9, 0), (19, 1.2, 8)):
        _apply(st, {"event_type": "eval", "step": step,
                    "data": {"eval_loss": loss, "ends": ends, "answer_rows": 8}})
    head = agent_text.frame(st, "succeeded", None, None)[0]
    assert "Learned well · ends its answers 8 of 8 · 20 steps" in head
    assert "ends" not in agent_text.frame(st, "succeeded", None, None, rough=True)[0]
