"""`poros agents` on a Mac: preference pairs, short-run curves, loss frames, the download
question and the guide. No weights: fixture run folders, the simulator and tiny CPU models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from poros import runs
from tests.cli.test_agents import _agent, _chats

RUN = "1107-0900-aaaa"


@pytest.fixture
def home(tmp_path, monkeypatch):
    for variable in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros"))
    (tmp_path / "me").mkdir()
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    return tmp_path


def _pairs(home: Path, rows: list[dict], name: str = "prefs.jsonl") -> Path:
    path = home / "work" / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# Preference pairs that teach nothing


def test_a_file_of_pairs_that_teach_nothing_trains_and_then_offers_the_fix(home):
    good, same = {"prompt": "q", "chosen": "a", "rejected": "b"}, {"prompt": "q", "chosen": "x",
                                                                    "rejected": "x"}
    _pairs(home, [good, same, good])
    said = _agent("data", "prefs.jsonl")
    assert said["trains_as_is"] is True and said["rows"] == 3
    assert said["issues"] == [{"row": 2, "kind": "identical", "why": "row 2: chosen and rejected "
                               "are the same: it teaches nothing"}]
    assert said["next"][0].startswith("poros agents train ") and said["next"][0].endswith(
        "--data prefs.jsonl")
    assert said["next"][1:] == ["poros agents data prefs.jsonl --fix"]


def test_a_pairs_runs_learned_says_how_many_pairs_could_count(home):
    folder = runs.create(RUN, {"model": "Qwen/Qwen2.5-0.5B", "steps": 40})
    runs.update(folder, state="succeeded")
    with open(folder / "events.jsonl", "w") as f:
        print(json.dumps({"event_type": "setup", "data": {
            "recipe": "dpo", "data": "pairs", "rows": 80, "same_pairs": 15}}), file=f)
        for step in range(40):  # the last eighth is the last 8
            print(json.dumps({"event_type": "step", "step": step,
                              "data": {"loss": 0.6, "accuracy": 0.7}}), file=f)
    assert _agent("run", "info", RUN)["learned"] == (
        "chosen over rejected on 70% of training pairs over its last 8 steps; training pairs, "
        "not held out; 65 of 80 pairs could count: 15 are the same on both sides as it trains "
        "them")
    with open(folder / "events.jsonl", "w") as f:  # none the same: nothing more is said
        print(json.dumps({"event_type": "setup", "data": {
            "recipe": "dpo", "data": "pairs", "rows": 80, "same_pairs": 0}}), file=f)
    assert _agent("run", "info", RUN)["learned"] == "no held-out measure for preference training"


# Short Mac runs keep only their end


def test_the_plan_says_which_checkpoints_a_run_keeps(home):
    """A 284-step run on a Mac, whose table row saves every 1000, kept only its end: the plan says
    so, and how to set it."""
    from types import SimpleNamespace as Bag

    from poros.cli.verbs.train_run import plan_said

    plan = Bag(s_per_step=Bag(value=None, why="not measured"), load_time=Bag(value=None))
    run = dict(shape="chats", rows=300, steps=284, seq_len=512, lr=2e-5, recipe="SFT", plan=plan,
               model="Qwen/Qwen2.5-0.5B-Instruct")
    mac = plan_said(Bag(config={}, preset={"save_steps": 1000}, **run))
    assert (mac["save_steps"], mac["keeps"]) == (1000, (
        "only at its end (and at a stop): it saves every 1,000 steps and runs 284; save_steps=N "
        "sets it"))
    off = plan_said(Bag(config={"save_steps": "0"}, preset={"save_steps": 1000}, **run))
    assert off["keeps"] == ("only at its end (and at a stop): save_steps=0 saves none between; "
                            "save_steps=N sets it")
    _chats(home / "work" / "d.jsonl", 200)
    # a CUDA card's run, simulated: on a Mac its presets are the Mac's (one row a step, no
    # accumulation), so 200 rows are 380 steps there, not 48
    if sys.platform != "darwin":
        cuda = _agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl")["plan"]
        assert (cuda["steps"], cuda["save_steps"]) == (48, 500)
        assert cuda["keeps"].startswith("only at its end (and at a stop): it saves every 500 steps")
        mine = _agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl",
                      "save_steps=10")["plan"]
        assert (mine["save_steps"], mine["keeps"]) == (10, (  # 5 saves, and 3 kept on disk
            "every 10 steps, at a stop and at its end, the newest 3 kept (save_total_limit=N); "
            "save_steps=N sets it"))
    from poros.cli.verbs.train_run import keeps

    assert keeps(10, 48, 0) == "every 10 steps, at a stop and at its end; save_steps=N sets it"
    assert keeps(50, 120, 3) == "every 50 steps, at a stop and at its end; save_steps=N sets it"


def test_the_guide_says_when_a_run_saves_on_both_platforms_and_how_to_set_it():
    from poros.cli.verbs import docs

    guide = " ".join(docs._files()[docs.GUIDE].split())
    assert "500 unless a `save_steps=N` field says" not in guide  # untrue on a Mac
    assert "500 on CUDA; on a Mac the Mac table sets it" in guide
    assert "`poros agents train <model> --data <file> save_steps=50`" in guide
    assert "`keeps` which checkpoints" in guide


# A finished run's last step and final are one moment


def test_a_finished_runs_last_step_and_final_each_name_the_other(home):
    from tests.cli.test_agents import _a_run

    folder = _a_run(run_id=RUN)  # steps 10 and 20 saved, 20 steps, finished
    rows = {row["path"]: row for row in _agent("checkpoint", "list")["checkpoints"]}
    final, last, ten = (f"poros://{RUN}/weights/{s}" for s in ("final", 20, 10))
    assert rows[final]["same_as"] == last and rows[last]["same_as"] == final
    assert rows[ten]["same_as"] is None
    assert _agent("checkpoint", "info", last)["checkpoints"][0]["same_as"] == final
    runs.update(folder, state="cancelled")  # stopped since: its adapter may be an earlier end's
    assert all(row["same_as"] is None for row in _agent("checkpoint", "list")["checkpoints"])


def test_the_guide_says_what_same_as_means():
    from poros.cli.verbs import docs

    guide = " ".join(docs._files()[docs.GUIDE].split())
    assert "the same moment, and each one's `same_as` names the other's path" in guide


# The frame's train loss, like for like


def _ended(losses: list[float], state: str = "succeeded") -> runs.Run:
    folder = runs.create(RUN, {"model": "Qwen/Qwen2.5-0.5B-Instruct", "steps": len(losses)})
    runs.update(folder, state=state)
    with open(folder / "events.jsonl", "w") as f:
        for i, loss in enumerate(losses):
            print(json.dumps({"event_type": "step", "step": i, "timestamp": 1000.0 + i,
                              "data": {"loss": loss, "s_per_step": 1.0}}), file=f)
    return runs.read(folder)


def test_the_frames_train_loss_is_the_first_tenth_against_the_last(home):
    """It read "train loss 1.4400, first step 1.0628" under "17% lower": the curve's smoothed end
    against one row's first step. Now both are means of a tenth of the steps."""
    from poros.cli.verbs import runs_live

    losses = [1.0628] + [2.0 - i / 400 for i in range(1, 284)]  # a low first row, then falling
    live = runs_live.snapshot(_ended(losses), frame=True)
    first, last = sum(losses[:28]) / 28, sum(losses[-28:]) / 28
    assert f"   train loss {first:.4f} to {last:.4f}" in live.frame
    assert "   means of its first and last tenth: 28 steps each" in live.frame
    assert not any("first step 1.0628" in row for row in live.frame)
    # the card and loss_smooth are as they were: the first step's loss, and the line's end
    assert live.card[1].startswith("   loss 1.06 ")
    assert live.card[1].rstrip().endswith(f" {live.loss_smooth:.2f}")
    assert max(len(row) for row in live.frame) <= 56


def test_a_rising_train_loss_stays_shown_as_it_is(home):
    from poros.cli.verbs import runs_live

    losses = [1.0 + i / 100 for i in range(50)]
    frame = runs_live.snapshot(_ended(losses), frame=True).frame
    assert "   train loss 1.0200 to 1.4700" in frame  # 5 steps a tenth, and it rose


def test_the_legend_says_what_plus_n_on_the_axis_means(home):
    from poros.cli.verbs import runs_live

    losses = [3.0 - i / 200 for i in range(400)]
    for at in (200, 203, 206):  # side by side: one label, and +2
        losses[at] += 1.0
    frame = runs_live.snapshot(_ended(losses), frame=True).frame
    axis = next(row for row in frame if "▲" in row and row.strip().startswith("1 "))
    assert "▲ 201 +2" in axis
    assert "   ▲ a loss spike at that step" in frame
    assert "   +2: 2 more spikes beside it" in frame
    assert max(len(row) for row in frame) <= 56


# A run at its last step, still scoring its held-out rows


def test_a_run_at_its_last_step_says_it_is_scoring_its_held_out_rows(home):
    """It read `state: running` at 60/60 with nothing saying why."""
    import os

    folder = runs.create(RUN, {"model": "Qwen/Qwen2.5-0.5B-Instruct", "steps": 60})
    runs.update(folder, state="running", pid=os.getpid())

    def wrote(*events):
        with open(folder / "events.jsonl", "a") as f:
            for event in events:
                print(json.dumps(event), file=f)

    wrote({"event_type": "setup", "data": {"recipe": "sft", "data": "chats"}},
          {"event_type": "eval", "step": 0, "data": {"eval_loss": 2.0, "rows": 10}},
          *({"event_type": "step", "step": i, "data": {"loss": 2.0 - i / 100}} for i in range(30)))
    halfway = _agent("run", "watch", RUN)
    assert halfway["phase"] == "training" and len(halfway["card"]) == 2
    wrote(*({"event_type": "step", "step": i, "data": {"loss": 2.0 - i / 100}}
            for i in range(30, 60)))
    scoring = _agent("run", "watch", RUN, "--frame")
    assert (scoring["state"], scoring["step"], scoring["phase"]) == ("running", 60, "scoring")
    assert scoring["card"][2] == "   scoring its 10 held-out rows"
    assert "   scoring its 10 held-out rows" in scoring["frame"]
    wrote({"event_type": "eval", "step": 59, "data": {"eval_loss": 1.5, "rows": 10}})
    saving = _agent("run", "watch", RUN)
    assert saving["phase"] == "saving"
    assert saving["card"][2] == "   last step done: saving what it trained"
    runs.update(folder, state="succeeded")
    assert "phase" not in _agent("run", "watch", RUN)  # over: its state says it


# The download question's units


@pytest.mark.parametrize("gib, said", [
    (0.25, "0.27 GB"),  # SmolLM2-135M: `arch` said 0.25 GiB, the question 0.2 GB
    (0.01, "under 0.1 GB"),
    (0.5, "0.54 GB"),
    (14.6, "15.7 GB"),
    (135.4, "145.4 GB"),
])
def test_the_download_question_says_decimal_gb_to_two_places_under_one(gib, said):
    from poros.cli.verbs import shared

    assert shared.download_confirm("Qwen/Qwen3-0.6B", gib, None) == said


# The guide, shorter: the rare paths in the docs it points at


def test_the_guide_points_at_the_ladder_and_a_macs_new_family_and_the_docs_hold_them(home):
    from poros.cli.verbs import docs

    files = docs._files()
    guide = files[docs.GUIDE]
    # the guide stays under 22 kB, with eval --checkpoint and --show, eval_rows and the
    # ipo/kto/orpo/simpo pair loops documented
    assert len(guide.encode()) < 22_000
    heads = [h for _, level, h in docs._headings(guide) if level == 2]
    ladder = guide[guide.index("## Validating a new family"):guide.index("## Rules")]
    assert "## Validating a new family" in guide and len(ladder.strip().splitlines()) == 4
    said = " ".join(guide.split())
    assert "The ladder: `poros agents docs models any other model`." in said
    assert "How: `poros agents docs mac validated`." in said
    assert "Rules" in heads and "docs/" not in guide
    rules = guide[guide.index("## Rules"):guide.index("## On a Mac")]
    assert "200-step `prove --unvalidated` pass's `next` says in words" in " ".join(rules.split())
    # what the guide said is in the doc it points at, and the pointers read it
    whole = " ".join(_agent("docs", "models", "any", "other", "model")["text"].split())
    assert whole.startswith("## Any other model")
    for fact in ("no `next` carries a consent flag", "`{surface, step, tensor}`",
                 "Poros never marks a family validated itself",
                 "`submit` says where each file goes", "within 1e-4",
                 "`train <model> --data <file> --unvalidated` trains the family at any step"):
        assert fact in whole, fact
    mac = " ".join(_agent("docs", "mac", "validated")["text"].split())
    assert mac.startswith("## A model Poros has not validated")
    for fact in ("the nearest model of its kind in the Mac table", "`store_path` when two disks "
                 "have room, then `unvalidated`, `download` and `all`", "sends a stored token",
                 "not in bf16", "`export` says `support: user_asserted`"):
        assert fact in mac, fact


@pytest.mark.parametrize("env, shown", [
    ({}, "~/.cache/huggingface/hub"),
    ({"HF_HOME": "/data/hf"}, "/data/hf/hub"),
    ({"HF_HOME": "/data/hf", "HF_HUB_CACHE": "/fast/hub"}, "/fast/hub"),
    ({"HUGGINGFACE_HUB_CACHE": "/data/legacy"}, "/data/legacy"),
    ({"XDG_CACHE_HOME": "/xdg"}, "/xdg/huggingface/hub"),
])
def test_the_download_question_names_the_cache_hugging_face_uses(monkeypatch, env, shown):
    """The download and dataset questions say where the files land: HF_HUB_CACHE, HF_HOME, or
    the default, as huggingface_hub reads them, never a hardcoded ~/.cache/huggingface."""
    from poros.cli.verbs import shared

    for k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert shared.hf_cache_shown() == shown
    assert f"into {shown}." in shared.download_question("Qwen/Qwen2.5-7B", "x", "--download",
                                                        "Yes").prompt
    assert f"into {shown}," in shared.dataset_question("org/data").prompt
