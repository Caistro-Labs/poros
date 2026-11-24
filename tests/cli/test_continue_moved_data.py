"""A verb-form run goes on after its data file moved: `runs continue <run> --data FILE`.

It could not: with the file gone, `runs continue` said "Dataset file not found" in a config
file's words ("Point 'dataset:' at ..."), and `--data` on continue was refused as a changed
setting. A moved file is no change: the resume check compares its contents with what the
checkpoint recorded, and refuses other data. On the simulated trainer, with the record a real
checkpoint keeps written beside its stand-ins.
"""

from __future__ import annotations

import json
import shutil

import pytest

from poros import runs
from tests.cli.test_run_flow_wording import MODEL, _chats
from tests.cli.test_train_runs import home, say  # noqa: F401  (the simulated machine)


@pytest.fixture
def stopped(home, capsys, monkeypatch):  # noqa: F811
    """A run stopped at its fifth step with a checkpoint that records its data, as a trainer's
    does (poros_config.json, inputs.json): (the run, its data file)."""
    from poros.cli.ui import simulate
    from poros.preflight import resume

    data = home / _chats(home, 60, "g3data.jsonl")
    real = simulate.SimTrainer._event

    def event(self, i, *a, **k):
        if i == 4:
            (runs.root() / runs.all_runs()[0].id / "stop").touch()
        return real(self, i, *a, **k)

    monkeypatch.setattr(simulate.SimTrainer, "_event", event)
    code, _ = say(["train", MODEL, "--data", "g3data.jsonl", "--yes", "--steps", "12"], home,
                  capsys)
    monkeypatch.setattr(simulate.SimTrainer, "_event", real)
    (run,) = runs.all_runs()
    assert code == 7 and run.state == "cancelled"
    for ckpt in (runs.root() / run.id).glob("checkpoint-*"):
        (ckpt / "poros_config.json").write_text(json.dumps({"dataset": str(data)}))
        (ckpt / resume.INPUTS_RECORD).write_text(json.dumps(
            resume.inputs_record(resume.trained_on(str(data)), None)))
    return run, data


def _action(run) -> dict:
    return json.loads((runs.root() / run.id / "action.json").read_text())


def test_a_moved_data_file_is_said_in_this_commands_words(home, capsys, stopped):  # noqa: F811
    run, data = stopped
    (home / "sub").mkdir()
    shutil.move(data, home / "sub" / data.name)
    code, obj = say(["runs", "continue", run.id], home, capsys)
    err = obj["error"]
    assert (err["code"], code) == ("E_DATASET_NOT_FOUND", 6), err
    assert f"run {run.id} trained on {data}, and nothing is there now" in err["message"]
    assert err["hint"] == f"poros runs continue {run.id} --data <file>"
    assert "dataset:" not in err["message"] + err["hint"]  # a config file's words
    assert next(r for r in runs.all_runs() if r.id == run.id).state == "cancelled"


def test_continue_takes_the_same_data_at_another_path(home, capsys, stopped):  # noqa: F811
    run, data = stopped
    (home / "sub").mkdir()
    moved = home / "sub" / data.name
    shutil.move(data, moved)
    code, obj = say(["runs", "continue", run.id, "--data", "sub/g3data.jsonl"], home, capsys)
    assert code == 0 and obj["runs"][0]["state"] == "succeeded", obj
    assert obj["runs"][0]["step"] == 12
    assert _action(run)["data"] == str(moved)  # its record says where its data is now
    assert f"--data {moved}" in _action(run)["command"]


def test_continue_refuses_other_data_and_leaves_the_run_as_it_was(home, capsys,  # noqa: F811
                                                                  stopped):
    run, data = stopped
    before = _action(run)
    other = home / _chats(home, 20, "other.jsonl")
    code, obj = say(["runs", "continue", run.id, "--data", str(other)], home, capsys)
    err = obj["error"]
    assert (err["code"], code) == ("E_CONFIG", 2), err
    assert "holds other data" in err["message"] and "sha256" in err["message"]
    assert _action(run) == before
    assert next(r for r in runs.all_runs() if r.id == run.id).state == "cancelled"
    # a file that is not there, and a changed setting, are refused as before
    code, obj = say(["runs", "continue", run.id, "--data", "nowhere.jsonl"], home, capsys)
    assert obj["error"]["code"] == "E_DATASET_NOT_FOUND"
    code, obj = say(["runs", "continue", run.id, "--steps", "50"], home, capsys)
    assert obj["error"]["code"] == "E_USAGE" and "--steps" in obj["error"]["message"]


def test_the_agent_form_names_its_own_command(monkeypatch):
    from poros.cli import contract

    monkeypatch.setenv("POROS_AGENT", "1")
    assert contract.as_agent("poros runs continue 1117-1416-8499 --data <file>") == (
        "poros agents run continue 1117-1416-8499 --data <file>")
