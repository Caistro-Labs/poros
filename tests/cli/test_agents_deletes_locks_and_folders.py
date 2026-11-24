"""`poros agents`: deletes, locks, folders and refusals.
No weights: fixture run folders and the simulator."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from poros import runs
from poros.core import checkpoints as saved
from poros.core.errors import EXIT_CODES
from poros.core.refusal import Refusal
from tests.cli.test_agents import _a_run, _again, _agent, _poros

RUN = "1105-0900-aaaa"


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


def _hold(folder: Path, name: str = "run.lock"):
    import fcntl

    lock = open(folder / name, "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock


# a confirmed delete is what the question named


def test_a_confirmed_delete_takes_only_what_it_named(home):
    folder = _a_run(steps=(10, 20, 29), final=False, state="cancelled")
    asked = _agent("checkpoint", "delete", "--run-id", RUN, code=2)
    command = asked["questions"][0]["choices"][0]["command"]
    words = shlex.split(command)
    assert "--run-id" not in words and words[-1] == "--yes"
    assert sorted(w for w in words if w.startswith("poros://")) == sorted(
        f"poros://{RUN}/weights/{n}" for n in (10, 20, 29))
    for step in (30, 40):  # a continue since: new checkpoints and the finished adapter
        (folder / f"checkpoint-{step}").mkdir()
        (folder / f"checkpoint-{step}" / ".poros_checkpoint_complete").touch()
    (folder / "adapter_model.safetensors").write_bytes(b"f")
    done = _again(command)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [r["path"] for r in json.loads(done.stdout)["checkpoints"]] == [
        f"poros://{RUN}/weights/{n}" for n in (29, 20, 10)]
    assert sorted(p.name for p in folder.glob("checkpoint-*")) == ["checkpoint-30",
                                                                   "checkpoint-40"]
    assert (folder / "adapter_model.safetensors").exists()


def test_a_direct_yes_with_filters_still_deletes_what_they_match(home):
    folder = _a_run(steps=(10, 20), final=False, state="cancelled")
    _agent("checkpoint", "delete", "--run-id", RUN, "--yes")
    assert not list(folder.glob("checkpoint-*"))


def test_delete_holds_the_run_so_a_continue_cannot_race_it(home):
    folder = _a_run(state="cancelled")
    held = _hold(folder)  # what a continue holds while it trains
    try:
        hit = _agent("checkpoint", "delete", f"{RUN}/weights/10", "-y", code=2)["error"]
    finally:
        held.close()
    assert hit["code"] == "E_USAGE" and hit["hint"] == f"poros agents run stop {RUN}"
    assert (folder / "checkpoint-9").exists()


# continue holds the run before it writes, and a new id is never one taken


def test_continue_takes_the_lock_before_it_writes_more_steps(home):
    from poros.cli.verbs import train_run

    folder = _a_run(state="cancelled")
    before = (folder / "action.json").read_text()
    held = _hold(folder)
    try:
        with pytest.raises(Refusal) as hit:
            train_run.continue_run(runs.read(folder), more=5)
    finally:
        held.close()
    assert hit.value.title == "Already running"
    assert (folder / "action.json").read_text() == before


def test_a_new_id_is_never_one_a_run_has(home, monkeypatch):
    import secrets

    said = iter(["aaaa", "aaaa", "bbbb"])
    monkeypatch.setattr(secrets, "token_hex", lambda n: next(said))
    first = runs.new_id()
    runs.create(first, {"model": "m"})
    assert runs.new_id().endswith("-bbbb")


# a step's folder is the one listed, by the name the trainer gives it


def test_a_step_named_another_way_is_no_step_and_a_link_is_never_walked(home):
    folder = _a_run(steps=(10,), final=False)
    ssh = home / "me" / ".ssh"
    ssh.mkdir()
    (ssh / "id_fake").write_text("PRIVATE")
    (ssh / ".poros_checkpoint_complete").touch()
    (folder / "checkpoint-007").mkdir()
    (folder / "checkpoint-007" / ".poros_checkpoint_complete").touch()
    (folder / "checkpoint-7").symlink_to(ssh)
    (folder / "checkpoint-²").mkdir()
    (folder / "checkpoint-²" / ".poros_checkpoint_complete").touch()
    listed = _agent("checkpoint", "list", "--run-id", RUN)["checkpoints"]
    assert [r["path"] for r in listed] == [f"poros://{RUN}/weights/10"]
    assert _agent("checkpoint", "storage")["schema"] == "poros.checkpoints.v1"
    assert _agent("run", "info", RUN)["runs"][0]["id"] == RUN
    miss = _agent("checkpoint", "download", f"{RUN}/weights/7", code=2)["error"]
    assert miss["code"] == "E_NOT_FOUND"
    assert not [p for p in (home / "work").rglob("*") if p.name == "id_fake"]


def test_the_trainer_reads_a_step_as_it_writes_one(tmp_path):
    for name, step in (("checkpoint-12", 12), ("checkpoint-0", 0), ("checkpoint-007", None),
                       ("checkpoint-²", None), ("checkpoint-١", None), ("checkpoint-best", None)):
        (tmp_path / name).mkdir()
        assert saved.checkpoint_step(tmp_path / name) == step, name


def test_a_file_with_a_second_name_is_not_copied(home):
    folder = _a_run(steps=(10,), final=False)
    mine = home / "me" / "notes"
    mine.write_text("mine")
    os.link(mine, folder / "checkpoint-9" / "tokenizer.json")
    hit = _agent("checkpoint", "download", f"{RUN}/weights/10", code=2)["error"]
    assert hit["code"] == "E_USAGE" and "one name" in hit["message"]
    assert not list((home / "work").iterdir())


def test_a_hint_to_copy_again_quotes_its_folder(home):
    _a_run(steps=(10,), final=False)
    _agent("checkpoint", "download", f"{RUN}/weights/10", "-o", "my models")
    again = _agent("checkpoint", "download", f"{RUN}/weights/10", "-o", "my models", code=2)
    words = shlex.split(again["error"]["hint"])
    assert words[words.index("-o") + 1].endswith("my models") and words[-1] == "--force"
    assert _again(again["error"]["hint"]).returncode == 0


# one folder that cannot be read is skipped, and named it says so


def _broken(home: Path, name: str, status=None, action=None, events: str = "") -> Path:
    folder = home / "poros" / "runs" / name
    folder.mkdir(parents=True)
    good = {"schema": "poros.run.status.v1", "id": name, "state": "succeeded", "pid": None,
            "created_at": "2026-11-05T09:00:00+00:00", "output_dir": str(folder)}
    (folder / "status.json").write_text(json.dumps(good if status is None else status))
    (folder / "action.json").write_text(json.dumps({"model": "m"} if action is None else action))
    if events:
        (folder / "events.jsonl").write_text(events)
    return folder


@pytest.mark.parametrize("status, action, events", [
    ([], None, ""),
    ({"id": "1105-0800-bad0", "state": "succeeded", "created_at": None, "output_dir": "."}, None,
     ""),
    ({"id": "1105-0800-bad0", "state": "running", "pid": 10**30, "created_at": "x",
      "output_dir": "."}, None, ""),
    ({"id": "1105-0800-bad0", "state": "running", "pid": "12", "created_at": "x",
      "output_dir": "."}, None, ""),
    ({"id": "1105-0800-bad0", "state": "succeeded", "created_at": "x", "output_dir": 5}, None, ""),
    (None, [], ""),
    (None, None, "[" * 100_000 + "\n"),
])
def test_a_folder_that_cannot_be_read_breaks_no_command(home, status, action, events):
    _a_run()
    _broken(home, "1105-0800-bad0", status, action, events)
    assert RUN in [r["id"] for r in _agent("run", "list")["runs"]]
    # machine answers, ready or not: a Mac with no torch installed is not (E_MACHINE_NOT_READY)
    machine = _poros("agents", "machine")
    assert machine.returncode in (0, EXIT_CODES["E_MACHINE_NOT_READY"]), machine.stderr[-400:]
    assert json.loads(machine.stdout.strip().splitlines()[-1])["schema"] == "poros.machine.v1"
    assert _agent("checkpoint", "list")["schema"] == "poros.checkpoints.v1"
    assert _agent("run", "watch", RUN)["schema"] == "poros.run_live.v1"


def test_a_run_named_whose_folder_cannot_be_read_is_refused_plainly(home):
    _broken(home, "1105-0800-bad0", status=[])
    for command in (("run", "info", "1105-0800-bad0"),
                    ("checkpoint", "list", "--run-id", "1105-0800-bad0")):
        hit = _agent(*command, code=2)["error"]
        assert hit["code"] == "E_NOT_FOUND" and "can't be read" in hit["message"], command


def test_a_simulated_train_beside_a_broken_folder(home):
    (home / "work" / "d.jsonl").write_text(json.dumps({"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}) + "\n")
    _broken(home, "1105-0800-bad0", action=[])
    done = _agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--steps", "2",
                  env={"POROS_DEV_SIMULATE_LOAD": "0", "POROS_DEV_SIMULATE_STEP": "0.01"})
    assert done["schema"] == "poros.train.v1"


# a command is rewritten word by word, and a sentence only where a command is whole


def test_a_command_keeps_its_words_whatever_they_say(monkeypatch):
    from poros.cli import contract

    monkeypatch.setenv("POROS_AGENT", "1")
    path = "./poros runs/it's \"x\"\nnotes.jsonl"
    said = contract.as_agent(shlex.join(["poros", "train", "m", "--data", path, "--detach"]))
    assert shlex.split(said) == ["poros", "agents", "train", "m", "--data", path]
    monkeypatch.setattr(contract, "_waiting", True)
    said = contract.as_agent(shlex.join(["poros", "train", "m", "--data", path]))
    assert shlex.split(said) == ["poros", "agents", "train", "m", "--data", path, "--wait"]
    monkeypatch.setattr(contract, "_waiting", False)
    out = contract._for_agent({"message": f"read {path} with poros runs",
                               "hint": "see the log; poros jobs resume 7f3a retries it",
                               "remediation": "then `poros runs continue 7f3a`"})
    assert out["message"] == f"read {path} with poros runs"
    assert out["hint"] == "see the log; poros jobs resume 7f3a retries it"
    assert out["remediation"] == "then `poros agents run continue 7f3a`"


def test_datas_next_runs_as_written_from_a_folder_named_like_a_command(home):
    (home / "work" / "poros data").mkdir()
    (home / "work" / "poros data" / "notes.jsonl").write_text(json.dumps({"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}) + "\n")
    said = _agent("data", "./poros data/notes.jsonl")["next"][0]
    words = shlex.split(said)
    assert words[words.index("--data") + 1].endswith("poros data/notes.jsonl")
    done = _again(said + " --steps 1",
                  env={"POROS_DEV_SIMULATE_LOAD": "0", "POROS_DEV_SIMULATE_STEP": "0.01"})
    assert done.returncode == 0 and json.loads(done.stdout)["schema"] == "poros.train.v1"


# a folder named like a word is no key, and a path is kept whole


def test_a_folder_named_sk_something_is_no_key():
    from poros.core.errors import no_keys

    for word in ("/runs/sk-learn-baseline-v2/adapter", "sk-learn-a-much-longer-folder-name-here"):
        assert no_keys(word) == word
    for key in ("sk-proj-abc123def456ghi789jkl", "sk-" + "Ab3" * 16, "sk-ant-api03-" + "x" * 20,
                "sk-or-v1-" + "0f" * 32, "sk-" + "a" * 20):
        assert key not in no_keys(f"mine is {key} ok"), key


def test_a_path_field_keeps_its_path(monkeypatch):
    from poros.cli import contract

    monkeypatch.setenv("POROS_AGENT", "1")
    odd = "/w/sk-" + "a" * 24 + "/adapter"
    out = contract._for_agent({"output": odd, "load": f"f({odd!r})", "note": odd})
    assert out["output"] == odd and odd in out["load"] and odd not in out["note"]


def test_an_export_to_a_folder_named_sk_says_its_path(home):
    from tests.cli.test_agents import _exportable

    _exportable(home)
    out = _agent("export", "last", "--output", "./sk-learn-baseline-v2/adapter")
    assert out["output"] == str(home / "work" / "sk-learn-baseline-v2" / "adapter")
    assert repr(out["output"]) in out["load"]


# an export over an earlier one replaces its own names and no one else's


def test_an_export_over_another_keeps_a_users_links_and_drops_its_own_stale_files(home):
    from tests.cli.test_agents import _exportable

    _exportable(home)
    out = home / "work" / "adapter"
    out.mkdir()
    (out / "AGENTS.md").write_text("rules")
    (out / "CLAUDE.md").symlink_to("AGENTS.md")
    stale = ("tokenizer_config.json", "chat_template.jinja", "adapter_model.bin")
    for name in stale:
        (out / name).write_text("old")
    mine = home / "me" / ".bashrc"
    mine.write_text("mine")
    (out / "adapter_config.json").symlink_to(mine)
    _agent("export", "last", "--output", "./adapter", "--overwrite")
    assert (out / "CLAUDE.md").is_symlink() and (out / "AGENTS.md").read_text() == "rules"
    assert not [name for name in stale if os.path.lexists(out / name)]
    assert not (out / "adapter_config.json").is_symlink() and mine.read_text() == "mine"
    assert (out / "adapter_model.safetensors").is_file()


@pytest.mark.parametrize("where", ["home", "file", "run", "root"])
def test_an_export_where_it_cannot_go_is_a_usage_refusal(home, where):
    from tests.cli.test_agents import _exportable

    folder = _exportable(home)
    (home / "work" / "a-file").write_text("x")
    target = {"home": str(home / "me"), "file": "a-file", "run": str(folder), "root": "/"}[where]
    hit = _agent("export", "last", "--output", target, code=2)["error"]
    assert hit["code"] == "E_USAGE", hit
    if where == "root":  # no name to stage a merge beside: said before a download
        merged = _agent("export", "last", "--format", "merged", "--output", "/", code=2)["error"]
        assert "root" in hit["message"] and "root" in merged["message"], (hit, merged)
    assert (folder / "adapter_model.safetensors").is_file()


# a file's name in bytes that are not UTF-8


def _bytes_named(home: Path) -> bytes:
    name = b"bad\xff.jsonl"
    with open(os.path.join(os.fsencode(home / "work"), name), "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "hi"},
                                         {"role": "assistant", "content": "yo"}]}) + "\n")
    return name


@pytest.mark.skipif(sys.platform == "darwin", reason="APFS refuses a file name that is not "
                    "UTF-8 (EILSEQ): no file on a Mac has such a name")
def test_a_name_that_is_not_utf8_is_said_and_trains(home):
    import subprocess

    name = _bytes_named(home)
    for args in ([b"data", name], [b"train", b"qwen2.5-0.5b-instruct", b"--data", name,
                                    b"--steps", b"1"]):
        proc = subprocess.run(
            [sys.executable, "-c", "from poros.cli.main import cli; cli()", "agents", *args],
            capture_output=True, stdin=subprocess.DEVNULL,
            env={**os.environ, "POROS_DEV_SIMULATE": "1", "POROS_DEV_SIMULATE_LOAD": "0",
                 "POROS_DEV_SIMULATE_STEP": "0.01"})
        said = json.loads(proc.stdout.decode())
        assert proc.returncode == 0 and said["schema"] != "poros.error.v1", said
        assert "�" in json.dumps(said, ensure_ascii=False)


def test_emit_writes_a_lone_surrogate_as_a_replacement(capsys):
    from poros.cli import contract
    from poros.cli.schemas import VersionOutput

    contract.emit(VersionOutput(version="bad\udcff"))
    assert json.loads(capsys.readouterr().out)["version"] == "bad�"


# another word for a command names it, and no state-changing command is a guess


@pytest.mark.parametrize("said, hint", [
    (("run", "status", RUN), f"poros agents run info {RUN}"),
    (("run", "show", RUN), f"poros agents run info {RUN}"),
    (("run", "start", RUN), f"poros agents run continue {RUN}"),
    (("run", "cancel", RUN), f"poros agents run stop {RUN}"),
    (("checkpoint", "get", "x y"), "poros agents checkpoint download 'x y'"),
    (("checkpoint", "rm", "x"), "poros agents checkpoint delete x"),
    (("chekpoint", "list"), "poros agents checkpoint list"),
])
def test_a_miss_offers_what_it_meant(home, said, hint):
    assert _agent(*said, code=2)["error"]["hint"] == hint


def test_a_guess_never_changes_anything(home):
    changes = {"train", "run stop", "run continue", "run report", "checkpoint download",
               "checkpoint delete", "export", "recipes", "prove"}
    for said in ("run stp", "run contineu", "checkpoint delet", "checkpoint downlod", "expor",
                 "trian"):
        hint = _agent(*said.split(), code=2)["error"]["hint"]
        assert hint.removeprefix("poros agents ") not in changes, (said, hint)


# a hint runs as written


def _not_its_form(proc) -> bool:
    """The command ran past its own words: whatever it then said, it was a command."""
    try:
        said = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return proc.returncode == 0  # help is a document
    message = (said.get("error") or {}).get("message", "")
    return not any(bad in message for bad in (
        "no poros agents", " here; its flags", "it takes at most", "said twice", "Too many"))


def test_every_hint_in_a_sweep_of_refusals_runs_as_written(home):
    _a_run(steps=(10,))
    runs.create("1105-0901-bbbb", {"model": "Qwen/Qwen2.5-0.5B", "name": "my run", "steps": 2})
    _agent("checkpoint", "download", f"{RUN}/weights/10", "-o", "my models")
    refusals = [("chekpoint", "list"), ("run", "status", RUN), ("schema", "evnts"),
                ("schema", "zzzz"), ("run", "info"), ("export", "nope"),
                ("checkpoint", "download", f"{RUN}/weights/10", "-o", "my models"),
                ("train", "qwen2.5-0.5b-instruct", "--data", "nope.jsonl"),
                ("eval", "my run"), ("fit", "qwen2.5-7b", "--nope", "1"),
                ("checkpoint", "info", f"{RUN}/weights/99")]
    env = {"POROS_DEV_SIMULATE_LOAD": "0", "POROS_DEV_SIMULATE_STEP": "0.001"}
    for said in refusals:
        hint = json.loads(_poros("agents", *said).stdout)["error"]["hint"]
        assert hint.startswith("poros agents "), (said, hint)
        assert _not_its_form(_again(hint, env=env)), (said, hint)


# the contract's edges


def _cli(*args, **how):
    import subprocess
    import sys

    return subprocess.Popen(
        [sys.executable, "-c", "from poros.cli.main import cli; cli()", "agents", *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "POROS_DEV_SIMULATE": "1"}, **how)


def test_sigterm_says_interrupted_as_ctrl_c_does(home):
    import signal
    import time

    _a_run(state="running")
    proc = _cli("run", "watch", RUN, "--wait", "1000")
    time.sleep(3)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 130, err[-800:]
    assert json.loads(out)["error"]["code"] == "E_INTERRUPTED"


def test_a_reader_that_went_is_no_traceback(home):
    _a_run()
    proc = _cli("run", "list")
    proc.stdout.close()  # before it writes: the pipe has no reader
    proc.wait(timeout=60)
    err = proc.stderr.read()
    assert proc.returncode == 0 and "Traceback" not in err, err[-800:]


def test_a_flag_like_value_is_that_flags_value(home):
    _a_run()
    for value in ("-h", "--json"):
        hit = _agent("eval", RUN, "--data", value, code=2)["error"]
        assert hit["code"] == "E_NOT_FOUND" and hit["message"].startswith(f"{value} is not a ")


def test_a_name_too_long_for_the_disk_is_a_usage_refusal(home):
    _a_run(steps=(10,))
    assert _agent("export", "x" * 300, code=2)["error"]["code"] == "E_NOT_FOUND"
    hit = _agent("checkpoint", "download", f"{RUN}/weights/10", "-o", "y" * 300, code=2)
    assert hit["error"]["code"] == "E_USAGE"


def test_a_file_where_a_folder_goes_is_a_usage_refusal(home):
    _a_run(steps=(10,))
    (home / "work" / "a-file").write_text("x")
    hit = _agent("checkpoint", "download", f"{RUN}/weights/10", "-o", "a-file", code=2)
    assert hit["error"]["code"] == "E_USAGE" and "is a file" in hit["error"]["message"]


def test_a_home_it_cannot_write_is_the_disk(home):
    (home / "work" / "d.jsonl").write_text(json.dumps({"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}) + "\n")
    locked = home / "locked"
    (locked / "runs").mkdir(parents=True)
    (locked / "runs").chmod(0o500)
    try:
        hit = _agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--steps", "1",
                     code=5, env={"POROS_HOME": str(locked), "POROS_DEV_SIMULATE_LOAD": "0"})
    finally:
        (locked / "runs").chmod(0o700)
    assert hit["error"]["code"] == "E_DISK"


def test_the_guide_names_every_exit_code():
    from poros.cli.verbs import agents
    from poros.core.errors import EXIT_CODES

    guide = (Path(agents.__file__).parents[2] / "agents.md").read_text()
    rules = guide[guide.index("Errors are"):guide.index("`command`, `next`")]
    for code in sorted(set(EXIT_CODES.values())):
        assert f"{code} " in rules, code


# a flag or a run said twice


@pytest.mark.parametrize("said", [
    ("checkpoint", "list", "--run-id", "a", "--run-id", "b"),
    ("checkpoint", "list", "--limit", "1", "--limit=2"),
    ("run", "info", RUN, "1105-0901-bbbb"),
    ("run", "watch", RUN, "--wait", "1", "--wait", "2"),
    ("train", "qwen2.5-0.5b-instruct", "--data", "a.jsonl", "--data", "b.jsonl"),
    ("eval", RUN, "--rows", "1", "--rows", "2"),
    ("eval", RUN, "1105-0901-bbbb"),
    ("export", RUN, "1105-0901-bbbb"),
])
def test_a_flag_or_a_run_said_twice_is_refused(home, said):
    _a_run()
    hit = _agent(*said, code=2)["error"]
    assert hit["code"] == "E_USAGE", hit
