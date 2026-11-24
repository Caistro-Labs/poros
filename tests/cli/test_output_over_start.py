"""A new run never trains into, or clears, the run it starts from; a replaced run is cleared last.

`poros train --from last --data more.jsonl --output out --overwrite`, where `out` is the folder
of the run it starts from, deleted that run's checkpoints and adapter (the overwrite's clearing
ran first) and then refused: E_NOT_FOUND, the checkpoint it starts from gone. It is refused
before any question and before anything is removed, in the verb form and the config form
(`output_dir` against `init_adapter:`). And an agreed `--overwrite` clears the old run only once
every check before the load has passed: a refusal on the way leaves that run whole.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poros.core.refusal import Refusal
from poros.preflight.disk import output_over_start
from tests.cli.test_run_flow_wording import MODEL, _chats, _screen
from tests.cli.test_train_runs import home, say  # noqa: F401  (the simulated machine)
from tests.conftest import checkout_env


def _held(folder: Path) -> dict:
    """Every file under ``folder`` with its contents' hash."""
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _flat(screen: str) -> str:
    """A screen's words on one line: a long sentence wraps on its rail."""
    return " ".join(screen.replace("│", " ").split())


def _poros(cwd: Path, *argv: str, agent: bool = False) -> subprocess.CompletedProcess:
    # the simulated CUDA card, whatever the host: on a Mac the simulator previews the Mac itself,
    # whose store must have the model's 20 GB and 10% of its real disk free (test_mac's to check)
    env = checkout_env({**os.environ, "POROS_HOME": str(cwd / "home"), "HF_HUB_OFFLINE": "1",
                        "POROS_DEV_SIMULATE": "1", "POROS_DEV_SIMULATE_STEP": "0.001",
                        "POROS_DEV_SIMULATE_GPU": "RTX 4090:24.0", "NO_COLOR": "1",
                        "COLUMNS": "200"})
    for name in ("POROS_OUTPUT", "POROS_AGENT"):
        env.pop(name, None)
    if agent:
        env["POROS_AGENT"] = "1"
    return subprocess.run([sys.executable, "-c", "from poros.cli.main import cli; cli()", *argv],
                          capture_output=True, text=True, env=env, cwd=cwd,
                          stdin=subprocess.DEVNULL)


@pytest.fixture
def world(tmp_path):
    """A folder with data files and a finished run in ./out: (the folder, what ./out holds)."""
    _chats(tmp_path, 20)
    _chats(tmp_path, 40, "more.jsonl")
    done = _poros(tmp_path, "train", MODEL, "--data", "chats.jsonl", "--steps", "8", "--yes",
                  "--output", "out")
    assert done.returncode == 0, done.stdout + done.stderr
    held = _held(tmp_path / "out")
    assert any(name.startswith("checkpoint-") for name in held)
    return tmp_path, held


@pytest.mark.parametrize("agent", [False, True], ids=["person", "agent"])
@pytest.mark.parametrize("flags", [("--overwrite",), ("--overwrite", "--yes"), ()],
                         ids=["overwrite", "overwrite-yes", "no-flag"])
def test_from_a_run_into_its_own_folder_is_refused_and_removes_nothing(world, agent, flags):
    cwd, held = world
    verb = ("agents", "train") if agent else ("train",)
    said = _poros(cwd, *verb, "--from", "last", "--data", "more.jsonl", "--output", "out",
                  *flags, agent=agent)
    assert said.returncode == 2, said.stdout + said.stderr
    assert _held(cwd / "out") == held  # the run it starts from: every file, as it was
    if agent:
        err = json.loads(said.stdout)["error"]
        assert err["code"] == "E_CONFIG" and "nothing was removed" in err["message"]
        text = err["message"]
    else:
        text = _flat(said.stderr)
        assert "Not an output folder" in text and "nothing was removed" in text
        assert "Needs an answer" not in text  # refused before any question
    assert "is the folder of the run this one starts from" in text
    assert "name another output folder" in text
    assert len(list((cwd / "home" / "runs").iterdir())) == 1  # no run recorded


@pytest.mark.parametrize("agent", [False, True], ids=["person", "agent"])
@pytest.mark.parametrize("start", ["./out", "./out/checkpoint-7"], ids=["folder", "checkpoint"])
def test_a_configs_output_dir_on_its_init_adapter_is_refused_and_removes_nothing(world, agent,
                                                                                 start):
    cwd, held = world
    assert (cwd / "out" / "checkpoint-7").is_dir()
    (cwd / "run.yaml").write_text(
        f"model_name_or_path: {MODEL}\ndataset: ./more.jsonl\noutput_dir: ./out\n"
        f"max_steps: 4\ninit_adapter: {start}\n")
    said = _poros(cwd, "train", "run.yaml", "--overwrite", *(["--json"] if agent else []),
                  agent=agent)
    assert said.returncode == 2, said.stdout + said.stderr
    assert _held(cwd / "out") == held
    text = json.loads(said.stdout)["error"]["message"] if agent else _flat(said.stderr)
    assert "the run this one starts from" in text and "nothing was removed" in text
    if agent:
        assert json.loads(said.stdout)["error"]["code"] == "E_CONFIG"


@pytest.mark.parametrize("agent", [False, True], ids=["person", "agent"])
def test_every_name_for_the_run_it_starts_from_is_refused(world, agent):
    """The same folder by any spelling: relative, with a trailing slash, absolute, through `..`,
    a symlink to it, a folder inside it or one that holds it, and `--from <run>/weights/<n>`."""
    cwd, held = world
    verb = ("agents", "train") if agent else ("train",)
    for start, output in (("last", "."),  # before the link: a folder holding one is refused
                          ("last", "out/"), ("last", str(cwd / "out")), ("last", "./x/../out"),
                          ("last", "link"), ("last", "link/"), ("last", "out/nested"),
                          ("last/weights/8", "out"), ("last/weights/8", "out/checkpoint-7"),
                          ("last/weights/8", "link/sub")):
        if output == "out/" and not (cwd / "link").exists():
            (cwd / "link").symlink_to(cwd / "out")
        said = _poros(cwd, *verb, "--from", start, "--data", "more.jsonl", "--output", output,
                      "--overwrite", "--yes", agent=agent)
        assert said.returncode == 2, (start, output, said.stdout + said.stderr)
        text = (json.loads(said.stdout)["error"]["message"] if agent else _flat(said.stderr))
        assert "nothing was removed" in text and "the run this one starts from" in text, (
            start, output, text)
        assert _held(cwd / "out") == held, (start, output)
        assert not (cwd / "out" / "nested").exists() and not (cwd / "out" / "sub").exists()
    assert len(list((cwd / "home" / "runs").iterdir())) == 1  # no run recorded
    apart = _poros(cwd, *verb, "--from", "last", "--data", "more.jsonl", "--output", "out2",
                   "--steps", "2", "--yes", agent=agent)
    assert apart.returncode == 0, apart.stdout + apart.stderr  # another folder trains
    assert _held(cwd / "out") == held


@pytest.mark.parametrize("agent", [False, True], ids=["person", "agent"])
def test_a_configs_init_adapter_by_another_name_is_refused(world, agent):
    """The config form, its paths relative, through a symlink, with trailing slashes."""
    cwd, held = world
    (cwd / "link").symlink_to(cwd / "out")
    for out, start in (("./out/", "./link/"), ("./link", "./out/checkpoint-7/"),
                       ("./out/deeper", "./link"), (str(cwd), "./out")):
        (cwd / "run.yaml").write_text(
            f"model_name_or_path: {MODEL}\ndataset: ./more.jsonl\noutput_dir: {out}\n"
            f"max_steps: 4\ninit_adapter: {start}\n")
        said = _poros(cwd, "train", "run.yaml", "--overwrite", *(["--json"] if agent else []),
                      agent=agent)
        assert said.returncode == 2, (out, start, said.stdout + said.stderr)
        text = json.loads(said.stdout)["error"]["message"] if agent else _flat(said.stderr)
        assert "the run this one starts from" in text and "nothing was removed" in text, text
        assert _held(cwd / "out") == held and not (cwd / "out" / "deeper").exists()


def test_which_folders_are_the_run_it_starts_from(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoint-7").mkdir(parents=True)
    for out, start, how in (
            (run, run, "is the folder of the run"),
            (run, run / "checkpoint-7", "holds the run"),
            (tmp_path, run, "holds the run"),
            (run / "again", run, "is inside the run"),
            (str(run) + "/", run / "checkpoint-7", "holds the run")):
        refused = output_over_start(out, start)
        assert isinstance(refused, Refusal) and refused.code == "E_CONFIG", (out, start)
        assert how in refused.reason and "nothing was removed" in refused.context
    link = tmp_path / "link"
    link.symlink_to(run)
    assert output_over_start(link, run) is not None  # the same folder by another name
    assert output_over_start(tmp_path / "other", run, run / "checkpoint-7") is None  # apart
    assert output_over_start(tmp_path / "run2", run) is None  # a name it begins, not its folder
    assert output_over_start(run, None) is None and output_over_start(None, run) is None
    assert output_over_start(run, run, resume_from=str(run / "checkpoint-7")) is None  # a resume


# ── clearing happens last ──────────────────────────────────────────────────────────────────────


def test_an_overwrite_refused_on_the_way_leaves_the_old_run_whole(home, capsys, monkeypatch):  # noqa: F811
    """`--overwrite` cleared the folder before the run's checks: one that then refused (the
    model, the data, the fit) had already deleted the run it was to replace."""
    from poros.preflight import command

    _chats(home, 20)
    _chats(home, 40, "more.jsonl")
    code, first = say(["train", MODEL, "--data", "chats.jsonl", "--steps", "8", "--output",
                       "out"], home, capsys)
    assert code == 0, first
    held = _held(home / "out")
    real = command.arch_refusal  # a check of the run's own flow, after its settings are written
    monkeypatch.setattr(command, "arch_refusal", lambda *a, **k: Refusal(
        "E_NETWORK", "No network", "the Hub cannot be reached"))
    code, text = _screen(["train", MODEL, "--data", "more.jsonl", "--steps", "5", "--output",
                          "out", "--overwrite", "--rank", "8"], home, capsys)
    assert code != 0 and "the Hub cannot be reached" in text
    assert _held(home / "out") == held  # its checkpoints, adapter, events, and its own settings
    monkeypatch.setattr(command, "arch_refusal", real)
    code, again = say(["train", MODEL, "--data", "more.jsonl", "--steps", "5", "--output", "out",
                       "--overwrite"], home, capsys)
    assert code == 0, again
    now = _held(home / "out")
    assert not any(name.startswith("checkpoint-7") for name in now)  # replaced, as agreed
    assert any(name.startswith("checkpoint-4") for name in now)
    assert json.loads((home / "out" / "config.resolved.json").read_text())["max_steps"] == 5
