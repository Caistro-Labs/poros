"""A first run's mistakes: each is a refusal with its own code and a sentence, in seconds, before
any download. None is a traceback or E_INTERNAL, and none leaves a run behind to list.

On the simulated trainer (no GPU): the verb form (`poros train <model> --data`), the config form
(`poros train run.yaml`) and the Python API's own check.
"""

from __future__ import annotations

import json
import os

import pytest

from poros import runs
from tests.cli.test_train_runs import home, say  # noqa: F401  (the simulated machine)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
needs_a_user = pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                                  reason="root writes a folder whatever its mode")


def _config(home, capsys, **fields):  # noqa: F811
    """`poros train run.yaml --json` in process: (exit code, the one JSON object)."""
    from typer.testing import CliRunner

    from poros.cli import main

    said = {"model_name_or_path": MODEL, "dataset": str(home / "notes.jsonl"),
            "output_dir": str(home / "out"), "max_steps": 3, **fields}
    cfg = home / "run.yaml"
    cfg.write_text("".join(f"{k}: {json.dumps(v)}\n" for k, v in said.items()))
    res = CliRunner().invoke(main.app, ["train", str(cfg), "--json"])
    capsys.readouterr()
    line = [x for x in res.output.splitlines() if x.startswith('{"schema"')]
    assert len(line) == 1, res.output
    return res.exit_code, json.loads(line[0])


# ── the output folder (the first check, on every form) ─────────────────────────────────────────


def _folders(home):  # noqa: F811
    """(what --output names, the code it is refused with, a word its sentence holds)."""
    (home / "afile").write_text("x")
    locked = home / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    return [(home / "afile", "E_CONFIG", "is a file, not a folder"),
            (home / "afile" / "sub", "E_CONFIG", "is a file, not a folder"),
            (locked / "out", "E_DISK", "cannot be written")]


@needs_a_user
def test_an_output_folder_that_cannot_be_made_is_refused_by_the_verb(home, capsys):  # noqa: F811
    for folder, code, why in _folders(home):
        exit_code, obj = say(["train", MODEL, "--data", "notes.jsonl", "--yes", "--steps", "3",
                              "--output", str(folder)], home, capsys)
        err = obj["error"]
        assert err["code"] == code and exit_code == err["exit_code"], err
        assert str(folder) in err["message"] and why in err["message"], err
        assert "refused before the model download" in err["message"]
        assert not folder.exists() or folder.is_file()  # nothing was made on the way
    assert runs.all_runs() == []  # and no run is recorded for a refusal


@needs_a_user
def test_an_output_folder_that_cannot_be_made_is_refused_by_a_config(home, capsys):  # noqa: F811
    for folder, code, why in _folders(home):
        exit_code, obj = _config(home, capsys, output_dir=str(folder))
        err = obj["error"]
        assert err["code"] == code and exit_code == err["exit_code"], err
        assert str(folder) in err["message"] and why in err["message"], err


@needs_a_user
def test_a_home_that_cannot_hold_runs_is_refused_before_a_run_is_recorded(home, capsys,  # noqa: F811
                                                                          monkeypatch):
    locked = home / "nohome"
    locked.mkdir()
    locked.chmod(0o555)
    monkeypatch.setenv("POROS_HOME", str(locked))
    exit_code, obj = say(["train", MODEL, "--data", "notes.jsonl", "--yes", "--steps", "3"],
                         home, capsys)
    err = obj["error"]
    assert (err["code"], exit_code) == ("E_DISK", 5), err
    assert "POROS_HOME" in err["message"] and str(locked) in err["message"]


def test_a_full_disk_at_the_output_folder_is_a_disk_refusal(home, capsys, monkeypatch):  # noqa: F811
    """What no check sees first (the disk fills as the folder is made) is said the same way."""
    import errno
    from pathlib import Path

    real = Path.mkdir

    def full(self, *a, **k):
        if self.name == "out":
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", full)
    exit_code, obj = _config(home, capsys)
    err = obj["error"]
    assert (err["code"], exit_code) == ("E_DISK", 5), err
    assert "no space left on device" in err["message"] and str(home / "out") in err["message"]


@needs_a_user
def test_the_python_api_refuses_an_output_folder_before_the_download(tmp_path):
    from poros.core.errors import PorosConfigError, PorosDiskError
    from poros.preflight.disk import output_fault, require_output_dir

    (tmp_path / "afile").write_text("x")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    with pytest.raises(PorosConfigError, match="afile is a file, not a folder") as said:
        require_output_dir(tmp_path / "afile")
    assert said.value.code == "E_CONFIG"
    with pytest.raises(PorosDiskError, match="cannot be written") as said:
        require_output_dir(locked / "out")
    assert said.value.code == "E_DISK"
    # a folder that is there and read-only: a load that only reads a run passes, a run does not
    assert output_fault(locked, read_only_ok=True) is None
    assert output_fault(locked)[0] == "E_DISK"
    assert output_fault(tmp_path / "new" / "deeper") is None  # made when the run starts
    assert not (tmp_path / "new").exists()


@needs_a_user
def test_run_create_and_the_trainer_check_the_folder_first(tmp_path, monkeypatch):
    """Run.create and PorosTrainer.from_pretrained(output_dir=...) refuse before anything loads."""
    pytest.importorskip("torch")
    import poros
    from poros.core.errors import PorosConfigError
    from poros.train.trainer import PorosTrainer

    def loaded(*a, **k):
        raise AssertionError("the model was loaded")

    (tmp_path / "afile").write_text("x")
    monkeypatch.setattr(PorosTrainer, "__init__", loaded)
    with pytest.raises(PorosConfigError, match="is a file, not a folder"):
        poros.Run.create("Qwen/Qwen2.5-0.5B", output_dir=str(tmp_path / "afile"))
    with pytest.raises(PorosConfigError, match="is a file, not a folder"):
        PorosTrainer.from_pretrained("Qwen/Qwen2.5-0.5B", output_dir=str(tmp_path / "afile"))


# ── the model: a Hub id that is there, before any consent is offered for it ────────────────────

TYPO = "Qwen/Qwen2.5-0.5B-Instrcut"


def _hub_says(monkeypatch, kind: str, offline: bool = False) -> list:
    """The Hub answers every config.json fetch with ``kind`` (a huggingface_hub error's name);
    returns the models asked for."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    asked = []

    def fetch(model, revision=None):
        asked.append(model)
        cls = getattr(hub.utils, kind)
        raise cls.__new__(cls)  # its text is the library's: nothing here reads it

    monkeypatch.setattr("poros.plan.configs.fetch_config", fetch)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    return asked


def _typo(home, capsys, *flags):  # noqa: F811
    return say(["train", TYPO, "--data", "notes.jsonl", "--yes", *flags], home, capsys)


def test_a_hub_id_typo_is_not_found_before_any_consent_is_offered(home, capsys,  # noqa: F811
                                                                  monkeypatch):
    """`Qwen/Qwen2.5-0.5B-Instrcut` was offered "Train it anyway (--allow-unvalidated)", then
    crashed in the load with the Hub's traceback, E_INTERNAL, and a failed run to list."""
    asked = _hub_says(monkeypatch, "RepositoryNotFoundError")
    for flags in ((), ("--allow-unvalidated",), ("--unvalidated",)):
        code, obj = _typo(home, capsys, *flags)
        assert obj["schema"] == "poros.error.v1", obj  # an error, never a question
        err = obj["error"]
        assert (err["code"], code) == ("E_NOT_FOUND", 2), err
        assert f"{TYPO} is not on the Hugging Face Hub" in err["message"]
        assert "did you mean Qwen/Qwen2.5-0.5B-Instruct?" in err["message"]
        assert "Traceback" not in err["message"]
    assert asked and set(asked) == {TYPO}
    assert runs.all_runs() == []  # a refusal records no run


def test_a_name_near_no_catalog_model_offers_none(home, capsys, monkeypatch):  # noqa: F811
    _hub_says(monkeypatch, "RepositoryNotFoundError")
    _, obj = say(["train", "someone/a-model-of-their-own", "--data", "notes.jsonl", "--yes"],
                 home, capsys)
    assert obj["error"]["code"] == "E_NOT_FOUND" and "did you mean" not in obj["error"]["message"]


def test_a_gated_repo_and_no_network_each_have_a_code_and_a_sentence(home, capsys,  # noqa: F811
                                                                     monkeypatch):
    monkeypatch.delenv("POROS_DEV_SIMULATE")  # a real run: the cache and the network matter
    _hub_says(monkeypatch, "GatedRepoError")
    code, obj = _typo(home, capsys)
    assert (obj["error"]["code"], code) == ("E_HUB_AUTH", 5)
    assert "is gated" in obj["error"]["message"] and "hf auth login" in obj["error"]["message"]

    _hub_says(monkeypatch, "LocalEntryNotFoundError", offline=True)
    code, obj = _typo(home, capsys)
    assert (obj["error"]["code"], code) == ("E_NETWORK", 5)
    said = obj["error"]["message"]
    assert "HF_HUB_OFFLINE is set" in said and "unset HF_HUB_OFFLINE" in said

    _hub_says(monkeypatch, "LocalEntryNotFoundError", offline=False)
    code, obj = _typo(home, capsys)
    assert (obj["error"]["code"], code) == ("E_NETWORK", 5)
    assert "could not be reached" in obj["error"]["message"]
    assert "HF_HUB_OFFLINE" not in obj["error"]["message"]


def test_a_catalog_model_is_not_looked_up_as_the_words_resolve(monkeypatch):
    """The catalog names what is on the Hub: the verb asks the Hub about any other name only."""
    from poros.cli.context import Context
    from poros.cli.verbs import train
    from poros.core.refusal import Refusal

    ctx = Context(fixed_cached=frozenset(), fixed_interactive=False)
    asked = _hub_says(monkeypatch, "RepositoryNotFoundError")
    train.require_model(MODEL, ctx)
    assert asked == []
    with pytest.raises(Refusal) as refused:
        train.require_model(TYPO, ctx)
    assert refused.value.code == "E_NOT_FOUND" and asked == [TYPO]


def test_a_mac_and_a_preview_leave_the_cache_to_their_own_checks(monkeypatch):
    """Offline with nothing cached is a refusal where a load reads the Hugging Face cache; a
    Mac's weights can be in a store of its own, and a preview loads nothing."""
    from poros.preflight import command

    _hub_says(monkeypatch, "LocalEntryNotFoundError", offline=True)
    assert command.model_refusal(TYPO).code == "E_NETWORK"
    assert command.model_refusal(TYPO, cache_decides=False) is None
    _hub_says(monkeypatch, "RepositoryNotFoundError")  # the Hub's own answer: either way
    assert command.model_refusal(TYPO, cache_decides=False).code == "E_NOT_FOUND"


def test_a_config_names_a_model_that_is_not_there(home, capsys, monkeypatch):  # noqa: F811
    _hub_says(monkeypatch, "RepositoryNotFoundError")
    code, obj = _config(home, capsys, model_name_or_path=TYPO)
    err = obj["error"]
    assert (err["code"], code) == ("E_NOT_FOUND", 2), err
    assert "did you mean Qwen/Qwen2.5-0.5B-Instruct?" in err["message"]
    assert not (home / "out" / "events.jsonl").exists()
    code, obj = _config(home, capsys, model_name_or_path="./no-such-folder")
    assert (obj["error"]["code"], code) == ("E_NOT_FOUND", 2)
    assert "no model folder at ./no-such-folder" in obj["error"]["message"]


def test_a_load_that_meets_the_hub_late_says_it_without_the_librarys_words():
    """What the checks could not see (a file that 404s mid-load, the network dropping): the
    load's error still has its code and our sentence, never E_INTERNAL."""
    from poros.cli.errors import _HUB_SAID, _error_code

    missing = OSError("x/y is not a local folder and is not a valid model identifier listed on "
                      "'https://huggingface.co/models'\nIf this is a private repository, ...")
    offline = OSError("We couldn't connect to 'https://huggingface.co' to load the files, and "
                      "couldn't find them in the cached files.")
    gated = OSError("You are trying to access a gated repo.\nMake sure to have access to it.")
    assert [_error_code(e) for e in (missing, offline, gated)] == [
        "E_NOT_FOUND", "E_NETWORK", "E_HUB_AUTH"]
    wrapped = RuntimeError("load failed")
    wrapped.__cause__ = offline
    assert _error_code(wrapped) == "E_NETWORK"
    assert _error_code(FileNotFoundError(2, "No such file or directory", "x.json")) == "E_NOT_FOUND"
    assert _error_code(KeyError("a bug")) == "E_INTERNAL"
    assert all("huggingface.co/models" not in said for said in _HUB_SAID.values())


# ── a bug: said plainly, with where its traceback is ───────────────────────────────────────────


def _crash(monkeypatch, capsys, tmp_path, *, env=()):
    """A command that raises a bug, in a person's terminal mode: (exit code, stdout, stderr)."""
    from poros.cli import contract, errors

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POROS_OUTPUT", raising=False)
    monkeypatch.delenv("POROS_JOB", raising=False)
    for name, value in env:
        monkeypatch.setenv(name, value)
    (tmp_path / "home").mkdir(exist_ok=True)
    contract.reset()
    try:
        try:
            raise KeyError("a bug, with hf_ABCDEFGHIJKLMNOPQRSTUV in it")
        except KeyError as exc:
            errors._friendly_error(exc)
    except Exception as stop:  # typer.Exit
        code = stop.exit_code
    said = capsys.readouterr()
    return code, said.out, said.err


def test_a_bug_says_where_its_traceback_is_and_how_to_report_it(monkeypatch, capsys, tmp_path):
    code, out, err = _crash(monkeypatch, capsys, tmp_path)
    said = out + err  # an error's screen is stderr's in a one-shot command
    assert code == 1
    assert "Internal error" in said and "KeyError" in said
    assert "crash.log" in said and "github.com/Caistro-Labs/poros/issues" in said
    assert "POROS_TRACEBACK=1" in said
    assert "Traceback" not in said  # none on the screen unasked
    log = (tmp_path / "home" / "crash.log").read_text()
    assert "Traceback (most recent call last)" in log and "KeyError" in log
    assert "hf_ABCDEFGHIJKLMNOPQRSTUV" not in log + said  # nothing shaped like a key is kept


def test_poros_traceback_prints_it(monkeypatch, capsys, tmp_path):
    code, out, err = _crash(monkeypatch, capsys, tmp_path, env=[("POROS_TRACEBACK", "1")])
    assert code == 1 and "Traceback (most recent call last)" in err


# ── a failed run says why ──────────────────────────────────────────────────────────────────────


def test_a_failed_runs_page_says_why_and_offers_what_works(home, capsys, monkeypatch):  # noqa: F811
    """It read `error  exit_code 1` and offered `runs continue` with no checkpoint to continue."""
    from poros.cli.ui import simulate
    from poros.cli.views import run_screens
    from poros.core.errors import PorosOOMError

    def out_of_memory(self, *a, **k):
        raise PorosOOMError("GPU out of memory during blockwise training.", block_size=4)

    monkeypatch.setattr(simulate.SimTrainer, "train", out_of_memory)
    code, obj = say(["train", MODEL, "--data", "notes.jsonl", "--yes", "--steps", "3"], home,
                    capsys)
    assert (obj["error"]["code"], code) == ("E_OOM", 3)
    (run,) = runs.all_runs()
    assert run.state == "failed"
    assert run.error == {"exit_code": 3, "code": "E_OOM",
                         "message": "GPU out of memory during blockwise training."}
    assert run_screens.error_said(run.error) == (
        "GPU out of memory during blockwise training. (E_OOM, exit code 3)")
    assert run_screens.error_said({"exit_code": 1}) == "exit code 1: no reason was recorded"
    code, obj = say(["runs", run.id], home, capsys)
    page = "\n".join("".join(t for t, _ in line) for line in run_screens.run_page(
        obj["runs"][0], {"action": {"command": "poros train again"}}, 100))
    assert "GPU out of memory during blockwise training. (E_OOM, exit code 3)" in page
    assert "poros train again" in page and "runs continue" not in page


# ── a config file that is not one ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("make, said", [
    (lambda p: p.mkdir(), "is a folder, not a YAML config file"),
    (lambda p: p.write_bytes(b"\xff\xfemodel: x\n"), "is not UTF-8 text"),
    (lambda p: p.write_text(""), "is empty: a config names a model"),
    (lambda p: p.write_text("- a\n- b\n"), "holds a YAML list"),
    (lambda p: p.write_text("model_name_or_path: [unclosed\n"), "does not parse"),
])
def test_a_config_that_is_not_one_says_what_it_is(tmp_path, make, said):
    from typer.testing import CliRunner

    from poros.cli import main

    cfg = tmp_path / "run.yaml"
    make(cfg)
    res = CliRunner().invoke(main.app, ["train", str(cfg), "--json"])
    line = next(x for x in res.output.splitlines() if x.startswith('{"schema"'))
    err = json.loads(line)["error"]
    assert (err["code"], res.exit_code) == ("E_USAGE", 2), err
    assert said in err["message"] and "Errno" not in err["message"]


def test_a_file_is_no_checkpoint_to_resume_from(home, capsys):  # noqa: F811
    (home / "notes.txt").write_text("x")
    from typer.testing import CliRunner

    from poros.cli import main

    cfg = home / "run.yaml"
    cfg.write_text(f"model_name_or_path: {MODEL}\ndataset: {home / 'notes.jsonl'}\n"
                   f"output_dir: {home / 'out'}\n")
    res = CliRunner().invoke(main.app, ["train", str(cfg), "--resume-from",
                                        str(home / "notes.txt"), "--json"])
    capsys.readouterr()
    err = json.loads(next(x for x in res.output.splitlines() if x.startswith('{"schema"')))["error"]
    assert err["code"] == "E_NOT_FOUND" and "is a file, not a checkpoint folder" in err["message"]


# ── words that name nothing ────────────────────────────────────────────────────────────────────


def test_train_with_no_data_files_here_says_so(home, capsys):  # noqa: F811
    """`poros train <model>` in a folder with no data file was an IndexError (E_INTERNAL)."""
    (home / "notes.jsonl").unlink()
    code, obj = say(["train", MODEL], home, capsys)
    err = obj["error"]
    assert (err["code"], code) == ("E_USAGE", 2), err
    assert "train needs data" in err["message"] and "--data FILE" in err["message"]


@pytest.mark.parametrize("flag, needs", [("--data", "a file"), ("--output", "a folder"),
                                         ("--name", "a name"), ("--steps", "a number"),
                                         ("--seq-len", "a number"), ("--rank", "a number")])
def test_a_flag_with_no_value_is_refused_not_read_as_true(home, capsys, flag, needs):  # noqa: F811
    """`--output` alone trained into a folder named True, `--data` alone trained on "True", and
    `--steps` alone trained the default count."""
    argv = ["train", MODEL, "--yes", "--steps", "3"]
    argv += [flag] if flag == "--data" else ["--data", "notes.jsonl", flag]
    code, obj = say(argv, home, capsys)
    err = obj["error"]
    assert (err["code"], code) == ("E_USAGE", 2), err
    assert f"{flag} needs {needs} after it" in err["message"]
    assert runs.all_runs() == [] and not (home / "True").exists()


# ── words a person can act on ──────────────────────────────────────────────────────────────────


def test_an_output_folder_that_is_home_or_holds_links_says_which(tmp_path):
    """`--output /` said "/ holds this user's home folder, or links, and a run writes through no
    link; ... remove the links": two reasons in one sentence, and neither plainly."""
    from poros.preflight import command

    above = command.output_refusal("/")
    assert above.code == "E_CONFIG" and "links" not in above.reason
    assert "/ is your home folder, or a folder above it" in above.reason
    linked = tmp_path / "out"
    linked.mkdir()
    (linked / "a.txt").symlink_to(tmp_path / "elsewhere")
    said = command.output_refusal(str(linked))
    assert f"{linked} holds links (a.txt)" in said.reason and "home folder" not in said.reason
    assert said.context[0].startswith("remove them, or name a folder of its own")


def test_a_config_forms_unknown_flag_offers_the_nearest_one(tmp_path):
    """`poros train run.yaml --stepz 5` listed "Possible options: --help, --steps, --store" and
    then how to write a first config, with no nearest flag: the verb form's own sentence."""
    import subprocess
    import sys

    from tests.conftest import checkout_env

    cfg = tmp_path / "run.yaml"
    cfg.write_text("model_name_or_path: x\ndataset: synthetic\n")
    env = checkout_env({**os.environ, "POROS_HOME": str(tmp_path / "home")})
    env.pop("POROS_OUTPUT", None)

    def poros(*argv):
        return subprocess.run([sys.executable, "-c", "from poros.cli.main import cli; cli()",
                               *argv], capture_output=True, text=True, env=env,
                              stdin=subprocess.DEVNULL)

    asked = poros("train", str(cfg), "--stepz", "5", "--json")
    err = json.loads(asked.stdout)["error"]
    assert (err["code"], asked.returncode) == ("E_USAGE", 2)
    assert err["message"] == "train has no --stepz; did you mean --steps?"
    assert err["hint"] == "poros train --help"
    human = poros("train", str(cfg), "--stepz", "5")
    assert "train has no --stepz; did you mean --steps?" in human.stderr
    assert "poros init" not in human.stderr and "Possible options" not in human.stderr
    assert "poros train --help" in human.stderr


def test_the_note_under_a_next_step_names_its_command_in_the_callers_form(monkeypatch):
    """Agent-form output read `Continue with: poros agents export ...`, then `poros eval <id>
    --data FILE` in a person's form."""
    from poros.cli import contract

    note = "No rows were held out: `poros eval 1117-1124-9889 --data FILE` scores a file of yours."
    assert contract.as_agent(note) == note  # a person's own form
    monkeypatch.setenv("POROS_AGENT", "1")
    assert "`poros agents eval 1117-1124-9889 --data FILE`" in contract.as_agent(note)
