"""The agent contract on a first run: a missing data file answered with other files to train
on, a `machine` with no reason, a guide that named fields and flags the commands do not have,
a message that stopped mid-sentence, error objects with empty fields, a chat with no answer left
unsaid, and `data`'s help and -o. No weights: the real entry point in a subprocess, the
simulator, and the verbs' own functions.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
from pathlib import Path

import pytest
import typer

import poros
from poros.cli import contract
from poros.cli.context import Context
from poros.cli.schemas import DoctorCheck, ErrorBody, PlanGpu, result_schema
from poros.cli.verbs import agents
from poros.cli.words import data as data_words
from poros.cli.words.grammar import parse
from poros.core.errors import EXIT_CODES
from poros.core.refusal import Refusal
from tests.cli.test_agents import _again, _agent, _exportable, _poros
from tests.conftest import Tok, machine_exit_here

MODEL = "qwen2.5-0.5b-instruct"
CHAT = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
CARD = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=None, source="spec")
GUIDE = " ".join((Path(poros.__file__).parent / "agents.md").read_text().split())


@pytest.fixture
def home(tmp_path, monkeypatch):
    for variable in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "me").mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    for name in ("notes.jsonl", "prefs.jsonl", "bad.jsonl"):
        (work / name).write_text((json.dumps(CHAT) + "\n") * 3)
    yield tmp_path
    contract.reset()


# A missing data file


def test_a_missing_data_file_is_an_error_that_offers_no_other_file(home):
    out = _agent("train", MODEL, "--data", "nope.jsonl", code=EXIT_CODES["E_DATASET_NOT_FOUND"])
    assert out["schema"] == "poros.error.v1" and out["error"]["code"] == "E_DATASET_NOT_FOUND"
    said = json.dumps(out)
    assert not any(name in said for name in ("notes.jsonl", "prefs.jsonl", "bad.jsonl"))
    # its hint asks which file to read, and each answer reads one: none trains
    assert out["error"]["hint"] == "poros agents data"
    asked = json.loads(_again(out["error"]["hint"]).stdout)
    assert asked["schema"] == "poros.needs_input.v1"
    commands = [c["command"] for q in asked["questions"] for c in q["choices"]]
    assert commands and all(c.startswith("poros agents data ") for c in commands), commands


@pytest.mark.parametrize("asked", ["note.jsonl", "notes.json", "notse.jsonl"])
def test_a_near_miss_is_named_and_its_hint_trains_nothing(home, asked):
    error = _agent("train", MODEL, "--data", asked, code=6)["error"]
    assert error["code"] == "E_DATASET_NOT_FOUND"
    assert error["message"] == (f"no file or folder named {asked}; the closest name here is "
                                "notes.jsonl")
    # the hint reads that file, and says what it holds: no command here trains on it unasked
    assert error["hint"] == "poros agents data notes.jsonl"
    read = _again(error["hint"])
    assert read.returncode == 0 and json.loads(read.stdout)["schema"] == "poros.data.v1"


def test_only_a_near_miss_of_the_name_is_one():
    here = ["notes.jsonl", "prefs.jsonl", "bad.jsonl", "data/"]
    assert data_words.near_misses("nope.jsonl", here) == []
    assert data_words.near_misses("x.jsonl", here) == []
    assert data_words.near_misses("note.jsonl", here) == ["notes.jsonl"]
    assert data_words.near_misses("notes.csv", here) == ["notes.jsonl"]
    assert data_words.near_misses("dat", here) == ["data/"]
    assert data_words.near_misses("a.jsonl", ["a.json", "a.csv", "b.jsonl"]) == ["a.csv", "a.json"]


def test_a_person_at_a_terminal_is_asked_about_a_near_miss_alone(home):
    from poros.cli.verbs import train

    def asked(data, interactive):
        ctx = Context(cwd=home / "work", fixed_gpu=CARD, fixed_cached=frozenset({MODEL}),
                      fixed_interactive=interactive)
        return train.resolve(parse(["train", MODEL, "--data", data]), ctx)

    [question] = asked("note.jsonl", True).questions
    assert [c.value for c in question.choices] == ["notes.jsonl"] and question.default is None
    for data, interactive in (("nope.jsonl", True), ("note.jsonl", False)):
        with pytest.raises(Refusal) as hit:  # nothing near, or no one to pick: the error
            asked(data, interactive)
        assert hit.value.code == "E_DATASET_NOT_FOUND"


# `machine` says why, and what to run


def test_machine_says_why_it_is_not_ready_and_what_to_run():
    from poros.cli.verbs import machine

    ok = [DoctorCheck(name="Python", value="3.12.1", status="ok")]
    assert machine.why("ready", ok, CARD) == ("", [])
    reason, to_run = machine.why("not ready", ok, None)
    assert "GPU" in reason or "Apple silicon" in reason
    assert to_run == ["poros recipes"]
    # a card and no torch: one install, said once, however many packages it brings
    missing = ok + [DoctorCheck(name=n, value="not installed", status="warn")
                    for n in ("torch", "peft")]
    reason, to_run = machine.why("ready with warnings", missing, CARD)
    assert reason == "the training stack needs an install: torch, peft"
    assert to_run == ["pip install poros-train"]
    # with no GPU that install trains nothing here: it is not offered
    assert machine.why("not ready", missing, None)[1] == ["poros recipes"]
    low = ok + [DoctorCheck(name="Disk free", value="3.0 GB", status="error"),
                DoctorCheck(name="CUBLAS_WORKSPACE_CONFIG", value="not set", status="warn")]
    assert machine.why("not ready", low, CARD) == (
        "Disk free: 3.0 GB; CUBLAS_WORKSPACE_CONFIG: not set", [])


def test_machines_result_carries_its_reason_and_its_next_runs(home):
    ready = _agent("machine", code=machine_exit_here())  # the simulated card
    assert "reason" in ready and isinstance(ready["next"], list)
    assert (ready["verdict"] == "ready") == (ready["reason"] == "")
    here = json.loads(_poros("agents", "machine", env={"POROS_DEV_SIMULATE": "0"}).stdout)
    assert (here["verdict"] == "ready") == (here["reason"] == "")
    if here["gpu"] is None:  # this machine has none: the reason says so, and its next runs
        assert here["verdict"] == "not ready" and "this machine has none" in here["reason"]
        listed = _again(here["next"][0])
        assert listed.returncode == 0 and json.loads(listed.stdout)["schema"] == "poros.recipes.v1"


# The guide and the commands agree


def test_model_list_has_evidence_and_support_is_a_plans(home):
    row = _agent("model", "list", "0.5b-instruct")["models"][0]
    assert "evidence" in row and "support" not in row


def test_a_word_that_is_no_catalog_name_is_not_found_and_says_what_a_model_is(home):
    error = _agent("train", "gpt2", "--data", "notes.jsonl", code=2)["error"]
    assert error["code"] == "E_NOT_FOUND" and "owner/name" in error["message"]
    assert error["hint"] == "poros agents model list"


def test_export_says_merged_one_way_in_its_help_and_takes_it_in_both_forms(home):
    page = _poros("agents", "export", "-h").stdout
    assert "[--merged]" in page and "    --merged " in page and "--format NAME" not in page
    assert "--format [table|json]" in page  # the one --format here: the output's
    _exportable(home)
    for form in (("export",), ("agents", "export")):
        merged = _poros(*form, "last", "--merged", "--json").stdout
        # read as --format merged: what stops it here is the merge's own need (torch, the base)
        assert "has no --merged" not in merged
        assert json.loads(merged).get("error", {}).get("code") != "E_USAGE", merged
    both = json.loads(_poros("export", "last", "--merged", "--format", "adapter", "--json").stdout)
    assert both["error"]["message"].startswith("--merged and --format adapter: say one")


# No message stops mid-sentence


# "with" is not among them: "the settings the checkpoint trained with" is a whole sentence
_OPEN = {"to", "the", "a", "an", "of", "and", "or", "for", "at", "by", "into", "than"}


def _unfinished(text: str) -> bool:
    """A message that stops before it is said: on a word that leads to another, or on a mark
    that does."""
    text = text.strip()
    return bool(text) and (text[-1] in ":;,(" or text.split()[-1].lower() in _OPEN)


def test_a_run_that_returns_at_once_names_the_flag_to_drop(home):
    proc = _poros("agents", "train", MODEL, "--data", "notes.jsonl", "--output", "./x",
                  env={"POROS_DEV_SIMULATE": "0"})
    error = json.loads(proc.stdout)["error"]
    if error["code"] == "E_NO_TORCH":  # the base install refuses before it reads the words
        pytest.skip("the base install refuses training first")
    assert error["message"] == ("a run that returns at once trains in its own folder, which its "
                                "result names; drop --output, or add --wait to train into that "
                                "folder")
    person = json.loads(_poros("train", MODEL, "--data", "notes.jsonl", "--output", "./x",
                               "--detach", "--json", env={"POROS_DEV_SIMULATE": "0"}).stdout)
    assert person["error"]["message"].endswith("drop --output, or drop --detach to train into "
                                               "that folder")


def _literals(node: ast.AST):
    """The text a call's string arguments end on: a literal, or an f-string's last literal part
    (None where it ends on a value)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.JoinedStr) and node.values and isinstance(
            node.values[-1], ast.Constant):
        yield str(node.values[-1].value)
    elif isinstance(node, ast.BinOp):
        yield from _literals(node.right)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for item in node.elts:
            yield from _literals(item)


def test_no_refusal_in_the_code_stops_mid_sentence():
    """Every refusal's reason, context line and fix, as it is written: none ends on a word or a
    mark that leads to more (`drop to`, `start it again with:`)."""
    found = []
    for path in sorted(Path(poros.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = getattr(getattr(node, "func", None), "id", None) or getattr(
                getattr(node, "func", None), "attr", "")
            if not isinstance(node, ast.Call) or name not in ("Refusal", "refuse", "Fix"):
                continue
            for arg in [*node.args[1:], *(k.value for k in node.keywords)]:
                found += [f"{path.name}:{arg.lineno}: {text!r}" for text in _literals(arg)
                          if " " in text.strip() and _unfinished(text)]
    assert not found, found


# An error says its next step and its docs, and no empty field


def _docs_open(ref: str) -> bool:
    """`poros docs <page> <words>` finds a page, and the section its words name."""
    from poros.cli.verbs import docs as verb

    page, *words = ref.removeprefix("poros agents docs ").removeprefix("poros docs ").split()
    files = verb._files()
    return page in files and (not words or verb._section(files[page], words) is not None)


def _whole(error: dict) -> dict:
    """An error object with nothing empty and nothing cut: a message that ends, a hint that is
    one line and no bare flag, docs that open."""
    assert {"code", "exit_code", "message"} <= error.keys(), error
    assert all(error[k] not in ("", {}, None) for k in error), error
    assert error["message"].strip() and not _unfinished(error["message"]), error
    if "hint" in error:
        assert "\n" not in error["hint"] and not error["hint"].startswith("-"), error
        if error["hint"].startswith(("poros ", "pip ")):
            shlex.split(error["hint"])  # a command a shell reads
    if "docs" in error:
        assert error["docs"].startswith("poros ") and _docs_open(error["docs"]), error
    return error


def _refused(refusal: Refusal, capsys) -> dict:
    """A refusal as its JSON error, through the CLI's own printer."""
    contract.force_json()
    capsys.readouterr()
    with pytest.raises(typer.Exit) as stop:
        contract.refused(refusal)
    error = json.loads(capsys.readouterr().out)["error"]
    assert stop.value.exit_code == error["exit_code"] == EXIT_CODES[refusal.code]
    contract.reset()
    return error


def test_an_error_leaves_out_what_it_has_none_of():
    bare = ErrorBody(code="E_INTERNAL", exit_code=1, message="RuntimeError")
    assert bare.model_dump() == {"code": "E_INTERNAL", "exit_code": 1, "message": "RuntimeError"}
    full = ErrorBody(code="E_OOM", exit_code=3, message="m", hint="h", remediation={"b": 2},
                     docs="poros docs limitations")
    assert set(full.model_dump()) == {"code", "exit_code", "message", "hint", "remediation",
                                      "docs"}
    body = result_schema("error")["$defs"]["ErrorBody"]  # and its schema still names each
    assert set(body["properties"]) == set(full.model_dump())
    assert body["required"] == ["code", "exit_code", "message"]


def test_every_docs_section_a_code_names_opens():
    assert contract.DOCS.keys() <= EXIT_CODES.keys()
    from poros.links import doc_ref

    for code, doc in contract.DOCS.items():
        assert _docs_open(doc_ref(doc)), (code, doc)


def test_the_refusals_a_first_run_meets_say_a_next_step_and_their_docs(home, capsys, monkeypatch):
    from poros.preflight import command
    from poros.preflight.resume import resume_clash

    # no torch, no CUDA: the command to run, and where the docs say what training needs
    no_torch = _whole(_refused(command.NO_TORCH, capsys))
    assert no_torch["hint"] == "pip install poros-train"
    assert no_torch["docs"] == "poros docs install install"
    no_cuda = _whole(_refused(command.NO_CUDA, capsys))
    assert (no_cuda["hint"], no_cuda["docs"]) == ("poros machine",
                                                  "poros docs install requirements")
    # an unsupported architecture, a download past the disk, a checkpoint that is another run's
    arch = _whole(_refused(command.not_trainable("org/m", "it needs its own code", "x"), capsys))
    assert arch["hint"] == "poros arch org/m" and arch["docs"] == (
        "poros docs models any other model")
    disk = _whole(_refused(command.disk_refusal(("1 GB", "", "a 9 GB download, 2 GB free")),
                           capsys))
    assert disk["code"] == "E_DISK" and disk["docs"] == "poros docs training refusals"
    assert "--force downloads anyway" in disk["message"] and "hint" not in disk
    saved = home / "checkpoint-4"
    saved.mkdir()
    (saved / "poros_config.json").write_text(json.dumps(
        {"model_name_or_path": "Qwen/Qwen2.5-0.5B", "quantization": "nf4", "lora_rank": 16,
         "seed": 42}))
    from poros.core.config import PorosConfig

    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", dataset="synthetic", seed=7)
    assert resume_clash(str(saved), cfg)
    resume = _whole(_refused(command.resume_refusal(str(saved), cfg), capsys))
    assert resume["code"] == "E_CONFIG" and resume["docs"] == (
        "poros docs training checkpoints and resume")
    # a yes it needs: the whole command again with the flag, never the flag alone
    monkeypatch.setattr(contract, "_argv", ["train", "t.yaml", "--json"])
    contract.force_json()
    capsys.readouterr()
    with pytest.raises(typer.Exit):
        contract.confirm("./out already holds a trained adapter; this run overwrites it.", False)
    yes = _whole(json.loads(capsys.readouterr().out)["error"])
    assert (yes["code"], yes["hint"]) == ("E_CONFIRM_REQUIRED", "poros train t.yaml --yes")
    assert yes["docs"] == "poros docs cli names and flags"


def test_the_same_refusals_from_the_commands_themselves(home):
    work = home / "work"
    # a missing data file, a model that is none, a folder that is not there
    for args, code, docs in (
            (("train", MODEL, "--data", "zz.csv"), "E_DATASET_NOT_FOUND",
             "poros agents docs training data"),
            (("train", "gpt2", "--data", "notes.jsonl"), "E_NOT_FOUND",
             "poros agents docs cli names and flags"),
            (("arch", "no-such-model"), "E_NOT_FOUND", "poros agents docs cli names and flags"),
            (("run", "info", "no-such-run"), "E_NOT_FOUND",
             "poros agents docs cli names and flags"),
            (("data", "zz.csv"), "E_DATASET_NOT_FOUND", "poros agents docs training data"),
            (("train", MODEL, "--data", "notes.jsonl", "--step", "3"), "E_USAGE",
             "poros agents docs cli names and flags")):
        error = _whole(_agent(*args, code=EXIT_CODES[code])["error"])
        assert (error["code"], error["docs"]) == (code, docs), error
        assert _docs_open(error["docs"]) and error["hint"].startswith("poros agents "), error
    # a config file's unknown key and wrong type: the command that prints every field
    (work / "bad.yaml").write_text("model_name_or_path: Qwen/Qwen2.5-0.5B\ndataset: notes.jsonl\n"
                                   "max_stepz: 3\nlora_rank: many\n")
    bad = _whole(json.loads(_poros("train", "bad.yaml", "--json").stdout)["error"])
    assert bad["code"] == "E_CONFIG" and "did you mean max_steps?" in bad["message"]
    assert (bad["hint"], bad["docs"]) == ("poros schema config", "poros docs cli config files")
    # a family no flag trains: its own code
    (work / "own").mkdir()
    (work / "own" / "config.json").write_text(json.dumps(
        {"model_type": "mything", "auto_map": {"AutoModelForCausalLM": "m.M"}}))
    arch = _whole(_agent("train", "./own", "--data", "notes.jsonl", code=4)["error"])
    assert arch["code"] == "E_ARCH_UNSUPPORTED" and arch["hint"].startswith("poros agents arch ")
    assert arch["docs"] == "poros agents docs models any other model"


# A chat with no assistant turn


def test_a_chat_with_no_assistant_turn_is_said_dropped_as_the_trainer_drops_it(home):
    from poros import data as poros_data
    from poros.preflight.data import _Checked

    odd = home / "work" / "odd.jsonl"
    rows = [CHAT] * 9 + [{"messages": [{"role": "user", "content": "only user"}]}] + [CHAT]
    odd.write_text("".join(json.dumps(row) + "\n" for row in rows))
    # what training does with it: the chats builder keeps no row with nothing to score
    built = poros_data.chats(odd, Tok(), length=64, batch_size=1)
    assert built.rows == 10 and 10 not in built.numbers
    assert _Checked(odd).rows == built.rows
    out = _agent("data", "odd.jsonl")
    assert (out["rows"], out["lines"], out["trains_as_is"]) == (10, 11, True)
    assert out["issues"] == [{"row": 10, "kind": "dropped",
                              "why": "row 10 has no assistant turn; it is left out"}]
    assert out["next"][-1] == "poros agents data odd.jsonl --fix"  # it trains; the fix drops it
    fixed = _agent("data", "odd.jsonl", "--fix")
    assert fixed["kept"] == 10 and fixed["dropped"] == [
        "row 10 has no assistant turn; it is left out"]
    kept = [json.loads(line) for line in Path(fixed["wrote"]).read_text().splitlines()]
    assert len(kept) == 10 and all(row == CHAT for row in kept)


# `data`'s help, and where -o cannot write


def test_each_flag_in_a_verbs_help_is_a_row_of_its_own():
    """A flag's line is its name, two spaces, what it does: one space runs it into the flag
    above (`-o, --output FILE where --fix writes` read as --fix's words)."""
    import importlib

    from poros.cli.app import BUILT

    for verb, module in BUILT.items():
        doc, flags = importlib.import_module(module).__doc__ or "", False
        for line in doc.splitlines():
            text = line.strip()
            if text == "Flags:":
                flags = True
            elif flags and not text:
                break
            elif flags and text.startswith("-"):  # as views.verb_help splits it
                assert text.partition("  ")[2].strip(), (verb, text)
    page = _poros("data", "--help").stdout
    assert re.search(r"-o, --output FILE +where --fix writes it instead", page)
    assert "never over a file -o" not in " ".join(page.split())


def test_where_o_cannot_write_is_said_whole_and_is_the_disks_refusal(home):
    target = home / "locked"
    target.mkdir()
    target.chmod(0o555)
    try:
        if (target / "probe").parent.stat().st_uid == 0:  # root writes anywhere
            pytest.skip("root")
        error = _whole(_agent("data", "notes.jsonl", "--fix", "-o", str(target / "fixed.jsonl"),
                              code=EXIT_CODES["E_DISK"])["error"])
    finally:
        target.chmod(0o755)
    assert error["code"] == "E_DISK"
    assert error["message"].startswith(f"{target}/fixed.jsonl: Permission denied")
    gone = _agent("data", "notes.jsonl", "--fix", "-o", str(home / "none" / "fixed.jsonl"),
                  code=2)["error"]
    assert gone["code"] == "E_USAGE" and gone["message"].startswith(f"{home}/none/fixed.jsonl: ")


# The guide says what the commands do


def test_the_guide_agrees_with_the_commands():
    """Each line a first-run agent read that the commands did not bear out, as it reads now."""
    for said in (
            # a result and its schema; an error's fields; machine's reason
            "each result validates against it, and a field outside its `required` is left out",
            "An error adds, where there is one: `hint`, the next step; `docs`",
            "a `hint` that begins `poros` or `pip` are commands to copy",
            "a verdict (not ready: exit 5) with its `reason` and `next`",
            # machine checks the machine; a bare word is no Hub id
            "the GPU, driver, host RAM, disk and installed stack checked",
            "or a folder: any other word is `E_NOT_FOUND` (exit 2)",
            # a data file that is not there, a chat with no answer
            "is `E_DATASET_NOT_FOUND` (exit 6), never a question",
            "(a chat with no assistant turn, a question with no answer)"):
        assert said in GUIDE, said
    assert "a measured run says fits" not in GUIDE and "- `support`:" not in GUIDE
    assert "`export <run> [--merged] [--output DIR]`" in GUIDE
    assert agents.COMMANDS["export"][0] == "<RUN>|<PATH> [--merged] [--output DIR]"
    assert len((Path(poros.__file__).parent / "agents.md").read_bytes()) < 22_000
