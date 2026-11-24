"""The words layer: grammar, model/data/run resolution, Questions (no machine)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
import yaml

from poros.cli import questions
from poros.cli.schemas import Choice, Question
from poros.cli.words import data as data_words
from poros.cli.words import models as model_words
from poros.cli.words import runs as run_words
from poros.cli.words.grammar import parse
from poros.core.refusal import Refusal

SENTENCES = yaml.safe_load((Path(__file__).parent / "sentences.yaml").read_text())


@pytest.mark.parametrize("case", SENTENCES, ids=[c["say"] for c in SENTENCES])
def test_sentence_grammar(case):
    s = parse(case["say"])
    want = case["grammar"]
    assert s.verb == want.get("verb")
    assert s.action == want.get("action")
    assert s.subject == want.get("subject", [])
    assert {k: c.value for k, c in s.clauses.items()} == want.get("clauses", {})
    assert s.flags == want.get("flags", {})


@pytest.mark.parametrize(
    "case",
    [c for c in SENTENCES if "model" in c],
    ids=[c["say"] for c in SENTENCES if "model" in c],
)
def test_sentence_model(case):
    assert model_words.by_name(parse(case["say"]).subject[0]).id == case["model"]


def test_flag_wins_over_key_value():
    s = parse("train Qwen2.5-7B steps=300 --steps 50")
    assert s.value("steps") == "50"


def test_only_the_first_word_is_a_verb():
    assert parse("can i export last").verb is None
    assert parse("will Qwen2.5-7B fit").verb is None
    assert parse("/runs stop last").action == "stop"
    assert parse("models stop").subject == ["stop"]  # an action word only after its verb


def test_a_model_is_named_exactly():
    from poros.cli.context import Context
    from poros.cli.verbs import shared

    ctx = Context(fixed_cached=frozenset())
    assert shared.resolve_model(["qwen2.5-7b"], ctx, verb="train")[0] == "Qwen/Qwen2.5-7B"
    for words in (["qwen", "7b"], ["qwen7b"], ["72b"]):  # part of a name: which one, never a guess
        model, _, ask = shared.resolve_model(words, ctx, verb="train")
        assert model is None and ask.id == "model" and len(ask.choices) >= 2
    with pytest.raises(Refusal) as r:
        shared.resolve_model(["qwen-9b"], ctx, verb="train")
    assert "Qwen" in r.value.context[0]


def test_cache_only_ranks():
    m = model_words.resolve(["qwen", "7b"], cached=["Qwen/Qwen2.5-7B-Instruct"])
    assert [e.id for e in m.entries] == ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B"]
    assert model_words.resolve(["llama", "8b"], cached=["meta-llama/Llama-3.1-8B"]).entries == []


def test_exact_ids():
    m = model_words.resolve(["meta-llama/Llama-3.1-8B"])
    assert m.unique == "meta-llama/Llama-3.1-8B" and m.entries == []


def test_local_path_never_becomes_a_hub_id(tmp_path):
    assert model_words.looks_local("./org/model")
    assert not model_words.looks_like_hub_id("./org/model")
    miss = data_words.resolve("./nope.jsonl", tmp_path)
    assert isinstance(miss, data_words.DataMiss)


def test_data_resolution(tmp_path):
    (tmp_path / "notes.jsonl").write_text("{}\n")
    (tmp_path / "folder").mkdir()
    assert data_words.resolve("notes.jsonl", tmp_path).kind == "file"
    assert data_words.resolve("synthetic", tmp_path).kind == "synthetic"
    assert data_words.resolve("OpenAssistant/oasst1", tmp_path).kind == "hub"
    empty = data_words.resolve("folder", tmp_path)
    assert isinstance(empty, data_words.DataMiss) and "no .jsonl" in empty.reason
    near = data_words.resolve("note.jsonl", tmp_path)
    assert isinstance(near, data_words.DataMiss) and near.near == ["notes.jsonl"]


def test_run_resolution():
    ids = ["7f3a91c2", "7f3b0000", "a1b2c3d4"]
    assert run_words.resolve("a1", ids) == "a1b2c3d4"
    assert run_words.resolve("last", ids, newest="a1b2c3d4") == "a1b2c3d4"
    amb = run_words.resolve("7f3", ids)
    assert isinstance(amb, run_words.RunMiss) and len(amb.candidates) == 2
    assert isinstance(run_words.resolve("last", []), run_words.RunMiss)


def _q(**kw) -> Question:
    base = dict(
        id="model",
        prompt="Which Qwen2.5 7B?",
        default="Qwen/Qwen2.5-7B-Instruct",
        choices=[
            Choice(value="Qwen/Qwen2.5-7B-Instruct", label="Instruct"),
            Choice(value="Qwen/Qwen2.5-7B", label="base"),
        ],
    )
    return Question(**(base | kw))


def test_answering_is_editing_the_command():
    s = parse("train --data notes.jsonl")
    q = questions.commands_for(_q(), s)
    assert q.choices[0].command == "poros train Qwen/Qwen2.5-7B-Instruct --data notes.jsonl"
    again = parse(shlex.split(q.choices[1].command)[1:])
    assert again.subject == ["Qwen/Qwen2.5-7B"] and again.value("data") == "notes.jsonl"


def test_a_flag_answer_replaces_the_flag():
    s = parse("train Qwen2.5-7B --data notes.jsonl --steps 30")
    q = questions.commands_for(
        _q(id="steps", flag="--steps", default=None, choices=[Choice(value="60", label="60")]), s
    )
    assert q.choices[0].command == "poros train Qwen2.5-7B --data notes.jsonl --steps 60"


@pytest.mark.parametrize("flag", ["--yes", "--force=true", "--trust-remote-code", "--detach"])
def test_no_answer_adds_an_override_flag(flag):
    s = parse("train Qwen2.5-7B --data notes.jsonl")
    bad = _q(
        id="steps",
        flag="--steps",
        choices=[Choice(value="1", label="x", command=f"poros train x {flag}")],
    )
    with pytest.raises(RuntimeError):
        questions.commands_for(bad, s)


def test_a_confirmation_is_one_flag_never_preselected():
    s = parse("train Qwen2.5-7B --data notes.jsonl")
    yes = Choice(value="--yes", label="Yes")
    q = questions.commands_for(questions.confirmation("start", "Start?", "--yes", "Yes"), s)
    assert q.choices[0].command.endswith("--yes") and q.default is None
    for bad in (
        _q(id="start", confirm=True, default="--yes", choices=[yes]),
        _q(id="start", confirm=True, default=None,
           choices=[yes, Choice(value="--force", label="f")]),
        _q(id="start", confirm=True, default=None, choices=[Choice(value="--steps", label="s")]),
    ):
        with pytest.raises(RuntimeError):
            questions.commands_for(bad, s)


def test_confirmations_each_say_what_they_confirm_and_one_command_answers_them_all():
    """An agent got three questions, two of them `confirm`, each command adding one flag."""
    s = parse("train someone/model --data notes.jsonl")
    asked = [questions.confirmation("unvalidated", "u", "--unvalidated", "u"),
             _q(id="check", default=None, flag="",
                choices=[Choice(value="arch try", label="c", command="poros arch try x")]),
             questions.confirmation("download", "d", "--download", "d")]
    qs = questions.with_all([questions.commands_for(q, s) for q in asked], s)
    assert [q.id for q in qs] == ["unvalidated", "check", "download", "all"]
    (every,) = qs[-1].choices
    assert qs[-1].default is None and every.command == (
        "poros train someone/model --data notes.jsonl --unvalidated --download")
    one = [questions.commands_for(asked[0], s)]
    assert questions.with_all(one, s) == one  # one confirmation is its own command
    twice = [questions.commands_for(questions.confirmation("download", "d", f, "d"), s)
             for f in ("--download", "--force")]
    with pytest.raises(RuntimeError):
        questions.with_all(twice, s)


def test_canonical_round_trip():
    for case in SENTENCES:
        s = parse(case["say"])
        again = parse(s.canonical())
        assert again.verb == (s.verb or "models") and again.subject == s.subject
        for name in s.clauses:
            assert again.value(name) == str(s.value(name)) or again.value(name) == s.value(name)


def test_preferences_never_apply_without_a_tty(tmp_path, monkeypatch):
    import tomllib

    monkeypatch.setenv("POROS_HOME", str(tmp_path))
    questions.remember("variant", "instruct")
    questions.remember("export.format", 'C:\\Users\\me "x"')
    prefs = tomllib.loads((tmp_path / "preferences.toml").read_text())
    assert prefs == {"variant": "instruct", "export.format": 'C:\\Users\\me "x"'}
    assert questions.preference("variant") is None  # pytest has no TTY


def test_needs_input_json(monkeypatch, capsys):
    import typer

    from poros.cli import contract

    monkeypatch.setenv("POROS_OUTPUT", "json")
    q = questions.commands_for(_q(), parse("train Qwen2.5-7B --data notes.jsonl"))
    with pytest.raises(typer.Exit) as exc:
        questions.needs_input("train", [q])
    assert exc.value.exit_code == 2
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "poros.needs_input.v1"
    assert out["questions"][0]["choices"][0]["command"].endswith("--json")
    contract.reset()


def test_the_sentence_table_in_docs_is_current():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, str(root / "scripts" / "gen_cli_words.py"), "--check"])
    assert proc.returncode == 0, "run: python scripts/gen_cli_words.py"
