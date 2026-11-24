"""`poros data`: what a file holds and what it trains; --convert writes chats."""

from __future__ import annotations

import json

import pytest

from poros import data as poros_data
from poros.cli import contract
from poros.cli.context import Context
from poros.core.refusal import Refusal
from tests.cli.test_verbs import RTX3090, run  # noqa: F401  (the fixture machine)


@pytest.fixture
def ctx(tmp_path):
    return Context(cwd=tmp_path, fixed_gpu=RTX3090, fixed_cached=frozenset(),
                   fixed_interactive=False)


@pytest.fixture(autouse=True)
def json_mode(monkeypatch):
    monkeypatch.setenv("POROS_OUTPUT", "json")
    yield
    contract.reset()


def test_chats_train_sft(ctx, tmp_path, capsys):
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    (tmp_path / "c.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    code, obj = run(["data", "c.jsonl"], ctx, capsys)
    assert code == 0 and obj["schema"] == "poros.data.v1"
    assert (obj["shape"], obj["rows"], obj["trains"]) == ("chats", 2, "sft")
    assert obj["command"] == "poros train <model> --data c.jsonl"


def test_pairs_train_dpo_and_a_bad_row_is_named(ctx, tmp_path, capsys):
    pair = json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"})
    (tmp_path / "p.jsonl").write_text(f"{pair}\nnot json\n{pair}\n")
    code, obj = run(["data", "p.jsonl"], ctx, capsys)
    assert (obj["shape"], obj["rows"], obj["trains"]) == ("pairs", 2, "dpo")
    assert obj["problems"] == ["row 2 is not JSON"]


@pytest.mark.parametrize("name, body", [
    ("t.csv", 'question,answer\n"Reset?","Hold it 10s."\n'),
    ("a.json", json.dumps([{"instruction": "Translate", "input": "hola", "output": "hello"}])),
    ("pc.jsonl", json.dumps({"prompt": "Reset?", "completion": "Hold it 10s."}) + "\n"),
])
def test_question_answer_rows_train_as_they_are_and_convert(ctx, tmp_path, capsys, name, body):
    (tmp_path / name).write_text(body)
    code, obj = run(["data", name], ctx, capsys)
    assert obj["shape"] == "convertible" and obj["trains"] == "sft"
    assert obj["command"] == f"poros train <model> --data {name}"
    assert poros_data.detect(tmp_path / name) == "chats"  # no converting first
    contract.reset()
    code, obj = run(["data", name, "--convert"], ctx, capsys)
    out = tmp_path / obj["converted"].rsplit("/", 1)[-1]
    assert code == 0 and out.name.endswith(".chats.jsonl") and obj["trains"] == "sft"
    assert poros_data.detect(out) == "chats"  # what train reads
    turns = json.loads(out.read_text().splitlines()[0])["messages"]
    assert [t["role"] for t in turns] == ["user", "assistant"]
    if name == "a.json":
        assert turns[0]["content"] == "Translate\n\nhola"  # Alpaca's input joins its instruction


def test_no_file_asks_which_from_the_ones_here(ctx, tmp_path, capsys):
    (tmp_path / "tickets.csv").write_text("question,answer\nq,a\n")
    code, obj = run(["data"], ctx, capsys)
    assert code == 2 and obj["schema"] == "poros.needs_input.v1"
    assert [c["value"] for c in obj["questions"][0]["choices"]] == ["tickets.csv"]


def test_a_missing_file_is_refused(ctx, capsys):
    code, obj = run(["data", "nope.jsonl"], ctx, capsys)
    assert code != 0 and obj["error"]["code"] == "E_DATASET_NOT_FOUND"


def test_a_csv_reads_as_chats_and_skips_rows_with_no_answer(tmp_path):
    (tmp_path / "t.csv").write_text('Question,Answer\n"Reset?","Hold it 10s."\n"Why?",""\n')
    rows = [poros_data._turns(r, n) for n, r in poros_data._rows(tmp_path / "t.csv")]
    assert rows[0][-1] == {"role": "assistant", "content": "Hold it 10s."} and rows[1] is None


def test_holdout_keeps_rows_out_by_seed_only_from_40_up():
    from poros.data import Data

    small = Data(list(range(39)), batch_size=4)
    assert small.holdout(0.05) == (small, None)
    big = Data(list(range(400)), batch_size=4, seed=7)
    train, held = big.holdout(0.05)
    assert (train.rows, held.rows) == (380, 20)
    assert set(train.items).isdisjoint(held.items)
    assert big.holdout(0.05)[1].items == held.items  # a resume holds out the same rows


def test_a_missing_eval_file_is_refused(ctx, tmp_path, capsys):
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    (tmp_path / "c.jsonl").write_text(json.dumps(row) + "\n")
    code, obj = run(["train", "Qwen/Qwen2.5-0.5B-Instruct", "--data", "c.jsonl",
                     "--eval", "nope.jsonl"], ctx, capsys)
    assert code != 0 and obj["error"]["code"] == "E_DATASET_NOT_FOUND"


def test_from_keeps_the_parent_lora_shape(tmp_path):
    from types import SimpleNamespace

    from poros.cli.verbs.train import TrainAction, _inherit, preview

    (tmp_path / "config.resolved.json").write_text(
        json.dumps({"lora_rank": 32, "lora_alpha": 64}))
    parent = SimpleNamespace(id="7f3a", output_dir=str(tmp_path))
    a = TrainAction(model="m", data="d")
    _inherit(a, parent, None)
    assert a.rank == 32 and a.config["lora_alpha"] == 64
    assert "lora_alpha=64" in a.command()
    assert any("lora_alpha" in str(row) and "64" in str(row) for row in preview(a, 100))
    with pytest.raises(Refusal):
        _inherit(TrainAction(model="m", data="d"), parent, 16)


def test_a_row_whose_answer_is_past_any_length_says_so(ctx, tmp_path, capsys):
    long = "word " * 5000
    rows = [{"messages": [{"role": "user", "content": long},
                          {"role": "assistant", "content": "a"}]},
            {"prompt": long, "chosen": "a", "rejected": "b"}]
    for i, row in enumerate(rows):
        (tmp_path / f"{i}.jsonl").write_text(json.dumps(row) + "\n")
        code, obj = run(["data", f"{i}.jsonl"], ctx, capsys)
        assert code == 0 and obj["problems"] == [
            "row 1: the answer starts past 4096 tokens, and is cut away"]
    short = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": long}]}
    (tmp_path / "s.jsonl").write_text(json.dumps(short) + "\n")
    _, obj = run(["data", "s.jsonl"], ctx, capsys)
    assert obj["problems"] == []  # a long answer is only shortened


# --fix: a clean copy, built from what Poros trains on alone


def _chat(q, a, **more):
    return json.dumps({"messages": [{"role": "user", "content": q},
                                    {"role": "assistant", "content": a}], **more})


def test_a_files_shape_is_most_rows_and_a_first_odd_row_is_the_one_named(ctx, tmp_path, capsys):
    pair = json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"})
    (tmp_path / "c.jsonl").write_text("\n".join([pair, _chat("q", "a"), _chat("r", "b")]) + "\n")
    code, obj = run(["data", "c.jsonl"], ctx, capsys)
    assert (obj["shape"], obj["rows"]) == ("chats", 2)
    assert obj["problems"] == ["row 1 is pairs, not chats"]
    contract.reset()
    code, obj = run(["data", "c.jsonl", "--fix"], ctx, capsys)
    assert code == 0 and obj["kept"] == 2 and obj["dropped"] == ["row 1 is pairs, not chats"]


def test_fix_keeps_pairs_as_pairs_with_only_their_three_fields(ctx, tmp_path, capsys):
    rows = [{"prompt": "q", "chosen": "a", "rejected": "b", "score": 7},
            {"prompt": [{"from": "human", "value": "q", "x": 1}],
             "chosen": {"role": "assistant", "content": "a", "name": "n"}, "rejected": "b"},
            {"prompt": "q", "chosen": " ", "rejected": "b"}]
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, obj = run(["data", "p.jsonl", "--fix"], ctx, capsys)
    assert code == 0 and obj["kept"] == 2
    assert obj["dropped"] == ["row 3 has an empty chosen answer"]
    assert obj["command"] == "poros train <model> --data p.clean.jsonl"
    out = [json.loads(line) for line in (tmp_path / "p.clean.jsonl").read_text().splitlines()]
    assert out[0] == {"prompt": "q", "chosen": "a", "rejected": "b"}
    assert out[1] == {"prompt": [{"role": "user", "content": "q"}],
                      "chosen": [{"role": "assistant", "content": "a"}], "rejected": "b"}
    assert poros_data.detect(tmp_path / "p.clean.jsonl") == "pairs"


def test_the_trainer_reads_every_row_of_the_copy(ctx, tmp_path, capsys):
    from poros.preflight import data as preflight_data

    rows = [_chat("q", "a", id=1), "nope", json.dumps({"conversations": [
        {"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo", "weight": 0}]}),
        json.dumps({"instruction": "Translate", "input": "hola", "output": "hello"}),
        json.dumps([1, 2]), json.dumps({"messages": "just text"})]
    (tmp_path / "c.jsonl").write_text("\n".join(rows) + "\n")
    code, obj = run(["data", "c.jsonl", "--fix"], ctx, capsys)
    assert obj["converted_rows"] == ["row 4 is convertible, not chats"]
    # row 6 is a chat with no assistant turn: the trainer keeps no such row, and neither does it
    assert obj["dropped"] == ["row 2 is not JSON", "row 5 is not an object",
                              "row 6 has no assistant turn; it is left out"]
    clean = tmp_path / "c.clean.jsonl"
    for n, row in poros_data._rows(clean):  # as the chats builder reads them, every one
        assert set(row) == {"messages"} and poros_data._turns(row, n)
        assert all(set(t) == {"role", "content"} for t in row["messages"])
    assert preflight_data.stops(clean) == [] and obj["kept"] == 3
    turns = json.loads(clean.read_text().splitlines()[2])["messages"]
    assert turns[0]["content"] == "Translate\n\nhola"  # Alpaca's input joins its instruction


@pytest.mark.parametrize("name, body, wrote", [
    ("a.json", json.dumps([{"messages": [{"role": "user", "content": "q"},
                                         {"role": "assistant", "content": "a"}]}, "x"]),
     "a.clean.json"),
    ("t.csv", "question,answer\nhi,yo\nhow,\n", "t.clean.jsonl"),
    ("x.jsonl", json.dumps({"text": "words"}) + "\n" + json.dumps({"text": 5}) + "\n",
     "x.clean.jsonl"),
])
def test_fix_writes_each_kind_of_file_as_train_reads_it(ctx, tmp_path, capsys, name, body, wrote):
    (tmp_path / name).write_text(body)
    code, obj = run(["data", name, "--fix"], ctx, capsys)
    assert code == 0 and obj["wrote"] == str(tmp_path / wrote) and obj["kept"] == 1
    assert len(obj["dropped"]) == 1
    assert poros_data.detect(tmp_path / wrote) in ("chats", "text")
    assert sum(1 for _ in poros_data._rows(tmp_path / wrote)) == 1


def test_fix_never_writes_through_a_link(ctx, tmp_path, capsys):
    (tmp_path / "c.jsonl").write_text(_chat("q", "a") + "\n")
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("keep me\n")
    (tmp_path / "c.clean.jsonl").symlink_to(elsewhere)
    code, obj = run(["data", "c.jsonl", "--fix"], ctx, capsys)
    assert code == 2 and "c.clean.jsonl already exists" in obj["error"]["message"]
    assert elsewhere.read_text() == "keep me\n"
    (tmp_path / "c.clean.jsonl").unlink()
    (tmp_path / "c.clean.jsonl").symlink_to(tmp_path / "gone.jsonl")  # a link to nothing
    contract.reset()
    code, obj = run(["data", "c.jsonl", "--fix"], ctx, capsys)
    assert code == 2 and not (tmp_path / "gone.jsonl").exists()
    from poros.cli.verbs import data as data_verb
    from poros.core.refusal import Refusal

    with pytest.raises(Refusal):  # the open itself, as when a link lands after the first look
        data_verb._write_clean(tmp_path / "c.jsonl", tmp_path / "c.clean.jsonl", "chats")
    assert not (tmp_path / "gone.jsonl").exists() and elsewhere.read_text() == "keep me\n"


@pytest.mark.parametrize("argv, said", [
    (["data", "c.jsonl", "-o", "x.jsonl"], "-o names where --fix writes"),
    (["data", "c.jsonl", "--fix", "--convert"], "--fix and --convert: say one"),
    (["data", "n.txt", "--fix"], "--fix cleans a .jsonl, .json or .csv file"),
    (["data", "c.jsonl", "--fix", "-o", "x.csv"], "-o names a .jsonl or .json file"),
])
def test_fix_is_refused_where_it_cannot_help(ctx, tmp_path, capsys, argv, said):
    (tmp_path / "c.jsonl").write_text(_chat("q", "a") + "\n")
    (tmp_path / "n.txt").write_text("words\n")
    code, obj = run(argv, ctx, capsys)
    assert code == 2 and said in obj["error"]["message"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.jsonl", "n.txt"]


@pytest.mark.parametrize("body", [json.dumps({"a": 1}) + "\n", "nope\n" + _chat("q", " ") + "\n"])
def test_fix_with_nothing_to_keep_leaves_no_file(ctx, tmp_path, capsys, body):
    (tmp_path / "c.jsonl").write_text(body)
    code, obj = run(["data", "c.jsonl", "--fix"], ctx, capsys)
    assert code == 6 and obj["error"]["code"] == "E_DATASET"
    assert not (tmp_path / "c.clean.jsonl").exists()


def test_lines_count_a_files_lines_and_rows_the_rows_that_train(ctx, tmp_path, capsys):
    (tmp_path / "t.csv").write_text("question,answer\nhi,yo\nhow,\nwhy,because")
    code, obj = run(["data", "t.csv"], ctx, capsys)
    assert (obj["rows"], obj["lines"]) == (2, 4)  # its header, and a row with no answer
    (tmp_path / "c.jsonl").write_text("\n" + _chat("q", "a") + "\n\nnope\n")
    contract.reset()
    code, obj = run(["data", "c.jsonl"], ctx, capsys)
    assert (obj["rows"], obj["lines"], obj["problems"]) == (1, 4, ["row 4 is not JSON"])


def test_messages_as_one_string_or_one_message_are_read_not_a_crash(ctx, tmp_path, capsys):
    rows = [{"messages": "hi"}, {"messages": {"role": "user", "content": "hi"}},
            json.loads(_chat("q", "a"))]
    (tmp_path / "c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, obj = run(["data", "c.jsonl"], ctx, capsys)
    # read as one user turn each: no assistant turn, so the trainer leaves both out
    assert code == 0 and (obj["shape"], obj["rows"]) == ("chats", 1)
    assert obj["samples"] == ["q → a"]
    assert obj["problems"] == ["row 1 has no assistant turn; it is left out",
                               "row 2 has no assistant turn; it is left out"]


def test_a_file_a_row_stops_says_it_does_not_train_as_it_is_and_issues_say_each_row(
    ctx, tmp_path, capsys
):
    """`data` said `rows: 41` of a file `train` refused: `trains_as_is` says which, and `issues`
    are `problems`' rows as data, from the same checks."""
    rows = [_chat("q", "a"), "{nope", _chat("e", " "),
            json.dumps({"question": "Reset?", "answer": "Hold it."}),
            json.dumps({"question": "No answer?", "answer": ""}), _chat("q2", "a2")]
    (tmp_path / "c.jsonl").write_text("\n".join(rows) + "\n")
    code, obj = run(["data", "c.jsonl"], ctx, capsys)
    assert code == 0 and obj["trains_as_is"] is False and obj["rows"] == 4, obj
    assert obj["problems"] == ["row 2 is not JSON",
                               "row 3 has an empty answer: it would train the model to say nothing",
                               "row 4 is convertible, not chats",
                               "row 5 has no answer; it is left out"]
    assert obj["issues"] == [
        {"row": 2, "kind": "fails", "why": "row 2 is not JSON"},
        {"row": 3, "kind": "empty_answer", "why": obj["problems"][1]},
        {"row": 4, "kind": "converted", "why": "row 4 is convertible, not chats"},
        {"row": 5, "kind": "dropped", "why": "row 5 has no answer; it is left out"}]
    assert obj["command"] == "poros data c.jsonl --fix"
    (tmp_path / "ok.jsonl").write_text("\n".join(rows[:1] + rows[2:]) + "\n")
    contract.reset()
    code, obj = run(["data", "ok.jsonl"], ctx, capsys)
    assert obj["trains_as_is"] is True and [i["kind"] for i in obj["issues"]] == [
        "empty_answer", "converted", "dropped"]
    long = {"messages": [{"role": "user", "content": "word " * 5000},
                         {"role": "assistant", "content": "a"}]}
    (tmp_path / "l.jsonl").write_text(json.dumps(long) + "\n" + _chat("q", "a") + "\n")
    contract.reset()
    _, obj = run(["data", "l.jsonl"], ctx, capsys)
    assert obj["issues"] == [{"row": 1, "kind": "cut", "why": obj["problems"][0]}]


def test_issues_list_the_first_rows_and_say_how_many_more(ctx, tmp_path, capsys):
    (tmp_path / "c.jsonl").write_text(_chat("q", "a") + "\n" + "nope\n" * 25)
    code, obj = run(["data", "c.jsonl"], ctx, capsys)
    assert len(obj["issues"]) == 20 and obj["issues_more"] == 5
    assert obj["issues"][-1] == {"row": 21, "kind": "fails", "why": "row 21 is not JSON"}


def test_a_question_answer_row_in_chats_trains_as_the_trainer_reads_it(ctx, tmp_path, capsys):
    """The trainer's chats builder reads a question/answer row as a chat, and a chat among them:
    neither stops a run, so `train` does not refuse the file. A pair among chats still does."""
    from poros.preflight import data as preflight_data

    qa = json.dumps({"question": "Reset?", "answer": "Hold it."})
    path = tmp_path / "mixed.jsonl"
    for main, other in ((_chat("q", "a"), qa), (qa, _chat("q", "a"))):
        path.write_text("\n".join([main, main, other]) + "\n")
        assert preflight_data.stops(path) == []
        said = [poros_data._turns(row, n)[-1]["content"] for n, row in poros_data._rows(path)]
        assert said == (["a", "a", "Hold it."] if main != qa else ["Hold it.", "Hold it.", "a"])
        assert poros_data.detect(path) == "chats"  # the builder that reads both
        contract.reset()
        _, obj = run(["data", "mixed.jsonl"], ctx, capsys)
        assert obj["trains_as_is"] is True and obj["rows"] == 3
        assert [i["kind"] for i in obj["issues"]] == ["converted"]
    pair = json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"})
    path.write_text("\n".join([_chat("q", "a"), _chat("q", "a"), pair]) + "\n")
    assert preflight_data.stops(path) == ["row 3 is pairs, not chats"]


def test_a_pair_whose_two_sides_are_the_same_is_said_and_fix_drops_it(ctx, tmp_path, capsys):
    """A prefs file where 15 of 80 pairs have chosen == rejected trains 16 steps at exactly
    ln 2. Such a pair is said, by its number alone, as
    `identical`; it still trains, as an empty answer does, and --fix drops it."""
    from poros.preflight import data as preflight_data

    rows = [{"prompt": "q", "chosen": "good", "rejected": "bad"},
            {"prompt": "q", "chosen": "secret words", "rejected": "secret words"},
            {"prompt": "q", "chosen": [{"role": "assistant", "content": "same"}],
             "rejected": "same"},  # a message and its text: the same turns to the trainer
            {"prompt": "q", "chosen": "Same", "rejected": "same"}]
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, obj = run(["data", "p.jsonl"], ctx, capsys)
    assert code == 0 and obj["rows"] == 4 and obj["trains_as_is"] is True
    assert obj["problems"] == [f"row {n}: chosen and rejected are the same: it teaches nothing"
                               for n in (2, 3)]
    assert obj["issues"] == [{"row": n, "kind": "identical", "why": obj["problems"][i]}
                             for i, n in enumerate((2, 3))]
    assert "secret" not in json.dumps({k: obj[k] for k in ("problems", "issues")})
    assert preflight_data.stops(tmp_path / "p.jsonl") == []  # train does not refuse it
    contract.reset()
    code, obj = run(["data", "p.jsonl", "--fix"], ctx, capsys)
    assert code == 0 and obj["kept"] == 2
    assert obj["dropped"] == [f"row {n}: chosen and rejected are the same: it teaches nothing"
                              for n in (2, 3)]
    kept = [json.loads(line) for line in (tmp_path / "p.clean.jsonl").read_text().splitlines()]
    assert [r["chosen"] for r in kept] == ["good", "Same"]
