"""The CLI's recipe routing: which loop a config trains with, and the
recipe path run end to end on the tiny CPU engine."""

from __future__ import annotations

import json

import pytest

from poros import recipes
from poros.core.config import PorosConfig
from tests.test_run_api import Tok


def _cfg(tmp_path, rows, **kw):
    p = tmp_path / "data.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return PorosConfig(dataset=str(p), output_dir=str(tmp_path / "out"), **kw)


CHAT = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
PAIR = {"prompt": "q", "chosen": "good", "rejected": "bad"}


def test_auto_reads_the_data(tmp_path):
    assert recipes.choose(_cfg(tmp_path, [CHAT])) == recipes.Choice("sft", "chats")
    assert recipes.choose(_cfg(tmp_path, [PAIR])).name == "dpo"
    alpaca = [{"instruction": "x", "output": "y"}]  # question/answer rows are chats
    assert recipes.choose(_cfg(tmp_path, alpaca)) == recipes.Choice("sft", "chats")
    assert recipes.choose(PorosConfig(dataset="synthetic")).name == "classic"


def test_a_set_recipe_must_fit_the_data(tmp_path):
    with pytest.raises(ValueError, match="DPO needs preference pairs"):
        recipes.choose(_cfg(tmp_path, [CHAT], recipe="dpo"))
    with pytest.raises(ValueError, match="--recipe dpo"):
        recipes.choose(_cfg(tmp_path, [PAIR], recipe="sft"))
    with pytest.raises(ValueError, match="local .jsonl"):
        recipes.choose(PorosConfig(dataset="synthetic", recipe="sft"))
    assert recipes.choose(_cfg(tmp_path, [{"text": "x"}], recipe="sft")).shape == "text"


def test_a_loop_file(tmp_path):
    loop = tmp_path / "mine.py"
    loop.write_text("def main(run, data, *, steps=None, stop=None):\n    return {'ok': 1}\n")
    c = recipes.choose(_cfg(tmp_path, [CHAT], loop=str(loop)))
    assert c.name == str(loop) and c.label == "mine.py" and c.source == "set"
    assert recipes.load_loop(c.name)(None, None) == {"ok": 1}
    with pytest.raises(ValueError, match="does not exist"):
        recipes.choose(_cfg(tmp_path, [CHAT], loop=str(tmp_path / "nope.py")))


@pytest.mark.parametrize("rows,recipe", [([PAIR] * 4, "dpo"), ([CHAT] * 4, "sft")])
def test_the_recipe_path_trains(tmp_path, rows, recipe):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from poros.cli.train_recipe import train_recipe
    from tests.test_run_api import Tok
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    torch.manual_seed(0)
    model, _ = _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj"))
    tr = _make_trainer(torch, model, str(tmp_path / "out"), max_steps=4)
    tr._tokenizer = Tok()
    cfg = tr.config
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg.dataset, cfg.save_steps = str(p), 0
    cfg.per_device_batch_size, cfg.gradient_accumulation_steps = 1, 2
    choice = recipes.choose(cfg)
    assert choice.name == recipe
    metrics = train_recipe(tr, cfg, choice, None, None)
    assert metrics["total_steps"] == 4
    events = [json.loads(x) for x in (tmp_path / "out" / "events.jsonl").read_text().splitlines()]
    setup = next(e for e in events if e["event_type"] == "setup")
    assert setup["data"]["recipe"] == recipe and setup["data"]["rows"] == 4
    steps = [e for e in events if e["event_type"] == "step"]
    assert len(steps) == 4
    if recipe == "dpo":
        assert "accuracy" in steps[-1]["data"]


def test_a_pairs_run_counts_its_pairs_that_teach_nothing_and_trains_as_before(tmp_path,
                                                                              monkeypatch):
    """15 of 80 pairs have chosen == rejected. The setup event counts the pairs whose
    two sides train as the same tokens, after the length cut too, which `data` cannot see: it reads
    no tokenizer. Counting reads the rendered rows and changes no step."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from poros import data as pdata
    from poros.cli.train_recipe import train_recipe
    from tests.test_run_api import Tok
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    rows = [PAIR, {"prompt": "q", "chosen": "same", "rejected": "same"},
            {"prompt": "q", "chosen": "abcdefZ", "rejected": "abcdefY"}, PAIR]

    def trained(out: str) -> list:
        torch.manual_seed(0)
        model, _ = _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj"))
        tr = _make_trainer(torch, model, str(tmp_path / out), max_steps=4)
        tr._tokenizer = Tok()
        cfg = tr.config
        p = tmp_path / "d.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg.dataset, cfg.save_steps, cfg.max_seq_length = str(p), 0, 9  # cuts row 3's last letter
        cfg.per_device_batch_size, cfg.gradient_accumulation_steps = 1, 2
        train_recipe(tr, cfg, recipes.choose(cfg), None, None)
        return [json.loads(x) for x in (tmp_path / out / "events.jsonl").read_text().splitlines()]

    counted = trained("counted")
    setup = next(e for e in counted if e["event_type"] == "setup")["data"]
    assert (setup["rows"], setup["same_pairs"]) == (4, 2)
    monkeypatch.setattr(pdata, "same_pairs", lambda data: 0)  # as before: nothing counted
    before = trained("before")
    assert [e["data"]["loss"] for e in counted if e["event_type"] == "step"] == [
        e["data"]["loss"] for e in before if e["event_type"] == "step"]


def test_a_finished_run_trains_further(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from poros.cli.train_recipe import train_recipe
    from tests.test_run_api import Tok
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    def engine():
        torch.manual_seed(0)
        model, _ = _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj"))
        tr = _make_trainer(torch, model, str(tmp_path / "out"), max_steps=4)
        tr._tokenizer = Tok()
        p = tmp_path / "d.jsonl"
        p.write_text("".join(json.dumps(CHAT) + "\n" for _ in range(4)))
        tr.config.dataset, tr.config.save_steps = str(p), 0
        return tr

    first = engine()
    train_recipe(first, first.config, recipes.choose(first.config), None, None)
    assert (tmp_path / "out" / "checkpoint-3").is_dir()
    more = engine()
    more.config.max_steps = 6
    metrics = train_recipe(more, more.config, recipes.choose(more.config),
                           str(tmp_path / "out" / "checkpoint-3"), None)
    assert metrics["total_steps"] == 2  # this sitting's steps, as a resumed run counts them
    events = [json.loads(x) for x in (tmp_path / "out" / "events.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events if e["event_type"] == "step"][-2:] == [4, 5]
    assert (tmp_path / "out" / "checkpoint-5").is_dir()


def test_runs_continue_takes_more_and_asks_how_many(tmp_path, monkeypatch):
    from poros import runs as runs_mod
    from poros.cli.context import Context
    from poros.cli.verbs import runs as runs_verb
    from poros.cli.words.grammar import parse

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    d = runs_mod.create("1025-0000-abcd", {"model": "m", "name": "mine", "steps": 300})
    runs_mod.update(d, state="succeeded")
    draft = runs_verb.resolve(parse("runs continue 1025-0000-abcd --more 50"), Context())
    assert draft.action.more == 50
    assert draft.action.canonical()[-2:] == ["--more", "50"]
    ask = runs_verb.resolve(parse("runs continue 1025-0000-abcd --more"), Context()).questions[0]
    assert ask.flag == "--more" and ask.choices[0].label == "300 more"
    with pytest.raises(Exception, match="succeeded"):
        runs_verb._continue(runs_mod.read(d))


class _Ending(Tok):
    """The test tokenizer, its end-of-turn token a special token of its own, as a real one's is."""

    all_special_ids = [1]

    def encode(self, s, add_special_tokens=False):
        return self.enc(s)


def _engine(torch, out, rows, **cfg):
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    torch.manual_seed(0)
    model, _ = _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj"))
    tr = _make_trainer(torch, model, str(out), max_steps=5)
    tr._tokenizer = _Ending()
    p = out.parent / "d.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tr.config.dataset, tr.config.save_steps = str(p), 2
    for k, v in cfg.items():
        setattr(tr.config, k, v)
    return tr


def _said(out) -> list:
    return [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]


def test_a_chat_run_scores_each_checkpoint_and_says_whether_its_answers_end(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from poros.cli.train_recipe import train_recipe

    chats = [{"messages": [{"role": "user", "content": f"q{i}"},
                           {"role": "assistant", "content": f"a{i}"}]} for i in range(60)]
    tr = _engine(torch, tmp_path / "out", chats)
    metrics = train_recipe(tr, tr.config, recipes.choose(tr.config), None, None)
    evals = [e for e in _said(tmp_path / "out") if e["event_type"] == "eval"]
    assert [e["step"] for e in evals] == [0, 2, 4]
    assert all(e["data"]["answer_rows"] == 3 for e in evals)
    assert (metrics["eval_answer_rows"], metrics["eval_ends"]) == (3, evals[-1]["data"]["ends"])
    for step in (2, 4):
        said = json.loads((tmp_path / "out" / f"checkpoint-{step}" / "eval.json").read_text())
        assert said["answer_rows"] == 3 and said["rows"] == 3


def test_dpo_holds_pairs_out_and_a_run_begun_before_that_keeps_every_pair(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from poros.cli.train_recipe import train_recipe

    pairs = [{"prompt": f"q{i}", "chosen": f"yes {i}", "rejected": f"no no {i}"}
             for i in range(60)]
    (tmp_path / "new").mkdir()
    tr = _engine(torch, tmp_path / "new" / "out", pairs)
    metrics = train_recipe(tr, tr.config, recipes.choose(tr.config), None, None)
    events = _said(tmp_path / "new" / "out")
    assert next(e for e in events if e["event_type"] == "setup")["data"]["rows"] == 57
    evals = [e["data"] for e in events if e["event_type"] == "eval"]
    assert len(evals) == 3 and all(e["rows"] == 3 and "prefers" in e for e in evals)
    assert metrics["eval_prefers_start"] == evals[0]["prefers"]

    old = tmp_path / "old" / "out"  # its first start trained on all 60, as DPO did then
    old.mkdir(parents=True)
    (old / "events.jsonl").write_text(json.dumps(
        {"event_type": "setup", "data": {"recipe": "dpo", "rows": 60}}) + "\n")
    first = _engine(torch, old, pairs, eval_split=0.0, max_steps=3)
    train_recipe(first, first.config, recipes.choose(first.config), None, None)
    again = _engine(torch, old, pairs)
    train_recipe(again, again.config, recipes.choose(again.config),
                 str(old / "checkpoint-2"), None)
    setups = [e["data"]["rows"] for e in _said(old) if e["event_type"] == "setup"]
    assert setups == [60, 60, 60] and not [e for e in _said(old) if e["event_type"] == "eval"]
