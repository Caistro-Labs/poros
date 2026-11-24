"""``poros.Run``, ``poros.data``, ``poros.losses`` and the recipes, on CPU.

The engine is the tiny PEFT Qwen2 the train-loop tests build (no download,
no GPU); the tokenizer is a byte-level fake with a chat template, so the
rendering and answer-only weights are checked token by token.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from poros import data as pdata
from poros.train.batch import Batch, Datum
from tests.conftest import ROLE, Tok

CHAT = {"messages": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "yo"},
    {"role": "user", "content": "ok?"},
    {"role": "assistant", "content": "ok"},
]}


def _scored(d: Datum):
    return [t for t, w in zip(d.target_tokens, d.weights) if w]


# ── data, torch-free ───────────────────────────────────────────────────


def test_chats_score_the_answers_only():
    tok = Tok()
    d = pdata.chats([CHAT], tok).items[0]
    # each answer's content and its end token; never the role headers or the user
    assert _scored(d) == tok.enc("yo") + [1] + tok.enc("ok") + [1]


def test_chats_last_and_all():
    tok = Tok()
    last = pdata.chats([CHAT], tok, train_on="last").items[0]
    assert _scored(last) == tok.enc("ok") + [1]
    every = pdata.chats([CHAT], tok, train_on="all").items[0]
    assert all(every.weights)


def test_pairs_score_the_answer_and_batch_chosen_then_rejected():
    tok = Tok()
    rows = [{"prompt": "q", "chosen": "good", "rejected": "bad"},
            {"prompt": [{"role": "user", "content": "r"}],
             "chosen": {"role": "assistant", "content": "y"}, "rejected": "n"}]
    data = pdata.pairs(rows, tok, batch_size=2)
    c, r = data.items[0]
    assert _scored(c) == tok.enc("good") + [1] and _scored(r) == tok.enc("bad") + [1]
    b = next(data.ordered())
    assert b.pairs and b.ids == [0, 1, 0, 1] and b.datums[2] is r


LONG = "q" * 40


def test_a_row_with_its_answer_cut_away_is_skipped():
    """It would train on nothing and report a loss of 0.0."""
    tok, cut = Tok(), {"messages": [{"role": "user", "content": LONG},
                                    {"role": "assistant", "content": "ok"}]}
    data = pdata.chats([cut, CHAT], tok, length=16)
    assert data.rows == 1
    assert data.items[0].tokens == pdata.chats([CHAT], tok, length=16).items[0].tokens
    with pytest.raises(ValueError, match="no rows to train on"):
        pdata.chats([cut], tok, length=16)


def test_a_pair_with_either_answer_cut_away_is_skipped():
    class Plain(Tok):
        chat_template = None

    tok, kept = Plain(), {"prompt": "q", "chosen": "good", "rejected": "bad"}
    both = {"prompt": LONG, "chosen": "good", "rejected": "bad"}
    one = {"prompt": "q", "chosen": "good",
           "rejected": [{"role": "user", "content": LONG}, {"role": "assistant", "content": "n"}]}
    assert pdata.pairs([kept], tok, length=16).rows == 1
    assert pdata.pairs([both, one, kept], tok, length=16).rows == 1
    with pytest.raises(ValueError, match="no rows to train on"):
        pdata.pairs([both, one], tok, length=16)


def test_any_answer_left_after_the_cut_keeps_the_row_as_it_was():
    tok = Tok()
    whole = pdata.chats([CHAT], tok).items[0]
    first = whole.weights.index(1.0) + 1  # a Datum's position i predicts token i + 1
    kept = pdata.chats([CHAT], tok, length=first).items[0]
    assert sum(kept.weights) == 1.0
    assert kept.tokens == whole.tokens[:first] and kept.weights == whole.weights[:first]
    with pytest.raises(ValueError, match="no rows to train on"):
        pdata.chats([CHAT], tok, length=first - 1)


def test_a_row_too_short_to_read_is_still_a_bad_row_and_text_is_never_skipped():
    with pytest.raises(ValueError, match="row 1 renders to fewer than two tokens"):
        pdata.chats([CHAT], Tok(), length=0)
    assert pdata.text([{"text": "a b c"}], Tok(), length=1).rows == 1


def test_detect_names_the_shape_or_the_bad_row():
    assert pdata.detect([CHAT]) == "chats"
    assert pdata.detect([{"prompt": "q", "chosen": "a", "rejected": "b"}]) == "pairs"
    assert pdata.detect([{"text": "x"}]) == "text"
    with pytest.raises(ValueError, match="row 1 .*fields: foo"):
        pdata.detect([{"foo": 1}])


def test_bad_rows_name_their_row(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(CHAT) + "\n" + json.dumps({"text": "x"}) + "\n")
    with pytest.raises(ValueError, match="row 2 has no messages"):
        pdata.chats(p, Tok())


def test_batches_are_a_function_of_the_step():
    rows = [{"text": f"row {i}"} for i in range(10)]
    data = pdata.text(rows, Tok(), batch_size=3, seed=7)
    assert len(data) == 4
    again = pdata.text(rows, Tok(), batch_size=3, seed=7)
    assert [data.batch(i).ids for i in range(8)] == [again.batch(i).ids for i in range(8)]
    epoch0 = sorted(i for s in range(4) for i in data.batch(s).ids)
    assert epoch0 == list(range(10))
    assert data.batch(0).ids != data.batch(4).ids  # the next epoch reshuffles


def test_datum_checks_lengths():
    with pytest.raises(ValueError, match="weights has 1 positions"):
        Datum(tokens=[1, 2], target_tokens=[2, 3], weights=[1.0])


# ── the Run on the engine ──────────────────────────────────────────────


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _run(torch, outdir, **cfg):
    from poros.train.run import Run
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    torch.manual_seed(0)
    model, _ = _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj", "gate_proj"))
    tr = _make_trainer(torch, model, str(outdir), max_steps=cfg.pop("max_steps", 5))
    for k, v in cfg.items():
        setattr(tr.config, k, v)
    return Run.from_trainer(tr, Tok())


def _chats(n=8, batch_size=2):
    rows = [{"messages": [{"role": "user", "content": f"say {i}"},
                          {"role": "assistant", "content": f"{i} {i} {i}"}]} for i in range(n)]
    return pdata.chats(rows, Tok(), batch_size=batch_size)


def _events(run):
    lines = (run.dir / "events.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_cross_entropy_matches_the_engine_loss(ml, tmp_path):
    torch = ml
    run = _run(torch, tmp_path)
    seq = list(range(30, 46))
    batch = Batch([Datum.from_sequence(seq)])
    ours = run.forward(batch).loss
    ids = torch.tensor([seq])
    with torch.no_grad():
        legacy = run._engine._forward_eval_blockwise({"input_ids": ids, "labels": ids.clone()})
    assert math.isclose(ours, float(legacy), rel_tol=1e-5)


def test_sft_learns_writes_events_and_a_checkpoint(ml, tmp_path):
    from poros.recipes import sft

    run = _run(ml, tmp_path, learning_rate=1e-2, save_steps=0)
    data = _chats()
    first = run.forward(data.batch(0)).loss
    metrics = sft.main(run, data, steps=12)
    assert run.forward(data.batch(0)).loss < first
    steps = [e for e in _events(run) if e["event_type"] == "step"]
    assert [e["step"] for e in steps] == list(range(12))
    assert all("tokens" in e["data"] for e in steps)
    assert metrics["total_steps"] == 12
    assert (tmp_path / "checkpoint-11").is_dir()


def test_the_eval_before_the_first_step_outlives_it(ml, tmp_path):
    from poros.cli.train_recipe import with_eval
    from poros.recipes import sft

    run = _run(ml, tmp_path, learning_rate=1e-2, save_steps=0)
    metrics = sft.main(run, _chats(), steps=6, eval_data=_chats())
    evals = run._engine.history.evals
    assert [e["step"] for e in evals] == [0, 5]
    shown = with_eval(metrics, evals)
    assert shown["eval_loss_start"] == evals[0]["eval_loss"] != shown["eval_loss"]


def test_the_held_out_loss_is_the_same_rows_and_mask_before_and_after(ml, tmp_path):
    """On a real tiny run, from its events: rows are counted, scoring them is repeatable, and they
    are the rows the run never trained on."""
    from poros.cli.verbs import runs_report
    from poros.recipes import sft

    run = _run(ml, tmp_path, learning_rate=1e-2, save_steps=0)
    data, held = _chats(n=60, batch_size=2).holdout(0.05)
    assert (data.rows, held.rows) == (57, 3) and not set(map(id, data.items)) & set(
        map(id, held.items))
    assert run.evaluate(held)["eval_loss"] == run.evaluate(held)["eval_loss"]
    run._engine.history.evals.clear()
    (tmp_path / "events.jsonl").unlink()
    sft.main(run, data, steps=6, eval_data=held)
    found = runs_report.held_out(str(tmp_path))
    evals = [e for e in _events(run) if e["event_type"] == "eval"]
    assert [e["data"]["rows"] for e in evals] == [3, 3]
    assert found == {"eval_loss_start": evals[0]["data"]["eval_loss"],
                     "eval_loss": evals[1]["data"]["eval_loss"], "eval_rows": 3,
                     "eval_ends": None, "eval_answer_rows": None, "eval_prefers_start": None,
                     "eval_prefers": None}
    assert found["eval_loss"] < found["eval_loss_start"]
    assert set(run._engine.history.evals[0]) == {"step", "eval_loss", "eval_perplexity"}


def test_resume_is_bitwise(ml, tmp_path):
    from poros.core.errors import PorosCancelledError
    from poros.recipes import sft

    torch = ml
    data = _chats()
    whole = _run(torch, tmp_path / "a", learning_rate=1e-2, save_steps=0)
    sft.main(whole, data, steps=6)

    part = _run(torch, tmp_path / "b", learning_rate=1e-2, save_steps=0)
    with pytest.raises(PorosCancelledError) as stopped:
        sft.main(part, data, steps=6, stop=lambda: part.step == 3)
    assert stopped.value.step == 2

    resumed = _run(torch, tmp_path / "c", learning_rate=1e-2, save_steps=0)
    resumed.load(stopped.value.checkpoint)
    assert resumed.step == 3
    sft.main(resumed, data, steps=6)

    a = dict(whole._engine.model.named_parameters())
    for name, p in resumed._engine.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, a[name]), name


def test_dpo_starts_at_log2_and_learns(ml, tmp_path):
    from poros.recipes import dpo

    run = _run(ml, tmp_path, learning_rate=1e-2, save_steps=0)
    rows = [{"prompt": f"pick {i}", "chosen": f"yes {i}", "rejected": f"no no {i}"}
            for i in range(4)]
    data = pdata.pairs(rows, Tok(), batch_size=2)
    ref = run.reference(data)
    out = run.forward(data.batch(0), loss="dpo", ref=ref)
    assert math.isclose(out.loss, math.log(2), rel_tol=1e-4)
    assert out.metrics["accuracy"] == 0.0 and abs(out.metrics["margin"]) < 1e-5
    dpo.main(run, data, steps=8, beta=0.5)
    after = run.forward(data.batch(0), loss="dpo", ref=ref)
    assert after.loss < out.loss and after.metrics["accuracy"] > 0
    step = [e for e in _events(run) if e["event_type"] == "step"][-1]
    assert {"accuracy", "margin", "chosen_reward", "rejected_reward"} <= set(step["data"])
    assert (run.dir / "reference" / f"{data.fingerprint}.pt").exists()


def test_a_steps_tokens_are_what_it_trained_on_for_every_loss(ml, tmp_path):
    """A DPO step event carried the run's token count so far (its loss counts none), and a watch,
    which adds a recipe's steps up, read 3,687 then 5,332 tok/s. A step's `tokens` are the
    positions its losses scored: a pair's two answers, a chat's answers, every call of the step."""
    from poros.cli.ui.screens import TrainState
    from poros.cli.verbs import runs_live
    from poros.recipes import dpo

    run = _run(ml, tmp_path / "dpo", learning_rate=1e-2, save_steps=0)
    rows = [{"prompt": f"pick {i}", "chosen": f"yes {i}", "rejected": f"no no {i}"}
            for i in range(4)]
    data = pdata.pairs(rows, Tok(), batch_size=2)
    dpo.main(run, data, steps=6, beta=0.5)
    steps = [e["data"] for e in _events(run) if e["event_type"] == "step"]
    trained = [data.batch(i).trained for i in range(6)]
    assert [s["tokens"] for s in steps] == trained
    first = data.batch(0)
    answers = sum(w for d in first.datums for w in d.weights)  # chosen's and rejected's
    assert first.trained == answers < first.tokens  # no prompt, no padding
    with open(run.dir / "events.jsonl") as f:  # as a watch reads the run: a recipe's, by steps
        said = f.read()
    (run.dir / "events.jsonl").write_text(json.dumps(
        {"event_type": "setup", "data": {"recipe": "dpo", "data": "pairs"}}) + "\n" + said)
    events = runs_live._Events(TrainState(model="m", output_dir=str(run.dir)))
    events.read()
    assert events.tokens == sum(trained)
    assert events.seconds == pytest.approx(sum(s["s_per_step"] for s in steps))

    sft = _run(ml, tmp_path / "sft", learning_rate=1e-2, save_steps=0)
    chats = _chats()
    for step in sft.steps(2):  # a loop of your own: two calls to a step
        for half in (2 * step, 2 * step + 1):
            sft.forward_backward(chats.batch(half))
        sft.optim_step()
    said = [e["data"]["tokens"] for e in _events(sft) if e["event_type"] == "step"]
    assert said == [chats.batch(0).trained + chats.batch(1).trained,
                    chats.batch(2).trained + chats.batch(3).trained]


def test_a_custom_loss_drives_the_step_and_its_metrics(ml, tmp_path):
    from poros.train.batch import Loss

    run = _run(ml, tmp_path, per_device_batch_size=2)
    seen = {}

    def mine(lp, batch, *, scale):
        seen["shape"] = tuple(lp.shape)
        value = -lp.sum() * scale
        return Loss(value, {"mine": 7.0})

    data = _chats(4)
    for step in run.steps(2):
        run.forward_backward(data.batch(step), loss=mine, scale=0.5)
        run.optim_step()
    assert seen["shape"][0] == 2
    assert _events(run)[-2]["data"]["mine"] == 7.0  # the last step event, before teardown


def test_micro_batches_cost_time_not_the_answer(ml, tmp_path):
    torch = ml
    rows = [{"prompt": f"pick {i}", "chosen": f"yes {i}", "rejected": f"no no {i}"}
            for i in range(4)]
    data = pdata.pairs(rows, Tok(), batch_size=4)
    run = _run(torch, tmp_path, per_device_batch_size=4)
    ref = run.reference(data) - 0.5  # off zero, so the margins differ by pair
    whole = run.forward(data.batch(0), loss="dpo", ref=ref, beta=0.5)
    run.config.per_device_batch_size = 1
    sliced = run.forward(data.batch(0), loss="dpo", ref=ref, beta=0.5)
    assert math.isclose(whole.loss, sliced.loss, rel_tol=1e-5)
    assert torch.allclose(whole.logprobs, sliced.logprobs, atol=1e-5)
    assert math.isclose(whole.metrics["margin"], sliced.metrics["margin"], abs_tol=1e-5)


def test_the_loop_is_checked(ml, tmp_path):
    run = _run(ml, tmp_path)
    with pytest.raises(RuntimeError, match="no forward_backward"):
        run.optim_step()
    with pytest.raises(RuntimeError, match="ended without run.optim_step"):
        for _ in run.steps(2):
            pass
    with pytest.raises(ValueError, match="unknown loss 'nope'"):
        run.forward(_chats(2).batch(0), loss="nope")


def test_rl_losses(ml):
    torch = ml
    from poros import losses

    d = Datum(tokens=[5, 6, 7], target_tokens=[6, 7, 8],
              logprobs=[0.0, -1.0, -2.0], advantages=[0.0, 1.0, 1.0])
    b = Batch([d])
    lp = torch.tensor([[0.0, -1.0, -2.0]], requires_grad=True)
    out = losses.importance_sampling(lp, b)  # ratio 1: -sum(advantages)
    assert float(out.value) == -2.0
    out = losses.ppo(lp + torch.tensor([[0.0, 1.0, 0.0]]), b, clip=0.2)
    # position 1 ratio e (clipped to 1.2), position 2 ratio 1
    assert math.isclose(float(out.value), -(1.2 + 1.0), rel_tol=1e-6)
    assert math.isclose(out.metrics["clip_frac"], 1 / 3, rel_tol=1e-6)
    with pytest.raises(ValueError, match="needs Datum.advantages"):
        losses.ppo(lp, Batch([Datum(tokens=[1], target_tokens=[2], logprobs=[0.0])]))


def test_recipes_copy(tmp_path):
    from poros import recipes

    out = recipes.copy("dpo", tmp_path / "loop.py")
    assert "def main(" in out.read_text()
    with pytest.raises(FileExistsError):
        recipes.copy("dpo", out)


def test_init_adapter_starts_from_the_weights_with_a_fresh_optimizer(ml, tmp_path):
    from poros.recipes import sft

    torch = ml
    first = _run(torch, tmp_path / "a", learning_rate=1e-2, save_steps=0)
    sft.main(first, _chats(), steps=4)
    then = _run(torch, tmp_path / "b", learning_rate=1e-2, save_steps=0)
    then._engine.load_checkpoint(str(tmp_path / "a" / "checkpoint-3"),
                                 load_optimizer=False, load_rng=False)
    assert then.step == 0
    a = dict(first._engine.model.named_parameters())
    for name, p in then._engine.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, a[name]), name


@pytest.mark.parametrize("name", ["ipo", "kto", "orpo", "simpo"])
def test_each_pair_loop_learns_the_preference(ml, tmp_path, name):
    """IPO, KTO, ORPO and SimPO train the same pairs DPO does, through the same Run; a loop needs
    the reference only where its loss reads one."""
    import importlib

    from poros import losses

    run = _run(ml, tmp_path, learning_rate=1e-2, save_steps=0)
    rows = [{"prompt": f"pick {i}", "chosen": f"yes {i}", "rejected": f"no no {i}"}
            for i in range(4)]
    data = pdata.pairs(rows, Tok(), batch_size=2)
    beta = 2.0 if name == "simpo" else 0.5
    args = {"beta": beta, **({"ref": run.reference(data)} if name in losses.REFERENCE else {})}
    before = run.forward(data.batch(0), loss=name, **args)
    if name == "ipo":  # no margin yet: (0 - 1/(2 beta))^2
        assert math.isclose(before.loss, (1 / (2 * beta)) ** 2, rel_tol=1e-4)
    if name == "kto":  # every log ratio 0: 1 - sigmoid(0), chosen and rejected alike
        assert math.isclose(before.loss, 0.5, rel_tol=1e-4)
    importlib.import_module(f"poros.recipes.{name}").main(run, data, steps=8, beta=beta,
                                                           eval_data=data)
    after = run.forward(data.batch(0), loss=name, **args)
    assert after.loss < before.loss and after.metrics["accuracy"] > 0
    evals = [e["data"] for e in _events(run) if e["event_type"] == "eval"]
    assert len(evals) >= 2 and "prefers" in evals[-1]
    assert (run.dir / "reference").exists() == (name in losses.REFERENCE)


def test_eval_show_answers_greedily_and_says_each_rows_own_words(ml, tmp_path):
    """--show: each held-out row's prompt answered a token at a time through the run's forward,
    stopping at the end of a turn, and the row's question and answer read back from its file,
    plain: no escape reaches the terminal."""
    from poros.cli.verbs import evaluate_pass as ep

    run = _run(ml, tmp_path, save_steps=0)
    rows = [{"messages": [{"role": "user", "content": f"q{i} \x1b[31mred"},
                          {"role": "assistant", "content": f"a{i}"}]} for i in range(6)]
    path = tmp_path / "chats.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = pdata.chats(str(path), Tok(), batch_size=2)
    train, held = data.holdout(0.0, 2)
    assert held.rows == 2 and sorted(train.numbers + held.numbers) == list(range(1, 7))
    scored = ep.Rows(held, "chats", "held_out", 2, str(path))
    asked = ep.prompts(scored, 2)
    assert [a.number for a in asked] == held.numbers
    for a, item in zip(asked, held.items):  # the prompt is the row up to its answer
        assert a.prompt[-2:] == [2, ROLE["assistant"]] and len(a.prompt) < len(item.tokens) + 1
    said = ep.answers(run._engine, asked, end=1)
    assert len(said) == 2 and all(len(s) <= ep.MOST_NEW and 1 not in s for s in said)

    class Words(Tok):
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids) + "\x1b]0;x\x07"

    out = ep.samples(Words(), asked, said, said)
    for x, n in zip(out, held.numbers):
        assert x.prompt == f"q{n - 1} [31mred" and x.expected == f"a{n - 1}"
        assert "\x1b" not in x.base + x.trained and "\x07" not in x.trained


def test_a_save_before_the_first_step_resumes_at_step_0(ml, tmp_path):
    """Run.save() before any optim_step writes no checkpoint-0 (the name of the state after step
    0, which loads as a resume at step 1): an unnamed save then is refused,
    and a named one keeps the starting state, which resumes at step 0 and trains as a fresh run."""
    from poros.recipes import sft

    run = _run(ml, tmp_path / "a", learning_rate=1e-2, save_steps=0)
    with pytest.raises(ValueError, match="before the first optim_step"):
        run.save()
    assert not list((tmp_path / "a").glob("checkpoint-*"))
    start = run.save("start")
    fresh = sft.main(_run(ml, tmp_path / "b", learning_rate=1e-2, save_steps=0), _chats(),
                     steps=4)
    resumed = _run(ml, tmp_path / "c", learning_rate=1e-2, save_steps=0)
    resumed.load(start)
    assert resumed.step == 0
    again = sft.main(resumed, _chats(), steps=4)
    assert again["loss_history"] == fresh["loss_history"]


def test_a_new_run_among_another_runs_checkpoints_is_refused_before_the_load(ml, tmp_path,
                                                                             monkeypatch):
    """Run.create into a folder another run left a checkpoint-N in refuses before the load, not
    at steps() after it; Run.open continues that run."""
    from poros.core import checkpoints
    from poros.core.config import PorosConfig
    from poros.core.errors import PorosConfigError
    from poros.train import run as run_api

    saved = tmp_path / "checkpoint-4"
    saved.mkdir()
    for marker, said in ((checkpoints.FORMAT_MARKER, "1"), (checkpoints.COMPLETE_MARKER, "4"),
                         (checkpoints.RUN_MARKER, "b" * 16)):
        (saved / marker).write_text(said)
    (saved / "poros_config.json").write_text(
        PorosConfig(model_name_or_path="Org/Model", output_dir=str(tmp_path)).model_dump_json())
    loads = []
    monkeypatch.setattr(run_api.PorosTrainer, "from_pretrained",
                        staticmethod(lambda model, **k: loads.append(model) or SimpleNamespace(
                            config=k["config"], _dataset_record=None)))
    with pytest.raises(PorosConfigError, match="checkpoint-4"):
        run_api.Run.create("Org/Model", output_dir=str(tmp_path), tokenizer=Tok())
    assert loads == []
    monkeypatch.setattr(run_api.Run, "load", lambda self, ckpt: None)
    run_api.Run.open(tmp_path, tokenizer=Tok())
    assert loads == ["Org/Model"]
