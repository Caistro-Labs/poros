"""`poros eval <run>`: a real tiny run, trained on the CPU engine, scored by the verb.

The engine the verb builds is the tiny PEFT Qwen2 the train-loop tests use, from the same seed,
so its adapter starts at zero as a run's does; the tokenizer is the byte-level fake with a chat
template and an end-of-turn token. No weights, no network, no GPU.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from poros import data as pdata
from poros import jobs, runs
from poros.cli import app, contract
from poros.cli.context import Context
from poros.cli.schemas import EvalOutput, EvalSide, PlanGpu
from poros.cli.verbs import evaluate, evaluate_pass
from tests.conftest import Tok

CARD = PlanGpu(name="RTX", total_gib=8.0, free_gib=8.0, source="nvidia-smi")


class Toks(Tok):
    """The fake, with what a real tokenizer says of its special tokens: 1 ends a turn."""

    all_special_ids = [0, 1, 2]

    def encode(self, s, add_special_tokens=True):
        return self.enc(s)


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


@pytest.fixture
def home(tmp_path, monkeypatch, cpu_only):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.delenv("POROS_DEV_SIMULATE", raising=False)
    monkeypatch.delenv("POROS_AGENT", raising=False)
    monkeypatch.setattr("poros.core.card.detect_gpu", lambda *a, **k: CARD)
    monkeypatch.setattr(jobs, "gpu_uuid", lambda: "GPU-x")
    monkeypatch.setattr(evaluate, "_ready", lambda cfg, ctx: None)  # the tiny engine is on CPU
    monkeypatch.setattr(evaluate_pass, "tokenizer", lambda cfg: Toks())
    (tmp_path / "home").mkdir()
    yield tmp_path
    contract.reset()


def _trainer(torch, outdir, **cfg):
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    torch.manual_seed(0)
    model, _ = _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj", "gate_proj"))
    tr = _make_trainer(torch, model, str(outdir), grad_accum=2, max_steps=cfg.pop("steps", 8))
    for k, v in cfg.items():
        setattr(tr.config, k, v)
    if "learning_rate" in cfg:  # the optimizer takes its rate when it is made
        tr._initialize_optimizer()
    return tr


def _chats(path: Path, n: int = 60) -> Path:
    rows = [{"messages": [{"role": "user", "content": f"say {i}"},
                          {"role": "assistant", "content": f"{i % 7} {i % 7}"}]} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _pairs(path: Path, n: int = 6) -> Path:
    rows = [{"prompt": f"pick {i}", "chosen": f"yes {i}", "rejected": f"no no {i}"}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _trained(torch, home: Path, data: Path, shape: str = "chats", steps: int = 8,
             group: bool = False) -> runs.Run:
    """A run as the train verb leaves it: its folder, its settings, its events, its adapter."""
    from poros import recipes
    from poros.cli.train_cmd import _write_resolved
    from poros.core.events import PorosEvent
    from poros.recipes import dpo, sft
    from poros.train.run import Run

    rid = runs.new_id()
    folder = runs.create(rid, {"model": "dummy", "data": str(data), "steps": steps})
    runs.update(folder, state="running", started_at=runs.now())
    tr = _trainer(torch, folder, steps=steps, learning_rate=1e-2, dataset=str(data))
    c = tr.config
    common = dict(length=c.max_seq_length, seed=c.seed,
                  batch_size=c.per_device_batch_size * c.gradient_accumulation_steps)
    run = Run.from_trainer(tr, Toks())
    if shape == "pairs":
        rows = pdata.pairs(str(data), Toks(), **common)
        tr._event_writer.emit(PorosEvent(event_type="setup", data={
            "recipe": "dpo", "data": "pairs", "rows": rows.rows}))
        dpo.main(run, rows, steps=steps, beta=0.5)
        tr.config.dpo_beta = 0.5
    else:
        built = pdata.chats(str(data), Toks(), **common)
        built.group_by_length = tr.config.group_by_length = group
        rows, held = built.holdout(tr.config.eval_split)
        assert rows.group_by_length is group  # the split keeps it
        tr._event_writer.emit(PorosEvent(event_type="setup", data={
            "recipe": "sft", "data": "chats", "rows": rows.rows}))
        sft.main(run, rows, steps=steps, eval_data=held)
    tr.save(str(folder))
    _write_resolved(tr.config, recipes.Choice("dpo" if shape == "pairs" else "sft", shape))
    runs.update(folder, state="succeeded", finished_at=runs.now())
    return next(r for r in runs.all_runs() if r.id == rid)


@pytest.fixture
def engine(ml, monkeypatch):
    """The verb's engine: the same tiny model from the same seed, its adapter at zero."""
    built = []

    def make(cfg, *where):  # the run's folder and a way to say a line: nothing to say here
        built.append(_trainer(ml, cfg.output_dir))
        return built[-1]

    monkeypatch.setattr(evaluate_pass, "engine", make)
    return built


def _say(argv, home, capsys):
    code = app.dispatch(argv, Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False))
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    return code, json.loads(out[-1])


def _files(folder: Path) -> dict:
    return {str(p.relative_to(folder)): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_the_base_is_the_runs_own_first_score_and_the_trained_its_last(ml, home, engine, capsys):
    """Bit for bit: the same rows, the same forward, the same mean the run took of them."""
    from poros.cli.verbs import runs_report

    r = _trained(ml, home, _chats(home / "chats.jsonl"))
    held = runs_report.held_out(r.output_dir)
    before = _files(Path(r.folder))
    code, out = _say(["eval", r.id], home, capsys)
    assert code == 0, out
    assert out["base"]["loss"] == held["eval_loss_start"]
    assert out["trained"]["loss"] == held["eval_loss"] < held["eval_loss_start"]
    assert (out["rows"], out["of"], out["rows_from"], out["shape"]) == (3, 3, "held_out", "chats")
    assert out["checkpoint"] == f"poros://{r.id}/weights/final"
    assert out["change"] == pytest.approx((held["eval_loss"] - held["eval_loss_start"])
                                          / held["eval_loss_start"])
    assert 0 <= out["improved"] <= 3 and out["metrics"]["eval_loss"] == out["trained"]["loss"]
    assert (out["turn_end"], out["answer_rows"]) == (1, 3)
    for side in ("base", "trained"):
        assert 0 <= out[side]["ends"] <= 3 and 0 < out[side]["end_p"] < 1
    assert out["verdict"].startswith(("Learned well", "Barely moved", "Held-out loss rose"))
    assert _files(Path(r.folder)) == before  # it wrote nothing into the run
    assert "say" not in json.dumps(out)  # no row's words
    assert len(engine) == 1  # one load: the base, then the run's adapter on it


def test_a_run_grouped_by_length_is_scored_as_it_scored_itself(ml, home, engine, capsys):
    """group_by_length reorders a step's rows; eval keeps the setting and still matches the run."""
    from poros.cli.verbs import runs_report

    r = _trained(ml, home, _chats(home / "chats.jsonl"), group=True)
    cfg, _ = evaluate_pass.settings(r, Path(r.output_dir))
    assert cfg.group_by_length is True
    held = runs_report.held_out(r.output_dir)
    code, out = _say(["eval", r.id], home, capsys)
    assert code == 0, out
    assert out["base"]["loss"] == held["eval_loss_start"]
    assert out["trained"]["loss"] == held["eval_loss"]


def test_the_loss_falls_before_the_answers_end_and_eval_tells_the_two_apart(ml, home, engine,
                                                                            capsys):
    """The lesson, on the tiny engine: 40 steps take a tenth off the held-out loss, which the card
    calls Learned well, and still no answer ends; by 120 every one does."""
    early = _trained(ml, home, _chats(home / "early.jsonl"), steps=40)
    code, out = _say(["eval", early.id], home, capsys)
    assert code == 0 and out["trained"]["loss"] < 0.95 * out["base"]["loss"], out
    assert out["base"]["ends"] == 0 and 2 * out["trained"]["ends"] < 3
    assert out["verdict"].startswith("Learned well · ends its answers ")
    assert out["warning"].startswith("It does not end its answers")
    later = _trained(ml, home, _chats(home / "later.jsonl"), steps=120)
    code, out = _say(["eval", later.id], home, capsys)
    assert out["trained"]["ends"] == 3 and out["warning"] is None, out
    assert out["trained"]["end_p"] > out["base"]["end_p"]


def test_a_checkpoint_is_scored_by_its_path_and_rows_are_capped(ml, home, engine, capsys):
    r = _trained(ml, home, _chats(home / "chats.jsonl"))
    _, final = _say(["eval", r.id], home, capsys)
    code, out = _say(["eval", f"poros://{r.id}/weights/8", "--rows", "2"], home, capsys)
    assert code == 0, out
    assert (out["rows"], out["of"], out["checkpoint"]) == (2, 3, f"poros://{r.id}/weights/8")
    code, out = _say(["eval", f"poros://{r.id}/weights/3"], home, capsys)
    assert code and out["error"]["code"] == "E_NOT_FOUND"
    assert out["error"]["hint"] == f"poros runs checkpoints {r.id}"
    assert final["trained"]["loss"] != final["base"]["loss"]


def test_a_changed_data_file_is_refused_and_data_scores_it_as_it_is(ml, home, engine, capsys):
    data = _chats(home / "chats.jsonl")
    r = _trained(ml, home, data)
    later = os.stat(data).st_mtime + 30
    os.utime(data, (later, later))  # the same rows, written to after the run began
    code, out = _say(["eval", r.id], home, capsys)
    assert out["error"]["code"] == "E_DATASET" and "after run" in out["error"]["message"]
    assert out["error"]["hint"] == f"poros eval {r.id} --data {data}"
    _chats(data, 80)  # more rows: another split
    code, out = _say(["eval", r.id], home, capsys)
    assert out["error"]["code"] == "E_DATASET"
    assert "now splits as 76 rows trained and 4 held out" in out["error"]["message"]
    assert "counted 57 rows trained and 3 held out" in out["error"]["message"]
    code, out = _say(["eval", r.id, "--data", str(data), "--rows", "5"], home, capsys)
    assert code == 0 and (out["rows_from"], out["rows"], out["of"]) == ("file", 5, 80), out


def test_data_of_another_shape_is_refused_before_anything_loads(ml, home, engine, capsys):
    r = _trained(ml, home, _chats(home / "chats.jsonl"))
    engine.clear()
    code, out = _say(["eval", r.id, "--data", str(_pairs(home / "p.jsonl"))], home, capsys)
    assert out["error"]["code"] == "E_USAGE" and "trained on chats" in out["error"]["message"]
    code, out = _say(["eval", r.id, "--data", str(home / "nope.jsonl")], home, capsys)
    assert out["error"]["code"] == "E_NOT_FOUND"
    code, out = _say(["eval", r.id, "--rows", "0"], home, capsys)
    assert out["error"]["code"] == "E_USAGE"
    assert engine == []


def test_pairs_say_how_often_the_chosen_answer_wins(ml, home, engine, capsys):
    """A pairs run holds none out: its own pairs, said to be; DPO's loss with the base as its
    reference, which is log 2 at the base."""
    r = _trained(ml, home, _pairs(home / "prefs.jsonl"), shape="pairs")
    code, out = _say(["eval", r.id], home, capsys)
    assert code == 0, out
    assert (out["rows_from"], out["shape"], out["rows"], out["metrics"]) == ("trained", "pairs", 6,
                                                                              {})
    assert out["base"]["loss"] == pytest.approx(math.log(2))
    assert out["trained"]["loss"] < out["base"]["loss"] and out["improved"] > 0
    assert 0 <= out["base"]["prefers"] <= 6 and 0 <= out["trained"]["prefers"] <= 6
    assert out["trained"]["margin"] > out["base"]["margin"]
    assert out["answer_rows"] == 6  # the chosen answers' ends


def test_an_agent_gets_only_what_its_rows_have(ml, home, engine, capsys, monkeypatch):
    r = _trained(ml, home, _chats(home / "chats.jsonl"))
    monkeypatch.setenv("POROS_AGENT", "1")
    code, out = _say(["eval", r.id], home, capsys)
    assert code == 0, out
    assert "adapter" not in out and "simulated" not in out and "support" not in out
    assert set(out["base"]) == {"loss", "ends", "end_p"}  # a chats run: no pairs numbers
    assert out["warning"].startswith("It does not end its answers")  # 8 steps teach no end
    assert out["next"] == [f"poros agents export {r.id} --output ./adapter-{r.id[-4:]}",
                           f"poros agents recipes save {r.id}"]


def test_a_classic_run_holds_nothing_out_and_says_what_scores(home, capsys):
    rid = runs.new_id()
    folder = runs.create(rid, {"model": "dummy", "data": "synthetic", "steps": 3})
    runs.update(folder, state="succeeded", started_at=runs.now())
    (folder / "adapter_model.safetensors").write_bytes(b"x")
    (folder / "config.resolved.json").write_text(json.dumps(
        {"model_name_or_path": "dummy", "dataset": "synthetic", "trained_with": "classic",
         "data_shape": None}))
    code, out = _say(["eval", rid], home, capsys)
    assert out["error"]["code"] == "E_USAGE" and "classic loop" in out["error"]["message"]
    assert out["error"]["hint"] == f"poros eval {rid} --data <file>"


def _out(ends: int, model_rows: int = 8, rows_from: str = "held_out") -> EvalOutput:
    return EvalOutput(adapter="", dataset="", metrics={}, rows_from=rows_from,
                      base=EvalSide(loss=2.4, ends=0), trained=EvalSide(loss=1.7, ends=ends),
                      answer_rows=model_rows)


def test_a_model_that_never_ends_its_answers_is_said_plainly_with_its_instruct_twin():
    """The lesson: a base model trained on chats lost 30% of its held-out loss and never said its
    <|im_end|>, so every reply ran to the limit. The loss alone said Learned well."""
    word, warn = evaluate.verdict(_out(0), "Qwen/Qwen2.5-0.5B")
    assert word == "Learned well · ends its answers 0 of 8"
    assert warn.startswith("It does not end its answers: in 8 of 8 rows")
    assert warn.endswith("Qwen2.5-0.5B is a base model, and Qwen2.5-0.5B-Instruct ends its answers")
    word, warn = evaluate.verdict(_out(0), "Qwen/Qwen2.5-0.5B-Instruct")
    assert warn and "Instruct ends" not in warn
    assert evaluate.verdict(_out(5), "Qwen/Qwen2.5-0.5B") == (
        "Learned well · ends its answers 5 of 8", None)
    rose = _out(8, rows_from="file")
    rose.trained.loss = 3.0
    assert evaluate.verdict(rose, "m")[0] == "Loss rose · ends its answers 8 of 8"


def test_the_person_view_says_both_and_the_warning(capsys):
    from poros.cli import views
    from poros.cli.ui.text import plain

    out = _out(0)
    out.run, out.rows, out.of, out.shape, out.change, out.improved = "1108-1530-7f3a", 8, 8, \
        "chats", -0.29, 8
    out.base.end_p, out.trained.end_p = 0.02, 0.05
    out.verdict, out.warning = evaluate.verdict(out, "Qwen/Qwen2.5-0.5B")
    text = "\n".join(plain(line) for line in views.evaluated(out, 100))
    assert "Scored run 1108-1530-7f3a  8 held-out rows" in text
    assert "2.400 base   1.700 trained   29% lower" in text
    assert "0 of 8 base   0 of 8 trained   p 0.02 to 0.05" in text
    assert "Learned well · ends its answers 0 of 8" in text and "✕ It does not end" in text


def test_ends_are_the_last_end_token_of_each_answer():
    from poros import ends

    assert ends._ends_in([5, 1, 9, 6, 1, 1], [1, 1, 0, 1, 1, 0], 1) == [1, 4]
    assert ends._ends_in([5, 6, 1, 7], [1, 1, 1, 1], 1) == [2]  # one answer, then text
    assert ends._ends_in([5, 6, 7], [1, 1, 1], 1) == []  # cut before its end
