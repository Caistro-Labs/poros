"""`poros train` on text rows, through the engine's own loop, on the tiny CPU engine: not the
simulator. Only the hardware gates before the load are stood in for (no GPU here), and the base
is the tiny PEFT Qwen2 the train-loop tests use, from the same seed each time it loads.

Without a held-out file the run says it holds none out, trains every row and scores nothing.
With one (``eval_dataset``), it scores it before the first step, on the way and after the last,
and `poros eval` reads and scores that file as the loop did: its base loss is the run's first
score, its trained loss the run's last, bit for bit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poros import runs
from poros.cli import app, contract
from poros.cli.context import Context
from tests.conftest import Tok


class TextTok(Tok):
    """The byte-level fake, called as the engine's loop calls a tokenizer: a batch of texts, cut
    at ``max_length``, with no end token added."""

    pad_token, eos_token = "<pad>", "<eos>"

    def __call__(self, texts, truncation=True, max_length=None, **kw):
        one = isinstance(texts, str)
        ids = [self.enc(t)[:max_length] if truncation and max_length else self.enc(t)
               for t in ([texts] if one else texts)]
        if one:
            return {"input_ids": ids[0]}
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    return torch


@pytest.fixture
def home(ml, tmp_path, monkeypatch, cpu_only):
    """The tiny engine for every load, train's and eval's; the fake tokenizer for the loop's
    rows; no GPU gate (the engine is on the CPU)."""
    from poros.cli import train_cmd
    from poros.cli.verbs import evaluate, evaluate_pass
    from poros.train.trainer import PorosTrainer

    torch = ml
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POROS_DEV_SIMULATE", raising=False)
    monkeypatch.delenv("POROS_AGENT", raising=False)
    monkeypatch.delenv("POROS_OUTPUT", raising=False)
    (tmp_path / "home").mkdir()
    loads = []

    def tiny(cfg):
        from tests.test_train_loop_cpu import _tiny_peft_qwen2

        torch.manual_seed(0)
        model, _ = _tiny_peft_qwen2(torch, dropout=cfg.lora_dropout)
        cfg.blockwise_arch_check = "off"
        tr = PorosTrainer(cfg)
        tr.model, tr.device, tr._adapter_backend = model, torch.device("cpu"), "peft"
        tr._setup_blockwise_training()
        loads.append(tr)
        return tr

    monkeypatch.setattr(PorosTrainer, "from_pretrained",
                        classmethod(lambda cls, model, **kw: tiny(kw["config"])))
    monkeypatch.setattr("poros.core.tokenizer.load_tokenizer", lambda *a, **k: TextTok())

    def ready(cfg, **kw):  # what the checks before the load decide, on a machine with no GPU
        from poros import recipes

        choice = recipes.choose(cfg)
        train_cmd._write_resolved(cfg, choice)
        return train_cmd._Ready(choice, "qwen2", None, False, False, "CPU", 8.0, "")

    monkeypatch.setattr(train_cmd, "_before_the_load", ready)
    monkeypatch.setattr(train_cmd, "_download_note", lambda cfg: None)
    monkeypatch.setattr(train_cmd, "_checkpoint_note", lambda cfg, model_config: None)
    monkeypatch.setattr(evaluate, "_ready", lambda cfg, ctx: None)
    monkeypatch.setattr(evaluate_pass, "engine", lambda cfg, *where: tiny(cfg))
    yield tmp_path, loads
    contract.reset()


def _text(path: Path, n: int, seed: int = 0) -> Path:
    rows = [{"text": f"row {seed + i}: the {i % 5} sat on the {i % 3} mat " * 2} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _config(folder: Path, data: Path, **more):
    from poros.core.config import PorosConfig

    return PorosConfig(model_name_or_path="dummy", dataset=str(data), output_dir=str(folder),
                       block_size=2, deterministic=True, enable_prefetch=False,
                       blockwise_arch_check="off", per_device_batch_size=1,
                       gradient_accumulation_steps=2, max_steps=7, warmup_steps=0,
                       save_steps=2, eval_steps=3, lora_dropout=0.1, learning_rate=1e-2,
                       max_seq_length=32, logging_steps=1, **more)


def _run(cfg):
    from poros.cli import train_cmd

    def raise_it(exc):
        raise exc

    return train_cmd.run(cfg, config_path="run.yaml", resume_from=None, trust_remote_code=False,
                         friendly_error=raise_it)


def _events(folder: Path) -> list:
    return [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines()]


def _flat(text: str) -> str:
    return " ".join(text.replace("│", " ").split())


def test_a_text_run_with_no_held_out_file_says_so_and_trains_every_row(ml, home, capsys):
    tmp, loads = home
    folder = tmp / "plain"
    data = _text(tmp / "notes.jsonl", 60)  # 60 rows: a recipe's run would hold 3 out
    from poros.data import held_rows

    cfg = _config(folder, data)
    assert held_rows(cfg) == 3  # what the screen once said were held out
    metrics = _run(cfg)
    said = capsys.readouterr()
    shown = _flat(said.out + said.err)
    assert "rows held out" not in shown and "3 rows" not in shown
    assert "none held out this loop trains every row" in shown
    assert "give a file of them: eval_dataset, or --eval FILE" in shown
    assert "recipe sft (--recipe sft), which holds 5% of 40 rows or more out" in shown
    events = _events(folder)
    assert not [e for e in events if e["event_type"] == "eval"]
    assert not list(folder.glob("checkpoint-*/eval.json"))
    assert "eval_loss" not in metrics and "eval_loss_start" not in metrics
    resolved = json.loads((folder / "config.resolved.json").read_text())
    assert resolved["trained_with"] == "classic"
    # every row of the file trained: the loop's rows are the file's, none set aside
    assert len(loads[0]._prepare_dataloader(str(data), 32).dataset) == 60


def test_a_held_out_file_is_scored_as_the_config_says_and_eval_reads_it_as_the_loop_did(
        ml, home, capsys, monkeypatch):
    from poros.cli.verbs import runs_report

    tmp, loads = home
    rid = runs.new_id()
    folder = runs.create(rid, {"model": "dummy", "data": "notes.jsonl", "steps": 7})
    runs.update(folder, state="running", started_at=runs.now())
    data, held = _text(tmp / "notes.jsonl", 60), _text(tmp / "held.jsonl", 5, seed=100)
    monkeypatch.setenv("POROS_OUTPUT", "json")
    cfg = _config(folder, data, eval_dataset=str(held))
    written = cfg.model_dump(mode="json", exclude_unset=True)  # the config file, as written
    metrics = _run(cfg)
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    said = json.loads(out[-1])
    runs.update(folder, state="succeeded", finished_at=runs.now())

    evals = [e for e in _events(folder) if e["event_type"] == "eval"]
    # before the first step, every 3, each save (2, 4), after the last: once a step
    assert [e["step"] for e in evals] == [0, 2, 3, 4, 6]
    assert all(e["data"]["rows"] == 5 for e in evals)
    assert metrics["eval_loss_start"] == evals[0]["data"]["eval_loss"]
    assert metrics["eval_loss"] == evals[-1]["data"]["eval_loss"]
    assert said["metrics"]["eval_loss"] == metrics["eval_loss"]  # --json carries them
    assert said["metrics"]["eval_loss_start"] == metrics["eval_loss_start"]
    scores = {d.name: json.loads((d / "eval.json").read_text())
              for d in sorted(folder.glob("checkpoint-*")) if (d / "eval.json").exists()}
    assert scores["checkpoint-6"]["eval_loss"] == metrics["eval_loss"]  # the finished run's
    assert {"checkpoint-2", "checkpoint-4", "checkpoint-6"} <= set(scores)
    held_said = runs_report.held_out(str(folder))
    assert (held_said["eval_loss_start"], held_said["eval_loss"], held_said["eval_rows"]) == (
        metrics["eval_loss_start"], metrics["eval_loss"], 5)

    # `poros eval <run>`: the held-out file, read and scored as the loop read and scored it
    code = app.dispatch(["eval", rid], Context(cwd=tmp, fixed_cached=frozenset(),
                                               fixed_interactive=False))
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    scored = json.loads(out[-1])
    assert code == 0, scored
    assert scored["base"]["loss"] == metrics["eval_loss_start"]
    assert scored["trained"]["loss"] == metrics["eval_loss"]
    assert (scored["rows"], scored["of"], scored["rows_from"]) == (5, 5, "eval_file")
    assert scored["dataset"] == str(held)

    # `poros eval <config>`: the held-out file, not the training rows
    from poros.cli import main as cli

    cfg_file = tmp / "run.yaml"
    cfg_file.write_text(json.dumps(written))
    monkeypatch.setattr("poros.cli.root._require_torch", lambda *a, **k: None)
    from poros.cli import root as cli_root
    from tests.conftest import as_linux

    as_linux(monkeypatch, cli_root)
    cli.evaluate(config=cfg_file, adapter=None, trust_remote_code=False, json_output=True)
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    form = json.loads(out[-1])
    assert form["dataset"] == str(held)
    assert form["metrics"]["eval_loss"] == metrics["eval_loss"]
    assert form["metrics"]["scored"] == "every token of the held-out rows"


def test_a_run_on_the_engines_loop_with_no_held_out_file_is_refused_by_eval_in_words(
        ml, home, capsys, monkeypatch):
    tmp, _ = home
    rid = runs.new_id()
    folder = runs.create(rid, {"model": "dummy", "data": "notes.jsonl", "steps": 7})
    runs.update(folder, state="running", started_at=runs.now())
    _run(_config(folder, _text(tmp / "notes.jsonl", 60)))
    runs.update(folder, state="succeeded", finished_at=runs.now())
    capsys.readouterr()
    monkeypatch.setenv("POROS_OUTPUT", "json")
    code = app.dispatch(["eval", rid], Context(cwd=tmp, fixed_cached=frozenset(),
                                               fixed_interactive=False))
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    said = json.loads(out[-1])["error"]
    assert code == 2 and said["code"] == "E_USAGE"
    assert "trains every row and holds none out" in said["message"]
    assert said["hint"] == f"poros eval {rid} --data <file>"


def test_a_resumed_run_says_the_start_it_had_on_the_card_in_json_and_in_its_record(
        ml, home, capsys, monkeypatch):
    """Stopped after update 3 and continued: --json, the end card and the run's record all give
    the score of the model it started from, the uninterrupted run's, not the first one after."""
    import click

    from poros.cli import train_cmd
    from poros.cli.ui import train_view
    from poros.cli.verbs import runs_report

    tmp, _ = home
    data, held = _text(tmp / "notes.jsonl", 60), _text(tmp / "held.jsonl", 5, seed=100)
    monkeypatch.setenv("POROS_OUTPUT", "json")
    whole = _run(_config(tmp / "whole", data, eval_dataset=str(held)))
    capsys.readouterr()

    part = tmp / "part"
    cfg = _config(part, data, eval_dataset=str(held))
    asked = []  # once a step: the fourth time is after update 3

    def raise_it(exc):
        raise exc

    from poros.core.errors import PorosCancelledError

    with pytest.raises((click.exceptions.Exit, SystemExit, PorosCancelledError)):
        train_cmd.run(cfg, config_path="run.yaml", resume_from=None, trust_remote_code=False,
                      friendly_error=raise_it,
                      should_stop=lambda: asked.append(1) or len(asked) == 4)
    assert (part / "checkpoint-3").is_dir()
    capsys.readouterr()
    shown = {}
    real = train_view.TrainView.finish

    def finish(view, result, w):
        shown["start"], shown["end"] = result.eval_start, result.eval_end
        return real(view, result, w)

    monkeypatch.setattr(train_view.TrainView, "finish", finish)
    metrics = _run_from(_config(part, data, eval_dataset=str(held)), str(part / "checkpoint-3"))
    said = json.loads([x for x in capsys.readouterr().out.splitlines() if x.startswith("{")][-1])
    record = runs_report.held_out(str(part))
    assert metrics["eval_loss_start"] == whole["eval_loss_start"]
    assert said["metrics"]["eval_loss_start"] == whole["eval_loss_start"]
    assert shown["start"] == record["eval_loss_start"] == whole["eval_loss_start"]
    assert shown["end"] == said["metrics"]["eval_loss"] == record["eval_loss"] == whole["eval_loss"]


def _run_from(cfg, checkpoint):
    from poros.cli import train_cmd

    def raise_it(exc):
        raise exc

    return train_cmd.run(cfg, config_path="run.yaml", resume_from=checkpoint,
                         trust_remote_code=False, friendly_error=raise_it)
