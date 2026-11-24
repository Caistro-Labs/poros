"""Small things a run's screens got wrong: one result card for an export and a serve line with
nothing left to fill in, a labelled runs list that says what it lists, the model named whole,
a folder README that is the run's own, a card whose frame holds, a continued run's whole time.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _text(lines) -> str:
    return "\n".join("".join(t for t, st in line if st != "raw") for line in lines)


def test_an_adapter_exports_serve_line_names_its_base():
    from poros.cli.views import run_screens

    r = SimpleNamespace(format="adapter", run="1117-1124-9889", output="./adapter-9889",
                        fidelity="the adapter exactly as trained, bit for bit",
                        base_model="Qwen/Qwen2.5-0.5B-Instruct")
    said = _text(run_screens.exported(r, 120))
    assert ("vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-lora --lora-modules "
            "run=./adapter-9889") in said
    assert "<base>" not in said and said.count("Continue with") == 1


def test_a_merged_export_through_the_verb_shows_one_card():
    """The command's own "Exported merged weights" card and its `Continue with: vllm serve`
    came above the verb's card and its `Continue with: poros runs <id>`."""
    from poros.cli.commands import export as export_command

    shown = []
    t = SimpleNamespace(print=shown.append)
    export_command._EXPORT_RESULT_TO_CALLER = True
    try:
        export_command._show_done(t, [["a card"]])
    finally:
        export_command._EXPORT_RESULT_TO_CALLER = False
    assert shown == []  # the verb shows its own
    export_command._show_done(t, [["a card"]])
    assert shown == [[["a card"]]]  # `poros export <folder> -o <out>` keeps its card


def test_the_runs_list_has_a_header_and_says_what_it_lists():
    from poros.cli.views import run_screens

    run = {"id": "1117-1124-9889", "name": "qwen2.5-0.5b-instruct-chats-1117",
           "state": "succeeded", "step": 100, "total": 100, "loss": 1.17,
           "created_at": "2026-11-17T11:24:00+00:00", "started_at": "2026-11-17T11:24:01+00:00",
           "finished_at": "2026-11-17T11:27:05+00:00", "output_dir": None}
    lines = run_screens.runs(SimpleNamespace(do="list", runs=[run], more=0), 120)
    head, row = _text([lines[1]]), _text([lines[2]])
    for name in ("run", "steps", "loss", "held-out", "started", "took"):
        assert name in head
    # each name sits over its column: the loss column was the unlabelled one
    assert head.index("loss") + len("loss") == row.index("1.17") + len("1.17")
    assert head.index("steps") + len("steps") == row.index("100/100") + len("100/100")
    said = _text(lines)
    assert "poros train <model> --data" in said and "their own output_dir" in said
    narrow = run_screens.runs(SimpleNamespace(do="list", runs=[run], more=0), 64)
    assert c_width(narrow[1]) <= 64 and c_width(narrow[2]) <= 64
    empty = _text(run_screens.runs(SimpleNamespace(do="list", runs=[], more=0), 120))
    assert "No runs yet" in empty and "their own output_dir" in empty


def c_width(line) -> int:
    return sum(len(t) for t, st in line if st != "raw")


def test_the_compact_card_names_the_model_whole():
    from poros.cli.ui import agent_text

    assert agent_text._name("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen2.5-0.5B-Instruct"
    assert agent_text._name("Qwen/Qwen2.5-0.5B") == "Qwen2.5-0.5B"


def test_the_run_folders_readme_is_the_runs_own(tmp_path):
    """Each run folder's README.md was PEFT's template: "# Model Card for Model ID"."""
    from poros.cli import train_readme

    out = tmp_path / "1117-1124-9889"
    out.mkdir()
    (out / "adapter_model.safetensors").write_bytes(b"x")
    (out / "README.md").write_text(
        "---\nbase_model: Qwen/Qwen2.5-0.5B-Instruct\nlibrary_name: peft\n---\n\n"
        "# Model Card for Model ID\n\n[More Information Needed]\n")
    cfg = SimpleNamespace(output_dir=str(out), dataset="/data/chats.jsonl",
                          model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct", lora_rank=16,
                          lora_alpha=32, max_seq_length=256, quantization="nf4")
    train_readme.write(cfg, "SFT", 100, "0.3.0")
    said = (out / "README.md").read_text()
    assert said.startswith("---\nbase_model: Qwen/Qwen2.5-0.5B-Instruct\nlibrary_name: peft\n---\n")
    assert "Model Card for Model ID" not in said and "More Information Needed" not in said
    assert "# 1117-1124-9889" in said and "A LoRA adapter for `Qwen/Qwen2.5-0.5B-Instruct`" in said
    assert "| steps | 100 |" in said and "`chats.jsonl`" in said and "/data/" not in said
    # a folder with no adapter (a preview's) is left as it is, and a checkpoint is never touched
    bare = tmp_path / "bare"
    bare.mkdir()
    train_readme.write(SimpleNamespace(**{**vars(cfg), "output_dir": str(bare)}), "SFT", 1, "x")
    assert not (bare / "README.md").exists()


def test_the_trained_cards_frame_holds_with_a_long_axis_label():
    """A loss scale labelled 0.0002 is six columns where the axis had four: every row of the
    card ran two past the frame's right edge."""
    from poros.cli.ui import run_card as rc
    from poros.cli.ui.curve import Series

    def rows(values):
        series = Series(len(values)).extend(values)
        k = rc.Card(series=series, last=len(values) - 1, rows=rc.SHORT, first=values[0],
                    final=values[-1], meta="40 steps · SFT · 10s")
        return rc.card(k, 100, rc.FINAL)

    for values in ([2.9 - i * 0.04 for i in range(40)],
                   [0.0009 - i * 0.00002 for i in range(40)]):
        card = rows(values)
        widths = {c_width(line) for line in card if line}
        assert len(widths) == 1, [(_text([line]), c_width(line)) for line in card]
    assert any(len(v) > 4 for v in rc.Card(
        series=Series(40).extend([0.0009 - i * 0.00002 for i in range(40)]), last=39,
        rows=rc.SHORT).scale().labels().values())  # the case is the long label


def test_a_continued_runs_page_says_all_its_training_time(tmp_path):
    """It said `100 of 100 in 3m 04s`: the last stretch's wall time, as the receipt has it."""
    from poros.cli.verbs import runs_report
    from poros.cli.views import run_screens

    out = tmp_path / "run"
    out.mkdir()
    events = [{"event_type": "step", "step": i, "data": {"loss": 2.0, "s_per_step": 2.0}}
              for i in range(60)]
    events.append({"event_type": "resume", "step": 60, "data": {"checkpoint": "checkpoint-59"}})
    events += [{"event_type": "step", "step": i, "data": {"loss": 1.5, "s_per_step": 2.0}}
               for i in range(60, 100)]
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (out / "receipt.json").write_text(json.dumps({"wall_s": 84.0}))
    f = runs_report.facts(str(out), str(out))
    assert f["parts"] == 2
    x = {"id": "r", "name": "a run", "state": "succeeded", "step": 100, "total": 100,
         "output_dir": str(out), "created_at": None, "finished_at": None, "model": "m"}
    page = _text(run_screens.run_page(x, f, 70))
    assert "100 of 100" in page and "3m 20s of steps, over 2 stretches" in page
    one = {**f, "parts": 1}
    assert "in 1m 24s" in _text(run_screens.run_page(x, one, 70))


def test_the_python_api_says_where_it_saved():
    """The README's snippet trained into ./poros_output and printed nothing about it."""
    pytest.importorskip("torch")
    from poros.train.progress import _Progress

    said = []
    me = SimpleNamespace(config=SimpleNamespace(output_dir="./poros_output"),
                         _progress_enabled=lambda: True, _progress_print=said.append)
    metrics = {"total_steps": 3, "final_loss": 1.0, "total_time_s": 1.0, "avg_step_time_ms": 1.0,
               "tokens_processed": 10, "tokens_per_sec": 10.0, "peak_vram_allocated_gb": 0.0,
               "peak_vram_reserved_gb": 0.0}
    _Progress._log_final_summary(me, metrics)
    assert "Output folder:    ./poros_output" in said[0]



def test_a_pair_runs_accuracy_is_one_measure_over_one_named_window(tmp_path):
    """One DPO run read "acc 100% chosen ranked first" on its page (its last step) and "chosen
    over rejected on 33% of training pairs over its last 3 steps" to an agent."""
    from poros.cli.ui.screens import metric_rows
    from poros.cli.verbs import runs_report

    out = tmp_path / "run"
    out.mkdir()
    accs = [0.0, 0.0, 1.0]
    (out / "events.jsonl").write_text("".join(json.dumps(
        {"event_type": "step", "step": i, "data": {"loss": 0.6, "accuracy": a, "margin": 0.1}})
        + "\n" for i, a in enumerate(accs)))
    f = runs_report.facts(str(out), str(out))
    (steps, mean), pairs, _ = runs_report.preference(str(out))
    assert pairs and (steps, round(mean, 4)) == (3, 0.3333)
    assert (f["acc_over"], round(f["extra"]["accuracy"], 4)) == (3, 0.3333)  # the same window
    said = _text(metric_rows(f["extra"], 8, 16, f["acc_over"]))
    assert "33%" in said and "chosen ranked first over its last 3 steps" in said
    live = _text(metric_rows({"accuracy": 1.0}, 8, 16))  # while it trains: the step just taken
    assert "100%" in live and "over its last" not in live


def test_a_three_step_runs_axis_names_no_step_twice():
    """A three-step run's axis read `1 ... 1 ... 3`: its half is its first step."""
    from poros.cli.ui import run_card as rc
    from poros.cli.ui.curve import Series

    def axis(n):
        k = rc.Card(series=Series(n).extend([1.0] * n), last=n - 1, rows=rc.SHORT)
        return "".join(t for t, _ in rc._axis(k, 60)).split()

    assert axis(3) == ["1", "3"] and axis(2) == ["1", "2"]
    assert axis(9) == ["1", "4", "9"] and axis(100) == ["1", "50", "100"]
