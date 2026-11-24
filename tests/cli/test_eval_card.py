"""/eval as a card: the run, its weights, the rows and the answers to read are each a pick, and
the runs list puts the held-out loss beside the training loss."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from poros.cli import sheets
from poros.cli.context import Context
from poros.cli.ui import sheet as sh
from poros.cli.ui.style import resolve
from poros.cli.ui.text import render_line
from poros.cli.words.grammar import parse

ROOT = Path(__file__).resolve().parents[2]


def _poros(*args, cwd, env):
    code = "import sys; sys.argv[0] = 'poros'; from poros.cli.main import cli; sys.exit(cli())"
    return subprocess.run([sys.executable, "-c", code, *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Two previewed runs: one holding rows out of its data, one scoring an eval file."""
    import os

    home = tmp_path_factory.mktemp("eval-card")
    env = {**os.environ, "POROS_HOME": str(home / "home"), "POROS_DEV_SIMULATE": "1",
           "PYTHONPATH": str(ROOT)}
    (home / "home").mkdir()
    rows = [{"messages": [{"role": "user", "content": f"where is billing {i}?"},
                          {"role": "assistant", "content": "under Settings."}]} for i in range(400)]
    (home / "chats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (home / "held.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:50]))
    for extra in (["--steps", "40", "save_steps=20"], ["--steps", "40", "--eval", "held.jsonl"]):
        done = _poros("train", "Qwen/Qwen2.5-0.5B-Instruct", "--data", "chats.jsonl", *extra,
                      "--yes", cwd=home, env=env)
        assert done.returncode == 0, done.stdout + done.stderr
    return home, env


@pytest.fixture
def ctx(trained, monkeypatch):
    home, env = trained
    monkeypatch.setenv("POROS_HOME", env["POROS_HOME"])
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    return Context(cwd=home, fixed_interactive=True)


def _row(sheet, key):
    return next(f for f in sheet.view.fields if f.key == key)


def test_eval_opens_a_card_with_every_choice_a_pick(ctx):
    assert sheets.wanted("eval", parse(["eval"]))
    assert not sheets.wanted("eval", parse(["eval", "x"]))
    sheet = sh.Sheet(sheets.eval_form(parse(["eval"]), ctx))
    assert sheet.view.ready and _row(sheet, "subject").source == "the latest"
    rows = _row(sheet, "--data")
    assert rows.shown == "held.jsonl" and rows.fact == "50 chats"  # the newest scored its file
    weights = [o.label for o in _row(sheet, "--checkpoint").options()]
    assert weights[0] == "final"
    assert _row(sheet, "--show").shown == "3 answers"
    assert sheet.view.command.endswith("eval last --show 3")


def test_a_checkpoint_and_rows_picked_are_the_command(ctx):
    from poros import runs

    first = runs.all_runs()[-1]  # held rows out, saved every 20 (index 20: step 21)
    sheet = sh.Sheet(sheets.eval_form(parse(["eval", first.id]), ctx))
    assert _row(sheet, "--data").shown.endswith("held out")
    labels = [o.label for o in _row(sheet, "--checkpoint").options()]
    assert "best" in labels and "step 21" in labels
    sheet.sel = [f.key for f in sheet.view.fields].index("--checkpoint")
    sheet.key("enter")
    sheet.key("down")
    sheet.key("enter")  # best
    assert sheet.form.values["--checkpoint"] == "best"
    assert "--checkpoint best" in sheet.view.command and not sheet.view.error
    text = "\n".join(render_line(r, resolve(depth="plain")) for r in sheet.render(90))
    assert "the best" in text and all(len(r) <= 90 for r in text.splitlines())


def test_a_url_on_the_command_line_fills_the_weights_row(ctx):
    from poros import runs

    first = runs.all_runs()[-1]
    sheet = sh.Sheet(sheets.eval_form(parse(["eval", f"poros://{first.id}/weights/21"]), ctx))
    assert sheet.form.values["subject"] == first.id
    assert _row(sheet, "--checkpoint").shown == "step 21"


def test_show_writes_the_rows_words_for_a_person_and_none_for_an_agent(trained):
    home, env = trained
    said = _poros("eval", "last", "--show", "2", cwd=home, env=env)
    assert said.returncode == 0 and "where is billing" in said.stdout and "expected" in said.stdout
    data = json.loads(_poros("eval", "last", "--show", "2", "--json", cwd=home, env=env).stdout)
    assert "samples" not in data and "billing" not in json.dumps(data)
    refused = _poros("eval", "last", "--show", "50", cwd=home, env=env)
    assert refused.returncode != 0 and "at most" in refused.stdout + refused.stderr
    both = _poros("eval", "poros://last/weights/21", "--checkpoint", "best", cwd=home, env=env)
    assert both.returncode != 0


def test_the_runs_list_puts_held_out_beside_loss(ctx):
    L = sheets.runs_listing(ctx)
    head = "".join(t for t, _ in L.head)
    assert "held out" in head
    text = "\n".join(render_line(r, resolve(depth="plain")) for r in sh.Lister(L).render(110))
    assert "held out " in text  # the chosen run's detail: its held-out loss, start to end
