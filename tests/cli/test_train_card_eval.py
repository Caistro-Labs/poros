"""The train card sets what a run scores, when, and for how long, by picking: no flag typed."""

from __future__ import annotations

import json

import pytest

import poros.plan.configs
from poros.cli import sheets
from poros.cli.context import Context
from poros.cli.schemas import Figure, PlanGpu
from poros.cli.ui.sheet import Sheet
from poros.cli.ui.style import resolve
from poros.cli.ui.text import render_line
from poros.cli.words.grammar import parse
from poros.data import held_count


def _chats(path, n, said="hi"):
    path.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"{said} {i}"}, {"role": "assistant", "content": "yo"}]}) + "\n"
        for i in range(n)))


@pytest.fixture
def card(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="offline in tests"))
    _chats(tmp_path / "chats.jsonl", 1240)
    _chats(tmp_path / "held.jsonl", 80, "other")
    (tmp_path / "pairs.jsonl").write_text(
        '{"prompt": "q", "chosen": "a", "rejected": "b"}\n' * 300)
    gpu = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=None, source="spec")
    ctx = Context(cwd=tmp_path, fixed_gpu=gpu, fixed_cached=frozenset({"Qwen/Qwen2.5-7B-Instruct"}),
                  fixed_interactive=True)

    def make(data="chats.jsonl", *more):
        return Sheet(sheets.train_form(parse(["train", "Qwen2.5-7B-Instruct", "--data", data,
                                              *more]), ctx))

    return make


def _row(sheet, key):
    return next(f for f in sheet.view.fields if f.key == key)


def _pick(sheet, key, typed="", down=0):
    sheet.sel = [f.key for f in sheet.view.fields].index(key)
    sheet.key("enter")
    for ch in typed:
        sheet.key("char", ch)
    for _ in range(down):
        sheet.key("down")
    sheet.key("enter")


def test_the_card_says_what_it_holds_out_and_scores_at_each_checkpoint(card):
    sheet = card()
    held = _row(sheet, "eval")
    assert held.shown == "62 chats" and "5%" in held.source  # the real count, not a share
    saves = _row(sheet, "save_steps=")
    assert "scored: loss + endings" in saves.source
    length = _row(sheet, "length")
    assert length.shown == "2 epochs" and length.source == "SFT recipe"  # the recipe's two
    text = "\n".join(render_line(r, resolve(depth="plain")) for r in sheet.render(90))
    assert "held out" in text and "checkpoints" in text and "train for" in text
    assert "empty for one epoch" not in text


def test_picking_sets_the_flags_a_run_takes(card):
    sheet = card()
    _pick(sheet, "length", "3")
    assert sheet.form.values.get("--epochs") == "3" and "--steps" not in sheet.form.values
    _pick(sheet, "length", "500")  # a number typed is steps; the last pick wins
    assert sheet.form.values.get("--steps") == "500" and "--epochs" not in sheet.form.values
    _pick(sheet, "eval", "200")
    assert sheet.form.values.get("eval_rows=") == "200"
    assert _row(sheet, "eval").shown == "200 chats"
    _pick(sheet, "eval", "held.jsonl")  # a file of the data's shape, from the list
    assert sheet.form.values.get("--eval") == "held.jsonl" and "eval_rows=" not in sheet.form.values
    _pick(sheet, "save_steps=", down=2)
    assert sheet.form.values.get("save_steps=") == "100"
    assert sheet.view.command.endswith("--steps 500 --eval held.jsonl save_steps=100") or (
        "--eval held.jsonl" in sheet.view.command and "save_steps=100" in sheet.view.command)
    _pick(sheet, "eval", "off")
    assert sheet.form.values.get("eval_split=") == "0" and "--eval" not in sheet.form.values
    assert "nothing held out" in _row(sheet, "save_steps=").source
    sheet.sel = [f.key for f in sheet.view.fields].index("eval")
    sheet.key("backspace")  # back to its default: every key it set goes
    assert not {"eval_split=", "eval_rows=", "--eval"} & set(sheet.form.values)


def test_an_eval_file_on_the_command_line_shows_on_its_row(card):
    sheet = card("chats.jsonl", "--eval", "held.jsonl")
    row = _row(sheet, "eval")
    assert row.shown == "held.jsonl" and row.fact == "80 chats"


def test_pairs_score_preference_and_offer_only_pair_files(card):
    sheet = card("pairs.jsonl")
    assert "prefers" in _row(sheet, "save_steps=").source
    labels = [o.label for o in _row(sheet, "eval").options()]
    assert "held.jsonl" not in labels and "chats.jsonl" not in labels


def test_too_few_rows_say_so_and_a_count_still_holds_some_out(card, tmp_path):
    _chats(tmp_path / "tiny.jsonl", 31)
    sheet = card("tiny.jsonl")
    assert _row(sheet, "eval").shown == "none" and "under 40" in _row(sheet, "eval").source
    assert held_count(31, 0.05) == 0 and held_count(31, 0.05, 6) == 6
    assert held_count(31, 0.05, 50) == 15  # never more than half: the rest must train
    assert held_count(10_000, 0.05) == 100 and held_count(10_000, 0.05, 500) == 500


def test_the_card_rows_are_not_listed_again_under_more(card):
    sheet = card()
    more = [o.value for o in _row(sheet, "more").options()]
    assert not {"save_steps=", "eval_split=", "eval_rows="} & set(more)


# -- the live card: a row a checkpoint, beside the curve ---------------------------------------


def _live(kind: str, room: int = 40):
    import math

    from poros.cli.ui import run_card as rc
    from poros.cli.ui import screens

    st = screens.TrainState(model="Qwen2.5-0.5B-Instruct")
    st.total, st.step, st.recipe = 2356, 1700, "SFT"
    st.losses = [1.1 + 1.75 * math.exp(-i / 589) for i in range(1700)]
    st.step_time, st.tokens_per_sec, st.vram_gb, st.vram_peak_gb, st.card_gb = .42, 2468, 4, 4.8, 24
    for step, loss, k in [(0, 2.91, 12), (500, 1.70, 41), (1000, 1.36, 55), (1500, 1.26, 58)]:
        data = ({"eval_loss": loss, "prefers": k * 4, "rows": 300} if kind == "pairs"
                else {"eval_loss": loss, "ends": k, "answer_rows": 62})
        st.scores.append(rc.score(step, data))
        st.evals.append((step, loss))
    return st, room


@pytest.mark.parametrize("kind,word", [("chats", "ends"), ("pairs", "prefers")])
@pytest.mark.parametrize("w", [100, 80])
def test_the_live_card_lists_each_checkpoint_beside_the_curve(monkeypatch, kind, word, w):
    from poros.cli.ui import screens
    from poros.cli.ui.text import width

    st, room = _live(kind)
    monkeypatch.setattr(screens, "_room", lambda: room)
    lines = screens.train_surface(st, 0, w, animate=False)
    text = [render_line(r, resolve(depth="plain")) for r in lines]
    assert all(width(r) <= w for r in lines)
    head = next(t for t in text if "held out" in t)
    assert "step" in head and word in head
    rows = [t for t in text if "│" in t and ("○ " in t or "● " in t)]
    said = " ".join(rows)
    for step in ("base", "500", "1,000"):
        assert step in said
    best = next(t for t in text if "│" in t and "● " in t.split("│")[-1])
    assert "1,500" in best and "1.2600" in best  # the lowest held-out loss, marked


def test_the_ledger_keeps_the_base_the_best_and_the_latest():
    from poros.cli.ui import run_card as rc

    losses = [2.9, 1.9, 1.5, 1.2, 1.3, 1.35, 1.4, 1.45]  # best at 600, then it rises
    scores = [rc.Score(i * 200, v) for i, v in enumerate(losses)]
    rows = [render_line(r, resolve(depth="plain")) for r in rc.ledger(scores, 4)]
    assert len(rows) == 4 and "held out" in rows[0]
    assert "base" in rows[1] and "●" in rows[2] and "600" in rows[2] and "1,400" in rows[3]


def test_a_score_counts_the_steps_done_and_the_one_before_any_is_the_base(tmp_path):
    from poros.cli.verbs import runs_report

    evs = [{"event_type": "eval", "step": 0, "data": {"eval_loss": 2.9, "ends": 1,
                                                       "answer_rows": 8}}]
    evs += [{"event_type": "step", "step": i, "data": {"loss": 2.0}} for i in range(20)]
    evs += [{"event_type": "eval", "step": 19, "data": {"eval_loss": 1.5, "ends": 6,
                                                        "answer_rows": 8}}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))
    got = runs_report.scores(str(tmp_path))
    assert [(s.step, s.k, s.n, s.word) for s in got] == [(0, 1, 8, "ends"), (20, 6, 8, "ends")]
