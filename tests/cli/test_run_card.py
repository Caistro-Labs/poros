"""The drawn curve, and the run card it finishes into."""

import sys

import pytest

from poros.cli.ui import curve
from poros.cli.ui import run_card as rc
from poros.cli.ui import screens as sc
from poros.cli.ui.text import width


def _text(line):
    return "".join(s for s, _ in line)


def _result(**kw):
    losses = [2.4 * 0.995 ** i + 0.3 for i in range(400)]
    base = dict(steps=400, wall_s=312.0, final_loss=losses[-1], first_loss=losses[0], losses=losses,
                peak_vram_gb=18.1, card_gb=24.0, peak_ram_gb=31.0, ram_total_gb=64.0,
                output_dir="./runs/qwen-notes", receipt="./runs/qwen-notes/receipt.json",
                tokens_per_sec=1665.0, next_cmd="/prove qwen-notes", recipe="sft")
    return sc.TrainResult(**{**base, **kw})


def _state():
    r = _result()
    return sc.TrainState(model="Qwen/Qwen2.5-7B", arch="dense", gpu="RTX 4090", card_gb=24.0,
                         output_dir=r.output_dir, run="x", step=400, total=400, losses=r.losses,
                         step_time=0.8, step_times=[0.8] * 20, tokens_per_sec=1665.0, vram_gb=17.6,
                         vram_peak_gb=18.1, ram_gb=30.0)


def test_the_verdict_only_speaks_from_held_out_data():
    assert rc.verdict(None, 1.0) == "" and rc.verdict(2.0, None) == ""
    assert rc.verdict(2.0, 1.0) == "Learned well"
    assert rc.verdict(2.0, 1.95) == "Barely moved"
    assert rc.verdict(2.0, 2.1) == "Held-out loss rose"


def test_no_verdict_word_without_held_out_data():
    text = "\n".join(_text(row) for row in sc.train_done(_result(), 100))
    assert not any(w in text for w in ("Learned well", "Barely moved", "Held-out loss rose"))


def test_the_line_never_moves_once_drawn():
    losses = [2.4 * 0.995 ** i + 0.3 for i in range(400)]
    whole = curve.Series(400).extend(losses)
    part = curve.Series(400).extend(losses[:120])
    assert whole.line[:120] == part.line[:120]


def test_the_scale_only_widens_and_lands_on_cell_centres():
    a = curve.fit_scale([2.0, 1.0], 10)
    b = curve.fit_scale([2.0, 0.5], 10)
    assert b.top >= a.top and b.top - 9 * b.step <= a.top - 9 * a.step
    assert sorted(a.labels()) == [2, 5, 8]
    assert sorted(curve.fit_scale([2.0, 1.0], 7).labels()) == [1, 3, 5]


@pytest.mark.parametrize("w", [76, 80, 100, 140])
def test_finish_rests_on_the_finished_screen_and_stays_in_bounds(w):
    r, st = _result(), _state()
    rest = [_text(row).rstrip() for row in sc.train_done(r, w)]
    end = sc.finish_frame(r, st, w, rc.FINAL)
    assert [_text(row).rstrip() for row in end] == ["", *rest]
    for t in (0, 80, 200, curve.LAND, sc.finish_length(r, w)):
        assert all(width(row) <= max(w, 100) for row in sc.finish_frame(r, st, w, t))


def test_finish_length_keeps_room_for_every_row_below():
    r = _result()
    assert curve.LAND < sc.finish_length(r, 100) < 2000


def test_narrow_terminals_keep_todays_screen():
    assert not any("╭" in _text(row) for row in sc.train_done(_result(), 70))
    assert any("final loss" in _text(row) for row in sc.train_done(_result(), 56))  # plain pairs


def test_without_images_the_curve_is_cells():
    series = curve.Series(400).extend(_result().losses)
    scale = curve.fit_scale(series.line, 7)
    rows = curve.block_rows(series, scale, 62, 399, 0, [], [], rc.FINAL)
    assert len(rows) == 7
    assert all(width(row) == 62 for row in rows)
    assert not any("\x1b" in _text(row) for row in rows)


def test_the_drawn_curve_is_kitty_only(monkeypatch):
    monkeypatch.setattr(curve, "protocol", lambda stream=None: "iterm")
    assert curve.Painter.maybe() is None
    monkeypatch.setattr(curve, "protocol", lambda stream=None: None)
    assert curve.Painter.maybe() is None


def _page(ww=100, state="succeeded", losses=None):
    from poros.cli import views

    pairs = losses or [(i + 1, 2.4 * 0.99 ** i + 0.3) for i in range(200)]
    x = {"id": "abc", "name": "qwen-notes", "state": state, "model": "Qwen/Qwen2.5-7B",
         "total": 200, "step": pairs[-1][0], "output_dir": "/nonexistent/abc",
         "created_at": None, "finished_at": None}
    return views.run_page(x, {"losses": pairs, "action": {}, "receipt": {}}, ww)


def test_the_run_page_draws_the_curve_in_its_rail():
    rows = [_text(row) for row in _page()]
    assert not any("╭" in row or "╰" in row for row in rows)  # no second frame: the page has a rail
    assert any("final loss" in row for row in rows)
    assert any("200" in row and row.strip().endswith("200") for row in rows)  # the axis, in steps
    assert all(width(row) <= 100 for row in _page())


def test_a_continued_run_keeps_its_own_steps_on_the_axis():
    pairs = [(i, 1.5 - i * 0.001) for i in range(101, 201)]
    rows = [_text(row) for row in _page(losses=pairs)]
    assert any("final loss" in row for row in rows)
    gaps = [(1, 2.0), (2, 1.9), (10, 1.5), (200, 1.0)]  # steps it didn't write hold the last value
    assert any("final loss" in _text(row) for row in _page(losses=gaps))


def test_a_narrow_run_page_keeps_the_sparks():
    rows = [_text(row) for row in _page(ww=70)]
    assert any(row.lstrip(" │").startswith("loss") for row in rows)
    assert not any("final loss" in row for row in rows)


def test_a_stopped_run_ends_on_its_last_real_loss():
    pairs = [(i + 1, 2.0 - i * 0.01) for i in range(50)]
    rows = "\n".join(_text(row) for row in _page(state="cancelled", losses=pairs))
    assert f"{pairs[-1][1]:.4f}" in rows


def test_watching_reads_held_out_scores_from_the_events():
    from poros.cli.verbs.runs_follow import _apply

    st = _state()
    st.evals = []
    _apply(st, {"event_type": "eval", "step": 0, "data": {"eval_loss": 2.9}})
    _apply(st, {"event_type": "eval", "step": 40, "data": {"accuracy": 0.5}})
    assert st.evals == [(0, 2.9)]


def test_the_memory_numbers_stay_in_their_cells_at_the_finish():
    r, st = _result(), _state()

    def col(t):
        # the memory row: "vram", or on a Mac (one pool) "memory"
        row = next(_text(x) for x in sc.finish_frame(r, st, 100, t) if rc.MEMORY in _text(x))
        return row.index("GiB")

    assert col(0) == col(200) == col(rc.FINAL)
    assert any("GiB peak" in _text(x) for x in sc.train_done(r, 100))


def test_the_finish_stays_quick():
    assert sc.finish_length(_result(), 100) <= 800


def test_ctrl_c_during_the_finish_still_shows_the_finished_card():
    from poros.cli.ui.train_view import TrainView

    class Interrupted:
        live = True
        closed = None

        def play(self, *a, **k):
            raise KeyboardInterrupt

        def close(self, final=None):
            self.closed = final

    view = TrainView.__new__(TrainView)
    view.st, view.live, view.stream = _state(), Interrupted(), None
    view.st.series = curve.Series(400).extend(_result().losses)
    view.finish(_result(), 100)
    assert any("final loss" in _text(row) for row in view.live.closed)


def test_no_image_motion_over_ssh_or_when_motion_is_off(monkeypatch):
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):  # a local terminal, wherever run
        monkeypatch.delenv(name, raising=False)
    series = curve.Series(400).extend(_result().losses)
    scale = curve.fit_scale(series.line, 10)
    p = curve.Painter(_palette(), None)
    p.frame(series, scale, 62, 399)
    assert p.halo(series, 399, 0, 0.5, 5, 9)
    monkeypatch.setenv("POROS_NO_ANIM", "1")
    assert p.halo(series, 399, 0, 0.5, 5, 9) == ""
    monkeypatch.delenv("POROS_NO_ANIM")
    p.remote = True
    assert p.halo(series, 399, 0, 0.5, 5, 9) == ""


def _palette():
    from poros.cli.ui.style import Palette

    return Palette(depth="truecolor", dark=True)


def test_watching_says_what_ctrl_c_does():
    st = _state()
    assert any("ctrl-c to stop" in _text(row) for row in sc._status(st, 140))
    st.watching = True
    assert any("ctrl-c stops watching" in _text(row) for row in sc._status(st, 140))


def test_the_statusline_names_the_run_once_and_keeps_the_key_longest(monkeypatch):
    monkeypatch.setenv("HOME", "/home/u")
    st = _state()
    st.run, st.output_dir = "1105-2231-d22a", "/home/u/.poros/runs/1105-2231-d22a"
    wide = "".join(_text(row) for row in sc._status(st, 140))
    assert "~/.poros/runs/1105-2231-d22a" in wide and "run 1105" not in wide
    assert "/home/u" not in wide
    assert "ctrl-c to stop" in "".join(_text(row) for row in sc._status(st, 40))


def test_a_version_is_not_a_reading():
    from poros.cli.ui import components as c

    def bright(text):
        return [t for t, st in c.inked([(text, "dm")]) if st == "tx"]

    assert bright("macOS 27.2 · arm64 · Python 3.12.12") == []
    assert bright("driver 550.54 · CUDA 12.4 · PCIe 5 x16") == []
    assert bright("3 GiB free") == ["3"]
    hi = c.inked([("22 W", "hi")])
    assert hi == [("22", "hi"), (" W", "rl")]
    assert c.inked([("31°C", "hi")]) == [("31", "hi"), ("°C", "rl")]
    assert c.inked([("0%", "hi")]) == [("0", "hi"), ("%", "rl")]


def test_the_head_of_the_block_curve_is_not_a_solid_column():
    series = curve.Series(100).extend([1.0 + i * 0.04 for i in range(50)])
    scale = curve.fit_scale(series.line, 10)
    rows = curve.block_rows(series, scale, 40, 49)
    head = [row[19] for row in rows if row[19][0].strip()]
    assert len(head) > 3
    assert [st for _, st in head].count("hi") <= 2


def test_a_short_run_ends_where_its_numbers_do():
    series = curve.Series(2).extend([2.8289, 1.7019])
    assert series.line == series.raw


def test_a_two_step_block_curve_slopes_and_does_not_step():
    series = curve.Series(2).extend([2.8, 1.7])
    scale = curve.fit_scale(series.line, 10)
    rows = curve.block_rows(series, scale, 62, 1, t=rc.FINAL)
    tops = [next(r for r, row in enumerate(rows) if row[c][0].strip()) for c in range(62)]
    assert tops == sorted(tops) and len(set(tops)) >= 4
    assert max(b - a for a, b in zip(tops, tops[1:])) <= 1


def test_a_short_block_curve_draws_only_the_steps_so_far():
    series = curve.Series(4).extend([2.8])
    scale = curve.fit_scale(series.line, 10)
    rows = curve.block_rows(series, scale, 62, 0)
    drawn = [c for c in range(62) if any(row[c][0].strip() for row in rows)]
    assert drawn == [0]


def test_the_image_fill_never_rises_above_a_short_line():
    painter = curve.Painter(curve.resolve(sys.stdout), sys.stdout)
    series = curve.Series(2).extend([2.8, 1.7])
    scale = curve.fit_scale(series.line, 10)
    painter._reset(series, scale, 62)
    x = int(painter.px_of(1, 2)) - 2
    painter._column(x, series, 0)
    line_y = painter.y_of(series.line[1])
    above = int(line_y - 4 * painter.s)
    grid = set(painter.grid)
    alphas = [painter.px[(y * painter.w + x) * 4 + 3] for y in range(above) if y not in grid]
    assert not any(alphas)


def test_a_chat_runs_card_says_whether_its_answers_end():
    """A loss can fall while a model never stops: the verdict says both, from the same pass."""
    assert rc.with_ends("Learned well", (8, 8)) == "Learned well · ends its answers 8 of 8"
    assert rc.with_ends("", (0, 8)) == "Ends its answers 0 of 8"
    assert rc.with_ends("Barely moved", None) == rc.with_ends("Barely moved", (0, 0)) == (
        "Barely moved")
    r = _result(eval_start=2.9, eval_end=1.4, ends=(8, 8))
    top = _text(sc.train_done(r, 100)[0])
    assert "Trained  Learned well · ends its answers 8 of 8" in top and "400 steps" in top
    assert "ends its answers" not in _text(sc.train_done(_result(eval_start=2.9, eval_end=1.4),
                                                         100)[0])
