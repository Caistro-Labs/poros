"""The tabbed full-screen session (poros/cli/ui/tui.py)."""

from __future__ import annotations

import io
import re

from poros.cli.ui import keys, tui
from poros.cli.ui.style import Palette
from poros.cli.ui.text import plain


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _screen(n_tabs: int = 1):
    scr = tui.Screen(_Tty(), Palette(depth="none", dark=True))
    scr.open(tui.Tab("home", lines=[[("the box", None)]], centre=True, keep=False))
    for i in range(1, n_tabs):
        scr.open(tui.Tab(f"t{i}", lines=[[(f"receipt {i}", None)]]))
    return scr


def test_the_tab_row_underlines_the_active_tab():
    tabs = [tui.Tab("home"), tui.Tab("train", label=lambda: "train 42%"), tui.Tab("runs")]
    top, under = (plain(r) for r in tui.tab_row(tabs, 1, 60))
    assert "▛▜" not in top and top.startswith(" 1 home ") and "2 train 42% ✕" in top
    assert "3 runs" in top and top.rstrip().endswith("ctrl+w close")
    start = top.index(" 2 train 42% ✕ ")
    assert set(under[start : start + len(" 2 train 42% ✕ ")]) == {"━"}
    assert "━" not in under[: start] and len(under) == 60


def test_tabs_that_do_not_fit_fold_into_a_count():
    tabs = [tui.Tab(f"tab-number-{i}") for i in range(12)]
    top = plain(tui.tab_row(tabs, 11, 50)[0])
    assert "‹" in top and "tab-number-11" in top and len(top) <= 50


def test_the_canvas_keeps_the_newest_rows_and_scrolls_up():
    t = tui.Tab("log", lines=[[(f"row {i}", None)] for i in range(30)])
    rows = [plain(r) for r in tui.canvas(t, 40, 10)]
    assert rows[-1] == "row 29" and len(rows) == 10
    t.scroll = 5
    rows = [plain(r) for r in tui.canvas(t, 40, 10)]
    assert rows[0] == "row 15" and "5 rows below" in rows[-1]


def test_home_sits_in_the_middle():
    t = tui.Tab("home", lines=[[("box", None)]], centre=True)
    rows = [plain(r) for r in tui.canvas(t, 40, 9)]
    assert rows.index("box") == 4


def test_closing_a_tab_steps_left_and_home_never_closes():
    scr = _screen(3)
    scr.switch(2)
    scr.close()
    assert [t.title for t in scr.tabs] == ["home", "t1"] and scr.active == 1
    scr.switch(0)
    scr.close()
    assert len(scr.tabs) == 2 and scr.active == 0


def test_the_frame_is_tab_row_canvas_and_footer_with_the_caret_in_it():
    scr = _screen(2)
    scr.footer = lambda w: ([[("─" * w, "rl")], [("▛▜ ❯ ", None)]], (1, 5))
    rows, caret = scr.rows(40, 12)
    text = [plain(r) for r in rows]
    assert len(rows) == 12 and text[0].startswith(" 1 home") and text[-1].startswith("▛▜ ❯")
    assert caret == (11, 5)
    assert "receipt 1" in text


def test_an_open_card_hides_the_footer_and_the_caret():
    scr = _screen(2)
    scr.footer = lambda w: ([[("prompt", None)]], (0, 0))
    scr.tab.card = lambda w: [[("the card", None)]]
    rows, caret = scr.rows(40, 10)
    text = [plain(r) for r in rows]
    assert caret is None and "prompt" not in text and text[tui.HEAD + tui.CARD_TOP] == "the card"


def test_paint_writes_only_the_rows_that_changed():
    scr = _screen(2)
    scr.enter()
    scr.paint()
    scr.out.truncate(0)
    scr.out.seek(0)
    scr.paint()
    assert "receipt" not in scr.out.getvalue()
    scr.tab.lines.append([("new row", None)])
    scr.paint()
    assert "new row" in scr.out.getvalue()


def test_quitting_leaves_the_shell_clean():
    scr = _screen(3)
    scr.enter()
    scr.out.truncate(0)
    scr.out.seek(0)
    scr.leave()
    out = scr.out.getvalue()
    assert out == tui.SCHEME_OFF + tui.ALT_OFF + "\x1b[?25h"  # nothing printed into scrollback


def test_a_command_writes_into_its_own_tab():
    scr = _screen(2)
    t = tui.TabTerm(scr, scr.tab)
    t.print([[("printed", None)]])
    t.raw('{"a": 1}\n')
    assert [plain(r) for r in scr.tab.lines][-2:] == ["printed", '{"a": 1}']
    t.erase(1)
    assert plain(scr.tab.lines[-1]) == "printed"
    live = t.live(lambda f, w: [[("live row", None)]])
    live.start()
    assert scr.tab.surface is not None
    live.close([[("final", None)]])
    assert scr.tab.surface is None and plain(scr.tab.lines[-1]) == "final"


def test_a_settled_card_retells_the_tab_line_and_esc_drops_the_tab(monkeypatch):
    scr = _screen(1)
    tab = scr.open(tui.Tab("train", lines=[[("▛▜ ❯ /train", None)], []], echo=0))
    monkeypatch.setattr(tui, "current", scr)
    tui.retell("/train qwen --data x.jsonl")
    assert "/train qwen --data x.jsonl" in plain(tab.lines[0])
    tui.retell(None)
    assert tab.gone


def test_alt_digits_and_ctrl_arrows_are_keys():
    import os

    from poros.cli.ui.keys import read_key

    for seq, want in ((b"\x1b2", ("alt", "2")), (b"\x1b[1;5C", ("ctrl-right", "")),
                      (b"\x1b[1;5D", ("ctrl-left", "")), (b"\x1b[5~", ("page-up", ""))):
        r, w = os.pipe()
        os.write(w, seq)
        try:
            assert read_key(r) == want
        finally:
            os.close(r)
            os.close(w)


def test_a_stray_print_from_a_command_thread_lands_in_its_tab():
    import threading

    scr = _screen(2)
    real = _Tty()
    out = tui.ThreadOut(real)
    tab = scr.tab

    def work():
        tui.local.tab = tab
        out.write("\x1b[1mplain\x1b[0m line\npart")
        out.write("ial\n")
        tui.local.tab = None

    th = threading.Thread(target=work)
    th.start()
    th.join()
    out.write("to the terminal")
    assert [plain(r) for r in tab.lines][-2:] == ["plain line", "partial"]
    assert real.getvalue() == "to the terminal"


def test_ctrl_c_stops_the_command_in_a_tab():
    import threading
    import time

    scr = _screen(1)
    tab = scr.open(tui.Tab("train"))
    seen = []

    def work():
        try:
            while True:
                time.sleep(0.01)
        except KeyboardInterrupt:
            seen.append("stopped")

    tab.thread = threading.Thread(target=work, daemon=True)
    tab.thread.start()
    assert tab.running and tui.interrupt(tab)
    tab.thread.join(timeout=2)
    assert seen == ["stopped"] and not tab.running


def test_a_card_in_a_tab_reads_the_keys_handed_to_it(monkeypatch):
    import threading

    scr = _screen(1)
    scr.enter()
    monkeypatch.setattr(tui, "current", scr)
    tab = scr.open(tui.Tab("menu"))
    got = []

    def work():
        tui.local.tab = tab
        got.append(tui.TabTerm(scr, tab).choose("Pick", [("one", ""), ("two", "")]))

    th = threading.Thread(target=work, daemon=True)
    th.start()
    for _ in range(200):
        if tab.listening:
            break
        th.join(0.01)
    assert tab.listening and tab.card is not None
    tab.keys.put(("down", ""))
    tab.keys.put(("enter", ""))
    th.join(timeout=2)
    assert got == [1] and tab.card is None and not tab.listening


def test_a_tall_card_loses_its_middle_and_keeps_its_keys():
    card = lambda w: [[(f"row {i}", None)] for i in range(20)] + [[("keys", None)]]  # noqa: E731
    rows = [plain(r).strip() for r in tui.card_canvas(card, 60, 10)]
    assert len(rows) == 10 and rows[-1] == "keys" and "⋮" in rows
    assert rows[tui.CARD_TOP] == "row 0"


def test_the_train_tab_label_carries_its_progress():
    import threading

    scr = _screen(1)
    tab = scr.open(tui.Tab("train"))
    done = threading.Event()
    tab.thread = threading.Thread(target=done.wait, daemon=True)
    tab.thread.start()
    st = {"step": 42, "total": 100}
    tui.TabTerm(scr, tab).progress(lambda: (st["step"], st["total"]))
    assert tab.name() == "train 42%"
    done.set()
    tab.thread.join()
    tab.thread = None
    assert tab.name() == "train ○"
    st["step"] = 100
    assert tab.name() == "train ✓"


def test_the_base_term_ignores_progress():
    from poros.cli.ui import term

    term.Term().progress(lambda: (1, 2))


def test_the_intro_hands_its_frames_over_and_ends_on_the_settled_box(monkeypatch):
    from poros.cli.ui import intro

    monkeypatch.setattr(intro, "DURATION", 0.1)
    words = ("Fine-tune models bigger than your GPU.", "What are we training today?")
    shown = []
    intro.run(shown.append, intro.frames(100, *words))
    assert len(shown) > 1 and len({len(f) for f in shown}) == 1  # no row count jumps
    assert [plain(r) for r in shown[-1]] == [plain(r) for r in intro.final(100, *words)]


def test_tabs_are_the_default_with_ways_back_to_inline(monkeypatch):
    for var in ("POROS_TABS", "POROS_INLINE", "ZELLIJ"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(tui, "size", lambda stream=None: (100, 30))
    assert tui.enabled(_Tty())
    assert not tui.enabled(io.StringIO())  # not a terminal
    monkeypatch.setattr(tui, "size", lambda stream=None: (100, tui.MIN_ROWS - 1))
    assert not tui.enabled(_Tty())  # too short
    monkeypatch.setattr(tui, "size", lambda stream=None: (100, 30))
    for var, value in (("POROS_TABS", "0"), ("POROS_INLINE", "1"), ("ZELLIJ", "0")):
        monkeypatch.setenv(var, value)
        assert not tui.enabled(_Tty()), var
        monkeypatch.delenv(var)


def test_the_list_under_the_prompt_lies_over_the_canvas_without_moving_it():
    scr = _screen(1)
    tab = scr.open(tui.Tab("log", lines=[[(f"row {i}", None)] for i in range(40)]))
    rest = [[("─", "rl")], [("prompt", None)], [("─", "rl")], [("status", None)]]
    scr.footer = lambda w: (rest, (1, 2))
    still = [plain(r) for r in scr.rows(40, 20)[0]]
    scr.footer = lambda w: (rest[:2] + [[(f"/cmd {i}", None)] for i in range(6)], (1, 2))
    rows, caret = scr.rows(40, 20)
    text = [plain(r) for r in rows]
    assert len(rows) == 20 and text[-1] == "/cmd 5"
    body = text[tui.HEAD : text.index("prompt") - 1]
    assert body == still[tui.HEAD : tui.HEAD + len(body)]  # rows above the list stay put
    assert caret == (text.index("prompt"), 2)
    assert tab.lines


def test_terminal_app_option_arrows_step_tabs_not_fit(monkeypatch):
    import os


    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    for seq, want in ((b"\x1bf", ("ctrl-right", "")), (b"\x1bb", ("ctrl-left", ""))):
        r, w = os.pipe()
        os.write(w, seq)
        try:
            assert keys.read_key(r) == want
        finally:
            os.close(r)
            os.close(w)
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")  # option sent as alt: ⌥F is fit
    r, w = os.pipe()
    os.write(w, b"\x1bf")
    try:
        assert keys.read_key(r) == ("ctrl-f", "")
    finally:
        os.close(r)
        os.close(w)


def test_mac_option_digits_and_shift_arrows_switch_tabs():
    import os

    from poros.cli.ui.keys import read_key

    for seq, want in (("¡".encode(), ("alt", "1")), ("ª".encode(), ("alt", "9")),
                      (b"\x1b[1;2C", ("ctrl-right", "")), (b"\x1b[1;2D", ("ctrl-left", ""))):
        r, w = os.pipe()
        os.write(w, seq)
        try:
            assert read_key(r) == want
        finally:
            os.close(r)
            os.close(w)


def test_an_image_belongs_to_the_tab_that_shows_it():
    scr = _screen(1)
    scr.pal = Palette(depth="256", dark=True)  # a plain palette drops images
    img = "\x1b_Ga=T,i=7;PNG\x1b\\"
    scr.open(tui.Tab("machine", lines=[[(img, "raw"), ("    ", None)], [("RTX 4090", None)]]))
    scr.enter()
    scr.paint()
    assert img in scr.out.getvalue()
    for go, sent in ((0, False), (1, True)):  # home: taken off; back: placed again
        scr.out.truncate(0)
        scr.out.seek(0)
        scr.switch(go)
        scr.paint()
        out = scr.out.getvalue()
        assert (img in out) == sent and (tui.KITTY_FORGET in out) == (not sent)


def test_the_terminal_says_when_it_goes_light_or_dark():
    import os

    from poros.cli.ui.keys import read_key

    for seq in (b"\x1b[?997;1n", b"\x1b[?997;2n"):
        r, w = os.pipe()
        os.write(w, seq)
        try:
            assert read_key(r) == ("theme", "")
        finally:
            os.close(r)
            os.close(w)


def test_a_tab_laid_out_by_width_follows_a_resize():
    t = tui.Tab("home", centre=True, fit=lambda w: [[("x" * w, None)]])
    assert [plain(r) for r in tui.canvas(t, 30, 3)][1] == "x" * 30
    assert [plain(r) for r in tui.canvas(t, 12, 3)][1] == "x" * 12


def test_a_result_shown_in_a_tab_follows_a_resize():
    scr = _screen(1)
    t = tui.Tab("machine")
    tui.TabTerm(scr, t).print([[("> /machine", None)]])
    tui.TabTerm(scr, t).show(lambda w: [[("y" * w, None)], [("end", None)]])
    assert [plain(r) for r in tui.canvas(t, 30, 3)][1] == "y" * 30
    assert [plain(r) for r in tui.canvas(t, 12, 3)][1] == "y" * 12
    assert len(tui.laid(t.lines, 12)) == 3  # counted as rows, not as one entry


def test_a_resize_repaints_off_the_signal_handler():
    import threading

    scr = _screen(1)
    scr.enter()
    with scr._lock:  # a paint is half way when the signal lands
        scr.resized()
    for th in threading.enumerate():
        if th is not threading.current_thread() and th.daemon:
            th.join(timeout=2)
    assert "the box" in scr.out.getvalue()


def test_a_tab_hides_its_typed_command_and_a_twin_comes_forward():
    scr = _screen(1)
    t = scr.open(tui.Tab("machine", lines=[[("❯ /machine", None)], [], [("ready", None)]],
                         echo=0, command="/machine"))
    assert [plain(r) for r in tui.shown(t, 40)] == ["", "ready"]
    assert scr.twin("/machine") == 1 and scr.twin("/runs") == -1
    scr.switch(0)
    scr.replace(1, tui.Tab("runs", command="/runs"))
    assert scr.active == 1 and scr.tabs[1].title == "runs" and len(scr.tabs) == 2


def test_a_card_over_the_tab_makes_no_tab():
    scr = _screen(1)
    over = tui.Tab("settings", card=lambda w: [[("the card", None)]])
    scr.over = over
    assert scr.tab is over and len(scr.tabs) == 1
    frame, _ = scr.rows(60, 20)
    top = plain(frame[0])
    assert top.startswith(" 1 settings") and "close" not in top  # it names the tab; esc is its own
    assert any("the card" in plain(r) for r in frame)


def test_a_list_never_cuts_the_start_box_in_half():
    scr = tui.Screen(_Tty(), Palette(depth="none", dark=True))
    box = [[("╭─╮", None)]] + [[("│ │", None)] for _ in range(12)] + [[("╰─╯", None)]]
    scr.open(tui.Tab("home", lines=box, centre=True, keep=False))
    rest = [[("─", None)], [("❯", None)], [("─", None)], [("", None)]]
    scr.footer = lambda w: (rest + [[(f"/cmd {i}", None)] for i in range(10)], None)
    short = [plain(r) for r in scr.rows(80, 31)[0]]
    assert not any("│" in r for r in short)  # no room above the list: the box steps away
    tall = [plain(r) for r in scr.rows(80, 60)[0]]
    assert "╭─╮" in tall and "╰─╯" in tall  # room: whole, where it was


def test_where_the_box_and_the_list_both_fit_the_box_rises_clear_of_it():
    # 163x43 Terminal.app: centred, the box reached one row into the list's
    # room and used to vanish the moment / was typed
    scr = tui.Screen(_Tty(), Palette(depth="none", dark=True))
    box = [[("╭─╮", None)]] + [[("│ │", None)] for _ in range(12)] + [[("╰─╯", None)]]
    scr.open(tui.Tab("home", lines=box, centre=True, keep=False))
    rest = [[("─", None)], [("❯", None)], [("─", None)], [("", None)]]
    scr.footer = lambda w: (rest + [[(f"/cmd {i}", None)] for i in range(10)], None)
    rows = [plain(r) for r in scr.rows(163, 43)[0]]
    top, end = rows.index("╭─╮"), rows.index("╰─╯")
    assert end - top == 13 and end < rows.index("/cmd 0") - 3  # whole, above the prompt


def test_terminal_app_rules_are_underlined_blanks_not_font_dashes():
    from poros.cli.ui.text import render_line

    pal = Palette(depth="256", dark=True, drawn_rules=True)
    out = render_line([("─" * 10, "rl"), (" home", "hi")], pal)
    assert "─" not in out and "\x1b[4;" in out and " home" in out
    assert "─" in render_line([("─" * 10, "rl")], Palette(depth="256", dark=True))


def test_a_drawn_rule_breaks_for_one_cell_where_its_color_changes():
    from poros.cli.ui.text import render_line

    pal = Palette(depth="256", dark=True, drawn_rules=True)
    out = render_line([("━" * 6, "a"), ("─", "rl"), ("─" * 9, "rl")], pal)
    assert out.count("\x1b[4;") == 3 and "\x1b[0m \x1b[4;" in out  # one plain cell, then grey
    assert len(re.sub(r"\x1b\[[0-9;]*m", "", out)) == 16


def test_a_held_surface_hands_its_image_to_the_result_with_no_frame_between():
    # /machine's chip scan ends and its still takes its place: a frame with
    # neither would take every image off the screen and send the still from
    # nothing, and the chip blinks
    scr = _screen(1)
    scr.pal = Palette(depth="256", dark=True)
    t = tui.Tab("machine")
    scr.open(t)
    scr.enter()
    scan = "\x1b_Ga=d,d=i,i=7,q=2\x1b\\\x1b_Ga=p,i=8,q=2\x1b\\"
    live = tui.TabTerm(scr, t).live(lambda f, w: [[(scan, "raw"), ("    ", None)]])
    live.start()
    scr.out.truncate(0)
    scr.out.seek(0)
    live.close(hold=True)
    scr.paint()  # a ticker's paint while the result is on its way
    still = "\x1b_Ga=T,i=7;PNG\x1b\\"
    tui.TabTerm(scr, t).print([[(still, "raw"), ("    ", None)]])
    out = scr.out.getvalue()
    assert still in out and tui.KITTY_FORGET not in out and "\x1b[2J" not in out
    assert t.surface is None and not t.held


def test_a_held_surface_goes_when_nothing_comes_after_it():
    scr = _screen(1)
    t = tui.Tab("machine")
    scr.open(t)
    live = tui.TabTerm(scr, t).live(lambda f, w: [[("scan", None)]])
    live.start()
    live.close(hold=True)
    assert t.held and t.surface is not None
    tui.let_go(t)
    assert t.surface is None and not t.held
