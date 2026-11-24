"""The session holds: quitting and stopping never lose work, mistakes
make no tabs, units are GiB, and the cards and lists hold their shape."""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from poros.cli import session as sm
from poros.cli import sheets
from poros.cli.ui import components as c
from poros.cli.ui import keys, tui
from poros.cli.ui import sheet as sh
from poros.cli.ui.style import Palette
from poros.cli.ui.text import plain
from poros.cli.views.machine_screens import _machine_tail, _to_fix


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _text(lines) -> str:
    return "\n".join(plain(line) for line in lines)


@pytest.fixture
def work():
    """A thread that runs until stopped (ctrl+c) or released."""
    done = threading.Event()
    seen = []

    def run():
        try:
            while not done.is_set():
                time.sleep(0.01)
        except KeyboardInterrupt:
            seen.append("stopped")

    th = threading.Thread(target=run, daemon=True)
    th.start()
    yield th, seen
    done.set()
    th.join(timeout=2)


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(c, "next_step", None)
    s = sm.Session(animate=False)
    s._runs = False
    s.pal = Palette("none", True)
    s._gpu = "RTX 4090 24.0 GiB"
    scr = tui.Screen(_Tty(), Palette(depth="none", dark=True))
    scr.open(tui.Tab("home", lines=[[("the box", None)]], centre=True, keep=False))
    scr.paint = lambda force=False: None
    s.screen = scr
    monkeypatch.setattr(s, "draw", lambda: None)
    monkeypatch.setattr(s, "erase", lambda: None)
    return s


def _drive(monkeypatch, s, keys, look=None):
    """Feed ``keys`` to the session's line editor; ``look(i)`` runs before key i."""
    seq = iter(enumerate(keys))

    def read(fd):
        i, k = next(seq)
        if look:
            look(i)
        return k if isinstance(k, tuple) else (k, "")

    monkeypatch.setattr(sm, "read_key", read)
    return s.edit()


# -- quitting and stopping -------------------------------------------------------


def test_ctrl_d_with_training_running_asks_once_then_stops_it_and_quits(session, monkeypatch,
                                                                         work):
    th, seen = work
    tab = session.screen.open(tui.Tab("train"))
    tab.thread = th
    session.screen.switch(0)
    said = []
    out = _drive(monkeypatch, session, ["ctrl-d", "ctrl-d"],
                 look=lambda i: said.append(plain(session._status_row(100))) if i == 1 else None)
    assert out is None and seen == ["stopped"]
    assert "training is still running · ctrl+d again stops it and quits" in said[0]


def test_ctrl_c_on_home_never_quits_past_running_work_without_asking(session, monkeypatch,
                                                                    work):
    th, seen = work
    tab = session.screen.open(tui.Tab("train"))
    tab.thread = th
    session.screen.switch(0)
    said = []
    out = _drive(monkeypatch, session, ["ctrl-c", "ctrl-c", "ctrl-c"],
                 look=lambda i: said.append(plain(session._status_row(100))))
    assert out is None and seen == ["stopped"]
    assert "ctrl+c again stops it and quits" in said[2]  # the third press is the one that quits


def test_one_ctrl_c_on_the_training_tab_only_arms_the_stop(session, monkeypatch, work):
    th, seen = work
    tab = session.screen.open(tui.Tab("train"))
    tab.thread = th
    said = []

    def look(i):
        said.append(plain(session._status_row(100)))
        if i == 2:
            th.join(timeout=2)

    _drive(monkeypatch, session, ["ctrl-c", "ctrl-c", "ctrl-d"], look=look)
    assert "ctrl+c again stops training" in said[1]
    assert seen == ["stopped"]  # the second press, not the first


def test_ctrl_c_reaches_the_session_as_a_key_not_a_signal():
    import os
    import pty
    import termios

    master, slave = pty.openpty()
    try:
        sm._keys_mode(slave)
        attrs = termios.tcgetattr(slave)
        assert attrs[6][termios.VINTR] != b"\x03"  # ctrl+c is read, never raised mid-draw
        assert attrs[3] & termios.ISIG  # ctrl+z still suspends
        os.write(master, b"\x03")
        assert keys.read_key(slave) == ("ctrl-c", "")
    finally:
        os.close(master)
        os.close(slave)


def test_quit_and_stop_keys_never_collide_with_a_card():
    from inspect import getsource

    from poros.cli.sheets import export, fit, listings

    src = "".join(getsource(m) for m in (sheets, fit, listings, export))  # every sheet's keys
    assert '("ctrl-g", "download"' in src and '("ctrl-x", "stop"' in src
    assert '("ctrl-d", ' not in src and '("ctrl-s", "stop"' not in src



def test_a_digit_on_an_empty_prompt_goes_to_that_tab_and_never_into_a_card(session):
    session.screen.open(tui.Tab("runs", lines=[[("r", None)]]))
    session.screen.open(tui.Tab("fit", lines=[[("f", None)]]))
    assert session._tab_key("char", "2") and session.screen.active == 1
    assert session._tab_key("char", "9") and session.screen.active == 1  # no tab 9: stays
    session.s.buf = "/fit 7"
    assert not session._tab_key("char", "1")  # typing a value keeps its digit
    session.s.buf = ""
    session.screen.tab.listening = 1
    assert not session._tab_key("char", "1")  # a card's own number box keeps it


def test_alt_and_ctrl_reach_the_same_action_and_the_hints_say_one(monkeypatch):
    import pty

    master, slave = pty.openpty()
    try:
        sm._keys_mode(slave)  # keys one at a time, ctrl+s not flow control
        for seq, want in ((b"\x1bs", "ctrl-s"), (b"\x13", "ctrl-s"), (b"\x1bF", "ctrl-f"),
                          (b"\x1bn", "ctrl-n"), (b"\x1bw", "ctrl-w"), (b"\x1bp", "ctrl-p"),
                          (b"\x10", "ctrl-p")):
            os.write(master, seq)
            assert keys.read_key(slave) == (want, "")
    finally:
        os.close(master)
        os.close(slave)
    monkeypatch.setenv("POROS_KEYS", "alt")
    assert c.key_name("ctrl-s") == "alt+s" and c.key_name("ctrl-c") == "ctrl+c"
    assert c.key_name("ctrl-u") == "ctrl+u"  # line editing stays ctrl
    monkeypatch.delenv("POROS_KEYS")
    monkeypatch.setattr(sys, "platform", "linux")
    assert c.key_name("ctrl-n") == "ctrl+n"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(c, "mac_glyphs", lambda: True)
    assert c.key_name("ctrl-n") == "⌥N"  # option, beside command, on a Mac


def test_option_letters_a_mac_terminal_types_reach_the_same_actions(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    for typed, want in (("ƒ", "ctrl-f"), ("ß", "ctrl-s"), ("∑", "ctrl-w"), ("˜", "ctrl-n"),
                        ("π", "ctrl-p")):
        r, w = os.pipe()
        os.write(w, typed.encode())
        try:
            assert keys.read_key(r) == (want, "")
        finally:
            os.close(r)
            os.close(w)

# -- mistakes make no tabs ---------------------------------------------------------


def test_a_word_that_is_not_a_command_makes_no_tab_and_says_so(session):
    assert session._run_in_tab("/hello") and len(session.screen.tabs) == 1
    assert plain(session._error).startswith("✕ /hello isn't a command   ⏎ lists every command")
    session._run_in_tab("/rnus")
    assert "did you mean /runs?" in plain(session._error)
    session._tab_key("char", "x")
    assert session._error == []  # the next key clears it


def test_a_tab_whose_whole_output_is_an_error_becomes_one_status_line():
    lines = [[("✕ ", "e"), ("No runs yet", "hi")],
             [("  │ ", "rl"), ("train one, then talk to it", "dm")]]
    assert plain(sm.parts._only_error(lines)) == "✕ No runs yet   train one, then talk to it"
    fixed = lines + [[("  │ ", "rl"), ("fix  ", "dm"), ("/train", "a"), ("  starts one", "dm")]]
    assert plain(sm.parts._only_error(fixed)).endswith("talk to it   /train starts one")
    one_row = [[("✕ ", "e"), ("No runs yet", "hi"), ("  train one, then talk to it", "tx")]]
    assert plain(sm.parts._only_error(one_row)) == "✕ No runs yet   train one, then talk to it"
    assert sm.parts._only_error([[("● Training", "hi")]]) == []
    assert sm.parts._only_error([[("✕ x", "e")]] * 5) == []  # a long story keeps its tab


def test_watching_a_run_that_trains_in_a_tab_brings_that_tab_forward(monkeypatch, work):
    from poros.cli.verbs import runs_follow

    th, _ = work
    scr = tui.Screen(_Tty(), Palette(depth="none", dark=True))
    scr.open(tui.Tab("home", centre=True, keep=False))
    training = scr.open(tui.Tab("train", run="r1"))
    training.thread = th
    here = scr.open(tui.Tab("runs"))
    monkeypatch.setattr(tui, "current", scr)
    monkeypatch.setattr(tui.local, "tab", here, raising=False)
    assert runs_follow.show_tab(SimpleNamespace(id="r1"))
    assert scr.tab is training and here.gone
    assert not runs_follow.show_tab(SimpleNamespace(id="r2"))


# -- units --------------------------------------------------------------------------


def test_memory_is_said_in_gib_everywhere():
    assert c.gb(24.0) == "22.4 GiB" and round(c.gib(1.0), 4) == 0.9313  # decimal GB in


# -- the cards and lists keep their shape ---------------------------------------------


def _items(names):
    return [sh.Item(cells=[(n.ljust(31), "hi"), ("7B".ljust(6), "tx"), *c.chips(0.25, 16, "a"),
                           ("    6.2 GiB  ", "tx"), ("bounded".ljust(10), "dm")],
                    text=n,
                    actions=[("enter", "train", ["train", n]), ("ctrl-f", "fit", ["fit", n])])
            for n in names]


def test_searching_keeps_the_focused_row_and_esc_gives_it_back():
    lister = sh.Lister(sh.Listing("models", "Models", [], _items(["alpha", "beta", "gamma"])))
    lister.key("down")
    lister.key("down")  # on gamma
    for ch in "am":
        lister.key("char", ch)
    assert lister.shown()[lister.sel].text == "gamma"  # still a match: still focused
    lister.key("esc")
    assert lister.buf == "" and lister.listing.items[lister.sel].text == "gamma"
    assert lister.key("ctrl-f") == "go" and lister.argv == ["fit", "gamma"]


def test_the_search_sits_under_the_title():
    lister = sh.Lister(sh.Listing("models", "Models", [("model", "rl")], _items(["alpha"])))
    lister.key("char", "a")
    lines = [plain(x) for x in lister.render(100)]
    assert lines[1].strip().startswith("│ search  a▏")


def test_a_cards_value_column_never_moves_as_values_are_chosen():
    def col(value):
        form = sh.Form("t", lambda v: sh.View("T", [
            sh.Field("data", "data", value=value, shown=value, source="the note")]))
        line = next(plain(x) for x in sh.Sheet(form).render(100) if "the note" in plain(x))
        return line.index("the note")

    assert col("") == col("x.jsonl") == col("artifacts-graduation/")


def test_narrow_lists_shrink_meters_before_names_and_keep_columns_in_line():
    names = ["Qwen2.5-7B", "Qwen3-30B-A3B-Instruct-2507", "Qwen2.5-0.5B-Instruct"]
    head = [("model".ljust(31), "rl"), ("size".ljust(6), "rl"), ("peak on RTX 4090", "rl")]
    lister = sh.Lister(sh.Listing("models", "Models", head, _items(names)))
    lines = [plain(x) for x in lister.render(64)]
    rows = [x for x in lines if "GiB" in x]
    assert len({x.index("━") for x in rows}) == 1  # one meter column
    assert "Qwen2.5-0.5B-Instruct" in "\n".join(rows)  # names whole where they can be
    assert all(len(x) <= 64 for x in lines)
    assert lines[2].index("size") == rows[1].index("7B")  # the name has no 7B in it
    assert "size" in lines[2]  # a column keeps room for its name


def test_export_with_nothing_finished_is_an_empty_state(monkeypatch):
    from poros import runs

    monkeypatch.setattr(runs, "all_runs", lambda: [])
    assert sheets._nothing_to_export() == "no runs yet: /train starts one"
    monkeypatch.setattr(runs, "all_runs", lambda: [SimpleNamespace(state="running")])
    assert sheets._nothing_to_export().startswith("no finished runs yet")
    monkeypatch.setattr(runs, "all_runs", lambda: [SimpleNamespace(state="succeeded")])
    assert sheets._nothing_to_export() == ""


def test_settings_offer_no_choice_when_there_is_nothing_to_choose(monkeypatch):
    from poros.cli import settings

    monkeypatch.setattr(sheets, "saved_defaults", lambda: {})
    form = settings.settings_form()
    view = form.build(dict(form.values))
    assert "defaults" not in [f.key for f in view.fields]
    assert view.facts[0][:2] == ("defaults", "none saved")


def test_a_settings_list_opens_between_rows_and_hides_none():
    from poros.cli import settings

    sheet = sh.Sheet(settings.settings_form())
    before = [plain(x).split()[1] for x in sheet.render(100) if plain(x).startswith("  │ ")
              and len(plain(x).split()) > 1]
    sheet.key("enter")  # the first row's list
    after = _text(sheet.render(100))
    for label in ("export", "motion", "theme"):
        assert label in after
    assert "● watch here" in after and len(before)


def test_esc_on_a_lone_slash_clears_it(session, monkeypatch):
    session.screen = None
    seen = []
    _drive(monkeypatch, session, [("char", "/"), "esc", "ctrl-d"],
           look=lambda i: seen.append(session.s.buf))
    assert seen == ["", "/", ""]


def test_the_statusline_says_keys_one_way(session):
    session.s.closed_for = ""  # the list under the prompt closed
    scr = session.screen
    scr.open(tui.Tab("machine", lines=[[("done", None)]]))
    rows, _ = session._tab_footer(100)
    assert plain(rows[-1]).rstrip().endswith("⏎ here   ctrl+n new tab   1-9 switch")  # no ⌥
    scr.switch(0)
    rows, _ = session._tab_footer(100)
    assert plain(rows[-1]).rstrip().endswith("switch tabs")


# -- /machine -------------------------------------------------------------------------


def test_machine_counts_what_it_shows_and_says_the_install_once():

    def ck(name, status, value=""):
        return SimpleNamespace(name=name, status=status, value=value)

    r = SimpleNamespace(checks=[ck("torch", "ok", "2.4"), ck("bitsandbytes", "warn"),
                                ck("CUDA", "warn", "none")], gpu=None, can_train={})
    assert _to_fix(r) == 1
    assert _machine_tail(r, 100) == []  # the bitsandbytes row already says it
    r.checks[0] = ck("torch", "warn")
    assert "pip install" in _text(_machine_tail(r, 100))  # "the command below"


def test_a_narrow_cards_bar_gives_way_before_its_note():
    fit = ("fit", "~6.2 of 24.0 GiB", "estimated, not measured")
    form = sh.Form("t", lambda v: sh.View("T", [sh.Field("data", "data", value="x")],
                                          result=fit, meter=(0.26, "dm")))
    for w in (100, 64):
        line = next(plain(x) for x in sh.Sheet(form).render(w) if "~6.2" in plain(x))
        assert line.endswith("estimated, not measured") and "━" in line and len(line) <= w


def test_a_watched_runs_speed_is_per_step_not_cumulative():
    from poros.cli.ui.screens import TrainState
    from poros.cli.verbs.runs_follow import _apply

    st = TrainState(model="m")
    for step in range(3):
        _apply(st, {"event_type": "step", "step": step,
                    "data": {"s_per_step": 0.5, "tokens": 2048 * (step + 1)}})
    assert st.tokens_per_sec == 4096  # 2,048 tokens a step, at 0.5 s a step


def test_a_watched_run_counts_its_time_from_its_first_step():
    from poros.cli.ui.screens import TrainState
    from poros.cli.verbs.runs_follow import _apply

    st = TrainState(model="m")
    _apply(st, {"event_type": "step", "step": 0, "timestamp": time.time() - 600, "data": {}})
    assert 599 < time.monotonic() - st.started < 601  # ten minutes in, not just now


def test_on_a_mac_the_hints_use_the_macs_own_key_symbols(monkeypatch):
    from poros.cli.ui import components as c

    monkeypatch.setattr(c, "mac_glyphs", lambda: True)
    monkeypatch.delenv("POROS_KEYS")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert c.key_name("ctrl-w") == "⌥W" and c.key_name("ctrl-c") == "⌃C"  # ctrl+c stays ctrl
    monkeypatch.setenv("POROS_KEYS", "ctrl")
    assert c.key_name("ctrl-s") == "⌃S"


def test_settings_and_theme_keep_what_the_terminal_said_about_its_colors(session, monkeypatch):
    """/settings and /theme run off the key-reading thread: asking the terminal
    again there lost its reply to the key reader, cached "no answer", and the
    start box fell back from its filled ground to the outlined box."""
    from poros.cli import theme

    forgot = []
    monkeypatch.setattr(theme, "forget_colors", lambda: forgot.append(1))
    session._retheme()  # what /settings (each change, and on close) and /theme call
    assert forgot == []
    session._tab_key("theme", "")  # the terminal itself went light or dark: ask again
    assert forgot == [1]
