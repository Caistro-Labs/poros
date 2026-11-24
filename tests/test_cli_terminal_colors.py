"""The palette follows the terminal it is drawn in: built from the colors the
terminal reports, or drawn in the terminal's own colors when it reports none."""

from __future__ import annotations

import re

import pytest

from poros.cli import theme
from poros.cli.ui import style
from poros.cli.ui.style import ACCENT, mix, resolve
from poros.cli.ui.text import render

SGR = re.compile(r"\x1b\[([0-9;]*)m")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("POROS_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    theme._reset_cache()
    yield
    theme._reset_cache()


def _terminal(monkeypatch, fg, bg):
    monkeypatch.setattr(theme, "_query_terminal_colors", lambda: (fg, bg))


@pytest.mark.parametrize(
    "reply, expected",
    [
        (b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\\x1b]11;rgb:1e1e/2a2a/4b4b\x1b\\\x1b[?62c",
         ((255, 255, 255), (30, 42, 75))),
        (b"\x1b]11;rgb:fc/fc/fc\x07", (None, (252, 252, 252))),
        (b"\x1b]10;#0a3d0a\x07", ((10, 61, 10), None)),
        (b"\x1b[?1;2c", (None, None)),  # only the DA1 answer: the terminal won't say
    ],
)
def test_osc_replies_parse(reply, expected):
    assert theme._parse_osc(reply) == expected


def test_a_white_terminal_gets_dark_ink(monkeypatch):
    _terminal(monkeypatch, (0, 0, 0), (255, 255, 255))
    pal = resolve(depth="truecolor")
    assert not pal.dark and pal.sgr("hi") == "1;38;2;0;0;0"
    assert pal.sgr("dm") == "38;2;%d;%d;%d" % mix((0, 0, 0), (255, 255, 255), 0.5)


@pytest.mark.parametrize("bg", [(0, 43, 54), (12, 40, 12), (30, 42, 75), (250, 240, 220)])
def test_every_grey_leans_toward_the_terminals_own_ground(monkeypatch, bg):
    fg = (238, 232, 213) if theme.luminance(bg) < 0.18 else (40, 40, 40)
    _terminal(monkeypatch, fg, bg)
    pal = resolve(depth="truecolor")
    for tier, share in style.TIERS.items():
        code = pal.sgr(tier)
        assert code.endswith("38;2;%d;%d;%d" % mix(fg, bg, share)), (tier, code)
    assert pal.sgr("a") == "38;2;%d;%d;%d" % ACCENT


def test_an_unreadable_reported_foreground_is_not_used(monkeypatch):
    _terminal(monkeypatch, (20, 20, 20), (10, 10, 10))  # fg ~ bg: a bad report
    pal = resolve(depth="truecolor")
    assert pal.fg == style.PAPER


def test_an_orange_ground_keeps_the_accent_readable(monkeypatch):
    _terminal(monkeypatch, (0, 0, 0), (255, 110, 60))
    pal = resolve(depth="truecolor")
    assert pal.sgr("a") == "38;2;0;0;0"


def test_a_terminal_that_wont_say_is_drawn_in_its_own_colors(monkeypatch):
    _terminal(monkeypatch, None, None)
    pal = resolve(depth="truecolor")
    assert pal.adaptive
    assert (pal.sgr("hi"), pal.sgr("tx"), pal.sgr("dm"), pal.sgr("rl")) == ("1", "39", "2", "90")
    assert pal.sgr("a") == "38;2;255;107;53"
    from poros.cli.ui import components as c
    from poros.cli.ui.text import S, ln

    lines = [ln(S("● ", "a"), S("Ready to train", "a"), S("  RTX 5090", "tx")),
             c.rail(S("gpu".ljust(11), "dm"), *c.bar(0.4, 16), *c.flow_bar(0.6, 16)),
             *c.key_hints(["↑↓ move", "esc cancel"], 80)]
    out = render(lines, pal)
    greys = [m for m in SGR.findall(out) if m.startswith("38;2") or ";38;2" in m]
    allowed = {"255;107;53", "229;72;77"}
    for code in greys:  # no grey is ever guessed: only the accent, red and spectrum
        rgb = code.split("38;2;")[1]
        r, g, b = map(int, rgb.split(";"))
        assert rgb in allowed or max(r, g, b) - min(r, g, b) > 30, code


def test_a_pinned_theme_ignores_the_terminal(monkeypatch):
    _terminal(monkeypatch, (0, 0, 0), (255, 255, 255))
    theme.set_mode("dark")
    pal = resolve(depth="truecolor")
    assert pal.dark and pal.ground is None and not pal.adaptive


def test_no_color_never_asks_the_terminal(monkeypatch):
    def boom():
        raise AssertionError("queried the terminal under NO_COLOR")

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(theme, "_query_terminal_colors", boom)
    assert theme.terminal_colors() == (None, None)


def test_the_mark_and_the_composer_carry_the_brand_on_any_ground(monkeypatch):
    from poros.cli import session as sm

    _terminal(monkeypatch, None, None)
    pal = resolve(depth="truecolor")
    out = render([sm.parts.prompt()], pal)
    assert "38;2;255;107;53m▛▜" in out  # the arch mark in the accent
    assert "\x1b[2m" not in out  # nothing in the composer is faint


def test_a_narrow_terminal_still_gets_the_intro_words_under_the_mark():
    from poros.cli.ui import intro
    from poros.cli.ui.text import plain

    rows = [plain(r) for r in intro.final(46, "Fine-tune here.", "RTX 4090")]
    assert any("██" in r for r in rows)
    assert rows[-3:] == ["  poros", "  Fine-tune here.", "  RTX 4090"]  # no version here
    assert all(len(r) <= 46 for r in rows)


def test_a_wide_terminal_starts_in_the_box_and_the_intro_lands_in_it():
    from poros.cli.ui import intro
    from poros.cli.ui.text import plain

    rows = [plain(r) for r in intro.final(100, "Fine-tune here.", "RTX 4090")]
    assert rows[0].strip().startswith("╭") and rows[-1].strip().startswith("╰")
    assert len({len(r) for r in rows}) == 1 and len(rows[0]) <= 100  # one box, centred
    assert sum("██" in r for r in rows) == 5  # the mark, inside
    assert any(r.rstrip("│ ").endswith("ctrl+d") for r in rows)  # keys on the right
    assert not any(" v" + __import__("poros").__version__ in r for r in rows)  # names no version
    for t in (0.3, 0.9, 1.4):  # every frame is the same box: nothing jumps
        assert len(intro.box_frame(t, 100, intro.box_column("a", "b", 50))) == len(rows)
    words = "Fine-tune models bigger than your GPU."
    narrow = [plain(r) for r in intro.final(intro.BOX_MIN, words, "b")]
    assert narrow[0].strip().startswith("╭") and max(map(len, narrow)) <= intro.BOX_MIN
    assert any("GPU." in r for r in narrow)  # the words wrap, never cut
    assert not plain(intro.final(intro.BOX_MIN - 1, "a", "b")[0]).strip().startswith("╭")


def test_a_note_gives_way_in_whole_parts():
    from poros.cli.ui.sheet.frame import _fit

    assert _fit("7B · fits · validated", 12) == "7B · fits"
    assert _fit("a catalog name, a Hub id or a path", 20).endswith("…")
    assert not _fit("a catalog name, a Hub id or a path", 20).endswith(" …")


def test_the_loss_curve_is_the_run_and_a_spike_is_never_averaged_away():
    import math
    import random

    from poros.cli.ui import components as c

    rng = random.Random(7)
    clean = [1.1 + 1.75 * math.exp(-i / 50) + rng.uniform(-0.06, 0.06) for i in range(200)]
    assert c.spikes(clean) == set()  # noise is not a spike
    assert c.spikes(clean[:12]) == set()  # nor a short run's steep start
    spiky = list(clean)
    spiky[119] += 1.4
    assert c.spikes(spiky) == {119}
    row = c.spark(spiky, 28)[0]
    assert [st for _, st in row].count("e") == 1  # its column, in red, at its peak
    assert c.spark(clean, 28) != c.spark(list(reversed(clean)), 28)  # drawn from the data


@pytest.mark.parametrize("bg", [(255, 255, 255), (12, 40, 12), (30, 42, 75)])
def test_the_intro_ends_on_the_mark_in_the_brand(monkeypatch, bg):
    from poros.cli.ui import intro

    fg = (34, 34, 34) if theme.luminance(bg) > 0.18 else (230, 230, 230)
    _terminal(monkeypatch, fg, bg)
    pal = resolve(depth="truecolor")
    last = render(intro.final(100, "a", "b"), pal)
    ground = "48;2;%d;%d;%d" % pal.panel()
    assert set(re.findall(r"\x1b\[([0-9;]*)m█", last)) == {"1;38;2;255;107;53;" + ground}


@pytest.mark.parametrize("fg, bg, lighter", [
    ((230, 230, 230), (22, 22, 22), True),  # a dark ground: the panel lifts
    ((34, 34, 34), (255, 255, 255), False),  # a light one: it sinks a hair
    ((131, 148, 150), (0, 43, 54), True),  # Solarized dark, whose grey is its ground
])
def test_the_start_box_is_a_filled_panel_where_the_ground_is_known(monkeypatch, fg, bg, lighter):
    from poros.cli.ui import intro

    _terminal(monkeypatch, fg, bg)
    pal = resolve(depth="truecolor")
    panel = pal.panel()
    assert (theme.luminance(panel) > theme.luminance(bg)) == lighter
    rows = [render([r], pal) for r in intro.final(100, "a", "b")]
    assert rows[0] == rows[-1] == ""  # the edges draw nothing: no half block to seam
    assert all("48;2;%d;%d;%d" % panel in r for r in rows[1:-1])  # the rest is the panel
    assert not any(ch in "".join(rows) for ch in "╭╮╰╯│─▄▔")  # no line, no half row


def test_where_the_ground_is_unknown_the_box_is_a_faint_line_in_the_text_color(monkeypatch):
    from poros.cli.ui import intro
    from poros.cli.ui.style import Palette

    pal = Palette(depth="256", dark=True, adaptive=True)
    rows = render(intro.final(100, "a", "b"), pal).splitlines()
    assert "\x1b[2m╭" in rows[0] and "\x1b[2m╰" in rows[-1]
    assert "\x1b[90m" not in "".join(rows)  # never the theme's own grey


def test_the_words_fade_in_row_by_row_and_settle_plain():
    from poros.cli.ui import intro

    right = intro.box_column("Fine-tune here.", "RTX 4090", 50)
    early = intro.box_frame(1.0, 100, right)
    styles = {st[0] for r in early for _, st in r if isinstance(st, tuple)}
    assert "fd" in styles  # mid-fade
    mid = "\n".join("".join(t for t, _ in r) for r in intro.box_frame(1.12, 100, right))
    assert "poros" in mid and "Quit" not in mid  # the top row arrives before the last
    final = intro.box_frame(intro.DURATION + 1.0, 100, right)
    assert not any(isinstance(st, tuple) and st[0] == "fd" for r in final for _, st in r)


@pytest.mark.parametrize("fg, bg", [
    ((131, 148, 150), (0, 43, 54)),  # Solarized dark
    ((101, 123, 131), (253, 246, 227)),  # Solarized light
    ((216, 222, 233), (46, 52, 64)),  # Nord
    ((200, 200, 200), (90, 90, 90)),  # a low-contrast grey
    ((40, 254, 20), (0, 0, 0)),  # green on black
])
@pytest.mark.parametrize("depth", ["truecolor", "256"])
def test_a_meter_keeps_fill_estimate_and_track_apart(monkeypatch, fg, bg, depth):
    from poros.cli.ui import components as c

    _terminal(monkeypatch, fg, bg)
    pal = resolve(depth=depth)
    styles = (c.chips(.5, 4)[0][1], c.chips(.5, 4, "dm")[0][1], "tk")
    fill, est, track = (pal.sgr(st) for st in styles)
    assert len({fill, est, track}) == 3, (fill, est, track)


def test_a_meter_in_the_terminals_own_colors_keeps_three_steps(monkeypatch):
    monkeypatch.setattr(theme, "_query_terminal_colors", lambda: (None, None))
    pal = resolve(depth="truecolor")
    assert pal.adaptive
    assert (pal.sgr("mf"), pal.sgr("me"), pal.sgr("tk")) == ("39", "90", "2")
