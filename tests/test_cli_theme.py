"""Light/dark theme detection + resolution for the Poros CLI.

Covers the pure, tty-free logic: ``COLORFGBG`` parsing, ``POROS_THEME``
resolution, the ``/theme`` cycle, the palette flip, and -- crucially -- that
detection never touches the terminal (no probe bytes) and a non-tty invocation
does not hang.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from poros.cli import theme

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI = [sys.executable, "-m", "poros.cli.main"]


@pytest.fixture(autouse=True)
def _clean_theme_state(monkeypatch):
    """Isolate every test from ambient theme env + cached detection."""
    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.delenv("POROS_THEME", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    theme._reset_cache()
    yield
    theme._reset_cache()


# ---------------------------------------------------------------------------
# COLORFGBG parsing.  Format is fg;bg (or fg;default;bg) -- the BACKGROUND is
# the LAST field, so "0;15" is light (bg 15) and "15;0" is dark (bg 0).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0;15", "light"),  # white bg -> light
        ("15;0", "dark"),  # order matters: bg is the LAST field
        ("0;7", "light"),  # light-gray bg -> light
        ("7;0", "dark"),
        ("15;default;0", "dark"),  # three-part: bg=0
        ("0;default;15", "light"),  # three-part: bg=15
        ("15;default", "light"),  # literal "default" bg -> light
        ("1;5", "dark"),  # bg 5 -> dark
        ("1;9", "dark"),  # bg 9 -> dark
    ],
)
def test_colorfgbg_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("COLORFGBG", value)
    assert theme.detect_background() == expected


def test_colorfgbg_light_drives_light_palette(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "0;15")
    theme._reset_cache()
    assert theme.is_dark() is False
    assert theme.palette()["text"] == 235  # dark-on-light ink (256 index)


def test_colorfgbg_dark_drives_dark_palette(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "15;0")
    theme._reset_cache()
    assert theme.is_dark() is True
    assert theme.palette()["text"] == 231  # the historical cream (256 index)


def test_colorfgbg_absent_is_none_without_tty(monkeypatch):
    # No COLORFGBG and stdio is not a tty under pytest -> undetermined.
    assert theme.detect_background() is None


# ---------------------------------------------------------------------------
# POROS_THEME env + set_mode override
# ---------------------------------------------------------------------------


def test_poros_theme_env_dark(monkeypatch):
    monkeypatch.setenv("POROS_THEME", "dark")
    assert theme.current_mode() == "dark"
    assert theme.is_dark() is True


def test_poros_theme_env_light(monkeypatch):
    monkeypatch.setenv("POROS_THEME", "light")
    assert theme.current_mode() == "light"
    assert theme.is_dark() is False


def test_poros_theme_env_invalid_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("POROS_THEME", "banana")
    assert theme.current_mode() == "auto"


def test_poros_theme_unset_is_auto(monkeypatch):
    assert theme.current_mode() == "auto"


def test_auto_falls_back_to_dark_when_undetectable(monkeypatch):
    # auto + no COLORFGBG + no tty -> detection None -> DARK (no regression).
    assert theme.current_mode() == "auto"
    assert theme.is_dark() is True


def test_set_mode_overrides_env(monkeypatch):
    monkeypatch.setenv("POROS_THEME", "dark")
    theme.set_mode("light")
    assert theme.current_mode() == "light"
    assert theme.is_dark() is False


def test_set_mode_invalid_becomes_auto():
    assert theme.set_mode("nonsense") == "auto"


# ---------------------------------------------------------------------------
# /theme cycle logic
# ---------------------------------------------------------------------------


def test_next_mode_cycles():
    assert theme.next_mode("auto") == "dark"
    assert theme.next_mode("dark") == "light"
    assert theme.next_mode("light") == "auto"
    # unknown -> restart the cycle at auto
    assert theme.next_mode("???") == "auto"


def test_theme_cycle_via_set_mode():
    theme.set_mode("auto")
    for expected in ("dark", "light", "auto"):
        theme.set_mode(theme.next_mode(theme.current_mode()))
        assert theme.current_mode() == expected


# ---------------------------------------------------------------------------
# Palette + SGR
# ---------------------------------------------------------------------------


def test_accent_identical_across_themes():
    theme.set_mode("dark")
    dark_accent = theme.accent()
    theme.set_mode("light")
    assert theme.accent() == dark_accent  # accent identical across themes
    assert theme.palette()["accent"] == 208  # the brand cinnabar (256 index)


# ---------------------------------------------------------------------------
# HARD CONSTRAINT: detection is a pure env read -- it must never touch the tty
# (no OSC probe bytes) and a non-tty run must not hang.
# ---------------------------------------------------------------------------


def _run_piped(args, env_extra=None, timeout=30):
    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        CLI + args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        env=env,
    )


def test_non_tty_check_does_not_emit_osc_and_does_not_hang():
    # `poros check --help` renders through the theme; with piped stdio the OSC 11
    # probe must be skipped entirely (no `\x1b]11` bytes) and it must not hang.
    proc = _run_piped(["check", "--help"], timeout=30)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert b"\x1b]11" not in combined  # never queried the terminal
    assert b"]11;?" not in combined


def test_non_tty_bare_poros_light_colorfgbg_no_hang():
    # Even with a light COLORFGBG set, a piped bare `poros` must exit fast with
    # help (no REPL, no OSC, no hang).
    proc = _run_piped([], env_extra={"COLORFGBG": "0;15"}, timeout=30)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0  # the help, as `poros --help`
    assert b"\x1b]11" not in combined
