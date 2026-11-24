"""Single source of truth for the Poros CLI color theme.

The CLI's palette was tuned for *dark* terminals: the light warm neutrals
(cream menu text, muted gray) wash out or vanish on a *white* background. This
module resolves a light/dark theme -- detected automatically by default -- and
exposes each color as a Rich ``color(N)`` token. Colors are 256-palette indices,
never 24-bit hex -- see the palette note below for why.

The card UI (poros/cli/ui/style.py) goes further: ``terminal_colors`` asks the
terminal for its real foreground and background (OSC 10/11), and every grey is
mixed between those two, so a white, green or blue theme gets tiers of its own
colors. A terminal that answers neither is drawn in its own default colors.

Detection order (both hang-proof): (1) the ``COLORFGBG`` env var -- a plain env
read that never touches the tty (set by iTerm2, konsole, ...); (2) a *cached*
one-time OSC 11 terminal query for terminals that don't export it (e.g. macOS
Terminal). The OSC query only runs on a real tty, in cbreak mode with a short
``select`` timeout, always restoring terminal attrs, and is primed once at
shell startup before the REPL reads input -- so it cannot hang, corrupt the
terminal, or eat a keystroke. When neither resolves, ``auto`` falls back to the
historical **dark** look, and the user can always pin ``POROS_THEME=light`` or
use ``/theme``.

Respect ``NO_COLOR``: when set, the SGR accessors return ``""`` (uncolored);
Rich itself already drops color from the styles.

Torch-free, rich-free: stdlib only.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from poros.palette import (
    ACCENT,
    console_color_system,
)

Mode = str  # "auto" | "dark" | "light"
Background = str  # "light" | "dark"

_VALID_MODES = ("auto", "dark", "light")

# ---------------------------------------------------------------------------
# Palettes, given as 256-color indices (not 24-bit hex).
#
# The CLI renders in 256-color, so we name the palette indices directly rather
# than converting from hex. The reason we don't use hex at all: a terminal (or
# Rich) downgrades a 24-bit hex to the nearest 256 color, and the nearest to our
# brand orange #FF6B35 is index 203 -- a pink. Index 208 is a clean orange and
# renders identically on 256- and truecolor terminals. Each line notes the
# source brand hex.
#
# ``accent`` is the Caistro orange in BOTH themes (it reads on light and dark);
# everything else flips for a light terminal.
# ---------------------------------------------------------------------------

_DARK = {
    "accent": 208,  # #FF6B35 cinnabar
    "text": 231,  # #F4F1EC parchment cream
    "dim": 241,  # warm muted gray
    "ok": 114,  # green
    "warn": 220,  # yellow
    "error": 203,  # soft red
}

_LIGHT = {
    "accent": 208,  # same orange -- reads on white too
    "text": 235,  # near-ink, dark on a white bg
    "dim": 243,  # warm mid gray
    "ok": 29,  # forest green (the dark green is faint on white)
    "warn": 130,  # amber (the dark yellow is invisible on white)
    "error": 124,  # deep red
}


# ---------------------------------------------------------------------------
# Mode resolution. ``set_mode`` (from ``/theme`` or ``--theme``) overrides the
# ``POROS_THEME`` env var, which itself defaults to ``auto`` when unset/invalid.
# ---------------------------------------------------------------------------

_mode_override: Mode | None = None


def current_mode() -> Mode:
    """The active theme mode: ``auto`` (default), ``dark``, or ``light``."""
    if _mode_override is not None:
        return _mode_override
    env = os.environ.get("POROS_THEME", "").strip().lower()
    return env if env in _VALID_MODES else "auto"


def set_mode(mode: str) -> Mode:
    """Set the theme mode live (for ``/theme`` and ``--theme``). Returns it.

    An unknown value resolves to ``auto`` rather than raising -- callers pass
    raw user input.
    """
    global _mode_override
    m = (mode or "").strip().lower()
    _mode_override = m if m in _VALID_MODES else "auto"
    return _mode_override


def next_mode(mode: str) -> Mode:
    """The next mode in the ``auto -> dark -> light -> auto`` cycle."""
    order = ["auto", "dark", "light"]
    try:
        i = order.index(mode)
    except ValueError:
        i = -1
    return order[(i + 1) % len(order)]


_UNSET = object()
_detected: object = _UNSET  # cached background from the one-time terminal query


def _reset_cache() -> None:
    """Clear the mode override + detection cache (tests only)."""
    global _mode_override, _detected, _colors, _answer
    _mode_override = None
    _detected = _UNSET
    _colors = _UNSET
    _answer = _UNSET


def forget_colors() -> None:
    """Drop what the terminal said about its colors, so the next palette asks
    again (it switched light/dark under a running session)."""
    global _detected, _colors, _answer
    _detected = _colors = _answer = _UNSET


# ---------------------------------------------------------------------------
# Background detection -- COLORFGBG only (a plain env read; cannot hang).
# ---------------------------------------------------------------------------


def _bg_field_is_light(bg: str) -> bool | None:
    """Interpret the *background* field of ``COLORFGBG``.

    Color indices ``7`` (light gray) and ``15`` (white) mean a light
    background; ``0``-``6`` / ``8``-``9`` mean dark. Some light terminals emit
    the literal ``default`` -- treat that as light. Anything else -> ``None``.
    """
    bg = bg.strip().lower()
    if bg == "default":
        return True
    try:
        n = int(bg)
    except ValueError:
        return None
    if n in (7, 15):
        return True
    if 0 <= n <= 15:
        return False
    return None


RGB = tuple  # (r, g, b), 0..255


def _parse_rgb(spec: str) -> "RGB | None":
    """An X11 color spec as terminals answer OSC 10/11: ``rgb:R/G/B`` with 1-4
    hex digits per channel, or ``#RRGGBB``."""
    import re

    m = re.match(r"rgba?:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})", spec)
    if m:
        return tuple(round(int(h, 16) * 255 / (16 ** len(h) - 1)) for h in m.groups())
    m = re.match(r"#([0-9a-fA-F]{6})", spec)
    if m:
        h = m.group(1)
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return None


def luminance(c: "RGB") -> float:
    """Relative luminance (WCAG), 0 black .. 1 white."""

    def lin(v: float) -> float:
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _parse_osc(resp: bytes) -> "tuple[RGB | None, RGB | None]":
    """(foreground, background) from the replies to OSC 10 and OSC 11."""
    import re

    text = resp.decode("latin-1", "ignore")
    got = {}
    for m in re.finditer(r"\x1b\]1([01]);([^\x07\x1b]*)", text):
        got[m.group(1)] = _parse_rgb(m.group(2))
    return got.get("0"), got.get("1")


_colors: object = _UNSET  # cached (fg, bg) from the one-time terminal query


_answer: object = _UNSET  # the raw reply to the one-time terminal query


def _ask() -> bytes:
    """The terminal's reply to the one query, asked once and kept."""
    global _answer
    if _answer is _UNSET:
        _answer = _ask_terminal()
    return _answer  # type: ignore[return-value]


def _query_terminal_colors() -> "tuple[RGB | None, RGB | None]":
    """The terminal's real foreground and background (OSC 10, 11)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None, None
    return _parse_osc(_ask())


@dataclass(frozen=True)
class Graphics:
    """What the terminal says it can draw: kitty graphics, sixel, and the
    size of a cell in pixels (None when it won't say)."""

    kitty: bool = False
    sixel: bool = False
    cell: "tuple[int, int] | None" = None


def terminal_graphics() -> Graphics:
    """From the same one-time query as the colours: a kitty graphics probe,
    the cell size (CSI 16 t) and the device attributes (4 = sixel)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return Graphics()
    import re

    resp = _ask()
    da = re.search(rb"\x1b\[\?([0-9;]*)c", resp)
    cell = re.search(rb"\x1b\[6;(\d+);(\d+)t", resp)
    return Graphics(kitty=b"_Gi=31;OK" in resp,
                    sixel=bool(da) and b"4" in da.group(1).split(b";"),
                    cell=(int(cell.group(2)), int(cell.group(1))) if cell else None)


def _ask_terminal() -> bytes:
    """Ask the terminal, once: its colours (OSC 10, 11), whether it speaks
    kitty graphics, its cell size, and its device attributes (DA1).

    Hang-proof: only on a real tty; cbreak mode, attrs always restored. DA1
    goes last: every terminal answers it, so the read stops as soon as that
    reply arrives instead of waiting out a timeout on terminals that ignore
    the other questions."""
    from poros.core.env import under_agent

    if under_agent() or not (sys.stdin.isatty() and sys.stdout.isatty()):  # no one to answer
        return b""
    try:
        import select
        import termios
        import time
        import tty
    except Exception:
        return b""
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return b""
    resp = b""
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        # Apple's Terminal shows an APC string as text: it gets no kitty probe
        kitty = ("" if os.environ.get("TERM_PROGRAM") == "Apple_Terminal"
                 else "\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\")
        sys.stdout.write(f"\x1b]10;?\x1b\\\x1b]11;?\x1b\\{kitty}\x1b[16t\x1b[c")
        sys.stdout.flush()
        end = time.monotonic() + 0.4
        import re

        while time.monotonic() < end:
            r, _w, _e = select.select([fd], [], [], max(end - time.monotonic(), 0))
            if not r:
                break
            chunk = os.read(fd, 256)
            if not chunk:
                break
            resp += chunk
            if re.search(rb"\x1b\[\?[0-9;]*c", resp):  # the DA1 reply: nothing more is coming
                break
    except Exception:
        return resp
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, old)
        except Exception:
            pass
    return resp


def terminal_colors() -> "tuple[RGB | None, RGB | None]":
    """The terminal's own (foreground, background), each None when it does
    not say. Queried once and cached; never under NO_COLOR."""
    global _colors
    if _no_color():
        return None, None
    if _colors is _UNSET:
        _colors = _query_terminal_colors()
    return _colors  # type: ignore[return-value]


def _query_terminal_background() -> Background | None:
    """light/dark from the terminal's real background, or None."""
    bg = terminal_colors()[1]
    if bg is None:
        return None
    return "light" if luminance(bg) > 0.18 else "dark"


def detect_background() -> Background | None:
    """Terminal background: ``"light"``, ``"dark"``, or ``None``.

    ``COLORFGBG`` first (a plain env read, instant); otherwise a *cached*
    one-time OSC 11 terminal query (for terminals that don't export it, e.g.
    macOS Terminal). The result is cached so repeated calls -- one per menu
    render -- never re-query the tty. Returns ``None`` only when nothing can
    determine it, and ``auto`` then falls back to dark.
    """
    global _detected
    val = os.environ.get("COLORFGBG")
    if val:
        light = _bg_field_is_light(val.split(";")[-1])
        if light is not None:
            return "light" if light else "dark"
    if _detected is _UNSET:
        _detected = _query_terminal_background()
    return _detected  # type: ignore[return-value]


def prime_detection() -> None:
    """Run + cache detection once, now. Call at shell startup, BEFORE the REPL
    menu, so the OSC 11 query happens at a controlled moment (no keystroke to
    eat) rather than lazily mid-render.

    Skipped entirely when the outcome cannot matter: an explicit dark/light
    mode discards the detection, and NO_COLOR means no ANSI should be
    written at all."""
    global _detected
    if current_mode() != "auto" or _no_color():
        return
    terminal_colors()  # the real tones, for palettes built from them
    if not os.environ.get("COLORFGBG") and _detected is _UNSET:
        _detected = _query_terminal_background()


# ---------------------------------------------------------------------------
# Resolved palette
# ---------------------------------------------------------------------------


def is_dark() -> bool:
    """True when the dark palette is active.

    ``dark``/``light`` modes are explicit. ``auto`` detects the background and
    -- crucially -- falls back to **dark** (today's look) when detection is
    inconclusive, so undetectable terminals see no regression.
    """
    mode = current_mode()
    if mode == "dark":
        return True
    if mode == "light":
        return False
    if _no_color():
        # No color means the background never matters -- and the OSC 11 probe
        # reads stdin, so running it lazily mid-session (say, inside a command
        # dispatched from the shell) would swallow type-ahead keystrokes.
        return True
    return detect_background() != "light"  # None (undetected) -> dark


def palette() -> dict[str, int]:
    """The active palette (dark or light) as ``{role: 256-index}``."""
    return _DARK if is_dark() else _LIGHT


def _no_color() -> bool:
    return "NO_COLOR" in os.environ


# ---------------------------------------------------------------------------
# Color output. Each palette color is offered in two forms:
#   - a Rich style token  ``color(N)``     -- for ``console.print`` styles/markup
#   - a raw SGR sequence  ``\x1b[38;5;Nm``  -- for the shell's raw-mode menu
# Both are plain 256-color; see the palette note for why we never emit 24-bit.
# ---------------------------------------------------------------------------


RESET = "\x1b[0m"


def _rich(idx: int) -> str:
    return f"color({idx})"


def accent() -> str:
    return _rich(palette()["accent"])


def text() -> str:
    return _rich(palette()["text"])


def dim() -> str:
    return _rich(palette()["dim"])


def ok() -> str:
    return _rich(palette()["ok"])


def warn() -> str:
    return _rich(palette()["warn"])


def error() -> str:
    return _rich(palette()["error"])


def apply_typer_help_theme() -> None:
    """Brand typer's --help rendering; its defaults are cyan (off-brand).

    Overrides only rich style strings on typer.rich_utils -- layout constants
    stay untouched, and a typer that has moved these attributes degrades to
    its defaults rather than crashing.
    """
    try:
        from typer import rich_utils
    except Exception:  # pragma: no cover - typer always present for the CLI
        return
    styles = {
        "STYLE_OPTION": f"bold {ACCENT}",
        "STYLE_SWITCH": f"bold {ACCENT}",
        "STYLE_COMMANDS_TABLE_FIRST_COLUMN": f"bold {ACCENT}",
        "STYLE_USAGE": f"bold {ACCENT}",
        "STYLE_USAGE_COMMAND": "bold",
        "STYLE_METAVAR": "dim",
        "STYLE_METAVAR_SEPARATOR": "dim",
        "STYLE_OPTION_DEFAULT": "dim",
        "STYLE_OPTION_ENVVAR": "dim",
    }
    for name, style in styles.items():
        if hasattr(rich_utils, name):
            setattr(rich_utils, name, style)
    # Typer builds its own Console for --help, which re-detects terminal
    # capabilities -- on a 16-color TERM (or TERM unset over ssh) Rich
    # downgrades color(208) to bright red. Pin the same color system our
    # own console uses so help never falls back off-palette.
    if hasattr(rich_utils, "COLOR_SYSTEM"):
        rich_utils.COLOR_SYSTEM = console_color_system()
