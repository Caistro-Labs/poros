"""Tokens and color resolution for the Poros CLI (see the CLI visual design).

Every grey is a share of the terminal's foreground tone mixed toward its
ground -- the caistrolabs.com ``color-mix`` tiers -- so the hierarchy holds on
dark and light terminals alike. ``#FF6B35`` is the only accent; the flow
spectrum (sampled from the v2 hero) is reserved for things that flow.

Depth: truecolor when ``COLORTERM`` says so, else the nearest xterm-256 index
(the accent is pinned to 208: the nearest-index downgrade of ``#FF6B35`` is
203, a pink). ``NO_COLOR`` keeps bold and drops everything else.

Torch-free, rich-free: stdlib only.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Union

from poros.cli import theme

RGB = Tuple[int, int, int]

INK: RGB = (22, 22, 22)
PAPER: RGB = (252, 252, 252)
ACCENT: RGB = (255, 107, 53)
ERROR: RGB = (229, 72, 77)

SPECTRUM_DARK: tuple[RGB, ...] = (
    (110, 111, 176),
    (156, 138, 204),
    (230, 149, 174),
    (246, 160, 161),
    (255, 185, 167),
    (255, 221, 134),
    (255, 249, 149),
)
SPECTRUM_LIGHT: tuple[RGB, ...] = (
    (112, 122, 214),
    (160, 112, 196),
    (214, 108, 146),
    (226, 116, 98),
    (230, 136, 74),
    (222, 150, 44),
    (205, 128, 20),
)

# token -> share of the foreground tone (the site's color-mix tiers)
TIERS = {"hi": 1.0, "tx": 0.78, "dm": 0.50, "rl": 0.30, "tk": 0.17}

# The start box's edges: top, bottom, side. Where the terminal's colors are
# known the box is a filled panel and these become its half-row padding (see
# ``text.render_line``); where they are not, a faint line in the text color --
# never the theme's own grey, which Solarized sets to its ground.
EDGES = ("et", "eb", "es")

# A meter's fill against its ``tk`` track: ``mf`` a real figure, ``me`` an
# estimate. Each starts at its tier and steps toward the foreground until it
# stands apart from the track (and the estimate from the fill), so a
# low-contrast theme or a coarse 256-color ramp can't fold them together.
METER = {"me": (0.30, 1.30), "mf": (0.50, 1.70)}  # token: (share, least contrast to the track)

# A style is a token name, or a parametric pair: ("sp", 0..1) a point on the
# flow spectrum, ("sh", 0..1) a shimmer level between dim and hi. For a moment
# in motion: ("fd", style, 0..1) any style faded into the ground, ("ax", 0..1)
# text turning accent, ("gl", x, 0..1) a point on the spectrum lit by a glint.
Style = Union[str, Tuple[str, float], None]


def mix(fg: RGB, bg: RGB, share: float) -> RGB:
    return tuple(round(b + (f - b) * share) for f, b in zip(fg, bg))  # type: ignore[return-value]


def _spectrum_at(stops: tuple[RGB, ...], x: float) -> RGB:
    x = min(max(x, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    return mix(stops[i + 1], stops[i], x - i)


# --------------------------------------------------------------------------
# xterm-256 fallback
# --------------------------------------------------------------------------

_CUBE = (0, 95, 135, 175, 215, 255)


def _nearest_256(c: RGB) -> int:
    if c == ACCENT:
        return 208
    r, g, b = c
    if max(c) - min(c) < 12:  # grey ramp 232..255 (8..238), plus 16 / 231
        avg = (r + g + b) / 3
        if avg < 4:
            return 16
        if avg > 246:
            return 231
        return 232 + max(0, min(23, round((avg - 8) / 10)))

    def idx(v: int) -> int:
        return min(range(6), key=lambda i: abs(_CUBE[i] - v))

    return 16 + 36 * idx(r) + 6 * idx(g) + idx(b)


# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """Resolved output mode. ``depth``: ``truecolor`` | ``256`` | ``none`` | ``plain``.

    The greys are mixed between ``ink`` and ``ground``: the terminal's own
    foreground and background when it reports them, else Poros' ink and paper.
    ``adaptive``: the terminal did not say what its colors are, so text is
    drawn in its default foreground (bold, plain, faint) and the terminal
    renders it against whatever ground it has -- white, green, blue."""

    depth: str
    dark: bool
    ink: Optional[RGB] = None
    ground: Optional[RGB] = None
    adaptive: bool = False
    # Terminal.app draws ─ from the font, with a gap in every cell; there a
    # rule is drawn as underlined blanks, which the terminal draws whole
    drawn_rules: bool = False

    @property
    def fg(self) -> RGB:
        return self.ink or (PAPER if self.dark else INK)

    @property
    def bg(self) -> RGB:
        return self.ground or (INK if self.dark else PAPER)

    def _color(self, c: RGB) -> str:
        if self.depth == "truecolor":
            return "38;2;%d;%d;%d" % c
        return "38;5;%d" % _nearest_256(c)

    def sgr(self, st: Style) -> Optional[str]:
        """SGR parameter string for a style, or ``None`` for plain text."""
        if st is None or self.depth == "plain":
            return None
        if isinstance(st, tuple) and st[0] in ("fd", "ax", "gl"):
            return self._moving(st)
        if self.depth == "none":
            if st in ("hi", "a", "e") or (isinstance(st, tuple) and st[0] == "sh"):
                return "1"
            if isinstance(st, tuple) and st[0] == "mk":
                return "1" if st[1] > 0.5 else None
            if isinstance(st, tuple) and st[0] == "fd":
                return "1" if st[1] == "hi" and st[2] > 0.5 else None
            return None
        if self.adaptive:
            return self._adaptive(st)
        if isinstance(st, tuple):
            kind, v = st[0], st[1]
            if kind == "sp":  # flow spectrum; an optional third item fades it into the ground
                stops = SPECTRUM_DARK if self.dark else SPECTRUM_LIGHT
                col = _spectrum_at(stops, v)
                if len(st) > 2:
                    col = mix(self.bg, col, max(0.0, min(1.0, st[2])))
                return self._color(col)
            if kind == "mk":  # the mark resolving: track tier -> the brand
                lo = mix(self.fg, self.bg, TIERS["tk"])
                return ("1;" if v > 0.5 else "") + self._color(mix(self._legible(ACCENT), lo, v))
            if kind == "fd":  # a token fading in from the ground
                k = st[2]
                col = mix(mix(self.fg, self.bg, TIERS[v]), self.bg, k)
                return ("1;" if v == "hi" and k > 0.5 else "") + self._color(col)
            lo = mix(self.fg, self.bg, TIERS["dm"])
            return "1;" + self._color(mix(self.fg, lo, v))
        if st == "a":
            return self._color(self._legible(ACCENT))
        if st == "e":
            return self._color(self._legible(ERROR))
        if st in METER:
            return self._color(self._meter(st))
        share = 0.22 if st in EDGES else TIERS[st]
        code = self._color(mix(self.fg, self.bg, share))
        return "1;" + code if st == "hi" else code

    def _moving(self, st: tuple) -> Optional[str]:
        """A colour between two others. Where the terminal can't draw one it
        takes the nearer of the two: a fade steps, it never disappears early."""
        kind = st[0]
        if self.depth == "none" or self.adaptive:
            if kind == "fd":
                return self.sgr(st[1]) if st[2] >= 0.5 else None
            if kind == "ax":
                return self.sgr("a" if st[1] >= 0.5 else "tx")
            return self.sgr(("sp", st[1]))
        if kind == "fd":
            return self._color(mix(self._rgb(st[1]), self.bg, st[2]))
        if kind == "ax":
            return self._color(mix(self._legible(ACCENT), self._rgb("tx"), st[1]))
        return self._color(mix(self.fg, self._rgb(("sp", st[1])), st[2]))

    def _rgb(self, st: Style) -> RGB:
        if isinstance(st, tuple):
            if st[0] == "sp":
                stops = SPECTRUM_DARK if self.dark else SPECTRUM_LIGHT
                col = _spectrum_at(stops, st[1])
                return mix(self.bg, col, st[2]) if len(st) > 2 else col
            return self.fg
        if st == "a":
            return self._legible(ACCENT)
        if st == "e":
            return self._legible(ERROR)
        if st in METER:
            return self._meter(st)
        return mix(self.fg, self.bg, TIERS.get(st or "tx", 0.78))

    def _meter(self, st: str) -> RGB:
        track = mix(self.fg, self.bg, TIERS["tk"])
        floor = track
        for tok in ("me", "mf"):  # the estimate first: the fill must clear it too
            share, least = METER[tok]
            col = mix(self.fg, self.bg, share)
            while share < 1.0 and (_contrast(col, track) < least or self._same(col, floor)):
                share = min(1.0, share + 0.05)
                col = mix(self.fg, self.bg, share)
            if tok == st:
                return col
            floor = col
        return col

    def _same(self, a: RGB, b: RGB) -> bool:
        """Two colors the output can't tell apart: equal, or one xterm-256 index."""
        return a == b or (self.depth != "truecolor" and _nearest_256(a) == _nearest_256(b))

    def band(self) -> Optional[str]:
        """The focused row's ground: the foreground a few steps into the
        background. None where a ground can't be drawn without guessing."""
        if self.depth in ("none", "plain") or self.adaptive:
            return None
        col = mix(self.fg, self.bg, 0.08)
        return "48;2;%d;%d;%d" % col if self.depth == "truecolor" else "48;5;%d" % _nearest_256(col)

    def panel(self) -> Optional[RGB]:
        """The start box's ground: a step off the terminal's, lifted toward
        the foreground on a dark ground and sunk a hair on a light one. None
        where the ground is unknown, so a fill would be a guess."""
        if self.depth in ("none", "plain") or self.adaptive:
            return None
        return mix(self.fg, self.bg, 0.07 if self.dark else 0.045)

    def ground_code(self, c: RGB) -> str:
        """``c`` as a cell background."""
        return "48;2;%d;%d;%d" % c if self.depth == "truecolor" else "48;5;%d" % _nearest_256(c)

    def _legible(self, c: RGB) -> RGB:
        """A fixed color, unless the terminal's ground is too close to it
        (an orange theme): then the foreground, which always reads."""
        if self.ground is not None and _contrast(c, self.ground) < 2.0:
            return self.fg
        return c

    def _adaptive(self, st: Style) -> Optional[str]:
        """The terminal's own foreground, made strong or weak with bold and
        faint; only the accent, the error red and the spectrum are colors."""
        if isinstance(st, tuple):
            kind, v = st[0], st[1]
            if kind == "sp":
                col = self._color(_spectrum_at(SPECTRUM_DARK if self.dark else SPECTRUM_LIGHT, v))
                return ("2;" + col) if len(st) > 2 and st[2] < 0.6 else col
            if kind == "mk":
                return ("1;" if v > 0.5 else "2;") + self._color(ACCENT)
            if kind == "fd":  # fading in: faint, then the token itself
                return self._adaptive(v) if st[2] > 0.5 else "2"
            return "1" if v > 0.66 else ("39" if v > 0.33 else "2")  # shimmer
        if st == "a":
            return self._color(ACCENT)
        if st == "e":
            return self._color(ERROR)
        # rules and separators take the theme's own grey (bright black): faint
        # text at a hairline's weight all but vanishes on a white ground
        # a meter keeps three steps: the fill in the foreground, an estimate
        # in bright black, the track faint
        return {"hi": "1", "tx": "39", "rl": "90", "mf": "39", "me": "90"}.get(st, "2")


def _contrast(a: RGB, b: RGB) -> float:
    la, lb = theme.luminance(a), theme.luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _depth(stream) -> str:
    """``plain`` (no escapes: pipes, files, TERM=dumb), ``none`` (NO_COLOR on a
    terminal: bold only), ``256`` or ``truecolor``."""
    isatty = getattr(stream, "isatty", None)
    if os.environ.get("TERM", "") == "dumb":
        return "plain"
    if not (isatty and isatty()) and not os.environ.get("POROS_FORCE_COLOR"):
        return "plain"
    if "NO_COLOR" in os.environ:
        return "none"
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    return "256"


def resolve(stream=None, *, depth: Optional[str] = None, dark: Optional[bool] = None) -> Palette:
    """The palette for ``stream`` (default stdout). Arguments override detection."""
    pal = _resolve(stream, depth=depth, dark=dark)
    if pal.depth != "plain" and os.environ.get("TERM_PROGRAM") == "Apple_Terminal":
        pal = replace(pal, drawn_rules=True)
    return pal


def _resolve(stream=None, *, depth: Optional[str] = None, dark: Optional[bool] = None) -> Palette:
    stream = stream or sys.stdout
    d = depth or _depth(stream)
    if dark is not None or d in ("none", "plain"):
        # NO_COLOR never probes the terminal; the tone is irrelevant there
        return Palette(depth=d, dark=True if dark is None else dark)
    if theme.current_mode() != "auto":  # pinned with /settings or POROS_THEME
        return Palette(depth=d, dark=theme.is_dark())
    fg, bg = theme.terminal_colors()
    if bg is not None:
        ink = fg if fg is not None and _contrast(fg, bg) >= 3.0 else None
        is_dark = theme.luminance(bg) <= 0.18
        return Palette(depth=d, dark=is_dark, ink=ink or (PAPER if is_dark else INK), ground=bg)
    # The terminal won't say: draw in its own colors. The spectrum still
    # needs a side; COLORFGBG can give it, else the one that reads on both.
    return Palette(depth=d, dark=theme.detect_background() == "dark", adaptive=True)
