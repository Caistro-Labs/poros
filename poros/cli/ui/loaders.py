"""Loading states, handcrafted for the terminal (after loading.dev's set).

Each loader is a pure function ``frame(t) -> Line`` drawing a fixed 3-cell
slot, so swapping loaders or ticking never shifts the label beside it. ``t``
is the surface frame at 10 fps. Every loader has a still frame for
``POROS_NO_ANIM`` / reduced motion / logs (``still(kind)``).

Which loader means what -- one motion per kind of wait:

    river   loading weights        a comet runs a short track, the flow spectrum
                                   fading behind a gold head (the hero's river)
    wave    training               five ... three bars rising and falling in a
                                   phase-offset wave, in the accent
    dots    saving / writing       three dots lighting in turn (linear dots)
    trace   evaluating             one braille dot tracing a square, two-dot
                                   fading trail (trace / swirl)
    leap    merging / exporting    three dots; the last leaps to the front
    ripple  resolving / network    a ring spreading out from a point (ripple)
    mark    starting up            the arch mark breathing ink <-> faint
"""

from __future__ import annotations

import math
from typing import Callable, Dict

from poros.cli.ui.text import Line

WIDTH = 3


def _pad(line: Line) -> Line:
    w = sum(len(t) for t, _ in line)
    return line + [(" " * (WIDTH - w), None)] if w < WIDTH else line


def river(t: int) -> Line:
    """A comet on a 3-cell track: gold head, spectrum tail, sub-cell motion
    via half blocks so it glides instead of stepping."""
    track = 6  # half-cell positions
    head = t % (track + 3)
    cells = []
    for i in range(WIDTH):
        left, right = 2 * i, 2 * i + 1
        on_l = 0 <= head - left < 3
        on_r = 0 <= head - right < 3
        age = min((head - left) if on_l else 9, (head - right) if on_r else 9)
        ch = "━" if (on_l and on_r) else "╸" if on_l else "╺" if on_r else "─"
        if ch == "─":
            cells.append((ch, "tk"))
        else:
            cells.append((ch, ("sp", 1.0 - age / 3.0, age / 4.0)))
    return cells


_BARS = "▁▂▃▄▅▆▇█"


def wave(t: int) -> Line:
    """Three bars rising and falling, each a step behind the last."""
    out = []
    for i in range(WIDTH):
        v = (math.sin((t * 0.55) - i * 0.9) + 1) / 2
        out.append((_BARS[min(7, int(v * 7.99))], "a"))
    return out


def dots(t: int) -> Line:
    """Three dots, the bright one walking left to right, then a beat of rest."""
    lit = t % 5
    out = []
    for i in range(WIDTH):
        st = "hi" if i == lit else ("dm" if i == lit - 1 else "rl")
        out.append(("•" if i == lit else "·", st))
    return out


# perimeter of a braille cell (dot bits), clockwise from top-left
_RING = [0x01, 0x08, 0x10, 0x20, 0x80, 0x40, 0x04, 0x02]


def trace(t: int) -> Line:
    """One braille dot tracing the cell's outline with a two-dot fading trail."""
    i = t % len(_RING)
    cell = chr(0x2800 | _RING[i] | _RING[i - 1] | _RING[i - 2])
    return _pad([(" ", None), (cell, "hi")])


_LOW, _MID, _TOP = "⡀", "⠄", "⠂"
# (cells, index of the dot in the air): the last dot rises, arcs over the
# other two, and lands in front -- identical dots, so it reads as a leap
_LEAP = [
    ((_LOW, _LOW, _LOW), None),
    ((_LOW, _LOW, _LOW), None),
    ((_LOW, _LOW, _MID), 2),
    ((_LOW, _LOW, _TOP), 2),
    ((_LOW, "⡂", " "), 1),
    ((_TOP, _LOW, _LOW), 0),
    ((_MID, _LOW, _LOW), 0),
    ((_LOW, _LOW, _LOW), None),
]


def leap(t: int) -> Line:
    """Three dots in a row; the last one leaps over the others to the front."""
    cells, air = _LEAP[t % len(_LEAP)]
    return [(ch, "hi" if k == air else "dm") for k, ch in enumerate(cells)]


_RIPPLE = ["·", "∘", "○", "◯", " "]


def ripple(t: int) -> Line:
    """A ring spreading out from a point, fading as it grows."""
    k = t % len(_RIPPLE)
    st = ["hi", "hi", "tx", "dm", None][k]
    return _pad([(" ", None), (_RIPPLE[k], st)])


def mark(t: int) -> Line:
    """The arch mark breathing: ink, through the dim tiers, and back."""
    v = (math.cos(t * 0.35) + 1) / 2
    return _pad([("▛▜", ("mk", 0.25 + 0.75 * v))])


LOADERS: Dict[str, Callable[[int], Line]] = {
    "river": river,
    "wave": wave,
    "dots": dots,
    "trace": trace,
    "leap": leap,
    "ripple": ripple,
    "mark": mark,
}

_STILL = {
    "river": "━╸─",
    "wave": "▃▅▇",
    "dots": "•··",
    "trace": " ⠏",
    "leap": "⡀⡀⡀",
    "ripple": " ○",
    "mark": "▛▜",
}


def still(kind: str) -> Line:
    st = "a" if kind == "wave" else "hi"
    return _pad([(_STILL.get(kind, "•"), st)])


def frame(kind: str, t: int, animate: bool = True) -> Line:
    fn = LOADERS.get(kind)
    if fn is None or not animate:
        return still(kind)
    return fn(t)


def for_label(label: str) -> str:
    """The loader that fits a wait, from the activity's own words."""
    lowered = label.lower()
    for words, kind in (
        (("loading", "downloading", "streaming"), "river"),
        (("training", "benchmarking", "running"), "wave"),
        (("saving", "writing"), "dots"),
        (("evaluating",), "trace"),
        (("merging", "exporting"), "leap"),
        (("checking", "resolving", "fetching"), "ripple"),
        (("starting",), "mark"),
    ):
        if lowered.startswith(words):
            return kind
    return "dots"
