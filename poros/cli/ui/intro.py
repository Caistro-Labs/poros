"""The shell's intro: the v2 hero's river threading the arch mark.

Streams in the flow spectrum fall from above, converge into the one-column
passage between the mark's pillars, pass through, and fan out of the arch
below -- the caistrolabs.com hero, in a terminal. Then the mark resolves
from its faint outline to ink, top-down, the words type in beside it, and
the river drains. ~1.8 s; any key skips; the last frame is exactly the
static banner, so nothing jumps.

Pure frames (``frame(t, w)`` -> lines) plus a small player. Stdlib only.
"""

from __future__ import annotations

import math
import os
import random
import sys
import textwrap
import time
from typing import Callable, List, Optional, Sequence

from poros.cli.ui import components as c
from poros.cli.ui.style import TIERS
from poros.cli.ui.text import PANEL, Line, merge

DURATION = 2.0
T_DRAIN = 1.05  # no new pass enters the field after this
FPS = 30
TOP = 3  # stream rows above the mark
LEFT = 2  # mark column offset (aligns with the hint line)
GAP = 6  # the passage: the mark's one empty column in its top rows
TEXT_COL = LEFT + len(c.MARK[0]) + 5
ROWS = TOP + len(c.MARK)  # the river exits through the arch on the mark's last rows
SIDE = 70  # from here the words sit beside the mark; narrower, they go under it
MIN_WIDTH = 30  # narrower than this the intro is skipped for the one-line welcome


def _smooth(a: float, b: float, x: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    x = (x - a) / (b - a)
    return x * x * (3 - 2 * x)


class _Stream:
    __slots__ = ("x0", "speed", "phase", "born")

    def __init__(self, rng: random.Random, i: int, n: int) -> None:
        # spread across the field, edges wider; born staggered so the river arrives
        u = (i + 0.5) / n * 2 - 1
        self.x0 = u * 15 + rng.uniform(-0.8, 0.8)
        self.speed = rng.uniform(9.0, 13.0)  # rows per second
        self.phase = rng.uniform(0, ROWS + 3)
        self.born = rng.uniform(0.0, 0.5)


def _x_at(s: _Stream, y: float) -> float:
    """Column offset from the passage at height y (0 = top of field)."""
    if y < TOP + 0.5:  # converge into the passage
        k = 1 - _smooth(-1.0, TOP + 0.5, y)
        return s.x0 * k
    m = y - TOP  # inside the mark: straight through the gap, then the arch opens
    if m < 2:
        return 0.0
    spread = _smooth(2.0, len(c.MARK), m)
    return s.x0 * 0.32 * spread


_STREAMS: Optional[List[_Stream]] = None


def _streams() -> List[_Stream]:
    global _STREAMS
    if _STREAMS is None:
        rng = random.Random(7)
        n = 30
        _STREAMS = [_Stream(rng, i, n) for i in range(n)]
    return _STREAMS


def _mark_cell(r: int, x: int) -> bool:
    row = c.MARK[r]
    return 0 <= x < len(row) and row[x] != " "


def _field(t: float, width: int) -> List[List[tuple]]:
    """The river and the mark at time ``t``: ROWS rows of (char, style) cells."""
    grid: List[List[tuple]] = [[(" ", None)] * width for _ in range(ROWS)]
    heat = [[0.0] * width for _ in range(ROWS)]
    hue = [[0.0] * width for _ in range(ROWS)]

    # the river: flows, then drains -- a segment whose pass through the field
    # would begin after T_DRAIN never appears, so the last ones fall out
    cx = LEFT + GAP
    span = ROWS + 4
    for s in _streams():
        if t < s.born:
            continue
        travel = (t - s.born) * s.speed
        for k in range(4):  # head + trail
            p = s.phase + travel - k * 0.9  # unwrapped distance along the field
            cycle = math.floor(p / span)
            pass_start = s.born + (cycle * span - s.phase + k * 0.9) / s.speed
            if cycle > 0 and pass_start > T_DRAIN:
                continue
            y = p - cycle * span - 2
            if y < -0.5 or y >= ROWS:
                continue
            x = int(round(cx + _x_at(s, y)))
            r = int(y)
            if not (0 <= x < width) or not (0 <= r < ROWS):
                continue
            mr = r - TOP
            if 0 <= mr < len(c.MARK) and _mark_cell(mr, x - LEFT):
                continue  # the river passes through the openings, never the stone
            v = 1.0 - k * 0.28
            if v > heat[r][x]:
                heat[r][x] = v
                hue[r][x] = 1 - min(abs(s.x0) / 15.5, 1.0)  # edges blue-violet, core gold

    for r in range(ROWS):
        for x in range(width):
            v = heat[r][x]
            if v <= 0.05:
                continue
            ch = "┃" if v > 0.8 else "│" if v > 0.55 else "╎" if v > 0.3 else "·"
            grid[r][x] = (ch, ("sp", hue[r][x], 1 - v))

    # the mark: faint outline, then resolves to ink top-down
    for mr, row in enumerate(c.MARK):
        lit = _smooth(0.75 + mr * 0.07, 1.1 + mr * 0.07, t)
        for i, ch in enumerate(row):
            if ch == " ":
                continue
            x = LEFT + i
            if x < width:
                grid[TOP + mr][x] = (ch, ("mk", lit))

    return grid


def frame(t: float, w: int, right: Sequence[Line]) -> List[Line]:
    """One intro frame at time ``t`` (seconds). ``right``: the 5 text rows."""
    width = min(w, c.CAP)
    grid = _field(t, width)
    # words beside the mark (only when there is room)
    out: List[Line] = []
    for r in range(ROWS):
        line = merge(grid[r])
        mr = r - TOP
        if 0 <= mr < len(right) and w >= SIDE:
            line = _overlay(line, TEXT_COL, _typed(right[mr], mr, t), width)
        out.append(line)
    if w < SIDE:  # a narrow terminal: the words type in under the mark
        out.append([])
        for mr, text_line in enumerate(right[:-1]):
            if mr == 1:
                continue  # the gap under the name belongs to the side layout
            out.append(_overlay([], LEFT, _typed(text_line, mr, t), width))
    return out


def _typed(text_line: Line, mr: int, t: float, step: float = 0.08) -> Line:
    """``text_line`` as far as it has typed in at time ``t`` (row ``mr``)."""
    reveal = _smooth(1.05 + mr * step, 1.45 + mr * step, t)
    left_n = int(round(sum(len(s_) for s_, _ in text_line) * reveal))
    shown: Line = []
    for s_, st in text_line:
        if left_n <= 0:
            break
        shown.append((s_[:left_n], st))
        left_n -= len(s_)
    return shown


def _faded(text_line: Line, mr: int, t: float, step: float = 0.07) -> Line:
    """``text_line`` at time ``t``: each row fades in whole as the mark row
    beside it lights, top down; settled, it is the line itself."""
    k = _smooth(0.95 + mr * step, 1.2 + mr * step, t)
    if k <= 0:
        return [(" " * len(s_), None) for s_, _ in text_line]
    if k >= 1:
        return list(text_line)
    return [(s_, ("fd", st, k) if st in TIERS else st) for s_, st in text_line]


def _overlay(base: Line, col: int, add: Line, width: int) -> Line:
    flat = [(ch, st) for s_, st in base for ch in s_]
    flat += [(" ", None)] * max(0, col - len(flat))
    for s_, st in add:
        for ch in s_:
            if col < width:
                if col < len(flat):
                    flat[col] = (ch, st)
                else:
                    flat.append((ch, st))
            col += 1
    return merge(flat)


def right_column(line1: str, line2: str) -> List[Line]:
    return [
        [("poros", "hi")],  # the home screen names no version: poros --version does
        [],
        [(line1, "tx")],
        [(line2, "dm")],
        [],
    ]


def final(w: int, line1: str, line2: str) -> List[Line]:
    """The settled banner -- identical to the intro's last frame."""
    if w >= BOX_MIN:
        return box_frame(DURATION + 1.0, w, box_column(line1, line2, box_room(w)))
    return frame(DURATION + 1.0, w, right_column(line1, line2))


# -- the start box: the mark on the left, the name and what to do beside it --

BOX_W = 76  # the box, border to border, where there is room
BOX_MIN = 57  # narrower terminals keep the open banner (the commands need 29 columns)
PAD = 2  # columns between the border and what it holds, on each side
# What the box offers, and the key that does it: each one works on the
# prompt under it (a letter starts its slash command, ⏎ opens the list).
BOX_COMMANDS = (
    ("Start from a recipe", "/recipes"),
    ("Set up your own run", "/train"),
    ("Watch your runs", "/runs"),
    ("Every command", "⏎"),
    ("Quit", "ctrl-d"),  # as key_name says it here
)


def box_width(w: int) -> int:
    """The box narrows with the terminal, down to what BOX_MIN leaves."""
    return min(BOX_W, w - 2)


def box_room(w: int) -> int:
    """Columns for the words beside the mark."""
    return box_width(w) - 2 - PAD - TEXT_COL - PAD


def balanced(text: str, room: int) -> List[str]:
    """``text`` wrapped in ``room`` columns into lines of even length: the
    fewest lines it needs, at the narrowest width that still takes that many,
    so the last line is never one word left alone."""
    lines = textwrap.wrap(text, room) or [""]
    for narrower in range(room - 1, 0, -1):
        tried = textwrap.wrap(text, narrower)
        if len(tried) > len(lines):
            break
        lines = tried
    return lines


def box_column(line1: str, line2: str, room: int) -> List[Line]:
    """The words beside the mark in the box, ``room`` columns wide: the name,
    what Poros does, this machine, then the commands with their keys."""
    rows: List[Line] = [
        [("poros", "hi")],
        [],
        *[[(part, "tx")] for part in balanced(line1, room)],  # wraps in a narrow box
        *[[(part, "dm")] for part in balanced(line2, room)],
        [],
    ]
    for label, key in BOX_COMMANDS:
        key = c.key_name(key)
        gap = max(2, room - len(label) - len(key))
        rows.append([(label, "tx"), (" " * gap, None), (key, "dm")])
    return rows


def _box_top(right: Sequence[Line]) -> int:
    return 1 + (len(right) - len(c.MARK) + 1) // 2 - TOP  # the mark sits mid-box


def box_mark_row(right: Sequence[Line]) -> int:
    """The line of ``box_frame`` the mark's top row is on."""
    return 1 + _box_top(right) + TOP


def box_frame(t: float, w: int, right: Sequence[Line],
              field: Optional[Callable[[float, int], List[List[tuple]]]] = None) -> List[Line]:
    """One frame of the boxed intro: the river falls inside the box, the mark
    resolves in it, the words fade in beside it row by row. Centred in the
    terminal; ``text.PANEL`` lets a terminal that reports its colors draw the
    box as a filled panel. ``field`` draws the river and mark in place of
    ``_field`` (the turning entrance draws only the mark)."""
    bw = box_width(w)
    inner = bw - 2
    field = (field or _field)(t, inner - PAD)
    n = len(right) + 2  # a row of air above and below
    top = _box_top(right)
    margin = [(" " * max(0, (w - bw) // 2), None)]
    out: List[Line] = [margin + [PANEL, ("╭" + "─" * inner + "╮", "et")]]
    for i in range(n):
        f = i - top
        line = merge(field[f]) if 0 <= f < ROWS else []
        if 1 <= i <= len(right):
            line = _overlay(line, TEXT_COL, _faded(right[i - 1], i - 1, t), inner - PAD)
        line = [(" " * PAD, None)] + line
        pad = inner - sum(len(s_) for s_, _ in line)
        out.append(margin + [PANEL, ("│", "es")] + line + [(" " * pad, None), ("│", "es")])
    out.append(margin + [PANEL, ("╰" + "─" * inner + "╯", "eb")])
    return out


def frames(w: int, line1: str, line2: str) -> Callable[[float], List[Line]]:
    """The intro at time ``t``: boxed when there is room, open otherwise. On
    the full screen, where the terminal shows images, the mark turns in
    (intro_turn); everywhere else the river brings it."""
    from poros.cli.ui import intro_turn

    turn = intro_turn.maybe(w, line1, line2)
    if turn is not None:
        return turn
    if w >= BOX_MIN:
        right = box_column(line1, line2, box_room(w))
        return lambda t: box_frame(t, w, right)
    right = right_column(line1, line2)
    return lambda t: frame(t, w, right)


def run(show: Callable[[List[Line]], None], draw: Callable[[float], List[Line]]) -> None:
    """Hand ``show`` each frame for ``DURATION`` (or the ``duration`` the
    frames carry; any key skips), then the final one. Keys typed meanwhile
    are dropped, so none lands in the prompt."""
    duration = getattr(draw, "duration", DURATION)
    fd = None
    old = None
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd, termios.TCSANOW)
    except Exception:
        fd = None
    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            done = t >= duration or _key_pressed(fd)
            show(draw(duration + 1.0 if done else t))
            if done:
                break
            time.sleep(1 / FPS)
    except KeyboardInterrupt:
        show(draw(duration + 1.0))
    finally:
        if fd is not None and old is not None:
            import termios

            termios.tcsetattr(fd, termios.TCSAFLUSH, old)


def play(stream, pal, w: int, line1: str, line2: str) -> None:
    """Animate on a tty (any key skips), then leave the final frame."""
    from poros.cli.ui.live import SYNC_OFF, SYNC_ON, up
    from poros.cli.ui.text import render_line

    drawn = [0]

    def show(rows: List[Line]) -> None:
        if drawn[0]:
            stream.write(up(drawn[0]))
        stream.write(SYNC_ON + "".join(render_line(r, pal) + "\x1b[K\n" for r in rows) + SYNC_OFF)
        stream.flush()
        drawn[0] = len(rows)

    stream.write("\x1b[?25l")
    try:
        run(show, frames(w, line1, line2))
    finally:
        stream.write("\x1b[?25h")
        stream.flush()


def _key_pressed(fd: Optional[int]) -> bool:
    if fd is None:
        return False
    import select

    r, _, _ = select.select([fd], [], [], 0)
    if r:
        try:
            data = os.read(fd, 64)
        except OSError:
            return False
        # the OSC 11 theme reply can arrive late; it is not a keypress
        return not data.startswith(b"\x1b]")
    return False


def enabled(stream) -> bool:
    isatty = getattr(stream, "isatty", None)
    return (
        bool(isatty and isatty())
        and sys.stdin.isatty()
        and not os.environ.get("POROS_NO_ANIM")
        and os.environ.get("TERM", "") != "dumb"
    )
