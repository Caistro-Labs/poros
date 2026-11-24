"""Styled text as plain data: a ``Line`` is a list of ``(text, style)`` segments.

Screens build lines; ``render`` turns them into ANSI for one palette. Keeping
layout separate from escape codes is what lets every screen be tested on CPU
as plain text and rendered identically in color, 256-color and NO_COLOR.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple, Union

from poros.cli.ui.style import Palette, Style
from poros.core.errors import CONTROL

Seg = Tuple[str, Style]
Line = List[Seg]
Part = Union[str, Seg, Line, None]

BAND: Seg = ("", "band")  # first segment of a line drawn on the focus band
PANEL: Seg = ("", "panel")  # what follows sits on the start box's filled ground
MARK = "\x00mark:"  # a raw segment standing for a logo (ui/logos): drawn when rendered


def banded(line: Line, w: int) -> Line:
    """``line`` on the focus band, padded to ``w`` columns."""
    return [BAND, *line, (" " * max(0, w - width(line)), None)]


def S(text: object, style: Style = "tx") -> Seg:
    return (str(text), style)


def ln(*parts: Part) -> Line:
    """Build a line from strings (``tx``), segments, nested lines; ``None`` skips."""
    out: Line = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, str):
            out.append((p, "tx"))
        elif isinstance(p, list):
            out.extend(p)
        else:
            out.append(p)
    return out


def width(line: Sequence[Seg]) -> int:
    return sum(len(t) for t, st in line if st != "raw")


def plain(line: Sequence[Seg]) -> str:
    """A line with no styling, as a pipe or NO_COLOR gets it: still with no one's escapes."""
    return CONTROL.sub(_gone, "".join(t for t, st in line if st != "raw")).rstrip()


def merge(line: Iterable[Seg]) -> Line:
    out: Line = []
    for t, s in line:
        if out and out[-1][1] == s:
            out[-1] = (out[-1][0] + t, s)
        else:
            out.append((t, s))
    return out


def _gone(control: re.Match) -> str:
    return " " if control.group() in "\t\n\r" else ""  # words stay apart


def render_line(line: Sequence[Seg], pal: Palette) -> str:
    if PANEL in line:
        return _render_panel(line, pal)
    bg = pal.band() if line and line[0] == BAND else None
    if bg is None:
        line = [seg for seg in line if seg != BAND]
    buf = []
    ruled = None  # the style of the underlined rule just drawn
    for t, st in line:
        if st == "band":
            continue
        if st == "raw":  # a terminal escape that takes no cells (an image)
            if t.startswith(MARK):  # a logo, drawn in this palette's inks
                from poros.cli.ui import logos

                buf.append(logos.escape(t, pal))
            elif pal.depth not in ("none", "plain"):
                buf.append(t)
            continue
        t = CONTROL.sub(_gone, t)  # a file's or a run's words never steer the terminal
        code = pal.sgr(st) if t.strip() else None
        if pal.drawn_rules and t and not t.strip("─━"):
            # Terminal.app gives a whole underlined run its first cell's
            # color, so a rule changing color breaks for one cell
            gap = " " if ruled not in (None, st) else ""
            ruled, t = st, " " * (len(t) - len(gap))
            code = f"4;{code}" if code else "4"
            if gap:
                buf.append(gap)
        else:
            ruled = None
        if bg:
            code = f"{code};{bg}" if code else bg
        buf.append(f"\x1b[{code}m{t}\x1b[0m" if code else t)
    return "".join(buf) if bg else "".join(buf).rstrip()


def _render_panel(line: Sequence[Seg], pal: Palette) -> str:
    """A start-box row. With the ground known, the box is a filled panel:
    everything inside sits on it and its top and bottom edges draw nothing
    (the rows of air inside pad it; half blocks would leave a seam wherever
    the terminal draws them from the font). Without, the edges stay a faint
    line."""
    at = line.index(PANEL)
    ground = pal.panel()
    if ground is None:
        return render_line([seg for seg in line if seg != PANEL], pal)
    bg = pal.ground_code(ground)
    buf = [t for t, _ in line[:at]]  # the margin: blank, on the terminal's own ground
    for t, st in line[at + 1:]:
        if not t:
            continue
        if st in ("et", "eb"):
            return ""
        if st == "es":
            t, st = " " * len(t), None
        code = pal.sgr(st) if t.strip() else None
        if code and pal.drawn_rules and "█" in t and "38;" in code:
            # Terminal.app draws █ from the font, a seam under every row:
            # the mark's solid cells are filled with color instead
            fill = code.replace("38;", "48;", 1)
            t = t.replace("▀", "█")  # a half block is font-drawn too: round it up
            for run in re.split("(█+)", t):
                if run.startswith("█"):
                    buf.append(f"\x1b[{fill}m{' ' * len(run)}\x1b[0m")
                elif run:
                    buf.append(f"\x1b[{code};{bg}m{run}\x1b[0m")
            continue
        buf.append(f"\x1b[{code};{bg}m{t}\x1b[0m" if code else f"\x1b[{bg}m{t}\x1b[0m")
    return "".join(buf)


def render(lines: Sequence[Sequence[Seg]], pal: Palette) -> str:
    return "\n".join(render_line(line, pal) for line in lines)
