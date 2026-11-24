"""The frame a sheet and a list run in: the alternate screen (``fullscreen``), the key loop
(``_loop``), the canvas a card is drawn on, and the column widths both lay rows out with.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Literal, Optional

from poros.cli.ui import components as c
from poros.cli.ui.text import Line

METER_MIN = 6  # a meter narrows to this before a name or a note is cut


NAME_WHOLE = 16  # a name column cut to this keeps most names whole


NAME_MIN = 12  # and no name column is cut under this


Outcome = Literal["go", "cancel"]


def _fit(text: str, room: int) -> str:
    """``text`` in ``room`` columns: a ``a · b · c`` note loses whole parts
    from the right, anything else is cut at a word with an ellipsis."""
    if room <= 1:
        return ""
    if len(text) <= room:
        return text
    parts = text.split(" · ")
    while len(parts) > 1:
        parts.pop()
        if len(" · ".join(parts)) <= room:
            return " · ".join(parts)
    cut = text[: room - 1]
    if " " in cut[room // 2 :]:
        cut = cut[: cut.rindex(" ")].rstrip(" ,·:")
    return cut + "…"


TOP = 2  # rows of air above a card


_depth = 0


def full_screen_ok(stream) -> bool:
    """Cards take the full screen on a terminal, but not under Zellij or tmux
    control mode (their panes keep scrollback their own way), nor when asked
    not to (``POROS_INLINE=1``)."""
    import os

    isatty = getattr(stream, "isatty", None)
    return (bool(isatty and isatty()) and not os.environ.get("POROS_INLINE")
            and not os.environ.get("ZELLIJ") and os.environ.get("TERM", "") != "dumb")


@contextmanager
def fullscreen(stream=None):
    """The card canvas for everything inside; nested uses share it."""
    import sys

    global _depth
    stream = stream or sys.stdout
    from poros.cli.ui import tui

    if _depth or tui.current is not None:  # already on a canvas: this card shares it
        _depth += 1
        try:
            yield
        finally:
            _depth -= 1
        return
    if not full_screen_ok(stream):
        yield
        return
    _depth = 1
    stream.write("\x1b[?1049h\x1b[H\x1b[2J")
    stream.flush()
    try:
        yield
    finally:
        _depth = 0
        stream.write("\x1b[?1049l")
        stream.flush()


def canvas(rows: List[Line], w: int, h: int) -> List[Line]:
    """``rows`` placed on a ``w`` x ``h`` screen: centred across, TOP rows down,
    cut to fit (a card never scrolls its canvas)."""
    margin = [(" " * max(0, (w - min(w, c.CAP)) // 2), None)]
    return [[] for _ in range(min(TOP, h - 1))] + [margin + r for r in rows[: max(1, h - TOP)]]


def _loop(sheet, stream=None) -> Optional[Outcome]:
    import shutil
    import signal
    import sys
    import termios
    import tty

    from poros.cli.ui import tui
    from poros.cli.ui.keys import _read_paste, read_key
    from poros.cli.ui.live import SYNC_OFF, SYNC_ON, columns, defer, led, take_pending, up
    from poros.cli.ui.style import resolve
    from poros.cli.ui.text import render_line

    if tui.current is not None:  # in a tab: the session reads the keys
        return _tab_loop(sheet)
    stream = stream or sys.stdout
    pal = resolve(stream)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    height = 0
    line_rows = 0  # the typed command above the card, when the card showed it
    full = bool(_depth)

    def draw() -> None:
        nonlocal height, line_rows
        w = columns(stream)
        if full:  # from the top of the canvas, every frame; the card stays put
            cw = min(w, c.CAP)
            rows = canvas(sheet.render(cw), w, shutil.get_terminal_size((w, 24)).lines)
            body = "\r\n".join(render_line(_clip(r, w), pal) + "\x1b[K" for r in rows)
            stream.write(SYNC_ON + "\x1b[H" + body + "\x1b[J" + SYNC_OFF)
            stream.flush()
            return
        rows = sheet.render(w)
        body = "".join(render_line(_clip(r, w), pal) + "\x1b[K\n" for r in rows)
        if height:
            climb = up(height)
        else:
            climb = take_pending() or "\r"
            line_rows = led()
        stream.write(SYNC_ON + climb + body + "\x1b[J" + SYNC_OFF)
        stream.flush()
        height = len(rows)

    outcome: Optional[Outcome] = None
    resized = signal.signal(signal.SIGWINCH, lambda *_: draw()) if full else None
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        attrs = termios.tcgetattr(fd)
        attrs[0] &= ~termios.IXON  # ^s is a key here, not flow control
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        stream.write("\x1b[?25l\x1b[?2004h")  # bracketed paste: a dropped file arrives whole
        draw()
        while outcome is None:
            kind, ch = read_key(fd)
            if kind == "eof":
                outcome = "cancel"
                break
            if kind == "paste" and hasattr(sheet, "drop"):
                sheet.drop(_read_paste(fd))
                draw()
                continue
            outcome = sheet.key(kind, ch)
            if outcome is None:
                draw()
    finally:
        if full:
            signal.signal(signal.SIGWINCH, resized)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        stream.write("\x1b[?2004l\x1b[?25h")
        stream.flush()
        if not full:
            # not erased here: what comes next paints over the card, in one
            # frame; closed with esc, the command that opened it goes too
            defer(stream, h=height + (line_rows if outcome == "cancel" else 0))
    return outcome


def _tab_loop(sheet) -> Optional[Outcome]:
    """A card in a tab of the full-screen session: drawn under the tab row,
    fed the keys the session hands this tab."""
    from poros.cli.ui import tui

    outcome: Optional[Outcome] = None
    with tui.listening():
        try:
            tui.show_card(sheet.render)
            while outcome is None:
                kind, ch = tui.key()
                if kind == "eof":
                    return "cancel"
                if kind == "paste":
                    if hasattr(sheet, "drop"):
                        sheet.drop(ch)
                        tui.show_card(sheet.render)
                    continue
                outcome = sheet.key(kind, ch)
                if outcome is None:
                    tui.show_card(sheet.render)
        finally:
            tui.show_card(None)
    return outcome


def _clip(line: Line, w: int) -> Line:
    out: Line = []
    left = w
    for t, st in line:
        if left <= 0:
            break
        if st == "raw":
            out.append((t, st))
            continue
        out.append((t[:left], st))
        left -= len(t)
    return out
