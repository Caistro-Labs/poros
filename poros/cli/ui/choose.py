"""The fx-style choice surface: ``❯`` menu, digits/arrows/enter/esc.

Returns the chosen index, or ``None`` for cancel / no TTY. Drawn in place
and erased afterwards, so scrollback keeps only the decision.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence, Tuple

from poros.cli.ui.live import columns, interactive, up
from poros.cli.ui.screens import choice_menu
from poros.cli.ui.style import resolve
from poros.cli.ui.text import render_line


def read_key(fd: int) -> str:
    """One logical key: 'up' 'down' 'enter' 'esc' 'ctrl-c' or the character
    (``poros.cli.ui.keys.read_key``, the one decoder)."""
    from poros.cli.ui.keys import read_key as key

    kind, char = key(fd)
    return char if kind == "char" else kind


def choose(
    title: str,
    options: Sequence[Tuple[str, str]],
    *,
    stream=None,
    tag: str = "fix",
    start: int = 0,
) -> Optional[int]:
    stream = stream or sys.stdout
    if not (interactive(stream) and sys.stdin.isatty()):
        return None
    try:
        import termios
        import tty
    except ImportError:  # Windows: no raw mode, the printed fix line stands
        return None
    pal = resolve(stream)
    n = len(options) + 1  # + Cancel
    sel = min(max(start, 0), n - 1)
    height = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def draw() -> None:
        nonlocal height
        rows = choice_menu(title, options, sel, columns(stream), tag)
        stream.write(
            up(height) + "".join(render_line(r, pal) + "\x1b[K\n" for r in rows) + "\x1b[J"
        )
        stream.flush()
        height = len(rows)

    try:
        tty.setcbreak(fd, termios.TCSANOW)
        stream.write("\x1b[?25l")
        draw()
        while True:
            k = read_key(fd)
            if k in ("up", "k"):
                sel = (sel - 1) % n
            elif k in ("down", "j"):
                sel = (sel + 1) % n
            elif k.isdigit() and 1 <= int(k) <= n:
                sel = int(k) - 1
                break
            elif k == "enter":
                break
            elif k in ("esc", "ctrl-c", "q"):
                sel = n - 1
                break
            else:
                continue
            draw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if height:
            stream.write(up(height) + "\x1b[J")
        stream.write("\x1b[?25h")
        stream.flush()
    return None if sel == n - 1 else sel
