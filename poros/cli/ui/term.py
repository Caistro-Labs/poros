"""Where every command's output goes.

Commands never write to stdout directly: they call ``term.print`` (lines into
scrollback), ``term.activity`` (a live one-liner around a blocking call),
``term.live`` (a redrawn surface) and ``term.choose`` (the ❯ menu). In a
one-shot ``poros <cmd>`` the default ``Term`` writes straight to the
terminal. The interactive session installs its own ``Term`` (see
``poros.cli.session``) so the same commands run inside it with the composer
pinned at the bottom -- no command knows which one it is talking to.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, List, Optional, Sequence, Tuple

from poros.cli.ui.text import Line


class Term:
    """Direct-to-terminal sink (one-shot commands)."""

    stream = None

    def _out(self):
        return self.stream or sys.stdout

    def width(self) -> int:
        from poros.cli.ui.live import columns

        return columns()

    def print(self, lines: Sequence[Line]) -> None:
        from poros.cli.ui.live import SYNC_OFF, SYNC_ON, take_pending
        from poros.cli.ui.style import resolve
        from poros.cli.ui.text import render, render_line

        out = self._out()
        pal = resolve(out)
        climb = take_pending()
        if climb:  # paint over a just-closed live area in this same write
            body = "".join(render_line(line, pal) + "\x1b[K\n" for line in lines)
            out.write(SYNC_ON + climb + body + "\x1b[J" + SYNC_OFF)
        else:
            out.write(render(list(lines), pal) + "\n")
        out.flush()

    def show(self, render: Callable[[int], List[Line]]) -> None:
        """A result laid out at a width: printed once here; a sink that can
        lay it out again when the width changes (a tab) keeps ``render``."""
        self.print(render(self.width()))

    def raw(self, text: str) -> None:
        """Unstyled text (JSON, schemas) -- exactly as given, pipe-safe."""
        from poros.cli.ui.live import take_pending

        out = self._out()
        climb = take_pending()
        if climb:
            out.write(climb + "\x1b[J")
        out.write(text if text.endswith("\n") else text + "\n")
        out.flush()

    def erase(self, n: int) -> None:
        """Clear the ``n`` lines just printed, for a screen redrawn in place.
        Off a terminal nothing is erased: the log keeps every version."""
        from poros.cli.ui.live import interactive, up

        out = self._out()
        if n and interactive(out):
            out.write(up(n) + "\x1b[J")
            out.flush()

    def live(self, surface: Callable[[int, int], List[Line]], **kw):
        from poros.cli.ui.live import Live

        return Live(surface, stream=self._out(), **kw)

    def activity(self, label: str, *, statusline: Optional[List[str]] = None, tag: str = "",
                 below=None, delay: float = 0.0):
        from poros.cli.ui.train_view import Activity

        return Activity(label, stream=self._out(), statusline=statusline, tag=tag, below=below,
                        delay=delay)

    def progress(self, fn: Callable[[], Tuple[int, int]]) -> None:
        """How far a long command is, as (done, total): where the sink can
        show it outside the command's own screen (a tab's label). Nothing
        here: the live surface already says it."""

    def choose(
        self, title: str, options: Sequence[Tuple[str, str]], tag: str = "fix", start: int = 0
    ) -> Optional[int]:
        from poros.cli.ui.choose import choose

        return choose(title, options, stream=self._out(), tag=tag, start=start)


term: Term = Term()
# The command line being run, after `poros`: argv for a one-shot command, the
# typed line inside the session (which sets it). Retry commands are built from it.
argv: Optional[List[str]] = None


def current_argv() -> List[str]:
    import sys

    return list(argv) if argv is not None else sys.argv[1:]


def install(t: Term) -> Term:
    """Swap the active sink; returns the previous one."""
    global term
    prev, term = term, t
    return prev


# A command running in a thread of its own (a tab of the full-screen session)
# writes to its own sink; every other thread uses the installed one.
_local = threading.local()


def bind(t: Optional[Term]) -> None:
    """This thread's own sink (None: back to the installed one)."""
    _local.term = t


def get() -> Term:
    return getattr(_local, "term", None) or term


def out(lines: List[Line]) -> None:
    """Print lines through the active sink (terminal, or the session)."""
    get().print(lines)


def err(lines: Sequence[Line]) -> None:
    """An error's lines. A one-shot command writes them to stderr: its stdout carries results
    only, so a script reading a pipe never takes a refusal for output. Inside the session, whose
    sink is its own, they are shown where everything else is."""
    t = get()
    if type(t) is not Term:
        t.print(lines)
        return
    from poros.cli.ui.live import interactive, take_pending

    where = t._out()
    if interactive(where) and not interactive(sys.stderr):
        # stderr goes elsewhere (2>file): a live area just closed is cleared where it was drawn
        climb = take_pending()
        if climb:
            where.write(climb + "\x1b[J")
            where.flush()
    sink = Term()
    sink.stream = sys.stderr
    sink.print(lines)
