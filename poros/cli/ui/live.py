"""The live surface: scrollback above, one redrawn region below (fx-style).

Finished things are ``print``-ed into normal scrollback, so the run stays
scrollable, searchable and copyable. Only the surface (live metrics, activity
line, rule, statusline) redraws in place, at <= 10 fps from a daemon thread.
On ``close`` the surface is erased and replaced by whatever the caller prints
last, so nothing half-drawn is left in scrollback.

Not a TTY (CI, pipes, ``TERM=dumb``) or ``POROS_NO_ANIM``: there is no
surface. ``print`` writes plain lines, and ``update`` writes one heartbeat line
at most every ``heartbeat_s`` seconds.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import threading
import time
from typing import Callable, List, Optional, Sequence

from poros.cli.ui.style import Palette, resolve
from poros.cli.ui.text import Line, Seg, plain, render_line

SurfaceFn = Callable[[int, int], List[Line]]  # (frame, columns) -> lines

_HIDE, _SHOW, _CLEAR_DOWN = "\x1b[?25l", "\x1b[?25h", "\x1b[J"


def clip(line: Sequence[Seg], w: int) -> List[Seg]:
    """Cut a line to ``w`` columns (surface rows must never wrap, or the
    redraw's cursor arithmetic drifts)."""
    out: List[Seg] = []
    left = w
    for t, st in line:
        if st == "raw":  # takes no cells
            out.append((t, st))
            continue
        if left <= 0:
            break
        out.append((t[:left], st))
        left -= len(t)
    return out


def columns(stream=None) -> int:
    """Usable columns, read from the terminal itself on every call.

    Never ``$COLUMNS``: it is fixed when the process starts and goes stale the
    moment the window is resized, and a redraw computed for the wrong width
    wraps and leaves stale rows behind. One column is held back so no row
    ever touches the last cell -- terminals disagree on whether that wraps.
    """
    for s in (stream, sys.stdout, sys.stderr, sys.stdin):
        try:
            n = os.get_terminal_size(s.fileno()).columns
        except (AttributeError, OSError, ValueError):
            continue
        if n > 0:  # a fresh pty with no size set reports 0
            return max(20, n - 1)
    try:
        return max(20, shutil.get_terminal_size((101, 24)).columns - 1)
    except Exception:  # pragma: no cover
        return 100


# Synchronized output (DEC 2026): a frame appears whole or not at all.
# Terminals that don't know it ignore it.
SYNC_ON, SYNC_OFF = "\x1b[?2026h", "\x1b[?2026l"


# A live area that closed with nothing to say is not erased on the spot: the
# next write to the terminal paints over it (``take_pending``), so there is no
# blank frame between "• Checking 0.4s" and the output that replaces it.
# A typed command can wait here too: the composer stays on screen until the
# command's first frame climbs over it (``climb``) and shows the line (``lead``).
# A card can retell that line (``retell``): the command it settled on, or none.
_pending = {"h": 0, "stream": None, "climb": "", "lead": "", "lead_rows": 0, "led": 0}


def take_pending(stream=None) -> str:
    """The climb back over a closed live area (and any command line still
    to show), once; "" when there is none."""
    h, climb, lead = _pending["h"], _pending["climb"], _pending["lead"]
    # the rows the command line took, if this write is the one showing it
    _pending["led"] = _pending["lead_rows"] if lead else 0
    if not h and not climb and not lead:
        return ""
    _pending["h"], _pending["climb"], _pending["lead"], _pending["lead_rows"] = 0, "", "", 0
    return up(h) + climb + lead


def led() -> int:
    """Rows of command line the last ``take_pending`` drew above its output:
    a card closed with esc climbs over them too, and the line goes with it."""
    return _pending["led"]


def defer(stream, lead: str = "", h: int = 0, lead_rows: int = 0, climb: str = "") -> None:
    """Leave what is on screen until the next write paints over it."""
    _pending["stream"], _pending["lead"], _pending["h"] = stream, lead, h
    _pending["climb"] = climb
    _pending["lead_rows"] = lead_rows if lead else 0


def retell(lead: str) -> None:
    """The waiting command line becomes ``lead``; "" leaves only the climb, so
    the next write erases the composer and shows no line at all."""
    _pending["lead"] = lead


def waiting() -> bool:
    """A typed command still waits to be shown (the composer is on screen)."""
    return bool(_pending["climb"])


def flush_pending() -> None:
    """Clear a closed live area now, for output that does not paint over it."""
    s = _pending["stream"]
    if (_pending["h"] or _pending["climb"] or _pending["lead"]) and s is not None:
        try:
            s.write(take_pending() + "\x1b[J")
            s.flush()
        except Exception:
            pass


atexit.register(flush_pending)


def up(n: int) -> str:
    """Back to column 0, ``n`` rows up (CUU + CR: supported everywhere, unlike CPL)."""
    return (f"\x1b[{n}A" if n else "") + "\r"


def interactive(stream=None) -> bool:
    stream = stream or sys.stdout
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty()) and os.environ.get("TERM", "") != "dumb"


def animated(stream=None) -> bool:
    return interactive(stream) and not os.environ.get("POROS_NO_ANIM")


class Live:
    def __init__(
        self,
        surface: SurfaceFn,
        *,
        stream=None,
        palette: Optional[Palette] = None,
        fps: float = 10.0,
        heartbeat: Optional[Callable[[int], str]] = None,
        heartbeat_s: float = 30.0,
        delay: float = 0.0,
    ) -> None:
        self.stream = stream or sys.stdout
        self.pal = palette or resolve(self.stream)
        self.surface = surface
        self.live = animated(self.stream)
        self.fps = fps
        self.heartbeat = heartbeat
        self.heartbeat_s = heartbeat_s
        self._lock = threading.RLock()
        self._height = 0
        self._shown: List[str] = []  # the rows on screen, as written, with the width
        self._shown_w = 0
        self._frame = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_beat = 0.0
        self._closed = False
        self.delay = delay  # nothing is drawn for work done sooner than this

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Live":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if not self.live:
            return
        self._write(_HIDE)
        atexit.register(self._restore)
        if not self.delay:
            self._draw()
        self._thread = threading.Thread(target=self._run, name="poros-live", daemon=True)
        self._thread.start()

    def close(self, final: Optional[Sequence[Line]] = None) -> None:
        """Erase the surface and print ``final`` (if any) in its place."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            if self.live and final:
                # the final lines paint over the surface in the same write
                body = "".join(render_line(line, self.pal) + "\x1b[K\n" for line in final)
                self._write(SYNC_ON + up(self._height) + body + _CLEAR_DOWN + _SHOW + SYNC_OFF)
                self._height = 0
            elif self.live:
                # nothing to leave behind: whoever writes next paints over it
                self._write(_SHOW)
                _pending["h"], _pending["stream"] = self._height, self.stream
                self._height = 0
            elif final:
                self._emit(final)
            self._flush()

    # -- api ---------------------------------------------------------------

    def play(self, surface: SurfaceFn, ms: float, fps: float = 30.0, land: float = 0.0) -> None:
        """Draw ``surface`` as fast as ``fps`` for ``ms``: a moment in motion,
        on the caller's thread. Nothing on a pipe or with POROS_NO_ANIM."""
        if not self.live or self._closed:
            return
        self.surface = surface
        t0 = time.monotonic()
        while (time.monotonic() - t0) * 1000 < ms:
            tick = time.monotonic()
            self._draw()
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))

    def print(self, lines: Sequence[Line]) -> None:
        """Lines into scrollback, above the surface."""
        with self._lock:
            if self.live:
                # one write: climb, the new scrollback lines, then the surface
                # again -- each row overwrites in place, nothing is blanked first
                self._draw(locked=True, before=lines)
            else:
                self._emit(lines)
                self._flush()

    def update(self) -> None:
        """New data: redraw now (TTY) or maybe emit a heartbeat line (not a TTY)."""
        if self.live:
            self._draw()
            return
        now = time.monotonic()
        if self.heartbeat and now - self._last_beat >= self.heartbeat_s:
            self._last_beat = now
            self._write(self.heartbeat(self._frame) + "\n")
            self._flush()

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        period = 1.0 / self.fps
        if self.delay and not self._stop.wait(self.delay):
            self._draw()
        while not self._stop.wait(period):
            self._frame += 1
            try:
                self._draw()
            except Exception:  # a draw bug must never kill training
                self.live = False
                self._restore()
                return

    def _draw(self, locked: bool = False, before: Sequence[Line] = ()) -> None:
        """Repaint without ever blanking: climb to the surface's first row,
        overwrite each row and clear only its tail (EL), then clear what is
        left below (ED) in case the new frame is shorter. A terminal without
        synchronized output never shows an empty intermediate state, so
        nothing flickers."""
        if self._closed and not locked:
            return
        with self._lock:
            w = columns(self.stream)
            rows = [render_line(clip(line, w), self.pal) for line in self.surface(self._frame, w)]
            body = "".join(render_line(line, self.pal) + "\x1b[K\n" for line in before)
            # a row the same as last frame is stepped over, not written again:
            # an image on it (gpu_image) is sent once and stays
            same = self._shown if (self._height and not before and w == self._shown_w) else []
            body += "".join("\n" if i < len(same) and same[i] == r else r + "\x1b[K\n"
                            for i, r in enumerate(rows))
            climb = up(self._height) if self._height else take_pending()
            self._write(SYNC_ON + climb + body + _CLEAR_DOWN + SYNC_OFF)
            self._height = len(rows)
            self._shown, self._shown_w = rows, w
            self._flush()

    def _emit(self, lines: Sequence[Line]) -> None:
        if self.pal.depth in ("none", "plain") and not self.live:
            self._write("".join(plain(line) + "\n" for line in lines))
        else:
            self._write("".join(render_line(line, self.pal) + "\n" for line in lines))

    def _restore(self) -> None:
        try:
            self._write(_SHOW)
            self._flush()
        except Exception:
            pass

    def _write(self, s: str) -> None:
        self.stream.write(s)

    def _flush(self) -> None:
        try:
            self.stream.flush()
        except Exception:
            pass
