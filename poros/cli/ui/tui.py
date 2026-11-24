"""The full-screen session: every command in its own tab.

The whole session lives on the terminal's second screen:

    ▛▜  home   train 42%   runs                  the tab row: the active tab
      ━━━━━━──────────────────────────────────   bright and underlined
                                                 the canvas: what the active
      (the start box, a card, a live run ...)    tab holds, cut to fit
                                                 the footer: the composer,
    ──────────────────────────────────────────   its picker, the statusline
    ▛▜ ❯
    ──────────────────────────────────────────
    RTX 4090 24.0 GB · train.yaml · ~/support-bot

A tab holds printed lines (``lines``) and, while something is running in it,
a live surface under them (``surface``). Frames are painted with absolute
cursor moves, only the rows that changed, inside synchronized output, so a
redraw never flickers and never scrolls the screen.

On quit the second screen goes away and the terminal is as it was before
Poros started: nothing from the tabs is printed into its scrollback
(``poros runs`` keeps every run).
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from poros.cli.ui.live import SYNC_OFF, SYNC_ON, clip
from poros.cli.ui.style import Palette
from poros.cli.ui.text import MARK, Line, render_line

SurfaceFn = Callable[[int, int], List[Line]]

ALT_ON, ALT_OFF = "\x1b[?1049h", "\x1b[?1049l"
SCHEME_ON, SCHEME_OFF = "\x1b[?2031h", "\x1b[?2031l"  # tell us when it goes light/dark
KITTY_FORGET = "\x1b_Ga=d,d=A,q=2\x1b\\"  # take every kitty image off the screen
HEAD = 2  # the tab row and its underline


@dataclass
class Tab:
    title: str
    lines: List[Line] = field(default_factory=list)
    surface: Optional[SurfaceFn] = None  # live rows under the lines, while running
    label: Optional[Callable[[], str]] = None  # a title that changes (train 42%)
    centre: bool = False  # the home tab: its lines sit in the middle of the canvas
    scroll: int = 0  # rows up from the bottom
    keep: bool = True  # printed into scrollback on quit
    surface_rows: List[Line] = field(default_factory=list)  # the surface's last frame
    held: bool = False  # a closed surface left up until the lines after it come in
    card: Optional[Callable[[int], List[Line]]] = None  # an open card, drawn at a width
    echo: int = -1  # the row holding the typed command, retold by the card it opened
    gone: bool = False  # closed with esc: the tab goes when its command ends
    thread: Optional[threading.Thread] = None  # the command running in this tab
    listening: int = 0  # a card or a menu in this tab is reading keys
    keys: "queue.Queue[Tuple[str, str]]" = field(default_factory=queue.Queue)
    partial: str = ""  # stray stdout text waiting for its newline
    fit: Optional[Callable[[int], List[Line]]] = None  # lines laid out again at each width
    command: str = ""  # the line typed to open it
    closing: bool = False  # ctrl+w once on a running tab: again stops it and closes
    run: str = ""  # the run this tab trains (its id), so /runs can bring it forward
    check_until: float = 0.0  # its ✓ stays in the accent until then

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def name(self) -> str:
        if self.label is not None:
            try:
                return self.label() or self.title
            except Exception:  # a label must never break a frame
                return self.title
        return self.title


MIN_ROWS = 16  # shorter, the tab row and the prompt leave no room: inline instead


def enabled(stream) -> bool:
    """The session runs on the tabbed screen wherever cards may take the full
    screen (not under ``POROS_INLINE=1`` or Zellij) and the terminal is tall
    enough. ``POROS_TABS=0`` keeps the inline session."""
    from poros.cli.ui.sheet.frame import full_screen_ok

    return (os.environ.get("POROS_TABS", "1") != "0" and full_screen_ok(stream)
            and size(stream)[1] >= MIN_ROWS)


def size(stream=None) -> Tuple[int, int]:
    """(columns, rows), read from the terminal on every call; one column held
    back, as everywhere (live.columns)."""
    from poros.cli.ui.live import columns

    return columns(stream), shutil.get_terminal_size((100, 30)).lines


def close_hint() -> str:
    from poros.cli.ui import components as c

    return f"{c.key_name('ctrl-w')} close "


def tab_row(tabs: Sequence[Tab], active: int, w: int, over: Optional[Tab] = None) -> List[Line]:
    """`` 1 home   2 machine ✕   3 train 42%``, numbered for the 1-9 keys, the open
    one with its ✕ (ctrl+w) and underlined in the accent, in a rule that runs
    to the edge. Tabs that don't fit collapse from the left of the active
    one into ``‹ n``."""
    names = [f"{i + 1} {t.name()}" + (" ✕" if i == active and i and over is None else "  ")
             for i, t in enumerate(tabs)]
    if over is not None:  # a card over the open tab (/settings) names it while it is up
        names[active] = f"{active + 1} {over.name()}  "
    first = 0

    def width_from(i: int) -> int:
        return sum(len(n) + 3 for n in names[i:]) + (len(f"‹ {i}") + 2 if i else 0)

    while first < active and width_from(first) > w:
        first += 1
    top: Line = []
    under: Line = []
    col = 0
    if first:
        more = f"‹ {first}"
        top += [(more, "dm"), ("  ", None)]
        under.append(("─" * (len(more) + 2), "rl"))
        col += len(more) + 2
    for i in range(first, len(tabs)):
        n = names[i]
        if col + len(n) + 2 > w:
            break
        on = i == active
        num = str(i + 1)
        ink = "hi" if on else "tx" if tabs[i].running else "dm"  # a running tab stays readable
        name = n[len(num) + 1 : -2]
        words: Line = [(name, ink)]
        if name.endswith("✓") and getattr(tabs[i], "check_until", 0.0) > time.monotonic():
            words = [(name[:-1], ink), ("✓", "a")]  # the run landing, for a moment
        top += [(" " + num + " ", "a" if on else "rl"), *words, (n[-2:] + " ", "dm"), (" ", None)]
        under += [("━" * (len(n) + 2), "a") if on else ("─" * (len(n) + 2), "rl"), ("─", "rl")]
        col += len(n) + 3
    hint = close_hint()  # a card over the tab says its own esc, at its foot
    if active and over is None and col + len(hint) < w:  # how a tab goes, where its ✕ is
        top += [(" " * (w - col - len(hint)), None), (hint, "dm")]
    used = sum(len(t) for t, _ in under)
    if used < w:
        under.append(("─" * (w - used), "rl"))
    return [top, under]


class Fitted(list):
    """A block of a tab's lines laid out again at each width (a command's
    result): it stands as one entry in ``tab.lines`` and ``laid`` opens it."""

    def __init__(self, render: Callable[[int], List[Line]]) -> None:
        super().__init__()
        self.render = render
        self._at: Tuple[int, List[Line]] = (-1, [])

    def at(self, w: int) -> List[Line]:
        if self._at[0] != w:
            self._at = (w, list(self.render(w)))
        return self._at[1]


def laid(lines: List[Line], w: int) -> List[Line]:
    """A tab's lines at width ``w``, each fitted block laid out for it."""
    out: List[Line] = []
    for r in lines:
        if isinstance(r, Fitted):
            out += r.at(w)
        else:
            out.append(r)
    return out


def shown(tab: Tab, w: int) -> List[Line]:
    """What a tab's canvas holds at width ``w``: its lines, without the typed
    command on top (the tab's label says it; scrollback keeps it on quit)."""
    if tab.fit is not None:
        return list(tab.fit(w))
    rows = laid(tab.lines, w)
    if 0 <= tab.echo < len(rows):
        del rows[tab.echo]
    return rows


def canvas(tab: Tab, w: int, h: int) -> List[Line]:
    """The ``h`` rows the tab shows: its lines then its live surface, the
    newest at the bottom (scrolled up by ``tab.scroll``); the home tab sits
    in the middle."""
    body = shown(tab, w)
    if tab.surface is not None:
        body += tab.surface_rows
    if len(body) <= h:
        tab.scroll = 0
        top = max(0, (h - len(body)) // 2) if tab.centre else 0
        return [[] for _ in range(top)] + body + [[] for _ in range(h - top - len(body))]
    tab.scroll = max(0, min(tab.scroll, len(body) - h))
    end = len(body) - tab.scroll
    rows = body[end - h : end]
    if tab.scroll:
        note = f"↓ {tab.scroll} rows below · end jumps back"
        rows[-1] = [(" " * max(0, w - len(note)), None), (note, "dm")]
    return rows


FOOT = 4  # the footer at rest: rule, prompt, rule, statusline
OVER_MAX = 13  # the tallest list the prompt opens: its rows and its keys
CARD_TOP = 1  # rows of air between the tab row and a card


KEEP = 3  # a card's last rows (the command, the rule, the keys) always show


def card_canvas(card: Callable[[int], List[Line]], w: int, h: int) -> List[Line]:
    """A card centred across (CAP wide), ``CARD_TOP`` rows down, its last
    rows (the command, the rule, the keys) on the floor, where the prompt and
    the statusline sit on every other tab. Taller than the room, it loses
    rows from its middle, never its keys."""
    from poros.cli.ui import components as c

    cw = min(w, c.CAP)
    margin = [(" " * max(0, (w - cw) // 2), None)]
    rows = [margin + r for r in card(cw)]
    room = max(1, h - CARD_TOP)
    if len(rows) > room:
        top = max(0, room - KEEP - 1)
        rows = rows[:top] + [margin + [("  ⋮", "rl")]] + rows[-KEEP:] if top else rows[-room:]
    # the floor: the rule and keys land on the footer's last rows; with a
    # command line just above them it sits where the prompt is
    cut = len(rows) - (KEEP if len(rows) > KEEP else 0)
    gap = [[] for _ in range(h - CARD_TOP - len(rows))]
    return [[] for _ in range(CARD_TOP)] + rows[:cut] + gap + rows[cut:]


# The screen the session is running on, if it is the tabbed one.
current: Optional["Screen"] = None
# The tab a command thread writes to (unset on the session's own thread).
local = threading.local()


def mine() -> Optional[Tab]:
    """The tab of the command running on this thread (else the active one)."""
    tab = getattr(local, "tab", None)
    if tab is None and current is not None and current.tabs:
        tab = current.tab
    return tab


def show_card(card: Optional[Callable[[int], List[Line]]]) -> None:
    """Open (or, with None, close) a card in this command's tab."""
    tab = mine()
    if tab is not None:
        tab.card = card
        current.paint()


def key() -> Tuple[str, str]:
    """The next key for this command's tab, as ``session.read_key`` gives it.
    The session reads the keyboard and hands keys to the tab on screen that
    is listening; a stopped command gets its ``KeyboardInterrupt`` here."""
    tab = mine()
    while True:
        try:
            return tab.keys.get(timeout=0.1)  # short waits: ctrl+c lands between them
        except queue.Empty:
            continue


class listening:
    """While a card or menu in this command's tab reads keys."""

    def __enter__(self) -> None:
        tab = mine()
        tab.listening += 1
        while not tab.keys.empty():  # keys typed before it opened were for the prompt
            tab.keys.get_nowait()

    def __exit__(self, *exc) -> None:
        mine().listening -= 1


def interrupt(tab: Tab) -> bool:
    """Stop the command running in ``tab`` (ctrl+c): a ``KeyboardInterrupt``
    raised in its thread, which the command handles as on a terminal."""
    import ctypes

    th = tab.thread
    if th is None or not th.is_alive() or th.ident is None:
        return False
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident),
                                               ctypes.py_object(KeyboardInterrupt))
    return True


_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[_P][^\x1b]*\x1b\\")


class ThreadOut:
    """``sys.stdout`` while the tabs are up: what a command thread prints
    directly (not through its sink) lands in its tab as plain lines instead
    of over the screen; every other write goes to the terminal."""

    def __init__(self, real) -> None:
        self.real = real

    def write(self, text: str) -> int:
        tab = getattr(local, "tab", None)
        if tab is None:
            return self.real.write(text)
        text = tab.partial + _CSI.sub("", text).replace("\r\n", "\n")
        *done, tab.partial = text.split("\n")
        tab.lines += [[(row.split("\r")[-1], "tx")] for row in done]
        return len(text)

    def flush(self) -> None:
        if getattr(local, "tab", None) is None:
            self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


def retell(command: Optional[str]) -> None:
    """The typed line at the top of the tab becomes the command its card
    settled on; None (esc) and the tab goes when the command ends."""
    t = mine()
    if t is None:
        return
    if command is None:
        t.gone = True
    elif 0 <= t.echo < len(t.lines):
        from poros.cli.session.parts import prompt

        t.lines[t.echo] = prompt() + [(command, "hi")]
        t.title = command.lstrip("/").split()[0] if command.strip() else t.title


class Screen:
    """The tabs, the frame they make with the footer, and the painting."""

    def __init__(self, out, pal: Palette) -> None:
        self.out = out
        self.pal = pal
        self.tabs: List[Tab] = []
        self.active = 0
        self.over: Optional[Tab] = None  # a card over the open tab (/settings), not a tab
        self.on = False
        self._shown: List[str] = []
        self._images: List[Tuple[int, str]] = []  # (row, escape) of each image on screen
        self._size: Tuple[int, int] = (0, 0)
        self._lock = threading.RLock()
        self._frame = 0
        # what sits under the canvas: rows plus where the caret goes in them
        # (None hides the caret, for a card or a running command)
        self.footer: Callable[[int], Tuple[List[Line], Optional[Tuple[int, int]]]] = (
            lambda w: ([], None))

    # -- tabs ----------------------------------------------------------------

    def open(self, tab: Tab, focus: bool = True) -> Tab:
        with self._lock:
            self.tabs.append(tab)
            if focus:
                self.active = len(self.tabs) - 1
        return tab

    def close(self, i: Optional[int] = None) -> None:
        """Close a tab (never the home tab, the first)."""
        with self._lock:
            i = self.active if i is None else i
            if i <= 0 or i >= len(self.tabs):
                return
            del self.tabs[i]
            if self.active >= i:  # the closed one or one after it: step left
                self.active -= 1

    def switch(self, i: int) -> None:
        with self._lock:
            if 0 <= i < len(self.tabs):
                self.active = i

    def step(self, d: int) -> None:
        with self._lock:
            if self.tabs:
                self.active = (self.active + d) % len(self.tabs)

    @property
    def tab(self) -> Tab:
        """The tab on screen: a card opened over the open tab, else that tab."""
        return self.over if self.over is not None else self.tabs[self.active]

    def replace(self, i: int, tab: Tab) -> Tab:
        """A command typed in a finished tab takes its place, like a link."""
        with self._lock:
            self.tabs[i] = tab
            self.active = i
        return tab

    def twin(self, line: str) -> int:
        """The tab already showing ``line`` (else -1): it comes forward."""
        for i, t in enumerate(self.tabs):
            if i and t.command == line:
                return i
        return -1

    def resized(self) -> None:
        """The terminal changed size (SIGWINCH): draw it all again, off the
        signal handler, which may have interrupted a paint holding the lock."""
        threading.Thread(target=self.paint, kwargs={"force": True}, name="poros-repaint",
                         daemon=True).start()

    def scroll(self, rows: int) -> None:
        """Positive scrolls up, into the tab's past; ``0`` jumps to the bottom."""
        with self._lock:
            t = self.tab
            t.scroll = 0 if rows == 0 else max(0, t.scroll + rows)

    # -- the screen ------------------------------------------------------------

    def enter(self) -> None:
        self.out.write(ALT_ON + SCHEME_ON + "\x1b[H\x1b[2J")
        self.out.flush()
        self.on = True
        self._shown, self._images = [], []

    def leave(self) -> None:
        """Back to the terminal's own screen, clean: what the tabs showed
        stays in the session, not in the shell's scrollback."""
        if not self.on:
            return
        self.on = False
        self.out.write(SCHEME_OFF + ALT_OFF + "\x1b[?25h")
        self.out.flush()

    def rows(self, w: int, h: int) -> Tuple[List[Line], Optional[Tuple[int, int]]]:
        """The whole frame and the caret's (row, col), or None."""
        foot, caret = self.footer(w)
        if not self.tabs:
            return [[] for _ in range(h - len(foot))] + foot, None
        t = self.tab
        if t.surface is not None:
            try:
                t.surface_rows = t.surface(self._frame, w)
            except Exception:  # a surface bug shows as nothing, never a dead session
                t.surface_rows = []
        if t.card is not None:  # a card owns the canvas and carries its own keys
            foot, caret = [], None
        # The canvas keeps the room it has under the resting footer; a list
        # opening under the prompt lies over its last rows instead of
        # squeezing it, so nothing above moves while you type.
        room = max(1, h - HEAD - min(len(foot), FOOT))
        body = card_canvas(t.card, w, room) if t.card is not None else canvas(t, w, room)
        over = max(0, len(foot) - FOOT)
        if over and t.centre and t.card is None:
            # the start box is whole, never cut by the list: it rises just
            # clear of the tallest list (so it doesn't move again as typing
            # narrows one), and only a terminal too short for both loses it
            box = shown(t, w)
            n, top = len(box), (room - len(box)) // 2
            if top + n > room - OVER_MAX:
                lift = room - OVER_MAX - n
                body = [[] for _ in body]
                if lift >= 0:
                    body[lift : lift + n] = box
        if over:
            body = body[: max(1, room - over)]
        frame = tab_row(self.tabs, self.active, w, self.over) + body + foot
        frame = frame[:h]
        if caret is not None:
            caret = (HEAD + len(body) + caret[0], caret[1])
        if t.listening and t.card is None:  # the keys go to the tab, not the prompt
            caret = None
        return frame, caret

    def paint(self, force: bool = False) -> None:
        """Draw the frame, writing only the rows that changed."""
        if not self.on:
            return
        with self._lock:
            w, h = size(self.out)
            if (w, h) != self._size:
                force, self._size = True, (w, h)
            rows, caret = self.rows(w, h)
            # An image stays where it was placed whatever the cells under it
            # show, so it belongs to the frame: when the images in view change
            # (another tab, a card or the list over it, a scroll) take them all
            # off and draw every row again, which places the new ones.
            images = [(i, image_key(t)) for i, r in enumerate(rows) for t, st in r if st == "raw"]
            moved = images != self._images
            drawn = [render_line(clip(r, w), self.pal) for r in rows]
            drawn += [""] * (h - len(drawn))
            buf = [SYNC_ON, "\x1b[?25l"]
            if (moved or force) and self._images:
                force = True  # a logo's ink follows the theme: drawn again, in it
                if any("\x1b_G" in t or t.startswith(MARK) for _, t in self._images):
                    buf.append(KITTY_FORGET)
            self._images = images
            if force:
                buf.append("\x1b[H\x1b[2J")
            for i, r in enumerate(drawn):
                if force or i >= len(self._shown) or self._shown[i] != r:
                    buf.append(f"\x1b[{i + 1};1H{r}\x1b[K")
            if caret is not None:
                buf.append(f"\x1b[{caret[0] + 1};{caret[1] + 1}H\x1b[?25h")
            buf.append(SYNC_OFF)
            self._shown = drawn
            self._frame += 1
            self.out.write("".join(buf))
            self.out.flush()


_KITTY_ID = re.compile(r"\x1b_G[^;]*?\bi=(\d+)")


def image_key(raw: str) -> str:
    """What makes an image the same one: a kitty image sent again on its id
    replaces itself where it is, so only its row and id say whether the screen
    has to be cleared of it. Anything else: the whole escape."""
    m = _KITTY_ID.search(raw)
    return f"\x1b_G{m.group(1)}" if m else raw


class Ticker:
    """Repaints the screen at ``fps`` while something in a tab is live."""

    def __init__(self, screen: Screen, fps: float = 10.0) -> None:
        self.screen = screen
        self.period = 1.0 / fps
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="poros-tabs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            if any(t.surface is not None or t.label is not None for t in self.screen.tabs):
                try:
                    self.screen.paint()
                except Exception:  # never let a paint kill the session
                    pass


class TabLive:
    """``Live``'s interface, drawn inside a tab instead of under the cursor:
    ``print`` adds lines to the tab, the surface redraws in it, and ``close``
    leaves the final lines in its place."""

    def __init__(self, screen: Screen, tab: Tab, surface: SurfaceFn, **_kw) -> None:
        self.screen = screen
        self.tab = tab
        self.surface = surface
        self.live = True  # surfaces animate, as on a terminal
        self._closed = False

    def __enter__(self) -> "TabLive":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        let_go(self.tab)
        self.tab.surface = self.surface
        self.screen.paint()

    def update(self) -> None:
        pass  # the ticker repaints

    def play(self, surface: SurfaceFn, ms: float, fps: float = 30.0, land: float = 0.0) -> None:
        """``Live.play``, in the tab. At ``land`` the tab's label says it's done,
        in the accent for a moment; off screen the tab shows it all the same."""
        self.tab.surface = surface
        t0 = time.monotonic()
        said = False
        if os.environ.get("POROS_NO_ANIM"):  # still: only the tab says it's done
            ms, land = 0, 0
            self.tab.label = (lambda name: lambda: f"{name} ✓")(self.tab.title)
        while (el := (time.monotonic() - t0) * 1000) < ms:
            tick = time.monotonic()
            if land and not said and el >= land:
                said = True
                tab, name = self.tab, self.tab.title
                tab.label = lambda: f"{name} ✓"
                tab.check_until = time.monotonic() + 1.5
            self.screen.paint()
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))

    def print(self, lines: Sequence[Line]) -> None:
        self.tab.lines += list(lines)
        self.screen.paint()

    def close(self, final: Optional[Sequence[Line]] = None, hold: bool = False) -> None:
        """``hold``: the surface's last frame stays up until the tab's next
        lines take its place in the same paint, never an empty frame between
        them (an image there would be taken off the screen and sent again:
        a blink)."""
        if self._closed:
            return
        self._closed = True
        if hold and not final:
            self.tab.held = True
            return
        self.tab.surface = None
        if final:
            self.tab.lines += [r for r in final]
        self.screen.paint()


def let_go(tab: Tab) -> None:
    """A held surface goes: what comes next in the tab is drawn in its place."""
    if tab.held:
        tab.held, tab.surface = False, None


class TabTerm:
    """Where a command's output goes in the tabbed session: into its own tab
    (see poros/cli/ui/term.py for the interface every command writes to)."""

    def __init__(self, screen: Screen, tab: Tab) -> None:
        self.screen = screen
        self.tab = tab
        self.stream = screen.out

    def width(self) -> int:
        return size(self.screen.out)[0]

    def print(self, lines: Sequence[Line]) -> None:
        let_go(self.tab)
        self.tab.lines += list(lines)
        self.screen.paint()

    def show(self, render: Callable[[int], List[Line]]) -> None:
        let_go(self.tab)
        self.tab.lines.append(Fitted(render))
        self.screen.paint()

    def raw(self, text: str) -> None:
        let_go(self.tab)
        self.tab.lines += [[(row, None)] for row in text.rstrip("\n").split("\n")]
        self.screen.paint()

    def erase(self, n: int) -> None:
        if n:
            del self.tab.lines[-n:]

    def live(self, surface: SurfaceFn, **kw) -> TabLive:
        return TabLive(self.screen, self.tab, surface, **kw)

    def activity(self, label: str, *, statusline: Optional[List[str]] = None, tag: str = "",
                 below=None, delay: float = 0.0):
        from poros.cli.ui.train_view import Activity

        return Activity(label, statusline=statusline, tag=tag, below=below, delay=delay,
                        live=lambda fn, **kw: TabLive(self.screen, self.tab, fn, **kw))

    def progress(self, fn: Callable[[], Tuple[int, int]]) -> None:
        """The tab's label carries the command's progress: ``train 42%``
        while it runs, then ``train ✓`` (all done) or ``train ○`` (stopped)."""
        tab = self.tab

        def label() -> str:
            done, total = fn()
            name = tab.title
            if tab.running:
                # room for two figures always, so the tabs after it never step
                # as it counts; the spare column trails, where it reads as air
                pct = f"{min(99, int(100 * done / total))}%"
                return f"{name} {pct:<3}" if total else name
            return f"{name} {'✓' if total and done >= total else '○'}"

        tab.label = label
        self.screen.paint()

    def choose(self, title: str, options, tag: str = "fix", start: int = 0) -> Optional[int]:
        """The ❯ menu as a card in the tab: ↑↓, digits, ⏎; esc cancels."""
        from poros.cli.ui.screens import choice_menu

        n = len(options) + 1  # + Cancel
        sel = min(max(start, 0), n - 1)
        with listening():
            try:
                while True:
                    show_card(lambda w, s=sel: choice_menu(title, options, s, w, tag))
                    kind, ch = key()
                    if kind == "up" or ch == "k":
                        sel = (sel - 1) % n
                    elif kind == "down" or ch == "j":
                        sel = (sel + 1) % n
                    elif ch.isdigit() and 1 <= int(ch) <= n:
                        sel = int(ch) - 1
                        break
                    elif kind == "enter":
                        break
                    elif kind in ("esc", "ctrl-c") or ch == "q":
                        sel = n - 1
                        break
            finally:
                show_card(None)
        return None if sel == n - 1 else sel
