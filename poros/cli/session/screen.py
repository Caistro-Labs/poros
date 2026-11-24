"""What the interactive session draws: the home screen and the picker, the footer and the
status rows, the frame it paints and erases, and the full screen it opens and closes (the
session itself is ``poros.cli.session.Session``)."""

from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from poros.cli.session.parts import (
    ARM_S,
    HOME,
    MENU,
    NEXT,
    PICK_ROWS,
    PROMPT,
    TRAINS,
    Keys,
    Picker,
    _clip,
    _cols,
    _fit_text,
    _spread,
    _workspace,
    prompt,
)
from poros.cli.ui import components as c
from poros.cli.ui.live import SYNC_OFF, SYNC_ON, take_pending, up
from poros.cli.ui.text import Line, S, banded, ln, render_line, width


class _Screen:
    """The session's screen: its home, the picker, the footer and status rows, drawing, and the
    stderr it quiets (mixed into Session)."""

    def status_items(self) -> List[str]:
        now = time.monotonic()
        if now - self._status_cache[0] < 2.0:
            return self._status_cache[1]
        items = [self.gpu()]
        try:
            from poros.cli.ui.complete import configs

            cfgs = [x for x in configs() if "/" not in x.text]
            if cfgs:
                pick = next((x for x in cfgs if x.text == "train.yaml"), cfgs[0])
                items.append(pick.text + (f" · {pick.desc.split('/')[-1]}" if pick.desc else ""))
        except Exception:
            pass
        items.append(_workspace())
        self._status_cache = (now, items)
        return items

    # -- picker --------------------------------------------------------------

    def home(self) -> List[Tuple[str, str]]:
        """(command, what it does): the next step first, then HOME."""
        out = []
        nxt = self._top = c.next_step or self._found_data()
        if nxt:  # ⏎ runs it whole, however it is cut to fit
            label = nxt if len(nxt) <= 34 else nxt[:33] + "…"
            what = ("next · " + NEXT.get(nxt.split()[0].lstrip("/"), "") if c.next_step
                    else self._found[1])  # your data, recognised: training it is next
            out.append((label, what))
        has_runs = self._has_runs()
        for cmd, desc in HOME:
            if cmd == "/eval" and not has_runs:
                continue
            if cmd != nxt:  # the next step is this one, already on top
                out.append((cmd, desc))
        return out

    def _found_data(self) -> str:
        """``/train --data <file>`` for the newest data file here, said with what
        it holds; "" when there is none."""
        if self._found is None:
            self._found = ("", "")
            try:
                from poros.cli.verbs.data import about
                from poros.cli.verbs.train_data import local_files

                files = local_files(Path.cwd())
                holds, trains = about(Path.cwd() / files[0]) if files else ("", "")
                if holds:
                    self._found = (shlex.join(["/train", "--data", files[0]]),
                                   f"your data · {holds}, trains {trains}")
            except Exception:  # unreadable: /data says why; the list stays as it is
                self._found = ("", "")
        return self._found[0]

    def _has_runs(self) -> bool:
        if self._runs is None:
            from poros import runs

            try:
                self._runs = any(r.state == "succeeded" for r in runs.all_runs())
            except Exception:
                self._runs = False
        return self._runs

    def picker(self) -> Optional[Picker]:
        st = self.s
        if st.closed_for == st.buf or st.caret != len(st.buf):
            return None
        from poros.cli.ui import complete as cm

        buf = st.buf
        if not buf:  # home: the commands, the next step first
            items = self.home()
            self._home_cmds = [x[0] for x in items]
            if self._top and items:
                self._home_cmds[0] = self._top
            return Picker("home", items, 0, len(items))
        if buf.startswith("/") and " " not in buf:
            # The nine commands, in the order of the work; nothing else is listed
            cands = [cm.Candidate("/" + n, d, "command") for n, d in MENU]
            ranked = cm.rank(buf, cands)
            return Picker("commands", [(x.text, x.desc) for x in ranked], 0, len(ranked))
        if " " in buf.lstrip():
            try:
                comp = cm.complete(buf, st.hist)
            except Exception:
                return None
            items = [(x.text, x.desc) for x in comp.candidates if x.text != comp.token]
            if items:
                return Picker("results", items, comp.start, len(items))
        return None

    def ghost(self) -> str:
        st = self.s
        if st.caret != len(st.buf):
            return ""
        if not st.buf:
            if self.picker() is not None or st.closed_for == "":  # the list is up, or esc
                return ""  # just closed it: a blank prompt, not a suggestion
            from poros.cli.shell import _starter

            return _starter() if c.next_step is None else c.next_step
        if st.buf.startswith("/") and " " not in st.buf:  # the picker's choice, completed
            pk = self.picker()
            top = pk.items[st.sel][0] if pk is not None and pk.items else ""
            return top[len(st.buf):] if top.startswith(st.buf) else ""
        try:
            from poros.cli.ui.complete import complete

            return complete(st.buf, st.hist).ghost
        except Exception:
            return ""

    # -- footer ----------------------------------------------------------------

    def footer(self, w: int, busy: Keys = "") -> Tuple[List[Line], int, int]:
        """(rows, cursor_row, cursor_col) for the idle footer."""
        st = self.s
        ghost = self.ghost()
        if self.pal.depth in ("none", "plain"):
            ghost = ""  # without color a ghost would read as typed text
        # a line longer than the terminal scrolls sideways around the caret
        # (one row, so the footer never changes height while typing)
        field_w = max(8, w - len(PROMPT) - 1)
        start = 0
        if len(st.buf) > field_w:
            start = max(0, min(st.caret - field_w + 1, len(st.buf) - field_w + 1))
            start = max(start, 0)
        shown = st.buf[start : start + field_w]
        lead = []
        if start:
            shown = shown[1:]
            lead = [("…", "dm")]
        room = max(0, field_w - len(shown) - len(lead))
        ghost = ghost.rstrip(" ")[:room] if st.caret == len(st.buf) else ""
        # the composer sits in a box (a rule above and below) so it reads as
        # the place to type on any ground; the picker opens under its floor
        box: Line = [("─" * w, "rl")]
        rows: List[Line] = [
            box,
            prompt() + lead + ln(S(shown, "tx"), S(ghost, "dm") if ghost else None),
            box,
        ]
        crow, ccol = 1, min(len(PROMPT) + st.caret - start, w - 1)
        pk = self.picker()
        if pk is not None:
            # the menu drops down under the box; the keys sit under it
            rows += self._menu_rows(pk, w)
            rows.append(self._picked_row(pk, w))
        else:
            rows.append(self._status_row(w, busy=busy))
        self.floor = max(self.floor, len(rows))
        rows += [[] for _ in range(self.floor - len(rows))]
        return rows, crow, ccol

    def _menu_rows(self, pk: Picker, w: int) -> List[Line]:
        """The matches under the prompt, one per row: name and what it does,
        the one ⏎ takes on the focus band."""
        st = self.s
        n = len(pk.items)
        if not n:
            return [[("  nothing matches", "dm")]]
        st.sel = max(0, min(st.sel, n - 1))
        st.offset = min(st.offset, st.sel)
        if st.sel >= st.offset + PICK_ROWS:
            st.offset = st.sel - PICK_ROWS + 1
        shown = pk.items[st.offset : st.offset + PICK_ROWS]
        lw = max(len(lbl) for lbl, _ in shown) + 3
        out: List[Line] = []
        for i, (lbl, desc) in enumerate(shown, st.offset):
            on = i == st.sel
            mark = "❯ " if on and self.pal.depth in ("none", "plain") else "  "
            row = ln(S(mark, "a"), S(lbl.ljust(lw), "hi" if on else "tx"),
                     S(_fit_text(desc, w - lw - 3), "tx" if on else "dm") if desc else None)
            out.append(banded(row, w) if on else row)
        return out

    def _picked_row(self, pk: Picker, w: int) -> Line:
        """While choosing: what the chosen one is, and the keys."""
        st = self.s
        n = len(pk.items)
        count = (f"{st.offset + 1}–{min(n, st.offset + PICK_ROWS)} of {n}"
                 if n > PICK_ROWS else "")
        keys = {"home": [("⏎", "run"), ("↑↓", "move"), ("type", "to filter"), ("esc", "close")],
                "commands": [("⏎", "run"), ("↑↓", "move"), ("esc", "close")]}.get(
            pk.kind, [("⏎", "run"), ("↑↓", "move"), ("tab", "use"), ("esc", "close")])
        while keys and c.keys_width(keys) + len(count) + 2 > w:
            keys = keys[:-1]
        return _spread(c.key_segs(keys), [(count, "rl")] if count else [], w)

    def _status_row(self, w: int, *, busy: Keys = "") -> Line:
        st = self.s
        tag = busy
        armed = st.armed if st.armed and time.monotonic() - st.armed[1] < ARM_S else None
        if self._error:
            return _clip(self._error, w)
        if armed and armed[0] == "ctrl-c":
            return [(f"press {c.key_name('ctrl-c')} again to exit", "dm")]
        if armed and armed[0] == "esc":
            tag = [("esc", "again to clear")]
        if armed and armed[0] == "quit":
            return [(f"{self._what_runs()} is still running · ", "dm"),
                    (f"{self._quit_key} again", "hi"), (" stops it and quits", "dm")]
        if armed and armed[0] == "stop" and self.screen is not None:
            what = "training" if self.screen.tab.title in TRAINS else "it"
            tag = [(c.key_name("ctrl-c"), f"again stops {what}")]
        items = list(self.status_items())
        # keys say themselves one way everywhere: the key, then what it does
        right: Line = [(tag, "tx")] if isinstance(tag, str) and tag else c.key_segs(tag or [])
        while items and len(" · ".join(items)) + (width(right) + 2 if right else 0) > w:
            items.pop()
        left: Line = []
        for i, it in enumerate(items):
            if i:
                left.append((" · ", "rl"))
            left.append((it, "dm"))
        return _spread(left, right, w)

    def busy_footer(self, items, w: int, tag: str) -> List[Line]:
        """Under a running command's live surface: the composer rail, idle,
        and the statusline with how to stop."""
        # the idle footer's shape, so nothing jumps when the command ends
        box: Line = [("─" * w, "rl")]
        stop = [(c.key_name("ctrl-c"), "to stop")]
        return [box, prompt(busy=True), box, self._status_row(w, busy=stop)]

    # -- drawing ---------------------------------------------------------------

    def draw(self) -> None:
        if self.screen is not None:
            self.screen.paint()
            return
        w = _cols()
        rows, crow, ccol = self.footer(w)
        # overwrite in place, row by row (EL clears each tail), then ED for
        # anything a taller previous footer left below: never a blank frame
        buf = [SYNC_ON, up(self.cursor_row) if self.footer_h else (take_pending() or "\r")]
        for i, r in enumerate(rows):
            buf.append(render_line(_clip(r, w), self.pal) + "\x1b[K")
            if i < len(rows) - 1:
                buf.append("\r\n")
        buf.append("\x1b[J")
        climb = len(rows) - 1 - crow
        if climb:
            buf.append(f"\x1b[{climb}A")
        buf.append(f"\r\x1b[{ccol}C" if ccol else "\r")
        buf.append(SYNC_OFF)
        self.out.write("".join(buf))
        self.out.flush()
        self.footer_h = len(rows)
        self.cursor_row = crow

    def erase(self) -> None:
        if self.screen is not None:
            return
        if not self.footer_h:
            return
        self.out.write(up(self.cursor_row) + "\x1b[J")
        self.out.flush()
        self.footer_h = 0
        self.cursor_row = 0
        self.floor = 0

    def emit(self, lines: List[Line]) -> None:
        self.out.write("".join(render_line(line, self.pal) + "\n" for line in lines))
        self.out.flush()

    # -- the tabbed screen ---------------------------------------------------------

    def _open_screen(self) -> None:
        """The session on the full screen: the home tab holds the start box."""
        from poros.cli.banner import _gpu_line
        from poros.cli.ui import intro, tui

        scr = tui.Screen(self.out, self.pal)
        w = _cols()
        words = ("Fine-tune models bigger than your GPU.",
                 _gpu_line() or "What are we training today?")
        home = scr.open(tui.Tab("home", centre=True, keep=False,
                                fit=lambda w: intro.final(w, *words)))  # sized on each paint
        scr.footer = self._tab_footer
        scr.enter()
        tui.current = self.screen = scr
        sys.stdout = tui.ThreadOut(self.out)
        self._quiet_stderr()
        import signal

        self._winch = signal.signal(signal.SIGWINCH, lambda *_: scr.resized())
        self._ticker = tui.Ticker(scr)
        self._ticker.start()
        if self.animate and self.pal.depth not in ("none", "plain") and intro.enabled(self.out):
            # the river lands in the start box, inside the home tab; the
            # prompt under it waits closed, as it does after
            self.s.closed_for = ""
            def show(rows):
                home.lines = rows
                scr.paint()

            fit, home.fit = home.fit, None  # the frames are drawn for the width it began at
            try:
                intro.run(show, intro.frames(w, *words))
            finally:
                home.fit, home.lines = fit, []

    def _quiet_stderr(self) -> None:
        """While the full screen is up, what libraries write to stderr (a Hub
        warning, a C extension's notice) goes to ``~/.poros/session.log``,
        never over the frame."""
        from poros import runs

        try:
            log = runs.home() / "session.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError:  # nowhere to write it: stderr stays as it was
            return
        sys.stderr.flush()
        self._stderr = os.dup(2)
        os.dup2(fd, 2)
        os.close(fd)

    def _loud_stderr(self) -> None:
        if self._stderr is not None:
            sys.stderr.flush()
            os.dup2(self._stderr, 2)
            os.close(self._stderr)
            self._stderr = None

    def _close_screen(self) -> None:
        from poros.cli.ui import term, tui

        if self.screen is None:
            return
        import signal

        self._ticker.stop()
        signal.signal(signal.SIGWINCH, self._winch)
        sys.stdout = self.out
        self._loud_stderr()
        self.screen.leave()
        tui.current = self.screen = None
        term.install(term.Term())
