"""The interactive session's tabs: a command runs in a tab of its own while the prompt stays
usable, with a footer of its own; the keys that move between tabs, close one, and leave."""

from __future__ import annotations

from poros.cli.session.parts import (
    EXIT_WORDS,
    OVER,
    TRAINS,
    Keys,
    _blank,
    _known,
    _only_error,
    _unknown,
    prompt,
)
from poros.cli.ui import components as c


class _Tabs:
    """The session's tabs: a command run in its own tab, its footer, the keys between tabs,
    closing and leaving (mixed into Session)."""

    def _tab_footer(self, w: int):
        """Under the canvas: the composer (caret in it), or, while a command
        runs, its resting shape."""
        scr = self.screen
        running = scr.tab.running
        reused = scr.active and not running  # a finished tab: a command goes in its place
        new = c.key_name("ctrl-n")
        hint: Keys = ([("⏎", "here"), (new, "new tab"), ("1-9", "switch")] if reused
                      else [("1-9", "switch tabs")] if len(scr.tabs) > 1 else [])
        if self._note:
            hint = self._note
        busy = [(c.key_name("ctrl-c"), "twice stops it")] if running else hint
        rows, crow, ccol = self.footer(w, busy=busy)
        while rows and not rows[-1]:  # the floor's padding is for the inline footer
            rows.pop()
        return rows, (crow, ccol)

    def _run_in_tab(self, line: str) -> bool:
        """A typed command runs in a tab of its own, on a thread of its own:
        the prompt stays free for the next one."""
        import threading

        from poros.cli import views
        from poros.cli.ui import complete, term, tui

        scr = self.screen
        head = line.split()[0].lstrip("/").lower()
        if head in EXIT_WORDS:
            self._stop_tabs()  # as ctrl-d does: nothing of the shell's is still drawing at exit
            return False
        if head == "clear":  # the finished tabs go; home and running ones stay
            for t in [t for t in scr.tabs[1:] if not t.running]:
                scr.close(scr.tabs.index(t))
            return True
        if head not in _known() and head not in EXIT_WORDS:  # a typo makes no tab
            self._error = _unknown(head)
            scr.paint()
            return True
        new, self._new_tab = self._new_tab, False
        tab = tui.Tab(head or "run", lines=[prompt() + [(line, "hi")], []], echo=0, command=line)
        here = scr.active
        at = scr.twin(line)
        swapped = None  # the finished tab this one took the place of
        if head in OVER:  # a quick card over the open tab: it makes no tab
            scr.over = tab
        elif at >= 0 and scr.tabs[at].running:  # already running: go to it
            scr.switch(at)
            scr.paint()
            return True
        elif at >= 0:  # already open: it comes forward, run again
            swapped = scr.tabs[at]
            scr.replace(at, tab)
        elif not new and here and not scr.tabs[here].running and not scr.tabs[here].listening:
            swapped = scr.tabs[here]
            scr.replace(here, tab)  # a finished tab: the command goes in its place
        else:
            scr.open(tab)
        c.next_step = None

        def work() -> None:
            tui.local.tab = tab
            term.bind(tui.TabTerm(scr, tab))
            try:
                self.dispatch(line)
            except KeyboardInterrupt:
                term.get().print(views.note("Stopped"))
            except Exception as exc:  # noqa: BLE001 - a command never takes the session down
                term.get().print(c.error_event("Error", f"{type(exc).__name__}: {exc}",
                                               term.get().width()))
            finally:
                term.bind(None)
                tui.local.tab = None
                tui.let_go(tab)  # a surface held for lines that never came
                tab.thread = None  # done: the footer stops offering ctrl+c
                complete._CACHE.clear()
                self._status_cache = (0.0, [])
                self._runs = self._found = None
                shown = tui.laid(tab.lines, tui.size(scr.out)[0])
                said = _only_error(tab.lines[2:])
                if scr.over is tab:  # the card over the tab closes with its command
                    scr.over = None
                elif said and not tab.gone:
                    # it could not run: the error is said under the prompt, and
                    # the tab it took the place of comes back
                    self._error = said
                    with scr._lock:
                        if tab in scr.tabs:
                            i = scr.tabs.index(tab)
                            if swapped is not None:
                                scr.tabs[i] = swapped
                            else:
                                scr.close(i)
                elif tab.gone or (len(shown) <= 2 and tab.surface is None):
                    with scr._lock:  # esc, or nothing to show: no tab
                        if tab in scr.tabs:
                            scr.close(scr.tabs.index(tab))
                scr.paint()

        tab.thread = threading.Thread(target=work, name=f"poros-{head}", daemon=True)
        tab.thread.start()
        scr.paint()
        return True

    # -- loop ------------------------------------------------------------------

    def _working(self) -> list:
        """The tabs doing work: a command running, not a card waiting on you."""
        tabs = self.screen.tabs if self.screen is not None else []
        return [t for t in tabs if t.running and not t.listening]

    def _quit_asked(self, armed, now: float, key: str) -> bool:
        """True when the session may quit now. With work running, the first
        press only says so; the next one stops it and quits."""
        if not self._working() or (armed and armed[0] == "quit"):
            return True
        self.s.armed, self._quit_key = ("quit", now), key
        return False

    def _what_runs(self) -> str:
        work = self._working()
        return "training" if any(t.title in TRAINS for t in work) else "a command"

    def _leave(self) -> None:
        """Quit: what still runs stops first, then the footer goes."""
        self._stop_tabs()
        self.erase()

    def _stop_tabs(self) -> None:
        """Stop what still runs in a tab before the session quits, then wait, briefly, for the
        shell's own threads that are still drawing (a tab's last paint, a card's picture). A
        daemon thread caught writing when Python exits aborts it (_enter_buffered_busy)."""
        import threading
        import time

        from poros.cli.ui import tui

        scr = self.screen
        # the tabs and the card over them as they are now: a tab that has left its loop takes
        # itself off the live list while its thread may still be writing its last lines, and it
        # is waited for all the same
        first = [*scr.tabs, getattr(scr, "over", None)] if scr else []
        # every tab is told first, from a copy of the list: a tab that ends takes itself off the
        # live one, and the tab after it would be passed over
        told = []
        for t in list(scr.tabs) if scr else []:
            th = t.thread
            if th is not None and t.running and tui.interrupt(t):
                told.append(th)
        for th in told:
            th.join(timeout=5)
        # a card still reading your keys (/settings over a tab) is not drawing: not waited for
        cards = [*first, *scr.tabs, getattr(scr, "over", None)] if scr else []
        reading = {t.thread for t in cards if t is not None and t.listening}
        drawing = {t.thread for t in cards if t is not None and t.thread is not None} - reading
        until = time.monotonic() + 2.0
        # the ticker and a live surface are stopped by their owners; a run's own workers (a
        # layer reader, a watchdog) draw nothing and may stay idle for good: not waited for
        for th in threading.enumerate():
            if th is threading.current_thread() or th in reading:
                continue
            if th in drawing or th.name in ("poros-draw", "poros-repaint"):
                th.join(timeout=max(0.0, until - time.monotonic()))

    def _tab_key(self, kind: str, ch: str) -> bool:
        """The keys that move between tabs; True when the key was one."""
        scr = self.screen
        self._note, self._error = "", []
        if kind == "theme":
            self._retheme(switched=True)
            return True
        if scr.over is not None:  # a card over the tab has the keys until it closes
            return False
        if (kind == "char" and ch in "123456789" and not self.s.buf
                and not scr.tab.listening):  # no command starts with a digit
            scr.switch(int(ch) - 1)
        elif kind == "alt":  # ⌥1-9, still, where the terminal sends it
            scr.switch(int(ch) - 1)
        elif kind == "ctrl-right":
            scr.step(1)
        elif kind == "ctrl-left":
            scr.step(-1)
        elif kind == "page-up":
            scr.scroll(10)
        elif kind == "page-down":
            scr.scroll(-10) if scr.tab.scroll > 10 else scr.scroll(0)
        elif kind == "ctrl-w" and not self.s.buf:
            self._close_tab()
        elif kind == "ctrl-n" and not self.s.buf:  # ctrl+n (or ⌥n): an empty new tab
            from poros.cli.ui import tui

            scr.open(tui.Tab("new", centre=True, fit=_blank))
        else:
            return False
        return True

    def _close_tab(self) -> None:
        """ctrl+w: the open tab goes. A running one asks first: the second
        ctrl+w stops its command, then closes it."""
        from poros.cli.ui import tui

        scr = self.screen
        t = scr.tab
        if not scr.active:
            return
        if t.running and not t.listening and not t.closing:  # work, not a card waiting on you
            t.closing = True
            self._note = f"still running · {c.key_name('ctrl-w')} again stops it and closes the tab"
            return
        if t.running:
            tui.interrupt(t)
            t.gone = True  # it goes when its command ends
        scr.close()
