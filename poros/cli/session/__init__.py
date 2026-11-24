"""``poros`` on a terminal: the interactive session (fx's interaction model).

Scrollback holds the conversation: each submitted line stays as typed
and the command's own events follow it. A footer stays pinned under the
last row of scrollback and is redrawn on every key:

    ─────────────────────────────────      the composer's box
    ▛▜ ❯ train train.yaml --steps 50       mark in the accent, ghost in dim
    ─────────────────────────────────      then the statusline, or the picker:
    Commands 15 · type to filter     1–6
                                           (gap)
      /init      Write a ready-to-run …    selected row bright, the rest dim
      /train     Train with a YAML config
    ─────────────────────────────────
    ↑↓ navigate     enter use     esc close
    RTX 4090 24.0 GB · train.yaml · Qwen2.5-7B · ~/runs (main)      [right tag]

Commands run inside the session through the same output sink as one-shot
``poros <cmd>`` (poros/cli/ui/term.py); their live surfaces keep a busy
footer under them via ``components.status_hook``. Keys: enter run · tab
complete · → take the suggestion · ↑↓ history/picker · esc close, esc esc
clear · ctrl+c clear, twice on empty to exit · ctrl+d exit · ctrl+l clear
screen · ctrl+u / ctrl+w / ctrl+a / ctrl+e. Tabs: 1-9 on an empty prompt goes
to one · ctrl+n a new tab · ctrl+w on an empty prompt closes it. Every app key
is one modifier + one letter, the same letter on every card: alt on Linux,
ctrl on a Mac (both work on both; see components.key_name).

This module holds the session: its keys, its state and its dispatch. What it draws (home, picker,
footer) is ``session.screen``; its tabs are ``session.tabs``.
"""

from __future__ import annotations

import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from poros.cli import registry, theme
from poros.cli.session.parts import (
    ARM_S,
    EXIT_WORDS,
    MENU,
    Picker,
    _cols,
    _known,
    _unknown,
    prompt,
)
from poros.cli.session.screen import _Screen
from poros.cli.session.tabs import _Tabs
from poros.cli.ui import components as c
from poros.cli.ui.keys import (
    _read_paste,
    read_key,
)
from poros.cli.ui.live import SYNC_OFF, SYNC_ON, defer, take_pending, up
from poros.cli.ui.style import Palette, resolve
from poros.cli.ui.text import Line, S, render_line


def echo_lead(line: str, pal: Palette) -> str:
    """A command as it stays in scrollback, under the composer's cleared top
    rule and with a blank after: ``▛▜ ❯ /train``."""
    return ("\x1b[K\n" + render_line(prompt() + [(line, "hi")], pal) + "\x1b[K\n"
            + "\x1b[K\n")


SHEET_VERBS = frozenset(
    "train export prove runs models report settings data recipes eval".split())


SHELL_WORDS = {
    "help": ("[command]", "Show commands, or one command's flags"),
    "clear": ("", "Clear the screen"),
    "settings": ("", "Theme, motion and what Poros remembers"),
    "report": ("[last|<run>]", "A run in depth, the same as /runs <run>"),
    "theme": ("[auto|dark|light]", "Switch the color theme"),
    "version": ("", "Show the Poros version"),
    "quit": ("", "Leave the session"),
}
# The verbs first, in the order of the work (poros.cli.registry.help_order); the commands kept for
# researchers and agents after.
VERB_ORDER = registry.help_order()  # poros.cli.registry
SECTION = {c.name: c.section for c in registry.COMMANDS if c.section}


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------


def _keys_mode(fd: int) -> None:
    """Keys one at a time, and ctrl+c as a key, not a signal: a SIGINT that
    lands while the screen draws would end the session past its quit guard."""
    import termios
    import tty

    tty.setcbreak(fd, termios.TCSANOW)
    attrs = termios.tcgetattr(fd)
    try:
        off = os.fpathconf(fd, "PC_VDISABLE")
    except (OSError, ValueError):
        off = 0xFF if sys.platform == "darwin" else 0
    attrs[6][termios.VINTR] = bytes([off])
    attrs[0] &= ~termios.IXON  # ctrl+s is a key here (save), not flow control
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


@dataclass
class State:
    buf: str = ""
    caret: int = 0
    sel: int = 0
    offset: int = 0
    closed_for: Optional[str] = None  # picker dismissed (esc) for this buffer
    hist: List[str] = field(default_factory=list)
    hist_idx: int = 0
    armed: Optional[Tuple[str, float]] = None  # ("esc"|"ctrl-c"|"quit", when)


class Session(_Screen, _Tabs):
    def __init__(self, animate: bool = True) -> None:
        self.out = sys.stdout
        self.animate = animate
        self.pal: Palette = resolve(self.out)
        self.s = State()
        self.footer_h = 0
        self.cursor_row = 0
        # the tallest the footer has been since the last command: it never
        # shrinks back, so closing the menu leaves room instead of a jump
        self.floor = 0
        self.fd = None  # set when the session starts reading keys
        self._gpu = None
        self._status_cache: Tuple[float, List[str]] = (0.0, [])
        self._runs: Optional[bool] = None  # any finished run: the home list offers /eval
        self._found: Optional[Tuple[str, str]] = None  # (command, what) for the data here
        self._top = ""  # the command the home list's first row runs
        self._home_cmds: List[str] = []
        self._quiet = False  # the first prompt, under the start box: the list waits closed
        self._new_tab = False  # ⇧⏎: the next command opens a tab of its own
        self._note = ""  # one line on the status row until the next key (ctrl+w on a run)
        self._quit_key = c.key_name("ctrl-d")  # the key that asked to quit while work runs
        self._error: Line = []  # a command that could not run: said here, not in a tab
        self._stderr: Optional[int] = None  # the terminal's stderr, while the log has it
        self.screen = None  # the tabbed full screen, when the session runs on it (ui/tui.py)

    # -- environment -------------------------------------------------------

    def gpu(self) -> str:
        if self._gpu is None:
            from poros.cli.ui import simulate

            if simulate.enabled():
                self._gpu = f"{simulate.GPU[0]} {simulate.GPU[1]:.1f} GiB"
            else:
                from poros.cli.banner import _gpu_line

                line = _gpu_line()
                none = "no Apple silicon GPU" if sys.platform == "darwin" else "no CUDA GPU"
                self._gpu = line.replace(" · ", " ") if line else none
        return self._gpu

    def echo(self, line: str) -> Line:
        """The typed line as it stays in scrollback: ``▛▜ ❯ /train``."""
        return prompt() + [(line, "hi")]

    def run(self) -> None:
        import termios

        from poros import __version__
        from poros.cli.banner import print_banner
        from poros.cli.shell import _load_history_lines, _save_history_lines, console
        from poros.cli.ui import intro, tui

        self.fd = sys.stdin.fileno()
        theme.prime_detection()
        self.pal = resolve(self.out)
        if tui.enabled(self.out):
            self._open_screen()
        else:
            print_banner(console, __version__, animate=self.animate)
        # under the start box the first prompt waits closed: the box shows
        # what to do, and ⏎ opens the list it promises
        self._quiet = _cols() >= intro.BOX_MIN
        self.s.hist = _load_history_lines()
        self.s.hist_idx = len(self.s.hist)
        old = termios.tcgetattr(self.fd)
        # under a live surface: the composer at rest (in tabs the screen draws it)
        c.status_hook = (lambda *_: []) if self.screen is not None else self.busy_footer
        c.in_session = True
        try:
            while True:
                _keys_mode(self.fd)
                self.out.write("\x1b[?2004h")  # bracketed paste
                line = self.edit()
                self.out.write("\x1b[?2004l")
                termios.tcsetattr(self.fd, termios.TCSADRAIN, old)
                if line is None:
                    break
                if not self.submit(line, old):
                    break
        finally:
            self._close_screen()
            c.status_hook = None
            c.in_session = False
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old)
            self.out.write("\x1b[?2004l\x1b[?25h")
            self.out.flush()
            _save_history_lines(self.s.hist)

    def edit(self) -> Optional[str]:
        """Edit one line in the footer. None = leave the session."""
        st = self.s
        st.buf, st.caret, st.sel, st.offset = "", 0, 0, 0
        st.closed_for, self._quiet = ("" if self._quiet else None), False
        st.hist_idx = len(st.hist)
        self.draw()
        while True:
            kind, ch = read_key(self.fd)
            pk = self.picker()
            now = time.monotonic()
            armed = st.armed if st.armed and now - st.armed[1] < ARM_S else None
            if kind != "ctrl-c" and kind != "esc":
                st.armed = None

            if kind == "ctrl-d" and not st.buf:  # quit, from anywhere: one meaning
                if self._quit_asked(armed, now, c.key_name("ctrl-d")):
                    return self._leave()
                self.draw()
                continue
            if self.screen is not None and self._tab_key(kind, ch):
                self.draw()
                continue
            if self.screen is not None and self.screen.tab.listening:
                owner = getattr(self.screen.tab.card, "__self__", None)
                if (kind == "char" and ch == "/" and owner is not None
                        and getattr(owner, "takes_slash", lambda: False)()):
                    # a command typed over a card: the card closes, the prompt has it
                    tab = self.screen.tab
                    tab.keys.put(("ctrl-c", ""))
                    until = time.monotonic() + 0.5  # the next key is the prompt's, not the card's
                    while tab.listening and time.monotonic() < until:
                        time.sleep(0.01)
                    self._set("/")
                    self.draw()
                    continue
                # a card or menu in the tab on screen has the keys
                if kind == "paste":
                    ch = _read_paste(self.fd)
                self.screen.tab.keys.put((kind, ch))
                continue
            if (kind == "ctrl-c" and self.screen is not None and not st.buf
                    and self.screen.tab.running):
                # a habit key never ends hours of work: the second press stops it
                if armed and armed[0] == "stop":
                    from poros.cli.ui import tui

                    tui.interrupt(self.screen.tab)
                    st.armed = None
                else:
                    st.armed = ("stop", now)
                self.draw()
                continue
            if kind == "shift-enter":  # the same as enter, into a tab of its own
                self._new_tab, kind = True, "enter"
            if kind == "enter":
                if pk is not None and pk.kind == "home":
                    if not pk.items:
                        self.draw()
                        continue
                    st.buf = self._home_cmds[st.sel]
                    return self._finish()
                if not st.buf and st.closed_for == "":  # esc closed the list: ⏎ reopens it
                    st.closed_for = None
                    self.draw()
                    continue
                if not st.buf and c.next_step:  # the step the last command suggested
                    st.buf = c.next_step
                if pk is not None and pk.kind == "commands" and pk.items:
                    name = pk.items[st.sel][0]
                    if _needs_args(name.lstrip("/")):
                        self._set(name + " ")
                        self.draw()
                        continue
                    st.buf = name
                return self._finish()
            if kind == "tab" and pk is not None and pk.kind == "home":
                pass  # the home list has nothing to complete; ⏎ opens
            elif kind == "tab":
                if pk is not None and pk.items:
                    self._accept(pk, pk.items[st.sel][0])
                else:
                    g = self.ghost()
                    if g:
                        self._set(st.buf + g)
            elif kind == "right":
                if st.caret < len(st.buf):
                    st.caret += 1
                else:
                    g = self.ghost()
                    if g:
                        self._set(st.buf + g)
            elif kind == "left":
                st.caret = max(0, st.caret - 1)
            elif kind in ("home", "ctrl-a"):
                st.caret = 0
            elif kind in ("end", "ctrl-e"):
                st.caret = len(st.buf)
            elif kind == "up":
                if pk is not None and pk.items:
                    st.sel = (st.sel - 1) % len(pk.items)
                elif st.hist and st.hist_idx > 0:
                    st.hist_idx -= 1
                    self._set(st.hist[st.hist_idx])
                    st.closed_for = st.buf
            elif kind == "down":
                if pk is not None and pk.items:
                    st.sel = (st.sel + 1) % len(pk.items)
                elif st.hist_idx < len(st.hist):
                    st.hist_idx += 1
                    self._set(st.hist[st.hist_idx] if st.hist_idx < len(st.hist) else "")
                    st.closed_for = st.buf
            elif kind == "backspace":
                if st.caret:
                    st.buf = st.buf[: st.caret - 1] + st.buf[st.caret :]
                    st.caret -= 1
                    st.sel = st.offset = 0
            elif kind == "delete":
                st.buf = st.buf[: st.caret] + st.buf[st.caret + 1 :]
            elif kind == "ctrl-u":
                self._set("")
            elif kind == "ctrl-k":
                st.buf = st.buf[: st.caret]
            elif kind == "ctrl-w":
                head = st.buf[: st.caret].rstrip()
                cut = head.rfind(" ") + 1
                st.buf = head[:cut] + st.buf[st.caret :]
                st.caret = cut
            elif kind == "ctrl-l" and self.screen is not None:
                self.screen.paint(force=True)
            elif kind == "ctrl-l":
                self.out.write("\x1b[2J\x1b[H")
                self.footer_h = self.cursor_row = 0
            elif kind == "esc":
                if pk is not None and st.buf == "/":  # the list was all a lone / asked for
                    self._set("")
                    st.closed_for = ""
                elif pk is not None:
                    st.closed_for = st.buf
                elif st.buf:
                    if armed and armed[0] == "esc":
                        self._set("")
                        st.armed = None
                    else:
                        st.armed = ("esc", now)
            elif kind == "ctrl-c":
                if st.buf:
                    self._set("")
                    st.armed = None
                elif armed and armed[0] in ("ctrl-c", "quit"):
                    if self._quit_asked(armed, now, c.key_name("ctrl-c")):
                        return self._leave()
                else:
                    st.armed = ("ctrl-c", now)
            elif kind == "eof":
                if not st.buf:
                    return self._leave()
            elif kind == "paste":
                text = _read_paste(self.fd)
                self._insert(text.replace("\n", " ").replace("\r", ""))
            elif kind == "char":
                # every line is a slash command: a letter on an empty prompt starts one
                self._insert(ch if st.buf or ch in "/ " else "/" + ch)
                st.hist_idx = len(st.hist)
            self.draw()

    def _retheme(self, switched: bool = False) -> None:
        """Draw every tab in the palette the theme now gives. ``switched``: the
        terminal itself went light or dark (mode 2031), so ask it its colors
        again. Only then: /settings and /theme run off the key-reading thread,
        which would take the terminal's reply as keys, and the query would
        keep "no answer" -- the start box would lose its filled ground."""
        if switched:
            theme.forget_colors()
        self.pal = resolve(self.out)
        if self.screen is not None:
            self.screen.pal = self.pal
            self.screen.paint(force=True)

    def _set(self, text: str) -> None:
        self.s.buf, self.s.caret = text, len(text)
        self.s.sel = self.s.offset = 0

    def _insert(self, text: str) -> None:
        st = self.s
        st.buf = st.buf[: st.caret] + text + st.buf[st.caret :]
        st.caret += len(text)
        st.sel = st.offset = 0
        st.closed_for = None

    def _accept(self, pk: Picker, text: str) -> None:
        st = self.s
        if pk.kind == "commands":
            self._set(text + " ")
            return
        tail = "" if text.endswith("/") else " "
        self._set(st.buf[: pk.token_start] + text + tail)

    def _finish(self) -> str:
        # the footer stays until submit() paints the card over it in one write
        return self.s.buf.strip()

    # -- running ------------------------------------------------------------------

    def submit(self, line: str, cooked) -> bool:
        """Echo the line as a card, run it. False = leave the session."""
        import termios

        if not line:
            self.erase()
            return True
        st = self.s
        if not st.hist or st.hist[-1] != line:
            st.hist.append(line)
        if self.screen is not None:
            self._quiet = True  # the canvas shows the result; ⏎ opens the list
            return self._run_in_tab(line)
        # the line stays in scrollback exactly as it was typed, prompt and all,
        # painted over the footer in one write (no blank frame between)
        climb = up(self.cursor_row)
        words = line.split()
        if words[0].lstrip("/").lower() in SHEET_VERBS and "--json" not in words:
            # a card is coming: the composer stays until what the card settles
            # on replaces it, never a blank (the rows: the composer's top rule,
            # the line, a blank)
            defer(self.out, lead=echo_lead(line, self.pal), lead_rows=self.cursor_row + 2,
                  climb=climb)
        else:
            self.out.write(SYNC_ON + climb + echo_lead(line, self.pal)[:-4] + "\x1b[J\n"
                           + SYNC_OFF)
        self.out.flush()
        self.footer_h = self.cursor_row = self.floor = 0
        # commands see a normal terminal, minus echo: keys typed while a run
        # draws its live view must not scribble over it
        quiet = termios.tcgetattr(self.fd)
        quiet[3] &= ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSADRAIN, quiet)
        c.next_step = None  # a suggestion lasts until the next command
        try:
            keep = self.dispatch(line)
        finally:
            # type-ahead is kept (as in fx): what you typed during a run is
            # the next line; echo was off, so it never drew over the run
            termios.tcsetattr(self.fd, termios.TCSADRAIN, cooked)
        from poros.cli.ui import complete

        complete._CACHE.clear()
        self._status_cache = (0.0, [])
        self._runs = self._found = None
        if keep:
            # a live area the command closed without output is cleared here,
            # then one blank line separates the turns
            self.out.write(take_pending() + "\x1b[J")
        return keep

    def dispatch(self, line: str) -> bool:
        from poros.cli import views
        from poros.cli.ui import term

        t = term.get()
        try:
            argv = shlex.split(line)
        except ValueError as exc:
            t.print(c.error_event("Can't parse", str(exc), t.width()))
            return True
        if not argv:
            return True
        if argv[0] == "poros":
            argv = argv[1:] or ["/help"]
        if argv[0].lower() in EXIT_WORDS:
            return False
        if not argv[0].startswith("/"):
            # Slash commands only: a line of words is never guessed at.
            t.print(
                c.error_event("Commands start with /", line, t.width())
                + [c.rail(S("type ", "dm"), S("/", "hi"), S(" to see them, or ", "dm"),
                          S("/" + argv[0].lower(), "hi"), S(" if that was one", "dm"))]
                if argv[0].lower() in _known()
                else c.error_event("Commands start with /", line, t.width())
                + [c.rail(S("type ", "dm"), S("/", "hi"), S(" to see them", "dm"))]
            )
            return True
        head = argv[0].lstrip("/").lower()
        argv[0] = head
        if head in EXIT_WORDS:
            return False
        if head == "clear":
            self.out.write("\x1b[2J\x1b[3J\x1b[H")
            return True
        if head == "help" and len(argv) == 1:
            t.print(views.help_screen([r for r in _command_rows() if r[0] in dict(MENU)],
                                      t.width()))
            return True
        if head == "help":
            argv = [argv[1], "--help"]
        if head == "report":  # /report [last|<run>]: the run's page, nothing to write
            argv = ["runs", *(_last_done() if argv[1:] == ["last"] else argv[1:])]
            head = "runs"
        if head == "share":  # /share: the last finished run's recipe file, here
            argv, head = ["recipes", "share", *argv[1:]], "recipes"
        if head == "settings":
            from poros.cli import settings

            settings.open_settings(self)
            self._retheme()
            return True
        if head == "version":
            from poros import __version__

            t.print([c.event("Poros", f"v{__version__}")])
            return True
        if head == "theme":
            mode = (
                theme.set_mode(argv[1])
                if len(argv) > 1
                else theme.set_mode(theme.next_mode(theme.current_mode()))
            )
            self._retheme()
            t.print(
                [c.event("Theme", mode, "auto follows the terminal" if mode == "auto" else None)]
            )
            return True
        from poros.cli import app as words_app
        from poros.cli import contract
        from poros.cli.ui import complete

        contract.reset()
        term.argv = list(argv)
        try:
            handled = words_app.dispatch(argv)
        except KeyboardInterrupt:
            t.print(views.note("Stopped"))
            handled = 130
        except Exception as exc:  # noqa: BLE001 - the session must survive any command
            t.print(c.error_event("Error", f"{type(exc).__name__}: {exc}", t.width()))
            handled = 1
        finally:
            term.argv = None
        if handled is not None:
            return True
        if head not in complete.command_tree():
            t.print([_unknown(head)])
            return True
        _run_typer(argv)
        return True


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _last_done() -> List[str]:
    from poros import runs

    done = next((r for r in runs.all_runs() if r.state in ("succeeded", "failed", "cancelled")),
                None)
    return [done.id] if done else []


def _run_typer(argv: List[str]) -> None:
    import typer
    from typer.main import get_command

    from poros.cli import main as cli_main
    from poros.cli import root as cli_root
    from poros.cli import views
    from poros.cli.ui import term

    click_mod = cli_root._click_module()
    term.argv = list(argv)
    try:
        get_command(cli_main.app).main(args=argv, prog_name="poros", standalone_mode=False)
    except (typer.Exit, SystemExit):
        pass
    except (typer.Abort, KeyboardInterrupt):
        term.get().print(views.note("Stopped"))
    except click_mod.exceptions.UsageError as exc:
        cli_main._render_usage_error(exc)
    except click_mod.exceptions.ClickException as exc:
        t = term.get()
        t.print(c.error_event("Error", exc.format_message(), t.width()))
    except Exception as exc:  # noqa: BLE001 - the session must survive any command
        t = term.get()
        t.print(c.error_event("Error", f"{type(exc).__name__}: {exc}", t.width()))
    finally:
        term.argv = None


def _describe(name: str, fallback: str) -> str:
    """The curated one-liner (shell.COMMANDS), else the command's docstring."""
    from poros.cli.shell import COMMANDS

    if name in SHELL_WORDS:
        return SHELL_WORDS[name][1]
    return COMMANDS.get(name, ("", fallback or ""))[1]


def _command_rows():
    from poros.cli.ui.complete import command_tree

    rows = []
    for name, cmd in command_tree().items():
        if name in EXIT_WORDS:
            continue
        if name in SHELL_WORDS:
            args, desc = SHELL_WORDS[name]
            rows.append((name, args, desc, "shell"))
        else:
            from poros.cli.shell import COMMANDS

            args = " ".join(f"<{a}>" for a in cmd.args) if cmd.args else ""
            if cmd.sub:
                args = "|".join(cmd.sub)
            if SECTION.get(name) == "verbs":  # a verb's words, not its config-file form
                args = COMMANDS.get(name, (args,))[0]
            rows.append((name, args, _describe(name, cmd.help), SECTION.get(name, "advanced")))
    order = list(VERB_ORDER)
    return sorted(rows, key=lambda r: order.index(r[0]) if r[0] in order else len(order))


def _needs_args(name: str) -> bool:
    from poros.cli.ui.complete import command_tree

    cmd = command_tree().get(name)
    if cmd is None or name in SHELL_WORDS:
        return False
    from poros.cli.app import BUILT

    if name in BUILT:  # a command with a settings sheet: enter opens it
        return False
    if cmd.sub:
        return True
    # positionals that typer marks required show up as args without defaults;
    # init's path has a default, so it runs bare
    return bool(cmd.args) and name not in ("init",)


def available() -> bool:
    if os.environ.get("POROS_NO_MENU"):
        return False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except ImportError:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def run(animate: bool = True) -> None:
    from poros.cli import settings

    settings.apply_saved()
    Session(animate=animate).run()
    from poros.cli.ui.style import resolve as _r

    pal = _r(sys.stdout)
    sys.stdout.write(render_line([("bye.", "dm")], pal) + "\n")
    sys.stdout.flush()


