"""CLI shell / banner / machine guarantees.

The interactive shell and banner animation are strictly TTY-gated: any
piped, redirected, CI, or agent invocation of bare ``poros`` must show the
help and exit immediately -- no animation, no REPL, no hang. ``machine``
must keep working without torch, and ``machine --json`` must emit valid
machine-readable JSON. All tests here run without torch installed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI = [sys.executable, "-m", "poros.cli.main"]

# Escape sequences only the banner animation emits: hide-cursor and the
# cursor-up frame redraw.
_ANIMATION_MARKERS = ("\x1b[?25l", "\x1b[6A")


def _run(args, *, stdin=subprocess.DEVNULL, input_bytes=None, timeout=30, env=None):
    """Run the CLI with non-TTY stdio and return the completed process."""
    full_env = dict(os.environ)
    full_env.pop("NO_COLOR", None)  # gating must not depend on NO_COLOR
    # Nor on ambient color-forcing (CI runners and agent shells export these;
    # rich honors them even on pipes, which is opt-in behavior we don't test).
    full_env.pop("FORCE_COLOR", None)
    full_env.pop("CLICOLOR_FORCE", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        CLI + args,
        stdin=stdin if input_bytes is None else None,
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        env=full_env,
    )


# ---------------------------------------------------------------------------
# TTY gating: bare `poros` on a non-TTY must never animate or enter the REPL
# ---------------------------------------------------------------------------


def test_bare_poros_non_tty_shows_help_and_exits():
    """`poros | cat`-style invocation: the help and exit 0, as `poros --help`; no animation, no
    hang, and no error after the help."""
    proc = _run([])
    out = proc.stdout.decode(errors="replace")
    combined = out + proc.stderr.decode(errors="replace")
    assert proc.returncode == 0
    assert "Usage" in out and "✕" not in combined
    assert "train" in out and "check" in out and "doctor" not in out
    for marker in _ANIMATION_MARKERS:
        assert marker not in combined
    # The big banner art must not appear on a pipe.
    assert "██████ ██████" not in combined  # the banner's pixel mark
    assert "poros ❯" not in combined


def test_bare_poros_with_piped_stdin_does_not_enter_repl():
    """`echo | poros`: stdin is a pipe, so no REPL -- must exit immediately."""
    proc = _run([], input_bytes=b"\n/doctor\n/quit\n")
    out = proc.stdout.decode(errors="replace")
    assert proc.returncode == 0
    assert "Usage" in out
    assert "poros ❯" not in out + proc.stderr.decode(errors="replace")


def test_bare_poros_plain_flag_non_tty_still_safe():
    """The --plain escape hatch must not weaken the TTY gate."""
    proc = _run(["--plain"])
    assert proc.returncode == 0
    assert "Usage" in proc.stdout.decode(errors="replace")


def test_version_flag_output_unchanged():
    from poros import __version__

    proc = _run(["--version"])
    assert proc.returncode == 0
    assert proc.stdout.decode().strip() == f"poros {__version__}"


# ---------------------------------------------------------------------------
# Shell modules stay torch-free and import-light
# ---------------------------------------------------------------------------


def test_banner_and_shell_modules_do_not_import_torch():
    code = (
        "import sys\n"
        "import poros.cli.banner, poros.cli.shell\n"
        "assert 'torch' not in sys.modules, 'shell modules must stay torch-free'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=30,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stdout.decode().strip() == "ok"


# ---------------------------------------------------------------------------
# doctor: gone; `poros machine` is the environment check
# ---------------------------------------------------------------------------


def test_doctor_is_an_unknown_command():
    proc = _run(["doctor"])
    combined = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    assert proc.returncode == 2
    assert "No such command 'doctor'" in combined


# ---------------------------------------------------------------------------
# Real-TTY path: bare `poros` on a pty opens the shell and exits cleanly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(os, "openpty") or sys.platform == "win32",
    reason="requires a POSIX pty",
)
def test_bare_poros_on_pty_opens_shell_and_quits():
    import pty
    import select
    import time

    master, slave = pty.openpty()
    env = dict(os.environ, TERM="xterm-256color", NO_COLOR="1", COLUMNS="100")
    proc = subprocess.Popen(
        CLI,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        cwd=REPO_ROOT,
        start_new_session=True,
    )
    os.close(slave)
    out = b""
    try:
        time.sleep(1.5)  # banner is capped well under 1s; be generous in CI
        os.write(master, b"/version\n/quit\n")
        deadline = time.time() + 25
        while time.time() < deadline:
            r, _, _ = select.select([master], [], [], 0.5)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
            elif proc.poll() is not None:
                break
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)

    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", out.decode(errors="replace"))
    assert proc.returncode == 0
    from poros import __version__

    assert "██████ ██████" in text  # the start box's pixel mark
    assert __version__ in text
    assert "bye." in text


# ---------------------------------------------------------------------------
# Live slash-command menu: pure filter/hint logic (no tty needed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Live menu over a pty: filters, completes, dispatches, quits cleanly
# ---------------------------------------------------------------------------


# The shell under test runs in a child process, so a fixture's monkeypatch cannot reach it: it
# would read this machine's free host RAM (the cgroup room of a memory-capped test run, or what
# the rest of the suite left) and the user's real home -- runs, remembered settings -- and what a
# menu row runs could depend on either. The child gets a fresh home and a pinned free host RAM
# (as tests/cli/test_train_runs.py pins its child's) through a sitecustomize on its PYTHONPATH.
_PTY_HOST_FREE_GIB = 64.0


def _pty_env(env_extra=None) -> tuple[dict, Path]:
    """The child's environment, and the folder holding its home and pin (removed by the caller)."""
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="poros-pty-"))
    (root / "home").mkdir()
    (root / "pin").mkdir()
    (root / "pin" / "sitecustomize.py").write_text(
        "import poros.core.env\n"
        f"poros.core.env.host_ram_free = lambda: {_PTY_HOST_FREE_GIB}\n")
    path = [str(root / "pin"), str(REPO_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    env = dict(os.environ, TERM="xterm-256color", NO_COLOR="1", COLUMNS="100",
               HOME=str(root / "home"), PYTHONPATH=os.pathsep.join(path))
    for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        env.pop(name, None)
    if env_extra:
        env.update(env_extra)
    return env, root


def _drive_pty(script: bytes, settle=1.5, read_timeout=25, env_extra=None):
    """``_drive_pty_once``, tried once more if the shell did not exit in time. A shell that never
    exits fails both tries and so still fails the test; a slow machine gets a second go."""
    try:
        return _drive_pty_once(script, settle, read_timeout, env_extra)
    except subprocess.TimeoutExpired:
        return _drive_pty_once(script, settle, read_timeout, env_extra)


def _drive_pty_once(script: bytes, settle=1.5, read_timeout=25, env_extra=None):
    """Boot the shell on a pty (``_pty_env``), wait for the banner, feed ``script``, collect."""
    import pty
    import select
    import shutil
    import time

    master, slave = pty.openpty()
    env, root = _pty_env(env_extra)
    proc = subprocess.Popen(
        CLI, stdin=slave, stdout=slave, stderr=slave, env=env,
        cwd=REPO_ROOT, start_new_session=True,
    )
    os.close(slave)
    out = b""
    try:
        # The banner: the first output, then a quiet moment. A fixed sleep alone is too short on a
        # slow machine, and keys typed before the shell reads them are lost.
        time.sleep(settle)
        boot = time.time() + 20
        quiet_since = None
        while time.time() < boot:
            r, _, _ = select.select([master], [], [], 0.25)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
                quiet_since = time.time()
            elif out and quiet_since is not None and time.time() - quiet_since >= 0.75:
                break
        os.write(master, script)
        deadline = time.time() + read_timeout
        while time.time() < deadline:
            r, _, _ = select.select([master], [], [], 0.5)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
            elif proc.poll() is not None:
                break
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)
        shutil.rmtree(root, ignore_errors=True)
    return proc.returncode, out.decode(errors="replace")


@pytest.mark.skipif(
    not hasattr(os, "openpty") or sys.platform == "win32",
    reason="requires a POSIX pty",
)
def test_pty_menu_filters_and_shows_descriptions():
    """Typing a partial slash command renders a live, descriptive menu."""
    # Type '/machine' (the menu filters live to its row), Enter runs it, quit.
    rc, text = _drive_pty(b"/machine\n/quit\n")
    assert rc == 0
    # The command bar's description rendered while typing (it lives only in
    # the composer, never in the card's own output).
    assert "what this machine can do" in text
    # And the command actually dispatched: the /machine card opened.
    assert "this machine" in re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
    assert "bye." in text


@pytest.mark.skipif(
    not hasattr(os, "openpty") or sys.platform == "win32",
    reason="requires a POSIX pty",
)
def test_pty_menu_tab_completes_and_dispatches():
    """Tab accepts the highlighted command; the completed command dispatches."""
    # '/mach' + Tab -> '/machine ', then Enter runs it.
    rc, text = _drive_pty(b"/mach\t\n/quit\n")
    assert rc == 0
    assert "what this machine can do" in text  # the bar's description rendered
    assert "this machine" in re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)  # dispatched
    assert "bye." in text


@pytest.mark.skipif(
    not hasattr(os, "openpty") or sys.platform == "win32",
    reason="requires a POSIX pty",
)
def test_pty_menu_arrow_navigation_runs_selected_command():
    """Down-arrow moves the selection; Enter runs the highlighted command."""
    # Bare '/', Down once to move off the first row, Enter runs whatever is
    # selected. We only assert the shell stays healthy and exits cleanly -- the
    # navigation must not traceback or hang.
    rc, text = _drive_pty(b"/\x1b[B\x1b[B\x1b\n/quit\n")
    assert rc == 0, text
    assert "bye." in text, text
    assert "Traceback" not in text, text[text.find("Traceback"):]  # the whole traceback


def test_quitting_waits_for_the_shells_own_threads_and_a_typed_quit_stops_them_too():
    """A daemon thread of the shell caught writing when Python exits aborts it
    (_enter_buffered_busy). So the quit path waits for the shell's own threads, and a typed
    /quit takes that path as ctrl-d does."""
    import threading
    import time
    import types

    from poros.cli.session.tabs import _Tabs

    drawn = threading.Event()

    def draw():
        time.sleep(0.4)
        drawn.set()

    th = threading.Thread(target=draw, name="poros-draw", daemon=True)
    th.start()
    shell = types.SimpleNamespace(screen=types.SimpleNamespace(tabs=[]))
    _Tabs._stop_tabs(shell)
    assert drawn.is_set() and not th.is_alive()

    # a card that is still reading keys (/settings over a tab) is not drawing: quit does not wait
    release = threading.Event()
    card = threading.Thread(target=release.wait, name="poros-settings", daemon=True)
    card.start()
    over = types.SimpleNamespace(thread=card, running=False, listening=1)
    shell = types.SimpleNamespace(screen=types.SimpleNamespace(tabs=[], over=over))
    began = time.monotonic()
    _Tabs._stop_tabs(shell)
    assert time.monotonic() - began < 1.0
    release.set()
    card.join(timeout=5)

    # a run's own worker (the Mac's layer reader, idle after a run) draws nothing: not waited for
    idle = threading.Event()
    reader = threading.Thread(target=idle.wait, name="poros-read_0", daemon=True)
    reader.start()
    shell = types.SimpleNamespace(screen=types.SimpleNamespace(tabs=[]))
    began = time.monotonic()
    _Tabs._stop_tabs(shell)
    assert time.monotonic() - began < 1.0 and reader.is_alive()
    idle.set()
    reader.join(timeout=5)

    stopped = []
    shell = types.SimpleNamespace(screen=types.SimpleNamespace(tabs=[]),
                                  _stop_tabs=lambda: stopped.append(True))
    assert _Tabs._run_in_tab(shell, "/quit") is False
    assert stopped == [True]


def test_quitting_waits_for_a_tab_that_left_the_list_while_it_still_writes(monkeypatch):
    """A tab that has left its loop takes itself off the screen's tabs, and its thread may still
    be writing its last lines. The quit saw it at the start, so it waits for it all the same."""
    import threading
    import time
    import types

    from poros.cli.session.tabs import _Tabs
    from poros.cli.ui import tui

    tabs = []
    go, gone, written = threading.Event(), threading.Event(), threading.Event()

    def last_lines():
        go.wait(5)
        tabs.remove(ending)  # off the live list, as a tab that ends does
        gone.set()
        time.sleep(0.4)
        written.set()

    th = threading.Thread(target=last_lines, name="poros-ending", daemon=True)
    ending = types.SimpleNamespace(thread=th, running=False, listening=0)
    # a running tab the quit tells to stop (this thread stands in for its own, never joined)
    other = types.SimpleNamespace(thread=threading.current_thread(), running=True, listening=0)
    tabs.extend([ending, other])
    th.start()

    def interrupt(tab):
        # the ending tab leaves the list while the quit tells the others to stop
        go.set()
        gone.wait(5)
        return False

    monkeypatch.setattr(tui, "interrupt", interrupt)
    _Tabs._stop_tabs(types.SimpleNamespace(screen=types.SimpleNamespace(tabs=tabs)))
    assert gone.is_set() and written.is_set() and not th.is_alive()
