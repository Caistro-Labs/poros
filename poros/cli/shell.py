"""Interactive Poros shell -- a slash-command REPL over the Typer CLI.

Launched ONLY by the bare ``poros`` entry when BOTH stdin and stdout are
TTYs (see ``poros.cli.main``). Slash commands re-dispatch into the existing
Typer app -- no command logic is duplicated here. Torch-free at import.

stdlib + rich only: readline (when available) provides history and tab
completion; no new runtime dependencies.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from poros.cli import registry, theme
from poros.cli.banner import ACCENT, color_enabled, print_banner

# command -> (args hint, description, section)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "machine": ("", "What this machine can do", "verbs"),
    "models": ("[get <model>]", "The models Poros names, with recipes certified here", "verbs"),
    "data": ("<file> [--convert|--fix]", "Check a data file; convert it, or write a clean copy",
             "verbs"),
    "train": ("<model> on <data>", "Fine-tune (or a config: train run.yaml)", "verbs"),
    "recipes": ("[copy|show|save|share]", "Whole setups and the loops they train with", "verbs"),
    "arch": ("[try] <model>", "What a model would take, from its config alone", "verbs"),
    "runs": ("[<run>|stop|continue|report]", "Your runs: watch, stop, continue", "verbs"),
    "eval": ("[<run>] [--data FILE] [--rows N]", "Score a run beside its base: loss, answer ends",
             "verbs"),
    "export": ("<run> [as merged|adapter]", "Turn a run into a model you can serve", "verbs"),
    "prove": ("[<model>|<receipt>]", "Train streamed and fully resident, and compare", "verbs"),
    "docs": ("[<topic> [<words>]]", "the docs for people, from this install: no website", "verbs"),
    "init": ("[train.yaml] [-m MODEL]", "Write a ready-to-run starter config", "workflow"),
    "check": ("<model|dir|config.json>", "Is this architecture supported?", "inspect"),
    "schema": ("<config|events|artifact>", "Print a JSON Schema for tooling", "inspect"),
    "bench": ("run|matrix|compare|report …", "Benchmark harness", "results"),
    "report": ("<run_dir|results.jsonl>", "Render an HTML/Markdown report", "results"),
    "leaderboard": ("<results_dir|.jsonl>", "Rank runs by VRAM efficiency", "results"),
    "help": ("[command]", "Show commands (or one command's help)", "shell"),
    "theme": ("[auto|dark|light]", "Switch color theme (auto/dark/light)", "shell"),
    "version": ("", "Show the Poros version", "shell"),
    "clear": ("", "Clear the screen", "shell"),
    "quit": ("", "Exit the shell (also: exit, Ctrl-D)", "shell"),
}
_SECTIONS = ("verbs", "workflow", "inspect", "results", "shell")
# Commands that re-dispatch into the Typer app.
DISPATCHED = frozenset(registry.names(shell=True))  # poros.cli.registry
_EXIT_WORDS = frozenset({"quit", "exit", "q"})

console = Console(highlight=False, color_system=theme.console_color_system())


# ---------------------------------------------------------------------------
# Dispatch into the existing Typer app (no duplicated command logic)
# ---------------------------------------------------------------------------


def _words(argv: list[str]) -> bool:
    """Run ``argv`` as a sentence if it is one (poros/cli/app.py)."""
    from poros.cli import app, contract

    contract.reset()
    try:
        return app.dispatch(argv) is not None
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted.[/dim]")
        return True
    except Exception as exc:  # noqa: BLE001 -- shell must survive anything
        console.print(f"[red]error:[/red] {exc}")
        return True


def _dispatch(argv: list[str]) -> None:
    """Run one CLI invocation in-process, keeping the REPL alive afterwards."""
    from typer.main import get_command

    from poros.cli.main import app
    from poros.cli.root import _click_module

    click_mod = _click_module()
    cmd = get_command(app)
    try:
        cmd.main(args=argv, prog_name="poros", standalone_mode=False)
    except typer.Exit:
        pass  # command finished with an explicit code
    except typer.Abort:
        console.print("[dim]aborted.[/dim]")
    except click_mod.exceptions.UsageError as exc:
        from poros.cli.main import _render_usage_error

        _render_usage_error(exc)
    except click_mod.ClickException as exc:
        exc.show()  # non-usage click errors
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted.[/dim]")
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 -- shell must survive anything
        console.print(f"[red]error:[/red] {exc}")


# ---------------------------------------------------------------------------
# readline: history + tab completion (stdlib only, optional)
# ---------------------------------------------------------------------------


def _history_path() -> Path:
    state = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state"
    )
    return Path(state) / "poros" / "history"


def _completer_factory(readline_mod):  # pragma: no cover - interactive only
    import glob

    names = sorted(COMMANDS) + ["exit"]
    bench_sub = ("run", "matrix", "compare", "report")

    def complete(text: str, state: int):
        buf = readline_mod.get_line_buffer()
        begidx = readline_mod.get_begidx()
        options: list[str] = []
        if begidx == 0:
            for name in names:
                for cand in ("/" + name, name):
                    if cand.startswith(text):
                        options.append(cand + " ")
        else:
            head = buf.lstrip().lstrip("/").split(" ", 1)[0]
            if head == "bench" and not any(s in buf.split() for s in bench_sub):
                options.extend(s + " " for s in bench_sub if s.startswith(text))
            if not options:  # fall back to filesystem paths
                for p in sorted(glob.glob(glob.escape(text) + "*"))[:64]:
                    options.append(p + ("/" if os.path.isdir(p) else " "))
        # de-dup, keep order
        seen: dict[str, None] = dict.fromkeys(options)
        options = list(seen)
        return options[state] if state < len(options) else None

    return complete


def _setup_readline():
    try:
        import readline
    except ImportError:
        return None
    readline.set_completer_delims(" \t")
    readline.set_completer(_completer_factory(readline))
    if "libedit" in (getattr(readline, "__doc__", "") or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    readline.set_history_length(1000)
    hist = _history_path()
    try:
        readline.read_history_file(hist)
    except OSError:
        pass
    return readline


def _save_history(readline_mod) -> None:
    if readline_mod is None:
        return
    hist = _history_path()
    try:
        hist.parent.mkdir(parents=True, exist_ok=True)
        readline_mod.write_history_file(hist)
    except OSError:
        pass


def _starter() -> str:
    """The ghost on an empty prompt: the next step from where you are. Data
    here: train on it; a run: watch or export it; else see it work first."""
    try:
        from poros import runs
        from poros.cli.verbs.train_data import local_files

        last = runs.all_runs()[:1]
        if last and last[0].state == "running":
            return f"/runs {last[0].id}"
        if last and last[0].state == "succeeded":
            return "/export last"
        if local_files(Path.cwd()):
            return "/train"
    except Exception:
        pass
    return "/recipes"


def _ui_sgr(token) -> str:
    from poros.cli.ui.style import resolve

    code = resolve(depth=_ui_depth()).sgr(token)
    return f"\x1b[{code}m" if code else ""


def _ui_depth() -> str:
    if not color_enabled():
        return "none"
    colorterm = os.environ.get("COLORTERM", "").lower()
    return "truecolor" if colorterm in ("truecolor", "24bit") else "256"


def _load_history_lines() -> list[str]:
    """Load recent shell lines (plain text) for raw-mode history recall."""
    try:
        text = _history_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [ln for ln in text.splitlines() if ln.strip()][-1000:]


def _save_history_lines(lines: list[str]) -> None:
    hist = _history_path()
    try:
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text("\n".join(lines[-1000:]) + "\n", encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Help panel
# ---------------------------------------------------------------------------


def _print_help() -> None:
    color = color_enabled()
    accent = ACCENT if color else "bold"
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=2)
    grid.add_column(min_width=14)
    grid.add_column(min_width=26, style="dim")
    grid.add_column()
    for section in _SECTIONS:
        grid.add_row("", Text(section, style=f"bold {accent}" if color else "bold"), "", "")
        for name, (args, desc, sec) in COMMANDS.items():
            if sec != section:
                continue
            grid.add_row(
                "",
                Text(f"  /{name}", style=accent),
                Text(args, style="dim"),
                Text(desc),
            )
        grid.add_row("", "", "", "")
    console.print()
    console.print(grid)
    console.print(
        "  [dim]The slash is optional --[/dim] [bold]check Qwen/Qwen2.5-7B[/bold] "
        "[dim]works too. Any command accepts[/dim] [bold]--help[/bold][dim].[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# The REPL
# ---------------------------------------------------------------------------


def _prompt(readline_active: bool) -> str:
    if not color_enabled():
        return "poros > "
    # Same prompt as the raw editor. \001/\002 mark zero-width sequences so
    # readline keeps its column math right.
    hi, dm, reset = _ui_sgr("hi"), _ui_sgr("dm"), theme.RESET
    if readline_active:
        return f"\001{hi}\002▛▜\001{reset}\002 \001{dm}\002❯\001{reset}\002 "
    return f"{hi}▛▜{reset} {dm}❯{reset} "


def _handle_theme(rest: list[str]) -> None:
    """``/theme`` -- cycle auto->dark->light; ``/theme <mode>`` sets it directly.

    Prints a one-line confirmation rendered in the *new* theme's colors. The
    prompt is redrawn by the REPL loop, so the switch is visible immediately.
    """
    if rest:
        arg = rest[0].lstrip("/").lower()
        if arg not in ("auto", "dark", "light"):
            console.print(
                Text("  usage: ", style="dim").append(
                    "/theme [auto|dark|light]",
                    style=ACCENT if color_enabled() else "bold",
                )
            )
            return
        theme.set_mode(arg)
    else:
        theme.set_mode(theme.next_mode(theme.current_mode()))

    mode = theme.current_mode()
    resolved = "dark" if theme.is_dark() else "light"
    color = color_enabled()
    msg = Text("  theme ", style=theme.dim() if color else "dim")
    msg.append(mode, style=f"bold {theme.accent()}" if color else "bold")
    if mode == "auto":
        msg.append("  →  ", style=theme.dim() if color else "dim")
        msg.append(resolved, style=theme.text() if color else "")
        msg.append(" (detected)", style=theme.dim() if color else "dim")
    else:
        msg.append("  →  ", style=theme.dim() if color else "dim")
        msg.append(resolved, style=theme.text() if color else "")
    console.print(msg)


def _handle(line: str) -> bool:
    """Execute one shell line. Returns False when the shell should exit."""
    stripped = line.lstrip("/").strip()
    if stripped.lower() in _EXIT_WORDS:
        return False
    try:
        argv = shlex.split(stripped)
    except ValueError as exc:
        console.print(f"[red]parse error:[/red] {exc}")
        return True
    if not argv:
        return True
    head, rest = argv[0].lower(), argv[1:]
    if head == "poros" and rest:
        # Humans type "poros report ..." inside the poros shell -- just work.
        head, rest = rest[0].lstrip("/").lower(), rest[1:]

    if head == "help":
        if rest and rest[0].lstrip("/") in DISPATCHED:
            _dispatch([rest[0].lstrip("/"), "--help"])
        else:
            _print_help()
    elif head == "version":
        from poros import __version__

        console.print(f"poros [bold]{__version__}[/bold]")
    elif head == "clear":
        console.clear()
    elif head == "theme":
        _handle_theme(rest)
    elif _words([head, *rest]):
        pass  # a sentence: `train qwen 7b on notes.jsonl`, `will 72b fit?`
    elif head in DISPATCHED:
        _dispatch([head, *rest])
    else:
        import difflib

        msg = Text("  unknown command ", style="dim")
        msg.append(head, style="bold")
        near = difflib.get_close_matches(head, list(COMMANDS) + ["exit"], n=1)
        if near:
            msg.append("  — did you mean ", style="dim")
            msg.append(f"/{near[0]}", style=ACCENT if color_enabled() else "bold")
            msg.append("?", style="dim")
        else:
            msg.append("  — try /help", style="dim")
        console.print(msg)
    return True


def run_shell(animate: bool = True) -> None:
    """Entry point for the interactive shell (TTY-gated by the caller)."""
    from poros import __version__

    # The interactive session (fx's model: scrollback + a pinned footer).
    from poros.cli import session

    if session.available():
        session.run(animate=animate)
        return

    # Detect the terminal background once, now -- before the banner or the REPL
    # reads any input -- so auto-theme is right for the banner + menu.
    theme.prime_detection()
    print_banner(console, __version__, animate=animate)

    # Off the session (TERM=dumb, POROS_NO_MENU): readline (history + tab completion), or input().
    readline_mod = _setup_readline()
    prompt = _prompt(readline_active=readline_mod is not None)
    try:
        while True:
            try:
                line = input(prompt)
            except EOFError:
                console.print()
                break
            except KeyboardInterrupt:
                console.print("[dim]  (/quit or Ctrl-D to exit)[/dim]")
                continue
            line = line.strip()
            if not line:
                continue
            if not _handle(line):
                break
    finally:
        _save_history(readline_mod)
    console.print("[dim]bye.[/dim]")
