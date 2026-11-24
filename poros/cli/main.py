"""Poros CLI entry point: ``cli()``, the console script, and the usage-error screen.

The Typer app and what every command shares are ``poros.cli.root``; an exception's refusal is
``poros.cli.errors``, a command's own refusal ``poros.cli.contract.refuse``. The commands live
in the modules of ``poros.cli.commands`` (importing the package registers them on ``root.app``
in the order ``--help`` lists them): ``train`` (init, train, plan), ``jobs``, ``export`` (eval,
export), ``report`` (report, leaderboard, bench run/matrix/compare/report) and ``check``
(check, agents, schema, and the environment checks `poros machine` runs). The verbs of the
interactive shell are ``poros.cli.verbs``; what may start a run is ``poros.preflight``.

Multi-GPU (`launch`/distributed) is deferred to a future release -- single-GPU only.

Critical: poros --help, poros --version, poros check, and poros report
MUST work with NO torch/GPU installed. All torch-dependent imports are
lazy (inside the command functions that need them).
"""

from __future__ import annotations

import os
import sys

import typer

import poros.cli.commands  # noqa: F401 (registers every command on root.app, in --help order)
from poros.cli import contract

# The commands, importable from this module too (poros.cli.main.train, ...).
from poros.cli.commands.check import check, schema  # noqa: F401
from poros.cli.commands.export import evaluate, export  # noqa: F401
from poros.cli.commands.report import (  # noqa: F401
    bench_compare,
    bench_matrix,
    bench_report,
    bench_run,
    leaderboard,
    report,
)
from poros.cli.commands.train import train  # noqa: F401
from poros.cli.root import _click_module, _ui, app
from poros.core.errors import EXIT_CODES

# ---------------------------------------------------------------------------
# Friendly usage errors (replaces click's raw "Missing argument" traces)
# ---------------------------------------------------------------------------

_USAGE_EXAMPLES: dict[str, tuple[str, ...]] = {
    "train": (
        "poros init                # writes a ready-to-run train.yaml",
        "poros train train.yaml",
    ),
    "eval": ("poros eval train.yaml     # the config you trained with",),
    "export": (
        "poros export ./my-adapter -o ./merged",
        "(./my-adapter = the output_dir of a finished training run)",
    ),
    "report": (
        "poros bench run bench.yaml -o ./bench-out",
        "poros report ./bench-out",
        "(training-run progress lives in <output_dir>/events.jsonl, not report)",
    ),
    "leaderboard": ("poros leaderboard ./bench-out",),
    "check": ("poros check Qwen/Qwen2.5-7B    # HF id, local model dir, or config.json",),
    "schema": ("poros schema config|events|artifact",),
    "bench run": ("poros bench run bench.yaml -o ./bench-out    # spec guide: docs/bench.md",),
    "bench matrix": ("poros bench matrix matrix.yaml -o ./bench-out",),
    "bench compare": ("poros bench compare ./bench-a ./bench-b",),
    "bench report": ("poros bench report ./bench-out --md",),
}


def _no_such_flag(exc, path: str) -> str:
    """`train has no --stepz; did you mean --steps?`: a flag a command does not read, with the
    nearest one it does (click lists every flag that shares a prefix; one is what is meant)."""
    import difflib

    name = getattr(exc, "option_name", "") or "that flag"
    near = difflib.get_close_matches(name, list(getattr(exc, "possibilities", None) or ()), n=1,
                                     cutoff=0.5)
    verb = path.removeprefix("poros").strip() or "poros"
    return f"{verb} has no {name}" + (f"; did you mean {near[0]}?" if near else "")


# Commands an earlier release had, and the command that answers their question now.
_REMOVED = {"fit": ("poros recipes", "the runs measured on the release, machine by machine"),
            "plan": ("poros train <config>", "checks the run before anything downloads")}


def _unknown_command(exc) -> "str | None":
    """The command name a click "No such command" error is about, else None."""
    import re

    found = re.match(r"No such command '([^']+)'", exc.format_message())
    return found.group(1) if found else None


def _render_usage_error(exc) -> None:
    """Render a click UsageError as guidance, not a scolding usage trace.

    Missing required arguments are the #1 first-contact stumble: answer with
    what the argument IS (its own help text) and a copy-paste example.
    """
    from poros.cli.ui import components as uc
    from poros.cli.ui.live import columns
    from poros.cli.ui.term import err as out  # an error: stderr in a one-shot command
    from poros.cli.ui.text import S

    click_mod = _click_module()
    ctx = getattr(exc, "ctx", None)
    path = ctx.command_path if ctx is not None else "poros"
    param = getattr(exc, "param", None)
    w = min(columns(), uc.CAP)
    examples = True
    if isinstance(exc, click_mod.exceptions.MissingParameter) and param is not None:
        name = (param.metavar or (param.name or "").upper()).strip()
        help_text = (getattr(param, "help", "") or "").rstrip(".")
        lines = uc.error_event(f"{path} needs {name}", help_text + "." if help_text else "", w)
    elif isinstance(exc, click_mod.exceptions.NoSuchOption):
        # a flag it does not read: the nearest real one, as the verb form says it, and no
        # example of how to begin (the command was begun)
        lines, examples = uc.error_event("Unknown flag", _no_such_flag(exc, path), w), False
    else:
        lines = uc.error_event("Usage", exc.format_message(), w)
        gone = _REMOVED.get(_unknown_command(exc) or "")
        if gone:  # a command an earlier release had: where its answer is now
            lines.append(uc.rail(S(gone[0], "tx"), S("  # " + gone[1], "dm")))
    key = path.removeprefix("poros").strip()
    for ex in _USAGE_EXAMPLES.get(key, ()) if examples else ():
        cmd, _, note = ex.partition("#")
        lines.append(uc.rail(S(cmd.rstrip(), "tx"),
                             S("  # " + note.strip(), "dm") if note else None))
    lines.append(uc.rail(S("full options: ", "dm"), S(f"{path} --help", "tx")))
    out(lines)


def _plain_help(command) -> None:
    """Typer's --help without what reads as internals: no ``<bool>`` beside a flag that takes
    no value, and a required argument named in capitals (``CONFIG``), not in braces. Only where
    this typer has those hooks: any other renders its own way."""
    import types

    params = list(getattr(command, "params", ()))
    try:  # --help itself is made on first use, and kept: make it now
        from typer.core import _click

        params.append(command.get_help_option(_click.Context(command)))
    except Exception:  # another typer: its own rendering stands
        pass
    for param in filter(None, params):
        if getattr(param, "is_flag", False) and getattr(param, "metavar", None) is None:
            param.metavar = ""  # typer shows an option's metavar in place of its type
        if param.param_type_name == "argument" and hasattr(param, "_usage_display_name"):
            name = (param.metavar or param.name or "").upper()
            said = name if param.required else f"[{name}]"
            param._usage_display_name = types.MethodType(lambda self, said=said: said, param)
    for sub in getattr(command, "commands", {}).values():
        _plain_help(sub)


def cli() -> None:
    """Console-script entry: the typer app with friendly usage errors.

    In JSON mode (``--json`` anywhere, or POROS_OUTPUT=json) stdout is
    isolated first, so usage errors and crashes also come out as one JSON
    error object (docs/cli.md).
    """
    from poros.core.env import safe_path

    os.environ["PATH"] = safe_path()  # before anything here or in a library spawns a tool by name
    click_mod = _click_module()
    from rich.markup import escape
    from typer.main import get_command

    from poros.core.env import quiet_hub_notice
    from poros.runs import make_home

    make_home()
    quiet_hub_notice()  # every command's anonymous questions to the Hub: its notice of them
    # Decided from argv up front: a usage error is raised before any
    # command's --json option is parsed, and must still be JSON. Help and
    # --version are documents, not results: they keep the real stdout.
    argv = sys.argv[1:]
    lead = 0
    while lead < len(argv) and argv[lead] == "--json":
        lead += 1
    if lead and lead < len(argv) and argv[lead] not in ("--help", "-h"):
        argv = argv[lead:] + ["--json"]  # `poros --json train run.yaml`: --json is per command
    elif lead:  # bare `poros --json`: JSON mode for the "no command" error
        os.environ["POROS_OUTPUT"] = "json"
        argv = argv[lead:]
    if argv[:1] == ["help"]:  # `poros help [verb]`
        argv = argv[1:2] + ["--help"]
    if argv[:1] == ["agents"]:  # its own commands end here, and its verbs go on as the words said
        from poros.cli.verbs import agents as agent_commands

        argv = agent_commands.enter(argv[1:])
    sys.argv[1:] = argv
    contract.invoked(argv)
    json_requested = "--json" in argv or contract.json_mode()
    if json_requested and not {"--help", "--version", "-V"} & set(argv):
        contract.isolate_stdout()

    # The verbs first (`poros models`, `poros train Qwen2.5-7B-Instruct --data
    # notes.jsonl`); config files, adapter directories and the advanced commands go to Typer.
    from poros.cli import app as words_app

    try:
        code = words_app.dispatch(argv)
    except KeyboardInterrupt:
        if json_requested:
            contract.emit_error("E_INTERRUPTED", "interrupted")
        raise SystemExit(EXIT_CODES["E_INTERRUPTED"])
    if code is not None:
        raise SystemExit(code)

    cmd = get_command(app)
    _plain_help(cmd)
    try:
        # Non-standalone click RETURNS a typer.Exit's code instead of
        # raising -- losing it exits 0 on every failure path, which breaks
        # the exit-code contract agents script against.
        rv = cmd.main(prog_name="poros", standalone_mode=False)
    except typer.Exit as e:
        raise SystemExit(e.exit_code)
    except (typer.Abort, KeyboardInterrupt):
        if json_requested:
            contract.emit_error("E_INTERRUPTED", "interrupted")
        else:
            from poros.cli import views

            _ui()[0].print(views.note("Aborted"))
        raise SystemExit(EXIT_CODES["E_INTERRUPTED"])
    except click_mod.exceptions.UsageError as exc:
        if json_requested:
            path = exc.ctx.command_path if getattr(exc, "ctx", None) is not None else "poros"
            unknown = isinstance(exc, click_mod.exceptions.NoSuchOption)
            gone = _REMOVED.get(_unknown_command(exc) or "")
            contract.emit_error("E_USAGE", escape(_no_such_flag(exc, path) if unknown
                                                  else exc.format_message()),
                                hint=f"{path} --help" if unknown else gone[0] if gone else "")
        else:
            _render_usage_error(exc)
        raise SystemExit(2)
    except click_mod.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code)
    except Exception as exc:  # as a verb's: its own code and sentence, a bug's report for a bug
        from poros.cli.errors import _friendly_error

        try:
            _friendly_error(exc)
        except typer.Exit as e:
            raise SystemExit(e.exit_code)
    else:
        code = rv if isinstance(rv, int) else 0
        if code == EXIT_CODES["E_INTERRUPTED"] and json_requested:
            # typer turns Ctrl-C into Exit(130), which click returns.
            contract.emit_error("E_INTERRUPTED", "interrupted")
        raise SystemExit(code)


# ---------------------------------------------------------------------------
# Display helpers for benchmark artifacts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Top-level callback (--version, interactive shell on a bare TTY `poros`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # `python -m poros.cli.main`
    cli()
    raise SystemExit(0)
