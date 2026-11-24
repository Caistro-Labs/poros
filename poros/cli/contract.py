"""The CLI's machine contract: one JSON object on stdout, stable exit codes.

``--json`` (or ``POROS_OUTPUT=json``) turns it on. Stdout then carries exactly
one JSON object per result, written by :func:`emit`. Everything else the
process prints (Rich, training progress, library chatter, C extensions) is
moved to stderr at the file-descriptor level by :func:`isolate_stdout`, so no
code path has to remember which stream it is on. Exit codes and error codes
come from ``poros.core.errors.EXIT_CODES``. Reference: docs/cli.md.

Torch-free: imported by every command.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from typing import TYPE_CHECKING, Any, NoReturn, Optional

import typer
from rich.errors import MarkupError
from rich.text import Text

from poros.cli import registry
from poros.cli.schemas import ErrorBody, ErrorOutput, Output
from poros.core.env import under_agent
from poros.core.errors import EXIT_CODES, no_keys

if TYPE_CHECKING:
    from poros.core.refusal import Refusal

_forced = False  # --json seen for this invocation
_stdout = None  # the real stdout, once isolate_stdout() has redirected fd 1
_argv: Optional[list[str]] = None  # this process's command line, once cli() has read it


def invoked(argv: list[str]) -> None:
    """The command this process runs (``poros`` + ``argv``): what a refusal offers again, whole,
    with the flag that confirms it."""
    global _argv
    _argv = list(argv)


def force_json() -> None:
    """Turn JSON mode on for this invocation (a command's --json option)."""
    global _forced
    _forced = True


_emitted = False  # a result or error object went to stdout this invocation


def reset() -> None:
    """Start an invocation in human mode (the interactive shell runs many)."""
    global _forced, _emitted
    _forced = _emitted = False


def emitted() -> bool:
    return _emitted


def json_mode() -> bool:
    return _forced or os.environ.get("POROS_OUTPUT", "").lower() == "json"


def isolate_stdout() -> None:
    """Keep the real stdout for emit() and point fd 1 at stderr."""
    global _stdout
    if _stdout is None:
        sys.stdout.flush()
        _stdout = os.fdopen(os.dup(1), "w", encoding="utf-8")
        os.dup2(2, 1)


def stdout():
    """The real stdout: for emit(), and for documents (`poros schema`)."""
    return _stdout or sys.stdout


_VERBS = set(registry.names(agent=True))  # `poros agents <verb>` runs them (poros.cli.registry)
_GONE = {"--json", "--detach", "--live"}  # what `poros agents` says for itself, or means nothing
_WITH_VALUE = {"--data", "--output", "-o", "--name", "--eval", "--recipe", "--from"}
_IN_TICKS = re.compile(r"`(poros [^`\n]*)`")
_WITH_COMMANDS = {"command", "next", "hint", "train", "without_data", "docs"}  # whole commands
_IN_PROSE = {"message", "remediation"}  # sentences: only a command whole in backticks
_RUN_FILES = {"job_dir", "config", "log", "folder"}  # `run watch` reads a run: an agent needs none
_PATHS = {"output", "output_dir", "path", "converted", "wrote", "adapter", "events", "report",
          "load", "same_as", "best"}
# top-level fields `poros agents` shows only when set
_WHEN_SET = {"learned", "plan", "command", "from", "queued_behind", "starts_next", "lines",
             "converted", "converted_rows", "wrote", "kept", "dropped", "support", "model_type",
             "fidelity", "issues_more", "phase",
             # a run's held-out counts and its best checkpoint: chats have ends, pairs prefers
             "eval_ends", "eval_answer_rows", "eval_prefers_start", "eval_prefers", "best"}
# the same path as `output`, or a run's own: an eval's checkpoint names its weights
_LEFT_OUT = {"poros.export.v1": {"adapter"}, "poros.eval.v1": {"adapter"}}
# poros.checkpoints.v1: each of these only under the `do` it means something for
_CHECKPOINT_FIELDS = (("runs", ("storage",)), ("total_bytes", ("storage", "delete")),
                      ("output", ("download",)), ("more", ("list",)), ("warnings", ("delete",)))
_waiting = False  # `--wait` was said: this result's commands keep it


def keep_waiting() -> None:
    """This `poros agents` call holds its run open: a command it offers does the same."""
    global _waiting
    _waiting = True


def _words(text: str) -> Optional[list[tuple[str, str]]]:
    """Each word of a command line, as the shell reads it and as it is written there. None: no
    shell reads it as one."""
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split, lexer.commenters = True, ""
    out, start = [], 0
    try:
        while (word := lexer.get_token()) is not None:
            end = lexer.instream.tell()
            out.append((word, text[start:end].strip()))
            start = end
    except ValueError:  # an unclosed quote
        return None
    return out


def _lead(said: list[str]) -> tuple[int, list[str]]:
    """How many leading words a person's command takes to say what `poros agents` says in these
    (a run's id or name, as it was written, follows)."""
    verb, rest = said[1] if len(said) > 1 else "", said[2:]
    first = rest[0] if rest else ""
    if verb == "agents":
        return 0, []
    if verb == "runs" and first in ("stop", "continue", "report"):
        return 3, ["poros", "agents", "run", first]
    if verb == "jobs" and first == "resume":
        return 3, ["poros", "agents", "run", "continue"]
    if verb == "runs" and first == "checkpoints" and len(rest) > 1:  # a copy or a delete: below
        return (0, []) if {"--copy", "--delete"} & set(rest) else (
            3, ["poros", "agents", "checkpoint", "list", "--run-id"])
    if verb == "runs" and first and not first.startswith("-"):
        watch = {"--live", "--wait", "--frame"} & set(rest)
        return 2, ["poros", "agents", "run", "watch" if watch else "info"]
    if verb == "runs":
        return 2, ["poros", "agents", "run", "list"]
    if verb == "models" and first != "get":
        return 2, ["poros", "agents", "model", "list"]
    if verb in _VERBS:
        return 2, ["poros", "agents", verb]
    return 0, []


def _one_checkpoint(said: list[str]) -> Optional[list[str]]:
    """`poros runs checkpoints <id> --copy|--delete <step>` as this command's download or delete:
    its yes is a delete's --yes and a copy's --force."""
    if said[1:3] != ["runs", "checkpoints"] or len(said) < 6 or said[4] not in ("--copy",
                                                                               "--delete"):
        return None
    yes = {"--yes", "-y"} & set(said[6:])
    path = f"poros://{said[3]}/weights/{said[5]}"
    if said[4] == "--delete":
        return ["poros", "agents", "checkpoint", "delete", path, *(["--yes"] if yes else [])]
    return ["poros", "agents", "checkpoint", "download", path, *(["--force"] if yes else [])]


def _as_agent_command(text: str) -> str:
    """One person's command as `poros agents` runs it: its leading words in this command's words,
    and its flags dropped or changed as whole words. Every other word is kept as it was written,
    quotes and all; a line no shell reads as one is kept whole."""
    words = _words(text)
    if not words or words[0][0] != "poros":
        return text
    said = [w for w, _ in words]
    one = _one_checkpoint(said)
    if one:
        return shlex.join(one)
    lead, form = _lead(said)
    if not form and said[1:2] != ["agents"]:
        return text
    kept, i = list(form), lead
    while i < len(words):
        word, raw = words[i]
        after = said[i + 1] if i + 1 < len(said) else None
        if said[i - 1] in _WITH_VALUE and i > lead:  # a value, whatever it says
            kept.append(raw)
        elif word in _GONE:
            pass
        elif word == "--format" and after in ("adapter", "merged"):  # `-f` is the output's here
            kept += ["--merged"] if after == "merged" else []
            i += 1
        else:
            kept.append(raw)
        i += 1
    foreground = kept[2:3] == ["train"] or kept[2:4] == ["run", "continue"]
    if _waiting and foreground and "--wait" not in said:
        kept.append("--wait")
    return " ".join(kept)


def as_agent(text: str) -> str:
    """A command as it runs here: under `poros agents` its form, else as it is. A sentence keeps
    its words, and a command whole in backticks in it is changed. A person's `poros machine` has
    no other form, and still runs."""
    if not under_agent():
        return text
    if text.startswith("poros "):
        return _as_agent_command(text)
    return _IN_TICKS.sub(lambda hit: f"`{_as_agent_command(hit[1])}`", text)


def as_agent_page(text: str) -> str:
    """A help page as `poros agents` gives it: a line that is a command, and a command in
    backticks, in its form."""
    lines = []
    for line in text.split("\n"):
        said = line.lstrip()
        lead = line[: len(line) - len(said)]
        lines.append(lead + as_agent(said) if said.startswith("poros ") else as_agent(line))
    return "\n".join(lines)


def _for_agent(value: Any, key: str = "") -> Any:
    """A result as `poros agents` gives it: commands in its words, no path into a run's own folder,
    and in any string nothing a reader cannot see and nothing shaped like a key."""
    from poros.core.errors import CONTROL

    if isinstance(value, dict):
        return {_for_agent(k): _for_agent(v, k) for k, v in value.items()
                if not (k in _RUN_FILES and isinstance(v, str))}
    if isinstance(value, list):
        return [_for_agent(item, key) for item in value]
    if not isinstance(value, str):
        return value
    seen = CONTROL.sub(lambda hit: hit[0] if hit[0] in "\n\t" else "", _whole(value))
    seen = seen.replace("\u2028", " ").replace("\u2029", " ")  # a line's end to some readers
    if key in _PATHS:  # a path is a path: a folder named like a key is still that folder
        return seen
    if key in _WITH_COMMANDS:
        return no_keys(as_agent(seen) if seen.startswith("poros ") else seen)
    return no_keys(as_agent(seen) if key in _IN_PROSE else seen)


def _as_agent_result(said: dict) -> dict:
    """A result's top level as `poros agents` gives it: no field that says nothing. An eval says
    only what its rows have, and is simulated or does not say."""
    out = {k: v for k, v in said.items() if not (
        (v is None and k in _WHEN_SET) or k in _LEFT_OUT.get(said.get("schema"), ()))}
    if said.get("schema") == "poros.eval.v1":
        out = {k: v for k, v in out.items() if v is not None and (k != "simulated" or v)}
        for side in ("base", "trained"):  # a chats run has no pairs numbers, and pairs no ends
            if isinstance(out.get(side), dict):
                out[side] = {k: v for k, v in out[side].items() if v is not None}
    if said.get("schema") == "poros.checkpoints.v1":  # each field where it means something
        do = said.get("do")
        for key, where in _CHECKPOINT_FIELDS:
            if do not in where:
                out.pop(key, None)
        for row in out.get("checkpoints", []) if do not in ("info", "download") else []:
            row.pop("files", None)
    return _for_agent(out)


def _nullable(prop: dict) -> bool:
    return {"type": "null"} in prop.get("anyOf", ())


def published(doc: dict) -> dict:
    """A result's JSON Schema as every result satisfies it, a person's ``--json`` and `poros
    agents`' alike: ``required`` names the fields both carry, and a field `poros agents` leaves
    out (``_as_agent_result``, ``_for_agent``) is optional. The two change together:
    tests/cli/test_agent_schemas.py holds each command's result to its schema."""
    name, props = doc.get("$id"), doc.get("properties", {})
    unset = {k for k, prop in props.items() if _nullable(prop)}  # None: said only when set
    gone = (_WHEN_SET & unset) | _LEFT_OUT.get(name, set())
    if name == "poros.eval.v1":
        gone |= unset | {"simulated"}
    if name == "poros.checkpoints.v1":
        gone |= {key for key, _ in _CHECKPOINT_FIELDS}
    for part in (doc, *doc.get("$defs", {}).values()):  # a run's own files, at any depth
        if "required" in part:
            part["required"] = [k for k in part["required"] if k not in _RUN_FILES
                                and not (part is doc and k in gone)]
    return doc


_FIGURE_KEYS = frozenset({"value", "unit", "source"})


def _no_figure_receipts(said: Any) -> Any:
    """``said`` with every plan figure's ``receipt`` left out (poros.core.schemas.Figure): where
    a figure was read from is the planner's, never a result's."""
    if isinstance(said, dict):
        figure = _FIGURE_KEYS <= said.keys()
        return {k: _no_figure_receipts(v) for k, v in said.items()
                if not (figure and k == "receipt")}
    if isinstance(said, list):
        return [_no_figure_receipts(v) for v in said]
    return said


def emit(result: Output) -> None:
    """Write one result object as a single JSON line on the real stdout."""
    global _emitted
    from poros.cli.ui import live

    live.flush_pending()  # a closed live area is cleared first, not left half drawn
    out = stdout()
    try:
        text = result.model_dump_json(by_alias=True)
    except ValueError:  # a lone surrogate: a file's name in bytes that are not UTF-8
        text = json.dumps(_whole(result.model_dump(mode="json", by_alias=True)),
                          ensure_ascii=False, separators=(",", ":"))
    if '"receipt":' in text:  # a plan's figures name theirs: left out
        text = json.dumps(_no_figure_receipts(json.loads(text)), ensure_ascii=False,
                          separators=(",", ":"))
    if under_agent():
        text = json.dumps(_as_agent_result(json.loads(text)), ensure_ascii=False,
                          separators=(",", ":"))
    try:
        out.write(text + "\n")
        out.flush()
    except BrokenPipeError:  # its reader went: nothing more to say, and no traceback
        os.dup2(os.open(os.devnull, os.O_WRONLY), out.fileno())
    _emitted = True


_SURROGATE = re.compile("[\ud800-\udfff]")


def _whole(value: Any) -> Any:
    """``value`` with every lone surrogate in its strings as U+FFFD: UTF-8 can write it."""
    if isinstance(value, str):
        return _SURROGATE.sub("\ufffd", value)
    if isinstance(value, dict):
        return {_whole(k): _whole(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_whole(v) for v in value]
    return value


def _plain(markup: str) -> str:
    try:
        return Text.from_markup(markup).plain
    except MarkupError:  # unescaped user text: never turn an error into a crash
        return markup


# The docs section an error names where its refusal names none, by code: said as `poros docs
# <page> <section words>` (poros.links.doc_ref), which any install opens offline. A code that
# has no one section (E_INTERNAL is a bug, E_NETWORK and E_HUB_AUTH the Hub's) has none here:
# its refusal names its own, or the error says none.
DOCS = {
    "E_USAGE": "docs/cli.md#names-and-flags",
    "E_NOT_FOUND": "docs/cli.md#names-and-flags",  # what names a model, a run, a checkpoint
    "E_CONFIG": "docs/cli.md#config-files",
    "E_CONFIRM_REQUIRED": "docs/cli.md#names-and-flags",
    "E_OOM": "docs/training.md#what-fits",
    "E_OOM_PREDICTED": "docs/training.md#what-fits",
    "E_HOST_OOM_PREDICTED": "docs/training.md#what-fits",
    "E_ARCH_UNSUPPORTED": "docs/models.md#any-other-model",
    "E_EXPORT_UNSUPPORTED": "docs/limitations.md",
    "E_GPU_UNSUPPORTED": "docs/install.md#requirements",
    "E_STEP_CHECK": "docs/training.md#checkpoints-and-resume",
    "E_NO_TORCH": "docs/install.md#install",
    "E_DEPS": "docs/install.md#install",
    "E_NO_CUDA": "docs/install.md#requirements",
    "E_STACK_UNSUPPORTED": "docs/install.md#requirements",
    "E_MACHINE_NOT_READY": "docs/install.md#check-your-machine",
    "E_DISK": "docs/training.md#refusals",
    "E_DATASET": "docs/training.md#data",
    "E_DATASET_NOT_FOUND": "docs/training.md#data",
    "E_JOB_FAILED": "docs/training.md#checkpoints-and-resume",
    "E_JOB_CANCELLED": "docs/training.md#checkpoints-and-resume",
}


def emit_error(
    code: str, message: str, *, hint: str = "", fix: Optional[dict] = None, docs: str = ""
) -> None:
    """Emit the error object. ``message``/``hint`` may carry Rich markup; ``docs`` a page of
    docs/, said as ``poros docs <page>`` (poros.links.doc_ref), by default its code's (``DOCS``).
    A hint, a remediation or a docs section there is none of is left out (ErrorBody)."""
    from poros.links import doc_ref

    docs = doc_ref(docs or DOCS.get(code, ""))
    emit(ErrorOutput(error=ErrorBody(
        code=code,
        exit_code=EXIT_CODES[code],
        message=no_keys(_plain(message)),
        hint=no_keys(_plain(hint)),
        remediation=fix or {},
        docs=docs,
    )))


def refuse(
    code: str, title: str, reason: str, *, context=(), fix=None, note: str = "",
    docs: Optional[str] = None,
) -> NoReturn:
    """The CLI's one refusal: ``✕ title  reason`` + rails, or the JSON error, then the exit
    ``code`` has (``EXIT_CODES``).

    Plain text in (no markup). ``fix``: the command that fixes it (with ``note`` beside it), or a
    ``poros.cli.ui.screens.Fix``; its command becomes the JSON ``hint``. A ``Refusal`` raised by
    a verb or returned by the preflight is shown with ``refused``.
    """
    if isinstance(fix, str):
        from poros.cli.ui.screens import Fix

        fix = Fix("", fix, note) if fix else None
    if json_mode():
        from rich.markup import escape

        detail = "; ".join([reason, *context])
        emit_error(code, escape(detail), hint=escape(fix.command) if fix else "", docs=docs or "")
    else:
        from poros.cli.ui import term
        from poros.cli.ui.screens import error_screen
        from poros.links import doc_ref

        t = term.get()
        term.err(error_screen(title, reason, t.width(), context=list(context), fix=fix,
                              docs=doc_ref(docs) if docs else docs))  # stderr, off the session
        from poros.cli.ui import components as uc

        if fix and fix.command and uc.in_session:  # the fix waits in the prompt: ⏎ runs it
            cmd = uc.command(fix.command)
            if cmd.startswith("/"):
                uc.next_step = cmd
    raise typer.Exit(code=EXIT_CODES[code])


def confirm(action: str, yes: bool, *, flag: str = "--yes") -> None:
    """Proceed past a risky ``action`` only when confirmed.

    ``flag`` confirms. Otherwise a person at a terminal is asked with the ❯
    menu (default no); anything else (a pipe, CI, an agent, JSON mode) is
    never prompted and exits 2 naming the flag.
    """
    if yes:
        return
    if json_mode() or under_agent() or not (sys.stdin.isatty() and sys.stdout.isatty()):
        # the command to copy: this one, whole, with the flag (the flag alone where this
        # process read no command of its own: a line in the shell, a caller in Python)
        again = flag if not _argv else shlex.join(
            ["poros", *(a for a in _argv if a != "--json"), flag])
        refuse("E_CONFIRM_REQUIRED", "Needs a yes",
               f"{_plain(action)} Re-run with {flag} to confirm.", fix=again)
    from poros.cli.ui import term
    from poros.cli.ui.live import interactive

    try:
        import termios  # noqa: F401
    except ImportError:  # Windows: no raw mode for the menu
        termios = None
    if termios is None or not interactive(sys.stdout):  # TERM=dumb: a plain prompt
        if typer.confirm(f"{_plain(action)} Continue?", default=False):
            return
        refuse("E_CONFIRM_REQUIRED", "Needs a yes", "Not confirmed; nothing was changed.",
               fix=flag)
    pick = term.get().choose(_plain(action), [("No, stop here", ""), ("Yes, continue", "")],
                             tag="confirm")
    if pick != 1:
        refuse("E_CONFIRM_REQUIRED", "Needs a yes", "Not confirmed; nothing was changed.",
               fix=flag)


def refused(r: Refusal) -> NoReturn:
    """``refuse`` for a ``poros.core.refusal.Refusal``."""
    refuse(r.code, r.title, r.reason, context=r.context, fix=r.fix, note=r.note, docs=r.docs)
