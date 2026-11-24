"""poros agents: Poros for a coding agent's shell.

The same verbs and JSON as `poros <verb>`, in one table of commands. A shell command runs with the
agent's own permissions and the host's approval of it, so `--yes` and `--download` work here as they
do at a terminal, and nothing is fenced. What holds against hostile content holds for anyone, and
this adds the one switch for outside words: see poros.core.env.
"""

from __future__ import annotations

import difflib
import importlib
import json
import os
import shlex
import signal
import sys
from typing import Callable, Optional, Sequence, Union

from poros import __version__, runs
from poros.cli import contract, questions
from poros.cli.schemas import CheckpointsOutput, Output, VersionOutput
from poros.cli.ui.text import Line, S
from poros.cli.verbs import checkpoints
from poros.core.errors import EXIT_CODES
from poros.core.refusal import Refusal

_SHORT = {"-y": "--yes", "-o": "--output"}
# Flags whose value is anyone's words: what follows is the value, whatever it says (`--name -h`).
_TEXT = ("--name", "--data", "--output", "-o", "--eval", "--recipe")


def _attached(argv: Sequence[str]) -> list[str]:
    """argv with each such value attached to its flag, `--name=-h`: then nothing here or in a
    verb reads it as -h, --json or -f."""
    out, items = [], iter(argv)
    for item in items:
        value = next(items, None) if item in _TEXT else None
        out.append(item if value is None else f"{_SHORT.get(item, item)}={value}")
    return out


def _read(rest: Sequence[str], values: Sequence[str] = (), switches: Sequence[str] = (),
          most: Optional[int] = None) -> tuple[list[str], dict]:
    """A command's words and flags. A flag it does not read is refused, as a verb's is, and so is
    one said twice: which one it meant is not for this to guess."""
    words, said = [], {}
    items = iter(rest)
    for item in items:
        name, equals, value = item.partition("=")
        name = _SHORT.get(name, name)
        if name in said and (name in values or name in switches):
            raise Refusal("E_USAGE", "Said twice", f"{name} is said twice: say it once")
        if name in values:
            said[name] = value if equals else next(items, "")
            if not said[name]:
                raise Refusal("E_USAGE", "No value", f"{name} needs a value")
        elif name in switches and not equals:
            said[name] = True
        elif item.startswith("-"):
            known = " ".join([*values, *switches]) or "none"
            raise Refusal("E_USAGE", "Unknown flag", f"no {item} here; its flags: {known}")
        else:
            words.append(item)
    if most is not None and len(words) > most:
        raise Refusal("E_USAGE", "Too many words", f"{' '.join(words)}: it takes at most {most}")
    return words, said


def _limit(said: dict, default: int = 20) -> int:
    raw = said.get("--limit", default)
    if not str(raw).isdecimal():
        raise Refusal("E_USAGE", "Not a number", f"--limit {raw} is not 0 or more; 0 shows all")
    return int(raw)


def _show(result: Output, view: Callable[[Output, int], list[Line]]) -> int:
    if contract.json_mode():
        contract.emit(result)
    else:
        from poros.cli.ui import term

        term.get().show(lambda w: view(result, w))
    return 0


def _version(rest: Sequence[str]) -> int:
    _read(rest, most=0)
    return _show(VersionOutput(version=__version__), lambda r, w: [[S(f"poros {r.version}", "tx")]])


def _model_list(rest: Sequence[str]) -> int:
    words, _ = _read(rest, most=1)
    from poros.cli.context import Context
    from poros.cli.verbs import models

    out = models.run(models.ModelsAction(), Context())
    if words:
        out.models = [m for m in out.models if words[0].lower() in m.id.lower()]

    def view(r: models.ModelsOutput, w: int) -> list[Line]:
        rows = [[m.id, m.size.upper(), m.entry, ", ".join(m.certified_here),
                 "yes" if m.cached else ""] for m in r.models]
        heads = ("Model", "Size", "Kind", "Certified here", "Downloaded")
        return [[S(checkpoints.counted(len(rows), "model"), "dm")],
                *(checkpoints.table(heads, rows, w, right=(1,)) if rows else [])]

    return _show(out, view)


def _run_list(rest: Sequence[str]) -> int:
    _, said = _read(rest, values=("--limit",), most=0)
    from poros.cli.verbs import runs_report
    from poros.cli.verbs.runs import RunsOutput

    every, limit = runs.all_runs(), _limit(said)
    shown = every[:limit] if limit else every

    def view(r: RunsOutput, w: int) -> list[Line]:
        rows = [[x["id"], x["state"], x["model"], "" if x["step"] is None else f"{x['step']:,}",
                 "" if x["loss"] is None else f"{x['loss']:.4f}",
                 checkpoints.age_text(x["created_at"])] for x in r.runs]
        heads = ("Run", "State", "Model", "Step", "Loss", "Created")
        return [[S(checkpoints.counted(len(rows), "run", r.more), "dm")],
                *(checkpoints.table(heads, rows, w, right=(3, 4, 5)) if rows else [])]

    return _show(RunsOutput(do="list", runs=[runs_report.row(r) for r in shown],
                            more=len(every) - len(shown)), view)


def _checkpoint_list(rest: Sequence[str]) -> int:
    _, said = _read(rest, values=("--run-id", "--limit"), most=0)
    return _show(checkpoints.listing(said.get("--run-id"), _limit(said)), checkpoints.view)


def _checkpoint_info(rest: Sequence[str]) -> int:
    words, _ = _read(rest, most=1)
    if not words:
        raise Refusal("E_USAGE", "Which checkpoint", "it takes a checkpoint path",
                      fix="poros agents checkpoint list", note="lists them")
    return _show(checkpoints.info(words[0]), checkpoints.view)


def _checkpoint_download(rest: Sequence[str]) -> int:
    words, said = _read(rest, values=("--output",), switches=("--force",), most=1)
    if not words:
        raise Refusal("E_USAGE", "Which checkpoint", "it takes a checkpoint path",
                      fix="poros agents checkpoint list", note="lists them")
    return _show(checkpoints.download(words[0], said.get("--output", "."), "--force" in said),
                 checkpoints.view)


def _checkpoint_delete(rest: Sequence[str]) -> int:
    paths, said = _read(rest, values=("--run-id", "--before", "--after"), switches=("--yes",))
    if not paths and not said.keys() - {"--yes"}:
        raise Refusal("E_USAGE", "Which checkpoints", "name paths, --run-id, --before or --after",
                      fix="poros agents checkpoint list", note="lists them")
    rows = checkpoints.select(paths, said.get("--run-id"), said.get("--before"),
                              said.get("--after"))
    warned = checkpoints.warnings(rows)
    if rows and "--yes" not in said:
        names = ", ".join(x.path for x in rows[:20]) + (
            f" and {len(rows) - 20} more" if len(rows) > 20 else "")
        # the ones it named, by path: what a run writes or prunes before the yes is not among them
        command = shlex.join(["poros", "agents", "checkpoint", "delete", *(x.path for x in rows),
                              "--yes"])
        questions.needs_input("checkpoint delete", [questions.confirmation(
            "delete", " ".join([f"Delete for good: {names}.",
                                *(f"{w[0].upper()}{w[1:]}." for w in warned)]),
            "--yes", "Delete them", command=command)])
    checkpoints.delete(rows)
    return _show(CheckpointsOutput(do="delete", checkpoints=rows, warnings=warned,
                                   total_bytes=sum(x.size_bytes for x in rows)), checkpoints.view)


def _checkpoint_storage(rest: Sequence[str]) -> int:
    _read(rest, most=0)
    return _show(checkpoints.storage(), checkpoints.view)


# Each result this table names, as `schema` prints it: the verbs' own, past poros.cli.schemas's.


def _schema(rest: Sequence[str]) -> int:
    """A result's or a run's file's JSON Schema, a document as it is for a person. With no name, one
    object naming them all: without a terminal stdout stays one object."""
    words, _ = _read(rest, most=1)
    from poros.cli.main import schema
    from poros.cli.schemas import result_schema, schema_names

    names = sorted(schema_names())
    if not words:
        print(json.dumps({"$id": "poros.schemas", "names": names}), file=contract.stdout())
    elif words[0] not in names:
        near = difflib.get_close_matches(words[0], names, n=1)
        raise Refusal("E_USAGE", "Unknown schema", f"{words[0]} is not a schema's name",
                      fix=f"poros agents schema {near[0]}" if near else "poros agents schema",
                      note="the closest" if near else "names them all")
    elif (doc := result_schema(words[0])) is not None:
        print(json.dumps(doc), file=contract.stdout())
    else:
        schema(words[0], False)
    return 0


# Each command: its usage, what it does, and how: this file's function, or the verb it is.
COMMANDS: dict[str, tuple[str, str, Union[str, Callable[[Sequence[str]], int]]]] = {
    "version": ("", "Poros's version", _version),
    "machine": ("", "whether this machine is ready to train: GPU, driver, RAM, disk, stack",
                "machine"),
    "model list": ("[SUBSTRING]", "the catalog, with the recipes certified for each here",
                   _model_list),
    "data": ("<PATH> [--convert] [--fix [-o FILE]]",
             "what a file trains; --convert writes csv as chats, --fix a clean copy", "data"),
    "train": ("<MODEL> --data <DATA> [--wait]", "fine-tune: a run id at once, or --wait", "train"),
    "run list": ("[--limit N]", "the runs, newest first (default 20, 0 for all)", _run_list),
    "run info": ("<RUN_ID>", "one run", "runs"),
    "run watch": ("<RUN_ID> [--wait N] [--frame]",
                  "its live state, once N steps further; --frame adds its loss curve", "runs"),
    "run stop": ("<RUN_ID>", "stop it after the step it is on; returns once stopped",
                 "runs stop"),
    "run continue": ("<RUN_ID> [--more N] [--data FILE] [--wait]", "go on from there, bit for bit",
                     "runs continue"),
    "run report": ("<RUN_ID>", "write its report", "runs report"),
    "checkpoint list": ("[--run-id ID] [--limit N]",
                        "newest first, across runs (default 20, 0 for all)", _checkpoint_list),
    "checkpoint info": ("<PATH>", "one checkpoint", _checkpoint_info),
    "checkpoint download": ("<PATH> [-o DIR] [--force]", "copy it to <DIR>/<run>_weights_<step>/",
                            _checkpoint_download),
    "checkpoint delete": ("[PATHS...] [--run-id ID] [--before DATE] [--after DATE] [-y]",
                          "permanent", _checkpoint_delete),
    "checkpoint storage": ("", "the size of each run's checkpoints, and of all",
                           _checkpoint_storage),
    "export": ("<RUN>|<PATH> [--merged] [--output DIR]",
               "an adapter, or with --merged a whole model; a checkpoint's path too", "export"),
    "eval": ("[<RUN>|<PATH>] [--data FILE] [--rows N]",
             "score a run beside its base: loss, rows improved, whether its answers end", "eval"),
    "recipes": ("[show|save|share|copy] ...", "whole setups, and a run as one", "recipes"),
    "arch": ("[try] <MODEL>", "what any model would take, from its config", "arch"),
    "prove": ("[<MODEL>|<RECEIPT>] [--unvalidated]",
              "streamed against resident, bit for bit; a new family's GPU gates", "prove"),
    "docs": ("[<TOPIC> [<WORDS>]]", "Poros's docs from this install, offline; words: one section",
             "docs"),
    "schema": ("[NAME]", "a result's JSON Schema, or events.jsonl's; alone, the names", _schema),
}
# What a verb's page says that means nothing under this command: its flags, and whole lines.
_NOT_HERE = {"train": ("--detach", "In a terminal"),
             "runs": ("--follow", "--live", "--detach", "--limit", "--copy", "--delete")}
# A flag this command says its own way: here --format is the output's (table or json), and a
# merged export is --merged. The verb's line, and this command's for it.
_SAID_HERE = {"export": (
    "    --format NAME     merged (a standalone model) or adapter (the LoRA weights)",
    "    --merged          a standalone model; without it, the adapter (the LoRA weights)")}


def _page(verb: str, doc: str) -> str:
    """A verb's page in this command's words, without what means nothing here: a flag goes with the
    lines under it, and nothing is left for a stripped flag to leave behind."""
    kept, going = [], False
    theirs, ours = _SAID_HERE.get(verb, ("\0", ""))
    for line in doc.replace(theirs, ours).splitlines():
        said = line.strip()
        if said.startswith(_NOT_HERE.get(verb, ("\0",))):
            going = said.startswith("--")
        elif not going or said.startswith("--") or not line.startswith("     "):
            going = False
            kept.append(line)
    return contract.as_agent_page("\n".join(kept).replace("with --live: ", "")).strip()


def _help(words: Sequence[str]) -> str:
    """The commands these words begin, and a verb's own page when they name just one."""
    hits: list[str] = []
    for said in (" ".join(words[:2]), " ".join(words[:1])):
        hits = hits or [k for k in COMMANDS if k == said or k.startswith(f"{said} ")]
    hits = hits or list(COMMANDS)
    lines = [f"poros agents {k} {COMMANDS[k][0]}".rstrip() + f"\n    {COMMANDS[k][1]}"
             for k in hits]
    lines.append("\nOn every command: -f, --format [table|json]  (json without a terminal) and "
                 "-h, --help.")
    how = COMMANDS[hits[0]][2] if len(hits) == 1 else None
    if isinstance(how, str):
        from poros.cli.app import BUILT

        verb = how.split()[0]
        lines.append("\n" + _page(verb, importlib.import_module(BUILT[verb]).__doc__ or ""))
    return "\n".join(lines)


def _format(argv: Sequence[str]) -> tuple[Optional[str], list[str]]:
    """The format asked for, and argv without it. `--format` is export's flag too: only table and
    json are this one. Anything else stays, for the command to refuse or read."""
    fmt, kept = None, []
    items = iter(argv)
    for item in items:
        name, equals, value = item.partition("=")
        if item == "--json":
            fmt = "json"
        elif name in ("-f", "--format"):
            value = value if equals else next(items, "")
            if value in ("table", "json"):
                fmt = value
            else:
                kept += ["--format", *([value] if value else [])]
        else:
            kept.append(item)
    return fmt, kept


# Another word for a command's own: `run status` is `run info`.
_SAME = {"run": {"status": "info", "show": "info", "get": "info", "start": "continue",
                 "resume": "continue", "cancel": "stop", "kill": "stop", "ls": "list"},
         "checkpoint": {"get": "download", "copy": "download", "pull": "download",
                        "rm": "delete", "remove": "delete", "ls": "list", "show": "info",
                        "status": "info"},
         "model": {"ls": "list"}}
# Commands an earlier release had, and the command that answers their question now.
_REMOVED = {"fit": ("poros agents recipes",
                    "the runs measured on the release, machine by machine")}
# What changes something: never offered as the closest spelling of another word.
_CHANGES = {"train", "run stop", "run continue", "run report", "checkpoint download",
            "checkpoint delete", "export", "recipes", "prove"}


def _command(argv: Sequence[str]) -> tuple[str, list[str]]:
    """The command these words name, and what follows it. A miss offers the command another word
    names, or the closest that changes nothing, or the list."""
    for size in (2, 1):
        name = " ".join(argv[:size])
        if name in COMMANDS:
            return name, list(argv[size:])
    said = " ".join(a for a in argv[:2] if not a.startswith("-"))
    group, _, word = said.partition(" ")
    same = _SAME.get(group, {}).get(word)
    if same:  # what it said, in this command's words, with the rest as it was
        raise Refusal("E_USAGE", "No such command", f"no poros agents {said}",
                      fix=shlex.join(["poros", "agents", group, same, *argv[2:]]),
                      note=f"{word} is {same} here")
    if group in _REMOVED:  # a command an earlier release had: what answers it now
        raise Refusal("E_USAGE", "No such command", f"no poros agents {said}",
                      fix=_REMOVED[group][0], note=_REMOVED[group][1])
    near = difflib.get_close_matches(said, [k for k in COMMANDS if k not in _CHANGES], n=1,
                                     cutoff=0.4)
    raise Refusal("E_USAGE", "No such command", f"no poros agents {said}",
                  fix=f"poros agents {near[0]}" if near else "poros agents --help",
                  note="the closest" if near else "every command")


_ONE_SUBJECT = {"run info", "run watch", "run stop", "run continue", "run report", "export",
                "eval", "data"}


def _once(name: str, verb: str, rest: Sequence[str]) -> None:
    """A verb's flag said twice, or a run said twice, is refused: a verb keeps one of them, and
    which is not for anyone to guess."""
    from poros.cli.words.grammar import BOOL_FLAGS, _is_flag, parse

    flags, i = [], 0
    while i < len(rest):
        word = rest[i]
        if _is_flag(word):
            flags.append(_SHORT.get(word.partition("=")[0], word.partition("=")[0]))
            if "=" not in word and word not in BOOL_FLAGS and i + 1 < len(rest) and not (
                    _is_flag(rest[i + 1])):
                i += 1  # its value
        i += 1
    twice = next((f for f in flags if flags.count(f) > 1), None)
    if twice:
        raise Refusal("E_USAGE", "Said twice", f"{twice} is said twice: say it once")
    subject = parse([*verb.split(), *rest]).subject
    if name in _ONE_SUBJECT and len(subject) > 1:
        quote = ['a path with spaces goes in quotes: "my file.jsonl"'] if name == "data" else []
        raise Refusal("E_USAGE", "Too many words",
                      f"{' '.join(subject)}: it takes one {'file' if name == 'data' else 'run'}",
                      context=quote)


def _verb(name: str, verb: str, rest: list[str], json_out: bool) -> list[str]:
    """argv for `poros`: this command as its verb, in the form an agent's shell wants."""
    from poros.cli.ui import simulate

    _once(name, verb, rest)
    if name == "run info" and not any(not r.startswith("-") for r in rest):
        raise Refusal("E_USAGE", "Which run", "it takes a run id", fix="poros runs",
                      note="lists them")
    if name == "run watch":
        rest = [*rest, "--live"]
    if name == "export" and "--format" not in rest:
        rest = [r for r in rest if r != "--merged"] + [
            "--format", "merged" if "--merged" in rest else "adapter"]
    if name in ("train", "run continue"):
        if "--wait" in rest:
            rest = [r for r in rest if r != "--wait"]
            contract.keep_waiting()
        elif not simulate.enabled():  # a simulated run is foreground only
            rest = [*rest, "--detach"]
    return [*verb.split(), *rest, *(["--json"] if json_out else [])]


def _interrupted(*_) -> None:
    raise KeyboardInterrupt


def enter(argv: Sequence[str]) -> list[str]:
    """Sets the switch, and the format: json without a terminal, as an agent's shell has none.
    Hands back the argv for a verb. Its own commands, and help, end here."""
    from poros.cli.app import guarded

    os.environ["POROS_AGENT"] = "1"
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # no warnings about a token it doesn't need
    # As ctrl-c: a run stops at once, saves nothing and exits E_INTERRUPTED (`poros runs stop`
    # is the one that saves a checkpoint first).
    signal.signal(signal.SIGTERM, _interrupted)
    fmt, argv = _format(_attached(argv))
    if "-h" in argv or "--help" in argv:
        print(_help([a for a in argv if not a.startswith("-")][:2]))
        raise SystemExit(0)
    if not argv:
        return ["agents"]  # the guide
    if fmt == "table":
        os.environ.pop("POROS_OUTPUT", None)
    json_out = fmt == "json" or (fmt is None and (contract.json_mode() or not sys.stdout.isatty()))
    if json_out:
        contract.force_json()
        contract.isolate_stdout()
    handed: list[str] = []

    def work() -> int:
        name, rest = _command(argv)
        how = COMMANDS[name][2]
        if isinstance(how, str):
            handed.extend(_verb(name, how, rest, json_out))
            return 0
        return how(rest)

    try:
        code = guarded(work)
    except KeyboardInterrupt:
        if json_out:
            contract.emit_error("E_INTERRUPTED", "interrupted")
        code = EXIT_CODES["E_INTERRUPTED"]
    if handed:
        return handed
    raise SystemExit(code)
