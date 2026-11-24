"""data: what a file holds, what it would train, and a converted or clean copy.

    poros data chats.jsonl
    poros data tickets.csv --convert
    poros data alpaca.json --convert
    poros data messy.jsonl --fix

Flags:
    --convert          write it as chats.jsonl (or pairs) next to the file, or in the folder
    --fix              write a clean copy next to it, <name>.clean.jsonl; never over a file
    -o, --output FILE  where --fix writes it instead (.jsonl, or .json)

Reads the file, never the Hub. Chats ({"messages": [...]}, ShareGPT's
conversations too) train SFT; prompt/chosen/rejected pairs train DPO. A csv
with question/answer-like columns, Alpaca rows (instruction, input, output)
and prompt/completion rows train SFT as they are, read as chats; --convert
writes that copy out. Plain text files convert into {"text"} rows.

`rows` counts the rows that train, and `lines` the file's lines; `problems`
says what the others are, and `issues` the same rows as data (row, kind, why).
`trains_as_is` says whether `train` takes the file as it is: no row stops a run.
A question/answer row among chats trains as a chat, as the trainer reads it,
and a chat among question/answer rows too; a row of any other shape stops a
run. A chat with no assistant turn has no answer to train on: the trainer
leaves it out, and it is said as `dropped`. A blank line is no row, and in a .jsonl row N is
line N. --fix keeps each row that trains, makes a row of another shape the
file's own when nothing it trains on is lost (a question/answer row in chats
becomes a chat), and drops the rest: a row a run fails at, an empty answer, a
pair whose chosen and rejected are the same.
The copy holds only what Poros trains on. The file itself is never touched.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import stat
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from poros import data as poros_data
from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import Choice, Output, Question
from poros.cli.ui.text import Line
from poros.cli.verbs import shared, train_data
from poros.cli.words.grammar import Sentence
from poros.core.env import under_agent
from poros.core.refusal import Refusal
from poros.data import ASKS, SAYS, _pick, as_chat
from poros.preflight.data import (
    _BLANK,
    _FIXES,
    _KNOWN,
    LONGEST,
    Shape,
    _Checked,
    _read,
    _records,
    _said,
    _shape,
    _Unread,
)

# What a row does in a run, from the checks `problems` says in words: ``fails`` a run stops at it,
# ``dropped`` the trainer leaves it out, ``converted`` it trains as a chat, ``empty_answer`` it
# trains the model to say nothing, ``identical`` a pair's chosen and rejected are the same, so it
# teaches nothing, ``cut`` its answer starts past the longest length train picks.
IssueKind = Literal["fails", "dropped", "converted", "empty_answer", "identical", "cut"]


class DataAction(Action):
    verb: ClassVar[str] = "data"
    path: str
    convert: bool = False
    fix: bool = False
    output: Optional[str] = None  # --fix: where the clean copy goes

    def canonical(self) -> list[str]:
        return (["data", self.path] + (["--convert"] if self.convert else [])
                + (["--fix"] if self.fix else []) + (["--output", self.output] if self.output
                                                      else []))


class Issue(BaseModel):
    """One row `problems` names, as data: its number, what it does in a run, and why."""

    row: int  # in a .jsonl, its line
    kind: IssueKind
    why: str  # what `problems` says of it


class DataOutput(Output):
    schema_: Literal["poros.data.v1"] = Field("poros.data.v1", alias="schema")
    path: str
    shape: Shape
    rows: int  # the rows that train; `problems` says what the rest are
    lines: Optional[int] = None  # the file's lines (a folder has none): a blank line is no row
    trains: Optional[str] = None  # sft / dpo / classic; None until converted
    trains_as_is: bool = False  # `train` takes this file as it is: rows train, and none stops it
    samples: list[str]  # the first rows, one line each
    problems: list[str] = []  # rows that cannot train, by number
    issues: list[Issue] = []  # the same rows as data, the first LISTED of them
    issues_more: Optional[int] = None  # how many rows past those `issues` lists
    converted: Optional[str] = None  # --convert: the file it wrote
    converted_rows: Optional[list[str]] = None  # --fix: rows made the file's shape, data's reason
    wrote: Optional[str] = None  # --fix: the clean copy
    kept: Optional[int] = None  # --fix: its rows, the converted ones too
    dropped: Optional[list[str]] = None  # --fix: the rows left out of it, each with data's reason
    command: Optional[str] = None  # the train command for it
    next: list[str] = []  # `poros agents`: the same, with nothing to fill in


def resolve(s: Sentence, ctx: Context) -> Draft[DataAction]:
    if not s.subject:
        return Draft(None, [_which(ctx)])
    path = Path(s.subject[0]).expanduser()
    path = path if path.is_absolute() else ctx.cwd / path
    if not path.exists():
        raise Refusal("E_DATASET_NOT_FOUND", "No such file",
                      f"{s.subject[0]} does not exist",
                      fix="poros data", note="the data files here, to pick from")
    if not (path.is_file() or path.is_dir()):  # a pipe would hang this, and a device fill memory
        raise Refusal("E_DATASET", "Not a file", f"{s.subject[0]} is not a file or a folder")
    convert, fix = bool(s.flags.get("--convert")), bool(s.flags.get("--fix"))
    output = s.flags.get("--output", s.flags.get("-o"))
    if output is True or output == "":
        raise Refusal("E_USAGE", "No value", "-o needs a file's name: -o clean.jsonl")
    if output and not fix:
        raise Refusal("E_USAGE", "-o goes with --fix", "-o names where --fix writes")
    if fix and convert:
        raise Refusal("E_USAGE", "One at a time", "--fix and --convert: say one",
                      context=["--fix writes question/answer rows as chats too"])
    if fix and (path.is_dir() or path.suffix not in _FIXES):
        raise Refusal("E_USAGE", "Nothing to fix", "--fix cleans a .jsonl, .json or .csv "
                      "file", context=["a text file trains as it is"])
    if output:
        out = Path(str(output)).expanduser()
        out = out if out.is_absolute() else ctx.cwd / out
        if out.suffix not in (".jsonl", ".json"):
            raise Refusal("E_USAGE", "Not a data file", "-o names a .jsonl or .json file")
        output = str(out)
    return Draft(DataAction(path=str(path), convert=convert, fix=fix, output=output or None))


def _which(ctx: Context) -> Question:
    """The data files here, to pick from; none here is a refusal that says so."""
    files = train_data.local_files(ctx.cwd)[:9]
    if not files:
        raise Refusal("E_USAGE", "No data here", f"no csv, json or jsonl in {ctx.cwd}",
                      context=["drag a file into the terminal, or give its path"])
    return Question(id="data", prompt="Check which file?", flag="subject", default=files[0],
                    choices=[Choice(value=f, label=f, note="here") for f in files])


def confirmations(action: DataAction, ctx: Context) -> list:
    return []


# ── reading ───────────────────────────────────────────────────────────


HOLDS = {"chats": "chats", "pairs": "preference pairs", "text": "text rows",
         "convertible": "question/answer rows"}
TRAINS = {"chats": "SFT", "convertible": "SFT", "text": "SFT", "pairs": "DPO"}


def shape_of(path: Path) -> Shape:
    """What a file holds, from its first readable row; "unknown" if none."""
    try:
        first = next((r for r in _records(path) if isinstance(r, dict)), None)
    except (OSError, ValueError, csv.Error, RecursionError):
        return "unknown"
    return _shape(first) if first else "unknown"


def about(path: Path) -> tuple[str, str]:
    """(what a file holds, what it trains), in a person's words:
    ("61 question/answer rows", "SFT"); ("", "") when Poros can't read it."""
    shape = shape_of(path)
    try:
        rows = train_data.scan(str(path), path.parent).rows
    except (OSError, ValueError):
        return "", ""
    if shape not in HOLDS:
        return "", ""
    noun = HOLDS[shape][:-1] if rows == 1 else HOLDS[shape]  # 1 chat, 2 chats
    return (f"{rows:,} {noun}" if rows else noun), TRAINS[shape]


_PLAIN_PATH = re.compile(r"[\w ./-]{1,512}", re.ASCII)


LISTED = 20  # rows --fix names as converted, or dropped; past it, how many more


# ── --fix: a clean copy ───────────────────────────────────────────────


def _new_file(target: Path, mode: int) -> int:
    """A new file, or FileExistsError: never over one, never through a link."""
    return os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)


def _clean_name(path: Path) -> Path:
    """Where --fix writes by default: beside the file, ``<stem>.clean.jsonl`` (a .json's is
    .json). A csv's rows are written as chats, which a csv cannot hold."""
    return path.with_name(f"{path.stem}.clean{'.json' if path.suffix == '.json' else '.jsonl'}")


def _at(path: Path) -> Optional[str]:
    """A path as a command says it: its name here, else whole; None when it is not fit to
    repeat."""
    at = path.name if path.parent == Path.cwd() else str(path)
    return at if _PLAIN_PATH.fullmatch(at) else None


def _taken(target: Path, path: Path) -> Refusal:
    """--fix's file is there already: the same command, with -o naming the next free name."""
    stem, ext = target.stem, target.suffix
    free = next((c for c in (target.with_name(f"{stem}-{i}{ext}") for i in range(2, 100))
                 if not os.path.lexists(c)), None)
    at, to = _at(path), _at(free) if free else None
    return Refusal("E_USAGE", "Already there", f"{target.name} already exists",
                   context=["--fix never writes over a file"],
                   fix=shlex.join(["poros", "data", at, "--fix", "-o", to]) if at and to
                   else "", note="writes it under the next free name")


def _whole_chat(row: dict) -> bool:
    """A question/answer row a chat holds all of: each field of it Poros reads is read, and is
    text. A field Poros reads in no row is not one."""
    ask, say = _pick(row, ASKS), _pick(row, SAYS)
    used = {ask, say} | ({"system"} & set(row))
    if str(ask).lower() == "instruction" and "input" in row:
        used.add("input")  # Alpaca's: it joins its instruction
    named = {k for k in row if str(k).lower() in (*ASKS, *SAYS, "system")}
    text = all(isinstance(row[k], str) or (k in ("system", "input") and row[k] is None)
               for k in used)
    return named <= used and text and as_chat(row) is not None


def _side(value, role: str, n: int):
    """A pair's prompt or answer: text as it is; messages as role and content alone."""
    if isinstance(value, str):
        return value
    return [{"role": t["role"], "content": t["content"]}
            for t in poros_data._as_turns(value, role, n)]


def _kept(row: dict, this: Shape, n: int) -> Optional[dict]:
    """A row as the clean copy holds it, built from what Poros trains on alone: no other field of
    the row's is copied. None: a text row with no text."""
    if this == "chats":
        return {"messages": [{"role": t["role"], "content": t["content"]}
                             for t in poros_data._turns(row, n) or []]}
    if this == "convertible":
        return as_chat(row)
    if this == "pairs":
        return {key: _side(row[key], "user" if key == "prompt" else "assistant", n)
                for key in ("prompt", "chosen", "rejected")}
    text = row.get("text")
    return {"text": text} if isinstance(text, str) and text else None


def _into(row: dict, this: Shape, main: Shape, n: int) -> Optional[dict]:
    """A row of another shape in the file's, when nothing it trains on is lost: a question and its
    answer as a chat, in chats; a chat that trains as it is, in question/answer rows, which are
    written as chats. Nothing else: a pair is more than a chat, and text has no turns."""
    if main not in ("chats", "convertible"):
        return None
    if this == "convertible" and _whole_chat(row):
        return as_chat(row)
    if this == "chats" and all(kind == "say" for kind, _ in _said(row, this, n, LONGEST)):
        return _kept(row, this, n)
    return None


def _cleaned(row, n: int, main: Shape) -> tuple[str, Optional[dict], str]:
    """What --fix does with a row: ("kept", it, ""), ("converted", it, why), or ("dropped", None,
    why). Each why is what `data` says of it."""
    if isinstance(row, _Unread):
        return "dropped", None, str(row)
    if not isinstance(row, dict):
        return "dropped", None, f"row {n} is not an object"
    this = _shape(row)
    if this != main:
        why = f"row {n} is {this}, not {main}"
        made = _into(row, this, main, n)
        return ("converted", made, why) if made else ("dropped", None, why)
    bad = next((said for kind, said in _said(row, this, n, LONGEST) if kind != "say"), None)
    if bad:
        return "dropped", None, bad
    made = _kept(row, this, n)
    return ("kept", made, "") if made else ("dropped", None, f"row {n} has no 'text' text")


def _dumped(row: dict) -> bytes:
    try:
        return json.dumps(row, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: escaped, it reads back the same
        return json.dumps(row).encode("ascii")


def _listed(rows: list[str]) -> list[str]:
    return rows[:LISTED] + ([f"and {len(rows) - LISTED} more"] if len(rows) > LISTED else [])


def _write_clean(path: Path, target: Path, main: Shape) -> tuple[int, list[str], list[str]]:
    """The clean copy, a new file; on any failure, none. Every row is read again, past the count's
    bound too: `train` reads them all. (kept, converted, dropped)."""
    try:
        fd = _new_file(target, stat.S_IMODE(os.stat(path).st_mode) & 0o666)
    except FileExistsError:
        raise _taken(target, path) from None
    except OSError as exc:
        import errno

        # the path whole, as -o said it: /fixed.jsonl is not fixed.jsonl here. A folder that
        # takes no file from this user, or has no room, is the disk's refusal; any other is the
        # path's (a folder that is not there)
        disk = exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSPC)
        raise Refusal("E_DISK" if disk else "E_USAGE", "Can't write it",
                      f"{_at(target) or target.name}: "
                      f"{exc.strerror or 'it cannot be written'}",
                      context=["-o names a file in a writable folder"]) from None
    kept, converted, dropped, as_json = 0, [], [], target.suffix == ".json"
    try:
        with os.fdopen(fd, "wb") as fh:
            for n, row in enumerate(_read(path), 1):
                if row is _BLANK:
                    continue
                what, made, why = _cleaned(row, n, main)
                if made is None:
                    dropped.append(why)
                    continue
                if what == "converted":
                    converted.append(why)
                if as_json:
                    fh.write(b",\n" if kept else b"[\n")
                fh.write(_dumped(made) + (b"" if as_json else b"\n"))
                kept += 1
            if as_json:
                fh.write(b"\n]\n" if kept else b"[]\n")
    except BaseException:
        os.unlink(target)  # what it wrote is not the copy
        raise
    if not kept:
        os.unlink(target)
        raise Refusal("E_DATASET", "No row trains", f"no row of {path.name} trains, so "
                      "nothing is written", context=dropped[:3])
    return kept, converted, dropped


def run(action: DataAction, ctx: Context) -> DataOutput:
    path = Path(action.path)
    target = Path(action.output) if action.output else _clean_name(path)
    if action.fix and os.path.lexists(target):  # said before a long file is read
        raise _taken(target, path)
    checked = _Checked(path)
    rows, problems, samples, shape = (checked.rows, checked.problems, checked.samples,
                                      checked.shape)
    trains = {"chats": "sft", "pairs": "dpo", "text": "classic"}.get(shape)
    if shape == "convertible" and path.suffix in (".csv", ".json", ".jsonl"):
        trains = "sft"  # read as chats as it is
    more = [f"and {len(problems) - 5} more"] if len(problems) > 5 else []
    out = DataOutput(path=str(path), shape=shape, rows=rows, lines=checked.lines, trains=trains,
                     trains_as_is=bool(trains and rows and not checked.stops),
                     samples=samples, problems=problems[:5] + more,
                     issues=[Issue(row=n, kind=k, why=w) for n, k, w in checked.issues[:LISTED]],
                     issues_more=len(checked.issues) - LISTED if len(checked.issues) > LISTED
                     else None)
    if action.convert and shape in ("convertible", "text") and (
            path.suffix != ".jsonl" or shape == "convertible"):
        # Next to a file, and in a folder, not beside it.
        target = (path / path.name if path.is_dir() else path).with_name(
            f"{path.stem}.chats.jsonl" if shape == "convertible" else f"{path.stem}.text.jsonl")
        mode = stat.S_IMODE(os.stat(path).st_mode) & 0o666  # no one reads the copy who could not
        written = 0
        try:  # a new file, or nothing: never over one, never through a link
            fd = _new_file(target, mode)
        except FileExistsError:
            raise Refusal("E_USAGE", "Already there", f"{target.name} already exists",
                          context=["delete it, or train it as it is"]) from None
        try:
            with os.fdopen(fd, "w") as fh:
                for row in _read(path):
                    if not isinstance(row, dict):
                        continue
                    made = (as_chat(row) if shape == "convertible"
                            else {"text": row.get("text", "")})
                    if made:
                        fh.write(json.dumps(made, ensure_ascii=False) + "\n")
                        written += 1
        except Refusal:
            os.unlink(target)  # what it wrote is not the file
            raise
        out.converted, out.rows = str(target), written
        out.trains = "sft" if shape == "convertible" else "classic"
        path = target
    if action.fix:
        if shape not in _KNOWN:
            raise Refusal("E_DATASET", "Nothing to keep",
                          f"no row of {path.name} is one Poros trains",
                          context=['chats are {"messages": [...]}; pairs are '
                                   '{"prompt", "chosen", "rejected"}'])
        out.kept, converted, dropped = _write_clean(path, target, shape)
        out.converted_rows, out.dropped = _listed(converted), _listed(dropped)
        out.wrote = str(target)
        path = target
    # a name fit to repeat, quoted to copy; any other is anyone's words, and is not repeated
    named = shlex.quote(path.name) if train_data.PLAIN_NAME.fullmatch(path.name) else "<this file>"
    clean = shape in _KNOWN and not checked.stops
    fixes = shape in _KNOWN and path.suffix in _FIXES  # --fix can clean it
    if out.trains and (action.fix or out.converted or clean):  # a row that stops a run: refused
        out.command = f"poros train <model> --data {named}"
    elif fixes:
        out.command = f"poros data {named} --fix"
    elif shape == "convertible":
        out.command = f"poros data {named} --convert"
    # it trains, and --fix would still drop rows of it: an empty answer, a row left out, a pair
    # whose two sides are the same
    also = clean and fixes and checked.fixable and not (action.convert or action.fix)
    return _to_copy(out, path, also=bool(also)) if under_agent() else out


def _to_copy(out: DataOutput, path: Path, also: bool = False) -> DataOutput:
    """Under `poros agents`, commands to copy: one with a hole in it goes, and `next` runs from
    here, on the small model; with ``also``, --fix after it. A file's path fit to repeat, or
    none."""
    at = _at(path)
    if at and out.command:
        flag = out.command.rsplit(" ", 1)[-1]  # --fix, --convert
        out.next = [f"poros train {shared.START_WITH} --data {shlex.quote(at)}"
                    if "<model>" in out.command else f"poros data {shlex.quote(at)} {flag}"]
        out.next += [f"poros data {shlex.quote(at)} --fix"] if also else []
    out.command = None if "<" in (out.command or "") else out.command
    return out


def view(r: DataOutput, w: int) -> list[Line]:
    from poros.cli import views

    return views.data(r, w)
