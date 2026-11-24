"""The dataset a run reads, checked before the model download: a Hub id that is missing or
gated, and a local file's rows that would stop a run.

The rows are read here as the trainer reads them (``stops``, ``_Checked``): ``poros data``
shows the same reading, and ``poros train`` and ``PorosTrainer`` refuse from it. Torch-free.
"""

from __future__ import annotations

import csv
import json
import os
import re
import stat
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from poros import data as poros_data
from poros.core.env import outside_words_out
from poros.core.errors import CONTROL, KEY, no_keys
from poros.core.refusal import Refusal
from poros.data import ASKS, SAYS, _pick, as_chat


def checked_before_load(path: Path) -> bool:
    """Whether ``poros train`` reads the rows of the data file ``path`` before the load: a file
    `poros data` reads row for row as the trainer does (.jsonl, .json, .csv). Any other file the
    trainer reads (a .parquet, a .txt) is the trainer's to read."""
    return path.is_file() and path.suffix in _FIXES


def hub_dataset_refusal(dataset_id: str) -> Optional[Refusal]:
    """The refusal of a definitively missing or gated Hub dataset id; None on anything else
    (offline, rate limits): a network hiccup must never block a run that could still resolve
    from the local cache."""
    try:
        from huggingface_hub import dataset_info
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:
        return None
    try:
        dataset_info(dataset_id)
    except GatedRepoError:  # subclass of RepositoryNotFoundError: check first
        return Refusal("E_HUB_AUTH", "Gated dataset",
                       f"{dataset_id} needs its license accepted on huggingface.co",
                       fix="hf auth login", note="then retry")
    except RepositoryNotFoundError:
        return Refusal("E_DATASET_NOT_FOUND", "Dataset not found",
                       f"{dataset_id} is not on the Hub",
                       context=["if it is private or gated, authenticate first; "
                                "otherwise check the spelling"],
                       fix="hf auth login")
    except Exception:
        return None
    return None



MAX_ROWS = 200_000  # scan at most this many rows; past it, rows is a lower bound


CHARS_PER_TOKEN = 4



SAMPLES = 3


WALK_CAP = 100_000  # entries of a folder looked at


LONGEST = 4096  # the longest length train picks by itself; a row past it has its answer cut away


Shape = Literal["chats", "pairs", "text", "convertible", "unknown"]


def _open(path: Path, **how):
    """A regular file, open: a pipe cannot hang this, and a device cannot fill memory."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("it is not a regular file")
    return os.fdopen(fd, **how)


class _Unread(str):
    """What Poros says of a row or a folder it could not read. A row's own keys cannot be this."""


class _Blank:
    """A blank line of a .jsonl: no row, and it keeps row N line N."""


_BLANK = _Blank()


class _Tally:
    """The lines read so far: each newline ends one, and text after the last is one more."""

    def __init__(self) -> None:
        self.ended, self.open = 0, False

    def read(self, text: str) -> str:
        if text:
            self.ended += text.count("\n")
            self.open = not text.endswith("\n")
        return text

    @property
    def n(self) -> int:
        return self.ended + self.open


def _counted(lines: Iterator[str], tally: Optional[_Tally]) -> Iterator[str]:
    for line in lines:
        yield tally.read(line) if tally else line


def _whole(path: Path) -> str:
    with _open(path, encoding="utf-8-sig", errors="ignore") as fh:
        cap = poros_data.whole_cap()
        if os.fstat(fh.fileno()).st_size > cap:
            raise ValueError(f"it is over {cap:,} bytes to read at once")
        return fh.read()


def _paragraphs(path: Path, tally: Optional[_Tally] = None) -> Iterator[dict]:
    """A text file's blocks, as ``split("\\n\\n")`` gives them, one at a time: splitting a whole
    file of ab\\n\\n took 15 times its size. No block is held past the cap on a line."""
    with _open(path, encoding="utf-8-sig", errors="ignore") as fh:
        held = ""
        while chunk := fh.read(1 << 20):
            if tally:
                tally.read(chunk)
            *blocks, held = (held + chunk).split("\n\n")
            for block in blocks:
                if block.strip():
                    yield {"text": block.strip()}
            if len(held) > poros_data.LINE_CAP:
                raise ValueError(f"a block is longer than {poros_data.LINE_CAP:,} characters")
        if held.strip():
            yield {"text": held.strip()}


def _records(path: Path, tally: Optional[_Tally] = None) -> Iterator["dict | _Unread | _Blank"]:
    if path.is_dir():
        found = list(islice(path.rglob("*"), WALK_CAP + 1))
        for f in sorted(found[:WALK_CAP]):
            # a link in a folder was named by no one: it may reach anywhere
            if f.suffix in (".txt", ".md") and f.is_file() and not f.is_symlink():
                yield {"text": _whole(f)}
        if len(found) > WALK_CAP:
            yield _Unread(f"only the first {WALK_CAP:,} files here were read")
        return
    if path.suffix == ".csv":
        with _open(path, newline="", encoding="utf-8-sig", errors="ignore") as fh:
            yield from csv.DictReader(_counted(poros_data.lines(fh), tally))
        return
    if path.suffix in (".txt", ".md"):
        yield from _paragraphs(path, tally)
        return
    if path.suffix == ".json":
        text = _whole(path)
        if tally:
            tally.read(text)
        rows = json.loads(text)
        yield from (rows if isinstance(rows, list) else [rows])
        return
    with _open(path, encoding="utf-8-sig", errors="ignore") as fh:
        for n, line in enumerate(_counted(poros_data.lines(fh), tally), 1):
            if not line.strip():
                yield _BLANK
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield _Unread(f"row {n} is not JSON")


def _read(path: Path, tally: Optional[_Tally] = None) -> Iterator["dict | _Unread | _Blank"]:
    """The rows, with what cannot be read as one refusal: its reason names no row's words."""
    try:
        yield from _records(path, tally)
    except (OSError, ValueError, csv.Error, RecursionError) as exc:
        reason = "it nests too deeply" if isinstance(exc, RecursionError) else (
            str(exc) if not isinstance(exc, OSError) else exc.strerror or "it cannot be read")
        raise Refusal("E_DATASET", "Can't read it", f"{path.name}: {reason}") from None


def _shape(row: dict) -> Shape:
    if {"chosen", "rejected"} <= set(row):
        return "pairs"
    if any(k in row for k in ("messages", "conversations", "conversation")):
        return "chats"
    if _pick(row, ASKS) and _pick(row, SAYS):
        return "convertible"
    if "text" in row:
        return "text"
    return "unknown"


_PLAIN_KEY = re.compile(r"[\w .-]{1,40}", re.ASCII)


def _keys_only(row: dict, path: Path) -> str:
    """A row's field names and none of its values, for an agent: only chats, pairs and
    question/answer rows are training data for sure, and a file of anything else may hold anything.
    Its field names too: a name shaped like a key is a key, and a csv's first line is whatever
    anyone wrote."""
    def named(k: str) -> bool:
        return k == "text" or (path.suffix != ".csv" and bool(_PLAIN_KEY.fullmatch(k))
                               and not KEY.search(k))

    return " · ".join(str(k) if named(str(k)) else "?" for k in list(row)[:8])


def _turns_of(row: dict) -> list[dict]:
    """A chat's turns as the trainer reads them, role and content strings; none when it cannot:
    ``messages`` may be one string, or one message."""
    try:
        return poros_data._turns(row, 0) or []
    except ValueError:
        return []


def _line(row: dict, w: int = 96) -> str:
    if "messages" in row or "conversations" in row:
        text = " → ".join(t["content"][:60] for t in _turns_of(row)[:2])
    elif "chosen" in row:
        text = (f"{str(row.get('prompt', ''))[:40]} ✓ {str(row['chosen'])[:30]} "
                f"✕ {str(row['rejected'])[:30]}")
    else:
        text = " · ".join(f"{str(k)[:w]}: {str(v)[:w]}" for k, v in list(row.items())[:3])
    text = no_keys(CONTROL.sub("", " ".join(text.split())))
    return text if len(text) <= w else text[: w - 1] + "…"


def _before_the_answer(row: dict) -> int:
    """Characters ahead of a row's first answer, which is all a length has to cut to lose it."""
    if "prompt" in row:
        return len(str(row["prompt"]))
    before = 0
    for turn in _turns_of(row):
        if turn["role"] in ("assistant", "gpt"):
            break
        before += len(turn["content"])
    return before


# What is worth saying of a row and is no reason to refuse a file: the trainer trains through it.
_TRAINS_THROUGH = ("would train the model to say nothing", "empty chosen answer",
                   "empty rejected answer")


def _stops(row: dict, shape: Shape, n: int) -> Optional[str]:
    """What stops a run at this row, or trains the model on nothing, in the trainer's own words:
    its own check of a chat's or a pair's messages. None: the row trains."""
    try:
        if shape == "chats":
            turns = poros_data._turns(row, n) or []
            if any(t["role"] in ("assistant", "gpt") and not t["content"].strip() for t in turns):
                return f"row {n} has an empty answer: it would train the model to say nothing"
        elif shape == "pairs":
            if "prompt" not in row:
                return f"row {n} has no 'prompt'"
            poros_data._as_turns(row["prompt"], "user", n)
            for key in ("chosen", "rejected"):
                if not any(t["content"].strip()
                           for t in poros_data._as_turns(row[key], "assistant", n)):
                    return f"row {n} has an empty {key} answer"
    except ValueError as exc:  # the trainer's refusal: our words, none of the row's
        return str(exc)
    return None


def _same_sides(row: dict, n: int) -> bool:
    """A pair whose chosen and rejected the trainer reads as the same turns: DPO scores its two
    sides alike, so its loss sits at ln 2 and it teaches nothing. Asked once `_stops` passes it."""
    try:
        return (poros_data._as_turns(row["chosen"], "assistant", n)
                == poros_data._as_turns(row["rejected"], "assistant", n))
    except (KeyError, ValueError):
        return False


_KNOWN = ("chats", "pairs", "text", "convertible")  # the shapes Poros trains


_FIXES = (".jsonl", ".json", ".csv")  # the files --fix cleans: the ones `train` reads


def _said(row: dict, this: Shape, n: int, longest: int) -> tuple:
    """What `data` says of a row in its own shape, each with its kind: ``stop`` a run fails at it,
    ``out`` the trainer leaves it out, ``empty`` it trains the model to say nothing, ``same`` a
    pair's two sides are one answer, ``say`` it trains, and is worth a word. Nothing: it
    trains."""
    if this == "convertible" and as_chat(row) is None:
        return (("out", f"row {n} has no answer; it is left out"),)
    stop = _stops(row, this, n)
    if stop and not stop.endswith(_TRAINS_THROUGH):
        return (("stop", stop),)
    if this == "chats" and not any(t["role"] == "assistant" for t in _turns_of(row)):
        # the chats builder scores assistant turns alone (poros.data._render_chat): a row with
        # none has no token to learn from, and it keeps no such row (poros.data.chats)
        return (("out", f"row {n} has no assistant turn; it is left out"),)
    said: tuple = (("empty", stop),) if stop else ()
    if not stop and this == "pairs" and _same_sides(row, n):
        said += (("same", f"row {n}: chosen and rejected are the same: it teaches nothing"),)
    if this in ("chats", "pairs") and (
            _before_the_answer(row) > longest * CHARS_PER_TOKEN):
        said += (("say", f"row {n}: the answer starts past {longest} tokens, and is cut away"),)
    return said


def _trains(said: tuple) -> bool:
    return not any(kind in ("stop", "out") for kind, _ in said)


_AS_CHATS = ("chats", "convertible")  # what the chats builder reads alike, row by row


KINDS = {"stop": "fails", "out": "dropped", "empty": "empty_answer", "same": "identical",
         "say": "cut"}


_DROPS = ("out", "empty", "same")  # what --fix leaves out of a file that trains as it is


def _main(counts: dict) -> Shape:
    """A file's shape: the one most of its rows have, of those Poros trains; the first one's on a
    tie. A first row of another shape does not make every other row the odd one."""
    known = [shape for shape in counts if shape in _KNOWN]
    return max(known, key=counts.__getitem__) if known else "unknown"


class _Checked:
    """A file's rows as `data` reads them: what they hold, and what is wrong. ``stops`` are the
    problems a run fails at, or trains on nothing at; the rest are rows it leaves out or cuts, and
    a bound on the read. Up to MAX_ROWS rows, as the count is. Its shape is most rows' shape, and
    a row of another is a stop, but for a question/answer row among chats or a chat among them:
    the trainer reads both as chats. ``issues`` are the rows of ``problems``, as (row, kind, why).
    ``lines`` are the lines read, None for a folder."""

    def __init__(self, path: Path, longest: int = LONGEST) -> None:
        self.rows, self.problems, self.stops = 0, [], []
        self.issues: list[tuple] = []  # (row, kind, why): an Issue each, only for those shown
        self.fixable = 0  # rows --fix drops that are no stop: an empty answer, a row left out, a
        # pair that teaches nothing
        tally = None if path.is_dir() else _Tally()
        seen: list = []  # (row number, its shape, what is said of it), in the file's order
        counts: dict = {}  # each shape's rows, in the order first seen
        samples: dict = {}  # each shape's first rows that train
        read, capped = 0, False
        for n, row in enumerate(_read(path, tally), 1):
            if row is _BLANK:
                continue
            read += 1
            if read > MAX_ROWS:
                capped = True
                break
            if isinstance(row, _Unread) and not str(row).startswith("row "):
                seen.append((n, None, (("say", str(row)),)))  # a folder read only so far
                continue
            if not isinstance(row, dict):
                why = str(row) if isinstance(row, _Unread) else f"row {n} is not an object"
                seen.append((n, None, (("stop", why),)))
                continue
            this = _shape(row)
            counts[this] = counts.get(this, 0) + 1
            said = _said(row, this, n, longest)
            seen.append((n, this, said))
            mine = samples.setdefault(this, [])
            if _trains(said) and len(mine) < SAMPLES:
                real = this in ("chats", "pairs", "convertible") or not outside_words_out()
                sample = _line(row) if real else _keys_only(row, path)
                if real or sample not in mine:
                    mine.append(sample)
        self.shape: Shape = _main(counts)
        for n, this, said in seen:
            if this is not None and this != self.shape:
                if not {this, self.shape} <= set(_AS_CHATS):
                    self._stop(f"row {n} is {this}, not {self.shape}", n)
                    continue
                if _trains(said):  # the words --fix says of it too, converted or dropped
                    self._say(f"row {n} is {this}, not {self.shape}", n, "converted")
            for kind, problem in said:
                if kind == "stop":
                    self._stop(problem, n)
                else:
                    self._say(problem, n if this is not None else None, KINDS[kind])
            self.rows += this is not None and _trains(said)
            self.fixable += any(kind in _DROPS for kind, _ in said)
        if capped:
            self.problems.insert(0, f"stopped after {MAX_ROWS:,} rows; there may be "
                                    "more")
        self.samples: list[str] = samples.get(self.shape, [])
        self.lines: Optional[int] = tally.n if tally else None

    def _stop(self, problem: str, n: int) -> None:
        self._say(problem, n, "fails")
        self.stops.append(problem)

    def _say(self, problem: str, n: Optional[int], kind: str) -> None:
        """A problem, and the row it is of as data: a bound on the read is of no row."""
        self.problems.append(problem)
        if n is not None:
            self.issues.append((n, kind, problem))


def stops(path: Path, longest: int = LONGEST) -> list[str]:
    """What in a data file stops a run, before one is queued: `train` reads it as `data` does."""
    return _Checked(path, longest).stops


def rows_and_stops(path: Path, longest: int = LONGEST) -> tuple[int, list[str]]:
    """``stops``, with the rows read: a file with none trains nothing, and stops no row."""
    read = _Checked(path, longest)
    return read.rows, read.stops


def degenerate_end_row(head_weight: Any, end_id: int, special_ids: Any, *, frozen: bool,
                       train_on: str, end_name: str = "the turn's end token") -> Optional[str]:
    """Why a chat run cannot learn to end its answers, said before it trains: the LM head is
    frozen, the run trains on the answers (so on the end token), and the end token's head row is
    the same as another special token's (cosine 0.9999 or more: Qwen2.5 base models give
    ``<|im_end|>`` the row of ``<|im_start|>``). Its logit then moves only as that token's does,
    so no update can make it the likeliest next token: ``poros.ends`` reports "never ends". None
    where the head trains, the answers are not scored, or the row is its own."""
    if not frozen or train_on not in ("answers", "last") or head_weight is None:
        return None
    import torch

    others = sorted({int(i) for i in (special_ids or []) if int(i) != int(end_id)
                     and 0 <= int(i) < head_weight.shape[0]})
    if not others or not 0 <= int(end_id) < head_weight.shape[0]:
        return None
    with torch.no_grad():
        rows = head_weight[[int(end_id), *others]].detach().to("cpu", torch.float32)
        end, rest = rows[0], rows[1:]
        cos = (rest @ end) / (rest.norm(dim=1) * end.norm() + 1e-30)
    best = int(cos.argmax())
    if float(cos[best]) < 0.9999:
        return None
    return (f"{end_name} (id {int(end_id)}) has the LM head row of token {others[best]} (cosine "
            f"{float(cos[best]):.7f}), and the head is frozen: the run cannot learn to end its "
            "answers, so it will not stop on its own. Train the Instruct model of this base "
            "(its end token is trained), or adapt the head too (lm_head in lora_target_modules)")
