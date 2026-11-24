"""What train infers from the data without loading a model: rows and length; and the data's
own refusals, which come before the device's (``before_the_device``).

Length is estimated from characters (about 4 per token for English text),
so no tokenizer is downloaded before the confirm screen. It is labelled an
estimate wherever it is shown.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from poros.core.dataset_spec import folder_files
from poros.core.env import under_agent
from poros.core.errors import PorosDataError
from poros.core.refusal import Refusal
from poros.data import AGENT_CAP, lines
from poros.preflight import data as preflight_data  # the row rules: MAX_ROWS, CHARS_PER_TOKEN

if TYPE_CHECKING:
    from poros.cli.action import Draft
    from poros.cli.context import Context

SUFFIXES = (".jsonl", ".json", ".csv")
PLAIN_NAME = re.compile(r"[\w .-]{1,64}")  # a file's name fit to repeat: it is outside text
MAX_JSON_BYTES = 256 * 2**20  # a .json is parsed whole; past this, rows stay unknown


@dataclass
class Scan:
    rows: Optional[int]  # None: not countable here (a Hub dataset)
    seq_len: Optional[int]  # the 95th-percentile row, as a power of two in [256, 4096]
    p95_tokens: Optional[int] = None  # the same row, in tokens: what a Mac's table is read by
    chars: Sequence[int] = ()  # every row's length (``_weight``: characters, ASCII's), sorted

    def kept(self, seq: int) -> float:
        """The share of rows that fit in ``seq`` tokens."""
        return bisect_right(self.chars, seq * preflight_data.CHARS_PER_TOKEN) / len(self.chars)

    @property
    def longest_tokens(self) -> Optional[int]:
        """The longest row, in tokens (the same estimate as ``p95_tokens``); None with no rows."""
        return max(1, self.chars[-1] // preflight_data.CHARS_PER_TOKEN) if self.chars else None

    @property
    def whole_len(self) -> Optional[int]:
        """The length every row fits whole, as ``seq_len`` rounds it: what a run that cuts no row
        reaches (a config with no max_seq_length set)."""
        return length_for(self.longest_tokens) if self.chars else None


def length_for(tokens: int) -> int:
    """A row of ``tokens`` as the length Poros trains it at: a power of two in [256, 4096]."""
    return min(4096, max(256, 2 ** math.ceil(math.log2(max(1, tokens)))))


def local_files(cwd: Path) -> list[str]:
    """Data files in ``cwd``, newest first."""
    files = [p for p in cwd.iterdir() if p.is_file() and not p.name.startswith(".")
             and p.suffix.lower() in SUFFIXES]
    return [p.name for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]


def _weight(text: str) -> int:
    """A string's length as the token estimate reads it (CHARS_PER_TOKEN characters a token):
    its ASCII characters, and each other character as a whole token's worth. Chinese, Japanese,
    Korean, Arabic, emoji and the like run near a token a character (or more), not four to one,
    and a length read at four would pass a fit check the run then fails."""
    plain = len(text.encode("ascii", "ignore"))
    return plain + (len(text) - plain) * preflight_data.CHARS_PER_TOKEN


def _text(row) -> int:
    """The text in one row, as ``_weight`` counts it: every string value, nested included."""
    if isinstance(row, str):
        return _weight(row)
    if isinstance(row, dict):
        return sum(_text(v) for v in row.values())
    if isinstance(row, list):
        return sum(_text(v) for v in row)
    return 0


def _rows(path: Path) -> Iterator[int]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            for line in lines(f):
                if line.strip():
                    try:
                        yield _text(json.loads(line))
                    except ValueError:
                        yield _weight(line)
    elif suffix == ".json":
        cap = AGENT_CAP if under_agent() else MAX_JSON_BYTES
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            if os.fstat(f.fileno()).st_size > cap:
                raise ValueError("too large to count here")  # rows unknown, not guessed
            text = f.read()
        try:
            data = json.loads(text)
        except ValueError:  # JSON Lines saved as .json
            for line in text.splitlines():
                if line.strip():
                    yield _text(json.loads(line))
            return
        if isinstance(data, dict):
            data = data.get("data", [data])
        if not isinstance(data, list):
            raise ValueError("not a list of rows")
        for row in data:
            yield _text(row)
    elif suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            for row in csv.DictReader(lines(f)):  # the header names the columns: not a row
                yield sum(_weight(str(cell or "")) for cell in row.values())


def _in_folder(folder: Path) -> list[Path]:
    """The files of a folder a run would read, that can be counted here; none if it reads none."""
    try:
        return [p for p in map(Path, folder_files(folder)[1]) if p.suffix.lower() in SUFFIXES]
    except (OSError, PorosDataError):
        return []


def scan(data: str, cwd: Path) -> Scan:
    if data == "synthetic":
        return Scan(rows=None, seq_len=None)
    path = Path(data).expanduser()
    path = path if path.is_absolute() else cwd / path
    files = _in_folder(path) if path.is_dir() else [path] if path.is_file() else []
    if not files:
        return Scan(rows=None, seq_len=None)  # a Hub dataset: nothing to read here
    lengths: list[int] = []
    try:
        for f in files:
            for n in _rows(f):
                lengths.append(n)
                if len(lengths) >= preflight_data.MAX_ROWS:
                    break
    except (OSError, ValueError, csv.Error):  # unreadable: rows unknown, never a partial count
        return Scan(rows=None, seq_len=None)
    if not lengths:
        return Scan(rows=0, seq_len=None)
    lengths.sort()
    p95 = lengths[min(len(lengths) - 1, math.ceil(0.95 * len(lengths)) - 1)]
    tokens = max(1, p95 // preflight_data.CHARS_PER_TOKEN)
    return Scan(rows=len(lengths), seq_len=length_for(tokens), p95_tokens=tokens, chars=lengths)


def before_the_device(draft: "Draft", ctx: "Context") -> bool:
    """The data's own refusals, said before the device's (docs/training.md's order: the output
    folder, the data, then the GPU): a file with no rows, or rows a run would stop at. The
    output folder was checked as the words resolved. False where the question on the table is a
    data file that was not found: that is the data's refusal, and comes first too."""
    a = draft.action
    if a is None:
        asked = draft.questions[0] if draft.questions else None
        return not (asked is not None and asked.id == "data" and asked.default is None)
    if Path(a.data).is_file():
        if scan(a.data, ctx.cwd).rows == 0:
            raise Refusal("E_DATASET", "No rows", f"{a.data} has no rows to train on")
        refuse_stops(a.data)
    return True


def refuse_stops(path: str) -> None:
    """A row a run would fail at, or train on nothing at, refused now: in the queue it fails only
    when its turn comes. The first 200,000 rows are read, as `data` reads them."""
    from poros.cli.verbs import data as data_verb
    from poros.preflight import data as preflight_data

    stops = preflight_data.stops(Path(path))
    if stops:
        more = f"; and {len(stops) - 3} more" if len(stops) > 3 else ""
        plain = data_verb._PLAIN_PATH.fullmatch(path)  # a name fit to repeat, or none
        raise Refusal("E_DATASET", "Rows that stop a run", "; ".join(stops[:3]) + more,
                      context=["`poros data <file> --fix` writes a clean copy beside your "
                               "file: it leaves out the rows a run cannot train on, turns the "
                               "rest into one shape, and never changes your file"],
                      fix=shlex.join(["poros", "data", path, "--fix"]) if plain else "",
                      note="writes a clean copy, and the train command for it")


def same_shape(held: Optional[str], shape: Optional[str]) -> None:
    """An --eval file holds rows of the training data's shape: chats for chats, pairs for pairs.
    Refused now, not when the run reaches its first score."""
    from poros import data as pdata

    if not held or shape is None:
        return
    try:
        said = pdata.detect(Path(held))
    except ValueError as exc:
        raise Refusal("E_DATASET", "Can't read the eval file", str(exc)) from None
    if said != shape:
        raise Refusal("E_USAGE", "Not the data's shape",
                      f"{Path(held).name} holds {said}, and the data holds {shape}",
                      context=["--eval takes held-out rows of the data's shape"])
