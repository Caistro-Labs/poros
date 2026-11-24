"""Turn rows into batches for ``poros.Run``.

    data = poros.data.chats("chats.jsonl", run.tokenizer)          # SFT, answers only
    data = poros.data.text("notes.jsonl", run.tokenizer)           # every token
    data = poros.data.pairs("prefs.jsonl", run.tokenizer)          # DPO
    batch = data.batch(step)                                       # resumable by index

A source is a ``.jsonl`` / ``.json`` / ``.csv`` file, a list of dicts, or a
Hugging Face dataset name. Question/answer rows (a csv, Alpaca,
prompt/completion) are read as chats: no converting first. Rows are
rendered once, up front, so a bad row fails before
anything loads on the GPU, with its row number.

``batch(i)`` is a pure function of ``i``: epoch ``i // len(data)``, shuffled
by ``seed + epoch``. A resumed run asks for the same index and gets the same
batch, with nothing to fast-forward.
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

from poros.core.env import outside_words_out, under_agent
from poros.core.errors import KEY
from poros.core.template_guard import template_bound
from poros.train.batch import Batch, Datum

LINE_CAP = 64 * 2**20  # characters in one row of a text file
WHOLE_CAP = 2**30  # bytes of a file read at once: .json
AGENT_CAP = 64 << 20  # the same under an agent: 114 MiB of [{},...] reached 2.98 GB in memory
# Names a folder's listing leaves out: what such a file holds is not a training set.
SECRET_NAMES = ("*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_dsa*", "id_ecdsa*",
                "id_ed25519*", "credentials*", "secrets*", "*.tfstate*", "service-account*")

Source = Union[str, Path, Sequence[Dict[str, Any]]]
TrainOn = Literal["answers", "last", "all"]


class Data:
    """Rendered rows and how to batch them."""

    def __init__(
        self,
        items: List[Any],
        *,
        batch_size: int,
        seed: int = 0,
        pad_id: int = 0,
        pairs: bool = False,
        kind: str = "",
    ) -> None:
        if not items:
            raise ValueError("no rows to train on")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.items, self.batch_size, self.seed = items, batch_size, seed
        self.pad_id, self.pairs, self.kind = pad_id, pairs, kind
        self._orders: Dict[int, List[int]] = {}
        # Sort each step's rows by length before they are split into micro-batches, so a
        # micro-batch pads to rows of about its own length (same rows per step, same epoch
        # order; only the order inside a step changes). PorosConfig.group_by_length.
        self.group_by_length = False
        # each item's 1-based row in its file, where a builder knows it: what eval --show reads
        # a row's own words back from. Never part of the fingerprint.
        self.numbers: Optional[List[int]] = None

    @property
    def rows(self) -> int:
        return len(self.items)

    def __len__(self) -> int:
        """Batches per epoch."""
        return -(-self.rows // self.batch_size)

    @property
    def fingerprint(self) -> str:
        """Names this exact rendered data (caches key on it)."""
        h = hashlib.sha1(self.kind.encode())
        for item in self.items:
            for d in item if self.pairs else (item,):
                h.update(repr((d.tokens, d.target_tokens, d.weights)).encode())
        return h.hexdigest()[:16]

    def batch(self, index: int) -> Batch:
        """The batch for global step ``index``: epochs reshuffle by seed."""
        epoch, pos = divmod(index, len(self))
        order = self._order(epoch)
        idx = order[pos * self.batch_size:(pos + 1) * self.batch_size]
        if self.group_by_length and not self.pairs:
            idx = sorted(idx, key=lambda i: len(self.items[i]))  # stable: ties keep epoch order
        return self._make(idx)

    def ordered(self) -> Iterator[Batch]:
        """Every row once, in file order (eval, reference logprobs)."""
        for start in range(0, self.rows, self.batch_size):
            yield self._make(list(range(start, min(start + self.batch_size, self.rows))))

    def holdout(self, share: float, count: int = 0):
        """(train, held): ``share`` of the rows, picked by seed, kept out to
        score; held is None under 40 rows, where every row should train.
        ``count`` holds out that many instead, at most half the rows."""
        n = _held(self.rows, share, count)
        if not n:
            return self, None
        picked = set(random.Random(self.seed).sample(range(self.rows), n))
        keep = dict(batch_size=self.batch_size, seed=self.seed, pad_id=self.pad_id,
                    pairs=self.pairs)
        train = [x for i, x in enumerate(self.items) if i not in picked]
        held = [x for i, x in enumerate(self.items) if i in picked]
        train, held = (Data(train, kind=self.kind + ":train", **keep),
                       Data(held, kind=self.kind + ":held", **keep))
        train.group_by_length = held.group_by_length = self.group_by_length
        if self.numbers is not None:
            train.numbers = [x for i, x in enumerate(self.numbers) if i not in picked]
            held.numbers = [x for i, x in enumerate(self.numbers) if i in picked]
        return train, held

    def _order(self, epoch: int) -> List[int]:
        if epoch not in self._orders:
            order = list(range(self.rows))
            random.Random(self.seed + epoch).shuffle(order)
            self._orders = {epoch: order}  # one epoch at a time is all a run needs
        return self._orders[epoch]

    def _make(self, idx: List[int]) -> Batch:
        if not self.pairs:
            return Batch([self.items[i] for i in idx], ids=idx, pad_id=self.pad_id)
        chosen = [self.items[i][0] for i in idx]
        rejected = [self.items[i][1] for i in idx]
        return Batch(chosen + rejected, ids=idx + idx, pairs=True, pad_id=self.pad_id)


SHARE_CAP = 100  # a share holds out this many rows at most: every score passes over them all
FEWEST = 40  # under this many rows a share holds none out: every row should train


def _held(rows: int, share: float, count: int = 0) -> int:
    if count > 0:  # asked for by number: that many, so long as half the rows still train
        return min(count, rows // 2)
    return 0 if share <= 0 or rows < FEWEST else max(2, min(SHARE_CAP, round(rows * share)))


def held_count(rows: int, share: float, count: int = 0) -> int:
    """How many of ``rows`` ``holdout`` keeps out at ``share`` (or ``count``): what a run does
    not train on."""
    return _held(rows, share, count)


def held_of_kept(kept: int, share: float, count: int = 0) -> Optional[int]:
    """How many rows ``holdout`` kept out of a run that trained on ``kept``, or None if no count
    gives that. For a run whose eval events do not say."""
    if count > 0:
        return count if count <= kept else None
    return next((n for n in range(2, SHARE_CAP + 1) if _held(kept + n, share) == n), None)


def held_rows(cfg: Any) -> int:
    """How many of a config's local rows ``holdout`` keeps out (0: none)."""
    path = Path(str(cfg.dataset or "")).expanduser()
    if getattr(cfg, "eval_dataset", None) or not path.is_file():
        return 0
    try:
        return _held(sum(1 for _ in _rows(path)), getattr(cfg, "eval_split", 0) or 0,
                     getattr(cfg, "eval_rows", 0) or 0)
    except (OSError, ValueError):
        return 0


def train_rows(cfg: Any) -> Optional[int]:
    """How many rows of a config's local dataset a recipe trains on, at most, read before
    anything loads: the file's rows less those ``holdout`` keeps out (a builder may drop more).
    None where the data is not a local file. The plan gate reads it: a recipe of at most
    ``per_device_batch_size`` rows runs each step as one micro-batch."""
    path = Path(str(cfg.dataset or "")).expanduser()
    if not path.is_file():
        return None
    try:
        return sum(1 for _ in _rows(path)) - held_rows(cfg)
    except (OSError, ValueError):
        return None


# ── Builders ───────────────────────────────────────────────────────────


def chats(
    source: Source,
    tokenizer: Any,
    *,
    train_on: TrainOn = "answers",
    length: int = 2048,
    batch_size: int = 8,
    seed: int = 0,
) -> Data:
    """Conversations (``messages``, ``conversations``: role and content).

    ``train_on``: ``answers`` scores every assistant turn, ``last`` only the
    final one, ``all`` every token (the system prompt and user turns too).
    """
    items, numbers = [], []
    for n, row in _rows(source):
        turns = _turns(row, n)
        if turns is None:  # a question with no answer: nothing to learn
            continue
        seq, weights = _render_chat(tokenizer, turns, train_on)
        item = _datum(seq, weights, length, n)
        if any(item.weights):  # with its answer cut away at ``length``: nothing to learn
            items.append(item)
            numbers.append(n)
    out = Data(items, batch_size=batch_size, seed=seed, pad_id=_pad(tokenizer),
               kind=f"chats:{train_on}:{length}")
    out.numbers = numbers
    return out


def text(
    source: Source,
    tokenizer: Any,
    *,
    length: int = 2048,
    batch_size: int = 8,
    seed: int = 0,
    field: str = "text",
) -> Data:
    """Plain text rows (``text``, or ``field``): every token is scored."""
    items = []
    for n, row in _rows(source):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"row {n} has no {field!r} text")
        seq = _encode(tokenizer, value, special=True) + _eos(tokenizer)
        items.append(_datum(seq, [1.0] * len(seq), length, n))
    return Data(items, batch_size=batch_size, seed=seed, pad_id=_pad(tokenizer),
                kind=f"text:{field}:{length}")


def pairs(
    source: Source,
    tokenizer: Any,
    *,
    length: int = 2048,
    batch_size: int = 4,
    seed: int = 0,
) -> Data:
    """Preference rows: ``prompt``, ``chosen``, ``rejected``.

    The prompt is a string or a list of messages; each answer a string, an
    assistant message, or a list of messages. Only the answer is scored.
    """
    items, numbers = [], []
    for n, row in _rows(source):
        for key in ("prompt", "chosen", "rejected"):
            if key not in row:
                raise ValueError(f"row {n} has no {key!r}")
        prompt = _as_turns(row["prompt"], "user", n)
        both = []
        for key in ("chosen", "rejected"):
            answer = _as_turns(row[key], "assistant", n)
            seq, weights = _render_answer(tokenizer, prompt, answer)
            both.append(_datum(seq, weights, length, n))
        if all(any(item.weights) for item in both):  # a side cut away leaves nothing to prefer
            items.append(tuple(both))
            numbers.append(n)
    out = Data(items, batch_size=batch_size, seed=seed, pad_id=_pad(tokenizer),
               pairs=True, kind=f"pairs:{length}")
    out.numbers = numbers
    return out


def same_pairs(data: Data) -> int:
    """How many of ``data``'s pairs train their two sides as the same tokens, after the length
    cut: DPO scores such a pair alike on both sides, so it teaches nothing. Reads the items, and
    changes none of them."""
    if not data.pairs:
        return 0
    return sum((a.tokens, a.target_tokens, a.weights) == (b.tokens, b.target_tokens, b.weights)
               for a, b in data.items)


def detect(source: Source) -> str:
    """Which builder a source's first row is for: ``chats``, ``pairs``,
    ``text``. The CLI picks the recipe with it."""
    for n, row in _rows(source):
        if {"chosen", "rejected"} <= set(row):
            return "pairs"
        if any(k in row for k in ("messages", "conversations", "conversation")):
            return "chats"
        if _pick(row, ASKS) and _pick(row, SAYS):
            return "chats"
        if "text" in row:
            return "text"
        raise ValueError(
            f"row {n} is not chats (messages), pairs (prompt, chosen, rejected) "
            f"or text; its fields: {', '.join(_field_names(row, source)) or 'none'}"
        )
    raise ValueError("no rows")


def _field_names(row: Dict[str, Any], source: Source) -> List[str]:
    """A row's field names, which are anyone's words: not one in another's voice or shaped like a
    key, and under an agent or in a job, whose error an agent reads, none of a csv's: its first line
    is whatever anyone wrote there. A person's own command still names them, before a job is
    queued."""
    csv_line = outside_words_out() and Path(str(source)).suffix == ".csv"
    plain = [str(k) for k in row]
    return ["?" if csv_line or not re.fullmatch(r"[\w .-]{1,64}", k) or KEY.search(k) else k
            for k in sorted(plain)][:12]


# ── Rendering ──────────────────────────────────────────────────────────


Rendered = Tuple[List[int], List[float]]  # a sequence and each token's weight
Turns = List[Dict[str, str]]


def _render_chat(tok: Any, turns: Turns, train_on: TrainOn) -> Rendered:
    if not _has_template(tok):
        return _render_plain(tok, turns, train_on)
    full = _chat_ids(tok, turns, generation=False)
    if train_on == "all":
        return full, [1.0] * len(full)
    weights = [0.0] * len(full)
    answers = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    if train_on == "last":
        answers = answers[-1:]
    for i in answers:
        start = _common(full, _chat_ids(tok, turns[:i], generation=True))
        if i == len(turns) - 1:
            end = len(full)
        else:
            end = _common(full, _chat_ids(tok, turns[: i + 1], generation=False))
        for p in range(start, max(start, end)):
            weights[p] = 1.0
    return full, weights


def _render_answer(tok: Any, prompt: Turns, answer: Turns) -> Rendered:
    if not _has_template(tok):
        return _render_plain(tok, prompt + answer, "answers")
    full = _chat_ids(tok, prompt + answer, generation=False)
    start = _common(full, _chat_ids(tok, prompt, generation=True))
    return full, [0.0] * start + [1.0] * (len(full) - start)


def _render_plain(tok: Any, turns: Turns, train_on: TrainOn) -> Rendered:
    """No chat template: ``role: content`` lines, each turn tokenized alone."""
    seq: List[int] = _bos(tok)
    weights: List[float] = [0.0] * len(seq)
    last = max((i for i, t in enumerate(turns) if t["role"] == "assistant"), default=-1)
    for i, t in enumerate(turns):
        ids = _encode(tok, f"{t['role']}: {t['content']}\n", special=False)
        scored = train_on == "all" or (
            t["role"] == "assistant" and (train_on == "answers" or i == last)
        )
        seq += ids
        weights += [1.0 if scored else 0.0] * len(ids)
    eos = _eos(tok)  # ends the final turn: scored when that turn is
    scored = train_on == "all" or last == len(turns) - 1
    return seq + eos, weights + [1.0 if scored else 0.0] * len(eos)


def _datum(seq: List[int], weights: List[float], length: int, n: int) -> Datum:
    seq, weights = seq[: length + 1], weights[: length + 1]
    if len(seq) < 2:
        raise ValueError(f"row {n} renders to fewer than two tokens")
    return Datum.from_sequence(seq, weights)


def _common(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _chat_ids(tok: Any, turns: List[Dict[str, str]], *, generation: bool) -> List[int]:
    if not turns:
        return _bos(tok) if generation else []
    with template_bound():
        out = tok.apply_chat_template(
            turns, tokenize=True, add_generation_prompt=generation, return_dict=False
        )
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def _has_template(tok: Any) -> bool:
    return bool(getattr(tok, "chat_template", None))


def _encode(tok: Any, s: str, *, special: bool) -> List[int]:
    return list(tok(s, add_special_tokens=special)["input_ids"])


def _eos(tok: Any) -> List[int]:
    eos = getattr(tok, "eos_token_id", None)
    return [eos] if isinstance(eos, int) else []


def _bos(tok: Any) -> List[int]:
    bos = getattr(tok, "bos_token_id", None)
    return [bos] if isinstance(bos, int) and getattr(tok, "add_bos_token", False) else []


def _pad(tok: Any) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tok, name, None)
        if isinstance(value, int):
            return value
    return 0


def _turns(row: Dict[str, Any], n: int) -> Optional[List[Dict[str, str]]]:
    for key in ("messages", "conversations", "conversation"):
        if key in row:
            return _as_turns(row[key], "user", n)
    if _pick(row, ASKS) and _pick(row, SAYS):
        chat = as_chat(row)
        return chat["messages"] if chat else None
    raise ValueError(f"row {n} has no messages")


# Question/answer column names, first match wins (case-blind).
ASKS = ("prompt", "question", "instruction", "input", "query", "user", "human")
SAYS = ("completion", "answer", "response", "output", "reply", "assistant", "gpt")


def _pick(row: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    low = {str(k).lower(): k for k in row}
    return next((low[n] for n in names if n in low), None)


def as_chat(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A question/answer row as a chat, None when it has no answer. Alpaca's
    ``input`` joins its instruction; a ``system`` column is the system turn."""
    ask, say = _pick(row, ASKS), _pick(row, SAYS)
    if not ask or not say or not str(row[say] or "").strip():
        return None
    prompt = str(row[ask]).strip()
    extra = row.get("input") if str(ask).lower() == "instruction" else None
    if extra and str(extra).strip():
        prompt = f"{prompt}\n\n{str(extra).strip()}"
    system = row.get("system")
    turns = [{"role": "system", "content": str(system)}] if system else []
    turns += [{"role": "user", "content": prompt},
              {"role": "assistant", "content": str(row[say]).strip()}]
    return {"messages": turns}


def _as_turns(value: Any, role: str, n: int) -> List[Dict[str, str]]:
    """A string, one message, or a list of messages, as role/content dicts
    (ShareGPT's from/value is accepted)."""
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"row {n}: expected text or a list of messages")
    turns = []
    for t in value:
        if not isinstance(t, dict):
            raise ValueError(f"row {n}: a message must be an object with role and content")
        sharegpt = t.get("from")
        r = t.get("role") or {"human": "user", "gpt": "assistant"}.get(sharegpt, sharegpt)
        c = t.get("content", t.get("value"))
        if not r or not isinstance(c, str):
            raise ValueError(f"row {n}: a message needs a role and text content")
        turns.append({"role": str(r), "content": c})
    return turns


def secret_name(name: str) -> bool:
    """A file or folder named as a secret: hidden, or matching SECRET_NAMES, in any case."""
    name = name.lower()
    return name.startswith(".") and name not in (".", "..") or any(
        fnmatch.fnmatchcase(name, pattern) for pattern in SECRET_NAMES)


def lines(f: Any) -> Iterator[str]:
    """A text file's lines, exactly as iterating it gives them, none held past the cap: a file with
    no newlines cannot fill memory. Past it, a ValueError, in no one's words."""
    cap = LINE_CAP
    while line := f.readline(cap + 1):
        if len(line) > cap:
            raise ValueError(f"a line is longer than {cap:,} characters")
        yield line


def whole_cap() -> int:
    """Bytes of a file read at once, here: a person's own file has more room than an agent's."""
    return AGENT_CAP if under_agent() else WHOLE_CAP


def whole(path: Path) -> str:
    """A file read at once, as ``.json`` has to be, up to the cap."""
    with open(path, encoding="utf-8-sig") as f:
        if os.fstat(f.fileno()).st_size > whole_cap():
            raise ValueError(f"over {whole_cap():,} bytes to read at once: use .jsonl")
        return f.read()


def _rows(source: Source) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """(1-based row number, row) from a file, a list, or a Hub dataset."""
    if not isinstance(source, (str, Path)):
        rows: Any = source
    else:
        path = Path(source)
        if path.suffix == ".jsonl":
            rows = _jsonl(path)
        elif path.suffix == ".json":
            rows = json.loads(whole(path))
        elif path.suffix == ".csv":
            rows = _csv(path)
        elif path.exists():
            raise ValueError(f"{path}: use a .jsonl, .json or .csv file")
        else:
            from poros.core.dataset_spec import load_hub_dataset

            rows = load_hub_dataset(str(source))
    for n, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"row {n} is not an object")
        yield n, row


def _csv(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(lines(f))


def _jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8-sig") as f:  # a leading BOM (Windows editors) is no row
        for n, line in enumerate(lines(f), 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} line {n}: not JSON ({exc.msg})") from None
