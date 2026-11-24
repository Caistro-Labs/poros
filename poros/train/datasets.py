"""The data a PorosTrainer reads: the license-free synthetic tokens, a local file or a Hub
dataset tokenized into training rows (chat turns, prompt/response pairs, plain text), the
padding collate, and the remedies a dataset error is told with."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from poros.core.errors import PorosDataError

logger = logging.getLogger("poros.train.trainer")  # the trainer's own log, as before


# Synthetic-token range for the license-free systems benchmark. ``[100, 30000)``
# is in-vocabulary for every real LLM tokenizer (all have vocab >= 32k) and
# matches the resident/unsloth baselines, so all methods train on
# identically-distributed synthetic tokens. Token *content* does not affect
# VRAM / throughput / bitwise parity (those depend only on tensor shape and the
# block schedule), so synthetic measures the same systems metrics as real text
# while staying fully reproducible, zero-download, and free of any dataset
# license. This is the data the poros paper used for its headline numbers.
_SYNTHETIC_VOCAB_LOW = 100
_SYNTHETIC_VOCAB_HIGH = 30000
_SYNTHETIC_DEFAULT_SAMPLES = 512


def _synthetic_sample_count(
    spec: str, default: int = _SYNTHETIC_DEFAULT_SAMPLES
) -> int:
    """Parse the sample count from a ``"synthetic"`` / ``"synthetic:<n>"`` spec."""
    if ":" in spec:
        try:
            return max(1, int(spec.split(":", 1)[1]))
        except ValueError:
            return default
    return default


class _SyntheticTokenDataset(Dataset):
    """Deterministic random-token dataset for systems benchmarking.

    Token ids are drawn once from a seeded generator, so every machine sees
    identical inputs with no download or auth. Used for the parity / VRAM /
    throughput benchmark where token content is irrelevant to the measured
    quantity (it depends only on tensor shape and the block schedule).
    """

    def __init__(
        self, *, n_samples: int, seq_len: int, seed: int, low: int, high: int
    ) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        self._input_ids = torch.randint(
            low, high, (n_samples, seq_len), generator=generator
        )

    def __len__(self) -> int:
        return int(self._input_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self._input_ids[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def _build_synthetic_dataloader(
    *,
    n_samples: int,
    max_seq_length: int,
    batch_size: int,
    seed: int,
    low: int = _SYNTHETIC_VOCAB_LOW,
    high: int = _SYNTHETIC_VOCAB_HIGH,
    drop_last: bool = True,
) -> DataLoader:
    """Build a deterministic synthetic-token DataLoader (no download/auth).

    ``shuffle=False`` keeps iteration order fixed so the run is fully
    reproducible; the tokens are already random, so order adds nothing.
    """
    dataset = _SyntheticTokenDataset(
        n_samples=n_samples, seq_len=max_seq_length, seed=seed, low=low, high=high
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
    )


# Exception types that name the actual defect. Chain-walking prefers the
# SHALLOWEST of these over the deepest link: datasets' json builder chains
# ArrowInvalid("JSON parse error ... in row 1") -> a pandas fallback's
# ValueError("Trailing data"), and the deepest link is the misleading one.
_INFORMATIVE_ROOTS = (
    "UnicodeDecodeError", "ParserError", "EmptyDataError", "ArrowInvalid",
    "JSONDecodeError", "SchemaInferenceError", "CastError",
)


def _root_cause(exc: BaseException) -> BaseException:
    """Most informative link of the exception chain.

    HF wraps file problems in DatasetGenerationError, whose str() is the
    useless "An error occurred while generating the dataset" -- the actual
    answer (bad encoding, ragged row, empty file) is further down. Rules:
    follow ``__cause__`` (explicit chaining); follow ``__context__`` only
    when it was not suppressed (``raise X from None`` means the library
    deliberately hid it -- honoring that avoids surfacing an unrelated
    internal error). Prefer the first link whose type names the defect;
    otherwise the deepest link.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        if cur.__cause__ is not None:
            cur = cur.__cause__
        elif not cur.__suppress_context__:
            cur = cur.__context__
        else:
            cur = None
    for link in chain:
        if type(link).__name__ in _INFORMATIVE_ROOTS:
            return link
    return chain[-1]


def _utf8_error_location(path: Optional[str]) -> Optional[str]:
    """True file offset + line of the first invalid UTF-8 byte, or None.

    The UnicodeDecodeError raised by pandas/arrow carries an offset into the
    256 KiB block it was decoding, not into the file -- right for small
    files, silently wrong on the large ones people actually train on. Cost
    does not matter here: we are already in the error path.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read(256 * 1024 * 1024)  # bound the re-scan at 256 MB
        data.decode("utf-8")
    except UnicodeDecodeError as e:
        line = data.count(b"\n", 0, e.start) + 1
        return f"byte {e.start} (0x{data[e.start]:02x}), line {line}"
    except OSError:
        return None
    return None


def _dataset_remediation(root: BaseException, path: Optional[str]) -> str:
    """Remedy that matches the ACTUAL failure, not a generic format lecture."""
    name = type(root).__name__
    msg = str(root)
    low = msg.lower()
    if isinstance(root, UnicodeDecodeError):
        where = _utf8_error_location(path)
        loc = where or f"byte {root.start} of the failing block"
        return (
            f"The file is not valid UTF-8 (first bad byte: {loc}). "
            "Re-save it as UTF-8, e.g. "
            "iconv -f latin1 -t utf8 in.csv > out.csv"
        )
    if "ParserError" in name or ("Expected" in msg and ("fields" in msg or "columns" in msg)):
        at = re.search(r"line \d+", msg)  # the parser's pointer, and none of its words
        return (
            "A row does not match the header's column count"
            f"{f' ({at[0]})' if at else ''}. Quote fields containing commas or newlines, "
            "or drop the ragged rows."
        )
    if path and os.path.isfile(path) and os.path.getsize(path) == 0:
        return "The file is empty -- it has no rows to train on."
    if (
        isinstance(root, json.JSONDecodeError)
        or "json parse error" in low
        or msg.strip() in ("Trailing data", "Expected object or value")
    ):
        # Carry pyarrow's "in row N" pointer when it exists.
        row = ""
        m = re.search(r"in row (\d+)", msg)
        if m:
            row = f" (row {m.group(1)})"
        return (
            f"A line is not valid JSON{row}. Each .jsonl line "
            "must be ONE complete JSON object -- a pretty-printed, "
            "multi-line object must be collapsed to a single line."
        )
    return (
        "Rows must be one JSON object per line (.jsonl), a JSON array "
        "(.json), CSV with a header row, parquet, or plain text lines (.txt)."
    )


@contextlib.contextmanager
def _no_progress_bars() -> Iterator[None]:
    """While a dataset is read and tokenized: none of the datasets library's own bars
    ("Generating train split", "Map", "Filter"), which draw over a run's screen and fill a
    log. As they were once it is done."""
    import datasets

    was = datasets.is_progress_bar_enabled()
    datasets.disable_progress_bars()
    try:
        yield
    finally:
        if was:
            datasets.enable_progress_bars()


@contextlib.contextmanager
def _download_shown(name: str, say: Any) -> Iterator[Any]:
    """Yields what ``load_hub_dataset`` calls where a Hub dataset is not on this machine yet
    (None where ``say`` is None: a front end shows its own screen). It says one line through
    ``say`` and puts the datasets library's bars back on for the transfer: a large dataset is
    minutes of silence from a script otherwise. The bars go off again once it is here, so the
    read and the tokenizing after it stay quiet; a local file or a cached dataset never calls
    it."""
    import datasets

    shown = []

    def downloading() -> None:
        say(f"Downloading dataset {name} from the Hugging Face Hub (cached after this)")
        datasets.enable_progress_bars()
        shown.append(name)

    try:
        yield None if say is None else downloading
    finally:
        if shown:
            datasets.disable_progress_bars()


def _load_hf_dataset_as_dataloader(**how: Any) -> DataLoader:
    """Load a HuggingFace dataset, tokenize, and wrap in a DataLoader (``_load_and_tokenize``'s
    arguments), with the datasets library's progress bars off while it does, but for a Hub
    dataset's download (``say``, ``_download_shown``)."""
    with _no_progress_bars():
        return _load_and_tokenize(**how)


def _load_and_tokenize(
    *,
    dataset_name: str,
    model_name_or_path: str,
    max_seq_length: int,
    batch_size: int,
    seed: int,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    pad_to_longest: bool = True,
    for_eval: bool = False,
    say: Any = None,
) -> DataLoader:
    """Load a HuggingFace dataset, tokenize, and wrap in a DataLoader.

    Supports common instruction-tuning datasets like tatsu-lab/alpaca, plain
    ``text`` datasets, and chat/``messages`` datasets (applies the tokenizer's
    chat template when available). ``pad_to_longest``: each batch padded to its
    longest row; False pads every row to ``max_seq_length``. ``say`` (a line's printer):
    where a Hub dataset has to be downloaded first, that is said and its progress shown.
    """
    from datasets import load_dataset

    from poros.core.dataset_spec import (
        data_files,
        load_hub_dataset,
        resolve_dataset_spec,
    )
    from poros.core.template_guard import TemplateBound, template_bound
    from poros.core.tokenizer import load_tokenizer

    # Resolve the dataset spec BEFORE the tokenizer fetch so a bad spec fails
    # without any network work (the CLI preflights this too; API callers get
    # the same fail-fast here).
    kind, builder, data_path = resolve_dataset_spec(dataset_name)

    tokenizer = load_tokenizer(
        model_name_or_path, trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if kind == "local":
        # A file on disk (dataset: ./my_data.jsonl) loads without ceremony.
        try:
            dataset = load_dataset(builder, data_files=data_files(data_path), split="train")
        except PorosDataError:
            raise
        except Exception as exc:
            root = _root_cause(exc)
            detail = type(root).__name__
            if root is not exc and type(exc).__name__ != "DatasetGenerationError":
                detail += f"; via {type(exc).__name__}"
            raise PorosDataError(
                f"Could not read dataset file {dataset_name!r} ({detail})",
                remediation=_dataset_remediation(root, data_path),
                doc_url="docs/training.md#data",
            ) from exc
    else:
        try:
            with _download_shown(dataset_name, say) as downloading:
                dataset = (load_hub_dataset(dataset_name) if downloading is None
                           else load_hub_dataset(dataset_name, downloading=downloading))
        except PorosDataError:
            raise
        except Exception as exc:
            root = _root_cause(exc)
            raise PorosDataError(
                f"Could not load dataset {dataset_name!r} ({type(root).__name__})",
                remediation=(
                    "Point 'dataset:' at a local .jsonl/.json/.csv/.parquet/"
                    ".txt file, a Hugging Face dataset id, or 'synthetic'. If "
                    "the id is correct, this may be a network or auth issue -- "
                    "the original error is chained above."
                ),
                doc_url="docs/training.md#data",
            ) from exc

    def _format_chat(turns: Any) -> Optional[str]:
        """Render a chat/messages list into a single training string."""
        if not isinstance(turns, list) or not turns:
            return None
        # Normalise {role, content} or {from, value} turn dicts.
        norm = []
        for t in turns:
            if not isinstance(t, dict):
                return None
            role = t.get("role") or t.get("from") or "user"
            content = t.get("content") or t.get("value") or ""
            norm.append({"role": str(role), "content": str(content)})
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            try:
                with template_bound():
                    return tokenizer.apply_chat_template(norm, tokenize=False)
            except (TemplateBound, MemoryError):
                raise
            except Exception:
                pass
        return "\n".join(f"{t['role']}: {t['content']}" for t in norm)

    def _format_example(example: Dict[str, Any]) -> str:
        """Format a training example into a single string."""
        # Chat / messages format (ultrachat, OpenAssistant, ShareGPT, ...)
        for key in ("messages", "conversations", "conversation"):
            if key in example:
                rendered = _format_chat(example[key])
                if rendered is not None:
                    return rendered
        # Instruction-tuning format
        if "instruction" in example:
            parts = [example["instruction"]]
            if example.get("input"):
                parts.append(example["input"])
            if example.get("output"):
                parts.append(example["output"])
            return "\n".join(parts)
        if "text" in example:
            return example["text"]
        if "content" in example:
            return example["content"]
        # Fallback: concatenate all string values (warn -- likely wrong schema)
        logger.warning(
            "Dataset row has no recognized field (messages/instruction/text/"
            "content); falling back to concatenating string values. Provide a "
            "pre-tokenized Dataset/DataLoader for full control."
        )
        return " ".join(
            str(v) for v in example.values() if isinstance(v, str)
        )

    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = [_format_example(ex) for ex in _dict_to_list_of_dicts(examples)]
        if pad_to_longest:
            # No padding here: each batch is padded to its own longest row by
            # _PadToLongest below (as the SFT recipe's Batch does), so a short
            # row never pays for max_seq_length of compute.
            return tokenizer(texts, truncation=True, max_length=max_seq_length)
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        # Mask padding positions out of the loss. pad_token may equal
        # eos_token (e.g. Qwen2.5), so without this the loss is dominated
        # by trivially-predictable padding on short rows.
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    # Drop rows with fewer than two real tokens: the label shift consumes
    # one position, so a 1-token row leaves every target at -100 and
    # cross_entropy returns NaN forward / zero gradient — the row is wasted
    # and the logged step loss is poisoned.
    dataset = dataset.filter(lambda ex: sum(ex["attention_mask"]) >= 2)
    if len(dataset) < batch_size:
        raise PorosDataError(
            f"Dataset {dataset_name!r} has {len(dataset)} usable training "
            f"row(s) after tokenization (batch_size={batch_size}). Rows need "
            "a recognized text field and at least two tokens.",
            remediation=(
                "Check the file's rows: plain {'text': ...}, "
                "{'instruction'/'output': ...}, or chat {'messages': [...]} "
                "shapes are detected automatically."
            ),
            doc_url="docs/training.md#data",
        )
    dataset.set_format("torch")
    dataset = dataset.shuffle(seed=seed)

    # Seeded generator: batch order must not depend on the global-RNG
    # timeline, or resuming from a checkpoint (which restores mid-run RNG
    # state) would shuffle differently than the uninterrupted run and lose
    # the bitwise-resume guarantee.
    return DataLoader(  # held-out rows (for_eval): every one, in order, the last batch short
        dataset,
        batch_size=batch_size,
        shuffle=not for_eval,
        drop_last=not for_eval,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_PadToLongest(tokenizer.pad_token_id) if pad_to_longest else None,
    )


class _PadToLongest:
    """Right-pad a batch of token rows to its longest row.

    ``labels`` are ``input_ids`` with padding masked to -100: pad_token may
    equal eos_token (e.g. Qwen2.5), so without the mask the loss would be
    dominated by trivially-predictable padding on short rows.
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = int(pad_id)

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        width = max(len(r["input_ids"]) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            n = len(r["input_ids"])
            ids[i, :n] = torch.as_tensor(r["input_ids"], dtype=torch.long)
            mask[i, :n] = 1
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _dict_to_list_of_dicts(
    batch_dict: Dict[str, list],
) -> List[Dict[str, Any]]:
    """Convert {key: [values]} to [{key: value}, ...]."""
    keys = list(batch_dict.keys())
    if not keys:
        return []
    n = len(batch_dict[keys[0]])
    return [{k: batch_dict[k][i] for k in keys} for i in range(n)]


# The manifest fields that describe the ADAPTER (what export needs to
# rebuild the merge) rather than the RUN (machine paths, schedules, output
# dirs). trust_remote_code is deliberately absent: export ignores manifest
# authorization since the adapter must never authorize its own remote code.
