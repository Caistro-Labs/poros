"""One door to a tokenizer: the library is handed a private copy of a model's tokenizer files, built
from nothing, for every Poros load.

A tokenizer's config is its repo's words, and transformers hands them to the tokenizer as arguments.
``vocab`` as a string is a path: the file goes into the vocabulary, a run saves that as
tokenizer.json, and an export hands it on. ``init_inputs`` and special_tokens_map.json reach the
same arguments by other doors, and each list of what to refuse has had a way past it. So nothing is
refused for what it says: only fields known to be no path are copied, and a field no one listed
never reaches the library. Each file is read once, so what the library sees is what was looked at.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Optional

from poros.core.config import planted_base
from poros.core.errors import PorosConfigError

_PAPERS = ("tokenizer_config.json", "special_tokens_map.json")  # the library's arguments
_TABLE = "tokenizer.json"  # the library's arguments as well: a class is handed its model's vocab
_FILES = ("vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
          "sentencepiece.bpe.model", "added_tokens.json", "chat_template.jinja",
          "chat_template.json")
_CAP = 256 << 20  # the largest tokenizer.json in the catalog is about 30 MB
_QUIET = dict(_raise_exceptions_for_missing_entries=False, _raise_exceptions_for_gated_repo=False,
              _raise_exceptions_for_connection_errors=False)

_WORDS = {"chat_template", "tokenizer_class", "processor_class", "backend", "errors",
          "padding_side", "truncation_side", "pretokenize_regex"}
_DATA = {"added_tokens_decoder", "additional_special_tokens", "extra_special_tokens",
         "chat_template", "model_input_names", "response_schema", "response_template"}
_A_NAME = re.compile(r"[\w-]{1,64}")


class TokenizerRefused(PorosConfigError):
    """Fixed words: nothing from the model's own files is in the text."""

    def __init__(self, message: str) -> None:
        super().__init__(message, remediation="Use another copy of this model.")


def _kept(key: str, value: Any) -> bool:
    """Whether a paper's field reaches the library: a number, a token, or what is listed here."""
    if key == "init_inputs":  # positional: the first is the vocabulary
        return False
    if isinstance(value, (bool, int, float)) or value in (None, {}, []):
        return True
    if key.endswith("_token"):
        return isinstance(value, (str, dict))
    if key == "extra_special_tokens" and isinstance(value, dict):  # each key becomes an attribute
        return all(k.endswith("_token") for k in value)
    return key in _WORDS if isinstance(value, str) else key in _DATA


def read_once(found: str) -> bytes:
    """A file's bytes, read once, up to the cap, where its links lead. A pipe cannot hang this."""
    try:
        how = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(os.path.realpath(found), how), "rb") as f:
            seen = os.fstat(f.fileno())
            if stat.S_ISREG(seen.st_mode) and seen.st_size <= _CAP:
                return f.read(_CAP)
    except OSError:
        pass
    raise TokenizerRefused("A tokenizer file of this model cannot be read, and is not loaded.")


def _paper(raw: bytes) -> dict:
    try:
        said = json.loads(raw)
    except (ValueError, RecursionError):
        said = None
    if not isinstance(said, dict):
        raise TokenizerRefused("This model's tokenizer config cannot be read, and is not loaded.")
    return said


def _table(raw: bytes) -> bytes:
    """tokenizer.json, whose model's vocab and merges are tables. A class hands on what they are,
    and as a string, Rust opens it: the file was the vocabulary, saved with the run."""
    model = _paper(raw).get("model")
    if isinstance(model, dict) and not (isinstance(model.get("vocab"), (dict, list, type(None)))
                                        and isinstance(model.get("merges"), (list, type(None)))):
        raise TokenizerRefused("This model's tokenizer.json names a file for its vocabulary, and "
                               "it is not read.")
    return raw


def _stage(name: str, revision: Optional[str]) -> Path:
    """A private folder holding what the library may read of this model."""
    from transformers.utils import cached_file

    from poros.core.arch_registry import hub_token

    stage = Path(tempfile.mkdtemp(prefix="poros-tokenizer-"))
    atexit.register(shutil.rmtree, stage, True)

    def fetched(file: str) -> Optional[bytes]:
        try:
            found = cached_file(name, file, revision=revision, token=hub_token(), **_QUIET)
        except OSError:  # one that cannot be had: if nothing can, the load says why
            return None
        return read_once(found) if found else None

    for file in _FILES:
        if (raw := fetched(file)) is not None:
            (stage / file).write_bytes(raw)
    if (raw := fetched(_TABLE)) is not None:
        (stage / _TABLE).write_bytes(_table(raw))
    for paper in _PAPERS:
        if (raw := fetched(paper)) is not None:
            kept = {k: v for k, v in _paper(raw).items() if _kept(k, v)}
            (stage / paper).write_text(json.dumps(kept, ensure_ascii=False))
    if (raw := fetched("config.json")) is not None:  # which class a model_type forces
        kept = {k: v for k, v in _paper(raw).items() if k in ("model_type", "tokenizer_class")
                and isinstance(v, str) and _A_NAME.fullmatch(v)}
        # With no version, the library takes a local folder for an old Mistral and warns of it.
        (stage / "config.json").write_text(json.dumps({**kept, "transformers_version": "5.0.0"}))
    return stage


def load_tokenizer(name: Any, revision: Optional[str] = None,
                   trust_remote_code: bool = False) -> Any:
    """``AutoTokenizer.from_pretrained``, on the private copy. A person's --trust-remote-code runs
    the repo's own code, which can read anything anyway."""
    from transformers import AutoTokenizer
    from transformers.utils import cached_file

    from poros.core.arch_registry import hub_token

    name = str(name)
    if planted_base(name):
        raise TokenizerRefused("A folder here has this model's Hub name, and its tokenizer would "
                               "load in place of the Hub's.")
    if trust_remote_code:
        return AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    stage = _stage(name, revision)
    if not any(stage.iterdir()):
        cached_file(name, _PAPERS[0], revision=revision, token=hub_token())  # why not, in its words
    tokenizer = AutoTokenizer.from_pretrained(str(stage))
    tokenizer.name_or_path = name
    tokenizer.init_kwargs["is_local"] = os.path.isdir(name)  # it saves this: no trace of the copy
    return tokenizer
