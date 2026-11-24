"""What a recipe file may hold. A file can come from anywhere, so it is data and never code, and
whoever wrote it may be trying to steer whatever reads our output.

``read_file`` and ``parse`` are the one way a recipe is read: a plain file through no link, exact
fields, numbers within bounds, and no paths, env, flags or code. A refusal names a field of ours
and never repeats what it found. ``from_run`` is the one way a run becomes a recipe: it builds the
fields up from nothing, each because a line here names it, so nothing else in a run's config or
receipt can reach the file. ``write_new`` goes through no link and over no file. A recipe's numbers
are what someone reported, and Poros has not checked them.

A shared recipe (the reviewed repo ``poros.recipes.hub`` syncs, never a file in a folder here) may
also carry words for people -- its title, what it does, who wrote it, when -- in plain letters
and marks only: no control, escape, bidi or zero-width character can pass. The /recipes card
shows them; nothing made for an agent (``--json``) carries them.
"""

from __future__ import annotations

import os
import re
import stat
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Mapping

from poros.core.arch_registry import TYPE_WORD

CAP_BYTES = 4096
# A file's stem, a run's id, a saved recipe's name: what may be shown of anything from outside.
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# What a recipe's ``data`` takes, in a person's words.
DATA = {"chats": "chats", "qa": "question/answer rows", "pairs": "preference pairs",
        "text": "plain text", "any": "any data"}
# A Hub id, and nothing a path could be.
_HUB_ID = re.compile(r"(?!.*\.\.)[A-Za-z0-9][\w.-]{0,63}/[A-Za-z0-9][\w.-]{0,63}", re.ASCII)
_APPLE = re.compile(r"apple-m\d{1,2}(-(pro|max|ultra))?-\d{1,3}gpu-\d{1,4}gb")


def _machine_rule(value: Any) -> bool:
    from poros.core.card import GPU_SPECS_MIB

    return isinstance(value, str) and (bool(_APPLE.fullmatch(value)) or value in GPU_SPECS_MIB)


# Words a recipe file may not use of itself: a certification is Poros's own recipes' alone
# (poros.recipes.certified), and none of anyone's words may look like one.
CLAIMS = re.compile(r"certif|caistro", re.IGNORECASE)
# What a recipe is for: the tabs of /recipes. ``big`` and ``mac``: what "recipe" most often means
# in the field, the exact settings a large model or the Mac path needs.
USES = ("big", "mac", "start", "support", "writing", "code", "extract", "prefer")
# A shared recipe's words for people: letters of any language, digits, spaces and plain marks.
# No control, escape, bidi or zero-width characters (\w holds none), so it can't steer a terminal.
_WORDS = r"[\w][\w ,.;:!?()'&/+#%·–—-]"
# A GitHub login, which is who a shared recipe says wrote it.
_LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}")


def _date_rule(value: Any) -> bool:
    from datetime import date, datetime

    return isinstance(value, date) and not isinstance(value, datetime) and value.year >= 2024


# Each field's rule: a pattern, the words it may be, its numeric range, or a check of its own.
# `name` is for a file saved here: `from_run` does not fill it.
# `unvalidated`: the model_type of a family Poros has not validated, from a run that trained it
# with --unvalidated. A label only: train reads the model's own config, and asks for the flag.
_TOP = {
    "name": re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}"),
    "model": _HUB_ID,
    "data": tuple(DATA),
    "loop": ("sft", "dpo", "ipo", "kto", "orpo", "simpo"),
    "unvalidated": TYPE_WORD,
    "receipt": re.compile(r"[a-z0-9.-]{1,80}"),
    "use": USES,
    # a shared recipe's card (SHARED_ONLY): what it's called, what it does, who wrote it, when
    "title": re.compile(_WORDS + r"{0,47}"),
    "about": re.compile(_WORDS + r"{0,119}"),
    "by": _LOGIN,
    "updated": _date_rule,
}
SHARED_ONLY = ("title", "about", "by", "updated")
SETTINGS = {
    "max_seq_length": (1, 1 << 20),
    "max_steps": (1, 10**8),
    "num_epochs": (1, 1000),
    "lora_rank": (1, 1024),
    "lora_alpha": (1, 4096),
    "lora_dropout": (0.0, 0.99),
    "learning_rate": (1e-9, 1.0),
    "weight_decay": (0.0, 10.0),
    "max_grad_norm": (0.0, 1e4),
    "warmup_steps": (0, 10**8),
    "lr_scheduler": ("constant", "cosine", "linear"),
    "block_size": (1, 1024),
    "per_device_batch_size": (1, 1024),
    "gradient_accumulation_steps": (1, 1024),
    # how a CUDA run trains a step, and where a MoE model's experts live: words only, never a
    # path or a plugin's name (``fast+cache`` is train's own word: path fast with the SSD
    # store's host cache)
    "path": ("efficient", "fast", "sequential", "fast+cache"),
    "store": ("ram", "ssd"),
    # the embedding table and LM head in host RAM (a CUDA run's), said as true or false
    "offload_embeddings_and_head": lambda v: isinstance(v, bool),
    # a MoE model's device expert cache, GiB: a set one is the same size on every card
    "moe_expert_cache_gib": (0.0, 1024.0),
    "train_on": ("answers", "last", "all"),
    "dpo_beta": (1e-6, 1e3),
    "eval_split": (0.0, 0.49),
    "eval_rows": (0, 10**6),
    "save_steps": (0, 10**8),
    "eval_steps": (0, 10**8),
    "seed": (0, 2**32 - 1),
}
_MEASURED = {
    "poros": re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,4}((a|b|rc|\.dev)\d{1,6})?"),
    "machine": _machine_rule,
    "steps": (1, 10**9),
    "s_per_step": (0.0, 1e7),
    "peak_gib": (0.0, 1e5),
    "eval_start": (-1e6, 1e6),
    "eval_end": (-1e6, 1e6),
    "checked": (0, 10**13),  # numbers a run compared with their first reading
    "mismatched": (0, 10**13),
}


def _fits(rule: Any, value: Any) -> bool:
    if callable(rule):
        return rule(value)
    if isinstance(rule, re.Pattern):
        return isinstance(value, str) and rule.fullmatch(value) is not None
    if isinstance(rule[0], str):
        return isinstance(value, str) and value in rule
    low, high = rule
    return type(value) in ((int,) if type(low) is int else (int, float)) and low <= value <= high


def within(key: str, value: Any) -> bool:
    """A setting's value, within what a recipe may hold; text may say a number."""
    rule = SETTINGS[key]
    if isinstance(value, str) and not isinstance(rule[0], str):
        try:
            value = int(value) if type(rule[0]) is int else float(value)
        except ValueError:
            return False
    return _fits(rule, value)


def _check(table: Any, rules: Mapping[str, Any], what: str) -> Dict[str, Any]:
    if not isinstance(table, dict):
        raise ValueError(f"{what} is not a table")
    for key, value in table.items():
        if key not in rules:  # a field's name is outside text too
            raise ValueError(f"{what} holds a field no recipe may")
        if not _fits(rules[key], value):
            raise ValueError(f"{key} in {what} is outside what a recipe may hold")
    return table


def read_file(path: Path) -> str:
    """A file's text: plain, with one name, through no link, at most the cap's worth. A pipe cannot
    hang it."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("it could not be read, or is a link") from exc
    try:
        seen = os.fstat(fd)
        if not stat.S_ISREG(seen.st_mode) or seen.st_nlink != 1:
            raise ValueError("it is not a plain file")
        data = os.read(fd, CAP_BYTES + 1)
    finally:
        os.close(fd)
    try:
        return data.decode()
    except UnicodeDecodeError as exc:
        raise ValueError("it is not text") from exc


def write_new(folder: Path, key: str, text: str) -> Path:
    """``text`` as ``<key>.toml`` in ``folder``, or under the next name free: through no link and
    over no file. The same text stays where it is."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for n in range(1, 100):
        path = folder / (f"{key}.toml" if n == 1 else f"{key}-{n}.toml")
        try:
            fd = os.open(path, flags, 0o644)
        except FileExistsError:
            try:
                if read_file(path) == text:
                    return path
            except ValueError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return path
    raise ValueError("no name is free for it")


def next_for(run_id: str) -> List[str]:
    """What a finished run offers, built from its id alone, and with no consent in it."""
    return [f"poros recipes save {run_id}"] if WORD.fullmatch(run_id) else []


def parse(text: str, shared: bool = False) -> Dict[str, Any]:
    """The fields of a recipe file, or ValueError saying what is wrong and none of what it held.
    ``shared``: a file from the shared repo, which alone may hold words for people."""
    if len(text) > CAP_BYTES:
        raise ValueError(f"it is over {CAP_BYTES} bytes")
    try:
        fields = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError) as exc:  # brackets nested to the cap
        raise ValueError("it is not TOML") from exc
    settings, measured = fields.pop("settings", {}), fields.pop("measured", {})
    _check(fields, _TOP if shared else {k: v for k, v in _TOP.items() if k not in SHARED_ONLY},
           "a recipe")
    if "model" not in fields:
        raise ValueError("it names no model")
    for key in ("name", "title", "about", "by"):  # only Poros's own carry a certification
        if CLAIMS.search(str(fields.get(key, ""))):
            raise ValueError(f"its {key} claims a Caistro Labs certification, which only "
                             "Poros's own recipes carry")
    return {
        **fields,
        "settings": _check(settings, SETTINGS, "settings"),
        "measured": _check(measured, _MEASURED, "measured"),
    }


def dumps(fields: Mapping[str, Any], shared: bool = False) -> str:
    """``fields`` as a recipe file, which must read back through ``parse`` (as a shared one's,
    ``shared``)."""
    def line(key: str, value: Any) -> str:
        if isinstance(value, str):
            return f'{key} = "{value}"'
        if hasattr(value, "isoformat"):  # a date: TOML's own, unquoted
            return f"{key} = {value.isoformat()}"
        if isinstance(value, bool):  # TOML's own words
            return f"{key} = {str(value).lower()}"
        return f"{key} = {value!r}"

    out = [line(k, fields[k]) for k in _TOP if fields.get(k)]
    for table in ("settings", "measured"):
        if fields.get(table):
            out += ["", f"[{table}]", *(line(k, v) for k, v in fields[table].items())]
    text = "\n".join(out) + "\n"
    parse(text, shared)
    return text


def _number(value: Any) -> Any:
    """``learning_rate=1e-4`` arrives as words: a number, if it is one."""
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def pick(rules: Mapping[str, Any], said: Mapping[str, Any]) -> Dict[str, Any]:
    """What ``said`` holds of the fields ``rules`` names, in their order, that pass."""
    picked = ((k, _number(said.get(k))) for k in rules)
    return {k: v for k, v in picked if _fits(rules[k], v)}


# What a finished run's recipe keeps: the settings a recipe held before path, store and batch
# could be said in one, so ``poros recipes save`` writes what it wrote before.
_RUN_SETTINGS = {k: v for k, v in SETTINGS.items()
                 if k not in ("per_device_batch_size", "path", "store",
                              "offload_embeddings_and_head", "moe_expert_cache_gib")}


def _machine_of(receipt: Mapping[str, Any]) -> Any:
    """The class of machine, from closed lists and never the text as it came."""
    from poros.core.card import GPU_SPECS_MIB

    if _fits(_APPLE, receipt.get("device_class")):
        return receipt["device_class"]
    named = str((receipt.get("gpu") or {}).get("name") or "")
    return max((g for g in GPU_SPECS_MIB if named.endswith(g)), key=len, default=None)


def from_run(model: str, told: Mapping[str, Any], receipt: Mapping[str, Any],
             held_out: List[float], shape: str) -> Dict[str, Any]:
    """A finished run's recipe: its model, data shape, loop, allowed settings and measured numbers.
    ValueError, in one line, for a run a recipe cannot stand for."""
    if not _fits(_HUB_ID, model) or os.path.exists(os.path.expanduser(model)):
        raise ValueError("It trained a local model, and a recipe needs a public Hub model.")
    if told.get("init_adapter"):
        raise ValueError("It started from another run, and a recipe starts from the model.")
    loop = {"SFT": "sft", "DPO": "dpo", "IPO": "ipo", "KTO": "kto", "ORPO": "orpo",
            "SimPO": "simpo", "classic": "sft"}.get(receipt.get("recipe"))
    if loop is None:
        raise ValueError("It trained with a loop file of yours, and a recipe uses a built-in loop.")
    data = {"convertible": "qa", "chats": "chats", "pairs": "pairs", "text": "text"}.get(shape)
    config = receipt.get("config") or {}
    fields: Dict[str, Any] = {
        "model": model,
        "data": data or ("any" if loop == "sft" else "pairs"),
        "loop": loop,
        "settings": pick(_RUN_SETTINGS, {**config, "block_size": None, **told}),
    }
    if receipt.get("support") == "user_asserted":  # it trained with --unvalidated: it says so
        kind = receipt.get("model_type")
        fields["unvalidated"] = kind if _fits(TYPE_WORD, kind) else "unknown"
    if not receipt.get("simulated"):  # a preview measured nothing
        witness = receipt.get("witness") or {}
        peak = receipt.get("peak_vram_allocated_gb")  # decimal GB, in a receipt
        fields["measured"] = pick(_MEASURED, {
            "poros": receipt.get("poros"),
            "machine": _machine_of(receipt),
            "steps": receipt.get("steps"),
            "s_per_step": receipt.get("s_per_step"),
            "peak_gib": round(peak * 1e9 / 2**30, 3) if isinstance(peak, (int, float)) else None,
            "eval_start": held_out[0] if held_out else None,
            "eval_end": held_out[-1] if len(held_out) > 1 else None,
            "checked": witness.get("checked"),
            "mismatched": witness.get("mismatches"),
        })
    return fields
