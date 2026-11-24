"""Recipes: a whole training setup, named, by someone. The model, the data
it expects, the loop and every setting it changes; picking one opens the
train card filled in, ready to start on your data.

Poros ships a few, each certified by Caistro Labs: run at this release on a
named machine, with what that run measured (``poros.recipes.certified``).
Yours live in ``~/.poros/recipes/*.toml``; a project's (shared by whoever
wrote them) in ``./recipes/*.toml``. A recipe file, read
only through ``poros.recipes.rules``:

    name = "support-answers"
    model = "Qwen/Qwen2.5-7B-Instruct"
    data = "qa"            # chats, qa, pairs, text, or any
    loop = "sft"           # sft or dpo
    [settings]             # the training settings rules.SETTINGS names
    max_seq_length = 512
    lora_rank = 16
    learning_rate = 2e-4
    [measured]             # what a run reported, from `poros recipes save`
    s_per_step = 5.74
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import poros.recipes.rules as rules

DATA = rules.DATA
# Which file shapes (poros.cli.verbs.data) each ``data`` trains on.
TAKES = {"chats": {"chats", "convertible"}, "qa": {"convertible", "chats"},
         "pairs": {"pairs"}, "text": {"text"},
         "any": {"chats", "convertible", "pairs", "text"}}
# Settings with their own train flag (the rest go as key=value).
FLAGS = {"max_seq_length": "--seq-len", "lora_rank": "--rank", "block_size": "--block-size",
         "max_steps": "--steps", "store": "--store"}


@dataclass
class Recipe:
    key: str  # the file's stem, or the built-in's name: what `--recipe-file` would name
    name: str
    by: str
    about: str
    model: str
    data: str = "any"
    loop: str = "sft"
    settings: Dict[str, Any] = field(default_factory=dict)
    source: str = "poros"  # poros, yours, or the project's path
    receipt: str = ""  # the measured run it stands on, if any
    measured: Dict[str, Any] = field(default_factory=dict)  # what a run reported
    unvalidated: str = ""  # a family Poros has not validated: its model_type
    use: str = ""  # its tab in /recipes (rules.USES); "" for a file that names none
    updated: Optional[date] = None  # a shared one: when its author last changed it
    # its tab in /recipes where that is not its use: a Mac built-in (use "mac", which only says
    # whose it is) is listed where its CUDA counterpart was at 5d75e31a. Built-ins only: no file
    # sets it
    shelf: str = ""

    def tab(self) -> str:
        """Its tab: the use it names, else what it is. A run measured on a Mac is a Mac recipe; a
        model of 30B or more, a big one; else by the data it takes."""
        if self.shelf:
            return self.shelf
        if self.use:
            return self.use
        if str(self.measured.get("machine", "")).startswith("apple-"):
            return "mac"
        if (params(self.model) or 0) >= 30:
            return "big"
        return {"pairs": "prefer", "qa": "support", "chats": "writing", "text": "writing"}.get(
            self.data, "start")

    def title(self) -> str:
        """A name out of what it is, in no one's words: ``Qwen2.5-7B on chats: 256 tokens``."""
        return f"{self.model.rsplit('/', 1)[-1]} on {DATA[self.data]}: {self.summary()}"

    def fields(self) -> Dict[str, Any]:
        """What its file holds."""
        return {"name": self.name, "model": self.model, "data": self.data, "loop": self.loop,
                "unvalidated": self.unvalidated, "receipt": self.receipt,
                "settings": self.settings, "measured": self.measured}

    def says(self) -> str:
        """A recipe of a family Poros has not validated, in plain words; "" for any other."""
        if not self.unvalidated:
            return ""
        return (f"not validated: model_type {self.unvalidated}, no parity guarantee; its train "
                "asks for --unvalidated")

    def reported(self) -> str:
        """What a run reported, as a report: nothing here checks these numbers."""
        m = self.measured
        bits = [f"{m['s_per_step']:g} s/step" if "s_per_step" in m else "",
                f"peak {m['peak_gib']:g} GiB" if "peak_gib" in m else "",
                f"eval {m['eval_start']:g} to {m['eval_end']:g}" if "eval_end" in m else "",
                m.get("machine", "")]
        return " · ".join(filter(None, bits))

    def argv(self, data_file: Optional[str] = None, model: str = "") -> List[str]:
        """The train command it opens: model (``model``, its short name, if
        given), data, loop, its settings with flags, then the rest as
        ``key=value``; numbers as people write them (2e-4, 0)."""
        out = ["train", model or self.model]
        if data_file:
            out += ["--data", data_file]
        if self.loop and self.data != "any":
            out += ["--recipe", self.loop]
        rest = []
        for k, v in self.settings.items():
            v = str(v).lower() if isinstance(v, bool) else _num(v) if isinstance(v, float) \
                else str(v)
            # on a Mac --store names a folder for the weights: the store goes as store=
            if k in FLAGS and not (k == "store" and sys.platform == "darwin"):
                out += [FLAGS[k], v]
            else:
                rest.append(f"{k}={v}")
        return out + rest

    def takes(self, shape: str) -> bool:
        return shape in TAKES.get(self.data, set())

    def summary(self) -> str:
        """``512 tokens · rank 16 · lr 2e-4 · 2 epochs``: what it changes."""
        s = self.settings
        bits = []
        if "max_seq_length" in s:
            bits.append(f"{s['max_seq_length']:,} tokens")
        if "max_steps" in s:
            bits.append(f"{s['max_steps']} steps")
        if "lora_rank" in s:
            bits.append(f"rank {s['lora_rank']}")
        if "learning_rate" in s:
            bits.append(f"lr {_num(s['learning_rate'])}")
        if "num_epochs" in s:
            n = s["num_epochs"]
            bits.append(f"{n} epoch{'s' * (n != 1)}")
        if "dpo_beta" in s:
            bits.append(f"beta {s['dpo_beta']:g}")
        if "block_size" in s:
            bits.append(f"block {s['block_size']}")
        if "lora_dropout" in s:
            bits.append(f"dropout {s['lora_dropout']:g}")
        rest = len(s) - len(bits)
        return " · ".join(bits + ([f"{rest} more"] if rest > 0 else []))


def params(model: str) -> Optional[float]:
    """Billions of parameters, from a model's name (``Qwen3-235B-A22B``: 235); None if it says
    none."""
    m = re.search(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)[bB](?![A-Za-z0-9])", model.rsplit("/", 1)[-1])
    return float(m.group(1)) if m else None


def _num(v: float) -> str:
    """2e-4, not 0.0002: how learning rates are written."""
    return f"{v:.0e}".replace("e-0", "e-") if 0 < abs(v) < 1e-3 else f"{v:g}"


# Poros's own. Each says every setting that decides whether it fits and how fast it runs, so the
# run it opens is the run Caistro Labs measured, on any platform: none is left to a loop's preset
# (block 4 and 8 rows a step on CUDA, block 1 and one row on a Mac). Where each was certified, and
# what it measured there, is poros.recipes.certified's, generated from the receipts.
# Eight times the updates takes a tenth of the rate: one row a step at 2e-4 is unstable on CUDA too.
SFT_LR = 2e-5 if sys.platform == "darwin" else 2e-4
_FIT = {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.0, "per_device_batch_size": 1}
# The path and the tables are left to the run, as the preflight decides them on each card (it
# moves a run that would not fit to another path, or its tables to host RAM). A recipe says one
# only where its certified run deliberately took another: the fast recipes their path, the 72B
# its tables in host RAM. A Mac has one loop, and its table settles where the weights stream
# from: neither is a Mac setting.
_CUDA = {**_FIT, "block_size": 4, "gradient_accumulation_steps": 8, "store": "ram"}
# The 72B keeps its embedding table and LM head in host RAM: that holds less on the GPU on any
# card, so saying it takes no fit away from a larger one (the path stays the run's to decide).
_TABLES_OUT = {"offload_embeddings_and_head": True}
# A MoE model streams per expert: it has no block, and its path refuses one it is told
# (poros.preflight.start.moe_settings_refusal), so a MoE recipe says none.
_CUDA_MOE = {k: v for k, v in _CUDA.items() if k != "block_size"}
_MAC = {**_FIT, "block_size": 1, "gradient_accumulation_steps": 1, "learning_rate": 2e-5}
_SFT, _MOE = {"learning_rate": 2e-4, "num_epochs": 1}, "Qwen/Qwen3-30B-A3B-Instruct-2507"
BY = "Caistro Labs"
# first-run says what it said at 5d75e31a and nothing more: its data's loop sets the rest (SFT's
# rate for chats, DPO's for pairs, the engine's own defaults for text or synthetic rows)
_FIRST = {"max_seq_length": 256, "max_steps": 50, "lora_rank": 16}
# Keys a release before this one shipped that are gone, and what took their place.
RENAMED = {"moe": "moe-30b"}
# Each built-in's twin on the other platform: what `show` and `train --recipe` point to where
# a key's own recipe is not for this machine (a CUDA one on a Mac, a Mac one off it).
MAC_TWIN = {"first-run": "mac-first-run", "support": "mac-support", "voice": "mac-voice",
            "prefer": "mac-prefer", "big-32b": "mac-32b", "big-72b": "mac-72b",
            "big-72b-fast": "mac-72b", "moe-30b": "mac-moe"}
CUDA_TWIN = {"mac-first-run": "first-run", "mac-support": "support", "mac-voice": "voice",
             "mac-prefer": "prefer", "mac-32b": "big-32b", "mac-72b": "big-72b",
             "mac-moe": "moe-30b", "mac-7b": "voice"}


def twin(key: str, on_mac: bool) -> Optional[str]:
    """The built-in for this machine that stands for ``key`` (a renamed key first), where
    ``key``'s own is the other platform's; None where it is this one's, or has none."""
    key = RENAMED.get(key.lower(), key.lower())
    other = DEFINED_BY_KEY.get(key)
    if other is None:
        return None
    if on_mac and other.use != "mac":
        return MAC_TWIN.get(key)
    if not on_mac and other.use == "mac":
        return CUDA_TWIN.get(key)
    return None
# Every recipe Caistro Labs defines; BUILT_IN is these less what the release dropped.
DEFINED = [
    # any data, as it took before: with no file of your own, --data synthetic trains it (the
    # engine's own loop); chats train SFT. What decides fit and speed is the same either way.
    Recipe("first-run", "First run", BY, "the whole loop in minutes, on a small model",
           "Qwen/Qwen2.5-0.5B-Instruct", "any", "sft", _FIRST, use="start"),
    Recipe("support", "Support answers", BY, "answers like your tickets or FAQ",
           "Qwen/Qwen2.5-7B-Instruct", "qa", "sft",
           {"max_seq_length": 512, **_CUDA, "learning_rate": 2e-4, "num_epochs": 2},
           use="support"),
    Recipe("voice", "Your voice", BY, "writes like your chats: tone, format, length",
           "Qwen/Qwen2.5-7B-Instruct", "chats", "sft",
           {"max_seq_length": 512, **_CUDA, "learning_rate": 2e-4, "num_epochs": 3},
           use="writing"),
    Recipe("prefer", "Better answers", BY, "prefers your chosen answer over the rejected",
           "Qwen/Qwen2.5-7B-Instruct", "pairs", "dpo",
           {"max_seq_length": 512, **_CUDA, "learning_rate": 5e-6, "num_epochs": 1,
            "dpo_beta": 0.1}, use="prefer"),
    Recipe("big-32b", "32B, streamed", BY, "a 32B model, streamed a block at a time",
           "Qwen/Qwen2.5-32B-Instruct", "chats", "sft", {"max_seq_length": 512, **_CUDA, **_SFT},
           use="big"),
    Recipe("big-72b", "72B, streamed", BY, "the 72B, streamed a block at a time",
           "Qwen/Qwen2.5-72B-Instruct", "chats", "sft",
           {"max_seq_length": 512, **_CUDA, **_SFT, **_TABLES_OUT}, use="big"),
    Recipe("big-72b-fast", "72B, fast path", BY, "the 72B on the fastest path: more VRAM",
           "Qwen/Qwen2.5-72B-Instruct", "chats", "sft",
           {"max_seq_length": 512, **_CUDA, **_SFT, "path": "fast", **_TABLES_OUT}, use="big"),
    Recipe("moe-30b", "MoE, experts in RAM", BY, "a 30B mixture of experts, its experts in RAM",
           _MOE, "chats", "sft", {"max_seq_length": 512, **_CUDA_MOE, **_SFT}, use="big"),
    # The GPU expert cache is pinned: unset, path fast grows it to the card, so no run's memory
    # would speak for another's. 10.5 GiB is at or just below the 10.943 GiB the fast path sized
    # it to in the RTX 4090's formal runs of this setup at 5d75e31a (SSD store, fast + 4G host
    # cache: "22.97 GiB this process may hold - 7.72 GiB reserved peak - 4.59 GiB margin", 19.75
    # GiB reserved median), which keeps that run's headroom and 0.44 GiB more.
    Recipe("moe-30b-ssd", "MoE, experts on SSD", BY,
           "the 30B mixture of experts streamed from an SSD, with a host cache",
           _MOE, "chats", "sft",
           {"max_seq_length": 512, **_CUDA_MOE, **_SFT, "store": "ssd", "path": "fast+cache",
            "moe_expert_cache_gib": 10.5}, use="big"),
    Recipe("moe-30b-8gb", "MoE on a small card", BY,
           "the 30B mixture of experts from an SSD, at rank 8 and short rows",
           _MOE, "chats", "sft",
           {"max_seq_length": 256, **_CUDA_MOE, "lora_rank": 8, "lora_alpha": 16, **_SFT,
            "store": "ssd"}, use="big"),
    Recipe("mac-72b", "72B on a Mac", BY, "the 72B on a Mac, a layer at a time",
           "Qwen/Qwen2.5-72B-Instruct", "chats", "sft",
           {"max_seq_length": 128, **_MAC, "num_epochs": 1},
           use="mac", shelf="big"),
    Recipe("mac-32b", "32B on a Mac", BY, "a 32B model on a Mac, a layer at a time",
           "Qwen/Qwen2.5-32B-Instruct", "chats", "sft",
           {"max_seq_length": 128, **_MAC, "num_epochs": 1},
           use="mac", shelf="big"),
    Recipe("mac-moe", "MoE on a Mac", BY, "a 30B mixture of experts on a Mac",
           "Qwen/Qwen3-30B-A3B-Instruct-2507", "chats", "sft",
           {"max_seq_length": 128, **_MAC, "num_epochs": 1}, use="mac", shelf="big"),
    Recipe("mac-7b", "7B on a Mac", BY, "a 7B model on a Mac, with rows held out to score",
           "Qwen/Qwen2.5-7B-Instruct", "chats", "sft",
           {"max_seq_length": 256, **_MAC, "num_epochs": 1, "eval_split": 0.05},
           use="mac", shelf="mac"),
    # What first-run, support, voice and prefer were on a Mac at 5d75e31a: the same words, which
    # a Mac's loop presets (one row a step at block 1, its SFT rate) fill as they did there.
    Recipe("mac-first-run", "First run on a Mac", BY, "the whole loop in minutes, on a Mac",
           "Qwen/Qwen2.5-0.5B-Instruct", "any", "sft", _FIRST, use="mac", shelf="start"),
    Recipe("mac-support", "Support answers on a Mac", BY, "answers like your tickets or FAQ",
           "Qwen/Qwen2.5-7B-Instruct", "qa", "sft",
           {"max_seq_length": 512, "lora_rank": 16, "lora_dropout": 0.0, "learning_rate": 2e-5,
            "num_epochs": 2}, use="mac", shelf="support"),
    Recipe("mac-voice", "Your voice on a Mac", BY, "writes like your chats: tone, format, length",
           "Qwen/Qwen2.5-7B-Instruct", "chats", "sft",
           {"max_seq_length": 512, "lora_rank": 16, "lora_dropout": 0.0, "learning_rate": 2e-5,
            "num_epochs": 3}, use="mac", shelf="writing"),
    Recipe("mac-prefer", "Better answers on a Mac", BY,
           "prefers your chosen answer over the rejected, on a Mac",
           "Qwen/Qwen2.5-7B-Instruct", "pairs", "dpo",
           {"max_seq_length": 512, "lora_rank": 16, "lora_dropout": 0.0, "learning_rate": 5e-6,
            "num_epochs": 1, "dpo_beta": 0.1}, use="mac", shelf="prefer"),
]


DEFINED_BY_KEY = {r.key: r for r in DEFINED}


def loop_preset(r: Recipe) -> Dict[str, Any]:
    """What its loop sets where a built-in says nothing, on the platform it is for, whatever
    machine reads it (poros.recipes.CUDA_MEASURED; a Mac one row a step at block 1)."""
    from poros.recipes import CUDA_MEASURED

    out = dict(CUDA_MEASURED)
    if r.use == "mac":
        out.update(block_size=1, gradient_accumulation_steps=1)
    return out


def pointer(key: str, on_mac: bool) -> Optional[Tuple[str, str, str]]:
    """(why, the twin's key, its train command) where ``key`` names the other platform's
    built-in (or a renamed one's), for a refusal to say; None where it is this machine's."""
    import shlex

    to = twin(key, on_mac)
    if to is None:
        return None
    here, there = ("a Mac", "CUDA cards") if on_mac else ("this machine", "a Mac")
    why = f"{key.lower()} is a recipe for {there}; on {here} use {to}"
    return why, to, shlex.join(["poros", *DEFINED_BY_KEY[to].argv("<data>")])


def built_in() -> List[Recipe]:
    """The built-ins this release ships: every one defined, less what its generator dropped for
    having no certified machine (poros.recipes.certified, ``dropped``)."""
    from poros.recipes import certified

    gone = set(certified.book().dropped)
    return [r for r in DEFINED if r.key not in gone]


def __getattr__(name: str) -> Any:
    if name == "BUILT_IN":  # read when asked: the certified file is not read at import
        return built_in()
    raise AttributeError(name)


def _home() -> Path:
    return Path(os.environ.get("POROS_HOME") or Path.home() / ".poros")


def _read(path: Path, source: str, by: str) -> Recipe:
    """A recipe file through the one parser, or ValueError saying why it is refused."""
    if not rules.WORD.fullmatch(path.stem):
        raise ValueError("its file name is not a recipe's")
    if rules.CLAIMS.search(path.stem):
        raise ValueError("its file name claims a Caistro Labs certification, which only "
                         "Poros's own recipes carry")
    return from_fields(path.stem, rules.parse(rules.read_file(path)), source, by)


def from_fields(key: str, raw: Dict[str, Any], source: str, by: str) -> Recipe:
    """A recipe from fields ``rules.parse`` passed. Only a shared one (``by`` None) says who
    wrote it: a file in a folder here is yours or the project's, whatever it claims."""
    r = Recipe(key=key, name="", by=by or raw.get("by") or "someone", about=raw.get("about", ""),
               model=raw["model"], data=raw.get("data", "any"), loop=raw.get("loop", "sft"),
               settings=raw["settings"], source=source, receipt=raw.get("receipt", ""),
               measured=raw["measured"], unvalidated=raw.get("unvalidated", ""),
               use=raw.get("use", ""), updated=raw.get("updated"))
    r.name = raw.get("title") or raw.get("name") or r.title()
    return r


MOST_FILES = 64  # of a folder's recipes, read and listed


def _files(cwd: Optional[Path]) -> Iterator[Tuple[Optional[Path], str, str]]:
    for folder, source, by in (((_home() / "recipes"), "yours", "you"),
                               ((cwd or Path.cwd()) / "recipes", "recipes", "someone else")):
        if not folder.is_symlink():  # a link is someone else's folder
            found = sorted(islice(folder.glob("*.toml"), 100 * MOST_FILES))
            for p in found[:MOST_FILES]:
                yield p, source if by == "you" else f"{source}/{p.name}", by
            if len(found) > MOST_FILES:  # 6,000 files made a result of 1.8 MB
                yield None, "recipes" if by != "you" else source, by


def is_built_in(key: str) -> bool:
    """A name Poros defines: one a file here may not take, dropped from the release or not."""
    return key.lower() in {r.key for r in DEFINED}


def local(cwd: Optional[Path] = None) -> Tuple[List[Recipe], List[Tuple[str, str]]]:
    """Yours, then this project's; and each file refused, with why. A name fit to show is shown."""
    kept, refused = [], []
    for path, source, by in _files(cwd):
        if path is None:
            refused.append((source, f"only the first {MOST_FILES} files here were read"))
            continue
        try:
            if is_built_in(path.stem):  # Poros's own name means Poros's own recipe, whoever wrote
                raise ValueError("it takes the name of one of Poros's: rename the file")
            kept.append(_read(path, source, by))
        except ValueError as exc:
            shown = path.name if rules.WORD.fullmatch(path.stem) else "a file"
            refused.append((shown if by == "you" else f"recipes/{shown}", str(exc)))
    return kept, refused


def all_recipes(cwd: Optional[Path] = None) -> List[Recipe]:
    """Yours first, then this project's, then the shared ones last synced, then Poros's."""
    from poros.recipes import hub

    return local(cwd)[0] + hub.cached()[0] + built_in()


def yours() -> Path:
    """The folder your recipes go in. ValueError if it is a link: what is written stays here."""
    folder = _home() / "recipes"
    if folder.is_symlink():
        raise ValueError("your recipes folder is a link")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save(r: Recipe) -> Path:
    """Write ``r`` to ``~/.poros/recipes/``; the path written. ValueError if the file would not
    read back through the parser."""
    return rules.write_new(yours(), r.key, rules.dumps(r.fields()))
