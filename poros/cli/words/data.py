"""Data words: `on notes.jsonl`, `on ./data/`, `on OpenAssistant/oasst1`, `on synthetic`.

A word that looks local is a path, and a path that does not exist is an
error naming a local file only where the name is a near miss of one (a typo, another
suffix): it never becomes a Hub lookup, and never an offer of whatever else is here.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

DATA_SUFFIXES = (".jsonl", ".json", ".csv")


@dataclass
class Data:
    kind: Literal["synthetic", "file", "dir", "hub"]
    value: str  # what PorosConfig.dataset gets
    said: str


@dataclass
class DataMiss:
    said: str
    reason: str
    near: list[str] = field(default_factory=list)


def _local(text: str, cwd: Path) -> bool:
    """Meant as a path: it says so, has a data suffix, or starts in a folder here."""
    first = Path(text).parts[0] if Path(text).parts else ""
    return (
        text.startswith(("./", "../", "/", "~"))
        or Path(text).suffix.lower() in DATA_SUFFIXES
        or (cwd / first).is_dir()
    )


def _stem(name: str) -> str:
    """A name as two are compared: no data suffix (every file here ends alike), no case."""
    name = name.rstrip("/")
    return (name[: -len(Path(name).suffix)] if Path(name).suffix.lower() in DATA_SUFFIXES
            else name).lower()


def near_misses(asked: str, names: list[str]) -> list[str]:
    """The names ``asked`` is a near miss of, closest first: the same name with another suffix
    or case, or one a typo away. `nope.jsonl` is no near miss of `notes.jsonl`."""
    stems: dict[str, list[str]] = {}
    for name in names:
        stems.setdefault(_stem(name), []).append(name)
    close = difflib.get_close_matches(_stem(asked), list(stems), n=3, cutoff=0.8)
    return [name for stem in close for name in sorted(stems[stem])][:3]


def resolve(text: str, cwd: Optional[Path] = None) -> Data | DataMiss:
    cwd = cwd or Path.cwd()
    if text.lower() == "synthetic":
        return Data("synthetic", "synthetic", text)
    path = Path(text).expanduser()
    full = path if path.is_absolute() else cwd / path
    if full.is_file():
        return Data("file", text, text)
    if full.is_dir():
        has_data = any(p.suffix.lower() in DATA_SUFFIXES for p in full.iterdir())
        if not has_data:
            return DataMiss(text, f"{text} has no .jsonl, .json or .csv files")
        return Data("dir", text, text)
    if not _local(text, cwd) and re.fullmatch(r"[\w.\-]+/[\w.\-]+", text):
        return Data("hub", text, text)
    parent = full.parent if full.parent.is_dir() else cwd
    names = (
        [
            p.name + ("/" if p.is_dir() else "")
            for p in parent.iterdir()
            if p.is_dir() or p.suffix.lower() in DATA_SUFFIXES
        ]
        if parent.is_dir()
        else []
    )
    near = near_misses(full.name, names)
    prefix = "" if parent == cwd else str(Path(text).parent) + "/"
    return DataMiss(text, f"no file or folder named {text}", [prefix + n for n in near])
