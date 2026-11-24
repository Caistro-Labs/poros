"""Run words: `7f3a` (a unique id prefix), `last` (the newest run), or a run's name."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class RunMiss:
    said: str
    reason: str
    candidates: list[str] = field(default_factory=list)  # ids the word could mean


def resolve(
    word: Optional[str], ids: Sequence[str], *, newest: Optional[str] = None,
    names: Sequence[tuple[str, str]] = (),
) -> str | RunMiss:
    """``ids`` are every run's id; ``newest`` the most recent one; ``names``
    (name, id) newest first, so a reused name means its newest run."""
    if word is None or word.lower() in ("last", "latest", "newest"):
        if newest is None:
            return RunMiss(word or "", "there are no runs yet")
        return newest
    if word in ids:
        return word
    named = next((i for n, i in names if n.lower() == word.lower()), None)
    if named is not None:
        return named
    hits = [i for i in ids if i.lower().startswith(word.lower())]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return RunMiss(word, f"{word} matches {len(hits)} runs", hits)
    from poros.runs import miss

    return RunMiss(word, miss(word))
