"""Model words into catalog entries: `qwen 7b`, `72b`, `deepseek lite`.

    resolve(["qwen", "7b"]) -> Match(entries=[Qwen2.5-7B, Qwen2.5-7B-Instruct])

An exact Hub id or a path that looks local is taken as given. Anything else
is matched against poros/catalog.toml on tokens: every word must name the
entry (its family, size, variant, entry kind, an alias or a piece of its
id). Being cached only ranks; it never adds a model the catalog lacks.
Pure apart from reading the packaged catalog. Section 3 of
the CLI interface design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Optional

from poros import catalog as catalog_file

Evidence = Literal["validated", "measured", "same-shape", "none"]


@dataclass(frozen=True)
class MetalRow:
    """One row of the Mac table: a model at one length on one device class."""

    device_class: str
    seq: int
    tier: Literal["ram", "ssd"]
    grad_accum: int
    save_steps: int
    synthetic_steps: int
    evidence: str  # validated | checked | measured | none | derived
    default: bool = False
    method: str = ""
    receipt: str = ""
    macos_build: str = ""
    # Not in the catalog: read from a model's config by poros.metal.table, never from a run.
    derived: bool = False
    # Not in the catalog: planned by poros.metal.table for this Mac's memory from this class's
    # rows, its evidence "derived". No run on this Mac measured it.
    planned_from: str = ""


@dataclass(frozen=True)
class Entry:
    id: str
    arch: str
    family: str
    size: str
    variant: str
    entry: Literal["dense", "moe"]
    evidence: Evidence
    receipt: str = ""
    via: str = ""
    gated: bool = False
    aliases: tuple[str, ...] = ()
    metal: tuple[MetalRow, ...] = ()

    @property
    def label(self) -> str:
        """``Qwen2.5 7B Instruct``: the id's name, spaced for reading."""
        name = self.id.split("/", 1)[-1]
        return re.sub(r"[-_]", " ", name)

    @property
    def short(self) -> str:
        return self.id.split("/", 1)[-1]

    def words(self) -> frozenset[str]:
        """Every word that names this entry.

        The family, size, variant and entry kind come from their fields. A
        piece of the id counts only when it is not another family's word
        ("qwen" in DeepSeek-R1-Distill-Qwen) or a variant word ("instruct"
        in Qwen3-Coder-...-Instruct): those would name the wrong model.
        """
        root = self.family.rstrip("0123456789.")
        out = {self.family, root, self.size, self.variant, self.entry}
        out |= {self.family[len(root) :]} if self.family[len(root) :] else set()  # qwen2.5 -> 2.5
        out |= {
            p
            for p in _pieces(self.id)
            if p not in _FAMILY_ROOTS - {root} and p not in _VARIANT_WORDS
        }
        for alias in self.aliases:
            out |= set(alias.lower().split())
        out |= _VARIANT_ALIASES.get(self.variant, set())
        return frozenset(w for w in out if w)


_VARIANT_ALIASES = {"instruct": {"it", "chat"}, "base": {"pt", "pretrained"}}
_VARIANT_WORDS = frozenset({"instruct", "base", "coder", "it", "pt", "chat"})


def _pieces(model_id: str) -> Iterable[str]:
    name = model_id.lower()
    yield from re.split(r"[/\-_ ]+", name)
    yield name
    yield name.split("/", 1)[-1]


@lru_cache(maxsize=1)
def catalog() -> tuple[Entry, ...]:
    known = set(Entry.__dataclass_fields__)
    return tuple(Entry(**{**{k: v for k, v in m.items() if k in known},
                          "aliases": tuple(m.get("aliases", ())),
                          "metal": tuple(MetalRow(**r) for r in m.get("metal", ()))})
                 for m in _raw()["model"])


_raw = catalog_file.raw  # the file, read once (poros.catalog)


@lru_cache(maxsize=1)
def on_mac() -> frozenset[str]:
    """The ids that train on a Mac: those with Mac rows (``[[model.metal]]``),
    and a same-shape model whose base has them."""
    rows = {m["id"] for m in _raw()["model"] if m.get("metal")}
    return frozenset(rows | {m["id"] for m in _raw()["model"] if m.get("via") in rows})


@lru_cache(maxsize=1)
def all_words() -> frozenset[str]:
    return frozenset(w for e in catalog() for w in e.words())


_FAMILY_ROOTS = frozenset({"qwen", "gemma", "deepseek", "olmoe", "llama", "mistral", "phi"})


def by_id(model_id: str) -> Optional[Entry]:
    low = model_id.lower()
    return next((e for e in catalog() if e.id.lower() == low), None)


def instruct_twin(model_id: str) -> Optional[Entry]:
    """The catalog's instruct model of a base model's own shape, when it lists one."""
    base = by_id(model_id)
    if base is None or base.variant != "base":
        return None
    return next((e for e in catalog() if e.variant == "instruct" and e.via == base.id), None)


def base_on_chats(model_id: str) -> str:
    """What to say when chats train a base model with an instruct twin; "" otherwise. Measured on
    a Mac: Qwen2.5-0.5B on chats, 80 and 284 steps, never ended an answer; its Instruct, 80 steps,
    did. Its <|im_end|> is a token the base never learned to say: `poros eval` reads it."""
    twin = instruct_twin(model_id)
    if twin is None:
        return ""
    return (f"{by_id(model_id).short} is a base model: trained on chats it may never learn to end "
            f"an answer, and then each reply runs to the token limit; {twin.short} ends them")


def by_name(name: str) -> Optional[Entry]:
    """A catalog entry by its Hub id or its short name (`Qwen2.5-7B-Instruct`),
    any case. Exact only: `qwen 7b` names nothing."""
    low = name.lower()
    return by_id(name) or next((e for e in catalog() if e.short.lower() == low), None)


def matching(said: str) -> list:
    """Catalog entries whose name holds every part of ``said``: ``72b``,
    ``qwen 7b``, ``coder``. A part is a word or a number (``2.5``, ``7b``)."""
    import re

    def parts(text: str) -> list[str]:
        return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())

    want = parts(said)
    if not want:
        return []
    out = []
    for e in catalog():
        have = parts(e.short)
        if all(w in have for w in want):
            out.append(e)
    return out


def closest(name: str, n: int = 3) -> list[str]:
    """Catalog names that share a family or a size with ``name``, nearest
    first; none when nothing does (``llama-8b`` names no Qwen)."""
    import difflib
    import re

    def parts(text: str) -> set[str]:
        return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower()))

    want = parts(name)
    scored = []
    for e in catalog():
        shared = want & parts(e.short)
        if not any(p.isalpha() and len(p) > 2 for p in shared):  # the family must match
            continue
        ratio = difflib.SequenceMatcher(None, name.lower(), e.short.lower()).ratio()
        scored.append((len(shared) + ratio, e.short))
    scored.sort(key=lambda t: -t[0])
    return [short for _, short in scored[:n]]


def looks_local(text: str) -> bool:
    """A path the user meant on disk: never sent to the Hub (section 7)."""
    return text.startswith(("./", "../", "/", "~"))


def is_model_dir(text: str) -> bool:
    """An existing local model: a directory with a config.json, or that file."""
    p = Path(text).expanduser()
    return (p.is_dir() and (p / "config.json").is_file()) or (p.is_file() and p.suffix == ".json")


def looks_like_hub_id(text: str) -> bool:
    return bool(re.fullmatch(r"[\w.\-]+/[\w.\-]+", text)) and not looks_local(text)


@dataclass
class Match:
    """What the words named: one model, several to choose from, or none."""

    words: list[str]
    entries: list[Entry] = field(default_factory=list)  # catalog matches, best first
    exact: Optional[str] = None  # an id or path taken as given
    near: list[str] = field(default_factory=list)  # closest catalog names when none matched

    @property
    def unique(self) -> Optional[str]:
        if self.exact:
            return self.exact
        return self.entries[0].id if len(self.entries) == 1 else None


def _normal(word: str) -> str:
    w = word.lower().strip(",.")
    w = re.sub(r"^(\d+(?:\.\d+)?)\s*(b|bn|billion)$", r"\1b", w)  # 7B, 7bn -> 7b
    return w


def resolve(words: list[str], cached: Iterable[str] = ()) -> Match:
    m = Match(words=list(words))
    if not words:
        return m
    one = words[0] if len(words) == 1 else ""
    if one and (looks_local(one) or is_model_dir(one) or looks_like_hub_id(one)):
        hit = by_id(one)
        m.exact = hit.id if hit else one
        if hit:
            m.entries = [hit]
        return m
    known = all_words()
    query = [_normal(w) for w in words]
    # "7" or "72" alone mean a size, people drop the b; "3" in "gemma 3" is
    # a family's own word, so it stays.
    query = [w + "b" if re.fullmatch(r"\d+(\.\d+)?", w) and w not in known else w for w in query]
    hits = [e for e in catalog() if all(q in e.words() for q in query)]
    if not hits:
        import difflib

        names = {e.short.lower(): e.short for e in catalog()}
        m.near = [
            names[n] for n in difflib.get_close_matches(" ".join(query), names, n=3, cutoff=0.3)
        ]
        return m
    cached = {c.lower() for c in cached}
    # Cached first, then validated before same-shape, then catalog order.
    order = {e.id: i for i, e in enumerate(catalog())}
    hits.sort(key=lambda e: (e.id.lower() not in cached,
                             e.evidence not in ("validated", "measured"), order[e.id]))
    m.entries = hits
    return m
