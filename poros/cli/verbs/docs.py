"""docs: the docs for people, from this install. No website, no network.

    poros docs
    poros docs cli
    poros docs cli recipes

With no topic, the topics: each one's name, title and what it covers. With a topic, its text in
Markdown and its sections. With words after the topic, the one section whose heading holds them
all, so an agent reads what it needs and no more. `guide` is what `poros agents` prints.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import Output
from poros.cli.ui.text import Line, S
from poros.cli.words.grammar import Sentence
from poros.core.refusal import Refusal

GUIDE = "guide"  # poros/agents.md: what `poros agents` prints
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_HEADING = re.compile(r"(#{1,4})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"\s*(```|~~~)")


class DocsAction(Action):
    verb: ClassVar[str] = "docs"
    topic: str = ""
    section: list[str] = []

    def canonical(self) -> list[str]:
        return ["docs", *([self.topic] if self.topic else []), *self.section]


class Topic(BaseModel):
    name: str
    title: str
    about: str  # its first paragraph, cut to a line
    sections: int  # how many headings under its title


class DocsOutput(Output):
    schema_: Literal["poros.docs.v1"] = Field("poros.docs.v1", alias="schema")
    topics: Optional[list[Topic]] = None  # no topic named: every one
    topic: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None  # the heading the words found
    sections: list[str] = []  # the topic's headings, to ask for one by
    text: Optional[str] = None  # Markdown, as the file has it
    next: list[str] = []


def _files() -> dict[str, str]:
    """name -> the doc's text, for every doc this install carries, and the guide."""
    root = resources.files("poros")
    found: dict[str, str] = {}
    docs = root.joinpath("docs")
    if docs.is_dir():
        for f in sorted(docs.iterdir(), key=lambda t: t.name):
            stem = f.name[:-3] if f.name.endswith(".md") else ""
            if stem and _NAME.fullmatch(stem) and f.is_file():
                found[stem] = f.read_text(encoding="utf-8")
    found[GUIDE] = root.joinpath("agents.md").read_text(encoding="utf-8")
    return found


def _headings(text: str) -> list[tuple[int, int, str]]:
    """(line, level, heading) for each heading outside a code block."""
    out, fenced = [], False
    for n, line in enumerate(text.splitlines()):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        m = None if fenced else _HEADING.fullmatch(line)
        if m:
            out.append((n, len(m.group(1)), m.group(2)))
    return out


def _title(name: str, text: str) -> str:
    first = next((h for _, level, h in _headings(text) if level == 1), "")
    return first or name


def _about(text: str) -> str:
    """The first paragraph that is prose, as one line of at most 160 characters."""
    para: list[str] = []
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        s = line.strip()
        if fenced or s.startswith(("#", ">", "|", "<!--", "-", "*", "!", "[")):
            if para:
                break
            continue
        if not s:
            if para:
                break
            continue
        para.append(s)
    said = " ".join(para)
    return said if len(said) <= 160 else said[:157].rstrip() + "..."


def _section(text: str, words: list[str]) -> Optional[tuple[str, str]]:
    """(heading, its text up to the next heading at its level or above) for the first heading that
    holds every word, ignoring case."""
    want = [w.lower() for w in words]
    lines = text.splitlines()
    heads = _headings(text)
    for i, (n, level, heading) in enumerate(heads):
        if all(w in heading.lower() for w in want):
            end = next((m for m, lv, _ in heads[i + 1:] if lv <= level), len(lines))
            return heading, "\n".join(lines[n:end]).strip() + "\n"
    return None


def resolve(s: Sentence, ctx: Context) -> Draft[DocsAction]:
    words = list(s.subject or [])
    if not words:
        return Draft(DocsAction())
    topic = words[0].lower()
    if not _NAME.fullmatch(topic):
        raise Refusal("E_USAGE", "Which topic", "a topic is a name from `poros docs`",
                      fix="poros docs", note="lists them")
    return Draft(DocsAction(topic=topic, section=words[1:]))


def confirmations(action: DocsAction, ctx: Context) -> list:
    return []


def run(action: DocsAction, ctx: Context) -> DocsOutput:
    files = _files()
    if not action.topic:
        topics = [Topic(name=name, title=_title(name, text), about=_about(text),
                        sections=sum(1 for _, level, _ in _headings(text) if level >= 2))
                  for name, text in files.items()]
        return DocsOutput(topics=topics, next=["poros docs guide", "poros docs cli"])
    text = files.get(action.topic)
    if text is None:
        raise Refusal("E_NOT_FOUND", "No such topic",
                      f"no doc named {action.topic} in this install",
                      fix="poros docs", note="lists them")
    sections = [h for _, level, h in _headings(text) if level == 2]
    out = DocsOutput(topic=action.topic, title=_title(action.topic, text), sections=sections)
    if not action.section:
        out.text = text
        return out
    hit = _section(text, action.section)
    if hit is None:
        raise Refusal("E_NOT_FOUND", "No such section",
                      f"no heading in {action.topic} holds those words",
                      fix=f"poros docs {action.topic}", note="its sections are listed")
    out.section, out.text = hit
    return out


def view(r: DocsOutput, w: int) -> list[Line]:
    if r.topics is not None:
        width = max((len(t.name) for t in r.topics), default=0) + 2
        lines: list[Line] = [[S("Docs", "hi"), S("  from this install", "dm")], []]
        for t in r.topics:
            lines.append([S(t.name.ljust(width), "hi"), S(t.title, "tx")])
        lines += [[], [S("poros docs <topic>", "hi"), S("  reads one; words after it, one section",
                                                        "dm")]]
        return lines
    return [[S(line, None)] for line in (r.text or "").splitlines()]
