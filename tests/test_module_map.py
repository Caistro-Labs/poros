"""AGENTS.md's module map names every package and top-level module under poros/, and nothing
that is not there."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_module_map_covers_the_tree():
    text = (ROOT / "AGENTS.md").read_text()
    section = text[text.index("## Module map"):]
    section = section[:section.index("\n## ", 5)] if "\n## " in section[5:] else section
    packages = {p.name for p in (ROOT / "poros").iterdir()
                if p.is_dir() and (p / "__init__.py").exists()} - {"docs", "agent_docs"}
    named = set(re.findall(r"^poros/(\w+)/", section, re.M))
    assert packages - named == set(), "packages the map leaves out"
    assert named - packages == set(), "packages the map names that are not there"
    modules = {p.stem for p in (ROOT / "poros").glob("*.py")
               if not p.stem.startswith("_")}
    line = next(x for x in section.splitlines() if x.startswith("poros/*.py"))
    rest = section[section.index(line):].split("```")[0]
    assert modules - set(re.findall(r"\b(\w+)\b", rest)) == set()
