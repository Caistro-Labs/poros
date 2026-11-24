"""`poros docs` and `poros agents docs`: Poros's own docs, read from the install, no network."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import poros
from poros.cli.verbs import docs

REPO = Path(__file__).parents[2]


def _agent(*args, code=0):
    proc = subprocess.run([sys.executable, "-c", "from poros.cli.main import cli; cli()", "agents",
                           "docs", *args], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, env={**os.environ, "POROS_DEV_SIMULATE": "1"})
    assert proc.returncode == code, proc.stdout + proc.stderr[-600:]
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 1, proc.stdout
    return json.loads(lines[0])


def test_the_docs_ship_in_the_package_and_internal_ones_do_not():
    held = Path(poros.__file__).parent / "docs"
    assert held.resolve() == (REPO / "docs").resolve()  # one copy: the package links to docs/
    data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]
    assert "docs/*.md" in data["package-data"]["poros"]  # top level only: no subfolder ships
    names = set(docs._files())
    assert {p.stem for p in (REPO / "docs").glob("*.md")} | {"guide"} == names
    assert not any("internal" in n for n in names)


def test_alone_it_lists_every_topic_with_a_title_and_what_it_covers():
    said = _agent()
    assert said["schema"] == "poros.docs.v1"
    topics = {t["name"]: t for t in said["topics"]}
    assert {"guide", "cli", "quickstart", "install"} <= set(topics)
    assert topics["guide"]["title"] == "Poros for agents"
    assert all(t["title"] and not t["about"].startswith(("[", "#", "|")) for t in topics.values())
    assert said["next"] == ["poros agents docs guide", "poros agents docs cli"]


def test_a_topic_is_its_file_word_for_word_with_its_sections():
    said = _agent("cli")
    assert said["text"] == (REPO / "docs" / "cli.md").read_text()
    assert "Recipes" in said["sections"] and said["title"] == "Poros commands"


def test_words_after_a_topic_give_one_section_and_no_more():
    said = _agent("cli", "recipes")
    assert said["section"] == "Recipes" and said["text"].startswith("## Recipes\n")
    assert "\n## " not in said["text"][3:]  # it ends where the next section of its level begins
    guide = _agent("guide")
    assert guide["text"] == (Path(poros.__file__).parent / "agents.md").read_text()


@pytest.mark.parametrize("word", ["../etc", "internal/adding-architectures", "CLI.md", "a/b"])
def test_a_topic_is_a_name_from_the_list_never_a_path(word):
    said = _agent(word, code=2)
    assert said["error"]["code"] == "E_USAGE" and said["error"]["hint"] == "poros agents docs"


def test_an_unknown_topic_or_section_says_where_to_look():
    assert _agent("nope", code=2)["error"]["code"] == "E_NOT_FOUND"
    hit = _agent("cli", "no", "such", "heading", code=2)["error"]
    assert hit["code"] == "E_NOT_FOUND" and hit["hint"] == "poros agents docs cli"


def test_a_heading_inside_a_code_block_is_not_a_section():
    text = "# T\n\n## A\n\n```\n## not one\n```\n\n## B\nb\n"
    assert [h for _, _, h in docs._headings(text)] == ["T", "A", "B"]
    assert docs._section(text, ["a"]) == ("A", "## A\n\n```\n## not one\n```\n")


def test_every_page_an_error_names_opens_with_poros_docs():
    """An error's docs link is `poros docs <page> [<section words>]` (poros.links.doc_ref), which
    any install opens offline: every page the code names is a topic, and its section words find
    a heading there. A repository path (docs/x.md) means nothing to a pip user."""
    import re

    from poros.cli.verbs import docs as verb
    from poros.links import doc_ref

    root = Path(poros.__file__).parent
    named = set()
    for f in root.rglob("*.py"):
        named |= set(re.findall(r'(?:doc_url|docs)\s*=\s*"(docs/[a-z0-9-]+\.md(?:#[a-z0-9-]+)?)"',
                                f.read_text(encoding="utf-8")))
    assert len(named) >= 5
    files = verb._files()
    for doc in sorted(named):
        ref = doc_ref(doc)
        page, *words = ref.removeprefix("poros docs ").split()
        assert ref.startswith("poros docs ") and page in files, doc
        if words:
            assert verb._section(files[page], words) is not None, doc
    assert doc_ref("https://x.com") == "https://x.com"
    from poros.core.errors import PorosOOMError

    assert PorosOOMError().info.doc_url == "poros docs limitations"
