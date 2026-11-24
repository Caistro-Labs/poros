"""Write the command table in docs/cli.md from tests/cli/sentences.yaml.

    python scripts/gen_cli_words.py          # rewrite the table
    python scripts/gen_cli_words.py --check  # exit 1 if it is stale

One file, two uses: the tests parse every command in it, and this
table documents them (the CLI interface design, section 10).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "cli.md"
START, END = "<!-- sentences:start -->", "<!-- sentences:end -->"


def table() -> str:
    cases = yaml.safe_load((ROOT / "tests" / "cli" / "sentences.yaml").read_text())
    rows = ["| Command | Verb | Means |", "|---|---|---|"]
    for case in cases:
        g = case["grammar"]
        # the docs list what runs as written in a shell: a session's own forms (a leading /,
        # words with no verb) are parsed by the tests and shown where they work, in the session
        if case["say"].startswith("/") or not g.get("verb"):
            continue
        verb = g["verb"]
        if g.get("action"):
            verb += f" {g['action']}"
        parts = []
        if g.get("subject"):
            parts.append("subject: " + " ".join(str(w) for w in g["subject"]))
        for k, v in (g.get("clauses") or {}).items():
            parts.append(f"{k} = {v}")
        for k, v in (g.get("flags") or {}).items():
            parts.append(k if v is True else f"{k} {v}")
        if "model" in case:
            parts.append("model: " + case["model"])
        rows.append(f"| `poros {case['say']}` | {verb} | {'; '.join(parts) or '—'} |")
    return "\n".join(rows)


def render(doc: str) -> str:
    a, b = doc.index(START) + len(START), doc.index(END)
    return doc[:a] + "\n" + table() + "\n" + doc[b:]


if __name__ == "__main__":
    doc = DOC.read_text()
    new = render(doc)
    if "--check" in sys.argv:
        sys.exit(0 if new == doc else 1)
    DOC.write_text(new)
