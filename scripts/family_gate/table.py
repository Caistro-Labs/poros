#!/usr/bin/env python3
"""table.py CARD=RESULT.txt [CARD=RESULT.txt ...] -- family_gate RESULT files as one Markdown
matrix: a row per (family, check), a column per card, each cell PASS / FAIL / SKIP / NOTE (the
last record of that check on that card wins, so several RESULT files of one card may be given,
oldest first)."""

import re
import sys

cells: dict = {}
cards: list = []
order: list = []
for arg in sys.argv[1:]:
    card, path = arg.split("=", 1)
    if card not in cards:
        cards.append(card)
    for line in open(path, errors="replace"):
        m = re.match(r"^(PASS|FAIL|SKIP|NOTE)\s+(\S+)\s*(.*)$", line.rstrip("\n"))
        if not m or ":" not in m.group(2):
            continue
        st, name, detail = m.groups()
        fam, check = (
            name.rsplit(":", 1)
            if not name.startswith("any:")
            else (name.split(":")[1], ":".join(name.split(":")[2:]))
        )
        if name.startswith("any:"):
            fam = "any:" + fam
        if check.startswith("ran:"):  # per-run expectation lines: folded into "ran"
            check = "ran"
        key = (fam, check)
        if key not in order:
            order.append(key)
        prev = cells.get((key, card))
        if prev and prev[0] == "FAIL" and check == "ran" and st != "FAIL":
            continue
        cells[(key, card)] = (st, detail)
print("| family | check | " + " | ".join(cards) + " |")
print("|---|---|" + "---|" * len(cards))
for key in order:
    row = []
    for card in cards:
        st, detail = cells.get((key, card), ("", ""))
        short = re.sub(r"\s+", " ", detail)[:90].replace("|", "/")
        row.append(f"{st} {short}".strip() or "-")
    print(f"| {key[0]} | {key[1]} | " + " | ".join(row) + " |")
