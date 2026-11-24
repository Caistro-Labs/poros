"""Build poros/plan/cells.json from docs/validation/plan_sources.yaml and the Mac's own.

The check before a run decides only from numbers read from committed logs. This script
resolves every pointer in those files against its log, converts the value to GiB (or
seconds), and records where it came from as `receipt#ptr`. Run it after adding a log;
tests/test_plan.py fails if the committed table and the logs disagree.

The logs (docs/validation/) are read from POROS_BENCHMARKS, the root of a checkout of
github.com/Caistro-Labs/poros-benchmarks, when it is set, else from this repository's root.

    python scripts/build_plan_cells.py [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
# where docs/validation/ is: POROS_BENCHMARKS (a poros-benchmarks checkout), else this repository
ROOT = Path(os.environ["POROS_BENCHMARKS"]).expanduser().resolve() if os.environ.get(
    "POROS_BENCHMARKS") else REPO
SOURCES = (ROOT / "docs/validation/plan_sources.yaml",
           ROOT / "docs/validation/metal/m4_16gb/plan_sources.yaml",  # the Mac's own cells
           # the certified recipes' runs: scripts/certify/generate.py writes it
           ROOT / "docs/validation/recipes/plan_sources.yaml")
CELLS = REPO / "poros/plan/cells.json"
TO_BASE = {"GiB": 1.0, "GB": 1e9 / 2**30, "MB": 1e6 / 2**30, "bytes": 1 / 2**30, "s": 1.0}


def _scalar(text: str):
    """A runs log's value as the YAML its config was written from reads it."""
    return yaml.safe_load(text)


def _runs_log(text: str) -> list[dict]:
    """A runs log (the RTX 4090 sweep's runs.log): one row per run, its
    ``<time> start NAME key=value ...`` line (the settings it was given) merged with its
    ``RUN NAME key=value ...`` line (what it measured); a bracketed list is skipped."""
    rows: dict[str, dict] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "start":
            name, said = parts[2], parts[3:]
            rows[name] = {"name": name, "started": parts[0]}
        elif len(parts) >= 2 and parts[0] == "RUN":
            name, said = parts[1], parts[2:]
            rows.setdefault(name, {"name": name})
        else:
            continue
        for word in said:
            key, eq, value = word.partition("=")
            if eq and value and not value.startswith("["):
                rows[name][key] = _scalar(value.rstrip(","))
    return list(rows.values())


def _load(receipt: str, select: dict | None):
    text = (ROOT / receipt).read_text()
    if receipt.endswith(".log"):
        rows = _runs_log(text)
    elif receipt.endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines()]
    else:
        return json.loads(text)
    for row in rows:
        if all(row.get(k) == v for k, v in (select or {}).items()):
            return row
    raise ValueError(f"{receipt}: no row matches {select}")


def _resolve(doc, ptr: str):
    for part in ptr.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def build() -> list[dict]:
    if not (ROOT / "docs/validation").is_dir():
        raise SystemExit(f"{ROOT} has no docs/validation/: set POROS_BENCHMARKS to the root of a "
                         "checkout of github.com/Caistro-Labs/poros-benchmarks")
    cells = []
    for src in (s for path in SOURCES for s in yaml.safe_load(path.read_text())):
        # `select`: the row of a JSONL file or a runs log that is this cell's run
        main = _load(src["receipt"], src.get("select"))

        def value(v):
            if not isinstance(v, dict):
                return v
            doc = _load(v["receipt"], v.get("select")) if "receipt" in v else main
            return _resolve(doc, v["ptr"])

        fields = {}
        for name, f in src["fields"].items():
            receipt = f.get("receipt", src["receipt"])
            select = f.get("select", src.get("select") if receipt == src["receipt"] else None)
            raw = _resolve(_load(receipt, select), f["ptr"])
            if "match" in f:  # a number inside a recorded sentence (e.g. a sizing's reason)
                found = re.search(f["match"], str(raw))
                if found is None:
                    raise ValueError(f"{receipt}#{f['ptr']}: {f['match']!r} matches nothing")
                raw = found.group(1)
            fields[name] = {
                "value": round(float(raw) * TO_BASE[f["unit"]], 3),
                "unit": "s" if f["unit"] == "s" else "GiB",
                "receipt": f"{receipt}#{f['ptr']}",
                **({"method": f["method"]} if "method" in f else {}),
            }
        cells.append({
            "id": src["id"],
            "key": {k: value(v) for k, v in src["key"].items()},
            "host": {k: value(v) for k, v in src.get("host", {}).items()},
            "fields": fields,
            # a certification's measurement: its own machine's fit, never another's estimate
            **({"only_its_machine": True} if src.get("only_its_machine") else {}),
        })
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if cells.json is stale")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        if CELLS.read_text() != text:
            print(f"{CELLS} is stale: run python scripts/build_plan_cells.py", file=sys.stderr)
            return 1
        return 0
    CELLS.write_text(text)
    print(f"wrote {CELLS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
