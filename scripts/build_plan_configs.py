"""Build poros/plan/config_facts.json: the config.json fields the plan reads, for the models the
built-in recipes name and the bases their catalog records are planned from.

A fresh install has no config.json in its Hugging Face cache, and `poros recipes` reads no
network: the plan reads these facts instead (poros.plan.configs.shipped_config) until a download
brings the model's own file. Only the fields the plan reads (poros.plan.configs.PLAN_KEYS, and
any field naming one of the model's own random draws), never the file: numbers and names, as
shapes.json already ships. This script reads each config.json from the Hugging Face cache (no
network: `hf download MODEL config.json` first) and records the snapshot it came from; a model
whose config.json is not cached keeps its committed entry.

    python scripts/build_plan_configs.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FACTS = ROOT / "poros/plan/config_facts.json"


def models() -> list[str]:
    """Every model a built-in recipe names, the catalog base each is planned from, and the
    Instruct model of each base (what a chats recipe on that base trains)."""
    from poros.cli.words.models import by_id, instruct_twin
    from poros.recipes import presets

    named = {r.model for r in presets.DEFINED}
    vias = {e.via for m in named if (e := by_id(m)) is not None and e.via}
    twins = {t.id for m in named if (t := instruct_twin(m)) is not None}
    return sorted(named | vias | twins)


def fields(config: dict) -> dict:
    from poros.core.config import _MODEL_DRAWS
    from poros.plan.configs import PLAN_KEYS

    def keep(c: dict) -> dict:
        return {k: (keep(v) if isinstance(v, dict) else v) for k, v in c.items()
                if k in PLAN_KEYS or _MODEL_DRAWS.search(k)}

    return keep(config)


def build(committed: dict) -> dict:
    from huggingface_hub import try_to_load_from_cache

    out = {}
    for model in models():
        hit = try_to_load_from_cache(model, "config.json")
        if not isinstance(hit, str):
            if model in committed:
                out[model] = committed[model]
            continue
        path = Path(hit)
        out[model] = {"revision": path.parent.name, "fields": fields(json.loads(path.read_text()))}
    return {"about": "the config.json fields the plan reads, for the models the built-in recipes "
                     "name (scripts/build_plan_configs.py): what a fresh cache plans from",
            "models": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if config_facts.json is stale")
    args = ap.parse_args()
    committed = json.loads(FACTS.read_text())["models"] if FACTS.is_file() else {}
    text = json.dumps(build(committed), indent=1, sort_keys=True) + "\n"
    if args.check:
        if not FACTS.is_file() or FACTS.read_text() != text:
            print(f"{FACTS} is stale: run python scripts/build_plan_configs.py", file=sys.stderr)
            return 1
        return 0
    FACTS.write_text(text)
    print(f"wrote {FACTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
