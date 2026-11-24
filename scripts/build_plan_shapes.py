"""Build poros/plan/shapes.json: the shapes of each receipt model's config.json.

`poros plan` finds receipts by model id. A model no receipt names -- a local folder of a Hub
model, a fine-tune of one -- is planned from the receipts of the model whose config.json has the
same shapes (poros.plan.cells.shape_twin; the keys are poros.plan.cells.SHAPE_KEYS). This script
reads each receipt model's config.json from the Hugging Face cache (no network: download a model's
config.json first, e.g. `hf download MODEL config.json`) and records its signature and the
snapshot it came from. A model whose config.json is not cached keeps its committed entry, or has
none (no twin plans from it).

    python scripts/build_plan_shapes.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHAPES = ROOT / "poros/plan/shapes.json"


def _cached(model: str) -> tuple[dict, str] | None:
    """(config.json, snapshot revision) of ``model`` from the Hugging Face cache, or None."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(model, "config.json")
    if not isinstance(hit, str):
        return None
    path = Path(hit)
    return json.loads(path.read_text()), path.parent.name


def build(committed: dict) -> dict:
    from poros.plan.cells import CELLS, shape_signature

    models = sorted({c["key"]["model"] for c in CELLS if not c["id"].startswith("mac-")})
    out = {}
    for model in models:
        got = _cached(model)
        if got is None:
            if model in committed:
                out[model] = committed[model]
            continue
        config, revision = got
        out[model] = {"revision": revision, "signature": shape_signature(config)}
    return {"about": "config.json shapes of the receipt models (scripts/build_plan_shapes.py): "
                     "a model no receipt names is planned from the one it matches",
            "models": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if shapes.json is stale")
    args = ap.parse_args()
    committed = json.loads(SHAPES.read_text())["models"] if SHAPES.is_file() else {}
    text = json.dumps(build(committed), indent=1, sort_keys=True) + "\n"
    if args.check:
        if SHAPES.read_text() != text:
            print(f"{SHAPES} is stale: run python scripts/build_plan_shapes.py", file=sys.stderr)
            return 1
        return 0
    SHAPES.write_text(text)
    print(f"wrote {SHAPES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
