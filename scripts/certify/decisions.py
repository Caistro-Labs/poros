#!/usr/bin/env python3
"""What the planner decides for each built-in recipe on a named machine, before and after the
certified runs' cells are in. No GPU and no training: the machine's probes are given.

    python scripts/certify/decisions.py --gpu "NVIDIA GeForce RTX 4090:23.99" --ram 62.5 \
        --free 58 [--recipes big-72b ...] [--cells docs/validation/recipes/plan_sources.yaml]

For each recipe it prints, from poros's own functions (scripts/certify/run_recipe.py's
check_fit and gate_of):
  * the recipe as shipped: whether poros train starts, warns ("may not fit"), or asks before it
    starts (a refusal for --yes runs), with its plan's verdict, peak and source;
  * the same settings with path left unset, as a run without a recipe gives them: the path
    decide_path_to_fit picks, the tables decide_offload_for_run places, vram_error, and the row
    length fitted_length keeps for rows of 2048 tokens nobody asked for.
With --cells (a plan_sources.yaml of new cells, as generate.py writes), each is printed before
and after the cells are added, and the lines that differ are marked CHANGED. --json prints all.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "certify"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_recipe  # noqa: E402

from poros.cli.context import Context  # noqa: E402
from poros.core.schemas import PlanGpu  # noqa: E402
from poros.recipes import certified, presets  # noqa: E402


@dataclasses.dataclass
class Machine(Context):
    """A Context whose machine is given: its GPU, host RAM and the host RAM free."""

    ram_total: float = 0.0
    ram_free: float = 0.0

    @property
    def host_free_gib(self) -> Optional[float]:  # type: ignore[override]
        return self.ram_free


def machine(gpu: str, ram: float, free: float, cwd: Path) -> Machine:
    name, _, gib = gpu.rpartition(":")
    card = PlanGpu(name=name, total_gib=float(gib), free_gib=float(gib) - 0.4,
                   source="nvidia-smi")
    certified.host_ram_total_gib = lambda: ram  # this machine's, as certified.here reads it
    import poros.core.compute as compute

    compute.triton_installed = lambda: True  # a CUDA box has Triton: the fused compute's choice
    return Machine(cwd=cwd, fixed_gpu=card, fixed_interactive=False, ram_total=ram,
                   ram_free=free)


def unset_path(r: presets.Recipe, data: str) -> Any:
    """``r``'s config with path, the tables and the expert cache left unset: what a run without
    a recipe, at the same settings otherwise, gives the planner to decide."""
    cfg = run_recipe.recipe_config(r, data)
    return cfg.model_copy(update={"path": None, "offload_embeddings_and_head": None,
                                  "moe_expert_cache_gib": None})


def length_kept(r: presets.Recipe, ctx: Context, asked: int = 2048) -> int:
    """The row length the train verb keeps (shared.fitted_length) for rows of ``asked`` tokens
    nobody gave a --seq-len for, at the recipe's other settings and path unset."""
    from poros.cli.verbs import shared
    from poros.cli.verbs.train import sss_dir

    st = r.settings
    loop = {**presets.loop_preset(r), **st}  # rank and dropout: its loop's where it says none
    said = {k: st[k] for k in ("per_device_batch_size", "gradient_accumulation_steps",
                               "block_size") if k in st}
    return shared.fitted_length(asked, lambda n: shared.plan_for(
        r.model, ctx, seq_len=n, rank=int(loop["lora_rank"]), dropout=float(loop["lora_dropout"]),
        store=st.get("store", "ram"), path_settings=said,
        store_path=str(sss_dir()) if st.get("store") == "ssd" else None))[0]


def decide(r: presets.Recipe, ctx: Context, data: Path) -> Dict[str, Any]:
    shipped = run_recipe.check_fit(r, ctx, data)
    train = shipped["train"]
    starts = ("asks: " + "; ".join(train.get("asks_text") or train.get("asks", []))
              if train.get("asks") else
              "warns: " + shipped["gate"]["warning"] if shipped["gate"].get("warning")
              else "starts")
    unset = run_recipe.gate_of(r, ctx, str(data), unset_path(r, str(data)))
    return {"recipe": r.key, "train": train, "gate": shipped["gate"],
            "train_starts": starts, "no_recipe": {**unset, "length_kept_of_2048":
                                                   length_kept(r, ctx)}}


def with_cells(sources: Path, root: Path = ROOT) -> List[dict]:
    """The cells ``sources`` (a plan_sources.yaml whose receipts start at ``root``) builds,
    through the planner's cell builder (the development repository's)."""
    import build_plan_cells as b

    was = b.SOURCES, b.ROOT
    b.SOURCES, b.ROOT = (sources,), root
    try:
        return b.build()
    finally:
        b.SOURCES, b.ROOT = was


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", required=True, help='"name:GiB", as nvidia-smi names the card')
    ap.add_argument("--ram", type=float, required=True, help="host RAM, GiB (MemTotal)")
    ap.add_argument("--free", type=float, required=True, help="host RAM free, GiB")
    ap.add_argument("--recipes", nargs="+", help="default: every CUDA built-in")
    ap.add_argument("--cells", type=Path, help="new cells' plan_sources.yaml: before and after")
    ap.add_argument("--cells-root", type=Path, default=ROOT, help="where --cells' receipts start")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from poros.plan import cells as plan_cells

    with tempfile.TemporaryDirectory() as tmp:
        ctx = machine(a.gpu, a.ram, a.free, Path(tmp))
        data = Path(tmp) / "chats.jsonl"
        keys = a.recipes or [r.key for r in presets.DEFINED if r.use != "mac"]
        recipes = [r for r in presets.DEFINED if r.key in keys]
        out = []
        for r in recipes:
            kind = {"qa": "qa", "pairs": "pairs"}.get(r.data, "chats")
            data = Path(tmp) / f"{kind}.jsonl"
            if not data.exists():
                run_recipe.write_rows(data, run_recipe.sample_rows(kind, 64, 2048))
            row = {"before": decide(r, ctx, data)}
            if a.cells:
                was = plan_cells.CELLS, plan_cells.CERTIFIED
                new = [c for c in with_cells(a.cells, a.cells_root)
                       if not c["key"]["gpu"].startswith("Apple")]
                # as poros.plan.cells splits them: a certification run's apart, its card's only
                plan_cells.CELLS = was[0] + [c for c in new if not c.get("only_its_machine")]
                plan_cells.CERTIFIED = was[1] + [c for c in new if c.get("only_its_machine")]
                try:
                    row["after"] = decide(r, dataclasses.replace(ctx), data)
                finally:
                    plan_cells.CELLS, plan_cells.CERTIFIED = was
            out.append(row)
    if a.json:
        print(json.dumps(out, indent=1, default=str))
        return 0
    for row in out:
        print(f"== {row['before']['recipe']}")
        before = describe(row["before"])
        for when, d in row.items():
            for old, line in zip(before, describe(d)):
                mark = ("CHANGED " if old != line else "same    ") if when == "after" else ""
                print(f"  {when:6} {mark}{line}")
    return 0


def describe(d: Dict[str, Any]) -> List[str]:
    t, n = d["train"], d["no_recipe"]
    return [f"train: {d['train_starts']}; plan {t.get('verdict')} "
            f"{t.get('peak_gib')} GiB ({t.get('source')})",
            f"no recipe, path unset: path {n['path'] or n['ran_path']}, offload "
            f"{n['offload']}, refusal {n['vram_refusal'] or n['host_refusal']}, "
            f"length kept of 2048: {n['length_kept_of_2048']}"]


if __name__ == "__main__":
    sys.exit(main())
