"""The config.json facts Poros ships for the models its built-in recipes name
(poros/plan/config_facts.json): read only where the Hugging Face cache has no config.json, and
planned from exactly as the model's own file is."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from poros.plan import configs

ROOT = Path(__file__).resolve().parent.parent
FACTS = json.loads((ROOT / "poros/plan/config_facts.json").read_text())["models"]
CARDS = [("NVIDIA GeForce RTX 4090", 23.99, 23.5), ("NVIDIA GeForce RTX 3090", 24.0, 23.6),
         ("NVIDIA GeForce RTX 4060 Laptop GPU", 7.68, 7.6)]


def test_it_ships_facts_for_every_model_a_built_in_names_and_nothing_else():
    """Every built-in's model, and only the fields the plan reads (no file copies)."""
    from poros.core.config import _MODEL_DRAWS
    from poros.recipes import presets

    assert {r.model for r in presets.DEFINED} <= set(FACTS)

    def keys(c: dict) -> set:
        return set(c) | {k for v in c.values() if isinstance(v, dict) for k in keys(v)}

    for model, entry in FACTS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), model  # the Hub snapshot
        assert all(k in configs.PLAN_KEYS or _MODEL_DRAWS.search(k)
                   for k in keys(entry["fields"])), model


def _plans(model: str, config: dict, monkeypatch) -> list:
    """What poros fit and a recipe's plan read for ``model`` from ``config``, on each card and
    path: its figures, verdict, path decision and tables."""
    from poros.cli.context import Context
    from poros.cli.verbs import shared
    from poros.core.schemas import PlanGpu

    monkeypatch.setattr(configs, "cached_model_config", lambda m: dict(config))
    monkeypatch.setattr(configs, "read_model_config", lambda m, revision=None: dict(config))
    out = []
    for name, total, free in CARDS:
        ctx = Context(cwd=Path.cwd(), fixed_gpu=PlanGpu(name=name, total_gib=total,
                                                        free_gib=free, source="nvidia-smi"),
                      fixed_interactive=False)
        for path in (None, "efficient", "fast", "sequential"):
            p = shared.plan_for(model, ctx, seq_len=512, config_json=True,
                                path_settings={"path": path} if path else None)
            out.append((name, path, p.path, p.fits, p.peak_vram_allocated.value,
                        p.peak_vram_reserved.value, p.peak_host_ram.value, p.offload_reason,
                        p.fit_basis))
    return out


@pytest.mark.parametrize("model", sorted(FACTS))
def test_the_shipped_facts_plan_as_the_models_own_config_does(model, monkeypatch):
    """Where this machine's Hugging Face cache holds the model's own config.json at the snapshot
    the facts came from: the same figures, verdicts and path decisions from either."""
    from huggingface_hub import try_to_load_from_cache

    entry = FACTS[model]
    hit = try_to_load_from_cache(model, "config.json", revision=entry["revision"])
    if not isinstance(hit, str):
        pytest.skip("its config.json at that snapshot is not in this machine's Hub cache")
    real = json.loads(Path(hit).read_text())
    assert _plans(model, entry["fields"], monkeypatch) == _plans(model, real, monkeypatch)


def test_a_cached_config_is_read_and_the_shipped_facts_are_not(tmp_path, monkeypatch):
    from huggingface_hub import constants

    snap = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 1}))
    refs = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B" / "refs"
    refs.mkdir()
    (refs / "main").write_text("a" * 40)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    def never(model):
        raise AssertionError(f"read the shipped facts of {model}")

    monkeypatch.setattr(configs, "shipped_config", never)
    assert configs.cached_model_config("Qwen/Qwen2.5-7B") == {"model_type": "qwen2",
                                                               "hidden_size": 1}


def test_an_empty_cache_plans_from_the_shipped_facts_and_a_run_never_does(tmp_path, monkeypatch):
    """The list and show (cached_model_config) read the facts; a run, a fit and every gate read
    the model's own file (read_model_config), which an offline machine without it does not have."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    assert configs.cached_model_config("Qwen/Qwen2.5-7B-Instruct") == (
        FACTS["Qwen/Qwen2.5-7B-Instruct"]["fields"])
    assert configs.cached_model_config("Qwen/Qwen3-32B") is None  # no recipe names it
    read = []
    monkeypatch.setattr(configs, "shipped_config", lambda m: read.append(m))
    assert configs.read_model_config("Qwen/Qwen2.5-7B-Instruct") is None and read == []
