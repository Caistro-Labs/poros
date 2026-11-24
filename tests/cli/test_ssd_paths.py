"""A MoE model's experts on the SSD store: the train card's path list and the planner
(shared.path_plans) give efficient (the default), fast, fast+cache (fast with the store's host
cache) and sequential, each with its VRAM fit, host RAM and seconds an update; fast+cache sets
path fast with ``store_options: {cache: 4G}``, and nothing else changes a default."""

from __future__ import annotations

import json
import time

import pytest

from poros.cli import sheets
from poros.cli.context import Context
from poros.cli.schemas import Figure, PlanGpu
from poros.cli.ui.sheet import Sheet
from poros.cli.verbs import shared
from poros.cli.words.grammar import parse
from poros.core.refusal import Refusal
from tests.sss.test_store_surface import QWEN3_30B

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Qwen3-30B-A3B's shapes: its RTX 4090 receipts
CARD = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=23.5, source="spec")
AT = "--seq-len 512 lora_dropout=0 per_device_batch_size=8 gradient_accumulation_steps=1"


@pytest.fixture
def disk(monkeypatch, tmp_path):
    """The store folder, its disk probed at 3.2 GB/s (the cache a plan --store or a run
    writes); config.json read from the test, nothing from the Hub."""
    import poros.plan.configs
    import poros.preflight.arch
    from poros.sss import probe

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    store = tmp_path / "nvme"
    store.mkdir()
    probe.cache_path().parent.mkdir(parents=True)
    probe.cache_path().write_text(json.dumps({probe._cache_key(store): {
        "directory": str(store), "b_ssd": 3.2e9, "direct": True, "b_h2d": None,
        "measured_at": time.time(), "b_ssd_qd1": 0.5e9}}))
    for name in ("read_model_config", "cached_model_config"):
        monkeypatch.setattr(poros.plan.configs, name, lambda model, revision=None: QWEN3_30B)
    monkeypatch.setattr(poros.preflight.arch, "resolve_model_type",
                        lambda model: ("qwen3_moe", ""))
    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="offline in tests"))
    (tmp_path / "notes.jsonl").write_text('{"text": "hello"}\n' * 40)
    return store


def _ctx(where) -> Context:
    ctx = Context(cwd=where, fixed_gpu=CARD, fixed_cached=frozenset({MODEL}),
                  fixed_interactive=True)
    ctx.__dict__["host_free_gib"] = 60.0
    return ctx


def _card(disk, extra=""):
    s = parse(f"train {MODEL} --data notes.jsonl --store {disk} {AT} {extra}".strip())
    return Sheet(sheets.train_form(s, _ctx(disk.parent)))


def _row(sheet, key):
    return next(f for f in sheet.view.fields if f.key == key)


def test_the_card_lists_each_ssd_path_and_what_it_is(disk):
    sheet = _card(disk)
    sheet._add("path=", "")
    said = {o.label: o.note for o in _row(sheet, "path=").options()}
    assert list(said) == ["auto → efficient", "efficient", "fast", "fast+cache", "sequential"]
    assert "4G of host RAM" in said["fast+cache"]
    # what each path is; no planner figure for any of them
    assert not any("GiB" in s or "s/update" in s for s in said.values()), said


def test_picking_fast_cache_trains_fast_with_the_host_cache(disk):
    sheet = _card(disk)
    sheet._add("path=", "")
    sheet._set("path=", "fast+cache")
    a, _, _ = sheets.common._resolved("train", sheet.form.values, _ctx(disk.parent))
    assert a.config["path"] == "fast" and a.config["store_options"] == {"cache": "4G"}
    assert a.plan.path == "fast" and "host cache" in a.plan.peak_host_ram.method
    assert "path=fast+cache" in sheet.view.command and "store_options" not in sheet.view.command
    assert _row(sheet, "path=").shown == "fast+cache"
    # another pick drops the cache with it
    sheet._set("path=", "fast")
    a, _, _ = sheets.common._resolved("train", sheet.form.values, _ctx(disk.parent))
    assert a.config["path"] == "fast" and "store_options" not in a.config


def test_a_card_on_the_ram_store_offers_no_cache_and_changes_no_default(disk):
    s = parse(f"train {MODEL} --data notes.jsonl {AT}")
    sheet = Sheet(sheets.train_form(s, _ctx(disk.parent)))
    sheet._add("path=", "")
    assert [o.label for o in _row(sheet, "path=").options()] == [
        "auto → efficient", "efficient", "fast", "sequential"]
    a, _, _ = sheets.common._resolved("train", _card(disk).form.values, _ctx(disk.parent))
    assert a.config.get("path") is None and "store_options" not in a.config  # efficient, no cache
    assert a.plan.path == "efficient" and "host cache" not in a.plan.peak_host_ram.method
    from poros.cli.verbs import train

    with pytest.raises(Refusal, match="fast with the SSD store's host cache"):
        train.resolve(parse(f"train {MODEL} --data notes.jsonl path=fast+cache"),
                      _ctx(disk.parent))


def test_the_ssd_store_gives_each_path_and_what_it_sets(disk):
    ctx = _ctx(disk.parent)
    at = dict(seq_len=512, rank=16, dropout=0.0, store="ssd", store_path=str(disk))
    got = shared.path_plans(MODEL, ctx, config_json=True, **at)
    assert list(got) == ["efficient", "fast", "fast+cache", "sequential"]
    assert shared.path_config("fast+cache") == {"path": "fast", "store_options": {"cache": "4G"}}
    assert shared.path_config("efficient") == {"path": "efficient"}
    for p in got.values():
        assert p.peak_host_ram.value is not None and p.peak_host_ram.source == "estimated"
    assert got["fast+cache"].peak_host_ram.value > got["fast"].peak_host_ram.value + 3.9
    assert shared.plan_for(MODEL, ctx, **at).path == "efficient"
    chose = shared.plan_for(MODEL, ctx, path_settings=shared.path_config("fast+cache"), **at)
    assert chose.path == "fast" and "host cache" in chose.peak_host_ram.method


def test_the_canonical_command_says_fast_cache_as_one_choice(disk):
    sheet = _card(disk, "path=fast+cache")
    a, _, _ = sheets.common._resolved("train", sheet.form.values, _ctx(disk.parent))
    argv = a.canonical()
    assert "path=fast+cache" in argv and not any(x.startswith("store_options") for x in argv)
    assert shared.path_choice(a.config) == "fast+cache"
    from poros.cli.verbs.train_run import config_dict
    from poros.core.config import PorosConfig, ssd_cache_bytes

    cfg = PorosConfig(**config_dict(a, str(disk.parent / "out")))  # what the run trains
    assert cfg.path == "fast" and ssd_cache_bytes(cfg.store_options) == 4 << 30
    assert cfg.store == "ssd" and cfg.store_path == str(disk)
