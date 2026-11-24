"""Every Apple silicon Mac, simulated: the tools that read the machine are faked, and the rest is
the code a real Mac runs. Each class plans every catalog model; the measured class is as it was."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform

import pytest

from poros import metal
from poros.cli.words.models import MetalRow, catalog
from poros.core.config import PorosConfig
from poros.metal import machine, table, watchdog
from poros.metal.plan import unified_plan

REFERENCE = "apple-m4-10gpu-16gb"
# (chip as sysctl says it, GPU cores, memory in GB): the Apple silicon Macs simulated here.
MACS = [
    ("Apple M1", 8, 8), ("Apple M1", 8, 16),
    ("Apple M1 Pro", 16, 16), ("Apple M1 Pro", 16, 32),
    ("Apple M1 Max", 32, 32), ("Apple M1 Max", 32, 64),
    ("Apple M1 Ultra", 64, 64), ("Apple M1 Ultra", 64, 128),
    ("Apple M2", 10, 8), ("Apple M2", 10, 16), ("Apple M2", 10, 24),
    ("Apple M2 Pro", 19, 16), ("Apple M2 Pro", 19, 32),
    ("Apple M2 Max", 38, 32), ("Apple M2 Max", 38, 96),
    ("Apple M2 Ultra", 76, 192),
    ("Apple M3", 10, 8), ("Apple M3", 10, 16), ("Apple M3", 10, 24),
    ("Apple M3 Pro", 18, 18), ("Apple M3 Pro", 18, 36),
    ("Apple M3 Max", 40, 36), ("Apple M3 Max", 40, 128),
    ("Apple M4", 10, 16), ("Apple M4", 10, 24), ("Apple M4", 10, 32),
    ("Apple M4 Pro", 20, 24), ("Apple M4 Pro", 20, 48), ("Apple M4 Pro", 20, 64),
    ("Apple M4 Max", 40, 36), ("Apple M4 Max", 40, 128),
]
IDS = [f"{chip.removeprefix('Apple ')} {gb}GB".replace(" ", "-") for chip, _, gb in MACS]
MODELS = [e.id for e in catalog() if table.rows(e.id, REFERENCE)]
# Everything a run on the measured class reads from the table and its plan, digested the same way
# when it was recorded, before other Macs were supported. It must not drift.
REFERENCE_DIGEST = "774a46a97a51bfc0"
BASE_ROW_FIELDS = ("device_class", "seq", "tier", "grad_accum", "save_steps", "synthetic_steps",
                   "evidence", "default", "method", "receipt", "macos_build", "derived")
# Config fields added after the digest was recorded: CUDA's own (block-major NF4, MoE, the fused NF4
# compute, and Inkling 2 Mini's moe_expert_format), which the Mac validator holds at their
# defaults, and eval_checkpoints, a recipe's scoring, which plans nothing.
# A settled config is digested without them, as it was when recorded.
LATER_FIELDS = ("cuda_expandable_segments", "fill_uninitialized_memory", "moe_expert_cache_gib",
                "moe_expert_compute", "moe_expert_grouping", "nf4_dequant_cache_mb",
                "offload_boundaries", "offload_embeddings_and_head", "resident_blocks",
                "eval_checkpoints", "group_by_length", "path", "path_reason", "ran_path",
                "fused_compute", "ran_compute", "compute_reason", "eval_rows",
                "moe_expert_format")
# Config fields since removed, at the value every Mac settled them to: no Mac ran block-major
# accumulation (``path`` now says the loop).
REMOVED_FIELDS = {"block_major_accumulation": False}
# Plan fields added after the digest was recorded: path, the CUDA loop a plan assumed, None on a
# Mac.
LATER_PLAN_FIELDS = ("path",)


def reference_digest() -> str:
    """The measured class's rows, settled configs and plans, for every catalog model. The same
    function, when the digest was recorded, gave ``REFERENCE_DIGEST``."""
    gpu = table.plan_gpu(REFERENCE)
    out = [gpu.model_dump(mode="json")]
    for e in catalog():
        rows = [{f: getattr(r, f) for f in BASE_ROW_FIELDS} for r in table.rows(e.id, REFERENCE)]
        cfg = PorosConfig(model_name_or_path=e.id, dataset="synthetic")
        plan = unified_plan(cfg, model_type=e.arch, gpu=gpu, download=None, host_free_gib=None)
        config = {**{k: v for k, v in cfg.model_dump(mode="json").items()
                     if k not in LATER_FIELDS}, **REMOVED_FIELDS}
        assert all(getattr(plan, k) is None for k in LATER_PLAN_FIELDS)
        planned = {k: v for k, v in plan.model_dump(mode="json").items()
                   if k not in LATER_PLAN_FIELDS}
        out.append({"id": e.id, "rows": rows, "config": config, "plan": planned,
                    "floor": [table.floor_gb(e.id, s) for s in (64, 128, 256, 512)]
                    if rows else None})
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]


@pytest.fixture
def simulate(monkeypatch, tmp_path):
    """``simulate(chip, cores, gb)``: this machine is that Mac, as sysctl, ioreg and Metal say."""
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(metal.DEV_MEMORY_ENV, raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    def to(chip: str, cores: int, gb: int) -> str:
        tools = {
            ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"): chip,
            ("/usr/sbin/sysctl", "-n", "hw.memsize"): str(gb * 2**30),
            ("/usr/sbin/ioreg", "-rc", "AGXAccelerator", "-d", "1"): f'"gpu-core-count" = {cores}',
            ("/usr/bin/sw_vers", "-buildVersion"): "26B123",
        }
        monkeypatch.setattr(metal, "run_tool", lambda *cmd: tools.get(cmd, ""))
        # What macOS gives this Mac's GPU: the reference measured, others as reported.
        given = (table.WORKING_SET_BYTES[REFERENCE] if (chip, gb) == ("Apple M4", 16)
                 else table.estimated_working_set(gb * 2**30))
        monkeypatch.setattr(machine, "gpu_working_set", lambda: given)
        metal.device_class.cache_clear()
        return metal.device_class()

    yield to
    metal.device_class.cache_clear()


def _plan(model: str, here: str):
    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic")
    entry = next(e for e in catalog() if e.id == model)
    return cfg, unified_plan(cfg, model_type=entry.arch, gpu=table.plan_gpu(here), download=None,
                             host_free_gib=None)


@pytest.mark.parametrize("chip, cores, gb", MACS, ids=IDS)
def test_every_mac_plans_every_model(simulate, chip, cores, gb):
    here = simulate(chip, cores, gb)
    name = chip.lower().replace(" ", "-")
    assert here == f"{name}-{cores}gpu-{gb}gb"
    metal.require_class(here)
    room = table.budget(here)
    for model in MODELS:
        cfg, plan = _plan(model, here)
        rows = table.rows(model, here)
        if here == REFERENCE:
            assert all(r.evidence != "derived" and not r.planned_from for r in rows)
            continue
        if not rows:  # refused: and only because even the SSD cannot hold it
            why = table.refusal(model, here)
            assert "even reading its weights from the SSD" in why, (model, here)
            assert plan.support == "unsupported" and plan.fits is None and plan.fit_basis == why
            need, gpu = table.need_gb(model, 64, "ssd"), table.gpu_gb(model, 64, "ssd")
            assert need > room.pool or (gpu is not None and gpu > room.gpu), (model, here)
            continue
        seqs = [r.seq for r in rows]
        assert seqs == sorted(seqs) and set(seqs) <= set(table.LENGTHS)
        assert sum(r.default for r in rows) == 1
        for r in rows:
            assert (r.evidence, r.planned_from, r.device_class) == ("derived", REFERENCE, here)
            assert r.tier in ("ram", "ssd") and r.grad_accum == 1 and r.save_steps > 0
            assert not r.receipt and not r.macos_build and not r.method
            if r.tier == "ram":
                assert table.need_gb(model, r.seq, "ram") <= room.ram
            else:  # RAM cannot hold it, or the reference read it from the SSD and this Mac has
                # no more room than the reference
                assert (table.need_gb(model, r.seq, "ram") > room.ram
                        or room.ram <= table.budget(REFERENCE).ram)
                assert table.need_gb(model, r.seq, "ssd") <= room.pool
        default = next(r for r in rows if r.default)
        assert (cfg.max_seq_length, cfg.store) == (default.seq, default.tier)
        assert plan.support == "unknown" and plan.fits is None, (model, here, plan.fit_basis)
        assert plan.warnings[0].startswith("a row derived for this Mac's memory")
        assert "not measured here" in plan.fit_basis
        for f in (plan.peak_vram_allocated, plan.gpu_visible, plan.s_per_step, plan.total_time):
            assert f.value is None and f.source == "unknown" and "derived for this Mac" in f.why
        assert table.evidence(model, here) == "derived"


def test_more_memory_never_trains_less(simulate):
    """Across the matrix: a Mac with more memory trains every model a smaller one does, and holds
    in RAM every one the smaller one holds there."""
    trained = {}
    for chip, cores, gb in MACS:
        here = simulate(chip, cores, gb)
        trained[gb] = {m: {r.tier for r in table.rows(m, here)} for m in MODELS}
    sizes = sorted(trained)
    for small, big in zip(sizes, sizes[1:]):
        for m in MODELS:
            assert not trained[small][m] or trained[big][m], (m, small, big)
            assert "ram" not in trained[small][m] or "ram" in trained[big][m], (m, small, big)


def test_what_the_smallest_and_largest_macs_train(simulate):
    """The matrix's ends, in plain numbers: an 8 GB M1 trains the small models and 7B from the SSD,
    and refuses only what even the SSD cannot hold; a 192 GB M2 Ultra holds every model in RAM."""
    small = simulate("Apple M1", 8, 8)
    tiers = {m: {r.tier for r in table.rows(m, small)} for m in MODELS}
    assert tiers["Qwen/Qwen2.5-0.5B"] == {"ram"} and tiers["Qwen/Qwen3-0.6B"] == {"ram"}
    assert tiers["Qwen/Qwen2.5-7B"] == {"ssd"}
    assert tiers["Qwen/Qwen2.5-72B"] == set() and tiers["Qwen/Qwen2.5-32B"] == set()
    big = simulate("Apple M2 Ultra", 76, 192)
    for m in MODELS:
        rows = table.rows(m, big)
        assert rows and {r.tier for r in rows} == {"ram"}, m
        assert max(r.seq for r in rows) == 512  # a larger Mac takes the table's longer lengths
        default = next(r for r in rows if r.default)
        assert default.seq == table.pick(m, REFERENCE).seq  # and the reference's default


def test_the_measured_class_is_byte_for_byte_as_before(simulate):
    assert simulate("Apple M4", 10, 16) == REFERENCE
    assert reference_digest() == REFERENCE_DIGEST


def test_the_measured_class_keeps_its_figures():
    memory = 16 * 2**30
    assert table.os_reserve_gb(memory) == 3.5 == table.OS_RESERVE_GB
    assert table.working_set(REFERENCE) == (12_713_115_648, "measured")
    assert watchdog.swap_stop_bytes(memory) == 2_000_000_000
    assert watchdog.boot_stop_bytes(memory) == 3_000_000_000
    assert watchdog.spill_bytes(memory) == 250_000_000
    assert metal.watermarks() == ("1.05", "0.95")


@pytest.mark.parametrize("gb, reserve, swap", [
    (8, 2.5, 1_000_000_000), (16, 3.5, 2_000_000_000), (24, 5.25, 3_000_000_000),
    (128, 28.0, 16_000_000_000), (192, 42.0, 24_000_000_000),
])
def test_the_swap_stop_scales_with_memory_and_the_disk_floor_does_not(gb, reserve, swap):
    """A floor of a share of memory would stop a 192 GB Mac with 20 GB free before swap grew."""
    memory = gb * 2**30
    assert table.os_reserve_gb(memory) == pytest.approx(reserve)
    assert watchdog.swap_stop_bytes(memory) == swap
    assert watchdog.boot_stop_bytes(memory) == 3_000_000_000


def test_a_large_mac_with_a_fuller_disk_trains_on(simulate, monkeypatch):
    simulate("Apple M2 Ultra", 76, 192)
    monkeypatch.setattr(machine, "swap_used", lambda: 0)
    monkeypatch.setattr(machine, "pressure", lambda: 1)
    monkeypatch.setattr(watchdog, "_boot_free", lambda: 20_000_000_000)
    dog = watchdog.Watchdog(guard=False)
    try:
        assert dog._over() == ""
        monkeypatch.setattr(watchdog, "_boot_free", lambda: 2_500_000_000)
        assert dog._over().startswith("The boot disk")
    finally:
        dog.stop()


def test_a_watchdog_reads_its_figures_from_this_macs_memory(simulate, monkeypatch):
    simulate("Apple M2", 10, 8)
    monkeypatch.setattr(machine, "swap_used", lambda: 0)
    dog = watchdog.Watchdog(guard=True)
    try:
        assert (dog.swap_stop, dog.boot_stop) == (1_000_000_000, 3_000_000_000)
    finally:
        dog.stop()


def test_a_derived_row_is_a_metal_row_with_one_field_more():
    names = {f.name for f in dataclasses.fields(MetalRow)}
    assert names - set(BASE_ROW_FIELDS) == {"planned_from"}


@pytest.mark.parametrize("mib", [12200, 13312, 14336])
def test_a_16_gb_mac_whose_gpu_reports_more_still_gets_the_reference_rows(simulate, monkeypatch,
                                                                          mib):
    """A newer macOS, or a raised iogpu.wired_limit_mb, reads a larger working set: more room on
    the GPU is not more memory, and the reference chose the SSD where RAM swapped."""
    here = simulate("Apple M2", 10, 16)
    monkeypatch.setattr(machine, "gpu_working_set", lambda: mib * 2**20)
    assert table.working_set(here) == (mib * 2**20, "read")
    for m in MODELS:
        assert ([(r.seq, r.tier) for r in table.rows(m, here)]
                == [(r.seq, r.tier) for r in table.rows(m, REFERENCE)]), m


def test_a_mac_whose_memory_cannot_be_read_is_not_told_a_model_does_not_fit():
    assert table.refusal("Qwen/Qwen2.5-0.5B", "apple-m4-10gpu-0gb") == ""


@pytest.mark.parametrize("chip, cores", [("Apple M1", 8), ("Apple M2", 10), ("Apple M2 Pro", 19),
                                         ("Apple M3", 10), ("Apple M4", 8)])
def test_a_16_gb_mac_of_any_chip_gets_the_reference_rows(simulate, chip, cores):
    """Same memory, same table: only the evidence says derived, and its speed is unknown."""
    here = simulate(chip, cores, 16)
    assert here != REFERENCE

    def shape(rows):
        return sorted((r.seq, r.tier, r.default, r.grad_accum, r.save_steps, r.synthetic_steps)
                      for r in rows)

    for m in MODELS:
        assert shape(table.rows(m, here)) == shape(table.rows(m, REFERENCE)), m
