"""The plan on a Mac: one pool, one verdict, and no figure that crosses from CUDA."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import pytest

import poros.core.env as env
from poros.cli.schemas import PlanGpu
from poros.core.config import PorosConfig
from poros.metal import machine, table
from poros.plan import cells as plan_cells
from poros.plan import make as plan_make
from poros.preflight import fit as preflight_fit

M4 = table.plan_gpu("apple-m4-10gpu-16gb")
RTX3090 = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=None, source="spec")
GB = 1e9 / 2**30
BASE7 = "Qwen/Qwen2.5-7B"
KEY = dict(
    model=BASE7,
    quantization="nf4",
    seq_len=128,
    block_size=1,
    batch=1,
    grad_accum=1,
    lora_rank=16,
    lora_dropout=0.05,
    load_strategy="metal_ssd",
    selfcheck=False,
    entry="dense",
    gpu=M4.name,
)


def figure(value, unit="GiB"):
    return {"value": value, "unit": unit, "receipt": "docs/validation/metal/r.json#/x"}


def cell(**fields):
    return {"id": "mac", "key": KEY, "host": {}, "fields": fields}


def mac_plan(model=BASE7, *, max_steps=200, **config):
    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic", max_steps=max_steps, **config)
    return plan_make.make_plan(cfg, model_type="qwen2", support="validated", entry="dense", gpu=M4)


@pytest.fixture(autouse=True)
def no_cells(monkeypatch):
    """No receipt here but what a test gives: the committed ones change with each set of runs."""
    monkeypatch.setattr(plan_cells, "MAC_CELLS", [])


def test_with_no_receipt_every_figure_says_so_and_nothing_is_refused():
    p = mac_plan("Qwen/Qwen3.6-35B-A3B")
    assert p.fits is None and p.host_fits is None and p.fit_basis == "not measured on this Mac yet"
    for f in (p.peak_vram_allocated, p.gpu_visible, p.s_per_step, p.load_time, p.total_time):
        assert f.value is None and f.source == "unknown"
    assert p.gpu.unified and p.store == "ssd" and p.bound is None


def test_the_one_pool_has_one_peak():
    p = mac_plan()
    assert p.peak_vram_allocated.why == "not measured on this Mac yet"
    for f in (p.peak_vram_reserved, p.peak_host_ram):
        assert f.value is None and f.why == "one pool: the peak above"


def test_a_plan_reads_the_table_never_the_linux_defaults():
    p = mac_plan("Qwen/Qwen2.5-72B")
    assert p.store == "ssd"
    assert any("128 tokens" in w and "read from disk" in w for w in p.warnings)
    assert not any("streams MoE" in w or "receipt" in w for w in p.warnings)


def test_a_length_off_the_table_says_so():
    assert "no measured time for this length" in mac_plan(max_seq_length=1024).warnings
    assert "no measured time for this length" not in mac_plan(max_seq_length=512).warnings


def test_the_floor_is_the_only_thing_that_says_wont_fit_unmeasured():
    p = mac_plan("Qwen/Qwen2.5-72B", max_seq_length=4096)
    assert p.fits is False and p.host_fits is None
    assert "23.3 GiB" in p.fit_basis and "3.3 GiB for macOS" in p.fit_basis
    assert preflight_fit.host_ram_error(p) is None  # one pool, one refusal: E_OOM_PREDICTED


def test_a_model_with_no_row_is_unsupported_not_unknown():
    p = mac_plan("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    assert p.support == "unsupported" and p.fits is None and p.entry is None
    assert "does not train on a Mac" in p.fit_basis


def test_a_config_the_mac_cannot_honour_plans_as_unsupported_not_measured(monkeypatch):
    monkeypatch.setattr(
        plan_cells, "MAC_CELLS",
        [cell(peak_vram_allocated=figure(9.0), gpu_visible=figure(7.0))],
    )
    p = mac_plan(lora_bias="all")
    assert p.support == "unsupported" and p.fits is None and p.peak_vram_allocated.value is None
    assert "lora_bias" in p.fit_basis


def test_eager_attention_is_not_off_default_on_a_mac(monkeypatch):
    monkeypatch.setattr(
        plan_cells, "MAC_CELLS",
        [cell(peak_vram_allocated=figure(9.0), gpu_visible=figure(7.0))],
    )
    assert mac_plan(attn_implementation="eager").peak_vram_allocated.value == 9.0


@pytest.mark.parametrize(
    "peak, visible, fits",
    [
        (9.0, 7.0, True),
        (12.74, 11.84, True),  # at the edge of both: 16 GiB less 3.5 GB, and the working set
        (12.75, 7.0, False),  # the pool
        (9.0, 11.85, False),  # the working set, measured at 11.84 GiB
    ],
)
def test_a_receipt_on_this_class_decides(monkeypatch, peak, visible, fits):
    monkeypatch.setattr(
        plan_cells, "MAC_CELLS", [cell(peak_vram_allocated=figure(peak),
                                       gpu_visible=figure(visible))]
    )
    p = mac_plan()
    assert p.fits is fits and p.host_fits is None
    assert p.peak_vram_allocated.source == "measured" and p.host_fit_basis == "one pool"


def test_time_is_measured_or_unknown_and_the_total_is_for_the_steps_asked(monkeypatch):
    monkeypatch.setattr(
        plan_cells, "MAC_CELLS",
        [
            cell(
                s_per_step=figure(7.0, "s"),
                load_time=figure(40.0, "s"),
            )
        ],
    )
    p = mac_plan(max_steps=300)
    assert p.s_per_step.value == 7.0 and p.total_time.value == 40.0 + 300 * 7.0
    assert mac_plan(max_steps=-1).total_time.source == "unknown"


def test_another_device_class_never_reads_this_ones_cell(monkeypatch):
    monkeypatch.setattr(
        plan_cells, "MAC_CELLS", [cell(peak_vram_allocated=figure(9.0), gpu_visible=figure(7.0))]
    )
    other = table.plan_gpu("apple-m4-10gpu-24gb")  # it has a derived row: only the cell is at issue
    cfg = PorosConfig(model_name_or_path=BASE7, dataset="synthetic")
    table.fill(cfg, table.class_of(M4.name))
    p = plan_make.make_plan(cfg, model_type="qwen2", support="validated", entry="dense", gpu=other)
    assert p.peak_vram_allocated.value is None


def test_a_figure_never_crosses_between_cuda_and_a_mac():
    assert not any(c["key"]["gpu"].startswith("Apple") for c in plan_cells.CELLS)
    assert all(c["key"]["gpu"].startswith("Apple") for c in plan_cells.MAC_CELLS)


def test_host_ram_free_reads_vm_stat_on_darwin(monkeypatch):
    out = textwrap.dedent(
        """\
        Mach Virtual Memory Statistics: (page size of 16384 bytes)
        Pages free:                                     5390.
        Pages active:                                 372021.
        Pages inactive:                               371235.
        Pages speculative:                               182.
        Pages wired down:                             107465.
        """
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(machine.metal, "run_tool", lambda *_: out)
    assert env.host_ram_free() == round((5390 + 371235 + 182) * 16384 / 2**30, 2)
    monkeypatch.setattr(machine.metal, "run_tool", lambda *_: "")
    assert env.host_ram_free() is None


def test_nothing_about_a_macs_gpu_imports_torch():
    """In a fresh process, as `machine` and `fit` run: the Darwin branch comes first."""
    script = textwrap.dedent(
        """\
        import sys
        import platform
        sys.modules["torch"] = None  # `import torch` anywhere raises
        from poros import metal
        OUT = {
            ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4",
            ("/usr/sbin/sysctl", "-n", "hw.memsize"): "17179869184",
            ("/usr/sbin/ioreg", "-rc", "AGXAccelerator", "-d", "1"): '"gpu-core-count" = 10',
        }
        metal.run_tool = lambda *c: OUT.get(c, "")
        platform.machine = lambda: "arm64"
        from poros.core import card
        from poros.plan import cells as plan_cells
        from poros.plan import make as plan_make
        from poros.preflight import fit as preflight_fit
        sys.platform = "darwin"
        gpu = card.detect_gpu()
        assert gpu.name == "Apple M4 10-core GPU 16GB" and gpu.unified, gpu
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.stdout.strip() == "ok", done.stderr


def test_a_moe_model_plans_on_its_own_row():
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B-Instruct-2507", dataset="synthetic")
    p = plan_make.make_plan(cfg, model_type="qwen3_moe", support="validated", entry="moe", gpu=M4)
    assert p.entry == "moe" and p.store == "ssd"


def test_only_a_gate_on_the_whole_model_says_validated(monkeypatch):
    real = table.fill
    for evidence, support in (("validated", "validated"), ("checked", "validated"),
                              ("measured", "unknown"), ("none", "unknown")):
        monkeypatch.setattr(table, "fill", lambda cfg, c, e=evidence: dataclasses.replace(
            real(cfg, c), evidence=e))
        assert mac_plan().support == support, evidence
