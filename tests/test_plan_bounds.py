"""poros plan bounds: a verdict from a receipt that dominates, or is dominated
by, the config asked for. Never across models, never interpolated."""

from __future__ import annotations

import pytest

from poros.cli.schemas import PlanGpu
from poros.plan import cells as plan_cells
from poros.plan import make as plan_make
from poros.plan import receipts as plan_receipts

KEY = dict(model="M", quantization="nf4", seq_len=512, block_size=4, batch=1, grad_accum=8,
           lora_rank=16, lora_dropout=0.0, load_strategy="gpu", selfcheck=True, entry="dense",
           gpu="NVIDIA GeForce RTX 3090")


def cell(id_, peak, reserved=None, **key):
    fields = {"peak_vram_allocated": {"value": peak, "unit": "GiB", "receipt": "r#/a"}}
    if reserved is not None:
        fields["peak_vram_reserved"] = {"value": reserved, "unit": "GiB", "receipt": "r#/b"}
    return {"id": id_, "key": KEY | key, "host": {}, "fields": fields}


def req(**kw):
    base = {k: v for k, v in KEY.items() if k != "gpu"}
    return plan_receipts.Request(**(base | kw))


GPU24 = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=None, source="spec")
RTX5090 = PlanGpu(name="NVIDIA GeForce RTX 5090", total_gib=31.84, free_gib=None, source="spec")
RTX4090 = PlanGpu(name="NVIDIA GeForce RTX 4090", total_gib=23.99, free_gib=None, source="spec")


@pytest.fixture
def cells(monkeypatch):
    def set_(*cs):
        monkeypatch.setattr(plan_cells, "CELLS", list(cs))
    return set_


def test_a_larger_completed_run_proves_a_fit(cells):
    cells(cell("big", 10.0, reserved=20.0, seq_len=1024, lora_dropout=0.05))
    b = plan_receipts.bound(req(seq_len=512), GPU24)
    assert b.fits is True and b.receipt == "big" and "reserved at most 20.0" in b.basis
    assert b.differs == {"seq_len": [512, 1024], "lora_dropout": [0.0, 0.05]}


def test_a_smaller_run_over_budget_proves_a_misfit(cells):
    cells(cell("small", 30.0, seq_len=256))
    b = plan_receipts.bound(req(seq_len=2048), GPU24)
    assert b.fits is False and b.peak_gib == 30.0


def test_in_between_is_unknown(cells):
    # Larger on seq, smaller on rank: dominates in neither direction.
    cells(cell("mixed", 10.0, reserved=20.0, seq_len=4096, lora_rank=8))
    assert plan_receipts.bound(req(seq_len=512, lora_rank=16), GPU24) is None


def test_never_across_models(cells):
    cells(cell("other", 1.0, reserved=2.0, model="Other", seq_len=8192))
    assert plan_receipts.bound(req(), GPU24) is None


def test_reserved_over_free_proves_nothing(cells):
    cells(cell("big", 10.0, reserved=26.0, seq_len=1024))
    assert plan_receipts.bound(req(), GPU24) is None


def test_a_streamed_load_receipt_does_not_bound_a_gpu_load(cells):
    # A gpu load can leave more cached than the streamed load the receipt ran.
    cells(cell("streamed", 10.0, reserved=20.0, seq_len=1024, load_strategy="block_streamed"))
    assert plan_receipts.bound(req(load_strategy="gpu"), GPU24) is None
    assert plan_receipts.bound(req(load_strategy="block_streamed"), GPU24).fits is True


def test_unknown_selfcheck_does_not_bound_a_selfcheck_run(cells):
    cells(cell("maybe", 10.0, reserved=20.0, seq_len=1024, selfcheck=None))
    assert plan_receipts.bound(req(selfcheck=True), GPU24) is None
    assert plan_receipts.bound(req(selfcheck=False), GPU24).fits is True


def test_make_plan_uses_the_bound_only_without_an_exact_receipt(cells):
    """No reserved peak to decide on (the larger run's figures bound the allocated alone): the
    verdict is the bound's."""
    from poros.core.config import PorosConfig

    cells(cell("big", 6.0, model="Qwen/Qwen2.5-7B", seq_len=1024))
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-7B", dataset="synthetic",
                      max_seq_length=512, lora_dropout=0.0, nf4_load_strategy="gpu",
                      startup_parity_selfcheck="always", gradient_accumulation_steps=8,
                      path="sequential", fused_compute=False)  # the compute the cell ran
    p = plan_make.make_plan(cfg, model_type="qwen2", support="validated", entry="dense", gpu=GPU24)
    assert p.fits is True and p.bound is not None and p.fit_basis.startswith("bounded:")


def test_an_exact_receipt_on_this_gpu_is_never_overridden(cells):
    # The case: the exact config measured here fits by allocated,
    # a same-knob run elsewhere peaked a hair higher. No bound either way.
    cells(cell("exact", 6.19, load_strategy="block_streamed"),
          cell("other", 6.21, load_strategy="gpu", gpu="NVIDIA GeForce RTX 5090"))
    tight = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=6.2, source="torch")
    assert plan_receipts.bound(req(load_strategy="block_streamed"), tight) is None


def test_a_config_measured_twice_counts_by_its_lowest_peak(cells):
    cells(cell("synthetic", 5.4, seq_len=256), cell("oasst1", 6.7, seq_len=256))
    six = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=6.0, source="torch")
    assert plan_receipts.bound(req(seq_len=512), six) is None


def test_wont_fit_needs_more_than_the_cross_gpu_tolerance(cells):
    cells(cell("small", 8.42, seq_len=256))
    gpu = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=8.41, source="torch")
    assert plan_receipts.bound(req(seq_len=512), gpu) is None


def test_reserved_counts_only_on_its_own_gpu(cells):
    cells(cell("big", 10.0, reserved=12.0, seq_len=1024))  # on a 3090
    # On a 4090 the 3090's reserved peak says nothing; its card (24.0) > 23.99.
    assert plan_receipts.bound(req(), RTX4090) is None
    # On a 5090 the run's whole 24 GiB card fits in 31.84: it completed within it.
    b = plan_receipts.bound(req(), RTX5090)
    assert b.fits is True and "completed within" in b.basis


def test_a_wsl2_run_proves_nothing_beyond_its_own_gpu(cells):
    c = cell("wsl", 10.0, seq_len=1024)
    c["host"] = {"os": "WSL2"}
    cells(c)
    assert plan_receipts.bound(req(), RTX5090) is None
