"""A dense NF4 run's resident blocks in its plan (poros.plan.make._resident_hold): no receipt ran
any, so the peaks grow by the resident layers' NF4 bytes and are estimates; what the run certainly
holds -- those layers, the fp32 tables on the GPU, the adapters' training state -- is a floor that
refuses before the download when it passes the card."""

from __future__ import annotations

import pytest

from poros.core.card import detect_gpu
from poros.core.config import PorosConfig
from poros.plan.configs import read_model_config
from poros.preflight.fit import run_planner

MODEL = "Qwen/Qwen2.5-72B"


def _plan(gpu="RTX 4090", **kw):
    mc = read_model_config(MODEL, None)
    if mc is None:
        pytest.skip("no packaged config.json sizes for the 72B")
    plan = run_planner("qwen2", gpu=detect_gpu(gpu), model_config=mc, host_free_gib=256.0)
    return plan(PorosConfig(model_name_or_path=MODEL, dataset="synthetic", quantization="nf4",
                            max_seq_length=512, **kw))


def test_resident_blocks_add_their_bytes_and_make_the_peak_an_estimate():
    streamed = _plan(gpu="A100 80GB PCIe", path="sequential")
    kept = _plan(gpu="A100 80GB PCIe", path="sequential", resident_blocks=2)
    a, b = streamed.peak_vram_allocated, kept.peak_vram_allocated
    assert a.value is not None and b.value is not None
    assert b.value > a.value and b.source == "estimated" and "8 resident layers" in b.method


def test_a_floor_past_the_card_refuses_and_says_how_many_blocks_fit():
    p = _plan(resident_blocks=999, path="fast")
    assert p.fits is False and "80 resident layers" in p.fit_basis
    assert "resident_blocks=" in p.fit_basis and "or fewer leaves room" in p.fit_basis


def test_a_floor_under_the_card_never_refuses_by_itself():
    p = _plan(gpu="A100 80GB PCIe", resident_blocks=2, path="sequential")
    assert p.fits is not False
