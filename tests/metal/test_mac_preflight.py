"""The Mac's preflight (poros.preflight.mac): the fit before the download from PorosTrainer and
poros.Run, the refusal's "what would fit", the one pool's memory rule, and the validator's words for
a CUDA-only setting. Torch-free: the Mac is the measured class's plan, read from the table."""

from __future__ import annotations

import pytest

import poros.core.card as card
import poros.core.env as env
import poros.plan.configs as plan_mod
from poros import metal
from poros.core.config import PorosConfig
from poros.core.errors import PorosConfigError, PorosVramError
from poros.core.schemas import Figure
from poros.metal import table
from poros.preflight import fit, mac

REFERENCE = "apple-m4-10gpu-16gb"


@pytest.fixture
def on_the_mac(monkeypatch):
    """This machine is the measured M4 with ``free`` GiB free; config.json says qwen2."""
    state = {"free": 9.0}
    monkeypatch.setattr(card, "detect_gpu", lambda *a, **k: table.plan_gpu(REFERENCE))
    monkeypatch.setattr(env, "host_ram_free", lambda: state["free"])
    monkeypatch.setattr(metal, "device_class", lambda: REFERENCE)
    monkeypatch.setattr(plan_mod, "read_model_config", lambda *a, **k: {"model_type": "qwen2"})
    return state


def _cfg(**kw) -> PorosConfig:
    return PorosConfig(**{"model_name_or_path": "Qwen/Qwen2.5-7B", "dataset": "synthetic", **kw})


def test_a_run_the_mac_plan_refuses_is_refused_before_the_download_with_what_would(on_the_mac):
    with pytest.raises(PorosVramError) as refused:
        mac.before_download(_cfg(max_seq_length=4096))
    assert refused.value.code == "E_OOM_PREDICTED"
    assert "won't fit" in refused.value.info.message
    assert "GiB" not in refused.value.info.message  # the table's floor is not said
    assert refused.value.would.startswith("max_seq_length: 2048 ")
    assert refused.value.info.remediation.startswith(
        "What to change: max_seq_length: 2048 is not refused")


def test_check_fit_false_skips_the_refusal(on_the_mac):
    mac.before_download(_cfg(max_seq_length=4096), check_fit=False)


def test_a_run_that_fits_passes_and_its_config_is_not_changed(on_the_mac):
    cfg = _cfg()
    table.fill(cfg, REFERENCE)  # as metal.require settles it when the trainer is made
    before = cfg.model_dump()
    mac.before_download(cfg)
    assert cfg.model_dump() == before


def test_a_folder_with_no_row_plans_nothing(on_the_mac, tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "gate", lambda *a, **k: pytest.fail("planned a folder"))
    mac.before_download(PorosConfig(model_name_or_path=str(tmp_path), dataset="synthetic"))


def test_would_fit_tries_the_ssd_at_a_length_ram_cannot_hold():
    seen = []

    class Plan:
        def __init__(self, fits):
            self.fits, self.fit_basis = fits, "basis"

    def plan(cfg):
        seen.append((cfg.max_seq_length, cfg.store))
        return Plan(True if cfg.store == "ssd" and cfg.max_seq_length == 256 else False)

    said = mac.would_fit(_cfg(max_seq_length=1024, store="ram"), plan)
    assert said == "max_seq_length: 256, store: ssd is not refused"  # never "fits"
    assert seen == [(512, "ram"), (512, "ssd"), (256, "ram"), (256, "ssd")]
    assert mac.would_fit(_cfg(max_seq_length=64), plan) == ""


def test_one_pool_has_no_host_ram_refusal_and_memory_held_elsewhere_warns(on_the_mac):
    cfg = _cfg(model_name_or_path="Qwen/Qwen2.5-72B", lora_dropout=0.0)  # as its runs measured
    table.fill(cfg, REFERENCE)
    assert mac.host_ram_applies(table.plan_gpu(REFERENCE)) is False
    on_the_mac["free"] = 12.0  # past its measured 11.345 GiB peak
    got = fit.check(cfg, fit.planner_here("qwen2", {"model_type": "qwen2"}))
    assert got.host is None and got.warning is None and got.vram is None
    on_the_mac["free"] = 4.0
    got = fit.check(cfg, fit.planner_here("qwen2", {"model_type": "qwen2"}))
    assert got.host is None
    assert got.warning == ("This Mac has 4.0 GiB free right now, less than a run of "
                           "Qwen/Qwen2.5-72B used when it was measured on this Mac. Close other "
                           "apps, or expect it to swap: the memory guard stops a run whose swap "
                           "grows.")
    assert mac.warning_head(got.p, got.warning) == "memory is short right now"
    off = fit.check(cfg.model_copy(update={"memory_guard": False}),
                    fit.planner_here("qwen2", {"model_type": "qwen2"}))
    assert off.warning.endswith("Close other apps, or expect it to swap.")  # no guard, no claim
    assert mac.warning_head(off.p, off.warning) == "memory is short right now"
    assert mac.warning_head(got.p, "allocated peak, at most 9 GiB, may not fit") == "may not fit"


def test_memory_warning_reads_a_measured_peak_only():
    class P:
        model = "m"
        host_ram_free_gib = 1.0
        peak_vram_allocated = Figure(value=None, unit="GiB", source="unknown", why="x")

    assert mac.memory_warning(P()) is None


@pytest.mark.parametrize("field, value, why, fix", [
    ("path", "efficient", "a Mac trains one micro-batch at a time", "Set path: sequential"),
    ("path", "fast", "every path trains the same update bit for bit", "Set path: sequential"),
    ("fused_compute", True, "the fused kernels are CUDA's (Triton)", "Set fused_compute: false"),
])
def test_a_cuda_only_setting_is_refused_with_why_and_what_to_set(field, value, why, fix):
    with pytest.raises(PorosConfigError) as refused:
        metal.validate(_cfg(**{field: value}))
    assert refused.value.info.message.startswith(f"This Mac cannot honour {field}={value!r}: ")
    assert why in refused.value.info.message
    assert refused.value.info.remediation.startswith(fix)


def test_what_a_mac_runs_is_accepted_as_set():
    metal.validate(_cfg(path="sequential", fused_compute=False))


def test_other_settings_keep_their_words():
    with pytest.raises(PorosConfigError) as refused:
        metal.validate(_cfg(resident_blocks=2))
    assert refused.value.info.remediation == ("Remove these settings. On a Mac, Poros picks them "
                                              "for the model.")
