"""PorosTrainer.from_pretrained's refusals before the download: the CLI gate's, in its order.

The architecture first, then the card's compute (a set fused compute at a rank its kernels do not
fit), then the GPU fit and the host RAM, each on its own flag; poros.Run passes the flags on.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")


from poros.core import card  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import (  # noqa: E402
    PorosArchNotValidatedError,
    PorosConfigError,
    PorosHostRamError,
    PorosVramError,
)
from poros.plan import configs as plan_configs  # noqa: E402
from tests.test_plan_upper import CONFIGS, QWEN3_30B, RTX4090  # noqa: E402


def _card(monkeypatch, cap=(8, 9), name="NVIDIA GeForce RTX 4090"):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: cap)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda device=None: name)
    monkeypatch.setattr("torch.cuda.get_arch_list", lambda: ["sm_80", "sm_86", "sm_89"])


def test_a_dense_nf4_run_past_host_ram_is_refused_by_the_trainer_too(monkeypatch):
    """The 72B's frozen layers in NF4 take 33.7 GiB of host RAM: on a 31 GB host the trainer,
    like `poros train`, refuses before the 145 GB download (check_host_ram=False goes on)."""
    from poros.core import compute, env
    from poros.preflight import fit as F

    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: RTX4090.model_copy(
        update={"free_gib": 23.5, "source": "torch"}))
    monkeypatch.setattr(compute, "triton_installed", lambda: True)
    monkeypatch.setattr(env, "host_ram_free", lambda: 30.35)
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                      max_seq_length=512, lora_dropout=0.0)
    fit = F.check(cfg, F.planner_here("qwen2", CONFIGS["Qwen/Qwen2.5-72B"]))
    with pytest.raises(PorosHostRamError) as err:
        F.refuse(fit)
    said = err.value.info.message
    assert said.startswith("Qwen/Qwen2.5-72B needs more host RAM than the 30.35 GiB available")
    assert "frozen layers" in said and "33." not in said  # the plan's figure is not said
    assert "None" not in said and err.value.code == "E_HOST_OOM_PREDICTED"
    F.refuse(fit, check_host_ram=False)
    # the same refusal word for word as the CLI's gate reads it (poros.preflight.fit)
    assert F.host_ram_error(fit.plan(cfg)).info.message == err.value.info.message


def test_each_check_runs_on_its_own_flag(monkeypatch):
    """check_host_ram=False never turns the GPU fit off, nor check_fit=False the host RAM."""
    from poros.core import compute, env
    from poros.preflight import fit as F

    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: RTX4090.model_copy(
        update={"free_gib": 6.0, "source": "torch"}))
    monkeypatch.setattr(compute, "triton_installed", lambda: True)
    monkeypatch.setattr(env, "host_ram_free", lambda: 30.35)
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                      max_seq_length=512, lora_dropout=0.0)
    fit = F.check(cfg, F.planner_here("qwen2", CONFIGS["Qwen/Qwen2.5-72B"]))
    with pytest.raises(PorosVramError):
        F.refuse(fit, check_host_ram=False)
    with pytest.raises(PorosHostRamError):
        F.refuse(fit, check_fit=False)
    F.refuse(fit, check_fit=False, check_host_ram=False)


def test_a_moe_run_checks_the_fit_with_the_host_ram_check_off(monkeypatch):
    """_setup_moe adapts the path and runs each refusal on its own flag (check_host_ram=False
    used to skip the GPU fit, and the path's adaptation, too)."""
    from poros.train import trainer as T

    seen = {}

    class Refused(Exception):
        pass

    def refuse(fit, *, check_fit=True, check_host_ram=True):
        seen.update(check_fit=check_fit, check_host_ram=check_host_ram)
        raise Refused

    from poros.preflight import fit as F

    monkeypatch.setattr(F, "refuse", refuse)
    monkeypatch.setattr(plan_configs, "read_model_config", lambda *a, **k: QWEN3_30B)
    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: RTX4090)
    monkeypatch.setattr(F, "decide_path_to_fit",
                        lambda cfg, plan: seen.setdefault("path", 1) and None)
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic")
    t = T.PorosTrainer(cfg)
    t._metal = False
    with pytest.raises(Refused):
        t._setup_moe(check_host_ram=False, check_fit=True)
    assert seen == {"path": 1, "check_fit": True, "check_host_ram": False}


def test_an_unvalidated_dense_model_hears_the_architecture_gate_before_the_fit(monkeypatch):
    """A dense model_type no one validated is told so (PorosArchNotValidatedError), not handed a
    fit's remedy for a run that would be refused anyway."""
    from poros.train import trainer as T

    _card(monkeypatch)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    phi = dict(CONFIGS["Qwen/Qwen2.5-72B"], model_type="phi3")
    monkeypatch.setattr(plan_configs, "read_model_config", lambda *a, **k: phi)
    from poros.preflight import fit as F

    monkeypatch.setattr(F, "gate", lambda *a, **k: pytest.fail("planned the fit first"))
    with pytest.raises(PorosArchNotValidatedError):
        T.PorosTrainer.from_pretrained("someone/phi-like", dataset="synthetic")


def _compute(monkeypatch, cfg, cap=(8, 9)):
    from poros.core import compute
    from poros.train.compute import RunCompute

    _card(monkeypatch, cap)
    monkeypatch.setattr(compute, "triton_installed", lambda: True)
    return RunCompute(types.SimpleNamespace(config=cfg, device=torch.device("cuda", 0)))


@pytest.mark.parametrize("cap, rank, refused", [
    ((8, 9), 128, False), ((8, 9), 256, True), ((12, 0), 256, True), ((8, 6), 256, True),
    ((9, 0), 256, False), ((8, 0), 256, False)])
def test_a_set_fused_compute_at_a_rank_the_card_cannot_fit_is_refused_before_the_download(
        monkeypatch, cap, rank, refused):
    """fused_compute: true at rank 256 on a 99 KB card (RTX 3090 / 4090 / 5090): the shared-memory
    table says no before the download, not install_dense after the load."""
    pytest.importorskip("peft")
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-7B", dataset="synthetic",
                      fused_compute=True, lora_rank=rank, lora_alpha=2 * rank)
    rc = _compute(monkeypatch, cfg, cap)
    if not refused:
        rc.preflight("dense", CONFIGS["Qwen/Qwen2.5-7B"], "qwen2")
        return
    with pytest.raises(PorosConfigError) as err:
        rc.preflight("dense", CONFIGS["Qwen/Qwen2.5-7B"], "qwen2")
    assert "rank 256" in str(err.value) and "lora_rank: 128" in err.value.info.remediation


@pytest.mark.parametrize("setting", [{"moe_expert_compute": "fused"}, {"fused_compute": True,
                                                                       "moe_expert_compute":
                                                                       "fused"}])
def test_a_set_fused_expert_compute_at_a_rank_the_card_cannot_fit_is_refused(monkeypatch,
                                                                              setting):
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic",
                      lora_rank=256, lora_alpha=512, **setting)
    with pytest.raises(PorosConfigError) as err:
        _compute(monkeypatch, cfg).preflight("moe", QWEN3_30B, "qwen3_moe")
    assert "rank 256" in str(err.value) and "lora_rank: 128" in err.value.info.remediation
    ok = cfg.model_copy(update={"lora_rank": 128, "lora_alpha": 256})
    _compute(monkeypatch, ok).preflight("moe", QWEN3_30B, "qwen3_moe")
    # the default (unset) adapts after the load instead: the reference experts, no refusal
    unset = PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic",
                        lora_rank=256, lora_alpha=512)
    _compute(monkeypatch, unset).preflight("moe", QWEN3_30B, "qwen3_moe")


@pytest.mark.parametrize("flag", ["check_fit", "check_host_ram"])
def test_run_create_passes_the_checks_on(monkeypatch, flag):
    """The refusal's remedy names check_fit / check_host_ram: poros.Run.create takes them too
    (they are not PorosConfig fields)."""
    from poros.train import run as R

    seen = {}

    def loaded(model, **kw):
        seen.update(kw)
        raise RuntimeError("loaded")

    monkeypatch.setattr(R.PorosTrainer, "from_pretrained", loaded)
    with pytest.raises(RuntimeError, match="loaded"):
        R.Run.create("Qwen/Qwen2.5-0.5B", **{flag: False})
    assert seen[flag] is False
