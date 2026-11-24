"""A ``fused_compute: true`` that cannot run is refused with the reason it cannot."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from poros.core.compute import choose
from poros.core.config import PorosConfig

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("kw,cuda,words", [
    ({}, False, "needs CUDA"),
])
def test_choose_says_why_the_fused_compute_cannot_run_whatever_the_setting(kw, cuda, words):
    for fused in (None, True):
        c = choose(PorosConfig(dataset="synthetic", fused_compute=fused, **kw), entry="dense",
                   cuda=cuda, triton=True)
        assert not c.fused and words in c.blocked
        assert c.reason == (c.blocked if fused is None else None)  # recorded only for a default
    off = choose(PorosConfig(dataset="synthetic", fused_compute=False, **kw), entry="dense",
                 cuda=cuda, triton=True)
    assert (off.fused, off.reason, off.blocked) == (False, None, None)


@pytest.mark.parametrize("kw,words", [({}, "the fused compute needs CUDA.")])
def test_a_refused_fused_compute_names_its_reason(kw, words):
    from poros.core.errors import PorosConfigError
    from poros.train.compute import RunCompute

    cfg = PorosConfig(dataset="synthetic", fused_compute=True, **kw)
    t = SimpleNamespace(config=cfg, device=torch.device("cpu"), model=None,
                        _adapter_backend="peft")
    with pytest.raises(PorosConfigError) as err:
        RunCompute(t).install_dense()
    message = str(err.value).splitlines()[0]
    assert "None" not in message
    assert message.startswith(f"fused_compute=True cannot run here: {words}")


def test_the_preflight_refuses_before_the_load_what_install_would_after(monkeypatch):
    """RunCompute.preflight (called by from_pretrained before the download) raises the refusals
    install_dense, prepare_moe and install_moe would raise only once the model had loaded."""
    from poros.core.errors import PorosConfigError
    from poros.train.compute import RunCompute

    def rc(**kw):
        cfg = PorosConfig(dataset="synthetic", **kw)
        return RunCompute(SimpleNamespace(config=cfg, device=torch.device("cpu")))

    with pytest.raises(PorosConfigError, match="needs CUDA"):
        rc(fused_compute=True).preflight("dense")
    from poros.core import compute

    monkeypatch.setattr(compute, "triton_installed", lambda: True)  # a CUDA host's stack
    with pytest.raises(PorosConfigError, match="needs PEFT adapters"):
        RunCompute(SimpleNamespace(config=PorosConfig(dataset="synthetic", fused_compute=True,
                                                      adapter_backend="native"),
                                   device=torch.device("cuda"))).preflight("dense")
    gelu = {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text",
                                                    "enable_moe_block": True,
                                                    "hidden_activation": "gelu_pytorch_tanh"}}
    with pytest.raises(PorosConfigError, match="not SiLU"):
        rc(moe_expert_compute="fused").preflight("moe", gelu, "gemma4_moe")
    with pytest.raises(PorosConfigError, match="needs moe_expert_compute: fused"):
        rc(fused_compute=True).preflight("moe", {"model_type": "qwen3_moe"}, "qwen3_moe")
    rc().preflight("dense")  # unset: decided for the run, never refused
    rc().preflight("moe", gelu, "gemma4_moe")
