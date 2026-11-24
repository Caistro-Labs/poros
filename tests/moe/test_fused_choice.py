"""Which expert compute a MoE model gets (CPU): the fused kernels compute SiLU experts only, so a
model whose experts are not SiLU -- Gemma 4's are GELU-tanh -- takes the reference compute under the
default ("auto"), says why, and computes what transformers' eager experts compute; an explicit
"fused" is refused for it. The plan keys the same choice, and the config-level reasons are named."""

from __future__ import annotations

import copy

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from poros.core.compute import choose  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.moe import fused  # noqa: E402
from poros.moe.arch import experts_silu  # noqa: E402
from poros.moe.modules import ManagedExperts, ManagedMoeBlock  # noqa: E402
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402
from tests.moe.test_families import _model  # noqa: E402


class _CudaBf16:  # what supported() reads of a CUDA run's hidden states
    is_cuda = True
    dtype = torch.bfloat16


def _blocks(model):
    return [m for m in model.modules() if isinstance(m, ManagedMoeBlock)]


def test_auto_takes_the_reference_for_gemma4_and_matches_the_eager_experts():
    model = _model("gemma4_moe").to(torch.float32)
    layers = model.model.language_model.layers
    eager = [copy.deepcopy(layer.experts) for layer in layers]
    report = prepare_moe(model, MoeConfig(streamed=False, look_ahead=0, expert_compute="auto"))
    assert report.expert_compute == "reference"
    assert "GELUTanh" in report.expert_compute_reason and "SiLU" in report.expert_compute_reason
    assert all(b.expert_compute == "reference" for b in _blocks(model))
    hidden = torch.randn(32, 64)
    for ordinal, (layer, ref) in enumerate(zip(layers, eager, strict=True)):
        managed = layer.experts
        assert isinstance(managed, ManagedExperts)
        assert fused.supported(managed, _CudaBf16()) is not None  # refused even on CUDA in bf16
        managed.residency.ensure([(ordinal, e) for e in range(managed.num_experts)])
        with torch.no_grad():
            for tensor in ("gate_up_proj", "down_proj"):
                for e in range(managed.num_experts):
                    key = (ordinal, e)
                    slab = managed.residency.resident_qbytes(key)
                    getattr(ref, tensor)[e] = managed.store.dequantize_tensor(key, tensor, slab)
            _, weights, picks = layer.router(hidden)
            want = ref.forward(hidden, picks, weights)  # transformers' eager Gemma4 experts
            torch.testing.assert_close(managed(hidden, picks, weights), want, rtol=1e-5, atol=1e-6)


def test_an_explicit_fused_is_refused_for_gemma4():
    model = _model("gemma4_moe").to(torch.bfloat16)
    with pytest.raises(ValueError, match="cannot run this model: layer 0: .*GELUTanh"):
        prepare_moe(model, MoeConfig(streamed=False, look_ahead=0, expert_compute="fused"))


@pytest.mark.parametrize("family", ["olmoe", "mixtral", "qwen2_moe", "deepseek_v2", "qwen3_5_moe"])
def test_auto_takes_fused_for_a_silu_family(family):
    model = _model(family).to(torch.bfloat16)
    report = prepare_moe(model, MoeConfig(streamed=False, look_ahead=0, expert_compute="auto"))
    assert report.expert_compute == "fused" and report.expert_compute_reason is None
    assert all(b.expert_compute == "fused" for b in _blocks(model))
    assert all(fused.supported(b, _CudaBf16()) is None for b in _blocks(model))


def test_the_plan_keys_the_same_choice():
    assert experts_silu({"model_type": "qwen3_moe", "hidden_act": "silu"}, "qwen3_moe")
    assert not experts_silu({"model_type": "gemma4", "text_config": {}}, "gemma4_moe")
    assert not experts_silu(
        {"text_config": {"hidden_activation": "gelu_pytorch_tanh"}}, "gemma4_moe")
    assert experts_silu(None, "mixtral")


@pytest.mark.parametrize("kw,cuda,triton,why", [
    ({}, False, True, "needs CUDA"),
    ({}, True, False, "needs Triton"),
    ({"torch_dtype": "float32"}, True, True, "torch_dtype float32"),
    ({"fused_compute": False}, True, True, "fused_compute: false"),
])
def test_the_config_level_reasons(kw, cuda, triton, why):
    cfg = PorosConfig(model_name_or_path="x", dataset="synthetic", **kw)
    choice = choose(cfg, entry="moe", cuda=cuda, triton=triton)
    assert choice.moe_expert == "reference" and not choice.fused and why in choice.reason
    cfg = PorosConfig(model_name_or_path="x", dataset="synthetic")
    choice = choose(cfg, entry="moe", cuda=True, triton=True)  # the model's layers decide
    assert (choice.moe_expert, choice.fused, choice.reason) == ("auto", True, None)
    assert choose(cfg, entry="moe", cuda=True, triton=True, silu_experts=False).moe_expert \
        == "reference"


def test_a_grouping_is_the_reference_compute_s_setting():
    """moe_expert_grouping buckets the reference compute's rows: with the compute unset it picks
    the reference (and says why); with fused or grouped set it is refused, never ignored."""
    cfg = PorosConfig(model_name_or_path="x", dataset="synthetic", moe_expert_grouping="single")
    choice = choose(cfg, entry="moe", cuda=True, triton=True)
    assert choice.moe_expert == "reference" and "moe_expert_grouping: single" in choice.reason
    for compute in ("fused", "grouped"):
        with pytest.raises(ValueError, match="moe_expert_grouping"):
            PorosConfig(model_name_or_path="x", dataset="synthetic", moe_expert_grouping="pow2",
                        moe_expert_compute=compute)


@pytest.mark.parametrize("kw,cuda,triton,fused,why", [
    ({}, True, True, True, None),
    ({}, True, False, False, "needs Triton"),
    ({}, False, True, False, "needs CUDA"),
    ({"fused_compute": False}, True, True, False, None),  # as set: no reason to give
])
def test_the_dense_choice(kw, cuda, triton, fused, why):
    cfg = PorosConfig(model_name_or_path="x", dataset="synthetic", **kw)
    choice = choose(cfg, entry="dense", cuda=cuda, triton=triton)
    assert choice.fused is fused and choice.moe_expert is None
    assert (choice.reason is None) if why is None else why in choice.reason


def test_the_trainer_records_settings_what_ran_and_why(tmp_path, cpu_only):
    """The settings stay as set (None); what ran is ran_compute, why the default took the slower
    compute is compute_reason (here: a CPU run, also on a host with a GPU), and the setup event
    says the same."""
    from tests.moe.test_families import _loader, _trainer

    base = tmp_path / "base"
    _model("olmoe").to(torch.bfloat16).save_pretrained(base)
    tr = _trainer(base, tmp_path / "run", max_steps=1)
    cfg = tr.config
    assert cfg.fused_compute is None and cfg.moe_expert_compute is None  # the settings
    assert cfg.ran_compute["moe_expert_compute"] == "reference"
    assert cfg.ran_compute["fused_compute"] is False
    assert cfg.ran_compute["moe_expert_cache_gib"] >= 0  # the expert residency's, in GiB
    assert "needs CUDA" in cfg.compute_reason
    record = tr.compute.record()
    # and what steps the expert adapters: ExpertAdamW's reference on the CPU (the CPU has no
    # fused kernel and costs nothing to say, so no time is recorded or said)
    assert record == {"fused_compute": False, "moe_expert_compute": "reference",
                      "compute_reason": cfg.compute_reason,
                      "expert_optimizer": {"cpu": {"path": "reference"}}}
    assert "ExpertAdamW" not in cfg.compute_reason
    tr.train(_loader(2))
