"""The fused decoder layer computes a SwiGLU MLP: a Qwen2-architecture layer whose MLP activation is
not SiLU keeps its own forward, and ``install`` records why (CPU: only the routing is checked)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from poros.core import fused  # noqa: E402


def _tiny(act: str):
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, hidden_act=act)
    return transformers.Qwen2ForCausalLM(cfg)


def test_a_silu_layer_takes_the_fused_forward():
    model = _tiny("silu")
    try:
        counts = fused.install(model)
        assert counts["layers"] == 2 and fused.refusals(model) == []
    finally:
        fused.uninstall(model)
    assert fused.refusals(model) == []


@pytest.mark.parametrize("act", ["gelu", "gelu_pytorch_tanh", "relu"])
def test_another_activation_keeps_the_layers_own_forward_and_says_why(act, caplog):
    model = _tiny(act)
    with caplog.at_level("WARNING", logger=fused.__name__):
        counts = fused.install(model)
    try:
        assert counts["layers"] == 0 and counts["swiglu_mlps"] == 0
        assert fused.refusals(model) == [f"Qwen2DecoderLayer: MLP activation {act!r} is not SiLU"]
        assert "is not SiLU" in caplog.text
        assert all("_poros_orig_forward" not in vars(m) for m in model.modules())
    finally:
        fused.uninstall(model)


def test_one_silu_rule_for_the_dense_layers_and_the_moe_experts():
    """The dense fused compute and the fused MoE experts ask the same question of the callable a
    module runs (poros.core.activation.is_silu): F.silu, nn.SiLU and SiLUActivation are SiLU; a
    swapped activation is not, whatever config.json calls it."""
    from transformers.activations import ACT2FN

    from poros.core.activation import is_silu
    from poros.moe import fused as moe_fused

    for act in (torch.nn.functional.silu, torch.nn.SiLU(), ACT2FN["silu"], ACT2FN["swish"]):
        assert is_silu(act) and moe_fused._silu(act)
    for act in (ACT2FN["gelu"], torch.nn.GELU(), torch.relu, None):
        assert not is_silu(act) and not moe_fused._silu(act)
    model = _tiny("silu")
    for layer in model.model.layers:
        layer.mlp.act_fn = torch.nn.GELU()  # config.json still says silu
    try:
        assert fused.install(model)["layers"] == 0
        assert fused.refusals(model) == ["Qwen2DecoderLayer: MLP activation 'GELU' is not SiLU"]
    finally:
        fused.uninstall(model)
