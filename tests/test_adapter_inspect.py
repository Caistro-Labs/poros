"""Adapter naming tests (torch-free): which parameter names are an adapter's."""

from __future__ import annotations

from poros.adapters.inspect import is_adapter_param

# ---------------------------------------------------------------------------
# is_adapter_param
# ---------------------------------------------------------------------------


def test_is_adapter_param_matches_lora_names():
    assert is_adapter_param("model.layers.0.self_attn.q_proj.lora_A.default.weight")
    assert is_adapter_param("model.layers.0.self_attn.v_proj.lora_B.default.weight")
    assert is_adapter_param("base_model.model.layers.3.mlp.lora_magnitude_vector")


def test_is_adapter_param_rejects_base_names():
    assert not is_adapter_param("model.layers.0.self_attn.q_proj.weight")
    assert not is_adapter_param("model.embed_tokens.weight")
    assert not is_adapter_param("lm_head.weight")
