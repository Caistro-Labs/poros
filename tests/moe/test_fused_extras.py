"""``poros.core.fused.extras``: install_extras routes a MoE model's loss, SDPA attention and
RMSNorms through the fused compute's versions, and uninstall_extras gives the model its own back
(no kernel runs here: the norms' module needs Triton to import)."""

from __future__ import annotations

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.core.fused import attention, head, norm  # noqa: E402
from poros.core.fused.extras import install_extras, uninstall_extras  # noqa: E402


def test_install_then_uninstall_extras():
    cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
                         attn_implementation="sdpa")
    model = Qwen3MoeForCausalLM(cfg)
    own_loss = model.loss_function
    done = install_extras(model)
    assert done["attention"] and done["rms_norms"] > 0
    assert model.loss_function is head.CausalLMLoss
    assert model.config._attn_implementation == attention.NAME
    assert any(getattr(vars(m).get("forward"), "__func__", None) is norm.rms_norm
               for m in model.modules())
    uninstall_extras(model)
    assert model.loss_function == own_loss
    assert model.config._attn_implementation == "sdpa"
    assert not any("forward" in vars(m) for m in model.modules())
