"""The block-major expert dequant cache changes no value: forward and grads bitwise with it on and
off (CPU)."""
from __future__ import annotations

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.moe.prepare import prepare_moe, streaming_config  # noqa: E402
from poros.moe.store import EXPERT_DQ_CACHE  # noqa: E402


def _prepared():
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                         moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=4)
    model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
    prepare_moe(model, streaming_config(cfg, device="cuda" if torch.cuda.is_available() else "cpu"))
    return model


def _run(model, ids, cache_on):
    EXPERT_DQ_CACHE.enabled = cache_on
    EXPERT_DQ_CACHE.clear()
    try:
        model.zero_grad(set_to_none=True)
        ids = ids.to(model.get_input_embeddings().weight.device)  # on CUDA where there is one
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return out.logits.detach(), grads
    finally:
        EXPERT_DQ_CACHE.enabled = False
        EXPERT_DQ_CACHE.clear()


def test_cache_is_bitwise_and_hits():
    model = _prepared()
    ids = torch.randint(0, 64, (1, 24), generator=torch.Generator().manual_seed(2))
    ref_logits, ref_grads = _run(model, ids, False)
    h0 = EXPERT_DQ_CACHE.hits
    got_logits, got_grads = _run(model, ids, True)
    assert torch.equal(ref_logits, got_logits)
    assert ref_grads.keys() == got_grads.keys() and ref_grads
    for n, g in ref_grads.items():
        assert torch.equal(g, got_grads[n]), n
    assert EXPERT_DQ_CACHE.hits > h0  # the backward's remat reuses the layer's decoded experts
