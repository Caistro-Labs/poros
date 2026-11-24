"""Batched ``_experts_compute`` against the HF per-expert loop.

Not arch certification (that is the GPU gate, ``scripts/moe_absolute_gate.py``).
The reference is the loop as first written: ``torch.where`` per expert,
per-index adapter slicing, ``index_add_`` combine. Batched matmuls and the
slot sum change the summation order, so outputs and every grad must match
to bf16 rounding, not bitwise. Grads are compared directly, because a few
optimizer steps would hide a wrong one in the losses.
"""

from __future__ import annotations

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

import torch.nn.functional as F  # noqa: E402
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.moe.modules import ManagedMoeBlock, _experts_compute  # noqa: E402
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402


def _reference(hidden_states, top_k_index, top_k_weights, block):
    adapter = block.adapter

    def lora(x, a, b):
        update = F.linear(F.linear(adapter.dropout(x).to(a.dtype), a), b)
        return update.to(x.dtype) * adapter.scaling

    out = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=block.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    block.residency.ensure([(block.layer_idx, int(e[0])) for e in hit])
    for e in hit:
        e = int(e[0])
        key = (block.layer_idx, e)
        top_k_pos, token_idx = torch.where(mask[e])
        x = hidden_states[token_idx]
        slab = block.residency.resident_qbytes(key)
        gate_up = F.linear(x, block.store.dequantize_tensor(key, "gate_up_proj", slab))
        gate_up = gate_up + lora(x, adapter.lora_A_gate_up[e], adapter.lora_B_gate_up[e])
        gate, up = gate_up.chunk(2, dim=-1)
        h = block.act_fn(gate) * up
        down = F.linear(h, block.store.dequantize_tensor(key, "down_proj", slab))
        down = down + lora(h, adapter.lora_A_down[e], adapter.lora_B_down[e])
        down = down * top_k_weights[token_idx, top_k_pos, None]
        out.index_add_(0, token_idx, down.to(out.dtype))
    return out


def _block() -> ManagedMoeBlock:
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=4,
        norm_topk_prob=True,
    )
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    block = next(m for m in model.modules() if isinstance(m, ManagedMoeBlock))
    with torch.no_grad():  # nonzero B so every adapter grad carries signal
        block.adapter.lora_B_gate_up.normal_(0, 0.02)
        block.adapter.lora_B_down.normal_(0, 0.02)
    return block


def _run(fn, block, hidden_states, top_k_index, top_k_weights):
    block.adapter.zero_grad(set_to_none=True)
    x = hidden_states.clone().requires_grad_(True)
    w = top_k_weights.clone().requires_grad_(True)
    out = fn(x, top_k_index, w, block)
    out.float().square().sum().backward()
    return [out, x.grad, w.grad, *(p.grad for p in block.adapter.parameters())]


def _inputs(block):
    hidden_states = torch.randn(512, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        _, top_k_weights, top_k_index = block.gate(hidden_states)
    assert all((top_k_index == e).any() for e in range(block.num_experts))
    return hidden_states, top_k_index, top_k_weights


@pytest.mark.parametrize("expert_batch", [0, 1, 3])
def test_experts_compute_matches_reference(expert_batch):
    block = _block()
    inputs = _inputs(block)
    expected = _run(_reference, block, *inputs)
    block.expert_batch = expert_batch
    actual = _run(_experts_compute, block, *inputs)
    for want, got in zip(expected, actual, strict=True):
        error = (got.float() - want.float()).norm() / want.float().norm()
        assert error < 1e-2  # bf16 epsilon is 7.8e-3


def test_no_tokens():
    block = _block()
    _, top_k_index, top_k_weights = _inputs(block)
    inputs = torch.empty(0, 64, dtype=torch.bfloat16), top_k_index[:0], top_k_weights[:0]
    out, x_grad, *_ = _run(_experts_compute, block, *inputs)
    assert out.shape == x_grad.shape == (0, 64)
