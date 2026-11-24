"""``poros.core.fused.attention.attention_forward`` (CUDA): the memory-efficient causal SDPA
under Poros's deterministic mode -- the math backend's values up to rounding, and a backward that
gives the same bits run to run (Qwen3-30B-A3B's attention shape: 32 query heads over 4 KV heads,
head_dim 128)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the efficient SDPA kernel runs on CUDA", allow_module_level=True)
pytest.importorskip("transformers")

from poros.core.determinism import enforce_determinism  # noqa: E402
from poros.core.fused.attention import attention_forward  # noqa: E402


def _qkv(batch, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def t(heads):
        return torch.randn(batch, heads, 512, 128, generator=g, device="cuda").to(torch.bfloat16)

    return t(32).requires_grad_(), t(4).requires_grad_(), t(4).requires_grad_()


def _run(fn, batch, seed):
    q, k, v = _qkv(batch, seed)
    out = fn(q, k, v)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    out.backward(torch.randn(out.shape, generator=g, device="cuda").to(out.dtype))
    return [out.detach(), q.grad, k.grad, v.grad]


@pytest.mark.parametrize("batch", [1, 2])
def test_efficient_attention_is_deterministic_and_close_to_math(batch):
    enforce_determinism()
    module = types.SimpleNamespace(is_causal=True, num_key_value_groups=8)
    scale = 128 ** -0.5

    def ours(q, k, v):
        return attention_forward(module, q, k, v, None, scaling=scale)[0]

    def math(q, k, v):
        k = k.repeat_interleave(8, dim=1)
        v = v.repeat_interleave(8, dim=1)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        return o.transpose(1, 2)

    first, second, want = _run(ours, batch, 3), _run(ours, batch, 3), _run(math, batch, 3)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(first, want, strict=True):
        assert a.shape == b.shape
        err = ((a.float() - b.float()).norm() / b.float().norm()).item()
        assert err < 1e-2, err
