"""``poros.core.fused.norm`` (CUDA, bf16): transformers' Qwen3-MoE RMSNorm to bf16 rounding,
forward and input grad, the same bits run to run, and ``install_rms_norm`` routing a model's norms
through it."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the fused RMSNorm runs on CUDA", allow_module_level=True)
pytest.importorskip("triton")

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm  # noqa: E402

from poros.core.fused import norm  # noqa: E402


@pytest.mark.parametrize("shape", [(4, 512, 2048), (2, 512, 32, 128), (3, 7, 100)])
def test_fused_rmsnorm_matches_transformers(shape):
    g = torch.Generator(device="cuda").manual_seed(0)
    mod = Qwen3MoeRMSNorm(shape[-1], eps=1e-6).to("cuda", torch.bfloat16)
    with torch.no_grad():
        mod.weight.copy_(1 + 0.1 * torch.randn(shape[-1], generator=g, device="cuda"))
    mod.weight.requires_grad_(False)
    x = (torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dy = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)

    def run(fn):
        xi = x.clone().requires_grad_(True)
        out = fn(xi)
        out.backward(dy)
        return out.detach(), xi.grad

    want = run(mod)
    got = run(lambda t: norm.rms_norm(mod, t))
    again = run(lambda t: norm.rms_norm(mod, t))
    for a, b in zip(got, again, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(got, want, strict=True):
        err = ((a.float() - b.float()).norm() / b.float().norm()).item()
        assert err < 1e-2, err
    assert norm.install_rms_norm(torch.nn.Sequential(mod)) == 1
    assert torch.equal(mod(x), got[0])
