"""The fused expert compute on every MoE family it takes (CUDA): under the default ("auto"), each
tiny real-config family either runs fused -- and then its experts' output and adapter grads match
the reference compute's to bf16 rounding on the same inputs -- or takes the reference for a reason
(Gemma 4: GELU experts). Catches a family whose experts the kernels would compute as another
function."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from poros.moe.fused import FusedExpertFunction  # noqa: E402
from poros.moe.modules import ManagedMoeBlock, _ManagedExpertFunction  # noqa: E402
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402
from tests.moe.test_families import CONFIGS, _model  # noqa: E402


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _factors(ad):
    return [ad.lora_A_gate_up, ad.lora_B_gate_up, ad.lora_A_down, ad.lora_B_down]


def _call(block, fn, x, idx, w, gy):
    for p in _factors(block.adapter):
        p.grad = None
    xi = x.clone().requires_grad_(True)
    out = fn.apply(xi, idx, w, block)
    out.backward(gy)
    return out.detach(), [p.grad.detach().clone() for p in _factors(block.adapter)]


@pytest.mark.parametrize("family", sorted(CONFIGS))
def test_auto_runs_fused_only_where_it_computes_the_reference(family):
    model = _model(family).to(torch.bfloat16)
    report = prepare_moe(model, MoeConfig(streamed=False, look_ahead=0, device="cuda",
                                          expert_compute="auto"))
    blocks = [m for m in model.modules() if isinstance(m, ManagedMoeBlock)]
    if family == "gemma4_moe":
        assert report.expert_compute == "reference" and "SiLU" in report.expert_compute_reason
        return
    assert report.expert_compute == "fused", report.expert_compute_reason
    g = torch.Generator(device="cpu").manual_seed(0)
    for block in blocks:
        ad = block.adapter
        with torch.no_grad():
            for p in (ad.lora_B_gate_up, ad.lora_B_down):  # B = 0 would leave A's grad zero
                p.copy_((torch.randn(p.shape, generator=g) * 1e-2).to(p.dtype))
        h = block.gate.weight.shape[-1]
        x = (torch.randn(96, h, generator=g) * 0.5).to(torch.bfloat16).cuda()
        with torch.no_grad():
            _, w, idx = block.gate(x)
        gy = (torch.randn(x.shape, generator=g) * 1e-2).to(torch.bfloat16).cuda()
        out_f, grads_f = _call(block, FusedExpertFunction, x, idx, w, gy)
        block.expert_compute = "reference"
        out_r, grads_r = _call(block, _ManagedExpertFunction, x, idx, w, gy)
        block.expert_compute = "fused"
        assert _rel(out_f, out_r) < 2e-2, (family, block.layer_idx)
        for name, a, b in zip(("A_gu", "B_gu", "A_d", "B_d"), grads_f, grads_r, strict=True):
            assert _rel(a, b) < 2e-2, (family, block.layer_idx, name)
