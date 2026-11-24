"""``poros.core.fused.head.CausalLMLoss`` against transformers' ``ForCausalLMLoss``: the same loss
and logits grad to fp32 rounding (mean, ``num_items_in_batch``, ignored targets, several chunks),
and the same bits on a second run."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.loss.loss_utils import ForCausalLMLoss  # noqa: E402

from poros.core.fused import head  # noqa: E402


def _case(dtype):
    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(2, 700, 333, generator=g) * 3).to(dtype)
    labels = torch.randint(0, 333, (2, 700), generator=g)
    labels[0, :50] = -100
    labels[1, 600:] = -100
    return logits, labels


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_items", [None, 1234])
def test_chunked_loss_is_hf_loss(dtype, num_items, monkeypatch):
    monkeypatch.setattr(head, "MOE_ROWS", 256)  # several chunks and a partial one
    logits, labels = _case(dtype)
    runs = []
    for fn in (ForCausalLMLoss, head.CausalLMLoss, head.CausalLMLoss):
        x = logits.clone().requires_grad_(True)
        loss = fn(logits=x, labels=labels, vocab_size=333, num_items_in_batch=num_items)
        loss.backward()
        runs.append((loss.detach(), x.grad))
    (want, want_g), (got, got_g), (again, again_g) = runs
    assert got.dtype == torch.float32 and got_g.dtype == dtype
    assert torch.equal(got, again) and torch.equal(got_g, again_g)
    assert torch.allclose(got, want, rtol=1e-5, atol=0)
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    err = ((got_g.float() - want_g.float()).norm() / want_g.float().norm()).item()
    assert err < tol, err


_CUDA = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"), pytest.mark.gpu]


@pytest.mark.parametrize("device,fused_ce", [("cpu", True), pytest.param("cuda", True, marks=_CUDA),
                                             pytest.param("cuda", False, marks=_CUDA)])
def test_head_loss_is_the_head_then_hf_loss(monkeypatch, device, fused_ce):
    """head_loss(h, W, labels): transformers' loss of h @ W^T to rounding, loss and grad of h,
    without the full logits; the same bits on a second run -- on CUDA with the Triton
    cross-entropy (poros.core.fused.ce_kernel) and with the torch ops."""
    monkeypatch.setattr(head, "MOE_ROWS", 300)  # several chunks and a partial one
    if not fused_ce:  # the torch ops' path, as where Triton is not installed
        monkeypatch.setattr(head, "_fused_ce", lambda: False)
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2, 700, 64, generator=g).to(torch.bfloat16).to(device)
    W = (torch.randn(333, 64, generator=g) * 0.2).to(torch.bfloat16).to(device)
    labels = torch.randint(0, 333, (2, 700), generator=g).to(device)
    labels[1, :20] = -100
    runs = []
    for fn in ("hf", "ours", "ours"):
        x = h.clone().requires_grad_(True)
        if fn == "hf":
            loss = ForCausalLMLoss(logits=torch.nn.functional.linear(x, W), labels=labels,
                                   vocab_size=333)
        else:
            loss = head.head_loss(x, W, labels)
        loss.backward()
        runs.append((loss.detach(), x.grad))
    (want, want_g), (got, got_g), (again, again_g) = runs
    assert torch.equal(got, again) and torch.equal(got_g, again_g)
    assert got_g.dtype == torch.bfloat16
    assert torch.allclose(got, want, rtol=1e-5, atol=0)
    err = ((got_g.float() - want_g.float()).norm() / want_g.float().norm()).item()
    assert err < 1e-2, err
