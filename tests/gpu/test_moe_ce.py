"""``poros.core.fused.ce_kernel.cross_entropy_rows`` (CUDA, Triton) at the real vocabulary: each
row's loss term and ``(softmax - onehot)`` grad match the torch ops to rounding, ignored rows are
zero, a row's bits do not depend on the rows around it, and a second run gives the same bits."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from poros.core.fused import ce_kernel as ce  # noqa: E402

V = 151936  # Qwen3-30B-A3B's vocabulary: not a multiple of the block


def _case(rows=96, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(rows, V, device="cuda", generator=g) * 4).to(torch.bfloat16)
    target = torch.randint(0, V, (rows,), device="cuda", generator=g)
    valid = torch.ones(rows, dtype=torch.bool, device="cuda")
    valid[::7] = False
    return logits, target, valid


def _torch(logits, target, valid):
    lf = logits.float()
    lse = torch.logsumexp(lf, -1)
    loss = torch.where(valid, lse - lf.gather(1, target[:, None]).squeeze(1), 0.0)
    p = (lf - lse[:, None]).exp()
    p[torch.arange(len(target), device=p.device), target] -= 1.0
    return loss, (p * valid[:, None]).to(logits.dtype)


def test_matches_torch_and_repeats():
    logits, target, valid = _case()
    grad = torch.empty_like(logits)
    loss = ce.cross_entropy_rows(logits, target, valid, grad)
    want, want_g = _torch(logits, target, valid)
    assert torch.allclose(loss, want, rtol=1e-5, atol=1e-5)
    assert (loss[~valid] == 0).all() and (grad[~valid] == 0).all()
    err = ((grad.float() - want_g.float()).norm() / want_g.float().norm()).item()
    assert err < 1e-2, err
    grad2 = torch.empty_like(logits)
    assert torch.equal(ce.cross_entropy_rows(logits, target, valid, grad2), loss)
    assert torch.equal(grad2, grad)


def test_rows_independent_of_neighbours():
    logits, target, valid = _case()
    grad = torch.empty_like(logits)
    loss = ce.cross_entropy_rows(logits, target, valid, grad)
    part = slice(13, 50)
    g2 = torch.empty_like(logits[part])
    l2 = ce.cross_entropy_rows(logits[part].contiguous(), target[part].contiguous(),
                               valid[part].contiguous(), g2)
    assert torch.equal(l2, loss[part]) and torch.equal(g2, grad[part])


def test_a_masked_row_is_zero_whatever_its_logits():
    """An ignored row's loss and grad are 0 even where its logits are inf or NaN, as the torch
    path's torch.where gives (a product with the mask would make them NaN)."""
    logits, target, valid = _case(rows=8)
    logits[0, 5] = float("nan")
    logits[0, 7] = float("inf")
    logits[1, :] = float("-inf")
    valid[0] = valid[1] = False
    grad = torch.empty_like(logits)
    loss = ce.cross_entropy_rows(logits, target, valid, grad)
    assert torch.equal(loss[:2], torch.zeros(2, device=loss.device))
    assert torch.equal(grad[:2], torch.zeros_like(grad[:2]))
    assert torch.isfinite(loss[2:]).all() and torch.isfinite(grad[2:]).all()
