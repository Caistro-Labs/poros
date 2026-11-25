"""Exercise the actual Triton routing kernels on CUDA or with TRITON_INTERPRET=1."""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from poros.moe.route_kernel import route_rows, transpose_runs  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() and not INTERPRET else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu" and not INTERPRET,
                                reason="requires CUDA or TRITON_INTERPRET=1")
if DEVICE == "cuda":
    pytestmark = [pytestmark, pytest.mark.gpu]


@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("lengths,k", [
    ([0, 0], 1), ([1], 1), ([128, 128], 1), ([31, 0, 33], 4), ([0, 257, 0], 1),
    ([513], 8), ([300, 0, 2, 0, 513], 2), ([512] * 8, 8)])
def test_rows_and_runs(expert_dtype, lengths, k):
    gen = torch.Generator().manual_seed(21)
    n, experts = sum(lengths), 128
    idx = torch.randint(0, experts // 2, (n, k), generator=gen, dtype=expert_dtype)
    idx[:257] = 3  # one run crosses a tile; it can also share a tile with the next segment
    sorted_ids, order = idx.to(DEVICE).reshape(-1).sort(stable=True)
    segments = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths) * k)
    expected = torch.bincount(idx.ravel().long() * len(lengths) + segments,
                              minlength=experts * len(lengths))
    expect_seg = segments[order.cpu()]
    for count in (True, False):
        dest, tok, seg, counts = route_rows(sorted_ids, order, k, experts, lengths, count=count)
        assert torch.equal(dest.cpu(), order.cpu().int()) and dest.dtype == torch.int32
        assert torch.equal(tok.cpu(), order.cpu() // k) and tok.dtype == torch.int32
        if len(lengths) == 1:
            assert seg is None
        else:
            assert torch.equal(seg.cpu(), expect_seg) and seg.dtype == torch.int64
        if count:
            assert counts.dtype == torch.int64 and torch.equal(counts, expected)
        else:
            assert counts is None

    ends = expected.cumsum(0)
    nonempty = expected > 0
    transposed = expected.view(experts, -1).T.flatten()
    dst = (transposed.cumsum(0) - transposed).view(len(lengths), -1).T.flatten()[nonempty]
    old_order = torch.cat([(expect_seg == s).nonzero().flatten() for s in range(len(lengths))])
    new_order = transpose_runs((ends - expected)[nonempty].int().to(DEVICE),
                               ends[nonempty].int().to(DEVICE), dst.int().to(DEVICE), n * k)
    assert torch.equal(new_order.cpu(), old_order) and new_order.dtype == torch.int64


@pytest.mark.parametrize("bad,error", [(-1, RuntimeError), (4, ValueError),
                                       (2**62 + 7, ValueError), (-(2**62), RuntimeError)])
def test_invalid_ids_are_reported_in_the_count_transfer(bad, error):
    idx = torch.tensor([bad, 1, 2], device=DEVICE)
    ids, order = idx.sort(stable=True)
    with pytest.raises(error, match="non-negative|at or above"):
        route_rows(ids, order, 1, 4, [0, 2, 1], count=True)
