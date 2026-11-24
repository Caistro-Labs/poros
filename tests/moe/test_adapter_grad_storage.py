"""``poros.moe.fused.accumulate_adapter_grads`` as block-major's 1x8 accumulation runs it: eight
micro-batches' expert adapter grads, each a view into one ``(segments, *shape)`` buffer per
parameter (what ``experts_backward`` returns for a batched visit, ``poros.moe.batching``).

The grads it stores own their memory: a view as ``param.grad`` kept the visit's whole buffer alive
until the grads were cleared (9.4 GiB of Qwen3-30B-A3B's 1x8 peak,
docs/validation/int/4090/round5_moegrad/). They hold the bits of the in-order sum, -0.0 included
(a copy of the first segment, not zeros plus it: ``0 + -0.0`` is ``+0.0``).

The fused kernels need CUDA and Triton; tests/gpu/test_moe_fused_block_major.py checks the grads
of the trainer's real 1x8 run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.moe.adapters import ExpertLoraAdapter  # noqa: E402
from poros.moe.fused import accumulate_adapter_grads  # noqa: E402


def _params(adapter):
    return (adapter.lora_A_gate_up, adapter.lora_B_gate_up, adapter.lora_A_down,
            adapter.lora_B_down)


def _owns_storage(t: torch.Tensor) -> bool:
    return t.untyped_storage().nbytes() == t.numel() * t.element_size()


def _segment_grads(adapter, segments: int):
    """Per parameter one ``(segments, *shape)`` buffer, and each segment's grads as views into
    them, as ``experts_backward`` lays them out. One element is -0.0 in every segment."""
    g = torch.Generator().manual_seed(0)
    bufs = [torch.randn((segments, *p.shape), generator=g).to(p.dtype) for p in _params(adapter)]
    for buf in bufs:
        buf[:, 0, 0, 0] = -0.0
    return bufs, [tuple(buf[k] for buf in bufs) for k in range(segments)]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)  # bf16: -0.0 and +0.0 differ here, not under torch.equal


def test_eight_segments_store_owned_grads_with_the_in_order_bits():
    adapter = ExpertLoraAdapter(4, 16, 8, rank=4, alpha=8)
    bufs, segments = _segment_grads(adapter, 8)
    expected = [buf[0].clone() for buf in bufs]
    for k in range(1, 8):
        for want, buf in zip(expected, bufs, strict=True):
            want.add_(buf[k])
    for grads in segments:
        accumulate_adapter_grads(adapter, grads)
    for param, want in zip(_params(adapter), expected, strict=True):
        assert _owns_storage(param.grad)
        assert torch.equal(_bits(param.grad), _bits(want))
        assert torch.signbit(param.grad[0, 0, 0])


def test_one_segment_is_stored_as_is():
    """A call of its own (8x1, or no batching) returns a one-segment buffer: stored, not copied."""
    adapter = ExpertLoraAdapter(4, 16, 8, rank=4, alpha=8)
    bufs, segments = _segment_grads(adapter, 1)
    accumulate_adapter_grads(adapter, segments[0])
    for param, buf in zip(_params(adapter), bufs, strict=True):
        assert param.grad.data_ptr() == buf.data_ptr()
