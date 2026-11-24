"""Where the streamed loader quantizes (CPU): ``auto`` takes CUDA only when it is there, ``cpu``
never does, ``POROS_QUANTIZE_ON`` overrides, and asking for CUDA without it is an error, not a
silent CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.moe import load  # noqa: E402


def test_auto_falls_back_to_the_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("POROS_QUANTIZE_ON", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert load._quantize_device("auto") is None
    assert load._quantize_device("cpu") is None
    assert load._quantize_device(None) is None
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        load._quantize_device("cuda")


def test_auto_takes_the_training_gpu_and_the_env_overrides_auto_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("POROS_QUANTIZE_ON", raising=False)
    assert load._quantize_device("auto") == "cuda"
    assert load._quantize_device("auto", "cuda:1") == "cuda:1"  # the run's device, not cuda:0
    assert load._quantize_device("auto", "cpu") == "cuda"
    monkeypatch.setenv("POROS_QUANTIZE_ON", "cpu")
    assert load._quantize_device("auto", "cuda:1") is None
    assert load._quantize_device("cuda:1") == "cuda:1"  # an explicit choice stands


def test_a_device_failure_quantizes_the_rest_on_the_cpu(monkeypatch):
    """Out of memory or a CUDA error on the quantizing device: that tensor and the rest go to the
    CPU, which makes the same bytes, instead of aborting the load."""
    from poros.moe import quant
    from poros.moe.contracts import ExpertLayout
    from poros.moe.store import ExpertSlabArena

    layout = ExpertLayout(num_layers=1, experts_per_layer=2, top_k=1,
                          tensor_names=("gate_up_proj", "down_proj"))
    shapes = {"gate_up_proj": (8, 64), "down_proj": (64, 4)}
    real, calls = quant.quantize_tensor, []

    def flaky(weight, device=None):
        calls.append(device)
        if device is not None:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(weight)

    monkeypatch.setattr(quant, "quantize_tensor", flaky)
    arena = ExpertSlabArena(layout, shapes, quant_device="cuda")
    want = ExpertSlabArena(layout, shapes)
    for e in range(2):
        for name, shape in shapes.items():
            w = torch.randn(shape)
            arena.write((0, e), name, w)
            want.write((0, e), name, w)
    assert calls.count("cuda") == 1  # tried once, then the CPU for the rest
    a, b = arena.build(torch.float32), want.build(torch.float32)
    for e in range(2):
        assert torch.equal(a.source._slab((0, e)), b.source._slab((0, e)))


def test_quantize_tensor_on_the_cpu_device_is_the_default_path():
    from poros.moe import quant

    w = (torch.randn(3, 131) * 0.02).to(torch.bfloat16)  # a tail block too
    assert torch.equal(quant.quantize_tensor(w), quant.quantize_tensor(w, "cpu"))
