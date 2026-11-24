"""When the dense fused attention takes the bf16 flash kernel: by device and head size only, as
PyTorch's flash forward and backward accept them (CPU: the device query is faked)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.core import card  # noqa: E402
from poros.core.fused import attention  # noqa: E402


@pytest.mark.parametrize("cap, D, ok", [
    ((7, 5), 128, False),                      # pre-Ampere: no flash
    ((8, 0), 128, True), ((8, 0), 256, True),  # A100: up to 256
    ((9, 0), 256, True), ((10, 0), 256, True),
    ((8, 6), 192, True), ((8, 6), 256, False),  # RTX 3090: over 192 the backward is refused
    ((8, 9), 128, True), ((8, 9), 256, False),  # RTX 4090
    ((12, 0), 192, True), ((12, 0), 256, False),  # RTX 5090: held to 192
    ((8, 0), 100, False), ((8, 0), 264, False),  # not a multiple of 8, or over 256
])
def test_flash_by_device_and_head_size(monkeypatch, cap, D, ok):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cap)
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: True,
                        raising=False)
    monkeypatch.setattr(card, "_flash_runs", lambda index, D: True)  # the kernel runs here
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    q = torch.empty(1, 2, 3, D)
    assert attention.flash_usable(q) is ok
    assert attention.flash_usable(q.requires_grad_()) is ok  # the recompute: the same kernel


def test_a_build_without_flash_never_takes_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: False,
                        raising=False)
    assert attention.flash_usable(torch.empty(1, 2, 3, 128)) is False


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)])
def test_a_card_whose_flash_kernel_fails_takes_the_efficient_kernel(monkeypatch, cap):
    """A build that lists flash but has no flash kernel for this card (a capability it was not
    built for): the probe's failure sends every call, forward and recompute, to sdpa_efficient."""
    from torch.nn.attention import sdpa_kernel  # noqa: F401  (the probe's import)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cap)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: True,
                        raising=False)
    monkeypatch.setattr(card, "_FLASH_RUNS", {})
    calls = []

    def no_kernel(*a, **k):
        calls.append(1)
        raise RuntimeError("FlashAttention only supports Ampere GPUs or newer / no kernel image")

    monkeypatch.setattr(torch, "zeros", no_kernel)
    q = torch.empty(1, 2, 3, 128)
    assert attention.flash_usable(q) is False
    assert attention.flash_usable(q.requires_grad_()) is False
    assert len(calls) == 1  # tried once per (device, head size), then remembered


@pytest.mark.gpu
def test_the_flash_trial_saves_nothing_into_the_callers_hooks(monkeypatch):
    """The trial runs inside the first layer's forward: a checkpoint's hooks (which count what
    the forward saves, to match its recomputation) must not see its tensors, or the forward and
    the recomputation differ ("a different number of tensors was saved")."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    monkeypatch.setattr(card, "_FLASH_RUNS", {})
    seen = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: seen.append(t) or t, lambda t: t):
        card._flash_runs(torch.cuda.current_device(), 128)
    assert seen == [] and (torch.cuda.current_device(), 128) in card._FLASH_RUNS
