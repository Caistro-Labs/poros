"""The block Functions on ``mps``, faked on Linux.

What is faked: the device a tensor reports, and ``torch.mps``'s RNG calls, backed by a CPU
generator. So this checks the wiring (capture, fork, restore, refusal) and says nothing
about Metal.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from poros.core.blockwise import JITAdapterBlockFunction, JITFrozenBlockFunction  # noqa: E402
from poros.core.errors import EXIT_CODES, PorosGuaranteeUnavailableError  # noqa: E402
from poros.metal import watchdog  # noqa: E402


class _MpsTensor(torch.Tensor):
    """A CPU tensor reporting device ``mps``: all the Functions branch on."""

    device = property(lambda self: torch.device("mps"))


@pytest.fixture
def fake_mps_rng(monkeypatch):
    gen = torch.Generator().manual_seed(7)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda device="mps": gen.get_state())
    monkeypatch.setattr(
        torch.mps, "set_rng_state", lambda state, device="mps": gen.set_state(state)
    )
    return gen


class _Trainer:
    """What the Function touches. The layer's dropout draws from ``gen``, as an MPS op would."""

    def __init__(self, gen):
        self.gen = gen
        self.masks = []
        self.param = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(preserve_rng_state=True)
        self.block_trainable_params = {0: [("p", self.param)]}
        noop = lambda *a, **k: None  # noqa: E731
        self.block_manager = SimpleNamespace(load_block_for_inference=noop, release_block=noop)
        self._record_remat_event = noop

    def _get_adapter_prefetch_stream(self):
        return None

    def _forward_block_layers(self, block_id, hidden, attention, position, checkpoint):
        mask = torch.rand(hidden.shape, generator=self.gen) > 0.5
        self.masks.append(mask)
        return hidden * mask * self.param


def _hidden():
    return torch.ones(8, 4).as_subclass(_MpsTensor)


def _forward(trainer, hidden):
    ctx = SimpleNamespace()
    return ctx, JITAdapterBlockFunction.forward(ctx, hidden, None, None, trainer, 0)


def test_mps_remat_replays_the_forward_dropout_masks(fake_mps_rng):
    trainer = _Trainer(fake_mps_rng)
    ctx, output = _forward(trainer, _hidden())
    assert ctx.mps_rng_state is not None
    torch.rand(5, generator=fake_mps_rng)  # later layers' draws
    before_backward = fake_mps_rng.get_state()
    JITAdapterBlockFunction.backward(ctx, torch.ones_like(output))
    assert torch.equal(trainer.masks[0], trainer.masks[1])
    # The fork puts the stream back: a remat does not advance what comes next.
    assert torch.equal(fake_mps_rng.get_state(), before_backward)
    assert trainer.param.grad is not None


def test_without_the_mps_state_the_masks_differ(fake_mps_rng):
    """The control: what the capture prevents, and that this test can see it."""
    trainer = _Trainer(fake_mps_rng)
    ctx, output = _forward(trainer, _hidden())
    ctx.mps_rng_state = None
    JITAdapterBlockFunction.backward(ctx, torch.ones_like(output))
    assert not torch.equal(trainer.masks[0], trainer.masks[1])


def test_preserve_rng_off_captures_nothing_on_mps(fake_mps_rng):
    trainer = _Trainer(fake_mps_rng)
    trainer.config.preserve_rng_state = False
    ctx, _ = _forward(trainer, _hidden())
    assert ctx.cpu_rng_state is None and ctx.mps_rng_state is None


def test_cpu_path_makes_the_same_fork_call_and_never_touches_mps(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("torch.mps touched off mps")

    monkeypatch.setattr(torch.mps, "get_rng_state", boom)
    monkeypatch.setattr(torch.mps, "set_rng_state", boom)
    calls = []
    real_fork = torch.random.fork_rng

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fork(*args, **kwargs)

    monkeypatch.setattr(torch.random, "fork_rng", spy)
    trainer = _Trainer(torch.Generator().manual_seed(7))
    hidden = torch.ones(8, 4)
    before = torch.get_rng_state()
    ctx, output = _forward(trainer, hidden)
    assert torch.equal(torch.get_rng_state(), before)  # the added calls draw nothing
    assert ctx.mps_rng_state is None
    JITAdapterBlockFunction.backward(ctx, torch.ones_like(output))
    assert calls == [((), {"devices": [], "enabled": True})]
    assert torch.equal(torch.get_rng_state(), before)


def test_the_patched_path_refuses_mps(fake_mps_rng):
    with pytest.raises(PorosGuaranteeUnavailableError) as exc:
        JITFrozenBlockFunction.forward(
            SimpleNamespace(), _hidden(), None, None, _Trainer(fake_mps_rng), 0
        )
    assert EXIT_CODES[exc.value.code] == 4


def test_is_mps_oom_knows_both_forms():
    assert watchdog.is_mps_oom(RuntimeError("MPS backend out of memory (MPS allocated: 0 bytes)"))
    assert watchdog.is_mps_oom(torch.OutOfMemoryError("command buffer"))
    assert not watchdog.is_mps_oom(RuntimeError("shape mismatch"))


def test_backend_and_metal_import_without_torch():
    code = "import sys, poros.core.backend, poros.metal; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
