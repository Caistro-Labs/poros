"""GPU parity test: the headline bitwise-identity claim, on real hardware.

This is the test that backs Poros's core promise -- that the blockwise
load/compute/release path produces output *bitwise-identical* to the
all-resident path. It needs a CUDA device and torch, so it is
marked ``gpu`` and skips cleanly everywhere else (CPU CI, laptops).

Run it on a GPU box with:

    pytest -m gpu -v

or via the ``gpu-parity`` GitHub Actions workflow (self-hosted GPU runner).

The model defaults to a small validated-architecture checkpoint; override with
``POROS_GPU_TEST_MODEL`` to exercise a different family.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

# Smallest validated-arch (qwen2) checkpoint that exercises NF4 + blockwise.
DEFAULT_MODEL = os.environ.get("POROS_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


def _require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU parity test requires a CUDA device")
    return torch


def test_resident_vs_blockwise_is_bitwise_identical():
    """The headline claim: resident reference == blockwise streaming, bitwise.

    Both forwards share identical layer math; the only difference is whether
    weights are streamed (load/compute/release) or held resident. A 0.00e+00
    loss diff therefore proves the streaming mechanism preserves parity -- not
    a tautology, because the two paths genuinely differ in residency.
    """
    torch = _require_cuda()
    pytest.importorskip("bitsandbytes")

    from poros.train.trainer import PorosTrainer

    # startup_parity_selfcheck="always" turns on the resident-parity self-check
    # (OFF by default -- it is an internal parity-testing tool); a parity failure
    # would raise here.
    trainer = PorosTrainer.from_pretrained(
        DEFAULT_MODEL,
        quantization="nf4",
        block_size=4,
        blockwise_arch_check="error",
        startup_parity_selfcheck="always",
    )
    assert trainer.block_manager is not None

    vocab = getattr(getattr(trainer.model, "config", None), "vocab_size", 32000)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, 8), device="cuda:0")

    with torch.no_grad():
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        resident = trainer._forward_selfcheck_resident(trainer.model, input_ids)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        blockwise = trainer._forward_selfcheck(trainer.model, input_ids)

    # Bitwise: exact tensor equality, not allclose.
    assert torch.equal(resident, blockwise), (
        f"resident={resident.item()} blockwise={blockwise.item()} "
        f"diff={(resident - blockwise).abs().max().item():.3e}"
    )


def test_two_step_training_loss_is_finite():
    """A couple of real blockwise optimizer steps stay finite (sanity)."""
    torch = _require_cuda()
    pytest.importorskip("bitsandbytes")

    from poros.train.trainer import PorosTrainer

    trainer = PorosTrainer.from_pretrained(
        DEFAULT_MODEL,
        quantization="nf4",
        block_size=4,
        blockwise_arch_check="error",
    )

    vocab = getattr(getattr(trainer.model, "config", None), "vocab_size", 32000)
    input_ids = torch.randint(0, vocab, (1, 16), device="cuda:0")
    loss = trainer._forward_selfcheck(trainer.model, input_ids)
    assert torch.isfinite(loss).item()
