"""ExpertAdamW's fused bf16 update (poros/train/expert_adamw_cuda.py) on the card, asserted: the
whole update (weight, remainder, m, v) equals the rule's reference (computed on the CPU) bit for
bit, over inputs that exercise every operation (``_probe_inputs``), at three sets of AdamW
settings, three updates fed back; the kernel twice gives the same bits; and the optimizer
chooses it. One differing element fails, naming the element, its inputs and both outputs.
tests/moe/test_expert_adamw_kernel.py runs the same kernel in Triton's interpreter.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
pytest.importorskip("triton")

from poros.train import expert_adamw as ea  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(("lr", "betas", "eps", "wd"), [
    (2e-4, (0.9, 0.999), 1e-8, 0.01), (2e-5, (0.99, 0.95), 1e-6, 0.1),
    (1e-3, (0.0, 0.5), 0.0, 0.0)])
@pytest.mark.parametrize("n", [2 * ea.BLOCK + 1040, 2 * ea.BLOCK + 1037])  # both variants
def test_the_card_stores_the_reference_s_bits(lr, betas, eps, wd, n):
    inputs = ea._probe_inputs(n, 11)
    s = ea._scalars(lr, betas[0], betas[1], eps, wd, 4)
    key = ea.update_key(3, 4, "model.layers.2.mlp.adapter.lora_B_gate_up")
    got = ea._run_steps("kernel", "cuda", inputs, 3, key, s)
    want = ea._run_steps("reference", "cpu", inputs, 3, key, s)
    diff = ea._first_difference(got, want, inputs)
    assert diff is None, diff
    again = ea._run_steps("kernel", "cuda", inputs, 3, key, s)
    assert ea._first_difference(again, got, inputs) is None  # the same bits, run after run


def test_the_reference_on_the_card_stores_the_cpu_s_bits():
    """The path a card without Triton takes: the reference, on the card, in fp64."""
    inputs = ea._probe_inputs(ea.BLOCK + 517, 12)
    s = ea._scalars(2e-4, 0.9, 0.999, 1e-8, 0.01, 4)
    key = ea.update_key(3, 4, "w")
    got = ea._run_steps("reference", "cuda", inputs, 3, key, s)
    want = ea._run_steps("reference", "cpu", inputs, 3, key, s)
    assert ea._first_difference(got, want, inputs) is None, ea._first_difference(got, want,
                                                                                 inputs)


def test_the_optimizer_chooses_the_kernel_on_this_card():
    p = torch.nn.Parameter(torch.zeros(4096, dtype=torch.bfloat16, device="cuda"))
    opt = ea.ExpertAdamW([("w", p)], lr=1e-3)
    assert opt.implementation["cuda:0"]["path"] == "kernel", opt.implementation


def test_the_optimizer_s_own_probe_passes_on_this_card():
    """F's edge (products planted in (2**-126 - 2**-150, 2**-126]), a weight on a bf16 tie, and
    both Triton variants: what the optimizer checks before it uses the kernel."""
    assert ea._probe_differences("cuda") is None, ea._probe_differences("cuda")

