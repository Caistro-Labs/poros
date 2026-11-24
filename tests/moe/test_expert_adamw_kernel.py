"""The CUDA kernel of ExpertAdamW's bf16 update (poros/train/expert_adamw_cuda.py), run by Triton's
CPU interpreter against the rule's reference (``_reference_bf16``), every stored tensor compared
bit for bit over inputs that exercise every operation (``_probe_inputs``: many exponents, both
signs, zeros, subnormals and the smallest normals, the largest finite values, infinities, a NaN,
a weight that rounds to inf, the last partial block), three updates fed back. Skipped where
Triton is not installed (the CPU suite's own environment); tests/gpu/test_expert_adamw_cuda.py
runs the same check on the card, and the optimizer runs it before it uses the kernel.

What is under test is the kernel's own source: its loads, its integer draws, its rounding, its
operations in their order, its F after each, its stores, executed element by element by the
interpreter with numpy, whose fp32 +, -, *, / and square root are each one IEEE operation. One
primitive of the interpreter is not the card's, and the fixture below replaces exactly that one:
its casts between fp32 and bf16 (it rounds half up without carrying into the exponent, and it
mangles bf16 subnormals; the card's casts are exact one way and to nearest even the other, as
torch's, which the fixture uses). The rule has no fused multiply-add, so the interpreter's
twice-rounded fma is not reached.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported anywhere here
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("Triton's interpreter is off in this process", allow_module_level=True)

import numpy as np  # noqa: E402
from triton.runtime import interpreter  # noqa: E402

from poros.train import expert_adamw as ea  # noqa: E402
from poros.train import expert_adamw_cuda  # noqa: E402


@pytest.fixture(autouse=True)
def _the_card_s_casts(monkeypatch):
    real = interpreter._convert_float

    def convert(data, src, dst, rounding):
        import triton.language as tl

        if src == tl.float32 and dst == tl.bfloat16:
            out = torch.from_numpy(np.ascontiguousarray(data).view(np.float32)).bfloat16()
            return out.view(torch.int16).numpy().view(np.uint16)
        if src == tl.bfloat16 and dst == tl.float32:
            bits = torch.from_numpy(np.ascontiguousarray(data).view(np.int16))
            return bits.view(torch.bfloat16).float().numpy().view(np.uint32)
        return real(data, src, dst, rounding)

    monkeypatch.setattr(interpreter, "_convert_float", convert)


def _interpreted(hi, lo, g, m, v, coef, s):
    expert_adamw_cuda.step(hi, lo, g, m, v, coef, tile=1 << 16, **s)  # fewer, larger programs


def test_the_kernel_stores_the_reference_s_bits_over_the_probe(monkeypatch):
    """The optimizer's own probe, as it runs on a card before it uses the kernel: every setting
    of ``PROBE_SETTINGS`` (an ordinary run across a block boundary; three products planted in
    (2**-126 - 2**-150, 2**-126], F's edge; weights on an exact bf16 tie), each at an element
    count that is a multiple of 16 and one that is not."""
    monkeypatch.setitem(ea._KERNEL, "cpu", _interpreted)
    assert ea._probe_differences("cpu") is None, ea._probe_differences("cpu")


@pytest.mark.parametrize("n", [5000, 5008])
def test_the_kernel_at_other_settings(monkeypatch, n):
    monkeypatch.setitem(ea._KERNEL, "cpu", _interpreted)
    inputs = ea._probe_inputs(n, 3)
    s = ea._scalars(2e-5, 0.99, 0.95, 1e-6, 0.1, 5)
    key = ea.update_key(9, 5, "model.layers.0.mlp.adapter.lora_A_down")
    got = ea._run_steps("kernel", "cpu", inputs, 3, key, s)
    want = ea._run_steps("reference", "cpu", inputs, 3, key, s)
    assert ea._first_difference(got, want, inputs) is None, ea._first_difference(got, want, inputs)


def test_the_planted_products_land_in_f_s_window():
    """The probe's second setting does put products strictly inside (2**-126 - 2**-150,
    2**-126): exactly, in fp64 (24-bit times 8-bit fits), and IEEE fp32 rounds each up to
    2**-126, which F reads as 0 (a rule with < would keep it: the mutant this catches)."""
    lr, b1, b2, eps, wd, _ = ea.PROBE_SETTINGS[1]
    s = ea._scalars(lr, b1, b2, eps, wd, 7)
    assert (s["w1"], s["b2"], s["decay"]) == (ea.EDGE_C1, ea.EDGE_C2, ea.EDGE_C3)
    lo_edge, hi_edge = 2.0 ** -126 - 2.0 ** -150, 2.0 ** -126
    for scalar, k in zip((s["w1"], s["b2"], s["decay"]), ea.EDGE_K):
        operand = (1 + k / 128) * 2.0 ** -126
        assert float(torch.tensor(operand).bfloat16()) == operand  # a bf16 value, planted as is
        exact = scalar * operand
        assert lo_edge < exact < hi_edge, (scalar, k)
        assert float(torch.tensor(scalar) * torch.tensor(operand)) == hi_edge  # rounds up


def test_the_stored_weight_and_remainder_are_int16_bits_and_an_infinite_weight_keeps_none(
        monkeypatch):
    """The kernel's int16 types are its own, asserted at compile time (``tl.static_assert``,
    which this run executes), and where the weight rounds to inf its remainder is +0's bits."""
    import inspect

    source = inspect.getsource(expert_adamw_cuda)
    assert "tl.static_assert(r.dtype == tl.int16)" in source
    assert "tl.where((hf - hf) == 0.0, r, 0).to(tl.int16)" in source
    monkeypatch.setitem(ea._KERNEL, "cpu", _interpreted)
    inputs = ea._probe_inputs(64, 1)
    inputs["h"][:] = 3.3895313892515355e38
    inputs["l"][:] = 2.0 ** 119
    s = ea._scalars(0.0, 0.9, 0.999, 1e-8, 0.0, 1)
    got = ea._run_steps("kernel", "cpu", inputs, 1, 5, s)
    want = ea._run_steps("reference", "cpu", inputs, 1, 5, s)
    assert ea._first_difference(got, want, inputs) is None
    gone = ~torch.isfinite(got["h"])  # every element, inf or NaN (where g or m is NaN)
    assert bool(gone.all()) and int(torch.isinf(got["h"]).sum()) > 50
    assert bool((got["l"].view(torch.int16) == 0).all())
