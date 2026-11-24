"""The Metal kernel's square root, division and bf16 rounding (poros/metal/expert_adamw_shader.py),
on the CPU: its transcription into numpy and its own source compiled as C++, against the
correctly rounded results, over a sample of every class of input
(scripts/moe_precision/metal_exact_arith.py runs 10**8; a Mac's probe runs the shader itself)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "moe_precision"))
import metal_exact_arith as arith  # noqa: E402


@pytest.mark.parametrize("kind", ["sqrt", "div"])
def test_the_transcription_is_correctly_rounded_over_every_class(kind):
    _, n, bad, first = arith.work((kind, 11, 300_000, arith.GUESS))
    assert n > 290_000 and bad == 0, first


def test_the_exact_fma_is_the_correctly_rounded_one():
    assert arith.check_fma(np.random.default_rng(3), 20_000) == 0


def test_the_hard_inputs_lie_next_to_a_midpoint():
    from fractions import Fraction

    rng = np.random.default_rng(5)
    a, b = arith.hard_quotients(rng, 50)
    assert a.size == 50
    for x, y in zip(a.tolist(), b.tolist()):
        q = Fraction(x) / Fraction(y)
        c = np.float32(float(q))
        mids = [(Fraction(float(c)) + Fraction(float(np.nextafter(c, np.float32(s))))) / 2
                for s in (np.inf, -np.inf)]
        assert min(abs(q - m) for m in mids) < abs(q) * 2.0 ** -36


def test_the_integer_bf16_rounding_is_torch_s_cast():
    import torch

    rng = np.random.default_rng(2)
    u = rng.integers(0, 1 << 32, size=1 << 20, dtype=np.int64).astype(np.uint32)
    u[:6] = [0x3F808000, 0x3F818000, 0x7F7F8000, 0xFF7F8000, 0x7F800000, 0x7FC00001]
    x = u.view(np.float32)
    want = torch.from_numpy(x).bfloat16().view(torch.int16).numpy()
    want = np.where(np.isnan(x), np.int16(0x7FC0), want)
    assert (arith.rtne_bf16(x) == want).all()


@pytest.mark.skipif(not (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")),
                    reason="no C++ compiler")
def test_the_shader_s_own_source_compiled_as_cpp(tmp_path, monkeypatch):
    exe = arith.compiled_shader_helpers(tmp_path)
    said = arith.check_compiled(exe, 150_000, 13, arith.GUESS)
    assert said["values"] > 100_000
    assert said["sqrt_differences"] == said["div_differences"] == said["rtne_differences"] == 0
    # and the check would see a wrong one: the division without one of its midpoint tests
    from poros.metal import expert_adamw_shader as shader

    monkeypatch.setattr(shader, "EXPERT_ADAMW_SOURCE", shader.EXPERT_ADAMW_SOURCE.replace(
        "else if (r > (0.5f * (q - dn)) * mb) q = dn;", ""))
    wrong = arith.check_compiled(arith.compiled_shader_helpers(tmp_path), 150_000, 13,
                                 arith.GUESS)
    assert wrong["div_differences"] > 0
