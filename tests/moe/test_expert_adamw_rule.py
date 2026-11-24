"""The rule ExpertAdamW stores (poros/train/expert_adamw.py's written definition), held to it:
its reference implementation against an independent one in numpy, and the reference's bits the
same whatever CPU kernels torch dispatches to (ATEN_CPU_CAPABILITY default, AVX2, AVX-512).

numpy's float32 +, -, *, / and square root are each one IEEE operation, correctly rounded, so a
numpy transcription of the rule computes the rule; torch's reference uses torch's basic
operations and corrects its square root and division (``_sqrt``, ``_div``), and must agree bit
for bit. The stochastic rounding and its bits are integer functions, shared by both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from poros.train import expert_adamw as ea  # noqa: E402

MIN = np.float32(2.0 ** -126)


def _F(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(np.abs(x) <= MIN, np.float32(0), x).astype(np.float32)


def _numpy_rule(inputs, key, s):
    """The rule, transcribed into numpy float32, one operation at a time."""
    f = {k: _F(inputs[k].float().numpy()) for k in ("h", "l", "g", "m", "v")}
    c = {k: np.float32(x) for k, x in s.items()}
    g, m, v, h, lo = f["g"], f["m"], f["v"], f["h"], f["l"]
    with np.errstate(all="ignore"):
        m1 = _F(m + _F(c["w1"] * _F(g - m)))
        v1 = _F(_F(v * c["b2"]) + _F(_F(c["w2"] * g) * g))
        w = _F(_F(h + lo) * c["decay"])
        d = _F(_F(_F(np.sqrt(v1)) / c["bc2s"]) + c["eps"])
        w = _F(w + _F(_F(c["neg_step"] * m1) / d))
    wt = torch.from_numpy(w)
    hi = wt.bfloat16()
    hi = torch.where(torch.isnan(wt), torch.tensor(0x7FC0, dtype=torch.int16)
                     .view(torch.bfloat16), hi)
    with np.errstate(all="ignore"):
        r = torch.from_numpy(_F(w - hi.float().numpy()))
    n = wt.numel()
    rem = ea._stochastic_round_bf16(r, ea._keyed_bits(key, 0, n, ea.DRAW_REMAINDER))
    rem = torch.where(torch.isfinite(hi), rem, torch.zeros((), dtype=torch.bfloat16))
    return {"h": hi, "l": rem,
            "m": ea._stochastic_round_bf16(torch.from_numpy(m1), ea._keyed_bits(key, 0, n,
                                                                             ea.DRAW_M)),
            "v": ea._stochastic_round_bf16(torch.from_numpy(v1), ea._keyed_bits(key, 0, n,
                                                                             ea.DRAW_V)),
            "g": inputs["g"]}


@pytest.mark.parametrize(("lr", "betas", "eps", "wd"), [
    (2e-4, (0.9, 0.999), 1e-8, 0.01), (2e-5, (0.99, 0.95), 1e-6, 0.1),
    (1e-3, (0.0, 0.5), 0.0, 0.0)])
def test_the_reference_is_the_rule_as_numpy_computes_it(lr, betas, eps, wd):
    inputs = ea._probe_inputs(ea.BLOCK + 4321, 7)
    s = ea._scalars(lr, betas[0], betas[1], eps, wd, 3)
    key = ea.update_key(1, 3, "w")
    want = _numpy_rule(inputs, key, s)
    got = ea._run_steps("reference", "cpu", inputs, 1, key, s)
    assert ea._first_difference(got, want, inputs) is None, ea._first_difference(got, want,
                                                                                 inputs)


def test_the_corrected_square_root_and_division_are_numpy_s():
    gen = torch.Generator().manual_seed(2)
    n = 1 << 20
    x = (torch.rand(n, generator=gen) + 0.5) * torch.exp2(torch.randint(-140, 127, (n,),
                                                                        generator=gen).float())
    y = (torch.rand(n, generator=gen) + 0.5) * torch.exp2(torch.randint(-60, 60, (n,),
                                                                        generator=gen).float())
    x[:3] = torch.tensor([0.0, float("inf"), 3.4028234663852886e38])
    with np.errstate(all="ignore"):
        assert np.array_equal(ea._sqrt(x).numpy(), np.sqrt(x.numpy()))
        q = (x.numpy() / y.numpy()).astype(np.float32)
    assert np.array_equal(ea._div(x, y).numpy(), q)
    assert np.array_equal(ea._div(-x, y).numpy(), -q)


_DIGEST = r"""
import hashlib, json, sys, torch
sys.path.insert(0, sys.argv[1])
from poros.train import expert_adamw as ea
inputs = ea._probe_inputs(ea.BLOCK + 4321, 7)
s = ea._scalars(2e-4, 0.9, 0.999, 1e-8, 0.01, 3)
got = ea._run_steps("reference", "cpu", inputs, 3, ea.update_key(1, 3, "w"), s)
h = hashlib.sha256()
for k in ("h", "l", "m", "v"):
    h.update(got[k].view(torch.int16).numpy().tobytes())
print(json.dumps({"capability": torch.backends.cpu.get_cpu_capability(), "digest": h.hexdigest()}))
"""


# x86 kernel levels torch dispatches to, lowest first: a child may be told any level at or below
# the one torch picks here by default, never one above (an AVX-512 kernel on a CPU without it dies
# of an illegal instruction)
LEVELS = ("DEFAULT", "AVX2", "AVX512")
_KEPT = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "OMP_NUM_THREADS")


def _child(code: str, *args: str, capability: str | None = None):
    """A Python child with a minimal environment of named variables (never a copy of this
    process's: a failure must not be able to show what the shell holds), and the CPU kernel level
    asked for. Returns (returncode, its last stdout line, the tail of its stderr)."""
    env = {k: os.environ[k] for k in _KEPT if k in os.environ}
    if capability:
        env["ATEN_CPU_CAPABILITY"] = capability.lower()
    done = subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True,
                          text=True)
    lines = done.stdout.strip().splitlines()
    return done.returncode, lines[-1] if lines else "", done.stderr[-2000:]


_CAPABILITY = "import torch; print(torch.backends.cpu.get_cpu_capability())"


def _levels_here() -> list:
    """The kernel levels this CPU runs: torch's own default choice here and every level below
    it (an empty list where its choice is not an x86 level)."""
    code, top, err = _child(_CAPABILITY)
    assert code == 0, f"the child that asks torch for its CPU capability failed: {err}"
    return _levels_up_to(top)


def _levels_up_to(top: str) -> list:
    return list(LEVELS[:LEVELS.index(top) + 1]) if top in LEVELS else []


def test_a_child_is_never_asked_for_a_kernel_level_above_torch_s_own_choice():
    """An AVX2-only CPU (an EPYC 7502) is never sent to AVX-512: the case that died of
    SIGILL. And here, the levels end at torch's own default."""
    assert _levels_up_to("AVX2") == ["DEFAULT", "AVX2"]
    assert _levels_up_to("DEFAULT") == ["DEFAULT"]
    assert _levels_up_to("AVX512") == ["DEFAULT", "AVX2", "AVX512"]
    assert _levels_up_to("SVE256") == []
    code, top, _ = _child(_CAPABILITY)
    assert code == 0 and _levels_here()[-1:] == ([top] if top in LEVELS else [])


def test_the_reference_s_bits_do_not_depend_on_the_cpu_kernels():
    """The same three updates under every CPU kernel level this CPU runs (torch's default and
    each below it): one digest. (torch's own lerp_ and addcmul_ differ between them: they fuse
    the multiply-add in the AVX2 and AVX-512 kernels only.)"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    levels = _levels_here()
    if len(levels) < 2:
        pytest.skip(f"this CPU runs one x86 kernel level here ({levels or 'none'}): nothing to "
                    "compare")
    seen = {}
    for capability in levels:
        code, out, err = _child(_DIGEST, root, capability=capability)
        assert code == 0, f"the child at {capability} failed (exit {code}): {err}"
        said = json.loads(out)
        assert said["capability"] == capability, (capability, said["capability"])
        seen[said["capability"]] = said["digest"]
    assert len(set(seen.values())) == 1, seen


_LERP = r"""
import hashlib, json, torch
g = torch.Generator().manual_seed(0)
a, b = torch.randn(1 << 16, generator=g) * 1e-5, torch.randn(1 << 16, generator=g) * 1e-4
print(json.dumps({"capability": torch.backends.cpu.get_cpu_capability(),
                  "digest": hashlib.sha256(a.lerp_(b, 0.1).numpy().tobytes()).hexdigest()}))
"""


def test_torch_s_own_lerp_differs_between_cpu_kernels_here():
    """The control: what the reference avoids. Skipped where this CPU has one kernel set only."""
    seen = {}
    for capability in _levels_here():
        code, out, err = _child(_LERP, capability=capability)
        assert code == 0, f"the child at {capability} failed (exit {code}): {err}"
        said = json.loads(out)
        seen[said["capability"]] = said["digest"]
    if len(seen) < 2:
        pytest.skip(f"one CPU kernel set here: {seen}")
    assert len(set(seen.values())) > 1, seen
