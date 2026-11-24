"""``poros.train.expert_adamw`` at its edges: damaged or foreign states, gradients, scalars and
parameters it refuses in words before anything reaches a kernel, and the edges of its own
arithmetic."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from poros.core.errors import PorosConfigError  # noqa: E402
from poros.train import expert_adamw as ea  # noqa: E402
from poros.train.expert_adamw import ExpertAdamW  # noqa: E402


def _param(shape=(6, 4), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter((torch.randn(shape, generator=g) * 0.02).bfloat16())


def _stepped(shape=(6, 4), dtype=torch.bfloat16, steps=2, **kw):
    p = torch.nn.Parameter(_param(shape).detach().to(dtype))
    opt = ExpertAdamW([("w", p)], **{"lr": 1e-3, **kw})
    for _ in range(steps):
        p.grad = torch.full_like(p, 1e-3)
        opt.step()
    return p, opt


# ---- a saved state of other shapes, dtypes or indices; a gradient unlike its parameter ----


@pytest.mark.parametrize(("damage", "said"), [
    (lambda st: st["state"][0].update(exp_avg=torch.zeros(5, 4, dtype=torch.bfloat16)),
     r"exp_avg for w is torch.bfloat16 of shape \(5, 4\) \(20 elements\)"),
    (lambda st: st["state"][0].update(exp_avg_sq=torch.zeros(4, 6, dtype=torch.bfloat16)),
     r"exp_avg_sq for w is torch.bfloat16 of shape \(4, 6\)"),
    (lambda st: st["state"][0].update(remainder=torch.zeros(6, 4)),
     r"remainder for w is torch.float32"),
    (lambda st: st["state"][0].update(exp_avg=torch.zeros(24, dtype=torch.bfloat16)),
     r"shape \(24,\) \(24 elements\), and w is torch.bfloat16 of shape \(6, 4\)"),
    (lambda st: st["state"][0].pop("exp_avg_sq"), "no exp_avg_sq tensor for w"),
    (lambda st: st["state"][0].update(step=torch.tensor(-3)), "step count for w is"),
    (lambda st: st["state"][0].update(step=torch.tensor(2.5)), "step count for w is"),
    (lambda st: st["state"][0].update(step=2), "step count for w is 2; this optimizer saves"),
    (lambda st: st["state"][0].update(step=torch.tensor(2 ** 63 - 1)), "from 0 to 2\\*\\*62"),
    (lambda st: st["state"][0].update(step=torch.tensor([2, 2])), "one int64 tensor"),
    (lambda st: st["state"].update({-1: st["state"].pop(0)}), "parameter -1"),
    (lambda st: st["state"].update({1: st["state"].pop(0)}), "parameter 1"),
    (lambda st: st["state"].update({"0": st["state"].pop(0)}), "parameter '0'"),
])
def test_a_damaged_saved_state_is_refused_before_anything_is_restored(damage, said):
    p, opt = _stepped()
    sd = copy.deepcopy(opt.state_dict())
    damage(sd)
    fresh = ExpertAdamW([("w", p)], lr=1e-3)
    with pytest.raises(PorosConfigError, match=said):
        fresh.load_state_dict(sd)
    assert not fresh.state  # nothing restored


def test_an_fp32_parameter_s_state_is_checked_too():
    p, opt = _stepped(dtype=torch.float32)
    sd = copy.deepcopy(opt.state_dict())
    sd["state"][0]["exp_avg"] = torch.zeros(6, 4, dtype=torch.bfloat16)
    with pytest.raises(PorosConfigError, match="exp_avg for w is torch.bfloat16"):
        ExpertAdamW([("w", p)], lr=1e-3).load_state_dict(sd)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("grad", "said"), [
    (lambda p: torch.zeros(p.numel(), dtype=p.dtype), r"shape \(24,\)"),
    (lambda p: torch.zeros(p.shape, dtype=torch.float16), "torch.float16"),
])
def test_a_gradient_unlike_its_parameter_is_refused_before_a_kernel(dtype, grad, said):
    p, opt = _stepped(dtype=dtype, steps=1)
    before = {k: v.clone() for k, v in opt.state[p].items()}
    weight = p.detach().clone()
    p.grad.data = grad(p)  # past torch's own check on assignment, as a hook or a cast could
    with pytest.raises(RuntimeError, match=f"the gradient of w is .*{said}"):
        opt.step()
    assert torch.equal(p.detach(), weight)
    assert all(torch.equal(opt.state[p][k], v) for k, v in before.items())


# ---- a scalar that is not an fp32 normal or zero, at construction and from a scheduler ----


@pytest.mark.parametrize(("kw", "said"), [
    (dict(lr=1e-45), r"lr / \(1 - beta1\) \(the first update's step\) is 1.0+2e-44, nonzero but "
                     r"at most 2\*\*-126"),
    (dict(lr=1e-45, betas=(0.0, 0.999)), r"is 1e-45, nonzero but at most 2\*\*-126"),
    (dict(lr=2.0 ** -126), "at most 2\\*\\*-126"),
    (dict(lr=1e39), r"lr = 1e\+39 is above 1"),
    (dict(lr=1.0, weight_decay=1e39), r"1 - lr \* weight_decay is .*past fp32's largest"),
    (dict(lr=2.0 ** 116, betas=(0.0, 0.0), eps=2.0 ** -39), "is above 1, the largest it takes"),
    (dict(lr=1.0000001), "lr = 1.0000001 is above 1"),
    (dict(lr=1e-3, betas=(0.9, 1e-40)), "beta2 is 1e-40, nonzero"),
    (dict(lr=1e-3, eps=1e39), "eps is 1e\\+39, past fp32's largest"),
])
def test_a_scalar_the_rule_cannot_hold_is_refused_at_construction(kw, said):
    with pytest.raises(ValueError, match=said):
        ExpertAdamW([("w", _param())], **kw)


@pytest.mark.parametrize(("kw", "said"), [  # inputs each under 2**-150, so fp32
    (dict(lr=1e-47), r"lr / \(1 - beta1\).* is 1.0+2e-46"),  # rounds it to 0: let pass
    (dict(lr=5e-46, betas=(0.0, 0.999)), "is 5e-46"),
    (dict(lr=1e-3, betas=(0.9, 1e-46)), "beta2 is 1e-46"),
])
def test_a_nonzero_scalar_fp32_rounds_to_zero_is_refused_not_zeroed(kw, said):
    with pytest.raises(ValueError, match=f"{said}, nonzero but at most 2\\*\\*-126.*set it "
                       "to 0 or above 2\\*\\*-126"):
        ExpertAdamW([("w", _param())], **kw)
    p, opt = _stepped(steps=1)  # and from a decaying scheduler, before anything is updated
    opt.param_groups[0]["lr"] = 1e-47
    p.grad = torch.full_like(p, 1e-3)
    with pytest.raises(ValueError, match="at this step, lr / "):
        opt.step()


def test_a_tiny_lr_times_weight_decay_is_no_scalar_and_zeros_are_accepted():
    for kw in (dict(lr=1e-30, weight_decay=1e-30), dict(lr=0.0), dict(lr=1e-3, betas=(0.0, 0.0)),
               dict(lr=1e-3, weight_decay=0.0)):
        p = _param()
        p.grad = torch.full_like(p, 1e-3)
        ExpertAdamW([("w", p)], **kw).step()


def test_a_scheduler_s_lr_the_rule_cannot_hold_is_refused_before_any_update():
    p, opt = _stepped(steps=1)
    state = {k: v.clone() for k, v in opt.state[p].items()}
    weight = p.detach().clone()
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1e-43 if step >= 1 else 1.0)
    sched.step()
    p.grad = torch.full_like(p, 1e-3)
    with pytest.raises(ValueError, match="at this step, lr .*Nothing was updated"):
        opt.step()
    assert torch.equal(p.detach(), weight)
    assert all(torch.equal(opt.state[p][k], v) for k, v in state.items())


def test_every_scalar_follows_f_and_valid_ones_are_unchanged():
    assert ea._f32(1e-40) == 0.0 and ea._f32(-(2.0 ** -126)) == 0.0
    assert ea._f32(2.0 ** -125) == 2.0 ** -125 and ea._f32(1e-3) == 0.0010000000474974513
    with pytest.raises(ValueError, match="past fp32's largest"):
        ea._f32(1e39)
    assert ea._f32(float("inf")) == float("inf")


# ---- a parameter of no elements; a parameter moved after the optimizer was built ---------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_parameter_of_no_elements_is_skipped_on_every_path(dtype, monkeypatch):
    empty = torch.nn.Parameter(torch.zeros(0, 8, dtype=dtype))
    full = _param((4, 4))
    opt = ExpertAdamW([("e", empty), ("w", full)], lr=1e-3)
    for _ in range(2):
        empty.grad, full.grad = torch.zeros_like(empty), torch.full_like(full, 1e-3)
        opt.step()
    assert int(opt.state[empty]["step"]) == 2 and opt.state[empty]["exp_avg"].numel() == 0
    # the kernel path returns before a launch of no blocks
    opt.implementation["cpu"] = {"device": "cpu", "path": "kernel", "why": "test"}
    monkeypatch.setitem(ea._KERNEL, "cpu", lambda *a: pytest.fail("a kernel ran on 0 elements"))
    opt.param_groups[0]["params"] = [empty]
    empty.grad = torch.zeros_like(empty)
    opt.step()
    assert ea._keyed_bits(7, 0, 0, ea.DRAW_V).shape == (0,)
    assert ea._keyed_bits(7, ea.BLOCK, 0, ea.DRAW_M).dtype == torch.int32
    assert ea._coefficients(7, 0).numel() == 0
    h = torch.zeros(0, dtype=torch.bfloat16)
    out = [h.clone() for _ in range(4)]
    ea._reference_bf16(h, h, h, h, h, 7, 0, ea._scalars(1e-3, 0.9, 0.999, 1e-8, 0.01, 1), out)


def test_a_bf16_parameter_moved_after_the_optimizer_was_built_is_refused():
    p = _param()
    opt = ExpertAdamW([("w", p)], lr=1e-3)
    # as if built while the model was on a GPU and the model moved to the CPU after
    opt.implementation = {"cuda:0": opt.implementation.pop("cpu")}
    p.grad = torch.zeros_like(p)
    with pytest.raises(RuntimeError, match="w is on cpu, and the optimizer was built for its "
                                           "parameters on cuda:0"):
        opt.step()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_state_on_another_device_than_its_parameter_is_refused(dtype):
    p, opt = _stepped(dtype=dtype, steps=1)
    opt.state[p]["exp_avg"] = opt.state[p]["exp_avg"].to("meta")
    p.grad = torch.zeros_like(p)
    with pytest.raises(RuntimeError, match="w is on cpu and its exp_avg on meta"):
        opt.step()
    assert int(opt.state[p]["step"]) == 1


# ---- the corrected division and root at the largest values, infinities, and under FTZ ----

FLT_MAX = 3.4028234663852886e38


def _rounded_f32(x):
    """The exact rational x rounded to the nearest fp32, even on a tie (inf past the largest's
    upper midpoint), with every comparison exact."""
    from fractions import Fraction

    import numpy as np

    if x == 0:
        return 0.0
    mag = abs(x)
    if mag >= Fraction(2 ** 128 - 2 ** 103):
        out = float("inf")
    else:
        c = np.float32(float(mag))  # within an fp32 step
        for _ in range(2):
            up = np.nextafter(c, np.float32(np.inf))
            down = np.nextafter(c, np.float32(0))
            if np.isinf(up):  # c is the largest: above its midpoint was inf, handled
                pass
            elif mag > (Fraction(float(c)) + Fraction(float(up))) / 2 or (
                    mag == (Fraction(float(c)) + Fraction(float(up))) / 2
                    and int(np.float32(up).view(np.int32)) % 2 == 0):
                c = up
            elif mag < (Fraction(float(c)) + Fraction(float(down))) / 2 or (
                    mag == (Fraction(float(c)) + Fraction(float(down))) / 2
                    and int(np.float32(down).view(np.int32)) % 2 == 0):
                c = down
        out = float(c)
    return out if x > 0 else -out


def _boundary_pairs():
    import numpy as np

    big = [FLT_MAX, float(np.nextafter(np.float32(FLT_MAX), np.float32(0))), 2.0 ** 127,
           1.5 * 2.0 ** 127]
    near_one = [1.0, 1 - 2.0 ** -24, 1 + 2.0 ** -23, 1 - 2.0 ** -23, 0.5, 0.75 + 2.0 ** -24]
    tiny = [2.0 ** -126, 2.0 ** -125, 1.5 * 2.0 ** -126, 2.0 ** -149, 2.0 ** -127, 3 * 2.0 ** -149]
    pairs = [(a, b) for a in big + [-x for x in big] for b in near_one + tiny]
    pairs += [(a, b) for a in tiny + near_one for b in big + near_one]
    gen = torch.Generator().manual_seed(3)
    # random significands at the overflow edge and at F's edge
    sig = (torch.rand(4000, generator=gen, dtype=torch.float64) + 1).float()
    sig2 = (torch.rand(4000, generator=gen, dtype=torch.float64) + 1).float()
    pairs += [(float(x) * 2.0 ** 127, float(y)) for x, y in zip(sig[:2000], sig2[:2000])]
    pairs += [(float(x) * 2.0 ** -126, float(y)) for x, y in zip(sig[2000:], sig2[2000:])]
    return pairs


def test_the_corrected_division_and_root_are_the_correctly_rounded_ones_at_the_edges():
    from fractions import Fraction

    pairs = _boundary_pairs()
    a = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    q = ea._div(a, b)
    for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
        want = _rounded_f32(Fraction(x) / Fraction(y))
        if abs(want) > 2.0 ** -126:  # F reads the rest as zero, whatever their rounding
            assert q[i].item() == want, (x, y, q[i].item(), want)
    special = ea._div(torch.tensor([float("inf"), -FLT_MAX, 1.0, FLT_MAX]),
                      torch.tensor([1.0, 2.0 ** -126, float("inf"), 1 - 2.0 ** -24]))
    assert special.tolist() == [float("inf"), float("-inf"), 0.0, float("inf")]
    x = torch.tensor([FLT_MAX, float("inf"), 0.0, 2.0 ** -126, 2.0 ** -125 * 1.5, 1e38],
                     dtype=torch.float32)
    r = ea._sqrt(x)
    assert r[1].item() == float("inf")
    for got, xi in zip(r.tolist()[:1] + r.tolist()[2:], x.tolist()[:1] + x.tolist()[2:]):
        lo, hi = Fraction(got), Fraction(xi)
        up = Fraction(float(torch.nextafter(torch.tensor(got), torch.tensor(float("inf")))))
        down = Fraction(float(torch.nextafter(torch.tensor(got), torch.tensor(0.0))))
        assert ((lo + down) / 2) ** 2 < hi <= ((lo + up) / 2) ** 2 or xi == 0, (xi, got)


def test_no_quotient_of_fp32_values_lies_in_fp64_s_window_below_the_overflow_midpoint():
    """fp64's quotient rounds up onto 2**128 - 2**103 (then fp32's to inf) only for an exact
    quotient within 2**-53 of it, relatively. a / b = (ma / mb) * 2**k with 24-bit significands;
    below the midpoint (1 - 2**-25) * 2**128 means ma <= mb - 1, which falls short by
    (2**25 - mb) / (mb * (2**25 - 1)) relatively: checked for every mb, it is above 2**-26, so no
    pair meets the window (and ``_div`` compares exactly anyway)."""
    import numpy as np

    mb = np.arange(2 ** 23, 2 ** 24, dtype=np.int64)
    gap = (2 ** 25 - mb) / (mb.astype(np.float64) * (2 ** 25 - 1))
    assert gap.min() > 2.0 ** -26 > 2.0 ** -53


def test_the_reference_stores_the_same_bits_with_the_cpu_flushing_subnormals():
    """torch.set_flush_denormal(True) (the CPU's FTZ and DAZ): the reference reads every value
    at most 2**-126 as zero after every operation and corrects its root and division in fp64,
    whose midpoints are far from fp64's subnormals, so it stores the bits it stores without:
    over the probe at every setting, and the division and root at the edges, after F."""
    if not torch.set_flush_denormal(True):
        pytest.skip("this CPU has no flush-to-zero mode")
    torch.set_flush_denormal(False)

    def run():
        out = []
        for i, (lr, b1, b2, eps, wd, steps) in enumerate(ea.PROBE_SETTINGS):
            s = ea._scalars(lr, b1, b2, eps, wd, 7)
            for n in (4096, 4099):
                inputs = ea._probe_inputs(n, i)
                out.append(ea._run_steps("reference", "cpu", inputs, steps, 11 + i, s))
        pairs = _boundary_pairs()
        a = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
        b = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        edge = torch.tensor([0.0, 1e-45, 2.0 ** -126, 2.0 ** -125, 1e-38, FLT_MAX,
                             float("inf")], dtype=torch.float32)
        # as the rule meets them: every operand already through F
        a, b, edge = ea._flush(a), ea._flush(b), ea._flush(edge)
        b = torch.where(b == 0, torch.ones_like(b), b)  # the rule divides by d >= eps only
        return out, ea._flush(ea._div(a, b)), ea._flush(ea._sqrt(edge))

    plain = run()
    torch.set_flush_denormal(True)
    try:
        flushed = run()
    finally:
        torch.set_flush_denormal(False)
    for got, want in zip(flushed[0], plain[0]):
        for k in ("h", "l", "m", "v"):
            assert torch.equal(got[k].view(torch.int16), want[k].view(torch.int16)), k
    for got, want in zip(flushed[1:], plain[1:]):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))


# ---- a probe that raises is tried once more, then stops the run; it is never kept ---------


def _probing(monkeypatch, outcomes):
    """``_probe_kernel`` on the CPU as if it had a kernel, its probe giving ``outcomes`` in turn
    (an exception raised, else returned)."""
    calls = []
    monkeypatch.setattr(ea, "_PROBED", {})
    monkeypatch.setattr(ea, "_kernel_unavailable", lambda kind: None)
    freed = []
    monkeypatch.setattr(ea, "_free_memory", lambda dev: freed.append(str(dev)))

    def probe(dev):
        calls.append(str(dev))
        out = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(ea, "_probe_differences", probe)
    return calls, freed


def test_a_probe_that_raises_once_is_tried_again_after_freeing_memory(monkeypatch):
    calls, freed = _probing(monkeypatch, [torch.OutOfMemoryError("out of memory"), None])
    assert ea._probe_kernel("cpu") == (True, "the kernel stores the reference's bits")
    assert calls == ["cpu", "cpu"] and freed == ["cpu"]
    assert ea._probe_kernel("cpu")[0] and len(calls) == 2  # a definite verdict is kept


def test_a_probe_that_raises_twice_stops_the_run_in_words_and_is_not_kept(monkeypatch):
    calls, freed = _probing(monkeypatch, [torch.OutOfMemoryError("out of memory"),
                                          RuntimeError("launch failed")])
    with pytest.raises(RuntimeError, match=r"could not check its fused kernel on cpu: the "
                       r"check raised OutOfMemoryError \(out of memory\), and again after "
                       r".*RuntimeError: launch failed\. The run stops"):
        ea._probe_kernel("cpu")
    assert "cpu" not in ea._PROBED and freed == ["cpu"]
    with pytest.raises(RuntimeError):  # the next optimizer probes afresh
        ea._probe_kernel("cpu")
    assert len(calls) == 4


def test_a_kernel_whose_bits_differ_is_a_kept_verdict(monkeypatch):
    calls, _ = _probing(monkeypatch, ["setting 0: 3 elements of h differ"])
    ok, why = ea._probe_kernel("cpu")
    assert not ok and "does not store the reference's bits: setting 0" in why
    ea._probe_kernel("cpu")
    assert len(calls) == 1


# ---- the caller's state is left as it was; the saved settings are checked as at build ----


def test_loading_a_state_leaves_the_caller_s_dict_as_it_was():
    p, opt = _stepped()
    sd = opt.state_dict()
    tensors = {k: v for k, v in sd["state"][0].items()}
    groups = [dict(g) for g in sd["param_groups"]]
    fresh = ExpertAdamW([("w", p)], lr=1e-3)
    fresh.load_state_dict(sd)
    assert all(sd["state"][0][k] is v for k, v in tensors.items())
    assert sd["param_groups"] == groups
    values = {k: v.clone() for k, v in tensors.items()}
    p.grad = torch.full_like(p, 1e-3)
    fresh.step()  # updates its state in place: its own copy, not the caller's
    assert all(torch.equal(tensors[k], values[k]) for k in tensors)
    assert int(fresh.state[p]["step"]) == 3


@pytest.mark.parametrize(("key", "value", "said"), [
    ("betas", (1.0, 0.999), r"betas\[0\] = 1.0"),
    ("betas", (0.9,), "must be numbers"),
    ("lr", -1.0, "lr = -1.0"),
    ("lr", float("nan"), "lr = nan"),
    ("eps", 1e-30, "eps = 1e-30 is under"),
    ("weight_decay", "a lot", "must be numbers"),
    ("lr", 1e-45, "nonzero but at most 2\\*\\*-126"),
])
def test_a_saved_setting_construction_would_refuse_is_refused(key, value, said):
    p, opt = _stepped()
    sd = copy.deepcopy(opt.state_dict())
    sd["param_groups"][0][key] = value
    fresh = ExpertAdamW([("w", p)], lr=1e-3)
    with pytest.raises(PorosConfigError, match=f"settings are refused: .*{said}"):
        fresh.load_state_dict(sd)
    assert not fresh.state and fresh.param_groups[0]["betas"] == (0.9, 0.999)


# ---- smaller edges -----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [True, False])
def test_a_bool_seed_is_refused(seed):
    with pytest.raises(ValueError, match=f"seed {seed!r}"):
        ExpertAdamW([("w", _param())], lr=1e-3, seed=seed)


def test_the_shader_s_f_threshold_is_2_to_the_minus_126_written_exactly():
    from pathlib import Path

    source = (Path(ea.__file__).parent.parent / "metal" / "expert_adamw_shader.py").read_text()
    assert "fabs(x) <= 0x1p-126f ? 0.0f : x" in source
    assert float.fromhex("0x1p-126") == ea.FLT_MIN


def test_a_gradient_view_is_copied_once_with_the_same_bits(monkeypatch):
    copies = []
    real = ea._own
    monkeypatch.setattr(ea, "_own", lambda x: copies.append(x.is_contiguous()) or real(x))
    shape = (6, 4)
    a, b = _param(shape), _param(shape)
    oa = ExpertAdamW([("w", a)], lr=1e-3, seed=3)
    ob = ExpertAdamW([("w", b)], lr=1e-3, seed=3)
    big = (torch.randn(3, *shape) * 1e-3).bfloat16()
    views = (big[1],  # contiguous, offset in a larger buffer
             big[2].t().contiguous().t())  # not contiguous
    for view in views:
        a.grad = view.contiguous().clone()
        b.grad = view
        oa.step()
        ob.step()
    assert torch.equal(a, b) and copies == [True, False]
    for x in (torch.zeros(4, 6).t(), torch.zeros(10)[2:8]):
        y = ea._own(x)
        assert y.is_contiguous() and y.storage_offset() == 0 and torch.equal(x, y)
        assert y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()


def test_a_seed_or_block_refusal_names_the_group_that_differs():
    p, opt = _stepped(seed=5)
    sd = copy.deepcopy(opt.state_dict())
    sd["param_groups"][0]["seed"] = 9
    with pytest.raises(PorosConfigError, match="written with seed 9, and this optimizer has 5"):
        ExpertAdamW([("w", p)], lr=1e-3, seed=5).load_state_dict(sd)


# ---- a state restored onto other parameters than it was checked against --------


def _three():
    shapes = ((4, 8), (8, 2), (2, 8))
    ps = [torch.nn.Parameter(_param(s, i).detach()) for i, s in enumerate(shapes)]
    opt = ExpertAdamW([(f"p{i}", p) for i, p in enumerate(ps)], lr=1e-3)
    for _ in range(2):
        for p in ps:
            p.grad = torch.full_like(p, 1e-3)
        opt.step()
    return ps, opt


@pytest.mark.parametrize(("order", "said"), [
    ([1, 0, 2], r"group 0 restores its entries onto parameters \[1, 0, 2\]"),
    ([2, 1, 0], r"onto parameters \[2, 1, 0\]"),  # 4x8 and 2x8 ... same counts as 8x2
    ([10, 11, 12], r"onto parameters \[10, 11, 12\].*numbered 0 to 2, in order"),
])
def test_a_state_whose_groups_restore_onto_other_parameters_is_refused(order, said):
    ps, opt = _three()
    sd = copy.deepcopy(opt.state_dict())
    sd["param_groups"][0]["params"] = order
    fresh = ExpertAdamW([(f"p{i}", p) for i, p in enumerate(ps)], lr=1e-3)
    with pytest.raises(PorosConfigError, match=said):
        fresh.load_state_dict(sd)
    assert not fresh.state


@pytest.mark.parametrize("damage", ["missing", "extra", "empty"])
def test_a_state_missing_an_entry_or_holding_an_extra_one_is_refused(damage):
    ps, opt = _three()
    sd = copy.deepcopy(opt.state_dict())
    if damage == "missing":
        del sd["state"][1]
        said = r"no entry for 1 of its 3 parameters, the first p1 \(number 1\)"
    elif damage == "extra":
        sd["state"][3] = copy.deepcopy(sd["state"][0])
        said = "entry for parameter 3"
    else:
        sd["state"][2] = {}
        said = "the first p2"
    fresh = ExpertAdamW([(f"p{i}", p) for i, p in enumerate(ps)], lr=1e-3)
    with pytest.raises(PorosConfigError, match=said):
        fresh.load_state_dict(sd)
    assert not fresh.state


def test_every_parameter_has_an_entry_even_one_without_a_gradient_and_restores_by_position():
    ps, _ = _three()
    idle = _param((3, 3), 9)
    opt = ExpertAdamW([("p0", ps[0]), ("idle", idle)], lr=1e-3)
    ps[0].grad = torch.full_like(ps[0], 1e-3)
    opt.step()
    sd = opt.state_dict()
    assert sorted(sd["state"]) == [0, 1] and int(sd["state"][1]["step"]) == 0
    assert not opt.state[idle]["exp_avg"].any()
    again = ExpertAdamW([("p0", ps[0]), ("idle", idle)], lr=1e-3)
    again.load_state_dict(sd)
    assert torch.equal(again.state[ps[0]]["exp_avg"], opt.state[ps[0]]["exp_avg"])
    assert again.state[idle]["exp_avg"].shape == (3, 3)


# ---- every setting checked at every step, as at construction --------------------


@pytest.mark.parametrize(("key", "value", "said"), [
    ("lr", -1e-4, "lr = -0.0001"),  # a linear decay past its end
    ("eps", 0.0, "eps = 0.0 is under"),
    ("eps", 1e-20, "eps = 1e-20 is under"),
    ("betas", (0.9, 1.0), r"betas\[1\] = 1.0"),  # 1 - beta2 = 0: bc2s 0, a division by zero
    ("betas", (1.0, 0.999), r"betas\[0\] = 1.0"),  # a ZeroDivisionError before
    ("weight_decay", float("nan"), "weight_decay = nan"),
])
def test_a_setting_changed_after_construction_is_refused_before_anything_is_updated(key, value,
                                                                                      said):
    p, opt = _stepped(steps=1)
    state = {k: v.clone() for k, v in opt.state[p].items()}
    weight = p.detach().clone()
    opt.param_groups[0][key] = value
    p.grad = torch.full_like(p, 1e-3)
    with pytest.raises(ValueError, match=f"at this step, {said}.*Nothing was updated"):
        opt.step()
    assert torch.equal(p.detach(), weight)
    assert all(torch.equal(opt.state[p][k], v) for k, v in state.items())


# ---- the kernel launches on, and the timing waits for, the tensors' own card ----


def test_the_triton_launch_is_made_on_the_tensors_card(monkeypatch):
    """Without two cards: a stand-in Triton records the current device at launch, and a
    stand-in torch.cuda.device records which device it was asked to make current."""
    import contextlib
    import importlib.util
    import sys
    import types
    from pathlib import Path

    current, launched = ["cuda:0"], []

    class Launch:
        def __getitem__(self, grid):
            return lambda *a, **kw: launched.append(current[0])

    fake = types.ModuleType("triton")
    fake.jit = lambda f: Launch()
    fake.cdiv = lambda a, b: -(-a // b)
    lang = types.ModuleType("triton.language")
    lang.constexpr = object
    fake.language = lang
    monkeypatch.setitem(sys.modules, "triton", fake)
    monkeypatch.setitem(sys.modules, "triton.language", lang)

    @contextlib.contextmanager
    def device(d):
        was, current[0] = current[0], str(d)
        yield
        current[0] = was

    monkeypatch.setattr(torch.cuda, "device", device)
    path = Path(ea.__file__).parent / "expert_adamw_cuda.py"
    spec = importlib.util.spec_from_file_location("_cuda_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class OnCard1:
        device = torch.device("cuda:1")

        def numel(self):
            return 4096

    module.step(OnCard1(), None, None, None, None, None, w1=0.1, b2=0.999, w2=0.001,
                decay=1.0, bc2s=1.0, eps=1e-8, neg_step=-1e-3)
    assert launched == ["cuda:1"] and current == ["cuda:0"]


def test_the_reference_s_timing_waits_for_its_own_card(monkeypatch):
    synced = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: synced.append(d))
    monkeypatch.setattr(ea, "_run_steps", lambda *a, **kw: None)
    monkeypatch.setattr(ea, "_probe_inputs", lambda n, seed: None)
    ea._reference_seconds("cuda:1")
    assert synced and all(d == torch.device("cuda:1") for d in synced)


# ---- a Mac on either side of a resume; one choice a device; iterables once ------


def test_a_state_a_mac_s_kernel_stepped_is_refused_when_continued_elsewhere():
    from types import SimpleNamespace

    mac = {"mps:0": {"device": "mps", "path": "kernel", "why": "the kernel stores ..."}}
    here = SimpleNamespace(implementation={"cpu": {"device": "cpu", "path": "reference",
                                                   "why": "the CPU runs the reference"}})
    said = ea.other_implementation(mac, here)
    assert said and "stepped by ExpertAdamW's mps:0: kernel, and here it steps with its cpu: " \
        "reference" in said, said
    cuda = SimpleNamespace(implementation={"cuda:0": {"device": "cuda", "path": "reference",
                                                      "why": "t"}})
    assert ea.other_implementation({"cuda:0": {"device": "cuda", "path": "kernel"}}, cuda) is None
    assert ea.other_implementation(mac, SimpleNamespace(implementation=dict(mac))) is None


def test_the_implementation_is_chosen_once_a_device_and_iterables_are_read_once(monkeypatch):
    calls = []
    real = ea._implementation
    monkeypatch.setattr(ea, "_implementation", lambda d: calls.append(str(d)) or real(d))
    opt = ExpertAdamW(((f"w{i}", _param((4,), i)) for i in range(5)), lr=1e-3,
                      betas=(b for b in (0.9, 0.999)))
    assert calls == ["cpu"] and opt.param_groups[0]["betas"] == (0.9, 0.999)
    assert len(opt.param_groups[0]["params"]) == 5

    class Cfg:
        learning_rate, adam_beta1, adam_beta2, adam_epsilon, weight_decay, seed = (
            1e-3, 0.9, 0.999, 1e-8, 0.0, 0)

    built = ea.build_optimizer(((f"e{i}", _param((4,), i)) for i in range(3)), Cfg)
    assert isinstance(built, ExpertAdamW) and len(built.param_groups[0]["params"]) == 3


def test_lr_up_to_1_is_taken_and_a_tiny_lr_stays_within_its_bound():
    """lr 1 is accepted. The lr = 2**-100 case (w = 2**-100, g = 2**-26): F zeroes
    ns * m1 = 2**-126, so the weight does not move, where fp32 AdamW moves it by about 2**-101;
    the difference is under the 2**-87 the rule's comment bounds it by."""
    ExpertAdamW([("w", _param())], lr=1.0)
    w = torch.nn.Parameter(torch.full((4,), 2.0 ** -100).bfloat16())
    opt = ExpertAdamW([("w", w)], lr=2.0 ** -100, betas=(0.0, 0.0), eps=1e-8, weight_decay=0.0)
    w.grad = torch.full_like(w, 2.0 ** -26)
    opt.step()
    held = opt.weights()["w"].double()
    fp32 = 2.0 ** -100 - 2.0 ** -100 * (2.0 ** -26 / (2.0 ** -26 + 1e-8))
    assert bool(((held - fp32).abs() < 2.0 ** -87).all())


def test_a_state_or_parameter_changed_after_a_load_is_refused_before_a_kernel():
    p, opt = _stepped(steps=1)
    opt.state[p]["exp_avg"] = torch.zeros(8, 2, dtype=torch.bfloat16)  # 16 of 24 elements
    p.grad = torch.full_like(p, 1e-3)
    with pytest.raises(RuntimeError, match=r"w's exp_avg is torch.bfloat16 of shape \(8, 2\)"):
        opt.step()
    assert int(opt.state[p]["step"]) == 1
    q, opt2 = _stepped(steps=1)
    q.data = torch.zeros(4, 6, dtype=torch.bfloat16).t()  # same storage start, transposed
    q.grad = torch.zeros_like(q)
    with pytest.raises(RuntimeError, match="no longer contiguous"):
        opt2.step()


# ---- a dense run whose lora_bias trains a 16-bit base's own biases ---------------


def _peft_bias_model(dtype):
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
    model = get_peft_model(Qwen2ForCausalLM(cfg).to(dtype), LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], bias="all",
        task_type="CAUSAL_LM"))
    for n, p in model.named_parameters():  # PEFT's LoRA factors fp32, the biases the base's
        if "lora_" in n:
            p.data = p.data.float()
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


class _Cfg:
    learning_rate, adam_beta1, adam_beta2, adam_epsilon, weight_decay, seed = (
        1e-3, 0.9, 0.999, 1e-8, 0.0, 0)


def test_a_dense_run_s_bf16_biases_get_expert_adamw_and_the_run_says_why():
    named = _peft_bias_model(torch.bfloat16)
    assert any(p.dtype == torch.bfloat16 for _, p in named)
    opt = ea.build_optimizer(named, _Cfg)
    assert isinstance(opt, ExpertAdamW)
    assert opt.notice.startswith("ExpertAdamW, not torch's AdamW, steps this run's trainable "
                                 "tensors: ")
    assert "the base's own bf16 tensors (the first, " in opt.notice and ".bias)" in opt.notice
    assert "where every trainable tensor is fp32 (lora_bias: none)" in opt.notice


def test_a_moe_run_s_expert_adapters_say_nothing_more():
    named = [("model.layers.0.mlp.experts.adapter.lora_A_gate_up", _param((4, 8)))]
    assert ea.build_optimizer(named, _Cfg).notice is None


def test_a_dense_run_s_fp16_biases_are_refused_naming_them_and_lora_bias():
    named = _peft_bias_model(torch.float16)
    with pytest.raises(PorosConfigError, match=r"Trainable tensors in torch.float16 \(the first, "
                       r".*\.bias\)") as err:
        ea.build_optimizer(named, _Cfg)
    assert "lora_bias: none" in err.value.info.remediation


# ---- the older state's refusal says what 1.0.0 does for those adapters ----------


@pytest.mark.parametrize(("key", "now"), [
    ("base_model.model.layers.0.self_attn.q_proj.lora_A", "fp32 and updates them with torch's "
                                                          "AdamW in fp32"),  # a dense native run
    ("model.layers.0.mlp.experts.adapter.lora_A_gate_up", "with a remainder and keyed rounding"),
])
def test_an_older_bf16_checkpoint_is_refused_with_what_1_0_0_does_for_its_adapters(tmp_path, key,
                                                                                   now):
    from poros.preflight.resume import _old_update_rule

    torch.save({key: torch.zeros(2, 2, dtype=torch.bfloat16)}, tmp_path / "adapter_weights.pt")
    said = str(_old_update_rule(str(tmp_path)))
    assert "before 1.0.0" in said and now in said, said
    other = ("with a remainder" if "fp32" in now else "makes those adapters fp32")
    assert other not in said


def test_the_message_where_the_adapters_are_not_known_is_true_of_both():
    from poros.core.checkpoints import OLD_STATE_MESSAGE, old_state_message

    said = OLD_STATE_MESSAGE.format(where="x")
    assert "MoE model's bf16 expert adapters with a remainder" in said
    assert "dense run's adapters fp32 under torch's AdamW" in said
    assert old_state_message("x", None) == said
    p = _param()
    old = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.ones_like(p)
    old.step()
    with pytest.raises(PorosConfigError, match="a remainder and keyed rounding"):
        ExpertAdamW([("w", p)], lr=1e-3).load_state_dict(old.state_dict())


# ---- a state saved before the first step loads as an un-stepped optimizer -------


def _bf16_and_fp32():
    return [("h", _param((4, 4), 1)), ("f", torch.nn.Parameter(torch.zeros(3, 2)))]


def test_a_state_saved_before_the_first_step_loads_and_leaves_the_optimizer_unstepped():
    named = _bf16_and_fp32()
    sd = ExpertAdamW(named, lr=1e-3).state_dict()
    assert sd["state"] == {}
    again = ExpertAdamW(named, lr=1e-3)
    again.load_state_dict(sd)
    assert not any(again.state.values())
    for _, p in named:
        p.grad = torch.full_like(p, 1e-3)
    again.step()
    assert int(again.state[named[0][1]]["step"]) == 1


def test_a_state_with_entries_for_some_parameters_only_is_still_refused():
    named = _bf16_and_fp32()
    opt = ExpertAdamW(named, lr=1e-3)
    for _, p in named:
        p.grad = torch.full_like(p, 1e-3)
    opt.step()
    sd = copy.deepcopy(opt.state_dict())
    del sd["state"][1]
    with pytest.raises(PorosConfigError, match="no entry for 1 of its 2 parameters"):
        ExpertAdamW(named, lr=1e-3).load_state_dict(sd)


def _run_on_expert_adamw(torch_, outdir):
    """tests/test_run_api.py's tiny run, one module's LoRA factors bf16: ExpertAdamW steps it."""
    from poros.train.run import Run
    from tests.test_run_api import Tok
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    torch_.manual_seed(0)
    model, _ = _tiny_peft_qwen2(torch_, targets=("q_proj", "v_proj", "gate_proj"))
    for n, p in model.named_parameters():
        if "lora_" in n and ".q_proj." in n:
            p.data = p.data.bfloat16()
    tr = _make_trainer(torch_, model, str(outdir), max_steps=5)
    tr.config.learning_rate, tr.config.save_steps = 1e-2, 0
    run = Run.from_trainer(tr, Tok())
    assert isinstance(tr.optimizer, ExpertAdamW)
    return run


def test_a_run_on_expert_adamw_saved_at_its_start_resumes_at_step_0_as_a_fresh_run(tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from poros.recipes import sft
    from tests.test_run_api import _chats

    start = _run_on_expert_adamw(torch, tmp_path / "a").save("start")
    fresh = sft.main(_run_on_expert_adamw(torch, tmp_path / "b"), _chats(), steps=4)
    resumed = _run_on_expert_adamw(torch, tmp_path / "c")
    resumed.load(start)
    assert resumed.step == 0
    again = sft.main(resumed, _chats(), steps=4)
    assert again["loss_history"] == fresh["loss_history"]


@pytest.mark.parametrize("steps, said", [(1, "says 1 update was taken"),
                                         (2, "says 2 updates were taken")])
def test_a_checkpoint_that_took_updates_with_an_empty_state_is_refused_by_the_loader(
        tmp_path, steps, said):
    """The marker a save writes is left as it is (the index of the last update taken, so 0
    after one update): a checkpoint after a single update is refused too."""
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from poros.core import checkpoints
    from poros.recipes import sft
    from tests.test_run_api import _chats

    run = _run_on_expert_adamw(torch, tmp_path / "a")
    sft.main(run, _chats(), steps=steps)
    saved = Path(run.save("kept"))
    assert (saved / checkpoints.COMPLETE_MARKER).read_text().strip() == str(steps - 1)
    state = torch.load(saved / "optimizer_state.pt", weights_only=True)
    state["state"] = {}
    torch.save(state, saved / "optimizer_state.pt")
    other = _run_on_expert_adamw(torch, tmp_path / "b")
    with pytest.raises(PorosConfigError, match=said + r", and its optimizer state is empty"):
        other.load(saved)


# ---- betas that are not a pair, refused as every bad setting is --------------


@pytest.mark.parametrize(("betas", "said"), [
    (0.9, "betas 0.9 must be a pair of numbers"),
    (None, "betas None must be a pair of numbers"),
    ((0.9,), r"betas \(0.9,\), eps"),  # one number: the settings check's own words
])
def test_betas_that_are_not_a_pair_are_a_value_error_in_words(betas, said):
    with pytest.raises(ValueError, match=f"ExpertAdamW: .*{said}"):
        ExpertAdamW([("w", _param())], lr=1e-3, betas=betas)


# ---- a dense run on ExpertAdamW with settings it refuses: words, not an error ----


@pytest.mark.parametrize(("field", "value", "said", "setting"), [
    ("learning_rate", 1.5, "lr = 1.5 is above 1", "learning_rate"),
    ("adam_epsilon", 1e-30, "eps = 1e-30 is under", "adam_epsilon"),
    ("adam_beta2", 1.0, r"betas\[1\] = 1.0", "adam_beta2"),
])
def test_a_dense_run_s_refused_setting_is_a_config_refusal_naming_it(field, value, said, setting):
    class Cfg(_Cfg):
        pass

    setattr(Cfg, field, value)
    with pytest.raises(PorosConfigError, match=f"ExpertAdamW steps this run's bf16 trainable "
                       f"tensors, and it refuses its settings: {said}") as err:
        ea.build_optimizer(_peft_bias_model(torch.bfloat16), Cfg)
    assert err.value.info.remediation == f"Change {setting} as the words above say."
