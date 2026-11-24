"""``poros.train.expert_adamw``: AdamW for bf16 adapters, against an fp64 reference, and in the
trainer: what it draws, what it saves, what it refuses.

The quality evidence (2,000 updates against fp32 training) is
``scripts/expert_adam_precision.py`` and docs/validation/int/cpu/expert_adam_precision/.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import as_linux
from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")

from poros.core.checkpoints import (  # noqa: E402
    OLD_RULE,
    RULE,
    RULE_FILE,
    TORCH_RULE,
    checkpoint_rule,
)
from poros.core.errors import PorosConfigError  # noqa: E402
from poros.train import expert_adamw as ea  # noqa: E402
from poros.train.expert_adamw import (  # noqa: E402
    ExpertAdamW,
    build_optimizer,
    update_key,
)

stochastic_round_bf16 = ea._stochastic_round_bf16


def keyed_bits(key, start, n, device="cpu", draw=ea.DRAW_V):
    return ea._keyed_bits(key, start, n, draw, device)

BETAS, EPS, WD = (0.9, 0.999), 1e-8, 0.01


def _bf16_param(shape, seed=0, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter((torch.randn(shape, generator=g) * scale).bfloat16())


def _grads(shape, steps, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(shape, generator=g) * 1e-4).bfloat16() for _ in range(steps)]


def _reference(w0, grads, lr):
    """torch's AdamW formula in fp64, from the same bf16 grads: what ExpertAdamW approximates."""
    w, m, v = w0.double(), torch.zeros_like(w0, dtype=torch.float64), None
    v = torch.zeros_like(m)
    for t, g in enumerate(grads, 1):
        g = g.double()
        m = BETAS[0] * m + (1 - BETAS[0]) * g
        v = BETAS[1] * v + (1 - BETAS[1]) * g * g
        w = w * (1 - lr * WD)
        w = w - (lr / (1 - BETAS[0] ** t)) * m / ((v / (1 - BETAS[1] ** t)).sqrt() + EPS)
    return w, m, v


def _run(p, grads, lr, chunk=1 << 21, seed=0, name="w"):
    opt = ExpertAdamW([(name, p)], lr=lr, betas=BETAS, eps=EPS, weight_decay=WD, seed=seed,
                      chunk=chunk)
    for g in grads:
        p.grad = g
        opt.step()
    return opt


# ---- the update against fp64 ----------------------------------------------------------------


@pytest.mark.parametrize("lr", [2e-5, 2e-4])
def test_the_weight_follows_fp64_adamw_where_plain_bf16_stalls(lr):
    shape, steps = (32, 128), 200
    grads = _grads(shape, steps)
    p = _bf16_param(shape)
    w0 = p.detach().clone()
    opt = _run(p, grads, lr)
    ref, ref_m, ref_v = _reference(w0.float(), grads, lr)
    held = opt.weights()["w"].double()
    moved = (ref - w0.double()).norm()
    # hi + remainder holds the weight to about 16 bits; what is left is v's rounding noise (a
    # few percent per element, unbiased, entering the step through its square root): about 1
    # percent of the move, where plain bf16 loses a third to all of it (below)
    assert (held - ref).norm() / moved < 3e-2
    # the parameter itself is the weight rounded to nearest: within half a bf16 step of it
    hi = p.detach().double()
    assert bool(((held - hi).abs() <= hi.abs() * 2 ** -8).all())
    state = opt.state[p]
    # m rounded stochastically: its noise, unbiased, about half a percent of its norm
    assert (state["exp_avg"].double() - ref_m).norm() / ref_m.norm() < 1e-2
    v = state["exp_avg_sq"].double()
    assert abs(float(v.sum() / ref_v.sum()) - 1) < 2e-3  # unbiased: no ratchet
    # the control: torch's AdamW on the same bf16 parameter loses most of the move
    q = torch.nn.Parameter(w0.clone())
    plain = torch.optim.AdamW([q], lr=lr, betas=BETAS, eps=EPS, weight_decay=WD)
    for g in grads:
        q.grad = g
        plain.step()
    assert (q.detach().double() - ref).norm() / moved > 0.2


def test_fp32_parameters_get_torch_adamw_s_math():
    shape, lr = (32, 48), 1e-3
    grads = [g.float() for g in _grads(shape, 20)]
    p = torch.nn.Parameter(torch.randn(shape))
    q = torch.nn.Parameter(p.detach().clone())
    _run(p, grads, lr)
    ref = torch.optim.AdamW([q], lr=lr, betas=BETAS, eps=EPS, weight_decay=WD, foreach=False)
    for g in grads:
        q.grad = g
        ref.step()
    torch.testing.assert_close(p.detach(), q.detach(), rtol=0, atol=1e-7)


def test_the_result_does_not_depend_on_the_chunking():
    shape = (40, 77)  # 3080 elements: chunks of 1000 end mid-row
    grads = _grads(shape, 30)
    a, b = _bf16_param(shape), _bf16_param(shape)
    oa, ob = _run(a, grads, 2e-4, chunk=1 << 21), _run(b, grads, 2e-4, chunk=1000)
    assert torch.equal(a, b)
    for k in ("exp_avg", "exp_avg_sq", "remainder"):
        assert torch.equal(oa.state[a][k], ob.state[b][k]), k


def test_the_order_parameters_are_visited_in_does_not_matter():
    pa, pb = _bf16_param((16, 32), 1), _bf16_param((8, 64), 2)
    qa, qb = pa.detach().clone().requires_grad_(), pb.detach().clone().requires_grad_()
    ga, gb = _grads(pa.shape, 5, 3), _grads(pb.shape, 5, 4)
    one = ExpertAdamW([("a", pa), ("b", pb)], lr=1e-3, seed=7)
    two = ExpertAdamW([("b", qb), ("a", qa)], lr=1e-3, seed=7)
    for x, y in zip(ga, gb):
        pa.grad, pb.grad, qa.grad, qb.grad = x, y, x, y
        one.step()
        two.step()
    assert torch.equal(pa, qa) and torch.equal(pb, qb)
    assert torch.equal(one.state[pa]["exp_avg_sq"], two.state[qa]["exp_avg_sq"])


# ---- the keyed bits and the rounding ---------------------------------------------------------


def test_stochastic_rounding_is_unbiased_and_lands_on_a_neighbour():
    n = 1 << 20
    x = torch.full((n,), 1.0 + 2 ** -9)  # a quarter of a bf16 step above 1
    y = stochastic_round_bf16(x, keyed_bits(123, 0, n)).float()
    assert set(y.unique().tolist()) == {1.0, 1.0 + 2 ** -7}
    assert abs(float((y == 1.0 + 2 ** -7).float().mean()) - 0.25) < 3e-3
    x = torch.rand(n) * 1e-6 - 5e-7  # both signs, many exponents
    y = stochastic_round_bf16(x, keyed_bits(9, 0, n)).float()
    assert abs(float((y - x).mean())) < 1e-11
    lo = x.bfloat16().float()
    assert bool(((y - x).abs() < (lo.abs() * 2 ** -7 + 1e-45)).all())


def test_exact_and_special_values_pass_as_they_are():
    x = torch.tensor([0.0, -0.0, 1.0, -2.5, float("inf"), float("-inf"), float("nan")])
    y = stochastic_round_bf16(x, keyed_bits(5, 0, x.numel()))
    assert torch.equal(y[:6], x[:6].bfloat16())
    assert torch.signbit(y[1]) and torch.isnan(y[6])


def test_special_values_have_one_bit_pattern_whatever_the_bits_drawn():
    """A cast gives a NaN whose sign depends on the device (CUDA's differs from the CPU's), so
    the rounding writes these itself: zeros and infinities keep their bits, every NaN is 0x7fc0."""
    raw = torch.tensor([0x00000000, 0x80000000, 0x7F800000, 0xFF800000,  # +0, -0, +inf, -inf
                        0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF],  # NaNs, one signalling
                       dtype=torch.int64)
    x = (raw - ((raw >> 31) << 32)).to(torch.int32).view(torch.float32)
    want = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7FC0, 0x7FC0, 0x7FC0]
    for bits in (torch.zeros(8, dtype=torch.int64), torch.full((8,), 0xFFFF),
                 keyed_bits(5, 0, 8)):
        got = stochastic_round_bf16(x, bits).view(torch.int16).to(torch.int64) & 0xFFFF
        assert got.tolist() == want


def test_the_bits_are_a_pure_function_of_key_and_index():
    whole = keyed_bits(77, 0, 5000)
    assert torch.equal(keyed_bits(77, 1200, 800), whole[1200:2000])
    assert int(whole.min()) >= 0 and int(whole.max()) < 1 << 16
    assert not torch.equal(keyed_bits(78, 0, 5000), whole)
    assert abs(float(whole.float().mean()) / 65535 - 0.5) < 0.02
    assert update_key(1, 2, "x") != update_key(1, 3, "x") != update_key(1, 2, "y")


def test_the_device_draws_the_cpu_s_bits():
    """The hash runs where the parameter is; CPU, CUDA and MPS must agree bit for bit.
    (scripts/moe_precision/device_selftest.py checks it on a Mac.)"""
    devices = [d for d, ok in (("cuda", torch.cuda.is_available()),
                               ("mps", torch.backends.mps.is_available())) if ok]
    if not devices:
        pytest.skip("no CUDA or MPS device here")
    want = keyed_bits(2 ** 63 - 25, 1000, 1 << 16)
    x = torch.randn(1 << 16) * 1e-3
    sr = stochastic_round_bf16(x, want)
    for device in devices:
        assert torch.equal(keyed_bits(2 ** 63 - 25, 1000, 1 << 16, device).cpu(), want)
        got = stochastic_round_bf16(x.to(device), want.to(device)).cpu()
        assert torch.equal(got.view(torch.int16), sr.view(torch.int16))


def test_a_step_moves_no_torch_generator():
    p = _bf16_param((64, 64))
    opt = ExpertAdamW([("w", p)], lr=1e-3, seed=3)
    before = torch.get_rng_state()
    for g in _grads(p.shape, 3):
        p.grad = g
        opt.step()
    assert torch.equal(torch.get_rng_state(), before)


# ---- the state, saved and restored ----------------------------------------------------------


def test_a_restored_state_continues_bit_for_bit(tmp_path):
    shape = (24, 40)
    grads = _grads(shape, 10)
    whole = _bf16_param(shape)
    ow = _run(whole, grads, 1e-3, seed=4)
    part = _bf16_param(shape)
    op = _run(part, grads[:4], 1e-3, seed=4)
    torch.save(op.state_dict(), tmp_path / "opt.pt")
    torch.save(part.detach(), tmp_path / "w.pt")
    again = torch.nn.Parameter(torch.load(tmp_path / "w.pt"))
    oa = ExpertAdamW([("w", again)], lr=1e-3, seed=4)
    oa.load_state_dict(torch.load(tmp_path / "opt.pt", weights_only=True))
    for g in grads[4:]:
        again.grad = g
        oa.step()
    assert torch.equal(again, whole)
    for k in ("exp_avg", "exp_avg_sq", "remainder", "step"):
        assert torch.equal(oa.state[again][k], ow.state[whole][k]), k
    assert oa.state[again]["exp_avg_sq"].dtype == torch.bfloat16


def test_torch_s_bf16_adamw_state_is_refused():
    p = _bf16_param((4, 4))
    old = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.ones_like(p)
    old.step()
    with pytest.raises(PorosConfigError, match="before 1.0.0") as err:
        ExpertAdamW([("w", p)], lr=1e-3).load_state_dict(old.state_dict())
    assert "--from" in err.value.info.remediation


def test_the_trainer_s_choice_of_optimizer():
    class Cfg:
        learning_rate, adam_beta1, adam_beta2, adam_epsilon, weight_decay, seed = (
            2e-4, 0.9, 0.999, 1e-8, 0.01, 0)

    fp32 = [("a", torch.nn.Parameter(torch.zeros(2))), ("b", torch.nn.Parameter(torch.zeros(3)))]
    opt = build_optimizer(fp32, Cfg)
    assert type(opt) is torch.optim.AdamW
    assert [p for p in opt.param_groups[0]["params"]] == [p for _, p in fp32]
    bf16 = [("e", torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16)))]
    assert type(build_optimizer(bf16, Cfg)) is ExpertAdamW
    assert type(build_optimizer(fp32 + bf16, Cfg)) is ExpertAdamW  # no bf16 tensor left to torch
    fp16 = [("h", torch.nn.Parameter(torch.zeros(2, dtype=torch.float16)))]
    with pytest.raises(PorosConfigError, match="torch.float16"):
        build_optimizer(fp16 + bf16, Cfg)


# ---- the trainer: a tiny MoE model on the CPU ------------------------------------------------

require_tested_transformers()

from tests.moe.test_native_entry import _loader, _model, _trainer  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_moe_adamw")
    _model().save_pretrained(path)
    return path


@pytest.fixture(autouse=True)
def _a_cuda_hosts_planner(monkeypatch):
    import poros.core.card

    as_linux(monkeypatch, poros.core.card)


def test_a_moe_run_updates_its_expert_adapters_with_expert_adamw(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path)
    assert isinstance(trainer.optimizer, ExpertAdamW)
    assert all(p.dtype == torch.bfloat16 for p in trainer.trainable_parameters)
    trainer.train(_loader())
    state = trainer.optimizer.state[trainer.trainable_parameters[1]]
    assert int(state["step"]) == 2 and state["remainder"].abs().sum() > 0


def _expert_state(trainer):
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    return {f"{names[id(p)]}.{k}": v for p, s in trainer.optimizer.state.items()
            for k, v in s.items()}


def test_a_resume_in_the_middle_is_the_uninterrupted_run_remainder_and_all(checkpoint, tmp_path):
    """With dropout on (the default), saved at update 2 of 4: weights, remainder, both moments
    and the step equal, bit for bit."""
    knobs = dict(max_steps=4, gradient_accumulation_steps=2, learning_rate=1e-3,
                 warmup_steps=0)
    full = _trainer(checkpoint, tmp_path, save_steps=2, **knobs)
    full.train(_loader(n=3))
    said = json.loads((tmp_path / "checkpoint-2" / RULE_FILE).read_text())
    assert {k: said[k] for k in ("rule", "trainable_dtype", "torch_dtype")} == {
        "rule": RULE, "trainable_dtype": "bfloat16", "torch_dtype": "bfloat16"}
    assert said["implementation"] == {"cpu": {"device": "cpu", "path": "reference",
                                              "why": "the CPU runs the reference"}}
    resumed = _trainer(checkpoint, tmp_path, **knobs)
    resumed.train(_loader(n=3), resume_from=str(tmp_path / "checkpoint-2"))
    assert resumed.history.losses == full.history.losses[3:]
    after = dict(resumed.model.named_parameters())
    for name, param in full.model.named_parameters():
        if param.requires_grad:
            assert torch.equal(param, after[name]), name
    want, got = _expert_state(full), _expert_state(resumed)
    assert want.keys() == got.keys() and any(k.endswith(".remainder") for k in want)
    for key, value in want.items():
        assert torch.equal(got[key], value), key


def test_the_optimizer_draws_on_no_generator_a_run_uses(checkpoint, tmp_path, monkeypatch):
    """At a rate of 0 the weights never move, so two runs see the same model at every update:
    one with ExpertAdamW, one with torch's AdamW. Their dropout masks and data order (the losses,
    which they decide) and every generator's state at the end are equal: the keyed bits read
    and move no generator."""
    import poros.train.expert_adamw as ea

    def run(out, optimizer):
        monkeypatch.setattr(ea, "build_optimizer", optimizer)
        trainer = _trainer(checkpoint, out, max_steps=4, gradient_accumulation_steps=2,
                           lr_scheduler="constant", warmup_steps=0)
        for group in trainer.optimizer.param_groups:
            group["lr"] = 0.0
        assert trainer.config.lora_dropout > 0
        trainer.train(_loader(n=5))
        return trainer, torch.get_rng_state()

    real = ea.build_optimizer
    ours, rng_ours = run(tmp_path / "ours", real)
    assert isinstance(ours.optimizer, ExpertAdamW)
    theirs, rng_theirs = run(tmp_path / "theirs",
                             lambda named, cfg: torch.optim.AdamW([p for _, p in named], lr=0.0))
    assert ours.history.losses == theirs.history.losses
    assert torch.equal(rng_ours, rng_theirs)
    v = [s["exp_avg_sq"] for s in ours.optimizer.state.values()]
    assert any(bool(x.abs().sum() > 0) for x in v)  # the bits were drawn: v was rounded


def test_an_older_checkpoint_resume_is_refused_and_its_adapter_starts_a_new_run(
        checkpoint, tmp_path):
    """A checkpoint from before 1.0.0 (bf16 adapter weights, no RULE_FILE): the resume is
    refused before anything is restored, saying why and the way on; a new run from its
    adapter loads its weights exactly."""
    old = tmp_path / "old"
    trainer = _trainer(checkpoint, old, max_steps=2)
    trainer.train(_loader())
    ckpt = old / "checkpoint-1"
    trainer.save(str(ckpt))
    assert checkpoint_rule(ckpt) == RULE
    (ckpt / RULE_FILE).unlink()  # as a release candidate wrote it
    assert checkpoint_rule(ckpt) == OLD_RULE
    resumed = _trainer(checkpoint, old, max_steps=2)
    before = {n: p.detach().clone() for n, p in resumed.model.named_parameters()
              if p.requires_grad}
    with pytest.raises(PorosConfigError, match="before 1.0.0") as err:
        resumed.load_checkpoint(str(ckpt))
    assert "--from" in err.value.info.remediation
    for n, p in resumed.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, before[n]), n  # nothing restored
    fresh = _trainer(checkpoint, tmp_path / "new", max_steps=2, init_adapter=str(ckpt))
    assert fresh.start_from_adapter() == str(ckpt)  # as train() starts it
    saved = dict(trainer.model.named_parameters())
    for n, p in fresh.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, saved[n]), n


def test_a_checkpoint_without_a_rule_file_is_read_by_its_lora_tensors(tmp_path):
    from safetensors.torch import save_file

    save_file({"q.lora_A.weight": torch.zeros(2)}, tmp_path / "adapter_model.safetensors")
    assert checkpoint_rule(tmp_path) == TORCH_RULE
    save_file({"q.lora_A.weight": torch.zeros(2, dtype=torch.bfloat16)},
              tmp_path / "adapter_model.safetensors")
    assert checkpoint_rule(tmp_path) == OLD_RULE
    assert checkpoint_rule(tmp_path / "missing") is None


@pytest.mark.parametrize(("was", "now"), [("bfloat16", "float32"), ("float32", "bfloat16")])
def test_a_resume_at_another_torch_dtype_is_refused_naming_both(checkpoint, tmp_path, was, now):
    """bf16 expert adapters (ExpertAdamW) and fp32 ones (torch's AdamW) are other tensors with
    other state: either way round, the resume is refused before anything is restored, in words
    that name both dtypes, never as a checkpoint from before 1.0.0."""
    run = tmp_path / "run"
    trainer = _trainer(checkpoint, run, max_steps=2, torch_dtype=was)
    trainer.train(_loader())
    ckpt = run / "checkpoint-1"
    trainer.save(str(ckpt))
    resumed = _trainer(checkpoint, run, max_steps=2, torch_dtype=now)
    before = {n: p.detach().clone() for n, p in resumed.model.named_parameters()
              if p.requires_grad}
    with pytest.raises(PorosConfigError) as err:
        resumed.load_checkpoint(str(ckpt))
    said = str(err.value)
    assert f"torch_dtype {was}" in said and f"torch_dtype {now}" in said, said
    assert "before 1.0.0" not in said
    for n, p in resumed.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, before[n]), n


def _saved_with_implementation(checkpoint, run, recorded):
    trainer = _trainer(checkpoint, run, max_steps=2)
    trainer.train(_loader())
    ckpt = run / "checkpoint-1"
    trainer.save(str(ckpt))
    said = json.loads((ckpt / RULE_FILE).read_text())
    said["implementation"] = recorded
    (ckpt / RULE_FILE).write_text(json.dumps(said))
    return ckpt


def test_a_resume_on_a_mac_by_another_implementation_is_refused_naming_both(checkpoint, tmp_path,
                                                                          monkeypatch):
    ckpt = _saved_with_implementation(
        checkpoint, tmp_path / "run",
        {"cuda:0": {"device": "cuda", "path": "kernel", "why": "the kernel stores ..."}})
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    resumed = _trainer(checkpoint, tmp_path / "run", max_steps=2)
    resumed.optimizer.implementation = {  # as a Mac whose kernel failed its check would have
        "mps:0": {"device": "mps", "path": "reference on the CPU", "why": "test"}}
    before = {n: p.detach().clone() for n, p in resumed.model.named_parameters()
              if p.requires_grad}
    with pytest.raises(PorosConfigError) as err:
        resumed.load_checkpoint(str(ckpt))
    said = str(err.value)
    assert "stepped by ExpertAdamW's cuda:0: kernel, and here it steps with its mps:0: " \
           "reference on the CPU" in said, said
    assert "same implementation, or start again from its adapter (--from" in said, said
    assert ("This process chose (mps:0: its reference on the CPU because test), with "
            "PYTORCH_MPS_FAST_MATH unset") in said, said
    assert not resumed.optimizer.state
    for n, p in resumed.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, before[n]), n
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    with pytest.raises(PorosConfigError, match="PYTORCH_MPS_FAST_MATH set to '1'"):
        resumed.load_checkpoint(str(ckpt))
    same = _trainer(checkpoint, tmp_path / "run", max_steps=2)  # the same implementation: on
    same.optimizer.implementation = {"cuda:0": {"device": "cuda", "path": "kernel", "why": "t"}}
    same.load_checkpoint(str(ckpt))
    assert same.optimizer.state


def test_a_resume_on_cuda_by_another_implementation_goes_on_and_is_recorded(checkpoint, tmp_path,
                                                                            caplog):
    ckpt = _saved_with_implementation(
        checkpoint, tmp_path / "run",
        {"cuda:0": {"device": "cuda", "path": "kernel", "why": "the kernel stores ..."}})
    resumed = _trainer(checkpoint, tmp_path / "run", max_steps=2)  # the CPU's reference now
    with caplog.at_level("WARNING"):
        resumed.load_checkpoint(str(ckpt))
    assert "stepped by ExpertAdamW's cuda:0: kernel, and it now steps with its cpu: reference" \
        in caplog.text
    assert resumed.optimizer.state
    resumed.save(str(tmp_path / "again"))
    said = json.loads((tmp_path / "again" / RULE_FILE).read_text())
    assert said["implementation"]["cpu"]["resumed_from"] == "cuda:0: kernel"


def test_a_state_of_the_other_precision_is_refused_by_the_optimizer_itself():
    """Unreachable from the CLI after the resume check; refused in words all the same, where a
    bf16 parameter would otherwise meet a state with no remainder (a KeyError at its step)."""
    fp32 = torch.nn.Parameter(torch.zeros(4, 4))
    opt = ExpertAdamW([("w", fp32)], lr=1e-3)
    fp32.grad = torch.ones_like(fp32)
    opt.step()
    half = _bf16_param((4, 4))
    with pytest.raises(PorosConfigError, match="written for fp32 adapters"):
        ExpertAdamW([("w", half)], lr=1e-3).load_state_dict(opt.state_dict())
    other = ExpertAdamW([("w", half)], lr=1e-3)
    half.grad = torch.ones_like(half)
    other.step()
    with pytest.raises(PorosConfigError, match="written for bf16 adapters"):
        ExpertAdamW([("w", torch.nn.Parameter(torch.zeros(4, 4)))], lr=1e-3).load_state_dict(
            other.state_dict())


def test_the_disk_estimate_of_a_moe_checkpoint_is_its_real_state(checkpoint, tmp_path):
    """poros.preflight.disk sizes a MoE checkpoint as its weights and ExpertAdamW's state: the
    tiny model's trained adapters plus exp_avg, exp_avg_sq and remainder, in bytes."""
    from poros.preflight.disk import checkpoint_parts

    trainer = _trainer(checkpoint, tmp_path, max_steps=1)
    trainer.train(_loader())
    weights = sum(p.numel() * p.element_size() for p in trainer.trainable_parameters)
    state = sum(v.numel() * v.element_size() for s in trainer.optimizer.state.values()
                for k, v in s.items() if k != "step")
    config = json.loads((checkpoint / "config.json").read_text())
    one, adapter = checkpoint_parts(trainer.config, config)
    assert state == 3 * weights  # two moments and the remainder, each the weights' size
    # the estimate is the experts' exact bytes plus its fp32 allowance for other LoRA (the
    # MoE path trains none: it errs high, as the warning means to), 12 bytes a weight of it
    assert adapter >= weights and one >= weights + state
    assert one - (weights + state) == 3 * (adapter - weights)


def _round_v1(x, bits):
    """``stochastic_round_bf16`` as the optimizer's first rounding had it (int64 on the whole bit
    pattern): the int32 version must give the same bf16 for every input and every 16 bits."""
    x = x.contiguous()
    i = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    out = ((i + bits.reshape(x.shape).to(torch.int64)) & 0xFFFF0000) >> 16
    out = torch.where(torch.isfinite(x), out, i >> 16)
    out = torch.where(torch.isnan(x), torch.full_like(out, 0x7FC0), out)
    out = out - ((out >> 15) << 16)
    return out.to(torch.int16).view(torch.bfloat16)


def test_the_int32_rounding_is_the_int64_one_bit_for_bit():
    g = torch.Generator().manual_seed(3)
    n = 1 << 20
    pattern = torch.randint(-(2 ** 31), 2 ** 31, (n,), generator=g, dtype=torch.int64)
    x = pattern.to(torch.int32).view(torch.float32).clone()  # every exponent, both signs
    specials = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), -float("nan"),
                             3.3895e38, -3.3895e38, 1e-45, -1e-45, 1.17549435e-38])
    x[:specials.numel()] = specials
    x[specials.numel():specials.numel() + 4] = torch.tensor([0x7F7FFFFF, 0x7F800001, 0xFFC00001,
                                                            0x00FFFFFF], dtype=torch.int64).to(
        torch.int32).view(torch.float32)
    for bits in (torch.randint(0, 1 << 16, (n,), generator=g, dtype=torch.int32),
                 torch.zeros(n, dtype=torch.int32), torch.full((n,), 0xFFFF, dtype=torch.int32)):
        got, want = stochastic_round_bf16(x, bits), _round_v1(x, bits)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_the_bits_cross_a_block_as_they_would_alone():
    from poros.train.expert_adamw import BLOCK

    key = update_key(1, 2, "w")
    span = keyed_bits(key, BLOCK - 5, 10)
    assert torch.equal(span[:5], keyed_bits(key, BLOCK - 5, 5))
    assert torch.equal(span[5:], keyed_bits(key, BLOCK, 5))
    a, b = ea._block_coefficients(key, 1, ea.DRAW_V)
    assert 0 <= a < 2 ** 36 and 0 <= b < 2 ** 36
    assert [int(x) for x in keyed_bits(key, BLOCK + 7, 1)] == [((a * 7 + b) % 2 ** 36) >> 20]
    others = [keyed_bits(key, BLOCK + 7, 1, draw=d) for d in (ea.DRAW_M, ea.DRAW_REMAINDER)]
    assert len({int(t) for t in [keyed_bits(key, BLOCK + 7, 1), *others]}) == 3  # three draws


def test_each_element_s_bits_are_uniform_over_updates():
    """What stochastic rounding needs of one element: its 16 bits uniform, update after update
    (a fresh key each), so E[round(v)] = v. 4,000 updates of 8 elements: each element's mean and
    its share above half near 1/2."""
    draws = torch.stack([keyed_bits(update_key(0, u, "w"), 123456, 8) for u in range(4000)])
    share = draws.float() / 65536
    assert bool((share.mean(0) - 0.5).abs().max() < 0.02)
    assert bool(((draws >= 0x8000).float().mean(0) - 0.5).abs().max() < 0.03)


# ---- the rounding: every stored value an unbiased keyed rounding ---------------------------------


def _weights_near_every_exponent(n, seed):
    """fp32 weights over many exponents, both signs, and the cases a remainder can meet: just
    below and above powers of two (where the bf16 step halves), and exact bf16 values (r = 0)."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.rand(n, generator=g) + 0.5) * torch.exp2(torch.randint(-60, 60, (n,), generator=g)
                                                        .float())
    w[: n // 4] *= -1
    pw = torch.exp2(torch.randint(-40, 40, (n // 4,), generator=g).float())
    span = torch.randint(-300, 300, (n // 4,), generator=g)
    near = (pw.view(torch.int32) + span.to(torch.int32)).view(torch.float32)  # fp32 steps away
    w[n // 4: n // 2] = near
    w[n // 2: n // 2 + 64] = w[n // 2: n // 2 + 64].bfloat16().float()  # r = 0
    return w


def test_the_remainder_is_bounded_exact_and_unbiased_for_every_draw():
    """hi = the nearest bf16 to w, r = w - hi (exact in fp32), lo = r rounded stochastically.
    For every one of the 2**16 draws: |lo| is at most half of hi's bf16 step (a power of two,
    itself a bf16 value), hi + lo is exact in fp32; over the draws, lo's mean is r exactly."""
    w = _weights_near_every_exponent(256, 0)
    hi = w.bfloat16()
    r = w - hi.float()
    assert torch.equal((hi.double() + r.double()).float(), w)  # r is exact
    draws = torch.arange(1 << 16, dtype=torch.int32)
    total = torch.zeros(w.numel(), dtype=torch.float64)
    # half of hi's step below its magnitude, where |w| sits just under a power of two
    k = torch.floor(torch.log2(hi.float().abs().double()))
    bound = torch.exp2(k - 8)
    for i in range(w.numel()):
        lo = stochastic_round_bf16(r[i].expand(1 << 16).contiguous(), draws).double()
        assert bool((lo.abs() <= bound[i]).all()), (float(w[i]), float(r[i]))
        held = hi[i].double() + lo
        assert torch.equal(held.float().double(), held), float(w[i])  # exact in fp32
        total[i] = lo.sum()
    assert torch.equal(total / (1 << 16), r.double())  # the mean over the draws is r exactly
    assert bool((stochastic_round_bf16(torch.zeros(4), draws[:4]).float() == 0).all())


def test_every_stored_value_is_unbiased_over_updates():
    """One step from the same state at 600 updates' keys. Each stored m, v and remainder is one of
    the two bf16 values around its fp32 value x (the step's own, computed as the eager path does);
    over the keys it lands on the one above with probability (x - below) / (above - below), so
    its mean is x. Checked by the share of keys that rounded up, element by element (within 5
    standard deviations of a binomial share) and over all elements (within 3)."""
    n, keys = 2048, 600
    gen = torch.Generator().manual_seed(4)
    hi0 = (torch.randn(n, generator=gen) * 0.02).bfloat16()
    lo0 = (torch.randn(n, generator=gen) * 1e-5).bfloat16()
    m0 = (torch.randn(n, generator=gen) * 1e-5).bfloat16()
    v0 = (torch.rand(n, generator=gen) * 1e-9).bfloat16()
    grad = (torch.randn(n, generator=gen) * 1e-4).bfloat16()
    stored = {"m": [], "v": [], "r": []}
    hi = None
    for seed in range(keys):
        p = torch.nn.Parameter(hi0.clone())
        p.grad = grad.clone()
        opt = ExpertAdamW([("w", p)], lr=2e-4, seed=seed)
        opt.state[p].update({"step": torch.tensor(7, dtype=torch.int64),
                             "exp_avg": m0.clone(), "exp_avg_sq": v0.clone(),
                             "remainder": lo0.clone()})
        opt.step()
        stored["m"].append(opt.state[p]["exp_avg"].clone())
        stored["v"].append(opt.state[p]["exp_avg_sq"].clone())
        stored["r"].append(opt.state[p]["remainder"].clone())
        assert hi is None or torch.equal(p.detach(), hi)  # hi: the nearest bf16, every key
        hi = p.detach().clone()
    # the fp32 values, as the eager path computes them
    g32 = grad.float()
    m32 = m0.float().lerp_(g32, 1 - 0.9)
    v32 = v0.float().mul_(0.999).addcmul_(g32, g32, value=1 - 0.999)
    w = hi0.float().add_(lo0).mul_(1 - 2e-4 * 0.01)
    denom = v32.double().sqrt().float().div_((1 - 0.999 ** 8) ** 0.5).add_(1e-8)
    w.addcdiv_(m32, denom, value=-2e-4 / (1 - 0.9 ** 8))
    assert torch.equal(w.bfloat16(), hi)
    r32 = w - hi.float()
    for name, x in (("m", m32), ("v", v32), ("r", r32)):
        below = stochastic_round_bf16(x, torch.zeros(n, dtype=torch.int32)).double()
        above = stochastic_round_bf16(x, torch.full((n,), 0xFFFF, dtype=torch.int32)).double()
        got = torch.stack(stored[name]).double()
        assert bool(((got == below) | (got == above)).all()), name
        gap = above - below
        moving = gap != 0
        assert bool((got[:, ~moving] == below[~moving]).all()), name  # exact: never moves
        share = ((got - below) / torch.where(moving, gap, 1.0)).mean(0)[moving]
        want = ((x.double() - below) / torch.where(moving, gap, 1.0))[moving]
        sd = (want * (1 - want) / keys).sqrt()
        assert bool(((share - want).abs() <= 5 * sd + 1.5 / keys).all()), name
        total = float((share - want).sum() / (want * (1 - want) / keys).sum().sqrt())
        assert abs(total) < 3, (name, total)


def test_weight_decay_on_an_idle_expert_lands():
    """An expert no token reaches has a zero gradient: decoupled decay is its only change, 2e-7
    of the weight a step at lr 2e-5 and weight decay 0.01, 38 times under what a remainder rounded
    to nearest can hold (2**-17). Rounded stochastically, 2,000 steps shrink the weight as fp32's
    AdamW does (fp32's own rounding of 1 - 2e-7 included), to within 1 percent of the shrink."""
    n = 4096
    gen = torch.Generator().manual_seed(5)
    w0 = (torch.randn(n, generator=gen) * 0.02).bfloat16()
    p = torch.nn.Parameter(w0.clone())
    opt = ExpertAdamW([("w", p)], lr=2e-5, weight_decay=0.01, seed=3)
    for _ in range(2000):
        p.grad = torch.zeros_like(p)
        opt.step()
    q = torch.nn.Parameter(w0.float())  # fp32 AdamW from the same weights: the reference
    ref = torch.optim.AdamW([q], lr=2e-5, weight_decay=0.01)
    for _ in range(2000):
        q.grad = torch.zeros_like(q)
        ref.step()
    want = 1 - float(q.detach().double().norm() / w0.double().norm())  # about 3.8e-4
    got = 1 - float(opt.weights()["w"].double().norm() / w0.double().norm())
    assert abs(got / want - 1) < 0.01, (got, want)


@pytest.mark.parametrize("beta1", [0.95, 0.99])
def test_m_reaches_a_steady_gradient(beta1):
    """m rounded to nearest stops moving while |g - m| is under half a bf16 step over 1 - beta1
    (7.8 percent of |m| at beta1 0.95, 39 percent at 0.99). Rounded stochastically, a steady
    gradient's m reaches it, on average over the elements."""
    n = 4096
    g = (torch.rand(n, generator=torch.Generator().manual_seed(6)) * 1e-4 + 1e-4).bfloat16()
    p = torch.nn.Parameter(torch.zeros(n, dtype=torch.bfloat16))
    opt = ExpertAdamW([("w", p)], lr=0.0, betas=(beta1, 0.999), weight_decay=0.0, seed=1)
    steps = int(12 / (1 - beta1))
    for _ in range(steps):
        p.grad = g.clone()
        opt.step()
    ratio = opt.state[p]["exp_avg"].double() / g.double()
    assert abs(float(ratio.mean()) - (1 - beta1 ** steps)) < 2e-3
    # the control: the same moving average rounded to nearest stalls short of it
    m = torch.zeros(n, dtype=torch.bfloat16)
    for _ in range(steps):
        m = m.float().lerp_(g.float(), 1 - beta1).bfloat16()
    assert float((m.double() / g.double()).mean()) < 1 - beta1 ** steps - 0.02


# ---- hardening ---------------------------------------------------------------------------------


def test_the_step_counter_is_an_int64_on_the_cpu_whatever_the_defaults(tmp_path):
    """Under a bf16 default dtype a float counter would stop at 256 (and the bias correction and
    the keys with it). The counter is an int64 on the CPU, and stays one through a save and a
    load that maps the state elsewhere."""
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        p = _bf16_param((64,))
        opt = ExpertAdamW([("w", p)], lr=1e-4)
        for _ in range(300):
            p.grad = torch.full_like(p, 1e-4)
            opt.step()
    finally:
        torch.set_default_dtype(before)
    step = opt.state[p]["step"]
    assert step.dtype == torch.int64 and step.device.type == "cpu" and int(step) == 300
    sd = opt.state_dict()
    again = ExpertAdamW([("w", p)], lr=1e-4)
    again.load_state_dict(sd)
    assert again.state[p]["step"].dtype == torch.int64 and int(again.state[p]["step"]) == 300
    sd["state"][0]["step"] = sd["state"][0]["step"].double()  # not what this optimizer saves
    with pytest.raises(PorosConfigError, match="one int64 tensor"):
        ExpertAdamW([("w", p)], lr=1e-4).load_state_dict(sd)


def test_a_state_for_other_parameters_or_another_seed_is_refused():
    a, b = _bf16_param((8,), 1), _bf16_param((8,), 2)
    opt = ExpertAdamW([("x", a), ("y", b)], lr=1e-3, seed=5)
    a.grad, b.grad = torch.ones_like(a), torch.ones_like(b)
    opt.step()
    sd = opt.state_dict()
    with pytest.raises(PorosConfigError, match="number 0: x in the state, y here"):
        ExpertAdamW([("y", b), ("x", a)], lr=1e-3, seed=5).load_state_dict(sd)
    with pytest.raises(PorosConfigError, match="for 2 parameters and this optimizer has 1"):
        ExpertAdamW([("x", a)], lr=1e-3, seed=5).load_state_dict(sd)
    with pytest.raises(PorosConfigError, match="seed 5, and this optimizer has 6"):
        ExpertAdamW([("x", a), ("y", b)], lr=1e-3, seed=6).load_state_dict(sd)
    sd["param_groups"][0]["block"] = 1 << 20
    with pytest.raises(PorosConfigError, match="block"):
        ExpertAdamW([("x", a), ("y", b)], lr=1e-3, seed=5).load_state_dict(sd)


@pytest.mark.parametrize(("make", "said"), [
    (lambda p: dict(named_params=[("w", p), ("w", _bf16_param((4,)))]), "given to more than one"),
    (lambda p: dict(named_params=[("w", torch.nn.Parameter(torch.zeros(4, 6,
                                   dtype=torch.bfloat16).t()))]), "not contiguous"),
    (lambda p: dict(named_params=[("w", p)], lr=-1.0), "lr = -1.0"),
    (lambda p: dict(named_params=[("w", p)], lr=float("nan")), "lr = nan"),
    (lambda p: dict(named_params=[("w", p)], eps=-1e-8), "eps"),
    (lambda p: dict(named_params=[("w", p)], betas=(1.0, 0.999)), r"betas\[0\]"),
    (lambda p: dict(named_params=[("w", p)], betas=(0.9, -0.1)), r"betas\[1\]"),
    (lambda p: dict(named_params=[("w", p)], weight_decay=-0.01), "weight_decay"),
    (lambda p: dict(named_params=[("w", p)], seed=2 ** 63), "signed 64-bit"),
    (lambda p: dict(named_params=[("w", p)], seed=-(2 ** 63) - 1), "signed 64-bit"),
])
def test_construction_refuses_in_words(make, said):
    kw = {"lr": 1e-3, **make(_bf16_param((4,)))}
    with pytest.raises((ValueError, TypeError), match=said):
        ExpertAdamW(kw.pop("named_params"), **kw)


def test_the_extremes_of_the_accepted_seeds_key_and_a_huge_tensor_is_refused_first(monkeypatch):
    for seed in (-(2 ** 63), 2 ** 63 - 1):
        p = _bf16_param((4,))
        p.grad = torch.ones_like(p)
        ExpertAdamW([("w", p)], lr=1e-3, seed=seed).step()
    huge = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
    monkeypatch.setattr(type(huge), "numel", lambda self: 2 ** 32)
    with pytest.raises(ValueError, match="2\\*\\*32 - 1"):
        ExpertAdamW([("w", huge)], lr=1e-3)


def test_a_group_added_later_is_refused():
    opt = ExpertAdamW([("w", _bf16_param((4,)))], lr=1e-3)
    with pytest.raises(ValueError, match="at construction only"):
        opt.add_param_group({"params": [_bf16_param((4,))]})


def test_fp32_parameters_update_their_state_in_place_and_leave_the_grad():
    p = torch.nn.Parameter(torch.randn(64))
    opt = ExpertAdamW([("w", p)], lr=1e-3)
    p.grad = torch.randn(64)
    opt.step()
    m, v = opt.state[p]["exp_avg"].data_ptr(), opt.state[p]["exp_avg_sq"].data_ptr()
    grad = torch.randn(64)
    p.grad = grad.clone()
    opt.step()
    assert opt.state[p]["exp_avg"].data_ptr() == m and opt.state[p]["exp_avg_sq"].data_ptr() == v
    assert torch.equal(p.grad, grad) and "remainder" not in opt.state[p]


def test_an_overflowed_weight_keeps_no_remainder_and_subnormals_are_zero():
    """The rule's edges, through the optimizer: a weight whose nearest bf16 is inf keeps no
    remainder (inf - inf at the next step); a subnormal grad, m, v or remainder is read as 0; a
    LoRA B at exactly zero still takes its first update."""
    big = 3.3895313892515355e38
    p = torch.nn.Parameter(torch.tensor([big, 0.0, 1.0, 1.0], dtype=torch.bfloat16))
    opt = ExpertAdamW([("w", p)], lr=1e-3, weight_decay=0.0)
    p.grad = torch.tensor([0.0, 1e-3, 1e-40, 1e-3], dtype=torch.bfloat16)
    opt.step()
    st = opt.state[p]
    st["remainder"][0] = 2.0 ** 119  # big + half its step: rounds to inf
    st["remainder"][3] = 1e-40  # subnormal: read as 0
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.isinf(p[0]) and float(st["remainder"][0]) == 0.0
    assert float(p[1]) != 0.0  # the zero weight moved on its first update
    assert float(st["exp_avg"][2]) == 0.0 and float(st["exp_avg_sq"][2]) == 0.0


def _a_gpu_without_its_kernel(monkeypatch):
    """This CPU's parameters as a GPU's whose kernel is not used: the reference steps them."""
    monkeypatch.setattr(ea, "_implementation", lambda device: {
        "device": "cuda", "path": "reference", "why": "Triton is not installed"})


def test_a_gpu_on_the_reference_is_said_with_its_cost_measured_here(monkeypatch):
    _a_gpu_without_its_kernel(monkeypatch)
    p = _bf16_param((3, 1 << 19))  # 1,572,864 elements
    opt = ExpertAdamW([("w", p)], lr=1e-3)
    impl = opt.implementation["cpu"]
    assert impl["elements"] == 3 << 19 and impl["seconds_per_update"] > 0
    assert opt.notice.startswith("ExpertAdamW steps the 1,572,864 bf16 trainable elements on cpu "
                                 "with its reference, not its fused kernel: Triton is not "
                                 "installed. Measured on this machine now: about ")
    assert "s per update for the optimizer's step" in opt.notice
    assert opt.notice.endswith("(the same bits as the kernel would store; only the time "
                               "differs)")
    assert "would not have" not in opt.notice
    ok = _bf16_param((8,))
    monkeypatch.undo()
    assert ExpertAdamW([("w", ok)], lr=1e-3).notice is None  # the CPU's own reference: no word


def test_a_kernel_whose_bits_differ_is_said_as_such_not_as_the_same_bits(monkeypatch):
    """The real ``_implementation`` for a device whose probe found other bits (a meta device
    standing for the GPU, the probe's verdict given), its parameters stepped here."""
    monkeypatch.setattr(ea, "_PROBED", {})
    monkeypatch.setattr(ea, "_kernel_unavailable", lambda kind: None)
    monkeypatch.setattr(ea, "_probe_differences", lambda dev: "setting 1, 4096 elements: ...")
    real = ea._implementation
    monkeypatch.setattr(ea, "_implementation", lambda device: real("meta"))
    opt = ExpertAdamW([("w", _bf16_param((8,)))], lr=1e-3)
    impl = opt.implementation["cpu"]
    assert impl["differs"] and impl["why"].startswith(ea.KERNEL_DIFFERS)
    assert opt.notice.endswith("(the run stores the rule's bits, which the kernel would not "
                               "have)"), opt.notice
    assert "only the time differs" not in opt.notice
    assert real("cpu").get("differs") is None


def test_the_run_records_and_says_the_reference_and_its_cost(checkpoint, tmp_path, monkeypatch,
                                                             caplog):
    _a_gpu_without_its_kernel(monkeypatch)
    with caplog.at_level("WARNING"):
        trainer = _trainer(checkpoint, tmp_path)
    notice = trainer.optimizer.notice
    assert notice and notice in caplog.text
    assert notice in trainer.config.compute_reason
    ran = trainer.config.ran_compute["expert_optimizer"]["cpu"]
    assert ran["path"] == "reference" and ran["seconds_per_update"] > 0
    trainer.train(_loader())
    setup = next(json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()
                 if json.loads(x)["event_type"] == "setup")
    assert notice in setup["data"]["compute_reason"]
    assert setup["data"]["expert_optimizer"]["cpu"]["path"] == "reference"


def test_an_eps_under_the_least_is_refused_and_at_it_the_rule_is_fp32_adamw_s():
    """min_adam_epsilon: 2**-39 / sqrt(1 - beta2). Under it, refused in words at construction;
    at it, a gradient whose square the rule reads as zero (g = 1e-19: v = 1e-41, a subnormal fp32
    AdamW keeps) takes fp32 AdamW's step to within a rounding."""
    from poros.core.checkpoints import min_adam_epsilon

    least = min_adam_epsilon(0.999)
    assert abs(least - 2.0 ** -39 / (1 - 0.999) ** 0.5) < 1e-25 and 5.7e-11 < least < 5.8e-11
    with pytest.raises(ValueError, match="under 5.75e-11"):
        ExpertAdamW([("w", _bf16_param((4,)))], lr=1e-4, eps=1e-30)
    with pytest.raises(ValueError, match="eps"):
        ExpertAdamW([("w", _bf16_param((4,)))], lr=1e-4, eps=0.0)
    w0 = torch.tensor([1.0, -1.0, 0.5, 2.0 ** -20], dtype=torch.bfloat16)
    g = torch.full((4,), 1e-19)
    p = torch.nn.Parameter(w0.clone())
    opt = ExpertAdamW([("w", p)], lr=1e-4, eps=least * 1.0001, weight_decay=0.0)
    p.grad = g.bfloat16()
    opt.step()
    q = torch.nn.Parameter(w0.float())
    ref = torch.optim.AdamW([q], lr=1e-4, eps=least * 1.0001, weight_decay=0.0)
    q.grad = g.bfloat16().float()
    ref.step()
    held = opt.weights()["w"].double()
    assert bool(((held - q.detach().double()).abs()
                 <= q.detach().double().abs() * 2.0 ** -15 + 1e-30).all()), (held, q)
