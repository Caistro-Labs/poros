"""A non-finite loss or gradient norm stops a run at that update, before applying it.

The run (CPU here; the same code runs on CUDA, where no witness checks it) raises
``PorosStepCheckError`` with a remedy, writes an ``error`` event naming the value and the update,
leaves the adapter and the optimizer as the previous update left them, and keeps every
checkpoint written before. A finite run is the same bits with the check as without it.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from tests.test_train_loop_cpu import _loader, _make_trainer, _require_ml, _tiny_peft_qwen2


def _adapter(tr):
    return {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}


def _events(d):
    return [json.loads(line) for line in open(os.path.join(d, "events.jsonl"))]


def _run_to_failure(torch, d, poison, *, max_grad_norm=1.0, save_steps=1):
    """Train 5 updates; ``poison(tr)`` arranges update 3 (index 2) to go non-finite. Returns
    (trainer, the error, the adapter as update 2 left it)."""
    from poros.core.errors import PorosStepCheckError

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    tr = _make_trainer(torch, model, d)
    tr.config.save_steps = save_steps
    tr.config.max_grad_norm = max_grad_norm
    seen = {}
    step = tr._adapter_optimizer_step

    def watched():
        step()
        if tr.global_step == 1:  # after update 2 is applied
            seen["after2"] = _adapter(tr)

    tr._adapter_optimizer_step = watched
    poison(tr)
    with pytest.raises(PorosStepCheckError) as info:
        tr.train(_loader(torch, cfg.vocab_size), steps=5, seq_len=16)
    return tr, info.value, seen["after2"]


def _nan_loss_at_update_3(tr):
    real = tr._sequential_pass

    def p(batch, g, G):
        loss = real(batch, g, G)
        return loss * float("nan") if tr.global_step == 2 else loss

    tr._sequential_pass = p


def _inf_grad_at_update_3(tr):
    name, param = next((n, p) for n, p in tr.model.named_parameters() if "lora_B" in n)
    param.register_hook(lambda g: g * float("inf") if tr.global_step == 2 else g)


@pytest.mark.parametrize("poison, what", [(_nan_loss_at_update_3, "loss"),
                                          (_inf_grad_at_update_3, "grad_norm")])
def test_a_nonfinite_update_stops_the_run_before_it_is_applied(poison, what):
    torch = _require_ml()
    with tempfile.TemporaryDirectory() as d:
        tr, err, after2 = _run_to_failure(torch, d, poison)
        assert err.code == "E_STEP_CHECK" and "update 3" in str(err)
        assert "learning_rate" in err.info.remediation and "--resume-from latest" in str(err)
        assert what in err.witness and err.witness["step"] == 3
        now = _adapter(tr)  # update 3 was not applied: the adapter is update 2's
        assert all(torch.equal(now[n], after2[n]) for n in after2)
        assert all(torch.isfinite(t).all() for t in now.values())
        assert all(p.grad is None for p in tr.model.parameters() if p.requires_grad)
        errors = [e for e in _events(d) if e["event_type"] == "error"]
        assert len(errors) == 1 and errors[0]["data"]["code"] == "E_STEP_CHECK"
        said = errors[0]["data"]["nonfinite"]
        assert said["step"] == 3 and what in said
        assert errors[0]["step"] == 2  # the update's index, from 0, as a step event's
        steps = [e["step"] for e in _events(d) if e["event_type"] == "step"]
        assert steps == [0, 1]  # updates 1 and 2 logged, none after
        # checkpoint-1 (after update 2, index 1) was written before and is kept; none after
        assert os.path.isdir(os.path.join(d, "checkpoint-1"))
        assert not os.path.exists(os.path.join(d, "checkpoint-2"))
        assert "through update 2" in str(err)


def test_without_clipping_a_nonfinite_gradient_still_stops_the_run():
    torch = _require_ml()
    with tempfile.TemporaryDirectory() as d:
        tr, err, _ = _run_to_failure(torch, d, _inf_grad_at_update_3, max_grad_norm=0.0)
        assert err.witness["grad_norm"] in ("inf", "nan")


def test_float16_says_to_leave_torch_dtype_unset():
    torch = _require_ml()
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        tr = _make_trainer(torch, model, d)
        tr.config.torch_dtype = "float16"
        tr.global_step = 4
        from poros.core.errors import PorosStepCheckError

        with pytest.raises(PorosStepCheckError) as info:
            tr._stop_if_nonfinite(loss=math.inf)
        assert "leave torch_dtype unset" in info.value.info.remediation
        assert "Then start the run again" in info.value.info.remediation  # no checkpoint yet


def test_a_finite_run_is_the_same_bits_with_the_check():
    torch = _require_ml()

    def run(check):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        with tempfile.TemporaryDirectory() as d:
            tr = _make_trainer(torch, model, d)
            if not check:
                tr._stop_if_nonfinite = lambda **kw: None
            tr.train(_loader(torch, cfg.vocab_size), steps=4, seq_len=16)
            return list(tr.history.losses), list(tr.history.grad_norms), _adapter(tr)

    a, b = run(True), run(False)
    assert a[0] == b[0] and a[1] == b[1]
    assert all(torch.equal(a[2][n], b[2][n]) for n in a[2])
