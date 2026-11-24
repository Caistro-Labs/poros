"""Digests and counts from real tensors, and a real tiny run through the verifier."""

import struct

import pytest

torch = pytest.importorskip("torch")

from poros.metal.capture import TranscriptWriter, capture, digest_tensor  # noqa: E402
from poros.metal.verify import (  # noqa: E402
    PIN_KEYS,
    RECEIPT_SCHEMA,
    arm_from_transcript,
    digest_bytes,
    verify_receipt,
)

PIN = {key: "x" for key in PIN_KEYS} | {"ssd_sha256": "none"}


# -- -0.0 and 0.0 ------------------------------------------------------------------------------


def test_negative_zero_differs_under_the_digest_where_torch_equal_cannot_see_it():
    a = torch.tensor([0.0, 1.0], dtype=torch.float32)
    b = torch.tensor([-0.0, 1.0], dtype=torch.float32)
    assert torch.equal(a, b)
    assert (a == b).all()
    assert digest_tensor(a) != digest_tensor(b)
    assert capture(a)["b2"] != capture(b)["b2"]


def test_negative_zero_is_zero_to_the_nonzero_count_and_finite():
    s = capture(torch.tensor([-0.0, 0.0, 2.0]))
    assert s["nonzero"] == 1 and s["nonfinite_count"] == 0 and s["elements"] == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_negative_zero_differs_in_every_float_dtype(dtype):
    a = torch.zeros(4, dtype=dtype)
    b = a.clone()
    b[2] = -0.0
    assert torch.equal(a, b) and digest_tensor(a) != digest_tensor(b)


# -- NaN ---------------------------------------------------------------------------------------


def test_nan_same_bits_agree_under_the_digest_and_the_count_catches_them():
    a = torch.tensor([1.0, float("nan"), 3.0])
    b = a.clone()
    assert not torch.equal(a, b)  # torch.equal fails here for the opposite reason
    assert digest_tensor(a) == digest_tensor(b)
    assert capture(a)["nonfinite_count"] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bad, count", [
    ([float("nan")], 1),
    ([float("inf")], 1),
    ([float("-inf")], 1),
    ([float("nan"), float("inf"), 1.0, float("nan")], 3),
    ([1.0, 0.0], 0),
])
def test_nonfinite_count(dtype, bad, count):
    assert capture(torch.tensor(bad, dtype=dtype))["nonfinite_count"] == count


def test_nonfinite_counted_through_a_nested_surface():
    s = capture({"a": torch.tensor([float("nan")]), "b": [torch.tensor([float("inf"), 1.0])],
                 "lr": float("nan")})
    assert s["nonfinite_count"] == 3


def test_nan_differs_from_a_nan_with_other_bits():
    a = torch.frombuffer(bytearray(struct.pack("<I", 0x7FC00000)), dtype=torch.float32)
    b = torch.frombuffer(bytearray(struct.pack("<I", 0x7FC00001)), dtype=torch.float32)
    assert digest_tensor(a) != digest_tensor(b)
    assert capture(a)["nonfinite_count"] == capture(b)["nonfinite_count"] == 1


# -- the byte format ---------------------------------------------------------------------------


def test_digest_is_the_verifiers_over_little_endian_bytes():
    t = torch.tensor([[1.0, -0.0], [0.5, 2.0]], dtype=torch.float32)
    data = struct.pack("<4f", 1.0, -0.0, 0.5, 2.0)
    assert digest_tensor(t) == digest_bytes("float32", (2, 2), data)


def test_bf16_has_bytes_though_numpy_has_no_bf16():
    t = torch.tensor([1.0], dtype=torch.bfloat16)
    assert digest_tensor(t) == digest_bytes("bfloat16", (1,), b"\x80\x3f")


def test_dtype_and_shape_are_part_of_the_digest():
    assert digest_tensor(torch.zeros(4)) != digest_tensor(torch.zeros(2, 2))
    assert digest_tensor(torch.zeros(1, dtype=torch.float32)) != digest_tensor(
        torch.zeros(1, dtype=torch.int32)
    )


def test_layout_does_not_change_the_digest():
    t = torch.arange(6.0).reshape(2, 3)
    assert digest_tensor(t.t()) == digest_tensor(t.t().clone().contiguous())
    assert digest_tensor(t.t()) != digest_tensor(t.t().contiguous().flatten())  # shape, not bytes
    stride = torch.arange(6.0)[::2]
    assert not stride.is_contiguous()
    assert digest_tensor(stride) == digest_tensor(torch.tensor([0.0, 2.0, 4.0]))


@pytest.mark.parametrize("t, elements", [
    (torch.tensor(1.5), 1),
    (torch.empty(0), 0),
    (torch.empty(0, 3), 0),
    (torch.tensor([True, False]), 2),
    (torch.tensor([1, 0], dtype=torch.int64), 2),
    (torch.tensor([1, 0], dtype=torch.uint8), 2),
])
def test_odd_shapes_and_dtypes_digest(t, elements):
    s = capture(t)
    assert s["elements"] == elements and s["nonfinite_count"] == 0


def test_a_tensor_that_needs_grad_is_digested():
    t = torch.ones(2, requires_grad=True) * 2
    assert capture(t)["nonzero"] == 2


# -- nested surfaces ---------------------------------------------------------------------------


def _state():
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    model(torch.ones(1, 3)).sum().backward()
    opt.step()
    sched.step()
    return model, opt, sched


def test_real_adam_and_scheduler_state_digest_the_same_twice():
    torch.manual_seed(0)
    _, opt_a, sched_a = _state()
    torch.manual_seed(0)
    _, opt_b, sched_b = _state()
    assert capture(opt_a.state_dict()) == capture(opt_b.state_dict())
    assert capture(sched_a.state_dict()) == capture(sched_b.state_dict())


def test_adam_state_moves_the_digest():
    torch.manual_seed(0)
    model, opt, _ = _state()
    before = capture(opt.state_dict())["b2"]
    model(torch.ones(1, 3)).sum().backward()
    opt.step()
    assert capture(opt.state_dict())["b2"] != before


@pytest.mark.parametrize("a, b", [
    ({"x": torch.zeros(1)}, {"y": torch.zeros(1)}),
    ({"x": torch.zeros(1)}, {"x": torch.zeros(1), "y": torch.zeros(1)}),
    ({"x": torch.zeros(1)}, {"x": None}),
    ({"x": None}, {"x": ""}),
    ({"lr": 0.0}, {"lr": -0.0}),
    ({"lr": 1}, {"lr": 1.0}),
    ({"lr": 1}, {"lr": True}),
    ([torch.zeros(1), torch.ones(1)], [torch.ones(1), torch.zeros(1)]),
])
def test_a_nested_surface_loses_nothing(a, b):
    assert capture(a)["b2"] != capture(b)["b2"]


def test_entries_sum_their_counts():
    s = capture({"a": torch.zeros(2), "b": [torch.ones(3)], "lr": 0.5, "step": 3, "none": None})
    assert s["elements"] == 7 and s["nonzero"] == 5


def test_key_order_does_not_matter():
    assert capture({"a": 1, "b": 2}) == capture({"b": 2, "a": 1})


def test_unknown_leaf_raises_rather_than_vanish():
    with pytest.raises(TypeError, match="cannot digest set at '/x'"):
        capture({"x": {1}})


def test_capture_moves_no_rng():
    torch.manual_seed(7)
    before = torch.get_rng_state()
    capture({"a": torch.ones(3), "b": torch.tensor([float("nan")])})
    assert torch.equal(torch.get_rng_state(), before)


# -- a real tiny run through the verifier ------------------------------------------------------

STEPS, G, LAYERS = 3, 2, 2


def _run(path, arm, *, rng_off_at=None, negative_zero=False, nan_logits=False, stop_at=None):
    """Two dropout layers, AdamW and a scheduler: the shape of a gate arm, on one CPU thread."""
    torch.manual_seed(11)
    blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(LAYERS))
    dropout = torch.nn.Dropout(0.25)
    opt = torch.optim.AdamW(blocks.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (s + 1))
    data = torch.randn(STEPS * G, 4)
    last = STEPS * G if stop_at is None else stop_at
    w = TranscriptWriter(path, arm=arm, gate="repeat", model="tiny", seq=4, g=G, steps=STEPS,
                         lora_dropout=0.25, layers=LAYERS, expected_records=last, pin=PIN)
    for m in range(last):
        if m == rng_off_at:
            torch.manual_seed(1234)
        hidden, seen = data[m : m + 1], []
        for block in blocks:
            rng = torch.get_rng_state()
            inp = hidden
            hidden = dropout(block(hidden))
            hidden.retain_grad()
            seen.append((rng, inp, hidden))
        logits = hidden
        if nan_logits:
            logits = logits * float("nan")
        loss = logits.pow(2).mean() / G
        loss.backward()
        for n, (rng, inp, out) in enumerate(seen):
            w.layer(m, n, rng=rng, input=inp, output=out, cotangent=out.grad,
                    lora_grad=[p.grad for p in blocks[n].parameters()])
        w.microstep(m, loss=loss, logits=logits, grads={
            "lora_B": torch.zeros(2),
            **{f"{i}.{k}": v.grad for i, b in enumerate(blocks) for k, v in b.named_parameters()},
        })
        if (m + 1) % G == 0:
            opt.step()
            sched.step()
            opt.zero_grad()
    adapter = {k: v.detach().clone() for k, v in blocks.state_dict().items()}
    adapter["lora_B"] = torch.zeros(2)  # zero at init, so a true 0.0 to flip
    if negative_zero:
        adapter["lora_B"][0] = -0.0
    w.final(last, adapter=adapter, adam=opt.state_dict(), scheduler=sched.state_dict())
    w.close()
    return arm_from_transcript(path), adapter


def _receipt(arms):
    return {
        "schema": RECEIPT_SCHEMA, "gate": "repeat", "model": "tiny", "seq": 4, "g": G,
        "steps": STEPS, "lora_dropout": 0.25, "first_microstep": 0, "pin": dict(PIN),
        "expected_records": STEPS * G, "witness": {"checked": 12, "expected": 12, "mismatches": 0},
        "arms": arms, "pass": None,
    }


def test_two_real_runs_agree_and_pass(tmp_path):
    a, _ = _run(tmp_path / "one.jsonl", "one")
    b, _ = _run(tmp_path / "two.jsonl", "two")
    assert verify_receipt(_receipt([a, b]), tmp_path) == []
    assert a["digests"] == b["digests"]
    assert a["nonfinite"] == 0 and a["nonzero_grad_fraction"] > 0


def test_a_real_rng_fault_is_placed_at_its_first_layer(tmp_path):
    a, _ = _run(tmp_path / "one.jsonl", "one")
    b, _ = _run(tmp_path / "two.jsonl", "two", rng_off_at=4)
    failures = verify_receipt(_receipt([a, b]), tmp_path)
    assert any("first at microstep 4, layer 0, rng" in f for f in failures), failures


def test_mutation_negative_zero_in_the_final_adapter_is_caught_by_the_byte_comparator(tmp_path):
    """The mutation just before digesting. ``torch.equal`` misses it, on purpose."""
    a, clean = _run(tmp_path / "one.jsonl", "one")
    b, flipped = _run(tmp_path / "two.jsonl", "two", negative_zero=True)
    assert all(torch.equal(clean[key], flipped[key]) for key in clean)
    failures = verify_receipt(_receipt([a, b]), tmp_path)
    assert failures == ["arms one and two differ on adapter, first at final adapter"]


def test_real_nan_in_all_arms_digests_equal_and_fails_on_the_count(tmp_path):
    a, _ = _run(tmp_path / "one.jsonl", "one", nan_logits=True)
    b, _ = _run(tmp_path / "two.jsonl", "two", nan_logits=True)
    assert a["digests"] == b["digests"]  # same bits: every digest comparison passes
    failures = verify_receipt(_receipt([a, b]), tmp_path)
    assert not any("differ" in f for f in failures), failures
    assert any("first at microstep 0, logits" in f for f in failures), failures


def test_a_real_short_arm_fails_on_declared_counts(tmp_path):
    a, _ = _run(tmp_path / "one.jsonl", "one")
    b, _ = _run(tmp_path / "two.jsonl", "two", stop_at=5)
    failures = verify_receipt(_receipt([a, b]), tmp_path)
    assert any("5 records, declared 6" in f for f in failures), failures


def test_a_crash_mid_run_leaves_a_short_file_that_fails(tmp_path):
    a, _ = _run(tmp_path / "one.jsonl", "one")
    w = TranscriptWriter(tmp_path / "two.jsonl", arm="two", gate="repeat", model="tiny", seq=4,
                         g=G, steps=STEPS, lora_dropout=0.25, layers=LAYERS,
                         expected_records=STEPS * G, pin=PIN)
    w.microstep(0, loss=torch.ones(1), logits=torch.ones(1), grads=torch.ones(1))
    w.close()
    with pytest.raises(ValueError, match="bad transcript"):
        arm_from_transcript(tmp_path / "two.jsonl")
    b = dict(a, name="two", transcript_b2="0" * 64)
    assert any("transcript" in f for f in verify_receipt(_receipt([a, b]), tmp_path))
