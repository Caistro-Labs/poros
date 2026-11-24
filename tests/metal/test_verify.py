"""The verifier against tampered transcripts and receipts. No torch here: it must not need it."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from poros.metal.verify import (
    PIN_KEYS,
    RECEIPT_SCHEMA,
    TRANSCRIPT_SCHEMA,
    arm_from_transcript,
    check_stored,
    digest_bytes,
    digest_chain,
    main,
    pin_differences,
    seal,
    short_of,
    track_relative,
    verify_receipt,
    write_verdict,
)

ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / "poros" / "metal" / "verify.py"
PIN = {key: "x" for key in PIN_KEYS} | {"ssd_sha256": hashlib.sha256(b"s").hexdigest()}
ELEMENTS = 8


def surface(seed, nonfinite=0, nonzero=ELEMENTS):
    return {
        "b2": digest_bytes("float32", (2, 4), seed.encode()),
        "nonfinite_count": nonfinite,
        "elements": ELEMENTS,
        "nonzero": nonzero,
    }


def transcript_lines(arm, gate="direct", model="m/7B", steps=3, g=2, layers=2, records=None,
                     first=0, pin=PIN, final_microsteps=None):
    """One arm, every surface a function of its position only, so arms agree until edited."""
    records = steps * g if records is None else records
    lines = [{
        "schema": TRANSCRIPT_SCHEMA, "arm": arm, "gate": gate, "model": model, "seq": 128, "g": g,
        "steps": steps, "first_microstep": first, "lora_dropout": 0.05, "layers": layers,
        "expected_records": records, "pin": pin,
    }]
    for m in range(first, first + records):
        for n in range(layers):
            lines.append({
                "kind": "layer", "microstep": m, "layer": n,
                **{f: surface(f"{f}{m}{n}")
                   for f in ("rng", "input", "output", "cotangent", "lora_grad")},
            })
        lines.append({
            "kind": "microstep", "microstep": m, "loss": surface(f"loss{m}"),
            "logits": surface(f"logits{m}"), "grads": surface(f"grads{m}"),
        })
    done = first + steps * g if final_microsteps is None else final_microsteps
    lines.append({
        "kind": "final", "microsteps": done, "adapter": surface("adapter"),
        "adam": surface("adam"), "scheduler": surface("scheduler"),
    })
    return lines


def dump(lines):
    return "".join(json.dumps(line) + "\n" for line in lines)


def build(tmp_path, arms=("mono", "ram-loop"), gate="direct", model="m/7B", steps=3, g=2,
          layers=None, first=0, lines=None, extra=None):
    """A receipt and its transcripts, all honest unless ``lines`` edits an arm before hashing."""
    tdir = tmp_path / "t"
    tdir.mkdir(exist_ok=True)
    layers = layers or {}
    entries = []
    for arm in arms:
        recs = transcript_lines(arm, gate=gate, model=model, steps=steps, g=g, first=first,
                                layers=layers.get(arm, 2))
        if lines and arm in lines:
            lines[arm](recs)
        path = tdir / f"{arm}.jsonl"
        path.write_text(dump(recs))
        entries.append(arm_from_transcript(path))
    receipt = {
        "schema": RECEIPT_SCHEMA, "gate": gate, "model": model, "seq": 128, "g": g, "steps": steps,
        "lora_dropout": 0.05, "first_microstep": first, "pin": dict(PIN),
        "expected_records": steps * g, "witness": {"checked": 12, "expected": 12, "mismatches": 0},
        "arms": entries, "s_per_step": None, "peak_unified_gb": None, "gpu_visible_gb": None,
        "swap_delta_gb": None, "pass": None,
    }
    receipt.update(extra or {})
    return receipt, tdir


def fails(receipt, tdir, needle):
    failures = verify_receipt(receipt, tdir)
    assert any(needle in f for f in failures), failures


def test_an_honest_receipt_passes(tmp_path):
    receipt, tdir = build(tmp_path)
    assert verify_receipt(receipt, tdir) == []


# -- digest ------------------------------------------------------------------------------------


def test_digest_covers_dtype_shape_and_every_byte():
    data = bytes(range(24))
    base = digest_bytes("float32", (2, 3), data)
    assert digest_bytes("float32", (2, 3), data) == base
    assert digest_bytes("int32", (2, 3), data) != base
    assert digest_bytes("float32", (3, 2), data) != base
    assert digest_bytes("float32", (6,), data) != base
    assert digest_bytes("float32", (2, 3), data[:-1] + b"\x18") != base
    assert digest_bytes("float32", (1, 23), data) != base  # no shape run-together
    assert digest_bytes("float32", (12, 3), data) != digest_bytes("float32", (1, 23), data)


def test_chain_is_ordered_and_keeps_part_boundaries():
    assert digest_chain(["a", "b"]) != digest_chain(["b", "a"])
    assert digest_chain(["ab"]) != digest_chain(["a", "b"])


# -- tampered transcripts ----------------------------------------------------------------------


def _flip_digest(lines):
    rec = lines[5]
    field = next(f for f in rec if isinstance(rec[f], dict))
    rec[field]["b2"] = ("0" if rec[field]["b2"][0] != "0" else "1") + rec[field]["b2"][1:]


def _set(index, field, key, value):
    def edit(lines):
        lines[index][field][key] = value
    return edit


TEXT_TAMPERS = {
    "digest flipped": lambda t: _edit_text(t, _flip_digest),
    "microstep record deleted": lambda t: _edit_text(t, lambda ls: ls.pop(3)),
    "layer record deleted": lambda t: _edit_text(t, lambda ls: ls.pop(1)),
    "line duplicated": lambda t: _edit_text(t, lambda ls: ls.insert(2, ls[1])),
    "lines swapped": lambda t: _edit_text(t, lambda ls: ls.insert(2, ls.pop(1))),
    "record appended": lambda t: _edit_text(t, lambda ls: ls.append(ls[-1])),
    "final removed": lambda t: _edit_text(t, lambda ls: ls.pop()),
    "last microstep and final removed": lambda t: _edit_text(t, lambda ls: (ls.pop(), ls.pop())),
    "non-finite count raised": lambda t: _edit_text(t, _set(3, "logits", "nonfinite_count", 3)),
    "nonzero above elements": lambda t: _edit_text(t, _set(3, "grads", "nonzero", ELEMENTS + 1)),
    "count is None": lambda t: _edit_text(t, _set(3, "grads", "nonzero", None)),
    "digest is None": lambda t: _edit_text(t, _set(3, "loss", "b2", None)),
    "surface missing": lambda t: _edit_text(t, lambda ls: ls[3].pop("logits")),
    "pin changed": lambda t: _edit_text(t, lambda ls: ls[0]["pin"].update(macos_build="other")),
    "declared count lowered": lambda t: _edit_text(t, lambda ls: ls[0].update(expected_records=5)),
    "final count changed": lambda t: _edit_text(t, lambda ls: ls[-1].update(microsteps=5)),
    "torn last line": lambda t: t[: t.rindex("\n") - 7],
    "no last newline": lambda t: t[:-1],
    "blank line": lambda t: t.replace("\n", "\n\n", 1),
    "not json": lambda t: t.replace('"kind"', "kind", 1),
    "empty": lambda t: "",
    "NaN literal": lambda t: t.replace('"nonfinite_count": 0', '"nonfinite_count": NaN', 1),
    "duplicate key": lambda t: t.replace(
        '"nonfinite_count": 0', '"nonfinite_count": 3, "nonfinite_count": 0', 1),
}


def _edit_text(text, edit):
    lines = [json.loads(line) for line in text.splitlines()]
    edit(lines)
    return dump(lines)


@pytest.mark.parametrize("name", sorted(TEXT_TAMPERS))
def test_tampered_transcript_fails_with_the_receipt_as_written(tmp_path, name):
    receipt, tdir = build(tmp_path)
    assert verify_receipt(receipt, tdir) == []
    path = tdir / "ram-loop.jsonl"
    path.write_text(TEXT_TAMPERS[name](path.read_text()))
    assert verify_receipt(receipt, tdir) != []


@pytest.mark.parametrize("name", sorted(TEXT_TAMPERS))
def test_tampered_transcript_fails_with_the_receipt_rederived(tmp_path, name):
    """Past the file hash: the receipt is recomputed from the edited file, and it still fails."""
    receipt, tdir = build(tmp_path)
    path = tdir / "ram-loop.jsonl"
    path.write_text(TEXT_TAMPERS[name](path.read_text()))
    try:
        entry = arm_from_transcript(path)
    except ValueError:
        return  # refused outright
    receipt["arms"][1] = entry
    assert verify_receipt(receipt, tdir) != []


@pytest.mark.parametrize("field, value", [
    ("records", 5), ("nonfinite", 1), ("microsteps", 5), ("nonzero_grad_fraction", None),
    ("nonzero_grad_fraction", 0.0), ("transcript_b2", "0" * 64),
])
def test_edited_arm_fields_fail(tmp_path, field, value):
    receipt, tdir = build(tmp_path)
    receipt["arms"][1][field] = value
    assert verify_receipt(receipt, tdir) != []


def test_torn_transcript_says_so(tmp_path):
    receipt, tdir = build(tmp_path)
    path = tdir / "mono.jsonl"
    path.write_text(path.read_text()[:-1])
    fails(receipt, tdir, "torn: the last line has no newline")


@pytest.mark.parametrize("edit, needle", [
    (lambda ls: ls[3]["grads"].update(nonzero=ELEMENTS + 1), "malformed surface"),
    (lambda ls: ls[3]["logits"].update(nonfinite_count=ELEMENTS + 1), "malformed surface"),
    (lambda ls: ls[1].update(layer=1), "expected microstep 0 layer 0"),
    (lambda ls: ls[1].update(microstep=1), "expected microstep 0 layer 0"),
    (lambda ls: ls[3].update(microstep=1), "expected microstep 0"),
    (lambda ls: ls[3].update(kind="layer"), "expected a microstep record"),
    (lambda ls: ls[-1].update(kind="microstep"), "expected a final record"),
    (lambda ls: ls[-1].pop("microsteps"), "no microsteps count"),
])
def test_a_transcript_wrong_the_same_way_in_every_arm_is_refused_on_its_own(tmp_path, edit, needle):
    """No other arm to disagree with, and a matching hash: only the strict reader sees these."""
    lines = transcript_lines("mono")
    edit(lines)
    path = tmp_path / "mono.jsonl"
    path.write_text(dump(lines))
    with pytest.raises(ValueError, match=needle):
        arm_from_transcript(path)


@pytest.mark.parametrize("record, field", [
    (3, "loss"), (3, "logits"), (3, "grads"), (-1, "adapter"), (-1, "adam"), (-1, "scheduler"),
])
def test_every_surface_is_required(tmp_path, record, field):
    def drop(lines):
        lines[record].pop(field)

    with pytest.raises(ValueError, match="malformed surface"):
        build(tmp_path, lines={"mono": drop})


# -- the mutations this verifier owns ------------------------------------------------------------


def test_mutation_arm_stopped_short_is_caught_by_declared_counts(tmp_path):
    """One microstep fewer, and its own transcript is honest about that."""
    receipt, tdir = build(tmp_path)
    short = transcript_lines("ram-loop", records=5, final_microsteps=5)
    (tdir / "ram-loop.jsonl").write_text(dump(short))
    receipt["arms"][1] = arm_from_transcript(tdir / "ram-loop.jsonl")
    failures = verify_receipt(receipt, tdir)
    assert any("5 records, declared 6" in f for f in failures), failures
    assert any("ran to microstep 5, declared 6" in f for f in failures), failures


def test_every_arm_short_alike_still_fails(tmp_path):
    def cut(lines):
        keep = lines[:-4] + [lines[-1]]
        keep[0]["expected_records"] = 5
        keep[-1]["microsteps"] = 5
        lines[:] = keep

    receipt, tdir = build(tmp_path, lines={"mono": cut, "ram-loop": cut})
    failures = verify_receipt(receipt, tdir)
    assert any("mono: 5 records" in f for f in failures), failures
    assert any("ram-loop: 5 records" in f for f in failures), failures
    assert not any("differ" in f for f in failures)  # the digests agree: only the count sees it


def test_declared_count_must_follow_from_steps_and_g(tmp_path):
    receipt, tdir = build(tmp_path, extra={"expected_records": 5})
    fails(receipt, tdir, "not steps x g")


def test_mutation_nan_in_all_arms_is_caught_by_the_count_and_missed_by_digests(tmp_path):
    def nan(lines):
        lines[3]["logits"]["nonfinite_count"] = 2

    receipt, tdir = build(tmp_path, lines={"mono": nan, "ram-loop": nan})
    assert receipt["arms"][0]["digests"] == receipt["arms"][1]["digests"]
    failures = verify_receipt(receipt, tdir)
    assert not any("differ" in f for f in failures), failures
    assert any("2 non-finite elements, first at microstep 0, logits" in f for f in failures)


@pytest.mark.parametrize("record, field", [
    (1, "rng"), (1, "input"), (1, "output"), (1, "cotangent"), (1, "lora_grad"),
    (3, "loss"), (3, "logits"), (3, "grads"), (-1, "adapter"), (-1, "adam"), (-1, "scheduler"),
])
def test_nonfinite_is_refused_on_every_surface(tmp_path, record, field):
    def nan(lines):
        lines[record][field]["nonfinite_count"] = 1

    receipt, tdir = build(tmp_path, lines={"mono": nan, "ram-loop": nan})
    fails(receipt, tdir, "1 non-finite elements")


def test_mutation_secret_ram_arm_is_caught_by_disk_counters_and_missed_by_digests(tmp_path):
    honest = {"read_bytes": 96, "predicted_read_bytes": 100, "preflight_resident_fraction": 0.01}
    receipt, tdir = build(tmp_path, arms=("ram-loop", "ssd"), gate="streaming")
    receipt["arms"][1]["disk"] = honest
    assert verify_receipt(receipt, tdir) == []

    receipt["arms"][1]["disk"] = honest | {"read_bytes": 94}
    assert receipt["arms"][0]["digests"] == receipt["arms"][1]["digests"]
    fails(receipt, tdir, "under 95% of predicted")
    receipt["arms"][1]["disk"] = honest | {"read_bytes": 95}
    assert verify_receipt(receipt, tdir) == []
    receipt["arms"][1]["disk"] = honest | {"preflight_resident_fraction": 0.05}
    fails(receipt, tdir, "required under 5%")
    receipt["arms"][1].pop("disk")
    fails(receipt, tdir, "disk counters missing")


def test_mutation_rng_off_names_the_first_layer_that_differs(tmp_path):
    """Masks stop matching after the first update: the first G microsteps agree."""
    def rng_off(lines):
        for rec in lines[1:-1]:
            if rec["kind"] == "layer" and rec["microstep"] >= 2:
                rec["output"] = surface(f"wrong{rec['microstep']}{rec['layer']}")

    receipt, tdir = build(tmp_path, lines={"ram-loop": rng_off})
    fails(receipt, tdir, "differ on records, first at microstep 2, layer 0, output")


def test_first_difference_follows_execution_order(tmp_path):
    def later_layer_backward(lines):
        # backward runs top down, so layer 1's cotangent comes before layer 0's
        lines[1]["cotangent"] = surface("x")
        lines[2]["cotangent"] = surface("y")

    receipt, tdir = build(tmp_path, lines={"ram-loop": later_layer_backward})
    fails(receipt, tdir, "first at microstep 0, layer 1, cotangent")

    def forward_over_loss(lines):
        lines[2]["output"] = surface("x")
        lines[3]["loss"] = surface("y")
        lines[1]["cotangent"] = surface("z")

    receipt, tdir = build(tmp_path, arms=("mono", "ssd"), lines={"ssd": forward_over_loss})
    fails(receipt, tdir, "first at microstep 0, layer 1, output")


def test_mutation_witness_fire_fails(tmp_path):
    fired = {"checked": 12, "expected": 12, "mismatches": 1}
    receipt, tdir = build(tmp_path, extra={"witness": fired})
    fails(receipt, tdir, "witness: 1 mismatches")


@pytest.mark.parametrize("witness, needle", [
    (None, "witness missing or malformed"),
    ({"checked": 0, "expected": 0, "mismatches": 0}, "checked 0"),
    ({"checked": 11, "expected": 12, "mismatches": 0}, "checked 11, declared 12"),
    ({"checked": 12, "mismatches": 0}, "witness missing or malformed"),
    ({"checked": 12, "expected": 12, "mismatches": None}, "witness missing or malformed"),
])
def test_a_witness_that_saw_nothing_fails(tmp_path, witness, needle):
    receipt, tdir = build(tmp_path, extra={"witness": witness})
    fails(receipt, tdir, needle)


# -- nonzero gradients -------------------------------------------------------------------------


def _zero_grads(from_microstep, to_microstep):
    def edit(lines):
        for rec in lines:
            kind = rec.get("kind")
            if kind == "microstep" and from_microstep <= rec["microstep"] <= to_microstep:
                rec["grads"]["nonzero"] = 0
    return edit


def test_zero_grads_before_the_first_update_are_allowed(tmp_path):
    edit = _zero_grads(0, 1)  # g = 2: the first update comes after microstep 1
    receipt, tdir = build(tmp_path, lines={"mono": edit, "ram-loop": edit})
    assert verify_receipt(receipt, tdir) == []


@pytest.mark.parametrize("microstep", [2, 5])
def test_zero_grads_after_the_first_update_fail_in_any_arm(tmp_path, microstep):
    edit = _zero_grads(microstep, microstep)
    receipt, tdir = build(tmp_path, lines={"mono": edit, "ram-loop": edit})
    fails(receipt, tdir, "nonzero grad fraction after the first update is 0.0")


def test_fraction_is_the_worst_microstep(tmp_path):
    def edit(lines):
        lines[-2]["grads"]["nonzero"] = 2

    receipt, tdir = build(tmp_path, lines={"mono": edit, "ram-loop": edit})
    assert receipt["arms"][0]["nonzero_grad_fraction"] == 2 / ELEMENTS
    assert verify_receipt(receipt, tdir) == []


def test_a_run_with_nothing_after_the_first_update_cannot_pass(tmp_path):
    receipt, tdir = build(tmp_path, steps=1)
    fails(receipt, tdir, "nonzero grad fraction after the first update is None")


# -- empty surfaces ----------------------------------------------------------------------------


def _empty(*where):
    """Empties a surface: each ``where`` is an index into the arm's lines and a field."""

    def edit(lines):
        for index, field in where:
            lines[index][field].update(elements=0, nonzero=0, b2=digest_bytes("none", (), b""))
    return edit


@pytest.mark.parametrize("where, needle", [
    ((-1, "adapter"), "final adapter is empty"),
    ((-1, "adam"), "final adam is empty"),
    ((-1, "scheduler"), "final scheduler is empty"),
    ((3, "logits"), "microstep 0, logits is empty"),
    ((3, "loss"), "microstep 0, loss is empty"),
    ((1, "cotangent"), "microstep 0, layer 0, cotangent is empty"),
])
def test_a_surface_with_nothing_in_it_fails_though_the_arms_agree(tmp_path, where, needle):
    edit = _empty(where)
    receipt, tdir = build(tmp_path, lines={"mono": edit, "ram-loop": edit})
    fails(receipt, tdir, needle)


def _layer_lines(layer, records=6):
    return [1 + 3 * m + layer for m in range(records)]


def test_a_layer_may_hold_no_adapter_when_the_others_hold_all_of_it(tmp_path):
    edit = _empty(*((i, "lora_grad") for i in _layer_lines(0)))
    receipt, tdir = build(tmp_path, lines={"mono": edit, "ram-loop": edit})
    assert verify_receipt(receipt, tdir) == []


def test_a_layer_with_no_adapter_fails_when_the_gradients_are_not_all_elsewhere(tmp_path):
    edit = _empty(*((i, "lora_grad") for i in _layer_lines(0) + _layer_lines(1)))
    receipt, tdir = build(tmp_path, lines={"mono": edit, "ram-loop": edit})
    fails(receipt, tdir, "microstep 0, layer 0, lora_grad is empty")


def test_grads_narrowed_to_part_of_the_adapter_fail_though_the_arms_agree(tmp_path):
    def edit(lines):
        lines[3]["grads"]["elements"] = ELEMENTS // 2
        lines[3]["grads"]["nonzero"] = ELEMENTS // 2

    receipt, tdir = build(tmp_path, lines={"mono": edit, "ram-loop": edit})
    fails(receipt, tdir, "microstep 0 has 4 grad elements, and the final adapter 8")


# -- what looks outside Metal -------------------------------------------------------------------


def _measured(gate, **fields):
    return {"schema": RECEIPT_SCHEMA, "gate": gate, "model": "Qwen/Qwen2.5-7B", "seq": 128, "g": 1,
            "steps": 1, "first_microstep": 0, "expected_records": 1, "lora_dropout": 0.0,
            "pin": dict(PIN), "pass": None, **fields}


def _gradient(cosine=1.0, gpu=1.0, cpu=1.0):
    return {"dot": cosine * gpu * cpu, "gpu_norm": gpu, "cpu_norm": cpu}


def _cpu_ref(**fields):
    return _measured("cpu-ref", **{
        "forward_only": False,
        "adapter_init": {"gpu": "d", "cpu": "d"},
        "loss": {"gpu": 13.657, "cpu": 13.6575},
        "first_loss": {"here": 13.659, "cuda": 13.6575},
        "hidden": [1e-6, 2e-6],
        "lora_b": {"a.lora_B.weight": _gradient(),
                   "b.lora_B.weight": _gradient(0.9995, 1.0, 1.005)},
        **fields,
    })


def test_a_cpu_ref_within_its_tolerances_passes():
    assert verify_receipt(_cpu_ref(), None) == []
    assert short_of(_cpu_ref()) is None  # one microstep is the whole gate


@pytest.mark.parametrize("fields, needle", [
    ({"adapter_init": {"gpu": "d", "cpu": "e"}}, "did not start from one adapter"),
    ({"adapter_init": None}, "did not start from one adapter"),
    ({"loss": {"gpu": 13.0, "cpu": 13.6575}}, "over 0.005 apart"),
    ({"loss": {"gpu": None, "cpu": 13.6575}}, "over 0.005 apart"),
    ({"first_loss": {"here": 12.4, "cuda": 13.6575}}, "the CUDA receipt says 13.6575"),
    ({"first_loss": None}, "no CUDA receipt names a first loss"),
    ({"hidden": [1e-6, 0.02]}, "layer 1: output differs by 0.02"),
    ({"hidden": [1e-6, None]}, "layer 1: output differs by None"),
    ({"hidden": []}, "no layer outputs"),
    ({"lora_b": {}}, "no lora_B gradients"),
    ({"lora_b": {"x": _gradient(0.98)}}, "x: cosine"),
    ({"lora_b": {"x": _gradient(1.0, 1.06, 1.0)}}, "x: norm differs"),
    ({"lora_b": {"x": _gradient(0.998)}}, "the whole of lora_B: cosine"),
    ({"lora_b": {"x": _gradient(1.0, 1.02, 1.0)}}, "the whole of lora_B: norm differs"),
    ({"lora_b": {"x": _gradient(1.0, 0.0, 1.0)}}, "x: a gradient of zero"),
    ({"lora_b": {"x": {"dot": None, "gpu_norm": 1.0, "cpu_norm": 1.0}}}, "x: its dot and norms"),
    ({"forward_only": True}, "forward only is for 32B and 72B"),
])
def test_a_cpu_ref_outside_its_tolerances_fails(fields, needle):
    assert any(needle in f for f in verify_receipt(_cpu_ref(**fields), None))


def test_forward_only_is_enough_at_the_sizes_a_cpu_vjp_cannot_reach():
    receipt = _cpu_ref(forward_only=True, model="Qwen/Qwen2.5-32B", lora_b={})
    assert verify_receipt(receipt, None) == []


SETTINGS = {"gradient_accumulation_steps": 1, "learning_rate": 2e-5, "max_seq_length": 256}


def _trained(losses=(2.0, 1.0), **fields):
    return {"settings": SETTINGS, "losses": [*losses], "tokens": [5.0, 6.0], **fields}


def _track(mac=None, cuda=None, **fields):
    mac, cuda = mac or _trained(), cuda or _trained((2.05, 1.0))
    worst = track_relative(mac["losses"], cuda["losses"]) if len(cuda["losses"]) == 2 else None
    return _measured("track", **{"steps": 2, "expected_records": 2, "mac": mac, "cuda": cuda,
                                 "max_relative": worst, **fields})


def test_a_track_within_five_percent_passes_and_says_how_far():
    receipt = _track()
    assert receipt["max_relative"] == pytest.approx(0.05 / 2.05)
    assert verify_receipt(receipt, None) == []


@pytest.mark.parametrize("receipt, needle", [
    (_track(cuda=_trained((2.2, 1.0))), "over 0.05"),
    (_track(max_relative=0.0), "and the losses say"),
    (_track(cuda=_trained((2.05,))), "each run must hold 2"),
    (_measured("track", steps=2, expected_records=2, mac=_trained((0.0, 1.0)),
               cuda=_trained((0.0, 1.0))),
     "each run must hold 2"),  # a loss of 0.0 is the fault this looks for
    (_track(mac=_trained(tokens=[5.0, 7.0])), "different rows"),
    (_track(mac=_trained(settings={**SETTINGS, "max_seq_length": 128})), "max_seq_length"),
    (_track(mac=_trained(settings={}), cuda=_trained(settings={})), "track runs at learning_rate"),
    ({**_track(), "mac": "none"}, "track runs at"),
], ids=range(8))
def test_a_track_that_is_not_the_same_run_or_drifts_fails(receipt, needle):
    assert any(needle in f for f in verify_receipt(receipt, None))


def test_a_track_of_too_few_steps_is_a_probe():
    assert "needs 20" in short_of(_track())
    assert short_of({**_track(), "steps": 20}) is None


def test_a_ruled_gate_still_needs_its_declaration():
    receipt = _track()
    receipt["pin"]["poros_commit"] = "unknown"
    assert "pin.poros_commit is unknown" in verify_receipt(receipt, None)[0]


# -- pins --------------------------------------------------------------------------------------


def test_mixed_pins_refuse_to_compare(tmp_path):
    receipt, tdir = build(tmp_path)
    receipt["arms"][1]["pin"] = PIN | {"macos_build": "other"}
    receipt["arms"][1]["digests"]["adam"] = "0" * 64  # a difference that must not be reported
    failures = verify_receipt(receipt, tdir)
    assert any("refused to compare, mixed pins: arm ram-loop: macos_build" in f for f in failures)
    assert not any("differ" in f for f in failures), failures


def test_a_transcript_from_another_pin_refuses_to_compare(tmp_path):
    def other(lines):
        lines[0]["pin"] = PIN | {"torch_hash": "other"}

    receipt, tdir = build(tmp_path, lines={"ram-loop": other})
    receipt["arms"][1]["pin"] = dict(PIN)  # the receipt claims one pin, the transcript another
    fails(receipt, tdir, "refused to compare, mixed pins: arm ram-loop: torch_hash")


def test_pin_differences_sees_a_key_on_one_side():
    assert pin_differences({"a": 1, "b": 2}, {"a": 1, "c": 2}) == ["b", "c"]
    assert pin_differences({"a": 1}, {"a": 1}) == []


@pytest.mark.parametrize("key", PIN_KEYS)
@pytest.mark.parametrize("value", [None, ""])
def test_every_pin_key_is_required(tmp_path, key, value):
    receipt, tdir = build(tmp_path)
    receipt["pin"][key] = value
    assert f"pin.{key} missing" in verify_receipt(receipt, tdir)


def test_ssd_identity_must_be_hashed(tmp_path):
    receipt, tdir = build(tmp_path)
    receipt["pin"]["ssd_sha256"] = "S6XNNG0T123456"
    fails(receipt, tdir, "SSD identity must be hashed")
    receipt["pin"]["ssd_sha256"] = "none"
    assert not any("hashed" in f for f in verify_receipt(receipt, tdir))


# -- gates -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate", ["census", "decode", "pack", "sanity", "mutations", "seamless", None, []]
)
def test_a_gate_with_no_comparator_cannot_pass(tmp_path, gate):
    receipt, tdir = build(tmp_path, extra={"gate": gate})
    assert verify_receipt(receipt, tdir) == [f"no comparator for gate {gate}"]


def test_one_arm_compares_nothing_except_at_the_witness_gate(tmp_path):
    receipt, tdir = build(tmp_path, arms=("ram-loop",), gate="repeat")
    fails(receipt, tdir, "nothing to compare")
    receipt, tdir = build(tmp_path, arms=("ram-loop",), gate="witness", extra={"swap_delta_gb": 0})
    assert verify_receipt(receipt, tdir) == []


@pytest.mark.parametrize("swap", [None, 0.01])
def test_witness_gate_requires_swap_delta_zero(tmp_path, swap):
    extra = {"swap_delta_gb": swap}
    receipt, tdir = build(tmp_path, arms=("ram-loop",), gate="witness", extra=extra)
    fails(receipt, tdir, "required 0")


def test_direct_needs_its_named_arms(tmp_path):
    receipt, tdir = build(tmp_path, arms=("ram-loop", "ram-loop-2"))
    fails(receipt, tdir, "requires an arm named mono")


def test_direct_at_half_b_needs_plain(tmp_path):
    receipt, tdir = build(tmp_path, model="m/0.5B")
    fails(receipt, tdir, "requires an arm named plain")
    receipt, tdir = build(tmp_path, model="m/0.5B", arms=("plain", "mono", "ram-loop"),
                          layers={"plain": 0})
    assert verify_receipt(receipt, tdir) == []


def test_streaming_needs_both_sources(tmp_path):
    receipt, tdir = build(tmp_path, arms=("ram-loop", "ram-loop-2"), gate="streaming")
    fails(receipt, tdir, "requires an arm named ssd")


def test_duplicate_arm_names_fail(tmp_path):
    receipt, tdir = build(tmp_path)
    receipt["arms"][1]["name"] = "mono"
    fails(receipt, tdir, "duplicate arm name")


@pytest.mark.parametrize("name", ["../up", "a/b", "", ".hidden", None, "Mono"])
def test_a_bad_arm_name_reads_no_file(tmp_path, name):
    receipt, tdir = build(tmp_path)
    receipt["arms"][1]["name"] = name
    assert "bad arm name" in verify_receipt(receipt, tdir)


def test_missing_transcript_fails(tmp_path):
    receipt, tdir = build(tmp_path)
    (tdir / "mono.jsonl").unlink()
    fails(receipt, tdir, "arm mono: no transcript")


def test_arm_with_no_block_loop_is_compared_above_the_layers_only(tmp_path):
    def wrong_layer(lines):
        lines[1]["output"] = surface("x")

    receipt, tdir = build(tmp_path, arms=("plain", "ssd"), layers={"plain": 0}, gate="repeat",
                          lines={"ssd": wrong_layer})
    assert verify_receipt(receipt, tdir) == []


def test_a_pair_with_layers_is_still_compared_when_the_first_arm_has_none(tmp_path):
    def wrong_layer(lines):
        lines[1]["output"] = surface("x")

    receipt, tdir = build(tmp_path, arms=("plain", "mono", "ram-loop"), layers={"plain": 0},
                          lines={"ram-loop": wrong_layer})
    fails(receipt, tdir, "mono and ram-loop differ on records, first at microstep 0, layer 0")


def test_different_layer_counts_fail(tmp_path):
    receipt, tdir = build(tmp_path, layers={"mono": 2, "ram-loop": 3})
    fails(receipt, tdir, "record 2 and 3 layers")


def test_observer_compares_final_digests_with_capture_off(tmp_path):
    def capture_off(lines):
        lines[:] = [lines[0] | {"expected_records": 0, "layers": 0}, lines[-1]]

    receipt, tdir = build(tmp_path, arms=("on", "off"), gate="observer", lines={"off": capture_off},
                          extra={"witness": None})
    assert receipt["arms"][1]["records"] == 0
    assert verify_receipt(receipt, tdir) == []

    def wrong_adam(lines):
        capture_off(lines)
        lines[-1]["adam"] = surface("x")

    receipt, tdir = build(tmp_path, arms=("on", "off"), gate="observer", lines={"off": wrong_adam},
                          extra={"witness": None})
    fails(receipt, tdir, "arms on and off differ on adam")


def test_observer_still_catches_a_short_run(tmp_path):
    def short(lines):
        lines[:] = [lines[0] | {"expected_records": 0, "layers": 0}, lines[-1] | {"microsteps": 5}]

    receipt, tdir = build(tmp_path, arms=("on", "off"), gate="observer",
                          lines={"on": short, "off": short}, extra={"witness": None})
    fails(receipt, tdir, "ran to microstep 5, declared 6")


@pytest.mark.parametrize("key, value, needle", [
    ("seq", 0, "seq must be"), ("seq", "128", "seq must be"), ("g", None, "g must be"),
    ("steps", 1.5, "steps must be"), ("expected_records", True, "expected_records must be"),
    ("first_microstep", -1, "first_microstep must be"), ("lora_dropout", None, "lora_dropout"),
    ("model", "", "model missing"), ("pin", None, "pin missing"), ("arms", None, "arms missing"),
    ("schema", "other", "not a poros.metal.receipt.v1"),
])
def test_bad_declarations_fail(tmp_path, key, value, needle):
    receipt, tdir = build(tmp_path)
    receipt[key] = value
    fails(receipt, tdir, needle)


def test_a_header_that_disagrees_with_the_receipt_fails(tmp_path):
    def other(lines):
        lines[0]["seq"] = 512

    receipt, tdir = build(tmp_path, lines={"ram-loop": other})
    fails(receipt, tdir, "transcript seq is 512, receipt says 128")


# -- only verify.py writes pass -----------------------------------------------------------------


def test_verdict_written_and_rechecked_without_transcripts(tmp_path):
    receipt, tdir = build(tmp_path, steps=20)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    assert write_verdict(path, tdir) == []
    stored = json.loads(path.read_text())
    assert stored["pass"] is True and stored["failures"] == []
    assert check_stored(stored) == []
    assert check_stored(stored, tdir) == []
    assert not (tmp_path / "r.json.tmp").exists()


@pytest.mark.parametrize("gate, model, steps, probe", [
    ("direct", "Qwen/Qwen2.5-7B", 19, True),
    ("direct", "Qwen/Qwen2.5-7B", 20, False),
    ("repeat", "Qwen/Qwen2.5-7B", 4, True),
    ("repeat", "Qwen/Qwen2.5-72B", 4, False),
    ("witness", "Qwen/Qwen2.5-7B", 200, True),
    ("witness", "Qwen/Qwen2.5-72B", 200, False),
    ("witness", "Qwen/Qwen2.5-72B", 2, True),
])
def test_a_short_run_is_a_probe_and_never_a_pass(tmp_path, gate, model, steps, probe):
    receipt, tdir = build(tmp_path, gate=gate, model=model, steps=steps,
                          extra={"swap_delta_gb": 0})
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    answer = write_verdict(path, tdir)
    stored = json.loads(path.read_text())
    assert stored["pass"] is (not probe) and stored["failures"] == []
    assert bool(stored["probe"]) is probe and answer == ([stored["probe"]] if probe else [])
    wanted = [stored["probe"], "stored pass is False"] if probe else []
    assert check_stored(stored, tdir) == wanted


def test_a_resume_run_has_a_length_to_speak_for_its_gate():
    assert short_of({"gate": "resume", "model": "Qwen/Qwen2.5-32B", "steps": 7})
    assert short_of({"gate": "resume", "model": "Qwen/Qwen2.5-72B", "steps": 7}) is None


def test_a_short_run_that_failed_reads_as_failed(tmp_path):
    receipt, tdir = build(tmp_path, steps=2)
    receipt["arms"][1]["digests"]["adam"] = "0" * 64
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    assert write_verdict(path, tdir) != []
    stored = json.loads(path.read_text())
    assert stored["pass"] is False and stored["probe"] is None and stored["failures"]


def test_a_pass_stored_at_a_short_run_does_not_stand(tmp_path):
    receipt, tdir = build(tmp_path, steps=2)
    receipt["pass"] = True
    receipt["sealed_b2"] = seal(receipt)
    assert any("a probe" in f for f in check_stored(receipt))


def test_a_commit_nobody_passed_fails(tmp_path):
    receipt, tdir = build(tmp_path)
    receipt["pin"]["poros_commit"] = "unknown"
    assert any("pin.poros_commit" in f for f in verify_receipt(receipt, tdir))


def test_failed_verdict_is_written_false_with_its_reasons(tmp_path):
    receipt, tdir = build(tmp_path)
    receipt["arms"][1]["digests"]["adam"] = "0" * 64
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    assert write_verdict(path, tdir) != []
    stored = json.loads(path.read_text())
    assert stored["pass"] is False and stored["failures"]
    assert check_stored(stored) != []


def test_a_pass_set_by_hand_fails(tmp_path):
    receipt, tdir = build(tmp_path)
    receipt["pass"] = True
    assert any("seal does not match" in f for f in check_stored(receipt))
    assert any("seal does not match" in f for f in check_stored(receipt, tdir))


@pytest.mark.parametrize("edit", [
    lambda r: r.update({"s_per_step": 1.0}),
    lambda r: r["arms"][0].update({"records": 6, "note": "x"}),
    lambda r: r["pin"].update({"route": "other"}),
    lambda r: r.update({"failures": ["x"]}),
])
def test_a_receipt_edited_after_verification_fails(tmp_path, edit):
    receipt, tdir = build(tmp_path)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    write_verdict(path, tdir)
    stored = json.loads(path.read_text())
    edit(stored)
    assert any("seal does not match" in f for f in check_stored(stored))


def test_false_flipped_to_true_by_hand_fails(tmp_path):
    receipt, tdir = build(tmp_path)
    receipt["arms"][1]["nonfinite"] = 1
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    write_verdict(path, tdir)
    stored = json.loads(path.read_text())
    stored["pass"] = True
    stored["failures"] = []
    assert check_stored(stored) != []


def _verdict_lost_on_a_layer(tmp_path):
    """Only the transcripts can see this failure: the receipt's own fields all agree."""
    def wrong_layer(lines):
        lines[1]["output"] = surface("x")

    receipt, tdir = build(tmp_path, steps=20, lines={"ram-loop": wrong_layer})
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    assert write_verdict(path, tdir) != []
    return json.loads(path.read_text())


def test_a_verdict_of_false_stays_false_with_no_transcripts_at_hand(tmp_path):
    stored = _verdict_lost_on_a_layer(tmp_path)
    assert stored["pass"] is False
    assert check_stored(stored) == ["stored pass is False"]


def test_false_flipped_to_true_is_seen_by_the_seal_alone(tmp_path):
    stored = _verdict_lost_on_a_layer(tmp_path)
    stored["pass"] = True
    assert [f[:18] for f in check_stored(stored)] == ["seal does not matc"]


def test_no_receipt_writer_but_this_module_sets_pass():
    """Any file that writes a receipt leaves ``pass`` null: this one computes it."""
    setter = re.compile(r"""["']pass["']\s*(?:\]\s*=|:)\s*(?!None\b)""")
    offenders = []
    for folder in (ROOT / "poros" / "metal", ROOT / "scripts" / "metal"):
        for path in sorted(folder.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if path != VERIFY and "receipt.v1" in text and setter.search(text):
                offenders.append(path.name)
    assert offenders == []


# -- command line ------------------------------------------------------------------------------


def _run(capsys, *argv):
    code = main(list(argv))
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_cli_write_then_recheck(tmp_path, capsys):
    receipt, tdir = build(tmp_path, steps=20)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    code, out = _run(capsys, str(path))  # never verified
    assert code == 1 and out[0]["pass"] is False
    code, out = _run(capsys, str(path), "--transcripts", str(tdir))  # read-only, evidence fine
    assert code == 0 and out[0]["pass"] is True and out[0]["transcripts_checked"] is True
    assert json.loads(path.read_text())["pass"] is None
    code, out = _run(capsys, str(path), "--transcripts", str(tdir), "--write")
    assert code == 0 and json.loads(path.read_text())["pass"] is True
    code, out = _run(capsys, str(path))
    assert code == 0 and out == [
        {"receipt": "r.json", "pass": True, "transcripts_checked": False, "failures": []}
    ]


def test_cli_output_names_no_directory(tmp_path, capsys):
    receipt, tdir = build(tmp_path)
    receipt["arms"][1]["nonfinite"] = 1
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    _, out = _run(capsys, str(path), "--transcripts", str(tdir), "--write")
    assert str(tmp_path) not in json.dumps(out)
    assert str(tmp_path) not in path.read_text()


def test_cli_stored_true_against_bad_evidence_fails(tmp_path, capsys):
    receipt, tdir = build(tmp_path)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    _run(capsys, str(path), "--transcripts", str(tdir), "--write")
    t = tdir / "mono.jsonl"
    t.write_text(t.read_text().replace('"nonfinite_count": 0', '"nonfinite_count": 1', 1))
    code, out = _run(capsys, str(path), "--transcripts", str(tdir))
    assert code == 1 and out[0]["pass"] is False


def test_cli_write_refuses_no_transcripts(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--write"])
    assert exc.value.code == 2
    assert path.read_text() == "{}"


def test_cli_transcripts_take_one_receipt(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["a.json", "b.json", "--transcripts", str(tmp_path)])
    assert exc.value.code == 2


@pytest.mark.parametrize("text", ["not json", "[]", '{"a": NaN}', '{"a": 1, "a": 2}'])
def test_cli_bad_receipt_file_fails_cleanly(tmp_path, capsys, text):
    path = tmp_path / "r.json"
    path.write_text(text)
    code, out = _run(capsys, str(path))
    assert code == 1 and out[0]["pass"] is False


def test_cli_refuses_a_nan_anywhere_in_an_otherwise_honest_receipt(tmp_path, capsys):
    receipt, tdir = build(tmp_path)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt).replace('"s_per_step": null', '"s_per_step": NaN'))
    code, out = _run(capsys, str(path), "--transcripts", str(tdir))
    assert code == 1 and "JSON constant NaN" in out[0]["failures"][0]


def test_cli_missing_file_fails_cleanly(tmp_path, capsys):
    code, out = _run(capsys, str(tmp_path / "absent.json"))
    assert code == 1 and "unreadable" in out[0]["failures"][0]


# -- stdlib only --------------------------------------------------------------------------------

def test_import_through_the_package_needs_no_torch_and_no_numpy():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['numpy'] = None\n"
        "import poros.metal.verify as v\n"
        "print(v.digest_bytes('float32', (), b''))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


def test_it_runs_by_path_on_a_bare_python(tmp_path):
    receipt, tdir = build(tmp_path, steps=20)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    bare = [sys.executable, "-I", "-S", str(VERIFY)]
    r = subprocess.run([*bare, str(path), "--transcripts", str(tdir), "--write"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert json.loads(path.read_text())["pass"] is True
    r = subprocess.run([*bare, str(path)], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    t = tdir / "mono.jsonl"
    t.write_text(t.read_text()[:-1])
    r = subprocess.run([*bare, str(path), "--transcripts", str(tdir)],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 1 and '"pass": false' in r.stdout
