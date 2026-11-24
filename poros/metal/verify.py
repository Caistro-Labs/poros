"""Verifier for the Metal gate receipts and transcripts. Stdlib only, so it runs on any box.

``pass`` is computed and written here and nowhere else. Arm scripts leave it ``null``.

A digest is blake2b-256 over ``dtype|shape|`` and the tensor's little-endian bytes. So -0.0 and
0.0 differ, and the same NaN bits agree, which is why every surface also carries
``nonfinite_count`` and the verifier requires 0 on each.

A transcript (poros.metal.transcript.v1) is JSONL, in exactly this order:

    header    {"schema", "arm", "gate", "model", "seq", "g", "steps", "first_microstep",
               "lora_dropout", "layers", "expected_records", "pin"}
    for each of expected_records microsteps, from first_microstep:
      layers x {"kind": "layer", "microstep", "layer", "rng", "input", "output", "cotangent",
                "lora_grad"}
      1 x      {"kind": "microstep", "microstep", "loss", "logits", "grads"}
    1 x       {"kind": "final", "microsteps", "adapter", "adam", "scheduler"}

and each surface is {"b2", "nonfinite_count", "elements", "nonzero"}. The record counts are
declared before the run, from config, so a short arm fails. Nothing here truncates to the
shorter side or skips a missing value.

A receipt (poros.metal.receipt.v1) summarises one transcript per arm, and a verdict is made from
the transcripts alone. Without them, a stored verdict is only checked against its seal: a digest
over the receipt, so an edit after verification shows. It has no secret: it catches mistakes, not
someone who recomputes it. The ``cpu-ref`` and ``track`` gates compare numbers, within a stated
tolerance, against something outside Metal's own arms. Those numbers are in the receipt itself, and
so are the ``agents`` gate's: what `poros agents` was asked, and what it said.

    python poros/metal/verify.py RECEIPT --transcripts DIR --write
    python poros/metal/verify.py docs/validation/metal/m4_16gb/*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

RECEIPT_SCHEMA = "poros.metal.receipt.v1"
TRANSCRIPT_SCHEMA = "poros.metal.transcript.v1"

STEP_SURFACES = {"losses": "loss", "logits": "logits", "grads": "grads"}  # receipt -> record
FINAL_SURFACES = ("adapter", "adam", "scheduler")
FORWARD_FIELDS = ("rng", "input", "output")
BACKWARD_FIELDS = ("cotangent", "lora_grad")
FIELDS = {
    "layer": FORWARD_FIELDS + BACKWARD_FIELDS,
    "microstep": tuple(STEP_SURFACES.values()),
    "final": FINAL_SURFACES,
}
SURFACE_KEYS = {"b2", "nonfinite_count", "elements", "nonzero"}

# The gates whose arms compare byte for byte. A gate here or in RULED has a comparator, and any
# other's receipt cannot pass: pack, sanity, mutations, seamless.
COMPARED = {"direct", "direct-chats", "streaming", "repeat", "witness", "observer", "prove"}
FINAL_ONLY_GATE = "observer"  # one of its arms runs with per-step capture off
REQUIRED_ARMS = {
    "direct": {"mono", "ram-loop"},
    "direct-chats": {"mono", "ram-loop"},
    "streaming": {"ram-loop", "ssd"},
}
WITNESS_GATES = COMPARED - {"observer"}
# The steps a run needs to speak for its gate, then the sizes that differ. Fewer: a clean run is a
# probe. Prove has no such length.
MINIMUM_STEPS = {
    "direct": (20, {}),
    "direct-chats": (20, {}),
    "streaming": (20, {}),
    "repeat": (20, {"32B": 4, "72B": 4}),
    "observer": (20, {"32B": 4, "72B": 4}),
    "witness": (200, {"7B": 1000}),
    "resume": (12, {"72B": 7}),  # K = 10 and 5: checkpoint-K holds K + 1 steps, and one follows
    "track": (20, {}),
    "agents": (12, {}),
}

# What the gates that look outside Metal allow. Faults every Mac arm shares agree byte for byte, so
# only these can see them. Each tolerance is a judgement, and the reason is beside it.
# A wrong gradient moves the whole vector; rounding moves a tensor or two. At 0.5B x86's own two
# bf16 kernels, with no Metal in them, agree to 0.99990 whole, and on their worst tensor to 0.99863
# and 1.46%; the Mac's GPU and CPU to 0.99988, 0.99722 and 2.09%. The same math in fp32 lands at
# 0.99857, 0.96645 and 10.8%, so each of these still catches it. 0.999 and 1% on every tensor were
# set before any data, and correct kernels miss them.
GRAD_COSINE, GRAD_NORM = 0.9995, 1e-2  # the whole of lora_B
TENSOR_COSINE, TENSOR_NORM = 0.99, 5e-2  # any one tensor of it
FIRST_LOSS = 5e-3  # 7.6 times an honest CPU decode's own miss of its CUDA receipt at 0.5B
HIDDEN = 1.5e-2  # a layer's output: 1.5 times the worst at 7B, 1.03e-2, GPU against an x86 CPU
FORWARD_ONLY_SIZES = ("32B", "72B")  # a bf16 VJP on the CPU takes 74 and 166 minutes there
# Two runs at G=1 and lr 2e-5 drift apart a little by design: 0.7% at 0.5B and 2.1% at 7B, Mac
# against CUDA. A wrong gradient, or a rate the model cannot take, lands far above this, and 1e-3 of
# noise in one CUDA layer's weights spans 1.6% to 8.6%. So over 5% means look, and under it means no
# more.
TRACK_RELATIVE = 0.05
TRACK_AT = {"gradient_accumulation_steps": 1, "learning_rate": 2e-5}

# Every check an `agents` receipt must hold, on any machine: scripts/metal/agents_gate.py names
# them. What was called and said is the receipt's own. What a pass cannot argue with is ruled here.
AGENT_CHECKS = (
    "read.version", "read.models_fit", "read.data", "read.errors", "security.outside_words",
    "life.straight", "life.stopped", "life.same_bytes", "life.more", "life.from_run",
    "life.download", "beside.reads", "beside.busy", "checkpoint.list", "checkpoint.info",
    "checkpoint.download", "checkpoint.delete", "result.export", "result.eval",
    "result.recipes_arch", "offered.runs", "words.paths", "security.planted",
)
AGENT_ADAPTERS = ("straight", "stopped", "waited", "person")  # by four roads, one set of bytes
AGENT_FAMILIES = {"dense", "gemma", "moe"}  # what a Mac's receipt ran by these commands alone

PIN_KEYS = (
    "chip", "gpu_cores", "macos_build", "python", "torch", "torch_hash", "transformers", "peft",
    "shader_sha256", "route", "ssd_sha256", "poros_commit", "engine_tree_hash", "wheel_hash",
)
HEADER_KEYS = {
    "schema", "arm", "gate", "model", "seq", "g", "steps", "first_microstep", "lora_dropout",
    "layers", "expected_records", "pin",
}
HEX64 = re.compile(r"[0-9a-f]{64}")
ARM_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")


def digest_bytes(dtype: str, shape, data) -> str:
    """The byte digest of one tensor. ``data`` is its elements, little-endian, in C order."""
    h = hashlib.blake2b(digest_size=32)
    h.update(f"{dtype}|{','.join(str(int(d)) for d in shape)}|".encode())
    h.update(data)
    return h.hexdigest()


def digest_chain(parts) -> str:
    """An ordered digest of strings, for a surface made of many tensors or many microsteps."""
    h = hashlib.blake2b(digest_size=32)
    for part in parts:
        h.update(part.encode())
        h.update(b"\n")
    return h.hexdigest()


def _no_constants(name):
    raise ValueError(f"JSON constant {name} is not allowed")


def _no_duplicates(pairs):
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError(f"duplicate key in {[k for k, _ in pairs]}")
    return dict(pairs)


def _loads(text):
    return json.loads(text, parse_constant=_no_constants, object_pairs_hook=_no_duplicates)


def _is_count(x, minimum=0) -> bool:
    return type(x) is int and x >= minimum


def _is_surface(s) -> bool:
    return (
        isinstance(s, dict)
        and set(s) == SURFACE_KEYS
        and isinstance(s["b2"], str)
        and bool(HEX64.fullmatch(s["b2"]))
        and all(_is_count(s[k]) for k in ("nonfinite_count", "elements", "nonzero"))
        and max(s["nonzero"], s["nonfinite_count"]) <= s["elements"]
    )


class Transcript:
    """One arm's transcript, parsed strictly. ``problems`` is empty only if every line was the
    one its header promised, in order, with nothing left over."""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.b2 = hashlib.blake2b(raw, digest_size=32).hexdigest()
        self.header = None
        self.layer_records, self.step_records, self.final = [], [], None
        self.problems = []
        if not raw.endswith(b"\n"):
            self.problems.append("torn: the last line has no newline")
            return
        lines = raw[:-1].split(b"\n")
        try:
            header = _loads(lines[0])
        except ValueError as exc:
            self.problems.append(f"line 1: {exc}")
            return
        if (
            not isinstance(header, dict)
            or set(header) != HEADER_KEYS
            or header["schema"] != TRANSCRIPT_SCHEMA
            or not isinstance(header["pin"], dict)
            or not _is_count(header["g"], 1)
        ):
            self.problems.append("line 1: not a poros.metal.transcript.v1 header")
            return
        first, layers = header["first_microstep"], header["layers"]
        records = header["expected_records"]
        if not all(map(_is_count, (first, layers, records))):
            self.problems.append("line 1: bad counts")
            return
        self.header = header
        wanted = []  # (kind, microstep, layer), the whole file, from the header alone
        for m in range(first, first + records):
            wanted += [("layer", m, n) for n in range(layers)]
            wanted.append(("microstep", m, None))
        wanted.append(("final", None, None))
        if len(lines) - 1 != len(wanted):
            self.problems.append(f"{len(lines) - 1} lines after the header, declared {len(wanted)}")
            return
        for i, want in enumerate(wanted):
            problem = self._take(lines[i + 1], *want)
            if problem:
                self.problems.append(f"line {i + 2}: {problem}")
                return

    def _take(self, line, kind, microstep, layer):
        try:
            rec = _loads(line)
        except ValueError as exc:
            return str(exc)
        if not isinstance(rec, dict) or rec.get("kind") != kind:
            return f"expected a {kind} record"
        if kind == "layer":
            if rec.get("microstep") != microstep or rec.get("layer") != layer:
                return f"expected microstep {microstep} layer {layer}"
            self.layer_records.append(rec)
        elif kind == "microstep":
            if rec.get("microstep") != microstep:
                return f"expected microstep {microstep}"
            self.step_records.append(rec)
        else:
            if not _is_count(rec.get("microsteps")):
                return "final record has no microsteps count"
            self.final = rec
        for field in FIELDS[kind]:
            if not _is_surface(rec.get(field)):
                return f"{field}: malformed surface"
        return None

    def walk(self, layers=None):
        """(where, surface) for every surface, in execution order: forward up the layers, logits
        and loss, backward down the layers, grads, then the final state."""
        per = self.header["layers"]
        layers = per if layers is None else layers
        for i, step in enumerate(self.step_records):
            block, m = self.layer_records[i * per : (i + 1) * per], step["microstep"]
            for n in range(layers):
                for field in FORWARD_FIELDS:
                    yield f"microstep {m}, layer {n}, {field}", block[n][field]
            for field in ("logits", "loss"):
                yield f"microstep {m}, {field}", step[field]
            for n in reversed(range(layers)):
                for field in BACKWARD_FIELDS:
                    yield f"microstep {m}, layer {n}, {field}", block[n][field]
            yield f"microstep {m}, grads", step["grads"]
        for field in FINAL_SURFACES:
            yield f"final {field}", self.final[field]

    def layers_hold_all_grads(self) -> bool:
        per = self.header["layers"]
        return all(
            sum(r["lora_grad"]["elements"] for r in self.layer_records[i * per : (i + 1) * per])
            == step["grads"]["elements"]
            for i, step in enumerate(self.step_records)
        )

    def nonfinite(self) -> tuple:
        """(total non-finite elements across every surface, where the first one is)."""
        total, first = 0, None
        for where, surface in self.walk():
            if surface["nonfinite_count"] and first is None:
                first = where
            total += surface["nonfinite_count"]
        return total, first

    def nonzero_grad_fraction(self):
        """The smallest nonzero fraction after the first update. None if no record comes after."""
        fractions = [
            r["grads"]["nonzero"] / r["grads"]["elements"] if r["grads"]["elements"] else 0.0
            for r in self.step_records
            if r["microstep"] >= self.header["g"]
        ]
        return min(fractions) if fractions else None

    def arm(self) -> dict:
        """This arm's entry in a receipt, with every field derived from the transcript."""
        digests = {
            name: digest_chain(r[field]["b2"] for r in self.step_records)
            for name, field in STEP_SURFACES.items()
        }
        digests.update((field, self.final[field]["b2"]) for field in FINAL_SURFACES)
        return {
            "name": self.header["arm"],
            "records": len(self.step_records),
            "microsteps": self.final["microsteps"],
            "nonfinite": self.nonfinite()[0],
            "nonzero_grad_fraction": self.nonzero_grad_fraction(),
            "transcript_b2": self.b2,
            "pin": self.header["pin"],
            "digests": digests,
        }


def arm_from_transcript(path) -> dict:
    """Build a receipt's arm entry, for the script that writes receipts. Raises on a bad file."""
    t = Transcript(path)
    if t.problems:
        raise ValueError(f"bad transcript: {t.problems[0]}")
    return t.arm()


def first_difference(a: Transcript, b: Transcript):
    """Where two clean transcripts of one length first differ, in execution order. An arm with no
    block loop records no layers, so that pair compares above the layers only."""
    la, lb = a.header["layers"], b.header["layers"]
    layers = la if la and lb else 0
    for (where, x), (_, y) in zip(a.walk(layers), b.walk(layers), strict=True):
        if x != y:
            return where
    return None


def pin_differences(a: dict, b: dict) -> list:
    """The pin keys on which two receipts or arms disagree. Any at all refuses a comparison."""
    return sorted(k for k in set(a) | set(b) if k not in a or k not in b or a[k] != b[k])


def _config_failures(r: dict) -> list:
    out = []
    for key in ("seq", "g", "steps", "expected_records"):
        if not _is_count(r.get(key), 1):
            out.append(f"{key} must be a positive integer")
    if not _is_count(r.get("first_microstep", 0)):
        out.append("first_microstep must be a count")
    if type(r.get("lora_dropout")) not in (int, float):
        out.append("lora_dropout missing")
    if not isinstance(r.get("model"), str) or not r["model"]:
        out.append("model missing")
    if not out and r["expected_records"] != r["steps"] * r["g"]:
        out.append(f"expected_records {r['expected_records']} is not steps x g")
    pin = r.get("pin")
    if not isinstance(pin, dict):
        return out + ["pin missing"]
    out += [f"pin.{k} missing" for k in PIN_KEYS if pin.get(k) in (None, "")]
    if pin.get("poros_commit") == "unknown":
        out.append("pin.poros_commit is unknown: a sync with no git passes POROS_COMMIT")
    ssd = pin.get("ssd_sha256")
    if ssd not in (None, "", "none") and not (isinstance(ssd, str) and HEX64.fullmatch(ssd)):
        out.append("pin.ssd_sha256 is not a sha256: SSD identity must be hashed before writing")
    return out


def _gate_failures(r: dict, gate: str, names: list) -> list:
    out = [
        f"gate {gate} requires an arm named {name}"
        for name in sorted(REQUIRED_ARMS.get(gate, set()) - set(names))
    ]
    if gate == "direct" and r["model"].endswith("0.5B") and "plain" not in names:
        out.append("gate direct at 0.5B requires an arm named plain")
    if len(names) < (1 if gate == "witness" else 2):
        out.append(f"gate {gate} has {len(names)} arm(s): nothing to compare")

    w = r.get("witness")
    if gate in WITNESS_GATES:
        if not isinstance(w, dict) or not all(
            _is_count(w.get(k)) for k in ("checked", "expected", "mismatches")
        ):
            out.append("witness missing or malformed: wants checked, expected, mismatches")
        else:
            if w["mismatches"]:
                out.append(f"witness: {w['mismatches']} mismatches")
            if not w["checked"] or w["checked"] != w["expected"]:
                out.append(f"witness: checked {w['checked']}, declared {w['expected']}")

    if gate == "witness" and r.get("swap_delta_gb") != 0:
        out.append(f"swap_delta_gb is {r.get('swap_delta_gb')}, required 0")
    if gate == "streaming":
        for arm in r["arms"]:
            if arm["name"].startswith("ssd"):
                out += disk_failures(arm)
    return out


def disk_failures(arm: dict) -> list:
    """OS disk counters: the only check that sees an ``ssd`` arm serving from memory."""
    d, name = arm.get("disk"), arm["name"]
    if not isinstance(d, dict):
        return [f"arm {name}: disk counters missing"]
    read, predicted = d.get("read_bytes"), d.get("predicted_read_bytes")
    resident = d.get("preflight_resident_fraction")
    out = []
    if not (_is_count(read) and _is_count(predicted, 1)):
        out.append(f"arm {name}: disk read_bytes and predicted_read_bytes malformed")
    elif read * 100 < predicted * 95:
        out.append(f"arm {name}: read {read} bytes, under 95% of predicted {predicted}")
    if type(resident) not in (int, float) or not 0 <= resident < 0.05:
        out.append(f"arm {name}: preflight resident fraction is {resident}, required under 5%")
    return out


def _arm_failures(r: dict, arm: dict, t: Transcript, final_only: bool) -> list:
    """One arm against its own transcript and the receipt's declaration."""
    name, h, derived = arm["name"], t.header, t.arm()
    out = [
        f"arm {name}: transcript {key} is {h[key]}, receipt says {r.get(key, 0)}"
        for key in ("gate", "model", "seq", "g", "steps", "lora_dropout", "first_microstep")
        if h[key] != r.get(key, 0)
    ]
    declared, records = r["expected_records"], derived["records"]
    if records != declared and not (final_only and records == 0):
        out.append(f"arm {name}: {records} records, declared {declared}")
    last = r.get("first_microstep", 0) + declared
    if derived["microsteps"] != last:
        out.append(f"arm {name}: ran to microstep {derived['microsteps']}, declared {last}")
    out += [
        f"arm {name}: {key} in the receipt does not match its transcript"
        for key in sorted(derived)
        if derived[key] != arm.get(key)
    ]
    total, first = t.nonfinite()
    if total:
        out.append(f"arm {name}: {total} non-finite elements, first at {first}")
    if not final_only and not (derived["nonzero_grad_fraction"] or 0) > 0:
        fraction = derived["nonzero_grad_fraction"]
        out.append(f"arm {name}: nonzero grad fraction after the first update is {fraction}")
    # A digest of nothing agrees with a digest of nothing. A final-only arm has no records to walk.
    # A layer may hold no adapter, as a MoE model's dense one does, if the others hold all of it.
    out += [f"arm {name}: {where} is empty" for where, s in t.walk() if not s["elements"]
            and not (where.endswith("lora_grad") and t.layers_hold_all_grads())]
    adapter = t.final["adapter"]["elements"]
    out += [
        f"arm {name}: microstep {r['microstep']} has {r['grads']['elements']} grad elements, "
        f"and the final adapter {adapter}"
        for r in t.step_records
        if r["grads"]["elements"] != adapter
    ]
    return out


def _pair_failures(a: Transcript, b: Transcript, final_only: bool) -> list:
    name_a, name_b = a.header["arm"], b.header["arm"]
    digests_a, digests_b = a.arm()["digests"], b.arm()["digests"]
    surfaces = FINAL_SURFACES if final_only else (*STEP_SURFACES, *FINAL_SURFACES)
    differ = [s for s in surfaces if digests_a[s] != digests_b[s]]
    where = None
    if not final_only and len(a.step_records) == len(b.step_records):
        la, lb = a.header["layers"], b.header["layers"]
        if la and lb and la != lb:
            return [f"arms {name_a} and {name_b} record {la} and {lb} layers"]
        where = first_difference(a, b)
    if not differ and not where:
        return []
    at = f", first at {where}" if where else ""
    return [f"arms {name_a} and {name_b} differ on {', '.join(differ) or 'records'}{at}"]


def _is_number(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def _apart(a, b, tolerance: float) -> bool:
    """Whether two numbers are further apart than ``tolerance``. Anything else counts as apart."""
    return not (_is_number(a) and _is_number(b) and abs(a - b) <= tolerance)


def _gradient_failures(name: str, g, floor: float, limit: float) -> list:
    keys = ("dot", "gpu_norm", "cpu_norm")
    if not isinstance(g, dict) or not all(_is_number(g.get(k)) for k in keys):
        return [f"{name}: its dot and norms must be finite numbers"]
    dot, gpu, cpu = (g[k] for k in keys)
    if not gpu or not cpu:
        return [f"{name}: a gradient of zero, {gpu} on the GPU and {cpu} on the CPU"]
    cosine, relative = dot / (gpu * cpu), abs(gpu - cpu) / cpu
    out = [f"{name}: cosine {cosine} is under {floor}"] if cosine < floor else []
    if relative > limit:
        out.append(f"{name}: norm differs by {relative}, over {limit}")
    return out


def _whole(grads: dict) -> dict:
    """Every tensor's dot and norms as one vector's, or nothing if any is not a number."""
    try:
        return {"dot": math.fsum(g["dot"] for g in grads.values()),
                "gpu_norm": math.hypot(*(g["gpu_norm"] for g in grads.values())),
                "cpu_norm": math.hypot(*(g["cpu_norm"] for g in grads.values()))}
    except (KeyError, TypeError):
        return {}


def _part(r: dict, key: str) -> dict:
    return r[key] if isinstance(r.get(key), dict) else {}


def _cpu_ref_failures(r: dict) -> list:
    """The Metal arm against the same pack and code on the CPU, and its first loss against what a
    CUDA run on this checkpoint measured. The numbers are the receipt's own: the tensors are gone.
    """
    out = []
    loss, init, first = _part(r, "loss"), _part(r, "adapter_init"), _part(r, "first_loss")
    if not init.get("gpu") or init.get("gpu") != init.get("cpu"):
        out.append(f"the arms did not start from one adapter: {init}")
    if _apart(loss.get("gpu"), loss.get("cpu"), FIRST_LOSS):
        out.append(f"loss {loss.get('gpu')} on the GPU, {loss.get('cpu')} on the CPU: "
                   f"over {FIRST_LOSS} apart")
    if not first:
        out.append(f"no CUDA receipt names a first loss for {r['model']}")
    elif _apart(first.get("here"), first.get("cuda"), FIRST_LOSS):
        out.append(f"first loss {first.get('here')}, and the CUDA receipt says "
                   f"{first.get('cuda')}: over {FIRST_LOSS} apart")
    hidden = r.get("hidden")
    if not isinstance(hidden, list) or not hidden:
        out.append("no layer outputs to compare")
    else:
        out += [f"layer {i}: output differs by {h}, over {HIDDEN}"
                for i, h in enumerate(hidden) if _apart(h, 0.0, HIDDEN)]
    if r.get("forward_only") is True:
        if not any(size in r["model"] for size in FORWARD_ONLY_SIZES):
            out.append(f"forward only is for {' and '.join(FORWARD_ONLY_SIZES)}: {r['model']}")
        return out
    grads = r.get("lora_b")
    if not isinstance(grads, dict) or not grads:
        return out + ["no lora_B gradients to compare"]
    out += _gradient_failures("the whole of lora_B", _whole(grads), GRAD_COSINE, GRAD_NORM)
    for name, g in sorted(grads.items()):
        out += _gradient_failures(name, g, TENSOR_COSINE, TENSOR_NORM)
    return out


def track_relative(mac: list, cuda: list) -> float:
    """The largest relative difference at any step between two runs' losses."""
    return max(abs(m - c) / abs(c) for m, c in zip(mac, cuda, strict=True))


def _track_failures(r: dict) -> list:
    """A Mac run's losses, step by step, against a CUDA run's on the same rows and settings."""
    mac, cuda = _part(r, "mac"), _part(r, "cuda")
    settings, others = _part(mac, "settings"), _part(cuda, "settings")
    out = [f"track runs at {key} {want}" for key, want in TRACK_AT.items()
           if settings.get(key) != want]
    if not settings or settings != others:
        out.append(f"different settings: {pin_differences(settings, others)}")
    if mac.get("tokens") != cuda.get("tokens"):
        out.append("different rows: their steps scored different numbers of tokens")
    runs = [mac.get("losses"), cuda.get("losses")]
    if not all(isinstance(x, list) and len(x) == r["steps"] and all(map(_is_number, x)) and all(x)
               for x in runs):
        return out + [f"each run must hold {r['steps']} nonzero finite losses"]
    worst = track_relative(*runs)
    if worst != r.get("max_relative"):
        out.append(f"max_relative is {r.get('max_relative')}, and the losses say {worst}")
    if worst > TRACK_RELATIVE:
        out.append(f"losses differ by {worst} at worst, over {TRACK_RELATIVE}")
    return out


def _agents_failures(r: dict) -> list:
    """`poros agents` on real runs. Every check must have run and passed, and what no check may wave
    through is ruled from the receipt's facts: what stdout carried, what was offered, the adapter's
    bytes, and what the replies leaked."""
    checks, stdout, adapters = _part(r, "checks"), _part(r, "stdout"), _part(r, "adapters")
    out = [f"check {name} did not run" for name in AGENT_CHECKS if name not in checks]
    out += [f"check {name}: {_part(checks, name).get('why') or 'no verdict'}"
            for name in sorted(checks) if _part(checks, name).get("pass") is not True]
    invalid = stdout.get("invalid")
    if not _is_count(stdout.get("calls"), 1) or not isinstance(invalid, list):
        out.append("no call's stdout was read")
    else:
        out += [f"stdout carried what is not one object: {str(line)[:80]!r}" for line in invalid]
    offered = r.get("offered")
    if not isinstance(offered, list) or not offered:
        out.append("no command was offered, so none was tried")
    else:
        out += [f"an offered command fails, {o.get('command')}: {o['problem']}"
                for o in offered if o.get("problem")]
    missing = [name for name in AGENT_ADAPTERS if not adapters.get(name)]
    if missing:
        out.append(f"no adapter from {', '.join(missing)}")
    elif len({adapters[name] for name in AGENT_ADAPTERS}) != 1:
        out.append(f"the adapters differ: {adapters}")
    leaks = r.get("leaks")
    out += [f"a reply leaked: {leak}" for leak in leaks] if isinstance(leaks, list) else ["leaks?"]
    if r["pin"].get("macos_build") != "not a Mac":
        ran = {b.get("family") for b in _part(r, "big").values() if isinstance(b, dict)}
        if AGENT_FAMILIES - ran:
            out.append(f"no family of {sorted(AGENT_FAMILIES - ran)} ran by these commands alone")
    return out


RULED = {"cpu-ref": _cpu_ref_failures, "track": _track_failures, "agents": _agents_failures}


def verify_receipt(r, transcripts) -> list:
    """The reasons a receipt fails, [] if none. ``transcripts`` is a directory of <arm>.jsonl.

    Never reads ``pass``: see :func:`check_stored` for a receipt that already carries one.
    """
    if not isinstance(r, dict) or r.get("schema") != RECEIPT_SCHEMA:
        return [f"not a {RECEIPT_SCHEMA} receipt"]
    gate = r.get("gate")
    if not isinstance(gate, str) or gate not in COMPARED | RULED.keys():
        return [f"no comparator for gate {gate}"]
    final_only = gate == FINAL_ONLY_GATE

    failures = _config_failures(r)
    if gate in RULED:
        return failures or RULED[gate](r)
    arms = r.get("arms")
    if not isinstance(arms, list) or not all(isinstance(a, dict) for a in arms):
        return failures + ["arms missing"]
    names = [a.get("name") for a in arms]
    if not all(isinstance(n, str) and ARM_NAME.fullmatch(n) for n in names):
        return failures + ["bad arm name"]
    if len(set(names)) != len(names):
        failures.append("duplicate arm name")
    if failures:
        return failures  # a broken declaration leaves nothing sound to compare against
    failures += _gate_failures(r, gate, names)

    loaded = {}
    for name in names:
        path = Path(transcripts) / f"{name}.jsonl"
        if not path.is_file():
            failures.append(f"arm {name}: no transcript")
            continue
        t = Transcript(path)
        if t.problems:
            failures.append(f"arm {name}: transcript {t.problems[0]}")
        else:
            loaded[name] = t

    # Mixed pins refuse to compare: no digest is looked at.
    mixed = []
    for arm in arms:
        pins = [arm.get("pin")]
        if arm["name"] in loaded:
            pins.append(loaded[arm["name"]].header["pin"])
        for pin in pins:
            diff = pin_differences(r["pin"], pin) if isinstance(pin, dict) else ["pin missing"]
            if diff:
                mixed.append(f"arm {arm['name']}: {', '.join(diff)}")
    if mixed:
        return failures + [f"refused to compare, mixed pins: {'; '.join(mixed)}"]

    for arm in arms:
        if arm["name"] in loaded:
            failures += _arm_failures(r, arm, loaded[arm["name"]], final_only)
    # Every pair, since an arm with no layer records must not hide a pair that has them.
    clean = list(loaded.values())
    for i, a in enumerate(clean):
        for b in clean[i + 1 :]:
            failures += _pair_failures(a, b, final_only)
    return failures


def short_of(r) -> str | None:
    """Why a run is a probe and not its gate: too few steps for its model. None if it is not."""
    if not isinstance(r, dict) or not _is_count(r.get("steps")):
        return None
    least, sizes = MINIMUM_STEPS.get(r.get("gate"), (0, {}))
    least = sizes.get(str(r.get("model")).rsplit("-", 1)[-1], least)
    if r["steps"] >= least:
        return None
    return f"a probe, not a pass: {r['steps']} steps, and {r['gate']} at this size needs {least}"


def seal(r: dict) -> str:
    body = {k: v for k, v in r.items() if k != "sealed_b2"}
    return digest_chain([json.dumps(body, sort_keys=True, separators=(",", ":"))])


def check_stored(r, transcripts=None) -> list:
    """Recheck a receipt on disk: from its transcripts, or, with none at hand, by its seal."""
    failures = verify_receipt(r, transcripts) if transcripts is not None else []
    if not isinstance(r, dict):
        return failures or ["a receipt is a JSON object"]
    if probe := short_of(r):
        failures.append(probe)
    if transcripts is not None and r.get("pass") is None:
        return failures
    if r.get("pass") is not True:
        failures.append(f"stored pass is {r.get('pass')}")
    if r.get("sealed_b2") != seal(r):
        failures.append("seal does not match: edited after verification, or never verified")
    return failures


def write_verdict(path, transcripts) -> list:
    """Compute ``pass`` from a receipt and its transcripts, write it atomically, and return what
    stands in its way: the failures, or that the run was too short to be more than a probe."""
    path = Path(path)
    r = _loads(path.read_text(encoding="utf-8"))
    if not isinstance(r, dict):
        raise ValueError("a receipt is a JSON object")
    failures = verify_receipt(r, transcripts)
    probe = None if failures else short_of(r)
    r["pass"] = not failures and probe is None
    r["probe"] = probe
    r["failures"] = failures
    r["sealed_b2"] = seal(r)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return [probe] if probe else failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("receipts", nargs="+")
    ap.add_argument("--transcripts", help="directory of <arm>.jsonl, for one receipt")
    ap.add_argument("--write", action="store_true", help="write the verdict into the receipt")
    a = ap.parse_args(argv)
    if a.write and not a.transcripts:
        ap.error("--write needs --transcripts: a receipt alone cannot mint a pass")
    if a.transcripts and len(a.receipts) != 1:
        ap.error("--transcripts takes one receipt")

    ok = True
    for path in a.receipts:
        try:
            if a.write:
                failures = write_verdict(path, a.transcripts)
            else:
                r = _loads(Path(path).read_text(encoding="utf-8"))
                failures = check_stored(r, a.transcripts)
        except (OSError, ValueError) as exc:
            failures = [f"unreadable: {exc}"]
        ok = ok and not failures
        print(json.dumps({"receipt": os.path.basename(path), "pass": not failures,
                          "transcripts_checked": bool(a.transcripts), "failures": failures}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
