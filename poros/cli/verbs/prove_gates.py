"""`prove <model> --unvalidated`: the procedure's two GPU gates, for a family Poros has not
validated, on this machine.

1. The LOAD gate: the streamed NF4 load against the standard load, a sha256 of every tensor.
2. The TRAINING gate: streamed (block 4) against resident, one seed, G=1, dropout 0. The loss at
   every step, the adapter gradients, the adapter weights and the Adam state, ``torch.equal``.

The second runs only when the first holds. A pass writes a validation pack (``poros.validate.pack``)
to attach to an architecture request issue; a fail names the surface and the step, and writes no
pack. Either way
nothing changes a registry status: the family runs as ``user_asserted`` for this command alone.
"""

from __future__ import annotations

import re
import shlex
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field

from poros import runs
from poros.cli.schemas import Output
from poros.cli.ui import components as c
from poros.cli.ui import simulate, term
from poros.cli.ui.text import Line, S

if TYPE_CHECKING:
    from poros.cli.context import Context
    from poros.cli.verbs.prove import ProveAction

STREAMED_BLOCK = 4
_SHOWN = re.compile(r"[A-Za-z0-9_.:]{1,200}")
# internal surface -> the word a result uses
SURFACE = {"losses": "loss", "grads": "grads", "weights": "weights", "opt_state": "optimizer"}
ORDER = {"loss": 0, "grads": 1, "weights": 2, "optimizer": 3}
SAYS = ("the procedure's two GPU gates on this machine: the streamed NF4 load against the standard "
        "load, a sha256 of every tensor; then streamed (block 4) against resident from one seed, "
        "G=1, dropout 0, torch.equal on the loss at every step, the adapter gradients, weights and "
        "Adam state. It changes no status: support stays user_asserted until a maintainer "
        "commits the pack")


class LoadGate(BaseModel):
    identical: bool
    tensors_compared: int
    differs: list[str] = []  # the first tensors that differ, missing or extra
    n_differs: int = 0
    standard_peak_vram_gib: Optional[float] = None
    streamed_peak_vram_gib: Optional[float] = None
    standard_seconds: Optional[float] = None
    streamed_seconds: Optional[float] = None


class Surface(BaseModel):
    equal: bool
    compared: int  # steps for the loss, tensors for the rest
    max_abs_diff: float
    first_step: Optional[int] = None  # 1-based: the first optimizer step at which it differs
    first_tensor: Optional[str] = None


class TrainingGate(BaseModel):
    identical: bool
    steps: int
    seq_len: int
    block_size: int
    g: int
    lora_dropout: float
    seed: int
    loss: Surface
    grads: Surface
    weights: Surface
    optimizer: Surface
    # the first and last step's loss in each arm; every step's is in the pack's parity.jsonl
    losses: dict[str, Optional[float]]
    resident_peak_vram_gib: Optional[float] = None
    streamed_peak_vram_gib: Optional[float] = None
    resident_seconds: Optional[float] = None
    streamed_seconds: Optional[float] = None


class Difference(BaseModel):
    surface: Literal["load", "loss", "grads", "weights", "optimizer"]
    step: Optional[int] = None
    tensor: Optional[str] = None


class GatesOutput(Output):
    schema_: Literal["poros.prove_gates.v1"] = Field("poros.prove_gates.v1", alias="schema")
    model: str
    model_type: str
    status: str  # the registry's status for it: not-in-registry, target, ...
    support: Literal["user_asserted"] = "user_asserted"
    simulated: bool
    passed: bool
    load: LoadGate
    training: Optional[TrainingGate] = None  # None: not run, as the load differed
    differs: list[Difference] = []  # earliest first
    receipt: str
    pack: Optional[str] = None  # the folder a pass writes
    files: list[str] = []
    submit: Optional[dict[str, Any]] = None  # where the pack goes, and the registry line
    env: dict[str, Any]
    next: list[str] = []
    says: str = SAYS
    exit_code: int = Field(0, exclude=True)


def _name(tensor: Optional[str]) -> Optional[str]:
    """A tensor's name comes from this install's class: shown only if it reads like one."""
    if tensor is None:
        return None
    return tensor if _SHOWN.fullmatch(tensor) else "unknown"


def _env() -> dict[str, Any]:
    from poros.cli.verbs.prove_run import _env as base

    env = base()
    for name in ("peft", "bitsandbytes"):
        try:
            env[name] = __import__(name).__version__
        except Exception:  # absent, or broken: unknown
            pass
    return env


# -- The gates --------------------------------------------------------------------------------


def _simulated_load(a: "ProveAction") -> dict[str, Any]:
    return {"model": a.model, "dtype": "bfloat16", "compare_mode": "sha256-digest",
            "gpu": simulate.GPU[0], "torch": "simulated", "tensors_compared": 0,
            "missing_in_streamed": [], "extra_in_streamed": [], "mismatched": [],
            "n_missing": 0, "n_extra": 0, "n_mismatched": 0, "byte_identical": True,
            "standard_load_peak_vram_gb": None, "streamed_load_peak_vram_gb": None,
            "standard_load_seconds": 0.0, "streamed_load_seconds": 0.0}


def _simulated_training(a: "ProveAction") -> dict[str, Any]:
    losses = [round(10.0 / (1 + 0.05 * i), 6) for i in range(a.steps)]
    return {"model": a.model, "quantization": "nf4", "load_strategy": "gpu",
            "block_size": STREAMED_BLOCK, "g": 1, "steps": a.steps, "seq_len": a.seq_len,
            "block_major": False, "lora_dropout": 0.0, "seed": 1234,
            "resident_peak_vram_gb": None, "blockwise_peak_vram_gb": None,
            "losses_equal": True, "loss_max_abs_diff": 0.0, "resident_losses": losses,
            "blockwise_losses": list(losses), "grads_equal": True, "grads_count": 0,
            "grads_max_abs_diff": 0.0, "weights_equal": True, "weights_count": 0,
            "weights_max_abs_diff": 0.0, "opt_state_equal": True, "opt_state_count": 0,
            "opt_state_max_abs_diff": 0.0, "resident_seconds": 0.0, "blockwise_seconds": 0.0,
            "blockwise_h2d_bytes_total": None, "all_equal": True,
            "first_difference": {k: None for k in SURFACE}}


def _load_gate(a: "ProveAction") -> dict[str, Any]:
    if simulate.enabled():
        return _simulated_load(a)
    from poros.validate import load_gate

    return load_gate.run(a.model)


def _training_gate(a: "ProveAction") -> dict[str, Any]:
    if simulate.enabled():
        return _simulated_training(a)
    from poros.validate import training_gate

    return training_gate.run(a.model, steps=a.steps, seq_len=a.seq_len,
                             block_size=STREAMED_BLOCK, g=1, lora_dropout=0.0)


# -- From records to a result -----------------------------------------------------------------


def load_result(record: dict[str, Any]) -> LoadGate:
    named = [*record.get("mismatched", []), *record.get("missing_in_streamed", []),
             *record.get("extra_in_streamed", [])]
    return LoadGate(
        identical=bool(record["byte_identical"]),
        tensors_compared=int(record["tensors_compared"]),
        differs=[_name(x) or "unknown" for x in named[:10]],
        n_differs=int(record["n_mismatched"] + record["n_missing"] + record["n_extra"]),
        standard_peak_vram_gib=record.get("standard_load_peak_vram_gb"),
        streamed_peak_vram_gib=record.get("streamed_load_peak_vram_gb"),
        standard_seconds=record.get("standard_load_seconds"),
        streamed_seconds=record.get("streamed_load_seconds"),
    )


def training_result(record: dict[str, Any]) -> tuple[TrainingGate, list[Difference]]:
    first = record.get("first_difference") or {}
    surfaces: dict[str, Surface] = {}
    differs: list[Difference] = []
    for key, word in SURFACE.items():
        prefix = "loss" if key == "losses" else key
        equal = bool(record[f"{key}_equal"])
        at = first.get(key) or {}
        count = record["steps"] if key == "losses" else int(record[f"{key}_count"])
        surfaces[word] = Surface(equal=equal, compared=count,
                                 max_abs_diff=float(record[f"{prefix}_max_abs_diff"]),
                                 first_step=at.get("step"), first_tensor=_name(at.get("tensor")))
        if not equal or at:
            differs.append(Difference(surface=word, step=at.get("step"),
                                      tensor=_name(at.get("tensor"))))
    differs.sort(key=lambda d: (d.step if d.step is not None else 1 << 30, ORDER[d.surface]))
    gate = TrainingGate(
        identical=bool(record["all_equal"]) and not differs, steps=record["steps"],
        seq_len=record["seq_len"], block_size=record["block_size"], g=record["g"],
        lora_dropout=record["lora_dropout"], seed=record.get("seed", 1234), **surfaces,
        losses={f"{arm}_{end}": (record[key][i] if record[key] else None)
                for arm, key in (("resident", "resident_losses"), ("streamed", "blockwise_losses"))
                for end, i in (("first", 0), ("last", -1))},
        resident_peak_vram_gib=record.get("resident_peak_vram_gb"),
        streamed_peak_vram_gib=record.get("blockwise_peak_vram_gb"),
        resident_seconds=record.get("resident_seconds"),
        streamed_seconds=record.get("blockwise_seconds"),
    )
    return gate, differs


def run(a: "ProveAction", ctx: "Context") -> GatesOutput:
    from poros.core.arch_registry import assert_this_command
    from poros.validate import pack

    kind = a.unvalidated or "unknown"
    assert_this_command(kind)  # resolve asserted it already; a run from a sheet says it again
    if not simulate.enabled():
        try:
            import torch

            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        if not gpu:
            from poros.core.refusal import Refusal

            raise Refusal("E_NO_CUDA", "No CUDA GPU", "the gates run on a CUDA GPU, and none "
                          "is here", context=["arch try checks the CUDA route on the CPU"])
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    with term.get().activity("Load gate: the standard load, then the streamed one") as act:
        load_record = _load_gate(a)
        act.done([c.event("Loaded", "twice", c.duration(act.elapsed))])
    load = load_result(load_record)
    training, training_record, differs = None, None, []
    if not load.identical:
        differs = [Difference(surface="load", tensor=(load.differs or [None])[0])]
    else:
        from poros.cli.verbs.prove_run import _free_gpu

        _free_gpu()
        with term.get().activity(f"Training gate: {a.steps} steps resident, then streamed") as act:
            training_record = _training_gate(a)
            act.done([c.event("Trained", "both arms", c.duration(act.elapsed))])
        training, differs = training_result(training_record)
    passed = load.identical and training is not None and training.identical and not differs
    out = GatesOutput(model=a.model, model_type=kind, status=a.family_status or "not-in-registry",
                      simulated=simulate.enabled(), passed=passed, load=load, training=training,
                      differs=differs, receipt="", env=_env())
    for n in range(1, 100):  # one name for the receipt and the pack, never one already there
        name = stamp if n == 1 else f"{stamp}-{n}"
        receipt = runs.home() / "receipts" / f"prove-gates-{name}.json"
        folder = runs.home() / "validation" / f"{kind}-{name}"
        if not (receipt.exists() or folder.exists()):
            break
    receipt.parent.mkdir(parents=True, exist_ok=True)
    out.receipt = str(receipt)
    if passed:
        out.pack = str(folder)
        record = out.model_dump(mode="json", by_alias=True)
        out.files = pack.write(folder, record, load_record, training_record or {})
        line = pack.registry_line(kind, a.model, pack.clean_env(out.env))
        out.submit = {"to": pack.folder_for(kind), "files": out.files, "registry_line": line,
                      "into": pack.REGISTRY_FILE, "steps": a.steps,
                      "steps_the_procedure_asks": pack.SUBMIT_STEPS}
        if a.steps < pack.SUBMIT_STEPS:
            out.next = [shlex.join(["poros", "prove", a.model, "--steps",
                                    str(pack.SUBMIT_STEPS), "--seq-len", str(a.seq_len)])]
        else:
            out.next = [f"open an architecture request issue on the Poros GitHub repo and attach "
                        f"the files in {folder}"]
    else:
        out.next = [shlex.join(["poros", "arch", "try", a.model])]
    runs.write_json(receipt, out.model_dump(mode="json", by_alias=True))
    out.exit_code = 0 if passed else 1
    return out


# -- The screen -------------------------------------------------------------------------------


def view(r: GatesOutput, w: int) -> list[Line]:
    ww = max(40, min(w, 100))
    gpu = (r.env.get("gpu") or "this machine").removesuffix(" (simulated)")
    if r.passed:
        out = [c.event("Gates passed", r.model_type, f"on {gpu}" + (" · simulated"
                                                                      if r.simulated else ""))]
    else:
        first = r.differs[0] if r.differs else None
        where = (f"{first.surface}" + (f" at step {first.step}" if first and first.step else "")
                 if first else "a gate")
        out = c.error_event("Not identical", f"{where} differs", ww)
    out.append(c.rail(S(r.model, "hi"), S(f"  ·  model_type {r.model_type}, not validated", "dm")))
    out.append(c.rail())
    load = r.load
    said = (f"byte-identical, {load.tensors_compared} tensors" if load.identical else
            f"{load.n_differs} tensors differ")
    out.append(c.rail(S("load".ljust(10), "dm"), S(said, "a" if load.identical else "e")))
    if r.training is None:
        out.append(c.rail(S("training".ljust(10), "dm"), S("not run: the load differs", "dm")))
    else:
        t = r.training
        for word in ("loss", "grads", "weights", "optimizer"):
            s = getattr(t, word)
            text = ("equal" + (f" at all {s.compared} steps" if word == "loss" else
                               f", {s.compared} tensors") if s.equal else
                    f"differs from step {s.first_step}" + (f", {s.first_tensor}"
                                                           if s.first_tensor else ""))
            out.append(c.rail(S(word.ljust(10), "dm"), S(text, "a" if s.equal else "e")))
        out.append(c.rail(S(" " * 10), S(f"{t.steps} steps · length {t.seq_len} · block "
                                         f"{t.block_size} against resident · G=1 · dropout 0",
                                         "dm")))
    out.append(c.rail())
    if r.pack:
        out.append(c.rail(S("pack".ljust(10), "dm"), S(r.pack, "tx")))
    out.append(c.rail(S("receipt".ljust(10), "dm"), S(r.receipt, "dm")))
    if r.next:
        out.append([])
        nxt = r.next[0]
        out += (c.continue_with(nxt.removeprefix("poros "), ww) if nxt.startswith("poros ")
                else [c.rail(S(nxt, "tx"))])
    return out
