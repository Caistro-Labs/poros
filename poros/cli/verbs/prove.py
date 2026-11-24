"""prove: show on this machine that streaming changes nothing.

    poros prove
    poros prove qwen2.5-0.5b
    poros prove ~/.poros/receipts/prove-1022-7f3a.json
    poros prove HuggingFaceTB/SmolLM2-135M-Instruct --unvalidated

Flags:
    --steps N         steps per run (default: a short run)
    --seq-len N       tokens per row
    --download        yes to downloading the model
    --force           run both arms of a large model without asking first
    --block-sizes     compare blocks of 4 with one block of every layer, both
                      streamed (no resident reference; for a model the card
                      cannot hold whole)
    --unvalidated     run the GPU gates for a family Poros has not validated

A short fine-tune of a small validated model, twice from the same seed:
streamed (blocks of 4 layers) and with every layer held on the GPU for the
whole run, trained by ordinary autograd. The loss at every step, the adapter
gradients, the adapter weights and the optimizer state must be identical,
bit for bit. A model whose resident run the card cannot hold is said before
anything runs. Writes a receipt that names its reference; `prove <receipt>`
re-runs one and says identical, or which pinned item (GPU, driver, torch,
transformers) differs.

A model whose family Poros has not validated asks first, and `--unvalidated`
runs the two GPU gates a new family passes before anyone may call it
validated: the streamed load byte for byte against the standard load, then
streamed against resident training, bit for bit at every step. A pass writes
a validation pack to submit. Nothing here changes a family's status.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import Output
from poros.cli.ui.text import Line
from poros.cli.verbs import mac, shared
from poros.cli.words import models as model_words
from poros.cli.words.grammar import Sentence
from poros.core.refusal import Refusal
from poros.recipes.rules import within

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
STEPS, SEQ = 20, 256
MOST_STEPS = 2000  # what a receipt may ask for


class ProveAction(Action):
    verb: ClassVar[str] = "prove"
    model: str = DEFAULT_MODEL
    steps: int = STEPS
    seq_len: int = SEQ
    receipt: Optional[str] = None  # re-run this receipt
    force: bool = False  # --force: the resident arm of a large model, confirmed
    # what the streamed arm is compared with: every layer resident under ordinary autograd, or
    # (--block-sizes) one block of every layer, streamed too
    reference: Literal["resident", "block_sizes"] = "resident"
    download: bool = False  # --download: a large or unknown download, confirmed
    # A family Poros has not validated: its model_type and status, and --unvalidated's assertion
    family: Optional[str] = None
    family_status: Optional[str] = None
    unvalidated: Optional[str] = None

    def canonical(self) -> list[str]:
        if self.receipt:
            return ["prove", self.receipt]
        out = ["prove", self.model, "--steps", str(self.steps), "--seq-len", str(self.seq_len)]
        flags = (("--force", self.force), ("--download", self.download),
                 ("--block-sizes", self.reference == "block_sizes"),
                 ("--unvalidated", bool(self.unvalidated)))
        return out + [f for f, on in flags if on]


class Arm(BaseModel):
    block_size: int
    peak_vram_gib: Optional[float]
    s_per_step: Optional[float]
    final_loss: Optional[float]


class ProveOutput(Output):
    schema_: Literal["poros.prove.v1"] = Field("poros.prove.v1", alias="schema")
    model: str
    steps: int
    seq_len: int
    simulated: bool
    # CUDA's reference, said by the run that made it: resident (every layer on the GPU for the
    # whole run, ordinary autograd) or block_sizes (one block of every layer, streamed as the other
    # arm is). None on a Mac, whose receipt says its arms by ``method`` (ram-vs-ssd, ssd-twice)
    reference: Optional[Literal["resident", "block_sizes"]] = None
    streamed: Arm
    resident: Optional[Arm] = None  # reference resident
    one_block: Optional[Arm] = None  # reference block_sizes
    identical: bool
    compared: list[str]  # what was compared: losses, each adapter tensor
    differs: list[str] = []  # tensors (or "losses") that differ
    receipt: str
    env: dict[str, Any]
    against: Optional[str] = None  # the receipt re-run, if any
    reproduced: Optional[bool] = None  # matches that receipt's hashes
    pinned_differs: dict[str, list[Any]] = {}  # env item -> [receipt, here]
    # the built-in recipe certified on this machine whose model is the largest (certified.json,
    # as `poros recipes` reads it); null where none is certified here
    largest_certified: Optional[str] = None
    exit_code: int = Field(0, exclude=True)


class MacProveOutput(ProveOutput):
    """On a Mac ``resident`` is the RAM source and ``streamed`` the SSD source, in separate
    processes, and the digests go through ``poros.metal.verify``."""

    method: Literal["ram-vs-ssd", "ssd-twice"]
    witness: dict[str, int]
    machine: str


def _a_covered_model(model: str, ctx: Context) -> None:
    entry = model_words.by_id(model)
    mac.mac_gate(model, ctx)
    if not ctx.unified and (
        entry is None or entry.evidence not in ("validated", "measured") or entry.entry != "dense"
    ):
        raise Refusal(
            "E_USAGE",
            "Pick a validated dense model",
            f"prove runs a model a parity receipt covers; {model} is not one",
            fix="poros models",
            note="'validated', kind dense",
        )


def _new_family(model: str, ctx: Context, told: bool):
    """A family Poros has not validated, as `train` reads it, or None for a model plain `prove`
    takes or refuses. What no flag trains is refused here, as `train` refuses it."""
    if ctx.unified:
        if told:
            raise Refusal("E_GUARANTEE_UNAVAILABLE", "Not on a Mac yet",
                          "the gates for a new family run on a CUDA GPU",
                          context=["arch try checks the Mac's path on a tiny copy",
                                   "train --unvalidated trains it here, from its Hub id "
                                   "or a folder"])
        return None
    entry = model_words.by_id(model)
    if entry is not None:
        if told and entry.evidence in ("validated", "measured") and entry.entry == "dense":
            raise Refusal("E_USAGE", "Validated already",
                          f"{model} is a validated family: prove takes it as it is",
                          fix=shlex.join(["poros", "prove", model]))
        return None
    from poros.cli.verbs import train

    family = train.new_family(model, ctx, told=told)
    if family is None and told:  # a validated family the catalog does not list
        raise Refusal("E_USAGE", "Validated already",
                      f"{model}'s family is validated: a model of a validated family "
                      "inherits its parity, and the gates are for a family Poros has not "
                      "validated")
    return family


def unvalidated_questions(a: ProveAction) -> list:
    """The one rule for a family Poros has not validated: the gates run with --unvalidated, and
    arch try checks its math first."""
    from poros.cli import contract
    from poros.cli.schemas import Choice, Question

    said = {"not-in-registry": "not in Poros's registry",
            "target": "planned, not validated"}.get(a.family_status or "", a.family_status)
    from poros.cli.questions import confirmation

    confirm = confirmation(
        "unvalidated",
        f"{a.model} is model_type {a.family} (status {a.family_status}: {said}). Poros "
        "has not validated this family. prove runs the procedure's two GPU gates on this "
        f"machine: the streamed load against the standard load, byte for byte, then {a.steps} "
        "steps streamed against resident, bit for bit. A pass writes a validation pack to "
        "submit; it changes no status.",
        "--unvalidated", "Run the gates", "this command only")
    check = ["poros", "arch", "try", a.model, *(["--json"] if contract.json_mode() else [])]
    return [confirm, Question(
        id="check", prompt="Or first check its math on a tiny random copy of it.", flag="",
        choices=[Choice(value="arch try", label="Check it first",
                        note="on the CPU in seconds; nothing downloads but its config",
                        command=shlex.join(check))])]


def resolve(s: Sentence, ctx: Context) -> Draft[ProveAction]:
    a = ProveAction()
    told = bool(s.flags.get("--unvalidated"))
    if told and not s.subject:
        raise Refusal("E_USAGE", "Which model", "--unvalidated runs the gates on a model",
                      fix="poros prove <model> --unvalidated", note="a Hub id or a folder")
    if s.subject and s.subject[0].lower().endswith(".json"):
        path = Path(s.subject[0]).expanduser()
        if not path.is_file():
            raise Refusal("E_NOT_FOUND", "No receipt", f"nothing at {s.subject[0]}")
        from poros.plan.configs import read_config

        r = read_config(path)
        if told:
            raise Refusal("E_USAGE", "Not with a receipt",
                          "a receipt re-runs as it was: --unvalidated runs the gates on a "
                          "model", fix="poros prove <model> --unvalidated")
        if r is None or r.get("schema") != "poros.prove.v1":
            raise Refusal("E_USAGE", "Not a prove receipt", f"{path} is not poros.prove.v1")
        model, steps, seq = r.get("model"), r.get("steps"), r.get("seq_len")
        if not (isinstance(model, str) and type(steps) is int and 1 <= steps <= MOST_STEPS
                and within("max_seq_length", seq)):  # a file's model and sizes, as a person's are
            raise Refusal("E_USAGE", "Not a prove receipt",
                          f"{path} asks for a model or sizes prove does not take")
        shared.refuse_planted(model)
        _a_covered_model(model, ctx)
        # a receipt from before it named one compared two streamed block sizes
        ref = "resident" if r.get("reference") == "resident" else "block_sizes"
        a = ProveAction(model=model, steps=steps, seq_len=seq, receipt=str(path), reference=ref)
        a.given("receipt", s.subject[0])
        return Draft(a)
    if s.subject:
        model, said, ask = shared.resolve_model(s.subject, ctx, verb="prove")
        if ask is not None:
            return Draft(None, [ask])
        family = _new_family(model, ctx, told)
        if family is None:
            _a_covered_model(model, ctx)
        else:
            a.family, a.family_status = family.model_type, family.status
            if told:
                a.unvalidated = family.model_type
        a.model = model
        a.given("model", said)
    a.force = bool(s.flags.get("--force"))
    if s.flags.get("--block-sizes"):
        if told or ctx.unified:
            raise Refusal("E_USAGE", "Not here", "--block-sizes is plain prove's, on CUDA",
                          context=["--unvalidated runs the resident gate; a Mac compares its "
                                   "RAM and SSD sources"])
        a.reference = "block_sizes"
    a.download = bool(s.flags.get("--download") or s.flags.get("--yes"))
    for name in ("steps", "seq_len"):
        raw = s.value(name)
        if raw is not None:
            try:
                setattr(a, name, max(1, int(str(raw))))
            except ValueError:
                raise Refusal("E_USAGE", "Not a number", f"{name} {raw!r} is not a number")
    if a.family:
        if a.steps > MOST_STEPS:
            raise Refusal("E_USAGE", "Too many steps",
                          f"the gates run at most {MOST_STEPS} steps",
                          fix=shlex.join(["poros", "prove", a.model, "--steps", "200"]),
                          note="the procedure's training gate")
        if not within("max_seq_length", a.seq_len):
            raise Refusal("E_USAGE", "Not a length", f"seq_len {a.seq_len} is out of range")
    if ctx.unified:
        a.steps = max(2, a.steps)  # the check on gradients reads them after the first update
        if s.value("seq_len") is None:
            row = mac.mac_row(a.model, ctx)
            if row is None:  # the default model too: no row on this Mac is its refusal
                mac.mac_gate(a.model, ctx)
                raise Refusal("E_GUARANTEE_UNAVAILABLE", "Not on this Mac",
                              f"{a.model} has no row in this Mac's table")
            a.seq_len = row.seq
    return Draft(a)


# The resident arm holds every layer on the GPU at once: past this size no
# receipt says it fits a consumer card, so it is asked, not assumed.
RESIDENT_ASK_ABOVE_B = 4.0


def confirmations(a: ProveAction, ctx: Context) -> list:
    out = []
    if a.family and not a.unvalidated:
        out += unvalidated_questions(a)
    entry = model_words.by_id(a.model)
    disk = None
    if ctx.unified:
        from poros.metal.prove import store_for

        disk = store_for(a.model)
    if (entry is not None and shared.size_of(entry) > RESIDENT_ASK_ABOVE_B and not a.force
            and (ctx.unified or a.reference == "resident")):
        from poros.cli.questions import confirmation

        out.append(confirmation(
            "runs_twice" if ctx.unified else "resident_wont_fit",
            f"This trains {entry.short} twice, {a.steps} steps each. "
            "`poros recipes` says how long a step took on a certified Mac."
            if ctx.unified
            else f"The resident run holds all of {entry.short} on the GPU at once; "
            "no receipt measured that on this card.",
            "--force", "Run both arms anyway"))
    if not a.download:
        from poros.cli.verbs.shared import download_confirm, plan_for

        size = plan_for(a.model, ctx, download=True, store_path=disk).download.value
        from poros.cli.ui import simulate

        what = None if simulate.enabled() else download_confirm(a.model, size, ctx)
        if what:
            from poros.cli.verbs.shared import download_question

            out.append(
                download_question(a.model, what, "--download", "Yes, download and prove", disk)
            )
    return out


def run(a: ProveAction, ctx: Context) -> ProveOutput:
    if a.unvalidated:
        from poros.cli.verbs import prove_gates

        return prove_gates.run(a, ctx)  # type: ignore[return-value]
    from poros.cli.ui import simulate

    if ctx.unified and not simulate.enabled():  # a preview trains nothing, on a Mac as on CUDA
        from poros.metal import prove as mac_prove

        return mac_prove.run(a, ctx)
    from poros.cli.verbs import prove_run

    return prove_run.run(a, ctx)


def view(r: ProveOutput, w: int) -> list[Line]:
    from poros.cli.verbs import prove_gates

    if isinstance(r, prove_gates.GatesOutput):
        return prove_gates.view(r, w)
    if isinstance(r, MacProveOutput):
        from poros.metal import prove as mac_prove

        return mac_prove.view(r, w)
    from poros.cli import views

    return views.proved(r, w)
