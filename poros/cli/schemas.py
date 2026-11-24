"""Pydantic models for every ``--json`` result (the ``poros.<name>.v1`` contract).

Each output's JSON Schema is generated from the class that produces it, so the
published schema (``poros schema <name>``) cannot drift from the output.
Versioning: adding a field keeps ``v1``; removing, renaming or retyping one
is ``v2``. Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_serializer

# The plan's models are the library's (poros.core.schemas): published here with every result's.
from poros.core.schemas import (  # noqa: F401
    Figure,
    Output,
    PlanBound,
    PlanGpu,
    PlanNearest,
    PlanOutput,
    PlanStore,
)


class ErrorBody(BaseModel):
    """An error: its stable code, the exit code, and what happened. The three below are said
    only where there is something to say: an empty one is left out."""

    code: str
    exit_code: int
    message: str
    # The next step in one line: a command to copy where one does it, else a sentence.
    hint: str = ""
    # Machine-applicable fix: config field -> suggested value.
    remediation: dict[str, Any] = {}
    # Where the docs say more: `poros docs <page> [<section words>]`, offline from any install.
    docs: str = ""

    @model_serializer(mode="wrap")
    def _said(self, handler):  # no return type: one would replace the fields in its JSON Schema
        return {k: v for k, v in handler(self).items()
                if k not in ("hint", "remediation", "docs") or v not in ("", {})}


class ErrorOutput(Output):
    schema_: Literal["poros.error.v1"] = Field("poros.error.v1", alias="schema")
    error: ErrorBody


class DoctorCheck(BaseModel):
    name: str
    value: str
    status: Literal["ok", "warn", "error", "info"]


class CheckOutput(Output):
    schema_: Literal["poros.check.v1"] = Field("poros.check.v1", alias="schema")
    model: str
    model_type: Optional[str]
    status: Literal[
        "validated",
        "user_asserted",
        "target",
        "unsupported",
        "not-in-registry",
        "could-not-resolve",
        "not-on-this-device",
    ]
    entry: Optional[Literal["dense", "moe"]] = None
    reason: str = ""
    next: list[str] = []  # what to run next, where there is a step: [] where there is none


class InitOutput(Output):
    schema_: Literal["poros.init.v1"] = Field("poros.init.v1", alias="schema")
    path: str
    model: str
    dataset: str
    output_dir: str


class TrainOutput(Output):
    schema_: Literal["poros.train.v1"] = Field("poros.train.v1", alias="schema")
    learned: Optional[str] = None  # `poros agents`: its held-out loss in words
    output_dir: str
    events: str
    metrics: dict[str, Any]
    run: Optional[str] = None  # the run id (`poros runs <id>`), for the train verb
    command: Optional[str] = None  # its canonical command
    next: list[str] = []  # `poros agents`: what to do with it
    plan: Optional[dict[str, Any]] = None  # `poros agents`: what it trained at
    from_: Optional[str] = Field(None, alias="from")  # --from: the run it started from
    # --unvalidated: user_asserted, and the model_type: a family Poros has not validated
    support: Optional[str] = None
    model_type: Optional[str] = None


class EvalSide(BaseModel):
    """One pass of `poros eval <run>` over its rows: the base (the adapter off) or the run's."""

    loss: Optional[float] = None  # chats, text: the mean loss; pairs: DPO's loss against the base
    ends: Optional[int] = None  # chats: rows where the end-of-turn token is the top choice at
    end_p: Optional[float] = None  # the end of every answer, and its mean probability there
    prefers: Optional[int] = None  # pairs: the chosen answer scores over the rejected one
    margin: Optional[float] = None  # pairs: mean log-probability of chosen less rejected


class EvalOutput(Output):
    schema_: Literal["poros.eval.v1"] = Field("poros.eval.v1", alias="schema")
    adapter: str
    dataset: str
    metrics: dict[str, Any]
    # `poros eval <run>`: the run's own rows through its own streamed forward, adapter off and on
    run: Optional[str] = None
    checkpoint: Optional[str] = None  # poros://<run>/weights/<step|final>
    shape: Optional[str] = None  # chats, text or pairs
    rows_from: Optional[Literal["held_out", "eval_file", "trained", "file"]] = None
    rows: Optional[int] = None  # how many it scored
    of: Optional[int] = None  # how many there were, before --rows
    base: Optional[EvalSide] = None
    trained: Optional[EvalSide] = None
    change: Optional[float] = None  # trained loss against the base's: -0.3 is 30% lower
    improved: Optional[int] = None  # rows whose loss fell; pairs whose margin grew over the base's
    turn_end: Optional[int] = None  # the end-of-turn token's id, when the template has one
    answer_rows: Optional[int] = None  # rows with an answer end inside the length: what ends counts
    verdict: Optional[str] = None
    warning: Optional[str] = None  # it does not end its answers
    next: list[str] = []
    simulated: bool = False
    support: Optional[str] = None
    model_type: Optional[str] = None
    # --show: rows' words, for the terminal alone (EvalSample); never in --json, as nothing a file
    # says may reach an agent
    samples: list[Any] = Field([], exclude=True)


@dataclass
class EvalSample:
    """One row's prompt, the answer its file gives, and what the base and the run answer, each
    plain text: no control or format character, one line."""

    prompt: str
    expected: str
    base: Optional[str] = None  # None: no model ran (a preview)
    trained: Optional[str] = None


class ExportOutput(Output):
    schema_: Literal["poros.export.v1"] = Field("poros.export.v1", alias="schema")
    adapter: str
    output: str
    format: str


class ReportOutput(Output):
    schema_: Literal["poros.report.v1"] = Field("poros.report.v1", alias="schema")
    source: str
    artifacts: int
    written: list[str]


class LeaderboardRow(BaseModel):
    method: str
    model: str
    precision: str
    vram_peak_mb: float
    s_per_step: Optional[float]
    parity: Optional[float]
    bitwise: Optional[bool]


class LeaderboardOutput(Output):
    schema_: Literal["poros.leaderboard.v1"] = Field("poros.leaderboard.v1", alias="schema")
    rows: list[LeaderboardRow]


class JobOutput(Output):
    """A background training job (`poros train --detach`, `poros jobs ...`)."""

    schema_: Literal["poros.job.v1"] = Field("poros.job.v1", alias="schema")
    job_id: str
    state: Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled"]
    model: str
    name: Optional[str] = None  # the run's name, which `runs` finds it by too
    job_dir: str
    config: str  # the job's copy, with output_dir pointed at output_dir below
    output_dir: str  # checkpoints, the final adapter and events.jsonl
    log: str
    # the CUDA_VISIBLE_DEVICES entry the job runs on (PCI bus order); apple-<hash> on a Mac
    gpu: str
    attempt: int
    queued_at: str  # this attempt's place in the queue: a continue goes to its back
    created_at: Optional[str] = None  # when it was first queued: a continue keeps it
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None  # poros.train.v1 on success
    error: Optional[ErrorBody] = None
    next: list[str] = []  # just queued: the command that watches it
    plan: Optional[dict[str, Any]] = None  # `poros agents`, just queued: what it will train at
    queued_behind: Optional[str] = None  # queued: the run it waits for on its GPU
    from_: Optional[str] = Field(None, alias="from")  # --from: the run it started from
    # --unvalidated: user_asserted, and the model_type: a family Poros has not validated
    support: Optional[str] = None
    model_type: Optional[str] = None


class JobsOutput(Output):
    schema_: Literal["poros.jobs.v1"] = Field("poros.jobs.v1", alias="schema")
    jobs: list[JobOutput]


# name -> model, for `poros schema <name>` (config/events/artifact are the
# pre-existing data models, registered lazily in the CLI).
class Choice(BaseModel):
    """One answer to a Question. ``command`` is the full command that picks it."""

    value: str
    label: str
    note: str = ""
    command: str = ""


class Question(BaseModel):
    """Something Poros needs to know (the CLI interface design, section 6).

    ``id`` is the field it sets (``model``, ``data``...), or what a confirmation
    confirms (``unvalidated``, ``download``...): unique among a result's questions.
    ``default`` is the choice a person gets on enter; a confirmation has none.
    ``flag`` is how an answer is written in a command (``--data``), or
    ``subject`` when the answer replaces the subject words.
    """

    id: str
    prompt: str
    choices: list[Choice]
    default: Optional[str] = None
    remember: Optional[str] = None  # the preference a terminal answer can be saved as
    # How an answer is written into the command; internal, the choices carry it.
    flag: str = Field("subject", exclude=True)
    # A confirmation: one choice, an override flag, no default (questions.confirmation). Internal.
    confirm: bool = Field(False, exclude=True)


class NeedsInputOutput(Output):
    """A command that cannot go on without an answer. Exit 2, E_NEEDS_INPUT.

    Run one choice's ``command`` to answer. No command adds an override flag
    (--yes, --force...) the original command did not have.
    """

    schema_: Literal["poros.needs_input.v1"] = Field("poros.needs_input.v1", alias="schema")
    verb: str
    questions: list[Question]


class VersionOutput(Output):
    schema_: Literal["poros.version.v1"] = Field("poros.version.v1", alias="schema")
    version: str


class CheckpointRow(BaseModel):
    path: str  # poros://<run-id>/weights/<step>, or .../final
    run: str
    model: str
    type: Literal["train", "adapter"]  # a run continues from train; adapter is its finished weights
    step: Optional[int]
    loss: Optional[float]
    size_bytes: int
    created: str  # ISO 8601, UTC
    files: list[str] = []  # info: what it holds
    # a finished run's last step and its final adapter are one moment: each names the other's path
    same_as: Optional[str] = None
    # its held-out score, as the run wrote it then: the loss (a pairs run's, DPO's) on eval_rows
    # rows; on chats the rows whose answers all end on the end-of-turn token, of the rows with an
    # end to read; on pairs the pairs whose chosen answer scores over the rejected one
    eval_loss: Optional[float] = None
    eval_rows: Optional[int] = None
    eval_ends: Optional[int] = None
    eval_answer_rows: Optional[int] = None
    eval_prefers: Optional[int] = None
    best: bool = False  # the lowest held-out loss the run keeps; .../weights/best names it


class CheckpointRunSize(BaseModel):
    run: str
    checkpoints: int
    size_bytes: int


class CheckpointsOutput(Output):
    schema_: Literal["poros.checkpoints.v1"] = Field("poros.checkpoints.v1", alias="schema")
    do: Literal["list", "info", "download", "delete", "storage"]
    checkpoints: list[CheckpointRow] = []
    more: int = 0  # list: how many are past --limit
    output: Optional[str] = None  # download: the folder it is in
    runs: list[CheckpointRunSize] = []  # storage: a row per run
    total_bytes: Optional[int] = None  # storage, and what a delete freed
    warnings: list[str] = []  # delete: a stopped run's newest checkpoint gone


OUTPUTS: dict[str, type[Output]] = {
    "error": ErrorOutput,
    "check": CheckOutput,
    "init": InitOutput,
    "train": TrainOutput,
    "eval": EvalOutput,
    "export": ExportOutput,
    "report": ReportOutput,
    "leaderboard": LeaderboardOutput,
    "job": JobOutput,
    "jobs": JobsOutput,
    "needs_input": NeedsInputOutput,
    "version": VersionOutput,
    "checkpoints": CheckpointsOutput,
}

# A verb's ``--json`` result model by schema name, where it is defined (``poros.cli.verbs.<module>:
# <class>``), imported only when asked for. ``poros schema`` and ``poros agents schema`` read
# this one table with OUTPUTS. A name in both is the verb's model, the fuller one (export: the
# folder form's fields, and the run's).
RESULTS: dict[str, str] = {
    "runs": "runs:RunsOutput", "run_live": "runs_live:RunLiveOutput",
    "machine": "machine:MachineOutput", "models": "models:ModelsOutput",
    "data": "data:DataOutput", "export": "export:ExportOut", "recipes": "recipes:RecipesOutput",
    "arch": "arch:ArchOutput", "prove": "prove:ProveOutput",
    "prove_gates": "prove_gates:GatesOutput", "docs": "docs:DocsOutput",
}


def schema_names() -> list[str]:
    """Every name ``poros schema`` and ``poros agents schema`` print a schema for."""
    return ["config", "events", "artifact",
            *sorted(OUTPUTS.keys() | RESULTS.keys())]


def result_schema(name: str) -> Optional[dict]:
    """The JSON Schema of the ``--json`` result ``name`` (``$id`` ``poros.<name>.v1``); None for
    a name that is no result's. Its ``required`` is what every result carries: a field outside
    it is one `poros agents` leaves out when it is not set (poros.cli.contract.published)."""
    import importlib

    from poros.cli.contract import published

    if name in RESULTS:
        module, cls = RESULTS[name].split(":")
        model = getattr(importlib.import_module(f"poros.cli.verbs.{module}"), cls)
    elif name in OUTPUTS:
        model = OUTPUTS[name]
    else:
        return None
    return published({"$id": f"poros.{name}.v1",
                      **model.model_json_schema(mode="serialization", by_alias=True)})
