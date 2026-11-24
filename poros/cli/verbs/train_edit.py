"""Change a setting on the train confirm screen: pick a field, then a value.

Every value offered is valid, and the one the run takes now is marked,
so customizing a run never needs a flag. The answer goes back into the
sentence as a flag and the screen is drawn again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poros.cli.schemas import Choice, Question
from poros.cli.verbs import mac, shared
from poros.cli.verbs import train_data as data_scan
from poros.cli.words import models as model_words

if TYPE_CHECKING:
    from poros.cli.context import Context
    from poros.cli.verbs.train import TrainAction

FIELDS = (
    ("steps", "Steps"),
    ("seq_len", "Length"),
    ("rank", "LoRA rank"),
    ("data", "Data"),
    ("model", "Model"),
)


def field_question(a: "TrainAction") -> Question:
    now = {
        "steps": f"{a.steps:,}",
        "seq_len": str(a.seq_len),
        "rank": str(a.rank),
        "data": a.data,
        "model": a.model,
    }
    return Question(
        id="field",
        prompt="Change which setting?",
        default="steps",
        flag="",
        choices=[Choice(value=k, label=label, note=now[k]) for k, label in FIELDS],
    )


def value_question(field: str, a: "TrainAction", ctx: "Context") -> Question:
    if field == "steps":
        options = sorted({100, 300, 1000, 2000, a.steps})
        return Question(
            id="steps",
            prompt="How many steps?",
            flag="--steps",
            default=str(a.steps),
            choices=[
                Choice(value=str(n), label=f"{n:,}", note="now" if n == a.steps else "")
                for n in options
            ],
        )
    if field in ("seq_len", "rank"):
        options = (256, 512, 1024, 2048, 4096) if field == "seq_len" else (8, 16, 32, 64)
        if field == "seq_len" and ctx.unified:
            options = sorted({*mac.lengths(a.model, ctx), a.seq_len})
        now = a.seq_len if field == "seq_len" else a.rank

        def note(n: int) -> str:  # never the plan's verdict: the check says it at the start
            return "now" if n == now else ""

        prompt = "Tokens per example?" if field == "seq_len" else "LoRA rank?"
        flag = "--seq-len" if field == "seq_len" else "--rank"
        return Question(
            id=field,
            prompt=prompt,
            flag=flag,
            default=str(now),
            choices=[Choice(value=str(n), label=str(n), note=note(n)) for n in options],
        )
    if field == "data":
        files = data_scan.local_files(ctx.cwd)[:8]
        choices = [Choice(value=f, label=f) for f in files]
        return Question(
            id="data", prompt="Train on what data?", flag="--data", default=a.data, choices=choices
        )
    me = model_words.by_id(a.model)
    family = [e for e in mac.trainable(ctx) if me is None or e.family == me.family]
    return Question(
        id="model",
        prompt="Which model?",
        flag="subject",
        default=a.model,
        choices=[
            Choice(value=e.id, label=e.label, note=e.size.upper())  # never the plan's verdict
            for e in sorted(family, key=shared.size_of)
        ],
    )
