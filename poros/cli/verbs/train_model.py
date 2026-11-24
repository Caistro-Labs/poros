"""The model a train command names, before anything is offered for it: that it is there to
train (``require_model``), and what its family is where Poros has not validated it
(``new_family``, ``unvalidated_questions``). ``poros.cli.verbs.train`` resolves with these, and
prove reads a new family the same way."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from poros.cli import questions
from poros.cli.context import Context
from poros.cli.schemas import Choice, Question
from poros.cli.words import models as model_words
from poros.preflight import command

if TYPE_CHECKING:
    from poros.cli.verbs.train import TrainAction


def require_model(model: str, ctx: Context) -> None:
    """A model the catalog does not name is asked for before anything is offered for it (a
    consent to train it unvalidated, a download): a Hub id that is not there, is gated, or
    cannot be reached is refused here (``poros.preflight.command.model_refusal``). A preview
    and a Mac leave "not cached, and no Hub" to their own checks."""
    if model_words.by_id(model) is not None:  # the catalog names what is on the Hub
        return
    from poros.cli.ui import simulate

    gone = command.model_refusal(model, cache_decides=not (simulate.enabled() or ctx.unified))
    if gone is not None:
        raise gone


def new_family(model: str, ctx: Context, told: bool = False):
    """A model whose family Poros has not validated: arch's reading of it, or None for one the
    catalog or the registry vouches for. What no flag trains is refused here, in plain words. A
    family it trains runs as ``user_asserted`` for this command, so a plan or a Mac's table reads
    it as the env's assertion would; the run itself waits on --unvalidated (``confirmations``)."""
    if model_words.by_id(model) is not None:  # the catalog names validated families only
        return None
    from poros.cli.verbs import arch
    from poros.core.arch_registry import assert_this_command, status_outside_this_command

    seen = arch.reading(model, ctx)
    if seen.status == "validated":
        return None
    if seen.status == "unread" and ctx.unified:  # a Mac: no Hub, and in no cache of its own
        from poros.cli.verbs.mac import unread

        if (gone := unread(model)) is not None:
            raise gone
    if seen.status in ("gated", "unread"):  # what it is can't be read here: train's gate decides
        if told:
            raise command.unnamed_family(model, seen.refused)
        return None
    if seen.refused:
        raise command.not_trainable(model, seen.refused, seen.model_type)
    if status_outside_this_command(seen.model_type) == "user_asserted":  # the env's, or a plugin's
        seen = seen._replace(status="user_asserted")
    assert_this_command(seen.model_type)
    return seen


def unvalidated_questions(a: "TrainAction") -> list[Question]:
    """The one rule for a family Poros has not validated: it trains with --unvalidated, and arch
    try checks its math first."""
    from poros.cli import contract

    said = {"not-in-registry": "not in Poros's registry",
            "target": "planned, not validated"}.get(a.family_status, a.family_status)
    confirm = questions.confirmation(
        "unvalidated",
        f"{a.model} is model_type {a.family} (status {a.family_status}: {said}). Poros has not "
        "validated this family, so it trains with no parity guarantee, and every result about "
        "the run says so.",
        "--unvalidated", "Train it unvalidated", "this run only")
    check = ["poros", "arch", "try", a.model, *(["--json"] if contract.json_mode() else [])]
    return [confirm, Question(
        id="check", prompt="Or first check its math on a tiny random copy of it.", flag="",
        choices=[Choice(value="arch try", label="Check it first",
                        note="on the CPU in seconds; nothing downloads but its config",
                        command=shlex.join(check))])]
