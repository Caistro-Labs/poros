"""Actions: what a sentence resolved to, with where each value came from.

An Action is plain typed fields (exactly what its verb consumes) plus
``provenance``, a ``Source`` per field. Its canonical command is the
flags-only form that reproduces it anywhere.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import ClassVar, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

from poros.cli.schemas import Question


class Source(BaseModel):
    kind: Literal["given", "inferred", "preference", "default"]
    detail: str = ""  # the words that said it, or what it was inferred from

    def __str__(self) -> str:
        if self.kind == "given":
            if self.detail.startswith("chosen"):  # an answer to a Question
                return self.detail
            return "given"
        if self.kind == "inferred":
            return f"inferred · {self.detail}" if self.detail else "inferred"
        return self.kind


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verb: ClassVar[str] = ""
    provenance: dict[str, Source] = {}

    def canonical(self) -> list[str]:
        """argv after ``poros``: the verb, its subject, then flags in a fixed order."""
        raise NotImplementedError

    def command(self, *, json: bool = False) -> str:
        return shlex.join(["poros", *self.canonical(), *(["--json"] if json else [])])

    def given(self, name: str, said: str = "") -> None:
        self.provenance[name] = Source(kind="given", detail=said)

    def inferred(self, name: str, why: str) -> None:
        self.provenance[name] = Source(kind="inferred", detail=why)

    def default(self, name: str) -> None:
        self.provenance[name] = Source(kind="default")


A = TypeVar("A", bound=Action)


@dataclass
class Draft(Generic[A]):
    """A resolve step's answer: the action so far, and what it still needs."""

    action: Optional[A]
    questions: list[Question] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.action is not None and not self.questions
