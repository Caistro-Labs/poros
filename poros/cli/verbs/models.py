"""models: the catalog, and the recipes certified for each on this machine. Download one.

    poros models
    poros models get qwen2.5-7b-instruct

Flags:
    --yes             yes to the download

A download shows its size and destination first, and every one is asked (--yes answers it).
No model's memory or speed is shown here: what the release measured is a certified recipe's
(poros recipes), and poros train checks a run before it downloads anything.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from poros.cli.action import Action, Draft, Source
from poros.cli.context import Context
from poros.cli.schemas import Output, PlanGpu, Question
from poros.cli.ui.text import Line
from poros.cli.verbs import mac, shared
from poros.cli.words import models as model_words
from poros.cli.words.grammar import Sentence

CONFIRM_ABOVE_GIB = 5.0


class ModelsAction(Action):
    verb: ClassVar[str] = "models"
    get: Optional[str] = None  # a model to download
    yes: bool = False
    disk: Optional[str] = Field(None, exclude=True)  # on a Mac: where it goes
    size_gib: Optional[float] = Field(None, exclude=True)  # inspect: bytes still to fetch

    def canonical(self) -> list[str]:
        if self.get is None:
            return ["models"]
        where = ["--store", self.disk] if self.disk else []
        return ["models", "get", self.get, *where] + (["--yes"] if self.yes else [])


class ModelRow(BaseModel):
    id: str
    family: str
    size: str
    variant: str
    entry: str
    evidence: str
    # the built-in recipes of this model certified on this very machine ([] where none)
    certified_here: list[str] = Field(default_factory=list)
    cached: bool
    gated: bool


class ModelsOutput(Output):
    schema_: Literal["poros.models.v1"] = Field("poros.models.v1", alias="schema")
    gpu: Optional[PlanGpu]
    models: list[ModelRow]
    downloaded: Optional[str] = None  # `models get`: the local snapshot path


def resolve(s: Sentence, ctx: Context) -> Draft[ModelsAction]:
    words = list(s.subject)
    if s.action != "get":
        return Draft(ModelsAction())
    if not words:
        return Draft(None, [shared.pick_model_question(ctx)])
    model, said, ask = shared.resolve_model(words, ctx, verb="models")
    if ask is not None:
        return Draft(None, [ask])
    mac.mac_gate(model, ctx)
    action = ModelsAction(get=model, yes=bool(s.value("yes") or s.flags.get("-y")),
                          disk=s.flags.get("--store"))
    action.provenance["get"] = Source(kind="given", detail=said)
    return Draft(action)


def inspect(action: ModelsAction, ctx: Context) -> ModelsAction:
    if action.get is not None:
        from poros.core.config import PorosConfig
        from poros.plan.configs import download_size

        if ctx.unified:
            action.disk = mac.mac_store(action.get, action.disk)
        f = download_size(PorosConfig(model_name_or_path=action.get, dataset="synthetic"))
        action.size_gib = f.value
    return action


def confirmations(action: ModelsAction, ctx: Context) -> list[Question]:
    if action.get is None:
        return []
    if ctx.unified and action.disk is None:
        return [mac.store_question(action.get)]
    if action.size_gib and not ctx.unified:  # train's own check: a download past the free disk
        from poros.core.config import PorosConfig
        from poros.preflight.command import disk_refusal
        from poros.preflight.disk import download_note

        refused = disk_refusal(download_note(
            PorosConfig(model_name_or_path=action.get, dataset="synthetic")))
        if refused is not None:
            raise refused
    if action.yes:
        return []
    what = shared.download_confirm(action.get, action.size_gib, ctx)
    if what is None:
        return []  # already on disk: nothing to fetch
    return [shared.download_question(action.get, what, "--yes", "Yes, download it", action.disk)]


def run(action: ModelsAction, ctx: Context) -> ModelsOutput:
    # Already all on disk (nothing left to fetch): no download, no network.
    fetch = action.get and action.size_gib != 0
    downloaded = _download(action.get) if fetch else None
    from poros.cli.sheets.listings import certified_models

    certified_here = certified_models(ctx)
    rows = []
    for e in model_words.catalog():
        rows.append(
            ModelRow(
                id=e.id,
                family=e.family,
                size=e.size,
                variant=e.variant,
                entry=e.entry,
                evidence=mac.evidence_of(e, ctx),
                certified_here=certified_here.get(e.id, []),
                cached=e.id in ctx.cached or e.id == action.get,
                gated=e.gated,
            )
        )
    return ModelsOutput(gpu=ctx.gpu, models=rows, downloaded=downloaded)


def _download(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    from poros.cli.ui import term
    from poros.core.hub_files import HUB_FILES

    with term.get().activity(f"Downloading {model_id.split('/')[-1]}") as act:
        path = snapshot_download(model_id, allow_patterns=HUB_FILES)
        act.done([])
    return path


def view(r: ModelsOutput, w: int) -> list[Line]:
    from poros.cli import views

    return views.models(r, w)
