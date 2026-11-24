"""What the verbs ask of a Mac, in one place: its row, what it trains, which disk. Torch-free."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from poros.cli.schemas import Choice, Question
from poros.cli.words import models as model_words

if TYPE_CHECKING:
    from poros.cli.context import Context


def _class(ctx: "Context") -> str:
    from poros.metal.table import class_of

    return class_of(ctx.gpu.name)


def mac_row(model_id: str, ctx: "Context", *, seq=None, tokens=None):
    """This model's row in the Mac table, or None: not a Mac, or no row on this one."""
    if not ctx.unified:
        return None
    from poros.metal.table import pick

    return pick(model_id, _class(ctx), seq=seq, tokens=tokens)


def lengths(model_id: str, ctx: "Context") -> list[int]:
    """The lengths in this model's row of this Mac's table."""
    from poros.metal.table import rows

    return [r.seq for r in rows(model_id, _class(ctx))]


def trainable(ctx: "Context", entries=None) -> list[model_words.Entry]:
    """The catalog (or ``entries``), less what this Mac does not train."""
    entries = model_words.catalog() if entries is None else entries
    return [e for e in entries if not ctx.unified or mac_row(e.id, ctx)]


def trains_here(recipe, ctx: "Context") -> bool:
    """A whole setup this machine can run: a Mac offers none it would refuse, and a machine that
    is not a Mac none of Poros's Mac recipes (block 1, one row a step: legal, never meant)."""
    if not ctx.unified:
        return not (recipe.source == "poros" and recipe.use == "mac")
    if recipe.source == "poros" and recipe.use != "mac":  # a CUDA built-in: its Mac twin is here
        return False
    if not mac_row(recipe.model, ctx):
        return False
    from poros.core.config import PorosConfig
    from poros.core.errors import PorosError
    from poros.metal import validate

    try:
        validate(PorosConfig(model_name_or_path=recipe.model, dataset="synthetic",
                             **recipe.settings))
    except (PorosError, ValueError):
        return False
    return True


def evidence_of(e: model_words.Entry, ctx: "Context") -> str:
    """The catalog's evidence, or on a Mac what this class of Mac holds for the model."""
    if not ctx.unified:
        return e.evidence
    from poros.metal.table import evidence

    return evidence(e.id, _class(ctx))


def unread(model_id: str, revision: Optional[str] = None):
    """On a Mac, the refusal for a Hub id whose config.json is in no cache here, neither the Hugging
    Face one nor any store's, with no Hub to ask (offline, or no network): the words off a Mac
    (E_NETWORK). None for anything else: a catalog model, a folder, a config that can be read."""
    if model_words.by_id(model_id) is not None or Path(model_id).expanduser().exists():
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # no huggingface_hub: the load says what it needs
        return None
    from poros.metal import store
    from poros.preflight.command import model_refusal

    if any(isinstance(try_to_load_from_cache(model_id, "config.json", revision=revision,
                                             cache_dir=hub), str) for hub in store.hubs()):
        return None
    return model_refusal(model_id, revision)  # the Hub's own answer, or E_NETWORK, or None


def mac_gate(model_id: str, ctx: "Context") -> None:
    """On a Mac, a model it does not train stops here, before anything is asked or fetched."""
    if not ctx.unified:
        return
    from poros.core.errors import PorosGuaranteeUnavailableError
    from poros.core.refusal import Refusal
    from poros.metal.table import require_row
    from poros.moe.carried import declares_visits
    from poros.plan.configs import read_model_config

    raw = read_model_config(model_id) or {}
    if declares_visits(raw):  # a looped decoder: not on a Mac
        from poros.preflight.arch import mac_looped

        raise Refusal("E_ARCH_UNSUPPORTED", "Not on a Mac", f"{model_id}: {mac_looped(raw)}",
                      fix="poros models", note="what this Mac trains")
    try:
        require_row(model_id, _class(ctx))
    except PorosGuaranteeUnavailableError as exc:
        raise Refusal(exc.code, "Not on this Mac", exc.info.message, fix=exc.info.remediation)


def mac_store(model_id: str, given: Optional[str]) -> Optional[str]:
    """The disk for a model on this Mac, in use from here on. None: `store_question` asks."""
    from poros.core.errors import PorosDiskError
    from poros.core.refusal import Refusal
    from poros.metal import store

    try:
        found = store.pick(model_id, given)
    except PorosDiskError as exc:
        raise Refusal(exc.code, "Not enough disk", exc.info.message, fix=exc.info.remediation)
    if found is None:
        return None
    store.use(found)
    return str(found)


def run_files(disk: Optional[str]) -> Optional[Path]:
    """The folder a run keeps its files under, if not its own. Only a Mac's store says so."""
    if not disk:
        return None
    from poros.metal import store

    return store.run_files(disk)


def run_room(model_id: str, rank: int, keep: int, folder: Path) -> str:
    """Why this run's checkpoints would fill the disk they go on, hours in, or ""."""
    from poros.metal.store import nearest_existing
    from poros.metal.table import run_gb

    need = run_gb(model_id, rank, keep)
    free = shutil.disk_usage(nearest_existing(folder)).free / 1e9
    if keep <= 0 or need <= free:
        return ""
    return (f"This run keeps about {need:.0f} GB of checkpoints in {folder}, and that disk has "
            f"{free:.0f} GB free. POROS_HOME moves them, save_total_limit=1 keeps fewer.")


def store_question(model_id: str) -> Question:
    from poros.metal import store

    fit = store.qualifying(model_id)
    return Question(
        id="store_path",
        prompt=f"Which disk for {model_id.split('/')[-1]}?",
        flag="--store",
        remember="store",
        default=str(fit[0].store),
        choices=[
            Choice(value=str(v.store), label=v.label, note=f"{v.free / 1e9:.0f} GB free")
            for v in fit
        ],
    )
