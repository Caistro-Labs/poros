"""recipes: whole setups to start from, the training loops they use, and your own.

    poros recipes
    poros recipes show first-run
    poros recipes save                  # the last finished run, as your recipe
    poros recipes save 7f3a --name "Support tone"
    poros recipes share                 # its recipe file, written to this folder
    poros recipes sync                  # the shared ones, from GitHub
    poros recipes copy dpo
    poros recipes copy sft --output my_loop.py

Flags:
    --name TEXT       save: what to call it (default: from its model, data and settings)
    --output FILE     where the copy goes (default ./<name>_loop.py)

A setup is a small data-only file: ~/.poros/recipes holds yours, and one placed in
./recipes shows in /recipes as it is. Its numbers are a report, not checked by Poros.
Poros's own are certified by Caistro Labs: each ran at this release on the machines it names.
show gives a recipe's certified figures where this machine is one it ran on; on any other
machine it says where it was certified and that it was not run here, with no figure.
Only sync uses the network: it fetches the shared recipes repo, with no token, every file
through the same parser. Nothing writes through a link or over a file. A loop is one short
Python file written on poros.Run's verbs (forward_backward, optim_step, save, steps).
/train picks one from the data, or --recipe sft|dpo; a copied file trains with
--recipe my_loop.py.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from poros import runs
from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import Output, Question
from poros.cli.ui.text import Line
from poros.cli.words import runs as run_words
from poros.cli.words.grammar import Sentence
from poros.core.refusal import Refusal
from poros.recipes import presets, rules

# what each built-in reads and optimises, in plain words; the files say the rest
ABOUT = {
    "sft": ("SFT", "chats or text", "cross_entropy",
            "learns to answer like your chats; the prompt is context"),
    "dpo": ("DPO", "prompt, chosen, rejected", "dpo",
            "learns to prefer your chosen answer to the rejected one"),
    "ipo": ("IPO", "prompt, chosen, rejected", "ipo",
            "DPO with its margin held to a mark, so clean pairs cannot overfit"),
    "kto": ("KTO", "prompt, chosen, rejected", "kto",
            "each chosen answer a good one, each rejected a bad one; steady on noisy pairs"),
    "orpo": ("ORPO", "prompt, chosen, rejected", "orpo",
             "learns the chosen answers and the preference at once, with no reference"),
    "simpo": ("SimPO", "prompt, chosen, rejected", "simpo",
              "prefers the chosen answer by its mean logprob a token, with no reference"),
}


class RecipesAction(Action):
    verb: ClassVar[str] = "recipes"
    fork: Optional[str] = None  # a recipe to copy into this folder
    output: Optional[str] = None
    do: Literal["list", "show", "save", "share", "sync"] = "list"
    key: Optional[str] = None  # show: a recipe, by its file's name
    run: Optional[str] = None  # save, share: a finished run
    name: Optional[str] = None  # save: what to call it

    def canonical(self) -> list[str]:
        if self.do != "list":
            words = [self.key or self.run or ""] + (["--name", self.name] if self.name else [])
            return ["recipes", self.do, *words]
        if self.fork is None:
            return ["recipes"]
        return ["recipes", "copy", self.fork] + (["--output", self.output] if self.output else [])


class RecipeRow(BaseModel):
    name: str
    title: str
    data: str
    loss: str
    about: str
    file: str


class CertifiedHere(BaseModel):
    """What a certified recipe's runs say for this machine."""

    # certified: this machine is the certified model (the same card with at least its host RAM;
    # a Mac of the same class); same_memory_class: another card or Mac with at least the memory
    # of one (the same platform; on CUDA at least its VRAM and host RAM), so it should fit and its
    # time an update will differ; not_certified: less memory than every one; unknown: no GPU
    # here, or one named for a plan, whose host RAM nothing here knows
    status: Literal["certified", "same_memory_class", "not_certified", "unknown"]
    machine: Optional[str] = None  # this machine, as it was read
    certified_on: Optional[str] = None  # the certified machine that speaks for it
    # its experts stream from a drive (the SSD store): its time an update depends on that drive
    depends_on_drive: bool = False
    # not certified, on the certified card: why (its host RAM is less than that run's machine's)
    reason: Optional[str] = None
    # certified or same_memory_class, where the card has less free now than the run used: both
    # figures, in words (the status stands: free memory changes, the machine does not)
    free_now: Optional[str] = None
    # a card named for a plan (--gpu): certification is not matched, its host RAM is not known
    named_card: Optional[str] = None
    # always null: recipes says only what a certified run measured, never a planner estimate
    # (kept so a poros.recipes.v1 reader that reads it still validates)
    estimate: Optional[dict[str, Any]] = None


class Certification(BaseModel):
    """A built-in's certification: generated from the receipts (poros.recipes.certified)."""

    status: Literal["certified", "pending"]
    by: str
    release: str
    machines: list[dict[str, Any]]  # each certified run: machine, settings, figures, receipt
    here: CertifiedHere


class SetupRow(BaseModel):
    key: str
    name: str
    by: str
    source: str  # poros, yours, or the file's path
    model: str
    data: str
    loop: str
    settings: dict[str, Any]
    measured: dict[str, Any]  # a run's report: Poros has not checked it
    train: str  # the command it opens, on your data
    # a run of a family Poros has not validated: user_asserted, its model_type, in plain words
    support: Optional[str] = None
    model_type: Optional[str] = None
    says: str = ""
    certified: Optional[Certification] = None  # Poros's own: who certified it, where, and here
    # a built-in that takes any data: its command with --data synthetic, which trains the whole
    # loop on generated rows, for someone with no file of their own yet
    without_data: Optional[str] = None


class RecipesOutput(Output):
    schema_: Literal["poros.recipes.v1"] = Field("poros.recipes.v1", alias="schema")
    recipes: list[RecipeRow]
    losses: list[str]  # every loss forward_backward takes by name
    copied: Optional[str] = None  # `recipes copy`: the file written
    do: str = "list"
    setups: list[SetupRow] = []  # whole setups: yours, this project's, the shared ones, Poros's
    refused: list[str] = []  # recipe files not read, each with why
    recipe: Optional[SetupRow] = None  # show: the one
    file: Optional[str] = None  # save, share: the file written
    synced: Optional[dict[str, Any]] = None  # sync: the repo, when, how many, what was refused
    content: Optional[str] = None  # and what it holds, exactly


def _finished(word: Optional[str]) -> runs.Run:
    """The run named, or the last that finished: only a finished run has numbers to report."""
    every = runs.all_runs()
    if word is None or word.lower() in ("last", "latest", "newest"):
        hit = next((r for r in every if r.state == "succeeded"), None)
        if hit is None:
            raise Refusal("E_NOT_FOUND", "No finished run", "no run has finished yet",
                          fix="poros runs", note="lists them")
        return hit
    found = run_words.resolve(word, [r.id for r in every], names=[(r.name, r.id) for r in every])
    if isinstance(found, run_words.RunMiss):
        raise Refusal("E_NOT_FOUND", "No such run", "no run by that name", fix="poros runs",
                      note="lists them")
    hit = next(r for r in every if r.id == found)
    if hit.state != "succeeded":
        raise Refusal("E_USAGE", "Not finished", "only a finished run becomes a recipe",
                      fix="poros runs", note="shows how each went")
    return hit


def _setup(key: str, ctx: Context) -> presets.Recipe:
    told = presets.pointer(key, ctx.unified)
    if told is not None:  # the other platform's built-in: its twin here, and how it trains
        why, to, trains = told
        raise Refusal("E_NOT_FOUND", "Another machine's recipe", why,
                      context=[f"it trains with: {trains}"], fix=f"poros recipes show {to}",
                      note="what it is and sets")
    hit = next((r for r in presets.all_recipes(ctx.cwd) if r.key == key), None)
    if hit is None:
        now = presets.RENAMED.get(key.lower())
        raise Refusal("E_NOT_FOUND", "No such recipe",
                      f"{key.lower()} is now {now}" if now else "no recipe by that name",
                      fix=f"poros recipes show {now}" if now else "poros recipes",
                      note="" if now else "lists them")
    return hit


def resolve(s: Sentence, ctx: Context) -> Draft[RecipesAction]:
    word = s.subject[0] if s.subject else None
    named = s.value("name")
    if s.action == "show":
        if word is None:
            raise Refusal("E_USAGE", "Which recipe", "show needs a recipe's name",
                          fix="poros recipes", note="lists them")
        return Draft(RecipesAction(do="show", key=_setup(word, ctx).key))
    if s.action == "share" and named:
        raise Refusal("E_USAGE", "No names", "a shared file holds no one's words",
                      fix="poros recipes share")
    if s.action == "sync":
        if s.subject:
            raise Refusal("E_USAGE", "Sync takes no name", "it fetches the whole repo",
                                 fix="poros recipes sync")
        return Draft(RecipesAction(do="sync"))
    if s.action in ("save", "share"):
        return Draft(RecipesAction(do=s.action, run=_finished(word).id,
                                   name=str(named) if named else None))
    if s.action != "copy":
        if s.subject:
            raise Refusal("E_USAGE", "Copy which way", "recipes takes no name by itself",
                          context=["poros recipes copy sft"])
        return Draft(RecipesAction())
    if not s.subject:
        return Draft(None, [_which(ctx)])
    name = s.subject[0].lower()
    if name not in ABOUT:
        raise Refusal("E_USAGE", "Unknown recipe", f"{name!r} is not a recipe",
                      context=[f"built in: {', '.join(ABOUT)}"])
    out = s.value("output")
    target = Path(str(out)).expanduser() if out and out is not True else ctx.cwd / f"{name}_loop.py"
    if not target.is_absolute():
        target = ctx.cwd / target
    if os.path.lexists(target):
        raise Refusal("E_USAGE", "Already there", f"{target.name} already exists",
                      context=["pick another name: --output other_loop.py"])
    return Draft(RecipesAction(fork=name, output=str(target)))


def _which(ctx: Context) -> Question:
    from poros.cli.schemas import Choice

    return Question(
        id="subject",
        prompt="Copy which recipe?",
        flag="",
        default="sft",
        choices=[Choice(value=n, label=t, note=d) for n, (t, d, _, _) in ABOUT.items()],
    )


def confirmations(action: RecipesAction, ctx: Context) -> list[Question]:
    return []


def _of_run(run_id: str) -> dict[str, Any]:
    """A finished run's recipe, from what the run recorded."""
    from poros.cli.verbs.data import shape_of
    from poros.cli.verbs.runs_report import facts

    r = next(x for x in runs.all_runs() if x.id == run_id)
    seen = facts(r.folder, r.output_dir)
    if r.kind == "job":  # a job's folder holds the config it ran
        import yaml

        told = yaml.safe_load(Path(r.folder, "config.yaml").read_text()) or {}
    else:
        action = seen["action"]
        told = {"dataset": action.get("data"), "max_seq_length": action.get("seq_len"),
                "lora_rank": action.get("rank"), "max_steps": action.get("steps"),
                "block_size": action.get("block_size"), **action.get("config", {})}
    shape = shape_of(Path(str(told.get("dataset"))))
    try:
        return rules.from_run(r.model, told, seen["receipt"], seen["held_out"], shape)
    except ValueError as exc:
        raise Refusal("E_USAGE", "Can't be a recipe", str(exc)) from None


def _file_key(fields: dict[str, Any]) -> str:
    """``qwen2.5-7b-chats-256``: its model, its data and its length."""
    words = [fields["model"].rsplit("/", 1)[-1], fields["data"],
             str(fields["settings"].get("max_seq_length", ""))]
    return re.sub(r"[^a-z0-9.]+", "-", " ".join(words).lower()).strip("-")


def _write(action: RecipesAction, ctx: Context) -> tuple[Path, str]:
    """Where it goes and what it says. Its name comes from checked fields, whoever asked, and what
    it says is what was written: the project can swap the file for another before a second look."""
    fields = _of_run(action.run or "")
    if action.do == "share":
        text = rules.dumps(fields)
        return rules.write_new(ctx.cwd, _file_key(fields), text), text
    if presets.is_built_in(action.name or ""):
        raise Refusal("E_USAGE", "Another name", f"{action.name} is one of Poros's own "
                      "recipes, and that name stays its own")
    try:
        text = rules.dumps({**fields, "name": action.name or ""})
    except ValueError:
        raise Refusal("E_USAGE", "Another name", "a name is one word of letters, digits, "
                      "dots and dashes, up to 40") from None
    key = (action.name or "").lower() or _file_key(fields)
    return rules.write_new(presets.yours(), key, text), text


# What a Mac the table has not measured is told in place of "should fit" (poros.metal.plan's own
# account of such a Mac's rows)
PLANNED_MAC = "planned from an M4 with 16 GB; not measured on this Mac"


def machine_check(r: presets.Recipe, ctx: Context, found: tuple, this: Any) -> tuple:
    """(the certified run that speaks for this machine and how, as certified.match says; why a
    card Poros does not train on is not certified, in the refusal's words; the words an unmeasured
    Mac class is told in place of "should fit"). The recipes list, the card and show read this."""
    from poros.recipes import certified

    hit = certified.match(found, this)
    refused = planned = None
    if this is not None and this.platform == "cuda":
        from poros.core.card import card_refusal
        from poros.core.config import PorosConfig

        why = card_refusal(PorosConfig(model_name_or_path=r.model, dataset="synthetic"), ctx.gpu)
        if why:  # no native bf16: not certified, and said as the card's refusal is
            hit, refused = None, f"Not a supported GPU: {why}"
    elif hit is not None and hit[1] == certified.SAME_MEMORY:
        from poros.metal import measured
        from poros.metal.table import class_of

        if not measured(class_of(ctx.gpu.name)):
            planned = PLANNED_MAC
    return hit, refused, planned


def mac_short(r: presets.Recipe, ctx: Context) -> Optional[str]:
    """On a Mac: the train start's own warning where the memory free now is short of what a run
    of this recipe peaked at when it was measured here ("memory is short right now"), else
    None. The verdict stands: free memory changes, the Mac does not."""
    if not ctx.unified or ctx.host_free_gib is None:
        return None
    from poros.cli.verbs import shared
    from poros.preflight.mac import SHORT, memory_warning

    st = {**presets.loop_preset(r), **r.settings}
    p = shared.plan_for(r.model, ctx, seq_len=st.get("max_seq_length"),
                        rank=int(st.get("lora_rank", 16)), dropout=st.get("lora_dropout"))
    if not memory_warning(p, guard=bool(st.get("memory_guard", True))):
        return None
    from poros.cli.sheets.fit import mac_memory_note

    # the train start's decision, said with this recipe's certified figure, never a cell's
    return f"{SHORT}: {mac_memory_note(p.model_copy(update={'model': r.model}), key=r.key)}"


def _certification(r: presets.Recipe, ctx: Context, show: bool) -> Optional[Certification]:
    from poros.recipes import certified

    if r.source != "poros":
        return None
    said = certified.status(r.key)
    if said == "none":
        return None
    found = certified.entries(r.key)
    this = certified.here(ctx)
    hit, refused, planned = machine_check(r, ctx, found, this)
    drive = r.settings.get("store") == "ssd"
    if this is None:
        named = ctx.gpu.name if ctx.gpu is not None and ctx.gpu_name is not None else None
        here = CertifiedHere(status="unknown", depends_on_drive=drive, named_card=named)
    else:
        here = CertifiedHere(status=hit[1] if hit else "not_certified", machine=this.label(),
                             certified_on=hit[0].machine.label() if hit else None,
                             depends_on_drive=drive,
                             reason=refused or (None if hit else certified.short_of(found, this)))
        free = ctx.gpu.free_gib if ctx.gpu is not None else None
        # a certified figure only on the machine it ran on (the owner's rule)
        used = hit[0].fit_gib() if hit and hit[1] == certified.CERTIFIED else None
        from poros.cli.recipes_card import plan_of
        from poros.cli.sheets.fit import host_said, host_short

        short = host_short(plan_of(r, ctx)) if hit and not ctx.unified else None
        if short:  # host RAM is short here now: said first, in the refusal's own words
            here.free_now = host_said(plan_of(r, ctx))
        elif ctx.unified and hit:  # one pool: the train start's own words for memory free now
            here.free_now = mac_short(r, ctx)
        elif free is not None and used is not None and free < used:
            here.free_now = (f"this card has {free:.1f} GiB free now, and the certified run used "
                             f"{used:.1f} GiB of GPU memory: free some before it starts")
    return Certification(status=said, by=certified.CERTIFIER, release=certified.book().release,
                         machines=[e.fields() for e in found], here=here)


def _row(r: presets.Recipe, ctx: Optional[Context] = None, show: bool = False) -> SetupRow:
    # a shared recipe's words are for the card: an agent gets its name made of what it is
    return SetupRow(key=r.key, name=r.title() if r.source == "shared" else r.name, by=r.by,
                    source=r.source, model=r.model, data=r.data,
                    loop=r.loop, settings=r.settings, measured=r.measured,
                    train=shlex.join(["poros", *r.argv("<data>")]),
                    without_data=shlex.join(["poros", *r.argv("synthetic")])
                    if r.source == "poros" and r.data == "any" else None,
                    support="user_asserted" if r.unvalidated else None,
                    model_type=r.unvalidated or None, says=r.says(),
                    certified=_certification(r, ctx or Context(), show))


def run(action: RecipesAction, ctx: Context) -> RecipesOutput:
    from poros import losses, recipes
    from poros.cli.verbs import mac
    from poros.recipes import hub

    copied = file = content = synced = None
    if action.do == "sync":
        try:
            synced = hub.sync()
        except ValueError as exc:
            raise Refusal("E_NETWORK", "Not synced", str(exc),
                          context=["the copy synced last stays as it was"]) from None
        except OSError:
            raise Refusal("E_DISK", "Not synced", "the shared recipes could not be "
                          "written here", context=["it goes through no link"]) from None
    if action.fork is not None:
        copied = str(recipes.copy(action.fork, action.output))
    if action.do in ("save", "share"):
        try:
            path, content = _write(action, ctx)
        except (OSError, ValueError):
            raise Refusal("E_DISK", "Not written", "the recipe file could not be written "
                          "here", context=["it goes through no link"]) from None
        file = re.sub(r"[\x00-\x1f\x7f]", "?", str(path))
    rows = [
        RecipeRow(name=n, title=t, data=d, loss=lo, about=ab, file=str(recipes.path(n)))
        for n, (t, d, lo, ab) in ABOUT.items()
    ]
    kept, refused = presets.local(ctx.cwd)
    return RecipesOutput(
        do=action.do, recipes=rows, losses=losses.names(), copied=copied, file=file,
        content=content, synced=synced,
        setups=[_row(r, ctx) for r in kept + hub.cached()[0] + presets.BUILT_IN
                if mac.trains_here(r, ctx)],
        refused=[f"{where}: {why}" for where, why in refused],
        recipe=_row(_setup(action.key, ctx), ctx, show=True) if action.do == "show" else None,
    )


def view(r: RecipesOutput, w: int) -> list[Line]:
    from poros.cli import views

    return views.recipes(r, w)
