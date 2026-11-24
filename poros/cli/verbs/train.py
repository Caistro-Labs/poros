"""train: fine-tune a model on your data.

    poros train qwen2.5-7b-instruct --data chats.jsonl
    poros train --from last --data prefs.jsonl
    poros train Qwen/Qwen2.5-7B-Instruct --data ./data/ --steps 300 --seq-len 1024

Flags:
    --data FILE       chats or text train SFT; prompt/chosen/rejected pairs train DPO
    --steps N         optimizer steps (default: the recipe's passes over your rows)
    --epochs N        passes over the data, instead of --steps
    --seq-len N       tokens per row (default: the length that holds 95% of your rows whole,
                      or a shorter one where the check refuses that one)
    --rank N          LoRA rank (default 16, as the recipes train)
    --block-size N    layers resident at once (default 4, as the recipes train)
    --recipe NAME     sft, dpo, ipo, kto, orpo, simpo, auto or your loop.py (default: by the data)
    --from RUN        start from that run's adapter: DPO after SFT (an id or last), or
                      from one checkpoint of it: poros://<run>/weights/<step>, or best
    --eval FILE       held-out rows, the data's shape: scored before training, at each
                      checkpoint and after (default: chats and pairs hold 5% out; text
                      rows train every row and hold none out, unless --recipe sft)
    --store WHERE     a MoE model's experts: ram (default, fastest), or ssd to
                      stream them from a .sss file in ~/.poros/sss (or a folder
                      you name: --store /mnt/nvme) and use a few GiB of RAM
    --name NAME       the run's name (default: model, data and date)
    --output DIR      where the adapter goes (default: the run's own folder); not
                      for a run in the background, which keeps its own
    --detach          train in the background; watch it with
                      poros runs <id> --live --wait 30 (two rows per update)
    --yes             no start screen: every confirmation given
    --download        yes to downloading the model, or a Hub dataset
    --force           start where the check refuses the run, or the disk would fill
    --overwrite       replace the previous run in the --output folder
    --unvalidated     train a model whose family Poros has not validated, this run
                      only, with no parity guarantee; every result about it says so.
                      `poros arch try <model>` checks its math first
    key=value         any config field: learning_rate=1e-4 dpo_beta=0.2 train_on=last
                      (poros schema config lists them). Not one that runs code: your
                      own loop is --recipe FILE.py.
                      path=NAME picks how a CUDA run trains a step, with the same
                      result each way: efficient (default) takes less VRAM than fast;
                      fast is the fastest; sequential is slower and can peak lower;
                      with --store ssd, fast+cache is the fastest. poros recipes
                      shows what each certified run measured.
                      eval_checkpoints=false skips the held-out score at each
                      checkpoint; eval_steps=N scores every N steps (default 500).

With a config file:
    poros train run.yaml          trains the config (poros init writes a starter one)
    --resume-from DIR|latest      continue from a checkpoint folder, or the newest one in
                                  the config's output_dir, bit for bit
    --steps N, --seq-len N        override max_steps, max_seq_length
    --block-size N, --store W     override block_size, the expert store (--sss-ram too)
    --output-dir DIR              override output_dir
    --force                       train even where the check refuses the run
    --trust-remote-code           for a base model that ships its own code
    --detach, --yes, --unvalidated  as above

Every recipe trains at the settings Poros's certified runs use (NF4, 4 layers
on the GPU at once, steps of 8 rows, rank 16, alpha 32, no dropout); SFT at lr
2e-4 for 2 epochs, DPO at lr 5e-6 for 1. What is not
said is inferred or defaulted, and every value shows where it came from.
In a terminal the run is shown first: Start, Change a setting, or Cancel.
The run gets a folder that `poros runs` reads: ~/.poros/runs/<id>/, or in the
background ~/.poros/jobs/<id>/artifacts/. Its result's output_dir says which.
On a Mac, Poros picks length, block and disk use for the model, and trains one row a step at
lr 2e-5.
"""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import Field

from poros.cli.action import Action, Draft, Source
from poros.cli.context import Context
from poros.cli.schemas import Choice, PlanOutput, Question, TrainOutput
from poros.cli.ui.text import Line
from poros.cli.verbs import mac, shared
from poros.cli.verbs import train_data as data_scan
from poros.cli.verbs.train_data import before_the_device  # noqa: F401 (app asks train for it)
from poros.cli.verbs.train_model import (  # noqa: F401 (named here for fit and prove)
    new_family,
    require_model,
    unvalidated_questions,
)
from poros.cli.words import data as data_words
from poros.cli.words import models as model_words
from poros.cli.words.grammar import Sentence, with_value
from poros.core.refusal import Refusal
from poros.preflight import command
from poros.recipes.rules import within

STEP_CAP = 2000
SYNTHETIC_STEPS = 200


class TrainAction(Action):
    verb: ClassVar[str] = "train"
    model: str
    data: str
    steps: int = 0  # filled by inspect when not given
    epochs: Optional[int] = None  # `--epochs 3`: inspect turns it into steps
    seq_len: int = 0
    rank: int = shared.DEFAULT_RANK
    block_size: Optional[int] = None
    name: str = ""
    output: Optional[str] = None  # `to ./out`; default: the run folder
    config: dict[str, Any] = {}  # key=value PorosConfig fields, and what --from inherits
    detach: bool = False
    yes: bool = False  # --yes: no start screen, every confirmation given
    download: bool = False  # --download: the download, confirmed
    force: bool = False
    overwrite: bool = False
    allow_unvalidated: bool = False
    # --unvalidated: the model_type this run asserts, a family Poros has not validated
    unvalidated: Optional[str] = None
    family: Optional[str] = Field(None, exclude=True)  # outside the catalog: arch's model_type
    family_status: str = Field("", exclude=True)  # and the registry's word for it
    disk: Optional[str] = Field(None, exclude=True)  # on a Mac: where the weights go
    kept: Optional[float] = Field(None, exclude=True)  # on a Mac: the rows that fit the length
    room: str = Field("", exclude=True)  # on a Mac: why its checkpoints may fill the disk
    from_run: Optional[str] = None  # --from <run>: start from its adapter (DPO after SFT)
    from_step: Optional[str] = None  # --from <run>/weights/<step>: from that checkpoint
    # inspect's facts: shown, never part of the command
    rows: Optional[int] = Field(None, exclude=True)
    plan: Optional[PlanOutput] = Field(None, exclude=True)
    # (recipe, its certified entry) where a built-in was certified on this very machine at these
    # settings (poros.cli.sheets.fit.certified_run): the only figures a card shows
    certified: Optional[Any] = Field(None, exclude=True)
    # plan_for's settings for this run besides its length and path, and the run's settings its
    # plan reads (shared.PLAN_KEYS): a card plans each path with them
    plan_kw: dict = Field({}, exclude=True)
    plan_said: dict = Field({}, exclude=True)
    recipe: str = Field("", exclude=True)  # SFT / DPO / classic / a loop file's name
    shape: Optional[str] = Field(None, exclude=True)  # chats / pairs / text
    batch: int = Field(1, exclude=True)  # rows through the model per step
    lr: Optional[float] = Field(None, exclude=True)  # the learning rate it will train at
    preset: dict = Field({}, exclude=True)  # the recipe's settings, under anything you set
    passes: int = Field(0, exclude=True)  # the epochs its steps make: yours, else the recipe's
    per_epoch: int = Field(0, exclude=True)  # steps a pass over the rows it trains on
    held: Optional[int] = Field(None, exclude=True)  # rows it scores; None: not known here

    def canonical(self) -> list[str]:
        out = [
            "train",
            self.model,
            *with_value("--data", self.data),
            "--steps",
            str(self.steps),
            "--seq-len",
            str(self.seq_len),
            "--rank",
            str(self.rank),
        ]
        if self.block_size:
            out += ["--block-size", str(self.block_size)]
        if self.output:
            out += with_value("--output", self.output)
        if self.name:  # always: a derived name carries the date, so it is pinned here
            out += with_value("--name", self.name)
        config = dict(self.config)
        recipe = config.pop("loop", None) or config.pop("recipe", None)
        config.pop("recipe", None)
        if recipe:
            out += ["--recipe", recipe]
        config.pop("init_adapter", None)
        if self.from_run:
            step = self.from_step
            out += ["--from", f"poros://{self.from_run}/weights/{step}" if step else self.from_run]
        if shared.path_choice(config) == shared.CACHE_PATH:  # as it was said: one choice
            config.pop("store_options")
            config["path"] = shared.CACHE_PATH
        out += [f"{k}={v}" for k, v in sorted(config.items())]
        for flag, on in (
            ("--detach", self.detach),
            ("--download", self.download),
            ("--force", self.force),
            ("--overwrite", self.overwrite),
            ("--allow-unvalidated", self.allow_unvalidated),
            ("--unvalidated", bool(self.unvalidated)),
        ):
            if on:
                out.append(flag)
        return out


_RULED = {"seq_len": "max_seq_length", "rank": "lora_rank", "block_size": "block_size"}


def _number(s: Sentence, name: str) -> tuple[Optional[int], Optional[Source]]:
    raw = s.value(name)
    if raw is None:
        return None, None
    if raw is True:  # the flag with nothing after it: never read as "the default"
        flag = "--" + name.replace("_", "-")
        raise Refusal("E_USAGE", "Missing a value", f"{flag} needs a number after it",
                      fix="poros train --help", note="every flag, with what it takes")
    try:
        n = int(str(raw).replace(",", "").replace("_", ""))
    except ValueError:
        label = {"seq_len": "length", "block_size": "block size"}.get(name, name)
        raise Refusal("E_USAGE", "Not a number", f"{label} {raw!r} is not a number")
    if n < 1:
        raise Refusal("E_USAGE", "Out of range", f"{name.replace('_', ' ')} must be >= 1")
    if name in _RULED and not within(_RULED[name], n):  # the most a recipe may hold, for any run
        raise Refusal("E_USAGE", "Out of range",
                      f"{name.replace('_', ' ')} {n:,} is past what a run can use")
    said = s.clauses[name].said if name in s.clauses else f"--{name.replace('_', '-')} {raw}"
    return n, Source(kind="given", detail=said)


# Config fields train sets itself: as key=value they would silently disagree
# with the screen, the fit verdict and the command. Each has its own flag.
OWN_FIELDS = {
    "model_name_or_path": "the model, first",
    "dataset": "--data <data>",
    "max_steps": "--steps N",
    "max_seq_length": "--seq-len N",
    "lora_rank": "--rank N",
    "output_dir": "--output <dir>",
    "block_size": "--block-size N",
}


# What would run someone's code. A key=value can come from anywhere a line is pasted from, so only
# a person's explicit flag grants these: --trust-remote-code on a config file, --recipe FILE.py.
GRANTS = {
    "trust_remote_code": "poros train <config.yaml> --trust-remote-code",
    "loop": "--recipe your_loop.py",
}


def _config_fields(s: Sentence) -> dict[str, str]:
    from poros.core.config import PorosConfig

    known = PorosConfig.model_fields
    out = {}
    for name, clause in s.clauses.items():
        if name in ("data", "steps", "epochs", "seq_len", "rank", "output", "name", "gpu"):
            continue
        if name in GRANTS:
            raise Refusal("E_USAGE", "A flag, not a setting",
                          f"{name} runs code, so no key=value grants it",
                          context=[f"use {GRANTS[name]}"])
        if name in OWN_FIELDS:
            raise Refusal(
                "E_USAGE",
                "Say it in words",
                f"{name} is set by train itself",
                context=[f"use {OWN_FIELDS[name]}"],
            )
        if name not in known:
            import difflib

            near = difflib.get_close_matches(name, known, n=1)
            raise Refusal(
                "E_CONFIG",
                "Unknown setting",
                f"{name} is not a config field",
                context=[f"did you mean {near[0]}?"] if near else [],
                fix="poros schema config",
                note="lists every field",
            )
        out[name] = str(clause.value)
    return out


RECIPES = ("auto", "sft", "dpo", "ipo", "kto", "orpo", "simpo")


def sss_dir() -> Path:
    """Where ``--store ssd`` keeps a model's .sss expert file unless told."""
    from poros.cli.context import home

    return home() / "sss"


def _store(s: Sentence, ctx: Context, config: dict[str, str]) -> None:
    """``--store ram|ssd|<folder>`` and ``--sss-ram 4G`` into the config.
    A folder means ssd there; ssd alone means the folder under ~/.poros."""
    from poros.core.config import parse_bytes, parse_store

    raw = s.flags.get("--store")
    if ctx.unified and raw is not None:
        if raw is True or not str(raw).startswith(("/", "~", ".")):
            raise Refusal("E_USAGE", "Which disk", "--store names a folder on a Mac",
                          context=["Poros picks memory or disk for each model"])
        config["store_path"] = _absolute(str(raw), ctx)
        return
    if raw is True:
        raise Refusal("E_USAGE", "Which store", "--store needs a value",
                      context=["ram: experts in host RAM (the default)",
                               "ssd, or a folder on an SSD: streamed from a .sss file"])
    if raw is not None:
        raw = str(raw)
        if raw.startswith(("/", "~", ".")):
            raw = "ssd:" + raw
        try:
            kind, path = parse_store(raw)
        except ValueError as exc:
            raise Refusal("E_USAGE", "Which store", str(exc)) from None
        config["store"] = kind
        if path:
            config["store_path"] = path
    if config.get("store") == "ssd" and not ctx.unified:  # a Mac's store picks its own folder
        path = config.get("store_path") or str(sss_dir())
        config["store_path"] = _absolute(path, ctx)
    if config.get("path") == shared.CACHE_PATH:  # the SSD store's choice: fast + its host cache
        if config.get("store") != "ssd":
            raise Refusal("E_USAGE", "SSD only",
                          f"path={shared.CACHE_PATH} is fast with the SSD store's host cache",
                          context=["add --store ssd (or --store /a/folder/on/an/ssd)",
                                   "or path=fast, with the experts in RAM"])
        config.update(shared.path_config(shared.CACHE_PATH))
    ram = s.flags.get("--sss-ram")
    if ram is not None:
        try:
            parse_bytes(str(ram))
        except ValueError as exc:
            raise Refusal("E_USAGE", "How much RAM", str(exc)) from None
        config["sss_ram"] = str(ram)


def _recipe(s: Sentence, ctx: Context, config: dict[str, str]) -> None:
    """``--recipe sft|dpo|auto|<file.py>`` into the config's recipe or loop."""
    raw = s.flags.get("--recipe")
    if raw is None:
        return
    if raw is True:
        raise Refusal("E_USAGE", "Which recipe", "--recipe needs a value",
                      context=["sft, dpo, ipo, kto, orpo, simpo, auto, or a loop .py"])
    raw = str(raw)
    if raw.endswith(".py"):
        path = _absolute(raw, ctx)
        if not Path(path).is_file():
            raise Refusal("E_USAGE", "No such loop", f"{raw} does not exist",
                          fix="poros recipes copy sft",
                          note="writes the SFT loop here to start from")
        config["loop"] = path
        config.pop("recipe", None)
    elif raw.lower() in RECIPES:
        if raw.lower() != "auto":
            config["recipe"] = raw.lower()
    else:
        _refuse_setup(s, ctx)
        raise Refusal("E_USAGE", "Unknown recipe", f"{raw!r} is not a recipe",
                      context=["sft, dpo, ipo, kto, orpo, simpo, auto, or a loop .py"])


def _refuse_setup(s: Sentence, ctx: Context) -> None:
    """``--recipe <a built-in setup's key>``: refused at once, model or none, with the command
    that trains that setup (the user's --data in it), never a question first. A setup runs as its
    own train command; --recipe names a loop."""
    from poros.recipes import presets

    raw = s.flags.get("--recipe")
    if not isinstance(raw, str) or raw.endswith(".py") or raw.lower() in RECIPES:
        return
    told = presets.pointer(raw, ctx.unified)  # the other platform's built-in: its twin here
    key = told[1] if told else raw.lower()
    setup = next((r for r in presets.built_in() if r.key == key), None)
    setup = setup or (presets.DEFINED_BY_KEY[key] if told else None)
    if setup is None:
        return
    data = s.value("data")
    trains = shlex.join(["poros", *setup.argv(data if isinstance(data, str) else "<data>")])
    why = told[0] if told else f"{setup.key} is one of Poros's recipes"
    raise Refusal("E_USAGE", "A setup, not a loop",
                  f"{why}: a whole setup, which --recipe (the loop) does not take",
                  context=[f"it trains with: {trains}",  # first: what a terminal's sheet shows
                           f"poros recipes show {setup.key} says what it is and sets"],
                  fix=trains, note=f"trains {setup.key}")


def _absolute(path: str, ctx: Context) -> str:
    """Stored absolute: a run continued from another directory must read the
    same files (and the canonical command reproduces it anywhere)."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else (ctx.cwd / p).resolve())


def data_question(ctx: Context, near: list[str] = (), missed: str = "") -> Question:
    """Which data. After a miss it says so, and nothing is pre-selected: a
    correction is the user's to pick, never a default an agent takes."""
    files = list(near) or data_scan.local_files(ctx.cwd)
    if not ctx.interactive:  # anyone's words, on their way to an agent: a plain name or none
        files = [f for f in files if data_scan.PLAIN_NAME.fullmatch(f.rstrip("/"))]
    files = files[:6]
    if not files and not missed:  # nothing here to offer: said, not asked
        raise Refusal("E_USAGE", "Which data", "train needs data, and this folder holds no "
                      ".jsonl, .json or .csv file",
                      context=["--data FILE: chats or text train SFT, preference pairs DPO",
                               "--data synthetic trains on random tokens, to try a run"],
                      fix="poros docs training data", note="the shapes a row can take")
    choices = [Choice(value=f, label=f, note="closest name" if missed else "here") for f in files]
    prompt = (
        f"No file named {missed}. Did you mean one of these?" if missed else ("Train on what data?")
    )
    return Question(
        id="data",
        prompt=prompt,
        flag="--data",
        choices=choices,
        default=None if missed else choices[0].value,
    )


def _from(s: Sentence) -> Optional[tuple]:
    """``--from <run>``: the run, its last complete checkpoint and None; or with a checkpoint's
    path, that checkpoint (best as the step it is) and its step. None without --from."""
    raw = s.flags.get("--from")
    if raw is None:
        return None
    from poros import runs
    from poros.cli.verbs import checkpoints as listed
    from poros.cli.words import runs as run_words
    from poros.core import checkpoints

    every = runs.all_runs()
    ids = [r.id for r in every]
    word, step = (None if raw is True else str(raw)), None
    if word and "/weights/" in word:
        word, step = listed.parse(word)
    found = run_words.resolve(word, ids, newest=ids[0] if ids else None,
                              names=[(r.name, r.id) for r in every])
    if isinstance(found, run_words.RunMiss):
        raise Refusal("E_NOT_FOUND", "No such run", found.reason,
                      context=[", ".join(found.candidates)] if found.candidates else [],
                      fix="poros runs", note="lists them")
    run = next(r for r in every if r.id == found)
    if step == listed.FINAL:
        if run.state != "succeeded" or not checkpoints.has_weights(Path(run.output_dir)):
            raise Refusal("E_NOT_FOUND", "No finished adapter",
                          f"run {run.id} keeps no finished adapter",
                          fix=f"poros runs checkpoints {run.id}", note="lists what it kept")
        return run, run.output_dir, step
    if step is not None:
        step = listed.step_of(run, step)
        folder = listed._folders(Path(run.output_dir)).get(int(step))
        if folder is None:
            raise Refusal("E_NOT_FOUND", "No such checkpoint",
                          f"run {run.id} keeps none after step {step}",
                          fix=f"poros runs checkpoints {run.id}", note="lists them")
        return run, str(folder), step
    saved = checkpoints.list_checkpoints(Path(run.output_dir))
    if not saved:
        raise Refusal("E_NOT_FOUND", "Nothing to start from",
                      f"run {run.id} saved no checkpoint", context=[run.output_dir])
    return run, str(saved[-1][1]), None


def _inherit(a: "TrainAction", parent, rank: Optional[int]) -> None:
    """--from keeps the parent's LoRA shape: its adapter only loads into the same one."""
    import json

    try:
        saved = json.loads((Path(parent.output_dir) / "config.resolved.json").read_text())
    except (OSError, ValueError):
        return
    was = saved.get("lora_rank")
    if rank and was and rank != was:
        raise Refusal("E_USAGE", "A different rank",
                      f"run {parent.id} trained a rank {was} adapter; --rank {rank} "
                      "can't load it", context=[f"drop --rank, or train fresh at {rank}"])
    if was:
        a.rank = was
        a.provenance["rank"] = Source(kind="inferred", detail=f"from run {parent.id}")
    for key in ("lora_alpha", "lora_target_modules"):
        if saved.get(key) is not None and key not in a.config:
            a.config[key] = saved[key]


def resolve(s: Sentence, ctx: Context) -> Draft[TrainAction]:
    _refuse_setup(s, ctx)  # a setup's key: its own command, before any question
    start = _from(s)
    if not s.subject and start is not None:
        s.subject = [start[0].model]  # --from alone: the earlier run's model
    if not s.subject:
        return Draft(None, [shared.pick_model_question(ctx)])
    model, said, ask = shared.resolve_model(s.subject, ctx, verb="train")
    if ask is not None:
        return Draft(None, [ask])
    require_model(model, ctx)
    family = new_family(model, ctx, told=bool(s.flags.get("--unvalidated")))
    mac.mac_gate(model, ctx)
    if start is not None and start[0].model and model != start[0].model:
        raise Refusal("E_USAGE", "A different model",
                      f"run {start[0].id} trained {start[0].model}, not {model}",
                      context=[f"poros train --from {start[0].id} --data <file>"])
    raw = s.value("data")
    for flag in ("data", "output", "name"):  # a flag with nothing after it names nothing
        if s.value(flag) is True:
            what = {"data": "a file", "output": "a folder", "name": "a name"}[flag]
            raise Refusal("E_USAGE", "Missing a value", f"--{flag} needs {what} after it",
                          fix="poros train --help", note="every flag, with what it takes")
    if raw is None:
        return Draft(None, [data_question(ctx)])
    found = data_words.resolve(str(raw), ctx.cwd)
    if isinstance(found, data_words.Data) and found.kind == "dir":
        shared.refuse_planted(str(raw))  # a Hub dataset is asked about, and a folder would not be
    if isinstance(found, data_words.DataMiss):
        if found.near and ctx.interactive:  # a person picks the name they meant, or none
            return Draft(None, [data_question(ctx, found.near, missed=str(raw))])
        # with no person to pick, a missing file is an error, and no command it offers trains
        # on another file: its fix reads a file (the near miss it names, or one of those here),
        # which trains nothing. A person's asks which file, as `train` with no data does.
        near = [n for n in found.near
                if all(data_scan.PLAIN_NAME.fullmatch(part) for part in Path(n).parts)][:1]
        if near:
            fix, note = shlex.join(["poros", "data", near[0]]), "says what that file holds"
        elif ctx.interactive:
            fix, note = shlex.join(["poros", "train", model]), "asks which file, from those here"
        else:
            fix, note = "poros data", "the data files here, and what each holds"
        raise Refusal(
            "E_DATASET_NOT_FOUND",
            "No such data",
            found.reason,
            context=[f"the closest name here is {near[0]}"] if near else [
                "give a .jsonl, .json or .csv file, a folder of them, or a Hub dataset id"],
            fix=fix, note=note,
        )
    output = _absolute(str(s.value("output")), ctx) if s.value("output") else None
    if output and (unwritable := command.output_refusal(output)) is not None:
        raise unwritable  # before any question, and before the device is asked for
    steps, steps_src = _number(s, "steps")
    epochs, epochs_src = _number(s, "epochs")
    if steps and epochs:
        raise Refusal("E_USAGE", "Steps or epochs", "give one: --steps N, or --epochs N")
    if epochs and found.kind not in ("file", "dir"):
        raise Refusal(
            "E_USAGE",
            "Epochs need rows",
            f"{found.value} has no rows to count here",
            context=["give --steps N instead"],
        )
    config = _config_fields(s)
    _recipe(s, ctx, config)
    _store(s, ctx, config)
    seq, seq_src = _number(s, "seq_len")
    rank, rank_src = _number(s, "rank")
    block, block_src = _number(s, "block_size")
    a = TrainAction(
        model=model,
        data=found.value if found.kind in ("synthetic", "hub") else _absolute(found.value, ctx),
        steps=steps or 0,
        seq_len=seq or 0,
        rank=rank or shared.DEFAULT_RANK,
        block_size=block,
        output=output,
        name=str(s.value("name") or ""),
        config=config,
        detach=bool(s.flags.get("--detach") or s.flags.get("-d")),
        yes=bool(s.flags.get("--yes") or s.flags.get("-y")),
        download=bool(s.flags.get("--download")),
        force=bool(s.flags.get("--force")),
        overwrite=bool(s.flags.get("--overwrite")),
        allow_unvalidated=bool(s.flags.get("--allow-unvalidated")),
    )
    if family is not None:
        a.family, a.family_status = family.model_type, family.status
        if s.flags.get("--unvalidated") or family.status == "user_asserted":
            a.unvalidated = family.model_type
    if start is not None:
        from poros.preflight.disk import output_over_start

        # --output on the run it starts from: before any question, and before anything is cleared
        if (over := output_over_start(a.output, start[0].output_dir, start[1])) is not None:
            raise over
        a.from_run, a.from_step = start[0].id, start[2]
        a.config["init_adapter"] = start[1]
        a.given("from_run", f"--from {start[0].id}" + (f"/weights/{start[2]}" if start[2] else ""))
        _inherit(a, start[0], rank)
    from poros.cli.ui import simulate

    if a.detach and simulate.enabled():  # before any run folder exists
        raise Refusal("E_USAGE", "Not simulated",
                      "a simulated run is foreground only", context=["drop --detach"])
    held = s.value("eval")
    if held is not None:
        hit = data_words.resolve(str(held), ctx.cwd)
        if held is True or isinstance(hit, data_words.DataMiss) or hit.kind != "file":
            raise Refusal("E_DATASET_NOT_FOUND", "No such eval file",
                          f"--eval {held if held is not True else ''}".strip(),
                          context=["held-out rows, the training data's shape"])
        a.config["eval_dataset"] = _absolute(hit.value, ctx)
    a.provenance["model"] = Source(kind="given", detail=said)
    a.given("data", s.clauses["data"].said if "data" in s.clauses else f"--data {raw}")
    if s.value("name"):
        a.given("name", f"as {s.value('name')}")
    if s.value("gpu"):
        raise Refusal(
            "E_USAGE",
            "Train runs here",
            "--gpu plans for another card; train always runs on this machine's GPU",
            context=["poros recipes lists the machines each certified run used"],
        )
    if a.detach and a.output:
        from poros.core.env import under_agent

        # what to drop, as it was said: --output or output=. `poros agents train` detaches
        # unless it is told --wait, so there --detach is a word the caller never wrote
        said = "output=" if "output" in s.clauses and "--output" not in s.flags else "--output"
        raise Refusal(
            "E_USAGE",
            "Detached runs pick their folder",
            "a run that returns at once trains in its own folder, which its result names"
            if under_agent() else "a --detach run trains in its job's own folder",
            context=[f"drop {said}, or add --wait to train into that folder" if under_agent()
                     else f"drop {said}, or drop --detach to train into that folder"],
        )
    if epochs:
        a.steps, a.epochs = 0, epochs
        a.provenance["steps"] = epochs_src
    for name, src in (
        ("steps", steps_src),
        ("seq_len", seq_src),
        ("rank", rank_src),
        ("block_size", block_src),
    ):
        if src is not None:
            a.provenance[name] = src
    a.provenance.setdefault("rank", Source(kind="default"))
    return Draft(a)


def inspect(a: TrainAction, ctx: Context) -> TrainAction:
    """The facts the confirm screen shows: rows, steps, length, the fit verdict."""
    from poros.core.config import PorosConfig

    a = a.model_copy(deep=True)
    epochs = a.epochs or 0
    scan = data_scan.scan(a.data, ctx.cwd)
    a.rows = scan.rows
    fields = PorosConfig.model_fields

    if scan.rows == 0:
        raise Refusal("E_DATASET", "No rows", f"{a.data} has no rows to train on")
    if Path(a.data).is_file():  # read as `data` reads it, before a run is queued on it
        data_scan.refuse_stops(a.data)
    from poros import recipes

    try:
        told = PorosConfig(dataset=a.data, **a.config)  # each setting as the config reads it
    except ValueError as exc:  # learning_rate=fast: the setting's fault, by name
        said = exc.errors()[0] if hasattr(exc, "errors") else {}
        raise Refusal("E_CONFIG", "Bad setting", ": ".join(
            [".".join(map(str, said.get("loc", ()))), said.get("msg", str(exc))]).lstrip(": "))

    picked = {k: a.config[k] for k in ("recipe", "loop", "train_on", "dpo_beta") if k in a.config}
    try:
        choice = recipes.choose(PorosConfig(dataset=a.data, **picked))
    except ValueError as exc:
        fix, note = "poros recipes", "the recipes and the data each reads"
        other = {"holds preference pairs": "dpo", "needs preference pairs": "sft"}
        hit = next((r for k, r in other.items() if k in str(exc)), None)
        if hit:
            fix = f"poros train {shlex.quote(a.model)} --data {shlex.quote(a.data)} --recipe {hit}"
            note = f"trains it with {hit.upper()}"
        raise Refusal("E_CONFIG", "Can't train this", str(exc), fix=fix, note=note)
    data_scan.same_shape(a.config.get("eval_dataset"), choice.shape)
    preset = choice.settings
    row = mac.mac_row(a.model, ctx, seq=a.seq_len or None,
                         tokens=None if a.seq_len else scan.p95_tokens)
    if row:
        from poros.metal.table import defaults

        preset = {**preset, **defaults(row)}
    a.preset = dict(preset)

    def setting(name: str):
        """What the run trains at: yours, else the recipe's, else the config's."""
        if name in a.config:
            return a.config[name]
        return preset.get(name, fields[name].default)

    batch = int(setting("gradient_accumulation_steps")) * int(setting("per_device_batch_size"))
    a.batch = batch
    a.recipe, a.shape = choice.label, choice.shape
    a.lr = float(setting("learning_rate"))
    if not epochs and not a.steps and scan.rows:  # num_epochs=, else the loop's passes
        epochs = (told.num_epochs if "num_epochs" in a.config else 0) or choice.epochs or 0
    a.provenance["recipe"] = Source(
        kind="given" if choice.source == "set" else "inferred",
        detail=("--recipe" if choice.source == "set" else "a test run" if a.data == "synthetic"
                else _DATA_SAYS.get(choice.shape or "", "")),
    )
    noun = {"chats": "chats", "pairs": "pairs"}.get(a.shape or "", "rows")
    held = trained = 0
    if a.data != "synthetic" and scan.rows:
        from poros.data import held_count

        # an epoch is a pass over the rows it trains on: the held-out rows score, never train.
        # The engine's own loop holds none out: it trains every row, and scores only a file.
        held = 0 if a.config.get("eval_dataset") or choice.name == "classic" else held_count(
            scan.rows, float(setting("eval_split")), int(setting("eval_rows")))
        trained = max(1, scan.rows - held)
        a.per_epoch, a.held = max(1, math.ceil(trained / batch)), held
    if a.config.get("eval_dataset"):
        a.held = data_scan.scan(a.config["eval_dataset"], ctx.cwd).rows
    if not a.steps:
        if a.data == "synthetic":
            a.steps = row.synthetic_steps if row else SYNTHETIC_STEPS
            a.provenance["steps"] = Source(kind="default", detail="a test run")
        elif scan.rows:
            per_epoch = a.per_epoch
            a.passes = epochs or (1 if per_epoch <= STEP_CAP else 0)
            a.steps = per_epoch * epochs if epochs else min(per_epoch, STEP_CAP)
            out = f", {held} held out" if held else ""
            why = (
                f"{epochs} epoch{'s' * (epochs != 1)} of {trained:,} {noun}{out}"
                if epochs
                else (
                    f"one epoch of {trained:,} {noun}{out}"
                    if per_epoch <= STEP_CAP
                    else f"one epoch is {per_epoch:,}; capped"
                )
            )
            a.provenance["steps"] = Source(kind="inferred", detail=why)
        else:
            a.steps = STEP_CAP
            a.provenance["steps"] = Source(kind="default", detail="rows unknown; capped")
    if not a.seq_len:
        a.seq_len = row.seq if row else scan.seq_len or shared.DEFAULT_SEQ
        a.provenance["seq_len"] = (
            Source(kind="inferred", detail="95% of your rows are no longer")
            if scan.seq_len
            else Source(kind="default")
        )
    if not a.name:
        base = derived_name(a)
        from poros import runs

        taken = {r.name for r in runs.all_runs()}
        a.name = next(n for n in [base] + [f"{base}-{i}" for i in range(2, 999)]
                      if n not in taken)
    dropout = setting("lora_dropout")
    if row:
        a.disk = mac.mac_store(a.model, a.config.get("store_path"))
        a.kept = scan.kept(a.seq_len) if scan.chars else None
        from poros import runs

        a.room = mac.run_room(a.model, a.rank, int(setting("save_total_limit")),
                                 Path(a.output or mac.run_files(a.disk) or runs.root()))

    a.plan_kw = dict(rank=a.rank, dropout=float(dropout), store=a.config.get("store", "ram"),
                     store_path=a.disk or a.config.get("store_path"), steps=a.steps if row else 0)
    if ctx.unified and a.config.get("store") == "ram":  # a Mac told to hold every layer
        a.plan_kw["held"] = True
    # the run's settings its plan reads (shared.PLAN_KEYS): yours, else the recipe's
    said = a.plan_said = {**{k: preset[k] for k in shared.PLAN_KEYS if k in preset}, **a.config,
                          **({"block_size": a.block_size} if a.block_size else {})}

    def plan(seq_len: int) -> PlanOutput:
        p = shared.plan_for(a.model, ctx, seq_len=seq_len, download=True, path_settings=said,
                            **a.plan_kw)
        if a.config.get("path") not in (None, ""):
            return p
        # a path left unset: the one the run takes, where host RAM decides it
        took = shared.auto_path(p, lambda path: shared.plan_for(
            a.model, ctx, seq_len=seq_len, config_json=True,
            path_settings={**said, "path": path}, **a.plan_kw))
        return p if took is p else took.model_copy(update={"download": p.download})

    a.plan = plan(a.seq_len)
    if a.config.get("store", "ram") != "ram" and a.plan.entry != "moe" and not ctx.unified:
        raise Refusal("E_USAGE", "RAM only", f"{a.model.split('/')[-1]} is a dense model; "
                      "only MoE experts stream from an SSD",
                      context=["drop --store: dense models train from RAM"])
    # not ``said``: plan() above reads that name (the plan's settings) when fitted_length calls it
    length_from = a.provenance.get("seq_len")
    if a.plan.fits is False and (length_from is None or length_from.kind != "given"):
        # Your rows are longer than this card trains: the longest length that
        # fits, measured or bounded, rather than a run that won't start.
        seq, p = shared.fitted_length(a.seq_len, plan)
        if seq != a.seq_len:
            a.seq_len, a.plan = seq, p
            a.provenance["seq_len"] = Source(
                kind="inferred",
                detail="the longest the check does not refuse on this GPU; longer rows are cut")
    if a.plan.entry == "moe":  # what the MoE path refuses, refused here in its own words
        _moe_refusal(a, ctx)
    from poros.cli.sheets.fit import certified_run, run_settings

    a.certified = certified_run(a.model, run_settings(a), ctx, (a.recipe or "").lower())
    return a


def _moe_refusal(a: TrainAction, ctx: Context) -> None:
    """A setting the MoE path refuses at its load (poros.preflight.start.moe_settings_refusal:
    a block_size, a LoRA bias, ...), raised before the card, in that refusal's words."""
    from poros.cli.verbs.train_run import config_dict
    from poros.core.config import PorosConfig
    from poros.preflight.start import moe_settings_refusal

    try:
        cfg = PorosConfig(**config_dict(a, str(ctx.cwd / "out")))
    except Exception:  # the run's own config check says what is wrong with it
        return
    err = moe_settings_refusal(cfg, ctx.unified)
    if err is not None:
        raise err


def _ssd_room(a: TrainAction, ctx: Context, p: PlanOutput) -> Optional[float]:
    """For a MoE run on the RAM store short of host RAM: the SSD store's peak host RAM where the
    check passes it there (as poros.preflight.fit.ssd_alternative asks), else None: what lets
    the refusal's words offer the SSD store only where it helps."""
    if p.entry != "moe" or a.config.get("store", "ram") != "ram" or p.host_fits is not False:
        return None
    try:
        q = shared.plan_for(a.model, ctx, seq_len=a.seq_len, rank=a.rank, store="ssd",
                            store_path=a.config.get("store_path") or "ssd", config_json=True,
                            path_settings={**a.plan_said, "store": "ssd"})
    except Exception:  # advice only: never the reason a card fails
        return None
    return q.peak_host_ram.value if q.host_fits is not False else None


def _open(a: TrainAction, ctx: Context, p: PlanOutput) -> tuple:
    """The paths the check refuses neither for host RAM nor the GPU, for a run that set path:
    efficient by hand (poros.preflight.fit.open_paths, on the card's plan); else ()."""
    if a.config.get("path") != "efficient" or p.host_fits is not False:
        return ()
    out = []
    for alt in ("fast", "sequential"):
        try:
            q = shared.plan_for(a.model, ctx, seq_len=a.seq_len, config_json=True,
                                path_settings={**a.plan_said, "path": alt}, **a.plan_kw)
        except Exception:  # advice only: never the reason a card fails
            continue
        if q.host_fits is not False and q.fits is not False:
            out.append(alt)
    return tuple(out)


def derived_name(a: TrainAction) -> str:
    """A run's name from its model, data and date: plain, as a file's name is anyone's words."""
    from datetime import datetime

    entry = model_words.by_id(a.model)
    short = (entry.short if entry else a.model.split("/")[-1]).lower()
    data = Path(a.data).stem if a.data != "synthetic" else "synthetic"
    base = f"{short}-{data.split('/')[-1]}-{datetime.now():%m%d}"[:53]
    return re.sub(r"[^\w.-]+", "-", base)
# a setting a run can take from the nearest receipt, and its flag
MATCHABLE = {"seq_len": "--seq-len", "lora_dropout": "lora_dropout=", "lora_rank": "--rank"}


def matched(a: TrainAction, ctx: Context) -> dict[str, tuple[str, str]]:
    """{flag: (value, why)} that turn an unknown fit into a known one: the
    nearest receipt's length, dropout or rank, for settings nobody chose.
    Never a shorter length than the data asked for: that would cut its rows."""
    p = a.plan
    if p is None or shared.fit_note(p) == "fits" or p.nearest is None:
        return {}
    differs = {k: v for k, v in p.nearest.differs.items() if k != "selfcheck"}
    chosen = {"seq_len": a.provenance.get("seq_len"), "lora_rank": a.provenance.get("rank")}
    if not differs or any(k not in MATCHABLE for k in differs):
        return {}
    if any(chosen[k] is not None and chosen[k].kind == "given" for k in differs if k in chosen):
        return {}
    if "lora_dropout" in differs and "lora_dropout" in a.config:
        return {}
    to = {k: v[1] for k, v in differs.items()}
    if to.get("seq_len", a.seq_len) < a.seq_len:
        return {}
    dropout = to.get("lora_dropout", a.config.get("lora_dropout"))
    near = shared.plan_for(a.model, ctx, seq_len=to.get("seq_len", a.seq_len),
                           rank=to.get("lora_rank", a.rank), download=True,
                           dropout=None if dropout is None else float(dropout),
                           path_settings=a.plan_said or a.config)
    if shared.fit_note(near) != "fits":
        return {}
    why = "matched to settings the check knows"
    return {MATCHABLE[k]: (f"{v:g}" if isinstance(v, float) else str(v), why)
            for k, v in to.items()}


def settle(a: TrainAction, ctx: Context) -> TrainAction:
    """A run with no card opens where the card would: at the receipt's settings."""
    match = matched(a, ctx)
    if not match:
        return a
    why = next(iter(match.values()))[1]
    if "--seq-len" in match:
        a.seq_len = int(match["--seq-len"][0])
        a.provenance["seq_len"] = Source(kind="inferred", detail=why)
    if "--rank" in match:
        a.rank = int(match["--rank"][0])
        a.provenance["rank"] = Source(kind="inferred", detail=why)
    if "lora_dropout=" in match:
        a.config["lora_dropout"] = match["lora_dropout="][0]  # as typed: config holds text
    return inspect(a, ctx)


_DATA_SAYS = {"chats": "your data is chats", "pairs": "your data is preference pairs",
              "": "your data is plain text"}


def _confirm(id: str, prompt: str, flag: str, label: str, note: str = "") -> Question:
    """A confirmation, by what it confirms: no two of a run's share an id."""
    from poros.cli import questions

    return questions.confirmation(id, prompt, flag, label, note)


def _what_fits(a: TrainAction, ctx: Context) -> str:
    """"; --seq-len 1024 fits" for a run its plan says will not fit: the longest shorter length
    that does (poros.preflight.command.shorter_that_fits), at the run's other settings; "" where
    none does."""
    if a.plan is None or a.plan.fits is not False:
        return ""

    def plan(seq: int):
        return shared.plan_for(a.model, ctx, seq_len=seq, config_json=True,
                               path_settings=a.plan_said or a.config, **(a.plan_kw or {}))

    floor = 128
    if ctx.unified:
        from poros.metal.table import LENGTHS

        floor = LENGTHS[0]
    found = command.shorter_that_fits(a.seq_len, floor, plan)
    from poros.cli.sheets.fit import plain

    return f"; {plain(command.fits_said('--seq-len', found[0], found[1]))}" if found else ""


def confirmations(a: TrainAction, ctx: Context) -> list[Question]:
    if mac.mac_row(a.model, ctx) and a.disk is None:
        return [mac.store_question(a.model)]
    out = []
    p = a.plan
    if a.family and not a.unvalidated:
        out += unvalidated_questions(a)
    uncovered = model_words.by_id(a.model) is None and not mac.mac_row(a.model, ctx)
    if uncovered and not a.allow_unvalidated and not a.family:  # a new family's says it all
        out.append(
            _confirm(
                "not_in_catalog",
                f"{a.model} is not in the catalog: no Poros receipt covers it.",
                "--allow-unvalidated",
                "Train it anyway",
                p.support if p else "",
            )
        )
    if p is not None and (p.fits is False or p.host_fits is False) and not a.force:
        from poros.cli.sheets.fit import host_said

        if p.fits is False and p.fit_basis.startswith("refused on this card: "):  # no bf16
            said = "Not a supported GPU: " + p.fit_basis.removeprefix("refused on this card: ")
        elif p.host_fits is False:  # host RAM first, in words; the GPU too where it is short
            said = (f"Not enough host RAM: {host_said(p, _ssd_room(a, ctx, p), _open(a, ctx, p))}"
                    + (". The GPU may not fit it either" if p.fits is False else "")
                    + ". Start anyway only if you can free that memory first")
        else:
            held = ("; drop store=ram to read the frozen layers from the disk"
                    if ctx.unified and a.config.get("store") == "ram" else "")
            where = "this Mac, with every layer held from RAM" if held else "the GPU"
            said = (f"This run may not fit {where} at these settings{_what_fits(a, ctx)}{held}. "
                    "Start anyway to try it: it stops with an out-of-memory error if it does not "
                    "fit")
        out.append(_confirm("wont_fit", said, "--force", "Start anyway"))
    if a.room and not a.force:
        out.append(_confirm("disk_fills", a.room, "--force", "Start anyway"))
    size = p.download.value if p is not None else None
    from poros.cli.ui import simulate

    # A simulated run loads nothing, so it downloads nothing.
    what = None if simulate.enabled() else shared.download_confirm(a.model, size, ctx)
    if what and not (a.yes or a.download):
        out.append(shared.download_question(a.model, what, "--download", "Yes, download and train",
                                            a.disk))
    data = data_words.resolve(a.data, ctx.cwd)
    if (isinstance(data, data_words.Data) and data.kind == "hub" and not simulate.enabled()
            and not (a.yes or a.download) and not shared.dataset_on_disk(a.data)):
        out.append(shared.dataset_question(a.data))
    out_dir = Path(a.output).expanduser() if a.output else None
    if out_dir is not None and _has_run(out_dir) and not a.overwrite:
        out.append(
            _confirm(
                "overwrite",
                f"{a.output} holds a previous run; this one replaces its adapter, "
                "checkpoints and events.",
                "--overwrite",
                "Overwrite it",
            )
        )
    if not out and ctx.interactive and not a.yes and not ctx.sheet:
        out.append(start_question())
    return out


def start_question() -> Question:
    return Question(
        id="start",
        prompt="Start this run?",
        default="start",
        choices=[
            Choice(value="start", label="Start"),
            Choice(value="change", label="Change a setting"),
        ],
    )


def _has_run(path: Path) -> bool:
    """Anything a previous run left: an adapter, a checkpoint, its events."""
    from poros.core import checkpoints

    return path.is_dir() and (
        checkpoints.has_weights(path)
        or any(path.glob("checkpoint-*"))
        or (path / "events.jsonl").exists()
    )


def preview(a: TrainAction, w: int) -> list[Line]:
    from poros.cli import views

    return views.train_ready(a, w)


def edit_questions(a: TrainAction, ctx: Context):
    from poros.cli.verbs import train_edit

    return train_edit.field_question(a), lambda field: train_edit.value_question(field, a, ctx)


def run(a: TrainAction, ctx: Context) -> TrainOutput:
    from poros.cli.verbs import train_run

    return train_run.run(a, ctx)


def view(r: Any, w: int) -> list[Line]:
    from poros.cli import views
    from poros.cli.schemas import JobOutput

    if isinstance(r, JobOutput):
        return views.job_submitted(r.job_id, r.state, r.output_dir, w)
    return []  # the live screen and the summary were drawn while it ran
