"""One settings sheet per command: the rows, and what each change means.

A form turns the sheet's values into a command line, resolves it with the
verb's own ``resolve``, ``inspect`` and ``confirmations`` (so the sheet can
never disagree with what the command would do), and reads every row back
from the resolved action with its source. Nothing here runs a command: the
sheet hands back a command line, and poros/cli/app.py runs it like any other.

This module holds the sheets' machinery and the train and fit sheets; ``sheets.fit`` what they
say of the fit, ``sheets.listings`` the lists they browse, ``sheets.export`` the export and
prove sheets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from poros.cli.context import Context
from poros.cli.sheets.common import (
    LENGTHS,
    OVERRIDES,
    _eval_files,
    _int,
    _model_options,
    _ready,
    _resolved,
    _short,
    _warnings,
    _words,
    command_line,
    form_argv,
    initial,
)
from poros.cli.sheets.export import _home, eval_form, export_form, prove_form
from poros.cli.sheets.fit import (
    OFFLOAD,
    _matched,
    _offload_said,
    _path_row,
    _train_fit,
    recommended,
)
from poros.cli.sheets.listings import (
    checkpoints_listing,
    data_listing,
    models_listing,
    recipes_listing,
    runs_listing,
)
from poros.cli.ui import sheet as sh
from poros.cli.ui.sheet import Field, Form, Option, Value, View
from poros.cli.ui.text import S
from poros.cli.verbs import mac
from poros.cli.words.grammar import Sentence, parse

# The sheets' API: this module's forms and saved defaults, and the lists (listings) and the
# export and prove sheets (export). A private name is imported from the module that defines it.
__all__ = ["FORMS", "LISTS", "SAVED", "checkpoints_listing", "config_fields", "data_listing",
           "export_form", "models_listing", "open_sheet",
           "prove_form", "recipes_listing", "recommended", "runs_listing", "save_defaults",
           "save_recipe", "saved_defaults", "train_form", "wanted"]


def wanted(verb: str, s: Sentence) -> bool:
    """Whether a terminal command opens its sheet: whenever it would start
    work, and for a read-only verb only when its subject is missing."""
    if s.flags.get("--yes") or s.flags.get("-y") or s.flags.get("--help"):
        return False
    if verb in ("train", "prove"):
        return True
    if verb == "export":
        return not (s.subject and s.value("format"))
    if verb == "data":
        return not s.subject
    if verb == "eval":  # a run, its weights, the rows and the answers, each a pick
        return not s.subject
    if verb in ("models", "runs", "recipes"):  # --limit asks for the printed list
        return ((s.action is None or (verb, s.action) == ("runs", "report")) and not s.subject
                and "--limit" not in s.flags)
    return False


def open_sheet(verb: str, s: Sentence, ctx: Context) -> Optional[List[str]]:
    """The command line the person settled on, or None if they cancelled.
    Chained cards (a recipe, then its train card) share one canvas."""
    with sh.fullscreen():
        return _open(verb, s, ctx)


def _open(verb: str, s: Sentence, ctx: Context) -> Optional[List[str]]:
    if verb in LISTS:  # `runs report` with no run named too: pick one, see its page
        if verb == "recipes":  # /models' card, its tabs underlined and its foot on one edge
            from poros.cli import recipes_card

            argv = sh.pick(LISTS[verb](ctx), lister=recipes_card.Lister)
        else:
            argv = sh.pick(LISTS[verb](ctx))
        if argv and argv[0] in FORMS:  # a recipe or a model fills the train card: you
            _retitle(argv[0])           # settle it there, in the same tab
            return _open(argv[0], parse(argv), ctx)
        if argv and argv[:2] == ["runs", "checkpoints"]:  # a run's checkpoints: their card next
            return sh.pick(checkpoints_listing(argv[2]))
        return argv
    if verb == "export" and not s.subject:  # nothing finished: say so, as /runs does
        empty = _nothing_to_export()
        if empty:
            return sh.pick(sh.Listing(tag="export", title="Export a run", head=[], items=[],
                                      empty=empty))
    if ctx.unified and verb == "train":
        s = _likeliest(s, ctx)
    form = FORMS[verb](s, ctx)
    values = sh.run(form)
    return None if values is None else form_argv(verb, values)


def _likeliest(s: Sentence, ctx: Context) -> Sentence:
    """A Mac's ``train qwen 0.5b``: the card opens on the one you have, an instruct model before
    a base one, in view and one key from changing."""
    from poros.cli.words.models import by_name, matching

    said = " ".join(s.subject)
    hits = mac.trainable(ctx, matching(said)) if said and not by_name(said) else []
    if hits:
        best = max(hits, key=lambda e: (e.id in ctx.cached, "instruct" in e.short.lower()))
        s.subject = [best.short]
    return s


def _nothing_to_export() -> str:
    """Why /export has nothing to offer, or "" when a run has finished."""
    from poros import runs

    every = runs.all_runs()
    if any(r.state == "succeeded" for r in every):
        return ""
    return ("no finished runs yet: /runs shows how they're going" if every
            else "no runs yet: /train starts one")


def _retitle(verb: str) -> None:
    """The session's tab says the card it shows now (``recipes`` became ``train``)."""
    from poros.cli.ui import tui

    tab = getattr(tui.local, "tab", None)
    if tab is not None:
        tab.title = verb


# --------------------------------------------------------------------------
# values <-> command lines
# --------------------------------------------------------------------------


def _src(action, name: str) -> str:
    if action is None:
        return ""
    src = action.provenance.get(name)
    return str(src) if src is not None else "default"


def _dropout(text: str) -> Optional[str]:
    try:
        v = float(text)
    except ValueError:
        return "dropout is a number like 0.05"
    return None if 0 <= v < 1 else "dropout is at least 0 and below 1"


# --------------------------------------------------------------------------
# every other training setting: the "more" row and saved defaults
# --------------------------------------------------------------------------

# Rows of their own, or never offered here: trust_remote_code would run code
# from the model's repo, and Poros downloads weights only.
HIDDEN = frozenset(
    "model_name_or_path dataset max_steps max_seq_length lora_rank output_dir block_size "
    "num_epochs resolved_base_model_path trust_remote_code progress "
    "store store_path sss_ram".split()  # the experts row, in words; --sss-ram for agents
)


def config_fields(ctx: Optional[Context] = None) -> Dict[str, object]:
    from poros.core.config import FIXED_FIELDS, PorosConfig

    fields = {n: f for n, f in PorosConfig.model_fields.items()
              if n not in HIDDEN and n not in FIXED_FIELDS}
    if ctx is not None and ctx.unified:  # what a Mac honours; the table says the rest
        from poros.metal import ALLOWED

        return {n: f for n, f in fields.items() if n in ALLOWED or n == "save_steps"}
    return {n: f for n, f in fields.items() if n != "memory_guard"}  # a Mac's alone


def _choices(f) -> Optional[List[str]]:
    """A Literal's words or a bool's yes/no; None for a free value."""
    import typing

    t = f.annotation
    if t is bool:
        return ["true", "false"]
    if typing.get_origin(t) is typing.Literal:
        return [str(a) for a in typing.get_args(t)]
    return None


def _typed(name: str, f) -> Callable[[str], Optional[str]]:
    t = f.annotation

    def check(text: str) -> Optional[str]:
        try:
            if t is int:
                int(text)
            elif t is float:
                float(text)
        except ValueError:
            return f"{name} is {'a whole number' if t is int else 'a number'}"
        return None

    return check


FLAG_ROWS = (("--rank", "rank", "LoRA", "adapter size; more learns more, uses more memory",
              "16"),
             ("--name", "name", "saving & checks", "what /runs lists it as", ""),
             ("--output", "output", "saving & checks", "the folder the adapter goes to",
              "~/.poros/runs/<id>"))


def _now(name: str, f, a, values: Dict[str, Value]) -> str:
    """What the run trains at for ``name``: yours, else the recipe's, else the config's."""
    if values.get(f"{name}=") not in (None, ""):
        return str(values[f"{name}="])
    v = a.preset[name] if a is not None and name in a.preset else f.default
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        from poros.cli.views import lr_text

        return lr_text(v) if v and abs(v) < 1e-3 else f"{v:g}"  # 2e-4, never 0.0002
    return "" if v is None else str(v)


def _said(value: str) -> str:
    """A value as the list reads it: an unset field is "auto", not a blank."""
    return {"": "auto", "true": "yes", "false": "no"}.get(value, value)


# a setting only one kind of data reads: the others list it under "advanced"
RELEVANT = {"dpo_beta": "pairs", "train_on": "chats"}
# settings the train card's own rows set (held out, checkpoints): never a row of their own too
OWN_ROWS = frozenset({"eval_split=", "eval_rows=", "save_steps="})


def _more_options(values: Dict[str, Value], a=None, ctx=None) -> Callable[[], List[Option]]:
    """Every setting not on the card yet, grouped and in words: what it is
    called, what it trains at now, and what changing it does."""
    from poros.cli.train_settings import GROUPS, SETTINGS, setting

    def opts() -> List[Option]:
        out = [Option(k, label, about, sets="", group=group, now=now)
               for k, label, group, about, now in FLAG_ROWS if k not in values]
        for n, f in config_fields(ctx).items():
            if f"{n}=" in values or f"{n}=" in OWN_ROWS:
                continue
            st = setting(n, f.description or "")
            now = _now(n, f, a, values)
            off = a is not None and a.shape and RELEVANT.get(n, a.shape) != a.shape
            said = _said(now)
            if n == "path" and now == "" and a is not None and a.plan is not None and (
                    a.plan.path and "path" not in a.config):
                said = f"auto → {a.plan.path}"  # the loop a run of the card takes
            if n == OFFLOAD and now == "" and a is not None and a.plan is not None and (
                    a.plan.path and OFFLOAD not in a.config):
                said = _offload_said(a.plan)  # where a run of the card keeps its tables
            out.append(Option(f"{n}=", st.label, st.about, sets=now,
                              group="advanced" if off else st.group,
                              now=said, words=f"{n} {st.words}"))
        order = {f"{st.name}=": i for i, st in enumerate(SETTINGS)}
        return sorted(out, key=lambda o: (GROUPS.index(o.group), order.get(o.value, len(order))))

    return opts


def _setting_rows(values: Dict[str, Value], a=None, ctx=None) -> List[Field]:
    """A row for each ``name=value`` setting the person added, its choices in words; the path
    row says what an unset path takes, and what each path needs on this card."""
    from poros.cli.train_settings import setting

    fields = config_fields()
    out = []
    for key in values:
        if not key.endswith("=") or key in OWN_ROWS:
            continue
        name = key[:-1]
        f = fields.get(name)
        st = setting(name, (f.description or "") if f is not None else "")
        words = _choices(f) if f is not None else None
        now = _now(name, f, a, {}) if f is not None else ""
        picks = list(st.choices) or [(w, "") for w in (words or ())]
        if picks and now not in [v for v, _ in picks]:
            picks.insert(0, (now, ""))
        v = values[key]

        def options(picks=picks, now=now) -> List[Option]:
            return [Option(val, _said(val), " · ".join(x for x in (
                "what it trains at" if val == now else "", note) if x)) for val, note in picks]

        shown, source = _said(str(v or "")), "default" if str(v) == now else "given"
        if (name == "path" and ctx is not None and not ctx.unified and a is not None
                and a.plan is not None and a.plan.path is not None):
            said, why, options = _path_row(a, ctx, now)
            if v in (None, ""):
                shown, source = said, why or source
        if (name == OFFLOAD and v in (None, "") and ctx is not None and not ctx.unified
                and a is not None and a.plan is not None and a.plan.path is not None):
            shown, source = _offload_said(a.plan), a.plan.offload_reason or source
        out.append(
            Field(
                key, _label(st.label),
                kind="pick" if picks else "text", value=v, shown=shown,
                free=bool(picks) and not words,
                source=source,
                options=options if picks else None,
                check=None if words or f is None else _typed(name, f),
                hint=st.about,
            )
        )
    return out


def _label(name: str) -> str:
    """``lora_dropout`` reads as "dropout": the label column is 14 wide at most."""
    said = {"learning_rate": "lr", "num_epochs": "epochs", "dpo_beta": "beta",
            "gradient_accumulation_steps": "grad accum", "per_device_batch_size": "batch"}
    words = said.get(name) or name.removeprefix("lora_").replace("_", " ")
    return words if len(words) <= 14 else words[:13] + "…"


SAVED = "train_defaults"
UNSAVED = ("subject", "--data", "--name", "--output", "--store", *OVERRIDES)


def saved_defaults() -> Dict[str, Value]:
    from poros.cli.questions import preference

    try:
        got = json.loads(preference(SAVED) or "{}")
    except ValueError:
        return {}
    return got if isinstance(got, dict) else {}


def save_defaults(values: Dict[str, Value]) -> str:
    from poros.cli.questions import remember

    keep = {k: v for k, v in values.items() if k not in UNSAVED and v not in (None, "", False)}
    remember(SAVED, json.dumps(keep, sort_keys=True))
    n = len(keep)
    return f"saved {n} setting{'s' if n != 1 else ''} as your /train defaults" if n else (
        "cleared your /train defaults"
    )


def save_recipe(values: Dict[str, Value], ctx: Optional[Context] = None) -> str:
    """The card's setup as one of your recipes: its model, the kind of data
    it trains on (not the file), the loop and every setting you changed."""

    from poros.cli.verbs.data import shape_of
    from poros.cli.words.models import catalog
    from poros.recipes import presets, rules

    model = str(values.get("subject") or "")
    if not model:
        return "pick a model first: a recipe starts from one"
    e = next((x for x in catalog() if model in (x.id, x.short)), None)
    data = str(values.get("--data") or "")
    shape = shape_of(Path(data)) if data and Path(data).exists() else "unknown"
    kind = {"convertible": "qa", "chats": "chats", "pairs": "pairs", "text": "text"}.get(shape,
                                                                                       "any")
    flags = {v: k for k, v in presets.FLAGS.items()}
    settings: Dict[str, object] = {}
    for k, v in values.items():
        if v in (None, "", False):
            continue
        if k in flags:
            settings[flags[k]] = int(v) if str(v).isdigit() else v
        elif k.endswith("="):
            settings[k[:-1]] = _toml_value(str(v))
    a = _resolved("train", values, ctx)[0] if ctx is not None else None
    if a is not None:  # what the card shows, not only what was changed on it
        settings.setdefault("max_seq_length", a.seq_len)
        settings.setdefault("lora_rank", a.rank)
        if a.lr:
            settings.setdefault("learning_rate", a.lr)
    short = e.short if e else model.rsplit("/", 1)[-1]
    name = str(values.get("--name") or "")
    key = re.sub(r"[^a-z0-9.]+", "-", (name or f"{short} {kind}").lower()).strip("-.")
    loop = str(values.get("--recipe") or ("dpo" if kind == "pairs" else "sft"))
    try:
        path = presets.save(presets.Recipe(key, name, "you", "", e.id if e else model, kind, loop,
                                           rules.pick(rules.SETTINGS, settings), source="yours"))
    except (OSError, ValueError):
        return "a recipe needs a public Hub model, and a name of one plain word"
    return f'saved as your recipe "{path.stem}" · /recipes · {_home(str(path))}'


def _toml_value(text: str) -> object:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(text.lower(), text)


def _data_options(ctx: Context) -> Callable[[], List[Option]]:
    def opts() -> List[Option]:
        from poros.cli.verbs.train_data import local_files

        out = [Option(f, f, "here") for f in local_files(ctx.cwd)[:30]]
        dirs = sorted(
            p.name + "/"
            for p in ctx.cwd.iterdir()
            if p.is_dir() and not p.name.startswith(".") and _has_data(p)
        )
        out += [Option(d, d, "a folder of data files") for d in dirs[:10]]
        from poros.cli.ui import simulate

        if simulate.enabled():  # a preview has its own data, so its checkpoints are scored
            from poros.cli.ui import preview_data

            out[:0] = [
                Option(str(preview_data.chats()), preview_data.CHATS,
                       f"made-up chats, {preview_data.CHAT_ROWS} · trains SFT"),
                Option(str(preview_data.pairs()), preview_data.PAIRS,
                       f"made-up pairs, {preview_data.PAIR_ROWS} · trains DPO"),
            ]
        return out

    return opts


_RECIPE_SAID = {"": "auto", "sft": "SFT", "dpo": "DPO", "ipo": "IPO", "kto": "KTO", "orpo": "ORPO",
                "simpo": "SimPO"}


def _rows_fact(rows: Optional[int], shape: Optional[str]) -> str:
    if not rows:
        return ""
    noun = {"chats": "chat", "pairs": "pair"}.get(shape or "", "row")
    return f"{rows:,} {noun}{'s' if rows != 1 else ''}"


def _recipe_options(ctx: Context, a) -> Callable[[], List[Option]]:
    """auto, SFT, the pair loops, then any loop file here (a .py with ``def main(``)."""
    def opts() -> List[Option]:
        auto = f"from your data: {a.recipe}" if a is not None and a.recipe else "from your data"
        out = [
            Option("", "auto", auto),
            Option("sft", "SFT", "chats or text · learns the answers"),
            Option("dpo", "DPO", "prompt, chosen, rejected · learns the preference"),
            Option("ipo", "IPO", "pairs · DPO that cannot overfit clean pairs"),
            Option("kto", "KTO", "pairs · steadier where the pairs are noisy"),
            Option("orpo", "ORPO", "pairs · SFT and preference at once, from a base model"),
            Option("simpo", "SimPO", "pairs · no reference pass, long answers win no points"),
        ]
        for p in sorted(ctx.cwd.glob("*.py"))[:20]:
            try:
                if "def main(" in p.read_text(errors="ignore")[:20000]:
                    out.append(Option(p.name, p.name, "your loop"))
            except OSError:
                continue
        return out

    return opts


def _lengths(a, ctx: Context, notes: Dict[str, str] = {}) -> Callable[[], List[Option]]:
    """The lengths worth offering: on a Mac, the ones in this model's row of its table."""
    if a is None or not ctx.unified:
        return _words(LENGTHS, notes)
    return _words([str(seq) for seq in mac.lengths(a.model, ctx)])


def _has_data(path) -> bool:
    from poros.cli.verbs.train_data import SUFFIXES

    try:
        return any(p.suffix.lower() in SUFFIXES for p in path.iterdir())
    except OSError:
        return False


def _gpu_options() -> List[Option]:
    from poros.core.card import GPU_SPECS_MIB

    out = [Option("", "this machine", "the GPU Poros finds here")]
    for name, mib in GPU_SPECS_MIB.items():
        out.append(Option(name, name, f"{mib / 1024:.0f} GiB"))
    return out


def _disk_rows(a, values: Dict[str, Value]) -> List[Field]:
    """A Mac picks RAM or SSD itself. A row only when more than one disk can take the model."""
    if a is None or a.plan is None or (a.disk and not values.get("--store")):
        return []
    from poros.metal import store

    return [Field(
        "--store", "disk", value=values.get("--store"), required=True, free=True,
        shown=_home(str(values.get("--store") or "")), source="where the model goes",
        options=lambda: [Option(str(v.store), v.label, f"{v.free / 1e9:.0f} GB free")
                         for v in store.qualifying(a.model)],
        hint="pick one, or type a folder",
    )]


def _store_rows(a, values: Dict[str, Value], ctx: Context) -> List[Field]:
    """Where a MoE model's experts live: RAM, or streamed from a file on an SSD.
    A dense model's row says it trains from RAM: the choice is always in view."""
    if ctx.unified:
        return _disk_rows(a, values)
    if a is None or a.plan is None:
        return []
    if a.plan.entry != "moe" and values.get("--store") in (None, "", "ram"):  # no experts here
        return [Field(
            "--store", "experts", value=values.get("--store") or None, shown="in RAM", quiet=True,
            source="dense model: SSD streaming is for MoE",
            options=lambda: [Option(None, "in RAM", "dense models train from RAM")],
            hint="only a MoE model's experts stream from an SSD",
        )]
    ssd = a.config.get("store") == "ssd"
    where = str(a.config.get("store_path") or "")
    home = str(Path.home())
    where = "~" + where[len(home):] if where.startswith(home) else where
    short = a.plan.host_fits is False  # the run's own check: its experts would not fit host RAM
    if ssd:
        said = f"streams a .sss file · {where}"
    elif short:
        said = "not enough host RAM for the experts: use SSD"
    else:
        said = "every expert in host RAM"
    return [Field(
        "--store", "experts", value=values.get("--store"), free=True,
        shown="from SSD" if ssd else "in RAM", source=said, quiet=not values.get("--store"),
        options=lambda: [Option(None, "in RAM", "fastest · every expert in host RAM"),
                         Option("ssd", "from SSD", "less host RAM · streams a .sss file")],
        check=_store_check,
        hint="in RAM, from SSD, or type a folder on a fast SSD for the .sss file",
    )]


def _n(n: int, noun: str) -> str:
    """``1 step``, ``50 steps``: a count, its noun in the number it takes."""
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


# what a held-out score reads, by the data's shape
SCORES = {"chats": "loss + endings", "pairs": "loss + prefers", "text": "loss"}
EPOCHS = (1, 2, 3, 5)
HELD = (50, 100, 200, 500)


def _settle(values: Dict[str, Value]) -> None:
    """The card's own rows into the flags and settings a run takes, in place: ``length`` is
    ``--epochs`` or ``--steps`` (whichever was picked last), ``eval`` is ``--eval FILE``,
    ``eval_rows=N`` or ``eval_split=0``."""
    if "length" in values:
        said = str(values.pop("length") or "").lower().replace(",", "").replace("_", "").strip()
        values.pop("--epochs", None)
        values.pop("--steps", None)
        if m := re.fullmatch(r"e(\d+)|(\d+) ?epochs?", said):
            values["--epochs"] = m.group(1) or m.group(2)
        elif m := re.fullmatch(r"(\d+)(?: ?steps?)?", said):
            values["--steps"] = m.group(1)
    if "eval" in values:
        said = values.pop("eval")
        for key in ("--eval", "eval_split=", "eval_rows="):
            values.pop(key, None)
        text = str(said or "").strip()
        if text == "off":
            values["eval_split="] = "0"
        elif m := re.fullmatch(r"(?:rows:)?(\d+)", text.replace(",", "")):
            values["eval_rows="] = m.group(1)
        elif text not in ("", "auto"):
            values["--eval"] = text


def _length_row(a, values: Dict[str, Value]) -> Field:
    """How long it trains: epochs, the recipe's unless you pick, or a number of steps typed."""
    steps, epochs = values.get("--steps"), values.get("--epochs")
    per = a.per_epoch if a is not None else 0
    if a is None:
        shown, fact, src = str(steps or (f"{epochs} epochs" if epochs else "")), "", ""
    elif steps or not a.passes:
        shown, fact = _n(a.steps, "step"), f"{a.steps / per:.1f} epochs" if per else ""
        src = "given" if steps else _src(a, "steps")
    else:
        n = a.passes
        shown, fact = f"{n} epoch{'s' * (n != 1)}", f"{a.steps:,} steps"
        src = ("given" if epochs else f"{a.recipe} recipe" if a.preset.get("num_epochs")
               else "default")

    def options() -> List[Option]:
        if not per:
            return []
        return [Option(f"e{n}", f"{n} epoch{'s' * (n != 1)}", f"{per * n:,} steps") for n in EPOCHS]

    return Field(
        "length", "train for", value=f"e{epochs}" if epochs else steps, free=True,
        shown=shown, fact=fact, source=src, quiet=not (steps or epochs), options=options,
        clears=("--epochs", "--steps"),
        hint="pick the passes over your data, or type a number of steps",
    )


def _text_file(a) -> bool:
    """Whether the run's data is a local file of text rows (which --recipe sft trains, holding
    5% of them out)."""
    from poros import data

    path = Path(str(a.data or "")).expanduser()
    if not path.is_file() or path.suffix not in (".jsonl", ".json", ".csv"):
        return False
    try:
        return data.detect(path) == "text"
    except (OSError, ValueError):
        return False


def _eval_rows(a, values: Dict[str, Value], ctx: Context) -> List[Field]:
    """What it scores and when: the held-out rows, and the checkpoints each score lands in."""
    from poros.core.config import PorosConfig
    from poros.data import FEWEST, SHARE_CAP

    fields = PorosConfig.model_fields
    file = str(values.get("--eval") or "")

    def now(name: str) -> str:
        return _now(name, fields[name], a, values)

    split, count = float(now("eval_split") or 0), int(now("eval_rows") or 0)
    rows = a.rows if a is not None else None
    shape = a.shape if a is not None else None
    noun = {"chats": "chats", "pairs": "pairs"}.get(shape or "", "rows")
    held = a.held if a is not None else None
    off = not file and (split <= 0 and not count)
    synthetic = a is not None and a.data == "synthetic"
    classic = a is not None and a.recipe == "classic"  # the engine's own loop holds none out
    if file:
        value, shown = file, Path(file).name
        fact, src = _rows_fact(held, shape), "your file, kept out of training"
    elif synthetic:
        value, shown, fact, src = None, "none", "", "a test run"
    elif classic:
        value, shown, fact = None, "none", ""
        src = ("this loop trains every row: give a file of held-out rows, or --recipe sft"
               if _text_file(a) else
               "this loop trains every row: give a file of held-out rows")
    elif off:
        value, shown, fact = "off", "off", ""
        src = "given" if values.get("eval_split=") else "the Mac table: on SSD a score costs steps"
    elif held == 0 and rows is not None:
        value, shown, fact = None, "none", ""
        src = f"under {FEWEST} {noun}: pick how many, or a file"
    elif count:
        value, shown = f"rows:{count}", f"{held if held is not None else count} {noun}"
        fact, src = f"of {rows:,}" if rows else "", "given"
    else:
        value, shown = None, f"{held} {noun}" if held else f"{split:.0%}"
        capped = rows is not None and held == SHARE_CAP and round(rows * split) > SHARE_CAP
        fact = f"of {rows:,}" if rows else ""
        src = f"{split:.0%}, at most {SHARE_CAP}" if capped else f"{split:.0%} of your data"

    def options() -> List[Option]:
        if classic:  # nothing of its data held out to pick: a file, or none
            return _eval_files(ctx, values, shape)
        share = split or 0.05
        out = [Option("auto", f"{share:.0%} of your {noun}",
                      f"at most {SHARE_CAP}; none under {FEWEST}")]
        out += [Option(f"rows:{n}", f"{n} {noun}", f"{n / rows:.0%} of them")
                for n in HELD if rows and n <= rows // 2]
        out += _eval_files(ctx, values, shape)
        return out + [Option("off", "off", "trains on every row; nothing scored")]

    scores = not off and not synthetic and held != 0
    save = int(now("save_steps") or 0)
    steps = a.steps if a is not None else 0
    saves = (steps - 1) // save + 1 if save and steps else 0  # and the finished run's
    what = SCORES.get(shape or "", "loss")
    if not steps:
        shown2, fact2 = (f"every {_n(save, 'step')}" if save else "at the end"), ""
    elif not save or save >= steps:
        shown2, fact2 = "at the end", "1 checkpoint"
    else:
        keep = int(now("save_total_limit") or 0)
        shown2 = f"every {_n(save, 'step')}"
        fact2 = f"{saves}, last {keep} kept" if 0 < keep < saves else f"{saves} checkpoints"
    src2 = (f"scored: {what}" if scores and now("eval_checkpoints") == "true"
            else f"scored at the ends: {what}" if scores else "nothing held out to score")

    def saves_at() -> List[Option]:
        picks = [n for n in (25, 50, 100, 250, 500, 1000) if not steps or n < steps]
        keep = int(now("save_total_limit") or 0)

        def note(n: int) -> str:
            k = (steps - 1) // n + 1 if steps else 0
            return "" if not k else f"{k} checkpoints" + (f", last {keep} kept" if 0 < keep < k
                                                          else "")

        out = [Option(str(n), f"every {n:,} steps", note(n)) for n in picks]
        return out + [Option("0", "at the end", "one checkpoint, the finished run")]

    return [
        Field(
            "eval", "held out", value=value, free=True, gap=True, files=True,
            shown=shown, fact=fact, source=src, quiet=value is None,
            options=options, clears=("--eval", "eval_split=", "eval_rows="),
            hint="how many rows to keep back and score, or a file of them: drag it in",
        ),
        Field(
            "save_steps=", "checkpoints", value=values.get("save_steps="), free=True,
            shown=shown2, fact=fact2, source=src2, quiet=not values.get("save_steps="),
            options=saves_at, check=_int("checkpoints", 0),
            hint="steps between checkpoints; each keeps its held-out score",
        ),
    ]


def _store_check(text: str) -> Optional[str]:
    if text.startswith(("/", "~", ".")) or text in ("ram", "ssd") or text.startswith("ssd:"):
        return None
    return "ram, ssd, or a folder like /mnt/nvme"


def train_form(s: Sentence, ctx: Context) -> Form:
    def build(values: Dict[str, Value]) -> View:
        _settle(values)
        _settle(form.values)  # what is picked on a row of the card's own is a flag from now on
        a, error, asks = _resolved("train", values, ctx)
        rows = None if a is None else a.rows

        def why(key: str, name: str) -> str:  # a prefilled value says why it was picked
            if key in guessed and values.get(key) == guessed[key][0]:
                if key not in ("subject", "--data") and values.get("subject") != model0:
                    return _src(a, name)  # matched for another model
                return guessed[key][1]
            return _src(a, name)

        model = _short(a.model) if a else str(values.get("subject") or "")
        data = str(values.get("--data") or "")
        from poros.cli.views import recipe_text
        from poros.cli.words.models import instruct_twin

        twin = instruct_twin(a.model) if a and a.shape == "chats" else None

        fields = [
            Field(
                "subject", "model", value=values.get("subject"), required=True, free=True,
                shown=model, source=(f"base: on chats {twin.short} ends its answers" if twin
                                     else why("subject", "model")),
                options=_model_options(ctx, values),
                hint="a catalog name, a Hub id, or a path",
            ),
            Field(
                "--data", "data", value=values.get("--data"), required=True, free=True,
                shown=data, source=why("--data", "data"),
                fact=_rows_fact(rows, a.shape if a else None),
                options=_data_options(ctx),
                hint="pick one, or drag a file into the terminal: csv, json, jsonl",
            ),
            Field(
                "--recipe", "recipe", value=values.get("--recipe") or "", gap=True,
                shown=recipe_text(a.recipe, a.shape, a.config, a.lr) if a else _RECIPE_SAID.get(
                    str(values.get("--recipe") or ""), str(values.get("--recipe") or "")),
                source=_src(a, "recipe"), quiet=not values.get("--recipe"),
                options=_recipe_options(ctx, a),
                hint="how it learns; auto reads your data",
            ),
            *([Field(
                "--from", "from", kind="text", value=values.get("--from"),
                shown=f"run {a.from_run}'s adapter" if a and a.from_run else str(values["--from"]),
                source=_src(a, "from_run"),
                hint="a run id or last: start from its adapter, a fresh optimizer",
            )] if values.get("--from") else []),
            _length_row(a, values),
            Field(
                "--seq-len", "length", value=values.get("--seq-len"), free=True,
                shown=f"{a.seq_len:,} tokens" if a else "", source=why("--seq-len", "seq_len"),
                quiet=values.get("--seq-len") == guessed.get("--seq-len", (None,))[0],
                options=_lengths(a, ctx), check=_int("length"),
                hint="tokens per example",
            ),
            *_store_rows(a, values, ctx),
            *_eval_rows(a, values, ctx),
            Field(
                "--detach", "run", value=bool(values.get("--detach")), gap=True,
                shown="in the background" if values.get("--detach") else "watch here",
                options=lambda: [Option(False, "watch here"), Option(True, "in the background")],
                hint="in the background, /runs watches it and this terminal stays free",
            ),
        ] + [f for f in (
            Field(
                "--rank", "rank", value=values.get("--rank"), free=True,
                shown=str(a.rank) if a else "", source=why("--rank", "rank"),
                quiet=values.get("--rank") == guessed.get("--rank", (None,))[0],
                options=_words(("8", "16", "32", "64"), {"16": "the default"}),
                check=_int("rank"), hint="LoRA rank",
            ),
            Field(
                "--name", "name", kind="text", value=values.get("--name"),
                shown=a.name if a else "", hint="what /runs lists it as",
            ),
            Field(
                "--output", "output", kind="text", value=values.get("--output"),
                shown=a.output if (a and a.output) else "", hint="a folder for the adapter",
            ),
        ) if f.key in values] + [
            f for f in _setting_rows(values, a, ctx) if f.key not in guessed  # rides unseen
        ] + [
            Field(
                "more", "more", adds=True,
                shown=f"all {len(_more_options(values, a, ctx)())} settings",
                source="learning · LoRA · saving" if ctx.unified
                else "learning · LoRA · memory · saving",
                options=_more_options(values, a, ctx),
                hint="type what you want to change: learning rate, warmup, save…",
            ),
        ] + _warnings(asks, values)
        title = [S("Train ", "hi"), S(model or "a model", "hi" if model else "dm")]
        if data:
            title += [S(" on ", "dm"), S(Path(data).name or data, "hi")]
        view = View(
            title=title,
            fields=fields,
            error=error,
            go="Start training",
            command=command_line("train", values),
            **(_train_fit(a, ctx) if a is not None else {}),
        )
        view.ready = a is not None and _ready(view)
        return view

    from poros.cli.questions import preference
    from poros.cli.verbs.train_data import local_files

    values0 = {**saved_defaults(), **initial(s)}
    guessed: Dict[str, Tuple[Value, str]] = {}  # key -> (the value picked, why)
    if not values0.get("subject"):
        rec = recommended(ctx)
        if rec:
            values0["subject"] = rec[0]
            guessed["subject"] = rec
    if not values0.get("--data"):
        files = local_files(ctx.cwd)
        from poros.cli.ui import simulate

        if simulate.enabled():  # a preview opens on its own chats, wherever it was started
            from poros.cli.ui import preview_data

            values0["--data"] = str(preview_data.chats())
            guessed["--data"] = (values0["--data"], "the preview's made-up chats")
        elif files:
            values0["--data"] = files[0]
            guessed["--data"] = (files[0], "newest file here")
    model0 = values0.get("subject")
    if model0 and values0.get("--data"):
        for key, (v, why) in _matched(values0, ctx).items():
            values0[key] = v
            guessed[key] = (v, why)
    if "--detach" not in values0 and preference("train_where") == "background":
        values0["--detach"] = True
    _settle(values0)
    form = Form("train", build, values0, keys={"ctrl-s": ("save", [
        ("defaults", save_defaults), ("recipe", lambda v: save_recipe(v, ctx))])})
    return form


LISTS = {"models": models_listing, "runs": runs_listing, "recipes": recipes_listing,
         "data": data_listing}

FORMS = {
    "train": train_form,
    "eval": eval_form,
    "export": export_form,
    "prove": prove_form,
}
