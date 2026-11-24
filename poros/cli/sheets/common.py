"""What every sheet shares: the command line a sheet's values say, its first values, the warnings
it carries, the field checks and the model options.
"""

from __future__ import annotations

import importlib
import shlex
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from poros.cli.context import Context
from poros.cli.ui.sheet import Field, Option, Value, View
from poros.cli.verbs import shared
from poros.cli.words.grammar import Sentence, parse
from poros.core.refusal import Refusal

# A confirmation's override flag -> the row that asks for it.
WARN_LABEL = {
    "--download": "download",
    "--allow-unvalidated": "unvalidated",
    "--unvalidated": "not validated",
    "--force": "start anyway",  # its row says why the run may not fit
    "--overwrite": "overwrite",
}


OVERRIDES = tuple(WARN_LABEL)


LENGTHS = ("256", "512", "1024", "2048", "4096", "8192")


def command_line(verb: str, values: Dict[str, Value]) -> str:
    """The command these values run, as the user would type it here."""
    from poros.cli.ui import components as c

    return c.command(shlex.join(form_argv(verb, values)))


def initial(s: Sentence) -> Dict[str, Value]:
    values: Dict[str, Value] = {}
    if s.subject:
        values["subject"] = s.subject[0]
    if s.action:
        values["do"] = s.action
    for flag, v in s.flags.items():
        if flag not in ("--json",):
            values[flag] = v
    for key, clause in s.clauses.items():
        values[f"{key}="] = str(clause.value)
    return values


def form_argv(verb: str, values: Dict[str, Value], *, drop: Sequence[str] = ()) -> List[str]:
    out = [verb]
    if values.get("do"):
        out.append(str(values["do"]))
    if values.get("subject"):
        out.append(str(values["subject"]))
    for key, v in values.items():
        if key in ("subject", "do") or key in drop or v in (None, "", False):
            continue
        if key.endswith("="):
            out.append(f"{key}{v}")
        elif v is True:
            out.append(key)
        else:
            out += [key, str(v)]
    return out


def _resolved(verb: str, values: Dict[str, Value], ctx: Context):
    """(action or None, why it cannot run, the confirmations it needs).

    Resolved without the override flags, so every warning is found and shown
    as its own row whether or not it has been answered yet."""
    from poros.cli.app import BUILT

    mod = importlib.import_module(BUILT[verb])
    s = parse(form_argv(verb, values, drop=OVERRIDES))
    try:
        draft = mod.resolve(s, ctx)
        if draft.action is None:
            return None, "", []
        action = mod.inspect(draft.action, ctx) if hasattr(mod, "inspect") else draft.action
        asks = [q for q in mod.confirmations(action, ctx) if q.confirm]
        return action, "", asks
    except Refusal as r:
        return None, "; ".join([r.reason, *r.context]), []


def _warnings(asks, values: Dict[str, Value]) -> List[Field]:
    out = []
    for q in asks:
        flag = q.choices[0].value
        out.append(
            Field(
                key=flag,
                label=WARN_LABEL.get(flag, flag.lstrip("-")),
                kind="toggle",
                warn=True,
                value=bool(values.get(flag)),
                shown="yes" if values.get(flag) else "",
                source=q.prompt,
            )
        )
    return out


def _ready(view: View) -> bool:
    return not view.error and all(
        (not f.required or f.value not in (None, "")) and (not f.warn or f.value)
        for f in view.fields
    )


def _int(label: str, low: int = 1) -> Callable[[str], Optional[str]]:
    def check(text: str) -> Optional[str]:
        try:
            n = int(text.replace(",", "").replace("_", ""))
        except ValueError:
            return f"{label} is a whole number"
        return None if n >= low else f"{label} is at least {low}"

    return check


def _words(values: Sequence[str], notes: Dict[str, str] = {}) -> Callable[[], List[Option]]:
    return lambda: [Option(v, v, notes.get(v, "")) for v in values]


def _short(model: Optional[str]) -> str:
    from poros.cli.words.models import by_id

    if not model:
        return ""
    e = by_id(model)
    return e.short if e else model


def _model_options(ctx: Context, values: Dict[str, Value], entries=None):
    def opts() -> List[Option]:
        plan = {}
        for key, name in (("--seq-len", "seq_len"), ("--rank", "rank")):
            try:
                plan[name] = int(str(values[key]))
            except (KeyError, ValueError):
                pass
        drop = values.get("--lora-dropout") or values.get("lora_dropout=")
        if drop not in (None, True):
            try:
                plan["dropout"] = float(str(drop))
            except ValueError:
                pass
        return [
            Option(ch.label, ch.label, ch.note)
            for ch in shared.model_choices(ctx, entries, **plan)
        ]

    return opts


def _eval_files(ctx: Context, values: Dict[str, Value], shape: Optional[str]) -> List[Option]:
    """Files here that hold rows of the training data's shape, the data itself left out."""
    from poros import data as pdata
    from poros.cli.verbs.train_data import local_files

    data = Path(str(values.get("--data") or "")).name
    out = []
    for f in local_files(ctx.cwd)[:20]:
        if f == data:
            continue
        try:
            said = pdata.detect(ctx.cwd / f)
        except (OSError, ValueError):
            continue
        if shape is None or said == shape:
            out.append(Option(f, f, f"a file here · {said}"))
    return out[:8]
