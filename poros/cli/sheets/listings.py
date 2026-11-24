"""The lists a sheet browses: models (and the recipes certified for each on this machine), runs
and their checkpoints, recipes, data files."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Tuple

from poros.cli.context import Context
from poros.cli.sheets.common import _short
from poros.cli.sheets.fit import _billions, recommended
from poros.cli.ui import sheet as sh
from poros.cli.verbs import mac


def _cell(text: str, width: int, style) -> Tuple[str, object]:
    text = text if len(text) <= width - 2 else text[: width - 3] + "…"  # two spaces after
    return (text.ljust(width), style)


def certified_models(ctx: Context) -> dict:
    """{model id: the built-in recipes certified on this very machine for it}: the one thing the
    model list says of this machine beyond what trains here."""
    from poros.recipes import certified, presets

    this = certified.here(ctx)
    out: dict = {}
    if this is None:
        return out
    for r in presets.built_in():
        hit = certified.match(certified.entries(r.key), this)
        if hit is not None and hit[1] == certified.CERTIFIED:
            out.setdefault(r.model, []).append(r.key)
    return out


def largest_certified(ctx: Context) -> Optional[str]:
    """The built-in recipe certified on this very machine whose model is the largest (the first
    of them in the recipe list where two share a size), or None where none is: what prove says
    trains here, read from certified.json as poros recipes reads it."""
    from poros.cli.words.models import by_id
    from poros.recipes import presets

    certified_here = certified_models(ctx)
    keys = [r.key for r in presets.built_in()
            if r.key in certified_here.get(r.model, [])]
    if not keys:
        return None
    model = {r.key: r.model for r in presets.built_in()}

    def size(key: str) -> float:
        e = by_id(model[key])
        return _billions(e.size) if e else 0.0

    return max(keys, key=lambda k: (size(k), -keys.index(k)))


def models_listing(ctx: Context) -> sh.Listing:
    """Every catalog model this machine trains, smallest first, with the recipes certified for
    it on this machine: enter trains it. No model's memory or speed is shown: what was measured
    on the release is a certified recipe's (/recipes)."""
    import sys

    from poros.cli.ui import logos
    from poros.cli.words.models import by_id, catalog, on_mac

    entries = mac.trainable(ctx)  # a Mac lists what it trains
    image = logos.images()
    on_a_mac = on_mac()
    certified_here = certified_models(ctx)

    slots = {e.id: 2 * n for n, e in enumerate(catalog())}  # a placement per row and mark

    def paths(e) -> list:
        """Where it trains: NVIDIA's mark (Linux, CUDA: every model), then
        Apple's where the Mac path has rows for it. Without images, only
        the Mac's, as its own glyph: the column every row shares says CUDA."""
        apple = (logos.cells("apple", "tx", image=image, slot=slots[e.id] + 1,
                             apple_glyph=sys.platform == "darwin")
                 if e.id in on_a_mac else [("", None), ("  ", None)] if image else [("  ", None)])
        nvidia = logos.cells("nvidia", "tx", image=image, slot=slots[e.id]) if image else []
        return nvidia + [(" ", None)] * bool(nvidia) + apple + [("  ", None)]

    rec = (recommended(ctx) or (None, ""))[0]
    ordered = sorted(entries, key=lambda e: (e.id not in certified_here, _billions(e.size)))
    items = []
    for e in ordered:
        here = e.id in ctx.cached
        actions = [("enter", "train", ["train", e.short])]
        if not here:
            actions.append(("ctrl-g", "download", ["models", "get", e.short]))
        tag = ([("recommended", "a")] if e.short == rec else
               [("downloaded", "rl")] if here else
               [("gated", "rl")] if e.gated else [])
        via = by_id(e.via) if e.evidence == "same-shape" and e.via else None
        keys = certified_here.get(e.id, [])
        detail = [(e.id, "tx")] + [(" · " + x, "dm") for x in (
            "" if ctx.unified else "MoE" if e.entry == "moe" else "dense",
            "" if ctx.unified else f"same shapes as {via.short}" if via else e.evidence,
            f"certified here: {', '.join(keys)}" if keys else "",
            "downloaded" if here else "",
            "gated: accept its license on the Hub first" if e.gated and not here else "",
        ) if x]
        items.append(
            sh.Item(
                group=logos.lab(e.id)[0],
                cells=[
                    *paths(e),
                    _cell(e.short, 29, "hi"),
                    _cell(e.size.upper(), 6, "tx"),
                    _cell("moe" if e.entry == "moe" else "", 5, "dm"),
                    _cell(f"certified: {', '.join(keys)}" if keys else "", 30, "a"),
                ],
                tag=tag,
                text=" ".join([e.short, e.id, e.size, e.entry, e.family]),
                actions=actions,
                detail=detail,
            )
        )
    labs: dict = {}
    for e in ordered:
        name, mark = logos.lab(e.id)
        labs[name] = (mark, labs.get(name, (mark, 0))[1] + 1)
    groups = [(name, (logos.cells(mark, "tx", image=True, slot=1000 + n) + [(" ", None)]
                      if image and mark else []) + [(name, None)])
              for n, (name, (mark, _)) in enumerate(sorted(labs.items(), key=lambda kv: -kv[1][1]))]
    marks = 7 if image else 4
    head = [("".ljust(marks), "rl"), ("model".ljust(29), "rl"), ("size".ljust(11), "rl"),
            ("recipes certified on this machine", "rl")]
    return sh.Listing(
        tag="models",
        title="Models",
        aside="what this Mac trains" if ctx.unified else "certified here first",
        head=head,
        items=items,
        groups=groups,
        start=next((i for i, e in enumerate(ordered) if e.short == rec), 0),
    )


RUN_GLYPH = {"running": ("●", "running"), "queued": ("●", "queued"), "succeeded": ("✓", "done"),
             "failed": ("✕", "failed"), "cancelled": ("○", "stopped")}


RUN_MARK = {"running": "a", "queued": "a", "succeeded": "tx", "failed": "e", "cancelled": "dm"}


def _ago(stamp: Optional[str]) -> str:
    from datetime import datetime, timezone

    if not stamp:
        return ""
    try:
        t = datetime.fromisoformat(stamp)
    except ValueError:
        return ""
    s = (datetime.now(timezone.utc) - t).total_seconds()
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if s >= size:
            return f"{int(s // size)}{unit} ago"
    return "just now"


def _run_actions(r) -> List[Tuple[str, str, List[str]]]:
    """Enter opens the run, whatever its state; the keys do the rest, its checkpoints last."""
    page = ("enter", "open", ["runs", r.id])
    saved = ("ctrl-p", "checkpoints", ["runs", "checkpoints", r.id])  # S saves, on every card
    if r.state in ("running", "queued"):
        return [("enter", "watch", ["runs", r.id]), ("ctrl-x", "stop", ["runs", "stop", r.id]),
                saved]
    model = getattr(r, "model", None)
    again = ("ctrl-t", "train again", ["train", _short(model)] if model else ["train"])
    if r.state == "succeeded":
        return [page, ("ctrl-e", "export", ["export", r.id]),
                ("ctrl-r", "train more", ["runs", "continue", r.id, "--more"]), again, saved]
    if r.state in ("failed", "cancelled"):
        return [page, ("ctrl-r", "resume", ["runs", "continue", r.id]), again, saved]
    return [page, saved]


def _run_detail(r) -> list:
    """The chosen run in one line: model, its curve's ends, data, how long, its peak."""
    from poros.cli.ui import components as c
    from poros.cli.verbs import runs_report
    from poros.cli.views.run_screens import _file_name, _took, error_said

    f = runs_report.facts(r.folder, r.output_dir)
    a, rc, losses = f["action"], f["receipt"], f["losses"]
    peak = rc.get("peak_vram_allocated_gb") or rc.get("peak_vram_reserved_gb")  # the end card's
    held = run_held(r)
    start, end = held.get("eval_loss_start"), held.get("eval_loss")
    k, n = held.get("eval_ends"), held.get("eval_answer_rows")
    chose, rows = held.get("eval_prefers"), held.get("eval_rows")
    # what matters most first: a narrow line loses its end, never the loss
    parts = [
        _short(r.model) if r.model else "",  # its own name: an Instruct is not its base
        f"loss {losses[0][1]:.2f} → {losses[-1][1]:.2f}" if len(losses) >= 2 else "",
        (f"held out {start:.2f} → {end:.2f}" if start is not None and end is not None
         else f"held out {end:.2f}" if end is not None else ""),
        f"ends {k} of {n}" if k is not None and n else
        f"prefers {chose} of {rows}" if chose is not None and rows else "",
        _file_name(str(a["data"])) if a.get("data") else "",
        _took(r.started_at, r.finished_at) if r.started_at else "",
        f"{c.ceil1(c.gib(peak)):.1f} GiB peak" if peak else "",
    ]
    if r.state == "failed" and r.error:
        parts.append(error_said(r.error))
    parts = [p for p in parts if p]
    return [(parts[0], "tx")] + [(" · " + p, "dm") for p in parts[1:]] if parts else []


def runs_listing(ctx: Context) -> sh.Listing:
    """Every run, newest first: enter opens it, in depth."""
    from poros import runs
    from poros.cli.ui import components as c

    items = []
    for r in runs.all_runs():
        frac = (r.step or 0) / r.total if r.total else None
        fill = {"running": "a", "queued": "a", "failed": "e"}.get(
            r.state, "tx" if frac == 1 else "dm")
        steps = f"{r.step or 0:,}/{r.total:,}" if r.total else f"{r.step or 0:,}"
        loss = "" if (shown := card_loss(r)) is None else f"{shown:.2f}"
        end = run_held(r).get("eval_loss")
        held = "" if end is None else f"{end:.2f}"
        glyph, word = RUN_GLYPH.get(r.state, ("○", r.state))
        items.append(
            sh.Item(
                cells=[
                    (glyph + " ", RUN_MARK.get(r.state, "dm")),
                    _cell(r.name, 40, "hi"),
                    *c.chips(frac, 10, fill),
                    (steps.rjust(10), "tx"),
                    (loss.rjust(7), "tx"),
                    (held.rjust(10), "hi" if held else "tx"),
                    ("    " + _ago(r.created_at), "rl"),
                ],
                text=" ".join([r.name, r.id, r.state, r.model or ""]),
                actions=_run_actions(r),
                detail=_run_detail(r),
            )
        )
    head = [(" " * 2, None), ("run".ljust(40), "rl"), ("progress".ljust(20), "rl"),
            ("loss".rjust(7), "rl"), ("held out".rjust(10), "rl"), ("    when", "rl")]
    return sh.Listing(
        tag="runs",
        title="Runs",
        aside=f"{len(items)} · newest first",
        head=head if items else [],
        items=items,
        empty="no runs yet: /train starts one",
    )


def _curve_at(curve: list, step: int) -> Optional[float]:
    """The run card's curve at a step: the point whose rows hold it (one row a step reads by row
    means), else that step's own loss, so a checkpoint reads as the card did there."""
    return next((v for end, v in curve if end >= step), None)


def checkpoints_listing(run_id: str) -> sh.Listing:
    """One run's checkpoints, as `poros agents checkpoint list` has them: the finished adapter,
    then each step, newest first. Enter copies one here. The keys are the run card's where they
    can be: export the finished adapter, continue from the newest step (a run continues from
    there only), and delete one of a run that is not going. A run that scored held-out rows at
    its checkpoints shows each score beside the loss, marks the lowest best, and says how many
    rows end their answers (chats) or prefer the chosen answer (pairs)."""
    from poros import runs
    from poros.cli.verbs import checkpoints, runs_report

    r = next(x for x in runs.all_runs() if x.id == run_id)
    rows = checkpoints.of_run(r)
    newest = next((x.path for x in rows if x.type == "train"), None)
    curve = runs_report.facts(r.folder, r.output_dir)["losses"]
    held = any(x.eval_loss is not None for x in rows)
    counts = {"ends": [x.eval_ends for x in rows], "chosen": [x.eval_prefers for x in rows]}
    counted = next((k for k, v in counts.items() if any(n is not None for n in v)), "")
    items = []
    for x in rows:
        loss = _curve_at(curve, x.step) if x.step and curve else x.loss
        step = x.path.rsplit("/", 1)[1]
        keys = [("enter", "copy here", ["runs", "checkpoints", r.id, "--copy", step])]
        if x.type == "adapter" and r.state == "succeeded":
            keys.append(("ctrl-e", "export", ["export", r.id]))
        if x.path == newest:
            keys += [k for k in _run_actions(r) if k[0] == "ctrl-r"]
        if r.state not in checkpoints.LIVE:
            keys.append(("ctrl-x", "delete", ["runs", "checkpoints", r.id, "--delete", step]))
        final = x.type == "adapter"
        said = ("the finished adapter" + ("" if x.step is None else f", after step {x.step:,}")
                if final else "the newest: a run goes on from here" if x.path == newest else "")
        of = {"ends": (x.eval_ends, x.eval_answer_rows), "chosen": (x.eval_prefers, x.eval_rows)}
        k, n = of.get(counted, (None, None))
        score = [(("" if x.eval_loss is None else f"{x.eval_loss:.4f}").rjust(10), "tx"),
                 ("  best" if x.best else "      ", "a"),
                 *([(("" if k is None or not n else f"{k} of {n}").rjust(11), "tx")]
                   if counted else [])] if held else []
        items.append(sh.Item(
            cells=[_cell("final" if final else f"step {x.step:,}", 14, "hi"),
                   (("" if loss is None else f"{loss:.4f}").rjust(7), "tx"), *score,
                   (checkpoints.size_text(x.size_bytes).rjust(11), "tx"),
                   ("    " + _ago(x.created), "rl")],
            text=" ".join([step, "final" if final else f"step {step}", x.path,
                           "best" if x.best else ""]).rstrip(),
            actions=keys,
            detail=[(said, "tx"), (" · " + x.path, "dm")] if said else [(x.path, "dm")],
        ))
    head = [("checkpoint".ljust(14), "rl"), ("loss".rjust(7), "rl"),
            *([("held out".rjust(10), "rl"), ("      ", "rl"),
               *([(counted.rjust(11), "rl")] if counted else [])] if held else []),
            ("size".rjust(11), "rl"), ("    saved", "rl")]
    total = sum(x.size_bytes for x in rows)
    return sh.Listing(tag="checkpoints", title="Checkpoints",
                      aside=f"{r.name} · {checkpoints.size_text(total)} · newest first",
                      head=head if items else [], items=items,
                      empty=f"no checkpoints: {r.name} saved none")


def recipes_listing(ctx: Context) -> sh.Listing:
    """Whole training setups, yours first: see ``poros.cli.recipes_card``."""
    from poros.cli import recipes_card

    return recipes_card.listing(ctx)


def data_listing(ctx: Context) -> sh.Listing:
    """The data files here: enter shows what one holds, ctrl/alt+t trains on it."""
    from poros.cli.ui.components import key_name
    from poros.cli.verbs.data import about
    from poros.cli.verbs.train_data import local_files

    items = []
    for f in local_files(ctx.cwd)[:40]:
        holds, trains = about(ctx.cwd / f)
        items.append(sh.Item(
            cells=[_cell(f, 30, "hi"), _cell(holds or "not a shape Poros reads", 30,
                                              "tx" if holds else "e"), (trains, "dm")],
            text=f,
            actions=[("enter", "see what it holds", ["data", f]),
                     ("ctrl-t", "train on it", ["train", "--data", f])],
            detail=[(f"trains {trains} · {key_name('ctrl-t')} trains on it" if trains
                     else "Poros reads chats, Q&A rows, pairs and text", "dm")],
        ))
    head = [("file".ljust(30), "rl"), ("holds".ljust(30), "rl"), ("trains", "rl")]
    return sh.Listing(tag="data", title="Data", aside="files here, newest first", head=head,
                      items=items, empty="no csv, json or jsonl in this folder",
                      window=max(3, min(len(items), 10)))


def card_loss(r) -> Optional[float]:
    """The loss a finished run's card and its /runs line end on (one row a step reads by row
    means); a run still going, its newest step's."""
    if r.state != "succeeded" or r.loss is None:
        return r.loss
    try:
        at = os.stat(os.path.join(r.output_dir, "events.jsonl"))
    except OSError:
        return r.loss
    end = _curve_end(r.folder, r.output_dir, (at.st_mtime_ns, at.st_size))
    return r.loss if end is None else end


@lru_cache(maxsize=512)
def _curve_end(folder: str, output_dir: str, stamp: tuple) -> Optional[float]:
    from poros.cli.verbs import runs_report

    losses = runs_report.facts(folder, output_dir)["losses"]
    return losses[-1][1] if losses else None


@lru_cache(maxsize=512)
def _held_at(output_dir: str, stamp: tuple) -> dict:
    from poros.cli.verbs import runs_report

    return runs_report.held_out(output_dir)


def run_held(r) -> dict:
    """A run's held-out scores (runs_report.held_out), read again only when its events change."""
    try:
        at = os.stat(os.path.join(r.output_dir, "events.jsonl"))
    except OSError:
        return {}
    return _held_at(r.output_dir, (at.st_mtime_ns, at.st_size))
