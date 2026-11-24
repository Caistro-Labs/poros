"""The screens of what Poros trains: models, recipes, a model's architecture, a data file, an
evaluation, a train card, the runs and a run's page, an export and a proof (pure functions
from data to ``Line`` lists)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from poros.cli.ui import components as c
from poros.cli.ui.text import Line, S, Seg, ln, width
from poros.cli.views.machine_screens import _clip_text, _cut
from poros.cli.views.parts import (
    _knob,
    _unified,
    _w,
    command_lines,
    rail_text,
    table,
)

# What a run card's "ran" line says in its own words first, so a key=value of it is not said again
_SAID = ("recipe", "loop", "max_seq_length", "lora_rank", "lora_alpha", "learning_rate",
         "gradient_accumulation_steps", "quantization")
_RAN = {"lora_dropout": "dropout", "per_device_batch_size": "batch", "num_epochs": "epochs"}


def models(r: Any, w: int) -> List[Line]:
    """``poros models``: the catalog, each model with the recipes certified for it on this
    machine; no memory or speed (what the release measured is a certified recipe's)."""
    ww = _w(w)
    gpu = f"{r.gpu.name} {r.gpu.total_gib:.1f} GiB" if r.gpu else "no GPU detected"
    out: List[Line] = []
    if r.downloaded:
        out.append(c.event("Downloaded", None, None))
        out += c.rail_path("to", r.downloaded, ww)
        out.append([])
    if r.gpu is not None:  # the rows the /models card draws
        from poros.cli import sheets
        from poros.cli.context import Context
        from poros.cli.ui import sheet as sh

        listing = sheets.models_listing(Context())
        n = len(listing.items)
        about = f"{n} in the catalog · {listing.aside}"
        about = about if len(about) + 10 <= ww else f"{n} in the catalog"
        if r.gpu.unified:
            about = f"{n} train here · {len(r.models) - n} more do not, in this release"
        out.append(c.event("Models", None, about))
        out.append(c.rail())
        out.append(c.rail(*sh.whole(listing.head, ww - len(c.RAIL))))
        out += [_cut(c.rail(*i.cells), ww) for i in listing.items]
        out.append([])
        out += c.continue_with("recipes", ww)
        return out
    out.append(c.event("Models", None, f"{len(r.models)} in the catalog · {gpu}"))
    evidence = {"validated": "validated", "measured": "measured", "same-shape": "same shapes"}
    body, styles = [], []
    for m in r.models:
        flags = " · ".join(
            f for f in ("downloaded" if m.cached else "", "gated" if m.gated else "") if f
        )
        body.append((m.id, m.size.upper(), m.entry, evidence.get(m.evidence, m.evidence), flags))
        styles.append(("hi", "tx", "dm", "dm", "dm"))
    out += table(("model", "size", "kind", "evidence", ""), body, right=(1,), styles=styles, w=w)
    out += rail_text("no GPU detected; recipes lists the machines each certified run used", w)
    out.append([])
    out += c.continue_with("recipes", ww)
    return out


def recipes(r: Any, w: int) -> List[Line]:
    """``poros recipes``: the loops /train runs; after a copy, the file and
    the command that trains with it."""
    ww = _w(w)
    out: List[Line] = []
    if r.file:
        from pathlib import Path

        out.append(c.event("Shared as a file" if r.do == "share" else "Saved as your recipe",
                           Path(r.file).name, None))
        out += c.rail_path("to", r.file, ww)
        out.append(c.rail())
        out += [c.rail(S(line, "tx")) for line in (r.content or "").splitlines()]
        out.append(c.rail())
        out.append(c.rail(S("send it to anyone: in ./recipes it shows in /recipes, and nothing "
                            "left this machine" if r.do == "share" else "it shows in /recipes",
                            "dm")))
        return out
    if r.synced is not None:
        from poros.recipes import hub

        said = r.synced
        n = said.get("count", 0)
        out.append(c.event("Synced", f"{n} shared recipe{'s' * (n != 1)}",
                           hub.shown(str(said.get("repo", ""))) +
                           (" · demo" if said.get("demo") else "")))
        for where, why in said.get("refused", [])[:8]:
            out.append(c.rail(S("refused ", "dm"), S(f"{where}: {why}", "tx")))
        out.append(c.rail(S("their numbers are what their authors reported, not checked by Poros",
                            "dm")))
        out.append([])
        out += c.continue_with("recipes", ww)
        return out
    if r.recipe:
        one = r.recipe
        out.append(c.event(one.name, None, f"by {one.by}"))
        out.append(c.rail(S(f"{one.model} · {one.loop.upper()} · {one.data}", "tx")))
        if one.says:  # a family Poros has not validated
            out.append(c.rail(S(one.says, "e")))
        out += [c.rail(S(f"{k} = {v}", "dm")) for k, v in one.settings.items()]
        if one.measured:
            out.append(c.rail(S("reported, not checked by Poros", "rl")))
            out += [c.rail(S(f"{k} = {v}", "dm")) for k, v in one.measured.items()]
        if one.certified:
            out.append(c.rail())
            out += _certified(one.certified)
        out.append([])
        out += c.continue_with(one.train.removeprefix("poros "), ww)
        if one.without_data:  # no file of your own yet: the whole loop on generated rows
            out.append(c.rail(S("No data of your own yet? This tries the whole loop on "
                                "generated rows:", "dm")))
            out.append(c.rail(S(f"  {one.without_data}", "tx")))
        return out
    if r.copied:
        from pathlib import Path

        name = Path(r.copied).name
        out.append(c.event("Forked", name, None))
        out += c.rail_path("to", r.copied, ww)
        out.append(c.rail(S("edit main(run, data, ...); every step is forward_backward, "
                            "then optim_step", "dm")))
        out.append([])
        out += c.continue_with(f"train <model> --data <file> --recipe {name}", ww)
        return out
    from poros.cli import sheets
    from poros.cli.context import Context
    from poros.cli.ui import sheet as sh

    listing = sheets.recipes_listing(Context())
    out.append(c.event("Recipes", None, listing.aside))
    out.append(c.rail())
    out.append(c.rail(*sh.whole(listing.head, ww - len(c.RAIL))))
    for item in listing.items:
        out.append(_cut(c.rail(*item.cells), ww))
        out += [_cut(c.rail(S("  " + text.lstrip(" ·"), "dm")), ww) for text, _ in item.detail]
    out.append(c.rail())
    out.append(c.rail(S("losses".ljust(8), "dm"), S(" · ".join(r.losses), "tx")))
    out.append([])
    out += [c.rail(S("refused ", "dm"), S(line, "tx")) for line in r.refused]
    out += c.continue_with("recipes copy sft", ww)
    return out


def _certified(cert: Any) -> List[Line]:
    """A built-in's certification on ``recipes show``: the machines it ran on with what each run
    measured, then this machine: certified on its class with those figures, or not."""
    if cert.status == "pending":
        return [c.rail(S("Certification pending: its runs at this release are not in yet", "e"))]
    out = [c.rail(S(f"Certified by {cert.by} at Poros {cert.release} on", "rl"))]
    for m in cert.machines:
        out.append(c.rail(S(m["machine"]["label"], "tx"),
                          S(f"  {m['machine']['os']}" if m["machine"].get("os") else "", "dm")))
        out.append(c.rail(S("  certified: " + m["figures"], "dm")))
        if m.get("torch_figures"):
            out.append(c.rail(S("  " + m["torch_figures"], "dm")))
    here = cert.here
    drive = [c.rail(S("  its time an update also depends on the drive the experts stream from",
                      "dm"))] if here.depends_on_drive else []
    free = [c.rail(S(f"  {here.free_now}", "e"))] if here.free_now else []
    if here.status == "certified":
        out.append(c.rail(S("Certified on this machine", "a"), S(f"  {here.machine}", "dm")))
        out += drive + free
    elif here.status == "same_memory_class":
        out.append(c.rail(S(f"Certified on {here.certified_on}.", "tx")))
        out.append(c.rail(S("  Not run on this machine: its memory and time per step here are "
                            "not given.", "dm")))
        out += drive + free
    elif here.status == "not_certified":
        out.append(c.rail(S("Not certified on this machine", "e"), S(f"  {here.machine}", "dm")))
        if here.reason:
            out.append(c.rail(S(f"  {here.reason}", "dm")))
    elif here.named_card:
        out.append(c.rail(S(f"Planned for the {here.named_card} you named", "tx")))
        out.append(c.rail(S("  certification is matched to the machine it runs on: its host RAM "
                            "is not known here", "dm")))
    else:
        out.append(c.rail(S("No GPU was found here to compare with those machines", "dm")))
    return out


def arch(r: Any, w: int) -> List[Line]:
    """``poros arch``: what it found, and what it would take. Sizes are arithmetic, not a run."""
    ww = _w(w)
    said = {"validated": "a family Poros has checked", "user_asserted": "runs on your word",
            "target": "planned, not yet", "gated": "gated"}.get(
                r.status, "a family Poros has not checked")
    out = [c.event(r.model, r.model_type or "", said)]
    if r.weights.format:
        out.append(c.rail(S("weights  ".ljust(10), "dm"), S(r.weights.format, "tx")))
    if r.weights.why:
        out.append(c.rail(S(r.weights.why, "dm")))
    if r.layers:
        out.append(c.rail(S("layers".ljust(10), "dm"),
                          S(f"{r.layers['count']} at {r.layers['path']}, "
                            f"up to {r.layers['linears']} linears each", "tx")))
    if r.experts:
        out.append(c.rail(S("experts".ljust(10), "dm"),
                          S(f"{r.experts['per_layer']} a layer, {r.experts['per_token']} a token",
                            "tx")))
    if r.memory:
        out.append(c.rail(S("memory".ljust(10), "dm"),
                          S(f"adapter {r.memory['adapter_at_rank_16_gib']:g} GiB at rank 16, "
                            f"one layer {r.memory['one_layer_gib']:g}, "
                            f"embed and head {r.memory['embed_head_gib']:g}", "tx")))
    if r.disk.get("pack_gib") is not None:
        from poros.preflight.disk import disk_text

        def size(gib: float) -> str:  # on disk: decimal GB, as every download is said
            return disk_text(gib * 2**30 / 1e9)

        got = r.disk.get("download_gib")
        out.append(c.rail(S("disk".ljust(10), "dm"),
                          S(f"pack {size(r.disk['pack_gib'])}"
                            + (f", download {size(got)}" if got else ""), "tx")))
    out += [c.rail(S("can't place ", "e"), S(line, "tx")) for line in r.cannot_place]
    if getattr(r, "refused", None):
        out.append(c.rail(S("✕ no flag trains it: ", "e"), S(r.refused["reason"], "tx")))
    if r.tried is not None:
        first = r.tried.get("first_difference")
        routes = r.tried.get("routes") or {}
        # what was checked, once: each route's loss and adapter gradients against the model's
        # own transformers forward, and how closely they agreed
        same = ("bit for bit" if routes and all(x.get("bitwise") for x in routes.values())
                else "within 1e-4")
        passed = ("the blockwise forward a run trains with matches the model's own transformers "
                  f"forward, {same}: the loss and every adapter gradient"
                  if r.tried.get("route") == "cuda" else "bit for bit against a dense reference")
        said = (passed if r.tried["ok"] else
                f"differs first at {first['module']}, {first['tensor']}" if first else
                r.tried.get("why") or "no adapter gradient moved")
        out.append(c.rail(S("tried".ljust(10), "dm"), S(said, "tx" if r.tried["ok"] else "e")))
        out.append(c.rail(S(r.tried["says"], "dm")))
    if r.next:
        out.append([])
        out += c.continue_with(r.next[0].removeprefix("poros "), ww)
    return out


_SHAPE = {"chats": "chats", "pairs": "preference pairs", "text": "plain text",
          "convertible": "question / answer rows", "unknown": "rows Poros can't read"}


_TRAINS = {"sft": "SFT on the answers", "dpo": "DPO", "ipo": "IPO", "kto": "KTO", "orpo": "ORPO",
           "simpo": "SimPO", "classic": "the classic loop, every token"}


def data(r: Any, w: int) -> List[Line]:
    """``poros data``: what a file holds, three of its rows, what it trains. With --fix, the
    clean copy: where, its rows, and each row made the file's shape or left out."""
    ww = _w(w)
    out: List[Line] = []
    written = r.converted  # --convert's file
    ok = r.shape in ("chats", "pairs", "text") or bool(r.trains) or written or r.wrote
    mark, style = ("●", "tx") if ok else ("○", "dm")
    count = f"{r.rows:,} rows"
    if r.lines is not None and r.lines != r.rows and not written:  # a blank line is no row
        count += f" · {r.lines:,} line{'' if r.lines == 1 else 's'}"
    title = "Fixed" if r.wrote else "Converted" if written else "Data"
    out.append(c.event(title, _short_path(r.path), count, mark=mark, mark_style=style))
    out.append(c.rail())
    # an empty file holds nothing: never "rows Poros can't read", of which it has none
    holds = "no rows: the file is empty" if not r.rows and not r.problems else _SHAPE[r.shape]
    out.append(c.rail(S("holds".ljust(10), "dm"), S(holds, "tx")))
    if written or r.wrote:
        out.append(c.rail(S("wrote".ljust(10), "dm"), S(_short_path(written or r.wrote), "hi")))
    if r.wrote:
        out.append(c.rail(S("kept".ljust(10), "dm"), S(f"{r.kept:,} rows", "tx")))
    if r.trains:
        out.append(c.rail(S("trains".ljust(10), "dm"), S(_TRAINS.get(r.trains, r.trains), "a")))
    for i, sample in enumerate(r.samples):
        out.append(_cut(c.rail(S(("rows" if i == 0 else "").ljust(10), "dm"), S(sample, "tx")), ww))
    for p in r.converted_rows or []:
        out.append(c.rail(S("converted".ljust(10), "a"), S(p, "tx")))
    for p in r.dropped if r.wrote else r.problems:
        out.append(c.rail(S(("dropped" if r.wrote else "✕ ").ljust(10), "e"), S(p, "tx")))
    if r.shape == "unknown":
        out.append(c.rail(S("", "dm"), S("chats are {\"messages\": [...]}; pairs are "
                                          "{\"prompt\", \"chosen\", \"rejected\"}", "dm")))
    out.append([])
    if r.command:
        out += c.continue_with(r.command.removeprefix("poros "), ww)
    return out


_SCORED = {"held_out": "held-out rows", "eval_file": "rows of its --eval file",
           "trained": "pairs it trained on, none held out", "file": "rows"}


def evaluated(r: Any, w: int, name: str = "", note: str = "", then: str = "") -> List[Line]:
    """``poros eval``: the base beside the run on the same rows, then the verdict, and a warning
    when its answers do not end. A config file's run has no id: ``name`` is its folder, ``note``
    what its rows are, ``then`` its next command."""
    from pathlib import Path

    ww, lw = _w(w), 10
    of =f"{r.rows} of {r.of}" if r.rows and r.of and r.of > r.rows else f"{r.rows or 0}"
    where = _SCORED.get(r.rows_from or "", "rows")
    if r.rows_from == "file":
        where = f"rows of {Path(r.dataset).name}"
    out = [c.event("Scored", name or f"run {r.run}",
                   "simulated" if r.simulated else f"{of} {where}" if r.rows else None)]
    out += [c.rail(S(note, "dm"))] if note else []
    out.append(c.rail())
    base, ours = r.base, r.trained
    if base is not None and ours is not None and ours.loss is not None:
        change = (f"{abs(r.change):.0%} {'lower' if r.change < 0 else 'higher'}"
                  if r.change is not None else "")
        out.append(c.rail(S("loss".ljust(lw), "dm"), S(f"{base.loss:.3f}", "tx"),
                          S(" base   ", "dm"), S(f"{ours.loss:.3f}", "a"), S(" trained   ", "dm"),
                          S(change, "tx")))
    if r.improved is not None and r.rows:
        noun = "pairs moved toward the chosen answer" if r.shape == "pairs" else "rows"
        out.append(c.rail(S("better".ljust(lw), "dm"), S(f"{r.improved} of {r.rows}", "tx"),
                          S(f" {noun}", "dm")))
    if base is not None and ours is not None and ours.prefers is not None:
        out.append(c.rail(S("prefers".ljust(lw), "dm"), S(f"{base.prefers} of {r.rows}", "tx"),
                          S(" base   ", "dm"), S(f"{ours.prefers} of {r.rows}", "a"),
                          S(" trained   the chosen answer", "dm")))
    if r.answer_rows and ours is not None and ours.ends is not None:
        n = r.answer_rows
        out.append(c.rail(S("ends".ljust(lw), "dm"), S(f"{base.ends} of {n}", "tx"),
                          S(" base   ", "dm"), S(f"{ours.ends} of {n}", "a"),
                          S(" trained   ", "dm"),
                          S(f"p {base.end_p:.2f} to {ours.end_p:.2f}", "dm")))
    out += _samples(r, ww, lw)
    if r.verdict:
        out += [c.rail(), c.rail(S(r.verdict, "hi"))]
    if r.warning:
        room = max(20, ww - len(c.RAIL) - 2)
        for i, line in enumerate(c.wrap(r.warning + ".", room)):
            out.append(c.rail(S("✕ " if i == 0 else "  ", "e"), S(line, "tx")))
    out.append([])
    return out + c.continue_with(then or f"export {r.run}", ww)


def _samples(r: Any, ww: int, lw: int) -> List[Line]:
    """eval --show: each row's question, the answer its file gives, then the base's and the run's,
    greedy. The words are a file's and a model's, made plain (evaluate_pass.plain) before here."""
    said = list(getattr(r, "samples", []) or [])
    if not said:
        return []
    ran = any(x.base is not None for x in said)
    where = {"held_out": "held-out", "eval_file": "held-out", "trained": "training"}.get(
        r.rows_from or "", "")
    head = f"{len(said)} {where + ' ' if where else ''}prompt{'s' * (len(said) != 1)}"
    out = [c.rail(), c.rail(S("answers".ljust(lw), "dm"), S(head, "tx"), S(
        "   greedy, the first 48 tokens" if ran else "   no model ran in a preview", "dm"))]
    room = max(20, ww - len(c.RAIL) - lw)
    for i, x in enumerate(said, 1):
        out.append(c.rail())
        rows = [(str(i), x.prompt or "(no question)", "hi"), ("expected", x.expected, "dm")]
        if ran:
            rows += [("base", x.base or "(nothing)", "tx"),
                     ("trained", x.trained or "(nothing)", "a")]
        for label, text, style in rows:
            for j, line in enumerate(c.wrap(text, room)[:3]):  # three lines of any one text
                line = line if len(line) <= room else line[: room - 1] + "…"  # a long word
                out.append(c.rail(S((label if j == 0 else "").ljust(lw), "dm"), S(line, style)))
    return out


def _source(src: Any) -> str:
    return str(src) if src is not None else ""


_TRAIN_ON = {"answers": "answers only", "last": "last answer", "all": "every token"}


def recipe_text(recipe: str, shape: Optional[str], config: Dict[str, Any],
                lr: Optional[float] = None) -> str:
    """"SFT · answers only · lr 2e-4", "DPO · beta 0.1 · lr 1e-5"."""
    if not recipe:
        return ""
    if recipe == "classic" or shape == "text":
        parts = [recipe, "every token"]
    elif shape == "pairs":
        parts = [recipe, f"beta {float(config.get('dpo_beta', 0.1)):g}"]
    elif shape == "chats":
        parts = [recipe, _TRAIN_ON[str(config.get("train_on", "answers"))]]
    else:
        parts = [recipe]
    if lr:
        parts.append("lr " + lr_text(lr))
    return " · ".join(parts)


def lr_text(lr: float) -> str:
    """2e-4, 1e-5, 3.5e-5: how a learning rate is written."""
    mant, exp = f"{lr:.2e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def train_ready(a: Any, w: int) -> List[Line]:
    """The resolved train action: every value, where it came from, the fit
    verdict and download, and the canonical command, before anything runs."""
    ww = _w(w)
    data = a.data if a.data == "synthetic" else _file_name(a.data)
    out = [ln(S("● ", "tx"), S("Ready to train", "hi"), S(f"  {a.model} on {data}", "dm"))]
    p = a.plan
    rows = [
        ("model", a.model, a.provenance.get("model")),
        ("data", _short_path(a.data) + (f"  {a.rows:,} {a.shape or 'rows'}" if a.rows else ""),
         a.provenance.get("data")),
        ("recipe", recipe_text(a.recipe, a.shape, a.config, a.lr), a.provenance.get("recipe")),
        ("steps", f"{a.steps:,}", a.provenance.get("steps")),
        ("length", f"{a.seq_len} tokens", a.provenance.get("seq_len")),
        ("rank", str(a.rank), a.provenance.get("rank")),
    ]
    if p is not None:  # the run's check, in words; figures only from a certified run here
        from poros.cli.sheets.fit import host_short
        from poros.cli.verbs.shared import path_why

        cert = getattr(a, "certified", None)
        if host_short(p):
            rows.append(("fit", "won't fit", host_short(p)))
        elif p.fits is False:
            rows.append(("fit", "won't fit", "refused before the download at these settings"))
        elif cert is not None:
            from poros.cli.sheets.fit import on_a_drive

            drive = on_a_drive(a)  # the SSD store: memory certified, time the drive's
            rows.append(("fit", "certified here", f"{cert[0]}'s run: "
                         + cert[1].figures(time=not drive)
                         + (" · its time an update was certified on another drive and is not "
                            "given" if drive else "")))
        else:
            rows.append(("fit", "checked at the start", "no certified run of these settings here"))
        if p.path is not None and not _unified(p):
            rows.append(("path", p.path, path_why(p)))
            if p.offload_reason:
                rows.append(("tables", "host RAM", "to leave GPU memory for training"))
        if p.download.value is not None:
            rows.append(("download", f"{p.download.value:.1f} GiB", "from the Hub"))
    for key, value in sorted(a.config.items()):  # key=value settings are shown too
        if key not in ("recipe", "loop"):  # the recipe row says them
            rows.append((key, str(value), "given"))
    rows.append(("output", a.output or "a new run folder in ~/.poros/runs", ""))
    lw = max(10, max(len(k) for k, _, _ in rows) + 2)
    vw = min(34, max(len(v) for _, v, _ in rows) + 2)
    for label, value, src in rows:
        tail = _source(src)
        if len(c.RAIL) + lw + vw + len(tail) <= ww:
            out.append(c.rail(S(label.ljust(lw), "dm"), S(value.ljust(vw), "tx"), S(tail, "dm")))
        else:
            out.append(c.rail(S(label.ljust(lw), "dm"), S(value, "tx")))
            for line in c.wrap(tail, max(20, ww - len(c.RAIL) - lw)):
                out.append(c.rail(S(" " * lw + line, "dm")))
    if a.kept is not None:
        out += rail_text(f"{a.kept:.0%} of rows are no longer than {a.seq_len} tokens; the "
                         "rest are cut to it", w)
    if p is not None and _unified(p):  # the Mac table's words; whose figures they are goes
        for note in p.warnings:
            if not note.startswith("memory and speed are"):
                out += rail_text(note, w)
    out.append([])
    out += command_lines(a.command(), w)
    out.append([])
    return out


def _short_path(path: str) -> str:
    """A path as people read it: relative to here, else ~/; the command keeps it whole."""
    return c.short_path(path)


def _file_name(path: str) -> str:
    from pathlib import Path

    return Path(path).name or path


_RUN_MARK = {"queued": ("○", "dm"), "running": ("●", "a"), "stopping": ("○", "dm"),
             "succeeded": ("✓", "tx"), "failed": ("✕", "e"), "cancelled": ("○", "dm")}


def _took(start: Any, end: Any) -> str:
    """How long a run ran: from start to finish, or to now while it runs."""
    from datetime import datetime, timezone

    if not start:
        return "—"
    try:
        t0 = datetime.fromisoformat(str(start))
        t1 = datetime.fromisoformat(str(end)) if end else datetime.now(timezone.utc)
        return c.duration(max(0.0, (t1 - t0).total_seconds()))
    except ValueError:
        return "—"


def _when(iso: Any) -> str:
    """A timestamp as local time, short (run ids use local time too)."""
    from datetime import datetime

    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(str(iso)).astimezone().strftime("%m-%d %H:%M")
    except ValueError:
        return str(iso)[:16]


def runs(r: Any, w: int) -> List[Line]:
    """``poros runs``: every run newest first, or one run's page."""
    ww = _w(w)
    if r.do == "list":
        if not r.runs:
            return [c.event("No runs yet", None, None, mark="○"),
                    c.rail(S("a run gets a folder here, its curve and its receipt", "dm")),
                    *_LISTED(ww), [], *c.continue_with("train", ww)]
        more = f" · {r.more} more not shown" if getattr(r, "more", 0) else ""
        out = [c.event("Runs", None, f"{len(r.runs)} · newest first{more}")]
        nw = min(36, max(len(x["name"]) for x in r.runs) + 2)
        out.append(_runs_header(ww, nw))
        for x in r.runs:
            out.append(_run_row(x, ww, nw))
        # a run whose process ended without finishing (killed, its machine stopped) says so in
        # words, and how it goes on, as its own page does: a ✕ alone reads as any failure
        cut = [x for x in r.runs if (x.get("error") or {}).get("code") == "E_INTERRUPTED"]
        for x in cut[:3]:
            out += [c.rail(), *rail_text(f"{x['name']} was interrupted: {x['error']['message']}",
                                         ww)]
        out += [c.rail(), *_LISTED(ww), []]
        return out + c.continue_with(f"runs continue {cut[0]['id']}" if cut else "runs <run>",
                                     ww)
    if r.do == "continue":  # the trainer drew the run and its summary; one next step is enough
        going = r.runs[0]["state"] in ("running", "queued")  # in the background: watch it there
        return c.continue_with(f"runs {r.runs[0]['id']}", ww) if going else []
    return run_page(r.runs[0], getattr(r, "detail", None) or {}, ww, r.do, r.report)


def _LISTED(ww: int) -> List[Line]:
    """Which runs the list holds, said under it: a run started another way is not lost, it is
    in the folder its user named."""
    said = ("lists runs started with poros train <model> --data, and jobs; a config file's run "
            "and a Python run keep their record in their own output_dir")
    return [c.rail(S(line, "dm")) for line in c.wrap(said, ww - len(c.RAIL))]


def _runs_header(ww: int, nw: int) -> Line:
    """The list's column names, over ``_run_row``'s columns, the last given up as its are."""
    tail = [S("   held-out", "dm"), S("  started    ", "dm"), S("  took", "dm")]
    while True:
        row = c.rail(S("  ", None), S("run".ljust(nw), "dm"), S(" " * 10, None),
                     S("steps".rjust(10), "dm"), S("loss".rjust(7), "dm"), *tail)
        if width(row) <= ww or not tail:
            return row if width(row) <= ww else _cut(row, ww)
        tail.pop()


def _run_row(x: dict, ww: int, nw: int) -> Line:
    """``✓ name  ▰▰▰▰▰▰▰▰  200/200  1.17  10-25 00:16  1m 26s``"""
    mark, style = _RUN_MARK.get(x["state"], ("○", "dm"))
    total, step = x.get("total"), x.get("step") or 0
    frac = step / total if total else None
    fill = {"running": "a", "failed": "e"}.get(x["state"], "tx" if frac == 1 else "dm")
    steps = f"{step:,}/{total:,}" if total else (f"{step:,}" if step else "—")
    loss = f"{x['loss']:.2f}" if x.get("loss") is not None else "—"
    end = _held_end(x.get("output_dir"))
    held = [S("  held ", "dm"), S(f"{end:.2f}", "hi")] if end is not None else [S(" " * 11, "dm")]
    tail = [held, [S(f"  {_when(x['created_at'])}", "dm")],
            [S(f"  {_took(x.get('started_at'), x.get('finished_at'))}", "dm")]]
    while True:  # narrow: the last facts go whole, never cut mid-word
        row = c.rail(S(mark + " ", style), S(_clip_text(x["name"], nw - 2).ljust(nw), "hi"),
                     *c.chips(frac, 10, fill), S(steps.rjust(10), "tx"), S(loss.rjust(7), "tx"),
                     *(cell for t in tail for cell in t))
        if width(row) <= ww or not tail:
            return row if width(row) <= ww else _cut(row, ww)
        tail.pop()


def _held_end(output_dir: Optional[str]) -> Optional[float]:
    """The run's last held-out loss, from its own events; none when it scored none."""
    if not output_dir:
        return None
    from poros.cli.verbs import runs_report

    try:
        end = runs_report.held_out(output_dir).get("eval_loss")
    except (OSError, ValueError):
        return None
    return end if isinstance(end, (int, float)) else None


_RUN_WORD = {"succeeded": "trained", "cancelled": "stopped"}


def run_page(x: dict, f: dict, ww: int, do: str = "show",
             report: Optional[str] = None) -> List[Line]:
    """One run in depth, from what it recorded: the loss curve, how far it
    got, memory against the card, speed, the settings it ran, its files,
    and the one thing to do next."""
    from poros.cli.words.models import by_id

    action, receipt = f.get("action") or {}, f.get("receipt") or {}
    losses = [v for _, v in f.get("losses") or []]
    mark, style = _RUN_MARK.get(x["state"], ("○", "dm"))
    word = {"stop": "Stop requested"}.get(do) or _RUN_WORD.get(x["state"], x["state"]).capitalize()
    when = _when(x.get("finished_at") or x.get("created_at"))
    # the mark heads it, as on every card; the state's own color carries a failure
    head = ln(S(c.MARK_INLINE + " ", "a"), S(word, "e" if style == "e" else "hi"),
              S(f"  {x['name']}", "tx"))
    gap = ww - width(head) - len(when)
    out = [head + ([S(" " * gap + when, "rl")] if gap > 2 else [])]
    e = by_id(x.get("model") or "")
    model = e.short if e else (x.get("model") or "")
    data = action.get("data") or receipt.get("config", {}).get("dataset") or ""
    gpu = (receipt.get("gpu") or {}).get("name", "")
    what = [S(model, "hi")] + ([S(" on ", "dm"), S(_file_name(str(data)), "tx")] if data else [])
    # the rest give way, last first, so the line never wraps
    tail = ([S(f" · {gpu}", "dm")] if gpu else []) + [S(f" · {x['id']}", "rl")]
    tail += [S(" · simulated", "rl")] if receipt.get("simulated") and "simulated" not in gpu else []
    while tail and width(what + tail) > ww - len(c.RAIL):
        tail.pop()
    out.append(_cut(c.rail(*what, *tail), ww))
    if x.get("support") == "user_asserted":  # --unvalidated: a family Poros has not checked
        from poros.cli.verbs.runs_report import UNVALIDATED

        out.append(_cut(c.rail(S(UNVALIDATED.format(x.get("model_type")), "dm")), ww))
    out.append(c.rail())
    lw = 8
    room = ww - len(c.RAIL) - lw

    def row(label: str, *parts: Seg) -> Line:
        return c.rail(S(label.ljust(lw), "dm"), *parts)

    # the curve: drawn where there's width, once the page's height is known;
    # narrower, three rows tall with the numbers that matter beside it
    curve_at = len(out) if len(losses) >= 2 and ww >= _RUN_CURVE_W else None
    if curve_at is not None:
        pass
    elif len(losses) >= 2:
        first, last = losses[0], losses[-1]
        side = [ln(S(f"{last:.4f}", "a")), ln(S(f"from {first:.2f}", "dm")),
                ln(S(f"{'↓' if last <= first else '↑'} {abs(last - first) / first * 100:.0f}%",
                     "tx")) if first else []]
        cw = max(16, min(room - 14, 64))
        chart = c.spark(losses, cw, 3)
        for i, line in enumerate(chart):
            out.append(row("loss" if i == 0 else "", *line, S("  ", None), *side[i]))
        start, end = f"step {f['losses'][0][0]:,}", f"{f['losses'][-1][0]:,}"
        out.append(row("", S(start + " " * max(1, cw - len(start) - len(end)) + end, "rl")))
        jumps = sorted(c.spikes(losses))
        if jumps:
            at = ", ".join(str(f["losses"][j][0]) for j in jumps[:3]) + (" …" if len(jumps) > 3
                                                                        else "")
            out.append(row("", S(f"{len(jumps)} spike{'s' if len(jumps) > 1 else ''}", "e"),
                           S(f" at step {at}", "dm")))
    elif x.get("loss") is not None:
        out.append(row("loss", S(f"{x['loss']:.4f}", "a")))
    else:
        out.append(row("loss", S("no steps recorded yet", "dm")))
    from poros.cli.ui.screens import metric_rows

    out += metric_rows(f.get("extra") or {}, lw, 16, f.get("acc_over") or 0)
    out.append(c.rail())

    # how far, how much, how fast
    total, step = x.get("total") or action.get("steps"), x.get("step") or 0
    # the training time, as the run's card said it; a run with no receipt: start to end
    took = (c.duration(receipt["wall_s"]) if receipt.get("wall_s")
            else _took(x.get("started_at"), x.get("finished_at")))
    parts = f.get("parts") or 1
    if parts > 1 and f.get("speeds"):
        # a run that stopped and went on: its receipt times the last stretch alone, so the
        # time said is every step's, over all of them
        took = f"{c.duration(sum(f['speeds']))} of steps, over {parts} stretches"
    fill = {"running": "a", "failed": "e"}.get(x["state"], "tx")
    if total:
        out.append(row("steps", *c.chips(step / total, 16, fill),
                       S(f"  {step:,} of {total:,}", "tx"), S(f"  in {took}", "dm")))
    card = (receipt.get("gpu") or {}).get("total_gb")
    peak = receipt.get("peak_vram_allocated_gb") or receipt.get("peak_vram_reserved_gb")
    peak = c.gib(peak) if peak is not None else None  # recorded in GB, like the card
    from poros.cli.ui.screens import MEMORY

    if peak is None and f.get("vram_gib"):
        peak = max(f["vram_gib"])
    if peak is not None:
        card = c.gib(card) if card else None  # recorded in GB; every screen says GiB
        meter = c.chips(peak / card, 16, "a") if card else []
        out.append(row(MEMORY, *meter, S(f"{'  ' if meter else ''}{c.ceil1(peak):.1f} GiB", "tx"),
                       S(f" peak of {card:.1f} GiB" if card else " peak", "dm")))
    sps = receipt.get("s_per_step") or (sum(f["speeds"]) / len(f["speeds"])
                                        if f.get("speeds") else None)
    if sps:
        tps = receipt.get("tokens_per_sec")
        out.append(row("speed", *([S(f"{tps:,.0f}", "tx"), S(" tok/s", "dm"), S(" · ", "dm")]
                                  if tps else []),
                       S(f"{sps:.2f}", "tx"), S(" s/step", "dm")))
    if receipt.get("peak_host_ram_gb") is not None:
        ram = c.ceil1(c.gib(receipt["peak_host_ram_gb"]))  # as the run's end card says it
        out.append(row("ram", S(f"{ram:.1f} GiB", "tx"), S(" peak", "dm")))
    out.append(c.rail())

    # what it ran with, then where it is
    cfg = receipt.get("config") or {}
    knobs = [
        str(receipt["recipe"]) if receipt.get("recipe") else "",
        f"length {action.get('seq_len') or cfg.get('max_seq_length')}"
        if action.get("seq_len") or cfg.get("max_seq_length") else "",
        f"rank {action.get('rank') or cfg.get('lora_rank')}"
        if action.get("rank") or cfg.get("lora_rank") else "",
        f"alpha {cfg['lora_alpha']}" if cfg.get("lora_alpha") else "",
        "lr " + lr_text(cfg["learning_rate"])
        if cfg.get("learning_rate") else "",
        f"accum {cfg['gradient_accumulation_steps']}"
        if cfg.get("gradient_accumulation_steps") else "",
        str(cfg["quantization"]) if cfg.get("quantization") else "",
    ] + [_knob(_RAN.get(k, k), v) for k, v in sorted((action.get("config") or {}).items())
         if k not in _SAID]  # the receipt names the recipe; the words above, each once
    knobs = [k for k in knobs if k]
    if knobs:
        for i, line in enumerate(c.wrap(" · ".join(knobs), room)):
            out.append(row("ran" if i == 0 else "", S(line, "tx")))
    out += c.rail_path("files", _short_path(x["output_dir"]), ww, lw)
    if f.get("saved"):  # what its checkpoints take on disk, the finished adapter with them
        from poros.cli.verbs.checkpoints import counted, size_text

        held = f["saved"]
        out.append(row("saved", S(size_text(held.size_bytes), "tx"),
                       S(f" in {counted(held.checkpoints, 'checkpoint')}", "dm")))
    if report:
        out += c.rail_path("report", _short_path(report), ww, lw)
    if x.get("error"):
        for i, line in enumerate(c.wrap(error_said(x["error"]), room)[:3]):
            out.append(row("error" if i == 0 else "", S(line, "e")))
    out.append([])
    nxt = {"running": f"runs stop {x['id']}", "stopping": f"runs {x['id']}",
           "queued": f"runs {x['id']}", "cancelled": f"runs continue {x['id']}",
           "failed": f"runs continue {x['id']}", "succeeded": f"export {x['id']}"}
    step = nxt.get(x["state"], "runs")
    if x["state"] in ("cancelled", "failed") and not _has_checkpoint(x.get("output_dir")):
        # nothing to continue from: the command that starts it again, or the list
        step = str(action.get("command") or "runs")
    out += c.continue_with(step, ww)
    if curve_at is not None:
        out[curve_at:curve_at] = _run_curve(x, f, ww, len(out))
    return out


_RUN_CURVE_W = 76  # run_card.MIN_W


def _run_curve(x: dict, f: dict, ww: int, around: int) -> List[Line]:
    """The curve the run drew, in the page's rail. The tallest that lets the page
    fit; in a tab, a page still too tall gets cells, which clip a row at a time."""
    from poros.cli.ui import curve as cv
    from poros.cli.ui import run_card as rc
    from poros.cli.ui import screens
    from poros.cli.verbs import runs_report

    by_step = dict(f["losses"])  # a run continued after a crash: its later word on a step
    steps = sorted(by_step)
    values, held = [], by_step[steps[0]]
    for step in range(steps[0], steps[-1] + 1):  # steps it didn't write hold the last value
        held = by_step.get(step, held)
        values.append(held)
    action = f.get("action") or {}
    total = max(x.get("total") or action.get("steps") or 0, steps[-1])
    room = screens._room()
    rows = rc.TALL if around + rc.TALL + 1 <= room else rc.SHORT
    fits = around + rows + 1 <= room or not c.in_session
    evals = screens._clamped(runs_report.evals(x["output_dir"]), total)
    k = rc.Card(series=cv.Series(total).extend(values), last=steps[-1] - 1, rows=rows,
                offset=steps[0] - 1, evals=evals, spikes=sorted(c.spikes(values)),
                first=values[0], final=values[-1], framed=False,
                painter=cv.Painter.maybe() if fits else None,
                scores=runs_report.scores(x["output_dir"]))
    going = x["state"] in ("running", "queued", "stopping")
    return rc.card(k, ww, None if going else rc.FINAL)


def _has_checkpoint(output_dir: Optional[str]) -> bool:
    """Whether a run kept a complete checkpoint to continue from."""
    from pathlib import Path

    from poros.core import checkpoints

    try:
        return bool(output_dir) and bool(checkpoints.list_checkpoints(Path(output_dir)))
    except OSError:
        return False


def error_said(err: Any) -> str:
    """A failed run's error as its page says it: the reason first, then its code and exit code;
    a record with no reason says only how it exited."""
    if not isinstance(err, dict):
        return str(err)
    how = [str(err["code"])] if err.get("code") and err.get("ended") != "unfinished" else []
    how += [f"exit code {err['exit_code']}"] if err.get("exit_code") is not None else []
    if err.get("message"):
        return str(err["message"]) + (f" ({', '.join(how)})" if how else "")
    return ", ".join(how) + ": no reason was recorded" if how else json_brief(err)


def json_brief(d: Any) -> str:
    if isinstance(d, dict):
        return " · ".join(f"{k} {v}" for k, v in d.items() if v not in (None, ""))
    return str(d)


def exported(r: Any, w: int) -> List[Line]:
    ww = _w(w)
    what = "a standalone model" if r.format == "merged" else "the adapter"
    out = [ln(S("● ", "tx"), S("Exported", "hi"), S(f"  {what}", "tx"),
              S(f"  run {r.run}" if r.run else "", "dm"))]
    out += c.rail_path("to", r.output, ww)
    out += rail_text(r.fidelity, w)
    # the base it trained on, by name: the line is one to run, with nothing left to fill in
    base = getattr(r, "base_model", None) or "<base model>"
    serve = (f"vllm serve {r.output}" if r.format == "merged"
             else f"vllm serve {base} --enable-lora --lora-modules run={r.output}")
    if getattr(r, "note", None):  # no serve line that would fail: what holds instead
        out += rail_text(r.note, w)
    else:
        out += c.rail_path("serve", serve, ww)
    out.append([])
    return out + c.continue_with(f"runs {r.run}" if r.run else "runs", ww)


def proved(r: Any, w: int) -> List[Line]:
    """``poros prove``: the two arms side by side, memory as a share of the
    larger, then the result: bit for bit, or what differs. Then the receipt."""
    from poros.cli.words.models import by_id

    ww = _w(w)
    gpu = (r.env.get("gpu") or "this machine").removesuffix(" (simulated)")
    where = f"on {gpu}" + (" · simulated" if r.simulated else "")
    if r.identical:
        said = ("  the block size changes nothing" if getattr(r, "one_block", None) is not None
                else "  streaming changes nothing")
        head = ln(S("● ", "tx"), S("Proved", "hi"), S(said, "tx"))
    else:
        head = c.error_event("Not identical", f"{len(r.differs)} item(s) differ", ww)[0]
    gap = ww - width(head) - len(where)
    out = [head + ([S(" " * gap + where, "rl")] if gap > 2 else [])]
    e = by_id(r.model)
    out.append(c.rail(S(e.short if e else r.model, "hi"),
                      S(f"  ·  {r.steps} steps · length {r.seq_len} · one seed, two runs", "dm")))
    out.append(c.rail())
    lw = 10
    # the reference by what it was: every layer held on the GPU, or one streamed block of them
    ref = r.resident if r.resident is not None else r.one_block
    arms = (("streamed", r.streamed, f"block {r.streamed.block_size}"),
            ("resident" if r.resident is not None else "one block", ref,
             f"all {ref.block_size}"))
    most = max([a.peak_vram_gib or 0 for _, a, _ in arms]) or None
    for label, a, blocks in arms:
        mem = a.peak_vram_gib
        fill = "a" if label == "streamed" else "tx"
        out.append(c.rail(S(label.ljust(lw), "dm"),
                          *c.chips(mem / most if mem is not None and most else None, 16, fill),
                          S(f"{c.ceil1(mem):.1f} GiB".rjust(10) if mem is not None
                            else "—".rjust(10),
                            "tx"),
                          S(f"  {blocks}".ljust(10), "dm"),
                          S(f"  {a.s_per_step:.2f} s/step" if a.s_per_step else "", "dm")))
    sm, rm = r.streamed.peak_vram_gib, ref.peak_vram_gib
    if sm and rm and rm > sm:
        slow = (f", {r.streamed.s_per_step / ref.s_per_step:.1f}× the time per step"
                if r.streamed.s_per_step and ref.s_per_step else "")
        out.append(c.rail(S(" " * lw), S(f"{(1 - sm / rm) * 100:.0f}% less memory", "tx"),
                          S(slow, "dm")))
    out.append(c.rail())
    if r.identical:
        out.append(c.rail(S("result".ljust(lw), "dm"), S("identical", "a"),
                          S("  " + " · ".join(x for x in r.compared if not x.startswith("0 "))
                            + ", bit for bit", "dm")))
    else:
        out.append(c.rail(S("result".ljust(lw), "dm"), S("differs", "e"),
                          S("  " + ", ".join(r.differs[:4]), "tx")))
    if r.against:
        same = "identical to the receipt it re-ran" if r.reproduced else "not the receipt it re-ran"
        out.append(c.rail(S("again".ljust(lw), "dm"), S(same, "tx" if r.reproduced else "e")))
        for k, (then, now) in r.pinned_differs.items():
            out += rail_text(f"{k}: {then} then, {now} here", w)
    if r.largest_certified:  # a certified recipe's own, never a plan's
        out.append(c.rail(S("certified".ljust(lw), "dm"), S(r.largest_certified, "hi"),
                          S("  the largest model certified on this card; poros recipes lists all",
                            "dm")))
    out += c.rail_path("receipt", _short_path(r.receipt), ww, lw=lw)
    out.append([])
    return out + c.continue_with("recipes" if r.identical else "machine", ww)
