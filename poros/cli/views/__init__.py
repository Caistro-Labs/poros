"""Every command's screen.

Pure functions from data to ``Line`` lists. The commands gather the data and
print these through ``poros.cli.ui.term``; nothing here touches torch or
the terminal. This module holds the command screens (eval, export, report, bench,
help, plan, fit, jobs); ``views.parts`` the pieces every screen is built from;
``views.machine_screens`` the machine's screens,
``views.run_screens`` those of what trains (models, recipes, arch, data, train cards, runs,
exports, proofs).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from poros.cli.ui import components as c
from poros.cli.ui.text import Line, S, ln, width
from poros.cli.views.machine_screens import (
    machine,
    machine_card,
    machine_drawn,
    machine_head,
    machine_show,
    machine_side,
    show_size,
)
from poros.cli.views.parts import (
    W,
    _w,
    command_lines,
    rail_text,
    table,
)
from poros.cli.views.run_screens import (
    arch,
    data,
    evaluated,
    exported,
    json_brief,
    lr_text,
    models,
    proved,
    recipe_text,
    recipes,
    run_page,
    runs,
    train_ready,
)

# The screens (this module's, and machine_screens' and run_screens'). Their parts are
# poros.cli.views.parts; a private name is imported from the module that defines it.
__all__ = [
    "arch", "bench_compare", "bench_results", "bench_start", "data", "eval_done",
    "eval_start", "evaluated", "export_done", "export_start", "exported", "header",
    "help_screen", "job", "job_submitted", "jobs", "json_brief", "leaderboard", "lr_text",
    "machine", "machine_card", "machine_drawn", "machine_head", "machine_show", "machine_side",
    "models", "needs_input", "note", "proved", "recipe_text", "recipes",
    "report_skipped", "report_start", "report_written", "run_page", "runs", "show_size",
    "train_ready", "verb_help",
]


def header(cmd: str, *items: str, w: int = W) -> List[Line]:
    line = c.welcome(cmd, *items, max_width=_w(w))
    return [line, []] if line else []


# --------------------------------------------------------------------------
# eval / export
# --------------------------------------------------------------------------


def eval_start(model: str, dataset: str, adapter: str, w: int) -> List[Line]:
    ww = _w(w)
    out = header("poros eval", model, w=w)
    out.append(c.event("Checked", "adapter"))
    out += c.rail_path("adapter", adapter, ww)
    out.append(c.kv("dataset", str(dataset), lw=10, vw=0))
    out.append([])
    return out


def eval_done(results: dict, took: float, w: int) -> List[Line]:
    out = [c.event("Evaluated", None, c.duration(took))]
    keys = list(results)
    lw = max([len(k) for k in keys] + [8]) + 2
    for k in keys:
        v = results[k]
        if isinstance(v, float):
            vs = f"{v:.4f}"
        elif isinstance(v, int):
            vs = c.num(v)
        else:
            vs = str(v)
        key_result = k in ("eval_loss", "perplexity", "eval_perplexity")
        out.append(c.rail(S(k.ljust(lw), "dm"), S(vs.rjust(10), "a" if key_result else "tx")))
    return out


def note(text: str, detail: Optional[str] = None) -> List[Line]:
    """A soft notice (``○``): not an error, worth reading."""
    out = [c.event(text, None, None, mark="○")]
    if detail:
        out.append(c.rail(S(detail, "dm")))
    return out


def export_start(adapter: str, base: str, fmt: str, out_path: str, w: int) -> List[Line]:
    ww = _w(w)
    out = header("poros export", base, w=w)
    out.append(c.event("Checked", "adapter", fmt))
    out += c.rail_path("adapter", adapter, ww)
    out += c.rail_path("output", out_path, ww)
    out.append([])
    return out


def export_done(out_path: str, took: float, w: int, tokenizer: bool = True) -> List[Line]:
    ww = _w(w)
    out = [
        c.event(
            "Exported", "merged weights" + (" + tokenizer" if tokenizer else ""), c.duration(took)
        )
    ]
    out += c.rail_path("written", out_path, ww)
    out.append(
        c.rail(
            S(
                "loads in transformers, vLLM and TGI · "
                "for GGUF run llama.cpp convert_hf_to_gguf.py",
                "dm",
            )
        )
    )
    out.append([])
    out += c.continue_with(f"vllm serve {out_path}", ww)
    return out


# --------------------------------------------------------------------------
# report / leaderboard / bench
# --------------------------------------------------------------------------


def report_start(cmd: str, source: str, n: int, w: int) -> List[Line]:
    ww = _w(w)
    out = header(cmd, w=w)
    out.append(c.event("Read", f"{c.num(n)} artifact" + ("s" if n != 1 else "")))
    out += c.rail_path("source", source, ww)
    return out


def report_written(kind: str, path: str, w: int) -> List[Line]:
    return [c.event("Wrote", kind)] + c.rail_path("path", path, _w(w))


def report_skipped(kind: str, why: str) -> List[Line]:
    return note(f"Skipped {kind}", why)


def leaderboard(rows: Sequence[Any], fmt_parity, fmt_sps, w: int) -> List[Line]:
    out = header("poros leaderboard", "ranked by peak VRAM", w=w)
    out.append(c.event("Ranked", f"{len(rows)} run" + ("s" if len(rows) != 1 else "")))
    body, styles = [], []
    for i, a in enumerate(rows, 1):
        parity = fmt_parity(a)
        body.append(
            [
                str(i),
                a.method,
                a.display_name,
                a.precision,
                f"{a.vram_peak_mb / 1024:,.1f} GB",
                fmt_sps(a),
                parity,
            ]
        )
        styles.append(
            [
                "dm",
                "hi",
                "tx",
                "dm",
                "a" if i == 1 else "tx",
                "tx",
                "tx" if a.bitwise_exact else "dm",
            ]
        )
    out += table(
        ["#", "method", "model", "prec", "peak vram", "s/step", "parity"],
        body,
        right={0, 4, 5},
        styles=styles,
        w=w,
    )
    return out


def bench_start(cmd: str, spec_id: str, rows: Sequence[Tuple[str, str]], w: int) -> List[Line]:
    ww = _w(w)
    out = header(cmd, spec_id, w=w)
    out.append(c.event("Planned", spec_id))
    for k, v in rows:
        if k == "output":
            out += c.rail_path("output", v, ww)
        else:
            out.append(c.kv(k, v, lw=10, vw=0))
    out.append(
        c.rail(
            S(
                "arms share configuration, not a data pipeline: "
                "s/step is indicative, VRAM comparable",
                "dm",
            )
        )
    )
    out.append([])
    return out


def bench_results(
    spec_id: str, artifacts: Sequence[Any], fmt_parity, w: int, took: float
) -> List[Line]:
    out = [
        c.event(
            "Benchmarked",
            f"{len(artifacts)} cell" + ("s" if len(artifacts) != 1 else ""),
            c.duration(took),
        )
    ]
    body, styles = [], []
    best = min((a.vram_peak_mb for a in artifacts if a.status == "ok"), default=None)
    for a in artifacts:
        body.append(
            [
                a.method,
                a.display_name,
                a.precision,
                f"{a.vram_peak_mb / 1024:,.1f} GB",
                f"{a.s_per_step:.2f} s" if a.s_per_step else "–",
                fmt_parity(a),
                a.status,
            ]
        )
        styles.append(
            [
                "hi",
                "tx",
                "dm",
                "a" if a.vram_peak_mb == best else "tx",
                "tx",
                "tx" if a.bitwise_exact else "dm",
                "tx" if a.status == "ok" else "e",
            ]
        )
    out += table(
        ["method", "model", "prec", "peak vram", "s/step", "parity", "status"],
        body,
        right={3, 4},
        styles=styles,
        w=w,
    )
    return out


def bench_compare(
    label_a: str,
    label_b: str,
    n_a: int,
    n_b: int,
    rows: Sequence[Sequence[str]],
    styles: Sequence[Sequence[Any]],
    w: int,
) -> List[Line]:
    ww = _w(w)
    out = header("poros bench compare", w=w)
    out.append(c.event("Compared", f"{c.num(n_a)} vs {c.num(n_b)} artifacts"))
    out += c.rail_path("A", label_a, ww, lw=4)
    out += c.rail_path("B", label_b, ww, lw=4)
    out.append(c.rail())
    out += table(
        ["cell", "vram A", "vram B", "Δ vram", "s/step A", "s/step B", "Δ s/step"],
        rows,
        right={1, 2, 3, 4, 5, 6},
        styles=styles,
        w=w,
    )
    return out


# --------------------------------------------------------------------------
# help / version
# --------------------------------------------------------------------------


def help_screen(commands: Sequence[Tuple[str, str, str, str]], w: int) -> List[Line]:
    """``/help``: commands grouped like the docs, name then what it does. No
    argument column: each command opens a sheet that asks for what it needs."""
    ww = _w(w)
    sections = {"verbs": "Commands", "advanced": "Advanced", "shell": "Session"}
    by: dict = {}
    for name, _args, desc, sec in commands:
        by.setdefault(sec, []).append((name, desc))
    nw = max(len(n) for n, _, _, _ in commands) + 4
    dw = ww - len(c.RAIL) - nw
    out: List[Line] = []
    for key, title in sections.items():
        rows = by.get(key)
        if not rows:
            continue
        out.append(c.titled(c.event(title)) if not out else c.event(title))
        out.append(c.rail())
        for name, desc in rows:
            d = desc if len(desc) <= dw else desc[: max(0, dw - 1)] + "…"
            out.append(c.rail(S(("/" + name).ljust(nw), "hi"), S(d, "dm")))
        out.append([])
    hint = ln(
        S("/train", "tx"),
        S(" alone opens its settings · ", "dm"),
        S("tab", "tx"),
        S(" completes · ", "dm"),
        S("/help <command>", "tx"),
        S(" for more", "dm"),
    )
    if width(hint) > ww:
        hint = ln(S("/train", "tx"), S(" opens its settings · ", "dm"), S("/help <command>", "tx"))
    out.append(hint)
    return out


_STATE_MARK = {
    "queued": ("○", "dm"),
    "running": ("●", "a"),
    "cancelling": ("○", "dm"),
    "succeeded": ("●", "tx"),
    "failed": ("✕", "e"),
    "cancelled": ("○", "dm"),
}


def job_submitted(job_id: str, state: str, output_dir: str, w: int) -> List[Line]:
    ww = _w(w)
    return [
        c.event("Queued", job_id, state),
        *c.rail_path("output", output_dir, ww),
        [],
        *c.continue_with(f"poros runs {job_id}", ww),
    ]


def job(j: Any, w: int) -> List[Line]:
    """One background job: state, model, times, where its files are."""
    ww = _w(w)
    mark, style = _STATE_MARK.get(j.state, ("○", "dm"))
    out = [
        ln(
            S(mark + " ", style),
            S(j.job_id, "hi"),
            S("  " + j.state, "tx"),
            S("  " + j.model, "dm"),
        )
    ]
    for label, value in (
        ("queued", j.queued_at),
        ("started", j.started_at),
        ("finished", j.finished_at),
    ):
        if value:
            out.append(c.kv(label, value, lw=10, vw=0))
    out += c.rail_path("output", j.output_dir, ww)
    out += c.rail_path("log", j.log, ww)
    if j.error is not None:
        out.append(c.rail(S(j.error.code + "  ", "e"), S(j.error.message, "tx")))
    return out


def jobs(rows: Sequence[Any], w: int) -> List[Line]:
    if not rows:
        return [
            c.event("No jobs", None, "nothing queued or run yet", mark="○"),
            [],
            *c.continue_with("poros train <config> --detach", _w(w)),
        ]
    body = [(j.job_id, j.state, j.queued_at.replace("T", " ")[:16], j.model) for j in rows]
    styles = [("hi", _STATE_MARK.get(j.state, ("", "dm"))[1], "dm", "tx") for j in rows]
    return table(("job", "state", "queued", "model"), body, styles=styles, w=w)


def needs_input(questions: Sequence[Any], w: int) -> List[Line]:
    """A Question without a terminal to ask it in: the choices and their commands."""
    ww = _w(w)
    out: List[Line] = []
    for q in questions:
        out.append(c.event("Needs an answer", None, q.prompt, mark="○"))
        for ch in q.choices:
            room = ww - len(c.RAIL)
            if ch.note and len(ch.label) + 2 + len(ch.note) > room:
                out.append(c.rail(S(ch.label, "hi")))
                out += rail_text(ch.note, w)
            else:
                out.append(c.rail(S(ch.label, "hi"), S("  " + ch.note, "dm") if ch.note else None))
            for line in command_lines(ch.command, room, "  ", "tx"):
                out.append(c.rail(*line))
        out.append([])
    out += c.continue_with("run one of the commands above", ww)
    return out


def _plain_help(text: str) -> str:
    """Help prose as a terminal reads it: no RST double backticks, and outside the session a
    command said the way it is typed there (``poros train``, never the session's ``/train``)."""
    import re

    text = text.replace("``", "")
    if c.in_session:
        return text
    from poros.cli import registry

    names = "|".join(sorted(registry.names(), key=len, reverse=True))
    return re.sub(rf"(?<![\w./~>-])/({names})\b", r"poros \1", text)


def verb_help(verb: str, doc: str, w: int) -> List[Line]:
    """A verb's docstring as its help: the summary, the example commands, the
    flags in a column (``Flags:``, and any other block headed by a line that ends
    with a colon, under its own title), then the prose."""
    first, _, rest = _plain_help(doc.strip()).partition("\n")
    out = [c.event(verb)] + rail_text(first.split(":", 1)[-1].strip(), w, "tx")
    examples, blocks, prose, part = [], [], [], "examples"
    for line in rest.splitlines():
        text = line.strip()
        if text.endswith(":") and not line.startswith(" ") and len(text.split()) <= 5:
            part = "flags"  # "Flags:", or a titled block ("With a config file:")
            blocks.append(("" if text == "Flags:" else text, []))
        elif text.startswith("poros ") and part == "examples":
            examples.append(text)
        elif part == "flags" and text:
            flags = blocks[-1][1]
            name, _, about = text.partition("  ")
            if about or not flags:
                flags.append((name.strip(), about.strip()))
            else:  # a description that ran onto the next line
                flags[-1] = (flags[-1][0], f"{flags[-1][1]} {text}")
        elif part == "flags":
            part = "prose"
        elif text and (part == "prose" or examples):  # prose, with or without a flags block
            part = "prose"
            prose.append(text)
    out += [c.rail(S(e, "hi")) for e in examples]
    for title, flags in blocks:
        if not flags:
            continue
        col = max(len(n) for n, _ in flags) + 2
        out.append(c.rail())
        if title:
            out.append(c.rail(S(title, "dm")))
        for name, about in flags:
            lines = c.wrap(about, max(20, _w(w) - len(c.RAIL) - col)) or [""]
            out.append(c.rail(S(name.ljust(col), "tx"), S(lines[0], "dm")))
            out += [c.rail(S(" " * col), S(more, "dm")) for more in lines[1:]]
    if prose:
        out.append(c.rail())
        out += rail_text(" ".join(prose), w)
    out.append([])
    return out
