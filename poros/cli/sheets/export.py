"""The export and prove sheets: which run and checkpoint, in what format, and the loss card
beside them."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from poros.cli.context import Context
from poros.cli.sheets.common import (
    LENGTHS,
    _eval_files,
    _int,
    _model_options,
    _ready,
    _resolved,
    _short,
    _warnings,
    _words,
    command_line,
    initial,
)
from poros.cli.sheets.fit import _gpu_name
from poros.cli.sheets.listings import _ago, card_loss, run_held
from poros.cli.ui.sheet import Field, Form, Option, Value, View
from poros.cli.ui.text import S
from poros.cli.words.grammar import Sentence

EXPORT_AS = {
    "merged": ("a standalone model", "ready to serve · bf16"),
    "adapter": ("the adapter alone", "loads on its base · PEFT, vLLM"),
}


def export_form(s: Sentence, ctx: Context) -> Form:
    """One card, ready as it opens: the latest finished run, merged (or the
    saved preference), into the run's own export folder. Enter exports."""
    from poros import runs
    from poros.cli import questions

    done = [r for r in runs.all_runs() if r.state == "succeeded"]
    values0 = initial(s)
    if not values0.get("subject") and done:
        values0["subject"] = "last"
    if not values0.get("--format"):
        values0["--format"] = questions.preference("export") or "merged"
        if ctx.unified and done:  # a Mac merges up to a size; the adapter is always there
            from poros.metal.export import merges

            run = next((r for r in done if r.id == values0.get("subject")), done[0])
            if not merges(run.model):
                values0["--format"] = "adapter"

    def build(values: Dict[str, Value]) -> View:
        a, error, asks = _resolved("export", values, ctx)
        run = next((r for r in done if a is not None and r.id == a.run), None)
        fmt = str(values.get("--format") or "merged")
        given = values.get("subject") not in (None, "last")
        target = str(values.get("--output") or "")
        if not target and run is not None:
            target = _home(str(Path(run.folder) / "export" / fmt))
        fields = [
            Field(
                "subject", "run", value=values.get("subject"), required=True,
                shown=run.name if run else str(values.get("subject") or ""), bold=True,
                fact=_progress_fact(run) if run else "",
                source="" if given or not run else "the latest",
                options=_export_runs(done), hint="a finished run",
            ),
            Field(
                "--format", "as", value=fmt, required=True, quiet=True,
                shown=EXPORT_AS.get(fmt, (fmt, ""))[0], source=EXPORT_AS.get(fmt, ("", ""))[1],
                options=lambda: [Option(k, v[0], v[1]) for k, v in EXPORT_AS.items()],
            ),
            Field(
                "--output", "to", kind="text", value=values.get("--output"),
                shown=target, hint="a folder",
            ),
        ] + _warnings(asks, values)
        view = View(
            title=[S("Export ", "hi"), S(run.name if run else "a run", "hi" if run else "dm")],
            fields=fields, error=error,
            go="Export", command=command_line("export", values),
        )
        view.ready = a is not None and _ready(view)
        return view

    return Form("export", build, values0)


SHOW = (("", "none", "numbers only"), ("3", "3 answers", "base, then trained"),
        ("5", "5 answers", "base, then trained"), ("10", "10 answers", "a token at a time: slower"))


def eval_form(s: Sentence, ctx: Context) -> Form:
    """A finished run beside its base, every choice a pick: which run, which of its checkpoints,
    which rows, and how many answers to read. Enter scores it."""
    from poros import runs
    from poros.cli.verbs import checkpoints

    done = [r for r in runs.all_runs() if r.state == "succeeded" or r.step]
    values0 = initial(s)
    word = str(values0.get("subject") or "")
    if "/weights/" in word:  # poros://<run>/weights/<step>: the run, and the checkpoint row
        run_word, step = checkpoints.parse(word)
        values0["subject"], values0["--checkpoint"] = run_word, str(step)
    if not values0.get("subject") and done:
        values0["subject"] = "last"
    values0.setdefault("--show", "3")

    def build(values: Dict[str, Value]) -> View:
        a, error, asks = _resolved("eval", values, ctx)
        word = str(values.get("subject") or "")
        run = next((r for r in done if r.id == (a.run if a is not None else word)
                    or (a is None and word in (r.name,))), None)
        if run is None and a is None and word == "last":
            run = next((r for r in done if r.state == "succeeded"), None)
        given = values.get("subject") not in (None, "last")
        shape = _run_shape(run)
        noun = {"chats": "chats", "pairs": "pairs"}.get(shape or "", "rows")
        own, rows = _held_of(run)
        data = str(values.get("--data") or "")
        picked = str(values.get("--checkpoint") or "")
        show = str(values.get("--show") or "")
        fields = [
            Field(
                "subject", "run", value=values.get("subject"), required=True, bold=True,
                shown=_middle(run.name, 24) if run else str(values.get("subject") or ""),
                fact=_progress_fact(run) if run else "",
                source="" if given or not run else "the latest",
                options=_eval_runs(done), hint="a run, finished or with a checkpoint",
            ),
            Field(
                "--checkpoint", "weights", value=values.get("--checkpoint"), free=True,
                quiet=not picked,
                shown=("final" if not picked or picked == "final" else
                       f"step {int(a.step):,}" if a is not None and a.step else picked),
                source=_weights_said(run, a.step if a is not None else None),
                options=_weights(run), check=None,
                hint="the finished adapter, the best by held-out loss, or a step",
            ),
            Field(
                "--data", "rows", value=values.get("--data"), free=True, files=True,
                quiet=not data,
                shown=Path(data).name if data else own or (f"{rows} held out" if rows else
                                                            "its own"),
                fact=f"{rows} {noun}" if own and rows else "" if data or not rows else noun,
                source="your file" if data else "its eval file" if own else (
                    "the ones it never trained on" if rows else
                    "it held none out: pick a file of its shape"),
                options=_eval_data(ctx, run, shape),
                hint="the rows it held out, or a file of its shape: drag it in",
            ),
            Field(
                "--show", "answers", value=values.get("--show"), quiet=False,
                shown=dict((k, v) for k, v, _ in SHOW).get(show, f"{show} answers"),
                source=("the model's words: base, then trained" if show not in ("", "0")
                        else "loss and counts only"),
                options=lambda: [Option(k, v, n) for k, v, n in SHOW
                                 if not (shape == "text" and k)],
                hint="read what the base and the run answer, on the same prompts",
            ),
        ] + _warnings(asks, values)
        if shape == "text":  # no question to answer: --show is for chats and pairs
            values.pop("--show", None)
        view = View(
            title=[S("Score ", "hi"), S(run.name if run else "a run", "hi" if run else "dm"),
                   S(" beside its base", "dm")],
            fields=fields, error=error, go="Score it",
            command=command_line("eval", values),
        )
        view.ready = a is not None and _ready(view)
        return view

    return Form("eval", build, values0)


def _middle(name: str, most: int) -> str:
    """A name cut in its middle: runs of one model and file differ at their ends."""
    if len(name) <= most:
        return name
    head = (most - 1) // 2
    return name[:head] + "…" + name[len(name) - (most - 1 - head):]


def _held_of(r) -> Tuple[str, Optional[int]]:
    """(its eval file's name, or "" for rows held out of its data; how many rows), from what the
    run recorded, else from its settings."""
    from types import SimpleNamespace

    from poros import data as poros_data
    from poros.cli.verbs import runs_report
    from poros.cli.verbs.train_data import scan

    if r is None:
        return "", None
    cfg = runs_report._read(Path(r.output_dir) / "config.resolved.json") or {}
    said = run_held(r).get("eval_rows")
    if cfg.get("eval_dataset"):
        path = Path(str(cfg["eval_dataset"]))
        return path.name, said or (scan(str(path), path.parent).rows if path.is_file() else None)
    if said:
        return "", said
    if cfg.get("trained_with") == "classic":  # the engine's own loop trained every row
        return "", None
    held = poros_data.held_rows(SimpleNamespace(dataset=cfg.get("dataset"), eval_dataset=None,
                                                eval_split=cfg.get("eval_split", 0.05),
                                                eval_rows=cfg.get("eval_rows", 0)))
    return "", held or None


def _run_shape(r) -> Optional[str]:
    from poros.cli.verbs import runs_report

    if r is None:
        return None
    return (runs_report._read(Path(r.output_dir) / "config.resolved.json") or {}).get("data_shape")


def _eval_runs(done) -> Callable[[], List[Option]]:
    """Runs, newest first, each with its held-out loss: the one to score is the one to find."""
    def opts() -> List[Option]:
        out = []
        for r in sorted(done, key=lambda r: r.created_at or "", reverse=True):
            end = run_held(r).get("eval_loss")
            out.append(Option(r.id, r.name, " · ".join(filter(None, (
                f"held out {end:.2f}" if end is not None else "", _short(r.model),
                _ago(r.created_at))))))
        return out

    return opts


def _weights(run) -> Callable[[], List[Option]]:
    """The finished adapter, the best checkpoint, then each step, newest first, with its score."""
    def opts() -> List[Option]:
        from poros.cli.verbs import checkpoints

        out = [Option("", "final", "the finished adapter")]
        if run is None:
            return out
        rows = [x for x in checkpoints.of_run(run) if x.type == "train"]
        best = next((x for x in rows if x.best), None)
        if best is not None:
            out.append(Option("best", "best",
                              f"step {best.step:,} · held out {best.eval_loss:.4f}"))
        for x in rows:
            score = f"held out {x.eval_loss:.4f}" if x.eval_loss is not None else "not scored"
            out.append(Option(str(x.step), f"step {x.step:,}", score))
        return out

    return opts


def _weights_said(run, step: Optional[str]) -> str:
    if run is None:
        return ""
    if not step:
        return "the finished adapter"
    from poros.cli.verbs import checkpoints

    x = next((x for x in checkpoints.of_run(run) if x.type == "train" and str(x.step) == str(step)),
             None)
    if x is None or x.eval_loss is None:
        return "a checkpoint"
    return f"held out {x.eval_loss:.4f}" + (" · the best" if x.best else "")


def _eval_data(ctx: Context, run, shape: Optional[str]) -> Callable[[], List[Option]]:
    def opts() -> List[Option]:
        out = [Option("", "the rows it held out", "never trained on")]
        trained = ""
        if run is not None:
            from poros.cli.verbs import runs_report

            cfg = runs_report._read(Path(run.output_dir) / "config.resolved.json") or {}
            trained = Path(str(cfg.get("dataset") or "")).name
        return out + [o for o in _eval_files(ctx, {"--data": trained}, shape)]

    return opts


def _progress_fact(r) -> str:
    """``5 steps · loss 1.18``: what a run did, from its own events."""
    loss = card_loss(r)
    bits = [f"{r.step:,} steps" if r.step else "", f"loss {loss:.2f}" if loss is not None else ""]
    return " · ".join(b for b in bits if b)


def _export_runs(done) -> Callable[[], List[Option]]:
    """Finished runs, newest first: the model and when tell same-named runs apart."""
    return lambda: [
        Option(r.id, r.name, " · ".join(filter(None, (_short(r.model), _ago(r.created_at), r.id))))
        for r in sorted(done, key=lambda r: r.created_at or "", reverse=True)
    ]


def _home(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path


def prove_form(s: Sentence, ctx: Context) -> Form:
    """Two runs from one seed, streamed and resident, compared bit for bit."""
    from poros.cli.words.models import catalog

    dense = [e for e in catalog() if e.evidence == "validated" and e.entry == "dense"]
    here = f"this {_gpu_name(ctx.gpu)}" if ctx.gpu else "this machine"

    def build(values: Dict[str, Value]) -> View:
        a, error, asks = _resolved("prove", values, ctx)
        model = _short(a.model) if a else ""
        e = next((d for d in dense if d.short == model), None)
        facts = " · ".join(x for x in ("validated" if e else "",
                                         "downloaded" if e and e.id in ctx.cached else "") if x)
        fields = [
            Field(
                "subject", "model", value=values.get("subject"), free=True,
                shown=model, fact=facts, bold=True,
                options=_model_options(ctx, values, dense),
                hint="a validated dense model",
            ),
            Field(
                "--steps", "steps", kind="text", value=values.get("--steps"),
                shown=str(a.steps) if a else "", fact="each run · two runs from one seed",
                check=_int("steps"), hint="steps per run",
            ),
        ] + [f for f in (
            Field(
                "--seq-len", "length", value=values.get("--seq-len"), free=True,
                shown=f"{a.seq_len:,} tokens" if a else "",
                options=_words(LENGTHS[:3]), check=_int("length"), hint="tokens per example",
            ),
        ) if f.key in values] + _warnings(asks, values)
        view = View(
            title=[S("Prove streaming changes nothing", "hi"), S(f" on {here}", "dm")],
            fields=fields,
            facts=[("expect", "the same losses and adapter, bit for bit", ""),
                   ("writes", "a receipt anyone can re-run", "")],
            error=error,
            go="Prove it",
            command=command_line("prove", dict(values, subject=values.get("subject") or model)),
        )
        view.ready = a is not None and _ready(view)
        return view

    return Form("prove", build, initial(s))
