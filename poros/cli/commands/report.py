"""`poros report`, `poros leaderboard` and `poros bench run|matrix|compare|report`: benchmark
RunArtifact records rendered, ranked and produced."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from poros.cli import contract, root
from poros.cli.contract import refuse
from poros.cli.errors import _friendly_error
from poros.cli.root import (
    _JSON_OPTION,
    _load_yaml_config,
    _no_artifacts,
    _no_grants,
    _ui,
    app,
    bench_app,
    load_run_artifacts,
)


def _format_parity(a: "RunArtifact") -> str:  # noqa: F821 (forward ref, lazy import)
    """Human display for a cell's parity.

    ``parity_loss_max_diff is None`` means parity was NOT measured in this
    speed/VRAM cell (parity is measured by the dedicated GPU parity test, not
    by a throughput cell) -- so we show an em dash, never a fake ``0.00e+00``.
    """
    if a.parity_loss_max_diff is None:
        return "—"
    if a.bitwise_exact:
        return "0.00e+00"
    return f"{a.parity_loss_max_diff:.2e}"


def _format_s_per_step(a: "RunArtifact") -> str:  # noqa: F821
    """Human display for s/step: median plus +/- spread on multi-pass runs."""
    if a.s_per_step is None:
        return "—"
    if (a.n_passes or 1) > 1 and a.s_per_step_stddev is not None:
        return f"{a.s_per_step:.3f} ±{a.s_per_step_stddev:.3f}"
    return f"{a.s_per_step:.3f}"


def _render_reports(cmd: str, src: Path, html: bool, markdown: bool, output: Optional[str]) -> None:
    """Shared body of ``poros report`` and ``poros bench report``."""
    from poros.cli import views

    t, uc, w = _ui()
    artifacts = load_run_artifacts(src)
    if not artifacts:
        _no_artifacts(src)
    # Output goes into a run directory when one was passed; for a single file
    # it goes to the current directory, never back into the source tree.
    out_base = src if src.is_dir() else Path.cwd()
    t.print(views.report_start(cmd, str(src), len(artifacts), w))

    html_out = output
    md_out = output
    if output and html and markdown:
        # One --output cannot hold both formats; derive a path per format.
        html_out = str(Path(output).with_suffix(".html"))
        md_out = str(Path(output).with_suffix(".md"))
    jobs = []
    if html:
        jobs.append(("HTML report", html_out or str(out_base / "report.html"), "poros.report.html",
                     "generate_html_report"))
    if markdown:
        jobs.append(("Markdown report", md_out or str(out_base / "report.md"),
                     "poros.report.markdown", "generate_markdown_report"))
    written: list[str] = []
    skipped: list[tuple[str, str]] = []  # (code, why)
    for kind, out_path, mod, fn in jobs:
        try:
            import importlib

            result = getattr(importlib.import_module(mod), fn)(artifacts, out_path)
            t.print(views.report_written(kind, str(result), w))
            written.append(str(result))
        except NotImplementedError:
            t.print(views.report_skipped(kind, "not implemented yet"))
            skipped.append(("E_NO_RESULTS", f"{kind} is not implemented yet"))
        except (ImportError, ValueError) as exc:
            t.print(views.report_skipped(kind, str(exc)))
            skip_code = "E_DEPS" if isinstance(exc, ImportError) else "E_NO_RESULTS"
            skipped.append((skip_code, f"{kind}: {exc}"))

    if not (html or markdown):
        refuse("E_USAGE", "Nothing to write", "both formats are off",
               fix=f"poros {cmd.split(' ', 1)[1]} {src} --md")
    if not written:
        codes = [code for code, _ in skipped] or ["E_NO_RESULTS"]
        refuse("E_DEPS" if "E_DEPS" in codes else codes[0], "No report written",
               "; ".join(why for _, why in skipped) or "no format is implemented")
    if contract.json_mode():
        from poros.cli.schemas import ReportOutput

        contract.emit(ReportOutput(source=str(src), artifacts=len(artifacts), written=written))
    elif written[0].endswith(".html"):
        t.print([[]] + uc.continue_with(f"open {written[0]}", min(w, uc.CAP)))


@app.command()
def report(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to a run/benchmark directory, or a .json/.jsonl results file.",
        exists=True,
    ),
    html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML report."),
    markdown: bool = typer.Option(
        False, "--markdown", "--md", help="Also generate a Markdown report."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for the report."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Generate a report from benchmark results.

    Accepts a directory of RunArtifact JSON files, OR a single .json /
    .jsonl results file (e.g. poros_grid_pro.jsonl). Training-run
    events.jsonl is NOT report input (its rows are per-step events, not
    benchmark cells). Works without torch.
    """
    _render_reports("poros report", run_dir, html, markdown, output)


@app.command()
def leaderboard(
    results_dir: Path = typer.Argument(
        ...,
        help="Benchmark results: a directory, or a .json/.jsonl results file.",
        exists=True,
    ),
    top_n: int = typer.Option(10, "--top", "-t", help="Number of entries to show."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Display a leaderboard from benchmark results.

    Ranks methods by VRAM efficiency across the results' models. Accepts a
    directory or a single .json/.jsonl results file. Works without torch.
    """
    artifacts = load_run_artifacts(results_dir)

    if contract.json_mode():
        from poros.cli.schemas import LeaderboardOutput

        if not artifacts:
            _no_artifacts(results_dir)
        ranked = sorted(
            [a for a in artifacts if a.status == "ok"],
            key=lambda a: a.vram_peak_mb,
        )[:top_n]
        rows = [
            {
                "method": a.method,
                "model": a.display_name,
                "precision": a.precision,
                "vram_peak_mb": a.vram_peak_mb,
                "s_per_step": a.s_per_step,
                "parity": a.parity_loss_max_diff,
                "bitwise": a.bitwise_exact,
            }
            for a in ranked
        ]
        contract.emit(LeaderboardOutput(rows=rows))
        return

    from poros.cli import views

    t, uc, w = _ui()
    if not artifacts:
        _no_artifacts(results_dir)

    # Sort by VRAM efficiency (lowest peak VRAM first)
    ranked = sorted(
        [a for a in artifacts if a.status == "ok"],
        key=lambda a: a.vram_peak_mb,
    )[:top_n]
    t.print(views.leaderboard(ranked, _format_parity, _format_s_per_step, w))


@bench_app.command(name="run")
def bench_run(
    spec: Path = typer.Argument(
        ...,
        help="Path to a benchmark spec YAML.",
        exists=True,
        readable=True,
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Override output directory."
    ),
    passes: Optional[int] = typer.Option(
        None, "--passes", "-p",
        help="Repeat each cell N times; report median s/step + spread "
        "(default: the spec's n_passes, normally 3).",
    ),
) -> None:
    """Run a single benchmark spec.

    Loads a BenchmarkSpec from YAML, executes each (method, model)
    cell, and writes RunArtifact results. Requires torch.
    """
    root._require_torch("poros bench run")

    from poros.bench.specs import BenchmarkSpec

    raw = _load_yaml_config(spec)
    _no_grants(raw, spec)
    try:
        bench_spec = BenchmarkSpec(**raw)
    except Exception as exc:
        _friendly_error(exc)
        return

    if output_dir is not None:
        bench_spec.reporting.output_dir = output_dir
    if passes is not None:
        bench_spec.training.n_passes = max(1, passes)

    import time as _time

    from poros.cli import views

    t, uc, w = _ui()
    t.print(views.bench_start("poros bench run", bench_spec.spec_id, [
        ("models", ", ".join(m.display_name for m in bench_spec.models)),
        ("methods", ", ".join(bench_spec.methods)),
        ("precision", ", ".join(bench_spec.precisions)),
        ("passes", str(bench_spec.training.n_passes)),
        ("output", str(bench_spec.reporting.output_dir)),
    ], w))
    try:
        from poros.bench.runner import run

        t0 = _time.monotonic()
        with t.activity(f"Benchmarking {bench_spec.spec_id}") as act:
            artifacts = run(bench_spec)
            act.done([])
        elapsed = _time.monotonic() - t0
        t.print(views.bench_results(bench_spec.spec_id, artifacts, _format_parity, w, elapsed))
        report_cmd = f"poros report {bench_spec.reporting.output_dir}"
        t.print([[]] + uc.continue_with(report_cmd, min(w, uc.CAP)))
    except NotImplementedError:
        refuse("E_INTERNAL", "Not available", "the bench runner module is not fully implemented")
    except typer.Exit:
        raise
    except Exception as exc:
        _friendly_error(exc)


@bench_app.command(name="matrix")
def bench_matrix(
    spec: Path = typer.Argument(
        ...,
        help="Path to a benchmark matrix spec YAML.",
        exists=True,
        readable=True,
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Override output directory."
    ),
    passes: Optional[int] = typer.Option(
        None, "--passes", "-p",
        help="Repeat each cell N times; report median s/step + spread "
        "(default: the spec's n_passes, normally 3).",
    ),
) -> None:
    """Run a full benchmark matrix: every model in the spec with every method.

    Expands the spec into all cells and runs each sequentially.
    Requires torch.
    """
    root._require_torch("poros bench matrix")

    from poros.bench.specs import BenchmarkSpec

    raw = _load_yaml_config(spec)
    _no_grants(raw, spec)
    try:
        bench_spec = BenchmarkSpec(**raw)
    except Exception as exc:
        _friendly_error(exc)
        return

    if output_dir is not None:
        bench_spec.reporting.output_dir = output_dir
    if passes is not None:
        bench_spec.training.n_passes = max(1, passes)

    import time as _time

    from poros.cli import views

    t, uc, w = _ui()
    n_cells = len(bench_spec.models) * len(bench_spec.methods) * len(bench_spec.precisions)
    t.print(views.bench_start("poros bench matrix", bench_spec.spec_id, [
        ("cells", uc.num(n_cells)),
        ("passes", str(bench_spec.training.n_passes)),
        ("output", str(bench_spec.reporting.output_dir)),
    ], w))
    try:
        from poros.bench.runner import matrix

        t0 = _time.monotonic()
        with t.activity(f"Running {uc.num(n_cells)} cells") as act:
            artifacts = matrix(bench_spec)
            act.done([])
        elapsed = _time.monotonic() - t0
        t.print(views.bench_results(bench_spec.spec_id, artifacts, _format_parity, w, elapsed))
        report_cmd = f"poros report {bench_spec.reporting.output_dir}"
        t.print([[]] + uc.continue_with(report_cmd, min(w, uc.CAP)))
    except NotImplementedError:
        refuse("E_INTERNAL", "Not available", "the bench runner module is not fully implemented")
    except typer.Exit:
        raise
    except Exception as exc:
        _friendly_error(exc)


@bench_app.command(name="compare")
def bench_compare(
    dir1: Path = typer.Argument(
        ...,
        help="First benchmark output directory.",
        exists=True,
    ),
    dir2: Path = typer.Argument(
        ...,
        help="Second benchmark output directory.",
        exists=True,
    ),
) -> None:
    """Compare two benchmark runs side by side.

    Loads RunArtifact JSON files from both directories and produces
    a comparison table. Works without torch for pre-recorded results.
    """
    from poros.bench.artifacts import RunArtifact
    from poros.cli import views

    t, uc, w = _ui()
    arts1 = load_run_artifacts(dir1)
    arts2 = load_run_artifacts(dir2)
    for d, arts in ((dir1, arts1), (dir2, arts2)):
        if not arts:
            _no_artifacts(d)

    # Build comparison keyed by (method, model_id, precision)
    def _key(a: RunArtifact) -> str:
        return f"{a.method}|{a.model_id}|{a.precision}"

    map1 = {_key(a): a for a in arts1}
    map2 = {_key(a): a for a in arts2}
    rows, styles = [], []
    for k in sorted(set(map1) | set(map2)):
        a, b = map1.get(k), map2.get(k)
        one = a or b
        # the model too: two models' rows of one method and precision would read the same
        model = getattr(one, "display_name", None) or one.model_id.rsplit("/", 1)[-1]
        cell = f"{one.method} / {model} / {one.precision}"
        dv = ds = ""
        st_dv = st_ds = "dm"
        if a and b:
            d_v = (b.vram_peak_mb - a.vram_peak_mb) / 1024
            d_s = b.s_per_step - a.s_per_step
            dv, ds = f"{d_v:+.1f} GB", f"{d_s:+.2f} s"
            st_dv = "a" if d_v < 0 else "tx"
            st_ds = "a" if d_s < 0 else "tx"
        rows.append([cell,
                     f"{uc.ceil1(a.vram_peak_mb / 1024):.1f} GB" if a else "–",
                     f"{uc.ceil1(b.vram_peak_mb / 1024):.1f} GB" if b else "–", dv,
                     f"{a.s_per_step:.2f} s" if a else "–",
                     f"{b.s_per_step:.2f} s" if b else "–", ds])
        styles.append(["hi", "tx", "tx", st_dv, "tx", "tx", st_ds])
    t.print(views.bench_compare(str(dir1), str(dir2), len(arts1), len(arts2), rows, styles, w))


@bench_app.command(name="report")
def bench_report(
    output_dir: Path = typer.Argument(
        ...,
        help="Benchmark output directory containing RunArtifact JSON files.",
        exists=True,
    ),
    html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML report."),
    markdown: bool = typer.Option(
        False, "--markdown", "--md", help="Also generate a Markdown report."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Override output file path."
    ),
) -> None:
    """Generate an HTML/Markdown report from benchmark artifacts.

    Accepts a directory or a single .json/.jsonl results file. Works without
    torch installed -- reads pre-recorded RunArtifact JSON/JSONL.
    """
    _render_reports("poros bench report", output_dir, html, markdown, output)
