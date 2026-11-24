"""Markdown report generator for Poros benchmarks.

Renders a results JSONL (or list of RunArtifact objects) into a clean
Markdown report with the following sections:

    Summary -> Hardware -> Methods -> VRAM -> Speed -> Parity table
    (0.00e+00) -> OOM -> Config

Honest by design: the parity column shows the real max|delta| values.
Rows whose measured parity is not bitwise are flagged NOT BITWISE.

This module must work WITHOUT torch installed.  It depends only on
pydantic (for RunArtifact) and the Python standard library.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# JSONL record -> RunArtifact bridge
# ---------------------------------------------------------------------------

def _load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file, warning loudly about unparseable lines.

    A dropped line is a missing benchmark cell; a report that renders
    without it reads as complete.
    """
    records: List[Dict[str, Any]] = []
    n_skipped = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                n_skipped += 1
    if n_skipped:
        import warnings

        warnings.warn(
            f"{n_skipped} unparseable line(s) in {path} were skipped; "
            "the report may be missing cells.",
            stacklevel=2,
        )
    return records


def _quant_display(quant: str) -> str:
    """Map internal quantisation tags to user-facing labels."""
    _MAP = {
        "qlora4": "NF4 (QLoRA)",
        "nf4": "NF4 (QLoRA)",
        "int8": "INT8",
        "lora16": "BF16 (LoRA)",
        "bf16": "BF16 (LoRA)",
        "fp16": "FP16 (LoRA)",
    }
    if not isinstance(quant, str):
        return "?"
    return _MAP.get(quant, quant.upper())


def _parity_note(record: Dict[str, Any]) -> str:
    """Return a footnote marker when the record's measured parity is not bitwise."""
    bitwise = record.get("bitwise_exact", False)
    if (
        not bitwise
        and record.get("parity_loss_max_diff") is None
        and record.get("parity_weight_max_diff") is None
    ):
        # Parity was not measured in this cell -- no claim either way.
        return ""
    if not bitwise:
        return " [NOT BITWISE]"
    return ""


def _fmt_mb(val: Optional[float]) -> str:
    """Format MB value with comma separator, or dash if missing/zero."""
    if val is None or val == 0:
        return "--"
    return f"{val:,.0f}"


def _fmt_pct(val: Optional[float]) -> str:
    """Format a percentage value, or dash."""
    if val is None:
        return "--"
    return f"{val:+.1f}%" if val < 0 else f"{val:.1f}%"


def _fmt_slowdown(val: Optional[float]) -> str:
    """Format a slowdown multiplier, or dash."""
    if val is None:
        return "--"
    return f"{val:.2f}x"


def _fmt_sstep(record: Dict[str, Any]) -> str:
    """Format median s/step with the multi-pass stddev when available."""
    s = record.get("s_per_step")
    if not s:
        # 0.0 is a schema default from legacy loaders, never a measurement
        return "--"
    stddev = record.get("s_per_step_stddev")
    if (record.get("n_passes") or 1) > 1 and stddev is not None:
        return f"{s:.3f} ± {stddev:.3f}"
    return f"{s:.3f}"


def _is_ok(record: Dict[str, Any]) -> bool:
    return record.get("status", "ok") == "ok"


def _is_blocked(record: Dict[str, Any]) -> bool:
    return record.get("status", "") in ("disk_blocked", "oom", "error", "skipped")


# ---------------------------------------------------------------------------
# Normalise heterogeneous input
# ---------------------------------------------------------------------------

def _fill(r: Dict[str, Any], key: str, *fallbacks: Any) -> None:
    """Set ``r[key]`` from the first non-None fallback when the key is
    missing OR present-but-None. RunArtifact.model_dump() always emits every
    key, so plain setdefault silently never fires on schema input -- the
    schema's documented primary field (e.g. vram_peak_mb) was not what the
    report read."""
    if r.get(key) is None:
        for fb in fallbacks:
            v = r.get(fb) if isinstance(fb, str) and fb in r else (
                None if isinstance(fb, str) else fb
            )
            if v is not None:
                r[key] = v
                return


def _normalise_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a raw JSONL dict into a stable internal representation.

    The raw JSONL from poros_grid_pro.jsonl uses field names like
    ``resident_vram_mb``, ``poros_vram_mb``, ``quant``, ``tag``, etc.
    RunArtifact uses ``vram_peak_mb``, ``precision``, ``display_name``.
    This helper unifies both into a common dict shape.
    """
    r: Dict[str, Any] = dict(raw)  # shallow copy

    # Identity fields
    _fill(r, "display_name", "tag", "model_id", "unknown")
    _fill(r, "precision", "quant", "nf4")
    _fill(r, "arch", "unknown")

    # VRAM
    _fill(r, "resident_vram_mb", "vram_resident_mb")
    _fill(r, "poros_vram_mb", "vram_poros_mb", "vram_peak_mb")
    _fill(r, "unsloth_vram_mb", "vram_unsloth_mb")

    # Comparisons
    _fill(r, "vram_save_vs_resident_pct", "vram_saved_vs_resident_pct")
    _fill(r, "vram_save_vs_unsloth_pct", "vram_saved_vs_unsloth_pct")

    # Parity
    r.setdefault("parity_loss_max_diff", None)  # None = not measured in this cell
    r.setdefault("parity_weight_max_diff", None)
    r.setdefault("bitwise_exact", False)

    # Status
    r.setdefault("status", "ok")

    return r


def _records_from_artifacts(
    artifacts: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Convert a list of RunArtifact (pydantic) objects to dicts."""
    out: List[Dict[str, Any]] = []
    for a in artifacts:
        if hasattr(a, "model_dump"):
            out.append(a.model_dump())
        elif isinstance(a, dict):
            out.append(a)
        else:
            out.append(dict(a))
    return out


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _poros_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Completed POROS cells only, for the cross-method tables.

    The runner copies the resident/unsloth cross-references onto EVERY
    method's artifact, so iterating all ok records rendered each model once
    per method -- with the baseline row's own vs-columns, producing tables
    that claimed Poros both saved 54.9% and used 50.3% more than Unsloth on
    the same model. Legacy artifacts without a method field are poros runs.
    """
    return [
        r for r in records
        if _is_ok(r) and r.get("method", "poros") == "poros"
    ]


def _section_summary(records: List[Dict[str, Any]], title: str) -> str:
    """Top-level summary paragraph."""
    ok = [r for r in records if _is_ok(r)]
    blocked = [r for r in records if _is_blocked(r)]
    # Speed/VRAM cells record no parity by design; only parity-measured runs
    # belong in the bitwise / not-bitwise buckets.
    measured = [
        r
        for r in ok
        if r.get("bitwise_exact")
        or r.get("parity_loss_max_diff") is not None
        or r.get("parity_weight_max_diff") is not None
    ]
    bitwise = [r for r in measured if r.get("bitwise_exact")]
    not_bitwise = [r for r in measured if not r.get("bitwise_exact")]

    archs = sorted({r.get("arch", "?") for r in ok}, key=str)
    quants = sorted({r.get("precision", r.get("quant", "?")) for r in ok}, key=str)

    lines = [
        f"# {title}",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        f"**Runs:** {len(ok)} completed, {len(blocked)} blocked/skipped  ",
        f"**Architectures:** {', '.join(archs)}  ",
        f"**Precisions:** {', '.join(_quant_display(q) for q in quants)}  ",
        f"**Bitwise (0.00e+00):** {len(bitwise)} / {len(measured)} parity-measured runs  ",
    ]
    if not_bitwise:
        lines.append(
            f"**Not bitwise:** {len(not_bitwise)} "
            f"-- NOT claimed as bitwise; see the parity table."
        )
    lines.append("")
    return "\n".join(lines)


def _section_hardware(records: List[Dict[str, Any]]) -> str:
    """Hardware info section (extracted from records or a separate dict)."""
    gpu_names = sorted({r.get("hardware", "unknown") for r in records if r.get("hardware")})
    if not gpu_names:
        return ""
    lines = [
        "## Hardware",
        "",
    ]
    for gpu in gpu_names:
        subset = [r for r in records if r.get("hardware") == gpu]
        vram_total = next((r.get("vram_total_mb") for r in subset if r.get("vram_total_mb")), None)
        vram_str = f" ({vram_total / 1024:.0f} GB)" if vram_total else ""
        lines.append(f"- **{gpu}**{vram_str}")
    lines.append("")
    return "\n".join(lines)


def _section_methods(records: List[Dict[str, Any]]) -> str:
    """Methods overview table.

    Only rendered when records carry explicit ``method`` fields (i.e.,
    they came from RunArtifact objects with multiple methods compared).
    Raw JSONL from ``poros_grid_pro.jsonl`` has no ``method`` field --
    each record IS the Poros run and references resident/unsloth inline
    as comparison columns, so a methods table would be misleading.
    """
    ok = [r for r in records if _is_ok(r)]
    # Only show this section when at least one record has an explicit method
    has_explicit_method = any("method" in r for r in ok)
    if not has_explicit_method:
        return ""

    methods = sorted({r.get("method", "poros") for r in ok}, key=str)
    lines = [
        "## Methods",
        "",
        "| Method | Runs |",
        "|--------|------|",
    ]
    for m in methods:
        count = sum(1 for r in ok if r.get("method", "poros") == m)
        lines.append(f"| {m} | {count} |")
    lines.append("")
    return "\n".join(lines)


def _section_vram(records: List[Dict[str, Any]]) -> str:
    """VRAM comparison table -- the hero section."""
    ok = _poros_rows(records)
    if not ok:
        return ""

    # A directory can mix campaigns from different GPUs/protocols; without a
    # label the rows read as one comparison. Add the column only when needed.
    multi_hw = len({r.get("hardware") or "?" for r in ok}) > 1
    hw_head = " Hardware |" if multi_hw else ""
    hw_rule = "----------|" if multi_hw else ""
    lines = [
        "## VRAM (Peak, MB)",
        "",
        "| Model |" + hw_head + " Precision | Resident | Unsloth "
        "| Poros | vs Resident | vs Unsloth | Note |",
        "|-------|" + hw_rule + "-----------|----------|---------"
        "|-------|-------------|------------|------|",
    ]
    for r in ok:
        name = r.get("display_name", r.get("tag", "?"))
        if multi_hw:
            name = f"{name} | {r.get('hardware') or '?'}"
        quant = _quant_display(r.get("precision", r.get("quant", "?")))
        res = _fmt_mb(r.get("resident_vram_mb"))
        uns = _fmt_mb(r.get("unsloth_vram_mb"))
        por = _fmt_mb(r.get("poros_vram_mb"))
        vs_res = _fmt_pct(r.get("vram_save_vs_resident_pct"))
        vs_uns = _fmt_pct(r.get("vram_save_vs_unsloth_pct"))
        note = _parity_note(r).strip()
        lines.append(
            f"| {name} | {quant} | {res} | {uns} | {por} | {vs_res} | {vs_uns} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_speed(records: List[Dict[str, Any]]) -> str:
    """Speed comparison table."""
    ok = _poros_rows(records)
    if not ok:
        return ""

    # Only include if speed data is present. A poros-only run has a real
    # s_per_step but no resident baseline to compare against -- its measured
    # speed must still publish.
    has_speed = any(
        r.get("s_per_step") or r.get("slowdown_vs_resident_x") is not None
        for r in ok
    )
    if not has_speed:
        return ""

    multipass = any((r.get("n_passes") or 1) > 1 for r in ok)
    lines = [
        "## Speed",
        "",
        "| Model | Precision | s/step (median ± stddev) | Passes | "
        "Slowdown vs Resident | Slowdown vs Unsloth |",
        "|-------|-----------|--------------------------|--------|"
        "---------------------|---------------------|",
    ]
    for r in ok:
        name = r.get("display_name", r.get("tag", "?"))
        quant = _quant_display(r.get("precision", r.get("quant", "?")))
        vs_res = _fmt_slowdown(r.get("slowdown_vs_resident_x"))
        vs_uns = _fmt_slowdown(r.get("slowdown_vs_unsloth_x"))
        lines.append(
            f"| {name} | {quant} | {_fmt_sstep(r)} | {r.get('n_passes') or 1} | "
            f"{vs_res} | {vs_uns} |"
        )
    if multipass:
        lines.append("")
        lines.append(
            "*s/step is the median across passes; ± is the population stddev "
            "(speed is noisy -- a single pass is not a defensible headline).*"
        )
    lines.append("")
    return "\n".join(lines)


def _section_parity(records: List[Dict[str, Any]]) -> str:
    """Parity table -- the trust-building section.

    Shows the real max|delta| for every cell.  Bitwise cells show
    0.00e+00.  Non-bitwise cells are flagged.
    """
    ok = _poros_rows(records)
    if not ok:
        return ""

    lines = [
        "## Parity (max |delta|)",
        "",
        "Parity is the maximum absolute difference in loss and adapter",
        "weights between blockwise Poros and the fully-resident reference.",
        "A value of **0.00e+00** means bitwise-identical training.",
        "",
        "| Model | Precision | Loss max|delta| | Weight max|delta| | Bitwise | Status |",
        "|-------|-----------|-----------------|-------------------|---------|--------|",
    ]
    for r in ok:
        name = r.get("display_name", r.get("tag", "?"))
        quant = _quant_display(r.get("precision", r.get("quant", "?")))
        loss_d = r.get("parity_loss_max_diff")
        wt_d = r.get("parity_weight_max_diff")
        bitwise = r.get("bitwise_exact", False)
        measured = bitwise or loss_d is not None or wt_d is not None
        bw_str = "0.00e+00" if bitwise else ("—" if not measured else "NO")

        if loss_d is None:
            loss_str = "—"
        else:
            loss_str = f"{loss_d:.2e}" if loss_d != 0.0 else "0.00e+00"
        if wt_d is None:
            wt_str = "—"
        else:
            wt_str = f"{wt_d:.2e}" if wt_d != 0.0 else "0.00e+00"

        status = "VALIDATED"
        if not measured:
            status = "n/a -- parity not measured in this cell"
        elif not bitwise:
            status = "NOT BITWISE"

        lines.append(
            f"| {name} | {quant} | {loss_str} | {wt_str} | {bw_str} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_oom(records: List[Dict[str, Any]]) -> str:
    """OOM / blocked / error table."""
    blocked = [r for r in records if _is_blocked(r)]
    if not blocked:
        return ""
    lines = [
        "## OOM / Blocked / Skipped",
        "",
        "| Model | Precision | Status | Reason |",
        "|-------|-----------|--------|--------|",
    ]
    for r in blocked:
        name = r.get("display_name", r.get("tag", "?"))
        quant = _quant_display(r.get("precision", r.get("quant", "?")))
        status = r.get("status", "blocked")
        reason = r.get("reason") or r.get("error_message") or "--"
        lines.append(f"| {name} | {quant} | {status} | {reason} |")
    lines.append("")
    return "\n".join(lines)


def _section_config(records: List[Dict[str, Any]]) -> str:
    """Training config section (shared across runs)."""
    ok = [r for r in records if _is_ok(r)]
    if not ok:
        return ""

    # Pull common config across records
    steps_vals = sorted({r.get("steps", "?") for r in ok}, key=str)
    seq_vals = sorted({r.get("seq_len", "?") for r in ok}, key=str)
    data_vals = sorted({r.get("data", r.get("dataset", "?")) for r in ok})

    lines = [
        "## Configuration",
        "",
        f"- **Steps:** {', '.join(str(s) for s in steps_vals)}",
        f"- **Sequence length:** {', '.join(str(s) for s in seq_vals)}",
        f"- **Dataset:** {', '.join(str(d) for d in data_vals)}",
    ]

    block_sizes = sorted({r.get("block_size") for r in ok if r.get("block_size") is not None})
    if block_sizes:
        lines.append(f"- **Block size (Poros):** {', '.join(str(b) for b in block_sizes)}")

    lines.append("")
    lines.append("---")
    lines.append("")
    from poros import __version__

    lines.append(
        f"*Report generated by [Poros](https://github.com/Caistro-Labs/poros) "
        f"v{__version__}. Source measurements come from the supplied run "
        "artifacts; ratios and percentages are derived from them.*"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_jsonl_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load and normalise records from a JSONL file.

    This is the primary entry point when working with raw benchmark
    output (e.g., ``poros_grid_pro.jsonl``).  Each line is parsed as
    JSON and normalised into the internal dict shape that the renderer
    understands.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of normalised record dicts.
    """
    raw = _load_jsonl(path)
    return [_normalise_record(r) for r in raw]


def generate_markdown_report(
    artifacts: Union[List[Any], str, Path],
    output_path: Union[str, Path],
    *,
    title: str = "Poros Benchmark Report",
    hardware_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a Markdown benchmark report.

    Accepts either:
    - a list of ``RunArtifact`` pydantic objects,
    - a list of plain dicts (from ``load_jsonl_records``), or
    - a path to a JSONL file (str or Path).

    Sections: Summary -> Hardware -> Methods -> VRAM -> Speed ->
    Parity table (0.00e+00) -> OOM -> Config.

    The parity column shows the real max|delta|; rows whose measured parity
    is not bitwise are flagged NOT BITWISE.

    Args:
        artifacts: Input data -- RunArtifact list, dict list, or JSONL path.
        output_path: Where to write the .md file.
        title: Report title.
        hardware_info: Optional hardware dict (gpu_name, vram_gb, etc.).

    Returns:
        Path to the generated Markdown file.
    """
    # -- resolve input -------------------------------------------------
    if isinstance(artifacts, (str, Path)):
        records = load_jsonl_records(artifacts)
    else:
        records = [_normalise_record(d) for d in _records_from_artifacts(artifacts)]

    # Inject hardware info if provided
    if hardware_info:
        for r in records:
            if not r.get("hardware"):
                r["hardware"] = hardware_info.get("gpu_name", "")
            if "vram_gb" in hardware_info:
                if not r.get("vram_total_mb"):
                    r["vram_total_mb"] = hardware_info["vram_gb"] * 1024

    # -- build sections ------------------------------------------------
    sections = [
        _section_summary(records, title),
        _section_hardware(records),
        _section_methods(records),
        _section_vram(records),
        _section_speed(records),
        _section_parity(records),
        _section_oom(records),
        _section_config(records),
    ]

    body = "\n".join(s for s in sections if s)

    # -- write ---------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out


def render_markdown_string(
    artifacts: Union[List[Any], str, Path],
    *,
    title: str = "Poros Benchmark Report",
    hardware_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the Markdown report as a string (no file write).

    Same arguments as ``generate_markdown_report`` except ``output_path``
    is not needed.
    """
    # -- resolve input -------------------------------------------------
    if isinstance(artifacts, (str, Path)):
        records = load_jsonl_records(artifacts)
    else:
        records = [_normalise_record(d) for d in _records_from_artifacts(artifacts)]

    if hardware_info:
        for r in records:
            if not r.get("hardware"):
                r["hardware"] = hardware_info.get("gpu_name", "")
            if "vram_gb" in hardware_info:
                if not r.get("vram_total_mb"):
                    r["vram_total_mb"] = hardware_info["vram_gb"] * 1024

    sections = [
        _section_summary(records, title),
        _section_hardware(records),
        _section_methods(records),
        _section_vram(records),
        _section_speed(records),
        _section_parity(records),
        _section_oom(records),
        _section_config(records),
    ]
    return "\n".join(s for s in sections if s)
