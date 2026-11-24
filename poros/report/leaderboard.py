"""Leaderboard aggregation for Poros benchmarks.

Ranks methods/configurations by VRAM efficiency, speed, and parity
status across models and precisions.  Produces a sorted list of
leaderboard entries that can be rendered as Markdown, HTML, or JSON.

Honest by design: an entry whose measured parity is not bitwise is tagged NOT BITWISE and
never ranked as "bitwise validated"; a speed/VRAM cell that measured no parity says so.

This module must work WITHOUT torch installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Helpers (shared logic with markdown.py, kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _quant_display(quant: str) -> str:
    _MAP = {
        "qlora4": "NF4",
        "nf4": "NF4",
        "int8": "INT8",
        "lora16": "BF16",
        "bf16": "BF16",
        "fp16": "FP16",
    }
    return _MAP.get(quant, quant.upper())


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


def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a raw record dict into stable field names."""
    r: Dict[str, Any] = dict(raw)
    _fill(r, "display_name", "tag", "model_id", "unknown")
    _fill(r, "precision", "quant", "nf4")
    _fill(r, "arch", "unknown")
    _fill(r, "resident_vram_mb", "vram_resident_mb")
    _fill(r, "poros_vram_mb", "vram_poros_mb", "vram_peak_mb")
    _fill(r, "unsloth_vram_mb", "vram_unsloth_mb")
    _fill(r, "vram_save_vs_resident_pct", "vram_saved_vs_resident_pct")
    _fill(r, "vram_save_vs_unsloth_pct", "vram_saved_vs_unsloth_pct")
    r.setdefault("slowdown_vs_resident_x", None)
    r.setdefault("slowdown_vs_unsloth_x", None)
    r.setdefault("parity_loss_max_diff", None)  # None = not measured in this cell
    r.setdefault("parity_weight_max_diff", None)
    r.setdefault("bitwise_exact", False)
    r.setdefault("status", "ok")
    return r


def _records_from_input(
    artifacts: Union[List[Any], str, Path],
) -> List[Dict[str, Any]]:
    """Accept RunArtifact list, dict list, or JSONL path -> normalised dicts."""
    if isinstance(artifacts, (str, Path)):
        raw: List[Dict[str, Any]] = []
        with open(artifacts) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return [_normalise(r) for r in raw]

    out: List[Dict[str, Any]] = []
    for a in artifacts:
        if hasattr(a, "model_dump"):
            out.append(_normalise(a.model_dump()))
        elif isinstance(a, dict):
            out.append(_normalise(a))
        else:
            out.append(_normalise(dict(a)))
    return out


# ---------------------------------------------------------------------------
# Leaderboard entry
# ---------------------------------------------------------------------------

class LeaderboardEntry:
    """A single row in the leaderboard.

    Attributes are plain types so the entry serialises cleanly to JSON.
    """

    __slots__ = (
        "rank",
        "display_name",
        "arch",
        "precision",
        "precision_display",
        "poros_vram_mb",
        "resident_vram_mb",
        "unsloth_vram_mb",
        "vram_save_vs_resident_pct",
        "vram_save_vs_unsloth_pct",
        "slowdown_vs_resident_x",
        "slowdown_vs_unsloth_x",
        "parity_loss_max_diff",
        "parity_weight_max_diff",
        "bitwise_exact",
        "status_tag",
        "steps",
    )

    def __init__(self, record: Dict[str, Any], rank: int = 0):
        self.rank = rank
        self.display_name: str = record.get("display_name", "?")
        self.arch: str = record.get("arch", "?")
        self.precision: str = record.get("precision", "?")
        self.precision_display: str = _quant_display(self.precision)
        self.poros_vram_mb: Optional[float] = record.get("poros_vram_mb")
        self.resident_vram_mb: Optional[float] = record.get("resident_vram_mb")
        self.unsloth_vram_mb: Optional[float] = record.get("unsloth_vram_mb")
        self.vram_save_vs_resident_pct: Optional[float] = record.get("vram_save_vs_resident_pct")
        self.vram_save_vs_unsloth_pct: Optional[float] = record.get("vram_save_vs_unsloth_pct")
        self.slowdown_vs_resident_x: Optional[float] = record.get("slowdown_vs_resident_x")
        self.slowdown_vs_unsloth_x: Optional[float] = record.get("slowdown_vs_unsloth_x")
        self.parity_loss_max_diff: Optional[float] = record.get("parity_loss_max_diff")
        self.parity_weight_max_diff: Optional[float] = record.get("parity_weight_max_diff")
        self.bitwise_exact: bool = record.get("bitwise_exact", False)
        self.steps: int = record.get("steps", 0)

        # Status tag: VALIDATED / NOT BITWISE / not measured.
        # Three-state on purpose: a speed/VRAM cell records no parity at all,
        # and tagging it NOT BITWISE reads as "the run is suspect" when the
        # truth is "parity was not measured in this cell".
        parity_measured = (
            self.parity_loss_max_diff is not None
            or self.parity_weight_max_diff is not None
            or self.bitwise_exact
        )
        if not parity_measured:
            self.status_tag = "parity not measured here"
        elif not self.bitwise_exact:
            self.status_tag = "NOT BITWISE"
        else:
            self.status_tag = "VALIDATED"

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


# ---------------------------------------------------------------------------
# Sorting / ranking
# ---------------------------------------------------------------------------

def _sort_key_vram_efficiency(entry: LeaderboardEntry) -> tuple:
    """Primary sort: best VRAM savings vs resident (descending).

    Tie-break: bitwise > not bitwise, then by name.
    """
    vram_pct = entry.vram_save_vs_resident_pct
    if vram_pct is None:
        vram_pct = 0.0
    return (
        -vram_pct,                   # higher savings first
        0 if entry.bitwise_exact else 1,  # bitwise first
        entry.display_name,
    )


def _sort_key_speed(entry: LeaderboardEntry) -> tuple:
    """Sort by lowest slowdown vs resident."""
    slowdown = entry.slowdown_vs_resident_x
    if slowdown is None:
        slowdown = float("inf")
    return (
        slowdown,
        entry.display_name,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_leaderboard(
    artifacts: Union[List[Any], str, Path],
    *,
    sort_by: str = "vram",
    include_not_bitwise: bool = True,
) -> List[Dict[str, Any]]:
    """Build a ranked leaderboard from benchmark artifacts.

    Ranks methods by VRAM efficiency (default) or speed across models.
    Only ``status == "ok"`` records are included.

    Args:
        artifacts: RunArtifact list, dict list, or path to a JSONL file.
        sort_by: ``"vram"`` (default) or ``"speed"``.
        include_not_bitwise: If False, exclude non-bitwise rows.

    Returns:
        Sorted list of leaderboard entry dicts, each containing:
        ``rank``, ``display_name``, ``arch``, ``precision``,
        ``poros_vram_mb``, ``vram_save_vs_resident_pct``,
        ``bitwise_exact``, ``status_tag``, etc.
    """
    records = _records_from_input(artifacts)
    # Poros cells only: the runner copies cross-references onto every
    # method's artifact, so without this filter resident/unsloth baseline
    # rows ranked as Poros results. Legacy rows without a method are poros.
    ok = [
        r for r in records
        if r.get("status") == "ok" and r.get("method", "poros") == "poros"
    ]

    entries = [LeaderboardEntry(r) for r in ok]

    if not include_not_bitwise:
        entries = [e for e in entries if e.bitwise_exact]

    if sort_by == "speed":
        entries.sort(key=_sort_key_speed)
    else:
        entries.sort(key=_sort_key_vram_efficiency)

    for i, entry in enumerate(entries, start=1):
        entry.rank = i

    return [e.to_dict() for e in entries]


def leaderboard_to_markdown(
    entries: List[Dict[str, Any]],
    *,
    title: str = "Poros Leaderboard",
) -> str:
    """Render a leaderboard (from ``build_leaderboard``) as Markdown.

    Args:
        entries: List of leaderboard entry dicts.
        title: Heading text.

    Returns:
        Markdown string.
    """
    if not entries:
        return f"# {title}\n\nNo results to display.\n"

    def _fmt(val: Optional[float], suffix: str = "") -> str:
        if val is None:
            return "--"
        return f"{val:,.0f}{suffix}" if suffix == " MB" else f"{val:.1f}{suffix}"

    lines = [
        f"# {title}",
        "",
        "| Rank | Model | Arch | Precision | Poros VRAM | vs Resident | vs Unsloth | Slowdown | Parity | Status |",
        "|------|-------|------|-----------|-----------|-------------|------------|----------|--------|--------|",
    ]

    for e in entries:
        rank = e.get("rank", "?")
        name = e.get("display_name", "?")
        arch = e.get("arch", "?")
        prec = e.get("precision_display", "?")
        vram = _fmt(e.get("poros_vram_mb"), " MB")
        vs_res = _fmt(e.get("vram_save_vs_resident_pct"), "%")
        vs_uns = _fmt(e.get("vram_save_vs_unsloth_pct"), "%")

        slowdown = e.get("slowdown_vs_resident_x")
        slow_str = f"{slowdown:.2f}x" if slowdown is not None else "--"

        bitwise = e.get("bitwise_exact", False)
        loss_diff = e.get("parity_loss_max_diff")
        if bitwise:
            parity = "0.00e+00"
        elif loss_diff is None:
            parity = "not measured"
        else:
            parity = "drift"

        status = e.get("status_tag", "?")

        lines.append(
            f"| {rank} | {name} | {arch} | {prec} | {vram} | {vs_res} | {vs_uns} | {slow_str} | {parity} | {status} |"
        )

    from poros import __version__

    lines.append("")
    lines.append(
        f"*Leaderboard generated by Poros v{__version__}. "
        "All numbers from measured runs.*"
    )
    lines.append("")
    return "\n".join(lines)


def leaderboard_to_json(
    entries: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Write leaderboard entries to a JSON file.

    Args:
        entries: Leaderboard dicts from ``build_leaderboard``.
        output_path: Destination file path.

    Returns:
        Path to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return out
