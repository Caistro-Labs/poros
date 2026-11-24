"""The parts every screen is built from: the width (``_w``), a table, a rail of text, a command
line, how a figure, a knob and the nearest receipt are said.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Iterable, List, Optional, Sequence

from poros.cli.ui import components as c
from poros.cli.ui.text import Line, S, ln

W = c.CAP


def _w(w: int) -> int:
    return min(w, c.CAP)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    right: Iterable[int] = (),
    styles: Optional[Sequence[Sequence[Any]]] = None,
    w: int = W,
    rail: bool = True,
    keep: Iterable[int] = (),
) -> List[Line]:
    """A borderless table on the rail: dim header row, tabular columns,
    numbers right-aligned. Columns shrink from the widest text column when
    the terminal is narrow."""
    right = set(right)
    keep = set(keep)  # columns never shortened (an id you have to type)
    cols = len(headers)
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    gap = 3
    budget = _w(w) - (len(c.RAIL) if rail else 0)
    while sum(widths) + gap * (cols - 1) > budget:
        i = max((j for j in range(cols) if j not in right | keep), key=lambda j: widths[j],
                default=None)
        if i is None or widths[i] <= 8:
            break
        widths[i] -= 1

    def cell(text: str, i: int) -> str:
        t = str(text)
        if len(t) > widths[i]:
            t = t[: widths[i] - 1] + "…"
        return t.rjust(widths[i]) if i in right else t.ljust(widths[i])

    out: List[Line] = []
    head: Line = []
    for i, h in enumerate(headers):
        head += [(cell(h, i), "dm"), (" " * gap if i < cols - 1 else "", None)]
    out.append(c.rail(head) if rail else head)
    for ri, r in enumerate(rows):
        line: Line = []
        for i, v in enumerate(r):
            st = (
                styles[ri][i]
                if styles and styles[ri][i] is not None
                else ("hi" if i == 0 else "tx")
            )
            line += [(cell(v, i), st), (" " * gap if i < cols - 1 else "", None)]
        out.append(c.rail(line) if rail else line)
    return out


_SOURCE_TAIL = {
    "measured": "measured",
    "estimated": "estimated",
    "hub_metadata": "from the Hub",
    "local": "on disk",
    "unknown": "unknown",
}


def _figure_value(f: Any) -> str:
    if f.value is None:
        return "—"
    if f.unit == "s" and f.value >= 60:
        return c.duration(f.value)
    if f.unit == "GiB":  # a peak: never shown below it
        return f"{c.ceil1(f.value):.1f} GiB"
    return f"{f.value:g} {f.unit}"


def _download_value(f: Any) -> str:
    """A plan's download (its figure is GiB) as every screen says a download: decimal GB."""
    from poros.preflight.disk import disk_text

    if f.value is None:
        return "—"
    return disk_text(f.value * 2**30 / 1e9) if f.value else "0 GB"


def rail_text(text: str, w: int, style: str = "dm") -> List[Line]:
    """Prose on the rail, wrapped at word boundaries (a path is never split)."""
    room = max(20, _w(w) - len(c.RAIL))
    return [c.rail(S(line, style)) for line in c.wrap(text, room)]


def command_lines(cmd: str, w: int, indent: str = "  ", style: str = "dm") -> List[Line]:
    """A command to copy: one line when it fits, else broken before flags with
    shell continuations, so pasting it still runs it."""
    cmd = c.command(cmd)
    room = _w(w) - len(indent)
    if len(cmd) <= room:
        return [ln(S(indent + cmd, style))]
    parts, cur = [], ""
    for tok in re.split(r" (?=--?[a-z])", cmd):
        if cur and len(cur) + 1 + len(tok) + 4 > room:
            parts.append(cur)
            cur = tok
        else:
            cur = f"{cur} {tok}" if cur else tok
    parts.append(cur)
    last = len(parts) - 1
    return [ln(S(indent + ("  " if i else "") + part + (" \\" if i < last else ""), style))
            for i, part in enumerate(parts)]


def _needs_gpu() -> str:
    return "needs Apple silicon" if sys.platform == "darwin" else "needs a CUDA GPU"


def _unified(p: Any) -> bool:
    return p.gpu is not None and p.gpu.unified


_KNOB = {"seq_len": "length", "lora_rank": "rank", "lora_dropout": "dropout", "batch": "batch",
         "block_size": "block size", "load_strategy": "load", "selfcheck": "startup self-check",
         "quantization": "quantization", "entry": "path"}


def _knob(name: str, value: Any) -> str:
    if name == "selfcheck":
        value = {True: "on", False: "off", None: "unknown"}.get(value, value)
    return f"{_KNOB.get(name, name)} {value}"


def _nearest(p: Any, w: int) -> List[Line]:
    """Beside an unknown verdict: what the closest receipt measured, and how it differs."""
    n = p.nearest
    if n is None:
        return []
    peak = (f"{c.ceil1(n.peak_gib):.1f} GiB allocated" if n.peak_gib is not None
            else "no VRAM recorded")
    differs = ", ".join(_knob(k, b) for k, (a, b) in n.differs.items())
    text = f"nearest receipt: {n.receipt} on {n.gpu}, {peak}"
    return rail_text(text + (f" (it ran {differs})" if differs else ""), w)
