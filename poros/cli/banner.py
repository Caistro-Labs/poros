"""Poros banner -- animated ASCII wordmark for the interactive shell.

Torch-free: stdlib + rich only. This module must never import poros ML
code. Animation is strictly TTY-gated by the caller (the shell entry in
``poros.cli.main``) and defensively re-checked here; it also honors
``NO_COLOR``, ``TERM=dumb``, and ``POROS_NO_ANIM``. The full sweep is
hard-capped well under one second and any keypress skips it instantly.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console

from poros.cli import theme

ACCENT = theme.ACCENT  # the brand orange, color(208) (see theme.py)

# ANSI-shadow "POROS" wordmark (42 columns x 6 rows).
LOGO: list[str] = [
    "██████╗  ██████╗ ██████╗  ██████╗ ███████╗",
    "██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗██╔════╝",
    "██████╔╝██║   ██║██████╔╝██║   ██║███████╗",
    "██╔═══╝ ██║   ██║██╔══██╗██║   ██║╚════██║",
    "██║     ╚██████╔╝██║  ██║╚██████╔╝███████║",
    "╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
]
LOGO_WIDTH = max(len(row) for row in LOGO)

# Total animation duration (seconds). Hard cap: well under 1s.
_DURATION = 0.72
_FRAME_DT = 1 / 30
_FADE = 9.0  # sweep-head width in columns


def color_enabled() -> bool:
    """True when color output is appropriate (NO_COLOR / dumb-term aware)."""
    return "NO_COLOR" not in os.environ and os.environ.get("TERM", "") != "dumb"


def _gpu_line() -> str | None:
    """One local hardware line ("RTX 4090 · 23.6 GiB free"), or None.

    Read via nvidia-smi only: local and fast, and this module must stay
    torch-free. Deliberately nothing personal -- no username, no date/time,
    no network lookups; CLI users read those as tracking.
    """
    if sys.platform == "darwin":
        from poros.metal.table import this_mac

        mac = this_mac()
        return mac.name if mac else None
    from poros.core.card import nvidia_smi

    rows = nvidia_smi("gpu=name,memory.free", timeout=0.4)
    if not rows:
        return None
    try:
        *name_parts, free_mib = rows[0]
        name = ", ".join(name_parts)
        free_gb = int(free_mib) / 1024
    except ValueError:
        return None
    name = name.strip().removeprefix("NVIDIA ").removeprefix("GeForce ")
    extra = f"  (+{len(rows) - 1} more)" if len(rows) > 1 else ""
    return f"{name} · {free_gb:.1f} GiB free{extra}"


def print_banner(console: Console, version: str, animate: bool = True) -> None:
    """The shell's welcome: the caistrolabs arch mark in pixels, words beside
    it, then the key hints (the CLI visual design). Only the shell
    entry calls this."""
    from poros.cli.ui import components as c
    from poros.cli.ui import intro
    from poros.cli.ui.style import resolve
    from poros.cli.ui.text import S, ln, render_line, width

    out = console.file
    pal = resolve(out, depth=None if (color_enabled() and console.is_terminal) else "none")
    from poros.cli.ui.live import columns

    # the terminal's width now, not rich's (which trusts a stale $COLUMNS)
    w = columns() if console.is_terminal else console.width
    gpu = _gpu_line()
    line1 = "Fine-tune models bigger than your GPU."
    line2 = gpu or "What are we training today?"
    hint = ln(S("  ", None), S("↑↓", "tx"), S(" pick a command · ", "dm"), S("⏎", "tx"),
              S(" opens its card · ", "dm"), S("type", "tx"),
              S(" to filter", "dm"))
    if width(hint) > w:
        hint = ln(S("  ", None), S("↑↓", "tx"), S(" pick · ", "dm"), S("⏎", "tx"),
                  S(" open", "dm"))
    if w < intro.MIN_WIDTH:
        out.write("\n" + render_line(ln(S(c.MARK_INLINE, "a"), S(" ", None), S("poros", "hi")), pal)
                  + "\n")  # no version on the home screen: poros --version says it
    elif animate and pal.depth not in ("none", "plain") and intro.enabled(out):
        intro.play(out, pal, w, line1, line2)
    else:
        out.write("\n".join(render_line(r, pal) for r in intro.final(w, line1, line2)) + "\n")
    if w < intro.BOX_MIN:  # the box carries its own keys
        out.write("\n" + render_line(hint, pal) + "\n")
    out.write("\n")
    out.flush()
