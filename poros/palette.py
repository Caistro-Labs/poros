"""The brand accent (``ACCENT``) and the color system a Rich console uses here
(``console_color_system``), for the CLI theme (``poros.cli.theme``). Torch-free.
"""

from __future__ import annotations

import os
import sys

# The brand accent as a Rich token. It never flips between themes, so it's a
# plain constant -- use it anywhere a ``#FF6B35`` style is wanted (that hex
# would otherwise render pink).
ACCENT = "color(208)"


def console_color_system() -> str:
    """``color_system`` for every CLI ``rich.Console``: ``"256"`` on a real,
    non-dumb tty, else ``"auto"``.

    Rich auto-detects unrecognised terminals (e.g. ``TERM=xterm-ghostty`` with
    no ``COLORTERM``) as 16-color ``standard``, where orange collapses to bright
    red. Pinning 256 (universally supported) keeps the accent orange; ``auto``
    on a non-tty leaves pipes/redirects uncolored. ``NO_COLOR`` is honored too.
    """
    if sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb":
        return "256"
    return "auto"
