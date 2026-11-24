"""What the session's screen and tabs share with it: the menu and the prompt, the key names, the
picker, the words that end a session, and the narrow-text helpers.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from poros.cli.ui.text import Line, S, width
from poros.core.env import outside_tool

PICK_ROWS = 12  # the commands and the next step fit without scrolling


PROMPT = "▛▜ ❯ "


def prompt(busy: bool = False) -> Line:
    """The arch mark, in the brand, and the caret: ``▛▜ ❯``. While a command
    runs the mark breathes (the ``mark`` loader) and the caret rests."""
    if not busy:
        return [("▛▜", "a"), (" ", None), ("❯", "tx"), (" ", None)]
    from poros.cli.ui import loaders

    still = bool(os.environ.get("POROS_NO_ANIM"))
    mark = [("▛▜", "a")] if still else loaders.mark(int(time.monotonic() * 10))[:1]
    return mark + [(" ", None), ("❯", "rl"), (" ", None)]


# The home list on an empty prompt: the commands, in the order of the work,
# the step the last one suggested first; ⏎ runs the focused one.
HOME = (
    ("/recipes", "a whole setup, ready: pick one, start on your data"),
    ("/train", "set up your own: model, data, every setting"),
    ("/eval", "score the model you trained beside the original: loss, and answers that end"),
    ("/runs", "watch, stop, continue, see how each went"),
    ("/models", "what this machine can train"),
    ("/data", "what a data file holds and would train"),
    ("/export", "turn a run into a model you can serve"),
    ("/machine", "what this machine can do"),
    ("/settings", "what every command opens with"),
)


# The only commands the session lists, in the order of the work. Anything
# else still runs when typed in full; it is just never offered. Each says
# what the home list says of it, so a command reads the same everywhere.
_SAYS = dict(HOME)


MENU = tuple((n, _SAYS["/" + n]) for n in (
    "recipes", "train", "data", "eval", "models", "runs", "export", "machine", "settings",
)) + (("prove", "see streaming change nothing, bit for bit"),
      ("share", "your last finished run as a recipe file, to send to anyone"))


# What a suggested next step says beside it.
NEXT = {"eval": "how the run you just trained does, beside its base",
        "export": "turn it into a model to serve",
        "runs": "how it went, step by step", "report": "how it went, step by step",
        "train": "its settings, ready to start",
        "data": "what it holds and would train"}


# commands that open a card (a sheet or a list) in a terminal
# Quick cards that open over the tab you are in and make no tab of their own.
OVER = frozenset({"settings", "theme"})


def _blank(w: int) -> List[Line]:
    """A tab opened with ctrl+n, before a command lands in it."""
    tip = "type a command · ⏎ lists them"
    return [[S(" " * max(0, (w - len(tip)) // 2)), S(tip, "dm")]]


TRAINS = ("train", "prove")  # tabs whose work is a training run


Keys = Union[str, List[Tuple[str, str]]]  # a note, or (key, what it does) pairs


ARM_S = 2.0  # how long "esc again" / "ctrl+c again" stays armed


EXIT_WORDS = {"quit", "exit", "q"}


@dataclass
class Picker:
    kind: str  # "commands" | "results"
    items: List[Tuple[str, str]]  # (label, description)
    token_start: int = 0
    total: int = 0


def _known() -> set:
    from poros.cli.ui.complete import command_tree

    return set(command_tree()) | {"share"}


def _unknown(head: str) -> Line:
    """``✕ /hello isn't a command`` and the listed one it was nearly."""
    import difflib

    near = difflib.get_close_matches(head, [n for n, _ in MENU], n=1)
    tail = ([("   did you mean ", "dm"), ("/" + near[0], "hi"), ("?", "dm")] if near
            else [("   ⏎ lists every command", "dm")])
    return [("✕ ", "e"), ("/" + head, "hi"), (" isn't a command", "tx")] + tail


def _only_error(lines) -> Line:
    """A tab whose whole output is one short error (``✕ No runs yet`` and its
    rails), as one line for the status row; [] when it showed anything else."""
    body = [x for x in lines if x]
    if not body or len(body) > 4 or not all(isinstance(x, list) for x in body):
        return []
    if not body[0] or not body[0][0][0].startswith("✕"):
        return []
    out: Line = list(body[0][:2])  # the mark and what went wrong
    for row in [body[0][2:], *body[1:]]:  # then each part, three spaces apart
        segs = [(t.strip(), st) for t, st in row if t.strip() not in ("", "│", "fix")]
        if segs:
            out += [("   ", "dm")] + [(" " * bool(i) + t, st) for i, (t, st) in enumerate(segs)]
    return out


def _workspace() -> str:
    cwd = Path.cwd()
    home = Path.home()
    p = "~/" + str(cwd.relative_to(home)) if cwd.is_relative_to(home) and cwd != home else str(cwd)
    if len(p) > 32:
        p = "…" + p[-31:]
    try:
        br = subprocess.run(
            [outside_tool("git"), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=0.3,
            cwd=cwd,
        ).stdout.strip()
    except Exception:
        br = ""
    return f"{p} ({br})" if br else p


def _cols() -> int:
    from poros.cli.ui.live import columns

    return columns()


def _clip(line: Line, w: int) -> Line:
    out: Line = []
    left = w
    for t, st in line:
        if st == "raw":  # takes no cells
            out.append((t, st))
            continue
        if left <= 0:
            break
        out.append((t[:left], st))
        left -= len(t)
    return out


def _spread(left: Line, right: Line, w: int) -> Line:
    gap = w - width(left) - width(right)
    if gap < 1:
        return left
    return left + [(" " * gap, None)] + right


def _fit_text(text: str, room: int) -> str:
    return text if len(text) <= room else text[: max(0, room - 1)] + "…"
