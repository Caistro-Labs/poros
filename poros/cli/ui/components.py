"""The grammar's building blocks (the CLI visual design).

Every function returns ``Line`` data; nothing here writes to a terminal.
"""

from __future__ import annotations

import math
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

from poros.cli.ui.text import Line, S, ln, merge, width

# The letters that are Poros actions (save, fit, get, export, resume, train,
# stop, new tab, close tab, a run's checkpoints). Each goes with one modifier: alt on Linux, ctrl
# on a Mac, where option types characters (ß, ƒ, @ on many layouts) unless the
# terminal is set to send it as alt. Both always work; the hints say the one.
APP_LETTERS = "efgnprstwx"


def mod() -> str:
    """``alt`` or ``ctrl``: the key the hints pair the app's letters with.
    Option on a Mac (beside command, under the thumb), ctrl on Linux.
    ``POROS_KEYS=alt|ctrl`` picks the other. Both reach every action."""
    want = os.environ.get("POROS_KEYS", "").strip().lower()
    if want in ("alt", "ctrl"):
        return want
    return "alt" if sys.platform == "darwin" else "ctrl"


def key_name(kind: str) -> str:
    """``ctrl-s`` as the hints say it: ``ctrl+s`` on Linux; on a Mac in the
    Mac's own symbols, ``⌥S``, as the hints already draw ⏎ and ↑↓.
    ctrl+c, ctrl+d and the line-editing keys stay ctrl everywhere
    (``ctrl+c`` on Linux, ``⌃C`` on a Mac)."""
    if not kind.startswith("ctrl-") or len(kind) != 6:
        return kind.replace("ctrl-", "ctrl+")
    letter = kind[-1]
    held = mod() if letter in APP_LETTERS else "ctrl"
    if mac_glyphs():
        return ("⌥" if held == "alt" else "⌃") + letter.upper()
    return f"{held}+{letter}"


def mac_glyphs() -> bool:
    """Key hints in the Mac's own symbols (⌃S), as every Mac app writes them."""
    return sys.platform == "darwin"

VERSION_FALLBACK = "dev"
CAP = 100  # content never stretches past this, even in a 140-col terminal
RAIL = "  │ "

# The arch mark (caistrolabs brand/mark.svg, 25x19) at 13x10 pixels in half blocks.
MARK = [
    "██████ ██████",
    "██████ ██████",
    "█████▀ ▀█████",
    "██▀       ▀██",
    "██         ██",
]
MARK_INLINE = "▛▜"


def version() -> str:
    try:
        from poros import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - partial installs
        return VERSION_FALLBACK


def num(n: float) -> str:
    return f"{n:,}"


def steps(n: int) -> str:
    """``1 step``, ``16 steps``: a count of steps, its noun in the number it takes."""
    return f"{n:,} step" + ("" if n == 1 else "s")


# The trainer measures memory in decimal GB; every screen says GiB, as the
# fit checks and the catalog do, so one number never shows in two units.
GIB = 1e9 / 2**30


def gib(v: float) -> float:
    """Decimal GB as GiB."""
    return v * GIB


def gb(v: float) -> str:
    return f"{gib(v):.1f} GiB"


def ceil1(v: float) -> float:
    """``v`` to 0.1 rounded up: a peak, an upper bound or a fit shown never reads below its
    figure (8.539 GiB shows as 8.6). The JSON keeps the figure as it is."""
    return math.ceil(round(v * 10, 6)) / 10


def floor1(v: float) -> float:
    """``v`` to 0.1 rounded down: a floor (``≥``, a run no larger's peak) never reads above
    what it bounds."""
    return math.floor(round(v * 10, 6)) / 10


def duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{s}s"
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# --------------------------------------------------------------------------
# headers
# --------------------------------------------------------------------------


# Set by the interactive session: the typed ``▛▜ ❯ <command>`` line already
# carries the mark, so a command's own header becomes a subtitle under it.
in_session = False
# The last "Continue with" a command printed in the session, when it is a
# slash command: the empty prompt's ghost, and what enter on it runs.
next_step: Optional[str] = None


def welcome(cmd: str, *items: str, max_width: int = CAP) -> Line:
    """``▛▜ poros v1.0.0 · item · item`` -- items drop from the right to fit.

    In the session: just the items, dim, aligned under the typed command."""
    keep = [i for i in items if i]
    if in_session:
        while keep and 5 + len(" · ".join(keep)) > max_width:
            keep.pop()
        segs: Line = [("     ", None)] if keep else []
        for i, it in enumerate(keep):
            if i:
                segs.append((" · ", "rl"))
            segs.append((it, "dm"))
        return segs
    while True:
        segs = ln(S(MARK_INLINE, "a"), S(" ", None), S(cmd, "hi"), S(f" v{version()}", "dm"))
        for it in keep:
            segs += ln(S(" · ", "rl"), S(it, "dm"))
        if width(segs) <= max_width or not keep:
            return segs
        keep.pop()


def banner(cmd: str, line1: str, line2: str, w: int) -> List[Line]:
    """First-run banner: the arch mark in pixels with words beside it.

    Below 70 columns it degrades to the one-line welcome.
    """
    if w < 70:
        return [welcome(cmd, max_width=w), []]
    right = [
        ln(S(cmd, "hi"), S(f"  v{version()}", "dm")),
        [],
        ln(S(line1, "tx")),
        ln(S(line2, "dm")),
        [],
    ]
    return [ln(S(m, "a"), S("     ", None), r) for m, r in zip(MARK, right)] + [[]]


# --------------------------------------------------------------------------
# transcript
# --------------------------------------------------------------------------


def event(
    verb: str,
    obj: Optional[str] = None,
    tail: Optional[str] = None,
    mark: str = "●",
    mark_style="tx",
) -> Line:
    """``● Verb object  tail`` -- a finished thing, printed into scrollback."""
    return ln(
        S(mark + " ", mark_style),
        S(verb, "hi"),
        S(" " + obj, "tx") if obj else None,
        S("  " + tail, "dm") if tail else None,
    )


def rail(*parts) -> Line:
    return ln(S(RAIL, "rl"), *parts)


def kv(
    label: str, value: str, *, lw: int = 10, vw: int = 9, style="tx", tail: Optional[str] = None
) -> Line:
    """A rail row: dim label column, right-aligned value, dim tail."""
    return rail(
        S(label.ljust(lw), "dm"), S(value.rjust(vw), style), S("  " + tail, "dm") if tail else None
    )


def short_path(path: str) -> str:
    """A path as people read it: relative to here, else ~/ (a /tmp that is
    /private/tmp too). Display only: commands and JSON keep it whole."""
    import os
    from pathlib import Path

    p = Path(path)
    if not p.is_absolute():
        return path
    real = Path(os.path.realpath(p))
    for base, label in ((Path.cwd(), ""), (Path.home(), "~/")):
        for mine in (p, real):
            try:
                rel = str(mine.relative_to(base))
            except ValueError:
                continue
            return (label + rel) if rel != "." else (label.rstrip("/") or ".")
    return path


def rail_path(label: str, path: str, w: int, lw: int = 10) -> List[Line]:
    """Paths are never truncated: a path that doesn't fit gets its own rail line."""
    path = short_path(path)
    if len(RAIL) + lw + len(path) <= w:
        return [rail(S(label.ljust(lw), "dm"), S(path))]
    return [rail(S(label, "dm")), rail(S(path))]


def _subcommands() -> frozenset:
    """The words a command shown here starts with after ``poros``: every registered command
    (poros.cli.registry, so none is shown without its ``poros``), and the words of a run's own
    that a shown command can start with."""
    from poros.cli import registry

    return frozenset(registry.names()) | {"stop", "continue", "download"}


_SUBCOMMANDS = _subcommands()


def command(cmd: str) -> str:
    """A command as the user types it here: ``poros train Qwen2.5-72B`` in a
    shell, ``/train Qwen2.5-72B`` inside the session (slash commands only)."""
    bare = cmd[len("poros "):] if cmd.startswith("poros ") else cmd
    if in_session:
        return "/" + bare if bare.split(" ", 1)[0] in _SUBCOMMANDS else bare
    from poros.cli.contract import as_agent

    return as_agent(f"poros {bare}" if bare.split(" ", 1)[0] in _SUBCOMMANDS else cmd)


# A next step that names a command this release removed, and the one that answers it now (a
# Mac prove's screen, poros.metal.prove, still offers fit).
REPLACED = {"fit": "recipes"}


def continue_with(cmd: str, w: int) -> List[Line]:
    global next_step
    cmd = command(REPLACED.get(cmd, cmd))
    if in_session and cmd.startswith("/"):  # a <placeholder> is left for its sheet to ask
        next_step = " ".join(t for t in cmd.split(" ") if not (t[:1] == "<" and t[-1:] == ">"))
        return []  # it heads the home list under this, ⏎ away
    elif in_session and next_step is None:  # a shell command can't run here: no ghost
        next_step = ""
    if in_session:  # not a slash command: it runs after you leave
        return [ln(S("  In a shell  ", "dm"), S(cmd, "hi"))] if len(cmd) + 14 <= w else [
            ln(S("  In a shell", "dm")), ln(S("  " + cmd, "hi"))]
    if len("Continue with: ") + len(cmd) <= w:
        return [ln(S("Continue with: ", "dm"), S(cmd, "hi"))]
    # Too long: the command below, broken with shell continuations so it
    # still pastes as one command.
    lines, cur = [], ""
    for word in cmd.split(" "):
        if cur and 4 + len(cur) + 1 + len(word) + 2 > w:  # indent + word + " \\"
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    lines.append(cur)
    last = len(lines) - 1
    return [ln(S("Continue with:", "dm"))] + [
        ln(S(("  " if i == 0 else "    ") + part + (" \\" if i < last else ""), "hi"))
        for i, part in enumerate(lines)
    ]


def wrap(text: str, room: int) -> List[str]:
    """Words into lines of at most ``room`` columns; a word longer than that
    (a path) keeps its own line, never split."""
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > room:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    return lines + ([cur] if cur else [])


def error_event(title: str, reason: str, w: int) -> List[Line]:
    head = ln(S("✕ ", "e"), S(title, "hi"))
    msg = "  " + reason
    if width(head) + len(msg) <= w:
        return [head + [(msg, "tx")]]
    return [head] + [rail(S(line)) for line in wrap(reason, max(20, w - len(RAIL)))]


# --------------------------------------------------------------------------
# live surface pieces
# --------------------------------------------------------------------------


# The interactive session replaces command statuslines with its own footer.
status_hook = None


def statusline(items: Sequence[str], w: int, tag: str = "") -> List[Line]:
    """Full-width rule + dim items; items drop from the right to fit."""
    if status_hook is not None:
        return status_hook(items, w, tag)
    return key_hints(items, w, tag)


def key_hints(items: Sequence[str], w: int, tag: str = "") -> List[Line]:
    """The rule + dim items row itself (never replaced by the session)."""
    keep = [i for i in items if i]
    lead = len(tag) + 2 if tag else 0
    while keep and 1 + lead + len(" · ".join(keep)) > w:
        keep.pop()
    segs: Line = [(" ", None)]
    if tag:
        segs += [(tag, "dm"), ("  ", None)]
    for i, it in enumerate(keep):
        if i:
            segs.append((" · ", "rl"))
        segs.append((it, "dm"))
    return [[("─" * w, "rl")], segs]


def keyline(keys: Sequence[Tuple[str, str]], w: int, tag: str = "") -> List[Line]:
    """The rule, then the tag and ``(key, what it does)`` pairs: the key in
    ink (⏎ in the accent, it is the one that runs), what it does dim."""
    keep = list(keys)
    lead = len(tag) + 3 if tag else 0
    while keep and 1 + lead + keys_width(keep) > w:
        keep.pop(-2 if len(keep) > 2 else -1)  # esc, the way out, is the last to go
    segs: Line = [(" ", None)]
    if tag:
        segs += [(tag, "dm"), ("   ", None)]
    return [[("─" * w, "rl")], segs + key_segs(keep)]


def keys_width(keys: Sequence[Tuple[str, str]]) -> int:
    return len("   ".join(f"{k} {d}" for k, d in keys))


def key_segs(keys: Sequence[Tuple[str, str]]) -> Line:
    """``⏎ run   ↑↓ move``: every key line in Poros says its keys this way."""
    segs: Line = []
    for i, (k, d) in enumerate(keys):
        if i:
            segs.append(("   ", None))  # space, not dots: the keys are the marks
        segs += [(k, "a" if k == "⏎" else "tx"), (" " + d, "dm")]
    return segs


METER = "━"  # a meter's fill; its track is ─


def chips(frac: Optional[float], n: int = 16, style="a", track="tk", lit: bool = False) -> Line:
    """A quiet meter of ``n`` cells, ``━━━━────``: how much of a whole is taken.
    The river belongs to training, so a meter is grey: ``style`` says what the
    figure is (``a`` real, ``dm`` estimated, ``tx`` neutral, ``e`` won't fit)
    and only won't-fit keeps its red. ``lit`` puts it in the accent: the row
    under the cursor, or a card's one result. Anything above nothing fills
    one cell; ``None`` is the track alone."""
    if frac is None:
        return [("─" * n, track)]
    frac = min(max(frac, 0.0), 1.0)
    full = min(n, max(1, round(n * frac))) if frac > 0.004 else 0
    ink = "e" if style == "e" else "a" if lit else "me" if style == "dm" else "mf"
    return [x for x in ((METER * full, ink), ("─" * (n - full), track)) if x[0]]


def light(line: Line) -> Line:
    """The focused row's meter, lit: its grey fill turns to the accent."""
    return [(t, "a") if t and set(t) == {METER} and st != "e" else (t, st) for t, st in line]


# a version is a name, not a reading: "macOS 27.2" and "PCIe 5" stay one tone
_VERSIONED = "macOS |Linux |Windows |Python |PCIe |CUDA |driver |torch "
_FIGURE = re.compile("".join(f"(?<!{v})" for v in _VERSIONED.split("|"))
                     + r"(?<![\w.\-/])(~|≤ )?(\d[\d,]*(?:\.\d+)?)"
                       r"( (?:GiB|GB|tok/s|s/step|tokens|chats|rows|W|%)|°C|%)?(?![\w.\-/])")


def inked(line: Line) -> Line:
    """Contrast in the type: a figure one tier up, its unit and ``~`` faint;
    measured and bounded read as facts; finished ticks recede."""
    out: Line = []
    for t, st in line:
        if st in ("tx", "hi", "dm") and _FIGURE.search(t):
            pos = 0
            for m in _FIGURE.finditer(t):
                if m.start() > pos:
                    out.append((t[pos:m.start()], st))
                lead, n, unit = m.groups()
                if lead:
                    out.append((lead, "rl" if st == "dm" else "dm"))
                out.append((n, "tx" if st == "dm" else "hi"))
                if unit:
                    out.append((unit, "rl"))
                pos = m.end()
            if pos < len(t):
                out.append((t[pos:], st))
            continue
        if t.strip() in ("tested", "measured") and st in ("dm", "rl"):
            st = "tx"
        elif t.startswith("✓ ") and st in ("tx", "dm"):
            st = "rl"
        out.append((t, st))
    return out


def titled(line: Line) -> Line:
    """A card's title line: the ▛▜ mark, in the accent, where the dot was."""
    if not line or line[0] != ("● ", "tx"):
        return line
    rest = list(line[1:])
    for i, (t, st) in enumerate(rest):  # the mark is a column wider: the aside's gap gives it
        if t.startswith("  ") and len(t) - len(t.lstrip(" ")) > 2:
            rest[i] = (t[1:], st)
            break
    return [(MARK_INLINE + " ", "a"), *rest]


def bar(frac: float, w: int, peak: Optional[float] = None, fill_style="a") -> Line:
    """Thin meter: ``━`` fill, ``─`` track, ``│`` peak marker."""
    frac = min(max(frac, 0.0), 1.0)
    n = round(frac * w)
    p = None if peak is None else min(round(min(max(peak, 0.0), 1.0) * w), w - 1)
    segs = []
    for i in range(w):
        if i == p:
            segs.append(("│", "hi"))
        elif i < n:
            segs.append(("━", fill_style))
        else:
            segs.append(("─", "tk"))
    return merge(segs)


def flow_bar(frac: float, w: int) -> Line:
    """Progress as the hero's river: the spectrum spans the filled part, so the
    leading edge is always the gold core."""
    frac = min(max(frac, 0.0), 1.0)
    n = round(frac * w)
    return [("━", ("sp", i / max(n - 1, 1))) for i in range(n)] + (
        [("─" * (w - n), "tk")] if w > n else []
    )


_SPARK = "▁▂▃▄▅▆▇█"


def spark(values: Sequence[float], w: int, height: int = 1, log: bool = True) -> List[Line]:
    """Block sparkline in the flow spectrum, ``height`` rows, latest point gold.

    ``log`` plots log(values): loss decays roughly exponentially, and a linear
    plot is a cliff followed by a flat line.
    """
    n = len(values)
    if n == 0 or w <= 0:
        return [[(" " * max(w, 0), None)] for _ in range(height)]
    jumps = spikes(values)
    cols, hot = [], []
    for i in range(w):
        a = int(i * n / w)
        b = max(int((i + 1) * n / w), a + 1)
        chunk = values[a : min(b, n)] or values[-1:]
        # a column is the mean of its steps -- unless one of them spiked:
        # then its peak, in red, so averaging never smooths a spike away
        hit = [values[j] for j in range(a, min(b, n)) if j in jumps]
        v = max(hit) if hit else sum(chunk) / len(chunk)
        hot.append(bool(hit))
        cols.append(math.log(v) if log and v > 0 else v)
    lo, hi = min(cols), max(cols)
    rows = []
    for r in range(height - 1, -1, -1):
        row = []
        for x, v in enumerate(cols):
            lvl = 1 + round((v - lo) / ((hi - lo) or 1) * (height * 8 - 1))
            fill = lvl - r * 8
            ch = " " if fill <= 0 else ("█" if fill >= 8 else _SPARK[fill - 1])
            row.append((ch, "e" if hot[x] else ("sp", x / max(w - 1, 1))))
        rows.append(merge(row))
    return rows


def row_means(losses: Sequence[float]) -> List[Tuple[int, float]]:
    """A step of one row swings with the row, so a curve of them shows the hard
    rows, not the run: (steps in, the mean of about eight rows). Under four such
    points nothing, since that few rows can't say which way a run went."""
    n, parts = len(losses), round(len(losses) / 8)
    if parts < 4:
        return []
    cuts = [n * i // parts for i in range(parts + 1)]
    return [(end, math.fsum(losses[at:end]) / (end - at)) for at, end in zip(cuts, cuts[1:])]


def spikes(values: Sequence[float], window: int = 5) -> set:
    """Steps whose loss jumps well above the run's own local level: more than
    six typical deviations over the median of the steps around it (and at
    least 5% of that level, so a flat, quiet run never flags its noise)."""
    n = len(values)
    if n < 2 * window + 1:
        return set()
    resid = []
    for i in range(n):
        near = sorted(values[max(0, i - window) : i + window + 1])
        resid.append(values[i] - near[len(near) // 2])
    typical = sorted(abs(r) for r in resid)[n // 2] or 1e-9
    # only steps with a full window on both sides, and only a rise: the steep
    # first steps of a run sit above their median without being a spike
    return {
        i for i in range(window, n - window)
        if resid[i] > 6 * typical and resid[i] > 0.05 * abs(values[i] - resid[i])
        and values[i] > values[i - 1]
    }


def activity(
    label: str,
    elapsed: str,
    t: int,
    *,
    frac: Optional[float] = None,
    eta: Optional[str] = None,
    w: int = CAP,
    animate: bool = True,
    loader: Optional[str] = None,
) -> Line:
    """``    Training  1h 07m 12s  ━━━━━──────  41%  eta 1h 36m`` -- the one
    live line. A wait with no bar shows a loader (poros.cli.ui.loaders) in a
    fixed 3-cell slot; a line with a bar leaves the slot empty, since the bar
    already moves (and the label then sits in the rail's column). The label
    holds still: only the loader, the clock and the bar move. The clock is a
    fixed-width field, so nothing after it ever moves."""
    from poros.cli.ui import loaders

    kind = loader or loaders.for_label(label)
    line = ln(
        S(" " * loaders.WIDTH, None) if frac is not None else loaders.frame(kind, t, animate),
        S(" ", None),
        S(label, "hi"),
        S(" " + elapsed.ljust(10), "tx"),
    )
    tail = ln(
        S(f"  {round(frac * 100):>3}%", "tx") if frac is not None else None,
        S(f"  eta {eta}", "dm") if eta else None,
    )
    if frac is not None:
        pw = min(w - width(line) - width(tail) - 2, 40)
        if pw >= 10:
            line += [("  ", None)] + flow_bar(frac, pw)
    return line + tail


# --------------------------------------------------------------------------
# < 60 cols
# --------------------------------------------------------------------------


def plain_pairs(pairs: Sequence[tuple], w: int) -> List[Line]:
    """``key: value`` lines; a value that doesn't fit drops to its own line."""
    import textwrap

    out: List[Line] = []
    for k, v, st in pairs:
        head = f"{k}: " if k else ""
        if len(head) + len(v) <= w:
            out.append(ln(S(head, "dm") if head else None, S(v, st)))
            continue
        if head:
            out.append(ln(S(head.rstrip(), "dm")))
        # words wrap; a path stays whole (it must stay copyable)
        chunks = [v] if v.startswith(("./", "/", "~")) else textwrap.wrap(v, w) or [v]
        out += [ln(S(x, st)) for x in chunks]
    return out
