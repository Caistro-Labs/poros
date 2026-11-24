"""The run's card: the curve with its numbers beside it, live and finished.

One layout from the load to the receipt. While the run goes it is open:
a header row and a rail. When it ends the frame traces around it while the
glint runs the curve; both arrive together at ``curve.LAND``, the end of the
line turns accent, and the numbers become the run's result. ``t`` is the
milliseconds since the last step, None while live, ``FINAL`` at rest; every
frame is a pure function of it.

Before a number exists its place reads "–": the loss and memory before the first step, the
steps before the run knows how many it takes, a held-out score before the base is scored.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from poros.cli.ui import components as c
from poros.cli.ui.curve import LAND, Scale, Series, block_rows, ease, fit_scale, seq, x_of
from poros.cli.ui.text import Line, S, merge, width

FINAL = 1e9
MIN_W = 76  # below this the curve is too narrow to read: the old rows
SIDE = 24  # the readout's column
TALL, SHORT = 10, 7
MEMORY = "memory" if sys.platform == "darwin" else "vram"  # a Mac has one pool, not a card's
NONE = "–"  # a value not known yet


def fade(line: Sequence[Tuple[str, Any]], a: float) -> Line:
    """A row on its way in or out; fully in, it is the row itself."""
    if a >= 1:
        return list(line)
    return [(t, ("fd", st, max(0.0, a))) if st not in (None, "raw") else (t, st) for t, st in line]


def verdict(start: Optional[float], end: Optional[float]) -> str:
    """What the held-out rows say. Only they can: a falling training loss is
    not learning, so with none there is no word at all."""
    if not start or end is None:
        return ""
    gain = (start - end) / start
    return "Learned well" if gain >= 0.05 else "Barely moved" if gain >= 0 else "Held-out loss rose"


def with_ends(word: str, ends: Optional[Tuple[int, int]]) -> str:
    """The verdict and, on chats, whether the answers end: "Learned well · ends its answers 8 of
    8". A loss can fall while a model never stops, so the card says both."""
    if not ends or not ends[1]:
        return word
    k, n = ends
    return f"{word} · ends its answers {k} of {n}" if word else f"Ends its answers {k} of {n}"


@dataclass(frozen=True)
class Score:
    """One pass over the held-out rows: the base before step 1, then each checkpoint. ``k`` of
    ``n`` is what the rows show beside the loss: on chats the answers that end, on pairs the
    ones that prefer the chosen answer."""

    step: int
    loss: float
    k: Optional[int] = None
    n: Optional[int] = None
    word: str = ""  # "ends" or "prefers"


def score(step: int, data: Any) -> Optional[Score]:
    """A score from an eval's numbers, as the trainer reports them and events.jsonl keeps them."""
    if not isinstance(data, dict):
        return None
    loss = data.get("eval_loss")
    if type(loss) not in (int, float) or loss != loss:
        return None
    k, n, word = data.get("ends"), data.get("answer_rows"), "ends"
    if type(data.get("prefers")) is int:
        k, n, word = data["prefers"], data.get("rows"), "prefers"
    whole = type(k) is int and type(n) is int and 0 <= k <= n
    return Score(int(step or 0), float(loss), k if whole else None, n if whole else None,
                 word if whole else "")


def curve_rows(room: int) -> int:
    """How tall the curve can be in ``room`` rows; 0 for no room at all."""
    return TALL if room >= 22 else SHORT if room >= 18 else 0


@dataclass
class Card:
    series: Series
    last: int  # the last step shown, 0-based, in the run's own count
    rows: int = TALL
    offset: int = 0  # a resumed run's steps before its first loss
    evals: List[Tuple[int, float]] = field(default_factory=list)
    spikes: Sequence[int] = ()
    first: Optional[float] = None
    final: Optional[float] = None  # the result the loss lands on
    vram: float = 0.0  # now, then what it rolls to when the run ends
    vram_peak: float = 0.0
    ram: Optional[float] = None
    ram_peak: Optional[float] = None
    meta: str = ""  # the top edge's right end: steps · recipe · time
    word: str = ""  # the top edge's verdict
    pct: int = 0
    painter: Any = None
    framed: bool = True  # False: curve and axis alone, inside a page's own rail
    scores: Sequence[Score] = ()  # each held-out pass: beside the curve, a row a checkpoint
    memory: bool = False  # the memory row stands before its first number
    measured: bool = True  # a step has reported memory: a 0 there is a number, not "–"
    scored: bool = False  # the base is scored before step 1: its table stands from the start
    counted: bool = True  # False: the run's steps are not known yet
    anchored: bool = False  # the memory rows on the readout's last rows (the live card's)

    def scale(self) -> Optional[Scale]:
        """None before the first step: the chart stands empty, without labels."""
        if not self.series.line:
            return None
        vals = [self.series.lo, self.series.hi] + [v for _, v in self.evals]
        return fit_scale(vals, self.rows)


def _cells(line: Sequence[Tuple[str, Any]], n: int) -> List[Tuple[str, Any]]:
    out = [(ch, st) for t, st in line if st != "raw" for ch in t][:n]
    return out + [(" ", None)] * (n - len(out))


def _pad(line: Line, n: int) -> Line:
    return [*line, (" " * max(0, n - width(line)), None)]


def _axis(k: Card, cw: int) -> Line:
    total = k.series.total
    cells: List[Tuple[str, Any]] = [(" ", None)] * cw

    def put(text: str, col: int, st: Any) -> bool:
        if col < 0 or col + len(text) > cw:
            return False
        for i, ch in enumerate(text):
            cells[col + i] = (ch, st)
        return True

    def col_of(step: int) -> int:
        return round(x_of(step, total, cw) - 0.5)

    end = (c.num(total) if total > 1 else "") if k.counted else NONE  # one step: its "1" alone
    jumps = [s + k.offset for s in k.spikes if s + k.offset <= k.last]
    taken: List[Tuple[int, int]] = []
    shown = 0
    for s in jumps:  # the spikes first: the axis says where, in steps
        text, at = f"▲ {c.num(s + 1)}", col_of(s) - 1
        if not any(at <= b + 1 and at + len(text) >= a - 1 for a, b in taken) and (
                at >= 2 and at + len(text) < cw - len(end) - 1):
            put(text, at, "e")
            taken.append((at, at + len(text)))
            shown += 1
    if shown < len(jumps) and taken:
        more = f"+{len(jumps) - shown}"
        put(more, taken[-1][1] + 1, "e")
        taken[-1] = (taken[-1][0], taken[-1][1] + 1 + len(more))
    put("1", 0, "rl")
    mid = c.num(total // 2)
    at = col_of(total // 2) - len(mid) // 2
    # a run of two or three steps has no middle of its own: its half is its first step
    if k.counted and 1 < total // 2 < total and not any(at <= b + 1 and at + len(mid) >= a - 1
                                           for a, b in taken):
        put(mid, at, "rl")
    if end:
        put(end, cw - len(end), "rl")
    return merge(cells)


def _readout(k: Card, t: Optional[float], done: bool) -> List[Line]:
    """The numbers beside the curve, the same rows from the first step on."""
    rise = ease(seq(t, 0, LAND))
    landed = seq(t, LAND, LAND + 160)
    now = k.series.line[-1] if k.series.line else None
    end = k.final if k.final is not None else now
    # no point drawn yet: the result itself where the run has one, else "–"
    loss = end if now is None else now + (end - now) * rise
    first = k.first if k.first is not None else k.series.raw[0] if k.series.raw else None

    def label(live: str, end: str) -> Line:
        return fade([S(end, "dm")], seq(t, LAND, LAND + 140)) if done else [S(live, "dm")]

    # the number stays in its cell when the run ends: what it now means comes after it
    peak: Line = fade([S(" peak", "dm")], seq(t, LAND, LAND + 140)) if done else []

    def fell(a: Optional[float], b: float) -> str:
        return f"  {'↓' if b <= a else '↑'} {abs(b - a) / a * 100:.0f}%" if a else ""

    ink: Any = ("ax", landed) if 0 < landed < 1 else ("a" if landed else "tx")
    change = fell(first, loss) if loss is not None and len(k.series.raw) > 1 else ""  # of one: none
    out: List[Line] = [label("loss", "final loss"),
                       [S(NONE, "tx")] if loss is None else [S(f"{loss:.4f}", ink), S(change)]]
    memory = bool(k.vram or k.vram_peak or k.memory)
    below = memory + (k.ram is not None or k.ram_peak is not None)
    room = k.rows - len(out) - 1 - below  # under a blank row, over the memory rows
    if k.scores and room >= 2:
        table = ledger(k.scores, room)
    else:  # the base's row stands before its score, so nothing moves when it comes
        table = waiting_ledger() if k.scored and room >= 2 else []
    if table:  # the checkpoints say where the loss came from: no "from" row
        out += [[], *table]
        out += [[]] * (k.rows - below - len(out) > 0)
    elif k.rows == TALL:
        out += [[S(f"from {first:.4f}" if first is not None else f"from {NONE}", "dm")], []]
    if not table and k.evals:
        step, val = k.evals[-1]
        out += [[S("○ ", "tx"), S("held out", "dm")],
                [S(f"{val:.4f}"), S(fell(k.evals[0][1], val) if len(k.evals) > 1 else "")]]
        if k.rows == TALL:
            sub = (f"from {k.evals[0][1]:.4f}" if done else f"step {c.num(step)}")
            out += [[S(sub, "dm")], []]
        else:
            out += [[]]

    def amount(name: str, a: float, b: float) -> Line:
        if not a and not b and not k.measured:  # no step has measured it yet
            return [S(name, "dm"), S(" ", None), S(NONE, "tx")]
        v = c.gib(a + (b - a) * rise)
        v = c.ceil1(v) if peak else v  # landed: the peak, never shown below it
        return [S(name, "dm"), S(" ", None), S(f"{v:.1f}", "tx"), S(" GiB", "rl"), *peak]

    # the memory rows sit on the readout's last rows: the table above them grows as
    # checkpoints are scored, and they stay where they are
    out += [[]] * (max(0, k.rows - below - len(out)) if k.anchored else 0)
    if memory:
        out.append(amount(MEMORY, k.vram, k.vram_peak or k.vram))
    if k.ram is not None or k.ram_peak is not None:
        ram = k.ram if k.ram is not None else k.ram_peak
        top = k.ram_peak if k.ram_peak is not None else ram
        out.append(amount("ram ", ram, top))
    return out


def waiting_ledger() -> List[Line]:
    """The checkpoints' table before the base is scored: its header and the base's row, the
    score a dash, in the columns ``ledger`` gives them."""
    return [[S("  " + "step".rjust(4) + "  " + "held out", "rl")],
            [S("○ ", "dm"), S("base", "dm"), S("  " + NONE.rjust(8), "tx")]]


def ledger(scores: Sequence[Score], room: int, wide: int = SIDE + 4) -> List[Line]:
    """The checkpoints beside the curve: a header, then a row each, the base first.

          step  held out   ends
        ○ base    2.9100  12/62
        ○  500    1.7007  41/62
        ● 1,000   1.2630  58/62    ● the lowest held-out loss, in the accent

    ○ is the curve's own mark for a held-out score. With more than fit, the base, the best and
    the latest stay, and the rows between give way from the oldest."""
    rows = list(scores)
    best = min((s for s in rows if s.step), key=lambda s: s.loss, default=None)
    keep = room - 1
    if len(rows) > keep:
        must = {id(rows[0]), id(rows[-1])} | ({id(best)} if best is not None else set())
        rest = [s for s in rows if id(s) not in must]
        drop = {id(s) for s in rest[:len(rows) - keep]}
        rows = [s for s in rows if id(s) not in drop][-keep:] if keep > 0 else []
    if not rows:
        return []

    def step_of(s: Score) -> str:
        return "base" if s is scores[0] and s.step == 0 else c.num(s.step)

    word = next((s.word for s in rows if s.word), "")
    counts = [f"{s.k}/{s.n}" if s.n is not None else "" for s in rows]
    sw = max(4, *(len(step_of(s)) for s in rows))
    cw = max([len(word)] + [len(x) for x in counts]) if word else 0
    if 2 + sw + 2 + 8 + (2 + cw if cw else 0) > wide:
        cw, word = 0, ""  # the loss outlasts the count
    head: Line = [S("  " + "step".rjust(sw) + "  " + "held out", "rl")]
    if word:
        head.append(S("  " + word.rjust(cw), "rl"))
    out: List[Line] = [head]
    for s, n in zip(rows, counts):
        top = best is not None and s is best
        line: Line = [S("● " if top else "○ ", "a" if top else "dm"),
                      S(step_of(s).rjust(sw), "dm"),
                      S("  " + f"{s.loss:.4f}".rjust(8), "a" if top else "tx")]
        if word:
            line.append(S("  " + n.rjust(cw), "tx" if n else "dm"))
        out.append(line)
    return out


def card(k: Card, wc: int, t: Optional[float] = None) -> List[Line]:
    """Header, curve and readout, axis, bottom edge. ``t`` as in the module."""
    right = wc - 3  # the frame's last column
    done = t is not None and t >= LAND
    side = _readout(k, t, done)
    scale = k.scale()
    labels = scale.labels() if scale is not None else {}
    # the loss axis's labels sit in 4 columns; a longer one (0.0002) takes its own from the
    # curve, so every row stays as wide as the frame
    gutter = max([4, *(len(v) for v in labels.values())])
    # the readout's column, wider by what the checkpoints' table needs: the curve gives it
    reserve = SIDE
    if k.anchored and k.counted and (k.scored or k.scores):  # the table at its widest step,
        # with counts up to "nn/nn", held from the first frame: the curve never narrows for it
        reserve = max(SIDE, 2 + max(4, len(c.num(k.series.total))) + 2 + 8 + 2 + 5 + 1)
    cw = min(62, wc - 9 - (gutter - 4) - 2 - max(reserve, *(width(r) + 1 for r in side)) - 3)
    rows_n = k.rows
    span = right - 2
    total_edge = span + (rows_n + 1) + span
    kk = ease(seq(t, 0, LAND)) * total_edge
    head_to = 2 + kk

    def closed(r: int) -> bool:  # the right edge, reached row by row from the top
        return k.framed and t is not None and kk >= span and r <= kk - span

    title: Line = [S("Trained", "hi")]
    if k.word:
        title += [S("  ", None), *fade([S(k.word, "tx")], seq(t, LAND + 40, LAND + 260))]
    meta = f" {k.meta} " if k.meta else " "
    shown = k.last >= 0  # a step is done, or a resumed run's count is known
    live_head = c.event("Step", f"{c.num(k.last + 1) if shown else NONE} of "
                        f"{c.num(k.series.total) if k.counted else NONE}",
                        f"{k.pct}%" if shown and k.counted else NONE)
    out: List[Line] = []
    if not k.framed:
        pass
    elif t is None:
        out.append(live_head)
    else:
        lead: Line = [S("  ╭─ ", "rl"), *title, S(" ", "rl")]
        fill = max(1, right - width(lead) - len(meta) - 1)
        top: Line = [*lead, S("─" * fill, "rl"), S(meta, "dm"), S("─╮", "rl")]
        a, b = _cells(top, right + 1), _cells(live_head, right + 1)
        out.append(merge([a[i] if 2 <= i <= head_to else ((" ", None) if i < 2 else b[i])
                          for i in range(right + 1)]))

    cells = None
    raw = None
    if scale is None or not k.series.line:  # nothing to plot yet: the chart's cells stand empty
        cells = [[(" ", None)] * cw for _ in range(rows_n)]
    elif k.painter is not None:
        raw = k.painter.frame(k.series, scale, cw, k.last, k.offset, k.evals, k.spikes, t)
    else:
        cells = block_rows(k.series, scale, cw, k.last, k.offset, k.evals, k.spikes, t)
    def row(r: int, body: Line) -> Line:
        line: Line = [S(c.RAIL, "rl"), *body]
        return [*_pad(line, right), S("│", "rl")] if closed(r) else line

    for i in range(rows_n):
        y = f"{labels[i]:>{gutter}} " if i in labels else " " * (gutter + 1)
        body: Line = [S(y, "rl")]
        if raw is not None and i == 0:
            body.append((raw, "raw"))
        body += merge(cells[i]) if cells is not None else [(" " * cw, None)]
        if i < len(side) and side[i]:
            body += [S("  ", None), *side[i]]
        out.append(row(i + 1, body))
    out.append(row(rows_n + 1, [S(" " * (gutter + 1), None), *_axis(k, cw)]))
    if not k.framed:
        return out
    edge: List[Tuple[str, Any]] = [(" ", None)] * (right + 1)
    edge[2] = ("│", "rl")
    if t is not None:
        if kk >= span + rows_n + 1:
            from_col = right - 1 - (kk - span - rows_n - 1)
            for i in range(3, right):
                if i >= from_col:
                    edge[i] = ("─", "rl")
            edge[right] = ("╯", "rl")
        if kk >= total_edge - 0.5:
            edge[2] = ("╰", "rl")
    out.append(merge(edge))
    return out
