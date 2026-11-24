"""A training run as an agent shows it in a chat: plain monospace text, no
color, narrow enough for a phone (about 50 columns). The card is two rows
for every update; the frame is the run card the live screen draws, for when
the user asks to see the run: its curve, axis, numbers and verdict, the line
unbroken in box-drawing characters, as plain text draws one. Both are drawn
from the same TrainState as the live screen."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from poros.cli.ui import components as c
from poros.cli.ui import run_card as rc
from poros.cli.ui.curve import Series
from poros.cli.ui.screens import TrainState
from poros.cli.ui.text import plain

MARK = "▛▜"
SPARK = "▁▂▃▄▅▆▇█"
CURVE_W = 40
NAME_MAX = 24
ENDED = {"succeeded": "trained", "cancelled": "stopped", "failed": "failed"}
_BEFORE = {"failed": "failed before its first step", "cancelled": "stopped before its first step",
           "queued": "queued", "succeeded": "no step recorded"}


def _smooth(xs: List[float]) -> List[float]:
    k = max(1, len(xs) // 20)
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1])
            for i in range(len(xs))]


def _buckets(xs: List[float], n: int) -> List[float]:
    """``xs`` as ``n`` means; fewer when there are fewer points."""
    n = min(n, len(xs))
    out = []
    for i in range(n):
        a = i * len(xs) // n
        b = max(a + 1, (i + 1) * len(xs) // n)
        out.append(sum(xs[a:b]) / (b - a))
    return out


def _bar(frac: float, n: int) -> str:
    k = round(max(0.0, min(1.0, frac)) * n)
    return "━" * k + "─" * (n - k)


def _name(model: str) -> str:
    """The model as the run names it, whole: Qwen2.5-0.5B-Instruct is not Qwen2.5-0.5B."""
    name = model.rsplit("/", 1)[-1]
    return name if len(name) <= NAME_MAX else name[: NAME_MAX - 1] + "…"


def _where(st: TrainState, state: str, eta: Optional[float], ended: bool = True) -> str:
    """``78/120  eta 23s`` while it trains, ``36/200  stopped`` once over
    (the frame's title says so already: ``ended=False``)."""
    said = f"{st.step}/{st.total}" if st.total else f"step {st.step}"
    if state in ENDED:
        return f"{said}  {ENDED[state]}" if ended else said
    return f"{said}  eta {c.duration(eta)}" if eta else said


def line(st: TrainState) -> Series:
    """The run card's line through the steps: a running mean, the number the screen shows."""
    return Series(max(st.total, len(st.losses))).extend(st.losses)


def _vram(st: TrainState, card_gib: Optional[float]) -> str:
    """The peak, as a receipt reads it: a step's end holds less, its streamed blocks gone."""
    if not st.vram_peak_gb:
        return ""
    peak = c.ceil1(c.gib(st.vram_peak_gb))
    return f"peak {peak:.1f}/{card_gib:.1f} GiB" if card_gib else f"peak {peak:.1f} GiB"


def card(st: TrainState, state: str, eta: Optional[float],
         card_gib: Optional[float]) -> List[str]:
    """Two rows: where the run is, and how its loss is going.

        ▛▜ Qwen2.5-0.5B-Instruct  ━━━━━━━━━━──────  78/120  eta 23s
           loss 2.83 █▇▅▄▄▃▂▂▂▁▁▁ 1.26 · peak 5.7/24.0 GiB

    The first loss is the first step's; the last, where the run card's line ends.
    """
    frac = st.step / st.total if st.total else 0.0
    top = f"{MARK} {_name(st.model)}  {_bar(frac, 16)}  {_where(st, state, eta)}"
    if not st.losses:  # no step yet: why, from its state
        return [top, "   " + _BEFORE.get(state, "starting")]
    ys = _buckets(_smooth(st.losses), 12)
    hi, lo = max(ys), min(ys)
    spark = "".join(SPARK[round((y - lo) / (hi - lo) * 7)] if hi > lo else SPARK[0] for y in ys)
    loss = f"   loss {st.losses[0]:.2f} {spark} {line(st).line[-1]:.2f}"
    vram = _vram(st, card_gib)
    return [top, loss + (f" · {vram}" if vram else "")]


def _columns(series: Series, cols: int, last: int, offset: int) -> List[Optional[float]]:
    """The line's value at each column, as the run card's cells sample it; None past the run."""
    n, total = len(series.line), series.total
    out: List[Optional[float]] = []
    for col in range(cols):
        if total < cols:  # fewer steps than columns: the line runs between them
            pos = col / max(cols - 1, 1) * (total - 1)
            if pos > last + 1e-9 or pos - offset > n - 1 + 1e-9:
                out.append(None)
                continue
            i = max(0.0, min(n - 1.0, pos - offset))
            j = min(int(i), max(n - 2, 0))
            out.append(series.line[j] + (series.line[min(j + 1, n - 1)] - series.line[j]) * (i - j))
        else:
            a = int(col * total / cols)
            b = max(a + 1, int((col + 1) * total / cols))
            out.append(series.line[max(0, min(n - 1, min(b - 1, last) - offset))]
                       if a <= last and a - offset < n else None)
    return out


def _drawn(series: Series, scale: Any, cols: int, last: int, offset: int,
           evals: List[Tuple[int, float]]) -> List[str]:
    """The line in box-drawing characters on the card's rows, unbroken where it climbs or falls,
    and each held-out score a ○ where it sits."""
    ys = _columns(series, cols, last, offset)
    rows = [None if y is None else max(0, min(scale.rows - 1, round(scale.row(y)))) for y in ys]
    grid = [[" "] * cols for _ in range(scale.rows)]
    for x, r in enumerate(rows):
        if r is None:
            continue
        p = rows[x - 1] if x and rows[x - 1] is not None else r
        if r == p:
            grid[r][x] = "─"
            continue
        grid[p][x] = "╮" if r > p else "╯"  # leaving the row it was on
        grid[r][x] = "╰" if r > p else "╭"  # arriving on its new one
        for k in range(min(p, r) + 1, max(p, r)):
            grid[k][x] = "│"
    total = series.total
    for step, v in evals:
        x = (round(step / max(total - 1, 1) * (cols - 1)) if total < cols
             else min(cols - 1, int(step * cols / total)))
        grid[max(0, min(scale.rows - 1, round(scale.row(v))))][x] = "○"
    return ["".join(row) for row in grid]


def _trained(st: TrainState) -> float:
    """The training time: the steps' own seconds, without a stop's pause or a queue's wait."""
    return sum(st.step_times)


def tenths(losses: List[float]) -> Tuple[float, float, int]:
    """(the mean of the first tenth of ``losses``, of the last tenth, how many steps each): at
    least one step each, and no step in both."""
    k = max(1, len(losses) // 10)
    return sum(losses[:k]) / k, sum(losses[-k:]) / k, k


def _fell(a: float, b: float) -> str:
    return f"{'↓' if b <= a else '↑'} {abs(b - a) / a * 100:.0f}%" if a else ""


def frame(st: TrainState, state: str, eta: Optional[float], card_gib: Optional[float],
          learned: str = "", rough: bool = False) -> List[str]:
    """The run card in plain text: where the run is, its curve with the held-out marks
    (○) and the steps under it, then its numbers. ``learned`` says the held-out result in
    words, where the card says it in numbers; on ``rough`` (few held-out rows) the title gives
    no verdict, as those words say why.

        ▛▜ Trained  Learned well · 120 steps · 4m 44s
           Qwen2.5-7B-Instruct
           held-out loss 2.620 to 1.910 on 40 rows, 27% lower

                ○

              2                                        ○
                ─╮
                 ╰───╮
              1      ╰───────╮╭╮  ╭╮
                             ╰╯╰──╯╰────────────────────

              0
                1                  60                120

           train loss 1.6012 to 0.5634
           means of its first and last tenth: 12 steps each
           2.37 s/step · 169 tok/s · peak 1.0/8 GiB
    """
    done = state in ENDED
    name = f"   {_name(st.model)}"
    if len(st.losses) < 2:  # nothing to draw yet: the card's words alone
        return [f"{MARK} {ENDED.get(state, 'training').capitalize()}", name,
                "   " + _where(st, state, eta)]
    series = line(st)
    evals = [(min(step, series.total - 1), v) for step, v in st.evals]
    k = rc.Card(series=series, last=max(0, min(st.step, series.total) - 1), rows=rc.TALL,
                offset=max(0, st.step - len(st.losses)), evals=evals,
                spikes=sorted(c.spikes(st.losses)), first=st.losses[0])
    scale = k.scale()
    lines = _drawn(series, scale, CURVE_W, k.last, k.offset, evals)
    labels = scale.labels()
    word = rc.with_ends(rc.verdict(evals[0][1], evals[-1][1]) if len(evals) >= 2 else "",
                        st.ends) if done and not rough else ""
    if done:
        head = " · ".join([f"{ENDED[state].capitalize()}{'  ' + word if word else ''}",
                           c.steps(st.step),
                           *([c.duration(_trained(st))] if st.step_times else [])])
    else:
        pct = f" · {st.step / st.total:.0%}" if st.total else ""
        head = (f"Step {c.num(st.step)} of {c.num(st.total)}{pct}" if st.total
                else f"Step {c.num(st.step)}") + (f" · eta {c.duration(eta)}" if eta else "")
    out = [f"{MARK} {head}", name, *([f"   {learned}"] if learned else []), ""]
    rows = [f"   {labels.get(i, ''):>4} {lines[i]}".rstrip() for i in range(scale.rows)]
    while rows and not rows[0]:  # the card's air above and below the line, which a chat lacks
        rows.pop(0)
    while rows and not rows[-1]:
        rows.pop()
    out += rows
    axis = plain(rc._axis(k, CURVE_W)).rstrip()
    out.append("        " + axis)
    # like for like: the mean of the run's first tenth of steps against its last tenth's. A step
    # is one row's loss, and the line's end is smoothed: neither is the other's measure. The
    # verdict is left to the held-out rows
    first, last, k = tenths(st.losses)
    out += ["", f"   train loss {first:.4f} to {last:.4f}",
            f"   means of its first and last tenth: {c.num(k)} steps each" if k > 1
            else "   its first step and its last"]
    if "▲" in axis:  # the axis says where, in steps; this says what
        out.append("   ▲ a loss spike at that step")
        more = re.search(r"▲ [\d,]+ \+(\d+)", axis)
        if more:
            out.append(f"   +{more[1]}: {more[1]} more spike{'s' * (more[1] != '1')} beside it")
    if evals and not learned:  # said in words at the top, once the run is over
        step, v = evals[-1]
        out.append(f"   ○ held out {v:.4f}  " + (f"{_fell(evals[0][1], v)}  from {evals[0][1]:.4f}"
                                                if len(evals) > 1 else f"at step {c.num(step)}"))
    numbers = [f"{st.step_time:.2f} s/step" if st.step_time else "",
               f"{st.tokens_per_sec:,.0f} tok/s" if st.tokens_per_sec else "",
               _vram(st, card_gib),
               c.duration(_trained(st)) if st.step_times and not done else ""]
    if any(numbers):
        out.append("   " + " · ".join(n for n in numbers if n))
    return out
