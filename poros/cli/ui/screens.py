"""Screens: pure functions from run data to ``Line`` lists.

Each screen takes the terminal width and returns lines; callers render them
with a palette (``text.render``) or into a ``live.Live`` surface. Below 60
columns every screen switches to plain ``key: value`` lines.
"""

from __future__ import annotations

import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from poros.cli.ui import components as c
from poros.cli.ui import run_card as rc
from poros.cli.ui.curve import LAND, RING, Series, seq
from poros.cli.ui.text import Line, S, ln, width

PLAIN_BELOW = 60
MEMORY, PEAK = rc.MEMORY, "peak" if sys.platform == "darwin" else "peak vram"


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


@dataclass
class TrainState:
    """Everything the live training surface shows. Mutated by the reporter."""

    model: str
    arch: str = ""  # "dense · 80 blocks" / "MoE · 48 layers"
    gpu: str = ""
    card_gb: float = 0.0
    output_dir: str = ""
    run: str = ""
    step: int = 0
    total: int = 0
    loss: Optional[float] = None
    losses: List[float] = field(default_factory=list)
    step_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    tokens_seen: float = 0.0  # a watched run's token count so far, from its events
    vram_gb: float = 0.0
    vram_peak_gb: float = 0.0
    load_s: float = 0.0
    ram_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    stream_unit: str = ""
    stream_pos: Optional[int] = None
    stream_count: int = 0
    stream_gbs: Optional[float] = None
    started: float = 0.0  # monotonic, first step
    elapsed: float = 0.0
    label: str = "Training"
    last_checkpoint: Optional[str] = None
    recipe: str = ""  # SFT / DPO / a loop file's name
    extra: Dict[str, float] = field(default_factory=dict)  # the loss's numbers, latest step
    accs: List[float] = field(default_factory=list)  # a pair run: each step's accuracy
    evals: List[Tuple[int, float]] = field(default_factory=list)  # held-out loss, as it comes
    scores: List[rc.Score] = field(default_factory=list)  # each held-out pass, whole
    ends: Optional[Tuple[int, int]] = None  # chats, the last score: rows that end, of those read
    series: Optional[Series] = None  # the line through ``losses``
    painter: Any = None  # draws the curve as an image, where the terminal can show one
    jumps: Tuple[float, int, set] = (0.0, 0, set())  # spikes, found at most once a second
    watching: bool = False  # /runs: ctrl-c leaves the screen, the run goes on
    one_row: bool = False  # one row a step: the finished card reads it by row means
    # what the card holds a row for before its first number, so nothing moves when it comes:
    scored: bool = False  # the base is scored before step 1 (the checkpoints' table)
    pairs: bool = False  # a pair objective (its accuracy and margin rows)
    stream_soon: bool = False  # the stream's row, until the trainer says what streams

    def first(self, losses: Optional[List[float]] = None,
              total: Optional[int] = None) -> Optional[float]:
        """Where the card's loss starts. One row a step: the mean of the first rows the finished
        card will average (``c.row_means`` over the planned steps), once they are in, so the
        number does not change when the run ends. Before them, and otherwise, the first step.
        ``losses`` and ``total``: a frame's own reading of them (default: the state's)."""
        losses = self.losses if losses is None else losses
        total = self.total if total is None else total
        if not losses:
            return None
        parts = round(total / 8) if self.one_row and total else 0
        if parts >= 4 and len(losses) >= (end := total // parts):
            return math.fsum(losses[:end]) / end
        return losses[0]

    def eta(self) -> Optional[str]:
        if not self.step_times or not self.total or self.step >= self.total:
            return None
        recent = self.step_times[-20:]
        return c.duration(sum(recent) / len(recent) * (self.total - self.step))

    def loss_smooth(self) -> Optional[float]:
        if not self.losses:
            return None
        tail = self.losses[-5:]
        return sum(tail) / len(tail)


def train_surface(st: TrainState, t: int, w: int, *, animate: bool = True) -> List[Line]:
    """The redrawn region while training runs: the loss curve, the card's
    memory against its size, speed, and the base streaming block by block.
    Where the card fits it is the one layout from the load on; a number not
    known yet reads "–" in its place."""
    loading = not st.losses and not st.step and st.label != "Training"
    if w < PLAIN_BELOW and not loading:  # the load, this narrow: its one line, as before
        return _train_plain(st, w)
    tall_card = rc.curve_rows(_room()) if w >= rc.MIN_W else 0
    if tall_card:
        return _run_surface(st, t, w, tall_card, animate)
    tall = 3 if w >= 90 else 1
    wc = min(w, c.CAP)
    out: List[Line] = [[]]  # breathing room under the last scrollback event
    if st.step:
        out.append(
            c.event(
                "Step",
                f"{c.num(st.step)} of {c.num(st.total)}",
                f"{round(st.step / max(st.total, 1) * 100)}%",
            )
        )
        # the run page's grammar: the meter first, then its number
        lw = 8
        viz = min(wc - len(c.RAIL) - lw - 14, 56)
        loss = st.loss_smooth()
        loss_s = f"{loss:.4f}" if loss is not None else "–"
        if w >= 80 and len(st.losses) >= 2:
            first = st.first()
            side = [S(loss_s, "a"), S(f"from {first:.2f}", "dm"), None]
            for i, r in enumerate(c.spark(st.losses, viz, tall)):
                out.append(c.rail(S(("loss" if i == 0 else "").ljust(lw), "dm"), r,
                                  S("  ", None), side[i]))
        else:
            out.append(c.rail(S("loss".ljust(lw), "dm"), S(loss_s, "a")))
        n = 16
        out += metric_rows(st.extra, lw, n)
        if st.card_gb > 0:
            out.append(c.rail(S(MEMORY.ljust(lw), "dm"),
                              *c.chips(st.vram_gb / st.card_gb, n, "a"),
                              S(f"  {c.gb(st.vram_gb):>8}", "tx"),
                              S(f"  peak {c.ceil1(c.gib(st.vram_peak_gb)):.1f}"
                                f" of {c.gib(st.card_gb):.1f} GiB", "dm")))
        if st.ram_gb is not None and st.ram_total_gb:
            out.append(c.rail(S("ram".ljust(lw), "dm"),
                              *c.chips(st.ram_gb / st.ram_total_gb, n, "tx"),
                              S(f"  {c.gb(st.ram_gb):>8}", "tx"),
                              S(f"  of {c.gib(st.ram_total_gb):.0f} GiB", "dm")))
        tps = round(st.tokens_per_sec or 0)  # a watched run's file may not carry it
        out.append(c.rail(S("speed".ljust(lw), "dm"), S(f"{st.step_time:5.2f} s"),
                          S("/step", "dm"),
                          *([S(" · ", "dm"), S(c.num(tps)), S(" tok/s", "dm")] if tps else [])))
        if st.stream_unit and st.stream_count:
            pos = (f"{st.stream_unit} {st.stream_pos} of {st.stream_count}" if st.stream_pos
                   else f"{st.stream_count} {st.stream_unit}s")
            row = c.rail(S("stream".ljust(lw), "dm"), S(pos, "dm"))  # words, no moving bar
            if st.stream_gbs is not None:
                row += ln(S(" · ", "rl"), S(f"{c.gib(st.stream_gbs):.1f}"), S(" GiB/s", "dm"))
            out.append(row)
        out.append([])
    training = st.label == "Training"
    frac = st.step / st.total if st.total and training else None
    out.append(
        c.activity(
            st.label,
            c.duration(st.elapsed),
            t,
            frac=frac,
            eta=st.eta() if training else None,
            w=wc,
            animate=animate,
        )
    )
    return out + _status(st, w)


def _room() -> int:
    """Rows a run's screen can use: a session tab holds fewer than the window;
    alone it needs one to spare, or the redraw climbs past the top."""
    lines = shutil.get_terminal_size((100, 40)).lines
    return lines - 6 if c.in_session else lines - 2


def _series(st: TrainState, losses: Optional[List[float]] = None,
            total: Optional[int] = None) -> Series:
    losses = st.losses if losses is None else losses
    if st.series is None:
        st.series = Series((st.total if total is None else total) or len(losses))
    return st.series.extend(losses)


def _spikes(st: TrainState, losses: Optional[List[float]] = None) -> set:
    losses = st.losses if losses is None else losses
    at, n, found = st.jumps
    now = time.monotonic()
    if len(losses) != n and now - at >= 1.0:
        st.jumps = (now, len(losses), c.spikes(losses))
    return st.jumps[2]


def _clamped(evals: Sequence[Tuple[int, float]], total: int) -> List[Tuple[int, float]]:
    return [(min(step, max(total - 1, 0)), v) for step, v in evals]


def _run_surface(st: TrainState, t: int, w: int, rows: int, animate: bool) -> List[Line]:
    wc = min(w, c.CAP)
    # one reading of what the trainer's thread appends to, and everything drawn from it
    losses = list(st.losses)
    step = max(st.step, len(losses))
    total = st.total
    # before the first step the series waits for the step count: one made now would fix it
    series = _series(st, losses, total) if losses else Series(total)
    offset = max(0, step - len(losses))
    # the image draws a line: from its second point; one point is the cells' dot
    painter = st.painter if len(losses) >= 2 else None
    k = rc.Card(series=series, last=min(step, series.total) - 1, rows=rows,
                offset=offset, evals=_clamped(st.evals, series.total),
                spikes=sorted(_spikes(st, losses)), first=st.first(losses, total),
                vram=st.vram_gb, ram=st.ram_gb, pct=round(step / max(total, 1) * 100),
                painter=painter, scores=st.scores, memory=st.card_gb > 0,
                measured=bool(losses), scored=st.scored, counted=total > 0, anchored=True)
    lw = 11
    out: List[Line] = [[]] + rc.card(k, wc) + [c.rail()]
    curve_at = 2
    out += metric_rows({**_PAIRS_WAIT, **st.extra} if st.pairs else st.extra, lw, 16)
    out += _speed_row(st.tokens_per_sec, lw, st.step_time) or [
        c.rail(S("speed".ljust(lw), "dm"), S(rc.NONE, "tx"))]
    if st.stream_unit and st.stream_count:
        pos = (f"{st.stream_unit} {st.stream_pos} of {st.stream_count}" if st.stream_pos
               else f"{st.stream_count} {st.stream_unit}s")
        row = c.rail(S("stream".ljust(lw), "dm"), S(pos, "dm"))  # words, no moving bar
        if st.stream_gbs is not None:
            row += ln(S(" · ", "rl"), S(f"{c.gib(st.stream_gbs):.1f}"), S(" GiB/s", "dm"))
        out.append(row)
    elif st.stream_soon:
        out.append(c.rail(S("stream".ljust(lw), "dm"), S(rc.NONE, "dm")))
    out.append([])
    training = st.label == "Training"
    activity = c.activity(st.label, c.duration(st.elapsed), t,
                          frac=step / total if total and training else None,
                          eta=st.eta() if training else None, w=wc, animate=animate)
    if painter is not None and animate:  # the head breathes between steps
        halo = painter.halo(series, k.last, offset, time.monotonic() / 2.4 % 1,
                               len(out) - curve_at, 9)
        activity = [(halo, "raw"), *activity] if halo else activity
    out.append(activity)
    return out + _status(st, w)


def _status(st: TrainState, w: int) -> List[Line]:
    """One line, dropped from the right when it won't fit: the key outlasts the path,
    and a run's own folder already names it."""
    path = c.short_path(st.output_dir) if st.output_dir else ""
    named = st.run and path.endswith(st.run)
    return c.statusline(
        ["" if named or not st.run else f"run {st.run}", st.recipe, st.arch.split(" ·")[0],
         "ctrl-c stops watching" if st.watching else "ctrl-c to stop", path], w)


def _speed_row(tps: float, lw: int, step_time: float = 0.0) -> List[Line]:
    """Tokens a second, and while the run goes its seconds a step."""
    tail = [S(" · ", "dm"), S(f"{step_time:.2f}"), S(" s/step", "dm")] if step_time else []
    if not tps:
        return [c.rail(S("speed".ljust(lw), "dm"), *tail[1:])] if tail else []
    return [c.rail(S("speed".ljust(lw), "dm"), S(c.num(round(tps)), "tx"), S(" tok/s", "dm"),
                   *tail)]


_SHOWN = ("accuracy", "margin", "chosen_reward", "rejected_reward")
_PAIRS_WAIT: Dict[str, Optional[float]] = {"accuracy": None, "margin": None}  # before step 1


def metric_rows(extra: Dict[str, float], lw: int, n: int, over: int = 0) -> List[Line]:
    """The loss's own numbers under the loss row, meter first. DPO gets its
    accuracy meter and the margin split into its two rewards; a custom loss
    gets its first two numbers. Nothing for plain SFT. ``over``: the steps the accuracy is the
    mean of, once a run has ended (one step swings with its pairs: the run's page, its card
    and what an agent is told all say this window); 0 while it trains: the step just taken."""
    out: List[Line] = []
    if "accuracy" in extra and extra["accuracy"] is None:  # no step yet: the rows, waiting
        out.append(c.rail(S("acc".ljust(lw), "dm"), *c.chips(0.0, n, "tx"),
                          S(f"  {rc.NONE}", "tx")))
    elif "accuracy" in extra:
        acc = max(0.0, min(1.0, extra["accuracy"]))
        window = f" over its last {over} step{'' if over == 1 else 's'}" if over else ""
        out.append(c.rail(S("acc".ljust(lw), "dm"), *c.chips(acc, n, "tx"),
                          S(f"  {acc * 100:.0f}%", "tx"),
                          S(f"  chosen ranked first{window}", "dm")))
    if "margin" in extra and extra["margin"] is None:
        out.append(c.rail(S("margin".ljust(lw), "dm"), S(rc.NONE, "tx")))
    elif "margin" in extra:
        row = c.rail(S("margin".ljust(lw), "dm"), S(f"{extra['margin']:.3f}", "tx"))
        if "chosen_reward" in extra and "rejected_reward" in extra:
            row += ln(S(f"  chosen {extra['chosen_reward']:+.2f} · "
                        f"rejected {extra['rejected_reward']:+.2f}", "dm"))
        out.append(row)
    rest = [k for k in extra if k not in _SHOWN][:2]
    for k in rest:
        out.append(c.rail(S(k[:lw - 1].ljust(lw), "dm"), S(f"{extra[k]:.4g}", "tx")))
    return out


def _train_plain(st: TrainState, w: int) -> List[Line]:
    loss = st.loss_smooth()
    pairs = [
        ("poros", f"{st.label.lower()} {c.num(st.step)}/{c.num(st.total)}", "tx"),
        ("loss", f"{loss:.4f}" if loss is not None else "–", "a"),
    ]
    if "accuracy" in st.extra:
        pairs.append(("acc", f"{st.extra['accuracy'] * 100:.0f}%", "tx"))
    if st.card_gb:
        pairs.append(
            (
                MEMORY,
                f"{c.gb(st.vram_gb)} (peak {c.ceil1(c.gib(st.vram_peak_gb)):.1f}"
                f" of {c.gib(st.card_gb):.1f} GiB)",
                "tx",
            )
        )
    if st.ram_gb is not None:
        pairs.append(("ram", c.gb(st.ram_gb), "tx"))
    pairs.append(
        ("speed", f"{st.step_time:.2f} s/step, {c.num(round(st.tokens_per_sec))} tok/s", "tx")
    )
    if st.eta():
        pairs.append(("eta", st.eta() or "", "tx"))
    return c.plain_pairs(pairs, w)


def heartbeat(st: TrainState) -> str:
    """One plain line for logs (no TTY): every field a CI log needs."""
    loss = st.loss_smooth()
    parts = [
        f"step {st.step}/{st.total}",
        f"loss {loss:.4f}" if loss is not None else "",
        f"{st.step_time:.2f} s/step",
        (f"{MEMORY} {c.gb(st.vram_gb)} (peak {c.ceil1(c.gib(st.vram_peak_gb)):.1f})"
         if st.card_gb else ""),
        f"ram {c.gb(st.ram_gb)}" if st.ram_gb is not None else "",
        f"eta {st.eta()}" if st.eta() else "",
    ]
    return "  ".join(p for p in parts if p)


@dataclass
class TrainResult:
    steps: int
    wall_s: float
    final_loss: float
    first_loss: Optional[float]
    losses: List[float]
    peak_vram_gb: float
    card_gb: float
    peak_ram_gb: Optional[float]
    ram_total_gb: Optional[float]
    output_dir: str
    receipt: Optional[str]
    tokens_per_sec: float = 0.0
    next_cmd: str = ""
    save_cmd: str = ""  # keeps this run as a recipe
    next_note: str = ""  # said under the next step: why it is not the usual one
    extra: Dict[str, float] = field(default_factory=dict)  # the loss's numbers, last step
    acc_over: int = 0  # a pair run: the steps its accuracy is the mean of (its last ones)
    recipe: str = ""
    eval_start: Optional[float] = None  # held-out loss before the first step
    eval_end: Optional[float] = None  # and after the last
    eval_note: str = ""  # why there is no start, where a resume dropped it
    every: int = 1  # the steps one point of the curve stands for
    evals: List[Tuple[int, float]] = field(default_factory=list)  # every held-out loss, by step
    scores: List[rc.Score] = field(default_factory=list)  # and each pass whole, for the ledger
    ends: Optional[Tuple[int, int]] = None  # chats, the last score: rows that end, of those read
    series: Optional[Series] = None  # the line the live card drew, kept so nothing moves
    painter: Any = None  # the image the live card drew on
    offset: int = 0  # a resumed run's steps before its first loss
    vram_now: float = 0.0  # where the finish's numbers roll from
    ram_now: Optional[float] = None
    room: Optional[int] = None  # rows it has to fit in; None: the window's


def _done_card(r: TrainResult, wc: int, t: float, painter: Any) -> List[Line]:
    series = r.series or Series(len(r.losses)).extend(r.losses)
    every = r.every if r.series is not None else 1  # the live line has a point a step
    per = 1 if r.series is not None else max(1, r.every)  # else a point is ``every`` steps
    evals = _clamped([(step // per, v) for step, v in (
        r.evals or ([(0, r.eval_start), (r.steps, r.eval_end)]
                    if r.eval_start is not None and r.eval_end is not None else []))],
                     series.total)
    recipe = f" · {r.recipe}" if r.recipe else ""
    meta = f"{c.steps(r.steps)}{recipe} · {c.duration(r.wall_s)}"
    # the run's own start and end where it knows them; a resumed run's first score here was taken
    # on the way, so with no start carried there is no verdict to give
    if r.eval_start is not None and r.eval_end is not None:
        said = rc.verdict(r.eval_start, r.eval_end)
    else:
        said = rc.verdict(evals[0][1], evals[-1][1]) if len(evals) >= 2 and not r.offset else ""
    word = rc.with_ends(said, r.ends)
    if wc - 5 - 8 - (len(word) + 2 if word else 0) - len(meta) - 4 < 1:
        meta = f"{c.steps(r.steps)} · {c.duration(r.wall_s)}"
    rows = rc.curve_rows(_room()) or rc.SHORT
    painter = painter if len(series.line) >= 2 else None  # one point: the cells' dot
    k = rc.Card(series=series, last=series.total - 1 if r.series is None else
                min(r.steps, series.total) - 1, rows=rows, offset=r.offset, evals=evals,
                spikes=sorted((j + 1) * every - 1 for j in c.spikes(r.losses)),
                first=r.first_loss if r.first_loss is not None else
                r.losses[0] if r.losses else None,
                final=r.final_loss, vram=r.vram_now or r.peak_vram_gb,
                vram_peak=r.peak_vram_gb if r.card_gb else 0.0,
                ram=r.ram_now if r.ram_now is not None else r.peak_ram_gb,
                ram_peak=r.peak_ram_gb, meta=meta, word=word, pct=100, painter=painter,
                anchored=c.in_session,  # where the live card held them; elsewhere as before
                scores=r.scores)
    return rc.card(k, wc, t)


def _session_below(r: TrainResult, wc: int, lw: int) -> List[Line]:
    """In the session, under the card: where it is, and what next on one row (⏎ runs the
    first). Speed was on the live card; the receipt is in the folder."""
    after = bool(metric_rows(r.extra, lw, 16, r.acc_over))  # a pair run's accuracy rows are above
    out = [c.rail()] * after + c.rail_path("saved", r.output_dir, wc, lw if after else 7)
    said = c.continue_with(r.next_cmd, wc) if r.next_cmd else []  # sets the ⏎ step
    save = c.command(r.save_cmd) if r.save_cmd else ""
    if said:  # not a slash command: it runs in a shell, after you leave
        return out + [[], *said] + ([ln(S("  ", None), S(save, "tx"))] if save else [])
    if not c.next_step:
        return out + ([[], ln(S("  Keep it  ", "dm"), S(save, "hi"))] if save else [])
    head = c.next_step.split(" ")[0]
    row = [S("  Next  ", "dm"), S(head, "hi")]
    if head == "/eval":
        row += [S("  beside its base", "dm"), S("    /export", "tx")]
    rows = [ln(*row)]
    if save and width(rows[0]) + 4 + len(save) <= wc:
        rows[0] = ln(*row, S(f"    {save}", "tx"))
    elif save:
        rows.append(ln(S("        " + save, "tx")))
    return out + [[], *rows]


def _below(r: TrainResult, w: int, wc: int, lw: int) -> List[Line]:
    """Under the card: speed, the files, the next step."""
    if c.in_session:
        return _session_below(r, wc, lw)
    out = _speed_row(r.tokens_per_sec, lw)
    out.append(c.rail())
    out += c.rail_path("adapter", r.output_dir, wc, lw)
    if r.receipt:
        out += c.rail_path("receipt", r.receipt, wc, lw)
    if r.next_cmd:
        out.append([])
        out += c.continue_with(r.next_cmd, wc)
        if c.in_session and c.next_step:  # the slash command it heads the list with, said
            out.append(ln(S("  Next  ", "dm"), S(c.next_step, "hi"),
                          *((S("  beside its base   ", "dm"), S("/export", "hi"),
                             S("  a model to serve", "dm")) if c.next_step.startswith("/eval")
                            else ())))
    if r.next_note:
        out.append(ln(S(("  " if c.in_session else "") + r.next_note, "dm")))
    if r.save_cmd:
        out.append(ln(S("  Keep it as a recipe  " if c.in_session else "Keep it as a recipe: ",
                        "dm"), S(c.command(r.save_cmd), "hi")))
    return out


_STAGGER, _FADE = 24, 120  # ms between the rows under the card, and each one's fade


def finish_length(r: TrainResult, w: int) -> float:
    """Milliseconds the finish takes: until the last row under the card is in."""
    wc = min(w, c.CAP)
    return max(LAND + 60 + _STAGGER * len(_below(r, w, wc, 11)) + _FADE, LAND + RING)


def finish_frame(r: TrainResult, st: TrainState, w: int, t: float, frame: int = 0) -> List[Line]:
    """The moment ``t`` ms after the last step. Out: the run's rows go in the
    first 160 ms. In: from the landing on, the rows under the card, one by one."""
    wc, lw = min(w, c.CAP), 11
    fits = _fits(r, w)
    out: List[Line] = [[]] + _done_card(r, wc, t, r.painter if fits else None)
    gone = 1 - seq(t, 0, 160)
    out.append([] if t >= LAND else rc.fade(c.rail(), 1 - seq(t, 0, LAND)))
    out += metric_rows(r.extra, lw, 16, r.acc_over)
    below = _below(r, w, wc, lw)
    step = st.step_time if gone > 0 else 0.0
    stays = 1 if r.tokens_per_sec and not c.in_session else 0  # the speed row stays on
    if gone > 0:
        speed = _speed_row(r.tokens_per_sec or st.tokens_per_sec, lw)
        tail = [S(" · ", "dm"), S(f"{step:.2f}"), S(" s/step", "dm")] if step else []
        out += [[*row, *rc.fade(tail, gone)] if stays else rc.fade([*row, *tail], gone)
                for row in speed]
        out.append([])
        out.append(rc.fade(c.activity(st.label, c.duration(st.elapsed), frame, frac=1.0, eta=None,
                                      w=wc, animate=False), gone))
        out += [rc.fade(row, gone) for row in _status(st, w)]
        return out
    for i, row in enumerate(below):
        at = LAND + 60 + _STAGGER * i
        a = 1.0 if i < stays else seq(t, at, at + _FADE)
        out.append(rc.fade(row, a))
    return out


def _fits(r: TrainResult, w: int) -> bool:
    """The image is placed on the curve's first row: a card too tall for the
    window would scroll that row away and take the curve with it, so there
    the curve is drawn in cells, which clip a row at a time."""
    wc = min(w, c.CAP)
    rows = (2 + (rc.curve_rows(_room()) or rc.SHORT) + 3 + 1
            + len(metric_rows(r.extra, 11, 16, r.acc_over)))
    return rows + len(_below(r, w, wc, 11)) <= _room() or not c.in_session


def train_done(r: TrainResult, w: int) -> List[Line]:
    if w < PLAIN_BELOW:
        pairs = [
            ("poros", f"trained {c.steps(r.steps)} in {c.duration(r.wall_s)}", "tx"),
            ("final loss", f"{r.final_loss:.4f}", "tx"),
        ]
        if "accuracy" in r.extra:
            pairs.append(("accuracy", f"{r.extra['accuracy'] * 100:.0f}%", "tx"))
        if r.card_gb:
            peak = c.ceil1(c.gib(r.peak_vram_gb))
            pairs.append((PEAK, f"{peak:.1f} of {c.gib(r.card_gb):.1f} GiB", "a"))
        if r.peak_ram_gb is not None:
            pairs.append(("peak ram", c.gb(r.peak_ram_gb), "tx"))
        pairs.append(("adapter", r.output_dir, "tx"))
        if r.receipt:
            pairs.append(("receipt", r.receipt, "tx"))
        return c.plain_pairs(pairs, w)
    wc = min(w, c.CAP)
    lw, vw = 11, 9
    # the card from its second point; in the session from the first, as the live card drew it
    drawn = c.in_session and r.series is not None and bool(r.series.line)
    if w >= rc.MIN_W and (len(r.losses) >= 2 or drawn):
        card = _done_card(r, wc, rc.FINAL, r.painter if _fits(r, w) else None)
        return [*card, [], *metric_rows(r.extra, lw, 16, r.acc_over), *_below(r, w, wc, lw)]
    steps = c.steps(r.steps) + (f" · {r.recipe}" if r.recipe else "")
    out = [c.event("Trained", steps, c.duration(r.wall_s))]
    if w >= 80 and len(r.losses) >= 2:
        # the curve the run drew, three rows tall, the numbers that matter
        # beside it: the same block the run's page draws
        cw = max(16, min(wc - len(c.RAIL) - lw - 14, 64))
        first = r.first_loss if r.first_loss is not None else r.losses[0]
        drop = (f"{'↓' if r.final_loss <= first else '↑'} "
                f"{abs(r.final_loss - first) / first * 100:.0f}%") if first else ""
        side = [(f"{r.final_loss:.4f}", "a"), (f"from {first:.2f}", "dm"), (drop, "tx")]
        for i, row in enumerate(c.spark(r.losses, cw, 3)):
            out.append(c.rail(S(("loss" if i == 0 else "").ljust(lw), "dm"), *row,
                              S("  " + side[i][0], side[i][1])))
        end = c.num(r.steps)
        out.append(c.rail(S(" " * lw, None),
                          S("step 1" + " " * max(1, cw - 6 - len(end)) + end, "rl")))
    else:
        first = r.first_loss
        drop = (f"  ·  {'↓' if r.final_loss <= first else '↑'} "
                f"{abs(r.final_loss - first) / first * 100:.0f}%") if first else ""
        out.append(c.rail(S("final loss".ljust(lw), "dm"), S(f"{r.final_loss:.4f}", "a"),
                          S(f"  from {first:.4f}{drop}" if first is not None else "", "dm")))
    out += metric_rows(r.extra, lw, 16, r.acc_over)
    if r.eval_start is not None and r.eval_end is not None:
        better = (r.eval_start - r.eval_end) / r.eval_start * 100 if r.eval_start else 0.0
        out.append(c.rail(S("eval loss".ljust(lw), "dm"), S(f"{r.eval_end:.4f}", "tx"),
                          S(f"  from {r.eval_start:.4f} on held-out rows  ·  "
                            f"{'↓' if better >= 0 else '↑'} {abs(better):.0f}%", "dm")))
    elif r.eval_end is not None:  # its start not known (resumed from an older checkpoint)
        out.append(c.rail(S("eval loss".ljust(lw), "dm"), S(f"{r.eval_end:.4f}", "tx"),
                          S("  on held-out rows", "dm")))
        if r.eval_note:  # and why, where it was dropped: the held-out rows changed
            out.append(c.rail(S(" " * lw, None), S(r.eval_note, "dm")))
    jumps = sorted(c.spikes(r.losses))
    if jumps:  # the curve shows them in red; say where, in steps
        at = ", ".join(str((j + 1) * r.every) for j in jumps[:3])
        at += " …" if len(jumps) > 3 else ""
        out.append(c.kv("spikes", str(len(jumps)), lw=lw, vw=vw, style="e", tail=f"at step {at}"))
    out.append(c.rail())
    if r.card_gb:
        out.append(c.rail(S(PEAK.ljust(lw), "dm"),
                          *c.chips(r.peak_vram_gb / r.card_gb, 16, "a"),
                          S(f"  {c.ceil1(c.gib(r.peak_vram_gb)):.1f} GiB", "tx"),
                          S(f" of {c.gib(r.card_gb):.1f} GiB", "dm")))
    if r.peak_ram_gb is not None:
        meter = (c.chips(r.peak_ram_gb / r.ram_total_gb, 16, "tx") if r.ram_total_gb else [])
        out.append(c.rail(S("peak ram".ljust(lw), "dm"), *meter,
                          S(f"{'  ' if meter else ''}{c.gb(r.peak_ram_gb)}", "tx"),
                          S(f" of {c.gib(r.ram_total_gb):.0f} GiB" if r.ram_total_gb else "",
                            "dm")))
    if 0 < r.tokens_per_sec < 1:  # a long step on short rows
        out.append(c.rail(S("speed".ljust(lw), "dm"), S(f"{r.wall_s / r.steps:.0f}", "tx"),
                          S(" s/step", "dm")))
    elif r.tokens_per_sec:
        out.append(c.rail(S("speed".ljust(lw), "dm"), S(c.num(round(r.tokens_per_sec)), "tx"),
                          S(" tok/s", "dm")))
    out.append(c.rail())
    out += c.rail_path("adapter", r.output_dir, wc, lw)
    if r.receipt:
        out += c.rail_path("receipt", r.receipt, wc, lw)
    if r.next_cmd:
        out.append([])
        out += c.continue_with(r.next_cmd, wc)
        if c.in_session and c.next_step:  # the slash command it heads the list with, said
            out.append(ln(S("  Next  ", "dm"), S(c.next_step, "hi"),
                          *((S("  beside its base   ", "dm"), S("/export", "hi"),
                             S("  a model to serve", "dm")) if c.next_step.startswith("/eval")
                            else ())))
    if r.next_note:
        out.append(ln(S(("  " if c.in_session else "") + r.next_note, "dm")))
    if r.save_cmd:
        out.append(ln(S("  Keep it as a recipe  " if c.in_session else "Keep it as a recipe: ",
                        "dm"), S(c.command(r.save_cmd), "hi")))
    return out


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


@dataclass
class Fix:
    label: str  # "Retry with --block-size 2"
    command: str  # the exact command line
    note: str = ""  # "halves resident blocks"


def error_screen(
    title: str,
    reason: str,
    w: int,
    *,
    context: Sequence[str] = (),
    fix: Optional[Fix] = None,
    docs: Optional[str] = None,
) -> List[Line]:
    """``✕ Title  reason`` + rail context + rail ``fix <command>`` (accent)."""
    if fix:  # as it is typed here: /runs in the session, poros runs in a shell
        fix = Fix(fix.label, c.command(fix.command), fix.note)
    if w < PLAIN_BELOW:
        pairs = [("error", title.lower(), "e"), ("", reason, "tx")]
        pairs += [("", x, "dm") for x in context]
        if fix:
            pairs.append(("fix", fix.command, "a"))
        return c.plain_pairs(pairs, w)
    wc = min(w, c.CAP)
    out = c.error_event(title, reason, wc)
    for x in context:
        out.append(c.rail(S(x, "dm")))
    if fix:
        out.append(
            c.rail(
                S("fix  ", "dm"),
                S(fix.command, "a"),
                S("  " + fix.note, "dm")
                if fix.note and len(fix.command) + len(fix.note) + 11 <= wc
                else None,
            )
        )
    if docs:
        out.append(c.rail(S("docs ", "dm"), S(docs, "dm")))
    return out


def choice_menu(
    title: str, options: Sequence[Tuple[str, str]], selected: int, w: int, tag: str = "fix"
) -> List[Line]:
    """fx-style ``❯`` menu. ``options`` = [(label, note)]; Cancel is appended. Its last three
    rows are a card's floor (the command, the rule, the keys): the command's is empty, so in a
    tab every option stays with the question."""
    out: List[Line] = [ln(S(title.rstrip("."), "hi"), S(" · Choose one", "dm"))]
    rows = list(options) + [("Cancel", "")]
    lw = max(len(label) for label, _ in rows) + 2
    for i, (label, note) in enumerate(rows):
        sel = i == selected
        out.append(
            ln(
                S("❯ " if sel else "  ", "hi"),
                S(f"{i + 1}  ", "dm"),
                S(label.ljust(lw), "hi" if sel else "tx"),
                S(note, "tx" if sel else "dm") if note else None,
            )
        )
    n = len(rows)
    out.append([])
    out += c.key_hints(
        [f"1–{n} choose", "↑↓ options", "enter confirm", "esc cancel"], w
    )
    return out
