"""The run's loss curve: one drawing from the first step to the finished card.

Where the terminal shows kitty images (Ghostty, kitty, Konsole) the curve is a
real image laid over empty cells; anywhere else it is the same curve in eighth
blocks, at the same scale, beside the same labels. Nothing moves when the run
ends: the line is a causal average (it never knows the future) and the scale
only ever widens, so a column once drawn stays right. Stdlib only.

The finish is a pure function of milliseconds since the last step, so a frame
can be drawn alone: a test can ask for any moment of it.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from poros.cli.ui.gpu_image import _cell, _ground, _png, escape, new_id, protocol
from poros.cli.ui.style import ACCENT, ERROR, SPECTRUM_DARK, SPECTRUM_LIGHT, Palette, resolve

LAND = 420.0  # ms: the frame closes and the glint reaches the last step
RING = 360.0  # ms the landing ring takes to open and fade

_SHAPES = {10: (2, 3), 7: (1, 2)}  # rows: (the first gridline's row, rows between gridlines)
_NICE = (1.0, 2.0, 2.5, 5.0)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def ease(k: float) -> float:
    """Fast start, soft landing."""
    return 1 - (1 - _clamp(k)) ** 3


def seq(t: Optional[float], a: float, b: float) -> float:
    """How far ``t`` is from ``a`` to ``b``, 0..1; 0 while live (no ``t``)."""
    return 0.0 if t is None else _clamp((t - a) / (b - a))


class Series:
    """The steps so far and the line through them, grown as they come."""

    def __init__(self, total: int) -> None:
        self.total = max(total, 1)
        # a short run has no noise to smooth away: its line goes through its steps
        self.alpha = max(0.035, min(1.0, 42.0 / self.total))
        self.raw: List[float] = []
        self.line: List[float] = []
        self.lo, self.hi = math.inf, -math.inf

    def extend(self, losses: Sequence[float]) -> "Series":
        for v in losses[len(self.raw):]:
            ok = isinstance(v, (int, float)) and math.isfinite(v)
            e = self.line[-1] if self.line else (v if ok else 0.0)
            e = e + (v - e) * self.alpha if ok and self.line else e
            self.raw.append(float(v) if ok else e)
            self.line.append(e)
            self.lo, self.hi = min(self.lo, e), max(self.hi, e)
        return self


@dataclass(frozen=True)
class Scale:
    """Gridlines sit on cell centres, so the text labels beside them line up."""

    top: float  # the value on the first gridline
    step: float
    first: int
    gap: int
    rows: int

    def row(self, v: float) -> float:
        return self.first + (self.top - v) / self.step * self.gap

    def labels(self) -> Dict[int, str]:
        d = 0 if self.step >= 1 else max(1, -math.floor(math.log10(self.step) + 1e-9))
        if abs(round(self.step, d) - self.step) > 1e-9:
            d += 1
        return {self.first + i * self.gap: f"{self.top - i * self.step:.{d}f}" for i in range(3)}


def fit_scale(values: Sequence[float], rows: int) -> Scale:
    """The tightest round scale that holds ``values``. Their range only grows
    as a run goes on, so this only ever widens."""
    first, gap = _SHAPES[rows]
    hi, lo = max(values), min(values)
    up = (first + 0.5) / gap - 0.08  # a little air, so the line never rides the edge
    down = (rows - 0.5 - first - 2 * gap) / gap - 0.08
    span = max(hi - lo, abs(hi) * 0.02, 1e-3)
    exp = math.floor(math.log10(span / (2 + up + down)))
    while True:
        for nice in _NICE:
            s = nice * 10.0 ** exp
            top = math.ceil((hi - up * s) / s - 1e-9) * s
            if lo >= 0 and top < 2 * s:  # no gridline under zero for a loss
                top = 2 * s
            if top - (2 + down) * s <= lo and top + up * s >= hi:
                return Scale(round(top, 10), round(s, 10), first, gap, rows)
        exp += 1


def x_of(step: int, total: int, cols: int) -> float:
    """A step's place across ``cols`` cells, in cells."""
    return 0.5 + step / max(total - 1, 1) * (cols - 1)


def block_rows(series: Series, scale: Scale, cols: int, last: int, offset: int = 0,
               evals: Sequence[Tuple[int, float]] = (), spikes: Sequence[int] = (),
               t: Optional[float] = None) -> List[List[Tuple[str, object]]]:
    """No images: the same curve in cells. ``last`` is the last step drawn."""
    n, total = len(series.line), series.total
    out: List[List[Tuple[str, object]]] = [[] for _ in range(scale.rows)]
    k = seq(t, 0, LAND)
    p = k * k * (3 - 2 * k)
    gliding = t is not None and 0 < t < LAND
    landed = t is not None and t >= LAND
    for c in range(cols):
        if total < cols:  # fewer steps than cells: the line runs between them
            pos = c / max(cols - 1, 1) * (total - 1)
            near = round(pos)
            a = near
            # a step belongs to one cell, so a mark on it is drawn once
            b = near + 1 if round(near / max(total - 1, 1) * (cols - 1)) == c else near
            shown = pos <= last + 1e-9 and pos - offset <= n - 1 + 1e-9
            head_here = c == cols - 1 or (c + 1) / max(cols - 1, 1) * (total - 1) > last + 1e-9
            i = max(0.0, min(n - 1.0, pos - offset))
            j = min(int(i), max(n - 2, 0))
            line_v = (series.line[j] + (series.line[min(j + 1, n - 1)] - series.line[j]) * (i - j)
                      if shown else 0.0)
        else:
            a = int(c * total / cols)
            b = max(a + 1, int((c + 1) * total / cols))
            shown = a <= last and a - offset < n
            head_here = last < b
            line_v = series.line[max(0, min(n - 1, min(b - 1, last) - offset))] if n else 0.0
        if not shown:
            for row in out:
                row.append((" ", None))
            continue
        hot = [s for s in spikes if a <= s + offset < b]
        v = series.raw[hot[0]] if hot else line_v
        eighths = (scale.rows - (scale.row(v) + 0.5)) * 8
        ev = next((e for e in evals if a <= e[0] < b and e[0] <= last), None)
        ev_row = round(scale.row(ev[1])) if ev else -1
        f = c / max(cols - 1, 1)
        head = head_here and not landed
        ink: object = ("sp", f)
        if gliding and -0.1 < f - p <= 0:
            ink = ("gl", f, 0.7 * (1 + (f - p) / 0.1))
        if landed and c == cols - 1:
            ink = "a"
        if head:
            ink = "hi"
        if hot:
            ink = "e"
        if n == 1:  # one step is a point: a dot where it is, nothing under it
            if c:  # a run of one step reaches every column; its point is the first
                for row in out:
                    row.append((" ", None))
                continue
            at = max(0, min(scale.rows - 1, round(scale.row(v))))
            for r in range(scale.rows):
                mark = ("○", "hi") if r == ev_row and r != at else (" ", None)
                out[r].append(("●", "a" if landed else "hi") if r == at else mark)
            continue
        for r in range(scale.rows):
            if r == ev_row:
                out[r].append(("○", "hi"))
                continue
            over = eighths - (scale.rows - 1 - r) * 8
            cell = int(max(0, min(8, round(over))))
            near = over < 12
            # the head, the landing and a spike mark the line; the fill under them stays quiet
            # and fades with depth, as the image's does (the third item is how far it fades)
            fade = min(0.94, 0.7 + (over - 12) / 8 * 0.05)
            out[r].append((" ▁▂▃▄▅▆▇█"[cell], ink if near else ("sp", f, fade)))
    return out


# --------------------------------------------------------------------------
# the image
# --------------------------------------------------------------------------


class Painter:
    """Draws the curve as an image, keeping the columns already right: a new
    step costs the columns it adds. One per run, on one image id."""

    def __init__(self, pal: Palette, stream=None) -> None:
        self.ident, self.halo_id = new_id(), new_id()
        self.fg = _ground(pal)[0]
        self.stops = SPECTRUM_DARK if pal.dark else SPECTRUM_LIGHT
        self.cell = _cell(stream)
        # over ssh every image crosses the wire: send them less often
        self.remote = bool(os.environ.get("SSH_CONNECTION"))  # each image crosses the network
        self.every = 2.0 if self.remote else 0.5
        self._key: tuple = ()
        self._sent, self._last = 0.0, ""
        self._drawn = -1
        self._halos: Dict[int, str] = {}

    @classmethod
    def maybe(cls, stream=None) -> Optional["Painter"]:
        """A painter where the terminal shows kitty images. Asked on the main
        thread: the live thread must not read the terminal."""
        stream = stream or sys.stdout
        return cls(resolve(stream), stream) if protocol(stream) == "kitty" else None

    # -- geometry ------------------------------------------------------------

    def _reset(self, series: Series, scale: Scale, cols: int) -> None:
        cw, ch = self.cell
        self.cols, self.rows, self.scale_ = cols, scale.rows, scale
        self.w, self.h = (cols + 2) * cw, scale.rows * ch
        self.s = max(1.0, ch / 17.0)  # one drawn unit, in pixels
        self.px = bytearray(self.w * self.h * 4)
        self.lines: Dict[int, List[Tuple[int, int, int, int, float]]] = {}
        self.base = self.h - max(1, round(self.s))
        self._drawn = -1
        self.spec = [self._spec(x) for x in range(self.w)]
        grid = [(round(ch * (r + 0.5)), 0.055) for r in scale.labels()] + [(self.base, 0.16)]
        self.grid = dict(grid)
        self.span = (cols - 1) * cw
        self.x0 = cw * 0.5
        for y, a in grid:
            self._hline(y, 0, cols * cw, a)
        ticks = [self.x0, self.x0 + self.span / 2, self.x0 + self.span]
        for x in ticks:
            for y in range(self.base - round(3 * self.s), self.base):
                self._over(int(x), y, self.fg, 0.16)

    def _spec(self, x: int) -> Tuple[int, int, int]:
        f = _clamp((x - self.cell[0] * 0.5) / max((self.cols - 1) * self.cell[0], 1))
        v = f * (len(self.stops) - 1)
        i = min(int(v), len(self.stops) - 2)
        a, b, m = self.stops[i], self.stops[i + 1], v - i
        return (round(a[0] + (b[0] - a[0]) * m), round(a[1] + (b[1] - a[1]) * m),
                round(a[2] + (b[2] - a[2]) * m))

    def px_of(self, step: float, total: int) -> float:
        return self.x0 + step / max(total - 1, 1) * self.span

    def y_of(self, v: float) -> float:
        return max(0.0, self.cell[1] * (self.scale_.row(v) + 0.5))

    # -- pixels --------------------------------------------------------------

    def _over(self, x: int, y: int, c: Tuple[int, int, int], a: float,
              px: Optional[bytearray] = None) -> None:
        if not (0 <= x < self.w and 0 <= y < self.h) or a <= 0:
            return
        p = self.px if px is None else px
        i = (y * self.w + x) * 4
        da = p[i + 3] / 255
        oa = a + da * (1 - a)
        if oa <= 0:
            return
        k = a / oa
        p[i] = round(p[i] + (c[0] - p[i]) * k)
        p[i + 1] = round(p[i + 1] + (c[1] - p[i + 1]) * k)
        p[i + 2] = round(p[i + 2] + (c[2] - p[i + 2]) * k)
        p[i + 3] = round(oa * 255)

    def _hline(self, y: int, x0: int, x1: int, a: float) -> None:
        for x in range(max(0, x0), min(self.w, x1)):
            self._over(x, y, self.fg, a)

    def _column(self, x: int, series: Series, offset: int) -> None:
        """One column of the drawing, from nothing."""
        px, w, h = self.px, self.w, self.h
        for y in range(h):
            px[(y * w + x) * 4 + 3] = 0
        for y, a in self.grid.items():
            self._over(x, y, self.fg, a)
        total, n = series.total, len(series.line)
        at = (x - self.x0) / max(self.span, 1) * (total - 1) - offset

        def value(arr: List[float], i: float) -> float:
            i = max(0.0, min(n - 1.0, i))
            j = min(int(i), n - 2) if n > 1 else 0
            return arr[j] + (arr[min(j + 1, n - 1)] - arr[j]) * (i - j)

        v = value(series.line, at)
        y = self.y_of(v)
        col = self.spec[x]
        s, base = self.s, self.base
        # the area: the spectrum, fading to nothing before the baseline
        top = max(0, int(y))
        rise = max(1.0, base - self.y_of(self.scale_.top + self.scale_.step * 0.8))
        for yy in range(top, base):
            f = 1 - (yy - (base - rise)) / rise
            self._over(x, yy, col, 0.2 * max(0.0, f) ** 2 * (1 if yy > y else yy + 1 - y))
        # the steps themselves: quiet texture, so noise shows without being the result
        per = (total - 1) / max(self.span, 1)
        i0, i1 = at - per / 2, at + per / 2
        lo = int(max(0, math.ceil(i0)))
        hi = int(min(n - 1, math.floor(i1)))
        # only what this column covers: the path between steps, and any steps inside it
        vals = [value(series.raw, i0), value(series.raw, i1), *series.raw[lo:hi + 1]]
        if i1 >= 0 and i0 <= n - 1:
            ya, yb = self.y_of(max(vals)), self.y_of(min(vals))
            for yy in range(max(0, int(ya)), min(h, int(yb) + 1)):
                self._over(x, yy, self.fg, 0.17 * (0.55 if yb - ya < 1 else 1))
        # the line, with a soft bed under it
        dv = (value(series.line, at + 3 * per) - value(series.line, at - 3 * per)) / 6
        slope = abs(self.y_of(v + dv) - y)
        cosine = 1 / math.hypot(1, slope)
        for half, alpha, keep in ((2.25 * s, 0.16, False), (0.8 * s, 1.0, True)):
            reach = (half + 1) / cosine
            for yy in range(max(0, int(y - reach)), min(h, int(y + reach) + 2)):
                cov = half + 0.5 - abs(yy + 0.5 - y) * cosine
                if cov > 0:
                    a = min(1.0, cov) * alpha
                    self._over(x, yy, col, a)
                    if keep:
                        self.lines.setdefault(x, []).append(((yy * w + x) * 4, *col, a))

    def _advance(self, series: Series, last: int, offset: int) -> int:
        """Columns up to the head: the one the head was in is drawn again, as
        its steps were not all in yet."""
        hx = min(self.w - 1, int(self.px_of(last, series.total)))
        for x in range(max(int(self.x0), self._drawn), hx + 1):
            self.lines.pop(x, None)
            self._column(x, series, offset)
        self._drawn = hx
        return hx

    def _disc(self, px: bytearray, cx: float, cy: float, r: float,
              c: Tuple[int, int, int], a: float) -> None:
        for y in range(int(cy - r - 1), int(cy + r + 2)):
            for x in range(int(cx - r - 1), int(cx + r + 2)):
                cov = r + 0.5 - math.hypot(x + 0.5 - cx, y + 0.5 - cy)
                if cov > 0:
                    self._over(x, y, c, min(1.0, cov) * a, px)

    def _ring(self, px: bytearray, cx: float, cy: float, r: float, wt: float,
              c: Tuple[int, int, int], a: float, cut: bool = False) -> None:
        reach = r + wt + 1
        for y in range(int(cy - reach), int(cy + reach) + 2):
            for x in range(int(cx - reach), int(cx + reach) + 2):
                d = abs(math.hypot(x + 0.5 - cx, y + 0.5 - cy) - r)
                cov = wt / 2 + 0.5 - d
                if cov > 0:
                    self._over(x, y, c, min(1.0, cov) * a, px)

    def _clear(self, px: bytearray, cx: float, cy: float, r: float) -> None:
        for y in range(max(0, int(cy - r - 1)), min(self.h, int(cy + r + 2))):
            for x in range(max(0, int(cx - r - 1)), min(self.w, int(cx + r + 2))):
                if math.hypot(x + 0.5 - cx, y + 0.5 - cy) <= r:
                    px[(y * self.w + x) * 4 + 3] = 0

    def _marks(self, px: bytearray, series: Series, last: int, offset: int,
               evals: Sequence[Tuple[int, float]], spikes: Sequence[int],
               t: Optional[float]) -> None:
        s, total, base = self.s, series.total, self.base
        for j in spikes:  # where it happened, down to the axis
            if j + offset > last:
                continue
            x, y = self.px_of(j + offset, total), self.y_of(series.raw[j])
            for yy in range(int(y), base):
                self._over(int(x), yy, ERROR, 0.3, px)
            self._disc(px, x, y, 2 * s, ERROR, 1.0)
        # held out: a dotted thread through open dots, so the gap to the
        # training line reads at a glance
        seen = [(self.px_of(i, total), self.y_of(v)) for i, v in evals if i <= last]
        for (xa, ya), (xb, yb) in zip(seen, seen[1:]):
            d = math.hypot(xb - xa, yb - ya)
            for k in range(1, int(d / (4.2 * s))):
                u = k * 4.2 * s / d
                self._disc(px, xa + (xb - xa) * u, ya + (yb - ya) * u, 0.55 * s, self.fg, 0.32)
        for x, y in seen:
            self._clear(px, x, y, 2.5 * s)
            self._ring(px, x, y, 2.5 * s, 1.0 * s, self.fg, 0.82)
        hx = self.px_of(last, total)
        hy = self.y_of(series.line[min(last - offset, len(series.line) - 1)])
        if t is None or t < LAND:
            fade = 1 - seq(t, 0, LAND)
            if fade > 0:
                for yy in range(int(hy), base):
                    self._over(int(hx), yy, self.fg, 0.2 * fade, px)
            self._disc(px, hx, hy, 2.6 * s, self.fg, 1.0)
        else:
            k = ease(seq(t, LAND, LAND + RING))
            if k < 1:
                self._ring(px, hx, hy, (3 + 9 * k) * s, (1.5 - k) * s, ACCENT, 0.6 * (1 - k))
            self._disc(px, hx, hy, 6.5 * s, ACCENT, 0.2 * k)
            self._disc(px, hx, hy, 3 * s, ACCENT, 1.0)

    def _glint(self, px: bytearray, t: float) -> None:
        """Runs the curve from the first step to the last and arrives as the
        card closes: the line itself, lit, behind a travelling edge."""
        k = seq(t, 0, LAND)
        p = k * k * (3 - 2 * k)
        edge = self.x0 + p * self.span
        for band, a in ((0.18, 0.35), (0.09, 0.55), (0.035, 0.8)):
            lit = a * (1 - 0.3 * k)
            for x in range(max(0, int(edge - band * self.w)), min(self.w, int(edge) + 1)):
                for i, r, g, b, cov in self.lines.get(x, ()):
                    m = cov * lit
                    px[i] = min(255, round(px[i] + r * m))
                    px[i + 1] = min(255, round(px[i + 1] + g * m))
                    px[i + 2] = min(255, round(px[i + 2] + b * m))

    # -- frames --------------------------------------------------------------

    def frame(self, series: Series, scale: Scale, cols: int, last: int, offset: int = 0,
              evals: Sequence[Tuple[int, float]] = (), spikes: Sequence[int] = (),
              t: Optional[float] = None) -> str:
        """The image as an escape, placed at the cursor. While live an unchanged
        moment returns the same string, and a row the same as last frame isn't
        written again, so the image isn't sent again either."""
        now = time.monotonic()
        key = (scale, cols)
        forced = key != self._key
        if forced:
            self._key = key
            self._reset(series, scale, cols)
        state = (last, tuple(evals), tuple(spikes))
        if (t is None and not forced and self._last
                and (now - self._sent < self.every or state == getattr(self, "_state", None))):
            return self._last
        self._advance(series, last, offset)
        px = bytearray(self.px)
        self._marks(px, series, last, offset, evals, spikes, t)
        if t is not None and 0 < t < LAND:
            self._glint(px, t)
        self._sent, self._state = now, state
        self._last = escape(_png(self.w, self.h, px, level=1), cols + 2, self.ident)
        return self._last

    def halo(self, series: Series, last: int, offset: int, phase: float, up: int,
             col: int) -> str:
        """The soft breath around the head, its own small image above the
        curve, sent from the row ``up`` rows under the curve's first. The
        curve itself is sent only as steps come."""
        if not hasattr(self, "px") or self.remote or os.environ.get("POROS_NO_ANIM"):
            return ""
        s = self.s
        size = int(16 * s) | 1
        step = int(phase * 8) % 8
        data = self._halos.get(step)
        if data is None:
            b = (math.sin(step / 8 * 2 * math.pi) + 1) / 2
            r, a = (4.5 + 2.5 * b) * s, 0.16 - 0.09 * b
            buf = bytearray(size * size * 4)
            m = size / 2
            for y in range(size):
                for x in range(size):
                    cov = r + 0.5 - math.hypot(x + 0.5 - m, y + 0.5 - m)
                    if cov > 0:
                        i = (y * size + x) * 4
                        buf[i:i + 3] = bytes(self.fg)
                        buf[i + 3] = round(min(1.0, cov) * a * 255)
            data = _b64(_png(size, size, buf, level=1))
            self._halos[step] = data
        cw, ch = self.cell
        # the head as drawn: the curve is sent at most every self.every seconds
        last = min(last, getattr(self, "_state", (last,))[0])
        hx = self.px_of(last, series.total) - size / 2
        hy = self.y_of(series.line[min(last - offset, len(series.line) - 1)]) - size / 2
        cx, x_off = max(0, int(hx // cw)), max(0, int(hx % cw))
        cy, y_off = max(0, int(hy // ch)), max(0, int(hy % ch))
        return (f"\x1b7\x1b[{up - cy}A\x1b[{col + cx + 1}G"
                f"\x1b_Ga=T,f=100,t=d,i={self.halo_id},p=1,X={x_off},Y={y_off},z=1,C=1,q=2;"
                f"{data}\x1b\\\x1b8")


def _b64(data: bytes) -> str:
    import base64

    return base64.standard_b64encode(data).decode()
