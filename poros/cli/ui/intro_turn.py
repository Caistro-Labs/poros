"""The start box's entrance where the terminal shows kitty images: the arch
mark turns into place.

It turns about its upright axis, one and a quarter turns, with perspective,
growing from 0.6x as it comes round; a glint crosses it as it faces you; then
it hands over to the mark in text cells, in the same place and color, and the
words fade in beside it as they do after the river. ~1.4 s, any key skips.

A turn about the upright axis keeps vertical edges vertical, so each column
of pixels is one column of the mark, stretched: the frame is drawn column by
column with byte slices, a few milliseconds in pure Python. Stdlib only.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

from poros.cli.ui import components as c
from poros.cli.ui.text import Line

RGB = Tuple[int, int, int]

DUR = 0.95  # the turn; then the mark is text
WORDS_AT = 0.6  # the words begin to fade in
WORDS_RATE = 1.7  # the words' clock runs this much faster than the turn's
SETTLED = 2.25  # the words' clock when the last row has faded in
TURNS = 450.0  # degrees still to turn at the start
COLS, ROWS = len(c.MARK[0]) + 4, len(c.MARK) + 4  # the mark and two cells all round
EDGE = 2  # cells between the region's edge and the mark
SUB = 4  # samples across each pixel column
FULL = 63  # one sample's full coverage: four of them still fit in a byte


def ease(u: float) -> float:
    """cubic-bezier(0.16, 1, 0.3, 1): fast out of the start, long settle."""
    u = min(1.0, max(0.0, u))
    lo, hi = 0.0, 1.0
    for _ in range(30):
        s = (lo + hi) / 2
        x = 3 * (1 - s) ** 2 * s * 0.16 + 3 * (1 - s) * s * s * 0.3 + s ** 3
        lo, hi = (s, hi) if x < u else (lo, s)
    s = (lo + hi) / 2
    return 3 * (1 - s) ** 2 * s + 3 * (1 - s) * s * s + s ** 3


def words_clock(t: float) -> float:
    """Where the boxed intro's words are, as the river's clock counts."""
    return 0.95 + max(0.0, t - WORDS_AT) * WORDS_RATE


def settle_time() -> float:
    return WORDS_AT + (SETTLED - 0.95) / WORDS_RATE


def _mix(a: RGB, b: RGB, k: float) -> RGB:
    return tuple(round(x + (y - x) * k) for x, y in zip(a, b))  # type: ignore[return-value]


def _lin(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _gam(v: float) -> float:
    v = min(1.0, max(0.0, v))
    return v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


_SRGB_TO_P3 = ((0.8225, 0.1774, 0.0), (0.0332, 0.9669, 0.0), (0.0171, 0.0724, 0.9108))


def to_p3(rgb: RGB) -> RGB:
    """An sRGB color in Display P3 numbers: the same color, where a terminal
    reads image pixels as P3."""
    lin = [_lin(v / 255) for v in rgb]
    return tuple(round(_gam(sum(m * x for m, x in zip(row, lin))) * 255)  # type: ignore[return-value]
                 for row in _SRGB_TO_P3)


def images_in_p3() -> bool:
    """Ghostty on macOS shows image pixels as Display P3 and text as sRGB:
    measured, the accent (255,107,53) as an image shows as (255,94,22)."""
    return sys.platform == "darwin" and os.environ.get("TERM_PROGRAM", "").lower() == "ghostty"


def _spans() -> List[List[Tuple[float, float]]]:
    """Each column of the mark: its filled runs, top to bottom, in rows."""
    out: List[List[Tuple[float, float]]] = []
    for i in range(len(c.MARK[0])):
        runs: List[Tuple[float, float]] = []
        for r, row in enumerate(c.MARK):
            g = row[i]
            if g == " ":
                continue
            a, b = (r, r + 0.5) if g == "▀" else (r + 0.5, r + 1) if g == "▄" else (r, r + 1)
            if runs and abs(runs[-1][1] - a) < 1e-9:
                runs[-1] = (runs[-1][0], b)
            else:
                runs.append((a, b))
        out.append(runs)
    return out


class Turn:
    """Frames of the turn for one cell size and accent."""

    def __init__(self, cell: Tuple[int, int], accent: RGB, p3: bool = False) -> None:
        self.cw, self.ch = cell
        self.W, self.H = COLS * self.cw, ROWS * self.ch
        self.Lw, self.Lh = len(c.MARK[0]) * self.cw, len(c.MARK) * self.ch
        self.cx = EDGE * self.cw + self.Lw / 2
        self.cy = EDGE * self.ch + self.Lh / 2
        self.accent, self.p3 = accent, p3
        self.spans = _spans()
        self.ident = 0

    # -- one frame ---------------------------------------------------------------

    def pose(self, t: float) -> dict:
        """Where the turn is at ``t``: angle, scale, opacity, color, glint."""
        p = ease(t / DUR)
        th = -TURNS * (1 - p)
        cos_t = math.cos(math.radians(th))
        light = 0.55 + 0.45 * abs(cos_t)
        face = self.accent if cos_t >= 0 else _mix(self.accent, (0, 0, 0), 0.4)
        return dict(th=th, s=0.6 + 0.4 * p, alpha=1 - (1 - min(1.0, t / 0.15)) ** 2,
                    color=_mix((0, 0, 0), face, light), glint=(t - 0.55) / 0.4 * 1.6 - 0.3)

    def coverage(self, th: float, s: float) -> Optional[bytes]:
        """The mark's coverage, one byte a pixel (0..252), column after
        column; None when it is edge-on."""
        cos_t, sin_t = math.cos(math.radians(th)), math.sin(math.radians(th))
        if abs(cos_t) < 0.02:
            return None
        H, D, hw = self.H, 3.0 * self.Lw, self.Lw / 2
        empty = bytes(H)
        full = bytes([FULL]) * H
        cols: List[bytes] = []
        for px in range(self.W):
            total = 0
            for k in range(SUB):
                X = px + (k + 0.5) / SUB - self.cx
                den = s * (cos_t * D - X * sin_t)
                if abs(den) < 1e-9:
                    continue
                x = X * D / den  # where on the mark (unscaled) this sample looks
                if not -hw <= x < hw:
                    continue
                f = D / (D + x * s * sin_t)
                if f <= 0:
                    continue
                runs = self.spans[int((x + hw) // self.cw)]
                if not runs:
                    continue
                ky, top = s * f * self.ch, self.cy - s * f * self.Lh / 2
                col = bytearray(empty)
                for a, b in runs:
                    y0, y1 = max(0.0, top + a * ky), min(float(H), top + b * ky)
                    if y1 <= y0:
                        continue
                    i0, i1 = int(y0), int(y1)
                    if i0 == i1:
                        col[i0] += round(FULL * (y1 - y0))
                        continue
                    col[i0] += round(FULL * (i0 + 1 - y0))
                    if i1 > i0 + 1:
                        col[i0 + 1:i1] = full[:i1 - i0 - 1]
                    if i1 < H:
                        col[i1] += round(FULL * (y1 - i1))
                total += int.from_bytes(col, "big")
            cols.append(total.to_bytes(H, "big") if total else empty)
        return b"".join(cols)

    def rgba(self, t: float) -> Optional[bytes]:
        """The frame at ``t`` as RGBA rows, or None when nothing shows."""
        pose = self.pose(t)
        cov = self.coverage(pose["th"], pose["s"])
        if cov is None or pose["alpha"] <= 0.003:
            return None
        W, H = self.W, self.H
        a = pose["alpha"] * 255 / (FULL * SUB)
        alpha = bytes(min(255, round(v * a)) for v in range(256))
        base = pose["color"]
        if self.p3:
            base = to_p3(base)
        # the glint: a band along x + 0.45 y, so each row is a slice of one strip
        n = W + int(0.45 * H) + 2
        u, span = pose["glint"], self.Lw + 0.45 * self.Lh
        x0 = self.cx - self.Lw / 2 + 0.45 * (self.cy - self.Lh / 2)
        strip = [base] * n
        if -0.4 < u < 1.4:
            strip = [_mix(base, (255, 255, 255),
                          0.38 * math.exp(-(((q - x0) / span - u) ** 2) / (2 * 0.07 ** 2)))
                     for q in range(n)]
        chan = [bytes(px[i] for px in strip) for i in range(3)]
        out = bytearray(W * H * 4)
        for y in range(H):
            o, row = int(0.45 * y), y * W * 4
            line = out[row:row + W * 4]
            line[0::4], line[1::4], line[2::4] = (ch[o:o + W] for ch in chan)
            line[3::4] = cov[y::H].translate(alpha)
            out[row:row + W * 4] = line
        return bytes(out)

    def escape(self, t: float, up: int, col: int) -> str:
        """The frame as a raw segment for the row the mark's top is on: the
        image goes ``up`` rows higher at column ``col``, and the cursor comes
        back. Empty when nothing shows."""
        from poros.cli.ui.gpu_image import _png, escape, new_id

        px = self.rgba(t)
        if px is None:
            return ""
        if not self.ident:
            self.ident = new_id()
        img = escape(_png(self.W, self.H, bytearray(px), level=1), COLS, self.ident)
        img = img.replace("C=1,", f"r={ROWS},C=1,z=1,", 1)
        move = (f"\x1b[{up}A" if up else "") + f"\x1b[{col + 1}G"
        return "\x1b7" + move + img + "\x1b8"


# -- the boxed intro with the turn ------------------------------------------------


def _mark_field(lit: Optional[float]) -> Callable[[float, int], List[List[tuple]]]:
    from poros.cli.ui import intro

    def field(_t: float, width: int) -> List[List[tuple]]:
        grid: List[List[tuple]] = [[(" ", None)] * width for _ in range(intro.ROWS)]
        if lit is not None:
            for mr, row in enumerate(c.MARK):
                for i, g in enumerate(row):
                    x = intro.LEFT + i
                    if g != " " and x < width:
                        grid[intro.TOP + mr][x] = (g, ("mk", lit))
        return grid

    return field


def frames(w: int, right: Sequence[Line], turn: Turn) -> Callable[[float], List[Line]]:
    """The boxed intro at time ``t`` with the mark turning in; from
    ``settle_time()`` on, exactly the settled box."""
    from poros.cli.ui import intro

    dark, ink = _mark_field(None), _mark_field(1.0)
    margin = (w - intro.box_width(w)) // 2
    col = margin + 1 + intro.PAD + intro.LEFT - EDGE

    def draw(t: float) -> List[Line]:
        if t >= DUR:
            return intro.box_frame(min(words_clock(t), intro.DURATION + 1.0), w, right, field=ink)
        lines = intro.box_frame(words_clock(t), w, right, field=dark)
        row = intro.box_mark_row(right)
        raw = turn.escape(t, EDGE, col)
        if raw:
            lines[row] = [(raw, "raw")] + lines[row]
        return lines

    draw.duration = settle_time()  # type: ignore[attr-defined]
    return draw


def maybe(w: int, line1: str, line2: str) -> Optional[Callable[[float], List[Line]]]:
    """The turning intro where it can run: the full screen is up, the box
    fits, and the terminal shows kitty images locally. None elsewhere."""
    from poros.cli.ui import intro, tui
    from poros.cli.ui.gpu_image import _cell, protocol
    from poros.cli.ui.style import ACCENT

    scr = tui.current
    if scr is None or w < intro.BOX_MIN or os.environ.get("SSH_CONNECTION"):
        return None
    stream = sys.__stdout__
    try:
        if protocol(stream) != "kitty":
            return None
        cell = _cell(stream)
    except Exception:  # noqa: BLE001 -- a terminal that won't say: the river
        return None
    pal = scr.pal
    accent = ACCENT if getattr(pal, "adaptive", False) else pal._legible(ACCENT)
    turn = Turn(cell, accent, p3=images_in_p3())
    return frames(w, intro.box_column(line1, line2, intro.box_room(w)), turn)
