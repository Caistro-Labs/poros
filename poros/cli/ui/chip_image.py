"""A Mac's chip for /machine, drawn as a real image where the terminal shows one.

The dies are seen from straight above in the GPU cards' own style: the
terminal's soft faces, hairline ink, one accent, through each GPU's L2.
Every block is drawn to its kind from the chip's floorplan (chip_plans,
traced from Apple's images): GPU cores with their SIMD partitions, register
files, ALU lanes and tile memory, the lit ones as many as the chip has on;
CPU cores with their caches and front ends; the Neural Engine's MAC arrays;
SRAM in banks; memory PHYs in channel slices; logic in cell rows. Detail is
dropped, never smeared, where it would fall under a few pixels. No text, no
logo, nothing outside the dies.

Its tones are the terminal's own, so it sits on any ground: a soft theme
(Solarized) gets its lines stepped further out, a ground of the accent's own
hue (Red Sands) gets the ink for the accent, and where Ghostty reads image
pixels as Display P3 every tone is converted, or a tinted ground shifts under
the chip. It enters with a scan, top to bottom (kitty only, 0.6 s). Drawn
once per chip, look and size, then kept in ~/.cache/poros/chip. Stdlib only.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import os
import random
import re
import zlib
from functools import lru_cache
from typing import Dict, Optional, Tuple

from poros.cli.ui import chip_plans, gpu_image, intro_turn
from poros.cli.ui.style import Palette, _contrast, mix

RGB = Tuple[int, int, int]
Pt = Tuple[float, float]

DUR = 0.6  # the scan, in seconds
_VERSION = 1
_PAD = 0.02  # of the chip's width, air all round

# -- the pen ---------------------------------------------------------------------

FINE = 3.2  # px: nothing finer is drawn


class Pen:
    """Ops in die units (mm, y down), knowing how many pixels a unit will be."""

    def __init__(self, ppu, flip_x=False, flip_y=False, ox=0.0, oy=0.0, W=0.0, H=0.0):
        self.ops, self.ppu = [], ppu
        self.fx, self.fy, self.ox, self.oy, self.W, self.H = flip_x, flip_y, ox, oy, W, H

    def P(self, x, y):
        if self.fx:
            x = self.W - x
        if self.fy:
            y = self.H - y
        return (self.ox + x, self.oy + y)

    def px(self, u):
        return u * self.ppu

    def rect(self, x, y, w, h, fill=None, a=1.0, edge=None, wt=0.7):
        if w <= 0 or h <= 0:
            return
        pts = [self.P(x, y), self.P(x + w, y), self.P(x + w, y + h), self.P(x, y + h)]
        if fill:
            self.ops.append(("fill", pts, fill, 1.0))
        if edge and min(self.px(w), self.px(h)) >= 2:
            for p0, p1 in zip(pts, pts[1:] + pts[:1]):
                self.ops.append(("line", p0, p1, edge, wt))

    def line(self, x0, y0, x1, y1, tone, wt=0.5):
        self.ops.append(("line", self.P(x0, y0), self.P(x1, y1), tone, wt))

    def lines(self, x, y, w, h, n, vertical, tone, wt=0.45, gap=FINE):
        """n - 1 rules across a box, fewer if they would come closer than gap px."""
        L = w if vertical else h
        n = min(n, int(self.px(L) / gap))
        for k in range(1, n):
            if vertical:
                xx = x + w * k / n
                self.line(xx, y, xx, y + h, tone, wt)
            else:
                yy = y + h * k / n
                self.line(x, yy, x + w, yy, tone, wt)
        return n

    def fits(self, w, h, px=FINE * 3):
        return min(self.px(w), self.px(h)) >= px


# -- the parts ---------------------------------------------------------------------


def sram(pen, x, y, w, h, banks=None, tone="end", vertical=None):
    """An SRAM macro: banks of bitcells with their word lines, a decoder spine
    down each bank, a row of sense amplifiers along one edge."""
    pen.rect(x, y, w, h, tone, 1.0, "hair", 0.6)
    if not pen.fits(w, h, 10):
        return
    vertical = (w >= h) if vertical is None else vertical  # banks side by side
    L = w if vertical else h
    banks = banks or max(2, round(L / max(min(w, h), 0.4) * 1.6))
    banks = max(1, min(banks, int(pen.px(L) / 9)))
    m = min(w, h) * 0.06
    for b in range(banks):
        if vertical:
            bx, by, bw, bh = x + m + b * (w - m) / banks, y + m, (w - m) / banks - m, h - 2 * m
        else:
            bx, by, bw, bh = x + m, y + m + b * (h - m) / banks, w - 2 * m, (h - m) / banks - m
        if bw <= 0 or bh <= 0:
            continue
        pen.rect(bx, by, bw, bh, "recess", 0.55, "hair", 0.45)
        if vertical:
            sa = bh * 0.1  # the sense amps
            pen.rect(bx, by + bh - sa, bw, sa, "top", 0.9)
            dx = bw * 0.46
            pen.rect(bx + dx, by, bw * 0.08, bh - sa, "top", 0.9)  # the decoder
            for side in ((bx, dx), (bx + dx + bw * 0.08, bw - dx - bw * 0.08)):
                pen.lines(side[0], by, side[1], bh - sa, 40, False, "hair", 0.4)
        else:
            sa = bw * 0.1
            pen.rect(bx + bw - sa, by, sa, bh, "top", 0.9)
            dy = bh * 0.46
            pen.rect(bx, by + dy, bw - sa, bh * 0.08, "top", 0.9)
            for side in ((by, dy), (by + dy + bh * 0.08, bh - dy - bh * 0.08)):
                pen.lines(bx, side[0], bw - sa, side[1], 40, True, "hair", 0.4)


def cells(pen, x, y, w, h, rng, tone="top", depth=None):
    """Logic: the block cut into hard macros and placed standard-cell regions,
    the way a die shot reads -- tidy rectangles, cell rows at their pitch."""
    pen.rect(x, y, w, h, tone, 1.0, "hair", 0.6)
    if not pen.fits(w, h, 12):
        return
    m = min(w, h) * 0.05
    leaves = []

    def cut(x, y, w, h, d):
        if d == 0 or not pen.fits(w, h, 22):
            leaves.append((x, y, w, h))
            return
        f = rng.uniform(0.35, 0.65)
        if w >= h:
            cut(x, y, w * f, h, d - 1)
            cut(x + w * f, y, w * (1 - f), h, d - 1)
        else:
            cut(x, y, w, h * f, d - 1)
            cut(x, y + h * f, w, h * (1 - f), d - 1)

    cut(x + m, y + m, w - 2 * m, h - 2 * m, depth or 3)
    for i, (lx, ly, lw, lh) in enumerate(leaves):
        q = min(lw, lh) * 0.08
        lx, ly, lw, lh = lx + q, ly + q, lw - 2 * q, lh - 2 * q
        if rng.random() < 0.38:  # a hard macro: a small SRAM
            pen.rect(lx, ly, lw, lh, "end", 1.0, "hair", 0.45)
            pen.lines(lx, ly, lw, lh, 12, lw < lh, "hair", 0.4)
        else:  # placed cells: rows at their pitch
            pen.rect(lx, ly, lw, lh, "face", 1.0, "hair", 0.45)
            pen.lines(lx, ly, lw, lh, 30, False, "end", 0.4, gap=4.5)


def gpu_core(pen, x, y, w, h, on, rot=False):
    """A GPU core: four SIMD partitions, each a register file over its ALU
    lanes; then the texture unit and the core's tile memory and L1."""
    body = "top" if on else "recess"
    ink = "hair" if on else "faint"
    pen.rect(x, y, w, h, body, 1.0, "line", 0.85)
    if not pen.fits(w, h, 14):
        return
    m = min(w, h) * 0.07
    ix, iy, iw, ih = x + m, y + m, w - 2 * m, h - 2 * m
    top = ih * 0.64
    # the four partitions
    pw = (iw - 3 * m * 0.6) / 4
    for k in range(4):
        px_ = ix + k * (pw + m * 0.6)
        rf = top * 0.38  # the register file
        pen.rect(px_, iy, pw, rf, "end" if on else "top", 0.85, ink, 0.45)
        pen.lines(px_, iy, pw, rf, 6, False, "hair" if on else "end", 0.4)
        pen.rect(px_, iy + rf + m * 0.4, pw, top - rf - m * 0.4, "face" if on else "top", 1.0,
                 ink, 0.45)
        pen.lines(px_, iy + rf + m * 0.4, pw, top - rf - m * 0.4, 8, True,
                  "hair" if on else "end", 0.4)  # its lanes
    # the texture unit, the tile memory and L1
    by = iy + top + m * 0.7
    bh = ih - top - m * 0.7
    tw = iw * 0.52
    pen.rect(ix, by, tw, bh, "end" if on else "top", 0.85, ink, 0.45)
    if pen.fits(tw, bh, 12):
        q = min(tw, bh) * 0.12
        pen.rect(ix + q, by + q, tw * 0.45 - q, bh - 2 * q, "face" if on else "top", 1.0, ink, 0.4)
        pen.rect(ix + tw * 0.5, by + q, tw * 0.5 - q, bh - 2 * q, "face" if on else "top", 1.0,
                 ink, 0.4)
    lx = ix + tw + m * 0.6
    lw = iw - tw - m * 0.6
    pen.rect(lx, by, lw, bh, "end" if on else "top", 0.85, ink, 0.45)
    pen.lines(lx, by, lw, bh, 5, True, "hair" if on else "end", 0.4)


def p_core(pen, x, y, w, h, flip, big=True):
    """A CPU core: L1 instruction and data caches at its outer end, the front
    end, then integer and load/store units beside the vector/FP unit."""
    pen.rect(x, y, w, h, "top", 1.0, "line", 0.85)
    if not pen.fits(w, h, 14):
        return
    m = min(w, h) * 0.06
    ix, iw = x + m, w - 2 * m
    ih = h - 2 * m
    l1 = ih * (0.3 if big else 0.34)
    fe = ih * 0.14
    rest = ih - l1 - fe - 2 * m * 0.6
    if not flip:
        y_l1, y_fe, y_ex = y + m, y + m + l1 + m * 0.6, y + m + l1 + fe + 1.2 * m
    else:
        y_ex = y + m
        y_fe = y_ex + rest + m * 0.6
        y_l1 = y_fe + fe + m * 0.6
    half = (iw - m * 0.6) / 2
    sram(pen, ix, y_l1, half, l1, banks=2)
    sram(pen, ix + half + m * 0.6, y_l1, half, l1, banks=2)
    pen.rect(ix, y_fe, iw, fe, "end", 0.8, "hair", 0.45)  # decode, rename
    pen.lines(ix, y_fe, iw, fe, 6 if big else 4, True, "hair", 0.4)
    vw = iw * (0.4 if big else 0.32)
    gw = iw - vw - m * 0.6
    # integer units and load/store: a grid of blocks
    cols, rows = (3, 3) if big else (2, 2)
    cw, ch = gw / cols, rest / rows
    for i in range(cols * rows):
        cx, cy = i % cols, i // cols
        pen.rect(ix + cx * cw + m * 0.2, y_ex + cy * ch + m * 0.2, cw - m * 0.4, ch - m * 0.4,
                 "face" if (i + cy) % 2 else "end", 0.9, "hair", 0.4)
    # the vector unit: its lanes
    vx = ix + gw + m * 0.6
    pen.rect(vx, y_ex, vw, rest, "end", 0.9, "hair", 0.45)
    pen.lines(vx, y_ex, vw, rest, 4, True, "hair", 0.45)


def ne_core(pen, x, y, w, h):
    """A Neural Engine core: its MAC array, its buffer."""
    pen.rect(x, y, w, h, "top", 1.0, "line", 0.75)
    if not pen.fits(w, h, 10):
        return
    m = min(w, h) * 0.1
    aw = (w - 2 * m) * 0.72
    ah = h - 2 * m
    pen.rect(x + m, y + m, aw, ah, "end", 0.9, "hair", 0.45)
    n = max(1, min(6, int(pen.px(min(aw, ah)) / 5)))
    if n >= 2:
        s = min(aw, ah) / n
        for i in range(n):
            for j in range(n):
                pen.rect(x + m + i * aw / n + s * 0.18, y + m + j * ah / n + s * 0.18,
                         aw / n - s * 0.36, ah / n - s * 0.36, "face", 0.95)
    bx = x + m + aw + m * 0.6
    pen.rect(bx, y + m, w - (bx - x) - m, ah, "end", 0.75, "hair", 0.45)
    pen.lines(bx, y + m, w - (bx - x) - m, ah, 5, False, "hair", 0.4)


def phy(pen, x, y, w, h, chans, vertical):
    """A memory PHY: one slice per channel, its data bumps in two rows, its
    calibration block."""
    pen.rect(x, y, w, h, "top", 1.0, "line", 0.85)
    L = h if vertical else w
    chans = max(1, min(chans, int(pen.px(L) / 8)))
    for c in range(chans):
        if vertical:
            sx, sy, sw, sh = x, y + c * h / chans, w, h / chans
        else:
            sx, sy, sw, sh = x + c * w / chans, y, w / chans, h
        q = min(sw, sh) * 0.1
        pen.rect(sx + q, sy + q, sw - 2 * q, sh - 2 * q, "end", 0.8, "hair", 0.45)
        if not pen.fits(sw, sh, 9):
            continue
        # the bumps
        across = sw if vertical else sh
        along = sh if vertical else sw
        n = max(2, min(6, int(pen.px(along - 2 * q) / 4.5)))
        b = min((along - 2 * q) / n * 0.5, across * 0.12)
        for r in (0.28, 0.58):
            for k in range(n):
                t = (k + 0.5) / n
                if vertical:
                    pen.rect(sx + sw * r, sy + q + (sh - 2 * q) * t - b / 2, b, b, "recess", 0.9)
                else:
                    pen.rect(sx + q + (sw - 2 * q) * t - b / 2, sy + sh * r, b, b, "recess", 0.9)
        if vertical:
            pen.rect(sx + sw * 0.8, sy + q * 2, sw * 0.12, sh - 4 * q, "face", 1.0)
        else:
            pen.rect(sx + q * 2, sy + sh * 0.8, sw - 4 * q, sh * 0.12, "face", 1.0)


# -- the blocks ----------------------------------------------------------------------


def grid(n, w, h, aspect=1.25):
    """cols x rows for n equal cells in a w x h box: every slot filled, cells
    near the wanted aspect."""
    best = None
    for cols in range(1, n + 1):
        if n % cols:
            continue
        rows = n // cols
        a = (w / cols) / (h / rows)
        s = abs(math.log(a / aspect))
        if best is None or s < best[0]:
            best = (s, cols, rows)
    return best[1], best[2]


def gpu_block(pen, x, y, w, h, cores, lit, cols=None, band=True, rec=None):
    """The GPU: its cores in a grid, the L2 through its middle."""
    if cols:
        rows = math.ceil(cores / cols)
    else:
        cols, rows = grid(cores, w, h * (0.9 if band and cores > 4 else 1.0))
    bh = h * 0.1 if band and rows >= 2 else 0.0
    ch = (h - bh) / rows
    cw = w / cols
    mid = rows // 2
    g = min(cw, ch) * 0.06
    order = []
    for i in range(cores):
        cx, cy = i % cols, i // cols
        yy = y + cy * ch + (bh if cy >= mid and bh else 0)
        order.append((cx, cy, x + cx * cw, yy))
    # which cores are off: binned at the ends of rows, spread over the rows
    off = set()
    k = cores - lit
    if k > 0:
        cand = sorted(range(cores), key=lambda i: (-(abs(order[i][0] - (cols - 1) / 2)),
                                                    order[i][1] % 2, order[i][1]))
        off = set(cand[:k])
    for i, (cx, cy, xx, yy) in enumerate(order):
        gpu_core(pen, xx + g / 2, yy + g / 2, cw - g, ch - g, i not in off)
    if bh:
        by = y + mid * ch
        sram(pen, x + g / 2, by + bh * 0.18, w - g, bh * 0.64, banks=cols * 2)
        if rec is not None:
            rec.append((pen.P(x + g / 2, by + bh / 2), pen.P(x + w - g / 2, by + bh / 2)))


def cpu_block(pen, x, y, w, h, cores, big, l2="mid", cols=None, l2frac=0.26, rows=None):
    """A CPU cluster: its cores in rows facing the shared L2 between them.
    l2: "mid" between two rows, "top"/"bottom" beside them, "none" (Apple's
    diagram draws no L2 for this cluster)."""
    if cores <= 0:
        return
    rows = rows or (2 if cores >= 4 else 1)
    cols = cols or math.ceil(cores / rows)
    rows = math.ceil(cores / cols)
    if l2 == "mid" and rows != 2:
        l2 = "bottom"
    l2h = 0.0 if l2 == "none" else h * l2frac
    ch = (h - l2h) / rows
    g = min(w / cols, ch) * 0.05
    if rows == 2 and l2 == "mid":
        ys = [y, y + ch + l2h]
        ly = y + ch
    else:
        ys = [y + (l2h if l2 == "top" else 0) + r * ch for r in range(rows)]
        ly = y if l2 == "top" else y + rows * ch
    for i in range(cores):
        cx, cy = i % cols, i // cols
        p_core(pen, x + cx * w / cols + g / 2, ys[cy] + g / 2, w / cols - g, ch - g,
               flip=(cy == 1 and l2 == "mid") or l2 == "top", big=big)
    if l2h:
        sram(pen, x + g / 2, ly + l2h * 0.08, w - g, l2h * 0.84, banks=max(2, cols * 2))


def ne_block(pen, x, y, w, h, cores=16, cols=None):
    cols = cols or grid(cores, w, h, 1.0)[0]
    rows = math.ceil(cores / cols)
    g = min(w / cols, h / rows) * 0.04
    for i in range(cores):
        cx, cy = i % cols, i // cols
        ne_core(pen, x + cx * w / cols + g / 2, y + cy * h / rows + g / 2, w / cols - g,
                h / rows - g)


def draw_die(pen, plan, lit_total, rng, rec):
    """One die: its body, seal ring and pads, then each block of its plan."""
    W, H = plan["size"]
    pen.rect(0, 0, W, H, "face", 1.0)
    for k, tone, wt in ((0.1, "hair", 0.6), (0.22, "faint", 0.5)):
        pen.rect(k, k, W - 2 * k, H - 2 * k, None, 1.0, tone, wt)
    # the pad ring: centred along each edge, clear of the corners
    pad, pitch, e = 0.13, 0.36, 0.34
    for L, along in ((W, True), (H, False)):
        n = int((L - 2 * e - 0.5) / pitch) + 1
        s0 = (L - (n - 1) * pitch) / 2
        for i in range(n):
            t = s0 + i * pitch - pad / 2
            for c in (e - pad / 2, (H if along else W) - e - pad / 2):
                if along:
                    pen.rect(t, c, pad, pad, "end", 1.0)
                else:
                    pen.rect(c, t, pad, pad, "end", 1.0)
    m = 0.6
    CW, CH = W - 2 * m, H - 2 * m
    g = min(CW, CH) * 0.006
    for b in plan["blocks"]:
        kind, fx, fy, fw, fh = b[:5]
        o = b[5] if len(b) > 5 else {}
        x, y, w, h = m + fx * CW + g, m + fy * CH + g, fw * CW - 2 * g, fh * CH - 2 * g
        if kind == "gpu":
            gpu_block(pen, x, y, w, h, o["cores"], o.get("lit", lit_total), o.get("cols"),
                      o.get("band", True), rec)
        elif kind in ("s", "p", "e"):
            big = kind == "s" or (kind == "p" and not o.get("mid"))
            cpu_block(pen, x, y, w, h, o["cores"], big, o.get("l2", "mid"), o.get("cols"),
                      o.get("l2frac", 0.26), o.get("rows"))
        elif kind == "ne":
            ne_block(pen, x, y, w, h, o.get("cores", 16), o.get("cols"))
        elif kind == "sram":
            sram(pen, x, y, w, h, o.get("banks"), vertical=o.get("vertical"))
        elif kind == "phy":
            phy(pen, x, y, w, h, o.get("chans", 8), h > w)
        elif kind == "logic":
            cells(pen, x, y, w, h, rng)
    pen.rect(0, 0, W, H, None, 1.0, "line", 1.6)


def shadow(ops, x0, y0, x1, y1):
    """The cards' soft shadow, a little below. The die is opaque, so only the
    parts that show past it are filled."""
    span = x1 - x0
    out = []

    def box(a, b, c, d):
        if c > a and d > b:
            out.append(("fill", [(a, b), (c, b), (c, d), (a, d)], "shadow", 0.016))

    for i in range(9):
        g = span * 0.003 * i
        dy = span * (0.004 + 0.003 * i)
        a, b, c, d = x0 + span * 0.02 - g, y0 + dy, x1 - span * 0.02 + g, y1 + dy + g
        box(a, max(b, y1), c, d)  # below
        box(a, b, min(c, x0), min(d, y1))  # past the left side
        box(max(a, x1), b, c, min(d, y1))  # past the right
    return out + ops


def chip_ops(name, lit, px, plans):
    """Every die of a chip, laid side by side as packaged, in GPU-card tones;
    the accent through each GPU's L2. lit: GPU cores switched on in all."""
    spec = plans[name]
    dies = spec["dies"]  # [(plan, ox, oy, flip_x, flip_y)]
    total_w = max(ox + p["size"][0] for p, ox, oy, fx, fy in dies)
    ppu = px * 0.96 / total_w
    rec = []
    ops = []
    bodies = []
    rng = random.Random(name)
    per = [sum(b[5]["cores"] for b in p["blocks"] if b[0] == "gpu") for p, *_ in dies]
    left, togo = lit, sum(per)
    for j, (plan, ox, oy, fx, fy) in enumerate(dies):
        W, H = plan["size"]
        pen = Pen(ppu, fx, fy, ox, oy, W, H)
        mine = round(left * per[j] / togo) if togo else 0
        left, togo = left - mine, togo - per[j]
        rng = random.Random(repr(plan["size"]) + name)  # a die looks the same, rotated or not
        draw_die(pen, plan, mine, rng, rec)
        ops += pen.ops
        bodies.append((ox, oy, ox + W, oy + H))
    under = []
    for x0, y0, x1, y1 in bodies:  # each die casts its own: none under the gaps
        under = shadow(under, x0, y0, x1, y1)
    ops = under + ops
    for a, b in rec:
        ops.append(("line", a, b, "accent", 2.4))
    return ops, bodies


# -- which chip ------------------------------------------------------------------------


def parse(gpu_name: str) -> Optional[Tuple[str, Optional[int]]]:
    """``Apple M4 Pro 20-core GPU 48GB`` -> ("M4 Pro", 20); None for anything
    that is no chip drawn here."""
    m = re.match(r"Apple (M\d+(?: Pro| Max| Ultra)?)(?: (\d+)-core GPU)?", gpu_name)
    if not m or m.group(1) not in chip_plans.PLANS:
        return None
    return m.group(1), int(m.group(2)) if m.group(2) else None


def lit(chip: str, cores: Optional[int]) -> int:
    """GPU cores drawn lit: the chip's own, at most its top config."""
    top = chip_plans.gpu_max(chip)
    return min(cores or top, top)


def subtitle(chip: str, cores: Optional[int], mem_gib: Optional[float]) -> str:
    """``10-core GPU · 16 GB LPDDR5X · 120 GB/s``."""
    g = lit(chip, cores)
    mem = f"{mem_gib:.0f} GB " if mem_gib else ""
    return (f"{g}-core GPU · {mem}{chip_plans.CHIPS[chip][4]} · "
            f"{chip_plans.bandwidth(chip, g)} GB/s")


# -- tones -----------------------------------------------------------------------------


def tones(pal: Palette) -> Dict[str, Tuple[RGB, RGB]]:
    """The GPU cards' tones, fitted to any ground (see the module's notes)."""
    t = dict(gpu_image._tones(pal))
    fg, bg = gpu_image._ground(pal)
    cr = _contrast(fg, bg)
    if cr < 9:  # a soft theme: its ink sits near its ground, so lines step further out
        k = min(1.6, (9 / cr) ** 0.5)
        for name, share in (("line", 0.48), ("hair", 0.26), ("faint", 0.13)):
            c_ = mix(fg, bg, min(0.9, share * k))
            t[name] = (c_, c_)

    def hsv(c_: RGB) -> Tuple[float, float, float]:
        h, s, v = colorsys.rgb_to_hsv(*(x / 255 for x in c_))
        return h * 360, s, v

    ah, gh = hsv(t["accent"][0]), hsv(bg)
    if gh[1] > 0.3 and gh[2] > 0.25 and min(abs(ah[0] - gh[0]), 360 - abs(ah[0] - gh[0])) < 35:
        t["accent"] = (fg, fg)  # the accent on a ground of its own hue is lost
    if intro_turn.images_in_p3():
        t = {k_: (intro_turn.to_p3(v[0]), intro_turn.to_p3(v[1])) for k_, v in t.items()}
    return t


# -- the picture -----------------------------------------------------------------------


class _Space:
    """Op coordinates onto one pixel grid."""

    def __init__(self, ops: list, px: int) -> None:
        xs, ys = [], []
        for op in ops:
            for x, y in (op[1] if op[0] == "fill" else (op[1], op[2])):
                xs.append(x)
                ys.append(y)
        self.minx, self.miny = min(xs), min(ys)
        sx, sy = max(xs) - self.minx, max(ys) - self.miny
        self.padu = max(sx, sy) * _PAD
        self.sc = px / (sx + 2 * self.padu)
        self.w = round(px)
        self.h = round((sy + 2 * self.padu) * self.sc)

    def P(self, p: Pt) -> Pt:
        return ((p[0] - self.minx + self.padu) * self.sc, (p[1] - self.miny + self.padu) * self.sc)


@lru_cache(maxsize=32)
def aspect(chip: str) -> float:
    """Height over width of the drawn chip, its air included."""
    ops, _ = chip_ops(chip, chip_plans.gpu_max(chip), 400, chip_plans.PLANS)
    sp = _Space(ops, 400)
    return sp.h / sp.w


def _raster(ops: list, t: Dict[str, Tuple[RGB, RGB]], sp: _Space) -> bytearray:
    """Line weights in thousandths of the image's width: legible at any size."""
    cv = gpu_image._Canvas(sp.w, sp.h, sp.sc)
    unit = sp.w / 1000
    for op in ops:
        if op[0] == "fill":
            col = t[op[2]]
            poly = [sp.P(p) for p in op[1]]
            ys = [p[1] for p in poly]
            if max(ys) - min(ys) < 12:  # no gradient to see, and none to overshoot
                col = (col[0], col[0])
            cv.fill(poly, col, op[3])
        else:
            cv.line(sp.P(op[1]), sp.P(op[2]), t[op[3]][0], max(0.5, op[4] * unit))
    return cv.px


class Picture:
    """The chip at one size: the still, and the wash the scan carries over it."""

    def __init__(self, w: int, h: int, px: bytes, y0: int, y1: int, x0: int, x1: int,
                 ink: RGB, scan: RGB) -> None:
        self.w, self.h, self.px = w, h, px
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
        self.band = max(6, (y1 - y0) // 12)
        self.ink, self.scan = ink, scan

    @property
    def wash(self) -> bytes:
        """The scan's light, ``band + 1`` rows: the band behind the line lit
        toward the ink, brightest at the line, then the line itself. Laid over
        the still it lights what it passes; only across the dies."""
        w, band = self.w, self.band
        out = bytearray(w * (band + 1) * 4)
        n = self.x1 - self.x0
        for i in range(band):
            k = (i / band) ** 2
            if k > 0.05:
                a = (i * w + self.x0) * 4
                out[a:a + n * 4] = bytes((*self.ink, round(255 * 0.22 * k))) * n
        a = (band * w + self.x0) * 4
        out[a:a + n * 4] = bytes((*self.scan, 255)) * n
        return bytes(out)

    def at(self, t: float) -> int:
        """The scan line's row at ``t`` seconds: in fast, settling as its
        light leaves the dies' foot (ease-out cubic), so the motion fills the
        whole DUR and the still follows at once."""
        u = min(1.0, max(0.0, t / DUR))
        return int(self.y0 + (self.y1 + self.band - self.y0) * (1 - (1 - u) ** 3))


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "poros", "chip")


@lru_cache(maxsize=8)
def picture(chip: str, cores: Optional[int], pal: Palette, px: int) -> Picture:
    """The chip ``px`` wide in ``pal``'s tones: drawn once, then from the cache."""
    t = tones(pal)
    fg, bg = gpu_image._ground(pal)
    p3 = intro_turn.to_p3 if intro_turn.images_in_p3() else (lambda c_: c_)
    ink, scan = p3(fg), p3(mix(fg, bg, 0.8))
    on = lit(chip, cores)
    key = hashlib.sha1(repr((_VERSION, chip, on, px, sorted(t.items()))).encode()).hexdigest()
    path = os.path.join(_cache_dir(), key + ".rgba")
    try:
        with open(path, "rb") as fh:
            head, body = fh.read().split(b"\n", 1)
        w, h, y0, y1, x0, x1 = (int(v) for v in head.split())
        data = zlib.decompress(body)
        if len(data) == w * h * 4:
            return Picture(w, h, data, y0, y1, x0, x1, ink, scan)
    except (OSError, ValueError, zlib.error):
        pass
    ops, bodies = chip_ops(chip, on, px, chip_plans.PLANS)
    sp = _Space(ops, px)
    data = bytes(_raster(ops, t, sp))
    corners = [sp.P((b[0], b[1])) for b in bodies] + [sp.P((b[2], b[3])) for b in bodies]
    x0 = max(0, int(min(p[0] for p in corners)))
    x1 = min(sp.w, int(math.ceil(max(p[0] for p in corners))))
    y0 = max(0, int(min(p[1] for p in corners)))
    y1 = min(sp.h, int(math.ceil(max(p[1] for p in corners))))
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(f"{sp.w} {sp.h} {y0} {y1} {x0} {x1}\n".encode() + zlib.compress(data, 1))
        os.replace(tmp, path)
    except OSError:
        pass
    return Picture(sp.w, sp.h, data, y0, y1, x0, x1, ink, scan)


# -- out to the terminal ---------------------------------------------------------------


def size(chip: str, cols: int, cell: Tuple[int, int]) -> Tuple[int, int]:
    """Pixels across, and rows tall, for the chip ``cols`` cells wide."""
    cw, ch = cell
    return cols * cw, max(1, round(cols * cw * aspect(chip) / ch))


def kitty(img: bytes, w: int, h: int, cols: int, ident: int, z: int = 0) -> str:
    """Raw RGBA, zlib'd, ``cols`` wide at the cursor; the cursor stays put.
    Sent again with the same ``ident`` it replaces the one on screen."""
    return _send(img, w, h, ident, f"a=T,c={cols},z={z},C=1,")


def _send(img: bytes, w: int, h: int, ident: int, how: str = "a=t,") -> str:
    """The image's pixels under ``ident``; with ``how`` a=t, kept and not shown."""
    import base64

    data = base64.standard_b64encode(zlib.compress(img, 1)).decode()
    out = []
    for i in range(0, len(data), 4096):
        more = 1 if i + 4096 < len(data) else 0
        head = (f"{how}f=32,s={w},v={h},o=z,i={ident},q=2,m={more}" if i == 0
                else f"m={more}")
        out.append(f"\x1b_G{head};{data[i:i + 4096]}\x1b\\")
    return "".join(out)


def _up(n: int, seg: str) -> str:
    """``seg`` written ``n`` rows up from the cursor, which then goes back."""
    climb = f"\x1b[{n}A" if n > 0 else ""  # CUU 0 would still go up one
    return f"\x1b7{climb}{seg}\x1b8"


def entrance(pic: Picture, cols: int, rows: int, cell: Tuple[int, int],
             ids: Tuple[int, int, int], t: float, send: bool = False) -> str:
    """One frame of the scan, written on the chip's last row (so a screen that
    scrolls to show it doesn't move it). ``ids``: the still's and the two the
    scan draws with. With ``send`` the chip and its wash go up first, kept and
    not shown; every frame then only moves crops of them, so motion costs a
    few bytes and nothing is sent again when it lands. The still (``placed``)
    takes its place in one frame, the same pixels, and takes it off."""
    still, chip, wash = ids
    # names the frame for the session as the still it ends in (tui.image_key):
    # nothing to take off, so nothing is cleared between them
    out = [f"\x1b_Ga=d,d=i,i={still},q=2\x1b\\"]
    if send:
        out += [_send(pic.px, pic.w, pic.h, chip), _send(pic.wash, pic.w, pic.band + 1, wash)]
    w, h, band = pic.w, pic.h, pic.band
    place = "\x1b_Ga=p,i={i},p=1,x=0,y={y},w={w},h={n},c={c},Y={dy},z={z},C=1,q=2\x1b\\"
    if t >= DUR:
        out.append(_up(rows - 1, place.format(i=chip, y=0, w=w, n=h, c=cols, dy=0, z=0)))
        out.append(f"\x1b_Ga=d,d=I,i={wash},q=2\x1b\\")
        return "".join(out)
    y = pic.at(t)
    top = min(y, h)
    if top > 0:
        out.append(_up(rows - 1, place.format(i=chip, y=0, w=w, n=top, c=cols, dy=0, z=0)))
    else:
        out.append(f"\x1b_Ga=d,d=i,i={chip},q=2\x1b\\")
    ja, jb = max(0, pic.y0 - (y - band)), min(band + 1, pic.y1 - (y - band))
    if jb > ja:
        cw, ch = cell
        yd = (y - band + ja) * cols * cw / w  # where its first row lands, in pixels
        r = min(rows - 1, int(yd // ch))
        seg = place.format(i=wash, y=ja, w=w, n=jb - ja, c=cols, dy=int(yd - r * ch), z=2)
        out.append(_up(rows - 1 - r, seg))
    else:
        out.append(f"\x1b_Ga=d,d=i,i={wash},q=2\x1b\\")
    return "".join(out)


def placed(pic: Picture, cols: int, rows: int, proto: str, ident: int,
           stream=None) -> Tuple[str, int]:
    """The still ready to lay over a block of empty cells: the escape and the
    row of the block it goes on (gpu_image.placed's rule)."""
    if proto == "kitty":
        image = kitty(pic.px, pic.w, pic.h, cols, ident, z=3)
    elif proto == "iterm":
        image = gpu_image._iterm(gpu_image._png(pic.w, pic.h, bytearray(pic.px)), cols)
    else:
        ground = gpu_image._ground(_pal(stream))[1]
        image = gpu_image._sixel(pic.w, pic.h, bytearray(pic.px), ground).decode()
    return _up(rows - 1, image), rows - 1


def _pal(stream=None) -> Palette:
    import sys

    from poros.cli.ui.style import resolve

    return resolve(stream or sys.stdout)
