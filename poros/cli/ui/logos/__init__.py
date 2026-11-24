"""The marks /models draws: each lab's logo, and Apple's and NVIDIA's for
the two paths a model trains on (a Mac, through Metal; Linux, through CUDA).

The SVGs are LobeHub's monochrome icons (MIT, see LICENSE here), one path
each on a 24 by 24 box. They are drawn in Poros' own inks, never the brands'
colors, as an image where the terminal shows one (kitty's protocol). A
terminal without images (Terminal.app) names each lab in words and shows
the Mac path as Apple's own glyph. Pure Python: the paths are flattened and
filled even-odd.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from importlib import resources
from typing import Dict, List, Sequence, Tuple

from poros.cli.ui.text import MARK  # a mark's placeholder in a line, drawn by escape()

Pt = Tuple[float, float]

# the lab behind each Hugging Face org, and the file of its mark
LABS: Dict[str, Tuple[str, str]] = {
    "Qwen": ("Qwen", "qwen"),
    "google": ("Gemma", "gemma"),
    "deepseek-ai": ("DeepSeek", "deepseek"),
    "allenai": ("Ai2", "ai2"),
    "meta-llama": ("Llama", "meta"),
    "mistralai": ("Mistral", "mistral"),
    "microsoft": ("Phi", "microsoft"),
}
PATHS = ("nvidia", "apple")  # Linux / CUDA, then Mac / Metal


def lab(model_id: str) -> Tuple[str, str]:
    """``(name, mark)`` for a model id's org; an unknown org is its own name."""
    org = model_id.split("/", 1)[0]
    return LABS.get(org, (org, ""))


_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TOK = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@lru_cache(maxsize=None)
def path(mark: str) -> str:
    svg = resources.files(__name__).joinpath(f"{mark}.svg").read_text()
    return " ".join(re.findall(r'<path[^>]*\sd="([^"]+)"', svg))


def subpaths(d: str, steps: int = 12) -> List[List[Pt]]:
    """The path flattened to closed polygons."""
    polys: List[List[Pt]] = []
    cur: List[Pt] = []
    x = y = sx = sy = 0.0
    last_c = last_q = None
    cmd = ""
    chunks = re.findall(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)", d)
    for cmd, args in chunks:
        if cmd in "Aa":
            nums = _arc_args(args)
        else:
            nums = [float(v) for v in _NUM.findall(args)]
        rel = cmd.islower()
        C = cmd.upper()
        if C == "Z":
            if cur:
                polys.append(cur)
            cur, x, y = [], sx, sy
            last_c = last_q = None
            continue
        size = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}[C]
        first = True
        for i in range(0, len(nums) - size + 1, size):
            a = nums[i:i + size]
            ox, oy = (x, y) if rel else (0.0, 0.0)
            if C == "M":
                if first:
                    if cur:
                        polys.append(cur)
                    x, y = a[0] + ox, a[1] + oy
                    sx, sy, cur = x, y, [(x, y)]
                else:  # further pairs after a move are lines
                    x, y = a[0] + ox, a[1] + oy
                    cur.append((x, y))
                last_c = last_q = None
            elif C == "L":
                x, y = a[0] + ox, a[1] + oy
                cur.append((x, y))
                last_c = last_q = None
            elif C == "H":
                x = a[0] + (x if rel else 0.0)
                cur.append((x, y))
                last_c = last_q = None
            elif C == "V":
                y = a[0] + (y if rel else 0.0)
                cur.append((x, y))
                last_c = last_q = None
            elif C in "CS":
                if C == "C":
                    c1 = (a[0] + ox, a[1] + oy)
                    c2, e = (a[2] + ox, a[3] + oy), (a[4] + ox, a[5] + oy)
                else:
                    c1 = (2 * x - last_c[0], 2 * y - last_c[1]) if last_c else (x, y)
                    c2, e = (a[0] + ox, a[1] + oy), (a[2] + ox, a[3] + oy)
                p0 = (x, y)
                for k in range(1, steps + 1):
                    t = k / steps
                    u = 1 - t
                    w0, w1, w2, w3 = u ** 3, 3 * u * u * t, 3 * u * t * t, t ** 3
                    cur.append(tuple(w0 * p0[i] + w1 * c1[i] + w2 * c2[i] + w3 * e[i]
                                     for i in (0, 1)))
                x, y = e
                last_c, last_q = c2, None
            elif C in "QT":
                if C == "Q":
                    q, e = (a[0] + ox, a[1] + oy), (a[2] + ox, a[3] + oy)
                else:
                    q = (2 * x - last_q[0], 2 * y - last_q[1]) if last_q else (x, y)
                    e = (a[0] + ox, a[1] + oy)
                p0 = (x, y)
                for k in range(1, steps + 1):
                    t = k / steps
                    u = 1 - t
                    cur.append((u * u * p0[0] + 2 * u * t * q[0] + t * t * e[0],
                                u * u * p0[1] + 2 * u * t * q[1] + t * t * e[1]))
                x, y = e
                last_q, last_c = q, None
            elif C == "A":
                e = (a[5] + ox, a[6] + oy)
                cur += _arc((x, y), a[0], a[1], a[2], a[3] != 0, a[4] != 0, e, steps)
                x, y = e
                last_c = last_q = None
            first = False
    if cur:
        polys.append(cur)
    return [p for p in polys if len(p) > 2]


def _arc_args(args: str) -> List[float]:
    """An arc's numbers: rx ry rotation then two one-digit flags, which SVG
    lets run together ("01" is large=0, sweep=1), then x y."""
    out: List[float] = []
    s = args.strip()
    i = 0
    while i < len(s):
        if s[i] in " ,\t\n":
            i += 1
            continue
        k = len(out) % 7
        if k in (3, 4):
            out.append(float(s[i]))
            i += 1
            continue
        m = _NUM.match(s, i)
        if not m:
            i += 1
            continue
        out.append(float(m.group()))
        i = m.end()
    return out


def _arc(p0: Pt, rx: float, ry: float, rot: float, large: bool, sweep: bool, p1: Pt,
         steps: int) -> List[Pt]:
    """An SVG elliptical arc as points (the endpoint parameterisation)."""
    if rx == 0 or ry == 0:
        return [p1]
    phi = math.radians(rot)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1, y1 = cp * dx + sp * dy, -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    co = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        co = -co
    cx1, cy1 = co * rx * y1 / ry, -co * ry * x1 / rx
    cx = cp * cx1 - sp * cy1 + (p0[0] + p1[0]) / 2
    cy = sp * cx1 + cp * cy1 + (p0[1] + p1[1]) / 2

    def ang(ux: float, uy: float, vx: float, vy: float) -> float:
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = ang(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = ang((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(2, int(steps * abs(dt) / (math.pi / 2)))
    pts = []
    for k in range(1, n + 1):
        t = t1 + dt * k / n
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        pts.append((cp * ex - sp * ey + cx, sp * ex + cp * ey + cy))
    return pts


@lru_cache(maxsize=None)
def coverage(mark: str, w: int, h: int, sub: int = 4) -> Tuple[Tuple[float, ...], ...]:
    """The mark filled even-odd on a ``w`` by ``h`` grid, centred and kept
    square: each cell's share covered, 0..1 (``sub`` by ``sub`` samples)."""
    polys = subpaths(path(mark))
    edges = [(a, b) for p in polys for a, b in zip(p, p[1:] + p[:1]) if a[1] != b[1]]
    s = min(w, h) / 24.0
    ox, oy = (w - 24 * s) / 2, (h - 24 * s) / 2
    rows = []
    for y in range(h):
        acc = [0] * w
        for j in range(sub):
            yc = ((y + (j + 0.5) / sub) - oy) / s
            xs = sorted(xa + (yc - ya) * (xb - xa) / (yb - ya)
                        for (xa, ya), (xb, yb) in edges if (ya <= yc < yb) or (yb <= yc < ya))
            for a, b in zip(xs[::2], xs[1::2]):
                for x in range(w):
                    for i in range(sub):
                        xc = ((x + (i + 0.5) / sub) - ox) / s
                        if a <= xc < b:
                            acc[x] += 1
        rows.append(tuple(v / (sub * sub) for v in acc))
    return tuple(rows)


def names() -> Sequence[str]:
    files = resources.files(__name__).iterdir()
    return sorted(p.name[:-4] for p in files if p.name.endswith(".svg"))


def ink(pal, st: str) -> Tuple[int, int, int]:
    """The RGB a style token draws in, for a mark drawn as an image: the
    terminal's own ink toward its ground, as the GPU card is drawn, so a
    mark follows the terminal light or dark."""
    from poros.cli.ui.gpu_image import _ground
    from poros.cli.ui.style import ACCENT, TIERS, mix

    if st == "a":
        return pal._legible(ACCENT)
    fg, bg = _ground(pal)
    return mix(fg, bg, TIERS.get(st, TIERS["tx"]))


def protocol(stream=None):
    """How marks are drawn here: as the GPU card is (kitty, iTerm2, sixel),
    or None (Terminal.app, tmux): then in words."""
    from poros.cli.ui import gpu_image

    return gpu_image.protocol(stream)


def images(stream=None) -> bool:
    return protocol(stream) is not None


_IDS: Dict[Tuple[str, int], int] = {}


def cells(mark: str, st: str, *, image: bool, apple_glyph: bool = False, slot: int = 0):
    """``mark`` two cells wide and one row tall. Where the terminal shows
    images: two blank cells and, after them, a placeholder that ``escape``
    turns into the image when the line is drawn, in the inks of that moment
    (the terminal going light or dark redraws it). Else Apple's own glyph
    for the Mac (a Mac's fonts draw U+F8FF as the Apple logo), else blank.
    ``slot`` tells a mark's places apart: one image each."""
    if image:
        return [("  ", None), (f"{MARK}{mark}:{st}:{slot}", "raw")]
    if mark == "apple" and apple_glyph:
        return [("\uf8ff ", st)]
    return [("  ", None)]


def escape(text: str, pal) -> str:
    """The image a placeholder stands for, drawn over the two cells before
    the cursor and the cursor put back after them; nothing in a plain
    palette or a terminal without images."""
    if pal.depth in ("none", "plain"):
        return ""
    proto = protocol()
    if proto is None:
        return ""
    mark, st, slot = text[len(MARK):].split(":")
    from poros.cli.ui import gpu_image

    cw, ch = gpu_image._cell()
    color = ink(pal, st)
    if proto == "kitty":
        ident = _IDS.setdefault((mark, int(slot)), gpu_image.new_id())  # sent again: replaced
        px = max(16, min(2 * cw, ch) * 2)  # twice the cell: sharp on retina
        image = _kitty(_png_cached(mark, px, px, color), ident)
    elif proto == "iterm":
        image = gpu_image._iterm(_png_cached(mark, 2 * cw * 2, ch * 2, color), 2)
    else:
        image = _sixel_cached(mark, 2 * cw, ch, color, gpu_image._ground(pal)[1]).decode()
    return f"\x1b[2D\x1b7{image}\x1b8\x1b[2C"


def _kitty(png_: bytes, ident: int) -> str:
    """kitty: send ``png_`` and place it two cells wide at the cursor, the
    cursor kept. Sent again under its id, the image and its place are
    replaced (a new ink), so each place a mark sits has its own id."""
    import base64

    data = base64.standard_b64encode(png_).decode()
    parts = [data[i:i + 4096] for i in range(0, len(data), 4096)]
    out = []
    for i, part in enumerate(parts):
        more = 1 if i < len(parts) - 1 else 0
        head = f"a=T,f=100,t=d,c=2,r=1,C=1,q=2,i={ident},m={more}" if i == 0 else f"m={more}"
        out.append(f"\x1b_G{head};{part}\x1b\\")
    return "".join(out)


def _rgba(mark: str, w: int, h: int, color: Tuple[int, int, int]) -> bytearray:
    cov = coverage(mark, w, h, sub=4)
    rgba = bytearray(w * h * 4)
    for y, row in enumerate(cov):
        for x, a in enumerate(row):
            if a:
                i = (y * w + x) * 4
                rgba[i:i + 4] = bytes((*color, round(a * 255)))
    return rgba


@lru_cache(maxsize=None)
def _png_cached(mark: str, w: int, h: int, color: Tuple[int, int, int]) -> bytes:
    from poros.cli.ui.gpu_image import _png

    return _png(w, h, _rgba(mark, w, h, color))


@lru_cache(maxsize=None)
def _sixel_cached(mark: str, w: int, h: int, color, ground) -> bytes:
    from poros.cli.ui.gpu_image import _sixel

    return _sixel(w, h, _rgba(mark, w, h, color), ground)


def png(mark: str, px: int, color: Tuple[int, int, int]) -> bytes:
    """The mark as a ``px`` square PNG in one ink on a clear ground."""
    return _png_cached(mark, px, px, color)
