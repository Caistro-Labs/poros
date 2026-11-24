"""The /machine card as a real image, for terminals that can show one.

Three ways to show it, asked of the terminal rather than guessed: the kitty
graphics protocol (kitty, Ghostty, Konsole), iTerm2's inline images (iTerm2,
WezTerm, Tabby, mintty, Rio, VS Code) and sixel (Windows Terminal, foot,
xterm, Black Box, VS Code). The card is drawn
here in its own isometric view, after the hardware drawings on
caistrolabs.com: a long, low shroud seen from above and to the side, three
fans filling its face, thin grey lines on soft grey faces, vents on the
bracket end, and the accent only on a stripe along the top and on the hubs.
A passive datacenter card has fins instead of fans. Rasterized in pure
Python (antialiased edges and lines), encoded as a PNG with zlib, and placed
over a block of empty cells without moving the cursor, so the text beside it
lines up. Anything else (Apple's Terminal, tmux, a pipe), or
``POROS_IMAGES=0``, gets the braille drawing (gpu_art). Stdlib only.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import struct
import sys
import threading
import zlib
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from poros.cli.ui import gpu_art
from poros.cli.ui.style import ACCENT, Palette, mix

COLS = 46  # the biggest card's width, in cells
MM = 2.6  # pixels a millimetre: the biggest card about 940 wide, sharp on retina

RGB = Tuple[int, int, int]
Pt = Tuple[float, float]


PROTOCOLS = ("kitty", "iterm", "sixel")
_ids = iter(range(1, 1 << 30))
_base = (os.getpid() % 1000) << 20  # another poros in another tab picks other ids


def new_id() -> int:
    """A kitty image id of its own: a later card never replaces an earlier one."""
    return _base + next(_ids) % (1 << 20)


def forget(proto: Optional[str], ident: int, stream=None) -> None:
    """Take a kitty image off the screen: text written over it does not."""
    if proto == "kitty":
        stream = stream or sys.stdout
        stream.write(f"\x1b_Ga=d,d=I,i={ident},q=2\x1b\\")
        stream.flush()


_ITERM = ("iterm.app", "wezterm", "tabby", "mintty", "rio", "vscode")


def protocol(stream=None) -> Optional[str]:
    """How this terminal shows images: "kitty", "iterm", "sixel", or None.
    ``POROS_IMAGES`` pins it (0 for none, 1 for kitty, or a name)."""
    stream = stream or sys.stdout
    pin = os.environ.get("POROS_IMAGES", "")
    if pin == "0" or "NO_COLOR" in os.environ:
        return None
    if pin in PROTOCOLS:
        return pin
    if pin == "1":
        return "kitty"
    if not getattr(stream, "isatty", lambda: False)():
        return None
    env = os.environ
    term, prog = env.get("TERM", ""), env.get("TERM_PROGRAM", "").lower()
    if env.get("TMUX") or term.startswith(("screen", "tmux")) or prog == "apple_terminal":
        return None  # tmux would need passthrough; Apple's Terminal draws none
    from poros.cli import theme

    g = theme.terminal_graphics()
    if g.kitty or term in ("xterm-kitty", "xterm-ghostty") or prog == "ghostty" or (
            "KITTY_WINDOW_ID" in env):
        return "kitty"
    if prog in _ITERM or env.get("LC_TERMINAL", "").lower() == "iterm2":
        return "iterm"
    return "sixel" if g.sixel else None


def supported(stream=None) -> bool:
    """True where the terminal can show the card as an image."""
    return protocol(stream) is not None


def _cell(stream=None) -> Tuple[int, int]:
    """A cell's width and height in pixels: as the terminal says (CSI 16 t,
    else the window size), else 10 by 20."""
    from poros.cli import theme

    cell = theme.terminal_graphics().cell
    if cell and cell[0] and cell[1]:
        return cell
    try:
        import fcntl
        import termios

        fd = (stream or sys.stdout).fileno()
        r, c, xp, yp = struct.unpack("HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8))
        if r and c and xp and yp:
            return xp // c, yp // r
    except Exception:  # noqa: BLE001 -- not a terminal, or it won't say
        pass
    return 10, 20


def cols(name: str, maker: str = "", biggest: int = COLS) -> int:
    """How many cells wide the card is drawn: smaller cards, fewer."""
    return max(20, round(biggest * gpu_art.share(gpu_art.form(name, maker))))


def rows(name: str, maker: str = "", stream=None, proto: str = "kitty",
         biggest: int = COLS) -> int:
    """How many rows the image takes, from the cell's shape. iTerm2 and sixel
    get one more: the image is drawn from the row under it (see placed)."""
    cw, ch = _cell(stream)
    _, _, sx, sy = gpu_art.bounds(gpu_art.form(name, maker))
    shape = (sy + 2 * _PAD) / (sx + 2 * _PAD)  # the image, its air included
    tall = max(4, math.ceil(cols(name, maker, biggest) * shape / (ch / cw)))
    return tall if proto == "kitty" else tall + 1


def _ground(pal: Palette) -> Tuple[RGB, RGB]:
    """The ink and the ground the card is drawn with: the terminal's own, or
    a plain light or dark pair when it won't say."""
    if pal.adaptive or pal.depth in ("none", "plain"):
        return ((236, 236, 236), (22, 22, 22)) if pal.dark else ((22, 22, 22), (252, 252, 252))
    return pal.fg, pal.bg


def _tones(pal: Palette) -> Dict[str, Tuple[RGB, RGB]]:
    """Each tone as its colour at the top and at the bottom of a shape: lit
    from above, so faces fall off toward the ground."""
    fg, bg = _ground(pal)
    dark = sum(bg) < sum(fg)

    def g(share: float, fall: float = 0.0) -> Tuple[RGB, RGB]:
        # lit: nearer the ground in light mode, nearer the ink in dark mode
        hi = share * (1 + fall) if dark else share * (1 - fall)
        lo = share * (1 - fall * 0.6) if dark else share * (1 + fall)
        return mix(fg, bg, max(0.0, hi)), mix(fg, bg, min(1.0, lo))

    t = {"face": g(0.035, 0.7), "top": g(0.065, 0.3), "end": g(0.13, 0.35),
         "shroud": g(0.055, 0.5), "recess": g(0.14, 0.25), "fin": g(0.05, 0.4),
         "blade": g(0.018), "blade2": g(0.055), "hub": g(0.06), "cap": g(0.02),
         "metal": g(0.10, 0.45), "port": g(0.42), "slot": g(0.30),
         "line": g(0.48), "hair": g(0.26), "faint": g(0.13)}
    shadow = (0, 0, 0) if dark else fg
    t["shadow"] = (shadow, shadow)
    acc = ACCENT if pal.adaptive or pal.depth in ("none", "plain") else pal._legible(ACCENT)
    t["accent"] = (acc, acc)  # on an orange ground, the accent gives way to the ink
    return t


def _png(w: int, h: int, rgba: bytearray, level: int = 6) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4:(y + 1) * w * 4]) for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level)) + chunk(b"IEND", b""))


class _Canvas:
    """RGBA pixels on a transparent ground; antialiased polygons and lines."""

    def __init__(self, w: int, h: int, scale: float) -> None:
        self.w, self.h, self.sc = w, h, scale
        self.px = bytearray(w * h * 4)

    def _blend(self, x: int, y: int, c: RGB, a: float) -> None:
        px, i = self.px, (y * self.w + x) * 4
        dst = px[i + 3]
        if dst == 0 or a >= 0.998:  # nothing under it, or nothing shows through
            px[i], px[i + 1], px[i + 2] = c
            px[i + 3] = 255 if a >= 0.998 else max(px[i + 3], round(a * 255))
            return
        if dst == 255:  # the common case: over an opaque pixel
            b = 1 - a
            px[i] = round(c[0] * a + px[i] * b)
            px[i + 1] = round(c[1] * a + px[i + 1] * b)
            px[i + 2] = round(c[2] * a + px[i + 2] * b)
            return
        da = dst / 255
        oa = a + da * (1 - a)
        for k in range(3):
            px[i + k] = round((c[k] * a + px[i + k] * da * (1 - a)) / oa)
        px[i + 3] = round(oa * 255)

    def fill(self, poly: List[Pt], col: Tuple[RGB, RGB], alpha: float = 1.0) -> None:
        """Four samples a row, exact coverage along it; the colour runs from
        the shape's top to its bottom. Fully covered runs of an opaque shape
        are written whole; only the edges are blended."""
        w, sub = self.w, 4
        xs_all = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, x1 = max(0, int(min(xs_all))), min(w, int(max(xs_all)) + 2)
        y_top, span = min(ys), max(1.0, max(ys) - min(ys))
        edges = list(zip(poly, poly[1:] + poly[:1]))
        px = self.px
        for y in range(max(0, int(min(ys))), min(self.h, int(max(ys)) + 1)):
            cov = [0.0] * (x1 - x0 + 1)
            run = [0.0] * (x1 - x0 + 2)  # whole cells covered, as a difference array
            for s in range(sub):
                yc = y + (s + 0.5) / sub
                xs = sorted(xa + (yc - ya) * (xb - xa) / (yb - ya)
                            for (xa, ya), (xb, yb) in edges if (ya <= yc < yb) or (yb <= yc < ya))
                for a, b in zip(xs[::2], xs[1::2]):
                    a, b = max(float(x0), a), min(float(x1), b)
                    if b <= a:
                        continue
                    ia, ib = int(a), int(b)
                    if ia == ib:
                        cov[ia - x0] += (b - a) / sub
                        continue
                    cov[ia - x0] += (ia + 1 - a) / sub
                    run[ia + 1 - x0] += 1 / sub
                    run[ib - x0] -= 1 / sub
                    if ib < x1:
                        cov[ib - x0] += (b - ib) / sub
            here = col[0] if col[0] == col[1] else mix(col[1], col[0], (y - y_top) / span)
            solid = bytes((*here, 255))
            acc, start = 0.0, -1
            for i in range(len(cov)):
                acc += run[i]
                c = cov[i] + acc
                if alpha >= 1.0 and c >= 0.999:
                    if start < 0:
                        start = i
                    continue
                if start >= 0:  # a run of whole cells: written at once
                    a = (y * w + x0 + start) * 4
                    px[a:a + (i - start) * 4] = solid * (i - start)
                    start = -1
                if c > 0.001:
                    self._blend(x0 + i, y, here, min(1.0, c) * alpha)
            if start >= 0:
                a = (y * w + x0 + start) * 4
                px[a:a + (len(cov) - start) * 4] = solid * (len(cov) - start)

    def line(self, a: Pt, b: Pt, col: RGB, wt: float = 1.0) -> None:
        """Antialiased by distance to the segment, walked along its longer
        axis: only the few pixels across it are looked at."""
        (x1, y1), (x2, y2) = a, b
        half = max(0.55, wt * 0.55)
        dx, dy = x2 - x1, y2 - y1
        ll = dx * dx + dy * dy or 1e-9
        pad = half + 1.5
        steep = abs(dy) > abs(dx)
        lo, hi = (min(y1, y2), max(y1, y2)) if steep else (min(x1, x2), max(x1, x2))
        limit, other = (self.h, self.w) if steep else (self.w, self.h)
        for u in range(max(0, int(lo - pad)), min(limit, int(hi + pad) + 1)):
            q = u + 0.5
            if steep:  # where the line crosses this row
                c = x1 + (q - y1) * dx / dy if dy else x1
            else:
                c = y1 + (q - x1) * dy / dx if dx else y1
            for v in range(max(0, int(c - pad - 1)), min(other, int(c + pad + 1) + 1)):
                qx, qy = (v + 0.5, q) if steep else (q, v + 0.5)
                t = max(0.0, min(1.0, ((qx - x1) * dx + (qy - y1) * dy) / ll))
                cv = half + 0.5 - math.hypot(qx - (x1 + t * dx), qy - (y1 + t * dy))
                if cv > 0:
                    if steep:
                        self._blend(v, u, col, min(1.0, cv))
                    else:
                        self._blend(u, v, col, min(1.0, cv))

    def loop(self, pts: List[Pt], col: RGB, wt: float = 1.0) -> None:
        for a, b in zip(pts, pts[1:] + pts[:1]):
            self.line(a, b, col, wt)


_LOCK = threading.Lock()
_VERSION = 5  # bump when the drawing changes: the disk cache keys on it


def render(name: str, pal: Palette, maker: str = "", proto: str = "kitty",
           width: int = 0) -> bytes:
    """The card as PNG bytes on a transparent ground, or for sixel as a
    sixel image ``width`` pixels wide on the terminal's own ground. Drawn
    once per card and look, then kept in memory and in ~/.cache/poros/gpu."""
    f = gpu_art.form(name, maker)
    tones = _tones(pal)
    label = gpu_art.brand(name, maker)
    sixel = (width, _ground(pal)[1]) if proto == "sixel" else None
    key = hashlib.sha1(repr((_VERSION, f, label, sorted(tones.items()), sixel)).encode()
                       ).hexdigest()
    with _LOCK:
        return _cached(key, f, label, tuple(sorted(tones.items())), sixel)


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "poros", "gpu")


@lru_cache(maxsize=16)
def _cached(key: str, f, label: str, tones, sixel=None) -> bytes:
    path = os.path.join(_cache_dir(), key + (".six" if sixel else ".png"))
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data.startswith(b"\x1bP" if sixel else b"\x89PNG"):
            return data
    except OSError:
        pass
    if sixel:
        width, ground = sixel
        sx = gpu_art.bounds(f)[2]
        cv = _draw(f, label, dict(tones), width / (sx + 2 * _PAD))
        data = _sixel(cv.w, cv.h, cv.px, ground)
    else:
        cv = _draw(f, label, dict(tones), MM)
        data = _png(cv.w, cv.h, cv.px)
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:
        pass
    return data


_PAD = 2.0  # mm of air around the card


def _draw(f, label: str, t: Dict[str, Tuple[RGB, RGB]], mm: float) -> "_Canvas":
    minx, miny, sx, sy = gpu_art.bounds(f)
    pad = _PAD
    cv = _Canvas(round((sx + 2 * pad) * mm), round((sy + 2 * pad) * mm), mm)

    def P(p: Tuple[float, float]) -> Tuple[float, float]:
        return ((p[0] - minx + pad) * mm, (p[1] - miny + pad) * mm)

    for op in gpu_art.drawing(f, label=label):
        if op[0] == "fill":
            cv.fill([P(p) for p in op[1]], t[op[2]], op[3])
        else:
            cv.line(P(op[1]), P(op[2]), t[op[3]][0], op[4])
    return cv


def _sixel(w: int, h: int, rgba: bytearray, ground: RGB) -> bytes:
    """RGBA as sixel: laid over the terminal's ground, at most 255 colours
    (the most used, every other pixel to its nearest), and the empty pixels
    left unpainted so the terminal's own ground shows through."""
    import re

    keys = [-1] * (w * h)  # 15-bit colour per pixel, -1 where nothing is drawn
    count: Dict[int, int] = {}
    total: Dict[int, List[int]] = {}
    for i in range(w * h):
        r, g, b, a = rgba[i * 4:i * 4 + 4]
        if a < 8:
            continue
        s = a / 255
        r, g, b = (round(r * s + ground[0] * (1 - s)), round(g * s + ground[1] * (1 - s)),
                   round(b * s + ground[2] * (1 - s)))
        k = (r >> 3) << 10 | (g >> 3) << 5 | (b >> 3)
        keys[i] = k
        count[k] = count.get(k, 0) + 1
        acc = total.setdefault(k, [0, 0, 0])
        acc[0] += r
        acc[1] += g
        acc[2] += b
    top = sorted(count, key=count.__getitem__, reverse=True)[:255]
    colours = [tuple(round(v / count[k]) for v in total[k]) for k in top]
    index = {k: n for n, k in enumerate(top)}
    for k in count:
        if k not in index:  # to the nearest colour kept
            c = tuple(v / count[k] for v in total[k])
            index[k] = min(range(len(colours)),
                           key=lambda n: sum((colours[n][j] - c[j]) ** 2 for j in range(3)))
    out = [f'\x1bP0;1;0q"1;1;{w};{h}']
    out += [f"#{n};2;{round(c[0] / 2.55)};{round(c[1] / 2.55)};{round(c[2] / 2.55)}"
            for n, c in enumerate(colours)]
    table = bytes(range(63, 127)) + bytes(192)

    def rle(m: "re.Match[bytes]") -> bytes:
        return b"!%d%c" % (len(m.group(0)), m.group(0)[0])

    body = []
    for y0 in range(0, h, 6):
        bands: Dict[int, bytearray] = {}
        for dy in range(min(6, h - y0)):
            row = (y0 + dy) * w
            bit = 1 << dy
            for x in range(w):
                k = keys[row + x]
                if k >= 0:
                    n = index[k]
                    band = bands.get(n)
                    if band is None:
                        band = bands[n] = bytearray(w)
                    band[x] |= bit
        runs = []
        for n, band in bands.items():
            chars = bytes(band).translate(table).rstrip(b"?")
            runs.append(b"#%d" % n + re.sub(rb"(.)\1{3,}", rle, chars))
        body.append(b"$".join(runs) + b"-")
    return "".join(out).encode() + b"".join(body) + b"\x1b\\"


def _iterm(png: bytes, cols: int) -> str:
    """iTerm2's inline image (OSC 1337), ``cols`` cells wide."""
    data = base64.standard_b64encode(png).decode()
    return (f"\x1b]1337;File=inline=1;size={len(png)};width={cols};"
            f"preserveAspectRatio=1:{data}\x07")


def placed(name: str, pal: Palette, maker: str = "", proto: str = "kitty",
           stream=None, ident: int = 0, biggest: int = COLS) -> Tuple[str, int, int, int]:
    """The card ready to lay over a block of empty cells: the escape, the
    block's width and height in cells, and the row of the block it goes on.

    kitty draws from the block's first row and leaves the cursor put. iTerm2
    and sixel move the cursor past the image and can scroll the screen when
    it reaches the bottom, so they go on the block's last row, an empty one
    under the image: the cursor is kept, moved up to the top, and put back."""
    aw, ah = cols(name, maker, biggest), rows(name, maker, stream, proto, biggest)
    if proto == "kitty":
        return escape(render(name, pal, maker), aw, ident), aw, ah, 0
    if proto == "iterm":
        image = _iterm(render(name, pal, maker), aw)
    else:
        image = render(name, pal, maker, "sixel", aw * _cell(stream)[0]).decode()
    return f"\x1b7\x1b[{ah - 1}A{image}\x1b8", aw, ah, ah - 1


def escape(png: bytes, cols: int, ident: int = 0) -> str:
    """Place ``png`` at the cursor, ``cols`` wide (its height follows); the
    cursor stays put. With an ``ident``, sent again it replaces the one on
    screen instead of piling up (the loading card, redrawn)."""
    data = base64.standard_b64encode(png).decode()
    parts = [data[i:i + 4096] for i in range(0, len(data), 4096)]
    out = []
    for i, part in enumerate(parts):
        more = 1 if i < len(parts) - 1 else 0
        head = (f"a=T,f=100,t=d,c={cols},C=1,q=2,i={ident or new_id()},m={more}" if i == 0
                else f"m={more}")
        out.append(f"\x1b_G{head};{part}\x1b\\")
    return "".join(out)
