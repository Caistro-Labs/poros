"""The graphics card on /machine, drawn as the card that is in the machine.

Every card from the GTX 10 series on has a form: how long, how tall and how
many slots thick it is, and how it is cooled (open fans, a blower, the
flow-through Founders Editions, a passive datacenter heatsink). ``form``
reads it from the card's name and, where nvidia-smi gives it, the board
maker: a Founders Edition looks like one, a partner card gets the partner
shape for its tier. Measurements are in millimetres, so a GTX 1060 is drawn
shorter than an RTX 4090.

One drawing, seen from a little above and to the side like the hardware on
caistrolabs.com: thin grey lines, soft grey faces, the accent on a stripe
along the top and on the hubs. Two ways to show it: a real image where the
terminal can show one (gpu_image), else braille, eight dots a cell, where
the fans turn while the checks run. Stdlib only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from poros.cli.ui.text import Line


@dataclass(frozen=True)
class Form:
    """``style``: open (axial fans) | blower | flow (fan and flow-through fins)
    | dual (two fans, fins between: the 50 series FE) | passive | box."""

    style: str
    fans: int
    length: float  # mm
    height: float  # mm
    slots: float
    edition: str = ""  # "Founders Edition", "reference", "" for a partner card


_FE = "Founders Edition"
_SLOT = 20.32  # mm

# PCI subsystem vendor ids: who made the board
BOARDS = {"10de": "NVIDIA", "1043": "ASUS", "1462": "MSI", "1458": "Gigabyte",
          "3842": "EVGA", "19da": "Zotac", "10b0": "Gainward", "1569": "Palit",
          "196e": "PNY", "1b4c": "Galax", "7377": "Colorful", "1682": "XFX",
          "1da2": "Sapphire", "148c": "PowerColor", "1849": "ASRock", "1028": "Dell",
          "103c": "HP", "17aa": "Lenovo", "1002": "AMD", "8086": "Intel", "1acc": "Point of View",
          "1d17": "Inno3D", "1b0a": "Pegatron", "1da3": "Yeston", "1771": "Biostar"}

# the Founders Editions: (generation, tier, suffix) -> form
_FOUNDERS: Dict[Tuple[int, int, str], Form] = {}
for _k, _f in (
    ([(10, 80, ""), (10, 80, "ti"), (10, 70, ""), (10, 70, "ti")],
     Form("blower", 1, 267, 111, 2, _FE)),
    ([(10, 60, "")], Form("blower", 1, 250, 111, 2, _FE)),
    ([(20, 80, "ti"), (20, 80, ""), (20, 80, "super"), (20, 70, ""), (20, 70, "super")],
     Form("open", 2, 267, 116, 2, _FE)),
    ([(20, 60, ""), (20, 60, "super")], Form("open", 2, 229, 113, 2, _FE)),
    ([(30, 90, ""), (30, 90, "ti")], Form("flow", 1, 313, 138, 3, _FE)),
    ([(30, 80, ""), (30, 80, "ti")], Form("flow", 1, 285, 112, 2, _FE)),
    ([(30, 70, ""), (30, 70, "ti"), (30, 60, "ti")], Form("open", 2, 242, 112, 2, _FE)),
    ([(40, 90, "")], Form("flow", 1, 304, 137, 3, _FE)),
    ([(40, 80, ""), (40, 80, "super")], Form("flow", 1, 310, 140, 3, _FE)),
    ([(40, 70, ""), (40, 70, "super")], Form("flow", 1, 244, 112, 2, _FE)),
    ([(40, 60, "ti")], Form("flow", 1, 244, 98, 2, _FE)),
    ([(50, 90, ""), (50, 80, "")], Form("dual", 2, 304, 137, 2, _FE)),
    ([(50, 70, "")], Form("dual", 2, 242, 112, 2, _FE)),
):
    for _key in _k:
        _FOUNDERS[_key] = _f


def _partner(gen: int, tier: int) -> Form:
    """The usual partner card for a tier: bigger every generation."""
    new = gen >= 30
    if tier >= 90:
        return Form("open", 3, 340 if new else 300, 140 if new else 125, 3.5 if new else 2.5)
    if tier >= 80:
        return Form("open", 3, 320 if new else 290, 135 if new else 122, 3 if new else 2.5)
    if tier >= 70:
        return Form("open", 3 if new else 2, 300 if new else 270, 125 if new else 120, 2.5)
    if tier >= 60:
        return Form("open", 2, 245, 120, 2)
    if tier >= 50:
        return Form("open", 2 if gen >= 20 else 1, 210, 115, 2)
    return Form("open", 1, 170, 105, 2)


# by name: the cards that are not GeForce (first match wins)
_NAMED: List[Tuple[str, Form]] = [
    (r"gb10|dgx spark", Form("box", 0, 150, 50.5, 7.4, "")),
    (r"sxm|hbm3|b200|b100|gb200|gh200|h200 (?!nvl)|mi300|mi325|mi250|mi355|oam",
     Form("passive", 0, 140, 102, 4, "module")),
    (r"rtx pro 6000.*(server)", Form("passive", 0, 267, 111, 2, "")),
    (r"rtx pro 6000.*max-q|rtx pro [45]", Form("blower", 1, 267, 111, 2, "")),
    (r"rtx pro 6000", Form("dual", 2, 304, 137, 2, "")),
    (r"\b(l4|t4|a2)\b", Form("passive", 0, 169, 69, 1, "")),
    (r"a100|h100|h200|l40|a40|a30|\ba10\b|a16|v100|p100|p40|k80|mi210|mi100|mi50|instinct",
     Form("passive", 0, 267, 111, 2, "")),
    (r"a2000|rtx 2000", Form("blower", 1, 167, 69, 2, "")),
    (r"a4000|rtx 4000", Form("blower", 1, 241, 111, 1, "")),
    (r"rtx a\d|rtx [5-8]000|quadro|tesla|a6000|a5000|a4500|w7900|w7800|w6800",
     Form("blower", 1, 267, 111, 2, "")),
    (r"7900 ?xtx|7900 ?xt|7900 gre", Form("open", 3, 287, 123, 2.5, "reference")),
    (r"9070", Form("open", 3, 310, 130, 3, "")),
    (r"9060", Form("open", 2, 240, 120, 2, "")),
    (r"7800 ?xt|7700 ?xt", Form("open", 2, 267, 111, 2, "reference")),
    (r"7600", Form("open", 2, 204, 115, 2, "reference")),
    (r"69[05]0|6800 ?xt", Form("open", 3, 267, 120, 2.5, "reference")),
    (r"6800", Form("open", 3, 267, 120, 2, "reference")),
    (r"67[05]0", Form("open", 2, 267, 110, 2, "reference")),
    (r"66[05]0|65[05]0|64[05]0", Form("open", 2, 190, 110, 2, "")),
    (r"5700|5600|vega|radeon vii", Form("blower", 1, 267, 111, 2, "reference")),
    (r"arc a7[57]0|arc b580|arc b570", Form("open", 2, 270, 113, 2, "Limited Edition")),
    (r"arc", Form("open", 1, 190, 110, 2, "")),
]


def vendor(name: str) -> str:
    """Who made the chip, from its name; "" when the name doesn't say."""
    n = f" {name.lower()} "
    if any(k in n for k in ("rtx", "gtx", "geforce", "quadro", "tesla", "nvidia", " a100",
                            " h100", " h200", " b200", " l40", " l4 ", " a10", "a6000", "gb10")):
        return "NVIDIA"
    if any(k in n for k in ("radeon", "instinct", " amd", "mi300", "mi250", " rx ")):
        return "AMD"
    if any(k in n for k in (" arc ", "intel")):
        return "INTEL"
    return ""


def board(sub_device_id: str) -> str:
    """The board maker from nvidia-smi's ``pci.sub_device_id`` ("0x16101043")."""
    s = sub_device_id.strip().lower().removeprefix("0x")
    return BOARDS.get(s[-4:], "") if len(s) >= 4 else ""


def form(name: str, maker: str = "") -> Form:
    """The card's form from its name and its board maker ("" when unknown:
    then the Founders Edition where there is one)."""
    n = " " + re.sub(r"\s+", " ", re.sub(r"\((r|tm)\)", "", name.lower())) + " "
    m = re.search(r"\b(?:gtx|rtx)\s?(\d{2})(\d)0\b\s?(ti|super)?", n)
    gen = int(m.group(1)) if m else 0
    if m and gen in (10, 16, 20, 30, 40, 50) and not re.search(r"rtx (pro|a\d)", n):
        tier, suf = int(m.group(2)) * 10, m.group(3) or ""
        fe = _FOUNDERS.get((gen, tier, suf))
        if fe is not None and maker in ("", "NVIDIA"):
            return fe
        return _partner(gen, tier)
    for pat, f in _NAMED:
        if re.search(pat, n):
            return f
    return Form("open", 2, 267, 111, 2)


def passive(name: str) -> bool:
    """Cooled by the server's fans, not its own."""
    return form(name).style in ("passive", "box")


# -- the drawing --------------------------------------------------------------

_A1, _A2 = math.radians(15), math.radians(30)
_U = (math.cos(_A1), math.sin(_A1))  # along the card, down to the right
_D = (math.cos(_A2) * 0.55, -math.sin(_A2) * 0.55)  # into it, up to the right
REF = Form("open", 3, 340, 140, 3.5)  # the biggest card: full width

Pt = Tuple[float, float]
# ("fill", outline, tone, alpha) or ("line", from, to, tone, weight), in order
Op = Tuple


def _raw(u: float, v: float, d: float = 0.0) -> Pt:
    return (u * _U[0] + d * _D[0], u * _U[1] - v + d * _D[1])


# a stroke font for the name on the shroud: a 4 x 6 grid, y up
_GLYPHS: Dict[str, List[List[Tuple[int, int]]]] = {
    "A": [[(0, 0), (0, 4), (2, 6), (4, 4), (4, 0)], [(0, 3), (4, 3)]],
    "B": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)],
          [(3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "C": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)]],
    "D": [[(0, 0), (0, 6), (2, 6), (4, 4), (4, 2), (2, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3), (3, 3)]],
    "G": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 3)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "I": [[(1, 6), (3, 6)], [(2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "J": [[(4, 6), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "K": [[(0, 0), (0, 6)], [(4, 6), (0, 2)], [(1, 3), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [[(1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)]],
    "P": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)]],
    "Q": [[(1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)],
          [(2, 2), (4, 0)]],
    "R": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(2, 3), (4, 0)]],
    "S": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (3, 3), (4, 2), (4, 1), (3, 0),
           (1, 0), (0, 1)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1, 0), (2, 3), (3, 0), (4, 6)]],
    "X": [[(0, 0), (4, 6)], [(0, 6), (4, 0)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0, 6), (4, 6), (0, 0), (4, 0)]],
}


def brand(name: str, maker: str = "") -> str:
    """The name on the shroud: the partner's, else the line's."""
    if maker and maker not in ("NVIDIA", "AMD", "Intel"):
        return "".join(ch for ch in maker.upper() if ch in _GLYPHS or ch == " ")
    n, v = name.lower(), vendor(name)
    if v == "NVIDIA":
        if re.search(r"\bgtx\b", n):
            return "GEFORCE GTX"
        if re.search(r"\brtx\s?\d{4}\b", n):
            return "GEFORCE RTX"
        return "NVIDIA"
    if v == "AMD":
        return "AMD INSTINCT" if "instinct" in n or re.search(r"\bmi\d", n) else "RADEON"
    if v == "INTEL":
        return "INTEL ARC"
    return ""


@lru_cache(maxsize=32)
def bounds(f: Form, detail: bool = True) -> Tuple[float, float, float, float]:
    """min x, min y, width, height of the drawing, in mm on the page."""
    xs: List[float] = []
    ys: List[float] = []
    for op in drawing(f, detail=detail):
        pts = op[1] if op[0] == "fill" else (op[1], op[2])
        for x, y in pts:
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def share(f: Form) -> float:
    """How wide the card is drawn against the biggest one (0..1)."""
    return max(0.45, bounds(f)[2] / bounds(REF)[2])


def drawing(f: Form, angle: float = 0.0, *, detail: bool = True, label: str = "") -> List[Op]:
    """The card as fills and lines, back to front, in page millimetres. Fill
    tones: shadow top end face shroud recess fin blade blade2 hub cap metal port
    slot; line tones: line hair faint accent. Without ``detail`` (braille): the
    outline, the fans, the fins and the stripe."""
    L, T, K = f.length, f.height, f.slots * _SLOT
    ops: List[Op] = []
    P = _raw

    def fill(poly: List[Pt], tone: str, alpha: float = 1.0) -> None:
        ops.append(("fill", poly, tone, alpha))

    def line(a: Pt, b: Pt, tone: str, wt: float = 1.0) -> None:
        ops.append(("line", a, b, tone, wt))

    def loop(pts: List[Pt], tone: str, wt: float = 1.0) -> None:
        for a, b in zip(pts, pts[1:] + pts[:1]):
            line(a, b, tone, wt)

    def ring(cu: float, cv: float, r: float, n: int = 96, d: float = 0.0) -> List[Pt]:
        return [P(cu + r * math.cos(2 * math.pi * i / n), cv + r * math.sin(2 * math.pi * i / n),
                  d) for i in range(n)]

    def quad(u0: float, u1: float, v0: float, v1: float, d: float = 0.0) -> List[Pt]:
        return [P(u0, v0, d), P(u1, v0, d), P(u1, v1, d), P(u0, v1, d)]

    def block(u0: float, u1: float, v0: float, v1: float, d0: float, d1: float,
              tone: str = "shroud") -> None:
        """A small box sitting on the card: its top, front and end."""
        top = [P(u0, v1, d0), P(u1, v1, d0), P(u1, v1, d1), P(u0, v1, d1)]
        front = [P(u0, v0, d0), P(u1, v0, d0), P(u1, v1, d0), P(u0, v1, d0)]
        end = [P(u1, v0, d0), P(u1, v0, d1), P(u1, v1, d1), P(u1, v1, d0)]
        for poly, t in ((front, tone), (end, "end"), (top, "top")):
            fill(poly, t)
            loop(poly, "line", 0.7)

    closed = f.style in ("blower", "passive", "box")
    ks = K if closed else K * (0.30 if f.style in ("flow", "dual") else 0.45)  # the shroud
    bp = 0.0 if f.style == "box" else 2.2  # the backplate

    if detail:  # a soft shadow on the ground
        for i in range(8):
            e = 2.0 + i * 2.6
            fill([P(-e * 0.6, -1.0, -e * 0.5), P(L + e * 0.8, -1.0, -e * 0.5),
                  P(L + e * 0.8, -1.0, K + e), P(-e * 0.6, -1.0, K + e)], "shadow", 0.035)

    face = quad(0, L, 0, T)
    top = [P(0, T), P(L, T), P(L, T, K), P(0, T, K)]
    end = [P(L, 0), P(L, 0, K), P(L, T, K), P(L, T)]
    fill(top, "top")
    fill(end, "end")
    fill(face, "face")

    # -- the top: shroud, fin stack, heat pipes, backplate, power, the name --
    if detail:
        if ks < K - bp:
            fin_top = [P(0, T, ks), P(L, T, ks), P(L, T, K - bp), P(0, T, K - bp)]
            fill(fin_top, "fin")
            u = 2.0
            while u < L - 1.5:
                line(P(u, T, ks + 0.4), P(u, T, K - bp - 0.4), "hair", 0.6)
                u += 2.3
            line(P(0, T, ks), P(L, T, ks), "line", 0.9)
            if f.style == "open" and not f.edition:  # partner cards show their heat pipes
                n = 5 if f.fans >= 3 else 4
                for i in range(n):
                    u = L * 0.30 + i * 7.5
                    arc = [P(u, T + 4.2 * math.sin(math.pi * s / 10), ks + 1.5
                             + (K - bp - ks - 3) * s / 10) for s in range(11)]
                    for a, b in zip(arc, arc[1:]):
                        line(a, b, "line", 3.2)
                    for a, b in zip(arc, arc[1:]):
                        line(a, b, "cap", 1.5)
        if bp:
            line(P(0, T, K - bp), P(L, T, K - bp), "hair", 0.8)
            fill([P(0, T, K - bp), P(L, T, K - bp), P(L, T, K), P(0, T, K)], "metal")
            loop([P(0, T, K - bp), P(L, T, K - bp), P(L, T, K), P(0, T, K)], "hair", 0.6)
        if not closed or f.style == "blower":  # power: 12V-2x6 on big new cards, else 8-pin
            new = f.style in ("flow", "dual") or f.length >= 290
            w = 19.0 if new else 21.0
            cu = L * (0.52 if f.style in ("flow", "dual") else 0.70)
            d0 = (K - bp - 9.5) if not closed else K * 0.45
            block(cu, cu + w, T, T + 3.2, d0, d0 + 7.5)
            pins = (6, 2) if new else (4, 2)
            for i in range(pins[0]):
                for j in range(pins[1]):
                    c = P(cu + w * (i + 0.5) / pins[0], T + 3.2, d0 + 7.5 * (j + 0.5) / 2)
                    fill([(c[0] + dx * 0.55, c[1] + dy * 0.35) for dx, dy in
                          ((-1, 0), (0, -1), (1, 0), (0, 1))], "port")
        if label:
            span = (ks if not closed else K) - 2
            h = max(3.0, min(8.0, span * 0.45))
            s = h / 6
            total = len(label) * 6 * s
            if total > L * 0.55:
                s *= L * 0.55 / total
            d_mid = (ks * 0.62) if not closed else K * 0.55
            u0 = L * 0.07
            for ch in label:
                for stroke in _GLYPHS.get(ch, []):
                    pts = [P(u0 + x * s, T, d_mid - 3 * s + y * s) for x, y in stroke]
                    for a, b in zip(pts, pts[1:]):
                        line(a, b, "line", 1.0)
                u0 += 6 * s
    if f.style != "box":  # the stripe
        dd = ks * 0.22 if not closed else K * 0.15
        line(P(L * 0.04, T, dd), P(L * 0.86, T, dd), "accent", 1.6)

    # -- the face --------------------------------------------------------------
    m = T * 0.055
    if detail:
        loop(quad(m, L - m, m, T - m), "hair", 0.8)

    def fan(cu: float, cv: float, rr: float, spin: float, blades: int = 9,
            tip: bool = True) -> None:
        fill(ring(cu, cv, rr * 1.07), "recess")
        if detail:
            loop(ring(cu, cv, rr * 1.07), "hair", 0.8)
        loop(ring(cu, cv, rr * 0.995), "line", 0.9)
        r0, r1 = rr * 0.31, rr * (0.90 if tip else 0.93)
        n = blades if detail else 5
        for b in range(n):
            th0 = 2 * math.pi * b / n + spin
            if detail:
                lead, mid, trail = [], [], []
                for k in range(15):
                    s = k / 14
                    r, sw, wd = r0 + (r1 - r0) * s, 0.95 * s, 0.20 + 0.34 * s
                    for arr, off in ((lead, 0.0), (mid, wd * 0.55), (trail, wd)):
                        arr.append(P(cu + r * math.cos(th0 + sw + off),
                                     cv + r * math.sin(th0 + sw + off)))
                fill(lead + trail[::-1], "blade")
                fill(mid + trail[::-1], "blade2")
                loop(lead + trail[::-1], "hair", 0.6)
            else:  # one curve a blade, hub to rim
                prev = P(cu + r0 * math.cos(th0), cv + r0 * math.sin(th0))
                for k in range(1, 9):
                    r, th = r0 + (rr * 0.85 - r0) * k / 8, th0 + 0.8 * k / 8
                    nxt = P(cu + r * math.cos(th), cv + r * math.sin(th))
                    line(prev, nxt, "line", 0.9)
                    prev = nxt
        if detail and tip:  # the ring that joins the blade tips
            pts = ring(cu, cv, rr * 0.93)
            for a, b in zip(pts, pts[1:] + pts[:1]):
                line(a, b, "blade", 2.4)
            loop(ring(cu, cv, rr * 0.955), "hair", 0.6)
            loop(ring(cu, cv, rr * 0.905), "hair", 0.6)
        fill(ring(cu, cv, rr * 0.31, 48), "hub")
        loop(ring(cu, cv, rr * 0.31, 48), "line", 0.9)
        if detail:
            fill(ring(cu, cv, rr * 0.24, 48), "cap")
            loop(ring(cu, cv, rr * 0.24, 48), "hair", 0.6)
        loop(ring(cu, cv, rr * 0.17, 48), "accent", 1.3)

    def impeller(cu: float, cv: float, rr: float, spin: float) -> None:
        fill(ring(cu, cv, rr * 1.10), "recess")
        loop(ring(cu, cv, rr * 1.10), "line", 0.9)
        for b in range(30 if detail else 12):
            th = 2 * math.pi * b / (30 if detail else 12) + spin
            line(P(cu + rr * 0.55 * math.cos(th), cv + rr * 0.55 * math.sin(th)),
                 P(cu + rr * 0.97 * math.cos(th + 0.25), cv + rr * 0.97 * math.sin(th + 0.25)),
                 "hair" if detail else "line", 0.8)
        fill(ring(cu, cv, rr * 0.52, 48), "hub")
        loop(ring(cu, cv, rr * 0.52, 48), "line", 0.9)
        if detail:
            fill(ring(cu, cv, rr * 0.40, 48), "cap")
        loop(ring(cu, cv, rr * 0.30, 48), "accent", 1.3)

    def fins_up(u0: float, u1: float, v0: float, v1: float, pitch: float = 2.6) -> None:
        """A window onto fins standing across the card, each with its depth."""
        fill(quad(u0, u1, v0, v1), "recess")
        n = max(3, int((u1 - u0) / pitch)) if detail else max(3, int((u1 - u0) / 7))
        for i in range(n):
            u = u0 + (u1 - u0) * (i + 0.5) / n
            if detail:
                plate = [P(u, v0 + 0.5), P(u, v1 - 0.5), P(u, v1 - 0.5, 5.0), P(u, v0 + 0.5, 5.0)]
                fill(plate, "fin")
                line(P(u, v0 + 0.5), P(u, v1 - 0.5), "hair", 0.7)
            else:
                line(P(u, v0 + 1), P(u, v1 - 1), "line", 0.9)
        loop(quad(u0, u1, v0, v1), "line", 0.9)

    def fins_along(u0: float, u1: float, v0: float, v1: float, pitch: float = 3.2) -> None:
        """A window onto fins running the length of the card."""
        fill(quad(u0, u1, v0, v1), "recess")
        n = max(3, int((v1 - v0) / pitch)) if detail else max(3, int((v1 - v0) / 9))
        for i in range(n):
            v = v0 + (v1 - v0) * (i + 0.5) / n
            if detail:
                plate = [P(u0 + 0.5, v), P(u1 - 0.5, v), P(u1 - 0.5, v, 5.0), P(u0 + 0.5, v, 5.0)]
                fill(plate, "fin")
                line(P(u0 + 0.5, v), P(u1 - 0.5, v), "hair", 0.7)
            else:
                line(P(u0 + 1, v), P(u1 - 1, v), "line", 0.9)
        loop(quad(u0, u1, v0, v1), "line", 0.9)

    def spin(i: int) -> float:
        return angle * (1 if i % 2 == 0 else -1) + i * 0.3

    def screw(u: float, v: float) -> None:
        if detail:
            fill(ring(u, v, 1.1, 16), "port")
            loop(ring(u, v, 1.1, 16), "hair", 0.5)

    rr = T * 0.42
    if f.style == "open":
        n = max(1, f.fans)
        rr = min(rr, (L - 2 * m) / n * 0.46)
        step = (L - 2 * m) / n
        centres = [m + step * (i + 0.5) for i in range(n)]
        if detail and not f.edition:  # a partner shroud: angled cuts between the fans
            for g in [m + step * i for i in range(n + 1)]:
                g = min(max(g, m + 3.5), L - m - 3.5)
                for v0, v1 in ((T * 0.10, T * 0.36), (T * 0.64, T * 0.90)):
                    cut = [P(g - 2.8, v0), P(g + 2.8, v0 + (v1 - v0) * 0.25),
                           P(g + 2.8, v1), P(g - 2.8, v1 - (v1 - v0) * 0.25)]
                    fill(cut, "shroud")
                    loop(cut, "hair", 0.6)
            line(P(m, T * 0.06), P(L - m, T * 0.06), "hair", 0.6)
        for i, cu in enumerate(centres):
            fan(cu, T / 2, rr, spin(i), 9 if f.edition else 11)
            for a in (45, 135, 225, 315):
                ang = math.radians(a)
                screw(cu + rr * 1.16 * math.cos(ang), T / 2 + rr * 1.16 * math.sin(ang) * 0.9)
    elif f.style == "blower":  # the impeller at the far end, a window down the shroud
        rr = T * 0.34
        cu = L * 0.17
        if detail:  # the faceted shroud
            for a, b in ((P(m, m), P(L * 0.34, T * 0.12)), (P(m, T - m), P(L * 0.34, T * 0.88)),
                         (P(L - m, m), P(L * 0.92, T * 0.12)),
                         (P(L - m, T - m), P(L * 0.92, T * 0.88))):
                line(a, b, "hair", 0.7)
        fins_along(L * 0.36, L * 0.90, T * 0.18, T * 0.82)
        impeller(cu, T / 2, rr, spin(0))
    elif f.style == "flow":  # the fan by the bracket, fins open at the far end
        frame = quad(m * 2, L - m * 2, m * 2, T - m * 2)
        if detail:
            fill(frame, "shroud")
            loop(frame, "hair", 0.7)
        fins_up(m * 3.2, L * 0.42, T * 0.13, T * 0.87)
        fan(L * 0.70, T / 2, min(rr, L * 0.2), spin(0), 11)
    elif f.style == "dual":  # a fan at each end, fins between
        r2 = min(rr, L * 0.17)
        frame = quad(m * 2, L - m * 2, m * 2, T - m * 2)
        if detail:
            fill(frame, "shroud")
            loop(frame, "hair", 0.7)
        fins_up(m + r2 * 2.45, L - m - r2 * 2.45, T * 0.13, T * 0.87)
        fan(m + r2 * 1.14, T / 2, r2, spin(0), 11)
        fan(L - m - r2 * 1.14, T / 2, r2, spin(1), 11)
    elif f.style == "passive":
        fins_along(m * 2, L - m * 2, T * 0.14, T * 0.86)
        for u in (m * 2 + 3, L - m * 2 - 3):
            for v in (T * 0.07, T * 0.93):
                screw(u, v)
    elif f.style == "box":  # a perforated front
        nx, ny = (int(L / 4.2), int(T / 4.2)) if detail else (14, 4)
        for i in range(nx):
            for j in range(ny):
                cu, cv = L * (i + 0.5) / nx, T * (j + 0.5) / ny
                if detail:
                    fill(ring(cu, cv, 1.25, 12), "port", 0.55)
                else:
                    c = P(cu, cv)
                    line(c, (c[0] + 0.6, c[1]), "line", 1.6)

    # -- the bracket end: ports, vents, the screw tab ---------------------------
    if detail and f.style != "box":
        ub = L + 0.8
        plate = [P(ub, -6, -0.6), P(ub, -6, K + 0.6), P(ub, T + 1.5, K + 0.6), P(ub, T + 1.5, -0.6)]
        fill(plate, "metal")
        loop(plate, "line", 0.9)
        tab = [P(ub, T + 1.5, -0.6), P(ub + 7, T + 1.5, -0.6), P(ub + 7, T + 1.5, min(K, 21)),
               P(ub, T + 1.5, min(K, 21))]
        fill(tab, "metal")
        loop(tab, "line", 0.8)
        c = P(ub + 3.5, T + 1.5, min(K, 21) * 0.5)
        fill([(c[0] + 1.6 * math.cos(a / 8 * math.pi), c[1] + 0.9 * math.sin(a / 8 * math.pi))
              for a in range(16)], "port")
        dp = K - bp - 4.5  # the ports sit by the board
        if f.style != "passive":
            n = 4
            top_v, bot_v = T * 0.86, T * 0.12
            h = (top_v - bot_v) / n
            for i in range(n):
                v0, v1 = bot_v + i * h + 1.6, bot_v + (i + 1) * h - 1.6
                d0, d1 = dp - 3.2, dp + 3.2
                port = [P(ub, v0, d0), P(ub, v0, d1), P(ub, v1, d1), P(ub, v1, d0 + 1.6),
                        P(ub, v1 - 1.6, d0)]
                fill(port, "port")
                loop(port, "line", 0.6)
            vent_d1 = dp - 6.5
        else:
            vent_d1 = K - 2
        cols = int((vent_d1 - 1.5) / 4.2)
        for i in range(max(0, cols)):
            d = 1.5 + i * 4.2
            for j in range(int((T * 0.84) / 8.5)):
                v = T * 0.08 + j * 8.5
                slot = [P(ub, v, d), P(ub, v, d + 2.0), P(ub, v + 6.5, d + 2.0), P(ub, v + 6.5, d)]
                fill(slot, "slot")
    elif not detail:  # vents on the end
        cols = max(2, round(K / 9))
        for i in range(cols):
            d = K * (i + 1) / (cols + 1)
            for j in range(6):
                v0 = T * 0.1 + j * T * 0.8 / 6
                line(P(L, v0 + T * 0.02, d), P(L, v0 + T * 0.8 / 6 - T * 0.03, d), "hair", 0.9)

    for poly in (face, top, end):
        loop(poly, "line", 1.1)
    return ops


# -- braille -------------------------------------------------------------------

W = 54  # the biggest card's columns in braille
_RANK = ("a", "dm", "rl")  # where strokes share a cell, the earlier style wins
_TOKEN = {"accent": "a", "line": "dm", "hair": "rl"}
_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def size(f: Form, cols: int = W) -> Tuple[int, int]:
    """Columns and rows for the card in braille (dots are about square)."""
    _, _, sx, sy = bounds(f, False)
    c = max(20, round(cols * share(f)))
    return c, max(4, math.ceil(c * 2 * (sy / sx) / 4))


def card(name: str, angle: float = 0.0, maker: str = "") -> List[Line]:
    """The card in braille; ``angle`` turns the fans."""
    f = form(name, maker)
    cw, ch = size(f)
    minx, miny, sx, sy = bounds(f, False)
    sc = min((cw * 2 - 1) / sx, (ch * 4 - 1) / sy)
    dots: Dict[Tuple[int, int], str] = {}

    def dot(x: float, y: float, st: str) -> None:
        p = (int(round(x)), int(round(y)))
        if 0 <= p[0] < cw * 2 and 0 <= p[1] < ch * 4:
            old = dots.get(p)
            if old is None or _RANK.index(st) < _RANK.index(old):
                dots[p] = st

    for op in drawing(f, angle, detail=False):
        if op[0] != "line" or op[3] not in _TOKEN:
            continue
        _, a, b, tone, _wt = op
        (x1, y1), (x2, y2) = [((x - minx) * sc, (y - miny) * sc) for x, y in (a, b)]
        n = max(1, int(max(abs(x2 - x1), abs(y2 - y1))))
        for i in range(n + 1):
            t = i / n
            dot(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, _TOKEN[tone])
    out: List[Line] = []
    for cy in range(ch):
        row: Line = []
        for cx in range(cw):
            bits, best = 0, None
            for dy in range(4):
                for dx in range(2):
                    st = dots.get((cx * 2 + dx, cy * 4 + dy))
                    if st is not None:
                        bits |= _BITS[dy][dx]
                        if best is None or _RANK.index(st) < _RANK.index(best):
                            best = st
            ch_ = chr(0x2800 + bits) if bits else " "
            if row and row[-1][1] == best:
                row[-1] = (row[-1][0] + ch_, best)
            else:
                row.append((ch_, best))
        out.append(row)
    return out


def label(name: str, maker: str = "") -> Optional[str]:
    """What to call the board: "Founders Edition", "ASUS", "reference"…"""
    f = form(name, maker)
    if maker and maker not in ("NVIDIA", "AMD", "Intel"):
        return maker
    return f.edition or None
