"""A list a sheet browses: its items (``Item``, ``Listing``), the lister that shows and picks
from them, and the narrow-column text helpers it lays rows out with."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from poros.cli.ui import components as c
from poros.cli.ui.sheet.frame import (
    METER_MIN,
    NAME_MIN,
    NAME_WHOLE,
    Outcome,
    _clip,
    _fit,
    _loop,
    fullscreen,
)
from poros.cli.ui.text import Line, S, banded, ln, width


@dataclass
class Item:
    """One row of a list card: its cells, the words typing matches, and what
    each key does on it -- ``(key kind, hint, command line)``, enter first."""

    cells: Line
    text: str
    actions: List[Tuple[str, str, List[str]]]
    detail: Line = field(default_factory=list)  # under the list while it is the chosen row
    tag: Line = field(default_factory=list)  # after the cells; the first thing cut when narrow
    panel: List[Line] = field(default_factory=list)  # a card's detail panel, in place of detail
    group: str = ""  # the Listing group it belongs to (a lab, for /models)


@dataclass
class Listing:
    tag: str
    title: str
    head: Line  # column names; empty when the cells speak for themselves
    items: List[Item]
    empty: str = "nothing here yet"
    window: int = 12
    aside: str = ""  # right of the title: how the list is ordered
    start: int = 0  # the row it opens on (the recommended one)
    panel: int = 0  # rows under the list for the focused item's panel; 0: one detail line
    # (key, chip): the groups ←→ steps through, after "All"; a chip is what
    # names the group in the strip under the title (a lab: its mark and name)
    groups: List[Tuple[str, Line]] = field(default_factory=list)


class Lister:
    """Move through a list, type to narrow it; a key picks what runs.

    ``key`` returns "go" (``self.argv`` is the command), "cancel" or None."""

    def __init__(self, listing: Listing) -> None:
        self.listing = listing
        self.buf = ""
        self.sel = min(listing.start, max(len(listing.items) - 1, 0))
        self.top = max(0, self.sel - listing.window + 1)
        self.argv: Optional[List[str]] = None
        self.group = 0  # 0: every group; else the group at groups[group - 1]

    def shown(self) -> List[Item]:
        q = self.buf.strip().lower()
        g = self.listing.groups[self.group - 1][0] if self.group else None
        return [i for i in self.listing.items
                if (not q or q in i.text.lower()) and (g is None or i.group == g)]

    def takes_slash(self) -> bool:
        """A ``/`` before any filter is typed means a command."""
        return not self.buf

    def key(self, kind: str, ch: str = "") -> Optional[Outcome]:
        items = self.shown()
        if kind in ("esc", "ctrl-c"):
            if self.buf and kind == "esc":
                self._search("", items)
                return None
            return "cancel"
        if kind == "up":
            self.sel = max(self.sel - 1, 0)
        elif kind in ("down", "tab"):
            self.sel = min(self.sel + 1, max(len(items) - 1, 0))
        elif kind in ("left", "right") and self.listing.groups:
            n = len(self.listing.groups) + 1
            self.group = (self.group + (1 if kind == "right" else -1)) % n
            self._search(self.buf, items)
        elif kind == "backspace":
            self._search(self.buf[:-1], items)
        elif kind == "char" and ch.isprintable():
            self._search(self.buf + ch, items)
        elif items:
            item = items[min(self.sel, len(items) - 1)]
            want = "enter" if kind == "enter" else kind
            for k, _, argv in item.actions:
                if k == want:
                    self.argv = argv
                    return "go"
        n = self.listing.window
        if self.sel < self.top:
            self.top = self.sel
        elif self.sel >= self.top + n:
            self.top = self.sel - n + 1
        return None

    def _strip(self, w: int) -> List[Line]:
        """The groups as chips under the title, each with its count; the
        chosen one lit, its count in the accent."""
        L = self.listing
        if not L.groups:
            return []
        chips = [("All", [("All", None)], len(L.items))] + [
            (k, list(chip), sum(1 for i in L.items if i.group == k)) for k, chip in L.groups]
        row: Line = [S(c.RAIL, "rl")]
        room = w - len(c.RAIL)
        for n, (_, chip, count) in enumerate(chips):
            on = n == self.group
            body = [(t, st) if st == "raw" else (t, ("hi" if on else "dm") if st is None else st)
                    for t, st in chip] + [S(f" {count}", "a" if on else "rl")]
            gap = "    " if n else ""
            if width(body) + len(gap) > room:
                break
            row += [S(gap)] + body
            room -= width(body) + len(gap)
        return [row]

    def _search(self, buf: str, was: List[Item]) -> None:
        """Search for ``buf``, the focus kept on the row it was on while that
        row still matches (else the first match)."""
        on = was[self.sel] if was and self.sel < len(was) else None
        self.buf = buf
        now = self.shown()
        self.sel = next((i for i, x in enumerate(now) if x is on), 0)
        self.top = max(0, self.sel - self.listing.window + 1)

    def render(self, w: int) -> List[Line]:
        L = self.listing
        items = self.shown()
        n = min(L.window, max(len(L.items), 1))
        w = min(w, c.CAP)
        said = L.aside if L.items else ""  # an empty list has no order and no count
        gap = w - 2 - len(L.title) - len(said)
        aside = S(" " * gap + said, "rl") if gap > 1 and said else None
        title = c.titled(ln(S("● ", "tx"), S(L.title, "hi"), aside))
        # the search sits under the title, where a card's filter does, and says so before it's used
        out: List[Line] = [title] + ([c.rail()] if L.groups else []) + self._strip(w) + [
            c.rail(S("search  ", "dm"), S(self.buf, "hi"), S("▏", "a"),
                   S(f"  {len(items)} of {len(L.items)}", "rl")) if self.buf
            else c.rail(S("type to search", "rl")) if L.items else c.rail()]
        # too narrow: the widest column (the name) gives way, so numbers stay whole
        # (a row's tag, like "recommended", is cut instead)
        over = max((width(i.cells) for i in L.items), default=0) - (w - len(c.RAIL))
        fit = _Narrow(L.items, over, L.head)  # one set of widths: the columns stay in line
        if L.head and L.items:  # no column names over nothing
            out.append(c.rail(*whole(fit.head(L.head), w - len(c.RAIL))))
        page = items[self.top:self.top + n]
        for i, item in enumerate(page):
            on = self.top + i == self.sel
            gutter = S("  ❯ ", "a") if on else S(c.RAIL, "rl")
            cells = fit.row(item.cells) + list(item.tag)
            cells = c.light(cells) if on else [(t, _dim(st)) for t, st in cells]
            row = whole(ln(gutter, *cells), w)
            out.append(banded(row, w) if on else row)
        if not items:
            why = f"nothing matches {self.buf!r}" if self.buf else L.empty
            out.append(c.rail(S(why, "dm")))
            page = [None]
        out += [c.rail()] * (n - len(page))  # the card keeps its height as typing narrows it
        sel = items[min(self.sel, len(items) - 1)] if items else None
        where = (f"{self.top + 1}–{self.top + len(page)} of {len(items)}"
                 if len(items) > len(page) and page != [None] else "")
        if L.panel:  # the focused item, in full; the same height for every item
            out.append(c.rail(S(" " * max(0, w - len(c.RAIL) - len(where)) + where, "rl")))
            body = sel.panel if sel is not None else []
            out += [c.rail(*_clip(list(x), w - len(c.RAIL))) for x in body[:L.panel]]
            out += [c.rail()] * (L.panel - len(body[:L.panel]))
            where, sel_detail = "", False
        else:
            sel_detail = True
        if sel_detail and sel and width(list(sel.detail)) > w - len(c.RAIL) - len(where) - 2:
            where = ""  # the focused row's detail outranks the count
        if sel_detail:
            about = _clip(list(sel.detail), w - len(c.RAIL) - len(where) - 2) if sel else []
            gap = w - len(c.RAIL) - width(about) - len(where)
            out.append(c.rail())
            out.append(c.rail(*about, S(" " * gap + where, "rl") if where else None))
        if sel is not None:
            first = sel.actions[0]
            out.append(ln(S("  "), S(_fit(c.command(" ".join(first[2])), w - 2), "rl")))
        else:
            out.append([])
        keys = [("⏎" if k == "enter" else c.key_name(k), h)
                for k, h, _ in (sel.actions if sel is not None else [])]
        if L.groups:
            keys.append(("←→", "lab"))
        keys.append(("esc", "clear") if self.buf else ("esc", "close"))
        out += c.keyline(keys, w)
        return [c.inked(line) for line in out]


def _dim(st):
    """An unfocused row's cell: one tier down, the accent kept."""
    return {"hi": "tx", "tx": "dm"}.get(st, st) if isinstance(st, str) else st


def pick(listing: Listing, *, stream=None, lister: type = Lister) -> Optional[List[str]]:
    """Show a list card (``lister`` draws it); the command its key picked, or None."""
    lister = lister(listing)
    with fullscreen(stream):
        return lister.argv if _loop(lister, stream) == "go" else None


def _meter(line: Line) -> Optional[Tuple[int, int]]:
    """(first, last) segment of the line's meter (``━``/``─`` runs), if any."""
    run = [i for i, (t, _) in enumerate(line) if t and set(t) <= {c.METER, "─"}]
    return (run[0], run[-1]) if run and run == list(range(run[0], run[-1] + 1)) else None


def _tail(line: Line) -> bool:
    """Whether the line has a word after its meter's figure."""
    m = _meter(line)
    return m is not None and len(line) > m[1] + 2


class _Narrow:
    """How a list too wide for its card gives way, the same for every row so
    the columns stay in line. In order: the meter (to ``METER_MIN`` cells,
    its share kept), the padding past each column's longest entry, the names
    (a column at a time, the widest first, to ``NAME_WHOLE``), the word
    after the figure ("bounded"), then the names again to ``NAME_MIN``.
    Whatever is still over, ``whole`` drops from the right."""

    def __init__(self, items: Sequence[Item], over: int, head: Line = ()) -> None:
        first = items[0].cells if items else []
        m = _meter(first)
        bar = sum(len(t) for t, _ in first[m[0]:m[1] + 1]) if m else 0
        self.drop = max(0, min(over, bar - METER_MIN))
        over -= self.drop
        lead = m[0] if m else len(first)  # the columns before the meter
        self.cols = [max(_cw(it.cells[i]) for it in items if len(it.cells) > i)
                     for i in range(lead)]
        self.sizes = list(self.cols)
        over = self._share(self._trim(items, head, over), NAME_WHOLE)
        tails = [len(it.cells[-1][0]) for it in items if _tail(it.cells)]
        self.tail = over > 0 and bool(tails)
        self._share(over - (max(tails) if self.tail else 0), NAME_MIN)

    def _trim(self, items: Sequence[Item], head: Line, over: int) -> int:
        """Cut each column to its longest entry or name and two spaces."""
        named, at = {}, 0  # a column's name, by where it starts
        for t, _ in head:
            named[at] = len(t.rstrip())
            at += len(t)
        at = 0
        for i, size in enumerate(self.cols):
            longest = max([_cw(it.cells[i], strip=True) for it in items if len(it.cells) > i]
                          + [named.get(at, 0)])
            spare = min(over, max(0, size - longest - 2))
            self.sizes[i] -= spare
            over -= spare
            at += size
        return over

    def _share(self, over: int, floor: int) -> int:
        """Take ``over`` from the widest columns a cell at a time, none under ``floor``."""
        while over > 0 and self.sizes:
            k = max(range(len(self.sizes)), key=lambda i: self.sizes[i])
            if self.sizes[k] <= floor:
                break
            self.sizes[k] -= 1
            over -= 1
        return over

    def row(self, line: Line) -> Line:
        if self.tail and _tail(line):
            line = line[:-1]
        out = [(_cut(t, self.sizes[i]), st) if i < len(self.sizes) and st != "raw" else (t, st)
               for i, (t, st) in enumerate(line)]
        m = _meter(out)
        if m is None or not self.drop:
            return out
        return [*out[:m[0]], *self._shrink(out[m[0]:m[1] + 1]), *out[m[1] + 1:]]

    def _shrink(self, run: Line) -> Line:
        """A meter ``drop`` cells shorter, its filled share kept."""
        n = sum(len(t) for t, _ in run)
        to = max(METER_MIN, n - self.drop)
        full = sum(len(t) for t, _ in run if c.METER in t)
        keep = min(to, max(1, round(full * to / n))) if full else 0
        fill = next((st for t, st in run if c.METER in t), None)
        track = next((st for t, st in run if c.METER not in t), "tk")
        return [x for x in [(c.METER * keep, fill), ("─" * (to - keep), track)] if x[0]]

    def head(self, line: Line) -> Line:
        """Column names: each loses what the column it starts over lost."""
        lost, at = {}, 0
        for size, now in zip(self.cols, self.sizes):
            lost[at] = size - now
            at += size
        out, at = [], 0
        for t, st in line:
            less = lost.get(at, 0)
            at += len(t)
            out.append((_cut(t, len(t) - less), st) if less else (t, st))
        return out


def _cw(seg, strip: bool = False) -> int:
    """A cell's width in columns: an image takes none of its own."""
    t, st = seg
    return 0 if st == "raw" else len(t.rstrip() if strip else t)


def _cut(text: str, keep: int) -> str:
    """``text`` in ``keep`` columns, padded; an ellipsis where it is cut."""
    if keep >= len(text):
        return text
    body = text.rstrip()  # whole while two spaces still follow it
    return (body if len(body) <= keep - 2 else body[:keep - 3].rstrip() + "…").ljust(keep)


def whole(line: Line, w: int) -> Line:
    """``line`` without the segments that would not fit whole: column names
    give way at the right edge rather than break mid-word."""
    out: Line = []
    for text, style in line:
        if style == "raw":  # an image: no cells of its own
            out.append((text, style))
            continue
        if width(out) + len(text) > w:
            text = text.rstrip()  # the padding after a name can go; the name can't
            if width(out) + len(text) <= w:
                out.append((text, style))
            break
        out.append((text, style))
    return out


