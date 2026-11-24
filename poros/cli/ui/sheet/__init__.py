"""The settings sheet: a command's whole setup on one screen, set from the keys.

    ▛▜ Train Qwen2.5-7B-Instruct on notes.jsonl
      │
      │ model     Qwen2.5-7B-Instruct             recommended here
      │ data      notes.jsonl                     1,204 chats · newest file here
      │
      │ recipe    SFT · answers only · lr 2e-4    your data is chats
      │ steps     151                             one epoch of 1,204 chats
      │ length    1,024 tokens                    fits 95% of your rows
      │
      │ run       Live
      │ more      all 47 settings                 lr · LoRA · memory · saving
      │
      │ fits      ━━━━────────────  6.2 of 24.0 GiB  measured
      │
      ❯ Start training

      /train Qwen2.5-7B-Instruct --data notes.jsonl
    ──────────────────────────────────────────────────────────
     ⏎ start training   ↑ change   ctrl+s save   esc cancel

It opens filled in, on its action: enter runs it. ↑ walks into the rows,
enter or ←→ changes one, typing searches every setting, esc goes back to
the action. A value set (given, picked, or prefilled) is bold; one Poros
worked out is plain, with its reason beside it.

Every row is a value, where it came from, and the way to change it: a pick
list (type to filter), a value typed in place, or an on/off. A warning that
needs a yes (a download, an unvalidated model) is its own row and is never
pre-set. The result row is the one accent on the screen. Under the action
row: what the focused row is for, or the command the sheet will run.

A ``Lister`` is the other card: a list you move through (/models, /runs),
where enter does the obvious thing for the row and a ctrl key the others.

The sheet is pure state: ``key()`` takes one key and ``render()`` returns
lines, so every behavior is tested without a terminal. ``run()`` is the
thin loop that draws it in place. What the rows are, and what changing one
means, comes from a ``Form`` (``poros.cli.sheets`` builds one per verb).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from poros.cli.ui import components as c
from poros.cli.ui.sheet.frame import (
    METER_MIN,
    TOP,
    Outcome,
    _clip,
    _fit,
    _loop,
    canvas,
    fullscreen,
)
from poros.cli.ui.sheet.lister import Item, Lister, Listing, pick, whole
from poros.cli.ui.text import Line, S, banded, ln, width

# The package's API: the sheet (this module), the frame it runs in (frame) and the lists it browses
# (lister). Private names are imported from the module that defines them.
__all__ = ["Field", "Form", "Item", "Kind", "Lister", "Listing", "Option", "Outcome", "Sheet",
           "TOP", "Value", "View", "canvas", "dropped_path", "fullscreen", "pick", "run", "whole"]

Kind = Literal["pick", "text", "toggle"]
Value = Union[str, bool, None]
VISIBLE = 7  # pick-list rows shown at once
TAIL = 44  # the facts and reasons beside a value stop here: a note, never a paragraph
GAP = 4  # between a value and its note
VALUE = 22  # the value column: a value up to 24 columns shows whole
BROWSE = 9  # settings shown at once in the "more" list
BAR = 16  # a card's result bar, before it narrows


@dataclass
class Option:
    value: str
    label: str
    note: str = ""
    sets: Value = None  # for a row that adds rows: the new row's first value
    group: str = ""  # the "more" list's heading this option sits under
    now: str = ""  # the "more" list: the value it trains at now
    words: str = ""  # other words that find it


@dataclass
class Field:
    """One row. ``key`` is what it sets: "subject", or a flag like "--data"."""

    key: str
    label: str
    kind: Kind = "pick"
    value: Value = None  # what goes on the command line (None: not set)
    shown: str = ""  # the value as resolved for reading ("151", "Qwen2.5-7B")
    source: str = ""  # given | default | inferred · why
    options: Optional[Callable[[], List[Option]]] = None
    free: bool = False  # a pick that also takes a typed value
    required: bool = False
    hint: str = ""  # what to give, shown while the row is empty or being typed
    warn: bool = False  # a confirmation: a yes the run needs
    check: Optional[Callable[[str], Optional[str]]] = None  # a typed value's problem
    adds: bool = False  # picking an option adds the row it names (the "more" row)
    inline: bool = False  # a pick of two or three: every choice on the row, enter steps
    fact: str = ""  # a measured fact beside the value, before its source ("1,204 rows")
    quiet: bool = False  # a set value that reads as a setting, not a choice: never bold
    bold: bool = False  # the card's subject: bold even when worked out
    gap: bool = False  # a new group starts here: a quiet line above it
    files: bool = False  # a file dropped in while this row is chosen is its value, not the data
    clears: Tuple[str, ...] = ()  # what this row sets under other keys: back to its default too


@dataclass
class View:
    """What a form shows for the current values."""

    title: Union[str, Line]  # a Line for a sentence with its own weights
    fields: List[Field]
    result: Optional[Tuple[str, str, str]] = None  # (label, value, detail)
    result_style: str = "hi"  # the value's weight: "e" when it will not fit
    error: str = ""  # why it cannot run as set
    go: str = "Start"  # the action row's words
    ready: bool = True  # every required row set, every warning answered
    meter: Optional[Tuple[float, str]] = None  # the result as a bar: (share, fill style)
    command: str = ""  # the command line these values run, shown under the action
    facts: List[Tuple[str, str, str]] = field(default_factory=list)  # (label, what, from where)


@dataclass
class Form:
    """``build(values)`` is called after every change; ``values`` maps a
    field key to its value. A form never runs anything itself."""

    tag: str
    build: Callable[[Dict[str, Value]], View]
    values: Dict[str, Value] = field(default_factory=dict)
    # extra keys: kind -> (hint, fn(values) -> the note to show after), or
    # (hint, [(choice, fn), ...]): the key asks which, on the line under the action
    keys: Dict[str, Tuple[str, Any]] = field(default_factory=dict)
    browse: bool = False  # every row has a value: open on the first, not on the action
    # kept as you change it (settings): no action row, and esc closes, done
    live: Optional[Callable[[Dict[str, Value]], None]] = None


class Sheet:
    def __init__(self, form: Form) -> None:
        self.form = form
        self.view = form.build(dict(form.values))
        self.sel = 0  # row index; len(fields) is the action row
        self.mode: Literal["nav", "pick", "text"] = "nav"
        self.buf = ""  # the filter (pick) or the value being typed (text)
        self.opt = 0  # selected option in the pick list
        self.top = 0  # first option shown
        self.problem = ""  # a typed value that was refused, shown on its row
        self.note = ""  # what an extra key did, shown under the action row
        self.choice: List[Tuple[str, Callable[[Dict[str, Value]], str]]] = []  # a key asking which
        self.ci = 0  # the choice under the cursor
        self._options: List[Option] = []
        self._vis = VISIBLE  # options the open list has room to show
        self._auto = False  # the open list was opened by the card, not by a key
        self.sel = len(self.view.fields)
        if form.browse:
            self.sel = 0
        elif any(self._empty(f) and not f.warn for f in self.view.fields):
            self._advance(start=-1)  # straight into the first thing still to choose
        # else on the action: a question that needs a yes waits until it runs

    # -- state ----------------------------------------------------------------

    @property
    def rows(self) -> int:
        return len(self.view.fields) + (0 if self.form.live else 1)

    @property
    def field(self) -> Optional[Field]:
        fs = self.view.fields
        return fs[self.sel] if self.sel < len(fs) else None

    @staticmethod
    def _empty(f: Field) -> bool:
        return (f.required and f.value in (None, "")) or (f.warn and not f.value)

    def _set(self, key: str, value: Value) -> None:
        if value in (None, "", False):
            self.form.values.pop(key, None)
        else:
            self.form.values[key] = value
        self.view = self.form.build(dict(self.form.values))
        if self.form.live is not None:
            self.form.live(dict(self.form.values))
        keys = [f.key for f in self.view.fields]
        # rows can come and go (a warning appears): stay on the same key
        self.sel = keys.index(key) if key in keys else min(self.sel, self.rows - 1)

    def _add(self, key: str, value: Value) -> None:
        """A new row, kept even while its value is still empty."""
        self.form.values[key] = value
        self.view = self.form.build(dict(self.form.values))
        keys = [f.key for f in self.view.fields]
        self.sel = keys.index(key) if key in keys else self.sel

    def options(self) -> List[Option]:
        f = self.field
        if f is None or f.options is None:
            return []
        q = self.buf.strip().lower()
        opts = [o for o in self._options if all(
            part in " ".join((o.label, o.note, o.group, o.words)).lower() for part in q.split())]
        if not (f.free and q):
            return opts
        named = [o for o in opts if all(part in o.label.lower() for part in q.split())]
        rest = [o for o in opts if o not in named]  # the words matched only a note: "3" in "2,356"
        if not any(str(o.value or "").lower() == q or o.label.lower() == q for o in opts):
            named.append(Option(self.buf.strip(), f'use "{self.buf.strip()}"', f.hint))
        return named + rest

    # -- keys -----------------------------------------------------------------

    def drop(self, text: str) -> None:
        """A file dragged into the terminal (or a pasted path): a data file
        becomes the data, a model folder the model; other text is typed."""
        path = dropped_path(text)
        keys = {f.key for f in self.view.fields}
        if path is not None:
            is_model = path.is_dir() and (path / "config.json").is_file()
            f = self.field
            key = "subject" if is_model else f.key if f is not None and f.files else "--data"
            if key in keys:
                self.mode, self.buf = "nav", ""
                self._set(key, str(path))
                self.note = f"{path.name} dropped in"
                return
        if self.mode in ("pick", "text"):
            self.buf += text.replace("\n", " ")

    def takes_slash(self) -> bool:
        """A ``/`` typed here means a command, not a setting: on the rows, or
        in a list that takes no typed value. A path being typed keeps it."""
        if self.mode == "nav":
            return True
        return self.mode == "pick" and not self.buf and not self.field.free

    def key(self, kind: str, ch: str = "") -> Optional[Outcome]:
        self.note = ""
        if self.choice:
            return self._choice_key(kind, ch)
        if self.mode == "nav" and kind in self.form.keys:
            do = self.form.keys[kind][1]
            if callable(do):
                self.note = do(dict(self.form.values))
            else:
                self.choice, self.ci = list(do), 0
            return None
        if self.mode == "pick":
            return self._pick_key(kind, ch)
        if self.mode == "text":
            return self._text_key(kind, ch)
        return self._nav_key(kind, ch)

    def _choice_key(self, kind: str, ch: str) -> None:
        """A key that asks which: ←→ move, ⏎ or the first letter does it,
        esc asks nothing."""
        n = len(self.choice)
        letters = [label[0].lower() for label, _ in self.choice]
        if kind in ("right", "down", "tab"):
            self.ci = (self.ci + 1) % n
        elif kind in ("left", "up"):
            self.ci = (self.ci - 1) % n
        elif kind == "enter" or (kind == "char" and ch.lower() in letters):
            at = letters.index(ch.lower()) if kind == "char" else self.ci
            do = self.choice[at][1]
            self.choice = []
            self.note = do(dict(self.form.values))
        elif kind in ("esc", "ctrl-c") or kind in self.form.keys:
            self.choice = []
        return None

    def _open(self) -> None:
        f = self.field
        if f is None:
            return
        self.problem = ""
        if f.kind == "toggle":
            self._set(f.key, not bool(f.value))
            if f.warn and self.form.values.get(f.key):
                self._advance(start=self.sel)  # answered: on to what is left, or the action
            return
        if f.kind == "text":
            self.mode, self.buf = "text", "" if f.value in (None, True) else str(f.value)
            return
        if f.inline:
            self._cycle(f, 1)
            return
        self.mode, self.buf, self.top, self._auto = "pick", "", 0, False
        self._options = f.options() if f.options else []
        self.opt = next((i for i, o in enumerate(self._options) if o.value == f.value), 0)

    def _nav_key(self, kind: str, ch: str) -> Optional[Outcome]:
        f = self.field
        if kind == "up":
            self.sel = (self.sel - 1) % self.rows
        elif kind in ("down", "tab"):
            self.sel = (self.sel + 1) % self.rows
        elif kind in ("esc", "ctrl-c") and self.form.live is not None:
            return "go"  # every change is already kept
        elif kind == "esc" and f is not None:
            self.sel = len(self.view.fields)  # back to the action
        elif kind in ("esc", "ctrl-c"):
            return "cancel"
        elif kind == "char" and ch != " " and self._search_row() is not None:
            self.sel = self._search_row()  # typing searches every setting
            self._open()
            return self.key(kind, ch)
        elif kind == "char" and ch != " " and f is not None and f.kind != "toggle" and not f.inline:
            self._open()  # typing on a row starts changing it
            return self.key(kind, ch)
        elif kind in ("left", "right") and f is not None and f.kind == "pick" and not f.adds:
            self._cycle(f, 1 if kind == "right" else -1)
        elif kind == "enter" or ch == " ":
            if self.field is None:
                return "go" if self.view.ready else self._to_first_empty()
            self._open()
        elif kind in ("backspace", "delete") and self.field is not None:
            f = self.field
            if not f.required and not f.warn:
                for key in f.clears:
                    self.form.values.pop(key, None)
                self._set(f.key, None)  # back to its default
        return None

    def _cycle(self, f: Field, step: int) -> None:
        """←→ on a pick row: the next option, without opening the list."""
        opts = f.options() if f.options else []
        if not opts:
            return
        i = next((i for i, o in enumerate(opts) if o.value == f.value), -1 if step > 0 else 0)
        self._set(f.key, opts[(i + step) % len(opts)].value)

    def _to_first_empty(self) -> None:
        i = next((i for i, f in enumerate(self.view.fields) if self._empty(f)), None)
        if i is not None:
            self.sel = i
        return None

    def _pick_key(self, kind: str, ch: str) -> Optional[Outcome]:
        opts = self.options()
        if kind == "esc" or (kind == "left" and not self.buf):
            self.mode, self.buf = "nav", ""
            if kind == "esc" and self._auto and not self.form.live:
                self.sel = len(self.view.fields)  # so a second esc leaves
        elif kind == "ctrl-c":
            return "cancel"
        elif kind == "char" and ch == " " and not self.buf:
            return self._pick_key("enter", "")
        elif kind == "up":
            self.opt = (self.opt - 1) % max(1, len(opts))
        elif kind in ("down", "tab"):
            self.opt = (self.opt + 1) % max(1, len(opts))
        elif kind in ("enter", "right"):
            if opts:
                pick = opts[min(self.opt, len(opts) - 1)]
                f = self.field
                self.mode, self.buf = "nav", ""
                if f.adds:  # a new row for the setting picked, opened to set it
                    self._add(pick.value, "" if pick.sets is None else pick.sets)
                    if self.field is not None and self.field.key == pick.value:
                        self._open()
                    return None
                was_empty = self._empty(f)
                self._set(f.key, pick.value)
                self._settle(f.key, was_empty)
        elif kind == "backspace":
            self.buf, self.opt, self.top = self.buf[:-1], 0, 0
        elif kind == "ctrl-u":
            self.buf, self.opt, self.top = "", 0, 0
        elif kind in ("char", "paste") and ch:
            self.buf, self.opt, self.top = self.buf + ch, 0, 0
        n = len(self.options())
        self.opt = min(self.opt, max(0, n - 1))
        if self.opt < self.top:
            self.top = self.opt
        elif self.opt >= self.top + self._vis:
            self.top = self.opt - self._vis + 1
        return None

    def _text_key(self, kind: str, ch: str) -> Optional[Outcome]:
        f = self.field
        if kind == "esc":
            self.mode, self.buf, self.problem = "nav", "", ""
        elif kind == "ctrl-c":
            return "cancel"
        elif kind == "enter":
            text = self.buf.strip()
            problem = f.check(text) if (f.check and text) else None
            if problem:
                self.problem = problem
                return None
            self.mode, self.buf, self.problem = "nav", "", ""
            was_empty = self._empty(f)
            self._set(f.key, text or None)
            self._settle(f.key, was_empty)
        elif kind == "backspace":
            self.buf = self.buf[:-1]
            self.problem = ""
        elif kind == "ctrl-u":
            self.buf = ""
        elif kind in ("char", "paste") and ch:
            self.buf += ch
            self.problem = ""
        return None

    def _settle(self, key: str, was_empty: bool) -> None:
        """After a row is set: filling in a blank goes on to the next one
        still to choose; changing a value already there stays on its row."""
        if was_empty:
            self._advance(start=self.sel)
            return
        keys = [f.key for f in self.view.fields]
        if key in keys:
            self.sel = keys.index(key)

    def _advance(self, start: int) -> None:
        """After a change: on to the next row still to set (opened, if it is
        a choice), else to the action row. Rows after ``start`` only, so
        moving back to fix one row never jumps away from it."""
        fs = self.view.fields
        i = next((i for i, f in enumerate(fs) if i > start and self._empty(f)), None)
        if i is None and not any(self._empty(f) for f in fs):
            self.sel = len(fs)
        elif i is not None:
            self.sel = i
            if fs[i].kind == "pick" and not fs[i].warn:
                self._open()
                self._auto = True  # opened for you: esc steps out to the action

    # -- drawing --------------------------------------------------------------

    def render(self, w: int) -> List[Line]:
        """The approved card: title, rows, status, the action, one line under
        it, the keys. The focused row (or the option under the cursor) sits
        on the band; an unfocused action recedes."""
        w = min(w, c.CAP)
        v = self.view
        title = [S(v.title, "hi")] if isinstance(v.title, str) else list(v.title)
        out: List[Line] = [c.titled(ln(S("● ", "tx"), *title)), c.rail()]
        labels = [self._label(f) for f in v.fields] + [t[0] for t in v.facts]
        lw = max([10] + [len(label) + 2 for label in labels])
        # one value column for the card's whole life: choosing a long value
        # never pushes the notes over (a longer one is cut, its end kept)
        vw = max(12, min(VALUE + GAP, w - len(c.RAIL) - lw - 8))
        anchor = 0  # the focused row's line: an open list draws from under it
        for i, f in enumerate(v.fields):
            if i and ((f.warn and not v.fields[i - 1].warn) or f.gap):
                out.append(c.rail())
            if i == self.sel:
                anchor = len(out)
            out += self._row(f, i == self.sel, lw, vw, w)
        status: List[Line] = []
        if v.result:
            label, value, detail = v.result
            bar: Line = []
            room = w - len(c.RAIL) - lw - len(value) - 2
            # the share first, as on /machine; the bar narrows before the note is cut
            cells = max(METER_MIN, min(BAR, room - len(detail) - 4))
            if v.meter is not None and room - cells - 2 >= 10:
                share, fill = v.meter
                bar = c.chips(share, cells, fill, lit=True) + [S("  ", None)]  # its one result
                room -= cells + 2
            status.append(c.rail(S(label.ljust(lw), "dm"), *bar, S(value, v.result_style),
                                 S("  " + _fit(detail, room), "dm") if detail else None))
        for label, what, where in v.facts:  # what running it gives, read-only
            if where and w - len(c.RAIL) - lw - vw >= 12 and len(what) < vw:
                what = what.ljust(vw)  # its note in the rows' note column
            elif where:
                what += "  "
            room = min(TAIL, w - len(c.RAIL) - lw - len(what))
            status.append(c.rail(S(label.ljust(lw), "dm"), S(what, "tx"),
                                 S(_fit(where, room), "rl") if where else None))
        if status:
            out += [c.rail()] + status
        out.append(c.rail())
        on = self.sel == len(v.fields) and self.mode == "nav"
        if self.form.live is None:
            go = [S(v.go, "hi" if on and v.ready else "dm")]
            if not v.ready:
                go.append(S("  " + self._missing(), "rl"))
            out.append(ln(S("  ❯ " if on else c.RAIL, "a" if on else "rl"), *go))
        out.append([])
        out += self._under(on, w)
        out += c.keyline(self._hints(), w)
        if self.mode == "pick" and self.field is not None:
            out = self._overlay(out, anchor, lw, vw, w)
        return [c.inked(line) for line in out]

    def _overlay(self, out: List[Line], anchor: int, lw: int, vw: int, w: int) -> List[Line]:
        """An open list draws over the lines below its row, like a dropdown:
        the card keeps its height and nothing under the list moves. Only a
        card too short to hold a few options grows, by what they need."""
        foot = out[-2:]
        if self.form.browse and not self.field.adds:
            # settings: every row is a setting you came to read, so the
            # list opens between rows and the ones under it move down
            lines = self._list(self.field, lw, vw, w, VISIBLE)
            return out[:anchor + 1] + lines + out[anchor + 1:-2] + foot
        if self.field.adds:  # every setting: in place of the rows, under the title
            start, need = 2, 4
        else:
            start, need = anchor + 1, 3
        room = max(len(out) - 2 - start, need)
        body = out[:-2] + [[] for _ in range(start + room - (len(out) - 2))]
        lines = (self._browse(w, room) if self.field.adds
                 else self._list(self.field, lw, vw, w, room))
        body[start:start + len(lines)] = lines
        if self.field.adds:  # the rows the list did not cover are not the card now
            body[start + len(lines):start + room] = [[] for _ in range(room - len(lines))]
        return body + foot

    @staticmethod
    def _label(f: Field) -> str:
        return ("! " if f.warn else "") + f.label

    def _under(self, on: bool, w: int) -> List[Line]:
        """Under the action: what an extra key did, why it can't run, what
        the focused row is for, or the command these values run."""
        if self.choice:
            parts: List[S] = [S("  "), S("save as ", "dm")]
            for i, (label, _) in enumerate(self.choice):
                parts += [S("  ")] + ([S("❯ ", "a"), S(label, "hi")] if i == self.ci
                                      else [S("  "), S(label, "dm")])
            return [ln(*parts)]
        if self.note:
            return [ln(S("  "), S("✓ ", "a"), S(_fit(self.note, w - 4), "tx"))]
        if self.view.error:
            return [ln(S("  "), S(line, "e")) for line in c.wrap(self.view.error, w - 4)]
        f = self.field
        if self.mode == "nav" and f is not None and f.hint and not on:
            return [ln(S("  "), S(_fit(f.hint, w - 2), "rl"))]
        if self.view.command:
            return [ln(S("  "), S(_fit(self.view.command, w - 2), "rl"))]
        return [[]]

    def _missing(self) -> str:
        empty = [f for f in self.view.fields if self._empty(f)]
        if not empty:
            return ""
        f = empty[0]
        if f.warn:  # a yes it needs: which row, and what saying it does
            return f"say yes on “{f.label}” to start"
        return f"choose {f.label} first"

    def _hints(self) -> List[Tuple[str, str]]:
        """(key, what it does): the key in ink, ⏎ in the accent."""
        if self.mode == "pick" and self.field.adds:
            return [("⏎", "change"), ("↑↓", "move"), ("esc", "back")]
        if self.mode == "pick":
            return [("⏎", "choose"), ("↑↓", "move"), ("esc", "back")]
        if self.mode == "text":
            return [("⏎", "set " + self.field.label), ("empty", "resets"), ("esc", "back")]
        if self.choice:
            return [("⏎", "save"), ("←→", "move"), ("esc", "back")]
        f = self.field
        extra = [(c.key_name(k), h) for k, (h, _) in self.form.keys.items()]
        if f is None:
            up = [("↑", "change")] if self.view.fields else []
            return [("⏎", self.view.go.lower()), *up, *extra, ("esc", "cancel")]
        hints = [("⏎", ("no" if f.value else "yes") if f.warn
                  else "switch" if f.kind == "toggle"
                  else "all settings" if f.adds else "change")]
        if f.kind == "pick" and not f.adds:
            hints.append(("←→", "cycle"))
        if self._search_row() is not None:
            hints.append(("type", "to find any setting"))
        if self.form.live is not None:
            return hints + [("↑↓", "move"), ("esc", "close")]
        return hints + [("esc", "done")]

    def _search_row(self) -> Optional[int]:
        """The row that searches every setting (the "more" row), if there is one."""
        return next((i for i, f in enumerate(self.view.fields) if f.adds), None)

    def _row(self, f: Field, sel: bool, lw: int, vw: int, w: int) -> List[Line]:
        on = sel and self.mode != "pick"  # an open list takes the focus to its options
        gutter = S("  ❯ ", "a") if on else S(c.RAIL, "rl")
        label = S(self._label(f).ljust(lw), "hi" if on else "dm")
        strong = f.bold or (f.value not in (None, "", False) and not f.quiet)  # chosen
        text = f.shown or (_display(f.value) if f.kind == "toggle" or f.value else "")
        text = _fit_path(text, vw - 2) if text[:1] in "~/" else _fit(text, vw - 2)
        typing = sel and (self.mode == "text" or (self.mode == "pick" and self.buf))
        if typing:  # what is typed, the caret, and the old value until it is replaced
            value = [S(self.buf, "hi"), S("▏", "a"), S("" if self.buf else text, "rl")]
            value.append(S(" " * max(0, vw - width(value)), None))
        elif sel and self.mode == "pick" and self.form.browse:
            value = [S(" " * vw, None)]  # the list open under the row says it: said once
        elif f.inline:
            value = self._choices(f, on, vw)
        elif self._empty(f):
            value = [S(("not yet: ⏎ for yes" if f.warn else "choose").ljust(vw),
                       "a" if on else "dm")]
        else:
            value = [S(text.ljust(vw), "hi" if strong else "tx")] if text else [
                S("—".ljust(vw), "dm")
            ]
        room = min(TAIL, w - len(c.RAIL) - lw - vw)
        if sel and self.mode == "text" and self.problem:
            tail = [S(_fit(self.problem, room), "e")]
        elif f.warn:  # the question itself, always
            tail = [S(_fit(f.source, room), "tx" if on else "dm")] if room >= 8 else []
        elif typing or (on and self._empty(f)):
            # on the rows the line under the card says it whole: said once
            tail = [S(_fit(f.hint, room), "rl")] if f.hint and typing else []
        elif room >= 8:
            src = _plain(f.source)
            tail = [S(_fit(f.fact, room), "dm")] if f.fact else []  # measured: a step up
            if src and room - len(f.fact) - 3 >= 8:  # from the reason it was picked
                gap = " · " if f.fact else ""
                tail += [S(gap, "rl"), S(_fit(src, room - len(f.fact) - len(gap)), "rl")]
        else:
            tail = []
        row = _clip(ln(gutter, label, *value, *tail), w)
        return [banded(row, w) if on else row]

    def _browse(self, w: int, room: int) -> List[Line]:
        """All settings: a search line, then the settings under their group,
        each with what it trains at now and what changing it does."""
        opts = self.options()
        q = self.buf
        said = ((f"   {len(opts)} setting{'s' if len(opts) != 1 else ''}" if opts
                 else "   nothing called that") if q
                else "type what you want to change · ↑↓ to browse")
        out = [c.rail(S("search  ", "dm"), S(q, "hi"), S("▏", "a"), S(said, "rl")), c.rail()]
        every = self._options  # columns from every setting: a search never moves them
        nw = min(16, max([10] + [len(o.label) for o in every])) + 2
        vw = min(12, max([4] + [len(o.now) for o in every])) + 2
        rows: List[Tuple[Optional[int], Line]] = []  # (option index or a heading, line)
        group = None
        for i, o in enumerate(opts):
            if o.group and o.group != group:
                if rows:
                    rows.append((None, c.rail()))
                rows.append((None, c.rail(S(o.group, "dm"))))
                group = o.group
            on = i == self.opt
            note = w - len(c.RAIL) - 2 - nw - vw
            row = _clip(ln(S("  ❯ " if on else c.RAIL, "a" if on else "rl"), S("  ", None),
                           S(_fit(o.label, nw - 2).ljust(nw), "hi" if on else "tx"),
                           S(_fit(o.now, vw - 2).ljust(vw), "tx" if on else "dm"),
                           *([S(_fit(o.note, note), "dm" if on else "rl")] if note >= 8 else [])),
                        w)
            rows.append((i, banded(row, w) if on else row))
        n = max(3, room - 3)
        at = next((k for k, (i, _) in enumerate(rows) if i == self.opt), 0)
        top = max(0, min(at - n + 2, len(rows) - n))
        if top and rows[top - 1][0] is None and at - top < n - 1:
            top -= 1  # keep the heading of the group the cursor is in
        out += [line for _, line in rows[top:top + n]]
        if len(rows) > n:
            out.append(c.rail(S(f"{self.opt + 1} of {len(opts)}", "rl")))
        return out

    def _choices(self, f: Field, on: bool, vw: int) -> List:
        """``● merged   ○ adapter``: the choice in ink, the others receding."""
        out: List = []
        used = 0
        for o in f.options() if f.options else []:
            chosen = o.value == f.value
            seg = ("● " if chosen else "○ ") + o.label + "   "
            out += [S(seg[:2], "a" if chosen and on else ("hi" if chosen else "rl")),
                    S(seg[2:], ("hi" if on else "tx") if chosen else "dm")]
            used += len(seg)
        return out + [S(" " * max(0, vw - used), None)]

    def _list(self, f: Field, lw: int, vw: int, w: int, room: int) -> List[Line]:
        """The options under their row: ● the current value, ○ the others,
        the one under the cursor on the band."""
        opts = self.options()
        pad = S(" " * lw, None)
        if not opts:
            return [c.rail(pad, S("nothing matches", "dm"))]
        n = min(VISIBLE, len(opts) if len(opts) <= room else room - 1)
        self._vis = max(1, n)
        self.top = max(0, min(self.top, self.opt, len(opts) - self._vis))
        if self.opt >= self.top + self._vis:
            self.top = self.opt - self._vis + 1
        shown = opts[self.top : self.top + self._vis]
        # names come first: a note gives way (whole parts) before a name is cut
        ow = min(48, w - len(c.RAIL) - lw - 4, max([vw - 2] + [len(o.label) + 2 for o in opts]))
        room = w - len(c.RAIL) - lw - 2 - ow
        out = []
        for i, o in enumerate(shown):
            on = self.top + i == self.opt
            note = _fit(o.note, room) if o.note and room >= 4 else ""
            row = _clip(c.rail(pad, S("● " if o.value == f.value else "○ ", "a" if on else "rl"),
                               S(_fit(o.label, ow - 2).ljust(ow), "hi" if on else "tx"),
                               S(note, "rl") if note else None), w)
            out.append(banded(row, w) if on else row)
        if len(opts) > len(shown):
            out.append(c.rail(S(" " * (lw + 2), None),
                              S(f"{self.top + 1}–{self.top + len(shown)} of {len(opts)}", "rl")))
        return out


def _plain(source: str) -> str:
    """Where a value came from, said only when it says something: "given"
    and "default" go unsaid, "inferred · one epoch" is just the reason."""
    return " · ".join(p for p in source.split(" · ") if p not in ("given", "default", "inferred"))


def _display(value: Value) -> str:
    if value is True:
        return "on"
    if value in (None, False):
        return "off"
    return str(value)


def _fit_path(path: str, room: int) -> str:
    """A path keeps both ends (where it is, what it is called): ``~/…/export/merged``."""
    if len(path) <= room or room < 8:
        return _fit(path, room)
    tail = path[-(room - 3):]
    tail = tail[tail.find("/"):] if "/" in tail else tail
    return path[:2] + "…" + tail if len(path[:2]) + 1 + len(tail) <= room else _fit(path, room)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def run(form: Form, *, stream=None) -> Optional[Dict[str, Value]]:
    """Show ``form`` until go (its values) or cancel (None). Drawn in place,
    erased afterwards: scrollback keeps only what the command prints."""
    with fullscreen(stream):
        return dict(form.values) if _loop(Sheet(form), stream) == "go" else None


def dropped_path(text: str):
    """The path in a terminal's drop: quoted, or with backslash-escaped
    spaces, or a file:// URL. None unless it names something that exists."""
    import shlex
    from pathlib import Path
    from urllib.parse import unquote

    raw = text.strip()
    if raw.startswith("file://"):
        raw = unquote(raw[7:])
    try:
        words = shlex.split(raw)
    except ValueError:
        words = [raw]
    if len(words) != 1:
        return None
    path = Path(words[0]).expanduser()
    return path if path.exists() else None


# -- full screen ---------------------------------------------------------------
# A card is a moment: it opens on the terminal's second screen, centred on a
# clean canvas, and closing it brings back the scrollback exactly as it was.
# What it settled on is told there once, as one line (``live.retell``).
# Chained cards (a recipe opening the train card) share one canvas.


# --------------------------------------------------------------------------
# the list card: /models, /runs
# --------------------------------------------------------------------------
