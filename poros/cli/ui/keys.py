"""Keys off a terminal in cbreak mode: ``read_key`` (one logical key as (kind, char), escape
sequences and a Mac's option characters decoded) and ``_read_paste`` (a bracketed paste). The one
key decoder: the session, a sheet's frame and the choice menu read keys here.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

from poros.cli.ui import components as c


def read_key(fd: int) -> Tuple[str, str]:
    """One logical key as (kind, char)."""
    import select

    try:
        b = os.read(fd, 1)
    except KeyboardInterrupt:
        return ("ctrl-c", "")
    if not b:
        return ("eof", "")
    o = b[0]
    if b in (b"\r", b"\n"):
        return ("enter", "")
    if b == b"\t":
        return ("tab", "")
    if b in (b"\x7f", b"\x08"):
        return ("backspace", "")
    if o < 32 and b != b"\x1b":
        return (
            {
                1: "ctrl-a",
                3: "ctrl-c",
                4: "ctrl-d",
                5: "ctrl-e",
                6: "ctrl-f",
                7: "ctrl-g",
                11: "ctrl-k",
                12: "ctrl-l",
                14: "ctrl-n",
                16: "ctrl-p",
                18: "ctrl-r",
                19: "ctrl-s",
                20: "ctrl-t",
                21: "ctrl-u",
                23: "ctrl-w",
                24: "ctrl-x",
            }.get(o, "other"),
            "",
        )
    if b == b"\x1b":
        r, _, _ = select.select([fd], [], [], 0.03)
        if not r:
            return ("esc", "")
        nxt = os.read(fd, 1)
        if nxt.isdigit():  # alt+1..9: a tab
            return ("alt", nxt.decode())
        if nxt in (b"b", b"f") and os.environ.get("TERM_PROGRAM") == "Apple_Terminal":
            # Terminal.app sends option+←/→ as esc b / esc f (back and forward a
            # word): steps between tabs there; its option+f types ƒ (below)
            return ("ctrl-left" if nxt == b"b" else "ctrl-right", "")
        if nxt.decode("latin-1").lower() in c.APP_LETTERS:  # alt+s is ctrl+s: one action
            return ("ctrl-" + nxt.decode("latin-1").lower(), "")
        if nxt in (b"\r", b"\n"):  # alt+enter: the command in a new tab
            return ("shift-enter", "")
        if nxt not in (b"[", b"O"):
            return ("other", "")
        seq = b""
        while True:
            r, _, _ = select.select([fd], [], [], 0.03)
            if not r:
                break
            ch = os.read(fd, 1)
            seq += ch
            if ch.isalpha() or ch == b"~":
                break
        return (
            {
                b"A": "up",
                b"B": "down",
                b"C": "right",
                b"D": "left",
                b"H": "home",
                b"F": "end",
                b"1~": "home",
                b"4~": "end",
                b"3~": "delete",
                b"5~": "page-up",
                b"6~": "page-down",
                b"1;5C": "ctrl-right",
                b"1;5D": "ctrl-left",
                b"1;3C": "ctrl-right",  # alt+arrows, where ctrl+arrows move spaces (macOS)
                b"1;3D": "ctrl-left",
                b"1;2C": "ctrl-right",  # shift+arrows: free on every Mac terminal
                b"1;2D": "ctrl-left",
                b"200~": "paste",
                b"13;2u": "shift-enter",  # where the terminal tells shift+enter apart
                b"27;2;13~": "shift-enter",
                b"?997;1n": "theme",  # the terminal went dark (mode 2031)
                b"?997;2n": "theme",  # ... or light
            }.get(seq, "other"),
            "",
        )
    if o >= 0x80:  # utf-8 multibyte
        n = 2 if o >> 5 == 0b110 else 3 if o >> 4 == 0b1110 else 4
        b += os.read(fd, n - 1)
    try:
        ch = b.decode()
    except UnicodeDecodeError:
        return ("other", "")
    if ch in OPTION_DIGITS:  # option+1..9 on a Mac that doesn't send it as alt
        return ("alt", str(OPTION_DIGITS.index(ch) + 1))
    if ch in OPTION_LETTERS and sys.platform == "darwin":
        # option+letter where the terminal types its character (Terminal.app
        # unless "Use Option as Meta key" is on): the same action as ⌥ sent as alt
        return ("ctrl-" + OPTION_LETTERS[ch], "")
    return ("char", ch)


# What option+1..9 types on a Mac (US, UK and most Latin layouts) when the
# terminal keeps option for characters instead of sending alt.
OPTION_DIGITS = "¡™£¢∞§¶•ª"


# What option+letter types there (US layout) for the app's letters. E and N
# are dead keys: the terminal holds them for the next key, so ⌥E and ⌥N need
# option sent as alt (Ghostty: macos-option-as-alt; Terminal.app: Use Option
# as Meta key), or a space after them.
OPTION_LETTERS = {"´": "e", "ƒ": "f", "©": "g", "˜": "n", "π": "p", "®": "r", "ß": "s",
                  "†": "t", "∑": "w", "≈": "x"}


def _read_paste(fd: int) -> str:
    data = b""
    while not data.endswith(b"\x1b[201~"):
        ch = os.read(fd, 1)
        if not ch:
            break
        data += ch
    return (
        data[: -len(b"\x1b[201~")].decode("utf-8", "replace")
        if data.endswith(b"\x1b[201~")
        else data.decode("utf-8", "replace")
    )
