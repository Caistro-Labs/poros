"""`poros machine` and the machine card: this machine's GPU, memory and stack, what to fix,
and the live GPU gauges (pure functions from data to ``Line`` lists)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from poros.cli.ui import components as c
from poros.cli.ui.text import Line, S, Seg, ln, width
from poros.cli.views.parts import _needs_gpu, _w

# The packages training needs. One install brings all of them, so however
# many are missing they are one thing to fix, said as torch's.
_ML = ("torch", "peft", "bitsandbytes", "accelerate", "transformers")


_INFO = ("Python", "Platform", "Poros", "Validated archs", "Disk free", "CUDA")


_ML_FIX = "pip install poros-train"


def _problems(r: Any) -> List[Any]:
    return [ch for ch in r.checks if ch.status in ("warn", "error")]


def _to_fix(r: Any) -> int:
    """What the card asks you to fix, counted as the card shows it: the
    training stack as one install, Python and disk where they say why."""
    shown = set(_INFO) - {"Python", "Disk free"}
    probs = [ch for ch in _problems(r) if ch.name not in shown]
    return any(ch.name in _ML for ch in probs) + sum(ch.name not in _ML for ch in probs)


def _fix_names(r: Any) -> str:
    """What ``_to_fix`` counts, by name: "the training stack and disk space"."""
    shown = set(_INFO) - {"Python", "Disk free"}
    names: List[str] = []
    for ch in _problems(r):
        if ch.name in shown:
            continue
        name = ("the training stack" if ch.name in _ML else
                "disk space" if ch.name.startswith("Disk ") else
                {"Python": "a newer Python"}.get(ch.name, ch.name))
        if name not in names:
            names.append(name)
    if len(names) > 2:
        return f"{len(names)} to fix"
    return "needs " + " and ".join(names) if names else ""


def machine(r: Any, w: int, art: bool = True) -> List[Line]:
    """``poros machine``: one card. The verdict; the graphics card drawn with
    what it is doing now; the host and the training stack, said only as far
    as they matter for training; then the command to run next. ``art`` False (stdout is a pipe or a
    file): the same card in rows, with no drawing."""
    from poros.cli.ui import gpu_art

    if not art:
        lines = _machine_rows(r, w)
        return [c.titled(lines[0]), *(c.inked(line) for line in lines[1:])]
    if r.gpu is not None and r.gpu.unified:  # a Mac: its chip, where the terminal shows images
        try:
            chip = machine_chip(r, w)
        except Exception:  # noqa: BLE001 -- a drawing bug must not cost the result: the rows
            chip = None
        if chip is not None:
            return [c.titled(chip[0]), *(c.inked(line) for line in chip[1:])]
    drawn = None
    card = r.gpu is not None and not r.gpu.unified  # a Mac's chip is no card to draw
    if card and _w(w) >= _SHOW_W:
        drawn = machine_drawn(r, w)
    # the drawing must fit whole, or its title scrolls out of view
    if (not drawn or len(drawn) > _show_room()) and card and _w(w) >= gpu_art.W + 38:
        drawn = machine_side(r, w)
    lines = drawn if drawn and len(drawn) <= _show_room() else _machine_rows(r, w)
    return [c.titled(lines[0]), *(c.inked(line) for line in lines[1:])] if lines else lines


def machine_head(r: Any, w: int) -> Line:
    n = _to_fix(r)
    if r.verdict == "not ready":
        why = f"training {_needs_gpu()}" if r.gpu is None else f"{n} to fix"
        return c.error_event("Not ready", why, _w(w))[0]
    if not n:
        return ln(S("● ", "tx"), S("Ready to train", "hi"))  # the card's ▛▜ goes on the dot
    ml = any(ch.name in _ML for ch in _problems(r))
    why = ("the training stack is one install away" if ml and n == 1 else
           _fix_names(r) or f"{n} to fix")
    return ln(S("● ", "tx"), S("Almost ready", "hi"), S(f"  ·  {why}", "dm"))


def machine_card(name: str, w: int, angle: float = 0.0, side: Sequence[Line] = (),
                 image: bool = False, maker: str = "", ident: int = 0) -> List[Line]:
    """The drawn card, indented, with ``side`` beside it row for row. With
    ``image`` the card is a real image laid over empty cells (gpu_image)."""
    from poros.cli.ui import gpu_art

    if image:
        import sys

        from poros.cli.ui import gpu_image
        from poros.cli.ui.style import resolve

        proto = gpu_image.protocol() or "kitty"
        seg, aw, ah, at = gpu_image.placed(name, resolve(sys.stdout), maker, proto, ident=ident)
        art: List[Line] = [[(" " * aw, None)] for _ in range(ah)]
        art[at] = [(seg, "raw"), *art[at]]
    else:
        art = gpu_art.card(name, angle, maker)
        aw, ah = gpu_art.size(gpu_art.form(name, maker))
    top = max(0, (ah - len(side)) // 2)  # the text sits level with the card
    side = [[]] * top + list(side)
    out = []
    for i in range(max(len(art), len(side))):
        row = art[i] if i < len(art) else []
        right = side[i] if i < len(side) else []
        pad = aw - width(row)
        out.append([("  ", None), *row, (" " * (pad + 4), None), *right] if right
                   else [("  ", None), *row])
    return out


def _gauge(label: str, frac: Optional[float], value: str, aside: str = "", *,
           lw: int, n: int, style="tx") -> Line:
    """``label  ▰▰▰▱▱▱▱▱  value aside``: one reading as a share of its whole."""
    return ln(S(label.ljust(lw), "dm"), *c.chips(frac, n, style), S("  " + value, "tx"),
              S(aside, "dm") if aside else None)


def _vram(r: Any) -> Tuple[float, Optional[float]]:
    """The card's memory as every row of the screen says it: (its total, what is in use), GiB.
    One total: the card's own as CUDA gives it to a run (``gpu.total_gib``, what the plan
    reads), never nvidia-smi's larger figure beside it. In use: what nvidia-smi says is used
    now, else the total less the free memory the card had when it was found."""
    gpu, now = r.gpu, getattr(r, "now", None)
    total = gpu.total_gib
    if now is not None and now.used_gib is not None:
        return total, min(total, now.used_gib)
    return total, (max(0.0, total - gpu.free_gib) if gpu.free_gib is not None else None)


def _gpu_gauges(r: Any, room: int, lw: int) -> List[Line]:
    """The card right now: memory in use, load, temperature, power, one gauge
    each so they read at a glance; memory, what training needs, in the accent."""
    now = getattr(r, "now", None)
    n = max(6, min(16, room - lw - 2 - 22))
    total, used = _vram(r)
    out: List[Line] = []
    if used is not None and total:
        out.append(_gauge("vram", used / total, f"{total - used:.1f} GiB free",
                          f" of {total:.1f}", lw=lw, n=n, style="a"))
    else:
        out.append(_gauge("vram", None, f"{total:.1f} GiB", lw=lw, n=n))
    if now is None:
        return out
    if now.util_pct is not None:
        out.append(_gauge("load", now.util_pct / 100, f"{now.util_pct}%", lw=lw, n=n))
    if now.temp_c is not None:
        hot = now.temp_c >= 83
        out.append(_gauge("temp", now.temp_c / 90, f"{now.temp_c}°C",
                          "  hot" if hot else "", lw=lw, n=n, style="e" if hot else "tx"))
    if now.power_w is not None:
        cap = now.power_limit_w
        out.append(_gauge("power", now.power_w / cap if cap else None, f"{now.power_w:.0f} W",
                          f" of {cap:.0f}" if cap else "", lw=lw, n=n))
    return out


def _gpu_now(r: Any, room: int, lw: int) -> List[Line]:
    """Who is on the card: a Poros run first, then anything else holding memory."""
    now = getattr(r, "now", None)
    busy = list(getattr(r, "busy", []) or [])
    apps = list(now.apps) if now is not None else []

    def row(*parts: Seg) -> Line:
        return ln(S("now".ljust(lw), "dm"), *parts)

    if busy:
        b = busy[0]
        prog = f" · step {b.step:,} of {b.total:,}" if b.step is not None and b.total else ""
        more = f" · {len(busy) - 1} more" if len(busy) > 1 else ""
        return [row(S("● ", "a"), S(_clip_text(b.name, room - lw - 2), "hi")),
                ln(S(" " * lw), S(_clip_text(b.state + prog + more, room - lw), "dm"))]
    if apps:
        a = max(apps, key=lambda x: x.used_gib or 0)
        held = f"  {a.used_gib:.1f} GiB" if a.used_gib is not None else ""
        others = f" · {len(apps) - 1} more" if len(apps) > 1 else ""
        return [row(S(_clip_text(f"{a.name} (pid {a.pid})", room - lw - len(held)), "tx"),
                    S(held, "dm")),
                ln(S(" " * lw), S("not Poros" + others, "dm"))]
    return [row(S("idle", "tx"), S("  nothing is running on it", "dm"))]


# Showcase: the rows around the image, and how wide it may be drawn
_SHOW_AROUND = 19


_SHOW_COLS = 56


_SHOW_LEAST = 30


_SHOW_W = 76


def _show_room() -> int:
    """Rows Showcase has: it must be seen whole, verdict first, so outside the
    session too it takes no more than the window."""
    import shutil

    return min(_room(), shutil.get_terminal_size((100, 40)).lines - 2)


def show_size(name: str, maker: str, proto: Optional[str]) -> Optional[int]:
    """How wide the biggest card is drawn so the screen fits its room; None
    if it can't. Without images the braille drawing has one size."""
    from poros.cli.ui import gpu_art, gpu_image

    room = _show_room() - _SHOW_AROUND
    if not proto:
        return _SHOW_COLS if gpu_art.size(gpu_art.form(name, maker))[1] <= room else None
    for biggest in range(_SHOW_COLS, _SHOW_LEAST - 1, -2):
        if gpu_image.rows(name, maker, proto=proto, biggest=biggest) <= room:
            return biggest
    return None


def machine_show(name: str, w: int, angle: float = 0.0, maker: str = "", image: bool = False,
                 ident: int = 0, biggest: int = _SHOW_COLS) -> List[Line]:
    """The card large and centred: the image, or braille where there is none."""
    from poros.cli.ui import gpu_art

    if image:
        import sys

        from poros.cli.ui import gpu_image
        from poros.cli.ui.style import resolve

        proto = gpu_image.protocol() or "kitty"
        seg, aw, ah, at = gpu_image.placed(name, resolve(sys.stdout), maker, proto, ident=ident,
                                           biggest=biggest)
        art: List[Line] = [[(" " * aw, None)] for _ in range(ah)]
        art[at] = [(seg, "raw"), *art[at]]
    else:
        art = gpu_art.card(name, angle, maker)
        aw = gpu_art.size(gpu_art.form(name, maker))[0]
    # rounds right: the drawing's own margins leave it a little left of the text's axis
    pad = (" " * max(0, (_w(w) - aw + 1) // 2), None)
    return [[pad, *row] for row in art]


def _centred(line: Line, ww: int) -> Line:
    return [(" " * max(0, (ww - width(line)) // 2), None), *line]


def _tiles(r: Any, ww: int) -> List[Line]:
    """Four readings under the card: what it is doing, read at a glance."""
    now = getattr(r, "now", None)
    total, used = _vram(r)
    tiles: List[Tuple[str, Line, Optional[float], str, bool]] = []
    if used is not None and total:
        tiles.append(("vram", [S(f"{total - used:.1f} GiB", "hi"), S(" free", "dm")],
                      used / total, "a", True))
    else:
        tiles.append(("vram", [S(f"{total:.1f} GiB", "hi")], None, "tx", False))
    temp = now.temp_c if now else None
    util = now.util_pct if now else None
    power = now.power_w if now else None
    cap = now.power_limit_w if now else None
    tiles.append(("load", [S(f"{util}%", "hi")] if util is not None else [S("–", "dm")],
                  util / 100 if util is not None else None, "tx", False))
    hot = temp is not None and temp >= 83
    tiles.append(("temp", ([S(f"{temp}°C", "e" if hot else "hi")] if temp is not None
                           else [S("–", "dm")]),
                  temp / 90 if temp is not None else None, "e" if hot else "tx", False))
    tiles.append(("power", ([S(f"{power:.0f} W", "hi")] if power is not None else [S("–", "dm")]),
                  power / cap if power is not None and cap else None, "tx", False))
    tw = min(23, (ww - 4) // 4 - 1)

    def row(cell) -> Line:
        out: Line = [("    ", None)]
        for t in tiles:
            part = cell(t)
            out += [*part, (" " * (tw + 1 - width(part)), None)]
        return out

    return [row(lambda t: [S(t[0], "dm")]), row(lambda t: t[1]),
            row(lambda t: c.chips(t[2], tw - 2, t[3], lit=t[4]))]


def _now_row(r: Any, row) -> Line:
    """Who is on the card, in one row."""
    now = getattr(r, "now", None)
    busy = list(getattr(r, "busy", []) or [])
    apps = list(now.apps) if now is not None else []
    if busy:
        b = busy[0]
        prog = f"  step {b.step:,} of {b.total:,}" if b.step is not None and b.total else ""
        more = f" · {len(busy) - 1} more" if len(busy) > 1 else ""
        return row("now", S("● ", "a"), S(b.name, "hi"), S((prog or f"  {b.state}") + more, "dm"))
    if apps:
        a = max(apps, key=lambda x: x.used_gib or 0)
        held = f"  {a.used_gib:.1f} GiB" if a.used_gib is not None else ""
        return row("now", S(f"{a.name} (pid {a.pid})", "tx"), S(held + "  not Poros", "dm"))
    return row("now", S("idle", "tx"))


def machine_drawn(r: Any, w: int) -> Optional[List[Line]]:
    """Showcase: the verdict, the card large and centred with its name, four
    readings, then the host, the training stack and what fits. None where it
    has no room: the card in rows says the same."""
    from poros.cli.ui import gpu_art, gpu_image

    ww = _w(w)
    gpu = r.gpu
    name = gpu.name.removesuffix(" (simulated)")
    brand = gpu_art.vendor(name)
    shown = ("GeForce " + name if brand == "NVIDIA" and name.split()[0] in ("RTX", "GTX")
             else name)
    value = {ch.name: str(ch.value) for ch in r.checks}
    now = getattr(r, "now", None)
    cuda = value.get("CUDA", "")
    maker = (now.board if now else None) or ""
    proto = gpu_image.protocol()
    biggest = show_size(name, maker, proto)
    if biggest is None:
        return None
    ident = " · ".join(x for x in (
        f"driver {now.driver}" if now and now.driver and now.driver != "simulated" else "",
        f"CUDA {cuda}" if cuda[:1].isdigit() else "",
        f"PCIe {now.pcie}" if now and now.pcie else "") if x)
    edition = gpu_art.label(name, maker)
    out = [machine_head(r, w)]
    out += machine_show(name, w, maker=maker, image=proto is not None, biggest=biggest)
    out.append(_centred([S(shown, "hi"), *([S(f"  {edition}", "dm")] if edition else [])], ww))
    if ident:
        out.append(_centred([S(ident, "dm")], ww))
    out += [[], *_tiles(r, ww), []]
    out += _machine_body(r, ww, show=True)
    return out + _machine_tail(r, ww)


def machine_side(r: Any, w: int) -> List[Line]:
    """For a short window, where Showcase can't fit: the card on the left, what
    it is and its gauges on the right, the rest under it."""
    from poros.cli.ui import gpu_art, gpu_image

    ww = _w(w)
    gpu = r.gpu
    name = gpu.name.removesuffix(" (simulated)")
    brand = gpu_art.vendor(name)
    shown = ("GeForce " + name if brand == "NVIDIA" and name.split()[0] in ("RTX", "GTX")
             else name)
    value = {ch.name: str(ch.value) for ch in r.checks}
    now = getattr(r, "now", None)
    cuda = value.get("CUDA", "")
    maker = (now.board if now else None) or ""
    room = ww - 2 - gpu_art.size(gpu_art.form(name, maker))[0] - 4
    lw = 7
    ident = " · ".join(x for x in (
        f"driver {now.driver}" if now and now.driver and now.driver != "simulated" else "",
        f"CUDA {cuda}" if cuda[:1].isdigit() else "",
        f"PCIe {now.pcie}" if now and now.pcie else "") if x)
    edition = gpu_art.label(name, maker)
    head = _clip_text(shown, room)
    tail = f"  {edition}" if edition and len(head) + 2 + len(edition) <= room else ""
    side: List[Line] = [ln(S(head, "hi"), S(tail, "dm"))]
    if ident:
        side.append(ln(S(_clip_text(ident, room), "dm")))
    side += [[]] + _gpu_gauges(r, room, lw) + [[]] + _gpu_now(r, room, lw)
    out = [machine_head(r, w), []] + machine_card(name, w, 0.0, side, maker=maker,
                                                  image=gpu_image.supported()) + [[]]
    out += _machine_body(r, ww)
    return out + _machine_tail(r, ww)


_CHIP_LEAST = 20  # cells: narrower, the chip's cores no longer read
_chip_ids: dict = {}


def chip_ident(r: Any) -> Tuple[int, int, int]:
    """Kitty image ids per result: its still, and the chip and wash its
    entrance draws with (chip_image.entrance); a later /machine has its own."""
    from poros.cli.ui import gpu_image

    if id(r) not in _chip_ids:
        _chip_ids.clear()
        _chip_ids[id(r)] = (gpu_image.new_id(), gpu_image.new_id(), gpu_image.new_id())
    return _chip_ids[id(r)]


def machine_chip(r: Any, w: int, t: Optional[float] = None,
                 send: bool = False) -> Optional[List[Line]]:
    """A Mac's /machine: the verdict, its chip drawn large and centred over its
    name, then the host, the stack and what it trains. ``t``: the time into
    the chip's entrance (chip_image.entrance, kitty only; ``send`` on its
    first frame), else the still. None where the terminal shows no images or
    there is no room: the rows say the same."""
    import sys

    from poros.cli.ui import chip_image, gpu_image
    from poros.cli.ui.style import resolve

    got = chip_image.parse(r.gpu.name.removesuffix(" (simulated)"))
    proto = gpu_image.protocol()
    ww = _w(w)
    if got is None or proto is None or ww < _SHOW_W:
        return None
    chip, cores = got
    below = [[], _centred([S(f"Apple {chip}", "hi")], ww),
             _centred([S(chip_image.subtitle(chip, cores, r.gpu.total_gib), "dm")], ww), [],
             *_machine_body(r, ww), *_machine_tail(r, ww)]
    cw, ch = gpu_image._cell(sys.stdout)
    cell = (cw or 8, ch or 17)
    room = _show_room() - 2 - len(below)
    cols = _SHOW_COLS
    rows = chip_image.size(chip, cols, cell)[1]
    if rows > room:  # drawn smaller, to be seen whole
        cols = int(cols * room / rows)
        if cols < _CHIP_LEAST:
            return None
        rows = chip_image.size(chip, cols, cell)[1]
    # kitty and iTerm2 scale the image to its cells: one drawing serves every size
    px = chip_image.size(chip, cols if proto == "sixel" else _SHOW_COLS, cell)[0]
    pic = chip_image.picture(chip, cores, resolve(sys.stdout), px)
    ids = chip_ident(r)
    if t is not None and proto == "kitty":
        seg, at = chip_image.entrance(pic, cols, rows, cell, ids, t, send), rows - 1
    else:
        seg, at = chip_image.placed(pic, cols, rows, proto, ids[0])
        if proto == "kitty":
            # the scan's chip and light, if it ran, go in the same frame: left
            # under the still, its see-through shadow would lie on the still's
            # and darken it the moment the result lands
            seg += f"\x1b_Ga=d,d=I,i={ids[1]},q=2\x1b\\\x1b_Ga=d,d=I,i={ids[2]},q=2\x1b\\"
    art: List[Line] = [[(" " * cols, None)] for _ in range(rows)]
    art[at] = [(seg, "raw"), *art[at]]
    pad = (" " * max(0, (ww - cols + 1) // 2), None)
    return [machine_head(r, w), [], *[[pad, *row] for row in art], *below]


def _room() -> int:
    """Rows a session tab holds under its tab row and over the prompt;
    outside the session, any height (the card scrolls with the shell)."""
    import shutil

    return shutil.get_terminal_size((100, 40)).lines - 6 if c.in_session else 10**6


def _machine_tail(r: Any, ww: int) -> List[Line]:
    """The command to run next, unless the row it fixes already says it."""
    nxt = _machine_next(r)
    said = nxt == _ML_FIX and any(ch.name == "torch" and ch.status == "ok" for ch in r.checks)
    return [] if said else [[]] + c.continue_with(nxt, ww)


def _os(platform: str) -> str:
    """``macOS-15.1-arm64-arm-64bit`` as people say it: ``macOS 15.1 · arm64``."""
    parts = platform.split("-")
    arch = next((p for p in parts if p in ("arm64", "aarch64", "x86_64", "AMD64")), "")
    ver = ".".join(parts[1].split(".")[:2]) if len(parts) > 1 and parts[1][:1].isdigit() else ""
    return " · ".join(x for x in (f"{parts[0]} {ver}".strip(), arch) if x)


def _machine_body(r: Any, ww: int, show: bool = False) -> List[Line]:
    """Under the card: the host and the training stack. Only what
    decides whether training runs here, each said once. With ``show`` it is
    Showcase's: who is on the card first, and the stack in one row."""

    lw, n = 8, 16
    value = {ch.name: str(ch.value) for ch in r.checks}
    status = {ch.name: ch.status for ch in r.checks}
    gpu = r.gpu

    def row(label: str, *parts: Seg) -> Line:
        return _cut(c.rail(S(label.ljust(lw), "dm"), *parts), ww)

    def aside(text: str, used: int) -> Optional[Seg]:
        room = ww - len(c.RAIL) - lw - used - 2
        return S("  " + _clip_text(text, room), "rl") if text and room > 8 else None

    out: List[Line] = []
    if show:
        out.append(_now_row(r, row))
    host = " · ".join(x for x in (_os(value.get("Platform", "")),
                                  f"Python {value['Python']}" if "Python" in value else "") if x)
    if host:
        bad = status.get("Python") in ("warn", "error")
        out.append(row("host", S(host, "tx"), aside("Python 3.11 or newer" if bad else "",
                                                    len(host))))
    ram_total = getattr(r, "ram_total_gib", None)
    if r.host_ram_free_gib is not None and ram_total:
        out.append(row("memory" if gpu is not None and gpu.unified else "ram",
                       *c.chips(1 - r.host_ram_free_gib / ram_total, n, "tx"),
                       S(f"  {r.host_ram_free_gib:.0f} GiB free", "tx"),
                       S(f" of {ram_total:.0f}", "dm")))
    elif ram_total:
        out.append(row("ram", S(f"{ram_total:.0f} GiB", "tx")))
    for name, val in value.items():  # a Mac names each of its disks
        if name.startswith("Disk ") and name != "Disk free":
            out.append(row("disk", S(f"{float(val.split()[0]):.0f} GB free", "tx"),
                           S(f"  {name.removeprefix('Disk ')}", "dm")))
    disk = value.get("Disk free", "")
    disk_total = getattr(r, "disk_total_gb", None)
    if disk[:1].isdigit():
        free = float(disk.split()[0])
        low = status.get("Disk free") in ("warn", "error")
        meter = c.chips(1 - free / disk_total, n, "e" if low else "tx") if disk_total else []
        text = f"{free:.0f} GB free"  # a disk is said in GB, as a download is; memory in GiB
        out.append(row("disk", *meter, S(("  " if meter else "") + text, "tx"),
                       S(f" of {disk_total:.0f}" if disk_total else "", "dm"),
                       aside("big downloads may not fit" if low else "",
                             (n + 2 if meter else 0) + len(text) + 8)))
    # the training stack: torch speaks for the packages one install brings
    missing = [ch.name for ch in r.checks if ch.name in _ML and ch.status != "ok"]
    torch = value.get("torch", "")
    if "torch" in missing:
        out.append(row("torch", S(torch, "e"),
                       aside("the command below installs the whole stack", 13)))
    elif torch and show and not missing:
        out.append(row("stack", S("✓ ", "tx"), S(f"torch {torch}", "tx"),
                       S(" · transformers · peft · bitsandbytes · accelerate", "dm")))
    elif torch:
        stack = f"{torch}" + (f" · CUDA {value['CUDA']}" if value.get("CUDA", "")[:1].isdigit()
                              else "")
        out.append(row("torch", S(stack, "tx")))
        if missing:
            gone = " · ".join(missing)
            out.append(row("", S(gone, "tx"), S("  not installed", "dm"),
                           aside(_ML_FIX, len(gone) + 15)))
    for ch in r.checks:  # anything else that needs you, by name
        if (ch.status in ("warn", "error") and ch.name not in _ML and ch.name not in _INFO
                and not ch.name.startswith("Disk ")):
            mark = "✕" if ch.status == "error" else "!"
            out.append(row(mark, S(ch.name, "tx"), S(f"  {ch.value}", "dm")))
    if not show:
        out.append(c.rail())

    return out


def _machine_next(r: Any) -> str:
    from poros.cli.verbs.shared import START_WITH

    if any(ch.name in _ML for ch in _problems(r)):
        return _ML_FIX
    if r.gpu is None:  # as the result's `next` says it: the machines the recipes ran on
        return "recipes"
    if getattr(r, "busy", None):
        return "runs"
    if r.gpu.unified:
        return "train"  # the card opens on what suits a first run
    # a first run: the small model every guide starts on, with the data still to name
    return f"train {START_WITH.split('/')[-1]} --data <file>"


def _machine_rows(r: Any, w: int) -> List[Line]:
    """Too narrow to draw the card, or no card: the same card in rows."""
    ww = _w(w)
    lw = 8
    gpu = r.gpu
    out = [machine_head(r, w)]
    if gpu is not None:
        name = gpu.name.removesuffix(" (simulated)")
        if gpu.unified:
            out.append(c.rail(S("gpu".ljust(lw), "dm"), S(name, "hi")))
        else:
            out.append(c.rail(S("gpu".ljust(lw), "dm"), S(name, "hi"),
                              S(f"  {gpu.total_gib:.1f} GiB", "dm")))
            room = ww - len(c.RAIL)
            out += [c.rail(*g) for g in _gpu_gauges(r, room, lw)[:1]]
    else:
        out.append(c.rail(S("gpu".ljust(lw), "dm"), S("none found", "tx"),
                          S(f"  {c.command('recipes')} says what each certified machine ran",
                            "rl")))
    out.append(c.rail())
    out += _machine_body(r, ww)
    return out + _machine_tail(r, ww)


def _clip_text(text: str, room: int) -> str:
    return text if len(text) <= room else text[: max(room - 1, 0)] + "…"


def _cut(line: Line, w: int) -> Line:
    out: Line = []
    left = w
    for t, st in line:
        if left <= 0:
            break
        out.append((t[:left], st))
        left -= len(t)
    return out
