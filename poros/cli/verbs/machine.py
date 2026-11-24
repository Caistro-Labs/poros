"""machine: whether this machine is ready to train.

    poros machine
    poros machine --json

The environment checks (Python, the training stack, the GPU, the driver, host RAM, the disk),
one verdict, and the command to run next. What trains on which machine is `poros recipes`:
each built-in recipe, with the machines its certified runs used.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import DoctorCheck, Output, PlanGpu
from poros.cli.ui.text import Line
from poros.cli.words.grammar import Sentence
from poros.core.card import nvidia_smi


class MachineAction(Action):
    verb: ClassVar[str] = "machine"

    def canonical(self) -> list[str]:
        return ["machine"]


class App(BaseModel):
    pid: int
    name: str
    used_gib: Optional[float] = None


class GpuNow(BaseModel):
    """The card as it is right now, from nvidia-smi (or the simulator)."""

    source: Literal["nvidia-smi", "simulated"]
    driver: Optional[str] = None
    used_gib: Optional[float] = None
    total_gib: Optional[float] = None
    util_pct: Optional[int] = None
    temp_c: Optional[int] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    fan_pct: Optional[int] = None
    pcie: Optional[str] = None  # "5 x16"
    board: Optional[str] = None  # who made the board: "NVIDIA" (Founders Edition), "ASUS"…
    apps: list[App] = Field(default_factory=list)


class Busy(BaseModel):
    """A Poros run using (or about to use) the card."""

    id: str
    name: str
    model: str
    state: str
    step: Optional[int] = None
    total: Optional[int] = None


class MachineOutput(Output):
    schema_: Literal["poros.machine.v1"] = Field("poros.machine.v1", alias="schema")
    verdict: Literal["ready", "ready with warnings", "not ready"]
    # not ready, or ready with warnings: why, in a line, and the commands that act on it, in
    # order ([] where no command helps); "" and [] when ready
    reason: str = ""
    next: list[str] = []
    version: str
    gpu: Optional[PlanGpu]
    host_ram_free_gib: Optional[float]
    checks: list[DoctorCheck]
    now: Optional[GpuNow] = None  # what the card is doing as it is checked
    busy: list[Busy] = Field(default_factory=list)  # Poros runs on it
    # for the screen: the wholes the gauges are shares of
    ram_total_gib: Optional[float] = Field(None, exclude=True)
    disk_total_gb: Optional[float] = Field(None, exclude=True)
    exit_code: int = Field(0, exclude=True)


def _totals() -> tuple[Optional[float], Optional[float]]:
    """Host RAM (GiB) and the disk here (GB, as the disk check says it)."""
    import shutil

    from poros.train.reporting import host_ram_total_gb

    ram = host_ram_total_gb()
    try:
        disk = shutil.disk_usage(".").total / 1e9
    except OSError:
        disk = None
    return (ram * 1e9 / 1024**3 if ram else None), disk


def resolve(s: Sentence, ctx: Context) -> Draft[MachineAction]:
    return Draft(MachineAction())


def confirmations(action: MachineAction, ctx: Context) -> list:
    return []


def _num(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


def _on(gpu: Optional[PlanGpu]) -> list[str]:
    """nvidia-smi's flag for the card torch would train on: its numbers are not CUDA's."""
    return ["-i", gpu.uuid] if gpu is not None and gpu.uuid else []


def gpu_board(gpu: Optional[PlanGpu] = None) -> str:
    """Who made the board, from its PCI subsystem vendor; "" when unknown."""
    import os

    from poros.cli.ui import gpu_art, simulate

    if simulate.enabled() and simulate.this_mac() is None:
        return os.environ.get("POROS_DEV_SIMULATE_BOARD", "")
    lines = nvidia_smi("gpu=pci.sub_device_id", units=True, on=_on(gpu), timeout=2.0, raw=True)
    rows = "\n".join(lines).split() if lines is not None else []
    return gpu_art.board(rows[0]) if rows else ""


def gpu_now(gpu: Optional[PlanGpu] = None) -> Optional[GpuNow]:
    """Memory in use, load, temperature, power and who is on the card."""

    from poros.cli.ui import simulate

    if simulate.enabled() and simulate.this_mac() is None:
        return GpuNow(source="simulated", driver="simulated", used_gib=0.4,
                      total_gib=simulate.GPU[1], util_pct=0, temp_c=31, power_w=22.0,
                      power_limit_w=575.0, fan_pct=0, pcie="5 x16")
    fields = ("driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,"
              "power.draw,power.limit,fan.speed,pcie.link.gen.current,pcie.link.width.current")
    rows = nvidia_smi(f"gpu={fields}", on=_on(gpu), timeout=2.0)
    if not rows:
        return None
    v = rows[0]
    if len(v) < 10:
        return None
    num = [_num(x) for x in v]
    now = GpuNow(
        source="nvidia-smi", driver=v[0] or None,
        used_gib=round(num[1] / 1024, 1) if num[1] is not None else None,
        total_gib=round(num[2] / 1024, 1) if num[2] is not None else gpu and gpu.total_gib,
        util_pct=int(num[3]) if num[3] is not None else None,
        temp_c=int(num[4]) if num[4] is not None else None,
        power_w=num[5], power_limit_w=num[6],
        fan_pct=int(num[7]) if num[7] is not None else None,
        pcie=f"{v[8]} x{v[9]}" if num[8] is not None and num[9] is not None else None,
    )
    for parts in nvidia_smi("compute-apps=pid,process_name,used_memory", on=_on(gpu),
                            timeout=2.0) or []:
        if len(parts) >= 3 and parts[0].isdigit():
            mib = _num(parts[2])
            now.apps.append(App(pid=int(parts[0]), name=parts[1].rsplit("/", 1)[-1],
                                used_gib=round(mib / 1024, 1) if mib is not None else None))
    return now


def busy_runs() -> list[Busy]:
    from poros import runs

    try:
        live = [r for r in runs.all_runs() if r.state in ("running", "queued", "stopping")]
    except OSError:
        return []
    return [Busy(id=r.id, name=r.name, model=r.model, state=r.state, step=r.step, total=r.total)
            for r in live]


def run(action: MachineAction, ctx: Context) -> MachineOutput:
    from poros import __version__
    from poros.cli.commands.check import _environment_checks
    from poros.cli.ui import term
    from poros.core.errors import EXIT_CODES

    below = None
    chip = _chip_drawing(ctx) if ctx.gpu is not None and ctx.unified else None
    # A discrete card is the one nvidia-smi describes: its board, its live row. A unified pool (a
    # Mac's) is not, and an nvidia-smi that answers beside one speaks of another device.
    drawn = ctx.gpu is not None and not ctx.unified
    maker, proto = "", None
    if drawn:
        from poros.cli import views
        from poros.cli.ui import gpu_art, gpu_image
        from poros.cli.views.machine_screens import _SHOW_W

        name = ctx.gpu.name.removesuffix(" (simulated)")
        maker = gpu_board(ctx.gpu)
        proto = gpu_image.protocol()  # asked here: the terminal is not read off this thread
        ready, ident = None, gpu_image.new_id()  # the loading card's own image
        biggest = views.show_size(name, maker, proto)
        if proto:  # the image is drawn while the checks run, and shown once it is
            import sys
            import threading

            from poros.cli.ui.style import resolve

            ready = threading.Event()
            pal = resolve(sys.stdout)

            def draw() -> None:
                try:
                    gpu_image.placed(name, pal, maker, proto,
                                     biggest=biggest or gpu_image.COLS)
                finally:
                    ready.set()

            threading.Thread(target=draw, name="poros-draw", daemon=True).start()

        def below(t: int, w: int) -> list:
            # the card the result is drawn on: the image where the terminal
            # shows one (the same block of rows before it is ready, so
            # nothing moves), else braille with its fans turning
            if biggest is None or min(w, 100) < _SHOW_W:  # Showcase won't fit: the side card
                if min(w, 100) < gpu_art.W + 2 + 36:
                    return []
                if ready is None:
                    return [[]] + views.machine_card(name, w, angle=t * 0.35, maker=maker)
                if ready.is_set():
                    return [[]] + views.machine_card(name, w, image=True, maker=maker,
                                                     ident=ident)
                aw, ah = gpu_image.cols(name, maker), gpu_image.rows(name, maker, proto=proto)
                return [[]] + [[("  " + " " * aw, None)] for _ in range(ah)]
            if ready is None:
                return views.machine_show(name, w, angle=t * 0.35, maker=maker)
            if ready.is_set():
                return views.machine_show(name, w, image=True, maker=maker,
                                          ident=ident, biggest=biggest)
            ah = gpu_image.rows(name, maker, proto=proto, biggest=biggest)
            return [[] for _ in range(ah)]

    # instant on most machines: the loading card shows only if the checks
    # take long enough to notice (nvidia-smi on a card that has to wake)
    with term.get().activity("Checking this machine", below=below, delay=0.25) as act:
        checks, _ = _environment_checks(quietly=True)
        now = gpu_now(ctx.gpu) if drawn else None
        if now is not None:
            now.board = maker or None
        busy = busy_runs()
        act.done([])
    if drawn and proto:
        gpu_image.forget(proto, ident)  # the result draws its own
    status = {s for _, _, s in checks}
    ram_total, disk_total = _totals()
    # Training needs a CUDA GPU: without one this machine is not ready for it.
    verdict = (
        "not ready"
        if "error" in status or ctx.gpu is None
        else ("ready with warnings" if "warn" in status else "ready")
    )
    reason, to_run = why(verdict, [DoctorCheck(name=n.strip(), value=v, status=s)
                                   for n, v, s in checks], ctx.gpu)
    out = MachineOutput(
        verdict=verdict,
        reason=reason,
        next=to_run,
        version=__version__,
        gpu=ctx.gpu,
        host_ram_free_gib=ctx.host_free_gib,
        now=now,
        busy=busy,
        checks=[DoctorCheck(name=n.strip(), value=v, status=s) for n, v, s in checks],
        ram_total_gib=ram_total, disk_total_gb=disk_total,
        exit_code=EXIT_CODES["E_MACHINE_NOT_READY"] if verdict == "not ready" else 0,
    )
    if chip is not None:
        _chip_entrance(out, chip)
    return out


def why(verdict: str, checks: list[DoctorCheck], gpu: Optional[PlanGpu]) -> tuple[str, list[str]]:
    """A verdict that is not a plain ready, for a result: why in a line, as the card's head and
    its rows say it to a person, and the commands that act on it. With no GPU nothing installed
    here trains, so it names no install: its command plans for a card elsewhere."""
    from poros.cli.views.machine_screens import _ML, _ML_FIX
    from poros.cli.views.parts import _needs_gpu

    if verdict == "ready":
        return "", []
    if gpu is None:
        return (f"training {_needs_gpu()}, and this machine has none",
                ["poros recipes"])
    wrong = sorted((c for c in checks if c.status in ("error", "warn")),
                   key=lambda c: c.status != "error")  # what stops a run first
    stack = [c.name for c in wrong if c.name in _ML]  # one install brings them all
    said = [f"{c.name}: {c.value}" for c in wrong if c.name not in _ML]
    if stack:
        said.insert(0, "the training stack needs an install: " + ", ".join(stack))
    return "; ".join(said), [_ML_FIX] if stack else []


def _chip_drawing(ctx: Context):
    """A Mac's chip, drawn while the checks run: a second or two the first
    time, then from the cache. The event is set once it is ready; None where
    the terminal shows no image this way."""
    import sys
    import threading

    from poros.cli.ui import chip_image, gpu_image
    from poros.cli.ui.style import resolve
    from poros.cli.views.machine_screens import _SHOW_COLS

    got = chip_image.parse(ctx.gpu.name.removesuffix(" (simulated)"))
    proto = gpu_image.protocol()  # asked here: the terminal is not read off this thread
    if got is None or proto not in ("kitty", "iterm"):
        return None
    cw, ch = gpu_image._cell(sys.stdout)
    px = chip_image.size(got[0], _SHOW_COLS, (cw or 8, ch or 17))[0]
    pal = resolve(sys.stdout)
    ready = threading.Event()

    def draw() -> None:
        try:
            chip_image.picture(got[0], got[1], pal, px)
        finally:
            ready.set()

    threading.Thread(target=draw, name="poros-draw", daemon=True).start()
    return ready


def _chip_entrance(r: MachineOutput, ready) -> None:
    """The chip comes in with a scan, top to bottom, where kitty images are
    shown locally and motion is on; the result then lays its still in the
    same place, as the same image. Any failure: straight to the still."""
    import os
    import time

    from poros.cli import contract
    from poros.cli.ui import chip_image, components, gpu_image, term
    from poros.cli.views.machine_screens import machine_chip

    ready.wait(10)
    if (contract.json_mode() or gpu_image.protocol() != "kitty"
            or os.environ.get("SSH_CONNECTION") or os.environ.get("POROS_NO_ANIM")):
        return
    sent = []  # the chip goes up to the terminal once, on the first frame drawn

    def surface(_frame: int, w: int) -> list:
        lines = machine_chip(r, w, time.monotonic() - t0, send=not sent)
        if lines is None:
            return []
        sent.append(1)
        return [components.titled(lines[0]), *(components.inked(x) for x in lines[1:])]

    try:
        if machine_chip(r, term.get().width(), chip_image.DUR) is None:  # room for it
            return
        t0 = time.monotonic()
        with term.get().live(surface) as live:
            # 60 fps costs nothing now (a frame is a few bytes); a little past
            # DUR, so the last frame drawn is the chip whole
            live.play(surface, chip_image.DUR * 1000 + 50, fps=60)
            live.update()
            if hasattr(live, "tab"):  # in a tab: the still takes the scan's place, no frame between
                live.close(hold=True)
    except (Exception, KeyboardInterrupt):  # a drawing bug or an impatient ctrl-c: the still
        pass


def view(r: MachineOutput, w: int) -> list[Line]:
    import sys

    from poros.cli import views
    from poros.cli.ui.live import interactive

    # the drawn card is for a terminal: a pipe or a file gets the same facts in rows
    return views.machine(r, w, art=interactive(sys.stdout))
