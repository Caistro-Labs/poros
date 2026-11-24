"""The CLI's reporter: drives the live training surface from trainer hooks.

The trainer calls ``on_step`` once per optimizer step from its own thread;
the surface redraws at 10 fps from the ``Live`` thread and reads the shared
``TrainState`` (plain attribute writes, read-only on the UI side). Streaming
position and host RAM are sampled at draw time, so the surface stays live
between steps -- a 72B step is ten seconds long.
"""

from __future__ import annotations

import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from poros.cli.ui import components as c
from poros.cli.ui import curve, gpu_image
from poros.cli.ui import run_card as rc
from poros.cli.ui.live import Live
from poros.cli.ui.screens import (
    TrainResult,
    TrainState,
    finish_frame,
    finish_length,
    heartbeat,
    train_done,
    train_surface,
)
from poros.cli.ui.text import Line, S
from poros.train.reporting import (
    Reporter,
    StepReport,
    StreamProbe,
    host_ram_gb,
    host_ram_total_gb,
)


def new_run_id() -> str:
    return secrets.token_hex(4)


class Activity:
    """A one-line live activity (``• Loading Qwen2.5-72B  12s``) around a
    blocking call, closed into a finished ``●`` event."""

    def __init__(
        self,
        label: str,
        *,
        stream=None,
        statusline: Optional[List[str]] = None,
        tag: str = "",
        live=None,
        below=None,
        delay: float = 0.0,
    ) -> None:
        self.label = label
        self._below = below  # (frame, width) -> rows drawn under the line
        self.t0 = time.monotonic()
        self._status = statusline or []
        self._tag = tag
        make = live or (lambda fn, **kw: Live(fn, stream=stream, **kw))
        self.live = make(self._surface, heartbeat=None, **({"fps": 15.0} if below else {}),
                         **({"delay": delay} if delay else {}))

    def _surface(self, t: int, w: int) -> List[Line]:
        rows = [
            c.activity(
                self.label,
                c.duration(time.monotonic() - self.t0),
                t,
                w=min(w, c.CAP),
                animate=self.live.live,
            )
        ]
        if self._below is not None:
            rows += self._below(t, w)
        if self._status or c.status_hook is not None:
            rows += c.statusline(self._status, w, self._tag)
        return rows

    def __enter__(self) -> "Activity":
        self.live.start()
        return self

    def done(self, lines: List[Line]) -> None:
        self.live.close(lines)

    def __exit__(self, exc_type, *exc) -> None:
        self.live.close()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0


class TrainView(Reporter):
    _under_blank = False  # see __init__
    def __init__(self, state: TrainState, *, stream=None, live=None, under_blank: bool = False
                 ) -> None:
        self.st = state
        # the line above is already blank (the session's start block ends with one): the
        # card's own leading blank would make two, until a line is printed between them
        self._under_blank = under_blank
        self.probe: Optional[StreamProbe] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.peak_ram: Optional[float] = None
        self.interrupted: Optional[tuple] = None
        self.checkpoints: List[str] = []
        self.st.ram_total_gb = host_ram_total_gb()
        make = live or (lambda fn, **kw: Live(fn, stream=stream, **kw))
        self.live = make(self._surface, heartbeat=lambda _t: heartbeat(self.st), heartbeat_s=30.0)
        self.stream = stream
        if getattr(self.live, "live", False) and self.st.painter is None:
            self.st.painter = curve.Painter.maybe(stream)  # asked here, on the main thread
        self._t0 = time.monotonic()
        self._frame = threading.Lock()  # a frame, or the switch from the load to training

    # surface ---------------------------------------------------------------

    def _surface(self, t: int, w: int) -> List[Line]:
        with self._frame:
            return self._drawn(t, w)

    def _drawn(self, t: int, w: int) -> List[Line]:
        st = self.st
        st.elapsed = time.monotonic() - self._t0
        st.ram_gb = None if sys.platform == "darwin" else host_ram_gb()  # the memory bar says it
        if st.ram_gb is not None:
            self.peak_ram = max(self.peak_ram or 0.0, st.ram_gb)
        if self.probe is not None:
            st.stream_pos = self.probe.position()
            st.stream_gbs = self.probe.sample() or st.stream_gbs
        return self._gap(train_surface(st, t, w, animate=self.live.live))

    def _gap(self, rows: List[Line]) -> List[Line]:
        return rows[1:] if self._under_blank and rows and not rows[0] else rows

    def start(self) -> None:
        self.live.start()

    def loaded(self) -> None:
        """The load is done: the card's activity line says training, its clock from now, in one
        step a frame cannot fall between. The host RAM peak counts from here, as it did when the
        view started after the load."""
        with self._frame:
            self._t0 = time.monotonic()
            self.st.elapsed = 0.0
            self.st.label = "Training"
            self.peak_ram = None

    def close(self, final: Optional[List[Line]] = None) -> None:
        """Whatever of the curve the final lines don't draw again comes off."""
        p = self.st.painter
        if p is not None:
            out = self.stream or sys.stdout
            gpu_image.forget("kitty", p.halo_id, out)
            drawn = any(st == "raw" and f"i={p.ident}," in t for r in final or [] for t, st in r)
            if not drawn:
                gpu_image.forget("kitty", p.ident, out)
        self.live.close(final)

    def finish(self, r: TrainResult, w: int) -> None:
        """The run's end, drawn: the live card closes into the finished one."""
        st = self.st
        if st.series is None and c.in_session and st.losses:  # no frame drew it before the end:
            from poros.cli.ui.screens import _series  # the line the live card would have drawn

            _series(st, list(st.losses))
        r.series, r.painter = st.series, st.painter
        r.evals, r.offset = list(st.evals), max(0, st.step - len(st.losses))
        r.scores = list(st.scores)
        r.vram_now, r.ram_now = st.vram_gb, st.ram_gb
        if st.accs:  # a pair run's accuracy, as its page and an agent's sentence say it: the
            from poros.cli.verbs.runs_report import last_rows  # mean over its last steps

            r.acc_over, r.extra["accuracy"] = last_rows(st.accs)
        final = self._gap([[]] + train_done(r, w))
        play = getattr(self.live, "play", None)
        far = st.painter is not None and st.painter.remote  # 70 KB a frame, over SSH: no
        if play is not None and st.series is not None and st.series.line and not far:
            if st.painter is not None:
                gpu_image.forget("kitty", st.painter.halo_id, self.stream or sys.stdout)
            t0 = time.monotonic()

            def clock() -> float:
                return (time.monotonic() - t0) * 1000

            try:
                play(lambda f, wide: self._gap(finish_frame(r, st, wide, clock(), f)),
                     finish_length(r, w), land=curve.LAND)
            except (Exception, KeyboardInterrupt):  # a drawing bug or an impatient ctrl-c
                pass  # must never cost the run's result: straight to the finished card
        self.close(final)

    def print(self, lines: List[Line]) -> None:
        self._under_blank = False
        self.live.print(lines)

    # reporter hooks ------------------------------------------------------------

    def on_start(self, trainer: Any, total: int, start_step: int) -> None:
        self.st.total = total
        self.st.step = start_step
        # a Mac's layers come whole from one pool: there is no stream to watch
        self.probe = None if sys.platform == "darwin" else StreamProbe.attach(trainer)
        if self.probe is not None:
            self.st.stream_unit = self.probe.unit
            self.st.stream_count = self.probe.count
        self.st.stream_soon = False  # the stream's row now stands on what is there
        if start_step:  # steps done, as every line counts them: the next one is start_step + 1
            self.print([c.event("Resumed", f"after step {c.num(start_step)}/{c.num(total)}")])

    def on_step(self, r: StepReport) -> None:
        st = self.st
        st.step, st.total = r.step, r.total
        st.losses.append(r.loss)
        st.step_times.append(r.step_time)
        st.step_time = r.step_time
        st.tokens_per_sec = r.tokens_per_sec
        st.vram_gb, st.vram_peak_gb = r.vram_gb, r.vram_peak_gb
        st.extra = dict(r.extra)
        if isinstance(r.extra.get("accuracy"), (int, float)):
            st.accs.append(float(r.extra["accuracy"]))
        self.live.update()

    def on_checkpoint(self, step: int, path: str) -> None:
        self.checkpoints.append(path)
        self.st.last_checkpoint = path
        if getattr(self.live, "live", False):
            return  # the live card's table lists it; a log has no card, so it gets a line
        self.print([c.event("Saved", checkpoint_said(path, self.st.step or step),
                            c.short_path(path))])

    def on_eval(self, step: int, metrics: Dict[str, Any]) -> None:
        """A held-out pass: a row in the card's ledger, and in a log a line."""
        st = self.st
        # the steps done, as the run shows them (a recipe counts from 0); before any: the base
        got = rc.score(st.step if step else 0, metrics)
        if got is None:
            return
        st.evals.append((step, got.loss))
        st.scores.append(got)
        if got.word == "ends":
            st.ends = (got.k, got.n)
        if getattr(self.live, "live", False):
            return  # the card's table shows it, the best marked
        base = st.scores[0].loss if len(st.scores) > 1 else None
        tail = f"held out {got.loss:.4f}"
        if base:
            tail += f" {'↓' if got.loss <= base else '↑'} {abs(got.loss - base) / base * 100:.0f}%"
        if got.word:
            said = "ends its answers" if got.word == "ends" else "prefers"
            tail += f" · {said} {got.k} of {got.n}"
        best = min((s.loss for s in st.scores[1:-1] if s.step), default=None)
        if got.step and len(st.scores) > 2 and best is not None and got.loss < best:
            tail += " · best"
        self.print([c.event("Scored", "the base" if not got.step else f"step {c.num(got.step)}",
                            tail)])

    def on_interrupt(self, step: int, checkpoint: Optional[str]) -> None:
        self.interrupted = (step, checkpoint)

    def on_done(self, metrics: Dict[str, Any]) -> None:
        self.metrics = metrics


def steps_said(n: int) -> str:
    """A count of steps, as every line a person reads says one: "1 step", "1,200 steps"."""
    return f"{c.num(n)} step{'' if n == 1 else 's'}"


def saved_checkpoint(output_dir: str) -> Optional[tuple]:
    """(the steps it holds, its folder) for the newest complete checkpoint in a run's folder, as
    the disk has it now; None where it saved none."""
    from poros.cli.verbs import checkpoints as listed

    found = listed._folders(Path(output_dir))
    return (max(found), str(found[max(found)])) if found else None


def steps_in(checkpoint: Optional[str]) -> Optional[int]:
    """The steps a checkpoint folder holds, as its run's listing counts them; None for a
    folder that is no complete checkpoint."""
    from poros.cli.verbs import checkpoints as listed

    if not checkpoint:
        return None
    path = Path(checkpoint)
    return next((n for n, p in listed._folders(path.parent).items() if p.name == path.name),
                None)


def finished_lines(done: int, total: int, output_dir: str, more: str, next_cmd: str,
                   w: int) -> List[Line]:
    """A resume of a run that has no step left to train: said as that, with the command that
    trains it further. No loss: none was computed."""
    wc = min(w, c.CAP)
    out = [c.event("Already finished", f"{c.num(done)} of {c.steps(total)}", mark="○"),
           c.rail(S("no step was left to train, so none ran", "dm"))]
    out += c.rail_path("adapter", output_dir, wc)
    out.append(c.rail(S("to train it further: ", "dm"), S(more, "tx")))
    if next_cmd:
        out.append([])
        out += c.continue_with(next_cmd, wc)
    return out


def interrupt_lines(step: int, checkpoint: Optional[str], config_path: str, w: int,
                    resume: Optional[str] = None, held: Optional[int] = None) -> List[Line]:
    """A stopped run: how many steps it had done, the checkpoint it resumes from (``held``: the
    steps that one holds, where they are fewer: a ctrl-c saves none, so its last save is behind)
    and the command that resumes it (``resume``: a job's own, for jobs)."""
    wc = min(w, c.CAP)
    out = [c.event("Stopped", f"after {steps_said(step)}", mark="○")]
    if checkpoint:
        out += c.rail_path("saved", checkpoint, wc)
        # the folder is named by an index, the line above counts steps: what it holds, said
        held = held if held is not None else steps_in(checkpoint)
        if held is not None and held >= step:
            out.append(c.rail(S(" " * 10, None), S(f"it holds all {steps_said(held)}", "dm")))
        elif held is not None:
            lost = (f"; the {steps_said(step - held)} after it "
                    f"train{'s' if step - held == 1 else ''} again")
            out.append(c.rail(S(" " * 10, None),
                              S(f"it holds {steps_said(held)}: a resume starts there{lost}",
                                "dm")))
        out.append([])
        out += c.continue_with(resume or f"poros train {config_path} --resume-from latest", wc)
    else:
        out.append(c.rail(S("no checkpoint was saved: ctrl-c saves none, and save_steps "
                            "says how often a run does", "dm")))
    return out


def checkpoint_said(path: str, held: int) -> str:
    """A saved checkpoint as the run's line names it: as its folder is (checkpoint-<n>), so the
    two agree; a trainer numbers its folders from 0 (the last update's index), so there it also
    says how many steps the folder holds: "checkpoint-12 (13 steps)"."""
    name = Path(path).name if path else f"checkpoint-{held}"
    return name if name == f"checkpoint-{held}" else f"{name} ({steps_said(held)})"
