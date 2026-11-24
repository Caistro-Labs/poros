"""Watch a run that is already going, from its events.jsonl: the same live
screen `train` draws, fed by the file instead of the trainer. ctrl-c stops
watching; the run carries on."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from poros import runs
from poros.cli.ui import components as c
from poros.cli.ui import curve, gpu_image, term
from poros.cli.ui import run_card as rc
from poros.cli.ui.screens import TrainState, heartbeat, train_surface
from poros.core.events import resumes

# what _apply folds in from the events: a resume marker sets them back and folds them again
_FOLDED = ("step", "loss", "losses", "step_time", "step_times", "tokens_per_sec", "tokens_seen",
           "vram_gb", "vram_peak_gb", "evals", "scores", "ends", "series", "jumps")


def parsed(raw: bytes) -> list:
    """events.jsonl's complete lines, as objects; a torn or foreign line skips only itself."""
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            ev = json.loads(line)
        except (ValueError, RecursionError):  # a torn line, or one nested past the stack
            continue
        if isinstance(ev, dict) and isinstance(ev.get("data", {}), dict):
            out.append(ev)
    return out


def refold(st: TrainState, path: Path, end: int) -> list:
    """A resume marker came in: ``st`` set back, and the events the file's first ``end`` bytes
    hold as the run's record stands (``events.current``: the interrupted run's steps past its
    checkpoint superseded) returned, to fold in again."""
    import dataclasses

    from poros.core.events import current

    for f in dataclasses.fields(st):
        if f.name in _FOLDED:
            value = f.default_factory() if f.default_factory is not dataclasses.MISSING else (
                f.default)
            setattr(st, f.name, value)
    return current(parsed(runs.read_own(path, 0, end) or b""))


def _apply(st: TrainState, ev: dict) -> None:
    data = ev.get("data") or {}
    if ev.get("event_type") == "resume" and type(ev.get("step")) is int:
        st.step = ev["step"]  # the updates its checkpoint holds: the steps done
    if ev.get("event_type") == "setup":
        st.recipe = st.recipe or data.get("recipe") or ""
    if ev.get("event_type") == "eval" and (got := rc.score(st.step, data)):  # the steps done
        st.evals.append((ev.get("step") or 0, got.loss))
        st.scores.append(got)
        st.ends = (got.k, got.n) if got.word == "ends" else None
    if ev.get("event_type") != "step":
        return
    if ev.get("step") is not None:
        st.step = ev["step"] + 1  # events count from 0; the screen counts steps done
    if data.get("loss") is not None:
        st.loss = float(data["loss"])
        st.losses.append(st.loss)
    if data.get("s_per_step"):
        st.step_time = float(data["s_per_step"])
        st.step_times.append(st.step_time)
        if data.get("tokens"):  # a recipe's are the step's; the classic loop's the count so far
            seen = float(data["tokens"])
            # the classic loop counts from 0 again in a resumed process
            made = (seen if st.recipe or seen < st.tokens_seen
                    else seen - st.tokens_seen if st.tokens_seen else 0.0)
            st.tokens_per_sec = made / st.step_time
            st.tokens_seen = seen
    if data.get("footprint"):  # a Mac: what its one pool holds, and the process's peak
        st.vram_gb, st.vram_peak_gb = data["vram_reserved_mb"] / 1000, data["footprint"] / 1e9
    elif data.get("vram_allocated_mb") is not None:
        st.vram_gb = data["vram_allocated_mb"] / 1000  # the trainer writes decimal MB
        # the high-water mark a receipt reads, which an older run's events do not hold
        st.vram_peak_gb = max(st.vram_peak_gb, st.vram_gb, data.get("vram_peak_mb", 0.0) / 1000)
    if not st.started:  # when the run began, not when watching did
        since = time.time() - ev["timestamp"] if ev.get("timestamp") else 0.0
        st.started = time.monotonic() - max(0.0, since)


def show_tab(run: runs.Run) -> bool:
    """In the tabbed session, the tab already training ``run`` comes forward
    (its live screen has every row; a second one would only copy it), and
    this tab goes. False when no tab trains it."""
    from poros.cli.ui import tui

    scr = tui.current
    if scr is None:
        return False
    at = next((i for i, t in enumerate(scr.tabs) if t.run == run.id and t.running), -1)
    if at < 0:
        return False
    here = getattr(tui.local, "tab", None)
    if here is not None:
        here.gone = True
    scr.switch(at)
    return True


def follow(run: runs.Run, poll: float = 0.5) -> None:
    st = TrainState(
        model=run.model,
        output_dir=run.output_dir,
        run=run.id,
        total=run.total or 0,
        label="Training",
        watching=True,
    )
    events = Path(run.output_dir) / "events.jsonl"
    pos = 0
    live = term.get().live(
        lambda t, w: train_surface(st, t, w), heartbeat=lambda _t: heartbeat(st), heartbeat_s=30.0
    )
    if getattr(live, "live", False):  # the same drawn curve the trainer's own screen has
        st.painter = curve.Painter.maybe()
    term.get().progress(lambda: (st.step, st.total))
    live.start()
    note = [c.event("Watched", run.id, "the run finished")]
    try:
        while True:
            if events.exists():
                with open(events, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                complete = chunk[: chunk.rfind(b"\n") + 1]  # a half-written line waits
                pos += len(complete)
                evs = parsed(complete)
                if any(map(resumes, evs)):  # a resumed run: its record again, as it stands
                    evs = refold(st, events, pos)
                for ev in evs:
                    try:
                        _apply(st, ev)
                    except (TypeError, ValueError, KeyError):
                        continue
                st.elapsed = time.monotonic() - st.started if st.started else 0.0
                live.update()
            state = next((r.state for r in runs.all_runs() if r.id == run.id), "failed")
            if state not in ("running", "queued", "stopping"):
                note = [c.event("Watched", run.id, f"it {state}")]
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        note = [c.event("Stopped watching", run.id, "the run carries on", mark="○")]
    finally:
        if st.painter is not None:  # the note replaces the card: its images come off
            gpu_image.forget("kitty", st.painter.halo_id, sys.stdout)
            gpu_image.forget("kitty", st.painter.ident, sys.stdout)
        live.close([[]] + note)
