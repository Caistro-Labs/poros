"""A run as an agent watches it: `poros runs <id> --live --json`. The state
the live screen draws, read from the run's events.jsonl; nothing else runs
and the trainer writes nothing more. `card` is the run in two rows of plain
text for an agent's reply; --frame adds a larger one with the loss curve,
and --wait N returns once the run is N steps further or has ended."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from poros import runs
from poros.cli.schemas import Output
from poros.cli.ui import agent_text
from poros.cli.ui import components as c
from poros.cli.ui.screens import TrainState
from poros.cli.verbs import runs_report
from poros.cli.verbs.runs_follow import _apply, parsed, refold
from poros.core.env import under_agent
from poros.core.events import resumes

LIVE = runs_report.LIVE
UNVALIDATED = runs_report.UNVALIDATED
WAIT_MAX_S = 100.0  # returns inside the 120 s an agent's shell call often allows


class RunLiveOutput(Output):
    schema_: Literal["poros.run_live.v1"] = Field("poros.run_live.v1", alias="schema")
    learned: Optional[str] = None  # `poros agents`, once over: its held-out loss in words
    run: str
    name: str
    state: str
    # running: what it does now. training its steps; scoring its held-out rows, its last step done;
    # saving what it trained, its last step done and scored
    phase: Optional[Literal["training", "scoring", "saving"]] = None
    model: str
    step: int
    total: int
    loss: Optional[float]  # the latest step's
    loss_smooth: Optional[float]  # where the run card's line ends, as the screen shows it
    loss_first: Optional[float]
    s_per_step: Optional[float]
    tokens_per_s: Optional[float]
    vram_gib: Optional[float]  # allocated at the last step's end, its streamed blocks gone
    vram_peak_gib: Optional[float]  # the receipt's measure: the allocator's high-water mark
    card_gib: Optional[float]
    elapsed_s: float  # from the first step
    eta_s: Optional[float]
    eval_loss_start: Optional[float] = None  # the held-out rows' loss before the first step
    eval_loss: Optional[float] = None  # and after the last, once scored (pairs: DPO's loss)
    eval_rows: Optional[int] = None  # how many rows that is
    # after the last: on chats the rows whose answers all end on the end-of-turn token, of those
    # with an end to read; on pairs the pairs whose chosen answer scores over the rejected one,
    # before the first step and after the last
    eval_ends: Optional[int] = None
    eval_answer_rows: Optional[int] = None
    eval_prefers_start: Optional[int] = None
    eval_prefers: Optional[int] = None
    best: Optional[str] = None  # the kept checkpoint with the lowest held-out loss, by its path
    error: Optional[dict] = None  # why a run failed: the card only says that it did
    from_: Optional[str] = Field(None, alias="from")  # --from: the run it started from
    queued_behind: Optional[str] = None  # queued: the run it waits for on its GPU
    # --unvalidated: user_asserted, and the model_type: a family Poros has not validated
    support: Optional[str] = None
    model_type: Optional[str] = None
    card: list[str]  # two rows to show the user as they are, in a code block
    frame: Optional[list[str]] = None  # --frame: the run with its loss curve
    next: list[str] = []  # the commands a finished run offers


class _Events:
    """events.jsonl read forward: each call applies only the lines written
    since the last one, and a half-written line waits."""

    def __init__(self, st: TrainState) -> None:
        self.st, self.pos = st, 0
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        self.path = Path(st.output_dir) / "events.jsonl"
        self.tokens = self.seconds = self.seen = 0.0  # the whole run's, for its mean tok/s

    def _count(self, data: dict) -> None:
        """A step's tokens and time, into the run's whole: a recipe's events hold a step's tokens,
        the classic loop's the count so far (from 0 again after a continue)."""
        tokens, took = data.get("tokens"), data.get("s_per_step")
        if not (isinstance(tokens, (int, float)) and isinstance(took, (int, float)) and took > 0):
            return
        made = tokens if self.st.recipe or tokens < self.seen else tokens - self.seen
        self.tokens, self.seconds, self.seen = self.tokens + made, self.seconds + took, tokens

    def read(self) -> None:
        chunk = runs.read_own(self.path, self.pos)
        if not chunk:
            return
        complete = chunk[: chunk.rfind(b"\n") + 1]
        self.pos += len(complete)
        evs = parsed(complete)
        if any(map(resumes, evs)):  # a resumed run: its record again, as it stands
            evs = refold(self.st, self.path, self.pos)
            self.first = self.last = None
            self.tokens = self.seconds = self.seen = 0.0
        for ev in evs:
            try:
                _apply(self.st, ev)
            except (TypeError, ValueError, KeyError):  # a field of a kind no trainer writes
                continue
            if ev.get("event_type") == "step":
                self._count(ev.get("data") or {})
            if ev.get("event_type") == "step" and isinstance(ev.get("timestamp"), (int, float)):
                self.first = self.first or ev["timestamp"]
                self.last = ev["timestamp"]


def _phase(state: str, st: TrainState, held: dict) -> Optional[str]:
    """What a running run does: its steps; at their end, the held-out rows it scored before them
    scored again; then what it trained saved. A score after its last step is one at that step."""
    if state != "running":
        return None
    if not st.total or st.step < st.total:
        return "training"
    scored = len(st.evals) >= 2 and st.evals[-1][0] >= st.total - 1
    return "scoring" if held["eval_loss_start"] is not None and not scored else "saving"


def _doing(phase: Optional[str], rows: Optional[int]) -> list[str]:
    """The line a card adds once a run's steps are done and it still runs, saying why."""
    if phase == "scoring":
        return [f"   scoring its {rows} held-out rows" if rows else "   scoring its held-out rows"]
    return ["   last step done: saving what it trained"] if phase == "saving" else []


def _now(r: runs.Run) -> runs.Run:
    """The run as it is now; a run whose folder went counts as failed."""
    hit = next((x for x in runs.all_runs() if x.id == r.id), None)
    return hit if hit is not None else replace(r, state="failed")


def snapshot(r: runs.Run, *, frame: bool = False, wait: Optional[int] = None,
             card_gib: Optional[float] = None) -> RunLiveOutput:
    st = TrainState(model=r.model, output_dir=r.output_dir, run=r.id, total=r.total or 0)
    events = _Events(st)
    events.read()
    if wait and r.state in LIVE:
        target, until = st.step + wait, time.monotonic() + WAIT_MAX_S
        while st.step < target and time.monotonic() < until:
            time.sleep(1.0)
            events.read()
            r = _now(r)
            if r.state not in LIVE:
                events.read()  # the last steps, written before it ended
                break
    end = time.time() if r.state in LIVE else events.last
    st.elapsed = (end - events.first) if events.first and end else 0.0
    live = r.state in LIVE
    recent = st.step_times[-20:]
    left = st.total - st.step if live and st.total else 0
    eta = sum(recent) / len(recent) * left if recent and left else None
    held = runs_report.held_out(r.output_dir)
    if not live:  # what it ended on, the same number its report says
        said = (runs_report._read(Path(r.output_dir) / "receipt.json") or {}).get(
            "peak_vram_allocated_gb")
        st.vram_peak_gb = said if isinstance(said, (int, float)) and said > 0 else st.vram_peak_gb
    learned = runs_report.learned(held, *runs_report.preference(r.output_dir)) if (
        under_agent() and r.state in runs.TERMINAL) else None
    behind = runs_report.queued_behind(r)
    st.tokens_per_sec = events.tokens / events.seconds if events.seconds else 0.0  # the run's mean
    card = agent_text.card(st, r.state, eta, card_gib)
    if behind and not st.losses:
        card[1] = f"   queued behind {behind}"
    phase = _phase(r.state, st, held)
    doing = _doing(phase, held["eval_rows"])  # at 60/60 and running: why it still runs
    rough = bool(held["eval_rows"]) and held["eval_rows"] < runs_report.FEW_ROWS
    drawn = agent_text.frame(st, r.state, eta, card_gib, learned or "", rough) if frame else None
    if drawn and doing:
        drawn[2:2] = doing
    if drawn and r.unvalidated:  # under the model's name, as the person reads the card
        drawn.insert(2, f"   {UNVALIDATED.format(r.unvalidated)}")
    return RunLiveOutput(
        learned=learned,
        run=r.id, name=r.name, state=r.state, phase=phase, model=r.model, step=st.step,
        total=st.total,
        loss=st.loss, loss_smooth=agent_text.line(st).line[-1] if st.losses else None,
        loss_first=st.losses[0] if st.losses else None,
        s_per_step=st.step_time or None, tokens_per_s=st.tokens_per_sec or None,
        vram_gib=round(c.gib(st.vram_gb), 2) if st.vram_gb else None,
        vram_peak_gib=round(c.gib(st.vram_peak_gb), 2) if st.vram_peak_gb else None,
        card_gib=card_gib, elapsed_s=round(st.elapsed, 1),
        eta_s=round(eta, 1) if eta else None,
        eval_loss_start=held["eval_loss_start"], eval_loss=held["eval_loss"],
        eval_rows=held["eval_rows"], eval_ends=held["eval_ends"],
        eval_answer_rows=held["eval_answer_rows"], eval_prefers_start=held["eval_prefers_start"],
        eval_prefers=held["eval_prefers"], best=runs_report.best(r),
        error=r.error,
        from_=r.parent, queued_behind=behind,
        support="user_asserted" if r.unvalidated else None, model_type=r.unvalidated,
        card=[*card, *doing, *([f"   {learned}"] if learned else []),
              *([f"   {UNVALIDATED.format(r.unvalidated)}"] if r.unvalidated else [])],
        frame=drawn,
        next=runs_report.offers(r),
    )
