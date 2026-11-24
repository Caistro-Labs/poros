"""One run, in depth: what ``runs <id>`` shows and ``runs report <id>``
writes as Markdown next to it.

Everything in it is read from the run's own files (action, receipt,
events): the settings, the loss curve, the peaks. No number is computed
that the run did not record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Optional

from poros import data as poros_data
from poros import runs
from poros.cli.ui.components import row_means
from poros.core import checkpoints
from poros.core.env import under_agent
from poros.recipes import rules

FEW_ROWS = 20  # under this many held-out rows, a loss moves with a few answers
LIVE = ("running", "queued", "stopping")


def _read(path: Path):
    try:
        return json.loads(runs.read_own(path) or b"null")
    except ValueError:
        return None


def _events(out: Path) -> Iterator[dict]:
    """A run's events, as its record stands: a resumed run's superseded steps folded out
    (``events.current``). A torn line (a crash, then continue) skips only itself."""
    from poros.cli.verbs.runs_follow import parsed
    from poros.core.events import current

    yield from current(parsed(runs.read_own(out / "events.jsonl") or b""))


def _count(data: dict, key: str) -> Optional[int]:
    n = data.get(key)
    return n if type(n) is int and n >= 0 else None


def held_out(output_dir: str) -> dict:
    """What a run's eval events say: the held-out loss before its first step and after its last,
    and how many rows. Its own record of the rows, or for an older run the rows its setup event
    trained on and the split in its config. A first score after training began is no "before".
    From its last score, on chats the rows whose answers all end (of those with an end to read);
    on pairs the pairs whose chosen answer scores over the rejected one, before and after."""
    out = Path(output_dir)
    scores, rows, kept, began, simulated, said = [], None, None, False, False, []
    for ev in _events(out):
        data = ev.get("data") or {}
        if ev.get("event_type") == "setup":
            kept = data.get("rows")
        loss = data.get("eval_loss")
        if ev.get("event_type") != "eval" or not isinstance(loss, (int, float)) or (
                not math.isfinite(loss)):
            continue
        began = began or (not scores and not ev.get("step"))
        scores.append(loss)
        said.append(data)
        rows = data.get("rows") if isinstance(data.get("rows"), int) else rows
        simulated = simulated or bool(data.get("simulated"))
    if scores and rows is None and isinstance(kept, int):
        config = _read(out / "config.resolved.json") or {}
        if not config.get("eval_dataset") and (config.get("eval_split") or config.get("eval_rows")):
            rows = poros_data.held_of_kept(kept, config.get("eval_split") or 0,
                                           config.get("eval_rows") or 0)
    if began and said[0].get("held") and said[-1].get("held") != said[0].get("held"):
        began = False  # its last score is of other rows (a resume with another held-out file):
        # no start to compare it with, as the run itself says (poros.train.evals.settle_start)
    after = said[-1] if len(scores) > int(began) else {}
    return {"eval_loss_start": scores[0] if began else None,
            "eval_loss": scores[-1] if len(scores) > int(began) else None,
            "eval_rows": rows if scores else None,
            "eval_ends": _count(after, "ends"), "eval_answer_rows": _count(after, "answer_rows"),
            "eval_prefers_start": _count(said[0], "prefers") if began else None,
            "eval_prefers": _count(after, "prefers"),
            **({"simulated": True} if simulated else {})}


def preference(output_dir: str) -> tuple[Optional[tuple[int, float]], bool,
                                         Optional[tuple[int, int]]]:
    """A pairs run's own signal, as its step events record it: over its last steps, how often the
    chosen answer scored over the rejected one (DPO's accuracy, on training pairs). Whether it
    trained on pairs at all. And (pairs whose two sides it trains as the same tokens, its pairs),
    from its last setup event, when it counted them: such a pair scores alike on both sides, so
    it never counts as chosen over rejected."""
    out, pairs, said, same = Path(output_dir), False, [], None
    for ev in _events(out):
        data = ev.get("data") or {}
        if ev.get("event_type") == "setup" and data.get("data") == "pairs":
            pairs = True
            n, rows = data.get("same_pairs"), data.get("rows")
            if type(n) is int and type(rows) is int and 0 <= n <= rows:
                same = (n, rows)
        accuracy = data.get("accuracy")
        if ev.get("event_type") == "step" and isinstance(accuracy, (int, float)) and (
                math.isfinite(accuracy)):
            said.append(float(accuracy))
    pairs = pairs or (_read(out / "config.resolved.json") or {}).get("data_shape") == "pairs"
    return (last_rows(said) if said else None), pairs or bool(said), same


def learned(held: dict, prefs: Optional[tuple[int, float]] = None, pairs: bool = False,
            same: Optional[tuple[int, int]] = None) -> str:
    """The held-out loss in plain words, from those numbers alone, and on chats whether the answers
    end. A pairs run's held-out loss is DPO's, with how many held-out pairs prefer the chosen
    answer; one that held none out says its training pairs' accuracy, said to be on training
    pairs, or that it has no measure; and how many of its pairs could count, when some have the
    same two sides."""
    start, end, rows = held["eval_loss_start"], held["eval_loss"], held["eval_rows"]
    noun = "pairs" if pairs else "rows"
    could = (f"; {same[1] - same[0]} of {same[1]} pairs could count: {same[0]} "
             f"{'is' if same[0] == 1 else 'are'} the same on both sides as it trains them"
             if same and same[0] else "")
    if start is None and end is None and prefs:
        steps, accuracy = prefs
        said = (f"chosen over rejected on {accuracy:.0%} of training pairs over its last {steps} "
                f"step{'s' * (steps != 1)}; training pairs, not held out{could}")
    elif start is None and end is None and pairs:
        said = f"no held-out measure for preference training{could}"
    elif start is None and end is None:
        said = "no held-out loss: this run scored no held-out rows"
    elif end is None:
        said = f"held-out loss {start:.3f} before training, not scored after it"
    elif start is None:
        said = f"held-out loss {end:.3f} after training, not scored before it"
    else:
        said = (f"held-out {'DPO ' * pairs}loss {start:.3f} to {end:.3f}"
                + (f" on {rows} {noun}" if rows else ""))
        if start > 0:
            said += f", {abs(end - start) / start:.0%} {'higher' if end > start else 'lower'}"
        if rows and rows < FEW_ROWS:
            said += f"; few {noun}, so this is rough"
    k, n = held.get("eval_ends"), held.get("eval_answer_rows")
    if k is not None and n:
        said += f"; ends its answers {k} of {n}"
    chose, before = held.get("eval_prefers"), held.get("eval_prefers_start")
    if chose is not None and rows:
        said += f"; chosen over rejected on {chose} of {rows} held-out pairs" + (
            f", from {before}" if before is not None else "") + (could if pairs else "")
    return f"simulated: {said}" if held.get("simulated") else said


def queued_behind(r: runs.Run) -> Optional[str]:
    """What a queued job waits for on its GPU, or None."""
    if r.kind != "job" or r.state != "queued":
        return None
    from poros import jobs

    every = jobs.all_jobs()
    me = next((j for j in every if j.job_id == r.id), None)
    return jobs.waits_behind(me, every) if me else None


def row(r: runs.Run) -> dict:
    """A run as a result lists it: with its held-out numbers, what a queued one waits for, and
    once over, under `poros agents`, those in words."""
    held = held_out(r.output_dir)
    words = {"learned": learned(held, *preference(r.output_dir))} if (
        under_agent() and r.state in runs.TERMINAL) else {}
    waits = {"queued_behind": queued_behind(r)} if r.state == "queued" else {}
    return {**r.to_dict(), **held, "best": best(r), **waits, **words}


def best(r: runs.Run) -> Optional[str]:
    """The checkpoint with the lowest held-out loss of those the run keeps, by its path."""
    from poros.cli.verbs import checkpoints as listed

    return next((x.path for x in listed.of_run(r) if x.best and x.type == "train"), None)


def scores_itself(output_dir: str) -> bool:
    """Whether `poros eval <run>` has rows to score with no --data: the rows or pairs the run
    held out, its --eval file, or a pairs run's own pairs. A chats run that held none out (a
    file under 40 rows, or eval_split=0) and a classic run have none, and eval refuses them."""
    config = _read(Path(output_dir) / "config.resolved.json")
    if not isinstance(config, dict):  # no record of what it trained on: not known, not withheld
        return True
    if config.get("eval_dataset") or config.get("data_shape") == "pairs":
        return True
    held = held_out(output_dir)
    return held["eval_loss"] is not None or held["eval_loss_start"] is not None


def offers(r: runs.Run) -> list[str]:
    """What a run offers next. Under `poros agents` by where it is: going, it is watched; stopped or
    failed, it is continued from its checkpoint, or with none started again; finished, it is
    scored (where it has rows to score: ``scores_itself``), exported and kept, the first two
    while its adapter is there. Built from a checked id alone."""
    out = Path(r.output_dir)
    # a run started from another's adapter is no recipe (`recipes save` refuses it): not offered
    keep = [] if r.parent or (_read(out / "config.resolved.json") or {}).get(
        "init_adapter") else rules.next_for(r.id)
    if not under_agent():
        return keep if r.state == "succeeded" else []
    if not rules.WORD.fullmatch(r.id):
        return []
    if r.state in LIVE:
        return [f"poros runs {r.id} --live --wait 30"]
    if r.state in ("cancelled", "failed"):  # why it failed is the result's own `error`
        if checkpoints.list_checkpoints(out):
            return [f"poros runs continue {r.id}"]
        again = restart_command(r.folder)
        return [again] if again else []
    if r.state != "succeeded":
        return []
    # its own folder: a second run's export would land in the first one's
    use = [*([f"poros eval {r.id}"] if scores_itself(r.output_dir) else []),  # one that works
           f"poros export {r.id} --format adapter --output ./adapter-{r.id[-4:]}"]
    simulated = (_read(out / "receipt.json") or {}).get("simulated")  # it writes no adapter
    there = checkpoints.has_weights(out) or simulated
    return [*(use if there else []), *keep]


# what a command's user said yes to: a command offered again asks again
_CONSENTS = ("--yes", "-y", "--download", "--force", "--overwrite", "--allow-unvalidated",
             "--unvalidated", "--trust-remote-code")


def restart_command(folder: str) -> Optional[str]:
    """The command that starts a run again, from its own record (``action.json``), with no
    consent in it: a run with nothing to continue from is started as it was. None for a run with
    a loop file: a record's Python file is never offered as a command to run."""
    import shlex

    record = _read(Path(folder) / "action.json") or {}
    said = record.get("command")
    if not isinstance(said, str) or not said.startswith("poros train "):
        return None
    try:
        words = shlex.split(said)
    except ValueError:
        return None
    config = record.get("config")
    if (isinstance(config, dict) and config.get("loop")) or any(
            w.endswith(".py") for w in words):
        return None
    return shlex.join(w for w in words if w not in _CONSENTS)


def last_rows(losses: list[float]) -> tuple[int, float]:
    """(how many steps, their mean) at the run's end: one step swings with its rows. As row_means
    cuts them, or the whole of a run too short for it."""
    points = row_means(losses)
    if not points:
        return len(losses), math.fsum(losses) / len(losses)
    return points[-1][0] - points[-2][0], points[-1][1]


def facts(folder: str, output_dir: str) -> dict:
    """What the run recorded: its action, its receipt, and per step the loss,
    the step time and the memory in use (steps counted from 1)."""
    out = Path(output_dir)
    action = _read(Path(folder) / "action.json") or {}
    receipt = _read(out / "receipt.json") or {}
    losses, speeds, vram, held, accs = [], [], [], [], []
    extra: dict = {}  # the loss's own numbers at the last step (DPO: accuracy, margin)
    parts = 1  # the stretches it trained in: one more for each resume
    for ev in _events(out):
        data = ev.get("data") or {}
        parts += ev.get("event_type") == "resume"
        if ev.get("event_type") == "eval" and data.get("eval_loss") is not None:
            held.append(data["eval_loss"])
        if ev.get("event_type") != "step" or data.get("loss") is None:
            continue
        step = ev["step"] + 1  # events count from 0
        losses.append((step, data["loss"]))
        if data.get("s_per_step") is not None:
            speeds.append(data["s_per_step"])
        if data.get("footprint"):  # a Mac: the one pool's peak
            vram.append(data["footprint"] / 2**30)
        elif data.get("vram_allocated_mb") is not None:
            vram.append(data["vram_allocated_mb"] / 1024)
        extra = {k: v for k, v in data.items()
                 if k not in _STEP_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)}
        if "accuracy" in extra and math.isfinite(extra["accuracy"]):
            accs.append(float(extra["accuracy"]))
    tail = last_rows([v for _, v in losses]) if losses else None
    acc_over = 0
    if accs:  # one step's accuracy swings with its pairs: the mean over the run's last steps,
        acc_over, extra["accuracy"] = last_rows(accs)  # the window `learned` names (preference)
    if (receipt.get("config") or {}).get("gradient_accumulation_steps") == 1:
        # one row a step: the run, not its hardest rows, as on the card
        losses = [(losses[n - 1][0], v) for n, v in row_means([v for _, v in losses])]
    return {"action": action, "receipt": receipt, "losses": losses, "speeds": speeds,
            "vram_gib": vram, "extra": extra, "held_out": held, "tail": tail, "parts": parts,
            "acc_over": acc_over}


def scores(output_dir: str) -> list:
    """Each held-out pass the run recorded, whole: the run card's ledger."""
    from poros.cli.ui import run_card as rc

    out, done = [], 0  # the steps done when it scored: a pass before any step is the base
    for ev in _events(Path(output_dir)):
        if ev.get("event_type") == "resume" and type(ev.get("step")) is int:
            done = ev["step"]
        if ev.get("event_type") == "step" and type(ev.get("step")) is int:
            done = ev["step"] + 1  # events count from 0
        elif ev.get("event_type") == "eval":
            got = rc.score(done, ev.get("data"))
            if got is not None:
                out.append(got)
    return out


def evals(output_dir: str) -> list:
    """The held-out loss the run recorded: (step, loss), in order."""
    out = []
    for ev in _events(Path(output_dir)):
        loss = (ev.get("data") or {}).get("eval_loss")
        if ev.get("event_type") == "eval" and loss is not None:
            out.append((ev.get("step") or 0, float(loss)))
    return out


# what every step event carries; anything else a loss added
_STEP_KEYS = frozenset({"loss", "grad_norm", "lr", "s_per_step", "vram_allocated_mb",
                        "vram_reserved_mb", "vram_peak_mb", "tokens", "simulated",
                        "footprint", "swap", "pressure"})


# What a run trained with --unvalidated says where a person reads it, with its model_type.
UNVALIDATED = "not validated: model_type {}, no parity guarantee"


def write(run: runs.Run) -> str:
    f = facts(run.folder, run.output_dir)
    action, receipt, losses = f["action"], f["receipt"], f["losses"]
    lines = [f"# {run.name}", "", f"Run `{run.id}`, {run.state}."]
    if receipt.get("simulated"):
        lines += ["", "**Simulated**: no GPU ran this; its numbers are a preview."]
    if run.unvalidated:
        said = UNVALIDATED.format(run.unvalidated)
        lines += ["", f"**{said[0].upper()}{said[1:]}**: it trained with `--unvalidated`."]
    held = held_out(run.output_dir)
    if held["eval_loss_start"] is not None or held["eval_loss"] is not None:
        said = learned({**held, "simulated": False})
        lines += ["", said[0].upper() + said[1:] + "."]
    elif f["tail"]:
        steps, mean = f["tail"]
        lines += ["", f"Mean loss over the last {steps} steps: {mean:.4f}. No rows were held out."]
    lines += ["", "| | |", "|---|---|"]
    rows = [
        ("model", run.model),
        ("support", "user_asserted" if run.unvalidated else None),
        ("model_type", run.unvalidated),
        ("data", action.get("data")),
        ("steps", f"{run.step or 0} of {action.get('steps')}" if action.get("steps") else None),
        ("length", action.get("seq_len")),
        ("rank", action.get("rank")),
        ("recipe", receipt.get("recipe")),
        ("last step's loss", receipt.get("final_loss")),
        *[(k.replace("_", " "), f"{v:.4g}") for k, v in (f.get("extra") or {}).items()],
        ("wall time, s", receipt.get("wall_s")),
        ("s/step", receipt.get("s_per_step")),
        ("peak memory, GB" if receipt.get("platform") == "darwin" else "peak VRAM allocated, GB",
         receipt.get("peak_vram_allocated_gb")),
        ("peak host RAM, GB", receipt.get("peak_host_ram_gb")),
        ("GPU", (receipt.get("gpu") or {}).get("name")),
    ]
    lines += [f"| {k} | {v} |" for k, v in rows if v is not None]
    if losses:
        every = max(1, len(losses) // 20)
        lines += ["", "## Loss", "", "| step | loss |", "|---:|---:|"]
        lines += [f"| {s} | {v:.4f} |" for s, v in losses[::every]]
        if losses[-1] not in losses[::every]:
            lines.append(f"| {losses[-1][0]} | {losses[-1][1]:.4f} |")
    if action.get("command"):
        lines += ["", "Reproduce:", "", "```", action["command"], "```"]
    path = Path(run.folder) / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
