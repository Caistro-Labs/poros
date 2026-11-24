"""A run's checkpoints, by one path, for whoever lists, reads, copies or deletes them.

    poros://<run-id>/weights/<step>    the checkpoint after <step> steps: a run continues from it
    poros://<run-id>/weights/final     the run's finished adapter
    poros://<run-id>/weights/best      the step with the lowest held-out loss of those it keeps

``<run-id>/weights/final`` is the same path. A run id is looked up among the runs and never joined
to a folder, and a step is digits, so a path stays in its run's own files. A step's folder is the
one the listing found, by the name the trainer gives it: checkpoint-007 is no step. A run's folder
can sit in a project: a link, a pipe or a file with a second name there is no file of a checkpoint,
and is neither read nor copied. What deletes holds the run, as training does.
"""

from __future__ import annotations

import datetime as dt
import heapq
import json
import math
import os
import re
import shlex
import shutil
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from poros import runs
from poros.cli.schemas import CheckpointRow, CheckpointRunSize, CheckpointsOutput
from poros.cli.ui import components as c
from poros.cli.ui.text import Line, S
from poros.cli.verbs.export import ADAPTER_FILES
from poros.cli.words import runs as run_words
from poros.core import checkpoints as saved
from poros.core.hub_files import portable_manifest
from poros.core.refusal import Refusal

FINAL = "final"
BEST = "best"
LIVE = ("running", "queued", "stopping")
_PATH = re.compile(r"(?:poros://)?([A-Za-z0-9][\w.-]*)/weights/(final|best|[0-9]{1,18})",
                   re.ASCII)
_COUNTS = {"rows": "eval_rows", "ends": "eval_ends", "answer_rows": "eval_answer_rows",
           "prefers": "eval_prefers"}  # a score's counts, by the name a row gives each
_FOLDER = re.compile(r"checkpoint-(0|[1-9][0-9]{0,17})", re.ASCII)  # as the trainer names one


def parse(path: str) -> tuple[str, str]:
    """(run id, step, "final" or "best"), or a refusal that shows the form."""
    hit = _PATH.fullmatch(path)
    if hit is None:
        raise Refusal("E_USAGE", "Not a checkpoint path", "a checkpoint path is "
                      "poros://<run-id>/weights/<step>, poros://<run-id>/weights/final, or "
                      "poros://<run-id>/weights/best")
    run, step = hit.groups()
    return run, step if step in (FINAL, BEST) else str(int(step))


def step_of(r: runs.Run, step: str) -> str:
    """``step`` as the checkpoint it names: best is the step with the lowest held-out loss of
    those the run keeps, refused when none has a score."""
    if step != BEST:
        return step
    hit = next((x for x in of_run(r) if x.best and x.type == "train"), None)
    if hit is None:
        raise Refusal("E_NOT_FOUND", "No best checkpoint",
                      f"run {r.id} keeps no checkpoint with a held-out score",
                      context=["a recipe run with held-out rows scores them at each checkpoint"],
                      fix=f"poros runs checkpoints {r.id}", note="lists them")
    return parse(hit.path)[1]


def score(folder: Path) -> dict:
    """A checkpoint's held-out score, as its run wrote it: the loss and the counts, each of its
    kind, or nothing. A run's folder can sit in a project, so the file is read for numbers only."""
    try:
        said = json.loads(runs.read_own(folder / saved.SCORE_FILE, cap=4096) or b"null")
    except (ValueError, RecursionError):
        return {}
    loss = said.get("eval_loss") if isinstance(said, dict) else None
    if type(loss) not in (int, float) or not math.isfinite(loss):
        return {}
    out = {"eval_loss": float(loss)}
    for name, field in _COUNTS.items():
        n = said.get(name)
        if type(n) is int and 0 <= n <= 10**9:
            out[field] = n
    return out


def _run(said: str, whole: bool = False) -> runs.Run:
    """A run by the forms `run` takes. ``whole``: its whole id alone, for what deletes."""
    every = runs.all_runs()
    ids = [r.id for r in every]
    found = said if whole else run_words.resolve(said, ids, newest=ids[0] if ids else None,
                                                 names=[(r.name, r.id) for r in every])
    hit = next((r for r in every if r.id == found), None)
    if hit is None:
        why = found.reason if isinstance(found, run_words.RunMiss) else runs.miss(said)
        raise Refusal("E_NOT_FOUND", "No such run", why, fix="poros runs", note="lists them")
    return hit


def _plain(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:  # gone already: a run prunes its checkpoints as it goes
        return False


def _adapter(out: Path) -> list[Path]:
    """The finished adapter's files, if weights are among them: at the top of the run's folder."""
    files = [out / name for name in ADAPTER_FILES if _plain(out / name)]
    return files if any(f.name in saved._WEIGHT_FILES for f in files) else []


def _folders(out: Path) -> dict[int, Path]:
    """Steps it holds -> a complete checkpoint folder, by its exact name. Every trainer, and
    the dev simulator, names a checkpoint by its last step counted from 0
    (``checkpoint-<global_step>``): checkpoint-13 holds 14 steps, and a resume from it starts at
    the 15th. A link is none, whatever it leads to."""
    try:
        entries = list(os.scandir(out))
    except OSError:
        return {}
    found = {}
    for entry in entries:
        hit = _FOLDER.fullmatch(entry.name)
        if hit and entry.is_dir(follow_symlinks=False) and saved.is_complete(Path(entry.path)):
            found[int(hit[1]) + 1] = Path(entry.path)
    return found


def _under(folder: Path) -> list[Path]:
    """The regular files under ``folder``, itself no link: a walk follows none."""
    try:
        if not stat.S_ISDIR(folder.lstat().st_mode):
            return []
    except OSError:
        return []
    return sorted(Path(top, name) for top, _, names in os.walk(folder) for name in names
                  if _plain(Path(top, name)))


def _losses(out: Path) -> dict[int, float]:
    """Each finished step's loss. A step event's ``step`` is the step before it, from 0."""
    from poros.cli.verbs.runs_follow import parsed
    from poros.core.events import current

    losses = {}
    for event in current(parsed(runs.read_own(out / "events.jsonl") or b"")):  # as it stands
        if (isinstance(event, dict) and event.get("event_type") == "step"
                and isinstance(event.get("step"), int) and isinstance(event.get("data"), dict)):
            loss = event["data"].get("loss")
            if isinstance(loss, (int, float)):
                losses[event["step"] + 1] = loss
    return losses


def _row(r: runs.Run, step: str, files: Sequence[Path], anchor: Path,
         losses: dict[int, float], held: Optional[dict] = None) -> CheckpointRow:
    final = step == FINAL
    try:
        when = anchor.stat().st_mtime
        size = sum(f.lstat().st_size for f in files)
    except OSError:  # a run pruned it while this looked
        when, size = 0.0, 0
    return CheckpointRow(
        path=f"poros://{r.id}/weights/{step}", run=r.id, model=r.model,
        type="adapter" if final else "train", step=r.step if final else int(step),
        loss=r.loss if final else losses.get(int(step)), size_bytes=size,
        created=dt.datetime.fromtimestamp(when, dt.timezone.utc).isoformat(timespec="seconds"),
        **(held or {}))


def of_run(r: runs.Run) -> list[CheckpointRow]:
    """A run's checkpoints: the finished adapter, then the newest step first. A finished run's
    last step and its adapter are one moment, and each says the other's path in ``same_as``: a run
    stopped or going on since may hold an adapter of an earlier end, so only a finished one. Each
    step holds its held-out score, the adapter its last step's, and the lowest is ``best``."""
    out, losses = Path(r.output_dir), _losses(Path(r.output_dir))
    adapter = _adapter(out)
    rows = [_row(r, FINAL, adapter, adapter[0], losses)] if adapter else []
    for step, folder in sorted(_folders(out).items(), reverse=True):
        rows.append(_row(r, str(step), _under(folder), folder, losses, score(folder)))
    last = next((row for row in rows[1:] if row.step == r.step), None) if adapter else None
    if last is not None and r.state == "succeeded" and r.step is not None:
        rows[0].same_as, last.same_as = last.path, rows[0].path
        for field in ("eval_loss", *_COUNTS.values()):
            setattr(rows[0], field, getattr(last, field))
    scored = [row for row in rows if row.type == "train" and row.eval_loss is not None]
    if scored:  # on a tie the earlier step: as good, on less training
        best = min(scored, key=lambda row: (row.eval_loss, row.step))
        best.best = True
        if rows[0].type == "adapter":
            rows[0].best = rows[0].same_as == best.path
    return rows


def _every(run_id: Optional[str], whole: bool = False) -> list[CheckpointRow]:
    rows = [_run(run_id, whole)] if run_id else runs.all_runs()
    return [row for r in rows for row in of_run(r)]


def _one(path: str, whole: bool = False) -> tuple[runs.Run, CheckpointRow]:
    run_id, step = parse(path)
    r = _run(run_id, whole)
    step = step_of(r, step)
    row = next((x for x in of_run(r) if parse(x.path)[1] == step), None)
    if row is None:
        raise Refusal("E_NOT_FOUND", "No such checkpoint", f"run {r.id} has no checkpoint {step}",
                      fix=f"poros runs checkpoints {r.id}", note="lists them")
    return r, row


def _files(r: runs.Run, step: str) -> tuple[Optional[Path], list[Path]]:
    """Where a checkpoint is, and its files: the folder the listing finds, never one named here.
    None: it is gone."""
    out = Path(r.output_dir)
    if step == FINAL:
        return out, _adapter(out)
    folder = _folders(out).get(int(step))
    return folder, _under(folder) if folder else []


def listing(run_id: Optional[str], limit: int) -> CheckpointsOutput:
    """Newest first, across runs, by when each was saved: a run continued later lists above one
    created after it. A run's own keep their order (its finished adapter, then its newest step).
    ``limit`` of them, 0 for all."""
    which = [_run(run_id)] if run_id else runs.all_runs()
    every = list(heapq.merge(*map(of_run, which), key=lambda row: row.created, reverse=True))
    shown = every[:limit] if limit else every
    return CheckpointsOutput(do="list", checkpoints=shown, more=len(every) - len(shown))


def info(path: str) -> CheckpointsOutput:
    r, row = _one(path)
    root, files = _files(r, parse(row.path)[1])
    row.files = [str(f.relative_to(root)) for f in files] if root else []
    return CheckpointsOutput(do="info", checkpoints=[row])


def _copy(file: Path, to: Path, shown: str) -> None:
    """One file of a checkpoint, copied from what was opened: no link, no pipe, and no file with a
    second name, which could be anyone's."""
    try:
        fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise Refusal("E_USAGE", "Not copied", f"{shown} could not be read, or is a link") from None
    with os.fdopen(fd, "rb") as src:
        seen = os.fstat(src.fileno())
        if not stat.S_ISREG(seen.st_mode) or seen.st_nlink != 1:
            raise Refusal("E_USAGE", "Not copied", f"{shown} is not a plain file with one name",
                          context=["a checkpoint's files have one name: this one has another"])
        with open(to, "xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    os.utime(to, ns=(seen.st_atime_ns, seen.st_mtime_ns))


def download(path: str, parent: str, force: bool) -> CheckpointsOutput:
    """A copy, in a new folder of its own in ``parent``, named after it. It lands whole or not at
    all, and only --force replaces one already there: a link by that name goes, and is not gone
    through."""
    r, row = _one(path)
    root, files = _files(r, parse(row.path)[1])
    if root is None:
        raise Refusal("E_NOT_FOUND", "No such checkpoint", f"{row.path} is gone",
                      fix=f"poros runs checkpoints {r.id}", note="lists them")
    target = copy_of(row.path, parent)
    into = target.parent
    if os.path.lexists(target) and not force:
        raise Refusal("E_USAGE", "Already there", f"{target} already exists",
                      fix=shlex.join(["poros", "agents", "checkpoint", "download", row.path,
                                      "-o", str(into), "--force"]), note="replaces it")
    if os.path.lexists(into) and not into.is_dir():
        raise Refusal("E_USAGE", "Not a folder", f"{into} is a file: -o names a folder")
    into.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=into))
    try:
        for file in files:
            copy = stage / file.relative_to(root)
            copy.parent.mkdir(parents=True, exist_ok=True)
            _copy(file, copy, str(file.relative_to(root)))
        if (stage / "poros_config.json").is_file() and (kept := portable_manifest(root)):
            (stage / "poros_config.json").write_bytes(kept)  # a copy leaves the run: no one's paths
        if os.path.islink(target) or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        os.rename(stage, target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    row.files = [str(f.relative_to(root)) for f in files]
    return CheckpointsOutput(do="download", checkpoints=[row], output=os.path.abspath(target))


def copy_of(path: str, parent: str) -> Path:
    """Where a copy of this checkpoint lands: <parent>/<run>_weights_<step>."""
    run, step = parse(path)
    return Path(parent).expanduser() / f"{run}_weights_{step}"


def _instant(word: str) -> dt.datetime:
    """A date or a date and time, in ISO 8601. Without a zone it is UTC."""
    try:
        when = dt.datetime.fromisoformat(word)
    except ValueError:
        raise Refusal("E_USAGE", "Not a date", f"{word} is not a date",
                      context=["dates are ISO 8601, in UTC: 2026-11-05, 2026-11-05T08:00:00Z"]
                      ) from None
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


def select(paths: Iterable[str], run_id: Optional[str] = None, before: Optional[str] = None,
           after: Optional[str] = None) -> list[CheckpointRow]:
    """What these words name to delete. A run still going holds its files: none of it goes."""
    upper, lower = before and _instant(before), after and _instant(after)
    rows = [_one(path, whole=True)[1] for path in paths]
    if run_id or not rows:
        rows += _every(run_id, whole=True)
    rows = list({row.path: row for row in rows
                 if (not upper or dt.datetime.fromisoformat(row.created) < upper)
                 and (not lower or dt.datetime.fromisoformat(row.created) >= lower)}.values())
    going = {r.id: r.state for r in runs.all_runs() if r.state in LIVE}
    for run in {*(row.run for row in rows), run_id}:
        if run in going:
            raise Refusal("E_USAGE", "Still running", f"run {run} is {going[run]}, and its "
                          "checkpoints stay until it ends", fix=f"poros runs stop {run}",
                          note="stops it at a checkpoint")
    return rows


def warnings(rows: Sequence[CheckpointRow]) -> list[str]:
    """What a delete takes from a stopped run: the checkpoint it continues from."""
    said, going = [], {row.path for row in rows}
    for r in {row.run: None for row in rows}:
        run = next((x for x in runs.all_runs() if x.id == r), None)
        if run is None or run.state not in ("cancelled", "failed"):
            continue
        steps = sorted(_folders(Path(run.output_dir)), reverse=True)
        kept = [n for n in steps if f"poros://{run.id}/weights/{n}" not in going]
        if steps and f"poros://{run.id}/weights/{steps[0]}" in going:
            said.append(f"run {run.id} continues from its newest checkpoint: with step "
                        f"{steps[0]} gone, " + (f"from step {kept[0]}" if kept else
                                                "it cannot continue"))
    return said


@contextmanager
def _holding(r: runs.Run) -> Iterator[None]:
    """The run held, as training holds it (a job's runner its lock, a run its run.lock): a continue
    that starts now refuses, and one going refuses this. Without flock the state check stands
    alone, as it does for training."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(Path(r.folder) / ("runner.lock" if r.kind == "job" else "run.lock"), "a") as lock:
        for _ in range(20):  # a reader's probe holds it for an instant
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.05)
        else:
            raise Refusal("E_USAGE", "Still running", f"run {r.id} is training, and its "
                          "checkpoints stay until it ends", fix=f"poros runs stop {r.id}",
                          note="stops it at a checkpoint")
        yield


def delete(rows: Iterable[CheckpointRow]) -> None:
    """Permanent. The finished adapter's files, or a checkpoint's folder, each run held while its
    go. What its run pruned since it was listed is gone already."""
    every = {r.id: r for r in runs.all_runs()}
    going: dict[str, list[CheckpointRow]] = {}
    for row in rows:
        going.setdefault(row.run, []).append(row)
    for run_id, its in going.items():
        if run_id not in every:
            continue
        with _holding(every[run_id]):
            for row in its:
                step = parse(row.path)[1]
                root, files = _files(every[run_id], step)
                if step == FINAL:
                    for file in files:
                        file.unlink(missing_ok=True)
                elif root is not None:
                    shutil.rmtree(root)


def storage(run: Optional[runs.Run] = None) -> CheckpointsOutput:
    """The size of each run's checkpoints, and of them all. With ``run``, that run's alone."""
    rows = []
    for r in [run] if run else runs.all_runs():
        held = of_run(r)
        if held:
            rows.append(CheckpointRunSize(run=r.id, checkpoints=len(held),
                                          size_bytes=sum(x.size_bytes for x in held)))
    return CheckpointsOutput(do="storage", runs=rows, total_bytes=sum(x.size_bytes for x in rows))


def size_text(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n} B"


def age_text(iso: Optional[str]) -> str:
    """How long ago, in its largest unit; nothing for a time a folder's file cannot say."""
    try:
        when = dt.datetime.fromisoformat(iso or "")
        gone = (dt.datetime.now(dt.timezone.utc) - when).total_seconds()
    except (ValueError, TypeError):  # TypeError: a time with no zone
        return ""
    for unit, scale in (("d", 86400), ("h", 3600), ("m", 60)):
        if gone >= scale:
            return f"{int(gone // scale)}{unit}"
    return "now"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], w: int,
          right: Iterable[int] = ()) -> list[Line]:
    """Rows that fit: the first column whole, and past the width the last columns go. A wrapped row
    reads as nothing, and a cut id cannot be copied."""
    from poros.cli.views.parts import table as view_table

    wide = min(w, c.CAP)
    shown = len(headers)

    def needs(n: int) -> int:
        widest = (max(len(str(row[i])) for row in [headers, *rows]) for i in range(n))
        return sum(widest) + 3 * (n - 1)

    while shown > 1 and needs(shown) > wide:
        shown -= 1
    return view_table(headers[:shown], [row[:shown] for row in rows], w=wide, rail=False,
                      keep=(0,), right=[i for i in right if i < shown])


def counted(n: int, noun: str, more: int = 0) -> str:
    """The line over a list: "3 checkpoints (1789 more not shown)"."""
    said = f"{n} {noun}" + ("" if n == 1 else "s")
    return f"{said} ({more} more not shown)" if more else said


def held_text(x: CheckpointRow) -> str:
    """Its held-out loss, and best on the lowest."""
    if x.eval_loss is None:
        return ""
    return f"{x.eval_loss:.4f}" + (" best" if x.best else "")


def ends_text(x: CheckpointRow) -> str:
    """What its held-out score counted besides the loss, in words: answers that end, or pairs."""
    if x.eval_ends is not None and x.eval_answer_rows:
        return f"ends its answers {x.eval_ends} of {x.eval_answer_rows}"
    if x.eval_prefers is not None and x.eval_rows:
        return f"chosen over rejected {x.eval_prefers} of {x.eval_rows}"
    return ""


def view(out: CheckpointsOutput, w: int) -> list[Line]:
    if out.do == "list":
        rows = [[x.path, x.type, size_text(x.size_bytes), age_text(x.created),
                 "" if x.step is None else f"{x.step:,}",
                 "" if x.loss is None else f"{x.loss:.4f}", held_text(x)]
                for x in out.checkpoints]
        heads = ("Checkpoint ID", "Type", "Size", "Created", "Step", "Loss", "Held out")
        return [[S(counted(len(rows), "checkpoint", out.more), "dm")],
                *(table(heads, rows, w, right=(2, 3, 4, 5)) if rows else [])]
    if out.do == "storage":
        rows = [[x.run, str(x.checkpoints), size_text(x.size_bytes)] for x in out.runs]
        return [*(table(("Run", "Checkpoints", "Size"), rows, w, right=(1, 2)) if rows else []),
                [S(f"{size_text(out.total_bytes or 0)} in all", "dm")]]
    if out.do == "delete":  # what was done, then what it was done to, as every result says it
        return [c.event("Deleted", counted(len(out.checkpoints), "checkpoint")),
                *(c.rail(S(x.path, "dm")) for x in out.checkpoints)]
    x = out.checkpoints[0]
    if out.do == "download":
        return [c.event("Copied", x.path), *c.rail_path("to", str(out.output), w, lw=4)]
    facts = [("Checkpoint ID", x.path), ("Type", x.type), ("Model", x.model),
             ("Size", size_text(x.size_bytes)), ("Created", x.created),
             ("Step", "" if x.step is None else f"{x.step:,}"),
             ("Loss", "" if x.loss is None else f"{x.loss:.4f}"),
             *([("Held out", " · ".join(filter(None, (held_text(x), ends_text(x)))))]
               if x.eval_loss is not None else []),
             ("Files", ", ".join(x.files)),
             *([("Same moment as", x.same_as)] if x.same_as else [])]
    return [[S(k.ljust(15), "dm"), S(v, "tx")] for k, v in facts]
