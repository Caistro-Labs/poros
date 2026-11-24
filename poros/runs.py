"""Training runs as folders: ``$POROS_HOME/runs/<id>/`` (default ``~/.poros``).

A run folder is the source of truth for everything ``poros runs`` shows, so
it reads the same after a crash, from another terminal, or for an agent
reading files:

    action.json     the resolved train Action (poros.action.train.v1)
    status.json     state, pid, times, the outcome (poros.run.status.v1)
    events.jsonl    the trainer's per-step events (poros schema events)
    checkpoint-N/   resumable checkpoints; the adapter files at the top level
    receipt.json    what the run measured, written when it finishes
    stop            present = a stop request: checkpoint at the next step, exit

One writer per file at a time: the process running the run writes
status.json; `runs stop` only creates `stop`. Writes are atomic (temp file +
rename). A run whose status says running but whose process is gone died: it
reads as failed, and its status is given the reason (``_ended_unfinished``).
Background runs are #40's jobs (``poros.jobs``); this module lists both.
Torch-free.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from poros.core.arch_registry import TYPE_WORD
from poros.core.env import under_agent
from poros.core.errors import error_to_keep

TERMINAL = ("succeeded", "failed", "cancelled")


def home() -> Path:
    return Path(os.environ.get("POROS_HOME") or Path.home() / ".poros")


def make_home() -> None:
    """POROS_HOME, its owner's alone when it is made here: it holds runs, data paths and adapters.
    One already there is left as it is."""
    try:
        if not home().exists():
            home().mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:  # what needs it will say so
        pass


def root() -> Path:
    return home() / "runs"


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    """Sortable and short: ``1022-1530-7f3a`` (month-day, hour-minute, random). Not one a run here
    has: the minute and 16 random bits can meet."""
    stamp = _dt.datetime.now().strftime("%m%d-%H%M")
    for _ in range(64):
        said = f"{stamp}-{secrets.token_hex(2)}"
        if not os.path.lexists(root() / said):
            return said
    return f"{stamp}-{secrets.token_hex(4)}"


def write_json(path: Path, payload: dict) -> None:
    write_bytes_atomic(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def write_bytes_atomic(path: Path, raw: bytes, *, sync: bool = False) -> None:
    """``raw`` at ``path`` whole or not at all: written to a new temporary file beside it (one
    no one else can have planted, never a link), then renamed over it; the rename replaces a link
    at ``path`` itself, never the file it points to."""
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)  # as a plain write gives it
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            if sync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def open_log(path: Path, mode: str = "a"):
    """A log file of a folder that may not be this user's own, for writing (``a``ppend, or
    ``w`` to start it over), never through a link: a link there is refused (OSError)."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == "a" else os.O_TRUNC)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    elif path.is_symlink():
        raise OSError(f"{path} is a link")
    return os.fdopen(os.open(path, flags, 0o644), mode)


def create(run_id: str, action: dict, *, output_dir: Optional[str] = None) -> Path:
    """The run's folder, with its action and a queued status. ``output_dir``
    (the user's `to ./out`) holds the files instead; the run folder points at it."""
    make_home()
    d = root() / run_id
    d.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(d / "action.json", {"schema": "poros.action.train.v1", **action})
    write_json(
        d / "status.json",
        {
            "schema": "poros.run.status.v1",
            "id": run_id,
            "state": "queued",
            "pid": None,
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "output_dir": str(output_dir or d),
            "error": None,
        },
    )
    return d


def update(run_dir: Path, **fields: Any) -> dict:
    status = json.loads((run_dir / "status.json").read_text())
    status.update(fields, error=error_to_keep(fields.get("error", status.get("error"))))
    write_json(run_dir / "status.json", status)
    return status


def _alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or not 0 < pid < 2**31:
        return False  # no process, or none a folder's word can name
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_started(pid: int) -> Optional[str]:
    """What tells this process from a later one given the same pid: on Linux the boot it runs
    in and its start time in that boot's clock ticks, on a Mac the start time `ps` says. None
    where neither can be read: the pid alone then says whether it lives."""
    try:
        if os.path.isdir("/proc/self"):
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            # after the command's name, which may hold spaces and brackets: field 22
            fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
            return f"{boot}:{fields[19]}"
        import subprocess

        # in one locale and one time zone, whoever asks: the status is written by the run's
        # shell and read from any other, and `lstart` is a date in the asker's words
        said = subprocess.run(["/bin/ps", "-o", "lstart=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=5,
                              env={**os.environ, "LC_ALL": "C", "TZ": "UTC"}).stdout.strip()
        return said or None
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def running_here() -> dict:
    """What a run's status keeps of the process that trains it, beside its pid: the machine it
    is on and ``process_started``."""
    import socket

    return {"host": socket.gethostname(), "pid_started": process_started(os.getpid())}


def _lock_held(run_dir: Optional[Path]) -> bool:
    """Whether a process holds the run's lock (``run.lock``, held for as long as it trains):
    asked with a shared lock that waits for nothing, and let go at once."""
    if run_dir is None:
        return False
    try:
        import fcntl

        with open(run_dir / "run.lock", "rb") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    return False


def _gone(status: dict, run_dir: Optional[Path] = None) -> bool:
    """Whether the process a status names as running is gone. Never said of a run on another
    machine (a home shared over a network: its pid means nothing here), nor of a run whose
    lock a process holds (it is training, whatever the pid's start reads as); said of a pid
    that now belongs to another process (it started at another time)."""
    import socket

    host = status.get("host")
    if isinstance(host, str) and host != socket.gethostname():
        return False
    pid = status.get("pid")
    began = status.get("pid_started")
    if _alive(pid) and not (isinstance(began, str)
                            and process_started(pid) not in (None, began)):
        return False
    return not _lock_held(run_dir)


def _unfinished_said(step: Any, checkpoint: Any) -> str:
    """The sentence for a run whose process ended with no outcome, from how far it got and the
    checkpoint it kept: built here from a count and a folder's name, so it is ours to show
    anyone."""
    done = (f"after {step} step{'' if step == 1 else 's'}" if type(step) is int and step > 0
            else "before its first step")
    kept = (f"{checkpoint} is its newest checkpoint"
            if isinstance(checkpoint, str) and re.fullmatch(r"checkpoint-[0-9]{1,12}", checkpoint)
            else "no checkpoint was saved")
    return (f"the process ended without finishing {done} (killed, or its machine stopped); "
            f"{kept}")


def _ended_unfinished(run_dir: Path, status: dict, output_dir: str) -> dict:
    """The record of a run whose process ended with no outcome written (killed, out of memory,
    the machine stopped): failed, with how far it got and what it kept, written once so every
    later reader says the same. A folder that cannot be written still reads so."""
    from poros.core import checkpoints

    step, _ = _progress(output_dir)
    try:
        saved = checkpoints.list_checkpoints(Path(output_dir))
    except OSError:
        saved = []
    newest = saved[-1][1].name if saved else None
    try:  # its last sign of life: the last step it wrote
        ended = _dt.datetime.fromtimestamp(
            (Path(output_dir) / "events.jsonl").stat().st_mtime, _dt.timezone.utc)
        finished = ended.isoformat(timespec="seconds")
    except OSError:
        finished = now()
    error = {"code": "E_INTERRUPTED", "ended": "unfinished", "step": step,
             "checkpoint": newest, "message": _unfinished_said(step, newest)}
    try:
        # read again, and written only if it still names that process as running: a run that
        # wrote its own outcome, or was continued, since ``status`` was read keeps its word
        latest = json.loads((run_dir / "status.json").read_text())
        if not (isinstance(latest, dict) and latest.get("state") == "running"
                and latest.get("pid") == status.get("pid")
                and latest.get("pid_started") == status.get("pid_started")):
            return latest if isinstance(latest, dict) else status
        return update(run_dir, state="failed", finished_at=finished, error=error)
    except (OSError, ValueError):
        return {**status, "state": "failed", "finished_at": finished, "error": error}


@dataclass
class Run:
    """One run as ``runs`` shows it: a foreground run folder or a #40 job."""

    id: str
    kind: str  # "run" | "job"
    state: str
    model: str
    name: str
    output_dir: str
    created_at: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    step: Optional[int]
    total: Optional[int]
    loss: Optional[float]
    error: Optional[dict]
    folder: str
    parent: Optional[str] = None  # the run it started from, with --from
    unvalidated: Optional[str] = None  # --unvalidated: the model_type it asserted

    def to_dict(self) -> dict:
        """The run as a result says it. One trained with --unvalidated says ``support:
        user_asserted`` and its ``model_type``: no parity guarantee applies to it."""
        said = dict(vars(self))
        said["from"] = said.pop("parent")
        kind = said.pop("unvalidated")
        if kind:
            said.update(support="user_asserted", model_type=kind)
        return said


def read_own(path: Path, start: int = 0, cap: int = 256 << 20) -> Optional[bytes]:
    """A run's file, which can sit in a project: no link, a pipe hangs nothing, and /dev/zero or a
    file made huge fills no memory. None if it is no such file."""
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return None
            f.seek(start)
            return f.read(cap)
    except OSError:
        return None


def _words_kept(run: Run) -> Run:
    """Under `poros agents`, a run in a person's terminal says what kind of error it ended on and
    not its words: no switch kept outside words out of them. A job's, which POROS_JOB keeps out, are
    ours, and say why; and so is the sentence for a process that ended unfinished."""
    if run.error and under_agent() and run.kind == "run":
        kept = {k: v for k, v in run.error.items() if k in ("code", "exit_code")}
        if run.error.get("ended") == "unfinished":  # ours, said again from its count and name
            kept["message"] = _unfinished_said(run.error.get("step"), run.error.get("checkpoint"))
        run.error = kept
    return run


def _progress(output_dir: str) -> tuple[Optional[int], Optional[float]]:
    """Steps done and the last loss, from the newest step event in events.jsonl.

    Read backwards in growing windows: a finished run ends with a teardown
    event that can be larger than any fixed tail. A step event's ``step`` is
    0-based (the trainer's global step before the increment), so done = step+1.
    """
    path = Path(output_dir) / "events.jsonl"
    try:  # a run's files can sit in a project: no link, and a pipe does not hang this
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return None, None
            f.seek(0, os.SEEK_END)
            end = f.tell()
            window = 65536
            while True:
                start = max(0, end - window)
                f.seek(start)
                lines = f.read(end - start).decode("utf-8", "replace").splitlines()
                for line in reversed(lines[1:] if start else lines):  # [0] may be cut
                    try:
                        ev = json.loads(line)
                    except (ValueError, RecursionError):  # a line nested past Python's stack
                        continue
                    if (isinstance(ev, dict) and ev.get("event_type") == "resume"
                            and type(ev.get("step")) is int):  # resumed, no step since: its own
                        return ev["step"], None
                    if (isinstance(ev, dict) and ev.get("event_type") == "step"
                            and isinstance(ev.get("step"), int)):
                        data = ev.get("data")
                        loss = data.get("loss") if isinstance(data, dict) else None
                        return ev["step"] + 1, loss if isinstance(loss, (int, float)) else None
                if start == 0 or window >= 64 * 2**20:
                    return None, None
                window *= 4
    except OSError:
        return None, None


def _job_config(config_path: str) -> tuple[Optional[int], Optional[str]]:
    """A job's max_steps, and the checkpoint it starts from with --from: the config it runs."""
    try:
        import yaml

        said = yaml.safe_load(Path(config_path).read_text()) or {}
        steps, start = said.get("max_steps"), said.get("init_adapter")
        return (int(steps) if steps and int(steps) > 0 else None,
                start if isinstance(start, str) else None)
    except (OSError, ValueError, TypeError, AttributeError, ImportError):
        return None, None


def _said(said: dict, key: str, kind: type) -> Any:
    """A field of a run's file, if it is of its kind: a run's folder can sit in a project."""
    value = said.get(key)
    return value if isinstance(value, kind) and not isinstance(value, bool) else None


def _family(said: Any) -> Optional[str]:
    """The model_type a run asserted with --unvalidated, in a word fit to show; one that is not
    such a word still says it was asserted."""
    if not said:
        return None
    return said if isinstance(said, str) and TYPE_WORD.fullmatch(said) else "unknown"


def read(run_dir: Path) -> Run:
    """A run from its folder. ValueError: its files are not what Poros writes."""
    status = json.loads((run_dir / "status.json").read_text())
    action = json.loads((run_dir / "action.json").read_text())
    if not (isinstance(status, dict) and isinstance(action, dict)):
        raise ValueError("a run's status and action are objects")
    run_id, state, output_dir = (_said(status, k, str) for k in ("id", "state", "output_dir"))
    if not (run_id and state and output_dir):
        raise ValueError("a run's status names its id, state and folder")
    if state == "running" and _gone(status, run_dir):  # it died without writing its outcome
        status = _ended_unfinished(run_dir, status, output_dir)
        state = _said(status, "state", str) or "failed"  # failed, or what it wrote since
    if state == "running" and (run_dir / "stop").exists():
        state = "stopping"
    step, loss = _progress(output_dir)
    return _words_kept(Run(
        id=run_id,
        kind="run",
        state=state,
        model=_said(action, "model", str) or "",
        name=_said(action, "name", str) or run_id,
        output_dir=output_dir,
        created_at=_said(status, "created_at", str),
        started_at=_said(status, "started_at", str),
        finished_at=_said(status, "finished_at", str),
        step=step,
        total=_said(action, "steps", int),
        loss=loss,
        error=_said(status, "error", dict),
        folder=str(run_dir),
        parent=_said(action, "from_run", str),
        unvalidated=_family(action.get("unvalidated")),
    ))


def all_runs() -> list[Run]:
    """Every readable run and job, newest first. A folder that cannot be read is skipped: one
    being made or deleted right now, or one whose files are not what Poros writes."""
    out: list[Run] = []
    r = root()
    for d in r.iterdir() if r.is_dir() else ():
        try:
            out.append(read(d))
        except (OSError, ValueError, RecursionError):
            continue
    jobs, starts = _jobs()
    out += jobs
    by_folder = {os.path.normpath(run.output_dir): run.id for run in out}
    for run in jobs:  # a job records the checkpoint it started from: its run is the one holding it
        if run.id in starts:
            run.parent = by_folder.get(os.path.normpath(os.path.dirname(starts[run.id])))
    return sorted(out, key=lambda run: str(run.created_at or ""), reverse=True)


def miss(word: str) -> str:
    """Why no run is ``word``: its folder is here and cannot be read, or there is none."""
    if re.fullmatch(r"[A-Za-z0-9][\w.-]{0,63}", word, re.ASCII) and (root() / word).is_dir():
        return (f"run {word}'s folder can't be read: its status.json or action.json is not what "
                "Poros writes")
    return f"no run {word}"


def _jobs() -> tuple[list[Run], dict[str, str]]:
    """Every job as a run, and the checkpoint each started from, by id."""
    if os.name != "posix":
        return [], {}
    try:
        from poros import jobs
    except ImportError:
        return [], {}
    out, starts = [], {}
    for j in jobs.all_jobs():
        step, loss = _progress(j.output_dir)
        state = j.state.replace("cancelling", "stopping")
        total, start = _job_config(j.config)
        if start:
            starts[j.job_id] = start
        out.append(_words_kept(
            Run(
                id=j.job_id,
                kind="job",
                state=state,
                model=j.model,
                name=j.name or j.job_id,
                output_dir=j.output_dir,
                created_at=j.created_at or j.queued_at,  # newest first stays put on a continue
                started_at=j.started_at,
                finished_at=j.finished_at,
                step=step,
                total=total,
                loss=loss,
                error=j.error.model_dump() if j.error else None,
                folder=j.job_dir,
                unvalidated=_family(j.model_type if j.support == "user_asserted" else None),
            )
        ))
    return out, starts


def stop(run: Run) -> None:
    """Ask a run to stop: it checkpoints at the next step boundary and exits."""
    Path(run.folder, "stop").touch()
