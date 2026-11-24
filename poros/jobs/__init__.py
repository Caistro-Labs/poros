"""Background training jobs: `poros train --detach` and `poros jobs ...`.

A job is a directory, ``$POROS_HOME/jobs/<id>/`` (default ``~/.poros``),
and one runner process (``python -m poros.jobs.runner``) that owns it:

- The runner holds an flock on ``runner.lock`` for its whole life (and so
  does its training child), so "the lock is free" means "nothing is running
  this job", whatever pid the OS has reused since. A job whose status says
  queued or running with a free lock died without a result: it reads as
  failed.
- ``status.json`` has one writer at a time: the submitter or ``jobs resume``
  while no runner holds the lock, the runner after.
- Jobs queue per GPU (by UUID): a runner takes ``$POROS_HOME/gpu-<uuid>.lock``
  when no live job for that GPU was queued before it, then runs ``poros
  train`` there.
- ``stop`` is a cancel request. The training run checks it once per step,
  writes a checkpoint and exits; ``jobs resume`` continues from it.

POSIX only (flock), with POROS_HOME on a local filesystem (flock over NFS is
emulated per process). Torch-free: the training runs in the runner's child.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from poros.cli.schemas import ErrorBody, JobOutput
from poros.core.errors import EXIT_CODES, error_to_keep
from poros.runs import make_home, open_log, write_bytes_atomic

TERMINAL = ("succeeded", "failed", "cancelled")
QUEUED_CAP = 8  # jobs waiting at once: a queue with no end spends the machine with no one asked


def home() -> Path:
    return Path(os.environ.get("POROS_HOME") or Path.home() / ".poros")


def jobs_root() -> Path:
    return home() / "jobs"


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def gpu_uuid() -> str:
    """The GPU a foreground `poros train` here would use (CUDA device 0 under
    this CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER), as its UUID. A Mac is one
    GPU: its own identity, with no torch."""
    if sys.platform == "darwin":
        from poros.metal import machine_id

        return machine_id()
    from poros.core.card import visible_gpus

    if cards := visible_gpus():
        return cards[0].uuid
    import torch

    return f"GPU-{torch.cuda.get_device_properties(0).uuid}"


def hold_machine() -> Optional[Any]:
    """What loads a model outside a job takes the lock jobs queue on: a foreground run on a Mac,
    whose one pool cannot host two runs, a merged export, an eval. The open file holds it, for as
    long as the caller keeps it. None: another has it."""
    home().mkdir(parents=True, exist_ok=True)
    lock = open(home() / f"gpu-{gpu_uuid()}.lock", "a")
    return lock if try_lock(lock) else None


def is_live(job_dir: Path) -> bool:
    """A runner (or its training child) holds this job's lock."""
    try:
        with open(job_dir / "runner.lock", "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False
    except FileNotFoundError:
        return False


def try_lock(f: Any) -> bool:
    """Take an exclusive lock, riding out readers' momentary shared probes;
    False if another process holds it."""
    for _ in range(20):
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            time.sleep(0.05)
    return False


def write_status(job_dir: Path, /, **fields: Any) -> None:
    """Merge ``fields`` into status.json atomically."""
    path = job_dir / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(fields)
    if "error" in fields:
        status["error"] = error_to_keep(fields["error"])
    write_bytes_atomic(path, json.dumps(status, indent=1).encode(), sync=True)  # never a link


def error(code: str, message: str, hint: str = "") -> dict:
    return ErrorBody(code=code, exit_code=EXIT_CODES[code], message=message,
                     hint=hint).model_dump()


def read(job_dir: Path) -> JobOutput:
    """The job as a reader sees it: status.json plus what the locks say."""
    status = json.loads((job_dir / "status.json").read_text())
    if status["state"] in TERMINAL:
        return JobOutput(**status)
    stop = (job_dir / "stop").exists()
    if is_live(job_dir):
        status["state"] = "cancelling" if stop else status["state"]
    elif stop:  # its runner died, then its training stopped as asked
        status.update(state="cancelled", error=error(
            "E_JOB_CANCELLED", "cancelled; the runner exited before recording it"))
    else:
        status.update(state="failed", error=error(
            "E_JOB_FAILED", "the runner exited without writing a result",
            hint=f"see {status['log']}; poros jobs resume {status['job_id']} retries it",
        ))
    return JobOutput(**status)


def waits_behind(job: JobOutput, every: list[JobOutput]) -> Optional[str]:
    """What a queued job waits for on its GPU: the last one queued before it, else the one
    running. None: it waits for nothing, or is not queued."""
    if job.state != "queued":
        return None
    same = [j for j in every if j.gpu == job.gpu and j.job_id != job.job_id]
    ahead = sorted((j.queued_at, j.job_id) for j in same if j.state == "queued"
                   and (j.queued_at, j.job_id) < (job.queued_at, job.job_id))
    going = [j.job_id for j in same if j.state in ("running", "cancelling")]
    return ahead[-1][1] if ahead else going[0] if going else None


def first_queued(gpu: str, every: list[JobOutput]) -> Optional[str]:
    """The job that starts next on this GPU: the first one queued."""
    queued = sorted((j.queued_at, j.job_id) for j in every if j.gpu == gpu and j.state == "queued")
    return queued[0][1] if queued else None


def find(job_id: str) -> Optional[Path]:
    """The job's folder under the jobs root, by its id alone: one path component, never an
    absolute path, ``..`` or a link (a folder elsewhere is not a job of this machine's)."""
    if (not job_id or job_id in (".", "..") or "/" in job_id or "\\" in job_id
            or os.path.basename(job_id) != job_id):
        return None
    job_dir = jobs_root() / job_id
    if job_dir.is_symlink() or not job_dir.is_dir():
        return None
    return job_dir if (job_dir / "status.json").is_file() else None


def own_file(job_dir: Path, name: str) -> Optional[Path]:
    """``job_dir / name`` when it is a regular file and not a link, else None."""
    path = job_dir / name
    return path if path.is_file() and not path.is_symlink() else None


def all_jobs() -> list[JobOutput]:
    """Every readable job, oldest first. A job dir someone deleted or broke
    mid-read is skipped, not fatal: queued runners call this every second."""
    root = jobs_root()
    out = []
    for d in sorted(root.iterdir()) if root.is_dir() else ():
        try:
            out.append(read(d))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def launch(job_dir: Path) -> None:
    """Start the runner detached, and return once it holds the job lock."""
    spec = json.loads((job_dir / "spec.json").read_text())
    cwd = spec["cwd"]
    # Whoever reads a job's status, an agent can, so POROS_JOB keeps a file's words out of it.
    env = {**os.environ, "POROS_JOB": "1"}
    with open_log(job_dir / "log.txt") as log:  # never through a planted link
        proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "poros.jobs.runner", str(job_dir)], cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for _ in range(300):  # without this a reader could see "queued, no runner"
        if is_live(job_dir) or proc.poll() is not None:
            return
        time.sleep(0.05)


@contextlib.contextmanager
def _a_place_in_the_queue() -> Iterator[None]:
    """The queue's ceiling, checked and kept: one at a time from the check until a new job's runner
    holds it, or 24 parallel submits would all pass the same count. A full queue is a refusal."""
    make_home()
    jobs_root().mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(jobs_root() / "queue.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if sum(j.state == "queued" for j in all_jobs()) >= QUEUED_CAP:
            from poros.cli.contract import refuse

            refuse("E_USAGE", "The queue is full", f"{QUEUED_CAP} runs are waiting already",
                   context=["let them finish, or stop one: poros runs stop <id>"])
        yield


def submit(
    raw: dict, *, model: str, gpu: str, force: bool = False, files: Optional[Path] = None,
    name: Optional[str] = None, unvalidated: Optional[str] = None,
) -> JobOutput:
    """Create a job dir for this config and start its runner. ``files``: a folder to keep its
    artifacts under, on another disk. ``unvalidated``: the model_type --unvalidated asserted, which
    every attempt asserts again."""
    with _a_place_in_the_queue():
        return _submit_now(raw, model, gpu, force, files, name, unvalidated)


def _submit_now(raw: dict, model: str, gpu: str, force: bool, files: Optional[Path],
                name: Optional[str], unvalidated: Optional[str] = None) -> JobOutput:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_id = f"{stamp}-{secrets.token_hex(2)}"
    job_dir = jobs_root() / job_id
    make_home()
    job_dir.mkdir(parents=True, mode=0o700)
    artifacts = files / job_id if files else job_dir / "artifacts"
    artifacts.mkdir(parents=True, mode=0o700)
    spec = {**raw, "output_dir": str(artifacts)}
    (job_dir / "config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    # Relative paths in the config (a local dataset) resolve against the
    # submitter's directory on every attempt, wherever `jobs resume` runs.
    (job_dir / "spec.json").write_text(json.dumps(
        {"config": spec, "cwd": os.getcwd(), "force": force,
         **({"unvalidated": True} if unvalidated else {}),
         # what the loop file held when its submitter named it: a resume names it again
         **({"loop_sha256": loop_digest(raw["loop"])} if raw.get("loop") else {})}, indent=1))
    write_status(
        job_dir, job_id=job_id, state="queued", model=model, name=name, job_dir=str(job_dir),
        config=str(job_dir / "config.yaml"), output_dir=spec["output_dir"],
        log=str(job_dir / "log.txt"), gpu=gpu, attempt=1, queued_at=(at := now()), created_at=at,
        **({"support": "user_asserted", "model_type": unvalidated} if unvalidated else {}),
    )
    launch(job_dir)
    job = read(job_dir)
    job.next = [f"poros runs {job_id} --live --wait 30"]
    return job


def _require_settled(job_dir: Path, job: JobOutput) -> None:
    """No resume on a folder's grant: one the job's config holds was settled for this attempt."""
    from poros.core.refusal import Refusal

    try:  # no config, no grant: the runner then has nothing to train and says so
        config = yaml.safe_load((job_dir / "config.yaml").read_text()) or {}
    except OSError:
        return
    held = [k for k in GRANTS if isinstance(config, dict) and config.get(k)]
    if held and json.loads((job_dir / "spec.json").read_text()).get("settled") != job.attempt:
        raise Refusal("E_USAGE", "Not settled",
                      f"job {job.job_id}'s config holds {', '.join(held)}",
                      context=["a resume takes them from its caller, not its folder"],
                      fix=f"poros runs continue {job.job_id}")


def _more_steps(job_dir: Path, more: int) -> None:
    """A job's total, ``more`` steps further, in the config it runs and the record of it."""
    config = yaml.safe_load((job_dir / "config.yaml").read_text())
    spec = json.loads((job_dir / "spec.json").read_text())
    config["max_steps"] = spec["config"]["max_steps"] = config["max_steps"] + more
    (job_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (job_dir / "spec.json").write_text(json.dumps(spec, indent=1))


def _moved_data(job_dir: Path, data: str) -> None:
    """Where a job's data file is now, in the config it runs and the record of it: the same
    data at another path (its resume compares the contents)."""
    config = yaml.safe_load((job_dir / "config.yaml").read_text())
    spec = json.loads((job_dir / "spec.json").read_text())
    config["dataset"] = spec["config"]["dataset"] = data
    (job_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (job_dir / "spec.json").write_text(json.dumps(spec, indent=1))


def loop_digest(path: "str | Path") -> Optional[str]:
    """The sha256 of a loop file's bytes; None where it cannot be read."""
    import hashlib

    try:
        return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
    except OSError:
        return None


# What a job's folder does not say for a resume, as for poros.Run.open and `runs continue`: a
# folder can come from anyone. Its config holds what its submitter's flags said, for its first
# attempt; a resume runs the base's code or a loop file, or pushes, only on its caller's word.
GRANTS = ("trust_remote_code", "loop", "push_to_hub", "hub_model_id", "init_adapter")


def settle_grants(job_dir: Path, *, loop: Optional[str] = None, trust_remote_code: bool = False,
                  push_to: Optional[str] = None) -> list[str]:
    """Before a resume (``retry``): the job's config with what its caller said in place of what
    its folder says, or a Refusal naming the command that says it. Returns lines to show."""
    from poros.core.checkpoints import list_checkpoints
    from poros.core.refusal import Refusal

    status = json.loads((job_dir / "status.json").read_text())
    job_id = job_dir.name
    own = job_dir / "config.yaml"
    if Path(str(status.get("config", ""))) != own or own.is_symlink():
        raise Refusal("E_USAGE", "Not this job's config",
                      f"job {job_id}'s status names {status.get('config')}, not its own "
                      "config.yaml", context=["a resume trains on the job's own config only"])
    if not own.is_file():  # nothing to grant: its runner says the config is missing
        return []
    config = yaml.safe_load(own.read_text()) or {}
    spec = json.loads((job_dir / "spec.json").read_text())
    if str(config.get("output_dir")) != str(status.get("output_dir")):
        raise Refusal("E_USAGE", "Not this job's folder",
                      f"job {job_id}'s status and config name two output folders",
                      context=["a resume writes where the job's own config does"])
    said: list[str] = []
    again = f"poros runs continue {job_id}"
    if config.get("trust_remote_code") and not trust_remote_code:
        # no ready command: a hint is copied, and this grant is its caller's to give (_no_grants)
        raise Refusal("E_USAGE", "A flag, not a setting",
                      f"job {job_id}'s config says trust_remote_code, which grants nothing",
                      context=["for a base you trust: poros runs continue <id> "
                               "--trust-remote-code"])
    if config.get("loop"):
        said += _loop_given(f"job {job_id}", str(config["loop"]), spec.get("loop_sha256"), loop)
        config["loop"] = spec["config"]["loop"] = loop
    if config.get("push_to_hub") and not push_to:
        said.append(f"job {job_id} pushed to the Hub when it ended: this resume does not push; "
                    f"{again} push_to_hub=true hub_model_id=<repo> pushes")
    for key in ("push_to_hub", "hub_model_id"):
        config.pop(key, None)
        spec["config"].pop(key, None)
    if push_to:
        config["push_to_hub"] = spec["config"]["push_to_hub"] = True
        config["hub_model_id"] = spec["config"]["hub_model_id"] = push_to
    if config.get("init_adapter"):
        if not list_checkpoints(Path(str(status["output_dir"]))):
            raise Refusal("E_USAGE", "Start it again",
                          f"job {job_id} starts from another adapter and saved no checkpoint",
                          context=["a resume starts from its own checkpoint, never from an "
                                   "adapter its folder names"],
                          fix="poros train --from <run> --data <file> --detach")
        config.pop("init_adapter")  # a resume reads its own checkpoint
        spec["config"].pop("init_adapter", None)
    from poros.stores import recorded_store

    # a store_path is the folder's to READ a store there, never to build one (Run.open's rule)
    config, _ = recorded_store(config, say=said.append)
    spec["settled"] = status.get("attempt")
    own.write_text(yaml.safe_dump(config, sort_keys=False))
    (job_dir / "spec.json").write_text(json.dumps(spec, indent=1))
    return said


def _loop_given(who: str, recorded: str, digest: Optional[str],
                given: Optional[str]) -> list[str]:
    """A run's loop file, named again by its caller: the same bytes the run recorded, or a
    Refusal; a run that recorded none is said, once, as not checked. The refusal offers no ready
    command and names the file by its name alone: a folder's own path never reaches a hint."""
    from poros.core.refusal import Refusal

    if not given:
        raise Refusal("E_USAGE", "Name its loop",
                      f"{who} trains with a loop file ({Path(recorded).name})",
                      context=["a continue runs a Python file only when you name it",
                               "name it again: poros runs continue <id> --recipe <its loop file>"])
    if digest is None:
        return [f"the loop file cannot be checked: {who} recorded no hash of it, so "
                f"{Path(given).name} runs as it is now"]
    if loop_digest(given) != digest:
        raise Refusal("E_USAGE", "Another loop",
                      f"{given} is not the loop file {who} trained with (its contents differ)",
                      context=["a continue trains with the same loop, or it is another run"])
    return []


def retry(job_dir: Path, job: JobOutput, more: int = 0,
          data: Optional[str] = None) -> Optional[JobOutput]:
    """Queue a job again, ``more`` steps further if asked; None if something holds it. A resume
    goes back to where it was first queued, ahead of jobs queued after it; ``more`` goes last.
    ``data``: where its data file is now. A job whose config holds a grant (GRANTS) resumes only
    once ``settle_grants`` has put its caller's word in place for this attempt."""
    with _a_place_in_the_queue():
        with open(job_dir / "runner.lock", "a") as lock:
            if not try_lock(lock):  # a live attempt, or a concurrent resume
                return None
            if json.loads((job_dir / "status.json").read_text())["attempt"] != job.attempt:
                return None  # another resume got here first
            _require_settled(job_dir, job)
            (job_dir / "stop").unlink(missing_ok=True)
            if more:
                _more_steps(job_dir, more)
            if data:
                _moved_data(job_dir, data)
            # a resume takes back the place it was first queued at; --more is new work, at the back
            place = now() if more else (job.created_at or job.queued_at)
            write_status(job_dir, state="queued", attempt=job.attempt + 1, queued_at=place,
                         started_at=None, finished_at=None, result=None, error=None)
        launch(job_dir)
        return read(job_dir)
