"""The process that owns one job: ``python -m poros.jobs.runner <job_dir>``.

It holds the job lock for its whole life, waits its turn for the GPU, runs
``poros train --json`` on the job's config in a child process (the same code
path as a foreground run, resuming from the job's newest checkpoint if it has
one), and records the outcome in status.json.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Optional

from poros import jobs
from poros.core.checkpoints import list_checkpoints


def _wait_for_gpu(job_dir: Path, gpu: str, queued_at: str) -> Optional[IO]:
    """The GPU lock once no live job for this GPU was queued earlier; None if
    the job is cancelled first."""
    lock = open(jobs.home() / f"gpu-{gpu}.lock", "a")
    while not (job_dir / "stop").exists():
        # Not our own dir: probing our own lock would drop it where flock is
        # emulated per process (NFS).
        ahead = [j for j in jobs.all_jobs() if j.job_id != job_dir.name and j.gpu == gpu
                 and j.state == "queued" and (j.queued_at, j.job_id) < (queued_at, job_dir.name)]
        if not ahead:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                if not (job_dir / "stop").exists():
                    return lock
                break
        time.sleep(1)
    lock.close()
    return None


def _result(stdout: str) -> dict:
    lines = stdout.strip().split("\n")  # a name can hold U+2028: a line is what print wrote
    try:
        return json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        return {}


def main(job_dir: Path) -> int:
    lock = open(job_dir / "runner.lock", "a")
    if not jobs.try_lock(lock):
        print(f"another runner owns {job_dir}", file=sys.stderr)
        return 2
    status = json.loads((job_dir / "status.json").read_text())
    gpu_lock = _wait_for_gpu(job_dir, status["gpu"], status["queued_at"])
    if gpu_lock is None:
        jobs.write_status(job_dir, state="cancelled", finished_at=jobs.now(), error=jobs.error(
            "E_JOB_CANCELLED", "cancelled while queued; nothing ran"))
        return 0
    jobs.write_status(job_dir, state="running", started_at=jobs.now())
    cmd = [sys.executable, "-P", "-m", "poros.cli.main", "train", status["config"], "--yes",
           "--json", "--stop-file", str(job_dir / "stop")]
    if list_checkpoints(Path(status["output_dir"])):
        cmd += ["--resume-from", "latest"]
    spec = json.loads((job_dir / "spec.json").read_text())
    if spec["force"]:
        cmd.append("--force")
    if spec.get("unvalidated") is True:  # what --unvalidated asserted, asserted again
        cmd.append("--unvalidated")
    print(f"[poros job] attempt {status['attempt']}: {' '.join(cmd)}", flush=True)
    # The child inherits both locks: if this runner is killed, the job still
    # reads as live and the GPU as taken until the training itself ends.
    pin = {} if sys.platform == "darwin" else {"CUDA_VISIBLE_DEVICES": status["gpu"]}
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
                          env={**os.environ, **pin},
                          pass_fds=(lock.fileno(), gpu_lock.fileno()))
    out = _result(proc.stdout)
    if proc.returncode == 0:
        fields = {"state": "succeeded", "result": out}
    else:
        err = out.get("error") or jobs.error(
            "E_JOB_FAILED", f"poros train exited {proc.returncode} without a result",
            hint=f"see {status['log']}")
        state = "cancelled" if err["code"] == "E_JOB_CANCELLED" else "failed"
        fields = {"state": state, "error": err}
    jobs.write_status(job_dir, finished_at=jobs.now(), **fields)
    gpu_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
