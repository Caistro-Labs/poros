"""`poros jobs`: background training jobs (``poros train --detach``): list, status, logs,
cancel, resume."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from poros.cli import contract
from poros.cli.contract import refuse
from poros.cli.root import _JSON_OPTION, _ui
from poros.core.errors import EXIT_CODES
from poros.core.refusal import Refusal

jobs_app = typer.Typer(
    name="jobs",
    help="Background training jobs (poros train --detach): list, status, logs, cancel, resume.",
    no_args_is_help=True,
)


def _jobs():
    """poros.jobs, or a clear refusal where flock does not exist."""
    if os.name != "posix":
        refuse("E_USAGE", "Can't run that",
               "background jobs need a POSIX system (Linux, WSL or macOS).")
    from poros import jobs

    return jobs


def _job_dir(job_id: str) -> Path:

    jobs = _jobs()
    job_dir = jobs.find(job_id)
    if job_dir is None:
        refuse("E_NOT_FOUND", "Not found", f"no job {job_id} under {jobs.jobs_root()}.",
               fix="poros jobs list", note="shows them")
    return job_dir


def _show_job(job: "JobOutput") -> None:  # noqa: F821 (lazy import)
    if contract.json_mode():
        contract.emit(job)
        return
    from poros.cli import views

    t, _, w = _ui()
    t.print(views.job(job, w))


@jobs_app.command("list")
def jobs_list(json_output: bool = _JSON_OPTION) -> None:
    """List jobs, oldest first. Example: poros jobs list --json"""
    jobs = _jobs()
    from poros.cli import views
    from poros.cli.schemas import JobsOutput

    rows = jobs.all_jobs()
    if contract.json_mode():
        contract.emit(JobsOutput(jobs=rows))
        return
    t, _, w = _ui()
    t.print(views.jobs(rows, w))


@jobs_app.command("status")
def jobs_status(job_id: str = typer.Argument(..., help="The job id."),
                json_output: bool = _JSON_OPTION) -> None:
    """Show one job. Exits 0 whenever it can read the job; the outcome is in
    its state. Example: poros jobs status 20261022-101500-ab12 --json"""
    jobs = _jobs()

    _show_job(jobs.read(_job_dir(job_id)))


@jobs_app.command("logs")
def jobs_logs(
    job_id: str = typer.Argument(..., help="The job id."),
    follow: bool = typer.Option(
        False, "--follow", "-f",
        help="Keep printing until the job ends, then exit 0 if it succeeded, 7 if not.",
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Print a job's log. With --json, print its events.jsonl rows instead
    (poros schema events). Example: poros jobs logs 20261022-101500-ab12 -f"""
    import time

    jobs = _jobs()

    job_dir = _job_dir(job_id)
    job = jobs.read(job_dir)
    # the job's own log, never a path its status.json names; events: its run's events.jsonl
    events = contract.json_mode()
    where, name = (Path(job.output_dir), "events.jsonl") if events else (job_dir, "log.txt")
    out, pos = contract.stdout(), 0
    while True:
        job = jobs.read(job_dir)  # before the read, so the last lines are never missed
        path = jobs.own_file(where, name)
        if path is not None:
            with open(path, "rb") as f:  # bytes: a write may end mid-character
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            out.flush()
            out.buffer.write(chunk)
            out.buffer.flush()
        if not follow:
            return
        if job.state in jobs.TERMINAL:
            raise typer.Exit(0 if job.state == "succeeded" else EXIT_CODES["E_JOB_FAILED"])
        time.sleep(1)


@jobs_app.command("cancel")
def jobs_cancel(job_id: str = typer.Argument(..., help="The job id."),
                json_output: bool = _JSON_OPTION) -> None:
    """Stop a job at the next step boundary, with a checkpoint (a queued job
    never starts). Model loading is not interrupted, so a job can show
    'cancelling' for minutes. Example: poros jobs cancel 20261022-101500-ab12"""
    jobs = _jobs()

    job_dir = _job_dir(job_id)
    job = jobs.read(job_dir)
    if job.state in jobs.TERMINAL:
        refuse("E_USAGE", "Can't run that", f"job {job.job_id} already {job.state}.")
    (job_dir / "stop").touch()
    _show_job(jobs.read(job_dir))


@jobs_app.command("resume")
def jobs_resume(job_id: str = typer.Argument(..., help="The job id."),
                json_output: bool = _JSON_OPTION) -> None:
    """Run a failed or cancelled job again, from its newest checkpoint (from
    the start if it has none). Example: poros jobs resume 20261022-101500-ab12"""
    jobs = _jobs()

    job_dir = _job_dir(job_id)
    job = jobs.read(job_dir)
    if job.state not in ("failed", "cancelled"):
        refuse("E_USAGE", "Can't run that",
               f"job {job.job_id} is {job.state}; only a failed or cancelled job resumes.")
    try:  # its folder grants nothing: a loop, a push, a base's code ask runs continue
        for line in jobs.settle_grants(job_dir):
            typer.echo(f"Note: {line}", err=True)
    except Refusal as r:
        contract.refused(r)
    queued = jobs.retry(job_dir, job)
    if queued is None:
        refuse("E_USAGE", "Can't run that",
               f"job {job.job_id} is held by another attempt or resume; try again.")
    _show_job(queued)
