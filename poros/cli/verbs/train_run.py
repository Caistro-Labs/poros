"""Running a resolved train Action: its run folder, the live screen, the outcome.

The folder is written before training starts and its status is kept true
whatever happens: succeeded, cancelled (a stop request or ctrl-c, with a
checkpoint to continue from when one was written), or failed with the
error's code. One process trains a run at a time (``run.lock``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Union

import typer

from poros import runs
from poros.cli import contract
from poros.cli.schemas import JobOutput, TrainOutput
from poros.cli.verbs import train_data
from poros.cli.verbs.mac import run_files
from poros.cli.words.models import base_on_chats
from poros.core.env import under_agent
from poros.core.refusal import Refusal
from poros.preflight import command

if TYPE_CHECKING:
    from poros.cli.context import Context
    from poros.cli.verbs.train import TrainAction

_CANCELLED = (7, 130)  # E_JOB_CANCELLED (a stop request), E_INTERRUPTED (ctrl-c)
# a run's own files, besides checkpoints
_PREVIOUS = ("events.jsonl", "receipt.json", "report.md", "config.resolved.json")


def config_dict(a: "TrainAction", output_dir: str) -> dict:
    raw = {
        "model_name_or_path": a.model,
        "dataset": a.data,
        "max_steps": a.steps,
        "max_seq_length": a.seq_len,
        "lora_rank": a.rank,
        "lora_alpha": 2 * a.rank,
        "output_dir": output_dir,
        **a.config,
    }
    if a.block_size:
        raw["block_size"] = a.block_size
    raw.pop("trust_remote_code", None)  # only a config file's --trust-remote-code grants it
    return raw


def run(a: "TrainAction", ctx: "Context") -> Union[TrainOutput, JobOutput]:
    from poros.core.config import PorosConfig

    run_id = runs.new_id()
    out_dir = a.output or str((run_files(a.disk) or runs.root()) / run_id)
    try:
        cfg = PorosConfig(**config_dict(a, out_dir))
    except Exception as exc:  # a bad key=value: the config error screen
        from poros.cli.errors import _friendly_error

        _friendly_error(exc)
    _refuse_local_push(cfg)
    if a.detach:
        job = _submit(a, cfg)
        job.plan = plan_said(a) if under_agent() else None
        job.from_ = a.from_run
        if job.state == "queued":
            from poros import jobs

            job.queued_behind = jobs.waits_behind(job, jobs.all_jobs())
        return job
    _require_folders(out_dir, own=not a.output)
    if sys.platform == "darwin":  # before a folder records a run that never began; after
        from poros.cli.train_cmd import hold_mac  # POROS_HOME's check: the lock is kept in it

        hold_mac().close()
    # --overwrite: the old run's files would mix in. They are cleared last, once every check
    # before the load has passed (``began``): a refusal before it leaves that run whole.
    replaces = bool(a.output and a.overwrite)
    resolved = Path(out_dir) / "config.resolved.json"
    was = resolved.read_bytes() if replaces and resolved.is_file() else None

    while True:  # an id is the minute and 16 random bits: one taken since is a new id
        try:
            run_dir = runs.create(
                run_id, {**a.model_dump(mode="json"), "command": a.command(), **_loop_record(a)},
                output_dir=out_dir)
            break
        except FileExistsError:
            run_id = runs.new_id()
            out_dir = a.output or str((run_files(a.disk) or runs.root()) / run_id)
            cfg = PorosConfig(**config_dict(a, out_dir))
        except OSError as exc:  # what the check above could not see: a full disk
            from poros.preflight.disk import output_error

            raise command.unwritable_output(output_error(exc, runs.root()), own=True) from None
    began: list = []

    def begin() -> None:
        if replaces:
            _clear_previous(Path(out_dir))
        began.append(True)

    try:
        with _lock(run_dir):
            metrics = _train(a, cfg, run_id, run_dir, resume_from=None, began=begin,
                             replaces=replaces)
    except BaseException:
        if not began:  # refused (or stopped) before the load: no run to list, or to be `last`
            shutil.rmtree(run_dir, ignore_errors=True)  # made new above (exist_ok=False)
            if replaces:  # the settings this run's checks wrote over the old run's: put back
                resolved.write_bytes(was) if was is not None else resolved.unlink(missing_ok=True)
        raise
    result = TrainOutput(
        output_dir=out_dir,
        events=str(Path(out_dir) / "events.jsonl"),
        metrics=metrics or {},
        run=run_id,
        command=a.command(),
        **said_support(a.unvalidated),
    )
    result.from_ = a.from_run
    if under_agent():
        from poros.cli.verbs import runs_report

        finished = runs.read(run_dir)
        result.learned = runs_report.row(finished).get("learned")
        result.next = runs_report.offers(finished)
        result.plan = plan_said(a)
    return result


def _require_folders(out_dir: str, *, own: bool) -> None:
    """The run's folder under POROS_HOME, and the output folder where one was named: both can be
    made and written, or the run is refused before a folder records it."""
    from poros.preflight.disk import output_fault

    for folder, its_own in ((runs.root(), True), (out_dir, own)):
        refusal = command.unwritable_output(output_fault(folder), own=its_own)
        if refusal is not None:
            raise refusal


def keeps(save: int, steps: int, limit: int = 0) -> str:
    """Which checkpoints a run of ``steps`` keeps, saving every ``save`` steps and keeping the
    newest ``limit`` on disk (none pruned at 0): a short run on a Mac, whose table row saves every
    1000, keeps only its end and a stop's."""
    sets = "save_steps=N sets it"
    if save <= 0:
        return f"only at its end (and at a stop): save_steps=0 saves none between; {sets}"
    if steps and steps <= save:
        return (f"only at its end (and at a stop): it saves every {save:,} steps and runs "
                f"{steps:,}; {sets}")
    saves = (steps - 1) // save + 1 if steps else None  # its saves on the way, and its end's
    newest = (f", the newest {limit} kept (save_total_limit=N)"
              if limit > 0 and (saves is None or saves > limit) else "")
    return f"every {save:,} steps, at a stop and at its end{newest}; {sets}"


def plan_said(a: "TrainAction") -> dict:
    """What a run trains at, from what train resolved. An eta only from a certified run of these
    settings on this machine, and it says which; with none, ``eta_from`` says why.
    ``save_steps`` is what the run saves at: yours, else the recipe's or the Mac table row's, else
    the config's; ``keeps`` says in words which checkpoints that leaves this run."""
    from poros import data as poros_data
    from poros.core.config import PorosConfig

    def setting(name: str):
        return a.config.get(name, a.preset.get(name, PorosConfig.model_fields[name].default))

    split = setting("eval_split")
    if a.config.get("eval_dataset"):
        held = train_data.scan(a.config["eval_dataset"], Path.cwd()).rows
    elif a.shape is None or a.rows is None:
        held = 0 if a.rows is not None else None
    else:
        held = poros_data._held(a.rows, float(split), int(setting("eval_rows")))
    save = int(setting("save_steps"))
    said = {"steps": a.steps, "seq_len": a.seq_len, "rows": a.rows, "eval_rows": held,
            "learning_rate": a.lr, "recipe": a.recipe, "save_steps": save,
            "keeps": keeps(save, a.steps, int(setting("save_total_limit")))}
    if a.shape == "chats" and (base := base_on_chats(a.model)):
        said["warning"] = base
    cert = getattr(a, "certified", None)  # a built-in certified here at these settings, same loop
    from poros.cli.sheets.fit import on_a_drive

    if cert is not None and on_a_drive(a):  # its memory is certified; its time is its drive's
        from poros.cli.words.models import by_id

        e = by_id(a.model)  # a dense model's layers come from the drive, a MoE model's experts
        what = "experts" if e is not None and e.entry == "moe" else "weights"
        said["eta_from"] = (f"no estimate: the certified run of {cert[0]} read its {what} from a "
                            "drive its record does not name, and a drive sets the time an "
                            "update; run watch gives one from this run's own steps")
        return said
    if cert is not None:
        key, e = cert
        said["eta_s"] = round(a.steps * e.s_per_step)
        said["eta_from"] = (f"{a.steps} steps at {e.s_per_step:.2f} s a step, the certified run "
                            f"of {key} on this machine; loading comes on top")
        return said
    said["eta_from"] = ("no estimate: no certified run of these settings on this machine; run "
                        "watch gives one from this run's own steps")
    return said


def _clear_previous(out: Path) -> None:
    """What a previous run left in ``out``: its checkpoints, adapter and events. Called once
    this run's checks have passed; config.resolved.json is by then this run's own."""
    for d in out.glob("checkpoint-*"):
        shutil.rmtree(d, ignore_errors=True)
    for name in (
        *(n for n in _PREVIOUS if n != "config.resolved.json"),
        "adapter_model.safetensors",
        "adapter_weights.pt",
        "adapter_config.json",
    ):
        (out / name).unlink(missing_ok=True)


@contextmanager
def _lock(run_dir: Path) -> Iterator[None]:
    """Hold the run: a second `runs continue` of it refuses instead of training
    into the same folder. POSIX flock; elsewhere the status check stands alone."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(run_dir / "run.lock", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refusal(
                "E_USAGE", "Already running", f"another process is training {run_dir.name}"
            ) from None
        yield


def _next(run_id: str) -> str:
    """What a run's screen offers at its end: scoring it beside its base, on CUDA or a Mac, as it
    streams the same weights the run did."""
    return f"poros eval {run_id}"


def _train(
    a: "TrainAction", cfg, run_id: str, run_dir: Path, resume_from: Optional[str],
    began: Optional[Callable[[], None]] = None, replaces: bool = False,
) -> Optional[dict]:
    """The run trained, its status kept true. Its caller holds the run's lock. ``began``: called
    once the checks before the load have passed. ``replaces``: ``began`` clears a run the
    output folder holds (train_cmd.run's)."""
    from poros.cli import errors as cli_errors
    from poros.cli import train_cmd
    from poros.cli.commands import train as cli_commands_train

    runs.update(
        run_dir,
        state="running",
        pid=os.getpid(),
        **runs.running_here(),  # which machine, and which process of that pid
        started_at=runs.now(),
        finished_at=None,
        error=None,
    )
    (run_dir / "stop").unlink(missing_ok=True)
    failed: dict = {}  # why it failed, for the run's page: the code, and a sentence

    def friendly_error(exc: Exception) -> None:
        from poros.core.errors import PorosError, library_error

        failed.update(code=cli_errors._error_code(exc),
                      message=exc.info.message if isinstance(exc, PorosError)
                      else library_error(exc))  # a library's own text stays out of the record
        cli_errors._friendly_error(exc)

    try:
        metrics = train_cmd.run(
            cfg,
            config_path=a.command(),
            resume_from=resume_from,
            trust_remote_code=False,
            friendly_error=friendly_error,
            plan_gate=lambda c, mt: cli_commands_train._plan_gate(c, mt, a.force),
            should_stop=(run_dir / "stop").exists,
            checked=(a.name, f"run {run_id}"),
            shown_run=run_id,
            resume_cmd=f"poros runs continue {run_id}",
            next_cmd=_next(run_id),
            next_unscored=(f"poros export {run_id}",
                           f"No rows were held out: `poros eval {run_id} --data FILE` scores "
                           "a file of yours."),
            emit_json=False,
            unvalidated=bool(a.unvalidated),
            force=bool(a.force),
            began=began,
            replaces=replaces,
        )
    except typer.Exit as e:
        code = e.exit_code or 0
        state = "succeeded" if code == 0 else "cancelled" if code in _CANCELLED else "failed"
        runs.update(
            run_dir,
            state=state,
            finished_at=runs.now(),
            error={"exit_code": code, **failed} if state == "failed" else None,
        )
        raise
    except KeyboardInterrupt:  # ctrl-c before the live view took it (preflight, load)
        runs.update(run_dir, state="cancelled", finished_at=runs.now(), error=None)
        raise
    except BaseException as exc:
        runs.update(
            run_dir,
            state="failed",
            finished_at=runs.now(),
            error={"exit_code": 1, "message": type(exc).__name__},
        )
        raise
    runs.update(run_dir, state="succeeded", finished_at=runs.now())
    return metrics


# What a run's folder does not say for whoever continues it, as for poros.Run.open (its
# _NOT_THE_FOLDERS): a folder can come from anyone. Its base's own code, a loop file and a push
# run on the caller's word alone; it writes where its checkpoints are; and it resumes its own
# checkpoint, never an adapter a path in it names.
NOT_THE_FOLDERS = ("trust_remote_code", "push_to_hub", "hub_model_id", "init_adapter", "loop",
                   "output_dir")


def _loop_record(a: "TrainAction") -> dict:
    """What action.json keeps of a loop file: its bytes' sha256, which a continue checks."""
    from poros import jobs

    loop = a.config.get("loop")
    return {"loop_sha256": jobs.loop_digest(loop)} if loop else {}


def _refuse_local_push(cfg) -> None:
    """push_to_hub on a base that is a folder here: refused now, before the load, by the rule the
    push itself holds (no downloader resolves the folder's path, and it would leave with it)."""
    from poros.cli.verbs.export import local_base

    if cfg.push_to_hub and local_base(str(cfg.model_name_or_path)):
        raise Refusal("E_CONFIG", "A local base",
                      f"push_to_hub with a base that is a folder here ({cfg.model_name_or_path})",
                      context=["no one the adapter is pushed to can load that folder",
                               "train on the base's Hub id to push"])


def _own_output(run: runs.Run, saved: dict) -> None:
    """Where a continue writes: the run's folder, the output folder its command named, or a
    folder of its id (a Mac's disk); never a link. Anything else its status names is refused."""
    out = Path(run.output_dir)
    named = saved.get("output")
    allowed = {os.path.normpath(run.folder)}
    if isinstance(named, str) and named:
        allowed.add(os.path.normpath(os.path.abspath(os.path.expanduser(named))))
    if (out.is_symlink() or (os.path.normpath(str(out)) not in allowed and out.name != run.id)):
        raise Refusal("E_USAGE", "Not this run's folder",
                      f"run {run.id}'s status names {out} to write in",
                      context=["a continue writes in the run's own folder, or the output "
                               "folder its command named"])


def continue_run(run: runs.Run, more: Optional[int] = None,
                 data: Optional[str] = None, loop: Optional[str] = None,
                 push_to: Optional[str] = None) -> TrainOutput:
    """Resume a stopped run from its newest complete checkpoint, bitwise, with
    the action it started with. With no checkpoint there is nothing to resume:
    it refuses rather than quietly start over.

    ``data``: where the run's data file is now. A moved file is no changed setting: the resume
    check compares its contents with what the checkpoint recorded, and refuses other data.

    ``more`` trains a run further: ``more`` steps past its total, from the
    checkpoint at its last step, the schedule stretched to the new total
    (so this part is not bitwise to anything; the steps before it are).

    ``loop``: the run's loop file, named again by its caller (NOT_THE_FOLDERS); ``push_to``: the
    Hub repo to push to when it ends, said on this command, never by the folder."""
    from poros.cli.verbs.train import TrainAction
    from poros.core import checkpoints
    from poros.core.config import PorosConfig

    run_dir = Path(run.folder)
    with _lock(run_dir):  # before anything is read or written: a delete or a second continue waits
        saved = json.loads((run_dir / "action.json").read_text())
        a = TrainAction(**{k: v for k, v in saved.items()
                           if k in TrainAction.model_fields and k != "provenance"})
        folder_said = {k: a.config.pop(k) for k in NOT_THE_FOLDERS if k in a.config}
        _own_output(run, saved)
        notes = _granted(run, a, folder_said, saved, loop, push_to)
        checkpoints.recover_checkpoint_dirs(Path(run.output_dir))
        done = checkpoints.list_checkpoints(Path(run.output_dir))
        if not done:
            from poros.cli.verbs import runs_report

            raise Refusal(
                "E_NOT_FOUND",
                "No checkpoint",
                f"run {run.id} stopped before its first checkpoint",
                context=["nothing to continue from"],
                fix=runs_report.restart_command(run.folder) or "", note="starts it again",
            )
        if data:
            a.data = saved["data"] = data
        else:
            require_data_there(run.id, a.data)
        if more:
            require_last_checkpoint(run, a.steps)
            a.steps += more
            saved["steps"] = a.steps
        if (more or data) and "command" in saved:
            saved["command"] = a.command()
        from poros.stores import recorded_store

        fields, build_path = recorded_store(config_dict(a, run.output_dir))  # read, never written
        cfg = PorosConfig(**fields)
        cfg._store_build_path = build_path
        # what the run said before this attempt: a refusal before the load leaves it as it was
        # (its state, its step count), and the new total is written only once the load begins
        before = {name: (run_dir / name).read_bytes() for name in ("status.json", "action.json")}
        began: list = []

        def begin() -> None:
            began.append(True)
            if more or data:  # past every check: the run's record says its new total, or where
                runs.write_json(run_dir / "action.json", saved)  # its data is now

        for line in notes:  # what the folder said that this continue does not do
            print(f"Note: {line}", file=sys.stderr)
        try:
            metrics = _train(a, cfg, run.id, run_dir, resume_from=str(done[-1][1]), began=begin)
        except BaseException:
            if not began:
                for name, raw in before.items():
                    runs.write_bytes_atomic(run_dir / name, raw)
            raise
    return TrainOutput(
        output_dir=run.output_dir,
        events=str(Path(run.output_dir) / "events.jsonl"),
        metrics=metrics or {},
        run=run.id,
        command=a.command(),
        **said_support(a.unvalidated),
    )


def _granted(run: runs.Run, a: "TrainAction", folder_said: dict, saved: dict,
             loop: Optional[str], push_to: Optional[str]) -> list[str]:
    """The settings of NOT_THE_FOLDERS this continue takes from its caller, set on ``a``; and
    what it says about the rest. A loop run continues only with its loop named again."""
    from poros import jobs

    said: list[str] = []
    if folder_said.get("loop"):
        said += jobs._loop_given(f"run {run.id}", str(folder_said["loop"]),
                                 saved.get("loop_sha256"), loop)
        a.config["loop"] = loop
    elif loop:
        raise Refusal("E_USAGE", "No loop", f"run {run.id} trained with no loop file",
                      context=["a continue keeps the run's recipe"])
    if folder_said.get("push_to_hub") and not push_to:
        said.append(f"run {run.id} pushed to the Hub when it ended: this continue does not push; "
                    f"poros runs continue {run.id} push_to_hub=true hub_model_id=<repo> pushes")
    if push_to:
        a.config.update(push_to_hub=True, hub_model_id=push_to)
    return said


def require_data_there(run_id: str, data: str) -> None:
    """A run's data file, where the run recorded it: gone, the next step is to say where it is
    now (`continue --data`), in this command's words and not a config file's."""
    path = Path(str(data))
    if not path.is_absolute() or path.exists():  # a Hub id or synthetic names no file here
        return
    raise Refusal("E_DATASET_NOT_FOUND", "Data moved",
                  f"run {run_id} trained on {data}, and nothing is there now",
                  context=["if the file moved, say where it is: its contents are checked "
                           "against what the run trained on"],
                  fix=f"poros runs continue {run_id} --data <file>",
                  note="continues it on the same data at another path")


def require_job_data(run: runs.Run, data: Optional[str]) -> None:
    """A job's continue: its data is there (or ``data`` says where it is now), and ``data``
    holds what its newest checkpoint trained on. Checked here, before it is queued again: in
    the queue a refusal is only read later."""
    import yaml

    from poros.core import checkpoints
    from poros.core.config import PorosConfig

    try:
        said = yaml.safe_load((Path(run.folder) / "config.yaml").read_text()) or {}
    except (OSError, ValueError):
        return
    if not data:
        require_data_there(run.id, str(said.get("dataset") or ""))
        return
    done = checkpoints.list_checkpoints(Path(run.output_dir))
    if not done:
        return
    try:
        cfg = PorosConfig(**{**said, "dataset": data})
    except ValueError:
        return  # its own run says what its config lacks
    clash = command.resume_refusal(str(done[-1][1]), cfg)
    if clash is not None:
        raise clash


def said_support(model_type: Optional[str]) -> dict:
    """What a result about a run trained with --unvalidated says: no parity guarantee applies."""
    return {"support": "user_asserted", "model_type": model_type} if model_type else {}


def require_last_checkpoint(run: runs.Run, steps: int) -> None:
    """--more trains on from the checkpoint at the last step; the classic loop may not keep one."""
    from poros.core import checkpoints

    done = checkpoints.list_checkpoints(Path(run.output_dir))
    if not done or done[-1][0] < steps - 1:
        raise Refusal(
            "E_USAGE",
            "No final checkpoint",
            f"run {run.id} kept no checkpoint at its last step (step {steps:,})",
            context=["runs trained by the SFT and DPO recipes keep one",
                     "the classic loop saves every save_steps only"],
        )


def continue_detached(run: runs.Run, more: Optional[int] = None,
                      data: Optional[str] = None, loop: Optional[str] = None,
                      push_to: Optional[str] = None) -> None:
    """`runs continue` in a session of its own, for a caller that cannot wait: returns once the
    run holds its folder. An attempt that ends first refuses with what it said."""
    argv = [sys.executable, "-P", "-m", "poros.cli.main", "runs", "continue", run.id, "--yes",
            "--json", *(["--more", str(more)] if more else []),
            *(["--data", data] if data else []), *(["--recipe", loop] if loop else []),
            *(["push_to_hub=true", f"hub_model_id={push_to}"] if push_to else [])]
    log = Path(run.folder) / "continue.log"
    with runs.open_log(log, "w") as f:  # never through a planted link
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
    for _ in range(600):
        if proc.poll() is not None or runs.read(Path(run.folder)).state == "running":
            break
        time.sleep(0.05)
    if proc.poll():
        said = log.read_text().strip().split("\n")[-1:]  # a name can hold U+2028
        err = json.loads(said[0]).get("error") if said[0].startswith("{") else None
        raise Refusal((err or {}).get("code", "E_INTERNAL"), "Did not continue",
                      (err or {}).get("message", f"it exited {proc.returncode}: see {log}"),
                      fix=(err or {}).get("hint", ""))


def _submit(a: "TrainAction", cfg) -> JobOutput:
    """--detach: the same preflights, then a #40 job, queued per GPU; the job
    runs the fit check when its turn comes. `poros runs` lists it."""
    from poros.cli import errors as cli_errors
    from poros.cli import train_cmd
    from poros.cli.commands import jobs as cli_commands_jobs

    jobs = cli_commands_jobs._jobs()
    queued: list[JobOutput] = []

    def submit() -> None:
        raw = config_dict(a, "./poros_output")  # the job points output_dir at its own folder
        queued.append(jobs.submit(raw, model=a.model, gpu=jobs.gpu_uuid(), force=a.force,
                                  files=run_files(a.disk), name=a.name,
                                  unvalidated=a.unvalidated))

    train_cmd.run(
        cfg,
        config_path=a.command(),
        resume_from=None,
        trust_remote_code=False,
        friendly_error=cli_errors._friendly_error,
        submit=submit,
        emit_json=False,
        unvalidated=bool(a.unvalidated),
    )
    if contract.json_mode():  # stdout is the one JSON object: the next step goes to stderr
        watch = contract.as_agent(f"poros runs {queued[0].job_id} --live --wait 30")
        from poros.cli.verbs.runs_live import WAIT_MAX_S

        print(f"Watch it: {watch} (returns after 30 more steps, or {WAIT_MAX_S:.0f} s; show its "
              "card rows as they are)", file=sys.stderr)
    return queued[0]
