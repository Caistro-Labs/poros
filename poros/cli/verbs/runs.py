"""runs: your runs. Watch one, stop it, continue it, report it, see what it saved.

    poros runs
    poros runs last
    poros runs stop 7f3a
    poros runs continue last --more 200
    poros runs report last
    poros runs checkpoints last

Flags:
    --follow          watch a running run live
    --live            the run's live state, read once (for agents, with --json)
    --frame           with --live: the run with its loss curve, as text
    --wait N          with --live: return once the run is N steps further or ends
    --more N          continue: train N steps past the run's total
    --data FILE       continue: where the run's data file is now, if it moved; its
                      contents must be the ones the run trained on
    --detach          continue: in the background, for a caller that cannot wait
    --recipe FILE     continue: the loop file a run trained with, named again (a run's
                      folder never runs a Python file on its own word)
    --trust-remote-code  continue a job whose base runs its own code, on your word
    --limit N         the newest N runs
    --copy STEP       checkpoints: copy one here, a step, final, or best (the lowest
                      held-out loss)
    --delete STEP     checkpoints: delete one, once asked
    --yes             no confirmation

The list holds the runs started with poros train <model> --data, and jobs. A
config file's run and a Python run keep their record in their own output_dir.
A run is its id, a unique prefix of it, or last (the newest). Stopping is a
request: the run checkpoints at its next step and exits. Continuing resumes
from that checkpoint, bitwise, with the action the run started with;
--more trains a finished run further. checkpoints lists what a run saved:
final, its finished adapter, then each step, newest first, each with its
held-out loss when the run scored one and the lowest marked best. A run
continues from its newest step only; train --from <run>/weights/best starts
a new one from the best.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import typer
from pydantic import Field

from poros import runs
from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import CheckpointsOutput, Choice, Output, Question
from poros.cli.ui.text import Line, S
from poros.cli.words import runs as run_words
from poros.cli.words.grammar import Sentence
from poros.core.env import under_agent
from poros.core.refusal import Refusal

Do = Literal["list", "show", "stop", "continue", "report", "checkpoints"]


class RunsAction(Action):
    verb: ClassVar[str] = "runs"
    do: Do = "list"
    run: Optional[str] = None
    follow: bool = False
    more: Optional[int] = None  # continue: train this many steps past the run's total
    data: Optional[str] = None  # continue: where the run's data file is now
    detach: bool = False  # continue: in the background
    # continue: what only its caller says, never the run's folder (train_run.NOT_THE_FOLDERS)
    loop: Optional[str] = None  # --recipe FILE: the run's loop file, named again
    push_to: Optional[str] = None  # push_to_hub=true hub_model_id=REPO: push when it ends
    trust_remote_code: bool = False  # --trust-remote-code: a job's base runs its own code
    live: bool = False  # the live state, once: poros.run_live.v1
    frame: bool = False
    wait: Optional[int] = None
    limit: Optional[int] = None  # list: the newest this many
    copy_step: Optional[str] = None  # checkpoints: this one, a step or final, copied here
    delete_step: Optional[str] = None  # checkpoints: this one deleted, once asked
    yes: bool = Field(False, exclude=True)  # --yes: what it asks, agreed to; never in its command

    def canonical(self) -> list[str]:
        out = ["runs"] + ([] if self.do in ("list", "show") else [self.do])
        out += [self.run] if self.run else []
        out += ["--limit", str(self.limit)] if self.limit else []
        out += ["--copy", self.copy_step] if self.copy_step else []
        out += ["--delete", self.delete_step] if self.delete_step else []
        out += ["--more", str(self.more)] if self.more else []
        out += ["--data", self.data] if self.data else []
        out += ["--detach"] if self.detach else []
        out += ["--recipe", self.loop] if self.loop else []
        out += ["push_to_hub=true", f"hub_model_id={self.push_to}"] if self.push_to else []
        out += ["--trust-remote-code"] if self.trust_remote_code else []
        out += ["--follow"] if self.follow else []
        out += ["--live"] if self.live and not (self.frame or self.wait) else []
        out += ["--frame"] if self.frame else []
        return out + (["--wait", str(self.wait)] if self.wait else [])


class RunsOutput(Output):
    schema_: Literal["poros.runs.v1"] = Field("poros.runs.v1", alias="schema")
    learned: Optional[str] = None  # `poros agents`, one run over: its held-out loss in words
    do: Do
    runs: list[dict[str, Any]]  # every run (list), or the one acted on
    report: Optional[str] = None  # `runs report`: the file written
    detail: Optional[dict] = Field(None, exclude=True)  # the run page: what the run recorded
    next: list[str] = []  # the commands a finished run offers
    more: int = 0  # --limit: how many are not shown
    starts_next: Optional[str] = None  # stop: the run queued behind it, which starts as it ends


def _pick(do: Do, all_runs: list[runs.Run]) -> Question:
    wanted = {"stop": ("running", "queued"), "continue": ("cancelled", "failed")}.get(do)
    cands = [r for r in all_runs if wanted is None or r.state in wanted][:9]
    if not cands:
        what = {"stop": "running", "continue": "stopped or failed"}.get(do, "")
        raise Refusal(
            "E_NOT_FOUND",
            "No runs",
            "no runs yet" if do == "checkpoints" else f"no {what} run to {do}".replace("  ", " "),
            fix="poros runs",
            note="lists them",
        )
    return Question(
        id="run",
        prompt=f"Which run to {do}?" if do not in ("show", "checkpoints") else "Which run?",
        flag="subject",
        default=cands[0].id,
        choices=[Choice(value=r.id, label=r.name, note=f"{r.state} · {r.model}") for r in cands],
    )


def resolve(s: Sentence, ctx: Context) -> Draft[RunsAction]:
    do: Do = s.action or ("show" if s.subject else "list")  # type: ignore[assignment]
    follow = bool(s.flags.get("--follow") or s.flags.get("-f"))
    frame = bool(s.flags.get("--frame"))
    wait = _count(s, "--wait")
    live = bool(s.flags.get("--live")) or frame or wait is not None
    if live and do != "show":
        raise Refusal("E_USAGE", "One run", "--live reads one run",
                      fix="poros runs last --live", note="the newest run")
    if do == "list":
        return Draft(RunsAction(do="list", limit=_count(s, "--limit")))
    more = _count(s, "--more") if do == "continue" else None
    if do == "continue":  # the run's own action, or the bitwise guarantee breaks
        # --data is no change of setting: it says where the same data is now, and the resume
        # check compares its contents with what the run trained on
        # nor are the grants only its caller gives: a loop file, a push, a base's own code
        changed = [c.said for name, c in s.clauses.items()
                   if name not in ("data", "push_to_hub", "hub_model_id")] + [
            f for f in s.flags
            if f not in ("--json", "--follow", "-f", "--more", "--yes", "-y", "--detach",
                         "--data", "--recipe", "--trust-remote-code")
        ]
        if changed:
            raise Refusal(
                "E_USAGE",
                "Continue changes nothing",
                f"a continued run keeps its settings: {', '.join(changed)}",
                context=[
                    "changing them would break the bitwise guarantee",
                    "for other data or settings, start a new run from its adapter: "
                    "poros train --from <run> --data <file>",
                ],
            )
    all_runs = runs.all_runs()
    ids = [r.id for r in all_runs]
    word = s.subject[0] if s.subject else None
    if word is None:
        if do == "continue":  # `continue` alone: the newest stopped run
            stopped = [r for r in all_runs if r.state in ("cancelled", "failed")]
            if len(stopped) == 1:
                return Draft(RunsAction(do=do, run=stopped[0].id))
        return Draft(None, [_pick(do, all_runs)])
    found = run_words.resolve(word, ids, newest=ids[0] if ids else None,
                              names=[(r.name, r.id) for r in all_runs])
    if isinstance(found, run_words.RunMiss):
        if found.candidates:
            pool = [r for r in all_runs if r.id in found.candidates]
            return Draft(None, [_pick(do, pool)])
        raise Refusal(
            "E_NOT_FOUND", "No such run", found.reason, fix="poros runs", note="lists them"
        )
    if s.flags.get("--more") is True:  # `--more` with no number: how many
        r = next(x for x in all_runs if x.id == found)
        return Draft(None, [_more_question(r)])
    copy = delete = None
    if do == "checkpoints":  # --copy and --delete each name one of its checkpoints
        copy, delete = _saved(s, "--copy", found), _saved(s, "--delete", found)
        if copy and delete:
            raise Refusal("E_USAGE", "One at a time", "--copy or --delete, not both")
    grants = _grants(s, ctx) if do == "continue" else {}
    return Draft(RunsAction(do=do, run=found, follow=follow, more=more,
                            data=_moved_data(s, ctx) if do == "continue" else None, **grants,
                            detach=bool(s.flags.get("--detach")),
                            live=live, frame=frame, wait=wait, copy_step=copy, delete_step=delete,
                            yes=bool(s.flags.get("--yes") or s.flags.get("-y"))))


def _grants(s: Sentence, ctx: Context) -> dict:
    """`continue`'s own grants: --recipe FILE (a .py, absolute), push_to_hub=true with
    hub_model_id=REPO, --trust-remote-code."""
    out: dict = {"trust_remote_code": bool(s.flags.get("--trust-remote-code"))}
    loop = s.flags.get("--recipe")
    if loop is not None:
        path = Path(str(loop)).expanduser() if loop is not True else None
        if path is None or path.suffix != ".py":
            raise Refusal("E_USAGE", "A loop file", "continue's --recipe names the run's loop .py")
        path = path if path.is_absolute() else ctx.cwd / path
        if not path.is_file():
            raise Refusal("E_USAGE", "No such loop", f"{loop} does not exist")
        out["loop"] = str(path.resolve())
    push = str(s.value("push_to_hub") or "").lower()
    repo = s.value("hub_model_id")
    if push in ("true", "1", "yes") and repo and repo is not True:
        out["push_to"] = str(repo)
    elif push or repo:
        raise Refusal("E_USAGE", "Push where", "a continue pushes with push_to_hub=true and "
                      "hub_model_id=<repo>, both")
    return out


def _moved_data(s: Sentence, ctx: Context) -> Optional[str]:
    """`continue --data FILE`: where the run's data file is now, absolute. A file that is there;
    whether it holds what the run trained on is the resume check's to say."""
    said = s.value("data")
    if said is None:
        return None
    path = Path(str(said)).expanduser() if said is not True else None
    path = path if path is None or path.is_absolute() else ctx.cwd / path
    if path is None or not path.exists():
        raise Refusal("E_DATASET_NOT_FOUND", "No such data",
                      "--data needs the file the run trained on, where it is now" if path is None
                      else f"no file or folder at {said}")
    return str(path.resolve())


def _count(s: Sentence, flag: str) -> Optional[int]:
    raw = s.flags.get(flag)
    if raw is None or raw is True:
        return None
    try:
        n = int(str(raw).replace(",", "").replace("_", ""))
    except ValueError:
        raise Refusal("E_USAGE", "Not a number", f"{flag} {raw!r} is not a number")
    if n < 1:
        raise Refusal("E_USAGE", "Out of range", f"{flag} must be >= 1")
    return n


def _saved(s: Sentence, flag: str, run_id: str) -> Optional[str]:
    """A checkpoint by its step, final, or best (as the step it is, so the command says which):
    what --copy and --delete take."""
    from poros.cli.verbs import checkpoints

    said = s.flags.get(flag)
    if said is None or said == "final":
        return said
    if said == checkpoints.BEST:
        return checkpoints.step_of(_find(run_id), said)
    if not re.fullmatch(r"[0-9]{1,18}", str(said), re.ASCII):
        raise Refusal("E_USAGE", "Which checkpoint", f"{flag} takes a step, final or best",
                      fix=f"poros runs checkpoints {run_id}", note="lists them")
    return str(int(said))


def _more_question(r: runs.Run) -> Question:
    total = r.total or 0
    picks = [(total, "as many again")] if total else []
    picks += [(n, "") for n in (100, 500, 1000) if n != total]
    return Question(
        id="more",
        prompt=f"Train {r.name} how many more steps?",
        flag="--more",
        default=str(picks[0][0]),
        choices=[Choice(value=str(n), label=f"{n:,} more", note=note) for n, note in picks],
    )


def confirmations(action: RunsAction, ctx: Context) -> list:
    """A stop is a checkpointed request, never a loss. More steps spend the machine: whoever typed
    them at a terminal said so, and anyone else is asked. A deleted checkpoint is gone for good, so
    everyone is asked, and a copy asks before it replaces one."""
    from poros.cli.questions import confirmation

    if action.more and not action.yes and not ctx.interactive:
        return [confirmation("more", f"Train run {action.run} {action.more:,} steps more?",
                             "--yes", "Train it further")]
    if action.yes or action.do != "checkpoints":
        return []
    from poros.cli.verbs import checkpoints

    if action.delete_step:  # a running run's is refused here, before anyone is asked
        row = checkpoints.select([_weights(action.run, action.delete_step)])[0]
        where = _find(action.run).output_dir  # the folder it deletes in, as the run says it
        return [confirmation("delete", f"Delete for good: {row.path}, in {where}.", "--yes",
                             "Delete it")]
    there = action.copy_step and checkpoints.copy_of(_weights(action.run, action.copy_step), ".")
    if there and os.path.lexists(there):
        return [confirmation("replace", f"{there} is already here; the copy replaces it.",
                             "--yes", "Replace it")]
    return []


def _weights(run_id: str, step: str) -> str:
    return f"poros://{run_id}/weights/{step}"


def _find(run_id: str) -> runs.Run:
    hit = next((r for r in runs.all_runs() if r.id == run_id), None)
    if hit is None:
        raise Refusal("E_NOT_FOUND", "No such run", f"no run {run_id}")
    return hit


def run(action: RunsAction, ctx: Context) -> Output:
    if action.do == "list":
        every = runs.all_runs()
        shown = every[:action.limit] if action.limit else every
        return RunsOutput(do="list", runs=[r.to_dict() for r in shown],
                          more=len(every) - len(shown))
    r = _find(action.run)
    if action.do == "checkpoints":
        return _checkpoints(r, action)
    if action.live:
        from poros.cli.verbs import runs_live

        card = ctx.quiet_gpu.total_gib if ctx.quiet_gpu else None
        return runs_live.snapshot(r, frame=action.frame, wait=action.wait, card_gib=card)
    after = None
    if action.do == "stop":
        if r.state not in ("running", "queued", "stopping"):
            raise Refusal("E_USAGE", "Not running", f"run {r.id} is {r.state}")
        after = _queued_after(r)
        runs.stop(r)
        r = _stopped(r) if under_agent() else _find(r.id)
    elif action.do == "continue":
        return _continue(r, action.more, action.detach, action.data, loop=action.loop,
                         push_to=action.push_to, trust_remote_code=action.trust_remote_code)
    elif action.do == "report":
        from poros.cli.verbs import runs_report

        path = runs_report.write(r)
        return RunsOutput(do="report", runs=[r.to_dict()], report=path,
                          detail=runs_report.facts(r.folder, r.output_dir))
    elif r.state == "running" and (action.follow or ctx.interactive):
        from poros.cli.verbs import runs_follow

        if runs_follow.show_tab(r):  # trained in a tab of this session: that tab comes forward
            raise typer.Exit(code=0)
        runs_follow.follow(r)
        r = _find(r.id)
    from poros.cli.verbs import checkpoints, runs_report

    row = runs_report.row(r)
    held = checkpoints.storage(r).runs
    return RunsOutput(do=action.do, learned=row.pop("learned", None), runs=[row],
                      detail={**runs_report.facts(r.folder, r.output_dir),
                              "saved": held[0] if held else None},
                      next=runs_report.offers(r), starts_next=after)


def _queued_after(r: runs.Run) -> Optional[str]:
    """The job that starts on this one's GPU once a running job stops: the first one queued."""
    if r.kind != "job" or r.state not in ("running", "stopping"):
        return None
    from poros import jobs

    every = jobs.all_jobs()
    me = next((j for j in every if j.job_id == r.id), None)
    return jobs.first_queued(me.gpu, every) if me else None


def _stopped(r: runs.Run) -> runs.Run:
    """Under `poros agents`, a stop returns once the run has stopped at its checkpoint, so what it
    says is where it stopped: 100 s at most, as `run watch --wait` does."""
    import time

    from poros.cli.verbs.runs_live import WAIT_MAX_S

    until = time.monotonic() + WAIT_MAX_S
    while (r := _find(r.id)).state in ("running", "queued", "stopping") and (
            time.monotonic() < until):
        time.sleep(0.5)
    return r


def _checkpoints(r: runs.Run, action: RunsAction) -> CheckpointsOutput:
    """The run's checkpoints, as `poros agents checkpoint list` gives them; or one copied here, or
    deleted. Its confirmation came first."""
    from poros.cli.verbs import checkpoints

    if action.copy_step:
        return checkpoints.download(_weights(r.id, action.copy_step), ".", force=action.yes)
    if action.delete_step:
        rows = checkpoints.select([_weights(r.id, action.delete_step)])
        checkpoints.delete(rows)
        return CheckpointsOutput(do="delete", checkpoints=rows)
    return checkpoints.listing(r.id, 0)


def _continue(r: runs.Run, more: Optional[int] = None, detach: bool = False,
              data: Optional[str] = None, *, loop: Optional[str] = None,
              push_to: Optional[str] = None, trust_remote_code: bool = False) -> RunsOutput:
    ok = ("cancelled", "failed", "succeeded") if more else ("cancelled", "failed")
    if r.state not in ok:
        if r.state == "succeeded":
            raise Refusal("E_USAGE", "Already finished", f"run {r.id} is {r.state}",
                          fix=f"poros runs continue {r.id} --more 200",
                          note="trains it 200 steps further")
        raise Refusal("E_USAGE", "Nothing to continue", f"run {r.id} is {r.state}",
                      context=["only a stopped or failed run continues"])

    from poros.cli.verbs import train_run

    # a Mac run's files can sit on a disk of its own: gone is not "no checkpoint"
    if sys.platform == "darwin" and r.output_dir != r.folder and not Path(r.output_dir).exists():
        raise Refusal("E_DISK", "Not here", f"{r.output_dir} is not here",
                      context=[f"run {r.id} keeps its files there"],
                      fix="Attach its disk, then run this again")
    if r.kind == "job":
        from poros.cli.commands.jobs import _jobs

        if more and not r.total:
            raise Refusal("E_USAGE", "No total to extend", f"job {r.id} trains by epochs")
        if more:
            train_run.require_last_checkpoint(r, r.total)
        jobs = _jobs()
        train_run.require_job_data(r, data)  # there, and what the job trained on
        for line in jobs.settle_grants(Path(r.folder), loop=loop, push_to=push_to,
                                       trust_remote_code=trust_remote_code):
            print(f"Note: {line}", file=sys.stderr)
        queued = jobs.retry(Path(r.folder), jobs.read(Path(r.folder)), more=more or 0,
                            data=data)
        if queued is None:
            raise Refusal("E_USAGE", "Busy", f"job {r.id} is held by another resume")
    elif trust_remote_code:
        raise Refusal("E_USAGE", "Not a job's", f"run {r.id} never ran its base's own code",
                      context=["a run started with poros train <model> does not"])
    elif detach:
        train_run.continue_detached(r, more, data, loop=loop, push_to=push_to)
    else:
        train_run.continue_run(r, more=more, data=data, loop=loop, push_to=push_to)
    from poros.cli.verbs import runs_report

    now = _find(r.id)
    return RunsOutput(do="continue", runs=[runs_report.row(now)],  # queued: what it waits for
                      next=runs_report.offers(now) if under_agent()
                      else [f"poros runs {r.id} --live --wait 30"])


def view(result: Output, w: int) -> list[Line]:
    from poros.cli import views
    from poros.cli.verbs.runs_live import RunLiveOutput

    if isinstance(result, RunLiveOutput):  # the text an agent shows, the mark in the bright tier
        rows = result.frame or result.card
        return [[S(r[:2], "hi"), S(r[2:])] if r.startswith("▛▜") else [S(r, "tx")] for r in rows]
    if isinstance(result, CheckpointsOutput):  # the agent's table: one view of a checkpoint
        from poros.cli.verbs import checkpoints

        return checkpoints.view(result, w)
    return views.runs(result, w)
