"""eval: score a run beside the model it started from, on rows it never trained on.

    poros eval
    poros eval 7f3a --rows 20
    poros eval last --data more_chats.jsonl
    poros eval poros://7f3a/weights/120

Flags:
    --data FILE       score this file's rows, of the run's shape (default: the rows or pairs
                      the run held out, or its --eval file; a pairs run that held none out, its
                      own pairs)
    --rows N          the most rows it scores, from the first (default 100: a run holds out
                      at most 100)
    --checkpoint S    which weights: a step, best (the lowest held-out loss) or final
                      (default); the same as poros://<run>/weights/<S>
    --show N          also writes N of the rows' answers, greedy, from the base and then the
                      run: on chats and pairs, in the terminal only (never in --json)

With a config file:
    poros eval train.yaml         the config's dataset through its trained adapter (CUDA)
    --adapter DIR                 the adapter to score (default: the config's output_dir)
    --trust-remote-code           for a base model that ships its own code

Which model: a run's id, its name, or last (default: the newest finished run) scores its
finished adapter; poros://<run>/weights/<step> scores a checkpoint, and
poros://<run>/weights/best the one with the lowest held-out loss (poros runs checkpoints <run>
lists them).

The rows go through the streamed forward the run trained with, twice: with its adapter off,
which is the base exactly (as the run scored it before its first step, unless it started --from
another run), then with it on. Nothing loads at full precision, so it scores any model Poros
trains, at any size, on CUDA or a Mac, and no training byte changes. It says each loss, the
change and how many rows got better. On chats it reads each answer's end: whether the template's
end-of-turn token is the model's top choice there, as it must be for a reply to stop. On
preference pairs: how often the chosen answer scores over the rejected one. Only numbers come
back, never a row's words, except --show's answers, which a person reads and --json never carries.
It takes the GPU (on a Mac, the Mac) as a run does, so beside a run
it is refused as busy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar, Optional

from poros import runs
from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import EvalOutput, EvalSide, Question
from poros.cli.ui.text import Line
from poros.cli.verbs import shared
from poros.cli.words import runs as run_words
from poros.cli.words.grammar import Sentence
from poros.core.refusal import Refusal

ROWS = 100  # a run holds out at most 100 (poros.data.holdout)
SHOW = 10  # answers --show writes at most: each is a token at a time, twice


class EvalAction(Action):
    verb: ClassVar[str] = "eval"
    run: str
    step: Optional[str] = None  # a checkpoint's step; None: the finished adapter
    data: Optional[str] = None
    rows: int = ROWS
    show: int = 0  # --show N: N rows' answers, base and trained, for a person to read

    def canonical(self) -> list[str]:
        out = ["eval", f"poros://{self.run}/weights/{self.step}" if self.step else self.run]
        out += ["--data", self.data] if self.data else []
        out += ["--show", str(self.show)] if self.show else []
        return out + (["--rows", str(self.rows)] if self.rows != ROWS else [])


def resolve(s: Sentence, ctx: Context) -> Draft[EvalAction]:
    from poros.cli.verbs import checkpoints

    word, step = s.subject[0] if s.subject else "last", None
    if "/weights/" in word:
        word, step = checkpoints.parse(word)
        step = None if step == checkpoints.FINAL else step
    every = runs.all_runs()
    ids = [r.id for r in every]
    done = [r.id for r in every if r.state == "succeeded"]
    found = run_words.resolve(word, ids, newest=(done or ids or [None])[0],
                              names=[(r.name, r.id) for r in every])
    if isinstance(found, run_words.RunMiss) and not ids:
        raise Refusal("E_NOT_FOUND", "No runs yet", "train one, then score it",
                      fix="poros train", note="starts one")
    if isinstance(found, run_words.RunMiss):
        raise Refusal("E_NOT_FOUND", "No such run", found.reason,
                      context=[", ".join(found.candidates)] if found.candidates else [],
                      fix="poros runs", note="lists them")
    picked = s.value("checkpoint")
    if picked is not None:
        if step is not None or "/weights/" in str(s.subject[0] if s.subject else ""):
            raise Refusal("E_USAGE", "Which checkpoint",
                                 "give one: poros://<run>/weights/<step>, or --checkpoint")
        said = str(picked).lower()
        named = (checkpoints.BEST, checkpoints.FINAL)
        if picked is True or not (said.isdecimal() or said in named):
            raise Refusal("E_USAGE", "Which checkpoint",
                                 f"--checkpoint {'' if picked is True else picked}".strip(),
                                 context=["a step, best or final"])
        step = None if said == checkpoints.FINAL else said
    if step == checkpoints.BEST:  # the step it is, so the command says which
        step = checkpoints.step_of(_run(found), step)
    a = EvalAction(run=found, step=step, data=_data(s.value("data"), ctx),
                   rows=_rows(s.value("rows")), show=_show(s.value("show")))
    a.given("run", word)
    return Draft(a)


def _data(said, ctx: Context) -> Optional[str]:
    """--data FILE, absolute: a file of rows, never a folder, the Hub or a secret's name."""
    from poros import data as pdata
    from poros.cli.words.data import DATA_SUFFIXES

    if said is None:
        return None
    if said is True or not str(said):
        raise Refusal("E_USAGE", "Which file", "--data needs a file of rows")
    path = Path(str(said)).expanduser()
    path = path if path.is_absolute() else (ctx.cwd / path)
    if not path.is_file() or path.suffix.lower() not in DATA_SUFFIXES or pdata.secret_name(
            path.name):
        raise Refusal("E_NOT_FOUND", "No such file",
                      f"{said} is not a .jsonl, .json or .csv file here")
    return str(path.resolve())


def _rows(said) -> int:
    if said is None:
        return ROWS
    if said is True or not str(said).isdecimal() or int(said) < 1:
        raise Refusal("E_USAGE", "Not a number", f"--rows {said} is not 1 or more")
    return int(said)


def _show(said) -> int:
    if said is None:
        return 0
    if said is True or not str(said).isdecimal():
        raise Refusal("E_USAGE", "Not a number", f"--show {said} is not 0 or more")
    if int(said) > SHOW:
        raise Refusal("E_USAGE", "Too many", f"--show {said}: {SHOW} at most",
                      context=["each answer is written a token at a time, twice"])
    return int(said)


def _run(run_id: str) -> runs.Run:
    return next(x for x in runs.all_runs() if x.id == run_id)


def _adapter(r: runs.Run, step: Optional[str]) -> Path:
    """The weights it scores: a checkpoint's folder, or the finished adapter."""
    from poros.cli.ui import simulate
    from poros.cli.verbs import checkpoints
    from poros.core import checkpoints as saved

    out = Path(r.output_dir)
    if step is not None:
        found = checkpoints._folders(out).get(int(step))
        if found is None:
            raise Refusal("E_NOT_FOUND", "No such checkpoint",
                          f"run {r.id} keeps none after step {step}",
                          fix=f"poros runs checkpoints {r.id}", note="lists them")
        return found
    if r.state == "succeeded" and (saved.has_weights(out) or simulate.enabled()):
        return out  # a simulated run writes no adapter
    raise Refusal("E_USAGE", "Nothing to score yet", f"run {r.id} is {r.state}",
                  context=["eval a run that finished, or one of its checkpoints"],
                  fix=f"poros runs checkpoints {r.id}", note="lists them")


def confirmations(a: EvalAction, ctx: Context) -> list[Question]:
    r = _run(a.run)
    _adapter(r, a.step)
    if a.data:
        _same_shape(r, a.data)
    return []


def _same_shape(r: runs.Run, path: str) -> None:
    """--data is of the run's shape: chats for a chats run, pairs for a pairs run."""
    from poros import data as pdata
    from poros.cli.verbs import runs_report

    said = (runs_report._read(Path(r.output_dir) / "config.resolved.json") or {}).get("data_shape")
    try:
        shape = pdata.detect(Path(path))
    except ValueError as exc:
        raise Refusal("E_DATASET", "Can't read it", str(exc)) from None
    shape = "chats" if shape == "convertible" else shape
    if said in ("chats", "text", "pairs") and shape != said:
        raise Refusal("E_USAGE", "Not its shape",
                      f"{Path(path).name} holds {shape}, and run {r.id} trained on {said}",
                      context=["eval scores rows of the shape the run trained on"])


def run(a: EvalAction, ctx: Context) -> EvalOutput:
    from poros.cli.ui import simulate

    r = _run(a.run)
    adapter = _adapter(r, a.step)
    # Never queued, as a tool call is never left behind a run for hours: refused as busy. The
    # open file holds the GPU, or on a Mac the Mac, until this returns.
    held = shared.hold_gpu("eval")  # noqa: F841
    shared.refuse_planted(r.model)  # a folder here named as the base is not read as it
    if simulate.enabled():
        return _simulated(a, r, adapter)
    if r.unvalidated:  # the family its run asserted, for this command alone
        from poros.core.arch_registry import assert_this_command

        assert_this_command(r.unvalidated)
    out = score(r, adapter, ctx, data=a.data, cap=a.rows, show=a.show)
    out.checkpoint = _checkpoint(a)
    return _finished(out, r)


def score(r: runs.Run, adapter: Path, ctx: Context, *, data: Optional[str] = None,
          cap: int = ROWS, show: int = 0, dataset: Optional[str] = None,
          trust_remote_code: Optional[bool] = None) -> EvalOutput:
    """The run's rows through its own streamed forward, the adapter off and then on: what
    `poros eval <run>` says, and `poros eval <config>` of a recipe's run (``r`` is then the
    run's folder as a run). ``data``: a file to score whole. ``dataset``: where the run's data
    is now, where the config that names it says another path than the run recorded; with no
    rows held out, that file is scored whole and said so. ``trust_remote_code``: the calling
    command's own flag (`poros eval <config> --trust-remote-code`); a run's recorded settings
    never grant it."""
    from poros.cli import views
    from poros.cli.ui import term
    from poros.cli.verbs import evaluate_pass as ep
    from poros.ends import turn_end

    cfg, shape = ep.settings(r, adapter)
    if dataset:
        cfg.dataset = dataset
    if trust_remote_code is not None:
        cfg.trust_remote_code = trust_remote_code
    shared.refuse_planted(cfg.model_name_or_path)
    t, short = term.get(), cfg.model_name_or_path.rstrip("/").split("/")[-1]
    if ep.on_engine_loop(r):  # its rows as that loop reads and scores them
        return _score_on_loop(r, adapter, ctx, cfg, data=data, cap=cap)
    with ep.scratch() as folder:
        cfg.output_dir = folder
        _ready(cfg, ctx)
        tok = ep.tokenizer(cfg)
        end = turn_end(tok)
        try:
            rows = ep.rows(cfg, shape, r, data, cap, tok, end)
        except Refusal:
            if not dataset:
                raise
            # a config's eval with nothing held out (or a file that no longer splits as the
            # run counted): the config's data, whole, as it is now
            rows = ep.rows(cfg, shape, r, str(cfg.dataset), cap, tok, end)
        with t.activity(f"Loading {short}") as act:
            said: list = []  # the host-RAM line, where the run's own measurement stood for it
            trainer = ep.engine(cfg, r.output_dir, said.append)
            act.done(views.note("host RAM", said[0]) if said else [])
        with t.activity(f"Scoring the base · {rows.data.rows} rows") as act:
            base = ep.score(trainer, rows, end)
            act.done([])
        asked = ep.prompts(rows, show) if show else []
        if asked:
            with t.activity(f"The base answers · {len(asked)}") as act:
                said_base = ep.answers(trainer, asked, end, tok.eos_token_id)
                act.done([])
        ep.trained(trainer, adapter)
        with t.activity(f"Scoring run {r.id} · {rows.data.rows} rows") as act:
            ours = ep.score(trainer, rows, end)
            act.done([])
        if asked:
            with t.activity(f"Run {r.id} answers · {len(asked)}") as act:
                said_ours = ep.answers(trainer, asked, end, tok.eos_token_id)
                act.done([])
            samples = ep.samples(tok, asked, said_base, said_ours)
    said = ep.numbers(base, ours, rows, cfg.dpo_beta)
    loss = said["trained"].get("loss")
    out = EvalOutput(
        adapter=str(adapter), dataset=rows.path, run=r.id,
        metrics={} if rows.data.pairs else {"eval_loss": loss, "eval_perplexity": _ppl(loss)},
        shape=rows.shape, rows_from=rows.source, rows=rows.data.rows, of=rows.total,
        base=EvalSide(**said["base"]), trained=EvalSide(**said["trained"]),
        change=said["change"], improved=said["improved"], answer_rows=said.get("answer_rows"),
        turn_end=end if rows.ends else None, **_support(r))
    if asked:
        out.samples = samples
    return out


def _score_on_loop(r: runs.Run, adapter: Path, ctx: Context, cfg: Any, *,
                   data: Optional[str], cap: int) -> EvalOutput:
    """A run on the engine's own loop: its file of held-out rows (or ``data``) read and scored as
    that loop read and scored them, every token, the adapter off and then on, through the same
    eval pass (``_eval_pass``) at the run's batch and length. With the same rows and weights its
    loss after training is the one the run reported after its last step, as both are taken the
    same way, in chunks on CUDA's fused head (to fp32 rounding of the whole logits' loss)."""
    from poros.cli import views
    from poros.cli.ui import term
    from poros.cli.verbs import evaluate_pass as ep

    rows = ep.loop_rows(cfg, r, data)
    t, short = term.get(), cfg.model_name_or_path.rstrip("/").split("/")[-1]
    with ep.scratch() as folder:
        cfg.output_dir = folder
        _ready(cfg, ctx)
        with t.activity(f"Loading {short}") as act:
            said: list = []
            trainer = ep.engine(cfg, r.output_dir, said.append)
            act.done(views.note("host RAM", said[0]) if said else [])
        ep.read_loop_rows(trainer, rows, r, cap)
        with t.activity(f"Scoring the base · {rows.rows} rows") as act:
            base = ep.score_loop(trainer, rows)
            act.done([])
        ep.trained(trainer, adapter)
        with t.activity(f"Scoring run {r.id} · {rows.rows} rows") as act:
            ours = ep.score_loop(trainer, rows)
            act.done([])
    b, loss = base.loss, ours.loss
    return EvalOutput(
        adapter=str(adapter), dataset=rows.path, run=r.id,
        metrics={"eval_loss": loss, "eval_perplexity": _ppl(loss)},
        shape=None, rows_from=rows.source, rows=rows.rows, of=rows.total,
        base=EvalSide(loss=b), trained=EvalSide(loss=loss),
        change=(loss - b) / b if b and math.isfinite(b) and math.isfinite(loss) else None,
        improved=sum(o < x for x, o in zip(base.rows, ours.rows)), **_support(r))


def folder_run(folder: Path, model: str) -> runs.Run:
    """A config file's run, by its output folder, as the run `score` reads: its settings and
    events are there, and it has no id among the runs `poros runs` lists."""
    return runs.Run(id=folder.name or str(folder), kind="run", state="succeeded", model=model,
                    name=folder.name, output_dir=str(folder), created_at=None, started_at=None,
                    finished_at=None, step=None, total=None, loss=None, error=None,
                    folder=str(folder))


def _ready(cfg, ctx: Context) -> None:
    """Where the engine runs, checked as a run's is: a Mac's gate, else a CUDA GPU."""
    if ctx.unified:
        from poros import metal
        from poros.cli import train_cmd

        metal.require(cfg, check=False)
        train_cmd.mac_ready()
        return
    import torch

    if not torch.cuda.is_available():
        raise Refusal("E_NO_CUDA", "No GPU", "eval streams the run's base on a CUDA GPU, "
                      "and none is here", fix="poros machine", note="inspects this machine")


def _checkpoint(a: EvalAction) -> str:
    return f"poros://{a.run}/weights/{a.step or 'final'}"


def _ppl(loss: Optional[float]) -> Optional[float]:
    return math.exp(loss) if loss is not None and math.isfinite(loss) and loss < 700 else None


def _support(r: runs.Run) -> dict:
    return {"support": "user_asserted", "model_type": r.unvalidated} if r.unvalidated else {}


def _simulated(a: EvalAction, r: runs.Run, adapter: Path) -> EvalOutput:
    """No weights in a preview: the run's own simulated held-out numbers, said to be simulated."""
    from poros.cli.verbs import runs_report

    held = runs_report.held_out(r.output_dir)
    start, end = held["eval_loss_start"], held["eval_loss"]
    shape = (runs_report._read(Path(r.output_dir) / "config.resolved.json") or {}).get(
        "data_shape")
    scored = a.data is None and start is not None and end is not None
    out = EvalOutput(
        adapter=str(adapter), dataset=a.data or "", run=r.id, checkpoint=_checkpoint(a),
        metrics={"eval_loss": end, "eval_perplexity": _ppl(end)} if scored else {},
        shape=shape, rows_from="file" if a.data else "held_out",
        rows=held["eval_rows"] if scored else None,
        base=EvalSide(loss=start) if scored else None,
        trained=EvalSide(loss=end) if scored else None,
        change=(end - start) / start if scored and start else None,
        simulated=True, **_support(r))
    if a.show:  # no model ran: the rows' own prompts and answers, and nothing said for either
        from poros.cli.verbs import evaluate_pass as ep

        out.samples = ep.file_samples(a.data or _data_of(r), a.show)
    return _finished(out, r)


def _data_of(r: runs.Run) -> str:
    from poros.cli.verbs import runs_report

    config = runs_report._read(Path(r.output_dir) / "config.resolved.json") or {}
    return str(config.get("eval_dataset") or config.get("dataset") or "")


def _finished(out: EvalOutput, r: runs.Run) -> EvalOutput:
    from poros.cli.verbs import runs_report

    out.verdict, out.warning = verdict(out, r.model)
    out.next = [x for x in runs_report.offers(r) if not x.startswith("poros eval ")]
    return out


def verdict(out: EvalOutput, model: str) -> tuple[Optional[str], Optional[str]]:
    """(the verdict, a warning): the loss in the run card's words, and on chats whether the
    trained model ends its answers. One that does not is said plainly, with an Instruct model of
    the base's own shape when the catalog lists one."""
    from poros.cli.ui.run_card import verdict as loss_word
    from poros.cli.ui.run_card import with_ends
    from poros.cli.words.models import by_id, instruct_twin

    if out.base is None or out.trained is None:
        return None, None
    word = loss_word(out.base.loss, out.trained.loss)
    if word == "Held-out loss rose" and out.rows_from in ("trained", "file"):
        word = "Loss rose"  # these rows were not held out
    k, n = out.trained.ends, out.answer_rows
    if not n or k is None:
        return word or None, None
    word = with_ends(word, (k, n))
    if 2 * k >= n:
        return word, None
    warn = (f"It does not end its answers: in {n - k} of {n} rows the end-of-turn token is not "
            "its top choice after the answer, so a reply can run to the token limit")
    twin = instruct_twin(model)
    if twin:
        warn += f"; {by_id(model).short} is a base model, and {twin.short} ends its answers"
    return word, warn


def view(r: EvalOutput, w: int) -> list[Line]:
    from poros.cli import views

    return views.evaluated(r, w)
