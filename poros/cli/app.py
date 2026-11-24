"""argv or a session line → sentence → Action → the verb's stages.

    dispatch(["models"])              # handled: returns the exit code
    dispatch(["train", "run.yaml"])   # a config file: None, Typer runs it

One loop for every verb: resolve; while there
are Questions, ask them (or return them as needs_input) and fold each answer
into the sentence as a flag; then confirmations, the same way; then run and
view. Forms that name a config file or an adapter directory, and the
advanced commands, stay with the Typer app.
"""

from __future__ import annotations

import dataclasses
import importlib
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

import typer

from poros.cli import contract, questions, registry
from poros.cli.context import Context
from poros.cli.words.grammar import VERBS, Sentence, parse
from poros.core.errors import EXIT_CODES
from poros.core.refusal import Refusal

# The verbs built on words, and the module that runs each (poros.cli.registry).
BUILT = {c.name: c.verb for c in registry.COMMANDS if c.verb}
# Typer commands that are not v1 verbs (advanced, or kept for one release).
TYPER_ONLY = set(registry.names(typer_first=True))


def _typer_commands() -> set[str]:
    from poros.cli.main import app as typer_app

    names = {
        c.name or (c.callback.__name__ if c.callback else "") for c in typer_app.registered_commands
    }
    return names | {g.name for g in typer_app.registered_groups if g.name}


def _typer_form(argv: Sequence[str]) -> bool:
    """The old shapes: `train run.yaml`, `plan run.yaml`, `eval run.yaml`, `export ./adapter`."""
    head = argv[0].lower().lstrip("/")
    if head.startswith("-") or head in TYPER_ONLY:
        return True
    if head not in VERBS and head in _typer_commands():
        return True  # an advanced command: Typer runs it
    args = argv[1:]
    if head in ("train", "eval") and any(a.lower().endswith((".yaml", ".yml"))
                                                 for a in args):
        return True  # a config file anywhere: `plan --gpu "RTX 4090" run.yaml`
    positional = [a for a in args if not a.startswith("-")]
    if head == "export" and positional and _is_dir(positional[0]):
        return not _names_a_run(positional[0])  # a project can hold a folder called last
    return False


def _is_dir(word: str) -> bool:
    try:
        return Path(word).expanduser().is_dir()
    except (OSError, ValueError):  # a name too long for the file system names no folder
        return False


def _names_a_run(word: str) -> bool:
    from poros import runs
    from poros.cli.words import runs as run_words

    every = runs.all_runs()
    ids = [r.id for r in every]
    if not ids:  # run_words' own answer to `last`, when there is none: that is still a run's word
        return word.lower() in ("last", "latest", "newest")
    found = run_words.resolve(word, ids, newest=ids[0], names=[(r.name, r.id) for r in every])
    return not isinstance(found, run_words.RunMiss)


def dispatch(argv: Sequence[str], ctx: Optional[Context] = None) -> Optional[int]:
    """Run ``argv`` (after ``poros``) as a sentence; None hands it to Typer."""
    argv = list(argv)
    lead = 0
    while lead < len(argv) and argv[lead] == "--json":
        lead += 1
    if lead and lead < len(argv):  # `poros --json machine`
        argv = argv[lead:] + ["--json"]
    if not argv:
        return None
    head = argv[0].lower().lstrip("/")
    if head in BUILT and ({"--help", "-h"} & set(argv[1:])):
        return _help(head)  # one help per verb, its config-file form too (`train run.yaml -h`)
    if _typer_form(argv):
        return None
    s = parse(argv)
    verb = s.verb
    if verb is None or verb not in BUILT:
        return None
    if "--json" in argv:  # decided from argv, as main.cli isolates stdout
        contract.force_json()
    if "--help" in s.flags or "-h" in s.flags:
        return _help(verb)

    def work() -> int:
        _known_flags(verb, s)
        return run_sentence(verb, s, ctx or Context())

    return guarded(work)


def guarded(work) -> int:
    """``work()``'s exit code, however it ended: a refusal, or a config, network or Hub error, is
    the contract's error. What ``--unvalidated`` asserts ends with it."""
    from poros.core.arch_registry import one_command

    with one_command():
        return _guarded(work)


def _guarded(work) -> int:
    try:
        return work()
    except Refusal as r:
        try:
            contract.refuse(r.code, r.title, r.reason, context=r.context, fix=_fix(r),
                            docs=r.docs)
        except typer.Exit as e:
            return e.exit_code
    except typer.Exit as e:
        if contract.json_mode() and not contract.emitted() and e.exit_code:
            # The contract: an exit with nothing on stdout still says why.
            code = (
                "E_INTERRUPTED"
                if e.exit_code == 130
                else next((k for k, v in EXIT_CODES.items() if v == e.exit_code), "E_INTERNAL")
            )
            contract.emit_error(code, "stopped" if e.exit_code == 130 else "failed")
        return e.exit_code
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        from poros.cli.errors import _friendly_error

        try:
            _friendly_error(exc)
        except typer.Exit as e:
            return e.exit_code
    return 1


# Flags a verb reads that tab does not offer: short forms and quiet aliases.
_QUIET_FLAGS = {
    "train": ("-y", "-d", "--download", "--gpu", "--sss-ram"),
    "models": ("-y",),
    "runs": ("-f", "-y"),
    # -o: the folder form's short --output; --merged: --format merged, as `poros agents` says it
    "export": ("--yes", "-y", "-o", "--download", "--merged"),
    "prove": ("--yes", "-y"),
    "data": ("-o",),
}
if sys.platform == "darwin":  # a Mac's question for which disk answers with this
    _QUIET_FLAGS["models"] += ("--store",)
_EVERY_VERB = ("--json", "--help", "-h", "--plain", "--no-anim", "--theme")


def _known_flags(verb: str, s) -> None:
    """A flag the verb does not read is refused, never ignored: a typo
    that trains the wrong thing is worse than a question."""
    from poros.cli.ui.complete import VERB_FLAGS

    if verb == "runs" and s.action == "continue":
        return  # it says why itself: a continue takes the run's own settings
    shown = VERB_FLAGS.get(verb, ())
    known = set(shown) | set(_QUIET_FLAGS.get(verb, ())) | set(_EVERY_VERB)
    unknown = [f for f in s.flags if f not in known]
    if not unknown:
        return
    import difflib

    near = difflib.get_close_matches(unknown[0], shown, n=1)
    raise Refusal(
        "E_USAGE",
        "Unknown flag",
        f"{verb} has no {unknown[0]}" + (f"; did you mean {near[0]}?" if near else ""),
        context=["its flags: " + " ".join(f for f in shown if f != "--json")],
        fix=f"poros {verb} --help",
        note="every flag, with what it does",
    )


def _fix(r: Refusal):
    from poros.cli.ui.screens import Fix

    return Fix("", r.fix, r.note) if r.fix else None


def _device_first(verb: str, s: Sentence, ctx: Context) -> None:
    """What a verb that trains or merges needs on this machine, refused before it asks anything,
    checks a disk or downloads: train and prove torch and (off a Mac) CUDA, a merged export torch.
    --unvalidated's gates say what they need themselves."""
    from poros.cli.verbs.shared import refuse_without_a_device

    if verb in ("train", "prove") and not s.flags.get("--unvalidated"):
        refuse_without_a_device(ctx)
    elif verb == "export" and str(s.value("format") or "").lower() in (
            "merged", "safetensors", "model", "standalone"):  # a copy of the adapter needs none
        refuse_without_a_device(ctx, cuda=False)


def run_sentence(verb: str, s: Sentence, ctx: Context) -> int:
    if ctx.interactive and not ctx.sheet and not contract.json_mode():
        from poros.cli import sheets

        if sheets.wanted(verb, s):
            argv = sheets.open_sheet(verb, s, ctx)
            told = _tell(argv)  # in the session, the typed line becomes the real command
            if argv is None:  # esc on a card: back to the prompt, nothing to report
                from poros.cli.ui import components as c

                return EXIT_CODES["E_INTERRUPTED"] if c.in_session else _cancelled()
            if not told:
                _settled(argv)
            s, verb = parse(argv), argv[0]
            ctx = dataclasses.replace(ctx, sheet=True)
    mod = importlib.import_module(BUILT[verb])
    answered: dict = {}  # field -> (the value chosen, its Source)

    def resolved(s: Sentence):
        draft = mod.resolve(s, ctx)
        # after the input's own refusals, before any question. A verb with checks of its input
        # that come before the device's says them first (train: its data), and says when the
        # question on the table is such a refusal itself
        ready = getattr(mod, "before_the_device", None)
        if ready is None or ready(draft, ctx):
            _device_first(verb, s, ctx)
        while draft.questions:
            s = _answer(verb, s, draft.questions, ctx, answered)
            if s is None:
                return None, s
            draft = mod.resolve(s, ctx)
        action = draft.action
        for field, (value, source) in answered.items():
            if getattr(action, field, None) == value:  # still the value that was chosen
                action.provenance[field] = source
        action = mod.inspect(action, ctx) if hasattr(mod, "inspect") else action
        # no card opened: the same opening settings a card would have given
        return (mod.settle(action, ctx) if hasattr(mod, "settle") and not ctx.sheet
                else action), s

    action, s = resolved(s)
    if action is None:
        return _cancelled()
    shown = 0  # lines of the resolved-action screen on the terminal now
    while True:
        confirms = mod.confirmations(action, ctx)
        if not confirms:
            break
        if confirms[0].id == "start":  # the resolved-action screen: start, change, cancel
            edited, shown = _start_screen(mod, action, s, ctx, shown)
            if edited is True:
                break
            if edited is None:
                return _cancelled()
            s = edited
        else:
            s = _answer(verb, s, confirms, ctx, answered)
            if s is None:
                return _cancelled()
        action, s = resolved(s)
        if action is None:
            return _cancelled()
    result = mod.run(action, ctx)
    if contract.json_mode():
        contract.emit(result)
    else:
        from poros.cli.ui import term

        t = term.get()
        t.show(lambda w: mod.view(result, w))
    return getattr(result, "exit_code", 0) or 0


def _tell(argv: Optional[Sequence[str]]) -> bool:
    """After a full-screen card in the session: the line left in scrollback is
    the command the card settled on, as if typed (nothing at all after esc).
    False when there is no such line to tell (one-shot, or an inline card)."""
    import sys

    from poros.cli.session import echo_lead
    from poros.cli.ui import live, tui
    from poros.cli.ui.style import resolve

    if tui.current is not None:  # the tab's typed line becomes the command
        tui.retell("/" + shlex.join(argv) if argv else None)
        return True
    if not live.waiting():
        return False
    live.retell(echo_lead("/" + shlex.join(argv), resolve(sys.stdout)) if argv else "")
    return True


def _settled(argv: Sequence[str]) -> None:
    """What the sheet settled on, as the command that does it again."""
    from poros.cli import views
    from poros.cli.ui import components as c
    from poros.cli.ui import term

    term.get().print(views.note(c.command(shlex.join(argv)), "the same, next time"))


def _start_screen(mod, action, s: Sentence, ctx: Context, shown: int):
    """(True to start | None to cancel | the edited sentence, lines now shown).

    The screen is redrawn in place after a change. Only ever shown in a
    terminal: a verb asks for it only then.
    """
    from poros.cli.ui import term

    t = term.get()
    t.erase(shown)
    lines = mod.preview(action, t.width())
    t.print(lines)
    start = mod.confirmations(action, ctx)[0]
    while True:
        value = questions.ask(start)
        if value == "start":
            return True, len(lines)
        if value is None:
            return None, len(lines)
        field_q, value_q = mod.edit_questions(action, ctx)
        field = questions.ask(field_q)
        if field is None:
            continue  # back to start / change / cancel
        q = questions.commands_for(value_q(field), s)
        picked = questions.ask(q)
        if picked is None:
            continue
        chosen = next(c for c in q.choices if c.value == picked)
        return parse(shlex.split(chosen.command)[1:]), len(lines)


def _answer(verb: str, s: Sentence, qs, ctx: Context, answered: dict) -> Optional[Sentence]:
    """The sentence with the first question answered, or None if cancelled.

    Without a terminal nothing is asked: every question goes back as data.
    """
    from poros.cli.action import Source

    qs = [questions.commands_for(q, s) for q in qs]
    if not ctx.interactive:
        questions.needs_input(verb, questions.with_all(qs, s))
    q = qs[0]
    value = questions.ask(q)
    if value is None:
        return None
    if q.remember == "variant":
        from poros.cli.words.models import by_id

        hit = by_id(value)
        if hit is not None and hit.variant in ("base", "instruct"):
            questions.remember("variant", hit.variant)
            from poros.cli import views
            from poros.cli.ui import term

            term.get().print(
                views.note(
                    f"Remembered: {hit.variant}",
                    "used when a size has both · ~/.poros/preferences.toml to change",
                )
            )
    if q.remember and q.remember != "variant":  # e.g. export's format
        from poros.cli import views
        from poros.cli.ui import term

        questions.remember(q.remember, value)
        term.get().print(views.note(f"Remembered: {value}", "~/.poros/preferences.toml to change"))
    if not q.confirm:
        said = " ".join(s.subject) if q.flag == "subject" else ""
        answered[q.id] = (
            value,
            Source(kind="given", detail=f'chosen for "{said}"' if said else "chosen"),
        )
    chosen = next(c for c in q.choices if c.value == value)
    return parse(shlex.split(chosen.command)[1:])


def _cancelled() -> int:
    from poros.cli.ui import components as c
    from poros.cli.ui import term

    term.get().print([c.event("Cancelled", None, "nothing ran", "○", "dm")])
    return EXIT_CODES["E_INTERRUPTED"]


def _help(verb: str) -> int:
    """A verb's docstring: what it does and the sentences it takes."""
    from poros.cli import views
    from poros.cli.ui import term

    doc = importlib.import_module(BUILT[verb]).__doc__ or verb
    t = term.get()
    t.print(views.verb_help(verb, doc, t.width()))
    return 0
