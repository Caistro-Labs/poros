"""Questions, asked the same way of people and agents.

In a terminal a Question is the ❯ menu. In JSON mode or without a TTY the
command never prompts: it prints ``poros.needs_input.v1``, each choice with
the full command that picks it, and exits 2 (E_NEEDS_INPUT).
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import NoReturn, Optional, Sequence

import typer

from poros.cli import contract
from poros.cli.schemas import Choice, NeedsInputOutput, Question
from poros.cli.words.grammar import Sentence
from poros.core.env import under_agent
from poros.core.errors import EXIT_CODES

# Override flags: a confirmation's one choice adds one of these; no other
# answer may add any of them (section 6).
OVERRIDES = frozenset(
    {
        "--download",
        "--yes",
        "-y",
        "--force",
        "--overwrite",
        "--allow-unvalidated",
        "--unvalidated",
        "--trust-remote-code",
        "--detach",
        "-d",
    }
)


def interactive() -> bool:
    return (not contract.json_mode() and not under_agent()
            and sys.stdin.isatty() and sys.stdout.isatty())


def _flag_names(tokens: Sequence[str]) -> set[str]:
    return {t.split("=", 1)[0] for t in tokens if t.startswith("-")}


def confirmation(id: str, prompt: str, flag: str, label: str, note: str = "",
                 command: str = "") -> Question:
    """A yes Poros needs before it goes on: one choice, the override ``flag``, and no default.
    ``id`` says what it confirms, so two in one result never share one."""
    return Question(id=id, prompt=prompt, default=None, flag="", confirm=True,
                    choices=[Choice(value=flag, label=label, note=note, command=command)])


ALL = "all"  # the question that answers every confirmation of a result in one command


def commands_for(q: Question, sentence: Sentence) -> Question:
    """Fill each choice's ``command``: the sentence, in canonical form, with
    that answer applied. Enforces the safety rule: only a confirmation adds
    an override flag, exactly one, never pre-selected."""
    if q.confirm and (
        q.default is not None or len(q.choices) != 1 or q.choices[0].value not in OVERRIDES
    ):
        raise RuntimeError("a confirmation has one choice, an override flag, and no default")

    def cmd(choice: Choice) -> list[str]:
        if q.confirm:
            tokens = sentence.canonical(set_flags={choice.value: None})
        elif q.flag == "subject":
            tokens = sentence.canonical(subject=[choice.value])
        else:
            tokens = sentence.canonical(set_flags={q.flag: choice.value})
        if contract.json_mode() and "--json" not in tokens:
            tokens.append("--json")
        return tokens

    q = q.model_copy(deep=True)
    had = _flag_names(sentence.tokens)
    for c in q.choices:
        tokens = shlex.split(c.command)[1:] if c.command else cmd(c)
        added = (_flag_names(tokens) - had) & OVERRIDES
        if added and not (q.confirm and added == {c.value}):
            raise RuntimeError(f"answer {c.value!r} would add {sorted(added)}")
        c.command = shlex.join(["poros", *tokens])
    return q


def with_all(qs: Sequence[Question], sentence: Sentence) -> list[Question]:
    """The questions, and when they hold more than one confirmation, a last one that answers them
    all: one command with each one's flag, and no other, never pre-selected. Without a terminal
    alone: a person is asked one at a time."""
    if any(q.id == ALL for q in qs) or len({q.id for q in qs}) != len(qs):
        raise RuntimeError("a result's questions each have an id of their own")
    asks = [q for q in qs if q.confirm]
    if len(asks) < 2:
        return list(qs)
    flags = list(dict.fromkeys(q.choices[0].value for q in asks))
    tokens = sentence.canonical(set_flags=dict.fromkeys(flags))
    if contract.json_mode() and "--json" not in tokens:
        tokens.append("--json")
    had = _flag_names(sentence.tokens)
    if (_flag_names(tokens) - had) & OVERRIDES != set(flags) - had:
        raise RuntimeError(f"all of them would add more than {flags}")
    ids = ", ".join(q.id for q in asks)
    return [*qs, Question(
        id=ALL, default=None, flag="",
        prompt=f"Or answer all {len(asks)} confirmations ({ids}) in one command. Each one's own "
               "prompt says what it agrees to: show the person each of them first.",
        choices=[Choice(value=" ".join(flags), label="Yes to all of them", note=" ".join(flags),
                        command=shlex.join(["poros", *tokens]))])]


def ask(q: Question) -> Optional[str]:
    """A person's answer from the ❯ menu, or None when they cancel."""
    from poros.cli.ui import term

    start = next((i for i, c in enumerate(q.choices) if c.value == q.default), 0)
    pick = term.get().choose(
        q.prompt, [(c.label, c.note) for c in q.choices], tag=q.id, start=start
    )
    return None if pick is None else q.choices[pick].value


def needs_input(verb: str, questions: Sequence[Question]) -> NoReturn:
    """Return the questions as data and exit 2: nobody waits on a hidden prompt."""
    if contract.json_mode():
        contract.emit(NeedsInputOutput(verb=verb, questions=list(questions)))
    else:  # a pipe or CI without --json: the same facts, readable
        from poros.cli import views
        from poros.cli.ui import term

        t = term.get()
        t.print(views.needs_input(questions, t.width()))
    raise typer.Exit(code=EXIT_CODES["E_NEEDS_INPUT"])


# ---- preferences: a terminal answer, remembered --------------------------------


def _prefs_path() -> Path:
    return Path(os.environ.get("POROS_HOME") or Path.home() / ".poros") / "preferences.toml"


def setting(name: str) -> Optional[str]:
    """What this machine's preferences file says, with or without a terminal."""
    try:
        import tomllib

        return tomllib.loads(_prefs_path().read_text()).get(name)
    except (OSError, ValueError):
        return None


def preference(name: str) -> Optional[str]:
    """A remembered answer. Never consulted in JSON mode or without a TTY, so an
    agent always gets the same Question for the same words."""
    return setting(name) if interactive() else None


def remember(name: str, value: str) -> None:
    """Save a terminal answer. Keys and values are written as TOML basic
    strings (JSON's escapes are TOML's), so any text round-trips."""
    import json
    import tomllib

    path = _prefs_path()
    try:
        prefs = tomllib.loads(path.read_text())
    except (OSError, ValueError):
        prefs = {}
    prefs[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(f"{json.dumps(k)} = {json.dumps(str(v))}\n" for k, v in sorted(prefs.items()))
    )
    tmp.replace(path)
