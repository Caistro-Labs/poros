"""A command line into parts: the verb, its subject, its flags.

    poros train Qwen2.5-7B-Instruct --data notes.jsonl --steps 300
    poros models
    poros runs stop 7f3a

Strict on purpose: a verb, at most one subject (a model or a run), flags,
and ``key=value`` config fields. No filler words, no synonyms, no guessing:
the interactive session opens a settings sheet for anything left out.
Pure: no I/O, no machine state.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from poros.cli import registry

VERBS = registry.names(verb=True)  # the sentence verbs (poros.cli.registry)

# Words that are a verb's own action: `runs stop 7f3a`, `models get <id>`.
RUN_ACTIONS = {"stop": "stop", "continue": "continue", "report": "report",
               "checkpoints": "checkpoints"}
MODEL_ACTIONS = {"get": "get"}
RECIPE_ACTIONS = {"copy": "copy", "show": "show", "save": "save", "share": "share", "sync": "sync"}
ARCH_ACTIONS = {"try": "try"}

# Flags that take no value. Every other --flag takes the next token.
BOOL_FLAGS = frozenset(
    "--yes -y --force --json --detach -d --overwrite --allow-unvalidated --unvalidated "
    "--trust-remote-code "
    "--download "
    "--md --markdown --html --no-html --follow -f --help -h --plain --no-anim --version -V "
    "--list --convert --fix --live --frame --merged --explain".split()
)

# Field -> its flag in a canonical command.
FLAG_OF = {
    "data": "--data",
    "steps": "--steps",
    "epochs": "--epochs",
    "seq_len": "--seq-len",
    "rank": "--rank",
    "output": "--output",
    "name": "--name",
    "gpu": "--gpu",
}

@dataclass
class Clause:
    """One ``key=value`` config field, and how it was typed."""

    field: str
    value: Union[str, int]
    said: str


@dataclass
class Sentence:
    verb: Optional[str]  # one of VERBS, None when the line named none
    said_verb: str = ""
    action: Optional[str] = None  # runs: stop | continue | report | checkpoints; models: get
    subject: list[str] = field(default_factory=list)  # a model or a run; one word expected
    clauses: dict[str, Clause] = field(default_factory=dict)
    flags: dict[str, Union[str, bool]] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)  # what was typed

    def value(self, name: str) -> Optional[Union[str, int, bool]]:
        """A field's value: its flag, else its key=value."""
        flag = FLAG_OF.get(name, "--" + name.replace("_", "-"))
        if flag in self.flags:
            return self.flags[flag]
        clause = self.clauses.get(name)
        return clause.value if clause else None

    def canonical(
        self,
        *,
        subject: Optional[Sequence[str]] = None,
        set_flags: Optional[dict[str, Optional[str]]] = None,
        verb: Optional[str] = None,
    ) -> list[str]:
        """The line as flags: verb, action, subject, then every flag and
        key=value. ``subject`` replaces the subject; ``set_flags`` adds or
        replaces flags (None: a flag with no value)."""
        out = [verb or self.verb or "models"]
        if self.action:
            out.append(self.action)
        out += list(self.subject if subject is None else subject)
        flags: dict[str, Union[str, bool, None]] = {}
        extra: list[str] = []
        for name, clause in self.clauses.items():
            if name in FLAG_OF:
                flags[FLAG_OF[name]] = str(clause.value)
            else:
                extra.append(f"{name}={clause.value}")
        flags.update(self.flags)
        flags.update(set_flags or {})
        for flag, value in flags.items():
            if value is True or value is None:
                out.append(flag)
            elif value is not False:
                out += with_value(flag, str(value))
        return out + extra


def tokenize(text: Union[str, Sequence[str]]) -> list[str]:
    """A typed line (shell quoting honored) or argv, as tokens."""
    if isinstance(text, str):
        try:
            return shlex.split(text)
        except ValueError:  # an unclosed quote: split plainly rather than refuse
            return text.split()
    return list(text)


def _is_flag(token: str) -> bool:
    return token.startswith("-") and len(token) > 1 and not token[1:].replace(".", "").isdigit()


def is_clause(token: str) -> bool:
    """``key=value``: what a word outside any flag is read as, before it could be a subject."""
    return "=" in token and token.split("=", 1)[0].replace("_", "").isalnum() and token[0].isalpha()


def with_value(flag: str, value: str) -> list[str]:
    """A flag and its value as words that read back the same: one that looks like a flag is kept
    attached, or `--name --force` would read as a yes."""
    return [f"{flag}={value}"] if _is_flag(value) else [flag, value]


def parse(text: Union[str, Sequence[str]]) -> Sentence:
    tokens = tokenize(text)
    s = Sentence(verb=None, tokens=tokens)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if _is_flag(tok):
            # a flag that takes no value takes none this way either: `--force=1` is no yes
            if "=" in tok and tok.startswith("--") and tok.partition("=")[0] not in BOOL_FLAGS:
                name, _, val = tok.partition("=")
                s.flags[name] = val
            elif tok in BOOL_FLAGS or nxt is None or _is_flag(nxt):
                s.flags[tok] = True  # `--gpu --json`: a flag is never another flag's value
            else:
                s.flags[tok] = nxt
                i += 1
            i += 1
            continue
        word = tok[1:] if i == 0 and tok.startswith("/") else tok  # `/train` in the session
        if s.verb is None and not s.subject and word.lower() in VERBS:
            s.verb = s.said_verb = word.lower()
        elif s.verb == "runs" and s.action is None and not s.subject and word in RUN_ACTIONS:
            s.action = RUN_ACTIONS[word]
        elif s.verb == "models" and s.action is None and not s.subject and word in MODEL_ACTIONS:
            s.action = MODEL_ACTIONS[word]
        elif s.verb == "recipes" and s.action is None and not s.subject and word in RECIPE_ACTIONS:
            s.action = RECIPE_ACTIONS[word]
        elif s.verb == "arch" and s.action is None and not s.subject and word in ARCH_ACTIONS:
            s.action = ARCH_ACTIONS[word]
        elif is_clause(tok):
            key, _, val = tok.partition("=")
            s.clauses[key] = Clause(key, val, tok)
        else:
            s.subject.append(tok)
        i += 1
    return s
