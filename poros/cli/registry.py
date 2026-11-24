"""Every ``poros`` command, once: its name and what kind of command it is.

The lists the CLI keeps of its commands are derived from ``COMMANDS``: the sentence verbs
(``poros.cli.words.grammar.VERBS``, the verb modules ``poros.cli.app.BUILT``), the commands Typer
takes first (``poros.cli.app.TYPER_ONLY``), what ``poros agents <verb>`` runs
(``poros.cli.contract``), the session's sections and the order its help lists the verbs in
(``poros.cli.session``) and the commands the plain shell hands to Typer (``poros.cli.shell``).
The tables that say more about a command (its help line, its completions, its examples, its
sheet) name only commands registered here: tests/cli/test_registry.py checks each one, and that
every Typer command (``poros.cli.commands``) is registered.

A new command is one ``Command`` line here, its module (``poros.cli.verbs`` for a verb,
``poros.cli.commands`` for a Typer command), and whichever of those tables it needs. Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Command:
    name: str
    verb: Optional[str] = None  # a sentence verb: the module in poros.cli.verbs that runs it
    # a Typer command or group of this name (poros.cli.commands): every verb has one, so that
    # `poros --help` lists it; a verb with no config-file form is a stub that runs the verb
    typer: bool = False
    typer_first: bool = False  # Typer runs it before the words are parsed (TYPER_ONLY)
    agent: bool = False  # `poros agents <name>` runs it as `poros <name>` (poros.cli.contract)
    section: Optional[str] = None  # where the session's help lists it: "verbs" or "advanced"
    help_rank: Optional[int] = None  # its place in the session's help, before every unranked one
    shell: bool = False  # the plain shell hands it to Typer (poros.cli.shell.DISPATCHED)
    about: str = ""  # a verb's one line in `poros --help`'s command list


V = "poros.cli.verbs."
COMMANDS: tuple[Command, ...] = (
    # the verbs (the order completions list them in); help_rank: where the session's help puts it
    Command("train", verb=V + "train", typer=True, agent=True, section="verbs", help_rank=4,
            shell=True, about="Fine-tune a model on your data, or from a config file."),
    Command("runs", verb=V + "runs", typer=True, section="verbs", help_rank=6,
            about="Your runs: watch, stop, continue, report."),
    Command("machine", verb=V + "machine", typer=True, agent=True, section="verbs", help_rank=0,
            about="Whether this machine is ready to train."),
    Command("export", verb=V + "export", typer=True, agent=True, section="verbs", help_rank=8,
            shell=True, about="Turn a run into a model you can serve."),
    Command("prove", verb=V + "prove", typer=True, agent=True, section="verbs", help_rank=9,
            about="Train streamed and fully resident, and compare."),
    Command("models", verb=V + "models", typer=True, section="verbs", help_rank=1,
            about="The models Poros names, with the recipes certified on this machine."),
    Command("recipes", verb=V + "recipes", typer=True, agent=True, section="verbs", help_rank=5,
            about="Whole setups to start from, and the loops they train with."),
    Command("arch", verb=V + "arch", typer=True, agent=True,
            about="What any model would take to train, from its config alone."),
    Command("data", verb=V + "data", typer=True, agent=True, section="verbs", help_rank=3,
            about="What a file trains; convert it, or write a clean copy."),
    Command("eval", verb=V + "evaluate", typer=True, agent=True, section="verbs", help_rank=7,
            shell=True, about="Score a run beside the model it started from."),
    Command("docs", verb=V + "docs", typer=True, agent=True,
            about="The docs, offline, from this install."),
    # the config-file forms and the advanced tools (Typer's)
    Command("report", typer=True, typer_first=True, section="verbs", shell=True),
    Command("init", typer=True, typer_first=True, section="advanced", shell=True),
    Command("check", typer=True, typer_first=True, section="advanced", shell=True),
    Command("schema", typer=True, typer_first=True, agent=True, section="advanced", shell=True),
    Command("leaderboard", typer=True, typer_first=True, section="advanced", shell=True),
    Command("bench", typer=True, typer_first=True, section="advanced", shell=True),
    Command("jobs", typer=True, typer_first=True, section="advanced"),
    Command("agents", typer=True, typer_first=True),
    Command("help", typer_first=True),  # `poros help [verb]`: --help, said as a word
)


def names(**kind) -> tuple[str, ...]:
    """The registered commands whose fields equal ``kind`` (``verb=...`` matches any verb when
    given True), in registry order."""
    def fits(c: Command) -> bool:
        for key, want in kind.items():
            have = getattr(c, key)
            if want is True and key == "verb":
                if have is None:
                    return False
            elif have != want:
                return False
        return True

    return tuple(c.name for c in COMMANDS if fits(c))


def help_list() -> tuple[str, ...]:
    """Every verb in the order `poros --help` lists them: the ranked ones (``help_order``), then
    the rest as registered."""
    ranked = help_order()
    return ranked + tuple(c.name for c in COMMANDS if c.verb and c.name not in ranked)


def help_order() -> tuple[str, ...]:
    """The commands the session's help lists first, in their order (``help_rank``)."""
    ranked = sorted((c.help_rank, c.name) for c in COMMANDS if c.help_rank is not None)
    return tuple(name for _, name in ranked)


def by_name(name: str) -> Optional[Command]:
    return next((c for c in COMMANDS if c.name == name), None)
