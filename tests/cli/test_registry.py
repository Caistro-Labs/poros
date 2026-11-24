"""One registry of the CLI's commands (poros.cli.registry): every list of commands is derived from
it, and every table that says more about a command names a registered one."""

from __future__ import annotations

from poros.cli import registry

NAMES = {c.name for c in registry.COMMANDS}
# The words of the shell and the session that are not commands of their own.
SHELL_ONLY = {"help", "theme", "version", "clear", "quit", "settings", "share", "exit"}


def test_every_typer_command_is_registered_and_none_more():
    from poros.cli.main import app

    typer = {c.name or c.callback.__name__ for c in app.registered_commands}
    typer |= {g.name for g in app.registered_groups}
    assert typer == set(registry.names(typer=True))


def test_the_lists_derive_from_the_registry():
    from poros.cli import app, contract, session, shell
    from poros.cli.words import grammar

    assert set(grammar.VERBS) == set(registry.names(verb=True)) == set(app.BUILT)
    assert app.TYPER_ONLY == set(registry.names(typer_first=True))
    assert contract._VERBS == set(registry.names(agent=True))
    assert session.VERB_ORDER == registry.help_order()
    assert set(session.SECTION) == {c.name for c in registry.COMMANDS if c.section}
    assert shell.DISPATCHED == set(registry.names(shell=True))


def test_every_table_about_commands_names_registered_ones():
    from poros.cli import session, sheets, shell
    from poros.cli.main import _USAGE_EXAMPLES
    from poros.cli.session.parts import MENU
    from poros.cli.ui.complete import POSITIONAL
    from poros.cli.verbs.agents import COMMANDS as AGENT_COMMANDS

    tables = {
        "shell.COMMANDS": set(shell.COMMANDS),
        "session MENU": {name for name, _ in MENU},
        "session SHELL_WORDS": set(session.SHELL_WORDS),
        "session SHEET_VERBS": set(session.SHEET_VERBS),
        "completions": {name for name, _ in POSITIONAL},
        "usage examples": {key.split()[0] for key in _USAGE_EXAMPLES},
        "sheets FORMS": set(sheets.FORMS),
        "sheets LISTS": set(sheets.LISTS),
        "agents' commands": {target.split()[0] for _, _, target in AGENT_COMMANDS.values()
                             if isinstance(target, str)},
    }
    for table, names in tables.items():
        assert names - NAMES - SHELL_ONLY == set(), table
