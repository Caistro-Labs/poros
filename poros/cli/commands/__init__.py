"""The ``poros`` commands, one module per group, registered on ``poros.cli.root.app``:
``train`` (init, train, plan), ``jobs``, ``export`` (eval, export), ``report`` (report,
leaderboard, bench), ``check`` (check, agents, schema).

Importing this package registers all of them, whichever command module is imported first, and
then every verb that has no Typer command of its own as a stub that runs the verb
(``_list_every_command``), so that ``poros --help`` lists every command once: the verbs first,
in ``poros.cli.registry.help_list``'s order, then the config-file forms and tools.
"""

# isort: off
from poros.cli.root import app
from poros.cli.commands import train  # noqa: F401
from poros.cli.commands import jobs

app.add_typer(jobs.jobs_app, name="jobs")
from poros.cli.commands import export  # noqa: E402, F401
from poros.cli.commands import report  # noqa: E402, F401
from poros.cli.commands import check  # noqa: E402, F401
# isort: on


VERBS_PANEL, TOOLS_PANEL = "Commands", "Config files and tools"
TOOLS_ORDER = ("init", "check", "schema", "agents", "jobs", "report", "leaderboard",
               "bench")


def _verb_stub(name: str):
    import typer

    def run(ctx: typer.Context) -> None:
        from poros.cli import app as words

        raise typer.Exit(words.dispatch([name, *ctx.args]) or 0)

    run.__name__ = name
    return run


def _list_every_command() -> None:
    """One command list for ``poros --help``: a stub for each verb Typer has no command for
    (``poros.cli.main.cli`` runs a verb before Typer sees it; the stub does the same for a line
    that reaches Typer, as ``poros --theme dark machine`` does), each verb's one line
    (``registry.Command.about``), the two panels, and the order."""
    from poros.cli import registry

    have = {c.name or c.callback.__name__ for c in app.registered_commands}
    have |= {g.name for g in app.registered_groups}
    verbs = registry.help_list()
    for name in verbs:
        if name not in have:
            app.command(name, add_help_option=False, context_settings={
                "allow_extra_args": True, "ignore_unknown_options": True})(_verb_stub(name))
    for info in (*app.registered_commands, *app.registered_groups):
        name = info.name or info.callback.__name__
        about = registry.by_name(name).about
        if about:
            info.short_help = about
        info.rich_help_panel = VERBS_PANEL if name in verbs else TOOLS_PANEL
    order = [*verbs, *TOOLS_ORDER, *(c.name for c in registry.COMMANDS)]
    app.registered_commands.sort(key=lambda c: order.index(c.name or c.callback.__name__))
    app.registered_groups.sort(key=lambda g: order.index(g.name))


_list_every_command()
