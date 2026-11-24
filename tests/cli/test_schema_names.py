"""`poros schema` and `poros agents schema` publish one set of schemas from one table
(poros.cli.schemas: OUTPUTS and RESULTS), so `poros schema` has every verb result the agents'
command has."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from poros.cli.main import app
from poros.cli.schemas import OUTPUTS, RESULTS, schema_names

RESULT_NAMES = ["machine", "models", "runs", "run_live", "data", "arch", "recipes", "prove",
          "prove_gates", "docs"]


def _run(*args):
    out = CliRunner().invoke(app, list(args))
    assert out.exit_code == 0, out.output
    return out.stdout


def test_poros_schema_lists_every_result_the_agents_command_does():
    listed = _run("schema", "--list").split()
    assert listed == schema_names()
    for name in RESULT_NAMES + ["config", "events", "artifact", *OUTPUTS]:
        assert name in listed, name


def test_each_name_prints_the_same_schema_from_both_commands(capsys):
    from poros.cli.verbs.agents import _schema

    capsys.readouterr()
    for name in sorted(OUTPUTS.keys() | RESULTS.keys()):
        doc = json.loads(_run("schema", name))
        assert doc["$id"] == f"poros.{name}.v1", name
        assert _schema([name]) == 0
        assert doc == json.loads(capsys.readouterr().out), name


def test_export_is_the_fuller_model_in_both():
    doc = json.loads(_run("schema", "export"))
    assert {"adapter", "output", "format", "run", "base_model", "load"} <= doc["properties"].keys()
