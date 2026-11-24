"""The verbs through app.dispatch, against a fixture machine (no GPU, no network)."""

from __future__ import annotations

import json

import pytest

from poros.cli import app, contract
from poros.cli.context import Context
from poros.cli.schemas import PlanGpu
from tests.conftest import needs_ml

RTX3090 = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=None, source="spec")


@pytest.fixture
def ctx(tmp_path):
    return Context(
        cwd=tmp_path, fixed_gpu=RTX3090, fixed_cached=frozenset(), fixed_interactive=False
    )


@pytest.fixture(autouse=True)
def json_mode(monkeypatch):
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=None, unit="GiB", source="unknown", why="offline in tests"),
    )
    yield
    contract.reset()


def run(argv, ctx, capsys):
    code = app.dispatch(argv, ctx)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out  # the contract: exactly one JSON object
    return code, json.loads(out[0])


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "run.yaml"],
        ["plan", "x.yml"],
        ["check", "Qwen/Qwen2.5-7B"],
        ["--version"],
        [],
    ],
)
def test_old_forms_stay_with_typer(argv):
    assert app.dispatch(argv) is None


@needs_ml
def test_a_missing_model_is_a_question_never_a_guess(ctx, capsys):
    code, obj = run(["train"], ctx, capsys)
    assert code == 2 and obj["schema"] == "poros.needs_input.v1"
    q = obj["questions"][0]
    assert q["id"] == "model" and q["default"] is None
    assert q["choices"] and all(c["command"].startswith("poros train ") for c in q["choices"])


def test_a_catalog_name_is_exact_and_case_blind(ctx, capsys):
    code, obj = run(["arch", "qwen2.5-7b-instruct"], ctx, capsys)
    assert obj["schema"] == "poros.arch.v1" and obj["model"] == "Qwen/Qwen2.5-7B-Instruct"


@pytest.mark.parametrize("argv", [["qwen", "7b"], ["/qwen"]])
def test_a_line_without_a_verb_or_with_loose_words_does_nothing(argv, ctx, capsys):
    code = app.dispatch(argv, ctx)
    out = capsys.readouterr().out.strip()
    if code is None:  # no verb: Typer's own "no such command"
        assert argv[0].lstrip("/") not in app.BUILT
    else:
        assert code == 2 and json.loads(out)["error"]["code"] == "E_USAGE"


def test_no_model_matches(ctx, capsys):
    code, obj = run(["arch", "llama-8b"], ctx, capsys)
    assert code == 2 and obj["error"]["code"] == "E_NOT_FOUND"
    assert "closest" not in obj["error"]["message"]  # no family in common: no guesses
    code, obj = run(["arch", "qwen-9b"], ctx, capsys)
    assert "closest: Qwen" in obj["error"]["message"]


@pytest.mark.parametrize("argv", [["arch", "72b"], ["arch", "qwen2.5", "7b"],
                                  pytest.param(["train", "qwen7b"], marks=needs_ml)])
def test_part_of_a_name_asks_which_never_guesses(argv, ctx, capsys):
    code, obj = run(argv, ctx, capsys)
    assert code == 2 and obj["schema"] == "poros.needs_input.v1"
    q = obj["questions"][0]
    assert q["id"] == "model" and len(q["choices"]) >= 2
    assert all(c["command"].startswith(f"poros {argv[0]} Qwen/") for c in q["choices"])


def test_models_lists_the_catalog(ctx, capsys):
    code, obj = run(["models"], ctx, capsys)
    assert code == 0 and obj["schema"] == "poros.models.v1" and len(obj["models"]) == 23


def test_a_download_of_unknown_size_needs_a_yes(ctx, capsys, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)  # the size lookup is stubbed
    code, obj = run(["models", "get", "DeepSeek-V2-Lite"], ctx, capsys)
    assert code == 2 and obj["schema"] == "poros.needs_input.v1"
    (q,) = obj["questions"]
    assert q["id"] == "download" and q["default"] is None
    assert q["choices"][0]["command"].split() == [
        "poros",
        "models",
        "get",
        "DeepSeek-V2-Lite",
        "--yes",
        "--json",
    ]


def test_machine(ctx, capsys):
    code, obj = run(["machine"], ctx, capsys)
    assert obj["schema"] == "poros.machine.v1" and obj["gpu"]["name"] == RTX3090.name
    assert "can_train" not in obj  # what trains where is `poros recipes`


def test_models_get_refuses_a_download_past_the_free_disk(ctx, capsys, monkeypatch):
    """`poros models get <model> --yes` checks the free disk first: it refuses as train does
    (E_DISK), before anything is fetched."""
    import shutil
    from collections import namedtuple

    import poros.plan.configs
    from poros.cli.schemas import Figure
    from poros.cli.verbs import models

    monkeypatch.setattr(poros.plan.configs, "download_size",
                        lambda cfg: Figure(value=15.2, unit="GiB", source="hub_metadata"))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(2**40, 2**40 - 2**30, 2**30))
    monkeypatch.setattr(models, "_download", lambda m: pytest.fail("downloaded"))
    code, obj = run(["models", "get", "Qwen/Qwen2.5-7B", "--yes"], ctx, capsys)
    assert obj["error"]["code"] == "E_DISK" and "is a 16.3 GB download" in obj["error"]["message"]
