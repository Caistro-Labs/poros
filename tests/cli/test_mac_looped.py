"""On a Mac, a decoder that runs its layers as visits (Inkling 2 Mini's loop) is refused before
anything downloads, by check, arch and train alike, --unvalidated or not, validated or not, in
words true to the family's registry status: a validated family trains on Linux with an NVIDIA
GPU; a target is not validated on any machine yet, and nothing is said of Linux (where it does
not train either). The fixture Mac of tests/cli/test_mac.py: no Mac, no GPU, no network."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tests.cli.test_mac import _check, as_json, ctx, mac  # noqa: F401 - the fixture Mac
from tests.conftest import needs_ml

RELEASED = json.loads((Path(__file__).parents[1] / "metal" / "real_configs"
                       / "inkling_2_mini.json").read_text())
LINUX = "trains on Linux with an NVIDIA GPU"
TARGET = "has not validated it on any machine yet"


def words(status):
    """What the refusal must say, and what it must not, for the family's status."""
    return (LINUX, TARGET) if status == "validated" else (TARGET, "Linux")


@pytest.fixture
def folder(tmp_path):
    here = tmp_path / "inkling"
    here.mkdir()
    (here / "config.json").write_text(json.dumps(RELEASED))
    with open(here / "model.safetensors", "wb") as f:
        f.truncate(10**6)
    return str(here)


@pytest.fixture(params=["target", "validated"])
def status(request, monkeypatch):
    """As the registry has it now (target), and as it will once validated on Linux."""
    import poros.core.arch_registry as registry

    if request.param == "validated":
        for name in ("inkling2_loop", "inkling2_loop_text"):
            spec = dataclasses.replace(registry._BUILTIN_ARCHITECTURES[name], status="validated")
            monkeypatch.setitem(registry._BUILTIN_ARCHITECTURES, name, spec)
            monkeypatch.setitem(registry._ARCHITECTURES, name, spec)
    return request.param


def test_the_family_declares_its_visits_and_the_recorded_list_agrees():
    from poros.moe import carried

    for kind in carried.CARRIED:
        assert carried.declares_visits(kind) == (kind in carried.LOOPED)
    assert carried.declares_visits(RELEASED) and not carried.declares_visits("qwen2")


def test_check_says_it_does_not_train_on_a_mac(mac, folder, status):  # noqa: F811
    code, said, then, drawn = _check(folder)
    assert (code, said, then) == (4, "not-on-this-device", [])
    must, never = words(status)
    assert must in " ".join(drawn.split()) and never not in drawn


@needs_ml
@pytest.mark.parametrize("flag", [[], ["--unvalidated"]])
def test_train_refuses_it_before_anything_downloads(mac, ctx, capsys, folder, status,  # noqa: F811
                                                    flag):
    code, obj = as_json(["train", folder, "--data", "notes.jsonl", "--yes", *flag], ctx, capsys)
    assert code == 4 and obj["error"]["code"] == "E_ARCH_UNSUPPORTED", obj
    must, never = words(status)
    assert must in json.dumps(obj) and never not in json.dumps(obj), obj


@needs_ml
def test_arch_refuses_it(mac, ctx, capsys, folder, status):  # noqa: F811
    code, obj = as_json(["arch", folder], ctx, capsys)
    assert code == 4 and obj["refused"]["code"] == "E_ARCH_UNSUPPORTED", obj
    must, never = words(status)
    assert must in obj["refused"]["reason"] and never not in obj["refused"]["reason"]


@needs_ml
def test_the_trainer_refuses_it_on_a_mac(mac, folder, status):  # noqa: F811
    """poros.Run and PorosTrainer reach the route without the CLI: refused there too."""
    from poros.preflight.arch import LoopedOnAMacError
    from poros.preflight.start import Startup, route

    class _Trainer:
        _metal = True
        config = type("C", (), {"model_revision": None, "moe_expert_format": None})()

    must, never = words(status)
    with pytest.raises(LoopedOnAMacError, match=must) as err:
        route(_Trainer(), folder, Startup())
    told = str(err.value) + str(err.value.info.remediation)
    assert never not in told


def test_the_metal_path_and_the_mac_gate_say_the_same(mac, folder, status):  # noqa: F811
    """The Mac path's own refusal (poros.metal) and the agent verbs' gate say it alike."""
    from types import SimpleNamespace

    from poros.metal import _refuse_a_looped_decoder
    from poros.preflight.arch import LoopedOnAMacError

    must, never = words(status)
    with pytest.raises(LoopedOnAMacError, match=must) as err:
        _refuse_a_looped_decoder(SimpleNamespace(model_name_or_path=folder, model_revision=None))
    assert never not in str(err.value) + str(err.value.info.remediation)
