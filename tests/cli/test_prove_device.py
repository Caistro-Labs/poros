"""`poros prove` runs train's first check: without torch it refuses with E_NO_TORCH, and without
a GPU with E_NO_CUDA (exit 5), as train does, before any question or download."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.conftest import checkout_env

# no torch, on a machine whose GPU nvidia-smi shows (a refusal without torch looks for a GPU
# first, so these say which machine they are on) ...
NO_TORCH = ("import sys; sys.modules['torch'] = None\n"
            "import poros.core.card as card\n"
            "from poros.core.schemas import PlanGpu\n"
            "card.visible_gpus = lambda: [PlanGpu(name='NVIDIA GeForce RTX 4090', total_gib=23.5,"
            " free_gib=23.0, source='nvidia-smi')]\n"
            "from poros.cli.main import cli; cli()")
# ... and on one where it shows none
NO_TORCH_NO_GPU = NO_TORCH.replace("[PlanGpu(name='NVIDIA GeForce RTX 4090', total_gib=23.5,"
                                   " free_gib=23.0, source='nvidia-smi')]", "[]")


def test_prove_without_torch_refuses_as_train_does(tmp_path):
    env = checkout_env({**os.environ, "HOME": str(tmp_path), "POROS_HOME": str(tmp_path / "ph"),
                        "HF_HUB_OFFLINE": "1"})
    for var in ("POROS_DEV_SIMULATE", "POROS_AGENT"):
        env.pop(var, None)
    proc = subprocess.run([sys.executable, "-P", "-c", NO_TORCH, "prove", "qwen2.5-7b", "--json"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path,
                          env=env)
    error = json.loads(proc.stdout)["error"]
    assert proc.returncode == 5 and error["code"] == "E_NO_TORCH"
    assert error["hint"] == "pip install poros-train" and "Traceback" not in proc.stderr


def test_prove_without_a_gpu_refuses_e_no_cuda_before_any_question(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from poros.cli.app import _device_first
    from poros.cli.context import Context
    from poros.cli.words.grammar import parse
    from poros.core.refusal import Refusal
    from poros.preflight import command

    monkeypatch.delenv("POROS_DEV_SIMULATE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = Context(cwd=tmp_path, fixed_cached=frozenset(), fixed_interactive=False)
    if ctx.unified:
        pytest.skip("a Mac proves on its own runtime")
    for argv in (["prove", "qwen2.5-7b"], ["prove"], ["train", "qwen2.5-7b", "--data", "x"]):
        with pytest.raises(Refusal) as refused:
            _device_first(argv[0], parse(argv), ctx)
        assert refused.value is command.NO_CUDA and refused.value.code == "E_NO_CUDA"
    _device_first("prove", parse(["prove", "m", "--unvalidated"]), ctx)  # the gates' own


@pytest.mark.parametrize("argv", [["prove"], ["train", "qwen2.5-7b", "--data", "notes.jsonl"],
                                  ["export", "./ad", "-o", "./full"]])
def test_without_torch_nothing_is_asked_first(tmp_path, argv):
    """Without torch the refusal comes first: bare `poros prove` asks no model, train asks no
    download and the folder export asks no overwrite before E_NO_TORCH."""
    (tmp_path / "ad").mkdir()
    (tmp_path / "ad" / "poros_config.json").write_text("{}")
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("")
    (tmp_path / "notes.jsonl").write_text('{"text": "a"}\n')
    env = checkout_env({**os.environ, "HOME": str(tmp_path), "POROS_HOME": str(tmp_path / "ph"),
                        "HF_HUB_OFFLINE": "1"})
    env.pop("POROS_DEV_SIMULATE", None)
    proc = subprocess.run([sys.executable, "-P", "-c", NO_TORCH, *argv, "--json"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path,
                          env=env)
    assert json.loads(proc.stdout)["error"]["code"] == "E_NO_TORCH", proc.stdout


@pytest.mark.skipif(sys.platform == "darwin", reason="every Mac trains: only torch can be missing")
@pytest.mark.parametrize("argv", [["prove", "qwen2.5-7b"],
                                  ["train", "qwen2.5-7b", "--data", "notes.jsonl"]])
def test_without_a_gpu_the_ml_extra_is_not_what_is_missing(tmp_path, argv):
    """On a machine with no GPU the first refusal sent its user to a multi-GB install,
    and only then said there is no GPU. The GPU is looked for without torch first."""
    (tmp_path / "notes.jsonl").write_text('{"text": "a"}\n')
    env = checkout_env({**os.environ, "HOME": str(tmp_path), "POROS_HOME": str(tmp_path / "ph"),
                        "HF_HUB_OFFLINE": "1"})
    for var in ("POROS_DEV_SIMULATE", "POROS_AGENT"):
        env.pop(var, None)
    proc = subprocess.run([sys.executable, "-P", "-c", NO_TORCH_NO_GPU, *argv, "--json"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path,
                          env=env)
    error = json.loads(proc.stdout)["error"]
    assert (proc.returncode, error["code"]) == (5, "E_NO_CUDA"), error
    assert "none was found here" in error["message"]
    assert "pip install poros-train" in error["message"]  # said, for a machine with one
    assert error["hint"] == "poros machine"


def test_the_device_step_looks_for_a_gpu_before_torch(monkeypatch):
    """`poros train <config>`'s device step, and prove's: no GPU is said before torch."""
    import importlib.util

    import poros.core.card as card
    from poros.core.schemas import PlanGpu
    from poros.preflight import command

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)  # no torch
    monkeypatch.setattr(card, "visible_gpus", lambda: [])
    assert command.device_refusal().code == "E_NO_CUDA"
    monkeypatch.setattr(card, "visible_gpus", lambda: None)  # no nvidia-smi either
    assert command.device_refusal().code == "E_NO_CUDA"
    monkeypatch.setattr(card, "visible_gpus", lambda: [PlanGpu(
        name="NVIDIA GeForce RTX 4090", total_gib=23.5, free_gib=23.0, source="nvidia-smi")])
    assert command.device_refusal() is command.NO_TORCH


@pytest.mark.parametrize("data, code", [("nope.jsonl", "E_DATASET_NOT_FOUND"),
                                        ("empty.jsonl", "E_DATASET"),
                                        ("stops.jsonl", "E_DATASET")])
def test_a_data_mistake_is_said_before_the_device(tmp_path, data, code):
    """`--data nope.jsonl` answered "No GPU": the verb form refused the device before it read
    the data, where docs/training.md and the config form check the data first."""
    (tmp_path / "empty.jsonl").write_text("")
    (tmp_path / "stops.jsonl").write_text('{"text": "a fine row"}\nnot a row\n')
    env = checkout_env({**os.environ, "HOME": str(tmp_path), "POROS_HOME": str(tmp_path / "ph"),
                        "HF_HUB_OFFLINE": "1"})
    for var in ("POROS_DEV_SIMULATE", "POROS_AGENT"):
        env.pop(var, None)
    for harness in (NO_TORCH, NO_TORCH_NO_GPU):
        verb = subprocess.run(
            [sys.executable, "-P", "-c", harness, "train", "qwen2.5-7b", "--data", data, "--json"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path, env=env)
        said = json.loads(verb.stdout)
        # a missed file with near names here is asked about, as a question on the data
        asked = said.get("questions") and said["questions"][0]["id"] == "data"
        assert asked or said["error"]["code"] == code, verb.stdout
        cfg = tmp_path / "run.yaml"
        cfg.write_text(f"model_name_or_path: Qwen/Qwen2.5-7B\ndataset: ./{data}\n"
                       f"output_dir: {tmp_path / 'out'}\n")
        config = subprocess.run(
            [sys.executable, "-P", "-c", harness, "train", str(cfg), "--json"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path, env=env)
        assert json.loads(config.stdout)["error"]["code"] == code, config.stdout
