"""`poros arch try` without torch refuses (E_NO_TORCH, exit 5, the pip line) instead of exiting 0
with tried: null and itself as the next command; `poros arch` without torch offers no `try` it
would refuse; and every shown command keeps its `poros`."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.conftest import checkout_env

NO_TORCH = "import sys; sys.modules['torch'] = None; from poros.cli.main import cli; cli()"


@pytest.fixture
def model(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "model_type": "phi3", "architectures": ["Phi3ForCausalLM"], "hidden_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
        "intermediate_size": 512, "vocab_size": 1000}))
    return folder


def _poros(tmp_path, *args):
    env = checkout_env({**os.environ, "HOME": str(tmp_path), "POROS_HOME": str(tmp_path / "ph"),
                        "HF_HUB_OFFLINE": "1"})
    return subprocess.run([sys.executable, "-P", "-c", NO_TORCH, *args], capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, cwd=tmp_path, env=env)


def test_arch_try_without_torch_refuses_with_the_pip_line(tmp_path, model):
    proc = _poros(tmp_path, "arch", "try", str(model), "--json")
    error = json.loads(proc.stdout)["error"]
    assert proc.returncode == 5 == error["exit_code"] and error["code"] == "E_NO_TORCH"
    assert error["hint"] == "pip install poros-train"


def test_arch_without_torch_offers_no_try_it_would_refuse(tmp_path, model):
    proc = _poros(tmp_path, "arch", str(model), "--json")
    out = json.loads(proc.stdout)
    assert proc.returncode == 0 and out["next"] == [] and "pip install poros-train" in out[
        "cannot_place"][0]


def test_a_shown_command_keeps_its_poros(monkeypatch):
    from poros.cli import registry
    from poros.cli.ui import components as c

    monkeypatch.setattr(c, "in_session", False)
    for name in registry.names():
        assert name in c._SUBCOMMANDS, name
    assert c.command("arch try ./m") == "poros arch try ./m"
    assert c.command("docs cli") == "poros docs cli"
