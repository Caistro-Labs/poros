"""`poros export --format adapter` of a run: an adapter gets passed around, so no file or line it
writes carries a base that is a folder on this machine; and a native adapter (a CUDA MoE run's
adapter_weights.pt) is never given a PEFT or vLLM line that cannot load it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from poros import runs


def _run(home: Path, base: str, files: dict) -> str:
    d = runs.create("1118-1300-abcd", {"model": base, "data": "synthetic", "steps": 1,
                                        "seq_len": 8, "rank": 4, "command": "poros train ..."})
    runs.update(d, state="succeeded")
    for name, body in files.items():
        (d / name).write_text(body)
    return d.name


def _export(home: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "POROS_HOME": str(home), "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, "-m", "poros.cli.main", "export", "1118-1300-abcd",
                           "--format", "adapter", "--output", str(out), "--yes", *extra],
                          env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def test_a_local_base_folder_leaves_no_path_in_the_export(tmp_path, monkeypatch):
    base = tmp_path / "private-models" / "my-base"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    _run(tmp_path / "home", str(base), {
        "adapter_config.json": json.dumps({"base_model_name_or_path": str(base), "r": 4}),
        "adapter_model.safetensors": "w",
        "poros_config.json": json.dumps({"model_name_or_path": str(base), "lora_rank": 4,
                                         "dataset": str(tmp_path / "rows.jsonl")})})
    out = tmp_path / "shared"
    for extra in ((), ("--json",)):
        proc = _export(tmp_path / "home", out, *extra)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert str(base) not in proc.stdout + proc.stderr and "private-models" not in proc.stdout
    for f in out.iterdir():
        assert "private-models" not in f.read_text(errors="replace"), f.name
    said = json.loads(_export(tmp_path / "home", out, "--json").stdout)
    assert said["base_model"] == "<the base model's Hub id>" and said["load"] is None
    assert "set base_model_name_or_path in adapter_config.json" in said["note"]
    assert json.loads((out / "adapter_config.json").read_text())["base_model_name_or_path"] == \
        "<the base model's Hub id>"
    assert json.loads((out / "poros_config.json").read_text())["model_name_or_path"] == \
        "<the base model's Hub id>"


def test_a_native_adapter_gets_no_line_that_cannot_load_it(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    _run(tmp_path / "home", "Qwen/Qwen3-30B-A3B", {
        "adapter_weights.pt": "w",
        "poros_config.json": json.dumps({"model_name_or_path": "Qwen/Qwen3-30B-A3B",
                                         "lora_rank": 8})})
    out = tmp_path / "moe"
    said = json.loads(_export(tmp_path / "home", out, "--json").stdout)
    assert said["load"] is None and (out / "adapter_weights.pt").is_file()
    assert "loads with Poros" in said["note"] and "--format merged" in said["note"]
    # what opens an adapter-only folder: a new run started from it (Run.open needs a run's own)
    assert f"init_adapter={str(out)!r}" in said["note"] and "Run.open" not in said["note"]
    plain = _export(tmp_path / "home", out, "--plain").stdout
    assert "vllm serve" not in plain and "PeftModel" not in plain
    assert "--format merged gives a model any" in " ".join(plain.split())
