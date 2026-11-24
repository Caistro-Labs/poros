"""adapter_config.json is the same bytes in every process: PEFT lists target_modules (a set) in
the order a string's hash gives, which PYTHONHASHSEED changes from run to run; Poros's save lists
every set-valued field sorted (``_sorted_config_sets``), and PEFT reads it back as the same set."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAVE = r"""
import sys, torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from poros.train.checkpointing import _sorted_config_sets
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"))
model.save_pretrained(sys.argv[1])
if sys.argv[2] == "sorted":
    _sorted_config_sets(model, __import__("pathlib").Path(sys.argv[1]))
"""


def _save(tmp_path, seed, how):
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    out = tmp_path / f"{how}-{seed}"
    # named variables only, never a copy of this process's environment: nothing the shell holds
    # can reach a failure's output
    kept = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}
    done = subprocess.run([sys.executable, "-c", SAVE, str(out), how], capture_output=True,
                          text=True, env={**kept, "PYTHONHASHSEED": str(seed),
                                          "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"})
    assert done.returncode == 0, f"the save failed (exit {done.returncode}): {done.stderr[-2000:]}"
    return (out / "adapter_config.json").read_bytes()


def test_two_saves_under_other_hash_seeds_write_the_same_adapter_config(tmp_path):
    import json

    # a light install has no peft: skip here, before the name is needed
    LoraConfig = pytest.importorskip("peft").LoraConfig

    written = {_save(tmp_path, seed, "sorted") for seed in (1, 2, 3, 4)}
    assert len(written) == 1
    said = json.loads(written.pop())
    assert said["target_modules"] == sorted(said["target_modules"])
    assert LoraConfig(**{k: v for k, v in said.items() if k != "peft_type"}).target_modules == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    # PEFT's own file differs between hash seeds: what this test guards
    raw = {_save(tmp_path, seed, "raw") for seed in (1, 2, 3, 4)}
    assert len(raw) > 1
