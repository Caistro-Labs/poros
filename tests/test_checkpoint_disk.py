"""The checkpoints a run keeps against its output disk (poros.preflight.disk.checkpoint_note): a
warning before the download, never a refusal. Torch-free."""

from __future__ import annotations

from collections import namedtuple

import pytest

from poros.core.config import PorosConfig
from poros.preflight import disk

DENSE = {"model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28,
         "num_attention_heads": 28, "num_key_value_heads": 4, "intermediate_size": 18944,
         "vocab_size": 152064}
MOE = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
       "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
       "intermediate_size": 6144, "moe_intermediate_size": 768, "num_experts": 128,
       "num_experts_per_tok": 8, "vocab_size": 151936}
usage = namedtuple("usage", "total used free")


def _cfg(tmp_path, **kw):
    return PorosConfig(**{"model_name_or_path": "m", "dataset": "synthetic",
                          "output_dir": str(tmp_path / "out"), **kw})


def test_a_checkpoint_is_the_adapter_and_its_two_moments(tmp_path):
    from poros.plan.estimate import _sizes

    dense = disk.checkpoint_bytes(_cfg(tmp_path), DENSE)
    assert dense == 12 * 16 * _sizes(DENSE).per_rank  # fp32 weights and two fp32 moments
    moe = disk.checkpoint_bytes(_cfg(tmp_path), MOE)
    experts = 48 * 128 * 16 * (2 * 2048 + 3 * 768)  # poros.moe.adapters' shapes, bf16
    # weights, two moments and the remainder (ExpertAdamW), 2 bytes each
    assert 8 * experts <= moe < 8 * experts * 1.1
    assert disk.checkpoint_bytes(_cfg(tmp_path), None) is None


@pytest.mark.parametrize("limit, keep", [(3, 3), (1, 1)])
def test_too_little_disk_for_what_the_run_keeps_warns(tmp_path, monkeypatch, limit, keep):
    cfg = _cfg(tmp_path, save_steps=100, save_total_limit=limit)
    one, adapter = disk.checkpoint_parts(cfg, MOE)
    assert adapter * 4 == pytest.approx(one, rel=0.1)  # the weights: a quarter of a MoE's
    need = (keep + 1) * one + adapter
    monkeypatch.setattr("shutil.disk_usage", lambda p: usage(10**13, 0, int(need) - 1))
    said = disk.checkpoint_note(cfg, MOE)
    assert said and f"{keep} of about {one / 1e9:.1f} GB each" in said
    assert "save_total_limit=1" in said and str(tmp_path / "out") in said
    monkeypatch.setattr("shutil.disk_usage", lambda p: usage(10**13, 0, int(need) + 1))
    assert disk.checkpoint_note(cfg, MOE) is None


def test_no_save_steps_keeps_the_last_one_and_no_limit_keeps_every_save(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda p: usage(10**13, 0, 1))
    assert "1 of about" in disk.checkpoint_note(_cfg(tmp_path, save_steps=0), DENSE)
    every = _cfg(tmp_path, save_steps=10, save_total_limit=0, max_steps=95)
    assert "10 of about" in disk.checkpoint_note(every, DENSE)
