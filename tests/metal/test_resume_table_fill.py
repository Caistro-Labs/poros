"""On a Mac the table fills what a config leaves unset (gradient_accumulation_steps, max_seq_length,
save_steps, the store), and the run trains with, and its checkpoints record, those values. The
resume check before the load compares a checkpoint against them, so a run that left
gradient_accumulation_steps to the table (1) resumes rather than being held to the config
default (8).
CPU-runnable: the Mac is the measured M4 class, named; nothing loads."""

from __future__ import annotations

import json
import sys

import pytest

from poros.cli import train_cmd
from poros.core.config import PorosConfig
from poros.metal import table
from poros.preflight import command
from poros.preflight.resume import resume_clash

MODEL = "Qwen/Qwen2.5-0.5B"
CLASS = "apple-m4-10gpu-16gb"


@pytest.fixture
def a_mac(monkeypatch):
    from poros import metal
    from poros.cli.ui import simulate

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(metal, "device_class", lambda: CLASS)
    monkeypatch.setattr(simulate, "enabled", lambda: False)


def _classic(tmp_path, **kw) -> PorosConfig:
    """`poros train <model> --data synthetic` as the train verb builds it: the length set, the
    accumulation left to the table."""
    return PorosConfig(model_name_or_path=MODEL, dataset="synthetic", max_steps=12,
                       max_seq_length=256, lora_rank=16, lora_alpha=32,
                       output_dir=str(tmp_path / "run"), **kw)


def _checkpoint(tmp_path, **said) -> str:
    d = tmp_path / "run" / "checkpoint-10"
    d.mkdir(parents=True)
    (d / "poros_config.json").write_text(json.dumps({
        "model_name_or_path": MODEL, "quantization": "nf4", "lora_rank": 16, "seed": 42,
        "per_device_batch_size": 1, "dataset": "synthetic", "max_seq_length": 256, **said}))
    return str(d)


def test_the_table_sets_every_compared_field_it_fills():
    """resume_clash compares accumulation and length, and the table fills both: once filled, no
    other field the table sets is one the check reads (batch and seed stay the config's)."""
    row = table.pick(MODEL, CLASS)
    filled = set(table.defaults(row))
    compared = {"quantization", "lora_rank", "lora_target_modules", "model_revision",
                "gradient_accumulation_steps", "per_device_batch_size", "seed", "dataset",
                "max_seq_length"}
    assert filled & compared == {"gradient_accumulation_steps", "max_seq_length"}
    assert row.grad_accum == 1 != PorosConfig.model_fields["gradient_accumulation_steps"].default


def test_a_classic_run_left_to_the_table_resumes(a_mac, tmp_path):
    ckpt = _checkpoint(tmp_path, gradient_accumulation_steps=1)
    cfg = _classic(tmp_path)
    assert "gradient_accumulation_steps" in resume_clash(ckpt, cfg)  # what the refusal was
    train_cmd._fill_as_this_mac(cfg)
    assert cfg.gradient_accumulation_steps == 1
    assert command.resume_refusal(ckpt, cfg) is None


def test_a_checkpoint_trained_at_another_accumulation_is_still_refused(a_mac, tmp_path):
    ckpt = _checkpoint(tmp_path, gradient_accumulation_steps=4)
    cfg = _classic(tmp_path)
    train_cmd._fill_as_this_mac(cfg)
    refused = command.resume_refusal(ckpt, cfg)
    assert refused is not None and "gradient_accumulation_steps 4" in str(refused.reason)


def test_one_set_by_the_user_stays_set(a_mac, tmp_path):
    """The table fills only what the config left unset: an accumulation you set is compared as
    set, and refused against a checkpoint that trained at the table's."""
    ckpt = _checkpoint(tmp_path, gradient_accumulation_steps=1)
    cfg = _classic(tmp_path, gradient_accumulation_steps=8)
    train_cmd._fill_as_this_mac(cfg)
    assert cfg.gradient_accumulation_steps == 8
    assert command.resume_refusal(ckpt, cfg) is not None


def test_off_a_mac_and_in_the_simulator_nothing_is_filled(monkeypatch, tmp_path):
    from poros.cli.ui import simulate

    cfg = _classic(tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    train_cmd._fill_as_this_mac(cfg)
    assert "gradient_accumulation_steps" not in cfg.model_fields_set
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(simulate, "enabled", lambda: True)
    train_cmd._fill_as_this_mac(cfg)
    assert "gradient_accumulation_steps" not in cfg.model_fields_set


def test_the_resume_check_before_the_load_sees_the_filled_values(a_mac, tmp_path, monkeypatch):
    """The order in `poros train`: the table's fill comes before the resume check."""
    ckpt = _checkpoint(tmp_path, gradient_accumulation_steps=1)
    seen = {}

    class Checked(Exception):
        pass

    def resume_refusal(resume_from, cfg, progress=None):
        seen["accumulation"] = cfg.gradient_accumulation_steps
        seen["clash"] = resume_clash(resume_from, cfg)
        raise Checked

    monkeypatch.setattr(command, "resume_refusal", resume_refusal)
    with pytest.raises(Checked):
        train_cmd._before_the_load(
            _classic(tmp_path), resume_from=ckpt, trust_remote_code=False,
            friendly_error=lambda exc: pytest.fail(str(exc)), plan_gate=None, submit=None,
            job_id=None, unvalidated=False)
    assert seen == {"accumulation": 1, "clash": None}
