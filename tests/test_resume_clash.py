"""Resuming a checkpoint into another run is refused before the load, naming both values: the
base and its revision, and the settings that put rows behind each step.
Torch-free."""

from __future__ import annotations

import json

import pytest

from poros.core.config import PorosConfig
from poros.preflight.command import resume_refusal
from poros.preflight.resume import resume_clash

SAVED = {"model_name_or_path": "Qwen/Qwen2.5-0.5B", "quantization": "nf4", "lora_rank": 16,
         "lora_target_modules": None, "model_revision": "a" * 40,
         "gradient_accumulation_steps": 8, "per_device_batch_size": 1, "seed": 42,
         "resolved_base_model_path": "/hf/hub/models--Qwen--Qwen2.5-0.5B/snapshots/" + "a" * 40}


def _ckpt(tmp_path, **over):
    d = tmp_path / "checkpoint-4"
    d.mkdir()
    (d / "poros_config.json").write_text(json.dumps({**SAVED, **over}))
    return str(d)


def _cfg(**kw):
    return PorosConfig(**{"model_name_or_path": "Qwen/Qwen2.5-0.5B", "dataset": "synthetic", **kw})


def test_the_same_run_resumes(tmp_path):
    assert resume_clash(_ckpt(tmp_path), _cfg()) is None


@pytest.mark.parametrize("kw, name", [
    ({"gradient_accumulation_steps": 4}, "gradient_accumulation_steps 8"),
    ({"per_device_batch_size": 2}, "per_device_batch_size 1"),
    ({"seed": 7}, "seed 42"),
    ({"model_revision": "b" * 40}, "model_revision 'aaaa"),
    ({"model_name_or_path": "Qwen/Qwen2.5-7B"}, "base 'Qwen/Qwen2.5-0.5B'"),
])
def test_another_run_is_refused_naming_both(tmp_path, kw, name):
    clash = resume_clash(_ckpt(tmp_path), _cfg(**kw))
    assert clash and name in clash and repr(next(iter(kw.values()))) in clash
    refusal = resume_refusal(str(tmp_path / "checkpoint-4"), _cfg(**kw))
    assert refusal is not None and refusal.code == "E_CONFIG"


def test_a_hub_id_and_its_snapshot_are_one_base(tmp_path):
    snap = tmp_path / "hub" / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    ck = _ckpt(tmp_path, model_name_or_path=str(snap), resolved_base_model_path=str(snap))
    assert resume_clash(ck, _cfg()) is None  # trained from the folder, resumed by its id
    assert "base" in resume_clash(ck, _cfg(model_name_or_path="Qwen/Qwen3-0.6B"))


def test_a_checkpoint_that_does_not_say_resumes(tmp_path):
    ck = _ckpt(tmp_path, seed=None, gradient_accumulation_steps=None, model_revision=None)
    assert resume_clash(ck, _cfg(seed=7, gradient_accumulation_steps=2)) is None


@pytest.mark.parametrize("over, kw, name", [
    ({"dataset": "synthetic"}, {"dataset": "OpenAssistant/oasst1"}, "dataset 'synthetic'"),
    ({"dataset": "synthetic", "max_seq_length": 512}, {"max_seq_length": 1024},
     "max_seq_length 512"),
])
def test_another_dataset_or_length_is_refused(tmp_path, over, kw, name):
    """The classic loop's resume skips step x accumulation micro-batches
    of its stream; another dataset or max_seq_length makes them, and the next ones, other rows."""
    said = resume_clash(_ckpt(tmp_path, **over), _cfg(**kw))
    assert said is not None and name in said


def test_a_local_dataset_named_two_ways_is_one(tmp_path, monkeypatch):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    ckpt = _ckpt(tmp_path, dataset=str(rows), max_seq_length=2048)
    assert resume_clash(ckpt, _cfg(dataset="rows.jsonl")) is None


def test_a_resume_with_no_revision_takes_its_checkpoints_before_the_download(tmp_path):
    from poros.preflight.resume import adopt_revision

    cfg = _cfg()
    assert cfg.model_revision is None
    assert adopt_revision(_ckpt(tmp_path, dataset="synthetic"), cfg) == "a" * 40
    assert cfg.model_revision == "a" * 40
    pinned = _cfg(model_revision="c" * 40)
    assert adopt_revision(str(tmp_path / "checkpoint-4"), pinned) is None  # said: kept, refused
    assert pinned.model_revision == "c" * 40
    other = _cfg(model_name_or_path="Qwen/Qwen2.5-7B")
    assert adopt_revision(str(tmp_path / "checkpoint-4"), other) is None
