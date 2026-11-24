"""Inputs that would train or save something wrong must stop instead."""

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")  # the Mac path needs torch
from safetensors.torch import load_file, save_file  # noqa: E402

from poros import metal  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosConfigError  # noqa: E402
from poros.metal import load, pack  # noqa: E402
from poros.metal.export import merged  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

arms = tiny.arms


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_a_checkpoint_that_is_not_bf16_is_refused_at_the_pack(tmp_path, dtype):
    tiny.write_checkpoint(tmp_path, tiny.tiny_config(vocab_size=256, layers=1))
    tensors = load_file(tmp_path / "model.safetensors")
    tensors["model.embed_tokens.weight"] = tensors["model.embed_tokens.weight"].to(dtype)
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="holds bf16 only"):
        pack.pack(tmp_path, tmp_path / "store", model_id="t", revision="r")


@pytest.mark.parametrize("clip", [1.0, 0.0])
def test_a_non_finite_gradient_stops_before_the_optimizer_steps(clip):
    """On a Mac (a witness) as on CUDA, clipped or not: the one stop reads the norm first."""
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.tensor([float("nan"), 1.0])
    seen = []

    def stop(*, loss=None, grad_norm=None):
        seen.append(grad_norm)
        if not math.isfinite(grad_norm):
            raise RuntimeError("step check")

    run = SimpleNamespace(
        optimizer=torch.optim.SGD([param], lr=1.0),
        config=SimpleNamespace(max_grad_norm=clip), trainable_parameters=[param],
        history=SimpleNamespace(grad_norms=[]), _witness=SimpleNamespace(check=None),
        global_step=7, _stop_if_nonfinite=stop,
    )
    with pytest.raises(RuntimeError, match="step check"):
        PorosTrainer._adapter_optimizer_step(run)
    assert len(seen) == 1 and math.isnan(seen[0])
    assert param.data.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("platform, g, rate", [("darwin", 1, 2e-5), ("linux", 8, 2e-4)])
def test_one_row_a_step_trains_at_a_tenth_of_the_rate(monkeypatch, platform, g, rate):
    import importlib
    import sys

    from poros import recipes
    from poros.recipes import presets

    monkeypatch.setattr(sys, "platform", platform)
    importlib.reload(presets)
    importlib.reload(recipes)
    try:
        assert recipes.PRESETS["sft"]["gradient_accumulation_steps"] == g
        assert recipes.PRESETS["sft"]["learning_rate"] == rate
        assert recipes.LOOP["learning_rate"] == rate
    finally:
        monkeypatch.undo()
        importlib.reload(presets)
        importlib.reload(recipes)


# -- the base a run trained on ------------------------------------------------------------------


def _config(model, root, output_dir):
    return PorosConfig(model_name_or_path=str(model), store="ram", store_path=str(root / "store"),
                       max_seq_length=16, gradient_accumulation_steps=1, max_steps=2,
                       learning_rate=1e-3, warmup_steps=0, save_steps=0, eval_steps=0,
                       output_dir=str(output_dir), progress=False)


def _trained(model, root):
    """A run saved by the Mac path, and its output folder."""
    run_dir = root / "run"
    trainer = arms.CpuDouble.from_pretrained(str(model), config=_config(model, root, run_dir))
    trainer._save_tokenizer = lambda _path: None
    trainer.train(arms.token_stream(trainer.model.config.vocab_size, 16, 2), steps=2)
    trainer.save(str(run_dir))
    return run_dir


def _again(model, root):
    return arms.CpuDouble.from_pretrained(str(model), config=_config(model, root, root / "again"))


def _saved(run_dir):
    return PorosConfig(**json.loads((run_dir / "poros_config.json").read_text()))


@pytest.fixture
def local(tmp_path):
    base = tmp_path / "base"
    tiny.write_checkpoint(base, tiny.tiny_config())
    return base, _trained(base, tmp_path)


def _adapter(trainer):
    return {n: p.clone() for n, p in trainer.model.named_parameters() if p.requires_grad}


def test_a_local_base_records_what_fetch_says(local, tmp_path):
    base, run_dir = local
    assert _saved(run_dir).model_revision == load.fetch(_config(base, tmp_path, run_dir))[2]


def test_a_rewritten_local_base_refuses_a_merge_before_anything_is_written(local, tmp_path):
    base, run_dir = local
    merged(_saved(run_dir), run_dir, tmp_path / "same")  # untouched, it merges
    tiny.write_checkpoint(base, tiny.tiny_config(), seed=2)
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        merged(_saved(run_dir), run_dir, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_rewritten_local_base_refuses_a_resume_before_anything_is_restored(local, tmp_path):
    base, run_dir = local
    _again(base, tmp_path).load_checkpoint(str(run_dir))  # untouched, it resumes
    tiny.write_checkpoint(base, tiny.tiny_config(), seed=2)
    other = _again(base, tmp_path)
    before = _adapter(other)
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        other.load_checkpoint(str(run_dir))
    assert all(torch.equal(p, before[n]) for n, p in _adapter(other).items())
    # Chaining onto another base is a new run: only a full resume claims the same one.
    other.load_checkpoint(str(run_dir), load_optimizer=False, load_rng=False)


def _copied(base, to):
    """``base`` copied as cp copies it: the same bytes, at another path, with other times."""
    import os
    import shutil

    shutil.copytree(base, to, copy_function=shutil.copy)
    for shard in to.glob("*.safetensors"):
        seen = shard.stat()
        os.utime(shard, ns=(seen.st_atime_ns, seen.st_mtime_ns + 10**9))
    return to


def test_a_copied_local_base_resumes_and_merges(local, tmp_path):
    """A local base's revision is its shards' sizes and times, which a copy changes: its
    contents say it is the same model (the run's inputs.json), wherever the copy is."""
    base, run_dir = local
    assert "base" in json.loads((run_dir / "inputs.json").read_text())
    copy = _copied(base, tmp_path / "elsewhere" / "base72")
    moved = _again(copy, tmp_path)
    assert moved.config.model_revision != _saved(run_dir).model_revision  # other times
    moved.load_checkpoint(str(run_dir))
    merged(_saved(run_dir).model_copy(update={"model_name_or_path": str(copy)}), run_dir,
           tmp_path / "merged")
    assert (tmp_path / "merged" / "config.json").exists()
    # the same folder, its files only touched: other times, the same model
    for shard in base.glob("*.safetensors"):
        shard.touch()
    _again(base, tmp_path).load_checkpoint(str(run_dir))
    # a copy rewritten is another model, wherever it is
    tiny.write_checkpoint(copy, tiny.tiny_config(), seed=2)
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        _again(copy, tmp_path).load_checkpoint(str(run_dir))
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        merged(_saved(run_dir).model_copy(update={"model_name_or_path": str(copy)}), run_dir,
               tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_older_runs_copied_local_base_is_compared_or_warned_of_once(local, tmp_path, caplog):
    """A run saved before inputs.json has only the times to go on. Its own base still here is
    compared with the copy; with nothing to compare, the resume goes on with one warning, and
    what it saves next records the base's contents."""
    import logging
    import shutil

    base, run_dir = local
    (run_dir / "inputs.json").unlink()  # as an older Poros saved it
    copy = _copied(base, tmp_path / "elsewhere" / "base72")
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        _again(copy, tmp_path).load_checkpoint(str(run_dir))  # both here: the same bytes
    assert not [r for r in caplog.records if "A base that could not be checked" in r.getMessage()]
    other = tmp_path / "elsewhere" / "other"
    tiny.write_checkpoint(other, tiny.tiny_config(), seed=2)
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        _again(other, tmp_path).load_checkpoint(str(run_dir))  # both here: another model

    shutil.rmtree(base)  # another machine: the base it names is not here
    resumed = _again(copy, tmp_path)
    resumed._save_tokenizer = lambda _path: None
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        resumed.load_checkpoint(str(run_dir))
    warned = [r.getMessage() for r in caplog.records
              if "A base that could not be checked" in r.getMessage()]
    assert len(warned) == 1 and str(base) in warned[0] and str(copy) in warned[0]
    resumed.save(str(tmp_path / "next"))
    assert "base" in json.loads((tmp_path / "next" / "inputs.json").read_text())
    _again(_copied(copy, tmp_path / "third"), tmp_path).load_checkpoint(str(tmp_path / "next"))


def test_an_older_runs_base_changed_where_it_is_stays_refused(local, tmp_path):
    """A run saved before inputs.json, and the folder it names holds other sizes or times: the
    revision is all it kept, and nothing in it tells a rewrite from a touch, so it is refused,
    a rewrite and a touch alike. A run that recorded its base is asked its contents."""
    base, run_dir = local
    (run_dir / "inputs.json").unlink()  # as an older Poros saved it
    _again(base, tmp_path).load_checkpoint(str(run_dir))  # untouched, it resumes
    for shard in base.glob("*.safetensors"):
        seen = shard.stat()
        __import__("os").utime(shard, ns=(seen.st_atime_ns, seen.st_mtime_ns + 10**9))
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        _again(base, tmp_path).load_checkpoint(str(run_dir))
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        merged(_saved(run_dir), run_dir, tmp_path / "out")
    tiny.write_checkpoint(base, tiny.tiny_config(), seed=2)
    with pytest.raises(PorosConfigError, match="trained on base revision"):
        _again(base, tmp_path).load_checkpoint(str(run_dir))
    assert not (tmp_path / "out").exists()


def test_a_hub_revision_that_moved_refuses_a_resume(tmp_path, monkeypatch):
    def snapshot(sha):
        path = tmp_path / "hub" / "snapshots" / sha
        if not path.exists():
            tiny.write_checkpoint(path, tiny.tiny_config())
        return str(path)

    head = ["a" * 40]
    monkeypatch.setattr(load, "snapshot_download",
                        lambda name, revision=None, allow_patterns=None, token=None:
                        snapshot(revision or head[0]))  # Q/M is no catalog model: no token
    monkeypatch.setattr("poros.preflight.arch.cheap_model_type", lambda *_args: "qwen2")
    run_dir = _trained("Q/M", tmp_path)
    assert _saved(run_dir).model_revision == head[0]
    head[0] = "b" * 40
    other = _again("Q/M", tmp_path)
    assert other.config.model_revision == head[0]
    with pytest.raises(PorosConfigError, match=f"trained on base revision {'a' * 12}"):
        other.load_checkpoint(str(run_dir))
    # A run reopened with its own pin finds its own snapshot.
    pinned = arms.CpuDouble.from_pretrained("Q/M", config=_saved(run_dir))
    pinned.load_checkpoint(str(run_dir))


def test_a_finished_pack_trains_with_its_shards_gone_and_never_fetches_them(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub" / ("c" * 40)
    tiny.write_checkpoint(snapshot, tiny.tiny_config())
    asked = []

    def snapshot_download(name, revision=None, allow_patterns=None, token=None):
        assert token is False  # Q/M is no catalog model: anyone's repo, read with no token
        asked.append("*.safetensors" in allow_patterns)
        return str(snapshot)

    monkeypatch.setattr(load, "snapshot_download", snapshot_download)
    monkeypatch.setattr("poros.preflight.arch.cheap_model_type", lambda *_args: "qwen2")
    monkeypatch.setattr(metal, "settle", lambda _: None)
    run_dir = _trained("Q/M", tmp_path)
    assert True in asked  # no pack yet: it takes them

    asked.clear()
    merged(_saved(run_dir), run_dir, tmp_path / "merged")
    assert asked == [True]  # a merge reads every shard

    asked.clear()
    for shard in snapshot.glob("*.safetensors"):
        shard.unlink()
    trainer = _again("Q/M", tmp_path)
    assert asked and True not in asked
    trainer.load_checkpoint(str(run_dir))


def test_off_a_mac_a_resume_does_not_look_at_the_base(local, tmp_path, monkeypatch):
    """Off a Mac a resume never reads the base (metal.same_base hashes it): it compares the
    revision the checkpoint recorded with the one this run names, and a different one is
    refused naming both, before anything is restored."""
    base, run_dir = local
    monkeypatch.setattr(metal, "same_base", lambda *_a: pytest.fail("read the base"))
    trainer = _again(base, tmp_path)
    trainer._metal, trainer.config.model_revision = False, "another"
    before = _adapter(trainer)
    with pytest.raises(PorosConfigError, match="model_revision .*and this run has 'another'"):
        trainer.load_checkpoint(str(run_dir))
    assert all(torch.equal(p, before[n]) for n, p in _adapter(trainer).items())
    trainer.config.model_revision = _saved(run_dir).model_revision
    trainer.load_checkpoint(str(run_dir))
