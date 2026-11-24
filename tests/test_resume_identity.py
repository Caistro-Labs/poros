"""A resume compares what a run read, not where it is: a run moved to another machine names its
dataset and its base by other paths. Each checkpoint records what its run trained on, taken when
the run read it (inputs.json: a file and a base folder by content); an older checkpoint with no
record is compared with its own copy if that is still here, and goes on with a warning where
nothing is left to compare. Another file or another model is still refused, whichever way the
run was started, and what has no contents to compare (rows in memory) is said, once.
The rules are torch-free; the last tests train."""

from __future__ import annotations

import json
import logging
import os
import shutil

import pytest

from poros.core.config import PorosConfig
from poros.preflight import resume
from poros.preflight.command import resume_refusal
from poros.preflight.resume import (
    INPUTS_RECORD,
    adopt_revision,
    base_identity,
    dataset_identity,
    inputs_record,
    resume_clash,
    resume_unverified,
    trained_on,
    warn_unverified,
)

ELSEWHERE = "/workspace/run/data/train.jsonl"  # where the run's first machine kept its data
SHA = "a" * 40
SAVED = {"model_name_or_path": "Qwen/Qwen2.5-72B", "quantization": "nf4", "lora_rank": 16,
         "lora_target_modules": None, "model_revision": SHA, "gradient_accumulation_steps": 8,
         "per_device_batch_size": 1, "seed": 42, "max_seq_length": 2048, "dataset": ELSEWHERE,
         "resolved_base_model_path": f"/hf/hub/models--Qwen--Qwen2.5-72B/snapshots/{SHA}"}


@pytest.fixture(autouse=True)
def _each_test_warns_anew():
    resume._warned.clear()
    yield
    resume._warned.clear()


def _rows(path, text='{"text": "one"}\n{"text": "two"}\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _base(folder, weights=b"w" * 4096, config='{"model_type": "qwen2"}'):
    """A base model's folder: config.json and two shards an index names."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(config)
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for name in names:
        (folder / name).write_bytes(weights)
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": names[0], "b": names[1]}}))
    return folder


def _later(*files):
    """A rewrite's time, where a file system's clock is too coarse to show one made at once."""
    for f in files:
        seen = f.stat()
        os.utime(f, ns=(seen.st_atime_ns, seen.st_mtime_ns + 10**9))


def _ckpt(tmp_path, inputs=None, **over):
    d = tmp_path / "run" / "checkpoint-4"
    d.mkdir(parents=True)
    (d / "poros_config.json").write_text(json.dumps({**SAVED, **over}))
    if inputs is not None:
        (d / INPUTS_RECORD).write_text(json.dumps(inputs))
    return str(d)


def _cfg(**kw):
    return PorosConfig(**{"model_name_or_path": "Qwen/Qwen2.5-72B", "dataset": "synthetic", **kw})


# ── the dataset ──────────────────────────────────────────────────────────────────────────────


def test_the_same_file_at_another_path_resumes(tmp_path):
    """The refusal a moved 72B run met: the same train.jsonl (same sha256) at another path."""
    here = _rows(tmp_path / "home" / "train.jsonl")
    ckpt = _ckpt(tmp_path, inputs_record(trained_on(here), None))
    cfg = _cfg(dataset=str(here))
    assert resume_clash(ckpt, cfg) is None and resume_unverified(ckpt, cfg) == []
    moved = _rows(tmp_path / "another machine" / "rows.jsonl")  # another name too
    assert resume_clash(ckpt, _cfg(dataset=str(moved))) is None


def test_another_file_at_a_new_path_is_refused_with_the_next_step(tmp_path):
    ckpt = _ckpt(tmp_path, inputs_record(trained_on(_rows(tmp_path / "was.jsonl")), None))
    other = _rows(tmp_path / "home" / "train.jsonl", '{"text": "other rows"}\n')
    clash = resume_clash(ckpt, _cfg(dataset=str(other)))
    assert ELSEWHERE in clash and str(other) in clash and "holds other data" in clash
    assert clash.fix.startswith("set dataset to a copy of the data the checkpoint trained on")
    refusal = resume_refusal(ckpt, _cfg(dataset=str(other)))
    assert refusal.code == "E_CONFIG" and clash.fix in refusal.context


def test_the_same_path_holding_other_rows_is_refused(tmp_path):
    """The path proves nothing: on another machine it can hold another file."""
    rows = _rows(tmp_path / "train.jsonl")
    ckpt = _ckpt(tmp_path, inputs_record(trained_on(rows), None), dataset=str(rows))
    assert resume_clash(ckpt, _cfg(dataset=str(rows))) is None
    rows.write_text('{"text": "one"}\n{"text": "two"}\n{"text": "a row added"}\n')
    assert "holds other data" in resume_clash(ckpt, _cfg(dataset=str(rows)))
    rows.unlink()  # the name alone, with no file behind it, is not the file it recorded
    assert "is no file or folder here" in resume_clash(ckpt, _cfg(dataset=str(rows)))


def test_a_recorded_file_is_never_swapped_for_a_hub_dataset(tmp_path):
    ckpt = _ckpt(tmp_path, inputs_record(trained_on(_rows(tmp_path / "was.jsonl")), None))
    clash = resume_clash(ckpt, _cfg(dataset="OpenAssistant/oasst1"))
    assert "is no file or folder here" in clash and clash.fix.startswith("set dataset")


def test_a_folder_of_data_is_its_files_contents(tmp_path):
    for folder in ("a", "b"):
        _rows(tmp_path / folder / "train-0.jsonl")
        _rows(tmp_path / folder / "train-1.jsonl", '{"text": "three"}\n')
    assert dataset_identity(tmp_path / "a") == dataset_identity(tmp_path / "b")
    _rows(tmp_path / "b" / "train-1.jsonl", '{"text": "four"}\n')
    assert dataset_identity(tmp_path / "a") != dataset_identity(tmp_path / "b")
    for none in ("synthetic", "OpenAssistant/oasst1", str(tmp_path / "gone.jsonl"), None, [{}]):
        assert dataset_identity(none) is None


def test_a_file_is_read_once_while_it_stays_the_same_and_says_how_far_it_is(tmp_path,
                                                                             monkeypatch):
    """A multi-GB file costs one read (about 2 GB a second), not one per check or per save, and
    a caller that shows the wait is told after each piece (the resume check before the load)."""
    monkeypatch.setattr(resume, "_READ", 16)
    rows = _rows(tmp_path / "train.jsonl")  # 32 bytes: two pieces
    told = []
    first = dataset_identity(rows, lambda done, size: told.append((done, size)))
    assert told == [(16, 32), (32, 32)]
    assert first["sha256"] == __import__("hashlib").sha256(rows.read_bytes()).hexdigest()
    assert dataset_identity(rows, lambda *a: told.append(a)) == first and len(told) == 2
    rows.write_text('{"text": "edited, and longer"}\n')
    ckpt = _ckpt(tmp_path, inputs_record(first, None), dataset=str(rows))
    told.clear()
    refusal = resume_refusal(ckpt, _cfg(dataset=str(rows)), progress=lambda *a: told.append(a))
    assert refusal is not None and told and told[-1][0] == told[-1][1] == rows.stat().st_size
    assert dataset_identity(rows) != first


def test_what_a_run_trains_on_is_a_file_a_name_or_rows_in_memory(tmp_path):
    rows = _rows(tmp_path / "train.jsonl")
    assert trained_on(str(rows)) == dataset_identity(rows) and "sha256" in trained_on(rows)
    assert trained_on("OpenAssistant/oasst1") == {"name": "OpenAssistant/oasst1"}
    assert trained_on("synthetic") == {"name": "synthetic"}
    assert trained_on([{}, {}, {}]) == {"in_memory": "list", "length": 3}
    assert trained_on(iter(())) == {"in_memory": type(iter(())).__name__, "length": None}
    assert trained_on(None) is None


@pytest.mark.parametrize("kept, here, said", [
    # rows in memory have no contents on record: another length is refused, anything else said
    ({"in_memory": "DataLoader", "length": 32}, {"in_memory": "DataLoader", "length": 32}, None),
    ({"in_memory": "DataLoader", "length": 32}, {"in_memory": "DataLoader", "length": 8},
     "trained on rows handed over in memory (a DataLoader of 32), and this run is handed rows "
     "handed over in memory (a DataLoader of 8)"),
    ({"in_memory": "Dataset", "length": None}, {"in_memory": "Dataset", "length": 8}, None),
    ({"in_memory": "DataLoader", "length": 32}, {"name": "synthetic"}, None),
    ({"name": "synthetic"}, {"in_memory": "DataLoader", "length": 32}, None),
    (None, {"in_memory": "DataLoader", "length": 32}, None),
    # a run that does not know its data (a loop of the caller's own), a checkpoint that kept none
    ({"name": "synthetic"}, None, None),
    (None, None, None),
    (None, {"name": "synthetic"}, None),
])
def test_data_with_nothing_to_compare_is_said_and_never_passed_in_silence(tmp_path, kept, here,
                                                                          said):
    inputs = {"version": 1, **({"dataset": kept} if kept else {})}
    ckpt = _ckpt(tmp_path, inputs, dataset=(kept or {}).get("name"))
    cfg = _cfg(dataset=(here or {}).get("name"))
    clash, notes = resume.resume_verdict(ckpt, cfg, dataset=here)
    if said:
        assert said in clash and clash.fix.startswith("hand the run the rows") and notes == []
    else:
        assert clash is None and len(notes) == 1 and "Data that could not be checked" in notes[0]


def test_an_older_checkpoints_moved_data_goes_on_with_one_warning_naming_both(tmp_path, caplog):
    """No record, and the file it names is on another machine: nothing can be compared, so the
    run goes on, told once what could not be checked."""
    here = _rows(tmp_path / "home" / "train.jsonl")
    ckpt, cfg = _ckpt(tmp_path, model_name_or_path="Q/M"), _cfg(dataset=str(here),
                                                                 model_name_or_path="Q/M")
    assert resume_clash(ckpt, cfg) is None
    (said,) = resume_unverified(ckpt, cfg)
    assert ELSEWHERE in said and str(here) in said and "could not be checked" in said
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        assert resume_refusal(ckpt, cfg) is None  # before the load
        warn_unverified(ckpt, cfg)  # and the trainer after it
    assert [r.getMessage() for r in caplog.records] == [said]


def test_an_older_checkpoints_data_still_here_is_compared(tmp_path):
    was = _rows(tmp_path / "old" / "train.jsonl")
    ckpt = _ckpt(tmp_path, dataset=str(was), model_name_or_path="Q/M")
    same = _rows(tmp_path / "new" / "train.jsonl")
    cfg = _cfg(dataset=str(same), model_name_or_path="Q/M")
    assert resume_clash(ckpt, cfg) is None and resume_unverified(ckpt, cfg) == []
    other = _rows(tmp_path / "new" / "other.jsonl", '{"text": "other rows"}\n')
    clash = resume_clash(ckpt, _cfg(dataset=str(other), model_name_or_path="Q/M"))
    assert str(was) in clash and str(other) in clash and "holds other data" in clash


@pytest.mark.parametrize("was, now", [
    ("OpenAssistant/oasst1", "train.jsonl"),  # a Hub dataset is no file that moved
    ("synthetic", "train.jsonl"),
    (ELSEWHERE, "OpenAssistant/oasst1"),
    ("OpenAssistant/oasst1", "tatsu-lab/alpaca"),
])
def test_an_older_checkpoints_other_dataset_is_still_refused(tmp_path, monkeypatch, was, now):
    _rows(tmp_path / "train.jsonl")
    monkeypatch.chdir(tmp_path)
    ckpt = _ckpt(tmp_path, dataset=was, model_name_or_path="Q/M")
    cfg = _cfg(dataset=now, model_name_or_path="Q/M")
    clash = resume_clash(ckpt, cfg)
    assert f"dataset {was!r}" in clash and repr(now) in clash
    assert clash.fix.startswith("set dataset to the one the checkpoint trained on")
    assert resume_unverified(ckpt, cfg) == []


# ── the base model ───────────────────────────────────────────────────────────────────────────


def test_a_copy_of_the_base_in_a_local_folder_resumes(tmp_path):
    """The other refusal a moved 72B run met: trained by its Hub id (from the Hub cache), resumed
    from a plain folder holding the same files. No Hub-cache layout is asked of the folder."""
    snapshot = _base(tmp_path / "hub" / "models--Qwen--Qwen2.5-72B" / "snapshots" / SHA)
    ckpt = _ckpt(tmp_path, inputs_record(None, snapshot), dataset="synthetic")
    shutil.rmtree(tmp_path / "hub")  # another machine: the cache it trained from is not here
    copy = _base(tmp_path / "models" / "base72")
    cfg = _cfg(model_name_or_path=str(copy))
    assert resume_clash(ckpt, cfg) is None and resume_unverified(ckpt, cfg) == []
    assert adopt_revision(ckpt, cfg) == SHA  # the same files: the revision it trained on


def test_a_sibling_model_of_the_same_shapes_is_refused_with_the_next_step(tmp_path):
    """Same config.json, same shard names and sizes, other weights: what no name or shape check
    sees (an Instruct or fine-tuned sibling)."""
    ckpt = _ckpt(tmp_path, inputs_record(None, _base(tmp_path / "was")), dataset="synthetic")
    sibling = _base(tmp_path / "models" / "instruct", weights=b"v" * 4096)
    clash = resume_clash(ckpt, _cfg(model_name_or_path=str(sibling)))
    assert "'Qwen/Qwen2.5-72B'" in clash and str(sibling) in clash
    assert "does not hold the same model" in clash
    assert clash.fix.startswith("set model_name_or_path to the base the checkpoint trained on")
    assert clash.fix in resume_refusal(ckpt, _cfg(model_name_or_path=str(sibling))).context
    other_config = _base(tmp_path / "models" / "eos", config='{"model_type": "qwen2", "eos": 2}')
    assert resume_clash(ckpt, _cfg(model_name_or_path=str(other_config))) is not None
    no_model = tmp_path / "models" / "empty"
    no_model.mkdir()
    assert resume_clash(ckpt, _cfg(model_name_or_path=str(no_model))) is not None


def test_the_same_folder_path_holding_another_model_is_refused(tmp_path):
    base = _base(tmp_path / "base")
    ckpt = _ckpt(tmp_path, inputs_record(None, base), dataset="synthetic",
                 model_name_or_path=str(base), resolved_base_model_path=None)
    assert resume_clash(ckpt, _cfg(model_name_or_path=str(base))) is None
    _later(*_base(base, weights=b"v" * 4096).glob("*.safetensors"))
    assert "does not hold the same model" in resume_clash(ckpt, _cfg(model_name_or_path=str(base)))
    for shard in base.glob("*.safetensors"):
        shard.unlink()  # the folder it named, with nothing left to read: its name, as before
    assert resume_clash(ckpt, _cfg(model_name_or_path=str(base))) is None


def test_a_large_shard_is_sampled_not_read_whole(tmp_path):
    """A shard past 16 MiB is named by its size and 256 evenly spaced 64 KiB pieces."""
    size = 64 << 20
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "config.json").write_text("{}")
        with open(tmp_path / folder / "model.safetensors", "wb") as f:
            f.truncate(size)  # sparse: no disk
    assert base_identity(tmp_path / "a") == base_identity(tmp_path / "b")
    assert base_identity(tmp_path / "a")["bytes"] == size
    piece = (size - resume._PIECE) * 100 // (resume._PIECES - 1)  # where the 101st piece starts
    with open(tmp_path / "b" / "model.safetensors", "r+b") as f:
        f.seek(piece + 7)
        f.write(b"x")
    _later(tmp_path / "b" / "model.safetensors")
    assert base_identity(tmp_path / "a") != base_identity(tmp_path / "b")
    with open(tmp_path / "b" / "model.safetensors", "r+b") as f:
        f.truncate(size + 1)  # the same pieces at the start, another size
    assert base_identity(tmp_path / "a") != base_identity(tmp_path / "b")


def test_an_older_checkpoints_base_in_a_folder_goes_on_with_one_warning(tmp_path, caplog):
    """No record of the base's contents and the cache it trained from is on another machine:
    the folder cannot be checked. It is not refused (no folder has to be renamed into a Hub-cache
    layout), and it is not silent; the revision it cannot show is not claimed."""
    folder = _base(tmp_path / "models" / "base72")
    ckpt, cfg = _ckpt(tmp_path, dataset="synthetic"), _cfg(model_name_or_path=str(folder))
    assert resume_clash(ckpt, cfg) is None
    (said,) = resume_unverified(ckpt, cfg)
    assert "'Qwen/Qwen2.5-72B'" in said and str(folder) in said and "could not be checked" in said
    assert adopt_revision(ckpt, cfg) is None and cfg.model_revision is None
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        resume_refusal(ckpt, cfg)
        warn_unverified(ckpt, cfg)
    assert [r.getMessage() for r in caplog.records] == [said]


def test_an_older_checkpoints_base_still_here_is_compared(tmp_path):
    snapshot = _base(tmp_path / "hub" / "models--Qwen--Qwen2.5-72B" / "snapshots" / SHA)
    ckpt = _ckpt(tmp_path, dataset="synthetic", resolved_base_model_path=str(snapshot))
    same = _cfg(model_name_or_path=str(_base(tmp_path / "models" / "base72")))
    assert resume_clash(ckpt, same) is None and resume_unverified(ckpt, same) == []
    sibling = _base(tmp_path / "models" / "instruct", weights=b"v" * 4096)
    clash = resume_clash(ckpt, _cfg(model_name_or_path=str(sibling)))
    assert str(sibling) in clash and "does not hold the same model" in clash


def test_the_caches_snapshot_of_another_revision_is_refused(tmp_path):
    other = _base(tmp_path / "hub" / "models--Qwen--Qwen2.5-72B" / "snapshots" / ("b" * 40))
    clash = resume_clash(_ckpt(tmp_path, dataset="synthetic"), _cfg(model_name_or_path=str(other)))
    assert "at revision aaaaaaaaaaaa" in clash and "revision bbbbbbbbbbbb" in clash


def test_another_hub_id_is_still_refused_before_any_download(tmp_path):
    for inputs in (None, inputs_record(None, _base(tmp_path / "was"))):
        ckpt = _ckpt(tmp_path / str(bool(inputs)), inputs, dataset="synthetic")
        cfg = _cfg(model_name_or_path="Qwen/Qwen2.5-72B-Instruct")
        clash = resume_clash(ckpt, cfg)
        assert "base 'Qwen/Qwen2.5-72B'" in clash and "'Qwen/Qwen2.5-72B-Instruct'" in clash
        assert clash.fix.startswith("set model_name_or_path") and resume_unverified(ckpt, cfg) == []


def test_the_next_step_names_the_base_the_checkpoint_trained_on(tmp_path):
    """A run trained from a local folder and resumed by a Hub id is refused (a Hub id is no
    folder to compare), and the next step names the folder it trained on: it never had a Hub
    id to go back to."""
    trained = "/data/models/base72"
    ckpt = _ckpt(tmp_path, dataset="synthetic", model_name_or_path=trained, model_revision=None,
                 resolved_base_model_path=None)
    clash = resume_clash(ckpt, _cfg())
    assert f"trained on base {trained!r}, and this run's base is 'Qwen/Qwen2.5-72B'" in clash
    assert clash.fix == (
        f"set model_name_or_path to the base the checkpoint trained on ({trained!r}), or to a "
        "folder holding a copy of its files (at any path), or start a new run in another "
        "output_dir")
    by_id = resume_clash(_ckpt(tmp_path / "hub", dataset="synthetic"),
                         _cfg(model_name_or_path="Qwen/Qwen2.5-72B-Instruct"))
    assert "trained on ('Qwen/Qwen2.5-72B')" in by_id.fix


def test_a_record_this_code_cannot_read_is_no_record(tmp_path):
    """Another version's record, or a file that is not one: the older checkpoint's rules."""
    here = _rows(tmp_path / "home" / "train.jsonl")
    for inputs in ({"version": 2, "dataset": {"sha256": "0" * 64}}, ["not", "a", "record"],
                   {"version": 1, "dataset": "not an identity"}):
        ckpt = _ckpt(tmp_path / str(len(str(inputs))), inputs, model_name_or_path="Q/M")
        cfg = _cfg(dataset=str(here), model_name_or_path="Q/M")
        assert resume_clash(ckpt, cfg) is None and len(resume_unverified(ckpt, cfg)) == 1


# ── through the trainer ──────────────────────────────────────────────────────────────────────


def _trainers(torch, tmp_path):
    """A maker of tiny trainers whose ``train(dataset)`` takes any dataset, a file's name too:
    the tokenizer a file would need is left out (every run trains the same in-vocabulary rows),
    so what is tested is what ``train`` records of its dataset and compares on a resume."""
    from tests.test_train_loop_cpu import _loader, _make_trainer, _tiny_peft_qwen2

    def trainer(base="dummy", steps=4, rows=32):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        tr = _make_trainer(torch, model, str(tmp_path / "run"), max_steps=steps)
        tr.config.model_name_or_path = str(base)
        tr.loader = _loader(torch, cfg.vocab_size, n=rows)
        tr._prepare_dataloader = lambda dataset, length, for_eval=False: (
            tr.loader if isinstance(dataset, str) else dataset)
        return tr

    return trainer


def _warned(caplog):
    """What the resume check warned of (a tiny model's trainer warns of other things too)."""
    return [r.getMessage() for r in caplog.records if r.name == "poros.preflight.resume"]


def _resume_marks(tmp_path):
    events = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    return [e["data"] for e in events if e["event_type"] == "resume"]


def test_train_on_a_file_then_resume_on_another_is_refused(tmp_path):
    """trainer.train("a.jsonl") then trainer.train("b.jsonl", resume_from=ckpt), with no dataset
    in the config: it resumed onto the other rows with no word. The dataset train() is given is
    what the checkpoint records and what the resume compares."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from pathlib import Path

    from poros.core.errors import PorosConfigError

    trainer = _trainers(torch, tmp_path)
    a, b = _rows(tmp_path / "a.jsonl"), _rows(tmp_path / "b.jsonl", '{"text": "other rows"}\n')
    first = trainer(steps=2)
    assert first.config.dataset is None
    first.train(str(a), steps=2)
    ckpt = first._save_checkpoint()
    assert json.loads((Path(ckpt) / "poros_config.json").read_text())["dataset"] == str(a)
    assert json.loads((Path(ckpt) / INPUTS_RECORD).read_text())["dataset"] == dataset_identity(a)

    other = trainer()
    before = {n: p.clone() for n, p in other.model.named_parameters() if p.requires_grad}
    with pytest.raises(PorosConfigError, match="holds other data") as refused:
        other.train(str(b), resume_from=ckpt)
    assert str(a) in str(refused.value) and str(b) in str(refused.value)
    assert "Set dataset to a copy of the data the checkpoint trained on" in str(refused.value)
    assert other.global_step == 0 and all(  # refused before anything was restored
        torch.equal(p, before[n]) for n, p in other.model.named_parameters() if p.requires_grad)

    same = trainer()
    same.train(str(a), resume_from=ckpt)  # the same file resumes
    moved = shutil.copy(a, tmp_path / "moved.jsonl")
    trainer().train(str(moved), resume_from=ckpt)  # and so does a copy of it anywhere
    assert _resume_marks(tmp_path) == [{"checkpoint": ckpt}] * 2  # checked: nothing left unsaid
    with pytest.raises(PorosConfigError, match="'synthetic' is no file or folder here"):
        trainer().train("synthetic", resume_from=ckpt)


def test_rows_handed_over_in_memory_are_said_once_and_another_length_is_refused(tmp_path,
                                                                                caplog):
    """A DataLoader has no contents a checkpoint can record (reading it would draw from its
    sampler): the checkpoint says what it was, a resume handed rows of another length is
    refused, and one handed rows of the same length says once that it could not check them."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from pathlib import Path

    from poros.core.errors import PorosConfigError

    trainer = _trainers(torch, tmp_path)
    first = trainer(steps=2)
    first.train(first.loader, steps=2)
    ckpt = first._save_checkpoint()
    assert json.loads((Path(ckpt) / INPUTS_RECORD).read_text())["dataset"] == {
        "in_memory": "DataLoader", "length": 32}

    again = trainer()
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        again.train(again.loader, resume_from=ckpt)
    (said,) = _warned(caplog)
    assert "Data that could not be checked" in said and "a DataLoader of 32" in said
    assert _resume_marks(tmp_path) == [{"checkpoint": ckpt, "unverified": [said]}]

    shorter = trainer(rows=8)
    with pytest.raises(PorosConfigError, match=r"a DataLoader of 32\), and this run is handed"):
        shorter.train(shorter.loader, resume_from=ckpt)
    assert shorter.global_step == 0


def test_the_dataset_is_taken_when_the_run_reads_it_not_at_a_save(tmp_path, monkeypatch):
    """A file edited after the run read it is not what its checkpoints say trained, and no save
    reads the dataset: the first one does not stop to hash it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from pathlib import Path

    from poros.core.errors import PorosConfigError

    trainer = _trainers(torch, tmp_path)
    rows = _rows(tmp_path / "train.jsonl")
    read = dataset_identity(rows)
    first = trainer(steps=2)
    first.train(str(rows), steps=2)
    rows.write_text('{"text": "one"}\n{"text": "two"}\n{"text": "added after the run read it"}\n')
    # a context of its own: monkeypatch.undo() would undo the conftest's too (a Mac's CPU trainer)
    with monkeypatch.context() as m:
        m.setattr(resume, "_file_sha256", lambda *_a: pytest.fail("a save read the data"))
        ckpt = first._save_checkpoint()
    assert json.loads((Path(ckpt) / INPUTS_RECORD).read_text())["dataset"] == read
    with pytest.raises(PorosConfigError, match="holds other data"):
        trainer().train(str(rows), resume_from=ckpt)  # the edited file is not what trained


def test_a_checkpoint_that_recorded_no_dataset_is_resumed_with_a_warning(tmp_path, caplog):
    """A checkpoint train() wrote with ``dataset: null`` and no record: nothing to compare, so
    the resume says so once, and the run's next checkpoint records the file."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from pathlib import Path

    trainer = _trainers(torch, tmp_path)
    rows = _rows(tmp_path / "train.jsonl")
    first = trainer(steps=2)
    first.train(first.loader, steps=2)
    ckpt = Path(first._save_checkpoint())
    (ckpt / INPUTS_RECORD).unlink()  # as it was written
    again = trainer()
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        again.train(str(rows), resume_from=str(ckpt))
    (said,) = _warned(caplog)
    assert "did not record what it trained on" in said and str(rows) in said
    newest = Path(again._save_checkpoint())
    assert json.loads((newest / INPUTS_RECORD).read_text())["dataset"] == dataset_identity(rows)


def test_a_checkpoint_records_its_base_once_and_a_moved_run_resumes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from pathlib import Path

    from poros.core.errors import PorosConfigError

    trainer = _trainers(torch, tmp_path)
    rows, base = _rows(tmp_path / "a" / "train.jsonl"), _base(tmp_path / "a" / "base")
    tr = trainer(base, steps=2)
    tr.train(str(rows), steps=2)
    ckpt = tr._save_checkpoint()
    kept = json.loads((Path(ckpt) / INPUTS_RECORD).read_text())
    assert kept == {"version": 1, "dataset": dataset_identity(rows), "base": base_identity(base)}
    assert json.dumps(kept).count(str(tmp_path)) == 0  # contents only: no path of this machine

    # the base is sampled at the first save and kept: no later save reads it again
    with monkeypatch.context() as m:  # not undo(): that would undo the conftest's too
        m.setattr("poros.train.checkpointing.inputs_record",
                  lambda *_a: pytest.fail("read the base again"))
        assert json.loads((Path(tr._save_checkpoint()) / INPUTS_RECORD).read_text()) == kept

    # another machine: both at other paths, and what the checkpoint names is gone
    moved_rows, moved_base = _rows(tmp_path / "b" / "rows.jsonl"), _base(tmp_path / "b" / "base72")
    shutil.rmtree(tmp_path / "a")
    again = trainer(moved_base)
    again.train(str(moved_rows), resume_from=ckpt)
    assert again.global_step == 3  # the fourth update, counted from 0
    assert _resume_marks(tmp_path) == [{"checkpoint": ckpt}]

    sibling = trainer(_base(tmp_path / "sibling", weights=b"v" * 4096))
    with pytest.raises(PorosConfigError, match="does not hold the same model") as refused:
        sibling.train(str(moved_rows), resume_from=ckpt)
    assert "Set model_name_or_path to the base the checkpoint trained on" in str(refused.value)


def test_an_older_checkpoint_resumes_on_moved_inputs_and_its_next_checkpoint_records_them(
        tmp_path, caplog):
    """A checkpoint written before the record existed, resumed on another machine: it goes on
    with a warning for each input it could not check, and from its next checkpoint on the run is
    checked by content."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from pathlib import Path

    trainer = _trainers(torch, tmp_path)
    tr = trainer(_base(tmp_path / "a" / "base"), steps=2)
    tr.train(str(_rows(tmp_path / "a" / "train.jsonl")), steps=2)
    ckpt = tr._save_checkpoint()
    (Path(ckpt) / INPUTS_RECORD).unlink()  # as an older Poros wrote it
    moved_rows, moved_base = _rows(tmp_path / "b" / "rows.jsonl"), _base(tmp_path / "b" / "base72")
    shutil.rmtree(tmp_path / "a")
    again = trainer(moved_base)
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        again.train(str(moved_rows), resume_from=ckpt)
    warned = [r.getMessage() for r in caplog.records if "could not be checked" in r.getMessage()]
    assert len(warned) == 2 and str(moved_rows) in warned[1] and str(moved_base) in warned[0]
    assert _resume_marks(tmp_path) == [{"checkpoint": ckpt, "unverified": warned}]

    newest = Path(again._save_checkpoint())
    assert json.loads((newest / INPUTS_RECORD).read_text()) == inputs_record(
        trained_on(moved_rows), moved_base)
    resume._warned.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        trainer(moved_base, steps=6).train(str(moved_rows), resume_from=str(newest))
    assert not _warned(caplog)


# ── through poros.Run ────────────────────────────────────────────────────────────────────────


def test_a_run_driven_by_a_loop_of_its_own_says_its_rows_are_unchecked(tmp_path, caplog):
    """poros.Run sees batches, not a dataset: with nothing recorded and nothing to compare, a
    resume says so once (and the resume event keeps it) instead of passing in silence."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from poros.recipes import sft
    from tests.test_run_api import _chats, _run

    run = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0)
    sft.main(run, _chats(), steps=2)
    ckpt = str(tmp_path / "run" / "checkpoint-1")
    assert "dataset" not in json.loads((tmp_path / "run" / "checkpoint-1" / INPUTS_RECORD)
                                       .read_text())
    again = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0)
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        again.load(ckpt)
        again.load(ckpt)
    (said,) = _warned(caplog)
    assert "handed its rows by a loop of its own" in said and ckpt in said
    assert _resume_marks(tmp_path)[0] == {"checkpoint": ckpt, "unverified": [said]}


def test_a_recipes_rows_are_compared_by_the_trainer_and_its_file_is_taken_as_it_starts(
        tmp_path, monkeypatch, caplog):
    """A recipe run holds its rows' record (data.json): the trainer compares it on a resume,
    whoever asked first, and other rows are refused before anything is restored. The file the
    rows came from is taken as the run starts, before step 0, not inside a save."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from poros.core.errors import PorosConfigError
    from poros.preflight.resume import data_record
    from poros.recipes import sft
    from tests.test_run_api import _chats, _run

    rows = _rows(tmp_path / "chats.jsonl")

    def run(data, dataset=rows):
        made = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0,
                    dataset=str(dataset))
        made._engine._data_record = data_record(data)  # as `poros train` hands a recipe's rows
        return made

    data = _chats()
    first = run(data)
    taken = []
    monkeypatch.setattr(first, "forward_backward", lambda *a, **k: (
        taken.append(first._engine._dataset_record), type(first).forward_backward(first, *a, **k)
    )[1])
    sft.main(first, data, steps=2)
    assert taken[0] == dataset_identity(rows)  # before the first step
    ckpt = tmp_path / "run" / "checkpoint-1"
    assert json.loads((ckpt / INPUTS_RECORD).read_text())["dataset"] == dataset_identity(rows)

    with pytest.raises(PorosConfigError, match="trained on 8 rows .* and this run reads 6 rows"):
        run(_chats(n=6)).load(ckpt)
    moved = _rows(tmp_path / "elsewhere" / "chats.jsonl")
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        run(data, moved).load(ckpt)  # the same rows, from a file at another path
        run(data, _rows(tmp_path / "reworded.jsonl", '{"text": "x"}\n')).load(ckpt)  # rows decide
    assert not _warned(caplog)
