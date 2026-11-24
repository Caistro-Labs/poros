"""The init_adapter check: a new run that starts from an earlier adapter is checked once, by one
function, before the download, and the folder it reads is treated as data from anywhere. What
it says of its shape is checked before it is used, a save that did not finish is no adapter, and
its base is compared by revision too. Torch-free."""

from __future__ import annotations

import json
import logging

import pytest

from poros.core import checkpoints
from poros.core.config import PorosConfig
from poros.preflight import command, resume
from poros.preflight.resume import start_adapter, start_clash


def start_from(cfg, *how):
    return resume.start_from(cfg, *how)

SHA, OTHER = "a" * 40, "b" * 40
SAID = {"model_name_or_path": "Org/Model", "model_revision": SHA, "lora_rank": 8,
        "lora_alpha": 24, "lora_target_modules": ["q_proj", "v_proj"]}


def _adapter(folder, said=SAID, *, complete=True, raw=None, name="poros_config.json"):
    """A checkpoint folder as Poros saves one, with ``said`` (or the text ``raw``) in ``name``."""
    folder.mkdir(parents=True)
    (folder / checkpoints.FORMAT_MARKER).write_text("1")
    (folder / name).write_text(raw if raw is not None else json.dumps(said))
    (folder / "adapter_weights.pt").write_bytes(b"")
    if complete:
        (folder / checkpoints.COMPLETE_MARKER).write_text("4")
    return folder


def _cfg(adapter, **over):
    return PorosConfig(**{"model_name_or_path": "Org/Model", "dataset": "synthetic",
                          "init_adapter": str(adapter), **over})


# ── one check, and what it takes ─────────────────────────────────────────────────────────────


def test_the_one_check_names_the_checkpoint_takes_the_shape_and_compares(tmp_path):
    ckpt = _adapter(tmp_path / "run" / "checkpoint-4")
    cfg = _cfg(tmp_path / "run")  # the run's folder: its newest checkpoint
    start = start_from(cfg)
    assert start.checkpoint == str(ckpt) and start.clash is None
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules) == (8, 24, ["q_proj", "v_proj"])
    assert start.said["lora_alpha"] == 24
    named = _cfg(ckpt, lora_alpha=64, lora_target_modules=["v_proj", "q_proj"])
    assert start_from(named).clash is None and named.lora_alpha == 64  # what it says stands
    kept = _cfg(ckpt, lora_rank=8)
    start_from(kept, ())  # the trainer's later check takes nothing
    assert kept.lora_alpha == 32 and kept.lora_target_modules is None
    assert start_from(PorosConfig(model_name_or_path="Org/Model")) == (None, None, {})
    named = _cfg(tmp_path / "run")
    assert command.start_refusal(None, named) is None and named.init_adapter == str(ckpt)
    assert command.start_refusal(str(ckpt), _cfg("/nowhere")) is None  # a resume reads its own


def test_every_entry_asks_the_one_check(tmp_path, monkeypatch):
    """poros train's preflight here; Run.create and PorosTrainer in tests/train."""
    asked = []
    real = resume.start_from
    monkeypatch.setattr(resume, "start_from", lambda cfg, *a: asked.append(a) or real(cfg, *a))
    assert command.start_refusal(None, _cfg(_adapter(tmp_path / "checkpoint-4"))) is None
    assert asked == [()]
    resume.require_start(_cfg(tmp_path / "checkpoint-4", lora_rank=8))
    assert asked == [(), ((),)]


# ── what the folder says is checked before it is used ────────────────────────────────────────


@pytest.mark.parametrize("field, value, why", [
    ("lora_rank", "8", "no whole number"),
    ("lora_rank", 0, "no whole number"),
    ("lora_rank", -4, "no whole number"),
    ("lora_rank", True, "no whole number"),
    ("lora_rank", 8.5, "no whole number"),
    ("lora_rank", [8], "no whole number"),
    ("lora_alpha", "sixteen", "no whole number"),
    ("lora_alpha", {"a": 1}, "no whole number"),
    ("lora_alpha", 0, "no whole number"),
    ("lora_target_modules", [["q_proj"]], "no module name"),
    ("lora_target_modules", {"q_proj": 1}, "no module name"),
    ("lora_target_modules", ["q_proj", 7], "no module name"),
    ("lora_target_modules", ["../../x; rm -rf"], "no module name"),
    ("lora_target_modules", [], "no module name"),
    ("lora_target_modules", ["q"] * 600, "no module name"),
    ("model_name_or_path", ["Org/Model"], "not text"),
    ("model_revision", 7, "not text"),
])
def test_a_shape_that_is_no_shape_is_refused_naming_the_file(tmp_path, field, value, why):
    ckpt = _adapter(tmp_path / "checkpoint-4", {**SAID, field: value})
    cfg = _cfg(ckpt)
    start = start_from(cfg)
    assert str(ckpt / "poros_config.json") in start.clash and field in start.clash
    assert why in start.clash and start.clash.fix.startswith("name an adapter whose records read")
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules) == (16, 32, None)  # untaken
    refusal = command.start_refusal(None, _cfg(ckpt))
    assert refusal.code == "E_CONFIG" and str(ckpt / "poros_config.json") in refusal.reason
    assert str(ckpt / "poros_config.json") in start_clash(str(ckpt), _cfg(ckpt))  # never a crash


def test_alpha_as_a_whole_float_and_one_pattern_are_taken(tmp_path):
    ckpt = _adapter(tmp_path / "checkpoint-4",
                    {**SAID, "lora_alpha": 16.0, "lora_target_modules": r".*\.(q|v)_proj"})
    cfg = _cfg(ckpt)
    assert start_from(cfg).clash is None
    assert cfg.lora_alpha == 16 and cfg.lora_target_modules == r".*\.(q|v)_proj"


@pytest.mark.parametrize("raw", ["[" * 200_000, '{"lora_rank": ', '["a", "list"]', "\xff\xfe"])
def test_a_file_that_does_not_read_is_a_refusal_not_an_internal_error(tmp_path, raw):
    """Nested past Python's stack, cut short, or no object: the file is named, before any
    download, where json.loads raised RecursionError out of the check."""
    ckpt = _adapter(tmp_path / "checkpoint-4", raw=raw)
    refusal = command.start_refusal(None, _cfg(ckpt))
    assert refusal.code == "E_CONFIG"
    assert f"{ckpt / 'poros_config.json'} does not read as a JSON object" in refusal.reason
    peft = _adapter(tmp_path / "peft", raw=raw, name="adapter_config.json")
    assert f"{peft / 'adapter_config.json'} does not read" in start_from(_cfg(peft)).clash
    assert resume.resume_clash(str(ckpt), _cfg(ckpt)) is None  # a resume: unread, as before


# ── its base, by revision and for a PEFT folder too ──────────────────────────────────────────


def test_another_revision_of_the_base_is_refused(tmp_path):
    ckpt = _adapter(tmp_path / "checkpoint-4")
    assert start_from(_cfg(ckpt, model_revision=SHA)).clash is None
    assert start_from(_cfg(ckpt)).clash is None  # a run that pins none
    clash = start_from(_cfg(ckpt, model_revision=OTHER)).clash
    assert "trained on revision aaaaaaaaaaaa" in clash and "this run pins bbbbbbbbbbbb" in clash
    assert clash.fix == f"set model_revision to {SHA!r}, or drop it"
    assert command.start_refusal(None, _cfg(ckpt, model_revision=OTHER)).code == "E_CONFIG"
    folder = tmp_path / "base"  # a base in a folder has no revision: its contents are compared
    folder.mkdir()
    local = _adapter(tmp_path / "local" / "checkpoint-4",
                     {**SAID, "model_name_or_path": str(folder)})
    assert start_from(_cfg(local, model_name_or_path=str(folder),
                           model_revision=OTHER)).clash is None


def test_a_peft_folder_with_no_poros_config_has_its_base_checked(tmp_path):
    """adapter_config.json names the base it trained on: another one was taken with no word."""
    peft = {"r": 8, "lora_alpha": 16, "base_model_name_or_path": "Org/Model", "revision": SHA,
            "target_modules": ["q_proj"]}
    folder = _adapter(tmp_path / "peft", peft, name="adapter_config.json")
    (folder / checkpoints.FORMAT_MARKER).unlink()  # PEFT's own save
    (folder / checkpoints.COMPLETE_MARKER).unlink()
    cfg = _cfg(folder)
    start = start_from(cfg)
    assert start.clash is None and (cfg.lora_rank, cfg.lora_alpha) == (8, 16)
    assert start.said == {"lora_rank": 8, "lora_alpha": 16, "model_name_or_path": "Org/Model",
                          "model_revision": SHA}
    other = start_from(_cfg(folder, model_name_or_path="Org/Other")).clash
    assert "trained on base 'Org/Model'" in other and "'Org/Other'" in other
    assert other.fix == "set model_name_or_path to the base the adapter trained on ('Org/Model')"
    assert "this run pins bbbbbbbbbbbb" in start_from(_cfg(folder, model_revision=OTHER)).clash
    bare = _adapter(tmp_path / "bare", {"r": 8, "lora_alpha": 16}, name="adapter_config.json")
    assert start_from(_cfg(bare, model_name_or_path="Any/Base")).clash is None  # it names none


# ── a save that did not finish is no adapter ─────────────────────────────────────────────────


def test_a_torn_checkpoint_or_a_staging_folder_is_refused_naming_the_newest_complete(tmp_path):
    run = tmp_path / "run"
    done = _adapter(run / "checkpoint-4")
    torn = _adapter(run / "checkpoint-6", complete=False)  # killed mid-save: no marker
    staged = _adapter(run / "checkpoint-8.tmp")  # complete, never renamed
    for bad in (torn, staged, _adapter(run / "checkpoint-9.old")):
        with pytest.raises(FileNotFoundError) as refused:
            start_adapter(bad)
        assert "a save that did not finish" in str(refused.value)
        assert f"the newest complete checkpoint beside it is {done}" in str(refused.value)
        refusal = command.start_refusal(None, _cfg(bad))
        assert refusal.code == "E_NOT_FOUND" and str(done) in refusal.reason
    assert start_adapter(run) == str(done)  # the run's folder still names its newest complete
    alone = _adapter(tmp_path / "alone" / "checkpoint-2", complete=False)
    with pytest.raises(FileNotFoundError, match="no complete checkpoint is beside it"):
        start_adapter(alone)
    # a run's own folder whose last save was cut short: its checkpoints, not its torn adapter
    (run / checkpoints.FORMAT_MARKER).write_text("1")
    (run / "poros_config.json").write_text(json.dumps(SAID))
    assert start_adapter(run) == str(done)
    cut = _adapter(tmp_path / "cut", complete=False)
    with pytest.raises(FileNotFoundError, match="a save that did not finish"):
        start_adapter(cut)


# ── a checkpoint does not choose which of a repo's code runs ─────────────────────────────────


def test_a_checkpoints_revision_is_not_taken_where_the_run_loads_the_repos_code(tmp_path, caplog):
    """A resume takes its checkpoint's revision so the download is the base it trained on. A run
    that loads the base's own code (trust_remote_code) names its revision itself: a folder from
    elsewhere could name a commit whose code does harm."""
    resume._warned.clear()
    ckpt = _adapter(tmp_path / "checkpoint-4")
    plain = PorosConfig(model_name_or_path="Org/Model", dataset="synthetic")
    assert resume.adopt_revision(str(ckpt), plain) == SHA and plain.model_revision == SHA
    code = PorosConfig(model_name_or_path="Org/Model", dataset="synthetic", trust_remote_code=True)
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        assert resume.adopt_revision(str(ckpt), code) is None
    assert code.model_revision is None
    (said,) = [r.getMessage() for r in caplog.records]
    assert f"Set model_revision: {SHA}" in said and "trust_remote_code" in said
    pinned = PorosConfig(model_name_or_path="Org/Model", dataset="synthetic",
                         trust_remote_code=True, model_revision=SHA)
    assert resume.adopt_revision(str(ckpt), pinned) is None and pinned.model_revision == SHA
