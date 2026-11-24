"""A run's folder does not say where a store is written. Its recorded `store_path` is read where a
store of this model already is, and nothing is created there: with none, `store: ssd` builds at
the default place (`~/.poros/sss`, still on the SSD) and says so in one line. The caller's own
`store_path=` stands. The same holds where `poros eval <run>` and `poros export` build a config
from a run's records (`poros.stores.recorded_store`)."""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros import stores  # noqa: E402
from poros.core import checkpoints  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.moe.load import load_moe_streamed, sss_file_path  # noqa: E402
from poros.train.run import Run  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

NOTE = "No store of this model to reuse at the recorded store_path"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny MoE base on disk, whose store a load builds in seconds on the CPU."""
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=96, hidden_size=64, intermediate_size=96, moe_intermediate_size=40,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2,
    )
    model = Qwen3MoeForCausalLM(config)
    path = tmp_path_factory.mktemp("qwen3_moe")
    model.save_pretrained(path)
    return str(path)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """POROS_HOME, so the default store place is the test's own: (home, its sss folder)."""
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    return tmp_path / "home" / "sss"


# The rule for a recorded store_path here is a CUDA host's: a Mac picks its store as a new run
# does (test_a_mac_takes_a_recorded_store_only_where_it_holds_the_model)
cuda_store_rule = pytest.mark.cuda_host(reason="a CUDA host's rule for a recorded store_path: a "
                                        "Mac picks its store as a new run does")


@pytest.fixture
def loads(monkeypatch):
    """The config each `Run.open` hands the engine, without loading anything."""
    seen = []
    monkeypatch.setattr(PorosTrainer, "from_pretrained", staticmethod(
        lambda model, *, config, **how: seen.append(config) or object()))
    monkeypatch.setattr("poros.core.tokenizer.load_tokenizer",
                        lambda *a, **k: SimpleNamespace(pad_token="<pad>"))
    monkeypatch.setattr(Run, "load", lambda self, ckpt: None)
    monkeypatch.setattr(Run, "_declare_data", lambda self: None)
    return seen


def _run(folder, **said):
    """A run folder whose checkpoint's poros_config.json says ``said``."""
    ckpt = folder / "checkpoint-4"
    ckpt.mkdir(parents=True)
    (ckpt / checkpoints.FORMAT_MARKER).write_text("1")
    (ckpt / "poros_config.json").write_text(
        json.dumps({**PorosConfig().model_dump(mode="json"), **said}))
    (ckpt / checkpoints.COMPLETE_MARKER).write_text("4")
    return folder


def _tree(path):
    return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in path.rglob("*"))


@cuda_store_rule
def test_a_hostile_store_path_is_never_created(tmp_path, home, loads, caplog):
    hostile = tmp_path / "their" / "disk"
    there = tmp_path / "theirs"  # one that exists, empty
    there.mkdir()
    for named in (hostile, there, there / "x.moe.sss"):
        folder = _run(tmp_path / f"run{len(loads)}", model_name_or_path="Org/Moe", store="ssd",
                      store_path=str(named))
        with caplog.at_level(logging.WARNING, logger="poros.stores"):
            Run.open(folder)
        config = loads[-1]
        assert (config.store, config.store_path) == ("ssd", str(home))  # on the SSD, not RAM
        assert config._store_build_path is None
        notes = [r.getMessage() for r in caplog.records if NOTE in r.getMessage()]
        assert notes == [stores.moved_store_note(named, home)] and "\n" not in notes[0]
        caplog.clear()
    assert not hostile.exists() and not hostile.parent.exists()
    assert list(there.iterdir()) == []


@cuda_store_rule
def test_an_existing_store_at_the_recorded_path_is_read(tmp_path, home, loads, checkpoint,
                                                        caplog):
    recorded = tmp_path / "mine"
    load_moe_streamed(checkpoint, store="ssd", store_path=recorded)._poros_moe_store.source.close()
    before = _tree(recorded)
    for path in (recorded, sss_file_path(recorded, checkpoint)):  # the folder, or the file
        folder = _run(tmp_path / f"run{len(loads)}", model_name_or_path=checkpoint, store="ssd",
                      store_path=str(path))
        with caplog.at_level(logging.WARNING):
            Run.open(folder)
        assert loads[-1].store_path == str(path)  # read there
        assert loads[-1]._store_build_path == str(home)  # never built there
    assert NOTE not in caplog.text
    # a store of another model there is not this model's: the default place
    other = _run(tmp_path / "other", model_name_or_path="Org/Other", store="ssd",
                 store_path=str(sss_file_path(recorded, checkpoint)))
    Run.open(other)
    assert loads[-1].store_path == str(home)
    assert _tree(recorded) == before and not home.exists()


def test_the_callers_store_path_wins(tmp_path, home, loads, caplog):
    mine = tmp_path / "my" / "ssd"
    folder = _run(tmp_path / "run", model_name_or_path="Org/Moe", store="ssd",
                  store_path=str(tmp_path / "their" / "disk"))
    with caplog.at_level(logging.WARNING):
        Run.open(folder, store_path=str(mine))
    assert loads[-1].store_path == str(mine) and loads[-1]._store_build_path is None
    assert NOTE not in caplog.text
    assert not (tmp_path / "their").exists()


def test_a_store_that_cannot_be_reused_is_built_at_the_default_place(tmp_path, home, checkpoint,
                                                                     caplog):
    """The recorded store is this model's, but not reusable (here another dtype; another
    revision or Poros version alike): the load builds at build_path and leaves it as it was."""
    recorded = tmp_path / "mine"
    load_moe_streamed(checkpoint, store="ssd", store_path=recorded)._poros_moe_store.source.close()
    before = _tree(recorded)
    with caplog.at_level(logging.WARNING, logger="poros.moe.load"):
        model = load_moe_streamed(checkpoint, torch.float32, store="ssd", store_path=recorded,
                                  build_path=home)
    model._poros_moe_store.source.close()
    assert _tree(recorded) == before  # not a byte, not a temp file
    assert sss_file_path(home, checkpoint).exists() and model._poros_store_path == str(home)
    assert stores.moved_store_note(recorded, home) in caplog.text
    # reusable: read where it is, and nothing at build_path
    elsewhere = tmp_path / "unused"
    load_moe_streamed(checkpoint, store="ssd", store_path=recorded,
                      build_path=elsewhere)._poros_moe_store.source.close()
    assert not elsewhere.exists() and _tree(recorded) == before


@cuda_store_rule
def test_poros_eval_takes_the_same_rule(tmp_path, home):
    from poros.cli.verbs import evaluate_pass

    hostile = tmp_path / "their" / "disk"
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.resolved.json").write_text(json.dumps(
        {"model_name_or_path": "Org/Moe", "store": "ssd", "store_path": str(hostile)}))
    cfg, _ = evaluate_pass.settings(SimpleNamespace(id="r1", output_dir=str(run)), run)
    assert (cfg.store, cfg.store_path, cfg._store_build_path) == ("ssd", str(home), None)
    assert not hostile.parent.exists()


@cuda_store_rule
def test_poros_export_takes_the_same_rule(tmp_path, home, monkeypatch):
    from typer.testing import CliRunner

    import poros.train.export as pe
    from poros.cli.main import app

    hostile = tmp_path / "their" / "disk"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "poros_config.json").write_text(json.dumps({
        "model_name_or_path": "Org/Dense", "model_revision": "a" * 40, "store": "ssd",
        "store_path": str(hostile)}))
    seen = []
    monkeypatch.setattr(pe, "export_merged_dense", lambda cfg, *a, **k: seen.append(cfg) or 1 / 0)
    CliRunner().invoke(app, ["export", str(adapter), "-y"])
    assert seen and seen[0].store_path == str(home)
    assert not hostile.parent.exists()


def test_a_plugin_store_takes_no_path_or_options_from_a_folder():
    said = []
    fields = {"model_name_or_path": "Org/Moe", "store": "mystore", "store_path": "/their/disk",
              "store_options": {"scratch": "/their/tmp"}}
    out, build = stores.recorded_store(fields, say=said.append)
    assert "store_path" not in out and "store_options" not in out and build is None
    assert len(said) == 1 and "pass them yourself" in said[0]
    out, _ = stores.recorded_store(fields, given={"store_path", "store_options"}, say=said.append)
    assert out == fields and len(said) == 1  # the caller's own stand


def test_ram_reads_a_store_and_never_needs_one(tmp_path, checkpoint, monkeypatch):
    """store: ram only reads a store at store_path: with none there it takes no path."""
    monkeypatch.setattr(sys, "platform", "linux")  # a CUDA host's rule (a Mac's: the next test)
    said = []
    out, build = stores.recorded_store({"model_name_or_path": checkpoint, "store": "ram",
                                        "store_path": str(tmp_path / "none")}, say=said.append)
    assert "store_path" not in out and build is None and said == []


def test_a_mac_takes_a_recorded_store_only_where_it_holds_the_model(monkeypatch):
    from poros.metal import store

    held = {"/mine"}
    monkeypatch.setattr(store, "holds", lambda path, model: str(path) in held)
    monkeypatch.setattr(sys, "platform", "darwin")
    said = []
    fields = {"model_name_or_path": "Org/Moe", "store": "ssd"}
    assert stores.recorded_store({**fields, "store_path": "/mine"}, say=said.append) == (
        {**fields, "store_path": "/mine"}, None)
    out, _ = stores.recorded_store({**fields, "store_path": "/their/disk"}, say=said.append)
    assert out == {"model_name_or_path": "Org/Moe"}  # the table and the store pick, as new
    assert len(said) == 1 and "/their/disk" in said[0]


def test_the_trainer_hands_the_load_its_build_path(tmp_path, monkeypatch):
    """`Run.open` sets it on the config; the MoE setup passes it to the load, and records where
    the store was built."""
    import poros.moe.load as moe_load
    import poros.preflight.start as start
    import poros.sss.probe as sss_probe

    seen = {}
    built = str(tmp_path / "default")

    def load(*a, **how):
        seen.update(how)
        raise RuntimeError("stop")

    monkeypatch.setattr(start, "moe_settings_refusal", lambda *a: None)
    monkeypatch.setattr(start, "moe_on_cuda", lambda *a, **k: None)
    monkeypatch.setattr(sss_probe, "probe", lambda path: pytest.fail(f"probed {path}"))
    monkeypatch.setattr(moe_load, "load_moe_streamed", load)
    trainer = PorosTrainer.__new__(PorosTrainer)
    trainer.config = PorosConfig(model_name_or_path="Org/Moe", store="ram")
    trainer.config._store_build_path = built
    trainer._metal, trainer.device = False, torch.device("cpu")
    trainer.compute = SimpleNamespace(moe_request=lambda: None)
    with pytest.raises(RuntimeError, match="stop"):
        trainer._setup_moe()
    assert seen["build_path"] is None  # store: ram builds nothing
    trainer.config.store, trainer.config.store_path = "ssd", str(tmp_path / "mine")
    monkeypatch.setattr(sss_probe, "probe", lambda path: SimpleNamespace(
        bandwidth=1, b_ssd_qd1=1, queue_depth=1, b_ssd=1, direct=True, b_h2d=None))
    with pytest.raises(RuntimeError, match="stop"):
        trainer._setup_moe()
    assert (seen["store_path"], seen["build_path"]) == (str(tmp_path / "mine"), built)
    model = SimpleNamespace(_poros_store_path=built, _poros_moe_store=SimpleNamespace(
        source=object()), config=SimpleNamespace(name_or_path="Org/Moe"))
    monkeypatch.setattr(moe_load, "load_moe_streamed", lambda *a, **k: model)
    trainer.compute = SimpleNamespace(moe_request=lambda: None, prepare_moe=load)
    with pytest.raises(RuntimeError, match="stop"):  # past the record
        trainer._setup_moe()
    assert trainer.config.store_path == built
