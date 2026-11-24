"""A run's folder is data, from wherever it came. `poros.Run.open(folder)` reads what the run
trained with from it, and nothing in it speaks for the caller: the base's own code runs only on
the caller's `trust_remote_code=True`, a push goes where the caller says, and the run writes into
the folder it was opened from. A new run from an adapter (`init_adapter`) is checked by the one
check before any download, from `Run.create` and from `PorosTrainer.from_pretrained` alike."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from poros.core import checkpoints  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosConfigError  # noqa: E402
from poros.preflight import resume  # noqa: E402
from poros.train import run as run_api  # noqa: E402
from poros.train.run import Run  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

HOSTILE = {"model_name_or_path": "Someone/Repo", "trust_remote_code": True, "push_to_hub": True,
           "hub_model_id": "someone/takes-it", "init_adapter": "/their/adapter",
           "output_dir": "/their/old/path", "lora_rank": 8}


def _saved(folder, said, step=4):
    """A complete save in ``folder`` whose poros_config.json says ``said``."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / checkpoints.FORMAT_MARKER).write_text("1")
    (folder / "poros_config.json").write_text(
        json.dumps({**PorosConfig().model_dump(mode="json"), **said}))
    (folder / checkpoints.COMPLETE_MARKER).write_text(str(step))
    return folder


@pytest.fixture
def loads(monkeypatch):
    """What `Run.open` would load, without loading it: (model, config) of the engine's load and
    (model, trust_remote_code) of the tokenizer's."""
    seen = {"engine": [], "tokenizer": []}

    def engine(model, *, config, **how):
        seen["engine"].append((model, config))
        return object()

    def tokenizer(name, revision=None, trust_remote_code=False):
        seen["tokenizer"].append((name, trust_remote_code))
        return type("Tok", (), {"pad_token": "<pad>"})()

    monkeypatch.setattr(PorosTrainer, "from_pretrained", staticmethod(engine))
    monkeypatch.setattr("poros.core.tokenizer.load_tokenizer", tokenizer)
    monkeypatch.setattr(Run, "load", lambda self, ckpt: None)
    monkeypatch.setattr(Run, "_declare_data", lambda self: None)
    return seen


def test_a_hostile_folder_never_turns_remote_code_on(tmp_path, loads):
    """Its poros_config.json said trust_remote_code: true and named a repo: the documented
    `Run.open(folder)` handed both to the model's and the tokenizer's loads."""
    ckpt = _saved(tmp_path / "run" / "checkpoint-4", HOSTILE)
    Run.open(tmp_path / "run")
    Run.open(ckpt)
    assert [trust for _, trust in loads["tokenizer"]] == [False, False]
    for model, config in loads["engine"]:
        assert model == "Someone/Repo"  # the base the run names is what continuing it loads
        assert config.trust_remote_code is False
        assert (config.push_to_hub, config.hub_model_id, config.init_adapter) == (False, None, None)
        assert config.output_dir == str(tmp_path / "run")
        assert config.lora_rank == 8  # what the run trained with is still the folder's to say


def test_the_callers_own_flag_still_grants_it(tmp_path, loads):
    _saved(tmp_path / "run" / "checkpoint-4", {**HOSTILE, "trust_remote_code": False})
    Run.open(tmp_path / "run", trust_remote_code=True, hub_model_id="me/mine",
             output_dir=str(tmp_path / "elsewhere"))
    (_, config), = loads["engine"]
    assert config.trust_remote_code is True and loads["tokenizer"] == [("Someone/Repo", True)]
    assert config.hub_model_id == "me/mine" and config.output_dir == str(tmp_path / "elsewhere")


def test_no_engine_takes_remote_code_from_a_runs_records():
    """`poros eval` builds its engine from a run's records by an allowlist, and `poros export`
    from the flag alone: neither list holds the setting."""
    from poros.cli.verbs import evaluate_pass

    assert not {"trust_remote_code", "push_to_hub", "hub_model_id", "output_dir", "loop",
                "init_adapter"} & set(evaluate_pass.KEPT)
    assert "trust_remote_code" in run_api._NOT_THE_FOLDERS


# ── an opened run writes into the folder it was opened from ──────────────────────────────────


@pytest.mark.parametrize("recorded", ["out/run", "/their/old/path", "./run"])
def test_an_opened_run_writes_where_it_was_opened_from(tmp_path, loads, monkeypatch, recorded):
    """A finished run's folder kept the output_dir its config recorded: a relative one resolved
    against the current folder, and a moved run wrote to its old path."""
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "moved" / "run"
    _saved(run, {**HOSTILE, "output_dir": recorded})  # a finished run: its adapter in its folder
    (run / "events.jsonl").write_text("")
    _saved(run / "checkpoint-4", {**HOSTILE, "output_dir": recorded})
    _saved(run / "start", {**HOSTILE, "output_dir": recorded}, step=-1)  # run.save("start")
    for opened in (run, run / "checkpoint-4", run / "start"):
        Run.open(opened)
    assert [config.output_dir for _, config in loads["engine"]] == [str(run)] * 3
    alone = _saved(tmp_path / "copied" / "adapter", {**HOSTILE, "output_dir": recorded})
    Run.open(alone)  # a save copied out on its own: the folder it is
    assert loads["engine"][-1][1].output_dir == str(alone)
    Run.open(run, output_dir=str(tmp_path / "named"))  # unless the caller says otherwise
    assert loads["engine"][-1][1].output_dir == str(tmp_path / "named")


# ── a new run from an adapter: one check, before the download ────────────────────────────────


def _adapter(folder, **said):
    return _saved(folder, {"model_name_or_path": "Org/Model", "lora_rank": 8, "lora_alpha": 24,
                           "lora_target_modules": ["q_proj", "v_proj"], **said})


def test_run_create_takes_the_adapters_alpha_and_target_modules(tmp_path, loads, monkeypatch):
    """With no targets named, the run took the architecture's own and the adapter's weights
    failed on a key mismatch after the load: its target modules are taken as its alpha is."""
    ckpt = _adapter(tmp_path / "old" / "checkpoint-4")
    monkeypatch.setattr(Run, "_loaded", classmethod(
        lambda cls, model, config, *a: loads["engine"].append((model, config))
        or type("R", (), {"_engine": type("E", (), {"start_from_adapter": lambda s: None})()})()))
    Run.create("Org/Model", rank=8, init_adapter=str(tmp_path / "old"),
               output_dir=str(tmp_path / "new"))
    config = loads["engine"][-1][1]
    assert config.init_adapter == str(ckpt)
    assert (config.lora_alpha, config.lora_target_modules) == (24, ["q_proj", "v_proj"])
    Run.create("Org/Model", rank=8, alpha=16, lora_target_modules=["v_proj", "q_proj"],
               init_adapter=str(ckpt), output_dir=str(tmp_path / "new2"))
    config = loads["engine"][-1][1]
    assert (config.lora_alpha, config.lora_target_modules) == (16, ["v_proj", "q_proj"])
    Run.create("Org/Model", rank=8, output_dir=str(tmp_path / "new3"))  # no adapter: as ever
    assert loads["engine"][-1][1].lora_alpha == 16
    own = PorosConfig(model_name_or_path="Org/Model", lora_rank=8, init_adapter=str(ckpt),
                      output_dir=str(tmp_path / "new4"))
    Run.create("Org/Model", config=own)  # a config of the caller's own: checked, nothing taken
    assert (own.lora_alpha, own.lora_target_modules) == (32, None)
    before = len(loads["engine"])
    with pytest.raises(PorosConfigError, match="trained with lora_target_modules") as refused:
        Run.create("Org/Model", rank=8, lora_target_modules=["k_proj"], init_adapter=str(ckpt),
                   output_dir=str(tmp_path / "new5"))
    assert "Set lora_target_modules to ['q_proj', 'v_proj']" in str(refused.value)
    assert len(loads["engine"]) == before  # refused before the load


def test_from_pretrained_checks_its_init_adapter_before_the_download(tmp_path, monkeypatch):
    """PorosTrainer.from_pretrained(init_adapter=) downloaded and loaded the base before train()
    found the adapter was another shape, another base, or not there."""
    ckpt = _adapter(tmp_path / "old" / "checkpoint-4")
    monkeypatch.setattr(PorosTrainer, "__init__",
                        lambda self, config: pytest.fail("built a trainer: past the check"))
    with pytest.raises(PorosConfigError, match="trained with lora_rank 8, and this run has 16"):
        PorosTrainer.from_pretrained("Org/Model", init_adapter=str(ckpt))
    with pytest.raises(PorosConfigError, match="trained on base 'Org/Model'"):
        PorosTrainer.from_pretrained("Org/Other", lora_rank=8, init_adapter=str(ckpt))
    with pytest.raises(FileNotFoundError, match="no folder at"):
        PorosTrainer.from_pretrained("Org/Model", init_adapter=str(tmp_path / "nowhere"))
    torn = _adapter(tmp_path / "old" / "checkpoint-6")
    (torn / checkpoints.COMPLETE_MARKER).unlink()
    with pytest.raises(FileNotFoundError, match="a save that did not finish"):
        PorosTrainer.from_pretrained("Org/Model", lora_rank=8, init_adapter=str(torn))


def test_the_three_entries_ask_the_one_check(tmp_path, loads, monkeypatch):
    """poros train's preflight, Run.create and the trainer's own later check each held a copy."""
    ckpt = _adapter(tmp_path / "old" / "checkpoint-4")
    asked = []
    real = resume.start_from
    monkeypatch.setattr(resume, "start_from",
                        lambda cfg, *a, **k: asked.append(cfg) or real(cfg, *a, **k))
    monkeypatch.setattr(Run, "_loaded", classmethod(
        lambda cls, model, config, *a: type(
            "R", (), {"_engine": type("E", (), {"start_from_adapter": lambda s: None})()})()))
    Run.create("Org/Model", rank=8, init_adapter=str(ckpt), output_dir=str(tmp_path / "new"))
    assert len(asked) == 1
    trainer = PorosTrainer.__new__(PorosTrainer)
    trainer.config = PorosConfig(model_name_or_path="Org/Model", lora_rank=8, lora_alpha=24,
                                 init_adapter=str(ckpt))
    loaded = []
    trainer.load_checkpoint = lambda path, **how: loaded.append((path, how))
    assert trainer.start_from_adapter() == str(ckpt) and len(asked) == 2
    assert loaded == [(str(ckpt), {"load_optimizer": False, "load_rng": False})]
