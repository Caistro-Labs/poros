"""Train further on new or more data: a new run that starts from a checkpoint's adapter.

A resume is the same run, so other data is refused. The other route, for someone who wants the
new data, is a new run from the adapter: its weights, a fresh optimizer and schedule, its own
folder, the new data on its record. `poros train --from` did it for a recipe's run only: the
classic loop refused, a config's `init_adapter:` was read by recipes only (the classic loop
trained from scratch with no word), `poros.Run.create` ignored it, and the resume's refusal
never named the route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from poros.core.errors import PorosConfigError  # noqa: E402
from poros.preflight import resume  # noqa: E402
from poros.preflight.resume import INPUTS_RECORD, dataset_identity  # noqa: E402


def _rows(path, n=4, word="row"):
    path.write_text("".join(json.dumps({"text": f"{word} {i}"}) + "\n" for i in range(n)))
    return path


def _trainer(out, steps=4, rank=None, **cfg):
    """A tiny trainer whose ``train(dataset)`` takes a file's name (as test_resume_identity's)."""
    from tests.test_train_loop_cpu import _loader, _make_trainer, _tiny_peft_qwen2

    torch.manual_seed(0)
    model, small = _tiny_peft_qwen2(torch)
    tr = _make_trainer(torch, model, str(out), max_steps=steps)
    tr.config.model_name_or_path = "dummy"
    for name, value in cfg.items():
        setattr(tr.config, name, value)
    if rank is not None:
        tr.config.lora_rank = rank
    tr.loader = _loader(torch, small.vocab_size, n=32)
    tr._prepare_dataloader = lambda dataset, length, for_eval=False: (
        tr.loader if isinstance(dataset, str) else dataset)
    return tr


def _adapter(tr):
    return {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}


@pytest.fixture
def first(tmp_path):
    """A run of four steps on a.jsonl: (its trainer, its newest checkpoint, a.jsonl)."""
    a = _rows(tmp_path / "a.jsonl")
    tr = _trainer(tmp_path / "a", learning_rate=1e-2)
    tr.train(str(a), steps=4)
    return tr, tr._save_checkpoint(), a


def test_the_classic_loop_starts_a_new_run_from_an_adapter_on_other_data(tmp_path, first):
    """The classic loop read no init_adapter: `--from` was refused there, and a config's
    init_adapter trained from scratch. It loads the adapter, and all else is the new run's."""
    old, ckpt, a = first
    b = _rows(tmp_path / "b.jsonl", n=9, word="more")
    new = _trainer(tmp_path / "b", init_adapter=str(tmp_path / "a"))  # the run's folder
    seen = {}
    real = new.load_checkpoint

    def load(path, **how):
        real(path, **how)
        seen.update(path=path, step=new.global_step, weights=_adapter(new), **how)

    new.load_checkpoint = load
    new.train(str(b), steps=2)
    assert seen["path"] == ckpt  # a run's folder: its newest checkpoint
    assert (seen["load_optimizer"], seen["load_rng"], seen["step"]) == (False, False, 0)
    then = _adapter(old)
    assert all(torch.equal(w, then[n]) for n, w in seen["weights"].items())  # where it stopped
    events = [json.loads(x) for x in (tmp_path / "b" / "events.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events if e["event_type"] == "step"] == [0, 1]  # its own steps
    kept = Path(new._save_checkpoint())
    assert kept.parent == tmp_path / "b"
    assert json.loads((kept / INPUTS_RECORD).read_text())["dataset"] == dataset_identity(b)
    said = json.loads((kept / "poros_config.json").read_text())
    assert said["dataset"] == str(b) and said["init_adapter"] == str(tmp_path / "a")


def test_a_run_with_no_init_adapter_loads_nothing(tmp_path):
    tr = _trainer(tmp_path / "n")
    tr.load_checkpoint = lambda *a, **k: pytest.fail("a fresh run loads no adapter")
    assert tr.start_from_adapter() is None
    tr.train(str(_rows(tmp_path / "a.jsonl")), steps=1)


def test_an_adapter_of_another_shape_or_base_is_refused_with_both_named(tmp_path, first):
    _, ckpt, _ = first
    cfg = json.loads((Path(ckpt) / "poros_config.json").read_text())
    was = cfg["lora_rank"]
    new = _trainer(tmp_path / "b", init_adapter=ckpt, rank=was * 2)
    before = _adapter(new)
    with pytest.raises(PorosConfigError, match="Can't start from this adapter") as refused:
        new.train(str(_rows(tmp_path / "b.jsonl")), steps=1)
    said = str(refused.value)
    assert f"lora_rank {was}" in said and f"this run has {was * 2}" in said
    assert f"Set lora_rank to {was}" in said
    assert all(torch.equal(p, before[n]) for n, p in _adapter(new).items())  # nothing loaded
    other = _trainer(tmp_path / "c", init_adapter=ckpt, model_name_or_path="Org/Other")
    with pytest.raises(PorosConfigError, match="trained on base 'dummy'"):
        other.train(str(_rows(tmp_path / "c.jsonl")), steps=1)
    with pytest.raises(FileNotFoundError, match="no folder at"):
        _trainer(tmp_path / "d", init_adapter=str(tmp_path / "nowhere")).train(
            str(_rows(tmp_path / "d.jsonl")), steps=1)


def test_what_init_adapter_names(tmp_path, first):
    _, ckpt, _ = first
    assert resume.start_adapter(ckpt) == ckpt  # a checkpoint
    assert resume.start_adapter(tmp_path / "a") == ckpt  # a run's folder: its newest
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="holds no adapter"):
        resume.start_adapter(empty)
    (empty / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}))
    assert resume.start_adapter(empty) == str(empty)  # a PEFT adapter folder
    from poros.core.config import PorosConfig

    assert resume.start_from(PorosConfig(model_name_or_path="x", init_adapter=str(empty))).said == {
        "lora_rank": 8, "lora_alpha": 16}
    # a finished run's folder holds its adapter too: the later of the two, the checkpoint on a tie
    from poros.core import checkpoints

    run = tmp_path / "a"
    (run / "adapter_config.json").write_text("{}")
    assert resume.start_adapter(run) == ckpt  # the folder does not say its step
    step = checkpoints.checkpoint_step(Path(ckpt))
    (run / checkpoints.COMPLETE_MARKER).write_text(str(step))
    assert resume.start_adapter(run) == ckpt  # the same step
    (run / checkpoints.COMPLETE_MARKER).write_text(str(step + 5))
    assert resume.start_adapter(run) == str(run)  # trained on past its newest checkpoint


def test_a_resume_refused_for_other_data_names_the_new_run_from_its_adapter(tmp_path, first):
    """The refusal said "set dataset to a copy ... or start a new run in another output_dir":
    nothing for someone who wants the new data."""
    _, ckpt, a = first
    b = _rows(tmp_path / "b.jsonl", n=9, word="more")
    with pytest.raises(PorosConfigError, match="holds other data") as refused:
        _trainer(tmp_path / "a").train(str(b), resume_from=ckpt)
    said = str(refused.value)
    assert "Set dataset to a copy of the data the checkpoint trained on" in said
    assert "To train its adapter on other data or settings, start a new run from it" in said
    assert "poros train <model> --from <run> --data <file>" in said
    assert f"init_adapter: {ckpt} with another output_dir" in said


def test_the_route_is_named_where_a_new_run_may_differ_and_nowhere_else(tmp_path, first):
    from poros.preflight import command

    tr, ckpt, a = first

    def clash(**other):
        return resume.resume_clash(ckpt, tr.config.model_copy(update=other))

    assert clash() is None
    for other in ({"max_seq_length": 64}, {"gradient_accumulation_steps": 4},
                  {"per_device_batch_size": 3}, {"seed": 7}):
        assert clash(**other).route == resume.new_run_from(ckpt), other
    for other in ({"lora_rank": 64}, {"model_name_or_path": "Org/Other"}):
        assert clash(**other).route is None, other  # its adapter would not load there either
    refused = command.resume_refusal(ckpt, tr.config.model_copy(update={"max_seq_length": 64}))
    assert refused.context[-1] == resume.new_run_from(ckpt)
    assert "--from <run>" in refused.context[-1] and "init_adapter" in refused.context[-1]


def test_run_create_starts_from_an_adapter_and_refuses_another_before_the_download(
        tmp_path, first, monkeypatch):
    """`poros.Run.create(..., init_adapter=...)` built its config with it and loaded nothing."""
    from poros.train import run as run_api
    from tests.test_run_api import Tok

    old, ckpt, _ = first
    was = json.loads((Path(ckpt) / "poros_config.json").read_text())
    loads = []

    def loaded(model, **how):
        loads.append(model)
        tr = _trainer(how["config"].output_dir)
        tr.config.init_adapter = how["config"].init_adapter
        return tr

    monkeypatch.setattr(run_api.PorosTrainer, "from_pretrained", staticmethod(loaded))
    with pytest.raises(PorosConfigError, match="Can't start from this adapter"):
        run_api.Run.create("dummy", rank=was["lora_rank"] * 2, output_dir=str(tmp_path / "x"),
                           init_adapter=ckpt, tokenizer=Tok())
    assert loads == []  # refused before the download
    run = run_api.Run.create("dummy", rank=was["lora_rank"], output_dir=str(tmp_path / "b"),
                             init_adapter=str(tmp_path / "a"), tokenizer=Tok())
    assert loads == ["dummy"] and run.step == 0
    then = _adapter(old)
    assert all(torch.equal(p, then[n]) for n, p in _adapter(run._engine).items())


def test_run_create_takes_the_adapters_alpha_where_none_is_named(tmp_path, first, monkeypatch):
    from poros.train import run as run_api
    from tests.test_run_api import Tok

    _, ckpt, _ = first
    was = json.loads((Path(ckpt) / "poros_config.json").read_text())
    seen = {}

    def loaded(model, **how):
        seen["config"] = how["config"]
        raise RuntimeError("stop before the load")

    monkeypatch.setattr(run_api.PorosTrainer, "from_pretrained", staticmethod(loaded))
    for alpha, out, want in ((None, "p", was["lora_alpha"]), (7, "q", 7)):
        with pytest.raises(RuntimeError):
            run_api.Run.create("dummy", rank=was["lora_rank"], alpha=alpha, init_adapter=ckpt,
                               output_dir=str(tmp_path / out), tokenizer=Tok())
        assert seen["config"].lora_alpha == want and seen["config"].init_adapter == ckpt
    with pytest.raises(RuntimeError):  # no adapter: twice the rank, as before
        run_api.Run.create("dummy", rank=12, output_dir=str(tmp_path / "r"), tokenizer=Tok())
    assert seen["config"].lora_alpha == 24
