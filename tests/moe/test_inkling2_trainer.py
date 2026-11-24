"""PorosTrainer on a tiny Inkling 2 Mini checkpoint in the released layout, on the CPU.

Inkling 2 Mini is a registry target until its GPU gate passes; here it is marked validated for
the test only (the trainer refuses a MoE family that is not), so the route it will take is
exercised end to end: the streamed load of the released experts, prepare_moe with the attention
and expert adapters, the visits, the keyed random states per pass, a save and a resume that ends
where the uninterrupted run did.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()

import poros.core.arch_registry as registry  # noqa: E402
from tests.moe import inkling2_tiny as tiny  # noqa: E402


@pytest.fixture
def validated(monkeypatch):
    for name in ("inkling2_loop", "inkling2_loop_text"):
        spec = dataclasses.replace(registry._BUILTIN_ARCHITECTURES[name], status="validated")
        monkeypatch.setitem(registry._BUILTIN_ARCHITECTURES, name, spec)
        monkeypatch.setitem(registry._ARCHITECTURES, name, spec)


def _loader(n, seq=16):
    rows = [{"input_ids": x, "labels": x.clone()} for x in tiny.ids(n, seq)]
    return torch.utils.data.DataLoader(rows, batch_size=1)


def _trainer(base, output_dir, **overrides):
    from poros.train.trainer import PorosTrainer

    knobs = dict(output_dir=str(output_dir), max_steps=2, save_steps=0, progress=False)
    return PorosTrainer.from_pretrained(str(base), **{**knobs, **overrides})


def test_the_trainer_resumes_bitwise(validated, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base)
    full = _trainer(base, tmp_path, max_steps=4, save_steps=2, gradient_accumulation_steps=2)
    assert full.block_manager is None and getattr(full.model, "poros_visits", False)
    assert full.model.get_decoder().poros_visit_runner is not None
    full.train(_loader(3))
    resumed = _trainer(base, tmp_path, max_steps=4, gradient_accumulation_steps=2)
    resumed.train(_loader(3), resume_from=str(tmp_path / "checkpoint-2"))
    assert resumed.history.losses == full.history.losses[3:]
    after = dict(resumed.model.named_parameters())
    trained = [(n, p) for n, p in full.model.named_parameters() if p.requires_grad]
    assert any(".adapter.lora_" in n for n, _ in trained)
    assert any(".self_attn.q_proj.linear.lora_A" in n for n, _ in trained)
    for name, param in trained:
        assert torch.equal(param, after[name]), name
    # the optimizer's state too, ExpertAdamW's remainder included (the experts' adapters are
    # bf16, the attention's fp32: one optimizer over both)
    from poros.train.expert_adamw import ExpertAdamW

    assert isinstance(full.optimizer, ExpertAdamW) and isinstance(resumed.optimizer, ExpertAdamW)
    dtypes = {p.dtype for _, p in trained}
    assert dtypes == {torch.bfloat16, torch.float32}
    resumed_params = dict(resumed.model.named_parameters())
    for name, param in trained:
        a, b = full.optimizer.state[param], resumed.optimizer.state[resumed_params[name]]
        assert a.keys() == b.keys() and ("remainder" in a) == (param.dtype == torch.bfloat16)
        for key in a:
            assert torch.equal(a[key], b[key]), (name, key)


def test_a_block_major_path_is_refused_in_words():
    """Block-major order does not walk the loop's visits: unset, the path is sequential for that
    reason; set to a block-major path, it is refused before the download, in words."""
    from poros.core.config import PorosConfig
    from poros.preflight.start import path_refusal_error

    released = {"model_type": "inkling2_loop", "text_config": {}}
    unset = PorosConfig(model_name_or_path="x")
    unset.apply_cuda_defaults(released)
    assert unset.path == "sequential" and "reruns its middle layers" in unset.path_reason
    err = path_refusal_error(
        PorosConfig(model_name_or_path="x", path="sequential").model_copy(update={"path": "fast"}),
        released,
    )
    assert err is not None and "reruns its middle layers" in str(err)


def _read_merged(path):
    import json

    from safetensors import safe_open

    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(path / shard), "pt") as f:
            out.update({n: f.get_tensor(n) for n in f.keys()})
    return json.loads((path / "config.json").read_text()), out


def test_the_run_records_its_expert_format_and_export_reads_it(validated, tmp_path, monkeypatch):
    """The run records released (the default) in its resolved config, every checkpoint and its
    saved adapter; an export with POROS_INKLING_EXPERTS=nf4 set still merges against the
    released weights the run trained on (logits equal the trained model's)."""
    import json

    from typer.testing import CliRunner

    from poros.cli.main import app
    from poros.moe.inkling2 import codec
    from poros.moe.inkling2.config import text_config_of
    from poros.moe.inkling2.modeling import Inkling2LoopForCausalLM

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32)
    run = tmp_path / "run"
    trainer = _trainer(
        base,
        run,
        torch_dtype="float32",
        learning_rate=1e-2,
        warmup_steps=0,
        max_steps=2,
        save_steps=1,
    )
    trainer.train(_loader(2))
    trainer.save(str(run))
    assert trainer.config.moe_expert_format == "released"
    for folder in [run, *sorted(run.glob("checkpoint-*"))]:
        saved = json.loads((folder / "poros_config.json").read_text())
        assert saved["moe_expert_format"] == "released", folder

    monkeypatch.setenv("POROS_INKLING_EXPERTS", "nf4")  # the export must not read it
    out = tmp_path / "merged"
    res = CliRunner().invoke(app, ["export", str(run), "-o", str(out)])
    assert res.exit_code == 0, res.output
    config, merged = _read_merged(out)
    reloaded = Inkling2LoopForCausalLM(text_config_of(config))
    reloaded.load_state_dict(
        codec.convert_state_dict(
            merged.__getitem__,
            list(merged),
            unpadded_vocab_size=reloaded.config.unpadded_vocab_size,
            dtype=torch.float32,
            expert_dtype=torch.float32,
        )
    )
    ids = next(iter(_loader(1)))["input_ids"]
    trainer.model.eval()
    with torch.no_grad():
        want = trainer.model(input_ids=ids.to(trainer.device)).logits.cpu()
        got = reloaded(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_a_resume_with_another_expert_format_is_refused(validated, tmp_path, monkeypatch):
    from poros.core.errors import PorosConfigError

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base)
    full = _trainer(base, tmp_path, max_steps=2, save_steps=1)
    full.train(_loader(2))
    checkpoint = str(sorted(tmp_path.glob("checkpoint-*"))[0])
    with pytest.raises(Exception, match="moe_expert_format"):  # set in the config: a clash
        resumed = _trainer(base, tmp_path, max_steps=2, moe_expert_format="nf4")
        resumed.train(_loader(2), resume_from=checkpoint)
    # unset in the config, the variable asking nf4: refused too, naming both formats
    monkeypatch.setenv("POROS_INKLING_EXPERTS", "nf4")
    with pytest.raises(PorosConfigError, match="moe_expert_format 'released'.*'nf4'"):
        resumed = _trainer(base, tmp_path, max_steps=2)
        resumed.train(_loader(2), resume_from=checkpoint)


def test_a_merge_without_the_record_is_refused(tmp_path):
    from poros.core.errors import PorosConfigError
    from poros.moe.inkling2.export import export_merged

    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base)
    with pytest.raises(PorosConfigError, match="does not record"):
        export_merged(str(base), {}, str(tmp_path / "out"), scaling=2.0, expert_codec=None)


@pytest.mark.parametrize("quantize", [False, {0, 1, 2}])
def test_a_record_the_base_cannot_bear_out_is_refused(tmp_path, quantize):
    """A run from before the format was decided, on an unquantized base, recorded released while
    it held NF4: its merge is refused in words, before anything is written, as is any merge on
    a base quantized on some layers only."""
    from poros.core.errors import PorosConfigError
    from poros.moe.inkling2.export import export_merged

    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32, quantize=quantize)
    # nf4 of an unquantized base merges (test_an_unquantized_base_is_held_in_nf4_...)
    for record in ("released",) if quantize is False else ("released", "nf4"):
        out = tmp_path / f"out_{record}"
        with pytest.raises(PorosConfigError, match="no merge") as err:
            export_merged(
                str(base), {}, str(out), scaling=2.0, dtype=torch.float32, expert_codec=record
            )
        said = "unquantized" if quantize is False else "quantized on some layers only"
        assert f"held as {record}" in str(err.value) and said in str(err.value)
        assert not out.exists()


def _merged_logits_equal(trainer, out):
    from poros.moe.inkling2 import codec
    from poros.moe.inkling2.config import text_config_of
    from poros.moe.inkling2.modeling import Inkling2LoopForCausalLM

    config, merged = _read_merged(out)
    reloaded = Inkling2LoopForCausalLM(text_config_of(config))
    reloaded.load_state_dict(
        codec.convert_state_dict(
            merged.__getitem__,
            list(merged),
            unpadded_vocab_size=reloaded.config.unpadded_vocab_size,
            dtype=torch.float32,
            expert_dtype=torch.float32,
        )
    )
    ids = next(iter(_loader(1)))["input_ids"]
    trainer.model.eval()
    with torch.no_grad():
        want = trainer.model(input_ids=ids.to(trainer.device)).logits.cpu()
        got = reloaded(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    return config


def _train_and_merge(base, run, out):
    import json

    from typer.testing import CliRunner

    from poros.cli.main import app

    trainer = _trainer(
        base,
        run,
        torch_dtype="float32",
        learning_rate=1e-2,
        warmup_steps=0,
        max_steps=2,
        save_steps=1,
    )
    trainer.train(_loader(2))
    trainer.save(str(run))
    saved = [
        json.loads((f / "poros_config.json").read_text())["moe_expert_format"]
        for f in [run, *sorted(run.glob("checkpoint-*"))]
    ]
    res = CliRunner().invoke(app, ["export", str(run), "-o", str(out)])
    assert res.exit_code == 0, res.output
    return trainer, saved


def test_an_unquantized_base_is_held_in_nf4_and_says_so_two_generations(
    validated, tmp_path, monkeypatch
):
    """A base whose experts are unquantized (a merged export, a bf16 save) has no released bytes
    to keep: its experts are held in NF4, and the run, its checkpoints and its adapter say nf4;
    its merge is against that NF4 round trip, so the merged model computes what was trained.
    Training on that merged export again, and merging again, holds the same way."""
    from poros.moe.store import MoeExpertStore

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32, quantize=False)
    first, saved = _train_and_merge(base, tmp_path / "run1", tmp_path / "merged1")
    assert first.config.moe_expert_format == "nf4" and set(saved) == {"nf4"}
    assert first.model._poros_expert_format == "nf4"
    assert type(first.model._poros_moe_store) is MoeExpertStore
    _merged_logits_equal(first, tmp_path / "merged1")
    second, saved = _train_and_merge(tmp_path / "merged1", tmp_path / "run2", tmp_path / "merged2")
    assert second.config.moe_expert_format == "nf4" and set(saved) == {"nf4"}
    _merged_logits_equal(second, tmp_path / "merged2")


def test_released_is_refused_where_the_checkpoint_has_none(validated, tmp_path, monkeypatch):
    from poros.core.errors import PorosConfigError

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, quantize=False)
    with pytest.raises(PorosConfigError, match="there are none to keep"):
        _trainer(base, tmp_path / "run", moe_expert_format="released")
    monkeypatch.setenv("POROS_INKLING_EXPERTS", "released")
    with pytest.raises(PorosConfigError, match="there are none to keep"):
        _trainer(base, tmp_path / "run")
    from poros.moe.load import load_moe_streamed

    with pytest.raises(PorosConfigError, match="there are none to keep"):
        load_moe_streamed(str(base), torch.bfloat16, expert_format="released")


def test_a_partly_quantized_checkpoint_is_refused(validated, tmp_path, monkeypatch):
    from poros.core.errors import PorosConfigError

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, quantize={0, 1})
    with pytest.raises(PorosConfigError, match="quantized on some layers and unquantized"):
        _trainer(base, tmp_path / "run")


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_other_families_are_refused_the_field_and_record_nothing_new(kind, tmp_path):
    """moe_expert_format given for a model with no such choice is refused in words before
    anything loads; and a run of any other family writes poros_config.json, and its adapter's
    manifest, without the field (as before it existed: byte for byte against the engine before
    the field, checked by hand for a dense and a MoE run's config when this was written)."""
    import json

    import transformers

    from poros.core.config import PorosConfig
    from poros.core.errors import PorosConfigError
    from poros.core.hub_files import portable_manifest
    from poros.train.checkpointing import recorded_config

    base = tmp_path / "base"
    if kind == "dense":
        cfg = transformers.Qwen2Config(
            vocab_size=96,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    else:
        cfg = transformers.OlmoeConfig(
            vocab_size=96,
            hidden_size=64,
            intermediate_size=40,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=12,
            num_experts_per_tok=3,
        )
    cfg.save_pretrained(base)
    with pytest.raises(PorosConfigError, match="has no such choice"):
        _trainer(base, tmp_path / "run", moe_expert_format="nf4")
    run = tmp_path / "saved"
    run.mkdir()
    config = PorosConfig(model_name_or_path=str(base), output_dir=str(run))
    (run / "poros_config.json").write_text(json.dumps(recorded_config(config), indent=2))
    assert "moe_expert_format" not in (run / "poros_config.json").read_text()
    assert b"moe_expert_format" not in (portable_manifest(run) or b"")
    assert "moe_expert_format" not in config.model_dump()


def test_a_variant_without_a_softcap_takes_the_fused_moe_head(validated, tmp_path):
    """With no final softcap the fused MoE head applies (poros.train.compute), and the trainer
    reads the decoder's output as transformers' decoders give it (``last_hidden_state``, the
    final norm's output): its loss is the model's own."""
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32, final_logit_softcapping=None)
    trainer = _trainer(base, tmp_path / "run", torch_dtype="float32")
    assert trainer.model.config.final_logit_softcapping is None
    trainer.head.moe_fused = True  # as compute.py sets it on a CUDA run of such a model
    batch = next(iter(_loader(1)))
    batch = {k: v.to(trainer.device) for k, v in batch.items()}
    with torch.no_grad():
        fused = trainer._forward_blockwise_adapter_model(batch)
        own = trainer.model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
    torch.testing.assert_close(fused, own, rtol=1e-5, atol=1e-6)


def test_eval_scores_against_the_experts_the_run_held(
    validated, tmp_path, monkeypatch, capsys, cpu_only
):
    """``poros eval <run>`` rebuilds the run's engine from its record, the expert format
    included: a run that held a released checkpoint's experts as nf4 is scored on nf4 experts,
    with POROS_INKLING_EXPERTS set to released at eval time. Its base loss is the run's
    eval_loss_start and its trained loss the run's eval_loss, bit for bit."""
    import json

    from poros import runs
    from poros.cli import app, contract, train_cmd
    from poros.cli.context import Context
    from poros.cli.verbs import evaluate
    from poros.core.config import PorosConfig
    from tests.cli.test_engine_loop_held_out import TextTok, _text

    class SmallTok(TextTok):  # ids inside the tiny model's vocabulary
        def enc(self, s):
            return [2 + ord(c) % (tiny.VOCAB - 2) for c in s]

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    for name in ("POROS_DEV_SIMULATE", "POROS_AGENT", "POROS_INKLING_EXPERTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("poros.core.tokenizer.load_tokenizer", lambda *a, **k: SmallTok())

    def ready(cfg, **kw):  # the checks before the load, on a machine with no GPU
        from poros import recipes

        choice = recipes.choose(cfg)
        train_cmd._write_resolved(cfg, choice)
        return train_cmd._Ready(choice, "inkling2_loop", None, False, False, "CPU", 8.0, "")

    monkeypatch.setattr(train_cmd, "_before_the_load", ready)
    monkeypatch.setattr(train_cmd, "_download_note", lambda cfg: None)
    monkeypatch.setattr(train_cmd, "_checkpoint_note", lambda cfg, model_config: None)
    monkeypatch.setattr(evaluate, "_ready", lambda cfg, ctx: None)

    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32)
    rid = runs.new_id()
    folder = runs.create(rid, {"model": str(base), "data": "notes.jsonl", "steps": 3})
    runs.update(folder, state="running", started_at=runs.now())
    data, held = _text(tmp_path / "notes.jsonl", 12), _text(tmp_path / "held.jsonl", 3, seed=50)
    cfg = PorosConfig(
        model_name_or_path=str(base),
        dataset=str(data),
        eval_dataset=str(held),
        output_dir=str(folder),
        torch_dtype="float32",
        moe_expert_format="nf4",
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,
        warmup_steps=0,
        save_steps=3,
        learning_rate=1e-2,
        max_seq_length=16,
        lora_dropout=0.0,
        logging_steps=1,
        progress=False,
    )

    def raise_it(exc):
        raise exc

    monkeypatch.setenv("POROS_OUTPUT", "json")
    metrics = train_cmd.run(
        cfg,
        config_path="run.yaml",
        resume_from=None,
        trust_remote_code=False,
        friendly_error=raise_it,
    )
    runs.update(folder, state="succeeded", finished_at=runs.now())
    capsys.readouterr()
    recorded = json.loads((folder / "config.resolved.json").read_text())
    assert recorded["moe_expert_format"] == "nf4"
    assert metrics["eval_loss_start"] != metrics["eval_loss"]

    monkeypatch.setenv("POROS_INKLING_EXPERTS", "released")  # the other value: not read
    try:
        code = app.dispatch(
            ["eval", rid], Context(cwd=tmp_path, fixed_cached=frozenset(), fixed_interactive=False)
        )
    finally:
        contract.reset()
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    scored = json.loads(out[-1])
    assert code == 0, scored
    assert scored["base"]["loss"] == metrics["eval_loss_start"]
    assert scored["trained"]["loss"] == metrics["eval_loss"]


def test_eval_rebuilds_the_recorded_expert_format(tmp_path, monkeypatch):
    """The settings eval rebuilds keep the run's moe_expert_format, and mark it as the run's own
    record, which POROS_INKLING_EXPERTS does not contradict."""
    import json
    from types import SimpleNamespace

    from poros.cli.verbs.evaluate_pass import settings
    from poros.moe.inkling2 import formats

    (tmp_path / "config.resolved.json").write_text(
        json.dumps({"model_name_or_path": "x/y", "moe_expert_format": "nf4"})
    )
    cfg, _ = settings(SimpleNamespace(output_dir=str(tmp_path), id="r1"), tmp_path)
    assert cfg.moe_expert_format == "nf4"
    monkeypatch.setenv("POROS_INKLING_EXPERTS", "released")
    assert formats.asked_format(cfg) == "nf4"


def test_a_run_from_an_adapter_holds_its_experts_as_it_did(tmp_path):
    """``--from`` (init_adapter): a new run holds the experts as the adapter trained against
    them, where it says nothing; one that says the other format is refused, as another revision
    is, before the download; a record that is neither format does not read."""
    import json

    from poros.core.config import PorosConfig
    from poros.preflight.resume import start_from

    start = tmp_path / "earlier"
    start.mkdir()
    said = {"model_name_or_path": "org/inkling", "lora_rank": 16, "moe_expert_format": "nf4"}
    (start / "poros_config.json").write_text(json.dumps(said))
    torch.save({}, start / "adapter_weights.pt")

    cfg = PorosConfig(model_name_or_path="org/inkling", init_adapter=str(start))
    got = start_from(cfg)
    assert got.clash is None and cfg.moe_expert_format == "nf4"
    other = PorosConfig(
        model_name_or_path="org/inkling", init_adapter=str(start), moe_expert_format="released"
    )
    clash = start_from(other).clash
    assert clash is not None and "held as nf4" in str(clash) and "'nf4'" in clash.fix
    (start / "poros_config.json").write_text(json.dumps({**said, "moe_expert_format": "fp8"}))
    assert "neither released nor nf4" in str(
        start_from(PorosConfig(model_name_or_path="org/inkling", init_adapter=str(start))).clash
    )


def _unreadable_names(monkeypatch, files=True):
    """Before the download, the tensor names cannot be read (offline, nothing cached), and,
    with ``files`` False, nor the file list."""
    from poros.moe.inkling2 import formats

    monkeypatch.setattr(formats, "checkpoint_names", lambda *a, **k: None)
    if not files:
        monkeypatch.setattr(formats, "_files", lambda *a, **k: None)


def test_a_quantized_checkpoint_with_only_hf_quant_config_is_sized_released(tmp_path, monkeypatch):
    """Some quantized checkpoints carry no quantization_config in config.json, only an
    hf_quant_config.json beside it (the tiny checkpoint, a slice of the released one). With the
    names unreadable, the checks that read no weights still size the experts as released."""
    import json
    from types import SimpleNamespace

    from poros.moe.inkling2 import formats

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base)
    config = json.loads((base / "config.json").read_text())
    assert "quantization_config" not in config
    run = SimpleNamespace(model_name_or_path=str(base), model_revision=None, moe_expert_format=None)
    _unreadable_names(monkeypatch)
    assert formats.expected(run, config) == ("nf4", None)  # no marker: judged unquantized
    (base / "hf_quant_config.json").write_text(json.dumps({"quantization": {}}))
    assert formats.expected(run, config) == ("released", None)


def test_where_nothing_can_be_read_the_check_says_it_cannot_know_yet(
    validated, tmp_path, monkeypatch
):
    """Names and file list unreadable before the download: the checks size released (the
    larger) and say in words that they cannot tell yet; the trainer tells the load nothing, the
    load decides from its files, and the run records what it held."""
    import json
    from types import SimpleNamespace

    from poros.moe.inkling2 import formats
    from poros.train.trainer import _expert_format

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32, quantize=False)
    config = json.loads((base / "config.json").read_text())
    run = SimpleNamespace(model_name_or_path=str(base), model_revision=None, moe_expert_format=None)
    _unreadable_names(monkeypatch, files=False)
    fmt, note = formats.expected(run, config)
    assert (fmt, note) == ("released", formats.NOT_YET) and "cannot tell yet" in note
    sized_told = _expert_format(
        SimpleNamespace(model_name_or_path=str(base), model_revision=None, moe_expert_format=None)
    )
    assert sized_told == ("released", None)
    trainer = _trainer(base, tmp_path / "run", torch_dtype="float32")
    assert trainer.config.moe_expert_format == "nf4"  # what the load held, from its files


def test_a_leftover_quantization_config_on_an_unquantized_base_trains_offline(
    validated, tmp_path, monkeypatch
):
    """An unquantized base whose config.json still names a quantization config, offline (its
    names unreadable before the download): nothing is refused, since the user asked for nothing;
    the load sees unquantized experts, holds them in NF4 and the run records nf4. Asking for
    released of it is refused in words that name both settings."""
    import json

    from poros.core.errors import PorosConfigError

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32, quantize=False)
    config = json.loads((base / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "modelopt"}
    (base / "config.json").write_text(json.dumps(config))
    _unreadable_names(monkeypatch)
    trainer = _trainer(base, tmp_path / "run", torch_dtype="float32")
    assert trainer.config.moe_expert_format == "nf4"
    with pytest.raises(PorosConfigError) as err:
        _trainer(base, tmp_path / "again", torch_dtype="float32", moe_expert_format="released")
    words = str(err.value) + str(err.value.info.remediation)
    assert "there are none to keep" in words
    assert "Unset moe_expert_format and POROS_INKLING_EXPERTS" in words


@pytest.mark.parametrize("broken", ["index_without_weight_map", "index_not_json", "header"])
def test_a_checkpoint_whose_names_do_not_read_is_refused_naming_the_file(
    validated, tmp_path, monkeypatch, broken
):
    """A local folder whose model.safetensors.index.json has no weight_map (or is no JSON), or
    whose safetensors header is corrupt: the trainer and the export's early check refuse it in
    words that name the file, before anything loads, not with a KeyError or a JSON error."""
    from poros.core.errors import PorosConfigError
    from poros.moe.inkling2.export import export_merged

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32)
    index = base / "model.safetensors.index.json"
    if broken == "index_without_weight_map":
        index.write_text('{"metadata": {}}')
        bad = index
    elif broken == "index_not_json":
        index.write_text("{not json")
        bad = index
    else:
        index.unlink()
        bad = sorted(base.glob("model-*.safetensors"))[1]
        data = bytearray(bad.read_bytes())
        data[8:24] = b"\x00garbage\xff\xfe!!!!!!!"
        bad.write_bytes(bytes(data))
    with pytest.raises(PorosConfigError, match=str(bad.name)) as err:
        _trainer(base, tmp_path / "run", torch_dtype="float32")
    assert "cannot be read" in str(err.value) and str(bad) in str(err.value)
    with pytest.raises(PorosConfigError, match=str(bad.name)):
        export_merged(
            str(base),
            {},
            str(tmp_path / "out"),
            scaling=2.0,
            dtype=torch.float32,
            expert_codec="released",
        )
    assert not (tmp_path / "out").exists()
