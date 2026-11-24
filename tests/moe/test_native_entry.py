"""MoE is native: ``PorosTrainer`` and ``poros train`` route it to poros.moe."""

from __future__ import annotations

import pytest

from tests.conftest import as_linux, roomy_disk_on_a_mac
from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.core.errors import PorosConfigError  # noqa: E402
from poros.moe.modules import ManagedMoeBlock  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

VOCAB = 64


def _model():
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=4,
    )
    return Qwen3MoeForCausalLM(config).to(torch.bfloat16)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_moe")
    _model().save_pretrained(path)
    return path


def _loader(n=8, seq=16):
    ids = torch.randint(0, VOCAB, (n, seq), generator=torch.Generator().manual_seed(1))
    rows = [{"input_ids": x, "labels": x.clone()} for x in ids]
    return torch.utils.data.DataLoader(rows, batch_size=1)


def _trainer(checkpoint, tmp_path, **overrides):
    knobs = dict(output_dir=str(tmp_path), max_steps=2, save_steps=0, progress=False)
    return PorosTrainer.from_pretrained(str(checkpoint), **{**knobs, **overrides})


def test_trainer_trains_moe_through_the_streamed_loader(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path, gradient_accumulation_steps=2)
    model = trainer.model
    assert trainer.block_manager is None
    assert model._poros_moe_store is not None  # quantized while loading, not after
    assert sum(isinstance(m, ManagedMoeBlock) for m in model.modules()) == 2
    assert trainer.trainable_parameter_names
    assert all(".mlp.adapter.lora_" in n for n in trainer.trainable_parameter_names)
    metrics = trainer.train(_loader())
    assert metrics["total_steps"] == 2
    assert all(torch.isfinite(torch.tensor(trainer.history.losses)))
    assert any(p.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)


@pytest.mark.parametrize(
    "knob",
    [
        {"lora_target_modules": ["q_proj"]},
        {"lora_bias": "all"},
        {"block_size": 2},
        {"torch_dtype": "float16"},  # fp16 expert adapters: no optimizer or kernel takes them
        {"adam_epsilon": 1e-30},  # under ExpertAdamW's least eps (min_adam_epsilon)
        {"learning_rate": 1.5},  # above ExpertAdamW's largest lr (MAX_LR)
    ],
)
def test_trainer_refuses_dense_only_knobs_before_loading(checkpoint, tmp_path, knob, monkeypatch):
    import poros.moe.load as load

    monkeypatch.setattr(load, "load_moe_streamed", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(PorosConfigError, match=next(iter(knob))):
        _trainer(checkpoint, tmp_path, **knob)


def test_trainer_refuses_short_host_ram_before_loading(checkpoint, tmp_path, monkeypatch):
    import poros.core.card
    import poros.moe.load as load
    from poros.core.errors import PorosHostRamError

    # the CPU suite's trainer is the CUDA route's (tests/conftest.py): its planner reads this host
    # as that route does, not as a Mac's one pool, which has no host RAM of its own to check
    as_linux(monkeypatch, poros.core.card)
    monkeypatch.setattr(load, "load_moe_streamed", lambda *a, **k: pytest.fail("loaded"))
    monkeypatch.setattr("poros.core.env.host_ram_free", lambda: 0.5)
    with pytest.raises(PorosHostRamError, match=r"0\.5 GiB available"):
        _trainer(checkpoint, tmp_path)  # needs 1.5 GiB: the process, for a tiny model
    monkeypatch.setattr(load, "load_moe_streamed", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):  # reached the load
        _trainer(checkpoint, tmp_path, check_host_ram=False)


def test_resume_from_a_checkpoint_is_bitwise(checkpoint, tmp_path):
    """A run resumed from checkpoint-2 ends bitwise where the uninterrupted run did.

    With dropout on (the default 0.05), so the restored RNG stream is under test.
    """
    full = _trainer(checkpoint, tmp_path, max_steps=4, save_steps=2, gradient_accumulation_steps=2)
    full.train(_loader(n=3))  # 8 micro-batches, so the data wraps around
    resumed = _trainer(checkpoint, tmp_path, max_steps=4, gradient_accumulation_steps=2)
    resumed.train(_loader(n=3), resume_from=str(tmp_path / "checkpoint-2"))
    assert resumed.history.losses == full.history.losses[3:]
    after = dict(resumed.model.named_parameters())
    for name, param in full.model.named_parameters():
        if param.requires_grad:
            assert torch.equal(param, after[name]), name


def test_export_merges_the_experts_into_a_plain_checkpoint(tmp_path, monkeypatch):
    """Train, ``poros export``, reload with transformers: the logits match.

    fp32 so the merge is the only difference; small shards split some
    experts' gate and up halves across files.
    """
    roomy_disk_on_a_mac(monkeypatch)  # a CUDA run's adapter, merged on a Mac too
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from typer.testing import CliRunner

    from poros.cli.main import app
    from poros.moe.export import export_merged

    base = tmp_path / "base"
    _model().float().save_pretrained(base, max_shard_size="100KB")
    run = tmp_path / "run"
    trainer = _trainer(base, run, torch_dtype="float32", learning_rate=1e-2, warmup_steps=0)
    trainer.train(_loader())
    trainer.save(str(run))

    out = tmp_path / "merged"
    res = CliRunner().invoke(app, ["export", str(run), "-o", str(out)])
    assert res.exit_code == 0, res.output
    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).to(trainer.device)
    ids = next(iter(_loader()))["input_ids"].to(trainer.device)
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model(input_ids=ids).logits
        assert torch.allclose(merged(input_ids=ids).logits, expected, atol=1e-5)
    shards = sorted(p.name for p in base.glob("*.safetensors"))
    assert len(shards) > 1 and shards == sorted(p.name for p in out.glob("*.safetensors"))
    for shard in shards:
        before, after = load_file(base / shard), load_file(out / shard)
        assert before.keys() == after.keys()
        for name, tensor in before.items():
            assert (".experts." in name) != torch.equal(tensor, after[name]), name
    with pytest.raises(ValueError, match="base checkpoint itself"):
        export_merged(str(base), {}, str(base), scaling=1.0)

