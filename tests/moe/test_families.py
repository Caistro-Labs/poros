"""The MoE families beyond qwen3_moe (``MOE_FAMILIES``) on tiny random models, on CPU.

Per family: the streamed loader is byte-identical to ``from_pretrained``
plus ``prepare_moe``'s quantization, the managed block computes the HF
sparse block's forward on the same (NF4 round-tripped) expert weights, and
streamed and resident ``prepare_moe`` train bitwise equal. Arch
certification is the GPU gate (``scripts/moe_absolute_gate.py``).
"""

from __future__ import annotations

import copy
import types

import pytest

from tests.conftest import roomy_disk_on_a_mac
from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

import transformers  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from poros.moe.arch import MOE_FAMILIES, MoeArch  # noqa: E402
from poros.moe.load import load_moe_streamed  # noqa: E402
from poros.moe.modules import ManagedExperts, ManagedMoeBlock  # noqa: E402
from poros.moe.prepare import (  # noqa: E402
    MoeConfig,
    default_backends,
    prepare_moe,
    streaming_config,
)
from tests.metal import families as fam  # noqa: E402

_SHARED = dict(vocab_size=96, hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
CONFIGS = {
    # A dense mlp_only layer and a shared expert.
    "qwen2_moe": lambda: transformers.Qwen2MoeConfig(
        **_SHARED,
        intermediate_size=96,
        moe_intermediate_size=40,
        shared_expert_intermediate_size=48,
        num_hidden_layers=3,
        mlp_only_layers=[1],
        num_experts=12,
        num_experts_per_tok=3,
    ),
    "olmoe": lambda: transformers.OlmoeConfig(
        **_SHARED, intermediate_size=40, num_hidden_layers=2, num_experts=12, num_experts_per_tok=3
    ),
    # w1/w2/w3 under block_sparse_moe in the checkpoint; fp32 routing weights.
    "mixtral": lambda: transformers.MixtralConfig(
        **_SHARED,
        intermediate_size=40,
        num_hidden_layers=2,
        num_local_experts=12,
        num_experts_per_tok=2,
    ),
    # A dense first layer, two shared experts, group-limited routing, MLA attention.
    "deepseek_v2": lambda: transformers.DeepseekV2Config(
        **{**_SHARED, "num_key_value_heads": 4},  # MLA: one KV head per query head
        intermediate_size=96,
        moe_intermediate_size=40,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_shared_experts=2,
        n_routed_experts=12,
        num_experts_per_tok=3,
        n_group=3,
        topk_group=2,
        topk_method="group_limited_greedy",
        kv_lora_rank=32,
        q_lora_rank=None,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
    ),
    # The newest checkpoints: real config.json, sizes aside. Gated DeltaNet and a shared expert...
    "qwen3_5_moe": lambda: fam.real_config("qwen3_6_35b_a3b", hidden=64, vocab=96),
    # ... and a decoder layer that routes on its residual, its experts beside a dense mlp.
    "gemma4_moe": lambda: fam.real_config("gemma_4_26b_a4b", hidden=64, vocab=96),
}
# Their checkpoints store each layer's experts fused, as no other family's do.
FUSED = ("qwen3_5_moe", "gemma4_moe")


def test_every_family_is_covered():
    # test_load.py covers qwen3_moe; test_inkling2.py Inkling 2 Mini, whose decoder Poros carries
    # (no transformers class to build it from at this pin).
    assert CONFIGS.keys() == MOE_FAMILIES.keys() - {"qwen3_moe", "qwen3_5_moe_text",
                                                     "inkling2_loop", "inkling2_loop_text"}


def _model(family):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(CONFIGS[family]())
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.05)
    return model


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def checkpoint(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    if request.param in FUSED:
        fam.write_fused(path, CONFIGS[request.param]())
    else:
        _model(request.param).save_pretrained(path, max_shard_size="60KB")  # splits gate/up pairs
    return path


def _store(model):
    return default_backends().quantize_experts(model, MoeArch.from_config(model.config))


def test_streamed_load_is_byte_identical(checkpoint):
    reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
    expected = _store(reference)
    model = load_moe_streamed(str(checkpoint), torch.bfloat16)
    store = _store(model)

    assert store.layout == expected.layout and store.tensor_shapes == expected.tensor_shapes
    for key, slab in expected._slabs.items():
        assert torch.equal(store._slabs[key], slab), key
    ref_params = dict(reference.named_parameters())
    params = {n: p for n, p in model.named_parameters() if not p.is_meta}
    assert params.keys() == {n for n in ref_params if ".experts." not in n}
    for name, param in params.items():
        assert torch.equal(param, ref_params[name]), name
    ref_buffers = dict(reference.named_buffers())
    for name, buf in model.named_buffers():
        assert torch.equal(buf, ref_buffers[name]), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("family", sorted(CONFIGS))
def test_managed_block_computes_the_hf_block(family, dtype):
    if family == "gemma4_moe":  # no block: test_managed_experts_compute_the_hf_experts
        pytest.skip("its layer routes for itself")
    model = _model(family).to(dtype)
    blocks = {
        name: copy.deepcopy(module.mlp)
        for name, module in model.model.layers.named_children()
        if hasattr(module.mlp, "experts")
    }
    report = prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    shared = [n for n in dict(model.named_parameters()) if ".shared_expert" in n]
    assert all(MoeArch.from_config(model.config).classify_parameter(n) == "dense" for n in shared)
    assert report.params_classified["dense"] > 0
    hidden = torch.randn(2, 64, 64, dtype=dtype)
    for ordinal, (name, reference) in enumerate(blocks.items()):
        managed = model.model.layers.get_submodule(f"{name}.mlp")
        assert isinstance(managed, ManagedMoeBlock)
        managed.residency.ensure([(ordinal, e) for e in range(managed.num_experts)])
        with torch.no_grad():  # the weights training sees: the NF4 round trip
            for tensor in ("gate_up_proj", "down_proj"):
                for e in range(managed.num_experts):
                    key = (ordinal, e)
                    slab = managed.residency.resident_qbytes(key)
                    weight = managed.store.dequantize_tensor(key, tensor, slab)
                    getattr(reference.experts, tensor)[e] = weight
            # bf16 sums round in a thread-count-dependent order: torch's bf16 defaults
            tol = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else {}
            torch.testing.assert_close(managed(hidden), reference(hidden), **tol)


def test_managed_experts_compute_the_hf_experts():
    model = _model("gemma4_moe").to(torch.float32)
    layers = model.model.language_model.layers
    reference = [copy.deepcopy(layer.experts) for layer in layers]
    report = prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    assert report.blocks_installed == len(layers)
    hidden = torch.randn(32, 64)
    for ordinal, (layer, fused) in enumerate(zip(layers, reference, strict=True)):
        managed = layer.experts
        assert isinstance(managed, ManagedExperts) and layer.adapter is managed.adapter
        assert not [n for n, _ in managed.named_parameters() if "lora" in n]  # the layer's to hold
        managed.residency.ensure([(ordinal, e) for e in range(managed.num_experts)])
        with torch.no_grad():
            for tensor in ("gate_up_proj", "down_proj"):
                for e in range(managed.num_experts):
                    key = (ordinal, e)
                    slab = managed.residency.resident_qbytes(key)
                    weight = managed.store.dequantize_tensor(key, tensor, slab)
                    getattr(fused, tensor)[e] = weight
            _, weights, picks = layer.router(hidden)
            base = fused.forward(hidden, picks, weights)
            torch.testing.assert_close(managed(hidden, picks, weights), base, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", sorted(CONFIGS))
def test_streamed_trains_bitwise_equal_to_resident(family, cpu_bf16_backward):
    ids = torch.randint(0, 96, (2, 16), generator=torch.Generator().manual_seed(1))
    runs = []
    for config in (
        streaming_config(CONFIGS[family](), lora_dropout=0.05),
        MoeConfig(streamed=False, look_ahead=0, lora_dropout=0.05),
    ):
        model = _model(family).to(torch.bfloat16)
        prepare_moe(model, config)
        model.train()
        torch.manual_seed(2)  # both arms draw the same dropout masks
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        runs.append([loss.detach(), *(p.grad for p in model.parameters() if p.requires_grad)])
    streamed, resident = runs
    assert len(streamed) == len(resident) > 1
    assert all(torch.equal(s, r) for s, r in zip(streamed, resident, strict=True))


def test_mixtral_router_jitter_is_refused():
    config = CONFIGS["mixtral"]()
    config.router_jitter_noise = 0.1
    with pytest.raises(ValueError, match="router_jitter_noise"):
        MoeArch.from_config(config)


# The families validated for poros.moe (mixtral stays gated): every native
# entry point routes them, as it does qwen3_moe (test_native_entry.py).
VALIDATED = ("deepseek_v2", "olmoe", "qwen2_moe", *FUSED)


def _loader(n, seq=16):
    ids = torch.randint(0, 96, (n, seq), generator=torch.Generator().manual_seed(1))
    rows = [{"input_ids": x, "labels": x.clone()} for x in ids]
    return torch.utils.data.DataLoader(rows, batch_size=1)


def _trainer(checkpoint, output_dir, **overrides):
    from poros.train.trainer import PorosTrainer

    knobs = dict(output_dir=str(output_dir), max_steps=2, save_steps=0, progress=False)
    return PorosTrainer.from_pretrained(str(checkpoint), **{**knobs, **overrides})


@pytest.mark.parametrize("family", VALIDATED)
def test_trainer_resumes_bitwise(family, tmp_path):
    """The trainer routes the family to poros.moe; a resumed run ends where the
    uninterrupted one did, at the default dropout (0.05)."""
    base = tmp_path / "base"
    _model(family).to(torch.bfloat16).save_pretrained(base)
    full = _trainer(base, tmp_path, max_steps=4, save_steps=2, gradient_accumulation_steps=2)
    assert full.block_manager is None and full.model._poros_moe_store is not None
    full.train(_loader(3))
    resumed = _trainer(base, tmp_path, max_steps=4, gradient_accumulation_steps=2)
    resumed.train(_loader(3), resume_from=str(tmp_path / "checkpoint-2"))
    assert resumed.history.losses == full.history.losses[3:]
    after = dict(resumed.model.named_parameters())
    trained = [(n, p) for n, p in full.model.named_parameters() if p.requires_grad]
    assert trained and all(".adapter.lora_" in n for n, _ in trained)
    for name, param in trained:
        assert torch.equal(param, after[name]), name


@pytest.mark.parametrize("family", VALIDATED)
def test_export_merges_only_the_routed_experts(family, tmp_path, monkeypatch):
    """``poros export`` folds the LoRA into the routed experts; shared experts
    and dense layers are copied unchanged, and transformers reloads the result
    with the trained model's logits (fp32, so the merge is the only change)."""
    roomy_disk_on_a_mac(monkeypatch)  # a CUDA run's adapter, merged on a Mac too
    from safetensors.torch import load_file
    from typer.testing import CliRunner

    from poros.cli.main import app

    base = tmp_path / "base"
    if family in FUSED:  # fused, as the real ones store it, in fp32 to test the merge
        fam.write_fused(base, CONFIGS[family](), dtype=torch.float32)
    else:
        _model(family).save_pretrained(base, max_shard_size="60KB")
    run = tmp_path / "run"
    trainer = _trainer(base, run, torch_dtype="float32", learning_rate=1e-2, warmup_steps=0)
    trainer.train(_loader(8))
    trainer.save(str(run))

    out = tmp_path / "merged"
    res = CliRunner().invoke(app, ["export", str(run), "-o", str(out)])
    assert res.exit_code == 0, res.output
    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).to(trainer.device)
    ids = next(iter(_loader(1)))["input_ids"].to(trainer.device)
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model(input_ids=ids).logits
        assert torch.allclose(merged(input_ids=ids).logits, expected, atol=1e-5)
    for shard in sorted(p.name for p in base.glob("*.safetensors")):
        before, after = load_file(base / shard), load_file(out / shard)
        assert before.keys() == after.keys()
        for name, tensor in before.items():
            routed = ".experts." in name and not name.startswith("mtp.")
            assert routed != torch.equal(tensor, after[name]), name


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="dense streaming needs CUDA")
@pytest.mark.parametrize("family", VALIDATED)
def test_cuda_streamed_bf16_is_validated(family):
    model = _model(family).to(torch.bfloat16)
    report = prepare_moe(model, streaming_config(model.config, device="cuda"))
    assert report.guarantee == "validated_bitwise", report.warnings


def test_mixtral_stays_gated(tmp_path):
    from poros.core.errors import PorosArchNotValidatedError

    _model("mixtral").save_pretrained(tmp_path / "base")
    with pytest.raises(PorosArchNotValidatedError):
        _trainer(tmp_path / "base", tmp_path / "run")


def test_a_remote_code_config_is_refused():
    """DeepSeek-V2's Hub config (trust_remote_code=True) names its expert count
    n_routed_experts, with no num_experts alias, and pairs with per-expert modules."""

    class HubDeepseekV2Config(transformers.PretrainedConfig):
        model_type = "deepseek_v2"

    config = HubDeepseekV2Config(num_hidden_layers=2, n_routed_experts=12)
    with pytest.raises(ValueError, match="trust_remote_code"):
        MoeArch.from_config(config)


def test_the_gate_trains_on_a_padded_batch_when_given_one():
    from poros.moe.gates import _train_adapter_steps

    seen = []

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, **inputs):
            seen.append(inputs)
            return types.SimpleNamespace(loss=(self.weight * 2).sum())

    ids = torch.tensor([[5, 6, 0]])
    mask = torch.tensor([[1, 1, 0]])
    labels = torch.tensor([[5, 6, -100]])
    _train_adapter_steps(Recorder(), ids, steps=1, lr=0.1)
    assert seen[-1]["labels"] is ids and "attention_mask" not in seen[-1]
    _train_adapter_steps(Recorder(), ids, steps=1, lr=0.1, attention_mask=mask, labels=labels)
    assert seen[-1]["attention_mask"] is mask and seen[-1]["labels"] is labels
