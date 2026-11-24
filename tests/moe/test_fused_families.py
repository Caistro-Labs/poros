"""Qwen3.5/3.6 MoE and Gemma 4 MoE on tiny random models, on CPU, as the Hub stores them.

Both ship their experts already fused, under a multimodal prefix, beside tensors the model
does not load (vision, Qwen3.6's MTP head). tests/moe/test_families.py runs them from their
real config.json; this file runs the Hub's checkpoint layout. Arch certification is the GPU
gate (``scripts/moe_absolute_gate.py``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

import transformers  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from poros.core.arch_registry import registry_type, validated_arch_types  # noqa: E402
from poros.moe.arch import MoeArch  # noqa: E402
from poros.moe.export import export_merged  # noqa: E402
from poros.moe.load import load_moe_streamed  # noqa: E402
from poros.moe.modules import ManagedExperts, ManagedMoeBlock  # noqa: E402
from poros.moe.prepare import (  # noqa: E402
    MoeConfig,
    default_backends,
    prepare_moe,
    streaming_config,
)

_SHARED = dict(vocab_size=96, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
               num_hidden_layers=4, intermediate_size=96, tie_word_embeddings=False,
               num_experts=12)


def _qwen():
    return transformers.Qwen3_5MoeTextConfig(
        **_SHARED, moe_intermediate_size=40, shared_expert_intermediate_size=48,
        num_experts_per_tok=3, head_dim=16, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, full_attention_interval=2,
        layer_types=["linear_attention", "full_attention"] * 2,
    )


def _gemma_text():
    return transformers.Gemma4TextConfig(
        **_SHARED, moe_intermediate_size=40, top_k_experts=3, enable_moe_block=True,
        head_dim=16, global_head_dim=16,
    )


def _gemma():  # the wrapper the Hub's config.json names, as AutoModelForCausalLM builds it
    vision = transformers.Gemma4VisionConfig(
        hidden_size=16, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, global_head_dim=8,
    )
    return transformers.Gemma4Config(text_config=_gemma_text(), vision_config=vision)


CONFIGS = {"qwen": _qwen, "gemma": _gemma}


def _model(name, dtype=torch.bfloat16):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(CONFIGS[name]())
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.05)
    return model.to(dtype).eval()


def _hub_checkpoint(path, name):
    """The layout on the Hub: fused experts, the multimodal prefix, and tensors not loaded."""
    model = _model(name)
    tensors = {}
    for key, value in model.state_dict().items():
        if name == "qwen":
            key = key.replace("model.", "model.language_model.", 1)
        tensors[key] = value.clone()
    if name == "qwen":
        tensors["model.visual.blocks.0.weight"] = torch.zeros(2, dtype=torch.bfloat16)
        tensors["mtp.layers.0.mlp.experts.down_proj"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    config = model.config.to_dict()
    if name == "qwen":
        config = {"model_type": "qwen3_5_moe", "text_config": config,
                  "architectures": ["Qwen3_5MoeForConditionalGeneration"]}
    (path / "config.json").write_text(json.dumps(config))
    return model


def test_gemma_4_routes_by_its_moe_block_not_its_model_type():
    dense = {"model_type": "gemma4", "text_config": {"enable_moe_block": False}}
    moe = {"model_type": "gemma4", "text_config": {"enable_moe_block": True}}
    assert registry_type(dense) == "gemma4"
    assert registry_type(moe) == "gemma4_moe"
    assert registry_type({"model_type": "gemma4_text", "enable_moe_block": True}) == "gemma4_moe"
    assert registry_type({"model_type": "qwen2"}) == "qwen2"
    with pytest.raises(ValueError, match="'gemma4_text'"):  # dense: the dense engine's
        # (transformers' bare Gemma4TextConfig is the per-layer-input variant: gemma4_per_layer)
        MoeArch.from_config(transformers.Gemma4TextConfig(enable_moe_block=False,
                                                          hidden_size_per_layer_input=0))


@pytest.mark.parametrize("from_hub", [False, True])
def test_the_cli_routes_gemma_4_by_its_moe_block(from_hub, tmp_path, monkeypatch):
    from poros.preflight.arch import resolve_model_type as _resolve_model_type

    path = tmp_path / "config.json"
    path.write_text(json.dumps(_gemma().to_dict()))
    if from_hub:
        monkeypatch.setattr("poros.plan.configs.fetch_config", lambda model, revision=None: path)
    assert _resolve_model_type("org/model" if from_hub else str(path))[0] == "gemma4_moe"


def test_they_are_validated_for_their_own_path_alone():
    moe = validated_arch_types("moe")
    assert {"qwen3_5_moe", "qwen3_5_moe_text", "gemma4_moe"} <= moe  # receipts on real layers
    assert not {"qwen3_5_moe", "gemma4_moe"} & validated_arch_types()
    assert "gemma4" in validated_arch_types()  # the dense route is untouched


def test_layouts():
    wrapped = SimpleNamespace(model_type="gemma4", text_config=_gemma_text())
    arch = MoeArch.from_config(wrapped)
    assert arch.experts_module(1) == "model.language_model.layers.1.experts"
    assert arch.router_module(1) == "model.language_model.layers.1.router"
    assert arch.classify_parameter("model.language_model.layers.1.experts.down_proj") == "expert"
    for pinned in ("proj.weight", "scale", "per_expert_scale"):
        name = f"model.language_model.layers.1.router.{pinned}"
        assert arch.classify_parameter(name) == "pinned"
    assert arch.classify_parameter("model.language_model.layers.1.mlp.up_proj.weight") == "dense"
    arch = MoeArch.from_config(SimpleNamespace(model_type="qwen3_5_moe", text_config=_qwen()))
    assert arch.experts_module(1) == "model.layers.1.mlp.experts"
    assert arch.classify_parameter("model.layers.0.linear_attn.in_proj_a.weight") == "dense"
    assert arch.classify_parameter("model.layers.0.linear_attn.A_log") == "dense"
    assert arch.classify_parameter("model.layers.0.mlp.shared_expert_gate.weight") == "dense"


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_streamed_load_of_a_fused_checkpoint_is_byte_identical(name, tmp_path):
    reference = _hub_checkpoint(tmp_path, name)
    expected = default_backends().quantize_experts(
        reference, MoeArch.from_config(reference.config)
    )
    model = load_moe_streamed(str(tmp_path), torch.bfloat16)
    store = model._poros_moe_store
    assert store.layout == expected.layout and store.tensor_shapes == expected.tensor_shapes
    for key, slab in expected._slabs.items():
        assert torch.equal(store._slabs[key], slab), key
    ref_params = dict(reference.named_parameters())
    params = {n: p for n, p in model.named_parameters() if not p.is_meta}
    assert params.keys() == {n for n in ref_params if ".experts." not in n}
    assert all(torch.equal(p, ref_params[n]) for n, p in params.items())


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_managed_model_computes_the_hf_model(name):
    reference = _model(name, torch.float32)  # fp32: only summation order differs
    arch = MoeArch.from_config(reference.config)
    model = _model(name, torch.float32)
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    kind = ManagedExperts if name == "gemma" else ManagedMoeBlock
    managed = [m for m in model.modules() if isinstance(m, kind)]
    assert len(managed) == arch.layout.num_layers
    with torch.no_grad():
        for ordinal, module in enumerate(managed):  # the weights training sees: NF4 round trip
            experts = reference.get_submodule(arch.experts_module(arch.decoder_index(ordinal)))
            module.residency.ensure([(ordinal, e) for e in range(module.num_experts)])
            for tensor in ("gate_up_proj", "down_proj"):
                for e in range(module.num_experts):
                    slab = module.residency.resident_qbytes((ordinal, e))
                    weight = module.store.dequantize_tensor((ordinal, e), tensor, slab)
                    getattr(experts, tensor)[e] = weight
        ids = torch.randint(0, 96, (2, 16), generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(model(input_ids=ids).logits, reference(input_ids=ids).logits,
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_streamed_trains_bitwise_equal_to_resident(name, cpu_bf16_backward):
    ids = torch.randint(0, 96, (2, 16), generator=torch.Generator().manual_seed(1))
    runs = []
    for config in (
        streaming_config(CONFIGS[name](), lora_dropout=0.05),
        MoeConfig(streamed=False, look_ahead=0, lora_dropout=0.05),
    ):
        model = _model(name)
        prepare_moe(model, config)
        model.train()
        torch.manual_seed(2)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        runs.append([loss.detach(), *(p.grad for p in model.parameters() if p.requires_grad)])
    assert len(runs[0]) == len(runs[1]) > 1
    assert all(torch.equal(s, r) for s, r in zip(*runs, strict=True))


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_export_merges_fused_experts_and_copies_the_rest(name, tmp_path):
    from safetensors import safe_open

    source = tmp_path / "base"
    source.mkdir()
    reference = _hub_checkpoint(source, name)
    arch = MoeArch.from_config(reference.config)
    model = load_moe_streamed(str(source), torch.bfloat16)
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    torch.manual_seed(3)
    adapter = {
        n: torch.randn_like(p) * 0.01 for n, p in model.named_parameters() if p.requires_grad
    }
    out = tmp_path / "merged"
    export_merged(str(source), adapter, str(out), scaling=2.0)

    def read(path):
        with safe_open(path / "model.safetensors", framework="pt") as f:
            return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118

    before, after = read(source), read(out)
    assert before.keys() == after.keys()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed == {
        k for k in before
        if k.endswith(("experts.gate_up_proj", "experts.down_proj")) and not k.startswith("mtp.")
    }
    assert after[next(iter(changed))].dtype == torch.bfloat16
    assert arch.layout.num_layers * 2 == len(changed)
