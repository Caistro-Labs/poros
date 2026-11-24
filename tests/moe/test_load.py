"""``load_moe_streamed``: byte-identical to ``from_pretrained`` + ``prepare_moe``'s quantization."""

from __future__ import annotations

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

import poros.moe.load as load  # noqa: E402
import poros.moe.store as moe_store  # noqa: E402
from poros.moe.arch import MoeArch  # noqa: E402
from poros.moe.contracts import ExpertLayout  # noqa: E402
from poros.moe.load import load_moe_streamed  # noqa: E402
from poros.moe.prepare import MoeConfig, default_backends, prepare_moe  # noqa: E402
from poros.sss import RamSink  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """12 experts (natural vs lexical order differ), a dense mlp_only layer,
    an all-zero expert tensor, and shards small enough to split experts'
    gate and up halves apart."""
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=40,
        num_hidden_layers=4,
        mlp_only_layers=[2],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=12,
        num_experts_per_tok=3,
    )
    model = Qwen3MoeForCausalLM(config)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.05)
        model.model.layers[0].mlp.experts.down_proj[5].zero_()
    path = tmp_path_factory.mktemp("qwen3_moe")
    model.save_pretrained(path, max_shard_size="60KB")
    return path


def _reference(path, dtype):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cpu")
    return model, default_backends().quantize_experts(model, MoeArch.from_config(model.config))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_byte_identical_to_from_pretrained(checkpoint, dtype, monkeypatch):
    reference, expected = _reference(checkpoint, dtype)
    monkeypatch.setattr(RamSink, "_CHUNK_BYTES", 5 * expected.slice_nbytes((0, 0)) + 1)
    model = load_moe_streamed(str(checkpoint), dtype)
    store = default_backends().quantize_experts(model, MoeArch.from_config(model.config))

    assert store.layout == expected.layout
    assert store.tensor_shapes == expected.tensor_shapes
    assert store.compute_dtype == expected.compute_dtype == dtype
    for key, slab in expected._slabs.items():
        assert torch.equal(store._slabs[key], slab), key
    ref_params = dict(reference.named_parameters())
    params = {n: p for n, p in model.named_parameters() if not p.is_meta}
    assert params.keys() == {n for n in ref_params if ".mlp.experts." not in n}
    for name, param in params.items():
        assert param.dtype == ref_params[name].dtype and torch.equal(param, ref_params[name]), name
    ref_buffers = dict(reference.named_buffers())
    for name, buf in model.named_buffers():
        assert torch.equal(buf, ref_buffers[name]), name


def test_leaves_keys_the_class_does_not_define(checkpoint, tmp_path):
    """from_pretrained never reads a checkpoint key the model does not define, and nor does the
    streamed load: a key absent on the class, a bias where it holds None, a buffer outside the
    state dict. Before, each became a parameter or overwrote the buffer."""
    import json
    import shutil

    from safetensors.torch import save_file

    path = tmp_path / "strays"
    shutil.copytree(checkpoint, path)
    strays = {
        "model.layers.0.self_attn.bias": torch.ones(1, 1, 8, 8, dtype=torch.uint8),
        "model.layers.1.self_attn.o_proj.bias": torch.ones(64),
        "model.rotary_emb.inv_freq": torch.zeros(8),
    }
    save_file(strays, path / "strays.safetensors", metadata={"format": "pt"})
    index = json.loads((path / "model.safetensors.index.json").read_text())
    index["weight_map"].update(dict.fromkeys(strays, "strays.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(index))

    reference, expected = _reference(path, torch.bfloat16)
    model = load_moe_streamed(str(path), torch.bfloat16)
    store = default_backends().quantize_experts(model, MoeArch.from_config(model.config))
    for key, slab in expected._slabs.items():
        assert torch.equal(store._slabs[key], slab), key
    ref_params = {n: p for n, p in reference.named_parameters() if ".mlp.experts." not in n}
    params = {n: p for n, p in model.named_parameters() if not p.is_meta}
    assert params.keys() == ref_params.keys()
    for name, param in params.items():
        assert param.dtype == ref_params[name].dtype and torch.equal(param, ref_params[name]), name
    ref_buffers = dict(reference.named_buffers())
    buffers = dict(model.named_buffers())
    assert buffers.keys() == ref_buffers.keys()
    for name, buf in buffers.items():
        assert torch.equal(buf, ref_buffers[name]), name


def test_prepare_moe_trains_identically(checkpoint):
    ids = torch.randint(0, 96, (2, 16), generator=torch.Generator().manual_seed(1))
    runs = []
    for model in (_reference(checkpoint, torch.bfloat16)[0], load_moe_streamed(str(checkpoint))):
        report = prepare_moe(model, MoeConfig(streamed=True, device="cpu"))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        runs.append((report.experts_quantized, loss.detach(), grads))
    (n_ref, loss_ref, grads_ref), (n, loss, grads) = runs
    assert n == n_ref and torch.equal(loss, loss_ref)
    assert len(grads) == len(grads_ref) > 0
    assert all(torch.equal(g, r) for g, r in zip(grads, grads_ref, strict=True))


def test_arena_refuses_unwritten_tensors():
    layout = ExpertLayout(
        num_layers=1, experts_per_layer=2, top_k=1, tensor_names=("gate_up_proj", "down_proj")
    )
    arena = moe_store.ExpertSlabArena(layout, {"gate_up_proj": (4, 64), "down_proj": (64, 2)})
    arena.write((0, 0), "gate_up_proj", torch.ones(4, 64))
    with pytest.raises(ValueError, match="never written"):
        arena.build(torch.float32)


def test_refuses_an_unknown_expert_conversion(checkpoint, monkeypatch):
    monkeypatch.setattr(load, "_FUSIONS", {})
    with pytest.raises(RuntimeError, match="converts the MoE expert weights"):
        load_moe_streamed(str(checkpoint))


def test_refuses_a_missing_gate_or_up_half(checkpoint, monkeypatch):
    real = load._shard_map
    dropped = "model.layers.0.mlp.experts.3.up_proj.weight"

    def without_up(path):
        return {shard: [n for n in names if n != dropped] for shard, names in real(path).items()}

    monkeypatch.setattr(load, "_shard_map", without_up)
    with pytest.raises(RuntimeError, match="lacks the gate or up half"):
        load_moe_streamed(str(checkpoint))
