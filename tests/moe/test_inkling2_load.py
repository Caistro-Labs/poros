"""``load_moe_streamed`` on an Inkling 2 Mini checkpoint in the released layout.

The checkpoint: the tiny model (tests/moe/inkling2_tiny.py) written in the released layout and
names (``codec.to_checkpoint``), its experts encoded in NVFP4 and its last layer's
in MXFP8 as the release stores them, sharded small so that tensors which make one parameter
land in different shards, with the released config's wrapper (``model_type: inkling2_loop``) and
a vision tensor the decoder does not read. The streamed load must give exactly what decoding the
whole checkpoint and quantizing it gives: every expert slab byte for byte, every other tensor
equal, the official fp32 modules in fp32.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()


from poros.moe.arch import MoeArch  # noqa: E402
from poros.moe.inkling2 import codec  # noqa: E402
from poros.moe.inkling2.modeling import FP32_MODULES, Inkling2LoopForCausalLM, cast  # noqa: E402
from poros.moe.load import load_moe_streamed  # noqa: E402
from poros.moe.prepare import default_backends  # noqa: E402
from tests.moe import inkling2_tiny as tiny  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("inkling2")
    return path, tiny.write_released(path)


def _reference(tensors, dtype):
    """The whole checkpoint decoded at once (codec.convert_state_dict), the model cast as the
    official one is, the experts quantized by prepare_moe's backend."""
    model = Inkling2LoopForCausalLM(tiny.config())
    state = codec.convert_state_dict(
        tensors.__getitem__,
        list(tensors),
        unpadded_vocab_size=model.config.unpadded_vocab_size,
        dtype=dtype,
    )
    model.load_state_dict(state)
    cast(model, dtype)
    return model, default_backends().quantize_experts(model, MoeArch.from_config(model.config))


def test_the_nf4_route_is_the_decoded_checkpoint_quantized(checkpoint, monkeypatch):
    monkeypatch.setenv("POROS_INKLING_EXPERTS", "nf4")
    path, tensors = checkpoint
    reference, expected = _reference(tensors, torch.bfloat16)
    model = load_moe_streamed(str(path), torch.bfloat16)
    assert isinstance(model, Inkling2LoopForCausalLM)
    store = default_backends().quantize_experts(model, MoeArch.from_config(model.config))
    assert store.layout == expected.layout and store.tensor_shapes == expected.tensor_shapes
    for key, slab in expected._slabs.items():
        assert torch.equal(store._slabs[key], slab), key
    ref = dict(reference.named_parameters())
    params = {n: p for n, p in model.named_parameters() if not p.is_meta}
    assert params.keys() == {n for n in ref if ".mlp.experts." not in n}
    for name, param in params.items():
        assert param.dtype == ref[name].dtype and torch.equal(param, ref[name]), name
        fp32 = any(key in name for key in FP32_MODULES)
        assert param.dtype == (torch.float32 if fp32 else torch.bfloat16), name
    ref_buffers = dict(reference.named_buffers())
    for name, buf in model.named_buffers():
        assert buf.dtype == ref_buffers[name].dtype and torch.equal(buf, ref_buffers[name]), name


def test_the_released_route_keeps_the_released_bytes(checkpoint):
    """The default: each expert decodes, from its slab, to exactly what decoding the whole
    checkpoint gives; the NVFP4 and MXFP8 layers keep their own (different) slab sizes."""
    from poros.moe.inkling2.released import ReleasedExpertStore

    path, tensors = checkpoint
    reference, _ = _reference(tensors, torch.bfloat16)
    model = load_moe_streamed(str(path), torch.bfloat16)
    store = model._poros_moe_store
    assert isinstance(store, ReleasedExpertStore)
    last = model.config.num_hidden_layers - 1
    assert store.formats[0] == "nvfp4" and store.formats[last] == "mxfp8"
    assert store.slice_nbytes((0, 0)) < store.slice_nbytes((last, 0))
    for layer, block in enumerate(reference.model.layers):
        for tensor in ("gate_up_proj", "down_proj"):
            want = getattr(block.mlp.experts, tensor)
            for e in range(want.shape[0]):
                slab = torch.empty(store.slice_nbytes((layer, e)), dtype=torch.uint8)
                store.load_slice((layer, e), slab)
                got = store.dequantize_tensor((layer, e), tensor, slab)
                assert torch.equal(got, want[e]), (layer, tensor, e)


def test_released_experts_train_streamed_as_resident(checkpoint, cpu_bf16_backward):
    from poros.moe.prepare import MoeConfig, prepare_moe, streaming_config

    path, _ = checkpoint
    runs = []
    for streamed in (True, False):
        model = load_moe_streamed(str(path), torch.bfloat16)
        store = model._poros_moe_store
        config = (
            streaming_config(model.config, store=store, lora_dropout=0.05)
            if streamed
            else MoeConfig(streamed=False, look_ahead=0, lora_dropout=0.05)
        )
        prepare_moe(model, config)
        tiny.randomize_lora_b(model)
        model.train()
        torch.manual_seed(2)
        ids = tiny.ids()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        runs.append([loss.detach(), *(p.grad for p in model.parameters() if p.requires_grad)])
    assert len(runs[0]) == len(runs[1]) > 1
    assert all(torch.equal(a, b) for a, b in zip(*runs, strict=True))


def test_a_released_expert_decodes_to_what_it_encodes():
    x = torch.randn(2, 4, 64)
    nv = codec.decode_experts(tiny.encode_nvfp4(x), torch.float32)
    mx = codec.decode_experts(tiny.encode_mxfp8(x), torch.float32)
    assert _rel(nv, x) < 0.2 and _rel(mx, x) < 0.05  # the formats' own precision, roughly


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_the_checkpoint_declares_a_looped_decoder(checkpoint, tmp_path):
    """The family gate learns a model is a looped decoder from its decoder class (poros_visits),
    not from a list of names."""
    from poros.moe.load import declares_visits

    path, _ = checkpoint
    assert declares_visits(str(path))
    import transformers

    other = tmp_path / "qwen2"
    transformers.Qwen2Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=96,
    ).save_pretrained(other)
    assert not declares_visits(str(other))


def test_the_ssd_cache_sizes_a_store_by_each_layer(checkpoint, monkeypatch):
    """poros.sss.tune sizes the expert cache from every layer's own slab size: the released
    store's MXFP8 layer is counted at its size, not the first layer's; a store of one slab size
    gives what it always did (layers x experts x that size)."""
    from poros.sss.tune import layer_nbytes, store_nbytes

    path, _ = checkpoint
    released = load_moe_streamed(str(path), torch.bfloat16)._poros_moe_store
    lay = released.layout
    every = sum(released.slice_nbytes((layer, e)) for layer in range(lay.num_layers)
                for e in range(lay.experts_per_layer))
    first = lay.num_layers * lay.experts_per_layer * released.slice_nbytes((0, 0))
    assert store_nbytes(released) == every > first
    last = lay.num_layers - 1
    assert layer_nbytes(released) == lay.experts_per_layer * released.slice_nbytes((last, 0))
    monkeypatch.setenv("POROS_INKLING_EXPERTS", "nf4")
    nf4 = load_moe_streamed(str(path), torch.bfloat16)._poros_moe_store
    n = nf4.layout
    assert store_nbytes(nf4) == n.num_layers * n.experts_per_layer * nf4.slice_nbytes((0, 0))
    assert layer_nbytes(nf4) == n.experts_per_layer * nf4.slice_nbytes((0, 0))
