"""CPU tests for the block-streamed NF4 load plumbing.

The quantization itself requires CUDA (covered by
tests/gpu/test_streamed_load_gate.py and scripts/streamed_load_gate.py);
these tests cover what must hold on any machine: the flag plumbing, the
default path being untouched, and the shard-walking helpers.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


def test_config_default_is_auto():
    from poros.core.config import PorosConfig

    cfg = PorosConfig(model_name_or_path="x")
    assert cfg.nf4_load_strategy == "auto"


def test_auto_refuses_nf4_without_cuda(monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("exercised via GPU suite on CUDA boxes")
    # On a no-CUDA machine "auto" must never invoke the streamed loader --
    # and nf4 must refuse honestly rather than silently training an
    # unquantized model that records "nf4" in its artifacts.
    import poros.train.model as pm
    from poros.core.errors import PorosConfigError

    called = {}

    def boom(*a, **k):
        called["streamed"] = True
        raise RuntimeError("should not be called without CUDA")

    monkeypatch.setattr(
        "poros.train.streamed_load.load_nf4_block_streamed", boom,
        raising=True,
    )

    with pytest.raises(PorosConfigError, match="needs a CUDA GPU"):
        pm._load_model("x", torch.bfloat16, nf4_load_strategy="auto")
    assert "streamed" not in called


def test_unknown_strategy_rejected():
    from poros.train.model import _load_model

    with pytest.raises(ValueError, match="nf4_load_strategy"):
        _load_model("x", torch.bfloat16, nf4_load_strategy="bogus")


def test_streamed_requires_cuda():
    if torch.cuda.is_available():
        pytest.skip("CPU-only check")
    from poros.train.streamed_load import load_nf4_block_streamed

    with pytest.raises(RuntimeError, match="requires CUDA"):
        load_nf4_block_streamed("x", torch.bfloat16)


def test_shard_map_indexed(tmp_path):
    from poros.train.streamed_load import _shard_map

    index = {
        "weight_map": {
            "model.layers.0.q.weight": "model-00001.safetensors",
            "model.layers.1.q.weight": "model-00002.safetensors",
            "model.embed.weight": "model-00001.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    got = _shard_map(str(tmp_path))
    assert sorted(got) == ["model-00001.safetensors", "model-00002.safetensors"]
    assert sorted(got["model-00001.safetensors"]) == [
        "model.embed.weight",
        "model.layers.0.q.weight",
    ]


def test_shard_map_missing_checkpoint(tmp_path):
    from poros.train.streamed_load import _shard_map

    with pytest.raises(FileNotFoundError):
        _shard_map(str(tmp_path))


def test_locate_resolves_nested_attr():
    import torch.nn as nn

    from poros.train.streamed_load import _locate

    model = nn.Sequential()
    model.add_module("block", nn.Linear(2, 2))
    module, attr = _locate(model, "block.weight")
    assert module is model.block and attr == "weight"
    module, attr = _locate(model.block, "bias")
    assert attr == "bias"


# -- Keys a checkpoint holds and the model's class does not define ---------------------------------
#
# transformers 5 loads a checkpoint tensor only when its key is in the model's state dict (every
# parameter and persistent buffer the class defines) and leaves every other key unread
# (``convert_and_load_state_dict_in_model``); ``_keys_to_ignore_on_load_unexpected`` only quiets
# its report. The streamed load must leave the same keys: a key added as a parameter is a tensor
# the standard load does not have (pythia-160m's legacy ``attention.bias``/``masked_bias``, 24 of
# them, failed the load gate so), a bias where the class holds None changes the layer's forward,
# and a non-persistent buffer overwritten from the checkpoint differs from the one the model
# computes.


def _tiny(kind):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    if kind == "gpt_neox":
        config = GPTNeoXConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4,
                               max_position_embeddings=16, rotary_pct=0.25)
        model = GPTNeoXForCausalLM(config)
        causal = torch.tril(torch.ones(16, 16, dtype=torch.uint8)).view(1, 1, 16, 16)
        legacy = (("bias", causal), ("masked_bias", torch.tensor(-1e9)),
                  ("rotary_emb.inv_freq", torch.ones(1)))  # pythia's checkpoint holds all three
        strays = {f"gpt_neox.layers.{i}.attention.{name}": value.clone()
                  for i in range(2) for name, value in legacy}
    else:
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        model = Qwen2ForCausalLM(config)
        strays = {
            "model.layers.0.self_attn.o_proj.bias": torch.ones(32),  # the class holds None
            "model.rotary_emb.inv_freq": torch.zeros(4),  # a buffer the state dict leaves out
        }
    return model, strays


def _checkpoint_with(tmp_path, kind):
    from safetensors.torch import load_file, save_file

    model, strays = _tiny(kind)
    model.save_pretrained(tmp_path)
    tensors = load_file(tmp_path / "model.safetensors")
    assert not set(strays) & set(tensors)
    save_file({**tensors, **strays}, tmp_path / "model.safetensors", metadata={"format": "pt"})


def _streamed_on_cpu(path, dtype):
    """The streamed load's CPU part on its skeleton: the shard walk and the tie, no 4-bit layers."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from poros.train.streamed_load import _load_shards

    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), dtype=dtype)
    _load_shards(model, str(path), dtype)
    model.tie_weights()
    return model


@pytest.mark.parametrize("kind", ["gpt_neox", "qwen2"])
def test_streamed_load_leaves_keys_the_class_does_not_define(tmp_path, kind):
    from transformers import AutoModelForCausalLM

    from poros.validate.load_gate import compare, tensor_state

    _checkpoint_with(tmp_path, kind)
    standard = tensor_state(AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32))
    streamed = tensor_state(_streamed_on_cpu(tmp_path, torch.float32))

    found = compare(standard, streamed)
    assert (found["missing"], found["extra"], found["mismatched"]) == ([], [], [])
    assert found["compared"] == len(standard)


def test_streamed_load_says_which_keys_it_left(tmp_path, caplog):
    """As transformers' load report: one warning naming the keys the class does not list, and
    nothing for the legacy keys its ``_keys_to_ignore_on_load_unexpected`` names or for an old
    per-layer ``rotary_emb.inv_freq``."""
    import logging

    def said():
        return [r for r in caplog.records if r.name == "poros.train.streamed_load"]

    for kind in ("gpt_neox", "qwen2"):
        (tmp_path / kind).mkdir()
        _checkpoint_with(tmp_path / kind, kind)
    with caplog.at_level(logging.WARNING, logger="poros.train.streamed_load"):
        _streamed_on_cpu(tmp_path / "gpt_neox", torch.float32)
        assert not said()
        _streamed_on_cpu(tmp_path / "qwen2", torch.float32)
    (record,) = said()  # the rotary buffer's checkpoint copy is quiet, as in transformers
    assert "1 checkpoint tensor is" in record.getMessage()
    assert "model.layers.0.self_attn.o_proj.bias" in record.getMessage()
    assert "inv_freq" not in record.getMessage()


def test_assign_never_adds_a_tensor_the_class_does_not_define():
    import torch.nn as nn

    from poros.train.streamed_load import _assign

    linear = nn.Linear(2, 2, bias=False)
    linear.register_buffer("scratch", torch.zeros(2), persistent=False)
    for attr in ("bias", "legacy", "scratch"):  # None, absent, outside the state dict
        with pytest.raises(RuntimeError, match="defines no"):
            _assign(linear, attr, torch.ones(2), torch.float32)
    linear.register_buffer("kept", torch.zeros(2))
    _assign(linear, "kept", torch.ones(2, dtype=torch.float64), torch.float32)
    _assign(linear, "weight", torch.full((2, 2), 3.0), torch.float32)
    assert torch.equal(linear.kept, torch.ones(2)) and linear.kept.dtype == torch.float32
    assert torch.equal(linear.weight, torch.full((2, 2), 3.0))
    assert set(dict(linear.named_parameters())) == {"weight"}
    assert set(dict(linear.named_buffers())) == {"scratch", "kept"}


# nf4_load_strategy='auto' falls back to the standard loader only on an input the streamed load
# cannot stream byte-exactly; any other failure is the run's own, with its own cause (an OOM
# retried as a whole-model GPU load would fail again, misleadingly).
def test_auto_falls_back_only_on_an_unsupported_input():
    from poros.train.model import _streamed_or_none
    from poros.train.streamed_load import StreamedLoadUnsupported, _NoSafetensors

    def unsupported():
        raise StreamedLoadUnsupported("non-renaming weight transform")

    assert _streamed_or_none(unsupported) is None
    assert issubclass(_NoSafetensors, FileNotFoundError)  # a missing file to any other caller
    assert _streamed_or_none(lambda: (_ for _ in ()).throw(_NoSafetensors("no .safetensors"))) \
        is None
    for real in (torch.OutOfMemoryError("CUDA out of memory"), OSError("couldn't connect"),
                 RuntimeError("block-streamed load left parameters unmaterialized")):
        def fails(e=real):
            raise e
        with pytest.raises(type(real)):
            _streamed_or_none(fails)
    assert _streamed_or_none(lambda: "model") == "model"


def test_no_safetensors_is_an_unsupported_input(tmp_path):
    from poros.train.streamed_load import StreamedLoadUnsupported, _shard_map

    with pytest.raises(StreamedLoadUnsupported, match="safetensors"):
        _shard_map(str(tmp_path))
