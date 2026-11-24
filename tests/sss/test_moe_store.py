"""The MoE expert store over an SSD source, on CPU.

- ``load_moe_streamed(store="ssd")`` writes a ``.sss`` whose every slab is
  ``torch.equal`` to the RAM build, reuses it without requantizing, and
  rebuilds when the reuse key changes;
- RAM vs SSD training is ``torch.equal`` in losses,
  logits, adapter grads and adapter weights over several steps, with a
  residency small enough to evict and rematerialize, and schedule replay on.

The CUDA gate (dense layers streamed, side-stream copies) is
``scripts/sss/gate_moe_ram_vs_ssd.py``.
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_fadvise
from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

import poros.moe.quant as quant  # noqa: E402
from poros.moe.load import load_moe_streamed, sss_file_path  # noqa: E402
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402
from poros.sss import RamSource, SsdSource  # noqa: E402

VOCAB = 96


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=96, moe_intermediate_size=40,
        num_hidden_layers=4, mlp_only_layers=[2], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_experts=12, num_experts_per_tok=3,
    )
    model = Qwen3MoeForCausalLM(config)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.05)
    path = tmp_path_factory.mktemp("qwen3_moe")
    model.save_pretrained(path, max_shard_size="60KB")
    return path


def _keys(store):
    layout = store.layout
    return [(layer, e) for layer in range(layout.num_layers)
            for e in range(layout.experts_per_layer)]


def _slab(store, key):
    dst = torch.empty(store.slice_nbytes(key), dtype=torch.uint8)
    store.load_slice(key, dst)
    return dst


@needs_fadvise
def test_ssd_store_holds_the_ram_bytes_and_is_reused(checkpoint, tmp_path, monkeypatch):
    ram = load_moe_streamed(str(checkpoint))._poros_moe_store
    model = load_moe_streamed(str(checkpoint), store="ssd", store_path=tmp_path)
    ssd = model._poros_moe_store
    path = sss_file_path(tmp_path, str(checkpoint))
    assert isinstance(ssd.source, SsdSource) and path.exists()
    assert ssd.layout == ram.layout and ssd.tensor_shapes == ram.tensor_shapes
    for key in _keys(ram):
        assert torch.equal(_slab(ssd, key), _slab(ram, key)), key
    ssd.source.close()

    def no_quantize(*_):
        raise AssertionError("a valid .sss must be reused, not requantized")

    monkeypatch.setattr(quant, "quantize_tensor", no_quantize)
    reused = load_moe_streamed(str(checkpoint), store="ssd", store_path=tmp_path)
    for key in _keys(ram):
        assert torch.equal(_slab(reused._poros_moe_store, key), _slab(ram, key)), key
    reused._poros_moe_store.source.close()
    # the RAM store fills from the same file when pointed at it
    warm = load_moe_streamed(str(checkpoint), store="ram", store_path=tmp_path)
    assert isinstance(warm._poros_moe_store.source, RamSource)
    for key in _keys(ram):
        assert torch.equal(_slab(warm._poros_moe_store, key), _slab(ram, key)), key
    # another reuse key (here the source dtype) rebuilds
    with pytest.raises(AssertionError, match="reused, not requantized"):
        load_moe_streamed(str(checkpoint), torch.float32, store="ssd", store_path=tmp_path)
    assert path.exists() and not list(tmp_path.glob("*.tmp"))  # the old file survives


def test_ssd_needs_a_path():
    with pytest.raises(ValueError, match="store_path"):
        load_moe_streamed("unused", store="ssd")
    from poros.stores import PorosStoreNotFoundError

    with pytest.raises(PorosStoreNotFoundError, match="no installed package registers it"):
        load_moe_streamed("unused", store="nvme")


def _train(model, steps, batches, *, mark, trainers=False):
    report = prepare_moe(model, MoeConfig(streamed=True, device="cpu", capacity_bytes=1,
                                          look_ahead=1, lora_dropout=0.0))
    assert report.streamed
    residency = next(m.residency for m in model.modules() if hasattr(m, "residency"))
    # one layer's slices plus look-ahead: forces eviction and rematerialization
    layout = model._poros_moe_store.layout
    residency._capacity = (layout.experts_per_layer + 1) * model._poros_moe_store.slice_nbytes(
        (0, 0))
    params = [p for p in model.parameters() if p.requires_grad]
    if trainers:  # the trainer's own choice: ExpertAdamW on bf16 expert adapters
        from types import SimpleNamespace

        from poros.train.expert_adamw import build_optimizer

        opt = build_optimizer(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            SimpleNamespace(learning_rate=1e-2, adam_beta1=0.9, adam_beta2=0.999,
                            adam_epsilon=1e-8, weight_decay=0.01, seed=0))
    else:
        opt = torch.optim.AdamW(params, lr=1e-2)
    out = []
    for step in range(steps):
        ids = batches[step]
        result = model(input_ids=ids, labels=ids)
        result.loss.backward()
        if mark:
            model._poros_moe_store.source.mark_step()
        grads = [p.grad.clone() for p in params]
        opt.step()
        opt.zero_grad(set_to_none=True)
        out.append((result.loss.detach(), result.logits.detach(), grads))
    if trainers:
        return out, [p.detach().clone() for p in params], opt
    return out, [p.detach().clone() for p in params]


@needs_fadvise
def test_ram_and_ssd_training_are_bitwise_equal(checkpoint, tmp_path):
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, VOCAB, (2, 12), generator=g) for _ in range(5)]
    torch.manual_seed(0)
    ram_steps, ram_final = _train(load_moe_streamed(str(checkpoint)), 5, batches, mark=False)
    torch.manual_seed(0)
    model = load_moe_streamed(str(checkpoint), store="ssd", store_path=tmp_path,
                              ssd_options={"depth": 4, "chunk_bytes": 4096})
    ssd_steps, ssd_final = _train(model, 5, batches, mark=True)
    src = model._poros_moe_store.source
    assert src.stats["prefetch_hits"] > 0  # replay was live, not just synchronous reads
    assert src.stats["hinted"] > 0  # the router's picks reached the source
    for step, ((la, xa, ga), (lb, xb, gb)) in enumerate(zip(ram_steps, ssd_steps, strict=True)):
        assert torch.equal(la, lb), step
        assert torch.equal(xa, xb), step
        assert len(ga) == len(gb) > 0
        assert all(torch.equal(a, b) for a, b in zip(ga, gb, strict=True)), step
    assert all(torch.equal(a, b) for a, b in zip(ram_final, ssd_final, strict=True))
    src.close()


@needs_fadvise
def test_ram_and_ssd_training_with_the_trainers_optimizer_are_bitwise_equal(
        checkpoint, tmp_path, cpu_bf16_backward):
    """The same as above, with the optimizer the trainer builds for these adapters
    (``build_optimizer``: ExpertAdamW on bf16 expert adapters): losses, logits, grads, weights,
    and the optimizer's whole state (both moments, the remainder, the step) equal, bit for
    bit."""
    from poros.train.expert_adamw import ExpertAdamW

    g = torch.Generator().manual_seed(2)
    batches = [torch.randint(0, VOCAB, (2, 12), generator=g) for _ in range(4)]
    torch.manual_seed(0)
    ram_steps, ram_final, ram_opt = _train(load_moe_streamed(str(checkpoint)), 4, batches,
                                           mark=False, trainers=True)
    torch.manual_seed(0)
    model = load_moe_streamed(str(checkpoint), store="ssd", store_path=tmp_path,
                              ssd_options={"depth": 4, "chunk_bytes": 4096})
    ssd_steps, ssd_final, ssd_opt = _train(model, 4, batches, mark=True, trainers=True)
    assert isinstance(ram_opt, ExpertAdamW) and isinstance(ssd_opt, ExpertAdamW)
    assert all(p.dtype == torch.bfloat16 for p in ram_final)
    for step, ((la, xa, ga), (lb, xb, gb)) in enumerate(zip(ram_steps, ssd_steps, strict=True)):
        assert torch.equal(la, lb) and torch.equal(xa, xb), step
        assert all(torch.equal(a, b) for a, b in zip(ga, gb, strict=True)), step
    assert all(torch.equal(a, b) for a, b in zip(ram_final, ssd_final, strict=True))
    a, b = ram_opt.state_dict()["state"], ssd_opt.state_dict()["state"]
    assert a.keys() == b.keys() and a
    for i in a:
        assert set(a[i]) == {"step", "exp_avg", "exp_avg_sq", "remainder"}
        for k in a[i]:
            assert torch.equal(a[i][k], b[i][k]), (i, k)
    assert any(bool(s["remainder"].abs().sum() > 0) for s in a.values())
    model._poros_moe_store.source.close()
