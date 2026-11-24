"""Block-major MoE == sequential MoE, bit for bit, for every validated MoE family (CUDA).

Tiny random models of each family (tests/moe/test_families.py configs + qwen3_moe), padded
masks and varied lengths, RAM and SSD stores, the default expert compute (fused where the family
qualifies) and the reference compute's two groupings (set on both arms: pow2 and single), and
block boundaries parked in pinned slabs (path:
efficient), kept on the GPU (path: fast), or parked in ordinary RAM where pinning is refused;
at lora_dropout 0 and the default 0.05, whose masks a MoE model keys by (seed, update,
micro-batch, layer) in either loop (poros.core.sequential_rng.KeyedRng). At 0.05 also: dense
blocks of 1, 2, 4 and 8 layers, the model fully resident (experts too, no dense streaming), and
a run stopped and resumed on either loop.
"""
from __future__ import annotations

import dataclasses

import pytest

from tests.moe import require_tested_transformers

pytestmark = pytest.mark.gpu  # CUDA only: the GPU lane (-m gpu) runs it
torch = pytest.importorskip("torch")
require_tested_transformers()
if not torch.cuda.is_available():
    pytest.skip("block-major MoE composes with the streamed dense engine (CUDA)",
                allow_module_level=True)

import transformers  # noqa: E402

from poros.sss import pinning  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402
from tests.moe.test_families import CONFIGS, VALIDATED  # noqa: E402

FAMILIES = ("qwen3_moe",) + tuple(VALIDATED)


def _config(family):
    if family == "qwen3_moe":
        return transformers.Qwen3MoeConfig(vocab_size=96, hidden_size=64, intermediate_size=96,
                                           moe_intermediate_size=40, num_hidden_layers=3,
                                           num_attention_heads=4, num_key_value_heads=2,
                                           head_dim=16, num_experts=12, num_experts_per_tok=3)
    return CONFIGS[family]()


@pytest.fixture(scope="module", params=FAMILIES)
def checkpoint(request, tmp_path_factory):
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(_config(request.param))
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    path = tmp_path_factory.mktemp(request.param)
    model.to(torch.bfloat16).save_pretrained(path)
    return path


@pytest.fixture(scope="module", params=FAMILIES)
def deep(request, tmp_path_factory):
    """The family's tiny model with nine decoder layers, so dense blocks of 1, 2, 4 and 8 layers
    split it four ways."""
    cfg = _config(request.param)
    text = getattr(cfg, "text_config", None) or cfg
    text.num_hidden_layers = 9
    if getattr(text, "layer_types", None):
        text.layer_types = [text.layer_types[i % len(text.layer_types)] for i in range(9)]
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    path = tmp_path_factory.mktemp(f"{request.param}_deep")
    model.to(torch.bfloat16).save_pretrained(path)
    return path


def _loader():
    g = torch.Generator().manual_seed(1)
    rows = []
    for seq in (16, 11, 16, 7, 13, 16, 9, 16):
        x = torch.randint(0, 96, (seq,), generator=g)
        rows.append({"input_ids": x, "labels": x.clone(), "attention_mask": torch.ones_like(x)})
    return torch.utils.data.DataLoader(rows, batch_size=1)


def _run(checkpoint, out, loader=None, **kw):
    knobs = dict(output_dir=str(out), max_steps=2, save_steps=0, progress=False,
                 gradient_accumulation_steps=4, lora_dropout=0.0, learning_rate=1e-3,
                 warmup_steps=0)
    tr = PorosTrainer.from_pretrained(str(checkpoint), **{**knobs, **kw})
    tr.train(loader if loader is not None else _loader())
    offl = getattr(tr, "_boundary_offload", None)
    params = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    return tr.history.losses, params, None if offl is None else dict(offl.stats)


@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize("offload", ["on", "off", "ram"])
@pytest.mark.parametrize("grouping", [None, "pow2", "single"])  # None: the default compute
@pytest.mark.parametrize("store", ["ram", "ssd"])
def test_block_major_equals_sequential(checkpoint, tmp_path, monkeypatch, grouping, store, offload,
                                       dropout):
    extra = {"lora_dropout": dropout}
    if grouping is not None:  # the reference compute's setting (it selects that compute)
        extra["moe_expert_grouping"] = grouping
    if store == "ssd":
        extra.update(store="ssd", store_path=str(tmp_path / "sss"))
    # sequential pinned: on CUDA block-major accumulation is the default
    seq_l, seq_p, _ = _run(checkpoint, tmp_path / "seq", path="sequential", **extra)
    if offload == "on":  # the slabs reserved for a model this small round down to none
        monkeypatch.setenv("POROS_BOUNDARY_PIN", "256M")
    elif offload == "ram":  # a host that locks nothing: ordinary-RAM slabs
        monkeypatch.setattr(pinning, "reserve", lambda n: False)
    bm_l, bm_p, st = _run(checkpoint, tmp_path / "bm",
                          path="fast" if offload == "off" else "efficient", **extra)
    if offload == "off":
        assert st is None
    else:
        assert st["offloaded"] > 0 and st["kept_on_device"] == 0, st
        assert offload != "ram" or (st["pageable_slabs"] > 0 and st["slabs"] == 0), st
    assert seq_l == bm_l
    assert seq_p.keys() == bm_p.keys() and seq_p
    for n, p in seq_p.items():
        assert torch.equal(p, bm_p[n]), n


def _padded(rows=2, n=20):
    """Rows of 5 to 16 tokens, ``rows`` to a micro-batch, each padded to its longest, shuffled by
    a seeded generator (as the trainer's own loader: resumable)."""
    g = torch.Generator().manual_seed(2)
    seqs = [torch.randint(3, 96, (int(k),), generator=g)
            for k in torch.randint(5, 17, (n,), generator=g)]

    def collate(items):
        width = max(len(x) for x in items)
        ids = torch.zeros(len(items), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(items):
            ids[i, :len(x)], mask[i, :len(x)] = x, 1
        return {"input_ids": ids, "attention_mask": mask,
                "labels": ids.masked_fill(mask == 0, -100)}

    return torch.utils.data.DataLoader(seqs, batch_size=rows, shuffle=True, collate_fn=collate,
                                       generator=torch.Generator().manual_seed(0))


# the default dropout and expert compute (fused where the family qualifies), padded micro-batches
# of two rows
AT_DROPOUT = dict(lora_dropout=0.05, per_device_batch_size=2, max_steps=3)


def _arm(checkpoint, out, monkeypatch, *, dense_block=None, resident=False, **kw):
    """One run's losses, grad norms, adapter and CUDA generator after it. ``dense_block``: layers
    per streamed dense block; ``resident``: the experts and dense layers all held, nothing
    streamed (MoeConfig.streamed=False)."""
    from poros.moe import prepare

    real = prepare.streaming_config

    def made(*a, **k):
        c = real(*a, **k)
        if resident:
            return dataclasses.replace(c, streamed=False, look_ahead=0, capacity_bytes=0)
        return c if dense_block is None else dataclasses.replace(c, dense_block_size=dense_block)

    monkeypatch.setattr(prepare, "streaming_config", made)
    knobs = dict(output_dir=str(out), save_steps=0, progress=False, gradient_accumulation_steps=4,
                 learning_rate=1e-3, warmup_steps=0, **AT_DROPOUT)
    tr = PorosTrainer.from_pretrained(str(checkpoint), **{**knobs, **kw})
    tr.train(_padded())
    params = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    return list(tr.history.losses), list(tr.history.grad_norms), params, torch.cuda.get_rng_state()


def _same(a, b):
    assert a[0] == b[0] and a[1] == b[1]  # every step's loss and grad norm
    assert a[2].keys() == b[2].keys() and a[2]
    for n, p in a[2].items():
        assert torch.equal(p, b[2][n]), n
    assert torch.equal(a[3], b[3])  # the CUDA generator, where what follows starts


def test_at_dropout_every_loop_block_size_and_residency_trains_the_same_update(
        deep, tmp_path, monkeypatch):
    """The default dropout: the sequential loop (dense blocks of the engine's default size), the
    sequential loop at block 1, block-major at dense blocks of 1, 2, 4 and 8 layers, and the
    model fully resident: every step's loss and grad norm, every adapter tensor, and the CUDA
    generator after, bit for bit."""
    ref = _arm(deep, tmp_path / "seq", monkeypatch, path="sequential")
    _same(ref, _arm(deep, tmp_path / "seq1", monkeypatch, dense_block=1,
                    path="sequential"))
    for b in (1, 2, 4, 8):
        _same(ref, _arm(deep, tmp_path / f"bm{b}", monkeypatch, dense_block=b,
                        path="efficient"))
    _same(ref, _arm(deep, tmp_path / "resident", monkeypatch, resident=True,
                    path="sequential"))


@pytest.mark.parametrize("first, then", [(True, True), (False, True), (True, False)],
                         ids=["block-major", "sequential-then-block-major",
                              "block-major-then-sequential"])
def test_at_dropout_a_resumed_run_is_the_uninterrupted_one(deep, tmp_path, monkeypatch,
                                                            first, then):
    """Stopped after its first of three updates and resumed, on either loop: the uninterrupted
    sequential run's later steps, adapter and CUDA generator. 20 rows are 10 micro-batches an
    epoch: the second epoch, and its shuffle, starts inside the third update."""
    from poros.core.errors import PorosCancelledError

    whole = _arm(deep, tmp_path / "whole", monkeypatch, path="sequential")
    part = PorosTrainer.from_pretrained(
        str(deep), output_dir=str(tmp_path / "part"), save_steps=0, progress=False,
        gradient_accumulation_steps=4, learning_rate=1e-3, warmup_steps=0,
        path="efficient" if first else "sequential", **AT_DROPOUT)
    with pytest.raises(PorosCancelledError) as stopped:
        part.train(_padded(), should_stop=lambda: part.global_step == 0)
    resumed = PorosTrainer.from_pretrained(
        str(deep), output_dir=str(tmp_path / "part"), save_steps=0, progress=False,
        gradient_accumulation_steps=4, learning_rate=1e-3, warmup_steps=0,
        path="efficient" if then else "sequential", **AT_DROPOUT)
    resumed.train(_padded(), resume_from=str(stopped.value.checkpoint))
    assert list(resumed.history.losses) == whole[0][1:]
    assert list(resumed.history.grad_norms) == whole[1][1:]
    for n, p in whole[2].items():
        assert torch.equal(dict(resumed.model.named_parameters())[n], p), n
    assert torch.equal(torch.cuda.get_rng_state(), whole[3])
