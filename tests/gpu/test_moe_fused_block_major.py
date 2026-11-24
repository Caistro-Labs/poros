"""The fused expert compute through the trainer (CUDA): block-major, with each MoE layer's experts
run once for all micro-batches (``poros.moe.batching``), is the sequential loop's update bit for
bit -- losses and every trained parameter -- on the RAM and SSD expert stores."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the fused expert compute runs on CUDA", allow_module_level=True)
pytest.importorskip("triton")

from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.moe.modules import ManagedMoeBlock  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

VOCAB = 64


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=VOCAB, hidden_size=128, intermediate_size=256, moe_intermediate_size=96,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        num_experts=8, num_experts_per_tok=2,
    )
    path = tmp_path_factory.mktemp("qwen3_moe_fused_bm")
    model = Qwen3MoeForCausalLM(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" not in name:
                p.normal_(0.0, 0.05)
    model.to(torch.bfloat16).save_pretrained(path)
    return path


def _loader(lengths):
    g = torch.Generator().manual_seed(1)
    rows = []
    for seq in lengths:
        x = torch.randint(0, VOCAB, (seq,), generator=g)
        rows.append({"input_ids": x, "labels": x.clone(), "attention_mask": torch.ones_like(x)})
    return torch.utils.data.DataLoader(rows, batch_size=1)


def _run(checkpoint, tmp_path, lengths, **overrides):
    knobs = dict(output_dir=str(tmp_path), max_steps=2, save_steps=0, progress=False,
                 gradient_accumulation_steps=4, learning_rate=1e-3,
                 warmup_steps=0, moe_expert_compute="fused")
    trainer = PorosTrainer.from_pretrained(str(checkpoint), **{**knobs, **overrides})
    trainer.train(_loader(lengths))
    params = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    hits = sum(m.remat_hits for m in trainer.model.modules() if isinstance(m, ManagedMoeBlock))
    _run.batch = getattr(trainer, "_moe_batch", None)  # its batching state (block-major only)
    return trainer.history.losses, params, hits


# equal lengths, and varied ones (each micro-batch routes a different number of tokens); expert
# LoRA dropout on and off (the masks are keyed per micro-batch and layer)
@pytest.mark.parametrize("lengths,dropout,store,ga", [
    ((16,) * 8, 0.0, "ram", 4),
    ((16, 11, 16, 7, 13, 16, 9, 16), 0.0, "ram", 4),
    ((16, 11, 16, 7, 13, 16, 9, 16), 0.0, "ssd", 4),
    ((16, 11, 16, 7, 13, 16, 9, 16), 0.1, "ram", 4),
    ((16, 11, 16, 7, 13, 16, 9, 16), 0.0, "ram", 1),  # one micro-batch a step
])
def test_fused_block_major_equals_sequential(checkpoint, tmp_path, lengths, dropout, store, ga):
    extra = {"lora_dropout": dropout, "gradient_accumulation_steps": ga}
    if store == "ssd":
        extra.update(store="ssd", store_path=str(tmp_path / "sss"))
    seq_losses, seq_params, _ = _run(checkpoint, tmp_path / "seq", lengths, path="sequential",
                                     **extra)
    bm_losses, bm_params, hits = _run(checkpoint, tmp_path / "bm", lengths, path="efficient",
                                      **extra)
    assert seq_losses == bm_losses
    assert seq_params.keys() == bm_params.keys() and seq_params
    for name, p in seq_params.items():
        assert torch.equal(p, bm_params[name]), name
    assert hits > 0
    batch = _run.batch  # the block-major run's
    assert batch.checked and all(batch.checked.values())  # the layer class batched
    if ga > 1:
        assert batch.visits["forward"] > 0 and batch.visits["backward"] > 0
        # every batched forward visit's row counts went to its layer's backward visit (the one
        # backward visit more: the first layer's, whose forward visit checked the layer class)
        assert batch.visits["counts_reused"] == batch.visits["forward"]
        assert batch.visits["backward"] == batch.visits["forward"] + 1


def test_one_by_eight_expert_grads_own_their_memory(checkpoint, tmp_path):
    """1x8: at the optimizer step every expert adapter grad owns its memory. A batched visit
    computes the eight micro-batches' grads into one buffer per parameter; a view of it as
    ``param.grad`` kept the whole buffer alive (docs/validation/int/4090/round5_moegrad/)."""
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    trainer = PorosTrainer.from_pretrained(
        str(checkpoint), output_dir=str(tmp_path), max_steps=1, save_steps=0, progress=False,
        gradient_accumulation_steps=8, learning_rate=1e-3, warmup_steps=0,
        moe_expert_compute="fused", path="efficient")
    owned = []

    def check(optimizer, args, kwargs):
        for block in trainer.model.modules():
            if isinstance(block, ManagedMoeBlock):
                for p in block.adapter.parameters():
                    g = p.grad
                    owned.append(g is not None and
                                 g.untyped_storage().nbytes() == g.numel() * g.element_size())

    handle = register_optimizer_step_pre_hook(check)
    try:
        trainer.train(_loader((16, 11, 16, 7, 13, 16, 9, 16)))
    finally:
        handle.remove()
    assert trainer._moe_batch.visits["backward"] > 0  # the eight micro-batches were batched
    assert owned and all(owned)


def test_batched_backward_visits_do_not_wait_on_the_gpu(checkpoint, tmp_path, monkeypatch):
    """A batched backward visit that plans on its forward visit's row counts queues all of its
    work without one host sync (torch's sync debug mode raises on any), and a batched forward
    visit syncs only for its plan's row counts (CUDA bincount reads its input's min and max, then
    the counts come back) -- so the host runs ahead of the GPU through a layer's expert compute."""
    import warnings

    from poros.moe import batching

    real_backward, real_finish = batching.BatchState.backward, batching.ForwardVisit.finish
    checked, forward_syncs = [], []

    def backward(self, modules, grads):
        block = batching._block_of(modules)
        reuse = block is not None and block._fwd_counts is not None
        if reuse:
            torch.cuda.set_sync_debug_mode("error")
        try:
            return real_backward(self, modules, grads)
        finally:
            torch.cuda.set_sync_debug_mode("default")
            checked.append(reuse)

    def finish(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                return real_finish(self)
            finally:
                torch.cuda.set_sync_debug_mode("default")
                forward_syncs.append(sum("synchroniz" in str(w.message) for w in caught))

    monkeypatch.setattr(batching.BatchState, "backward", backward)
    monkeypatch.setattr(batching.ForwardVisit, "finish", finish)
    # without LoRA dropout: each micro-batch's masks are drawn into its own rows, which a
    # nonzero over the plan's row segments finds (a host sync of the masks, not of the plan)
    _run(checkpoint, tmp_path, (16, 11, 16, 7, 13, 16, 9, 16), path="efficient", lora_dropout=0.0)
    batch = _run.batch
    assert sum(checked) == batch.visits["counts_reused"] > 0  # each such visit ran sync-free
    assert forward_syncs and max(forward_syncs) <= 3
