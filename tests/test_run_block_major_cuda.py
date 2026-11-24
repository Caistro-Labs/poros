"""poros.Run (the chat/SFT and DPO recipe loops) under block-major accumulation == its sequential
loop, bit for bit (CUDA).

Tiny random NF4 Qwen2 and Qwen3-MoE models, chat rows of uneven length (answers-only weights), 4
micro-batches per step, with and without group_by_length (pinned on both arms), at lora_dropout
0 and 0.1 (block-major draws the sequential loop's masks: a dense model's in that loop's order, a
MoE model's keyed by layer in both), and preference pairs. The MoE model runs the default expert
compute (fused on CUDA); the masks test runs the reference compute too.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu  # CUDA only: the GPU lane (-m gpu) runs it
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("block-major composes with the streamed engine (CUDA)", allow_module_level=True)
pytest.importorskip("bitsandbytes")
transformers = pytest.importorskip("transformers")

from poros import data as pdata  # noqa: E402
from poros.recipes import sft  # noqa: E402
from poros.train.run import Run  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402
from tests.conftest import Tok  # noqa: E402
from tests.moe import fused_mask_digests  # noqa: E402


def _rows():
    g = torch.Generator().manual_seed(3)
    out = []
    for n in (5, 17, 9, 30, 12, 7, 22, 15):
        words = " ".join(str(int(x)) for x in torch.randint(0, 50, (n,), generator=g))
        out.append({"messages": [{"role": "user", "content": words},
                                 {"role": "assistant", "content": words[::-1]}]})
    return out


@pytest.fixture(scope="module", params=["qwen2", "qwen3_moe"])
def checkpoint(request, tmp_path_factory):
    torch.manual_seed(0)
    if request.param == "qwen2":
        cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                                       num_hidden_layers=3, num_attention_heads=4,
                                       num_key_value_heads=2)
    else:
        cfg = transformers.Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=96,
                                          moe_intermediate_size=40, num_hidden_layers=3,
                                          num_attention_heads=4, num_key_value_heads=2,
                                          head_dim=16, num_experts=8, num_experts_per_tok=2)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    path = tmp_path_factory.mktemp(request.param)
    model.to(torch.bfloat16).save_pretrained(path)
    return request.param, path


def _engine(checkpoint, out, block_major, dropout=0.0, **extra):
    kind, path = checkpoint
    knobs = dict(output_dir=str(out), max_steps=2, save_steps=0, progress=False,
                 lora_dropout=dropout, learning_rate=1e-3, warmup_steps=0,
                 gradient_accumulation_steps=4, per_device_batch_size=1,
                 path="efficient" if block_major else "sequential")
    if kind == "qwen2":
        knobs.update(quantization="nf4", block_size=1, blockwise_arch_check="off")
    return PorosTrainer.from_pretrained(str(path), **{**knobs, **extra})


def _train(checkpoint, out, block_major, group, dropout=0.0):
    tr = _engine(checkpoint, out, block_major, dropout)
    data = pdata.chats(_rows(), Tok(), batch_size=4, length=128)
    data.group_by_length = group
    run = Run.from_trainer(tr, Tok())
    sft.main(run, data, steps=2)
    params = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    return list(tr.history.losses), list(tr.history.grad_norms), params, torch.cuda.get_rng_state()


def _same(a, b):
    assert a[0] == b[0] and a[1] == b[1]  # losses, grad norms
    assert a[2].keys() == b[2].keys() and a[2]
    for n, p in a[2].items():
        assert torch.equal(p, b[2][n]), n
    assert torch.equal(a[3], b[3])  # the CUDA generator after the run


@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("group", [False, True])
def test_run_block_major_equals_sequential(checkpoint, tmp_path, group, dropout):
    _same(_train(checkpoint, tmp_path / "seq", False, group, dropout),
          _train(checkpoint, tmp_path / "bm", True, group, dropout))


@pytest.mark.parametrize("path", ["efficient", "fast"])
@pytest.mark.parametrize("block_size", [1, 2, 3])
@pytest.mark.parametrize("compute", [None, "reference"])
@pytest.mark.parametrize("micro_batches", [4, 1])
def test_every_recompute_draws_its_forwards_masks(checkpoint, tmp_path, path, block_size, compute,
                                                   micro_batches):
    """The CUDA generator at dropout 0.1, both paths, blocks of 1, 2 and 3 (all) layers: every
    mask a block's recompute draws is one its forward drew, block-major draws the sequential
    loop's masks (on Qwen2 one per micro-batch and LoRA dropout; a MoE model's expert adapters
    draw once per expert group under the reference compute, once per layer call under the
    default, fused, which a call of several micro-batches draws a micro-batch at a time), and the
    update is the same, at 4 micro-batches an update and at 1."""
    from poros.moe.modules import ManagedMoeBlock
    from tests.test_block_major_dropout import _mask_digests

    moe = checkpoint[0] != "qwen2"
    if moe and block_size != 1:
        pytest.skip("a MoE model's block sizes: tests/moe/test_block_major_families.py")
    if not moe and (compute is not None or micro_batches != 4):
        pytest.skip("a MoE model's expert compute")
    fused_moe = moe and compute is None
    masks, states = {}, []
    for name, arm in (("seq", {"block_major": False}),
                      ("bm", {"block_major": True, "path": path})):
        # (path in extra overrides the efficient that block_major sets)
        if not moe:
            arm["block_size"] = block_size
        if compute is not None:
            arm["moe_expert_compute"] = compute
        tr = _engine(checkpoint, tmp_path / name, dropout=0.1,
                     gradient_accumulation_steps=micro_batches, **arm)
        assert not moe or tr.config.ran_compute["moe_expert_compute"] == (
            "fused" if fused_moe else "reference")
        data = pdata.chats(_rows(), Tok(), batch_size=micro_batches, length=128)
        with pytest.MonkeyPatch.context() as mp:
            if fused_moe:
                hooks = []
                got = fused_mask_digests(mp)
                drawers = sum(isinstance(m, ManagedMoeBlock) for m in tr.model.modules())
            else:
                got, hooks = _mask_digests(torch, tr.model)
                drawers = len(hooks)
            sft.main(Run.from_trainer(tr, Tok()), data, steps=2)
        for h in hooks:
            h.remove()
        # every adapter's dropout drew: Qwen2's LoRA-wrapped linears, a MoE layer's experts
        assert len(got) == drawers >= (3 if moe else 3 * 4)
        for module, calls in got.items():
            seen = {d: calls.count(d) for d in calls}
            assert moe or len(seen) == 2 * 4, module  # a mask a micro-batch
            assert not fused_moe or len(seen) == 2 * micro_batches, module  # a layer call each
            assert min(seen.values()) >= 2, module  # each drawn again by its recompute
        masks[name] = {module: set(calls) for module, calls in got.items()}
        params = {n: p.detach().clone() for n, p in tr.model.named_parameters()
                  if p.requires_grad}
        states.append((list(tr.history.losses), list(tr.history.grad_norms), params,
                       torch.cuda.get_rng_state()))
    assert masks["seq"] == masks["bm"]
    _same(*states)


def test_preference_pairs_block_major_equal_sequential_with_dropout(checkpoint, tmp_path):
    from poros.recipes import dpo

    if checkpoint[0] != "qwen2":
        pytest.skip("dense")
    g = torch.Generator().manual_seed(5)
    rows = []
    for i, n in enumerate((4, 11, 7, 15, 9, 6, 13, 8)):
        words = " ".join(str(int(x)) for x in torch.randint(0, 50, (n,), generator=g))
        rows.append({"prompt": words, "chosen": words[::-1], "rejected": f"no {i}"})
    out = []
    for name, bm in (("seq", False), ("bm", True)):
        tr = _engine(checkpoint, tmp_path / name, bm, 0.1)
        run = Run.from_trainer(tr, Tok())
        data = pdata.pairs(rows, Tok(), batch_size=4, length=128)
        dpo.main(run, data, steps=2, beta=0.5)
        params = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
        out.append((list(tr.history.losses), list(tr.history.grad_norms), params,
                    torch.cuda.get_rng_state()))
    _same(*out)


def test_a_run_of_one_micro_batch_a_step_runs_block_major_and_reserves_nothing_at_load(
        checkpoint, tmp_path):
    """poros.Run.create on CUDA's default (block-major, efficient): a step's micro-batches are
    known only when it runs, so nothing is page-locked for block inputs at load; the first step
    makes them. Steps of one micro-batch run block-major too (no slower there), the same
    update as the sequential loop's; the run's poros_config.json says efficient, and Run.open
    reads the setting back."""
    import json
    from pathlib import Path

    if checkpoint[0] != "qwen2":
        pytest.skip("dense")
    _, path = checkpoint

    def made(name, **kw):
        return Run.create(str(path), rank=8, lr=1e-3, length=128, tokenizer=Tok(),
                          output_dir=str(tmp_path / name), save_steps=0, progress=False,
                          warmup_steps=0, quantization="nf4", block_size=1,
                          blockwise_arch_check="off", gradient_accumulation_steps=4,
                          per_device_batch_size=1, lora_dropout=0.1, **kw)

    def trained(run, rows):
        sft.main(run, pdata.chats(_rows(), Tok(), batch_size=rows, length=128), steps=2)
        saved = json.loads((Path(run.save()) / "poros_config.json").read_text())
        params = {n: p.detach().clone() for n, p in run._engine.model.named_parameters()
                  if p.requires_grad}
        return saved, params

    for rows in (1, 4):
        run = made(f"bm{rows}")
        assert run.config.path == "efficient"  # CUDA's default
        assert run._engine._boundary_offload is None  # nothing locked at load
        saved, got = trained(run, rows)
        assert run._engine._boundary_offload is not None  # made by its first step
        assert (saved["path"], saved["ran_path"]) == ("efficient", "efficient")
        _, want = trained(made(f"seq{rows}", path="sequential"), rows)
        assert got.keys() == want.keys() and all(torch.equal(got[n], want[n]) for n in got)
    reopened = Run.open(tmp_path / "bm4", tokenizer=Tok())  # the setting, not what ran
    assert reopened.config.path == "efficient"
