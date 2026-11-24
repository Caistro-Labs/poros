"""Every MoE family through the real Mac path on the CPU, against what the model itself computes.

The reference is the real transformers model from ``from_pretrained``, its own router, attention,
shared experts and head under plain autograd. Its experts run the model's own loop over the NF4
round trip of each expert, with the LoRA written out at each projection, and a test holds that loop
to the model's. Equal means ``torch.equal`` on the loss and on every adapter gradient, row by row,
so an expert no token routed to must come out exactly zero. It says nothing about Metal or real
shapes.
Each mutation below is a fault this exists to catch.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("accelerate")

import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from poros import metal  # noqa: E402
from poros.cli.train_cmd import _arch_label  # noqa: E402
from poros.core import blockwise  # noqa: E402
from poros.core.arch_registry import discover_layers  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosCancelledError, PorosStepCheckError  # noqa: E402
from poros.metal import experts as experts_mod  # noqa: E402
from poros.metal import load as metal_load  # noqa: E402
from poros.metal.experts import Experts  # noqa: E402
from poros.metal.export import merged  # noqa: E402
from poros.metal.pack import SssFile, pack  # noqa: E402
from poros.metal.tiny import DenseExperts, _decoded  # noqa: E402
from poros.metal.tiny import moe_reference as _reference  # noqa: E402
from poros.metal.tiny import sparse_blocks as _sparse_blocks  # noqa: E402
from poros.metal.tiny import vanilla_moe as _vanilla  # noqa: E402
from poros.moe.export import export_merged  # noqa: E402
from poros.moe.load import load_moe_streamed  # noqa: E402
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402
from poros.train.batch import Batch, Datum  # noqa: E402
from poros.train.run import Run  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402
from tests.metal import families as fam  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

arms = tiny.arms
FACTORS = ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down")
VOCAB = 96
SEQ = 16
_SHARED = dict(vocab_size=VOCAB, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
               use_cache=False, attn_implementation="eager")
_DEEPSEEK = {**_SHARED, "num_key_value_heads": 4}  # MLA: one KV head per query head
CONFIGS = {
    # A dense mlp_only layer among the sparse ones.
    "qwen3_moe": lambda: transformers.Qwen3MoeConfig(
        **_SHARED, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=3,
        mlp_only_layers=[1], head_dim=16, num_experts=12, num_experts_per_tok=3,
        norm_topk_prob=True),
    # A shared expert with its sigmoid gate, and a dense layer.
    "qwen2_moe": lambda: transformers.Qwen2MoeConfig(
        **_SHARED, intermediate_size=96, moe_intermediate_size=40,
        shared_expert_intermediate_size=48, num_hidden_layers=3, mlp_only_layers=[1],
        num_experts=12, num_experts_per_tok=3, use_sliding_window=False, sliding_window=32768),
    # A clip that bites. Its config has no layer types, and so no mask of its own on the block loop.
    "olmoe": lambda: transformers.OlmoeConfig(
        **_SHARED, intermediate_size=40, num_hidden_layers=2, num_experts=12,
        num_experts_per_tok=3, clip_qkv=0.1),
    # A dense first layer, two shared experts, group-limited routing, MLA attention.
    "deepseek_v2": lambda: transformers.DeepseekV2Config(
        **_DEEPSEEK, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=3,
        first_k_dense_replace=1, n_shared_experts=2, n_routed_experts=12, num_experts_per_tok=3,
        n_group=3, topk_group=2, topk_method="group_limited_greedy", kv_lora_rank=32,
        q_lora_rank=None, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16),
    # What DeepSeek-V2-Lite sets: plain top-k, one group.
    "deepseek_v2_greedy": lambda: transformers.DeepseekV2Config(
        **_DEEPSEEK, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=3,
        first_k_dense_replace=1, n_shared_experts=2, n_routed_experts=12, num_experts_per_tok=3,
        n_group=1, topk_method="greedy", kv_lora_rank=32, q_lora_rank=None, qk_nope_head_dim=16,
        qk_rope_head_dim=8, v_head_dim=16, max_position_embeddings=64,
        rope_scaling={"type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1,
                      "mscale": 0.707, "mscale_all_dim": 0.707,
                      "original_max_position_embeddings": 32}),
}
# The newest checkpoints: real config.json, sizes aside, and experts stored fused, under a wrapper.
REAL = {
    "qwen3_5_moe": lambda: fam.real_config("qwen3_6_35b_a3b", hidden=64, vocab=VOCAB),
    "gemma4_moe": lambda: fam.real_config("gemma_4_26b_a4b", hidden=64, vocab=VOCAB),
}
CONFIGS.update(REAL)
SHARED_EXPERT = ("qwen2_moe", "deepseek_v2", "deepseek_v2_greedy", "qwen3_5_moe")


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _checkpoint(root, family) -> SimpleNamespace:
    """A bf16 checkpoint with vanilla names, each expert's tensors of its own, split across shards.
    Fresh init has unit norms and zero routers, so nothing could see them."""
    cfg = CONFIGS[family]()
    if family in REAL:
        fam.write_fused(root / "source", cfg)
        return SimpleNamespace(family=family, cfg=cfg, source=root / "source", root=root)
    torch.manual_seed(1)
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    model.to(torch.bfloat16).save_pretrained(root / "source", max_shard_size="60KB")
    return SimpleNamespace(family=family, cfg=cfg, source=root / "source", root=root)


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def checkpoint(request, tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp(request.param), request.param)


def _trainer(checkpoint, out, **over):
    config = PorosConfig(
        model_name_or_path=str(checkpoint.source), store=over.pop("store", "ram"),
        max_seq_length=SEQ, store_path=str(checkpoint.root / "store"),
        gradient_accumulation_steps=1, max_steps=over.pop("steps", 2),
        lora_dropout=over.pop("dropout", 0.0), learning_rate=1e-2, warmup_steps=0,
        save_steps=over.pop("save_steps", 0), eval_steps=0, output_dir=str(out), progress=False,
        lora_rank=4, lora_alpha=8, **over,
    )
    # A Mac plans a local checkpoint of a family its table has as that family's smallest model
    # (poros.metal.table._as_catalog), 13 to 25 GiB for these: a Mac with 14 GB refuses it. What
    # is tested here is the arithmetic, not that fit: the gate is not asked.
    trainer = arms.CpuDouble.from_pretrained(str(checkpoint.source), config=config,
                                             check_fit=False)
    torch.set_num_threads(1)  # a trainer picks its own
    return trainer


def _lora(trainer) -> dict[str, nn.Parameter]:
    return {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}


def _randomize(lora: dict[str, nn.Parameter]) -> None:
    """lora_B starts at zero, and then no gradient of lora_A could be seen."""
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for param in lora.values():
            param.copy_(0.03 * torch.randn(param.shape, generator=gen, dtype=param.dtype))


def _ids():
    return torch.randint(0, VOCAB, (2, SEQ), generator=torch.Generator().manual_seed(3))


def _batch(padded):
    ids = _ids()
    batch = {"input_ids": ids, "labels": ids.clone()}
    if padded:
        real = torch.arange(SEQ) < 9 if padded == "right" else torch.arange(SEQ) >= 7
        mask = real.long().unsqueeze(0).expand_as(ids)
        batch.update(attention_mask=mask, labels=ids.masked_fill(mask == 0, -100))
    return batch


def _compare(checkpoint, trainer, *, padded=None, batch=None) -> SimpleNamespace:
    """One microstep on the real path and on the reference, from the same random adapter."""
    lora = _lora(trainer)
    _randomize(lora)
    scaling = trainer.config.lora_alpha / trainer.config.lora_rank
    reference, pairs = _reference(checkpoint.source, lora, scaling)
    batch = batch or _batch(padded)
    loss = trainer._forward_blockwise_adapter_model(batch)
    loss.backward()
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    # The model's own loss rounds a little otherwise: it scores a last token, ignored, on each row.
    want = trainer._compute_loss(reference(**inputs).logits.float(), batch["labels"])
    want.backward()
    routed = {name: reference.get_submodule(name).experts.routed
              for name, _ in _sparse_blocks(reference)}
    return SimpleNamespace(
        same_loss=torch.equal(loss, want),
        same_grads=all(torch.equal(p.grad, pairs[n].grad) for n, p in lora.items()),
        lora=lora,
        routed=routed,
    )


@pytest.mark.parametrize("padded", [None, "right", "left"], ids=["full", "right", "left"])
def test_grads_are_bit_equal_to_plain_autograd_on_the_real_model(checkpoint, tmp_path, padded):
    got = _compare(checkpoint, _trainer(checkpoint, tmp_path), padded=padded)
    assert got.same_loss
    assert got.same_grads


def test_run_on_chats_is_bit_equal_too(checkpoint, tmp_path):
    """Rows of unequal length, part of each scored, across two slices, as ``poros train`` has."""
    per_slice, rows, prompt = 2, (6, 14, 10), 3
    config = PorosConfig(
        model_name_or_path=str(checkpoint.source), store="ram", max_seq_length=SEQ,
        store_path=str(checkpoint.root / "store"), per_device_batch_size=per_slice,
        lora_dropout=0.0, warmup_steps=0, save_steps=0, eval_steps=0, lora_rank=4, lora_alpha=8,
        output_dir=str(tmp_path), progress=False,
    )
    trainer = arms.CpuDouble.from_pretrained(str(checkpoint.source), config=config)
    torch.set_num_threads(1)
    lora = _lora(trainer)
    _randomize(lora)
    reference, pairs = _reference(checkpoint.source, lora, config.lora_alpha / config.lora_rank)
    ids = torch.randint(0, VOCAB, (3, SEQ), generator=torch.Generator().manual_seed(3)).tolist()
    tokens = [row[:n] for row, n in zip(ids, rows)]
    batch = Batch(
        [Datum.from_sequence(t, [0.0] * prompt + [1.0] * (len(t) - prompt)) for t in tokens],
        pad_id=7, pad_to=SEQ,
    )
    loss = Run.from_trainer(trainer).forward_backward(batch).loss

    total = 0.0
    for start in range(0, len(tokens), per_slice):
        part = tokens[start : start + per_slice]
        inputs = torch.full((len(part), SEQ), 7)
        mask = torch.zeros(len(part), SEQ, dtype=torch.long)
        targets = torch.zeros(len(part), SEQ, dtype=torch.long)
        weights = torch.zeros(len(part), SEQ)
        for i, row in enumerate(part):
            inputs[i, : len(row) - 1] = torch.tensor(row[:-1])
            mask[i, : len(row) - 1] = 1
            targets[i, : len(row) - 1] = torch.tensor(row[1:])
            weights[i, prompt - 1 : len(row) - 1] = 1.0
        logits = reference(input_ids=inputs, attention_mask=mask).logits.float()
        logprobs = torch.stack([-F.cross_entropy(logits[i], targets[i], reduction="none")
                                for i in range(len(part))])
        value = -(logprobs * mask.float() * weights).sum() / weights.sum()
        share = len(part) / len(tokens)
        (value * share).backward()
        total += share * float(value.detach())
    assert loss == total
    assert all(torch.equal(p.grad, pairs[n].grad) for n, p in lora.items())


def test_a_run_says_what_it_is(checkpoint, tmp_path):
    assert _arch_label(_trainer(checkpoint, tmp_path)).startswith("MoE · ")


def test_the_reference_runs_the_models_own_experts(checkpoint):
    """With no LoRA its loop is the model's, on the weights training sees."""
    model = _vanilla(checkpoint.source)
    gen = torch.Generator().manual_seed(9)
    for _, block in _sparse_blocks(model):
        block.experts.gate_up_proj.data = _decoded(block.experts.gate_up_proj)
        block.experts.down_proj.data = _decoded(block.experts.down_proj)
        hidden = torch.randn(32, 64, generator=gen, dtype=torch.bfloat16)
        _, weights, picks = (block.gate if hasattr(block, "gate") else block.router)(hidden)
        own = block.experts(hidden, picks, weights)
        written = DenseExperts(block.experts, None, 0.0)
        written.gate_up_proj.data = block.experts.gate_up_proj.data
        written.down_proj.data = block.experts.down_proj.data
        assert torch.equal(own, written(hidden, picks, weights))


def test_only_a_routed_expert_gets_a_gradient_and_moves(checkpoint, tmp_path):
    """A routed expert can have none too: the last token's own is not scored."""
    trainer = _trainer(checkpoint, tmp_path, weight_decay=0.0)
    ids = _ids()[:1, :3]  # few enough tokens that some experts see none
    got = _compare(checkpoint, trainer, batch={"input_ids": ids, "labels": ids.clone()})
    assert got.same_grads
    before = {n: p.detach().clone() for n, p in got.lora.items()}
    trainer.optimizer.step()
    for block, routed in got.routed.items():
        assert 0 < len(routed) < 12
        for factor in FACTORS:
            name = f"{block}.adapter.{factor}"
            moved = {e for e, row in enumerate(got.lora[name])
                     if not torch.equal(row, before[name][e])}
            graded = {e for e, row in enumerate(got.lora[name].grad) if row.abs().sum() > 0}
            assert graded and graded <= routed, name
            assert moved == graded, name


# -- The faults a gate must see ------------------------------------------------------------------


def _experts_of(trainer) -> list[Experts]:
    return [m for m in trainer.model.modules() if isinstance(m, Experts)]


def _wrong_matrix(trainer) -> None:
    for experts in _experts_of(trainer):
        a, b = experts.gate_up_proj[0], experts.gate_up_proj[1]
        a.start, b.start = b.start, a.start
        a, b = experts.down_proj[0], experts.down_proj[1]
        a.start, b.start = b.start, a.start


def _wrong_route(trainer) -> None:
    for _, block in _sparse_blocks(trainer.model):
        real = block.experts.forward
        block.experts.forward = lambda h, picks, w, real=real: real(h, (picks + 1) % 12, w)


def _wrong_adapter_row(trainer) -> None:
    for experts in _experts_of(trainer):
        lora = experts.adapter
        object.__setattr__(experts, "adapter", SimpleNamespace(
            delta=lora.delta, **{f: getattr(lora, f).roll(1, 0) for f in FACTORS}))


def _router_gone(trainer) -> None:
    for _, block in _sparse_blocks(trainer.model):
        if hasattr(block, "gate"):
            block.gate.weight.data.zero_()
        else:
            block.router.scale.data.zero_()


def _shared_expert_gone(trainer) -> None:
    for _, block in _sparse_blocks(trainer.model):
        for name in ("shared_expert", "shared_experts"):
            if hasattr(block, name):
                getattr(block, name).forward = lambda x: torch.zeros_like(x)


def _no_activation(trainer) -> None:
    for experts in _experts_of(trainer):
        experts.act_fn = nn.Identity()


FAULTS = {
    "another expert's matrices": (_wrong_matrix, sorted(CONFIGS)),
    "a token routed to the wrong expert": (_wrong_route, sorted(CONFIGS)),
    "the adapter of an expert no token picked": (_wrong_adapter_row, sorted(CONFIGS)),
    "the router dropped": (_router_gone, sorted(CONFIGS)),
    "the shared expert dropped": (_shared_expert_gone, SHARED_EXPERT),
    "the activation dropped": (_no_activation, sorted(CONFIGS)),
}


@pytest.mark.parametrize(
    ("fault", "family"),
    [(fault, family) for fault, (_, families) in FAULTS.items() for family in families],
)
def test_each_fault_shows(tmp_path, fault, family):
    checkpoint = _checkpoint(tmp_path, family)
    trainer = _trainer(checkpoint, tmp_path / "run")
    FAULTS[fault][0](trainer)
    got = _compare(checkpoint, trainer)
    assert not got.same_grads


@pytest.mark.parametrize("cap", [3.5, None], ids=["wrong", "none"])
def test_a_softcap_not_the_models_shows(tmp_path, monkeypatch, cap):
    checkpoint = _checkpoint(tmp_path, "gemma4_moe")
    trainer = _trainer(checkpoint, tmp_path / "run")
    monkeypatch.setattr(PorosTrainer, "_final_logit_softcapping", lambda self: cap)
    assert not _compare(checkpoint, trainer).same_grads


@pytest.mark.parametrize("family", sorted(REAL))
def test_a_swapped_norm_shows(tmp_path, monkeypatch, family):
    checkpoint = _checkpoint(tmp_path, family)
    real = load_file

    def swapped(path):
        small = real(path)
        for name in [n for n in small if ".input_layernorm." in n]:
            other = name.replace(".input_layernorm.", ".post_attention_layernorm.")
            small[name], small[other] = small[other], small[name]
        return small

    monkeypatch.setattr(metal_load, "load_file", swapped)
    assert not _compare(checkpoint, _trainer(checkpoint, tmp_path / "run")).same_grads


def test_a_head_taken_from_the_embed_shows(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path, "qwen3_5_moe")
    real = metal_load.parts_of
    monkeypatch.setattr(metal_load, "parts_of", lambda model: real(model)._replace(head=None))
    assert not _compare(checkpoint, _trainer(checkpoint, tmp_path / "run")).same_grads


def test_a_draft_head_is_left_behind_and_read_nothing_from(tmp_path):
    checkpoint = _checkpoint(tmp_path, "qwen3_5_moe")
    with safe_open(checkpoint.source / "model-00001-of-00002.safetensors", framework="pt") as f:
        names = list(f.keys())
    with safe_open(checkpoint.source / "model-00002-of-00002.safetensors", framework="pt") as f:
        names += list(f.keys())
    assert any(n.startswith("mtp.") and ".experts." in n for n in names)
    path = pack(checkpoint.source, tmp_path / "store", model_id="tiny/x",
                revision="0123456789abcdef")
    assert not [n for n in load_file(path / "small.safetensors") if n.startswith("mtp")]


def test_gate_and_up_fused_the_other_way_show(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path, "olmoe")
    fuse = experts_mod.fuse

    def reversed_halves(tensors):
        return {n: t._replace(halves=t.halves[::-1]) if isinstance(t, experts_mod.Fused) else t
                for n, t in fuse(tensors).items()}

    monkeypatch.setattr(experts_mod, "fuse", reversed_halves)
    got = _compare(checkpoint, _trainer(checkpoint, tmp_path / "run"))
    assert not got.same_grads


def test_a_restore_that_does_nothing_stops_the_run(tmp_path, monkeypatch):
    """The expert adapter's dropout draws in the block's Function: the witness sees it differ."""
    checkpoint = _checkpoint(tmp_path, "olmoe")
    fine = _trainer(checkpoint, tmp_path / "a", dropout=0.5)
    fine.train(arms.token_stream(VOCAB, SEQ, 2), steps=2)
    assert fine._witness.checked == 2 * 2 and fine._witness.mismatches == 0

    monkeypatch.setattr(blockwise, "torch", arms._NoRestore(torch))
    broken = _trainer(checkpoint, tmp_path / "b", dropout=0.5)
    with pytest.raises(PorosStepCheckError):
        broken.train(arms.token_stream(VOCAB, SEQ, 2), steps=2)


def test_nothing_adds_at_an_index(checkpoint, tmp_path):
    """MPS has no deterministic kernel for it, so a missed site would raise on the Mac alone."""
    trainer = _trainer(checkpoint, tmp_path, dropout=0.1)
    trainer._witness.remove()  # this runs each layer once, under grad
    _randomize(_lora(trainer))
    batch = _batch(None)
    hidden, mask, positions, _ = trainer._build_inputs(batch)
    names = set()
    for block in trainer.block_manager.blocks:
        trainer.block_manager.load_block_for_inference(block.block_id)
        hidden = hidden.detach().requires_grad_(True)
        out = trainer._forward_block_layers(block.block_id, hidden, mask, positions, False)
        seen, stack = set(), [out.grad_fn]
        while stack:
            fn = stack.pop()
            if fn is not None and fn not in seen:
                seen.add(fn)
                stack.extend(next_fn for next_fn, _ in fn.next_functions)
        names |= {type(fn).__name__ for fn in seen}
        hidden = out
    assert "_TakeBackward" in names
    assert not {n for n in names if "Index" in n}, names


# -- What lands on disk --------------------------------------------------------------------------


def test_a_slab_holds_the_bytes_cuda_holds(checkpoint):
    """Every expert's blobs are those of CUDA's slab, each other linear is the checkpoint's bf16."""
    path = pack(checkpoint.source, checkpoint.root / "pack", model_id="tiny/x",
                revision="0123456789abcdef")
    store = load_moe_streamed(str(checkpoint.source), torch.bfloat16)._poros_moe_store
    vanilla = AutoModelForCausalLM.from_pretrained(checkpoint.source,
                                                   dtype=torch.bfloat16).state_dict()
    decoder, path_to_layers = discover_layers(_vanilla(checkpoint.source))
    layers = [i for i, layer in enumerate(decoder)
              if hasattr(layer, "experts") or hasattr(layer.mlp, "experts")]
    expert_blobs = raw = 0
    with SssFile.open(path / "model.sss") as file:
        for layer in range(store.layout.num_layers + 8):
            try:
                entry = file.entry((layer, 0))
            except KeyError:
                break
            slab = torch.empty(entry.nbytes, dtype=torch.uint8)
            file.read_into((layer, 0), slab, verify=True)
            for view in entry.views:
                n = view["shape"][0] * view["shape"][1]
                if view.get("raw"):
                    assert ".experts." not in view["name"]
                    stored = slab[view["offset"] : view["offset"] + 2 * n]
                    want = vanilla[f"{path_to_layers}.{layer}.{view['name']}.weight"]
                    assert torch.equal(stored, want.flatten().view(torch.uint8)), view["name"]
                    raw += 1
                    continue
                kind, _, e = view["name"].rpartition(".")
                cuda = store._slabs[(layers.index(layer), int(e))]
                start, end = store._tensor_offsets[kind.rpartition(".")[2]]
                blob = slab[view["offset"] : view["offset"] + end - start]
                assert torch.equal(blob, cuda[start:end]), view["name"]
                expert_blobs += 1
    assert expert_blobs == 2 * store.layout.num_layers * store.layout.experts_per_layer
    assert raw > 0


def test_an_adapter_starts_as_cudas_does(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path)
    model = load_moe_streamed(str(checkpoint.source), torch.bfloat16)
    prepare_moe(model, MoeConfig(lora_rank=4, lora_alpha=8, seed=trainer.config.seed, look_ahead=0))
    theirs = {n: p for n, p in model.named_parameters() if p.requires_grad}
    mine = _lora(trainer)
    assert mine.keys() == theirs.keys() and mine
    for name, param in mine.items():
        assert param.dtype == theirs[name].dtype == torch.bfloat16
        assert torch.equal(param, theirs[name]), name


def _trained(checkpoint, out):
    trainer = _trainer(checkpoint, out, steps=2)
    _randomize(_lora(trainer))
    trainer.train(arms.token_stream(VOCAB, SEQ, 2), steps=2)
    trainer.save(str(out / "saved"))
    return trainer, out / "saved"


def test_a_saved_adapter_loads_in_vanilla_peft(checkpoint, tmp_path):
    trainer, saved = _trained(checkpoint, tmp_path)
    config = json.loads((saved / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["base_model_name_or_path"]
    assert config["target_parameters"] and not config["target_modules"]
    with safe_open(saved / "adapter_model.safetensors", framework="pt") as f:
        names = list(f.keys())
    blocks = _sparse_blocks(_vanilla(checkpoint.source))
    assert len(names) == 4 * len(blocks) > 0

    model = peft.PeftModel.from_pretrained(_vanilla(checkpoint.source), saved, is_trainable=False)
    assert sum("lora_" in n for n, _ in model.named_parameters()) == len(names)
    scaling = config["lora_alpha"] / config["r"]
    for name, block in blocks:
        wrapper = model.base_model.model.get_submodule(name).experts
        mine = trainer.model.get_submodule(name).adapter
        for weight, base, a, b in (
            ("down_proj", wrapper, mine.lora_A_down, mine.lora_B_down),
            ("gate_up_proj", wrapper.base_layer, mine.lora_A_gate_up, mine.lora_B_gate_up),
        ):
            assert base.parameter_name == weight
            delta = base.get_delta_weight("default")
            want = (b.float() @ a.float()) * scaling
            assert torch.allclose(delta.float(), want, rtol=2e-2, atol=1e-5), (name, weight)
    logits = model(input_ids=_ids()).logits
    assert torch.isfinite(logits).all()
    bare = _vanilla(checkpoint.source)(input_ids=_ids()).logits
    assert not torch.equal(logits, bare)


def test_a_saved_adapter_reads_back_byte_for_byte(checkpoint, tmp_path):
    trainer, saved = _trained(checkpoint, tmp_path)
    state = experts_mod.read_adapter(saved)
    lora = _lora(trainer)
    assert state.keys() == lora.keys()
    for name, param in lora.items():
        assert torch.equal(state[name].view(torch.int16), param.detach().view(torch.int16)), name


def test_a_merged_export_is_the_cuda_merge_of_the_same_adapter(checkpoint, tmp_path, monkeypatch):
    trainer, saved = _trained(checkpoint, tmp_path)
    monkeypatch.setattr(metal, "settle", lambda config: None)
    merged(trainer.config, saved, tmp_path / "merged")
    export_merged(str(checkpoint.source), experts_mod.read_adapter(saved), str(tmp_path / "cuda"),
                  scaling=trainer.config.lora_alpha / trainer.config.lora_rank)
    mine, theirs = (AutoModelForCausalLM.from_pretrained(tmp_path / d, dtype=torch.bfloat16)
                    for d in ("merged", "cuda"))
    vanilla = _vanilla(checkpoint.source).state_dict()
    moved = 0
    for name, tensor in mine.state_dict().items():
        assert torch.equal(tensor, theirs.state_dict()[name]), name
        moved += not torch.equal(tensor, vanilla[name])
    assert moved == 2 * len(_sparse_blocks(mine))
    assert torch.isfinite(mine(input_ids=_ids()).logits).all()


def test_resume_is_bit_equal_to_a_run_that_never_stopped(tmp_path):
    checkpoint = _checkpoint(tmp_path, "qwen2_moe")
    whole = _trainer(checkpoint, tmp_path / "whole", steps=4, dropout=0.05)
    whole.train(arms.token_stream(VOCAB, SEQ, 4), steps=4)

    first = _trainer(checkpoint, tmp_path / "split", steps=4, dropout=0.05)
    with pytest.raises(PorosCancelledError) as stopped:
        first.train(arms.token_stream(VOCAB, SEQ, 4), steps=4,
                    should_stop=lambda: first.global_step == 2)
    again = _trainer(checkpoint, tmp_path / "split", steps=4, dropout=0.05)
    again.train(arms.token_stream(VOCAB, SEQ, 4), steps=4, resume_from=stopped.value.checkpoint)
    assert again.global_step == whole.global_step
    for name, param in _lora(whole).items():
        assert torch.equal(param, _lora(again)[name]), name


def test_the_file_tier_changes_nothing(tmp_path):
    checkpoint = _checkpoint(tmp_path, "deepseek_v2")
    ram = _trainer(checkpoint, tmp_path / "a", dropout=0.05)  # seeds the one RNG: each in its turn
    ram.train(arms.token_stream(VOCAB, SEQ, 2), steps=2)
    disk = _trainer(checkpoint, tmp_path / "b", dropout=0.05, store="ssd")
    disk.train(arms.token_stream(VOCAB, SEQ, 2), steps=2)
    assert type(disk.model.poros_source).__name__ == "FileLayers"
    for name, param in _lora(ram).items():
        assert torch.equal(param, _lora(disk)[name]), name


def test_a_dense_model_is_not_one_of_these():
    assert not experts_mod.is_moe(AutoModelForCausalLM.from_config(
        tiny.tiny_config(vocab_size=64, layers=1)))
