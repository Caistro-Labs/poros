"""Host RAM: frozen tables kept compact, pinned slabs shared, resident blocks with no host copy.

Every change here moves bytes, never values: the tests compare with ``torch.equal``.
"""
from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


def _tiny_qwen(tie: bool = False, layers: int = 4):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=tie, pad_token_id=299,
    )
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16)


# ── poros.core.tables ──────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1000, 24), (37,), (1,), (2,), (3,), (0, 8), (5, 3, 4)])
@pytest.mark.parametrize("chunk", [256, 1 << 20])
def test_cast_to_equals_a_plain_cast_bit_for_bit(monkeypatch, dtype, shape, chunk):
    from poros.core import tables

    monkeypatch.setattr(tables, "CHUNK_BYTES", chunk)  # many chunks and a halving tail, or one
    t = torch.randn(shape).to(dtype)
    t.view(-1)[:4] = torch.tensor([float("inf"), -0.0, float("nan"), 1e-40])[: t.numel()].to(dtype)
    out = tables.cast_to(t, torch.device("cpu"), torch.float32)
    assert out.dtype == torch.float32 and out.shape == t.shape
    want = t.float()
    nan = want.isnan()
    assert torch.equal(out.isnan(), nan)  # a NaN stays a NaN (its payload is the CPU's choice)
    assert torch.equal(out.view(torch.int32)[~nan], want.view(torch.int32)[~nan])  # every bit
    if dtype == torch.bfloat16:  # bf16 -> fp32 is a shift: NaN payloads too
        assert torch.equal(out.view(torch.int32), want.view(torch.int32))


def test_cast_to_allocates_only_its_result(monkeypatch):
    """The widening happens inside the result: torch.empty is called once, for it."""
    from poros.core import tables

    calls = []
    real = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **k: calls.append(a) or real(*a, **k))
    tables.cast_to(torch.randn(300, 8).to(torch.bfloat16), torch.device("cpu"), torch.float32)
    assert len(calls) == 1


def test_cast_to_returns_the_tensor_when_nothing_changes():
    from poros.core.tables import cast_to

    t = torch.randn(4, 4)
    assert cast_to(t, torch.device("cpu"), torch.float32) is t


class _Scaled(nn.Embedding):
    """Gemma's ScaledWordEmbedding: scales in the table's dtype."""

    def __init__(self, n: int, d: int, padding_idx: int, scale: float) -> None:
        super().__init__(n, d, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


@pytest.mark.parametrize("make", [
    lambda: nn.Embedding(500, 48, padding_idx=499),
    lambda: nn.Embedding(500, 48),
    lambda: _Scaled(500, 48, padding_idx=0, scale=48 ** 0.5),
])
def test_embed_as_equals_the_fp32_table_bit_for_bit(make):
    import copy

    from poros.core.tables import embed_as

    torch.manual_seed(1)
    compact = make()  # the weight in its load dtype; buffers as they are (PEFT casts parameters)
    compact.weight.data = compact.weight.data.to(torch.bfloat16)
    full = copy.deepcopy(compact)
    full.weight.data = compact.weight.data.float()  # PEFT's cast: exact
    ids = torch.randint(0, 500, (2, 33))
    ids[0, :3] = 499  # the padding row and repeats
    before = compact.weight.detach().clone()
    got = embed_as(compact, ids, torch.float32)
    want = full(ids)
    assert got.dtype == want.dtype == torch.float32
    assert torch.equal(got, want)
    assert torch.equal(compact.weight, before) and compact.weight.dtype == torch.bfloat16
    assert compact.padding_idx == full.padding_idx  # set back after the call


def test_embed_as_is_the_module_itself_when_dtypes_match():
    from poros.core.tables import embed_as

    emb = nn.Embedding(10, 4)
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(embed_as(emb, ids, torch.float32), emb(ids))


@pytest.mark.parametrize("tie", [False, True])
def test_vocab_tables_are_the_16bit_embedding_and_head_once(tie):
    from poros.core.tables import vocab_tables

    model = _tiny_qwen(tie=tie)
    got = vocab_tables(model)
    assert got[0] is model.get_input_embeddings().weight
    assert len(got) == (1 if tie else 2)
    if not tie:
        assert got[1] is model.get_output_embeddings().weight
    model.float()
    assert vocab_tables(model) == []


def test_kbit_prep_holds_the_tables_out_and_the_cast_restores_peft_exactly():
    """PEFT's preparation on one copy, Poros's (tables held out, cast after) on another:
    every parameter equal and of the same dtype once the held tables are cast."""
    pytest.importorskip("peft")
    import copy

    from peft import prepare_model_for_kbit_training

    from poros.core.tables import cast_to, vocab_tables
    from poros.train.model import _prepare_model_for_kbit_training

    ref = _tiny_qwen()
    ours = copy.deepcopy(ref)
    prepare_model_for_kbit_training(ref, use_gradient_checkpointing=True,
                                    gradient_checkpointing_kwargs={"use_reentrant": False})
    held = _prepare_model_for_kbit_training(ours, vocab_tables(ours))
    assert [p.dtype for p in held] == [torch.bfloat16, torch.bfloat16]
    norm = dict(ours.named_parameters())["model.norm.weight"]
    assert norm.dtype == torch.float32  # everything else was cast as before
    for p in held:
        p.data = cast_to(p.data, torch.device("cpu"), torch.float32)
    a, b = dict(ref.named_parameters()), dict(ours.named_parameters())
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].dtype == b[name].dtype, name
        assert torch.equal(a[name], b[name]), name
        assert a[name].requires_grad == b[name].requires_grad, name


@pytest.mark.parametrize("targets", [
    ["q_proj", "lm_head", "embed_tokens"],  # both tables LoRA targets
    ["q_proj", "lm_head"],
    "model.*q_proj|lm_head",  # PEFT reads a string as a regular expression
])
def test_a_table_lora_targets_is_cast_before_its_adapter_is_made(targets):
    """PEFT makes an adapter in its base layer's dtype: a table LoRA targets is not held out, so
    every adapter starts from the values PEFT's own preparation gives it."""
    pytest.importorskip("peft")
    import copy

    from peft import prepare_model_for_kbit_training

    from poros.train.model import (
        _attach_peft_lora,
        _prepare_model_for_kbit_training,
        _untargeted_tables,
    )

    ref = _tiny_qwen()
    ours = copy.deepcopy(ref)
    prepare_model_for_kbit_training(ref, use_gradient_checkpointing=True,
                                    gradient_checkpointing_kwargs={"use_reentrant": False})
    held = _prepare_model_for_kbit_training(ours, _untargeted_tables(ours, targets))
    targeted = {"lm_head"} | ({"embed_tokens"} if "embed_tokens" in targets else set())
    assert len(held) == 2 - len(targeted)  # only an untargeted table waits for its cast
    assert ours.lm_head.weight.dtype == torch.float32
    adapters = []
    for model in (ref, ours):
        torch.manual_seed(1)
        _, peft_model = _attach_peft_lora(model, lora_rank=4, lora_alpha=8, lora_dropout=0.0,
                                          lora_bias="none", target_modules=targets)
        adapters.append({n: p for n, p in peft_model.named_parameters() if p.requires_grad})
    assert adapters[0].keys() == adapters[1].keys()
    assert any("lm_head" in n for n in adapters[0])
    for name, p in adapters[0].items():
        assert p.dtype == adapters[1][name].dtype and torch.equal(p, adapters[1][name]), name


def test_all_linear_and_a_list_without_them_hold_both_tables():
    from poros.train.model import _untargeted_tables

    model = _tiny_qwen()
    both = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
    for targets in ("all-linear", None, ["q_proj", "v_proj"], ("o_proj",)):
        assert _untargeted_tables(model, targets) == both, targets


@pytest.mark.parametrize("cast_first", [True, False])
def test_a_failed_preparation_leaves_the_tables_as_peft_did(monkeypatch, cast_first):
    """PEFT failing after its cast leaves the tables owed (fp32 once placed, as PEFT's would be);
    failing before it leaves them in their load dtype, as PEFT left everything."""
    peft = pytest.importorskip("peft")
    from poros.core.tables import vocab_tables
    from poros.train.model import _prepare_model_for_kbit_training

    def fails(model, **kw):
        if cast_first:
            for p in model.parameters():
                if p.dtype == torch.bfloat16:
                    p.data = p.data.float()
        raise RuntimeError("no gradient checkpointing here")

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", fails)
    model = _tiny_qwen()
    tables = vocab_tables(model)
    owed = _prepare_model_for_kbit_training(model, tables)
    assert owed == (tables if cast_first else [])
    assert all(t.dtype == torch.bfloat16 and t.numel() > 0 for t in tables)


@pytest.mark.parametrize("defer", [False, True])
def test_poros_model_keeps_peft_s_host_cast_unless_asked(monkeypatch, defer):
    """PorosModel.from_pretrained, the public API, returns fp32 tables as PEFT made them; only
    defer_table_cast=True (what PorosTrainer passes) leaves them for the caller to cast."""
    pytest.importorskip("peft")
    from poros.train import model as M

    monkeypatch.setattr(M, "_load_model", lambda *a, **k: _tiny_qwen())
    pm = M.PorosModel.from_pretrained("tiny", quantization="nf4", adapter_backend="peft",
                                      defer_table_cast=defer)
    head = pm.model.get_output_embeddings().weight
    emb = pm.model.get_input_embeddings().weight
    want = torch.bfloat16 if defer else torch.float32
    assert (emb.dtype, head.dtype) == (want, want)
    assert [id(t) for t in pm.uncast_tables] == ([id(emb), id(head)] if defer else [])


# ── poros.sss.pinning.PinnedArena ──────────────────────────────────────────


def _arena(slab: int | None = None, total: int | None = None, device: str = "cpu"):
    """The arena with ordinary RAM (or no memory at all: ``meta``) standing in for page-locked
    slabs, so it runs on CPU-only hosts."""
    from poros.sss.pinning import PinnedArena

    class Arena(PinnedArena):
        SLAB = slab or PinnedArena.SLAB

        def _new_slab(self, nbytes):
            return torch.empty(nbytes, dtype=torch.uint8, device=device)

    return Arena(total)


def test_arena_carves_exact_copies_first_fit(monkeypatch):
    from poros.sss import pinning

    monkeypatch.setattr(pinning, "reserve", lambda n: True)
    arena = _arena(slab=64 << 10)
    sizes = [20000, 20000, 20000, 3000, 20000, 1500, 500]
    srcs = [torch.randint(0, 255, (n,), dtype=torch.uint8) for n in sizes]
    srcs.append(torch.randn(100, 7).to(torch.bfloat16))  # not bytes: its dtype and shape survive
    outs = [arena.pin(s) for s in srcs]
    for s, o in zip(srcs, outs):
        assert o.dtype == s.dtype and o.shape == s.shape and torch.equal(o, s)
        assert o.untyped_storage().data_ptr() != s.untyped_storage().data_ptr()
        assert (o.data_ptr() - o.untyped_storage().data_ptr()) % arena.ALIGN == 0
    # three 20000s fill slab 1 past room for a fourth; the small ones go back into its tail
    assert len(arena._slabs) == 2
    first = arena._slabs[0][0].untyped_storage().data_ptr()
    assert outs[3].untyped_storage().data_ptr() == first  # 3000 B went into slab 1's tail


# bytes of one layer's NF4 weights, as the streamed loader pins them (MLP x3, q/o, k/v)
_LAYERS = {
    "0.5B": ([2179072] * 3 + [401408] * 2 + [57344] * 2, 24),
    "7B": ([33947648] * 3 + [6422528] * 2 + [917504] * 2, 28),
    "32B": ([70778880] * 3 + [13107200] * 2 + [2621440] * 2, 64),
    "72B": ([121110528] * 3 + [33554432] * 2 + [4194304] * 2, 80),
}


@pytest.mark.parametrize("model", sorted(_LAYERS))
def test_arena_locks_what_it_stores_where_one_by_one_pinning_rounds(monkeypatch, model):
    """Every Qwen2.5 size, through ``pin`` on meta slabs (nothing is allocated): what the slabs
    lock is what is stored plus under one slab, and never more than pinning one by one."""
    from poros.sss import pinning

    monkeypatch.setattr(pinning, "reserve", lambda n: True)
    layer, n = _LAYERS[model]
    sizes = layer * n
    arena = _arena(total=sum(sizes), device="meta")
    for size in sizes:  # torch.empty touches no page: no RAM is used
        arena.pin(torch.empty(size, dtype=torch.uint8))
    locked = sum(slab.numel() for slab, _ in arena._slabs)
    stored, one_by_one = sum(sizes), sum(pinning.pinned_size(s) for s in sizes)
    assert all(slab.numel() & (slab.numel() - 1) == 0 for slab, _ in arena._slabs)  # powers of 2
    assert stored <= locked < stored + pinning.PinnedArena.SLAB
    assert locked <= one_by_one
    if model == "32B":  # 26.5 GiB pinned one by one -> under 15.5 GiB of slabs
        assert one_by_one / 2**30 == 26.5 and locked / 2**30 < 15.5


def test_arena_falls_back_to_the_tensor_when_refused(monkeypatch):
    from poros.sss import pinning

    t = torch.ones(1000, dtype=torch.uint8)
    monkeypatch.setattr(pinning, "reserve", lambda n: False)
    assert _arena(slab=1 << 16).pin(t) is t
    released = []
    monkeypatch.setattr(pinning, "reserve", lambda n: True)
    monkeypatch.setattr(pinning, "release", released.append)

    class Broken(pinning.PinnedArena):
        def _new_slab(self, nbytes):
            raise RuntimeError("no page-locked memory")

    assert Broken().pin(t) is t and released == [pinning.PinnedArena.SLAB]


def test_arena_stops_asking_once_the_host_refuses(monkeypatch, caplog):
    """A failed page-locked allocation is not retried for every later tensor (on some WSL2 hosts
    a failed attempt can break the process), and BlockManager says so once, as it did."""
    from poros.sss import pinning

    asked = []
    monkeypatch.setattr(pinning, "reserve", lambda n: asked.append(n) or True)
    monkeypatch.setattr(pinning, "release", lambda n: None)

    class Broken(pinning.PinnedArena):
        def _new_slab(self, nbytes):
            raise RuntimeError("no page-locked memory")

    arena = Broken()
    t, u = torch.ones(1000, dtype=torch.uint8), torch.ones(10, dtype=torch.uint8)
    assert arena.pin(t) is t and arena.refused and len(asked) == 1
    assert arena.pin(u) is u and len(asked) == 1

    from poros.core.residency import BlockManager

    bm = BlockManager(_tiny_qwen(layers=2), block_size=1)
    bm._pin_memory_enabled = True
    with caplog.at_level(logging.WARNING, logger="poros.core.residency"):
        for _ in range(2):
            assert torch.equal(bm._host_copy(t, Broken()), t)
    assert bm._pin_memory_enabled is False
    assert sum("Pinned-memory allocation is unavailable" in r.getMessage()
               for r in caplog.records) == 1


def test_arena_pins_a_tensor_alone_where_the_budget_has_no_room_for_a_slab(monkeypatch):
    """At the budget's tail a whole slab is refused, but the tensor's own power of two may still
    be granted: pinned alone, as one by one would have."""
    from poros.sss import pinning

    monkeypatch.setattr(pinning, "reserve", lambda n: n <= 2048)
    pinned = []
    t = torch.ones(1500, dtype=torch.uint8)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: pinned.append(self) or self)
    arena = _arena(slab=1 << 16)
    assert arena.pin(t) is t and pinned == [t] and not arena._slabs


# ── resident blocks ────────────────────────────────────────────────────────


def test_resident_ids_and_layers_spread_evenly():
    from poros.core.residency import resident_block_ids, resident_layer_names

    assert resident_block_ids(10, 0) == []
    assert resident_block_ids(10, 3) == [2, 5, 9]
    assert resident_block_ids(4, 9) == [0, 1, 2, 3]
    model = _tiny_qwen(layers=6)
    assert resident_layer_names(model, 2, 1) == ["model.layers.4", "model.layers.5"]
    # a short last block
    assert resident_layer_names(model, 4, 1) == ["model.layers.4", "model.layers.5"]


def test_resident_blocks_keep_no_host_copy_and_the_same_tensors(cpu_only):
    from poros.core.residency import BlockManager

    model = _tiny_qwen(layers=4)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    bm = BlockManager(model, block_size=1)
    bm.offload_all_to_cpu(keep=(1, 3))
    assert sorted(bm._cpu_weight_store) == [0, 2]
    bm.keep_resident([1, 3], torch.device("cpu"))
    assert sorted(bm._cpu_weight_store) == [0, 2]
    for bid in (0, 1, 2, 3):
        bm.load_block_for_inference(bid, device=torch.device("cpu"))
    got = model.state_dict()
    for k, v in ref.items():
        assert torch.equal(got[k], v), k
    bm.release_block(1)  # resident: stays
    assert bm.is_block_loaded(1)
    bm.release_block(0)
    assert not bm.is_block_loaded(0)
    bm.load_block_for_inference(0, device=torch.device("cpu"))  # released, not lost
    got = model.state_dict()
    for k, v in ref.items():
        assert torch.equal(got[k], v), k
