"""Every dense family through the real Mac path on the CPU, against what the model itself computes.

The reference is the real transformers model from ``from_pretrained``, under plain autograd, with
nothing of the Mac path in it but the NF4 round trip every frozen weight takes. Equal means
``torch.equal`` on the loss and on every adapter gradient. It says nothing about Metal or about real
shapes. Each mutation below is a fault this exists to catch: a gate that cannot fail proves nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

import peft  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2ForCausalLM  # noqa: E402

from poros.core.arch_registry import discover_layers  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.metal import load as load_mod  # noqa: E402
from poros.metal import pack as pack_mod  # noqa: E402
from poros.metal.export import merged  # noqa: E402
from poros.metal.head import ChunkedHead, EmbedRows  # noqa: E402
from poros.metal.nf4 import Nf4Linear  # noqa: E402
from poros.metal.pack import pack, parts_of, skeleton  # noqa: E402
from poros.moe import quant  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402
from tests.metal import families as fam  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

arms = tiny.arms
GEMMA = ("gemma3_text", "gemma3", "gemma4_text", "gemma4")
QWEN3_5 = ("qwen3_5_text", "qwen3_5", "qwen3_8")
FAMILIES = ("qwen3", *QWEN3_5, *GEMMA)
ATTENTION = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP = {"gate_proj", "up_proj", "down_proj"}
LINEAR_ATTENTION = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
CASES = [dict(family=f, tied=tied) for tied in (True, False) for f in FAMILIES]
# Gemma 3's wrapper has no softcap in its forward, and no real Gemma 3 sets one.
CASES += [dict(family=f, tied=True, softcap=3.0) for f in GEMMA if f != "gemma3"]


def _case_id(case: dict) -> str:
    return "-".join([case["family"], "tied" if case["tied"] else "untied",
                     *(["softcap"] if "softcap" in case else [])])


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _checkpoint(root, family, *, vocab=fam.SMALL_VOCAB, **kwargs):
    cfg = fam.config(family, vocab=vocab, **kwargs)
    fam.write_checkpoint(root / "source", family, cfg)
    return SimpleNamespace(family=family, cfg=cfg, source=root / "source", root=root, vocab=vocab)


@pytest.fixture(scope="module", params=CASES, ids=_case_id)
def checkpoint(request, tmp_path_factory):
    case = dict(request.param)
    return _checkpoint(tmp_path_factory.mktemp(_case_id(case)), case.pop("family"),
                       vocab=fam.VOCAB, **case)


@pytest.fixture(scope="module", params=FAMILIES)
def one_per_family(request, tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp(request.param), request.param)


def _trainer(checkpoint, out, **over):
    config = PorosConfig(
        model_name_or_path=str(checkpoint.source), store="ram", max_seq_length=16,
        store_path=str(checkpoint.root / "store"), gradient_accumulation_steps=1, max_steps=2,
        lora_dropout=over.pop("dropout", 0.0), learning_rate=1e-2, warmup_steps=0, save_steps=0,
        eval_steps=0, output_dir=str(out), progress=False, lora_rank=4, lora_alpha=8, **over,
    )
    # A Mac plans a local checkpoint of a family its table has as that family's smallest model
    # (poros.metal.table._as_catalog): this tiny Gemma 4 as the 26B, 22 GiB, which a Mac with
    # 14 GB refuses. What is tested here is the arithmetic, not that fit: the gate is not asked.
    return arms.CpuDouble.from_pretrained(str(checkpoint.source), config=config,
                                          check_fit=False)


def _lora(trainer) -> dict[str, nn.Parameter]:
    return {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}


def _ids(checkpoint):
    return torch.randint(0, checkpoint.vocab, (1, 16), generator=torch.Generator().manual_seed(3))


def _hf_kwargs(family, ids) -> dict:
    """What a wrapper asks of a text-only batch in training."""
    return {"gemma3": {"token_type_ids": torch.zeros_like(ids)},
            "gemma4": {"mm_token_type_ids": torch.zeros_like(ids)}}.get(family, {})


def _compare(checkpoint, trainer, *, padded=None) -> SimpleNamespace:
    """One microstep on the real path and on the reference, from the same random adapter."""
    gen = torch.Generator().manual_seed(5)
    lora = _lora(trainer)
    with torch.no_grad():
        for param in lora.values():  # lora_B starts at zero, which no gradient of lora_A could see
            param.copy_(0.03 * torch.randn(param.shape, generator=gen, dtype=param.dtype))
    reference, pairs = fam.dense_reference(checkpoint.source, lora, trainer.config.lora_alpha
                                           / trainer.config.lora_rank)
    ids, mask = _ids(checkpoint), None
    batch = {"input_ids": ids, "labels": ids.clone()}
    kwargs = _hf_kwargs(checkpoint.family, ids)
    if padded:
        real = torch.arange(16) < 9 if padded == "right" else torch.arange(16) >= 7
        mask = real.long().unsqueeze(0)
        labels = ids.masked_fill(mask == 0, -100)
        batch.update(attention_mask=mask, labels=labels)
        kwargs["attention_mask"] = mask
    loss = trainer._forward_blockwise_adapter_model(batch)
    loss.backward()
    want = trainer._compute_loss(reference(input_ids=ids, **kwargs).logits, batch["labels"])
    want.backward()
    return SimpleNamespace(
        same_loss=torch.equal(loss, want),
        same_grads=all(torch.equal(p.grad, pairs[n].grad) for n, p in lora.items()),
        nonzero=sum(bool(p.grad.abs().sum() > 0) for p in lora.values()),
        n=len(lora),
    )


@pytest.mark.parametrize("padded", [None, "right", "left"], ids=["full", "right", "left"])
def test_grads_are_bit_equal_to_plain_autograd_on_the_real_model(checkpoint, tmp_path, padded):
    got = _compare(checkpoint, _trainer(checkpoint, tmp_path), padded=padded)
    assert got.same_loss
    assert got.same_grads
    assert got.nonzero == got.n > 0


# -- The real trainer, across arms and steps -----------------------------------------------------


@pytest.fixture(scope="module")
def packed(one_per_family):
    path = pack(one_per_family.source, one_per_family.root / "store", model_id="tiny/x",
                revision="0123456789abcdef")
    cfg = AutoConfig.from_pretrained(one_per_family.source, attn_implementation="eager",
                                     use_cache=False)
    return SimpleNamespace(path=path, cfg=cfg)


def _differing(a, b) -> set[str]:
    return {name for name, value in a.surfaces().items() if value != b.surfaces()[name]}


@pytest.mark.parametrize("dropout", [0.05, 0.0])
@pytest.mark.parametrize("arm", ["mono", "plain"])
def test_the_loop_is_byte_equal_to_each_arm(packed, tmp_path, arm, dropout):
    loop, _ = tiny.run_arm(packed.path, packed.cfg, "loop", tmp_path / "a", dropout=dropout)
    other, _ = tiny.run_arm(packed.path, packed.cfg, arm, tmp_path / "b", dropout=dropout)
    assert _differing(loop, other) == set()
    assert loop.nonfinite == other.nonfinite == 0


def test_the_file_tier_changes_nothing(packed, tmp_path):
    ram, _ = tiny.run_arm(packed.path, packed.cfg, "loop", tmp_path / "a", dropout=0.05)
    disk, _ = tiny.run_arm(packed.path, packed.cfg, "loop", tmp_path / "b", dropout=0.05,
                           file_rows=True)
    assert _differing(ram, disk) == set()


# -- What the model says, packed and placed ------------------------------------------------------


def test_every_tensor_a_bf16_load_holds_is_in_the_pack(one_per_family):
    checkpoint = one_per_family
    path = pack(checkpoint.source, checkpoint.root / "store", model_id="tiny/x",
                revision="0123456789abcdef")
    vanilla = AutoModelForCausalLM.from_pretrained(checkpoint.source, dtype=torch.bfloat16)
    state = vanilla.state_dict()
    model = skeleton(AutoConfig.from_pretrained(checkpoint.source))
    parts = parts_of(model)
    small = load_file(path / "small.safetensors")
    assert small.keys() == set(parts.small)
    for name, tensor in small.items():
        assert tensor.dtype == state[name].dtype == torch.bfloat16, name
        assert torch.equal(tensor, state[name]), name

    file = pack_mod.SssFile.open(path / "model.sss")
    for layer, matrices in enumerate(parts.matrices):
        entry = file.entry((layer, 0))
        slab = torch.empty(entry.nbytes, dtype=torch.uint8)
        file.read_into((layer, 0), slab, verify=True)
        assert [v["name"] for v in entry.views] == matrices
        for view in entry.views:
            weight = state[f"{parts.layers}.{layer}.{view['name']}.weight"]
            blob = quant.quantize_tensor(weight)
            assert torch.equal(slab[view["offset"] : view["offset"] + blob.numel()], blob)
    for block, name in ((pack_mod.EMBED_BLOCK, parts.embed), (pack_mod.HEAD_BLOCK, parts.head)):
        assert ((block, 0) in {e.key for e in file.entries}) == (name is not None)
        if name is not None:
            raw = torch.empty(file.entry((block, 0)).nbytes, dtype=torch.uint8)
            file.read_into((block, 0), raw, verify=True)
            assert torch.equal(raw, state[name].view(torch.uint8).reshape(-1)), name


def test_no_weight_is_left_empty_and_the_rest_of_a_wrapper_is_gone(checkpoint, tmp_path):
    model = _trainer(checkpoint, tmp_path).model
    assert not [n for n, p in model.named_parameters() if p.is_meta]
    assert isinstance(model.get_input_embeddings(), EmbedRows)
    assert isinstance(model.get_output_embeddings(), ChunkedHead)
    tied = model.get_output_embeddings().rows is model.get_input_embeddings().rows
    assert tied == (checkpoint.cfg.get_text_config().tie_word_embeddings)
    for layer in discover_layers(model)[0]:
        assert not [n for n, m in layer.named_modules() if type(m) is nn.Linear and "lora" not in n]
    assert not [n for n, _ in model.named_modules() if "vision" in n or "visual" in n]


@pytest.mark.parametrize(("family", "targets"), [
    ("qwen3", ATTENTION | MLP),
    ("qwen3_5_text", ATTENTION | MLP | LINEAR_ATTENTION),
    ("qwen3_5", ATTENTION | MLP | LINEAR_ATTENTION),
    ("qwen3_8", ATTENTION | MLP | LINEAR_ATTENTION),
    ("gemma3_text", ATTENTION | MLP),
    ("gemma3", ATTENTION | MLP),
    ("gemma4_text", ATTENTION | MLP),
    ("gemma4", ATTENTION | MLP),
])
def test_lora_targets_exactly_the_linears_in_the_decoder_layers(tmp_path, family, targets):
    trainer = _trainer(_checkpoint(tmp_path, family), tmp_path / "run")
    layers, path = discover_layers(trainer.model)
    linears = [m for layer in layers for m in layer.modules() if isinstance(m, Nf4Linear)]
    lora = _lora(trainer)
    assert len(lora) == 2 * len(linears) > 0
    assert all(name.startswith(path) for name in lora)
    assert {name.split(".lora_")[0].rpartition(".")[2] for name in lora} == targets


def test_the_embedding_scale_is_the_one_a_bf16_load_rounds_to(tmp_path):
    checkpoint = _checkpoint(tmp_path, "gemma3_text")
    embed = _trainer(checkpoint, tmp_path / "run").model.get_input_embeddings()
    vanilla = AutoModelForCausalLM.from_pretrained(checkpoint.source, dtype=torch.bfloat16)
    want = vanilla.get_input_embeddings().embed_scale
    assert want.dtype == torch.bfloat16 and embed.scale == float(want)
    assert embed.scale != fam.SHARED["hidden_size"] ** 0.5
    assert not [*embed.buffers()]  # a buffer would follow the module to the GPU, away from its rows


def test_a_skeleton_moves_no_rng(tmp_path):
    """Qwen3.5 draws in its constructor. A pack that is reused must train what a fresh one does."""
    checkpoint = _checkpoint(tmp_path, "qwen3_5_text")
    torch.manual_seed(11)
    before = torch.get_rng_state()
    pack(checkpoint.source, tmp_path / "store", model_id="tiny/x", revision="0123456789abcdef")
    load_mod.build(tmp_path / "store" / "tiny--x-01234567", checkpoint.cfg)
    assert torch.equal(before, torch.get_rng_state())


# -- Qwen2 stays as it was -----------------------------------------------------------------------


def test_a_qwen2_skeleton_has_the_buffers_a_plain_build_has():
    cfg = fam.config("qwen2", vocab=fam.SMALL_VOCAB)
    plain = dict(Qwen2ForCausalLM(cfg).named_buffers())
    built = dict(skeleton(cfg).named_buffers())
    assert plain.keys() == built.keys()
    for name, buffer in built.items():
        assert buffer.dtype == plain[name].dtype and torch.equal(buffer, plain[name]), name


def test_qwen2_norms_stay_bf16_and_its_rows_unscaled(tmp_path):
    cfg = tiny.tiny_config(vocab_size=fam.SMALL_VOCAB)
    _, path = tiny.packed(tmp_path, cfg)
    model = load_mod.build(path, cfg)
    for name, param in model.named_parameters():
        if "norm" in name or name.endswith(".bias"):
            assert param.dtype == torch.bfloat16, name
    assert model.model.embed_tokens.scale is None


# -- A fault this must catch ---------------------------------------------------------------------


@pytest.fixture
def softcapped(tmp_path):
    return _checkpoint(tmp_path, "gemma4_text", softcap=3.0)


def _assert_caught(checkpoint, trainer, *, padded=None):
    got = _compare(checkpoint, trainer, padded=padded)
    assert not (got.same_loss and got.same_grads)


def test_mutation_a_wrong_softcap_is_caught(softcapped, tmp_path, monkeypatch):
    monkeypatch.setattr(PorosTrainer, "_final_logit_softcapping", lambda self: 3.5)
    _assert_caught(softcapped, _trainer(softcapped, tmp_path / "run"))


def test_mutation_no_softcap_is_caught(softcapped, tmp_path, monkeypatch):
    monkeypatch.setattr(PorosTrainer, "_final_logit_softcapping", lambda self: None)
    _assert_caught(softcapped, _trainer(softcapped, tmp_path / "run"))


@pytest.mark.parametrize("family", FAMILIES)
def test_mutation_a_swapped_norm_is_caught(tmp_path, monkeypatch, family):
    checkpoint = _checkpoint(tmp_path, family)
    real = load_file

    def swapped(path):
        small = real(path)
        for name in [n for n in small if ".input_layernorm." in n]:
            other = name.replace(".input_layernorm.", ".post_attention_layernorm.")
            small[name], small[other] = small[other], small[name]
        return small

    monkeypatch.setattr(load_mod, "load_file", swapped)
    _assert_caught(checkpoint, _trainer(checkpoint, tmp_path / "run"))


@pytest.mark.parametrize("family", FAMILIES)
def test_mutation_a_head_taken_from_the_embed_is_caught(tmp_path, monkeypatch, family):
    checkpoint = _checkpoint(tmp_path, family, tied=False)
    real = load_mod.parts_of
    monkeypatch.setattr(load_mod, "parts_of", lambda model: real(model)._replace(head=None))
    _assert_caught(checkpoint, _trainer(checkpoint, tmp_path / "run"))


@pytest.mark.parametrize("family", GEMMA)
def test_mutation_no_embedding_scale_is_caught(tmp_path, monkeypatch, family):
    checkpoint = _checkpoint(tmp_path, family)
    monkeypatch.setattr(load_mod, "EmbedRows", lambda rows, scale: EmbedRows(rows))
    _assert_caught(checkpoint, _trainer(checkpoint, tmp_path / "run"))


@pytest.mark.parametrize("family", GEMMA)
def test_mutation_an_embedding_scale_not_rounded_is_caught(tmp_path, monkeypatch, family):
    checkpoint = _checkpoint(tmp_path, family)
    exact = fam.SHARED["hidden_size"] ** 0.5 + 0.1
    monkeypatch.setattr(load_mod, "EmbedRows", lambda rows, scale: EmbedRows(rows, scale=exact))
    _assert_caught(checkpoint, _trainer(checkpoint, tmp_path / "run"))


@pytest.mark.parametrize("family", QWEN3_5)
def test_mutation_a_conv_left_in_bf16_is_caught(tmp_path, family):
    checkpoint = _checkpoint(tmp_path, family)
    trainer = _trainer(checkpoint, tmp_path / "run")
    convs = [m for m in trainer.model.modules() if isinstance(m, nn.Conv1d)]
    assert convs and all(m.weight.dtype == torch.float32 for m in convs)
    for conv in convs:
        conv.weight.data = conv.weight.data.to(torch.bfloat16)
    _assert_caught(checkpoint, trainer)


@pytest.mark.parametrize("family", QWEN3_5)
def test_mutation_linear_attention_with_no_padding_mask_is_caught(tmp_path, monkeypatch, family):
    import transformers.masking_utils as masking

    checkpoint = _checkpoint(tmp_path, family)
    monkeypatch.setattr(masking, "create_recurrent_attention_mask", lambda **kw: None)
    # Left: padding on the right comes after every real token, where it cannot matter.
    _assert_caught(checkpoint, _trainer(checkpoint, tmp_path / "run"), padded="left")


def test_mutation_a_layer_scalar_left_at_its_default_is_caught(tmp_path):
    checkpoint = _checkpoint(tmp_path, "gemma4_text")
    trainer = _trainer(checkpoint, tmp_path / "run")
    for layer in discover_layers(trainer.model)[0]:
        layer.layer_scalar.fill_(1.0)
    _assert_caught(checkpoint, trainer)


# -- What it cannot place ------------------------------------------------------------------------


def _edited(checkpoint, tmp_path, *, add=None, change=None, cfg=None):
    source = tmp_path / "edited"
    source.mkdir()
    tensors = load_file(checkpoint.source / "model.safetensors")
    tensors.update(add or {})
    for name, fn in (change or {}).items():
        tensors[name] = fn(tensors[name])
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text(cfg or (checkpoint.source / "config.json").read_text())
    return source


def _pack(source, tmp_path):
    return pack(source, tmp_path / "store", model_id="tiny/x", revision="0123456789abcdef")


def test_what_the_model_says_to_ignore_is_left_behind(tmp_path):
    checkpoint = _checkpoint(tmp_path, "qwen3_5")
    add = {"mtp.fc.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
           "model.visual.extra.weight": torch.zeros(4, dtype=torch.bfloat16)}
    path = _pack(_edited(checkpoint, tmp_path, add=add), tmp_path)
    assert not [n for n in load_file(path / "small.safetensors") if "mtp" in n or "visual" in n]


def test_a_tensor_the_model_has_no_name_for_is_refused(tmp_path):
    checkpoint = _checkpoint(tmp_path, "qwen3_5")
    add = {"model.language_model.extra.weight": torch.zeros(4, dtype=torch.bfloat16)}
    with pytest.raises(ValueError, match="no place for .*extra"):
        _pack(_edited(checkpoint, tmp_path, add=add), tmp_path)
    assert not (tmp_path / "store").exists()


def test_a_linear_outside_the_decoder_layers_is_refused(tmp_path):
    """Per-layer embeddings feed each layer an input the block loop has no way to hand it."""
    cfg = fam.config("gemma4_text", vocab=fam.SMALL_VOCAB)
    cfg.hidden_size_per_layer_input = 8
    cfg.vocab_size_per_layer_input = 100
    fam.write_checkpoint(tmp_path / "source", "gemma4_text", cfg)
    with pytest.raises(ValueError, match="no place for .*per_layer_model_projection"):
        _pack(tmp_path / "source", tmp_path)


def test_a_head_with_a_bias_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "source"
    tiny.write_checkpoint(source, tiny.tiny_config(vocab_size=fam.SMALL_VOCAB, layers=1))
    real = pack_mod.skeleton

    def biased(config):
        model = real(config)
        hidden, vocab = model.lm_head.in_features, model.lm_head.out_features
        with pack_mod.init_empty_weights():
            model.lm_head = nn.Linear(hidden, vocab)
        return model

    monkeypatch.setattr(pack_mod, "skeleton", biased)
    with pytest.raises(ValueError, match="no place for .*the head's bias"):
        _pack(source, tmp_path)


def test_a_matrix_that_disagrees_with_the_config_is_refused(tmp_path):
    checkpoint = _checkpoint(tmp_path, "gemma3_text")
    name = "model.layers.1.mlp.up_proj.weight"
    source = _edited(checkpoint, tmp_path, change={name: lambda t: t[:-1]})
    with pytest.raises(ValueError, match="up_proj.* its config says"):
        _pack(source, tmp_path)


def test_a_matrix_that_is_not_bf16_is_refused_and_a_small_one_is_rounded(tmp_path):
    checkpoint = _checkpoint(tmp_path, "qwen3_5_text")
    stored = load_file(checkpoint.source / "model.safetensors")
    assert stored["model.layers.0.linear_attn.A_log"].dtype == torch.float32
    path = _pack(checkpoint.source, tmp_path)
    small = load_file(path / "small.safetensors")
    assert all(t.dtype == torch.bfloat16 for t in small.values())
    a_log = small["model.layers.0.linear_attn.A_log"]
    assert torch.equal(a_log, stored["model.layers.0.linear_attn.A_log"].to(torch.bfloat16))
    assert not torch.equal(a_log.float(), stored["model.layers.0.linear_attn.A_log"])

    name = "model.layers.0.mlp.up_proj.weight"
    source = _edited(checkpoint, tmp_path, change={name: lambda t: t.float()})
    with pytest.raises(ValueError, match="up_proj.*holds bf16 only"):
        pack(source, tmp_path / "other", model_id="tiny/x", revision="0123456789abcdef")


# -- An adapter out, a merged model out ----------------------------------------------------------


def _trained(checkpoint, out):
    trainer = _trainer(checkpoint, out)
    trainer.train(arms.token_stream(checkpoint.vocab, 16, 3), steps=2)
    trainer.save(str(out))
    return trainer


def test_the_adapter_loads_in_vanilla_peft(one_per_family, tmp_path):
    checkpoint = one_per_family
    trainer = _trained(checkpoint, tmp_path / "run")
    saved = load_file(tmp_path / "run" / "adapter_model.safetensors")
    lora = _lora(trainer)
    assert len(saved) == len(lora)

    vanilla = AutoModelForCausalLM.from_pretrained(checkpoint.source, dtype=torch.bfloat16)
    loaded = peft.PeftModel.from_pretrained(vanilla, tmp_path / "run")
    params = dict(loaded.named_parameters())
    wrapped = {n for n, p in params.items() if "lora_" in n}
    assert wrapped == {f"base_model.model.{name}" for name in lora}  # and nothing in a vision tower
    for name, param in lora.items():
        other = params[f"base_model.model.{name}"]
        assert torch.equal(other.detach().float(), param.detach().float()), name
    ids = _ids(checkpoint)
    logits = loaded(input_ids=ids, **_hf_kwargs(checkpoint.family, ids)).logits
    assert torch.isfinite(logits).all()


def test_merged_export_folds_the_adapter_under_the_checkpoints_own_names(one_per_family, tmp_path):
    checkpoint = one_per_family
    trainer = _trained(checkpoint, tmp_path / "run")
    trainer.config.model_revision = load_mod.fetch(trainer.config)[2]
    merged(trainer.config, tmp_path / "run", tmp_path / "merged")
    before = load_file(checkpoint.source / "model.safetensors")
    after = load_file(tmp_path / "merged" / "model.safetensors")
    assert before.keys() == after.keys()

    base = AutoModelForCausalLM.from_pretrained(checkpoint.source, dtype=torch.bfloat16)
    out = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.bfloat16)
    scaling = trainer.config.lora_alpha / trainer.config.lora_rank
    lora, state = _lora(trainer), base.state_dict()
    folded = 0
    for name, value in out.state_dict().items():
        a = lora.get(name.replace(".weight", ".lora_A.default.weight"))
        if a is None:
            assert torch.equal(value, state[name]), name
            continue
        b = lora[name.replace(".weight", ".lora_B.default.weight")]
        seen = quant.dequantize(quant.quantize_tensor(state[name]), state[name].shape,
                                torch.bfloat16)
        delta = scaling * (b.detach().float() @ a.detach().float())
        want = (seen.float() + delta).to(torch.bfloat16)
        assert torch.equal(value, want), name
        folded += 1
    assert folded == len(lora) // 2 > 0
