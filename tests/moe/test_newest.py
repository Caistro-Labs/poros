"""poros.moe on the newest families, against what the model itself computes, with the faults a gate
must see. In fp32, since in bf16 the batched experts round in another order than the model's loop:
this holds the math, and streamed against resident holds the bits (``test_families``)."""

from __future__ import annotations

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from safetensors.torch import load_file, save_file  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from poros.core.arch_registry import blockwise_arch_check, registry_type  # noqa: E402
from poros.core.errors import PorosArchNotValidatedError  # noqa: E402
from poros.moe import quant  # noqa: E402
from poros.moe.adapters import ExpertLoraAdapter  # noqa: E402
from poros.moe.arch import MoeArch  # noqa: E402
from poros.moe.load import load_moe_streamed  # noqa: E402
from poros.moe.modules import ManagedMoeBlock  # noqa: E402
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402
from tests.metal import families as fam  # noqa: E402
from tests.metal.test_moe import DenseExperts  # noqa: E402

VOCAB = 96
SEQ = 16
FAMILIES = {
    "qwen3_5_moe": lambda: fam.real_config("qwen3_6_35b_a3b", hidden=64, vocab=VOCAB),
    "gemma4_moe": lambda: fam.real_config("gemma_4_26b_a4b", hidden=64, vocab=VOCAB),
}
FACTORS = ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down")


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def checkpoint(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    fam.write_fused(path, FAMILIES[request.param](), dtype=torch.float32)
    return path


def _prepared(path) -> nn.Module:
    model = load_moe_streamed(str(path), torch.float32)
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0, lora_rank=4, lora_alpha=8))
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.copy_(0.03 * torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model


def _blocks(model):
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "experts")]


class _Round(DenseExperts):
    """The reference's experts in fp32, over the NF4 round trip the store holds."""

    def __init__(self, fused, lora, scaling):
        nn.Module.__init__(self)
        self.num_experts, self.act_fn, self.scaling = fused.num_experts, fused.act_fn, scaling
        self.gate_up_proj = nn.Parameter(self._seen(fused.gate_up_proj), requires_grad=False)
        self.down_proj = nn.Parameter(self._seen(fused.down_proj), requires_grad=False)
        self.factors = nn.ParameterDict(
            {f: nn.Parameter(lora[f].detach().clone()) for f in FACTORS}
        )
        self.routed = set()

    @staticmethod
    def _seen(fused):
        return torch.stack([
            quant.dequantize(quant.quantize_tensor(w), w.shape, torch.float32) for w in fused
        ])


def _reference(path, lora):
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager", experts_implementation="eager"
    )
    model.config.get_text_config().use_cache = False
    model.requires_grad_(False)
    pairs = {}
    for name, block in _blocks(model):
        factors = {f: lora[f"{name}.adapter.{f}"] for f in FACTORS}
        block.experts = _Round(block.experts, factors, 2.0)
        pairs.update({f"{name}.adapter.{f}": block.experts.factors[f] for f in FACTORS})
    return model.train(), pairs


def _step(model):
    ids = torch.randint(0, VOCAB, (2, SEQ), generator=torch.Generator().manual_seed(3))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    return loss


def _agrees(mine, path) -> bool:
    lora = {n: p for n, p in mine.named_parameters() if p.requires_grad}
    reference, pairs = _reference(path, lora)
    mine.train()
    close = dict(rtol=1e-4, atol=1e-6)
    if not torch.allclose(_step(mine), _step(reference), **close):
        return False
    assert lora.keys() == pairs.keys() and lora
    return all(torch.allclose(p.grad, pairs[n].grad, **close) for n, p in lora.items())


def test_gemma_4_is_a_moe_model_only_with_its_block():
    moe, dense = FAMILIES["gemma4_moe"](), fam.config("gemma4", vocab=VOCAB)
    assert (registry_type(moe), registry_type(dense)) == ("gemma4_moe", "gemma4")
    assert registry_type(moe.to_dict()) == "gemma4_moe"
    assert registry_type(dense.to_dict()) == "gemma4"
    with pytest.raises(ValueError, match="prepare_moe supports"):
        MoeArch.from_config(dense)
    with pytest.raises(PorosArchNotValidatedError):  # the dense engine cannot quantize its experts
        blockwise_arch_check(registry_type(moe))


def test_experts_stored_one_by_one_load_as_those_stored_fused(tmp_path):
    """Transformers maps Qwen3.6's older layout onto the fused one: a checkpoint may hold either."""
    cfg = FAMILIES["qwen3_5_moe"]()
    fam.write_fused(tmp_path / "fused", cfg)
    split = {}
    for shard in (tmp_path / "fused").glob("*.safetensors"):
        for name, tensor in load_file(shard).items():
            if name.startswith("mtp."):
                continue
            if ".gate_up_proj" in name:
                head = name.removesuffix(".gate_up_proj")
                for e, row in enumerate(tensor):
                    gate, up = row.chunk(2)
                    split[f"{head}.{e}.gate_proj.weight"] = gate.clone()
                    split[f"{head}.{e}.up_proj.weight"] = up.clone()
            elif ".experts.down_proj" in name:
                head = name.removesuffix(".down_proj")
                split.update({f"{head}.{e}.down_proj.weight": row.clone()
                              for e, row in enumerate(tensor)})
            else:
                split[name] = tensor
    (tmp_path / "split").mkdir()
    save_file(split, tmp_path / "split" / "model.safetensors", metadata={"format": "pt"})
    cfg.save_pretrained(tmp_path / "split")
    fused = load_moe_streamed(str(tmp_path / "fused"), torch.bfloat16)._poros_moe_store
    split = load_moe_streamed(str(tmp_path / "split"), torch.bfloat16)._poros_moe_store
    assert fused._slabs.keys() == split._slabs.keys()
    assert all(torch.equal(slab, split._slabs[key]) for key, slab in fused._slabs.items())


def test_loss_and_every_grad_are_the_models_own(checkpoint):
    assert _agrees(_prepared(checkpoint), checkpoint)


def test_the_adapter_is_named_for_its_block(checkpoint):
    model = _prepared(checkpoint)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    arch = MoeArch.from_config(model.config)
    blocks = map(arch.block_module, arch.moe_layer_indices)
    assert names == {f"{b}.adapter.{f}" for b in blocks for f in FACTORS}
    assert all(isinstance(m, ExpertLoraAdapter) for n, m in model.named_modules()
               if n.endswith(".adapter"))


def test_a_wrong_expert_shows(checkpoint):
    model = _prepared(checkpoint)
    for module in model.modules():
        if isinstance(module, ManagedMoeBlock):
            real = module._remat_reuse
            module._remat_reuse = lambda h, i, w, real=real: real(h, (i + 1) % 12, w)
    assert not _agrees(model, checkpoint)


def test_a_wrong_norm_shows(checkpoint):
    model = _prepared(checkpoint)
    for _, layer in _layers(model):
        a, b = layer.input_layernorm, layer.post_attention_layernorm
        a.weight.data, b.weight.data = b.weight.data, a.weight.data
    assert not _agrees(model, checkpoint)


def _layers(model):
    from poros.core.arch_registry import discover_layers

    decoder, path = discover_layers(model)
    return [(f"{path}.{i}", layer) for i, layer in enumerate(decoder)]


def test_a_wrong_softcap_shows(tmp_path):
    fam.write_fused(tmp_path, FAMILIES["gemma4_moe"](), dtype=torch.float32)
    model = _prepared(tmp_path)
    model.config.get_text_config().final_logit_softcapping = 0.05
    assert not _agrees(model, tmp_path)


def test_a_head_taken_from_the_embed_shows(tmp_path):
    fam.write_fused(tmp_path, FAMILIES["qwen3_5_moe"](), dtype=torch.float32)
    model = _prepared(tmp_path)
    model.lm_head.weight.data = model.get_input_embeddings().weight.data
    assert not _agrees(model, tmp_path)


def test_a_router_given_another_input_shows(tmp_path):
    """Gemma 4 routes on its residual and runs its experts on a norm of it: why they stay apart."""
    fam.write_fused(tmp_path, FAMILIES["gemma4_moe"](), dtype=torch.float32)
    model = _prepared(tmp_path)
    for _, layer in _layers(model):
        real = layer.router.forward
        norm = layer.pre_feedforward_layernorm_2
        layer.router.forward = lambda h, real=real, norm=norm: real(norm(h))
    assert not _agrees(model, tmp_path)
