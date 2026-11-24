"""The trainer's layer-by-layer path equals HF's own forward, padding mask included (CPU).

Configs without ``layer_types`` (qwen3_moe, olmoe, llama) once got the raw 2-D mask handed
to their layers -- non-causal attention -- while HF's Model.forward builds a causal one.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from poros.core.config import PorosConfig  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

COMMON = dict(vocab_size=64, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2)


def _model(kind):
    torch.manual_seed(0)
    if kind == "olmoe":
        cfg = transformers.OlmoeConfig(num_experts=8, num_experts_per_tok=4, **COMMON)
        return transformers.OlmoeForCausalLM(cfg)
    if kind == "qwen3_moe":
        cfg = transformers.Qwen3MoeConfig(head_dim=16, moe_intermediate_size=32, num_experts=8,
                                          num_experts_per_tok=4, **COMMON)
        return transformers.Qwen3MoeForCausalLM(cfg)
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(**COMMON))


@pytest.mark.parametrize("kind", ["qwen3_moe", "olmoe", "llama"])
@pytest.mark.parametrize("with_mask", [True, False])
def test_layerwise_equals_hf_forward(kind, with_mask):
    model = _model(kind).float().eval()
    tr = PorosTrainer.__new__(PorosTrainer)
    tr.model, tr.device = model, torch.device("cpu")
    tr.block_manager, tr._signature_cache = None, {}
    tr._metal = False  # what __init__ sets off a Mac: the CUDA path's masks
    tr._compact_tables = {}  # what __init__ sets: no table kept compact
    tr.config = PorosConfig(model_name_or_path="x", lora_dropout=0.0)
    ids = torch.randint(0, 64, (1, 13), generator=torch.Generator().manual_seed(1))
    batch = {"input_ids": ids, "labels": ids.clone()}
    if with_mask:
        batch["attention_mask"] = torch.ones_like(ids)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=batch.get("attention_mask")).logits
        h, attn, pos, _ = tr._build_inputs(batch)
        dec = tr._prepare_decoder_attention(h, attn, pos)
        for i, layer in enumerate(model.model.layers):
            h = tr._call_layer(layer, h, pos, layer_index=i, dec=dec)
        got = tr._logits_from_hidden(h)
    assert torch.equal(got, ref)
