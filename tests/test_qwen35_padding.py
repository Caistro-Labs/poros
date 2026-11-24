"""A padded batch through the blockwise layer driver, on Qwen3.5's mixed layers.

Gated DeltaNet layers take a 2-D padding mask, attention layers a 4-D causal one. The
driver must hand each layer what the HF model hands it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from poros.train.trainer import PorosTrainer  # noqa: E402


def _backbone():
    torch.manual_seed(0)
    config = transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        tie_word_embeddings=False,
    )
    return modeling.Qwen3_5TextModel(config).eval()


def _driver(backbone):
    trainer = PorosTrainer.__new__(PorosTrainer)
    trainer._signature_cache = {}
    trainer._get_decoder_backbone = lambda: backbone
    trainer._get_backbone_module = lambda: backbone
    trainer._get_rotary_embedding_module = lambda: backbone.rotary_emb
    return trainer


def _drive(backbone, input_ids, attention_mask, *, with_index=True):
    trainer = _driver(backbone)
    hidden = backbone.embed_tokens(input_ids)
    positions = torch.arange(input_ids.shape[1])[None]
    dec = trainer._prepare_decoder_attention(hidden, attention_mask, positions)
    for index, layer in enumerate(backbone.layers):
        hidden = trainer._call_layer(
            layer,
            hidden,
            positions,
            layer_index=index if with_index else None,
            dec=dec,
        )
    return backbone.norm(hidden)


@pytest.mark.parametrize("with_index", [True, False])
@torch.no_grad()
def test_padded_batch_matches_the_hf_model(with_index):
    backbone = _backbone()
    input_ids = torch.randint(0, 64, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 7:] = 0  # the shape that crashed: right padding, as oasst1 pads
    expected = backbone(input_ids, attention_mask=attention_mask).last_hidden_state
    assert torch.equal(
        _drive(backbone, input_ids, attention_mask, with_index=with_index), expected
    )


@torch.no_grad()
def test_padding_reaches_the_gated_deltanet_layers():
    backbone = _backbone()
    input_ids = torch.randint(0, 64, (2, 12))
    padded = torch.ones_like(input_ids)
    padded[0, 7:] = 0
    masked = _drive(backbone, input_ids, padded)
    unmasked = _drive(backbone, input_ids, torch.ones_like(input_ids))
    assert not torch.equal(masked, unmasked)
    assert torch.equal(masked[1], unmasked[1])
