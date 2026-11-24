"""Merged export of an adapter that adapts the input embedding (lora_target_modules naming
embed_tokens): its LoRA is folded as PEFT folds it, and a trained tensor the export does not
recognise is refused, never dropped."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file, save_file  # noqa: E402

from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosExportError  # noqa: E402
from poros.train.export import export_merged_dense, nf4_roundtrip  # noqa: E402


def _base_and_adapter(tmp_path):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=False)
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(base)
    model = transformers.Qwen2ForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    pm = peft.get_peft_model(model, peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["embed_tokens", "q_proj"], init_lora_weights=False))
    pm.save_pretrained(adapter)
    return base, adapter


def test_the_embedding_lora_is_merged_as_peft_merges_it(tmp_path):
    base, adapter = _base_and_adapter(tmp_path)
    export_merged_dense(PorosConfig(model_name_or_path=str(base)),
                        str(adapter), str(tmp_path / "merged"), nf4_device="cpu")
    merged = load_file(str(tmp_path / "merged" / "model.safetensors"))
    # The embedding (never NF4): PEFT's merge in fp32, rounded once to the checkpoint's bf16
    fp32 = transformers.Qwen2ForCausalLM.from_pretrained(base, dtype=torch.float32)
    expected = peft.PeftModel.from_pretrained(fp32, str(adapter)).merge_and_unload().state_dict()
    before = load_file(str(base / "model.safetensors"))
    key = "model.embed_tokens.weight"
    assert torch.equal(merged[key], expected[key].to(torch.bfloat16))
    assert not torch.equal(merged[key], before[key])  # the delta is in it
    for key in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.self_attn.q_proj.weight"):
        assert not torch.equal(merged[key], before[key]), key  # NF4's round trip plus its delta
    # a linear with no delta: the NF4 round trip training saw
    seen, _ = nf4_roundtrip(torch.bfloat16, "cpu")
    key = "model.layers.0.mlp.up_proj.weight"
    assert torch.equal(merged[key], seen(before[key]).to(torch.bfloat16))


@pytest.mark.parametrize("extra", [
    "base_model.model.model.layers.0.mlp.up_proj.lora_magnitude_vector",
    "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",  # half a pair
])
def test_a_trained_tensor_the_export_does_not_fold_is_refused(tmp_path, extra):
    base, adapter = _base_and_adapter(tmp_path)
    path = adapter / "adapter_model.safetensors"
    state = load_file(str(path))
    state[extra] = torch.zeros(4, 64, dtype=torch.bfloat16)
    save_file(state, str(path))
    with pytest.raises(PorosExportError, match="does not match this model|does not fold|half"):
        export_merged_dense(PorosConfig(model_name_or_path=str(base)),
                            str(adapter), str(tmp_path / "merged"), nf4_device="cpu")
    assert not (tmp_path / "merged" / "model.safetensors").exists()


@pytest.mark.parametrize("target", ["embed_tokens", "lm_head"])
def test_a_tied_models_table_delta_is_refused_before_anything_is_written(tmp_path, target):
    """On a model that ties its head to its embedding table, PEFT trains
    the embedding's LoRA with the head still the base table, and a merged checkpoint stores one
    table for both, so the merged model's head carried the delta (logits off by 600 on a tiny
    model). The merge is refused, naming the table, and nothing is written."""
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=True)
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "merged"
    transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(base)
    model = transformers.Qwen2ForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    peft.get_peft_model(model, peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=[target, "q_proj"],
        init_lora_weights=False)).save_pretrained(adapter)
    with pytest.raises(PorosExportError, match="ties its LM head to its embedding table"):
        export_merged_dense(PorosConfig(model_name_or_path=str(base)),
                            str(adapter), str(out), nf4_device="cpu")
    assert not out.exists()


def test_the_train_preflight_says_a_tied_tables_lora_will_not_merge(tmp_path):
    import json

    from poros.preflight.start import tied_tables_note

    folder = tmp_path / "m"
    folder.mkdir()
    for tied, said in ((True, True), (False, False)):
        (folder / "config.json").write_text(json.dumps({"model_type": "qwen2",
                                                        "tie_word_embeddings": tied}))
        cfg = PorosConfig(model_name_or_path=str(folder),
                          lora_target_modules=["embed_tokens", "q_proj"])
        assert (tied_tables_note(cfg) is not None) is said
    assert tied_tables_note(PorosConfig(model_name_or_path=str(folder))) is None
