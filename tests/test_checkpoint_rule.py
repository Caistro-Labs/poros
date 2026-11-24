"""The update rule a checkpoint records (``optimizer.json``, ``poros.core.checkpoints``), and what a
resume reads from it: every checkpoint 1.0.0 writes carries it; a checkpoint from before it is
read by its LoRA tensors alone, never by a frozen base table an adapter saved beside them.

The case that found it: a dense bf16 run whose adapter targets lm_head saves PEFT's copy of the
frozen head (``lm_head.base_layer.weight``) in bf16 beside fp32 LoRA tensors, and its own
checkpoint was refused as "written before 1.0.0".
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tests.test_train_loop_cpu import _loader, _make_trainer, _require_ml


def _bf16_qwen2_with_lm_head(torch):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                             target_modules=["q_proj", "lm_head"], bias="none",
                                             task_type="CAUSAL_LM"))
    return model, cfg


def test_a_dense_run_targeting_lm_head_resumes_its_own_checkpoint_bitwise():
    torch = _require_ml()
    from poros.core.checkpoints import RULE_FILE, TORCH_RULE, checkpoint_record
    from poros.core.errors import PorosCancelledError

    def run(d, **kw):
        torch.manual_seed(0)
        model, cfg = _bf16_qwen2_with_lm_head(torch)
        tr = _make_trainer(torch, model, d, max_steps=4)
        return tr, tr.train(_loader(torch, cfg.vocab_size), steps=4, **kw)

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        whole, full = run(a)
        checks = []
        with pytest.raises(PorosCancelledError):
            run(b, should_stop=lambda: checks.append(1) or len(checks) > 1)
        ckpt = Path(b) / "checkpoint-1"
        names = json.loads((ckpt / "adapter_model.safetensors").read_bytes()[
            8:8 + int.from_bytes((ckpt / "adapter_model.safetensors").read_bytes()[:8],
                                 "little")])
        assert any("lm_head.base_layer" in k and v["dtype"] == "BF16" for k, v in names.items())
        assert checkpoint_record(ckpt) == {"rule": TORCH_RULE, "trainable_dtype": "float32",
                                           "torch_dtype": "bfloat16", "experts": False}
        (ckpt / RULE_FILE).unlink()  # as a build before the rule file wrote it
        assert checkpoint_record(ckpt)["rule"] == TORCH_RULE  # the base table is not read
        resumed, rest = run(b, resume_from=str(ckpt))
    assert rest["loss_history"] == full["loss_history"][2:]
    for x, y in zip(resumed.trainable_parameters, whole.trainable_parameters, strict=True):
        assert torch.equal(x, y)


def _save_adapter(folder: Path, tensors: dict) -> None:
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "adapter_model.safetensors")


def test_without_a_rule_file_bf16_lora_tensors_are_refused_and_fp32_ones_are_not(tmp_path):
    torch = _require_ml()
    from poros.core.checkpoints import OLD_RULE, TORCH_RULE, checkpoint_record

    old = tmp_path / "old"
    _save_adapter(old, {"m.q_proj.lora_A.weight": torch.zeros(2, 4, dtype=torch.bfloat16),
                        "m.q_proj.lora_B.weight": torch.zeros(4, 2, dtype=torch.bfloat16)})
    assert checkpoint_record(old)["rule"] == OLD_RULE
    table = tmp_path / "table"
    _save_adapter(table, {"m.q_proj.lora_A.weight": torch.zeros(2, 4),
                          "m.q_proj.lora_B.weight": torch.zeros(4, 2),
                          "m.lm_head.base_layer.weight": torch.zeros(8, 4, dtype=torch.bfloat16)})
    assert checkpoint_record(table) == {"rule": TORCH_RULE, "trainable_dtype": "float32",
                                        "torch_dtype": None, "experts": False}
    native = tmp_path / "native"
    native.mkdir()
    torch.save({"l.adapter.lora_A_gate_up": torch.zeros(2, 2, dtype=torch.bfloat16),
                "embed.weight": torch.zeros(2)}, native / "adapter_weights.pt")
    assert checkpoint_record(native)["rule"] == OLD_RULE
    torch.save({"l.adapter.lora_A_gate_up": torch.zeros(2, 2),
                "embed.weight": torch.zeros(2, dtype=torch.bfloat16)},
               native / "adapter_weights.pt")
    assert checkpoint_record(native)["rule"] == TORCH_RULE
    assert checkpoint_record(tmp_path / "missing") is None


def test_the_resume_check_refuses_an_old_bf16_checkpoint_in_words(tmp_path):
    torch = _require_ml()
    from poros.core.config import PorosConfig
    from poros.preflight.resume import resume_verdict

    ckpt = tmp_path / "ck"
    _save_adapter(ckpt, {"m.q_proj.lora_A.weight": torch.zeros(2, 4, dtype=torch.bfloat16)})
    (ckpt / "poros_config.json").write_text(json.dumps({"model_name_or_path": "x"}))
    clash, _ = resume_verdict(str(ckpt), PorosConfig(model_name_or_path="x"))
    assert clash and "before 1.0.0" in clash and "--from" in clash.fix


@pytest.mark.parametrize("rule", ["poros.expert-adamw.v1", "poros.expert-adamw.v2",
                                  "poros.expert-adamw.v3"])
def test_a_release_candidate_s_rule_is_refused_naming_both(tmp_path, rule):
    _require_ml()
    from poros.core.checkpoints import RULE
    from poros.core.config import PorosConfig
    from poros.preflight.resume import resume_verdict

    ckpt = tmp_path / "ck"
    ckpt.mkdir()
    (ckpt / "poros_config.json").write_text(json.dumps({"model_name_or_path": "x"}))
    (ckpt / "optimizer.json").write_text(json.dumps({"rule": rule}))
    clash, _ = resume_verdict(str(ckpt), PorosConfig(model_name_or_path="x"))
    assert clash and rule in clash and RULE in clash and "--from" in clash.fix


def test_an_oversized_pickle_is_refused_in_words_not_read(tmp_path, monkeypatch):
    torch = _require_ml()
    import poros.core.checkpoints as ck
    from poros.core.config import PorosConfig
    from poros.preflight.resume import resume_verdict

    ckpt = tmp_path / "ck"
    ckpt.mkdir()
    torch.save({"a.lora_A": torch.zeros(4, dtype=torch.bfloat16)}, ckpt / "adapter_weights.pt")
    (ckpt / "poros_config.json").write_text(json.dumps({"model_name_or_path": "x"}))
    monkeypatch.setattr(ck, "PICKLE_LIMIT", 16)
    with pytest.raises(ck.CheckpointUnreadable, match="more than the 16"):
        ck.checkpoint_record(ckpt)
    clash, _ = resume_verdict(str(ckpt), PorosConfig(model_name_or_path="x"))
    assert clash and "data.pkl" in clash
