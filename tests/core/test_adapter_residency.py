"""Adapter residency test (CPU only).

Uses a tiny mock transformer with LoRA-injected linears.
After patch_hf_model + blockwise setup, every ``lora_`` param must:
  - have ``requires_grad == True``
  - have ``numel() > 0``
  - live on ``trainer.device``
A forward/backward then asserts every LoRA param gets a non-None grad.
"""
from __future__ import annotations

import pytest


def _require_ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _tiny_model(torch):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg)
    model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                   target_modules=["q_proj", "v_proj"]),
    )
    return model, cfg


def _prepare_cpu(model):
    from poros.patch.hf import patch_hf_model

    return patch_hf_model(model, block_size=2, deterministic=False)


@pytest.mark.usefixtures("cpu_only")
class TestAdapterResidency:
    """Adapter params are resident, trainable, and grad-producing."""

    def test_lora_params_resident_after_setup(self):
        torch = _require_ml()
        torch.manual_seed(0)
        model, cfg = _tiny_model(torch)
        model = _prepare_cpu(model)

        device = torch.device("cpu")
        lora_count = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                lora_count += 1
                assert p.requires_grad, f"{name} lost requires_grad"
                assert p.data.numel() > 0, f"{name} has empty data"
                assert p.device == device, (
                    f"{name} on {p.device}, expected {device}"
                )
        assert lora_count > 0, "No lora_ params found"

    def test_lora_params_get_grad_after_fwd_bwd(self):
        torch = _require_ml()
        torch.manual_seed(0)
        model, cfg = _tiny_model(torch)
        model = _prepare_cpu(model)

        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()

        lora_grads = 0
        for name, p in model.named_parameters():
            if "lora_" in name and p.requires_grad:
                assert p.grad is not None, f"{name} has no grad after backward"
                lora_grads += 1
        assert lora_grads > 0, "No LoRA grads found"

    def test_patched_forward_and_backward_give_adapter_grads(self):
        """The patched blockwise forward and its backward run and reach the adapters."""
        torch = _require_ml()
        torch.manual_seed(42)
        model, cfg = _tiny_model(torch)
        model = _prepare_cpu(model)

        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        out = model(input_ids=ids, labels=ids)
        assert out.loss.item() > 0
        out.loss.backward()

        grad_count = sum(
            1 for _, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        )
        assert grad_count > 0
