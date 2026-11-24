"""Poros train: trainer, model loading, and adapters.

This subpackage provides:
    PorosTrainer   -- The main training orchestrator (from_pretrained, train, save)
    PorosModel     -- Model loading with quantization + LoRA adapter attachment
    NativeLoraLinear, NativeAdapterManager -- Native LoRA fallback (no PEFT)
    get_trainable_state_dict -- Extract adapter-only state dict
"""
