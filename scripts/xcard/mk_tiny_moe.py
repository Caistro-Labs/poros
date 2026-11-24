"""mk_tiny_moe.py OUT_DIR TOKENIZER_MODEL -- a small random Qwen3-MoE checkpoint (bf16
safetensors) with TOKENIZER_MODEL's tokenizer beside it (``poros train`` loads one), its vocabulary
the tokenizer's. Only the tokenizer is downloaded (already cached by the dense runs)."""
import sys

import torch
import transformers

torch.manual_seed(0)
tok = transformers.AutoTokenizer.from_pretrained(sys.argv[2])
cfg = transformers.Qwen3MoeConfig(vocab_size=len(tok), hidden_size=512, intermediate_size=1024,
                                  moe_intermediate_size=256, num_hidden_layers=6,
                                  num_attention_heads=8, num_key_value_heads=4, head_dim=64,
                                  num_experts=16, num_experts_per_tok=4)
model = transformers.AutoModelForCausalLM.from_config(cfg)
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0, 0.02)
model.to(torch.bfloat16).save_pretrained(sys.argv[1])
tok.save_pretrained(sys.argv[1])
print("tiny qwen3_moe at", sys.argv[1])
