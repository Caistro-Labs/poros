"""rank_probe.py OUT_DIR -- the fused compute at LoRA ranks 8..256 on this card, through
PorosTrainer (2 updates, a tiny random NF4 Qwen2, fused_compute left at its default), with any
tree on PYTHONPATH: run it against a tree before the shared-memory fit (to see a rank fail) and
after (to see it run, fused or on PEFT's modules with the reason recorded). One line per rank:
RANK <r> OK fused=<bool> reason=<...> | RANK <r> FAIL <exception>."""
import sys
import traceback

import torch
import transformers

from poros.train.trainer import PorosTrainer

out = sys.argv[1]
torch.manual_seed(0)
cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
m = transformers.AutoModelForCausalLM.from_config(cfg)
with torch.no_grad():
    for p in m.parameters():
        p.normal_(0, 0.05)
m.to(torch.bfloat16).save_pretrained(f"{out}/tiny_qwen2")
rows = []
g = torch.Generator().manual_seed(5)
for _ in range(8):
    ids = torch.randint(3, 250, (64,), generator=g)
    rows.append({"input_ids": ids, "attention_mask": torch.ones(64, dtype=torch.long),
                 "labels": ids.clone()})
for rank in (8, 16, 32, 64, 128, 256):
    try:
        tr = PorosTrainer.from_pretrained(
            f"{out}/tiny_qwen2", quantization="nf4", lora_rank=rank, lora_alpha=2 * rank,
            lora_dropout=0.05, output_dir=f"{out}/rank{rank}", max_steps=2, save_steps=0,
            progress=False, gradient_accumulation_steps=2, per_device_batch_size=2,
            max_seq_length=64, blockwise_arch_check="off",
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                 "up_proj", "down_proj"])
        tr.train(torch.utils.data.DataLoader(rows, batch_size=2, shuffle=False), steps=2,
                 seq_len=64)
        ran = tr.config.ran_compute or {}
        print(f"RANK {rank} OK fused={ran.get('fused_compute')} reason={tr.config.compute_reason}"
              f" loss={tr.history.losses}", flush=True)
    except Exception as exc:  # noqa: BLE001 - the probe reports whatever happens
        print(f"RANK {rank} FAIL {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        traceback.print_exc(file=sys.stderr)
    torch.cuda.empty_cache()
