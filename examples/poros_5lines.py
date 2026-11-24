"""The 5-line PorosTrainer API.

The shortest way to fine-tune with Poros: NF4 QLoRA with the default settings. ``block_size``
defaults to 4 whatever the GPU; a lower one holds less GPU memory and takes longer a step.

Run (needs a GPU: ``pip install poros-train``):

    python examples/poros_5lines.py

Swap the model / dataset / steps for your own. Use a smaller model (e.g.
"Qwen/Qwen2.5-0.5B") for a quick smoke test on a small card.
"""

from poros import PorosTrainer

trainer = PorosTrainer.from_pretrained(
    "Qwen/Qwen2.5-7B",       # any supported family (poros check <model>)
    quantization="nf4",       # the 4-bit base Poros streams
    lora_rank=16,
    lora_alpha=32,
    block_size=4,             # layers per streamed block: larger is faster, holds more GPU memory
)
trainer.train("OpenAssistant/oasst1", steps=200, seq_len=512)  # any HF dataset, or "synthetic"
trainer.save("./poros_output/qwen7b-lora")   # examples/inference.py loads from here
print("Done. Adapter saved to ./poros_output/qwen7b-lora")

# ── Smoke run (small card / no Hub access) ───────────────────────────────────
# Swap the two calls above for a tiny model + the built-in synthetic dataset, so
# the whole train -> save -> serve chain runs verbatim on any GPU:
#   trainer = PorosTrainer.from_pretrained("Qwen/Qwen2.5-0.5B", quantization="nf4")
#   trainer.train("synthetic", steps=20, seq_len=128)  # then trainer.save(...) as above
