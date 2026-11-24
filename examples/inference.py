"""Use a Poros-trained adapter for inference.

After `trainer.save("./poros_output/qwen7b-lora")` (or `poros train ...`), the saved
directory contains the LoRA adapter, the tokenizer, and poros_config.json.
There are two ways to run it; pick one for where it will run.

Run (needs a GPU):

    python examples/inference.py
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-7B"
ADAPTER_DIR = "./poros_output/qwen7b-lora"  # what trainer.save() wrote
PROMPT = "Explain blockwise residency in one sentence."

# Option A: load the adapter on the base model (no merge).
# The base is loaded in bf16 here; PEFT
# applies the LoRA deltas at runtime.
base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# The tokenizer was saved alongside the adapter, so load it from there.
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# Option B: merge the adapter into the base once, and serve the merged model.
# For vLLM / TGI / a plain transformers load, merge the adapter into the base
# and save a standalone model directory (this is what `poros export` does):
#
#     poros export ./poros_output/qwen7b-lora -o ./merged
#
# then load ./merged like any HuggingFace model. The merged directory already
# contains the tokenizer, so it is directly servable.
