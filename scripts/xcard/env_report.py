"""env_report.py OUT.json -- the card, the driver and the stack, against pyproject's pins."""
import importlib.metadata as md
import json
import platform
import subprocess
import sys

import torch

out = {"python": sys.version.split()[0], "platform": platform.platform()}
for pkg in ("torch", "triton", "bitsandbytes", "transformers", "peft", "accelerate", "datasets",
            "safetensors"):
    try:
        out[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        out[pkg] = None
out["cuda_runtime"] = torch.version.cuda
out["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    out.update(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)),
               vram_gib=round(p.total_memory / 2**30, 2), arch_list=torch.cuda.get_arch_list(),
               shared_per_block_optin=getattr(p, "shared_memory_per_block_optin", None),
               bf16_native=torch.cuda.is_bf16_supported(including_emulation=False)
               if "including_emulation" in torch.cuda.is_bf16_supported.__code__.co_varnames
               else torch.cuda.is_bf16_supported())
    try:
        out["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                        "--format=csv,noheader"], capture_output=True,
                                       text=True, timeout=10).stdout.strip()
    except OSError:
        out["driver"] = None
    from poros.core import card

    cap = tuple(out["capability"])
    out["poros"] = {"class": card.class_name(cap), "build_reason": card.build_reason(
        cap, out["arch_list"]), "bf16_reason": card.bf16_reason(cap),
        "triton_reason": card.triton_reason(cap), "shared_limit": card.shared_memory_limit(cap)}
    try:
        with open("/proc/version") as f:
            out["wsl"] = "microsoft" in f.read().lower()
    except OSError:
        out["wsl"] = None
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out))
