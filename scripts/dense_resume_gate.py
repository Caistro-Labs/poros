"""Dense resume gate through the CLI: train, resume from a checkpoint, compare bitwise.

Run A trains 4 steps with a checkpoint at step 2; run B resumes from it. At
the PorosConfig defaults (NF4, block_size 4, lora_dropout 0.05) the final
adapters and optimizer state of both runs must be ``torch.equal``.

    python scripts/dense_resume_gate.py Qwen/Qwen2.5-7B ./resume_gate
"""

from __future__ import annotations

import json
import subprocess
import sys

import torch
from safetensors.torch import load_file

CLI = [sys.executable, "-c", "from poros.cli.main import cli; cli()"]


def _cli(*args: str) -> None:
    proc = subprocess.run([*CLI, *args], capture_output=True, text=True)
    print(proc.stdout[-2500:], proc.stderr[-2500:], flush=True)
    if proc.returncode:
        raise SystemExit(f"poros {' '.join(args)} exited {proc.returncode}")


def _tensors(obj, prefix=""):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _tensors(value, f"{prefix}/{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from _tensors(value, f"{prefix}/{i}")


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def main(model: str, root: str) -> None:
    config = dict(
        model_name_or_path=model, dataset="synthetic", max_steps=4, save_steps=2,
        max_seq_length=256, gradient_accumulation_steps=2, logging_steps=1, eval_steps=0,
        progress=False,
    )
    for run in ("a", "b"):
        with open(f"{root}/{run}.yaml", "w") as f:
            json.dump({**config, "output_dir": f"{root}/run_{run}"}, f)
    _cli("train", f"{root}/a.yaml")
    _cli("train", f"{root}/b.yaml", "--resume-from", f"{root}/run_a/checkpoint-2")

    def state(run: str) -> tuple[dict, dict]:
        adapter = load_file(f"{root}/run_{run}/adapter_model.safetensors")
        optim = torch.load(f"{root}/run_{run}/optimizer_state.pt", map_location="cpu")
        return adapter, dict(_tensors(optim))

    (adapter_a, optim_a), (adapter_b, optim_b) = state("a"), state("b")
    result = {
        "model": model,
        "adapter_tensors": len(adapter_a),
        "adapters_bitwise_equal": _equal(adapter_a, adapter_b),
        "optimizer_state_bitwise_equal": _equal(optim_a, optim_b),
    }
    result["passed"] = result["adapters_bitwise_equal"] and result["optimizer_state_bitwise_equal"]
    print("DENSE_RESUME_RESULT", json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
