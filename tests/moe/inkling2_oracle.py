"""The official Inkling 2 Mini implementation (transformers >= 5.24) as a test oracle, in its own
process: Poros pins an older transformers, so tests/moe/test_inkling2_oracle.py runs this file with
``POROS_INKLING_ORACLE`` (a directory holding transformers 5.24, e.g. from
``pip install --target DIR --no-deps transformers==5.24.0``) first on ``PYTHONPATH``.

    python inkling2_oracle.py <job.pt> <out.pt>

The job holds a text config (dict), a state dict, the input ids and labels (and optionally an
attention mask, and ``dtype: bfloat16`` to run the official model in bf16 with its own fp32
modules kept, as it holds a bf16 model), and optionally a
LoRA spec: ``{"rank", "alpha", "targets": [module names], "A": {name: A}, "B": {name: B},
"experts": {layer: {"A_gate_up", "B_gate_up", "A_down", "B_down"}}}``. With LoRA, every base
parameter is frozen and the LoRA factors are the trainable ones, with the math of
poros.train.adapters.NativeLoraLinear and poros.moe.adapters.ExpertLoraAdapter written out here
from their definitions (``base + (B A x) * alpha / rank``; dropout 0). The output holds the loss,
the logits and the gradient of every trainable tensor, all fp32.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import nn


class _Lora(nn.Module):
    def __init__(self, base: nn.Linear, a: torch.Tensor, b: torch.Tensor, scaling: float) -> None:
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(a.clone())
        self.lora_B = nn.Parameter(b.clone())
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + update.to(out.dtype) * self.scaling


def _expert_lora(experts: nn.Module, spec: dict, scaling: float) -> None:
    """Replace the experts' forward with the official one plus a per-expert LoRA on each fused
    projection (the delta added at the projection output, before the activation / weighting)."""
    params = nn.ParameterDict({k: nn.Parameter(v.clone()) for k, v in spec.items()})
    experts.poros_lora = params

    def forward(hidden, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden, dtype=torch.float32)
        mask = F.one_hot(top_k_index, num_classes=experts.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            if e == experts.num_experts:
                continue
            slot, token = torch.where(mask[e])
            x = hidden[token]
            gate_up = F.linear(x, experts.gate_up_proj[e])
            delta = F.linear(F.linear(x, params["A_gate_up"][e]), params["B_gate_up"][e])
            gate_up = gate_up + delta.to(gate_up.dtype) * scaling
            h = experts._apply_gate(gate_up)
            y = F.linear(h, experts.down_proj[e])
            delta = F.linear(F.linear(h, params["A_down"][e]), params["B_down"][e])
            y = y + delta.to(y.dtype) * scaling
            y = y * top_k_weights[token, slot, None]
            out.index_add_(0, token, y.to(out.dtype))
        return out.to(hidden.dtype)

    experts.forward = forward


def main(job_path: str, out_path: str) -> None:
    import transformers
    from transformers.models.inkling2_loop import (
        Inkling2LoopForCausalLM,
        Inkling2LoopTextConfig,
    )

    job = torch.load(job_path, weights_only=False)
    if job.get("checkpoint"):  # a released-format checkpoint, through transformers' own loading
        model = Inkling2LoopForCausalLM.from_pretrained(
            job["checkpoint"], dtype=torch.float32, attn_implementation="eager"
        )
    else:
        config = Inkling2LoopTextConfig(**job["config"])
        config._attn_implementation = "eager"
        with torch.device("cpu"):
            model = Inkling2LoopForCausalLM(config).float()
        missing, unexpected = model.load_state_dict(job["state"], strict=False)
        if missing or unexpected:
            raise SystemExit(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    if job.get("dtype") == "bfloat16":  # as the official class holds a bf16 model: its fp32
        keep = tuple(model._keep_in_fp32_modules_strict or ())  # modules kept in fp32
        model.to(torch.bfloat16)
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            if any(k in name for k in keep):
                tensor.data = tensor.data.float()
    # the loaded tensors themselves, not copies (nothing below changes a weight; a copy of the
    # tiny checkpoint's 200k-row embedding and head would take the child past the tests' cap)
    loaded = dict(model.state_dict()) if job.get("return_state") else None
    lora = job.get("lora")
    if lora is None:
        trainable = dict(model.named_parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        scaling = lora["alpha"] / lora["rank"]
        for name in lora["targets"]:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            setattr(
                parent,
                child,
                _Lora(getattr(parent, child), lora["A"][name], lora["B"][name], scaling),
            )
        for layer, spec in lora["experts"].items():
            _expert_lora(model.model.layers[int(layer)].mlp.experts, spec, scaling)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    out = model(
        input_ids=job["input_ids"], labels=job["labels"], attention_mask=job.get("attention_mask")
    )
    out.loss.backward()
    torch.save(
        {
            "transformers": transformers.__version__,
            "loss": out.loss.detach(),
            "logits": out.logits.detach().float(),
            "grads": {
                n: p.grad.detach().float() for n, p in trainable.items() if p.grad is not None
            },
            "state": loaded,
        },
        out_path,
    )
    _drop_from_cache(out_path)


def _drop_from_cache(path: str) -> None:
    """Write the file through and drop it from the page cache: the tests run under a memory cap
    that counts a process's cached file pages, and the tiny checkpoint's output is about 2 GB."""
    import os

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
