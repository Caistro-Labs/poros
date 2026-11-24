"""Opt-in grouped expert compute for ``ManagedMoeBlock`` (``MoeConfig.expert_compute="grouped"``).

The reference compute (``poros.moe.modules._experts_compute``) buckets experts by
power-of-two token count, zero-pads each bucket to its busiest expert and runs
``bmm``; its backward rematerializes the whole layer with grad, so every decoded
weight of the layer stays alive in the autograd graph until the layer's backward
ends (18-24 GiB at a 288-expert, 4096-wide layer on an RTX 3090).

This path:

* sorts the ``tokens x top_k`` rows by expert once (stable: token order within an
  expert) and runs each fixed chunk of ``expert_batch`` consecutive hit experts
  with ``torch.nn.functional.grouped_mm`` over the exact rows -- no padding, no
  layout copies; LoRA through the same grouped GEMM; clamped SwiGLU between;
* combines with an index scatter into each row's own ``(token, slot)`` place and
  a fixed-order sum over ``top_k`` (no ``index_add``, no atomics);
* runs backward chunk by chunk: decode a chunk, recompute it with grad, backprop
  it, free it -- one chunk's decoded weights alive at a time.

Determinism: every op is deterministic for fixed shapes on a fixed torch / GPU
(``grouped_mm`` is a CUTLASS grouped GEMM with no split-K reduction; the token
gradient's ``index_put(accumulate=True)`` runs torch's sorted deterministic
kernel). It is not bitwise the reference (other GEMM kernels and grouping), so it
is a second self-consistent backend: activation refuses unless torch and the GPU
match a recorded pin (``grouped_pins/``, ``scripts/moe_grouped_pin.py``), and a
cross-process proof backs each pin. On CPU (tests) the grouped GEMM is a per-expert
``mm`` loop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PIN_DIR = Path(__file__).parent / "grouped_pins"
_ACTIVE: dict[str, Any] | None = None


class GroupedPinError(RuntimeError):
    """The grouped expert backend is not pinned for this host."""


def environment() -> dict[str, Any]:
    import triton

    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def default_pin_path(env: dict[str, Any] | None = None) -> Path:
    env = env or environment()
    gpu = re.sub(r"[^0-9A-Za-z]+", "_", env["gpu"]).strip("_")
    return PIN_DIR / f"torch-{env['torch']}_{gpu}.json"


def activate(pin_file: str | Path | None = None) -> dict[str, Any]:
    """Allow the grouped backend on this CUDA host (idempotent); ``GroupedPinError`` otherwise."""
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    if not torch.cuda.is_available():
        raise GroupedPinError("expert_compute='grouped' is pinned per CUDA host; no CUDA here")
    if not hasattr(F, "grouped_mm"):
        raise GroupedPinError(f"torch {torch.__version__} has no torch.nn.functional.grouped_mm")
    env = environment()
    path = Path(pin_file) if pin_file is not None else default_pin_path(env)
    if not path.exists():
        raise GroupedPinError(f"no grouped-expert pin for this host: {path} (env {env})")
    pin = json.loads(path.read_text())
    if pin["env"] != env:
        raise GroupedPinError(
            f"grouped-expert pin {path.name} was recorded on {pin['env']}, host is {env}"
        )
    _ACTIVE = {"pin_file": str(path), "env": env}
    return _ACTIVE


def _grouped_mm(
    x: torch.Tensor, w_t: torch.Tensor, offs: torch.Tensor, ends: list[int]
) -> torch.Tensor:
    """``out[rows of e] = x[rows of e] @ w_t[e]`` for consecutive row groups ending at ``ends``."""
    if x.is_cuda:
        return F.grouped_mm(x, w_t, offs=offs)
    outs, start = [], 0
    for e, end in enumerate(ends):
        outs.append(x[start:end] @ w_t[e])
        start = end
    return torch.cat(outs)


def _chunk_forward(x, chunk, ends, offs, block, lora):
    """One chunk of consecutive experts over their sorted rows ``x`` -> unweighted down rows."""
    from poros.moe.store import decode_experts

    keys = [(block.layer_idx, e) for e in chunk]
    w = decode_experts(block, keys, "gate_up_proj")
    gate_up = _grouped_mm(x, w.mT, offs, ends)
    del w
    if lora:
        a, b = lora[0], lora[1]
        upd = _grouped_mm(
            _grouped_mm(block.adapter.dropout(x).to(a.dtype), a.mT, offs, ends), b.mT, offs, ends
        )
        gate_up = gate_up + upd.to(gate_up.dtype) * block.adapter.scaling
    gate, up = gate_up.chunk(2, dim=-1)
    if block.swiglu_limit is not None:
        gate = gate.clamp(min=None, max=block.swiglu_limit)
        up = up.clamp(min=-block.swiglu_limit, max=block.swiglu_limit)
    hidden = block.act_fn(gate) * up
    w = decode_experts(block, keys, "down_proj")
    down = _grouped_mm(hidden, w.mT, offs, ends)
    del w
    if lora:
        a, b = lora[2], lora[3]
        upd = _grouped_mm(
            _grouped_mm(block.adapter.dropout(hidden).to(a.dtype), a.mT, offs, ends),
            b.mT,
            offs,
            ends,
        )
        down = down + upd.to(down.dtype) * block.adapter.scaling
    return down


def _plan(top_k_index: torch.Tensor, block) -> dict[str, Any]:
    """Expert-sorted row order and the fixed chunks of consecutive hit experts."""
    flat = top_k_index.reshape(-1)  # row r: token r // top_k, slot r % top_k
    counts = torch.bincount(flat, minlength=block.num_experts).tolist()  # one host sync
    hit = [e for e, n in enumerate(counts) if n]
    batch = block.expert_batch or len(hit) or 1
    chunks, start = [], 0
    for i in range(0, len(hit), batch):
        chunk = hit[i : i + batch]
        ends, total = [], 0
        for e in chunk:
            total += counts[e]
            ends.append(total)
        chunks.append((chunk, start, start + total, ends))
        start += total
    order = flat.argsort(stable=True)  # rows grouped by expert id, token order within
    return {"flat": flat, "order": order, "hit": hit, "chunks": chunks}


def _chunk_lora(block, chunk, device):
    if block.adapter is None:
        return ()
    ids = torch.tensor(chunk, device=device)
    return block.adapter.factors(ids)


def _offs(ends, device):
    return torch.tensor(ends, dtype=torch.int32, device=device)


def _forward(hidden, top_k_index, top_k_weights, block, plan):
    num_tokens, top_k = top_k_index.shape
    from poros.moe.modules import _prefetch_hint

    _prefetch_hint(block, [(block.layer_idx, e) for e in plan["hit"]])
    block.residency.ensure([(block.layer_idx, e) for e in plan["hit"]])
    order = plan["order"]
    rows = hidden[order // top_k]
    outs = []
    for chunk, lo, hi, ends in plan["chunks"]:
        lora = _chunk_lora(block, chunk, hidden.device)
        outs.append(
            _chunk_forward(rows[lo:hi], chunk, ends, _offs(ends, hidden.device), block, lora)
        )
    out = torch.cat(outs) if outs else hidden.new_zeros(0, hidden.shape[-1])
    weighted = out * top_k_weights.reshape(-1)[order, None]
    unsorted = torch.empty_like(weighted)
    unsorted[order] = weighted
    return unsorted.view(num_tokens, top_k, -1).sum(1).to(hidden.dtype)


class GroupedExpertFunction(torch.autograd.Function):
    """Autograd boundary of the grouped path: saves inputs + RNG, backward chunk by chunk."""

    @staticmethod
    def forward(ctx, hidden, top_k_index, top_k_weights, block):
        ctx.block = block
        ctx.cpu_rng_state = torch.get_rng_state()
        ctx.cuda_rng_state = torch.cuda.get_rng_state(hidden.device) if hidden.is_cuda else None
        ctx.save_for_backward(hidden, top_k_index, top_k_weights)
        precomputed = block._remat_hit
        if precomputed is not None:
            block._remat_hit = None
            return precomputed.clone()
        return _forward(hidden, top_k_index, top_k_weights, block, _plan(top_k_index, block))

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        hidden, top_k_index, top_k_weights = ctx.saved_tensors
        block = ctx.block
        num_tokens, top_k = top_k_index.shape
        plan = _plan(top_k_index, block)
        from poros.moe.modules import _prefetch_hint

        _prefetch_hint(block, [(block.layer_idx, e) for e in plan["hit"]])
        block.residency.ensure([(block.layer_idx, e) for e in plan["hit"]])
        order = plan["order"]
        w_sorted = top_k_weights.reshape(-1)[order]
        # combine backward: out = sum_k unsorted[t, k] (promoted dtype), cast to hidden's
        acc_dtype = torch.promote_types(hidden.dtype, top_k_weights.dtype)
        g_rows_out = grad_out.to(acc_dtype)[order // top_k]  # d unsorted, in sorted row order
        rows = hidden[order // top_k].detach()
        g_rows = torch.empty_like(rows)
        g_w_sorted = torch.empty_like(w_sorted)
        adapter = block.adapter
        params = list(adapter.parameters()) if adapter is not None else []
        fork = (
            [
                hidden.device.index
                if hidden.device.index is not None
                else torch.cuda.current_device()
            ]
            if hidden.is_cuda
            else []
        )
        with torch.random.fork_rng(devices=fork):
            torch.set_rng_state(ctx.cpu_rng_state)
            if ctx.cuda_rng_state is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng_state, device=hidden.device)
            for chunk, lo, hi, ends in plan["chunks"]:
                with torch.enable_grad():
                    x = rows[lo:hi].detach().requires_grad_(True)
                    lora = _chunk_lora(block, chunk, hidden.device)
                    down = _chunk_forward(x, chunk, ends, _offs(ends, hidden.device), block, lora)
                g_chunk = g_rows_out[lo:hi]
                w_chunk = w_sorted[lo:hi, None]
                # weighted = down * w  ->  d down = g * w (in down's dtype), d w = sum(g * down)
                g_down = (g_chunk * w_chunk).to(down.dtype)
                g_w_sorted[lo:hi] = (
                    (g_chunk * down.detach().to(acc_dtype)).sum(-1).to(w_sorted.dtype)
                )
                grads = torch.autograd.grad(down, [x, *params], g_down, allow_unused=True)
                g_rows[lo:hi] = grads[0]
                for param, grad in zip(params, grads[1:], strict=True):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad.detach()
                    else:
                        param.grad.add_(grad.detach())
                del down, grads, x
        g_hidden = torch.zeros_like(hidden).index_put_((order // top_k,), g_rows, accumulate=True)
        g_weights = torch.empty_like(top_k_weights).reshape(-1)
        g_weights[order] = g_w_sorted
        return g_hidden, None, g_weights.view_as(top_k_weights), None
