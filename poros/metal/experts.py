"""MoE on the Mac: every expert is two more frozen linears in its layer's slab.

A family's own sparse block keeps running, its router and shared experts with it. Only its fused
``experts`` give way to ``Experts``, called as they are, with one ``nn.Linear`` per expert per
projection. The pack, the decode, the block loop, the remat and the witness already know those, so
nothing here streams or restores anything. Experts are NF4 and all else stays bf16, as on CUDA.

The math is the family's own expert loop with ``poros.moe``'s LoRA at each projection, one expert at
a time and in its order, so a token's shares add up as the model adds them. The one departure is
``_Take`` and ``_Add``: autograd's indexing adds at an index, which has no deterministic kernel on
MPS. Each id appears once in a call, so a copy gives the same bytes, and the strict flag stays on to
catch any site missed.

An adapter keeps ``poros.moe``'s names and fused shape in memory. On disk it is PEFT's own layout
for a fused parameter, so vanilla PEFT loads it on the vanilla model.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from poros.moe.adapters import ExpertLoraAdapter

# The factor -> its name in PEFT's file: ``experts.gate_up_proj`` is the inner of its wrappers.
_PEFT_NAMES = {
    "lora_A_gate_up": "experts.base_layer.lora_A.weight",
    "lora_B_gate_up": "experts.base_layer.lora_B.weight",
    "lora_A_down": "experts.lora_A.weight",
    "lora_B_down": "experts.lora_B.weight",
}
_PEFT_TARGETS = ["experts.gate_up_proj", "experts.down_proj"]
_PEFT_PREFIX = "base_model.model."
_ROLE = re.compile(r"^(.+\.experts)\.(gate|up|down)\.(\d+)\.weight$")
_WHOLE = re.compile(r"^.+\.experts\.(gate_up_proj|down_proj)$")


class _Take(torch.autograd.Function):
    """``x[ids]``, its VJP a copy into zeros where autograd's would add."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        ctx.shape, ctx.ids = x.shape, ids
        return x.index_select(0, ids)

    @staticmethod
    @once_differentiable
    def backward(ctx, g: torch.Tensor):
        return g.new_zeros(ctx.shape).index_copy_(0, ctx.ids, g), None


class _Add(torch.autograd.Function):
    """``total[ids] += rows``, written with a copy. Its VJP is ``index_add_``'s."""

    @staticmethod
    def forward(ctx, total: torch.Tensor, ids: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        ctx.ids = ids
        return total.index_copy(0, ids, total.index_select(0, ids) + rows)

    @staticmethod
    @once_differentiable
    def backward(ctx, g: torch.Tensor):
        return g, None, g.index_select(0, ctx.ids)


class Experts(nn.Module):
    """Takes the place of a family's fused experts, and is called as they are."""

    def __init__(self, fused: nn.Module):
        super().__init__()
        hidden, mid, count = fused.hidden_dim, fused.intermediate_dim, fused.num_experts
        self.gate_up_proj = nn.ModuleList(
            nn.Linear(hidden, 2 * mid, bias=False) for _ in range(count)
        )
        self.down_proj = nn.ModuleList(nn.Linear(mid, hidden, bias=False) for _ in range(count))
        self.act_fn = fused.act_fn

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        total = torch.zeros_like(hidden_states)
        device, top_k = hidden_states.device, top_k_index.shape[1]
        picks = top_k_index.T.cpu()  # [top_k, tokens]: one sync per layer per pass
        weights = top_k_weights.reshape(-1)
        lora = self.adapter
        fused = lora.lora_A_gate_up, lora.lora_B_gate_up, lora.lora_A_down, lora.lora_B_down
        # One backward per factor, a stack, where indexing per expert would zero the whole factor.
        a_gate_up, b_gate_up, a_down, b_down = (p.unbind(0) for p in fused)
        for e in picks.unique().tolist():
            slots, tokens = torch.where(picks == e)  # by slot, then by token, as the model has them
            ids = tokens.to(device)
            x = _Take.apply(hidden_states, ids)
            gate_up = self.gate_up_proj[e](x) + lora.delta(x, a_gate_up[e], b_gate_up[e])
            gate, up = gate_up.chunk(2, dim=-1)
            y = self.act_fn(gate) * up
            y = self.down_proj[e](y) + lora.delta(y, a_down[e], b_down[e])
            y = y * _Take.apply(weights, (tokens * top_k + slots).to(device)).unsqueeze(-1)
            total = _Add.apply(total, ids, y.to(total.dtype))
        return total


def _sparse_blocks(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """(name, block) of each module holding experts, fused or not, in the model's own order."""
    for name, module in model.named_modules():
        if hasattr(getattr(module, "experts", None), "gate_up_proj"):
            yield name, module


def split(model: nn.Module) -> None:
    """Fused experts become ``Experts``, on the meta device if that is where the model is."""
    for _, block in _sparse_blocks(model):
        block.experts = Experts(block.experts)


def is_moe(model: nn.Module) -> bool:
    return next(_sparse_blocks(model), None) is not None


def attach_adapters(model: nn.Module, *, rank: int, alpha: int, dropout: float, seed: int) -> None:
    """``poros.moe``'s adapter on each block, drawn in its order from its seed, so the same bytes:
    the block owns it, and its experts read it without holding it a second time."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _, block in _sparse_blocks(model):
            experts = block.experts
            block.adapter = ExpertLoraAdapter(
                len(experts.gate_up_proj), experts.gate_up_proj[0].in_features,
                experts.down_proj[0].in_features, rank=rank, alpha=alpha, dropout=dropout,
            )
            object.__setattr__(experts, "adapter", block.adapter)


class Fused(NamedTuple):
    """One expert's gate rows, then its up rows: one tensor in a slab, two in a checkpoint."""

    halves: tuple[Any, Any]
    shape: list[int]
    dtype: str


class Row(NamedTuple):
    """One expert of a tensor a checkpoint stores fused: all a slab wants of it."""

    whole: Any
    index: int
    shape: list[int]
    dtype: str


def checkpoint_names(model: nn.Module) -> Callable[[str], str]:
    """A checkpoint's tensor name -> the model's. An expert's says which projection it is, for
    ``fuse``: no renaming can, and the loader's own check refuses a fusion it does not know."""
    from poros.moe.load import _EXPERT_KEY, _check_fusion
    from poros.train.streamed_load import _key_remapper

    remap = _key_remapper(model)
    if not is_moe(model):
        return remap
    roles = functools.cache(lambda: _check_fusion(model))  # a checkpoint of fused experts asks not

    def name(tensor: str) -> str:
        found = _EXPERT_KEY.match(tensor)
        if found is None:
            return remap(tensor)
        experts, index, proj = found.groups()
        return f"{remap(experts)}.{roles().get(proj, proj)}.{index}.weight"

    return name


def fuse(tensors: dict[str, Any]) -> dict[str, Any]:
    """Each expert under the model's names, gate and up as ``poros.moe.load`` fuses them, or each
    row of a tensor already fused, so its slab holds the bytes CUDA's holds. Anything else passes as
    it is."""
    fused = {}
    for name, stored in tensors.items():
        if _WHOLE.match(name):
            for e in range(stored.shape[0]):
                fused[f"{name}.{e}.weight"] = Row(stored, e, stored.shape[1:], stored.dtype)
            continue
        found = _ROLE.match(name)
        if found is None:
            fused[name] = stored
            continue
        experts, role, index = found.groups()
        if role == "down":
            fused[f"{experts}.down_proj.{index}.weight"] = stored
        elif role == "gate":
            up = tensors[f"{experts}.up.{index}.weight"]
            fused[f"{experts}.gate_up_proj.{index}.weight"] = Fused(
                (stored, up), [stored.shape[0] + up.shape[0], stored.shape[1]], stored.dtype
            )
    return fused


def save_adapter(model: nn.Module, directory: Path, config: Any) -> None:
    """PEFT's file for fused parameters: per expert, (r, in) rows of A and (out, r) columns of B."""
    from peft import LoraConfig
    from safetensors.torch import save_file

    tensors = {}
    for name, block in _sparse_blocks(model):
        for factor, peft_name in _PEFT_NAMES.items():
            p = getattr(block.adapter, factor).detach().cpu()
            flat = p.flatten(0, 1) if "lora_A" in factor else p.permute(1, 2, 0).flatten(1)
            tensors[f"{_PEFT_PREFIX}{name}.{peft_name}"] = flat
    save_file(tensors, directory / "adapter_model.safetensors", metadata={"format": "pt"})
    local = Path(config.model_name_or_path).is_dir()  # its "revision" is only a pack's key
    LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, task_type="CAUSAL_LM", target_modules=[],
        target_parameters=_PEFT_TARGETS, base_model_name_or_path=config.model_name_or_path,
        revision=None if local else config.model_revision,
    ).save_pretrained(str(directory))


def read_adapter(directory: Path) -> dict[str, torch.Tensor]:
    """What ``save_adapter`` wrote, under the names and in the fused shape the model has them."""
    from safetensors.torch import load_file

    rank = json.loads((directory / "adapter_config.json").read_text())["r"]
    saved = load_file(directory / "adapter_model.safetensors")
    state = {}
    for name in sorted(saved):
        for factor, peft_name in _PEFT_NAMES.items():
            if name.endswith(f".{peft_name}"):
                flat, block = saved[name], name[len(_PEFT_PREFIX) : -len(peft_name) - 1]
                state[f"{block}.adapter.{factor}"] = (
                    flat.reshape(-1, rank, flat.shape[1]) if "lora_A" in factor
                    else flat.reshape(flat.shape[0], rank, -1).permute(2, 0, 1)
                )
    return state
