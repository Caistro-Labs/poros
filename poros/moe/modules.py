"""Managed MoE sparse block (compute layer).

``ManagedMoeBlock`` is a drop-in replacement for HF's sparse block of every
``poros.moe.arch.MOE_FAMILIES`` member; they share ``Qwen3MoeSparseMoeBlock``'s
layout (transformers 5.14.1
``models/qwen3_moe/modeling_qwen3_moe.py:275-286``). Forward computes the
reference's math:

- router matmul, fp32 softmax, top-k, optional prob normalization — the
  reference ``Qwen3MoeTopKRouter`` module itself, reused verbatim (lines
  254-272); the router stays dense, resident, frozen, and unadapted
 ;
- the ``Qwen3MoeExperts.forward`` SwiGLU (lines 233-251) for every selected
  expert, from resident quantized slabs, as batched matmuls over
  expert-sorted rows instead of a per-expert loop;
- weighted combine as a sum over each token's top-k slots;
- plus any shared experts, dense and resident-or-streamed by the dense
  engine (qwen2_moe gates its one with a sigmoid).

Expert weights are never materialized as parameters: a custom
``autograd.Function`` pulls each selected expert's slab through
``contracts.ExpertResidency``, dequantizes via ``contracts.ExpertStore``,
and saves no weight tensors; backward re-``ensure``s the same slabs and
rematerializes the forward bit-exactly — the dense engine's remat
discipline (``poros/core/blockwise.py``). Input grads are exact; base
weights are frozen and receive no grads.

The quantized slabs it computes from are ``poros.moe.store``'s (``MoeExpertStore``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from poros.moe.contracts import (
    ExpertKey,
    ExpertResidency,
    ExpertStore,
    RouteDecision,
    RouteObserver,
)
from poros.moe.pinned import try_pin
from poros.moe.store import EXPERT_DQ_CACHE, MoeExpertStore
from poros.sss.source import RamSource

if TYPE_CHECKING:
    from poros.moe.adapters import ExpertLoraAdapter
    from poros.moe.arch import MoeArch


@dataclass
class RematEntry:
    """One MoE layer call kept for its backward rematerialization (``_remat_reuse``): the call's
    inputs, its expert output (None while ``poros.moe.batching`` defers it) and the random state
    before and after it."""

    hidden: torch.Tensor
    index: torch.Tensor
    weights: torch.Tensor
    out: torch.Tensor | None
    rng_before: Any
    rng_after: Any


__all__ = [
    "ManagedExperts",
    "ManagedMoeBlock",
    "install_managed_blocks",
]


def _padded_slots(counts: list[int]) -> torch.Tensor:
    """Slot of each expert-sorted row once every expert is padded to ``max(counts)`` rows."""
    lengths, m = torch.tensor(counts), max(counts)
    # Row i of expert j (whose rows start at s_j) lands in slot j * m + i - s_j.
    shift = torch.arange(len(counts)) * m - (lengths.cumsum(0) - lengths)
    return torch.arange(int(lengths.sum())) + shift.repeat_interleave(lengths)


def _expert_group(
    x: torch.Tensor,
    experts: list[int],
    counts: list[int],
    pos: torch.Tensor,
    lora: Sequence[torch.Tensor],
    block: ManagedMoeBlock,
) -> torch.Tensor:
    """One group's expert MLPs over its expert-sorted rows ``x``, as batched matmuls.

    Each expert's rows are zero-padded to the group's busiest expert (slots
    ``pos``), so a projection is one batched dequantize and one ``bmm`` (one
    more per LoRA factor) on every device; the padding rows are computed and
    dropped.
    """
    keys = [(block.layer_idx, e) for e in experts]
    x = x.new_zeros(len(counts) * max(counts), x.shape[-1]).index_put_((pos,), x)
    x = x.unflatten(0, (len(counts), -1))
    gate_up = x @ EXPERT_DQ_CACHE.get(block, keys, "gate_up_proj").mT
    if lora:
        gate_up = gate_up + block.adapter.delta(x, lora[0], lora[1])
    gate, up = gate_up.chunk(2, dim=-1)
    if block.swiglu_limit is not None:
        # Experts that clamp their SwiGLU (an ``experts.swiglu_limit``), as transformers does.
        gate = gate.clamp(min=None, max=block.swiglu_limit)
        up = up.clamp(min=-block.swiglu_limit, max=block.swiglu_limit)
    hidden = (block.act_fn(gate, up) if getattr(block, "act_takes_up", False)
              else block.act_fn(gate) * up)
    down = hidden @ EXPERT_DQ_CACHE.get(block, keys, "down_proj").mT
    if lora:
        down = down + block.adapter.delta(hidden, lora[2], lora[3])
    return down.flatten(0, 1)[pos]


def _prefetch_hint(block: ManagedMoeBlock, keys: list[ExpertKey]) -> None:
    """Tell a streaming slab source which experts the router just picked.

    ``ensure`` then reads them one by one; an ``SsdSource`` already has them in
    flight. Only keys not already resident on the device are hinted. A hint
    moves no bytes into the residency and changes no numerics.
    """
    source = getattr(block.store, "source", None)
    prefetch = getattr(source, "prefetch", None)
    if prefetch is None or isinstance(source, RamSource):
        return
    # ExpertBlockManager then loads each picked expert's next ``look_ahead`` experts
    # when there is room: hint those too, after the picked ones.
    look_ahead = int(getattr(block.residency, "_look_ahead", 0) or 0)
    picked = set(keys)
    extra = [
        (layer, e + j)
        for layer, e in keys
        for j in range(1, look_ahead + 1)
        if e + j < block.num_experts and (layer, e + j) not in picked
    ]
    keys = list(dict.fromkeys(keys + extra))
    resident = getattr(block.residency, "_resident", None)
    if resident is not None:
        keys = [k for k in keys if k not in resident]
    if keys:
        prefetch(keys)


def prefetch_neighbor(block: ManagedMoeBlock, step: int) -> None:
    """Start the next MoE layers' experts (``layer_idx + step``, ``+ 2 * step``, ...: +1 in a
    forward pass, -1 in a backward one) toward the device while this layer computes: every
    expert, since a layer call of a few hundred tokens routes to all of them. Layers already
    resident are passed over; the first that finds no room ends it (the layers between this one
    and it are never evicted for it). So the copies of the layers a pass streams start as soon as
    the layers it has run make room, and run beside its compute instead of after it.

    From a streaming source (the SSD store) only what it can hand over without a disk read is
    started -- slabs in its host cache, or whose read has landed (``ready``) -- so this never
    waits on the disk; a layer it cannot hand over whole ends it. ``POROS_MOE_PREFETCH``: how
    many layers ahead at most (default: all; 0 turns it off)."""
    depth_env = os.environ.get("POROS_MOE_PREFETCH", "")
    try:
        depth = int(depth_env) if depth_env else block.store.layout.num_layers
    except ValueError:
        depth = block.store.layout.num_layers
    prefetch = getattr(block.residency, "prefetch", None)
    if depth <= 0 or prefetch is None:
        return
    source = getattr(block.store, "source", None)
    ready = None
    if not isinstance(source, RamSource):
        ready = getattr(source, "ready", None)
        if ready is None:  # a source that cannot say what it holds: its own read-ahead only
            return
    resident = getattr(block.residency, "layer_resident", None)
    held = getattr(block.residency, "_resident", {})
    keep = [block.layer_idx]
    for d in range(1, depth + 1):
        nxt = block.layer_idx + d * step
        if not 0 <= nxt < block.store.layout.num_layers:
            break
        if resident is not None and resident(nxt):
            keep.append(nxt)
            continue
        keys = [(nxt, e) for e in range(block.num_experts)]
        whole = True
        if ready is not None:  # the slabs it can hand over now, and whether that is all of them
            missing = [k for k in keys if k not in held]
            keys = [k for k in missing if ready(k)]
            whole = len(keys) == len(missing)
            if not keys:
                break
        moved = prefetch(keys, keep=keep)
        keep.append(nxt)
        if not moved or not whole:
            break


def _experts_compute(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    block: ManagedMoeBlock,
) -> torch.Tensor:
    """The HF expert MLPs, batched: weights dequantized from resident slabs.

    ``Qwen3MoeExperts.forward``'s math (``modeling_qwen3_moe.py:233-251``):
    fused gate/up linear, chunk, ``act(gate) * up``, down linear,
    router-weight scaling, sum over each token's top-k. Experts run in
    groups of similar token count (power-of-two buckets, so padding at most
    doubles a group's rows) of at most ``block.expert_batch`` experts (0: no
    cap), so per layer the op count no longer scales with the expert count.
    Batched matmuls and the slot sum replace HF's per-expert linears and
    ``index_add_``, so results match HF up to rounding and summation order
    (Mixtral's fp32 routing weights make the weighted sum fp32, rounded
    once, where HF rounds each expert's share),
    and group shapes (set by the routing and ``expert_batch``) may change
    bits; streamed and resident runs share this path and stay bitwise equal.
    Adapter deltas add ``base + delta`` at each projection output, the
    NativeLoraLinear op order (``poros.train.adapters.NativeLoraLinear.forward``).
    """
    num_tokens, top_k = top_k_index.shape
    with torch.no_grad():
        flat = top_k_index.reshape(-1)  # row r: token r // top_k, slot r % top_k
        counts = torch.bincount(flat, minlength=block.num_experts).tolist()  # one host sync
    hit = [e for e, n in enumerate(counts) if n]
    if not hit:  # no tokens: an empty result, still on the autograd graph
        return hidden_states.clone()
    _prefetch_hint(block, [(block.layer_idx, e) for e in hit])
    block.residency.ensure([(block.layer_idx, e) for e in hit])
    if getattr(block, "expert_grouping", "pow2") == "single":
        bucket = dict.fromkeys(hit, 0)  # one load class: one group (per expert_batch chunk)
    else:
        bucket = {e: counts[e].bit_length() for e in hit}  # power-of-two load class
    hit.sort(key=bucket.get)  # stable: ascending expert ids within a bucket
    groups = []
    for _, same in groupby(hit, key=bucket.get):
        members = list(same)
        batch = block.expert_batch or len(members)
        groups += [members[i : i + batch] for i in range(0, len(members), batch)]
    host_rank = torch.zeros(block.num_experts, dtype=torch.long)
    host_rank[hit] = torch.arange(len(hit))
    slots = [_padded_slots([counts[e] for e in g]) for g in groups]
    host = torch.cat([host_rank, torch.tensor(hit), *slots])
    if hidden_states.is_cuda:  # the caching host allocator keeps it alive for the copy
        host = try_pin(host)
    sizes = [block.num_experts, len(hit), *map(len, slots)]
    rank, experts, *pos = host.to(hidden_states.device, non_blocking=True).split(sizes)  # one copy
    with torch.no_grad():
        order = rank[flat].argsort(stable=True)  # rows in ``hit`` order, token order within
    lora = [()] * len(groups)
    if block.adapter is not None:  # one gather per factor; split's backward is one cat
        factors = block.adapter.factors(experts)
        lora = list(zip(*(f.split([len(g) for g in groups]) for f in factors)))
    rows = hidden_states[order // top_k].split([len(p) for p in pos])
    # A block may bring its own group compute (e.g. a store plugin's block); same contract.
    group_fn = getattr(block, "expert_group", _expert_group)
    out = torch.cat(
        [
            group_fn(x, g, [counts[e] for e in g], p, group_lora, block)
            for x, g, p, group_lora in zip(rows, groups, pos, lora, strict=True)
        ]
    )
    out = out * top_k_weights.reshape(-1)[order, None]
    unsorted = torch.empty_like(out)
    unsorted[order] = out
    return unsorted.view(num_tokens, top_k, -1).sum(1).to(hidden_states.dtype)


class _ManagedExpertFunction(torch.autograd.Function):
    """Autograd boundary for one MoE layer's expert compute.

    Forward runs under no_grad and saves only what the dense engine's
    blockwise functions save — the input activations and the RNG state
    (INV-1, ``poros.core.blockwise.JITFrozenBlockFunction.forward``) — never weight
    tensors. Backward re-``ensure``s the same slabs (bit-identical; the store is
    immutable), restores the RNG, rematerializes the forward with grad, and
    returns exact input and routing-weight grads; adapter grads accumulate
    into ``param.grad`` (``poros.core.blockwise.JITAdapterBlockFunction.backward``'s
    pattern). Unselected experts' adapter
    params legitimately receive no grad (``allow_unused=True``).
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        block: ManagedMoeBlock,
    ) -> torch.Tensor:
        ctx.block = block
        ctx.cpu_rng_state = torch.get_rng_state()
        if hidden_states.device.type == "cuda" and torch.cuda.is_available():
            device_index = (
                hidden_states.device.index
                if hidden_states.device.index is not None
                else torch.cuda.current_device()
            )
            ctx.cuda_rng_state = torch.cuda.get_rng_state(device_index)
        else:
            ctx.cuda_rng_state = None
        ctx.save_for_backward(hidden_states, top_k_index, top_k_weights)
        precomputed = block._remat_hit
        if precomputed is not None:  # ManagedMoeBlock verified it is this call's exact output
            block._remat_hit = None
            return precomputed.clone()
        return _experts_compute(hidden_states, top_k_index, top_k_weights, block)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        hidden_states, top_k_index, top_k_weights = ctx.saved_tensors
        block = ctx.block
        device = grad_output.device
        fork_devices: list[int] = []
        if device.type == "cuda" and torch.cuda.is_available():
            fork_devices = [
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=fork_devices):
            torch.set_rng_state(ctx.cpu_rng_state)
            if ctx.cuda_rng_state is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng_state, device=device)
            with torch.enable_grad():
                remat_input = hidden_states.detach().requires_grad_(True)
                remat_weights = top_k_weights.detach().requires_grad_(True)
                remat_output = _experts_compute(remat_input, top_k_index, remat_weights, block)
            adapter = block.adapter
            params = list(adapter.parameters()) if adapter is not None else []
            grads = torch.autograd.grad(
                remat_output,
                [remat_input, remat_weights, *params],
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for param, grad in zip(params, grads[2:], strict=True):
                if grad is None:
                    continue
                if param.grad is None:
                    param.grad = grad.detach()
                else:
                    param.grad.add_(grad.detach())
        return grads[0], None, grads[1], None


class ManagedMoeBlock(nn.Module):
    """Drop-in replacement for a ``MOE_FAMILIES`` sparse block.

    Built from the HF block it replaces. The router is that block's
    ``TopKRouter`` module itself — dense, resident, frozen.
    Expert compute pulls quantized slabs through ``residency`` and
    dequantizes via ``store``; ``adapter`` is an optional resident
    ``ExpertLoraAdapter``. Shared experts move over unchanged — qwen2_moe's
    ``shared_expert`` with its sigmoid ``shared_expert_gate``, deepseek_v2's
    ``shared_experts`` — dense, frozen, unadapted, added to the routed
    output as HF does. ``on_route`` observes
    each committed ``RouteDecision``; default None, zero overhead.

    ``layer_idx`` is the MoE-layer ordinal (0-based over the model's
    MoE-carrying layers), matching the store's build order.
    """

    def __init__(
        self,
        block: nn.Module,
        layer_idx: int,
        store: ExpertStore,
        residency: ExpertResidency,
        adapter: ExpertLoraAdapter | None = None,
        on_route: RouteObserver | None = None,
    ) -> None:
        super().__init__()
        experts = block.experts
        self.layer_idx = int(layer_idx)
        self.num_experts = int(experts.num_experts)
        self.hidden_size = int(experts.hidden_dim)
        self.intermediate_size = int(experts.intermediate_dim)
        # Experts with no act_fn compute a SiLU SwiGLU, clamped where they say swiglu_limit.
        self.act_fn = getattr(experts, "act_fn", None) or nn.functional.silu
        # Experts whose activation takes the up projection too, ``act_fn(gate, up)`` (Inkling
        # 2's softcapped SwiGLU), instead of ``act_fn(gate) * up``.
        self.act_takes_up = bool(getattr(experts, "act_takes_up", False))
        self.swiglu_limit = getattr(experts, "swiglu_limit", None)
        self.store = store
        self.residency = residency
        self.adapter = adapter
        self.on_route = on_route
        self.expert_batch = 0  # max experts per batched call; 0 = no cap (MoeConfig)
        # "pow2": experts grouped by power-of-two token count (padding at most doubles a
        # group); "single": one group per call, every expert padded to the busiest -- far
        # fewer kernel launches for more padded math (MoeConfig.expert_grouping)
        self.expert_grouping = "pow2"
        # Remat output reuse (see ``_remat_reuse``); prepare_moe turns it on when the dense
        # engine streams the decoder blocks, the one setting with a no-grad pass followed by
        # a rematerializing grad pass of the same inputs.
        self.reuse_remat_output = False
        # "reference" (_experts_compute), "grouped" (poros.moe.grouped, opt-in, pinned) or
        # "fused" (poros.moe.fused: Triton kernels over the exact rows, CUDA)
        self.expert_compute = "reference"
        self.remat_hits = 0
        # how many no-grad outputs wait for their grad pass (1: the sequential path, a
        # block's forward then its remat; block-major MoE holds one per microstep)
        self.remat_slots = 1
        self._remat_cache: list | None = None
        self._remat_hit: torch.Tensor | None = None
        # block-major batching (poros.moe.batching): a no-grad call records its inputs here and
        # returns -0.0 (shadow: not into the remat cache); backward results computed ahead
        self._defer: list | None = None
        self._defer_shadow = False
        self._pending = None
        # a batched forward visit's row counts for the layer's backward visit (its routes, counts)
        self._fwd_counts: tuple | None = None
        # a batched backward visit's equality checks, as device booleans read once an update
        self._checks: list | None = None
        self.gate = getattr(block, "gate", None)
        if self.gate is not None:
            # The router is never quantized, streamed, or adapted (B8/B12).
            self.gate.weight.requires_grad_(False)
        self.shared_expert = getattr(block, "shared_expert", None)
        self.shared_expert_gate = getattr(block, "shared_expert_gate", None)
        self.shared_experts = getattr(block, "shared_experts", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The HF sparse blocks' forward (Qwen3MoeSparseMoeBlock,
        # modeling_qwen3_moe.py:281-286; Qwen2MoeSparseMoeBlock and
        # DeepseekV2Moe add their shared experts).
        # Grad seeding (mirrors poros.patch.hf.forward._install_blockwise_forward): with a
        # frozen base, nothing upstream of the first managed block requires grad, so the
        # custom autograd boundary below would never fire and
        # ``loss.backward()`` would raise "does not require grad". Seed the
        # block input — a metadata op, numerics untouched — as the dense
        # engine does for its first block.
        hidden_states = self._seeded(hidden_states)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self._routed(
            hidden_states_reshaped, selected_experts, routing_weights
        )
        if self.shared_expert is not None:
            shared = self.shared_expert(hidden_states_reshaped)
            gate = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            final_hidden_states = final_hidden_states + gate * shared
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(hidden_states_reshaped)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    @staticmethod
    def _seeded(hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and not hidden_states.requires_grad:
            return hidden_states.detach().requires_grad_(True)
        return hidden_states

    def _routed(
        self, hidden: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """The experts on a committed routing, which ``on_route`` sees."""
        if self.on_route is not None:
            self.on_route(
                RouteDecision(
                    layer=self.layer_idx,
                    expert_ids=tuple(top_k_index.reshape(-1).tolist()),
                    expert_weights=tuple(top_k_weights.detach().reshape(-1).tolist()),
                )
            )
        return self._remat_reuse(hidden, top_k_index, top_k_weights)

    def attach(self, adapter: ExpertLoraAdapter) -> None:
        self.adapter = adapter


class ManagedExperts(ManagedMoeBlock):
    """A family's fused experts, called as they are, where the decoder layer routes for itself.

    Gemma 4's layer routes on its residual and runs its experts on a norm of that, so only the
    expert compute is managed here (``arch.LAYER_ROUTED``). The layer owns the adapter, named as a
    block's, and this reads it without holding it a second time.
    """

    def __init__(self, layer: nn.Module, *args, **kwargs) -> None:
        super().__init__(layer, *args, **kwargs)
        object.__setattr__(self, "_layer", layer)  # its parent: not a submodule

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        return self._routed(self._seeded(hidden_states), top_k_index, top_k_weights)

    def attach(self, adapter: ExpertLoraAdapter) -> None:
        self._layer.adapter = adapter
        object.__setattr__(self, "adapter", adapter)


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cuda = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        cuda = torch.cuda.get_rng_state(index)
    return torch.get_rng_state(), cuda


def _set_rng_state(device: torch.device, state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device=device)


def _same_rng(a: tuple, b: tuple) -> bool:
    if not torch.equal(a[0], b[0]):
        return False
    if a[1] is None or b[1] is None:
        return a[1] is None and b[1] is None
    return torch.equal(a[1], b[1])


def _remat_reuse(
    self: ManagedMoeBlock,
    hidden: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """The expert compute, reusing the block forward's output in the backward rematerialization.

    Under dense streaming every decoder block runs twice before its backward:
    the block forward (no grad), then the rematerialization (grad) from the
    same saved block input -- bitwise the same activations, so the same
    expert output, decoded and computed a second time. With
    ``reuse_remat_output`` the no-grad pass keeps (inputs, output, RNG
    before / after); a grad pass whose inputs are ``torch.equal`` to them and
    whose RNG state equals the saved "before" returns that output (and
    advances the RNG to the saved "after", exactly as the compute would)
    instead of recomputing it. Anything else computes as before. Backward is
    unchanged. The result is bitwise the recompute's by construction, at the
    cost of holding two ``[tokens, hidden]`` tensors per MoE layer between a
    block's forward and its backward.
    """
    grad_pass = torch.is_grad_enabled()
    if self.expert_compute == "fused":
        from poros.moe.fused import FusedExpertFunction, supported

        why = supported(self, hidden)
        if why is not None:
            raise RuntimeError(f"expert_compute='fused' cannot run this call: {why}")
        function = FusedExpertFunction
    elif self.expert_compute == "grouped":
        from poros.moe.grouped import GroupedExpertFunction

        function = GroupedExpertFunction
    else:
        function = _ManagedExpertFunction
    if not self.reuse_remat_output:
        return function.apply(hidden, top_k_index, top_k_weights, self)
    device = hidden.device
    if not grad_pass and self._defer is not None:  # poros.moe.batching runs the experts later
        state = _rng_state(device)
        entry = RematEntry(hidden, top_k_index, top_k_weights, None, state, state)
        self._defer.append(entry)
        if not self._defer_shadow:
            slots = max(1, int(getattr(self, "remat_slots", 1)))
            queue = self._remat_cache if (slots > 1 and self._remat_cache) else []
            queue.append(entry)
            self._remat_cache = queue[-slots:]
        return hidden.new_full(hidden.shape, -0.0)
    if not grad_pass:
        before = _rng_state(device)
        out = function.apply(hidden, top_k_index, top_k_weights, self)
        entry = RematEntry(hidden, top_k_index, top_k_weights, out, before, _rng_state(device))
        slots = max(1, int(getattr(self, "remat_slots", 1)))
        queue = self._remat_cache if (slots > 1 and self._remat_cache) else []
        queue.append(entry)
        self._remat_cache = queue[-slots:]
        return out
    queue = self._remat_cache or []
    cached = queue.pop(0) if queue else None  # FIFO: grad passes come in forward order
    self._remat_cache = queue or None
    if cached is not None:
        c_hidden, c_index, c_weights = cached.hidden, cached.index, cached.weights
        c_out, before, after = cached.out, cached.rng_before, cached.rng_after
        checks = self._checks
        if (
            checks is not None  # a batched visit (poros.moe.batching): checked on the device
            and c_out is not None
            and c_hidden.shape == hidden.shape
            and c_index.shape == top_k_index.shape
            and _same_rng(before, _rng_state(device))
        ):
            checks.append(torch.stack([
                (c_hidden == hidden.detach()).all(),
                (c_index == top_k_index).all(),
                (c_weights == top_k_weights.detach()).all(),
            ]).all())
            self._remat_hit = c_out
            out = function.apply(hidden, top_k_index, top_k_weights, self)
            _set_rng_state(device, after)
            self.remat_hits += 1
            return out
        if (
            c_hidden.shape == hidden.shape
            and torch.equal(c_hidden, hidden.detach())
            and torch.equal(c_index, top_k_index)
            and torch.equal(c_weights, top_k_weights.detach())
            and _same_rng(before, _rng_state(device))
        ):
            self._remat_hit = c_out
            out = function.apply(hidden, top_k_index, top_k_weights, self)
            _set_rng_state(device, after)
            self.remat_hits += 1
            return out
    return function.apply(hidden, top_k_index, top_k_weights, self)


ManagedMoeBlock._remat_reuse = _remat_reuse


def install_managed_blocks(
    model: nn.Module,
    arch: MoeArch,
    store: MoeExpertStore,
    residency: ExpertResidency,
    on_route: RouteObserver | None = None,
) -> int:
    """Swap the sparse block of each of ``arch``'s MoE layers for a managed block (B10).

    Pure module replacement: each managed block keeps the original router
    (and shared-expert) modules, same parameter objects, and drops the
    fused expert parameters,
    whose weights now live in ``store``. The rest of the model is
    untouched. Returns the number of replaced blocks. Raises before
    touching the model when the store does not match the model's MoE
    layers.
    """
    targets = [
        (name, model.get_submodule(name))
        for name in map(arch.block_module, arch.moe_layer_indices)
    ]
    if len(targets) != store.layout.num_layers:
        raise ValueError(
            f"model has {len(targets)} MoE blocks but the store covers "
            f"{store.layout.num_layers} layer(s); build the store from this model"
        )
    shapes = store.tensor_shapes
    for name, module in targets:
        for tensor_name, shape in shapes.items():
            fused = getattr(getattr(module, "experts", None), tensor_name, None)
            if fused is None or tuple(fused.shape[1:]) != shape:
                raise ValueError(
                    f"{name}.experts.{tensor_name} per-expert shape "
                    f"{None if fused is None else tuple(fused.shape[1:])} does not match "
                    f"the store's {shape}; build the store from this model"
                )
    for ordinal, (name, module) in enumerate(targets):
        if arch.layer_routed:
            module.experts = ManagedExperts(module, ordinal, store, residency, on_route=on_route)
            continue
        managed = ManagedMoeBlock(module, ordinal, store, residency, on_route=on_route)
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, managed)
    return len(targets)
