"""Block-major MoE: one expert compute per layer visit for all of an update's micro-batches.

Under block-major accumulation (``poros.train.block_major``) a MoE layer runs for every
micro-batch of the update in a row, forward and backward. With the fused expert compute
(``poros.moe.fused``) a micro-batch's values do not depend on the other rows in its call, so the
visit runs the experts once for all micro-batches ("segments") -- each expert's GEMMs get every
micro-batch's rows, its weights are read once -- and each micro-batch still gets exactly what
its own call gives: the update stays the sequential loop's, bit for bit.

Forward: each micro-batch's layer runs with its MoE block deferring: it routes, records its input
and returns -0.0, so the layer's output is its residual stream exactly (``x + -0.0 == x`` for
every x). ``ForwardVisit.finish`` then runs the experts once and adds each micro-batch's expert
output into its layer output in place -- ``residual + experts``, the add the layer makes. That
holds for a layer whose output is ``residual + mlp(...)`` with the MoE block as the whole MLP;
the first visit of each decoder-layer class checks it against the layer's own forward, bit for
bit, before any visit of that class batches (a class that fails runs per micro-batch).

Backward: the expert output's grad is then the layer output's grad, so ``backward`` computes every
micro-batch's input grad, routing-weight grad and adapter grads in one call before the visit's
per-micro-batch backward; each micro-batch's backward takes its own and adds the adapter grads in
micro-batch order, as the sequential loop does. That its recomputed expert inputs and its grad are
the ones the call used is checked on the device and read once per update (``verify``, before the
optimizer step: a mismatch stops the update), so the visit never waits on the GPU.

Expert LoRA dropout: each micro-batch's masks are drawn from the random state its own call starts
in (``poros.moe.fused.Masks``), so they are the masks its own call draws.

Batching needs the fused compute, no shared experts, and one MoE layer per streamed block
(block-major MoE streams one layer per block). ``POROS_MOE_BATCH=0`` turns it off.

State lives in a ``BatchState``, one per trainer (``poros.train.block_major``): which decoder-layer
classes passed their check, the visit counts, and the current update's device checks.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

__all__ = ["BatchState", "ForwardVisit"]


@dataclass
class BatchState:
    """One trainer's batching state."""

    #: decoder-layer class -> True (checked: batches) / False (failed: per micro-batch)
    checked: dict[type, bool] = field(default_factory=dict)
    #: batched visits so far, and the backward visits that planned on their forward visit's row
    #: counts (tests, logs)
    visits: dict[str, int] = field(
        default_factory=lambda: {"forward": 0, "backward": 0, "counts_reused": 0})
    #: this update's checks of its batched backward visits (device booleans; ``verify``)
    checks: list = field(default_factory=list)

    def forward(self, modules: Sequence[torch.nn.Module],
                micro_batches: int) -> ForwardVisit | None:
        """A forward visit of these streamed-block layers with their MoE layer batched (or, at
        one micro-batch, computed as before but keeping no expert output for the backward), or
        None: run as before."""
        block = _block_of(modules)
        if block is None or micro_batches < 1 or block.remat_slots != micro_batches:
            return None
        layer_class = type(modules[0])
        if self.checked.get(layer_class) is False:
            return None
        return ForwardVisit(self, block, layer_class, micro_batches)

    def backward(self, modules: Sequence[torch.nn.Module],
                 grads: Sequence[torch.Tensor]) -> None:
        """Before a block visit's per-micro-batch backward: every micro-batch's expert backward
        in one call, assuming expert output grad == layer output grad (``grads``, micro-batch
        order); each micro-batch's ``FusedExpertFunction.backward`` checks and takes its own."""
        block = _block_of(modules)
        if block is None or len(grads) < 2 or not self.checked.get(type(modules[0])):
            return
        entries = list(block._remat_cache or [])
        if len(entries) != len(grads) or any(e.out is None for e in entries):
            return
        gys = [g.reshape(-1, g.shape[-1]) for g in grads]
        if any(gy.shape != e.hidden.shape for gy, e in zip(gys, entries)):
            return
        from poros.moe.fused import Plan, experts_backward

        hidden, index, weights, segments = _cat(entries)
        counts = _take_counts(block, entries)
        plan = Plan.build(index, block.num_experts, segments=segments, counts=counts)
        if counts is not None:
            self.visits["counts_reused"] += 1
        with torch.no_grad():
            dx, dw, seg_grads = experts_backward(hidden, weights, torch.cat(gys), plan, block,
                                                 _states(block, entries))
        dxs, dws = dx.split(segments), dw.split(segments)
        block._pending = deque(
            (e.hidden, gy, dxs[g], dws[g], seg_grads[g] if seg_grads else ())
            for g, (e, gy) in enumerate(zip(entries, gys)))
        block._checks = self.checks  # each micro-batch's reuse checked on the device (verify)
        self.visits["backward"] += 1

    @staticmethod
    def end_backward(modules: Sequence[torch.nn.Module]) -> None:
        """After a block visit's backward: drop what no micro-batch took."""
        block = _block_of(modules)
        if block is not None:
            block._pending = None
            block._checks = None
            block._fwd_counts = None

    def reset(self) -> None:
        """Drop the checks of an update that did not reach ``verify`` (it failed)."""
        self.checks.clear()

    def verify(self) -> None:
        """Once an update's passes end, before its step: every batched backward visit's
        assumption -- each micro-batch's recomputed expert inputs equal to its forward's, its
        expert output grad equal to its layer output grad -- held, bit for bit (one device read
        for the update). Raises otherwise: the update is not applied (fail closed)."""
        if not self.checks:
            return
        ok = bool(torch.stack(self.checks).all())
        self.checks.clear()
        if not ok:
            raise RuntimeError(
                "block-major MoE batching: a micro-batch's recomputed expert inputs or expert "
                "output grad differed from the ones its batched backward used; the update is "
                "not applied. Set POROS_MOE_BATCH=0 to compute each micro-batch's experts on "
                "their own.")


def _block_of(modules: Sequence[torch.nn.Module]):
    """The one batchable MoE block of a streamed block's layers, or None."""
    from poros.moe.modules import ManagedMoeBlock

    if os.environ.get("POROS_MOE_BATCH", "1") == "0" or len(modules) != 1:
        return None
    found = [m for m in modules[0].modules() if isinstance(m, ManagedMoeBlock)]
    if len(found) != 1:
        return None
    block = found[0]
    if type(block) is not ManagedMoeBlock or block.expert_compute != "fused":
        return None
    if block.shared_expert is not None or block.shared_experts is not None:
        return None
    if not block.reuse_remat_output:
        return None
    return block


def _cat(entries):
    hidden = torch.cat([e.hidden for e in entries])
    index = torch.cat([e.index for e in entries])
    weights = torch.cat([e.weights for e in entries])
    return hidden, index, weights, [e.hidden.shape[0] for e in entries]


def _keep_counts(block, entries, plan) -> None:
    """Keep a batched forward visit's row counts per (expert, segment) -- host memory, a few KB --
    for the layer's backward visit (``_take_counts``); the plan's device tensors are freed."""
    block._fwd_counts = (tuple(e.index for e in entries), plan.counts)


def _take_counts(block, entries):
    """The row counts the layer's forward visit found, for its backward visit over ``entries``,
    so its plan is built without the host sync that counts them; None (count them) unless the
    forward routed exactly these entries' index tensors -- the same tensor objects, in the same
    order, which nothing writes to between the passes -- so they are the counts ``Plan.build``
    finds for them. Taken once.

    Identity cannot see a write into one of those tensors between the passes. Nothing makes one;
    if something did, the plan built on stale counts would be wrong, and no micro-batch would
    take its result unchecked: each one's rematerialization compares its recomputed routes with
    the entry's index on the device (``c_index == top_k_index`` in
    ``poros.moe.modules._remat_reuse``, read at ``verify``), over every element, so the update
    stops before its optimizer step. A second check here of each group's first and last sorted
    rows was left out: it would see only some such writes, where that one sees all."""
    held, block._fwd_counts = block._fwd_counts, None
    if held is None:
        return None
    routes, counts = held
    if len(routes) != len(entries) or any(a is not e.index for a, e in zip(routes, entries)):
        return None
    return counts


def _states(block, entries):
    """Each micro-batch's random state at its expert call (its dropout masks start there)."""
    from poros.moe.fused import dropout_p

    return [e.rng_before for e in entries] if dropout_p(block) else None


def _experts(block, entries, keep: bool = False) -> list[torch.Tensor]:
    """The deferred calls' expert outputs, one call; each entry's random state after its own
    draws becomes its ``after`` (the state its own call would leave). ``keep``: the plan's row
    counts wait for the layer's backward visit (``_take_counts``)."""
    from poros.moe.fused import Plan, _forward

    hidden, index, weights, segments = _cat(entries)
    plan = Plan.build(index, block.num_experts, segments=segments)
    with torch.no_grad():
        out, masks = _forward(hidden, weights, plan, block, _states(block, entries))
    if masks is not None:
        for entry, after in zip(entries, masks.after, strict=True):
            entry.rng_after = after
    if keep:
        _keep_counts(block, entries, plan)
    return list(out.split(segments))


def _unused(y: torch.Tensor) -> torch.Tensor:
    """What the rematerialization's expert call returns in place of ``y``: zeros of its shape,
    holding no memory. The value roots no gradient -- in a single-layer block whose output is
    ``residual + experts`` the layer output is the block output, read only as the root of
    ``autograd.grad`` -- so a verified visit keeps no expert output for its backward."""
    return torch.zeros((), dtype=y.dtype, device=y.device).expand(y.shape)


class ForwardVisit:
    """One block visit's forward for G micro-batches with its MoE layer batched (G >= 2)."""

    def __init__(self, state: BatchState, block, layer_class: type, micro_batches: int) -> None:
        self.state = state
        self.block = block
        self.layer_class = layer_class
        self.micro_batches = micro_batches
        self.checking = layer_class not in state.checked
        self.entries: list = []
        self.outs: list[torch.Tensor] = []

    @property
    def deferring(self) -> bool:
        return not self.checking and self.micro_batches > 1

    def run(self, g: int, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Micro-batch g's layer forward (``fn``), its experts deferred to ``finish``."""
        block = self.block
        if not self.deferring:  # computed per micro-batch; a checking visit's g = 0 also deferred
            out = fn()
            if self.checking and g == 0:
                self.state.checked[self.layer_class] = self._check(fn, out)
            return out
        block._defer, block._defer_shadow = self.entries, False
        try:
            out = fn()
        finally:
            block._defer = None
        self.outs.append(out)
        return out

    def _check(self, fn, out: torch.Tensor) -> bool:
        shadow: list = []
        block = self.block
        block._defer, block._defer_shadow = shadow, True
        try:
            deferred = fn()
        finally:
            block._defer, block._defer_shadow = None, False
        ok = len(shadow) == 1 and deferred.shape == out.shape
        if ok:
            deferred.add_(_experts(block, shadow)[0].view_as(deferred))
            ok = torch.equal(deferred, out)
        if not ok:
            logger.warning("block-major MoE: %s is not residual + MoE block, bit for bit; its "
                           "experts run per micro-batch", self.layer_class.__name__)
        return ok

    def finish(self) -> None:
        """Run the deferred experts once and add each micro-batch's output into its layer
        output. A verified visit then keeps no expert output for the rematerialization
        (``_unused``)."""
        if self.deferring and self.entries:
            if len(self.entries) != len(self.outs):
                raise RuntimeError("block-major MoE: a deferred layer call without its output")
            ys = _experts(self.block, self.entries, keep=True)
            for entry, out, y in zip(self.entries, self.outs, ys, strict=True):
                entry.out = y
                out.add_(y.view_as(out))
            self.state.visits["forward"] += 1
        queue = self.block._remat_cache
        if not self.state.checked.get(self.layer_class) or not queue:
            return
        for i in range(max(0, len(queue) - self.micro_batches), len(queue)):
            e = queue[i]
            if e.out is not None:
                queue[i] = dataclasses.replace(e, out=_unused(e.out))
