"""A streamed autograd boundary whose state is several tensors: one *visit* of a decoder's layers.

``poros.core.blockwise`` streams a block of decoder layers that maps one hidden state to the
next. Some decoders carry more between layers: Inkling 2 Mini reads a stack of earlier block
states at every layer and, in a loop pass, the previous pass's states too, and it runs some
layers more than once a forward (poros.moe.inkling2). ``JITVisitFunction`` is the same
boundary for them, unchanged in its discipline (INV-1 of poros.core.blockwise):

- forward: save the random state and the input tensors (no weights), load the block, run the
  visit under no_grad, release the block;
- backward: load the block again, restore the random state, rerun the visit with grad from the
  saved inputs, and take the gradient of every input tensor and of the visit's trainable
  parameters in one ``autograd.grad``; parameter gradients are added into ``param.grad``, so a
  layer visited several times a forward sums the gradients of all its visits there.

Families whose blocks carry one tensor never reach this module: the single-tensor functions in
poros.core.blockwise are what they run, unchanged.

The host (``VisitHost``) loads and releases what a visit needs (it may keep a block loaded across
consecutive visits of that block: loading is idempotent, and the numbers never depend on it),
runs the visit, and lists its trainable parameters.
"""

from __future__ import annotations

import contextlib
from typing import Any, List, Protocol, Sequence, Tuple

import torch
from torch.autograd.function import once_differentiable

__all__ = ["VisitHost", "JITVisitFunction", "run_visit"]


class VisitHost(Protocol):
    def load_visit(self, key: Any, device: torch.device) -> None: ...

    def release_visit(self, key: Any, backward: bool) -> None: ...

    def run_visit(self, key: Any, tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]: ...

    def visit_params(self, key: Any) -> List[torch.nn.Parameter]: ...


def _rng_state(device: torch.device) -> Tuple[Any, Any, Any]:
    cpu = torch.get_rng_state()
    cuda = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        cuda = torch.cuda.get_rng_state(index)
    mps = torch.mps.get_rng_state() if device.type == "mps" else None
    return cpu, cuda, mps


@contextlib.contextmanager
def _restored_rng(device: torch.device, state: Tuple[Any, Any, Any]):
    cpu, cuda, mps = state
    devices: List[int] = []
    kind = {}
    if device.type == "cuda" and torch.cuda.is_available():
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    if device.type == "mps":
        devices, kind = [0], {"device_type": "mps"}
    with torch.random.fork_rng(devices=devices, **kind):
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device=device)
        if mps is not None:
            torch.mps.set_rng_state(mps)
        yield


class JITVisitFunction(torch.autograd.Function):
    """``apply(host, key, *tensors) -> tuple of tensors``: one streamed visit (module doc)."""

    @staticmethod
    def forward(ctx: Any, host: VisitHost, key: Any, *tensors: torch.Tensor):  # type: ignore[override]
        device = tensors[0].device
        ctx.host, ctx.key, ctx.device = host, key, device
        ctx.rng = _rng_state(device)
        ctx.save_for_backward(*tensors)
        host.load_visit(key, device)
        try:
            with torch.no_grad():
                outs = tuple(host.run_visit(key, tensors))
        finally:
            host.release_visit(key, backward=False)
        if any(any(o is t for t in tensors) for o in outs):
            raise RuntimeError("[Poros visit] a visit returned one of its inputs as an output")
        return outs

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_outputs: torch.Tensor):  # type: ignore[override]
        host, key, device = ctx.host, ctx.key, ctx.device
        saved = ctx.saved_tensors
        params = host.visit_params(key)
        for param in params:
            if not param.requires_grad or param.data.numel() == 0:
                raise RuntimeError(
                    f"[Poros adapter residency failure] a visit's adapter param "
                    f"lost requires_grad or its data ({key})"
                )
        host.load_visit(key, device)
        try:
            with _restored_rng(device, ctx.rng):
                with torch.enable_grad():
                    inputs = [t.detach().requires_grad_(t.is_floating_point()) for t in saved]
                    outs = tuple(host.run_visit(key, inputs))
                pairs = [
                    (o, g) for o, g in zip(outs, grad_outputs) if o.requires_grad and g is not None
                ]
                targets = [t for t in inputs if t.requires_grad] + list(params)
                grads = torch.autograd.grad(
                    [o for o, _ in pairs],
                    targets,
                    grad_outputs=[g for _, g in pairs],
                    allow_unused=True,
                )
        finally:
            host.release_visit(key, backward=True)
        n_in = sum(1 for t in inputs if t.requires_grad)
        for param, grad in zip(params, grads[n_in:]):
            if grad is None:
                continue
            if param.grad is None:
                param.grad = grad.detach()
            else:
                param.grad.add_(grad.detach())
        it = iter(grads[:n_in])
        input_grads = [next(it) if t.requires_grad else None for t in inputs]
        return (None, None, *input_grads)


def run_visit(
    host: VisitHost, key: Any, tensors: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """One visit through ``JITVisitFunction``; under grad the first tensor is seeded to require
    grad (a frozen embedding output otherwise leaves the graph unconnected, as in
    poros.core.blockwise)."""
    tensors = list(tensors)
    if torch.is_grad_enabled() and not any(t.requires_grad for t in tensors):
        tensors[0] = tensors[0].detach().requires_grad_(True)
    return JITVisitFunction.apply(host, key, *tensors)
