"""Blockwise autograd.Functions for Poros.

Implements JITFrozenBlockFunction and JITAdapterBlockFunction: the custom
autograd.Functions that route forward/backward through blockwise residency
with per-block RNG save/restore for bitwise-exact rematerialization.

INV-1: Per-block RNG save/restore. Forward stores torch+CUDA (or torch+MPS)
RNG state; backward restores it via torch.random.fork_rng before recompute. This
guarantees dropout masks in the recomputed forward are bitwise identical
to those in the original forward.

Ported from the validated blockwise engine (training/remat.py).
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from torch.autograd.function import once_differentiable

from poros.core import backend


class BlockHost(Protocol):
    """What these autograd Functions call on the ``trainer`` they are given. Two classes provide
    it: ``poros.train.trainer.PorosTrainer`` and ``poros.patch.hf.forward._BlockwiseContext``
    (``patch_hf_model``). ``config`` needs ``preserve_rng_state`` and
    ``offload_suffix_inputs_to_cpu`` (read with defaults)."""

    config: Any
    block_manager: Any  # poros.core.residency.BlockManager
    block_trainable_params: Dict[int, List[Tuple[str, torch.nn.Parameter]]]

    def _record_remat_event(self, event: str, block_id: int) -> None: ...

    def _forward_block_layers(self, block_id: int, hidden_states: torch.Tensor,
                              attention_mask: Optional[Any], position_ids: Optional[Any],
                              checkpoint: bool = False) -> torch.Tensor: ...

    def _get_adapter_prefetch_stream(self) -> Optional[Any]: ...


class JITFrozenBlockFunction(torch.autograd.Function):
    """Custom autograd boundary for a frozen transformer block.

    Forward: load block, run with no_grad, save CPU+CUDA RNG state, release.
    Backward: reload block, fork_rng with restored state, rematerialize
    forward with enable_grad, compute grad w.r.t. input only, release.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        trainer: "BlockHost",
        block_id: int,
    ) -> torch.Tensor:
        backend.refuse_on_mps(hidden_states.device, "The patched frozen block path")
        ctx.trainer = trainer
        ctx.block_id = int(block_id)
        ctx.input_device = hidden_states.device
        ctx.pin_memory_enabled = bool(
            getattr(trainer.block_manager, "pin_memory_enabled", False)
        )
        ctx.offload_saved_input = bool(
            getattr(trainer.config, "offload_suffix_inputs_to_cpu", False)
            and hidden_states.device.type == "cuda"
        )

        # -- INV-1: save per-block RNG state --
        if getattr(trainer.config, "preserve_rng_state", True):
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
        else:
            ctx.cpu_rng_state = None
            ctx.cuda_rng_state = None

        # Detach and optionally offload saved tensors
        saved_hidden = hidden_states.detach()
        saved_attention = (
            attention_mask.detach()
            if isinstance(attention_mask, torch.Tensor)
            else None
        )
        saved_position = (
            position_ids.detach()
            if isinstance(position_ids, torch.Tensor)
            else None
        )

        if ctx.offload_saved_input:
            saved_hidden = saved_hidden.cpu()
            if saved_attention is not None:
                saved_attention = saved_attention.cpu()
            if saved_position is not None:
                saved_position = saved_position.cpu()

        ctx.saved_hidden_states = saved_hidden
        ctx.saved_attention_mask = saved_attention
        ctx.saved_position_ids = saved_position

        # Load block, run forward, release
        trainer._record_remat_event("forward_load", ctx.block_id)
        trainer.block_manager.load_block_for_inference(
            ctx.block_id, device=hidden_states.device
        )
        try:
            with torch.no_grad():
                output = trainer._forward_block_layers(
                    ctx.block_id,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    checkpoint=False,
                )
        finally:
            trainer.block_manager.release_block(ctx.block_id)
            trainer._record_remat_event("forward_release", ctx.block_id)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        trainer = ctx.trainer
        block_id = ctx.block_id
        device = grad_output.device
        non_blocking = bool(ctx.pin_memory_enabled and device.type == "cuda")

        # Restore saved tensors
        hidden_states = ctx.saved_hidden_states
        if hidden_states.device != device:
            hidden_states = hidden_states.to(device, non_blocking=non_blocking)

        attention_mask = ctx.saved_attention_mask
        if isinstance(attention_mask, torch.Tensor) and attention_mask.device != device:
            attention_mask = attention_mask.to(device, non_blocking=non_blocking)

        position_ids = ctx.saved_position_ids
        if isinstance(position_ids, torch.Tensor) and position_ids.device != device:
            position_ids = position_ids.to(device, non_blocking=non_blocking)

        # Reload block for rematerialization
        trainer._record_remat_event("backward_load", block_id)
        trainer.block_manager.load_block_for_inference(block_id, device=device)

        try:
            # -- INV-1: fork RNG and restore saved state --
            fork_devices: list[int] = []
            if device.type == "cuda" and torch.cuda.is_available():
                fork_devices = [
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ]

            with torch.random.fork_rng(
                devices=fork_devices, enabled=ctx.cpu_rng_state is not None
            ):
                if ctx.cpu_rng_state is not None:
                    torch.set_rng_state(ctx.cpu_rng_state)
                if fork_devices and ctx.cuda_rng_state is not None:
                    torch.cuda.set_rng_state(ctx.cuda_rng_state, device=device)

                # Rematerialize forward with grad tracking
                with torch.enable_grad():
                    remat_input = hidden_states.detach().requires_grad_(True)
                    remat_output = trainer._forward_block_layers(
                        block_id,
                        remat_input,
                        attention_mask,
                        position_ids,
                        checkpoint=False,
                    )

                grad_input = torch.autograd.grad(
                    remat_output,
                    remat_input,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
        finally:
            trainer.block_manager.release_block(block_id)
            trainer._record_remat_event("backward_release", block_id)

        return grad_input, None, None, None, None


class JITAdapterBlockFunction(torch.autograd.Function):
    """Custom autograd boundary for blocks with frozen backbone + trainable adapters.

    Like JITFrozenBlockFunction but also computes gradients for block-local
    adapter parameters and accumulates them into param.grad.  Includes
    backward prefetch overlap: while this block's backward is computing,
    the previous block is prefetched on a dedicated CUDA stream.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        trainer: "BlockHost",
        block_id: int,
    ) -> torch.Tensor:
        ctx.trainer = trainer
        ctx.block_id = int(block_id)
        ctx.input_device = hidden_states.device
        ctx.pin_memory_enabled = bool(
            getattr(trainer.block_manager, "pin_memory_enabled", False)
        )
        ctx.offload_saved_input = bool(
            getattr(trainer.config, "offload_suffix_inputs_to_cpu", False)
            and hidden_states.device.type == "cuda"
        )

        # -- INV-1: save per-block RNG state --
        if getattr(trainer.config, "preserve_rng_state", True):
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
            ctx.mps_rng_state = (
                torch.mps.get_rng_state() if hidden_states.device.type == "mps" else None
            )
        else:
            ctx.cpu_rng_state = None
            ctx.cuda_rng_state = None
            ctx.mps_rng_state = None

        # Detach and optionally offload saved tensors
        saved_hidden = hidden_states.detach()
        saved_attention = (
            attention_mask.detach()
            if isinstance(attention_mask, torch.Tensor)
            else None
        )
        saved_position = (
            position_ids.detach()
            if isinstance(position_ids, torch.Tensor)
            else None
        )

        if ctx.offload_saved_input:
            saved_hidden = saved_hidden.cpu()
            if saved_attention is not None:
                saved_attention = saved_attention.cpu()
            if saved_position is not None:
                saved_position = saved_position.cpu()

        ctx.saved_hidden_states = saved_hidden
        ctx.saved_attention_mask = saved_attention
        ctx.saved_position_ids = saved_position

        # Load block, run forward, release
        trainer._record_remat_event("adapter_forward_load", ctx.block_id)
        trainer.block_manager.load_block_for_inference(
            ctx.block_id, device=hidden_states.device
        )

        try:
            with torch.no_grad():
                output = trainer._forward_block_layers(
                    ctx.block_id,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    checkpoint=False,
                )
        finally:
            trainer.block_manager.release_block(ctx.block_id)
            trainer._record_remat_event("adapter_forward_release", ctx.block_id)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        trainer = ctx.trainer
        block_id = ctx.block_id
        device = grad_output.device
        non_blocking = bool(ctx.pin_memory_enabled and device.type == "cuda")

        # Restore saved tensors
        hidden_states = ctx.saved_hidden_states
        if hidden_states.device != device:
            hidden_states = hidden_states.to(device, non_blocking=non_blocking)

        attention_mask = ctx.saved_attention_mask
        if isinstance(attention_mask, torch.Tensor) and attention_mask.device != device:
            attention_mask = attention_mask.to(device, non_blocking=non_blocking)

        position_ids = ctx.saved_position_ids
        if isinstance(position_ids, torch.Tensor) and position_ids.device != device:
            position_ids = position_ids.to(device, non_blocking=non_blocking)

        # Collect trainable adapter parameters for this block. A lost
        # requires_grad or emptied data here means the residency invariant
        # was violated upstream -- fail loudly instead of silently skipping.
        trainable_params = _collect_block_trainable_params(trainer, block_id)

        # Wait for any pending prefetch (from prior backward block or forward tail)
        prefetch_stream = trainer._get_adapter_prefetch_stream()
        if prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(prefetch_stream)

        # Load this block for backward rematerialization
        trainer._record_remat_event("adapter_backward_load", block_id)
        trainer.block_manager.load_block_for_inference(block_id, device=device)

        # Overlap: prefetch next backward block while this one computes
        if prefetch_stream is not None and block_id > 0:
            trainer.block_manager.load_block_for_inference(
                block_id - 1,
                device=device,
                stream=prefetch_stream,
            )

        try:
            # -- INV-1: fork RNG and restore saved state --
            fork_devices: list[int] = []
            if device.type == "cuda" and torch.cuda.is_available():
                fork_devices = [
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ]
            # Only mps names its device type: off mps this is the call as it always was.
            fork_type = {}
            if device.type == "mps":
                fork_devices, fork_type = [0], {"device_type": "mps"}

            with torch.random.fork_rng(
                devices=fork_devices, enabled=ctx.cpu_rng_state is not None, **fork_type
            ):
                if ctx.cpu_rng_state is not None:
                    torch.set_rng_state(ctx.cpu_rng_state)
                if fork_devices and ctx.cuda_rng_state is not None:
                    torch.cuda.set_rng_state(ctx.cuda_rng_state, device=device)
                if ctx.mps_rng_state is not None:
                    torch.mps.set_rng_state(ctx.mps_rng_state)

                # Rematerialize forward with grad tracking
                scope = getattr(trainer, "_remat_scope", None)
                with torch.enable_grad(), (scope(block_id) if scope is not None
                                           else contextlib.nullcontext()):
                    remat_input = hidden_states.detach().requires_grad_(True)
                    remat_output = trainer._forward_block_layers(
                        block_id,
                        remat_input,
                        attention_mask,
                        position_ids,
                        checkpoint=False,
                    )

                # Compute grads for input AND adapter params
                grad_targets = [remat_input, *trainable_params]
                grads = torch.autograd.grad(
                    remat_output,
                    grad_targets,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_input = grads[0]

                # Accumulate adapter gradients
                for param, grad in zip(trainable_params, grads[1:]):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad.detach()
                    else:
                        param.grad.add_(grad.detach())
        finally:
            trainer.block_manager.release_block(block_id)
            trainer._record_remat_event("adapter_backward_release", block_id)

        return grad_input, None, None, None, None


def rematerialize_adapter_block(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    trainer: "BlockHost",
    block_id: int,
) -> torch.Tensor:
    """Apply a rematerialized block boundary that also accumulates adapter grads."""
    return JITAdapterBlockFunction.apply(
        hidden_states,
        attention_mask,
        position_ids,
        trainer,
        int(block_id),
    )


def _collect_block_trainable_params(trainer: "BlockHost", block_id: int) -> list:
    """Return this block's trainable adapter params, enforcing residency.

    Adapter params must stay trainable and materialized for the lifetime of a
    step (the adapter-residency invariant); any violation is a hard error,
    never a silent skip.
    """
    params = []
    for name, param in trainer.block_trainable_params.get(block_id, []):
        if not param.requires_grad:
            raise RuntimeError(
                "[Poros adapter residency failure] Adapter param lost "
                f"requires_grad=True in block {block_id}: {name}"
            )
        if param.data.numel() == 0:
            raise RuntimeError(
                "[Poros adapter residency failure] Adapter param has empty "
                f"data in block {block_id}: {name}"
            )
        params.append(param)
    return params
