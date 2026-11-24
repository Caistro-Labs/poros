"""Inkling 2 Mini's decoder as a chain of visits, streamed or resident, one number either way.

The decoder runs its layers as a schedule of visits (poros.moe.inkling2.modeling): the layers
before the loop once, the loop layers once per pass with shared weights, the layers after once.
Every visit, one layer in one pass, is one autograd boundary, ``poros.core.visits.
JITVisitFunction``, whose state is several tensors:

- the partial residual,
- every block state so far (the block attention residuals read all of them),
- in a loop pass after the first, the previous pass's message sources.

Its backward recomputes the visit from those saved inputs and returns a gradient for each of
them, so the gradient that flows through the block attention residuals and the messages reaches
the visits that produced them. A loop layer's adapters' gradients from every pass add up in
``param.grad``.

The boundaries are the same whether the weights stream or stay resident, and whatever the block
size: where gradients that fan out across visits are summed is fixed by the visits alone. So a
streamed run, a resident run and runs at any block size give the same numbers, bit for bit (a
plain autograd run of the same decoder sums those gradients in another order: it agrees to
rounding, tests/moe/test_inkling2.py).

Streaming: the dense engine's ``BlockManager`` holds the attention weights (the experts stream
through poros.moe's expert residency; the small resident pieces stay put). A visit loads its
layer's block; the block is released when the next visit to run, forward or backward, is in
another block, so consecutive visits of one block load it once.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from poros.core.visits import run_visit
from poros.moe.inkling2.modeling import VisitContext

__all__ = ["VisitKey", "VisitRunner", "install_visits", "install_visit_streaming"]


@dataclass(eq=False)
class VisitKey:
    """One visit: its layer, pass and place in the schedule, and its input tensors' layout."""

    layer: int
    pass_idx: int
    position: int  # its index in the forward's visit schedule
    n_blocks: int
    n_msg_blocks: Optional[int]  # None: no messages (the first pass, or outside the loop)
    first_appends_input: bool  # the layer is a block boundary: it appends its own input
    ctx: VisitContext = field(repr=False)


class VisitRunner:
    """Installed as ``Inkling2LoopTextModel.poros_visit_runner``; the host of the visits'
    ``JITVisitFunction`` boundaries. ``block_manager`` None: the weights are resident."""

    def __init__(self, text_model: nn.Module, block_manager: Any = None) -> None:
        self._model = weakref.ref(text_model)
        self.block_manager = block_manager
        self.schedule: List[Tuple[int, int]] = [
            (layer, p) for layers, p in text_model.segments() for layer in layers
        ]
        self._position = {visit: i for i, visit in enumerate(self.schedule)}
        self._layer_block: Dict[int, int] = {}
        if block_manager is not None:
            self._layer_block = {
                i: b.block_id for b in block_manager.blocks for i in b.layer_indices
            }
        self._params: Dict[Tuple[int, bool], List[nn.Parameter]] = {}
        self.visits = 0  # boundaries run forward (evidence for a test that they ran)
        self.loads = 0  # block loads asked for

    @property
    def model(self) -> nn.Module:
        model = self._model()
        if model is None:  # pragma: no cover - the runner lives on the model
            raise RuntimeError("the model this runner served is gone")
        return model

    def run(
        self,
        model: nn.Module,
        layers: Sequence[int],
        pass_idx: int,
        hidden: torch.Tensor,
        blocks: List[torch.Tensor],
        messages: Optional[Tuple[List[torch.Tensor], torch.Tensor]],
        ctx: VisitContext,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        blocks = list(blocks)
        for layer in layers:
            key = VisitKey(
                layer=layer,
                pass_idx=pass_idx,
                position=self._position[(layer, pass_idx)],
                n_blocks=len(blocks),
                n_msg_blocks=None if messages is None else len(messages[0]),
                first_appends_input=bool(model.layers[layer].is_block_boundary),
                ctx=ctx,
            )
            tensors = [hidden, *blocks]
            if messages is not None:
                tensors += [*messages[0], messages[1]]
            outs = run_visit(self, key, tensors)
            self.visits += 1
            if key.first_appends_input:
                blocks.append(hidden)
            blocks.extend(outs[1:])
            hidden = outs[0]
        return hidden, blocks

    # -- VisitHost ----------------------------------------------------------------------------

    def _block_at(self, position: int) -> Optional[int]:
        if not 0 <= position < len(self.schedule):
            return None
        return self._layer_block[self.schedule[position][0]]

    def load_visit(self, key: VisitKey, device: torch.device) -> None:
        if self.block_manager is not None:
            self.loads += 1
            self.block_manager.load_block_for_inference(self._layer_block[key.layer], device=device)

    def release_visit(self, key: VisitKey, backward: bool) -> None:
        if self.block_manager is None:
            return
        block = self._layer_block[key.layer]
        following = key.position + (-1 if backward else 1)
        if self._block_at(following) != block:
            self.block_manager.release_block(block)

    def run_visit(self, key: VisitKey, tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        hidden = tensors[0]
        blocks = list(tensors[1 : 1 + key.n_blocks])
        messages = None
        if key.n_msg_blocks is not None:
            rest = tensors[1 + key.n_blocks :]
            messages = (list(rest[: key.n_msg_blocks]), rest[key.n_msg_blocks])
        hidden, appended = self.model.visit(
            (key.layer,), key.pass_idx, hidden, blocks, messages, key.ctx
        )
        if key.first_appends_input:
            appended = appended[1:]  # the visit's own input: its caller appends it
        return (hidden, *appended)

    def visit_params(self, key: VisitKey) -> List[nn.Parameter]:
        cache = (key.layer, key.n_msg_blocks is not None)
        params = self._params.get(cache)
        if params is None:
            model = self.model
            params = [p for p in model.layers[key.layer].parameters() if p.requires_grad]
            if key.n_msg_blocks is not None:
                params += [p for p in model.message_input.parameters() if p.requires_grad]
            self._params[cache] = params
        return params


def install_visits(model: nn.Module, block_manager: Any = None) -> VisitRunner:
    """Run ``model``'s decoder (an ``Inkling2LoopForCausalLM``) as visits, its weights resident
    unless ``block_manager`` streams them."""
    text_model = model.get_decoder()
    if text_model.poros_visit_runner is not None:
        raise RuntimeError("this model's decoder already runs as visits")
    runner = VisitRunner(text_model, block_manager)
    text_model.poros_visit_runner = runner
    return runner


def install_visit_streaming(
    model: nn.Module, block_size: int = 1, managed_filter: Any = None
) -> VisitRunner:
    """Stream ``model``'s decoder (adapters already attached): its managed weights leave the
    device until a visit loads them. ``managed_filter``: which parameters stream
    (poros.moe.prepare.dense_managed_filter: the attention weights)."""
    from poros.patch.hf import (
        _block_manager,
        _move_resident_to_compute_device,
        _resident_compute_device,
    )
    from poros.patch.hf.forward import _move_trainable_params_to_model_device

    text_model = model.get_decoder()
    if text_model.poros_visit_runner is not None and text_model.poros_visit_runner.block_manager:
        raise RuntimeError("this model's visits already stream")
    text_model.poros_visit_runner = None
    adapters = any(p.requires_grad for p in model.parameters())
    manager = _block_manager(model, max(1, int(block_size)), adapters, managed_filter)
    manager.offload_all_to_cpu()
    _move_resident_to_compute_device(model, manager, _resident_compute_device(model))
    _move_trainable_params_to_model_device(model)
    model._poros_block_manager = manager
    return install_visits(model, manager)
