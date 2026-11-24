"""``UnifiedBlockManager``: one pool, so a block is one layer, and loading it moves a pointer."""

from __future__ import annotations

from poros.core.residency import BlockManager
from poros.metal.nf4 import Nf4Linear


class UnifiedBlockManager(BlockManager):
    """Points a layer's linears at bytes ``model.poros_source`` holds, and reads the next ahead.

    The read order is known, so it is written down: a microstep goes up the layers and back down.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.source = model.poros_source
        self._linears = [
            [m for m in layer.modules() if isinstance(m, Nf4Linear)]
            for _, layer in self.transformer_layers
        ]
        self._order = [*range(self.num_layers), *reversed(range(self.num_layers))]
        self._at = -1

    def offload_all_to_cpu(self) -> None:
        pass

    def _next_read(self, block_id: int) -> int:
        self._at = (self._at + 1) % len(self._order)
        if self._order[self._at] != block_id:  # off the schedule, as an evaluation is: back on it
            self._at = block_id
        return self._order[(self._at + 1) % len(self._order)]

    def load_block_for_inference(self, block_id, device=None, stream=None) -> None:
        slab = self.source.acquire(block_id)
        for linear in self._linears[block_id]:
            linear.slab = slab
        self.source.prefetch(self._next_read(block_id))

    def release_block(self, block_id: int) -> None:
        for linear in self._linears[block_id]:
            linear.slab = None
