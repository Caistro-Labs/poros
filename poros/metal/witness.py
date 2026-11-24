"""Every layer's forward output against its remat output, bit for bit, on every run.

Forward hooks on the decoder layers, so the block engine knows nothing of it. A forward's output is
already the next layer's saved input, so holding it costs nothing, and it is dropped at the remat.
Nothing sits on the autograd ``ctx``, where it would form a cycle only the cyclic GC frees. The
compare is encoded when the remat ends and read after the step's own sync, so it adds none.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch

from poros.core import checkpoints
from poros.core.errors import PorosStepCheckError
from poros.core.events import PorosEvent


class RematWitness:
    def __init__(self, layers, events, output_dir: str):
        self.events, self.output_dir = events, Path(output_dir)
        self.checked = self.mismatches = 0
        self._forward: dict[int, torch.Tensor] = {}
        self._flags: list[tuple[int, torch.Tensor]] = []
        self._hooks = [
            layer.register_forward_hook(partial(self._on_layer, index))
            for index, layer in enumerate(layers)
        ]

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def _on_layer(self, index: int, module, args, output: torch.Tensor) -> None:
        if not module.training:  # an evaluation has no remat
            return
        if not torch.is_grad_enabled():
            self._forward[index] = output
            return
        forward = self._forward.pop(index).view(torch.int32)
        self._flags.append((index, forward.ne(output.detach().view(torch.int32)).any()))

    def check(self, step: int, loss: float) -> None:
        """Call right after the step's loss read. Raises ``E_STEP_CHECK`` if a layer's remat
        differs from its forward. A non-finite loss or gradient norm alone is the trainer's stop
        (``PorosTrainer._stop_if_nonfinite``, the same on CUDA), which names the value and what
        to change; ``nonfinite_loss`` records it beside a mismatch."""
        differing = []
        if self._flags:
            layers, flags = zip(*self._flags)
            differing = [i for i, hit in zip(layers, torch.stack(flags).tolist()) if hit]
            self.checked += len(layers)
            self.mismatches += len(differing)
            self._flags.clear()
        if not differing:
            return
        record = {
            "mismatches": self.mismatches,
            "first_layer": min(differing, default=None),
            "step": step + 1,
            "nonfinite_loss": not math.isfinite(loss),
        }
        data = {"code": PorosStepCheckError.code, "witness": record}
        # the event's step is the update's index from 0, as a step event's; its record counts
        # the update from 1, as its message does
        self.events.emit(PorosEvent(event_type="error", step=step, data=data))
        saved = checkpoints.list_checkpoints(self.output_dir)
        kept = f"Nothing after step {saved[-1][0] + 1} was kept." if saved else "Nothing was kept."
        raise PorosStepCheckError(
            f"A check on step {step + 1} failed, so this run stopped. {kept}",
            remediation=(
                "`poros runs continue` goes on from there." if saved else "Start the run again."
            ),
            witness=record,
        )
