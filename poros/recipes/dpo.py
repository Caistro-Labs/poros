"""Direct preference optimisation (Rafailov et al., 2023) over LoRA.

The reference is the policy as it stands when DPO starts: the base model on
a fresh run, the SFT policy after SFT. ``run.reference`` works it out in one
pass over the pairs and keeps it in the run folder, so there is no second
model in memory and no extra trip through the base on each step.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import poros


def main(
    run: "poros.Run",
    data: "poros.data.Data",
    *,
    steps: Optional[int] = None,
    epochs: int = 1,
    beta: float = 0.1,
    eval_data: Optional["poros.data.Data"] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """``eval_data``, held-out pairs, is scored as SFT's held-out rows are: DPO's loss against the
    same reference, and how many pairs' chosen answer scores over the rejected one."""
    ref = run.reference(data)
    held = run.reference(eval_data) if eval_data is not None else None
    n = steps or epochs * len(data)
    if eval_data is not None and run.step == 0:  # the model it starts from
        run.evaluate(eval_data, loss="dpo", beta=beta, ref=held)
    for step in run.steps(n, stop=stop):
        # chosen and rejected share the batch: one trip through the base
        run.forward_backward(data.batch(step), loss="dpo", beta=beta, ref=ref)
        run.optim_step()
        if eval_data is not None and run.evals_at(step):
            run.evaluate(eval_data, loss="dpo", beta=beta, ref=held)
    if eval_data is not None:
        run.evaluate(eval_data, loss="dpo", beta=beta, ref=held)
    run.save()
    return run.metrics
