"""Kahneman-Tversky optimisation (Ethayarajh et al., 2024) over LoRA, on pairs taken apart: each
chosen answer a good one, each rejected a bad one, against the batch's KL to the reference.
Steadier than DPO where the pairs are noisy. The reference is worked out once (see dpo.py)."""

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
    """``eval_data``, held-out pairs, is scored with the same loss against the same reference, and
    how many pairs' chosen answer scores over the rejected one."""
    ref = run.reference(data)
    held = run.reference(eval_data) if eval_data is not None else None
    n = steps or epochs * len(data)
    if eval_data is not None and run.step == 0:  # the model it starts from
        run.evaluate(eval_data, loss="kto", beta=beta, ref=held)
    for step in run.steps(n, stop=stop):
        run.forward_backward(data.batch(step), loss="kto", beta=beta, ref=ref)
        run.optim_step()
        if eval_data is not None and run.evals_at(step):
            run.evaluate(eval_data, loss="kto", beta=beta, ref=held)
    if eval_data is not None:
        run.evaluate(eval_data, loss="kto", beta=beta, ref=held)
    run.save()
    return run.metrics
