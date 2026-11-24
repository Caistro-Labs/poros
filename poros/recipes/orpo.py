"""Odds-ratio preference optimisation (Hong et al., 2024) over LoRA: SFT on the chosen answers
and the preference in one loss, with no reference, so it trains straight from a base model
with no SFT first. ``beta`` weighs the preference against the SFT term."""

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
    """``eval_data``, held-out pairs, is scored with the same loss, and how many pairs' chosen
    answer scores over the rejected one. No reference: nothing is worked out before step one."""
    n = steps or epochs * len(data)
    if eval_data is not None and run.step == 0:  # the model it starts from
        run.evaluate(eval_data, loss="orpo", beta=beta)
    for step in run.steps(n, stop=stop):
        run.forward_backward(data.batch(step), loss="orpo", beta=beta)
        run.optim_step()
        if eval_data is not None and run.evals_at(step):
            run.evaluate(eval_data, loss="orpo", beta=beta)
    if eval_data is not None:
        run.evaluate(eval_data, loss="orpo", beta=beta)
    run.save()
    return run.metrics
