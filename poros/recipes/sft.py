"""Supervised fine-tuning: cross-entropy on the weighted tokens.

With ``poros.data.chats`` that is the answers only; with ``poros.data.text``
every token.
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
    eval_data: Optional["poros.data.Data"] = None,
    ends: Any = None,
    stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """``eval_data`` is scored before the first step, as ``run.evals_at`` says (every
    ``eval_steps``, and at each checkpoint) and after the last; ``ends`` (``poros.ends.Ends``)
    reads in the same passes whether the held-out chats' answers end."""
    n = steps or epochs * len(data)
    if eval_data is not None and run.step == 0:  # the model it starts from
        run.evaluate(eval_data, ends=ends)
    for step in run.steps(n, stop=stop):
        run.forward_backward(data.batch(step), loss="cross_entropy")
        run.optim_step()
        if eval_data is not None and run.evals_at(step):
            run.evaluate(eval_data, ends=ends)
    if eval_data is not None:
        run.evaluate(eval_data, ends=ends)
    run.save()
    return run.metrics
