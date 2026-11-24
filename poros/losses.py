"""Losses for ``Run.forward_backward``.

A loss is a function ``(logprobs, batch, **args) -> Loss``. ``logprobs`` is
``[B, T]``: log p(target_token) at each position, with gradients, 0 on
padding. It never sees the ``[B, T, vocab]`` logits, which keeps a custom
loss memory-safe and in the same shape as Tinker's.

    run.forward_backward(batch)                               # cross_entropy
    run.forward_backward(pairs, loss="dpo", beta=0.1, ref=ref)
    run.forward_backward(batch, loss=my_loss)                 # any callable

Built-in names: ``cross_entropy``; for pairs ``dpo``, ``ipo``, ``kto`` (against a reference),
``orpo`` and ``simpo`` (none needed); ``importance_sampling``, ``ppo``. ``register`` adds more.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

from poros.train.batch import Batch, Loss

LossFn = Callable[..., Loss]
_LOSSES: Dict[str, LossFn] = {}


def register(name: str) -> Callable[[LossFn], LossFn]:
    """Decorator: make a loss callable by name."""

    def add(fn: LossFn) -> LossFn:
        _LOSSES[name] = fn
        return fn

    return add


def get(loss: Union[str, LossFn]) -> LossFn:
    if callable(loss):
        return loss
    try:
        return _LOSSES[loss]
    except KeyError:
        raise ValueError(
            f"unknown loss {loss!r}; built in: {', '.join(sorted(_LOSSES))}"
        ) from None


def names() -> list:
    return sorted(_LOSSES)


def _need(t: Dict[str, Any], name: str, loss: str) -> Any:
    if name not in t:
        raise ValueError(f"{loss} needs Datum.{name} on every row of the batch")
    return t[name]


@register("cross_entropy")
def cross_entropy(lp: Any, batch: Batch) -> Loss:
    """Mean negative log-likelihood over the weighted positions.

    Without weights every real position counts (plain text). With weights,
    0 on the prompt and 1 on the answer trains on the answers only.
    """
    t = batch.tensors(lp.device)
    w = t["weights"] * t["mask"] if "weights" in t else t["mask"]
    n = w.sum().clamp(min=1.0)
    nll = -(lp * w).sum() / n
    return Loss(nll, {"tokens": float(w.sum())})


def sequence_logprobs(lp: Any, batch: Batch) -> Any:
    """``[B]``: each row's summed logprob over its weighted positions."""
    t = batch.tensors(lp.device)
    w = t["weights"] * t["mask"] if "weights" in t else t["mask"]
    return (lp * w).sum(-1)


@register("dpo")
def dpo(lp: Any, batch: Batch, *, ref: Any, beta: float = 0.1) -> Loss:
    """Direct preference optimisation over a pairs batch.

    ``ref`` is ``[N, 2]``: the reference model's sequence logprobs for each
    pair's (chosen, rejected), indexed by ``batch.ids``. ``Run.reference``
    computes it once, before training.
    """
    import torch.nn.functional as F

    if not batch.pairs or batch.ids is None:
        raise ValueError("dpo needs a pairs batch with ids (poros.data.pairs)")
    seq = sequence_logprobs(lp, batch)
    half = len(batch) // 2
    ids = batch.ids[:half]
    ref = ref.to(seq.device)[ids]
    chosen = beta * (seq[:half] - ref[:, 0])
    rejected = beta * (seq[half:] - ref[:, 1])
    margin = chosen - rejected
    loss = -F.logsigmoid(margin).mean()
    m, c, r = margin.detach(), chosen.detach(), rejected.detach()
    return Loss(loss, {
        "accuracy": float((m > 0).float().mean()),
        "margin": float(m.mean()),
        "chosen_reward": float(c.mean()),
        "rejected_reward": float(r.mean()),
    })


# The pair losses below share dpo's batch: chosen rows first, rejected after, ``ids`` naming each
# pair. ``REFERENCE`` names the ones that need ``ref`` (``Run.reference``).
REFERENCE = frozenset({"dpo", "ipo", "kto"})
PAIRS = ("dpo", "ipo", "kto", "orpo", "simpo")


def _pairs(lp: Any, batch: Batch, loss: str) -> tuple:
    """(chosen, rejected) summed logprobs and their weighted token counts, each ``[N]``."""
    if not batch.pairs or batch.ids is None:
        raise ValueError(f"{loss} needs a pairs batch with ids (poros.data.pairs)")
    t = batch.tensors(lp.device)
    w = t["weights"] * t["mask"] if "weights" in t else t["mask"]
    seq, n = (lp * w).sum(-1), w.sum(-1).clamp(min=1.0)
    half = len(batch) // 2
    return seq[:half], seq[half:], n[:half], n[half:]


def _logratios(lp: Any, batch: Batch, ref: Any, loss: str) -> tuple:
    """Each pair's (chosen, rejected) logprob over the reference's."""
    c, r, _, _ = _pairs(lp, batch, loss)
    ref = ref.to(c.device)[batch.ids[: len(batch) // 2]]
    return c - ref[:, 0], r - ref[:, 1]


def _said(margin: Any, **more: Any) -> Dict[str, float]:
    m = margin.detach()
    return {"accuracy": float((m > 0).float().mean()), "margin": float(m.mean()),
            **{k: float(v.detach().mean()) for k, v in more.items()}}


@register("ipo")
def ipo(lp: Any, batch: Batch, *, ref: Any, beta: float = 0.1) -> Loss:
    """Identity preference optimisation (Azar et al., 2023): the log-ratio margin pulled to
    ``1 / (2 beta)`` by a square, so it cannot run off as DPO's can on clean pairs."""
    c, r = _logratios(lp, batch, ref, "ipo")
    margin = c - r
    loss = ((margin - 1.0 / (2.0 * beta)) ** 2).mean()
    return Loss(loss, _said(margin, chosen_reward=beta * c, rejected_reward=beta * r))


@register("kto")
def kto(lp: Any, batch: Batch, *, ref: Any, beta: float = 0.1, desirable: float = 1.0,
        undesirable: float = 1.0) -> Loss:
    """Kahneman-Tversky optimisation (Ethayarajh et al., 2024) on pairs taken apart: each chosen
    answer is a good one and each rejected a bad one, judged against the batch's own KL to the
    reference (held, not trained through), each side weighted."""
    import torch

    c, r = _logratios(lp, batch, ref, "kto")
    z0 = torch.cat([c, r]).mean().detach().clamp(min=0.0)
    good = 1.0 - torch.sigmoid(beta * (c - z0))
    bad = 1.0 - torch.sigmoid(beta * (z0 - r))
    loss = (desirable * good.sum() + undesirable * bad.sum()) / (2 * len(c))
    return Loss(loss, {**_said(c - r, chosen_reward=beta * c, rejected_reward=beta * r),
                       "kl": float(z0)})


@register("orpo")
def orpo(lp: Any, batch: Batch, *, beta: float = 0.1) -> Loss:
    """Odds-ratio preference optimisation (Hong et al., 2024): SFT on the chosen answers plus
    ``beta`` times the log odds that chosen beats rejected, in one pass and with no reference,
    so it trains straight from the base."""
    import torch
    import torch.nn.functional as F

    c, r, nc, nr = _pairs(lp, batch, "orpo")
    avg_c, avg_r = c / nc, r / nr  # mean logprob a token: below 0

    def log_odds(x: Any) -> Any:
        return x - torch.log1p(-torch.exp(x).clamp(max=1 - 1e-6))

    ratio = log_odds(avg_c) - log_odds(avg_r)
    nll = -avg_c.mean()
    loss = nll - beta * F.logsigmoid(ratio).mean()
    return Loss(loss, {**_said(ratio), "nll": float(nll.detach())})


@register("simpo")
def simpo(lp: Any, batch: Batch, *, beta: float = 2.0, gamma: float = 1.0) -> Loss:
    """Simple preference optimisation (Meng et al., 2024): DPO's sigmoid on each answer's mean
    logprob a token, a ``gamma`` margin asked for; no reference, so no reference pass."""
    import torch.nn.functional as F

    c, r, nc, nr = _pairs(lp, batch, "simpo")
    gap = beta * (c / nc - r / nr)  # what accuracy reads: chosen over rejected, before gamma
    loss = -F.logsigmoid(gap - gamma).mean()
    return Loss(loss, _said(gap, chosen_reward=beta * c / nc, rejected_reward=beta * r / nr))


def _ratio(lp: Any, t: Dict[str, Any], loss: str) -> Any:
    import torch

    old = _need(t, "logprobs", loss)
    return torch.exp(lp - old) * t["mask"]


@register("importance_sampling")
def importance_sampling(lp: Any, batch: Batch) -> Loss:
    """Policy gradient corrected for the sampler: -sum(p/q * advantage),
    summed over positions and averaged over rows."""
    t = batch.tensors(lp.device)
    ratio = _ratio(lp, t, "importance_sampling")
    adv = _need(t, "advantages", "importance_sampling")
    loss = -(ratio * adv).sum() / len(batch)
    return Loss(loss, {"ratio": float(ratio.detach().sum() / t["mask"].sum())})


@register("ppo")
def ppo(lp: Any, batch: Batch, *, clip: float = 0.2) -> Loss:
    """Clipped surrogate: the ratio is held to [1 - clip, 1 + clip]."""
    import torch

    t = batch.tensors(lp.device)
    ratio = _ratio(lp, t, "ppo")
    adv = _need(t, "advantages", "ppo")
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    surrogate = torch.minimum(ratio * adv, clipped * adv) * t["mask"]
    clip_frac = ((ratio.detach() - 1.0).abs() > clip).float() * t["mask"]
    return Loss(-surrogate.sum() / len(batch), {
        "clip_frac": float(clip_frac.sum() / t["mask"].sum()),
    })
