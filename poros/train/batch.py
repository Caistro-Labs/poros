"""What a training step eats and returns: ``Datum``, ``Batch``, ``Loss``, ``Output``.

A ``Datum`` is one sequence, already shifted: position ``i`` reads
``tokens[i]`` and is scored on ``target_tokens[i]``. The loss inputs ride
along by name, the same names Tinker uses, so a loop written for one ports
to the other mostly by renames:

    cross_entropy          weights (0 on the prompt, 1 on the answer)
    importance_sampling    logprobs (the sampler's), advantages
    ppo                    logprobs, advantages

A ``Batch`` pads its datums to the longest one (never to the run's full
length) and builds the tensors on first use, so building batches stays
cheap and torch-free until a step needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The per-position loss inputs a Datum may carry, padded with 0.0.
FIELDS = ("weights", "logprobs", "advantages")


@dataclass
class Datum:
    """One shifted sequence and its per-position loss inputs."""

    tokens: List[int]
    target_tokens: List[int]
    weights: Optional[List[float]] = None
    logprobs: Optional[List[float]] = None
    advantages: Optional[List[float]] = None

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if n == 0:
            raise ValueError("a Datum needs at least one token")
        for name in ("target_tokens",) + FIELDS:
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(
                    f"Datum.{name} has {len(values)} positions, tokens has {n}"
                )

    @classmethod
    def from_sequence(
        cls, sequence: Sequence[int], weights: Optional[Sequence[float]] = None
    ) -> "Datum":
        """Shift a whole sequence; ``weights`` is per token of ``sequence``
        (the weight of token ``i`` scores the prediction of token ``i``)."""
        seq = list(sequence)
        if len(seq) < 2:
            raise ValueError("a sequence needs at least two tokens to train on")
        w = None if weights is None else [float(x) for x in weights][1:]
        return cls(tokens=seq[:-1], target_tokens=seq[1:], weights=w)

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class Batch:
    """Datums trained in one streamed pass, padded to the longest.

    ``ids`` names each row in its dataset (DPO looks reference logprobs up
    by it). ``pairs`` marks a preference batch: the first half are the
    chosen answers, the second half the rejected ones, row for row.
    ``pad_to`` is the least it pads to.
    """

    datums: List[Datum]
    ids: Optional[List[int]] = None
    pairs: bool = False
    pad_id: int = 0
    pad_to: int = 0
    _tensors: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.datums:
            raise ValueError("a Batch needs at least one Datum")
        if self.pairs and len(self.datums) % 2:
            raise ValueError("a pairs Batch holds chosen then rejected: an even count")

    def __len__(self) -> int:
        return len(self.datums)

    @property
    def length(self) -> int:
        return max(self.pad_to, *(len(d) for d in self.datums))

    @property
    def tokens(self) -> int:
        """Real (unpadded) positions in the batch."""
        return sum(len(d) for d in self.datums)

    @property
    def trained(self) -> float:
        """The positions a loss scores: each row's weights where it has them (the answers of a
        chat, a pair's two answers), else every real position. What a step trains on, counted on
        the CPU from the rows alone."""
        return float(sum(sum(d.weights) if d.weights is not None else len(d)
                         for d in self.datums))

    def split(self, size: int) -> List[Tuple["Batch", List[int]]]:
        """Micro-batches of ``size`` rows (``size`` pairs for a pairs batch),
        each with the positions its rows hold in this batch."""
        if self.pairs:
            half = len(self) // 2
            out = []
            for a in range(0, half, size):
                pos = list(range(a, min(a + size, half)))
                rows = pos + [half + p for p in pos]
                out.append((self._take(rows), rows))
            return out
        return [(self._take(list(range(a, min(a + size, len(self))))),
                 list(range(a, min(a + size, len(self)))))
                for a in range(0, len(self), size)]

    def _take(self, rows: List[int]) -> "Batch":
        ids = [self.ids[r] for r in rows] if self.ids is not None else None
        return Batch([self.datums[r] for r in rows], ids=ids, pairs=self.pairs,
                     pad_id=self.pad_id, pad_to=self.pad_to)

    def tensors(self, device: Any = None) -> Dict[str, Any]:
        """``input_ids``, ``attention_mask``, ``target``, ``mask`` and each
        loss input present, as ``[B, T]`` tensors on ``device``."""
        key = str(device)
        if key not in self._tensors:
            self._tensors[key] = self._build(device)
        return self._tensors[key]

    def _build(self, device: Any) -> Dict[str, Any]:
        import torch

        n, t = len(self.datums), self.length

        def pad(rows: List[Sequence[Any]], value: Any, dtype: Any) -> Any:
            out = torch.full((n, t), value, dtype=dtype)
            for i, row in enumerate(rows):
                out[i, : len(row)] = torch.tensor(list(row), dtype=dtype)
            return out.to(device) if device is not None else out

        ds = self.datums
        out = {
            "input_ids": pad([d.tokens for d in ds], self.pad_id, torch.long),
            "attention_mask": pad([[1] * len(d) for d in ds], 0, torch.long),
            "target": pad([d.target_tokens for d in ds], 0, torch.long),
        }
        out["mask"] = out["attention_mask"].float()
        for name in FIELDS:
            rows = [getattr(d, name) for d in ds]
            if any(r is not None for r in rows):
                out[name] = pad(
                    [r if r is not None else [0.0] * len(d) for r, d in zip(rows, ds)],
                    0.0, torch.float32,
                )
        return out


@dataclass
class Loss:
    """What a loss function returns: the scalar to backpropagate and the
    numbers worth drawing (they land in the step event and on the screen)."""

    value: Any
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Output:
    """What ``forward_backward`` / ``forward`` return."""

    loss: float
    metrics: Dict[str, float]
    logprobs: Any  # [B, T] log p(target_token), detached; 0 on padding
