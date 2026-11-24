"""Frozen contracts for the quarantined MoE training path (``poros.moe``).

Single source of truth shared by workstream A (residency), workstream B
(quant/modules/adapters), and workstream C (integration/gates). Types and
protocols only — no logic.

FROZEN: do not edit. If a contract cannot be implemented as written, stop
and report the conflict instead of editing (README §2).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

# An expert is addressed by (layer_index, expert_index), both zero-based.
ExpertKey = tuple[int, int]

# Callback receiving each committed RouteDecision (gate traces, debugging).
RouteObserver = Callable[["RouteDecision"], None]


@dataclass(frozen=True)
class RouteDecision:
    """A router's committed output for one forward pass at one MoE layer.

    ``expert_ids`` are in the reference implementation's exact selection
    order; ``expert_weights`` are the exact combining weights it applies.
    A committed decision must be computed in full: every selected expert,
    exact weights, no dropping, no substitution (invariant 3).
    """

    layer: int
    expert_ids: tuple[int, ...]
    expert_weights: tuple[float, ...]


@dataclass(frozen=True)
class ExpertLayout:
    """Static shape facts about a model's routed experts."""

    num_layers: int  # MoE layers carrying routed experts
    experts_per_layer: int
    top_k: int
    tensor_names: tuple[str, ...]  # logical per-expert tensors, e.g. ("gate_up_proj", "down_proj")


class ExpertStore(Protocol):
    """Owns expert weight bytes and exact dequantization (workstream B).

    Holds every expert's quantized weights as one contiguous flat uint8 slab
    per (layer, expert) in host memory. Immutable after construction; the
    slabs are the single source of truth, so a slice fetched twice is
    bit-identical both times (backward rematerialization relies on this).
    """

    @property
    def layout(self) -> ExpertLayout: ...

    def slice_nbytes(self, key: ExpertKey) -> int:
        """Quantized bytes of one expert's flat slab (all tensors included)."""

    def load_slice(self, key: ExpertKey, dst: torch.Tensor) -> None:
        """Copy one expert's flat slab into ``dst`` (uint8, ``slice_nbytes`` long).

        Pure byte copy, never requantizes. Copies on the current stream when
        ``dst`` lives on a device; the caller controls stream context.
        """

    def dequantize_tensor(
        self, key: ExpertKey, tensor_name: str, slab: torch.Tensor
    ) -> torch.Tensor:
        """Exactly dequantize one tensor of one expert out of its flat slab.

        ``slab`` was previously filled by ``load_slice`` for the same key;
        the store alone knows its internal offsets. The result must be
        bit-identical to the repo's NF4 dequantization of the same logical
        weights (README §5).
        """


class ExpertResidency(Protocol):
    """Owns expert residency and transfer scheduling (workstream A).

    Training schedule is router-independent: per layer, slices stream in
    static ascending order with a look-ahead window (README §1). With
    ``streamed=False`` everything is ensured up front and never evicted —
    the parity-reference configuration.
    """

    def ensure(self, keys: Sequence[ExpertKey]) -> None:
        """Block until every key is resident on the compute device.

        May evict slices that are no longer needed (LRU across layers).
        Never drops, substitutes, or approximates a requested key; a
        re-ensured key is bit-identical to its previous instance.
        """

    def resident_qbytes(self, key: ExpertKey) -> torch.Tensor:
        """The resident flat slab for ``key`` on the compute device.

        Raises ``KeyError`` if absent — that is a programming error; call
        ``ensure`` first. There is no silent fallback.
        """

    @property
    def bytes_transferred(self) -> int:
        """Total host→device slab bytes moved since construction, including
        backward rematerializations (G-BYTES accounting)."""
