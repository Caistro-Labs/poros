"""The byte-source interface under both streaming seams, and its RAM backend.

A ``SlabKey`` is ``(block_id, 0)`` on the dense path and
``(moe_layer, expert)`` on the MoE path. A source only moves bytes: it never
interprets them, so swapping one source for another cannot change numerics
as long as both hold the same bytes for every key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

SlabKey = tuple[int, int]

__all__ = ["RamSource", "SlabKey", "SlabSource", "check_dst"]


def check_dst(key: SlabKey, dst: torch.Tensor, nbytes: int) -> None:
    """Raise unless ``dst`` is a contiguous uint8 buffer of exactly ``nbytes`` bytes."""
    if dst.dtype != torch.uint8 or dst.numel() != nbytes or not dst.is_contiguous():
        raise ValueError(
            f"dst for slab {key} must be a contiguous uint8 tensor of {nbytes} bytes, "
            f"got {dst.dtype} with {dst.numel()} elements (contiguous={dst.is_contiguous()})"
        )


@runtime_checkable
class SlabSource(Protocol):
    """Where slab bytes come from: host RAM or a ``.sss`` file."""

    def keys(self) -> Sequence[SlabKey]:
        """Every key this source holds, in a fixed order."""
        ...

    def nbytes(self, key: SlabKey) -> int:
        """Byte size of ``key``'s slab."""
        ...

    def read_into(self, key: SlabKey, dst: torch.Tensor) -> None:
        """Copy ``key``'s bytes into ``dst`` (uint8, CPU or CUDA) on the current stream.

        A pure byte copy. On CUDA it may be asynchronous with respect to the
        host, exactly like ``dst.copy_(pinned_slab, non_blocking=True)``.
        """
        ...

    def prefetch(self, keys: Sequence[SlabKey]) -> None:
        """Hint that ``keys`` will be read soon, in this order. May be a no-op."""
        ...

    def release(self, key: SlabKey) -> None:
        """Hint that ``key`` will not be read again soon. May be a no-op."""
        ...


class RamSource:
    """Slabs held in host memory (pinned where the platform allows): today's behaviour.

    ``read_into`` is ``dst.copy_(slab, non_blocking=True)``, the same call
    ``MoeExpertStore.load_slice`` and the dense packed-block load make today.
    The slabs are borrowed, not copied; callers keep them immutable.
    """

    def __init__(self, slabs: Mapping[SlabKey, torch.Tensor]) -> None:
        self._slabs: dict[SlabKey, torch.Tensor] = {}
        for key, slab in slabs.items():
            if slab.dtype != torch.uint8 or slab.dim() != 1 or not slab.is_contiguous():
                raise ValueError(
                    f"slab {key} must be a contiguous 1-D uint8 tensor, "
                    f"got {slab.dtype} with shape {tuple(slab.shape)}"
                )
            if slab.device.type != "cpu":
                raise ValueError(f"slab {key} must live in host memory, got {slab.device}")
            self._slabs[tuple(key)] = slab
        self._keys = tuple(self._slabs)

    @classmethod
    def from_sss(cls, path, *, pin: bool = True, verify: bool = True) -> RamSource:
        """Fill RAM slabs from a ``.sss`` file (the startup cache), checksum-verified.

        The slabs are carved from ~1 GiB pinned chunks in the file's order, as a first load's
        ``RamSink`` carves them: a layer's experts lie back to back in pinned memory, so the
        expert residency moves a layer in one copy (pinned one by one, each slab took its own
        power-of-two block and the residency fell back to a copy per expert)."""
        from poros.sss.format import RamSink, SssFile

        with SssFile.open(path) as f:
            entries = list(f.entries)
            sink = RamSink([(entry.key, entry.nbytes) for entry in entries], pin=pin)
            for entry in entries:
                f.read_into(entry.key, sink.row(entry.key), verify=verify)
        return cls(sink.rows())

    def keys(self) -> Sequence[SlabKey]:
        return self._keys

    def nbytes(self, key: SlabKey) -> int:
        return self._slab(key).numel()

    def slab(self, key: SlabKey) -> torch.Tensor:
        """The host slab itself (read-only by contract), for writers and tests."""
        return self._slab(key)

    def read_into(self, key: SlabKey, dst: torch.Tensor) -> None:
        slab = self._slab(key)
        check_dst(key, dst, slab.numel())
        dst.copy_(slab, non_blocking=True)

    def prefetch(self, keys: Sequence[SlabKey]) -> None:  # already resident
        return None

    def release(self, key: SlabKey) -> None:  # nothing to free
        return None

    def items(self) -> Iterable[tuple[SlabKey, torch.Tensor]]:
        return self._slabs.items()

    def _slab(self, key: SlabKey) -> torch.Tensor:
        try:
            return self._slabs[tuple(key)]
        except KeyError:
            raise KeyError(f"slab {key} is not in this source") from None
