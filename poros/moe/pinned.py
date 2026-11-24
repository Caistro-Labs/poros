"""Page-locking a host tensor for an asynchronous host-to-device copy, best effort."""

from __future__ import annotations

import torch

__all__ = ["try_pin"]


def try_pin(host: torch.Tensor) -> torch.Tensor:
    """``host`` page-locked for an asynchronous copy to the device, or ``host`` itself where the
    machine refuses to pin (a pin ceiling, no CUDA host allocator).

    Either source is safe for ``.to(device, non_blocking=True)``: the caching host allocator keeps
    a page-locked block until its copy ran, and a copy from pageable memory has left the host
    tensor by the time the call returns. Unpinned only costs time (the call waits for the copy).
    Mirrors the dense loader's best-effort pin (``load_nf4_block_streamed`` in
    ``poros/train/streamed_load.py``).
    """
    try:
        return host.pin_memory()
    except RuntimeError:
        return host
