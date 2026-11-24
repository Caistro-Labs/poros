"""DeviceSampler: GPU VRAM samples and the card's facts for a benchmark run.

All torch access is deferred behind try/except so this module can be
imported and instantiated without torch installed (the sampler simply
records zero-valued samples on CPU-only machines).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List


def _torch_available() -> bool:
    """Check if torch + CUDA are usable without importing at module level."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class DeviceSampler:
    """Samples GPU VRAM and utilization during benchmark runs.

    On CPU-only machines the sampler is a no-op; all readings return 0.

    Args:
        device_index: CUDA device index (default 0).
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._samples: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------

    def sample(self) -> Dict[str, float]:
        """Take a single VRAM sample.

        Returns:
            Dict with ``allocated_mb``, ``reserved_mb``, and
            ``utilization_pct`` (0.0 on CPU).
        """
        if not _torch_available():
            entry: Dict[str, float] = {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "utilization_pct": 0.0,
                "timestamp": time.time(),
            }
            self._samples.append(entry)
            return entry

        import torch

        allocated = torch.cuda.memory_allocated(self._device_index) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self._device_index) / (1024 * 1024)
        # torch doesn't expose SM utilization directly; 0.0 as placeholder
        entry = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "utilization_pct": 0.0,
            "timestamp": time.time(),
        }
        self._samples.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def samples(self) -> List[Dict[str, float]]:
        """All collected samples."""
        return list(self._samples)

    def clear(self) -> None:
        """Discard all collected samples."""
        self._samples.clear()

    # ------------------------------------------------------------------
    # Hardware info (static, best-effort)
    # ------------------------------------------------------------------

    @staticmethod
    def gpu_info(device_index: int = 0) -> Dict[str, Any]:
        """Return GPU hardware info as a dict.

        Fields: ``gpu_name``, ``vram_total_mb``, ``cuda_version``,
        ``torch_version``.  Returns placeholder values on CPU.
        """
        info: Dict[str, Any] = {
            "gpu_name": "cpu",
            "vram_total_mb": 0.0,
            "cuda_version": None,
            "torch_version": None,
            "gpu_uuid": None,
        }
        if not _torch_available():
            return info
        import torch

        props = torch.cuda.get_device_properties(device_index)
        info["gpu_name"] = props.name
        info["vram_total_mb"] = props.total_memory / (1024 * 1024)
        info["cuda_version"] = torch.version.cuda
        info["torch_version"] = torch.__version__
        # GPU UUID distinguishes two identical GPU models in one box so a
        # public per-device leaderboard keys them apart (PyTorch 2.x exposes it).
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            info["gpu_uuid"] = str(uuid)
        return info
