"""Determinism protocol for Poros.

Centralizes all settings required for bitwise-reproducible training:
CUBLAS_WORKSPACE_CONFIG, deterministic algorithms, math-only SDPA.

INV-3: This is the determinism invariant. Do not weaken these settings.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def enforce_determinism(*, fill_uninitialized: bool = True) -> None:
    """Enable strict determinism for bitwise-reproducible training.

    Sets CUBLAS_WORKSPACE_CONFIG, enables deterministic algorithms (an op with no deterministic
    implementation raises), disables cuDNN benchmark mode, and restricts SDPA to math-only
    (flash and mem-efficient disabled).

    Args:
        fill_uninitialized: ``torch.utils.deterministic.fill_uninitialized_memory``.
            With deterministic algorithms on, torch fills every new ``empty``
            buffer (NaN / max int) before use; on Qwen2.5-72B NF4 that is ~36k
            bf16 fill launches (2.8 s GPU) per update, mostly whole dequantized
            weights that dequant overwrites at once. Turning it off changes no
            value a correct kernel reads; the deterministic algorithms, cuBLAS
            workspace and math-only SDPA stay on.

    Requires torch to be installed (raises ImportError otherwise).
    """
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.utils, "deterministic") and hasattr(torch.utils.deterministic,
                                                         "fill_uninitialized_memory"):
        torch.utils.deterministic.fill_uninitialized_memory = bool(fill_uninitialized)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable flash and mem-efficient SDPA (math-only)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    logger.info(
        "Determinism enforced: CUBLAS_WORKSPACE_CONFIG=:4096:8, "
        "deterministic_algorithms=True, math-only SDPA."
    )
