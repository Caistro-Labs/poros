"""What ``patch_hf_model`` fixed for a patched model (``_PatchConfig``), read by the forward it
installs (``poros.patch.hf.forward``) and by ``poros.core.blockwise`` as ``trainer.config``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _PatchConfig:
    """Minimal config matching the attributes that blockwise.py reads
    from ``trainer.config``."""

    preserve_rng_state: bool = True
    offload_suffix_inputs_to_cpu: bool = False
