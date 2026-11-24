"""Poros: fine-tune large models on small GPUs.

Streams the frozen quantized base through the GPU so only the adapters
stay resident.

All ML-dependent symbols are lazily imported. ``import poros`` succeeds
without torch installed (a light install: ``pip install --no-deps poros-train``). Accessing
ML symbols without torch raises a clear ``ImportError`` with install instructions.
"""

# The first import, before any other: nothing above it, not even a __future__, which is looked for
# on the path it cleans.
from poros import _path_rule  # noqa: F401

__version__ = "1.0.0"

import os as _os
import sys as _sys

# torch 2.14 routes some aten ops (bmm, ...) through Python "native" routers that may
# dispatch to new Triton kernels: per-call Python on every expert/LoRA bmm. Poros keeps
# torch's cuBLAS kernels instead (Qwen3-30B-A3B block-major update 18.4 -> 16.7 s;
# bitwise sequential == block-major either way). Read once when torch
# is imported, so it only applies if torch is not imported yet; POROS_KEEP_TORCH_NATIVE=1
# leaves torch's choice alone. The Mac path keeps torch's choice: it is a CUDA kernel
# choice, and the Mac's receipts ran without it.
if (
    "torch" not in _sys.modules
    and _sys.platform != "darwin"
    and _os.environ.get("POROS_KEEP_TORCH_NATIVE") != "1"
):
    _os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

# -- Pure-python exports (always available) --
from poros.bench.artifacts import RunArtifact
from poros.bench.specs import BenchmarkSpec
from poros.core.arch_registry import (
    architectures,
    register_architecture,
    runnable_arch_types,
    user_asserted_arch_types,
    validated_arch_types,
)
from poros.core.config import PorosConfig
from poros.core.env import quiet_libraries as _quiet_libraries
from poros.core.errors import (
    PorosArchNotValidatedError,
    PorosCancelledError,
    PorosConfigError,
    PorosDataError,
    PorosDatasetNotFoundError,
    PorosDiskError,
    PorosError,
    PorosExportError,
    PorosGpuUnsupportedError,
    PorosGuaranteeUnavailableError,
    PorosHostRamError,
    PorosOOMError,
    PorosStackUnsupportedError,
    PorosStepCheckError,
    PorosTorchBuildError,
    PorosVramError,
)

# -- ML-dependent exports (guarded) --
# These are wrapped so ``import poros`` never fails, but accessing
# the ML symbols without torch raises a clear ImportError.


# Two library notices a Poros user can do nothing about stay off the screen, for a script as
# for the CLI (each by its own words, on its own logger: poros.core.env.quiet_libraries).
_quiet_libraries()


def __getattr__(name: str):
    """Lazy import for ML-dependent symbols."""
    _ML_ATTRS = {
        "Run": ("poros.train.run", "Run"),
        "Datum": ("poros.train.batch", "Datum"),
        "Batch": ("poros.train.batch", "Batch"),
        "Loss": ("poros.train.batch", "Loss"),
        "PorosTrainer": ("poros.train.trainer", "PorosTrainer"),
        "PorosModel": ("poros.train.model", "PorosModel"),
        "enforce_determinism": ("poros.core.determinism", "enforce_determinism"),
    }

    if name in ("data", "losses", "recipes"):  # torch-free until a step runs
        import importlib

        return importlib.import_module(f"poros.{name}")

    if name in _ML_ATTRS:
        module_path, attr_name = _ML_ATTRS[name]
        try:
            import importlib

            mod = importlib.import_module(module_path)
            return getattr(mod, attr_name)
        except ImportError as e:
            raise ImportError(
                f"poros.{name} requires ML dependencies. "
                "Install them with: pip install poros-train\n"
                f"Original error: {e}"
            ) from e

    raise AttributeError(f"module 'poros' has no attribute {name!r}")


# Public API listing for documentation and IDE support
__all__ = [
    "__version__",
    # Pure-python
    "PorosConfig",
    "PorosError",
    "PorosArchNotValidatedError",
    "PorosOOMError",
    "PorosVramError",
    "PorosHostRamError",
    "PorosConfigError",
    "PorosGpuUnsupportedError",
    "PorosTorchBuildError",
    "PorosGuaranteeUnavailableError",
    # every error a run or an export raises, to catch by name (one per E_ code)
    "PorosStepCheckError",
    "PorosDiskError",
    "PorosExportError",
    "PorosCancelledError",
    "PorosDataError",
    "PorosDatasetNotFoundError",
    "PorosStackUnsupportedError",
    "BenchmarkSpec",
    "RunArtifact",
    # Extension API (pure-python: register new families)
    "register_architecture",
    "architectures",
    "runnable_arch_types",
    "user_asserted_arch_types",
    "validated_arch_types",
    # ML-dependent (lazy)
    "Run",
    "Datum",
    "Batch",
    "Loss",
    "PorosTrainer",
    "PorosModel",
    "enforce_determinism",
]
