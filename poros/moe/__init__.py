"""poros.moe — quarantined Mixture-of-Experts training path.

NF4 QLoRA for MoE models (fused expert tensors that stock bitsandbytes
cannot quantize), with expert-granular blockwise residency: expert slices
stream during forward and rematerialize bit-exactly in backward, while the
dense engine handles everything else untouched.

Guarantee discipline: the parity claim is streamed-vs-resident bitwise
equality (0.00e+00). ``prepare_moe`` reports ``validated_bitwise`` only for
architectures the registry validates for this path
(``validated_arch_types("moe")``, backed by the family gate in
``docs/validation/int/4090/round3/``) in the configuration those receipts
cover; anything else is outside that guarantee, proven per run by ``poros.moe.gates``.

Quarantine: the MoE path changes no dense-engine behavior; the registry only
records which path an architecture is validated for.
Entry point: ``poros.moe.prepare.prepare_moe``, optionally fed by
``poros.moe.load.load_moe_streamed`` (bounded host RAM at load). See
``docs/moe.md``.

All exports are lazy (PEP 562): importing ``poros.moe`` or
``poros.moe.arch`` pulls no torch, so the arch descriptor stays usable in
dependency-free tooling.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poros.moe.contracts import (
        ExpertKey,
        ExpertLayout,
        ExpertResidency,
        ExpertStore,
        RouteDecision,
        RouteObserver,
    )
    from poros.moe.load import load_moe_streamed
    from poros.moe.prepare import (
        MoeBackends,
        MoeConfig,
        MoeReport,
        default_backends,
        prepare_moe,
    )

_LAZY = {
    "ExpertKey": "poros.moe.contracts",
    "ExpertLayout": "poros.moe.contracts",
    "ExpertResidency": "poros.moe.contracts",
    "ExpertStore": "poros.moe.contracts",
    "RouteDecision": "poros.moe.contracts",
    "RouteObserver": "poros.moe.contracts",
    "load_moe_streamed": "poros.moe.load",
    "MoeBackends": "poros.moe.prepare",
    "MoeConfig": "poros.moe.prepare",
    "MoeReport": "poros.moe.prepare",
    "default_backends": "poros.moe.prepare",
    "prepare_moe": "poros.moe.prepare",
}

__all__ = [
    "ExpertKey",
    "ExpertLayout",
    "ExpertResidency",
    "ExpertStore",
    "MoeBackends",
    "MoeConfig",
    "MoeReport",
    "RouteDecision",
    "RouteObserver",
    "default_backends",
    "load_moe_streamed",
    "prepare_moe",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
