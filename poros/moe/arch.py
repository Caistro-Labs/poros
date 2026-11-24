"""MoE architecture descriptors and parameter classification.

Pure-python by design: this module never imports torch, so the descriptor
stays usable in dependency-free tooling. ``ExpertLayout``/``ExpertKey`` come
from the contracts and are imported lazily where values are constructed.

The descriptor answers the two questions ``prepare_moe`` asks:

* static shape facts about the routed experts (``contracts.ExpertLayout``),
* the residency class of every base parameter: ``"expert"`` (fused 3-D
  expert tensors, owned by the ExpertStore), ``"dense"`` (streamed by the
  existing blockwise engine), ``"pinned"`` (resident forever: embeddings,
  norms, router, LM head).

Mirrors the transformers 5.x sparse-MoE layout that ``MOE_FAMILIES``
share (``models/qwen3_moe/modeling_qwen3_moe.py``): on MoE layers
``model.layers.{i}.mlp`` is a sparse block (router ``mlp.gate`` + fused
``mlp.experts.gate_up_proj/down_proj``); on ``mlp_only_layers`` /
non-sparse steps it is a dense MLP.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Literal, Optional

from poros.core.arch_registry import registry_type
from poros.core.errors import PorosStackUnsupportedError

if TYPE_CHECKING:
    from torch import nn

    from poros.moe.contracts import ExpertKey, ExpertLayout

logger = logging.getLogger(__name__)

#: Residency classes assigned by ``MoeArch.classify_parameter``.
ParamClass = Literal["expert", "dense", "pinned"]

#: Logical per-expert tensors of the fused Qwen3-MoE expert block, in slab
#: order (modeling_qwen3_moe.py:223-224).
EXPERT_TENSOR_NAMES: tuple[str, str] = ("gate_up_proj", "down_proj")

#: The families whose sparse block (``mlp`` on MoE layers) is the qwen3_moe
#: one: a top-k router ``gate`` plus fused gated-MLP ``experts``, optionally
#: with dense shared experts (qwen2_moe's sigmoid-gated ``shared_expert``,
#: deepseek_v2's ``shared_experts``). Each maps to the config field holding
#: the per-expert intermediate size.
MOE_FAMILIES: dict[str, str] = {
    "qwen3_moe": "moe_intermediate_size",
    "qwen2_moe": "moe_intermediate_size",
    "olmoe": "intermediate_size",
    "mixtral": "intermediate_size",
    "deepseek_v2": "moe_intermediate_size",
    # Qwen3.6: a wrapper over a text backbone, on Gated DeltaNet layers, with a shared expert.
    "qwen3_5_moe": "moe_intermediate_size",
    "qwen3_5_moe_text": "moe_intermediate_size",
    # Gemma 4's MoE sibling, by registry_type: its experts run beside a dense mlp, see LAYER_ROUTED.
    "gemma4_moe": "moe_intermediate_size",
    # Inkling 2 Mini, Poros's own decoder (poros.moe.inkling2): a loop over its middle layers;
    # the released config is the multimodal wrapper, its text_config the decoder.
    # (config.json names it intermediate_size; Poros's config class maps that name too)
    "inkling2_loop": "intermediate_size",
    "inkling2_loop_text": "intermediate_size",
}

#: Families whose MoE config is ``config.text_config`` and whose decoder sits
#: under a multimodal wrapper; mapped to the decoder-stack path.
WRAPPED_FAMILIES: dict[str, str] = {
    "gemma4_moe": "model.language_model.layers",
    "qwen3_5_moe": "model.layers",  # ``AutoModelForCausalLM`` leaves its wrapper and vision behind
    "inkling2_loop": "model.layers",  # Poros's decoder leaves the vision tower and MTP head behind
}

#: Families with no sparse block: the decoder layer itself holds router and experts, and calls them
#: on different inputs, so there the experts alone are managed.
LAYER_ROUTED: frozenset[str] = frozenset({"gemma4_moe"})

#: The experts' activation where config.json names none: transformers' default for the family.
#: Every other family defaults to SiLU.
DEFAULT_EXPERT_ACT: dict[str, str] = {"gemma4_moe": "gelu_pytorch_tanh"}


def experts_silu(model_config: Optional[dict], model_type: Optional[str]) -> bool:
    """Whether a MoE model's experts are SiLU, the activation the fused expert compute computes,
    from its config.json (its text config's ``hidden_act`` or ``hidden_activation``, else the
    family's default). Torch-free: what the plan keys a run's compute by."""
    cfg = dict(model_config or {})
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    act = (text.get("hidden_act") or text.get("hidden_activation") or cfg.get("hidden_act")
           or cfg.get("hidden_activation") or DEFAULT_EXPERT_ACT.get(str(model_type), "silu"))
    return str(act) in ("silu", "swish")

#: The oldest and newest transformers minor releases ``tests/moe`` passes on
#: (CPU). Older ones fail it: before 5.13 deepseek_v2's config has no
#: ``num_experts``, and on 5.5 to 5.8 the expert conversion mapping has
#: another shape. pyproject.toml's transformers bound pins the same range, and
#: the canary leg of ``.github/workflows/compat.yml`` runs ``tests/moe`` against
#: the newest release nightly so the newest end moves with evidence.
TRANSFORMERS_TESTED: tuple[tuple[int, int], tuple[int, int]] = ((5, 13), (5, 22))


@functools.cache
def check_transformers() -> None:
    """Refuse a transformers older than ``TRANSFORMERS_TESTED``; log once on a newer one."""
    try:
        found = version("transformers")
    except PackageNotFoundError:
        return
    (lo_major, lo_minor), (hi_major, hi_minor) = TRANSFORMERS_TESTED
    tested = f"transformers>={lo_major}.{lo_minor},<{hi_major}.{hi_minor + 1}"
    match = re.match(r"(\d+)\.(\d+)", found)
    minor = (int(match[1]), int(match[2])) if match else (0, 0)
    if minor < TRANSFORMERS_TESTED[0]:
        raise PorosStackUnsupportedError(
            f"poros.moe needs {tested}, found {found}: older releases lay out MoE "
            "configs and expert weights differently.",
            remediation=f"pip install '{tested}'",
        )
    if minor > TRANSFORMERS_TESTED[1]:
        logger.warning(
            "poros.moe is tested on %s, found %s. The streamed loader still refuses "
            "an expert conversion it does not know; to stay on the tested range, "
            "run: pip install '%s'",
            tested,
            found,
            tested,
        )

_NORM_SUFFIXES = ("layernorm.weight", "_norm.weight", ".norm.weight")
_ROUTER_SUFFIX = ".mlp.gate.weight"
_ROUTER_INFIX = ".router."
_EXPERTS_INFIX = ".experts."


def _has_module(model: nn.Module, name: str) -> bool:
    try:
        model.get_submodule(name)
    except AttributeError:
        return False
    return True


def sparse_layers(config: object) -> tuple[int, ...]:
    """The decoder layers whose MLP is the sparse block, from a HF config (duck-typed)."""
    num_layers = int(config.num_hidden_layers)  # type: ignore[attr-defined]
    mlp_types = getattr(config, "mlp_layer_types", None)
    if mlp_types is not None:
        # An explicit per-layer "dense"/"sparse" schedule.
        return tuple(i for i, t in enumerate(mlp_types) if t == "sparse")
    num_experts = int(config.num_experts)  # type: ignore[attr-defined]
    mlp_only = frozenset(getattr(config, "mlp_only_layers", None) or ())
    step = int(getattr(config, "decoder_sparse_step", None) or 1)
    first = int(getattr(config, "first_k_dense_replace", None) or 0)
    # Qwen3MoeDecoderLayer.__init__ (modeling_qwen3_moe.py:310-321): a
    # layer is sparse iff it is not in mlp_only_layers, num_experts > 0,
    # and (layer_idx + 1) % decoder_sparse_step == 0; deepseek_v2 keeps
    # its first first_k_dense_replace layers dense. Families without
    # those fields (olmoe, mixtral) are sparse on every layer.
    return tuple(
        i
        for i in range(first, num_layers)
        if i not in mlp_only and num_experts > 0 and (i + 1) % step == 0
    )


@dataclass(frozen=True)
class MoeArch:
    """Static MoE facts derived from the HF config of a ``MOE_FAMILIES`` model.

    ``layout`` is the frozen-contract shape record; ``moe_layer_indices``
    lists the decoder layers whose ``mlp`` is a sparse MoE block (the rest
    are dense MLPs and classify ``"dense"``). ``intermediate_size`` is one
    expert's hidden width. ``layer_path`` is the dotted prefix of the
    decoder stack in an unwrapped ``*ForCausalLM``.
    """

    layout: ExpertLayout
    moe_layer_indices: tuple[int, ...]
    intermediate_size: int
    layer_path: str = "model.layers"
    layer_routed: bool = False

    @classmethod
    def from_config(cls, config: object) -> MoeArch:
        """Build the descriptor from a HF config object (duck-typed).

        Raises ``ValueError`` for a model_type outside ``MOE_FAMILIES``, and
        for a router with training-time jitter (Mixtral's), which the managed
        block does not apply. ``check_transformers`` refuses a transformers
        older than ``TRANSFORMERS_TESTED``.
        """
        from poros.moe.contracts import ExpertLayout

        model_type = registry_type(config)
        text = getattr(config, "text_config", None)
        layer_path = WRAPPED_FAMILIES.get(model_type, "model.layers") if text else "model.layers"
        if model_type in WRAPPED_FAMILIES and text:
            config = text
        if model_type not in MOE_FAMILIES:
            raise ValueError(
                f"prepare_moe supports model_type in {sorted(MOE_FAMILIES)} (got "
                f"{model_type!r}): fused-expert sparse blocks with the qwen3_moe "
                "layout (docs/moe.md)."
            )
        check_transformers()
        if (getattr(config, "router_jitter_noise", None) or 0) > 0:
            raise ValueError("prepare_moe does not apply router_jitter_noise; set it to 0")
        num_experts = getattr(config, "num_experts", None)
        if num_experts is None and model_type in WRAPPED_FAMILIES:
            num_experts = config.num_local_experts  # type: ignore[attr-defined]
        if num_experts is None:
            # e.g. DeepSeek-V2 with trust_remote_code=True: the Hub's own config
            # (n_routed_experts) and modeling code, not transformers' fused experts.
            raise ValueError(
                f"this {model_type} config has no num_experts: prepare_moe needs "
                "transformers' own config and model classes. Load without trust_remote_code."
            )
        moe_layers = sparse_layers(config)
        layout = ExpertLayout(
            num_layers=len(moe_layers),
            experts_per_layer=int(num_experts),
            top_k=int(getattr(config, "num_experts_per_tok", None) or config.top_k_experts),
            tensor_names=EXPERT_TENSOR_NAMES,
        )
        intermediate = int(getattr(config, MOE_FAMILIES[model_type]))
        return cls(
            layout=layout,
            moe_layer_indices=moe_layers,
            intermediate_size=intermediate,
            layer_path=layer_path,
            layer_routed=model_type in LAYER_ROUTED,
        )

    def on_model(self, model: nn.Module) -> MoeArch:
        """This arch with the decoder where ``model`` really keeps it.

        ``WRAPPED_FAMILIES`` has Qwen3.6 at ``model.layers``, which is what
        ``AutoModelForCausalLM`` and the streamed loader build. Its own wrapper, from
        ``AutoModelForImageTextToText``, keeps the decoder under ``model.language_model``.
        A model that has the arch's path keeps it.
        """
        wrapped = "model.language_model.layers"
        if self.layer_path == wrapped or _has_module(model, self.layer_path):
            return self
        if _has_module(model, wrapped):
            return dataclasses.replace(self, layer_path=wrapped)
        return self

    # -- module-name patterns (C1) ----------------------------------------

    def block_module(self, layer: int) -> str:
        """Dotted name of the module holding one MoE layer's experts, and its adapter."""
        return f"{self.layer_path}.{layer}" + ("" if self.layer_routed else ".mlp")

    def experts_module(self, layer: int) -> str:
        """Dotted name of one MoE layer's fused experts module."""
        return f"{self.block_module(layer)}.experts"

    def router_module(self, layer: int) -> str:
        """Dotted name of one MoE layer's router module (frozen, unadapted)."""
        return f"{self.block_module(layer)}.{'router' if self.layer_routed else 'gate'}"

    def expert_tensor(self, layer: int, tensor_name: str) -> str:
        """Full parameter name of one fused expert tensor on one layer."""
        return f"{self.experts_module(layer)}.{tensor_name}"

    def decoder_index(self, ordinal: int) -> int:
        """Decoder-layer index for a MoE-layer ordinal (store/managed-block key)."""
        return self.moe_layer_indices[ordinal]

    def expert_keys(self) -> tuple[ExpertKey, ...]:
        """Every (ordinal, expert) key in store order.

        ``layer`` is the MoE-layer ordinal (0..num_layers-1), matching
        ``MoeExpertStore`` and ``ManagedMoeBlock.layer_idx``. Decoder
        indices live in ``moe_layer_indices`` / ``decoder_index``; they
        coincide with ordinals only when every decoder layer is MoE.
        """
        return tuple(
            (ordinal, expert)
            for ordinal in range(self.layout.num_layers)
            for expert in range(self.layout.experts_per_layer)
        )

    # -- classification (C2) ----------------------------------------------

    def classify_parameter(self, name: str) -> ParamClass:
        """Classify a base-model parameter by residency.

        * ``"expert"`` — fused 3-D expert tensors (``...mlp.experts.*``);
          quantized into the ExpertStore and streamed by the ExpertResidency.
        * ``"pinned"`` — router (frozen, unadapted), all norms, embeddings,
          LM head: resident, never streamed.
        * ``"dense"`` — everything else inside the decoder stack (attention
          projections, linear attention, dense-MLP and shared-expert subtrees): streamed by
          the existing blockwise engine via ``managed_parameter_filter``.

        Runs on the base model BEFORE adapters attach; adapter parameters
        are added later and stay resident via the engine's own trainable
        exclusion. Unknown names outside the layer stack fall back to
        ``"pinned"`` — resident is the safe default.
        """
        if _EXPERTS_INFIX in name:
            return "expert"
        if name.endswith(_ROUTER_SUFFIX) or _ROUTER_INFIX in name:
            return "pinned"
        if name.endswith(_NORM_SUFFIXES) or "embed_tokens" in name:
            return "pinned"
        if name.startswith("lm_head"):
            return "pinned"
        if ".self_attn." in name or ".linear_attn." in name or ".mlp." in name:
            return "dense"
        return "pinned"
