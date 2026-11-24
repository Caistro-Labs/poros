"""Architecture registry and layer discovery for Poros.

INV-4: Maintains the validated architecture set and enforces the startup
parity self-check. An unvalidated architecture MUST fail loud -- never
train silently-wrong.

INV-5: Layer discovery handles nested / vision-language layouts by searching
multiple root and path candidates.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from poros.core.errors import PorosArchNotValidatedError

logger = logging.getLogger(__name__)

# "validated" is a FIRST-PARTY evidence class: a Caistro parity run with a
# committed artifact. It is not mintable at runtime -- user registrations
# land in "user_asserted", which is runnable but never carries a Poros
# bitwise guarantee.
ArchStatus = Literal["validated", "user_asserted", "target", "unsupported"]


@dataclass(frozen=True)
class ArchSpec:
    """One supported (or in-flight) architecture family.

    `status`:
      - "validated"   : parity confirmed at the arch's native NF4 precision and
                        training proceeds -- the 0.00e+00 bitwise guarantee.
      - "target"      : not validated yet; gated (``--unvalidated`` trains a dense one for a
                        run). What a passing ``poros prove --unvalidated`` pack submits
                        (``poros.validate.pack.registry_line``) until a maintainer's own run
                        validates it.
      - "unsupported" : a family or variant neither loop trains; refused before the download
                        whatever ``blockwise_arch_check`` says (``notes``: why).

    `entry` is the training path the status covers: "dense" (``PorosTrainer``'s blockwise
    forward) or "moe" (``poros.moe``, which ``PorosTrainer`` routes to).
    """

    model_type: str
    status: ArchStatus
    parity: str = ""          # e.g. "0.00e+00" once validated
    notes: str = ""
    entry: Literal["dense", "moe"] = "dense"
    # Submodules a text-only training forward never runs (another modality's tower and its
    # projector), by name. LoRA's "all-linear" default adapts them too; their adapters train
    # nothing on either loop, which the trainer checks after the first update.
    text_unreached: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# INV-4: the architecture registry is the SINGLE source of truth.
#
# To support a new family: add ONE line here (or call register_architecture()
# at runtime, or set POROS_USER_ASSERTED_ARCHS="gemma3,phi3" -- both make user_asserted
# entries, which run but carry no Poros parity guarantee). Nothing else
# in the codebase hardcodes the validated set.
# ---------------------------------------------------------------------------
# model_type strings are what ``config.model_type`` reports. Note the text
# backbones of multimodal families report a ``*_text`` type when loaded via
# AutoModelForCausalLM (gemma3 -> "gemma3_text", gemma4 -> "gemma4_text"), while
# the multimodal wrapper reports the bare family name -- both are registered.
# The release engine's evidence for every validated family: the family gate on one RTX 4090
# (every families.tsv row and the any-model ladder, streamed == resident and the four paths bit for
# bit), and the whole-model runs beside it.
EVIDENCE = "docs/validation/int/4090/round3/fg7_RESULT.txt"
WHOLE_MODELS = "docs/validation/int/4090/logs/fg_whole_*"

_ARCHITECTURES: Dict[str, ArchSpec] = {
    "qwen2": ArchSpec("qwen2", "validated", "0.00e+00",
                      "Qwen2.5; also covers DeepSeek-R1-Distill-Qwen, which is dense qwen2 "
                      f"(the same execution path). Qwen2.5-0.5B: {EVIDENCE}"),
    "qwen3": ArchSpec("qwen3", "validated", "0.00e+00", f"Qwen3-0.6B: {EVIDENCE}"),
    "qwen3_5": ArchSpec("qwen3_5", "validated", "0.00e+00",
                        "Qwen3.5 + Qwen3.6 dense; multimodal wrapper, text backbone is "
                        f"qwen3_5_text. Qwen3.5-0.8B and Qwen3.8-27B (8 layers): {EVIDENCE}; "
                        f"Qwen3.8-27B whole: {WHOLE_MODELS}"),
    "qwen3_5_text": ArchSpec("qwen3_5_text", "validated", "0.00e+00",
                             "Text backbone of the qwen3_5 wrapper: what model.config says once "
                             f"loaded, and what the qwen3_5 rows of {EVIDENCE} train"),
    "gemma2": ArchSpec("gemma2", "target", "", "Not validated yet: --unvalidated trains it"),
    "gemma3": ArchSpec(
        "gemma3", "validated", "0.00e+00",
        f"Gemma 3 multimodal wrapper, vision tower kept (gemma-3-4b-it, 6 layers): {EVIDENCE}",
        text_unreached=("vision_tower", "multi_modal_projector"),
    ),
    "gemma3_text": ArchSpec(
        "gemma3_text", "validated", "0.00e+00",
        f"Gemma 3 text backbone (gemma-3-270m): {EVIDENCE}",
    ),
    "gemma4": ArchSpec(
        "gemma4", "validated", "0.00e+00",
        "Gemma 4 multimodal wrapper (needs transformers>=5), towers kept (gemma-4-31B, 6 layers): "
        f"{EVIDENCE}. Not the E2B and E4B variants: gemma4_per_layer",
        text_unreached=("vision_tower", "audio_tower", "embed_vision", "embed_audio"),
    ),
    "gemma4_per_layer": ArchSpec(
        "gemma4_per_layer", "unsupported", "",
        "Gemma 4 E2B and E4B (hidden_size_per_layer_input or num_kv_shared_layers above 0): "
        "every decoder layer takes a per-layer input and layers share KV caches, which the "
        "block forward does not pass, so neither loop trains them",
    ),
    "gemma4_text": ArchSpec(
        "gemma4_text", "validated", "0.00e+00",
        f"Gemma 4 text backbone (needs transformers>=5; gemma-4-31B, 6 layers): {EVIDENCE}",
    ),
    "gpt_oss": ArchSpec(
        "gpt_oss", "unsupported", "",
        "gpt-oss is a mixture of experts stored in MXFP4, whose 3-D experts neither loop "
        "makes NF4",
        entry="moe",
    ),
    "qwen3_moe": ArchSpec(
        "qwen3_moe", "validated", "0.00e+00",
        "Qwen3-MoE, through poros.moe: NF4 experts streamed per expert, dense layers streamed "
        "in bf16. PorosTrainer and poros train route it there; the dense engine cannot "
        f"quantize its fused 3-D experts. Qwen3-30B-A3B (4 layers): {EVIDENCE}",
        entry="moe",
    ),
    # The families below take qwen3_moe's poros.moe path, each through the same family gate
    # (streamed == resident, the four paths bit for bit), on the first layers of the real
    # checkpoint.
    "olmoe": ArchSpec(
        "olmoe", "validated", "0.00e+00",
        f"OLMoE, through poros.moe as qwen3_moe. OLMoE-1B-7B (4 layers): {EVIDENCE}",
        entry="moe",
    ),
    "qwen2_moe": ArchSpec(
        "qwen2_moe", "validated", "0.00e+00",
        "Qwen2-MoE, through poros.moe as qwen3_moe; the shared expert stays frozen, without "
        f"LoRA. Qwen1.5-MoE-A2.7B (4 layers): {EVIDENCE}",
        entry="moe",
    ),
    "deepseek_v2": ArchSpec(
        "deepseek_v2", "validated", "0.00e+00",
        "DeepSeek-V2, through poros.moe as qwen3_moe; the shared experts stay frozen, without "
        f"LoRA. DeepSeek-V2-Lite (4 layers): {EVIDENCE}",
        entry="moe",
    ),
    "qwen3_5_moe": ArchSpec(
        "qwen3_5_moe", "validated", "0.00e+00",
        "Qwen3.6, through poros.moe as qwen3_moe; a wrapper over qwen3_5_moe_text. Gated "
        "DeltaNet layers and the shared expert stay frozen, without LoRA. Qwen3.6-35B-A3B "
        f"(4 layers): {EVIDENCE}; whole: {WHOLE_MODELS}",
        entry="moe",
    ),
    "qwen3_5_moe_text": ArchSpec(
        "qwen3_5_moe_text", "validated", "0.00e+00",
        "Text backbone of the qwen3_5_moe wrapper: what model.config says once loaded. "
        f"Qwen3.6-35B-A3B's text backbone (4 layers): {EVIDENCE}",
        entry="moe",
    ),
    "gemma4_moe": ArchSpec(
        "gemma4_moe", "validated", "0.00e+00",
        "Gemma 4 with its MoE block, by registry_type: config.json says gemma4. Through "
        "poros.moe, its experts managed where its decoder layer calls them; the dense mlp beside "
        "them stays frozen, without LoRA. Its GELU experts take the reference expert compute. "
        f"gemma-4-26B-A4B (6 layers): {EVIDENCE}",
        entry="moe",
    ),
    # Inkling 2 Mini: a looped decoder (its middle layers run once per pass) that Poros carries
    # itself (poros.moe.inkling2); not validated until its family gate runs on a GPU.
    "inkling2_loop": ArchSpec(
        "inkling2_loop", "target", "",
        "Inkling 2 Mini, through poros.moe with Poros's own decoder, text only: the vision "
        "tower and the multi-token-prediction head are left behind. Not validated yet: no "
        "GPU run has checked it",
        entry="moe",
    ),
    "inkling2_loop_text": ArchSpec(
        "inkling2_loop_text", "target", "",
        "Text decoder of the inkling2_loop wrapper: what model.config says once loaded. Not "
        "validated yet",
        entry="moe",
    ),
}


# Immutable snapshot of the first-party registry, taken before any runtime
# registration can mutate _ARCHITECTURES. validated_arch_types() reads THIS,
# so "validated" cannot be minted at runtime.
_BUILTIN_ARCHITECTURES: Dict[str, ArchSpec] = dict(_ARCHITECTURES)


def register_architecture(
    model_type: str,
    *,
    parity: str = "",
    notes: str = "",
) -> None:
    """Register an architecture as runnable at runtime (``user_asserted``).

    Lets plugins, CI, or a notebook extend Poros without editing the source --
    e.g. ``register_architecture("phi3")`` once you have satisfied yourself it
    works. The architecture becomes runnable and is reported honestly as
    ``user_asserted``.

    It deliberately CANNOT create a ``validated`` entry: that status means a
    first-party Caistro parity run with a committed artifact, and a library
    that lets callers mint its own evidence class has no evidence class. A
    prepared model on a user-asserted arch reports ``probed_not_validated``,
    never ``validated_bitwise``.

    Maintainers add first-party entries in ``_ARCHITECTURES`` with a parity
    run and a PR.
    """
    _ARCHITECTURES[model_type] = ArchSpec(
        model_type, "user_asserted", parity, notes
    )


@contextmanager
def asserted(model_type: str) -> Iterator[None]:
    """``model_type`` runs as ``user_asserted`` for this block alone."""
    before = _ARCHITECTURES.get(model_type)
    _ARCHITECTURES[model_type] = ArchSpec(model_type, "user_asserted", "", "")
    try:
        yield
    finally:
        if before is None:
            del _ARCHITECTURES[model_type]
        else:
            _ARCHITECTURES[model_type] = before


def architectures() -> Dict[str, ArchSpec]:
    """Return a copy of the full architecture registry."""
    return dict(_ARCHITECTURES)


# A model_type as a run's record or a recipe may say it: a word, never a sentence or a path.
TYPE_WORD = re.compile(r"[a-z][a-z0-9_]{0,39}")

# What `--unvalidated` asserts: the env's assertion, for one command alone.
_THIS_COMMAND: ContextVar[frozenset[str]] = ContextVar("poros_unvalidated", default=frozenset())


def assert_this_command(model_type: str) -> None:
    """``model_type`` runs as ``user_asserted`` until this command's :func:`one_command` ends,
    exactly as if ``POROS_USER_ASSERTED_ARCHS`` named it. It changes no status in the registry."""
    _THIS_COMMAND.set(_THIS_COMMAND.get() | {model_type})


@contextmanager
def asserted_inside(model_type: str) -> Iterator[None]:
    """``model_type`` as ``user_asserted`` inside this block alone: what ``--unvalidated`` would
    read, for a verb that says so and asserts nothing past it (``arch``)."""
    token = _THIS_COMMAND.set(_THIS_COMMAND.get() | {model_type})
    try:
        yield
    finally:
        _THIS_COMMAND.reset(token)


def status_outside_this_command(model_type: str) -> Optional[str]:
    """:func:`arch_status` as the registry and the env say it, without what this command asserted:
    a second reading in one command (a sheet's, an answer's) still asks for the flag."""
    token = _THIS_COMMAND.set(frozenset())
    try:
        return arch_status(model_type)
    finally:
        _THIS_COMMAND.reset(token)


def hub_token() -> Optional[bool]:
    """What a Hub call passes as ``token`` in this command: False once it asserted a family Poros
    has not validated, so no stored token reaches a repo nobody vetted (a gated one is refused
    before); None otherwise, the library's own default, so nothing else changes."""
    return False if _THIS_COMMAND.get() else None


@contextmanager
def one_command() -> Iterator[None]:
    """A command's scope: what ``assert_this_command`` said ends with it, so it reaches no other
    command a session runs, and no other thread."""
    token = _THIS_COMMAND.set(frozenset())
    try:
        yield
    finally:
        _THIS_COMMAND.reset(token)


def _env_asserted_archs() -> set[str]:
    """Architectures the operator asserts are runnable: ``POROS_USER_ASSERTED_ARCHS``
    (comma-separated), or for this command alone (``--unvalidated``)."""
    raw = os.environ.get("POROS_USER_ASSERTED_ARCHS", "")
    return {t.strip() for t in raw.split(",") if t.strip()} | _THIS_COMMAND.get()


# Python objects per layer, expert and label are real memory, so a config's counts are bounded
# before anything reads them: a 48-byte {"num_labels": 100000000000} filled 4 GiB in 5 s.
_MOST_LAYERS, _MOST_EXPERTS, _MOST_LINEARS, _MOST_LABELS = 1024, 4096, 100_000, 1024
_MOST_INT = 1 << 40  # no real field is this large: a size alone takes nothing on the meta device
_COUNT = re.compile(r"layers|blocks|depth|n_layer|heads|groups|stages|codebooks", re.IGNORECASE)
_EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts")


def bounded(raw: dict) -> None:
    """ValueError for a number that would take real memory, in this config or any nested in it."""
    def number(key: str, value: int) -> None:
        if abs(value) > _MOST_INT:
            raise ValueError("its numbers are too large")
        if _COUNT.search(key) and value > _MOST_LAYERS:
            raise ValueError("its layers are too many")

    def walk(table: dict) -> None:
        most = sum(table[k] for k in _EXPERTS if isinstance(table.get(k), int)) or 1
        layers = table.get("num_hidden_layers", 1)
        if not isinstance(layers, int):
            layers = 1
        if most > _MOST_EXPERTS or layers * most > _MOST_LINEARS:
            raise ValueError("its layers or experts are too many")
        labels = table.get("num_labels", 0)
        if isinstance(labels, int) and labels > _MOST_LABELS:
            raise ValueError("its labels are too many")
        if any(len(table.get(k) or ()) > _MOST_LABELS for k in ("id2label", "label2id")
               if isinstance(table.get(k), (dict, list))):
            raise ValueError("its labels are too many")
        for key, value in table.items():
            for one in value if isinstance(value, list) else (value,):
                if isinstance(one, dict):
                    walk(one)
                elif isinstance(one, int):
                    number(key, one)

    walk(raw)


def registry_type(config: Any) -> Optional[str]:
    """What the registry calls a model, from its config (a config.json's dict, or an HF config).

    Its ``model_type``, except where that one name covers what trains differently: a Gemma 4 with
    its MoE block has fused experts the dense engine cannot quantize, and is ``gemma4_moe``; a
    Gemma 4 with per-layer inputs or shared KV (E2B, E4B) is ``gemma4_per_layer``, unsupported.
    """

    def field(holder: Any, name: str) -> Any:
        return holder.get(name) if isinstance(holder, dict) else getattr(holder, name, None)

    model_type = field(config, "model_type")
    text = field(config, "text_config") or config
    if model_type in ("gemma4", "gemma4_text") and field(text, "enable_moe_block"):
        return "gemma4_moe"
    if model_type in ("gemma4", "gemma4_text") and any(
            (field(text, k) or 0) > 0
            for k in ("hidden_size_per_layer_input", "num_kv_shared_layers")):
        return "gemma4_per_layer"
    return model_type


def text_unreached(model_type: Optional[str]) -> Tuple[str, ...]:
    """The submodule names a text-only training forward of ``model_type`` never runs
    (``ArchSpec.text_unreached``)."""
    spec = _ARCHITECTURES.get(model_type or "")
    return spec.text_unreached if spec is not None else ()


def validated_arch_types(entry: Literal["dense", "moe"] = "dense") -> set[str]:
    """First-party validated model_types for one training path (``ArchSpec.entry``).

    Deliberately NOT influenced by ``register_architecture`` or the
    environment: this is the set backed by committed Caistro parity
    artifacts, and it is what a bitwise guarantee may be derived from. For
    "what will Poros agree to run", use :func:`runnable_arch_types`.
    """
    return {
        s.model_type for s in _BUILTIN_ARCHITECTURES.values()
        if s.status == "validated" and s.entry == entry
    }


def user_asserted_arch_types() -> set[str]:
    """Model types the caller asserted are runnable (runtime + env)."""
    registered = {
        s.model_type for s in _ARCHITECTURES.values()
        if s.status == "user_asserted"
    }
    return registered | _env_asserted_archs()


def runnable_arch_types() -> set[str]:
    """Everything the arch gate will let through: validated + user-asserted."""
    return validated_arch_types() | user_asserted_arch_types()


def arch_status(model_type: str) -> Optional[str]:
    """Return the registry status of a model_type, or None if unknown."""
    validated = validated_arch_types() | validated_arch_types("moe")
    if model_type in _env_asserted_archs() and model_type not in validated:
        # Asserted by the operator, not validated by Caistro. A built-in
        # validated entry always wins: listing it in the env must not
        # DOWNGRADE a first-party architecture.
        return "user_asserted"
    spec = _ARCHITECTURES.get(model_type)
    return spec.status if spec else None


# Why the dense engine refuses an arch validated for poros.moe only.
MOE_ENTRY_HINT = (
    "PorosTrainer and poros train route it to poros.moe; the "
    "dense engine cannot quantize its fused experts."
)

# -- INV-5: Root and path candidates for layer discovery --
ROOT_CANDIDATES: list[str] = [
    "",
    "model",
    "base_model",
    "base_model.model",
    "base_model.model.model",
    "language_model",           # Gemma3 / VLM
    "model.language_model",     # Gemma3 / VLM
    "language_model.model",     # Gemma3 / VLM
]

LAYER_PATHS: list[str] = [
    "model.layers",             # Qwen2, Qwen3, Phi3, Gemma
    "language_model.layers",    # VLM nested
    "gpt_neox.layers",          # GPT-NeoX, Pythia
    "transformer.h",            # GPT-2, GPT-J
    "model.decoder.layers",     # OPT
    "encoder.layer",            # encoder-only (tests)
]


def _get_nested_attr(obj: object, attr_path: str) -> object:
    """Resolve a dotted attribute path on an object."""
    if not attr_path:
        return obj
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def discover_layers(model: object) -> Tuple[object, str]:
    """Discover the transformer layer container in a model.

    Searches ROOT_CANDIDATES x LAYER_PATHS, plus get_base_model() as
    a dynamic root candidate.

    Args:
        model: A HuggingFace model (nn.Module).

    Returns:
        A tuple of (layer_container: nn.ModuleList, full_path: str).

    Raises:
        ValueError: If no transformer layers are found.
    """
    # Build root candidates including dynamic get_base_model()
    roots: List[Tuple[object, str]] = []
    seen_ids: set[int] = set()

    for root_path in ROOT_CANDIDATES:
        try:
            root_obj = _get_nested_attr(model, root_path)
            obj_id = id(root_obj)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                roots.append((root_obj, root_path))
        except AttributeError:
            continue

    # Try get_base_model() as a dynamic root
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()  # type: ignore[union-attr]
            obj_id = id(base)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                roots.append((base, "get_base_model()"))
        except Exception:
            pass

    # Search all root x path combinations
    for root_obj, root_name in roots:
        for layer_path in LAYER_PATHS:
            try:
                layers = _get_nested_attr(root_obj, layer_path)
                full = f"{root_name}.{layer_path}" if root_name else layer_path
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers, full
            except AttributeError:
                continue

    raise ValueError(
        "Could not auto-detect transformer layers. "
        f"Searched roots: {ROOT_CANDIDATES} x paths: {LAYER_PATHS}. "
        "If your model uses a non-standard layout, please open an architecture "
        "support request."
    )


def blockwise_arch_check(
    model_type: str,
    *,
    mode: Literal["error", "warn", "off"] = "error",
) -> bool:
    """Check whether a model architecture is validated for blockwise training.

    Args:
        model_type: The HuggingFace model_type string.
        mode: "error" raises, "warn" logs a warning, "off" skips.

    Returns:
        True if the architecture is validated, False otherwise.

    Raises:
        PorosArchNotValidatedError: If mode="error" and arch is not validated.
    """
    # The gate asks "may Poros run this", which includes user assertions.
    # Whether a bitwise guarantee applies is a different question, answered
    # by validated_arch_types() alone.
    valid_set = runnable_arch_types()
    trains = valid_set | validated_arch_types("moe")  # what the error lists: every route's
    spec = _ARCHITECTURES.get(model_type)
    if spec is not None and spec.status == "unsupported":  # in every mode: neither loop trains it
        raise PorosArchNotValidatedError(model_type, trains, unsupported=True,
                                         remediation=f"Unsupported: {spec.notes}.")
    if mode == "off":
        return model_type in valid_set

    is_valid = model_type in valid_set

    if not is_valid:
        if mode == "error":
            moe = model_type in validated_arch_types("moe")
            raise PorosArchNotValidatedError(model_type, trains, moe=moe,
                                             remediation=MOE_ENTRY_HINT if moe else None)
        elif mode == "warn":
            logger.warning(
                "Architecture '%s' has not been validated for blockwise parity. "
                "Architectures this install trains: %s. Proceeding at your own risk.",
                model_type,
                ", ".join(sorted(trains)),
            )

    return is_valid


def should_run_startup_selfcheck(
    selfcheck_mode: Literal["off", "always"] = "off",
) -> bool:
    """Decide whether the startup resident-vs-blockwise parity self-check runs.

    OFF by default. The self-check re-derives, at the start of a run, the parity
    that is already validated 0.00e+00 for the validated architectures -- so on
    the normal path it is pure overhead (and a fully resident reference forward
    can OOM on exactly the large models Poros exists to stream). It exists as an
    internal regression tool: internal parity testing opts in with ``"always"``
    or by exporting ``POROS_STARTUP_SELFCHECK=always`` (the env var overrides
    ``selfcheck_mode``).
    """
    env = os.environ.get("POROS_STARTUP_SELFCHECK", "").strip().lower()
    if env in {"off", "always"}:
        selfcheck_mode = env  # type: ignore[assignment]
    return selfcheck_mode == "always"


def discover_named_layers(model: object) -> List[Tuple[str, Any]]:
    """Discover transformer layers as a list of (dotted_name, module).

    Thin wrapper over `discover_layers` so callers that want a per-layer list
    (e.g. the BlockManager) share the exact same root/path search -- one place
    to add a new layout.
    """
    container, full_path = discover_layers(model)
    return [(f"{full_path}.{i}", layer) for i, layer in enumerate(container)]


def run_startup_parity_selfcheck(
    model: Any,
    block_manager: Any,
    forward_fn: Any,
    *,
    resident_forward_fn: Optional[Any] = None,
    device: Optional[Any] = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Dict[str, Any]:
    """Run the mandatory 1-step parity self-check before training.

    Two modes, chosen by whether a resident reference is available:

    - **resident-parity** (``resident_forward_fn`` given): Pass 1 runs the
      reference forward with all blocks held resident; Pass 2 runs the blockwise
      streaming forward. Because both share identical layer math, the only
      variable is residency, so a 0.00e+00 loss diff proves *streaming preserves
      parity* -- the headline claim. This is what ``PorosTrainer`` provides.

    - **determinism-only** (no ``resident_forward_fn``, or the resident
      reference would exceed VRAM): runs ``forward_fn`` twice. This catches a
      non-deterministic blockwise path but does NOT prove resident parity (there
      is no resident reference). Callers that can only produce the streaming
      forward (e.g. an already-patched HF model), and large models whose resident
      reference does not fit, get this weaker check; ``result["mode"]`` says
      which ran. Full resident parity at scale is covered by the GPU CI and the
      benchmark grid.

    Each ``forward_fn`` is responsible for its own block residency (load /
    compute / release); this function does not manage blocks itself.

    Args:
        model: The nn.Module model (with adapters attached if applicable).
        block_manager: The BlockManager (used only for arch reporting here).
        forward_fn: callable(model, input_ids) -> scalar loss; the blockwise
            streaming path.
        resident_forward_fn: optional callable(model, input_ids) -> scalar loss
            running the same math with all blocks resident. When given, enables
            the genuine resident-vs-blockwise comparison.
        device: CUDA device. Defaults to cuda:0 if available.
        atol/rtol: tolerances (0.0 = bitwise).

    Returns:
        Dict with {loss_max_diff, weight_max_diff, bitwise_exact, arch, mode}.
        ``weight_max_diff`` is always 0.0 here: a no-grad forward cannot change
        weights, so weight parity is covered by the full training parity test,
        not this forward-only probe.

    Raises:
        PorosArchNotValidatedError: If the losses differ past tolerance.
    """
    import torch

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arch = "unknown"
    config = getattr(model, "config", None)
    if config is not None:
        arch = getattr(config, "model_type", "unknown")

    vocab_size = 32000
    if config is not None:
        vocab_size = getattr(config, "vocab_size", 32000)

    def _seed() -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    _seed()
    input_ids = torch.randint(0, vocab_size, (1, 8), device=device)

    # -- Pass 1: resident reference (or first streaming pass) --
    # The resident reference holds ALL blocks on the GPU at once. For the large
    # models Poros targets, that may not fit -- the whole reason to stream. So we
    # attempt it and, on OOM, fall back to a determinism-only check rather than
    # crash the run we were trying to protect.
    mode = "resident-parity" if resident_forward_fn is not None else "determinism-only"
    loss_a = None
    if resident_forward_fn is not None:
        try:
            _seed()
            with torch.no_grad():
                loss_a = resident_forward_fn(model, input_ids).detach().cpu()
        except RuntimeError as exc:
            # torch.cuda.OutOfMemoryError subclasses RuntimeError; match on the
            # message so this works across torch versions.
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            loss_a = None
            mode = "determinism-only (resident reference exceeded VRAM)"
            logger.warning(
                "Resident reference did not fit in VRAM; falling back to a "
                "determinism-only self-check. Full resident-vs-blockwise parity "
                "for this model size is validated by the GPU CI / benchmark grid."
            )

    if loss_a is None:
        _seed()
        with torch.no_grad():
            loss_a = forward_fn(model, input_ids).detach().cpu()

    # -- Pass 2: blockwise streaming --
    _seed()
    with torch.no_grad():
        loss_b = forward_fn(model, input_ids).detach().cpu()

    loss_max_diff = float(torch.abs(loss_a - loss_b).max())
    # Forward-only, no_grad: weights cannot change. Weight parity is validated by
    # the full multi-step training parity test, not this probe.
    weight_max_diff = 0.0
    bitwise_exact = loss_max_diff == 0.0

    result = {
        "loss_max_diff": loss_max_diff,
        "weight_max_diff": weight_max_diff,
        "bitwise_exact": bitwise_exact,
        "arch": arch,
        "mode": mode,
    }

    if not torch.allclose(loss_a, loss_b, atol=atol, rtol=rtol):
        logger.error(
            "Startup parity check FAILED (%s): loss_max_diff=%.2e, arch=%s",
            mode,
            loss_max_diff,
            arch,
        )
        raise PorosArchNotValidatedError(
            arch, runnable_arch_types() | validated_arch_types("moe"),
            selfcheck=f"{mode}, loss_max_diff={loss_max_diff:.2e}",
            remediation="This is a bug in this path, not in the model: report it with the model "
                        "and this message (docs/models.md).")

    logger.info(
        "Startup parity check PASSED (%s): loss_max_diff=%.2e, "
        "bitwise_exact=%s, arch=%s",
        mode,
        loss_max_diff,
        bitwise_exact,
        arch,
    )
    return result
