"""The HF patch: blockwise residency installed on a HuggingFace model's own forward.

``patch_hf_model`` makes a model's decoder blocks stream through the GPU blockwise, the model
staying a compliant ``nn.Module``. It is an internal detail of the MoE path: ``poros.moe``'s
``prepare_moe`` streams a mixture-of-experts model's dense layers through it, and PorosTrainer
drives that model. A dense model trains through PorosTrainer's own blockwise forward.

Built on:
    core/blockwise.py  (autograd.Functions with INV-1 RNG save/restore)
    core/residency.py  (BlockManager with INV-2 record_stream safety)
    core/arch_registry.py (INV-4 validated registry + INV-5 layer discovery)
    core/determinism.py (INV-3 determinism protocol)

This module holds ``patch_hf_model`` and what it checks and sets up; the forward it installs on
the decoder layers is ``poros.patch.hf.forward``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, List, Literal, Optional, Tuple

from poros.patch.hf.config import _PatchConfig
from poros.patch.hf.forward import (
    _BlockwiseContext,
    _install_blockwise_forward,
    _move_trainable_params_to_model_device,
    _partition_block_trainable_params,
)

logger = logging.getLogger(__name__)


def _selfcheck_tolerance_for_model(model: Any) -> Tuple[float, float]:
    """``(atol, rtol)`` for the patch-path self-check, keyed off the model's frozen weights.

    A model carrying bitsandbytes 4-bit weights is the NF4 bitwise path -> 0.0 tolerance (any
    drift is a real bug). A MoE model's dense layers stream in bf16, where cuBLAS may pick
    different-but-valid kernels as the memory footprint changes between two passes: a structural
    tolerance there still catches a blown-up forward (NaN, a wrong-by-integer-multiple loss)
    without calling bf16 bitwise.
    """
    for module in model.modules():
        if type(module).__name__ in ("Linear4bit", "Params4bit"):
            return (0.0, 0.0)
        weight = getattr(module, "weight", None)
        if weight is not None and type(weight).__name__ == "Params4bit":
            return (0.0, 0.0)
    return (2.0, 0.05)


def _resident_compute_device(model: Any) -> Optional["Any"]:
    """The device the resident set (embeddings/norm/head/adapters) should sit on.

    The frozen base streams from CPU, but the always-resident modules and the
    per-step activations must live on the compute device. We honour an existing
    GPU placement (e.g. ``device_map="auto"``) and otherwise default to the
    current CUDA device, so a CPU-first load (``device_map={"": "cpu"}`` -- which
    avoids the full-resident NF4 load spike) still ends up computing on the GPU.
    """
    import torch

    for p in model.parameters():
        if p.data.numel() > 0 and p.data.device.type == "cuda":
            return p.data.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return None


def _move_resident_to_compute_device(
    model: Any, block_manager: Any, device: "Any"
) -> None:
    """Move every non-streamed param/buffer to the compute device.

    After ``offload_all_to_cpu`` the managed block layers live on CPU and stream
    on demand; everything else (token embeddings, final norm, LM head, rotary
    buffers, adapters) is *resident* and must be on the compute device. This is
    the patch-path analogue of the trainer's ``_move_static_modules_to_device``,
    but it is **layout-agnostic**: instead of hardcoding module paths (which miss
    VLM-nested backbones like Gemma 3/4), it moves anything that is not owned by a
    managed block module. That makes a CPU-first load work on every architecture
    without per-arch placement code.
    """
    if device is None:
        return
    managed_ids = {
        id(t)
        for block in block_manager.blocks
        for module in block_manager.get_block_modules(block.block_id)
        for t in list(module.parameters()) + list(module.buffers())
    }
    for param in model.parameters():
        if id(param) in managed_ids:
            continue
        if param.data.numel() > 0 and param.data.device != device:
            param.data = param.data.to(device)
    for buf in model.buffers():
        if id(buf) in managed_ids:
            continue
        if buf.numel() > 0 and buf.device != device:
            buf.data = buf.data.to(device)


# ---------------------------------------------------------------------------
# Lightweight config shim consumed by the autograd.Functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _BlockwiseContext -- the adapter object passed as ``trainer`` into the
# autograd.Functions.  The autograd.Functions expect the following interface:
#
#   .block_manager          : BlockManager
#   .config                 : object with preserve_rng_state, offload_suffix_inputs_to_cpu
#   .block_trainable_params : dict[int, list[(name, Parameter)]]
#   ._record_remat_event(event_name, block_id)
#   ._forward_block_layers(block_id, hidden_states, attn_mask, position_ids, checkpoint)
#   ._get_adapter_prefetch_stream()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Default block size
# ---------------------------------------------------------------------------

# Canonical, paper-validated block size. Used in every Poros canonical run
# (parity, 3090, Blackwell, seq-len sweep) and the only value with a measured
# parity + VRAM + throughput sweep behind it. block_size is a memory<->speed
# knob (larger = fewer H2D transfers = faster but more resident VRAM; smaller =
# less VRAM but slower); it does NOT affect parity. We do NOT guess it from free
# VRAM -- that was an untested per-machine heuristic that could pick a block too
# large for the card (OOM). Users override explicitly to trade speed vs memory.
DEFAULT_BLOCK_SIZE = 4


def _default_block_size(num_layers: int) -> int:
    """Return the canonical block size, clamped to the layer count."""
    block_size = min(DEFAULT_BLOCK_SIZE, num_layers) if num_layers else DEFAULT_BLOCK_SIZE
    logger.info(
        "Using validated default block_size=%d (override to trade speed vs memory)",
        block_size,
    )
    return block_size


# ---------------------------------------------------------------------------
# Trainable-parameter detection
# ---------------------------------------------------------------------------

def _has_adapters(model: Any) -> bool:
    """Return True if any parameter in the model is trainable (adapter)."""
    for param in model.parameters():
        if param.requires_grad:
            return True
    return False


def _warn_non_adapter_trainable(model: Any) -> None:
    """Warn if trainable parameters look like base-model weights, not adapters.

    Poros assumes a frozen base with only adapter (LoRA) parameters
    trainable.  If non-adapter parameters are trainable, the blockwise
    offload will lose their gradients.
    """
    adapter_keywords = {
        "lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B",
        "lora_a", "lora_b", "adapter", "ia3",
    }
    non_adapter_trainable: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not any(kw in name for kw in adapter_keywords):
                non_adapter_trainable.append(name)

    if non_adapter_trainable:
        sample = non_adapter_trainable[:5]
        msg = (
            f"Found {len(non_adapter_trainable)} trainable parameter(s) that "
            f"do not look like adapter weights (e.g. {sample}). Poros "
            f"blockwise residency assumes a frozen base with only LoRA / "
            f"adapter parameters trainable. Gradients for non-adapter "
            f"parameters in offloaded blocks will NOT be computed."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)


def _neutralize_gradient_checkpointing(model: Any) -> None:
    """Make the model's gradient-checkpointing toggles no-ops after patching.

    HF ``Trainer`` calls ``model.gradient_checkpointing_enable()`` by default
    (``TrainingArguments(gradient_checkpointing=True)`` is common for large
    models). Stock gradient checkpointing wraps the decoder layers in
    ``torch.utils.checkpoint`` -- but under Poros those layers are already
    streamed through a custom autograd.Function that rematerializes the block in
    backward. Letting HF also checkpoint them would double-rematerialize (wrong
    grads / crashes) and break bitwise parity.

    Blockwise residency IS the activation checkpointing here, so we:
      1. disable any checkpointing already installed, and
      2. replace ``gradient_checkpointing_enable``/``disable`` with no-ops that
         warn once, so HF Trainer's default cannot re-enable it.
    """
    # 1. Clear anything already installed.
    # by the name this transformers has: gradient_checkpointing_disable is its older alias
    # for activation_checkpointing_disable, and says so where both exist
    disable = (getattr(model, "activation_checkpointing_disable", None)
               or getattr(model, "gradient_checkpointing_disable", None))
    if callable(disable):
        try:
            disable()
        except Exception:
            pass

    # 2. Neutralize the toggles (idempotent: only patch the real methods once).
    def _enable_noop(*_args: Any, **_kwargs: Any) -> None:
        if not getattr(model, "_poros_gc_warned", False):
            warnings.warn(
                "gradient_checkpointing_enable() is a no-op on a Poros-patched "
                "model: blockwise rematerialization already provides activation "
                "checkpointing. Ignoring the request to avoid double-checkpointing "
                "(which would break parity).",
                UserWarning,
                stacklevel=2,
            )
            model._poros_gc_warned = True

    def _disable_noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    # Bind as plain attributes (shadow the bound methods on the instance).
    model.gradient_checkpointing_enable = _enable_noop  # type: ignore[assignment]
    model.gradient_checkpointing_disable = _disable_noop  # type: ignore[assignment]
    # HF reads this flag in some code paths; keep it consistent.
    try:
        model.gradient_checkpointing = False
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Install blockwise forward routing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layer-path resolution
# ---------------------------------------------------------------------------

def _resolve_layer_path_on_model(
    model: Any,
    full_path: str,
) -> Tuple[List[str], Any]:
    """Walk the model following ``full_path`` and return the attr chain
    plus the ModuleList at the end.

    The ``full_path`` comes from ``discover_layers`` and may have a
    dotted root prefix (possibly including ``get_base_model()``).
    For the root ``get_base_model()`` we resolve dynamically.
    """
    parts = full_path.split(".")
    # Filter out numeric indices (layer index) that discover_layers appends
    # e.g. "model.layers.0" -> we want "model.layers"
    # Actually discover_layers returns "model.layers" without indices.

    obj = model
    attr_chain: List[str] = []
    for part in parts:
        if part == "get_base_model()":
            obj = obj.get_base_model()
            # Don't add to attr_chain since it's a method call
            continue
        try:
            obj = getattr(obj, part)
            attr_chain.append(part)
        except AttributeError:
            raise ValueError(
                f"Could not resolve layer path '{full_path}' on model: "
                f"attribute '{part}' not found after "
                f"{'->'.join(attr_chain) if attr_chain else 'model root'}."
            )

    return attr_chain, obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_hf_model(
    model: Any,
    *,
    block_size: Optional[int] = None,
    enable_prefetch: bool = True,
    preserve_rng_state: bool = True,
    arch_check: Literal["error", "warn", "off"] = "error",
    deterministic: bool = True,
    managed_parameter_filter: Optional[Callable[[str, Any], bool]] = None,
) -> Any:
    """Patch a HuggingFace model for blockwise residency.

    The patched model is still a compliant ``nn.Module``: its own forward streams the frozen
    blocks, and autograd rematerializes each one in the backward.

    Args:
        model: A HuggingFace model (``nn.Module``), optionally with PEFT
            LoRA already applied.
        block_size: Layers resident per block.  ``None`` = the validated
            default ``4`` (not guessed from VRAM).  Memory<->speed knob
            (larger = faster + more VRAM; smaller = less VRAM + slower);
            never affects parity.
        enable_prefetch: Async prefetch of the next block on a dedicated
            CUDA stream.
        preserve_rng_state: Save/restore per-block RNG state for bitwise
            parity (INV-1).  Only disable for debugging.
        arch_check: Architecture validation mode.  ``"error"`` raises on
            unvalidated architectures, ``"warn"`` logs a warning,
            ``"off"`` skips.
        deterministic: Enforce strict determinism (math-only SDPA +
            deterministic algorithms).  Required to PROVE bitwise parity but
            disables flash attention and costs throughput.  The
            blockwise==resident equivalence holds regardless; set ``False``
            for the flash-attention fast path (default ``True``).
        managed_parameter_filter: Optional ``(name, param) -> bool`` predicate
            selecting which block parameters the ``BlockManager`` streams
            (offloads/reloads).  ``None`` (default) manages every block
            parameter.  ``prepare_moe`` passes one that keeps a MoE layer's
            experts, router and norms out of the dense streaming.

    Returns:
        The patched model (same object, mutated in place).

    Raises:
        PorosArchNotValidatedError: If ``arch_check="error"`` and the
            model architecture has not been validated for blockwise
            parity.
        ValueError: If transformer layers cannot be auto-detected.
    """

    from poros.core.backend import refuse_model_on_mps

    refuse_model_on_mps(model, "MoE training")

    from poros.core.arch_registry import discover_layers
    from poros.core.determinism import enforce_determinism

    # ----------------------------------------------------------------
    # 0. Refuse a double patch: a second call would wrap the collector
    #    proxies in a new blockwise list over the already-patched one.
    # ----------------------------------------------------------------
    if getattr(model, "_poros_patched", False):
        from poros.core.errors import PorosError

        raise PorosError(
            "This model is already blockwise-patched.",
            remediation="Build a fresh model: a patched model is never patched twice.",
        )

    # 1. Detect model.config.model_type -> validate against registry
    # ----------------------------------------------------------------
    model_type, config = _checked_model_type(model, arch_check)

    # ----------------------------------------------------------------
    # 2. Locate the layer container via discover_layers (INV-5)
    # ----------------------------------------------------------------
    layer_container, full_path = discover_layers(model)

    # The full_path from discover_layers tells us where the layers live
    # e.g. "model.layers" or "base_model.model.model.layers"
    # We need the attr chain on the original model to swap the ModuleList.
    attr_chain, resolved_layers = _resolve_layer_path_on_model(model, full_path)
    num_layers = len(resolved_layers)

    logger.info(
        "Discovered %d transformer layers at '%s' (model_type=%s)",
        num_layers,
        full_path,
        model_type or "unknown",
    )

    # ----------------------------------------------------------------
    # 3. Detect frozen-base assumption
    # ----------------------------------------------------------------
    adapters_present = _has_adapters(model)
    if adapters_present:
        _warn_non_adapter_trainable(model)
    else:
        logger.info(
            "No trainable (adapter) parameters detected. Will use "
            "JITFrozenBlockFunction (frozen-only, no adapter grads)."
        )

    # ----------------------------------------------------------------
    # 4. Resolve block_size (validated default if not specified)
    # ----------------------------------------------------------------
    if block_size is None:
        block_size = _default_block_size(num_layers)
    else:
        block_size = min(block_size, num_layers)
        if block_size < 1:
            block_size = 1

    logger.info("Using block_size=%d for %d layers", block_size, num_layers)

    # ----------------------------------------------------------------
    # 5. Enforce determinism (INV-3) -- only when requested. Strict
    #    determinism (math-only SDPA) is what PROVES bitwise parity but costs
    #    throughput; the blockwise==resident equivalence holds regardless.
    #    Pass deterministic=False for the flash-attention fast path.
    # ----------------------------------------------------------------
    if deterministic:
        enforce_determinism()

    # ----------------------------------------------------------------
    # 6. Create the BlockManager (INV-2, INV-5)
    #
    #    When adapters are present we must keep them GPU-resident and
    #    trainable: the streaming machinery (load/release) sets
    #    requires_grad=False on every *managed* parameter, so if the LoRA
    #    adapters were managed they'd be frozen and the backward would compute
    #    zero adapter grads.  Mirror the proven PorosTrainer setup: exclude the
    #    trainable (adapter) params from management and disable
    #    train_managed_parameters.  This is the §6.5 adapter-resident guarantee.
    # ----------------------------------------------------------------
    block_manager = _block_manager(model, block_size, adapters_present,
                                   managed_parameter_filter)

    # ----------------------------------------------------------------
    # 7. Build per-block trainable-param index BEFORE offloading
    #    (offloading sets requires_grad=False and empties param.data).
    #
    #    The index is built from the block's *modules* (not the managed-param
    #    enumeration), so adapters -- which are excluded from management above
    #    -- are still attributed to their block.  This mirrors
    #    PorosTrainer._partition_trainable_parameters_by_block.
    # ----------------------------------------------------------------
    pre_offload_trainable = _partition_block_trainable_params(
        model, block_manager
    )

    # Offload all (managed) blocks to CPU so they stream on demand.  Adapter
    # params are not managed, so they are left untouched (resident + trainable).
    block_manager.offload_all_to_cpu()

    # Place the resident set (embeddings, norm, head, rotary buffers, adapters)
    # on the compute device while the frozen base stays on CPU and streams. This
    # makes a CPU-first load (device_map={"": "cpu"}) work -- which skips the
    # full-resident NF4 load spike (~30 GB at 32B) so peak VRAM tracks the
    # streaming steady state (~11 GB). A device_map="auto" model is already
    # GPU-resident, so these moves are no-ops there.
    _compute_device = _resident_compute_device(model)
    _move_resident_to_compute_device(model, block_manager, _compute_device)
    _move_trainable_params_to_model_device(model)

    # ----------------------------------------------------------------
    # 8. Create the lightweight context (trainer adapter)
    # ----------------------------------------------------------------
    patch_config = _PatchConfig(
        preserve_rng_state=preserve_rng_state,
        offload_suffix_inputs_to_cpu=False,
    )
    context = _BlockwiseContext(
        block_manager=block_manager,
        layers=resolved_layers,
        config=patch_config,
        enable_prefetch=enable_prefetch,
        trainable_params_override=pre_offload_trainable,
        model=model,
    )

    # ----------------------------------------------------------------
    # 9. Install blockwise forward (INV-1)
    # ----------------------------------------------------------------
    _install_blockwise_forward(
        model=model,
        layers_attr_chain=attr_chain,
        context=context,
        use_adapter_fn=adapters_present,
    )

    # ----------------------------------------------------------------
    # 9.5 Neutralize HF gradient checkpointing (blockwise remat IS it).
    #     Done before the self-check so stale checkpoint wrappers can't
    #     perturb the probe.
    # ----------------------------------------------------------------
    _neutralize_gradient_checkpointing(model)

    # ----------------------------------------------------------------
    # 9.6 Disable the KV cache: _forward_block_layers strips
    #     past_key_values and forces use_cache=False, so incremental
    #     decode would hand each block an EMPTY cache while
    #     cache_position advances -- model.generate() then silently
    #     produces garbage after the first new token (measured). With
    #     the flags off, generation recomputes the full prefix each
    #     step: slower, bitwise-correct.
    # ----------------------------------------------------------------
    _disable_kv_cache(model)

    # ----------------------------------------------------------------
    # 10. Startup parity self-check. OFF by default -- validated archs are
    #     already validated 0.00e+00, so this is an internal regression tool
    #     only, enabled with POROS_STARTUP_SELFCHECK=always (parity testing).
    # ----------------------------------------------------------------
    _startup_selfcheck(model, block_manager)

    # ----------------------------------------------------------------
    # 11. Attach metadata for introspection
    # ----------------------------------------------------------------
    model._poros_patched = True
    model._poros_block_manager = block_manager
    model._poros_context = context
    model._poros_block_size = block_size
    model._poros_num_blocks = block_manager.num_blocks
    model._poros_arch = model_type or "unknown"

    logger.info(
        "patch_hf_model complete: %d layers -> %d blocks "
        "(block_size=%d, prefetch=%s, rng_state=%s, arch=%s)",
        num_layers,
        block_manager.num_blocks,
        block_size,
        enable_prefetch,
        preserve_rng_state,
        model_type or "unknown",
    )

    return model



def _checked_model_type(model: Any, arch_check: str) -> Tuple[Optional[str], Any]:
    """Step 1 of ``patch_hf_model``: the model's ``model_type`` (a PEFT model's from its base)
    through the architecture gate in ``arch_check``'s mode, and its config."""
    from poros.core.arch_registry import blockwise_arch_check, registry_type
    from poros.preflight.arch import routable_arch_types

    config = getattr(model, "config", None)
    model_type = registry_type(config) if config is not None else None

    if model_type is None:
        # PEFT models wrap the config; try the base model
        base = getattr(model, "get_base_model", lambda: model)()
        config = getattr(base, "config", config)
        model_type = registry_type(config) if config is not None else None

    if model_type is None:
        if arch_check == "error":
            from poros.core.errors import PorosArchNotValidatedError
            raise PorosArchNotValidatedError(
                "unknown (model.config.model_type not found)",
                routable_arch_types(),
            )
        elif arch_check == "warn":
            warnings.warn(
                "Could not detect model_type from model.config. "
                "Proceeding without architecture validation.",
                UserWarning,
                stacklevel=2,
            )
    else:
        blockwise_arch_check(model_type, mode=arch_check)
    return model_type, config


def _block_manager(model: Any, block_size: int, adapters_present: bool,
                   managed_parameter_filter: Optional[Callable[[str, Any], bool]]) -> Any:
    """Step 6 of ``patch_hf_model``: the BlockManager, with the adapters left out of
    management (resident and trainable)."""
    from poros.core.residency import BlockManager

    adapter_param_ids: set = set()
    if adapters_present:
        adapter_param_ids = {
            id(p) for p in model.parameters() if p.requires_grad
        }

    def _effective_managed_filter(name: str, param: Any) -> bool:
        # Honour a user-supplied filter first; then always exclude adapters
        # from management so they stay resident + trainable.
        if managed_parameter_filter is not None and not managed_parameter_filter(
            name, param
        ):
            return False
        if adapter_param_ids and id(param) in adapter_param_ids:
            return False
        return True

    use_filter = (
        _effective_managed_filter
        if (adapters_present or managed_parameter_filter is not None)
        else None
    )

    block_manager = BlockManager(
        model=model,
        block_size=block_size,
        freeze_embeddings=True,
        freeze_lm_head=True,
        managed_parameter_filter=use_filter,
        train_managed_parameters=False,
    )
    return block_manager


def _disable_kv_cache(model: Any) -> None:
    """Step 9.6 of ``patch_hf_model``: ``use_cache`` off on every config that has it."""
    for cfg_holder in (
        getattr(model, "config", None),
        getattr(
            getattr(model, "get_base_model", lambda: model)(), "config", None
        ),
        getattr(model, "generation_config", None),
    ):
        if cfg_holder is not None and hasattr(cfg_holder, "use_cache"):
            cfg_holder.use_cache = False


def _startup_selfcheck(model: Any, block_manager: Any) -> None:
    """Step 10 of ``patch_hf_model``: the startup parity self-check, where it is asked for."""
    from poros.core.arch_registry import run_startup_parity_selfcheck, should_run_startup_selfcheck

    if should_run_startup_selfcheck():
        import torch as _torch

        if _torch.cuda.is_available():
            # The model is already patched, so model(...) IS the blockwise
            # streaming forward; there is no separate resident reference on this
            # path. The self-check therefore runs in determinism-only mode here
            # (see run_startup_parity_selfcheck). Full resident-vs-blockwise
            # parity is validated by PorosTrainer's self-check and the GPU CI.
            def _selfcheck_forward(m: Any, input_ids: Any) -> Any:
                outputs = m(input_ids=input_ids, labels=input_ids)
                return outputs.loss if hasattr(outputs, "loss") else outputs[0]

            _atol, _rtol = _selfcheck_tolerance_for_model(model)
            # Opt-in (internal parity testing): let a failure raise, never swallow it.
            selfcheck_result = run_startup_parity_selfcheck(
                model,
                block_manager,
                _selfcheck_forward,
                atol=_atol,
                rtol=_rtol,
            )
            logger.info(
                "Startup parity self-check passed: %s", selfcheck_result
            )
        else:
            logger.info(
                "Startup parity self-check skipped (no CUDA device available)"
            )
