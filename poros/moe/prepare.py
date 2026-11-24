"""``prepare_moe``: the single entry point of the quarantined MoE path.

Orchestrates, in order (workstream-c.md C3):

1. classify every base parameter (``poros.moe.arch``),
2. freeze the base (QLoRA: adapters alone train; the router stays frozen),
3. quantize the fused expert tensors into an ``ExpertStore`` (workstream B),
4. build the ``ExpertResidency`` (workstream A; ``streamed=False`` pins
   everything — the parity-reference configuration, README §4),
5. install the managed MoE blocks (workstream B),
6. attach resident expert LoRA adapters to the managed blocks (workstream B),
7. stream the remaining *dense* parameters by composing the existing
   blockwise engine through its ``managed_parameter_filter`` seam
   (import-only reuse of ``poros.patch.hf.patch_hf_model``, zero edits).

Honesty (AGENTS.md §2): the report claims ``validated_bitwise`` only for the
configuration Caistro's parity artifacts cover — a model whose architecture
the registry validates for this path, loaded in bf16,
on the built-in NF4 store and residency, streamed,
with its dense layers streamed on CUDA. Otherwise the guarantee is
``"experimental_moe_streamed"`` or ``"resident_reference"``, and the
report's warnings say exactly what ran.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from poros.core.arch_registry import registry_type, validated_arch_types
from poros.core.backend import refuse_model_on_mps, refuse_on_mps
from poros.moe.arch import MoeArch
from poros.moe.contracts import (
    ExpertResidency,
    ExpertStore,
    RouteObserver,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MoeBackends",
    "MoeConfig",
    "MoeReport",
    "default_backends",
    "dense_managed_filter",
    "prepare_moe",
    "streaming_config",
]


@dataclass(frozen=True)
class MoeConfig:
    """Knobs for ``prepare_moe`` (workstream-c.md C4 — deliberately exhaustive).

    ``capacity_bytes`` bounds the expert residency budget; ``0`` means
    unbounded (no eviction pressure). ``seed`` drives adapter init so two
    prepares with the same seed produce identical adapters; ``lora_dropout``
    applies to the adapter branch only, as on the dense path. ``device`` is
    where expert slabs are made resident — it must be the device the model
    computes on at forward time (``"cpu"`` default; set ``"cuda"`` on GPU
    hosts). ``expert_batch`` caps how many experts each batched expert call
    dequantizes and multiplies (``0``: no cap; calls already group a layer's
    selected experts by similar token count). It trades transient VRAM for
    kernel launches: per Qwen3-30B-A3B expert, about 9.4 MB of bf16 weights
    plus about 19 MB of scratch while gate_up_proj decodes. Results can
    differ in the last bits between values; streamed and resident runs with
    the same value stay bitwise equal.
    """

    streamed: bool = True
    capacity_bytes: int = 0
    look_ahead: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    seed: int = 1234
    device: str = "cpu"
    expert_batch: int = 0
    #: Reuse each MoE layer's block-forward output in the backward rematerialization
    #: when dense blocks stream (verified equal inputs and RNG; bitwise the recompute).
    reuse_remat_output: bool = True
    #: "reference" (bucketed bmm; the bitwise reference), "grouped" (poros.moe.grouped:
    #: grouped GEMM over exact rows, chunked backward; opt-in, pinned per CUDA host), "fused"
    #: (poros.moe.fused, refused for a model it cannot run: poros.moe.fused.model_reason), or
    #: "auto": fused where it runs every layer, else reference (MoeReport says which and why).
    expert_compute: str = "reference"
    #: "pow2" (power-of-two token-count groups, the reference) or "single" (one padded group
    #: per layer call: fewer launches, more padded math; shapes differ, so a different but
    #: self-consistent bit pattern -- sequential and block-major stay bitwise equal).
    expert_grouping: str = "pow2"
    #: Layers per streamed dense block (None: the dense engine's default). Block size never
    #: changes the numbers; block-major MoE uses 1 so each MoE layer runs all G microsteps in a
    #: row and its decoded experts are reused across them (poros.moe.store.EXPERT_DQ_CACHE).
    dense_block_size: int | None = None


def streaming_config(model_config: object, *, store: Any = None, **knobs: Any) -> MoeConfig:
    """The streamed ``MoeConfig`` the receipts ran, for a ``MOE_FAMILIES`` HF config.

    The residency holds one MoE layer's expert slices plus one slice of
    look-ahead: the smallest budget it accepts, and the one the MoE parity receipts ran.
    ``knobs`` set any other ``MoeConfig`` field.
    With ``store`` (an ``ExpertStore``, e.g. a plugin's) the budget is its own
    largest MoE layer, measured from ``slice_nbytes`` instead of the NF4 sizes.
    """
    if store is not None:
        lay = store.layout
        capacity = max(
            sum(store.slice_nbytes((layer, e)) for e in range(lay.experts_per_layer))
            for layer in range(lay.num_layers)
        )
        return MoeConfig(capacity_bytes=capacity, look_ahead=1, **knobs)
    from poros.moe.quant import quantized_nbytes

    arch = MoeArch.from_config(model_config)
    model_config = getattr(model_config, "text_config", None) or model_config
    hidden, mid = int(model_config.hidden_size), arch.intermediate_size  # type: ignore[attr-defined]
    per_expert = quantized_nbytes((2 * mid, hidden)) + quantized_nbytes((hidden, mid))
    capacity = arch.layout.experts_per_layer * per_expert
    return MoeConfig(capacity_bytes=capacity, look_ahead=1, **knobs)


MoeGuarantee = Literal["validated_bitwise", "experimental_moe_streamed", "resident_reference"]


@dataclass
class MoeReport:
    """What ``prepare_moe`` did — and what it guarantees.

    The guarantee label is derived from what actually ran, never asserted
    ahead of evidence (see the module docstring for when
    ``"validated_bitwise"`` applies).
    """

    guarantee: MoeGuarantee
    streamed: bool
    dense_streamed: bool
    model_type: str
    experts_quantized: int
    blocks_installed: int
    params_classified: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    #: The expert compute the blocks run, and why "auto" took the reference (None otherwise).
    expert_compute: str = "reference"
    expert_compute_reason: str | None = None


@dataclass(frozen=True)
class MoeBackends:
    """The workstream A/B callables ``prepare_moe`` orchestrates.

    Contracts-only seam: ``poros/moe/contracts.py`` freezes the protocols,
    not constructor names, so this bundle is the integration surface.
    ``default_backends()`` binds the real workstream A/B implementations.

    * ``quantize_experts(model, arch) -> ExpertStore`` — quantize every
      fused expert tensor into host-side slabs (B).
    * ``attach_adapters(model, arch, rank, alpha, seed, dropout) -> None`` —
      resident, trainable LoRA on the expert projections, seeded init (B).
    * ``build_residency(store, streamed, capacity_bytes, look_ahead, device)
      -> ExpertResidency`` — the ExpertBlockManager (A); ``device`` is where
      slabs are made resident (the model's compute device).
    * ``install_modules(model, arch, store, residency, on_route) -> int`` —
      replace the HF MoE blocks with managed blocks computing the reference
      forward's math; returns blocks installed (B).
    """

    quantize_experts: Callable[[nn.Module, MoeArch], ExpertStore]
    attach_adapters: Callable[[nn.Module, MoeArch, int, int, int, float], None]
    build_residency: Callable[[ExpertStore, bool, int, int, int, str], ExpertResidency]
    install_modules: Callable[
        [nn.Module, MoeArch, ExpertStore, ExpertResidency, "RouteObserver | None"],
        int,
    ]


def default_backends() -> MoeBackends:
    """The real workstream A/B backends, wired at integration.

    Each callable adapts a workstream API to the bundle's call shape:
    B's ``MoeExpertStore``/``install_managed_blocks``/``attach_expert_adapters``
    and A's ``ExpertBlockManager``.
    """
    from poros.moe.adapters import attach_expert_adapters
    from poros.moe.modules import install_managed_blocks
    from poros.moe.residency import ExpertBlockManager
    from poros.moe.store import MoeExpertStore

    def _quantize(model: nn.Module, arch: MoeArch) -> ExpertStore:
        store = getattr(model, "_poros_moe_store", None)
        if store is not None:  # quantized at load (poros.moe.load)
            return store
        params = dict(model.named_parameters())
        state: dict[str, torch.Tensor] = {}
        for layer in arch.moe_layer_indices:
            for tensor_name in arch.layout.tensor_names:
                full_name = arch.expert_tensor(layer, tensor_name)
                # The store is host-side by design; stage GPU-loaded weights
                # to host before quantization (the NF4 codebooks are CPU).
                state[full_name] = params[full_name].detach().cpu()
        # Dequantize in the model's own dtype so the managed block's
        # F.linear inputs match (fp32 tiny-test models, bf16 real ones).
        first = state[arch.expert_tensor(arch.moe_layer_indices[0], arch.layout.tensor_names[0])]
        return MoeExpertStore.from_state_dict(
            state,
            num_layers=arch.layout.num_layers,
            experts_per_layer=arch.layout.experts_per_layer,
            top_k=arch.layout.top_k,
            key_format=arch.experts_module("{layer}") + ".{tensor}",
            tensor_names=arch.layout.tensor_names,
            moe_layer_ids=list(arch.moe_layer_indices),
            compute_dtype=first.dtype,
        )

    def _attach(
        model: nn.Module, arch: MoeArch, rank: int, alpha: int, seed: int, dropout: float
    ) -> None:
        del arch  # adapters discover managed blocks by type
        # Seeded fork: identical adapter init for identical seeds, without
        # disturbing the caller's RNG stream (adapter math mirrors the dense
        # path's kaiming-A / zero-B init).
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            attach_expert_adapters(model, rank=rank, alpha=alpha, dropout=dropout)

    def _build(
        store: ExpertStore,
        streamed: bool,
        capacity_bytes: int,
        look_ahead: int,
        device: str,
    ) -> ExpertResidency:
        if capacity_bytes <= 0:
            # MoeConfig's 0 means "unbounded": size the budget to hold every
            # slice, so eviction never triggers unless the caller opts in.
            layout = store.layout
            capacity_bytes = sum(
                store.slice_nbytes((layer, expert))
                for layer in range(layout.num_layers)
                for expert in range(layout.experts_per_layer)
            )
        return ExpertBlockManager(
            store,
            streamed=streamed,
            capacity_bytes=capacity_bytes,
            look_ahead=look_ahead,
            device=device,
        )

    def _install(
        model: nn.Module,
        arch: MoeArch,
        store: ExpertStore,
        residency: ExpertResidency,
        on_route: RouteObserver | None,
    ) -> int:
        return install_managed_blocks(model, arch, store, residency, on_route=on_route)

    return MoeBackends(
        quantize_experts=_quantize,
        attach_adapters=_attach,
        build_residency=_build,
        install_modules=_install,
    )


def dense_managed_filter(arch: MoeArch) -> Callable[[str, nn.Parameter], bool]:
    """The ``managed_parameter_filter`` predicate for the dense engine.

    Selects exactly the ``"dense"`` class for streaming: expert-fused
    tensors (owned by the ExpertStore/ExpertResidency) and pinned params
    (router, norms, embeddings, LM head) are excluded, so the BlockManager
    never touches them. Trainable adapter params are excluded by
    ``patch_hf_model`` itself (its adapter-resident guarantee).
    """

    def _filter(name: str, param: nn.Parameter) -> bool:
        return arch.classify_parameter(name) == "dense"

    return _filter


def _stream_dense_params(model: nn.Module, arch: MoeArch, block_size: int | None = None) -> None:
    """Compose the existing blockwise engine over the dense parameters.

    Import-only reuse of ``poros.patch.hf.patch_hf_model`` with the
    ``managed_parameter_filter`` seam — zero edits to shared files.
    ``arch_check="off"`` because the dense gate covers dense-engine
    architectures only; the registry validates MoE architectures for this
    path, and the guarantee lives in ``MoeReport``.
    """
    from poros.patch.hf import patch_hf_model

    patch_hf_model(
        model,
        arch_check="off",
        managed_parameter_filter=dense_managed_filter(arch),
        block_size=block_size,
        deterministic=False,  # prepare_moe enforced determinism above (its fill choice kept)
    )


_EXPERIMENTAL_WARNING = (
    "This run is outside the validated poros.moe configuration (a validated "
    "MoE architecture loaded in bf16, the built-in NF4 store and residency, "
    "streamed, dense layers streamed on CUDA), so no "
    "validated_bitwise "
    "guarantee applies. Its parity claim is streamed-vs-"
    "resident bitwise equality (0.00e+00) as proven by poros.moe.gates on "
    "this run's configuration."
)

_CPU_DENSE_WARNING = (
    "Model is not on CUDA: dense parameters stay RESIDENT and only expert "
    "residency is streamed. The dense engine's streamed-vs-resident bitwise "
    "parity is validated on CUDA, not CPU (measured drift ~5e-7 on CPU), so "
    "on this host the expert path is the only variable under test "
    "(workstream-c.md C8). Load or move the model onto CUDA for full "
    "composition."
)


def _already_moe_prepared(model: nn.Module) -> bool:
    """True once prepare_moe has installed managed blocks or attached a report.

    The report is written last, so a crash after ``install_modules`` would
    otherwise leave a model that dies on retry with a bare KeyError (fused
    expert params already dropped). Managed blocks are the durable signal.
    """
    if getattr(model, "_poros_moe_report", None) is not None:
        return True
    from poros.moe.modules import ManagedMoeBlock

    return any(isinstance(m, ManagedMoeBlock) for m in model.modules())


def _move_trainable_to_model_device(model: nn.Module, device: torch.device) -> None:
    """Keep adapter params on the model's compute device in both prepare arms.

    Streamed CUDA composition also does this inside ``patch_hf_model``. The
    resident reference never patches, so this helper is the shared move.
    Device is the one captured before adapters attach: after attach, the
    first parameter may be a CPU adapter, and scanning would no-op the move.
    """
    for param in model.parameters():
        if param.requires_grad and param.device != device:
            param.data = param.data.to(device)


def prepare_moe(
    model: nn.Module,
    config: MoeConfig,
    *,
    backends: MoeBackends | None = None,
    on_route: RouteObserver | None = None,
) -> MoeReport:
    """Prepare a sparse MoE model for QLoRA with blockwise expert residency.

    Mutates ``model`` in place (frozen base, quantized experts, managed MoE
    blocks, optional dense streaming) and returns an honest report; the same
    report is attached as ``model._poros_moe_report``.

    Args:
        model: An unwrapped ``*ForCausalLM`` of a ``MOE_FAMILIES`` model
            (``Qwen3MoeForCausalLM`` is the validated one; bf16 or wider; only
            bf16 is validated; the fused expert tensors must still be
            materialized — they are quantized into the ExpertStore here —
            or already quantized by ``poros.moe.load.load_moe_streamed``).
        config: ``MoeConfig``; ``streamed=False`` is the parity-reference
            configuration (everything pinned resident).
        backends: Workstream A/B implementations. ``None`` resolves to
            ``default_backends()`` (the integration wiring point).
        on_route: Optional ``RouteObserver`` receiving every committed
            ``RouteDecision`` (gate traces, debugging).

    Raises:
        ValueError: If ``model.config.model_type`` is not in ``MOE_FAMILIES``.
        RuntimeError: If ``model`` was already prepared by ``prepare_moe``
            (mirrors ``patch_hf_model``'s already-patched guard), including
            a previous call that failed after installing managed blocks.
    """
    # A checkpoint loads on CPU and config.device is where forward runs: check both.
    refuse_model_on_mps(model, "MoE training")
    refuse_on_mps(torch.device(config.device), "MoE training")
    if _already_moe_prepared(model):
        raise RuntimeError(
            "This model is already MoE-prepared (managed MoE blocks are "
            "installed or model._poros_moe_report is set). Build a fresh "
            "model instead of calling prepare_moe twice."
        )
    if config.expert_batch < 0:
        raise ValueError(f"MoeConfig.expert_batch must be >= 0, got {config.expert_batch}")
    if backends is None:  # a store plugin may name its own (poros.stores)
        backends = getattr(model, "_poros_moe_backends", None) or default_backends()
    arch = MoeArch.from_config(model.config).on_model(model)
    warnings: list[str] = []

    # The residency device must be the model's compute device; a mismatch
    # would crash at the first forward (slabs on one device, activations on
    # the other), so say so loudly at prepare time.
    model_device = next(model.parameters()).device
    target = torch.device(config.device)
    moving_cpu_to_cuda = target.type == "cuda" and model_device.type == "cpu"
    if target.type != model_device.type and not moving_cpu_to_cuda:
        warnings.append(
            f"MoeConfig.device={config.device!r} differs from the model's "
            f"device {str(model_device)!r}: expert slabs will be resident on "
            "the configured device. Set MoeConfig.device to the model's "
            "compute device."
        )

    # 1-2. Classify, then freeze the base. QLoRA trains adapters only; the
    # router is a base parameter and stays frozen + unadapted (README §1).
    params_classified: dict[str, int] = {"expert": 0, "dense": 0, "pinned": 0}
    for name, param in model.named_parameters():
        params_classified[arch.classify_parameter(name)] += 1
        param.requires_grad = False

    # 3-6. Quantize -> residency -> managed blocks -> adapters (A/B
    # backends). Quantization reads the fused expert parameters that block
    # installation drops, so it must come first; adapters attach to the
    # managed blocks, so they come last.
    store = backends.quantize_experts(model, arch)
    residency = backends.build_residency(
        store, config.streamed, config.capacity_bytes, config.look_ahead, config.device
    )
    blocks_installed = backends.install_modules(model, arch, store, residency, on_route)
    backends.attach_adapters(
        model, arch, config.lora_rank, config.lora_alpha, config.seed, config.lora_dropout
    )
    # A decoder that runs its layers as visits (Inkling 2: a loop, a stack of block states
    # between layers) adapts its attention too and streams through poros.moe.inkling2.visits.
    visits = bool(getattr(model, "poros_visits", False))
    if visits:
        model.poros_attach_attention_adapters(
            config.lora_rank, config.lora_alpha, config.lora_dropout, config.seed)
    from poros.moe.modules import ManagedMoeBlock
    from poros.moe.residency import ExpertBlockManager
    from poros.moe.store import MoeExpertStore

    # The receipts ran the built-in NF4 store and residency; a caller's own
    # backends may differ, so only these can earn validated_bitwise.
    real = isinstance(store, MoeExpertStore) and isinstance(residency, ExpertBlockManager)
    for module in model.modules():
        if isinstance(module, ManagedMoeBlock):
            module.expert_batch = config.expert_batch
            if config.expert_grouping not in ("pow2", "single"):
                raise ValueError("MoeConfig.expert_grouping must be 'pow2' or 'single', got "
                                 f"{config.expert_grouping!r}")
            module.expert_grouping = config.expert_grouping

    # After fused experts are dropped, the leftover model is small enough
    # to live on the residency device. Real checkpoints load on CPU (30B
    # BF16 does not fit a 24/48 GB card); config.device is where forward
    # runs. Dense streaming keys off that device, not cuda.is_available(),
    # so a CPU-resident model on a GPU host does not trip patch_hf_model.
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MoeConfig.device='cuda' but torch.cuda.is_available() is False")
        model.to(target)
        model_device = next(model.parameters()).device
    _move_trainable_to_model_device(model, model_device)

    # 7. Determinism (INV-3, import-only reuse) covers both configurations:
    # the resident reference needs it too (the expert unsort scatter and the
    # adapter gathers' backward take CUDA's deterministic kernels only then).
    # Dense streaming composes only where the dense engine's bitwise parity
    # is validated — CUDA; on CPU the dense parts stay resident and the
    # expert path is the variable under test (C8).
    from poros.core.determinism import enforce_determinism

    # A caller that already enforced determinism (the trainer) chose whether torch fills new
    # buffers (fill_uninitialized_memory, off in training: it changes no value a correct kernel
    # reads); keep that choice instead of turning the fill back on.
    fill = True
    if torch.are_deterministic_algorithms_enabled():
        fill = bool(getattr(torch.utils.deterministic, "fill_uninitialized_memory", True))
    enforce_determinism(fill_uninitialized=fill)
    # Experts left model.parameters() at install; the store kept their dtype.
    # (a visit decoder's own fp32 modules, which its official implementation keeps in fp32
    # whatever the model's dtype, aside)
    fp32_kept = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ()) if visits else ()
    bf16 = (
        real
        and store.compute_dtype == torch.bfloat16
        and all(p.dtype == torch.bfloat16 or any(k in n for k in fp32_kept)
                for n, p in model.named_parameters() if not p.requires_grad)
    )
    dense_streamed = False
    if config.streamed:
        if model_device.type == "cuda" and visits:
            from poros.moe.inkling2.visits import install_visit_streaming

            install_visit_streaming(model, config.dense_block_size or 1, dense_managed_filter(arch))
            dense_streamed = True
        elif model_device.type == "cuda":
            _stream_dense_params(model, arch, config.dense_block_size)
            dense_streamed = True
        else:
            warnings.append(_CPU_DENSE_WARNING)
    if visits and not dense_streamed:  # the same visit boundaries, the weights resident
        from poros.moe.inkling2.visits import install_visits

        install_visits(model)
    # Streamed dense blocks run each MoE layer twice per step (block forward, then
    # the backward rematerialization); reuse the first pass's verified output.
    if config.expert_compute not in ("reference", "grouped", "fused", "auto"):
        raise ValueError(
            "MoeConfig.expert_compute must be 'reference', 'grouped', 'fused' or 'auto', "
            f"got {config.expert_compute!r}"
        )
    if config.expert_compute == "grouped" and model_device.type == "cuda":
        from poros.moe import grouped

        grouped.activate()
    for module in model.modules():
        if isinstance(module, ManagedMoeBlock):
            # a visit decoder reruns a layer several times a forward, its backward in reverse
            # visit order: the reuse queue (one forward, then its remat) does not apply there
            module.reuse_remat_output = (dense_streamed and config.reuse_remat_output
                                         and not visits)
            if (
                config.expert_compute == "grouped"
                and getattr(module, "expert_group", None) is not None
            ):
                raise ValueError(
                    "expert_compute='grouped' needs fused gate_up/down experts; "
                    f"{type(module).__name__} brings its own group compute (a store "
                    "plugin's block): use its fused mode, or expert_compute='reference'"
                )
            if config.expert_compute == "grouped" and getattr(module, "act_takes_up", False):
                raise ValueError(
                    "expert_compute='grouped' computes act(gate) * up; this model's experts "
                    "take act(gate, up): use expert_compute='reference'"
                )
            module.expert_compute = config.expert_compute
    compute, compute_reason = config.expert_compute, None
    if compute in ("fused", "auto"):
        from poros.moe.fused import model_reason

        why = model_reason(model)
        if why is not None and compute == "fused":
            raise ValueError(f"expert_compute='fused' cannot run this model: {why}")
        compute = "fused" if why is None else "reference"
        compute_reason = None if why is None else f"the fused expert compute cannot run {why}"
        for module in model.modules():
            if isinstance(module, ManagedMoeBlock):
                module.expert_compute = compute

    model_type = str(registry_type(model.config) or "unknown")
    guarantee: MoeGuarantee
    validated = real and bf16 and model_type in validated_arch_types("moe")
    if not config.streamed:
        guarantee = "resident_reference"
    elif dense_streamed and validated and target.type == model_device.type:
        guarantee = "validated_bitwise"
    else:
        guarantee = "experimental_moe_streamed"
        warnings.insert(0, _EXPERIMENTAL_WARNING)
    report = MoeReport(
        guarantee=guarantee,
        streamed=config.streamed,
        dense_streamed=dense_streamed,
        model_type=model_type,
        experts_quantized=len(arch.moe_layer_indices) * arch.layout.experts_per_layer,
        blocks_installed=blocks_installed,
        params_classified=params_classified,
        warnings=warnings,
        expert_compute=compute,
        expert_compute_reason=compute_reason,
    )
    model._poros_moe_report = report
    logger.info(
        "prepare_moe complete: %d experts quantized, %d blocks installed, "
        "guarantee=%s, dense_streamed=%s",
        report.experts_quantized,
        report.blocks_installed,
        report.guarantee,
        report.dense_streamed,
    )
    return report
