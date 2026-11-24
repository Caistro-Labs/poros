"""The MoE gate suite (README §6): proves the three invariants.

The absolute GPU gate is ``scripts/moe_absolute_gate.py``: a real
``qwen3_moe`` checkpoint, streamed-vs-resident bitwise parity, step time,
and held-out eval CE. These functions are the instrumented probes used by
that script and by tiny-model CUDA checks.

Each gate is a function returning a ``GateResult``. Gates drive
``prepare_moe`` with an injected ``MoeBackends`` bundle.

Gate inventory:

* **G-TRAIN** (C8) — the money gate: streamed-vs-resident bitwise parity of
  per-step losses, grad norms, and final adapter weights.
* **G-QUANT** (C9) — quantize determinism + all-zero slice preservation,
  applied through ``prepare_moe``. NF4 bit-compatibility with the repo's
  dequant path is the real quant backend's contract (B) and activates when
  the real backend is wired; with stubs this gate proves the
  determinism/sparsity half.
* **G-ROUTE** (C10) — identical ``RouteDecision``s at every layer, every step.
* **G-BYTES** (C11) — ``bytes_transferred`` equals the static schedule's
  prediction exactly.
* **G-COVERAGE** (C12) — every selected expert's slice flowed through the
  parity-checked store path.
* **G-NOAPPROX** (C13) — the computed expert set equals the committed
  selection, per layer per step (the pragmatic form of "no route-filtering
  after selection": asserted at runtime on the event trace).
* **G-NOREGRESS** (C14) — documentation only. The no-dense-regression gate
  is the pre-existing suite itself; the orchestrator runs
  ``python3 -m pytest tests -q -m "not gpu"`` at integration, and quarantine
  (zero edits outside ``poros/moe/``) makes a dense regression structurally
  excluded. There is nothing to run here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn

from poros.moe.arch import MoeArch
from poros.moe.contracts import (
    ExpertKey,
    ExpertLayout,
    ExpertResidency,
    ExpertStore,
    RouteDecision,
)
from poros.moe.prepare import MoeBackends, MoeConfig, MoeReport, prepare_moe

__all__ = [
    "GateRecorder",
    "GateResult",
    "InstrumentedHandles",
    "gate_bytes",
    "gate_coverage",
    "gate_noapprox",
    "gate_quant",
    "gate_route",
    "gate_train",
    "instrumented_backends",
]


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome: a pass flag plus human/machine-readable details."""

    gate: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instrumentation: one ordered event log + delegating wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteEvent:
    """A committed router decision (via the ``on_route`` hook)."""

    decision: RouteDecision


@dataclass(frozen=True)
class DequantEvent:
    """One expert tensor dequantized out of a resident slab (compute path)."""

    key: ExpertKey
    tensor_name: str


@dataclass(frozen=True)
class EnsureEvent:
    """One ``ExpertResidency.ensure`` batch (the transfer schedule)."""

    keys: tuple[ExpertKey, ...]


GateEvent = RouteEvent | DequantEvent | EnsureEvent


class GateRecorder:
    """Ordered event log shared by the recording wrappers.

    Route decisions, dequantize calls, and ensure batches land in ONE ordered
    log so the gates can align what the router committed with what the
    compute path actually did.
    """

    def __init__(self) -> None:
        self.events: list[GateEvent] = []

    def route(self, decision: RouteDecision) -> None:
        """Record a committed decision; pass directly as ``on_route``."""
        self.events.append(RouteEvent(decision))

    def dequant(self, key: ExpertKey, tensor_name: str) -> None:
        self.events.append(DequantEvent(key, tensor_name))

    def ensure(self, keys: Sequence[ExpertKey]) -> None:
        self.events.append(EnsureEvent(tuple(keys)))

    @property
    def decisions(self) -> list[RouteDecision]:
        return [e.decision for e in self.events if isinstance(e, RouteEvent)]


class _RecordingExpertStore:
    """``ExpertStore`` delegation wrapper recording dequantize calls."""

    def __init__(self, inner: ExpertStore, recorder: GateRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        # Delegate concrete-store extras the protocol does not freeze (e.g.
        # ``MoeExpertStore.tensor_shapes``, read by ``install_managed_blocks``).
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    @property
    def layout(self) -> ExpertLayout:
        return self._inner.layout

    def slice_nbytes(self, key: ExpertKey) -> int:
        return self._inner.slice_nbytes(key)

    def load_slice(self, key: ExpertKey, dst: torch.Tensor) -> None:
        self._inner.load_slice(key, dst)

    def dequantize_tensor(
        self, key: ExpertKey, tensor_name: str, slab: torch.Tensor
    ) -> torch.Tensor:
        self._recorder.dequant(key, tensor_name)
        return self._inner.dequantize_tensor(key, tensor_name, slab)

    def dequantize_experts(
        self, keys: Sequence[ExpertKey], tensor_name: str, slabs: torch.Tensor
    ) -> torch.Tensor:
        for key in keys:
            self._recorder.dequant(key, tensor_name)
        return self._inner.dequantize_experts(keys, tensor_name, slabs)

    def dequantize_resident(
        self, keys: Sequence[ExpertKey], tensor_name: str, slabs: Sequence[torch.Tensor]
    ) -> torch.Tensor | None:
        resident = getattr(self._inner, "dequantize_resident", None)
        out = None if resident is None else resident(keys, tensor_name, slabs)
        if out is not None:  # recorded only when it decodes (else dequantize_experts records)
            for key in keys:
                self._recorder.dequant(key, tensor_name)
        return out


class _RecordingExpertResidency:
    """``ExpertResidency`` delegation wrapper recording ensure batches."""

    def __init__(self, inner: ExpertResidency, recorder: GateRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def ensure(self, keys: Sequence[ExpertKey]) -> None:
        self._recorder.ensure(keys)
        self._inner.ensure(keys)

    def resident_qbytes(self, key: ExpertKey) -> torch.Tensor:
        return self._inner.resident_qbytes(key)

    @property
    def bytes_transferred(self) -> int:
        return self._inner.bytes_transferred


@dataclass
class InstrumentedHandles:
    """Every wrapper created by ``instrumented_backends``, in creation order."""

    stores: list[ExpertStore] = field(default_factory=list)
    residencies: list[ExpertResidency] = field(default_factory=list)


def instrumented_backends(
    backends: MoeBackends, recorder: GateRecorder
) -> tuple[MoeBackends, InstrumentedHandles]:
    """Wrap a backend bundle so stores/residencies record into ``recorder``.

    The returned bundle is a drop-in ``MoeBackends``; the handles expose the
    recording wrappers (which delegate every protocol member) for gates that
    also need byte accounting or slab access.
    """
    handles = InstrumentedHandles()

    def _quantize(model: nn.Module, arch: MoeArch) -> ExpertStore:
        store = _RecordingExpertStore(backends.quantize_experts(model, arch), recorder)
        handles.stores.append(store)
        return store

    def _build(
        store: ExpertStore,
        streamed: bool,
        capacity_bytes: int,
        look_ahead: int,
        device: str,
    ) -> ExpertResidency:
        # `store` arrives as the recording wrapper from _quantize; the real
        # residency binds the inner store (load_slice is a transfer, not a
        # recorded compute event — only dequantize calls are).
        inner_store = store
        if isinstance(store, _RecordingExpertStore):
            inner_store = store._inner
        residency = backends.build_residency(
            inner_store, streamed, capacity_bytes, look_ahead, device
        )
        wrapped = _RecordingExpertResidency(residency, recorder)
        handles.residencies.append(wrapped)
        return wrapped

    return replace(backends, quantize_experts=_quantize, build_residency=_build), handles


# ---------------------------------------------------------------------------
# Shared training driver (C8's "plain SGD on adapter params only")
# ---------------------------------------------------------------------------


@dataclass
class _TrainTrace:
    losses: list[torch.Tensor]
    grad_norms: list[torch.Tensor]
    final_adapters: dict[str, torch.Tensor]


def _train_adapter_steps(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    steps: int,
    lr: float,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> _TrainTrace:
    """Run ``steps`` plain-SGD optimizer steps on the trainable params only."""
    inputs: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": input_ids if labels is None else labels,
    }
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "no trainable parameters after prepare_moe — adapters were not attached"
        )
    opt = torch.optim.SGD(params, lr=lr)
    losses: list[torch.Tensor] = []
    norms: list[torch.Tensor] = []
    route_blocks = [m for m in model.modules() if getattr(m, "on_route", None) is not None]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(**inputs)
        # Dense blockwise remat re-enters ManagedMoeBlock.forward during
        # backward. That is recomputation, not a new committed route; mute
        # observers so G-ROUTE / G-NOAPPROX count forward decisions only.
        observers = [m.on_route for m in route_blocks]
        for block in route_blocks:
            block.on_route = None
        try:
            out.loss.backward()
        finally:
            for block, observer in zip(route_blocks, observers, strict=True):
                block.on_route = observer
        with torch.no_grad():
            sq = torch.stack([p.grad.pow(2).sum() for p in params])
            norms.append(torch.sqrt(sq.sum()))
        losses.append(out.loss.detach().clone())
        opt.step()
    final = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    return _TrainTrace(losses, norms, final)


def _prepare_and_train(
    build_model: Callable[[], nn.Module],
    input_ids: torch.Tensor,
    *,
    backends: MoeBackends,
    config: MoeConfig,
    steps: int,
    lr: float,
    on_route: Callable[[RouteDecision], None] | None = None,
) -> tuple[MoeReport, _TrainTrace]:
    model = build_model()
    report = prepare_moe(model, config, backends=backends, on_route=on_route)
    return report, _train_adapter_steps(model, input_ids, steps=steps, lr=lr)


# ---------------------------------------------------------------------------
# G-TRAIN (C8) — the money gate
# ---------------------------------------------------------------------------


def gate_train(
    build_model: Callable[[], nn.Module],
    input_ids: torch.Tensor,
    *,
    backends: MoeBackends,
    steps: int = 4,
    seed: int = 1234,
    lr: float = 0.05,
) -> GateResult:
    """G-TRAIN: expert streaming is bitwise-identical to expert residency.

    Two ``prepare_moe`` runs — one ``streamed=True``, one ``streamed=False``
    — same seed, ``steps`` optimizer steps on the synthetic batch. Per-step
    losses, grad norms, and final adapter weights must be bitwise equal
    (``torch.equal``; 0.00e+00, never a tolerance).

    The streamed run's residency budget is exactly one layer's slices, so
    every layer switch evicts and every backward rematerializes — the parity
    claim is proven with the streaming machinery actually firing, not with
    everything silently fitting.

    On CPU the dense engine stays resident in BOTH runs (its streamed parity
    is validated on CUDA, not CPU — measured drift ~5e-7), so the only
    toggling variable is expert residency (workstream-c.md C8);
    ``details["dense_streamed"]`` records exactly which configuration ran.
    """
    # Probe: one throwaway prepare to size one MoE layer's slices.
    probe_recorder = GateRecorder()
    probe_bundle, probe_handles = instrumented_backends(backends, probe_recorder)
    prepare_moe(
        build_model(), MoeConfig(streamed=False, seed=seed), backends=probe_bundle
    )
    probe_store = probe_handles.stores[-1]
    one_layer_bytes = sum(
        probe_store.slice_nbytes((0, expert))
        for expert in range(probe_store.layout.experts_per_layer)
    )

    configs = {
        "streamed": MoeConfig(
            streamed=True, capacity_bytes=one_layer_bytes, seed=seed
        ),
        "resident": MoeConfig(streamed=False, seed=seed),
    }
    reports: dict[str, MoeReport] = {}
    traces: dict[str, _TrainTrace] = {}
    for label, config in configs.items():
        reports[label], traces[label] = _prepare_and_train(
            build_model,
            input_ids,
            backends=backends,
            config=config,
            steps=steps,
            lr=lr,
        )
    s, r = traces["streamed"], traces["resident"]

    losses_equal = all(torch.equal(a, b) for a, b in zip(s.losses, r.losses, strict=True))
    norms_equal = all(
        torch.equal(a, b) for a, b in zip(s.grad_norms, r.grad_norms, strict=True)
    )
    names_equal = set(s.final_adapters) == set(r.final_adapters)
    weights_equal = names_equal and all(
        torch.equal(s.final_adapters[name], r.final_adapters[name])
        for name in s.final_adapters
    )
    max_loss_diff = max(
        (float((a - b).abs().max()) for a, b in zip(s.losses, r.losses, strict=True)),
        default=0.0,
    )
    passed = losses_equal and norms_equal and weights_equal
    details = {
        "steps": steps,
        "streamed_capacity_bytes": one_layer_bytes,
        "losses_streamed": [float(x) for x in s.losses],
        "losses_resident": [float(x) for x in r.losses],
        "max_abs_loss_diff": max_loss_diff,
        "losses_bitwise_equal": losses_equal,
        "grad_norms_bitwise_equal": norms_equal,
        "adapter_weights_bitwise_equal": weights_equal,
        "adapter_params_compared": len(s.final_adapters),
        "dense_streamed": reports["streamed"].dense_streamed,
        "guarantee_streamed": reports["streamed"].guarantee,
        "guarantee_resident": reports["resident"].guarantee,
    }
    summary = (
        f"streamed-vs-resident over {steps} steps: max |loss diff| = "
        f"{max_loss_diff:.2e}; losses={losses_equal}, grad_norms={norms_equal}, "
        f"adapter_weights={weights_equal} "
        f"(dense_streamed={reports['streamed'].dense_streamed})"
    )
    return GateResult("G-TRAIN", passed, summary, details)


# ---------------------------------------------------------------------------
# G-QUANT (C9)
# ---------------------------------------------------------------------------


def gate_quant(
    build_model: Callable[[], nn.Module],
    *,
    backends: MoeBackends,
    seed: int = 1234,
) -> GateResult:
    """G-QUANT through ``prepare_moe``: determinism + zero-slice preservation.

    Two identically-seeded prepares must yield byte-identical slabs. Sparse
    preservation is checked on the dequantized side: NF4 encodes an all-zero
    slice as nonzero bytes (codebook indices + absmax), so raw slab bytes are
    never the signal — a slice counts as preserved exactly when every one of
    its tensors dequantizes to exact zeros. The fixture bakes one all-zero
    expert slice, so at least one must be found; a densifying quantizer
    would turn it nonzero and fail the gate.
    """
    recorder = GateRecorder()
    bundle, handles = instrumented_backends(backends, recorder)
    for _ in range(2):
        prepare_moe(build_model(), MoeConfig(streamed=False, seed=seed), backends=bundle)
    store_a, store_b = handles.stores
    arch = MoeArch.from_config(build_model().config)

    keys = arch.expert_keys()
    deterministic = True
    zero_slices = 0
    for key in keys:
        nbytes = store_a.slice_nbytes(key)
        if nbytes != store_b.slice_nbytes(key):
            deterministic = False
            continue
        slab_a = torch.empty(nbytes, dtype=torch.uint8)
        slab_b = torch.empty(nbytes, dtype=torch.uint8)
        store_a.load_slice(key, slab_a)
        store_b.load_slice(key, slab_b)
        if not torch.equal(slab_a, slab_b):
            deterministic = False
        tensors = [
            store_a.dequantize_tensor(key, name, slab_a)
            for name in store_a.layout.tensor_names
        ]
        if all(not bool(t.any()) for t in tensors):  # every element exactly 0
            zero_slices += 1
    passed = deterministic and zero_slices >= 1
    details = {
        "keys_compared": len(keys),
        "deterministic_bytes": deterministic,
        "zero_slices_found": zero_slices,
    }
    summary = (
        f"{len(keys)} expert slabs byte-identical across prepares: {deterministic}; "
        f"{zero_slices} slices dequantize to exact zeros (sparse preservation)"
    )
    return GateResult("G-QUANT", passed, summary, details)


# ---------------------------------------------------------------------------
# G-ROUTE (C10)
# ---------------------------------------------------------------------------


def gate_route(
    build_model: Callable[[], nn.Module],
    input_ids: torch.Tensor,
    *,
    backends: MoeBackends,
    steps: int = 2,
    seed: int = 1234,
    lr: float = 0.05,
) -> GateResult:
    """G-ROUTE: identical committed RouteDecisions, streamed vs resident.

    Captures decisions via the ``on_route`` hook in both runs and asserts
    the sequences are exactly equal — layer, selection order, and weights —
    at every layer, every step (implied by G-TRAIN; kept separate for
    debuggability).
    """
    decisions: dict[str, list[RouteDecision]] = {}
    for label, streamed in (("streamed", True), ("resident", False)):
        log: list[RouteDecision] = []
        _prepare_and_train(
            build_model,
            input_ids,
            backends=backends,
            config=MoeConfig(streamed=streamed, seed=seed),
            steps=steps,
            lr=lr,
            on_route=log.append,
        )
        decisions[label] = log
    s, r = decisions["streamed"], decisions["resident"]
    identical = len(s) == len(r) and all(a == b for a, b in zip(s, r, strict=True))
    details = {
        "decisions_streamed": len(s),
        "decisions_resident": len(r),
        "identical": identical,
    }
    summary = (
        f"{len(s)} vs {len(r)} route decisions over {steps} steps; "
        f"identical: {identical}"
    )
    return GateResult("G-ROUTE", identical, summary, details)


def _ensure_schedule_ok(
    batches: Sequence[tuple[ExpertKey, ...]], *, num_layers: int, steps: int
) -> bool:
    """True if ``batches`` follow fwd-ascending then bwd-descending ordinals.

    Each expected layer slot consumes one batch. Extra immediate repeats of
    the current layer (dense remat) are eaten only after leaving one batch
    per remaining consecutive expected slot of that layer. That keeps the
    last-layer fwd/bwd turn and the step-boundary L0/L0 (possibly different
    keys after SGD) from collapsing. ExpertKey.layer is the MoE ordinal.
    """
    if num_layers < 1 or not batches:
        return False
    expected: list[int] = []
    for _ in range(steps):
        expected.extend(range(num_layers))
        expected.extend(range(num_layers - 1, -1, -1))
    i = 0
    for k, layer in enumerate(expected):
        if i >= len(batches):
            return False
        batch = batches[i]
        experts = [key[1] for key in batch]
        if (
            not batch
            or batch[0][0] != layer
            or len({key[0] for key in batch}) != 1
            or experts != sorted(set(experts))
        ):
            return False
        i += 1
        remaining_needed = 0
        j = k + 1
        while j < len(expected) and expected[j] == layer:
            remaining_needed += 1
            j += 1
        run_rest = 0
        t = i
        while t < len(batches) and batches[t] and batches[t][0][0] == layer:
            run_rest += 1
            t += 1
        extras_here = run_rest - remaining_needed
        if extras_here < 0:
            return False
        for _ in range(extras_here):
            extra = batches[i]
            extra_experts = [key[1] for key in extra]
            if (
                extra != batch
                or len({key[0] for key in extra}) != 1
                or extra_experts != sorted(set(extra_experts))
            ):
                return False
            i += 1
    return i == len(batches)


# ---------------------------------------------------------------------------
# G-BYTES (C11)
# ---------------------------------------------------------------------------


def gate_bytes(
    build_model: Callable[[], nn.Module],
    input_ids: torch.Tensor,
    *,
    backends: MoeBackends,
    steps: int = 2,
    seed: int = 1234,
    lr: float = 0.05,
) -> GateResult:
    """G-BYTES: ``bytes_transferred`` equals the first-touch prediction.

    Real-backend schedule (workstreams A+B): per optimizer step, each MoE
    layer ensures its selected-expert union once in forward (layers
    ascending) and once more in backward rematerialization (layers
    descending, same key set — the saved top-k indices are reused). With
    ``look_ahead=0`` and an unbounded residency budget, every distinct slice
    transfers exactly once, on first touch, so ``bytes_transferred`` must
    equal Σ ``slice_nbytes`` over the union of all recorded ensure batches.
    Eviction-under-pressure accounting is covered by A's unit tests; parity
    under eviction is G-TRAIN's job (one-layer budget).
    """
    recorder = GateRecorder()
    bundle, handles = instrumented_backends(backends, recorder)
    _prepare_and_train(
        build_model,
        input_ids,
        backends=bundle,
        config=MoeConfig(streamed=True, look_ahead=0, seed=seed),
        steps=steps,
        lr=lr,
    )
    store = handles.stores[-1]
    residency = handles.residencies[-1]
    arch = MoeArch.from_config(build_model().config)
    recorded = [e.keys for e in recorder.events if isinstance(e, EnsureEvent)]
    structure_ok = _ensure_schedule_ok(
        recorded, num_layers=arch.layout.num_layers, steps=steps
    )
    first_touch = {key for batch in recorded for key in batch}
    expected_bytes = sum(store.slice_nbytes(key) for key in first_touch)
    actual_bytes = residency.bytes_transferred
    passed = structure_ok and actual_bytes == expected_bytes
    details = {
        "steps": steps,
        "ensure_batches_recorded": len(recorded),
        "schedule_structure_ok": structure_ok,
        "distinct_slices_touched": len(first_touch),
        "expected_bytes": expected_bytes,
        "bytes_transferred": actual_bytes,
    }
    summary = (
        f"bytes_transferred={actual_bytes} vs first-touch prediction "
        f"{expected_bytes} over {len(first_touch)} distinct slices; "
        f"schedule structure ok: {structure_ok}"
    )
    return GateResult("G-BYTES", passed, summary, details)


# ---------------------------------------------------------------------------
# G-COVERAGE (C12) + G-NOAPPROX (C13)
# ---------------------------------------------------------------------------


def _selected_and_computed(
    recorder: GateRecorder,
) -> tuple[set[ExpertKey], set[ExpertKey]]:
    selected: set[ExpertKey] = set()
    computed: set[ExpertKey] = set()
    for event in recorder.events:
        if isinstance(event, RouteEvent):
            selected |= {(event.decision.layer, e) for e in event.decision.expert_ids}
        elif isinstance(event, DequantEvent):
            computed.add(event.key)
    return selected, computed


def gate_coverage(
    build_model: Callable[[], nn.Module],
    input_ids: torch.Tensor,
    *,
    backends: MoeBackends,
    steps: int = 2,
    seed: int = 1234,
    lr: float = 0.05,
) -> GateResult:
    """G-COVERAGE: every selected expert flowed through the store path.

    The union of router-selected experts across the run must be non-empty
    and equal the set of experts whose slices were actually dequantized out
    of the ExpertStore (the parity-checked path) — count and identity.
    """
    recorder = GateRecorder()
    bundle, _ = instrumented_backends(backends, recorder)
    _prepare_and_train(
        build_model,
        input_ids,
        backends=bundle,
        config=MoeConfig(streamed=True, seed=seed),
        steps=steps,
        lr=lr,
        on_route=recorder.route,
    )
    selected, computed = _selected_and_computed(recorder)
    passed = bool(selected) and selected == computed
    details = {
        "selected_experts": len(selected),
        "computed_experts": len(computed),
        "counts_match": len(selected) == len(computed),
        "sets_match": selected == computed,
    }
    summary = (
        f"{len(selected)} distinct experts selected; {len(computed)} flowed "
        f"through the parity-checked store path; match: {selected == computed}"
    )
    return GateResult("G-COVERAGE", passed, summary, details)


def gate_noapprox(
    build_model: Callable[[], nn.Module],
    input_ids: torch.Tensor,
    *,
    backends: MoeBackends,
    steps: int = 2,
    seed: int = 1234,
    lr: float = 0.05,
) -> GateResult:
    """G-NOAPPROX: computed expert set == committed selection, always.

    Per MoE layer, each ``RouteDecision`` opens a group; dequantize calls
    join their layer's open group (a backward rematerialization re-joins the
    same group — it recomputes the same committed experts). Any drop,
    substitute, or extra-for-compute expert breaks set equality. This is the
    runtime form of the no-route-filtering-after-selection scan (C13).
    """
    recorder = GateRecorder()
    bundle, _ = instrumented_backends(backends, recorder)
    model = build_model()
    prepare_moe(
        model,
        MoeConfig(streamed=True, seed=seed),
        backends=bundle,
        on_route=recorder.route,
    )
    _train_adapter_steps(model, input_ids, steps=steps, lr=lr)
    arch = MoeArch.from_config(model.config)

    selected_groups: dict[int, list[set[int]]] = {}
    computed_groups: dict[int, list[set[int]]] = {}
    orphan_dequants = 0
    for event in recorder.events:
        if isinstance(event, RouteEvent):
            layer = event.decision.layer
            selected_groups.setdefault(layer, []).append(set(event.decision.expert_ids))
            computed_groups.setdefault(layer, []).append(set())
        elif isinstance(event, DequantEvent):
            layer, expert = event.key
            open_groups = computed_groups.get(layer)
            if open_groups:
                open_groups[-1].add(expert)
            else:
                orphan_dequants += 1

    groups_ok = orphan_dequants == 0 and selected_groups.keys() == computed_groups.keys()
    mismatches = 0
    if groups_ok:
        for layer in selected_groups:
            s_groups, c_groups = selected_groups[layer], computed_groups[layer]
            if len(s_groups) != len(c_groups):
                groups_ok = False
                continue
            mismatches += sum(
                1
                for s, c in zip(s_groups, c_groups, strict=True)
                if s != c
            )
    expected_groups = steps * len(arch.moe_layer_indices)
    total_groups = sum(len(g) for g in selected_groups.values())
    passed = groups_ok and mismatches == 0 and total_groups == expected_groups
    details = {
        "route_groups": total_groups,
        "expected_groups": expected_groups,
        "mismatched_groups": mismatches,
        "orphan_dequants": orphan_dequants,
    }
    summary = (
        f"{total_groups} layer-step groups checked; {mismatches} computed-set "
        f"mismatches; {orphan_dequants} dequants without a committed selection"
    )
    return GateResult("G-NOAPPROX", passed, summary, details)
