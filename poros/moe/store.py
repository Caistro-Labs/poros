"""The expert-slab store: the quantized experts ``poros.moe.modules`` computes with.

``MoeExpertStore`` owns the quantized slabs, read through a
``poros.sss.SlabSource`` (host RAM, or a ``.sss`` file on an SSD). Expert slab layout
the per-tensor blobs (``poros.moe.quant`` format) concatenated
in ``tensor_names`` order — a byte permutation of the fused source,
bit-exact-safe by construction. The store alone knows the per-tensor
offsets, which derive from the shapes alone. ``ExpertSlabArena`` holds a layer's slabs on the
device; ``decode_experts`` and the per-pass dequant cache (``EXPERT_DQ_CACHE``) turn them back
into weights.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import torch

from poros.moe import quant
from poros.moe.contracts import (
    ExpertKey,
    ExpertLayout,
    ExpertStore,
)
from poros.sss.format import RamSink, SlabSink
from poros.sss.source import RamSource, SlabSource

logger = logging.getLogger(__name__)


def slab_layout(
    layout: ExpertLayout, tensor_shapes: Mapping[str, Sequence[int]]
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, int]], int]:
    """Per-tensor shapes, ``[start, end)`` byte offsets and the slab size.

    A slab holds the ``poros.moe.quant`` blobs of ``layout.tensor_names`` in
    that order; the offsets derive from the shapes alone.
    """
    shapes = {name: tuple(tensor_shapes[name]) for name in layout.tensor_names}
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for name in layout.tensor_names:
        size = quant.quantized_nbytes(shapes[name])
        offsets[name] = (start, start + size)
        start += size
    return shapes, offsets, start


SinkFactory = Callable[[list[tuple[ExpertKey, int]]], SlabSink]


class ExpertSlabArena:
    """The slabs an ``MoeExpertStore`` is built from, filled per tensor.

    One flat uint8 slab per (layer, expert), written through a
    ``poros.sss.SlabSink``: by default a ``RamSink`` (preallocated host
    rows, pinned in place when the platform supports it -- pinning finished
    slabs would briefly double them), or e.g. an ``SssWriter`` for the SSD
    store. ``sink`` is a factory receiving every ``(key, nbytes)``. A slab
    holds the ``poros.moe.quant`` blobs of ``layout.tensor_names`` at fixed
    offsets. ``write`` quantizes one expert's tensor straight
    into its slab, so a builder holds at most that tensor in its source
    dtype.
    """

    def __init__(
        self,
        layout: ExpertLayout,
        tensor_shapes: Mapping[str, Sequence[int]],
        *,
        sink: SinkFactory | None = None,
        quant_device: torch.device | str | None = None,
    ) -> None:
        self._layout = layout
        self._quant_device = quant_device  # where quantize_tensor's NF4 level runs (None: CPU)
        self._tensor_shapes, self._offsets, self._slice_nbytes = slab_layout(
            layout, tensor_shapes
        )
        keys = [
            (layer, expert)
            for layer in range(layout.num_layers)
            for expert in range(layout.experts_per_layer)
        ]
        self._unwritten = {(key, name) for key in keys for name in layout.tensor_names}
        slabs = [(key, self._slice_nbytes) for key in keys]
        self._sink = sink(slabs) if sink is not None else RamSink(slabs)

    @property
    def slice_nbytes(self) -> int:
        return self._slice_nbytes

    def write(self, key: ExpertKey, tensor_name: str, weight: torch.Tensor) -> None:
        """Quantize one expert's ``tensor_name`` weight into its slab."""
        if tuple(weight.shape) != self._tensor_shapes[tensor_name]:
            raise ValueError(
                f"{tensor_name} of expert {key} has shape {tuple(weight.shape)}, "
                f"expected {self._tensor_shapes[tensor_name]}"
            )
        start, _ = self._offsets[tensor_name]
        blob = None
        if self._quant_device is not None:
            try:
                blob = quant.quantize_tensor(weight, self._quant_device)
            except (torch.OutOfMemoryError, RuntimeError) as exc:  # the CPU makes the same bytes
                logger.warning("NF4 quantization on %s failed (%s); the CPU quantizes the rest",
                               self._quant_device, str(exc).splitlines()[0][:200])
                self._quant_device = None
        if blob is None:
            blob = quant.quantize_tensor(weight)
        self._sink.write(key, start, blob)
        self._unwritten.discard((key, tensor_name))

    def build(self, compute_dtype: torch.dtype) -> MoeExpertStore:
        """The immutable store over the sink's source; every tensor must be written."""
        if self._unwritten:
            key, name = min(self._unwritten)
            raise ValueError(
                f"{len(self._unwritten)} expert tensor(s) never written, e.g. {name} of {key}"
            )
        return MoeExpertStore(
            self._sink.finish(),
            self._layout,
            self._tensor_shapes,
            self._offsets,
            self._slice_nbytes,
            compute_dtype,
        )

    def abort(self) -> None:
        """Drop a partial build (an ``SssWriter`` leaves no file behind)."""
        abort = getattr(self._sink, "abort", None)
        if abort is not None:
            abort()


class MoeExpertStore(ExpertStore):
    """Owner of the quantized expert slabs, read through a ``SlabSource``.

    Built once through an ``ExpertSlabArena`` (from a HF model's fused
    expert state dict, or per checkpoint tensor by ``poros.moe.load``),
    one flat uint8 slab per (layer, expert); immutable after build, so
    a slice fetched twice is bit-identical both times (backward
    rematerialization relies on this). The bytes live in a
    ``poros.sss.SlabSource``: pinned host RAM (``RamSource``, the default)
    or a ``.sss`` file streamed from an SSD (``SsdSource``). ``load_slice``
    is a pure byte copy either way, so the source cannot change numerics.
    """

    def __init__(
        self,
        source: SlabSource | dict[ExpertKey, torch.Tensor],
        layout: ExpertLayout,
        tensor_shapes: dict[str, tuple[int, ...]],
        tensor_offsets: dict[str, tuple[int, int]],
        slice_nbytes: int,
        compute_dtype: torch.dtype,
    ) -> None:
        """Assemble the store from built parts; use ``from_state_dict`` or an arena."""
        if isinstance(source, dict):  # host slabs, as before SSS
            source = RamSource(source)
        self._source = source
        self._layout = layout
        self._tensor_shapes = tensor_shapes
        self._tensor_offsets = tensor_offsets
        self._slice_nbytes = slice_nbytes
        self._compute_dtype = compute_dtype

    @classmethod
    def over_source(
        cls,
        source: SlabSource,
        layout: ExpertLayout,
        tensor_shapes: Mapping[str, Sequence[int]],
        compute_dtype: torch.dtype,
    ) -> MoeExpertStore:
        """A store over slabs that already exist (e.g. a reused ``.sss`` file).

        Every key of the layout must be in ``source`` with the slab size the
        layout implies; raises otherwise.
        """
        shapes, offsets, nbytes = slab_layout(layout, tensor_shapes)
        have = set(source.keys())
        for layer in range(layout.num_layers):
            for expert in range(layout.experts_per_layer):
                key = (layer, expert)
                if key not in have or source.nbytes(key) != nbytes:
                    raise ValueError(
                        f"source does not hold expert slab {key} of {nbytes} bytes"
                    )
        return cls(source, layout, shapes, offsets, nbytes, compute_dtype)

    @property
    def source(self) -> SlabSource:
        """Where the slab bytes come from (``RamSource``, ``SsdSource``, ...)."""
        return self._source

    @property
    def _slabs(self) -> dict[ExpertKey, torch.Tensor]:
        """The host slabs of a RAM-backed store (tests and tooling)."""
        if not isinstance(self._source, RamSource):
            raise TypeError(f"{type(self._source).__name__} holds no host slabs")
        return dict(self._source.items())

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        *,
        num_layers: int,
        experts_per_layer: int,
        top_k: int,
        key_format: str = "model.layers.{layer}.mlp.experts.{tensor}",
        tensor_names: tuple[str, ...] = ("gate_up_proj", "down_proj"),
        moe_layer_ids: Sequence[int] | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> MoeExpertStore:
        """Build the store from fused ``[num_experts, ...]`` expert tensors.

        ``key_format`` locates one MoE layer's fused tensor in ``state``;
        ``{layer}`` is the decoder-layer index and ``{tensor}`` the tensor
        name. ``moe_layer_ids`` maps MoE-layer ordinals (the contract's
        ``layer_index``) to decoder-layer indices for sparse-step archs;
        the default ``range(num_layers)`` matches Qwen3-MoE, where every
        decoder layer is MoE. ``compute_dtype`` is the dequantization
        (compute) dtype — the repo's NF4 compute dtype is bf16.
        """
        layer_ids = list(moe_layer_ids) if moe_layer_ids is not None else list(range(num_layers))
        if len(layer_ids) != num_layers:
            raise ValueError(
                f"moe_layer_ids has {len(layer_ids)} entries for num_layers={num_layers}"
            )
        tensor_shapes: dict[str, tuple[int, ...]] = {}
        fused_tensors: dict[tuple[int, str], torch.Tensor] = {}
        for name in tensor_names:
            for ordinal, decoder_idx in enumerate(layer_ids):
                key = key_format.format(layer=decoder_idx, tensor=name)
                fused = state.get(key)
                if fused is None:
                    raise KeyError(
                        f"expert tensor {key!r} not in the state dict; check key_format "
                        f"({key_format!r}) against the model's parameter names"
                    )
                if fused.dim() < 2 or fused.shape[0] != experts_per_layer:
                    raise ValueError(
                        f"{key!r} must be a fused [{experts_per_layer}, ...] tensor, "
                        f"got shape {tuple(fused.shape)}"
                    )
                per_expert = tuple(fused.shape[1:])
                previous = tensor_shapes.setdefault(name, per_expert)
                if previous != per_expert:
                    raise ValueError(
                        f"{key!r} per-expert shape {per_expert} differs from an earlier "
                        f"layer's {previous}; fused expert tensors must be uniform"
                    )
                fused_tensors[(ordinal, name)] = fused
        layout = ExpertLayout(
            num_layers=num_layers,
            experts_per_layer=experts_per_layer,
            top_k=top_k,
            tensor_names=tuple(tensor_names),
        )
        arena = ExpertSlabArena(layout, tensor_shapes)
        for (ordinal, name), fused in fused_tensors.items():
            for expert in range(experts_per_layer):
                arena.write((ordinal, expert), name, fused[expert])
        return arena.build(compute_dtype)

    @property
    def layout(self) -> ExpertLayout:
        return self._layout

    @property
    def compute_dtype(self) -> torch.dtype:
        """Dtype of dequantized expert weights (the adapter/base compute dtype)."""
        return self._compute_dtype

    @property
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-expert shape of each logical tensor (one expert's slice)."""
        return dict(self._tensor_shapes)

    def slice_nbytes(self, key: ExpertKey) -> int:
        """Quantized bytes of one expert's flat slab (all tensors included)."""
        self._check(key)
        return self._slice_nbytes

    def load_slice(self, key: ExpertKey, dst: torch.Tensor) -> None:
        """Copy one expert's flat slab into ``dst`` — a pure byte copy."""
        self._check(key)
        if dst.dtype != torch.uint8 or dst.numel() != self._slice_nbytes:
            raise ValueError(
                f"dst for {key} must be a uint8 tensor of {self._slice_nbytes} bytes, "
                f"got {dst.dtype} with {dst.numel()} elements"
            )
        self._source.read_into(key, dst)

    def dequantize_tensor(
        self, key: ExpertKey, tensor_name: str, slab: torch.Tensor
    ) -> torch.Tensor:
        """Exactly dequantize one tensor of one expert out of its flat slab.

        Bit-identical to the repo's NF4 dequantization of the same logical
        weights (``poros.moe.quant.dequantize``); the result lives on
        ``slab``'s device in the store's compute dtype.
        """
        return self.dequantize_experts([key], tensor_name, slab.unsqueeze(0))[0]

    def dequantize_experts(
        self, keys: Sequence[ExpertKey], tensor_name: str, slabs: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize one tensor of several experts in one batched call.

        Row ``i`` of ``slabs`` is ``keys[i]``'s slab and row ``i`` of the
        result its tensor, elementwise identical to ``dequantize_tensor``
        per key. A store extension beyond the frozen ``ExpertStore``
        protocol, like ``tensor_shapes``; gate wrappers forward it.
        """
        for key in keys:
            self._check(key)
        if tensor_name not in self._tensor_offsets:
            raise KeyError(
                f"unknown expert tensor {tensor_name!r}; known: {sorted(self._tensor_offsets)}"
            )
        if slabs.dtype != torch.uint8 or slabs.shape != (len(keys), self._slice_nbytes):
            raise ValueError(
                f"slabs for {list(keys)} must be uint8 [{len(keys)}, {self._slice_nbytes}], "
                f"got {slabs.dtype} {tuple(slabs.shape)}"
            )
        start, end = self._tensor_offsets[tensor_name]
        shape = self._tensor_shapes[tensor_name]
        return quant.dequantize(slabs[:, start:end], shape, self._compute_dtype)

    def dequantize_resident(
        self, keys: Sequence[ExpertKey], tensor_name: str, slabs: Sequence[torch.Tensor]
    ) -> torch.Tensor | None:
        """``dequantize_experts`` of slabs where they lie (no stacking), on the GPU decoder.

        ``slabs[i]`` is ``keys[i]``'s resident 1-D slab. Returns the same values bit for bit
        (``poros.moe.kernels``), or None where the decoder does not apply (CPU slabs, no
        Triton, a tensor that is not whole 64-element blocks): the caller then stacks the
        slabs for ``dequantize_experts``.
        """
        if not slabs or not slabs[0].is_cuda or tensor_name not in self._tensor_offsets:
            return None
        shape = self._tensor_shapes[tensor_name]
        try:
            from poros.moe import kernels
        except ImportError:  # no Triton: the torch decoder
            return None
        if not kernels.supports(shape):
            return None
        for key, slab in zip(keys, slabs, strict=True):
            self._check(key)
            if slab.dtype != torch.uint8 or slab.numel() != self._slice_nbytes:
                raise ValueError(f"slab of {key} must be uint8 [{self._slice_nbytes}]")
        out = torch.empty(
            (len(keys), *shape), dtype=self._compute_dtype, device=slabs[0].device
        )
        start, _ = self._tensor_offsets[tensor_name]
        return kernels.dequantize_into(out, kernels.slab_pointers(slabs), start, shape)

    def _check(self, key: ExpertKey) -> None:
        layer, expert = key
        layer_ok = 0 <= layer < self._layout.num_layers
        expert_ok = 0 <= expert < self._layout.experts_per_layer
        if not (layer_ok and expert_ok):
            raise KeyError(
                f"expert key {key} outside the layout "
                f"({self._layout.num_layers} layers x {self._layout.experts_per_layer} experts)"
            )


def decode_experts(block, keys: Sequence[ExpertKey], tensor_name: str) -> torch.Tensor:
    """One tensor of ``keys``' experts in the compute dtype, ``[len(keys), *shape]``, from their
    resident slabs: decoded in place on the GPU where the store can
    (``MoeExpertStore.dequantize_resident``), else stacked for the store's batched dequantize.
    Same values either way."""
    slabs = [block.residency.resident_qbytes(key) for key in keys]
    resident = getattr(block.store, "dequantize_resident", None)
    if resident is not None:
        out = resident(keys, tensor_name, slabs)
        if out is not None:
            return out
    return block.store.dequantize_experts(keys, tensor_name, torch.stack(slabs))


class _ExpertDequantCache:
    """Block-major MoE: each expert tensor dequantized once per layer visit.

    Under block-major accumulation one MoE layer runs for all G microsteps in a
    row (block forward, rematerialization, backward), so the same experts'
    slabs are dequantized again and again. While ``enabled`` this keeps the
    current layer's dequantized tensors (one layer at a time: another layer's
    call clears it): each batched decode whole, for a later call with the same
    experts, and per expert, stacked for a call with other ones.
    ``quant.dequantize`` is row-wise -- one expert's tensor is elementwise
    identical however it is batched -- so every group sees the same bf16
    values, bit for bit.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._layer: tuple | None = None
        self._store: dict = {}
        self._whole: dict = {}
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._layer, self._store, self._whole = None, {}, {}

    def get(self, block, keys, tensor_name) -> torch.Tensor:
        if not self.enabled:
            return decode_experts(block, keys, tensor_name)
        tag = (id(block.store), block.layer_idx)
        if tag != self._layer:
            self._layer, self._store, self._whole = tag, {}, {}
        whole = self._whole.get((tuple(keys), tensor_name))
        if whole is not None:
            self.hits += len(keys)
            return whole
        missing = [i for i, k in enumerate(keys) if (k, tensor_name) not in self._store]
        self.misses += len(missing)
        self.hits += len(keys) - len(missing)
        if len(missing) == len(keys):  # nothing cached: the batched call itself
            out = decode_experts(block, keys, tensor_name)
            for k, w in zip(keys, out):
                self._store[(k, tensor_name)] = w
            self._whole[(tuple(keys), tensor_name)] = out
            return out
        if missing:
            sub = [keys[i] for i in missing]
            for k, w in zip(sub, decode_experts(block, sub, tensor_name)):
                self._store[(k, tensor_name)] = w
        return torch.stack([self._store[(k, tensor_name)] for k in keys])


EXPERT_DQ_CACHE = _ExpertDequantCache()


__all__ = ["ExpertSlabArena", "MoeExpertStore", "slab_layout"]
