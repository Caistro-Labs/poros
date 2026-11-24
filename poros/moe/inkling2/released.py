"""The released expert store: Inkling 2 Mini's experts kept in their released NVFP4 / MXFP8 bytes.

``ExpertCodec.RELEASED`` (poros.moe.inkling2.codec): a run computes with exactly the released
weights, decoded for each use, never rounded again. The store is ``MoeExpertStore``'s, with
other bytes in its slabs:

- one flat uint8 slab per (MoE layer, expert), the expert's tensors in ``tensor_names`` order;
- an NVFP4 tensor ``[rows, cols]``: its fp32 factor (``amax / outer_scale_denominator``, as the
  decode computes it), then its float8 e4m3 scales ``[rows, cols / 16]``, then its codes
  ``[rows, cols / 2]``;
- an MXFP8 tensor: its float8 e4m3 values ``[rows, cols]``, then its exponent bytes
  ``[rows, cols / 32]``.

A layer's format is the checkpoint's (NVFP4 on most layers, MXFP8 on the last few), so slabs
differ in size between layers; every slab of one layer has the same size. The gate / up rows
are de-interleaved in the slab (a row permutation: every scale lies within a row), so the
decoded tensor is the model's ``gate_up_proj``. Decoding is ``codec.decode_nvfp4`` /
``decode_mxfp8``, the same values bit for bit as decoding the checkpoint whole.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from poros.moe.contracts import ExpertKey, ExpertLayout
from poros.moe.inkling2 import codec
from poros.moe.store import MoeExpertStore, SinkFactory
from poros.sss.format import RamSink
from poros.sss.source import SlabSource

__all__ = ["FORMATS", "ReleasedSlabArena", "ReleasedExpertStore", "tensor_nbytes", "pack"]

FORMATS = ("nvfp4", "mxfp8")


def _parts(fmt: str, shape: Sequence[int]) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    rows, cols = shape
    if fmt == "nvfp4":
        return [
            ("factor", torch.float32, ()),
            ("scale", torch.float8_e4m3fn, (rows, cols // 16)),
            ("codes", torch.uint8, (rows, cols // 2)),
        ]
    if fmt == "mxfp8":
        return [
            ("values", torch.float8_e4m3fn, (rows, cols)),
            ("scale", torch.uint8, (rows, cols // 32)),
        ]
    raise ValueError(f"unknown released expert format {fmt!r}")


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def tensor_nbytes(fmt: str, shape: Sequence[int]) -> int:
    """Bytes of one expert's tensor of ``shape`` in ``fmt``."""
    return sum(
        _numel(s) * torch.empty((), dtype=d).element_size() for _, d, s in _parts(fmt, shape)
    )


def pack(fmt: str, shape: Sequence[int], parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """One expert's tensor as slab bytes. ``parts``: ``factor``, ``scale``, ``codes`` (NVFP4) or
    ``values``, ``scale`` (MXFP8), for this expert alone."""
    blobs = []
    for name, dtype, part_shape in _parts(fmt, shape):
        tensor = parts[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != part_shape:
            raise ValueError(
                f"{fmt} {name}: expected {dtype} {part_shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        blobs.append(tensor.contiguous().reshape(-1).view(torch.uint8))
    return torch.cat(blobs)


def decode(fmt: str, shape: Sequence[int], blobs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[n, nbytes]`` slab bytes of one tensor of n experts to ``[n, *shape]`` in ``dtype``."""
    n = blobs.shape[0]
    views, start = {}, 0
    for name, part_dtype, part_shape in _parts(fmt, shape):
        size = _numel(part_shape) * torch.empty((), dtype=part_dtype).element_size()
        raw = blobs[:, start : start + size].contiguous()
        views[name] = raw.view(part_dtype).reshape(n, *part_shape)
        start += size
    if fmt == "nvfp4":
        one = torch.ones(1, dtype=torch.float32, device=blobs.device)
        return codec.decode_nvfp4(
            views["codes"], views["scale"], views["factor"].reshape(n), one, dtype
        )
    return codec.decode_mxfp8(views["values"], views["scale"], dtype)


def _layout(
    formats: Sequence[str], tensor_names: Sequence[str], shapes: Mapping[str, tuple[int, ...]]
):
    offsets: list[dict[str, tuple[int, int]]] = []
    sizes: list[int] = []
    for fmt in formats:
        start, layer = 0, {}
        for name in tensor_names:
            size = tensor_nbytes(fmt, shapes[name])
            layer[name] = (start, start + size)
            start += size
        offsets.append(layer)
        sizes.append(start)
    return offsets, sizes


class ReleasedSlabArena:
    """The slabs of a ``ReleasedExpertStore``, written one expert tensor at a time (no
    quantization: the released bytes go in as they are)."""

    def __init__(
        self,
        layout: ExpertLayout,
        tensor_shapes: Mapping[str, Sequence[int]],
        formats: Sequence[str],
        *,
        sink: SinkFactory | None = None,
    ) -> None:
        if len(formats) != layout.num_layers or any(f not in FORMATS for f in formats):
            raise ValueError(f"one released format per MoE layer, of {FORMATS}: {formats}")
        self._layout = layout
        self._formats = tuple(formats)
        self._shapes = {n: tuple(tensor_shapes[n]) for n in layout.tensor_names}
        self._offsets, self._sizes = _layout(self._formats, layout.tensor_names, self._shapes)
        keys = [
            (layer, e)
            for layer in range(layout.num_layers)
            for e in range(layout.experts_per_layer)
        ]
        self._unwritten = {(k, n) for k in keys for n in layout.tensor_names}
        slabs = [(k, self._sizes[k[0]]) for k in keys]
        self._sink = sink(slabs) if sink is not None else RamSink(slabs)

    def write_parts(
        self, key: ExpertKey, tensor_name: str, parts: Mapping[str, torch.Tensor]
    ) -> None:
        layer = key[0]
        blob = pack(self._formats[layer], self._shapes[tensor_name], parts)
        start, end = self._offsets[layer][tensor_name]
        assert blob.numel() == end - start
        self._sink.write(key, start, blob)
        self._unwritten.discard((key, tensor_name))

    def build(self, compute_dtype: torch.dtype) -> "ReleasedExpertStore":
        if self._unwritten:
            key, name = min(self._unwritten)
            raise ValueError(
                f"{len(self._unwritten)} expert tensor(s) never written, e.g. {name} of {key}"
            )
        return ReleasedExpertStore(
            self._sink.finish(), self._layout, self._shapes, self._formats, compute_dtype
        )

    def abort(self) -> None:
        abort = getattr(self._sink, "abort", None)
        if abort is not None:
            abort()


class ReleasedExpertStore(MoeExpertStore):
    """``MoeExpertStore`` over released-format slabs (module doc); per-layer slab sizes."""

    def __init__(
        self,
        source: SlabSource | dict,
        layout: ExpertLayout,
        tensor_shapes: Mapping[str, tuple[int, ...]],
        formats: Sequence[str],
        compute_dtype: torch.dtype,
    ) -> None:
        offsets, sizes = _layout(formats, layout.tensor_names, tensor_shapes)
        super().__init__(source, layout, dict(tensor_shapes), offsets[0], max(sizes), compute_dtype)
        self.formats = tuple(formats)
        self._layer_offsets = offsets
        self._layer_sizes = sizes

    @classmethod
    def over_released(
        cls,
        source: SlabSource,
        layout: ExpertLayout,
        tensor_shapes: Mapping[str, Sequence[int]],
        formats: Sequence[str],
        compute_dtype: torch.dtype,
    ) -> "ReleasedExpertStore":
        """A store over slabs that already exist (a reused ``.sss``); checks every size."""
        shapes = {n: tuple(tensor_shapes[n]) for n in layout.tensor_names}
        _, sizes = _layout(formats, layout.tensor_names, shapes)
        have = set(source.keys())
        for layer in range(layout.num_layers):
            for e in range(layout.experts_per_layer):
                if (layer, e) not in have or source.nbytes((layer, e)) != sizes[layer]:
                    raise ValueError(
                        f"source does not hold expert slab {(layer, e)} of {sizes[layer]} bytes"
                    )
        return cls(source, layout, shapes, formats, compute_dtype)

    @property
    def max_layer_nbytes(self) -> int:
        """The largest MoE layer's slabs, in bytes (residency sizes its one-layer floor by it)."""
        return self._layout.experts_per_layer * max(self._layer_sizes)

    def slice_nbytes(self, key: ExpertKey) -> int:
        self._check(key)
        return self._layer_sizes[key[0]]

    def load_slice(self, key: ExpertKey, dst: torch.Tensor) -> None:
        self._check(key)
        if dst.dtype != torch.uint8 or dst.numel() != self._layer_sizes[key[0]]:
            raise ValueError(
                f"dst for {key} must be a uint8 tensor of {self._layer_sizes[key[0]]} bytes"
            )
        self._source.read_into(key, dst)

    def dequantize_experts(
        self, keys: Sequence[ExpertKey], tensor_name: str, slabs: torch.Tensor
    ) -> torch.Tensor:
        for key in keys:
            self._check(key)
        layers = {k[0] for k in keys}
        if len(layers) != 1:
            raise ValueError(f"released slabs decode one layer at a time, got layers {layers}")
        (layer,) = layers
        size = self._layer_sizes[layer]
        if slabs.dtype != torch.uint8 or slabs.shape != (len(keys), size):
            raise ValueError(f"slabs for {list(keys)} must be uint8 [{len(keys)}, {size}]")
        start, end = self._layer_offsets[layer][tensor_name]
        return decode(
            self.formats[layer],
            self._tensor_shapes[tensor_name],
            slabs[:, start:end],
            self._compute_dtype,
        )

    def dequantize_resident(self, keys, tensor_name, slabs):
        return None  # no GPU decoder of its own: dequantize_experts on the stacked slabs
