"""BlockManager: blockwise GPU residency, prefetch, and offload.

Partitions transformer layers into blocks, manages CPU/GPU residency,
async prefetch on dedicated CUDA streams, pinned memory, and offload.

INV-2: Cross-stream record_stream safety. When a block is loaded on a
non-default CUDA stream (prefetch path), every transferred tensor MUST
call tensor.record_stream(default_stream) so the caching allocator does
not reuse its storage while the default stream is still reading.

INV-5: Layer discovery handles nested / VLM layouts via the expanded
root and path candidates (delegated to core.arch_registry.discover_layers).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from poros.sss.pinning import PinnedArena

logger = logging.getLogger(__name__)


def resident_block_ids(num_blocks: int, n: int) -> List[int]:
    """The ``n`` blocks kept on the device for a whole run, spread evenly so each streamed
    block's copy overlaps a resident block's compute."""
    n = min(int(n), num_blocks)
    return sorted({(k + 1) * num_blocks // n - 1 for k in range(n)}) if n > 0 else []


def resident_layer_names(model: nn.Module, block_size: int, n: int) -> List[str]:
    """Dotted names of the decoder layers in the ``n`` resident blocks of ``model`` (a
    loaded model or a loader's skeleton: the same discovery as ``BlockManager``)."""
    from poros.core.arch_registry import discover_named_layers

    names = [name for name, _ in discover_named_layers(model)]
    blocks = resident_block_ids(-(-len(names) // block_size), n)
    return [names[i] for b in blocks
            for i in range(b * block_size, min((b + 1) * block_size, len(names)))]


@dataclass
class BlockInfo:
    """Metadata for a single block of transformer layers."""

    block_id: int
    layer_indices: List[int]
    param_count: int
    weight_size_bytes: int
    module_names: List[str]
    is_frozen: bool = True


class BlockManager:
    """Partition a decoder stack into blocks and move blocks between CPU and GPU.

    The manager is intentionally conservative on CPU-only systems: pinned-memory
    transfers become regular CPU copies so that the same code path still works
    for correctness tests and smoke tests.

    Args:
        model: The full model (nn.Module).
        block_size: Number of transformer layers per block.
        freeze_embeddings: Freeze the embedding layer(s).
        freeze_lm_head: Freeze the LM head.
        managed_parameter_filter: Optional callable to filter which parameters
            are managed (offloaded). Signature: (full_name, param) -> bool.
        train_managed_parameters: Whether managed parameters should have
            requires_grad=True when activated on GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        freeze_embeddings: bool = True,
        freeze_lm_head: bool = True,
        managed_parameter_filter: Optional[Callable[[str, nn.Parameter], bool]] = None,
        train_managed_parameters: bool = True,
    ):
        self.model = model
        self.block_size = block_size
        self.freeze_embeddings = freeze_embeddings
        self.freeze_lm_head = freeze_lm_head
        self._pin_memory_enabled = torch.cuda.is_available()
        self._managed_parameter_filter = managed_parameter_filter
        self._train_managed_parameters = train_managed_parameters

        self.transformer_layers = self._discover_layers()
        self.num_layers = len(self.transformer_layers)

        self.blocks = self._partition_blocks()
        self.num_blocks = len(self.blocks)
        self.active_block_id: Optional[int] = None

        # Maps block_id -> full parameter name -> CPU tensor.
        self._cpu_weight_store: Dict[int, Dict[str, torch.Tensor]] = {}

        # Transfer metrics: total host-to-device bytes / copy calls issued by
        # block loads.
        self.h2d_bytes_total: int = 0
        self.h2d_copy_count: int = 0

        if self.freeze_embeddings:
            self._freeze_embeddings()
        if self.freeze_lm_head:
            self._freeze_lm_head()

        logger.info(
            "BlockManager initialized: %s layers -> %s blocks (block_size=%s)",
            self.num_layers,
            self.num_blocks,
            block_size,
        )

    @property
    def pin_memory_enabled(self) -> bool:
        return self._pin_memory_enabled

    # ------------------------------------------------------------------
    # Layer discovery -- INV-5
    # ------------------------------------------------------------------

    def _discover_layers(self) -> List[Tuple[str, nn.Module]]:
        """Introspect the model to find the transformer layer stack.

        Delegates to the shared discovery in `arch_registry` (INV-5), so the
        root/path candidates -- including nested VLM layouts (Gemma/GPT-OSS) --
        live in exactly one place. Add a new layout there, not here.
        """
        from poros.core.arch_registry import discover_named_layers

        layers = discover_named_layers(self.model)
        logger.info("Found %s transformer layers", len(layers))
        return layers

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def _partition_blocks(self) -> List[BlockInfo]:
        blocks: List[BlockInfo] = []
        for block_id, start in enumerate(
            range(0, self.num_layers, self.block_size)
        ):
            end = min(start + self.block_size, self.num_layers)
            layer_slice = self.transformer_layers[start:end]
            managed_params = [
                param
                for layer_name, layer in layer_slice
                for _, param in self._managed_named_parameters(layer_name, layer)
            ]
            param_count = sum(param.numel() for param in managed_params)
            weight_bytes = sum(
                param.numel() * param.element_size() for param in managed_params
            )
            blocks.append(
                BlockInfo(
                    block_id=block_id,
                    layer_indices=list(range(start, end)),
                    param_count=param_count,
                    weight_size_bytes=weight_bytes,
                    module_names=[name for name, _ in layer_slice],
                )
            )
        return blocks

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze_embeddings(self) -> None:
        embedding_attrs = [
            "model.embed_tokens",
            "gpt_neox.embed_in",
            "transformer.wte",
        ]
        for attr_path in embedding_attrs:
            try:
                module: Any = self.model
                for attr in attr_path.split("."):
                    module = getattr(module, attr)
                for param in module.parameters():
                    param.requires_grad = False
                logger.info("Froze embeddings at '%s'", attr_path)
                return
            except AttributeError:
                continue

    def _freeze_lm_head(self) -> None:
        head_attrs = ["lm_head", "embed_out"]
        for attr in head_attrs:
            try:
                module = getattr(self.model, attr)
                for param in module.parameters():
                    param.requires_grad = False
                logger.info("Froze LM head at '%s'", attr)
                return
            except AttributeError:
                continue

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _block_layer_items(
        self, block_id: int
    ) -> List[Tuple[str, nn.Module]]:
        return [
            self.transformer_layers[i]
            for i in self.blocks[block_id].layer_indices
        ]

    def _managed_named_parameters(
        self,
        layer_name: str,
        layer: nn.Module,
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for param_name, param in layer.named_parameters(recurse=True):
            full_name = f"{layer_name}.{param_name}"
            if (
                self._managed_parameter_filter is not None
                and not self._managed_parameter_filter(full_name, param)
            ):
                continue
            yield full_name, param

    def get_block_modules(self, block_id: int) -> List[nn.Module]:
        return [layer for _, layer in self._block_layer_items(block_id)]

    def get_block_named_parameters(
        self, block_id: int
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """Yield fully qualified parameter names for a block."""
        for layer_name, layer in self._block_layer_items(block_id):
            yield from self._managed_named_parameters(layer_name, layer)

    def get_block_parameters(self, block_id: int) -> Iterator[nn.Parameter]:
        for _, param in self.get_block_named_parameters(block_id):
            yield param

    # ------------------------------------------------------------------
    # CPU buffer allocation
    # ------------------------------------------------------------------

    def _host_copy(self, tensor: torch.Tensor, arena: PinnedArena) -> torch.Tensor:
        """A host copy of a device tensor: page-locked from ``arena`` (slabs shared by many
        tensors, so torch's power-of-two rounding locks nothing extra) while the budget
        (poros.sss.pinning) allows, else pageable."""
        cpu_tensor = tensor.detach().to("cpu")
        if self._pin_memory_enabled:
            cpu_tensor = arena.pin(cpu_tensor)
            if arena.refused:
                self._pin_memory_enabled = False
                logger.warning(
                    "Pinned-memory allocation is unavailable in this environment; "
                    "falling back to regular CPU tensors."
                )
        return cpu_tensor

    # ------------------------------------------------------------------
    # Offload / activate / deactivate
    # ------------------------------------------------------------------

    def offload_all_to_cpu(self, keep: Tuple[int, ...] = ()) -> None:
        """Give every block a host store and release it from the device; blocks in
        ``keep`` (resident, see ``keep_resident``) are left where they are, with no host copy."""
        moving = [b.block_id for b in self.blocks if b.block_id not in keep]
        # release_on_free: the store lives as long as this manager; when it goes (the trainer
        # closed or dropped), its pages unlock and its grant returns to the budget
        arena = PinnedArena(sum(p.data.numel() * p.data.element_size() for b in moving
                                for _, p in self.get_block_named_parameters(b)
                                if p.data.device.type != "cpu"), release_on_free=True)
        for block_id in moving:
            self._offload_block(block_id, copy_back=True, arena=arena)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Offloaded all %s blocks to CPU storage", self.num_blocks
        )

    def _offload_block(
        self, block_id: int, copy_back: bool = True, arena: Optional[PinnedArena] = None
    ) -> None:
        """Offload a block to CPU storage.

        When ``copy_back`` is False, we only release the device tensors and
        keep the already-saved CPU copy. This is useful after an asynchronous
        writeback has already been launched.
        """
        block_store = self._cpu_weight_store.setdefault(block_id, {})
        released_cuda_tensors = False
        if arena is None:
            arena = PinnedArena(sum(p.data.numel() * p.data.element_size()
                                    for _, p in self.get_block_named_parameters(block_id)
                                    if p.data.device.type != "cpu"), release_on_free=True)

        for full_name, param in self.get_block_named_parameters(block_id):
            tensor = param.data
            if tensor.numel() == 0:
                continue

            if copy_back:
                if tensor.device.type == "cpu":
                    block_store[full_name] = tensor.detach()
                else:
                    released_cuda_tensors = True
                    cpu_tensor = block_store.get(full_name)
                    if (
                        cpu_tensor is None
                        or cpu_tensor.shape != tensor.shape
                        or cpu_tensor.dtype != tensor.dtype
                    ):
                        block_store[full_name] = self._host_copy(tensor, arena)
                    else:
                        cpu_tensor.copy_(tensor.detach(), non_blocking=False)

            param.data = torch.empty(0, dtype=tensor.dtype, device="cpu")

        self.blocks[block_id].is_frozen = True
        if released_cuda_tensors and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_block_loaded(
        self,
        block_id: int,
        device: Optional[torch.device] = None,
    ) -> bool:
        params = list(self.get_block_parameters(block_id))
        if not params:
            return False
        if any(param.data.numel() == 0 for param in params):
            return False
        if device is None:
            return True
        return all(param.data.device == device for param in params)

    # ------------------------------------------------------------------
    # Inference loading with INV-2: cross-stream record_stream safety
    # ------------------------------------------------------------------

    def load_block_for_inference(
        self,
        block_id: int,
        device: torch.device = torch.device("cuda:0"),
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Load a frozen block to ``device`` for forward-only or frozen-gradient use.

        When ``stream`` is a non-default CUDA stream (prefetch path), the
        resulting tensor must be marked as consumed by the default stream via
        ``record_stream``. Otherwise the caching allocator only tracks
        ``stream``'s lifetime and may reuse the tensor's storage on a
        subsequent prefetch while the default stream is still reading from it,
        causing a silent data race that manifests as nondeterministic losses
        once prefetch and compute actually overlap.

        INV-2: This is the cross-stream safety invariant. Do not remove the
        record_stream call.
        """
        block_store = self._cpu_weight_store.get(block_id, {})
        use_non_blocking = self._pin_memory_enabled and device.type == "cuda"
        use_stream_ctx = (
            stream is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        ctx = torch.cuda.stream(stream) if use_stream_ctx else nullcontext()
        default_stream = (
            torch.cuda.default_stream(device) if use_stream_ctx else None
        )

        with ctx:
            for full_name, param in self.get_block_named_parameters(block_id):
                already_loaded = (
                    param.data.numel() > 0 and param.data.device == device
                )
                if not already_loaded and full_name in block_store:
                    src = block_store[full_name]
                    new_tensor = src.to(device, non_blocking=use_non_blocking)
                    if device.type == "cuda":
                        self.h2d_bytes_total += (
                            src.numel() * src.element_size()
                        )
                        self.h2d_copy_count += 1
                    if default_stream is not None:
                        # INV-2: Tell the caching allocator that the default
                        # stream will also consume this tensor's storage, so
                        # it is not freed for reuse on ``stream`` until the
                        # default stream has caught up.
                        new_tensor.record_stream(default_stream)
                    param.data = new_tensor
                param.requires_grad = False

        self.blocks[block_id].is_frozen = True

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def keep_resident(self, block_ids: List[int], device: torch.device) -> None:
        """Keep ``block_ids`` on ``device`` for good, with no host copy.

        The device tensors are the block's only store: later loads find them
        already there (no copy) and ``release_block`` skips them, so only the
        H2D traffic and where the bytes live change, never a value. A block a
        loader left on the device stays as it is; one still in host RAM moves
        up, and its host copy is dropped (memory carved from a shared slab
        returns only once the slab's other tensors go too).
        """
        resident = getattr(self, "resident_block_ids", set())
        for block_id in block_ids:
            store = self._cpu_weight_store.pop(int(block_id), {})
            for full_name, param in self.get_block_named_parameters(block_id):
                src = param.data if param.data.numel() > 0 else store.get(full_name)
                if src is not None and src.device != device:
                    param.data = src.to(device)
                param.requires_grad = False
            self.blocks[block_id].is_frozen = True
            resident.add(int(block_id))
        self.resident_block_ids = resident
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def release_block(self, block_id: int) -> None:
        """Release a frozen block from device memory without copying back."""
        if block_id == self.active_block_id:
            raise ValueError(
                "release_block cannot be used on the active block"
            )
        if block_id in getattr(self, "resident_block_ids", ()):
            return

        for _, param in self.get_block_named_parameters(block_id):
            tensor = param.data
            if tensor.numel() == 0:
                continue
            param.data = torch.empty(0, dtype=tensor.dtype, device="cpu")
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "Poros Block Manager Summary",
            "=" * 50,
            f"Total layers: {self.num_layers}",
            f"Block size: {self.block_size}",
            f"Number of blocks: {self.num_blocks}",
            "",
        ]
        for block in self.blocks:
            if block.block_id == self.active_block_id:
                status = "ACTIVE"
            elif self.is_block_loaded(block.block_id):
                status = "LOADED"
            else:
                status = "FROZEN"
            lines.append(
                f"  Block {block.block_id:3d}: "
                f"layers {block.layer_indices[0]:3d}"
                f"-{block.layer_indices[-1]:3d} | "
                f"{block.param_count / 1e6:8.1f}M params | "
                f"{block.weight_size_bytes / 1e9:6.2f} GB | "
                f"[{status}]"
            )
        return "\n".join(lines)
