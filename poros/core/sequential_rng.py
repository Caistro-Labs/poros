"""The random states the sequential loop gives each (micro-batch, block), for block-major order.

The sequential loop runs one micro-batch's forward block by block, then the next micro-batch's.
Every dropout in a block draws from the generator as the block runs, and the rematerialized
backward restores the state that block's forward started from (INV-1, ``poros.core.blockwise``),
so a block's masks are fixed by the state it starts from. Block-major accumulation runs the same
forwards in another order (block 0 for every micro-batch, then block 1, ...). To draw the same
masks, each (micro-batch, block) must start from the state the sequential loop would give it.

That state depends only on what the forwards before it drew. In a dense decoder block the draws
are the LoRA dropouts: one per adapter-wrapped linear a forward calls, over its input of tokens x
``in_features`` elements. ``SequentialRng`` walks the sequential order without the model: for
each micro-batch, block by block, it records the generator state and then advances the generator
as the block's dropouts would, by drawing the same dropout on a scratch buffer of the same size.
On CUDA a draw of n elements moves the Philox offset by an amount set by n and the device alone,
so each size is drawn once and later draws of it add that measured amount to the offset.

Block-major sets each recorded state before the block's forward and before its recompute, and
after each forward checks that the generator moved exactly as the walk did (``check``). A random
draw the walk does not know about (a model's own dropout, router noise) stops the update
there, before anything is applied; poros.train.block_major then trains it, and the rest of the
run, sequentially.

A MoE model's states are keyed instead (``KeyedRng``). Either is an ``UpdateRng``, what the
block-major driver (poros.train.block_major) takes.
"""

from __future__ import annotations

import functools
import hashlib
import struct
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from poros.core.errors import PorosConfigError

# One dropout a block's forward draws per token: (p, in_features, dtype of the dropped tensor).
Draw = Tuple[float, int, torch.dtype]
# The CPU generator's state and the device generator's (None off CUDA).
State = Tuple[torch.Tensor, Optional[torch.Tensor]]


class UnknownDraw(PorosConfigError):
    """A forward draws random numbers the sequential order's walk does not account for."""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"block-major accumulation cannot follow the sequential loop's random draws here: "
            f"{message}.",
            remediation="path: sequential trains it so from the start (the same update).")


class UpdateRng(Protocol):
    """The random states of one block-major update, for its micro-batches g and blocks b
    (``SequentialRng``, ``KeyedRng``)."""

    def add(self, tokens: int) -> None:
        """The next micro-batch was fetched; it runs ``tokens`` positions."""

    def set(self, g: int, b: int) -> None:
        """Before micro-batch g's forward or recompute of block b (b = blocks: the head)."""

    def check(self, g: int, b: int, what: str) -> None:
        """After ``what`` of micro-batch g: raises ``UnknownDraw`` where it drew unaccounted."""

    def finish(self) -> None:
        """The update ran: the generators as the sequential loop leaves them."""

    def rewind(self) -> None:
        """The update stopped before applying anything: the generators where it started."""


def _cuda_generator(device: torch.device) -> Optional[torch.Generator]:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[index]


def _lora_dropouts(module: nn.Module) -> List[Tuple[nn.Module, torch.dtype]]:
    """The dropout modules a LoRA layer's forward applies to its input, with the dtype dropped.

    PEFT's LoRA layers (``lora_dropout``, one per adapter; the input is cast to ``lora_A``'s
    dtype first) and Poros's native ones (``dropout``, ``lora_A`` a parameter).
    """
    per_adapter = getattr(module, "lora_dropout", None)
    if isinstance(per_adapter, nn.ModuleDict):
        if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
            return []
        names = getattr(module, "active_adapters", list(per_adapter.keys()))
        return [(per_adapter[a], module.lora_A[a].weight.dtype)
                for a in names if a in per_adapter and a in module.lora_A]
    a = getattr(module, "lora_A", None)
    dropout = getattr(module, "dropout", None)
    if isinstance(a, nn.Parameter) and isinstance(dropout, nn.Module):
        return [] if getattr(module, "merged", False) else [(dropout, a.dtype)]
    return []


def _draws(dropout: nn.Module) -> bool:
    return isinstance(dropout, nn.Dropout) and dropout.training and dropout.p > 0


def block_draws(layers: Iterable[nn.Module]) -> List[Draw]:
    """The dropouts one forward of these decoder layers draws, in module order.

    Raises ``UnknownDraw`` for an active ``nn.Dropout`` that is not a LoRA layer's input dropout:
    its input size is not known here, so the walk could not follow it.
    """
    out: List[Draw] = []
    known = set()
    modules = [m for layer in layers for m in layer.modules()]
    for m in modules:
        owned = getattr(m, "lora_dropout", None)  # PEFT: every adapter's, active or not
        known.update(id(d) for d in (owned.values() if isinstance(owned, nn.ModuleDict) else ()))
        for dropout, dtype in _lora_dropouts(m):
            known.add(id(dropout))
            if _draws(dropout):
                out.append((float(dropout.p), int(m.in_features), dtype))
    stray = [m for m in modules if _draws(m) and id(m) not in known]
    if stray:
        raise UnknownDraw(f"a dropout outside the LoRA adapters ({type(stray[0]).__name__}, "
                          f"p={stray[0].p}) draws in the block's forward")
    return out


def _state(device: torch.device) -> State:
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda


def _set(device: torch.device, state: State) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _same(a: State, b: State) -> bool:
    return torch.equal(a[0], b[0]) and (a[1] is None or torch.equal(a[1], b[1]))


class SequentialRng:
    """One update's walk: the state each (micro-batch g, block b) starts from in sequential order.

    ``add(tokens)`` takes the micro-batches in order, each right after the loop fetches it, as the
    sequential loop would run its forward there: the live generators advance by the forward's
    draws, so after the last one they hold what the sequential loop leaves behind. ``set(g, b)``
    puts a recorded state in place; ``finish()`` puts the end of the walk back.

    ``memo`` maps (elements, p, dtype) to the CUDA offset one draw of that size moves: kept by
    the caller across updates, it is this device's own and is filled by drawing.
    """

    def __init__(self, device: torch.device, draws: Sequence[Sequence[Draw]],
                 memo: Dict[Tuple[int, float, torch.dtype], int]) -> None:
        self.device = device
        self.draws = draws
        self.memo = memo
        self.states: List[List[State]] = []
        self._gen = _cuda_generator(device)
        self._end: Optional[State] = None

    def add(self, tokens: int) -> None:
        """The next micro-batch, of ``tokens`` positions (rows x length, padding included)."""
        row = [_state(self.device)]
        for block in self.draws:
            for p, width, dtype in block:
                self._advance(tokens * width, p, dtype)
            row.append(_state(self.device))
        self.states.append(row)
        self._end = row[-1]

    def _advance(self, n: int, p: float, dtype: torch.dtype) -> None:
        """Move the live generator as one dropout over ``n`` elements does."""
        key = (n, p, dtype)
        step = self.memo.get(key)
        if step is not None and self._gen is not None:
            self._gen.set_offset(self._gen.get_offset() + step)
            return
        before = self._gen.get_offset() if self._gen is not None else 0
        F.dropout(torch.empty(n, dtype=dtype, device=self.device), p, training=True)
        if self._gen is not None:
            self.memo[key] = self._gen.get_offset() - before

    def set(self, g: int, b: int) -> None:
        """Start micro-batch ``g``'s block ``b`` (``b`` = the block count: past its last)."""
        _set(self.device, self.states[g][b])

    def check(self, g: int, b: int, what: str) -> None:
        """After ``what`` of micro-batch ``g``: the generators must be at its state ``b``."""
        if not _same(_state(self.device), self.states[g][b]):
            raise UnknownDraw(f"{what} of micro-batch {g} drew random numbers its LoRA "
                              "dropouts do not account for")

    def finish(self) -> None:
        """The live generators as the sequential loop leaves them after the update."""
        if self._end is not None:
            _set(self.device, self._end)

    def rewind(self) -> None:
        """The live generators where the update's first forward starts."""
        if self.states:
            _set(self.device, self.states[0][0])


# KeyedRng's key tags: a decoder layer's forward, a fetch, the end of an update's passes.
_LAYER, _FETCH, _END = 0, 1, 2


class KeyedRng:
    """A MoE model's random states, the same in either loop: each decoder layer's forward, and
    each recompute of it, starts from a generator state that is a pure function of (run seed,
    update, micro-batch, layer).

    A MoE layer's expert dropouts draw over the rows routed to each expert, so how far one moves
    the generator is known only once the layer has routed; block-major order cannot replay the
    sequential loop's stream past it (``SequentialRng``). Keyed by layer instead, no draw moves
    another layer's masks, so the sequential loop, block-major order at any block size, the
    rematerializing backward and a fully resident model all draw the same masks. The same
    pure function sets the generators before each micro-batch is fetched (a new epoch's shuffle
    draws there) and when an update's passes end, so what follows, and a resume, is the same on
    either loop.

    ``begin(update, base)`` opens an update's passes (``base``: micro-batches already run this
    update, for ``poros.Run``'s repeated calls); ``fetch(g)`` comes before micro-batch g is
    fetched, ``start(g)`` before its forward or recompute, ``end()`` after the passes. ``set``,
    ``check``, ``add`` and ``finish`` are ``SequentialRng``'s, for the block-major drivers.
    """

    def __init__(self, seed: int, device: torch.device) -> None:
        self.seed = int(seed)
        self.device = device
        self.update = 0
        self.base = 0
        self.micro: Optional[int] = None  # None: no training pass under way, the hooks idle
        self._count = 0
        self._layers = 0  # how many decoder layers carry the hook (install)
        self._fired: set = set()  # (micro-batch, layer) keyed since begin
        self._gen = _cuda_generator(device)

    def key(self, *parts: int) -> int:
        """The 63-bit seed for ``parts`` under this run's seed."""
        data = struct.pack(f"<{len(parts) + 1}q", self.seed, *parts)
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little") >> 1

    def _put(self, *parts: int) -> None:
        seed = self.key(*parts)
        torch.default_generator.manual_seed(seed)
        if self._gen is not None:
            self._gen.manual_seed(seed)  # its Philox offset starts at 0 again

    @staticmethod
    def layers_of(model: nn.Module) -> List[nn.Module]:
        """The decoder layers a model runs, in order (a layer list the dense engine streams
        gives them through ``real_layers``: iterating it yields collector proxies)."""
        from poros.core.arch_registry import discover_layers

        layers, _ = discover_layers(model)
        real = getattr(layers, "real_layers", None)
        return list(real()) if real is not None else list(layers)

    def install(self, layers: Sequence[nn.Module]) -> List[torch.utils.hooks.RemovableHandle]:
        """Key every forward of these decoder layers by its index."""
        self._layers = len(layers)
        return [layer.register_forward_pre_hook(functools.partial(self._layer, index))
                for index, layer in enumerate(layers)]

    def _layer(self, index: int, module: nn.Module, args: tuple) -> None:
        if self.micro is not None:
            self._put(*self.layer_key(index, getattr(module, "poros_visit_pass", 0)))
            self._fired.add((self.micro, index))

    def layer_key(self, index: int, visit_pass: int = 0) -> Tuple[int, ...]:
        """The parts a layer's forward is keyed by. A decoder that runs a layer more than once a
        forward (Inkling 2's loop) names the pass on the layer (``poros_visit_pass``), so each
        pass draws its own masks; pass 0, every other model's only pass, keys exactly as it
        always has."""
        parts: Tuple[int, ...] = (_LAYER, self.update, self.micro, index)
        return parts + (int(visit_pass),) if visit_pass else parts

    def begin(self, update: int, base: int = 0) -> None:
        self.update, self.base, self.micro, self._count = int(update), int(base), None, 0
        self._fired = set()

    def fetch(self, g: int) -> None:
        self._put(_FETCH, self.update, self.base + g)

    def start(self, g: int) -> None:
        self.micro = self.base + g
        self._count = max(self._count, g + 1)

    def end(self, check: bool = True) -> None:
        """The update's passes ended. Fails closed: every micro-batch must have keyed every layer
        it ran (a hook the forward never reaches would leave the masks unkeyed)."""
        self.micro = None
        missing = [(self.base + g, i) for g in range(self._count) for i in range(self._layers)
                   if (self.base + g, i) not in self._fired] if check else []
        self._put(_END, self.update, self.base + self._count)
        if missing:
            raise RuntimeError(
                "[Poros keyed random states] the forward never reached the key of micro-batch "
                f"{missing[0][0]}, layer {missing[0][1]} ({len(missing)} unkeyed in all): its "
                "dropout masks would not be keyed, so the update stops before it is applied.")

    def add(self, tokens: int) -> None:
        """Nothing to walk: every state is keyed."""

    def set(self, g: int, b: int) -> None:
        self.start(g)

    def check(self, g: int, b: int, what: str) -> None:
        """Nothing to check: no draw moves another layer's state."""

    def finish(self) -> None:
        self.end()

    def rewind(self) -> None:
        self.micro = None
