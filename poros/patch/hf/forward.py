"""The blockwise forward ``patch_hf_model`` installs on a model's decoder layers: the context a
forward and its rematerialized backward share (``_BlockwiseContext``), how a layer is called
and what it returns, and the forward itself (``_install_blockwise_forward``). Moved verbatim from
``poros.patch.hf``."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from poros.patch.hf.config import _PatchConfig

logger = logging.getLogger("poros.patch.hf")  # the patch's own log, as before


def _find_rotary_embedding_module(model: Any) -> Any:
    """Locate the model-level rotary-embedding module, if any.

    Mirrors ``PorosTrainer._get_rotary_embedding_module`` so the one-line
    ``patch_hf_model`` path threads ``position_embeddings`` identically to the
    trainer path.  Returns ``None`` for architectures that compute rotary
    inside each layer (or have no rotary at all).
    """
    for path in ("model.rotary_emb", "transformer.rotary_emb", "gpt_neox.rotary_emb"):
        obj = model
        ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None:
            return obj
    return None


def _partition_block_trainable_params(
    model: Any, block_manager: Any
) -> Dict[int, List[Tuple[str, Any]]]:
    """Index trainable params per block by module membership.

    Mirrors ``PorosTrainer._partition_trainable_parameters_by_block``: a param
    is attributed to a block if it belongs to one of that block's modules
    (matched by ``id``).  This deliberately uses ``get_block_modules`` rather
    than the managed-parameter enumeration, so adapter params -- which are
    *excluded* from management so they stay resident/trainable -- are still
    indexed for the backward's grad accumulation.
    """
    block_param_ids: Dict[int, set] = {}
    for block in block_manager.blocks:
        ids: set = set()
        for module in block_manager.get_block_modules(block.block_id):
            for param in module.parameters():
                ids.add(id(param))
        block_param_ids[block.block_id] = ids

    result: Dict[int, List[Tuple[str, Any]]] = {
        block.block_id: [] for block in block_manager.blocks
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for block_id, ids in block_param_ids.items():
            if id(param) in ids:
                result[block_id].append((name, param))
                break
    return result


def _move_trainable_params_to_model_device(model: Any) -> None:
    """Move trainable (adapter) params to the model's primary device.

    On CPU this is a no-op.  On GPU it ensures adapters remain resident while
    the frozen base streams, matching the trainer's
    ``_move_trainable_parameters_to_device``.
    """
    device = None
    for p in model.parameters():
        if p.data.numel() > 0:
            device = p.data.device
            break
    if device is None:
        return
    for param in model.parameters():
        if param.requires_grad and param.data.device != device:
            param.data = param.data.to(device)


class _UnseenLayerOutput(tuple):
    """A pass-through's return while no real layer of this model has run, and the annotation says
    tuple. Annotations can be wrong (transformers annotates ``Gemma3DecoderLayer.forward`` as a
    tuple and returns a tensor), so this serves both loops: one that indexes or unpacks it gets the
    tensor, and one that assigns it hands it to the next proxy, which unwraps it. The block's
    terminal layer then runs for real, and every return after that has the shape it gave."""


def _layer_returns_tuple(layers: Any) -> bool:
    """Whether a decoder layer's annotation says it returns a tuple ``(hidden, ...)``.

    Older transformers (<4.53) decoder layers return a tuple and the model
    loop does ``hidden_states = layer_outputs[0]``; newer ones return a bare
    ``torch.Tensor`` and the model loop uses the result directly.  Our proxy /
    passthrough layers must match that contract, otherwise the next layer is
    handed a tuple (or a tensor sliced by ``[0]``).  The real layer's return
    settles it once one has run (``_BlockwiseContext._real_layer_arity``); this
    only covers the pass-throughs before that, from the ``forward`` return
    annotation, falling back to the transformers version.
    """
    import inspect as _inspect

    try:
        if len(layers) == 0:
            return False
        ann = _inspect.signature(layers[0].forward).return_annotation
        if ann is not _inspect.Signature.empty:
            text = str(ann)
            if "Tuple" in text or "tuple" in text:
                return True
            if "Tensor" in text:
                return False
    except (ValueError, TypeError, IndexError):
        pass

    # Fallback: the bare-tensor contract landed in the 4.53 masking refactor.
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return (major, minor) < (4, 53)
    except Exception:
        return False


def _layer_tuple_padding(layers: Any) -> int:
    """How many ``None`` slots follow ``hidden`` in a tuple-returning layer's output.

    A modern layer that returns a fixed-arity tuple has a model loop that
    unpacks every element, as in ``hidden_states, extra = decoder_layer(...)``
    for a layer annotated ``-> tuple[Tensor, Tensor | None]``. The proxies
    must return that arity or the loop raises (or, for a bare tensor with
    batch size 2, silently unpacks the batch). The extra slots are ``None``:
    Poros block boundaries carry only the residual stream. Layers annotated
    with a bare tuple, ``tuple[X, ...]`` or a legacy ``Tuple[...]`` keep the
    1-tuple contract (padding 0).
    """
    import inspect as _inspect
    import typing as _typing

    try:
        if len(layers) == 0:
            return 0
        ann = _inspect.signature(layers[0].forward).return_annotation
    except (ValueError, TypeError, IndexError):
        return 0
    if _typing.get_origin(ann) is not tuple:
        return 0
    args = _typing.get_args(ann)
    if not args or Ellipsis in args or not str(ann).startswith("tuple["):
        return 0
    return len(args) - 1


def _layer_positional_param_names(layer: Any) -> List[str]:
    """Names of a decoder layer's positional parameters after ``hidden_states``.

    Modern models may hand the layer arch-specific tensors positionally rather
    than by keyword (e.g. Gemma 4 passes ``per_layer_input`` as the second
    positional arg). Returning the ordered parameter names lets the collector
    proxy bind those positionals back to their names so they are captured and
    replayed faithfully instead of being dropped.
    """
    import inspect as _inspect

    try:
        params = list(_inspect.signature(layer.forward).parameters.values())
    except (ValueError, TypeError):
        return []
    names: List[str] = []
    for p in params:
        if p.kind in (
            _inspect.Parameter.POSITIONAL_ONLY,
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            names.append(p.name)
    # Drop the leading ``hidden_states`` (passed explicitly by the proxy).
    return names[1:] if names else names


class _BlockwiseContext:
    """The ``trainer`` ``JITFrozenBlockFunction`` and ``JITAdapterBlockFunction`` are given on the
    patch path: it provides ``poros.core.blockwise.BlockHost`` without the full ``PorosTrainer``."""

    def __init__(
        self,
        block_manager: Any,  # residency.BlockManager
        layers: Any,         # nn.ModuleList -- the original transformer layers
        config: _PatchConfig,
        enable_prefetch: bool = True,
        trainable_params_override: Optional[Dict[int, List[Tuple[str, Any]]]] = None,
        model: Any = None,   # the full HF model, used to locate rotary_emb
    ) -> None:
        import torch

        self.block_manager = block_manager
        self.config = config
        self._layers = layers
        # Per-pass capture of each layer's exact HF kwargs (masks + rotary),
        # populated by the collector proxies and replayed in
        # _forward_block_layers (forward + backward rematerialization).
        self._block_layer_kwargs: Dict[int, Dict[int, Dict[str, Any]]] = {}
        # What a real layer returned last: 0 for a bare tensor, else the tuple's
        # length. None until one has run. A dict, so the per-pass copies share it.
        self._real_layer_arity: Dict[str, Optional[int]] = {"arity": None}
        # Rotary-embedding module.  Modern transformers (>=4.43) compute the
        # rotary position embeddings ONCE at the model level and pass them into
        # every decoder layer; the layer no longer derives them internally
        # (Qwen2Attention does ``cos, sin = position_embeddings`` with no
        # fallback).  We recompute them per block from position_ids -- which is
        # deterministic, so the forward pass and the backward rematerialization
        # produce identical embeddings.
        self._rotary_emb = (
            _find_rotary_embedding_module(model) if model is not None else None
        )

        # Which kwargs the decoder layer's forward accepts, so we only pass
        # use_cache / output_attentions when the layer supports them.
        self._layer_call_params: set = set()
        try:
            import inspect

            if len(layers) > 0:
                self._layer_call_params = set(
                    inspect.signature(layers[0].forward).parameters
                )
        except (ValueError, TypeError, IndexError):
            self._layer_call_params = set()

        # Per-block trainable-param index used by JITAdapterBlockFunction.
        # Must be built BEFORE offloading (offload sets requires_grad=False),
        # so the caller passes the pre-computed mapping when available.
        if trainable_params_override is not None:
            self.block_trainable_params = trainable_params_override
        else:
            self.block_trainable_params = {}
            for block_id in range(block_manager.num_blocks):
                params: List[Tuple[str, Any]] = []
                for name, param in block_manager.get_block_named_parameters(block_id):
                    if param.requires_grad:
                        params.append((name, param))
                self.block_trainable_params[block_id] = params

        # Prefetch stream (one stream shared across blocks)
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        if enable_prefetch and torch.cuda.is_available():
            try:
                self._prefetch_stream = torch.cuda.Stream()
            except Exception:
                self._prefetch_stream = None

    # -- Interface methods consumed by blockwise.py --

    def _record_remat_event(self, event_name: str, block_id: int) -> None:
        """Log a remat event (forward/backward load/release)."""
        logger.debug("patch remat event: %s block=%d", event_name, block_id)

    def _forward_block_layers(
        self,
        block_id: int,
        hidden_states: Any,
        attention_mask: Any,
        position_ids: Any,
        checkpoint: bool = False,
    ) -> Any:
        """Run the original transformer layers belonging to ``block_id``.

        Faithful per-layer kwargs: when the model's real ``forward`` ran (patch
        path), each layer's exact kwargs -- the per-layer attention mask chosen
        from ``causal_mask_mapping[layer.attention_type]`` and the rotary
        embeddings (single ``position_embeddings`` OR Gemma's dual
        ``position_embeddings_global`` / ``position_embeddings_local``) -- were
        captured by the collector proxies into ``self._block_layer_kwargs``.
        Replaying them verbatim is bitwise-identical to the resident HF forward
        and correctly handles sliding-window archs (Gemma 3's local layers) and
        dual-rotary archs (gemma3) without any arch-specific code here.
        """
        block_info = self.block_manager.blocks[block_id]
        params = self._layer_call_params
        captured = getattr(self, "_block_layer_kwargs", {}).get(block_id, {})

        # Fallback rotary for the legacy/self-check path where no per-layer
        # kwargs were captured (single precomputed rotary from position_ids).
        fallback_pos_emb = None
        if not captured and "position_embeddings" in params \
                and self._rotary_emb is not None and position_ids is not None:
            fallback_pos_emb = self._rotary_emb(hidden_states, position_ids)

        for layer_idx in block_info.layer_indices:
            layer = self._layers[layer_idx]
            lk = captured.get(layer_idx)
            if lk is not None:
                # Replay HF's exact per-layer kwargs (masks + rotary). Drop any
                # KV-cache objects so the blockwise pass never allocates a cache.
                kwargs = {
                    k: v for k, v in lk.items()
                    if k not in ("past_key_value", "past_key_values")
                }
            else:
                # Legacy single-mask path (trainer self-check / no capture).
                kwargs = {}
                if "attention_mask" in params:
                    kwargs["attention_mask"] = attention_mask
                if "position_ids" in params:
                    kwargs["position_ids"] = position_ids
                if fallback_pos_emb is not None:
                    kwargs["position_embeddings"] = fallback_pos_emb
            if "use_cache" in params:
                kwargs["use_cache"] = False
            if "output_attentions" in params:
                kwargs["output_attentions"] = False
            out = layer(hidden_states, **kwargs)
            if isinstance(out, tuple):
                self._real_layer_arity["arity"] = len(out)
                hidden_states = out[0]
            else:
                self._real_layer_arity["arity"] = 0
                hidden_states = out
        return hidden_states

    def _get_adapter_prefetch_stream(self) -> Any:
        """Return the CUDA stream used for backward prefetch, or None."""
        return self._prefetch_stream


def _install_blockwise_forward(
    model: Any,
    layers_attr_chain: List[str],
    context: _BlockwiseContext,
    use_adapter_fn: bool,
) -> None:
    """Replace the model's layer ModuleList with a blockwise variant.

    Rather than overwriting ``forward()`` on the model (fragile -- HF
    Trainer, generation, and PEFT all assume specific forward signatures),
    we replace the layers ModuleList with a ``_BlockwiseModuleList`` that
    intercepts ``__iter__``.  When the model's forward iterates over its
    layers, the custom list yields proxy modules that route each block
    through the appropriate autograd.Function (``JITAdapterBlockFunction``
    or ``JITFrozenBlockFunction``).

    All standard ModuleList behaviour (``parameters()``,
    ``named_modules()``, ``state_dict()``, indexing) is preserved
    because ``_BlockwiseModuleList`` inherits from ``nn.ModuleList``
    and retains the real layer modules as children.
    """
    import torch
    import torch.nn as nn

    if use_adapter_fn:
        from poros.core.blockwise import JITAdapterBlockFunction
        autograd_fn = JITAdapterBlockFunction
    else:
        from poros.core.blockwise import JITFrozenBlockFunction
        autograd_fn = JITFrozenBlockFunction

    num_blocks = context.block_manager.num_blocks
    returns_tuple = _layer_returns_tuple(context._layers)
    # Fixed-arity tuple outputs (a layer's (hidden, extra)) are unpacked by the
    # model loop: pad the proxies' output to that arity.
    tuple_padding = (None,) * _layer_tuple_padding(context._layers)

    def _as_layer_output(tensor: Any) -> Any:
        """``tensor`` in the shape the real decoder layer returns: bare on modern
        transformers, else a tuple (optional present_kv / attn_weights are dropped
        since we run in no-cache mode), with ``None`` in any further slots."""
        arity = context._real_layer_arity["arity"]
        if arity is None:
            return _UnseenLayerOutput((tensor, *tuple_padding)) if returns_tuple else tensor
        return (tensor, *(None,) * (arity - 1)) if arity else tensor

    class _BlockwiseModuleList(nn.ModuleList):
        """Drop-in replacement for the model's layer ModuleList.

        When iterated normally by the model's ``forward()``, it yields
        ``_BlockwiseCollectorProxy`` wrappers (one per layer).  Each proxy
        captures HF's exact per-layer kwargs; the terminal layer of each block
        routes through the appropriate autograd.Function for its block.

        All standard ModuleList behaviour (indexing, parameters(),
        named_modules(), state_dict, etc.) is preserved because we
        inherit from ModuleList and keep the real layers as children.
        """

        def __init__(
            self,
            layers: nn.ModuleList,
            ctx: _BlockwiseContext,
            fn_cls: type,
        ):
            # Initialize as a fresh ModuleList with the same children
            super().__init__(list(layers))
            self._poros_ctx = ctx
            self._poros_fn_cls = fn_cls
            # Cache block_id for each layer index
            bm = ctx.block_manager
            self._layer_to_block: Dict[int, int] = {}
            for block in bm.blocks:
                for li in block.layer_indices:
                    self._layer_to_block[li] = block.block_id
            self._current_pass_id: Optional[int] = None

        def _reset_pass(self) -> None:
            self._current_pass_id = id(object())

        def __getitem__(self, idx):  # type: ignore[override]
            """Support slicing without breaking the blockwise routing.

            Modern transformers iterate the decoder with
            ``for layer in self.layers[: config.num_hidden_layers]:``.
            ``nn.ModuleList.__getitem__`` handles a slice by calling
            ``self.__class__(sliced)`` -- which fails here because
            ``_BlockwiseModuleList.__init__`` requires ``ctx``/``fn_cls``.
            For the full-range slice the model uses to iterate every layer,
            return ``self`` so iteration still flows through ``__iter__``
            (yielding block proxies).  Integer indexing and partial slices
            fall back to the real layer modules.
            """
            if isinstance(idx, slice):
                start, stop, step = idx.indices(len(self))
                if start == 0 and stop >= len(self) and step == 1:
                    return self
                return [
                    nn.ModuleList.__getitem__(self, i)
                    for i in range(start, stop, step)
                ]
            return nn.ModuleList.__getitem__(self, idx)

        def real_layers(self) -> List[nn.Module]:
            """The decoder layers themselves, in order: what the blocks run (iterating this list
            yields collector proxies instead; indexing gives the layers too)."""
            return [nn.ModuleList.__getitem__(self, i) for i in range(len(self))]

        def __iter__(self):  # type: ignore[override]
            """Yield collector proxies that capture HF's exact per-layer kwargs.

            Modern transformers resolve the per-layer attention mask
            (``causal_mask_mapping[layer.attention_type]``) and rotary
            embeddings at the model level and pass them into EVERY decoder
            layer.  We yield one collector per layer: every collector records
            the kwargs HF hands it (so masks for sliding-window archs and the
            dual global/local rotary for Gemma are all captured), and the
            collector for the LAST layer of each block runs the whole block
            through the autograd.Function.  Because the earlier layers in the
            block pass ``hidden_states`` through unchanged, the last layer
            receives the block's original input -- exactly what the block
            function needs.
            """
            self._reset_pass()
            # Reset per-pass capture buffer (read back by _forward_block_layers
            # in both the forward and the backward rematerialization).
            self._poros_ctx._block_layer_kwargs = {}
            for idx in range(len(self)):
                real_layer = nn.ModuleList.__getitem__(self, idx)
                attn_type = getattr(real_layer, "attention_type", "full_attention")
                positional_names = _layer_positional_param_names(real_layer)
                block_id = self._layer_to_block.get(idx)
                if block_id is not None:
                    layer_indices = self._poros_ctx.block_manager.blocks[
                        block_id
                    ].layer_indices
                    is_last = idx == layer_indices[-1]
                    yield _BlockwiseCollectorProxy(
                        self._poros_ctx, block_id, idx, is_last,
                        self._poros_fn_cls, attn_type,
                        positional_names=positional_names,
                    )
                else:
                    # Shouldn't happen, but fall through to the real layer
                    yield real_layer

    class _BlockwiseCollectorProxy(nn.Module):
        """Per-layer proxy that captures HF's kwargs and, on the block's last
        layer, routes the whole block through the Poros autograd.Function.

        Non-terminal layers in a block are pass-throughs (return
        ``hidden_states`` unchanged) so the block's original input reaches the
        terminal layer; only the terminal layer triggers the streamed block
        computation/rematerialization."""

        def __init__(
            self,
            ctx: _BlockwiseContext,
            block_id: int,
            layer_idx: int,
            is_last: bool,
            fn_cls: type,
            attention_type: str = "full_attention",
            positional_names: Optional[List[str]] = None,
        ):
            super().__init__()
            self._ctx = ctx
            self._block_id = block_id
            self._layer_idx = layer_idx
            self._is_last = is_last
            self._fn_cls = fn_cls
            self.attention_type = attention_type
            # Parameter names (after ``hidden_states``) so any positional args
            # the model passes -- e.g. Gemma 4's ``per_layer_input`` -- are
            # captured by name and replayed faithfully, not silently dropped.
            self._positional_names = positional_names or []

        def forward(self, hidden_states, *args, **kwargs):
            if isinstance(hidden_states, _UnseenLayerOutput):
                hidden_states = hidden_states[0]
            # Bind positional args to their parameter names so capture/replay is
            # faithful regardless of how the model calls the layer.
            for name, value in zip(self._positional_names, args):
                kwargs.setdefault(name, value)
            # Capture this layer's exact kwargs (per-layer mask + rotary + any
            # arch-specific per-layer tensors).
            self._ctx._block_layer_kwargs.setdefault(self._block_id, {})[
                self._layer_idx
            ] = kwargs

            if not self._is_last:
                # Pass-through: keep the block input unchanged for the terminal
                # layer, which recomputes the whole block from it.
                return _as_layer_output(hidden_states)

            # The blockwise forward runs each block inside a custom
            # autograd.Function under no_grad; its output only requires grad if
            # an input tensor does.  For the FIRST block the input is the frozen
            # embedding output (no grad), so the autograd chain would never
            # connect and ``loss.backward()`` raises "does not require grad".
            # Seed grad on the first block's input -- exactly what the trainer
            # path does (trainer.py: ``hidden_states.requires_grad_(True)``).
            if (
                self._block_id == 0
                and torch.is_grad_enabled()
                and isinstance(hidden_states, torch.Tensor)
                and not hidden_states.requires_grad
            ):
                hidden_states = hidden_states.detach().requires_grad_(True)

            # Thread the block's FIRST-layer mask/position_ids through the
            # autograd.Function for save-for-backward; the faithful per-layer
            # kwargs are replayed inside _forward_block_layers from the capture.
            first_idx = self._ctx.block_manager.blocks[
                self._block_id
            ].layer_indices[0]
            # Incremental-decode guard. The streamed blocks strip
            # past_key_values and never write to the cache, so under
            # generate(use_cache=True) the cache stays EMPTY while
            # cache_position advances -- each step then attends to one token
            # and produces silent garbage after the first new token. The
            # broken state is precisely "the caller believes there is past
            # context (cache_position > 0) but the cache holds none".
            # (Poros turns use_cache off at patch time; this fires only when
            # a caller forces use_cache=True explicitly.)
            # cache_position never reaches the layers; what does is the
            # decode-continuation signature: use_cache forced on, position_ids
            # starting past zero ("this token continues a sequence"), and an
            # EMPTY cache holding none of that past (verified by kwarg spy).
            pos = kwargs.get("position_ids")
            if (
                kwargs.get("use_cache")
                and pos is not None
                and pos.numel() > 0
                and int(pos.reshape(-1)[0].item()) > 0
            ):
                pkv = kwargs.get("past_key_values")
                try:
                    pkv_len = pkv.get_seq_length() if pkv is not None else 0
                except Exception:  # pragma: no cover - cache API drift
                    pkv_len = 0
                if not pkv_len:
                    raise RuntimeError(
                        "Blockwise-patched models cannot serve an "
                        "incremental KV cache: the streamed blocks hold no "
                        "past state, so generation must recompute the full "
                        "prefix each step. Poros disables use_cache at patch "
                        "time -- remove use_cache=True from generate() (or "
                        "pass use_cache=False)."
                    )
            first_kw = self._ctx._block_layer_kwargs[self._block_id].get(
                first_idx, kwargs
            )
            # Hand the Function a per-pass SNAPSHOT of the context, never the
            # live one: the shared context's kwarg capture is overwritten by
            # every forward, so a second forward between this one and its
            # loss.backward() (eval inside the step, a TRL-style reference
            # pass, manual accumulation) made the backward rematerialize with
            # the LAST forward's masks/rotary -- silently wrong adapter
            # gradients (measured max delta 1.7e-2) or a shape crash. The
            # snapshot pins the backward to the kwargs of ITS OWN forward.
            pass_ctx = copy.copy(self._ctx)
            pass_ctx._block_layer_kwargs = {
                self._block_id: dict(
                    self._ctx._block_layer_kwargs[self._block_id]
                )
            }
            output = self._fn_cls.apply(
                hidden_states,
                first_kw.get("attention_mask", None),
                first_kw.get("position_ids", None),
                pass_ctx,
                self._block_id,
            )
            return _as_layer_output(output)

    # -- Locate the parent module that owns the layer list --
    parent = model
    for attr_name in layers_attr_chain[:-1]:
        parent = getattr(parent, attr_name)
    list_attr = layers_attr_chain[-1]
    original_layers = getattr(parent, list_attr)

    # The rotary-embedding module lives on the same backbone that owns the
    # layer list (e.g. Qwen2Model.rotary_emb).  Resolving it relative to the
    # backbone is robust to PEFT/VLM nesting where a top-level "model.rotary_emb"
    # lookup misses (it's base_model.model.model.rotary_emb under PEFT).
    if getattr(context, "_rotary_emb", None) is None:
        context._rotary_emb = getattr(parent, "rotary_emb", None)

    # -- Swap in our blockwise ModuleList --
    blockwise_list = _BlockwiseModuleList(original_layers, context, autograd_fn)
    setattr(parent, list_attr, blockwise_list)

    logger.info(
        "Installed blockwise forward: %d blocks, fn=%s",
        num_blocks,
        autograd_fn.__name__,
    )
