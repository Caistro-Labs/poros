"""PorosTrainer's forward: the decoder's inputs (embeddings, masks, rotary), each block's
layers called as the model would call them, the modules found on the backbone, where the static
modules and the adapters live, and the self-check and eval forwards."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from poros.core.blockwise import rematerialize_adapter_block
from poros.core.config import train_path
from poros.core.tables import cast_to, embed_as
from poros.train.state import _TrainerState

logger = logging.getLogger("poros.train.trainer")  # the trainer's own log, as before



@dataclass
class _DecoderInputs:
    """Per-block decoder kwargs, reconstructed to match HF's model-level forward.

    Modern transformers compute the per-layer attention mask and rotary
    embeddings ONCE at the model level and hand them to every decoder layer.
    For sliding-window archs (Gemma 3's local layers) the mask differs per layer
    (``causal_mask_mapping[layer.attention_type]``); for dual-rotary archs
    (Gemma 3) each layer needs BOTH a global and a local rotary. We rebuild
    those here by calling the very same HF helpers with the same arguments, so
    the blockwise layer calls are bitwise-faithful to a resident HF forward.
    """

    # Either a single mask tensor (qwen-style) or a dict keyed by
    # ``layer.attention_type`` ({"full_attention": ..., "sliding_attention": ...}).
    mask: Any = None
    # Single shared rotary (Qwen).
    position_embeddings: Optional[Any] = None
    # Dual rotary (Gemma 3): global + local.
    position_embeddings_global: Optional[Any] = None
    position_embeddings_local: Optional[Any] = None
    # Per-attention-type rotary (Gemma 4): one shared rotary module whose
    # frequencies depend on a ``layer_type`` argument, so HF computes one
    # ``position_embeddings`` per unique layer type and dispatches per layer.
    position_embeddings_by_type: Optional[Dict[str, Any]] = None
    cache_position: Optional[Any] = None
    # Scratch dict some archs (Gemma 4) hand to every layer so a KV-sharing
    # layer can publish its full-length key/value states for later layers. One
    # dict per block keeps the write target alive across the block's layers,
    # matching HF's single model-level ``shared_kv_states``.
    shared_kv_states: Optional[Any] = None


def _rotary_takes_layer_type(rotary: nn.Module) -> bool:
    """Whether a rotary module's forward accepts a third ``layer_type`` arg.

    Gemma 4 uses a single rotary module whose frequencies depend on the layer
    type, so its forward is ``(x, position_ids, layer_type)``. Detecting the
    extra positional parameter keeps the dispatch generic -- no model-name
    checks -- so any future per-type rotary is handled the same way.
    """
    try:
        params = [
            p
            for p in inspect.signature(rotary.forward).parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
    except (ValueError, TypeError):
        return False
    # self is bound out; expect (x, position_ids[, layer_type]).
    return len(params) >= 3


def _recurrent_mask(
    config: Any, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """The padding mask of a Gated DeltaNet layer, as its HF model computes it."""
    from transformers import masking_utils

    create = getattr(masking_utils, "create_recurrent_attention_mask", None)
    if create is not None:
        return create(
            config=config, inputs_embeds=hidden_states, attention_mask=attention_mask
        )
    # Releases before that helper hand the layer the 2-D mask as it came.
    return (
        attention_mask
        if attention_mask is not None and attention_mask.ndim == 2
        else None
    )


class _Forward(_TrainerState):
    """PorosTrainer's forward through the streamed blocks (mixed into PorosTrainer)."""

    def _forward_blockwise_adapter_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through all blocks using blockwise adapter path."""
        if self.block_manager is None:  # poros.moe: the model streams itself
            assert self.model is not None
            if self.head.moe_fused:  # the fused MoE head: no [tokens, vocab] logits
                hidden = self.model.get_decoder()(
                    input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"),
                ).last_hidden_state
                return self.head.loss(hidden, batch["labels"], normed=True)
            # the model's own forward and loss (a router aux loss too, where it is on: such a
            # model trains sequentially); block-major's head computes the same from the logits
            return self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            ).loss
        hidden, labels = self._forward_train_logits(batch, hidden_only=True)
        return self.head.loss(hidden, labels, normed=False)

    def _forward_train_logits(
        self, batch: Dict[str, torch.Tensor], hidden_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The training forward up to the logits, ready for any loss.

        Streamed blocks are retained for the rematerialized backward exactly
        as in :meth:`_forward_blockwise_adapter_model`; the caller runs
        ``backward()`` and then :meth:`_release_retained_forward_blocks`.
        """
        if self.block_manager is None:  # poros.moe: the model streams itself
            assert self.model is not None
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            return out.logits, batch.get("labels")

        hidden_states, attention_mask, position_ids, labels = (
            self._build_inputs(batch)
        )
        self._retained_forward_blocks = []
        hidden_states = hidden_states.detach().requires_grad_(True)

        num_blocks = self.block_manager.num_blocks
        prefetch_stream = self._get_adapter_prefetch_stream()

        # Kick off async prefetch of first block
        if prefetch_stream is not None and num_blocks > 0:
            self.block_manager.load_block_for_inference(
                0, device=self.device, stream=prefetch_stream
            )

        for block_id in range(num_blocks):
            if prefetch_stream is not None:
                # GPU-side barrier: default stream waits for current block
                torch.cuda.current_stream().wait_stream(prefetch_stream)
                # Overlap: start loading next block while current one computes
                if block_id + 1 < num_blocks:
                    self.block_manager.load_block_for_inference(
                        block_id + 1,
                        device=self.device,
                        stream=prefetch_stream,
                    )

            hidden_states = rematerialize_adapter_block(
                hidden_states,
                attention_mask,
                position_ids,
                self,
                block_id,
            )

        # Pre-load last block for backward (processed first in reverse)
        if prefetch_stream is not None and num_blocks > 0:
            self.block_manager.load_block_for_inference(
                num_blocks - 1, device=self.device, stream=prefetch_stream
            )

        if hidden_only:  # the last block's output: the caller runs the head
            return hidden_states, labels
        return self._logits_from_hidden(hidden_states), labels

    def _forward_block_layers(
        self,
        block_id: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Any],
        position_ids: Optional[torch.Tensor],
        checkpoint: bool,
        dec: Optional[_DecoderInputs] = None,
    ) -> torch.Tensor:
        """Forward through the layers of a single block.

        ``dec`` (optional): decoder inputs prebuilt by the caller for these exact
        (hidden dtype/device/shape, mask, positions) -- reused with a fresh
        per-block ``shared_kv_states``; otherwise rebuilt here.
        """
        assert self.block_manager is not None
        if dec is None:
            dec = self._prepare_decoder_attention(
                hidden_states, attention_mask, position_ids
            )
        else:
            dec = _dc_replace(dec, shared_kv_states=None)
        block_layer_indices = self.block_manager.blocks[block_id].layer_indices
        for layer_index, layer in zip(
            block_layer_indices,
            self.block_manager.get_block_modules(block_id),
        ):
            hidden_states = self._forward_one_layer(
                layer,
                hidden_states,
                position_ids,
                checkpoint=checkpoint,
                layer_index=layer_index,
                dec=dec,
            )
        return hidden_states

    def _forward_one_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        checkpoint: bool,
        *,
        layer_index: Optional[int] = None,
        dec: Optional[_DecoderInputs] = None,
    ) -> torch.Tensor:
        """Forward through a single transformer layer."""
        if not checkpoint:
            return self._call_layer(
                layer,
                hidden_states,
                position_ids,
                layer_index=layer_index,
                dec=dec,
            )

        def layer_fn(x: torch.Tensor) -> torch.Tensor:
            return self._call_layer(
                layer,
                x,
                position_ids,
                layer_index=layer_index,
                dec=dec,
            )

        return torch.utils.checkpoint.checkpoint(
            layer_fn, hidden_states, use_reentrant=False
        )

    def _resolve_layer_attention_type(
        self, layer: nn.Module, layer_index: Optional[int]
    ) -> Optional[str]:
        """This layer's attention type, the key HF uses for per-type mappings.

        Resolution order, most authoritative first:
        1. the layer's own ``attention_type`` (Gemma 3),
        2. the attention submodule's ``layer_type`` (Gemma 4 records it on
           ``self_attn`` rather than the layer; Qwen3.5 on whichever of
           ``self_attn`` / ``linear_attn`` the layer has), and
        3. ``config.layer_types[layer_index]`` as a positional fallback.
        """
        attn_type = getattr(layer, "attention_type", None)
        for attention in ("self_attn", "linear_attn"):
            if attn_type is None:
                attn_type = getattr(
                    getattr(layer, attention, None), "layer_type", None
                )
        if attn_type is None and layer_index is not None:
            backbone = self._get_backbone_module()
            layer_types = getattr(
                getattr(backbone, "config", None), "layer_types", None
            )
            if isinstance(layer_types, list) and 0 <= layer_index < len(layer_types):
                attn_type = layer_types[layer_index]
        return attn_type

    def _select_from_type_mapping(
        self,
        mapping: Dict[str, Any],
        layer: nn.Module,
        layer_index: Optional[int],
    ) -> Optional[Any]:
        """Pick a layer's entry from a ``{attention_type: value}`` mapping."""
        attn_type = self._resolve_layer_attention_type(layer, layer_index)
        if attn_type is not None and attn_type in mapping:
            return mapping[attn_type]
        return mapping.get("full_attention", next(iter(mapping.values()), None))

    def _select_layer_mask(
        self, layer: nn.Module, layer_index: Optional[int], dec: _DecoderInputs
    ) -> Optional[Any]:
        """Pick this layer's attention mask from the (possibly per-type) mapping.

        For sliding-window archs HF resolves the mask per layer via
        ``causal_mask_mapping[layer.attention_type]``.
        """
        mask = dec.mask
        if not isinstance(mask, dict):
            return mask
        return self._select_from_type_mapping(mask, layer, layer_index)

    def _select_layer_position_embeddings(
        self, layer: nn.Module, layer_index: Optional[int], dec: _DecoderInputs
    ) -> Optional[Any]:
        """Pick this layer's rotary embeddings.

        For per-attention-type rotary archs (Gemma 4) HF resolves the rotary per
        layer via ``position_embeddings[layer.attention_type]``; otherwise a
        single shared ``position_embeddings`` applies to every layer.
        """
        by_type = dec.position_embeddings_by_type
        if isinstance(by_type, dict) and by_type:
            return self._select_from_type_mapping(by_type, layer, layer_index)
        return dec.position_embeddings

    def _call_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        *,
        layer_index: Optional[int] = None,
        dec: Optional[_DecoderInputs] = None,
    ) -> torch.Tensor:
        """Call a single transformer layer with HF-faithful per-layer kwargs."""
        if dec is None:
            dec = _DecoderInputs()
        layer_type = type(layer)
        if layer_type not in self._signature_cache:
            self._signature_cache[layer_type] = inspect.signature(layer.forward)
        signature = self._signature_cache[layer_type]
        params = signature.parameters

        kwargs: Dict[str, Any] = {}

        if "attention_mask" in params:
            layer_mask = self._select_layer_mask(layer, layer_index, dec)
            if layer_mask is not None:
                kwargs["attention_mask"] = layer_mask

        if "position_ids" in params and position_ids is not None:
            kwargs["position_ids"] = position_ids

        # Dual-rotary archs (Gemma 3) need BOTH global and local; single-rotary
        # archs (Qwen) take one shared ``position_embeddings``.
        if (
            "position_embeddings_global" in params
            and dec.position_embeddings_global is not None
        ):
            kwargs["position_embeddings_global"] = dec.position_embeddings_global
            if "position_embeddings_local" in params:
                kwargs["position_embeddings_local"] = dec.position_embeddings_local
        elif "position_embeddings" in params:
            pos_emb = self._select_layer_position_embeddings(
                layer, layer_index, dec
            )
            if pos_emb is None and position_ids is not None:
                rotary_emb = self._get_rotary_embedding_module()
                if rotary_emb is not None:
                    pos_emb = rotary_emb(hidden_states, position_ids)
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb

        if "use_cache" in params:
            kwargs["use_cache"] = False
        if "output_attentions" in params:
            kwargs["output_attentions"] = False
        if "cache_position" in params:
            kwargs["cache_position"] = dec.cache_position
        if "shared_kv_states" in params:
            # One dict per block (dec is rebuilt per block) so a KV-publishing
            # layer has a live write target, mirroring HF's model-level dict.
            if dec.shared_kv_states is None:
                dec.shared_kv_states = {}
            kwargs["shared_kv_states"] = dec.shared_kv_states

        output = layer(hidden_states, **kwargs)
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    # ── Input building ─────────────────────────────────────────────────

    def _build_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Build hidden_states, attention_mask, position_ids, labels from a batch."""
        head = getattr(self, "head", None)
        if head is not None and getattr(self.model, "training", False):  # a training forward:
            head.drop_eval_weight()  # the head's eval copy gives its room back first
        input_ids = batch.get("input_ids")
        inputs_embeds = batch.get("inputs_embeds")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "Batch must contain either input_ids or inputs_embeds"
                )
            embed_layer = self._get_input_embeddings()
            embed_device = next(embed_layer.parameters()).device
            input_ids = input_ids.to(
                embed_device, non_blocking=embed_device.type == "cuda"
            )
            hidden_states = embed_as(embed_layer, input_ids,
                                     self._table_dtype(embed_layer.weight))
            if (getattr(self.config, "offload_embeddings_and_head", False)
                    and hidden_states.device != self.device):  # offloaded table: rows go up
                hidden_states = hidden_states.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                input_ids = input_ids.to(self.device)

            # Handle absolute position embeddings (GPT-2 style)
            pos_emb = self._get_position_embeddings()
            if pos_emb is not None:
                seq_len = input_ids.shape[1]
                position_ids = batch.get("position_ids")
                if position_ids is None:
                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).unsqueeze(0)
                    position_ids = position_ids.expand(
                        input_ids.shape[0], seq_len
                    )
                else:
                    position_ids = position_ids.to(
                        hidden_states.device,
                        non_blocking=hidden_states.device.type == "cuda",
                    )
                hidden_states = hidden_states + pos_emb(position_ids)
        else:
            hidden_states = inputs_embeds.to(
                self.device, non_blocking=self.device.type == "cuda"
            )

        # Build position_ids for rotary embeddings
        position_ids = batch.get("position_ids")
        if position_ids is None and input_ids is not None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0)
            position_ids = position_ids.expand(hidden_states.shape[0], seq_len)
        elif position_ids is not None:
            position_ids = position_ids.to(
                hidden_states.device,
                non_blocking=hidden_states.device.type == "cuda",
            )

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                hidden_states.device,
                non_blocking=hidden_states.device.type == "cuda",
            )
            if (
                not torch.is_floating_point(attention_mask)
                and attention_mask.dtype is not torch.bool
            ):
                attention_mask = attention_mask.to(torch.bool)

        if labels is not None:
            labels = labels.to(
                hidden_states.device,
                non_blocking=hidden_states.device.type == "cuda",
            )

        return hidden_states, attention_mask, position_ids, labels

    def _prepare_decoder_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> _DecoderInputs:
        """Rebuild the per-layer masks + rotary HF computes at the model level.

        Modern decoder models (Qwen, Gemma 3, ...) compute the attention mask
        and rotary embeddings ONCE in ``Model.forward`` and pass them down to
        every layer. The blockwise loop drives layers directly, so we reproduce
        that step by calling the *same* HF helpers with the *same* arguments:

        * per-attention-type causal masks via ``create_causal_mask`` /
          ``create_sliding_window_causal_mask`` (sliding-window archs need a
          different mask on sliding vs full layers), and
        * dual global/local rotary for Gemma 3, single shared rotary otherwise.

        Qwen (no ``layer_types``, no sliding) keeps the exact prior behavior: a
        single raw mask + single rotary -- so the validated path is untouched.
        """
        backbone = self._get_decoder_backbone()
        cfg = getattr(backbone, "config", None) if backbone is not None else None

        # No KV cache in the blockwise pass: cache positions are just 0..seq-1.
        cache_position = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        )

        # Rotary: dual (global + local) for Gemma 3, single for everything else.
        rotary = (
            getattr(backbone, "rotary_emb", None)
            or self._get_rotary_embedding_module()
        )
        rotary_local = getattr(backbone, "rotary_emb_local", None)
        layer_types = getattr(cfg, "layer_types", None)
        pe = pe_global = pe_local = None
        pe_by_type: Optional[Dict[str, Any]] = None
        if position_ids is not None:
            if rotary is not None and rotary_local is not None:
                # Dual rotary (Gemma 3): separate global + local modules.
                pe_global = rotary(hidden_states, position_ids)
                pe_local = rotary_local(hidden_states, position_ids)
            elif (
                rotary is not None
                and isinstance(layer_types, list)
                and _rotary_takes_layer_type(rotary)
            ):
                # Per-attention-type rotary (Gemma 4): one module yielding
                # different frequencies per layer type. Build one embedding per
                # unique type, exactly as HF's Model.forward does.
                pe_by_type = {
                    lt: rotary(hidden_states, position_ids, lt)
                    for lt in dict.fromkeys(layer_types)
                }
            elif rotary is not None:
                pe = rotary(hidden_states, position_ids)

        dec = _DecoderInputs(
            mask=attention_mask,
            position_embeddings=pe,
            position_embeddings_global=pe_global,
            position_embeddings_local=pe_local,
            position_embeddings_by_type=pe_by_type,
            cache_position=cache_position,
        )

        # Per-attention-type masks are only needed for sliding-window archs.
        # Skip entirely for qwen-style single-mask models (behavior unchanged).
        # NOTE: build the masks even when the incoming ``attention_mask`` is
        # None -- HF derives them from config + cache_position regardless, and a
        # sliding layer with no mask would silently attend full-context.
        has_sliding = getattr(backbone, "has_sliding_layers", False) or (
            isinstance(layer_types, list) and "sliding_attention" in layer_types
        )
        if cfg is None or isinstance(attention_mask, dict) or position_ids is None:
            return dec
        typed = isinstance(layer_types, list) or has_sliding
        # On a Mac eager attention needs its causal mask, and OLMoE's config has no layer types:
        # there every config takes the mapping below, padded or not.
        if not typed and attention_mask is None and not self._metal:
            return dec  # no padding mask: every attention takes HF's causal default

        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        # The mask helpers' signature drifted across transformers versions: the
        # embeds arg was ``input_embeds`` (4.x) then ``inputs_embeds`` (5.x),
        # and ``cache_position`` was dropped in 5.x. Bind only the parameters
        # the installed version actually accepts so we stay faithful on both.
        available = {
            "config": cfg,
            "input_embeds": hidden_states,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        accepted = set(inspect.signature(create_causal_mask).parameters)
        mask_kwargs = {k: v for k, v in available.items() if k in accepted}
        if not typed and not self._metal:
            # A config without layer types (qwen3_moe, olmoe, ...): HF's Model.forward
            # still turns the 2-D padding mask into its causal mask with this helper.
            # Handing the raw 2-D mask to the layers made attention non-causal (block-
            # major OLMoE on oasst).
            dec.mask = create_causal_mask(**mask_kwargs)
            return dec
        mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if has_sliding:
            mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )
        if isinstance(layer_types, list) and "linear_attention" in layer_types:
            # Gated DeltaNet layers multiply by a 2-D padding mask, never a 4-D
            # causal one (HF's ``create_recurrent_attention_mask``). Eager on a Mac
            # builds a causal mask even with no padding, so it needs this as well.
            mapping["linear_attention"] = _recurrent_mask(
                cfg, hidden_states, attention_mask
            )
        dec.mask = mapping

        return dec

    # ── Module accessors ───────────────────────────────────────────────

    def _get_attr_by_path(self, root: nn.Module, path: str) -> Optional[Any]:
        """Resolve a dotted attribute path, searching through wrapper layers."""

        def _resolve(candidate: Any, candidate_path: str) -> Any:
            obj: Any = candidate
            for attr in candidate_path.split("."):
                obj = getattr(obj, attr)
            return obj

        candidates: List[Any] = [root]
        for wrapper_path in [
            "model",
            "base_model",
            "base_model.model",
            "base_model.model.model",
        ]:
            try:
                candidates.append(_resolve(root, wrapper_path))
            except AttributeError:
                continue

        get_base_model = getattr(root, "get_base_model", None)
        if callable(get_base_model):
            try:
                candidates.append(get_base_model())
            except Exception:
                pass

        seen: set[int] = set()
        for candidate in candidates:
            cid = id(candidate)
            if cid in seen:
                continue
            seen.add(cid)
            try:
                return _resolve(candidate, path)
            except AttributeError:
                continue
        return None

    def _get_input_embeddings(self) -> nn.Module:
        assert self.model is not None
        getter = getattr(self.model, "get_input_embeddings", None)
        if callable(getter):
            module = getter()
            if module is not None:
                return module
        for path in ["model.embed_tokens", "gpt_neox.embed_in", "transformer.wte"]:
            module = self._get_attr_by_path(self.model, path)
            if isinstance(module, nn.Module):
                return module
        raise AttributeError("Could not locate the model's input embedding layer")

    def _get_position_embeddings(self) -> Optional[nn.Module]:
        assert self.model is not None
        module = self._get_attr_by_path(self.model, "transformer.wpe")
        return module if isinstance(module, nn.Module) else None

    def _get_rotary_embedding_module(self) -> Optional[nn.Module]:
        assert self.model is not None
        for path in [
            "model.rotary_emb",
            "transformer.rotary_emb",
            "gpt_neox.rotary_emb",
        ]:
            module = self._get_attr_by_path(self.model, path)
            if isinstance(module, nn.Module):
                return module
        return None

    def _get_backbone_module(self) -> Optional[nn.Module]:
        assert self.model is not None
        for path in ["model", "gpt_neox", "transformer", "model.decoder"]:
            module = self._get_attr_by_path(self.model, path)
            if isinstance(module, nn.Module):
                return module
        return None

    def _get_decoder_backbone(self) -> Optional[nn.Module]:
        """The module that actually owns the decoder ``layers`` ModuleList.

        ``_get_backbone_module`` can land on the ``...ForCausalLM`` wrapper,
        whose ``rotary_emb`` / ``rotary_emb_local`` and ``config.layer_types``
        live one level deeper (on e.g. ``Gemma3TextModel``). Descend through the
        usual nesting attributes -- ``.model`` and the VLM ``.language_model`` /
        ``.text_model`` -- until we reach the module holding the layer list so
        mask + rotary reconstruction reads the right config and rotary submodules.
        """
        candidate = self._get_backbone_module()
        seen: set[int] = set()
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            layers = getattr(candidate, "layers", None)
            if isinstance(layers, nn.ModuleList):
                return candidate
            nxt = None
            for attr in ("model", "language_model", "text_model"):
                child = getattr(candidate, attr, None)
                if isinstance(child, nn.Module) and id(child) not in seen:
                    nxt = child
                    break
            candidate = nxt
        return self._get_backbone_module()

    def _move_static_modules_to_device(self, device: torch.device) -> None:
        """Move small, always-used modules (embed, norm, head) to device.

        A frozen table k-bit preparation left in its load dtype is cast to fp32 on the
        device first, a chunk at a time (the same tensor PEFT's host cast made). With
        ``offload_embeddings_and_head`` the embedding table and LM head stay in host RAM
        instead, in their load dtype, cast as each use reads them; norms still move.
        """
        assert self.model is not None
        uncast, self._uncast_tables = self._uncast_tables, []
        if getattr(self.config, "offload_embeddings_and_head", False):
            self._compact_tables = {id(p): torch.float32 for p in uncast}
            self._offload_embeddings_and_head()
            final_norm = self._get_final_norm()
            if isinstance(final_norm, nn.Module):
                final_norm.to(device)
            for path in ["transformer.wpe"]:
                module = self._get_attr_by_path(self.model, path)
                if isinstance(module, nn.Module):
                    module.to(device)
            return
        for table in uncast:  # PEFT's cast, made on the device: no fp32 copy in host RAM
            table.data = cast_to(table.data, device, torch.float32)
        for path in [
            "model.embed_tokens",
            "gpt_neox.embed_in",
            "transformer.wte",
            "transformer.wpe",
            "model.norm",
            "gpt_neox.final_layer_norm",
            "transformer.ln_f",
            "model.decoder.final_layer_norm",
            "lm_head",
            "embed_out",
        ]:
            module = self._get_attr_by_path(self.model, path)
            if isinstance(module, nn.Module):
                module.to(device)
        # VLM-nested backbones (e.g. Gemma 3 multimodal) keep these modules
        # under ``language_model``; resolve them through the same accessors
        # the forward path uses so they always land on the compute device.
        getter = getattr(self.model, "get_input_embeddings", None)
        resolved = [getter() if callable(getter) else None]
        resolved.append(self._get_final_norm())
        try:
            resolved.append(self._get_lm_head())
        except AttributeError:
            pass
        for module in resolved:
            if isinstance(module, nn.Module):
                module.to(device)

    def _offload_embeddings_and_head(self) -> None:
        """Keep the input embedding and LM head in host RAM, pinned if granted."""
        from poros.sss import pinning

        getter = getattr(self.model, "get_input_embeddings", None)
        embed = getter() if callable(getter) else None
        head = self._get_lm_head()
        for module in (embed, head):
            if not isinstance(module, nn.Module):
                continue
            module.to("cpu")
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    raise RuntimeError("offload_embeddings_and_head: a table is trainable")
                # The embedding lookup runs on the CPU (only the gathered rows go up), so
                # an untied input table gains nothing from page-locking (72B: 4 GiB locked
                # for nothing). The head's weight is copied up whole: pinned. (Registering
                # it in place with cudaHostRegister broke later pageable copies under WSL2,
                # so it stays a pin_memory copy.) A compact head is
                # not: torch would lock a power of two (72B: 4 GiB for 2.3), and it crosses
                # in half the bytes of the fp32 one it replaces.
                if (module is not head or p.data.is_pinned() or not torch.cuda.is_available()
                        or id(p) in self._compact_tables):
                    continue
                # The efficient path: the boundaries get the page-locked RAM instead. Hosts can
                # cap locked memory (a 57 GB WSL2 3090 node stopped at ~31 GiB), and
                # the head's pinned copy is 8 GiB (power of two) for one upload per update; a
                # pageable upload costs ~0.5 s per update, the boundaries save their whole VRAM.
                # A run that keeps no boundaries in host RAM keeps its pinned head.
                if train_path(self.config) == "efficient":
                    continue
                nbytes = pinning.pinned_size(p.data.numel() * p.data.element_size())
                if pinning.reserve(nbytes):
                    try:
                        p.data = p.data.pin_memory()
                    except RuntimeError:
                        pinning.release(nbytes)
                    else:  # given back with the table
                        import weakref

                        weakref.finalize(p.data.untyped_storage(), pinning._freed, nbytes)
        self.head.weight_device = None

    def _move_trainable_parameters_to_device(self) -> None:
        """Move trainable adapter parameters to the compute device."""
        assert self.model is not None
        use_non_blocking = self.device.type == "cuda"
        for param in self.model.parameters():
            if param.requires_grad and param.data.device != self.device:
                param.data = param.data.to(
                    self.device, non_blocking=use_non_blocking
                )

    def _partition_trainable_parameters_by_block(
        self,
    ) -> Tuple[
        Dict[int, List[Tuple[str, nn.Parameter]]],
        List[Tuple[str, nn.Parameter]],
    ]:
        """Split trainable params into per-block and static groups."""
        assert self.block_manager is not None
        assert self.model is not None

        block_params: Dict[int, List[Tuple[str, nn.Parameter]]] = {
            block.block_id: [] for block in self.block_manager.blocks
        }
        static_params: List[Tuple[str, nn.Parameter]] = []

        block_param_ids = {
            block.block_id: {
                id(param)
                for module in self.block_manager.get_block_modules(block.block_id)
                for param in module.parameters()
            }
            for block in self.block_manager.blocks
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            assigned = False
            for block_id, param_ids in block_param_ids.items():
                if id(param) in param_ids:
                    block_params[block_id].append((name, param))
                    assigned = True
                    break
            if not assigned:
                static_params.append((name, param))

        return block_params, static_params

    # ── Optimizer ──────────────────────────────────────────────────────

    def _forward_selfcheck(
        self, model: nn.Module, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Minimal forward pass for the startup parity self-check.

        Runs the blockwise forward path and returns a scalar loss.
        """
        batch = {"input_ids": input_ids, "labels": input_ids}
        batch = self._move_batch_to_device(batch)
        hidden_states, attention_mask, position_ids, labels = (
            self._build_inputs(batch)
        )
        assert self.block_manager is not None
        for block_id in range(self.block_manager.num_blocks):
            self.block_manager.load_block_for_inference(
                block_id, self.device
            )
            hidden_states = self._forward_block_layers(
                block_id, hidden_states, attention_mask, position_ids,
                checkpoint=False,
            )
            self.block_manager.release_block(block_id)
        return self.head.loss(hidden_states, labels, normed=False)  # the training loop's loss

    def _forward_selfcheck_resident(
        self, model: nn.Module, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Resident reference forward for the parity self-check.

        Identical layer math to ``_forward_selfcheck`` -- same ``_build_inputs``,
        ``_forward_block_layers``, norm, and head -- but with **all blocks loaded
        and kept resident** (no per-block release). The ONLY difference from the
        blockwise path is whether weights were streamed (load/compute/release) or
        held resident, so comparing the two isolates the streaming mechanism as
        the sole variable. Bitwise-equal losses => streaming preserves parity.
        """
        batch = {"input_ids": input_ids, "labels": input_ids}
        batch = self._move_batch_to_device(batch)
        hidden_states, attention_mask, position_ids, labels = (
            self._build_inputs(batch)
        )
        assert self.block_manager is not None
        num_blocks = self.block_manager.num_blocks
        # Loading is inside the try: if the resident load OOMs partway (the
        # model may simply not fit resident — the reason Poros exists), the
        # blocks loaded so far MUST be released, or the determinism-only
        # fallback self-check inherits a full card and dies on its first
        # kernel (observed at 72B on 24/32 GB cards: cublasCreate
        # CUBLAS_STATUS_ALLOC_FAILED in the fallback's rope forward).
        try:
            for block_id in range(num_blocks):
                self.block_manager.load_block_for_inference(block_id, self.device)
            for block_id in range(num_blocks):
                hidden_states = self._forward_block_layers(
                    block_id, hidden_states, attention_mask, position_ids,
                    checkpoint=False,
                )
            return self.head.loss(hidden_states, labels, normed=False)  # as the blockwise one
        finally:
            for block_id in range(num_blocks):
                try:
                    self.block_manager.release_block(block_id)
                except (ValueError, KeyError):
                    pass

    def _forward_eval_blockwise(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Blockwise forward for evaluation (no grad)."""
        return self._compute_loss(*self._forward_eval_logits(batch))

    @torch.no_grad()
    def _forward_eval_logits(
        self, batch: Dict[str, torch.Tensor], hidden_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The no-grad forward up to the logits (eval, reference logprobs). ``hidden_only``:
        the last block's output instead (a blockwise model; the caller runs the head)."""
        if self.block_manager is None:  # poros.moe: the model streams itself
            assert self.model is not None
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            return out.logits, batch.get("labels")
        hidden_states, attention_mask, position_ids, labels = (
            self._build_inputs(batch)
        )
        for block_id in range(self.block_manager.num_blocks):
            self.block_manager.load_block_for_inference(
                block_id, self.device
            )
            hidden_states = self._forward_block_layers(
                block_id,
                hidden_states,
                attention_mask,
                position_ids,
                checkpoint=False,
            )
            self.block_manager.release_block(block_id)

        if hidden_only:
            return hidden_states, labels
        return self._logits_from_hidden(hidden_states), labels

    # ── Dataset preparation ────────────────────────────────────────────
