"""The trainer's LM head: the training loss from the final decoder output, in both loops.

``DeviceHead.loss(hidden, labels, normed=...)`` is the one place the loss is chosen, for the
sequential loop (``PorosTrainer._forward_blockwise_adapter_model``) and block-major accumulation's
head (``poros.train.block_major``), so the two compute it the same way, op for op:

* a ``poros.moe`` model with the fused MoE head (``moe_fused``):
  ``poros.core.fused.head.head_loss``, the final norm first unless ``hidden`` is its output
  already (``normed``);
* a ``poros.moe`` model otherwise: its own loss of its logits (``hf_loss``, what its forward with
  labels computes);
* a dense model: ``poros.core.fused.head.HeadCrossEntropy`` where the fused compute ran and the
  head is a plain linear with no soft-capping (``fused_dense_loss``), else the logits and
  ``PorosTrainer._compute_loss``.

``_Head`` (mixed into PorosTrainer) is the head on the model: the final norm, the LM head,
soft-capping, the logits of the last block's output and the shifted cross-entropy.

It also owns the offloaded head's device copy (``offload_embeddings_and_head``): the upload on its
own stream (``begin_upload`` / ``end_upload``) -- from page-locked row chunks where the pin budget
allows (``pinned_chunks``), else from a helper thread over the pageable weight -- the copy itself
(``weight_device``, kept for one update) and whether the head's values are exactly bf16 ones.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
from torch import nn

from poros.core.tables import cast_to
from poros.train.state import _TrainerState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from poros.train.trainer import PorosTrainer


class DeviceHead:
    """One trainer's head (see the module docstring)."""

    def __init__(self, trainer: "PorosTrainer") -> None:
        self._trainer = trainer
        #: the fused MoE head runs (set where the fused MoE extras are installed)
        self.moe_fused = False
        #: the offloaded head's device copy for the current update (None: none)
        self.weight_device: Optional[torch.Tensor] = None
        #: an upload begun and not yet waited on (block-major lands it if an update is given up)
        self.upload_pending: Any = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._bf16_exact: Optional[bool] = None
        self._uploads = 0  # begin_upload calls so far
        self._pinned: Optional[List[Tuple[int, int, torch.Tensor]]] = None  # decided once
        #: the offloaded head's device copy for evaluation and decoding (model in eval mode):
        #: made once and kept across calls, dropped when a training forward begins
        self._eval_weight: Optional[torch.Tensor] = None
        self._eval_key: Optional[tuple] = None

    # rows of the offloaded head per page-locked chunk: at most 256 MiB, so the pin budget's
    # arena carves them from its slabs (no power-of-two rounding of the whole head)
    CHUNK_BYTES = 256 << 20

    # -- the eval copy -------------------------------------------------------------------------

    def eval_weight(self, weight: torch.Tensor, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        """The offloaded head's ``weight`` on ``device`` in ``dtype`` for a pass in eval mode:
        copied once and reused by every later eval or decode call (each one copied it again:
        2.5 GB a decode step at 72B), until a training forward drops it (``drop_eval_weight``).
        The same cast as a training call's, so the logits are the same."""
        key = (id(weight), weight._version, device, dtype)
        if self._eval_weight is None or self._eval_key != key:
            self._eval_weight = None  # the old copy goes before the new one is made
            self._eval_weight = cast_to(weight, device, dtype)
            self._eval_key = key
        return self._eval_weight

    def drop_eval_weight(self) -> None:
        """Free the eval copy: a training forward needs the room it held."""
        self._eval_weight = self._eval_key = None

    # -- the loss ----------------------------------------------------------------------------

    def loss(self, hidden: torch.Tensor, labels: torch.Tensor, *, normed: bool) -> torch.Tensor:
        """The training loss of the final decoder output ``hidden`` (``normed``: the final norm
        already applied, as a MoE model's decoder returns it)."""
        t = self._trainer
        if getattr(t.model, "_poros_moe_store", None) is not None:
            if self.moe_fused:
                return self.moe_loss(hidden, labels, normed=normed)
            if normed:
                raise ValueError("a MoE model's own loss takes the final norm's input")
            return self.hf_loss(t._logits_from_hidden(hidden), labels)
        loss = self.fused_dense_loss(hidden, labels)
        if loss is not None:
            return loss
        return t._compute_loss(t._logits_from_hidden(hidden), labels)

    def moe_loss(self, hidden: torch.Tensor, labels: torch.Tensor, *,
                 normed: bool) -> torch.Tensor:
        """The fused MoE path's loss (``poros.core.fused.head.head_loss``): the final norm
        (unless ``hidden`` is its output already), then the LM head and the cross-entropy a
        chunk of rows at a time."""
        from poros.core.fused import head

        t = self._trainer
        if not normed:
            final_norm = t._get_final_norm()
            if final_norm is not None:
                hidden = final_norm(hidden)
        return head.head_loss(hidden, t._get_lm_head().weight, labels)

    def hf_loss(self, logits: torch.Tensor, labels: Any) -> torch.Tensor:
        """The loss a ``poros.moe`` model's own forward computes from ``logits`` (HF's
        ``self.loss_function(logits=..., labels=..., vocab_size=...)``), so block-major MoE is op
        for op the sequential MoE path, which calls ``model(..., labels=...)``."""
        model = self._trainer.model
        assert model is not None
        cfg = getattr(model.config, "get_text_config", lambda: model.config)()
        return model.loss_function(logits=logits, labels=labels, vocab_size=cfg.vocab_size)

    def fused_dense_applies(self, hidden: torch.Tensor, labels: Optional[torch.Tensor]) -> bool:
        """Whether a dense model's training loss goes through poros.core.fused.head (a chunk of
        rows at a time): the fused compute ran, on CUDA, the head a plain fp32 linear with no
        soft-capping, labels given. Its eval pass then scores in chunks too
        (``_eval_loss_in_chunks``), so it never holds more than a training step's head does."""
        t = self._trainer
        if (not t.compute.fused_ran or labels is None or t._metal
                or hidden.device.type != "cuda" or t._final_logit_softcapping() is not None
                or getattr(t.model, "_poros_moe_store", None) is not None):
            return False
        lm_head = t._get_lm_head()
        return (isinstance(lm_head, nn.Linear) and lm_head.bias is None
                and t._table_dtype(lm_head.weight) == torch.float32)

    def fused_dense_loss(self, hidden: torch.Tensor,
                         labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """The training loss from the last block's output through poros.core.fused.head (the head
        and cross-entropy a chunk of rows at a time) when the fused compute ran and the head is a
        plain fp32 linear with no soft-capping; else None (the caller takes the logits and
        ``_compute_loss``)."""
        t = self._trainer
        if not self.fused_dense_applies(hidden, labels):
            return None
        lm_head = t._get_lm_head()
        final_norm = t._get_final_norm()
        if final_norm is not None:
            hidden = final_norm(hidden)
        hidden = hidden.to(torch.float32)
        weight = lm_head.weight
        if weight.device != hidden.device:  # offloaded: the update's uploaded copy, or one now
            weight = self.weight_device
            if weight is None or weight.device != hidden.device:
                weight = cast_to(lm_head.weight, hidden.device, self.upload_dtype())
        # Every placement multiplies the same way: in bf16 pieces where the head's values are
        # exactly bf16 ones (an offloaded compact table crosses as bf16; a resident fp32 one is
        # cast down, exactly), else in fp32.
        if self.values_bf16() and weight.dtype == torch.float32:
            weight = weight.to(torch.bfloat16)
        elif weight.dtype != torch.float32 and not self.values_bf16():
            weight = weight.float()
        from poros.core.fused.head import HeadCrossEntropy

        return HeadCrossEntropy.apply(hidden, weight, labels)

    # -- the offloaded head's device copy ---------------------------------------------------

    def begin_upload(self) -> Any:
        """Start the offloaded LM head weight's copy to the device (cast as _logits_from_hidden
        casts it: cast_to, exact) on its own stream: queued from the page-locked chunks where
        there are some (``pinned_chunks``), else from a helper thread, so a pageable copy overlaps
        the GPU's work. The device buffer is allocated here, on the current stream, so it is
        reused as any other once dropped. ``end_upload`` waits for it."""
        t = self._trainer
        head_w = t._get_lm_head().weight
        dtype = self.upload_dtype()
        out = torch.empty(head_w.shape, dtype=dtype, device=t.device)
        if t.device.type != "cuda":
            return None, None, cast_to(head_w, t.device, dtype, out=out), []
        if self._stream is None:
            self._stream = torch.cuda.Stream(t.device)
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())  # the buffer is free on this stream
        done = torch.cuda.Event()
        errors: List[BaseException] = []
        device = t.device
        chunks = self.pinned_chunks()
        self._uploads += 1
        if chunks is not None:  # page-locked: the copies are queued, and run beside the compute
            with torch.cuda.stream(stream):
                for r0, r1, chunk in chunks:
                    cast_to(chunk, device, dtype, out=out[r0:r1])
                done.record(stream)
            return None, done, out, errors

        def work() -> None:
            try:
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    cast_to(head_w, device, dtype, out=out)
                    done.record(stream)
            except BaseException as exc:  # noqa: BLE001 -- re-raised by end_upload
                errors.append(exc)

        thread = threading.Thread(target=work, name="poros-head-upload", daemon=True)
        thread.start()
        return thread, done, out, errors

    def pinned_chunks(self) -> Optional[List[Tuple[int, int, torch.Tensor]]]:
        """The offloaded head's weight as page-locked row chunks ``(first row, end row, rows)``,
        made once, at the second upload: the first update's own page-locked RAM (the efficient
        path's block inputs) is granted first, and the head takes what the budget still allows.
        Carved from shared slabs (poros.sss.pinning.PinnedArena), so 72B's 2.3 GiB head locks
        2.5 GiB, not the 4 GiB a single pin_memory copy rounds to. A copy beside the pageable
        weight, which stays as it is (the same bytes). None where nothing could be locked, or
        where the head is pinned already or not offloaded: the pageable upload then runs."""
        if self._pinned is not None or self._uploads < 1:
            return self._pinned or None
        self._pinned = []  # decided once, whatever the outcome
        t = self._trainer
        head_w = t._get_lm_head().weight
        if (not getattr(t.config, "offload_embeddings_and_head", False)
                or head_w.device.type != "cpu" or head_w.is_pinned() or head_w.dim() != 2
                or not head_w.is_contiguous() or not torch.cuda.is_available()):
            return None
        from poros.sss import pinning

        row_bytes = head_w.shape[1] * head_w.element_size()
        rows = max(1, self.CHUNK_BYTES // row_bytes)
        # the chunks are views of the arena's slabs: when this head (and so they) are freed,
        # the grants go back to the process's pin budget (a second trainer in one process)
        arena = pinning.PinnedArena(head_w.numel() * head_w.element_size(), release_on_free=True)
        chunks = []
        for r0 in range(0, head_w.shape[0], rows):
            r1 = min(r0 + rows, head_w.shape[0])
            chunks.append((r0, r1, arena.pin(head_w.data[r0:r1])))
        if not any(c.is_pinned() for _, _, c in chunks):
            return None
        self._pinned = chunks
        logger.info("offloaded LM head: %d of %d chunks page-locked for its uploads",
                    sum(c.is_pinned() for _, _, c in chunks), len(chunks))
        return chunks

    def end_upload(self, upload: Any) -> torch.Tensor:
        """Wait for ``begin_upload``'s copy (the compute stream after it); its device buffer."""
        thread, done, out, errors = upload
        if thread is not None:
            thread.join()
        if errors:
            raise errors[0]
        if done is not None:
            torch.cuda.current_stream().wait_event(done)
        return out

    def values_bf16(self) -> bool:
        """Whether the LM head's (fp32) values are exactly bf16 ones: a bf16 table (kept compact,
        or the model's own), or an fp32 one equal to its bf16 cast (decided once)."""
        w = self._trainer._get_lm_head().weight
        if w.dtype == torch.bfloat16:
            return True
        if w.dtype != torch.float32:
            return False
        if self._bf16_exact is None:
            with torch.no_grad():
                self._bf16_exact = bool(torch.equal(w.to(torch.bfloat16).float(), w))
        return self._bf16_exact

    def upload_dtype(self) -> torch.dtype:
        """The dtype an offloaded LM head crosses in for the fused head (poros.core.fused.head):
        a compact bf16 table as it is (its fp32 values are exactly its bf16 ones, and the head
        multiplies in bf16 pieces with fp32 sums), else its compute dtype (fp32)."""
        t = self._trainer
        head_w = t._get_lm_head().weight
        if (t.compute.fused_ran and head_w.dtype == torch.bfloat16
                and t._table_dtype(head_w) == torch.float32):
            return torch.bfloat16
        return t._table_dtype(head_w)


class _Head(_TrainerState):
    """PorosTrainer's final norm, LM head and loss (mixed into PorosTrainer)."""

    def _get_final_norm(self) -> Optional[nn.Module]:
        assert self.model is not None
        for path in [
            "model.norm",
            "gpt_neox.final_layer_norm",
            "transformer.ln_f",
            "model.decoder.final_layer_norm",
        ]:
            module = self._get_attr_by_path(self.model, path)
            if isinstance(module, nn.Module):
                return module
        # VLM-nested backbones (e.g. Gemma 3 multimodal) hold the final norm on
        # the text backbone (``language_model.norm``); resolve it generically.
        backbone = self._get_decoder_backbone()
        for attr in ("norm", "final_layer_norm", "ln_f"):
            module = getattr(backbone, attr, None) if backbone is not None else None
            if isinstance(module, nn.Module):
                return module
        return None

    def _get_lm_head(self) -> nn.Module:
        assert self.model is not None
        getter = getattr(self.model, "get_output_embeddings", None)
        if callable(getter):
            module = getter()
            if module is not None:
                return module
        for path in ["lm_head", "embed_out"]:
            module = self._get_attr_by_path(self.model, path)
            if isinstance(module, nn.Module):
                return module
        raise AttributeError("Could not locate the model's LM head")

    def _final_logit_softcapping(self) -> Optional[float]:
        """The model's ``final_logit_softcapping``, or None if it has none.

        Some families (Gemma 2 / Gemma 4) tanh-soft-cap the final logits in
        ``*ForCausalLM.forward``. The blockwise loop drives the head directly,
        so we read the value generically off the config (handling the
        text-config nesting of multimodal models) to replay it faithfully.
        Models without it (Qwen, Gemma 3) return None -> no change.
        """
        backbone = self._get_decoder_backbone()
        for cfg in (
            getattr(backbone, "config", None),
            getattr(self.model, "config", None),
        ):
            if cfg is None:
                continue
            get_text = getattr(cfg, "get_text_config", None)
            tcfg = get_text() if callable(get_text) else cfg
            cap = getattr(tcfg, "final_logit_softcapping", None)
            if cap is not None:
                return cap
        return None

    def _logits_from_hidden(
        self, hidden_states: torch.Tensor, normed: bool = False
    ) -> torch.Tensor:
        """Final norm + LM head + logit soft-capping, matching HF's forward.

        Centralizes the model-tail HF runs after the decoder layers so every
        blockwise path (train, eval, self-check) stays bitwise-faithful,
        including the Gemma 2/4 ``final_logit_softcapping`` transform.
        ``normed``: the final norm is applied already (an eval pass's chunks of rows).
        """
        final_norm = None if normed else self._get_final_norm()
        if final_norm is not None:
            hidden_states = final_norm(hidden_states)
        lm_head = self._get_lm_head()
        dtype = self._table_dtype(lm_head.weight)
        hidden_states = hidden_states.to(dtype)
        if lm_head.weight.device != hidden_states.device:
            # Offloaded head: the same nn.Linear forward on a device copy of
            # the weight (cached for a block-major update), dropped after; a
            # compact head is cast to its compute dtype on the way (exact).
            if not isinstance(lm_head, nn.Linear):
                raise RuntimeError("offload_embeddings_and_head needs an nn.Linear LM head")
            weight = self.head.weight_device
            if (weight is None or weight.device != hidden_states.device
                    or weight.dtype != dtype):
                if not getattr(self.model, "training", True):  # eval or decode: copied once
                    weight = self.head.eval_weight(lm_head.weight, hidden_states.device, dtype)
                else:
                    weight = cast_to(lm_head.weight, hidden_states.device, dtype)
            bias = lm_head.bias
            if bias is not None:
                bias = cast_to(bias, hidden_states.device, self._table_dtype(bias))
            logits = nn.functional.linear(hidden_states, weight, bias)
        else:
            logits = lm_head(hidden_states)
        cap = self._final_logit_softcapping()
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits.float() if self._metal else logits  # a MoE model's hidden is bf16, as on CUDA

    # ── Static module and parameter management ─────────────────────────

    def _table_dtype(self, param: torch.Tensor) -> torch.dtype:
        """The dtype a frozen table computes in: fp32 for one kept compact in host RAM (it owes
        k-bit preparation's cast, poros.core.tables), else its own."""
        return self._compact_tables.get(id(param), param.dtype)

    def _compute_loss(
        self, logits: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute shifted cross-entropy loss."""
        if labels is None:
            raise ValueError("PorosTrainer requires labels to compute loss")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
