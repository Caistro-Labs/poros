"""The trainer's compute: what it installs and what it records (``PorosTrainer.compute``).

``poros.core.compute.choose`` decides from the config and the machine, for the trainer and
its plan alike. Here the trainer applies the decision to the model that loaded -- the dense
fused compute (``install_dense``), a MoE model's expert compute (``moe_request``, then
``prepare_moe`` with ``"auto"`` where the model's layers decide) and its fused extras
(``install_moe``) -- and records, beside the settings (``fused_compute``,
``moe_expert_compute``: left as set), what ran (the config's ``ran_compute``, with the caches'
resolved sizes) and why a default took the slower compute (``compute_reason``), as ``ran_path``
and ``path_reason`` sit beside ``path``.

It also sizes a MoE model's device expert cache (``moe_expert_cache_gib``). Unset, ``path:
efficient`` (the lowest-VRAM path) keeps one layer's experts; elsewhere block-major (on either
store) and the SSD store's sequential loop size it before update 2 to what fits this GPU
(``poros.sss.tune.grow_expert_cache``): the VRAM this process could hold at setup, minus the run's
reserved peak through update 1 and its optimizer step (whose first step makes Adam's state;
``stepped``), minus what an update at the run's full shape would add where update 1 was smaller
(``poros.plan.estimate.token_bytes``, an estimate that errs high), minus a margin of ``max(3 GiB,
0.2 x budget)``; it grows again, never shrinking, after the first update at the full shape is
measured. An update beyond the full shape (rows longer than any measured, from a caller's loader)
drops it to one layer for the rest of the run. The cache decides when bytes move, never which, so no
value depends on its size.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch import nn

from poros.core.config import moe_grouping, train_path, trains_block_major
from poros.core.errors import PorosConfigError
from poros.preflight.compute import _RANK_REMEDY, _adapter_reason

if TYPE_CHECKING:
    from poros.moe.prepare import MoeReport
    from poros.train.trainer import PorosTrainer

logger = logging.getLogger(__name__)

class RunCompute:
    """One trainer's compute (see the module docstring)."""

    def __init__(self, trainer: "PorosTrainer") -> None:
        self._t = trainer
        self.ran: Dict[str, Any] = {}
        self.reasons: List[str] = []
        #: the VRAM this process could hold when training started (bytes; None: not measured)
        self.vram_budget: Optional[int] = None
        # the expert cache's sizing: the run's full update shape (count, rows, length), set by
        # update 1; whether the cache was sized, at the full shape, or dropped for good
        self._full: Optional[tuple[int, int, int]] = None
        self._pending: Optional[List[tuple[int, int]]] = None  # the update to size on next
        self._stepped = False  # an optimizer step ran since _pending's update: its peak is in
        self._sized = self._sized_full = self._dropped = False

    # -- what ran ---------------------------------------------------------------------------

    @property
    def fused_ran(self) -> bool:
        """Whether the fused compute ran (dense: the decoder layers; MoE: its extras)."""
        return bool(self.ran.get("fused_compute"))

    def record(self) -> Dict[str, Any]:
        """What computes this run, for its setup event: fused_compute as it ran, on a MoE model
        moe_expert_compute as it ran, and why a default took the slower compute (compute_reason,
        where it did)."""
        out: Dict[str, Any] = {"fused_compute": self.fused_ran}
        if "moe_expert_compute" in self.ran:
            out["moe_expert_compute"] = self.ran["moe_expert_compute"]
        if "expert_optimizer" in self.ran:
            out["expert_optimizer"] = self.ran["expert_optimizer"]
        if self._t.config.compute_reason:
            out["compute_reason"] = self._t.config.compute_reason
        return out

    def note(self, **ran: Any) -> None:
        """Record what computes the run into the config (ran_compute, compute_reason)."""
        self.ran.update(ran)
        self._t.config.ran_compute = dict(self.ran)
        self._t.config.compute_reason = "; ".join(self.reasons) or None

    def record_cache(self, *, moe_expert_cache_bytes: Optional[int] = None,
                     nf4_dequant_cache_mb: Optional[int] = None,
                     reason: Optional[str] = None) -> None:
        """Record a cache's resolved size (the expert cache in GiB, as its setting is, and how it
        was sized)."""
        if moe_expert_cache_bytes is not None:
            self.note(moe_expert_cache_gib=round(moe_expert_cache_bytes / 2**30, 3),
                      moe_expert_cache_reason=reason)
        if nf4_dequant_cache_mb is not None and self.ran.get("nf4_dequant_cache_mb") \
                != int(nf4_dequant_cache_mb):
            self.note(nf4_dequant_cache_mb=int(nf4_dequant_cache_mb))

    # -- a dense model ----------------------------------------------------------------------

    def preflight(self, entry: str, model_json: Optional[dict] = None,
                  model_type: Optional[str] = None) -> None:
        """Before the download: what install_dense, prepare_moe and install_moe would refuse
        once the model loaded (``poros.preflight.compute.refuse``)."""
        from poros.preflight.compute import refuse

        refuse(self._t, entry, model_json, model_type)

    def install_dense(self) -> None:
        """fused_compute (None: where it applies, poros.core.compute): route the decoder layers'
        LoRA-over-NF4 compute through poros.core.fused, and record what ran and why the default
        did not take it."""
        from poros.core.compute import choose

        t = self._t
        want = t.config.fused_compute
        cuda = t.device.type == "cuda"
        choice = choose(t.config, entry="dense", cuda=cuda)
        why = choice.blocked  # as set (True) too: a refusal says why
        if choice.fused:
            why = _adapter_reason(getattr(t, "_adapter_backend", None) == "peft"
                                  and t.model is not None)
        if choice.fused and why is None:
            from poros.core import fused

            why = fused.available(t.device)
        if not choice.fused or why is not None:
            if want:
                raise PorosConfigError(f"fused_compute=True cannot run here: {why}.",
                                       remediation="Set fused_compute: false (PEFT's modules).")
            if why is not None and cuda:  # a Mac or the CPU has nothing to route: no reason
                logger.info("fused compute off: %s", why)
                self.reasons.append(why)
            self.note(fused_compute=False)
            return
        from poros.core import fused

        counts = fused.install(t.model)
        ran = bool(counts["layers"] or counts["swiglu_mlps"] or counts["linears"])
        if not ran:
            why = "; ".join(fused.refusals(t.model)) or "no LoRA layer the fused compute takes"
            if want:
                raise PorosConfigError(f"fused_compute=True cannot run here: {why}.",
                                       remediation=_RANK_REMEDY)
            logger.info("fused compute off: %s", why)
            self.reasons.append(why)
        self.note(fused_compute=ran)

    def auto_dequant_cache_mb(self) -> int:
        """nf4_dequant_cache_mb=None (auto): on the fast path, the largest block's dequantized
        4-bit weights (bf16), so a whole block's are kept, capped at half the device's memory;
        0 elsewhere (the efficient path, the sequential loop). (A quarter of a 24 GB
        card held 6 of a 72B block's 7.0 GB at block size 4: the rest were dequantized again
        for every micro-batch, 2,000 dequantizations an update where 560 would do.)"""
        t = self._t
        bm = t.block_manager
        if bm is None or train_path(t.config) != "fast":
            return 0
        if t.device.type != "cuda" or not torch.cuda.is_available():
            return 0
        try:
            import bitsandbytes as bnb
        except ImportError:
            return 0
        largest = 0
        for block in bm.blocks:
            n = 0
            for module in bm.get_block_modules(block.block_id):
                for m in module.modules():
                    if isinstance(m, bnb.nn.Linear4bit):
                        n += int(m.in_features) * int(m.out_features) * 2
            largest = max(largest, n)
        cap = torch.cuda.get_device_properties(t.device).total_memory // 2
        return int(min(largest, cap) >> 20) + 1 if largest else 0

    # -- a poros.moe model ------------------------------------------------------------------

    def moe_request(self) -> str:
        """The expert compute prepare_moe is asked for (poros.core.compute.choose on this
        device: ``auto`` where it picks fused, the model's layers deciding); a reason the default
        took the reference is recorded. An explicit ``fused`` this machine cannot run is
        refused."""
        from poros.core.compute import choose, triton_installed

        t = self._t
        cfg = t.config
        cuda, triton = t.device.type == "cuda", triton_installed()
        card = None
        if cuda:
            from poros.core.card import capability, triton_reason

            card = triton_reason(capability(t.device))
        if cfg.moe_expert_compute == "fused" and not (cuda and triton and card is None):
            raise PorosConfigError(
                "moe_expert_compute: fused needs a CUDA GPU with Triton installed"
                + ("" if cuda else f"; this run is on {t.device}")
                + (f"; {card}" if card else ""),
                remediation="Leave moe_expert_compute unset (the reference compute where fused "
                            "cannot run), or set it to reference.")
        if cfg.moe_expert_compute == "grouped" and cuda:  # pinned per host: said before the load
            from poros.moe import grouped

            try:
                grouped.activate()
            except (grouped.GroupedPinError, ImportError) as exc:
                raise PorosConfigError(
                    f"moe_expert_compute: grouped cannot run on this host: {exc}",
                    remediation="Leave moe_expert_compute unset (fused where it runs, else the "
                                "reference), or record a pin for this host with "
                                "scripts/moe_grouped_pin.py.") from exc
        choice = choose(cfg, entry="moe", cuda=cuda, triton=triton, gpu_reason=card)
        if choice.reason:
            self.reasons.append(choice.reason)
        return choice.moe_expert

    def prepare_moe(self, model: nn.Module, compute: str) -> "MoeReport":
        """prepare_moe on the loaded model, asked for ``compute``; records the expert compute
        that runs (and, where "auto" took the reference, why). An explicit fused the model's
        experts cannot take is a PorosConfigError."""
        from poros.moe.prepare import prepare_moe, streaming_config

        t = self._t
        config = t.config
        try:
            report = prepare_moe(
                model,
                streaming_config(
                    model.config,
                    # a plugin store's slabs are not NF4-sized: budget its own largest layer
                    # (and a store whose layers differ in size, Inkling 2's released experts)
                    store=None if (getattr(model, "_poros_store_plugin", None) is None
                                   and not hasattr(model._poros_moe_store, "max_layer_nbytes"))
                    else model._poros_moe_store,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    seed=config.seed,
                    device=str(t.device),
                    expert_compute=compute,
                    expert_grouping=moe_grouping(config),
                    # block-major: one MoE layer per block, so its experts decode once for all G
                    dense_block_size=1 if trains_block_major(config) else None,
                ),
            )
        except ValueError as exc:  # an explicit fused the model's experts cannot take
            if "cannot run this model" not in str(exc):
                raise
            raise PorosConfigError(
                f"moe_expert_compute: fused: {exc}",
                remediation="Leave moe_expert_compute unset (the reference compute where fused "
                            "cannot run), or set it to reference.") from exc
        if report.expert_compute_reason:
            self.reasons.append(report.expert_compute_reason)
            logger.warning("moe_expert_compute: reference: %s", report.expert_compute_reason)
        self.note(moe_expert_compute=report.expert_compute)
        return report

    def install_moe(self) -> None:
        """With the fused expert compute running and fused_compute not false: the fused MoE
        path's extras (poros.core.fused.extras: attention through the memory-efficient SDPA
        kernel, RMSNorms in one kernel each way, the loss a chunk of rows at a time) and its head
        (``DeviceHead.moe_fused``), in every loop. The reference expert compute keeps
        transformers' modules. Records what ran."""
        t = self._t
        want = t.config.fused_compute
        if t.device.type != "cuda" or self.ran.get("moe_expert_compute") != "fused":
            if want:
                raise PorosConfigError(
                    "fused_compute=True on a MoE model needs moe_expert_compute: fused on a CUDA "
                    "GPU.", remediation="Set moe_expert_compute: fused, or fused_compute: false.")
            self.note(fused_compute=False)
            return
        if want is False:
            self.note(fused_compute=False)
            return
        from poros.core.fused.extras import install_extras

        install_extras(t.model)
        self.note(fused_compute=True)
        lm_head = t._get_lm_head()
        text = getattr(t.model.config, "get_text_config", lambda: t.model.config)()
        t.head.moe_fused = (isinstance(lm_head, nn.Linear) and lm_head.bias is None
                            and not lm_head.weight.requires_grad
                            and t._final_logit_softcapping() is None
                            and not getattr(text, "output_router_logits", False)
                            and not getattr(t.config, "offload_embeddings_and_head", False)
                            and callable(getattr(t.model, "get_decoder", None)))

    def moe_cache_cap_bytes(self) -> Optional[int]:
        """moe_expert_cache_gib as bytes. Unset: 0 (one layer) on path efficient, the lowest-VRAM
        path; None (no cap: what fits this GPU) elsewhere."""
        gib = getattr(self._t.config, "moe_expert_cache_gib", None)
        if gib is None:
            return 0 if train_path(self._t.config) == "efficient" else None
        return int(float(gib) * (1 << 30))

    # -- the MoE expert cache ---------------------------------------------------------------

    def measure_budget(self) -> None:
        """When training starts: the VRAM this process may hold (poros.sss.tune.vram_budget),
        within which the expert cache is sized."""
        if self._t.device.type == "cuda":
            from poros.sss.tune import vram_budget

            self.vram_budget = vram_budget(self._t.device)

    def cache_prepared(self, capacity: int) -> None:
        """Record the expert cache prepare_moe gave the model (one layer's experts)."""
        set_cap = getattr(self._t.config, "moe_expert_cache_gib", None) is not None
        path = train_path(self._t.config)
        reason = ("one layer: path efficient, the lowest-VRAM path"
                  if path == "efficient" and not set_cap
                  else "one layer: the sequential loop does not grow it"
                  if path == "sequential" and self._t.config.store != "ssd"
                  else "one layer until update 1 is measured")
        self.record_cache(moe_expert_cache_bytes=capacity, reason=reason)

    def cache_after_update(self, model: nn.Module, shapes: List[tuple[int, int]]) -> None:
        """A MoE update of micro-batches ``shapes`` (rows, length) ran: block-major on either
        store, or sequentially on the SSD store. Its peak is sized on before the next update
        (``cache_before_update``), once its optimizer step has allocated too (``stepped``:
        Adam's state is made at the first step). Update 1 sets the run's full shape."""
        cfg = self._t.config
        if self._full is None:
            self._full = (max(cfg.gradient_accumulation_steps, len(shapes)),
                          max([cfg.per_device_batch_size] + [r for r, _ in shapes]),
                          max([cfg.max_seq_length] + [n for _, n in shapes]))
        self._pending, self._stepped = list(shapes), False

    def stepped(self) -> None:
        """The optimizer stepped: the update before it is measured through its step."""
        self._stepped = True

    def cache_before_update(self, model: nn.Module, shapes: List[tuple[int, int]]) -> None:
        """Before a block-major MoE update of micro-batches ``shapes`` (rows, length). One beyond
        the run's full shape has no measured or estimated bound beside the cache, so the cache
        goes back to one layer for the rest of the run. Otherwise the update before it, measured
        through its optimizer step, sizes the cache (``_size``); one whose step has not run
        (``poros.Run``: another forward_backward before optim_step) waits for it."""
        pending = self._pending if self._stepped else None
        if pending is not None:
            self._pending = None
        if self._dropped or self._full is None:
            return
        count, rows, length = self._full
        if len(shapes) <= count and all(r <= rows and n <= length for r, n in shapes):
            if pending is not None:
                self._size(model, pending)
            return
        self._dropped = True
        residency = _residency(model)
        if residency is not None and hasattr(residency, "shrink_to_floor"):
            residency.shrink_to_floor()
        why = (f"one layer: an update beyond the run's full shape ({count} x {rows} rows x "
               f"{length} tokens) came, with no bound to size the cache beside")
        logger.warning("MoE expert cache: %s", why)
        self.record_cache(moe_expert_cache_bytes=getattr(residency, "_capacity", 0), reason=why)

    def _size(self, model: nn.Module, shapes: List[tuple[int, int]]) -> None:
        """Size the expert cache to what fits (the module docstring) on the run's peak through an
        update of ``shapes``: after update 1, and again, larger only, after the first update at
        the run's full shape, if update 1 was smaller."""
        cfg = self._t.config
        if self._sized_full:
            return
        if getattr(cfg, "moe_expert_cache_gib", None) is None and \
                train_path(cfg) == "efficient":
            return  # one layer: recorded at prepare
        count, rows, length = self._full
        full = len(shapes) >= count and all(r >= rows and n >= length for r, n in shapes)
        if self._sized and not full:
            return
        longer = 0.0
        if not full:
            from poros.plan.estimate import token_bytes

            text = getattr(model.config, "get_text_config", lambda: model.config)().to_dict()
            whole = token_bytes(text, path=train_path(cfg), entry="moe", block_size=None,
                                shapes=[(rows, length)] * count)
            this = token_bytes(text, path=train_path(cfg), entry="moe", block_size=None,
                               shapes=shapes)
            if whole is None or this is None:  # unsized: wait for an update at the full shape
                self.record_cache(
                    moe_expert_cache_bytes=getattr(_residency(model), "_capacity", 0),
                    reason="one layer until an update at the run's full shape is measured "
                           "(config.json does not size a larger one)")
                return
            longer = max(whole - this, 0.0)
        from poros.sss.tune import grow_expert_cache

        fit = grow_expert_cache(model, self._t.device, self.moe_cache_cap_bytes(),
                                budget=self.vram_budget, longer=int(longer))
        self._sized, self._sized_full = True, full
        if fit is not None:
            self.record_cache(moe_expert_cache_bytes=fit.capacity, reason=fit.reason())
            # block-major on either store, or the SSD store's sequential loop: said as what it is
            logger.warning("MoE expert cache %.2f GiB, sized on %s (%s)",
                           fit.capacity / 2**30,
                           "an update at the full shape" if full else "update 1", fit.reason())


def _residency(model: nn.Module) -> Any:
    """A prepared MoE model's expert residency (one, shared by its layers), or None."""
    from poros.moe.modules import ManagedMoeBlock

    return next((m.residency for m in model.modules() if isinstance(m, ManagedMoeBlock)), None)
