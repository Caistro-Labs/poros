"""PorosTrainer: the main training orchestrator for blockwise residency.

This module owns startup: ``from_pretrained`` (the preflight is ``poros.preflight``), the model's
load and the blockwise setup (dense, MoE, the Mac), ``train`` and ``evaluate``, and the
process's allocator settings. The rest of the class is mixed in from one module per job:

* ``poros.train.loop`` (``_Loop``): an update through the sequential loop or block-major, the
  optimizer and its schedule, the non-finite stop, batches and the dataloader;
* ``poros.train.forward`` (``_Forward``): the decoder inputs and each block's layers, the
  backbone's modules, where the static modules and the adapters live;
* ``poros.train.head`` (``_Head``): the final norm, the LM head and the loss
  (``DeviceHead.loss`` chooses which a step takes);
* ``poros.train.checkpointing`` (``_Checkpointing``): checkpoints, resume, the saved adapter;
* ``poros.train.progress`` (``_Progress``): step events, metrics, progress lines;
* ``poros.train.datasets``: the synthetic tokens and the dataset loaders.

The 5-line example must work verbatim:

    from poros import PorosTrainer, PorosConfig
    trainer = PorosTrainer.from_pretrained(
        "Qwen/Qwen2.5-32B", quantization="nf4",
        lora_rank=16, lora_alpha=32, block_size=4,
    )
    trainer.train("OpenAssistant/oasst1", steps=200, seq_len=512)
    trainer.save("./poros-qwen32b-lora")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from poros import metal
from poros.core.arch_registry import (
    blockwise_arch_check,
    registry_type,
    run_startup_parity_selfcheck,
    should_run_startup_selfcheck,
    text_unreached,
)
from poros.core.config import (
    PorosConfig,
    train_path,
    trains_block_major,
)
from poros.core.determinism import enforce_determinism
from poros.core.env import configure_torch_threads
from poros.core.errors import (
    PorosConfigError,
)
from poros.core.events import EventWriter, PorosEvent
from poros.core.residency import BlockManager, resident_block_ids
from poros.core.sequential_rng import KeyedRng
from poros.train.checkpointing import _Checkpointing
from poros.train.forward import _Forward
from poros.train.head import _Head
from poros.train.loop import _Loop
from poros.train.model import PorosModel
from poros.train.progress import RunHistory, _Progress
from poros.train.reporting import Reporter

logger = logging.getLogger(__name__)


def _set_allocator_settings(settings: str) -> None:
    """torch's process-wide allocator settings. torch has no public call for it (2.14 included):
    torch.cuda.memory._set_allocator_settings, deprecated since 2.10, only calls the one its
    warning names, called here directly, so the result is the same without the warning."""
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    (setter or torch.cuda.memory._set_allocator_settings)(settings)


# expandable_segments as this process last set it (apply_expandable_segments); None: never set.
_EXPANDABLE_SET: Optional[bool] = None


def _allocator_env_expandable() -> Optional[bool]:
    """None unless the environment configures torch's allocator (PYTORCH_ALLOC_CONF or
    PYTORCH_CUDA_ALLOC_CONF, its older name, not empty), whose choice then wins; else whether
    expandable segments are on in what torch reads: the first of its variables that is set."""
    if not any(os.environ.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")):
        return None
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if name in os.environ:
            said = dict(part.split(":", 1) for part in os.environ[name].replace(" ", "").split(",")
                        if ":" in part)
            return said.get("expandable_segments") == "True"
    return None  # pragma: no cover - one of the two above is set


def apply_expandable_segments(config: PorosConfig, device: torch.device) -> None:
    """``cuda_expandable_segments`` on CUDA, and afterwards the field says what this process's
    allocator runs with, so a run's records (recorded_config) say it too. Off CUDA nothing is
    done or changed.

    Expandable allocator segments for this process, unless the environment already configures
    the allocator (that choice wins, and is what is recorded). A driver that cannot map them (no
    CUDA VMM, as through some GPU-sharing layers) refuses the first allocation in one: one is
    made here, and on a refusal the default allocator stays (same values). A process that
    already holds CUDA memory keeps its allocator as it is: the check's allocation could come
    from a segment it holds and prove nothing. Once set, a later call keeps what was set."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    config.cuda_expandable_segments = _expandable_segments(config.cuda_expandable_segments,
                                                           device)


def _expandable_segments(wanted: bool, device: torch.device) -> bool:
    """What apply_expandable_segments leaves the allocator running with: True for expandable
    segments, False for torch's default."""
    global _EXPANDABLE_SET
    said = _allocator_env_expandable()
    if said is not None:
        return said
    if not wanted or _EXPANDABLE_SET is not None:
        return bool(_EXPANDABLE_SET)
    torch.cuda.empty_cache()  # cached free blocks would serve the check
    if torch.cuda.memory_reserved(device):
        logger.info("expandable_segments: this process already holds CUDA memory; its "
                    "allocator is left as it is")
        return False
    try:
        _set_allocator_settings("expandable_segments:True")
    except Exception as exc:  # pragma: no cover - platform without expandable segments
        logger.info("expandable_segments unavailable here (%s); default allocator kept", exc)
        return False
    _EXPANDABLE_SET = True
    try:
        torch.empty(1, device=device)
    except Exception as exc:  # noqa: BLE001 -- any failure: the check could not be made
        # No CUDA VMM ("CUDA driver error: operation not supported"), or no CUDA allocation at
        # all (a torch built without CUDA raises AssertionError): either way nothing proves
        # expandable segments work here, so the default allocator, which always does, is kept.
        _set_allocator_settings("expandable_segments:False")
        _EXPANDABLE_SET = False
        logger.warning(
            "expandable_segments: the first allocation in one failed (%s: %s); the default "
            "allocator is used instead, same results (PYTORCH_ALLOC_CONF="
            "expandable_segments:False skips this check)", type(exc).__name__, exc)
    torch.cuda.empty_cache()  # the allocator holds nothing from the check
    return _EXPANDABLE_SET


class PorosTrainer(_Checkpointing, _Loop, _Forward, _Head, _Progress):
    """Main training orchestrator for Poros blockwise residency.

    Implements the blockwise forward/backward loop with per-block RNG
    save/restore (INV-1), cross-stream record_stream safety (INV-2),
    determinism protocol (INV-3), and architecture validation (INV-4).
    """

    def __init__(self, config: PorosConfig) -> None:
        self.config = config
        configure_torch_threads()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        # A gate, not a preference: every entry path lands here, so on a Mac a run
        # is on mps or refused. Falling to CPU-like branches on mps is a wrong gradient.
        self._metal = self.device.type == "cpu" and sys.platform == "darwin"
        if self._metal:
            metal.require(config)
            self.device = torch.device("mps")

        self.model: Optional[nn.Module] = None
        self.block_manager: Optional[BlockManager] = None
        from poros.train.compute import RunCompute
        from poros.train.head import DeviceHead

        # what computes the run, recorded beside the settings (poros.train.compute)
        self.compute = RunCompute(self)

        # the LM head: the loss from the final decoder output in both loops (poros.train.head)
        self.head = DeviceHead(self)
        self.optimizer: Optional[optim.Optimizer] = None

        self._poros_model: Optional[PorosModel] = None
        self._peft_model: Optional[nn.Module] = None
        # Machine-local path the base was actually loaded from, when it
        # differs from the portable id save() writes (provenance).
        self._resolved_base_path: Optional[str] = None
        self._adapter_backend: Optional[str] = None
        self._adapter_summary: Optional[Dict[str, Any]] = None
        self.static_trainable_params: List[Tuple[str, nn.Parameter]] = []
        # Static adapters on submodules a text-only forward never runs (text_unreached): they
        # must get no gradient; checked once, after the first update.
        self._unreached_params: List[Tuple[str, nn.Parameter]] = []
        self.block_trainable_params: Dict[int, List[Tuple[str, nn.Parameter]]] = {}
        self.trainable_parameters: List[nn.Parameter] = []
        self.trainable_parameter_names: List[str] = []

        self.global_step = 0
        self.epoch = 0
        self._total_optimizer_steps = 0  # run length, for the progress display
        self._resume_step = 0  # first step index to run (>0 when resuming)
        # the run its checkpoints are stamped with (poros.core.checkpoints.RUN_MARKER); a resume
        # adopts its checkpoint's, so pruning and "newest" read this run's alone
        import uuid

        self._run_id = uuid.uuid4().hex[:16]
        self._is_setup = False
        self.history = RunHistory()  # what the run has done so far (poros.train.progress)
        # A front end's progress sink (poros.train.reporting). None = the
        # plain stdout progress lines below, unchanged.
        self.reporter: Optional[Reporter] = None
        self._retained_forward_blocks: List[int] = []
        self._remat_event_log: List[Tuple[str, int]] = []
        self._lr_scheduler: Optional[Any] = None
        self._signature_cache: Dict[type, Any] = {}
        self._event_writer: Optional[EventWriter] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        # Solid State Streaming (store="ssd", MoE only): the expert source,
        # told when each forward+backward ends (schedule replay).
        self._sss_source: Optional[Any] = None
        self._sss_probe: Optional[Any] = None
        self._sss_tuned = False
        # the SSD store's sequential loop: this update's micro-batch shapes, which size the
        # device expert cache once its optimizer step ran (RunCompute.cache_after_update)
        self._sss_pass_shapes: List[tuple[int, int]] = []
        # NF4's frozen embedding table and LM head that k-bit preparation left in their load
        # dtype (PorosModel.uncast_tables), until placed; and, once kept compact in host RAM,
        # the dtype each computes in (id(param) -> dtype, poros.core.tables).
        self._uncast_tables: List[nn.Parameter] = []
        self._compact_tables: Dict[int, torch.dtype] = {}
        self._witness: Optional[Any] = None
        # Block-major accumulation's walk of the sequential loop's random states: the CUDA
        # offset one dropout draw of a size moves on this device (poros.core.sequential_rng).
        self._rng_memo: Dict[Any, int] = {}
        # A MoE model's random states: every decoder layer's forward keyed by (seed, update,
        # micro-batch, layer) in either loop (_setup_moe); None for a dense one, whose
        # block-major updates walk the sequential loop's stream.
        self.keyed_rng: Optional[KeyedRng] = None
        # Which loop the updates ran: None before the first, then whether any ran block-major
        # (ran_path, the records' "ran_path").
        self._block_major_ran: Optional[bool] = None
        # Block-major given up for the run: a draw its walk cannot follow (block_major.update).
        self._block_major_off = False
        # path: efficient's host slabs for block inputs, made by the first update that needs them.
        self._boundary_offload: Optional[Any] = None
        self._watchdog: Optional[Any] = None
        # The newest held-out score: (updates done when it was scored, its numbers). The
        # checkpoint saved after that many updates keeps it in eval.json (poros.train.evals).
        self._held_score: Optional[Tuple[int, Dict[str, Any]]] = None
        # The run's first held-out score, of the model it started from; a resume reads it from
        # its checkpoint, and every checkpoint keeps it (poros.train.evals.read_start). None: not
        # known (no held-out rows, or a checkpoint from before checkpoints kept it).
        self._held_start: Optional[Dict[str, Any]] = None
        self._held_changed = False  # the start dropped: the held-out rows changed since then

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        quantization: Literal["nf4"] = "nf4",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: Optional[float] = None,
        lora_target_modules: Optional[Union[List[str], str]] = None,
        block_size: Optional[int] = None,
        config: Optional[PorosConfig] = None,
        check_host_ram: bool = True,
        check_fit: bool = True,
        store: str = "ram",
        store_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "PorosTrainer":
        """Load model + attach LoRA + set up blockwise residency.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            quantization: How the frozen base is stored: "nf4", the only value
                (any other is refused, PorosConfigError).
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha scaling.
            lora_dropout: LoRA dropout rate. None: the config's (0.05).
            lora_target_modules: Target modules for LoRA (None = auto).
            block_size: Layers resident per block (None = validated default 4,
                not guessed from VRAM; memory<->speed knob, never affects parity).
            config: Full PorosConfig (overrides individual args).
            store: Where a MoE model's quantized experts live: ``"ram"``
                (pinned host RAM, default) or ``"ssd"`` (a ``.sss`` file under
                ``store_path``, streamed from the SSD; written on the first
                run, reused after). Bit-identical training either way.
                Any other name is a store an installed package provides
                (``poros.stores``; e.g. a different expert codec).
            store_path: Directory on a local SSD for ``store="ssd"`` (a plugin
                store's own path otherwise).
            output_dir (in ``**kwargs``): where the run writes. One that cannot be created
                or written is refused here, before the download (``PorosConfigError``,
                ``PorosDiskError``); a ``config``'s is checked when ``train()`` starts.
            check_host_ram: On CUDA, refuse before the download when the plan's
                peak host RAM exceeds what is available (``PorosHostRamError``): a MoE
                model's, or a dense NF4 run whose frozen layers alone outgrow it. False
                skips the check (``poros train --force``).
            check_fit: On CUDA, refuse before the download when the plan says the
                run cannot fit the GPU on any path it could take (``PorosVramError``, with
                what would fit). False skips the check (``poros train --force``).
            **kwargs: Additional config overrides.

        Returns:
            Configured PorosTrainer ready for training.
        """
        if config is None:
            if "output_dir" in kwargs:  # named for this run: refused now, not after the download
                from poros.preflight.disk import require_output_dir

                require_output_dir(kwargs["output_dir"])
            config = PorosConfig(
                model_name_or_path=model_name_or_path,
                quantization=quantization,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                **({} if lora_dropout is None else {"lora_dropout": lora_dropout}),
                lora_target_modules=lora_target_modules,
                block_size=block_size,
                store=store,
                store_path=store_path,
                **kwargs,
            )
            from poros.core.checkpoints import fresh_run_clash

            # named for this run: refused now, not after the download. Not where the output
            # folder holds a run's checkpoints: no new run starts there (train() refuses one),
            # so this is a resume, which reads its own checkpoint, not the adapter it began from
            # (that folder may be gone by now)
            if config.init_adapter and not fresh_run_clash(config.output_dir):
                from poros.preflight.resume import require_start

                require_start(config)
        else:
            # config= is authoritative (documented: it overrides the individual
            # args), so the model load below must read from it -- otherwise a
            # supplied config's adapter/quantization settings are silently
            # replaced by the keyword defaults.
            quantization = config.quantization
            lora_rank = config.lora_rank
            lora_alpha = config.lora_alpha
            lora_dropout = config.lora_dropout
            lora_target_modules = config.lora_target_modules
            # EXCEPT the model id: the weights come from the positional
            # argument, so the config must record what was actually loaded.
            # A config built without model_name_or_path keeps the class
            # default ("Qwen/Qwen2.5-7B"), and that default must never end
            # up stamped on artifacts trained on a different model -- it
            # also drives the tokenizer fetch and adapter_config.json.
            if config.model_name_or_path != model_name_or_path:
                logger.warning(
                    "config.model_name_or_path=%r differs from the model "
                    "being loaded (%r); using the loaded one.",
                    config.model_name_or_path, model_name_or_path,
                )
                config.model_name_or_path = model_name_or_path

        trainer = cls(config)
        from poros import preflight
        from poros.core.env import quiet_hub_notice

        quiet_hub_notice()  # the load's own requests: the same notice, on every run

        # The GPU fit is a question about a length: with max_seq_length not set, the load plans
        # its path and tables as before, and the verdict (a warning, or a refusal) waits for
        # train(), which knows the length it trains at (seq_len=, or the config's) and asks then.
        # Before the download, only a run that fits at no length is refused (fit.at_no_length).
        explicit_seq = "max_seq_length" in config.model_fields_set
        trainer._fit_check = check_fit
        trainer._fit_host = check_host_ram
        trainer._fit_deferred = check_fit and not explicit_seq
        trainer._planned_seq = config.max_seq_length
        load_fit = check_fit and explicit_seq
        start = preflight.start.Startup()
        if trainer.device.type == "cuda":  # CUDA's own defaults, the card, the arch, the fit
            start = preflight.start.on_cuda(config, trainer.device, model_name_or_path,
                                            check_fit=load_fit, check_host_ram=check_host_ram,
                                            at_any_length=trainer._fit_deferred)
        elif trainer._metal:  # the Mac's branch (poros.preflight.mac)
            preflight.mac.before_download(config, check_fit=check_fit,
                                          check_host_ram=check_host_ram)
        lora_dropout = config.lora_dropout

        # Enforce determinism (INV-3) only when requested (the default). The bit-exact
        # guarantee is stated and checked with it on (math-only SDPA + deterministic
        # algorithms); opting out trades it for torch's faster nondeterministic kernels.
        if config.deterministic:
            enforce_determinism(
                # The Mac keeps torch's fill: its receipts ran with it.
                fill_uninitialized=trainer._metal
                or bool(getattr(config, "fill_uninitialized_memory", False)),
            )
        apply_expandable_segments(config, trainer.device)

        # Seed RNG for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        routed = preflight.start.route(trainer, model_name_or_path, start)
        trainer._fit_entry = "moe" if routed == "moe" else "dense"
        if routed == "moe":
            trainer._setup_moe(check_host_ram, load_fit)
            return trainer
        if routed == "metal_moe":
            trainer._setup_moe(check_host_ram)
            return trainer

        # Load model + attach adapters via PorosModel
        poros_model = PorosModel.from_pretrained(
            model_name_or_path,
            quantization=quantization,
            torch_dtype=config.torch_dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=config.lora_bias,
            adapter_backend=config.adapter_backend,
            revision=config.model_revision,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attn_implementation,
            nf4_load_strategy=getattr(config, "nf4_load_strategy", "auto"),
            loader=trainer._load_on_metal if trainer._metal else None,
            # resident blocks are quantized straight onto the GPU and never take host RAM
            resident=((config.block_size or _DEFAULT_BLOCK_SIZE, config.resident_blocks)
                      if config.resident_blocks and trainer.device.type == "cuda" else None),
            defer_table_cast=True,  # the tables are cast where they are placed (exact)
        )
        # Record the RESOLVED base revision for provenance: transformers
        # stamps the commit hash it actually loaded onto the config, which
        # covers the normal Hub-id path (the cache-snapshot path below also
        # sets it via the resolved path). Same-shaped later revisions pass
        # every key/shape check, so this pin is what eval/export re-load.
        if config.model_revision is None:
            sha = getattr(
                getattr(poros_model.model, "config", None), "_commit_hash", None
            )
            if (isinstance(sha, str) and len(sha) == 40
                    and all(c in "0123456789abcdef" for c in sha)):
                config.model_revision = sha

        trainer._poros_model = poros_model
        trainer.model = poros_model.model
        trainer._uncast_tables = list(getattr(poros_model, "uncast_tables", []))
        trainer._peft_model = poros_model.peft_model
        trainer._adapter_backend = poros_model.adapter_backend
        trainer._adapter_summary = poros_model.adapter_summary
        if trainer._metal and hasattr(trainer.model, "poros_lora_targets"):
            trainer._peft_model.peft_config["default"].target_modules = (
                trainer.model.poros_lora_targets
            )

        # Architecture check (INV-4)
        model_config = getattr(trainer.model, "config", None)
        if model_config is not None:
            # A Gemma 4 with its MoE block is gemma4_moe: never the dense engine's.
            model_type = registry_type(model_config) or "unknown"
            blockwise_arch_check(model_type, mode=config.blockwise_arch_check)

        # Resolve block_size to the validated default if not provided. We do
        # NOT guess it from free VRAM (untested per-machine heuristic that can
        # OOM); block_size is a memory<->speed knob the user overrides.
        if trainer._metal:  # one pool holds everything, so a block is one layer
            config.block_size = 1
        if config.block_size is None:
            config.block_size = _DEFAULT_BLOCK_SIZE
            logger.info(
                "Using validated default block_size=%s "
                "(override to trade speed vs memory)",
                config.block_size,
            )

        # Set up blockwise residency
        trainer._setup_blockwise_training()
        trainer.compute.install_dense()

        # Startup parity self-check. OFF by default -- the validated archs are
        # already validated 0.00e+00, so this is an internal regression tool
        # only. Enable with startup_parity_selfcheck="always" or
        # POROS_STARTUP_SELFCHECK=always (internal parity testing).
        run_selfcheck = should_run_startup_selfcheck(config.startup_parity_selfcheck)
        if run_selfcheck and trainer.block_manager is not None:
            if torch.cuda.is_available():
                def _selfcheck_forward(m: Any, input_ids: Any) -> Any:
                    return trainer._forward_selfcheck(m, input_ids)

                def _selfcheck_resident(m: Any, input_ids: Any) -> Any:
                    return trainer._forward_selfcheck_resident(m, input_ids)

                # NF4 is bitwise: the self-check's tolerance is zero.
                _atol, _rtol = 0.0, 0.0
                # Opt-in (internal parity testing): a failure is a hard signal,
                # so let run_startup_parity_selfcheck raise rather than swallow it.
                # Observationally neutral: the probe runs forwards that
                # consume RNG, so snapshot and restore CPU+CUDA state around
                # it. Without this, enabling the check silently changes the
                # weights it exists to validate (grad_norm at step 0 differs,
                # loss diverges by step 2, final adapter hash changes).
                _cpu_rng = torch.get_rng_state()
                _cuda_rng = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                )
                try:
                    selfcheck_result = run_startup_parity_selfcheck(
                        trainer.model,
                        trainer.block_manager,
                        _selfcheck_forward,
                        resident_forward_fn=_selfcheck_resident,
                        atol=_atol,
                        rtol=_rtol,
                    )
                finally:
                    torch.set_rng_state(_cpu_rng)
                    if _cuda_rng is not None:
                        torch.cuda.set_rng_state_all(_cuda_rng)
                logger.info(
                    "Startup parity self-check passed: %s", selfcheck_result
                )
                # Audible: this is the instrument behind the headline claim,
                # so say it ran rather than hiding it behind logging config.
                if trainer._progress_enabled():
                    trainer._progress_print(
                        f"startup parity self-check passed ({selfcheck_result})"
                    )
            else:
                logger.info(
                    "Startup parity self-check skipped (no CUDA device available)"
                )

        return trainer

    def _load_on_metal(self) -> nn.Module:
        from poros.metal.load import fetch, load

        # poros_config.json keeps it, so a resume or a merged export can tell the base has moved.
        self.config.model_revision = fetch(self.config)[2]
        model = load(self.config, self.device)
        if hasattr(model.poros_source, "mark_step"):  # the SSD store: recorded as CUDA's is
            self._sss_source = model.poros_source
        return model

    def _setup_blockwise_training(self) -> None:
        """Set up the block manager, optimizer, scheduler, and prefetch stream."""
        config = self.config
        assert self.model is not None

        # Identify adapter parameter IDs (they stay on GPU always)
        adapter_param_ids = {
            id(param)
            for _, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Create block manager, filtering out adapter params from offloading
        manager = BlockManager
        if self._metal:
            from poros.metal.residency import UnifiedBlockManager as manager
        self.block_manager = manager(
            self.model,
            block_size=config.block_size or 4,
            freeze_embeddings=config.freeze_embeddings,
            freeze_lm_head=config.freeze_lm_head,
            managed_parameter_filter=lambda _name, param: (
                id(param) not in adapter_param_ids
            ),
            train_managed_parameters=False,
        )
        # Blocks that stay on the device for the whole run: the same tensors,
        # never streamed again (load is then a no-op, release skips them), so
        # the math is unchanged and only H2D traffic drops. They keep no host
        # copy: each trades its weights' bytes of host RAM for as much VRAM.
        resident = (resident_block_ids(self.block_manager.num_blocks, config.resident_blocks or 0)
                    if self.device.type == "cuda" else [])
        if resident:
            self.block_manager.offload_all_to_cpu(keep=tuple(resident))
        else:
            self.block_manager.offload_all_to_cpu()

        # Move static modules (embeddings, norms, head) to device
        self._move_static_modules_to_device(self.device)

        # Move trainable adapter params to device
        self._move_trainable_parameters_to_device()

        # Partition trainable parameters by block
        self.block_trainable_params, self.static_trainable_params = (
            self._partition_trainable_parameters_by_block()
        )
        # Adapters on a tower a text-only forward never runs (a VLM's vision tower, under the
        # all-linear default) train nothing on either loop: checked after the first update.
        towers = text_unreached(getattr(getattr(self.model, "config", None), "model_type", None))
        self._unreached_params = [(n, p) for n, p in self.static_trainable_params
                                  if towers and set(n.split(".")) & set(towers)]
        outside = [n for n, p in self.static_trainable_params
                   if not any(p is q for _, q in self._unreached_params)]
        if trains_block_major(config) and outside:
            # block-major trains the blocks' adapters; sequential_reason names the usual cases
            raise PorosConfigError(
                f"path: {config.path} trains block-major, which trains the decoder blocks' "
                "adapters only, and these train outside them: " + ", ".join(outside[:5]),
                remediation="Set path: sequential.",
            )

        # Initialize optimizer
        self._initialize_optimizer()

        # Set up prefetch stream
        if config.enable_prefetch and self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

        if resident:
            self.block_manager.keep_resident(resident, self.device)

        # Set up event writer
        self._event_writer = EventWriter(config.output_dir)

        if self._metal:
            from poros.metal.witness import RematWitness

            self._witness = RematWitness(
                [layer for _, layer in self.block_manager.transformer_layers],
                self._event_writer,
                config.output_dir,
            )

        self._is_setup = True

        # Log setup summary
        logger.info(
            "Poros blockwise training initialized: "
            "%s layers -> %s blocks (block_size=%s), "
            "%s trainable params, backend=%s",
            self.block_manager.num_layers,
            self.block_manager.num_blocks,
            config.block_size,
            len(self.trainable_parameters),
            self._adapter_backend,
        )

    # What close() lets go of: by name, set to None, [] or {} (anything it does not name it leaves).
    _CLOSE_NONE = ("_boundary_offload", "block_manager", "optimizer", "_lr_scheduler", "model",
                   "_peft_model", "_poros_model", "_sss_source", "_sss_probe", "keyed_rng",
                   "_witness", "_prefetch_stream", "head", "compute", "_watchdog")
    _CLOSE_LISTS = ("trainable_parameters", "static_trainable_params", "trainable_parameter_names",
                    "_retained_forward_blocks", "_uncast_tables", "_unreached_params",
                    "_sss_pass_shapes", "_remat_event_log")
    _CLOSE_DICTS = ("block_trainable_params", "_compact_tables", "_rng_memo", "_signature_cache")

    def close(self) -> None:
        """Free what this trainer holds, now: an SSD store's reader threads, file descriptors and
        page-locked ring (its ``close``), the block store's page-locked host copies and the
        boundary slabs (their grants return to the process's pin budget, POROS_PIN_BUDGET), the
        LM head's pinned and eval copies, the model and every list of its parameters, the
        optimizer and the GPU's cache. Dropping a trainer frees the same once it is collected;
        ``close`` (or ``with PorosTrainer...``) frees it at once, so a second trainer in the
        process gets what the first had. Safe to call twice; the trainer cannot train after."""
        import gc

        model = getattr(self, "model", None)
        sources = [getattr(self, "_sss_source", None),
                   getattr(getattr(model, "_poros_moe_store", None), "source", None),
                   getattr(model, "poros_source", None)]
        closed = set()
        for source in sources:  # each once: the trainer and its model may name the same one
            if source is not None and id(source) not in closed and callable(
                    getattr(source, "close", None)):
                closed.add(id(source))
                source.close()
        head = getattr(self, "head", None)
        for name in ("_pinned", "_eval_weight", "weight_device", "_stream", "_trainer"):
            if head is not None and hasattr(head, name):
                setattr(head, name, None)
        if head is not None and isinstance(getattr(head, "_uploads", None), list):
            head._uploads.clear()
        bm = getattr(self, "block_manager", None)
        if bm is not None:
            getattr(bm, "_cpu_weight_store", {}).clear()
        compute = getattr(self, "compute", None)
        if compute is not None and hasattr(compute, "_t"):
            compute._t = None  # its back-reference to this trainer
        for name in self._CLOSE_NONE:
            if hasattr(self, name):
                setattr(self, name, None)
        for name in self._CLOSE_LISTS:
            if hasattr(self, name):
                setattr(self, name, [])
        for name in self._CLOSE_DICTS:
            if hasattr(self, name):
                setattr(self, name, {})
        self._is_setup = False
        del model, sources, head, bm, compute
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "PorosTrainer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _fit_at_length(self) -> None:
        """The fit at ``max_seq_length`` as train() runs it: asked where the load deferred it (no
        length was set) or planned another length, decided as the load decides it. A path the
        run left to Poros is chosen again at this length (the path a load at this length
        picks: the loop is chosen per run, and efficient's slabs are made by its first update);
        the tables stay where the load put them (they are placed). A warning where the plan
        puts the run past the card on an estimate; ``PorosVramError`` (with what would fit) or
        ``PorosHostRamError`` where it cannot fit, each on the load's own flag, before any data
        is read. CUDA only (a Mac's plan is its table's)."""
        if (not getattr(self, "_fit_check", False) or self._metal
                or getattr(self, "device", None) is None or self.device.type != "cuda"):
            return
        seq = self.config.max_seq_length
        if not getattr(self, "_fit_deferred", False) and seq == getattr(self, "_planned_seq", seq):
            return
        self._fit_deferred, self._planned_seq = False, seq
        from poros.core.arch_registry import registry_type
        from poros.plan.configs import read_model_config
        from poros.preflight import fit

        cfg = self.config
        model_json = read_model_config(cfg.model_name_or_path, cfg.model_revision)
        tables = (cfg.offload_embeddings_and_head, cfg.offload_reason)
        try:
            found = fit.check(cfg, fit.planner_here(
                registry_type(model_json) if model_json else None, model_json,
                entry=getattr(self, "_fit_entry", None)))
        finally:  # placed at the load: whatever the check would choose now
            cfg.offload_embeddings_and_head, cfg.offload_reason = tables
        if found.path is not None:
            logger.warning("path: %s: %s", *found.path)
        if found.warning:
            logger.warning("may not fit: %s", found.warning)
        fit.refuse(found, check_fit=True,
                   check_host_ram=getattr(self, "_fit_host", True))

    def _setup_moe(self, check_host_ram: bool = True, check_fit: bool = True) -> None:
        """Set up a ``poros.moe`` run: streamed load, then expert LoRA.

        ``prepare_moe`` makes the model stream its own experts and dense
        layers inside its forward, so there is no ``BlockManager``: the
        loop calls the model directly, and the rest of the trainer
        (optimizer, checkpoints, events) is shared with the dense path.
        """
        from poros.moe.load import load_moe_streamed
        from poros.preflight.start import moe_on_cuda, moe_settings_refusal

        config = self.config
        err = moe_settings_refusal(config, self._metal)
        if err is not None:
            raise err
        if self._metal:
            self._setup_moe_on_metal()
            return
        # How the experts are held, decided once from the checkpoint and the setting, before the
        # checks size them and before anything downloads; recorded: checkpoints, adapter, export
        config.moe_expert_format, told = _expert_format(config)
        moe_on_cuda(config, check_fit=check_fit, check_host_ram=check_host_ram,
                    at_any_length=bool(getattr(self, "_fit_deferred", False)))
        config.adapter_backend = "native"  # what save() writes is what ran
        ssd_options = None
        if config.store == "ssd":
            from poros.sss.probe import probe

            self._sss_probe = probe(config.store_path)
            from poros.core.config import ssd_cache_bytes

            ssd_options = {"ram_budget": config.sss_ram,
                           "cache_bytes": ssd_cache_bytes(config.store_options),
                           "bandwidth": self._sss_probe.bandwidth,
                           "latency_bandwidth": self._sss_probe.b_ssd_qd1,
                           # one reader thread per read the probe found the disk wants
                           "threads": max(8, self._sss_probe.queue_depth)}
            logger.info(
                "SSS store %s: disk %.2f GB/s (%s), H2D %s",
                config.store_path, self._sss_probe.b_ssd / 1e9,
                "O_DIRECT" if self._sss_probe.direct else "buffered",
                "n/a" if self._sss_probe.b_h2d is None
                else f"{self._sss_probe.b_h2d / 1e9:.2f} GB/s",
            )
        # the decision (poros.core.compute): "auto", fused where every layer qualifies, settled
        # by prepare_moe once the model's layers are known; else reference, for a reason
        compute = self.compute.moe_request()
        build = getattr(config, "_store_build_path", None)
        model = load_moe_streamed(
            config.model_name_or_path,
            getattr(torch, config.torch_dtype),
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attn_implementation,
            revision=config.model_revision,
            store=config.store,
            store_path=config.store_path,
            ssd_options=ssd_options,
            store_options=config.store_options,
            device=str(self.device),  # where "auto" quantizes: the training GPU
            # a run folder's recorded store_path is only read (poros.stores.recorded_store)
            build_path=build if config.store == "ssd" else None,
            expert_format=told,
        )
        if hasattr(model, "_poros_expert_format"):  # what the load held them as, from its files
            config.moe_expert_format = model._poros_expert_format
        if build is not None and getattr(model, "_poros_store_path", None) == build:
            config.store_path = build  # where it was built is what the run records
        source = model._poros_moe_store.source
        if hasattr(source, "mark_step"):
            self._sss_source = source
        if model.config.name_or_path != config.model_name_or_path:
            # A Hub snapshot directory: save() pins its revision from the path.
            self._resolved_base_path = model.config.name_or_path
        report = self.compute.prepare_moe(model, compute)
        for warning in report.warnings:
            logger.warning(warning)
        self.model = model
        self.compute.install_moe()
        from poros.moe.modules import ManagedMoeBlock

        residency = next((m.residency for m in model.modules()
                          if isinstance(m, ManagedMoeBlock)), None)
        if hasattr(residency, "_capacity"):  # the expert cache before any growth: one layer
            self.compute.cache_prepared(residency._capacity)
        self.keyed_rng = KeyedRng(config.seed, self.device)
        self.keyed_rng.install(KeyedRng.layers_of(model))
        self._adapter_backend = "native"
        self._initialize_optimizer()
        self._event_writer = EventWriter(config.output_dir)
        self._is_setup = True
        logger.info(
            "Poros MoE training initialized: %d experts quantized, guarantee=%s",
            report.experts_quantized,
            report.guarantee,
        )

    def _setup_moe_on_metal(self) -> None:
        """A MoE model on a Mac is a dense one whose layers hold more linears, so the block loop
        runs it. Its experts and its adapter are ``poros.metal.experts``."""
        from poros.metal import experts

        config = self.config
        config.block_size = 1
        self.model = self._load_on_metal()
        experts.attach_adapters(self.model, rank=config.lora_rank, alpha=config.lora_alpha,
                                dropout=config.lora_dropout, seed=config.seed)
        self._adapter_backend = "experts"
        self._setup_blockwise_training()

    def train(
        self,
        dataset: Union[str, Dataset, DataLoader],
        *,
        steps: Optional[int] = None,
        seq_len: Optional[int] = None,
        eval_dataset: Optional[Union[str, Dataset, DataLoader]] = None,
        resume_from: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run the training loop.

        Args:
            dataset: HF dataset name, Dataset object, or DataLoader.
            steps: Override max_steps.
            seq_len: Override max_seq_length.
            eval_dataset: Held-out rows, every one scored (their mean loss, as an eval event
                with the rows it scored): before the first step (not again on a resume),
                every ``eval_steps``, at each checkpoint where ``eval_checkpoints`` is on
                (into its eval.json, before the save), and after the last step. The metrics
                then carry ``eval_loss_start`` and ``eval_loss``. An eval pass leaves the
                random generators as they were, so the run trains the same bytes as without
                it. None: nothing is scored; this loop holds no rows out of ``dataset``.
            resume_from: Path to a checkpoint directory written by ``save()``;
                restores adapter weights, optimizer state, RNG state, and the
                step counter, then continues from the next step.
            should_stop: Checked once per optimizer step, after its save and
                eval. When it returns True the trainer writes a checkpoint and
                raises ``PorosCancelledError``.

        Returns:
            Training metrics dict.
        """
        if not self._is_setup:
            raise RuntimeError(
                "Trainer not initialized. Use PorosTrainer.from_pretrained()."
            )

        config = self.config
        if steps is not None:
            config.max_steps = steps
        if seq_len is not None:
            config.max_seq_length = seq_len
        # What this run trains on is ``dataset``, whatever the config named: a name goes into the
        # config its checkpoints save, and what it is (a file's contents, as it is read now) into
        # their record, which a resume compares before anything is restored.
        from poros.preflight.resume import trained_on

        if isinstance(dataset, str):
            config.dataset = dataset
        self._dataset_record = trained_on(dataset)
        self._fit_at_length()  # the GPU fit at the length this run trains at
        from poros.preflight.disk import require_output_dir

        require_output_dir(config.output_dir)  # before any data is read: events land there

        # Rebuild the LR scheduler on the EFFECTIVE step count now that the
        # steps override has been applied (the scheduler built at
        # from_pretrained time used the placeholder max_steps=1000; a short run
        # would otherwise train inside warmup at a near-zero LR). Done BEFORE
        # load_checkpoint so a resumed run's restored scheduler state is the
        # last word -- the rebuild swaps in the correct lambda, then the
        # checkpoint restores the step counter on top, leaving resume unchanged.
        # Every run is built on its own length, with warmup_for's cap (warmup_steps, at most
        # about a tenth of the steps), however the length was given: train(steps=), the
        # config's max_steps, or its epochs. An epoch-based run (no max_steps): its length is
        # the data's, ceil(batches x num_epochs / gradient_accumulation_steps) optimizer steps,
        # known once the data is (a run with max_steps set prepares its data where it did).
        dataloader = None
        epoch_steps = None
        if config.max_steps <= 0:
            dataloader = self._prepare_dataloader(dataset, config.max_seq_length)
            try:
                epoch_steps = self._epoch_steps(len(dataloader))
            except TypeError:  # an iterable with no length: an epoch of it cannot be counted
                raise PorosConfigError(
                    "The dataset is an iterable with no length, and max_steps is unset: a run "
                    "of epochs needs to count its rows.",
                    remediation="Set max_steps (train(..., steps=N) or max_steps: N).") from None
        effective_steps = config.max_steps if config.max_steps > 0 else (epoch_steps or 1000)
        self._rebuild_lr_scheduler_for_run(effective_steps)

        self._resume_step = 0
        self._held_start = None  # a new run's is scored before its first step
        self._held_changed = False
        if resume_from is None:  # a new run: never into another run's checkpoints
            from poros.core.checkpoints import fresh_run_clash

            if clash := fresh_run_clash(config.output_dir):
                raise PorosConfigError(
                    f"Can't start a new run: {clash}.",
                    remediation="Continue that run with train(..., resume_from=<its checkpoint>) "
                    "(poros train <config> --resume-from latest), or set another output_dir.")
            self.start_from_adapter()  # config.init_adapter: an earlier adapter, a fresh optimizer
        if resume_from is not None:
            from pathlib import Path as _Path

            from poros.core.checkpoints import recover_checkpoint_dirs

            # a save a kill left at checkpoint-N.tmp / .old is adopted or dropped before the load
            recover_checkpoint_dirs(_Path(resume_from).absolute().parent)
            self.load_checkpoint(resume_from)
            from poros.train.evals import read_start

            self._held_start = read_start(resume_from)  # the start it had, not a score on the way
            # A checkpoint saved during step K is written AFTER K's optimizer
            # step, so resume must start at K+1 -- otherwise step K is re-run
            # (duplicate optimizer/scheduler update, wrong data + RNG position).
            self._resume_step = self.global_step + 1
            self._lr_from_schedule()  # this run's schedule, where a larger --steps lengthened it

        # Prepare dataloader
        if dataloader is None:
            dataloader = self._prepare_dataloader(dataset, config.max_seq_length)
        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = self._prepare_dataloader(
                eval_dataset, config.max_seq_length, for_eval=True
            )

        # Emit setup event
        if self._event_writer is not None:
            self._event_writer.emit(
                PorosEvent(
                    event_type="setup",
                    data={
                        "model": config.model_name_or_path,
                        "quantization": config.quantization,
                        "block_size": config.block_size,
                        "steps": config.max_steps,
                        "path": train_path(config),  # sequential, efficient or fast
                        **({"path_reason": config.path_reason} if config.path_reason else {}),
                        "offload_embeddings_and_head": bool(config.offload_embeddings_and_head),
                        **({"offload_reason": config.offload_reason}
                           if config.offload_reason else {}),
                        **self.compute.record(),
                        "torch_num_threads": torch.get_num_threads(),
                    },
                )
            )

        return self._train_blockwise_adapters(
            dataloader, eval_dataloader=eval_dataloader, should_stop=should_stop
        )

    def evaluate(
        self,
        eval_dataset: Union[str, Dataset, DataLoader],
    ) -> Dict[str, float]:
        """Evaluate and return metrics.

        Args:
            eval_dataset: Evaluation dataset.

        Returns:
            Dict with eval_loss and eval_perplexity.
        """
        dataloader = self._prepare_dataloader(
            eval_dataset, self.config.max_seq_length, for_eval=True
        )
        return self._evaluate_batches(dataloader)

    @property
    def training_metrics(self) -> Dict[str, Any]:
        """Return the full metrics payload from the last train() call."""
        return self._compute_metrics()


_DEFAULT_BLOCK_SIZE = 4


def _expert_format(config: Any) -> Tuple[Optional[str], Optional[str]]:
    """(sized, told): how a MoE run holds its experts where a model has a choice (Inkling 2
    Mini: released or nf4, ``poros.moe.inkling2.formats``), judged before any weight
    downloads, for the checks that size the run; and what the load is told. Where the
    checkpoint's tensor names were read the judgement is certain and the load is told it (a
    refusal comes here, before the download); where only its quantization files or file list
    were, or nothing, the load is told only what was asked and decides from its files, and the
    run records what it held. (None, None) for every other model, which is refused a
    ``moe_expert_format`` it would not use."""
    from poros.core.arch_registry import registry_type
    from poros.core.errors import PorosConfigError
    from poros.plan.configs import read_model_config

    model_json = read_model_config(config.model_name_or_path, config.model_revision) or {}
    model_type = registry_type(model_json)
    if model_type in ("inkling2_loop", "inkling2_loop_text"):
        from poros.moe.inkling2 import formats

        asked = formats.asked_format(config)
        state, certain = formats.state_before_download(
            config.model_name_or_path, config.model_revision, model_json)
        if certain:
            held = formats.decide(state, asked)
            return held, held
        if state is None:
            if asked is None:
                logger.info(formats.NOT_YET)
            return asked or formats.RELEASED, asked
        try:
            return formats.decide(state, asked), asked
        except PorosConfigError:  # a marker may be left over: the load decides from its files
            return asked, asked
    if config.moe_expert_format is not None:
        raise PorosConfigError(
            f"moe_expert_format: {config.moe_expert_format} applies to Inkling 2 Mini only; "
            f"{config.model_name_or_path} holds its experts in NF4.",
            remediation="Leave moe_expert_format unset.")
    return None, None
