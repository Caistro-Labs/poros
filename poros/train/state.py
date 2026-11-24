"""The state ``PorosTrainer``'s mixins share, declared once (``_TrainerState``).

``PorosTrainer`` is its startup module (``poros.train.trainer``) and one mixin per job
(``_Loop``, ``_Forward``, ``_Head``, ``_Checkpointing``, ``_Progress``); every one of them
subclasses ``_TrainerState``, so what a mixin reads or writes on ``self`` is named here, with the
module that owns it. ``PorosTrainer.__init__`` sets each instance attribute; the few that a run
fills in later have a class default here, so no module creates an attribute on the trainer on
the fly. Annotations only (and those defaults): nothing here runs at train time.

What the engine (``poros.core.blockwise``) calls on a trainer is ``poros.core.blockwise.BlockHost``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from poros.core.config import PorosConfig
    from poros.core.events import EventWriter
    from poros.core.residency import BlockManager
    from poros.core.sequential_rng import KeyedRng
    from poros.train.compute import RunCompute
    from poros.train.head import DeviceHead
    from poros.train.model import PorosModel
    from poros.train.progress import RunHistory
    from poros.train.reporting import Reporter


class _TrainerState:
    """What a ``PorosTrainer`` holds, by the module that writes it (``trainer``: its
    ``__init__`` and ``from_pretrained``)."""

    # -- the run's settings and device (trainer) --
    config: "PorosConfig"
    device: "torch.device"
    _metal: bool  # a Mac's run: the device is mps
    reporter: Optional["Reporter"]  # a front end's progress sink; None: stdout lines

    # -- the model, its adapters and where they live (trainer; poros.train.model) --
    model: Optional["nn.Module"]
    block_manager: Optional["BlockManager"]
    _poros_model: Optional["PorosModel"]
    _peft_model: Optional["nn.Module"]
    _resolved_base_path: Optional[str]  # where the base was loaded from (checkpointing reads)
    _adapter_backend: Optional[str]
    _adapter_summary: Optional[Dict[str, Any]]
    static_trainable_params: List[Tuple[str, "nn.Parameter"]]
    _unreached_params: List[Tuple[str, "nn.Parameter"]]  # loop checks them after update one
    block_trainable_params: Dict[int, List[Tuple[str, "nn.Parameter"]]]
    trainable_parameters: List["nn.Parameter"]  # loop rebuilds them for the optimizer
    trainable_parameter_names: List[str]
    _uncast_tables: List["nn.Parameter"]  # forward places them (poros.core.tables)
    _compact_tables: Dict[int, "torch.dtype"]  # forward fills it
    _tokenizer: Any = None  # a cached tokenizer a recipe or a caller sets; checkpointing saves it
    # what a recipe trains on (poros.data.Data: fingerprint, rows, kind), written beside each
    # checkpoint (poros.preflight.resume.DATA_RECORD) so a resume can check it reads the same
    _data_record: Optional[Dict[str, Any]] = None
    # the held-out score the next save keeps in its eval.json, with the step whose weights it
    # scored (poros.train.evals.write_score): the loop and Run.evaluate set it, checkpointing
    # writes it; and the run's first score, which a resume reads back so its summary's "from"
    # is the run's own start
    _held_score: Optional[Tuple[int, Dict[str, Any]]]
    _held_start: Optional[Dict[str, Any]]
    _held_changed: bool  # a resumed run's start dropped: its held-out rows changed
    # what the run trains on, taken when it reads it (poros.preflight.resume.trained_on: train,
    # Run.steps); None where a loop of the caller's own hands it its batches. Each checkpoint
    # records it (INPUTS_RECORD) and a resume compares it
    _dataset_record: Optional[Dict[str, Any]] = None
    # the version and the base's folder by content of INPUTS_RECORD; checkpointing samples it once
    _inputs_record: Optional[Dict[str, Any]] = None

    # -- what computes the run (trainer) --
    compute: "RunCompute"
    head: "DeviceHead"
    keyed_rng: Optional["KeyedRng"]  # a MoE model's random states (_setup_moe)
    _rng_memo: Dict[Any, int]  # block-major's walk of the sequential loop's dropout offsets
    _signature_cache: Dict[type, Any]
    _prefetch_stream: Optional["torch.cuda.Stream"]  # loop makes it; block_major reads it
    _witness: Optional[Any]  # the non-finite witness

    # -- the optimizer and the run's progress (trainer; loop writes) --
    optimizer: Optional["optim.Optimizer"]
    _lr_scheduler: Optional[Any]
    global_step: int  # loop advances it; checkpointing restores it
    epoch: int
    _total_optimizer_steps: int  # the run's length, for the progress display
    _resume_step: int  # first step index to run (>0 when resuming)
    _run_id: str  # stamped on its checkpoints (checkpoints.RUN_MARKER); a resume adopts one
    # the unstamped checkpoint of its output_dir a resume started from (checkpoints.PARENT_MARKER)
    _run_parent: Optional[str] = None
    _is_setup: bool
    history: "RunHistory"  # losses, grad norms, step times, ... (poros.train.progress owns it)
    _event_writer: Optional["EventWriter"]
    _watchdog: Optional[Any]  # loop's hang watchdog

    # -- the sequential loop and block-major (loop, forward, block_major) --
    _retained_forward_blocks: List[int]  # forward keeps them; loop clears them
    _remat_event_log: List[Tuple[str, int]]
    _block_major_ran: Optional[bool]  # None before the first update (ran_path)
    _block_major_off: bool  # block-major given up for the run (block_major.update)
    _boundary_offload: Optional[Any]  # path: efficient's host slabs (block_major makes them)
    _moe_batch: Optional[Any] = None  # block_major's per-update MoE batch state
    _moe_dq_logged: bool = False  # block_major said once that MoE experts dequantize per pass
    _boundary_offload_logged: bool = False  # block_major said once where block inputs go
    _boundary_offload_stats: Optional[dict] = None  # and where they went, for the receipt
    _auto_cache_mb: Optional[int] = None  # loop: the NF4 dequant cache size it picked
    _nf4_dequant_cache: Optional[Any] = None  # loop makes it on first use

    # -- Solid State Streaming, MoE only (trainer; loop writes) --
    _sss_source: Optional[Any]
    _sss_probe: Optional[Any]
    _sss_tuned: bool
    _sss_pass_shapes: List[tuple]
