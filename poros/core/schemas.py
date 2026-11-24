"""The plan's data models: a figure and its source, a GPU, a store, a bound, a nearest receipt
and the plan itself (``poros.plan.v1``), and ``Output``, the base of every ``--json`` result.

The library builds these (``poros.core.card`` a GPU, ``poros.plan`` a plan); the CLI prints them,
and ``poros.cli.schemas`` publishes their JSON Schema with every other result's. Versioning:
adding a field keeps ``v1``; removing, renaming or retyping one is ``v2``. Torch-free.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema


class Output(BaseModel):
    """Base of every result. ``schema`` is aliased: the name shadows BaseModel."""

    # Serialization-mode schemas mark defaulted fields required: they are
    # always present in the output, so `schema` is a required constant.
    model_config = ConfigDict(
        populate_by_name=True, json_schema_serialization_defaults_required=True
    )


class Figure(BaseModel):
    """One planned quantity and where it comes from. Never an unlabeled guess."""

    value: Optional[float]
    unit: str
    source: Literal["measured", "estimated", "hub_metadata", "local", "unknown"]
    # measured/estimated: the receipt#pointer it was read from. The planner's own (cells.json, and
    # the tests that hold a figure to its receipt): no command shows it, and no --json result or
    # schema carries it (poros.cli.contract.emit leaves it out).
    receipt: SkipJsonSchema[str] = ""
    method: str = ""  # estimated: how it was derived
    why: str = ""  # unknown: why there is no number

    @model_validator(mode="after")
    def _labeled(self) -> "Figure":
        if self.source == "measured" and not self.receipt:
            raise ValueError("a measured figure needs its receipt")
        if self.source == "estimated" and not (self.method and self.receipt):
            raise ValueError("an estimated figure needs its method and receipt")
        if self.source == "unknown" and (self.value is not None or not self.why):
            raise ValueError("an unknown figure has no value and says why")
        return self


class PlanGpu(BaseModel):
    name: str
    total_gib: float
    free_gib: Optional[float]
    source: Literal["nvidia-smi", "torch", "spec"]
    unified: bool = False  # one pool of memory: a Mac
    working_set_gib: Optional[float] = None  # how much of it the GPU may use
    uuid: Optional[str] = Field(None, exclude=True)  # nvidia-smi's; which card, not for a report
    # the device's own compute capability (the driver's or torch's), for a card that is here;
    # None for a named card (--gpu), which its name maps (poros.core.card.capability_of_name)
    capability: Optional[Tuple[int, int]] = Field(None, exclude=True)


class PlanStore(BaseModel):
    """One expert-store backend's row in a MoE plan (``--store``)."""

    store: Literal["ram", "ssd"]
    peak_host_ram: Figure
    s_per_step: Figure
    bound_by: Optional[str] = None  # "compute" or "disk", when known
    # ssd: the probed disk (bytes/s, O_DIRECT or buffered) and H2D bandwidths
    b_ssd: Optional[float] = None
    b_h2d: Optional[float] = None
    direct: Optional[bool] = None


class PlanBound(BaseModel):
    """A verdict bounded by a receipt for the same model at a config that
    dominates (fits) or is dominated by (won't fit) the one asked for.

    Peak memory never decreases as seq_len, lora_rank, batch, block_size or
    dropout grow, so a completed run at least as large on a card no bigger
    than this one's free memory proves a fit, and a run at most as large
    whose allocated peak already exceeds free memory proves a misfit.
    """

    fits: bool
    receipt: str  # the cell id (poros/plan/cells.json)
    gpu: str  # the GPU that receipt ran on
    peak_gib: float  # its allocated peak: an upper bound (fits) or lower bound (won't fit)
    differs: dict[str, list[Any]]  # knob -> [asked, receipt], only the knobs that differ
    basis: str


class PlanNearest(BaseModel):
    """No receipt bounds the asked config: the closest one that ran this model,
    and how it differs. Context for a person, never a verdict."""

    receipt: str
    gpu: str
    peak_gib: Optional[float]  # its allocated peak, when it measured one
    differs: dict[str, list[Any]]  # knob -> [asked, receipt]


class PlanOutput(Output):
    schema_: Literal["poros.plan.v1"] = Field("poros.plan.v1", alias="schema")
    model: str
    model_type: Optional[str]
    support: Literal["validated", "user_asserted", "unsupported", "unknown"]
    entry: Optional[Literal["dense", "moe"]]
    gpu: Optional[PlanGpu]
    # The loop a CUDA run of the config trains (poros.core.config.train_path): sequential, or
    # block-major accumulation's efficient or fast path. None where no CUDA plan is made.
    path: Optional[Literal["sequential", "efficient", "fast"]] = None
    # Why a path left unset was decided as it was (PorosConfig.path_reason): the sequential loop
    # where block-major cannot train the same update, or the one that fits host RAM. None where
    # path was set, or the run takes block-major's first choice. Carried for the CLI's words, not
    # in poros.plan.v1's JSON, which stays as it was.
    path_reason: Optional[str] = Field(None, exclude=True)
    # Why offload_embeddings_and_head, left unset, is on for the run (PorosConfig.offload_reason:
    # the tables on the GPU would not fit, in host RAM they do); None where it stays as set or
    # off. In memory as path_reason is; the JSON's warnings say it.
    offload_reason: Optional[str] = Field(None, exclude=True)
    # True: the planned peak fits (measured, bounded by a receipt, or an estimate, an upper
    # bound, within free VRAM); False: a measured allocated peak, or a run no larger's (a floor),
    # exceeds free VRAM; None: no evidence either way, or only an estimate exceeds it, which
    # warns (never refuses).
    fits: Optional[bool]
    fit_basis: str
    # The same verdict for host RAM: peak_host_ram against what this machine
    # (or container) has available; null when either is unknown or --gpu
    # plans for another machine.
    host_ram_free_gib: Optional[float]
    host_fits: Optional[bool]
    host_fit_basis: str
    # where no figure sizes the host RAM but a dense NF4 run's frozen layers alone are past it:
    # that floor (GiB), which the refusal names (poros.preflight.fit.host_ram_error); not in the
    # JSON
    host_floor_gib: Optional[float] = Field(None, exclude=True)
    # The length the figures are for (tokens a row, max_seq_length), on CUDA; and where it is
    # from: the config, its data's longest row where the config sets none, or the default. For
    # the screen, as path_reason is: poros.plan.v1's JSON stays as it was, and
    # its warnings say a length the config did not set.
    seq_len: Optional[int] = Field(None, exclude=True)
    seq_len_from: Optional[str] = Field(None, exclude=True)
    peak_vram_allocated: Figure
    peak_vram_reserved: Figure
    peak_host_ram: Figure
    s_per_step: Figure
    load_time: Figure
    total_time: Figure
    download: Figure
    excludes: list[str]
    warnings: list[str]
    # Expert store (MoE): the configured backend, and both backends' rows
    # when `--store ssd` probed a disk.
    store: str = "ram"  # "ram", "ssd" or a plugin store's name (poros.stores)
    stores: list[PlanStore] = Field(default_factory=list)
    # Set when no receipt measured this exact config but one bounds it
    # (`fits` then carries the bounded verdict, `fit_basis` says so).
    bound: Optional[PlanBound] = None
    nearest: Optional[PlanNearest] = None  # set when the verdict is unknown
    # On a Mac peak_vram_allocated is the one pool's peak, this how much of it the GPU saw.
    gpu_visible: Optional[Figure] = None
