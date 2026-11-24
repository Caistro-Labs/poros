"""PorosConfig: the central configuration model for Poros.

All ML-dependent fields use Optional with None default so the config can be
instantiated without torch installed.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import unquote

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from poros.core.checkpoints import regular_only

_BYTE_UNITS = {"": 1, "B": 1, "K": 1 << 10, "KB": 1 << 10, "KIB": 1 << 10, "M": 1 << 20,
               "MB": 1 << 20, "MIB": 1 << 20, "G": 1 << 30, "GB": 1 << 30, "GIB": 1 << 30,
               "T": 1 << 40, "TB": 1 << 40, "TIB": 1 << 40}


def parse_bytes(v: str) -> int:
    """``"4G"``, ``"512M"``, ``"1.5GiB"`` or ``"4096"`` as a byte count (binary units)."""
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)\s*", v)
    if m is None or m.group(2).upper() not in _BYTE_UNITS:
        raise ValueError(f"not a byte size: {v!r} (use e.g. 4G, 512M)")
    return int(float(m.group(1)) * _BYTE_UNITS[m.group(2).upper()])


# What the built-in SSD store reads from ``store_options``: ``cache``, the pinned host RAM beyond
# its read ring that keeps expert slabs once read (poros.sss.ssd.SsdSource, cache_bytes).
SSD_STORE_OPTIONS = ("cache",)


def ssd_cache_bytes(store_options: Any) -> int:
    """The SSD store's host cache in bytes from ``store_options`` (``{"cache": "4G"}``, or a byte
    count); 0 where it is not set. Raises ValueError on an option the store does not read."""
    options = dict(store_options or {})
    unknown = sorted(set(options) - set(SSD_STORE_OPTIONS))
    if unknown:
        raise ValueError(f"store: ssd reads store_options {list(SSD_STORE_OPTIONS)}, "
                         f"not {unknown}")
    cache = options.get("cache") or 0
    n = parse_bytes(cache) if isinstance(cache, str) else int(cache)
    if n < 0:
        raise ValueError(f"store_options cache must be at least 0 bytes, got {cache!r}")
    return n


_STORE_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


def parse_store(spec: str) -> tuple[str, Optional[str]]:
    """``--store`` values -> (store, store_path): ``ram``, ``ssd`` or ``ssd:/path``, or a
    plugin store (``poros.stores``) as ``name`` or ``name:/path``."""
    kind, sep, path = spec.partition(":")
    if not _STORE_NAME.match(kind) or (kind == "ram" and sep):
        raise ValueError(
            f"--store takes 'ram', 'ssd', 'ssd:/path' or an installed plugin store "
            f"('name' or 'name:/path'), got {spec!r}"
        )
    return kind, (path or None)


def no_pull_request(revision: Optional[str]) -> Optional[str]:
    """``revision``, if it is not a Hub pull request: anyone can open one on any repo, and its
    weights and code would load under the trusted repo's name."""
    if revision is not None and "refs/pr" in unquote(str(revision)).lower():
        raise ValueError("a model_revision cannot name a Hub pull request (refs/pr/...)")
    return revision


def planted_base(name: str) -> bool:
    """A Hub-shaped name that also exists as a path here. Every loader takes the folder over the
    Hub, so whoever writes a project would decide the weights, and the files beside them, under a
    trusted name. ``./org/name`` says the folder, and a run then records it as one."""
    return (bool(re.fullmatch(r"[\w.\-]+/[\w.\-]+", name)) and not name.startswith(("./", "../"))
            and os.path.lexists(name))


def train_path(cfg: Any) -> str:
    """The loop a config trains with: ``path`` once set or decided (apply_cuda_defaults), else
    ``sequential`` (the Mac and the CPU leave it unset)."""
    return cfg.path or "sequential"


def trains_block_major(cfg: Any) -> bool:
    """Whether ``cfg`` trains block-major (``efficient`` or ``fast``)."""
    return train_path(cfg) != "sequential"


# Model config keys that turn on a model's own random draws in training: its dropouts and MoE
# router noise. Block-major accumulation follows the LoRA dropouts' draws only.
_MODEL_DRAWS = re.compile(r"(dropout|dropout_prob|pdrop|jitter_noise)$")


def sequential_reason(cfg: Any, model: Optional[dict] = None) -> Optional[str]:
    """Why block-major accumulation (``path: efficient`` or ``fast``) cannot train ``cfg`` as
    the sequential loop does, in words, or None. ``model`` is the model's config.json (None: not
    known here).

    Block-major trains the decoder blocks' adapters and draws the LoRA dropouts' masks the
    sequential loop draws (poros.core.sequential_rng: a dense model's in that loop's order, a
    MoE model's keyed by layer in both). It cannot follow: a trained embedding table or LM head
    (outside the blocks), dropout masks the recompute does not restore (preserve_rng_state:
    false), the model's own random draws (its config's dropout or router noise), or a MoE
    model's router aux loss."""
    targets = cfg.lora_target_modules
    names = [targets] if isinstance(targets, str) else list(targets or ())
    if any(re.search(r"\b(lm_head|embed_tokens)\b", n) for n in names):
        return "an embedding table or LM head that trains"
    if cfg.lora_dropout > 0 and not cfg.preserve_rng_state:
        return "lora_dropout with preserve_rng_state: false"
    if not model:
        return None
    text = model.get("text_config") if isinstance(model.get("text_config"), dict) else model
    own = sorted(k for k, v in text.items() if _MODEL_DRAWS.search(k)
                 and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0)
    if own:
        return f"the model's own random draws in training ({', '.join(own)})"
    from poros.core.arch_registry import registry_type, validated_arch_types

    if registry_type(model) in validated_arch_types("moe") and text.get("output_router_logits"):
        return "a MoE model's router aux loss (output_router_logits)"
    if registry_type(model) in ("inkling2_loop", "inkling2_loop_text"):
        return ("a decoder that reruns its middle layers once per loop pass (Inkling 2 Mini), "
                "whose layer visits block-major order does not walk")
    return None


def moe_grouping(cfg: Any) -> str:
    """The MoE expert grouping a config runs: ``moe_expert_grouping``, ``"pow2"`` (the reference,
    on every loop) unless set to ``"single"``."""
    return cfg.moe_expert_grouping or "pow2"


def path_refusal(path: str, why: str) -> str:
    """The words for a ``path`` you set that block-major cannot train as the sequential loop."""
    return (f"path: {path} trains block-major, which cannot give the sequential loop's update "
            f"with {why}. Set path: sequential, or leave path unset.")


def recorded_config(cfg: Any, **dump: Any) -> dict:
    """What a run's records (``poros_config.json``, ``config.resolved.json``) hold of its
    config: every field, the machine-written ``path_reason`` and ``ran_path`` among them.
    ``dump`` goes to ``model_dump``. A reader builds a config from the fields
    (``config_fields``)."""
    return cfg.model_dump(**dump)


def config_fields(said: dict) -> dict:
    """A record's config fields (``recorded_config``), less what a run added beside them."""
    return {k: v for k, v in said.items() if k in PorosConfig.model_fields}


# Fields every run's records hold at one value, which a config may repeat but never change: not
# settings, and not in the schema (``poros schema config``). Any other value is refused by name.
FIXED_FIELDS: Dict[str, Any] = {
    "adapter_runtime": "auto",
    "bf16_forward_math_mode": "auto",
    "packed_block_transfers": True,
    "bf16_cuda_graph_blocks": False,
}


def _refuse_other_precisions(value: Any) -> None:
    """Poros trains an NF4 base: any other ``quantization`` is refused, naming what it was."""
    if value != "nf4":
        from poros.core.errors import PorosConfigError

        raise PorosConfigError(
            f"quantization: {value!r} is not one Poros trains: the frozen base is NF4.",
            remediation="Set quantization: nf4, or leave it out (nf4 is the default).")


class PorosConfig(BaseModel):
    """Central configuration for Poros blockwise residency training."""

    # Unknown keys are rejected, not silently dropped: a typo'd field name
    # (per_device_train_batch_size, lora_r, ...) would otherwise train with
    # the default while the user believes their setting applied.
    model_config = ConfigDict(extra="forbid")

    # -- Model --
    model_name_or_path: str = Field("Qwen/Qwen2.5-7B", description=(
        "The base model: a Hugging Face Hub id (e.g. Qwen/Qwen2.5-7B) or a local folder "
        "holding its config.json and weights."))
    quantization: Literal["nf4"] = Field("nf4", description=(
        "How the frozen base is stored: nf4, 4-bit NormalFloat, the only value (the default). "
        "Streamed, it trains bit for bit as it does resident."))
    # Validated at parse (torch-free): a typo like "bfloat61" must not
    # silently train in the fallback dtype.
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field("bfloat16", description=(
        "The dtype the model computes in: bfloat16 (default, as measured), float16 or "
        "float32."))

    @field_validator("quantization", mode="before")
    @classmethod
    def _check_quantization(cls, v: Any) -> Any:
        _refuse_other_precisions(v)
        return v

    # Bounds below fail at config parse, not later as a modulo crash, an
    # empty training loop, or a silently degenerate optimizer. Sentinels
    # that legitimately disable a feature (save_steps=0, eval_steps=0,
    # max_grad_norm=0, max_steps=-1, save_total_limit<=0) stay allowed.

    # -- Blockwise Residency --
    # None = validated default 4 (not guessed from VRAM)
    block_size: Optional[int] = Field(None, ge=1, description=(
        "Frozen decoder layers streamed to the GPU together. null (default): 4, the validated "
        "value, the same on every machine (never guessed from free VRAM). Larger is faster "
        "and takes more VRAM; smaller takes less VRAM and is slower. It never changes a "
        "value."))
    freeze_embeddings: bool = Field(True, description=(
        "The input embedding table stays frozen. Poros does not train the table itself: false "
        "is refused (PorosConfigError). To adapt it, name embed_tokens in lora_target_modules."))
    freeze_lm_head: bool = Field(True, description=(
        "The LM head stays frozen. Poros does not train the head itself: false is refused "
        "(PorosConfigError). To adapt it, name lm_head in lora_target_modules."))
    preserve_rng_state: bool = Field(True, description=(
        "Save each block's RNG state in the forward and restore it for the backward's "
        "recompute, so its dropout masks are drawn again identically. Keep it true (default) "
        "whenever lora_dropout > 0: false trains such a run sequentially."))  # INV-1
    blockwise_arch_check: Literal["error", "warn", "off"] = Field("error", description=(
        "A model whose architecture is neither validated nor user-asserted: error (default) "
        "refuses the run, warn logs a warning and trains (you own correctness), "
        "off skips the check."))  # INV-4
    # Startup resident-vs-blockwise parity self-check. OFF by default: parity is
    # already validated 0.00e+00 for the validated archs, so re-deriving it on
    # every run only adds startup latency (and a resident reference forward can
    # OOM on the large models Poros streams). Internal parity testing opts in
    # with "always" (or POROS_STARTUP_SELFCHECK=always).
    startup_parity_selfcheck: Literal["off", "always"] = Field("off", description=(
        "always: before training, one forward pass without gradients on 8 random tokens, once "
        "with every block loaded on the GPU and once streamed block by block; the two losses must "
        "be equal bit for bit, or the run is refused. Where the GPU cannot hold every block, it "
        "runs the streamed forward twice instead, a repeat check with no resident reference. "
        "It checks no gradient, weight or training step (POROS_STARTUP_SELFCHECK=always also "
        "turns it on). off (default): parity is validated for the validated architectures, and "
        "a resident reference can run out of memory on the models Poros streams."))
    # Strict determinism (math-only SDPA + deterministic algorithms, INV-3). The bit-exact
    # guarantee (streamed == resident, every path the same update, resume == uninterrupted) is
    # stated and checked with it on; with it off nothing is guaranteed bit for bit. Off lets
    # torch pick faster nondeterministic kernels (flash attention in its own SDPA).
    deterministic: bool = Field(True, description=(
        "Strict determinism (math-only SDPA, deterministic algorithms). true (default): at "
        "NF4 on a supported family the bit-exact guarantee holds: the streamed run equals the "
        "fully resident one, every training path trains the same update, and two runs give "
        "the same bits. false allows faster nondeterministic kernels (flash attention): "
        "faster, but not bit-exact, neither against the resident run nor from one run to the "
        "next."))
    # Under ``deterministic``: let torch NaN-fill every new buffer
    # (torch.utils.deterministic.fill_uninitialized_memory). Costs ~36k fill
    # launches per 72B NF4 update; no correct kernel reads the fill, so it is off
    # by default (every bitwise gate passed with it off).
    fill_uninitialized_memory: bool = Field(False, description=(
        "Under deterministic: NaN-fill every new buffer "
        "(torch.utils.deterministic.fill_uninitialized_memory). Off by default: no correct "
        "kernel reads the fill, and on a 72B NF4 update it costs about 36k fill launches."))

    # -- LoRA --
    # Chat/text recipes: sort each optimizer step's rows by length before splitting them into
    # micro-batches of per_device_batch_size, so each pads to rows of about its own length
    # (short, uneven rows -- e.g. Akkadian lines -- waste far less compute on padding).
    group_by_length: bool = Field(False, description=(
        "Chat and text recipes: sort each optimizer step's rows by length before splitting "
        "them into micro-batches, so each micro-batch pads to about its own rows' length "
        "(less compute spent on padding with short, uneven rows)."))
    lora_rank: int = Field(16, ge=1, description=(
        "The rank r of each LoRA adapter."))
    lora_alpha: int = Field(32, ge=1, description=(
        "LoRA alpha: each adapter's output is scaled by lora_alpha / lora_rank."))
    lora_dropout: float = Field(0.05, ge=0.0, lt=1.0, description=(
        "Dropout on each LoRA adapter's input, in training. 0.05 is PorosConfig's default "
        "(PorosTrainer, poros.Run, poros train's classic loop); poros train's SFT and DPO "
        "recipes (chat or preference-pair rows) train at 0, the setting their receipts "
        "measured, unless you set it. Every loop draws the same masks: block-major's update is "
        "the sequential loop's at any value."))
    lora_target_modules: Optional[Union[List[str], str]] = Field(None, description=(
        "The modules that get adapters: a list of module names (e.g. [q_proj, v_proj]), one "
        "name or pattern, or null (default) for the architecture's own choice (its attention "
        "and MLP projections). Naming lm_head or embed_tokens trains that table: such a run "
        "trains sequentially."))
    lora_bias: Literal["none", "all", "lora_only"] = Field("none", description=(
        "Which biases train beside the adapters: none (default), all, or lora_only (PEFT's "
        "meanings)."))

    # -- Optimizer --
    learning_rate: float = Field(2e-5, gt=0.0, description=(
        "AdamW's learning rate after warmup."))
    adam_beta1: float = Field(0.9, ge=0.0, lt=1.0, description=(
        "AdamW's beta1 (first-moment decay)."))
    adam_beta2: float = Field(0.999, ge=0.0, lt=1.0, description=(
        "AdamW's beta2 (second-moment decay)."))
    adam_epsilon: float = Field(1e-8, gt=0.0, description=(
        "AdamW's epsilon."))
    weight_decay: float = Field(0.01, ge=0.0, description=(
        "AdamW's decoupled weight decay on the adapters."))
    max_grad_norm: float = Field(1.0, ge=0.0, description=(
        "Clip the adapters' gradient norm to this before each step; 0 turns clipping off (the "
        "norm is still read each step, one reduction over the adapters' gradients, so a "
        "non-finite one stops the run before it is applied)."))
    warmup_steps: int = Field(100, ge=0, description=(
        "Steps of linear learning-rate warmup; a run never warms up over more than about a "
        "tenth of its steps."))
    lr_scheduler: Literal["constant", "cosine", "linear"] = Field("cosine", description=(
        "The learning-rate schedule after warmup: cosine (default), linear or constant."))

    # -- Training --
    num_epochs: int = Field(1, ge=1, description=(
        "Passes over the data, when max_steps is -1."))
    per_device_batch_size: int = Field(1, ge=1, description=(
        "Rows in each micro-batch."))
    gradient_accumulation_steps: int = Field(8, ge=1, description=(
        "Micro-batches in each optimizer step: a step trains on per_device_batch_size x "
        "gradient_accumulation_steps rows."))
    max_steps: int = Field(-1, ge=-1, description=(
        "Optimizer steps to train; -1 (default) trains num_epochs passes over the data "
        "instead."))
    max_seq_length: int = Field(2048, ge=1, description=(
        "Tokens per row; longer rows are truncated."))
    seed: int = Field(42, description=(
        "Seed for the data order, the adapters' initialization and the dropout masks."))

    # Machine-written provenance (present in saved poros_config.json when the
    # base was loaded from a local HF-cache snapshot): the exact path the
    # weights came from. Ignored as an input setting.
    resolved_base_model_path: Optional[str] = Field(None, description=(
        "Machine-written provenance: the local snapshot path the base's weights were loaded "
        "from. Ignored as a setting."))
    # The Hub revision (snapshot commit) the base was loaded from. Written
    # by save(); consumed by eval/export so the adapter is re-applied onto
    # the SAME frozen weights -- a same-shaped later revision passes every
    # key/shape check while silently changing the base.
    model_revision: Optional[str] = Field(None, description=(
        "The Hub revision (snapshot commit) of the base. Written by save(); eval and export "
        "load the same revision, so an adapter is never applied to a same-shaped later base."))

    @field_validator("model_name_or_path")
    @classmethod
    def _base_is_what_it_says(cls, value: str) -> str:
        """The one place every run, export and prove passes before any base loads."""
        if planted_base(value):
            raise ValueError(f"a folder here is named {value}, and it would load in place of the "
                             f"Hub's: ./{value} takes that folder, or work in another one")
        if os.path.isdir(value):
            regular_only(value)
        return value

    def warmup_for(self, steps: int) -> int:
        """The steps a run of ``steps`` ramps over: warmup_steps, never past about a tenth of it."""
        return min(self.warmup_steps, max(1, steps // 10))

    @field_validator("model_revision")
    @classmethod
    def _check_revision(cls, v: Optional[str]) -> Optional[str]:
        return no_pull_request(v)

    # -- Memory --
    # NF4 load strategy. "auto" (default): try the bounded-GPU streamed
    # load (byte-identical to the standard load on every validated family)
    # and fall back to the standard loader
    # only for a checkpoint it cannot stream (StreamedLoadUnsupported); any other error raises.
    # "gpu" forces the standard loader; "block_streamed" forces the
    # streamed loader with no fallback.
    nf4_load_strategy: str = Field("auto", description=(
        "How the NF4 base is loaded. auto (default): the bounded-GPU streamed load "
        "(byte-identical to the standard load on every validated family), falling back to the "
        "standard loader only for a checkpoint it cannot stream (a weight transform beyond a "
        "rename, or a tensor the model has no place for); any other error stops the load. "
        "gpu: the standard loader. block_streamed: the streamed loader with no fallback."))
    enable_prefetch: bool = Field(True, description=(
        "Copy the next frozen block to the GPU while the current one computes (a side CUDA "
        "stream); same values. Default true."))
    # Where a MoE model's quantized expert slabs live (Solid State Streaming,
    # poros.sss). "ram": pinned host RAM (the default, today's behaviour).
    # "ssd": a .sss file under store_path, streamed from the SSD through a
    # pinned ring of at most sss_ram bytes; the first run writes the file,
    # later runs reuse it. Same bytes either way, so numerics are identical.
    # Dense models support "ram" only in this release. Any other name is a store
    # an installed package provides (poros.stores entry points); store_path and
    # store_options are then that plugin's (its documentation says what they mean).
    store: str = Field("ram", description=(
        "Where a MoE model's quantized expert slabs live (poros.sss). ram (default): pinned "
        "host RAM. ssd: a .sss file under store_path, streamed from the SSD through a pinned "
        "ring of at most sss_ram bytes (the first run writes the file, later runs reuse it); "
        "the same bytes, so the same values. Off a Mac, dense models take ram only. On a Mac, "
        "for any model: ram holds every layer in memory, ssd reads them from its 4-bit copy on "
        "the disk, and unset, the Mac's table picks. Any other name is a store an installed "
        "package provides (poros.stores entry points)."))
    # MoE expert compute: "reference" (bucketed bmm, the bitwise reference),
    # "grouped" (poros.moe.grouped: one CUTLASS grouped GEMM per expert chunk, far
    # fewer launches; a second self-consistent backend, refused on hosts without a
    # recorded pin -- sequential and block-major stay bitwise equal within it), or
    # "fused" (poros.moe.fused: Triton kernels over each expert's exact rows, CUDA,
    # bf16; a row's values do not depend on its call, so block-major runs a layer's
    # experts once for all micro-batches -- poros.moe.batching -- and stays the
    # sequential loop's update bit for bit).
    # None (default): poros.core.compute.choose -- "fused" where it qualifies (CUDA with Triton,
    # bf16, the built-in RAM or SSD store, fused_compute not false, every layer's experts SiLU),
    # else "reference". Fused qualified on its fp64 check: nearer the exact math than the
    # reference on every layer measured (docs/validation/moe/4090/). The trainer records what
    # ran (ran_compute) and why a default took the reference (compute_reason).
    moe_expert_compute: Optional[Literal["reference", "grouped", "fused"]] = Field(
        None, description=(
        "MoE models: the expert compute. null (default): fused (Triton kernels over each "
        "expert's exact rows, CUDA, bf16) where every layer qualifies, else reference "
        "(poros.core.compute); reference: transformers' math batched over groups of experts; "
        "grouped: one grouped GEMM per expert chunk. What ran is ran_compute."))
    # MoE expert grouping per layer call: "pow2" (the reference, what the MoE receipts ran) or
    # "single" (one padded group: far fewer kernel launches, OLMoE 4.9 -> 3.5 s, Qwen3-30B-A3B
    # 16.7 -> 13.3 s per update block-major; a different, self-consistent bit pattern). None
    # (default) = "pow2" on every loop. moe_grouping() resolves it.
    moe_expert_grouping: Optional[Literal["pow2", "single"]] = Field(None, description=(
        "MoE models: how each layer call groups its experts' rows. pow2 (null, the default): the "
        "reference, the same bits on every loop. single: one padded group, far fewer kernel "
        "launches, a different self-consistent bit pattern (its block-major and sequential "
        "runs agree with each other, not with pow2)."))
    # Inkling 2 Mini's experts as a run holds them (poros.moe.inkling2): "released", the
    # checkpoint's own NVFP4 / MXFP8 bytes decoded for each use, or "nf4", decoded and quantized
    # to NF4 as for other MoE families. None (default): POROS_INKLING_EXPERTS where set, else
    # what the checkpoint stores decides (poros.moe.inkling2.formats.decide): released for a
    # quantized checkpoint, nf4 for an unquantized one (a merged export, a bf16 save). The
    # trainer records the one the run held here, so its checkpoints, adapter, eval and export
    # read it, and a resume or a run from its adapter keeps it.
    moe_expert_format: Optional[Literal["released", "nf4"]] = Field(None, description=(
        "Inkling 2 Mini: how its experts are held. released: the checkpoint's own NVFP4 and "
        "MXFP8 bytes, decoded for each use, so training starts from exactly the released model. "
        "nf4: decoded and quantized to NF4, as other MoE families. null (default): "
        "POROS_INKLING_EXPERTS where set, else the checkpoint decides: released where it stores "
        "its experts quantized (the release), nf4 where it stores them unquantized (a merged "
        "export, a bf16 save); released of an unquantized checkpoint is refused. The run "
        "records the one it held; eval, export, a resume and a run from its adapter keep it."))
    # MoE: the device expert cache after update 1. None (default): on path efficient, the lowest-
    # VRAM path, one layer's experts (as 0); on fast, what fits this GPU (poros.train.compute: the
    # VRAM the process could hold at setup, minus the reserved peak through update 1 and its
    # optimizer step, minus what an update at the full shape would add, minus a max(3 GiB, 0.2 x
    # budget) margin). A number caps the whole cache at that many GiB (never below one layer's
    # experts), 0 keeps it at one layer. What it resolved to and why: ran_compute
    # (moe_expert_cache_gib, moe_expert_cache_reason).
    moe_expert_cache_gib: Optional[float] = Field(None, ge=0, description=(
        "MoE models: the device expert cache, GiB. null (default): one layer's experts on path "
        "efficient; on fast, sized to this GPU before update 2 (the VRAM the process "
        "could hold at setup, minus the reserved peak through update 1 and its optimizer step, "
        "minus room for a full-shape update, minus a max(3 GiB, 0.2 x that VRAM) margin). A "
        "number caps it (never below one layer); 0 keeps one layer. It decides when expert "
        "bytes move, never a value. What it resolved to: ran_compute."))
    store_path: Optional[str] = Field(None, description=(
        "store: ssd: the directory for the model's .sss file (or the file itself). A plugin "
        "store: what its documentation says."))
    store_options: dict[str, Any] = Field(default_factory=dict, description=(
        "A plugin store's options, handed to it as given. store: ssd reads cache: the pinned "
        "host RAM, beyond its read ring, that keeps expert slabs once read (4G, 512M; default "
        "none), so a slab asked for again before it is evicted comes from RAM, not the SSD: "
        "fewer disk reads for that much more host RAM, the same bytes. store: ram reads none."))
    # The distribution that provides a plugin store, named in the error when it is
    # not installed (e.g. "my-store-package"); optional.
    store_package: Optional[str] = Field(None, description=(
        "The distribution that provides a plugin store, named in the error when it is not "
        "installed (optional)."))
    sss_ram: int = Field(4 << 30, ge=1 << 20, description=(
        "store: ssd: bytes of pinned host RAM for the ring the expert slabs stream through "
        "(YAML and the CLI also take 4G, 512M). Default 4 GiB."))
    offload_suffix_inputs_to_cpu: bool = Field(False, description=(
        "Keep each streamed block's saved input in host RAM, not VRAM, until its backward "
        "reads it; less VRAM, same values."))

    # -- Data --
    dataset: Optional[str] = Field(None, description=(
        "The training data: a local .jsonl, .json or .csv file, a Hugging Face dataset id "
        "(e.g. OpenAssistant/oasst1), or synthetic (license-free random tokens, no download)."))
    # The loop that trains on it (poros.recipes). "auto" reads the data:
    # chats train SFT, preference pairs train DPO, anything else the
    # classic loop. ``loop`` is your own file with main(run, data, ...).
    recipe: Literal["auto", "sft", "dpo", "ipo", "kto", "orpo", "simpo"] = Field(
        "auto", description=(
        "The loop that trains on the data (poros.recipes). auto (default) reads the data: "
        "chats train SFT, preference pairs train DPO, anything else the classic loop. sft, "
        "dpo, ipo, kto, orpo or simpo forces one."))
    loop: Optional[str] = Field(None, description=(
        "Your own training loop: a Python file whose main(run, data, ...) trains with "
        "poros.Run's verbs, in place of a recipe."))
    train_on: Literal["answers", "last", "all"] = Field("answers", description=(
        "SFT on chat rows: which tokens the loss scores. answers (default): every assistant "
        "turn; last: the last assistant turn only; all: every token."))
    dpo_beta: float = Field(0.1, gt=0.0, description=(
        "DPO's beta: how strongly the preference loss holds the trained model to the "
        "reference."))
    # A new run from an earlier adapter: its weights (a checkpoint dir, or a run's folder for
    # its newest checkpoint), a fresh optimizer and schedule at step 0. More or other data
    # after a run, DPO on top of SFT. The verb form's --from.
    init_adapter: Optional[str] = Field(None, description=(
        "Start a new run from these adapter weights (a checkpoint directory, or a run's folder "
        "for its newest checkpoint) with a fresh optimizer and schedule at step 0: more or "
        "other data after a run, or DPO on top of SFT. Needs the same base, lora_rank and "
        "lora_target_modules, and another output_dir."))
    # Held-out rows, the training data's shape: their loss before the first
    # step (the model it started from), on the way (eval_steps, eval_checkpoints) and after the
    # last; pairs score DPO's. Every loop scores it, the engine's own too.
    eval_dataset: Optional[str] = Field(None, description=(
        "A file of held-out rows in the training data's shape: their loss is scored before "
        "the first step, every eval_steps, at each checkpoint (eval_checkpoints) and after the "
        "last (pairs score DPO's loss). On every loop, the engine's own too."))

    @field_validator("dataset", "eval_dataset")
    @classmethod
    def _data_is_what_it_says(cls, value: Optional[str]) -> Optional[str]:
        """A Hub dataset is asked about, as a download. A folder here under its name would load in
        its place with no one asked: the same rule as a base. A file is only ever itself."""
        if value and planted_base(value) and not os.path.isfile(value):
            raise ValueError(f"a folder here is named {value}, and it would load in place of the "
                             f"Hub's: ./{value} takes that folder, or work in another one")
        return value

    # Without eval_dataset, on a recipe's run: this share of the rows or pairs (2 to 100 of
    # them, none under 40 rows: poros.data._held) is held out and scored the same way. 0 trains
    # on every row. The engine's own loop holds none out: it trains every row.
    eval_split: float = Field(0.05, ge=0, lt=0.5, description=(
        "Recipe runs, without eval_dataset: the share of the rows (or pairs) held out and "
        "scored the same way: 2 to 100 of them, and none when the data has under 40 rows; 0 "
        "trains on every row. The engine's own loop (text rows, other files) holds none out."))
    eval_rows: int = Field(0, ge=0, description=(
        "Recipe runs: or this many rows held out, whatever the share says (at most half the "
        "rows). 0: the share."))

    # -- Output --
    output_dir: str = Field("./poros_output", description=(
        "Where checkpoints, the adapter, events.jsonl and the run's records are written."))
    save_steps: int = Field(500, ge=0, description=(
        "Write a checkpoint every this many steps; 0 turns periodic saves off."))
    eval_steps: int = Field(500, ge=0, description=(
        "Score the held-out rows every this many steps; 0 turns periodic eval off. A run with "
        "no held-out rows scores nothing."))
    # A run scores its held-out rows at each checkpoint too, into the checkpoint's eval.json:
    # one pass over them a save, never two on a step eval_steps also lands on.
    eval_checkpoints: bool = Field(True, description=(
        "Also score the held-out rows at each checkpoint, into its eval.json (once a save, "
        "never twice on a step eval_steps also lands on)."))
    logging_steps: int = Field(10, ge=1, description=(
        "Log a step's figures (and print the progress line) every this many steps."))
    save_total_limit: int = Field(3, description=(
        "Checkpoints kept in output_dir, the newest; 0 or less keeps every one."))
    # On a Mac: stop before the run hurts the machine (poros/metal/watchdog.py). Reads, so no output
    # holds it, and off a Mac nothing does.
    memory_guard: bool = Field(True, exclude=True, description=(
        "On a Mac: stop the run before memory pressure hurts the machine "
        "(poros/metal/watchdog.py). Read only; never written to a run's records, and nothing "
        "reads it off a Mac."))
    # Live stdout progress display. When True (default), PorosTrainer.train()
    # prints a compact per-`logging_steps` line and a final summary directly to
    # stdout, independent of any root-logger configuration -- so a plain script
    # (or the CLI) that never calls logging.basicConfig still sees live loss,
    # speed, and the headline peak-VRAM number. Set False to restore the
    # logger-only behavior (nothing prints unless you configure logging). The
    # POROS_PROGRESS env var overrides this (POROS_PROGRESS=0 forces it off).
    progress: bool = Field(True, description=(
        "Print a live progress line every logging_steps and a final summary to stdout, "
        "whatever the logging configuration (default true). false leaves output to the "
        "logger. POROS_PROGRESS=0 forces it off."))
    push_to_hub: bool = Field(False, description=(
        "After the run, upload the adapter to the Hugging Face Hub as hub_model_id (uses your "
        "own Hub login)."))
    hub_model_id: Optional[str] = Field(None, description=(
        "The Hub repository push_to_hub uploads to."))

    # -- Advanced --
    adapter_backend: Literal["auto", "peft", "native"] = Field("auto", description=(
        "What implements the LoRA adapters: peft, native (Poros's own), or auto (default): "
        "peft where it is installed, else native."))
    adapter_runtime: SkipJsonSchema[Literal["auto"]] = Field("auto", description=(
        "Fixed at auto (FIXED_FIELDS): not a setting."))
    bf16_forward_math_mode: SkipJsonSchema[Literal["auto"]] = Field("auto", description=(
        "Fixed at auto (FIXED_FIELDS): not a setting."))
    # The loop a CUDA run trains with. efficient and fast are block-major accumulation
    # (poros.train.block_major): each frozen block streams once a step for all its
    # micro-batches instead of once per micro-batch, bit for bit the sequential loop's update,
    # dropout included (poros.core.sequential_rng); at one micro-batch a step it skips the
    # per-layer recompute inside a block, and is no slower. efficient: less VRAM than fast
    # (block inputs in host RAM, every frozen weight dequantized where it is used); fast: the
    # fastest (block inputs on the GPU, one block's dequantized NF4 weights kept for all its
    # micro-batches). None (default): decided for the run (apply_cuda_defaults below); the Mac
    # and the CPU leave it unset and train sequentially. block_size stays the knob within
    # either block-major path: 1 holds one layer's activations in a block's backward, not four.
    path: Optional[Literal["efficient", "fast", "sequential"]] = Field(None, description=(
        "The training loop (CUDA). efficient, the default: block-major accumulation, less VRAM "
        "than fast: each frozen block streams to the GPU once a step for all its micro-batches, "
        "its inputs wait for the backward in host RAM (NF4 only) and every frozen weight is "
        "dequantized where it is used. fast: block-major, the fastest: block inputs stay on the "
        "GPU and one block's dequantized NF4 weights are "
        "kept for all its micro-batches; more VRAM. sequential: one micro-batch at a time "
        "through every block; slower, and its peak can be lower than efficient's. All three "
        "train the same update, bit for bit, at any number of micro-batches a step. null "
        "(default): efficient, else sequential where block-major "
        "cannot train the same update (a trained embedding table or LM head, the model's own "
        "dropout, a MoE model's router aux loss); that reason is logged and recorded as "
        "path_reason, and the loop the run's steps took as ran_path. The Mac and the CPU train "
        "sequentially."))
    # Machine-written, read back with a saved config, never a setting: why path was decided as it
    # was (apply_cuda_defaults, decide_path), and the loop the run's steps took
    # (PorosTrainer.ran_path, written into poros_config.json and config.resolved.json).
    path_reason: Optional[str] = Field(None, description=(
        "Machine-written: why the run's loop was decided as it was, where path was left unset "
        "and the run did not take block-major's first choice (e.g. a trained LM head)."))
    ran_path: Optional[Literal["efficient", "fast", "sequential"]] = Field(None, description=(
        "Machine-written: the loop the run's steps took, beside path as set (sequential "
        "where block-major gave up on an update: a draw it cannot follow)."))
    # Frozen blocks kept on the device for the whole run instead of streamed
    # (spread evenly). Same tensors, so numerics are unchanged; each one costs
    # its block's VRAM and saves its H2D copies. On CUDA a resident block has
    # no host copy (NF4: quantized straight onto the GPU), so each one moves
    # its weights' bytes from host RAM to VRAM. 0 = stream every block.
    resident_blocks: int = Field(0, ge=0, description=(
        "Frozen blocks kept on the GPU for the whole run instead of streamed, spread evenly; "
        "same values. Each costs its block's VRAM and saves its copies to the GPU. 0 "
        "(default): stream every block."))
    # The frozen input embedding table and LM head live in host RAM instead of
    # VRAM: the embedding lookup runs on the CPU and only its rows are copied
    # up; the head's weight is copied up for each head pass (once per update
    # under block-major) and dropped after. Same values, same kernels. On NF4
    # they stay in their load dtype there (bf16: half of fp32), and k-bit
    # preparation's exact fp32 cast is made on the rows a lookup reads and on
    # the head's device copy (poros.core.tables). None (default): decided for
    # a CUDA run (decide_offload): on where the plan says the run would not fit
    # the GPU with the tables on it and does with them in host RAM, recorded as
    # offload_reason; else off.
    offload_embeddings_and_head: Optional[bool] = Field(None, description=(
        "The frozen embedding table and LM head in host RAM instead of VRAM: the lookup's rows "
        "and the head's weight are copied up as they are used; same values. null (default): "
        "decided for a dense NF4 CUDA run: on where the check finds the run past the GPU with "
        "them on it and not with them in host RAM (offload_reason says so), else "
        "off."))
    offload_reason: Optional[str] = Field(None, description=(
        "Machine-written: why offload_embeddings_and_head, left unset, was turned on for the "
        "run (the check with the tables on the GPU and in host RAM)."))
    # NF4 block-major: keep each frozen weight's dequantized copy for the
    # current block visit and reuse it across the G microsteps (forward,
    # rematerialization and grad-input matmul) instead of dequantizing it
    # 3G times. bitsandbytes' own kernel choice is kept per call -- the cache
    # only stands in where bnb would itself dequantize then matmul -- so the
    # numbers are unchanged. Cap in MiB on the cached weights (0 = off);
    # weights past the cap are dequantized per call. None (default) = as path
    # says: on the fast path one block's dequantized weights, at most half of
    # the GPU's memory (poros.train.compute: auto_dequant_cache_mb); off on the
    # efficient path.
    nf4_dequant_cache_mb: Optional[int] = Field(None, ge=0, description=(
        "Advanced: a cap. NF4 block-major accumulation (dense models): MiB of VRAM for each "
        "frozen weight's dequantized copy, kept for one block's visit and reused by all its "
        "micro-batches instead of dequantized again; same values. null (default): as path "
        "says: fast, one block's dequantized weights, at most half of the GPU's memory; "
        "efficient, off. 0: off. Read by nothing on the sequential loop."))
    # NF4 on CUDA: run the decoder layers' LoRA-over-NF4 compute through poros.core.fused
    # (Triton): the LoRA sums, SwiGLU, RMSNorm, rotary and residual adds in a few fused kernels
    # at PEFT's precisions (bf16 frozen GEMMs, fp32 adapters and activations), causal attention
    # through the deterministic flash kernel in bf16 (fp32 accumulation; a measured precision
    # change: poros.core.fused.attention), the LM head and cross-entropy a chunk of rows at a
    # time, a head whose values are exactly bf16 multiplying the hidden state as a bf16 high
    # half plus a bf16 low half (its bits past 16 dropped: poros.core.fused.head); a recomputed
    # block skips its last frozen GEMM,
    # whose output no gradient reads. Every path (streamed, resident, sequential, block-major)
    # runs the same kernels, so they stay bitwise-equal to each other; against PEFT's own modules
    # the sums are formed in another order (fp32 rounding). None (default) = on where it
    # applies (NF4, PEFT adapters, CUDA with Triton); False keeps PEFT's modules.
    fused_compute: Optional[bool] = Field(None, description=(
        "NF4 on CUDA: the decoder layers' LoRA, SwiGLU, norm, rotary, attention and residual "
        "math, and the head's loss, in fused kernels at PEFT's precisions except two: causal "
        "attention runs in bf16 (flash, fp32 accumulation), and an LM head whose values are "
        "exactly bf16 (a bf16 checkpoint's) multiplies the fp32 hidden state as two bf16 halves, "
        "dropping its bits past 16 (about 2^-17 relative). Every Poros path stays bitwise-equal "
        "to the others. null (default): on where it applies; false: PEFT's modules. On a MoE "
        "model with the fused expert compute, its attention, RMSNorms and loss. What ran is "
        "ran_compute."))
    # Machine-written, read back with a saved config, never a setting: what computed the run
    # (poros.core.compute decides; the trainer records), beside the settings, as ran_path is.
    ran_compute: Optional[Dict[str, Any]] = Field(None, description=(
        "Machine-written: what computed the run -- fused_compute (bool) and, on a MoE model, "
        "moe_expert_compute (fused, reference or grouped) -- and the caches as resolved: "
        "moe_expert_cache_gib (the expert residency's capacity, after its growth), "
        "nf4_dequant_cache_mb (the dequantized-weight cache, its auto size resolved) and, where "
        "ExpertAdamW steps bf16 trainable tensors, expert_optimizer (its fused kernel or its "
        "reference, by device, with the reference's measured seconds per update)."))
    compute_reason: Optional[str] = Field(None, description=(
        "Machine-written: why a default took the slower compute, where it did (no CUDA or "
        "Triton, a non-bf16 model, another store, a grouping, fused_compute: false, or a model "
        "whose experts the fused kernels cannot compute, such as Gemma 4's GELU experts)."))
    # PYTORCH_CUDA_ALLOC_CONF expandable_segments for this process unless the
    # environment already sets an allocator config (PYTORCH_ALLOC_CONF or
    # PYTORCH_CUDA_ALLOC_CONF; fewer fragmented segments: 72B bs1 reserved
    # 15.0 -> 13.6 GiB, bitwise the same). A GPU that cannot map them (no CUDA
    # VMM) keeps the default allocator. On CUDA the trainer sets this field to what
    # the allocator runs with (apply_expandable_segments), so the records say it.
    cuda_expandable_segments: bool = Field(True, description=(
        "CUDA: expandable allocator segments for this process, fewer fragmented segments, same "
        "values. Not applied where PYTORCH_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF is set (that "
        "setting wins) or in a process that already holds CUDA memory; a GPU that cannot map "
        "them (no CUDA VMM) keeps the default allocator. A CUDA run's records say what its "
        "allocator ran with."))
    packed_block_transfers: SkipJsonSchema[Literal[True]] = Field(True, description=(
        "Fixed at true (FIXED_FIELDS): not a setting."))
    bf16_cuda_graph_blocks: SkipJsonSchema[Literal[False]] = Field(False, description=(
        "Fixed at false (FIXED_FIELDS): not a setting."))
    debug_memory: bool = Field(False, description=(
        "Diagnostics: record each block's rematerialization events on the trainer (its "
        "_remat_event_log). No effect on training."))
    # SECURITY: enabling trust_remote_code lets a model/dataset repo execute
    # arbitrary code on load (see SECURITY.md). Poros's validated bases
    # are all natively supported by transformers and need it OFF, so off is the default: a
    # typo'd or compromised repo id must not be able to cross from
    # "download weights" into "run this repo's Python". Turn it on
    # deliberately (poros train <config.yaml> --trust-remote-code, or this
    # field in your own Python; a config file or key=value cannot) only for a
    # base you trust that ships custom modeling code.
    trust_remote_code: bool = Field(False, description=(
        "Let a model or dataset repository run its own Python on load. Off by default: the "
        "validated bases need none of it, and a typo'd or compromised repository id must not "
        "run code. Turn it on (poros train --trust-remote-code, or this field in your own "
        "Python; a config file or key=value cannot) only for a base you trust."))
    # Attention backend passed to AutoModelForCausalLM.from_pretrained. None lets
    # transformers pick its default (sdpa); an architecture with no SDPA implementation needs
    # "eager".
    attn_implementation: Optional[str] = Field(None, description=(
        "The attention backend passed to transformers' from_pretrained. null (default): "
        "transformers' own choice (sdpa). An architecture with no SDPA implementation needs "
        "eager."))

    @model_validator(mode="before")
    @classmethod
    def _fixed_fields_stay_fixed(cls, data: Any) -> Any:
        """A fixed field (FIXED_FIELDS) given another value is refused by name, not trained
        past."""
        if isinstance(data, dict):
            moved = [f"{k}: {data[k]!r}" for k, v in FIXED_FIELDS.items()
                     if k in data and str(data[k]).lower() != str(v).lower()]
            if moved:
                from poros.core.errors import PorosConfigError

                raise PorosConfigError(
                    f"{', '.join(moved)}: not a setting; Poros trains its NF4 base one way.",
                    remediation="Remove the line from the config.")
            data = {**data, **{k: v for k, v in FIXED_FIELDS.items() if k in data}}
        return data

    @model_validator(mode="after")
    def _tables_stay_frozen(self) -> "PorosConfig":
        """freeze_embeddings / freeze_lm_head false trained nothing (the table's gradient never
        reached an optimizer): refused, naming the field, rather than accepted and ignored."""
        thawed = [n for n in ("freeze_embeddings", "freeze_lm_head") if not getattr(self, n)]
        if thawed:
            from poros.core.errors import PorosConfigError

            raise PorosConfigError(
                f"{' and '.join(f'{n}: false' for n in thawed)}: training the embedding table "
                "or the LM head itself is not supported.",
                remediation="Leave them frozen (true, the default); to adapt a table, name "
                "embed_tokens or lm_head in lora_target_modules.")
        return self

    @field_validator("sss_ram", mode="before")
    @classmethod
    def _parse_sss_ram(cls, v: object) -> object:
        return parse_bytes(v) if isinstance(v, str) else v

    @field_validator("store")
    @classmethod
    def _check_store_name(cls, v: str) -> str:
        if not _STORE_NAME.match(v):
            raise ValueError(f"store must be 'ram', 'ssd' or a plugin store name, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_store_installed(self) -> "PorosConfig":
        from poros.stores import check_store_name

        try:
            check_store_name(self.store, self.store_package)
        except ImportError as exc:
            raise ValueError(str(exc)) from None
        return self

    @model_validator(mode="after")
    def _check_moe_grouping(self) -> "PorosConfig":
        # The grouping buckets the reference compute's rows; the fused and grouped computes run
        # each expert's exact rows and would silently ignore it.
        if self.moe_expert_grouping is not None and self.moe_expert_compute in ("fused",
                                                                                "grouped"):
            raise ValueError(
                f"moe_expert_grouping: {self.moe_expert_grouping} groups the reference expert "
                f"compute's rows; moe_expert_compute: {self.moe_expert_compute} runs each "
                "expert's exact rows. Set moe_expert_compute: reference, or leave "
                "moe_expert_grouping unset.")
        return self

    @model_validator(mode="after")
    def _check_store(self) -> "PorosConfig":
        if self.store == "ssd" and not self.store_path:
            raise ValueError(
                "store='ssd' needs store_path: a directory on a local SSD for the "
                ".sss expert file (CLI: --store ssd:/path)."
            )
        if self.store == "ssd":
            ssd_cache_bytes(self.store_options)  # an option the SSD store does not read
        return self

    @model_validator(mode="after")
    def _check_hub_push(self) -> "PorosConfig":
        # Fail at parse, not after N GPU-hours: an explicitly requested
        # push with nowhere to push to cannot succeed.
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError(
                "push_to_hub=True requires hub_model_id='org/name'."
            )
        return self

    @model_validator(mode="after")
    def _check_path(self) -> "PorosConfig":
        if self.path in ("efficient", "fast"):  # block-major, set by you (or a saved run)
            why = sequential_reason(self)  # what the model's own config.json says: the trainer
            if why is not None:
                raise ValueError(path_refusal(self.path, why))
        return self

    # Whether path was decided for the run (apply_cuda_defaults, decide_path) rather than set.
    _path_decided: bool = PrivateAttr(False)
    # store: ssd with a store_path a run folder recorded (poros.stores.recorded_store): that
    # store is only read, and the load builds here instead when it cannot be reused. Not saved.
    _store_build_path: Optional[str] = PrivateAttr(None)
    # moe_expert_format read from a run's own record (poros eval): it is that run's base, so
    # POROS_INKLING_EXPERTS does not contradict it (poros.moe.inkling2.formats.asked_format).
    _expert_format_recorded: bool = PrivateAttr(False)

    @property
    def path_decided(self) -> bool:
        """Whether ``path`` was decided for the run (apply_cuda_defaults, ``decide_path``)
        rather than set."""
        return self._path_decided

    def decide_path(self, path: str, reason: str) -> None:
        """Decide ``path`` for the run, for ``reason`` (a path you set stays as set)."""
        if self.path is not None and not self._path_decided:
            return
        self.path, self.path_reason, self._path_decided = path, reason, True

    @model_serializer(mode="wrap")
    def _dump(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        """Every field, but offload_reason only where a run wrote one: a config that never
        decided the tables' place dumps as it did before the field was added; and
        moe_expert_format only where a run uses it (Inkling 2 Mini): every other family's
        poros_config.json, and so its adapter's manifest, is as it was before the field."""
        out = handler(self)
        if isinstance(out, dict):
            for name in ("offload_reason", "moe_expert_format"):
                if out.get(name) is None:
                    out.pop(name, None)
        return out

    def decide_offload(self, on: bool, reason: Optional[str] = None) -> None:
        """Decide ``offload_embeddings_and_head`` for the run where it was left unset, for
        ``reason`` where it is turned on (a value you set stays as set)."""
        if self.offload_embeddings_and_head is not None:
            return
        self.offload_embeddings_and_head = on
        self.offload_reason = reason if on else None

    def apply_cuda_defaults(self, model: Optional[dict] = None) -> None:
        """Decides ``path`` where this config left it unset, for a CUDA run: block-major's
        ``efficient``, else ``sequential`` for a reason this records as ``path_reason``: what
        sequential_reason(self, model) finds block-major cannot follow. ``model`` is the model's
        config.json (None: not known; the trainer checks the blocks it loads). A path you set is
        kept as set.

        Called once the run is known to be on CUDA (PorosTrainer.from_pretrained, poros
        train); the Mac and the CPU never call it, so they train sequentially."""
        if self.path is not None:
            return
        why = sequential_reason(self, model)
        if why is not None:
            self.decide_path("sequential", why)
        else:
            self.path, self._path_decided = "efficient", True
