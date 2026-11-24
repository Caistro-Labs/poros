"""Merged export: the adapter folded into the base, one checkpoint tensor at a time.

``poros export`` writes the model the run trained. Each weight is the
weight training saw plus its LoRA delta ``scaling * B @ A``, added in fp32
and rounded once to the checkpoint dtype. For an NF4 run the weight
training saw is the base weight's NF4 round trip: the bitsandbytes
quantization the training load applied (``BitsAndBytesConfig`` NF4,
double quant, uint8 storage, bnb's blocksize), dequantized to the run's
dtype. Every linear layer that load quantized is written as that
dequantized value, plus its delta when it is a LoRA target. Biases the
run trained (``lora_bias``) replace the base's. Everything else
(embeddings, norms, the output head) is copied byte for byte.

The model is never resident anywhere. The merge walks the base's
safetensors shards and reads one tensor at a time with plain file reads
(no mmap). It writes each result straight into its output shard, so peak
host memory is a few copies of the largest merged tensor plus the
adapter. Peak GPU memory, for the NF4 round trip, is one tensor. Each
output shard reuses its source shard's header verbatim: tensor names,
dtypes, shapes, offsets, shard files and the index are the base's, so the
output loads wherever the base does (transformers, vLLM, TGI).

The NF4 round trip runs on CUDA when a GPU is there. It is the same
``bnb.nn.Params4bit(...).to(cuda)`` call the training load makes, so it is
exact by construction. Without a GPU it runs ``poros.moe.quant``, which
matches bitsandbytes' CPU kernels bit for bit. bnb's CUDA quantize kernel
is built with fast math (an approximate reciprocal of each block's
absmax), so a CPU result can pick a neighbouring NF4 code for an element
that sits on a code boundary. The report says which device ran.

``poros.moe.export`` uses the same reader and writer for MoE experts.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from poros.core.errors import PorosExportError
from poros.train.streamed_load import _key_remapper, _resolve_checkpoint_dir, _shard_map

if TYPE_CHECKING:  # pragma: no cover
    from poros.core.config import PorosConfig

__all__ = [
    "CheckpointReader",
    "MergeReport",
    "export_merged_dense",
    "nf4_roundtrip",
    "write_rewritten",
]

# safetensors dtype names -> torch, for the tensors a merge reads.
_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_HEADER_LIMIT = 100 << 20  # safetensors' own cap on the JSON header
_COPY_CHUNK = 64 << 20


# ---------------------------------------------------------------------------
# Reading and writing safetensors one tensor at a time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    shard: str
    dtype: str
    shape: Tuple[int, ...]
    begin: int  # absolute file offsets
    end: int


class CheckpointReader:
    """A safetensors checkpoint's headers, and its tensors read one at a time.

    Plain ``seek`` + ``readinto``, no mmap: a mapped shard keeps every page
    it has touched resident until it is closed, which is a whole shard.
    """

    def __init__(self, source: str) -> None:
        if sys.byteorder != "little":  # pragma: no cover
            raise PorosExportError("merged export needs a little-endian host")
        self.source = source
        self.shards = sorted(_shard_map(source))
        self.headers: Dict[str, bytes] = {}
        self.order: Dict[str, list] = {}
        self.entries: Dict[str, _Entry] = {}
        for shard in self.shards:
            path = os.path.join(source, shard)
            with open(path, "rb") as f:
                prefix = f.read(8)
                (n,) = struct.unpack("<Q", prefix) if len(prefix) == 8 else (-1,)
                if not 0 < n <= _HEADER_LIMIT:
                    raise PorosExportError(f"{path} is not a safetensors file")
                raw = f.read(n)
            header = json.loads(raw)
            start = 8 + n
            names = []
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                if name in self.entries:
                    raise PorosExportError(f"tensor {name!r} is stored twice in {source}")
                lo, hi = spec["data_offsets"]
                self.entries[name] = _Entry(
                    shard, spec["dtype"], tuple(spec["shape"]), start + lo, start + hi
                )
                names.append(name)
            self.headers[shard] = prefix + raw
            self.order[shard] = sorted(names, key=lambda k: self.entries[k].begin)

    def get(self, name: str) -> torch.Tensor:
        """Tensor ``name``, read from its shard into a fresh CPU tensor."""
        e = self.entries[name]
        dtype = _DTYPES.get(e.dtype)
        if dtype is None:
            raise PorosExportError(f"cannot merge {name!r}: unsupported dtype {e.dtype}")
        n = e.end - e.begin
        if n == 0:
            return torch.empty(e.shape, dtype=dtype)
        buf = bytearray(n)
        with open(os.path.join(self.source, e.shard), "rb") as f:
            f.seek(e.begin)
            if f.readinto(buf) != n:
                raise PorosExportError(f"{e.shard} is truncated at {name!r}")
        return torch.frombuffer(buf, dtype=dtype).reshape(e.shape)


def shard_bytes(reader: "CheckpointReader") -> int:
    """The bytes of the base's shards: what its merged model takes on disk."""
    return sum(os.path.getsize(os.path.join(reader.source, shard)) for shard in reader.shards)


def room_for(need: int, output_dir: "str | os.PathLike") -> None:
    """Refuse, before anything is built or written, a merged model the disk under ``output_dir``
    has no room for (``need`` bytes: the base's shards, which the merge rewrites at their size).
    Files of the same names already in ``output_dir`` count for nothing: the merge is written
    whole beside them first (``staged_output``) and replaces them only then, so for a moment the
    disk holds both."""
    import shutil

    from poros.core.errors import PorosDiskError

    out = Path(output_dir)
    probe = out
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if need > free:
        raise PorosDiskError(
            f"A merged model needs {need / 1e9:.1f} GB at {out}: {free / 1e9:.1f} GB free.",
            remediation="Free some space, or name another place with --output.",
        )


def room_before_download(model: str, revision: Optional[str],
                         output_dir: "str | os.PathLike") -> None:
    """Before the base is fetched: the Hub's sizes against the disks it lands on. The weights not
    cached yet must fit the Hugging Face cache, and the merged model (every shard rewritten) the
    output disk; one disk holding both must hold the two. Refused (E_DISK) before any byte is
    downloaded; said nothing where the Hub cannot say (offline, a local folder)."""
    import shutil

    from huggingface_hub.constants import HF_HUB_CACHE

    from poros.core.errors import PorosDiskError
    from poros.plan.configs import weights_on_hub

    sizes = weights_on_hub(model, revision)
    if sizes is None:
        return
    total, fetch = sizes

    def nearest(p: Path) -> Path:
        return next((q for q in (p, *p.parents) if q.exists()), Path.home())

    cache, out = nearest(Path(HF_HUB_CACHE).expanduser()), nearest(Path(output_dir))
    cache_free = shutil.disk_usage(cache).free
    same = os.stat(cache).st_dev == os.stat(out).st_dev
    if fetch > cache_free or (same and fetch + total > cache_free):
        need = fetch + total if same else fetch
        raise PorosDiskError(
            f"Merging onto {model} downloads {fetch / 1e9:.1f} GB of its weights into the Hugging "
            f"Face cache ({cache})" + (f" and writes {total / 1e9:.1f} GB beside them" if same
                                        else "") + f": {need / 1e9:.1f} GB needed, "
            f"{cache_free / 1e9:.1f} GB free.",
            remediation="Free some space, point HF_HOME at a larger disk, or name an --output "
                        "on another disk.")
    room_for(total, output_dir)


# Written into a staging folder when it is made: a ``<output_dir>.partial`` without it is not one
# of Poros's, and is never deleted (poros/cli/commands/export.py checks it first, before the load).
STAGING_MARKER = ".poros-export-staging"


def foreign_stage(out: Path) -> Optional[str]:
    """Why ``<out>.partial`` cannot be used as the staging folder, or None: it exists and is not
    a staging folder an export made (a link, a file, or a folder without the marker)."""
    stage = out.with_name(out.name + ".partial")
    if not os.path.lexists(stage):
        return None
    if stage.is_dir() and not stage.is_symlink() and (stage / STAGING_MARKER).is_file():
        return None  # an earlier export's, stopped before it cleaned up
    return (f"{stage} is in the way: a merged export stages its files there, and that folder "
            "was not made by an export, so it is left as it is; move or rename it")


@contextlib.contextmanager
def staged_output(output_dir: "str | os.PathLike") -> Iterator[Path]:
    """Where an export writes: ``<output_dir>.partial`` beside it, moved into ``output_dir`` file
    by file only once the merged model is whole. However the export fails (a full disk, a
    refused tensor, ctrl-c) the staging folder goes and ``output_dir`` is as it was: a model
    there before is never left half old and half new. Files of other names in it stay."""
    import shutil

    out = Path(output_dir).resolve()  # "." or "./" has no name of its own to stage beside
    stage = out.with_name(out.name + ".partial")
    refusal = foreign_stage(out)
    if refusal:  # a folder of that name this export did not make: never deleted
        raise PorosExportError(refusal)
    if stage.is_symlink() or stage.is_file():
        stage.unlink()
    elif stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / STAGING_MARKER).write_text("a Poros export's staging folder\n")
    try:
        yield stage
        out.mkdir(parents=True, exist_ok=True)
        for item in sorted(stage.iterdir()):  # renames on one disk: quick, each one whole
            if item.name == STAGING_MARKER:
                continue
            target = out / item.name
            if target.is_dir() and not target.is_symlink() and not item.is_dir():
                shutil.rmtree(target)
            os.replace(item, target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _tensor_bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy())


def write_rewritten(
    reader: CheckpointReader,
    output_dir: str,
    transform: Callable[[str], Optional[torch.Tensor]],
) -> int:
    """Write every shard of ``reader`` to ``output_dir``, tensor by tensor.

    ``transform(name)`` returns the tensor to store under ``name`` (same
    dtype and shape as the source), or ``None`` to copy the source bytes
    unchanged. Each shard keeps its source header verbatim and lands under
    its own name only once complete. Returns the bytes written.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    for shard in reader.shards:
        src = os.path.join(reader.source, shard)
        dst = os.path.join(output_dir, shard)
        tmp = dst + ".partial"
        header = reader.headers[shard]
        try:
            _write_shard(reader, shard, src, tmp, header, transform)
        except BaseException:  # a .partial is never left, whatever stopped it
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        os.replace(tmp, dst)
        total += os.path.getsize(dst)
    return total


def _write_shard(reader: CheckpointReader, shard: str, src: str, tmp: str, header: bytes,
                 transform: Callable[[str], Optional[torch.Tensor]]) -> None:
    """One shard of :func:`write_rewritten`, written to ``tmp``."""
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        fout.write(header)
        pos = len(header)
        for name in reader.order[shard]:
            e = reader.entries[name]
            if e.begin != pos:
                raise PorosExportError(f"{shard} is not a contiguous safetensors file")
            out = transform(name)
            if out is None:
                fin.seek(e.begin)
                left = e.end - e.begin
                while left:
                    chunk = fin.read(min(left, _COPY_CHUNK))
                    if not chunk:
                        raise PorosExportError(f"{shard} is truncated at {name!r}")
                    fout.write(chunk)
                    left -= len(chunk)
            else:
                want = _DTYPES[e.dtype]
                if out.dtype != want or tuple(out.shape) != e.shape:
                    raise RuntimeError(
                        f"merge produced {out.dtype}{tuple(out.shape)} for {name!r}, "
                        f"the checkpoint stores {want}{e.shape}"
                    )
                data = _tensor_bytes(out)
                if data.nbytes != e.end - e.begin:  # pragma: no cover
                    raise RuntimeError(f"byte size mismatch for {name!r}")
                fout.write(data)
            pos = e.end
        if os.fstat(fin.fileno()).st_size != pos:
            raise PorosExportError(f"{shard} has bytes after its last tensor")


# ---------------------------------------------------------------------------
# The weight training saw
# ---------------------------------------------------------------------------


def _training_bnb_config(dtype: torch.dtype):
    """The ``BitsAndBytesConfig`` the NF4 training load uses (poros/train/model.py)."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def nf4_roundtrip(
    dtype: torch.dtype, device: str = "auto"
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """``(fn, device)``: ``fn(w)`` is ``w``'s NF4 round trip in ``dtype``, on CPU.

    ``device="cuda"`` quantizes with bitsandbytes on the GPU, one tensor at
    a time, exactly as the training load does. ``"cpu"`` uses
    ``poros.moe.quant`` (bnb's CPU kernels, bit for bit). ``"auto"`` picks
    CUDA when a GPU and bitsandbytes are there.
    """
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            try:
                import bitsandbytes  # noqa: F401

                device = "cuda"
            except ImportError:
                pass
    if device == "cpu":
        from poros.moe import quant

        def on_cpu(w: torch.Tensor) -> torch.Tensor:
            w = w.to(dtype)
            return quant.dequantize(quant.quantize_tensor(w), w.shape, dtype)

        return on_cpu, "cpu"
    if device != "cuda":
        raise ValueError(f"nf4 device must be 'auto', 'cuda' or 'cpu', got {device!r}")
    import bitsandbytes as bnb

    cfg = _training_bnb_config(dtype)
    # The parameter transformers' replace_with_bnb_linear creates for each
    # quantized linear; its attributes carry the quantization (bnb's own
    # default blocksize included), as Bnb4bitQuantize passes them on.
    with torch.device("meta"):
        like = bnb.nn.Linear4bit(
            1,
            1,
            False,
            cfg.bnb_4bit_compute_dtype,
            compress_statistics=cfg.bnb_4bit_use_double_quant,
            quant_type=cfg.bnb_4bit_quant_type,
            quant_storage=cfg.bnb_4bit_quant_storage,
        ).weight
    attrs = {
        k: getattr(like, k)
        for k in ("blocksize", "compress_statistics", "quant_type", "quant_storage")
    }
    cuda = torch.device("cuda", torch.cuda.current_device())

    def on_cuda(w: torch.Tensor) -> torch.Tensor:
        value = w.to(device=cuda, dtype=dtype)
        q = bnb.nn.Params4bit(value, requires_grad=False, **attrs).to(cuda)
        return bnb.functional.dequantize_4bit(q.data, q.quant_state).cpu()

    return on_cuda, "cuda"


def _skeleton(source: str, dtype: torch.dtype, trust_remote_code: bool,
              attn_implementation: Optional[str]) -> nn.Module:
    """The model class the training load builds, on the meta device (no memory)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from poros.moe.carried import ensure_registered

    ensure_registered(source)  # a family Poros carries registers itself first
    config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
    if getattr(config, "quantization_config", None):
        raise PorosExportError(
            "the base checkpoint is already quantized",
            remediation="merged export reads a bf16/fp16/fp32 base checkpoint",
        )
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config, dtype=dtype, trust_remote_code=trust_remote_code
        )
    model.name_or_path = source  # PEFT reads it back: a local dir, never a Hub lookup
    return model


def _nf4_quantized_weights(model: nn.Module) -> frozenset:
    """The weights the NF4 training load quantizes, as model keys.

    transformers' rule (``Bnb4BitHfQuantizer`` -> ``replace_with_bnb_linear``,
    mirrored by ``poros.train.streamed_load``): every exact ``nn.Linear``
    outside the not-convert set (the output head, keep-in-fp32 modules,
    config skips).
    """
    from transformers.pytorch_utils import Conv1D
    from transformers.quantizers.base import HfQuantizer
    from transformers.quantizers.quantizers_utils import should_convert_module

    skip = HfQuantizer.get_modules_to_not_convert(
        model,
        _training_bnb_config(torch.bfloat16).llm_int8_skip_modules,
        getattr(model, "_keep_in_fp32_modules", None),
    )
    keys = set()
    for name, module in model.named_modules():
        if not should_convert_module(name, skip):
            continue
        if isinstance(module, Conv1D):
            raise PorosExportError(
                f"{name} is a Conv1D layer (transposed weights)",
                remediation="merged export supports nn.Linear models (the validated families)",
            )
        if type(module) is nn.Linear:
            keys.add(f"{name}.weight")
    return frozenset(keys)


# ---------------------------------------------------------------------------
# The adapter, checked against the model before any byte is written
# ---------------------------------------------------------------------------


@dataclass
class _Plan:
    deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]]  # weight key -> (A, B)
    biases: Dict[str, torch.Tensor]  # bias key -> trained bias
    scaling: float
    adapter_format: str


# The frozen tables PEFT may save whole beside an adapter that adapts or resizes them
# (save_embedding_layers): the base's own weights, which the merge reads from the base.
_TABLE_COPIES = ("embed_tokens.weight", "lm_head.weight", "wte.weight", "embed_in.weight",
                 "embed_out.weight")


def _peft_plan(adapter_dir: Path, state: Dict[str, torch.Tensor], skeleton: nn.Module) -> _Plan:
    import inspect

    from peft import LoraConfig, get_peft_model

    from poros.train.adapters import validate_peft_adapter_state

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise PorosExportError(f"{adapter_dir} has PEFT weights but no adapter_config.json")
    if json.loads(config_path.read_text()).get("peft_type", "LORA") != "LORA":
        raise PorosExportError("merged export folds LoRA adapters only")
    lcfg = LoraConfig.from_pretrained(str(adapter_dir))
    unsupported = [
        name
        for name, on in (
            ("use_dora", getattr(lcfg, "use_dora", False)),
            ("fan_in_fan_out", getattr(lcfg, "fan_in_fan_out", False)),
            ("lora_bias", getattr(lcfg, "lora_bias", False)),
            ("rank_pattern", bool(getattr(lcfg, "rank_pattern", None))),
            ("alpha_pattern", bool(getattr(lcfg, "alpha_pattern", None))),
            ("modules_to_save", bool(getattr(lcfg, "modules_to_save", None))),
        )
        if on
    ]
    if unsupported:
        raise PorosExportError(
            f"the adapter uses {', '.join(unsupported)}, which merged export does not fold",
            remediation="serve the adapter files in the run directory with PEFT instead",
        )
    kwargs = {}
    if "low_cpu_mem_usage" in inspect.signature(get_peft_model).parameters:
        kwargs["low_cpu_mem_usage"] = True
    # Name the base as the skeleton does (the resolved checkpoint dir): PEFT
    # warns on a "renamed" base and checks the name for a config.json.
    lcfg.base_model_name_or_path = skeleton.name_or_path
    # The exact-match gate resume and eval use: a truncated, foreign or
    # reshaped adapter is refused here, before anything is written.
    from poros.train.model import pefts_checkpointing_names

    try:
        with pefts_checkpointing_names():  # PEFT's own reads of a deprecated name
            wrapped = get_peft_model(skeleton, lcfg, **kwargs)
        validate_peft_adapter_state(wrapped, state, str(adapter_dir))
    except RuntimeError as exc:
        raise PorosExportError(str(exc)) from exc
    scaling = lcfg.lora_alpha / (math.sqrt(lcfg.r) if lcfg.use_rslora else lcfg.r)
    prefix = "base_model.model."
    plan = _Plan({}, {}, scaling, "peft")
    a_of, b_of = {}, {}
    for key, tensor in state.items():
        k = key[len(prefix):] if key.startswith(prefix) else key
        if k.endswith(".lora_A.weight"):
            a_of[k[: -len(".lora_A.weight")]] = tensor
        elif k.endswith(".lora_B.weight"):
            b_of[k[: -len(".lora_B.weight")]] = tensor
        elif k.endswith(".lora_embedding_A"):  # an embedding's LoRA: A [r, vocab], B [dim, r];
            # PEFT folds it as (B @ A).T * scaling, which is (A.T as b) @ (B.T as a)
            b_of[k[: -len(".lora_embedding_A")]] = tensor.t()
        elif k.endswith(".lora_embedding_B"):
            a_of[k[: -len(".lora_embedding_B")]] = tensor.t()
        elif k.endswith(".bias"):
            plan.biases[k.replace(".base_layer.", ".")] = tensor
        elif k.endswith(".base_layer.weight") or k.endswith(_TABLE_COPIES):
            pass  # PEFT's copy of a frozen table it saves beside an adapter (save_embedding_layers)
        else:  # a trained tensor this export would drop: never write a model without it
            raise PorosExportError(
                f"{adapter_dir} holds {key!r}, which merged export does not fold",
                remediation="serve the adapter files in the run directory with PEFT instead")
    unpaired = sorted(set(a_of) ^ set(b_of))
    if unpaired:
        raise PorosExportError(f"{adapter_dir} holds half a LoRA pair for {', '.join(unpaired)}")
    plan.deltas = {f"{m}.weight": (a_of[m], b_of[m]) for m in a_of}
    return plan


def _native_plan(cfg: "PorosConfig", adapter_dir: Path, state: Dict[str, torch.Tensor],
                 skeleton: nn.Module) -> _Plan:
    from poros.train.adapters import NativeAdapterManager, validate_native_adapter_state
    from poros.train.model import _resolve_lora_target_modules

    # Inject exactly as PorosModel.from_pretrained does, then require the
    # saved state to be that trainable set, key for key and shape for shape.
    NativeAdapterManager(
        model=skeleton,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target_modules=_resolve_lora_target_modules(cfg.lora_target_modules),
        bias=cfg.lora_bias,
    ).inject()
    try:
        validate_native_adapter_state(skeleton, state, str(adapter_dir))
    except RuntimeError as exc:
        raise PorosExportError(str(exc)) from exc
    plan = _Plan({}, {}, float(cfg.lora_alpha) / float(cfg.lora_rank), "native")
    a_of, b_of = {}, {}
    for key, tensor in state.items():
        if key.endswith(".lora_A"):
            a_of[key[: -len(".lora_A")]] = tensor
        elif key.endswith(".lora_B"):
            b_of[key[: -len(".lora_B")]] = tensor
        elif key.endswith(".bias"):
            plan.biases[key.replace(".base_layer.", ".")] = tensor
        else:
            raise PorosExportError(f"unexpected trained tensor {key!r} in {adapter_dir}")
    plan.deltas = {f"{m}.weight": (a_of[m], b_of[m]) for m in a_of}
    return plan


def tied_table_keys(model: nn.Module) -> frozenset:
    """The weight keys of the input embedding and the LM head when the model ties them
    (``tie_word_embeddings``, on the config or its text config); empty when it does not."""
    config = getattr(model, "config", None)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    if not (getattr(config, "tie_word_embeddings", False)
            or getattr(text, "tie_word_embeddings", False)):
        return frozenset()
    names = {id(m): n for n, m in model.named_modules(remove_duplicate=False)}
    keys = set()
    for get in ("get_input_embeddings", "get_output_embeddings"):
        module = getattr(model, get, lambda: None)()
        if module is not None and id(module) in names:
            keys.add(f"{names[id(module)]}.weight")
    return frozenset(keys)


def _adapter_plan(cfg: "PorosConfig", adapter_dir: Path, skeleton: nn.Module) -> _Plan:
    st = adapter_dir / "adapter_model.safetensors"
    binary = adapter_dir / "adapter_model.bin"
    native = adapter_dir / "adapter_weights.pt"
    if st.is_file():
        from safetensors.torch import load_file

        return _peft_plan(adapter_dir, load_file(str(st)), skeleton)
    if binary.is_file():
        return _peft_plan(
            adapter_dir, torch.load(binary, map_location="cpu", weights_only=True), skeleton
        )
    if native.is_file():
        state = torch.load(native, map_location="cpu", weights_only=True)
        return _native_plan(cfg, adapter_dir, state, skeleton)
    raise PorosExportError(
        f"no adapter weights in {adapter_dir}",
        remediation="expected adapter_model.safetensors (PEFT) or adapter_weights.pt (native)",
    )


# ---------------------------------------------------------------------------
# The dense merge
# ---------------------------------------------------------------------------


@dataclass
class MergeReport:
    """What a merged export wrote."""

    output_dir: str
    nf4_roundtrip: Optional[str]  # "cuda" | "cpu" | None (no linear layer the load quantized)
    adapter_format: str  # "peft" | "native"
    merged: int  # weights with a LoRA delta folded in
    dequantized: int  # NF4 weights written as their round trip, without a delta
    biases: int  # trained biases written over the base's
    copied: int  # tensors copied byte for byte
    bytes_written: int


def merge_weight(
    seen: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float, dtype: torch.dtype
) -> torch.Tensor:
    """``seen + scaling * b @ a``, added in fp32 and rounded once to ``dtype``."""
    return (seen.float() + scaling * (b.float() @ a.float())).to(dtype)


def export_merged_dense(
    cfg: "PorosConfig",
    adapter_dir,
    output_dir: str,
    *,
    nf4_device: str = "auto",
) -> MergeReport:
    """Write the base of ``cfg`` with the adapter in ``adapter_dir`` merged, to ``output_dir``.

    ``cfg`` is the run's ``PorosConfig`` (``poros_config.json``): the base
    and its pinned revision, the dtype it computed in, and, for a native
    adapter, the LoRA settings it was injected with. ``nf4_device`` picks
    where the NF4 base is round-tripped (see :func:`nf4_roundtrip`).
    """
    adapter_dir = Path(adapter_dir)
    dtype = getattr(torch, cfg.torch_dtype)  # the dtype the run loaded its base in
    room_before_download(cfg.model_name_or_path, cfg.model_revision, output_dir)
    source = _resolve_checkpoint_dir(cfg.model_name_or_path, cfg.model_revision)
    if os.path.realpath(output_dir) == os.path.realpath(source):
        raise PorosExportError(f"the output directory {output_dir} is the base checkpoint itself")
    try:
        reader = CheckpointReader(source)
    except FileNotFoundError as exc:
        raise PorosExportError(
            f"{cfg.model_name_or_path} has no safetensors checkpoint",
            remediation="merged export reads safetensors shards (.bin checkpoints are not read)",
        ) from exc
    room_for(shard_bytes(reader), output_dir)  # before the skeleton and the adapter are built

    skeleton = _skeleton(source, dtype, cfg.trust_remote_code, cfg.attn_implementation)
    remap = _key_remapper(skeleton)  # before the adapter wraps the modules
    try:  # checkpoint name -> model key, as the training load maps it
        key_of = {name: remap(name) for name in reader.entries}
    except RuntimeError as exc:
        raise PorosExportError(f"cannot map the checkpoint onto the model: {exc}") from exc
    quantized = _nf4_quantized_weights(skeleton)
    plan = _adapter_plan(cfg, adapter_dir, skeleton)
    tied = sorted(tied_table_keys(skeleton) & (plan.deltas.keys() | plan.biases.keys()))
    if tied:
        raise PorosExportError(
            f"{cfg.model_name_or_path} ties its LM head to its embedding table "
            f"(tie_word_embeddings), and the adapter changes {tied[0]}: a merged checkpoint "
            "stores the one table, so its head would carry a delta training never applied to "
            "the head (or its embedding one it never applied to the embedding)",
            remediation="Export the adapter itself (poros export <run> --format adapter), or "
            "train without embed_tokens and lm_head in lora_target_modules.",
        )
    missing = sorted((plan.deltas.keys() | plan.biases.keys()) - set(key_of.values()))
    if missing:
        raise PorosExportError(
            f"{len(missing)} adapter tensor(s) have no weight in the base checkpoint "
            f"(e.g. {missing[0]!r})",
            remediation="check that the run's base model and revision are the ones it trained on",
        )
    roundtrip, device = nf4_roundtrip(dtype, nf4_device) if quantized else (None, None)

    counts = {"merged": 0, "dequantized": 0, "biases": 0, "copied": 0}
    done = set()

    def transform(name: str) -> Optional[torch.Tensor]:
        key = key_of[name]
        stored = _DTYPES.get(reader.entries[name].dtype)
        if key in plan.biases:
            done.add(key)
            counts["biases"] += 1
            return plan.biases[key].to(stored)
        delta = plan.deltas.get(key)
        if key not in quantized and delta is None:
            counts["copied"] += 1
            return None
        w = reader.get(name)
        seen = roundtrip(w) if key in quantized else w.to(dtype)
        if delta is None:
            counts["dequantized"] += 1
            return seen.to(w.dtype)
        a, b = delta
        if tuple(w.shape) != (b.shape[0], a.shape[1]):
            raise PorosExportError(
                f"{name} is {tuple(w.shape)} but its adapter is "
                f"{tuple(b.shape)} @ {tuple(a.shape)}: a different base model"
            )
        done.add(key)
        counts["merged"] += 1
        return merge_weight(seen, a, b, plan.scaling, w.dtype)

    from poros.moe.export import copy_base_files  # by name, and no link out of the base

    with staged_output(output_dir) as stage:
        written = write_rewritten(reader, str(stage), transform)
        if done != plan.deltas.keys() | plan.biases.keys():  # pragma: no cover - checked above
            raise RuntimeError("merged export skipped adapter tensors")
        copy_base_files(source, stage)
    if device == "cuda":
        torch.cuda.empty_cache()
    return MergeReport(
        output_dir=str(output_dir),
        nf4_roundtrip=device,
        adapter_format=plan.adapter_format,
        bytes_written=written,
        **counts,
    )
