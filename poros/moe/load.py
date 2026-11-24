"""Streamed checkpoint load for ``poros.moe``: experts quantized as they are read.

Loading a sparse MoE (``poros.moe.arch.MOE_FAMILIES``; Qwen3-MoE quoted
below) with ``from_pretrained`` before ``prepare_moe`` holds
the whole bf16 model in host RAM (about 61 GB at 30B-A3B) and then the
quantized copy of its experts on top. This loader never materializes the
fused expert tensors. It walks the safetensors shards one at a time, the
way ``poros.train.streamed_load`` does, and:

- quantizes each expert's tensor with ``poros.moe.quant`` straight into its
  row of a preallocated ``ExpertSlabArena``, then drops it: on the GPU when
  CUDA is available (``quantize_on="auto"``, the default; one tensor on the
  device at a time), else on the CPU, the same bytes either way;
- loads every other tensor (attention, norms, router, embeddings) into a
  ``meta`` skeleton in ``dtype``, as the dense loader does.

Host RAM peaks near the NF4 experts plus the dense weights plus one
shard. HF checkpoints store ``gate_proj``
and ``up_proj`` (Mixtral: ``w1`` and ``w3``) per expert; transformers fuses
them into ``gate_up_proj`` through its conversion mapping, and this loader
reproduces that fusion
(gate rows, then up rows), refusing any conversion it does not know.

Correctness contract (as for the dense loader): every slab, quantization
constant and non-expert parameter is byte-identical to ``from_pretrained``
followed by ``prepare_moe``'s quantization (``tests/moe/test_load.py``).
"""

from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path

import torch
import torch.nn as nn

from poros.moe.arch import MoeArch
from poros.moe.contracts import ExpertKey
from poros.sss.source import SlabSource
from poros.train.streamed_load import (
    _assign,
    _key_remapper,
    _left_out,
    _locate,
    _model_keys,
    _resolve_checkpoint_dir,
    _shard_map,
)

logger = logging.getLogger(__name__)

__all__ = ["load_moe_streamed", "sss_file_path", "sss_meta"]

# One expert's tensor in an HF MoE checkpoint: its experts module, index and
# projection.
_EXPERT_KEY = re.compile(r"^(.+\.experts)\.(\d+)\.(\w+)\.weight$")
# One layer's experts stored as the model holds them: its experts module and fused tensor.
_FUSED_KEY = re.compile(r"^(.+\.experts)\.(gate_up_proj|down_proj)$")


def _fusion(prefix: str, gate: str, up: str, down: str) -> tuple[frozenset, dict[str, str]]:
    """One expert conversion as ``_check_fusion`` reads it, and each projection's role."""
    conversion = frozenset(
        {
            (
                (f"{prefix}.*.{gate}.weight", f"{prefix}.*.{up}.weight"),
                (f"{prefix}.gate_up_proj",),
                (("MergeModulelist", 0), ("Concatenate", 1)),
            ),
            ((f"{prefix}.*.{down}.weight",), (f"{prefix}.down_proj",), (("MergeModulelist", 0),)),
        }
    )
    return conversion, {gate: "gate", up: "up", down: "down"}


# transformers' expert conversion mappings, each with the checkpoint name of
# every expert projection: per-expert gate/up stacked, then concatenated
# along the output dim into gate_up_proj; down stacked into down_proj. The
# "qwen2_moe" entry of conversion_mapping.py serves qwen3_moe, qwen2_moe and
# olmoe; "mixtral" names them w1/w3/w2. This loader performs exactly these
# fusions, so it refuses any other.
_FUSIONS = dict(
    [
        _fusion("mlp.experts", "gate_proj", "up_proj", "down_proj"),
        _fusion(".experts", "w1", "w3", "w2"),
    ]
)


def _check_fusion(model: nn.Module) -> dict[str, str]:
    """Each expert projection's role, or refuse a conversion we do not reproduce."""
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    found = frozenset(
        (
            tuple(t._original_source_patterns),
            tuple(t._original_target_patterns),
            tuple((type(op).__name__, getattr(op, "dim", None)) for op in t.operations),
        )
        for t in get_model_conversion_mapping(model)
        if not isinstance(t, WeightRenaming)
    )
    if found not in _FUSIONS:
        raise RuntimeError(
            "this transformers version converts the MoE expert weights as "
            f"{sorted(found)}, not a fusion the streamed MoE loader reproduces; "
            "load with from_pretrained instead"
        )
    return _FUSIONS[found]


def sss_file_path(store_path: str | os.PathLike, model_name_or_path: str) -> Path:
    """The ``.sss`` file of a model under ``store_path`` (a directory, or the file itself)."""
    path = Path(store_path)
    if path.suffix == ".sss":
        return path
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_name_or_path.strip("/")).strip("-")
    return path / f"{slug or 'model'}.moe.sss"


def _checkpoint_identity(checkpoint_dir: str, revision: str | None) -> dict:
    """The checkpoint's revision and a fingerprint of its weight files."""
    resolved = os.path.realpath(checkpoint_dir)
    sha = os.path.basename(resolved)
    hub = os.path.basename(os.path.dirname(resolved)) == "snapshots" and bool(
        re.fullmatch(r"[0-9a-f]{40}", sha)
    )
    files = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            st = os.stat(os.path.join(checkpoint_dir, name))
            # A Hub snapshot is content-addressed by its sha; a local directory
            # is identified by its files' sizes and modification times.
            files.append([name, st.st_size] if hub else [name, st.st_size, st.st_mtime_ns])
    return {"revision": sha if hub else revision, "files": files}


def sss_meta(
    model_name_or_path: str,
    checkpoint_dir: str,
    revision: str | None,
    arch: MoeArch,
    tensor_shapes: dict,
    dtype: torch.dtype,
) -> dict:
    """The reuse key of a model's expert ``.sss`` file: any difference means rebuild."""
    from poros import __version__
    from poros.moe import quant

    layout = arch.layout
    return {
        "kind": "poros.moe.expert_slabs",
        "model_id": model_name_or_path,
        "checkpoint": _checkpoint_identity(checkpoint_dir, revision),
        "layout": {
            "num_layers": layout.num_layers,
            "experts_per_layer": layout.experts_per_layer,
            "top_k": layout.top_k,
            "tensor_names": list(layout.tensor_names),
            "tensor_shapes": {k: list(v) for k, v in tensor_shapes.items()},
            "moe_layer_indices": list(arch.moe_layer_indices),
        },
        "quant": {
            "kind": "nf4",
            "blocksize": quant.NF4_BLOCKSIZE,
            "double_quant_blocksize": quant.DOUBLE_QUANT_BLOCKSIZE,
            "source_dtype": str(dtype),
        },
        "versions": {"poros": __version__, "torch": torch.__version__.split("+")[0]},
    }


def _reuse(path: Path, meta: dict, store: str, ssd_options: dict) -> SlabSource | None:
    """The source over a valid ``.sss`` at ``path`` for ``meta``, else None (rebuild)."""
    from poros.sss import RamSource, SsdSource, SssFile, SssFormatError

    try:
        with SssFile.open(path) as f:
            if not f.matches(meta):
                logger.info("SSS %s was built for another model/config; rebuilding", path)
                return None
    except (SssFormatError, OSError):
        return None
    if store == "ssd":
        return SsdSource(path, **ssd_options)
    return RamSource.from_sss(path)


def load_moe_streamed(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    *,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    revision: str | None = None,
    store: str = "ram",
    store_path: str | os.PathLike | None = None,
    ssd_options: dict | None = None,
    store_options: dict | None = None,
    quantize_on: str | None = "auto",
    device: str | None = None,
    build_path: str | os.PathLike | None = None,
    expert_format: str | None = None,
) -> nn.Module:
    """Load a sparse MoE checkpoint for ``prepare_moe`` with bounded host RAM.

    Returns the model on CPU with its expert weights already quantized: the
    fused expert parameters stay on ``meta`` and the store rides on the
    model for ``prepare_moe`` to adopt, so pass the model to ``prepare_moe``
    before moving it. Everything else is loaded in ``dtype``.

    ``store`` says where the quantized expert slabs live:

    - ``"ram"`` (default): pinned host RAM, as always. With ``store_path``,
      a valid ``.sss`` there fills RAM instead of requantizing (nothing is
      written when there is none).
    - ``"ssd"``: a ``.sss`` file under ``store_path`` streamed by
      ``poros.sss.SsdSource`` (``ssd_options`` are its keyword arguments).
      The first run quantizes straight into the file (crash-safe commit);
      later runs reuse it while ``sss_meta`` matches (same checkpoint
      revision and files, layout, quantization, poros and torch versions)
      and requantize otherwise. Host RAM then holds no expert slab.
    - any other name: a store an installed package provides (``poros.stores``);
      its plugin loads the model (``store_path`` and ``store_options`` are its
      own) and may name the ``MoeBackends`` ``prepare_moe`` should use.

    ``quantize_on`` says where each expert's NF4 level is computed: ``"auto"``
    (default) the training ``device`` where it is a CUDA device, else the
    current GPU when CUDA is available, else the CPU (``POROS_QUANTIZE_ON``
    overrides ``"auto"`` only); ``"cpu"``; or a device. The bytes are the same
    either way (``quant.quantize_tensor``); one tensor is on the device at a
    time, and a device failure (out of memory, a CUDA error) quantizes the rest
    on the CPU.

    ``build_path`` (``store="ssd"``) makes ``store_path`` read only: a valid
    ``.sss`` there is reused, and with none the store is built (or reused)
    under ``build_path`` instead, said in one line, and nothing is written at
    ``store_path``. ``model._poros_store_path`` is then ``build_path``. A run
    folder's recorded ``store_path`` comes this way
    (``poros.stores.recorded_store``).

    ``expert_format`` (Inkling 2 Mini): ``released`` or ``nf4``, how its experts are held (the
    run's ``moe_expert_format``); None reads ``POROS_INKLING_EXPERTS``, else the checkpoint
    decides (``poros.moe.inkling2.formats.decide``: released where its experts are quantized,
    nf4 where they are not); ``model._poros_expert_format`` is what it held.
    """
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    from poros.moe.store import ExpertSlabArena, MoeExpertStore
    from poros.stores import get_store_plugin, is_builtin

    if not is_builtin(store):
        plugin = get_store_plugin(store)
        model = plugin.load_model(
            model_name_or_path,
            dtype,
            store_path=store_path,
            options=dict(store_options or {}),
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            revision=revision,
        )
        if getattr(model, "_poros_moe_store", None) is None:
            raise RuntimeError(f"store plugin {store!r} returned a model without _poros_moe_store")
        if plugin.backends is not None:
            model._poros_moe_backends = plugin.backends(model._poros_moe_store)
        model._poros_store_plugin = store
        return model
    if store == "ssd" and store_path is None:
        raise ValueError("store='ssd' needs store_path (a directory on a local SSD)")
    ssd_options = dict(ssd_options or {})
    checkpoint_dir = _resolve_checkpoint_dir(model_name_or_path, revision)
    visits = _is_inkling_checkpoint(checkpoint_dir)
    from poros.moe.carried import ensure_registered

    ensure_registered(checkpoint_dir)  # a family Poros carries (Inkling 2 Mini) registers itself
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=trust_remote_code)
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    arch = MoeArch.from_config(config)
    # Parameters on meta (no memory); buffers real (rotary tables are not in
    # checkpoints).
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config, dtype=dtype, trust_remote_code=trust_remote_code
        )
    roles = functools.cache(lambda: _check_fusion(model))  # a checkpoint of fused experts asks not
    remap = (lambda name: name) if visits else _key_remapper(model)
    defined = _model_keys(model)  # what from_pretrained fills; any other key is left unread
    left: list[str] = []

    ordinal = {arch.experts_module(layer): i for i, layer in enumerate(arch.moe_layer_indices)}
    tensor_shapes = {
        name: tuple(
            model.get_parameter(arch.expert_tensor(arch.moe_layer_indices[0], name)).shape[1:]
        )
        for name in arch.layout.tensor_names
    }
    released = None  # Inkling 2 Mini: each MoE layer's released expert format, when kept
    if visits:
        from poros.moe.inkling2 import formats
        from poros.moe.inkling2.load import released_formats

        # the one decision (poros.moe.inkling2.formats): the run's format, checked against what
        # the checkpoint stores, never replaced silently
        names = [n for ns in _shard_map(checkpoint_dir).values() for n in ns]
        expert_format = formats.decide(formats.checkpoint_state(names),
                                       expert_format or formats.asked_format(None))
        if expert_format == formats.RELEASED:
            released = released_formats(_shard_map(checkpoint_dir), ordinal)
    source: SlabSource | None = None
    sss_path: Path | None = None
    reused = False
    sink = None
    if store_path is not None:
        sss_path = sss_file_path(store_path, model_name_or_path)
        meta = sss_meta(model_name_or_path, checkpoint_dir, revision, arch, tensor_shapes, dtype)
        if released is not None:  # the released bytes, not NF4: another store
            meta["quant"] = {"kind": "released", "formats": list(released)}
        source = _reuse(sss_path, meta, store, ssd_options)
        if source is None and store == "ssd" and build_path is not None:
            from poros.stores import moved_store_note

            logger.warning(moved_store_note(store_path, build_path))
            sss_path = sss_file_path(build_path, model_name_or_path)
            source = _reuse(sss_path, meta, store, ssd_options)
            model._poros_store_path = str(build_path)
        reused = source is not None
        if source is None and store == "ssd":
            sink = _ssd_sink(sss_path, meta)
    arena = None
    quant_device = None
    if source is None:
        quant_device = _quantize_device(quantize_on, device)
        if released is not None:
            from poros.moe.inkling2.released import ReleasedSlabArena

            arena = ReleasedSlabArena(arch.layout, tensor_shapes, released, sink=sink)
        else:
            arena = ExpertSlabArena(arch.layout, tensor_shapes, sink=sink,
                                    quant_device=quant_device)
    halves: dict[ExpertKey, dict[str, torch.Tensor]] = {}  # gate/up awaiting a partner
    try:
        shard_names = _shard_map(checkpoint_dir)
        if visits:  # its own names and expert formats (poros.moe.inkling2.load)
            from poros.moe.inkling2.load import fill

            left = fill(model, checkpoint_dir, shard_names, dtype=dtype, ordinal=ordinal,
                        arena=arena, assign=_assign, locate=_locate, released=released)
            shard_names = {}
        for shard, names in sorted(shard_names.items()):
            with safe_open(os.path.join(checkpoint_dir, shard), framework="pt", device="cpu") as f:
                for name in sorted(names):
                    fused = _FUSED_KEY.match(name)
                    if fused is not None and remap(fused[1]) in ordinal:
                        if arena is None:  # reused from the .sss file: never read
                            continue
                        whole = f.get_slice(name)
                        for index in range(whole.get_shape()[0]):
                            key = (ordinal[remap(fused[1])], index)
                            arena.write(key, fused[2], whole[index : index + 1][0].to(dtype))
                        continue
                    expert = _EXPERT_KEY.match(name)
                    if expert is not None:
                        experts_module, index, proj = expert.groups()
                        experts_module = remap(experts_module)  # e.g. mixtral's block_sparse_moe
                        if experts_module not in ordinal or proj not in roles():
                            raise RuntimeError(
                                f"checkpoint tensor {name!r} is not a known MoE expert"
                            )
                        if arena is None:  # reused from the .sss file: never read
                            continue
                        key = (ordinal[experts_module], int(index))
                        tensor = f.get_tensor(name)
                        if roles()[proj] == "down":
                            arena.write(key, "down_proj", tensor.to(dtype))
                            continue
                        # copy=True: a held mmap view would keep its shard mapped.
                        pair = halves.setdefault(key, {})
                        pair[roles()[proj]] = tensor.to(dtype, copy=True)
                        if len(pair) == 2:
                            gate_up = torch.cat([pair["gate"], pair["up"]])
                            arena.write(key, "gate_up_proj", gate_up)
                            del halves[key]
                        continue
                    model_key = remap(name)
                    if model_key not in defined:  # never read, as from_pretrained leaves it
                        left.append(model_key)
                        continue
                    module, attr = _locate(model, model_key)
                    _assign(module, attr, f.get_tensor(name), dtype)
        if halves:
            raise RuntimeError(
                f"checkpoint lacks the gate or up half of experts {sorted(halves)[:5]}"
            )
        _left_out(model, left)
        model.tie_weights()
        missing = [
            n
            for n, p in model.named_parameters()
            if p.is_meta and arch.classify_parameter(n) != "expert"
        ]
        if missing:
            raise RuntimeError(f"checkpoint does not cover parameters {missing[:5]}")
        if arena is not None:
            built = arena.build(dtype)
            arena = None
            if store == "ssd":  # committed: stream it
                built.source.close()
                from poros.sss import SsdSource

                source = SsdSource(sss_path, **ssd_options)
            else:
                model._poros_moe_store = built
    except BaseException:
        if arena is not None:
            arena.abort()
        if source is not None and hasattr(source, "close"):
            source.close()
        raise
    if source is not None and released is not None:
        from poros.moe.inkling2.released import ReleasedExpertStore

        model._poros_moe_store = ReleasedExpertStore.over_released(
            source, arch.layout, tensor_shapes, released, dtype)
    elif source is not None:
        model._poros_moe_store = MoeExpertStore.over_source(
            source, arch.layout, tensor_shapes, dtype
        )
    if visits:  # what the experts are held as: the run's record says the same
        model._poros_expert_format = expert_format
    model.eval()
    logger.info(
        "streamed MoE load: %d expert slabs %s (store=%s%s)",
        len(arch.expert_keys()),
        "reused" if reused else f"quantized on {quant_device or 'the CPU'}",
        store,
        f", {sss_path}" if sss_path is not None else "",
    )
    return model


def _is_inkling_checkpoint(checkpoint_dir: str) -> bool:
    """Whether the checkpoint's config.json names Inkling 2 Mini (read before transformers'
    AutoConfig, which knows it only once Poros registers its classes)."""
    import json

    path = os.path.join(checkpoint_dir, "config.json")
    try:
        with open(path) as f:
            model_type = json.load(f).get("model_type")
    except (OSError, ValueError):
        return False
    return model_type in ("inkling2_loop", "inkling2_loop_text")


def declares_visits(checkpoint_dir: str) -> bool:
    """Whether the decoder a checkpoint loads as runs its layers as visits (``poros_visits`` on
    its model class: a looped decoder, Inkling 2 Mini's), read from its config.json alone."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from poros.moe.carried import ensure_registered

    ensure_registered(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir)
    model_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
    return bool(getattr(model_class, "poros_visits", False))


def _quantize_device(quantize_on: str | None, device: str | None = None) -> str | None:
    """The device ``quant.quantize_tensor`` runs the NF4 level on (None: the CPU): ``"auto"``
    (``POROS_QUANTIZE_ON`` overrides it) takes the training ``device`` where it is CUDA."""
    want = quantize_on or "cpu"
    if want == "auto":
        want = os.environ.get("POROS_QUANTIZE_ON") or "auto"
    if want == "auto":
        if not torch.cuda.is_available():
            return None
        return device if device is not None and torch.device(device).type == "cuda" else "cuda"
    if want == "cpu":
        return None
    if torch.device(want).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"quantize_on={want!r} but CUDA is not available")
    return want


def _ssd_sink(path: Path, meta: dict):
    """A factory for the ``SssWriter`` the arena fills (checks free disk first)."""
    import shutil

    from poros.sss import SssWriter
    from poros.sss.format import ALIGNMENT, HEADER_NBYTES

    def factory(slabs):
        need = HEADER_NBYTES + sum(-(-n // ALIGNMENT) * ALIGNMENT for _, n in slabs)
        path.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path.parent).free
        if free < need:  # the preflight's own check (store_room) said it fit: the disk changed
            from poros.core.errors import PorosDiskError

            raise PorosDiskError(
                f"store_path {path.parent} has {free / 2**30:.1f} GiB free; the expert "
                f"store needs {need / 2**30:.1f} GiB",
                remediation="Free some space there, or name another store_path.")
        return SssWriter(path, slabs, meta=meta)

    return factory
