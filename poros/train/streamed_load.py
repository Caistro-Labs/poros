"""The block-streamed NF4 load: bounded-GPU quantization at load.

The standard NF4 path materializes the ENTIRE quantized base on the GPU
before the block manager evicts it to pinned host memory, so a base larger
than the card cannot load that way.

This loader bounds GPU residency during load the same way training already
bounds it: each checkpoint tensor is staged to the GPU alone, quantized by
bitsandbytes there (NF4 quantization requires CUDA), and immediately moved
to CPU. Peak load VRAM is one tensor's bf16 source plus its quantized copy
instead of the whole base. Host RSS also drops: bf16 shards are released
per-shard instead of accumulating.

Correctness contract: NF4 quantization is per-tensor and deterministic, so
this loader must produce a base that is BYTE-IDENTICAL to the standard
loader's (packed 4-bit data, absmax/quant-state tensors, and every
unquantized parameter compared with ``torch.equal``). The gate lives in
``scripts/streamed_load_gate.py``.

``nf4_load_strategy="auto"`` (the default) takes this load on CUDA and falls back to the
standard loader only for a checkpoint it cannot stream (``StreamedLoadUnsupported``);
``"block_streamed"`` takes it with no fallback; ``"gpu"`` takes the standard loader.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import torch
import torch.nn as nn

from poros.core.checkpoints import shard_name
from poros.core.hub_files import HUB_FILES
from poros.sss.pinning import PinnedArena

logger = logging.getLogger(__name__)

_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_SINGLE = "model.safetensors"


class StreamedLoadUnsupported(RuntimeError):
    """An input the block-streamed load cannot make byte-identical to the standard load (no
    safetensors, a weight transform that is not a renaming, a checkpoint tensor the model does
    not define): the one failure ``nf4_load_strategy='auto'`` answers with the standard loader.
    Any other failure (out of memory, the Hub, a model left unmaterialized) is the run's own."""


class _NoSafetensors(StreamedLoadUnsupported, FileNotFoundError):
    """No safetensors checkpoint: unsupported here, and a missing file to any caller."""


def _resolve_checkpoint_dir(model_name_or_path: str,
                            revision: Optional[str] = None) -> str:
    """Return a local directory containing the model's safetensors shards."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    from poros.core.arch_registry import hub_token
    from poros.plan.configs import weights_on_hub

    # With every weight file already in the cache nothing downloads, and the Hub's "Fetching N
    # files" bar is a line about nothing: off for this call. A real download keeps its bars.
    held = weights_on_hub(model_name_or_path, revision)
    quiet = held is not None and held[1] == 0 and not are_progress_bars_disabled()
    if quiet:
        disable_progress_bars()
    try:
        return snapshot_download(model_name_or_path, revision=revision,
                                 allow_patterns=HUB_FILES, token=hub_token())
    finally:
        if quiet:
            enable_progress_bars()


def _shard_map(checkpoint_dir: str) -> Dict[str, List[str]]:
    """Map each shard file to the parameter names stored in it."""
    index_path = os.path.join(checkpoint_dir, _SAFETENSORS_INDEX)
    by_shard: Dict[str, List[str]] = defaultdict(list)
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map: Dict[str, str] = json.load(f)["weight_map"]
        for name, shard in weight_map.items():
            by_shard[shard_name(shard)].append(name)
        return dict(by_shard)
    single = os.path.join(checkpoint_dir, _SAFETENSORS_SINGLE)
    if not os.path.exists(single):
        raise _NoSafetensors(
            f"No safetensors checkpoint found in {checkpoint_dir} "
            "(block-streamed load supports safetensors checkpoints only)"
        )
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as f:
        names = list(f.keys())
    return {_SAFETENSORS_SINGLE: names}


def _locate(model: nn.Module, param_name: str) -> Tuple[nn.Module, str]:
    """Resolve ``a.b.c.weight`` to (module ``a.b.c``, attribute ``weight``)."""
    parent_path, _, attr = param_name.rpartition(".")
    module = model.get_submodule(parent_path) if parent_path else model
    return module, attr


def _key_remapper(model: nn.Module) -> Callable[[str], str]:
    """Checkpoint key -> model key, as the standard loader maps it.

    Some checkpoints store keys under legacy layouts (e.g. gemma-3's
    multimodal "language_model.model.*" prefixes) that the standard loader
    remaps via the model's registered weight transforms. Apply the same
    renaming transforms here; refuse outright if a non-renaming transform
    (tensor fusion/splitting) would match — this loader cannot perform
    those, and silently skipping them would corrupt the model. The
    byte-identity gate is the backstop either way.
    """
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    transforms = get_model_conversion_mapping(model)
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    complex_transforms = [
        t for t in transforms if not isinstance(t, WeightRenaming)
    ]

    def remap(name: str) -> str:
        for t in complex_transforms:
            if t._scoped_match(name) is not None:
                raise StreamedLoadUnsupported(
                    "block-streamed load cannot apply non-renaming weight "
                    f"transform {type(t).__name__} matching checkpoint key "
                    f"{name!r}; use the standard loader for this model"
                )
        # Renaming transforms CHAIN in registration order (each applies to
        # the running key, scoped by its scope_prefix) — e.g. gemma-3:
        # '^language_model' -> 'model.language_model', then within scope
        # 'model': '^language_model.model' -> 'language_model'. First-match
        # -return produces doubled prefixes (measured on gemma3_27b).
        key = name
        for t in renamings:
            key, _matched = t.rename_source_key(key)
        return key

    return remap


def _model_keys(model: nn.Module) -> FrozenSet[str]:
    """The checkpoint keys a load fills: the model's own state dict.

    transformers 5 (``convert_and_load_state_dict_in_model``) loads a
    checkpoint tensor only when its renamed key is in ``model.state_dict()``,
    every parameter and persistent buffer the class defines. Any other key is
    recorded as unexpected and never read: no parameter is added, no
    non-persistent buffer is overwritten. ``_keys_to_ignore_on_load_unexpected``
    only quiets the report. Both streamed loads leave the same keys.
    """
    return frozenset(model.state_dict().keys())


def _left_out(model: nn.Module, keys: List[str]) -> None:
    """Say which checkpoint keys were left unread, as transformers' load report
    does (``_adjust_missing_and_unexpected_keys``): quietly for the keys the
    class names in ``_keys_to_ignore_on_load_unexpected`` (pythia's legacy
    ``attention.bias``) and for the old ``rotary_emb.inv_freq`` and
    ``position_ids`` copies of buffers the model has, in one warning for the
    rest. A key is checkpoint text, so it is shown escaped and cut short."""
    if not keys:
        return
    patterns = set(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ())
    buffers = [name for name, _ in model.named_buffers()]
    if any(name.endswith("rotary_emb.inv_freq") for name in buffers):
        patterns.add(r"rotary_emb\.inv_freq")
    if any(name.endswith("position_ids") for name in buffers):
        patterns.add(r"(^|\.)position_ids$")
    quiet = re.compile("|".join(f"({p})" for p in sorted(patterns))) if patterns else None
    loud = [k for k in keys if quiet is None or quiet.search(k) is None]
    logger.debug("%d checkpoint tensors the model does not define were not loaded",
                 len(keys))
    if loud:
        shown = ", ".join(repr(k[:120]) for k in loud[:5])
        logger.warning(
            "%d checkpoint %s not in the model and left unread, as the standard "
            "load leaves them: %s%s",
            len(loud), "tensor is" if len(loud) == 1 else "tensors are", shown,
            ", ..." if len(loud) > 5 else "",
        )


def _assign(module: nn.Module, attr: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    """Install an unquantized checkpoint tensor (floating point cast to ``dtype``).

    Only where the class defines one: a parameter, or a buffer the state dict
    holds. A checkpoint never adds a tensor to the model (``_model_keys``).

    Always a private copy: safetensors returns mmap-backed views, and a
    retained view keeps its whole shard mapping resident — 47 retained
    shard maps measured as +35 GB host RSS at 72B. The shard unmaps at
    context exit.
    """
    is_buffer = (module._buffers.get(attr) is not None
                 and attr not in module._non_persistent_buffers_set)
    if not is_buffer and module._parameters.get(attr) is None:
        raise StreamedLoadUnsupported(
            f"{type(module).__name__} defines no parameter or persistent buffer "
            f"{attr[:120]!r}; a checkpoint tensor is never added to the model"
        )
    tensor = tensor.to(dtype, copy=True) if tensor.is_floating_point() else tensor.clone()
    if is_buffer:  # persistent buffer present in the checkpoint
        module._buffers[attr] = tensor
    else:
        module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)


def _load_shards(
    model: nn.Module,
    checkpoint_dir: str,
    dtype: torch.dtype,
    quantize: Optional[Callable[[nn.Module, str, torch.Tensor], bool]] = None,
) -> Tuple[int, int]:
    """Fill the skeleton ``model`` from its checkpoint, one shard at a time.

    ``quantize(module, attr, tensor)`` places a tensor itself and returns
    True (the 4-bit parameters); every other tensor is copied in with
    ``_assign``. A key the model does not define is never read
    (``_model_keys``). Everything but ``quantize`` runs on the CPU. Returns
    how many tensors were quantized and how many copied.
    """
    from safetensors import safe_open

    _remap = _key_remapper(model)
    defined = _model_keys(model)

    quantized = 0
    direct = 0
    left: List[str] = []
    for shard, names in sorted(_shard_map(checkpoint_dir).items()):
        shard_path = os.path.join(checkpoint_dir, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in sorted(names):
                model_key = _remap(name)
                if model_key not in defined:
                    left.append(model_key)
                    continue
                module, attr = _locate(model, model_key)
                tensor = f.get_tensor(name)
                if quantize is not None and quantize(module, attr, tensor):
                    quantized += 1
                else:
                    _assign(module, attr, tensor, dtype)
                    direct += 1
        torch.cuda.empty_cache()
    _left_out(model, left)
    return quantized, direct


def dtype_argument(config: object) -> str:
    """The name this transformers takes a load's dtype by: ``dtype``, where its configs have
    one; ``torch_dtype`` is the older name for the same argument, and a load that passes it
    says "`torch_dtype` is deprecated" on every run."""
    return "dtype" if hasattr(config, "dtype") else "torch_dtype"


def load_nf4_block_streamed(
    model_name_or_path: str,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    revision: Optional[str] = None,
    resident: Optional[Tuple[int, int]] = None,
) -> nn.Module:
    """Load an NF4-quantized causal LM with bounded GPU residency.

    Produces a CPU-resident quantized model numerically identical to the
    standard ``from_pretrained(quantization_config=...)`` GPU load (see the
    byte-identity gate). Downstream (adapter attach, block manager pinning)
    is unchanged: the block manager already accepts CPU-resident bases.
    ``resident=(block_size, n)``: the 4-bit weights of the layers in the run's
    ``n`` resident blocks (``poros.core.residency.resident_layer_names``) stay
    on the GPU where they were quantized, so they never take host RAM.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "block-streamed NF4 load requires CUDA (bitsandbytes quantizes "
            "on the GPU); for CPU-only validation use the standard path"
        )

    import bitsandbytes as bnb
    from accelerate import init_empty_weights
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from transformers.integrations import replace_with_bnb_linear
    from transformers.quantizers.base import HfQuantizer

    device = torch.device("cuda:0")
    checkpoint_dir = _resolve_checkpoint_dir(model_name_or_path, revision)
    from poros.moe.carried import ensure_registered

    ensure_registered(checkpoint_dir)  # a family Poros carries registers itself first
    config = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=trust_remote_code
    )
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Skeleton: params on meta (no memory), buffers REAL (computed on CPU in
    # __init__ — rotary tables etc. are not in checkpoints and must survive).
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code, **{dtype_argument(config): dtype}
        )
    # Same not-convert set as the standard loader (lm_head, keep-in-fp32
    # modules, config skips) — required for byte identity. This mirrors
    # Bnb4BitHfQuantizer._process_model_before_weight_loading exactly.
    modules_to_not_convert = HfQuantizer.get_modules_to_not_convert(
        model,
        bnb_config.llm_int8_skip_modules,
        getattr(model, "_keep_in_fp32_modules", None),
    )
    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=bnb_config,
    )

    on_device = set()
    if resident and resident[1] > 0:
        from poros.core.residency import resident_layer_names

        for name in resident_layer_names(model, *resident):
            on_device.update(id(m) for m in model.get_submodule(name).modules())
    # what the host will hold: each 4-bit weight packs two values a byte
    # release_on_free: the model keeps these pages for its whole life, and gives them (and their
    # grant in the process's pin budget) back when it is freed
    arena = PinnedArena(sum((p.numel() + 1) // 2 for m in model.modules() if id(m) not in on_device
                            for p in m.parameters(recurse=False)
                            if isinstance(p, bnb.nn.Params4bit)), release_on_free=True)

    def quantize(module: nn.Module, attr: str, tensor: torch.Tensor) -> bool:
        current = getattr(module, attr, None)
        if not isinstance(current, bnb.nn.Params4bit):
            return False
        # Stage one tensor on GPU, quantize, evict. Mirrors the standard
        # loader's Bnb4bitQuantize op verbatim:
        # Params4bit(value, requires_grad=False,
        #            **skeleton.__dict__).to(device) — the skeleton attrs
        # carry the exact quant configuration, and .to(cuda) performs the
        # NF4 quantization. .cpu() then moves packed data + quant state off
        # the GPU.
        value = tensor.to(device=device, dtype=dtype)
        staged = bnb.nn.Params4bit(
            value, requires_grad=False, **current.__dict__
        ).to(device)
        if id(module) in on_device:  # a resident block's: its only copy stays here
            module._parameters[attr] = staged
            del value
            return True
        qparam = staged.cpu()
        # Match the standard path's steady state: packed data on CPU (the
        # block manager streams it per block), quant state GPU-resident. The
        # block manager only moves ``.data``; dequantize_4bit reads
        # quant-state tensors where they live, and CPU-resident state
        # against GPU packed data is an illegal access (measured: 0.5B and
        # 72B). Quant state is ~1/64 of packed size (<1 GB at 72B),
        # identical to the standard path's footprint.
        if qparam.quant_state is not None:
            qparam.quant_state.to(device)
        # Pin the packed data so the block manager adopts a pinned store
        # (async H2D stays truly async), carved from shared slabs: pinned one
        # by one, torch would lock each a power of two (up to twice its bytes).
        qparam.data = arena.pin(qparam.data)
        module._parameters[attr] = qparam
        del staged, value
        return True

    quantized, direct = _load_shards(model, checkpoint_dir, dtype, quantize)

    # Tied weights (e.g. lm_head <- embed_tokens) resolve meta leftovers.
    model.tie_weights()

    # Match the standard path's steady state for BUFFERS: a whole-model GPU
    # load leaves every buffer on the GPU, and the block manager never moves
    # buffers (parameters only). Some architectures device-cast buffers in
    # forward (rope inv_freq), but not all — gemma4's layer_scalar multiplies
    # raw and crashes against a CPU buffer (measured during validation).
    # Buffers are tiny; keep them GPU-resident.
    for mod in model.modules():
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and buf.device.type != device.type:
                mod._buffers[bname] = buf.to(device)

    remaining = [
        n for n, p in model.named_parameters() if p.is_meta
    ]
    if remaining:
        raise RuntimeError(
            "block-streamed load left parameters unmaterialized (likely a "
            f"checkpoint/model mismatch): {remaining[:5]}"
        )

    # Flags the rest of the stack keys on (PEFT kbit prep, registry probes).
    model.is_loaded_in_4bit = True
    model.config.quantization_config = bnb_config
    model.eval()

    logger.info(
        "block-streamed NF4 load complete: %d tensors quantized via a "
        "single-tensor GPU window, %d loaded directly to CPU",
        quantized, direct,
    )
    return model
