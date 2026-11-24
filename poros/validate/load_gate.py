"""The LOAD gate: the streamed NF4 load is byte-identical to the standard load.

The same model loads twice, once by the standard GPU load and once block by block, and every tensor
is compared: the packed 4-bit data, every quant-state tensor (absmax, nested absmax, quant maps,
offsets), and every unquantized parameter and persistent buffer. NF4 quantization is per tensor and
deterministic, so the two must be byte-identical; any difference is a bug in the streamed loader.

``digest_state`` holds a sha256 per tensor instead of the tensor, so one model is resident at a
time (two full copies of a 72B do not fit common host RAM).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Optional


def _params4bit():
    try:
        import bitsandbytes as bnb
    except ImportError:  # no 4-bit parameter can exist without it
        return None
    return bnb.nn.Params4bit


def _int_view(t):
    """A same-itemsize integer dtype numpy supports: a pure byte reinterpret, so the digest is over
    the exact stored bytes whatever the dtype (``.numpy()`` refuses bfloat16)."""
    import torch

    return {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]


def digest(t) -> str:
    """sha256 of a tensor's dtype, shape and raw bytes."""
    t = t.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.flatten().view(_int_view(t)).numpy().tobytes())
    return h.hexdigest()


def _state(model, keep: Callable[[Any], Any]) -> dict[str, Any]:
    import torch

    four_bit = _params4bit()
    state: dict[str, Any] = {}
    for name, param in model.named_parameters():
        state[f"param::{name}"] = keep(param)
        if four_bit is not None and isinstance(param, four_bit) and param.quant_state is not None:
            qs = param.quant_state.as_dict(packed=True)
            for key, value in sorted(qs.items()):
                if isinstance(value, torch.Tensor):
                    state[f"qstate::{name}::{key}"] = keep(value)
                else:
                    state[f"qmeta::{name}::{key}"] = value
    for name, buf in model.named_buffers():
        state[f"buffer::{name}"] = keep(buf)
    return state


def digest_state(model) -> dict[str, Any]:
    """name -> sha256 of every parameter, quant-state tensor and buffer (``repr`` of quant
    metadata). Digest equality over raw bytes is byte identity, up to a hash collision."""
    state = _state(model, digest)
    return {k: repr(v) if k.startswith("qmeta::") else v for k, v in state.items()}


def tensor_state(model) -> dict[str, Any]:
    """name -> a CPU copy of every parameter, quant-state tensor and buffer."""
    return _state(model, lambda t: t.detach().to("cpu", copy=True))


def compare(ref: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    """What differs between two states, digests or tensors: missing, extra and mismatched names,
    and how many tensors were compared."""
    import torch

    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    mismatched, compared = [], 0
    for key in sorted(set(ref) & set(got)):
        a, b = ref[key], got[key]
        if isinstance(a, torch.Tensor) != isinstance(b, torch.Tensor):
            mismatched.append(key)
            continue
        if isinstance(a, torch.Tensor):
            compared += 1
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                mismatched.append(key)
        else:
            compared += key.startswith(("param::", "qstate::", "buffer::"))
            if a != b:
                mismatched.append(key)
    return {"compared": compared, "missing": missing, "extra": extra, "mismatched": mismatched}


def run(model_id: str, dtype: str = "bfloat16", *, hash_compare: bool = True,
        revision: Optional[str] = None) -> dict[str, Any]:
    """Both loads of ``model_id``, one after the other on this GPU, and the record the gate writes.
    Never with a repo's own code: ``_load_model`` defaults ``trust_remote_code`` to False."""
    import torch

    from poros.train.model import _load_model

    torch_dtype = getattr(torch, dtype)
    state_fn = digest_state if hash_compare else tensor_state

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    standard = _load_model(model_id, torch_dtype, revision=revision).to("cpu")
    standard_peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    ref = state_fn(standard)
    del standard
    torch.cuda.empty_cache()
    t1 = time.time()

    torch.cuda.reset_peak_memory_stats()
    streamed = _load_model(model_id, torch_dtype, revision=revision,
                           nf4_load_strategy="block_streamed")
    streamed_peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    got = state_fn(streamed)
    del streamed
    t2 = time.time()
    torch.cuda.empty_cache()

    found = compare(ref, got)
    missing, extra, mismatched = found["missing"], found["extra"], found["mismatched"]
    return {
        "model": model_id,
        "dtype": dtype,
        "compare_mode": "sha256-digest" if hash_compare else "torch.equal",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tensors_compared": found["compared"],
        "missing_in_streamed": missing[:10],
        "extra_in_streamed": extra[:10],
        "mismatched": mismatched[:10],
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_mismatched": len(mismatched),
        "byte_identical": not (missing or extra or mismatched),
        "standard_load_peak_vram_gb": standard_peak,
        "streamed_load_peak_vram_gb": streamed_peak,
        "standard_load_seconds": round(t1 - t0, 1),
        "streamed_load_seconds": round(t2 - t1, 1),
    }
