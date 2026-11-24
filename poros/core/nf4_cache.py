"""Reuse dequantized NF4 weights across the G microsteps of one block visit.

Under NF4 block-major accumulation a block is loaded once for the forward of
all G microsteps and once for their backward. Stock bitsandbytes still
dequantizes every frozen ``Linear4bit`` weight on each use: the forward of each
microstep, its rematerialization, and the grad-input matmul of its backward --
3G dequantizations of the same packed bytes per update. On Qwen2.5-72B
(G=8, RTX 3090) that is ~12k ``kDequantizeBlockwise`` launches and ~3.9 s of
GPU time per update (a profile of one update).

``DequantCache.scope()`` swaps ``bitsandbytes.matmul_4bit`` for a version
that keeps each weight's dequantized copy for the scope (one block visit) and
otherwise mirrors bitsandbytes 0.50 op for op:

* the per-call choice between bnb's fused ``gemm_4bit`` kernel and
  dequantize + ``F.linear`` is bnb's own heuristic, evaluated with bnb's own
  function on the same (M, N, K); a call bnb would fuse is still fused;
* where bnb would dequantize then ``F.linear`` (``_dequant_linear_fallback``),
  the cached tensor is the same dequantization (same impls, same nested-absmax
  reconstruction, same output dtype), so ``F.linear`` sees the same bytes;
* the backward is ``MatMul4Bit.backward``'s ``torch.matmul(grad_output,
  dequantize_4bit(B, state).to(grad_output.dtype))`` with that dequantized
  weight reused.

The forward-fallback and backward dequantizations are two recipes; the cache
shares one tensor between them only after checking, on the first weight it
sees in the process, that the two recipes give equal bytes (otherwise it keeps
them under separate keys). Anything outside the mirrored cases -- a different
bitsandbytes, ROCm, CPU tensors, ``out=``, the deprecated [K, N] orientation,
empty inputs, fp32 compute -- goes to the original function untouched.

The cache is a pure speed/VRAM trade: bounded by ``max_bytes``; a weight that
does not fit is dequantized per call exactly as before.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SUPPORTED_BNB = ("0.50.",)

# The cache whose scope is open (one block visit), for dequantized_weight().
_ACTIVE: Optional["DequantCache"] = None


def _bnb_parts() -> Optional[Dict[str, Any]]:
    """The bitsandbytes internals the mirror needs, or None if not the tested shape."""
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnbF
        from bitsandbytes.backends.cuda import ops as cuda_ops
    except Exception:  # noqa: BLE001 -- any import problem means "do not patch"
        return None
    version = getattr(bnb, "__version__", "")
    if not version.startswith(_SUPPORTED_BNB) or torch.version.hip is not None:
        return None
    needed = (
        "_gemm_4bit_use_custom_fn",
        "_gemm_4bit_custom_max_m",
        "_dequantize_4bit_impl",
        "_dequantize_blockwise_impl",
    )
    if not all(hasattr(cuda_ops, n) for n in needed):
        return None
    return {"bnb": bnb, "F": bnbF, "ops": cuda_ops, "version": version}


class DequantCache:
    """Per-block-visit cache of dequantized NF4 weights (see module docstring)."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self._parts = _bnb_parts()
        self._store: Dict[Tuple[int, int, str, torch.dtype], torch.Tensor] = {}
        self._bytes = 0
        self._active = False
        self._orig = None
        # fwd/bwd recipes give equal bytes? decided once per (compute dtype, stored dtype)
        self._share: Dict[Tuple[torch.dtype, Any], bool] = {}
        self.stats = {"hits": 0, "misses": 0, "skipped_full": 0, "fused": 0, "passthrough": 0}
        if self._parts is None:
            logger.warning(
                "nf4_dequant_cache: bitsandbytes %s not in the mirrored set %s (or ROCm); "
                "the cache stays off and every call goes to stock bitsandbytes.",
                _bnb_version(), _SUPPORTED_BNB,
            )

    @property
    def enabled(self) -> bool:
        return self._parts is not None and self.max_bytes > 0

    # ---- scope -------------------------------------------------------------

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        """Patch ``bitsandbytes.matmul_4bit`` for one block visit; drop the cache after."""
        if not self.enabled or self._active:
            yield
            return
        global _ACTIVE
        bnb = self._parts["bnb"]
        self._orig = bnb.matmul_4bit
        bnb.matmul_4bit = self._matmul_4bit
        self._active = True
        _ACTIVE = self
        try:
            yield
        finally:
            _ACTIVE = None
            bnb.matmul_4bit = self._orig
            self._active = False
            self._store.clear()
            self._bytes = 0

    # ---- dequantization recipes (bnb 0.50 verbatim) --------------------------

    def _fwd_dequant(self, B: torch.Tensor, qs: Any, dtype: torch.dtype) -> torch.Tensor:
        """``_dequant_linear_fallback``'s B_dq: nested absmax rebuilt, then dequantized to
        ``dtype``."""
        ops = self._parts["ops"]
        absmax = qs.absmax
        if qs.nested:
            absmax_dq = torch.empty_like(qs.absmax, dtype=torch.float32)
            ops._dequantize_blockwise_impl(
                qs.absmax, qs.state2.absmax, qs.state2.code, 256, torch.float32, out=absmax_dq
            )
            absmax = absmax_dq + qs.offset
        out = torch.empty(qs.shape, dtype=dtype, device=B.device)
        ops._dequantize_4bit_impl(B, absmax, qs.blocksize, qs.quant_type, dtype, out=out)
        return out

    def _bwd_dequant(self, B: torch.Tensor, qs: Any, dtype: torch.dtype) -> torch.Tensor:
        """``MatMul4Bit.backward``'s weight: ``dequantize_4bit(B, state).to(dtype)``."""
        return self._parts["F"].dequantize_4bit(B, qs).to(dtype)

    def _weight(self, B: torch.Tensor, qs: Any, dtype: torch.dtype, kind: str) -> torch.Tensor:
        make = self._fwd_dequant if kind == "fwd" else self._bwd_dequant
        if not self._active:
            return make(B, qs, dtype)
        pair = (dtype, qs.dtype)
        if pair not in self._share:
            a = self._fwd_dequant(B, qs, dtype)
            b = self._bwd_dequant(B, qs, dtype)
            self._share[pair] = bool(torch.equal(a, b))
            del a, b
            if not self._share[pair]:
                logger.warning(
                    "nf4_dequant_cache: fwd/bwd dequant recipes differ for %s; caching them "
                    "separately", pair
                )
        key_kind = "w" if self._share[pair] else kind
        key = (B.data_ptr(), id(qs), key_kind, dtype)
        hit = self._store.get(key)
        if hit is not None:
            self.stats["hits"] += 1
            return hit
        self.stats["misses"] += 1
        w = make(B, qs, dtype)
        nbytes = w.numel() * w.element_size()
        if self._bytes + nbytes <= self.max_bytes:
            self._store[key] = w
            self._bytes += nbytes
        else:
            self.stats["skipped_full"] += 1
        return w

    # ---- the matmul_4bit mirror -------------------------------------------------

    def _mirrored(self, A: torch.Tensor, qs: Any, out: Optional[torch.Tensor]) -> bool:
        if qs is None or out is not None or A.device.type != "cuda":
            return False
        if A.dtype not in (torch.bfloat16, torch.float16) or A.numel() == 0:
            return False
        if len(qs.shape) != 2 or getattr(qs, "packing_format_for_cpu", False):
            return False
        K = A.shape[-1]
        if K == qs.shape[0] and K != qs.shape[1]:  # deprecated [K, N] weights
            return False
        if K != qs.shape[1]:
            return False
        if qs.nested and qs.state2.blocksize != 256:
            return False
        return True

    def _use_custom(self, A: torch.Tensor, qs: Any) -> bool:
        """bnb's ``gemm_4bit`` CUDA dispatch decision for this call."""
        ops = self._parts["ops"]
        K = A.shape[-1]
        M = A.numel() // K
        N = qs.shape[0]
        if M > ops._gemm_4bit_custom_max_m or K % qs.blocksize != 0:
            return False
        return bool(ops._gemm_4bit_use_custom_fn(A.device.index, A.dtype, M, N, K))

    def _matmul_4bit(self, A, B, quant_state, out=None, bias=None):  # noqa: ANN001 -- bnb signature
        qs = quant_state
        if not self._mirrored(A, qs, out):
            self.stats["passthrough"] += 1
            return self._orig(A, B, quant_state, out=out, bias=bias)
        B = B.view(-1, 1)
        use_custom = self._use_custom(A, qs)
        needs_grad = torch.is_grad_enabled() and (
            A.requires_grad or (bias is not None and bias.requires_grad)
        )
        if not needs_grad:
            if use_custom:
                self.stats["fused"] += 1
                return self._orig(A, B, quant_state, bias=bias)
            return F.linear(A, self._weight(B, qs, A.dtype, "fwd"), bias)
        return _CachedMatMul4Bit.apply(A, B, bias, qs, use_custom, self)


class _CachedMatMul4Bit(torch.autograd.Function):
    """``MatMul4Bit`` (bnb 0.50) with the dequantized weight taken from the cache."""

    @staticmethod
    def forward(ctx, A, B, bias, qs, use_custom, cache):  # noqa: ANN001
        if use_custom:
            cache.stats["fused"] += 1
            output = _fused(A, B, qs, bias)
        else:
            output = F.linear(A, cache._weight(B, qs, A.dtype, "fwd"), bias)
        ctx.state = qs
        ctx.cache = cache
        ctx.dtype_bias = None if bias is None else bias.dtype
        ctx.B = B if ctx.needs_input_grad[0] else None
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        req_gradA, _, req_gradBias = ctx.needs_input_grad[:3]
        grad_A = grad_bias = None
        if req_gradBias:
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)
        if req_gradA:
            W = ctx.cache._weight(ctx.B, ctx.state, grad_output.dtype, "bwd")
            grad_A = torch.matmul(grad_output, W)
        return grad_A, None, grad_bias, None, None, None


def dequantized_weight(weight: torch.Tensor, qs: Any, dtype: torch.dtype) -> torch.Tensor:
    """``MatMul4Bit.backward``'s weight for a 4-bit ``weight``: ``dequantize_4bit(B, state)
    .to(dtype)`` ([N, K]), taken from the open block visit's cache when there is one (the same
    bytes, shared with the forward's dequantization as ``_weight`` decides)."""
    B = weight.view(-1, 1)
    cache = _ACTIVE
    if cache is not None and cache._active:
        return cache._weight(B, qs, dtype, "bwd")
    import bitsandbytes.functional as bnbF

    return bnbF.dequantize_4bit(B, qs).to(dtype)


def _fused(A: torch.Tensor, B: torch.Tensor, qs: Any, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """``MatMul4Bit.forward``'s gemm_4bit call (bnb 0.50) -- the op re-runs bnb's dispatch."""
    if not qs.nested:
        return torch.ops.bitsandbytes.gemm_4bit.default(
            A, B, qs.shape, qs.absmax, qs.blocksize, qs.quant_type, bias=bias
        )
    return torch.ops.bitsandbytes.gemm_4bit.default(
        A, B, qs.shape, qs.state2.absmax, qs.blocksize, qs.quant_type, bias=bias,
        absmax_8bit=qs.absmax, absmax_code=qs.state2.code, absmax_offset=qs.offset,
    )


def _bnb_version() -> str:
    try:
        import bitsandbytes as bnb

        return getattr(bnb, "__version__", "?")
    except Exception:  # noqa: BLE001
        return "absent"
