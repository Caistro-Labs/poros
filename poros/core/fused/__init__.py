"""Fused compute for LoRA over a frozen NF4 base (CUDA, Triton).

PEFT's LoRA layer over a bitsandbytes ``Linear4bit`` runs ~10 elementwise passes per linear on
fp32 activations (cast, clone, scale, add, ...), and a SwiGLU MLP adds its activation and product
on fp32 tensors 3.6x the hidden width. ``install`` routes those modules through the autograd
Functions of ``poros.core.fused.lora`` instead: the same precisions (bf16 frozen GEMMs, fp32
adapter path and activations), a few fused kernels instead of the pass chains. It covers:

* a Qwen2 decoder layer whole (``_layer_forward``: transformers' layer and attention code over
  the fused Functions): RMSNorm, q/k/v with the rotary embedding in their epilogue, the model's
  attention (the memory-efficient SDPA kernel where it computes the call as specified:
  ``poros.core.fused.attention``), o with the residual add, RMSNorm, SwiGLU with the residual add;
* every other PEFT LoRA layer over a ``Linear4bit`` with one plain adapter (fp32 LoRA weights,
  no DoRA, no LoRA bias): its forward becomes ``LoraLinears`` on that linear alone;
* a Qwen2/Qwen3/Llama/Mistral-style attention's q/k/v (called in that order on one input): one
  ``LoraLinears`` over the three, the input cast once;
* a SwiGLU MLP of those classes: one ``SwiGLUMLP``.

Anything else keeps PEFT's forward. Every Poros path runs the same installed modules, so streamed,
resident, sequential and block-major stay bitwise-equal to each other; against PEFT's own modules
the values agree to fp32 rounding (the sums are formed in another order), within what a one-ulp
change of an fp32 input already moves the stock modules by (their frozen GEMMs take bf16
inputs). ``uninstall`` restores the original forwards.

``output_unused(layer)``: within it, the layer's MLP skips its frozen down GEMM -- for a
rematerialization whose output only roots the gradient (the last layer of a recomputed block):
no gradient reads that output's value, so every gradient is unchanged (tested bitwise).
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from poros.core.fused_tiles import pad, rope_kernel_head_dim

logger = logging.getLogger(__name__)

# Attention / MLP classes whose forward is the pattern fused here (checked by name: no import of
# any one family is needed). q/k/v are called in that order on one tensor; the MLP is
# down(act(gate(x)) * up(x)) with act SiLU; the decoder layer adds the MLP output straight to the
# residual (no norm after the MLP), which is what lets ``output_unused`` skip the down GEMM.
QKV_ATTENTION = ("Qwen2Attention", "Qwen3Attention", "LlamaAttention", "MistralAttention")
SWIGLU_MLP = ("Qwen2MLP", "Qwen3MLP", "LlamaMLP", "MistralMLP")
RESIDUAL_MLP_LAYERS = ("Qwen2DecoderLayer", "Qwen3DecoderLayer", "LlamaDecoderLayer",
                       "MistralDecoderLayer")

# Decoder layers run whole by ``_layer_forward`` (transformers' Qwen2 layer + attention code,
# over the fused Functions); their norms must be one of these RMSNorm classes.
FUSED_LAYERS = ("Qwen2DecoderLayer",)
RMS_NORMS = ("Qwen2RMSNorm",)

_ATTR = "_poros_fused"
_REFUSED_ATTR = "_poros_fused_refused"


def available(device: Any = None) -> Optional[str]:
    """None if the fused compute can run here (on ``device``: the current CUDA device by
    default), else why not (in words)."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    from poros.core import card

    why = card.triton_reason(card.capability(device))
    if why is not None:
        return why
    try:
        import triton  # noqa: F401
    except ImportError:
        return "triton is not installed"
    try:
        import bitsandbytes  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        return "bitsandbytes / peft are not installed"
    return None


def _adapter(mod: nn.Module) -> Optional[str]:
    """The adapter name if ``mod`` is a PEFT LoRA layer over a ``Linear4bit`` that the fused
    linear computes as PEFT's forward does; else None. The static part (types) is decided once per
    module, the rest (merged / disabled / active adapter / dtypes) on every call."""
    ok = getattr(mod, "_poros_lin_ok", None)
    if ok is None:
        ok = _static_ok(mod)
        try:
            mod._poros_lin_ok = ok
        except AttributeError:  # pragma: no cover - not a module
            return None
    if not ok or mod.merged or mod.disable_adapters:
        return None
    active = mod.active_adapters
    if len(active) != 1 or active[0] not in mod.lora_A:
        return None
    name = active[0]
    if mod.use_dora.get(name, False) or name in getattr(mod, "lora_variant", {}):
        return None
    A, B = mod.lora_A[name], mod.lora_B[name]
    if not (isinstance(A, nn.Linear) and isinstance(B, nn.Linear)) or A.bias is not None \
            or B.bias is not None:
        return None
    if A.weight.dtype != torch.float32 or B.weight.dtype != torch.float32:
        return None
    if not isinstance(mod.lora_dropout[name], (nn.Dropout, nn.Identity)):
        return None
    if mod.base_layer.compute_dtype not in (torch.bfloat16, torch.float16):
        return None
    return name


def _static_ok(mod: nn.Module) -> bool:
    try:
        import bitsandbytes as bnb
        from peft.tuners.lora import LoraLayer
    except ImportError:
        return False
    return isinstance(mod, LoraLayer) and isinstance(getattr(mod, "base_layer", None),
                                                     bnb.nn.Linear4bit)


def _lin(mod: nn.Module) -> Any:
    from poros.core.fused.lora import Lin

    return Lin(mod, mod.active_adapters[0])


def _params(lins: List[Any]) -> List[torch.Tensor]:
    return [p for lin in lins for p in (lin.A, lin.B)]


def _usable(x: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == torch.float32 and x.numel() > 0
            and not torch.is_autocast_enabled())


def _linear_forward(self: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """A LoRA layer's forward through ``LoraLinears`` (PEFT's own forward when not applicable)."""
    group = getattr(self, "_poros_qkv", None)
    if group is not None:
        out = group.take(self, x)
        if out is not None:
            return out
    if args or kwargs or not _usable(x) or _adapter(self) is None:
        return self._poros_peft_forward(x, *args, **kwargs)
    from poros.core.fused.lora import LoraLinears

    lin = _lin(self)
    return LoraLinears.apply(x, None, [lin], -1, None, lin.A, lin.B)[0]


class _QKV:
    """q/k/v of one attention: q's call computes all three over its input, k's and v's calls on
    the same tensor take theirs. The dropouts draw in q, k, v order at q's call, as PEFT's three
    calls would draw them (nothing else draws between them in these attention classes)."""

    def __init__(self, q: nn.Module, k: nn.Module, v: nn.Module) -> None:
        self.mods = (q, k, v)
        self.x: Optional[torch.Tensor] = None
        self.out: Dict[int, torch.Tensor] = {}

    def take(self, mod: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
        q, k, v = self.mods
        if mod is q:
            self.x, self.out = None, {}
            if not _usable(x) or any(_adapter(m) is None for m in self.mods):
                return None
            from poros.core.fused.lora import LoraLinears

            lins = [_lin(m) for m in self.mods]
            outs = LoraLinears.apply(x, None, lins, -1, None, *_params(lins))
            self.x = x
            self.out = {id(k): outs[1], id(v): outs[2]}
            return outs[0]
        if self.x is not None and x is self.x and id(mod) in self.out:
            out = self.out.pop(id(mod))
            if not self.out:
                self.x = None
            return out
        return None


def _mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    mods = (self.gate_proj, self.up_proj, self.down_proj)
    if not _usable(x) or any(_adapter(m) is None for m in mods):
        return self._poros_orig_forward(x)
    from poros.core.fused.lora import SwiGLUMLP

    gate, up, down = (_lin(m) for m in mods)
    if not (gate.r == up.r == down.r):
        return self._poros_orig_forward(x)
    skip = bool(getattr(self, "_poros_output_unused", False))
    return SwiGLUMLP.apply(x, None, gate, up, down, skip, gate.A, gate.B, up.A, up.B, down.A,
                           down.B)


def _silu(mod: nn.Module) -> bool:
    """Whether an MLP's activation is SiLU (poros.core.activation.is_silu, which the fused MoE
    experts check too)."""
    from poros.core.activation import is_silu

    return is_silu(getattr(mod, "act_fn", None))


def _norm_ok(norm: nn.Module) -> bool:
    w = getattr(norm, "weight", None)
    return (type(norm).__name__ in RMS_NORMS and isinstance(w, torch.Tensor)
            and w.dtype == torch.float32 and not w.requires_grad
            and isinstance(getattr(norm, "variance_epsilon", None), float))


def _rope(pe: Tuple[torch.Tensor, torch.Tensor], x: torch.Tensor, head_dim: int) -> Any:
    """The rotary embedding as the q/k epilogue applies it (transformers' apply_rotary_pos_emb
    with cos/sin [B or 1, S, head_dim] fp32), or None (then transformers' own function runs)."""
    cos, sin = pe
    B, S = x.shape[0], x.shape[1]
    if (cos.dtype != torch.float32 or sin.dtype != torch.float32 or cos.dim() != 3
            or cos.shape != sin.shape or cos.shape[1] != S or cos.shape[2] != head_dim
            or cos.shape[0] not in (1, B) or rope_kernel_head_dim(head_dim) is None):
        return None
    from poros.core.fused.lora import Rope

    return Rope(cos, sin, 2)


def _causal_kernel(attn: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask: Any, dropout: float, kwargs: Dict[str, Any]) -> bool:
    """Whether this attention call is transformers' SDPA attention with no mask -- causal over
    the whole sequence -- which ``poros.core.fused.attention`` computes; else the model's own."""
    if getattr(attn.config, "_attn_implementation", None) != "sdpa" or kwargs:
        return False
    if not getattr(attn, "is_causal", False) or q.shape[2] < 2:
        return False
    from poros.core.fused import attention

    return attention.usable(q, k, v, mask, dropout)


def _layer_forward(self: nn.Module, hidden_states: torch.Tensor,
                   attention_mask: Optional[torch.Tensor] = None,
                   position_ids: Optional[torch.Tensor] = None, past_key_values: Any = None,
                   use_cache: Optional[bool] = False,
                   position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   **kwargs: Any) -> torch.Tensor:
    """A Qwen2 decoder layer (transformers' ``Qwen2DecoderLayer`` + ``Qwen2Attention`` forward)
    over the fused Functions: norm, q/k/v, the model's own rotary and attention function, o with
    the residual add, norm, SwiGLU with the residual add. Falls back to the layer's own forward
    whenever any piece is not the fused case."""
    attn, mlp = self.self_attn, self.mlp
    ln1, ln2 = self.input_layernorm, self.post_attention_layernorm
    mods = (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.gate_proj, mlp.up_proj,
            mlp.down_proj)
    if (past_key_values is not None or use_cache or position_embeddings is None
            or not _usable(hidden_states) or getattr(attn, "sliding_window", None) is not None
            or not (_norm_ok(ln1) and _norm_ok(ln2)) or not _silu(mlp)
            or any(_adapter(m) is None for m in mods)):
        return self._poros_orig_forward(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs)
    from poros.core.fused.lora import LoraLinears, RMSNorm, SwiGLUMLP

    lq, lk, lv, lo, lg, lu, ld = (_lin(m) for m in mods)
    if not (lg.r == lu.r == ld.r):
        return self._poros_orig_forward(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs)
    hf = sys.modules[type(attn).__module__]
    lead = hidden_states.shape[:-1]
    shape = (*lead, -1, attn.head_dim)
    n1 = RMSNorm.apply(hidden_states, ln1.weight, ln1.variance_epsilon)
    rope = _rope(position_embeddings, hidden_states, attn.head_dim)
    q, k, v = LoraLinears.apply(n1, None, [lq, lk, lv], -1, rope, *_params([lq, lk, lv]))
    q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
    if rope is None:
        cos, sin = position_embeddings
        q, k = hf.apply_rotary_pos_emb(q, k, cos, sin)
    dropout = 0.0 if not attn.training else attn.attention_dropout
    if _causal_kernel(attn, q, k, v, attention_mask, dropout, kwargs):
        from poros.core.fused import attention as fa

        if fa._BF16 and fa.flash_usable(q):
            o = fa.sdpa_bf16(q, k, v, float(attn.scaling))
        else:
            o = fa.sdpa_efficient(q, k, v, float(attn.scaling))
    else:
        attention = hf.ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, hf.eager_attention_forward)
        o, _ = attention(attn, q, k, v, attention_mask, dropout=dropout, scaling=attn.scaling,
                         sliding_window=None, **kwargs)
    o = o.reshape(*lead, -1).contiguous()
    h1 = LoraLinears.apply(o, hidden_states, [lo], -1, None, lo.A, lo.B)[0]
    n2 = RMSNorm.apply(h1, ln2.weight, ln2.variance_epsilon)
    skip = bool(getattr(mlp, "_poros_output_unused", False))
    return SwiGLUMLP.apply(n2, h1, lg, lu, ld, skip, *_params([lg, lu, ld]))


def install(model: nn.Module) -> Dict[str, int]:
    """Route ``model``'s eligible LoRA linears, q/k/v groups and SwiGLU MLPs through the fused
    Functions. Idempotent. Returns counts of what was routed."""
    counts = {"layers": 0, "linears": 0, "qkv_groups": 0, "swiglu_mlps": 0}
    if getattr(model, _ATTR, None) is not None:
        return dict(model._poros_fused)
    refused: List[str] = _keep_unfit_ranks(model)
    for mod in model.modules():
        name = type(mod).__name__
        if name in FUSED_LAYERS and "_poros_orig_forward" not in vars(mod) and not any(
                getattr(m, "_poros_lin_ok", None) is False for m in mod.modules()):
            why = _layer_refusal(mod)
            if why is None:
                mod._poros_orig_forward = mod.forward
                mod.forward = types.MethodType(_layer_forward, mod)
                counts["layers"] += 1
            elif why not in refused:
                refused.append(why)
        if name in SWIGLU_MLP and _silu(mod) and all(
                _adapter(getattr(mod, p, None)) is not None
                for p in ("gate_proj", "up_proj", "down_proj")):
            mod._poros_orig_forward = mod.forward
            mod.forward = types.MethodType(_mlp_forward, mod)
            counts["swiglu_mlps"] += 1
        elif name in QKV_ATTENTION and all(_adapter(getattr(mod, p, None)) is not None
                                           for p in ("q_proj", "k_proj", "v_proj")):
            group = _QKV(mod.q_proj, mod.k_proj, mod.v_proj)
            for m in group.mods:
                m._poros_qkv = group
            counts["qkv_groups"] += 1
    for mod in model.modules():
        if _static_ok(mod) and getattr(mod, "_poros_lin_ok", True) \
                and "_poros_peft_forward" not in vars(mod):
            mod._poros_peft_forward = mod.forward
            mod.forward = types.MethodType(_linear_forward, mod)
            counts["linears"] += 1
    setattr(model, _ATTR, counts)
    setattr(model, _REFUSED_ATTR, refused)
    for why in refused:
        logger.warning("fused compute: %s; those layers run their own forward", why)
    logger.info("fused compute installed: %s", counts)
    return dict(counts)


def _keep_unfit_ranks(model: nn.Module) -> List[str]:
    """The LoRA layers whose rank the fused kernels cannot run on this GPU (their tiles exceed
    its shared memory: ``poros.core.fused.fit``) are marked to keep PEFT's forward; returns why,
    once per rank. Decided before training, so every pass of every path computes them alike."""
    if not torch.cuda.is_available():
        return []
    head_dim = next((int(m.self_attn.head_dim) for m in model.modules()
                     if type(m).__name__ in FUSED_LAYERS
                     and isinstance(getattr(getattr(m, "self_attn", None), "head_dim", None), int)),
                    None)
    head_dim = rope_kernel_head_dim(head_dim)  # None: transformers' rotary function (_rope)
    from poros.core.fused import fit

    why: List[str] = []
    for mod in model.modules():
        if not _static_ok(mod) or "_poros_peft_forward" in vars(mod):
            continue
        dtype = mod.base_layer.compute_dtype
        for A in mod.lora_A.values():
            r = int(A.weight.shape[0])
            unfit = fit.reason(pad(r), dtype, head_dim)
            if unfit is not None:
                mod._poros_lin_ok = False
                if unfit not in why:
                    why.append(unfit)
    return why


def _layer_refusal(layer: nn.Module) -> Optional[str]:
    """Why a decoder layer cannot take the fused layer forward, or None. The fused layer computes
    a SwiGLU MLP (silu(gate) * up): any other activation keeps the layer's own forward."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not _silu(mlp):
        act = getattr(getattr(mlp, "config", None), "hidden_act", None)
        if act in (None, "silu", "swish"):  # config.json's name is not what runs: the callable's
            fn = getattr(mlp, "act_fn", None)
            act = getattr(fn, "__name__", None) or type(fn).__name__
        return f"{type(layer).__name__}: MLP activation {act!r} is not SiLU"
    return None


def refusals(model: nn.Module) -> List[str]:
    """Why ``install`` left decoder layers on their own forward (empty when none were)."""
    return list(getattr(model, _REFUSED_ATTR, []) or [])


def uninstall(model: nn.Module) -> None:
    """Restore every forward ``install`` replaced."""
    for mod in model.modules():
        for attr in ("_poros_orig_forward", "_poros_peft_forward"):
            if attr in vars(mod):
                del mod.forward
                delattr(mod, attr)
        if "_poros_qkv" in vars(mod):
            del mod._poros_qkv
        if "_poros_output_unused" in vars(mod):
            del mod._poros_output_unused
    for attr in (_ATTR, _REFUSED_ATTR):
        if attr in vars(model):
            delattr(model, attr)


def installed(model: nn.Module) -> bool:
    return getattr(model, _ATTR, None) is not None


@contextlib.contextmanager
def output_unused(layers: List[nn.Module]) -> Iterator[None]:
    """For a rematerialization of ``layers`` whose output only roots ``autograd.grad``: the last
    layer's fused MLP skips its frozen down GEMM (its output becomes the residual plus zeros).
    A no-op unless that layer is a residual-after-MLP decoder layer with an installed SwiGLU."""
    mlp = None
    if layers:
        last = layers[-1]
        cand = getattr(last, "mlp", None)
        if (type(last).__name__ in RESIDUAL_MLP_LAYERS and cand is not None
                and "_poros_orig_forward" in vars(cand)):
            mlp = cand
    if mlp is None:
        yield
        return
    mlp._poros_output_unused = True
    try:
        yield
    finally:
        mlp._poros_output_unused = False
