"""The CPU rung of the CUDA route: a tiny random model of a real config, through the way a CUDA
machine trains a dense model (``trainer``: PorosTrainer's blockwise forward and its
rematerialized backward, as ``poros train`` and ``poros prove`` run it), against the model's own
transformers forward.

The reference is the same PEFT model under plain autograd. Equal means within
``TOLERANCE`` on the loss and on every adapter gradient: structural, on the CPU in fp32, as the
procedure says (docs/models.md, rung 1). It says nothing about NF4, the GPU,
real shapes or the real weights: that is what ``poros prove <model> --unvalidated`` checks.

A config may be anyone's. Its sizes are overridden before anything is built (``tiny.shrink``),
every dropout is set to 0, ``auto_map`` is ignored, nothing runs a repo's own code, and the only
weights are random ones made here.
"""

from __future__ import annotations

import contextlib
import copy
import re
import tempfile
import warnings
from typing import Any, Optional

TOLERANCE = 1e-4
SEQ = 16
HEADS, HEAD_DIM = 4, 8  # what tiny.shrink sets every config's attention to
ROUTES = ("trainer",)
_SHOWN = re.compile(r"[A-Za-z0-9_.]{1,160}")
# A mixture of experts on a GPU trains only through poros.moe, for a validated family.
MOE_VALIDATED = ("a validated mixture of experts trains on a GPU through poros.moe, which "
                 "measured runs on real cards cover; this check runs the dense route")


def _no_dropout(cfg: Any) -> None:
    """Every dropout in the config, and its text config's, at 0: a structural check compares two
    forwards, and dropout would draw differently in each."""
    for holder in {id(c): c for c in (cfg, cfg.get_text_config())}.values():
        for key, value in list(vars(holder).items()):
            if "dropout" in key and isinstance(value, float):
                setattr(holder, key, 0.0)


def _named(name: str) -> str:
    return name if _SHOWN.fullmatch(name) else "unknown"


def _grads(model) -> dict[str, Any]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def _verdict(loss, grads: dict, ref_loss, ref_grads: dict) -> dict[str, Any]:
    """Within TOLERANCE on the loss and every adapter gradient, bit for bit or not, and the first
    that is not."""
    import torch

    diffs = {"loss": float((loss.detach().float() - ref_loss.float()).abs().max())}
    first: Optional[dict[str, str]] = None
    if diffs["loss"] > TOLERANCE:
        first = {"module": "the loss", "tensor": "its value"}
    bitwise = torch.equal(loss.detach(), ref_loss)
    if set(grads) != set(ref_grads):
        missing = sorted(set(grads) ^ set(ref_grads))[0]
        return {"ok": False, "bitwise": False, "max_abs_diff": None,
                "first_difference": {"module": _named(missing), "tensor": "no gradient"}}
    for name in ref_grads:
        d = float((grads[name].float() - ref_grads[name].float()).abs().max())
        diffs[name] = d
        bitwise = bitwise and torch.equal(grads[name], ref_grads[name])
        if first is None and d > TOLERANCE:
            module, _, tensor = name.partition(".lora_")
            first = {"module": _named(module), "tensor": f"lora_{tensor} gradient"}
    return {"ok": first is None, "bitwise": bool(bitwise), "max_abs_diff": max(diffs.values()),
            "first_difference": first}


def _trainer_route(model, batch, kind: str):
    """PorosTrainer's blockwise forward and backward on the CPU, a block per layer."""
    from poros.core.config import PorosConfig
    from poros.train.trainer import PorosTrainer

    with tempfile.TemporaryDirectory(prefix="poros-try-") as out:
        cfg = PorosConfig(model_name_or_path=kind, block_size=1,
                          deterministic=False, output_dir=out, blockwise_arch_check="error",
                          enable_prefetch=False, lora_dropout=0.0, save_steps=0, eval_steps=0,
                          progress=False)
        trainer = PorosTrainer(cfg)
        trainer.model = model
        import torch

        trainer.device = torch.device("cpu")
        trainer._adapter_backend = "peft"
        trainer._setup_blockwise_training()
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = trainer._forward_blockwise_adapter_model(trainer._move_batch_to_device(batch))
        loss.backward()
        trainer._release_retained_forward_blocks()
        return loss.detach(), _grads(trainer.model)


def prove(raw: dict) -> dict[str, Any]:
    """One microstep of a tiny model of this config through the CUDA route and through its own
    forward, from the same random adapter. A family the registry does not name runs as
    ``user_asserted`` for this call alone. ValueError for a config that does not shrink."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from poros.core.arch_registry import (
        architectures,
        asserted,
        registry_type,
        validated_arch_types,
    )
    from poros.metal import tiny
    from poros.metal.layout import empty

    kind = registry_type(raw)
    validated = kind in validated_arch_types() | validated_arch_types("moe")
    status = "validated" if validated else "user_asserted"
    spec = architectures().get(kind)
    if spec is not None and spec.entry == "moe":
        # a mixture of experts by its registry entry is not this dense route's to check, and is
        # not shrunk to find that out (a looped one does not shrink: tiny.shrink says why)
        from poros.preflight.arch import MOE_ON_GPU, MOE_TARGET

        why = (MOE_VALIDATED if validated else MOE_TARGET if spec.status == "target"
               else MOE_ON_GPU)
        return {"route": "cuda", "status": status, "moe": True, "tolerance": TOLERANCE,
                "routes": {}, "ok": None, "first_difference": None, "compared": 0,
                "nonzero": 0, "why": why}
    # A head's width two ways at once: some families read head_dim, others (StableLM's attention
    # beside its rotary) divide hidden by heads. Equal here, so neither builds a model at odds.
    cfg = tiny.shrink(raw, hidden=HEADS * HEAD_DIM)
    skeleton = empty(cfg)
    held = sum(t.numel() for t in (*skeleton.parameters(), *skeleton.buffers()))
    if 2 * held > tiny.LARGEST_TINY_BYTES:
        raise ValueError("its config did not shrink to a tiny model")
    moe = bool(tiny.sparse_blocks(skeleton))
    del skeleton
    out: dict[str, Any] = {"route": "cuda", "status": status, "moe": moe,
                           "tolerance": TOLERANCE, "routes": {}}
    if moe:
        from poros.preflight.arch import MOE_ON_GPU

        # not checked: neither True nor a difference
        return {**out, "ok": None, "first_difference": None, "compared": 0, "nonzero": 0,
                "why": MOE_VALIDATED if validated else MOE_ON_GPU}
    _no_dropout(cfg)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(1)
        # SDPA, as a CUDA load gets it: the trainer leaves a causal mask to it (shrink's eager is
        # the Mac's, which builds its masks itself). A family transformers runs on SDPA not yet
        # (Gemma 2's soft-capped attention, from 5.21) loads eager on CUDA too: the same here.
        try:
            base = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
        except ValueError as exc:
            if "scaled_dot_product_attention" not in str(exc):
                raise
            base = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
        with torch.no_grad():
            for param in base.parameters():
                param.normal_(0.0, 0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = get_peft_model(base.float(), LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"))
        gen = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for param in model.parameters():  # lora_B starts at zero, and hides lora_A's grads
                if param.requires_grad:
                    param.copy_(0.03 * torch.randn(param.shape, generator=gen,
                                                   dtype=param.dtype))
        vocab = cfg.get_text_config().vocab_size
        ids = torch.randint(0, vocab, (1, SEQ), generator=torch.Generator().manual_seed(3))
        batch = {"input_ids": ids, "labels": ids.clone()}
        extra = tiny._text_only_batch(model, ids)
        for_trainer = copy.deepcopy(model)
        model.zero_grad(set_to_none=True)
        reference = model(**batch, **extra)
        reference.loss.backward()
        ref_loss, ref_grads = reference.loss.detach(), _grads(model)
        nonzero = sum(bool(g.abs().sum() > 0) for g in ref_grads.values())
        out.update(compared=len(ref_grads), nonzero=nonzero)
        with asserted(kind) if not validated else contextlib.nullcontext():
            for name, go in (("trainer", lambda: _trainer_route(for_trainer, batch, kind)),):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        loss, grads = go()
                    out["routes"][name] = _verdict(loss, grads, ref_loss, ref_grads)
                except Exception as exc:  # its name alone: its words may be the config's
                    out["routes"][name] = {"ok": False, "bitwise": False, "max_abs_diff": None,
                                           "first_difference": None,
                                           "why": f"it could not run ({type(exc).__name__})"}
    finally:
        torch.set_num_threads(threads)
    routes = out["routes"]
    first = next((r["first_difference"] for r in routes.values() if r.get("first_difference")),
                 None)
    out["ok"] = bool(routes) and all(r["ok"] for r in routes.values()) and out["nonzero"] > 0
    out["first_difference"] = first
    if not out["ok"] and first is None:
        out["why"] = next((r["why"] for r in routes.values() if r.get("why")),
                          "no adapter gradient was nonzero")
    return out
