"""The LM head and next-token cross-entropy, a chunk of rows at a time (fp32, deterministic).

PyTorch's path holds the whole ``[tokens, vocab]`` fp32 logits several times over: the logits, the
shifted copy, ``log_softmax``'s output kept for the backward, and their grads (Qwen2.5-72B, 4,096
tokens: 2.3 GiB each, ~9 GiB at once). Here the forward computes, per chunk of rows, the logits,
their log-sum-exp, the loss terms and -- the grad of a mean cross-entropy being known in closed
form, ``(softmax - onehot) / n`` -- the grad of the hidden states, then frees the chunk; the
backward only scales that grad. Same math in fp32 (the head GEMM, the softmax), summed in another
order than ``F.cross_entropy``; one chunk's logits are live at a time.

``poros.moe`` models (bf16): ``head_loss`` (the fused MoE path's training loss, in every loop)
takes the final norm's output and the LM head and never holds more than one chunk's logits; on
CUDA each chunk's cross-entropy and its grad are one Triton kernel (``ce_kernel``).
``CausalLMLoss`` (any other caller of the model's own loss, e.g. its forward with labels) takes
the logits: transformers' ``ForCausalLMLoss`` upcasts the whole ``[tokens, vocab]`` logits to
fp32 and ``cross_entropy`` keeps its ``log_softmax`` for the backward; with the bf16 logits and
their grad that is ~7 GB at 4,096 tokens of Qwen3-30B-A3B's 151,936-token vocabulary.
``install_loss(model)`` replaces the model's ``loss_function`` (read by its own forward, so by the
sequential loop, and by the block-major driver's head) with ``CausalLMLoss``: the same mean (or
``num_items_in_batch``) cross-entropy of the shifted labels in fp32, a chunk of ``MOE_ROWS`` rows
upcast at a time, its backward ``(softmax - onehot) / n`` per chunk, written in the logits' dtype.
Values differ from ``F.cross_entropy``'s by fp32 rounding (another summation order); every Poros
loop computes it the same way on the same rows.

The torch body of one chunk -- log-sum-exp, the target's term, ``(softmax - onehot)`` masked to
the valid rows, in place -- is ``chunk_ce_``, shared by ``HeadCrossEntropy`` and ``_HeadLoss``.
The chunk sizes, the GEMM (split fp32 or bf16) and where the mean's divisor is applied stay each
caller's own: each is part of its loss's bits.
"""

from __future__ import annotations

from typing import Any

import torch

CHUNK_ROWS = 1024
MOE_ROWS = 512  # the poros.moe losses' rows per chunk


def chunk_ce_(lc: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """One chunk's cross-entropy, in place: fp32 logits ``lc`` [rows, V] become the grad of the
    rows' summed loss terms, ``softmax - onehot`` (zero on rows not ``valid``); returns that sum
    (fp32 scalar) over the valid rows. ``tgt`` holds a class for every row, any for an invalid
    one."""
    lse = torch.logsumexp(lc, dim=-1)
    picked = lc.gather(1, tgt[:, None]).squeeze(1)
    total = torch.where(valid, lse - picked, torch.zeros_like(lse)).sum()
    lc.sub_(lse[:, None]).exp_()  # softmax, in place
    rows = torch.arange(lc.shape[0], device=lc.device)
    lc[rows, tgt] -= 1.0  # one target per row: no index repeats
    lc.mul_(valid[:, None].to(lc.dtype))
    return total


def _mm_f32(a: torch.Tensor, b: torch.Tensor, split: bool) -> torch.Tensor:
    """a @ b in fp32 for fp32 ``a``. With ``split`` (``b`` bf16, exactly the fp32 weight's values):
    ``a`` as a bf16 head plus a bf16 tail, two bf16 GEMMs summed in fp32 -- every product exact
    but for ``a``'s bits past 16 (2^-17 relative), at bf16 GEMM speed."""
    if not split:
        return a @ b
    hi = a.to(torch.bfloat16)
    lo = (a - hi.float()).to(torch.bfloat16)
    out = torch.mm(lo, b, out_dtype=torch.float32)
    return out.add_(torch.mm(hi, b, out_dtype=torch.float32))


class HeadCrossEntropy(torch.autograd.Function):
    """mean over the target tokens of ``-log softmax(h @ W^T)[next token]``.

    ``h`` [B, S, H] fp32 (the head's input), ``W`` [V, H] fp32, or bf16 when the fp32 weight's
    values are exactly bf16 ones (a compact head: see ``_mm_f32``), ``labels`` [B, S] with -100
    for no target; the target of position s is ``labels[:, s + 1]`` (HF's causal-LM shift).
    """

    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, W: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, S, H = h.shape
        target = torch.full((B, S), -100, dtype=torch.long, device=h.device)
        target[:, :-1] = labels[:, 1:]
        h2 = h.reshape(B * S, H)
        t2 = target.reshape(B * S)
        valid = t2 != -100
        n = valid.sum()
        loss = torch.zeros((), dtype=torch.float32, device=h.device)
        grad = torch.empty_like(h2)
        split = W.dtype == torch.bfloat16
        for r0 in range(0, B * S, CHUNK_ROWS):
            hc = h2[r0:r0 + CHUNK_ROWS]
            tc = t2[r0:r0 + CHUNK_ROWS]
            vc = valid[r0:r0 + CHUNK_ROWS]
            logits = _mm_f32(hc, W.t(), split)  # [rows, V] fp32
            loss += chunk_ce_(logits, tc.clamp(min=0), vc)
            grad[r0:r0 + CHUNK_ROWS] = _mm_f32(logits, W, split)
            del logits
        denom = n.clamp(min=1).to(torch.float32)
        grad.div_(denom)
        ctx.save_for_backward(grad)
        ctx.shape = h.shape
        return loss / denom

    @staticmethod
    def backward(ctx: Any, go: torch.Tensor):
        (grad,) = ctx.saved_tensors
        return (grad * go).view(ctx.shape), None, None


class _ChunkedCrossEntropy(torch.autograd.Function):
    """sum over rows i with target[i] != ignore of lse(logits[i]) - logits[i, target[i]] (fp32),
    divided by ``denom``."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, target: torch.Tensor, denom: torch.Tensor,
                ignore_index: int) -> torch.Tensor:
        valid = target != ignore_index
        tgt = torch.where(valid, target, torch.zeros_like(target))
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for r0 in range(0, logits.shape[0], MOE_ROWS):
            lc = logits[r0:r0 + MOE_ROWS].float()
            lse = torch.logsumexp(lc, dim=-1)
            picked = lc.gather(1, tgt[r0:r0 + MOE_ROWS, None]).squeeze(1)
            total += torch.where(valid[r0:r0 + MOE_ROWS], lse - picked,
                                 torch.zeros_like(lse)).sum()
        ctx.save_for_backward(logits, tgt, valid, denom)
        return total / denom

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        logits, tgt, valid, denom = ctx.saved_tensors
        scale = grad_out / denom
        grad = torch.empty_like(logits)
        for r0 in range(0, logits.shape[0], MOE_ROWS):
            p = torch.softmax(logits[r0:r0 + MOE_ROWS].float(), dim=-1)
            rows = torch.arange(p.shape[0], device=p.device)
            p[rows, tgt[r0:r0 + MOE_ROWS]] -= 1.0  # one target per row: no index repeats
            p.mul_((valid[r0:r0 + MOE_ROWS, None] * scale).to(p.dtype))
            grad[r0:r0 + MOE_ROWS] = p.to(grad.dtype)
        return grad, None, None, None


def CausalLMLoss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int,
                 num_items_in_batch: torch.Tensor | int | None = None, ignore_index: int = -100,
                 shift_labels: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
    """``ForCausalLMLoss``'s signature and semantics: labels shifted left one place (-100 past
    the end), mean over the targets -- or their sum over ``num_items_in_batch``."""
    if shift_labels is None:
        labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:]
    target = shift_labels.reshape(-1).to(logits.device)
    flat = logits.reshape(-1, vocab_size)
    if num_items_in_batch is None:
        denom = (target != ignore_index).sum().to(torch.float32)
    else:
        denom = torch.as_tensor(num_items_in_batch, device=logits.device).to(torch.float32)
    return _ChunkedCrossEntropy.apply(flat, target, denom, ignore_index)


def install_loss(model: torch.nn.Module) -> None:
    """The model's loss becomes ``CausalLMLoss`` (its forward and the block-major head)."""
    model.loss_function = CausalLMLoss


def uninstall_loss(model: torch.nn.Module) -> None:
    """The model's own loss again (``install_loss`` undone). transformers keeps a set loss in
    ``_loss_function`` (its ``loss_function`` property); a plain module, as an attribute."""
    if vars(model).get("_loss_function") is CausalLMLoss:
        del model._loss_function
    elif vars(model).get("loss_function") is CausalLMLoss:
        del model.loss_function


class _HeadLoss(torch.autograd.Function):
    """``CausalLMLoss`` of ``h @ W^T`` (``h`` [rows, H] the final norm's output, ``W`` [V, H] the
    frozen LM head), a chunk of rows at a time: no [rows, vocab] logits beyond one chunk's. Each
    chunk's logits are the head's own (the bf16 GEMM), their loss terms as ``CausalLMLoss``
    takes them, and -- the grad of the cross-entropy being ``(softmax - onehot)`` -- its rows'
    ``(softmax - onehot) @ W`` (the bf16 GEMM), kept in fp32; the backward scales that by the
    loss's grad over ``denom``."""

    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                denom: torch.Tensor, ignore_index: int) -> torch.Tensor:
        valid = target != ignore_index
        tgt = torch.where(valid, target, torch.zeros_like(target))
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        grad = torch.empty(h.shape, dtype=torch.float32, device=h.device)
        fused = h.is_cuda and _fused_ce()
        g = None
        for r0 in range(0, h.shape[0], MOE_ROWS):
            hc = h[r0:r0 + MOE_ROWS]
            vc = valid[r0:r0 + MOE_ROWS]
            tc = tgt[r0:r0 + MOE_ROWS]
            if fused:  # one kernel for the cross-entropy and its grad (ce_kernel)
                from poros.core.fused import ce_kernel

                logits = torch.nn.functional.linear(hc, weight)
                if g is None or g.shape != logits.shape:
                    g = torch.empty_like(logits)
                total += ce_kernel.cross_entropy_rows(logits, tc, vc, g).sum()
                del logits
                grad[r0:r0 + MOE_ROWS] = (g @ weight).float()
                continue
            lc = torch.nn.functional.linear(hc, weight).float()
            total += chunk_ce_(lc, tc, vc)
            grad[r0:r0 + MOE_ROWS] = (lc.to(h.dtype) @ weight).float()
            del lc
        ctx.save_for_backward(grad, denom)
        ctx.dtype = h.dtype
        return total / denom

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        grad, denom = ctx.saved_tensors
        return (grad * (grad_out / denom)).to(ctx.dtype), None, None, None, None


def _fused_ce() -> bool:
    """The Triton cross-entropy (ce_kernel) where Triton is installed, else the torch ops
    (another rounding of the same loss: a CPU run, or a test of the torch path)."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def head_loss(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
              ignore_index: int = -100) -> torch.Tensor:
    """The causal-LM loss of the LM head ``weight`` over the final norm's output ``hidden``
    [B, S, H] and ``labels`` [B, S] (shifted left one place, mean over the targets), without the
    [tokens, vocab] logits (``_HeadLoss``)."""
    labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    target = labels.reshape(-1).to(hidden.device)
    denom = (target != ignore_index).sum().to(torch.float32)
    return _HeadLoss.apply(hidden.reshape(-1, hidden.shape[-1]), weight, target, denom,
                           ignore_index)
