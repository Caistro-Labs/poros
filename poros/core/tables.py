"""The frozen embedding table and LM head, kept in the dtype they load in.

PEFT's k-bit preparation casts every 16-bit parameter to fp32, on the host. For the two
vocabulary tables that doubles their bytes (Qwen2.5-72B: 2 x 2.3 GiB in bf16, 2 x 4.6 GiB in
fp32) and, while it runs, holds a table in both dtypes at once. A bf16 or fp16 -> fp32 cast is
exact, so a table can stay in its load dtype and be cast where it is used: on the GPU, a chunk
of rows at a time, or only the rows an embedding lookup reads. Every value a forward sees is the
one the fp32 table would hold, so the results do not change by a bit.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

# Source bytes cast_to widens per step once the table has crossed (in one copy, into its result):
# it bounds each kernel's size only, as the cast allocates nothing beyond its result.
CHUNK_BYTES = 64 << 20

_SIXTEEN_BIT = (torch.float16, torch.bfloat16)


def vocab_tables(model: nn.Module) -> List[nn.Parameter]:
    """The model's input embedding and LM head weights that PEFT's k-bit preparation would cast
    to fp32 (16-bit, not 4-bit): each once, so a tied table appears once."""
    found: List[nn.Parameter] = []
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        fn = getattr(model, getter, None)
        try:
            module = fn() if callable(fn) else None
        except (AttributeError, NotImplementedError):
            module = None
        weight = getattr(module, "weight", None)
        if (isinstance(weight, nn.Parameter) and weight.dtype in _SIXTEEN_BIT
                and type(weight).__name__ != "Params4bit"
                and not any(weight is seen for seen in found)):
            found.append(weight)
    return found


def cast_to(t: torch.Tensor, device: torch.device, dtype: torch.dtype,
            out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``t.to(device, dtype)`` for a 16 -> 32-bit cast, made on ``device`` inside the result.

    ``Tensor.to`` with both a device and a dtype casts a host tensor on the host first, a full
    fp32 copy there. Here ``t``'s bytes cross once, into the front half of the result's own
    storage, and are widened in place from the last element back: a chunk ``[s, e)`` is read
    from bytes below where it is written once ``s >= e - s``, so no kernel reads what it writes
    and nothing beyond the result is allocated. An exact cast is the same on any device, so the
    result equals ``t.to(device, dtype)`` bit for bit (a NaN stays a NaN). ``out``: the result's
    storage, allocated by the caller (``t``'s shape, ``dtype``, on ``device``).
    """
    if t.dtype == dtype:  # nothing to cast: the plain copy
        if out is not None:
            return out.copy_(t, non_blocking=True)
        return t.to(device, non_blocking=True)
    if out is None:
        out = torch.empty(t.shape, dtype=dtype, device=device)
    n = t.numel()
    if n == 0:
        return out
    if out.element_size() != 2 * t.element_size():
        raise ValueError(f"cast_to widens 16 to 32 bits, not {t.dtype} to {dtype}")
    dst = out.view(-1)
    src = dst.view(t.dtype)[:n]  # the result's storage, read as t's dtype
    src.copy_(t.reshape(-1), non_blocking=True)
    chunk = max(1, CHUNK_BYTES // t.element_size())
    end = n
    while end > 1:
        start = end - min(chunk, end // 2)
        dst[start:end].copy_(src[start:end])
        end = start
    dst[:1].copy_(src[:1].clone())  # the first element's bytes are its own destination's
    return out


def embed_as(embed: nn.Module, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``embed(input_ids)`` as if ``embed``'s table were in ``dtype``: only the rows it reads are
    cast, and ``embed``'s own forward runs on them (a scaled embedding scales in ``dtype``, as it
    would). ``padding_idx`` is set aside for the call: it shapes only the gradient of a lookup,
    never its value, and the small table has none of the vocabulary's indices."""
    weight = embed.weight
    if weight.dtype == dtype:
        return embed(input_ids)
    if getattr(embed, "max_norm", None) is not None:  # renorms its own table in place
        raise ValueError("a compact embedding table cannot renormalize (max_norm)")
    rows = weight.index_select(0, input_ids.reshape(-1)).to(dtype)
    local = torch.arange(rows.shape[0], device=input_ids.device).view(input_ids.shape)
    if not hasattr(embed, "padding_idx"):
        return torch.func.functional_call(embed, {"weight": rows}, (local,))
    pad, embed.padding_idx = embed.padding_idx, None
    try:
        return torch.func.functional_call(embed, {"weight": rows}, (local,))
    finally:
        embed.padding_idx = pad
