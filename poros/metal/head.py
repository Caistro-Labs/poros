"""Embed and head: bf16 rows at rest, widened to fp32 on the CPU, with no standing memory.

Rows are any bf16 matrix indexed by a slice or by ids: a tensor on the RAM tier, ``FileRows`` on
the SSS tier. Stored bf16 is never named ``weight``. ``trainer.py`` calls
``next(embed.parameters())`` and casts hidden to ``lm_head.weight.dtype``, so both modules hold a
zero-size ``weight`` of the dtype activations have. That is fp32, as bitsandbytes leaves a dense
model: a bf16 one would round hidden to bf16 in every arm, and no comparison between arms could see
it. A MoE model computes in bf16, as it does on CUDA, and its logits are fp32 only for the loss.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from poros.sss.format import SssCorruptError, SssFile

CHUNK_ROWS = 16384
_PIECE = 64 << 20  # macOS refuses a read over 2 GiB, and a large model's table is more


class FileRows:
    """A bf16 matrix in one slab of a ``.sss``, read on demand with ``pread``.

    Its checksum is verified on first touch, in pieces, so the matrix is never whole in memory.
    """

    def __init__(self, file: SssFile, key: tuple[int, int]):
        self.file, self.entry = file, file.entry(key)
        self.shape = tuple(self.entry.views[0]["shape"])
        self.verified = False

    def _verify(self) -> None:
        e, h = self.entry, hashlib.blake2b(digest_size=8)
        for pos in range(e.offset, e.offset + e.nbytes, _PIECE):
            h.update(os.pread(self.file.fd, min(_PIECE, e.offset + e.nbytes - pos), pos))
        if h.hexdigest() != e.checksum:
            raise SssCorruptError(f"slab {e.key} in {self.file.path} fails its checksum")
        self.verified = True

    def __getitem__(self, index: slice | torch.Tensor) -> torch.Tensor:
        if not self.verified:
            self._verify()
        if isinstance(index, slice):
            runs = [(index.start, index.stop - index.start)]
        else:
            runs = [(row, 1) for row in index.tolist()]
        hidden = self.shape[1]
        out = torch.empty(sum(n for _, n in runs), hidden, dtype=torch.bfloat16)
        dst = memoryview(out.view(torch.uint8).numpy()).cast("B")
        for row, n in runs:
            pos, left = self.entry.offset + row * hidden * 2, n * hidden * 2
            while left:
                got = os.preadv(self.file.fd, [dst[:min(left, _PIECE)]], pos)
                if got <= 0:
                    raise SssCorruptError(f"short read of row {row} in {self.file.path}")
                dst, pos, left = dst[got:], pos + got, left - got
        return out


class EmbedRows(nn.Module):
    """Input embeddings: the unique token rows, widened on the CPU, then one blit. Exact. A model
    that scales its embeddings, as Gemma does, has them scaled here, in the fp32 it computes in."""

    def __init__(self, rows, *, scale: float | None = None):
        super().__init__()
        self.rows, self.scale = rows, scale
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        unique, inverse = input_ids.cpu().unique(return_inverse=True)
        found = self.rows[unique].to(self.weight.dtype)
        if self.scale is not None:
            found = found * self.scale
        return found[inverse].to(input_ids.device)


class ChunkedHead(nn.Module):
    """The LM head: an fp32 GEMM per chunk of rows. Hidden stays fp32 at the head."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.out_features = rows, rows.shape[0]
        self.spans = [
            (a, min(a + CHUNK_ROWS, self.out_features))
            for a in range(0, self.out_features, CHUNK_ROWS)
        ]
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)

    def chunk(self, start: int, stop: int, device: torch.device) -> torch.Tensor:
        return self.rows[start:stop].to(self.weight.dtype).to(device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = _ChunkedHeadFn.apply(hidden.reshape(-1, hidden.shape[-1]), self)
        return logits.reshape(*hidden.shape[:-1], -1)


class _ChunkedHeadFn(torch.autograd.Function):
    """Holds no chunk between its forward and its backward: each one is read again."""

    @staticmethod
    def forward(ctx, h2d: torch.Tensor, head: ChunkedHead) -> torch.Tensor:
        ctx.head = head
        out = h2d.new_empty(h2d.shape[0], head.out_features)
        for a, b in head.spans:
            out[:, a:b] = F.linear(h2d, head.chunk(a, b, h2d.device))
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, g: torch.Tensor):
        # Descending, because that is what autograd through torch.cat sums, so the plain arm
        # matches in bytes. Contiguous, because torch's own comment says mm on a view of a
        # matrix wider than 2^15 was silently wrong or crashed (ops/LinearAlgebra.mm:1171-1172).
        acc = None
        for a, b in reversed(ctx.head.spans):
            p = torch.mm(g[:, a:b].contiguous(), ctx.head.chunk(a, b, g.device))
            acc = p if acc is None else acc + p
        return acc, None
