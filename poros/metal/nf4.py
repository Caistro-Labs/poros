"""NF4 frozen linears on the Mac: weights live in a layer's slab, and every use decodes them.

``Nf4Linear`` is an ``nn.Linear`` with an empty weight, so PEFT's dispatch takes it and the
adapter it trains loads on vanilla weights. Nothing keeps a decoded matrix: a graph over any
number of layers stays small, at the cost of one more decode per matrix per microstep.
"""

from __future__ import annotations

from functools import cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from poros.metal import shaders
from poros.moe import quant


@cache
def _codebooks() -> tuple[torch.Tensor, torch.Tensor]:
    return quant.dynamic_codebook().to("mps"), quant.nf4_codebook().to("mps")


def _kernel_decode(out: torch.Tensor, slab: torch.Tensor, start: int, n: int) -> None:
    """One dispatch into ``out``, the blob's sections as views into the slab."""
    packed_b, nblocks, nabsmax2 = quant._section_sizes(n)
    pa = start + quant._align4(packed_b)
    a2 = pa + quant._align4(nblocks)
    off = a2 + 4 * nabsmax2
    shaders.library().nf4_decode_bf16(
        out,
        slab[start : start + packed_b],
        slab[pa : pa + nblocks],
        slab[a2:off],
        slab[off : off + 4],
        *_codebooks(),
        n,
        threads=[nblocks],
    )


class Nf4Matmul(torch.autograd.Function):
    """The frozen GEMM. It saves no weights, so a whole-model graph stays small. Forward and VJP
    are the pinned route: ``F.linear`` forward, ``torch.mm`` VJP."""

    @staticmethod
    def forward(ctx, x2d: torch.Tensor, lin: Nf4Linear) -> torch.Tensor:
        ctx.lin = lin
        return F.linear(x2d, lin.decoded())

    @staticmethod
    @once_differentiable
    def backward(ctx, g: torch.Tensor):
        return torch.mm(g, ctx.lin.decoded()), None


class Nf4Linear(nn.Linear):
    """A frozen linear with an empty weight. Input and output are fp32, the math is bf16.

    ``slab`` is its layer's bytes on the device, set by whoever made them current, and its blob
    sits at ``start``. ``scratch`` is the one bf16 buffer every matrix decodes into, at offset 0.
    ``raw``: its blob is plain bf16, as a MoE model's dense layers are, and "decoding" is a copy.
    """

    compute_dtype = torch.bfloat16  # read by PEFT: the same LoRA init path as bnb

    def __init__(self, in_features: int, out_features: int, *, scratch: torch.Tensor, start: int,
                 bias: torch.Tensor | None = None, raw: bool = False):
        nn.Module.__init__(self)
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.scratch, self.start, self.slab, self.raw = scratch, start, None, raw

    def decoded(self) -> torch.Tensor:
        """This matrix in the scratch: by the kernel on ``mps``, else on the CPU. Valid until the
        next decode."""
        n = self.in_features * self.out_features
        out = self.scratch[:n]
        if self.raw:
            out.view(torch.uint8).copy_(self.slab[self.start : self.start + 2 * n])
        elif out.device.type == "mps":
            _kernel_decode(out, self.slab, self.start, n)
        else:
            blob = self.slab[self.start : self.start + quant.quantized_nbytes((n,))]
            out.copy_(quant.dequantize(blob, (n,), torch.bfloat16))
        return out.view(self.out_features, self.in_features)

    def matmul(self, x2d: torch.Tensor) -> torch.Tensor:
        return Nf4Matmul.apply(x2d, self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.matmul(x.to(torch.bfloat16).reshape(-1, x.shape[-1]))
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], -1).to(x.dtype)
