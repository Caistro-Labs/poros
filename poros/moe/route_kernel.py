"""Row metadata and group boundaries of a stable expert sort, in one CUDA pass."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _segment(row, segment_ids, MODE: tl.constexpr, segment_width):
    if MODE == 0:  # one segment
        return tl.full(row.shape, 0, tl.int32)
    if MODE == 1:  # equally long segments
        return row // segment_width
    return tl.load(segment_ids + row).to(tl.int32)


@triton.jit(do_not_specialize=["rows", "experts", "nseg", "segment_width"])
def _route_rows(sorted_ids, order, segment_ids, dest, row_tok, row_seg, ends,
                rows, experts, nseg, segment_width, TOP_K: tl.constexpr, MODE: tl.constexpr,
                COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(order + i, mask=i < rows, other=0).to(tl.int32)
    seg = _segment(row, segment_ids, MODE, segment_width)
    tl.store(dest + i, row, mask=i < rows)
    tl.store(row_tok + i, row // TOP_K, mask=i < rows)
    if MODE != 0:
        tl.store(row_seg + i, seg.to(tl.int64), mask=i < rows)

    if COUNT:
        e = tl.load(sorted_ids + i, mask=i < rows, other=0)
        has_next = i < rows - 1
        next_e = tl.load(sorted_ids + i + 1, mask=has_next, other=0)
        last = (e != next_e) | ~has_next
        if MODE != 0:
            next_row = tl.load(order + i + 1, mask=has_next, other=0).to(tl.int32)
            last |= seg != _segment(next_row, segment_ids, MODE, segment_width)
        # Stable sorting orders rows by (expert, segment). Exactly one lane owns each
        # nonempty group's exclusive end, including a run crossing program boundaries.
        group = e.to(tl.int64) * nseg + seg
        tl.store(ends + group, i + 1, mask=(i < rows) & last & (e >= 0) & (e < experts))
        # The extremal sorted ids validate the entire input in the same host transfer.
        if tl.program_id(0) == 0:
            flags = ends + experts.to(tl.int64) * nseg
            tl.store(flags, tl.load(sorted_ids) < 0)
            tl.store(flags + 1, tl.load(sorted_ids + rows - 1) >= experts)


def route_rows(sorted_ids, order, top_k, experts, segments, *, count):
    """Return sorted row metadata and, when requested, the counts on the host."""
    from poros.moe.fused import _segment_of_rows

    rows = order.numel()
    device = order.device
    nseg = len(segments) if segments else 1
    mode, width, segment_ids = 0, 1, order
    if nseg > 1:
        equal = all(n == segments[0] for n in segments)
        mode, width = (1, max(1, segments[0] * top_k)) if equal else (2, 1)
        if not equal:
            segment_ids = _segment_of_rows(segments, top_k, rows, device)
    dest = torch.empty(rows, device=device, dtype=torch.int32)
    row_tok = torch.empty_like(dest)
    row_seg = torch.empty_like(order) if mode else None
    ends = torch.zeros(experts * nseg + 2, device=device, dtype=torch.int32) if count else None
    if rows:
        _route_rows[(triton.cdiv(rows, 256),)](
            sorted_ids, order, segment_ids, dest, row_tok, row_seg if mode else dest,
            ends if count else dest, rows, experts, nseg, width, TOP_K=top_k, MODE=mode,
            COUNT=count, BLOCK=256)
    counts = None
    if count:
        import numpy as np

        marks = ends.cpu().numpy()
        if marks[-2]:
            raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
        if marks[-1]:
            raise ValueError(f"an expert id at or above num_experts={experts}")
        # Empty groups inherit the previous exclusive end. Differencing also handles
        # leading/trailing empty groups, and exactly reconstructs the reference int64 counts.
        bounds = np.maximum.accumulate(marks[:-2])
        counts = torch.from_numpy(np.diff(bounds, prepend=np.int64(0)))
    return dest, row_tok, row_seg, counts


@triton.jit
def _transpose_runs(starts, ends, dest_starts, permutation, BLOCK: tl.constexpr):
    """A contiguous group moves as a unit from expert-major to segment-major row order."""
    g = tl.program_id(0)
    start, end = tl.load(starts + g), tl.load(ends + g)
    dest = tl.load(dest_starts + g)
    # All groups here are nonempty. Subtract before dividing to stay in int32.
    for tile in range(0, (end - start - 1) // BLOCK + 1):
        j = tile * BLOCK + tl.arange(0, BLOCK)
        tl.store(permutation + dest + j, (start + j).to(tl.int64), mask=j < end - start)


def transpose_runs(starts, ends, dest_starts, rows):
    permutation = torch.empty(rows, device=starts.device, dtype=torch.int64)
    if starts.numel():
        _transpose_runs[(starts.numel(),)](starts, ends, dest_starts, permutation, BLOCK=128)
    return permutation
