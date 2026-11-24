"""Digests and counts from tensors, for gate transcripts. See ``poros.metal.verify`` for the format.

Works from a CPU copy, so it adds no GPU kernels. ``.cpu()`` on an ``mps`` tensor does sync the
stream, which is one reason capture runs in gate arms only and the ``observer`` gate exists.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import torch

from poros.metal.verify import TRANSCRIPT_SCHEMA, digest_bytes, digest_chain


def _on_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous()


def digest_tensor(t: torch.Tensor) -> str:
    """Byte digest over dtype|shape|little-endian bytes. Numeric equality plays no part."""
    c = _on_cpu(t)
    # Through uint8, because numpy has no bf16.
    data = c.flatten().view(torch.uint8).numpy()
    return digest_bytes(str(c.dtype).removeprefix("torch."), c.shape, data)


def _leaves(obj, name: str):
    """(name, leaf) in a fixed order, for a tensor or a nested state dict. Anything else raises:
    a value this cannot digest must not vanish from the comparison."""
    if isinstance(obj, torch.Tensor) or obj is None or isinstance(obj, (bool, int, float, str)):
        yield name, obj
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            yield from _leaves(obj[key], f"{name}/{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from _leaves(value, f"{name}/{i}")
    else:
        raise TypeError(f"cannot digest {type(obj).__name__} at {name!r}")


class Surface(dict):
    """A captured surface. Capturing it again changes nothing, so a digest can be taken when the
    tensor is alive and written later."""

    def __init__(self, b2: str, nonfinite_count: int, elements: int, nonzero: int):
        super().__init__(
            b2=b2, nonfinite_count=nonfinite_count, elements=elements, nonzero=nonzero
        )


def _scalar(x) -> Surface:
    """A Python number or string in a state dict."""
    if x is None or isinstance(x, str):
        return Surface(digest_bytes(type(x).__name__, (), (x or "").encode()), 0, 0, 0)
    if isinstance(x, int):  # bool too, under its own name
        return Surface(digest_bytes(type(x).__name__, (), str(int(x)).encode()), 0, 1, int(x != 0))
    # The float's own bits, so -0.0 and 0.0 differ here too.
    bits = struct.pack("<d", x)
    return Surface(digest_bytes("float", (), bits), int(not math.isfinite(x)), 1, int(x != 0))


def capture(obj) -> Surface:
    """One surface of a transcript record, from a tensor or a nested dict or list of them.

    A dict's entries are hashed in sorted name order, with their names, so a renamed tensor
    differs. Non-finite elements count as such, and a NaN as nonzero. -0.0 counts as zero.
    """
    if isinstance(obj, Surface):
        return obj
    if isinstance(obj, torch.Tensor):
        c = _on_cpu(obj)
        nonfinite = c.numel() - int(torch.isfinite(c).sum())
        return Surface(digest_tensor(c), nonfinite, c.numel(), int(torch.count_nonzero(c)))
    leaves = [
        (name, capture(leaf) if isinstance(leaf, torch.Tensor) else _scalar(leaf))
        for name, leaf in _leaves(obj, "")
    ]
    return Surface(
        digest_chain(f"{name}={s['b2']}" for name, s in leaves),
        *(sum(s[key] for _, s in leaves) for key in ("nonfinite_count", "elements", "nonzero")),
    )


class TranscriptWriter:
    """Writes one arm's transcript, a line at a time, so a crash leaves a short file that fails."""

    def __init__(self, path, *, arm, gate, model, seq, g, steps, lora_dropout, layers,
                 expected_records, pin, first_microstep=0):
        self._f = Path(path).open("w", encoding="utf-8")
        self._write({
            "schema": TRANSCRIPT_SCHEMA, "arm": arm, "gate": gate, "model": model, "seq": seq,
            "g": g, "steps": steps, "first_microstep": first_microstep,
            "lora_dropout": lora_dropout, "layers": layers,
            "expected_records": expected_records, "pin": pin,
        })

    def _write(self, record: dict) -> None:
        self._f.write(json.dumps(record, allow_nan=False) + "\n")
        self._f.flush()

    def layer(self, microstep, layer, *, rng, input, output, cotangent, lora_grad) -> None:
        self._write({
            "kind": "layer", "microstep": microstep, "layer": layer, "rng": capture(rng),
            "input": capture(input), "output": capture(output),
            "cotangent": capture(cotangent), "lora_grad": capture(lora_grad),
        })

    def microstep(self, microstep, *, loss, logits, grads) -> None:
        self._write({
            "kind": "microstep", "microstep": microstep, "loss": capture(loss),
            "logits": capture(logits), "grads": capture(grads),
        })

    def final(self, microsteps, *, adapter, adam, scheduler) -> None:
        self._write({
            "kind": "final", "microsteps": microsteps, "adapter": capture(adapter),
            "adam": capture(adam), "scheduler": capture(scheduler),
        })

    def close(self) -> None:
        self._f.close()
