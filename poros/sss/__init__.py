"""Solid State Streaming (SSS): one slab store, two backends.

Both Poros streaming paths move flat uint8 slabs keyed by position through
one host-to-device seam: the MoE path one slab per ``(moe_layer, expert)``,
the dense path one packed slab per block. SSS puts a single byte-source
interface under that seam so that host RAM and an SSD file are a backend
swap, never a numerics change:

- ``SlabSource`` / ``RamSource`` (``source``): ``read_into(key, dst)`` is a
  pure byte copy; ``RamSource`` is exactly today's pinned-RAM behaviour.
- ``SlabSink`` / ``RamSink`` / ``SssWriter`` and ``SssFile`` (``format``):
  the on-disk ``model.sss`` + ``model.sss.idx`` pair, written once
  (temp + fsync + rename) and reused while its metadata matches.
- ``SyncFileSource`` (``format``): synchronous buffered reads of a ``.sss``
  file, the reference reader.
- ``SsdSource`` (``ssd``): the streaming reader -- O_DIRECT reads from a
  thread pool into a pinned chunk ring, with schedule replay.
- ``GatherSource`` / ``GatherIndex`` (``gather``): the same reader over slabs
  that already sit, uncompressed, in a model's own files: a
  sidecar index of file ranges per slab instead of a ``.sss`` copy.
- ``probe`` / ``tune``: disk and H2D bandwidth probe, prefetch depth and
  VRAM cache sizing (torch-free).

Bytes in == bytes out is the whole contract: every ``.sss`` slab carries a
checksum taken at write time, and the bit-exact gate compares sources with
``torch.equal``.

All exports are lazy (PEP 562): importing ``poros.sss`` or
``poros.sss.probe`` pulls no torch, so the plan can probe a disk.
"""

from importlib import import_module
from typing import Any

_LAZY = {
    **{name: "poros.sss.format" for name in (
        "ALIGNMENT", "CHECKSUM_ALGO", "FORMAT_VERSION", "RamSink", "SlabEntry", "SlabSink",
        "SssCorruptError", "SssFile", "SssFormatError", "SssWriter", "SyncFileSource",
        "slab_checksum")},
    **{name: "poros.sss.source" for name in ("RamSource", "SlabKey", "SlabSource", "check_dst")},
    "SsdSource": "poros.sss.ssd",
    **{name: "poros.sss.gather" for name in (
        "GatherIndex", "GatherIndexStale", "GatherSlab", "GatherSource", "Segment")},
    "ProbeResult": "poros.sss.probe",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
