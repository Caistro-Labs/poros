"""A process-wide budget for pinned (page-locked) host memory.

Some hosts cap page-locked memory far below their RAM: a cloud RTX 3090 container
container (WSL2) refuses a second GiB, and a process that pins up to that
ceiling then fails *later, unrelated* CUDA launches ("too many resources
requested for launch"), because the driver has no page-locked memory left
for itself. So every large pin in the streaming paths asks ``reserve``
first and falls back to pageable memory (same bytes, synchronous copies)
when the answer is no.

``reserve`` verifies headroom before granting: it registers the bytes it is
about to grant plus ``HEADROOM`` with ``cudaHostRegister`` on a scratch
mapping and releases them at once. Once a verification fails nothing more is
pinned in the process (on a cloud 3090 container even a quarter of the ceiling
broke later pageable copies). Under WSL2 this process never registers to probe:
a child process pins 4 GiB first (``_wsl_child_probe``). ``POROS_PIN_BUDGET`` (e.g. ``1G``;
``0`` disables pinning) skips the probing. ``status()`` reports what was granted and why,
for receipts and the plan.
"""

from __future__ import annotations

import logging
import mmap
import os
import threading
import weakref

logger = logging.getLogger(__name__)

__all__ = ["HEADROOM", "PinnedArena", "pinned_size", "release", "reserve", "reset", "status"]

HEADROOM = 512 << 20
_STEP = 1 << 30

_lock = threading.RLock()
_state: dict = {}


def reset() -> None:
    """Forget every grant and verification (tests)."""
    with _lock:
        _state.clear()
        _state.update(granted=0, verified=0, capped=False, refusals=0, env=_env_budget(),
                      wsl=_wsl())


def _wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


# Under WSL2 the ceiling is host-specific: one cloud 3090 container refused a second GiB,
# another pinned 44 GiB at 12.0 GB/s against 5.2 pageable.
# A failed registration poisons the process that tried it, so the probe runs in a child.
WSL_PROBE_BYTES = 4 << 30
_wsl_probe_result: bool | None = None

_PROBE_CODE = r"""
import sys, torch
n = int(sys.argv[1]) // (256 << 20)
chunks = [torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True) for _ in range(n)]
x = torch.ones(64 << 20, dtype=torch.uint8)            # a pageable copy after pinning
y = x.to("cuda"); m = torch.randn(512, 512, device="cuda")  # and a kernel launch
assert int(y.sum().item()) == x.numel() and torch.isfinite((m @ m).sum()).item()
print("PIN_OK", n)
"""


def _wsl_child_probe(nbytes: int = WSL_PROBE_BYTES) -> bool:
    """Pin ``nbytes`` in a throwaway child process, then copy and launch a kernel there."""
    import subprocess
    import sys

    try:
        # -P: the working folder is not put on the child's path (no planted torch.py runs)
        r = subprocess.run([sys.executable, "-P", "-c", _PROBE_CODE, str(int(nbytes))],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and "PIN_OK" in r.stdout


def _ram_budget() -> int:
    """Pinnable bytes: available host RAM minus max(4 GiB, 15% of total)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return max(0, int(vm.available - max(4 << 30, int(vm.total * 0.15))))
    except Exception:
        return 0


def _env_budget() -> int | None:
    """``POROS_PIN_BUDGET`` if set; under WSL2 a child-probed budget; else None (verify by
    registration).

    WSL2 can cap page-locked memory near 1-1.5 GiB, and there a *failed*
    registration leaves the process unable to make pageable copies ("CUDA
    error: out of memory" in ``model.to``, measured on a cloud 3090
    container), so this process never probes under WSL2. A child process pins
    ``WSL_PROBE_BYTES`` instead; if it succeeds the budget is the host's free
    RAM less a reserve (``_ram_budget``), else nothing is pinned.
    ``POROS_WSL_PIN_PROBE=0`` skips the child and pins nothing.
    """
    global _wsl_probe_result
    raw = os.environ.get("POROS_PIN_BUDGET")
    if raw is None or raw.strip() == "":
        if not _wsl():
            return None
        if os.environ.get("POROS_WSL_PIN_PROBE", "1") == "0" or not _cuda():
            _wsl_probe_result = False
            return 0
        if _wsl_probe_result is None:
            _wsl_probe_result = _wsl_child_probe()
            if not _wsl_probe_result:
                logger.warning(
                    "WSL2: a child process could not pin %d GiB of host memory, so nothing is "
                    "pinned in this process: host copies stay pageable (same bytes, synchronous "
                    "copies); the receipt's host_staging says so", WSL_PROBE_BYTES >> 30)
        return _ram_budget() if _wsl_probe_result else 0
    from poros.core.config import parse_bytes
    return parse_bytes(raw)


def pinned_size(nbytes: int) -> int:
    """What a ``pin_memory`` allocation of ``nbytes`` really locks: torch's caching
    host allocator rounds each block up to a power of two."""
    n = max(int(nbytes), 1)
    return 1 << (n - 1).bit_length()


def _register_ok(nbytes: int) -> bool:
    """Can ``nbytes`` more be page-locked right now? (registered, then released)."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    rt = torch.cuda.cudart()
    # Private anonymous memory, its pages faulted in by torch's threads before registering:
    # mmap(-1, n) alone is shared memory, which cudaHostRegister faults in one page at a time
    # (a 1.5 GiB probe 1.94 s on an RTX 4090 host; 1.18 s private, untouched). Same bytes.
    # Windows has no MAP_PRIVATE (nor a flags argument): its anonymous mapping as before.
    private = getattr(mmap, "MAP_PRIVATE", None)
    mm = mmap.mmap(-1, nbytes) if private is None else mmap.mmap(-1, nbytes, flags=private)
    t = torch.frombuffer(mm, dtype=torch.uint8)
    try:
        t.fill_(0)
        ok = int(rt.cudaHostRegister(t.data_ptr(), nbytes, 0)) == 0
        if ok:
            rt.cudaHostUnregister(t.data_ptr())
        return ok
    finally:
        del t
        try:
            mm.close()
        except BufferError:  # still exported; freed with the last reference
            pass


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def reserve(nbytes: int) -> bool:
    """Grant ``nbytes`` of page-locked memory to the caller, or refuse (use pageable)."""
    nbytes = int(nbytes)
    with _lock:
        if not _state:
            _state.update(granted=0, verified=0, capped=False, refusals=0, env=_env_budget(),
                      wsl=_wsl())
        granted = _state["granted"]
        if _state["env"] is not None:
            ok = granted + nbytes <= _state["env"]
        elif granted + nbytes <= _state["verified"]:
            ok = True
        elif _state["capped"] or not _cuda():
            ok = False
        else:
            want = max(granted + nbytes, _state["verified"] + _STEP)
            if _register_ok(want - granted + HEADROOM):
                _state["verified"] = want
                ok = True
            else:  # a ceiling: pin nothing more in this process
                # Measured on a cloud 3090 container (WSL2): with a ceiling near 1.5 GiB, any
                # further pinning -- even a quarter of it -- left the process unable to
                # make pageable copies ("CUDA error: out of memory" in model.to). So a
                # ceiling ends pinning; the bytes and the numerics do not change.
                _state["verified"] = granted
                _state["capped"] = True
                _state["ceiling_below"] = want - granted + HEADROOM
                logger.warning(
                    "page-locked host memory is capped on this host (%.2f GiB could not be "
                    "pinned): nothing more is pinned in this process; stores fall back to "
                    "pageable memory (same bytes, synchronous copies)",
                    (want - granted + HEADROOM) / 2**30,
                )
                ok = granted + nbytes <= _state["verified"]
        if ok:
            _state["granted"] = granted + nbytes
        else:
            _state["refusals"] += 1
        return ok


def release(nbytes: int) -> None:
    """Return a grant (the pinned memory was freed or unregistered)."""
    with _lock:
        if _state:
            _state["granted"] = max(0, _state["granted"] - int(nbytes))


def _freed(nbytes: int) -> None:
    """A page-locked tensor from torch's host allocator is gone: its block goes back to the
    driver (torch would keep it cached, and locked) and its grant to the budget."""
    try:
        import torch

        empty = getattr(torch._C, "_accelerator_emptyHostCache", None) or getattr(
            torch._C, "_host_emptyCache", None)
        if empty is not None:
            empty()
    except Exception:  # at interpreter exit torch may be half torn down: the grant still returns
        pass
    release(nbytes)


def staging(st: dict | None = None, *, ring_pinned: bool | None = None) -> dict:
    """The staging mode a run used and the pinned budget behind it (for receipts).

    ``mode`` is ``pinned`` (every pin request granted), ``partial`` (some
    refused: the rest is pageable) or ``unpinned`` (nothing pinned). The SSD
    ring's own state is passed as ``ring_pinned`` when there is one.
    """
    st = status() if st is None else st
    if st.get("env") is not None:
        budget = st["env"]
        if st.get("wsl") and "POROS_PIN_BUDGET" not in os.environ:
            basis = ("WSL2 (child probe pinned %d GiB; free RAM less reserve)"
                     % (WSL_PROBE_BYTES >> 30)
                     if _wsl_probe_result else "WSL2 (child probe failed or skipped)")
        else:
            basis = "POROS_PIN_BUDGET"
    elif st.get("capped"):
        budget, basis = st["verified"], "probed ceiling minus headroom"
    else:
        budget, basis = None, "no ceiling found"
    granted, refused = st.get("granted", 0), st.get("refusals", 0)
    if ring_pinned is False or (granted == 0 and refused):
        mode = "unpinned" if not granted else "partial"
    elif refused:
        mode = "partial"
    else:
        mode = "pinned" if granted or ring_pinned else "unpinned"
    return {"mode": mode, "pinned_budget_bytes": budget, "budget_basis": basis,
            "granted_bytes": granted, "refusals": refused}


def status() -> dict:
    """Granted bytes, the verified budget, whether a ceiling was found, refusals."""
    with _lock:
        if not _state:
            return {"granted": 0, "verified": 0, "capped": False, "refusals": 0, "env": None}
        return dict(_state)


class PinnedArena:
    """Page-locked copies of many tensors, carved from shared slabs.

    torch's host allocator rounds every ``pin_memory`` block up to a power of two, and the
    driver locks all of it: a 67.5 MiB NF4 weight (one of Qwen2.5-32B's MLP projections) locks
    128 MiB, so pinning a model's weights one by one can lock up to twice their size. Here each
    tensor is copied into the first slab with room for it. A slab is ``SLAB`` bytes, or, given
    ``total`` (the bytes the arena will be asked for), the power of two that covers what is
    still to come when that is less: every size is one torch locks exactly, and what is locked
    is what is stored plus little more than one slab's tail. Each slab is granted by
    ``reserve``; a tensor over half a slab, or one the budget will not grant a whole slab for,
    is pinned alone, as before. Once the host refuses an allocation (``refused``), nothing more
    is pinned. The bytes are the same; only where they sit changes. A slab lives as long as any
    tensor carved from it.
    """

    SLAB = 1 << 30
    ALIGN = 512

    def __init__(self, total: int | None = None, *, release_on_free: bool = False) -> None:
        self._slabs: list = []  # [slab, bytes used]
        self._left = total
        self.refused = False  # the host failed a page-locked allocation: stop asking
        # Give each grant back when its slab's (or lone tensor's) memory is freed: the finalizer
        # watches the storage, which lives as long as any view of it does (a copy, a view of a
        # copy, a tensor its memory was moved into), so a grant never returns while its pages
        # are still locked. Off for the arenas whose copies become parameters for good.
        self._release_on_free = release_on_free

    def _granted(self, t, nbytes: int):
        if self._release_on_free:
            weakref.finalize(t.untyped_storage(), _freed, nbytes)
        return t

    def pin(self, t):
        """A page-locked copy of ``t``, or ``t`` itself where the budget or the host says no."""
        nbytes = t.numel() * t.element_size()
        if nbytes == 0 or not t.is_contiguous() or t.is_pinned() or self.refused:
            return t
        left, self._left = self._left, None if self._left is None else max(0, self._left - nbytes)
        if nbytes > self.SLAB // 2:
            return self._alone(t, nbytes)
        entry = next((e for e in self._slabs if e[1] + nbytes <= e[0].numel()), None)
        if entry is None:
            size = self.SLAB if left is None else min(self.SLAB, pinned_size(max(nbytes, left)))
            if not reserve(size):  # the budget's tail: this tensor alone, as one by one would
                return self._alone(t, nbytes) if size > pinned_size(nbytes) else t
            try:
                entry = [self._granted(self._new_slab(size), size), 0]
            except RuntimeError:
                release(size)
                self.refused = True
                return t
            self._slabs.append(entry)
        slab, used = entry
        out = slab[used:used + nbytes].view(t.dtype).view(t.shape)
        entry[1] = used + -(-nbytes // self.ALIGN) * self.ALIGN
        out.copy_(t)
        return out

    def _alone(self, t, nbytes: int):
        """``t`` pinned by itself, granted its own power of two."""
        granted = pinned_size(nbytes)
        if not reserve(granted):
            return t
        try:
            return self._granted(t.pin_memory(), granted)
        except RuntimeError:
            release(granted)
            self.refused = True
            return t

    def _new_slab(self, nbytes: int):
        import torch

        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
