"""The card: what this NVIDIA GPU is and what it can run, in one place.

* **Capability** (``capability``): read from the device (torch, or the driver for a card that is
  only planned for: ``detect_gpu`` / ``visible_gpus``). A name maps to a capability only to plan
  for a card that is not here (``capability_of_name``: a card named with ``--gpu``).
* **The supported range** (``check``, ``refusal``, ``bf16_reason``, ``build_reason``,
  ``triton_reason``, ``class_name``): Poros measured its paths on Ampere and newer (bf16-capable)
  cards. A torch build with no kernels for this GPU (an old wheel on Blackwell, a CUDA 13 wheel on
  a pre-Turing card) is refused in seconds, never a "no kernel image" crash later; bf16 compute on
  a card without native bf16 (Volta sm_70, Turing sm_75 and older) is refused, as Poros supports
  bf16 cards only (Ampere sm_80 / 86, Ada sm_89, Hopper sm_90, Blackwell sm_100 / 103 / 120: NF4's
  bitwise path computes in bf16), ``torch_dtype: float16`` named as an unsupported, non-bitwise
  opt-in, not as the fix; the fused compute's Triton kernels on a card Triton does not compile for
  (pre-Volta): the fused compute stays off, the reason recorded (``poros.core.fused.available``,
  ``RunCompute.moe_request``).
* **Shared memory** (``shared_memory_limit``): the per-block ceiling the fused kernels' launch
  configurations are checked against (``poros.core.fused.kernels``).
* **Rank fit** (``FUSED_FIT``, ``fused_rank_reason``): the largest LoRA rank whose fused kernels
  fit each architecture, from the ahead-of-time compiles; a card's own run confirms it by
  launching them (``poros.core.fused.fit``).
* **Flash attention** (``flash_usable``): whether the bf16 flash kernel takes a head size on this
  device, forward and backward.
* **The cards here** (``detect_gpu``, ``visible_gpus``, ``card_refusal``, ``card_capability``,
  ``GPU_SPECS_MIB``): the GPU a plan or a run is for, asking nvidia-smi and the driver before
  torch.
* **A card that is not here** (``named_card``, ``CARD_GIB``, ``CARD_SIZES``): what ``--gpu``
  names, by a card's name, a memory size, or both.

Torch-free at import; ``check``, ``capability`` and ``flash_usable`` import torch when called.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import uuid
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

from poros.core.env import outside_tool
from poros.core.fused_tiles import pad

if TYPE_CHECKING:
    import torch

    from poros.core.config import PorosConfig
    from poros.core.schemas import PlanGpu

__all__ = ["CARD_GIB", "CARD_SIZES", "FUSED_FIT", "GPU_SPECS_MIB", "NATIVE_BF16", "TRITON_MIN",
           "CardNeedsSize", "UnknownCard", "bf16_reason", "build_reason", "capability",
           "capability_of_name", "card_capability", "card_refusal", "check", "class_name",
           "detect_gpu", "flash_usable", "fused_rank_reason", "named_card", "refusal",
           "shared_memory_limit", "triton_reason", "visible_gpus"]

Cap = Tuple[int, int]

NATIVE_BF16: Cap = (8, 0)  # Ampere: the first bf16 tensor cores
TRITON_MIN: Cap = (7, 0)  # Triton compiles for Volta and newer

# Max dynamic shared memory per block (opt-in), bytes, from the CUDA programming guide's
# per-capability table. A capability not listed takes its major's lowest listed figure.
_SHARED = {(7, 0): 98304, (7, 2): 98304, (7, 5): 65536, (8, 0): 166912, (8, 6): 101376,
           (8, 7): 166912, (8, 9): 101376, (9, 0): 232448, (10, 0): 232448, (10, 3): 232448,
           (11, 0): 232448, (12, 0): 101376, (12, 1): 101376}

_CLASSES = {(7, 0): "Volta", (7, 2): "Volta", (7, 5): "Turing", (8, 0): "Ampere",
            (8, 6): "Ampere", (8, 7): "Ampere", (8, 9): "Ada", (9, 0): "Hopper",
            (10, 0): "Blackwell", (10, 3): "Blackwell", (11, 0): "Blackwell",
            (12, 0): "Blackwell", (12, 1): "Blackwell"}


# A card's compute capability from its name (a plan's --gpu, or a card whose device cannot be
# asked), for a plan of a card that is not here: a local card's own capability is read from the
# device (``capability``, or the driver's for ``poros.core.card.detect_gpu``). Checked in order: the
# first pattern the name holds decides, so a workstation card's name comes before the series
# numbers its digits share ("Quadro RTX 5000" is Turing, not an RTX 50xx).
_NAMES: Tuple[Tuple[str, Cap], ...] = (
    ("QUADRO RTX", (7, 5)), ("TITAN RTX", (7, 5)), ("TITAN V", (7, 0)),
    ("GB300", (10, 3)), ("B300", (10, 3)), ("GB200", (10, 0)), ("B200", (10, 0)),
    ("B100", (10, 0)), ("RTX PRO", (12, 0)), ("RTX 50", (12, 0)), ("GH200", (9, 0)),
    ("H200", (9, 0)), ("H100", (9, 0)), ("H800", (9, 0)), ("H20", (9, 0)),
    ("RTX 40", (8, 9)), ("L40", (8, 9)), ("L4", (8, 9)),
    ("L20", (8, 9)), ("A100", (8, 0)), ("A800", (8, 0)), ("A30", (8, 0)),
    ("RTX 30", (8, 6)), ("RTX A2000", (8, 6)), ("RTX A4000", (8, 6)),
    ("RTX A4500", (8, 6)), ("RTX A5000", (8, 6)), ("RTX A5500", (8, 6)), ("RTX A6000", (8, 6)),
    ("A40", (8, 6)), ("A10", (8, 6)), ("A16", (8, 6)),
    ("A2", (8, 6)), ("V100", (7, 0)), ("T4", (7, 5)), ("RTX 20", (7, 5)), ("GTX 16", (7, 5)),
    ("GTX 10", (6, 1)), ("TITAN XP", (6, 1)), ("P100", (6, 0)), ("P40", (6, 1)), ("P4", (6, 1)),
)
# A generation word an RTX workstation name carries decides before any number in it: "RTX 2000
# Ada", "RTX 4000 SFF Ada", "RTX 5000 Ada Generation" are all Ada (sm_89).
_GENERATIONS: Tuple[Tuple[str, Cap], ...] = (("ADA", (8, 9)),)


def capability_of_name(name: Optional[str]) -> Optional[Cap]:
    """The compute capability a card of this name has, or None where the name does not say.
    A pattern matches whole words, its last word a word's start: "RTX 40" is any RTX 40-series,
    "A10" is an A10 or A10G but not an A100, "L4" is not an L40S."""
    if not name:
        return None
    words = name.upper().replace("-", " ").split()
    if "RTX" in words:
        for word, cap in _GENERATIONS:
            if word in words:
                return cap
    for pattern, cap in _NAMES:
        pw = pattern.split()
        for i in range(len(words) - len(pw) + 1):
            if words[i:i + len(pw) - 1] != pw[:-1]:
                continue
            w, last = words[i + len(pw) - 1], pw[-1]
            series = last.isdigit() and len(last) == 2  # RTX 40 -> 4090, 4080 ...
            if w == last or (w.startswith(last) and (series or not w[len(last)].isdigit())):
                return cap
    return None


def sm(cap: Cap) -> str:
    return f"sm_{cap[0]}{cap[1]}"


def class_name(cap: Cap) -> str:
    """Volta, Turing, Ampere, Ada, Hopper, Blackwell; "pre-Volta" below 7.0, else ``sm_XY``."""
    if tuple(cap) < (7, 0):
        return "pre-Volta"
    return _CLASSES.get(tuple(cap), sm(cap))


def shared_memory_limit(cap: Cap) -> int:
    """Max shared memory one block may take on ``cap`` (bytes)."""
    cap = tuple(cap)
    if cap in _SHARED:
        return _SHARED[cap]
    same = [v for k, v in _SHARED.items() if k[0] == cap[0]]
    return min(same) if same else min(_SHARED.values())


def capability(device: Any = None) -> Cap:
    import torch

    return tuple(torch.cuda.get_device_capability(device))  # type: ignore[return-value]


def _arch_num(arch: str) -> Optional[Tuple[str, int, int]]:
    """'sm_86' -> ('sm', 8, 6); 'compute_90' -> ('compute', 9, 0); 'sm_90a' -> None (an
    arch-specific build runs on that exact device only, never forward)."""
    kind, _, num = arch.partition("_")
    if kind not in ("sm", "compute") or not num.isdigit() or len(num) < 2:
        return None
    return kind, int(num[:-1]), int(num[-1])


def build_reason(cap: Cap, arch_list: Sequence[str]) -> Optional[str]:
    """Why this torch build cannot run kernels on a ``cap`` device, or None. Runs: SASS of the
    same major and a minor at most the device's (sm_86 code runs on sm_89), or PTX of a
    capability at most the device's (JIT-compiled forward). An empty list (a build that does not
    say) is taken as able."""
    if not arch_list:
        return None
    cap = tuple(cap)
    for arch in arch_list:
        parsed = _arch_num(arch)
        if parsed is None:
            continue
        kind, major, minor = parsed
        if kind == "sm" and major == cap[0] and minor <= cap[1]:
            return None
        if kind == "compute" and (major, minor) <= cap:
            return None
    return (f"this PyTorch build has no kernels for {sm(cap)} ({class_name(cap)}); it was built "
            f"for {', '.join(arch_list)}")


def bf16_reason(cap: Cap) -> Optional[str]:
    """Why bf16 compute is not run on ``cap``, or None."""
    if tuple(cap) >= NATIVE_BF16:
        return None
    return (f"{sm(cap)} ({class_name(cap)}) has no native bf16; Poros supports Ampere (sm_80 / "
            "sm_86), Ada (sm_89), Hopper (sm_90) and Blackwell (sm_100 / sm_103 / sm_120)")


def triton_reason(cap: Cap) -> Optional[str]:
    """Why the Triton kernels do not run on ``cap``, or None."""
    if tuple(cap) >= TRITON_MIN:
        return None
    return f"Triton does not compile for {sm(cap)} ({class_name(cap)}); it needs sm_70 or newer"


def _compute_bf16(cfg: Any) -> bool:
    """Whether a run of ``cfg`` computes in bf16 (its NF4 base's compute dtype)."""
    return getattr(cfg, "torch_dtype", "bfloat16") == "bfloat16"


def refusal(cfg: Any, cap: Optional[Cap]) -> Optional[str]:
    """Why a card of capability ``cap`` is refused for ``cfg`` (``check``'s bf16 refusal, which a
    plan of that card says too), or None. Torch-free."""
    if cap is None or not _compute_bf16(cfg):
        return None
    return bf16_reason(cap)


def check(cfg: Any, device: Any = None) -> None:
    """Refuse, before the download, what this GPU cannot run as asked, with the exact remedy: a
    torch build without kernels for it (PorosTorchBuildError, E_STACK_UNSUPPORTED, exit 5), or
    bf16 compute on a card without native bf16 (PorosGpuUnsupportedError, E_GPU_UNSUPPORTED,
    exit 4). Both are PorosConfigErrors."""
    import torch

    from poros.core.errors import PorosGpuUnsupportedError, PorosTorchBuildError

    cap = capability(device)
    name = torch.cuda.get_device_name(device)
    why = build_reason(cap, torch.cuda.get_arch_list())
    if why is not None:
        raise PorosTorchBuildError(
            f"{name}: {why}.",
            remediation=(f"Install a PyTorch build compiled for {sm(cap)}: pick the CUDA variant "
                         "that lists your GPU at https://pytorch.org/get-started/locally/, then "
                         "check that torch.cuda.get_arch_list() includes it."))
    why = refusal(cfg, cap)
    if why is not None:
        supported = ("Train on a supported GPU, sm_80 or newer: Ampere (A100, A10, RTX 30xx), "
                     "Ada (RTX 40xx, L4, L40), Hopper (H100, H200) or Blackwell (B200, RTX 50xx).")
        opt_in = (" Not a supported target, and not bitwise: torch_dtype: float16 runs the NF4 "
                  "compute in fp16 on this card (fp16 can overflow; a run whose loss or gradient "
                  "norm goes non-finite stops at that update).")
        raise PorosGpuUnsupportedError(
            f"Poros needs an NVIDIA GPU with bf16 (Ampere / RTX 30xx or newer); this card "
            f"({name}, {sm(cap)}) has none: {why}.",
            remediation=supported + opt_in)


# -- rank fit -----------------------------------------------------------------------------------

# The largest padded rank whose every kernel fits each architecture, from the ahead-of-time
# compiles (docs/validation/fused/rank_tiles/aot.json, Triton 3.9): dense with no rotary kernel,
# dense with the q/k rotary kernels at head dim 64 and at 128 (Qwen2's fused layer), and the MoE
# expert kernels (bf16, sm_80 and newer; None where they do not run). A card's own run decides
# by launching them (poros.core.fused.fit); this table is what a plan for a named card reads.
FUSED_FIT = {
    (7, 0): (128, 128, 128, None),
    (7, 5): (64, 64, 64, None),
    (8, 0): (256, 256, 256, 256),
    (8, 6): (128, 128, 128, 128),
    (8, 9): (128, 128, 128, 128),
    (9, 0): (256, 256, 256, 256),
    (10, 0): (256, 256, 256, 256),
    (12, 0): (128, 128, 128, 128),
}


def _fit_row(cap: Cap) -> Optional[tuple]:
    cap = tuple(cap)
    if cap in FUSED_FIT:
        return FUSED_FIT[cap]
    # another capability: the listed one with the most shared memory at most its own
    lim = shared_memory_limit(cap)
    below = [(shared_memory_limit(k), k) for k in FUSED_FIT
             if shared_memory_limit(k) <= lim and k <= cap]
    return FUSED_FIT[max(below)[1]] if below else None


def fused_rank_reason(cap: Cap, rank: int, *, entry: str = "dense",
                      head_dim: Optional[int] = None) -> Optional[str]:
    """Why the fused kernels do not fit capability ``cap`` at LoRA ``rank`` (None: they do), as
    the ahead-of-time compiles say. ``head_dim``: the q/k rotary kernels' head size, for a model
    whose fused layer runs them (Qwen2), else None. ``entry`` "moe": the expert kernels."""
    row = _fit_row(cap)
    R = pad(rank)
    if row is None:
        return None  # not known here: the run's own probe decides
    most = row[3] if entry == "moe" else row[0] if not head_dim else \
        row[1] if head_dim <= 64 else row[2]
    if most is not None and R <= most:
        return None
    what = "expert kernels" if entry == "moe" else "fused kernels"
    if most is None:
        return f"the {what} do not run on sm_{cap[0]}{cap[1]} (they compute in bf16)"
    return (f"LoRA rank {rank} (padded {R}): the {what} need more shared memory per block than "
            f"sm_{cap[0]}{cap[1]} has (they take up to rank {most})")


# -- flash attention ---------------------------------------------------------------------------

# Devices whose flash backward takes a head size up to 256 (A100, H100, B200: 228 KB of shared
# memory per SM). On the others PyTorch refuses a head size over 192 when grads are wanted
# (sm86-sm89: check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89), and newer consumer
# parts (sm120) are held to the same 192, conservatively.
_FLASH_D256 = ((8, 0), (9, 0), (10, 0))


def flash_usable(q: torch.Tensor) -> bool:
    """Whether the bf16 flash kernel takes this head size on this device, forward and backward.
    Decided by the device and the head size only, never by ``q.requires_grad``: the forward and
    its rematerialization must run the same kernel."""
    import torch

    D = q.shape[-1]
    if D % 8 or D > 256:
        return False
    available = getattr(torch.backends.cuda, "is_flash_attention_available", lambda: True)
    if not available():
        return False
    cap = torch.cuda.get_device_capability(q.device)
    if cap < (8, 0):
        return False
    if not (D <= 192 or cap in _FLASH_D256):
        return False
    return _flash_runs(q.device.index if q.device.index is not None
                       else torch.cuda.current_device(), D)


_FLASH_RUNS: dict = {}


def _kept(t: torch.Tensor) -> torch.Tensor:
    """A saved tensor as it is (the flash trial's own hooks: none of the caller's)."""
    return t


def _flash_runs(index: int, D: int) -> bool:
    """Whether the flash kernel runs a causal bf16 forward and backward at head size ``D`` on
    device ``index``: tried once per (device, head size) on a tiny input. A build may list flash
    attention yet have no flash kernel for this GPU (a capability it was not compiled for); that
    device then takes ``sdpa_efficient`` for the whole run, in every pass. The trial runs inside
    the first layer's forward, so it keeps its own saved tensors: hooks the caller has set (a
    checkpoint counting what the forward saves, an offload) never see them, and the forward and
    its recomputation save the same tensors."""
    key = (index, D)
    if key not in _FLASH_RUNS:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with torch.enable_grad(), torch.autocast("cuda", enabled=False), \
                    torch.autograd.graph.saved_tensors_hooks(_kept, _kept), \
                    sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                t = torch.zeros(1, 2, 16, D, dtype=torch.bfloat16, device=f"cuda:{index}",
                                requires_grad=True)
                torch.nn.functional.scaled_dot_product_attention(
                    t, t[:, :1], t[:, :1], is_causal=True, enable_gqa=True).sum().backward()
            _FLASH_RUNS[key] = True
        except RuntimeError:
            _FLASH_RUNS[key] = False
    return _FLASH_RUNS[key]


# -- the cards here ----------------------------------------------------------------------------

# nvidia-smi memory.total (MiB) for `--gpu NAME` when planning away from the
# card. Free VRAM is then assumed to be all of it.
GPU_SPECS_MIB = {
    "RTX 3090": 24576, "RTX 3090 Ti": 24564, "RTX 4090": 24564, "RTX 5090": 32607,
    "L40S": 46068, "A100 80GB PCIe": 81920, "H100 80GB HBM3": 81559,
    "RTX PRO 6000 Blackwell Workstation Edition": 97887,
}

# Cards a plan can name that no table above measured: the memory each is sold with, GiB. A plan
# for one takes all of it as free, and its figures come from other cards' runs, as estimates.
CARD_GIB = {
    "RTX 3060 Ti": 8, "RTX 3070": 8, "RTX 3070 Ti": 8, "RTX 3080 Ti": 12,
    "RTX 4060": 8, "RTX 4070": 12, "RTX 4070 SUPER": 12, "RTX 4070 Ti": 12,
    "RTX 4070 Ti SUPER": 16, "RTX 4080": 16, "RTX 4080 SUPER": 16,
    "RTX 5050": 8, "RTX 5060": 8, "RTX 5070": 12, "RTX 5070 Ti": 16, "RTX 5080": 16,
    "RTX A4000": 16, "RTX A4500": 20, "RTX A5000": 24, "RTX A5500": 24, "RTX A6000": 48,
    "RTX 2000 Ada": 16, "RTX 4000 Ada": 20, "RTX 4000 SFF Ada": 20, "RTX 4500 Ada": 24,
    "RTX 5000 Ada": 32, "RTX 6000 Ada": 48,
    "RTX PRO 4000 Blackwell": 24, "RTX PRO 4500 Blackwell": 32,
    "A10": 24, "A16": 16, "A30": 24, "A40": 48, "L4": 24, "L40": 48,
}
# Cards sold with two memory sizes: the name alone does not say which, so --gpu asks for it.
CARD_SIZES = {
    "RTX 3050": (6, 8), "RTX 3060": (8, 12), "RTX 3080": (10, 12), "RTX 4060 Ti": (8, 16),
    "RTX 5060 Ti": (8, 16), "RTX A2000": (6, 12), "RTX PRO 5000 Blackwell": (48, 72),
}
# Short names: `--gpu a100`, `--gpu 6000`.
_ALIASES = {"a100": "A100 80GB PCIe", "h100": "H100 80GB HBM3", "a10g": "A10",
            "6000": "RTX PRO 6000 Blackwell Workstation Edition"}
MAX_CARD_GIB = 1024  # a memory size past this is a typo, not a card
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(gib|gb|g)?", re.IGNORECASE)
_SIZED_NAME = re.compile(r"(.*[^\s-])[\s-]+(\d+(?:\.\d+)?)\s*(?:gib|gb|g)", re.IGNORECASE)


class UnknownCard(KeyError):
    """A --gpu value that names no card Poros knows and gives no memory size."""


class CardNeedsSize(ValueError):
    """A --gpu name sold with two memory sizes (``sizes``, GiB), or a size no card has."""

    def __init__(self, said: str, name: str = "", sizes: Tuple[int, ...] = ()):
        super().__init__(said)
        self.name, self.sizes = name, sizes


def _norm(name: str) -> str:
    # "(simulated)": the dev simulator's card plans as the card it stands in for
    drop = ("nvidia", "geforce", "(simulated)")
    return " ".join(w for w in name.lower().split() if w not in drop)


def _squash(name: str) -> str:
    """A card's name as people shorten it: no maker, no edition words, no spaces."""
    drop = ("nvidia", "geforce", "(simulated)", "blackwell", "workstation", "edition",
            "generation")
    return "".join(w for w in name.lower().replace("-", " ").split() if w not in drop)


def _known_cards() -> dict[str, str]:
    """Every way to say a card Poros knows -> its name: whole, and without its ``RTX``."""
    index = dict(_ALIASES)
    for name in (*GPU_SPECS_MIB, *CARD_GIB, *CARD_SIZES):
        key = _squash(name)
        index.setdefault(key, name)
        if key.startswith("rtx"):
            index.setdefault(key[3:], name)
    return index


def card_words() -> list[str]:
    """What a shell or the session offers after ``--gpu``: the usual memory sizes, then every
    card ``named_card`` knows, each as one word it reads back (``4070tisuper``, ``a100``,
    ``4060ti-16gb`` for a card sold in two sizes)."""
    short = {name: alias for alias, name in _ALIASES.items() if alias != "a10g"}

    def word(name: str) -> str:
        return short.get(name) or _squash(name).removeprefix("rtx")

    sizes = ["8GB", "12GB", "16GB", "24GB", "32GB", "48GB", "80GB"]
    two = [f"{word(name)}-{n}gb" for name, both in CARD_SIZES.items() for n in both]
    return sizes + [word(name) for name in (*GPU_SPECS_MIB, *CARD_GIB)] + two


def _size(text: str) -> float:
    gib = float(text)
    if not 1 <= gib <= MAX_CARD_GIB:
        raise CardNeedsSize(f"{text} GB is not a card's memory: 1 to {MAX_CARD_GIB}")
    return gib


def named_card(said: str) -> Tuple[str, "PlanGpu"]:
    """The card a plan's ``--gpu`` names, and the value that says it again (what a result's
    command repeats): a card Poros knows by name (``RTX 4060``, ``4070 ti super``, ``l4``), a
    memory size (``8GB``, ``12``: a number under 1000 is GiB, as cards are sold), or a name with
    its size (``RTX 4060 Ti 16GB``, any name: its generation is read from it where it says).
    All of its memory is taken as free. Raises ``UnknownCard`` for a name with no size that
    Poros does not know, ``CardNeedsSize`` for a card sold in two sizes named without one."""
    from poros.core.schemas import PlanGpu

    def card(name: str, gib: float) -> "PlanGpu":
        return PlanGpu(name=name, total_gib=round(gib, 2), free_gib=None, source="spec")

    text = " ".join(str(said).split())
    known = _known_cards()
    m = _SIZE.fullmatch(text)
    if m and (m.group(2) or float(m.group(1)) < 1000):  # 8GB, 12; 4090 is a card's number
        gib = _size(m.group(1))
        return f"{gib:g}GB", card("NVIDIA GPU", gib)
    name, given, gib = known.get(_squash(text)), text, None
    if name is None and (m := _SIZED_NAME.fullmatch(text)):
        given, gib = m.group(1), _size(m.group(2))
        name = known.get(_squash(given), given)
    if name is None:
        raise UnknownCard(text)
    measured = GPU_SPECS_MIB.get(name)
    if measured and (gib is None or abs(gib - measured / 1024) <= 0.1 * gib):
        return name, card(name, measured / 1024)  # nvidia-smi's own figure for it
    if measured:  # another size of a measured card's name: "A100 40GB"
        name = given
    if gib is None and name in CARD_SIZES:
        sizes = CARD_SIZES[name]
        raise CardNeedsSize(f"{name} is sold with {sizes[0]} or {sizes[1]} GB", name, sizes)
    if gib is None or gib == CARD_GIB.get(name):
        return name, card(name, CARD_GIB[name])
    sized = f"{name} {gib:g}GB"  # "RTX 4060 Ti 16GB", "A100 40GB": the name keeps its size
    return sized, card(sized, gib)



def _cuda_order() -> Optional[list[str]]:
    """The devices CUDA numbers, by UUID, in its order: only CUDA knows FASTEST_FIRST and every
    form CUDA_VISIBLE_DEVICES takes. cuInit makes no context, so this takes none of the card.
    [] when it sees no device; None when it cannot say."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1" if os.name == "posix" else "nvcuda.dll")
    except OSError:
        return None
    status = cuda.cuInit(0)
    if status == 100:  # CUDA_ERROR_NO_DEVICE: none, or CUDA_VISIBLE_DEVICES hides them
        return []
    count, device = ctypes.c_int(), ctypes.c_int()
    if status or cuda.cuDeviceGetCount(ctypes.byref(count)):
        return None
    order = []
    for i in range(count.value):
        raw = (ctypes.c_ubyte * 16)()
        if cuda.cuDeviceGet(ctypes.byref(device), i) or cuda.cuDeviceGetUuid(raw, device):
            return None
        order.append(f"GPU-{uuid.UUID(bytes=bytes(raw))}")
    return order


def _cuda_capabilities() -> dict[str, tuple[int, int]]:
    """Each device's compute capability by UUID, from the driver (cuDeviceGetAttribute): what the
    card is, not what its name suggests. cuInit makes no context. {} when the driver cannot say."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1" if os.name == "posix" else "nvcuda.dll")
        if cuda.cuInit(0):
            return {}
        count, device = ctypes.c_int(), ctypes.c_int()
        if cuda.cuDeviceGetCount(ctypes.byref(count)):
            return {}
        caps = {}
        for i in range(count.value):
            raw, major, minor = (ctypes.c_ubyte * 16)(), ctypes.c_int(), ctypes.c_int()
            if (cuda.cuDeviceGet(ctypes.byref(device), i) or cuda.cuDeviceGetUuid(raw, device)
                    or cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device)  # ..._MAJOR
                    or cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)):  # _MINOR
                return {}
            caps[f"GPU-{uuid.UUID(bytes=bytes(raw))}"] = (major.value, minor.value)
        return caps
    except Exception:  # no driver library, or one that does not have these calls
        return {}


def _with_own_cache(gpu: PlanGpu, index: int = 0) -> PlanGpu:
    """``gpu`` (device ``index``) with this process's own share counted as free: where torch has
    CUDA up here already, the free memory torch.cuda.mem_get_info reads plus what this process's
    allocator reserved and does not use (a trainer deleted without empty_cache), which the next
    run's allocator reuses. A process with no CUDA context yet keeps the driver's figure: asking
    torch would make one, and that context is no run's memory."""
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return gpu
        mine = str(torch.cuda.get_device_properties(index).uuid).lower().removeprefix("gpu-")
        if gpu.uuid is not None and gpu.uuid.lower().removeprefix("gpu-") != mine:
            return gpu  # not the card torch numbers ``index``
        free, _ = torch.cuda.mem_get_info(index)
        spare = torch.cuda.memory_reserved(index) - torch.cuda.memory_allocated(index)
    except Exception:  # advice for a fit check: the driver's figure stands
        return gpu
    return gpu.model_copy(update={"free_gib": round((free + max(0, spare)) / 2**30, 2)})


def nvidia_smi(query: str, *, units: bool = False, on: Sequence[str] = (), timeout: float = 10.0,
               raw: bool = False) -> Optional[list]:
    """``nvidia-smi --query-<query> --format=csv,noheader[,nounits] [on...]``: one list of
    stripped fields per row (``raw``: the output's lines as they are), or None where it did not
    run or did not exit 0. The one place Poros asks nvidia-smi (``query``: ``gpu=name,...`` or
    ``compute-apps=...``; ``on``: ``["-i", uuid]`` for one card)."""
    fmt = "--format=csv,noheader" + ("" if units else ",nounits")
    try:
        proc = subprocess.run([outside_tool("nvidia-smi"), f"--query-{query}", fmt, *on],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.splitlines()
    return lines if raw else [[x.strip() for x in row.split(",")] for row in lines if row.strip()]


def visible_gpus() -> Optional[list[PlanGpu]]:
    """The cards torch would number, in its order, asking neither torch nor the card. None when
    only torch can say: no nvidia-smi, or a MIG slice, which nvidia-smi has no row for."""
    from poros.core.schemas import PlanGpu

    try:
        rows = nvidia_smi("gpu=uuid,name,memory.total,memory.reserved,memory.free")
        if rows is None:
            return None
        cards = {}
        caps = _cuda_capabilities()
        for card, name, total, reserved, free in rows:
            usable = int(total) - int(reserved)  # what the driver keeps is not in CUDA's total
            cards[card] = PlanGpu(name=name, total_gib=round(usable / 1024, 2),
                                  free_gib=round(int(free) / 1024, 2), source="nvidia-smi",
                                  uuid=card, capability=caps.get(card))
        order = _cuda_order()
        return list(cards.values()) if order is None else [cards[card] for card in order]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None


def card_refusal(cfg: "PorosConfig", gpu: Optional[PlanGpu]) -> Optional[str]:
    """Why ``poros train`` refuses ``cfg`` on ``gpu`` before the download (poros.core.card.check's
    card refusal: bf16 compute on a card without native bf16), or None."""
    return refusal(cfg, card_capability(gpu))


def card_capability(gpu: Optional[PlanGpu]) -> Optional[tuple]:
    """The compute capability a plan of ``gpu`` reads: the device's own for a card that is here,
    its name's only for a named card (--gpu) or one whose device did not say."""
    if gpu is None or gpu.unified:
        return None
    return tuple(gpu.capability) if gpu.capability else capability_of_name(gpu.name)


def detect_gpu(name: Optional[str] = None, *, quietly: bool = False) -> Optional[PlanGpu]:
    """The GPU to plan for: ``--gpu`` (``named_card``: a name or a memory size), else the device
    torch will train on, from nvidia-smi: asking torch takes a process's worth of the card, beside
    a run. Torch only where nvidia-smi cannot say, and not ``quietly``: a reader's poll never
    takes that."""
    from poros.core.schemas import PlanGpu

    if name is not None:
        return named_card(name)[1]
    if sys.platform == "darwin":
        from poros.metal.table import this_mac

        return this_mac()
    cards = visible_gpus()
    if cards is not None:
        return _with_own_cache(cards[0]) if cards else None
    if quietly:
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    i = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(i)
    try:
        cap: Optional[tuple] = tuple(torch.cuda.get_device_capability(i))
    except Exception:  # advice for a plan: the name maps it
        cap = None
    return _with_own_cache(PlanGpu(
        name=torch.cuda.get_device_name(i), total_gib=round(total / 2**30, 2),
        free_gib=round(free / 2**30, 2), source="torch", capability=cap), i)
