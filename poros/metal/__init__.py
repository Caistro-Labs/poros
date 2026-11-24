"""The native Apple Metal path. Nothing here imports torch at module level.

``require`` is the gate every Mac entry path reaches: ``PorosTrainer`` calls it instead of
falling to CPU-like branches on ``mps``. It either passes or raises.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from poros.core.errors import (
    PorosConfigError,
    PorosGuaranteeUnavailableError,
    PorosStackUnsupportedError,
)

# The classes whose rows a run measured. Each chip, GPU core count and memory size is its own
# class. Every other Apple silicon Mac trains on rows poros.metal.table derives from the first of
# these for its own memory, after poros.metal.selfcheck has checked it once.
DEVICE_CLASSES = ("apple-m4-10gpu-16gb",)
# A test tool for a developer: plan and guard as if this Mac had this many GB, no more than it has.
DEV_MEMORY_ENV = "POROS_DEV_MAC_MEMORY_GB"
_CLASS = re.compile(r"apple-[a-z0-9-]+-\d+gpu-[1-9]\d*gb")

PINS = {
    "torch": "2.15.0",
    "transformers": "5.20.2",
    "peft": "0.20.0",
    "accelerate": "1.15.0",
    "safetensors": "0.8.0",
    "datasets": "5.1.0",
}
# The floor: Poros's kernel's bfloat needs MSL 3.1, which is macOS 14, as bf16 on MPS does.
# torch 2.15.0 runs there: its binaries are built for 14.0, its device needs only an Apple7 GPU
# (MPSDevice.mm), and its "macOS 15.0+" message is only the error for no GPU at all
# (EmptyTensor.cpp, mps_unavailable_error).
MIN_MACOS = (14, 0, 0)
# torch 2.15.0 works around stride-0 mm operands only below 26.4, and torch's main says the
# bug is back on exactly this release (ops/LinearAlgebra.mm:1188-1197, main Matmul.mm:674-682).
REFUSED_MACOS = ((27, 0, 0),)

# Before the first MPS allocation, the same in every arm: crossing low changes heap tier and
# commit behaviour, and the default high (1.7) would swap before any OOM.
HIGH_WATERMARK = "1.05"
LOW_WATERMARK = "0.95"

# Honoured as on Linux. The model's row in the Mac table is checked by poros/metal/table.py.
ALLOWED = frozenset({
    "model_name_or_path", "lora_rank", "lora_alpha", "lora_dropout",
    "learning_rate", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
    "max_grad_norm", "warmup_steps", "lr_scheduler", "num_epochs", "max_steps", "seed",
    "resolved_base_model_path", "model_revision", "dataset", "output_dir", "eval_steps",
    "eval_checkpoints",
    "logging_steps", "save_total_limit", "progress", "push_to_hub", "hub_model_id", "memory_guard",
    # poros.Run and its recipes run on the same engine.
    "recipe", "loop", "train_on", "dpo_beta", "init_adapter", "eval_dataset", "eval_split",
    "eval_rows",
})
# From the Mac table when unset (poros/metal/table.py). A set value runs if the budget allows.
_FROM_TABLE = frozenset({
    "gradient_accumulation_steps", "max_seq_length", "save_steps", "store_path", "sss_ram",
})
# Field -> the values the Mac can honour. Any field in neither list above takes its default only.
_ACCEPTED = {
    "block_size": (None, 1),
    "attn_implementation": (None, "eager"),
    "adapter_backend": ("auto", "peft"),
    "store": ("ram", "ssd"),
    # The Mac trains one micro-batch at a time: no block-major path.
    "path": (None, "sequential"),
    # PEFT's modules: false says what a Mac runs.
    "fused_compute": (None, False),
}
# Why a CUDA setting has no Mac counterpart, and what to set instead: a refusal says both.
_CUDA_ONLY = {
    "path": ("a Mac trains one micro-batch at a time through every layer (path: sequential), and "
             "every path trains the same update bit for bit, so no result is lost",
             "Set path: sequential, or leave path unset"),
    "fused_compute": ("the fused kernels are CUDA's (Triton); a Mac computes with PEFT's modules",
                      "Set fused_compute: false, or leave it unset"),
}


def macos_version(text: str | None = None) -> tuple[int, int, int]:
    """(major, minor, patch): ``27.0`` and ``27.0.0`` are the same release."""
    text = platform.mac_ver()[0] if text is None else text
    parts = [int(p) for p in text.split(".")] if text else []
    return tuple((parts + [0, 0, 0])[:3])  # type: ignore[return-value]


def run_tool(*cmd: str) -> str:
    """A system tool's stdout, empty if it cannot run. These ship with macOS: no compiler."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def real_memory_bytes() -> int:
    """``hw.memsize``: this Mac's memory, 0 if it cannot be read."""
    said = run_tool("/usr/sbin/sysctl", "-n", "hw.memsize")
    return int(said) if re.fullmatch(r"[0-9]{1,15}", said) else 0


def _chip_and_cores() -> str:
    """``apple-m4-10gpu``: a class less its memory, in letters, digits and dashes."""
    chip = run_tool("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
    cores = re.search(
        r'"gpu-core-count"\s*=\s*(\d{1,4})',
        run_tool("/usr/sbin/ioreg", "-rc", "AGXAccelerator", "-d", "1"),
    )
    return "-".join((
        re.sub(r"[^a-z0-9]+", "-", chip.lower()).strip("-")[:40] or "unknown",
        f"{cores.group(1) if cores else 0}gpu",
    ))


def dev_memory_gb() -> int | None:
    """The developer's budget (``DEV_MEMORY_ENV``): a whole number of GB from 4 to less than this
    Mac has. Anything else is no budget, and the Mac's own memory counts. A budget never makes a
    Mac a class a run measured: its figures would read as measured ones."""
    said = os.environ.get(DEV_MEMORY_ENV, "")
    if not re.fullmatch(r"[0-9]{1,4}", said):
        return None
    gb = int(said)
    if not (4 <= gb and gb * 2**30 < real_memory_bytes()):
        return None
    return None if f"{_chip_and_cores()}-{gb}gb" in DEVICE_CLASSES else gb


def memory_bytes() -> int:
    """This Mac's memory, or the developer's budget for it: what every plan and guard reads."""
    budget = dev_memory_gb()
    return budget * 2**30 if budget else real_memory_bytes()


@lru_cache(maxsize=1)
def device_class() -> str:
    """``apple-m4-10gpu-16gb``, from chip, GPU cores and memory, in letters, digits and dashes.
    A chip or memory that cannot be read yields a class ``require_class`` refuses."""
    return f"{_chip_and_cores()}-{round(memory_bytes() / 2**30)}gb"


def measured(here: str) -> bool:
    """Whether a run measured this class's rows. Every other class trains on derived ones."""
    return here in DEVICE_CLASSES


def macos_build() -> str:
    return run_tool("/usr/bin/sw_vers", "-buildVersion")


def machine_id() -> str:
    """``apple-<hash>``: names this Mac, and no hardware id reaches a job's JSON."""
    platform_device = run_tool("/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice")
    found = re.search(r'"IOPlatformUUID" = "([^"]+)"', platform_device)
    digest = hashlib.blake2b((found.group(1) if found else "").encode(), digest_size=8)
    return f"apple-{digest.hexdigest()}"


def installed(name: str) -> str | None:
    """A package's version, from its metadata: no import."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _stack_problems() -> list[str]:
    """What stops this machine running at all: chip architecture, macOS, exact pins."""
    problems = []
    if platform.machine() != "arm64":
        problems.append(f"this Mac is {platform.machine()}; Poros needs Apple silicon (arm64)")
    version = macos_version()
    shown = ".".join(map(str, version))
    if version < MIN_MACOS:
        problems.append(
            f"macOS {shown} is below 14, which Poros needs: its Metal kernel and bf16 on MPS "
            "need macOS 14 or later"
        )
    elif version in REFUSED_MACOS:
        problems.append(
            f"macOS {shown}: torch {PINS['torch']} has no workaround here for a bug that "
            "torch's own source names on this release (mm on stride-0 operands), so "
            "update macOS"
        )
    for name, want in PINS.items():
        have = installed(name)
        if have != want:
            problems.append(
                f"{name} is {have or 'not installed'}; this release needs exactly {want}"
            )
    return problems


def readable(here: str) -> bool:
    """Whether this class names an Apple chip and its memory: all a plan needs of it."""
    return measured(here) or bool(_CLASS.fullmatch(here))


def require_class(here: str) -> None:
    """Every Apple silicon Mac trains. Refused: one whose chip or memory could not be read, since
    nothing can be planned for it."""
    if readable(here):
        return
    raise PorosGuaranteeUnavailableError(
        f"Poros could not read this Mac's chip or memory ({here}).",
        remediation="Run `poros machine`. Poros reads them with /usr/sbin/sysctl, which every "
                    "macOS has.",
    )


def watermarks() -> tuple[str, str]:
    """(high, low) for torch's MPS allocator, as shares of the GPU's working set. Under the
    developer's budget they shrink to that budget's share of this GPU, so the allocator itself
    refuses past what a Mac with that memory would give it."""
    if dev_memory_gb() is None or measured(device_class()):
        return HIGH_WATERMARK, LOW_WATERMARK
    from poros.metal import machine, table

    live = machine.gpu_working_set()
    if not live:
        return HIGH_WATERMARK, LOW_WATERMARK
    share = table.working_set(device_class())[0] / live
    return f"{float(HIGH_WATERMARK) * share:.4f}", f"{float(LOW_WATERMARK) * share:.4f}"


def _no_mps() -> str:
    """Why torch found no MPS device: an Apple silicon Mac always has one, outside a VM."""
    return "torch reports no MPS device"


def validate(config: Any) -> None:
    """The Mac validator: refuse what the Mac cannot honour, naming every field at once."""
    fields = type(config).model_fields
    refused = []
    for name in sorted(set(fields) - ALLOWED - _FROM_TABLE):
        value = getattr(config, name)
        accepted = _ACCEPTED.get(name, (fields[name].get_default(call_default_factory=True),))
        if value not in accepted:
            refused.append(f"{name}={value!r}")
    if refused:
        names = [r.split("=", 1)[0] for r in refused]
        why = [_CUDA_ONLY[n][0] for n in names if n in _CUDA_ONLY]
        fix = [_CUDA_ONLY[n][1] for n in names if n in _CUDA_ONLY]
        rest = [n for n in names if n not in _CUDA_ONLY]
        raise PorosConfigError(
            f"This Mac cannot honour {', '.join(refused)}" + (f": {'; '.join(why)}" if why else "")
            + ".",
            remediation=". ".join(fix + ([f"Remove {', '.join(rest)}: on a Mac, Poros picks "
                                          "them for the model"] if rest else [])) + "."
            if fix else "Remove these settings. On a Mac, Poros picks them for the model.",
        )


def names_experts(raw: dict) -> bool:
    """Whether a config names experts, whatever its family is called."""
    text = raw.get("text_config", raw)
    return isinstance(text, dict) and any(
        k in text for k in ("num_experts", "num_local_experts", "n_routed_experts"))


def has_experts(model: str, revision: str | None) -> bool:
    """Whether a model's config names experts, whatever its family is called."""
    from poros.plan.configs import read_model_config

    return names_experts(read_model_config(model, revision) or {})


def settle(config: Any) -> None:
    """The Mac table and the store, for every entry path. A local directory of a family the table
    has takes that family's row, and a Hub id or a folder of a family someone asserted takes the
    row derived from its config. Any other folder is a developer's own model: no row and no store
    to pick, and the architecture check still sees it at load."""
    from poros.metal import store, table

    _refuse_a_looped_decoder(config)
    here = device_class()
    if table.fill(config, here) is not None:
        if _checking_here(config):  # the self-check's own few MB in its own folder: no disk rule
            store.use(config.store_path)
        else:
            store.settle(config)
    elif not Path(config.model_name_or_path).is_dir():
        table.require_row(config.model_name_or_path, here)


def _refuse_a_looped_decoder(config: Any) -> None:
    """A decoder that runs its layers as visits (a looped one, as its config class declares) is
    not the Mac's path's to train, whatever its status or assertion: refused before any row,
    store or download (poros.preflight.arch.MAC_LOOPED)."""
    from poros.moe.carried import declares_visits
    from poros.plan.configs import read_model_config

    raw = read_model_config(config.model_name_or_path, getattr(config, "model_revision", None))
    if raw and declares_visits(raw):
        from poros.preflight.arch import LoopedOnAMacError, mac_looped, mac_looped_fix

        raise LoopedOnAMacError(f"{config.model_name_or_path}: {mac_looped(raw)}.",
                                remediation=mac_looped_fix(raw))


def _checking_here(config: Any) -> bool:
    from poros.metal import selfcheck

    return selfcheck.inside() and config.store_path == selfcheck.scratch_store()


def same_base(recorded: str | None, here: str | None, adapter: Any = None,
              base: str | None = None) -> str | None:
    """Refuse an adapter on any base but the one it trained on: same-shaped weights pass every
    other check. ``recorded`` is what its poros_config.json says, ``here`` is ``load.fetch``'s.

    A Hub base's revision is its commit. A local folder's is its shards' sizes and times, which
    a copy changes as a rewrite does, so where ``base`` (the model as the config names it) is a
    folder and the revisions differ, its contents decide: what the adapter's folder ``adapter``
    recorded of its base against what the folder holds (``poros.preflight.resume``). The same
    model anywhere passes and another is refused. An adapter with no record has only the
    revision: the folder it names, with other sizes or times, is refused (nothing recorded can
    tell a rewrite from a touch, and neither a size nor a time is kept apart from the revision);
    a base at another place, with the one it names not here to compare, goes on with one
    warning, which is returned."""
    if recorded == here:
        return None
    if adapter is not None and base and Path(base).is_dir():
        from poros.preflight import resume

        same = resume.same_base_contents(adapter, base)
        if same is None and resume.base_moved(adapter, base):
            said = resume.unchecked_base(adapter, base)
            resume.warn_once(said)
            return said
        if same:
            return None
    if recorded is None:  # an older Poros's run, or one moved here with no record of its base
        raise PorosConfigError(
            f"This run recorded no revision of its base, so this Mac cannot check that the base "
            f"here ({str(here)[:12]}) is the one it trained on.",
            remediation="Run it on the machine it trained on, or set model_revision in its "
                        "poros_config.json to the base it trained on. Its adapter alone "
                        "(--format adapter) needs no check.")
    raise PorosConfigError(
        f"This run trained on base revision {str(recorded)[:12]}, and the base here is "
        f"{str(here)[:12]}.",
        remediation="Put back the base it trained on, or start a new run on this one.",
    )


def check_this_mac() -> None:
    """A Mac no run measured checks itself once, before its first run (``poros.metal.selfcheck``).
    Its caller holds this Mac: the check runs MPS, and one pool of memory hosts one run."""
    here = device_class()
    if not measured(here):
        from poros.metal import selfcheck

        selfcheck.ensure(here)


def require(config: Any, *, check: bool = True) -> None:
    """Pass, or raise. Called before the first MPS allocation, since it sets the watermarks.
    ``check=False`` leaves this Mac's own check to ``check_this_mac``, for a caller that takes
    this Mac's lock first."""
    problems = _stack_problems()
    if problems:
        raise PorosStackUnsupportedError(
            f"This Mac cannot train: {'; '.join(problems)}.",
            remediation="Run `poros machine`. `pip install poros-train` sets the pins.",
        )
    here = device_class()
    require_class(here)
    validate(config)
    settle(config)

    high, low = watermarks()
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = high
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = low
    os.environ.pop("PYTORCH_MPS_FAST_MATH", None)
    # It warns on import that a Mac has no stdout redirects, which nothing here uses.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").disabled = True

    import torch

    if not torch.backends.mps.is_available():
        raise PorosStackUnsupportedError(
            f"This Mac cannot train: {_no_mps()}.",
            remediation="Check this is not a virtual machine, which has no GPU for torch. "
                        "`poros machine` says more.",
        )
    if check:  # in a process of its own, so this one's MPS is still untouched
        check_this_mac()
