"""How a run holds Inkling 2 Mini's experts: decided once, from the checkpoint and the setting.

Torch-free (the pre-run checks read it). One decision, ``decide``, which every place that needs
the format calls: the trainer before it records ``moe_expert_format`` (and before anything
downloads), the streamed load, the pre-run size checks, and export through the run's record.

- A checkpoint whose experts are all stored quantized (the release: NVFP4, MXFP8 on its last
  layers) is held ``released`` (those bytes, decoded for each use; the default) or ``nf4`` when
  asked.
- A checkpoint whose experts are all unquantized (a merged export of a run, or any bf16 or fp32
  save) is held ``nf4``, as every other MoE family's experts are: there are no released bytes to
  keep. Asking ``released`` of it is refused in words.
- A checkpoint quantized on some layers and not others is refused in words: no one format holds
  its experts, and a record of one would be false for the rest.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Optional

__all__ = [
    "RELEASED",
    "NF4",
    "FAMILIES",
    "asked_format",
    "checkpoint_state",
    "decide",
    "from_config",
    "checkpoint_names",
    "state_before_download",
    "expected",
    "NOT_YET",
]

RELEASED, NF4 = "released", "nf4"
FORMATS = (RELEASED, NF4)
FAMILIES = ("inkling2_loop", "inkling2_loop_text")
QUANTIZED, UNQUANTIZED, MIXED = "quantized", "unquantized", "mixed"

_W13 = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.w13_weight(_amax|_scale)?$")


def _refuse(message: str, remediation: str) -> Exception:
    from poros.core.errors import PorosConfigError

    return PorosConfigError(message, remediation=remediation)


def asked_format(config: Any) -> Optional[str]:
    """The format a run asks for: its ``moe_expert_format`` (set, or recorded by the checkpoint
    it resumes), else ``POROS_INKLING_EXPERTS``, else None (the checkpoint decides). A variable
    that contradicts the run's own setting is refused rather than either silently winning,
    except where the setting is a finished run's record (``poros eval`` rebuilding it): that run
    held its experts so, and the variable does not apply to it."""
    env = os.environ.get("POROS_INKLING_EXPERTS")
    if env is not None and env not in FORMATS:
        raise _refuse(
            f"POROS_INKLING_EXPERTS={env!r}: one of released, nf4.",
            "Unset POROS_INKLING_EXPERTS, or set released or nf4.",
        )
    said = getattr(config, "moe_expert_format", None)
    if said is not None and getattr(config, "_expert_format_recorded", False):
        return said
    if said is not None and env is not None and env != said:
        raise _refuse(
            f"This run holds Inkling 2 Mini's experts as {said} (its config, or the checkpoint it "
            f"resumes, records that), and POROS_INKLING_EXPERTS says {env}: they would be other "
            "weights.",
            "Unset POROS_INKLING_EXPERTS, or set moe_expert_format to match it in a new run.",
        )
    return said or env


def checkpoint_state(names: Iterable[str]) -> Optional[str]:
    """``quantized`` (every layer's experts carry their scales), ``unquantized`` (none do) or
    ``mixed``, from the checkpoint's tensor names; None where it names no experts."""
    layers: dict[int, bool] = {}
    for name in names:
        m = _W13.match(name)
        if m:
            layers[int(m[1])] = layers.get(int(m[1]), False) or m[2] is not None
    if not layers:
        return None
    states = set(layers.values())
    return MIXED if len(states) > 1 else QUANTIZED if states == {True} else UNQUANTIZED


def decide(state: Optional[str], asked: Optional[str]) -> str:
    """The format a run holds the experts in (module doc), or a refusal in words. ``asked`` is
    what the run or POROS_INKLING_EXPERTS asked for, None where neither did."""
    if state == MIXED:
        raise _refuse(
            "This checkpoint stores Inkling 2 Mini's experts quantized on some layers and "
            "unquantized on others: no one format holds them, so it does not train.",
            "Train from the released checkpoint, or from a save whose experts are all unquantized.",
        )
    if state == UNQUANTIZED:
        if asked == RELEASED:
            raise _refuse(
                "Holding experts as released keeps a checkpoint's own NVFP4 and MXFP8 bytes, and "
                "this checkpoint stores its experts unquantized (a merged export, or a bf16 "
                "save): there are none to keep.",
                "Unset moe_expert_format and POROS_INKLING_EXPERTS (its experts are then held "
                "in NF4), or set either to nf4.",
            )
        return NF4
    return asked or RELEASED


#: Said where the checkpoint's experts cannot be judged before its weights are downloaded.
NOT_YET = (
    "Poros cannot tell yet whether this checkpoint stores Inkling 2 Mini's experts "
    "quantized (held as released) or unquantized (held in NF4): its tensor names and "
    "quantization files are not readable before the download. The load decides from its "
    "files; this sizes them as released, the larger."
)


def _files(model: str, revision: Optional[str]) -> Optional[list[str]]:
    """The checkpoint's file names without its weights: a folder's listing, or the Hub
    repository's (None where it cannot be read: offline, gated, not a Hub id)."""
    if os.path.isdir(model):
        return os.listdir(model)
    try:
        from huggingface_hub import list_repo_files

        return list(list_repo_files(model, revision=revision))
    except Exception:
        return None


def state_before_download(
    model: str, revision: Optional[str], model_config: Optional[dict]
) -> tuple[Optional[str], bool]:
    """(state, certain): what the checkpoint stores, from everything in hand without its
    weights. Its tensor names (a folder's index or headers, the Hub's safetensors metadata) are
    certain. Without them, a quantization marker (config.json's ``quantization_config``, or an
    ``hf_quant_config.json`` beside it, which some quantized checkpoints carry alone) says
    quantized, and a readable file list with neither says unquantized; neither is certain (a
    marker can be left over), so the load decides again from its files. (None, False) where
    nothing can be read."""
    names = checkpoint_names(model, revision)
    state = checkpoint_state(names) if names else None
    if state is not None:
        return state, True
    if model_config and model_config.get("quantization_config"):
        return QUANTIZED, False
    files = _files(model, revision)
    if files is None:
        return None, False
    return (QUANTIZED if "hf_quant_config.json" in files else UNQUANTIZED), False


def from_config(model_config: Optional[dict], asked: Optional[str]) -> Optional[str]:
    """The format a run of this config.json would hold, for size checks with config.json alone
    (``expected`` uses all that is in hand): a quantization config says quantized. None for
    every other family."""
    from poros.core.arch_registry import registry_type

    if not model_config or registry_type(model_config) not in FAMILIES:
        return None
    state = QUANTIZED if model_config.get("quantization_config") else UNQUANTIZED
    try:
        return decide(state, asked)
    except Exception:  # a refusal is the trainer's to say; a size check sizes what was asked
        return asked


def expected(config: Any, model_config: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    """(format, note) for checks before the download (``poros check``, the plan, the pre-run
    sizes): the run's recorded format, else what ``state_before_download`` and the asked format
    decide; where the checkpoint cannot be judged yet, released (the larger) with ``NOT_YET``.
    (None, None) for every other family, and for a run whose format is already decided."""
    from poros.core.arch_registry import registry_type

    said = getattr(config, "moe_expert_format", None)
    if said is not None or not model_config or registry_type(model_config) not in FAMILIES:
        return said, None
    try:
        asked = asked_format(config)
    except Exception:  # the trainer refuses it in words; a check sizes what config.json says
        asked = None
    state, _ = state_before_download(
        str(config.model_name_or_path), getattr(config, "model_revision", None), model_config
    )
    if state is None:
        return asked or RELEASED, (None if asked else NOT_YET)
    try:
        return decide(state, asked), None
    except Exception:  # refused at the run's start, in words; sized as asked here
        return asked, None


def _unreadable(path: str, what: str) -> Exception:
    return _refuse(
        f"{path} {what}, so which tensors this checkpoint holds, and how its experts are "
        "stored, cannot be read: nothing was loaded.",
        "Replace the file with the checkpoint's own (download it again), or name another "
        "checkpoint.",
    )


def checkpoint_names(model: str, revision: Optional[str] = None) -> Optional[list[str]]:
    """The checkpoint's tensor names before any weight is downloaded: a local folder's index or
    safetensors headers, or the Hub's safetensors metadata (headers read by range). None where
    neither can be read (offline, nothing cached): the load then decides from its own files."""
    import json

    if os.path.isdir(model):
        index = os.path.join(model, "model.safetensors.index.json")
        if os.path.isfile(index):
            try:
                with open(index) as f:
                    weight_map = json.load(f).get("weight_map")
            except (OSError, ValueError, AttributeError):
                weight_map = None
            if not isinstance(weight_map, dict):
                raise _unreadable(
                    index,
                    "is not a safetensors index (a JSON object with a weight_map of tensor names)",
                )
            return list(weight_map)
        names: list[str] = []
        for file in sorted(os.listdir(model)):
            if file.endswith(".safetensors") and not file.startswith("mtp"):
                path = os.path.join(model, file)
                header = None
                try:
                    with open(path, "rb") as f:
                        n = int.from_bytes(f.read(8), "little")
                        if 0 < n <= os.path.getsize(path) - 8:
                            header = json.loads(f.read(n))
                except (OSError, ValueError):
                    header = None
                if not isinstance(header, dict):
                    raise _unreadable(path, "has no safetensors header that reads")
                names += [k for k in header if k != "__metadata__"]
        return names or None
    try:
        from huggingface_hub import get_safetensors_metadata

        return list(get_safetensors_metadata(model, revision=revision).weight_map)
    except Exception:  # offline, gated, or not a Hub id: the load decides from its files
        return None
