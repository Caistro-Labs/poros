"""Before the download, for ``PorosTrainer.from_pretrained`` (and so ``poros.Run``): the checks
in the order a run takes them. ``poros train``'s order, and why it differs, is
``poros.preflight.command``.

``on_cuda`` (CUDA only; a Mac takes ``poros.preflight.mac.before_download``):

1. the card (``poros.core.card.check``: the torch build, bf16);
2. CUDA's defaults for what the config left unset (``PorosConfig.apply_cuda_defaults``), and a
   block-major ``path`` you set that cannot train this model's update refused
   (``path_refusal``);
3. the architecture gate for a dense model config.json names (``arch.require``), before the fit,
   so an unvalidated model hears that, not a fit's remedy;
4. the fit (``fit.gate``): the path and the tables adapted, then the GPU and the host RAM.

``route``, after the trainer's determinism and seeds: what the compute refuses
(``compute.refuse``), the route (``"moe"``, ``"metal_moe"`` or ``"dense"``), a store only a MoE
model takes, and the architecture gate for the type config.json gives.

``moe_settings_refusal``: what the MoE path does not support, refused before its load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from poros.core.errors import PorosConfigError

logger = logging.getLogger(__name__)


@dataclass
class Startup:
    """What ``on_cuda`` read and decided, for ``route``."""

    model_json: Optional[dict] = None  # config.json (read on CUDA)
    dense_mt: Optional[str] = None  # its model_type, where the dense gate already ran
    arch_checked: bool = False


def require_arch(model_type: str, mode: str) -> None:
    """The architecture gate (``arch.arch_refusal``), raised."""
    from poros.preflight.arch import arch_refusal

    err = arch_refusal(model_type, mode)
    if err is not None:
        raise err


def path_refusal_error(config: Any, model_json: Optional[dict]) -> Optional[PorosConfigError]:
    """A block-major ``path`` you set that cannot train this model's update (the reason
    ``sequential_reason`` finds), refused before the download; None otherwise."""
    from poros.core.config import path_refusal, sequential_reason, trains_block_major

    why = trains_block_major(config) and sequential_reason(config, model_json)
    if why:
        return PorosConfigError(path_refusal(config.path, why),
                                remediation="Set path: sequential, or leave path unset.")
    return None


def on_cuda(config: Any, device: Any, model_name_or_path: str, *, check_fit: bool = True,
            check_host_ram: bool = True, at_any_length: bool = False) -> Startup:
    """A CUDA run's checks before the download (the module docstring's 1-4). ``at_any_length``:
    the load leaves the fit at a length to train(), so only a run that fits at no length is
    refused here (``fit.at_no_length``)."""
    from poros.core import card
    from poros.core.arch_registry import registry_type, validated_arch_types
    from poros.plan.configs import read_model_config
    from poros.preflight import fit

    card.check(config, device)  # this card runs what the config asks, or a refusal
    model_json = read_model_config(model_name_or_path, config.model_revision)
    said = config.path_reason  # decided before here (poros train says that one itself)
    config.apply_cuda_defaults(model_json)
    err = path_refusal_error(config, model_json)
    if err is not None:  # a block-major path you set, refused before the download
        raise err
    if config.path_reason and config.path_reason != said:  # decided here, for a reason: once
        logger.warning("path: %s: %s", config.path, config.path_reason)
    # The architecture gate before the fit: an unvalidated dense model is told that,
    # not a fit's remedy (the arch gate in route, for the models config.json names).
    start = Startup(model_json=model_json)
    start.dense_mt = registry_type(model_json) if model_json else None
    if start.dense_mt is not None and start.dense_mt not in validated_arch_types("moe"):
        require_arch(start.dense_mt, config.blockwise_arch_check)
        start.arch_checked = True
        # a dense run adapted to fit, as `poros train`'s gate adapts it, then its refusals
        fit.gate(config, start.dense_mt, model_json, check_fit=check_fit,
                 check_host_ram=check_host_ram, say=logger.warning)
        if at_any_length and (err := fit.at_no_length(config, start.dense_mt, model_json)):
            raise err
        if config.offload_embeddings_and_head is None:  # no CUDA GPU the plan could read: off
            config.decide_offload(False)
    return start


def route(trainer: Any, model_name_or_path: str, start: Startup) -> str:
    """After the trainer's determinism and seeds, before the load: the compute's refusals, then
    where the model trains: ``"moe"`` (poros.moe), ``"metal_moe"`` (a Mac's expert layers, on the
    user's assertion) or ``"dense"``; a store only a MoE model takes is refused, and a dense
    model's architecture gated (a definitively resolved ``model_type`` only: anything
    unresolvable, offline, gated or trust_remote_code, falls through to the post-load gate)."""
    from poros import metal
    from poros.core.arch_registry import validated_arch_types
    from poros.preflight.arch import cheap_model_type

    config = trainer.config
    # Architecture gate BEFORE the weights download: config.json is a few KB and names
    # model_type; refusing an unvalidated arch after a 140 GB pull wastes the user's disk,
    # bandwidth, and hour.
    pre_mt = cheap_model_type(model_name_or_path, config.model_revision)
    if trainer._metal and pre_mt is not None:
        from poros.moe.carried import declares_visits

        if declares_visits(pre_mt):  # whatever its status or assertion: not on a Mac
            from poros.preflight.arch import LoopedOnAMacError, mac_looped, mac_looped_fix

            raise LoopedOnAMacError(f"{model_name_or_path} ({pre_mt}): {mac_looped(pre_mt)}.",
                                    remediation=mac_looped_fix(pre_mt))
    if config.moe_expert_format is not None and pre_mt is not None:
        from poros.moe.inkling2.formats import FAMILIES

        if pre_mt not in FAMILIES:  # a setting this model would not use: said, not ignored
            raise PorosConfigError(
                f"moe_expert_format: {config.moe_expert_format} says how Inkling 2 Mini's "
                f"experts are held; {model_name_or_path} ({pre_mt}) has no such choice.",
                remediation="Leave moe_expert_format unset.")
    # What the compute refuses, refused here, before the download: never after the load.
    trainer.compute.preflight("moe" if pre_mt in validated_arch_types("moe") else "dense",
                              start.model_json, pre_mt)
    if pre_mt in validated_arch_types("moe"):
        return "moe"
    if (trainer._metal and pre_mt is not None and pre_mt not in validated_arch_types()
            and metal.has_experts(model_name_or_path, config.model_revision)):
        # A family no one has validated still runs as what it is, on the user's assertion.
        require_arch(pre_mt, config.blockwise_arch_check)
        return "metal_moe"
    if config.store != "ram" and not trainer._metal:
        raise PorosConfigError(
            f"store={config.store!r} streams MoE experts from an SSD; "
            f"{model_name_or_path} is not a MoE model this release routes to poros.moe.",
            remediation="Use store='ram' (the default) for dense models.",
        )
    if pre_mt is not None and not (start.arch_checked and pre_mt == start.dense_mt):
        require_arch(pre_mt, config.blockwise_arch_check)
    if (said := tied_tables_note(config)) is not None:
        logger.warning(said)
    return "dense"


# The table modules a lora_target_modules entry can name (the last part of a module name).
_TABLES = ("embed_tokens", "lm_head", "wte", "embed_in", "embed_out", "word_embeddings")


def tied_tables_note(config: Any) -> Optional[str]:
    """A run whose LoRA targets the embedding table or the LM head of a model that ties the two
    (``tie_word_embeddings`` true in config.json, a few KB, read before the download) trains,
    but its merged export is refused: the merged checkpoint would store one table for both.
    Said before the download; None otherwise (also when config.json cannot be read)."""
    targets = config.lora_target_modules
    names = [targets] if isinstance(targets, str) else list(targets or ())
    hit = [n for n in names if str(n).split(".")[-1] in _TABLES]
    if not hit:
        return None
    from poros.plan.configs import read_model_config

    try:
        raw = read_model_config(config.model_name_or_path, config.model_revision) or {}
    except Exception:  # unreadable here: the post-load warning has nothing to add either
        return None
    text = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    if raw.get("tie_word_embeddings") is not True and text.get("tie_word_embeddings") is not True:
        return None
    return (f"lora_target_modules names {hit[0]}, and {config.model_name_or_path} ties its LM "
            "head to its embedding table (tie_word_embeddings): the run trains, and its adapter "
            "loads, but a merged export of it is refused (the merged checkpoint stores one table "
            "for both). To merge, train without the tables in lora_target_modules.")


def moe_settings_refusal(config: Any, metal: bool) -> Optional[PorosConfigError]:
    """What the MoE path does not support (target modules, a LoRA bias, the PEFT backend... ),
    refused before its load; None where it runs."""
    if config.torch_dtype == "float16":
        # The expert adapters are made in the store's dtype: fp16 ones would reach an optimizer
        # that updates bf16 and fp32 only (poros.train.expert_adamw), and the fused expert
        # kernels compute bf16. No MoE run in fp16 has been gated or certified.
        return PorosConfigError(
            f"{config.model_name_or_path} is a MoE model, and the MoE path does not train in "
            "torch_dtype float16: its experts and their adapters compute in bfloat16.",
            remediation="Set torch_dtype: bfloat16 (the default).",
        )
    from poros.core.checkpoints import MAX_LR, min_adam_epsilon

    if config.learning_rate > MAX_LR:
        # The expert adapters are bf16, so ExpertAdamW steps them, and it takes an lr up to 1:
        # above it a gradient its rule reads as zero would have moved a weight by more than a
        # rounding in fp32 AdamW (poros.train.expert_adamw, the rule's limits).
        return PorosConfigError(
            f"{config.model_name_or_path} is a MoE model, whose bf16 expert adapters ExpertAdamW "
            f"steps, and learning_rate {config.learning_rate:g} is above {MAX_LR:g}, the largest "
            "at which its rule and fp32 AdamW agree to within a rounding.",
            remediation=f"Set learning_rate to at most {MAX_LR:g} (a MoE recipe's is 2e-4).",
        )
    least = min_adam_epsilon(config.adam_beta2)
    if config.adam_epsilon < least:
        # The expert adapters are bf16, so ExpertAdamW steps them: under this eps its rule and
        # fp32 AdamW part by more than a rounding (poros.core.checkpoints.min_adam_epsilon).
        return PorosConfigError(
            f"{config.model_name_or_path} is a MoE model, whose bf16 expert adapters ExpertAdamW "
            f"steps, and adam_epsilon {config.adam_epsilon:g} is under {least:.3g}, the least at "
            f"which its rule and fp32 AdamW agree to within a rounding at adam_beta2 "
            f"{config.adam_beta2:g} (2**-39 / sqrt(1 - adam_beta2)).",
            remediation=f"Set adam_epsilon to at least {least:.3g} (the default 1e-8 is).",
        )
    unsupported = [
        f"{name}={getattr(config, name)!r}"
        for name, ok in (
            ("lora_target_modules", config.lora_target_modules is None),
            ("lora_bias", config.lora_bias == "none"),
            ("adapter_backend", config.adapter_backend != "peft"),
            ("block_size", config.block_size is None or metal),
            ("startup_parity_selfcheck", config.startup_parity_selfcheck == "off"),
        )
        if not ok
    ]
    if unsupported:
        return PorosConfigError(
            f"{config.model_name_or_path} is a MoE model, and the MoE path "
            f"does not support {', '.join(unsupported)}.",
            remediation="Remove these settings: MoE training quantizes the "
            "experts to NF4, streams them per expert and trains native LoRA on "
            "every expert projection.",
        )
    return None


def moe_on_cuda(config: Any, *, check_fit: bool = True, check_host_ram: bool = True,
                at_any_length: bool = False) -> None:
    """A MoE run's fit before its load: the path adapts whatever the checks, and each check runs
    on its own flag (``fit.gate``, entry ``moe``); ``at_any_length`` as ``on_cuda``'s."""
    from poros.core.arch_registry import registry_type
    from poros.plan.configs import read_model_config
    from poros.preflight import fit

    model_config = read_model_config(config.model_name_or_path, config.model_revision)
    store_room(config, model_config)
    model_type = registry_type(model_config) if model_config else None
    fit.gate(config, model_type, model_config,
             entry="moe", check_fit=check_fit, check_host_ram=check_host_ram, say=logger.warning)
    if at_any_length and (err := fit.at_no_length(config, model_type, model_config, entry="moe")):
        raise err


def store_room(config: Any, model_config: Optional[dict]) -> None:
    """The SSD store's .sss, sized from config.json before anything downloads: a store_path disk
    without room for every expert in NF4, laid out as the file is (its header, each slab padded
    to the next 4 KiB), is refused here (E_DISK), never after the download. A .sss already there
    is reused and needs none; an unreadable config.json is left to the load's own check."""
    import shutil

    from poros.core.errors import PorosDiskError
    from poros.plan.estimate import _sizes

    if config.store != "ssd" or not config.store_path:
        return
    from poros.moe.load import sss_file_path
    from poros.sss.format import ALIGNMENT, HEADER_NBYTES

    target = sss_file_path(config.store_path, config.model_name_or_path)
    from poros.moe.inkling2.formats import expected

    sizes = _sizes(model_config, expected(config, model_config)[0])
    if target.exists() or sizes is None or not sizes.experts_nf4:
        return
    slab = -(-sizes.experts_nf4 // sizes.expert_slabs)  # one (layer, expert) in NF4
    need = HEADER_NBYTES + sizes.expert_slabs * (-(-slab // ALIGNMENT) * ALIGNMENT)
    probe = target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if need > free:
        raise PorosDiskError(
            f"The SSD store needs {need / 2**30:.1f} GiB at {target.parent} for "
            f"{config.model_name_or_path}'s experts in NF4: {free / 2**30:.1f} GiB free.",
            remediation="Free some space there, or name another store_path (a local SSD).")
