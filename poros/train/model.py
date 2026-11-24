"""PorosModel: model loading and adapter attachment.

Handles NF4 4-bit loading via bitsandbytes, LoRA attachment via PEFT
or the native fallback, and provides access to trainable parameters.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re
import warnings
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from poros.core.errors import PorosConfigError
from poros.train.adapters import NativeAdapterManager

logger = logging.getLogger(__name__)


class PorosModel:
    """Wraps model loading + adapter attachment.

    Loads a HuggingFace model with its frozen base in NF4 and attaches LoRA
    adapters via PEFT or the native fallback.
    """

    def __init__(self) -> None:
        self._model: Optional[nn.Module] = None
        self._peft_model: Optional[nn.Module] = None
        self._adapter_backend: Optional[str] = None
        self._adapter_summary: Optional[Dict[str, Any]] = None
        self._trainable_parameter_names: List[str] = []
        # defer_table_cast: the frozen embedding table and LM head that k-bit preparation left
        # in their load dtype. They still owe it the fp32 cast (poros.core.tables), made where
        # they will live.
        self._uncast_tables: List[nn.Parameter] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        quantization: Literal["nf4"] = "nf4",
        torch_dtype: str = "bfloat16",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Union[List[str], str]] = None,
        lora_bias: str = "none",
        adapter_backend: Literal["auto", "peft", "native"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        nf4_load_strategy: Literal["auto", "gpu", "block_streamed"] = "auto",
        loader: Optional[Callable[[], nn.Module]] = None,
        resident: Optional[Tuple[int, int]] = None,
        defer_table_cast: bool = False,
    ) -> "PorosModel":
        """Load a HF model with its base in NF4 + LoRA adapters.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            quantization: "nf4", the only value (any other is refused,
                PorosConfigError).
            torch_dtype: Data type for the model.
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha scaling.
            lora_dropout: LoRA dropout rate.
            lora_target_modules: Target modules for LoRA (None = auto).
            lora_bias: "none", "all", or "lora_only".
            adapter_backend: "auto", "peft", or "native".
            revision: Hub revision (commit) to load the base from; None =
                default branch. Recorded by save() so eval/export re-pin it.
            loader: Builds the base model in place of transformers and
                bitsandbytes (the Mac path), with no kbit preparation.
            resident: ``(block_size, n)``: the streamed NF4 load keeps the
                run's ``n`` resident blocks on the GPU, out of host RAM.
            defer_table_cast: NF4 only. False (default): PEFT's k-bit
                preparation casts the frozen input embedding and LM head to
                fp32 on the host, as it does every 16-bit parameter. True:
                those tables (any a LoRA target names excepted) keep their
                load dtype and are listed in ``uncast_tables``; whoever places
                them makes that exact cast there (``poros.core.tables``), as
                PorosTrainer does.

        Returns:
            PorosModel with loaded model and adapters.
        """
        from poros.core.config import _refuse_other_precisions

        _refuse_other_precisions(quantization)

        instance = cls()
        backend = _resolve_adapter_backend(adapter_backend)
        instance._adapter_backend = backend

        # Load the model
        if nf4_load_strategy == "block_streamed":
            logger.info(
                "nf4_load_strategy='block_streamed': the bounded-GPU streamed load (one tensor "
                "at a time), with no fallback to the standard loader for a checkpoint it "
                "cannot stream."
            )
        dtype = getattr(torch, torch_dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"torch_dtype={torch_dtype!r} is not a torch dtype "
                "(expected e.g. 'bfloat16', 'float16', 'float32')."
            )
        model = loader() if loader is not None else _load_model(
            model_name_or_path, dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            revision=revision,
            nf4_load_strategy=nf4_load_strategy,
            resident=resident,
        )
        instance._model = model

        # Resolve target modules
        target_modules = _resolve_lora_target_modules(lora_target_modules)

        # Prepare for kbit training (the loader's own models are not bitsandbytes')
        if loader is None:
            instance._uncast_tables = _prepare_model_for_kbit_training(
                model, _untargeted_tables(model, target_modules) if defer_table_cast else [])

        # Attach adapters
        if backend == "peft":
            summary, peft_model = _attach_peft_lora(
                model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
                target_modules=target_modules,
            )
            # Keep the base module as the engine-facing model (the LoRA
            # layers are injected in place and shared), and keep the PEFT
            # wrapper for adapter-format save/load.
            instance._peft_model = peft_model
            instance._adapter_summary = summary
        else:
            manager = NativeAdapterManager(
                model=model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=target_modules,
                bias=lora_bias,
            )
            native_summary = manager.inject()
            instance._adapter_summary = {
                "backend": native_summary.backend,
                "target_module_names": native_summary.target_module_names,
                "trainable_param_count": native_summary.trainable_param_count,
                "total_param_count": native_summary.total_param_count,
            }

        # Collect trainable parameter names
        instance._trainable_parameter_names = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]

        logger.info(
            "PorosModel loaded: %s, quantization=nf4, backend=%s, "
            "trainable_params=%s",
            model_name_or_path,
            backend,
            len(instance._trainable_parameter_names),
        )

        return instance

    @property
    def model(self) -> nn.Module:
        """The underlying nn.Module."""
        if self._model is None:
            raise RuntimeError("Model has not been loaded yet.")
        return self._model

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        """All trainable (adapter) parameters."""
        if self._model is None:
            return []
        return [p for p in self._model.parameters() if p.requires_grad]

    @property
    def trainable_parameter_names(self) -> List[str]:
        """Names of all trainable parameters."""
        return list(self._trainable_parameter_names)

    @property
    def uncast_tables(self) -> List[nn.Parameter]:
        """Frozen vocabulary tables still in their load dtype that PEFT's preparation would have
        made fp32; whoever places them casts them (``poros.core.tables``)."""
        return list(self._uncast_tables)

    @property
    def peft_model(self) -> Optional[nn.Module]:
        """The PEFT wrapper around the model (None for the native backend)."""
        return self._peft_model

    @property
    def adapter_backend(self) -> Optional[str]:
        """The resolved adapter backend ("peft" or "native")."""
        return self._adapter_backend

    @property
    def adapter_summary(self) -> Optional[Dict[str, Any]]:
        """Summary of the adapter injection."""
        return self._adapter_summary

    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA into base weights and return the unloaded model.

        Returns:
            The merged nn.Module.
        """
        model = self.model
        if self._adapter_backend == "peft":
            # The PEFT wrapper lives in _peft_model; self.model deliberately
            # stays the base module (the engine-facing view), so merging must
            # go through the wrapper.
            peft_model = self._peft_model if self._peft_model is not None else model
            if hasattr(peft_model, "merge_and_unload"):
                return peft_model.merge_and_unload()
            raise RuntimeError(
                "PEFT model does not support merge_and_unload"
            )
        else:
            # The native manager can fold deltas into resident weights, but it
            # does not replace the wrapper modules with plain nn.Linear, and
            # folding a bf16 delta into a PACKED NF4 base is not valid merge
            # arithmetic: refused rather than returning the wrapped model.
            raise NotImplementedError(
                "merge_and_unload is not supported for adapter_backend="
                "'native': it cannot produce a plain, transformers-loadable "
                "module. Use poros export <run> (merged), or train with "
                "adapter_backend='peft' (the default)."
            )


# ── Private helpers ────────────────────────────────────────────────────


def _resolve_adapter_backend(backend: str) -> str:
    """Resolve 'auto' to 'peft' or 'native'."""
    if backend == "native":
        return "native"
    try:
        import peft  # noqa: F401

        return "peft"
    except ImportError:
        if backend == "peft":
            raise ImportError(
                "adapter_backend='peft' was requested but the peft package "
                "is not installed. Install with: pip install peft"
            )
        logger.info("PEFT is unavailable; using the native LoRA fallback")
        return "native"


def _load_model(
    model_name_or_path: str,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    revision: Optional[str] = None,
    nf4_load_strategy: str = "gpu",
    resident: Optional[Tuple[int, int]] = None,
) -> nn.Module:
    """Load a HuggingFace causal LM with its frozen base quantized to NF4."""
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load HuggingFace models. "
            "Install it with: pip install poros-train"
        ) from exc

    base_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        # Many target bases (Qwen, Phi, ...) ship custom modeling code.
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        base_kwargs["revision"] = revision
    from poros.core.arch_registry import hub_token

    if hub_token() is False:  # a family this command asserted: no stored token reaches its repo
        base_kwargs["token"] = False
    # An architecture with no SDPA implementation needs "eager". None = transformers default.
    if attn_implementation is not None:
        base_kwargs["attn_implementation"] = attn_implementation

    if nf4_load_strategy not in ("auto", "gpu", "block_streamed"):
        raise ValueError(
            f"Unknown nf4_load_strategy: {nf4_load_strategy!r} "
            "(expected 'auto', 'gpu' or 'block_streamed')"
        )
    if nf4_load_strategy == "block_streamed":
        # Bounded-GPU load (see poros/train/streamed_load.py). No
        # fallback: the caller opted into this path explicitly, so
        # failures must surface.
        from poros.train.streamed_load import load_nf4_block_streamed

        return load_nf4_block_streamed(
            model_name_or_path,
            dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            revision=revision,
            resident=resident,
        )
    if nf4_load_strategy == "auto" and torch.cuda.is_available():
        # Default route: the bounded-GPU streamed load (byte-identical to
        # the standard load on every validated family). It falls back to the standard
        # loader only where the input is one it cannot stream byte-exactly
        # (StreamedLoadUnsupported: no safetensors, a fusion-type weight
        # transform, a tensor the model does not define). Any other failure
        # -- out of memory, the Hub, a half-materialized model -- is raised
        # with its own cause: the standard loader holds the whole model on
        # the GPU, so retrying there would only fail again, misleadingly.
        from poros.train.streamed_load import load_nf4_block_streamed

        model = _streamed_or_none(lambda: load_nf4_block_streamed(
            model_name_or_path,
            dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            revision=revision,
            resident=resident,
        ))
        if model is not None:
            return model
    # nf4 means nf4: never silently substitute an unquantized model --
    # the run would train full-precision while its config, events.jsonl,
    # and saved artifacts all claim "nf4". Refuse honestly instead (the
    # CLI preflight gates CUDA the same way).
    if not torch.cuda.is_available():
        raise PorosConfigError(
            "The NF4 base needs a CUDA GPU (bitsandbytes 4-bit kernels), and "
            "none is available.",
            remediation="Run on a CUDA machine (on a Mac, poros train).",
            doc_url="docs/limitations.md",
        )
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            **base_kwargs,
        )
        return model
    except Exception as exc:
        # Route the remediation by what actually failed -- on the
        # exception TYPE first, text only as fallback. Two hard rules
        # from testing real failure strings: (1) install breakage is
        # checked BEFORE memory, because every bitsandbytes/driver
        # failure on this path mentions "cuda" somewhere; (2) HTTP
        # status codes are never matched as bare substrings -- an OOM
        # message's "404.06 MiB is free" must not become "model not
        # found".
        low = f"{type(exc).__name__}: {exc}".lower()
        try:
            from huggingface_hub.errors import (
                GatedRepoError,
                HFValidationError,
                RepositoryNotFoundError,
            )
        except ImportError:  # pragma: no cover
            GatedRepoError = HFValidationError = RepositoryNotFoundError = ()  # type: ignore[assignment]

        def _has(*needles: str) -> bool:
            return any(n in low for n in needles)

        if isinstance(exc, GatedRepoError):
            remedy = (
                f"{model_name_or_path!r} is a gated model: accept its "
                "license on the Hugging Face page, then run "
                "huggingface-cli login and retry."
            )
        elif isinstance(exc, HFValidationError):
            remedy = (
                f"{model_name_or_path!r} is not a valid model reference. "
                "Use 'org/name' for a Hub id or an existing local path "
                "(poros check <model> verifies one without downloading)."
            )
        elif isinstance(exc, RepositoryNotFoundError) or re.search(
            r"\b40[14]\b[^\n]{0,40}client error", low
        ):
            # HF answers unauthenticated requests for nonexistent AND
            # private repos with 401, so a typo and a login problem are
            # indistinguishable here -- say both.
            remedy = (
                f"Hugging Face could not serve {model_name_or_path!r}: "
                "the id may be misspelled (poros check <model> resolves "
                "it without downloading), or the repo is private/gated "
                "-- accept its license and run: huggingface-cli login"
            )
        elif _has(
            "bitsandbytes", "libbitsandbytes", "cuda setup",
            "no kernel image", "libcudart", "driver version",
            "not compiled with cuda",
        ):
            remedy = (
                "The CUDA / bitsandbytes install is broken for this GPU "
                "-- run poros machine to see what is wrong. Poros does "
                "not fall back to an unquantized model when nf4 was "
                "requested: the run's recorded precision must be true."
            )
        elif isinstance(
            exc, getattr(torch.cuda, "OutOfMemoryError", ())
        ) or _has("out of memory"):
            remedy = (
                "The GPU could not host the 4-bit load. Free VRAM, or "
                "start from a smaller model to confirm the setup."
            )
        elif _has(
            "connection", "timeout", "temporarily unavailable",
            "max retries",
        ):
            remedy = (
                "The download could not reach Hugging Face. Retry, or "
                "pre-fetch the model and point 'model_name_or_path' at "
                "the local directory."
            )
        elif _has("not a local folder", "does not appear to have"):
            remedy = (
                f"No model found at {model_name_or_path!r}. Verify the "
                "Hub id or the local path (poros check <model> tells you "
                "which it resolved, without downloading)."
            )
        else:
            remedy = (
                "Check the bitsandbytes install (poros machine). Poros "
                "does not fall back to an unquantized model when nf4 was "
                "requested -- the run's recorded precision must be true."
            )
        raise PorosConfigError(
            f"NF4 quantized load failed ({type(exc).__name__})",
            remediation=remedy,
            doc_url="docs/limitations.md",
        ) from exc


@contextlib.contextmanager
def pefts_checkpointing_names() -> Iterator[None]:
    """Around a call into PEFT: PEFT turns gradient checkpointing on, and reads whether it is
    on, by the names transformers has deprecated (``gradient_checkpointing_enable``,
    ``is_gradient_checkpointing``). Its calls, not this run's user's, so transformers'
    FutureWarning about each (with a path into the library) is not theirs to read. Those
    notices only, by their message; every other warning passes."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message=r"`(gradient_checkpointing_enable|gradient_checkpointing_disable|"
                    r"is_gradient_checkpointing)` is deprecated and will be removed")
        yield


def _prepare_model_for_kbit_training(
    model: nn.Module, hold: List[nn.Parameter]
) -> List[nn.Parameter]:
    """Prepare a quantized model for k-bit training if PEFT is available.

    PEFT casts every 16-bit parameter to fp32 where it lies, on the host here. The tables in
    ``hold`` (``poros.core.tables.vocab_tables``) are kept out of that cast, each a stand-in
    with no elements while it runs: a table in both dtypes at once, and kept in host RAM at
    twice its load size, is what holding them out saves. Returned: those of them PEFT did cast
    (their stand-in came back fp32). They still owe the cast, which the caller makes where
    each table lives (exact, so every value is PEFT's).
    """
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError:
        logger.info(
            "PEFT prepare_model_for_kbit_training unavailable; skipping. "
            "This is fine for native adapter training."
        )
        return []

    kwargs: Dict[str, Any] = {}
    sig = inspect.signature(prepare_model_for_kbit_training)
    if "use_gradient_checkpointing" in sig.parameters:
        kwargs["use_gradient_checkpointing"] = True
    if "gradient_checkpointing_kwargs" in sig.parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    held = [table.data for table in hold]
    for table in hold:
        table.data = table.data.new_empty(0)
    try:
        with pefts_checkpointing_names():
            prepare_model_for_kbit_training(model, **kwargs)
    except Exception as exc:
        logger.warning(
            "prepare_model_for_kbit_training failed (%s); continuing without it.",
            exc,
        )
    finally:
        owed = [table for table in hold if table.dtype == torch.float32]
        for table, data in zip(hold, held):
            table.data = data
    return owed


def _untargeted_tables(
    model: nn.Module, target_modules: Optional[Union[List[str], str]]
) -> List[nn.Parameter]:
    """The model's frozen vocabulary tables (``poros.core.tables.vocab_tables``) that no LoRA
    target names. An adapter is made in its base layer's dtype, so a table it targets must be
    fp32 first, as PEFT's preparation leaves it. Names are matched as PEFT or the native backend
    may match them (exactly, by suffix, or a string as a regular expression); "all-linear"
    leaves the head out and never takes an embedding."""
    from poros.core.tables import vocab_tables

    tables = vocab_tables(model)
    if not tables or target_modules is None or target_modules == "all-linear":
        return tables
    patterns = [target_modules] if isinstance(target_modules, str) else list(target_modules)

    def named(name: str) -> bool:
        return any(name == p or name.endswith(p) or (isinstance(target_modules, str)
                                                     and re.fullmatch(p, name) is not None)
                   for p in patterns)

    targeted = {id(p) for name, module in model.named_modules() if named(name)
                for p in module.parameters(recurse=False)}
    return [table for table in tables if id(table) not in targeted]


def _resolve_lora_target_modules(
    user_spec: Optional[Union[List[str], str]],
) -> Union[List[str], str]:
    """Which modules get LoRA: the ones named, else every linear layer ("all-linear")."""
    return user_spec if user_spec is not None else "all-linear"


def _attach_peft_lora(
    model: nn.Module,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    target_modules: Optional[Union[List[str], str]],
) -> tuple[Dict[str, Any], nn.Module]:
    """Attach PEFT LoRA to a model. Returns (adapter summary, PEFT model)."""
    from peft import LoraConfig, get_peft_model

    lora_kwargs: Dict[str, Any] = {
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": lora_bias,
    }

    sig = inspect.signature(LoraConfig)
    if target_modules is not None and "target_modules" in sig.parameters:
        lora_kwargs["target_modules"] = target_modules
    if "task_type" in sig.parameters:
        lora_kwargs["task_type"] = "CAUSAL_LM"

    lora_config = LoraConfig(**lora_kwargs)
    with pefts_checkpointing_names():
        peft_model = get_peft_model(model, lora_config)

    trainable = sum(
        param.numel() for param in peft_model.parameters() if param.requires_grad
    )
    total = sum(param.numel() for param in peft_model.parameters())
    # no print_trainable_parameters(): its stdout line lands inside the live run card, and these
    # counts are returned below for whoever needs them

    return {
        "backend": "peft",
        "target_module_names": target_modules,
        "trainable_param_count": trainable,
        "total_param_count": total,
    }, peft_model


def _streamed_or_none(load: Any) -> Any:
    """``load()`` (the block-streamed NF4 load), or None where the input is one it does not
    support (``StreamedLoadUnsupported``, said once): the standard loader takes those. Every
    other exception propagates."""
    from poros.train.streamed_load import StreamedLoadUnsupported

    try:
        return load()
    except StreamedLoadUnsupported as exc:
        logger.warning("Block-streamed NF4 load does not support this checkpoint (%s); using the "
                       "standard loader.", exc)
        return None
