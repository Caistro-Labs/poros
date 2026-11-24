"""Native LoRA adapter implementation for Poros.

Provides NativeLoraLinear, NativeAdapterManager, and get_trainable_state_dict
as a fallback when PEFT is not installed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

TargetSpec = Optional[Union[str, Sequence[str]]]


@dataclass
class NativeLoraSummary:
    """Summary of native LoRA injection."""

    backend: str
    target_module_names: List[str]
    trainable_param_count: int
    total_param_count: int


class NativeLoraLinear(nn.Module):
    """A minimal LoRA wrapper around ``nn.Linear``.

    The base linear layer stays frozen. The adapter update is

        y = base(x) + scale * B(A(dropout(x)))

    where A has shape ``(r, in_features)`` and B has shape ``(out_features, r)``.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("NativeLoraLinear only supports nn.Linear base layers")
        if rank < 1:
            raise ValueError("rank must be >= 1")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # fp32 adapters on any base, as PEFT makes them (its autocast_adapter_dtype): a quantized
        # base reports uint8, a bf16 or fp16 one its own dtype, and AdamW on adapters in either
        # half precision loses small updates. The forward casts the input to fp32 and the
        # update back to the base's output dtype.
        weight = base_layer.weight
        adapter_dtype = torch.float32
        adapter_device = weight.device

        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base_layer.in_features,
                device=adapter_device,
                dtype=adapter_dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.rank,
                device=adapter_device,
                dtype=adapter_dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base(x) + scaling * B(A(dropout(x)))."""
        result = self.base_layer(x)
        if self.merged:
            return result
        lora_x = self.dropout(x).to(self.lora_A.dtype)
        update = F.linear(lora_x, self.lora_A)
        update = F.linear(update, self.lora_B)
        return result + update.to(result.dtype) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Merge LoRA delta into base weights."""
        if self.merged:
            return
        delta = (
            (self.lora_B @ self.lora_A).to(self.base_layer.weight.dtype)
            * self.scaling
        )
        self.base_layer.weight.add_(delta)
        self.merged = True

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features


class NativeAdapterManager:
    """Inject and manage native LoRA adapters on a PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: TargetSpec = None,
        bias: str = "none",
        modules_to_save: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.bias = bias
        self.modules_to_save = tuple(modules_to_save or ())
        self.injected_module_names: List[str] = []
        self.modules_saved: List[str] = []

    @staticmethod
    def _iter_named_linears(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
        for name, module in model.named_modules():
            if not name:
                continue
            if isinstance(module, nn.Linear):
                yield name, module

    @staticmethod
    def _get_parent_module(
        model: nn.Module, module_name: str
    ) -> Tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    @staticmethod
    def _matches_any(name: str, patterns: Sequence[str]) -> bool:
        return any(name == pattern or name.endswith(pattern) for pattern in patterns)

    def resolve_target_module_names(self) -> List[str]:
        """Resolve which modules to apply LoRA to."""
        linear_names = [
            name
            for name, _ in self._iter_named_linears(self.model)
            if not name.endswith(("lm_head", "embed_out"))
        ]
        if not linear_names:
            return []

        spec = self.target_modules
        if spec is None or spec == "auto":
            attention_like = [
                name
                for name in linear_names
                if any(
                    token in name.split(".")[-1]
                    for token in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "query",
                        "key",
                        "value",
                        "c_attn",
                        "c_proj",
                    )
                )
            ]
            return attention_like or linear_names
        if isinstance(spec, str):
            if spec == "all-linear":
                return linear_names
            patterns = [spec]
        else:
            patterns = list(spec)
        return [name for name in linear_names if self._matches_any(name, patterns)]

    def _freeze_model(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def _mark_modules_to_save_trainable(self) -> None:
        if not self.modules_to_save:
            return
        for module_name, module in self.model.named_modules():
            if module_name and self._matches_any(module_name, self.modules_to_save):
                for param in module.parameters(recurse=True):
                    param.requires_grad = True
                self.modules_saved.append(module_name)

    def _mark_bias_trainable(self) -> None:
        if self.bias == "none":
            return
        if self.bias == "all":
            for name, param in self.model.named_parameters():
                if name.endswith("bias"):
                    param.requires_grad = True
            return
        # lora_only: only biases on injected modules
        targeted_bias_names = {
            f"{name}.base_layer.bias" for name in self.injected_module_names
        }
        for name, param in self.model.named_parameters():
            if name in targeted_bias_names:
                param.requires_grad = True

    def inject(self) -> NativeLoraSummary:
        """Inject LoRA adapters into the model."""
        self._freeze_model()
        target_names = self.resolve_target_module_names()
        if not target_names:
            raise ValueError(
                "No LoRA target modules were found. Set lora_target_modules "
                "explicitly or use a model with nn.Linear targets."
            )

        for name in target_names:
            parent, child_name = self._get_parent_module(self.model, name)
            child = getattr(parent, child_name)
            if not isinstance(child, nn.Linear):
                raise TypeError(
                    f"Native LoRA fallback only supports nn.Linear targets, "
                    f"but {name} is {type(child)}"
                )
            wrapped = NativeLoraLinear(
                base_layer=child,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
            )
            setattr(parent, child_name, wrapped)
            self.injected_module_names.append(name)

        self._mark_modules_to_save_trainable()
        self._mark_bias_trainable()

        trainable = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        total = sum(param.numel() for param in self.model.parameters())
        logger.info(
            "Native LoRA injected into %s modules; trainable params: %s / %s (%.4f%%)",
            len(self.injected_module_names),
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )
        return NativeLoraSummary(
            backend="native",
            target_module_names=list(self.injected_module_names),
            trainable_param_count=trainable,
            total_param_count=total,
        )

    @staticmethod
    def merge_and_unload(model: nn.Module) -> None:
        """Merge all native LoRA adapters in the model."""
        for module in model.modules():
            if isinstance(module, NativeLoraLinear):
                module.merge()


def validate_peft_adapter_state(model: nn.Module, state: Dict[str, "torch.Tensor"],
                                source: str) -> None:
    """Exact-match validation of a saved PEFT adapter BEFORE it is applied.

    The one shared gate for every loader (resume, eval, export): PEFT's own
    ``get_peft_model_state_dict`` defines the expected key set and shapes --
    the same convention ``save_pretrained`` writes -- so a truncated file
    (missing keys), a foreign file (extra keys), or a reshaped adapter all
    fail HERE, with the model untouched. Checking only unexpected_keys after
    ``set_peft_model_state_dict`` cannot catch truncation: every surviving
    key still has a slot, and the missing tensors silently stay at init.
    """
    from peft import get_peft_model_state_dict

    # NOTE: the oracle is computed here, at load time; PEFT's
    # save_embedding_layers="auto" is NOT a pure function of the model (it
    # resolves base_model_name_or_path, sometimes over the network), so the
    # embedding/lm_head keys can legitimately differ between what a save
    # wrote and what this call expects. Those keys are therefore ADVISORY:
    # warned about, never grounds for refusal. Every adapter (lora_*) key
    # is strict. (set_peft_model_state_dict may also rename keys for some
    # MoE families under transformers>=5; identity for every poros-validated
    # family, checked qwen2/gemma3_text.)
    from peft.utils.other import AuxiliaryTrainingWrapper

    from poros.adapters.inspect import is_adapter_param

    # TRAINED tensors are never advisory: adapter (lora_*) tensors by name,
    # and modules_to_save/trainable-tokens copies structurally (their saved
    # keys are name-identical to the optional base copies, so the wrapper
    # module set is the only reliable discriminator). Advisory = only the
    # OPTIONAL COPIED BASE embed/lm_head tensors, wrapped
    # (.base_layer.weight, when the module is a LoRA target) or bare.
    aux_prefixes = tuple(
        n + "." for n, m in model.named_modules()
        if isinstance(m, AuxiliaryTrainingWrapper)
    )

    def _advisory(k: str) -> bool:
        return (
            not is_adapter_param(k)
            and not k.startswith(aux_prefixes)
            and k.endswith((
                "embed_tokens.weight", "lm_head.weight",
                "embed_tokens.base_layer.weight", "lm_head.base_layer.weight",
            ))
        )

    expected = get_peft_model_state_dict(model)
    missing = sorted(expected.keys() - state.keys())
    extra = sorted(state.keys() - expected.keys())
    missing_hard = [k for k in missing if not _advisory(k)]
    extra_hard = [k for k in extra if not _advisory(k)]
    if missing_hard or extra_hard:
        raise RuntimeError(
            f"Adapter state in {source} does not match this model: "
            f"{len(missing_hard)} expected tensor(s) missing"
            + (f" (e.g. {missing_hard[0]!r})" if missing_hard else "")
            + f", {len(extra_hard)} unexpected"
            + (f" (e.g. {extra_hard[0]!r})" if extra_hard else "")
            + ". Partial or truncated saves, and adapters from a different "
            "base or lora_target_modules, are refused before any tensor is "
            "applied."
        )
    if missing or extra:
        import warnings

        warnings.warn(
            f"Adapter state in {source}: embedding/lm_head key set differs "
            f"from this load's expectation ({len(missing)} missing, "
            f"{len(extra)} extra) -- a save_embedding_layers resolution "
            "difference, not a truncated adapter. Proceeding.",
            stacklevel=2,
        )
    bad = [k for k in expected
           if k in state and tuple(state[k].shape) != tuple(expected[k].shape)]
    if bad:
        raise RuntimeError(
            f"Adapter state in {source} has {len(bad)} shape-mismatched "
            f"tensor(s) (e.g. {bad[0]!r}: {tuple(state[bad[0]].shape)} vs "
            f"expected {tuple(expected[bad[0]].shape)}) -- different "
            "lora_rank or base model."
        )


def validate_native_adapter_state(model: nn.Module,
                                  state: Dict[str, "torch.Tensor"],
                                  source: str) -> None:
    """Native counterpart: state keys ARE model parameter names, so the
    exact trainable key set AND shapes are required, before any tensor is
    applied."""
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    problems = (
        [f"missing {k!r}" for k in sorted(expected.keys() - state.keys())]
        + [f"unexpected {k!r}" for k in sorted(state.keys() - expected.keys())]
        + [f"shape {k!r} {tuple(state[k].shape)} != {tuple(v.shape)}"
           for k, v in expected.items()
           if k in state and tuple(state[k].shape) != tuple(v.shape)]
    )
    if problems:
        raise RuntimeError(
            f"Adapter state in {source} does not match the model's trainable "
            f"set ({len(problems)} problem(s): {'; '.join(problems[:3])}"
            + (" ..." if len(problems) > 3 else "") + ") -- partial, "
            "truncated, or saved from a different model/LoRA config."
        )


def get_trainable_state_dict(
    model: nn.Module,
    trainable_names: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Return a CPU state dict containing only trainable parameters.

    Keeps adapter checkpoints compact for both the native fallback and PEFT
    models where the trainable tensors are exposed as standard named parameters.
    """
    selected_names: Set[str]
    if trainable_names is None:
        selected_names = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
    else:
        selected_names = set(trainable_names)
    state = model.state_dict()
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if name in selected_names
    }
