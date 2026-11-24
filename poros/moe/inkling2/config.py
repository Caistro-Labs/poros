"""Inkling 2 Mini's text configuration, as Poros carries it (no transformers class needed).

The released checkpoint (``thinkingmachines/Inkling-2-Mini``) is a multimodal wrapper,
``model_type: inkling2_loop``, whose ``text_config`` describes the decoder Poros trains. Its keys
are Thinking Machines' own (``intermediate_size``, ``route_scale``, ``sconv_kernel_size``,
``local_layer_ids``, ``loop_config``); transformers 5.24 reads them through an attribute map, and
so does this class. The loop's bounds and pass count come from ``loop_config`` and are never
assumed: the variable-pass research checkpoint differs only by its pass count.

The field set and defaults follow transformers 5.24.0
``models/inkling2_loop/configuration_inkling2_loop.py`` (Apache-2.0, see NOTICE).
"""

from __future__ import annotations

from typing import Any

from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig

__all__ = ["Inkling2LoopTextConfig", "Inkling2LoopConfig", "text_config_of", "register"]


@strict
class Inkling2LoopTextConfig(PreTrainedConfig):
    """The decoder of Inkling 2 Mini: 24 layers, the middle ones run ``loop_num_passes`` times."""

    model_type = "inkling2_loop_text"
    #: Its decoder runs its layers as visits (the loop): what poros.moe.carried.declares_visits
    #: reads, so a path that runs each layer once (the Mac's) refuses it.
    poros_visits = True
    attribute_map = {
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
        "num_experts": "n_routed_experts",
        "sconv_kernel_size": "conv_kernel_size",
        "model_max_length": "max_position_embeddings",
        "intermediate_size": "moe_intermediate_size",
        "routed_scaling_factor": "route_scale",
    }

    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 24
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    layer_types: list[str] | None = None
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    hidden_act: str = "silu"
    moe_intermediate_size: int = 1152
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    route_scale: float = 8.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = 200006
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float | None = 30.0
    scoring_func: str = "sigmoid"
    use_cache: bool = False
    tie_word_embeddings: bool = False
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 3
    loop_gate_rank: int = 64

    def __post_init__(self, **kwargs: Any) -> None:
        loop = kwargs.pop("loop_config", None)
        if loop is not None:
            self.loop_layer_index_start = int(loop.get("core_start", self.loop_layer_index_start))
            self.loop_layer_index_end = int(loop.get("core_end", self.loop_layer_index_end))
            self.loop_num_passes = int(loop.get("num_passes", self.loop_num_passes))
            self.loop_gate_rank = int(loop.get("gate_rank", self.loop_gate_rank))
        local = kwargs.pop("local_layer_ids", None)
        if self.layer_types is None:
            # transformers 5.24 derives the layer types from this rule and ignores local_layer_ids;
            # the released config's local_layer_ids say the same. A config whose ids say
            # otherwise is refused rather than read two ways.
            self.layer_types = [
                "hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding"
                for i in range(self.num_hidden_layers)
            ]
            if local is not None:
                from_ids = [
                    "hybrid_sliding" if i in set(local) else "hybrid"
                    for i in range(self.num_hidden_layers)
                ]
                if from_ids != self.layer_types:
                    raise ValueError(
                        "local_layer_ids disagree with the layer types transformers derives for "
                        "this config; pass layer_types explicitly"
                    )
        super().__post_init__(**kwargs)
        self.validate_loop()

    def validate_loop(self) -> None:
        start, end, block = (
            self.loop_layer_index_start,
            self.loop_layer_index_end,
            self.n_layers_per_attn_res_block,
        )
        if self.loop_num_passes < 1:
            raise ValueError(f"loop_num_passes ({self.loop_num_passes}) must be at least 1")
        if not 0 < start < end <= self.num_hidden_layers:
            raise ValueError(
                f"the loop [{start}, {end}) must lie inside layers 1..{self.num_hidden_layers}"
            )
        if start % block or end % block:
            raise ValueError(
                f"the loop bounds [{start}, {end}) must be multiples of "
                f"n_layers_per_attn_res_block ({block})"
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must name every decoder layer")

    @property
    def visits_per_forward(self) -> int:
        """Decoder-layer calls per forward: every layer once, the loop layers once per pass."""
        loop = self.loop_layer_index_end - self.loop_layer_index_start
        return self.num_hidden_layers + (self.loop_num_passes - 1) * loop


@strict
class Inkling2LoopConfig(PreTrainedConfig):
    """The released wrapper (``model_type: inkling2_loop``): Poros reads its ``text_config`` only;
    the vision tower and the multi-token-prediction head are left behind."""

    model_type = "inkling2_loop"
    poros_visits = True  # its text decoder's (Inkling2LoopTextConfig.poros_visits)
    sub_configs = {"text_config": Inkling2LoopTextConfig}

    text_config: Inkling2LoopTextConfig | dict | None = None
    vision_config: dict | None = None
    mtp_config: dict | None = None
    quantization_config: dict | None = None
    image_token_id: int = 200054
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs: Any) -> None:
        if self.text_config is None:
            self.text_config = Inkling2LoopTextConfig()
        elif isinstance(self.text_config, dict):
            text = {k: v for k, v in self.text_config.items() if k != "model_type"}
            self.text_config = Inkling2LoopTextConfig(**text)
        super().__post_init__(**kwargs)

    def get_text_config(self, *args: Any, **kwargs: Any) -> Inkling2LoopTextConfig:
        return self.text_config


#: model_type -> its config class (poros.moe.carried reads it)
CONFIG_CLASSES: dict = {}


def text_config_of(config: Any) -> Inkling2LoopTextConfig:
    """The decoder config of a wrapper or text config (a dict as config.json holds it, too)."""
    if isinstance(config, dict):
        if config.get("model_type") == "inkling2_loop":
            config = dict(config.get("text_config") or {})
        config = {k: v for k, v in config.items() if k != "model_type"}
        return Inkling2LoopTextConfig(**config)
    if isinstance(config, Inkling2LoopConfig):
        return config.text_config
    if isinstance(config, Inkling2LoopTextConfig):
        return config
    raise TypeError(f"not an Inkling 2 Mini config: {type(config).__name__}")


def register() -> bool:
    """Register the config and model classes with transformers' Auto classes, unless this
    transformers already has its own (5.24 and newer, which Poros does not pin). Returns whether
    Poros's classes are the registered ones. Idempotent."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from poros.moe.inkling2.modeling import (
        Inkling2LoopForCausalLM,
        Inkling2LoopWrappedForCausalLM,
    )

    known = CONFIG_MAPPING._extra_content
    for model_type, cls in (
        ("inkling2_loop_text", Inkling2LoopTextConfig),
        ("inkling2_loop", Inkling2LoopConfig),
    ):
        if known.get(model_type) is cls:  # registered already
            continue
        if model_type in CONFIG_MAPPING:
            return False  # transformers' own Inkling 2 Mini
        AutoConfig.register(model_type, cls)
    for config_cls, model_cls in (
        (Inkling2LoopTextConfig, Inkling2LoopForCausalLM),
        (Inkling2LoopConfig, Inkling2LoopWrappedForCausalLM),
    ):
        AutoModelForCausalLM.register(config_cls, model_cls, exist_ok=True)
    return True


CONFIG_CLASSES.update({"inkling2_loop": Inkling2LoopConfig,
                       "inkling2_loop_text": Inkling2LoopTextConfig})
