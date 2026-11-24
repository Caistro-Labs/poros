"""A tiny random Inkling 2 Mini for the CPU tests: every kind of layer and every path of the loop.

Six layers, a block attention residual every two (so the stack grows inside and outside the
loop), layers 2 and 3 run three times; sliding and global layers alternate, the window (5) and
the relative-bias extent (8) are shorter than the test rows and the log scaling starts at
position 4, so every masking and scaling branch runs.
"""

from __future__ import annotations

import torch

from poros.moe.inkling2.codec import E2M1_VALUES
from poros.moe.inkling2.config import Inkling2LoopTextConfig
from poros.moe.inkling2.modeling import Inkling2LoopForCausalLM, cast

CONFIG = dict(
    vocab_size=128,
    unpadded_vocab_size=120,
    hidden_size=64,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=16,
    sliding_window_size=5,
    d_rel=4,
    rel_extent=8,
    log_scaling_n_floor=4,
    layer_types=[
        "hybrid_sliding",
        "hybrid",
        "hybrid_sliding",
        "hybrid",
        "hybrid_sliding",
        "hybrid",
    ],
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_layers_per_attn_res_block=2,
    loop_layer_index_start=2,
    loop_layer_index_end=4,
    loop_num_passes=3,
    loop_gate_rank=8,
)
VOCAB = 120

#: Loops of two residual blocks (the release's loop spans two: layers 6-17, blocks of 6), so a
#: pass after the first takes a block state from the one before (``blocks[len(pre) + 1:]``):
#: 8 layers looping 2-5 in blocks of 2, and 9 layers looping 3-8 in blocks of 3 with no layer
#: after the loop. CONFIG's own loop is one block.
SHAPES = {
    "one_block": {},
    "two_blocks_8": dict(
        num_hidden_layers=8,
        n_layers_per_attn_res_block=2,
        loop_layer_index_start=2,
        loop_layer_index_end=6,
        layer_types=["hybrid_sliding", "hybrid"] * 4,
    ),
    "two_blocks_9": dict(
        num_hidden_layers=9,
        n_layers_per_attn_res_block=3,
        loop_layer_index_start=3,
        loop_layer_index_end=9,
        layer_types=(["hybrid_sliding", "hybrid"] * 5)[:9],
    ),
}


def config(**overrides) -> Inkling2LoopTextConfig:
    return Inkling2LoopTextConfig(**{**CONFIG, **overrides})


def model(
    dtype: torch.dtype = torch.float32, seed: int = 0, **overrides
) -> Inkling2LoopForCausalLM:
    """Random weights, every parameter drawn (the residual queries large enough that the
    attention over the block states is far from uniform; scales and norm weights near one)."""
    torch.manual_seed(seed)
    m = Inkling2LoopForCausalLM(config(**overrides))
    with torch.no_grad():
        for name, param in m.named_parameters():
            param.normal_(0, 0.3 if "query" in name else 0.05)
            if name.endswith("scale") or "norm.weight" in name or "layernorm" in name:
                param.add_(1.0)
    return cast(m, dtype)


def ids(rows: int = 2, length: int = 16, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, VOCAB, (rows, length), generator=torch.Generator().manual_seed(seed))


def randomize_lora_b(m: torch.nn.Module, seed: int = 5) -> None:
    """LoRA B starts at zero, which leaves every A without a gradient: draw it."""
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(seed)
        for name, param in m.named_parameters():
            if param.requires_grad and "lora_B" in name:
                param.normal_(0, 0.05)


def encode_nvfp4(x: torch.Tensor) -> dict:
    """Encode ``[experts, rows, cols]`` as the release stores NVFP4 (nearest code; the bytes
    are what both loads decode, so their rounding needs no care)."""
    experts, rows, cols = x.shape
    x = x.float()
    amax = x.abs().amax(dim=(1, 2))
    denominator = torch.tensor([6.0 * 448.0])
    factor = amax / denominator
    blocks = x.view(experts, rows, cols // 16, 16)
    scale = (blocks.abs().amax(-1) / (6.0 * factor[:, None, None])).to(torch.float8_e4m3fn)
    lut = torch.tensor(E2M1_VALUES)
    unit = blocks / (scale.float()[..., None] * factor[:, None, None, None]).clamp(min=1e-30)
    codes = (unit[..., None] - lut).abs().argmin(-1).view(experts, rows, cols)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return {
        "weight": packed,
        "weight_scale": scale,
        "weight_amax": amax,
        "weight_outer_scale_denominator": denominator,
    }


def encode_mxfp8(x: torch.Tensor) -> dict:
    experts, rows, cols = x.shape
    blocks = x.float().view(experts, rows, cols // 32, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp(min=1e-30) / 448.0)) + 127
    values = (blocks / torch.exp2(exponent - 127)[..., None]).view(experts, rows, cols)
    return {"weight": values.to(torch.float8_e4m3fn), "weight_scale": exponent.to(torch.uint8)}


def write_released(path, dtype=None, quantize=True, **overrides) -> dict:
    """The tiny model written as the release stores Inkling 2 Mini: its names and layout
    (``codec.to_checkpoint``), every layer's experts in NVFP4 but the last's in MXFP8 (or left
    unquantized: ``quantize`` False, or a set of the layers to quantize), three shards
    interleaved so that tensors which make one parameter land apart, the wrapper config
    (``model_type: inkling2_loop``) and a vision tensor the decoder does not read. Returns the
    tensors written."""
    import json

    from safetensors.torch import save_file

    from poros.moe.inkling2 import codec

    m = model(dtype or torch.bfloat16, **overrides)
    config = m.config
    tensors = codec.to_checkpoint(m.state_dict(), config)
    last = config.num_hidden_layers - 1
    layers = (
        range(config.num_hidden_layers)
        if quantize is True
        else ()
        if quantize is False
        else sorted(quantize)
    )  # a set: those layers only
    for layer in layers:
        encode = encode_mxfp8 if layer == last else encode_nvfp4
        for which in ("w13", "w2"):
            name = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
            for part, tensor in encode(tensors.pop(name)).items():
                tensors[f"model.llm.layers.{layer}.mlp.experts.{which}_{part}"] = tensor
    tensors["model.visual.proj_in.weight"] = torch.zeros(2, 2)
    names = sorted(tensors)
    shards = [names[i::3] for i in range(3)]  # interleaved: parts of one tensor split apart
    weight_map = {}
    for i, shard in enumerate(shards):
        file = f"model-{i + 1:05d}-of-00003.safetensors"
        save_file({n: tensors[n].contiguous() for n in shard}, str(path / file))
        weight_map.update(dict.fromkeys(shard, file))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    text = {
        k: v
        for k, v in config.to_dict().items()
        if k in CONFIG or k in overrides or k == "model_type"
    }
    (path / "config.json").write_text(
        json.dumps({"model_type": "inkling2_loop", "text_config": text})
    )
    return tensors
