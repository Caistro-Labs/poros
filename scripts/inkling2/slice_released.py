"""Cut a smaller Inkling 2 Mini from the released checkpoint, in the released format.

For the oracle comparison on real weights (tests/moe/test_inkling2_oracle.py with
``POROS_INKLING_TINY_CHECKPOINT``): both implementations hold the whole slice in fp32 on the CPU,
which the full model (26B) does not allow. The slice keeps real weights and real NVFP4 / MXFP8
bytes, and is still a looped model:

- decoder layers: a list of the release's layers (default 0-17: the loop keeps both its
  residual blocks, 6-11 and 12-17, so a later pass reads a block state the pass before
  appended, and every layer kind is in it; MXFP8 layers 20-23 are not); the loop is the kept
  layers inside the release's loop;
- experts: the first ``--experts`` of every layer (and the router's rows for them);
- the vision tower and the MTP head are left out.

    python scripts/inkling2/slice_released.py SRC DST [--layers 0-17] [--experts 9]

SRC is the released checkpoint folder (config.json, model.safetensors.index.json, shards).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--layers", default="0-17")
    parser.add_argument("--experts", type=int, default=9)
    args = parser.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    keep = _layers(args.layers)
    new = {old: i for i, old in enumerate(keep)}
    loop = text["loop_config"]
    loop_kept = [i for i in keep if loop["core_start"] <= i < loop["core_end"]]
    per_block = int(text["n_layers_per_attn_res_block"])
    if len({i // per_block for i in loop_kept}) < 2:  # a one-block loop never carries a block
        raise SystemExit("keep at least two residual blocks of the loop (layers 6-17)")
    k = args.experts

    def target(name: str):
        """(new name, how to cut) for a checkpoint tensor, or None to leave it out."""
        if name.startswith(("model.visual.", "mtp.")):
            return None
        m = re.match(r"^model\.llm\.layers\.(\d+)\.(.*)$", name)
        if m:
            old, rest = int(m[1]), m[2]
            if old not in new:
                return None
            cut = None
            if rest.startswith("mlp.experts.") and not rest.endswith("outer_scale_denominator"):
                cut = "experts"
            elif rest in ("mlp.gate.weight", "mlp.gate.bias"):
                cut = "experts"
            return f"model.llm.layers.{new[old]}.{rest}", cut
        m = re.match(r"^model\.llm\.attn_res\.attn_queries\.(\d+)$", name)
        if m:  # attn_queries.j is layer j + 1's
            layer = int(m[1]) + 1
            if layer not in new or new[layer] == 0:
                return None
            return f"model.llm.attn_res.attn_queries.{new[layer] - 1}", None
        m = re.match(r"^model\.llm\.attn_res\.mlp_queries\.(\d+)$", name)
        if m:
            layer = int(m[1])
            return (f"model.llm.attn_res.mlp_queries.{new[layer]}", None) if layer in new else None
        if name.startswith("model.llm.middle_cycle_cross_pass_"):
            return name, "loop"
        return name, None

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    loop_rows = [2 * (i - loop["core_start"]) + r for i in loop_kept for r in (0, 1)]
    out: dict[str, "torch.Tensor"] = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(src / shard), "pt") as f:
            for name in f.keys():
                spec = target(name)
                if spec is None:
                    continue
                new_name, cut = spec
                tensor = f.get_tensor(name)
                if cut == "experts":
                    tensor = tensor[:k]
                elif cut == "loop":
                    tensor = tensor[loop_rows]
                out[new_name] = tensor.contiguous()
    save_file(out, str(dst / "model.safetensors"))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {}, "weight_map": dict.fromkeys(sorted(out), "model.safetensors")},
            indent=1,
        )
    )
    types = ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(len(keep))]
    original = ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in keep]
    if types != original:
        raise SystemExit("the kept layers break the every-sixth-layer-global pattern")
    text = dict(
        text,
        num_hidden_layers=len(keep),
        n_routed_experts=k,
        num_experts_per_tok=min(int(text["num_experts_per_tok"]), k),
        local_layer_ids=[i for i, t in enumerate(types) if t == "hybrid_sliding"],
        loop_config=dict(loop, core_start=new[loop_kept[0]], core_end=new[loop_kept[-1]] + 1),
    )
    # the slice's experts keep the release's bytes; transformers decodes them from their names.
    # config.json's quantization_config goes (transformers would quantize by it); the
    # hf_quant_config.json beside it is copied, so the slice still says it is quantized before
    # its weights are read (poros.moe.inkling2.formats.state_before_download)
    config = {
        key: v
        for key, v in config.items()
        if key not in ("vision_config", "mtp_config", "quantization_config")
    }
    config["text_config"] = text
    (dst / "config.json").write_text(json.dumps(config, indent=1))
    for extra in (
        "hf_quant_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        if (src / extra).is_file():
            (dst / extra).write_bytes((src / extra).read_bytes())
    print(
        f"{dst}: {len(keep)} layers, loop {text['loop_config']['core_start']}-"
        f"{text['loop_config']['core_end'] - 1}, {k} experts, {len(out)} tensors"
    )


if __name__ == "__main__":
    main()
