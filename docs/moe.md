# MoE models

A mixture-of-experts (MoE) model has many expert networks in each layer and sends each token to a
few of them. MoE models train with the same commands, configs and Python as dense ones.

```bash
poros train qwen3-30b-a3b-instruct-2507 --data chats.jsonl
poros train qwen3-30b-a3b-instruct-2507 --data chats.jsonl --store ssd   # experts on an SSD
```

`poros check <model> --json` says `"entry": "moe"` for a model that takes the MoE route.
[Supported families](models.md#supported-families) lists the MoE families.

## How it trains

- **Load.** Poros reads the checkpoint shard by shard and quantizes each expert to NF4 as it reads
  it, so the bf16 experts are never all in memory at once. The result has the same bytes as a full
  load followed by quantization.
- **Train.** The NF4 experts wait in host RAM, or on an SSD, and stream to the GPU expert by
  expert; the other layers stay bf16 and stream by block. LoRA trains every routed expert's
  `gate_up_proj` and `down_proj`; the router, attention and shared experts stay frozen.
- **Optimizer.** The expert adapters are bf16, where plain AdamW drops an update smaller than a
  weight's last bit. Poros steps them with its own AdamW: each update is computed in fp32 from
  the weight and a remainder kept beside it, and the remainder and both moments are rounded up
  or down by bits keyed to the seed, the step and the tensor, so two runs of one config round
  the same way. Each checkpoint records the rule in `optimizer.json`.
- **Expert cache.** On `path: fast`, after the first step, the experts kept on the GPU grow to what
  it leaves free, less the larger of 3 GiB and a fifth of the GPU memory (VRAM) the run could hold
  (`POROS_MOE_CACHE_HEADROOM` sets the fraction). `moe_expert_cache_gib` caps it. On
  `path: efficient` it holds one layer. The cache decides when bytes move.
- **Compute.** On CUDA the experts run as fused Triton kernels over each expert's exact rows
  (`moe_expert_compute: fused`), a micro-batch at a time, so they suit larger micro-batches
  (`per_device_batch_size: 8`, `gradient_accumulation_steps: 1`). Gemma 4 MoE's GELU experts take
  the reference compute.
- **Export.** `poros export` merges one expert at a time on the CPU, so it needs no GPU. On CUDA
  `--format adapter` writes Poros's own format, which loads with `poros.Run.open`; a Hub push
  carries its weights. `--format merged` gives a model any server loads.

[The SSD expert store](streaming.md#the-ssd-expert-store) keeps the experts in a file on an SSD
instead of host RAM, and says when to pick it and `path=fast+cache`.

## Store plugins

An installed package can add an expert store by registering a `poros.stores.StorePlugin` in the
`poros.stores` entry-point group:

```toml
[project.entry-points."poros.stores"]
mystore = "my_package.poros_plugin:STORE"
```

Select it with `--store mystore:/path`, `store: mystore`, or `store="mystore"`. Its `load_model`
returns the model with the store on `model._poros_moe_store`, as
`poros.moe.load.load_moe_streamed` does. The store must keep expert data immutable and make
`load_slice` a pure byte copy; showing [Reproducibility](training.md#reproducibility) holds on its
bytes is the plugin's job. A store no installed package provides is refused before anything loads,
naming `store_package` when the config sets it.

## Limitations

- **Families.** On a GPU, a MoE family that is not supported is refused, even with
  `--unvalidated`. Load DeepSeek-V2 without `trust_remote_code`.
- **Adapters.** `lora_target_modules` and `lora_bias` are refused before anything loads, and so is
  `block_size` on a GPU.
- **Settings.** `torch_dtype: float16`, an `adam_epsilon` under 2^-39 / sqrt(1 - beta2) (5.75e-11
  at the default beta2) and a learning rate above 1 are refused before anything loads.
- **Inkling 2 Mini.** Its looped decoder is in the code and not yet validated on a GPU, so it
  does not train in this release.
- **Host RAM.** Without the SSD store the experts wait in host RAM: [Host
  RAM](training.md#host-ram).
- **Router aux loss.** A model whose config sets `output_router_logits` trains on the sequential
  path.
- **transformers 5.13 to 5.22.** The MoE route is tested on these; Poros installs 5.20 to 5.22.
  An older release is refused and a newer one warns.

## Next

- [Streaming and memory](streaming.md): where the experts wait, and which path to pick.
- [Training](training.md): objectives, data, checkpoints and what fits.
