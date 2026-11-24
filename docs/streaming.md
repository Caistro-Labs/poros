# Streaming and memory

Poros trains models that are larger than your GPU's memory. It keeps the frozen base off the GPU
and copies it onto the card a few layers at a time, as training needs them. This page says where
the base waits and which settings to pick for your machine.

## What streaming means

Poros trains LoRA adapters, small matrices added to a model whose own weights stay frozen and
quantized to 4 bits (NF4). In each step:

1. Poros copies a block of layers to the GPU, 4 by default (`block_size`).
2. It runs the block and lets it go, while the next block is already being copied in.
3. In the backward pass, it copies each block in again, recomputes it, and computes the adapters'
   gradients.

What GPU memory (VRAM) must hold is one block and the adapters, not the whole model; the cost
is time, as the base crosses to the GPU twice a step. A dense model the GPU holds with room to spare can
keep some blocks on the GPU for the whole run ([Plenty of both](#plenty-of-both)).

## Where the weights wait

| Machine | Model | Where the frozen base waits |
|---|---|---|
| NVIDIA GPU | dense | host RAM |
| NVIDIA GPU | mixture of experts | the experts in host RAM (default) or in a file on an SSD (`--store ssd`); the other layers in host RAM |
| Mac | any | one pool of memory shared by CPU and GPU; the frozen layers held in memory or read from the drive |

### The SSD expert store

A mixture-of-experts (MoE) model has many experts in each layer. `--store ssd` keeps them in a
`.sss` file on a local SSD, so host RAM doesn't have to hold them:

```bash
poros train qwen3-30b-a3b-instruct-2507 --data chats.jsonl --store ssd
```

The file goes in `~/.poros/sss` unless you name a folder, such as `--store /mnt/nvme`. The run that
creates it quantizes the experts straight into it, after checking the disk has room (`E_DISK`
if not). Later runs reuse it while the model, the quantization and the Poros and torch versions
match, and rebuild it otherwise. Host RAM then holds the other layers and a read buffer of 4 GiB
by default (`sss_ram`). Each expert's bytes carry a checksum Poros checks on read, so training
reads the same bytes it would from RAM.

`poros train <model> --store ssd` with no folder uses `~/.poros/sss`. A config file (also with
`--store`), `Run.create` and `PorosTrainer.from_pretrained` need `store_path` with `store: ssd`.
Reopening or continuing a run reads its recorded `store_path`, or builds in `~/.poros/sss` if no
store is there. A dense model on an NVIDIA GPU streams from host RAM alone:
`--store ssd` on it is refused (`E_USAGE`, or `E_CONFIG` from a config file or Python).

### On a Mac

A Mac's CPU and GPU share one pool of memory, so a layer reaches the GPU without a copy. For each
model Poros holds the frozen layers in memory, which is a little faster, or reads them from the
drive, which roughly halves the peak memory of a 4B or 7B model; `store=ram` or `store=ssd`
sets it yourself. For the largest models on a 16 GB Mac the drive's read speed sets the
pace, so keep the 4-bit copy on your fastest drive: [Where the 4-bit copy
goes](mac.md#where-the-4-bit-copy-goes).

## The training paths

On an NVIDIA GPU, a step trains several micro-batches (8 at the settings `poros train` picks), and
the path decides how the blocks move through them:

| Path | How a step runs | VRAM | Host RAM | Time per step |
|---|---|---|---|---|
| `efficient` (default) | each block once a step; block inputs wait in host RAM | less than `fast` | the base and the block inputs | between the other two |
| `fast` | each block once a step; block inputs and dequantized weights stay on the GPU | the most | the base | the least |
| `sequential` | one micro-batch at a time through every block | can peak lower than `efficient` at long rows | the base | the most |

With the SSD expert store, `path=fast+cache` adds 4 GiB of host RAM that keeps expert data once
read, so data asked for again soon comes from RAM. In a config file or Python it is `path: fast` with `store_options: {cache: 4G}`; without the
SSD store it is refused (`E_USAGE`). On a MoE model, `fast` also keeps as many experts on the GPU
as its free memory allows, less a margin, while `efficient` keeps one layer's worth.

<!-- fill:experts -->
On an RTX 4090 with Qwen3-30B-A3B, at batch 1 with gradient accumulation, here is what the default
costs, with the experts in host RAM, beside the `moe-30b-ssd` recipe with them on the SSD:

| Experts wait in | Path | Time per step | VRAM reserved | Host RAM |
|---|---|---:|---:|---:|
| host RAM | `efficient` (default) | 4.6 s | 9.1 GiB | 19.9 GiB |
| SSD store (`moe-30b-ssd`) | `fast` | 4.3 s | 21.6 GiB | 11.2 GiB |
<!-- /fill:experts -->

A run that leaves `path` unset takes the first path the check does not turn down, and never moves the experts to the SSD by
itself. [Training paths](training.md#training-paths) and [What fits](training.md#what-fits) say
how it decides.

## Choose for your machine

Run `poros machine` on the machine itself: it checks the GPU, the driver, host RAM, the disk and
the installed software. `poros recipes` lists the certified runs and the machine each ran on. Then
adjust for what the machine is short of.

### Little VRAM

Keep the default path. On a dense model, hold fewer layers at once and shorten the rows:

```bash
poros train qwen2.5-32b-instruct --data chats.jsonl --block-size 1 --seq-len 512
```

`path=sequential` can peak lower still, and is slower. On a MoE model `--block-size` is refused.

### Little host RAM

For a MoE model, put the experts on the SSD with `--store ssd`. A dense model's base waits in host
RAM and needs room there; `path=fast` keeps the block inputs on the GPU instead, if it has the
VRAM.

### Plenty of both

```bash
poros train qwen2.5-32b-instruct --data chats.jsonl path=fast
```

For a dense model, `resident_blocks=N` keeps N blocks on the GPU for the whole run, so they are
not copied every step. Each moves its share of the base from host RAM to VRAM. Set it in a config
file; `poros train` checks it before the download. A MoE run doesn't use the setting, and a Mac refuses it.

### A laptop

For a MoE model on a laptop, start with the experts on its SSD. The `moe-30b-8gb` recipe was
measured on an 8 GB laptop GPU and is certified on the RTX 4090:

```bash
poros recipes show moe-30b-8gb
poros train qwen3-30b-a3b-instruct-2507 --data chats.jsonl --store ssd
```

### A Mac

There is no path to choose, and Poros picks memory or the drive for each model. `poros models` lists what trains on your Mac. For the largest models, keep
the 4-bit copy in a folder on your fastest drive:

```bash
poros train qwen2.5-32b-instruct --data chats.jsonl --store /Volumes/External/poros
```

## What stays the same

The path, the expert store and the block size change how much memory a run uses and how long a
step takes. They don't change the result: [Reproducibility](training.md#reproducibility).

## Next

- [What fits](training.md#what-fits): what a run checks before it starts.
- [MoE models](moe.md): how experts load, stream and train.
- [Poros on a Mac](mac.md): the settings Poros picks on a Mac.
