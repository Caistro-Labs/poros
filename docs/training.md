# Training

What a run trains, its data, the check before it starts, and resume:

```bash
poros train qwen2.5-0.5b-instruct --data chats.jsonl
```

## How it works

Poros trains LoRA adapters on a frozen 4-bit (NF4) base, streamed to the GPU a block of layers at
a time (`block_size`, 4 by default). The backward pass recomputes each block; Poros restores the
random state saved before it, so dropout draws the same masks both times.
[Streaming and memory](streaming.md) explains where the base waits and which settings to pick.

## Objectives

`poros train` picks SFT for chats and DPO for pairs; `--recipe kto` (or `recipe:` in a config:
`auto`, `sft`, `dpo`, `ipo`, `kto`, `orpo`, `simpo`) picks another.

| Objective | Data it expects | Learning rate | Epochs | Beta |
|---|---|---|---|---|
| SFT | chats or text; trains on the answers | 2e-4 | 2 | none |
| DPO, IPO | preference pairs | 5e-6 | 1 | 0.1 |
| KTO | pairs, each side scored on its own | 5e-6 | 1 | 0.1 |
| ORPO | pairs; no reference model, no SFT first | 8e-6 | 1 | 0.1 |
| SimPO | pairs; no reference model | 5e-6 | 1 | 2.0 |

Under `poros train` every objective also trains at NF4, block 4, rank 16, alpha 32, no dropout, and
8 micro-batches of one row a step. A flag or `key=value` you give wins, such as `dpo_beta=0.2` or
`learning_rate=1e-4`. On a Mac every objective trains one row a step at block 1, and SFT at lr
2e-5. In Python each objective is a loop, `poros.recipes.<name>.main`, at the settings you create
the run with ([Python API](api.md#porosrecipes)).

`--recipe my_loop.py` trains your own loop (`poros recipes copy dpo` writes one to start from). A
whole setup, model and every setting, is a [recipe](cli.md#recipes); Caistro Labs certifies Poros's own.

## Data

| Shape | A row | Trains |
|---|---|---|
| chats | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` | SFT on the answers (`train_on`: `answers`, `last` or `all`) |
| pairs | `{"prompt": "...", "chosen": "...", "rejected": "..."}` | DPO, or another pair objective |
| text | `{"text": "..."}` | every token, on the engine's own loop; `--recipe sft` trains it as SFT |

A local `.jsonl`, `.json` or `.csv` file trains as it is, its first row deciding the objective;
CSV question and answer columns, Alpaca and prompt/completion rows train as chats. A `.parquet` or
`.txt` file, a folder, a Hub dataset id, or `synthetic` (random tokens, no download) trains on the
engine's own loop.

```bash
poros data chats.jsonl             # what it holds and trains
poros data tickets.csv --convert   # as chats
poros data chats.jsonl --fix       # without the rows a run would stop at
```

A chats or pairs file of 40 rows or more sets 5% aside (2 to 100 rows), scored before the first
step, at each checkpoint and after the last; `eval_split=0` trains on every row, `--eval FILE`
scores your own file. The engine's own loop holds no rows out: it trains every row, and scores only
a file of held-out rows you give it (`--eval FILE`, `eval_dataset`), at the same points and every
`eval_steps`. For held-out rows of a text file, train it with `--recipe sft`. On a Mac, a model read from the drive holds none out unless you set
`eval_split=0.05`.

## Training paths

On an NVIDIA GPU the path decides how a step moves blocks through its micro-batches: `efficient`
(the default), `fast`, or `sequential`, and on the SSD expert store `fast+cache`. Every path trains
the same update ([Reproducibility](#reproducibility)) at a different GPU memory (VRAM), host RAM
and time; [the training paths](streaming.md#the-training-paths) compares them.

Set a path with `path=fast` on the command line, `path: fast` in a config, or `path="fast"` in
`Run.create`. `fast+cache` is a command-line name; elsewhere write `path: fast` with
`store_options: {cache: 4G}`. `block_size: 1` holds one layer, not four: a lower peak, slower
steps.

A run that leaves `path` unset takes `efficient`. If the check finds its block inputs past the
host RAM available, it takes `fast`, else `sequential`: the first the check does not turn down for
host RAM or the GPU. If the GPU is short, it tries `sequential`, then `fast`. It takes
`sequential` where the other paths cannot train the same update:

- LoRA on the embedding table or LM head.
- `lora_dropout` with `preserve_rng_state: false`.
- A model whose config turns on its own dropout or router noise.
- A MoE model's router aux loss.

With any of these, another `path` is refused before the download. A path other than the default
records why in `path_reason`, and the loop that ran in `ran_path`. A Mac has no path to choose.

## What fits

`poros recipes` lists Poros's own setups, each with the machines its certified runs used and the
GPU memory, host RAM and seconds a step each run measured. `poros recipes show <name>` says
whether this machine is one of them, or has at least the memory of one. Only a run on the
certified machine is certified; on another, the run is checked before it starts, and its time a
step differs. `poros machine` checks the GPU, the driver, host RAM, the disk and the installed
software before anything downloads. A card older than Ampere is refused (`E_GPU_UNSUPPORTED`).

### When the check refuses a run

`poros train`, `Run.create` and `PorosTrainer.from_pretrained` run the same check, in the same
words. Where the check refuses a run on the GPU, `poros train <model>` asks before it starts (a
question, exit 2; `--force` starts it), and `poros train <config>`, `Run.create` and
`PorosTrainer.from_pretrained` refuse it (exit 3). They first move the embedding table and LM head to host RAM, then try
another path, if you left those unset, and record why (`offload_reason`, `path_reason`). Where an
estimate still puts the run past the card, the run warns and trains. It is refused
(`E_OOM_PREDICTED`) when the check finds every path past the card. The refusal names the settings
the check found it does not refuse (fewer `resident_blocks`, a shorter length, `block_size: 1`,
another path, or a GPU with more free memory), or says it found none, and that `poros recipes`
lists the certified runs. It names the memory free on this machine, never a planned figure.
`--force` (or `check_fit=False`) trains anyway.

What holds less GPU memory: a lower `block_size` (4, 2, 1), shorter rows (`--seq-len`,
`max_seq_length`, `length=`), a lower rank, a freer GPU, or another path. A run out of VRAM once
started stops with `E_OOM`, naming a halved `block_size` and length.

### Host RAM

The frozen base and the default path's block inputs wait in host RAM; a run the check finds past
the host RAM available is refused before the download (`E_HOST_OOM_PREDICTED`). The refusal names
the host RAM available now, not a planned figure, and offers only what helps that run: for a MoE
model on the RAM store, the SSD store (`--store ssd`) where the check passes it; for a run that set
`path: efficient` itself, `path=fast` or `path=sequential` where the check refuses neither, by name.
A path left unset was chosen by the check from all three already, so no path is offered then, and
none where the frozen base alone is past the host RAM. `--force`, or `check_host_ram=False` in Python, tries anyway.

## Compute

On CUDA a run uses Poros's fused Triton kernels by default (`fused_compute`; `moe_expert_compute:
fused` on MoE), at PEFT's precisions but for causal attention (bf16, fp32 accumulation) and a bf16
LM head (two bf16 halves). `fused_compute: false` (dense) or
`moe_expert_compute: reference` (MoE) takes the reference compute: PEFT's modules and PyTorch
operations. A run takes it itself where the kernels can't run (no Triton, a rank too large for the
card's shared memory, non-SiLU experts), saying why in `compute_reason`. Every path runs the same
compute.

## Checkpoints and resume

`poros runs stop <run>` stops a run and saves a checkpoint.

| You want to | Command | What Poros checks |
|---|---|---|
| continue an interrupted run | `poros runs continue <run>`; `poros train run.yaml --resume-from latest`; `poros.Run.open(<folder>)`, then `run.steps(n)` | [the settings below](#what-a-resume-checks) and the data, by content |
| continue a finished run | `poros runs continue <run> --more N` | the same |
| train further on other data | `poros train --from <run> --data more.jsonl` | the base, rank and target modules ([Train further](#train-further-on-new-or-more-data)) |
| resume on another machine | `poros runs continue <run> --data FILE`; `poros.Run.open(<folder>, dataset=<its path>)` | the data file and a local base folder, by content |

### When a run saves

- Every `save_steps` steps (500), when stopped by `poros runs stop` or `poros jobs cancel`, and at
  the end of a CLI run, as `checkpoint-<N>`. The newest `save_total_limit` (3) are kept.
- A save is renamed into place once complete, so a resume never reads a torn one.
- On disk a step is an index from 0: `checkpoint-12` (and an event's `step`) holds 13 steps. On
  screen and in `poros://<run>/weights/<n>` it is a count from 1.

### When a run stops

- **Ctrl-C** stops at once (exit 130), saves nothing, and names the resume command.
- **A killed process** shows in `poros runs` as interrupted (`E_INTERRUPTED`), with its newest
  checkpoint and `poros runs continue <run>`.
- **A loss or gradient norm that is not finite** stops the run before that step (`E_STEP_CHECK`).
- **A finished run**: `--resume-from` says so (exit 0); `poros runs continue` without `--more N`
  is refused (`E_USAGE`).

### What a resume checks

- Refused (`E_CONFIG`) if it would change the base model or its revision, the quantization,
  `lora_rank`, `lora_target_modules`, `gradient_accumulation_steps`, `per_device_batch_size`,
  `seed`, `dataset` or `max_seq_length`. For that, start a new run from the adapter.
- Other settings are compared where the config sets them. One it leaves unset, and `lora_alpha`,
  takes the checkpoint's value, so a config file resumes its own run whatever filled them (an
  `init_adapter`, a recipe, a Mac's table).
- `poros runs continue` takes no loop file, Hub push, remote-code permission or starting adapter
  from the run's folder. Name them again: `--recipe FILE` (the same file), `push_to_hub=true
  hub_model_id=<repo>`, `--trust-remote-code`.
- A new run from a config file or `Run.create` into a folder with another run's checkpoints is
  refused (`E_CONFIG`). `poros train <model> --output DIR` into a folder with an earlier run asks
  first; `--overwrite` replaces it.

### A run moved to another machine

- Each checkpoint's `inputs.json` keeps the data file's sha256 and a fingerprint of a local base
  folder (its `config.json` and samples of each weight file, not a full hash). The same file or
  folder at another path resumes; another is refused. A Hub base is pinned by its revision.
- With nothing to compare (no `inputs.json`, rows passed in memory, a `poros.Run` loop that names
  no data), the resume warns once, also in the `resume` event's `data.unverified`.

### The newest checkpoint

- `--resume-from latest` and `poros.Run.open(<run folder>)` take the newest checkpoint of the
  folder's run; a folder with two runs' checkpoints has none, so name one.
- Each checkpoint names its run in `.poros_run`, and `save_total_limit` never deletes one that
  names no run or another run.

## Train further on new or more data

| | Continue the same run | Start a new run from its adapter |
|---|---|---|
| For | an interrupted run, or more steps on the same data | more rows, other data, another length or batch shape |
| It keeps | the adapter, optimizer, schedule, RNG and step | the adapter's weights |
| New | nothing (`--more N` adds steps) | the optimizer, schedule, run id, folder and data record |
| Data | the same, by content | any: chats, pairs or plain text |
| Command | [above](#checkpoints-and-resume) | `poros train --from <run> --data more.jsonl`; a config's `init_adapter: ./run` and another `output_dir`; `poros.Run.create(model, rank=16, init_adapter="./run", output_dir="./run2", dataset="more.jsonl")` |

`--from <run>` takes a run's newest checkpoint, `--from <run>/weights/<n>` the one after `n` steps
(`poros runs checkpoints <run>` lists them); `init_adapter` takes a checkpoint or run folder, in
`PorosTrainer.from_pretrained` too.

The adapter loads into the same base, `lora_rank` and `lora_target_modules` (another is
`E_CONFIG`, before the download). `--from` and a config that sets none of them take the adapter's,
with its `lora_alpha`. `Run.create` takes its `lora_alpha` and target modules but not its rank;
`PorosTrainer.from_pretrained` takes nothing. The new run needs its own folder (one inside the old
is `E_CONFIG`) and trains on the rows you give it alone.

## Reproducibility

Poros gives you the same result however it moves the weights. Change the path or the block size,
and the trained adapter comes out the same, bit for bit: every step's loss,
the gradients, the adapter weights and the optimizer state are equal under `torch.equal`. For a model the GPU can hold whole, it is also the same as training with the whole model
there. Stop a run and resume it, and it ends exactly where it would have.

This is at NF4, on a supported model family, with `deterministic: true` (the default), the same
settings, one GPU and one software stack. The whole-GPU comparison is for dense models. `deterministic: false` lets
torch pick faster kernels that don't repeat bit for bit.

`poros prove` checks this on your machine. It trains a dense model marked `validated` in
`poros models` streamed and with every layer on the GPU by ordinary autograd, and compares every
adapter tensor. Another model is refused (`E_USAGE`), and one the GPU can't hold whole is refused
before loading, offering `--block-sizes` (block 4 against one block). A Mac compares memory with
the drive, and repeats its own runs bit for bit; its results differ from a CUDA run's.

The test suite holds the same claims step by step. `tests/test_block_major_dropout.py`,
`tests/test_run_block_major_cuda.py`, `tests/moe/test_block_major_moe.py` and
`tests/moe/test_block_major_families.py` compare the paths and block sizes, with dropout on and
off. `tests/test_gpu_parity.py` compares streamed training with the whole model on the GPU. The
release's [benchmark logs](bench.md) show it on real models at this release: with dropout on, the
`efficient` and `fast` paths give the sequential loop's update, bit for bit, on Qwen2.5-7B,
Qwen2.5-32B and OLMoE-1B-7B; every supported family's check trains its paths and the whole model
on the GPU to the same adapter, and a resume of Qwen2.5 and of OLMoE matches a run straight
through; and every certified run but `mac-72b`'s was stopped and continued and ended at the same
adapter.

## Refusals

Every refusal comes before the download. `poros train` checks, in order:

1. The output folder: not home, no links, writable (`E_CONFIG`, or `E_DISK`).
2. The data: present, readable, no rows a run would stop at (`E_DATASET`, `E_DATASET_NOT_FOUND`,
   `E_HUB_AUTH` for a gated Hub dataset).
3. The recipe: one that trains this data.
4. The resume, or for a new run, a folder free of other runs' checkpoints (`E_CONFIG`).
5. The model: there, or a Hub id that exists (`E_NOT_FOUND`), not gated (`E_HUB_AUTH`), reachable
   or cached (`E_NETWORK`); then its family (`E_ARCH_UNSUPPORTED`).
6. The GPU, before PyTorch (`E_NO_CUDA`, then `E_NO_TORCH`), then `E_GPU_UNSUPPORTED` and
   `E_STACK_UNSUPPORTED`; on a Mac, PyTorch alone.
7. GPU memory and host RAM (`E_OOM_PREDICTED`, `E_HOST_OOM_PREDICTED`).
8. The disk: a download larger than the free disk (`E_DISK`; `--force` downloads anyway).

`poros train <model> --data <file>` checks the model's name (`E_NOT_FOUND`) and the data file
(`E_DATASET_NOT_FOUND`) first. `Run.create` checks the folder, GPU, family, GPU memory and host RAM at its
`length` before the download; `PorosTrainer.from_pretrained` does with `max_seq_length` set, and
otherwise checks GPU memory in `train()`, after the load. [Exit codes](cli.md#json-output-and-exit-codes) lists every code.

## Next

- [Streaming and memory](streaming.md): where the base waits, and which path to pick.
- [MoE models](moe.md): how experts load, stream and train.
- [Commands](cli.md): every command and flag.
