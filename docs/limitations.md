# Limits and troubleshooting

Each entry is a limit or an error, and what to do instead, most common first.

## The GPU runs out of memory during a run

The run stops with `E_OOM`: a run the check passed can still need more GPU memory (VRAM)
than is free. Halve `block_size` or the row length; the error names both, and in a
terminal Poros offers the retry. [What fits](training.md#what-fits) lists the other ways.

`big-72b-fast` runs close to the RTX 4090's limit, so PyTorch prints allocator warnings
("expandable_segments: memory mapping failed") during it; the allocator carries on, and the run
finishes and resumes as it should.

## A run is slower than training resident

Streaming copies the base every step. If the GPU holds a dense model with room to spare, keep
some blocks on the GPU with `resident_blocks=N`: [Plenty of both](streaming.md#plenty-of-both).

## One GPU per run

There is no multi-GPU or multi-node training.

## LoRA adapters on a frozen base

No full fine-tuning and no pretraining. The embedding table and LM head don't train themselves
(`freeze_embeddings: false` and `freeze_lm_head: false` are refused); add `embed_tokens` or
`lm_head` to `lora_target_modules` to adapt them.

## A merged export is refused

`E_EXPORT_UNSUPPORTED`: the model ties its embedding table to its LM head and the adapter changed
one, or the base is already quantized, or a layer type the merge does not take. With a tied table,
export with `--format adapter`.

## The base must be in bf16, fp16 or fp32

A release already at 8 bits (FP8, I8, U8) is refused. Use a bf16 repository of the same model.

## NVIDIA cards older than Ampere

A GPU needs compute capability 8.0 or newer, on Linux; older cards are refused before the download
(`E_GPU_UNSUPPORTED`). Without a CUDA GPU the NF4 base cannot load.
[Requirements](install.md#requirements) lists the cards.

## Your stack is refused

`E_STACK_UNSUPPORTED`: a PyTorch build with no kernels for your card, a transformers too old for
the MoE route, or a failed Mac self-check. The message names the fix.

## Macs

A Mac run repeats on that Mac and does not equal a CUDA run
([Reproducibility](training.md#reproducibility)). A config file's `eval`, `poros bench` and a model
the Mac does not train are refused with `E_GUARANTEE_UNAVAILABLE`; `poros models` lists what your
Mac trains. [Poros on a Mac](mac.md) has the rest.

## Windows

Windows is not supported. WSL2 is untested in this release. Train on Linux or on a Mac.

## Host RAM

The frozen base, the default path's block inputs and a MoE model's experts wait in host RAM,
checked before the download (`E_HOST_OOM_PREDICTED`). [Host RAM](training.md#host-ram) lists the
fixes, such as `--store ssd` for a MoE model. `poros eval` goes ahead past the check where
the run used less host RAM when it trained than is available now, with room to spare, and says so.

## Two runs in one output folder

A config file's or Python run doesn't lock its output folder, so two runs or two resumes in one
can overwrite each other's checkpoints; give each its own `output_dir`. `poros train <model>` locks
its folder.

## What a resume can check

A Hub dataset on the engine's own loop is recorded by name, not revision, so a newer copy resumes
without a refusal. A run from a local model folder resumes from a folder, not a Hub id.
[Checkpoints and resume](training.md#checkpoints-and-resume) has the rest.

## Models outside the supported families

[Reproducibility](training.md#reproducibility) is checked on the
[supported families](models.md#supported-families). Other models train without that check, and some
not at all: a repository that needs its own code, a gated repository, and on a GPU an unsupported
MoE family. [Any other model](models.md#any-other-model) says how to try one.

## Share what a run made, not its folder

A run's folder records the machine's name, in `status.json`. To share a run, use
`poros export <run> --format adapter`, a Hub push (`push_to_hub` with `hub_model_id`; it needs a base
with a Hub id) or `poros recipes share`. None carries that file or a local path (an adapter of a base folder names a
placeholder for it). Don't send the folder. Continue a run folder someone else gave you at your own risk.

## Next

- [What fits](training.md#what-fits): what the check refuses, and what to change.
- [Exit codes](cli.md#json-output-and-exit-codes): every error code and its exit number.
- [Other models and settings](other-models.md): which models train, and what no flag trains.
