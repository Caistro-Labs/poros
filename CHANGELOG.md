# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-24

The first open-source release: fine-tune models bigger than your GPU.

The PyPI releases of July 2026, numbered below 1.0, were early previews. 1.0.0 does not continue
their commands or their checkpoints.

Inkling 2 Mini, a looped MoE model, is not supported yet.

### Added

- **Fine-tuning models larger than GPU memory.** The frozen base is kept in host RAM as NF4 and
  streamed through the GPU one block of layers at a time, with only the LoRA adapters resident.
- **`poros.Run`**: a training run you drive one step at a time (`forward_backward`,
  `optim_step`, `save`, `load`, `steps`, `evaluate`), with `poros.data` for chats, pairs and text,
  and `poros.losses` for the built-in losses or your own.
- **Objectives**: SFT, DPO, IPO, KTO, ORPO and SimPO, as short loops in `poros.recipes`, picked
  from your data or by name.
- **Four training paths**: `efficient` (the default), `fast`, `fast+cache` (a MoE model on the
  SSD store) and `sequential`, each training the same update. Block-major accumulation streams
  each block once a step for all its micro-batches.
- **Fused Triton kernels** for the dense NF4 LoRA compute and a MoE model's experts, the default
  on CUDA.
- **Checks before download.** Every run checks GPU memory, host RAM and disk before anything
  downloads, adapts what you left unset, and refuses only what cannot fit. `poros machine` checks
  the GPU, driver, RAM, disk and installed software; `poros recipes` lists the certified runs and
  the machine each ran on.
- **Checkpoints and resume**, from the CLI and from `poros.Run`.
- **A run resumes on another machine**, on CUDA and on a Mac. A checkpoint records its dataset
  file and a base model in a local folder by content, so the same file or folder at another path
  resumes and another is refused. A base named by its Hub id is pinned by its revision.
- **One run, one newest checkpoint.** `--resume-from latest` and `poros.Run.open` take the newest
  checkpoint of a folder's run, a run resumed from a checkpoint that names no run included, and
  refuse a folder that holds two runs.
- **MoE models**: Qwen3-MoE, Qwen3.5 / 3.6 MoE, Gemma 4 MoE, OLMoE, Qwen2-MoE and DeepSeek-V2,
  with the experts in host RAM or on an SSD expert store.
- **Held-out scores on every loop.** A run on the engine's own loop scores a file of held-out
  rows (`--eval FILE`, `eval_dataset`) before the first step, every `eval_steps`, at each
  checkpoint and after the last, as the recipe loops score theirs. The run's metrics carry the
  first score as `eval_loss_start`, and a checkpoint keeps its scores (`eval.json`,
  `eval_start.json`), so a resumed run reports the same start.
- **An optimizer for 16-bit expert adapters.** A MoE model's expert adapters are bf16, where plain
  AdamW drops an update smaller than a weight's last bit. `ExpertAdamW` computes each update in
  fp32 from the weight and a remainder it keeps beside it, and rounds the remainder and both
  moments up or down by bits keyed to the seed, the step and the tensor, so two runs of one
  config round the same way. Its fused kernels (Triton on CUDA, Metal on a Mac) are compared
  with the reference when the optimizer is built and used only where every bit agrees. Each
  checkpoint records the rule in `optimizer.json`.
- **Any model**: `poros arch`, `poros arch try`, `poros train --unvalidated` and `poros prove
  --unvalidated` take a family Poros has not validated from a first look to a validation pack.
- **Apple silicon Macs**, dense and MoE.
- **The CLI**: `machine`, `models`, `recipes`, `data`, `train`, `runs`, `eval`, `export` and `prove`,
  an interactive session, recipes with `poros recipes sync`, `--json` output for every command but `poros bench`,
  and `poros agents` for coding agents.
- **`poros export`**: a merged bf16 model for vLLM or TGI, or the adapter alone.
- **`poros bench`**: Poros against resident QLoRA, PEFT, Unsloth, Accelerate offload and
  DeepSpeed ZeRO-3 under one configuration.
