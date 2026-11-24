# Poros documentation

Fine-tune models bigger than your GPU.

Poros keeps the frozen 4-bit base in
your computer's RAM, or a MoE model's experts on an SSD, and copies it to the GPU a few layers at a
time. The GPU holds the adapters you're training and the layers in use. Poros runs on NVIDIA GPUs
under Linux and on Apple silicon Macs, from the command line or step by step in Python.

<!-- fill:results -->
What it trains on hardware people own:

- **A 72B on one 24 GB card.** The built-in `big-72b` recipe trains Qwen2.5-72B-Instruct on an RTX 4090 with 18.2 GiB of GPU memory (VRAM) reserved.
  It needs about 45 GiB of system RAM (peak 45.2 GiB, measured on a machine with 59.6 GiB). A step takes 19.5 s. Qwen2.5-72B's 4-bit weights alone are 41.4 GiB. Qwen2.5-72B and its Instruct model are under the Qwen license; Poros itself is Apache-2.0.
- **A 30B MoE on an 8 GB laptop.** A measured run of the built-in `moe-30b-8gb` recipe (certified on the RTX 4090) trained Qwen3-30B-A3B-Instruct-2507 on an RTX 4060 Laptop GPU with
  6.0 GiB of VRAM reserved and 6.4 GiB of host RAM, at LoRA rank 8 and rows of 256 tokens.
  A step of 8 rows took 13.6 s. Its expert store was built on the laptop.
- **A 72B on a 16 GB Mac.** The built-in `mac-72b` recipe trains Qwen2.5-72B-Instruct on an M4 Mac with 16 GB at 11.4 GiB of peak memory, with
  rows of 128 tokens. A step takes 41.1 s.

[Benchmarks](bench.md) has the settings, and Poros beside Unsloth on one RTX 4090.
<!-- /fill:results -->

## Start

- [Installation](install.md): what you need, and the install command
- [Quickstart](quickstart.md): one run, from your data to a model you can serve

## Guides

- [Training](training.md): objectives, data, what fits, checkpoints and resume
- [Streaming and memory](streaming.md): where the base waits, and which path to pick
- [MoE models](moe.md): how experts load, stream and train
- [Poros on a Mac](mac.md): the settings Poros picks on a Mac
- [Models](models.md): the supported families, and any other model
- [Architectures](architectures.md): Poros supports models by architecture: any dense or mixture-of-experts model in a supported family trains with no extra flags.
- [Other models and settings](other-models.md): try a model outside those families, and change any setting
- [Coding agents](agents.md): hand a fine-tune to Claude Code, Cursor, Codex or another agent

## Reference

- [Commands](cli.md): every command and flag, JSON output and exit codes
- [Python API](api.md): `poros.Run`, data, losses, recipes and `PorosConfig`
- [Benchmarks](bench.md): measured results, and the harness to run your own
- [Limits](limitations.md): what Poros does not do, and what to do instead

These pages are also at https://caistrolabs.com/docs/, and `poros docs` serves them offline.
