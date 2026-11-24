# Architectures

> Poros supports models by architecture: any dense or mixture-of-experts model in a supported family trains with no extra flags.

Poros validates models by family, reading `model_type` from a model's `config.json`. If your model belongs to a supported family, you need no extra flags to train it, and model size does not matter for support. Both dense and mixture-of-experts (MoE) models train on Linux with an NVIDIA GPU (compute capability 8.0 or newer) or on an Apple silicon Mac on macOS 14 or later.

## Supported architectures

| Architecture | Kind | Models it covers (examples) |
| --- | --- | --- |
| Qwen2.5 | Dense | Qwen2.5-0.5B, 7B, 32B, 72B and their Instruct versions; DeepSeek-R1-Distill-Qwen models (same architecture) |
| Qwen3 | Dense | Qwen3-0.6B, Qwen3-32B |
| Qwen3.5, Qwen3.6, and Qwen3.8 | Dense | Qwen3.5-4B, Qwen3.8-27B |
| Gemma 3 | Dense | gemma-3-1b, gemma-3-27b |
| Gemma 4 | Dense | gemma-4-31B (Gemma 4 E2B and E4B are not supported) |
| Qwen3 MoE | MoE | Qwen3-30B-A3B, Qwen3-Coder-30B-A3B |
| Qwen3.5 / 3.6 MoE | MoE | Qwen3.6-35B-A3B |
| Gemma 4 MoE | MoE | gemma-4-26B-A4B |
| OLMoE | MoE | OLMoE-1B-7B |
| Qwen2 MoE | MoE | Qwen1.5-MoE-A2.7B |
| DeepSeek-V2 | MoE | DeepSeek-V2-Lite |

## Run a model

```bash
poros models                                              # list catalog models and support status
poros check Qwen/Qwen2.5-7B-Instruct                      # check whether the family is supported
poros train Qwen/Qwen2.5-7B-Instruct --data chats.jsonl   # train any model in a supported family
```

`poros check` accepts a catalog name, a Hub ID, a local folder, or a `config.json` path. It reads the model's config and exits with code 0 when the family is supported. When you run `poros train`, every refusal comes before the download, in seconds: the family, the GPU, GPU memory and host RAM, the disk, and the data. Run `poros recipes` to list built-in, certified setups.

## Mixture-of-experts models

A supported MoE model runs through the same `poros train` command with no extra flags. Poros makes the experts 4-bit and keeps the other layers in bf16; LoRA trains the routed experts. The experts wait in host RAM by default. On machines with little host RAM, pass `--store ssd` to keep them in an SSD store instead:

```bash
poros train Qwen/Qwen3-30B-A3B-Instruct-2507 --data chats.jsonl --store ssd   # keep experts in the SSD store
```

## What supported means

On the release build, each of the 15 architecture rows listed in [Benchmarks](bench.md#architectures-checked) passed the same checks on an RTX 4090. Every training path gives the same adapter as training fully on the GPU, and a repeated run is bit-identical. That table lists each row and the model it was checked on.

## A model from another architecture

For a dense model from an unknown family, `poros arch <model>` says whether any flag can train it, and `--unvalidated` trains one run at a time with no guarantee about its results:

```bash
poros train <model> --data <file> --unvalidated   # train one unvalidated dense run
```

A MoE model from an unknown family is refused on a CUDA GPU; see [Models](models.md) for the full steps.

## Next

- **[Models](models.md):** Checking any model before download, catalog support status, and steps for unknown families.
- **[MoE models](moe.md):** Training mixture-of-experts models and keeping experts in host RAM or the SSD store.
- **[Benchmarks](bench.md):** The 15 checked architecture rows and the model each was checked on.
