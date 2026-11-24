# Models

Poros supports a model by its family, the `model_type` in its `config.json`. A supported family
trains with no extra flag; any other can still train, one run at a time, with no guarantee:

```bash
poros check Qwen/Qwen2.5-7B     # is its family supported? exits 0 when it is
poros models                    # the models Poros names, with recipes certified here
```

## Supported families

| Family | `model_type` | Kind |
|---|---|---|
| Qwen2.5, DeepSeek-R1-Distill-Qwen | `qwen2` | dense |
| Qwen3 | `qwen3` | dense |
| Qwen3.5, Qwen3.6, Qwen3.8 | `qwen3_5`, `qwen3_5_text` | dense |
| Gemma 3 | `gemma3`, `gemma3_text` | dense |
| Gemma 4 (not E2B and E4B) | `gemma4`, `gemma4_text` | dense |
| Qwen3-MoE (Qwen3-30B-A3B, Qwen3-Coder-30B-A3B) | `qwen3_moe` | MoE |
| Qwen3.5 / 3.6 MoE (Qwen3.6-35B-A3B) | `qwen3_5_moe`, `qwen3_5_moe_text` | MoE |
| Gemma 4 MoE (gemma-4-26B-A4B) | `gemma4_moe` | MoE |
| OLMoE | `olmoe` | MoE |
| Qwen2-MoE (Qwen1.5-MoE-A2.7B) | `qwen2_moe` | MoE |
| DeepSeek-V2 (DeepSeek-V2-Lite) | `deepseek_v2` | MoE |

[MoE models](moe.md) says what differs for MoE. A Gemma 4 MoE's `config.json` says `gemma4`;
`enable_moe_block` tells it apart. Gemma 4 E2B and E4B are refused before the download: their
layers take per-layer inputs the block forward does not pass.

## Ask about a model

A family is supported, marked `validated`, once a maintainer has checked it on a CUDA GPU
([Reproducibility](training.md#reproducibility)).
`poros check` reads `config.json` without torch, from a catalog name, Hub id, folder or
`config.json`; a catalog model is answered offline too, noting a gated repository.

| `poros check` says | Means | Exit |
|---|---|---|
| `validated` | the family is supported | 0 |
| `user_asserted` | you registered it; it runs with no guarantee | 4 |
| `target` | known, not checked; `--unvalidated` trains it | 4 |
| `not-in-registry` | unknown; `--unvalidated` trains it where a flag can | 4 |
| `unsupported` | no flag trains this variant; the reason says why | 4 |
| `could-not-resolve` | no `config.json` could be read: offline, gated, or not found | 2 |
| `not-on-this-device` | this Mac has no settings for the model | 4 |

With `--json`, `entry` is the route (`dense` or `moe`) and `next` the step after. `poros models`
marks each model `validated` (checked whole or on its leading layers), `measured` (a run of it was
measured, but it was not checked), `same shapes as <model>` (an Instruct of a checked or measured
base; its own weights unchecked) or `none`.

## Any other model

Five steps take a model from unknown to supported. Each command's `next` names the following one, and no `next` carries a consent flag: say what a
step will run before you add `--unvalidated`.

| Step | Command | Says |
|---|---|---|
| 1 | `poros arch <model>` | from the config alone: its family, its layers, and whether any flag trains it |
| 2 | `poros arch try <model>` | on the CPU: a tiny random copy, streamed, against the model's own forward and backward |
| 3 | `poros train <model> --data <file> --unvalidated` | trains it, this run alone, no guarantee |
| 4 | `poros prove <model> --unvalidated` | on a CUDA GPU: the two checks a supported family passes; a pass writes a pack |
| 5 | an architecture request issue with the pack | a maintainer reruns the checks and adds the family |

`train <model> --data <file> --unvalidated` trains the family at any step. Its results say
`support: user_asserted`, `runs continue` keeps the flag, and no stored Hub token is sent.

No flag trains a repository that needs its own code, an FP8, I8 or U8 release, a gated repository, a `model_type` this install has no class for, or, on a
GPU, an unsupported MoE family: each is refused with `E_ARCH_UNSUPPORTED`, saying why.

**`arch try`** builds a tiny random model from the real config and checks, on the CPU, the route
this machine trains on: off a Mac within 1e-4 of the same PEFT model in fp32, on a Mac for exact
equality. `first_difference` names where they part. It checks the plumbing, not NF4.

**`prove --unvalidated`** runs two checks on this machine's CUDA GPU, where the GPU must hold
the whole model. One compares the streamed NF4 load with the standard load (a sha256 over every tensor), the
other streamed (block 4) with resident training: losses, adapter gradients, adapter weights and Adam
state under `torch.equal` at every step. `--steps` is 20 (the full procedure, 200). It prints
`poros.prove_gates.v1` (`passed`, each check's numbers, `differs` as `{surface, step, tensor}`,
earliest first) and exits 0 on a pass, 1 on a difference.

**The pack.** A pass writes `~/.poros/validation/<model_type>-<stamp>/`: `load_gate.jsonl`,
`parity.jsonl`, `receipt.json`, a `README.md` with how to reproduce it, and `registry_line.txt`,
the line for `_ARCHITECTURES` in `poros/core/arch_registry.py`. It holds no path, host or user of
the machine, and `submit` says where each file goes. Poros never marks a family validated itself:
a maintainer runs the checks on their own card. To ask for a family, attach the pack to an
architecture request ([Contributing](https://github.com/Caistro-Labs/poros/blob/main/CONTRIBUTING.md)).

## From Python

`blockwise_arch_check: "warn"` trains an unsupported family with a warning; `"off"` skips the
check. `poros.register_architecture("phi3")` or `POROS_USER_ASSERTED_ARCHS="phi3,starcoder2"` makes
a family `user_asserted`; [Model families](api.md#model-families) lists the registry functions. On
a Mac, see [A model Poros has not validated](mac.md#a-model-poros-has-not-validated).

## Next

- [MoE models](moe.md): how experts load, stream and train.
- [Python API](api.md): the registry functions and every other public name.
- [Other models and settings](other-models.md): try a model, then change any setting of its run.
