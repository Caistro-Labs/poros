# Other models and settings

Poros can tell you about any model on the Hugging Face Hub, or in a folder, before you download it:
whether a run can train it, and what that takes. A model of a [supported
family](models.md#supported-families) trains with no extra flag. Another dense model trains with
`--unvalidated`, one run at a time, with nothing to guarantee its results. Some no flag trains, and
Poros says why. This page shows you how to try a model, and then how to change any setting of a run.

```bash
poros check HuggingFaceTB/SmolLM2-135M    # supported family? exits 0 when it is
poros arch HuggingFaceTB/SmolLM2-135M     # what a run would take, and whether any flag trains it
```

## What Poros trains

| The model | What happens |
|---|---|
| A catalog model of a supported family | trains with no flag |
| Another model of a supported family, such as a fine-tune someone posted | its question, `not_in_catalog`, offers `--allow-unvalidated` |
| A dense model of a family transformers knows | its question, `unvalidated`, offers `--unvalidated` for this run, or `poros arch try` first |
| Anything in [What no flag trains](#what-no-flag-trains) | refused before the download, with the reason |

A supported family gives you the same result however Poros moves the weights
([Reproducibility](training.md#reproducibility)). A run with `--unvalidated` has no such check
behind it. Its results, its report and any recipe made from it say `support: user_asserted`.

## Try another model

1. Ask about it. `poros check <model>` reads its `config.json`, without torch. It says `validated`
   for a supported family, `not-in-registry` or `target` for one a flag can train, and
   `unsupported` with the reason for one none can.
2. Read what it would take. `poros arch <model>` places its layers and sizes its adapter, one layer
   and the embedding table from the config alone. With no flag that trains it, it says why.
3. Check the math on your CPU. `poros arch try <model>` builds a tiny random copy, runs the forward
   and backward a run would, and compares them with the model's own. `first_difference` names where
   they part.
4. Train it. `poros train <model> --data chats.jsonl --unvalidated` trains it for this run.
   `poros runs continue` keeps the flag.
5. Prove it on your GPU, if the GPU can hold the whole model. `poros prove <model> --unvalidated` runs the
   two checks a supported family passes, and a pass writes a pack you can attach to an
   architecture request issue.

`arch try` needs torch, and so does `arch` to place the layers. `arch try` checks the
plumbing on tiny shapes, not the 4-bit base or your GPU, which is why step 5 exists. [Any other model](models.md#any-other-model)
has the detail of each step and the pack.

A model outside the catalog of a supported family takes `--allow-unvalidated` instead:

```bash
poros train Qwen/Qwen2-0.5B --data chats.jsonl --allow-unvalidated
```

`poros.register_architecture("llama")` in Python, or `POROS_USER_ASSERTED_ARCHS=llama` in the
environment, marks a family `user_asserted` without the flag. [From Python](models.md#from-python) lists
the other switches.

## What no flag trains

Each of these is refused before any weights download. `poros arch` names the reason, and
`poros train` refuses with `E_ARCH_UNSUPPORTED`.

- A repository that needs its own code to load. Poros never runs a repository's code for a family it
  doesn't support.
- A release already at 8 bits, such as FP8, I8 or U8. Poros makes its 4-bit base from bf16, fp16 or
  fp32 weights, so look for a bf16 repository of the same model.
- A gated repository. Poros asks the Hub with no token for a family it doesn't support.
- A `model_type` the installed transformers has no class for, or a model whose layers Poros can't
  find.
- On a GPU, a mixture-of-experts family Poros doesn't support. It would take the dense route, which
  doesn't make its fused experts 4-bit.
- A variant Poros marks unsupported, such as gpt-oss or Gemma 4 E2B and E4B. `poros check` gives
  each one's reason.

The card and the system set limits too. A GPU older than Ampere is refused (`E_GPU_UNSUPPORTED`),
and Windows is not supported. [Limits](limitations.md) lists them all. A Mac trains another family
through settings derived from its config: [A model Poros has not
validated](mac.md#a-model-poros-has-not-validated).

## Change a run

Each setting below works on any model Poros trains, unless it says otherwise. Pass it as a flag, as `key=value` after the
command, or as a field in a config file. A field with its own flag, such as `max_steps`, takes the
flag on the command line. `poros schema config` lists every field with its default.

### Shape your data and objective

`poros data chats.jsonl` says what a file holds and which objective it trains. Chats train SFT and
preference pairs train DPO. `--recipe kto` picks another objective, and `train_on=last` scores the
last answer of each chat alone. A higher `dpo_beta`, such as `dpo_beta=0.2`, holds a pairs run
closer to the model it started from. [Data](training.md#data) and
[Objectives](training.md#objectives) list the shapes and their settings.

### Size the adapter

`--rank 32` gives the adapter more room to learn, and costs VRAM. `lora_alpha` scales its output by
alpha over rank. `poros train` keeps alpha at 32, so set `lora_alpha=64` beside a rank of 32 to keep
the same scale. `lora_dropout=0.05` adds dropout, which can help on a small file the run might
memorize. Set `lora_target_modules` in a config file to choose the layers that get adapters. Left unset, each family's attention and MLP projections get them.

### Set row length and batch

`--seq-len 1024` sets the tokens per row, and longer rows are cut. Left unset, `poros train` takes a
length that holds 95% of your rows whole, or a shorter one where the check refuses that one. A step trains `per_device_batch_size` rows a
micro-batch, 1 by default, times `gradient_accumulation_steps` micro-batches, 8 by default. Both
change what a step trains on, and a resume refuses a change to either.

### Set steps and learning rate

`--steps 300` sets the number of steps, or `--epochs 3` the passes over your data. Use
`learning_rate=1e-4` to change the rate. Each objective's default rate is in the Objectives table.
`lr_scheduler=constant` drops the cosine schedule, and `warmup_steps=` sets the warmup.

### Pick a path and store

`path=fast` trades VRAM for shorter steps. `--block-size 2` lowers the peak on a dense model.
`--store ssd` keeps a MoE model's experts on an SSD. On a supported family, none of these changes
the result. [Streaming and memory](streaming.md) says which to pick.

### Keep checkpoints

A run saves a checkpoint every `save_steps` steps, 500 by default, and keeps the newest
`save_total_limit`, 3 by default. Set `save_steps=50` on a short run you may stop and continue.
`poros runs continue <run>` resumes, and `poros train --from <run> --data more.jsonl` starts a new
run from its adapter. [Checkpoints and resume](training.md#checkpoints-and-resume) says what a
resume checks.

### Use a config file

Flags suit a run you type once. A config file suits a run you change, repeat or share.

```bash
poros init run.yaml -m Qwen/Qwen2.5-7B-Instruct -d ./chats.jsonl -o ./run
poros train run.yaml
```

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
dataset: ./chats.jsonl
output_dir: ./run
train_on: last
max_seq_length: 1024
lora_rank: 32
lora_alpha: 64
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
learning_rate: 1.0e-4
max_steps: 300
save_steps: 50
```

A field the file leaves out takes the default of the objective its data trains. The run's
`config.resolved.json` records every value it used. `poros train run.yaml --unvalidated` trains a
config of another family.

### Start from a recipe

A recipe is a whole setup: model, data, objective and settings.

```bash
poros recipes                       # every recipe, and where it is certified
poros recipes show support          # one recipe's settings and its train command
poros recipes copy sft              # the SFT loop as a Python file to change
```

Copy the command `recipes show` prints, swap in your data and change a setting. `poros recipes save`
keeps a finished run as your own recipe, and `poros recipes share` writes its file here. A recipe
file placed in `./recipes/` shows up in `poros recipes`. A loop file from `recipes copy` trains with
`--recipe sft_loop.py`. [Recipes](cli.md#recipes) has every command.

### Go further in Python

`poros.Run` runs one step at a time, so your own data and loss take a few lines. A loss is any
function that takes the per-token log-probabilities and the batch and returns a `poros.Loss`, and
`@poros.losses.register("name")` makes it callable by name in `run.forward_backward`.
`poros.Datum` builds rows of your own, with a weight on each token.
[poros.losses](api.md#poroslosses) and [Batches](api.md#batches) have the details, and
`blockwise_arch_check: "warn"` trains another family from Python.

## What can go wrong

| Code | Means | Do |
|---|---|---|
| `E_ARCH_UNSUPPORTED` | no flag trains this model, or the family needs `--unvalidated` | read `poros arch <model>`; add the flag if a flag trains it |
| `E_CONFIG` | an unknown field, or a resume that changes the model, rank, length or batch | fix the field it names; start a new run with `--from` |
| `E_USAGE` | a flag the command does not take, or a field that has its own flag | use the flag it names |
| `E_OOM_PREDICTED` | the check finds every path past this GPU at these settings | a shorter length, a smaller block or a lower rank |

[Refusals](training.md#refusals) lists every check in the order a run makes it.

## Next

- [Models](models.md): the supported families, and each step of trying another.
- [Training](training.md): every setting in depth.
- [Coding agents](agents.md): hand a run to a coding agent.
