# Python API

Every public name in the `poros` package, with its arguments and defaults:

```python
import poros

run = poros.Run.create("Qwen/Qwen3-8B", rank=16, length=1024, dataset="chats.jsonl")
data = poros.data.chats("chats.jsonl", run.tokenizer, length=1024, batch_size=8)
held = poros.data.chats("held_out.jsonl", run.tokenizer, length=1024)

for step in run.steps(600):                     # resumes, and saves every save_steps
    run.forward_backward(data.batch(step))
    run.optim_step()
    if run.evals_at(step):
        run.evaluate(held)
run.save()
```

`import poros` needs no torch; `Run`, `PorosTrainer`, `PorosModel`, `Datum`, `Batch`, `Loss` and
`enforce_determinism` need torch and say so.

## poros.Run

`poros.Run` is a training run you drive one step at a time.

| Member | Does |
|---|---|
| `Run.create(model, *, rank=32, alpha=None, length=2048, lr=2e-4, output_dir=None, dataset=None, config=None, tokenizer=None, check_fit=True, check_host_ram=True, **settings)` | loads the NF4 model with LoRA; checks GPU memory and host RAM at `length` before the download |
| `Run.open(path, *, tokenizer=None, **settings)` | reopens a run folder (its newest checkpoint) or one checkpoint, with its full state |
| `Run.from_trainer(trainer, tokenizer=None)` | drives a `PorosTrainer` you already loaded |
| `run.forward_backward(batch, loss="cross_entropy", **loss_args) -> Output` | one streamed forward and backward; gradients add up until `optim_step` |
| `run.forward(batch, loss="cross_entropy", **loss_args) -> Output` | the same pass without gradients |
| `run.optim_step(lr=None) -> dict` | clips, steps AdamW and the schedule; returns `loss`, `grad_norm` and the loss's metrics |
| `run.save(name=None) -> str` | saves the full state to `checkpoint-<N>`, or `<run folder>/<name>` |
| `run.load(checkpoint)` | restores a checkpoint; the next step follows it |
| `run.steps(n, *, stop=None)` | yields step numbers from the resume point to `n`; saves every `save_steps` |
| `run.evals_at(step) -> bool` | whether to score held-out rows after `step` (every `eval_steps`, and at checkpoints) |
| `run.evaluate(data, loss="cross_entropy", *, ends=None, **loss_args) -> dict` | `eval_loss` over `data`, and `prefers` on pairs |
| `run.reference(data) -> Tensor` | each pair's (chosen, rejected) log-probabilities under the adapter as it is, cached |
| `run.tokenizer`, `run.config`, `run.dir`, `run.device`, `run.step`, `run.metrics` | the tokenizer, config, folder, device, steps taken and summary so far |

Details that change what you write:

- `alpha=None` is twice `rank`; `length` sets `max_seq_length`. These defaults are not
  `poros train`'s.
- `settings` are any `PorosConfig` fields, such as `path="fast"`; `config` replaces them all.
- `dataset=` names the file your rows come from, so `Run.open` can check a resume reads the same
  data; on `Run.open` it says where the file moved to.
- `Run.open` takes the base, the adapter shape and the data from the folder and writes back into
  it unless you pass `output_dir=`. It never grants `trust_remote_code`, a Hub push or an
  `init_adapter`.
- `N` in `checkpoint-<N>` is the last step, counted from 0; before any step, a save needs a name.
- Each step of `run.steps` ends with `optim_step()`. When `stop()` returns True, the run saves and
  raises `PorosCancelledError`.
- `run.evaluate` leaves the RNG as it was; `run.reference` is what DPO, IPO and KTO compare
  against.

## poros.data

`poros.data` renders rows into batches from a `.jsonl`, `.json` or `.csv` file, a list of dicts,
or a Hub dataset id; a bad row fails with its row number.

| Builder | Rows | Scored |
|---|---|---|
| `chats(src, tokenizer, *, train_on="answers", length=2048, batch_size=8, seed=0)` | `messages` or `conversations` | `answers`: every assistant turn; `last`: the final one; `all`: every token |
| `pairs(src, tokenizer, *, length=2048, batch_size=4, seed=0)` | `prompt`, `chosen`, `rejected` | the answers, chosen and rejected in one batch |
| `text(src, tokenizer, *, length=2048, batch_size=8, seed=0, field="text")` | `text` | every token |
| `detect(src)` | any | `"chats"`, `"pairs"` or `"text"`, from the top row |

What a builder returns:

| Member | Does |
|---|---|
| `batch(i)` | step `i`'s batch; each epoch reshuffles by seed |
| `ordered()` | every row once, in order |
| `rows` | how many rows it holds |
| `holdout(share, count=0)` | returns (train, held); `count` holds out that many instead, at most half |

## Batches

Build batches yourself when the builders don't suit your data.

| Name | Does |
|---|---|
| `poros.Datum(tokens, target_tokens, weights=None, logprobs=None, advantages=None)` | one sequence: position `i` reads `tokens[i]`, scored on `target_tokens[i]` |
| `poros.Datum.from_sequence(seq, weights=None)` | a `Datum` from a whole sequence |
| `poros.Batch(datums)` | `Datum`s padded to the longest |
| `poros.Loss(value, metrics={})` | what a loss returns: its value, with gradients, and a dict of metrics |

## poros.losses

A loss is `(logprobs, batch, **args) -> poros.Loss`, where `logprobs` is `[B, T]` log p(target)
with gradients, 0 on padding.

| Name | Data | Arguments |
|---|---|---|
| `cross_entropy` | chats, text | none |
| `dpo`, `ipo` | pairs | `ref=run.reference(pairs)`, `beta=0.1` |
| `kto` | pairs | `ref=run.reference(pairs)`, `beta=0.1`, `desirable=1.0`, `undesirable=1.0` |
| `orpo` | pairs | `beta=0.1` |
| `simpo` | pairs | `beta=2.0`, `gamma=1.0` |
| `importance_sampling`, `ppo` | rows with `logprobs` and `advantages` | `ppo`: `clip=0.2` |

Pass a name or any callable: `run.forward_backward(batch, loss="dpo", ref=ref, beta=0.1)`.

| Function | Does |
|---|---|
| `@poros.losses.register("name")` | makes your own loss callable by name |
| `poros.losses.get(loss)` | the function for a name, or the callable |
| `poros.losses.names()` | every registered name |

## poros.recipes

Each built-in objective is a short loop on the verbs above.

| Loop | Arguments |
|---|---|
| `poros.recipes.sft.main` | `(run, data, *, steps=None, epochs=1, eval_data=None, ends=None, stop=None)` |
| `poros.recipes.dpo.main`, `ipo.main`, `kto.main`, `orpo.main` | `(run, pairs, *, steps=None, epochs=1, beta=0.1, eval_data=None, stop=None)` |
| `poros.recipes.simpo.main` | the same as `dpo`, with `beta=2.0` |
| `poros.recipes.copy(name, dest=None)` | writes a loop's file into your folder to change, and returns its path |

Each loop scores `eval_data` before, at `evals_at` steps and after training, then saves and returns
`run.metrics`. [Objectives](training.md#objectives) lists `poros train`'s settings for each.

## PorosTrainer and PorosModel

`PorosTrainer` is the engine `Run` drives and the CLI loads models through, with a plain loop of
its own that scores every token with cross-entropy.

```python
from poros import PorosTrainer

with PorosTrainer.from_pretrained("Qwen/Qwen2.5-7B", max_seq_length=512) as trainer:
    trainer.train("synthetic", steps=10)
    trainer.save("./out")
```

| Member | Does |
|---|---|
| `PorosTrainer.from_pretrained(model_name_or_path, *, quantization="nf4", lora_rank=16, lora_alpha=32, lora_dropout=None, lora_target_modules=None, block_size=None, config=None, check_host_ram=True, check_fit=True, store="ram", store_path=None, **kwargs)` | loads the model; `kwargs` are `PorosConfig` fields |
| `trainer.train(dataset, *, steps=None, seq_len=None, eval_dataset=None, resume_from=None, should_stop=None) -> dict` | trains; returns `final_loss`, `total_steps`, `peak_vram_allocated_gb` and more. `eval_dataset`, held-out rows, is scored before the first step (not again on a resume), every `eval_steps`, at each checkpoint (`eval_checkpoints`, into its `eval.json`) and after the last; the result then has `eval_loss_start` and `eval_loss`. Scoring leaves the random generators as they were. It holds no rows out of `dataset` |
| `trainer.save(output_dir, *, push_to_hub=False, hub_model_id=None)` | saves a standard PEFT adapter, with `poros_config.json` and the RNG state |
| `trainer.evaluate(eval_dataset) -> dict` | returns `eval_loss` and `eval_perplexity` |
| `trainer.close()`, or a `with` block | frees the model and its host memory |
| `PorosModel.from_pretrained(model_name_or_path, *, quantization="nf4", lora_rank=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, revision=None, trust_remote_code=False, ...)` | the loader below `PorosTrainer`: `.model` (NF4 with LoRA) and `.trainable_parameters` |

With `max_seq_length` set, `from_pretrained` runs the check before the download; otherwise
`train()` checks it after the load. `dataset` is a file, a Hub id, `"synthetic"`, a `Dataset` or a
`DataLoader`; `resume_from` is a checkpoint folder.

## PorosConfig

`PorosConfig(**yaml)` holds every setting, a Pydantic model that needs no torch; `poros schema
config` prints every field. The ones you are most likely to set:

| Field | Does | Default |
|---|---|---|
| `model_name_or_path` | the base: a Hub id or a local folder | `"Qwen/Qwen2.5-7B"` |
| `dataset` | a local file, a Hub dataset id, or `"synthetic"` | None |
| `recipe`, `loop` | `auto`, `sft`, `dpo`, `ipo`, `kto`, `orpo` or `simpo`; `loop` names your own file | `"auto"`, None |
| `output_dir` | where checkpoints, the adapter and `events.jsonl` go | `"./poros_output"` |
| `lora_rank`, `lora_alpha`, `lora_dropout` | the adapter's shape and dropout | 16, 32, 0.05 |
| `lora_target_modules` | the modules that get adapters | None: picked per family |
| `learning_rate`, `lr_scheduler`, `warmup_steps` | AdamW's rate, schedule and warmup (at most about a tenth of the run) | 2e-5, `"cosine"`, 100 |
| `max_steps`, `num_epochs` | steps, or with `max_steps` -1, passes over the data | -1, 1 |
| `per_device_batch_size`, `gradient_accumulation_steps` | rows in a micro-batch, and micro-batches in a step | 1, 8 |
| `max_seq_length` | tokens per row | 2048 |
| `block_size` | layers on the GPU at once | None (4) |
| `path` | `efficient`, `fast` or `sequential` | None (`efficient`) |
| `resident_blocks` | blocks kept on the GPU for the whole run | 0 |
| `offload_embeddings_and_head` | the embedding table and LM head in host RAM | None: where the run needs it |
| `fused_compute` | `False` takes the reference compute | None: on where it can run |
| `store`, `store_path`, `store_options`, `sss_ram` | where a MoE model's experts wait ([SSD store](streaming.md#the-ssd-expert-store)) | `"ram"`, None, `{}`, 4 GiB |
| `save_steps`, `save_total_limit` | how often to save, and how many checkpoints to keep | 500, 3 |
| `eval_split`, `eval_rows` | rows held out of the data, on the recipe paths | 0.05, 0 |
| `eval_dataset`, `eval_steps`, `eval_checkpoints` | a file of held-out rows, and when it is scored, on every loop | None, 500, True |
| `seed`, `deterministic` | the seed, and strict determinism | 42, True |
| `blockwise_arch_check` | `"warn"` or `"off"` runs a family that is not supported | `"error"` |

A run records what it resolved (`path_reason`, `ran_path`, `offload_reason`, `ran_compute`,
`compute_reason`) in `poros_config.json` and `config.resolved.json`.

## Model families

| Function | Returns |
|---|---|
| `poros.validated_arch_types(entry="dense")` | the supported dense families; `validated_arch_types("moe")` the supported MoE families |
| `poros.register_architecture(model_type, *, parity="", notes="")` | runs a family as `user_asserted`, with no guarantee |
| `poros.user_asserted_arch_types()` | what you registered, here or in `POROS_USER_ASSERTED_ARCHS` |
| `poros.runnable_arch_types()` | the supported dense families plus the `user_asserted` ones (no MoE) |
| `poros.architectures()` | every registry entry: `{model_type: ArchSpec}` |

Registering never changes the supported sets. [Models](models.md) explains what each status
means.

## Errors

Every error is a `PorosError`, importable from `poros`, with a stable `code` and `exc.info`
(`message`, `remediation`, `doc_url` as a `poros docs` command, and `fix`, the config fields to
apply): `except poros.PorosVramError as exc: print(exc.info.fix)`.
[Exit codes](cli.md#json-output-and-exit-codes) maps codes to exit numbers.

| Error | Code | Raised |
|---|---|---|
| `PorosError` | `E_USAGE` | the base of every error; alone, a misused API |
| `PorosArchNotValidatedError` | `E_ARCH_UNSUPPORTED` | a family neither supported nor registered |
| `PorosVramError` | `E_OOM_PREDICTED` | the check finds every path past this GPU; names what it does not refuse |
| `PorosHostRamError` | `E_HOST_OOM_PREDICTED` | the same, for host RAM |
| `PorosOOMError` | `E_OOM` | out of GPU memory (VRAM) while training; suggests a smaller `block_size` or length |
| `PorosGpuUnsupportedError` | `E_GPU_UNSUPPORTED` | a GPU below compute capability 8.0 |
| `PorosTorchBuildError` | `E_STACK_UNSUPPORTED` | a PyTorch build with no kernels for this GPU |
| `PorosStackUnsupportedError` | `E_STACK_UNSUPPORTED` | a failed Mac self-check, other Mac package versions, or a transformers too old for MoE |
| `PorosGuaranteeUnavailableError` | `E_GUARANTEE_UNAVAILABLE` | a model or feature a Mac cannot run |
| `PorosConfigError` | `E_CONFIG` | a refused setting or resume, or another run's folder |
| `PorosDataError` | `E_DATASET` | data a run cannot train on |
| `PorosDatasetNotFoundError` | `E_DATASET_NOT_FOUND` | data a run cannot find |
| `PorosDiskError` | `E_DISK` | no room for a download, a save, an export or an SSD store |
| `PorosExportError` | `E_EXPORT_UNSUPPORTED` | a merged export Poros will not write |
| `PorosStepCheckError` | `E_STEP_CHECK` | a loss or gradient norm that is not finite; the run stops before that step |
| `PorosCancelledError` | `E_JOB_CANCELLED` | a run stopped on request, after its checkpoint |

`PorosGpuUnsupportedError`, `PorosTorchBuildError` and `PorosDataError` subclass
`PorosConfigError`; `PorosDatasetNotFoundError` subclasses `PorosDataError`.

## Other names

| Name | Is |
|---|---|
| `poros.__version__` | the installed version, as a string |
| `poros.enforce_determinism(*, fill_uninitialized=True)` | strict determinism in torch; a run calls it when `deterministic` is true |
| `poros.BenchmarkSpec` | the Pydantic model of the spec `poros bench run` takes ([Benchmarks](bench.md)) |
| `poros.RunArtifact` | the Pydantic model of a benchmark cell's result (`poros schema artifact`) |

`enforce_determinism` turns on deterministic torch and cuDNN algorithms, sets
`CUBLAS_WORKSPACE_CONFIG` and keeps PyTorch's attention on its math kernel; Poros's fused attention
has deterministic backward passes. [Reproducibility](training.md#reproducibility) says what holds.

## Next

- [Training](training.md#objectives): the objectives and their settings.
- [Commands](cli.md): the same runs from the command line.
- [Other models and settings](other-models.md): which setting to change, and when.
