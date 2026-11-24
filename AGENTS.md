# AGENTS.md: driving Poros as an agent

Fine-tune models bigger than your GPU. Poros fine-tunes LoRA adapters while the frozen NF4 base
waits in host RAM and streams through the GPU one block of
layers at a time. Runs go from the CLI or step by step through `poros.Run`; SFT, DPO, IPO, KTO,
ORPO and SimPO ship as recipes; MoE models train with their experts in host RAM or on an SSD
store; Apple silicon Macs train too. On a supported family, how Poros moves the weights and a
resume don't change a run's results, and a dense model the GPU can hold whole trains to the same adapter
streamed or resident: `poros agents docs training reproducibility`.

Docs for agents: this file and `poros agents`. For people: `docs/`, also at
https://caistrolabs.com/docs/ and offline with `poros docs`. Where they disagree, the code wins.

## Orient

```bash
pip install poros-train         # everything training needs
poros agents                    # the agent guide: every command, its JSON, the rules
poros agents machine            # is this machine ready: GPU, driver, RAM, disk, software
poros agents recipes            # the certified runs, and the machine each ran on
poros check <model>             # supported family? exit 0 yes, 4 no, 2 could not resolve
poros schema --list             # every JSON Schema
```

No torch or GPU needed: `--help`, `machine`, `check`, `init`, `schema`, `report`,
`leaderboard`. A light install without torch (`pip install --no-deps poros-train`, then typer,
rich, pydantic, pyyaml, huggingface-hub and jinja2) runs those and refuses training with
`E_NO_TORCH`. Training needs the full install and a CUDA GPU of compute capability 8.0 or newer on Linux,
or an Apple silicon Mac on macOS 14 or later. Windows is not supported. WSL2 is untested in this release.

## The golden path

From a shell, in the agent form (JSON without a terminal, run ids at once, nothing prompts):

```bash
poros agents machine                                                # is this machine ready to train
poros agents data notes.jsonl                                       # shape, rows, problems
poros agents train Qwen/Qwen2.5-0.5B-Instruct --data notes.jsonl    # returns a run id at once
poros agents run watch <id> --wait 30                               # again until it ends
poros agents eval <id>                                              # held-out loss beside the base (text rows: give --eval FILE, or train with --recipe sft)
poros agents export <id> --output ./adapter                         # or --merged for a whole model
```

From a config file:

```bash
poros init run.yaml -m Qwen/Qwen2.5-7B-Instruct -d ./notes.jsonl -o ./run
poros train run.yaml --steps 200 --json        # progress in ./run/events.jsonl
poros train run.yaml --resume-from latest      # the same results at the same step count
poros export ./run -o ./merged                 # bf16, for transformers, vLLM or TGI
```

`--resume-from latest` needs a checkpoint: one every `save_steps` updates (500 unless set; `poros
init` leaves `save_steps: 50` commented out) and one when the run finishes. Ctrl-C writes none.

From Python, `poros.Run` (verbs shaped like Tinker's):

```python
import poros

run = poros.Run.create("Qwen/Qwen2.5-7B-Instruct", rank=16, length=512, dataset="notes.jsonl")
data = poros.data.chats("notes.jsonl", run.tokenizer, length=512)
for step in run.steps(200):
    run.forward_backward(data.batch(step))
    run.optim_step()
run.save()                                     # checkpoint-199; poros.Run.open(run.dir) resumes
```

| Objective | Data | CLI | Python |
|---|---|---|---|
| SFT | chats, text | chats pick it | `poros.recipes.sft.main(run, data)` |
| DPO | pairs | pairs pick it | `poros.recipes.dpo.main(run, pairs)` |
| IPO, KTO, ORPO, SimPO | pairs | `--recipe ipo` | `poros.recipes.ipo.main(run, pairs)` |

Chats are `{"messages": [...]}`, pairs `{"prompt", "chosen", "rejected"}`; a config's field is
`recipe:`, and `--from <run>` starts a new run from a run's adapter (DPO after SFT, more or other data; in a
config `init_adapter:`). Recipes train at NF4, block 4, rank 16,
alpha 32, dropout 0, 8 micro-batches of one row; SFT lr 2e-4, the pair objectives 5e-6 (ORPO
8e-6). A Mac trains one row a step at block 1, SFT at lr 2e-5. Whole setups are `poros recipes`:
Poros's own are certified by Caistro Labs, and `recipes show <name> --json` has `certified`: the
machines each ran on with what that run measured (`gpu_memory_used_gib`, nvidia-smi's, is the GPU memory a run needs;
`torch_reserved_peak_gib` is torch's) and where its record is, and `here.status`: `certified` (this
is that machine), `same_memory_class` (another with at least its memory: not run here, checked at the start, other speed;
never with a figure), or `not_certified` and, for a card
Poros does not train on, `here.reason`.

## What to know before a run

- **Every refusal comes before the download, in seconds**: the output folder, the data, the
  recipe, the resume, the family, the GPU, GPU memory and host RAM, the disk. Read the `poros.error.v1` object.
- **The check adapts, then refuses.** What you left unset adapts first: the embedding table and LM head
  move to host RAM (`offload_reason`), then another training path (`path_reason`). `block_size` is
  never changed for you. It refuses only where the check finds every path past the card
  (`E_OOM_PREDICTED`); `--force` / `check_fit=False` trains anyway.
- **Training paths**: `efficient` (default), `fast`, `fast+cache` (MoE on the SSD store) and
  `sequential` train the same update. Set with `path=fast`, `path: fast` or `path="fast"`. `fast+cache` is a
  command-line shorthand (`path=fast+cache`); in a config or in Python it is `path: fast` with
  `store_options: {cache: 4G}`.
- **Checkpoints**: `checkpoint-<N>` after update `N` (from 0), every `save_steps` (500) and on a
  stop (`run stop <id>`, `jobs cancel`; Ctrl-C writes none); atomic. A resume onto another model,
  rank, seed, dataset, length or batch shape is refused (`E_CONFIG`), as is a new run from a
  config or `Run.create` into another run's checkpoints (the verb form asks `--overwrite` and
  replaces it). A dataset file and a local base folder are compared by content (each
  checkpoint's `inputs.json`), so a run moved to other paths resumes (`train <config>
  --resume-from`, `run continue <id> --data <file>`, `Run.open(dir, dataset=...)`).
- **Any other model**: `arch <model>`, `arch try <model>`, `train <model> --data <file>
  --unvalidated`, `prove <model> --unvalidated`. Every result of such a run says `support:
  user_asserted`. Before adding `--unvalidated`, tell the person nothing guarantees its results.
  The ladder: `poros agents docs models any other model`.
- **Numbers**: quote only figures a command returned or the docs state, with the card and
  settings. Never invent or interpolate one.

## Machine-readable surfaces

- `--json` (or `POROS_OUTPUT=json`) on every command but `poros bench`: exactly one JSON object on
  stdout, `"schema": "poros.<name>.v1"`; logs on stderr. `poros schema <name>` gives its schema,
  which every result validates against.
- Failures: `poros.error.v1`: `code`, `exit_code`, `message` and, each where there is one, `hint`
  (the next step), `remediation` (fields), `docs`.
- Questions: `poros.needs_input.v1`, exit 2. Run one choice's `command`, whole.
- `events.jsonl` in each run folder: one event per line (`event_type`, `timestamp`, `step`,
  `data`; `poros schema events`). A `step` event carries `loss`, `grad_norm`, `lr`, `s_per_step`,
  `tokens` and memory; an `eval` event `eval_loss`, `rows` and `held` (which held-out rows, a
  fingerprint: a start is compared only with scores of the same rows). Append-only: fold it with
  `poros.core.events.current(lines)`.
- `config.resolved.json` and `poros_config.json`: every setting a run trained at, with
  `path_reason`, `ran_path`, `offload_reason`, `ran_compute`, `compute_reason`.
- `receipt.json`: a finished run's summary. `RunArtifact` (`poros schema artifact`): a bench cell.

## Exit codes

0 success; 1 a Poros bug; 2 usage, config or an answer needed; 3 won't fit; 4 unsupported, or a
non-finite loss; 5 environment (torch, CUDA, deps, disk, network, Hub access); 6 data; 7 a job
failed or was cancelled; 130 interrupted. The table is `poros.core.errors.EXIT_CODES`.

## Errors and their fix

| Error | Means | Do |
|---|---|---|
| `E_NO_TORCH` (5) | a light install, without torch | `pip install poros-train` |
| `E_ARCH_UNSUPPORTED` (4), `PorosArchNotValidatedError` | a family Poros does not support | a supported model, or `--unvalidated` for one run; the reason says when no flag trains it |
| `E_OOM_PREDICTED` (3), `PorosVramError` | the check finds every path past this GPU | read its `hint`: the settings the check does not refuse (fewer `resident_blocks`, a shorter `max_seq_length`, `block_size: 1`), or that it found none |
| `E_HOST_OOM_PREDICTED` (3), `PorosHostRamError` | not enough host RAM | more RAM free, or a MoE model's SSD store (`--store ssd`) |
| `E_OOM` (3), `PorosOOMError` | out of GPU memory while training | halve `block_size` or the length; the message names values |
| `E_GPU_UNSUPPORTED` (4) | compute capability below 8.0 | an Ampere or newer GPU |
| `E_STACK_UNSUPPORTED` (5) | PyTorch has no kernels for this GPU, or a Mac failed its self-check | the PyTorch build the message names |
| `E_DISK` (5), `PorosDiskError` | no room for a download, save, export or SSD store | free space, or point `HF_HOME`, `--output` or `store_path` elsewhere |
| `E_CONFIG` (2), `PorosConfigError` | a refused setting, resume or output folder | the message names the field and the fix |
| `E_DATASET` (6) | a row a run would stop at | `poros agents data <file> --fix`, then train on the copy |
| `E_EXPORT_UNSUPPORTED` (4) | a merged export Poros will not write | export the adapter |
| `E_STEP_CHECK` (4) | a non-finite loss or gradient norm; stopped before that update | lower `learning_rate`, check the data, resume from the last checkpoint |

## Module map

Each name lives in one module and is imported from there; no library layer imports the CLI.

```
poros/preflight/  every check before a download: card, family, compute, GPU memory, host RAM, disk, data, resume
poros/core/       the streaming engine: autograd, residency, determinism, registry, card, checkpoints
poros/train/      poros.Run (run.py), PorosTrainer, PorosModel, block-major, datasets, export
poros/recipes/    the objectives (sft, dpo, ipo, kto, orpo, simpo), shipped setups, recipes sync
poros/plan/       the check before a run: model configs and the cells it decides from
poros/moe/        MoE streaming: experts in host RAM or on an SSD, expert LoRA, load, export
poros/patch/      internal: streams a MoE model's dense layers through the GPU
poros/adapters/   which parameters are an adapter's
poros/sss/        the expert store's RAM and SSD (.sss) backends
poros/validate/   the checks a new family passes (load, training, pack)
poros/metal/      the Mac (Apple silicon) path
poros/jobs/       background jobs (poros train --detach, poros jobs)
poros/cli/        the CLI: verbs, the session and its screens, the JSON contract, agents
poros/bench/      the benchmark harness and RunArtifact
poros/report/     HTML and Markdown reports of benchmark artifacts
poros/*.py        catalog, data, losses, ends, runs, stores, links, palette
```
