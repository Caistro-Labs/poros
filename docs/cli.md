# Poros commands

Every `poros` command, flag, exit code and environment variable; `poros <command> --help` lists a
command's flags with their defaults.

## The verbs

| Command | Does |
|---|---|
| `machine` | checks the GPU, driver, host RAM, disk and installed software; says whether this machine is ready to train, and the next command |
| `models` | the models Poros names, with the recipes certified for each on this machine; `models get <model>` downloads one (`--yes`; on a Mac `--store FOLDER`) |
| `data <file>` | what a file holds and trains; `--convert` and `--fix` write a copy |
| `train <model> --data <file>` | fine-tunes a model |
| `runs` | your runs; `runs <run>`, `stop`, `continue`, `report`, `checkpoints` act on one |
| `eval <run>` | scores a run beside its base, on held-out rows |
| `export <run>` | a merged model (`--format merged`) or the adapter alone (`--format adapter`) |
| `prove [<model>]` | trains a validated dense model streamed and with every layer on the GPU; compares the adapters |

`recipes` lists whole setups and the machines their certified runs used. The others are under [Other commands](#other-commands). `poros` alone opens the interactive session; with `--json` or no
terminal, nothing prompts.

## Names and flags

| A name for | Is written as |
|---|---|
| a model | a catalog name in any case, a Hugging Face id (`owner/name`) or a folder |
| a run | its id, a unique prefix, its name, or `last` |
| a checkpoint | `poros://<run>/weights/<step>`, `.../final` or `.../best` (lowest held-out loss) |
| data | a file, a folder, a Hub dataset id, or `synthetic` |
| a config field | `key=value` after the verb, such as `path=fast` |

Part of a name (`qwen 7b`) makes Poros ask. An unknown flag is `E_USAGE`, naming the closest one.
These flags answer a confirmation in advance:

| Flag | Agrees to |
|---|---|
| `--yes`, `-y` | every confirmation |
| `--download` | downloading the model or a Hub dataset |
| `--force` | a run the GPU memory or host RAM check refuses, or a download larger than the free disk |
| `--overwrite` | training into a folder with a trained adapter, or exporting over files |
| `--allow-unvalidated` | a model outside the catalog, of a supported family |
| `--unvalidated` | a family Poros does not support, this run alone, no guarantee |

Before any command: `--version` (`-V`), `--json`, `--plain`, `--no-anim`, `--theme auto|dark|light`,
`--install-completion`, `--show-completion`, `--help`.

### `train`

| Flag | Sets | Default |
|---|---|---|
| `--data FILE` | what to train on: chats or text (SFT), pairs (DPO) | asked |
| `--steps N` | optimizer steps | the recipe's epochs |
| `--epochs N` | passes over the data, instead of `--steps` | SFT 2, pairs 1 |
| `--seq-len N` | tokens per row | 95% of your rows, within the GPU |
| `--rank N` | LoRA rank | 16 |
| `--block-size N` | layers on the GPU at once (dense) | 4 |
| `--recipe NAME` | `sft`, `dpo`, `ipo`, `kto`, `orpo`, `simpo`, `auto`, or `my_loop.py` | `auto` |
| `--from RUN` | start from a run's adapter, or a checkpoint of it | a fresh adapter |
| `--eval FILE` | rows to score | chats and pairs: 5% held out; text: none (`--recipe sft` holds 5% out) |
| `--store WHERE` | on CUDA, a MoE model's experts: `ram`, `ssd` (`~/.poros/sss`), `ssd:/path` or a folder | `ram` |
| `--store FOLDER` | on a Mac, the store folder for the 4-bit copy | a drive with room |
| `--sss-ram SIZE` | the SSD store's read buffer | `4G` |
| `--name NAME` | the run's name | model, data, date |
| `--output DIR` | the adapter's folder (not with `--detach`) | the run's |
| `--detach`, `-d` | train in the background | off |
| `key=value` | any config field: `path=fast`, `dpo_beta=0.2` | the recipe's |

`--store ssd` on a dense model and `--gpu` on `train` are refused. [Streaming and
memory](streaming.md) explains stores and paths.

### `data`

| Flag | Does | Default |
|---|---|---|
| `--convert` | writes it as chats (or pairs) beside it | off |
| `--fix` | a clean copy, `<name>.clean.jsonl` | off |
| `-o`, `--output FILE` | where `--fix` writes | beside the file |

`data` never reads the Hub; [Data](training.md#data) lists the shapes.

### `runs`

| Command or flag | Does | Default |
|---|---|---|
| `runs` | your runs, newest first | |
| `runs <run>` | shows one run | |
| `runs stop <run>` | saves a checkpoint at the next step and exits | |
| `runs continue <run>` | goes on from its newest checkpoint | |
| `runs report <run>` | writes `report.md` in its folder | |
| `runs checkpoints <run>` | what the run saved, `best` marked | |
| `--follow`, `-f` | watches a run live | off |
| `--limit N` | lists the newest N runs | all |
| `--live` | the state once | off |
| `--frame`, `--wait N` | with `--live`: the loss curve; return N steps on | off |
| `--more N` | on `continue`, N more steps | 0 |
| `--data FILE` | on `continue`, where the data file moved to (same contents) | the recorded path |
| `--recipe FILE` | on `continue`, the run's loop file again (same contents) | |
| `--trust-remote-code` | on `continue`, a job whose base runs its own code | off |
| `push_to_hub=true hub_model_id=REPO` | on `continue`, push again | no push |
| `--detach` | on `continue`, in the background | off |
| `--copy STEP`, `--delete STEP` | on `checkpoints`: copy one here, or delete one | |

The list holds `poros train <model>` runs and background jobs; config-file and Python runs keep
theirs in `output_dir`. What a continue takes from the folder is in [What a resume
checks](training.md#what-a-resume-checks). To start from `best`:
`train --from poros://<run>/weights/best`.

### `eval`

| Flag | Does | Default |
|---|---|---|
| `--data FILE` | scores this file, of the run's shape | the held-out rows |
| `--rows N` | the most rows it scores | 100 |
| `--checkpoint STEP` | a step, `best` or `final` | `final` |
| `--show N` | prints N answers | off |

### `export`

| Flag | Does | Default |
|---|---|---|
| `--format merged\|adapter` | a bf16 model, or the adapter alone | asked |
| `-o`, `--output DIR` | where it goes | the run's `export/<format>/` |
| `--overwrite` | replaces what's there | off |

`poros export poros://<run>/weights/best --format adapter` exports a checkpoint, to
`export/<format>-<step>/`. A merged export needs torch; an adapter does not. An adapter of a base folder
names a placeholder for the base and says what to set. A CUDA MoE adapter is Poros's own format
(it loads with `poros.Run.open`); `--format merged` gives a model any server loads.

### `prove`

| Flag | Does | Default |
|---|---|---|
| `--steps N` | steps in each run | 20 |
| `--seq-len N` | tokens per row | 256 |
| `--block-sizes` | block 4 against one block, both streamed | off |
| `--force` | runs both trainings of a large model without asking | asked |
| `--unvalidated` | a new family's two checks ([Any other model](models.md#any-other-model)) | off |

`poros prove` alone picks a small model (`--download` agrees to it). A model the GPU can't hold
whole is refused before loading, naming `--block-sizes`, and the saved result names which comparison
ran. On a Mac it compares memory with the drive. `poros prove <receipt>` reruns a saved check.
Where a built-in recipe is certified on this machine, the result names the one with the largest
model (`largest_certified`); `poros recipes` lists them all.

## Recipes

A recipe is a whole setup: model, data, objective and settings. `poros recipes` lists Poros's own,
yours (`~/.poros/recipes/`) and a project's (`./recipes/`). Each of Poros's own shows the GPU memory
its certified run used where this machine is the one it was certified on; on any other, it says
where it was certified, with no figure.

| Command | Does |
|---|---|
| `poros recipes` | every recipe |
| `poros recipes show <name>` | one recipe, with its train command and where it is certified |
| `poros recipes save [<run>] [--name TEXT]` | keeps a finished run as your recipe |
| `poros recipes share [<run>]` | writes its recipe file here, with no data, paths or names |
| `poros recipes copy <loop> [--output FILE]` | writes a loop here; `--recipe my_loop.py` trains it |
| `poros recipes sync` | fetches the shared recipes from `Caistro-Labs/poros-recipes` on GitHub |

Caistro Labs certifies Poros's own on the machines it runs them on, at this release: `first-run`,
`support`, `voice`, `prefer`, `big-32b`, `big-72b`, `big-72b-fast`, `moe-30b`, `moe-30b-ssd`,
`moe-30b-8gb`, and on a Mac `mac-first-run`, `mac-support`, `mac-voice`, `mac-prefer`, `mac-7b`,
`mac-32b`, `mac-moe`, `mac-72b`. Naming the other platform's recipe points to this machine's
counterpart; `first-run`'s card adds a command that needs no data. `first-run` is also certified on
an 8 GB RTX 4060 Laptop GPU; `moe-30b-8gb` is certified on the RTX 4090 and measured on the laptop.
<!-- fill:laptop -->
On an RTX 4060 Laptop GPU, a measured run of `moe-30b-8gb` trained Qwen3-30B-A3B-Instruct-2507 on the `fast` path at 13.6 s a step, with 6.0 GiB of VRAM reserved and 6.4 GiB of host RAM; its expert store was built on the laptop.
<!-- /fill:laptop -->
Qwen2.5-72B and its Instruct model, which `big-72b`, `big-72b-fast` and `mac-72b` train, are under
the Qwen license; Poros itself is Apache-2.0. `recipes show <name>` says:

| It says | When |
|---|---|
| Certified on this machine | the same card with at least its host RAM, or the same Mac model |
| Certified on another machine | this one has at least its memory; the screen says "Not run on this machine" and gives no figure (`here.status`: `same_memory_class`) |
| Not certified on this machine | less memory than any certified machine, or a card Poros does not train on, with the reason |
| Certification pending | its runs at this release are not in yet |

Only Poros's own recipes can be marked certified; a shared recipe's name, title, about and author may not claim it. `sync` alone uses the network: `owner/name` is
always GitHub, a folder starts `/`, `./`, `../` or `~/`, and a repo it can't fetch is `E_NETWORK`.

## After training

```bash
poros eval                              # the newest finished run
poros eval last --data more_chats.jsonl # another file of the run's shape
poros eval poros://7f3a/weights/best    # its best checkpoint
```

`eval` runs the rows through the streamed forward with the adapter off (the base) and on; no
row's words come back but `--show`'s. A pairs run with none held out is scored on the pairs it trained on.

| Result field | Holds |
|---|---|
| `base`, `trained` | each pass's `loss`; chats: `ends` (answers ending on end-of-turn); pairs: `prefers`, `margin` |
| `change` | trained loss against base: `-0.3` is 30% lower |
| `improved` | rows whose loss fell, or pairs whose margin grew |
| `rows`, `of`, `rows_from` | rows scored, of how many, from where |
| `verdict` | `Learned well` (loss 5% lower or more), `Barely moved`, or `Held-out loss rose` |
| `warning` | the trained model does not end its answers |

Held-out rows are set at train time ([Data](training.md#data)): `--eval FILE` (`eval_dataset`) on
any run, and on a recipe's run `eval_split` and `eval_rows`; `eval_checkpoints` (so `best` can be
picked) and `eval_steps` say when they are scored. A run on the engine's own loop holds no rows out:
`eval` scores its `--eval` file, read as that loop read it, or a file you name. A run folder holds `events.jsonl`,
`receipt.json`, the adapter, its checkpoints and `config.resolved.json`.

## Config files

| Command | Does |
|---|---|
| `poros init [PATH] [-m MODEL] [-d DATA] [-o OUTPUT] [--force]` | a starter config (`train.yaml`); `--force` replaces one |
| `poros train CONFIG.yaml` | trains the config, with the flags below |
| `poros eval CONFIG.yaml [-a DIR] [--trust-remote-code]` | scores the adapter in `output_dir` (or `--adapter DIR`) on its dataset (CUDA) |
| `poros export DIR [-o OUT] [--trust-remote-code] [--overwrite] [-y]` | merges a config run's adapter (`--format merged`, or `safetensors`); `--yes` agrees to the base's download |

| `poros train CONFIG.yaml` flag | Does |
|---|---|
| `-s`, `--steps N`; `--seq-len N`; `--block-size N`; `-o`, `--output-dir DIR` | override `max_steps`, `max_seq_length`, `block_size`, `output_dir` |
| `--store WHERE`, `--sss-ram SIZE` | override the expert store (`ram`, `ssd:/path`, a plugin) and its buffer |
| `-r`, `--resume-from DIR\|latest` | continues from a checkpoint, or the newest in `output_dir` |
| `-d`, `--detach` | runs as a background job and prints its id |
| `--force`, `--overwrite`, `-y`, `--unvalidated` | as in [Names and flags](#names-and-flags) |
| `--trust-remote-code` | lets the base's code run; a config file never grants it |

`poros train CONFIG.yaml` checks a run before anything downloads, and says the path, the tables'
place or a refusal in words. `poros recipes` lists what each certified run
measured. ([What fits](training.md#what-fits).)

## Other commands

| Command | Does |
|---|---|
| `poros check MODEL [--json]` | whether the family is supported, from `config.json` ([Models](models.md)) |
| `poros arch MODEL`, `poros arch try MODEL` | what a model would take; a tiny copy checked on the CPU ([Any other model](models.md#any-other-model)) |
| `poros schema [NAME]`, `poros schema --list` | a JSON Schema (`config`, `events`, `artifact`, a result) |
| `poros jobs list`, `poros jobs status ID` | background jobs; one job (exits 0 when it can read it) |
| `poros jobs logs ID [-f]` | a job's log, by its id alone (`--json`: its `events.jsonl` rows) |
| `poros jobs cancel ID`, `poros jobs resume ID` | stops a job with a checkpoint; reruns a stopped or failed one |
| `poros report PATH [--md] [--no-html] [-o FILE]`, `poros leaderboard PATH [-t N]`, `poros bench run\|matrix\|compare\|report` | benchmark reports, rankings and the harness ([Benchmarks](bench.md)) |
| `poros docs [TOPIC [WORDS]]` | these pages offline |
| `poros agents [COMMAND]` | the coding-agent guide, or a command in their form |
| `poros help` | `--help` |

Jobs keep files in `$POROS_HOME/jobs/<id>/artifacts/`; one job runs on a GPU at a time.

## JSON output and exit codes

With `--json` (or `POROS_OUTPUT=json`) stdout carries one `poros.<name>.v1` object (`poros schema
<name>`), the rest stderr; memory in GiB, downloads in GB. `poros bench` has no JSON mode. A
failure is one `poros.error.v1` object:

```json
{"schema":"poros.error.v1","error":{"code":"E_OOM","exit_code":3,
 "message":"GPU out of memory during blockwise training.",
 "hint":"Try reducing block_size or seq_len. Current block_size=4; try block_size=2.",
 "remediation":{"block_size":2},"docs":"poros docs limitations"}}
```

| Exit | Meaning | Codes |
|---|---|---|
| 0 | success | |
| 1 | a Poros bug, logged in `$POROS_HOME/crash.log` | `E_INTERNAL` |
| 2 | usage, config, or a question | `E_USAGE`, `E_CONFIG`, `E_CONFIRM_REQUIRED`, `E_NEEDS_INPUT`, `E_NOT_FOUND` |
| 3 | won't fit | `E_OOM_PREDICTED`, `E_HOST_OOM_PREDICTED`, `E_OOM` |
| 4 | not supported, or a step not finite | `E_ARCH_UNSUPPORTED`, `E_GPU_UNSUPPORTED`, `E_EXPORT_UNSUPPORTED`, `E_GUARANTEE_UNAVAILABLE`, `E_STEP_CHECK` |
| 5 | environment | `E_NO_TORCH`, `E_NO_CUDA`, `E_DEPS`, `E_DISK`, `E_NETWORK`, `E_HUB_AUTH`, `E_STACK_UNSUPPORTED`, `E_MACHINE_NOT_READY` |
| 6 | data | `E_DATASET`, `E_DATASET_NOT_FOUND`, `E_NO_RESULTS` |
| 7 | a job failed or cancelled | `E_JOB_FAILED`, `E_JOB_CANCELLED` |
| 130 | interrupted | `E_INTERRUPTED` |

The table is `poros.core.errors.EXIT_CODES`. `poros check` and `poros prove` exits are in
[Models](models.md). Others:

| Command | Exits |
|---|---|
| `poros machine` on Linux | 5 with no NVIDIA GPU, `pydantic` or `pyyaml` missing, a bad `POROS_NUM_THREADS`, or 5 GB or less free disk; missing torch only warns |
| `poros machine` on a Mac | 5 when a pinned package is missing or off its pin, or macOS is unsupported |
| `poros jobs logs --follow` | 0 when the job succeeded, 7 when it did not |

**Questions.** With no terminal, a command that needs an answer prints `poros.needs_input.v1`
(exit 2); each choice carries the `command` that answers it.

**Runs.** A run's files are in `$POROS_HOME/runs/<id>/`; progress goes to `events.jsonl`
(`poros schema events`), not stdout, and `poros.core.events.current(lines)` folds it.

## Environment variables

None of these changes a value a run computes.

| Variable | Does |
|---|---|
| `POROS_OUTPUT=json` | JSON output for every command |
| `POROS_PROGRESS=0` | no progress line on stdout |
| `POROS_HOME` | runs, jobs, recipes and the SSD store (default `~/.poros`) |
| `POROS_TRACEBACK=1` | an internal error's traceback on stderr |
| `POROS_RECIPES_REPO` | the repo (`owner/name`) or folder `recipes sync` reads |
| `POROS_USER_ASSERTED_ARCHS` | comma-separated `model_type`s to run as `user_asserted` |
| `POROS_NUM_THREADS` | torch's CPU thread count |
| `POROS_PIN_BUDGET` | host memory this process may pin (`4G`, `0`) |
| `POROS_WSL_PIN_PROBE=0` | under WSL2, no pin-budget probe and no pinning |
| `POROS_BOUNDARY_PIN`, `POROS_BOUNDARY_HOST_MAX` | `efficient`'s block inputs: pinned allowance, host RAM cap |
| `POROS_QUANTIZE_ON=cpu` | quantizes MoE experts on the CPU |
| `POROS_MOE_STICKY=0`, `POROS_MOE_PREFETCH=N`, `POROS_MOE_CACHE_HEADROOM`, `POROS_MOE_BATCH=0` | the MoE expert cache, read-ahead, margin and batched compute |
| `POROS_SSS_NATIVE=0`, `POROS_SSS_NATIVE_DIR` | the SSD store's compiled reader: off, or where |
| `POROS_KEEP_TORCH_NATIVE=1` | keeps torch's own native op routers |
| `POROS_STARTUP_SELFCHECK=always` | before training, one forward pass with every block loaded against one streamed (`startup_parity_selfcheck`) |
| `POROS_THEME`, `POROS_KEYS`, `POROS_FORCE_COLOR=1`, `POROS_NO_ANIM=1`, `POROS_NO_MENU=1`, `POROS_INLINE=1`, `POROS_TABS=0`, `POROS_IMAGES` | the terminal's look |
| `POROS_AGENT`, `POROS_JOB` | set by Poros itself; leave unset |
| `POROS_DEV_SIMULATE`, `POROS_DEV_SIMULATE_GPU`, `POROS_DEV_SIMULATE_LOAD`, `POROS_DEV_SIMULATE_STEP`, `POROS_DEV_SIMULATE_BOARD`, `POROS_DEV_MAC_MEMORY_GB`, `POROS_COMMIT` | development hooks; they train nothing real |

## Example commands

The last column is how Poros reads each command.

<!-- sentences:start -->
| Command | Verb | Means |
|---|---|---|
| `poros train Qwen2.5-7B-Instruct --data notes.jsonl` | train | subject: Qwen2.5-7B-Instruct; --data notes.jsonl; model: Qwen/Qwen2.5-7B-Instruct |
| `poros train --from last --data prefs.jsonl` | train | --from last; --data prefs.jsonl |
| `poros train Qwen2.5-7B-Instruct --data prefs.jsonl --recipe dpo dpo_beta=0.2` | train | subject: Qwen2.5-7B-Instruct; dpo_beta = 0.2; --data prefs.jsonl; --recipe dpo; model: Qwen/Qwen2.5-7B-Instruct |
| `poros train Qwen2.5-7B-Instruct --data ./data/ --steps 300 --seq-len 1024 --rank 32 --name notes-v1` | train | subject: Qwen2.5-7B-Instruct; --data ./data/; --steps 300; --seq-len 1024; --rank 32; --name notes-v1 |
| `poros train Qwen2.5-7B --data synthetic learning_rate=1e-4 --yes` | train | subject: Qwen2.5-7B; learning_rate = 1e-4; --data synthetic; --yes |
| `poros train DeepSeek-V2-Lite --data OpenAssistant/oasst1` | train | subject: DeepSeek-V2-Lite; --data OpenAssistant/oasst1; model: deepseek-ai/DeepSeek-V2-Lite |
| `poros train olmoe-1b-7b-0924 --data notes.jsonl --detach` | train | subject: olmoe-1b-7b-0924; --data notes.jsonl; --detach; model: allenai/OLMoE-1B-7B-0924 |
| `poros machine` | machine | — |
| `poros models` | models | — |
| `poros models get Qwen2.5-7B-Instruct` | models get | subject: Qwen2.5-7B-Instruct; model: Qwen/Qwen2.5-7B-Instruct |
| `poros runs` | runs | — |
| `poros runs 7f3a` | runs | subject: 7f3a |
| `poros runs stop 7f3a` | runs stop | subject: 7f3a |
| `poros runs continue last` | runs continue | subject: last |
| `poros runs continue last --more 200` | runs continue | subject: last; --more 200 |
| `poros runs report last` | runs report | subject: last |
| `poros data tickets.csv --convert` | data | subject: tickets.csv; --convert |
| `poros eval last --rows 20 --data more.jsonl` | eval | subject: last; --rows 20; --data more.jsonl |
| `poros export last --format merged --output ./merged` | export | subject: last; --format merged; --output ./merged |
| `poros prove Qwen2.5-0.5B --steps 20` | prove | subject: Qwen2.5-0.5B; --steps 20; model: Qwen/Qwen2.5-0.5B |
<!-- sentences:end -->

## Next

- [Quickstart](quickstart.md): the commands in the order a run uses them.
- [Training](training.md): what each training setting does.
- [Python API](api.md): the same runs from Python.
