# Poros on a Mac

Apple silicon Macs train dense and MoE models with the Linux commands; Poros picks the
settings.

## Install and train

You need Apple silicon (M1 or later) and macOS 14 or later (macOS 27.0 is refused: a torch matrix-multiply bug on that release). `uv` brings a
Python 3.12; nothing needs a compiler or sudo.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install poros-train
poros machine
poros train qwen2.5-7b-instruct --data chats.jsonl
```

On a Mac it pins torch 2.15.0, transformers 5.20.2, peft 0.20.0, accelerate 1.15.0,
safetensors 0.8.0 and datasets 5.1.0 (no bitsandbytes).

## What your Mac can train

`poros models` lists what trains on your Mac, with its row length and whether the frozen layers
stay in memory or come from the drive ([On a Mac](streaming.md#on-a-mac)).
Runs train one row a step at block 1, SFT at lr 2e-5. A model read from the drive holds no rows
out: set `eval_split=0.05`, or run `poros eval last --data more.jsonl`. On another Mac, the
settings come from the M4's, adjusted for its memory (`derived`), and no run there measured them.

## The built-in recipes on a 16 GB M4

The eight Mac recipes, as Caistro Labs certified them on an M4 with 16 GB at this release. Peak
memory is the process's, as `poros recipes` prints it. `mac-first-run` holds its model in memory; the others read their frozen
layers from the drive, here an external USB4 SSD.

| Recipe | Model | Tokens a row | Peak memory | Seconds a step | Steps measured |
|---|---|---|---|---|---|
| `mac-first-run` | Qwen2.5-0.5B-Instruct | 256 | 2.3 GiB | 0.7 | 16 |
| `mac-7b` | Qwen2.5-7B-Instruct | 256 | 4.4 GiB | 5.6 | 16 |
| `mac-support` | Qwen2.5-7B-Instruct | 512 | 5.4 GiB | 9.9 | 16 |
| `mac-voice` | Qwen2.5-7B-Instruct | 512 | 5.4 GiB | 9.9 | 16 |
| `mac-prefer` | Qwen2.5-7B-Instruct | 512 | 6.2 GiB | 18.7 | 16 |
| `mac-moe` | Qwen3-30B-A3B-Instruct-2507 (MoE) | 128 | 9.2 GiB | 18.4 | 16 |
| `mac-32b` | Qwen2.5-32B-Instruct | 128 | 7.2 GiB | 17.7 | 8 |
| `mac-72b` | Qwen2.5-72B-Instruct | 128 | 11.4 GiB | 41.1 | 6 |

A step's time is the median of the steps after the first.

## Where the 4-bit copy goes

A model's download and 4-bit copy go in a store: a folder on a drive that keeps 10% free after
them. Poros takes the folder you name (`--store`, or `store_path`), else
the store that holds the model, else the one this Mac remembers, else the one drive with room; with
several it asks, and remembers the answer in `~/.poros/preferences.toml`.

No drive with room is `E_DISK`; several, in a run that cannot ask, is `E_CONFIG` (set
`store_path`). The internal store is `~/.poros/store`, with the download in your Hugging Face cache
(and the 4-bit copy beside the cache `HF_HOME` or `HF_HUB_CACHE` names); on another drive, a
`poros` folder with a `hub` folder.

```bash
poros train qwen2.5-32b-instruct --data chats.jsonl --store /Volumes/External/poros
```

`--store` names the store folder, a path starting `/`, `~` or `.` (`--store ssd` is `E_USAGE`).
`store=ram` holds every layer in memory and `store=ssd` reads them from the drive, here as in a
config; a model this Mac can't hold in memory is refused
before anything loads (`--force` starts anyway). On an external store the run's files go in its
`runs` folder; set `POROS_HOME` to keep them elsewhere.

## What differs from an NVIDIA GPU

A Mac trains one micro-batch at a time (`path: sequential`) on PEFT's modules
(`fused_compute: false`), and refuses a setting it picks itself, such as `resident_blocks`.
A config file's `eval`, `poros bench` and
`poros prove --unvalidated` need CUDA; `poros eval <run>` runs, `poros prove` compares a run from
memory with one from the drive, and a Linux run exports here unless it recorded no base revision
(then export it where it trained, or with `--format adapter`).

## Checks that protect your Mac

**Self-check.** On an unmeasured Mac, Poros checks its NF4 kernel, bf16 matrix products and a tiny
run once; a failure is `E_STACK_UNSUPPORTED`.

**Memory guard.** A run stops, with a checkpoint where it can, when swap grows by an eighth of
memory (2 GB on 16 GB), the boot drive drops under 3 GB, or memory pressure is critical
(off: `memory_guard=false`).

## A model Poros has not validated

`poros check <model>` says first what this Mac does with a family: `not-in-registry` or `target`
(next, `poros arch try`) where `--unvalidated` trains it, `not-on-this-device` where it can't. `--unvalidated` trains it by its Hub id or from a folder with its `config.json` and
`.safetensors`. Its settings come from the nearest model of its kind in the Mac table. A Hub id asks what a
catalog model asks: `store_path` when two disks have room, then `unvalidated`, `download` and
`all`. Nothing it fetches sends a stored token, a release not in bf16 is refused, and the run's
`export` says `support: user_asserted`.

## Next

- [Streaming and memory](streaming.md#on-a-mac): memory or the drive, and when each pays.
- [Training](training.md): objectives, data and checkpoints.
- [Models](models.md#supported-families): the supported families.
