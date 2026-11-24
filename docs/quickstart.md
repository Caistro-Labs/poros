# Quickstart

Train a LoRA adapter on your own chats, see whether it learned, and export a model you can serve.
You need Linux with an NVIDIA GPU (Ampere or newer) or an Apple silicon Mac, and Python 3.11 or
newer. [Installation](install.md) has the details.

## Install and check your machine

```bash
pip install poros-train
poros machine
```

`poros machine` checks your GPU, driver, host RAM, disk and torch, says whether this machine is
ready to train, and prints the next command. `poros recipes` lists the certified runs and the
machine each ran on.

## Write your data

Put one chat per line in a `.jsonl` file:

```json
{"messages": [{"role": "user", "content": "What is Poros?"}, {"role": "assistant", "content": "A fine-tuning library."}]}
```

Chats train SFT on the answers. Preference pairs, one
`{"prompt": "...", "chosen": "...", "rejected": "..."}` per line, train DPO. Other shapes are in
[Data](training.md#data).

## Train

```bash
poros train qwen2.5-0.5b-instruct --data chats.jsonl
```

Start with this small model to try the whole loop. With no data yet,
`poros recipes show first-run` prints a command that needs none. Before anything downloads, `poros train` checks
your data, the model, your GPU, its memory and host RAM, in seconds. It shows every setting and asks before it
starts. The run's files go to `~/.poros/runs/<id>/`. `poros models` lists every model Poros names,
with the recipes certified on this machine, and `poros recipes` those runs.

## Check what it learned

```bash
poros eval last
```

`poros eval` scores the adapter beside the base model on rows the run set aside, 5% of a file of
40 rows or more, and ends with a verdict such as `Learned well`. For a smaller file, score another
with `poros eval last --data more_chats.jsonl`. A small file of pairs is scored on the pairs it
trained on, so its verdict is not a held-out result. On a Mac, a model read from the drive holds no
rows out unless you set `eval_split=0.05`. A file of text rows holds none out: give a held-out file
with `--eval FILE` when you train, or train it with `--recipe sft`.

## Export a model

```bash
poros export last --format merged --output ./merged
```

A merged export is a standard bf16 model that vLLM, TGI and transformers load. `--format adapter`
writes the adapter alone, for a server that loads adapters or for PEFT (on CUDA, a MoE model's adapter is in Poros's own format).

## Stop and resume

```bash
poros runs stop last       # in a second terminal: stops the run and saves a checkpoint
poros runs continue last   # goes on from that checkpoint
```

Ctrl-C stops at once and saves nothing. [Checkpoints and resume](training.md#checkpoints-and-resume)
covers the rest.

## From Python

`poros.Run` is a run you drive one step at a time:

```python
import poros

run = poros.Run.create("Qwen/Qwen2.5-7B-Instruct", rank=16, length=512, dataset="chats.jsonl")
data = poros.data.chats("chats.jsonl", run.tokenizer, length=512)
for step in run.steps(200):
    run.forward_backward(data.batch(step))
    run.optim_step()
run.save()
```

The built-in recipes, such as `poros.recipes.dpo.main(run, pairs)`, are loops like this one.
[Python API](api.md) has the rest.

## From a config file

```bash
poros init run.yaml -m Qwen/Qwen2.5-7B-Instruct -d ./chats.jsonl -o ./run
poros train run.yaml --steps 200
```

`poros schema config` lists every field. [Commands](cli.md#config-files) covers resuming and
exporting a config run.

## Good to know

- `poros` alone opens an interactive session, where `/train` sets up a run on one card.
- `poros train --from last --data prefs.jsonl` trains DPO on top of the SFT run.
- `poros train --detach` trains in the background; `poros runs` lists your runs.
- A model larger than your GPU trains the same way: [Streaming and memory](streaming.md).

## Next

- [Training](training.md): objectives, data, checkpoints and what fits.
- [Streaming and memory](streaming.md): where the weights wait, and which path to pick.
- [Commands](cli.md): every command and flag.
