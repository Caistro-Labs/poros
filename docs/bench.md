# Benchmarks

This page has Poros's measured results, on hardware people own and beside Unsloth on one RTX 4090.
It also covers `poros bench`, a harness for timing Poros and other methods on your own machine. The
results came from their own scripts, not this harness: each run's command, environment and log are
in the runs folder they link to.

<!-- fill:results -->
## Results

What Poros trains on hardware people own:

- **A 72B on one 24 GB card.** The built-in `big-72b` recipe trains Qwen2.5-72B-Instruct on an RTX 4090 with 18.2 GiB of GPU memory (VRAM) reserved.
  It needs about 45 GiB of system RAM (peak 45.2 GiB, measured on a machine with 59.6 GiB). A step takes 19.5 s. Qwen2.5-72B's 4-bit weights alone are 41.4 GiB. Qwen2.5-72B and its Instruct model are under the Qwen license; Poros itself is Apache-2.0.
- **A 30B MoE on an 8 GB laptop.** A measured run of the built-in `moe-30b-8gb` recipe (certified on the RTX 4090) trained Qwen3-30B-A3B-Instruct-2507 on an RTX 4060 Laptop GPU with
  6.0 GiB of VRAM reserved and 6.4 GiB of host RAM, at LoRA rank 8 and rows of 256 tokens.
  A step of 8 rows took 13.6 s. Its expert store was built on the laptop.
- **A 72B on a 16 GB Mac.** The built-in `mac-72b` recipe trains Qwen2.5-72B-Instruct on an M4 Mac with 16 GB at 11.4 GiB of peak memory, with
  rows of 128 tokens. A step takes 41.1 s.

The head-to-head runs, Poros and Unsloth on the same RTX 4090. Lower is better in every column.

| Model | Tool (setting) | Time per step | VRAM reserved | Host RAM | Host RAM, load | Host RAM, training | Checked |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-72B | Poros (efficient, default) | 17.7 s (17.7–17.7) | 13.5 GiB (13.5–13.8) | 45.1 GiB (45.1–45.2) | 39.1 GiB (39.1–39.1) | 45.1 GiB (45.1–45.2) | run-to-run repeat |
| Qwen2.5-72B | Unsloth 2026.11.4 (offload auto), November 18, 2026 | 22.1 s (20.9–22.1) | 21.1 GiB (21.1–21.1) | 43.0 GiB (42.7–43.0) | 43.0 GiB (42.7–43.0) | 25.7 GiB (25.7–25.8) | – |
| Qwen3-30B-A3B | Poros (efficient, default) | 4.6 s (4.6–4.6) | 9.1 GiB (9.1–9.1) | 19.9 GiB (19.8–19.9) | 18.7 GiB (18.7–18.8) | 19.9 GiB (19.8–19.9) | run-to-run repeat |
| Qwen3-30B-A3B | Unsloth 2026.11.10 (offload auto), November 24, 2026 | 8.9 s (8.8–8.9) | 19.7 GiB (19.7–19.7) | 22.9 GiB (22.9–23.2), 35 processes | 10.3 GiB (10.3–10.3) | 22.9 GiB (22.9–23.2), 35 processes | – |

**Key takeaways (Poros median vs. Unsloth's best run)**

- **Qwen2.5-72B:** Poros `efficient` is **1.1x faster** in **1.5x less VRAM** than Unsloth 2026.11.4 (Unsloth uses 6% less peak host RAM and 1.8x less while training).
- **Qwen3-30B-A3B:** Poros `efficient` is **1.9x faster** in **2.1x less VRAM** than Unsloth 2026.11.10 (Poros uses more host RAM: 19.9 GiB against about 11 GiB for Unsloth's own process).

### Notes on the comparison

- **Versions and dates.** The Qwen3-30B-A3B rows ran on November 24, 2026, with Poros at its release commit and Unsloth 2026.11.10, the newest on PyPI that day. The Qwen2.5-72B rows ran on November 18, 2026, with Unsloth 2026.11.4.
- **Unsloth's two versions.** Unsloth 2026.11.10 was slower than 2026.11.4 on the same harness and rows (8.8 s against 5.8 s at its best, 53% longer; the logs do not show why). Against 2026.11.4's figures Poros is 1.2x faster; the two were not measured in one session.
- **Host RAM.** Poros keeps the whole 4-bit base in host RAM (19.9 GiB on the 30B) and uses more of it than a tool that keeps part of the model on the GPU. Unsloth's figure is summed over its process tree, which reached 35 processes with 2026.11.10 (22.9 GiB); its main process peaked at 10.8 GiB, and 2026.11.4 peaked at 18.1 GiB with two or three processes. There is no host-RAM ratio on the 30B; every 72B run peaked with two processes.
- **What the VRAM ratio is set against.** Unsloth's automatic offload keeps layers on the GPU while there is room: on the 30B, 28 of 48 decoder layers and the embedding went to host RAM "so 13.32 GiB stays free for training".
- **All layers offloaded.** With all 48 layers and the embedding in host RAM (November 18, Unsloth 2026.11.4, the same harness), Unsloth reserved 13.9 GiB at 9.5 s an update, the median of three runs.
- **Other batch shapes.** With 8 rows as one micro-batch and no accumulation, the November 18 comparison had Unsloth faster. That shape has not been repeated on the release commit.
- **Quantization and adapters.** Poros makes the experts 4-bit (NF4) and keeps attention, norms, router and embeddings in bf16; Unsloth loads the model in 4 bits (`load_in_4bit`, bitsandbytes NF4 on its linear layers). Both train rank-16 LoRA adapters: on the 72B, fp32 on the attention and MLP projections; on the 30B, on the routed experts, where Poros trains bf16 adapters with an fp32 update ([MoE models](moe.md)) and Unsloth fp32 ones (and also attaches frozen adapters to `o_proj`). That is part of the gap in reserved memory.
- **transformers versions.** Each tool ran on the version its own requirements resolve to: 5.17.0 for Unsloth on both dates, 5.22.2 for Poros. The note beside the runs (`h2h/4090-r2/NOTE_unsloth_versions.md`) lists what else differs.
- **How the table is built.** Both tools load each model from the same bf16 copy and train at batch 1 with gradient accumulation (8 rows a step), on rows of 512 tokens. Each cell is the median of three interleaved runs, one session per model, range beside it; ratios set Poros's median against the other tool's best run, to one decimal (a whole percent under 1.1x), rounded against Poros. VRAM is the peak reserved; host RAM is the peak non-file-cache memory, split into the load and training (Unsloth's 72B peak comes during its load, Poros's during training). The Checked column names what each Poros row was compared against.

Time from launch to the end of the first step, on Qwen3-30B-A3B, on November 24, 2026.

| Tool (setting) | Launch to first step |
|---|---:|
| Poros (efficient, default) | 122.4 s |
| Unsloth 2026.11.10 (offload auto) | 139.0 s |

- Poros (efficient, default) reached its first step 9% sooner than Unsloth 2026.11.10's best run (133.5 s; 139.0 s is its median). Unsloth's previous version, 2026.11.4, reached its first step in 154.5 s at its best (158.7 s median) on November 18, 2026.

The standard way to load Qwen2.5-72B puts the whole model on the GPU, with transformers,
bitsandbytes and PEFT. On the same RTX 4090, with the same rows and adapters, that run stops with
`torch.OutOfMemoryError: CUDA out of memory.` (out of memory).

A run stopped and continued gives the same result as one that ran straight through: every certified run but `mac-72b`'s was stopped halfway and continued, and ended at the same adapter as its uninterrupted run, byte for byte. For a model the GPU can hold whole, streamed training is also the same as training with the whole model there.

- The whole model on the GPU: `poros prove` on Qwen2.5-7B on an RTX 4090, streamed and with the whole model on the GPU, ends 0.00e+00 apart after 200 steps of 512-token rows. A step took 0.47 s streamed and 0.30 s with the whole model on the GPU.

[Reproducibility](training.md#reproducibility) says what holds.

The raw benchmark logs behind every figure on this page, with each run's settings, are in the
companion repository [Caistro-Labs/poros-benchmarks](https://github.com/Caistro-Labs/poros-benchmarks),
at the same paths the docs and the paper cite: the runs above are in
[docs/validation/h2h](https://github.com/Caistro-Labs/poros-benchmarks/tree/main/docs/validation/h2h).

## Certified runs

Every built-in recipe's certified run on the release build.

| Machine | Recipe | Model | Step time | Memory (GPU) | Host RAM |
|---|---|---|---:|---:|---:|
| RTX 4090 | `first-run` | Qwen2.5-0.5B-Instruct | 2.3 s | 1.5 GiB | 2.5 GiB |
| RTX 4090 | `support` | Qwen2.5-7B-Instruct | 2.9 s | 7.3 GiB | 7.3 GiB |
| RTX 4090 | `voice` | Qwen2.5-7B-Instruct | 2.9 s | 7.3 GiB | 7.3 GiB |
| RTX 4090 | `prefer` | Qwen2.5-7B-Instruct | 3.5 s | 9.7 GiB | 7.3 GiB |
| RTX 4090 | `big-32b` | Qwen2.5-32B-Instruct | 8.3 s | 12.8 GiB | 19.2 GiB |
| RTX 4090 | `moe-30b` | Qwen3-30B-A3B-Instruct-2507 | 4.6 s | 8.9 GiB | 19.9 GiB |
| RTX 4090 | `moe-30b-ssd` | Qwen3-30B-A3B-Instruct-2507 | 4.3 s | 21.6 GiB | 11.2 GiB |
| RTX 4090 | `moe-30b-8gb` | Qwen3-30B-A3B-Instruct-2507 | 15.8 s | 5.7 GiB | 7.4 GiB |
| RTX 4090 | `big-72b` | Qwen2.5-72B-Instruct | 19.5 s | 18.2 GiB | 45.2 GiB |
| RTX 4090 | `big-72b-fast` | Qwen2.5-72B-Instruct | 15.8 s | 21.7 GiB | 42.9 GiB |
| M4 Mac, 16 GB | `mac-first-run` | Qwen2.5-0.5B-Instruct | 0.7 s | 2.3 GiB (unified) | - |
| M4 Mac, 16 GB | `mac-support` | Qwen2.5-7B-Instruct | 9.9 s | 5.4 GiB (unified) | - |
| M4 Mac, 16 GB | `mac-voice` | Qwen2.5-7B-Instruct | 9.9 s | 5.4 GiB (unified) | - |
| M4 Mac, 16 GB | `mac-prefer` | Qwen2.5-7B-Instruct | 18.7 s | 6.2 GiB (unified) | - |
| M4 Mac, 16 GB | `mac-7b` | Qwen2.5-7B-Instruct | 5.6 s | 4.4 GiB (unified) | - |
| M4 Mac, 16 GB | `mac-32b` | Qwen2.5-32B-Instruct | 17.7 s | 7.2 GiB (unified) | - |
| M4 Mac, 16 GB | `mac-moe` | Qwen3-30B-A3B-Instruct-2507 | 18.4 s | 9.2 GiB (unified) | - |
| M4 Mac, 16 GB | `mac-72b` | Qwen2.5-72B-Instruct | 41.1 s | 11.4 GiB (unified) | - |
| RTX 4060 Laptop | `first-run` | Qwen2.5-0.5B-Instruct | 1.4 s | 1.5 GiB | 2.5 GiB |

## Architectures checked

Poros supports models by [architecture](architectures.md). Each family below passed the same checks on the release build on an RTX 4090: every training path gives the same adapter as training fully on the GPU, and a repeated run is bit-identical; where run, a stopped run resumes to the same adapter.

| Family | Checked on |
|---|---|
| Qwen2.5 | Qwen/Qwen2.5-0.5B |
| Qwen3 | Qwen/Qwen3-0.6B |
| Qwen3.5 | Qwen/Qwen3.5-0.8B |
| Qwen3.8 | Qwen/Qwen3.8-27B, first 8 layers |
| Gemma 3, text | gemma-3-270m |
| Gemma 3 | gemma-3-4b-it, first 6 layers |
| Gemma 4 | google/gemma-4-31B, first 6 layers |
| Gemma 4, text | google/gemma-4-31B, text only, 6 layers |
| Qwen3 MoE | Qwen/Qwen3-30B-A3B, first 4 layers |
| OLMoE | allenai/OLMoE-1B-7B-0924, first 4 layers |
| Qwen2 MoE | Qwen/Qwen1.5-MoE-A2.7B, first 4 layers |
| DeepSeek-V2 | deepseek-ai/DeepSeek-V2-Lite, first 4 layers |
| Qwen3.5 / 3.6 MoE | Qwen/Qwen3.6-35B-A3B, first 4 layers |
| Qwen3.5 / 3.6 MoE, text | Qwen/Qwen3.6-35B-A3B, text only, first 4 layers |
| Gemma 4 MoE | google/gemma-4-26B-A4B-it, first 6 layers |

## More reference runs

These runs were made from November 17 to 20, 2026, before the release commit (Poros at 5d75e31a and earlier commits, Unsloth 2026.11.4), and were not repeated on it.

### The laptop run of November 18

Qwen3-30B-A3B trained on an RTX 4060 Laptop GPU (efficient, SSD) with
5.3 GiB of VRAM reserved and 7.4 GiB of host RAM, at LoRA rank 8 and rows of 256 tokens.
A step of 8 rows took 12.2 s (expert store built beforehand on the RTX 4090, not on the laptop).
The checkpoint copy left out the expert weights, which training never reads once the store exists.

### Head-to-head on November 18

| Model | Tool (setting) | Time per step | VRAM reserved | Host RAM | Host RAM, load | Host RAM, training | Checked |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-72B | Poros (efficient, default) | 17.7 s (17.7–17.7) | 13.5 GiB (13.5–13.8) | 45.1 GiB (45.1–45.2) | 39.1 GiB (39.1–39.1) | 45.1 GiB (45.1–45.2) | run-to-run repeat |
| Qwen2.5-72B | Poros (fast) | 13.7 s (13.7–13.7) | 21.6 GiB (21.6–21.6) | 42.9 GiB (42.9–43.0) | 39.1 GiB (39.1–39.1) | 42.9 GiB (42.9–43.0) | run-to-run repeat |
| Qwen2.5-72B | Unsloth 2026.11.4 (offload auto), November 18, 2026 | 22.1 s (20.9–22.1) | 21.1 GiB (21.1–21.1) | 43.0 GiB (42.7–43.0) | 43.0 GiB (42.7–43.0) | 25.7 GiB (25.7–25.8) | – |
| Qwen2.5-72B | Unsloth (80 of 80 requested) | did not train | – | – | – | – | – |
| Qwen3-30B-A3B | Poros (efficient, default) | 4.6 s (4.6–4.6) | 8.1 GiB (8.1–8.3) | 19.9 GiB (19.9–20.0) | 18.7 GiB (18.7–18.8) | 19.9 GiB (19.9–20.0) | run-to-run repeat |
| Qwen3-30B-A3B | Poros (SSD store, fast+cache) | 4.2 s (4.1–4.2) | 19.8 GiB (19.7–20.2) | 11.2 GiB (11.2–11.2) | 6.8 GiB (6.8–6.8) | 11.2 GiB (11.2–11.2) | SSD = RAM |
| Qwen3-30B-A3B | Poros (SSD store, fast) | 5.9 s (5.2–6.2) | 18.4 GiB (18.4–18.4) | 6.8 GiB (6.8–6.8) | 6.8 GiB (6.8–6.8) | 6.1 GiB (6.1–6.1) | SSD = RAM |
| Qwen3-30B-A3B | Poros (SSD store, GPU 7 + host 8 GiB) | 4.3 s (4.2–4.4) | 16.6 GiB (16.1–16.6) | 15.2 GiB (15.2–15.2) | 6.8 GiB (6.8–6.8) | 15.2 GiB (15.2–15.2) | SSD = RAM |
| Qwen3-30B-A3B | Poros (SSD store, efficient) | 18.0 s (17.9–19.2) | 7.7 GiB (7.7–7.8) | 7.9 GiB (7.9–7.9) | 6.8 GiB (6.8–6.8) | 7.9 GiB (7.9–7.9) | SSD = RAM |
| Qwen3-30B-A3B | Unsloth (offload auto) | 6.2 s (5.8–6.3) | 19.8 GiB (19.8–19.8) | 18.1 GiB (18.1–18.1) | 18.1 GiB (18.1–18.1) | 11.0 GiB (10.9–11.0) | – |
| Qwen3-30B-A3B | Unsloth (38 of 48 offloaded) | 8.1 s (7.5–8.4) | 16.9 GiB (16.9–16.9) | 23.8 GiB (23.7–24.0) | 23.8 GiB (23.7–24.0) | 13.8 GiB (13.7–13.8) | – |
| Qwen3-30B-A3B | Unsloth (all layers offloaded) | 9.5 s (9.3–10.7) | 13.9 GiB (13.9–13.9) | 29.8 GiB (29.7–29.8) | 29.8 GiB (29.7–29.8) | 16.7 GiB (16.7–16.7) | – |
| Qwen3-30B-A3B, no accumulation | Poros (efficient, default) | 2.5 s (2.4–2.5) | 8.1 GiB (8.1–8.1) | 19.8 GiB (19.8–19.9) | 18.7 GiB (18.7–18.7) | 19.8 GiB (19.8–19.9) | run-to-run repeat |
| Qwen3-30B-A3B, no accumulation | Poros (SSD store, fast+cache) | 1.4 s (1.4–1.4) | 20.2 GiB (20.1–20.2) | 11.2 GiB (11.2–11.2) | 6.8 GiB (6.8–6.8) | 11.2 GiB (11.2–11.2) | SSD = RAM |
| Qwen3-30B-A3B, no accumulation | Unsloth (offload auto) | 1.7 s (1.7–1.7) | 19.8 GiB (19.8–19.8) | 18.2 GiB (17.9–18.2) | 18.2 GiB (17.9–18.2) | 11.7 GiB (11.7–11.9) | – |

Unsloth (80 of 80 requested) on Qwen2.5-72B: did not train: killed by the kernel while loading at 58.1 GiB of host RAM on a 59.6 GiB machine; one attempt.

Both tools train the same LoRA adapters at batch 1 with gradient accumulation (8 rows a step), except the rows without accumulation; rows of 512 tokens, LoRA rank 16. On Qwen2.5-72B both trained fp32 adapters.

Time from launch to the end of the first step, on Qwen3-30B-A3B. With the SSD store, Poros's first start builds the store, and later starts reuse it.

| Tool (setting) | First start | Later starts |
|---|---:|---:|
| Poros (efficient, default) | – | 137.0 s |
| Poros (SSD store, fast+cache) | 175.2 s | 34.5 s |
| Poros (SSD store, fast) | 175.2 s | 33.9 s |
| Poros (SSD store, GPU 7 + host 8 GiB) | 175.2 s | 35.2 s |
| Poros (SSD store, efficient) | 175.2 s | 36.9 s |
| Unsloth (offload auto) | – | 158.7 s |
| Unsloth (38 of 48 offloaded) | – | 141.5 s |
| Unsloth (all layers offloaded) | – | 146.2 s |

On Qwen2.5-72B, Poros reached its first update in 309.8 s on its default path and 298.5 s on its fast path (median of three).

**Unsloth's default sources.** The runs above load both tools from the same bf16 copy of each model. In an earlier session, in a single run, Unsloth trained Qwen2.5-72B from its own pre-quantized 4-bit copy (offload auto, 8 updates): 21.1 s a step, 21.0 GiB of VRAM reserved and 42.9 GiB of host RAM. Its load took 140.2 s, and its first step ended 176.7 s after launch. Both lower-memory settings from that copy (all 80 layers and the embedding offloaded, and 70 of 80) were killed by the kernel while loading, at 58.0 and 58.3 GiB of host RAM on a 59.6 GiB host, one attempt each. From Unsloth's 4-bit copy of Qwen3-30B-A3B, a single run of an earlier session failed at the first training step.

### Exactness checks

Poros gives the same result however it moves the weights: change the path, the block size or the store, and the adapter comes out the same. A resumed run ends 0.00e+00 from an uninterrupted one after 16 steps. For a model the GPU can hold whole, it is also the same as training with the whole model there.

- The whole model on the GPU: DeepSeek-R1-Distill-Qwen-1.5B on an RTX 4060 Laptop GPU, streamed and with the whole model on the GPU, ends 0.00e+00 apart after 100 steps. A step took 0.47 s streamed and 0.38 s with the whole model on the GPU.
- The path: on Qwen2.5-72B, the fast path's adapter equals the default path's.
- The block size: on a 7B model, block size 4 and all layers in one block end 0.00e+00 apart after 100 steps.
- The store: on Qwen3-30B-A3B, the SSD store gives the same result as the RAM store.

### The RTX 4090 with 5.25 GiB of VRAM free

These runs use an RTX 4090 with 5.25 GiB of its VRAM left free. Another process held 18,318 MiB of the card's 24,564 MiB, leaving 5.25 GiB free by Poros's own reading at the start of each run. This was tighter than the 8 GB laptop on November 18, where the same reading was 6.73 GiB at its lowest (6.75, 6.75, 6.73). Host RAM was not limited.

Every run here is Qwen3-30B-A3B at rows of 256 tokens, LoRA rank 8, 8 rows a step.

| Tool (setting) | Time per step | VRAM reserved | Host RAM |
|---|---:|---:|---:|
| Poros (efficient, SSD) | 15.1 s (15.1–19.8) | 5.1 GiB (5.1–5.1) | 7.4 GiB (7.4–7.4) |
| Unsloth (all layers offloaded) | did not train | – | – |
| Unsloth (all layers offloaded), the whole card | 9.8 s (9.8–10.0) | 8.0 GiB (8.0–8.0) | 29.4 GiB (29.4–29.7) |

What Unsloth (all layers offloaded) did with 5.25 GiB free, as the run's record says: did not train: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB.

Against Unsloth with the whole card, Unsloth was 1.5x faster; Poros used 1.5x less VRAM and 4.0x less host RAM, and reached its first step 4.4x sooner.

Here too, Poros trains the expert adapters in bf16 and Unsloth trained fp32 adapters, so part of the VRAM difference is adapter precision.

Poros's row here was checked by run-to-run repeat.
<!-- /fill:results -->

## Running the harness

`poros bench` trains one model with several methods, all from one spec. The spec fixes the model,
the LoRA shape, the row length, the steps, the batch size and the accumulation. Each method's result
is saved as a `RunArtifact`.

```bash
pip install "poros-train[bench]"
poros bench run spec.yaml -o ./bench_output       # one spec
poros bench matrix matrix.yaml                    # several models
poros bench compare ./bench_output/a ./bench_output/b
poros bench report ./bench_output --md            # report.html, and report.md
```

`-p N` repeats each cell N times and reports the median. Without it, a cell repeats as often as
the spec's `n_passes` says. `poros report <dir>` and `poros leaderboard <dir>` read the same
artifacts without torch, from a folder or one `.json` or `.jsonl` file. `poros bench` has no
`--json` mode.

A spec:

```yaml
spec_id: qwen2_5_32b_4090

hardware:
  gpu_name: "NVIDIA GeForce RTX 4090"
  vram_gb: 24.0

models:
  - model_id: "Qwen/Qwen2.5-32B"
    display_name: "Qwen2.5-32B"

methods: [poros, resident, unsloth, peft, accelerate_offload]

training:
  steps: 200
  seq_len: 512
  batch_size: 1
  gradient_accumulation_steps: 8
  dataset: "synthetic"
  seed: 42

lora:
  rank: 16
  alpha: 32
  dropout: 0.0

poros:
  block_size: 4

reporting:
  output_dir: "./bench_output/qwen2_5_32b_4090"
```

`deepspeed_zero3` is a method you opt into, after `pip install deepspeed`. Its `zero3:` block sets
`offload_device` (`cpu` or `nvme`) and `nvme_path`. The `accelerate_offload:` block sets the GPU's
share of the device map (`gpu_max_memory_gb`) and `gradient_checkpointing`.
`hardware.vram_cap_gb` caps the VRAM of every method alike.

## Fairness guards

- Every method gets the same model, row length, LoRA rank, alpha, target modules, steps and
  tokens, from one spec.
- Every method runs in a fresh process and is measured by the same code.
- That code records peak VRAM in the training loop (`vram_peak_mb`) and in the load
  (`vram_load_peak_mb`), and times each optimizer step with a synchronized clock.
- Each cell records its environment: the driver, PCIe link, host RAM and container limits, thread
  count and package versions.

Two things differ between methods. The Poros method trains through `PorosTrainer`'s own data path
and schedule. Gradient checkpointing is on for DeepSpeed and Unsloth, and off for PEFT and
resident. So treat seconds per step as a guide, not a ranking. The harness measures speed and
memory, not results: `poros prove` checks what a run computes.

## Output

The output folder is `reporting.output_dir`, or `-o`. It gets `spec.json`, one
`{method}_{arch}_{precision}.json` artifact per cell, and `results.jsonl`, which grows as each cell
finishes. `poros schema artifact` prints the artifact's JSON Schema.

## Next

- [Streaming and memory](streaming.md): the settings behind each result, for your machine.
- [Limits](limitations.md): what Poros does not do.
