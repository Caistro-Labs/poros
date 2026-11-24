# scripts/

The release's own tools: the checks a release must pass, the drivers that certify the built-in
recipes, and the benchmark and parity drivers that wrote the logs in
[Caistro-Labs/poros-benchmarks](https://github.com/Caistro-Labs/poros-benchmarks). None of them is
part of the installed package. Each file's first lines say how to run it.

## Folders

| Folder | What it holds |
|---|---|
| `certify/` | `run_recipe.py` certifies one built-in recipe on one machine; `generate.py` writes `poros/recipes/certified.json` from the certification runs `sources.toml` names; `decisions.py` prints what the pre-run check decides for each recipe on a given machine |
| `family_gate/` | the helpers of `family_gate.sh`: the model lists (`families.tsv`, `catalog.tsv`, `anymodel.tsv`), the fetch of public Hub files, and the comparisons of two runs, of a resume, and of a refusal |
| `metal/` | the Mac's checks: losses against a CUDA run (`track.py`), stop and continue (`resume.py`), injected faults (`mutations.py`), `poros agents` on real runs (`agents_gate.py`), the training arms (`arms.py`) and the whole product path in a fresh home (`seamless.sh`) |
| `moe_precision/` | the expert optimizer's precision kit: a self-test of the fused kernel on a GPU (`device_selftest.py`), the Metal arithmetic transcribed and checked on the CPU (`metal_exact_arith.py`), the runs' configs, data, digests and verdict |
| `sss/` | the SSD expert store: every MoE family on it (`family_gate_ssd.sh`) and its result against the RAM store's (`gate_moe_ram_vs_ssd.py`) |
| `xcard/` | the fused kernels on a card: the environment, LoRA ranks 8 to 256, and the comparison with PEFT (`xcard_check.sh` runs them) |
| `inkling2/` | cuts a small slice of the released Inkling 2 Mini checkpoint for the CPU comparison with transformers (the family does not train in 1.0.0) |
| `cli/` | the CLI's screens without a GPU: a simulated training view and the out-of-memory screen, and `try`, which runs this tree's CLI |

## Scripts

| Script | What it does |
|---|---|
| `family_gate.sh` | every supported model family on this card: load, train, the paths bit for bit, resume, and "test any model" end to end |
| `xcard_check.sh` | the fused kernels' checks on this card |
| `training_parity.py` | streamed training against the whole model held on the GPU, bit for bit |
| `streamed_load_gate.py` | the block-streamed NF4 load against the standard load, tensor by tensor |
| `dense_resume_gate.py` | a dense run stopped and resumed through the CLI, against one run straight through |
| `parity_modes_sweep.py` | streamed against resident under PyTorch's execution modes |
| `moe_absolute_gate.py` | a MoE model: parity, speed and held-out loss |
| `moe_grouped_pin.py` | pins and checks the grouped expert backend on one wide MoE layer |
| `moe_synthetic_quality.py` | the MoE quality run of the paper's Appendix B |
| `expert_adam_precision.py` | the CPU study of the expert adapters' optimizer (its log: `docs/validation/int/cpu/expert_adam_precision/`) |
| `build_plan_cells.py` | writes `poros/plan/cells.json`, the measured figures the pre-run check reads, from the logs `docs/validation/plan_sources.yaml` names |
| `build_plan_configs.py`, `build_plan_shapes.py` | write the model configs and shapes that check reads, so it needs no download |
| `gen_cli_words.py` | writes the command table in `docs/cli.md` from `tests/cli/sentences.yaml` |
| `release.py` | the release helper: the version, the tag and the signing key's file |
| `smoke_wheel.sh` | installs the built wheel in a fresh environment and runs it from outside the repository |
| `repo_rules.sh` | applies the public repository's settings and branch rules with `gh` |

`build_plan_cells.py` finds the logs through `POROS_BENCHMARKS`, the root of a checkout of
poros-benchmarks, as the tests do.
