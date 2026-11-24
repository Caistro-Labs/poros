# Poros Technical Report

`poros.pdf` is the Caistro Labs technical report "Poros: QLoRA Fine-Tuning of Models Larger Than the GPU by Streaming the Frozen Base."

## Rebuilding
Check out the benchmark logs beside this repository and run `make` (needs `xelatex`, `latexmk`, and `matplotlib`; fonts are in `paper/fonts/`):
```sh
git clone https://github.com/Caistro-Labs/poros-benchmarks ../poros-benchmarks
make -C paper POROS_BENCHMARKS=$PWD/../poros-benchmarks
```
`make final` also refuses to build if any value is missing or if a log contradicts a comparison stated in the paper.

## Numbers and logs
No measured number is typed into the LaTeX source; the text prints a macro per value. `paper/src/launch_results.py` reads each value from a named log under `docs/validation/` in `github.com/Caistro-Labs/poros-benchmarks` and writes `values.tex`, the tables, and Figure 1. `paper/src/launch_values.json` records every value's source files and field. Every Poros figure, and every log, must name the release engine's commit, with one stated exception: the 18 November comparison session (Poros at commit 5d75e31a against Unsloth 2026.11.4), which gives the Qwen2.5-72B comparison and the earlier Unsloth version's Qwen3-30B-A3B figures. The script stops on any other log that names another commit. `paper/src/logs_appendix.py` writes Appendix A (each group of numbers with its logs) and `paper/log_digests.txt` (the SHA-256 of each cited log as the benchmark-logs repository holds it).

## Files
* `paper/src/main.tex`: LaTeX source (`xelatex`).
* `paper/src/paths.tex`: prose on what `path: efficient` and `path: fast` do.
* `paper/src/fig_design.tex`: Figure 2 (TikZ, no measured numbers).
* `paper/fonts/`: XCharter (Bitstream Charter license and OFL); Geist and Geist Mono (OFL).

## Provenance
The commit IDs in the logs and the paper come from Caistro Labs' development repository; the public history of `github.com/Caistro-Labs/poros` starts at the 1.0.0 export. The released `v1.0.0` package matches commit `1a8d7e912` except six display files under `poros/cli` (`train_cmd.py` and `ui/components.py`, `curve.py`, `run_card.py`, `screens.py`, `train_view.py`) and two generated files (`poros/recipes/certified.json` and `poros/plan/cells.json`). A run's computation is unchanged: the release's resolve check shows this, and `first-run` ends at its certified adapter with the released package on each machine, as stated on the front page of `github.com/Caistro-Labs/poros-benchmarks`. The 24 November Qwen3-30B-A3B comparison was measured at commit `1a8d7e912`. The Qwen2.5-72B comparison was not repeated at that commit; it is the 18 November session, Poros at commit `5d75e31a` against Unsloth 2026.11.4, and the paper says so where it gives it.
