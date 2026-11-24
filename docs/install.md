# Installation

Poros installs from PyPI as `poros-train` (the package called `poros` is a different project).

```bash
pip install poros-train
poros machine
```

On a Mac, whose stock Python may be older than 3.11, see [Poros on a Mac](mac.md#install-and-train).

## Requirements

- Python 3.11 or newer.
- **Linux with an NVIDIA GPU of compute capability 8.0 or newer**, the cards with native bf16:
  Ampere (sm_80, sm_86: A100, A10, RTX 30 series), Ada (sm_89: RTX 40 series, L4, L40), Hopper
  (sm_90: H100, H200) and Blackwell (sm_100, sm_103, sm_120: B200, B300, RTX 50 series, RTX PRO
  6000).
- **Or an Apple silicon Mac on macOS 14 or later.** macOS 27.0 is refused (a torch matrix-multiply bug on that release).
- Windows is not supported. WSL2 is untested in this release.

Older cards (V100, T4, RTX 20 series) are refused before anything downloads (`E_GPU_UNSUPPORTED`),
a PyTorch build with no kernels for your card with `E_STACK_UNSUPPORTED` and the build to install,
and a machine with no CUDA GPU with `E_NO_CUDA`.

## Install

| Command | Installs |
|---|---|
| `pip install poros-train` | Poros and everything it trains with: torch, transformers, peft, and bitsandbytes off a Mac |
| `pip install "poros-train[bench]"` | the benchmark harness as well (matplotlib) |

On Linux, it takes the releases Poros is tested on: torch 2.10 to 2.14, transformers 5.20 to
5.22, peft 0.19 to 0.20, bitsandbytes 0.43 to 0.50, accelerate 0.34 to 1.15, datasets 4.0 to 5.1,
safetensors 0.4 to 0.8. A Mac installs one exact set ([Poros on a Mac](mac.md#install-and-train)).

## Check your machine

```bash
poros --version
poros machine
poros recipes                              # the certified runs, and the machine each ran on
```

`poros machine` checks the GPU, the driver, host RAM, the disk and the installed packages before
anything downloads, says whether the machine is ready to train, and prints the next command.
[Exit codes](cli.md#json-output-and-exit-codes) lists when it exits
non-zero.

## From source

```bash
git clone https://github.com/Caistro-Labs/poros.git
cd poros
pip install -e ".[bench,dev]"
pytest
```

[Contributing](https://github.com/Caistro-Labs/poros/blob/main/CONTRIBUTING.md) says how to report
a bug, a hardware result or an idea.

## The latest from main

PyPI has the stable releases. To install what is on `main` today:

```bash
pip install git+https://github.com/Caistro-Labs/poros
```

## Next

- [Quickstart](quickstart.md): train, evaluate and export an adapter.
- [Poros on a Mac](mac.md): the settings Poros picks on a Mac.
- [Coding agents](agents.md): let a coding agent run the fine-tune for you.
