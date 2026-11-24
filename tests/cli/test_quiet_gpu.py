"""machine, models and fit beside a run: nvidia-smi and CUDA's own numbering, with no torch."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import poros
from poros import jobs
from poros.cli.commands import check as cli_commands_check
from poros.core import card
from tests.conftest import another_machine, as_linux, needs_ml

LAPTOP = "GPU-aaaa, NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 322, 7481"  # as a real one reads
DESK = "GPU-bbbb, NVIDIA GeForce RTX 3090, 24576, 317, 24000"


def _smi(folder: Path, rows: str) -> Path:
    """An nvidia-smi on PATH that keeps its arguments: the cards, and the card's live row."""
    (folder / "rows").write_text(rows)
    tool = folder / "nvidia-smi"
    tool.write_text(f'#!/bin/sh\necho "$*" >> {folder}/asked\ncase "$*" in\n'
                    f'*uuid,name*) cat {folder}/rows ;;\n'
                    '*driver_version*) echo 570.86, 1000, 8188, 3, 40, 20, 100, 0, 4, 16 ;;\n'
                    'esac\n')
    tool.chmod(0o755)
    return folder


@pytest.fixture
def two_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{_smi(tmp_path, f'{LAPTOP}{chr(10)}{DESK}{chr(10)}')}{os.pathsep}"
                               f"{os.environ['PATH']}")  # for a poros this test starts
    another_machine(monkeypatch, tmp_path)  # these two cards, and never the host's own
    # a CUDA host's readers, on any host: a Mac's own read its chip, not nvidia-smi
    as_linux(monkeypatch, card, jobs, cli_commands_check)
    return tmp_path


def _order(monkeypatch, order) -> None:
    monkeypatch.setattr(card, "_cuda_order", lambda: order)


def test_the_card_is_the_one_cuda_numbers_first_not_the_first_row(two_cards, monkeypatch):
    _order(monkeypatch, ["GPU-bbbb", "GPU-aaaa"])
    gpu = card.detect_gpu()
    assert (gpu.name, gpu.uuid, gpu.source) == ("NVIDIA GeForce RTX 3090", "GPU-bbbb", "nvidia-smi")
    assert jobs.gpu_uuid() == "GPU-bbbb"  # the lock and the pin are this read's card
    _order(monkeypatch, ["GPU-aaaa"])  # CUDA_VISIBLE_DEVICES, in any of its forms
    assert card.detect_gpu().uuid == "GPU-aaaa"
    assert "uuid" not in card.detect_gpu().model_dump()  # never in a report


def test_totals_are_what_cuda_can_use(two_cards, monkeypatch):
    """Measured on the 4060 Laptop this row is from: torch's total is 7866.3 MiB, nvidia-smi's less
    its memory.reserved; its free is the card's, without a reader's own context."""
    _order(monkeypatch, ["GPU-aaaa"])
    gpu = card.detect_gpu()
    assert (gpu.total_gib, gpu.free_gib) == (7.68, 7.31)


def test_no_device_visible_is_no_gpu_and_asks_no_torch(two_cards, monkeypatch):
    _order(monkeypatch, [])  # CUDA_VISIBLE_DEVICES="" or "-1"
    monkeypatch.setitem(sys.modules, "torch", None)
    assert card.detect_gpu() is None and card.visible_gpus() == []


def test_no_driver_library_reads_nvidia_smis_own_first_card(two_cards, monkeypatch):
    _order(monkeypatch, None)
    assert card.detect_gpu().uuid == "GPU-aaaa"


def _torch_says(monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0, get_device_name=lambda i: "Torch's",
        mem_get_info=lambda i: (2**30, 2**31)))
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_torch_only_where_nvidia_smi_cannot_say(tmp_path, monkeypatch):
    _torch_says(monkeypatch)
    another_machine(monkeypatch, tmp_path)  # none until _smi puts one there, nor the host's own
    assert card.detect_gpu().source == "torch"
    assert card.detect_gpu(quietly=True) is None  # a reader's poll never takes that
    _smi(tmp_path, f"{LAPTOP}\n")
    _order(monkeypatch, ["GPU-a-mig-slice"])  # nvidia-smi has no row for it, torch reads it
    assert card.detect_gpu().source == "torch"
    _order(monkeypatch, ["GPU-aaaa"])
    _smi(tmp_path, "GPU-aaaa, NVIDIA RTX, 8188, [N/A], 7481\n")  # an old driver's field
    assert card.detect_gpu().source == "torch"


class _Cuda:
    """libcuda as far as _cuda_order calls it."""

    def __init__(self, init=0):
        self.init = init

    def cuInit(self, flags):
        return self.init

    def cuDeviceGetCount(self, count):
        count._obj.value = 2
        return 0

    def cuDeviceGet(self, device, i):
        device._obj.value = i
        return 0

    def cuDeviceGetUuid(self, raw, device):
        raw[:] = bytes(range(device.value, device.value + 16))
        return 0


@pytest.mark.parametrize("init, order", [
    (0, ["GPU-00010203-0405-0607-0809-0a0b0c0d0e0f", "GPU-01020304-0506-0708-090a-0b0c0d0e0f10"]),
    (100, []),  # CUDA_ERROR_NO_DEVICE
    (999, None),  # no answer (nvidia_uvm not loaded, say): not the same as no device
])
def test_cuda_order_is_cudas_own(monkeypatch, init, order):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _Cuda(init))
    assert card._cuda_order() == order


class _CudaCaps(_Cuda):
    """libcuda with each device's compute capability: device 0 an sm_89, device 1 an sm_120."""

    def cuDeviceGetAttribute(self, value, attribute, device):
        value._obj.value = {(75, 0): 8, (76, 0): 9, (75, 1): 12, (76, 1): 0}[attribute,
                                                                           device.value]
        return 0


def test_a_card_heres_capability_is_the_drivers_not_its_names(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _CudaCaps())
    assert card._cuda_capabilities() == {
        "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f": (8, 9),
        "GPU-01020304-0506-0708-090a-0b0c0d0e0f10": (12, 0)}
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _Cuda())  # a driver without the call
    assert card._cuda_capabilities() == {}


def test_a_listed_cards_capability_comes_from_its_device(two_cards, monkeypatch):
    _order(monkeypatch, ["GPU-aaaa"])
    monkeypatch.setattr(card, "_cuda_capabilities", lambda: {"GPU-aaaa": (8, 9)})
    assert card.detect_gpu().capability == (8, 9)
    monkeypatch.setattr(card, "_cuda_capabilities", lambda: {})
    assert card.detect_gpu().capability is None  # the plan maps the name then


def _torch_holding(monkeypatch, *, uuid, reserved, allocated, free=2**30):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_initialized=lambda: True, mem_get_info=lambda i: (free, 2**34),
        get_device_properties=lambda i: types.SimpleNamespace(uuid=uuid),
        memory_reserved=lambda i: reserved, memory_allocated=lambda i: allocated))
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_this_processs_unused_cache_counts_as_free_for_the_fit(two_cards, monkeypatch):
    """A trainer deleted without empty_cache leaves its allocator's cache reserved: nvidia-smi
    counts it used, and the next run's allocator reuses it."""
    _order(monkeypatch, ["GPU-aaaa"])
    _torch_holding(monkeypatch, uuid="aaaa", reserved=3 * 2**30, allocated=2**30)
    assert card.detect_gpu().free_gib == 3.0  # 1 GiB free + 2 GiB reserved and unused
    _torch_holding(monkeypatch, uuid="bbbb", reserved=3 * 2**30, allocated=2**30)
    assert card.detect_gpu().free_gib == 7.31  # another card's process: nvidia-smi's figure


def test_a_process_with_no_cuda_context_keeps_the_drivers_free(two_cards, monkeypatch):
    _order(monkeypatch, ["GPU-aaaa"])
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert card.detect_gpu().free_gib == 7.31


def test_cuda_order_without_the_library(monkeypatch):
    def gone(name):
        raise OSError(name)

    monkeypatch.setattr(ctypes, "CDLL", gone)
    assert card._cuda_order() is None


@needs_ml
def test_machines_rows_read_torchs_build_from_its_files(two_cards, monkeypatch):
    import runpy

    monkeypatch.setitem(sys.modules, "torch", None)  # importing it would raise
    monkeypatch.setattr(runpy, "run_path",
                        lambda p: {"__version__": "2.11.0+cu128", "cuda": "12.8"})
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    _order(monkeypatch, ["GPU-aaaa"])
    assert cli_commands_check._torch_facts(quietly=True) == (
        "2.11.0+cu128", "12.8", [("NVIDIA GeForce RTX 4060 Laptop GPU", 7.68)], None)
    monkeypatch.setattr(runpy, "run_path", lambda p: {"__version__": "2.15.0+cpu", "cuda": None})
    assert cli_commands_check._torch_facts(quietly=True)[1:3] == (None,
                                                                   [])  # a CPU build: no card


@needs_ml
def test_machine_says_no_thread_count_it_cannot_know_and_still_refuses_a_bad_one(two_cards,
                                                                                 monkeypatch):
    """torch's own count was 8 on 16 CPUs on the real card's host: not a figure to guess."""
    import runpy

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(runpy, "run_path", lambda p: {"__version__": "2.11", "cuda": "12.8"})
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr("importlib.metadata.version", lambda dist: "1")
    _order(monkeypatch, ["GPU-aaaa"])
    rows = {row[0] for row in cli_commands_check._environment_checks(quietly=True)[0]}
    assert {"torch", "CUDA", "  GPU 0"} <= rows and "CPU threads" not in rows
    monkeypatch.setenv("POROS_NUM_THREADS", "many")
    rows = {row[0]: row[2] for row in cli_commands_check._environment_checks(quietly=True)[0]}
    assert rows["CPU threads"] == "error"


@pytest.mark.cuda_host(reason="a Mac's machine and models read its own chip in their own "
                               "process, never nvidia-smi")
@pytest.mark.parametrize("argv", [["machine"], ["models"]])
def test_a_reader_beside_a_run_never_imports_torch(two_cards, argv):
    """Each would hold 750 MB and a context's 94 MiB of the card, under the run it asks about."""
    (two_cards / "sitecustomize.py").write_text(
        'import poros.core.card\nporos.core.card._cuda_order = lambda: ["GPU-bbbb", "GPU-aaaa"]\n')
    env = {**os.environ, "POROS_HOME": str(two_cards / "home"), "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": f"{two_cards}{os.pathsep}{Path(poros.__file__).parent.parent}"}
    env.pop("POROS_DEV_SIMULATE", None)
    done = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "poros.cli.main", *argv, "--json"],
        env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    out = json.loads(done.stdout.splitlines()[-1])
    gpu = out.get("gpu") or out["plan"]["gpu"]
    assert gpu["name"] == "NVIDIA GeForce RTX 3090" and gpu["source"] == "nvidia-smi"
    assert not any(line.split("|")[-1].strip() == "torch" for line in done.stderr.splitlines())
    if argv == ["machine"]:  # its live rows are that card's, not the first row's
        asked = (two_cards / "asked").read_text().splitlines()
        assert any("driver_version" in row and "-i GPU-bbbb" in row for row in asked), asked
        assert any("compute-apps" in row and "-i GPU-bbbb" in row for row in asked), asked


@pytest.mark.parametrize("pool", [
    "apple-m4-10gpu-16gb",  # a Mac's
    None,  # one memory for the CPU and the GPU, on a machine whose nvidia-smi answers
])
def test_machine_on_one_pool_of_memory_answers_beside_an_nvidia_smi(two_cards, monkeypatch,
                                                                     capsys, pool):
    """nvidia-smi answers here (two cards), and the GPU Poros plans for is one pool of memory:
    machine says what this machine is, in words, and names no card's live row, which would be
    another device's. It once read the board's maker only for a card it draws, then used it for
    any GPU nvidia-smi answered for: an UnboundLocalError, exit 1."""
    from poros.cli import app, contract
    from poros.cli.context import Context
    from poros.cli.schemas import PlanGpu
    from poros.core.errors import EXIT_CODES
    from poros.metal import table

    _order(monkeypatch, ["GPU-bbbb", "GPU-aaaa"])
    one = table.plan_gpu(pool) if pool else PlanGpu(
        name="NVIDIA GB10", total_gib=119.7, free_gib=None, source="spec", unified=True)
    assert card.visible_gpus()  # nvidia-smi does answer
    code = app.dispatch(["machine", "--json"], Context(fixed_gpu=one, fixed_interactive=False))
    contract.reset()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["schema"] == "poros.machine.v1", out
    assert code == (EXIT_CODES["E_MACHINE_NOT_READY"] if out["verdict"] == "not ready" else 0)
    assert out["gpu"]["name"] == one.name and out["now"] is None
    assert out["verdict"] == "ready" or out["reason"], out  # a verdict short of ready says why
