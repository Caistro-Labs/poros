"""What `plan` and the preflight answer where no run measured the card, how plainly they say
it, and the one way every command names a model."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from poros.cli import contract
from poros.cli.context import Context
from poros.cli.schemas import DoctorCheck, Figure, PlanGpu
from poros.cli.verbs import shared
from poros.core.refusal import Refusal
from tests.conftest import checkout_env

HERE = PlanGpu(name="RTX 4090", total_gib=23.52, free_gib=23.1, source="nvidia-smi")
# what a plan's notes say of its method, in the planner's own terms
INTERNALS = ("block-major", "nf4_load_strategy", " x 4 B", "receipt", "scratch draw",
             "OfflineModeIsEnabled", "(path_extra)")


def _poros(*args, cwd=None, env=None):
    base = {k: v for k, v in os.environ.items()
            if k not in ("POROS_OUTPUT", "POROS_AGENT", "POROS_JOB", "POROS_DEV_SIMULATE")}
    return subprocess.run(
        [sys.executable, "-P", "-c", "from poros.cli.main import cli; cli()", *args],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=cwd,
        env=checkout_env({**base, "HF_HUB_OFFLINE": "1", **(env or {})}))


@pytest.fixture
def home(tmp_path, monkeypatch):
    for variable in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB", "POROS_DEV_SIMULATE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    yield tmp_path / "work"
    contract.reset()


def _ctx(where, gpu=None) -> Context:
    return Context(cwd=where, fixed_gpu=gpu, fixed_cached=frozenset(), fixed_interactive=False)


def _text(lines) -> str:
    return "\n".join("".join(seg[0] for seg in line) for line in lines)


def _flat(text: str) -> str:
    return " ".join(text.replace("│", " ").split())


def _plan(peak: float, card: float, source: str = "estimated"):
    """A CUDA plan with no verdict and an allocated peak of ``peak`` GiB on a ``card`` GiB card."""
    p = shared.plan_for("Qwen/Qwen2.5-7B", Context(fixed_gpu=PlanGpu(
        name="NVIDIA GPU", total_gib=card, free_gib=None, source="spec"),
        fixed_cached=frozenset()))
    figure = Figure(value=peak, unit="GiB", source=source, method="m", receipt="r")
    return p.model_copy(update={"fits": None, "host_fits": None, "bound": None,
                                "peak_vram_allocated": figure})


# -- an estimate past the card never refuses -----------------------------------------------------


def test_an_estimate_never_refuses_a_run(home):
    """An estimate past the card warns at most: the exit `train` would give is unchanged."""
    from poros.core.config import PorosConfig
    from poros.preflight import command

    tiny = PlanGpu(name="NVIDIA GPU", total_gib=1.0, free_gib=None, source="spec")
    p = shared.plan_for("Qwen/Qwen2.5-0.5B-Instruct", _ctx(home, tiny))
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct", dataset="synthetic")
    from poros.plan.make import over_card

    assert p.fits is None and over_card(p) and command.plan_exit(cfg, p) is None


# -- the same answer online and offline ----------------------------------------------------------


def test_plan_reads_the_catalogs_twin_with_no_config_json_at_all(home, tmp_path):
    """Offline with an empty cache the catalog says whose shapes a model has: `plan` reads that
    model's measured runs, and does not say unknown."""
    from poros.plan import cells

    assert cells.shape_twin("Qwen/Qwen2.5-0.5B-Instruct", None) == "Qwen/Qwen2.5-0.5B"
    assert cells.shape_twin("Qwen/Qwen2.5-0.5B", None) is None  # its own runs plan it
    assert cells.shape_twin("some-org/some-model", None) is None


# -- the paths, as they are ----------------------------------------------------------------------


def test_nothing_calls_the_default_path_the_least_vram(monkeypatch):
    """efficient is the default and holds less than fast; the sequential loop can peak lower."""
    from poros.cli.commands.train import _starter_config_yaml

    # the starter config off a Mac, which names the paths (a Mac's names none: test_path_settings)
    monkeypatch.setattr(sys, "platform", "linux")
    from poros.cli.train_settings import SETTINGS
    from poros.cli.verbs import train
    from poros.core.config import PorosConfig

    said = [train.__doc__, _starter_config_yaml("m", "d", "o"),
            PorosConfig.model_fields["path"].description,
            *(note for s in SETTINGS for _, note in (s.choices or ())),
            *shared.FIRST_CHOICE.values()]
    assert not [text for text in said if "least VRAM" in text]
    assert "less VRAM than fast" in train.__doc__ and "can peak lower" in train.__doc__
    assert "less VRAM than" in _starter_config_yaml("m", "d", "o")


# -- one way to name a model ---------------------------------------------------------------------


def test_check_and_arch_name_a_model_as_train_does(home):
    """A bare word that no catalog names is no Hub id, for every command alike."""
    with pytest.raises(Refusal) as trained:
        shared.resolve_model(["gpt2"], _ctx(home), verb="train")
    for command in ("check", "arch"):
        proc = _poros(command, "gpt2", "--json", cwd=home)
        error = json.loads(proc.stdout)["error"]
        assert proc.returncode == 2 and error["code"] == trained.value.code == "E_NOT_FOUND"
        assert error["message"].startswith(trained.value.reason), command
    part = _poros("check", "72b", cwd=home)
    assert part.returncode == 2 and "Qwen2.5-72B, Qwen2.5-72B-Instruct" in part.stderr


def test_the_catalog_answers_first_online_or_offline_gated_or_not(home, monkeypatch):
    """A catalog model's family is the catalog's record of it: the Hub is not asked, so a gated
    repository and an unreachable Hub answer as a reachable one does, with a note where a
    download needs a login."""
    from poros.plan import configs
    from poros.preflight import arch

    def never(model, revision=None):
        raise AssertionError(f"the Hub was asked for {model}")

    monkeypatch.setattr(configs, "fetch_config", never)
    assert arch.resolve_model_type("google/gemma-3-1b-pt") == ("gemma3_text", arch.GATED_NOTE)
    assert arch.resolve_model_type("qwen2.5-7b-instruct") == ("qwen2", "")
    proc = _poros("check", "google/gemma-3-1b-pt", "--json", cwd=home)
    out = json.loads(proc.stdout)
    assert proc.returncode == 0 and out["status"] == "validated" and "gated" in out["reason"]
    assert "accept its license" in _poros("check", "google/gemma-3-1b-pt", cwd=home).stdout


def test_arch_answers_for_a_gated_catalog_model_from_the_catalog(home):
    from poros.cli.verbs import arch

    seen = arch.see(arch.ArchAction(model="google/gemma-3-1b-pt"), arch._Seen(gated=True),
                    _ctx(home))
    assert (seen.model_type, seen.status) == ("gemma3_text", "validated")
    assert "it is gated" in seen.cannot_place[0] and seen.next == []
    other = arch.see(arch.ArchAction(model="some-org/gated"), arch._Seen(gated=True), _ctx(home))
    assert (other.model_type, other.status) == ("unknown", "gated")  # no catalog names it


# -- one total for the card ----------------------------------------------------------------------


def test_the_machine_screen_says_one_total_for_the_card():
    """`gpu 23.5 GiB` and `free of 23.5`: the card's own total in both rows, never nvidia-smi's
    larger figure beside it."""
    from poros.cli.verbs.machine import GpuNow, MachineOutput
    from poros.cli.views.machine_screens import _machine_rows, _vram

    r = MachineOutput(verdict="ready", version="0.3.0", gpu=HERE, host_ram_free_gib=40.0,
                      checks=[DoctorCheck(name="torch", value="2.14.1", status="ok")],
                      now=GpuNow(source="nvidia-smi", used_gib=0.4, total_gib=24.0))
    total, used = _vram(r)
    assert (total, used) == (23.52, 0.4)
    text = _text(_machine_rows(r, 100))
    assert "RTX 4090  23.5 GiB" in text and "23.1 GiB free of 23.5" in text
    assert "24.0" not in text


# -- small words ---------------------------------------------------------------------------------


def test_an_empty_data_file_holds_no_rows(home):
    (home / "empty.jsonl").write_text("")
    shown = _poros("data", "empty.jsonl", cwd=home).stdout
    assert "holds     no rows: the file is empty" in shown and "can't read" not in shown


def test_gpu_completes_with_every_card_fit_reads():
    from poros.cli.ui.complete import FLAG_VALUES
    from poros.core import card

    words = [c.text for c in FLAG_VALUES["--gpu"]()]
    assert {"8GB", "4070ti", "4060ti-16gb", "a100", "l4"} <= set(words) and len(words) > 40
    for word in words:  # each is one word `--gpu` reads back
        card.named_card(word)
        assert shared.gpu_name(word)
    assert card.named_card("4060ti-16gb")[1].total_gib == 16.0


def test_a_cuda_cards_plan_takes_the_presets_a_cuda_run_trains_at():
    """`plan --gpu` plans another card at that card's run's settings (CUDA_MEASURED). Off a Mac
    they are this machine's own, unchanged: MEASURED holds the same."""
    from poros import recipes

    assert recipes.CUDA_MEASURED == {"quantization": "nf4", "block_size": 4,
                                     "per_device_batch_size": 1, "gradient_accumulation_steps": 8,
                                     "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.0}
    if sys.platform != "darwin":
        assert recipes.MEASURED == recipes.CUDA_MEASURED
