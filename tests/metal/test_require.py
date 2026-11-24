"""``poros.metal.require`` and the Mac validator, with the machine faked on Linux."""

from __future__ import annotations

import importlib.metadata
import platform

import pytest

import poros.metal as metal
import poros.metal.selfcheck  # noqa: F401 - require reaches it for a Mac no run measured
import poros.metal.store  # noqa: F401 - require reaches it; the disk is not under test here
from poros.core.config import PorosConfig
from poros.core.errors import (
    EXIT_CODES,
    PorosConfigError,
    PorosGuaranteeUnavailableError,
    PorosStackUnsupportedError,
)

M4 = "apple-m4-10gpu-16gb"


def _config(**changes):
    # Past parse, so the value under test is what the validator sees even where a
    # parse validator would refuse it first.
    return PorosConfig().model_copy(update=changes)


@pytest.fixture
def this_mac(monkeypatch, tmp_path):
    """A machine that passes every check; a test then breaks one. Returns the classes it was asked
    to check itself on."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("27.2", ("", "", ""), "arm64"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: metal.PINS[name])
    monkeypatch.setattr(metal, "device_class", lambda: M4)
    monkeypatch.setattr(metal.store, "settle", lambda config: None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    for name in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(metal.DEV_MEMORY_ENV, raising=False)
    checked = []
    monkeypatch.setattr(metal.selfcheck, "ensure", checked.append)
    return checked


# ---------------------------------------------------------------------------
# Every field classified.
# ---------------------------------------------------------------------------

# A value off default for each refused field. A new refused field needs one here.
OFF_DEFAULT = {
    "torch_dtype": "float32",
    "block_size": 4,
    "freeze_embeddings": False,
    "freeze_lm_head": False,
    "preserve_rng_state": False,
    "blockwise_arch_check": "warn",
    "startup_parity_selfcheck": "always",
    "deterministic": False,
    "lora_target_modules": ["q_proj"],
    "lora_bias": "all",
    "per_device_batch_size": 2,
    "nf4_load_strategy": "gpu",
    "enable_prefetch": False,
    "store": "my-plugin",
    "store_options": {"a": 1},
    "store_package": "my-store-package",
    "offload_suffix_inputs_to_cpu": True,
    "adapter_backend": "native",
    "adapter_runtime": "qlora8",
    "bf16_forward_math_mode": "no_grad",
    "path": "fast",
    "packed_block_transfers": False,
    "bf16_cuda_graph_blocks": True,
    "debug_memory": True,
    "trust_remote_code": True,
    "attn_implementation": "sdpa",
}


def test_receipt_defaults_are_all_refused_off_default():
    from poros.plan.receipts import _RECEIPT_DEFAULTS

    assert not set(_RECEIPT_DEFAULTS) & (metal.ALLOWED | metal._FROM_TABLE)


@pytest.mark.parametrize("field", sorted(OFF_DEFAULT))
def test_a_refused_field_off_default_is_e_config(field):
    with pytest.raises(PorosConfigError) as exc:
        metal.validate(_config(**{field: OFF_DEFAULT[field]}))
    assert field in str(exc.value)
    assert exc.value.code == "E_CONFIG" and EXIT_CODES["E_CONFIG"] == 2


def test_every_refused_field_is_named_at_once():
    with pytest.raises(PorosConfigError) as exc:
        metal.validate(_config(block_size=4, lora_bias="all"))
    assert "block_size=4" in str(exc.value) and "lora_bias='all'" in str(exc.value)


@pytest.mark.parametrize("quantization", ["bf16", "fp16", "bf16_bitwise", "int8"])
def test_other_precisions_are_e_config(quantization):
    with pytest.raises(PorosConfigError) as exc:  # the config's own refusal
        PorosConfig(quantization=quantization)
    assert EXIT_CODES[exc.value.code] == 2
    with pytest.raises(PorosConfigError) as exc:  # and past parse, the Mac validator's
        metal.validate(_config(quantization=quantization))
    assert "quantization" in str(exc.value)


@pytest.mark.parametrize("changes", [
    {},
    {"block_size": 1},
    {"attn_implementation": "eager"},
    {"adapter_backend": "peft"},
    {"store": "ssd", "store_path": "/a/volume"},
    {"lora_dropout": 0.0, "lora_rank": 8, "learning_rate": 1e-4, "seed": 1},
    {"max_seq_length": 1024, "gradient_accumulation_steps": 2, "save_steps": 25},
])
def test_what_a_mac_honours_passes(changes):
    metal.validate(_config(**changes))


# ---------------------------------------------------------------------------
# The machine.
# ---------------------------------------------------------------------------


def test_this_class_passes_and_sets_the_allocator(this_mac, monkeypatch):
    import os

    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    metal.require(PorosConfig())
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == metal.HIGH_WATERMARK
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == metal.LOW_WATERMARK
    assert "PYTORCH_MPS_FAST_MATH" not in os.environ


@pytest.mark.parametrize("text,version", [
    ("27.0", (27, 0, 0)), ("27.0.0", (27, 0, 0)), ("27.0.1", (27, 0, 1)), ("26.4", (26, 4, 0)),
    ("", (0, 0, 0)),
])
def test_macos_version(text, version):
    assert metal.macos_version(text) == version


@pytest.mark.parametrize("version", ["27.0", "27.0.0"])
def test_macos_27_0_0_is_refused_however_it_is_written(this_mac, monkeypatch, version):
    monkeypatch.setattr(platform, "mac_ver", lambda: (version, ("", "", ""), "arm64"))
    with pytest.raises(PorosStackUnsupportedError, match="stride-0") as exc:
        metal.require(PorosConfig())
    assert EXIT_CODES[exc.value.code] == 5


@pytest.mark.parametrize("version", ["13.7", "12.0"])
def test_below_14_is_refused_in_what_it_needs(this_mac, monkeypatch, version):
    monkeypatch.setattr(platform, "mac_ver", lambda: (version, ("", "", ""), "arm64"))
    with pytest.raises(PorosStackUnsupportedError, match="below 14") as exc:
        metal.require(PorosConfig())
    assert "macOS 14 or later" in str(exc.value)


@pytest.mark.parametrize("version", ["14.0", "14.7", "15.0", "26.4", "27.0.1", "27.1", "27.2"])
def test_other_builds_run(this_mac, monkeypatch, version):
    monkeypatch.setattr(platform, "mac_ver", lambda: (version, ("", "", ""), "arm64"))
    metal.require(PorosConfig())


def test_no_mps_device_says_so_and_never_asks_for_an_update(this_mac, monkeypatch):
    """macOS 14 has MPS with torch 2.15.0; a Mac with none is a VM, whatever its macOS."""
    import torch

    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.6", ("", "", ""), "arm64"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(PorosStackUnsupportedError) as exc:
        metal.require(PorosConfig())
    assert "no MPS device" in str(exc.value) and "15" not in str(exc.value)
    assert "virtual machine" in exc.value.info.remediation


def test_intel_is_refused(this_mac, monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    with pytest.raises(PorosStackUnsupportedError, match="arm64"):
        metal.require(PorosConfig())


def test_a_pin_off_by_a_patch_is_refused(this_mac, monkeypatch):
    pins = {**metal.PINS, "peft": "0.20.1"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: pins[name])
    with pytest.raises(PorosStackUnsupportedError, match="peft is 0.20.1"):
        metal.require(PorosConfig())


def test_a_missing_package_is_refused(this_mac, monkeypatch):
    def version(name):
        if name == "datasets":
            raise importlib.metadata.PackageNotFoundError(name)
        return metal.PINS[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    with pytest.raises(PorosStackUnsupportedError, match="datasets is not installed"):
        metal.require(PorosConfig())


def test_mps_absent_at_preflight_is_refused(this_mac, monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(PorosStackUnsupportedError, match="no MPS"):
        metal.require(PorosConfig())


def test_require_applies_the_validator(this_mac):
    with pytest.raises(PorosConfigError):
        metal.require(PorosConfig(block_size=4))


@pytest.mark.parametrize("other", [
    "apple-m1-8gpu-8gb", "apple-m1-ultra-64gpu-128gb", "apple-m2-max-30gpu-32gb",
    "apple-m3-10gpu-16gb", "apple-m4-pro-16gpu-24gb", "apple-m4-10gpu-24gb", "apple-m5-10gpu-16gb",
    "apple-m4-0gpu-16gb",  # cores are a label: one that cannot be read stops nothing
])
def test_every_apple_silicon_mac_trains_once_it_checks_itself(this_mac, monkeypatch, other):
    monkeypatch.setattr(metal, "device_class", lambda: other)
    metal.require(PorosConfig())
    assert this_mac == [other]


def test_the_measured_class_runs_no_check(this_mac):
    metal.require(PorosConfig())
    assert this_mac == []


@pytest.mark.parametrize("other", ["unknown-0gpu-0gb", "apple-m4-10gpu-0gb", "unknown-10gpu-16gb"])
def test_a_mac_whose_chip_or_memory_cannot_be_read_is_refused(this_mac, monkeypatch, other):
    monkeypatch.setattr(metal, "device_class", lambda: other)
    with pytest.raises(PorosGuaranteeUnavailableError, match="could not read") as exc:
        metal.require(PorosConfig())
    assert "--allow-unvalidated" not in str(exc.value) and this_mac == []


@pytest.fixture
def clean_class_cache():
    yield
    metal.device_class.cache_clear()


def _fake_tools(
    monkeypatch, *, chip="Apple M4", mem="17179869184", ioreg='  "gpu-core-count" = 10\n'
):
    outputs = {"machdep.cpu.brand_string": chip, "hw.memsize": mem}

    def run(*cmd):
        return ioreg.strip() if "ioreg" in cmd[0] else outputs[cmd[-1]]

    monkeypatch.setattr(metal, "run_tool", run)
    metal.device_class.cache_clear()


def test_device_class_from_the_machine(monkeypatch, clean_class_cache):
    _fake_tools(monkeypatch)
    assert metal.device_class() == M4
    _fake_tools(monkeypatch, chip="Apple M4 Pro", mem=str(24 << 30), ioreg='"gpu-core-count" = 16')
    assert metal.device_class() == "apple-m4-pro-16gpu-24gb"


def test_a_part_that_cannot_be_read_fails_closed(monkeypatch, clean_class_cache):
    _fake_tools(monkeypatch, ioreg="")
    assert metal.device_class() not in metal.DEVICE_CLASSES
    _fake_tools(monkeypatch, chip="", mem="")
    assert metal.device_class() not in metal.DEVICE_CLASSES
    with pytest.raises(PorosGuaranteeUnavailableError, match="could not read"):
        metal.require_class(metal.device_class())


@pytest.mark.parametrize("chip, want", [
    ("Apple M1", "apple-m1-8gpu-16gb"),
    ("Apple M2 Ultra", "apple-m2-ultra-8gpu-16gb"),
    ("Apple M1 (Virtual)", "apple-m1-virtual-8gpu-16gb"),  # what reaches a file: letters, dashes
    ("Apple M5 Max\n../../x", "apple-m5-max-x-8gpu-16gb"),
])
def test_a_class_keeps_to_letters_digits_and_dashes(monkeypatch, clean_class_cache, chip, want):
    _fake_tools(monkeypatch, chip=chip, ioreg='"gpu-core-count" = 8')
    assert metal.device_class() == want


# ---------------------------------------------------------------------------
# The developer's budget: a test tool that plans and guards as if this Mac had less memory.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("said, gb", [
    ("8", 8), ("12", 12), ("15", 15), ("4", 4), ("16", None),
    ("3", None), ("17", None), ("8.5", None), (" 8", None), ("", None), ("x", None), ("-8", None),
])
def test_the_budget_is_whole_gb_from_4_to_less_than_this_macs_own(monkeypatch, clean_class_cache,
                                                                  said, gb):
    _fake_tools(monkeypatch)
    monkeypatch.setenv(metal.DEV_MEMORY_ENV, said)
    assert metal.dev_memory_gb() == gb
    assert metal.memory_bytes() == (gb or 16) * 2**30
    assert metal.device_class() == f"apple-m4-10gpu-{gb or 16}gb"


def test_a_budget_never_makes_a_mac_the_measured_one(monkeypatch, clean_class_cache):
    """A 24 GB M4 told 16 GB would be the reference by name, its figures read as measured."""
    _fake_tools(monkeypatch, mem=str(24 * 2**30))
    monkeypatch.setenv(metal.DEV_MEMORY_ENV, "16")
    assert metal.dev_memory_gb() is None
    assert metal.device_class() == "apple-m4-10gpu-24gb" and not metal.measured(
        metal.device_class())
    monkeypatch.setenv(metal.DEV_MEMORY_ENV, "12")
    metal.device_class.cache_clear()
    assert metal.dev_memory_gb() == 12 and metal.device_class() == "apple-m4-10gpu-12gb"


def test_with_no_budget_the_class_and_watermarks_are_as_measured(monkeypatch, clean_class_cache):
    _fake_tools(monkeypatch)
    monkeypatch.delenv(metal.DEV_MEMORY_ENV, raising=False)
    assert metal.device_class() == M4 and metal.measured(M4)
    assert metal.watermarks() == (metal.HIGH_WATERMARK, metal.LOW_WATERMARK) == ("1.05", "0.95")
    monkeypatch.setenv(metal.DEV_MEMORY_ENV, "16")  # the Mac's own memory: the measured class
    metal.device_class.cache_clear()
    assert metal.watermarks() == ("1.05", "0.95")


def test_under_a_budget_the_allocator_stops_at_that_macs_working_set(monkeypatch,
                                                                    clean_class_cache):
    from poros.metal import machine, table

    _fake_tools(monkeypatch)
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(machine, "gpu_working_set", lambda: table.WORKING_SET_BYTES[M4])
    monkeypatch.setenv(metal.DEV_MEMORY_ENV, "8")
    here = metal.device_class()
    assert here == "apple-m4-10gpu-8gb" and not metal.measured(here)
    budget, how = table.working_set(here)
    assert how == "budget" and budget == 8 * 2**30 * 2 // 3  # what an 8 GB Mac is given
    high, low = metal.watermarks()
    share = budget / table.WORKING_SET_BYTES[M4]
    assert float(high) == pytest.approx(1.05 * share, abs=1e-4)
    assert float(low) == pytest.approx(0.95 * share, abs=1e-4) and float(low) < float(high)
