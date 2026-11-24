"""``poros.metal.selfcheck`` off a Mac: its record, its gate, and each check's plumbing on the CPU
with the Mac path's CPU double. Whether a GPU passes it is for a Mac to say."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from poros import metal  # noqa: E402
from poros.core.errors import EXIT_CODES, PorosStackUnsupportedError  # noqa: E402
from poros.metal import nf4, selfcheck  # noqa: E402
from poros.moe import quant  # noqa: E402

HERE = "apple-m1-8gpu-8gb"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(metal, "macos_build", lambda: "26B123")
    monkeypatch.setattr(metal, "installed", lambda name: metal.PINS.get(name))
    return tmp_path / "home"


def _record(passed: bool, **more) -> dict:
    return {"schema": selfcheck.SCHEMA, "key": selfcheck.key(HERE), "when": "2026-11-08T00:00:00",
            "pass": passed, "failed": "" if passed else "the NF4 kernel differs", "checks": {},
            **more}


# -- The record and the gate ------------------------------------------------------------------


def test_a_first_run_checks_once_and_a_pass_stands(monkeypatch):
    calls = []
    monkeypatch.setattr(selfcheck, "run", lambda here: calls.append(here) or _record(True))
    selfcheck.ensure(HERE)
    assert calls == [HERE]
    selfcheck.path().parent.mkdir(parents=True, exist_ok=True)
    selfcheck.path().write_text(json.dumps(_record(True)))
    selfcheck.ensure(HERE)
    assert calls == [HERE]  # recorded: not again


def test_a_failure_refuses_says_what_to_send_and_is_checked_again(monkeypatch):
    calls = []
    monkeypatch.setattr(selfcheck, "run", lambda here: calls.append(here) or _record(False))
    for _ in range(2):
        with pytest.raises(PorosStackUnsupportedError) as exc:
            selfcheck.ensure(HERE)
        assert "the NF4 kernel differs" in str(exc.value)
        assert selfcheck.shown() in exc.value.info.remediation  # as said: ~ for the home folder
        assert "Poros team" in exc.value.info.remediation
        assert EXIT_CODES[exc.value.code] == 5
    assert calls == [HERE, HERE]


def test_what_it_says_to_send_names_no_user(monkeypatch, tmp_path):
    monkeypatch.setattr(selfcheck.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("POROS_HOME", str(tmp_path / ".poros"))
    monkeypatch.setattr(selfcheck, "run", lambda here: _record(False))
    with pytest.raises(PorosStackUnsupportedError) as exc:
        selfcheck.ensure(HERE)
    selfcheck.path().parent.mkdir(parents=True, exist_ok=True)
    selfcheck.path().write_text(json.dumps(_record(False)))
    for said in (exc.value.info.remediation, selfcheck.check_row(HERE)[1]):
        assert "~/.poros/machine/self-check.json" in said and str(tmp_path) not in said


def test_a_pass_the_check_recorded_for_another_mac_does_not_stand_for_this_one(monkeypatch):
    other = {**_record(True), "key": {**selfcheck.key(HERE), "device_class": "apple-m4-10gpu-8gb"}}
    monkeypatch.setattr(selfcheck, "run", lambda here: other)
    with pytest.raises(PorosStackUnsupportedError) as exc:
        selfcheck.ensure(HERE)
    assert "another Mac" in str(exc.value)


def test_the_checks_own_store_skips_the_disk_rule_and_nothing_else_does(monkeypatch, tmp_path):
    """A startup disk over 90% full refuses a model's store, not the check's few MB."""
    from poros.core.config import PorosConfig
    from poros.metal import store, table

    work = tmp_path / "check"
    monkeypatch.setattr(metal, "device_class", lambda: HERE)
    monkeypatch.setattr(table, "fill", lambda cfg, here: object())
    settled, used = [], []
    monkeypatch.setattr(store, "settle", lambda cfg: settled.append(cfg.store_path))
    monkeypatch.setattr(store, "use", lambda path: used.append(str(path)))
    cfg = PorosConfig(model_name_or_path=str(work / "model"), store="ssd",
                      store_path=str(work / "store"))
    metal.settle(cfg)  # not in the check: the rule applies
    monkeypatch.setattr(selfcheck, "_inside", True)
    monkeypatch.setattr(selfcheck, "_store", str(work / "store"))
    metal.settle(cfg)  # the check's own store: only pointed at
    other = cfg.model_copy(update={"store_path": str(tmp_path / "elsewhere")})
    metal.settle(other)  # any other store, even in the check's process: the rule
    assert settled == [str(work / "store"), str(tmp_path / "elsewhere")]
    assert used == [str(work / "store")]


@pytest.mark.parametrize("moved", ["device_class", "macos_build", "torch", "shader_sha256",
                                   "poros"])
def test_a_pass_stands_only_for_what_it_checked(moved):
    record = _record(True)
    record["key"][moved] = "something else"
    selfcheck.path().parent.mkdir(parents=True, exist_ok=True)
    selfcheck.path().write_text(json.dumps(record))
    assert selfcheck.recorded(HERE) is None
    assert selfcheck.check_row(HERE)[2] == "info"


def test_its_row_in_machine(home):
    assert selfcheck.check_row(HERE) == (
        "Self-check", "not run yet: the first `train` here runs it, in seconds", "info")
    selfcheck.path().parent.mkdir(parents=True, exist_ok=True)
    selfcheck.path().write_text(json.dumps(_record(True)))
    name, said, status = selfcheck.check_row(HERE)
    assert status == "ok" and said.startswith("passed 2026-11-08")
    selfcheck.path().write_text(json.dumps(_record(False)))
    name, said, status = selfcheck.check_row(HERE)
    assert status == "error" and "the NF4 kernel differs" in said and selfcheck.shown() in said


def test_inside_the_check_its_own_trainer_does_not_ask_again(monkeypatch):
    monkeypatch.setattr(selfcheck, "run", lambda here: pytest.fail("checked from inside"))
    monkeypatch.setattr(selfcheck, "_inside", True)
    selfcheck.ensure(HERE)


def test_a_check_process_that_dies_is_a_failure_with_no_path_in_it(monkeypatch, home):
    monkeypatch.setattr(selfcheck, "CHILD", [sys.executable, "-c", "import sys; sys.exit(3)"])
    record = selfcheck.run(HERE)
    assert record["pass"] is False and "exit 3" in record["failed"]
    assert json.loads(selfcheck.path().read_text()) == record
    assert str(Path.home()) not in selfcheck.path().read_text()


def test_what_the_check_process_writes_is_the_record(monkeypatch):
    write = ("import json, sys; out = sys.argv[sys.argv.index('--out') + 1]; "
             f"json.dump({_record(True)!r}, open(out, 'w'))")
    monkeypatch.setattr(selfcheck, "CHILD", [sys.executable, "-c", write])
    assert selfcheck.run(HERE)["pass"] is True
    assert selfcheck.recorded(HERE)["pass"] is True


def test_the_working_set_a_check_read_is_what_planning_falls_back_to():
    selfcheck.path().parent.mkdir(parents=True, exist_ok=True)
    record = _record(True, checks={"working_set": {"torch": 5_726_623_061, "pass": True}})
    selfcheck.path().write_text(json.dumps(record))
    assert selfcheck.recorded_working_set(HERE) == 5_726_623_061
    assert selfcheck.recorded_working_set("apple-m2-10gpu-8gb") is None


# -- The checks, on the CPU -------------------------------------------------------------------


def _cpu_decode(out, slab, start, n):
    """The kernel's CPU route, which the check compares against: what a correct GPU gives."""
    blob = slab[start : start + quant.quantized_nbytes((n,))]
    out.copy_(quant.dequantize(blob, (n,), torch.bfloat16))


def test_the_nf4_check_passes_a_decode_equal_to_the_cpu(monkeypatch):
    monkeypatch.setattr(nf4, "_kernel_decode", _cpu_decode)
    found = selfcheck.check_nf4("cpu")
    assert found["pass"] and found["bit_differences"] == 0 and found["elements"] > 4_000_000


def test_the_nf4_check_fails_one_bit_off(monkeypatch):
    def off_by_a_bit(out, slab, start, n):
        _cpu_decode(out, slab, start, n)
        out.view(torch.int16)[n // 2] ^= 1

    monkeypatch.setattr(nf4, "_kernel_decode", off_by_a_bit)
    found = selfcheck.check_nf4("cpu")
    assert not found["pass"] and found["bit_differences"] == 28  # 7 inputs, 2 offsets, twice
    assert "the NF4 kernel differs from the CPU" in found["says"] and found["first"]


def test_the_contracted_block_is_one_a_fused_multiply_add_would_change():
    from tests.metal.test_nf4 import _sections, kernel_port

    blob, n = selfcheck._contracted_block(), 64
    want = quant.dequantize(blob, (n,), torch.bfloat16)
    fused = torch.empty(n, dtype=torch.bfloat16)
    kernel_port(fused, *_sections(blob, n), quant.dynamic_codebook(), quant.nf4_codebook(), n,
                threads=[1], fma=True)
    assert not torch.equal(fused.view(torch.int16), want.view(torch.int16))


def test_the_matmul_check_passes_correct_products_and_fails_wrong_ones(monkeypatch):
    found = selfcheck.check_matmul("cpu")
    assert found["pass"] and found["repeat_equal"] and found["cases"] == 12
    assert found["relative_error"]["bf16"] < selfcheck.BF16_TOLERANCE / 4
    real = torch.nn.functional.linear
    monkeypatch.setattr(torch.nn.functional, "linear", lambda x, w: real(x, w) * 1.02)
    found = selfcheck.check_matmul("cpu")
    assert not found["pass"] and "matrix products on MPS are off" in found["says"]


def test_the_repeat_check_trains_the_tiny_model_from_ram_and_ssd_the_same(tmp_path):
    from poros.metal.tiny import CpuDouble

    found = selfcheck.check_repeat(tmp_path, CpuDouble)
    assert found["pass"], found
    assert found["ram"] == found["ssd"] and len(found["ram"]["losses"]) == selfcheck.STEPS


def test_the_repeat_check_fails_when_the_ssd_reads_other_bytes(tmp_path, monkeypatch):
    from poros.metal import source
    from poros.metal.tiny import CpuDouble

    real = source.FileLayers.acquire

    def acquire(self, layer):
        slab = real(self, layer)
        slab[:8] = 0x5A  # the first packed nibbles of its first matrix
        return slab

    monkeypatch.setattr(source.FileLayers, "acquire", acquire)
    found = selfcheck.check_repeat(tmp_path, CpuDouble)
    assert not found["pass"] and "differ in" in found["says"]


def test_off_a_mac_the_check_records_that_mps_is_missing_and_names_no_path(tmp_path, monkeypatch):
    monkeypatch.setattr(selfcheck, "_inside", False)  # main sets it: put it back after
    # a host with no MPS device, on any host (a Mac's real one is faked away, not skipped)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(platform, "mac_ver", lambda: ("27.2", ("", "", ""), "arm64"))
    monkeypatch.setattr(metal, "device_class", lambda: HERE)
    out = tmp_path / "out.json"
    assert selfcheck.main(["--out", str(out), "--work", str(tmp_path)]) == 1
    record = json.loads(out.read_text())
    assert record["pass"] is False and record["failed"] == "torch reports no MPS device"
    assert record["key"]["device_class"] == HERE and str(tmp_path) not in out.read_text()


def test_an_error_reaches_the_record_as_one_line_with_no_path():
    said = ("[Errno 2] No such file: '/Users/someone/x/model.sss' in /Volumes/Disk/poros/hf\n"
            "Traceback ...")
    clean = selfcheck._clean(said, "/var/folders/ab/poros-check-1")
    assert clean == "[Errno 2] No such file: '<path>' in <path>"
    assert selfcheck._clean("bad at /var/folders/ab/poros-check-1/store", "/var/folders/ab/"
                            "poros-check-1") == "bad at <check>/store"
