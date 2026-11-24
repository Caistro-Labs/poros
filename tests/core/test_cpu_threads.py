"""torch thread sizing from the container's CPU quota (poros.core.env).

Every test points the cgroup root at a temp dir and fixes the affinity mask,
so none depends on the host's real cgroups or core count.
"""

from __future__ import annotations

import os

import pytest

from poros.core import env
from poros.core.errors import PorosConfigError


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A 96-CPU host with no cgroup files and no thread env vars."""
    _write(tmp_path / "self_cgroup", "1:cpu,cpuacct:/\n0::/\n")
    monkeypatch.setattr(env, "_CGROUP", tmp_path)
    monkeypatch.setattr(env, "_PROC_CGROUP", tmp_path / "self_cgroup")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)), raising=False)
    monkeypatch.delenv("POROS_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    return tmp_path


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v2_quota_floors(host):
    _write(host / "cpu.max", "765000 100000\n")  # a cloud container: 7.65 cores
    assert env.cpu_threads(96) == (7, "cgroup CPU quota 7.65")


def test_quota_on_a_nested_cgroup_counts(host):
    # docker --cgroupns=host or a systemd slice: the root is unlimited and
    # the quota sits on the process's own cgroup below it.
    _write(host / "self_cgroup", "0::/kubepods/pod1\n")
    _write(host / "cpu.max", "max 100000\n")
    _write(host / "kubepods/pod1/cpu.max", "200000 100000\n")
    assert env.cpu_threads(96) == (2, "cgroup CPU quota 2")


def test_cgroup_v1_quota_floors_to_at_least_one(host):
    _write(host / "cpu/cpu.cfs_quota_us", "50000\n")
    _write(host / "cpu/cpu.cfs_period_us", "100000\n")
    assert env.cpu_threads(96)[0] == 1


@pytest.mark.parametrize("v2", ["max 100000\n", None])
def test_no_quota_caps_at_the_affinity_mask(host, monkeypatch, v2):
    if v2:
        _write(host / "cpu.max", v2)
    else:
        _write(host / "cpu/cpu.cfs_quota_us", "-1\n")
        _write(host / "cpu/cpu.cfs_period_us", "100000\n")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert env.cpu_threads(96) == (4, "4 usable CPUs")


def test_never_raises_torch_default(host):
    _write(host / "cpu.max", "1600000 100000\n")
    assert env.cpu_threads(6)[0] == 6
    assert env.cpu_threads(96)[0] == 16


def test_explicit_omp_num_threads_is_left_alone(host, monkeypatch):
    _write(host / "cpu.max", "200000 100000\n")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    assert env.cpu_threads(12) == (12, "OMP_NUM_THREADS")


def test_poros_num_threads_overrides_everything(host, monkeypatch):
    _write(host / "cpu.max", "200000 100000\n")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    monkeypatch.setenv("POROS_NUM_THREADS", "5")
    assert env.cpu_threads(12) == (5, "POROS_NUM_THREADS")


@pytest.mark.parametrize("value", ["0", "-2", "four"])
def test_bad_poros_num_threads_refuses(host, monkeypatch, value):
    monkeypatch.setenv("POROS_NUM_THREADS", value)
    with pytest.raises(PorosConfigError, match="POROS_NUM_THREADS"):
        env.cpu_threads(8)


@pytest.fixture
def torch_threads():
    """torch, with its thread count restored after the test."""
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    yield torch
    torch.set_num_threads(before)


def test_configure_sets_and_returns_torch_threads(host, torch_threads):
    _write(host / "cpu.max", "100000 100000\n")
    assert env.configure_torch_threads() == 1
    assert torch_threads.get_num_threads() == 1


def test_machine_reports_cpu_threads(host, monkeypatch, torch_threads):
    from poros.cli.commands import check
    from poros.cli.commands.check import _environment_checks
    from tests.conftest import as_linux

    as_linux(monkeypatch, check)  # a CUDA host's rows: a Mac's machine has its own, no thread row
    monkeypatch.setenv("POROS_NUM_THREADS", "3")
    rows = {name: (value, status) for name, value, status in _environment_checks()[0]}
    assert rows["CPU threads"] == ("3 (POROS_NUM_THREADS)", "ok")

    monkeypatch.setenv("POROS_NUM_THREADS", "zero")
    rows = {name: (value, status) for name, value, status in _environment_checks()[0]}
    assert rows["CPU threads"][1] == "error"
