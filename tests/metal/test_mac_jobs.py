"""Jobs on a Mac: its own identity, no CUDA in the runner's environment, one run per machine."""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys

import pytest

from poros import jobs, metal
from poros.jobs import runner

UUID = '"IOPlatformUUID" = "11111111-2222-3333-4444-555555555555"'


@pytest.fixture
def mac(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(metal, "run_tool", lambda *cmd: UUID)
    return tmp_path


def test_a_mac_is_named_by_a_hash_with_no_torch_and_no_hardware_id(mac):
    identity = jobs.gpu_uuid()
    assert identity.startswith("apple-") and len(identity) == len("apple-") + 16
    assert "11111111" not in identity
    assert identity == jobs.gpu_uuid()


def test_one_run_per_machine_and_a_job_queues_on_the_same_lock(mac):
    held = jobs.hold_machine()
    assert held is not None
    assert jobs.hold_machine() is None
    with open(mac / f"gpu-{jobs.gpu_uuid()}.lock", "a") as queued:
        with pytest.raises(BlockingIOError):
            fcntl.flock(queued, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held.close()
    again = jobs.hold_machine()
    assert again is not None
    again.close()


def test_the_runner_pins_no_cuda_device_on_a_mac(mac, monkeypatch):
    job = mac / "jobs" / "1"
    (job / "artifacts").mkdir(parents=True)
    (job / "spec.json").write_text(json.dumps({"force": False}))
    jobs.write_status(
        job,
        job_id="1",
        state="queued",
        gpu=jobs.gpu_uuid(),
        queued_at=jobs.now(),
        config=str(job / "config.yaml"),
        output_dir=str(job / "artifacts"),
        attempt=1,
        log="log",
    )
    seen = {}

    def run(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "kept-as-it-was")
    assert runner.main(job) == 0
    assert seen["CUDA_VISIBLE_DEVICES"] == "kept-as-it-was"
    assert json.loads((job / "status.json").read_text())["state"] == "succeeded"
