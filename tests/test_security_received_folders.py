"""A run or job folder someone else gave the user, placed under POROS_HOME: reading it never
writes through its links, its config never grants code, an export of a run never ships its
private record, and eval's measured-peak exception never goes below the frozen layers' floor."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from poros import runs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POROS_JOB", raising=False)
    runs.make_home()
    return tmp_path


def _target(tmp_path) -> Path:
    t = tmp_path / "victim.txt"
    t.write_text("mine\n")
    return t


# ── no write follows a planted link ─────────────────────────────────────────────────────


def test_listing_runs_never_writes_through_a_planted_status_link(home):
    """A run folder that says running with a dead pid is marked failed when runs are listed: the
    write lands in its own status.json, never through status.json.tmp, nor through status.json
    itself where that is the link."""
    victim = _target(home)
    for name, link in (("1120-0000-aaaa", "status.json.tmp"), ("1120-0000-bbbb", "status.json")):
        d = runs.create(name, {"model": "m", "data": "synthetic", "steps": 1})
        status = json.loads((d / "status.json").read_text())
        status.update(state="running", pid=2**22 + 4321)
        (d / "status.json").write_text(json.dumps(status))
        if link == "status.json":
            raw = (d / "status.json").read_text()
            (d / "status.json").unlink()
            victim.write_text(raw)  # the link's target reads as a status, as a planter would
        (d / link).symlink_to(victim)
        before = victim.read_text()
        assert {r.id: r for r in runs.all_runs()}[name].state == "failed"
        assert victim.read_text() == before, name
        assert not (d / "status.json").is_symlink()
        victim.write_text("mine\n")


def test_an_atomic_write_writes_the_same_bytes_and_leaves_no_temp_file(home):
    path = home / "x.json"
    runs.write_json(path, {"b": 1, "a": [2]})
    assert path.read_bytes() == (json.dumps({"b": 1, "a": [2]}, indent=2, sort_keys=True)
                                 + "\n").encode()
    assert sorted(p.name for p in home.iterdir() if p.name.startswith(".x.json")) == []


def test_a_jobs_status_write_never_follows_its_temp_name(home):
    from poros import jobs

    victim = _target(home)
    d = home / "home" / "jobs" / "j1"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": "queued"}))
    (d / "status.tmp").symlink_to(victim)
    jobs.write_status(d, state="running")
    assert victim.read_text() == "mine\n"
    assert json.loads((d / "status.json").read_text()) == {"state": "running"}


def test_a_jobs_launch_log_and_a_continue_log_refuse_a_planted_link(home, monkeypatch):
    import subprocess

    from poros import jobs
    from poros.cli.verbs import train_run

    victim = _target(home)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("it started"))
    d = home / "home" / "jobs" / "j2"
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"cwd": str(home)}))
    (d / "log.txt").symlink_to(victim)
    with pytest.raises(OSError):
        jobs.launch(d)
    r = runs.create("1120-0000-cccc", {"model": "m", "data": "synthetic", "steps": 1})
    (r / "continue.log").symlink_to(victim)
    run = {x.id: x for x in runs.all_runs()}["1120-0000-cccc"]
    with pytest.raises(OSError):
        train_run.continue_detached(run)
    assert victim.read_text() == "mine\n"


def test_a_log_opens_for_append_and_for_a_new_start(home):
    log = home / "a.log"
    with runs.open_log(log) as f:
        f.write("one\n")
    with runs.open_log(log) as f:
        f.write("two\n")
    assert log.read_text() == "one\ntwo\n"
    with runs.open_log(log, "w") as f:
        f.write("new\n")
    assert log.read_text() == "new\n"


# ── a received job folder's config grants nothing ──────────────────────────────────────


def _job_config(home, raw: dict, name: str = "j3") -> Path:
    d = home / "home" / "jobs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(json.dumps(raw))
    return d / "config.yaml"


def test_a_job_folders_config_grants_code_only_to_the_jobs_own_runner(home, monkeypatch):
    import typer

    from poros.cli.root import _no_grants

    raw = {"loop": "evil.py", "trust_remote_code": True}
    config = _job_config(home, raw)
    with pytest.raises((typer.Exit, SystemExit)):  # a person's poros train on a received job
        _no_grants(raw, config)
    monkeypatch.setenv("POROS_JOB", "1")
    _no_grants(raw, config)  # the job's runner, on the job's own config: its submitter's flags
    nested = home / "home" / "jobs" / "j4" / "deeper"
    nested.mkdir(parents=True)
    (nested / "config.yaml").write_text("{}")
    link = home / "home" / "jobs" / "j5"
    link.mkdir(parents=True)
    (link / "config.yaml").symlink_to(home / "elsewhere.yaml")
    (home / "elsewhere.yaml").write_text("{}")
    for other in (nested / "config.yaml", link / "config.yaml", home / "config.yaml"):
        with pytest.raises((typer.Exit, SystemExit)):
            _no_grants(raw, other)


# ── an export never ships the run's private record ─────────────────────────────────────


def _exported(home, record: str, out: Path, run_id: str = "1118-1300-abcd"):
    import subprocess
    import sys

    d = runs.create(run_id, {"model": "Qwen/Qwen2.5-0.5B", "data": "synthetic",
                             "steps": 1, "seq_len": 8, "rank": 4, "command": "x"})
    runs.update(d, state="succeeded")
    (d / "adapter_config.json").write_text(json.dumps({"r": 4}))
    (d / "adapter_model.safetensors").write_text("w")
    (d / "poros_config.json").write_text(record)
    env = {**os.environ, "POROS_HOME": str(home / "home"), "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, "-m", "poros.cli.main", "export", run_id,
                           "--format", "adapter", "--output", str(out), "--yes", "--json"],
                          env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def test_an_export_writes_only_the_portable_record(home):
    private = str(home / "private-data" / "rows.jsonl")
    out = home / "shared"
    proc = _exported(home, json.dumps({"model_name_or_path": "Qwen/Qwen2.5-0.5B",
                                       "lora_rank": 4, "dataset": private}), out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "private-data" not in (out / "poros_config.json").read_text()


def test_an_unreadable_record_is_refused_before_the_folder_is_touched(home):
    out = home / "shared"
    proc = _exported(home, "{not json", out)
    assert proc.returncode != 0 and not out.exists(), proc.stdout
    out.mkdir()
    (out / "poros_config.json").write_text(json.dumps({"dataset": "/private/x"}))
    proc = _exported(home, json.dumps({"model_name_or_path": "Qwen/Qwen2.5-0.5B"}), out,
                     "1118-1301-abcd")
    assert proc.returncode == 0 and "/private/x" not in (out / "poros_config.json").read_text()


# ── eval's exception never goes below the frozen layers' floor ─────────────────────────


def _refusing(monkeypatch, free: float, at_least: bool):
    from poros.core.errors import PorosHostRamError
    from poros.train.trainer import PorosTrainer

    calls = []

    class Built:
        _event_writer = "events"

    def build(model, **kw):
        calls.append(kw)
        if kw.get("check_host_ram", True):
            raise PorosHostRamError(model, 40.0, free, "an estimate", at_least=at_least)
        return Built()

    monkeypatch.setattr(PorosTrainer, "from_pretrained", build)
    return calls


def test_eval_keeps_the_frozen_layers_floor_whatever_the_run_measured(home, monkeypatch):
    """The 72B's frozen layers alone take more host RAM than is free: refused, though the run's
    receipt says it used less; above that floor the measured peak still stands."""
    pytest.importorskip("torch")
    from poros.cli.verbs import evaluate_pass as ep
    from poros.core.config import PorosConfig
    from poros.core.errors import PorosHostRamError

    run_dir = home / "run"
    run_dir.mkdir()
    model = "Qwen/Qwen2.5-72B-Instruct"
    (run_dir / "config.resolved.json").write_text(json.dumps(
        {"model_name_or_path": model, "quantization": "nf4", "store": "ram"}))
    (run_dir / "receipt.json").write_text(json.dumps({"peak_host_ram_gb": 10.0}))
    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic", path="sequential")
    floor = ep.frozen_floor(cfg)
    assert floor is not None and floor > 30
    calls = _refusing(monkeypatch, free=floor - 1, at_least=True)
    with pytest.raises(PorosHostRamError):
        ep.engine(cfg, str(run_dir), [].append)
    assert all(kw.get("check_host_ram", True) for kw in calls)
    calls = _refusing(monkeypatch, free=floor + 2, at_least=False)
    said = []
    ep.engine(cfg, str(run_dir), said.append)
    assert calls[-1].get("check_host_ram") is False and said


# ── prove offers --block-sizes only where its one-block arm fits ───────────────────────


def test_prove_offers_block_sizes_only_where_one_block_of_every_layer_fits(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from poros.cli.verbs import prove_run
    from poros.core.refusal import Refusal

    a = SimpleNamespace(model="Qwen/Qwen2.5-72B-Instruct")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *x: (int(20 * 2**30), 0))
    monkeypatch.setattr(prove_run, "resident_floor_gib", lambda m: 40.0)
    monkeypatch.setattr(prove_run, "one_block_floor_gib", lambda m: 33.7)
    with pytest.raises(Refusal) as found:
        prove_run._refuse_unheld(a)
    assert "--block-sizes" not in (found.value.fix or "") and "nor can" in found.value.context[0]
    monkeypatch.setattr(prove_run, "one_block_floor_gib", lambda m: 10.0)
    with pytest.raises(Refusal) as found:
        prove_run._refuse_unheld(a)
    assert found.value.fix.endswith("--block-sizes")
    with pytest.raises(Refusal) as found:  # the block-sizes comparison itself, before a load
        prove_run._refuse_one_block(a, 5.0)
    assert found.value.title == "Can't hold one block of every layer"
    prove_run._refuse_one_block(a, 20.0)  # it fits: nothing said
