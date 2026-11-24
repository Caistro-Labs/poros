"""`poros train --detach` and `poros jobs`: the lifecycle, end to end on CPU.

The jobs run the real runner and the real `poros train` child process. Only
the model is faked: a sitecustomize on PYTHONPATH swaps in a tiny CPU Qwen2
whose optimizer step waits while a hold file exists, so the test decides when
a job moves on instead of racing a timer.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from poros import jobs

_FAKE = '''
import os
if os.environ.get("POROS_FAKE_TRAIN"):
    import time
    import torch
    import poros.jobs
    import poros.core.card
    from poros.train import trainer as T

    real_cuda = torch.cuda.is_available
    torch.cuda.is_available = lambda: True  # for the CLI's GPU check only
    # and the card that check reads (poros.core.card.check): an RTX 4090, a torch built for it
    torch.cuda.get_device_capability = lambda device=None: (8, 9)
    torch.cuda.get_device_name = lambda device=None: "NVIDIA GeForce RTX 4090"
    torch.cuda.get_arch_list = lambda: ["sm_80", "sm_86", "sm_89"]
    poros.core.card.detect_gpu = lambda name=None, quietly=False: None
    poros.jobs.gpu_uuid = lambda: "GPU-fake"
    # the fake's model has no weights anywhere: nothing for the model check to look up
    import poros.preflight.arch
    poros.preflight.arch.hub_fault = lambda model, revision=None: None

    def from_pretrained(cls, name, *, config, **kw):
        # past the CLI's GPU check: what is imported from here on (peft, and bitsandbytes
        # through it) sees this torch's own CUDA, which a CPU-only build does not have
        torch.cuda.is_available = real_cuda
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=64)
        model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"))
        tr = cls(config)
        tr.model, tr.device, tr._adapter_backend = model, torch.device("cpu"), "peft"
        tr._setup_blockwise_training()
        return tr

    real_train, real_step = T.PorosTrainer.train, T.PorosTrainer._adapter_optimizer_step

    def train(self, dataset, **kw):
        ids = torch.randint(0, 256, (64, 16), generator=torch.Generator().manual_seed(0))
        rows = [{"input_ids": x, "labels": x.clone()} for x in ids]
        torch.cuda.is_available = real_cuda
        return real_train(self, torch.utils.data.DataLoader(rows, batch_size=1), **kw)

    def step(self):
        while self.global_step >= 1 and os.path.exists(os.environ["POROS_FAKE_HOLD"]):
            time.sleep(0.05)
        return real_step(self)

    T.PorosTrainer.from_pretrained = classmethod(from_pretrained)
    T.PorosTrainer.train, T.PorosTrainer._adapter_optimizer_step = train, step
'''

_CONFIG = """\
model_name_or_path: tiny
dataset: synthetic
block_size: 2
blockwise_arch_check: "off"
enable_prefetch: false
gradient_accumulation_steps: 1
max_steps: 30
max_seq_length: 16
warmup_steps: 0
save_steps: 0
eval_steps: 0
lora_dropout: 0.0
"""


# A job's runner here fakes a CUDA host in its own process (_FAKE); on a Mac the CLI it runs
# takes the Mac's gate, which refuses this config's block_size, prefetch and arch check
cuda_job = pytest.mark.cuda_host(reason="a job here fakes a CUDA host in its runner; on a Mac the "
                                 "CLI takes the Mac's gate, which refuses the CPU test config")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The CLI's environment; jobs held at step 1 until the test lets go."""
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "sitecustomize.py").write_text(_FAKE)
    (tmp_path / "run.yaml").write_text(_CONFIG)
    hold = tmp_path / "hold"
    hold.touch()
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))  # for in-process jobs.read
    env = {k: v for k, v in os.environ.items() if k not in ("POROS_OUTPUT", "CUDA_VISIBLE_DEVICES")}
    yield {**env, "PYTHONPATH": f"{fake}{os.pathsep}{env.get('PYTHONPATH', '')}",
           "POROS_FAKE_TRAIN": "1", "POROS_FAKE_HOLD": str(hold), "HF_HUB_OFFLINE": "1"}
    # Never leave runners behind: cancel everything, release the hold, wait.
    for job in jobs.all_jobs():
        (Path(job.job_dir) / "stop").touch()
    hold.unlink(missing_ok=True)
    deadline = time.time() + 60
    while any(jobs.is_live(Path(j.job_dir)) for j in jobs.all_jobs()) and time.time() < deadline:
        time.sleep(0.2)


def _poros(env, *args):
    return subprocess.run([sys.executable, "-m", "poros.cli.main", *args], env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)


def _job(env, *args):
    proc = _poros(env, *args, "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def _wait(job_id, until, timeout=120):
    deadline = time.time() + timeout
    job = jobs.read(jobs.find(job_id))
    while not until(job):
        assert time.time() < deadline, f"timed out: {job}"
        time.sleep(0.2)
        job = jobs.read(jobs.find(job_id))
    return job


def _stepped(job):
    # a step's own event: every event has a "step" key, the setup event before any step too
    events = Path(job.output_dir) / "events.jsonl"
    return events.exists() and '"event_type":"step"' in events.read_text()


def _done(job):
    return job.state in jobs.TERMINAL


@cuda_job
def test_cancel_at_a_step_boundary_then_resume_matches_an_uninterrupted_run(env, tmp_path):
    a = _job(env, "train", str(tmp_path / "run.yaml"), "--detach")
    b = _job(env, "train", str(tmp_path / "run.yaml"), "--detach")
    assert a["schema"] == "poros.job.v1" and a["state"] in ("queued", "running")
    assert b["state"] == "queued"

    _wait(a["job_id"], _stepped)  # held at step 1
    assert jobs.read(jobs.find(b["job_id"])).state == "queued"  # one job per GPU
    assert _job(env, "jobs", "cancel", a["job_id"])["state"] == "cancelling"
    Path(env["POROS_FAKE_HOLD"]).unlink()
    assert _poros(env, "jobs", "logs", a["job_id"], "--follow").returncode == 7
    a = jobs.read(jobs.find(a["job_id"]))
    assert a.state == "cancelled" and a.error.code == "E_JOB_CANCELLED"
    assert [p.name for p in Path(a.output_dir).glob("checkpoint-*")] == ["checkpoint-1"]

    b = _wait(b["job_id"], _done)
    assert b.state == "succeeded", Path(b.log).read_text()
    assert b.started_at >= a.finished_at

    assert _job(env, "jobs", "resume", a.job_id)["attempt"] == 2
    a = _wait(a.job_id, _done)
    assert a.state == "succeeded", Path(a.log).read_text()
    assert a.result["metrics"]["final_loss"] == b.result["metrics"]["final_loss"]

    listed = _job(env, "jobs", "list")
    assert listed["schema"] == "poros.jobs.v1"
    assert [j["job_id"] for j in listed["jobs"]] == [a.job_id, b.job_id]


@cuda_job
def test_a_queued_job_cancels_without_running(env, tmp_path):
    a = _job(env, "train", str(tmp_path / "run.yaml"), "--detach")
    b = _job(env, "train", str(tmp_path / "run.yaml"), "--detach")
    _job(env, "jobs", "cancel", b["job_id"])
    b = _wait(b["job_id"], lambda j: j.state == "cancelled")
    assert b.started_at is None and not list(Path(b.output_dir).iterdir())
    assert _wait(a["job_id"], lambda j: j.state == "running")


def _status(job_dir, state):
    jobs.write_status(job_dir, job_id=job_dir.name, state=state, model="m",
                      job_dir=str(job_dir), config="c", output_dir=str(job_dir / "a"),
                      log=str(job_dir / "log.txt"), gpu="GPU-x", attempt=1, queued_at="t")


def test_a_dead_runner_reads_as_failed_and_a_broken_dir_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path))
    dead = jobs.jobs_root() / "20260101-000000-dead"
    dead.mkdir(parents=True)
    _status(dead, "running")
    (jobs.jobs_root() / "20260101-000001-junk").mkdir()
    (jobs.jobs_root() / "20260101-000001-junk" / "status.json").write_text("{")
    [job] = jobs.all_jobs()
    assert job.state == "failed" and "runner exited" in job.error.message


def test_unknown_job_is_not_found(tmp_path):
    env = {**os.environ, "POROS_HOME": str(tmp_path)}
    proc = _poros(env, "jobs", "status", "nope", "--json")
    assert proc.returncode == 2 and json.loads(proc.stdout)["error"]["code"] == "E_NOT_FOUND"


@cuda_job
def test_from_keeps_the_alpha_of_a_run_started_as_a_job(env, tmp_path):
    """A job's record is in its artifacts, not its folder: SFT at alpha 16 was followed by DPO at
    alpha 32 on real runs, with no word."""
    from poros import runs
    from poros.cli.verbs.train import TrainAction, _inherit

    Path(env["POROS_FAKE_HOLD"]).unlink()
    (tmp_path / "run.yaml").write_text(
        _CONFIG.replace("max_steps: 30", "max_steps: 2") + "lora_rank: 8\nlora_alpha: 16\n")
    job = _wait(_job(env, "train", str(tmp_path / "run.yaml"), "--detach")["job_id"], _done)
    assert job.state == "succeeded", Path(job.log).read_text()
    a = TrainAction(model="tiny", data="d")
    _inherit(a, next(r for r in runs.all_runs() if r.id == job.job_id), None)
    assert (a.rank, a.config["lora_alpha"]) == (8, 16)


@cuda_job
def test_a_finished_job_trains_more(env, tmp_path):
    from poros import runs

    Path(env["POROS_FAKE_HOLD"]).unlink()
    (tmp_path / "run.yaml").write_text(
        _CONFIG.replace("max_steps: 30", "max_steps: 4").replace("save_steps: 0", "save_steps: 1"))
    a = _job(env, "train", str(tmp_path / "run.yaml"), "--detach")
    a = _wait(a["job_id"], _done)
    assert a.state == "succeeded", Path(a.log).read_text()

    listed = next(r for r in runs.all_runs() if r.id == a.job_id)
    assert listed.total == 4 and listed.name == a.job_id  # none given: its id
    out = _job(env, "runs", "continue", a.job_id, "--more", "2", "--yes", "--detach")
    assert out["runs"][0]["total"] == 6 and out["runs"][0]["state"] in ("queued", "running")
    a = _wait(a.job_id, _done)
    assert a.state == "succeeded" and a.attempt == 2, Path(a.log).read_text()
    assert (Path(a.output_dir) / "checkpoint-5").is_dir()


@cuda_job
def test_more_refuses_a_job_with_no_checkpoint_at_its_last_step(env, tmp_path):
    Path(env["POROS_FAKE_HOLD"]).unlink()
    (tmp_path / "run.yaml").write_text(_CONFIG.replace("max_steps: 30", "max_steps: 2"))
    a = _wait(_job(env, "train", str(tmp_path / "run.yaml"), "--detach")["job_id"], _done)
    for saved in Path(a.output_dir).glob("checkpoint-*"):
        shutil.rmtree(saved)
    proc = _poros(env, "runs", "continue", a.job_id, "--more", "2", "--yes", "--json")
    assert proc.returncode == 2 and "kept no checkpoint" in proc.stdout


def test_a_job_answers_to_its_name(tmp_path, monkeypatch):
    from poros import runs

    monkeypatch.setenv("POROS_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "launch", lambda job_dir: None)
    job = jobs.submit({"max_steps": 2}, model="m", gpu="GPU-x", name="notes-v1")
    assert job.name == "notes-v1"
    assert next(r for r in runs.all_runs() if r.id == job.job_id).name == "notes-v1"


@cuda_job
def test_more_refuses_a_job_with_no_total_to_extend(env, tmp_path):
    Path(env["POROS_FAKE_HOLD"]).unlink()
    (tmp_path / "run.yaml").write_text(_CONFIG.replace("max_steps: 30", "max_steps: 2"))
    a = _wait(_job(env, "train", str(tmp_path / "run.yaml"), "--detach")["job_id"], _done)
    config = Path(a.config)
    config.write_text(config.read_text().replace("max_steps: 2", "max_steps: 0"))
    proc = _poros(env, "runs", "continue", a.job_id, "--more", "2", "--yes", "--json")
    assert proc.returncode == 2 and "trains by epochs" in proc.stdout


def _umask() -> int:
    was = os.umask(0)
    os.umask(was)
    return was


def test_a_queued_jobs_result_holds_its_plan(monkeypatch):
    """Under `poros agents`, what it resolved; a person's job has none."""
    from types import SimpleNamespace as Bag

    from poros.cli.verbs import train_run
    from poros.cli.verbs.train import TrainAction

    job = Bag(plan=None, state="running")
    monkeypatch.setattr(train_run, "_submit", lambda a, cfg: job)
    action = TrainAction(model="m", data="synthetic", steps=9, seq_len=256, detach=True)
    monkeypatch.delenv("POROS_AGENT", raising=False)
    assert train_run.run(action, None).plan is None
    monkeypatch.setenv("POROS_AGENT", "1")
    assert train_run.run(action, None).plan["steps"] == 9


@cuda_job
def test_an_agent_runs_a_job_by_its_commands_alone(env, tmp_path):
    """A real job on the CPU: `train` returns at once and says what to run, a run's checkpoints stay
    while it trains, and what it offers runs. The tiny model is the fake's: no weights."""
    def agent(*words, code=0):
        proc = _poros(env, "agents", *words)
        assert proc.returncode == code, proc.stdout + proc.stderr
        assert len(proc.stdout.strip().splitlines()) == 1, proc.stdout
        return json.loads(proc.stdout)

    (tmp_path / "run.yaml").write_text(
        _CONFIG.replace("max_steps: 30", "max_steps: 4").replace("save_steps: 0", "save_steps: 2"))
    began = agent("train", str(tmp_path / "run.yaml"))
    run = began["job_id"]
    assert began["schema"] == "poros.job.v1" and "job_dir" not in began
    assert began["next"] == [f"poros agents run watch {run} --wait 30"]
    _wait(run, _stepped)
    live = agent("run", "watch", run)
    assert live["schema"] == "poros.run_live.v1" and live["state"] == "running"
    refused = agent("checkpoint", "delete", "--run-id", run, "-y", code=2)["error"]
    assert refused["hint"] == f"poros agents run stop {run}"
    Path(env["POROS_FAKE_HOLD"]).unlink()
    _wait(run, _done)
    info = agent("run", "info", run)
    assert info["runs"][0]["state"] == "succeeded" and info["next"][-1].endswith(f"save {run}")
    assert info["learned"] == "no held-out loss: this run scored no held-out rows"
    assert agent("checkpoint", "list", "--run-id", "last")["checkpoints"]
    listed = agent("checkpoint", "list", "--run-id", run)["checkpoints"]
    assert listed[0]["path"] == f"poros://{run}/weights/final" and listed[0]["step"] == 4
    steps = [c["step"] for c in listed if c["type"] == "train"]
    assert steps and steps == sorted(steps, reverse=True) and all(c["size_bytes"] for c in listed)
    assert all(c["loss"] is not None for c in listed)
    done = agent("checkpoint", "download", f"poros://{run}/weights/final", "-o", str(tmp_path))
    weights = Path(done["output"]) / "adapter_model.safetensors"
    assert weights.read_bytes() == (Path(jobs.read(jobs.find(run)).output_dir)
                                    / "adapter_model.safetensors").read_bytes()
    asked = agent("checkpoint", "delete", listed[-1]["path"], code=2)
    gone = _poros(env, *shlex.split(asked["questions"][0]["choices"][0]["command"])[1:])
    assert gone.returncode == 0 and json.loads(gone.stdout)["do"] == "delete"
    exported = agent("export", run, "--format", "adapter")
    assert (Path(exported["output"]) / "adapter_model.safetensors").exists()
    assert "adapter" not in exported and exported["base_model"] == "tiny"
    assert "from peft import PeftModel" in exported["load"]
    made = (Path(exported["output"]) / "adapter_model.safetensors").stat().st_mode & 0o777
    assert made == 0o666 & ~_umask()


def test_a_job_is_named_by_its_id_alone_and_logs_print_its_own_log(tmp_path):
    """A path in place of an id (absolute, or with ..) names no job; and `jobs logs` prints the
    job's own log.txt, never a file its status.json points at."""
    env = {**os.environ, "POROS_HOME": str(tmp_path / "home")}
    secret = tmp_path / "private.txt"
    secret.write_text("PRIVATE-MARKER\n")
    outside = tmp_path / "project" / "jobx"
    outside.mkdir(parents=True)
    _status(outside, "succeeded")
    status = json.loads((outside / "status.json").read_text())
    (outside / "status.json").write_text(json.dumps({**status, "log": str(secret)}))
    jobs_dir = Path(env["POROS_HOME"]) / "jobs"
    jobs_dir.mkdir(parents=True)
    for word in (str(outside), "../project/jobx", "../../project/jobx"):
        proc = _poros(env, "jobs", "logs", word)
        assert proc.returncode == 2 and "PRIVATE-MARKER" not in proc.stdout + proc.stderr, word
    own = jobs_dir / "20260101-000000-abcd"
    shutil.copytree(outside, own)
    (own / "log.txt").write_text("THE-JOBS-OWN-LOG\n")
    proc = _poros(env, "jobs", "logs", own.name)
    assert proc.returncode == 0 and "THE-JOBS-OWN-LOG" in proc.stdout
    assert "PRIVATE-MARKER" not in proc.stdout
    (own / "log.txt").unlink()
    (own / "log.txt").symlink_to(secret)  # a link in its place is not its log
    proc = _poros(env, "jobs", "logs", own.name)
    assert "PRIVATE-MARKER" not in proc.stdout


def test_find_refuses_every_path_form(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "status.json").write_text("{}")
    jobs.jobs_root().mkdir(parents=True)
    (jobs.jobs_root() / "linked").symlink_to(elsewhere)
    for word in (str(elsewhere), "../../elsewhere", "..", ".", "", "a/b", "linked"):
        assert jobs.find(word) is None, word
    (jobs.jobs_root() / "20260101-000000-abcd").mkdir()
    (jobs.jobs_root() / "20260101-000000-abcd" / "status.json").write_text("{}")
    assert jobs.find("20260101-000000-abcd") == jobs.jobs_root() / "20260101-000000-abcd"
