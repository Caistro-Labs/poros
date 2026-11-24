"""train and runs through words, on the simulated trainer (no GPU)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from poros import runs
from poros.cli import app, contract
from poros.cli.context import Context
from tests.conftest import needs_ml

# The planner's free host RAM, pinned: the simulated runs here plan real model sizes (a 7B wants
# ~15 GiB of host RAM), so a memory-capped shell or a small machine must not turn them into
# "predicted not to fit". In process it is monkeypatched; a child process gets PIN_RAM.
HOST_FREE_GIB = 64.0
PIN_RAM = f"import poros.core.env\nporos.core.env.host_ram_free = lambda: {HOST_FREE_GIB}\n"


# The steps a CUDA run's presets give (8 rows a step): a Mac's recipes train one row a step, and
# the simulated card's run takes this host's recipes
cuda_presets = pytest.mark.cuda_host(reason="the steps of the CUDA presets (8 rows a step); a "
                                     "Mac's recipes train one row a step, a simulated card too")


@pytest.fixture
def home(tmp_path, monkeypatch):
    import poros.core.env

    monkeypatch.setattr(poros.core.env, "host_ram_free", lambda: HOST_FREE_GIB)
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    # the simulated CUDA card these tests are written for, whatever the host is: on a Mac the
    # simulator previews the Mac itself (its disks, its table), which is test_mac's to check
    from poros.cli.ui import simulate

    monkeypatch.setenv("POROS_DEV_SIMULATE_GPU", "RTX 4090:24.0")
    simulate._gpu.cache_clear()
    monkeypatch.setenv("POROS_DEV_SIMULATE_STEP", "0.001")
    monkeypatch.setenv("POROS_DEV_SIMULATE_LOAD", "0")
    # huggingface_hub reads HF_HUB_OFFLINE once, at import: stub the Hub
    # lookup itself so no test depends on the network (or on import order).
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=None, unit="GiB", source="unknown", why="offline in tests"),
    )
    (tmp_path / "notes.jsonl").write_text(
        "".join(json.dumps({"text": "word " * (20 + i)}) + "\n" for i in range(40))
    )
    yield tmp_path
    contract.reset()
    simulate._gpu.cache_clear()


def say(argv, home, capsys):
    ctx = Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False)
    old = os.getcwd()
    os.chdir(home)
    try:
        code = app.dispatch(argv, ctx)
    finally:
        os.chdir(old)
    # In-process there is no fd-level isolation, so the human view shares
    # stdout here; the subprocess contract tests check stdout holds only JSON.
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert len(lines) == 1, lines
    return code, json.loads(lines[0])


def test_train_asks_for_missing_data_never_guesses(home, capsys):
    code, obj = say(["train", "Qwen2.5-7B-Instruct"], home, capsys)
    assert code == 2 and obj["questions"][0]["id"] == "data"
    commands = [c["command"] for c in obj["questions"][0]["choices"]]
    assert "poros train Qwen2.5-7B-Instruct --data notes.jsonl --json" in commands


def test_a_local_folder_on_a_card_still_asks_before_it_trains(home, capsys):
    model = home / "mine"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    code, obj = say(["train", str(model), "--data", "notes.jsonl"], home, capsys)
    assert code == 2 and "not in the catalog" in obj["questions"][0]["prompt"]
    assert "--allow-unvalidated" in obj["questions"][0]["choices"][0]["command"]


def test_train_runs_in_a_run_folder(home, capsys, path_receipts):
    code, obj = say(["train", "Qwen2.5-7B-Instruct", "--data", "notes.jsonl"], home, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1", obj
    # 40 rows at 8 examples per step: one epoch is 5 steps, inferred.
    # Paths are stored absolute: a run continued from elsewhere reads the same file.
    assert obj["command"].startswith(
        f"poros train Qwen/Qwen2.5-7B-Instruct --data {home / 'notes.jsonl'} --steps 5 "
        "--seq-len 512"  # short rows ask 256; the card and this both open at the measured length
    )
    status = json.loads((runs.root() / obj["run"] / "status.json").read_text())
    action = json.loads((runs.root() / obj["run"] / "action.json").read_text())
    assert status["state"] == "succeeded" and action["provenance"]["steps"]["kind"] == "inferred"


def test_unknown_setting_is_refused_with_a_suggestion(home, capsys):
    code, obj = say(
        ["train", "Qwen2.5-7B-Instruct", "--data", "synthetic", "learning_rat=1e-4"], home, capsys
    )
    assert obj["error"]["code"] == "E_CONFIG" and "learning_rate" in obj["error"]["message"]


def _fake_run(state: str, pid=None) -> str:
    rid = runs.new_id()
    d = runs.create(
        rid,
        {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "data": "synthetic",
            "steps": 3,
            "seq_len": 256,
            "rank": 16,
            "name": "fake",
        },
    )
    runs.update(d, state=state, pid=pid)
    return rid


def test_runs_list_and_report(home, capsys):
    say(["train", "Qwen2.5-7B-Instruct", "--data", "synthetic", "--steps", "3"], home, capsys)
    code, obj = say(["runs"], home, capsys)
    assert obj["schema"] == "poros.runs.v1" and obj["runs"][0]["state"] == "succeeded"
    assert obj["runs"][0]["step"] == 3
    code, obj = say(["runs", "report", "last"], home, capsys)
    assert Path(obj["report"]).read_text().startswith("# qwen2.5-7b-instruct-synthetic")


def test_stop_is_a_request_and_a_dead_run_reads_failed(home, capsys):
    live = _fake_run("running", pid=os.getpid())
    code, obj = say(["runs", "stop", live], home, capsys)
    assert code == 0 and obj["runs"][0]["state"] == "stopping"
    assert (runs.root() / live / "stop").exists()
    dead = _fake_run("running", pid=2**22 + 12345)
    assert next(r for r in runs.all_runs() if r.id == dead).state == "failed"
    done = _fake_run("succeeded")
    code, obj = say(["runs", "stop", done], home, capsys)
    assert obj["error"]["code"] == "E_USAGE"


def test_continue_resumes_from_the_newest_checkpoint(home, capsys, monkeypatch):
    from poros.core import checkpoints

    rid = _fake_run("cancelled")
    code, obj = say(["runs", "continue", rid], home, capsys)  # stopped before any checkpoint
    assert obj["error"]["code"] == "E_NOT_FOUND" and "first checkpoint" in obj["error"]["message"]
    ckpt = runs.root() / rid / "checkpoint-2"
    monkeypatch.setattr(checkpoints, "list_checkpoints", lambda d: [(2, ckpt)])
    code, obj = say(["runs", "continue", rid], home, capsys)
    assert code == 0 and obj["runs"][0]["state"] == "succeeded"


def test_continue_refuses_changed_settings(home, capsys):
    rid = _fake_run("cancelled")
    code, obj = say(["runs", "continue", rid, "--steps", "5"], home, capsys)
    assert obj["error"]["code"] == "E_USAGE" and "bitwise" in obj["error"]["message"]


def test_train_refuses_its_own_fields_as_settings(home, capsys):
    code, obj = say(
        ["train", "Qwen2.5-7B-Instruct", "--data", "synthetic", "max_steps=5"], home, capsys
    )
    assert obj["error"]["code"] == "E_USAGE" and "--steps N" in obj["error"]["message"]
    code, obj = say(
        [
            "train",
            "Qwen2.5-7B-Instruct",
            "--data",
            "notes.jsonl",
            "--epochs",
            "2",
            "--steps",
            "5",
        ],
        home,
        capsys,
    )
    assert obj["error"]["code"] == "E_USAGE"


def test_progress_reads_past_a_large_teardown_event(tmp_path):
    lines = [
        json.dumps({"event_type": "step", "step": i, "data": {"loss": 1.0 / (i + 1)}})
        for i in range(10)
    ]
    lines.append(json.dumps({"event_type": "teardown", "data": {"h": ["x" * 100] * 3000}}))
    (tmp_path / "events.jsonl").write_text("\n".join(lines) + "\n")
    assert runs._progress(str(tmp_path)) == (10, 0.1)  # 0-based events: 10 steps done


def test_stdout_is_one_json_object_end_to_end(home):
    import subprocess
    import sys

    pin = home / "pin"
    pin.mkdir()
    (pin / "sitecustomize.py").write_text(PIN_RAM)
    checkout = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(pin), checkout])}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "poros.cli.main",
            "train",
            "Qwen2.5-7B-Instruct",
            "--data",
            "synthetic",
            "--steps",
            "3",
            "--json",
        ],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr
    (line,) = proc.stdout.splitlines()
    assert json.loads(line)["schema"] == "poros.train.v1"
    assert "Trained" in proc.stderr  # the human screen went to stderr


def _chats(home, n=40):
    rows = [{"messages": [{"role": "user", "content": f"q {i}"},
                          {"role": "assistant", "content": f"a {i} " * 10}]} for i in range(n)]
    (home / "chats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pairs = [{"prompt": f"q {i}", "chosen": f"good {i}", "rejected": f"bad {i}"} for i in range(n)]
    (home / "prefs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pairs))


def test_unknown_flag_is_refused_with_the_nearest(home, capsys):
    code, obj = say(["train", "Qwen2.5-7B-Instruct", "--data", "notes.jsonl", "--step", "5"],
                    home, capsys)
    assert code == 2 and obj["error"]["code"] == "E_USAGE"
    assert "did you mean --steps?" in obj["error"]["message"]


@cuda_presets
def test_recipes_train_at_the_measured_preset(home, capsys):
    _chats(home)
    code, obj = say(["train", "Qwen2.5-7B-Instruct", "--data", "chats.jsonl"], home, capsys)
    assert code == 0, obj
    assert "--steps 10" in obj["command"]  # 40 chats, 8 a step, 2 epochs
    resolved = json.loads((Path(obj["output_dir"]) / "config.resolved.json").read_text())
    assert resolved["trained_with"] == "sft" and resolved["learning_rate"] == 2e-4
    assert (resolved["lora_dropout"], resolved["block_size"], resolved["quantization"]) == (
        0.0, 4, "nf4")
    code, obj = say(["train", "Qwen2.5-7B-Instruct", "--data", "prefs.jsonl",
                     "learning_rate=1e-6"], home, capsys)
    resolved = json.loads((Path(obj["output_dir"]) / "config.resolved.json").read_text())
    assert resolved["trained_with"] == "dpo" and resolved["learning_rate"] == 1e-6  # yours wins


def test_a_recipes_block_size_never_reaches_the_moe_path(home, capsys, monkeypatch):
    """poros.moe refuses a block size, and on CUDA the recipes' one would be why a MoE model
    could not train from chats. One you say yourself is refused before the card, in the MoE
    path's own words, and nothing starts."""
    _chats(home)
    monkeypatch.setattr("poros.preflight.arch.resolve_model_type", lambda name: ("olmoe", ""))
    model = "OLMoE-1B-7B-0924"
    code, obj = say(["train", model, "--data", "chats.jsonl"], home, capsys)
    assert code == 0, obj
    resolved = json.loads((Path(obj["output_dir"]) / "config.resolved.json").read_text())
    assert resolved["block_size"] is None and resolved["trained_with"] == "sft"
    code, obj = say(["train", model, "--data", "chats.jsonl", "--block-size", "2"], home, capsys)
    assert code != 0 and "output_dir" not in obj, obj
    assert obj["error"]["code"] == "E_CONFIG" and "block_size=2" in obj["error"]["message"], obj


def test_a_wrong_recipe_hints_the_command_that_trains_it(home, capsys):
    _chats(home)
    code, obj = say(["train", "Qwen2.5-7B-Instruct", "--data", "prefs.jsonl", "--recipe", "sft"],
                    home, capsys)
    assert code == 2 and obj["error"]["hint"].endswith("--recipe dpo")


@cuda_presets
def test_dpo_from_an_sft_run(home, capsys):
    _chats(home)
    _, sft = say(["train", "Qwen2.5-7B-Instruct", "--data", "chats.jsonl"], home, capsys)
    code, dpo = say(["train", "--from", "last", "--data", "prefs.jsonl"], home, capsys)
    assert code == 0, dpo
    assert dpo["command"].startswith("poros train Qwen/Qwen2.5-7B-Instruct ")
    assert f"--from {sft['run']}" in dpo["command"]
    resolved = json.loads((Path(dpo["output_dir"]) / "config.resolved.json").read_text())
    assert resolved["init_adapter"] == str(Path(sft["output_dir"]) / "checkpoint-9")
    code, obj = say(["train", "Qwen2.5-0.5B-Instruct", "--from", sft["run"], "--data",
                     "prefs.jsonl"], home, capsys)
    assert code == 2 and "trained Qwen/Qwen2.5-7B-Instruct" in obj["error"]["message"]


@pytest.mark.parametrize("where", [(), ("--output", "elsewhere")], ids=["its-folder", "--output"])
def test_from_keeps_the_alpha_of_a_run_started_directly(home, capsys, where):
    """The record is in the run's output_dir, which is its folder only by default."""
    _chats(home)
    code, parent = say(["train", "Qwen2.5-7B-Instruct", "--data", "chats.jsonl", "--rank", "8",
                        "lora_alpha=16", *where], home, capsys)
    assert code == 0, parent
    code, dpo = say(["train", "--from", parent["run"], "--data", "prefs.jsonl"], home, capsys)
    assert code == 0, dpo
    resolved = json.loads((Path(dpo["output_dir"]) / "config.resolved.json").read_text())
    assert (resolved["lora_rank"], resolved["lora_alpha"]) == (8, 16)


@cuda_presets
def test_a_finished_run_trains_more(home, capsys):
    _chats(home)
    _, first = say(["train", "Qwen2.5-7B-Instruct", "--data", "chats.jsonl"], home, capsys)
    code, obj = say(["runs", "continue", first["run"]], home, capsys)
    assert code == 2 and obj["error"]["hint"] == f"poros runs continue {first['run']} --more 200"
    code, obj = say(["runs", "continue", first["run"], "--more", "4", "--yes"], home, capsys)
    assert code == 0, obj
    assert (Path(first["output_dir"]) / "checkpoint-13").is_dir()


@cuda_presets
def test_detach_continues_in_a_session_of_its_own_and_returns_with_the_run(home, capsys):
    _chats(home)
    _, first = say(["train", "Qwen2.5-7B-Instruct", "--data", "chats.jsonl"], home, capsys)
    code, obj = say(["runs", "continue", first["run"], "--more", "2", "--yes", "--detach"], home,
                    capsys)
    assert code == 0, obj
    deadline = time.time() + 60
    while next(r for r in runs.all_runs() if r.id == first["run"]).state == "running":
        assert time.time() < deadline
        time.sleep(0.1)
    assert (Path(first["output_dir"]) / "checkpoint-11").is_dir()


def test_detach_reports_what_stopped_the_attempt(home, capsys):
    from poros.cli.verbs import train_run
    from poros.core.refusal import Refusal

    _chats(home)
    _, first = say(["train", "Qwen2.5-7B-Instruct", "--data", "chats.jsonl"], home, capsys)
    run = next(r for r in runs.all_runs() if r.id == first["run"])
    for saved in Path(first["output_dir"]).glob("checkpoint-*"):
        shutil.rmtree(saved)
    with pytest.raises(Refusal, match="before its first checkpoint"):
        train_run.continue_detached(run, 2)


def test_a_setting_of_the_wrong_kind_is_named_not_a_bug(home, capsys):
    _chats(home)
    code, obj = say(["train", "Qwen2.5-7B-Instruct", "--data", "chats.jsonl",
                     "learning_rate=fast"], home, capsys)
    assert code == 2 and obj["error"]["code"] == "E_CONFIG"
    assert "learning_rate" in obj["error"]["message"]


@needs_ml
def test_rows_with_no_answer_left_are_the_datas_fault_and_say_so(monkeypatch):
    from types import SimpleNamespace as NS

    from poros import data as pdata
    from poros.cli.train_recipe import train_recipe
    from poros.core.errors import PorosDataError

    cfg = NS(per_device_batch_size=1, gradient_accumulation_steps=8, max_seq_length=256, seed=0,
             train_on="answers", dataset="d.jsonl")

    def no_rows(*a, **k):
        raise ValueError("no rows to train on")

    monkeypatch.setattr(pdata, "chats", no_rows)
    with pytest.raises(PorosDataError, match="cut away at 256 tokens"):
        train_recipe(NS(_tokenizer=object()), cfg, NS(shape="chats"), None, None)


def test_the_receipt_records_the_batch_size(home, capsys):
    """The receipt's config names PorosConfig's own per_device_batch_size (it read
    TrainingArguments' per_device_train_batch_size, which a PorosConfig has not: always null)."""
    code, obj = say(["train", "Qwen2.5-7B-Instruct", "--data", "synthetic", "--steps", "3"],
                    home, capsys)
    assert code == 0, obj
    receipt = json.loads((Path(obj["output_dir"]) / "receipt.json").read_text())
    assert isinstance(receipt["config"]["per_device_batch_size"], int)
    assert "per_device_train_batch_size" not in receipt["config"]


def test_a_run_refused_before_the_load_is_no_run_and_never_last(home, capsys):
    """A run refused by the checks before the load leaves no run: it is not listed, does not
    become `last`, and the output folder it named is left as it was."""
    code, first = say(["train", "Qwen2.5-0.5B-Instruct", "--data", "synthetic", "--steps", "1"],
                      home, capsys)
    assert code == 0 and first["schema"] == "poros.train.v1"
    out = home / "out"
    out.mkdir()
    (out / "link").symlink_to(home / "notes.jsonl")  # a run writes through no link: refused
    code, refused = say(["train", "Qwen2.5-0.5B-Instruct", "--data", "synthetic", "--steps", "1",
                         "--output", str(out), "--yes"], home, capsys)
    assert code == 2 and refused["error"]["code"] == "E_CONFIG"
    assert [r.id for r in runs.all_runs()] == [first["run"]]
    assert sorted(p.name for p in out.iterdir()) == ["link"]
    code, last = say(["runs", "last"], home, capsys)
    assert code == 0 and first["run"] in json.dumps(last)


def test_export_names_a_missing_folder_and_takes_the_folder_forms_flags(home, capsys):
    """`poros export <missing folder> -o X` says the folder is missing, not that export has no
    -o: the run form reads -o as --output, as the folder form does."""
    code, obj = say(["export", "./nope", "-o", "X"], home, capsys)
    assert code == 2 and obj["error"]["code"] == "E_NOT_FOUND"
    assert "nothing at ./nope" in obj["error"]["message"]
    code, obj = say(["train", "Qwen2.5-0.5B-Instruct", "--data", "synthetic", "--steps", "1"],
                    home, capsys)
    assert code == 0
    from poros.cli.verbs import export as verb
    from poros.cli.words.grammar import parse

    draft = verb.resolve(parse(["export", "last", "--format", "merged", "-o", "X"]),
                         Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False))
    assert draft.action.output == "X" and draft.action.format == "merged"


def test_the_folder_form_takes_merged_as_the_run_form_does(tmp_path):
    """The folder form refused --format merged, which `poros export last` takes: it is
    safetensors by its other name in both."""
    from typer.testing import CliRunner

    from poros.cli.main import app as typer_app

    folder = tmp_path / "adapter"
    folder.mkdir()
    out = CliRunner().invoke(typer_app, ["export", str(folder), "--format", "merged", "--json"])
    assert "Unsupported format" not in out.output and "is not one" not in out.output
    out = CliRunner().invoke(typer_app, ["export", str(folder), "--format", "gguf", "--json"])
    assert "is not one" in out.output


@pytest.mark.parametrize("refusal", ["data gone", "no torch"])
def test_a_continue_refused_before_its_load_leaves_the_run_as_it_was(home, capsys, monkeypatch,
                                                                    refusal):
    """`runs continue <run> --more N` refused by its checks leaves the run as it was: a
    succeeded run stays succeeded, at the step count it trained."""
    import shutil as _shutil

    rows = home / "rows.jsonl"
    _shutil.copy(home / "notes.jsonl", rows)
    code, first = say(["train", "Qwen2.5-0.5B-Instruct", "--data", str(rows), "--steps", "3"],
                      home, capsys)
    assert code == 0, first
    folder = runs.root() / first["run"]
    before = {n: (folder / n).read_bytes() for n in ("status.json", "action.json")}
    if refusal == "data gone":
        rows.unlink()
    else:
        from poros.cli import contract as _contract
        from poros.cli import train_cmd
        from poros.preflight.command import NO_TORCH

        monkeypatch.setattr(train_cmd, "_before_the_load",
                            lambda *a, **k: _contract.refused(NO_TORCH))
    code, obj = say(["runs", "continue", first["run"], "--more", "2", "--yes"], home, capsys)
    assert code != 0 and obj["schema"] == "poros.error.v1", obj
    assert {n: (folder / n).read_bytes() for n in before} == before
    run = next(r for r in runs.all_runs() if r.id == first["run"])
    assert run.state == "succeeded"


def test_a_merge_that_would_download_its_base_asks_first(home, capsys, monkeypatch):
    """`poros export <run> --format merged` asks before it downloads the base ("Downloads always
    ask first"); with no terminal it needs --download or --yes."""
    import poros.plan.configs
    from poros.cli.schemas import Figure

    code, first = say(["train", "Qwen2.5-0.5B-Instruct", "--data", "synthetic", "--steps", "1"],
                      home, capsys)
    assert code == 0
    monkeypatch.setattr(poros.plan.configs, "download_size",
                        lambda cfg: Figure(value=0.9, unit="GiB", source="hub_metadata"))
    code, obj = say(["export", first["run"], "--format", "merged"], home, capsys)
    assert code == 2 and obj["schema"] == "poros.needs_input.v1", obj
    q = obj["questions"][0]
    assert q["id"] == "download" and "--download" in json.dumps(q)
