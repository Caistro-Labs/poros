"""The verbs on a Mac, through app.dispatch, against a fixture Mac (no Mac, no GPU, no network).

The word ban is the point of half of this file: a Mac's normal output speaks of memory and disks,
and never of what Poros does underneath.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.file_download  # noqa: F401 - imported on Linux: `sys.platform` is faked below
import huggingface_hub.hf_api  # noqa: F401
import pytest
import typer
from typer.testing import CliRunner

from poros import metal, recipes, runs
from poros.cli import app, contract
from poros.cli.context import Context
from poros.cli.main import app as typer_app
from poros.cli.schemas import Figure
from poros.core.errors import EXIT_CODES
from poros.core.refusal import Refusal
from poros.metal import store, table
from tests.conftest import (
    another_machine,
    hub_holding_configs,
    imported_on_this_platform,
    needs_ml,
    weighing,
)

M4 = table.plan_gpu("apple-m4-10gpu-16gb")
BASE7, INSTRUCT7, BASE72 = "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B"
TOOLS = {
    ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4",
    ("/usr/sbin/sysctl", "-n", "hw.memsize"): "17179869184",
    ("/usr/sbin/ioreg", "-rc", "AGXAccelerator", "-d", "1"): '"gpu-core-count" = 10',
    ("/usr/bin/sw_vers", "-buildVersion"): "26B123",
    (
        "/usr/bin/vm_stat",
    ): "(page size of 16384 bytes)\nPages free: 100.\nPages inactive: 300000.\n",
    ("/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"): '"IOPlatformUUID" = "ABC-123"',
}
BANNED = re.compile(
    r"resident|fit line|window|witness|validated|covered|unproven|cuda|vram|bitsandbytes|host ram",
    re.IGNORECASE,
)


@pytest.fixture(autouse=True)
def mac(tmp_path, monkeypatch):
    """This Mac: its tools, its pins, its two disks, an empty Hub cache, nothing written outside.
    And no other machine: no nvidia-smi and no CUDA driver answers, whatever card the host has."""
    imported_on_this_platform()  # each library picks its backend on this host, not the fake's
    another_machine(monkeypatch)

    import poros.plan.configs
    from poros.cli.schemas import Figure

    ssd = tmp_path / "Ext"
    ssd.mkdir()
    disks = [
        store.Volume(Path.home(), 245_075_738_624, 59_357_413_376),
        store.Volume(ssd, 499_898_105_856, 283_055_382_528),
    ]
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("POROS_OUTPUT", raising=False)
    monkeypatch.delenv(metal.DEV_MEMORY_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "mac_ver", lambda *_: ("27.2", ("", "", ""), "arm64"))
    monkeypatch.setattr(metal, "run_tool", lambda *cmd: TOOLS.get(cmd, ""))
    metal.device_class.cache_clear()
    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: metal.PINS.get(name) or real(name)
    )
    monkeypatch.setattr(store, "volumes", lambda: disks)
    monkeypatch.setattr(store.shutil, "disk_usage", lambda _: disks[1])
    monkeypatch.setattr(store, "_first_hub", lambda: (str(tmp_path / "default-hub"), False))
    monkeypatch.setattr(store, "use", lambda _: None)
    monkeypatch.setattr(store, "_holds", lambda *_: False)
    monkeypatch.setitem(recipes.MEASURED, "block_size", 1)
    monkeypatch.setitem(recipes.MEASURED, "gradient_accumulation_steps", 1)
    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=14.6, unit="GiB", source="hub_metadata", method="revision r"),
    )
    (tmp_path / "notes.jsonl").write_text(
        "".join(json.dumps({"text": "word " * n}) + "\n" for n in [40] * 19 + [400])
    )
    yield ssd
    metal.device_class.cache_clear()
    contract.reset()


@pytest.fixture
def ctx(tmp_path):
    return Context(cwd=tmp_path, fixed_gpu=M4, fixed_cached=frozenset(), fixed_interactive=False)


def as_json(argv, ctx, capsys):
    code = app.dispatch([*argv, "--json"], ctx)
    contract.reset()
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert len(lines) == 1, lines
    return code, json.loads(lines[0])


def as_person(argv, ctx, capsys):
    code = app.dispatch(argv, ctx)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def typer_run(*argv):
    result = CliRunner().invoke(typer_app, list(argv))
    return result.exit_code, result.output


# ---------------------------------------------------------------------------
# fit, plan
# ---------------------------------------------------------------------------


def _plan(model, ctx, **kw):
    from poros.cli.verbs import shared

    p = shared.plan_for(model, ctx, **kw)
    return {**p.model_dump(mode="json"), "seq_len": p.seq_len}


def test_the_plan_reads_the_table(ctx):
    from poros.cli.verbs import mac as mac_verb

    row = mac_verb.mac_row(BASE72, ctx)  # the length the table gives it
    p = _plan(BASE72, ctx, seq_len=row.seq)
    assert row.seq == 128 and p["gpu"]["unified"] is True and p["store"] == "ssd"
    assert p["fits"] is True and p["host_fit_basis"] == "one pool"
    assert p["peak_vram_allocated"]["source"] == "measured"


def test_a_same_shape_model_says_so_and_holds_the_base_row(ctx):
    p = _plan(INSTRUCT7, ctx)
    assert p["model"] == INSTRUCT7 and any("Qwen2.5-7B's" in w for w in p["warnings"])


def test_a_model_with_no_row_is_answered_not_refused(ctx):
    assert _plan("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", ctx)["support"] == "unsupported"


def test_past_the_floor_it_wont_fit(ctx):
    assert _plan(BASE72, ctx, seq_len=4096)["fits"] is False


def test_a_mac_has_the_same_grammar(ctx, capsys):
    code, obj = as_json(["train", BASE72, "on", "notes.jsonl"], ctx, capsys)
    assert code == 2 and obj["error"]["code"] == "E_USAGE"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="poros train asks its question only where torch is installed; "
                           "without it, it refuses with E_NO_TORCH first")
def test_a_question_on_a_mac_is_only_among_what_it_trains(ctx, capsys):
    code, obj = as_json(["train", "qwen", "32b"], ctx, capsys)
    (q,) = obj["questions"]
    assert [c["value"] for c in q["choices"]] == ["Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct",
                                                  "Qwen/Qwen3-32B"]
    code, obj = as_json(["train"], ctx, capsys)
    assert len(obj["questions"][0]["choices"]) == 22


def test_planning_for_a_card_from_a_mac_stays_cuda(tmp_path):
    ctx = Context(cwd=tmp_path, gpu_name="RTX 3090", fixed_cached=frozenset(),
                  fixed_interactive=False)
    p = _plan(BASE7, ctx)
    assert p["gpu"]["unified"] is False and p["seq_len"] == 512


# ---------------------------------------------------------------------------
# machine, models, check
# ---------------------------------------------------------------------------


def test_machine_is_ready_and_checks_the_mac(ctx, capsys):
    code, obj = as_json(["machine"], ctx, capsys)
    assert code == 0 and obj["verdict"] == "ready" and obj["schema"] == "poros.machine.v1"
    assert "can_train" not in obj  # it checks the Mac; what trains where is `poros recipes`
    checks = {c["name"]: c["value"] for c in obj["checks"]}
    assert checks["GPU"] == M4.name and checks["macOS"] == "27.2 (26B123)"
    assert checks["Unified memory"] == "17.2 GB, 12.7 GB of it for the GPU"
    assert checks["Disk Ext"] == "283.1 GB free" and checks["Models"] == (
        "DeepSeek, Gemma3, Gemma4, OLMoE, Qwen1.5, Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, Qwen3.8")
    assert "CUDA" not in checks and "bitsandbytes" not in checks


def test_a_pin_off_is_not_ready_with_the_code_for_it(ctx, capsys, monkeypatch):
    real = importlib.metadata.version

    def version(name):
        if name == "peft":
            raise importlib.metadata.PackageNotFoundError(name)
        return metal.PINS.get(name) or real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    code, obj = as_json(["machine"], ctx, capsys)
    assert code == EXIT_CODES["E_MACHINE_NOT_READY"] and obj["verdict"] == "not ready"
    assert {"name": "peft", "value": "not installed, needs 0.20.0", "status": "error"} in obj[
        "checks"
    ]


def test_a_build_other_than_the_tables_stays_ready_and_says_so_in_one_line(
    ctx, capsys, monkeypatch
):
    import dataclasses

    from poros.cli.words import models as words

    entry = words.by_id(BASE7)
    measured = dataclasses.replace(entry.metal[0], macos_build="26A5388g")
    monkeypatch.setattr(
        metal.machine, "catalog", lambda: [dataclasses.replace(entry, metal=(measured,))]
    )
    code, obj = as_json(["machine"], ctx, capsys)
    assert code == 0 and obj["verdict"] == "ready"
    assert {
        "name": "Times",
        "value": "measured on macOS 26A5388g; this Mac runs 26B123",
        "status": "info",
    } in obj["checks"]


@pytest.mark.parametrize(
    "version, verdict, value",
    [
        ("27.0", "not ready", "27.0 (26B123): update macOS"),
        ("13.6", "not ready", "13.6 (26B123): update macOS"),
    ],
)
def test_a_macos_that_cannot_run_is_named(ctx, capsys, monkeypatch, version, verdict, value):
    monkeypatch.setattr(platform, "mac_ver", lambda *_: (version, ("", "", ""), "arm64"))
    code, obj = as_json(["machine"], ctx, capsys)
    assert obj["verdict"] == verdict and code == 5
    assert {"name": "macOS", "value": value, "status": "error"} in obj["checks"]


def test_macos_14_is_a_mac_that_trains(ctx, capsys, monkeypatch):
    """torch 2.15.0 is built for 14.0 and its device needs only an Apple7 GPU."""
    monkeypatch.setattr(platform, "mac_ver", lambda *_: ("14.7", ("", "", ""), "arm64"))
    code, obj = as_json(["machine"], ctx, capsys)
    row = next(c for c in obj["checks"] if c["name"] == "macOS")
    assert code == 0 and row["status"] == "ok" and "update" not in row["value"]


def test_another_mac_is_ready_and_says_its_rows_are_derived(tmp_path, capsys, monkeypatch):
    plan_ctx = lambda: Context(cwd=tmp_path, fixed_cached=frozenset(), fixed_interactive=False)  # noqa: E731
    for chip, gb in (("Apple M4", 24), ("Apple M2", 16), ("Apple M1 Max", 64)):
        metal.device_class.cache_clear()
        monkeypatch.setitem(TOOLS, ("/usr/sbin/sysctl", "-n", "hw.memsize"), str(gb * 2**30))
        monkeypatch.setitem(TOOLS, ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"), chip)
        code, obj = as_json(["machine"], plan_ctx(), capsys)
        assert code == 0 and obj["verdict"] == "ready", obj
        said = {c["name"]: c for c in obj["checks"]}
        assert "derived" in said["This Mac"]["value"] and said["This Mac"]["status"] == "info"
        check = said["Self-check"]
        assert check["status"] == "info" and "not run yet" in check["value"]
        assert "OLMoE" in said["Models"]["value"] and "Qwen2.5" in said["Models"]["value"]
    metal.device_class.cache_clear()
    monkeypatch.setitem(TOOLS, ("/usr/sbin/sysctl", "-n", "hw.memsize"), "")
    code, obj = as_json(["machine"], plan_ctx(), capsys)
    assert code == 5 and obj["verdict"] == "not ready"


def test_an_intel_mac_has_no_gpu_to_train_on(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    ctx = Context(cwd=tmp_path, fixed_cached=frozenset(), fixed_interactive=False)
    code, text = as_person(["machine"], ctx, capsys)
    assert code == 5 and "Apple silicon" in text and not BANNED.search(text)


def test_models_lists_all_in_json_and_only_what_trains_for_a_person(ctx, capsys):
    code, obj = as_json(["models"], ctx, capsys)
    assert len(obj["models"]) == 23 and code == 0
    by_id = {m["id"]: m for m in obj["models"]}
    assert by_id[BASE7]["evidence"] != "absent" and "here" not in by_id[BASE72]
    assert by_id["Qwen/Qwen2.5-72B-Instruct"]["certified_here"] == ["mac-72b"]
    assert by_id[INSTRUCT7]["evidence"] == "same-shape"
    assert by_id["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]["evidence"] == "absent"
    code, text = as_person(["models"], ctx, capsys)
    assert "Qwen2.5-72B" in text and "Qwen3-32B" in text and "22 train here" in text
    assert "A3B" in text and "R1-Distill" not in text and "1 more do not" in text


def test_models_get_goes_to_the_store_and_refuses_what_a_mac_does_not_train(ctx, capsys, mac):
    code, obj = as_json(["models", "get", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"], ctx, capsys)
    assert code == 4 and obj["error"]["code"] == "E_GUARANTEE_UNAVAILABLE"
    code, obj = as_json(["models", "get", BASE72], ctx, capsys)
    (q,) = obj["questions"]
    assert str(mac / "poros" / "hub") in q["prompt"]
    assert f"41.2 GB to prepare it in {mac / 'poros'}" in q["prompt"]
    code, obj = as_json(["models", "get", BASE7], ctx, capsys)
    assert obj["questions"][0]["id"] == "store_path"


@pytest.mark.parametrize(
    "model, status, code",
    [(BASE7, "validated", 0),
     ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "not-on-this-device", 4)],
)
def test_check_exits_0_only_for_a_model_with_a_row(monkeypatch, model, status, code):
    monkeypatch.setattr("poros.preflight.arch.resolve_model_type", lambda m: ("qwen2", ""))
    exit_code, text = typer_run("check", model, "--json")
    assert exit_code == code and json.loads(text.splitlines()[-1])["status"] == status
    contract.reset()
    exit_code, text = typer_run("check", model)
    assert exit_code == code and not BANNED.search(text)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@pytest.fixture
def trained(monkeypatch):
    """Everything train hands the trainer, with the trainer itself left to the Mac."""
    import poros.cli.train_cmd as train_cmd

    calls = []

    def run(cfg, **kw):
        calls.append((cfg, kw))
        return {"final_loss": 1.0}

    monkeypatch.setattr(train_cmd, "run", run)
    return calls


@needs_ml
def test_a_big_model_goes_to_the_ssd_with_no_question_about_disks(
    ctx, capsys, mac, trained, tmp_path
):
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl"], ctx, capsys)
    (q,) = obj["questions"]
    assert q["id"] == "download" and str(mac / "poros") in q["prompt"]
    assert "--download" in q["choices"][0]["command"]
    assert "--store" not in q["choices"][0]["command"]  # the sentence, not the settled answer
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl", "--download"], ctx, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1"
    assert f"--store {mac / 'poros'}" not in obj["command"]
    assert "--seq-len 128" in obj["command"] and "--steps 20 " in obj["command"]
    assert trained[0][0].max_seq_length == 128


@needs_ml
def test_two_disks_that_qualify_ask_once_and_the_answer_is_a_flag(ctx, capsys, mac, trained):
    code, obj = as_json(["train", BASE7, "--data", "notes.jsonl"], ctx, capsys)
    (q,) = obj["questions"]
    assert q["id"] == "store_path" and q["default"] == str(mac / "poros")
    assert [c["label"] for c in q["choices"]] == ["Ext", "internal"]
    assert q["choices"][0]["note"] == "283 GB free"
    argv = [t for t in q["choices"][0]["command"].split()[1:] if t != "--json"]
    code, obj = as_json([*argv, "--download"], ctx, capsys)
    assert code == 0 and trained[0][0].store_path == str(mac / "poros")


@needs_ml
def test_a_run_keeps_its_files_on_a_store_off_the_internal_disk(
    ctx, capsys, mac, trained, monkeypatch
):
    monkeypatch.setattr(store, "run_files", lambda disk: Path(disk) / "runs")
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl", "--download"], ctx, capsys)
    assert code == 0 and Path(obj["output_dir"]).parent == mac / "poros" / "runs"
    assert trained[0][0].output_dir == obj["output_dir"]
    status = json.loads((runs.root() / obj["run"] / "status.json").read_text())
    assert status["output_dir"] == obj["output_dir"]


def test_settings_says_where_a_remembered_store_is(mac, monkeypatch):
    from poros.cli import questions, settings

    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    questions.remember("store", str(mac / "own"))
    facts = {fact[0]: fact[1] for fact in settings.settings_form().build({}).facts}
    assert facts["models"] == settings._short(mac / "own")  # as shown: ~ for the home folder


@needs_ml
def test_no_disk_with_room_is_e_disk_before_anything_is_asked(ctx, capsys, monkeypatch):
    monkeypatch.setattr(
        store, "volumes", lambda: [store.Volume(Path.home(), 245_000_000_000, 40_000_000_000)]
    )
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl"], ctx, capsys)
    assert code == 5 and obj["error"]["code"] == "E_DISK" and "186.6 GB" in obj["error"]["message"]
    code, text = as_person(["train", BASE72, "--data", "notes.jsonl"], ctx, capsys)
    assert code == 5 and "Not enough disk" in text and "internal 40 GB free" in text


def test_the_data_picks_the_length_and_the_screen_says_what_is_cut(ctx, capsys, mac, tmp_path):
    from poros.cli.verbs import train as verb
    from poros.cli.words.grammar import parse

    # 19 rows of 200 characters and one of 2,000: the 95th percentile is 50 tokens, seq 128.
    s = parse(["train", BASE7, "--data", "notes.jsonl", "--store", str(mac / "poros")])
    a = verb.inspect(verb.resolve(s, ctx).action, ctx)
    assert a.seq_len == 128 and a.steps == 20  # G=1: a step is one example
    assert a.kept == 0.95 and a.provenance["seq_len"].kind == "inferred"
    text = "".join(str(x) for line in verb.preview(a, 100) for x, _ in line)
    assert "95% of rows are no longer than 128 tokens; the rest are cut to it" in text
    assert "weights read from disk" in text and "no certified run of these settings" in text
    assert not BANNED.search(text)


@needs_ml
def test_an_off_table_length_runs_and_says_so(ctx, capsys, mac, trained):
    code, obj = as_json(
        [
            "train",
            BASE7,
            "--data",
            "notes.jsonl",
            "--seq-len",
            "1024",
            "--store",
            str(mac / "poros"),
            "--download",
        ],
        ctx,
        capsys,
    )
    assert code == 0 and trained[0][0].max_seq_length == 1024


@needs_ml
def test_an_agents_plan_says_its_rows_saves_and_why_there_is_no_eta(
    ctx, capsys, mac, trained, monkeypatch
):
    monkeypatch.setenv("POROS_AGENT", "1")
    asked = ["train", BASE7, "--data", "notes.jsonl", "--store", str(mac / "poros"), "--download"]
    for more in ([], ["--seq-len", "1024"]):
        code, obj = as_json([*asked, *more], ctx, capsys)
        plan = obj["plan"]
        row = table.pick(BASE7, "apple-m4-10gpu-16gb", seq=plan["seq_len"])
        assert code == 0 and plan["save_steps"] == row.save_steps, plan
        assert "eta_s" not in plan and plan["eta_from"].startswith("no estimate: "), plan
    assert "no certified run" in plan["eta_from"]  # the base 7B: no recipe of it is certified


@needs_ml
def test_synthetic_takes_the_tables_steps(ctx, capsys, mac, trained):
    code, obj = as_json(["train", BASE72, "--data", "synthetic", "--download"], ctx, capsys)
    assert code == 0 and "--steps 20 " in obj["command"]
    code, obj = as_json(
        ["train", BASE7, "--data", "synthetic", "--store", str(mac / "poros"), "--download"],
        ctx,
        capsys,
    )
    assert "--steps 200 " in obj["command"]


def test_train_refuses_what_a_mac_does_not_train(ctx, capsys):
    """A model whose config reads and that has no row on this Mac: not on this Mac."""
    code, obj = as_json(["train", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "--data",
                         "notes.jsonl"], ctx, capsys)
    assert code == 4 and obj["error"]["code"] == "E_GUARANTEE_UNAVAILABLE"
    assert "does not train on a Mac" in obj["error"]["message"]


@pytest.mark.parametrize("flags", [(), ("--unvalidated",)])
def test_a_model_no_cache_holds_with_no_hub_is_the_network_refusal(ctx, capsys, tmp_path,
                                                                   flags):
    """One whose config.json no cache here holds, and no Hub to ask: what is missing is the
    network, said as off a Mac, in the verb form and the config form alike."""
    code, obj = as_json(["train", "org/not-in-the-catalog", "--data", "notes.jsonl", *flags],
                        ctx, capsys)
    config_code, text = typer_run("train", str(_config(tmp_path, "org/not-in-the-catalog")),
                                  *flags, "--json")
    config = json.loads(text.splitlines()[-1])
    contract.reset()
    for code, obj in ((code, obj), (config_code, config)):
        assert (code, obj["error"]["code"]) == (5, "E_NETWORK"), obj
        assert obj["error"]["message"].startswith(
            "org/not-in-the-catalog is not in the local cache, and Hub access is off here "
            "(HF_HUB_OFFLINE is set)"), obj


def _config(tmp_path, model):
    path = tmp_path / "run.yaml"
    path.write_text(f"model_name_or_path: {model}\ndataset: {tmp_path / 'notes.jsonl'}\n"
                    f"output_dir: {tmp_path / 'out'}\n")
    return path


# -- check says what train then does: each kind of model, as JSON and drawn --------------------


def _check(model):
    """`poros check` as JSON and drawn: (exit, status, next, the drawn screen)."""
    code, text = typer_run("check", model, "--json")
    said = json.loads(text.splitlines()[-1])
    contract.reset()
    drawn_code, drawn = typer_run("check", model)
    contract.reset()
    assert drawn_code == code
    return code, said["status"], said["next"], " ".join(drawn.replace("│", " ").split())


@needs_ml
def test_check_says_a_family_this_mac_trains_with_unvalidated_and_train_does(
        ctx, capsys, mac, trained, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_holds", lambda s, m: s == mac / "poros")
    monkeypatch.setattr("poros.plan.configs.download_size",
                        lambda cfg: Figure(value=0.0, unit="GiB", source="local"))
    here = _local(tmp_path / "mine", "llama", 1.0, monkeypatch)
    code, status, then, drawn = _check(here)
    assert (code, status, then) == (4, "not-in-registry", [f"poros arch try {here}"])
    assert "Trains on this Mac with --unvalidated" in drawn
    assert "Continue with: poros arch try" in drawn  # a long path wraps on the screen
    code, obj = as_json(["train", here, "--data", "notes.jsonl", "--unvalidated"], ctx, capsys)
    assert code == 0 and obj["support"] == "user_asserted", obj


def test_check_says_why_no_machine_trains_a_family_and_train_refuses_it(ctx, capsys, mac,
                                                                        tmp_path):
    from tests.test_gemma4_variants import E2B

    here = tmp_path / "e2b"
    here.mkdir()
    (here / "config.json").write_text(json.dumps(E2B))
    code, status, then, drawn = _check(str(here))
    assert (code, status, then) == (4, "unsupported", [])
    assert "Unsupported" in drawn and "per-layer input" in drawn and "Continue with" not in drawn
    code, obj = as_json(["train", str(here), "--data", "notes.jsonl", "--store", str(mac / "poros"),
                         "--unvalidated", "--yes"], ctx, capsys)
    assert code == 4 and obj["error"]["code"] == "E_ARCH_UNSUPPORTED", obj


@needs_ml
def test_check_says_a_validated_family_trains_here_and_train_takes_it(ctx, capsys, mac,
                                                                     monkeypatch):
    monkeypatch.setattr("poros.preflight.arch.resolve_model_type", lambda m: ("qwen2", ""))
    code, status, then, drawn = _check(BASE7)
    assert (code, status, then) == (0, "validated", [])
    assert "Trains on this Mac" in drawn and "Continue with: poros recipes" in drawn
    code, obj = as_json(["train", BASE7, "--data", "notes.jsonl"], ctx, capsys)
    assert code == 2 and obj["schema"] == "poros.needs_input.v1", obj  # which disk: no refusal


def test_check_says_a_model_with_no_row_here_does_not_train_and_train_refuses_it(ctx, capsys,
                                                                                 monkeypatch):
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    monkeypatch.setattr("poros.preflight.arch.resolve_model_type", lambda m: ("qwen2", ""))
    code, status, then, drawn = _check(model)
    assert (code, status, then) == (4, "not-on-this-device", [])
    assert "Not on this Mac" in drawn and "Continue with: poros models" in drawn
    for flags in ((), ("--unvalidated",)):
        code, obj = as_json(["train", model, "--data", "notes.jsonl", *flags], ctx, capsys)
        assert code == 4 and obj["error"]["code"] == "E_GUARANTEE_UNAVAILABLE", obj


@needs_ml
@pytest.mark.parametrize("store", ["ram", "ssd"])
def test_the_command_line_takes_a_dense_models_store_as_a_config_file_does(ctx, capsys, mac,
                                                                          trained, store):
    """store=ram holds every layer, store=ssd reads them from the 4-bit copy: on a Mac a dense
    model takes either, as `store:` in a config file does, in the folder --store names."""
    code, obj = as_json(["train", BASE7, "--data", "notes.jsonl", "--store", str(mac / "poros"),
                         "--download", f"store={store}"], ctx, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1", obj
    cfg = trained[0][0]
    assert (cfg.store, cfg.store_path) == (store, str(mac / "poros"))


@needs_ml
def test_store_ram_for_a_model_this_mac_reads_from_the_ssd_is_said_not_to_fit_first(
        ctx, capsys, mac, trained, tmp_path):
    """Qwen2.5-32B held in memory is about 25 GiB: said before anything loads, with what fits."""
    model = "Qwen/Qwen2.5-32B"
    assert table.pick(model, table.class_of(M4.name)).tier == "ssd"
    code, obj = as_json(["train", model, "--data", "notes.jsonl", "--store", str(mac / "poros"),
                         "--download", "--seq-len", "128", "store=ram"], ctx, capsys)
    (q,) = obj["questions"]
    assert code == 2 and q["id"] == "wont_fit" and not trained, obj
    assert "from RAM" in q["prompt"] and "drop store=ram" in q["prompt"]
    assert "GiB" not in q["prompt"] and not trained


def test_a_cuda_card_named_from_a_mac_is_planned_at_a_cuda_runs_settings(
        ctx, capsys, tmp_path, monkeypatch):
    """A plan for a 12GB card on a Mac answers for that card's run (block 4, 8 rows a step), as
    it answers on Linux, not at the Mac's own presets. The model's config.json is
    here (the test's own cache): it sizes that run's path on top of the sequential loop's."""
    from poros.recipes import CUDA_MEASURED

    hub_holding_configs(monkeypatch, tmp_path / "hub", INSTRUCT7)

    assert (recipes.MEASURED["block_size"], recipes.MEASURED["gradient_accumulation_steps"]) == (
        1, 1)  # this Mac's, as the fixture makes them
    assert (CUDA_MEASURED["block_size"], CUDA_MEASURED["gradient_accumulation_steps"]) == (4, 8)
    import math

    from poros.cli.verbs import shared
    from poros.plan.estimate import _RESERVED_SLACK

    named = Context(cwd=ctx.cwd, gpu_name="12GB", fixed_cached=frozenset(),
                    fixed_interactive=False)
    p = shared.plan_for(INSTRUCT7, named, download=True)
    # the CUDA run's estimate (block 4, 8 rows a step): with the allocator's slack and the CUDA
    # context (1 GiB) it needs 11.6 GiB, as on Linux; never shown, it decides the check
    peak = p.peak_vram_allocated
    assert p.gpu is not None and not p.gpu.unified and peak.source == "estimated"
    assert math.ceil(round((peak.value * _RESERVED_SLACK + 1.0) * 10, 6)) / 10 == 11.6


def _local(path, arch, gb, monkeypatch):
    """A checkpoint folder of ``arch`` whose shards weigh ``gb``, taking no disk."""
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": arch}))
    weighing(monkeypatch, path, gb)
    return str(path)


@needs_ml
def test_a_local_folder_of_a_family_this_mac_trains_takes_no_flags(
    ctx, capsys, mac, trained, tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "_holds", lambda s, m: s == mac / "poros")
    monkeypatch.setattr("poros.plan.configs.download_size",
                        lambda cfg: Figure(value=0.0, unit="GiB", source="local"))
    here = _local(tmp_path / "mine", "qwen3", 7.0, monkeypatch)
    code, obj = as_json(["train", here, "--data", "notes.jsonl"], ctx, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1", obj
    cfg = trained[0][0]
    assert cfg.model_name_or_path == here
    assert table.fill(cfg, "apple-m4-10gpu-16gb") == table.pick("Qwen/Qwen3-32B",
                                                               "apple-m4-10gpu-16gb")
    assert (cfg.max_seq_length, cfg.gradient_accumulation_steps, cfg.store) == (128, 1, "ssd")


@needs_ml
def test_a_local_folder_of_a_family_no_one_validated_asks_then_trains_as_the_nearest(
    ctx, capsys, mac, trained, tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "_holds", lambda s, m: s == mac / "poros")
    monkeypatch.setattr("poros.plan.configs.download_size",
                        lambda cfg: Figure(value=0.0, unit="GiB", source="local"))
    here = _local(tmp_path / "mine", "llama", 1.0, monkeypatch)
    code, obj = as_json(["train", here, "--data", "notes.jsonl"], ctx, capsys)
    assert code == 2 and "model_type llama" in obj["questions"][0]["prompt"], obj
    assert "--unvalidated" in obj["questions"][0]["choices"][0]["command"].split()
    assert not trained
    code, obj = as_json(["train", here, "--data", "notes.jsonl", "--unvalidated"], ctx, capsys)
    assert code == 0 and obj["support"] == "user_asserted", obj
    cfg, kw = trained[0]
    assert kw["unvalidated"] is True and cfg.model_name_or_path == here
    from poros.core.arch_registry import arch_status

    assert arch_status("llama") is None  # for that command alone


SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
# SmolLM2-135M's config: a llama, which the registry does not name
LLAMA = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 576,
         "intermediate_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 9,
         "num_key_value_heads": 3, "vocab_size": 49152, "max_position_embeddings": 8192,
         "rms_norm_eps": 1e-5, "rope_theta": 100000, "tie_word_embeddings": True}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """The Hub as `arch` reads it, with no token: a test says what each id's config is, and the
    weights' format; an empty Hub cache; a download the size the Hub would list."""
    import huggingface_hub.constants

    import poros.plan.configs
    from poros.cli.verbs import arch

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    said = {SMOL: arch._Seen(raw=dict(LLAMA), counts={"BF16": 134_515_008}, download_gib=0.25)}
    monkeypatch.setattr(arch, "_from_hub", lambda model: said.get(model, arch._Seen()))
    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=0.0, unit="GiB", source="local") if Path(cfg.model_name_or_path).exists() else
        Figure(value=0.25, unit="GiB", source="hub_metadata", method="revision r"))
    return said


def _smol(path):
    """A folder of SmolLM2's files: its config, and a shard that says it is bf16."""
    path.mkdir()
    (path / "config.json").write_text(json.dumps(LLAMA))
    _shard(path, "BF16")
    return str(path)


def _shard(path, dtype):
    import struct

    header = json.dumps({"w": {"dtype": dtype, "shape": [64, 64], "data_offsets": [0, 8192]}})
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header.encode())


@needs_ml
def test_a_hub_id_of_a_family_no_one_validated_asks_what_a_catalog_model_does_then_trains(
    ctx, capsys, mac, trained, hub
):
    code, obj = as_json(["train", SMOL, "--data", "notes.jsonl"], ctx, capsys)
    assert code == 2 and [q["id"] for q in obj["questions"]] == ["store_path"], obj  # which disk
    asked = ["train", SMOL, "--data", "notes.jsonl", "--store", str(mac / "poros")]
    code, obj = as_json(asked, ctx, capsys)
    # each says what it confirms, and one more answers both in one command
    assert [q["id"] for q in obj["questions"]] == ["unvalidated", "check", "download", "all"]
    confirm, check, download, every = obj["questions"]
    assert "model_type llama" in confirm["prompt"] and confirm["choices"][0]["value"] == (
        "--unvalidated")
    assert check["choices"][0]["command"].startswith("poros arch try")
    # decimal GB, as Poros counts disk, to two places under 1 GB: 0.25 GiB is 0.27 GB
    assert download["prompt"].startswith("Download SmolLM2-135M-Instruct? 0.27 GB from")
    assert download["choices"][0]["value"] == "--download" and not trained
    assert every["default"] is None and "unvalidated, download" in every["prompt"]
    (both,) = every["choices"]
    words = [w for w in shlex.split(both["command"])[1:] if w != "--json"]
    assert words == [*asked, "--unvalidated", "--download"]
    code, obj = as_json([*asked, "--unvalidated"], ctx, capsys)
    assert code == 2 and [q["choices"][0]["value"] for q in obj["questions"]] == ["--download"]
    code, obj = as_json(words, ctx, capsys)
    assert code == 0 and (obj["support"], obj["model_type"]) == ("user_asserted", "llama"), obj
    cfg, kw = trained[0]
    assert kw["unvalidated"] is True and cfg.model_name_or_path == SMOL
    assert cfg.store_path == str(mac / "poros")
    near = table.pick("Qwen/Qwen2.5-0.5B", "apple-m4-10gpu-16gb")  # the nearest of its kind
    assert (cfg.max_seq_length, cfg.store) == (near.seq, near.tier)
    assert table.pick(SMOL, "apple-m4-10gpu-16gb") is None  # asserted for that command alone


@needs_ml
def test_a_folder_and_a_hub_id_of_a_new_family_train_alike(
    ctx, capsys, mac, trained, hub, tmp_path
):
    here = _smol(tmp_path / "smol")
    tail = ["--data", "notes.jsonl", "--store", str(mac / "poros"), "--unvalidated", "--download"]
    said = [as_json(["train", model, *tail], ctx, capsys) for model in (SMOL, here)]
    assert [code for code, _ in said] == [0, 0], said
    assert {(o["support"], o["model_type"]) for _, o in said} == {("user_asserted", "llama")}
    (hub_cfg, _), (folder_cfg, _) = trained
    fields = ("max_seq_length", "gradient_accumulation_steps", "save_steps", "store", "block_size",
              "eval_split", "max_steps", "store_path")
    assert [getattr(hub_cfg, f) for f in fields] == [getattr(folder_cfg, f) for f in fields]


@needs_ml
def test_arch_on_a_mac_says_why_no_flag_trains_one(ctx, capsys, hub):
    from poros.cli.verbs import arch

    hub["someone/model"] = arch._Seen(raw=dict(LLAMA), counts={"F16": 10**8})
    code, obj = as_json(["arch", "someone/model"], ctx, capsys)
    assert code == 0 and "a Mac makes NF4 from bf16 alone" in json.dumps(obj), obj
    assert "fits_here" not in obj  # arch says what a model is; recipes say what was measured


@needs_ml
def test_arch_try_on_a_mac_points_a_new_family_that_passed_at_train(
    ctx, capsys, hub, monkeypatch
):
    """Off a Mac a pass points at prove; on one, where prove's gates do not run, at train. Neither
    carries the consent flag: train asks for it."""
    from poros.cli.verbs import arch
    from poros.metal import tiny

    monkeypatch.setattr(tiny, "prove", lambda raw, root: {"ok": True, "status": "user_asserted"})
    code, obj = as_json(["arch", "try", SMOL], ctx, capsys)
    assert code == 0 and obj["tried"]["ok"] and obj["tried"]["route"] == "mac", obj
    assert obj["next"] == [f"poros train {SMOL} --data <file>"]  # the data to fill in
    assert not {"--unvalidated", "--download", "--yes"} & set(obj["next"][0].split())
    monkeypatch.setattr(tiny, "prove", lambda raw, root: {"ok": False, "why": "differs"})
    code, obj = as_json(["arch", "try", SMOL], ctx, capsys)
    assert code == 0 and obj["next"] == []
    hub["someone/model"] = arch._Seen(raw=dict(LLAMA), counts={"F16": 10**8})
    monkeypatch.setattr(tiny, "prove", lambda raw, root: {"ok": True})
    code, obj = as_json(["arch", "try", "someone/model"], ctx, capsys)
    assert code == 0 and obj["next"] == []  # a release a Mac does not pack: no train to offer


@needs_ml
def test_arch_try_on_a_macs_train_carries_its_data_as_the_gate_fills_it(
    ctx, capsys, hub, monkeypatch
):
    """`next` is a train command with --data: its data is a fill-in, `<file>`, as the guide says.
    The agents gate fills it in as a person would, and then the command is one `poros agents`
    reads, every flag known."""
    from types import SimpleNamespace as Bag

    from poros.cli.verbs import docs
    from poros.metal import tiny

    monkeypatch.setattr(tiny, "prove", lambda raw, root: {"ok": True, "status": "user_asserted"})
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "scripts" / "metal"))
    gate = __import__("agents_gate")
    monkeypatch.setenv("POROS_AGENT", "1")
    code, obj = as_json(["arch", "try", SMOL], ctx, capsys)
    assert obj["next"] == [f"poros agents train {SMOL} --data <file>"]
    assert gate.problem_with(Bag(a=Bag(model=SMOL), finished=None), obj["next"][0], None) == ""
    guide = " ".join(docs._files()[docs.GUIDE].split())
    assert "ready to run but for a fill-in: `<file>` is your data file" in guide


@needs_ml
def test_a_plan_on_a_mac_says_a_new_familys_numbers_are_arithmetic(ctx, capsys, hub):
    from poros.cli.verbs import mac as mac_verb
    from poros.cli.verbs import train
    from poros.core.arch_registry import one_command

    with one_command():  # the family's assertion ends with this command, as a CLI call's does
        train.new_family(SMOL, ctx)  # as train's card reads a family no one validated
        row = mac_verb.mac_row(SMOL, ctx)
        p = _plan(SMOL, ctx, download=True, seq_len=row.seq)
    assert p["support"] == "user_asserted" and p["fits"] is None, p
    assert "by arithmetic on its config" in p["fit_basis"] and "not measured" in p["fit_basis"]
    assert any("derived from its config" in w for w in p["warnings"])
    assert p["peak_vram_allocated"]["source"] == "unknown" and row.seq == 256


@needs_ml
@pytest.mark.parametrize("seen, words", [
    ({"counts": {"F8_E4M3": 10**8}}, "ships F8_E4M3"),
    ({"counts": {"I8": 10**8}}, "ships I8"),
    ({"counts": {"F32": 10**8}}, "a Mac makes NF4 from bf16 alone"),
    ({"raw": {"model_type": "evil_net", "auto_map": {"AutoModelForCausalLM": "evil.Net"}}},
     "its own code"),
    ({"raw": {"model_type": "not_a_type"}}, "no class"),
    ({"raw": {**LLAMA, "num_local_experts": 8}}, "names experts"),
    ({"gated": True}, "gated"),
])
def test_on_a_mac_what_no_flag_trains_by_hub_id_is_refused(ctx, capsys, hub, seen, words):
    from poros.cli.verbs import arch

    hub["someone/model"] = arch._Seen(**{"raw": dict(LLAMA), **seen})
    code, obj = as_json(["train", "someone/model", "--data", "notes.jsonl", "--unvalidated"],
                        ctx, capsys)
    assert code == 4 and obj["error"]["code"] == "E_ARCH_UNSUPPORTED", obj
    assert words in obj["error"]["message"] and "evil" not in obj["error"]["message"]


@needs_ml
def test_on_a_mac_a_folder_that_is_not_bf16_is_refused_as_its_hub_id_is(ctx, capsys, tmp_path):
    here = tmp_path / "smol"
    here.mkdir()
    (here / "config.json").write_text(json.dumps(LLAMA))
    _shard(here, "F16")
    code, obj = as_json(["train", str(here), "--data", "notes.jsonl", "--unvalidated"], ctx,
                        capsys)
    assert code == 4 and "this release ships F16" in obj["error"]["message"], obj


@needs_ml
def test_a_model_already_packed_asks_for_no_download(ctx, capsys, mac, trained):
    packed = mac / "poros" / f"{BASE72.replace('/', '--')}-abcd1234"
    packed.mkdir(parents=True)
    (packed / "model.sss.idx").touch()
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl"], ctx, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1", obj


@needs_ml
def test_model_list_says_cached_for_what_this_mac_holds_as_train_reads_it(tmp_path, capsys, mac,
                                                                           trained):
    """`cached` in `model list` is whether this machine holds what a run needs, and on a Mac that
    is a store's pack or a store's Hub cache."""
    def listed() -> dict:
        ctx = Context(cwd=tmp_path, fixed_gpu=M4, fixed_interactive=False)  # nothing fixed cached
        return {row["id"]: row["cached"] for row in as_json(["models"], ctx, capsys)[1]["models"]}

    assert not any(listed().values())
    packed = mac / "poros" / "Qwen--Qwen3-0.6B-abcd1234"
    packed.mkdir(parents=True)
    (packed / "model.sss.idx").touch()  # its pack, finished, and no Hub weights beside it
    _cached(mac / "poros" / "hub", "Qwen/Qwen2.5-0.5B")  # the weights, in that store's Hub cache
    said = listed()
    assert said["Qwen/Qwen3-0.6B"] is True and said["Qwen/Qwen2.5-0.5B"] is True
    assert [k for k, v in said.items() if v] == ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"]
    ctx = Context(cwd=tmp_path, fixed_gpu=M4, fixed_interactive=False)
    code, obj = as_json(["train", "Qwen/Qwen3-0.6B", "--data", "notes.jsonl", "--store",
                         str(mac / "poros")], ctx, capsys)  # the disk this fixture cannot find
    assert code == 0 and obj["schema"] == "poros.train.v1", obj  # no download asked: they agree


@needs_ml
def test_another_mac_trains_on_rows_derived_for_it_with_no_flag(tmp_path, capsys, mac, trained):
    other = table.plan_gpu("apple-m4-10gpu-24gb")
    ctx = Context(cwd=tmp_path, fixed_gpu=other, fixed_cached=frozenset(), fixed_interactive=False)
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl", "--download"], ctx, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1", obj


def test_a_mac_too_small_for_a_model_says_so_in_numbers(tmp_path, capsys, mac, trained):
    small = table.plan_gpu("apple-m1-7gpu-8gb")
    ctx = Context(cwd=tmp_path, fixed_gpu=small, fixed_cached=frozenset(), fixed_interactive=False)
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl", "--download"], ctx, capsys)
    assert code == 4 and "is too large for this Mac's 8 GB" in obj["error"]["message"], obj
    assert "even reading its weights from the SSD" in obj["error"]["message"]


def test_train_on_this_mac_is_not_another_card(ctx, capsys, trained):
    code, obj = as_json(
        ["train", BASE72, "--data", "notes.jsonl", "--gpu", "RTX 4090", "--download"], ctx, capsys
    )
    assert code == 2 and obj["error"]["code"] == "E_USAGE"


@needs_ml
def test_a_run_that_lost_its_volume_says_which(ctx, capsys, mac, trained):
    code, obj = as_json(
        ["train", BASE72, "--data", "notes.jsonl", "--store", "/gone/disk/poros", "--download"],
        ctx,
        capsys,
    )
    assert (
        code == 5 and obj["error"]["code"] == "E_DISK" and "/gone/disk" in obj["error"]["message"]
    )



# ---------------------------------------------------------------------------
# the cards
# ---------------------------------------------------------------------------


def card(ctx, *words):
    from poros.cli import sheets
    from poros.cli.words.grammar import parse

    form = sheets.train_form(sheets._likeliest(parse(["train", *words]), ctx), ctx)
    return form.build(form.values)


def said(view) -> str:
    rows = [(f.label, f.shown, f.source, f.hint) for f in view.fields]
    return " ".join(str(x) for row in rows for x in row) + " ".join(view.result or ())


def test_the_train_card_opens_ready_on_words_with_nothing_to_choose(ctx, mac, monkeypatch):
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setattr(poros.plan.configs, "download_size",
                        lambda cfg: Figure(value=0.0, unit="GiB", source="hub_metadata"))
    monkeypatch.setattr(store, "_holds", lambda at, _: "Ext" in str(at))
    view = card(ctx, "qwen", "0.5b", "--data", "notes.jsonl")
    assert view.ready and not view.error, said(view)
    assert view.fields[0].shown == "Qwen2.5-0.5B-Instruct"
    assert not {"--store", "experts", "--block-size"} & {f.key for f in view.fields}
    assert not BANNED.search(said(view)) and "block" not in said(view)


def test_two_disks_are_the_only_new_row_a_mac_ever_shows(ctx, mac, monkeypatch):
    view = card(ctx, BASE7, "--data", "notes.jsonl")
    assert next(f for f in view.fields if f.key == "--store").required and not view.ready
    assert [o.label for o in next(f for f in view.fields if f.key == "--store").options()] == [
        "Ext",
        "internal",
    ]
    view = card(ctx, BASE7, "--data", "notes.jsonl", "--store", str(mac / "poros"))
    assert next(f for f in view.fields if f.key == "--store").shown.endswith("/poros")


def test_more_offers_only_what_a_mac_honours_and_lengths_from_its_table(ctx):
    view = card(ctx, BASE7.replace("7B", "72B"), "--data", "notes.jsonl")
    fields = {f.key: f for f in view.fields}
    offered = fields["more"].options()
    names = {o.value for o in offered}
    assert {"learning_rate=", "warmup_steps=", "--rank"} <= names
    assert not names & {"quantization=", "torch_dtype=", "gradient_accumulation_steps=",
                        "per_device_batch_size=", "attn_implementation=", "enable_prefetch="}
    assert "memory & speed" not in {o.group for o in offered}
    # checkpoints are the card's own row, not one of "more"
    assert "save_steps=" not in names and fields["save_steps="].label == "checkpoints"
    assert [o.value for o in fields["--seq-len"].options()] == ["128"]


def test_a_mac_offers_only_recipes_it_would_run(ctx):
    from poros.cli import sheets

    names = {i.text.split("  ")[0] for i in sheets.listings.recipes_listing(ctx).items}
    # the Mac's own; a CUDA one (block 4, a path, the SSD expert store) is not offered
    assert names == {"72B on a Mac", "32B on a Mac", "MoE on a Mac", "7B on a Mac",
                     "First run on a Mac", "Better answers on a Mac", "Support answers on a Mac",
                     "Your voice on a Mac"}


def test_lists_are_what_this_mac_trains_and_a_first_run_is_the_small_one(ctx):
    from poros.cli import sheets

    assert len(sheets.listings.models_listing(ctx).items) == 22
    assert sheets.fit.recommended(ctx)[0] == "Qwen2.5-0.5B-Instruct"


def test_an_instruct_model_is_recommended_over_a_downloaded_base(tmp_path):
    """A base model trained on chats here never learned to end an answer; its Instruct did."""
    from poros.cli import sheets

    have_base = Context(cwd=tmp_path, fixed_gpu=M4, fixed_cached=frozenset({"Qwen/Qwen2.5-0.5B"}),
                        fixed_interactive=False)
    assert sheets.fit.recommended(have_base)[0] == "Qwen2.5-0.5B-Instruct"


def test_chats_on_a_base_model_say_which_model_ends_its_answers(ctx, mac):
    from poros.cli.words.models import base_on_chats, instruct_twin

    assert instruct_twin("Qwen/Qwen2.5-0.5B").short == "Qwen2.5-0.5B-Instruct"
    assert instruct_twin("Qwen/Qwen2.5-0.5B-Instruct") is None and base_on_chats("nope/x") == ""
    chat = [{"role": "user", "content": "pack for a hike?"},
            {"role": "assistant", "content": "a map"}]
    (ctx.cwd / "chats.jsonl").write_text("".join(json.dumps({"messages": chat}) + "\n"
                                                 for _ in range(20)))

    def note(model, data):
        view = card(ctx, model, "--data", data)
        return next(f for f in view.fields if f.key == "subject").source

    said = "base: on chats Qwen2.5-0.5B-Instruct ends its answers"
    assert note("Qwen2.5-0.5B", "chats.jsonl") == said
    assert "ends its answers" not in note("Qwen2.5-0.5B-Instruct", "chats.jsonl")
    assert "ends its answers" not in note("Qwen2.5-0.5B", "notes.jsonl")  # text: no turns to end


def test_a_card_that_will_not_fit_says_why_in_red(ctx, mac):
    view = card(ctx, BASE72, "--data", "notes.jsonl", "--seq-len", "4096")
    assert view.result[1] == "won't fit" and view.result_style == "e"
    assert "GiB" not in view.result[2]  # the decision, without the planner's figure


@needs_ml
def test_checkpoints_that_would_fill_their_disk_need_a_yes(ctx, capsys, mac, trained, monkeypatch):
    usage = SimpleNamespace(total=500_000_000_000, free=9_000_000_000)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage)
    code, obj = as_json(["train", BASE72, "--data", "notes.jsonl", "--download"], ctx, capsys)
    (q,) = obj["questions"]
    assert "about 10 GB of checkpoints" in q["prompt"] and "--force" in q["choices"][0]["command"]
    code, obj = as_json(
        ["train", BASE72, "--data", "notes.jsonl", "--download", "--force"], ctx, capsys
    )
    assert code == 0 and obj["schema"] == "poros.train.v1"
    code, obj = as_json(
        ["train", BASE72, "--data", "notes.jsonl", "--download", "save_total_limit=1"], ctx, capsys
    )
    assert code == 0  # 5 GB fits


def test_a_low_disk_among_others_stops_nothing_and_all_low_says_so(ctx, capsys, monkeypatch):
    low = store.Volume(Path.home(), 245_000_000_000, 14_000_000_000)
    roomy = store.Volume(Path("/Ext"), 500_000_000_000, 45_000_000_000)
    monkeypatch.setattr(store, "volumes", lambda: [low, roomy])
    code, obj = as_json(["machine"], ctx, capsys)
    assert code == 0 and obj["verdict"] == "ready"
    monkeypatch.setattr(store, "volumes", lambda: [low])
    code, text = as_person(["machine"], ctx, capsys)
    assert "needs disk space" in text and "Disk internal" not in text


def test_the_export_card_opens_on_what_a_mac_can_do_for_that_run(ctx, tmp_path):
    from poros.cli import sheets
    from poros.cli.words.grammar import parse

    folder, _ = finished_run(tmp_path)
    assert sheets.export.export_form(parse(["export"]), ctx).values["--format"] == "merged"
    runs.update(folder, model=BASE72)
    (folder / "action.json").write_text(json.dumps({"model": BASE72, "name": "n"}))
    assert sheets.export.export_form(parse(["export"]), ctx).values["--format"] == "adapter"


def test_eval_with_no_run_says_to_train_one(ctx, capsys):
    code, obj = as_json(["eval"], ctx, capsys)
    assert code == 2 and obj["error"]["code"] == "E_NOT_FOUND"
    assert obj["error"]["hint"] == "poros train"  # no run here: nothing to score or export


# ---------------------------------------------------------------------------
# eval on a Mac: the run's own engine, under the gate, lock and guard a run has
# ---------------------------------------------------------------------------

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"


def mac_run(tmp_path, model, *, run_id="1107-0100-c0de"):
    out = tmp_path / f"out-{run_id}"
    out.mkdir()
    (out / "adapter_model.safetensors").write_bytes(b"weights")
    (out / "adapter_config.json").write_text("{}")
    folder = runs.create(run_id, {"model": model, "name": "n"}, output_dir=str(out))
    runs.update(folder, state="succeeded", finished_at=runs.now())
    (out / "config.resolved.json").write_text(json.dumps(
        {"model_name_or_path": model, "dataset": str(tmp_path / "chats.jsonl"),
         "trained_with": "sft", "data_shape": "chats", "gradient_accumulation_steps": 1}))
    return folder, out


def test_a_mac_eval_holds_this_mac_and_passes_the_gate_a_run_does(ctx, capsys, tmp_path,
                                                                   monkeypatch):
    """No full-precision load: the same gate as a run, with the Mac held, before anything loads."""
    from poros import jobs
    from poros.cli import train_cmd
    from poros.cli.verbs import evaluate_pass

    folder, _ = mac_run(tmp_path, SMALL)
    seen = []
    monkeypatch.setattr(metal, "require", lambda cfg, check=True: seen.append(("require", check)))
    monkeypatch.setattr(train_cmd, "mac_ready", lambda: seen.append(("ready", None)))

    def tokenizer(cfg):
        seen.append(("held", jobs.hold_machine() is None, cfg.output_dir != str(folder)))
        raise Refusal("E_USAGE", "Stopped here", "the gate was passed")

    monkeypatch.setattr(evaluate_pass, "tokenizer", tokenizer)
    code, obj = as_json(["eval", folder.name], ctx, capsys)
    assert obj["error"]["message"] == "the gate was passed", obj
    assert seen == [("require", False), ("ready", None), ("held", True, True)]
    assert jobs.hold_machine() is not None  # let go of once it ends


def test_a_finished_mac_run_offers_eval_first(tmp_path, mac, monkeypatch):
    from poros.cli.verbs import runs_report

    monkeypatch.setenv("POROS_AGENT", "1")
    folder, out = mac_run(tmp_path, BASE72)
    r = next(x for x in runs.all_runs() if x.id == folder.name)
    export = f"poros export {folder.name} --format adapter --output ./adapter-{folder.name[-4:]}"
    # a chats run that scored no held-out rows has none for eval to score: export comes first
    assert runs_report.offers(r)[0] == export
    (out / "events.jsonl").write_text(json.dumps(  # it held rows out, and scored them
        {"event_type": "eval", "step": 0, "data": {"eval_loss": 2.5, "rows": 4}}) + "\n")
    assert runs_report.offers(r)[:2] == [f"poros eval {folder.name}", export]


def test_a_merge_a_mac_cannot_make_names_the_runs_own_adapter(ctx, capsys, tmp_path):
    folder, _ = finished_run(tmp_path)
    runs.update(folder, model=BASE72)
    (folder / "action.json").write_text(json.dumps({"model": BASE72, "name": "n"}))
    runs.create("1102-2359-ffff", {"model": BASE7, "name": "later"},
                output_dir=str(tmp_path / "later"))
    code, obj = as_json(["export", folder.name, "--format", "merged"], ctx, capsys)
    assert code == 4 and obj["error"]["code"] == "E_EXPORT_UNSUPPORTED"
    assert obj["error"]["hint"] == f"poros export {folder.name} --format adapter"


def test_a_mac_run_ends_on_scoring_itself_whatever_its_size(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from poros.cli import train_cmd
    from poros.cli.verbs import train_run

    folder = runs.create("1102-2300-abcd", {"model": BASE7, "name": "n"})
    seen = {}
    monkeypatch.setattr(train_cmd, "run", lambda cfg, **how: seen.update(how) or {})
    a = SimpleNamespace(name="n", force=False, unvalidated=None, command=lambda: "poros train")
    train_run._train(a, SimpleNamespace(model_name_or_path=BASE7), folder.name, folder, None)
    assert seen["next_cmd"] == f"poros eval {folder.name}"  # streamed, as it trained


def _cached(hub: Path, model_id: str) -> None:
    """A model's weights in a Hub cache, laid out as huggingface_hub keeps them."""
    repo = hub / f"models--{model_id.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc")
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "snapshots" / "abc" / "model.safetensors").write_bytes(b"w")


@needs_ml
def test_the_load_line_reads_the_base_where_this_mac_keeps_it(ctx, capsys, tmp_path, mac,
                                                               monkeypatch):
    """A store on another disk holds the base: the line loads it there, and fetches nothing."""
    import ast

    folder, _ = finished_run(tmp_path)
    hub = mac / "poros" / "hub"
    _cached(hub, BASE7)
    code, obj = as_json(["export", folder.name, "--format", "adapter", "--output",
                         str(tmp_path / "adapter")], ctx, capsys)
    assert code == 0, obj
    ast.parse(obj["load"])
    assert "\n" not in obj["load"] and f"cache_dir={str(hub)!r}" in obj["load"]
    # a folder named like code is no path the line repeats: the base is fetched as before
    odd = tmp_path / "x'), __import__('os').system('id'), ('" / "hub"
    _cached(odd, BASE7)
    monkeypatch.setattr(store, "hubs", lambda: [str(odd)])
    code, obj = as_json(["export", folder.name, "--format", "adapter", "--output",
                         str(tmp_path / "adapter"), "--overwrite"], ctx, capsys)
    assert code == 0 and "cache_dir" not in obj["load"]


def test_watching_a_run_reads_the_one_pool(monkeypatch):
    from poros.cli.ui.screens import TrainState
    from poros.cli.verbs import runs_follow

    st = TrainState(model="m")
    data = {"vram_allocated_mb": 300.0, "vram_reserved_mb": 1400.0, "footprint": 2_400_000_000}
    event = {"event_type": "step", "step": 0, "data": data}
    runs_follow._apply(st, event)
    assert (st.vram_gb, st.vram_peak_gb) == (1.4, 2.4)


# ---------------------------------------------------------------------------
# train_cmd: the gate on a Mac, the screens
# ---------------------------------------------------------------------------


def preflight_only(cfg, **kw):
    from unittest import mock

    import poros.cli.train_cmd as train_cmd

    # the gate's order, torch or none: the device step's torch check is the Mac's first refusal
    with mock.patch("poros.preflight.arch.resolve_model_type", lambda _: ("qwen2", "")), \
            mock.patch("poros.cli.root._require_torch", lambda *a, **k: None):
        train_cmd.run(
            cfg,
            config_path="run.yaml",
            resume_from=None,
            trust_remote_code=False,
            friendly_error=_raise,
            **kw,
        )


def _raise(exc):
    raise exc


def test_the_gate_is_the_backends_and_a_job_is_submitted_after_it(monkeypatch):
    from poros.core.config import PorosConfig

    order = []
    monkeypatch.setattr(metal, "require", lambda cfg, **kw: order.append(("require", kw)))
    monkeypatch.setattr(metal, "check_this_mac", lambda: order.append("check"))
    preflight_only(
        PorosConfig(model_name_or_path=BASE7, dataset="synthetic"),
        submit=lambda: order.append("submit"),
    )
    assert order == [("require", {"check": False}), "submit"]  # the job's turn checks the Mac


def test_a_mac_checks_itself_only_once_it_holds_the_machine(monkeypatch):
    """The self-check runs MPS: never beside another run, so after the lock and not before."""
    from poros.cli import train_cmd
    from poros.core.config import PorosConfig

    order = []
    monkeypatch.setattr(metal, "require", lambda cfg, **kw: order.append(("require", kw)))
    monkeypatch.setattr(metal, "check_this_mac", lambda: order.append("check"))
    monkeypatch.setattr(train_cmd, "hold_mac", lambda: order.append("lock") or object())
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: None)

    def stop_here(cfg, model_type):  # nothing past the gate: no weights load in a test
        order.append("plan")
        raise typer.Exit(code=9)

    with pytest.raises(typer.Exit):
        preflight_only(PorosConfig(model_name_or_path=BASE7, dataset="synthetic"),
                       plan_gate=stop_here)
    assert order == [("require", {"check": False}), "lock", "check", "plan"]


def test_a_foreground_run_holds_the_machine_and_a_second_is_told_to_queue(
    tmp_path, monkeypatch, capsys
):
    from poros import jobs
    from poros.core.config import PorosConfig

    monkeypatch.setattr(metal, "require", lambda cfg, **kw: None)
    monkeypatch.setattr(metal, "check_this_mac", lambda: pytest.fail("checked beside a run"))
    held = jobs.hold_machine()
    assert held is not None
    with pytest.raises(typer.Exit) as stopped:
        preflight_only(PorosConfig(model_name_or_path=BASE7, dataset="synthetic"))
    assert stopped.value.exit_code == 2
    assert "--detach" in capsys.readouterr().err  # a refusal: on stderr
    held.close()


def test_a_run_keeps_the_mac_awake_for_as_long_as_it_lives(monkeypatch):
    from poros.core.config import PorosConfig

    started = []
    monkeypatch.setattr(metal, "require", lambda cfg, **kw: None)
    monkeypatch.setattr(metal, "check_this_mac", lambda: None)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: started.append(cmd))

    def stop_here(cfg, model_type):
        raise typer.Exit(code=9)

    with pytest.raises(typer.Exit):
        preflight_only(
            PorosConfig(model_name_or_path=BASE7, dataset="synthetic"), plan_gate=stop_here
        )
    assert started == [["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]]


def test_mac_oom_says_what_to_do_not_block_size(capsys):
    from poros.cli import train_cmd
    from poros.cli.ui.screens import TrainState
    from poros.core.errors import PorosOOMError

    exc = PorosOOMError("This Mac ran out of memory at step 3.", remediation="Close other apps.")
    seen = []
    state = TrainState(model="m")
    train_cmd._oom(exc, object(), "run.yaml", state, seen.append)
    assert seen == [exc]


def test_the_memory_bar_says_memory(monkeypatch):
    from poros.cli.ui import run_card, screens
    from poros.cli.ui.screens import TrainResult, TrainState, heartbeat, train_done, train_surface

    monkeypatch.setattr(screens, "MEMORY", "memory")
    monkeypatch.setattr(run_card, "MEMORY", "memory")  # the run card's readout
    monkeypatch.setattr(screens, "PEAK", "peak")

    st = TrainState(
        model="m",
        gpu=M4.name,
        card_gb=17.18,
        total=20,
        step=2,
        vram_gb=7.0,
        vram_peak_gb=8.0,
    )
    text = "".join(x for line in train_surface(st, 0, 100, animate=False) for x, _ in line)
    assert "memory" in text and not BANNED.search(text)
    assert heartbeat(st).count("memory") == 1 and not BANNED.search(heartbeat(st))
    done = TrainResult(
        steps=2,
        wall_s=1.0,
        final_loss=1.0,
        first_loss=2.0,
        losses=[2.0, 1.0],
        peak_vram_gb=8.0,
        card_gb=17.18,
        peak_ram_gb=None,
        ram_total_gb=None,
        output_dir="o",
        receipt=None,
    )
    text = "".join(x for line in train_done(done, 100) for x, _ in line)
    assert "peak" in text and not BANNED.search(text)


def _card(curve, final, every=1):
    from poros.cli.ui.screens import TrainResult, train_done

    done = TrainResult(steps=100, wall_s=1.0, final_loss=final,
                       first_loss=curve[0] if curve else None, losses=curve, every=every,
                       peak_vram_gb=8.0, card_gb=17.18, peak_ram_gb=None, ram_total_gb=None,
                       output_dir="o", receipt=None)
    return "".join(x for line in train_done(done, 100) for x, _ in line)


def test_one_row_a_step_reads_by_the_run_not_its_hardest_row():
    from poros.cli.train_cmd import card_curve

    # a run that learned, ending on a hard row: the last row says up, the run says down
    rows = [2.0 - i * 0.01 for i in range(99)] + [7.0]
    assert rows[-1] > rows[0]
    curve, final, every = card_curve(1, rows, rows[-1])
    assert len(curve) == 12 and every == 8
    assert final == curve[-1] < curve[0]
    text = _card(curve, final, every)
    assert "↓" in text and "↑" not in text

    # more than one row a step: nothing changes
    assert card_curve(8, rows, 1.5) == (rows, 1.5, 1)


def test_too_few_single_rows_say_no_direction():
    from poros.cli.train_cmd import card_curve

    rows = [1.71, 2.39, 1.67, 1.85, 1.71, 1.61, 2.38, 1.71, 1.57, 1.5,
            1.8, 2.76, 4.79, 2.49, 1.98, 1.39, 2.24, 2.15, 3.27, 7.77]
    curve, final, every = card_curve(1, rows, rows[-1])
    assert curve == [] and every == 1 and final == pytest.approx(sum(rows) / 20)
    text = _card(curve, final)
    assert "↑" not in text and "↓" not in text and "spike" not in text
    assert card_curve(1, [], 1.0) == ([], 1.0, 1)


def test_a_run_page_reads_one_row_a_step_the_same_way(tmp_path):
    import json

    from poros.cli.verbs import runs_report

    def facts(g):
        (tmp_path / "receipt.json").write_text(
            json.dumps({"config": {"gradient_accumulation_steps": g}}))
        rows = [{"event_type": "step", "step": i, "data": {"loss": 2.0 - i * 0.01}}
                for i in range(40)]
        (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, rows)))
        return runs_report.facts(tmp_path, tmp_path)["losses"]

    one = facts(1)
    assert [s for s, _ in one] == [8, 16, 24, 32, 40]
    assert one[0][1] == pytest.approx(2.0 - 3.5 * 0.01)
    assert len(facts(8)) == 40


def test_the_header_says_the_mac_and_its_row_goes_in_the_receipt():
    from poros.cli.train_cmd import _gpu, _mac_receipt
    from poros.core.config import PorosConfig

    assert _gpu() == (M4.name, 16 * 2**30 / 1e9)
    cfg = PorosConfig(model_name_or_path=BASE72, dataset="synthetic")
    table.fill(cfg, table.class_of(M4.name))
    state = SimpleNamespace(gpu=M4.name, vram_peak_gb=7.5, load_s=3.14159)
    trainer = SimpleNamespace(
        _witness=SimpleNamespace(checked=8, mismatches=0),
        _watchdog=SimpleNamespace(swap_growth=740_000_000),
    )
    receipt = _mac_receipt(cfg, state, trainer)
    assert receipt == {
        "witness": {"checked": 8, "mismatches": 0},
        "spilled": True,
        "swap_growth_mb": 740.0,
        "lora_dropout": 0.05,
        "load_s": 3.14,
        "platform": "darwin",
        "device_class": "apple-m4-10gpu-16gb",
        "memory_budget_gb": None,
        "macos_build": "26B123",
        "tier": "ssd",
        "table_row": 128,
        "table_build": table.pick(BASE72, table.class_of(M4.name)).macos_build or None,
    }
    cfg.max_seq_length = 1024
    assert _mac_receipt(cfg, state, trainer)["table_row"] is None  # it left the table


@pytest.mark.parametrize(
    "grew, spilled, mb",
    [(0, False, 0.0), (26_000_000, False, 26.0), (51_000_000, False, 51.0),
     (249_999_999, False, 250.0), (250_000_000, True, 250.0), (740_000_000, True, 740.0)],
)
def test_a_spill_is_a_size_that_moves_the_figures_and_the_number_stays(grew, spilled, mb):
    from poros.cli.train_cmd import _mac_receipt
    from poros.core.config import PorosConfig

    cfg = PorosConfig(model_name_or_path=BASE72, dataset="synthetic")
    table.fill(cfg, table.class_of(M4.name))
    trainer = SimpleNamespace(
        _witness=SimpleNamespace(checked=8, mismatches=0),
        _watchdog=SimpleNamespace(swap_growth=grew),
    )
    receipt = _mac_receipt(cfg, SimpleNamespace(gpu=M4.name, load_s=1.0), trainer)
    assert receipt["spilled"] is spilled and receipt["swap_growth_mb"] == mb


def test_a_run_on_a_disk_that_is_gone_says_so_and_not_that_it_saved_nothing(ctx, capsys, tmp_path):
    from poros import runs

    drive = tmp_path / "drive" / "runs" / "1104-0101-aaaa"
    action = {"model": BASE7, "data": "notes.jsonl", "steps": 6, "command": "poros train"}
    folder = runs.create("1104-0101-aaaa", action, output_dir=str(drive))
    runs.update(folder, state="failed")
    code, obj = as_json(["runs", "continue", "1104-0101-aaaa"], ctx, capsys)
    assert code == 5 and obj["error"]["code"] == "E_DISK"
    assert str(drive) in obj["error"]["message"] and "Attach its disk" in json.dumps(obj)
    drive.mkdir(parents=True)  # the disk is back, and truly holds no checkpoint
    code, obj = as_json(["runs", "continue", "1104-0101-aaaa"], ctx, capsys)
    assert obj["error"]["code"] == "E_NOT_FOUND" and "first checkpoint" in obj["error"]["message"]


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(2, "No such file", "events.jsonl"), RuntimeError("Parent folder missing")],
)
def test_a_drive_pulled_during_a_run_is_a_disk_error_naming_the_folder(tmp_path, error):
    from poros.cli.train_cmd import disk_error

    run = tmp_path / "drive" / "run"
    got = disk_error(error, SimpleNamespace(output_dir=str(run)))
    assert got.code == "E_DISK" and str(run) in got.info.message
    assert "Attach its disk" in got.info.remediation and "runs continue" in got.info.remediation


@pytest.mark.parametrize(
    "error, writing",
    [(RuntimeError("unexpected pos 21369792 vs 21369684"), True),
     (OSError(28, "No space left on device"), False)],
)
def test_a_disk_that_fills_says_so_and_that_the_run_goes_on(tmp_path, error, writing):
    from poros.cli.train_cmd import disk_error

    got = disk_error(error, SimpleNamespace(output_dir=str(tmp_path)), writing=writing)
    assert got.code == "E_DISK" and str(tmp_path) in got.info.message
    assert "Free some space" in got.info.remediation


def test_a_folder_still_there_or_another_error_stays_what_it_was(tmp_path, monkeypatch):
    from poros.cli.train_cmd import disk_error

    missing = FileNotFoundError(2, "No such file", "tokenizer.json")
    here = SimpleNamespace(output_dir=str(tmp_path))
    assert disk_error(missing, here) is missing
    bug = RuntimeError("shape mismatch")
    assert disk_error(bug, here) is bug
    wrong = ValueError("shape")
    gone = SimpleNamespace(output_dir=str(tmp_path / "gone"))
    assert disk_error(wrong, gone, writing=True) is wrong
    monkeypatch.setattr(sys, "platform", "linux")
    assert disk_error(missing, gone, writing=True) is missing


def _config_folder(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    real = Path(__file__).parents[1] / "metal" / "real_configs" / f"{name}.json"
    shutil.copy(real, folder / "config.json")
    return str(folder)


def test_the_config_form_fills_from_the_table_and_a_set_value_stays(tmp_path, monkeypatch, mac):
    import poros.cli.train_cmd as train_cmd

    seen = {}
    monkeypatch.setattr(train_cmd, "run", lambda cfg, **kw: seen.update(cfg=cfg))
    monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))  # _require_torch only imports
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: {BASE72}\ndataset: synthetic\nsave_steps: 7\n")
    code, text = typer_run("train", str(cfg))
    assert code == 0, text
    from poros.metal.table import fill

    fill(seen["cfg"], table.class_of(M4.name))
    assert seen["cfg"].save_steps == 7 and seen["cfg"].gradient_accumulation_steps == 1


# ---------------------------------------------------------------------------
# export, prove, runs, init, eval, bench
# ---------------------------------------------------------------------------


def finished_run(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "adapter_model.safetensors").write_bytes(b"weights")
    (out / "adapter_config.json").write_text("{}")
    folder = runs.create("1102-2300-abcd", {"model": BASE7, "name": "n"}, output_dir=str(out))
    runs.update(folder, state="succeeded", finished_at=runs.now())
    return folder, out


def test_runs_and_their_report_speak_of_memory(ctx, capsys, tmp_path):
    from poros.cli.verbs import runs_report

    folder, out = finished_run(tmp_path)
    (out / "receipt.json").write_text(
        json.dumps({"platform": "darwin", "peak_vram_allocated_gb": 7.5, "s_per_step": 7.0})
    )
    code, text = as_person(["runs"], ctx, capsys)
    assert code == 0 and "✓ n" in text and not BANNED.search(text)
    report = Path(runs_report.write(next(iter(runs.all_runs())))).read_text()
    assert "peak memory, GB" in report and not BANNED.search(report)


def test_init_writes_a_template_with_no_length_or_block(tmp_path):
    path = tmp_path / "train.yaml"
    code, text = typer_run("init", str(path))
    assert code == 0, text
    written = path.read_text()
    assert "max_seq_length" not in written and "block_size" not in written
    assert not BANNED.search(written) and "Qwen/Qwen2.5-0.5B-Instruct" in written  # the starter


def test_eval_and_bench_are_refused_with_the_code_for_it(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: {BASE7}\ndataset: synthetic\n")
    code, text = typer_run("eval", str(cfg))
    assert code == 4 and "does not run on a Mac" in text
    contract.reset()
    code, text = typer_run("bench", "run", str(cfg))
    assert code == 4 and "does not run on a Mac" in text


# ---------------------------------------------------------------------------
# The word ban, verb by verb.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["machine"],
        ["models"],
        ["recipes"],
        ["train", BASE7, "--data", "notes.jsonl"],
        ["train", BASE72, "--data", "notes.jsonl"],
        ["train", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "--data", "notes.jsonl"],
        ["runs"],
    ],
)
def test_normal_output_never_says_what_poros_does_underneath(ctx, capsys, argv):
    code, text = as_person(argv, ctx, capsys)
    assert text.strip() and not BANNED.search(text), text


def test_the_welcome_names_the_mac():
    from poros.cli import banner

    assert banner._gpu_line() == M4.name
