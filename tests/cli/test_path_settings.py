"""The loop a CUDA run takes (path: efficient, fast or sequential): settable by a person (the
train card's "more" list) and by an agent (key=value, a config file, poros schema config), and
recorded in the run's config.resolved.json and its settings line."""

from __future__ import annotations

import json
import os

import pytest

from poros import runs
from poros.cli import app, contract
from poros.cli.context import Context
from tests.conftest import checkout_env, needs_ml

SWITCHES = ("path",)


def test_the_card_offers_one_path_in_words_and_the_rest_by_name():
    from poros.cli.sheets import config_fields
    from poros.cli.train_settings import SETTINGS, setting

    assert set(SWITCHES) <= set(config_fields())  # every one reachable
    path = setting("path")
    assert path.group == "memory & speed" and path.label == "path"
    first = next(s for s in SETTINGS if s.group == "memory & speed")
    assert first.name == "path"  # the first thing memory & speed offers
    said = dict(path.choices)
    assert list(said) == ["efficient", "fast", "sequential"]  # the default first
    # what is true of each: the default holds less than fast, and sequential can peak lower
    assert "less VRAM than fast" in said["efficient"] and "fastest" in said["fast"]
    assert "one micro-batch at a time" in said["sequential"] and "lower" in said["sequential"]
    assert not any("least VRAM" in words for words in said.values())
    assert not {"block_major_accumulation", "offload_boundaries"} & set(config_fields())


def test_a_mac_card_lists_neither():
    from poros.cli.schemas import PlanGpu
    from poros.cli.sheets import config_fields

    mac = Context(fixed_gpu=PlanGpu(name="Apple M4", total_gib=16.0, free_gib=None,
                                    source="spec", unified=True))
    assert not set(SWITCHES) & set(config_fields(mac))


def test_the_agents_help_says_the_paths():
    from poros.cli.verbs import train

    doc = " ".join(train.__doc__.split())
    assert "efficient (default) takes less VRAM than fast" in doc
    assert "fast is the fastest" in doc and "the same result each way" in doc
    assert "sequential is slower and can peak lower" in doc and "least VRAM" not in doc


def test_the_schema_offers_the_three_loops():
    from poros.core.config import PorosConfig

    path = PorosConfig.model_json_schema()["properties"]["path"]
    assert set(path["anyOf"][0]["enum"]) == {"efficient", "fast", "sequential"}
    assert path["default"] is None  # decided for the run
    assert "less VRAM than fast" in path["description"] and "the fastest" in path["description"]
    assert "least VRAM" not in path["description"]  # the sequential loop can peak lower


def test_every_config_field_says_what_it_does():
    """`poros schema config` is the machine contract (AGENTS.md §4): no field is a bare name."""
    from poros.core.config import FIXED_FIELDS, PorosConfig

    props = PorosConfig.model_json_schema()["properties"]
    assert set(props) == set(PorosConfig.model_fields) - set(FIXED_FIELDS)
    bare = sorted(k for k, v in props.items() if len(v.get("description", "").strip()) < 10)
    assert bare == []


def test_poros_init_offers_the_path_off_a_mac(monkeypatch):
    import sys

    from poros.cli.commands import train as cli_commands_train

    monkeypatch.setattr(sys, "platform", "linux")
    text = cli_commands_train._starter_config_yaml("m", "synthetic", "./out")
    assert "# path: fast" in text and "sequential: one micro-batch at a time" in text
    monkeypatch.setattr(sys, "platform", "darwin")
    text = cli_commands_train._starter_config_yaml("m", "synthetic", "./out")
    assert "# path:" not in text


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The simulated trainer (no GPU): it stands in for a CUDA run, as its card does."""
    from poros.cli.ui import simulate

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    # that card whatever the host: on a Mac the simulator previews the Mac itself (its disks, its
    # table), which test_mac checks
    monkeypatch.setenv("POROS_DEV_SIMULATE_GPU", "RTX 4090:24.0")
    simulate._gpu.cache_clear()
    monkeypatch.setenv("POROS_DEV_SIMULATE_STEP", "0.001")
    monkeypatch.setenv("POROS_DEV_SIMULATE_LOAD", "0")
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="offline in tests"))
    (tmp_path / "notes.jsonl").write_text(
        "".join(json.dumps({"text": "word " * (20 + i)}) + "\n" for i in range(40)))
    yield tmp_path
    contract.reset()
    simulate._gpu.cache_clear()


def _ran(argv, home, capsys) -> dict:
    """The settings a run trained at: its config.resolved.json."""
    ctx = Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False)
    old = os.getcwd()
    os.chdir(home)
    try:
        code = app.dispatch(argv, ctx)
    finally:
        os.chdir(old)
    out = capsys.readouterr().out
    obj = json.loads([x for x in out.splitlines() if x.startswith("{")][-1])
    assert code == 0, obj
    resolved = json.loads((runs.root() / obj["run"] / "config.resolved.json").read_text())
    return {**resolved, "_said": out}


TRAIN = ["train", "Qwen2.5-7B-Instruct", "--data", "notes.jsonl", "--steps", "2"]


@pytest.mark.parametrize("given, want, line", [
    ([], "efficient", " · efficient path"),  # the default
    (["path=efficient"], "efficient", " · efficient path"),
    (["path=fast"], "fast", " · fast path"),  # block inputs on the GPU
    (["path=sequential"], "sequential", None),
])
def test_key_value_reaches_the_run_and_is_recorded(home, capsys, given, want, line):
    ran = _ran([*TRAIN, *given], home, capsys)
    assert ran["path"] == want and ran["path_reason"] is None
    assert "block_major_accumulation" not in ran and "offload_boundaries" not in ran
    assert ran["lora_dropout"] == 0.05  # the default, on either loop
    settings = ran["_said"].split("seq ")[-1][:40]  # the settings line's tail after the length
    assert (line in ran["_said"]) if line else (" path" not in settings)


@pytest.mark.parametrize("dropout", ["0", "0.1"])
def test_a_dropout_you_set_trains_block_major(home, capsys, dropout):
    ran = _ran([*TRAIN, f"lora_dropout={dropout}"], home, capsys)
    assert (ran["lora_dropout"], ran["path"]) == (float(dropout), "efficient")
    assert "sequential" not in ran["_said"]


def test_pairs_train_block_major_and_the_record_says_so(home, capsys):
    """poros.Run sends a pair batch's slices through block-major too (a slice holds its pairs)."""
    (home / "pairs.jsonl").write_text("".join(
        json.dumps({"prompt": f"q{i} " * 5, "chosen": f"good {i}", "rejected": f"bad {i}"}) + "\n"
        for i in range(40)))
    ran = _ran(["train", "Qwen2.5-7B-Instruct", "--data", "pairs.jsonl", "--steps", "2"], home,
               capsys)
    assert ran["trained_with"] == "dpo"
    assert ran["path"] == "efficient"


@pytest.mark.parametrize("rows, per_device, planned", [
    (1, None, 1),  # one row, one a micro-batch: every step is one
    (3, 4, 1),
    (5, 4, 8),  # several micro-batches: as the config says
])
def test_the_plan_gate_plans_a_recipe_that_fits_one_micro_batch_as_it_runs(
        tmp_path, rows, per_device, planned):
    """Each of its steps is one micro-batch, so that is what its fit check plans, on the path
    the config trains."""
    from poros.cli.commands import train as cli_commands_train
    from poros.core.config import PorosConfig

    few = tmp_path / "few.jsonl"
    few.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"say {i}"}, {"role": "assistant", "content": f"ok {i}"}]})
        + "\n" for i in range(rows)))
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", dataset=str(few),
                      **({"per_device_batch_size": per_device} if per_device else {}))
    seen = cli_commands_train._as_it_runs(cfg)
    assert seen.gradient_accumulation_steps == planned and seen.path == cfg.path is None


@needs_ml
def test_a_config_file_says_the_same(home):
    import subprocess
    import sys

    (home / "run.yaml").write_text(
        "model_name_or_path: Qwen/Qwen2.5-7B-Instruct\ndataset: synthetic\nmax_steps: 2\n"
        f"output_dir: {home / 'out'}\npath: fast\n")
    subprocess.run([sys.executable, "-P", "-m", "poros.cli.main", "train", "run.yaml", "--json"],
                   cwd=home, check=True, capture_output=True, timeout=300, env=checkout_env())
    ran = json.loads((home / "out" / "config.resolved.json").read_text())
    assert (ran["path"], ran["lora_dropout"]) == ("fast", 0.05)
