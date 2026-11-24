"""`--from` and `init_adapter:` in every form: a new run from an earlier run's adapter.

For someone who wants more or other data after a run: the verb form's `--from` refused plain
text ("--from needs a recipe ... the classic loop starts fresh"), a config's `init_adapter:`
was checked nowhere before the download, and no refusal named the route. On the simulated
trainer (the weights' own load is tests/train/test_start_from_adapter.py's).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poros import runs
from poros.core.config import PorosConfig
from poros.preflight import command
from tests.cli.test_run_flow_wording import MODEL, _chats
from tests.cli.test_train_runs import home, say  # noqa: F401  (the simulated machine)


def _text(home, name="text.jsonl", n=30):  # noqa: F811
    (home / name).write_text("".join(
        json.dumps({"text": f"plain line {i} for the classic loop, with a few more words"}) + "\n"
        for i in range(n)))
    return name


def _action(run) -> dict:
    return json.loads((Path(run.folder) / "action.json").read_text())


def _resolved(run) -> dict:
    return json.loads((Path(run["output_dir"]) / "config.resolved.json").read_text())


def test_from_starts_the_classic_loop_from_a_runs_adapter(home, capsys):  # noqa: F811
    """`poros train --from last --data notes-as-text` was refused: "--from needs a recipe"."""
    _chats(home, 40)
    _, sft = say(["train", MODEL, "--data", "chats.jsonl", "--steps", "10"], home, capsys)
    code, more = say(["train", "--from", "last", "--data", _text(home), "--steps", "6"], home,
                     capsys)
    assert code == 0, more
    said = _resolved(more)
    assert said["trained_with"] == "classic"
    assert said["init_adapter"] == str(Path(sft["output_dir"]) / "checkpoint-9")
    assert said["lora_rank"] == _resolved(sft)["lora_rank"]  # the adapter's shape
    assert more["run"] != sft["run"] and more["output_dir"] != sft["output_dir"]  # a new run
    new = next(r for r in runs.all_runs() if r.id == more["run"])
    assert new.parent == sft["run"] and _action(new)["data"].endswith("text.jsonl")


def test_from_on_new_chats_is_a_new_run_on_that_data(home, capsys):  # noqa: F811
    _chats(home, 40)
    _, sft = say(["train", MODEL, "--data", "chats.jsonl", "--steps", "10"], home, capsys)
    _chats(home, 60, "more.jsonl")
    code, more = say(["train", "--from", sft["run"], "--data", "more.jsonl", "--steps", "6"],
                     home, capsys)
    assert code == 0, more
    assert _resolved(more)["init_adapter"] == str(Path(sft["output_dir"]) / "checkpoint-9")
    new = next(r for r in runs.all_runs() if r.id == more["run"])
    assert _action(new)["data"].endswith("more.jsonl") and new.step == 6  # its own steps, from 0


def test_continue_with_a_changed_setting_names_the_new_run_from_its_adapter(home, capsys):  # noqa: F811
    _chats(home, 40)
    say(["train", MODEL, "--data", "chats.jsonl", "--steps", "10"], home, capsys)
    code, obj = say(["runs", "continue", "last", "--seq-len", "64"], home, capsys)
    assert code == 2 and "start a new run from its adapter" in obj["error"]["message"]
    assert "train --from <run> --data <file>" in obj["error"]["message"]


def test_a_run_started_from_another_is_not_offered_as_a_recipe(home, capsys,  # noqa: F811
                                                               monkeypatch):
    """"Keep it as a recipe: poros recipes save last" was offered after a `--from` run, and
    `recipes save` then refused it: "It started from another run"."""
    from poros.cli.verbs import runs_report
    from tests.cli.test_run_flow_wording import _screen

    _chats(home, 40)
    _, plain = _screen(["train", MODEL, "--data", "chats.jsonl", "--steps", "10"], home, capsys)
    assert "poros recipes save last" in plain  # a run from the model: offered, as before
    _chats(home, 60, "more.jsonl")
    code, more = _screen(["train", "--from", "last", "--data", "more.jsonl", "--steps", "6"],
                         home, capsys)
    assert code == 0 and "Trained" in more
    assert "recipes save" not in more and "Keep it as a recipe" not in more
    # by what each recorded, not their order: two simulated runs can begin in one second
    (new,) = [r for r in runs.all_runs() if r.parent]
    (old,) = [r for r in runs.all_runs() if r.id != new.id]
    assert new.parent == old.id
    for agent in (False, True):
        monkeypatch.setenv("POROS_AGENT", "1") if agent else monkeypatch.delenv(
            "POROS_AGENT", raising=False)
        assert not any("recipes save" in cmd for cmd in runs_report.offers(new)), agent
        assert f"poros recipes save {old.id}" in runs_report.offers(old), agent
    code, refused = say(["recipes", "save", new.id], home, capsys)  # and it is still refused
    assert code != 0 and "started from another run" in refused["error"]["message"]


# ── a config's init_adapter, before the download ───────────────────────────────────────────────


def _adapter(tmp_path, **said):
    """A run's folder holding one checkpoint that recorded ``said`` (as a trainer's does)."""
    from poros.core import checkpoints

    ckpt = tmp_path / "old" / "checkpoint-7"
    ckpt.mkdir(parents=True)
    (ckpt / checkpoints.FORMAT_MARKER).write_text("1")
    (ckpt / checkpoints.COMPLETE_MARKER).write_text("7")
    (ckpt / "poros_config.json").write_text(json.dumps(
        {"model_name_or_path": "Org/Model", "lora_rank": 8, "lora_alpha": 24,
         "lora_target_modules": ["q_proj", "v_proj"], **said}))
    return ckpt


def _cfg(tmp_path, **said):
    return PorosConfig(model_name_or_path="Org/Model", output_dir=str(tmp_path / "new"), **said)


def test_init_adapter_names_a_checkpoint_and_takes_its_shape_where_none_is_said(tmp_path):
    ckpt = _adapter(tmp_path)
    cfg = _cfg(tmp_path, init_adapter=str(tmp_path / "old"))  # the run's folder
    assert command.start_refusal(None, cfg) is None
    assert cfg.init_adapter == str(ckpt)  # its newest checkpoint, as the record keeps it
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules) == (8, 24,
                                                                        ["q_proj", "v_proj"])
    assert {"lora_rank", "lora_alpha"} <= cfg.model_fields_set  # a recipe's presets leave them
    said = _cfg(tmp_path, init_adapter=str(ckpt), lora_rank=8, lora_alpha=64)
    assert command.start_refusal(None, said) is None and said.lora_alpha == 64  # as it says


def test_init_adapter_of_another_shape_or_base_is_refused_before_the_download(tmp_path):
    ckpt = _adapter(tmp_path)
    rank = command.start_refusal(None, _cfg(tmp_path, init_adapter=str(ckpt), lora_rank=4))
    assert (rank.code, rank.title) == ("E_CONFIG", "Can't start from this adapter")
    assert "trained with lora_rank 8, and this run has 4" in rank.reason
    assert rank.context == [command.BEFORE_THE_LOAD,
                            "set lora_rank to 8, or drop it to take the adapter's"]
    layers = command.start_refusal(None, _cfg(tmp_path, init_adapter=str(ckpt),
                                              lora_target_modules=["q_proj"]))
    assert "lora_target_modules" in layers.reason
    base = command.start_refusal(None, PorosConfig(model_name_or_path="Org/Other",
                                                   init_adapter=str(ckpt)))
    assert "trained on base 'Org/Model'" in base.reason
    gone = command.start_refusal(None, _cfg(tmp_path, init_adapter=str(tmp_path / "nowhere")))
    assert (gone.code, gone.title) == ("E_NOT_FOUND", "No adapter to start from")
    # a resume reads its own checkpoint: init_adapter, still in its config, is not looked at
    assert command.start_refusal(str(ckpt), _cfg(tmp_path, init_adapter="/nowhere")) is None


def test_a_config_files_run_starts_from_an_adapter(tmp_path):
    """`poros train run.yaml` with `init_adapter:` on the simulated machine: the new run's record
    names the checkpoint, its own folder and the new data; a folder with no adapter is refused."""
    from tests.conftest import checkout_env

    _adapter(tmp_path, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
    _chats(tmp_path, 30, "more.jsonl")
    env = checkout_env({**os.environ, "POROS_HOME": str(tmp_path / "home"),
                        "POROS_DEV_SIMULATE": "1", "HF_HUB_OFFLINE": "1"})
    env.pop("POROS_OUTPUT", None)

    def train(start):
        (tmp_path / "run.yaml").write_text(
            "model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct\ndataset: ./more.jsonl\n"
            f"output_dir: ./new\nmax_steps: 4\ninit_adapter: {start}\n")
        return subprocess.run(
            [sys.executable, "-c", "from poros.cli.main import cli; cli()", "train", "run.yaml",
             "--json"], capture_output=True, text=True, env=env, cwd=tmp_path,
            stdin=subprocess.DEVNULL)

    missing = train("./nowhere")
    err = json.loads(missing.stdout)["error"]
    assert err["code"] == "E_NOT_FOUND" and "no folder at ./nowhere" in err["message"]
    assert not (tmp_path / "new" / "events.jsonl").exists()
    done = train("./old")
    assert done.returncode == 0, done.stdout + done.stderr
    said = json.loads((tmp_path / "new" / "config.resolved.json").read_text())
    assert said["init_adapter"] == "old/checkpoint-7"  # the checkpoint, as the config named it
    assert said["lora_rank"] == 8 and said["dataset"].endswith("more.jsonl")


@pytest.mark.parametrize("field", ["init_adapter"])
def test_the_field_says_what_it_takes(field):
    said = PorosConfig.model_fields[field].description
    assert "Recipe runs only" not in said and "run's folder" in said and "output_dir" in said
