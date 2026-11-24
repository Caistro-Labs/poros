"""A family Poros has not validated: one question, --unvalidated for one run, the refusals no flag
gets past, and every result about such a run saying so. The simulator and config-only folders: no
weights."""

from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from poros import runs  # noqa: E402
from poros.cli import app, contract  # noqa: E402
from poros.cli.context import Context  # noqa: E402
from poros.cli.schemas import PlanGpu  # noqa: E402
from poros.cli.verbs import arch  # noqa: E402
from poros.core import arch_registry  # noqa: E402
from poros.recipes import presets, rules  # noqa: E402

REAL = Path(__file__).parent.parent / "metal" / "real_configs"
# SmolLM2-135M's config: a llama, which the registry does not name
LLAMA = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 576,
         "intermediate_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 9,
         "num_key_value_heads": 3, "vocab_size": 49152, "max_position_embeddings": 8192,
         "rms_norm_eps": 1e-5, "rope_theta": 100000, "tie_word_embeddings": True}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    # the simulated CUDA card these tests are written for, whatever the host is (a Mac's preview
    # is the Mac itself: test_mac's to check)
    from poros.cli.ui import simulate

    monkeypatch.setenv("POROS_DEV_SIMULATE_GPU", "RTX 4090:24.0")
    simulate._gpu.cache_clear()
    monkeypatch.setenv("POROS_DEV_SIMULATE_STEP", "0.001")
    monkeypatch.setenv("POROS_DEV_SIMULATE_LOAD", "0")
    monkeypatch.delenv("POROS_USER_ASSERTED_ARCHS", raising=False)
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="offline in tests"))
    (tmp_path / "notes.jsonl").write_text(
        "".join(json.dumps({"text": "word " * (20 + i)}) + "\n" for i in range(40)))
    yield tmp_path
    contract.reset()
    simulate._gpu.cache_clear()


def _folder(where: Path, config: dict, dtype: str = "") -> Path:
    where.mkdir()
    (where / "config.json").write_text(json.dumps(config))
    if dtype:  # a shard's header alone: what arch reads of a release's format
        header = json.dumps({"w": {"dtype": dtype, "shape": [64, 64], "data_offsets": [0, 4096]}})
        (where / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header.encode())
    return where


def say(argv, home, capsys, ctx=None):
    ctx = ctx or Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False)
    old = os.getcwd()
    os.chdir(home)
    try:
        code = app.dispatch(argv, ctx)
    finally:
        os.chdir(old)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert len(lines) == 1, lines
    return code, json.loads(lines[0])


def test_a_new_family_asks_once_with_the_flag_and_offers_arch_try(home, capsys):
    model = _folder(home / "smol", LLAMA)
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--steps", "3"], home, capsys)
    assert code == 2 and obj["schema"] == "poros.needs_input.v1", obj
    confirm, check = obj["questions"][:2]
    assert confirm["id"] == "unvalidated"
    assert "model_type llama" in confirm["prompt"] and "status not-in-registry" in confirm["prompt"]
    assert "no parity guarantee" in confirm["prompt"]
    assert confirm["choices"][0]["value"] == "--unvalidated"
    assert confirm["choices"][0]["command"].endswith("--unvalidated --json")
    assert check["choices"][0]["command"] == shlex.join(["poros", "arch", "try", str(model),
                                                         "--json"])
    # the family's question says it all: no second one about the catalog
    assert not any("--allow-unvalidated" in c["command"] for q in obj["questions"]
                   for c in q["choices"])
    assert arch_registry.arch_status("llama") is None  # asserted for that command alone


def test_the_flag_trains_it_and_every_result_says_so(home, capsys):
    model = _folder(home / "smol", LLAMA)
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--steps", "3",
                     "--unvalidated"], home, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1", obj
    assert (obj["support"], obj["model_type"]) == ("user_asserted", "llama")
    assert obj["command"].endswith("--unvalidated")
    folder = runs.root() / obj["run"]
    assert json.loads((folder / "action.json").read_text())["unvalidated"] == "llama"
    receipt = json.loads((Path(obj["output_dir"]) / "receipt.json").read_text())
    assert (receipt["support"], receipt["model_type"]) == ("user_asserted", "llama")
    code, info = say(["runs", obj["run"]], home, capsys)
    assert (info["runs"][0]["support"], info["runs"][0]["model_type"]) == ("user_asserted", "llama")
    code, listed = say(["runs"], home, capsys)
    assert listed["runs"][0]["support"] == "user_asserted"
    code, live = say(["runs", obj["run"], "--live", "--frame"], home, capsys)
    assert live["support"] == "user_asserted" and any("not validated" in r for r in live["card"])
    assert "not validated" in live["frame"][2]
    code, report = say(["runs", "report", obj["run"]], home, capsys)
    assert "Not validated: model_type llama" in Path(report["report"]).read_text()
    assert arch_registry.arch_status("llama") is None  # nothing stays asserted after the run


def test_a_validated_family_is_not_asked_and_not_marked(home, capsys):
    model = _folder(home / "mine", {"model_type": "qwen3"})
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--steps", "3"], home, capsys)
    assert code == 2 and "not in the catalog" in obj["questions"][0]["prompt"]
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--steps", "3",
                     "--allow-unvalidated"], home, capsys)
    assert code == 0 and obj["support"] is None and "--unvalidated" not in obj["command"].split()
    code, info = say(["runs", obj["run"]], home, capsys)
    assert "support" not in info["runs"][0]


@pytest.mark.parametrize("config, dtype, words", [
    ({"model_type": "evil_net", "auto_map": {"AutoModelForCausalLM": "evil.Net"}}, "",
     "its own code"),
    ({"model_type": "not_a_type"}, "", "no class"),
    (LLAMA, "F8_E4M3", "ships F8_E4M3"),
    (LLAMA, "I8", "ships I8"),
])
def test_what_no_flag_trains_is_refused_in_plain_words(home, capsys, config, dtype, words):
    model = _folder(home / "it", config, dtype)
    for flag in ([], ["--unvalidated"]):
        code, obj = say(["train", str(model), "--data", "notes.jsonl", *flag], home, capsys)
        assert obj["error"]["code"] == "E_ARCH_UNSUPPORTED" and code == 4, obj
        assert words in obj["error"]["message"]
        assert "evil" not in obj["error"]["message"]  # a config's words never reach a result


def test_an_unvalidated_moe_is_refused_on_a_gpu(home, capsys):
    raw = json.loads((REAL / "mixtral_8x7b.json").read_text())
    model = _folder(home / "mixtral", raw)
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--unvalidated"], home, capsys)
    assert code == 4 and "mixture of experts" in obj["error"]["message"]
    assert "dense route" in obj["error"]["message"] and "not made NF4" in obj["error"]["message"]


def test_a_gated_repo_is_refused_with_the_flag(home, capsys, monkeypatch):
    monkeypatch.setattr(arch, "_from_hub", lambda model: arch._Seen(gated=True))
    code, obj = say(["train", "someone/gated-model", "--data", "notes.jsonl", "--unvalidated"],
                    home, capsys)
    said = obj["error"]["message"]
    assert code == 4 and "gated" in said and "no token" in said


def test_on_a_mac_a_hub_id_outside_the_catalog_reads_as_its_folder_does(home, monkeypatch):
    monkeypatch.setattr(arch, "_from_hub", lambda model: arch._Seen(raw=dict(LLAMA),
                                                                    counts={"BF16": 10**8}))
    mac = Context(fixed_gpu=PlanGpu(name="Apple M4", total_gib=16.0, free_gib=None,
                                    source="spec", unified=True))
    folder = _folder(home / "smol", LLAMA, "BF16")
    said = [arch.reading(m, ctx) for m in ("HuggingFaceTB/SmolLM2-135M-Instruct", str(folder))
            for ctx in (mac, Context())]
    assert {(s.model_type, s.status, s.refused, s.moe) for s in said} == {
        ("llama", "not-in-registry", "", False)}


def test_an_export_of_an_unvalidated_run_says_so_and_merges_on_its_assertion(
    home, capsys, monkeypatch
):
    from poros.cli.commands import export as export_command

    model = _folder(home / "smol", LLAMA)
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--steps", "3",
                     "--unvalidated"], home, capsys)
    files = Path(obj["output_dir"])
    (files / "adapter_config.json").write_text("{}")
    (files / "adapter_model.safetensors").write_bytes(b"x")
    code, out = say(["export", obj["run"], "--format", "adapter"], home, capsys)
    assert code == 0 and (out["support"], out["model_type"]) == ("user_asserted", "llama"), out
    seen = []
    monkeypatch.setattr(export_command, "export", lambda **kw: seen.append(
        arch_registry.arch_status("llama")))
    code, out = say(["export", obj["run"], "--format", "merged", "--output",
                     str(home / "merged")], home, capsys)
    assert code == 0 and seen == ["user_asserted"] and out["support"] == "user_asserted", out
    assert arch_registry.arch_status("llama") is None  # for that export alone


def test_the_assertion_ends_with_its_command():
    with arch_registry.one_command():
        arch_registry.assert_this_command("llama")
        assert arch_registry.arch_status("llama") == "user_asserted"
        assert "llama" in arch_registry.runnable_arch_types()
        assert "llama" not in arch_registry.validated_arch_types()  # never a validation
    assert arch_registry.arch_status("llama") is None
    with arch_registry.one_command():
        arch_registry.assert_this_command("qwen3")
        assert arch_registry.arch_status("qwen3") == "validated"  # nothing is downgraded


def test_a_second_reading_in_one_command_still_asks(home):
    """A sheet or an answer resolves again inside the same command: what that command asserted
    to read its plan is no consent."""
    from poros.cli.verbs import train
    from poros.cli.words.grammar import parse

    model = _folder(home / "smol", LLAMA)
    ctx = Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False)
    with arch_registry.one_command():
        for _ in range(3):
            a = train.resolve(parse(["train", str(model), "--data", "notes.jsonl"]), ctx).action
            assert a.family == "llama" and a.unvalidated is None
            assert train.confirmations(train.inspect(a, ctx), ctx)[0].choices[0].value == (
                "--unvalidated")


def test_the_env_s_assertion_needs_no_question_and_is_recorded(home, capsys, monkeypatch):
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "llama")
    model = _folder(home / "smol", LLAMA)
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--steps", "3"], home, capsys)
    assert code == 0 and obj["support"] == "user_asserted", obj
    assert obj["command"].endswith("--unvalidated")  # the same run again needs no env


def _config_run(home, model, *flags):
    import shutil

    from typer.testing import CliRunner

    from poros.cli.main import app as typer_app

    shutil.rmtree(home / "out", ignore_errors=True)  # each call a new run (D3: never among one's)
    config = home / "run.yaml"
    config.write_text(json.dumps({"model_name_or_path": str(model), "dataset": "synthetic",
                                  "max_steps": 2, "output_dir": str(home / "out")}))
    said = CliRunner().invoke(typer_app, ["train", str(config), "--yes", "--json", *flags])
    lines = [x for x in said.stdout.splitlines() if x.startswith("{")]
    return said.exit_code, json.loads(lines[-1])


def test_a_config_file_takes_the_flag_too_and_never_with_remote_code(home):
    model = _folder(home / "smol", LLAMA)
    code, obj = _config_run(home, model)
    assert code == 4 and obj["error"]["code"] == "E_ARCH_UNSUPPORTED"
    assert "--unvalidated" in json.dumps(obj["error"])  # the way past, said where it refuses
    code, obj = _config_run(home, model, "--unvalidated")
    assert code == 0 and (obj["support"], obj["model_type"]) == ("user_asserted", "llama"), obj
    code, obj = _config_run(home, model, "--unvalidated", "--trust-remote-code")
    assert code == 2 and "own code" in obj["error"]["message"]
    raw = json.loads((REAL / "mixtral_8x7b.json").read_text())
    code, obj = _config_run(home, _folder(home / "mixtral", raw), "--unvalidated")
    assert code == 4 and "mixture of experts" in obj["error"]["message"]


def test_a_job_asserts_it_again_on_every_attempt(tmp_path, monkeypatch):
    from poros import jobs
    from poros.jobs import runner

    monkeypatch.setenv("POROS_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "launch", lambda job_dir: None)
    job = jobs.submit({"model_name_or_path": "x/y", "dataset": "synthetic"}, model="x/y",
                      gpu="0", unvalidated="llama")
    assert (job.support, job.model_type) == ("user_asserted", "llama")
    job_dir = Path(job.job_dir)
    assert json.loads((job_dir / "spec.json").read_text())["unvalidated"] is True
    said = []
    monkeypatch.setattr(runner, "_wait_for_gpu", lambda *a: open(os.devnull))
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: said.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0, stdout="{}"))
    runner.main(job_dir)
    assert "--unvalidated" in said[0]
    row = next(r for r in runs.all_runs() if r.id == job.job_id).to_dict()
    assert (row["support"], row["model_type"]) == ("user_asserted", "llama")


def test_a_continue_carries_the_flag(home, capsys, monkeypatch):
    from poros.cli import train_cmd
    from poros.cli.verbs import train_run

    model = _folder(home / "smol", LLAMA)
    code, obj = say(["train", str(model), "--data", "notes.jsonl", "--steps", "3",
                     "--unvalidated"], home, capsys)
    seen = {}
    monkeypatch.setattr(train_cmd, "run", lambda cfg, **kw: seen.update(kw) or {})
    run = next(r for r in runs.all_runs() if r.id == obj["run"])
    out = train_run.continue_run(run, more=2)
    assert seen["unvalidated"] is True and out.support == "user_asserted"


def test_a_run_of_a_new_family_saves_as_a_recipe_that_says_so(home, capsys, monkeypatch):
    receipt = {"recipe": "SFT", "support": "user_asserted", "model_type": "llama",
               "config": {"max_seq_length": 256}, "simulated": True}
    fields = rules.from_run("HuggingFaceTB/SmolLM2-135M-Instruct", {}, receipt, [], "chats")
    assert fields["unvalidated"] == "llama"
    text = rules.dumps(fields)
    assert 'unvalidated = "llama"' in text
    hostile = rules.from_run("a/b", {}, {**receipt, "model_type": "llama; poros train x"}, [],
                             "chats")
    assert hostile["unvalidated"] == "unknown"  # what it names is a word, or unknown
    with pytest.raises(ValueError):
        rules.parse(text.replace('"llama"', '"Llama Now"'))
    folder = Path(os.environ["POROS_HOME"]) / "recipes"
    folder.mkdir(parents=True)
    (folder / "smol.toml").write_text(text)
    code, obj = say(["recipes", "show", "smol"], home, capsys)
    one = obj["recipe"]
    assert (one["support"], one["model_type"]) == ("user_asserted", "llama")
    assert "no parity guarantee" in one["says"] and "--unvalidated" not in one["train"]
    code, listed = say(["recipes"], home, capsys)
    assert next(r for r in listed["setups"] if r["key"] == "smol")["support"] == "user_asserted"
    assert presets.local(home)[0][0].says().startswith("not validated")
    # training from it asks, as any new family does, and the answer carries the flag
    monkeypatch.setattr(arch, "_from_hub", lambda model: arch._Seen(raw=dict(LLAMA)))
    argv = shlex.split(one["train"].replace("<data>", "notes.jsonl"))[1:]
    code, asked = say(argv, home, capsys)
    assert code == 2 and asked["questions"][0]["choices"][0]["value"] == "--unvalidated"
    assert "--unvalidated" in asked["questions"][0]["choices"][0]["command"]


def test_an_agent_gets_the_question_and_the_flag_trains_it(tmp_path):
    """The real entry point, as an agent's shell runs it."""
    model = _folder(tmp_path / "smol", LLAMA)
    (tmp_path / "chats.jsonl").write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"hi {i}"}, {"role": "assistant", "content": f"hello {i}"}]})
        + "\n" for i in range(12)))
    env = {**os.environ, "POROS_HOME": str(tmp_path / "home"), "POROS_DEV_SIMULATE": "1",
           "POROS_DEV_SIMULATE_STEP": "0.001", "POROS_DEV_SIMULATE_LOAD": "0",
           "HF_HUB_OFFLINE": "1"}
    for name in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB", "POROS_USER_ASSERTED_ARCHS"):
        env.pop(name, None)

    def agent(*words):
        proc = subprocess.run([sys.executable, "-c", "from poros.cli.main import cli; cli()",
                               *words], capture_output=True, text=True, cwd=tmp_path, env=env,
                              stdin=subprocess.DEVNULL)
        return proc.returncode, json.loads(proc.stdout.strip().splitlines()[-1])

    code, asked = agent("agents", "train", str(model), "--data", "chats.jsonl", "--steps", "3",
                        "--wait")
    assert code == 2, asked
    command = asked["questions"][0]["choices"][0]["command"]
    assert command.startswith("poros agents train") and "--unvalidated" in command
    assert "--wait" in command
    assert asked["questions"][1]["choices"][0]["command"] == shlex.join(
        ["poros", "agents", "arch", "try", str(model)])
    code, done = agent(*shlex.split(command)[1:])
    assert code == 0 and done["support"] == "user_asserted" and done["model_type"] == "llama", done
    code, info = agent("agents", "run", "info", done["run"])
    assert info["runs"][0]["support"] == "user_asserted"


# -- no stored token reaches a family nobody validated ---------------------------------------


def test_an_asserted_family_sends_no_stored_token_to_the_hub(monkeypatch):
    """Under --unvalidated every training fetch passes token=False: the tokenizer's files, the
    streamed load's download and the model's own load. Outside it, the library's default."""
    import torch

    from poros.core import arch_registry, tokenizer
    from poros.train import model as model_mod
    from poros.train import streamed_load

    seen: dict = {}

    def cached_file(name, file, **kw):
        seen.setdefault("tokenizer", set()).add(kw.get("token"))
        return None

    monkeypatch.setattr("transformers.utils.cached_file", cached_file)

    def snapshot(name, **kw):
        seen["download"] = kw.get("token")
        return "/nowhere"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    class Auto:
        @staticmethod
        def from_pretrained(name, **kw):
            seen["model"] = kw.get("token", "unset")
            return torch.nn.Linear(1, 1)

    monkeypatch.setattr("transformers.AutoModelForCausalLM", Auto)

    def fetch():
        seen.clear()
        tokenizer._stage("org/new-family", None)
        streamed_load._resolve_checkpoint_dir("org/new-family", None)
        with monkeypatch.context() as m:  # the NF4 load asks for CUDA first
            m.setattr("torch.cuda.is_available", lambda: True)
            model_mod._load_model("org/new-family", torch.bfloat16, nf4_load_strategy="gpu")
        return dict(seen)

    with arch_registry.one_command():
        assert fetch() == {"tokenizer": {None}, "download": None, "model": "unset"}
        arch_registry.assert_this_command("llama")
        assert arch_registry.hub_token() is False
        assert fetch() == {"tokenizer": {False}, "download": False, "model": False}
    assert arch_registry.hub_token() is None  # it ends with the command
