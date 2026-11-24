"""The ladder a new family climbs, as an agent follows it from results alone:

    arch <model> -> arch try <model> -> prove <model> (asks) -> prove <model> --unvalidated
    -> a pack

The simulator stands in for the GPU and config-only folders for the models: no weights. What the
gates compare is tested on real tensors in tests/test_validate_gates.py."""

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
pytest.importorskip("peft")

from poros import runs  # noqa: E402
from poros.cli import app, contract  # noqa: E402
from poros.cli.context import Context  # noqa: E402
from poros.cli.schemas import PlanGpu  # noqa: E402
from poros.cli.verbs import arch, prove_gates  # noqa: E402
from poros.core import arch_registry  # noqa: E402

REAL = Path(__file__).parent.parent / "metal" / "real_configs"
# SmolLM2-135M's config: a llama, which the registry does not name
LLAMA = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 576,
         "intermediate_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 9,
         "num_key_value_heads": 3, "vocab_size": 49152, "max_position_embeddings": 8192,
         "rms_norm_eps": 1e-5, "rope_theta": 100000, "tie_word_embeddings": True}
CARD = PlanGpu(name="NVIDIA GeForce RTX 4060 Laptop GPU", total_gib=8.0, free_gib=8.0,
               source="spec")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    monkeypatch.setenv("POROS_DEV_SIMULATE_STEP", "0.001")
    monkeypatch.setenv("POROS_DEV_SIMULATE_LOAD", "0")
    monkeypatch.delenv("POROS_USER_ASSERTED_ARCHS", raising=False)
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="offline in tests"))
    yield tmp_path
    contract.reset()


def _folder(where: Path, config: dict, dtype: str = "") -> str:
    where.mkdir()
    (where / "config.json").write_text(json.dumps(config))
    if dtype:  # a shard's header alone: what arch reads of a release's format
        header = json.dumps({"w": {"dtype": dtype, "shape": [64, 64], "data_offsets": [0, 4096]}})
        (where / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header.encode())
    return str(where)


def say(argv, home, capsys, gpu=CARD):
    ctx = Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False, fixed_gpu=gpu)
    old = os.getcwd()
    os.chdir(home)
    try:
        code = app.dispatch(argv, ctx)
    finally:
        os.chdir(old)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert len(lines) == 1, lines
    return code, json.loads(lines[0])


def test_each_rung_points_at_the_next_and_no_next_carries_consent(home, capsys):
    model = _folder(home / "smol", LLAMA)
    code, seen = say(["arch", model], home, capsys)
    assert code == 0 and seen["next"] == [f"poros arch try {model}"]
    code, tried = say(["arch", "try", model], home, capsys)
    assert code == 0 and tried["tried"]["ok"] and tried["tried"]["route"] == "cuda"
    assert tried["next"] == [f"poros prove {model}"]
    code, asked = say(shlex.split(tried["next"][0])[1:], home, capsys)
    assert code == 2 and asked["schema"] == "poros.needs_input.v1", asked
    confirm, check = asked["questions"][:2]
    assert "model_type llama" in confirm["prompt"] and "changes no status" in confirm["prompt"]
    assert confirm["choices"][0]["value"] == "--unvalidated"
    assert check["choices"][0]["command"] == shlex.join(["poros", "arch", "try", model, "--json"])
    code, done = say(shlex.split(confirm["choices"][0]["command"])[1:], home, capsys)
    assert code == 0 and done["schema"] == "poros.prove_gates.v1" and done["passed"], done
    assert (done["support"], done["model_type"]) == ("user_asserted", "llama")
    for said in (seen["next"], tried["next"], done["next"]):
        assert not {"--unvalidated", "--yes", "--force", "--download"} & set(
            " ".join(said).split())
    # 20 steps is short of the procedure's 200: next runs those, and asks again
    assert done["next"] == [shlex.join(["poros", "prove", model, "--steps", "200", "--seq-len",
                                        "256"])]
    code, full = say([*shlex.split(done["next"][0])[1:], "--unvalidated"], home, capsys)
    assert code == 0 and full["passed"] and full["training"]["steps"] == 200
    assert full["next"] == ["open an architecture request issue on the Poros GitHub repo and "
                            f"attach the files in {full['pack']}"]
    assert full["submit"]["to"] == "docs/validation/llama/"
    assert full["submit"]["into"] == "poros/core/arch_registry.py"
    assert arch_registry.arch_status("llama") is None  # nothing became anything


# gemma2: a dense roadmap target. The any-model ladder takes it as it takes any dense family
GEMMA2 = {"model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"], "hidden_size": 256,
          "intermediate_size": 512, "num_hidden_layers": 4, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 4096,
          "max_position_embeddings": 2048, "rms_norm_eps": 1e-6}


def test_a_dense_roadmap_target_climbs_the_ladder_as_any_dense_family(home, capsys):
    """poros.preflight.arch.no_flag_trains refuses a mixture of experts and a variant no loop
    trains, never a dense family for being a roadmap target: arch, arch try and prove
    --unvalidated take gemma2 as they take llama, and its status stays what it was."""
    model = _folder(home / "g2", GEMMA2)
    code, seen = say(["arch", model], home, capsys)
    assert code == 0 and seen["refused"] is None and seen["next"] == [f"poros arch try {model}"]
    assert seen["status"] == "target"
    code, tried = say(["arch", "try", model], home, capsys)
    assert code == 0 and tried["tried"]["ok"] and tried["tried"]["route"] == "cuda", tried
    assert arch.reading(model, Context(cwd=home, fixed_gpu=CARD)).refused == ""
    code, done = say(["prove", model, "--unvalidated"], home, capsys)
    assert code == 0 and done["passed"] and done["model_type"] == "gemma2", done
    assert arch_registry.arch_status("gemma2") == "target"  # nothing became anything


def test_a_pass_writes_the_pack_in_poros_home(home, capsys):
    model = _folder(home / "smol", LLAMA)
    code, done = say(["prove", model, "--unvalidated"], home, capsys)
    folder = Path(done["pack"])
    assert folder.parent == runs.home() / "validation" and folder.name.startswith("llama-")
    assert sorted(p.name for p in folder.iterdir()) == done["files"] == [
        "README.md", "load_gate.jsonl", "parity.jsonl", "receipt.json", "registry_line.txt"]
    load = json.loads((folder / "load_gate.jsonl").read_text())
    parity = json.loads((folder / "parity.jsonl").read_text())
    assert load["byte_identical"] and parity["all_equal"] and len(parity["resident_losses"]) == 20
    assert (folder / "registry_line.txt").read_text() == done["submit"]["registry_line"]
    assert '"llama": ArchSpec(' in done["submit"]["registry_line"]
    text = "".join((folder / n).read_text() for n in done["files"])
    assert str(home) not in text  # the folder's own name, never its path
    receipt = json.loads(Path(done["receipt"]).read_text())
    assert receipt["schema"] == "poros.prove_gates.v1" and receipt["passed"]


SIMULATED_LOAD, SIMULATED_TRAINING = prove_gates._simulated_load, prove_gates._simulated_training


def _differs_at(step: int, surface: str):
    def training(a):
        rec = SIMULATED_TRAINING(a)
        key = {"loss": "losses", "grads": "grads", "weights": "weights",
               "optimizer": "opt_state"}[surface]
        rec[f"{key}_equal"] = rec["all_equal"] = False
        rec["loss_max_abs_diff" if key == "losses" else f"{key}_max_abs_diff"] = 2.0**-20
        rec["first_difference"][key] = {"step": step} if key == "losses" else {
            "step": step, "tensor": "base_model.model.model.layers.3.mlp.down_proj.lora_A"}
        if key != "losses":  # a weight off at step N moves the loss at N + 1
            rec["first_difference"]["losses"] = {"step": step + 1}
            rec["losses_equal"] = False
        return rec

    return training


@pytest.mark.parametrize("surface", ["grads", "weights", "optimizer"])
def test_a_fail_names_the_surface_and_the_step_and_writes_no_pack(home, capsys, monkeypatch,
                                                                    surface):
    monkeypatch.setattr(prove_gates, "_simulated_training", _differs_at(3, surface))
    model = _folder(home / "smol", LLAMA)
    code, done = say(["prove", model, "--unvalidated"], home, capsys)
    assert code == 1 and not done["passed"] and done["pack"] is None and done["submit"] is None
    first = done["differs"][0]
    assert (first["surface"], first["step"]) == (surface, 3)
    assert first["tensor"].endswith("down_proj.lora_A")
    assert done["differs"][1] == {"surface": "loss", "step": 4, "tensor": None}
    assert done["training"][surface]["first_step"] == 3
    assert not (runs.home() / "validation").exists()
    assert done["next"] == [f"poros arch try {model}"]
    assert json.loads(Path(done["receipt"]).read_text())["passed"] is False


def test_a_load_that_differs_stops_before_training(home, capsys, monkeypatch):
    def load(a):
        rec = SIMULATED_LOAD(a)
        rec.update(byte_identical=False, n_mismatched=1, mismatched=["qstate::model.x::absmax"])
        return rec

    def never(a):
        raise AssertionError("trained after a load that differs")

    monkeypatch.setattr(prove_gates, "_simulated_load", load)
    monkeypatch.setattr(prove_gates, "_simulated_training", never)
    model = _folder(home / "smol", LLAMA)
    code, done = say(["prove", model, "--unvalidated"], home, capsys)
    assert code == 1 and done["training"] is None and done["pack"] is None
    assert done["differs"] == [{"surface": "load", "step": None,
                                "tensor": "qstate::model.x::absmax"}]


@pytest.mark.parametrize("config, dtype, words", [
    ({"model_type": "evil_net", "auto_map": {"AutoModelForCausalLM": "evil.Net"}}, "",
     "its own code"),
    ({"model_type": "not_a_type"}, "", "no class"),
    (LLAMA, "F8_E4M3", "ships F8_E4M3"),
    (LLAMA, "I8", "ships I8"),
])
def test_what_no_flag_trains_no_flag_proves(home, capsys, config, dtype, words):
    model = _folder(home / "it", config, dtype)
    for flag in ([], ["--unvalidated"]):
        code, obj = say(["prove", model, *flag], home, capsys)
        assert obj["error"]["code"] == "E_ARCH_UNSUPPORTED" and code == 4, obj
        assert words in obj["error"]["message"] and "evil" not in obj["error"]["message"]


def test_an_unvalidated_moe_and_a_gated_repo_are_refused(home, capsys, monkeypatch):
    raw = json.loads((REAL / "mixtral_8x7b.json").read_text())
    code, obj = say(["prove", _folder(home / "mixtral", raw), "--unvalidated"], home, capsys)
    assert code == 4 and "mixture of experts" in obj["error"]["message"]
    monkeypatch.setattr(arch, "_from_hub", lambda model: arch._Seen(gated=True))
    code, obj = say(["prove", "someone/gated-model", "--unvalidated"], home, capsys)
    said = obj["error"]["message"]
    assert code == 4 and "gated" in said and "no token" in said


def test_the_validated_path_is_what_it_was(home, capsys):
    code, plain = say(["prove"], home, capsys)
    assert code == 0 and plain["schema"] == "poros.prove.v1" and plain["identical"]
    assert set(plain) == {"schema", "model", "steps", "seq_len", "simulated", "reference",
                          "streamed", "resident", "one_block", "identical", "compared", "differs",
                          "receipt", "env", "against", "reproduced", "pinned_differs",
                          "largest_certified"}
    assert plain["reference"] == "resident" and plain["one_block"] is None
    code, obj = say(["prove", "Qwen/Qwen2.5-0.5B", "--unvalidated"], home, capsys)
    assert code == 2 and "is a validated family" in obj["error"]["message"]
    assert obj["error"]["hint"] == "poros prove Qwen/Qwen2.5-0.5B"
    code, obj = say(["prove", plain["receipt"], "--unvalidated"], home, capsys)
    assert code == 2 and "a receipt re-runs as it was" in obj["error"]["message"]
    code, obj = say(["prove", _folder(home / "q", {"model_type": "qwen3"}), "--unvalidated"],
                    home, capsys)
    assert code == 2 and "inherits its parity" in obj["error"]["message"]


def test_steps_stop_at_the_cap_and_a_length_must_be_one(home, capsys):
    model = _folder(home / "smol", LLAMA)
    code, obj = say(["prove", model, "--unvalidated", "--steps", "2001"], home, capsys)
    assert code == 2 and "at most 2000" in obj["error"]["message"]
    code, obj = say(["prove", model, "--unvalidated", "--seq-len", "99999999"], home, capsys)
    assert code == 2 and obj["error"]["code"] == "E_USAGE"
    code, obj = say(["prove", "--unvalidated"], home, capsys)
    assert code == 2 and "on a model" in obj["error"]["message"]


def test_on_a_mac_the_gates_say_where_they_run(home, capsys):
    mac = PlanGpu(name="Apple M4", total_gib=16.0, free_gib=None, source="spec", unified=True)
    model = _folder(home / "smol", LLAMA)
    code, obj = say(["prove", model, "--unvalidated"], home, capsys, gpu=mac)
    assert code == 4 and obj["error"]["code"] == "E_GUARANTEE_UNAVAILABLE"
    assert "CUDA" in obj["error"]["message"]


def test_without_a_gpu_the_gates_refuse_before_loading_anything(home, capsys, monkeypatch):
    import torch

    monkeypatch.delenv("POROS_DEV_SIMULATE")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(prove_gates, "_load_gate", lambda a: pytest.fail("it loaded"))
    from poros.cli.verbs import prove

    monkeypatch.setattr(prove, "confirmations", lambda a, ctx: [])
    model = _folder(home / "smol", LLAMA)
    code, obj = say(["prove", model, "--unvalidated"], home, capsys)
    assert code == 5 and obj["error"]["code"] == "E_NO_CUDA", obj


def test_a_person_sees_each_gate_and_where_one_differs(home, capsys, monkeypatch):
    from poros.cli.ui.text import plain

    model = _folder(home / "smol", LLAMA)
    code, done = say(["prove", model, "--unvalidated"], home, capsys)
    shown = "\n".join(plain(x) for x in prove_gates.view(prove_gates.GatesOutput(**done), 80))
    assert "Gates passed" in shown and "byte-identical" in shown and "pack" in shown
    monkeypatch.setattr(prove_gates, "_simulated_training", _differs_at(5, "weights"))
    code, done = say(["prove", model, "--unvalidated"], home, capsys)
    shown = "\n".join(plain(x) for x in prove_gates.view(prove_gates.GatesOutput(**done), 80))
    assert "weights at step 5 differs" in shown and "differs from step 5" in shown
    assert "arch try" in shown and "pack " not in shown


@pytest.mark.cuda_host(reason="the CUDA ladder: on a Mac `arch try` takes the Mac's route, and a "
                               "simulated run previews the Mac (its store and disk)")
def test_an_agent_climbs_the_ladder_from_results_alone(tmp_path):
    """The real entry point, as an agent's shell runs it: every command it runs is one a result
    handed it."""
    model = _folder(tmp_path / "smol", LLAMA)
    env = {**os.environ, "POROS_HOME": str(tmp_path / "home"), "POROS_DEV_SIMULATE": "1",
           "POROS_DEV_SIMULATE_STEP": "0.001", "POROS_DEV_SIMULATE_LOAD": "0",
           "HF_HUB_OFFLINE": "1"}
    for name in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB", "POROS_USER_ASSERTED_ARCHS"):
        env.pop(name, None)

    def agent(command):
        words = shlex.split(command)
        assert words[:2] == ["poros", "agents"], command
        proc = subprocess.run([sys.executable, "-c", "from poros.cli.main import cli; cli()",
                               *words[1:]], capture_output=True, text=True, cwd=tmp_path, env=env,
                              stdin=subprocess.DEVNULL)
        return proc.returncode, json.loads(proc.stdout.strip().splitlines()[-1])

    code, seen = agent(f"poros agents arch {shlex.quote(model)}")
    code, tried = agent(seen["next"][0])
    assert code == 0 and tried["tried"]["route"] == "cuda" and tried["tried"]["ok"]
    code, asked = agent(tried["next"][0])
    assert code == 2 and asked["schema"] == "poros.needs_input.v1"
    command = asked["questions"][0]["choices"][0]["command"]
    assert command.startswith("poros agents prove") and command.endswith("--unvalidated")
    code, done = agent(command)
    assert code == 0 and done["schema"] == "poros.prove_gates.v1" and done["passed"], done
    code, schema = agent("poros agents schema prove_gates")
    assert schema["$id"] == "poros.prove_gates.v1" and "differs" in schema["properties"]


@pytest.mark.parametrize(("config", "says"), [
    ({**LLAMA, "model_type": "mixtral", "num_local_experts": 8, "num_experts_per_tok": 2},
     "mixture of experts"),
    ({"model_type": "gpt_oss", "hidden_size": 64, "num_hidden_layers": 2}, "mixture of experts"),
], ids=["mixtral", "gpt_oss"])
def test_arch_refuses_what_no_flag_trains_as_train_does(home, capsys, tmp_path, config, says):
    """docs/models.md, rung 0: what no flag trains is refused by arch itself
    (E_ARCH_UNSUPPORTED, exit 4, no next step), as train refuses it, and a target's layout is
    never read (gpt_oss showed an AttributeError there)."""
    folder = _folder(tmp_path / "m", config)
    for argv in (["arch", folder, "--json"], ["arch", "try", folder, "--json"]):
        code, out = say(argv, home, capsys)
        assert code == 4 and out["refused"]["code"] == "E_ARCH_UNSUPPORTED", out
        assert says in out["refused"]["reason"] and out["next"] == [] and out["tried"] is None
        assert not any("Error" in x for x in out["cannot_place"]), out
    assert arch.reading(folder, Context(cwd=home, fixed_gpu=CARD)).refused == (
        out["refused"]["reason"])


def test_arch_try_loads_eager_where_transformers_runs_no_sdpa(monkeypatch):
    """transformers 5.21+ refuses SDPA for Gemma 2 (soft-capped attention): its CUDA load is
    eager, and so is the tiny copy arch try checks, not a ValueError."""
    import transformers

    from poros.validate import tiny_route

    real = transformers.AutoModelForCausalLM.from_config.__func__
    asked = []

    def from_config(cls, cfg, **kw):
        asked.append(kw.get("attn_implementation"))
        if kw.get("attn_implementation") == "sdpa":
            raise ValueError("Gemma2ForCausalLM does not support an attention implementation "
                             "through torch.nn.functional.scaled_dot_product_attention yet")
        return real(cls, cfg, **kw)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config",
                        classmethod(from_config))
    out = tiny_route.prove(GEMMA2)
    assert [a for a in asked if a][:2] == ["sdpa", "eager"] and out["ok"], out
