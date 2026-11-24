"""The CLI's machine contract (docs/cli.md), end to end.

Each case runs the real `poros` entry point in a subprocess, because the
contract is about the process: what lands on stdout (exactly one JSON object),
what lands on stderr, and the exit code. Outputs are validated against the
schema `poros schema <name>` publishes, not against the model that made them.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from poros.core import errors
from poros.core.errors import EXIT_CODES

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict, "null": type(None)}


def _poros(*args, cwd=None, env=None):
    env = {**{k: v for k, v in os.environ.items()
              if k not in ("FORCE_COLOR", "CLICOLOR_FORCE", "POROS_OUTPUT")}, **(env or {})}
    return subprocess.run(
        [sys.executable, "-c", "from poros.cli.main import cli; cli()", *args],
        capture_output=True, text=True, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
    )


def _check(value, schema, defs):
    """Validate the subset of JSON Schema pydantic emits: refs, types, required, const."""
    if "$ref" in schema:
        schema = defs[schema["$ref"].rsplit("/", 1)[-1]]
    if "anyOf" in schema:
        errs = []
        for option in schema["anyOf"]:
            try:
                return _check(value, option, defs)
            except AssertionError as e:
                errs.append(str(e))
        raise AssertionError(f"{value!r} matches no anyOf option: {errs}")
    if "const" in schema:
        assert value == schema["const"], (value, schema["const"])
    if "enum" in schema:
        assert value in schema["enum"], (value, schema["enum"])
    if "type" in schema:
        expected = _TYPES[schema["type"]]
        assert isinstance(value, expected) and not (
            schema["type"] in ("integer", "number") and isinstance(value, bool)
        ), (value, schema["type"])
    for key in schema.get("required", ()):
        assert key in value, f"missing required {key!r}"
    for key, sub in schema.get("properties", {}).items():
        if isinstance(value, dict) and key in value:
            _check(value[key], sub, defs)
    if schema.get("type") == "array" and "items" in schema:
        for item in value:
            _check(item, schema["items"], defs)


def _one_object(proc, name):
    """stdout is exactly one JSON line, valid against `poros schema <name>`."""
    lines = proc.stdout.splitlines()
    assert len(lines) == 1, f"stdout must be one JSON line, got {proc.stdout!r}"
    obj = json.loads(lines[0])
    assert obj["schema"] == f"poros.{name}.v1"
    published = json.loads(_poros("schema", name).stdout)
    assert published["$id"] == f"poros.{name}.v1"
    _check(obj, published, published.get("$defs", {}))
    return obj


def _error(proc, code):
    obj = _one_object(proc, "error")
    assert obj["error"]["code"] == code
    assert proc.returncode == obj["error"]["exit_code"] == EXIT_CODES[code]
    return obj["error"]


def test_every_output_schema_is_registered_and_requires_its_schema_key():
    names = _poros("schema", "--list").stdout.split()
    assert {"config", "events", "artifact", "error", "check"} <= set(names)
    assert "doctor" not in names
    for name in names[3:]:
        doc = json.loads(_poros("schema", name).stdout)
        assert "schema" in doc["required"], name
        assert doc["properties"]["schema"]["const"] == f"poros.{name}.v1"


def test_doctor_is_an_unknown_command():
    """`poros machine` is the environment check: `doctor` is refused as any unknown command is."""
    proc = _poros("doctor", "--json")
    obj = _one_object(proc, "error")
    assert obj["error"]["code"] == "E_USAGE"
    assert "No such command 'doctor'" in obj["error"]["message"]
    assert proc.returncode == EXIT_CODES["E_USAGE"]


@pytest.mark.parametrize(("model_type", "status", "code", "entry"), [
    ("qwen2", "validated", 0, "dense"),
    # a Mac says not-on-this-device for a family it has no row for, before the registry's word
    pytest.param("gpt_oss", "unsupported", 4, "moe",  # a mixture, as its entry says
                 marks=pytest.mark.cuda_host),
    pytest.param("not_a_real_arch", "not-in-registry", 4, "dense", marks=pytest.mark.cuda_host),
])
def test_check_json_status_and_exit_code(tmp_path, model_type, status, code, entry):
    from poros.core.arch_registry import architectures

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    proc = _poros("check", str(tmp_path), "--json")
    obj = _one_object(proc, "check")
    assert (obj["status"], proc.returncode, obj["entry"]) == (status, code, entry)
    spec = architectures().get(model_type)
    assert spec is None or spec.entry == obj["entry"]


@pytest.mark.parametrize(("name", "model_type", "entry"), [
    ("qwen3_8_27b", "qwen3_5", "dense"),
    ("qwen3_6_35b_a3b", "qwen3_5_moe", "moe"),
])
def test_check_on_a_real_qwen_config_is_validated(tmp_path, name, model_type, entry):
    """Qwen/Qwen3.8-27B and Qwen/Qwen3.6-35B-A3B as their config.json says
    (tests/metal/real_configs/): validated, on the entry their family trains through, exit 0."""
    real = pathlib.Path(__file__).resolve().parent / "metal" / "real_configs" / f"{name}.json"
    (tmp_path / "config.json").write_text(real.read_text())
    proc = _poros("check", str(tmp_path), "--json")
    obj = _one_object(proc, "check")
    assert (obj["model_type"], obj["status"], obj["entry"], proc.returncode) == (
        model_type, "validated", entry, 0), obj


# a Mac's check says whether this Mac has a row for the model, and a bare config gives it none
_NO_MAC_ROW = pytest.mark.cuda_host(reason="a Mac's check says whether this Mac has a row for the "
                                    "model, and a bare config gives it none")


@pytest.mark.parametrize(("config", "says", "trains"), [
    pytest.param({"model_type": "llama"}, "--unvalidated", True, marks=_NO_MAC_ROW),
    pytest.param({"model_type": "mixtral", "num_local_experts": 8}, "mixture of experts", False,
                 marks=pytest.mark.cuda_host(reason="a GPU's rule for a new MoE family: a Mac "
                                             "trains one with --unvalidated, as check says")),
    ({"model_type": "gpt_oss"}, "mixture of experts", False),
    ({"model_type": "gemma4", "hidden_size_per_layer_input": 256}, "neither loop", False),
], ids=["llama", "mixtral", "gpt_oss", "gemma4_per_layer"])
def test_check_says_why_and_what_next_as_train_and_arch_do(tmp_path, config, says, trains):
    """One rule (poros.preflight.arch.no_flag_trains) for check, `agents arch` and train: a family
    --unvalidated trains gets a reason and its next rung; one no flag trains says why, offers no
    --unvalidated, and has no next step."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    proc = _poros("check", str(tmp_path), "--json")
    obj = _one_object(proc, "check")
    assert proc.returncode == 4 and says in obj["reason"], obj
    assert obj["next"] == ([f"poros arch try {tmp_path}"] if trains else []), obj
    shown = _poros("check", str(tmp_path)).stdout
    assert ("--unvalidated" in shown) is trains, shown
    if trains:  # the validated list names every family that trains, MoE ones too
        assert "qwen3_moe" in shown and "qwen2" in shown


def test_check_offers_a_dense_roadmap_target_the_any_model_ladder(tmp_path):
    """gemma2, a dense roadmap target: not validated (exit 4), and --unvalidated trains it, as
    any dense family (poros.preflight.arch.no_flag_trains refuses MoE and unsupported variants
    only)."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma2"}))
    proc = _poros("check", str(tmp_path), "--json")
    obj = _one_object(proc, "check")
    assert proc.returncode == 4 and obj["status"] == "target" and "--unvalidated" in obj["reason"]
    assert obj["next"] == [f"poros arch try {tmp_path}"], obj
    assert "--unvalidated" in _poros("check", str(tmp_path)).stdout


def test_the_arch_gate_names_what_trains_not_only_what_is_validated():
    from poros.core.arch_registry import blockwise_arch_check
    from poros.core.errors import PorosArchNotValidatedError

    with pytest.raises(PorosArchNotValidatedError) as err:
        blockwise_arch_check("gemma4_per_layer", mode="off")
    said = err.value.info.message
    assert "Validated architectures" not in said and "neither the dense nor the MoE loop" in said
    assert "Architectures this install trains:" in said and "qwen3_moe" in said
    with pytest.raises(PorosArchNotValidatedError) as err:
        blockwise_arch_check("llama")
    assert "has not been validated" in err.value.info.message


def test_check_unresolvable_is_a_usage_error_with_a_reason(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    proc = _poros("check", str(tmp_path), "--json")
    obj = _one_object(proc, "check")
    assert obj["status"] == "could-not-resolve" and obj["reason"]
    assert proc.returncode == 2


def test_env_var_turns_on_json_mode(tmp_path):
    proc = _poros("init", str(tmp_path / "t.yaml"), env={"POROS_OUTPUT": "json"})
    assert _one_object(proc, "init")["path"] == str(tmp_path / "t.yaml")
    assert proc.returncode == 0


def test_human_output_moves_to_stderr_in_json_mode(tmp_path):
    proc = _poros("init", str(tmp_path / "t.yaml"), "--json")
    _one_object(proc, "init")
    again = _poros("init", str(tmp_path / "t.yaml"), "--json")
    assert "already exists" in _error(again, "E_USAGE")["message"]


def test_usage_errors_are_json_errors():
    _error(_poros("report", "--json"), "E_USAGE")
    _error(_poros(env={"POROS_OUTPUT": "json"}), "E_USAGE")  # bare `poros`


@pytest.mark.cuda_host(reason="poros bench does not run on a Mac in this release: that refusal "
                               "comes first")
def test_bench_refuses_json_mode_rather_than_print_nothing(tmp_path):
    proc = _poros("bench", "compare", str(tmp_path), str(tmp_path),
                  env={"POROS_OUTPUT": "json"})
    assert "JSON mode" in _error(proc, "E_USAGE")["message"]


def test_documents_keep_the_real_stdout_in_json_mode():
    env = {"POROS_OUTPUT": "json"}
    assert json.loads(_poros("schema", "config", env=env).stdout)["title"] == "PorosConfig"
    assert _poros("--version", env=env).stdout.startswith("poros ")
    assert "Usage" in _poros("check", "--help", "--json").stdout


def _script(code: str, *args):
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)


def test_library_and_c_level_writes_land_on_stderr():
    proc = _script(
        "import os\n"
        "from poros.cli import contract\n"
        "from poros.cli.schemas import InitOutput\n"
        "contract.isolate_stdout()\n"
        "print('python print'); os.write(1, b'c-level write\\n')\n"
        "contract.emit(InitOutput(path='p', model='m', dataset='d', output_dir='o'))\n"
    )
    assert json.loads(proc.stdout)["schema"] == "poros.init.v1"
    assert "python print" in proc.stderr and "c-level write" in proc.stderr


_CRASH = (
    "import sys\n"
    "from poros.cli import main, root\n"
    "@main.app.command(name='boom')\n"
    "def boom(json_output: bool = root._JSON_OPTION):\n"
    "    raise {}\n"
    "sys.argv = ['poros', 'boom', '--json']\n"
    "main.cli()\n"
)


def test_a_crash_is_one_internal_error_with_the_traceback_on_stderr():
    proc = _script(_CRASH.format("RuntimeError('kaboom')"))
    assert _error(proc, "E_INTERNAL")["message"] == "RuntimeError"  # its text is on stderr alone
    assert "Traceback" in proc.stderr and "kaboom" in proc.stderr


def test_ctrl_c_exits_130():
    _error(_script(_CRASH.format("KeyboardInterrupt")), "E_INTERRUPTED")


def test_markup_in_user_input_is_a_config_error_not_a_crash(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text('model_name_or_path: x\ndataset: synthetic\n"[/x]": 1\n')
    assert "[/x]" in _error(_poros("train", str(cfg), "--json"), "E_CONFIG")["message"]


def test_json_mode_does_not_leak_into_the_next_in_process_command(tmp_path):
    from typer.testing import CliRunner

    from poros.cli.main import app

    runner = CliRunner()
    runner.invoke(app, ["check", str(tmp_path), "--json"])
    res = runner.invoke(app, ["init", str(tmp_path / "t.yaml")])
    assert "Wrote" in res.output and not res.output.startswith("{")


def test_config_errors_exit_two_with_field_detail(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("model_name_or_path: x\ndataset: synthetic\nlora_r: 8\n")
    assert "lora_r" in _error(_poros("train", str(cfg), "--json"), "E_CONFIG")["message"]
    human = _poros("train", str(cfg))  # an error: on stderr, and nothing on stdout
    assert human.returncode == 2 and "did you mean" in human.stderr and not human.stdout


def test_missing_dataset_file_is_a_data_error(tmp_path):
    pytest.importorskip("torch")  # without torch, train exits E_NO_TORCH first
    cfg = tmp_path / "t.yaml"
    cfg.write_text("model_name_or_path: x\ndataset: ./nope.jsonl\n")
    _error(_poros("train", str(cfg), "--json", cwd=tmp_path), "E_DATASET_NOT_FOUND")


def test_no_results_is_a_data_error(tmp_path):
    _error(_poros("leaderboard", str(tmp_path), "--json"), "E_NO_RESULTS")


def test_risky_action_never_prompts_without_a_tty(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "run"
    out.mkdir()
    (out / "adapter_weights.pt").write_bytes(b"")
    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"model_name_or_path: x\ndataset: synthetic\noutput_dir: {out}\n")
    # the question `poros train <model> --output` asks of a used folder, as data: one object and
    # one flag for both forms (it was E_CONFIRM_REQUIRED with --yes here, a question with
    # --overwrite there)
    asked = _poros("train", str(cfg), "--json")
    said = json.loads(asked.stdout)
    assert asked.returncode == 2 and said["schema"] == "poros.needs_input.v1", asked.stdout
    (q,) = said["questions"]
    assert q["id"] == "overwrite" and q["choices"][0]["value"] == "--overwrite"
    assert q["choices"][0]["command"] == f"poros train {cfg} --overwrite --json"
    human = _poros("train", str(cfg))  # stdin is /dev/null: asks as data, never waits
    assert human.returncode == 2 and "--overwrite" in human.stdout + human.stderr
    for flag in ("--overwrite", "--yes"):  # --yes: its name before, kept
        again = _poros("train", str(cfg), flag, "--json")
        assert "needs_input" not in again.stdout and "overwrite it" not in again.stdout.lower()


def test_export_into_a_non_empty_dir_asks_with_overwrite(tmp_path):
    """The folder form asked with E_CONFIRM_REQUIRED and --yes, where `poros export <run>` and
    both train forms ask the question as data with --overwrite: one contract for all four."""
    pytest.importorskip("torch")
    run = tmp_path / "run"
    run.mkdir()
    (run / "poros_config.json").write_text(json.dumps({"model_name_or_path": "x"}))
    (run / "adapter_weights.pt").write_bytes(b"")
    out = tmp_path / "merged"
    out.mkdir()
    (out / "old.txt").write_text("keep me")
    asked = _poros("export", str(run), "-o", str(out), "--json")
    said = json.loads(asked.stdout)
    assert asked.returncode == 2 and said["schema"] == "poros.needs_input.v1", asked.stdout
    assert said["verb"] == "export"
    (q,) = said["questions"]
    assert q["id"] == "overwrite" and "not empty" in q["prompt"]
    assert q["choices"][0]["value"] == "--overwrite"
    assert q["choices"][0]["command"] == f"poros export {run} -o {out} --overwrite --json"
    human = _poros("export", str(run), "-o", str(out))  # stdin is /dev/null: asks as data
    assert human.returncode == 2 and "--overwrite" in human.stdout + human.stderr
    assert (out / "old.txt").exists()
    for flag in ("--overwrite", "--yes"):  # --yes: its name before, kept
        again = _poros("export", str(run), "-o", str(out), flag, "--json")
        assert "needs_input" not in again.stdout and "not empty" not in again.stdout, flag


def _error_classes():
    seen, todo = [], [errors.PorosError]
    while todo:
        cls = todo.pop()
        seen.append(cls)
        todo.extend(cls.__subclasses__())
    return seen


def _instance(cls):
    for args in ((), ("msg",), ("arch",), ("model", 22.9, 14.0, "basis"), (1, "checkpoint-1")):
        try:
            return cls(*args)
        except TypeError:
            continue
    raise AssertionError(f"cannot construct {cls}")


@pytest.mark.parametrize("cls", _error_classes(), ids=lambda c: c.__name__)
def test_every_error_class_has_a_code_and_exit(cls, capsys):
    import typer

    from poros.cli import contract
    from poros.cli.errors import _friendly_error

    assert cls.code in EXIT_CODES, cls
    contract.force_json()
    try:
        with pytest.raises(typer.Exit) as exit_info:
            _friendly_error(_instance(cls))
    finally:
        contract.reset()
    assert exit_info.value.exit_code == EXIT_CODES[cls.code]
    assert json.loads(capsys.readouterr().out)["error"]["code"] == cls.code


def test_a_failed_step_check_exits_4(capsys):
    """Exit 4: failed on both run paths, export refuses it, runs continue accepts it. Exit 7
    would read as stopped."""
    import typer

    from poros.cli import contract
    from poros.cli.errors import _friendly_error

    assert EXIT_CODES["E_STEP_CHECK"] == 4
    contract.force_json()
    try:
        with pytest.raises(typer.Exit) as exit_info:
            _friendly_error(
                errors.PorosStepCheckError("layer 3 differs from its forward at step 12")
            )
    finally:
        contract.reset()
    assert exit_info.value.exit_code == 4
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "E_STEP_CHECK"


def test_oom_remediation_is_machine_applicable():
    err = errors.PorosOOMError(block_size=4, seq_len=1024)
    assert err.info.fix == {"block_size": 2, "max_seq_length": 512}


def test_a_gated_repo_is_a_login_to_do_not_a_bug_to_report():
    """transformers raises OSError where the Hub raised GatedRepoError."""
    from poros.cli.errors import _error_code

    assert _error_code(OSError("You are trying to access a gated repo.")) == "E_HUB_AUTH"
    assert _error_code(OSError("something else")) == "E_INTERNAL"


@pytest.mark.cuda_host  # a Mac's machine lists the models it trains, not the registry's rows
def test_machine_lists_the_validated_moe_families_too():
    from poros.core.arch_registry import validated_arch_types

    checks = {c["name"]: c["value"] for c in _one_object(_poros("machine", "--json"),
                                                          "machine")["checks"]}
    moe = checks["Validated MoE archs"].split(", ")
    assert moe == sorted({a.removesuffix("_text") for a in validated_arch_types("moe")})
    assert {"qwen3_moe", "qwen3_5_moe", "olmoe"} <= set(moe)
    assert "qwen2" in checks["Validated archs"].split(", ")


def test_the_human_check_says_why_it_could_not_resolve(tmp_path):
    """The human `check` says the resolver's reason for a model it cannot resolve (offline,
    gated, not found), not "no model_type in config.json" for all of them."""
    proc = _poros("check", "Org/Not-Cached-Anywhere", env={"COLUMNS": "200", "HF_HUB_OFFLINE": "1",
                                                          "POROS_HOME": str(tmp_path)})
    assert "no model_type in config.json" not in proc.stderr and not proc.stdout  # an error
    assert "HF_HUB_OFFLINE" in proc.stderr or "could not reach the Hub" in proc.stderr


def test_json_is_accepted_by_schema_and_version():
    """--json, which every command takes, is an option of `poros schema` and `poros --version`
    too: the schema is JSON already; --version gives its result."""
    assert json.loads(_poros("schema", "config", "--json").stdout)["title"] == "PorosConfig"
    assert json.loads(_poros("--version", "--json").stdout) == {
        "schema": "poros.version.v1", "version": json.loads(
            _poros("agents", "version").stdout)["version"]}
    assert _poros("--version").stdout.startswith("poros ")
