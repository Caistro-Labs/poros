"""Every result validates against the schema `poros schema <name>` publishes for it.

Each `poros agents` command is run as an agent's shell runs it (the real entry point, a
subprocess), in the shapes a CPU and the simulator reach: its result, a refusal and a question.
Each JSON object is held to the schema its own ``schema`` field names, with jsonschema, and may
carry no field that schema does not name. A person's ``--json`` forms are held to the same
schemas. A new command with no shapes here fails ``test_every_agent_command_has_shapes``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poros import jobs
from poros.cli import contract
from poros.cli.schemas import OUTPUTS, RESULTS, JobOutput, result_schema
from poros.cli.verbs import agents
from tests.cli.test_agents import _a_run, _exportable, _poros, _scored

jsonschema = pytest.importorskip(
    "jsonschema", reason="the [dev] extra installs it: pip install jsonschema")

RUN, OTHER = "1105-0900-aaaa", "1105-0901-bbbb"
FAST = {"POROS_DEV_SIMULATE_LOAD": "0", "POROS_DEV_SIMULATE_STEP": "0.01"}
CHAT = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
PAIR = {"prompt": "a", "chosen": "b", "rejected": "c"}
PHI3 = {"model_type": "phi3", "architectures": ["Phi3ForCausalLM"], "hidden_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
        "intermediate_size": 512, "vocab_size": 1000}


@pytest.fixture
def home(tmp_path, monkeypatch):
    for variable in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "me").mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    (work / "d.jsonl").write_text((json.dumps(CHAT) + "\n") * 50)
    (work / "p.jsonl").write_text((json.dumps(PAIR) + "\n") * 50)
    (work / "t.csv").write_text("question,answer\n1,2\n3,4\n")
    (work / "odd.jsonl").write_text(
        (json.dumps(CHAT) + "\n") * 5 + '{"x": 1}\n{not json\n'
        + json.dumps({"messages": [{"role": "user", "content": "only user"}]}) + "\n"
        + json.dumps({"question": "q", "answer": "a"}) + "\n")
    (work / "phi").mkdir()
    (work / "phi" / "config.json").write_text(json.dumps(PHI3))
    return tmp_path


def _strict(schema: dict) -> dict:
    """``schema`` refusing a field it does not name, in the result and in each model under it."""
    strict = copy.deepcopy(schema)
    for part in (strict, *strict.get("$defs", {}).values()):
        if "properties" in part:
            part["additionalProperties"] = False
    return strict


def held(text: str, said: str) -> dict:
    """One result, as it was printed: a single JSON object that its own schema takes."""
    lines = text.strip().splitlines()
    assert len(lines) == 1, f"{said}: {text[:400]!r}"
    out = json.loads(lines[0])
    kind, name, version = out["schema"].split(".")
    assert (kind, version) == ("poros", "v1"), said
    schema = result_schema(name)
    assert schema is not None, f"{said}: no schema is published as {name}"
    errors = [f"{'/'.join(map(str, e.absolute_path)) or '.'}: {e.message[:200]}"
              for e in jsonschema.Draft202012Validator(_strict(schema)).iter_errors(out)]
    assert not errors, f"{said} [{out['schema']}]: " + "; ".join(errors)
    return out


def agent(*args, env=None, **how) -> dict:
    proc = _poros("agents", *args, env={**FAST, **(env or {})}, **how)
    assert proc.stdout.strip(), f"{args}: {proc.stderr[-600:]}"
    out = held(proc.stdout, " ".join(("agents", *args)))
    if out["schema"] == "poros.error.v1":
        assert proc.returncode == out["error"]["exit_code"], args
    return out


def person(*args, env=None) -> dict:
    proc = _poros(*args, "--json", env={**FAST, **(env or {})})
    assert proc.stdout.strip(), f"{args}: {proc.stderr[-600:]}"
    return held(proc.stdout, " ".join(args))


def _a_job(home: Path, job_id: str, state: str):
    """A job its runner holds (its lock taken), as `train` leaves one: tests/cli/test_agents_run."""
    import fcntl

    job_dir = home / "poros" / "jobs" / job_id
    (job_dir / "artifacts").mkdir(parents=True)
    at = "2026-11-05T09:00:00.000+00:00"
    jobs.write_status(job_dir, job_id=job_id, state=state, model="Qwen/Qwen2.5-0.5B", name=None,
                      job_dir=str(job_dir), config=str(job_dir / "config.yaml"),
                      output_dir=str(job_dir / "artifacts"), log=str(job_dir / "log.txt"),
                      gpu="GPU-0", attempt=1, queued_at=at, created_at=at)
    lock = open(job_dir / "runner.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return job_dir, lock


def _schemas(*outs: dict) -> set[str]:
    return {out["schema"] for out in outs}


# Each command's shapes: what it prints for a result, a refusal and, where it asks, a question.


def _version(home):
    return [agent("version"), agent("version", "extra")]


def _machine(home):
    # a simulated card is ready; with none, whatever this machine is (no GPU: not ready, exit 5)
    proc = _poros("agents", "machine", env={"POROS_DEV_SIMULATE": "0"})
    return [agent("machine"), held(proc.stdout, "agents machine, this machine")]


def _model_list(home):
    return [agent("model", "list"), agent("model", "list", "qwen3"),
            agent("model", "list", "no-such-model"), agent("model", "list", "a", "b")]


def _data(home):
    outs = [agent("data", "d.jsonl"), agent("data", "p.jsonl"), agent("data", "t.csv"),
            agent("data", "t.csv", "--convert"), agent("data", "odd.jsonl"),
            agent("data", "odd.jsonl", "--fix"), agent("data", "odd.jsonl", "--fix"),
            agent("data", "odd.jsonl", "--fix", "-o", "clean.jsonl"),
            agent("data", "nope.jsonl"), agent("data")]
    assert _schemas(*outs) == {"poros.data.v1", "poros.error.v1", "poros.needs_input.v1"}
    return outs


def _train(home):
    outs = [agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--steps", "3"),
            agent("train", "qwen2.5-0.5b-instruct", "--data", "p.jsonl", "--steps", "3", "--wait"),
            agent("train", "72b", "--data", "d.jsonl"),  # which model: a question
            agent("train", "qwen2.5-0.5b-instruct", "--data", "nope.jsonl"),
            agent("train", "qwen2.5-0.5b-instruct", "--data", "odd.jsonl"),  # a row a run fails at
            agent("train", "gpt2", "--data", "d.jsonl"),
            agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--step", "3")]
    assert _schemas(*outs) == {"poros.train.v1", "poros.needs_input.v1", "poros.error.v1"}
    return outs


def _run_list(home):
    empty = agent("run", "list")
    _scored(_a_run(), 2.0, 1.5)
    _a_run(OTHER, state="failed")
    return [empty, agent("run", "list"), agent("run", "list", "--limit", "1"),
            agent("run", "list", "--limit", "x")]


def _run_info(home):
    _scored(_a_run(), 2.0, 1.5)
    failed = _a_run(OTHER, state="failed")
    from poros import runs

    runs.update(failed, error={"code": "E_DATASET", "exit_code": 6, "message": "a row"})
    return [agent("run", "info", RUN), agent("run", "info", OTHER), agent("run", "info", "last"),
            agent("run", "info", "nope"), agent("run", "info")]


def _run_watch(home):
    _scored(_a_run(), 2.0, 1.5)
    _a_run(OTHER, state="running")
    _, lock = _a_job(home, "20261105-090200-cccc", "queued")
    try:
        return [agent("run", "watch", RUN), agent("run", "watch", RUN, "--frame"),
                agent("run", "watch", OTHER, "--wait", "1"),
                agent("run", "watch", "20261105-090200-cccc"), agent("run", "watch", "nope")]
    finally:
        lock.close()


def _run_stop(home):
    import threading
    import time

    _a_run()
    job_dir, lock = _a_job(home, "20261105-090000-aaaa", "running")
    _, queued = _a_job(home, "20261105-090100-bbbb", "queued")

    def runner() -> None:  # the job's runner: it sees the stop and records it
        while not (job_dir / "stop").exists():
            time.sleep(0.05)
        jobs.write_status(job_dir, state="cancelled", finished_at=jobs.now())
        lock.close()

    threading.Thread(target=runner, daemon=True).start()
    try:
        stopped = agent("run", "stop", "20261105-090000-aaaa")
    finally:
        lock.close()
        queued.close()
    assert stopped["starts_next"] == "20261105-090100-bbbb"
    return [stopped, agent("run", "stop", RUN), agent("run", "stop", "nope")]


def _run_continue(home):
    done = agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--steps", "3")
    _a_run(state="cancelled")
    outs = [agent("run", "continue", done["run"], "--more", "2"),  # asks: more is new work
            agent("run", "continue", done["run"]), agent("run", "continue", RUN),
            agent("run", "continue", "nope")]
    assert "poros.needs_input.v1" in _schemas(*outs)
    return outs


def _run_report(home):
    _scored(_a_run(), 2.0, 1.5)
    return [agent("run", "report", RUN), agent("run", "report", "nope")]


def _checkpoint_list(home):
    empty = agent("checkpoint", "list")
    _scored(_a_run(), 2.0, 1.5)
    _a_run(OTHER, steps=(1,), final=False)
    return [empty, agent("checkpoint", "list"), agent("checkpoint", "list", "--run-id", RUN),
            agent("checkpoint", "list", "--limit", "1"), agent("checkpoint", "list", "--nope")]


def _checkpoint_info(home):
    _a_run()
    return [agent("checkpoint", "info", f"poros://{RUN}/weights/final"),
            agent("checkpoint", "info", f"{RUN}/weights/10"),
            agent("checkpoint", "info", f"{RUN}/weights/7"), agent("checkpoint", "info")]


def _checkpoint_download(home):
    _a_run()
    path = f"poros://{RUN}/weights/final"
    return [agent("checkpoint", "download", path, "-o", "models"),
            agent("checkpoint", "download", path, "-o", "models"),  # there already: refused
            agent("checkpoint", "download", path, "-o", "models", "--force"),
            agent("checkpoint", "download")]


def _checkpoint_delete(home):
    _a_run()
    _a_run(OTHER, state="cancelled", final=False)
    outs = [agent("checkpoint", "delete", "--run-id", RUN),  # names them, and asks
            agent("checkpoint", "delete", "--run-id", RUN, "-y"),
            agent("checkpoint", "delete", f"{OTHER}/weights/20", "-y"),  # warns: its newest
            agent("checkpoint", "delete"), agent("checkpoint", "delete", "nope/weights/1")]
    assert "poros.needs_input.v1" in _schemas(*outs)
    return outs


def _checkpoint_storage(home):
    empty = agent("checkpoint", "storage")
    _a_run()
    return [empty, agent("checkpoint", "storage"), agent("checkpoint", "storage", "x")]


def _export(home):
    _exportable(home)
    outs = [agent("export", "last", "--output", "./adapter"),
            agent("export", "last", "--output", "./adapter"),  # there already: asks
            agent("export", f"poros://{RUN}/weights/10", "--output", "./at-10"),
            agent("export", "nope"), agent("export", "last", "--merged", "--output", "/")]
    assert outs[0]["schema"] == "poros.export.v1" and "poros.error.v1" in _schemas(*outs)
    return outs


def _eval(home):
    done = agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--steps", "3")
    pairs = agent("train", "qwen2.5-0.5b-instruct", "--data", "p.jsonl", "--steps", "3")
    outs = [agent("eval", done["run"]), agent("eval", pairs["run"]), agent("eval"),
            agent("eval", done["run"], "--data", "d.jsonl", "--rows", "8"),
            agent("eval", done["run"], "--data", "nope.jsonl"), agent("eval", "nope")]
    assert _schemas(*outs) == {"poros.eval.v1", "poros.error.v1"}
    return outs


def _recipes(home):
    done = agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--steps", "3")
    listed = agent("recipes")
    saved = agent("recipes", "save", done["run"])
    return [listed, saved, agent("recipes"), agent("recipes", "show", "nope"),
            agent("recipes", "show", agent("recipes")["setups"][0]["name"])]


def _arch(home):
    outs = [agent("arch", "./phi"), agent("arch", "try", "./phi"), agent("arch", "./nope"),
            agent("arch")]
    assert outs[0]["schema"] == "poros.arch.v1"
    return outs


def _prove(home):
    return [agent("prove"), agent("prove", "./phi"), agent("prove", "./phi", "--unvalidated"),
            agent("prove", "nope.json")]


def _docs(home):
    outs = [agent("docs"), agent("docs", "cli"), agent("docs", "cli", "recipes"),
            agent("docs", "nope"), agent("docs", "cli", "zzzz no such heading")]
    assert _schemas(*outs) == {"poros.docs.v1", "poros.error.v1"}
    return outs


def _schema(home):
    """A document, not a result: the names, and for each a JSON Schema that is one."""
    names = json.loads(_poros("agents", "schema").stdout)["names"]
    assert set(names) >= OUTPUTS.keys() | RESULTS.keys()
    for name in names:
        jsonschema.Draft202012Validator.check_schema(
            json.loads(_poros("agents", "schema", name).stdout))
    return [agent("schema", "zzzz")]


SHAPES = {
    "version": _version, "machine": _machine, "model list": _model_list,
    "data": _data, "train": _train, "run list": _run_list, "run info": _run_info,
    "run watch": _run_watch, "run stop": _run_stop, "run continue": _run_continue,
    "run report": _run_report, "checkpoint list": _checkpoint_list,
    "checkpoint info": _checkpoint_info, "checkpoint download": _checkpoint_download,
    "checkpoint delete": _checkpoint_delete, "checkpoint storage": _checkpoint_storage,
    "export": _export, "eval": _eval, "recipes": _recipes, "arch": _arch, "prove": _prove,
    "docs": _docs, "schema": _schema,
}


def test_every_agent_command_has_shapes():
    assert set(SHAPES) == set(agents.COMMANDS)


@pytest.mark.parametrize("command", sorted(SHAPES))
def test_each_result_validates_against_its_published_schema(home, command):
    assert SHAPES[command](home)  # each one held to its schema as it was read (`held`)


def test_a_queued_job_validates_as_an_agent_and_a_person_see_it(home, monkeypatch, capsys):
    """`train` without --wait prints poros.job.v1, which needs a GPU to reach: its builder is
    held here, in both forms. An agent's has no path into the job's own folder."""
    job = JobOutput(job_id="j", state="queued", model="m", job_dir="/h/j", config="/h/c",
                    output_dir="/h/o", log="/h/l", gpu="0", attempt=1, queued_at="t",
                    next=["poros runs j --live --wait 30"], plan={"steps": 3})
    contract.emit(job)
    whole = held(capsys.readouterr().out, "a person's job")
    assert whole["job_dir"] == "/h/j" and whole["queued_behind"] is None
    monkeypatch.setenv("POROS_AGENT", "1")
    contract.emit(job)
    lean = held(capsys.readouterr().out, "an agent's job")
    assert not {"job_dir", "config", "log", "queued_behind"} & lean.keys()
    assert lean["next"] == ["poros agents run watch j --wait 30"]


def test_a_persons_json_validates_and_carries_every_field(home):
    _scored(_exportable(home), 2.0, 1.5)
    (home / "work" / "run.yaml").write_text(
        "model_name_or_path: Qwen/Qwen2.5-0.5B\ndataset: d.jsonl\noutput_dir: ./out\n")
    outs = [person("machine"), person("models"), person("arch", "qwen2.5-0.5b-instruct"),
            person("data", "d.jsonl"), person("data", "odd.jsonl", "--fix"), person("runs"),
            person("runs", RUN), person("runs", "checkpoints", RUN), person("recipes"),
            person("docs"), person("docs", "cli"), person("arch", "./phi"), person("prove"),
            person("check", "./phi"), person("init", "new.yaml"),
            person("jobs", "list"), person("export", "last", "--format", "adapter"),
            person("data", "nope.jsonl"), person("arch"), person("runs", "nope")]
    for out in outs:  # a person's result leaves nothing out: every field its schema names
        if out["schema"] not in ("poros.error.v1", "poros.needs_input.v1"):
            schema = result_schema(out["schema"].split(".")[1])
            assert schema["properties"].keys() <= out.keys(), out["schema"]
    assert len(_schemas(*outs)) >= 14


def test_required_is_what_both_forms_carry():
    """A field `poros agents` leaves out is optional in the schema, and no other: a parser may
    count on every required one."""
    train = result_schema("train")
    assert {"schema", "output_dir", "events", "metrics", "run", "next"} <= set(train["required"])
    assert not {"learned", "plan", "command", "from", "support"} & set(train["required"])
    jobs_list = result_schema("jobs")
    assert not {"job_dir", "config", "log"} & set(jobs_list["$defs"]["JobOutput"]["required"])
    assert "adapter" not in result_schema("export")["required"]
    assert "do" in result_schema("checkpoints")["required"]
    assert not {"runs", "more", "output"} & set(result_schema("checkpoints")["required"])
    for name in sorted(OUTPUTS.keys() | RESULTS.keys()):
        schema = result_schema(name)
        assert "schema" in schema["required"], name
        # each model under it names its fields: a serializer's return type once left an error
        # as any object at all, which every output "validates" against
        for model, part in schema.get("$defs", {}).items():
            assert "properties" in part or "enum" in part, (name, model)

