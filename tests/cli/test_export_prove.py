"""export and prove through words (simulated trainer; the comparison on real tensors)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poros import runs
from poros.cli import app, contract
from poros.cli.context import Context


@pytest.fixture
def home(tmp_path, monkeypatch):
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    # the simulated CUDA card these tests are written for, whatever the host is: on a Mac the
    # simulator previews the Mac itself (its disks, its table), which is test_mac's to check
    from poros.cli.ui import simulate

    monkeypatch.setenv("POROS_DEV_SIMULATE_GPU", "RTX 4090:24.0")
    simulate._gpu.cache_clear()
    monkeypatch.setenv("POROS_DEV_SIMULATE_STEP", "0.001")
    monkeypatch.setenv("POROS_DEV_SIMULATE_LOAD", "0")
    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=None, unit="GiB", source="unknown", why="offline in tests"),
    )
    monkeypatch.setattr(poros.plan.configs, "read_model_config", lambda *a,
                        **k: {"num_hidden_layers": 24})
    yield tmp_path
    contract.reset()
    simulate._gpu.cache_clear()


def say(argv, home, capsys):
    ctx = Context(cwd=home, fixed_cached=frozenset(), fixed_interactive=False)
    code = app.dispatch(argv, ctx)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert len(lines) == 1, lines
    return code, json.loads(lines[0])


def _finished_run(with_adapter: bool) -> runs.Run:
    rid = runs.new_id()
    d = runs.create(rid, {"model": "Qwen/Qwen2.5-0.5B", "data": "synthetic", "steps": 3})
    runs.update(d, state="succeeded")
    if with_adapter:
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"x")
    return next(r for r in runs.all_runs() if r.id == rid)


def test_export_asks_the_format_then_copies_the_adapter(home, capsys):
    r = _finished_run(with_adapter=True)
    code, obj = say(["export", "last"], home, capsys)
    assert code == 2 and obj["questions"][0]["id"] == "format"
    code, obj = say(["export", "last", "--format", "adapter"], home, capsys)
    assert code == 0 and obj["format"] == "adapter" and "bit for bit" in obj["fidelity"]
    assert (Path(obj["output"]) / "adapter_model.safetensors").read_bytes() == b"x"
    assert obj["run"] == r.id


def test_export_refuses_an_unfinished_run_and_a_nonempty_target(home, capsys):
    r = _finished_run(with_adapter=True)
    runs.update(Path(r.folder), state="running", pid=None)
    code, obj = say(["export", r.id, "--format", "adapter"], home, capsys)
    assert obj["error"]["code"] in ("E_USAGE",)
    runs.update(Path(r.folder), state="succeeded")
    say(["export", r.id, "--format", "adapter"], home, capsys)
    code, obj = say(["export", r.id, "--format", "adapter"], home, capsys)
    assert obj["questions"][0]["id"] == "overwrite"
    assert obj["questions"][0]["choices"][0]["command"].endswith("--overwrite --json")


def test_prove_simulated_is_identical_and_its_receipt_reproduces(home, capsys):
    code, obj = say(["prove"], home, capsys)
    assert code == 0 and obj["identical"] and obj["simulated"]
    assert obj["streamed"]["block_size"] == 4 and obj["resident"]["block_size"] == 24
    code, again = say(["prove", obj["receipt"]], home, capsys)
    assert again["reproduced"] is True and again["pinned_differs"] == {}


def test_prove_refuses_a_model_no_parity_receipt_covers(home, capsys):
    code, obj = say(["prove", "Qwen2.5-7B-Instruct"], home, capsys)
    assert obj["error"]["code"] == "E_USAGE"


def test_the_comparison_is_bitwise(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from poros.cli.verbs import prove_run

    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([1.0, 2.0 + 2**-20])}  # one ulp-scale change
    for name, t in (("a", a), ("b", b)):
        (tmp_path / name).mkdir()
        save_file(t, str(tmp_path / name / "adapter_model.safetensors"))
    ta, tb = prove_run._tensors(tmp_path / "a"), prove_run._tensors(tmp_path / "b")
    assert not torch.equal(ta["w"], tb["w"])
    assert prove_run._digest(ta, [1.0]) != prove_run._digest(tb, [1.0])
    assert prove_run._digest(ta, [1.0]) == prove_run._digest(ta, [1.0])
