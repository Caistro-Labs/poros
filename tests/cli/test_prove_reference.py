"""`poros prove` on CUDA compares the streamed run with a true resident reference (every layer on
the GPU for the whole run, ordinary autograd: poros.validate.training_gate's resident arm), and
its receipt says so; a model the card cannot hold resident is said before anything runs, and the
comparison of two streamed block sizes runs only by name (--block-sizes), never as "resident"."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from poros.cli.context import Context  # noqa: E402
from poros.cli.verbs import prove_run  # noqa: E402
from poros.cli.verbs.prove import ProveAction  # noqa: E402
from poros.core.refusal import Refusal  # noqa: E402
from poros.validate import training_gate  # noqa: E402


@pytest.fixture
def cuda_here(tmp_path, monkeypatch):
    """A CUDA machine as prove sees it, with no GPU work: the arms are the test's."""
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POROS_DEV_SIMULATE", raising=False)
    monkeypatch.setattr("poros.preflight.command.device_refusal", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (20 * 2**30, 24 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 3 * 2**30)
    monkeypatch.setattr(prove_run, "_layers", lambda model: 24)
    monkeypatch.setattr("poros.cli.sheets.listings.largest_certified", lambda ctx: None)
    return Context(cwd=tmp_path, fixed_cached=frozenset(), fixed_interactive=False)


def _gate_arms(monkeypatch, *, differ=False):
    built, ran = [], []
    monkeypatch.setattr(training_gate, "build_trainer", lambda model, **kw: built.append(kw) or
                        object())

    def run_arm(trainer, arm, **kw):
        ran.append(arm)
        w = torch.ones(2, 2) + (1 if differ and arm == "blockwise" else 0)
        loss = [torch.tensor(2.0), torch.tensor(1.5)]
        return {"losses": loss, "grads": {"a": w}, "weights": {"a": w}, "opt_state": {"s": w},
                "seconds": 4.0, "h2d_bytes_total": None}

    monkeypatch.setattr(training_gate, "run_arm", run_arm)
    return built, ran


def test_the_reference_is_training_gates_resident_arm_and_the_receipt_says_so(cuda_here,
                                                                              monkeypatch):
    built, ran = _gate_arms(monkeypatch)
    out = prove_run.run(ProveAction(model="Qwen/Qwen2.5-0.5B", steps=2, seq_len=32), cuda_here)
    assert ran == ["resident", "blockwise"] and all(b["block_size"] == 4 for b in built)
    assert out.identical and out.reference == "resident" and out.one_block is None
    assert out.resident.block_size == 24 and out.streamed.block_size == 4
    assert out.resident.peak_vram_gib == 3.0 and out.resident.s_per_step == 2.0
    assert out.compared == ["losses", "1 adapter gradients", "1 adapter tensors",
                            "1 optimizer tensors"]
    receipt = json.loads(Path(out.receipt).read_text())
    assert receipt["reference"] == "resident" and receipt["digest"]


def test_a_difference_is_named_by_its_surface(cuda_here, monkeypatch):
    _gate_arms(monkeypatch, differ=True)
    out = prove_run.run(ProveAction(model="Qwen/Qwen2.5-0.5B", steps=2, seq_len=32), cuda_here)
    assert not out.identical and out.exit_code == 1
    assert out.differs == ["adapter gradients", "adapter weights", "optimizer state"]


def test_a_model_the_card_cannot_hold_is_said_before_anything_runs(cuda_here, monkeypatch,
                                                                  recipe_configs_cached):
    built, _ = _gate_arms(monkeypatch)
    monkeypatch.setattr(prove_run, "resident_floor_gib", lambda model: 41.5)
    with pytest.raises(Refusal) as refused:
        prove_run.run(ProveAction(model="Qwen/Qwen2.5-72B", steps=2, seq_len=32), cuda_here)
    assert refused.value.code == "E_OOM_PREDICTED" and built == []
    assert "at least 41.50 GiB, and 20.00 GiB is free" in refused.value.reason
    # its one-block arm holds every layer too, past the card: no --block-sizes is offered
    assert not refused.value.fix and "nor can prove --block-sizes" in refused.value.context[0]
    monkeypatch.setattr(prove_run, "one_block_floor_gib", lambda model: 12.0)  # one that fits
    with pytest.raises(Refusal) as refused:
        prove_run.run(ProveAction(model="Qwen/Qwen2.5-72B", steps=2, seq_len=32), cuda_here)
    assert refused.value.fix == "poros prove Qwen/Qwen2.5-72B --block-sizes"


def test_block_sizes_runs_two_streamed_arms_and_never_says_resident(cuda_here, monkeypatch):
    blocks = []

    def arm(a, block, out):
        blocks.append(block)
        out.mkdir(parents=True)
        from safetensors.torch import save_file

        save_file({"a": torch.ones(2)}, str(out / "adapter_model.safetensors"))
        return ({"peak_vram_allocated_gb": 1.0, "avg_step_time_ms": 500, "final_loss": 1.5},
                [2.0, 1.5])

    monkeypatch.setattr(prove_run, "_arm", arm)
    out = prove_run.run(ProveAction(model="Qwen/Qwen2.5-0.5B", steps=2, seq_len=32,
                                    reference="block_sizes"), cuda_here)
    assert blocks == [4, 24] and out.identical
    assert out.reference == "block_sizes" and out.resident is None
    assert out.one_block.block_size == 24
    assert out.one_block.peak_vram_gib == pytest.approx(1e9 / 2**30, abs=1e-3)  # GB -> GiB
    receipt = json.loads(Path(out.receipt).read_text())
    assert receipt["reference"] == "block_sizes" and receipt["resident"] is None
    from poros.cli import views
    from poros.cli.ui.style import resolve
    from poros.cli.ui.text import render_line

    shown = "\n".join(render_line(r, resolve(depth="plain")) for r in views.proved(out, 100))
    assert "resident" not in shown and "one block" in shown
    assert "the block size changes nothing" in shown


def test_block_sizes_is_a_flag_and_an_old_receipt_reruns_as_what_it_was(tmp_path):
    from poros.cli.verbs import prove
    from poros.cli.words.grammar import parse

    ctx = Context(cwd=tmp_path, fixed_cached=frozenset(), fixed_interactive=False)
    if ctx.unified:
        pytest.skip("a Mac compares its RAM and SSD sources")
    a = prove.resolve(parse(["prove", "qwen2.5-0.5b", "--block-sizes"]), ctx).action
    assert a.reference == "block_sizes" and "--block-sizes" in a.command()
    old = tmp_path / "prove-old.json"
    old.write_text(json.dumps({"schema": "poros.prove.v1", "model": "Qwen/Qwen2.5-0.5B",
                               "steps": 20, "seq_len": 256}))
    assert prove.resolve(parse(["prove", str(old)]), ctx).action.reference == "block_sizes"


def test_the_floor_is_the_planners_resident_hold():
    floor = prove_run.resident_floor_gib("Qwen/Qwen2.5-7B")
    assert floor is None or 3.0 < floor < 12.0  # 28 layers in NF4, tables, the adapters' state


def test_a_mac_receipt_names_no_reference_and_each_cuda_kind_its_own(cuda_here, monkeypatch):
    """A Mac's prove says its arms by method (ram-vs-ssd, ssd-twice): its receipt's reference is
    null. A CUDA receipt says resident or block_sizes, as it ran."""
    from poros.cli.verbs.prove import Arm, MacProveOutput

    arm = Arm(block_size=1, peak_vram_gib=1.0, s_per_step=1.0, final_loss=1.0)
    for method in ("ram-vs-ssd", "ssd-twice"):
        mac = MacProveOutput(model="m", steps=2, seq_len=32, simulated=False, resident=arm,
                             streamed=arm, identical=True, compared=[], receipt="", env={},
                             method=method, witness={}, machine="m4")
        assert mac.model_dump(mode="json", by_alias=True)["reference"] is None
    _gate_arms(monkeypatch)
    resident = prove_run.run(ProveAction(model="Qwen/Qwen2.5-0.5B", steps=2, seq_len=32),
                             cuda_here)
    assert json.loads(Path(resident.receipt).read_text())["reference"] == "resident"

    def one(a, block, out):
        out.mkdir(parents=True)
        from safetensors.torch import save_file

        save_file({"a": torch.ones(2)}, str(out / "adapter_model.safetensors"))
        return {"final_loss": 1.0}, [1.0]

    monkeypatch.setattr(prove_run, "_arm", one)
    blocks = prove_run.run(ProveAction(model="Qwen/Qwen2.5-0.5B", steps=2, seq_len=32,
                                       reference="block_sizes"), cuda_here)
    assert json.loads(Path(blocks.receipt).read_text())["reference"] == "block_sizes"


def test_prove_names_the_recipe_certified_here_or_nothing(cuda_here, monkeypatch):
    """prove's "certified" line is largest_certified (certified.json, as poros recipes reads
    it): the recipe and the words around it where there is one, no line where there is none,
    and never the planner's "trains up to"."""
    from poros.cli import views
    from poros.cli.ui.style import resolve
    from poros.cli.ui.text import render_line

    def shown(out) -> str:
        return "\n".join(render_line(r, resolve(depth="plain")) for r in views.proved(out, 100))

    _gate_arms(monkeypatch)
    out = prove_run.run(ProveAction(model="Qwen/Qwen2.5-0.5B", steps=2, seq_len=32), cuda_here)
    assert out.largest_certified is None and "certified" not in shown(out).lower()
    monkeypatch.setattr("poros.cli.sheets.listings.largest_certified", lambda ctx: "big-32b")
    out = prove_run.run(ProveAction(model="Qwen/Qwen2.5-0.5B", steps=2, seq_len=32), cuda_here)
    assert out.largest_certified == "big-32b"
    text = shown(out)
    assert "big-32b  the largest model certified on this card; poros recipes lists all" in text
    assert "trains" not in text and "up to" not in text
