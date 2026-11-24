"""Merged export on a Mac, on the CPU: the one rounding, and plain transformers on what it makes."""

import json
import shutil

import pytest

torch = pytest.importorskip("torch")  # the Mac path: torch, transformers, safetensors
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from poros import runs  # noqa: E402
from poros.cli import app, contract, main  # noqa: E402
from poros.cli.context import Context  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.metal import table  # noqa: E402
from poros.metal.export import merged, merges  # noqa: E402
from poros.moe import quant  # noqa: E402
from tests.conftest import cpu_trainer_pick  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

arms = tiny.arms


def _bytes(tensor):
    return bytes(tensor.view(-1).view(torch.uint8))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A tiny base and an adapter the real trainer saved for it, after real updates."""
    root = tmp_path_factory.mktemp("export")
    base = root / "tiny-qwen2"
    tiny.write_checkpoint(base, tiny.tiny_config())
    config = PorosConfig(model_name_or_path=str(base), store="ram", store_path=str(root / "store"),
                         max_seq_length=16, gradient_accumulation_steps=1, max_steps=3,
                         learning_rate=1e-2, warmup_steps=0, save_steps=0, eval_steps=0,
                         output_dir=str(root / "run"), progress=False)
    with pytest.MonkeyPatch.context() as mp:  # module-scoped: before the autouse CPU pick
        cpu_trainer_pick(mp)
        trainer = arms.CpuDouble.from_pretrained(str(base), config=config)
        trainer.train(arms.token_stream(trainer.model.config.vocab_size, 16, 3), steps=3)
        trainer.save(str(root / "run"))
    return base, root / "run", config


def _config_of(run_dir):
    saved = json.loads((run_dir / "poros_config.json").read_text())
    return PorosConfig(**{k: v for k, v in saved.items() if k in PorosConfig.model_fields})


def test_merged_is_the_round_trip_plus_the_delta_rounded_once(run, tmp_path):
    base, run_dir, config = run
    merged(_config_of(run_dir), run_dir, tmp_path)
    before = load_file(base / "model.safetensors")
    after = load_file(tmp_path / "model.safetensors")
    adapter = load_file(run_dir / "adapter_model.safetensors")
    assert before.keys() == after.keys()
    folded = 0
    for name, weight in before.items():
        a = adapter.get(f"base_model.model.{name.removesuffix('.weight')}.lora_A.weight")
        if a is None:
            assert after[name].dtype == weight.dtype and _bytes(after[name]) == _bytes(weight)
            continue
        b = adapter[f"base_model.model.{name.removesuffix('.weight')}.lora_B.weight"]
        seen = quant.dequantize(quant.quantize_tensor(weight), weight.shape, torch.bfloat16)
        delta = config.lora_alpha / config.lora_rank * (b.float() @ a.float())
        want = (seen.float() + delta).to(weight.dtype)
        assert _bytes(after[name]) == _bytes(want)
        assert delta.abs().max() > 0
        folded += 1
    assert folded == 7 * tiny.tiny_config().num_hidden_layers


def test_merged_loads_in_vanilla_transformers_with_the_base_files(run, tmp_path):
    _, run_dir, _ = run
    merged(_config_of(run_dir), run_dir, tmp_path)
    model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    logits = model(torch.arange(8).unsqueeze(0)).logits
    assert logits.shape == (1, 8, tiny.tiny_config().vocab_size) and torch.isfinite(logits).all()


@pytest.mark.parametrize("model, yes", [
    ("Qwen/Qwen2.5-0.5B", True),
    ("Qwen/Qwen2.5-7B-Instruct", True),
    ("Qwen/Qwen2.5-32B", False),
    ("Qwen/Qwen2.5-72B-Instruct", False),
    ("/a/developers/own", True),
])
def test_this_release_merges_up_to_7b(model, yes):
    assert merges(model) is yes


def test_an_adapter_for_another_base_is_refused(run, tmp_path):
    _, run_dir, _ = run
    other = tmp_path / "other"
    other.mkdir()
    adapter = load_file(run_dir / "adapter_model.safetensors")
    save_file({k.replace("q_proj", "x_proj"): v for k, v in adapter.items()},
              other / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="does not match this base"):
        merged(_config_of(run_dir), other, tmp_path / "out")


def _finished_run(model, run_dir):
    run_id = runs.new_id()
    folder = runs.create(run_id, {"model": model, "data": "synthetic", "steps": 3},
                         output_dir=str(run_dir))
    runs.update(folder, state="succeeded")
    return run_id


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    yield Context(cwd=tmp_path, fixed_gpu=table.plan_gpu("apple-m4-10gpu-16gb"),
                  fixed_cached=frozenset(), fixed_interactive=False)
    contract.reset()


def _say(argv, ctx, capsys):
    code = app.dispatch(argv, ctx)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    return code, json.loads(lines[-1])


def test_above_7b_the_format_question_drops_and_the_adapter_is_the_default(cli, tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "adapter_config.json").write_text("{}")
    (run_dir / "adapter_model.safetensors").write_bytes(b"x")
    _finished_run("Qwen/Qwen2.5-72B", run_dir)
    code, obj = _say(["export", "last"], cli, capsys)
    assert code == 0 and obj["format"] == "adapter", obj
    assert "bit for bit" in obj["fidelity"]


def test_up_to_7b_the_question_stays(cli, tmp_path, capsys):
    _finished_run("Qwen/Qwen2.5-7B", tmp_path / "run")
    code, obj = _say(["export", "last"], cli, capsys)
    assert code == 2 and obj["questions"][0]["id"] == "format"


def test_the_config_form_on_a_mac_merges_there_and_refuses_above_7b(run, tmp_path, monkeypatch):
    base, run_dir, _ = run
    monkeypatch.setattr(main.sys, "platform", "darwin")
    done = CliRunner().invoke(main.app, ["export", str(run_dir), "-o", str(tmp_path / "out")])
    assert done.exit_code == 0, done.output
    assert (tmp_path / "out" / "model.safetensors").is_file()

    big = tmp_path / "big"
    shutil.copytree(run_dir, big)
    saved = json.loads((big / "poros_config.json").read_text())
    saved["model_name_or_path"] = "Qwen/Qwen2.5-72B"
    (big / "poros_config.json").write_text(json.dumps(saved))
    done = CliRunner().invoke(main.app, ["export", str(big), "-o", str(tmp_path / "out2")])
    assert done.exit_code == 4, done.output
    assert not (tmp_path / "out2").exists()
