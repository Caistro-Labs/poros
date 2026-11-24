"""Merged export writes the model the run trained, bf16, one tensor at a time.

(a) each weight is dequant(NF4(W)) + scaling * B @ A, added in fp32 and
    rounded once to the checkpoint dtype;
(b) the merged checkpoint's logits are the trained model's on the
    dequantized base, up to that one rounding;
(c) peak host memory stays at one shard plus the adapter;
(d) plain ``transformers`` loads the output.

Dense (PEFT and native adapters) and MoE, on tiny random models. CPU only:
the NF4 round trip here is ``poros.moe.quant`` (bitsandbytes' CPU kernels,
bit for bit); the GPU receipt in docs/validation/export/ checks the CUDA
round trip against the trainer's own dequantized weights.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosExportError  # noqa: E402
from poros.moe import quant  # noqa: E402
from poros.train.export import export_merged_dense  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BF16 = torch.bfloat16
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _qwen2(**kw):
    knobs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen2Config(**{**knobs, **kw})


def _base(path, config=None, shard="40KB"):
    torch.manual_seed(0)
    Qwen2ForCausalLM(config or _qwen2()).to(BF16).save_pretrained(path, max_shard_size=shard)
    return path


def _randomize(model, seed=1, scale=0.05):
    """Stand in for training: every trainable tensor gets non-zero values."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=g) * scale)


def _write_config(run, **fields):
    run.mkdir(parents=True, exist_ok=True)
    cfg = PorosConfig(**fields)
    (run / "poros_config.json").write_text(cfg.model_dump_json())
    return cfg


def _peft_run(base, run, *, bias="none", targets="all-linear"):
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(base, dtype=BF16)
    peft = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=targets, bias=bias,
        task_type="CAUSAL_LM",
    ))
    _randomize(peft)
    peft.save_pretrained(run)
    cfg = _write_config(run, model_name_or_path=str(base), quantization="nf4", lora_rank=4,
                        lora_alpha=8, lora_bias=bias, adapter_backend="peft")
    return peft, cfg


def _native_run(base, run):
    from poros.train.adapters import NativeAdapterManager, get_trainable_state_dict

    model = AutoModelForCausalLM.from_pretrained(base, dtype=BF16)
    # The targets PorosModel resolves: every linear.
    NativeAdapterManager(model=model, rank=4, alpha=8, dropout=0.0,
                         target_modules="all-linear").inject()
    _randomize(model)
    run.mkdir(parents=True, exist_ok=True)
    state = get_trainable_state_dict(model)
    torch.save(state, run / "adapter_weights.pt")
    cfg = _write_config(run, model_name_or_path=str(base), quantization="nf4",
                        lora_rank=4, lora_alpha=8, adapter_backend="native")
    return state, cfg


def _checkpoint(path):
    out = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        out.update(load_file(shard))
    return out


def _seen(w):
    """The weight NF4 training saw: W's NF4 round trip, in bf16."""
    return quant.dequantize(quant.quantize_tensor(w.to(BF16)), w.shape, BF16)


def _merged(seen, a, b, scaling):
    """The rule: added in fp32, rounded once to bf16."""
    return (seen.float() + scaling * (b.float() @ a.float())).to(BF16)


def _linear_weight(name):
    return name.endswith(".weight") and name.split(".")[-2] in LINEARS


def _same_layout(base, out):
    names = sorted(p.name for p in Path(base).iterdir())
    assert names == sorted(p.name for p in Path(out).iterdir())
    for name in names:
        if not name.endswith(".safetensors"):
            assert (Path(base) / name).read_bytes() == (Path(out) / name).read_bytes(), name


# ---------------------------------------------------------------------------
# (a) + (d): every tensor, exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", ["none", "all"])
def test_peft_merge_is_the_nf4_base_plus_delta_rounded_once(tmp_path, bias):
    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _, cfg = _peft_run(base, run, bias=bias)
    out = tmp_path / "merged"

    report = export_merged_dense(cfg, run, str(out), nf4_device="cpu")

    adapter = load_file(run / "adapter_model.safetensors")
    before, after = _checkpoint(base), _checkpoint(out)
    assert before.keys() == after.keys()
    trained_biases = 0
    for name, w in before.items():
        got = after[name]
        assert got.dtype == w.dtype == BF16 and got.shape == w.shape, name
        prefix = "base_model.model." + name.rsplit(".", 1)[0]
        if _linear_weight(name):
            a, b = adapter[f"{prefix}.lora_A.weight"], adapter[f"{prefix}.lora_B.weight"]
            assert torch.equal(got, _merged(_seen(w), a, b, 8 / 4)), name
        elif bias == "all" and name.endswith(".bias"):
            trained = adapter[f"{prefix}.base_layer.bias"]
            assert torch.equal(got, trained.to(BF16)) and not torch.equal(got, w), name
            trained_biases += 1
        else:  # embeddings, norms, the output head: byte for byte
            assert torch.equal(got, w), name
    assert report.merged == 2 * len(LINEARS) and report.dequantized == 0
    assert report.biases == trained_biases == (6 if bias == "all" else 0)
    assert report.nf4_roundtrip == "cpu" and report.adapter_format == "peft"
    _same_layout(base, out)
    assert len(list(out.glob("*.safetensors"))) > 1
    model = AutoModelForCausalLM.from_pretrained(out)  # (d)
    assert model.dtype == BF16


def test_non_target_linears_are_written_as_the_nf4_weights_training_saw(tmp_path):
    """Training quantized every linear, LoRA target or not: the output is that model."""
    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _, cfg = _peft_run(base, run, targets=["q_proj", "v_proj"])
    report = export_merged_dense(cfg, run, str(tmp_path / "merged"), nf4_device="cpu")

    adapter = load_file(run / "adapter_model.safetensors")
    before, after = _checkpoint(base), _checkpoint(tmp_path / "merged")
    for name, w in before.items():
        module = name.rsplit(".", 1)[0]
        if name.endswith(("q_proj.weight", "v_proj.weight")):
            a = adapter[f"base_model.model.{module}.lora_A.weight"]
            b = adapter[f"base_model.model.{module}.lora_B.weight"]
            assert torch.equal(after[name], _merged(_seen(w), a, b, 2.0)), name
        elif _linear_weight(name):
            assert torch.equal(after[name], _seen(w)), name
        else:
            assert torch.equal(after[name], w), name
    assert (report.merged, report.dequantized) == (4, 10)


def test_native_adapter_merges_by_the_same_rule(tmp_path):
    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    state, cfg = _native_run(base, run)
    report = export_merged_dense(cfg, run, str(tmp_path / "merged"), nf4_device="cpu")

    before, after = _checkpoint(base), _checkpoint(tmp_path / "merged")
    for name, w in before.items():
        module = name.rsplit(".", 1)[0]
        if _linear_weight(name):
            a, b = state[f"{module}.lora_A"], state[f"{module}.lora_B"]
            assert torch.equal(after[name], _merged(_seen(w), a, b, 2.0)), name
        else:
            assert torch.equal(after[name], w), name
    assert report.adapter_format == "native" and report.merged == 14


@pytest.mark.parametrize("model_type", ["qwen3", "gemma3_text", "qwen3_5_text", "gemma4_text"])
def test_every_validated_dense_family_merges_and_reloads(tmp_path, model_type):
    from transformers import CONFIG_MAPPING

    if model_type not in CONFIG_MAPPING:
        pytest.skip(f"{model_type} needs a newer transformers")
    config = CONFIG_MAPPING[model_type](
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).to(BF16).save_pretrained(
        tmp_path / "base", max_shard_size="200KB")
    peft, cfg = _peft_run(tmp_path / "base", tmp_path / "run")
    pairs = sum(n.endswith("lora_A.default.weight") for n, _ in peft.named_parameters())

    report = export_merged_dense(cfg, tmp_path / "run", str(tmp_path / "merged"),
                                 nf4_device="cpu")

    assert report.merged == pairs > 0 and report.dequantized == 0
    assert AutoModelForCausalLM.from_pretrained(tmp_path / "merged").dtype == BF16


# ---------------------------------------------------------------------------
# (b): the logits of the model training ran
# ---------------------------------------------------------------------------


def test_merged_logits_are_the_trained_models_up_to_one_bf16_rounding(tmp_path):
    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    peft, cfg = _peft_run(base, run)
    export_merged_dense(cfg, run, str(tmp_path / "merged"), nf4_device="cpu")

    # The trained model as training ran it: each quantized linear is its NF4
    # round trip under the (unmerged) LoRA. fp32 compute, so the merge's one
    # bf16 rounding is the only difference left.
    with torch.no_grad():
        for name, module in peft.named_modules():
            if name.endswith("base_layer"):
                module.weight.copy_(_seen(module.weight))
    trained = peft.float().eval()
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.float32).eval()
    ids = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = trained(input_ids=ids).logits
        got = merged(input_ids=ids).logits
        with trained.disable_adapter():
            untrained = trained(input_ids=ids).logits
    diff = (got - want).abs().max().item()
    effect = (want - untrained).abs().max().item()
    assert diff < 2e-2, diff  # bf16 weights: 2^-9 relative per weight
    assert diff < effect / 50, (diff, effect)  # far below what the adapter changes


# ---------------------------------------------------------------------------
# Refusals: before anything is written
# ---------------------------------------------------------------------------


def test_a_truncated_adapter_is_refused_before_any_output(tmp_path):
    from safetensors.torch import save_file

    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _, cfg = _peft_run(base, run)
    path = run / "adapter_model.safetensors"
    state = load_file(path)
    state.pop(sorted(state)[0])
    save_file(state, path)
    out = tmp_path / "merged"
    with pytest.raises(PorosExportError, match="partial|truncated"):
        export_merged_dense(cfg, run, str(out), nf4_device="cpu")
    assert not out.exists() or not any(out.iterdir())


def test_an_adapter_for_another_base_is_refused(tmp_path):
    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _, cfg = _peft_run(base, run)
    other = _base(tmp_path / "other", config=_qwen2(hidden_size=32, intermediate_size=64))
    cfg.model_name_or_path = str(other)
    with pytest.raises(PorosExportError, match="shape"):
        export_merged_dense(cfg, run, str(tmp_path / "merged"), nf4_device="cpu")


def test_the_base_itself_is_never_the_output(tmp_path):
    base = _base(tmp_path / "base")
    _, cfg = _peft_run(base, tmp_path / "run")
    with pytest.raises(PorosExportError, match="base checkpoint itself"):
        export_merged_dense(cfg, tmp_path / "run", str(base), nf4_device="cpu")


# ---------------------------------------------------------------------------
# MoE: the experts, by the same rule
# ---------------------------------------------------------------------------


def _moe_base(path, *, hidden=64, mid=32, experts=8, layers=2, shard="60KB"):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=2 * hidden,
        moe_intermediate_size=mid, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_experts=experts, num_experts_per_tok=2,
    )
    Qwen3MoeForCausalLM(config).to(BF16).save_pretrained(path, max_shard_size=shard)
    return config


def _moe_adapter(config, rank=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    e, h, m = config.num_experts, config.hidden_size, config.moe_intermediate_size
    shapes = {"lora_A_gate_up": (e, rank, h), "lora_B_gate_up": (e, 2 * m, rank),
              "lora_A_down": (e, rank, m), "lora_B_down": (e, h, rank)}
    return {
        f"model.layers.{i}.mlp.adapter.{f}": (torch.randn(s, generator=g) * 0.05).to(BF16)
        for i in range(config.num_hidden_layers) for f, s in shapes.items()
    }


def test_moe_experts_are_their_nf4_round_trip_plus_delta_rounded_once(tmp_path):
    from poros.moe.export import export_merged

    config = _moe_base(tmp_path / "base")
    adapter = _moe_adapter(config)
    export_merged(str(tmp_path / "base"), adapter, str(tmp_path / "merged"), scaling=2.0)

    before, after = _checkpoint(tmp_path / "base"), _checkpoint(tmp_path / "merged")
    assert before.keys() == after.keys()
    m = config.moe_intermediate_size
    for name, w in before.items():
        if ".mlp.experts." not in name:
            assert torch.equal(after[name], w), name
            continue
        layer, expert, proj = name.split(".mlp.experts.")[0], int(name.split(".")[5]), name
        f = {k: adapter[f"{layer}.mlp.adapter.{k}"][expert] for k in
             ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down")}
        if proj.endswith("down_proj.weight"):
            want = _merged(_seen(w), f["lora_A_down"], f["lora_B_down"], 2.0)
        else:  # gate and up are quantized fused, as the MoE load fuses them
            prefix = name.rsplit(".", 2)[0]
            fused = torch.cat([before[f"{prefix}.gate_proj.weight"],
                               before[f"{prefix}.up_proj.weight"]])
            full = _merged(_seen(fused), f["lora_A_gate_up"], f["lora_B_gate_up"], 2.0)
            want = full[:m] if proj.endswith("gate_proj.weight") else full[m:]
        assert torch.equal(after[name], want), name
    _same_layout(tmp_path / "base", tmp_path / "merged")
    AutoModelForCausalLM.from_pretrained(tmp_path / "merged")  # (d)


# ---------------------------------------------------------------------------
# (c): peak host memory, measured in a fresh process
# ---------------------------------------------------------------------------

_MEASURE = textwrap.dedent("""
    import json, sys
    import torch
    from poros.core.config import PorosConfig
    from poros.moe.export import export_merged
    from poros.train.export import export_merged_dense

    def status(key):
        for line in open("/proc/self/status"):
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024

    def run(kind, base, adapter, out):
        if kind == "dense":
            cfg = PorosConfig(**json.load(open(adapter + "/poros_config.json")))
            export_merged_dense(cfg, adapter, out, nf4_device="cpu")
        else:
            state = torch.load(adapter, weights_only=True)
            export_merged(base, state, out, scaling=2.0)

    kind, small, big = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
    run(kind, *small)  # imports, codebooks, transformers' model classes
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")  # reset the peak (VmHWM) to the current RSS
    before = status("VmRSS")
    run(kind, *big)
    print(json.dumps({"peak_over_start": status("VmHWM") - before}))
""")


def _peak(kind, small, big):
    env = {**os.environ, "PYTHONPATH": str(REPO), "MALLOC_MMAP_THRESHOLD_": "65536"}
    res = subprocess.run(
        [sys.executable, "-c", _MEASURE, kind, json.dumps(small), json.dumps(big)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])["peak_over_start"]


def _bytes(paths):
    return [p.stat().st_size for p in paths]


@pytest.mark.skipif(not Path("/proc/self/clear_refs").exists(), reason="needs Linux /proc")
def test_dense_export_peaks_at_one_shard_plus_the_adapter(tmp_path):
    big_cfg = _qwen2(vocab_size=4096, hidden_size=512, intermediate_size=1024,
                     num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=8)
    base = _base(tmp_path / "base", config=big_cfg, shard="32MB")
    _peft_run(base, tmp_path / "run")
    small = _base(tmp_path / "small")
    _peft_run(small, tmp_path / "small_run")
    shards = _bytes(base.glob("*.safetensors"))
    adapter = (tmp_path / "run" / "adapter_model.safetensors").stat().st_size
    assert len(shards) >= 4  # the whole model is several shards

    peak = _peak("dense", [str(small), str(tmp_path / "small_run"), str(tmp_path / "m0")],
                 [str(base), str(tmp_path / "run"), str(tmp_path / "m1")])

    assert peak < max(shards) + adapter, (peak, max(shards), adapter, sum(shards))


@pytest.mark.skipif(not Path("/proc/self/clear_refs").exists(), reason="needs Linux /proc")
def test_moe_export_peaks_at_one_shard_plus_the_adapter(tmp_path):
    config = _moe_base(tmp_path / "base", hidden=256, mid=256, experts=32, layers=4,
                       shard="16MB")
    torch.save(_moe_adapter(config), tmp_path / "adapter.pt")
    small = _moe_base(tmp_path / "small")
    torch.save(_moe_adapter(small), tmp_path / "small.pt")
    shards = _bytes((tmp_path / "base").glob("*.safetensors"))
    adapter = (tmp_path / "adapter.pt").stat().st_size
    assert len(shards) >= 3

    peak = _peak("moe", [str(tmp_path / "small"), str(tmp_path / "small.pt"), str(tmp_path / "m0")],
                 [str(tmp_path / "base"), str(tmp_path / "adapter.pt"), str(tmp_path / "m1")])

    assert peak < max(shards) + adapter, (peak, max(shards), adapter, sum(shards))


# ---------------------------------------------------------------------------
# The CLI routes a dense run here
# ---------------------------------------------------------------------------


@pytest.mark.cuda_host(reason="a Mac's merged export first checks the base against the revision "
                               "its run recorded (poros.metal.same_base); this run recorded none")
def test_poros_export_merges_a_dense_run_shard_by_shard(tmp_path, cpu_only):
    from typer.testing import CliRunner

    from poros.cli.main import app

    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _peft_run(base, run)
    out = tmp_path / "merged"
    res = CliRunner().invoke(app, ["export", str(run), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert "NF4 base rebuilt on the CPU" in res.output
    assert sorted(p.name for p in out.glob("*.safetensors")) == sorted(
        p.name for p in base.glob("*.safetensors"))
    assert AutoModelForCausalLM.from_pretrained(out).dtype == BF16


@pytest.mark.parametrize("given, cwd", [(".", "merged"), ("./", "merged"), ("merged", ".")])
def test_an_output_named_by_a_relative_path_or_dot_gets_the_merge(tmp_path, monkeypatch, given,
                                                                   cwd):
    """`cd merged && poros export ../run -o .` works: "." has no name to stage a sibling beside,
    so the merge lands in the folder the path names."""
    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _, cfg = _native_run(base, run)
    out = tmp_path / "merged"
    out.mkdir()
    monkeypatch.chdir(tmp_path / cwd)
    export_merged_dense(cfg, run, given, nf4_device="cpu")
    assert sorted(p.name for p in out.glob("*.safetensors")) == sorted(
        p.name for p in base.glob("*.safetensors"))
    assert not list(tmp_path.glob("**/*.partial"))


def test_the_root_as_output_is_refused_before_any_download(tmp_path, monkeypatch):
    """"/" has no name to stage the merge beside; `poros export -o /` refuses it
    as a usage error before the base's size is asked of the Hub or anything is downloaded (the
    verb form: tests/cli/test_agents_deletes_locks_and_folders.py)."""
    from typer.testing import CliRunner

    from poros.cli.main import app
    from poros.core.errors import EXIT_CODES
    from poros.plan import configs
    from poros.train import export as ex

    monkeypatch.setattr(configs, "download_size", lambda *a, **k: pytest.fail("asked the Hub"))
    monkeypatch.setattr(ex, "export_merged_dense", lambda *a, **k: pytest.fail("merged"))
    run = tmp_path / "run"
    _write_config(run, model_name_or_path="Org/Model", quantization="nf4")
    res = CliRunner().invoke(app, ["export", str(run), "-o", "/", "--yes"])
    assert res.exit_code == EXIT_CODES["E_USAGE"], res.output
    assert "root" in res.output


def _fail_on_shard(monkeypatch, n, exc):
    """The n-th shard's write raises ``exc`` after its .partial is open (a full disk, mid-file)."""
    from poros.train import export as ex

    real, calls = ex._write_shard, []

    def write(reader, shard, src, tmp, header, transform):
        calls.append(shard)
        if len(calls) == n:
            with open(tmp, "wb") as f:
                f.write(header)
            raise exc
        return real(reader, shard, src, tmp, header, transform)

    monkeypatch.setattr(ex, "_write_shard", write)
    return calls


@pytest.mark.parametrize("into", ["new", "existing"])
def test_a_full_disk_mid_export_leaves_no_partial_and_no_half_model(tmp_path, monkeypatch, into):
    """A write that fails (ENOSPC on the second shard) takes
    back every file the export made, .partial and finished shards alike; what the folder held
    before stays."""
    import errno

    base = _base(tmp_path / "base")
    assert len(list(base.glob("*.safetensors"))) >= 2
    run = tmp_path / "run"
    _, cfg = _native_run(base, run)
    out = tmp_path / "merged"
    if into == "existing":
        out.mkdir()
        (out / "notes.txt").write_text("mine")
    calls = _fail_on_shard(monkeypatch, 2, OSError(errno.ENOSPC, "No space left on device"))
    with pytest.raises(OSError, match="No space"):
        export_merged_dense(cfg, run, str(out), nf4_device="cpu")
    assert len(calls) == 2
    if into == "new":
        assert not out.exists()
    else:
        assert sorted(p.name for p in out.iterdir()) == ["notes.txt"]


def test_a_merge_with_no_room_is_refused_before_the_skeleton(tmp_path, monkeypatch):
    import shutil as _shutil
    from collections import namedtuple

    from poros.core.errors import PorosDiskError
    from poros.train import export as ex

    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _, cfg = _native_run(base, run)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(_shutil, "disk_usage", lambda p: usage(10**9, 10**9 - 1000, 1000))
    monkeypatch.setattr(ex, "_skeleton", lambda *a, **k: pytest.fail("built the skeleton"))
    out = tmp_path / "merged"
    with pytest.raises(PorosDiskError, match="needs .* GB free"):
        export_merged_dense(cfg, run, str(out), nf4_device="cpu")
    assert not out.exists()


def test_a_moe_merge_that_fails_takes_back_what_it_wrote(tmp_path, monkeypatch):
    from poros.moe.export import export_merged

    config = _moe_base(tmp_path / "base")
    calls = _fail_on_shard(monkeypatch, 2, KeyboardInterrupt())
    out = tmp_path / "merged"
    with pytest.raises(KeyboardInterrupt):
        export_merged(str(tmp_path / "base"), _moe_adapter(config), str(out), scaling=2.0)
    assert len(calls) == 2 and not out.exists()


def test_a_failed_overwrite_leaves_the_last_model_whole_and_a_good_one_replaces_it(
        tmp_path, monkeypatch):
    """An export over an earlier merged model never leaves a mix of new and old shards: the
    merge is written whole beside the folder first, so a failure leaves the earlier model byte
    for byte, and a success replaces its shards and keeps the folder's other files."""
    import errno

    base = _base(tmp_path / "base")
    run = tmp_path / "run"
    _, cfg = _native_run(base, run)
    out = tmp_path / "merged"
    export_merged_dense(cfg, run, str(out), nf4_device="cpu")
    (out / "notes.txt").write_text("mine")
    for shard in out.glob("*.safetensors"):  # an older model's bytes, to tell from a new one's
        shard.write_bytes(b"old " + shard.name.encode())
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    _fail_on_shard(monkeypatch, 2, OSError(errno.ENOSPC, "No space left on device"))
    with pytest.raises(OSError):
        export_merged_dense(cfg, run, str(out), nf4_device="cpu")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not (tmp_path / "merged.partial").exists()
    monkeypatch.undo()
    export_merged_dense(cfg, run, str(out), nf4_device="cpu")
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert after["notes.txt"] == b"mine"
    assert all(not v.startswith(b"old ") for k, v in after.items() if k.endswith(".safetensors"))


@pytest.mark.parametrize("free, fetch, total, refused", [
    (10 * 10**9, 15 * 10**9, 20 * 10**9, True),   # the download alone does not fit the cache
    (30 * 10**9, 15 * 10**9, 20 * 10**9, True),   # it does, but not with the merge beside it
    (40 * 10**9, 15 * 10**9, 20 * 10**9, False),  # both fit the one disk
])
def test_a_merge_checks_both_disks_before_it_downloads_its_base(tmp_path, monkeypatch, free,
                                                                fetch, total, refused):
    """The merge checks the Hub's sizes against the free cache and the output disk before it
    downloads its base, so a base past the free cache is refused, not fetched."""
    import shutil as _shutil
    from collections import namedtuple

    from poros.core.errors import PorosDiskError
    from poros.plan import configs
    from poros.train import export as ex

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(configs, "weights_on_hub", lambda m, r=None: (total, fetch))
    monkeypatch.setattr(_shutil, "disk_usage", lambda p: usage(10**13, 10**13 - free, free))
    monkeypatch.setattr(ex, "_resolve_checkpoint_dir", lambda *a: pytest.fail("downloaded"))
    import huggingface_hub.constants

    (tmp_path / "hub").mkdir()  # the cache and the output on one disk
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    if refused:
        with pytest.raises(PorosDiskError, match="downloads 15.0 GB"):
            ex.room_before_download("Org/Model", None, tmp_path / "out")
    else:
        ex.room_before_download("Org/Model", None, tmp_path / "out")
