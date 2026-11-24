"""Merged export of an Inkling 2 Mini run (poros.moe.inkling2.export), in fp32 on the CPU.

A tiny released-format checkpoint (NVFP4 and MXFP8 experts) is loaded as a run loads it (its
released experts kept), its expert and attention adapters drawn, and merged: the output keeps the
release's names and layout with unquantized experts, gate / up rows interleaved again, no
quantization config; decoding it gives a model whose logits are the trained model's, and every
tensor the run did not adapt is the base's, byte for byte.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()

from safetensors import safe_open  # noqa: E402

from poros.moe.inkling2 import codec  # noqa: E402
from poros.moe.inkling2.config import text_config_of  # noqa: E402
from poros.moe.inkling2.export import export_merged  # noqa: E402
from poros.moe.inkling2.modeling import Inkling2LoopForCausalLM  # noqa: E402
from poros.moe.load import load_moe_streamed  # noqa: E402
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402
from tests.moe import inkling2_tiny as tiny  # noqa: E402


def _read(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(path / shard), "pt") as f:
            out.update({n: f.get_tensor(n) for n in f.keys()})
    return out


def test_the_merged_model_is_the_trained_one(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32)
    model = load_moe_streamed(str(base), torch.float32)
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    tiny.randomize_lora_b(model)
    adapter = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert any("q_proj.linear.lora_A" in n for n in adapter)
    rank = next(m.rank for m in model.modules() if hasattr(m, "lora_A_gate_up"))
    alpha = next(m.alpha for m in model.modules() if hasattr(m, "lora_A_gate_up"))

    out = tmp_path / "merged"
    export_merged(
        str(base),
        adapter,
        str(out),
        scaling=alpha / rank,
        dtype=torch.float32,
        expert_codec="released",
    )

    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "inkling2_loop" and "quantization_config" not in config
    merged, before = _read(out), _read(base)
    assert not [n for n in merged if n.endswith(("_scale", "_amax", "_denominator"))]
    adapted = ("wq_du", "wk_dv", "wv_dv", "wo_ud", "w13_weight", "w2_weight")
    for name, tensor in merged.items():
        if not any(a in name for a in adapted):
            assert torch.equal(tensor, before[name]), name  # untouched, as it was
    assert merged["model.llm.layers.0.mlp.experts.w13_weight"].dtype == torch.float32

    reloaded = Inkling2LoopForCausalLM(text_config_of(config))
    reloaded.load_state_dict(
        codec.convert_state_dict(
            merged.__getitem__,
            list(merged),
            unpadded_vocab_size=reloaded.config.unpadded_vocab_size,
            dtype=torch.float32,
            expert_dtype=torch.float32,
        )
    )
    model.eval()
    ids = tiny.ids()
    with torch.no_grad():
        want = model(input_ids=ids).logits
        got = reloaded(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def _trained(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32)
    from safetensors.torch import save_file

    save_file(
        {"mtp.layers.0.w": torch.arange(64, dtype=torch.float32)}, str(base / "mtp.safetensors")
    )
    model = load_moe_streamed(str(base), torch.float32)
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    tiny.randomize_lora_b(model)
    adapter = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    return base, adapter


def test_the_disk_check_counts_what_the_merge_writes(tmp_path, monkeypatch):
    """The disk a merge needs is computed from the tensors it writes (shapes and dtypes: a
    released expert grows NVFP4 to fp32) and the MTP file copied beside: what it then writes is
    within that; a disk with less free is refused before anything is written, in words."""
    import shutil
    from collections import namedtuple

    from poros.core.errors import PorosDiskError

    base, adapter = _trained(tmp_path)
    out = tmp_path / "merged"
    export_merged(
        str(base), adapter, str(out), scaling=2.0, dtype=torch.float32, expert_codec="released"
    )
    written = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    data = written - sum(
        8 + int.from_bytes(f.read_bytes()[:8], "little") for f in out.glob("*.safetensors")
    )
    released = sum(f.stat().st_size for f in base.glob("model-*.safetensors"))
    assert data > released  # the experts grew from their released bytes
    Usage = namedtuple("Usage", "total used free")  # one byte short of what it wrote: refused
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10**12, 0, data - 1))
    with pytest.raises(PorosDiskError, match="A merged model needs"):
        export_merged(
            str(base),
            adapter,
            str(tmp_path / "again"),
            scaling=2.0,
            dtype=torch.float32,
            expert_codec="released",
        )
    assert not (tmp_path / "again").exists()


def test_the_mtp_file_is_copied_a_chunk_at_a_time(tmp_path, monkeypatch):
    import pathlib

    base, adapter = _trained(tmp_path)
    real = pathlib.Path.read_bytes

    def guarded(self):
        assert self.name != "mtp.safetensors", "the MTP file was read whole"
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", guarded)
    out = tmp_path / "merged"
    export_merged(
        str(base), adapter, str(out), scaling=2.0, dtype=torch.float32, expert_codec="released"
    )
    monkeypatch.setattr(pathlib.Path, "read_bytes", real)
    assert (out / "mtp.safetensors").read_bytes() == (base / "mtp.safetensors").read_bytes()
