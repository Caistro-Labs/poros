"""``poros.metal.pack`` on a tiny checkpoint: every tensor checked, nothing torn, reused by meta."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
pytest.importorskip("peft")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from poros.metal import pack as pack_mod  # noqa: E402
from poros.metal.pack import EMBED_BLOCK, HEAD_BLOCK, pack, pack_dir  # noqa: E402
from poros.moe import quant  # noqa: E402
from poros.sss.format import SssFile  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKED_FILES = ("model.sss", "model.sss.idx", "small.safetensors")
EMBED, HEAD = "model.embed_tokens.weight", "lm_head.weight"
# Qwen2's, in the order the stores on disk hold them: a rebuilt one must be the same bytes.
LAYER_MATRICES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _pack(source, store, **kwargs):
    return pack(source, store, model_id=tiny.MODEL_ID, revision=tiny.REVISION, **kwargs)


@pytest.fixture(scope="module")
def untied(tmp_path_factory):
    cfg = tiny.tiny_config()
    source, path = tiny.packed(tmp_path_factory.mktemp("untied"), cfg)
    return SimpleNamespace(cfg=cfg, source=source, path=path)


@pytest.fixture(scope="module")
def tied(tmp_path_factory):
    cfg = tiny.tiny_config(tied=True)
    source, path = tiny.packed(tmp_path_factory.mktemp("tied"), cfg)
    return SimpleNamespace(cfg=cfg, source=source, path=path)


def _digest(path: Path) -> dict:
    return {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in PACKED_FILES}


def _slab(file: SssFile, key) -> torch.Tensor:
    raw = torch.empty(file.entry(key).nbytes, dtype=torch.uint8)
    file.read_into(key, raw, verify=True)
    return raw


def test_store_path_is_slug_and_short_revision():
    got = pack_dir("/s", "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987")
    assert got == Path("/s/Qwen--Qwen2.5-0.5B-060db649")
    assert pack_dir("/s", "../evil", "abc").parent == Path("/s")


def test_every_blob_requantizes_from_source_to_the_same_bytes(untied):
    source = load_file(untied.source / "model.safetensors")
    with SssFile.open(untied.path / "model.sss") as file:
        for layer in range(untied.cfg.num_hidden_layers):
            slab = _slab(file, (layer, 0))
            views = file.entry((layer, 0)).views
            assert [v["name"] for v in views] == list(LAYER_MATRICES)
            for view in views:
                weight = source[f"model.layers.{layer}.{view['name']}.weight"]
                assert view["offset"] % 64 == 0
                assert view["shape"] == list(weight.shape)
                blob = quant.quantize_tensor(weight)
                assert torch.equal(slab[view["offset"] : view["offset"] + blob.numel()], blob)


def test_raw_slabs_are_the_source_bytes(untied):
    source = load_file(untied.source / "model.safetensors")
    with SssFile.open(untied.path / "model.sss") as file:
        for key, name in (((EMBED_BLOCK, 0), EMBED), ((HEAD_BLOCK, 0), HEAD)):
            assert torch.equal(_slab(file, key), source[name].view(torch.uint8).reshape(-1))
            assert file.entry(key).views[0]["shape"] == list(source[name].shape)


def test_small_is_biases_and_norms_in_bf16(untied):
    small = load_file(untied.path / "small.safetensors")
    source = load_file(untied.source / "model.safetensors")
    layers = untied.cfg.num_hidden_layers
    assert len(small) == 1 + layers * (2 + 3)  # final norm; two norms and q, k, v bias per layer
    for name, tensor in small.items():
        assert tensor.dtype == torch.bfloat16, name
        assert torch.equal(tensor.view(torch.int16), source[name].view(torch.int16)), name
    packed_names = {f"model.layers.{i}.{m}.weight" for i in range(layers) for m in LAYER_MATRICES}
    assert set(source) == set(small) | packed_names | {EMBED, HEAD}


def test_a_tied_model_has_one_rows_slab(tied):
    with safe_open(tied.source / "model.safetensors", framework="pt") as f:
        assert HEAD not in f.keys()
    with SssFile.open(tied.path / "model.sss") as file:
        assert [e.key for e in file.entries if e.kind == "rows"] == [(EMBED_BLOCK, 0)]
        file.verify_all()


def test_a_second_pack_reuses_the_store(untied, monkeypatch):
    before = {n: (untied.path / n).stat().st_mtime_ns for n in PACKED_FILES}

    def refuse(weight):
        raise AssertionError("a store whose meta matches must not be rebuilt")

    monkeypatch.setattr(quant, "quantize_tensor", refuse)
    assert _pack(untied.source, untied.path.parent) == untied.path
    assert before == {n: (untied.path / n).stat().st_mtime_ns for n in PACKED_FILES}


def test_a_different_meta_rebuilds(tmp_path, monkeypatch):
    cfg = tiny.tiny_config(vocab_size=256, layers=1)
    source, path = tiny.packed(tmp_path, cfg)
    monkeypatch.setattr(pack_mod, "PACK_VERSION", pack_mod.PACK_VERSION + 1)
    assert _pack(source, path.parent) == path
    with SssFile.open(path / "model.sss") as file:
        assert file.meta["pack"] == pack_mod.PACK_VERSION
        assert file.meta["torch"] == torch.__version__


def test_pack_twice_is_the_same_bytes(tmp_path, untied):
    assert _digest(_pack(untied.source, tmp_path)) == _digest(untied.path)


def test_a_failed_pack_leaves_no_store(tmp_path, untied, monkeypatch):
    real, calls = quant.quantize_tensor, []

    def fail_at_fifth(weight):
        calls.append(1)
        if len(calls) == 5:
            raise OSError(28, "No space left on device")
        return real(weight)

    monkeypatch.setattr(quant, "quantize_tensor", fail_at_fifth)
    with pytest.raises(OSError, match="No space"):
        _pack(untied.source, tmp_path)
    assert list(pack_dir(tmp_path, tiny.MODEL_ID, tiny.REVISION).iterdir()) == []


@pytest.mark.parametrize("at", [1, 20], ids=lambda n: f"tensor{n}")
def test_kill_nine_leaves_no_store_that_looks_finished(tmp_path, untied, at):
    script = textwrap.dedent(f"""
        import os, signal, sys
        from poros.metal.pack import pack
        from poros.moe import quant
        real, calls = quant.quantize_tensor, []
        def die(weight):
            calls.append(1)
            if len(calls) == {at}:
                os.kill(os.getpid(), signal.SIGKILL)
            return real(weight)
        quant.quantize_tensor = die
        pack(sys.argv[1], sys.argv[2], model_id={tiny.MODEL_ID!r}, revision={tiny.REVISION!r})
    """)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    args = [sys.executable, "-c", script, str(untied.source), str(tmp_path)]
    done = subprocess.run(args, env=env)
    assert done.returncode == -signal.SIGKILL
    out = pack_dir(tmp_path, tiny.MODEL_ID, tiny.REVISION)
    assert not (out / "model.sss.idx").exists()
    assert _digest(_pack(untied.source, tmp_path)) == _digest(untied.path)


def test_a_sharded_source_packs_to_the_same_bytes(tmp_path, untied):
    source = tmp_path / "sharded"
    tiny.write_checkpoint(source, untied.cfg, max_shard_size="2MB")
    assert (source / "model.safetensors.index.json").exists()
    assert _digest(_pack(source, tmp_path / "store")) == _digest(untied.path)


@pytest.mark.parametrize(
    ("add", "drop", "error"),
    [
        ({"model.extra.weight": torch.zeros(4).to(torch.bfloat16)}, None, "no place for"),
        (None, "model.layers.2.mlp.up_proj.weight", "up_proj"),
        (None, EMBED, "embed_tokens"),
        (None, HEAD, "lm_head"),
    ],
    ids=["unknown_tensor", "missing_matrix", "no_embed", "head_missing"],
)
def test_refuses_a_checkpoint_it_cannot_place(tmp_path, untied, add, drop, error):
    source = tmp_path / "edited"
    source.mkdir()
    tensors = {k: v for k, v in load_file(untied.source / "model.safetensors").items() if k != drop}
    save_file({**tensors, **(add or {})}, source / "model.safetensors")
    (source / "config.json").write_text((untied.source / "config.json").read_text())
    with pytest.raises(ValueError, match=error):
        _pack(source, tmp_path / "store")
    assert not (tmp_path / "store").exists()
