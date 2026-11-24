"""A Mac table row for a model the catalog does not list, derived from its config: a Hub id and a
folder alike, only for a family someone asserted, never said to be measured. Configs only, no
weights, no network."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from poros import metal  # noqa: E402
from poros.cli.verbs import arch  # noqa: E402
from poros.cli.words.models import catalog  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosGuaranteeUnavailableError  # noqa: E402
from poros.metal import layout, store, table  # noqa: E402

CLASS = "apple-m4-10gpu-16gb"
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REAL = Path(__file__).parent / "real_configs"
# SmolLM2-135M's config: a llama, which the registry does not name
LLAMA = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 576,
         "intermediate_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 9,
         "num_key_value_heads": 3, "vocab_size": 49152, "max_position_embeddings": 8192,
         "rms_norm_eps": 1e-5, "rope_theta": 100000, "tie_word_embeddings": True}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """An empty Hub cache, and a Hub that answers with whatever config a test hands it."""
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("POROS_USER_ASSERTED_ARCHS", raising=False)
    said = {}
    monkeypatch.setattr(arch, "_from_hub", lambda model: said.get(model, arch._Seen()))
    return said


def _folder(where: Path, config: dict, gb: float = 0.27) -> str:
    where.mkdir()
    (where / "config.json").write_text(json.dumps(config))
    with open(where / "model.safetensors", "wb") as f:
        f.truncate(int(gb * 1e9))
    return str(where)


def test_a_hub_id_of_an_asserted_family_has_a_row_derived_from_its_config(hub, monkeypatch):
    hub[SMOL] = arch._Seen(raw=dict(LLAMA))
    assert table.pick(SMOL, CLASS) is None  # no one asserted it: no row, as before
    with pytest.raises(PorosGuaranteeUnavailableError, match="does not train on a Mac"):
        table.require_row(SMOL, CLASS)
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "llama")
    table.require_row(SMOL, CLASS)
    row = table.pick(SMOL, CLASS)
    assert row.derived and row.evidence == "none" and not (row.receipt or row.macos_build)
    assert table.evidence(SMOL, CLASS) == "none"
    # its lengths and tier are the nearest model's of its kind, the smallest whose pack holds it
    near = table.pick("Qwen/Qwen2.5-0.5B", CLASS)
    assert (row.seq, row.tier, row.grad_accum, row.save_steps, row.synthetic_steps) == (
        near.seq, near.tier, near.grad_accum, near.save_steps, near.synthetic_steps)
    assert [r.seq for r in table.rows(SMOL, CLASS)] == [256, 512]
    # its sizes are arithmetic on its own config, not the nearest model's
    found = layout.layout_of(LLAMA)
    assert table.pack_gb(SMOL) == pytest.approx(found.pack_bytes / 1e9)
    assert table.disk_gb(SMOL) == pytest.approx((found.pack_bytes + found.weights_bytes) / 1e9)
    from_ram = (found.pack_bytes - found.one_layer_bytes) / 1e9  # every layer held, in RAM
    assert table.floor_gb(SMOL, 256) == pytest.approx(found.floor_gb(256, 16) + from_ram)
    assert table.floor_gb(SMOL, 256) < table.floor_gb("Qwen/Qwen2.5-0.5B", 256)


def test_a_folder_and_a_hub_id_of_the_same_model_have_one_row(hub, tmp_path, monkeypatch):
    hub[SMOL] = arch._Seen(raw=dict(LLAMA))
    here = _folder(tmp_path / "smol", LLAMA)
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "llama")
    assert table.rows(here, CLASS) == table.rows(SMOL, CLASS) != ()
    assert table.floor_gb(here, 512) == table.floor_gb(SMOL, 512)
    assert table.pack_gb(here) == table.pack_gb(SMOL)
    assert table.disk_gb(here) == table.pack_gb(here)  # a folder's shards are where they are
    assert table._entry(here).entry == table._entry(SMOL).entry == "dense"


def test_a_hub_id_reads_its_config_from_the_hub_cache_first(hub, tmp_path, monkeypatch):
    """What `arch` fetched once is read again from the cache: no second request."""
    snapshot = tmp_path / "hub" / "models--HuggingFaceTB--SmolLM2-135M-Instruct"
    (snapshot / "snapshots" / "abc").mkdir(parents=True)
    (snapshot / "refs").mkdir()
    (snapshot / "refs" / "main").write_text("abc")
    (snapshot / "snapshots" / "abc" / "config.json").write_text(json.dumps(LLAMA))
    monkeypatch.setattr(arch, "_from_hub", lambda model: pytest.fail("asked the Hub again"))
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "llama")
    assert table.pick(SMOL, CLASS).derived


@pytest.mark.parametrize("raw, asserted", [
    ({"model_type": "qwen2"}, "qwen2"),  # a validated family the catalog does not list: as before
    ({"model_type": "evil_net", "auto_map": {"AutoModelForCausalLM": "evil.Net"}}, "evil_net"),
    ({"model_type": "not_a_type"}, "not_a_type"),
    ({**LLAMA, "num_local_experts": 8}, "llama"),  # names experts its layers do not have
    ({**LLAMA, "num_hidden_layers": 10**9}, "llama"),  # bounded before anything is built
])
def test_what_no_row_is_derived_for(hub, monkeypatch, raw, asserted):
    hub["someone/model"] = arch._Seen(raw=raw)
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", asserted)
    assert table.pick("someone/model", CLASS) is None


def test_a_gated_repo_has_no_row(hub, monkeypatch):
    hub["someone/gated"] = arch._Seen(gated=True)
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "llama")
    assert table.pick("someone/gated", CLASS) is None


def test_a_new_moe_family_reads_the_nearest_moe_row(hub, monkeypatch):
    hub["mistralai/Mixtral-8x7B-v0.1"] = arch._Seen(
        raw=json.loads((REAL / "mixtral_8x7b.json").read_text()))
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "mixtral")
    row = table.pick("mistralai/Mixtral-8x7B-v0.1", CLASS)
    assert row.derived and row.tier == "ssd" and row.evidence == "none"
    assert table._entry("mistralai/Mixtral-8x7B-v0.1").entry == "moe"
    found = layout.layout_of(json.loads((REAL / "mixtral_8x7b.json").read_text()))
    assert table.floor_gb("mistralai/Mixtral-8x7B-v0.1", 128) == pytest.approx(
        found.floor_gb(128, 16))  # from disk, a layer at a time


def test_settle_takes_a_hub_id_of_an_asserted_family_to_its_store(hub, monkeypatch):
    hub[SMOL] = arch._Seen(raw=dict(LLAMA))
    calls = []
    monkeypatch.setattr(store, "settle", calls.append)
    monkeypatch.setattr(metal, "device_class", lambda: CLASS)
    with pytest.raises(PorosGuaranteeUnavailableError):
        metal.settle(PorosConfig(model_name_or_path=SMOL))
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "llama")
    cfg = PorosConfig(model_name_or_path=SMOL)
    metal.settle(cfg)
    row = table.pick(SMOL, CLASS)
    assert calls == [cfg] and (cfg.max_seq_length, cfg.store, cfg.save_steps) == (
        row.seq, row.tier, row.save_steps)


def test_no_catalog_row_says_it_is_derived():
    assert not any(r.derived for e in catalog() for r in e.metal)


def test_a_hub_id_outside_the_catalog_downloads_with_no_token(monkeypatch, tmp_path):
    from poros.metal import load

    said = []
    monkeypatch.setattr(load, "snapshot_download", lambda name, **kw: said.append(kw) or str(
        tmp_path / "abc"))
    for model in (SMOL, "Qwen/Qwen2.5-0.5B"):
        load.fetch(PorosConfig(model_name_or_path=model), weights=True)
    assert said[0]["token"] is False  # anyone's repo, read as anyone reads it
    assert said[1].get("token") is None  # a catalog model is fetched as it always was


class _Prompted(BaseException):
    """transformers asked a person whether to run a repo's own code."""


def test_a_pack_never_asks_to_run_a_repos_own_code(tmp_path, monkeypatch):
    """A config can change between the reading and the download: the pack itself runs no repo code
    and asks no one whether it may."""
    from poros.metal import pack

    folder = tmp_path / "evil"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "evil_net", "auto_map": {"AutoConfig": "evil.EvilConfig",
                                                "AutoModelForCausalLM": "evil.EvilModel"}}))
    (folder / "evil.py").write_text("raise SystemExit('ran')\n")

    def asked(*_):
        raise _Prompted

    monkeypatch.setattr(builtins, "input", asked)
    with pytest.raises(ValueError):
        pack.pack(folder, tmp_path / "store", model_id="evil", revision="0" * 8)
