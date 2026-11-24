"""poros.stores: expert stores provided by installed packages (entry points)."""

from __future__ import annotations

import types

import pytest

from poros import stores
from poros.stores import PorosStoreNotFoundError, StorePlugin


class _EP:
    def __init__(self, name, obj, dist="poros-store-fake", fail=None):
        self.name, self.value = name, f"fake_pkg.plugin:{name}"
        self.dist = types.SimpleNamespace(name=dist)
        self._obj, self._fail = obj, fail

    def load(self):
        if self._fail:
            raise ImportError(self._fail)
        return self._obj


@pytest.fixture
def registry(monkeypatch):
    eps = {}
    monkeypatch.setattr(stores, "_entry_points", lambda: dict(eps))
    return eps


def test_builtins_and_missing_store(registry):
    assert stores.available_stores() == {"ram": "poros-train (built in)",
                                         "ssd": "poros-train (built in)"}
    with pytest.raises(PorosStoreNotFoundError, match="no installed package registers it"):
        stores.get_store_plugin("fake")
    with pytest.raises(PorosStoreNotFoundError, match="No store plugins are installed"):
        stores.check_store_name("fake")


def test_registered_plugin_loads_and_errors_name_the_package(registry):
    plugin = StorePlugin(name="fake", load_model=lambda *a, **k: None)
    registry["fake"] = _EP("fake", plugin)
    assert stores.get_store_plugin("fake") is plugin
    assert stores.available_stores()["fake"] == "poros-store-fake"
    registry["broken"] = _EP("broken", None, dist="poros-store-broken", fail="no module x")
    with pytest.raises(PorosStoreNotFoundError, match="'poros-store-broken'.*no module x"):
        stores.get_store_plugin("broken")
    registry["wrong"] = _EP("wrong", object())
    with pytest.raises(TypeError, match="StorePlugin"):
        stores.get_store_plugin("wrong")
    with pytest.raises(PorosStoreNotFoundError, match="Installed store plugins: broken, fake"):
        stores.get_store_plugin("other")


def test_config_and_cli_accept_a_registered_store(registry):
    from poros.core.config import PorosConfig, parse_store

    registry["fake"] = _EP("fake", StorePlugin(name="fake", load_model=lambda *a, **k: None))
    cfg = PorosConfig(store="fake", store_path="/packs/x", store_options={"mode": "a"})
    assert (cfg.store, cfg.store_path, cfg.store_options) == ("fake", "/packs/x", {"mode": "a"})
    assert parse_store("fake:/packs/x") == ("fake", "/packs/x")
    with pytest.raises(ValueError, match="no installed package"):
        PorosConfig(store="other")
    with pytest.raises(ValueError, match="Install 'other-store-pkg', which provides it"):
        PorosConfig(store="other", store_package="other-store-pkg")


def test_load_and_prepare_use_the_plugin(registry):
    torch = pytest.importorskip("torch")
    from tests.moe import require_tested_transformers

    require_tested_transformers()
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from poros.moe.load import load_moe_streamed
    from poros.moe.modules import ManagedMoeBlock
    from poros.moe.prepare import MoeConfig, default_backends, prepare_moe, streaming_config
    from poros.moe.store import MoeExpertStore

    seen = {}

    def load_model(name, dtype, **kw):
        seen.update(kw, name=name, dtype=dtype)
        torch.manual_seed(0)
        cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                             moe_intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                             num_experts=4, num_experts_per_tok=2)
        model = Qwen3MoeForCausalLM(cfg).to(dtype)
        state = {n: p.detach() for n, p in model.named_parameters()}
        model._poros_moe_store = MoeExpertStore.from_state_dict(
            state, num_layers=2, experts_per_layer=4, top_k=2, compute_dtype=dtype)
        return model

    def backends(store):
        seen["backends_for"] = store
        return default_backends()

    registry["fake"] = _EP("fake", StorePlugin(name="fake", load_model=load_model,
                                               backends=backends))
    model = load_moe_streamed("org/m", torch.bfloat16, store="fake", store_path="/p",
                              store_options={"k": 1})
    assert seen["name"] == "org/m" and seen["store_path"] == "/p" and seen["options"] == {"k": 1}
    assert seen["backends_for"] is model._poros_moe_store
    store = model._poros_moe_store
    cfg = streaming_config(model.config, store=store, device="cpu")
    assert cfg.capacity_bytes == 4 * store.slice_nbytes((0, 0))
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    assert sum(isinstance(m, ManagedMoeBlock) for m in model.modules()) == 2
