"""PorosTrainer's GPU fit at the length it trains at: with no
max_seq_length set, the load plans its path and tables but asks the fit at train(), at seq_len;
the AGENTS.md example (32B, block 4, seq_len 512 on a 24 GB card) is not warned about 2048."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from poros.core.card import detect_gpu  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosVramError  # noqa: E402
from poros.plan.configs import read_model_config  # noqa: E402
from poros.preflight import fit  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

MODEL = "Qwen/Qwen2.5-32B"


@pytest.fixture
def on_a_4090(monkeypatch):
    if read_model_config(MODEL, None) is None:
        pytest.skip("no packaged config.json sizes")

    def here(model_type, model_config, *, entry=None):
        return fit.run_planner(model_type, gpu=detect_gpu("RTX 4090"), host_free_gib=None,
                               model_config=model_config, entry=entry)

    monkeypatch.setattr(fit, "planner_here", here)


def _stub(seq, deferred=True):
    cfg = PorosConfig(model_name_or_path=MODEL, quantization="nf4", lora_rank=16, lora_alpha=32,
                      block_size=4, path="efficient", offload_embeddings_and_head=False)
    t = SimpleNamespace(config=cfg, _fit_check=True, _fit_deferred=deferred, _planned_seq=2048,
                        _metal=False, device=SimpleNamespace(type="cuda"), _fit_entry="dense")
    cfg.max_seq_length = seq
    return t


def test_the_agents_example_at_512_is_not_warned(on_a_4090, caplog):
    t = _stub(512)
    with caplog.at_level(logging.WARNING, logger="poros.train.trainer"):
        PorosTrainer._fit_at_length(t)
    assert not [r for r in caplog.records if "may not fit" in r.getMessage()]
    assert t._planned_seq == 512 and not t._fit_deferred
    assert (t.config.path, t.config.offload_embeddings_and_head) == ("efficient", False)


def test_a_length_the_card_may_not_hold_is_said_at_train(on_a_4090, caplog):
    t = _stub(2048)
    with caplog.at_level(logging.WARNING, logger="poros.train.trainer"):
        PorosTrainer._fit_at_length(t)
    assert [r for r in caplog.records if "may not fit" in r.getMessage()]


def test_the_planned_length_is_not_asked_twice(on_a_4090, monkeypatch):
    t = _stub(2048, deferred=False)
    monkeypatch.setattr(fit, "check", lambda *a, **k: pytest.fail("asked again"))
    PorosTrainer._fit_at_length(t)  # the load already asked at 2048


def test_a_refusal_at_the_trained_length_raises_before_the_data(on_a_4090, monkeypatch):
    t = _stub(512)
    err = PorosVramError(MODEL, "RTX 4090", "allocated peak 30 GiB > 23.99 GiB", "")
    monkeypatch.setattr(fit, "check", lambda *a, **k: fit.Fit(
        plan=None, p=None, path=None, offload=None, warning=None, vram=err, host=None))
    with pytest.raises(PorosVramError):
        PorosTrainer._fit_at_length(t)


def _decided(model, seq):
    """A config as a load at ``seq`` leaves it: CUDA defaults, then the fit's own choices."""
    mc = read_model_config(model, None)
    cfg = PorosConfig(model_name_or_path=model, quantization="nf4", lora_rank=16, lora_alpha=32)
    cfg.max_seq_length = seq
    cfg.apply_cuda_defaults(mc)
    fit.check(cfg, fit.planner_here("qwen2", mc, entry="dense"))
    return cfg


@pytest.mark.parametrize("seq", [2560, 4096])
def test_a_longer_length_at_train_picks_the_path_a_load_at_it_picks(on_a_4090, seq):
    """The check at train(seq_len=) gives its verdict on the path the run then trains on: a path
    left to Poros is the one a load at that length picks, not the load's earlier one."""
    if read_model_config("Qwen/Qwen2.5-7B", None) is None:
        pytest.skip("no packaged config.json sizes")
    loaded = _decided("Qwen/Qwen2.5-7B", 2048)
    t = SimpleNamespace(config=loaded, _fit_check=True, _fit_deferred=False, _planned_seq=2048,
                        _metal=False, device=SimpleNamespace(type="cuda"), _fit_entry="dense",
                        _fit_host=True)
    tables = loaded.offload_embeddings_and_head
    loaded.max_seq_length = seq
    PorosTrainer._fit_at_length(t)
    assert loaded.path == _decided("Qwen/Qwen2.5-7B", seq).path
    assert loaded.offload_embeddings_and_head == tables  # placed at the load: stays


def test_host_ram_is_asked_at_the_trained_length_on_the_loads_flag(on_a_4090, monkeypatch):
    from poros.core.errors import PorosHostRamError

    err = PorosHostRamError("Qwen/Qwen2.5-32B", 90.0, 40.0, "peak host RAM 90 GiB")
    monkeypatch.setattr(fit, "check", lambda *a, **k: fit.Fit(
        plan=None, p=None, path=None, offload=None, warning=None, vram=None, host=err))
    t = _stub(512)
    t._fit_host = True
    with pytest.raises(PorosHostRamError):
        PorosTrainer._fit_at_length(t)
    t = _stub(512)
    t._fit_host = False  # from_pretrained(check_host_ram=False)
    PorosTrainer._fit_at_length(t)
