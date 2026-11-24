"""The store's surfaces: config, ``poros plan --store``, ``PorosTrainer(store=...)``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poros.cli.schemas import PlanGpu
from poros.core.config import PorosConfig, parse_bytes, parse_store, ssd_cache_bytes
from poros.plan.make import make_plan
from poros.sss.probe import ProbeResult
from tests.conftest import needs_fadvise

QWEN3_30B = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
             "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
             "vocab_size": 151936, "tie_word_embeddings": False, "intermediate_size": 6144,
             "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
             "decoder_sparse_step": 1, "mlp_only_layers": []}
GPU = PlanGpu(name="NVIDIA GeForce RTX 4060 Laptop GPU", total_gib=8.0, free_gib=7.5,
              source="torch")


def test_config_and_flag_parsing():
    assert parse_bytes("4G") == 4 << 30 and parse_bytes("512m") == 512 << 20
    assert parse_bytes("1.5GiB") == 3 << 29 and parse_bytes("4096") == 4096
    with pytest.raises(ValueError):
        parse_bytes("lots")
    assert parse_store("ram") == ("ram", None) and parse_store("ssd:/n/p") == ("ssd", "/n/p")
    assert parse_store("ssd") == ("ssd", None)
    with pytest.raises(ValueError):
        parse_store("NVMe!:/x")
    with pytest.raises(ValueError):
        parse_store("ram:/x")
    # any other lowercase name is a plugin store (poros.stores); the config refuses an
    # uninstalled one and says how to get it
    assert parse_store("nvme:/x") == ("nvme", "/x")
    with pytest.raises(ValueError, match="no installed package registers it"):
        PorosConfig(store="nvme")
    cfg = PorosConfig(store="ssd", store_path="/nvme", sss_ram="1G")
    assert cfg.sss_ram == 1 << 30
    assert ssd_cache_bytes(cfg.store_options) == 0
    cached = PorosConfig(store="ssd", store_path="/nvme", store_options={"cache": "2G"})
    assert ssd_cache_bytes(cached.store_options) == 2 << 30
    with pytest.raises(ValueError, match="reads store_options"):
        PorosConfig(store="ssd", store_path="/nvme", store_options={"cash": "2G"})
    assert PorosConfig().store == "ram"
    with pytest.raises(ValueError, match="store_path"):
        PorosConfig(store="ssd")


def _plan(store, probe=None, host_free=14.0, **kw):
    cfg = PorosConfig(model_name_or_path="org/qwen3-moe-unmeasured", dataset="synthetic",
                      max_seq_length=777, max_steps=10,
                      **{"gradient_accumulation_steps": 1, **kw},
                      **({"store": "ssd", "store_path": "/nvme", "sss_ram": "2G"}
                         if store == "ssd" else {}))
    return make_plan(cfg, model_type="qwen3_moe", support="validated", entry="moe", gpu=GPU,
                     model_config=QWEN3_30B, host_free_gib=host_free, probe=probe)


def test_plan_ssd_store_drops_the_experts_from_host_ram():
    ram = _plan("ram")
    assert ram.store == "ram" and not ram.stores and ram.host_fits is False  # 22.9 GiB
    probe = ProbeResult(directory="/nvme", b_ssd=2e9, direct=True, b_h2d=12e9)
    ssd = _plan("ssd", probe)
    assert ssd.store == "ssd" and [r.store for r in ssd.stores] == ["ram", "ssd"]
    assert ssd.stores[0].peak_host_ram.value == ram.peak_host_ram.value
    # 30B-A3B: ~14 GiB of NF4 experts leave RAM, a ring of 264 2.4 MB slabs arrives
    assert ssd.peak_host_ram.value == pytest.approx(ram.peak_host_ram.value - 13.9 + 0.6,
                                                    abs=0.15)
    assert ssd.host_fits is True
    # no RAM receipt for this config: the ssd row is the disk floor,
    # (2 - 1/48) x 14.9 GB of experts / 2 GB/s
    row = ssd.stores[1]
    assert row.bound_by == "disk" and row.b_ssd == 2e9 and row.direct
    assert ssd.s_per_step.source == "estimated" and 14.5 < ssd.s_per_step.value < 15
    unprobed = _plan("ssd")
    assert unprobed.s_per_step.source == "unknown" and unprobed.stores[1].b_ssd is None


def test_a_moe_run_short_of_host_ram_hears_of_the_ssd_store_first():
    """The RAM store's 22.9 GiB does not fit 14 GiB; the SSD store's ~9.6 GiB does: the refusal's
    remedy names it first, without the plan's figure. Where the SSD store would not fit either,
    it says nothing of it; an SSD run is never told to use the SSD store."""
    from poros.core.errors import PorosHostRamError
    from poros.preflight import fit

    def planner(host_free):
        return lambda cfg: make_plan(cfg, model_type="qwen3_moe", support="validated",
                                     entry="moe", gpu=GPU, model_config=QWEN3_30B,
                                     host_free_gib=host_free)

    cfg = PorosConfig(model_name_or_path="org/qwen3-moe-unmeasured", dataset="synthetic",
                      max_seq_length=777, gradient_accumulation_steps=1, max_steps=10)
    got = fit.check(cfg.model_copy(), planner(14.0))
    assert isinstance(got.host, PorosHostRamError)
    ssd = _plan("ssd").peak_host_ram.value
    assert got.host.info.remediation.startswith(
        "store: ssd keeps the experts in a .sss file on a local SSD instead, which the check does "
        "not refuse for the host RAM here")
    assert str(ssd) not in got.host.info.remediation and "GiB" not in got.host.info.remediation
    tight = fit.check(cfg.model_copy(), planner(ssd - 1))  # neither fits: no SSD advice
    assert isinstance(tight.host, PorosHostRamError)
    assert tight.host.info.remediation.startswith("Run on a machine")
    on_ssd = cfg.model_copy(update={"store": "ssd", "store_path": "/nvme"})
    assert fit.ssd_alternative(on_ssd, planner(1.0), planner(1.0)(on_ssd)) is None


def test_plan_ssd_store_counts_its_host_cache():
    probe = ProbeResult(directory="/nvme", b_ssd=2e9, direct=True, b_h2d=12e9)
    none, two = _plan("ssd", probe), _plan("ssd", probe, store_options={"cache": "2G"})
    assert two.peak_host_ram.value == pytest.approx(none.peak_host_ram.value + 2.0, abs=0.11)
    assert "host cache (sss_cache)" in two.peak_host_ram.method
    # never more than the experts it could hold: ~14 GiB of NF4 experts at 30B-A3B
    huge = _plan("ssd", probe, store_options={"cache": "64G"}, host_free=100.0)
    assert huge.peak_host_ram.value < none.peak_host_ram.value + 14.5


@pytest.mark.parametrize("path, reads", [("sequential", 8), ("efficient", 1)])
def test_plan_ssd_reads_the_experts_once_an_update_block_major(path, reads):
    """Eight micro-batches an update: the sequential loop streams every layer's experts for each,
    a block-major path once for all (a layer runs every micro-batch while they are on the GPU)."""
    probe = ProbeResult(directory="/nvme", b_ssd=2e9, direct=True, b_h2d=12e9)
    one = _plan("ssd", probe, path="sequential").s_per_step.value
    plan = _plan("ssd", probe, path=path, gradient_accumulation_steps=8)
    assert plan.s_per_step.value == pytest.approx(reads * one, rel=0.01)
    assert ("block-major" in plan.s_per_step.method) == (reads == 1)


def test_train_cli_store_flag_refuses_a_store_nothing_provides(tmp_path):
    """--store with a name no built-in or installed plugin provides is refused as a config error
    before anything loads, naming the store. (This drove the removed poros plan --store; poros
    train's own --store takes the same parse and the same check.)"""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model_name_or_path: Qwen/Qwen2.5-0.5B\ndataset: synthetic\n")
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", "from poros.cli.main import app; app()", "train", str(cfg),
         "--store", "nvme:/x", "--json"],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": "1",
             "POROS_HOME": str(tmp_path / "home")},
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    error = json.loads(proc.stdout)["error"]
    assert error["code"] == "E_CONFIG" and error["exit_code"] == 2
    assert "store 'nvme' is not built in" in error["message"]


def test_trainer_refuses_ssd_for_a_dense_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from poros.core.errors import PorosConfigError
    from poros.train.trainer import PorosTrainer

    monkeypatch.setattr("poros.preflight.arch.cheap_model_type", lambda *a: "qwen2")
    with pytest.raises(PorosConfigError, match="not a MoE model"):
        PorosTrainer.from_pretrained("org/dense", store="ssd", store_path=str(tmp_path),
                                     output_dir=str(tmp_path / "o"))


@needs_fadvise
def test_trainer_ssd_store_trains_like_ram(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tests.moe import require_tested_transformers

    require_tested_transformers()
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from poros.train.trainer import PorosTrainer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("poros.sss.probe.probe_disk",
                        lambda target, **kw: ProbeResult(directory=str(target), b_ssd=1e9,
                                                         direct=False))
    torch.manual_seed(0)
    config = Qwen3MoeConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                            moe_intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            num_experts=8, num_experts_per_tok=4)
    ckpt = tmp_path / "ckpt"
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(ckpt)
    ids = torch.randint(0, 64, (6, 16), generator=torch.Generator().manual_seed(1))
    rows = [{"input_ids": x, "labels": x.clone()} for x in ids]
    losses, lora = {}, {}
    for arm in ("ram", "ssd", "ssd_cache"):
        store = arm.split("_")[0]
        trainer = PorosTrainer.from_pretrained(
            str(ckpt), output_dir=str(tmp_path / arm), max_steps=3, save_steps=0,
            progress=False, gradient_accumulation_steps=2, store=store,
            path="sequential",  # one mark per micro-batch: the sequential loop's
            store_path=str(tmp_path / "store") if store == "ssd" else None,
            **({"store_options": {"cache": "1M"}} if arm == "ssd_cache" else {}),
        )
        trainer.train(torch.utils.data.DataLoader(rows, batch_size=1))
        losses[arm] = list(trainer.history.losses)
        lora[arm] = {n: p.detach().clone() for n, p in trainer.model.named_parameters()
                     if p.requires_grad}
        if store == "ssd":
            st = trainer._sss_source.stats
            assert st["steps"] == 6  # one mark per micro-batch
            assert list((tmp_path / "store").glob("*.moe.sss"))
            # the host cache serves the re-reads it still holds; without it, none
            assert st["cache"] is (arm == "ssd_cache") and (st["cache_hits"] > 0) is st["cache"]
            steps = [json.loads(line) for line in open(tmp_path / arm / "events.jsonl")]
            last = [e for e in steps if e["event_type"] == "step"][-1]["data"]["sss"]
            assert last["cache_hits"] == st["cache_hits"] and last["reads"] == st["reads"]
    for arm in ("ssd", "ssd_cache"):
        assert losses["ram"] == losses[arm]
        assert lora["ram"].keys() == lora[arm].keys()
        assert all(torch.equal(lora["ram"][n], lora[arm][n]) for n in lora["ram"])


@needs_fadvise
def test_a_nonfinite_update_stops_an_ssd_run_before_it_is_applied(tmp_path, monkeypatch):
    """The non-finite stop holds on the SSD store as on any: update 2's loss goes NaN, the run
    stops there with E_STEP_CHECK, the adapter is update 1's, and the source still closes."""
    torch = pytest.importorskip("torch")
    from tests.moe import require_tested_transformers

    require_tested_transformers()
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from poros.core.errors import PorosStepCheckError
    from poros.train.trainer import PorosTrainer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("poros.sss.probe.probe_disk",
                        lambda target, **kw: ProbeResult(directory=str(target), b_ssd=1e9,
                                                         direct=False))
    torch.manual_seed(0)
    config = Qwen3MoeConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                            moe_intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            num_experts=8, num_experts_per_tok=4)
    ckpt = tmp_path / "ckpt"
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(ckpt)
    ids = torch.randint(0, 64, (6, 16), generator=torch.Generator().manual_seed(1))
    rows = [{"input_ids": x, "labels": x.clone()} for x in ids]
    trainer = PorosTrainer.from_pretrained(
        str(ckpt), output_dir=str(tmp_path / "o"), max_steps=3, save_steps=0, progress=False,
        gradient_accumulation_steps=1, store="ssd", store_path=str(tmp_path / "store"),
        path="sequential")
    seen, step, real = {}, trainer._adapter_optimizer_step, trainer._sequential_pass

    def watched():
        step()
        seen["after1"] = {n: p.detach().clone() for n, p in trainer.model.named_parameters()
                          if p.requires_grad}

    def poisoned(batch, g, G):
        loss = real(batch, g, G)
        return loss * float("nan") if trainer.global_step == 1 else loss

    trainer._adapter_optimizer_step, trainer._sequential_pass = watched, poisoned
    with pytest.raises(PorosStepCheckError):
        trainer.train(torch.utils.data.DataLoader(rows, batch_size=1))
    now = {n: p.detach() for n, p in trainer.model.named_parameters() if p.requires_grad}
    assert all(torch.equal(now[n], seen["after1"][n]) for n in now)
    events = [json.loads(line) for line in open(tmp_path / "o" / "events.jsonl")]
    assert [e["step"] for e in events if e["event_type"] == "error"] == [1]  # index, from 0
    trainer._sss_source.close()


@needs_fadvise
def test_a_run_on_the_ssd_store_drives_its_schedule(tmp_path, monkeypatch):
    """``poros.Run`` (the recipes' engine) on the SSD store's sequential loop: each pass's reads
    become the next pass's schedule, as the classic loop's do (block-major drives its own, one an
    update), and the first one writes the store's setup record."""
    torch = pytest.importorskip("torch")
    from tests.moe import require_tested_transformers

    require_tested_transformers()
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from poros.train.batch import Batch, Datum
    from poros.train.run import Run
    from poros.train.trainer import PorosTrainer
    from tests.conftest import Tok

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("poros.sss.probe.probe_disk",
                        lambda target, **kw: ProbeResult(directory=str(target), b_ssd=1e9,
                                                         direct=False))
    torch.manual_seed(0)
    config = Qwen3MoeConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                            moe_intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            num_experts=8, num_experts_per_tok=4)
    ckpt = tmp_path / "ckpt"
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(ckpt)
    trainer = PorosTrainer.from_pretrained(
        str(ckpt), output_dir=str(tmp_path / "o"), max_steps=2, save_steps=0, progress=False,
        per_device_batch_size=1, store="ssd", store_path=str(tmp_path / "store"),
        path="sequential")
    run = Run.from_trainer(trainer, Tok())
    batch = Batch([Datum.from_sequence(list(range(3 + i, 19 + i))) for i in range(2)])
    for _ in range(2):  # two updates of two micro-batches
        run.forward_backward(batch)
        run.optim_step()
    st = trainer._sss_source.stats
    assert not st["recording"] and st["steps"] == 4  # a pass a micro-batch
    events = [json.loads(line) for line in open(tmp_path / "o" / "events.jsonl")]
    assert [e for e in events if e["event_type"] == "setup" and "sss" in (e.get("data") or {})]
    trainer._sss_source.close()


def test_plan_ssd_reads_less_through_its_caches():
    """The host cache serves twice its size a pass pair from RAM; path fast's expert cache on the
    GPU is read by neither pass. Unsized (no cap, no receipt of this config on this GPU), fast's
    speed is unknown, never a guess off by the cache; a cap sizes it."""
    probe = ProbeResult(directory="/nvme", b_ssd=2e9, direct=True, b_h2d=12e9)
    nf4 = 14.95e9  # 30B-A3B's experts in NF4, about
    plain = _plan("ssd", probe, path="efficient").s_per_step
    cached = _plan("ssd", probe, path="efficient", store_options={"cache": "2G"}).s_per_step
    assert cached.value == pytest.approx(plain.value - 2 * (2 << 30) / 2e9, abs=0.1)
    assert "less twice the 2.00 GiB host cache" in cached.method
    fast = _plan("ssd", probe, path="fast").s_per_step
    assert fast.value is None and "grows its expert cache" in fast.why
    capped = _plan("ssd", probe, path="fast", moe_expert_cache_gib=4.0).s_per_step
    assert capped.value == pytest.approx(2 * (nf4 - (4 << 30)) / 2e9, rel=0.03)
    both = _plan("ssd", probe, path="fast", moe_expert_cache_gib=4.0,
                 store_options={"cache": "8G"}).s_per_step
    assert both.value == pytest.approx(2 * (nf4 - (12 << 30)) / 2e9, rel=0.06)
    every = _plan("ssd", probe, path="fast", moe_expert_cache_gib=8.0,
                  store_options={"cache": "8G"}, host_free=40.0).s_per_step
    assert every.value == 0.0  # no RAM-store receipt here: the disk floor, nothing read


def test_plan_ssd_fast_cache_from_a_receipt_on_this_card():
    """Qwen3-30B-A3B on an RTX 4090 at 8 x 512: the fast receipt grew a 10.43 GiB cache within
    22.95 GiB; a plan reading 23.5 GiB free sizes 0.8 of the extra on top, and the s/step adds
    the RAM store's to what the two caches leave on the disk."""
    from tests.sss.test_store_surface import QWEN3_30B as config

    probe = ProbeResult(directory="/nvme", b_ssd=3.2e9, direct=True, b_h2d=None)
    card = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=23.5, source="spec")

    def plan(**kw):
        cfg = PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic",
                          max_seq_length=512, lora_dropout=0.0, per_device_batch_size=8,
                          gradient_accumulation_steps=1, max_steps=10, store="ssd",
                          store_path="/nvme", **kw)
        return make_plan(cfg, model_type="qwen3_moe", support="validated", entry="moe",
                         gpu=card, model_config=config, host_free_gib=60.0, probe=probe)

    from poros.plan.make import _fast_cache_gib
    from poros.plan.receipts import Request

    fast = plan(path="fast")
    req = Request.of(PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic",
                                 max_seq_length=512, lora_dropout=0.0, per_device_batch_size=8,
                                 gradient_accumulation_steps=1, path="fast"), "moe", triton=True)
    assert _fast_cache_gib(req, card, None) == pytest.approx(10.427 + 0.8 * (23.5 - 22.95))
    assert "less the 10.87 GiB expert cache on the GPU" in fast.s_per_step.method
    assert fast.s_per_step.value == pytest.approx(1.423 + 2 * (14.95e9 - 10.87 * 2**30) / 3.2e9,
                                                  abs=0.15)
    with_cache = plan(path="fast", store_options={"cache": "4G"}).s_per_step
    assert with_cache.value == pytest.approx(1.4, abs=0.05)  # the RAM store's: nothing read
    # the higher of the two efficient receipts (2.498, 2.98 s), plus the one-layer cache's reads
    assert plan(path="efficient").s_per_step.value == pytest.approx(2.98 + 9.25, abs=0.15)
