"""The plan never shows a number below what a run uses: every receipt, planned without itself, is
at most what the plan says (allocated and reserved); the rules that make it so (the fused
compute's ratio, the split, the tables, the expert cache) each from a measured pair."""

from __future__ import annotations

import re

import pytest

from poros.cli.schemas import PlanGpu
from poros.core import card
from poros.core.card import _norm
from poros.core.config import PorosConfig
from poros.plan import cells, configs
from poros.plan.cells import SHAPE_KEYS, prequantized, shape_signature
from poros.plan.estimate import _nf4_host_floor, _same_shapes, _sizes
from poros.plan.make import make_plan, over_card
from poros.plan.receipts import (
    _BUDGET_MATCH,
    _CACHE_OVERHEAD,
    Request,
    _grows_request,
    _ignored,
    _key_path,
    _key_value,
    bound,
)
from poros.preflight import fit as F
from poros.preflight.fit import (
    decide_offload_for_run,
    decide_path_to_fit,
    estimate_warning,
    path_that_fits_gpu,
    run_planner,
    vram_error,
)

# config.json's sizes of the models the receipts trained (from the Hub)
QWEN25 = dict(model_type="qwen2", tie_word_embeddings=False, vocab_size=152064)
CONFIGS = {
    "Qwen/Qwen2.5-0.5B": dict(QWEN25, hidden_size=896, num_hidden_layers=24,
                              num_attention_heads=14, num_key_value_heads=2,
                              intermediate_size=4864, vocab_size=151936,
                              tie_word_embeddings=True),
    "Qwen/Qwen2.5-7B": dict(QWEN25, hidden_size=3584, num_hidden_layers=28,
                            num_attention_heads=28, num_key_value_heads=4,
                            intermediate_size=18944),
    "Qwen/Qwen2.5-32B": dict(QWEN25, hidden_size=5120, num_hidden_layers=64,
                             num_attention_heads=40, num_key_value_heads=8,
                             intermediate_size=27648),
    "Qwen/Qwen2.5-72B": dict(QWEN25, hidden_size=8192, num_hidden_layers=80,
                             num_attention_heads=64, num_key_value_heads=8,
                             intermediate_size=29568),
}
QWEN3_30B = dict(model_type="qwen3_moe", hidden_size=2048, num_hidden_layers=48,
                 num_attention_heads=32, num_key_value_heads=4, head_dim=128, vocab_size=151936,
                 tie_word_embeddings=False, intermediate_size=6144, moe_intermediate_size=768,
                 num_experts=128, num_experts_per_tok=8, decoder_sparse_step=1, mlp_only_layers=[])
CONFIGS["Qwen/Qwen3-30B-A3B"] = CONFIGS["Qwen/Qwen3-30B-A3B-Instruct-2507"] = QWEN3_30B
RTX4090 = PlanGpu(name="RTX 4090", total_gib=23.99, free_gib=None, source="spec")


def _cfg(k: dict) -> PorosConfig:
    """The config a cell's key ran."""
    kw = dict(model_name_or_path=k["model"], dataset="synthetic", quantization=k["quantization"],
              max_seq_length=k["seq_len"], per_device_batch_size=k["batch"],
              gradient_accumulation_steps=k["grad_accum"], lora_rank=k["lora_rank"],
              lora_dropout=k["lora_dropout"], path=_key_path(k),
              startup_parity_selfcheck="always" if k["selfcheck"] else "off",
              offload_embeddings_and_head=_key_value(k, "offload"))
    if k["entry"] == "dense":
        kw.update(block_size=k["block_size"], fused_compute=_key_value(k, "fused_compute"),
                  nf4_load_strategy=k["load_strategy"] or "auto")
        if _key_value(k, "dequant_cache") is not None:
            kw["nf4_dequant_cache_mb"] = _key_value(k, "dequant_cache")
    else:
        kw["moe_expert_compute"] = _key_value(k, "moe_compute")
        if _key_value(k, "expert_cache") is not None:
            kw["moe_expert_cache_gib"] = _key_value(k, "expert_cache")
    return PorosConfig(**kw)


def _fake_card(monkeypatch) -> None:
    """A CUDA host's card as poros.core.card.check reads it: an RTX 4090, a torch built for it."""
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr("torch.cuda.get_device_name", lambda device=None: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr("torch.cuda.get_arch_list", lambda: ["sm_80", "sm_86", "sm_89"])


def _gpu(name: str) -> PlanGpu:
    try:
        return card.detect_gpu(name)
    except KeyError:  # a card with no spec here: as large as any, so only its peaks decide
        return PlanGpu(name=name, total_gib=80.0, free_gib=None, source="spec")


def _plan(cfg: PorosConfig, entry: str, gpu: PlanGpu):
    return make_plan(cfg, model_type=None, support="validated", entry=entry, gpu=gpu,
                       model_config=CONFIGS.get(cfg.model_name_or_path))


CUDA = [c for c in cells.CELLS if "peak_vram_allocated" in c["fields"]]


@pytest.mark.parametrize("cell", CUDA, ids=[c["id"] for c in CUDA])
def test_a_receipt_planned_without_itself_is_never_above_the_plan(cell, monkeypatch):
    """Every cell and its repeats (the plan's own "same config") left out: what the plan says of
    its config, from the other receipts, is at least what it measured, or no number at all."""
    k = cell["key"]
    cfg, gpu = _cfg(k), _gpu(k["gpu"])
    req = Request.of(cfg, k["entry"], triton=True)
    same = [c for c in CUDA if _norm(c["key"]["gpu"]) == _norm(k["gpu"]) and any(
        req.matches(c["key"], ignore=_ignored(n, req.path))
        for n in ("peak_vram_allocated", "peak_vram_reserved"))]
    monkeypatch.setattr(cells, "CELLS", [c for c in cells.CELLS if c not in same])
    p = _plan(cfg, k["entry"], gpu)
    for name, figure in (("peak_vram_allocated", p.peak_vram_allocated),
                         ("peak_vram_reserved", p.peak_vram_reserved)):
        measured = cell["fields"].get(name, {}).get("value")
        if measured is not None and figure.value is not None:
            assert figure.value >= measured, (name, measured, figure.value, figure.method)


# The RTX 4090 receipts behind the release's claims (GiB, allocated / reserved), each config as
# it ran: planned with every receipt, the plan is at least what each measured.
CLAIMS = [
    ("Qwen/Qwen2.5-72B", "dense", dict(path="efficient", block_size=1, per_device_batch_size=8,
                                       gradient_accumulation_steps=1,
                                       offload_embeddings_and_head=True), 10.31, 13.35),
    ("Qwen/Qwen2.5-72B", "dense", dict(path="fast", block_size=2, per_device_batch_size=8,
                                       gradient_accumulation_steps=1,
                                       offload_embeddings_and_head=True), 19.65, 22.22),
    ("Qwen/Qwen2.5-7B", "dense", dict(path="efficient", block_size=1, per_device_batch_size=8,
                                      gradient_accumulation_steps=1), 8.11, 10.71),
    ("Qwen/Qwen2.5-7B", "dense", dict(path="sequential", block_size=4, per_device_batch_size=8,
                                      gradient_accumulation_steps=1), 8.71, 11.07),
    ("Qwen/Qwen2.5-7B", "dense", dict(path="efficient", block_size=1, per_device_batch_size=1,
                                      gradient_accumulation_steps=8), 7.63, 8.54),
    ("Qwen/Qwen2.5-7B", "dense", dict(path="sequential", block_size=4, per_device_batch_size=1,
                                      gradient_accumulation_steps=8), 7.89, 9.36),
    ("Qwen/Qwen3-30B-A3B", "moe", dict(path="efficient", per_device_batch_size=8,
                                       gradient_accumulation_steps=1), 7.49, 8.94),
    ("Qwen/Qwen3-30B-A3B", "moe", dict(path="fast", per_device_batch_size=8,
                                       gradient_accumulation_steps=1), 10.80, 13.04),
    ("Qwen/Qwen3-30B-A3B", "moe", dict(path="fast", per_device_batch_size=8,
                                       gradient_accumulation_steps=1, moe_expert_cache_gib=9.0),
     16.33, 17.55),
]


@pytest.mark.parametrize("model, entry, kw, alloc, reserved", CLAIMS)
def test_the_rtx4090_claims_plan_at_least_what_they_measured(model, entry, kw, alloc, reserved):
    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic", max_seq_length=512,
                      lora_dropout=0.0, **kw)
    p = _plan(cfg, entry, RTX4090)
    # a cache grown to its card is measured only at that run's free VRAM (here the card's whole
    # 23.99 GiB, more than its run had: an estimate above it); every other config is measured
    grown = _grows_request(Request.of(cfg, entry, triton=True))
    assert p.peak_vram_allocated.source == p.peak_vram_reserved.source == (
        "estimated" if grown else "measured")
    # the figures above are the receipts' to two places
    assert round(p.peak_vram_allocated.value, 2) >= alloc
    assert round(p.peak_vram_reserved.value, 2) >= reserved
    assert p.fits is True


def test_the_tables_are_keyed_and_sized():
    """offload_embeddings_and_head keeps the embedding table and LM head off the GPU: the 72B's
    receipts ran it. Asked with the tables on the GPU, the plan adds them, fp32 as the trainer
    casts them (2 x 152064 x 8192 x 4 B), and never quotes the offloaded run."""
    base = dict(path="efficient", block_size=1, per_device_batch_size=8,
                gradient_accumulation_steps=1)
    on = _plan(PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                           max_seq_length=512, lora_dropout=0.0,
                           offload_embeddings_and_head=True, **base), "dense", RTX4090)
    off = _plan(PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                            max_seq_length=512, lora_dropout=0.0,
                            offload_embeddings_and_head=False, **base), "dense", RTX4090)
    tables = 2 * 152064 * 8192 * 4 / 2**30
    assert on.peak_vram_allocated.source == "measured"
    assert off.peak_vram_allocated.source == "estimated" and "tables" in (
        off.peak_vram_allocated.method)
    assert off.peak_vram_allocated.value >= on.peak_vram_allocated.value + tables


def test_a_72b_run_with_its_tables_on_a_4090_is_never_said_to_fit():
    """A `poros train` of the 72B on an RTX 4090 with the tables on the GPU (efficient, block 4,
    1 x 8) ran out of memory there on the PEFT-module tree
    (block-major-dropout/rtx4090_72b/q72_eff_b4_def.log.txt: 22.19 GiB allocated); the plan
    sizes it above what the GPU confirms, so no fit: an upper bound past the card warns."""
    for path in ("efficient", "fast"):
        p = _plan(PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                              max_seq_length=512, lora_dropout=0.0, path=path,
                              offload_embeddings_and_head=False), "dense", RTX4090)
        assert p.fits is None and over_card(p), path
        assert p.peak_vram_allocated.value >= 22.19 - 0.5, path


def test_a_default_72b_run_on_a_4090_offloads_its_tables():
    """offload_embeddings_and_head left unset: where the plan says the run would not fit with the
    tables on the GPU and does with them in host RAM, it plans (as the run takes) them offloaded,
    and says why; set either way, it stays as set."""
    kw = dict(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic", max_seq_length=512,
              lora_dropout=0.0)
    auto = _plan(PorosConfig(**kw), "dense", RTX4090)
    on = _plan(PorosConfig(offload_embeddings_and_head=True, **kw), "dense", RTX4090)
    assert auto.offload_reason == (
        "the check keeps the embedding table and LM head in host RAM to leave GPU memory for "
        "training: with them on the GPU an estimate puts the run past this GPU's memory")
    assert auto.fits is True and not over_card(auto)
    assert auto.peak_vram_allocated == on.peak_vram_allocated and on.offload_reason is None
    assert any(w.startswith("offload_embeddings_and_head: on") for w in auto.warnings)
    assert "offload_reason" not in auto.model_dump()  # poros.plan.v1's JSON is as it was
    off = _plan(PorosConfig(offload_embeddings_and_head=False, **kw), "dense", RTX4090)
    assert off.offload_reason is None and off.fits is None
    # a model that fits with its tables on the GPU keeps them there
    small = _plan(PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", dataset="synthetic",
                              max_seq_length=512, lora_dropout=0.0), "dense", RTX4090)
    assert small.offload_reason is None


def test_one_split_bounds_another_only_as_measured():
    """Dense block-major: a 1 x 8 run holds no more than an 8 x 1 (the 7B's fused efficient pair,
    and the 72B's); MoE: the other split only with the largest ratio measured between two."""
    def planned(model, entry, kw, without):
        # the receipts the split rule reads, without the certification runs: those bound a plan
        # by their own (a same-shape model's 4090 run, a larger cache) and are tested apart
        kept = [c for c in cells.CELLS if c["id"] not in without]
        old, cells.CELLS = cells.CELLS, kept
        certified, cells.CERTIFIED = cells.CERTIFIED, []
        try:
            return _plan(PorosConfig(model_name_or_path=model, dataset="synthetic",
                                     max_seq_length=512, lora_dropout=0.0, **kw), entry, RTX4090)
        finally:
            cells.CELLS, cells.CERTIFIED = old, certified

    one8 = planned("Qwen/Qwen2.5-7B", "dense",
                   dict(path="efficient", block_size=1, per_device_batch_size=1,
                        gradient_accumulation_steps=8),
                   {"qwen2.5-7b-rtx4090-oasst1-fused-efficient-b1-1x8",
                    "qwen2.5-7b-rtx4090-synthetic-efficient-b1"})  # and PEFT's 1x8
    assert "efficient-b1-8x1's peak" in one8.peak_vram_allocated.method
    assert one8.peak_vram_allocated.value >= 7.628
    moe = planned("Qwen/Qwen3-30B-A3B", "moe",
                  dict(path="fast", per_device_batch_size=1, gradient_accumulation_steps=8,
                       moe_expert_cache_gib=9.0),
                  {"qwen3-30b-a3b-rtx4090-oasst1-fused-fast-1x8-c9"})
    assert "its split 8x1 as 1x8" in moe.peak_vram_allocated.method
    assert moe.peak_vram_allocated.value >= 17.121 and moe.peak_vram_reserved.value >= 19.887


def test_a_same_shapes_model_plans_from_its_twin():
    """Qwen3-30B-A3B-Instruct-2507 has the base's config.json shapes: the base's RTX 4090 receipts
    of the fused default plan it, said so."""
    p = _plan(PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
                          dataset="synthetic", max_seq_length=512, lora_dropout=0.0,
                          per_device_batch_size=8, gradient_accumulation_steps=1), "moe", RTX4090)
    assert p.path == "efficient" and p.peak_vram_allocated.source == "estimated"
    assert "Qwen/Qwen3-30B-A3B's, the same shapes" in p.peak_vram_allocated.method
    assert p.peak_vram_allocated.value >= 7.488 and p.fits is True  # q30k_def_eff


GROWN = "qwen3-30b-a3b-rtx4090-oasst1-fused-fast"  # path fast, moe_expert_cache_gib unset


@pytest.mark.parametrize("free", [23.5, 23.1, 22.95, 22.0, 20.0, 17.0, 15.0, 9.0])
def test_a_cache_grown_to_its_card_neither_refuses_nor_understates_another_free(free):
    """Its expert cache grew to what its run could hold (22.95 GiB), so its peak is that run's
    alone: with less free the cache is smaller (never a refusal from it, nor "measured"), with
    more it is larger (the plan adds what the larger cache can hold); at its own free VRAM it is
    measured."""
    (cell,) = [c for c in cells.CELLS if c["id"] == GROWN]
    budget = cell["fields"]["vram_budget"]["value"]
    measured = cell["fields"]["peak_vram_allocated"]["value"]
    gpu = PlanGpu(name="NVIDIA GeForce RTX 4090", total_gib=23.99, free_gib=free, source="spec")
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic",
                      max_seq_length=512, lora_dropout=0.0, path="fast",
                      per_device_batch_size=8, gradient_accumulation_steps=1)
    p = _plan(cfg, "moe", gpu)
    alloc = p.peak_vram_allocated
    assert p.fits is not False and p.bound is None or p.bound.fits is True
    if budget <= free <= budget + _BUDGET_MATCH:
        assert alloc.source == "measured" and alloc.value == measured
    else:
        assert alloc.source == "estimated" and alloc.value >= measured
    if free > budget + _BUDGET_MATCH:  # the larger cache it grows is in the figure
        assert alloc.value >= measured + (free - budget) * _CACHE_OVERHEAD - 1e-3
    if free < measured:
        assert over_card(p) and p.fits is None  # it warns, as an estimate past the card does


def test_a_grown_receipt_is_no_floor_on_its_own_card():
    """bound() refused from the grown receipt on its own card whenever less was free (the
    fast default on a 4090 with a display server); only a fixed cache's peak can refuse."""
    gpu = PlanGpu(name="RTX 4090", total_gib=23.99, free_gib=12.0, source="spec")
    req = Request.of(PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic",
                                   max_seq_length=512, lora_dropout=0.0, path="fast",
                                   per_device_batch_size=8, gradient_accumulation_steps=1),
                       "moe", triton=True)
    assert _grows_request(req)
    b = bound(req, gpu)
    assert b is None or b.fits is True


@pytest.mark.parametrize("model", sorted(CONFIGS))
def test_a_model_no_receipt_names_plans_from_its_shape_twin(model):
    """A local folder (or a fine-tune) of a receipt model: no cell names it, so its config.json's
    shapes find the receipts (poros.plan.cells.SHAPES, from each model's Hub config.json)."""
    twin = cells.shape_twin("/models/local-copy", CONFIGS[model])
    assert twin == model or model in _same_shapes(twin) or (
        cells.SHAPES["models"][twin]["signature"] == shape_signature(CONFIGS[model]))
    assert cells.shape_twin(model, CONFIGS[model]) is None  # its own receipts name it
    other = dict(CONFIGS[model], num_hidden_layers=CONFIGS[model]["num_hidden_layers"] + 1)
    assert cells.shape_twin("/models/local-copy", other) is None  # another shape: no receipts


@pytest.mark.parametrize("quant", [{"quant_method": "awq", "bits": 4},
                                   {"quant_method": "gptq", "bits": 4},
                                   {"quant_method": "fp8", "fmt": "e4m3"}])
def test_a_prequantized_checkpoint_has_no_shape_twin(quant):
    """Qwen2.5-7B-Instruct-AWQ has Qwen2.5-7B's shapes, but its weights load through another
    quantizer than the bf16 checkpoints the receipts trained: their memory is not its."""
    config = dict(CONFIGS["Qwen/Qwen2.5-7B"], quantization_config=quant)
    assert prequantized(config) == quant["quant_method"]
    assert cells.shape_twin("Qwen/Qwen2.5-7B-Instruct-AWQ", config) is None
    assert cells.shape_twin("Qwen/Qwen2.5-7B-Instruct-AWQ", CONFIGS["Qwen/Qwen2.5-7B"]) is not None
    assert prequantized(CONFIGS["Qwen/Qwen2.5-7B"]) is None


def test_a_local_72b_folder_plans_and_offloads_as_the_hub_id_does(tmp_path):
    """A 72B loaded from a local folder got no figures and no automatic offload (it OOMed at the
    defaults on a 4090): planned from the Hub id's receipts, it takes the tables off the GPU."""
    import json

    folder = tmp_path / "Qwen2.5-72B-Instruct"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(
        {**CONFIGS["Qwen/Qwen2.5-72B"], "architectures": ["Qwen2ForCausalLM"]}))
    config = configs.read_model_config(str(folder))
    kw = dict(dataset="synthetic", max_seq_length=512, lora_dropout=0.0)
    local = make_plan(PorosConfig(model_name_or_path=str(folder), **kw), model_type="qwen2",
                        support="validated", entry="dense", gpu=RTX4090, model_config=config)
    hub = make_plan(PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", **kw),
                      model_type="qwen2", support="validated", entry="dense", gpu=RTX4090,
                      model_config=config)
    assert local.model == str(folder) and local.offload_reason and local.fits is True
    assert local.peak_vram_allocated == hub.peak_vram_allocated
    assert local.offload_reason == hub.offload_reason
    assert any("planned from the receipts of Qwen/Qwen2.5-72B" in w for w in local.warnings)


def test_the_trainer_offloads_a_local_72b_folder(tmp_path, monkeypatch):
    """PorosTrainer decides offload_embeddings_and_head as the plan does, for a local folder too."""
    import json

    pytest.importorskip("torch")
    from poros.core import env

    folder = tmp_path / "q72"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(CONFIGS["Qwen/Qwen2.5-72B"]))
    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: RTX4090)
    monkeypatch.setattr(env, "host_ram_free", lambda: 200.0)
    cfg = PorosConfig(model_name_or_path=str(folder), dataset="synthetic", max_seq_length=512,
                      lora_dropout=0.0)
    F.check(cfg, F.planner_here("qwen2", configs.read_model_config(str(folder))))
    assert cfg.offload_embeddings_and_head is True and cfg.offload_reason


def test_the_shape_table_is_keyed_as_the_plan_reads_it():
    """shapes.json holds each model's signature over SHAPE_KEYS: a key added there needs the
    table rebuilt (scripts/build_plan_shapes.py), or no model would match its twin."""
    assert cells.SHAPES["models"]
    for model, entry in cells.SHAPES["models"].items():
        assert set(entry["signature"]) == {"model_type", *SHAPE_KEYS}, model
        assert any(c["key"]["model"] == model for c in cells.CELLS), model


def test_a_twin_without_receipts_here_plans_nothing(monkeypatch):
    """With the receipts narrowed (a test's CELLS, a release without some), a model whose twin has
    none left has no twin: never a plan of a model by itself, round and round."""
    monkeypatch.setattr(cells, "CELLS",
                        [c for c in cells.CELLS if c["key"]["model"] != "Qwen/Qwen2.5-72B"])
    assert cells.shape_twin("/models/q72", CONFIGS["Qwen/Qwen2.5-72B"]) is None
    p = make_plan(PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic"),
                    model_type="qwen2", support="validated", entry="dense", gpu=RTX4090,
                    model_config=CONFIGS["Qwen/Qwen2.5-72B"])
    assert p.model == "Qwen/Qwen2.5-72B"


def test_a_too_big_default_run_exits_before_any_loader(monkeypatch, tmp_path,
                                                       recipe_configs_cached):
    """A run that cannot fit is refused in the preflight, in seconds, never after the download or
    the load. Qwen2.5-72B at the defaults on a 4090 with 6 GiB free: even with the tables in host
    RAM a run no larger peaked past it, on every path. `poros train` exits 3 naming what would fit,
    and nothing that downloads or loads a model was called."""
    pytest.importorskip("torch")
    import huggingface_hub
    import transformers
    from typer.testing import CliRunner

    from poros.cli import main
    from poros.core import compute, env
    from poros.moe import load as moe_load
    from poros.preflight import arch as preflight_arch
    from poros.train import model as train_model
    from poros.train.trainer import PorosTrainer

    called = []

    def loader(name):
        return lambda *a, **k: called.append(name) or pytest.fail(f"{name} was called")

    monkeypatch.setattr(PorosTrainer, "from_pretrained", loader("PorosTrainer.from_pretrained"))
    monkeypatch.setattr(train_model.PorosModel, "from_pretrained", loader("PorosModel"))
    monkeypatch.setattr(moe_load, "load_moe_streamed", loader("load_moe_streamed"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        loader("transformers"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", loader("snapshot_download"))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    _fake_card(monkeypatch)  # the card poros.core.card.check reads: the plan's RTX 4090
    small = RTX4090.model_copy(update={"free_gib": 6.0, "source": "torch"})
    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: small)
    monkeypatch.setattr(compute, "triton_installed", lambda: True)  # the 4090's stack
    monkeypatch.setattr(env, "host_ram_free", lambda: 200.0)
    monkeypatch.setattr(configs, "read_model_config",
                        lambda m, revision=None: CONFIGS["Qwen/Qwen2.5-72B"])
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen2", ""))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: Qwen/Qwen2.5-72B\ndataset: synthetic\n"
                   f"max_seq_length: 512\nlora_dropout: 0.0\noutput_dir: {tmp_path / 'out'}\n")
    res = CliRunner().invoke(main.app, ["train", str(cfg), "--json"])
    assert res.exit_code == 3 and "E_OOM_PREDICTED" in res.output, res.output
    assert "won't fit" in res.output and "--force trains anyway" in res.output
    # the decision and what to change: the GPU memory measured free here, no planned figure
    assert set(re.findall(r"([\d.]+) GiB", res.output)) <= {"6.0"}, res.output
    assert called == []


def _stub(fits: dict, over: dict = {}):
    """A plan function over a stand-in: each path's verdict, and whether it is past the card."""
    import types

    def plan(cfg):
        path = cfg.path or "efficient"
        f = fits[path]
        return types.SimpleNamespace(
            path=path, fits=f, path_reason=None, host_fits=None, fit_basis=f"{path} basis",
            gpu=types.SimpleNamespace(free_gib=10.0, total_gib=24.0),
            peak_vram_allocated=types.SimpleNamespace(value=30.0 if over.get(path) else 5.0),
            peak_vram_reserved=types.SimpleNamespace(value=None))
    return plan


@pytest.mark.parametrize("fits, over, want", [
    ({"efficient": False, "sequential": True, "fast": None}, {}, "sequential"),  # adapt
    ({"efficient": False, "sequential": False, "fast": None}, {}, "fast"),  # within, estimated
    ({"efficient": False, "sequential": False, "fast": None}, {"fast": True}, None),  # refuse
    ({"efficient": None, "sequential": True, "fast": None}, {"efficient": True}, "sequential"),
    # past the card by an estimate: a path whose estimate is within it (no late out-of-memory
    # where an adaptation fits)
    ({"efficient": None, "sequential": None, "fast": None}, {"efficient": True}, "sequential"),
    ({"efficient": None, "sequential": None, "fast": None},
     {"efficient": True, "sequential": True}, "fast"),
    ({"efficient": None, "sequential": None, "fast": None},
     {"efficient": True, "sequential": True, "fast": True}, None),  # none: it warns
    ({"efficient": True, "sequential": False, "fast": False}, {}, None),  # fits as it is
])
def test_a_run_left_to_decide_takes_the_path_that_fits(fits, over, want):
    """Adapt before refusing: a default run whose path would not fit the GPU (or is past it by an
    estimate) takes the path the plan says fits; refused only where none does."""
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-7B", dataset="synthetic")
    got = path_that_fits_gpu(cfg, _stub(fits, over))
    assert (got and got[0]) == want
    set_by_you = PorosConfig(model_name_or_path="Qwen/Qwen2.5-7B", dataset="synthetic",
                             path="efficient")
    assert path_that_fits_gpu(set_by_you, _stub(fits, over)) is None  # a path you set stays


def test_the_trainer_refuses_an_unfit_run_before_the_download(monkeypatch):
    """PorosTrainer.from_pretrained, without `poros train`'s gate: the same plan, the same
    adaptations, and the same refusal, before the download (check_fit=False skips it)."""
    pytest.importorskip("torch")
    from poros.core import compute, env
    from poros.core.errors import PorosVramError

    small = RTX4090.model_copy(update={"free_gib": 6.0, "source": "torch"})
    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: small)
    monkeypatch.setattr(compute, "triton_installed", lambda: True)
    monkeypatch.setattr(env, "host_ram_free", lambda: 200.0)
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                      max_seq_length=512, lora_dropout=0.0)
    with pytest.raises(PorosVramError) as err:
        F.refuse(F.check(cfg, F.planner_here("qwen2", CONFIGS["Qwen/Qwen2.5-72B"])))
    assert err.value.info.error_type == "PorosVramError" or "not to fit" in str(err.value)
    assert "What to change:" in err.value.info.remediation and "check_fit=False" in (
        err.value.info.remediation)
    roomy = small.model_copy(update={"free_gib": 23.5})
    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: roomy)
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                      max_seq_length=512, lora_dropout=0.0)
    F.refuse(F.check(cfg, F.planner_here("qwen2", CONFIGS["Qwen/Qwen2.5-72B"])))
    assert cfg.offload_embeddings_and_head is True  # adapted, not refused


def test_a_resume_from_another_rank_is_refused_before_the_load(tmp_path):
    """A checkpoint trained at another LoRA rank, quantization or target set holds other tensors:
    poros train says so from its poros_config.json before any model loads."""
    import json

    from poros.preflight.resume import resume_clash

    ckpt = tmp_path / "checkpoint-2"
    ckpt.mkdir()
    (ckpt / "poros_config.json").write_text(json.dumps(
        {"quantization": "nf4", "lora_rank": 8, "lora_target_modules": ["v_proj", "q_proj"]}))
    base = dict(model_name_or_path="Qwen/Qwen2.5-0.5B", dataset="synthetic")
    assert "lora_rank 8" in resume_clash(str(ckpt), PorosConfig(**base, lora_rank=16))
    assert resume_clash(str(ckpt), PorosConfig(**base, lora_rank=8,
                                               lora_target_modules=["q_proj", "v_proj"])) is None
    assert "lora_target_modules" in resume_clash(
        str(ckpt), PorosConfig(**base, lora_rank=8, lora_target_modules=["q_proj"]))
    assert resume_clash(str(tmp_path / "nothing"), PorosConfig(**base)) is None


# Starting a run never ends in a late out-of-memory where an adaptation fits.
# Qwen2.5-72B at 512 tokens, dropout 0, on a 4090 with 9 to 12 GiB free: no receipt bounds it,
# and the estimate with the tables on the GPU (at most 22.8 GiB allocated) is far past the card.


def _a_4090_with(monkeypatch, free: float) -> PlanGpu:
    from poros.core import compute, env

    gpu = RTX4090.model_copy(update={"free_gib": free, "source": "torch"})
    monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: gpu)
    monkeypatch.setattr(compute, "triton_installed", lambda: True)  # the 4090's stack
    monkeypatch.setattr(env, "host_ram_free", lambda: 200.0)
    monkeypatch.setattr(configs, "read_model_config", lambda m, revision=None: CONFIGS[m])
    return gpu


def _72b(**kw) -> PorosConfig:
    return PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic",
                       max_seq_length=512, lora_dropout=0.0, **kw)


def test_an_estimate_past_the_card_moves_the_tables_first(monkeypatch):
    """Past the card by an estimate with the tables on the GPU and past it, by less, with them in
    host RAM: they move (the same bits), the reason recorded; a value you set stays."""
    gpu = _a_4090_with(monkeypatch, 10.0)
    plan = run_planner("qwen2", gpu=gpu, model_config=CONFIGS["Qwen/Qwen2.5-72B"],
                         host_free_gib=200.0)
    cfg = _72b()
    on_gpu = plan(cfg.model_copy(update={"offload_embeddings_and_head": False}))
    assert on_gpu.fits is None and over_card(on_gpu)  # before: warned, and loaded like this
    decide_path_to_fit(cfg, plan)
    p = plan(cfg)
    decide_offload_for_run(cfg, p)
    assert cfg.offload_embeddings_and_head is True and "holds less" in cfg.offload_reason
    assert p.peak_vram_allocated.value < on_gpu.peak_vram_allocated.value
    assert p.fits is None and over_card(p)  # still past it: a warning, never a refusal
    warn = estimate_warning(cfg, plan)
    assert warn and "may run out of GPU memory at its first step" in warn
    assert "never changed for you" in warn and cfg.block_size is None  # named only
    yours = _72b(offload_embeddings_and_head=False)
    decide_path_to_fit(yours, plan)
    decide_offload_for_run(yours, plan(yours))
    assert yours.offload_embeddings_and_head is False and yours.offload_reason is None


def test_the_warning_names_the_block_size_that_fits(monkeypatch):
    gpu = _a_4090_with(monkeypatch, 12.0)
    plan = run_planner("qwen2", gpu=gpu, model_config=CONFIGS["Qwen/Qwen2.5-72B"],
                         host_free_gib=200.0)
    cfg = _72b()
    decide_path_to_fit(cfg, plan)
    decide_offload_for_run(cfg, plan(cfg))
    warn = estimate_warning(cfg, plan)
    assert "block_size: 1 draws no such warning" in warn and cfg.block_size is None
    one = plan(cfg.model_copy(update={"block_size": 1}))
    assert one.fits is True or not over_card(one)


def test_the_cli_gate_adapts_and_warns_instead_of_a_late_oom(monkeypatch, tmp_path,
                                                             recipe_configs_cached):
    """`poros train` on the 72B with 12 GiB free: before, it warned and loaded with the tables on
    the GPU. Now the trainer is handed the tables in host RAM, with why, and the warning names the
    block_size that fits, which the config keeps as set (unset: 4)."""
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    from poros.cli import main
    from poros.preflight import arch as preflight_arch
    from poros.train.trainer import PorosTrainer

    _a_4090_with(monkeypatch, 12.0)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    _fake_card(monkeypatch)  # the card poros.core.card.check reads: the plan's RTX 4090
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen2", ""))
    handed = []

    class Loaded(Exception):
        pass

    def loads(*a, config=None, check_fit=True, **k):
        handed.append((config, check_fit))
        raise Loaded

    monkeypatch.setattr(PorosTrainer, "from_pretrained", loads)
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: Qwen/Qwen2.5-72B\ndataset: synthetic\n"
                   f"max_seq_length: 512\nlora_dropout: 0.0\noutput_dir: {tmp_path / 'out'}\n")
    res = CliRunner().invoke(main.app, ["train", str(cfg)])
    assert "E_OOM_PREDICTED" not in res.output, res.output
    assert handed, res.output
    run, check_fit = handed[0]
    assert run.offload_embeddings_and_head is True and "holds less" in run.offload_reason
    assert run.block_size in (None, 4) and check_fit is False
    said = " ".join(res.output.split())
    assert "may not fit" in said and "block_size: 1 draws no such warning" in said
    assert "should fit" not in said and "estimated to fit" not in said
    assert "at most" not in said and "estimated" not in said  # no planner figure, no estimate


def test_from_pretrained_adapts_and_warns_before_the_load(monkeypatch, caplog):
    """PorosTrainer.from_pretrained, no CLI: the same adaptation and the same warning, before the
    download, and no refusal on an estimate alone."""
    pytest.importorskip("torch")
    from poros.train import trainer

    _a_4090_with(monkeypatch, 12.0)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    _fake_card(monkeypatch)  # the card poros.core.card.check reads: the plan's RTX 4090
    reached = []

    class Loaded(Exception):
        pass

    def stop(config, *a, **k):
        reached.append(config)
        raise Loaded

    monkeypatch.setattr(trainer, "apply_expandable_segments", stop)
    monkeypatch.setattr(trainer, "enforce_determinism", lambda *a, **k: None)
    with caplog.at_level("WARNING", logger=trainer.logger.name), pytest.raises(Loaded):
        trainer.PorosTrainer.from_pretrained("Qwen/Qwen2.5-72B", lora_dropout=0.0,
                                             max_seq_length=512, dataset="synthetic")
    run = reached[0]
    assert run.offload_embeddings_and_head is True and "holds less" in run.offload_reason
    assert run.block_size is None and run.path == "efficient"
    said = " ".join(caplog.text.split())
    assert "may not fit" in said and "block_size: 1 draws no such warning" in said
    assert "estimated to fit" not in said


def test_a_dense_run_whose_nf4_layers_outgrow_host_ram_is_refused_from_config_json():
    """No receipt sizes the 72B's host RAM at poros train's defaults, but the streamed base stays
    in host RAM between its visits: 80 layers in NF4 are 33.7 GiB, past a 31 GB host's 30.35 GiB
    available -- refused before the download, never a download then a late host OOM. A host
    with room plans on, and a MoE model is not held to it (its stores size it)."""
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", dataset="synthetic")
    gpu = card.detect_gpu("RTX 5090")

    def plan(free):
        return make_plan(cfg, model_type="qwen2", support="validated", entry="dense", gpu=gpu,
                           model_config=CONFIGS["Qwen/Qwen2.5-72B"], host_free_gib=free)

    small = plan(30.35)
    assert small.host_fits is False and "NF4 alone take 33.7" in small.host_fit_basis
    assert plan(64.0).host_fits is not False
    floor = _nf4_host_floor(cfg, "dense", _sizes(CONFIGS["Qwen/Qwen2.5-72B"]))
    assert 33 < floor < 34
    assert _nf4_host_floor(cfg, "moe", _sizes(QWEN3_30B)) is None


def test_a_plan_of_a_card_poros_train_refuses_says_so():
    """A pre-Ampere card (no native bf16): `poros train` refuses a bf16 run before the download,
    so its plan says "refused on this card" instead of a fit."""
    t4 = PlanGpu(name="Tesla T4", total_gib=14.58, free_gib=14.5, source="nvidia-smi",
                 capability=(7, 5))
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", dataset="synthetic")
    p = _plan(cfg, "dense", t4)
    assert p.fits is False and p.fit_basis.startswith("refused on this card: sm_75 (Turing)")
    assert any(w.startswith("refused on this card (Tesla T4)") for w in p.warnings)
    assert vram_error(cfg, lambda c: _plan(c, "dense", t4), p) is None  # not a fit's refusal
    fp16 = _plan(cfg.model_copy(update={"torch_dtype": "float16"}), "dense", t4)
    assert not fp16.fit_basis.startswith("refused")  # the experimental opt-in is planned
    assert _plan(cfg, "dense", RTX4090).fit_basis[:7] != "refused"


def _train_json(monkeypatch, tmp_path, gpu_free, host_free, model, seq):
    """``poros train <config> --json`` on a faked RTX 4090 with ``host_free`` GiB of host RAM:
    (exit code, the JSON error, the check's own Fit for the same run)."""
    import json

    from typer.testing import CliRunner

    from poros.cli import contract, main
    from poros.core import env
    from poros.preflight import arch as preflight_arch

    _a_4090_with(monkeypatch, gpu_free)  # its configs.read_model_config: CONFIGS, no download
    monkeypatch.setattr(env, "host_ram_free", lambda: host_free)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    _fake_card(monkeypatch)
    monkeypatch.setattr(preflight_arch, "resolve_model_type",
                        lambda m: (CONFIGS[m]["model_type"], ""))
    # the model's repo as a cache holding its config.json answers it (poros.preflight.command
    # .model_refusal asks hub_fault): no Hub and no download cache on a CI runner
    monkeypatch.setattr(preflight_arch, "hub_fault", lambda model, revision=None: None)
    # --json in this process: no stdout redirect and no module state left for later tests
    monkeypatch.setattr(contract, "isolate_stdout", lambda: None)
    for name in ("_stdout", "_argv", "_forced", "_emitted"):
        monkeypatch.setattr(contract, name, getattr(contract, name))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: {model}\ndataset: synthetic\nmax_seq_length: {seq}\n"
                   f"lora_dropout: 0.0\noutput_dir: {tmp_path / 'out'}\n")
    res = CliRunner().invoke(main.app, ["train", str(cfg), "--json"])
    lines = [x for x in res.output.splitlines() if x.startswith("{")]
    body = json.loads(lines[-1])["error"] if lines else None
    mt = CONFIGS[model]["model_type"]
    fit = F.check(PorosConfig(model_name_or_path=model, dataset="synthetic",
                              max_seq_length=seq, lora_dropout=0.0),
                  F.planner_here(mt, CONFIGS[model]))
    return res.exit_code, body, fit


@pytest.mark.parametrize("model, gpu_free, host_free, seq, code", [
    ("Qwen/Qwen2.5-72B", 23.5, 20.0, 512, "E_HOST_OOM_PREDICTED"),  # the frozen base: at least
    ("Qwen/Qwen2.5-32B", 23.5, 10.0, 512, "E_HOST_OOM_PREDICTED"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", 23.5, 14.0, 1024, "E_HOST_OOM_PREDICTED"),
    ("Qwen/Qwen2.5-7B", 2.0, 200.0, 2048, "E_OOM_PREDICTED"),
])
def test_poros_train_says_the_refusals_own_words(monkeypatch, tmp_path, model, gpu_free,
                                                 host_free, seq, code):
    """Through poros train itself (its gate, not the trainer, refuses): the code and exit, and
    the error's own head, cause and every piece of its advice, so one source says it on every
    surface; advice only where it helps (no path advice where the frozen base is the cause)."""
    pytest.importorskip("torch")
    from poros.core.errors import EXIT_CODES

    exit_code, body, fit = _train_json(monkeypatch, tmp_path, gpu_free, host_free, model, seq)
    err = fit.host if code == "E_HOST_OOM_PREDICTED" else fit.vram
    assert err is not None and err.code == code
    assert exit_code == EXIT_CODES[code] == 3 and body["code"] == code, body
    said, hint = body["message"], body["hint"]  # the trainer's shape: head and cause; advice
    assert said.startswith(err.head + (f": {err.cause}" if err.cause else "")), said
    for advice in err.advice:
        assert advice in hint, (advice, hint)
    assert "poros recipes lists the runs certified on" in hint and "--force trains anyway" in hint
    said = said + " " + hint
    assert "fit" not in said.replace("won't fit", "").lower(), said  # no fit asserted
    if code == "E_HOST_OOM_PREDICTED":
        # the path was left to the run, so the check tried every path already: none is offered
        assert "path:" not in said, said
