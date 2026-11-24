"""Block-major accumulation (``path: efficient`` or ``fast``) is a CUDA run's default; the Mac
and the CPU train sequentially.

A config parses the same everywhere: ``path`` unset, which a Mac or CPU run trains sequentially
(the Mac validator and its digests, tests/metal/test_any_mac.py, read that).
``apply_cuda_defaults()`` decides ``path`` for a CUDA run where the config left it unset, and
only a CUDA run calls it (PorosTrainer.from_pretrained, poros train, poros plan). Block-major
draws the sequential loop's dropout masks (tests/test_block_major_dropout.py), so the dropout
never decides the loop, and neither does the number of micro-batches a step (block-major is
no slower at one); what does is sequential_reason: what block-major
cannot train as the sequential loop does. Where it decides
sequential it says why (``path_reason``).
"""

from __future__ import annotations

import logging
import warnings

import pytest

from poros.core.config import PorosConfig, moe_grouping, recorded_config, train_path

BASE = {"model_name_or_path": "Qwen/Qwen2.5-0.5B", "dataset": "synthetic"}


def _cfg(**kw) -> PorosConfig:
    return PorosConfig(**{**BASE, **kw})


def _decided(cfg: PorosConfig) -> tuple:
    return cfg.path, cfg.path_reason


@pytest.mark.parametrize("kw", [{}, {"lora_dropout": 0.0}, {"gradient_accumulation_steps": 1}])
def test_a_config_parses_with_the_loop_left_to_the_run(kw):
    cfg = _cfg(**kw)
    assert (cfg.path, cfg.path_reason, train_path(cfg)) == (None, None, "sequential")


TABLES = ["q_proj", "v_proj", "lm_head"]
MOE = {"model_type": "qwen3_moe"}
TABLE = "an embedding table or LM head that trains"


@pytest.mark.parametrize("kw, model, want", [
    ({}, None, ("efficient", None)),  # the default: block-major, inputs in host RAM
    ({"lora_dropout": 0.0}, None, ("efficient", None)),
    ({"lora_dropout": 0.2}, None, ("efficient", None)),  # any dropout: its masks are drawn
    ({"path": "fast"}, None, ("fast", None)),  # yours
    ({"path": "sequential"}, None, ("sequential", None)),  # yours: no reason to give
    ({"nf4_dequant_cache_mb": 0}, None, ("efficient", None)),
    ({"gradient_accumulation_steps": 1}, None, ("efficient", None)),  # no slower
    # what block-major cannot train as the sequential loop does (sequential_reason)
    ({"lora_target_modules": TABLES}, None, ("sequential", TABLE)),
    ({"lora_target_modules": "lm_head|q_proj"}, None, ("sequential", TABLE)),
    ({"preserve_rng_state": False}, None,
     ("sequential", "lora_dropout with preserve_rng_state: false")),
    ({"preserve_rng_state": False, "lora_dropout": 0.0}, None, ("efficient", None)),
    ({}, {"model_type": "qwen2", "attention_dropout": 0.1},
     ("sequential", "the model's own random draws in training (attention_dropout)")),
    ({}, {"model_type": "qwen2", "attention_dropout": 0.0}, ("efficient", None)),
    ({}, MOE, ("efficient", None)),  # its states keyed by layer in either loop
    ({}, {**MOE, "output_router_logits": True},
     ("sequential", "a MoE model's router aux loss (output_router_logits)")),
    ({}, {"model_type": "x", "text_config": {**MOE, "router_jitter_noise": 0.1}},
     ("sequential", "the model's own random draws in training (router_jitter_noise)")),
])
def test_apply_cuda_defaults_decides_only_what_was_left_unset(kw, model, want):
    cfg = _cfg(**kw)
    assert cfg.apply_cuda_defaults(model) is None
    assert _decided(cfg) == want
    cfg.apply_cuda_defaults(model)
    assert _decided(cfg) == want  # decided once
    assert cfg.path_decided is ("path" not in kw)


@pytest.mark.parametrize("kw, why", [
    ({"lora_target_modules": TABLES}, "an embedding table or LM head that trains"),
    ({"preserve_rng_state": False}, "lora_dropout with preserve_rng_state: false"),
])
@pytest.mark.parametrize("path", ["efficient", "fast"])
def test_a_block_major_path_you_set_is_refused_where_it_cannot_be_the_sequential_loop(kw, why,
                                                                                    path):
    with pytest.raises(ValueError, match=f"path: {path} trains block-major.*{why}"):
        _cfg(path=path, **kw)


@pytest.mark.parametrize("kw, want", [
    ({}, "pow2"),  # the reference, on every loop
    ({"path": "fast"}, "pow2"),  # a block-major path you set: the same grouping
    ({"path": "sequential"}, "pow2"),
    ({"moe_expert_grouping": "single"}, "single"),  # only where you set it
    ({"moe_expert_grouping": "single", "path": "fast"}, "single"),
])
@pytest.mark.parametrize("cuda", [False, True])
def test_one_rule_says_which_expert_grouping_a_moe_run_takes(kw, want, cuda):
    cfg = _cfg(**kw)
    if cuda:
        cfg.apply_cuda_defaults(MOE)
    assert moe_grouping(cfg) == want
    again = PorosConfig(**cfg.model_dump())  # a resume or an eval reads the same grouping
    assert moe_grouping(again) == want


def test_an_explicit_contradiction_refuses_at_parse():
    with pytest.raises(ValueError):
        _cfg(path="eficient")


def test_a_copy_plans_what_cuda_would_train_and_leaves_the_original():
    cfg = _cfg()
    copy = cfg.model_copy()
    copy.apply_cuda_defaults()
    assert copy.path == "efficient" and cfg.path is None
    assert cfg.apply_cuda_defaults() is None and cfg.path == "efficient"


def test_a_saved_config_reloads_as_it_ran():
    """poros_config.json / config.resolved.json hold the decided path, with its reason where
    there is one (recorded_config); reloaded, the path is yours."""
    from poros.core.config import config_fields

    for kw in ({}, {"path": "fast"}, {"lora_target_modules": TABLES},
               {"gradient_accumulation_steps": 1}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ran = _cfg(**kw)
            ran.apply_cuda_defaults()
            said = recorded_config(ran)
            assert said["path"] == ran.path
            assert said["path_reason"] == ran.path_reason
            again = PorosConfig(**config_fields(said))  # a resume, Run.open, eval
        assert again.apply_cuda_defaults() is None
        assert again.path == ran.path and not again.path_decided, kw
    mac = PorosConfig(**config_fields(recorded_config(_cfg())))  # what a Mac or CPU run saved
    assert mac.path is None and train_path(mac) == "sequential"


@pytest.mark.parametrize("kw, cuda, path", [
    ({}, False, "sequential"),  # a Mac or CPU run
    ({"gradient_accumulation_steps": 1}, True, "efficient"),  # no slower at one
    ({"lora_target_modules": TABLES}, True, "sequential"),
    ({"path": "sequential"}, True, "sequential"),
    ({}, True, "efficient"),
    ({"path": "fast"}, True, "fast"),
])
def test_a_run_s_records_name_the_path(kw, cuda, path):
    cfg = _cfg(**kw)
    if cuda:
        cfg.apply_cuda_defaults()
    assert train_path(cfg) == path
    for mode in ("python", "json"):  # poros_config.json, config.resolved.json
        said = recorded_config(cfg, mode=mode)
        assert (said["path"] or "sequential") == path
        assert "block_major_accumulation" not in said and "offload_boundaries" not in said


def test_the_schema_says_what_each_setting_does():
    props = PorosConfig.model_json_schema()["properties"]
    said = str(props["path"])
    assert all(w in said for w in ("efficient", "fast", "sequential", "host RAM", "path_reason"))
    assert "block_major_accumulation" not in props and "offload_boundaries" not in props
    for name in ("nf4_dequant_cache_mb", "moe_expert_grouping", "cuda_expandable_segments",
                 "lora_dropout"):
        assert props[name].get("description"), name
    assert all(w in props["lora_dropout"]["description"] for w in ("0.05", "poros train", "0"))


def test_the_events_schema_says_what_the_setup_event_s_path_is():
    from poros.core.events import DATA_FIELDS

    said = DATA_FIELDS["path"]["description"]
    assert all(word in said for word in ("setup", "sequential", "efficient", "fast"))
    assert "sequential" in DATA_FIELDS["path_reason"]["description"]


# -- the trainer: CUDA applies them, nothing else does --------------------------------------------


class _Stop(Exception):
    pass


# Process-wide switches from_pretrained would flip (determinism, the allocator): off in these tests.
QUIET = {"deterministic": False, "cuda_expandable_segments": False}


def _built(monkeypatch, cuda: bool, gpu=None, model=None, **kw) -> PorosConfig:
    """The config PorosTrainer.from_pretrained trains ``model`` (BASE's) with, stopped before
    anything loads; on CUDA, ``gpu`` is the card its plan sees (None: none detected)."""
    torch = pytest.importorskip("torch")
    import poros.train.trainer as trainer_module
    from poros.train.trainer import PorosTrainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if cuda:  # the card poros.core.card.check reads: an RTX 4090 and a torch built for it
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
        monkeypatch.setattr(torch.cuda, "get_device_name",
                            lambda device=None: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_86", "sm_89"])
    monkeypatch.setattr("poros.core.card.detect_gpu", lambda *a, **k: gpu)
    # the host the plan sees: 59 GiB free, not whatever this machine has free
    # (a 72B's NF4 layers alone take 33.7 GiB of host RAM: on a smaller host the plan refuses)
    monkeypatch.setattr("poros.core.env.host_ram_free", lambda: 59.0)
    seen = {}

    def stop(config, device):
        seen["config"] = config
        raise _Stop

    monkeypatch.setattr(trainer_module, "apply_expandable_segments", stop)
    with pytest.raises(_Stop):
        PorosTrainer.from_pretrained(model or BASE["model_name_or_path"],
                                     **(kw if "config" in kw else {**QUIET, **kw}))
    return seen["config"]


def _model(monkeypatch, said):
    monkeypatch.setattr("poros.plan.configs.read_model_config", lambda model, revision=None: said)


def test_from_pretrained_takes_block_major_on_cuda(monkeypatch, tmp_path):
    _model(monkeypatch, {"model_type": "qwen2"})
    cfg = _built(monkeypatch, True, output_dir=str(tmp_path))
    assert _decided(cfg) == ("efficient", None)


def test_from_pretrained_offloads_the_tables_where_the_plan_says_they_would_not_fit(
        monkeypatch, tmp_path, caplog):
    """offload_embeddings_and_head left unset is decided as poros plan decides it, for the GPU
    the run sees: the 72B on an RTX 4090 at 512 tokens would not fit with its fp32 tables on the
    card, and does with them in host RAM; the run says why, once. The 0.5B keeps them on the
    GPU; a value you set stays as set; the CPU decides nothing."""
    from poros.cli.schemas import PlanGpu

    card = PlanGpu(name="RTX 4090", total_gib=23.99, free_gib=None, source="spec")
    q72 = dict(model_type="qwen2", tie_word_embeddings=False, vocab_size=152064,
               hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
               num_key_value_heads=8, intermediate_size=29568)
    kw = dict(output_dir=str(tmp_path), max_seq_length=512, lora_dropout=0.0)
    _model(monkeypatch, q72)
    with caplog.at_level(logging.WARNING, logger="poros.train.trainer"):
        cfg = _built(monkeypatch, True, card, "Qwen/Qwen2.5-72B", **kw)
    assert cfg.offload_embeddings_and_head is True and cfg.offload_reason.startswith(
        "the check keeps the embedding table and LM head in host RAM")
    assert "an estimate puts the run past" in cfg.offload_reason  # the cause: an estimate
    said = [r for r in caplog.records if r.getMessage().startswith("tables: host RAM: ")]
    assert len(said) == 1 and not [r for r in caplog.records
                                   if "offload_embeddings_and_head" in r.getMessage()]
    kept = _built(monkeypatch, True, card, "Qwen/Qwen2.5-72B", offload_embeddings_and_head=False,
                  **kw)
    assert kept.offload_embeddings_and_head is False and kept.offload_reason is None
    cpu = _built(monkeypatch, False, card, "Qwen/Qwen2.5-72B", **kw)
    assert cpu.offload_embeddings_and_head is None
    _model(monkeypatch, dict(q72, hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
                             num_key_value_heads=2, intermediate_size=4864, vocab_size=151936,
                             tie_word_embeddings=True))
    small = _built(monkeypatch, True, card, **kw)
    assert small.offload_embeddings_and_head is False and small.offload_reason is None


def test_from_pretrained_on_the_cpu_trains_sequentially(monkeypatch, tmp_path):
    cfg = _built(monkeypatch, False, output_dir=str(tmp_path))
    assert cfg.path is None and train_path(cfg) == "sequential"


def test_from_pretrained_says_why_it_decided_sequential(monkeypatch, tmp_path, caplog):
    """A MoE model takes block-major at any dropout; with router aux loss it trains
    sequentially, and the run says why, once."""
    _model(monkeypatch, MOE)
    with caplog.at_level(logging.INFO, logger="poros.train.trainer"):
        cfg = _built(monkeypatch, True, output_dir=str(tmp_path))
    assert _decided(cfg) == ("efficient", None) and "path: sequential" not in caplog.text
    _model(monkeypatch, {**MOE, "output_router_logits": True})
    with caplog.at_level(logging.WARNING, logger="poros.train.trainer"):
        cfg = _built(monkeypatch, True, output_dir=str(tmp_path))
    assert cfg.path == "sequential"
    said = [r for r in caplog.records if "path: sequential" in r.getMessage()]
    assert len(said) == 1 and "router aux loss" in said[0].getMessage()


@pytest.mark.parametrize("model", [{**MOE, "output_router_logits": True},
                                   {"model_type": "qwen2", "attention_dropout": 0.1}])
def test_a_block_major_path_you_set_where_it_cannot_follow_is_refused_before_the_download(
        monkeypatch, tmp_path, model):
    from poros.core.errors import PorosConfigError

    _model(monkeypatch, model)
    with pytest.raises(PorosConfigError, match="path: efficient trains block-major"):
        _built(monkeypatch, True, output_dir=str(tmp_path), path="efficient")


def test_a_config_you_pass_is_decided_the_same_way(monkeypatch, tmp_path):
    given = _cfg(output_dir=str(tmp_path), path="sequential", **QUIET)
    assert _built(monkeypatch, True, config=given) is given
    assert _decided(given) == ("sequential", None)
    fast = _cfg(output_dir=str(tmp_path), path="fast", **QUIET)
    _built(monkeypatch, True, config=fast)
    assert _decided(fast) == ("fast", None)


@pytest.mark.parametrize("kw, gives_up_the_head", [
    ({"path": "efficient"}, True),  # efficient: boundaries pin
    ({"path": "fast"}, False),
    ({"path": "sequential"}, False),
    ({}, False),  # a CPU run: sequential
])
def test_only_the_efficient_path_gives_up_the_head_s_pinned_copy(monkeypatch, kw,
                                                                 gives_up_the_head):
    """offload_embeddings_and_head pins the head's host copy unless the boundaries, on the
    efficient path, need the page-locked RAM: a sequential run keeps it."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from poros.sss import pinning
    from poros.train.trainer import PorosTrainer

    class Tables(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.head = nn.Embedding(8, 4), nn.Linear(4, 8, bias=False)
            self.requires_grad_(False)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    trainer = PorosTrainer(_cfg(offload_embeddings_and_head=True, **kw))
    trainer.model = Tables()
    asked = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pinning, "reserve", lambda nbytes: asked.append(nbytes) and False)
    trainer._offload_embeddings_and_head()
    assert (not asked) is gives_up_the_head


# -- the plan: the path a CUDA run takes, from its receipts, else the sequential loop's + more ---

QWEN32 = {"model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 64,
          "num_attention_heads": 40, "num_key_value_heads": 8, "intermediate_size": 27648}


def _plan(monkeypatch, cells=None, sized=True, free=31.0, **kw):
    import poros.plan.cells
    import poros.plan.configs
    import poros.plan.make
    from poros.cli.schemas import PlanGpu

    monkeypatch.delenv("POROS_STARTUP_SELFCHECK", raising=False)
    if cells is not None:
        monkeypatch.setattr(poros.plan.cells, "CELLS", cells)
    gpu = PlanGpu(name="NVIDIA GeForce RTX 5090", total_gib=31.84, free_gib=free, source="torch")
    # the receipts ran PEFT's modules (a default CUDA config plans the fused compute from them
    # only as an estimate: tests/test_plan.py)
    kw.setdefault("fused_compute", False)
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-32B", dataset="synthetic",
                      max_seq_length=512, lora_dropout=0.0, nf4_load_strategy="gpu", **kw)
    return poros.plan.make.make_plan(cfg, model_type="qwen2", support="validated", entry="dense",
                                gpu=gpu, model_config=QWEN32 if sized else None)


def _note(p) -> str:
    return next((w for w in p.warnings if "block-major accumulation's" in w), "")


def test_with_no_receipt_of_its_path_a_config_plans_the_sequential_loop_plus_the_path(monkeypatch):
    import poros.plan.cells
    import poros.plan.configs
    import poros.plan.make
    from poros.plan.estimate import PathExtra, path_extra

    # no receipt of either block-major path (the RTX 4090 ran both: the cells without them)
    monkeypatch.setattr(poros.plan.cells, "CELLS",
                        [c for c in poros.plan.cells.CELLS if "path" not in c["key"]])
    sequential = _plan(monkeypatch, path="sequential")
    assert sequential.path == "sequential" and sequential.peak_vram_allocated.source == "measured"
    assert sequential.fits is True and not _note(sequential)
    seq = sequential.peak_vram_allocated.value
    for kw, path in (({}, "efficient"), ({"path": "fast"}, "fast"),
                     ({"path": "efficient"}, "efficient")):
        p = _plan(monkeypatch, **kw)
        cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-32B", dataset="synthetic",
                          max_seq_length=512, lora_dropout=0.0, **kw)
        cfg.apply_cuda_defaults(QWEN32)
        extra = path_extra(cfg, QWEN32, p.gpu, "dense")
        assert isinstance(extra, PathExtra) and p.path == path
        assert p.peak_vram_allocated.source == "estimated"
        # the sequential loop's measured peak, with the 1% a peak moves between runs, plus it
        assert p.peak_vram_allocated.value == round(seq * 1.01 + extra.vram, 3)
        assert "sequential loop" in p.peak_vram_allocated.method
        assert f"{path} path" in _note(p) and extra.what in _note(p)
        # the estimate errs high, an upper bound: past the card it warns, never refuses (a
        # measured figure or a run no larger's does: poros train's E_OOM_PREDICTED)
        short = _plan(monkeypatch, free=round(seq + extra.vram / 2, 2), **kw)
        assert seq < short.gpu.free_gib < short.peak_vram_allocated.value
        assert short.fits is None and "may not fit" in short.fit_basis
        assert "estimated" in short.fit_basis and poros.plan.make.over_card(short)
    fast, efficient = _plan(monkeypatch, path="fast"), _plan(monkeypatch)
    assert fast.peak_vram_allocated.value > efficient.peak_vram_allocated.value
    assert not any("path" in w for w in _plan(monkeypatch, path="sequential").warnings)


def test_without_config_json_the_path_decides_no_fit(monkeypatch):
    """Only the sequential loop's receipts, and no config.json to size what the path holds on
    top: no memory figure (the path holds more than those), no verdict, and the plan says why."""
    import poros.plan.cells
    import poros.plan.configs
    import poros.plan.make

    sequential = [c for c in poros.plan.cells.CELLS if "path" not in c["key"]]
    p = _plan(monkeypatch, sequential, sized=False)
    assert p.peak_vram_allocated.source == "unknown" and p.fits is None
    assert "config.json" in p.peak_vram_allocated.why
    assert "config.json does not" in _note(p)


def test_a_receipt_of_a_path_is_all_it_takes_to_plan_that_path(monkeypatch):
    """A cell whose key says ``path: efficient`` (plan_sources.yaml): no code changes."""
    from poros.plan.cells import CELLS

    seq = next(c for c in CELLS if c["id"] == "qwen2.5-32b-rtx5090-synthetic")
    efficient = {**seq, "id": "efficient", "key": {**seq["key"], "path": "efficient",
                                                     "selfcheck": False},
                 "fields": {"peak_vram_allocated": {**seq["fields"]["peak_vram_allocated"],
                                                    "value": 9.0}}}
    cells = [*CELLS, efficient]
    auto = _plan(monkeypatch, cells)
    assert (auto.peak_vram_allocated.value, auto.peak_vram_allocated.source) == (9.0, "measured")
    fast = _plan(monkeypatch, cells, path="fast")  # a path of this model no receipt ran
    assert "fast path" in _note(fast) and fast.peak_vram_allocated.value != 9.0
    sequential = _plan(monkeypatch, cells, path="sequential")
    assert sequential.peak_vram_allocated.value == seq["fields"]["peak_vram_allocated"]["value"]


def test_the_train_card_plans_the_path_it_was_given(tmp_path, monkeypatch):
    from poros.cli.context import Context
    from poros.cli.schemas import PlanGpu
    from poros.cli.verbs import shared

    gpu = PlanGpu(name="NVIDIA GeForce RTX 5090", total_gib=31.84, free_gib=None, source="spec")
    ctx = Context(cwd=tmp_path, fixed_gpu=gpu, fixed_cached=frozenset(), fixed_interactive=False)

    def path(**given):
        return shared.plan_for("Qwen/Qwen2.5-32B", ctx, seq_len=512, path_settings=given).path

    assert path() == "efficient" and path(path="efficient") == "efficient"
    assert path(path="fast") == "fast"
    assert path(path="sequential") == "sequential"  # the receipts' own loop


# -- path: the one choice, in words ------------------------------------------------------------


def test_sequential_stays_reachable():
    for kw in ({"path": "sequential"}, {"lora_target_modules": TABLES}):
        cfg = _cfg(**kw)
        cfg.apply_cuda_defaults()
        assert train_path(cfg) == "sequential", kw
    one = _cfg(gradient_accumulation_steps=1)  # one micro-batch a step: block-major, no slower
    one.apply_cuda_defaults()
    assert train_path(one) == "efficient" and one.path_reason is None


@pytest.mark.parametrize("field", ["freeze_embeddings", "freeze_lm_head"])
def test_a_thawed_table_is_refused_not_silently_untrained(field):
    """freeze_embeddings / freeze_lm_head false would train nothing (the table's gradient
    reaches no optimizer): refused, naming the field, saying how to adapt a table instead."""
    from poros.core.errors import PorosConfigError

    with pytest.raises(PorosConfigError, match=f"{field}: false.*not supported") as e:
        _cfg(**{field: False})
    assert "lora_target_modules" in str(e.value)
