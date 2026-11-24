"""A run's screen holds no dependency's deprecation notice Poros's own calls cause, and none of
Poros's internal diagnostics: the current API names where Poros calls transformers, a narrow
filter around the one call PEFT makes by an old name, and the diagnostics in the debug log."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import needs_ml

GC_NOTICE = ("`gradient_checkpointing_enable` is deprecated and will be removed in v6. Use "
             "`activation_checkpointing_enable` instead.")


def test_a_load_names_its_dtype_as_this_transformers_does():
    pytest.importorskip("torch")
    from poros.train.streamed_load import dtype_argument

    assert dtype_argument(SimpleNamespace(dtype="bfloat16")) == "dtype"
    assert dtype_argument(SimpleNamespace()) == "torch_dtype"  # a transformers before `dtype`


@needs_ml
def test_pefts_deprecated_call_is_not_this_runs_users_to_read():
    """transformers 5.23 says `gradient_checkpointing_enable is deprecated` (a FutureWarning with
    a path into PEFT) when PEFT prepares a 4-bit model. PEFT makes that call, twice a run."""
    pytest.importorskip("peft")
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    from poros.train.model import _prepare_model_for_kbit_training

    model = AutoModelForCausalLM.from_config(Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32),
        dtype=torch.float32)
    model.is_loaded_in_4bit = True  # as an NF4 load is: PEFT then turns checkpointing on
    called = []

    def enable(gradient_checkpointing_kwargs=None):  # as 5.23's alias: it warns, then works
        warnings.warn(GC_NOTICE, FutureWarning)
        warnings.warn("something a user should read", UserWarning)
        called.append(gradient_checkpointing_kwargs)

    model.gradient_checkpointing_enable = enable
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _prepare_model_for_kbit_training(model, [])
    assert called == [{"use_reentrant": False}]  # the call itself is as it was
    said = [str(w.message) for w in seen]
    assert GC_NOTICE not in said
    assert "something a user should read" in said  # only that one notice is filtered


def test_pefts_read_of_the_deprecated_flag_is_filtered_and_nothing_else():
    """transformers 5.23 warns when PEFT's get_peft_model reads `is_gradient_checkpointing`."""
    pytest.importorskip("torch")
    from poros.train.model import pefts_checkpointing_names

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pefts_checkpointing_names():
            warnings.warn("`is_gradient_checkpointing` is deprecated and will be removed in v6. "
                          "Use `is_activation_checkpointing` instead.", FutureWarning)
            warnings.warn("`something_else` is deprecated and will be removed in v6.",
                          FutureWarning)
        warnings.warn(GC_NOTICE, FutureWarning)  # outside Poros's call: a user's own to read
    assert [str(w.message)[:18] for w in seen] == ["`something_else` i", "`gradient_checkpoi"]


def test_the_patch_turns_checkpointing_off_by_the_current_name():
    from poros.patch.hf import _neutralize_gradient_checkpointing

    calls = []

    class Model:
        def activation_checkpointing_disable(self):
            calls.append("activation")

        def gradient_checkpointing_disable(self):  # 5.23's deprecated alias
            warnings.warn("`gradient_checkpointing_disable` is deprecated", FutureWarning)
            calls.append("gradient")

    class Older:
        def gradient_checkpointing_disable(self):
            calls.append("older")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _neutralize_gradient_checkpointing(Model())
    _neutralize_gradient_checkpointing(Older())  # a transformers with the old name only
    assert calls == ["activation", "older"]


def test_internal_diagnostics_are_logged_for_debugging_not_shown():
    """`path efficient after update 1: block inputs {...}; pin budget {...}` was a warning, so
    Python printed it on every run's screen. It is the debug log's, and the receipt's."""
    source = (Path(__file__).resolve().parents[2] / "poros/train/block_major.py").read_text()
    for said in ("path efficient after update 1", "expert dequant cache after update 1"):
        at = source.index(said)
        call = source[source.rindex("logger.", 0, at):at]
        assert call.startswith("logger.info("), call
    # an info record reaches no screen unless a user configures logging to show it: Python's
    # handler of last resort prints warnings and above
    assert logging.lastResort.level == logging.WARNING


def test_the_receipt_keeps_where_the_block_inputs_went(monkeypatch):
    torch = pytest.importorskip("torch")
    from poros.cli.train_cmd import _staging_receipt
    from poros.sss import pinning

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pinning, "staging", lambda: {"mode": "pinned"})
    trainer = SimpleNamespace(_boundary_offload_stats={"offloaded": 40, "slabs": 1})
    assert _staging_receipt(trainer) == {"host_staging": {"mode": "pinned"},
                                         "block_inputs": {"offloaded": 40, "slabs": 1}}
    assert _staging_receipt(SimpleNamespace()) == {"host_staging": {"mode": "pinned"}}


def test_the_hubs_unauthenticated_notice_is_filtered_by_its_words_alone(caplog):
    """Every run printed "Warning: You are sending unauthenticated requests to the HF Hub": the
    Hub sends it with each answer to a request with no token, and Poros asks with none on
    purpose. Dropped on the one logger that says it; anything else that logger says passes."""
    from poros.core.env import quiet_hub_notice

    quiet_hub_notice()
    quiet_hub_notice()  # once, however often it is asked for
    log = logging.getLogger("huggingface_hub.utils._http")
    assert sum(type(f).__name__ == "_NoHubNotice" for f in log.filters) == 1
    with caplog.at_level(logging.WARNING, logger="huggingface_hub.utils._http"):
        log.warning("Warning: You are sending unauthenticated requests to the HF Hub. Please "
                    "set a HF_TOKEN to enable higher rate limits and faster downloads.")
        log.warning("429 Too Many Requests: you are being rate limited")
    said = [r.getMessage() for r in caplog.records]
    assert said == ["429 Too Many Requests: you are being rate limited"]
