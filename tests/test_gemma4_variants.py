"""Gemma 4 E2B and E4B are refused before the download, with the reason (no torch, no GPU).

Their decoder layers take a per-layer input and share KV caches across layers
(``hidden_size_per_layer_input``, ``num_kv_shared_layers``), which the block forward does not pass,
so neither loop trains them. The registry calls them ``gemma4_per_layer``, unsupported:
``poros check`` says so with the reason, and the trainer's architecture gate refuses them in every
``blockwise_arch_check`` mode.
"""
from __future__ import annotations

import json

import pytest

from poros.core.arch_registry import arch_status, blockwise_arch_check, registry_type
from poros.core.errors import PorosArchNotValidatedError

E2B = {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text",
                                                "hidden_size_per_layer_input": 256,
                                                "num_kv_shared_layers": 20}}


@pytest.mark.parametrize("text, variant", [
    ({"hidden_size_per_layer_input": 256}, True),
    ({"num_kv_shared_layers": 4}, True),
    ({"hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0}, False),  # 31B, 26B-A4B dense
    ({}, False),
])
def test_the_registry_names_the_variant(text, variant):
    for raw in ({"model_type": "gemma4", "text_config": text}, {"model_type": "gemma4_text",
                                                                 **text}):
        assert registry_type(raw) == ("gemma4_per_layer" if variant else raw["model_type"])


@pytest.mark.parametrize("mode", ["error", "warn", "off"])
def test_the_gate_refuses_it_in_every_mode(mode):
    assert arch_status("gemma4_per_layer") == "unsupported"
    with pytest.raises(PorosArchNotValidatedError, match="Unsupported: Gemma 4 E2B and E4B"):
        blockwise_arch_check("gemma4_per_layer", mode=mode)


def test_poros_check_says_unsupported_with_the_reason(tmp_path):
    from typer.testing import CliRunner

    from poros.cli.main import app

    (tmp_path / "config.json").write_text(json.dumps(E2B))
    res = CliRunner().invoke(app, ["check", str(tmp_path), "--json"])
    said = json.loads(res.output)
    assert res.exit_code != 0
    assert (said["status"], said["model_type"]) == ("unsupported", "gemma4_per_layer")
    assert "per-layer input" in said["reason"] and "neither loop" in said["reason"]
