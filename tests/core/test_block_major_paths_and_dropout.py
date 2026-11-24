"""Block-major draws the sequential loop's dropout masks: a dropout does not refuse a block-major
path. What it cannot train as that loop does still is refused."""
from __future__ import annotations

import pytest


class TestBlockMajorDropout:
    """Block-major draws the sequential loop's dropout masks (tests/test_block_major_dropout.py):
    a dropout does not refuse it. What it cannot train as that loop does still does."""

    @pytest.mark.parametrize("dropout", [0.0, 0.05])
    def test_block_major_takes_any_dropout(self, dropout):
        pytest.importorskip("torch")
        from poros.core.config import PorosConfig

        cfg = PorosConfig(path="fast", lora_dropout=dropout)
        assert (cfg.path, cfg.lora_dropout) == ("fast", dropout)

    def test_dropout_without_preserve_rng_state_is_refused(self):
        pytest.importorskip("torch")
        from poros.core.config import PorosConfig

        with pytest.raises(ValueError, match="preserve_rng_state"):
            PorosConfig(path="efficient", preserve_rng_state=False)
