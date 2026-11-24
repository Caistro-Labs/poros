"""The classic loop's length with no max_steps: num_epochs passes over the data, each optimizer
step taking gradient_accumulation_steps micro-batches, and the LR schedule built on that length.
Torch on the CPU, a stub engine: rows only."""

from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402

from poros.core.config import PorosConfig  # noqa: E402
from poros.train.progress import RunHistory  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402


class _Rows(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": torch.tensor([i, i]),
                "attention_mask": torch.ones(2, dtype=torch.long)}


class _Stub(PorosTrainer):
    """The loop with its engine replaced: records the rows each micro-batch takes and the LR each
    optimizer step uses."""

    def __init__(self, cfg):
        self.config = cfg
        self.device = torch.device("cpu")
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.model = torch.nn.Linear(1, 1)
        self.trainable_parameters = [self.p]
        self.optimizer = torch.optim.AdamW([self.p], lr=cfg.learning_rate)
        self._lr_scheduler = None
        self._is_setup = True
        self._event_writer = None
        self.reporter = None
        self.keyed_rng = None
        self._witness = None
        self._sss_source = None
        self._sss_pass_shapes = []
        self.block_manager = None
        self._retained_forward_blocks = []
        self._watchdog = None
        self._block_major_ran = None
        self._block_major_off = False
        self._unreached_params = []
        self.history = RunHistory()
        self.global_step = 0
        self.seen, self.lrs = [], []

    def _reset_runtime_counters(self):
        self.history = RunHistory()

    def _sequential_pass(self, batch, g, G):
        self.seen.extend(batch["input_ids"][:, 0].tolist())
        return torch.tensor(1.0)

    def _adapter_optimizer_step(self):
        self.lrs.append(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _emit_step_event(self, *a, **k):
        pass

    def _log_step(self, *a, **k):
        pass

    def _report(self, *a, **k):
        pass

    def _compute_metrics(self):
        return {}

    def _log_final_summary(self, *a, **k):
        pass


def _run(tmp_path, rows, G, epochs=1, **kw):
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", path="sequential",
                      gradient_accumulation_steps=G, num_epochs=epochs, learning_rate=1e-4,
                      save_steps=0, output_dir=str(tmp_path / "out"), **kw)
    t = _Stub(cfg)
    t.train(_Rows(rows))
    return t


@pytest.mark.parametrize("rows, G, epochs", [(40, 8, 1), (41, 8, 1), (40, 8, 2), (12, 1, 3)])
def test_an_epoch_is_one_pass_over_the_rows(tmp_path, rows, G, epochs):
    t = _run(tmp_path, rows, G, epochs)
    assert len(t.lrs) == -(-rows * epochs // G)  # ceil(batches x epochs / G) optimizer steps
    seen = Counter(t.seen)
    assert set(seen) == set(range(rows)) and min(seen.values()) >= epochs
    assert max(seen.values()) <= epochs + 1  # a last step may take the next pass's first rows


def test_the_schedule_ends_where_the_run_does(tmp_path):
    """Cosine over the run's own length: it falls to ~0 at the last step, never wraps."""
    t = _run(tmp_path, 4000, 1, warmup_steps=100, lr_scheduler="cosine")
    lr = t.lrs
    assert len(lr) == 4000 and lr[-1] < 1e-8
    assert all(b <= a + 1e-12 for a, b in zip(lr[100:], lr[101:]))  # past warmup, never rises


def test_a_set_max_steps_is_the_length(tmp_path):
    t = _run(tmp_path, 40, 8, max_steps=3)
    assert len(t.lrs) == 3 and len(t.seen) == 24
