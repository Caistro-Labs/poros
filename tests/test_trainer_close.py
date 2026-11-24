"""PorosTrainer.close() (and its `with`) lets go of everything a run holds, at once and safely
twice: the model and every list of its parameters, the LM head's copies,
and an SSD store's reader threads, file descriptors and ring."""

from __future__ import annotations

import gc
import os
import tempfile
import threading
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tests.test_train_loop_cpu import _loader, _make_trainer, _tiny_peft_qwen2  # noqa: E402


def _fds() -> int:
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0


def test_a_closed_trainer_holds_no_model_parameter_or_head_copy():
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        with _make_trainer(torch, model, d, max_steps=1) as tr:
            tr.train(_loader(torch, cfg.vocab_size), steps=1)
            refs = [weakref.ref(tr.model), weakref.ref(tr.model.get_input_embeddings().weight),
                    weakref.ref(tr.optimizer), *(weakref.ref(p) for p in tr.trainable_parameters)]
            head = tr.head
        del model
        gc.collect()
        assert [r() is None for r in refs] == [True] * len(refs)
        assert tr.model is None and tr.trainable_parameters == [] and tr.head is None
        assert all(getattr(head, n, None) is None for n in ("_pinned", "_eval_weight"))
        tr.close()  # twice: nothing left to free, nothing raised


@pytest.mark.cuda_host  # the SSD store is Linux's (os.posix_fadvise): a Mac keeps its own pack
def test_close_closes_an_ssd_store_its_threads_and_descriptors(tmp_path):
    from poros.sss.ssd import SsdSource
    from poros.train.trainer import PorosTrainer
    from tests.sss.test_ssd import _moe_sizes, _read, _slabs, _write

    slabs = _slabs(_moe_sizes(1, 3))
    path = _write(tmp_path, slabs)
    threads, fds = threading.active_count(), _fds()
    src = SsdSource(path, chunk_bytes=64 << 10, ram_budget=1 << 20)
    for key in slabs:
        _read(src, key)
    assert threading.active_count() > threads or _fds() > fds
    tr = PorosTrainer.__new__(PorosTrainer)
    tr._sss_source, tr.model = src, None
    tr.close()
    assert src._closed and tr._sss_source is None
    assert threading.active_count() == threads and _fds() == fds
    tr.close()
