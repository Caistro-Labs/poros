"""cuda_expandable_segments: an allocator the environment configures is left alone, a GPU that
cannot map expandable segments keeps the default allocator, and a process that already holds CUDA
memory keeps its allocator as it is; in each case the config, and so a run's records, then say
what the allocator runs with (CPU: the CUDA calls are faked)."""
from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from poros.core.config import PorosConfig, recorded_config  # noqa: E402
from poros.train import trainer as T  # noqa: E402

CUDA = torch.device("cuda")


@pytest.fixture
def calls(monkeypatch):
    seen = []
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(T, "_EXPANDABLE_SET", None)  # a process that has set nothing yet
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T, "_set_allocator_settings", seen.append)
    monkeypatch.setattr(T.torch.cuda, "empty_cache", lambda: seen.append("empty_cache"))
    monkeypatch.setattr(T.torch.cuda, "memory_reserved", lambda device=None: 0)
    return seen


def _ran(cfg: PorosConfig) -> bool:
    """What both of a run's records say of it (poros_config.json, config.resolved.json)."""
    said = {recorded_config(cfg)["cuda_expandable_segments"],
            recorded_config(cfg, mode="json")["cuda_expandable_segments"]}
    assert len(said) == 1
    return said.pop()


@pytest.mark.parametrize("name", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
@pytest.mark.parametrize("conf, on", [("expandable_segments:False", False),
                                      ("expandable_segments:True", True),
                                      ("max_split_size_mb:128", False)])
@pytest.mark.parametrize("wanted", [True, False])
def test_an_allocator_the_environment_configures_is_left_alone_and_recorded(
        calls, monkeypatch, name, conf, on, wanted):
    monkeypatch.setenv(name, conf)
    cfg = PorosConfig(cuda_expandable_segments=wanted)
    T.apply_expandable_segments(cfg, CUDA)
    assert calls == [] and _ran(cfg) is on


def test_torch_reads_the_older_name_first(calls, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    cfg = PorosConfig()
    T.apply_expandable_segments(cfg, CUDA)
    assert calls == [] and _ran(cfg) is False


def test_where_they_map_they_stay_on(calls, monkeypatch):
    monkeypatch.setattr(T.torch, "empty", lambda *a, **k: None)
    cfg = PorosConfig()
    T.apply_expandable_segments(cfg, CUDA)
    assert calls == ["empty_cache", "expandable_segments:True", "empty_cache"]
    assert _ran(cfg) is True
    again = PorosConfig()  # a second run in this process: what was set stays, unchecked
    T.apply_expandable_segments(again, CUDA)
    assert len(calls) == 3 and _ran(again) is True


def test_off_cuda_nothing_is_set_or_recorded_differently(calls):
    cfg = PorosConfig()  # the Mac and the CPU record the setting as they always have
    T.apply_expandable_segments(cfg, T.torch.device("cpu"))
    assert calls == [] and _ran(cfg) is True


def test_a_gpu_that_cannot_map_them_keeps_the_default_allocator(calls, monkeypatch, caplog):
    def refused(*a, **k):  # what a driver without CUDA VMM says at the first allocation
        raise RuntimeError("CUDA driver error: operation not supported")

    monkeypatch.setattr(T.torch, "empty", refused)
    cfg = PorosConfig()
    with caplog.at_level(logging.WARNING, logger=T.__name__):
        T.apply_expandable_segments(cfg, CUDA)
    assert calls == ["empty_cache", "expandable_segments:True", "expandable_segments:False",
                     "empty_cache"]
    assert "operation not supported" in caplog.text
    assert _ran(cfg) is False  # the record says the default allocator ran


def test_a_check_that_cannot_allocate_at_all_keeps_the_default_allocator(calls, monkeypatch,
                                                                         caplog):
    """Not only a RuntimeError: a torch built without CUDA raises AssertionError at the first
    CUDA allocation. The check could not be made, so the default allocator stays; no crash."""
    def no_cuda(*a, **k):
        raise AssertionError("Torch not compiled with CUDA enabled")

    monkeypatch.setattr(T.torch, "empty", no_cuda)
    cfg = PorosConfig()
    with caplog.at_level(logging.WARNING, logger=T.__name__):
        T.apply_expandable_segments(cfg, CUDA)
    assert calls == ["empty_cache", "expandable_segments:True", "expandable_segments:False",
                     "empty_cache"]
    assert "AssertionError: Torch not compiled with CUDA enabled" in caplog.text
    assert _ran(cfg) is False


def test_a_process_that_holds_cuda_memory_keeps_its_allocator(calls, monkeypatch):
    """The check's allocation could come from a segment the process already holds (a second
    trainer, or a script's own tensors), so it would prove nothing: the allocator stays as it is."""
    monkeypatch.setattr(T.torch.cuda, "memory_reserved", lambda device=None: 2 << 20)
    monkeypatch.setattr(T.torch, "empty", lambda *a, **k: pytest.fail("no check is made"))
    cfg = PorosConfig()
    T.apply_expandable_segments(cfg, CUDA)
    assert calls == ["empty_cache"] and _ran(cfg) is False


def test_poros_train_records_the_allocator_the_environment_chose(calls, monkeypatch, tmp_path,
                                                                 recipe_configs_cached):
    """config.resolved.json is written before the model loads: the allocator is set up first, so
    the record says what runs (here the environment's choice, over the config's default)."""
    import json

    from typer.testing import CliRunner

    from poros.cli import main
    from poros.cli.schemas import PlanGpu
    from poros.preflight import arch as preflight_arch

    big = PlanGpu(name="NVIDIA RTX PRO 6000", total_gib=96.0, free_gib=90.0, source="torch")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.setattr("poros.core.card.detect_gpu", lambda name=None: big)
    # the card poros.core.card.check reads (the plan's RTX PRO 6000) and a torch built for it
    monkeypatch.setattr(T.torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(T.torch.cuda, "get_device_name", lambda device=None: big.name)
    monkeypatch.setattr(T.torch.cuda, "get_arch_list", lambda: ["sm_90", "sm_120"])
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen2", ""))
    monkeypatch.setattr(T.PorosTrainer, "from_pretrained",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not loaded here")))
    cfg = tmp_path / "run.yaml"
    cfg.write_text("model_name_or_path: Qwen/Qwen2.5-0.5B\ndataset: synthetic\n"
                   f"max_seq_length: 512\noutput_dir: {tmp_path / 'out'}\n")
    CliRunner().invoke(main.app, ["train", str(cfg), "--json", "--force"])
    ran = json.loads((tmp_path / "out" / "config.resolved.json").read_text())
    assert ran["cuda_expandable_segments"] is False and calls == []


def test_the_setting_reaches_torch_without_its_deprecation_warning(monkeypatch):
    """torch.cuda.memory._set_allocator_settings is deprecated (a FutureWarning) and only calls
    torch._C._accelerator_setAllocatorSettings: Poros calls that one, with the same argument."""
    import warnings

    seen = []
    monkeypatch.setattr(T.torch._C, "_accelerator_setAllocatorSettings", seen.append,
                        raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        T._set_allocator_settings("expandable_segments:True")
    assert seen == ["expandable_segments:True"]
