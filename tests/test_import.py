"""Test that poros imports cleanly without torch."""

from __future__ import annotations


def test_version():
    """Import poros and check version is a valid version string."""
    import re

    import poros

    assert isinstance(poros.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", poros.__version__)


def test_pure_python_imports():
    """Pure-python symbols import without torch."""
    from poros import (
        PorosArchNotValidatedError,
        PorosConfigError,
        PorosError,
        PorosOOMError,
    )

    # Verify they are the right types
    assert issubclass(PorosError, Exception)
    assert issubclass(PorosArchNotValidatedError, PorosError)
    assert issubclass(PorosOOMError, PorosError)
    assert issubclass(PorosConfigError, PorosError)


def test_config_instantiation():
    """PorosConfig can be instantiated with defaults."""
    from poros import PorosConfig

    cfg = PorosConfig()
    assert cfg.quantization == "nf4"
    assert cfg.lora_rank == 16
    assert cfg.blockwise_arch_check == "error"


def test_benchmark_spec():
    """BenchmarkSpec can be instantiated."""
    from poros.bench.specs import BenchmarkSpec, HardwareSpec, ModelSpec

    spec = BenchmarkSpec(
        spec_id="test",
        hardware=HardwareSpec(gpu_name="RTX 3090", vram_gb=24.0),
        models=[ModelSpec(model_id="test/model", display_name="Test")],
        methods=["poros", "resident"],
    )
    assert spec.spec_id == "test"
    assert len(spec.models) == 1


def test_events():
    """PorosEvent can be instantiated."""
    from poros.core.events import PorosEvent

    event = PorosEvent(event_type="step", step=1, data={"loss": 0.5})
    assert event.event_type == "step"
    assert event.data["loss"] == 0.5


def test_errors_structured():
    """Errors carry structured info."""
    from poros.core.errors import PorosArchNotValidatedError

    try:
        raise PorosArchNotValidatedError("gemma", {"qwen3", "qwen2"})
    except PorosArchNotValidatedError as e:
        assert "gemma" in str(e)
        assert e.info.error_type == "PorosArchNotValidatedError"
        assert e.info.remediation != ""
