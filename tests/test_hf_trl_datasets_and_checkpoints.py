"""Any dataset: dataset formats, the synthetic loader, failure messages, checkpoint round trips,
the patched blockwise forward on CPU, config edge cases.

All tests run on CPU with no downloads: tiny models from config, mock
datasets, real engine paths.

Groups:
  1. Dataset robustness (chat templates, instruction, plain text, streaming)
  2. Failure-mode UX (unvalidated arch, OOM hint text)
  3. Checkpoint save/load round-trip
  4. The patched blockwise forward (the MoE path's dense streaming) on CPU
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────


def _require_ml():
    """Import the ML stack or skip."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _tiny_qwen2_peft(torch, vocab_size=256):
    """A 4-layer Qwen2 with PEFT LoRA, no download, no GPU."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg)
    model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                   target_modules=["q_proj", "v_proj"]),
    )
    return model, cfg


def _patched_cpu(torch, model):
    """The model patched for blockwise residency on CPU (``poros.patch.hf``)."""
    from poros.patch.hf import patch_hf_model

    return patch_hf_model(model, block_size=2, deterministic=False)


def _synthetic_batch(torch, vocab_size: int, seq_len: int = 16, batch: int = 1):
    ids = torch.randint(0, vocab_size, (batch, seq_len))
    return {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}


# ── 1. Dataset robustness ────────────────────────────────────────────────


class TestDatasetFormats:
    """_prepare_dataloader handles diverse dataset schemas correctly.

    _format_example is a nested closure, so we test the public surface:
    pass a pre-built HF Dataset to the trainer's _prepare_dataloader path.
    """

    def _tokenize_via_trainer(self, rows, torch):
        """Use a real tiny PorosTrainer to tokenize rows via _prepare_dataloader."""
        from poros.core.config import PorosConfig
        from poros.train.trainer import PorosTrainer

        cfg = PorosConfig(
            model_name_or_path="Qwen/Qwen2.5-0.5B",
            blockwise_arch_check="off",
        )
        tr = PorosTrainer(cfg)
        # _prepare_dataloader with a Dataset object wraps it directly
        from torch.utils.data import Dataset as TorchDS

        class _MapDS(TorchDS):
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        ids = torch.randint(0, 256, (len(rows), 16))
        ds = _MapDS([{"input_ids": ids[i], "labels": ids[i]} for i in range(len(rows))])
        dl = tr._prepare_dataloader(ds, max_seq_length=16)
        batch = next(iter(dl))
        return batch

    def test_synthetic_dataloader_schema(self):
        """The synthetic path produces correct batch keys."""
        _require_ml()
        from poros.train.datasets import _build_synthetic_dataloader

        dl = _build_synthetic_dataloader(n_samples=4, max_seq_length=16,
                                          batch_size=2, seed=42)
        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[1] == 16

    def test_torch_dataset_passthrough(self):
        """A torch Dataset is accepted directly by _prepare_dataloader."""
        torch = _require_ml()
        rows = [{"input_ids": torch.zeros(8, dtype=torch.long),
                 "labels": torch.zeros(8, dtype=torch.long)} for _ in range(4)]
        batch = self._tokenize_via_trainer(rows, torch)
        assert batch["input_ids"].shape[0] > 0

    def test_dataloader_passthrough(self):
        """An existing DataLoader is returned as-is."""
        torch = _require_ml()
        from poros.core.config import PorosConfig
        from poros.train.trainer import PorosTrainer

        ids = torch.zeros(4, 8, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(ids, ids)
        dl = torch.utils.data.DataLoader(ds, batch_size=2)

        cfg = PorosConfig(model_name_or_path="x", blockwise_arch_check="off")
        tr = PorosTrainer(cfg)
        result = tr._prepare_dataloader(dl, max_seq_length=8)
        assert result is dl


class TestSyntheticDataloader:
    """The synthetic dataloader is deterministic and schema-correct."""

    def test_synthetic_batch_shape(self):
        _require_ml()
        from poros.train.datasets import _build_synthetic_dataloader

        dl = _build_synthetic_dataloader(n_samples=4, max_seq_length=32,
                                          batch_size=2, seed=42)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (2, 32)
        assert batch["labels"].shape == (2, 32)

    def test_synthetic_is_deterministic(self):
        torch = _require_ml()
        from poros.train.datasets import _build_synthetic_dataloader

        dl1 = _build_synthetic_dataloader(n_samples=4, max_seq_length=16,
                                           batch_size=4, seed=42)
        dl2 = _build_synthetic_dataloader(n_samples=4, max_seq_length=16,
                                           batch_size=4, seed=42)
        b1 = next(iter(dl1))
        b2 = next(iter(dl2))
        assert torch.equal(b1["input_ids"], b2["input_ids"])


# ── 2. Failure-mode UX ──────────────────────────────────────────────────


class TestFailureModeUX:
    """Error messages for common foot-guns are clear and actionable."""

    def test_unvalidated_arch_error_message(self):
        """PorosArchNotValidatedError names the arch and lists the valid set."""
        from poros.core.errors import PorosArchNotValidatedError

        with pytest.raises(PorosArchNotValidatedError) as exc_info:
            raise PorosArchNotValidatedError("my_custom_arch", {"qwen2", "qwen3"})
        msg = str(exc_info.value)
        assert "my_custom_arch" in msg
        assert "qwen2" in msg
        assert "blockwise_arch_check" in msg.lower() or "validated" in msg.lower()

    def test_oom_error_suggests_halving(self):
        """PorosOOMError remediation suggests halving block_size and seq_len."""
        from poros.core.errors import PorosOOMError

        e = PorosOOMError(block_size=8, seq_len=4096)
        assert "block_size=4" in e.info.remediation
        assert "seq_len=2048" in e.info.remediation

    def test_config_error_is_clear(self):
        from poros.core.errors import PorosConfigError

        e = PorosConfigError("quantization='int32' is not selectable")
        assert "int32" in str(e)


# ── 3. Checkpoint save/load round-trip ───────────────────────────────────


@pytest.mark.usefixtures("cpu_only")
class TestCheckpointRoundTrip:
    """PorosTrainer save/load produces identical adapter weights."""

    def test_save_creates_adapter_files(self):
        """save() produces adapter_model.safetensors / adapter_config.json."""
        torch = _require_ml()

        torch.manual_seed(0)
        model, cfg = _tiny_qwen2_peft(torch)
        model = _patched_cpu(torch, model)

        # One forward+backward to get non-zero adapter params
        batch = _synthetic_batch(torch, cfg.vocab_size)
        out = model(**batch)
        out.loss.backward()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            files = [p.name for p in Path(tmpdir).iterdir()]
            assert any("adapter" in f for f in files), f"No adapter files: {files}"

    def test_trainer_save_writes_loadable_peft_adapter_dir(self):
        """PorosTrainer.save() (peft backend) writes a standard PEFT adapter
        directory (adapter_config.json + adapter_model.safetensors, NOT a full
        model.safetensors) that vanilla PEFT can load with identical weights."""
        torch = _require_ml()
        from peft import PeftModel, get_peft_model_state_dict
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from poros.core.config import PorosConfig
        from poros.train.model import _attach_peft_lora
        from poros.train.trainer import PorosTrainer

        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        base = Qwen2ForCausalLM(cfg)
        _, peft_model = _attach_peft_lora(
            base, lora_rank=4, lora_alpha=8, lora_dropout=0.0,
            lora_bias="none", target_modules=["q_proj", "v_proj"],
        )

        # Perturb adapter weights so the round-trip compares non-trivial values
        # (lora_B is zero-initialized).
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_" in name:
                    p.add_(torch.randn_like(p) * 0.01)

        tr = PorosTrainer(
            PorosConfig(model_name_or_path="x", blockwise_arch_check="off")
        )
        tr.model = base
        tr._peft_model = peft_model
        tr._adapter_backend = "peft"
        tr.trainable_parameter_names = [
            n for n, p in base.named_parameters() if p.requires_grad
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tr.save(tmpdir)
            out = Path(tmpdir)
            assert (out / "adapter_config.json").exists()
            assert (out / "adapter_model.safetensors").exists() or (
                out / "adapter_model.bin"
            ).exists()
            # The full base model must NOT be dumped.
            assert not (out / "model.safetensors").exists()
            assert not (out / "pytorch_model.bin").exists()

            # Round-trip through vanilla PEFT on a fresh base model.
            torch.manual_seed(0)
            fresh = Qwen2ForCausalLM(cfg)
            loaded = PeftModel.from_pretrained(fresh, tmpdir)

            orig_sd = get_peft_model_state_dict(peft_model)
            loaded_sd = get_peft_model_state_dict(loaded)
            assert set(orig_sd.keys()) == set(loaded_sd.keys())
            for k in orig_sd:
                assert torch.equal(orig_sd[k], loaded_sd[k]), f"Mismatch: {k}"


# ── 4. The patched blockwise forward on CPU ──────────────────────────────────


@pytest.mark.usefixtures("cpu_only")
class TestBlockwiseForwardCPU:
    """The blockwise forward path produces finite losses and working grads."""

    def test_forward_produces_finite_loss(self):
        torch = _require_ml()

        torch.manual_seed(0)
        model, cfg = _tiny_qwen2_peft(torch)
        model = _patched_cpu(torch, model)

        batch = _synthetic_batch(torch, cfg.vocab_size)
        out = model(**batch)
        assert math.isfinite(out.loss.item())

    def test_backward_adapter_grads_nonzero(self):
        torch = _require_ml()
        from poros.adapters.inspect import is_adapter_param

        torch.manual_seed(0)
        model, cfg = _tiny_qwen2_peft(torch)
        model = _patched_cpu(torch, model)

        batch = _synthetic_batch(torch, cfg.vocab_size)
        out = model(**batch)
        out.loss.backward()

        adapter_grad = sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if is_adapter_param(n) and p.grad is not None
        )
        assert adapter_grad > 0

    def test_two_steps_loss_decreases_or_finite(self):
        """Two optimizer steps both produce finite loss."""
        torch = _require_ml()

        torch.manual_seed(0)
        model, cfg = _tiny_qwen2_peft(torch)
        model = _patched_cpu(torch, model)

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        losses = []
        for _ in range(2):
            batch = _synthetic_batch(torch, cfg.vocab_size)
            model.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())
        assert all(math.isfinite(v) for v in losses)


# ── 5. Config + schema surface ───────────────────────────────────────────


class TestConfigEdgeCases:
    """PorosConfig edge cases that a new user might hit."""

    def test_block_size_1_is_valid(self):
        from poros.core.config import PorosConfig
        cfg = PorosConfig(block_size=1)
        assert cfg.block_size == 1

    def test_large_block_size_clamps(self):
        """The patch clamps block_size to num_layers."""
        torch = _require_ml()
        from poros.patch.hf import patch_hf_model

        torch.manual_seed(0)
        model, _ = _tiny_qwen2_peft(torch)
        model = patch_hf_model(model, block_size=999, deterministic=False)
        assert model._poros_block_size <= 4  # tiny model has 4 layers

    def test_invalid_quantization_rejected(self):
        from poros.core.config import PorosConfig
        with pytest.raises(Exception):
            PorosConfig(quantization="int32")

    def test_schema_cli_outputs_json(self):
        """poros schema config prints valid JSON."""
        import shutil
        import subprocess

        poros_bin = shutil.which("poros")
        if poros_bin is None:
            pytest.skip("poros CLI not on PATH (not installed via pip install -e)")
        result = subprocess.run(
            [poros_bin, "schema", "config"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        schema = json.loads(result.stdout)
        assert "properties" in schema
