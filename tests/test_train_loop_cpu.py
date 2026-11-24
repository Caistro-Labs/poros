"""CPU end-to-end coverage for the headline ``PorosTrainer.train()`` surface.

These exercise the full ``train()`` orchestration on CPU with a tiny model (no
GPU, no network download): ``train()`` itself, the resume path
(``save`` -> ``load_checkpoint``) and gradient-accumulation bookkeeping, end to end.

The manual-attach wiring builds a tiny PEFT-wrapped Qwen2, sets
``model``/``device``/``_adapter_backend`` and calls ``_setup_blockwise_training``,
so no bitsandbytes/CUDA is required (the model is the trainer's to stream as it is). The
dataset is a small in-vocabulary, shuffle-free ``DataLoader`` (passed through
as-is), which avoids the synthetic generator's out-of-vocab token IDs.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from tests.conftest import needs_ml


def _require_ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _tiny_peft_qwen2(torch, targets=("q_proj", "v_proj"), dropout=0.0):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = get_peft_model(
        Qwen2ForCausalLM(cfg),
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=dropout,
            target_modules=list(targets),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model, cfg


def _loader(torch, vocab, *, n=32, seq=16, seed=0):
    """Small, in-vocab, shuffle-free dataloader (returned as-is by the trainer)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, seq), generator=g)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            x = ids[i]
            return {"input_ids": x, "labels": x.clone()}

    return torch.utils.data.DataLoader(
        _DS(), batch_size=1, shuffle=False, drop_last=True
    )


def _make_trainer(torch, model, outdir, *, grad_accum=1, max_steps=5):
    from poros.core.config import PorosConfig
    from poros.train.trainer import PorosTrainer

    cfg = PorosConfig(
        model_name_or_path="dummy",
        block_size=2,
        deterministic=True,
        output_dir=outdir,
        blockwise_arch_check="off",
        enable_prefetch=False,
        per_device_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        warmup_steps=0,
        lr_scheduler="cosine",
        save_steps=0,
        eval_steps=0,
        logging_steps=1,
        lora_dropout=0.0,
    )
    tr = PorosTrainer(cfg)
    tr.model = model
    tr.device = torch.device("cpu")
    tr._adapter_backend = "peft"
    tr._setup_blockwise_training()
    return tr


def test_train_loop_runs_end_to_end_cpu():
    """The full train() loop runs on CPU and reports correct, finite metrics."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d)
        metrics = tr.train(_loader(torch, cfg.vocab_size), steps=5, seq_len=16)
        events = [json.loads(line) for line in open(os.path.join(d, "events.jsonl"))]

    setup = next(e for e in events if e["event_type"] == "setup")
    assert setup["data"]["path"] == "sequential"  # the loop that ran, for whoever parses events
    assert metrics["total_steps"] == 5
    assert math.isfinite(metrics["final_loss"]) and metrics["final_loss"] > 0
    assert len(metrics["loss_history"]) == 5
    assert metrics["device"] == "cpu"
    assert metrics["tokens_processed"] > 0
    assert metrics["torch_num_threads"] == torch.get_num_threads()


def test_resume_restores_step_and_optimizer_cpu():
    """save -> load_checkpoint restores the step counter and optimizer state,
    and the resumed loop continues without error."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=5)
        tr.train(_loader(torch, cfg.vocab_size), steps=5)
        saved_step = tr.global_step
        tr.save(ckpt)

        out = Path(ckpt)
        assert (out / "optimizer_state.pt").exists()
        assert (out / "rng_state.pt").exists()
        assert (out / "adapter_config.json").exists() or (
            out / "adapter_weights.pt"
        ).exists()

        # Fresh trainer + fresh model: load_checkpoint must restore the state.
        torch.manual_seed(0)
        model2, _ = _tiny_peft_qwen2(torch)
        tr2 = _make_trainer(torch, model2, d, max_steps=10)
        assert tr2.global_step == 0
        tr2.load_checkpoint(ckpt)
        assert saved_step > 0 and tr2.global_step == saved_step
        assert len(tr2.optimizer.state_dict()["state"]) > 0

        # The resumed run continues the loop and stays finite.
        metrics = tr2.train(_loader(torch, cfg.vocab_size), steps=10, resume_from=ckpt)
        assert math.isfinite(metrics["final_loss"]) and metrics["final_loss"] > 0


def test_grad_accum_run_is_deterministic_cpu():
    """Two identical gradient-accumulation runs produce an identical loss
    history -- the determinism contract holds through the full train() loop."""
    torch = _require_ml()

    def run():
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        with tempfile.TemporaryDirectory() as d:
            tr = _make_trainer(torch, model, d, grad_accum=2, max_steps=3)
            return tr.train(_loader(torch, cfg.vocab_size, seed=0), steps=3)[
                "loss_history"
            ]

    assert run() == run()


def test_checkpoint_saves_are_atomic_and_marked_cpu():
    """The staging/promote sequence of the in-loop save: real checkpoints
    land at their final names with the completion sentinel written last, and
    no staging (.tmp) or backup (.old) sibling survives a clean run."""
    torch = _require_ml()
    from poros.core import checkpoints

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=5)
        tr.config.save_steps = 2
        tr.train(_loader(torch, cfg.vocab_size), steps=5)

        # global_step runs 0..4, so the save grid fires at steps 2 and 4.
        names = sorted(p.name for p in Path(d).glob("checkpoint-*"))
        assert names == ["checkpoint-2", "checkpoint-4"]
        for name in names:
            ck = Path(d) / name
            assert (ck / checkpoints.COMPLETE_MARKER).exists(), name
            # The format marker is what stops a torn CURRENT save from
            # passing the legacy per-file heuristic: without it, a kill
            # between the optimizer and scheduler writes yields a
            # "complete" checkpoint that silently resets the LR schedule.
            assert (ck / checkpoints.FORMAT_MARKER).exists(), name
            assert checkpoints.is_complete(ck), name
        assert not list(Path(d).glob("checkpoint-*.tmp"))
        assert not list(Path(d).glob("checkpoint-*.old"))


def test_the_cli_shows_a_classic_checkpoint_as_the_steps_it_holds_cpu():
    """The classic loop names a checkpoint by its last step from 0, as a recipe does:
    checkpoint-2 holds 3 updates, and a resume from it starts at the 4th. `poros runs
    checkpoints` (and export, evaluate, train --from by step) must call it step 3, with step 3's
    loss, not step 2 with step 2's."""
    torch = _require_ml()
    from poros.cli.verbs import checkpoints as listed

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=5)
        tr.config.save_steps = 2
        out = tr.train(_loader(torch, cfg.vocab_size), steps=5)
        found = listed._folders(Path(d))
        assert {step: p.name for step, p in found.items()} == {3: "checkpoint-2",
                                                               5: "checkpoint-4"}
        for step, folder in found.items():
            state = torch.load(folder / "rng_state.pt", weights_only=False)
            assert state["global_step"] + 1 == step  # the updates it holds
        losses = listed._losses(Path(d))
        assert [losses[3], losses[5]] == pytest.approx(
            [out["loss_history"][2], out["loss_history"][4]], rel=1e-6)


def test_torn_checkpoint_is_refused_by_load_checkpoint_cpu():
    """A dir with adapter files but no optimizer/RNG state (a kill 20 ms into
    the save) must raise, not silently restart from step 0 with fresh Adam
    moments."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)

        (Path(ckpt) / "rng_state.pt").unlink()
        (Path(ckpt) / "optimizer_state.pt").unlink()
        from poros.core import checkpoints

        (Path(ckpt) / checkpoints.COMPLETE_MARKER).unlink()

        torch.manual_seed(0)
        model2, _ = _tiny_peft_qwen2(torch)
        tr2 = _make_trainer(torch, model2, d, max_steps=4)
        # The dir carries the FORMAT_MARKER of a current save, so the
        # completeness gate (missing sentinel) must be what refuses it --
        # pin that gate specifically, not the older per-file fallbacks.
        with pytest.raises(FileNotFoundError, match="not a complete checkpoint"):
            tr2.load_checkpoint(ckpt)


def test_missing_scheduler_state_is_refused_under_active_scheduler_cpu():
    """A checkpoint without lr_scheduler_state.pt must not resume a run that
    uses an LR scheduler: continuing would silently restart the warmup/decay
    curve at step 0 while weights/optimizer/RNG resume at step K."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)

        # Both markers stay: the dir is "complete" except for the scheduler
        # file, which is exactly the surface the load must now insist on.
        (Path(ckpt) / "lr_scheduler_state.pt").unlink()

        torch.manual_seed(0)
        model2, _ = _tiny_peft_qwen2(torch)
        tr2 = _make_trainer(torch, model2, d, max_steps=4)
        assert tr2._lr_scheduler is not None  # cosine: the guard must be live
        with pytest.raises(FileNotFoundError, match="lr_scheduler_state"):
            tr2.load_checkpoint(ckpt)


def test_truncated_adapter_is_refused_by_load_checkpoint_cpu():
    """One valid tensor removed from an otherwise-valid adapter must be
    refused BEFORE mutation -- `poros eval` rides load_checkpoint, so an
    unexpected-keys-only check would evaluate a part-random adapter while
    naming the intended run."""
    torch = _require_ml()
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)

        st = Path(ckpt) / "adapter_model.safetensors"
        state = load_file(str(st))
        dropped = sorted(state)[0]
        state.pop(dropped)
        save_file(state, str(st))

        torch.manual_seed(0)
        model2, _ = _tiny_peft_qwen2(torch)
        tr2 = _make_trainer(torch, model2, d, max_steps=4)
        probe = next(
            p for n, p in model2.named_parameters() if p.requires_grad
        ).detach().clone()
        with pytest.raises(RuntimeError, match="missing"):
            tr2.load_checkpoint(ckpt, load_optimizer=False, load_rng=False)
        # Refused BEFORE mutation: the model is untouched.
        assert torch.equal(
            probe,
            next(p for n, p in model2.named_parameters() if p.requires_grad),
        )


def test_resized_embedding_adapter_still_loads_cpu():
    """PEFT's save_embedding_layers='auto' resolves the base id (sometimes
    over the network), so a save made under the portable hub id can include
    embed/lm_head tensors a later local-identity load does not expect.
    Those keys are ADVISORY: the checkpoint must load (with a warning),
    never be refused as 'unexpected' -- the round-3 validator refused the
    trainer's own checkpoint here."""
    torch = _require_ml()

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    sha = "b" * 40
    with tempfile.TemporaryDirectory() as d:
        snap = (Path(d) / "hub" / "models--Qwen--Qwen2.5-0.5B"
                / "snapshots" / sha)
        snap.mkdir(parents=True)
        model.peft_config["default"].base_model_name_or_path = str(snap)
        model.get_base_model().config._name_or_path = str(snap)
        model.get_base_model().name_or_path = str(snap)

        out = Path(d) / "out"
        tr = _make_trainer(torch, model, str(out), max_steps=2)
        tr.config.model_name_or_path = "Qwen/Qwen2.5-0.5B"
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(str(out / "adapter"))

        from safetensors.torch import load_file
        saved = load_file(str(out / "adapter" / "adapter_model.safetensors"))
        if not any("embed" in k or "lm_head" in k for k in saved):
            pytest.skip("save_embedding_layers did not trigger (offline?)")

        # Must load; the key-set difference surfaces as the ADVISORY warning
        # (PEFT's own config-file warning also fires; pytest.warns allows it).
        with pytest.warns(UserWarning, match="embedding/lm_head"):
            tr.load_checkpoint(str(out / "adapter"),
                               load_optimizer=False, load_rng=False)


def test_shape_mismatched_adapter_is_refused_cpu():
    """A different lora_rank produces same-name, different-shape tensors --
    the validator must refuse before mutation."""
    torch = _require_ml()
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)

        st = Path(ckpt) / "adapter_model.safetensors"
        state = load_file(str(st))
        victim = next(k for k in sorted(state) if "lora_A" in k)
        state[victim] = state[victim][: state[victim].shape[0] // 2]
        save_file(state, str(st))

        torch.manual_seed(0)
        model2, _ = _tiny_peft_qwen2(torch)
        tr2 = _make_trainer(torch, model2, d, max_steps=4)
        with pytest.raises(RuntimeError, match="shape"):
            tr2.load_checkpoint(ckpt, load_optimizer=False, load_rng=False)


def test_native_validator_requires_exact_trainable_set():
    torch = _require_ml()
    from poros.train.adapters import validate_native_adapter_state

    model = torch.nn.Linear(4, 4)
    model.weight.requires_grad_(True)
    model.bias.requires_grad_(True)
    full = {"weight": model.weight.data, "bias": model.bias.data}
    with pytest.raises(RuntimeError, match="missing"):
        validate_native_adapter_state(model, {"weight": model.weight.data}, "x")
    with pytest.raises(RuntimeError, match="unexpected"):
        validate_native_adapter_state(
            model, {**full, "ghost": model.bias.data}, "x")
    with pytest.raises(RuntimeError, match="shape"):
        validate_native_adapter_state(
            model, {**full, "weight": model.weight.data[:2]}, "x")
    validate_native_adapter_state(model, full, "x")  # exact set passes


def test_lm_head_targeted_adapter_truncation_is_strict_cpu():
    """The advisory rule covers ONLY optional copied base tensors. A LoRA
    tensor on an lm_head/embed target is a real adapter tensor: removing it
    must refuse, not warn-and-proceed with that tensor at init."""
    torch = _require_ml()
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch, targets=("q_proj", "lm_head"))
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)

        st = Path(ckpt) / "adapter_model.safetensors"
        full = load_file(str(st))

        # The optional COPIED BASE tensor under the LoRA wrapper
        # (lm_head.base_layer.weight) is ADVISORY: a save made with
        # save_embedding_layers=False must load, with a warning.
        base_copy = next(
            (k for k in full if k.endswith("lm_head.base_layer.weight")), None)
        if base_copy is not None:
            state = {k: v for k, v in full.items() if k != base_copy}
            save_file(state, str(st))
            with pytest.warns(UserWarning, match="embedding/lm_head"):
                tr.load_checkpoint(ckpt, load_optimizer=False, load_rng=False)

        # A LoRA tensor on the same module is STRICT.
        victim = next(k for k in sorted(full)
                      if "lm_head" in k and ".lora_A." in k)
        state = {k: v for k, v in full.items() if k != victim}
        save_file(state, str(st))
        with pytest.raises(RuntimeError, match="missing"):
            tr.load_checkpoint(ckpt, load_optimizer=False, load_rng=False)


def test_modules_to_save_tensor_is_strict_cpu():
    """A modules_to_save copy is a TRAINED tensor whose saved key is
    name-identical to the optional base copy -- it must stay strict."""
    torch = _require_ml()
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = get_peft_model(
        Qwen2ForCausalLM(cfg),
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                   target_modules=["q_proj"], modules_to_save=["lm_head"],
                   bias="none", task_type="CAUSAL_LM"),
    )
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)

        st = Path(ckpt) / "adapter_model.safetensors"
        state = load_file(str(st))
        victim = next(k for k in state if k.endswith("lm_head.weight"))
        state.pop(victim)
        save_file(state, str(st))
        with pytest.raises(RuntimeError, match="missing"):
            tr.load_checkpoint(ckpt, load_optimizer=False, load_rng=False)


def test_foreign_tensor_same_count_is_refused_cpu():
    """Replacing one valid tensor with a same-count foreign key must fail:
    count-based checks pass this; exact key-set matching does not."""
    torch = _require_ml()
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)

        st = Path(ckpt) / "adapter_model.safetensors"
        state = load_file(str(st))
        victim = sorted(state)[0]
        state["not.a.real.adapter.key"] = state.pop(victim)
        save_file(state, str(st))

        torch.manual_seed(0)
        model2, _ = _tiny_peft_qwen2(torch)
        tr2 = _make_trainer(torch, model2, d, max_steps=4)
        with pytest.raises(RuntimeError, match="missing|unexpected"):
            tr2.load_checkpoint(ckpt, load_optimizer=False, load_rng=False)


def test_overwrite_save_invalidates_stale_sentinel_cpu(monkeypatch):
    """Saving over an already-complete directory must remove the OLD
    completion sentinel before any state lands: a kill mid-overwrite would
    otherwise leave a torn, mixed-generation dir that is_complete() blesses."""
    torch = _require_ml()
    from poros.core import checkpoints

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(Path(d) / "ckpt")
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(ckpt)
        assert checkpoints.is_complete(Path(ckpt))

        # Kill the overwrite at its first torch.save call (rng_state.pt --
        # the PEFT weights go through save_pretrained/safetensors), i.e.
        # after weights+config landed but before rng/optimizer/scheduler.
        real_save = torch.save
        calls = {"n": 0}

        def dying_save(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyboardInterrupt("simulated kill mid-overwrite")
            return real_save(*a, **kw)

        monkeypatch.setattr(torch, "save", dying_save)
        with pytest.raises(KeyboardInterrupt):
            tr.save(ckpt)
        monkeypatch.setattr(torch, "save", real_save)

        # The stale step-2 sentinel must NOT bless the torn overwrite.
        assert not checkpoints.is_complete(Path(ckpt))


def test_stranded_staging_dir_is_recovered_at_train_start_cpu():
    """A complete checkpoint stranded at checkpoint-N.tmp by a kill inside the promote window is
    adopted when the run resumes from it; a new run there is refused, naming the folders an
    earlier run left."""
    torch = _require_ml()
    from poros.core import checkpoints

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=2)
        # Simulate the strand: a complete save that never got promoted.
        staging = Path(d) / "checkpoint-99.tmp"
        tr.save(str(staging))
        # And a torn final sibling, as a mid-rmtree kill leaves it.
        torn = Path(d) / "checkpoint-99"
        torn.mkdir()
        (torn / "adapter_config.json").write_text("{}")

        from poros.core.errors import PorosConfigError

        with pytest.raises(PorosConfigError, match=r"checkpoint-99, checkpoint-99\.tmp"):
            tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2, resume_from=str(torn))

        promoted = Path(d) / "checkpoint-99"
        assert checkpoints.is_complete(promoted)
        assert (promoted / checkpoints.COMPLETE_MARKER).exists()
        assert not staging.exists()


def test_saved_adapter_identity_is_portable_and_schema_valid_cpu():
    """After a save with a snapshot-loaded base: adapter_config.json AND the
    PEFT model card carry the portable hub id (with the snapshot sha as the
    revision pin), and poros_config.json round-trips through PorosConfig
    (extra='forbid') -- poros must never write an artifact it rejects."""
    torch = _require_ml()
    import json as _json

    from poros.core.config import PorosConfig

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    sha = "b" * 40
    with tempfile.TemporaryDirectory() as d:
        snap = Path(d) / "hub" / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / sha
        snap.mkdir(parents=True)
        # what a cache-snapshot load records: the peft config, the base
        # config, AND the base model's name_or_path attribute (independent
        # of the config -- the model-card tag reads the attribute)
        model.peft_config["default"].base_model_name_or_path = str(snap)
        model.get_base_model().config._name_or_path = str(snap)
        model.get_base_model().name_or_path = str(snap)

        out = Path(d) / "out"
        tr = _make_trainer(torch, model, str(out), max_steps=2)
        tr.config.model_name_or_path = "Qwen/Qwen2.5-0.5B"
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        tr.save(str(out / "adapter"))

        acfg = _json.loads((out / "adapter" / "adapter_config.json").read_text())
        assert acfg["base_model_name_or_path"] == "Qwen/Qwen2.5-0.5B"
        assert acfg.get("revision") == sha
        card = (out / "adapter" / "README.md").read_text()
        assert str(snap) not in card, "model card still leaks the local path"

        pcfg = _json.loads((out / "adapter" / "poros_config.json").read_text())
        assert pcfg["resolved_base_model_path"] == str(snap)
        assert pcfg["model_revision"] == sha  # eval/export re-pin this base
        PorosConfig(**pcfg)  # strict read-back must accept poros's own file
        assert pcfg["path"] is None  # left to the run: a CPU run trains sequentially

        # runtime state restored after the save
        assert model.peft_config["default"].base_model_name_or_path == str(snap)


@pytest.mark.parametrize(("stop_after", "grad_accum", "save_steps", "eval_steps"), [
    (0, 1, 0, 0), (1, 1, 0, 0), (2, 1, 2, 0), (1, 2, 0, 0), (2, 1, 2, 2),
], ids=["first-step", "mid-run", "on-a-save-step", "grad-accum", "save-and-eval-step"])
def test_should_stop_checkpoints_and_resumes_bitwise_cpu(stop_after, grad_accum, save_steps,
                                                        eval_steps):
    """should_stop ends the run at a step boundary with a checkpoint, and
    resuming it gives the uninterrupted run's losses exactly."""
    torch = _require_ml()
    from poros.core.errors import PorosCancelledError

    def run(d, **train_kw):
        torch.manual_seed(0)
        # With LoRA dropout, training draws from the RNG; so does a shuffled
        # eval loader. The eval case is bitwise only if the checkpoint holds
        # the RNG state after eval.
        model, cfg = _tiny_peft_qwen2(torch, dropout=0.1 if eval_steps else 0.0)
        tr = _make_trainer(torch, model, d, grad_accum=grad_accum, max_steps=6)
        tr.config.save_steps, tr.config.eval_steps = save_steps, eval_steps
        evals = torch.utils.data.DataLoader(_loader(torch, cfg.vocab_size, n=4).dataset,
                                            batch_size=1, shuffle=True)
        return tr.train(_loader(torch, cfg.vocab_size), steps=6,
                        eval_dataset=evals if eval_steps else None, **train_kw)

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        full = run(a)
        checks = []
        with pytest.raises(PorosCancelledError) as exc:
            run(b, should_stop=lambda: checks.append(1) or len(checks) > stop_after)
        assert exc.value.step == stop_after  # the index; its words count the steps done
        assert f"after {stop_after + 1} step" in str(exc.value)
        resumed = run(b, resume_from=str(Path(b) / f"checkpoint-{stop_after}"))
    assert resumed["loss_history"] == full["loss_history"][stop_after + 1:]


def test_should_stop_on_the_last_step_lets_the_run_finish_cpu():
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=3)
        metrics = tr.train(_loader(torch, cfg.vocab_size), steps=3,
                           should_stop=lambda: tr.global_step == 2)
    assert len(metrics["loss_history"]) == 3


def _two_checkpoints_then_a_full_disk(torch, d, monkeypatch, fail):
    """A run that saved a whole checkpoint, then a disk that refuses the next save's optimizer
    state (``fail``: the exception torch.save raises); returns (the whole checkpoint's name, the
    error the next save raised)."""
    from poros.train import checkpointing

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    tr = _make_trainer(torch, model, d, max_steps=2)
    tr.train(_loader(torch, cfg.vocab_size), steps=2)
    whole = Path(tr._save_checkpoint()).name
    tr.global_step += 2
    real = torch.save

    def refuse(obj, f, *a, **k):
        if str(f).endswith("optimizer_state.pt"):
            raise fail
        return real(obj, f, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(checkpointing.torch, "save", refuse)
        with pytest.raises(BaseException) as raised:
            tr._save_checkpoint()
    assert not (Path(d) / f"checkpoint-{tr.global_step}").exists()
    return whole, raised.value


def test_a_full_disk_mid_checkpoint_is_a_disk_error_and_leaves_the_last_whole_one(monkeypatch):
    """The 35B resume on the RTX 4090: torch.save's short write on a full disk ended the run as
    E_INTERNAL 'RuntimeError'. It is E_DISK and says the disk is full; nothing of the failed save
    looks like a checkpoint, and --resume-from latest takes the whole one before it."""
    import shutil
    from collections import namedtuple

    torch = _require_ml()
    from poros.core import checkpoints
    from poros.core.errors import PorosDiskError
    from poros.train import checkpointing

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(checkpointing.shutil, "disk_usage",
                        lambda p: usage(10**12, 10**12 - 2 * 10**6, 2 * 10**6))
    with tempfile.TemporaryDirectory() as d:
        short = RuntimeError("[enforce fail at inline_container.cc:672] . unexpected pos "
                             "780200256 vs 780200144")
        whole, err = _two_checkpoints_then_a_full_disk(torch, d, monkeypatch, short)
        assert isinstance(err, PorosDiskError) and err.code == "E_DISK"
        assert "the disk is full (0.0 GB free" in str(err) and "--resume-from latest" in \
            err.info.remediation
        assert [p.name for _, p in checkpoints.list_checkpoints(Path(d))] == [whole]
        assert not list(Path(d).glob("checkpoint-*.tmp"))
        from poros.cli.commands.train import _resolve_resume_from

        assert _resolve_resume_from("latest", d).name == whole
    assert shutil.disk_usage is not None


def test_an_enospc_is_the_disk_and_another_error_stays_itself(monkeypatch):
    """An OSError's ENOSPC is the disk whatever the free space reads; a write error with the disk
    not full is not called a full disk."""
    import errno

    torch = _require_ml()
    from poros.core.errors import PorosDiskError

    with tempfile.TemporaryDirectory() as d:
        _, err = _two_checkpoints_then_a_full_disk(
            torch, d, monkeypatch, OSError(errno.ENOSPC, "No space left on device"))
        assert isinstance(err, PorosDiskError) and "the disk is full" in str(err)
    with tempfile.TemporaryDirectory() as d:
        other = RuntimeError("[enforce fail at inline_container.cc:672] . unexpected pos 1 vs 2")
        _, err = _two_checkpoints_then_a_full_disk(torch, d, monkeypatch, other)
        assert err is other  # this host's disk has room: not a disk error


@pytest.mark.parametrize("field, value", [("seed", 7), ("gradient_accumulation_steps", 4),
                                          ("per_device_batch_size", 2)])
def test_resuming_into_another_run_is_refused_through_the_api(field, value):
    """PorosTrainer.load_checkpoint on CUDA as on a Mac: a resume onto other settings is refused
    with both values named, before any tensor loads (it was a Mac-only check)."""
    torch = _require_ml()
    from poros.core.errors import PorosConfigError

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=2)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        ckpt = tr._save_checkpoint()
        torch.manual_seed(0)
        model2, _ = _tiny_peft_qwen2(torch)
        tr2 = _make_trainer(torch, model2, d, max_steps=4)
        tr2.load_checkpoint(ckpt)  # the same run resumes
        setattr(tr2.config, field, value)
        with pytest.raises(PorosConfigError, match=f"{field} .* and this run has {value}"):
            tr2.load_checkpoint(ckpt)


def test_a_resumed_runs_events_supersede_the_steps_it_takes_again_cpu():
    """A run resumed from an earlier checkpoint takes its later steps again: events.jsonl keeps
    every line (append-only), with a resume marker, and as the record stands (events.current)
    each step is there once, with the uninterrupted run's losses."""
    import json as _json

    torch = _require_ml()
    from poros.core.events import current

    def run(d, **train_kw):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        tr = _make_trainer(torch, model, d, max_steps=6)
        tr.config.save_steps = 2
        tr.train(_loader(torch, cfg.vocab_size), steps=6, **train_kw)

    def events(d):
        return [_json.loads(x) for x in (Path(d) / "events.jsonl").read_text().splitlines()]

    def steps(evs):
        return [(e["step"], e["data"]["loss"]) for e in evs if e["event_type"] == "step"]

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        run(a)
        run(b)
        run(b, resume_from=str(Path(b) / "checkpoint-2"))
        whole, written = events(a), events(b)
    marks = [e for e in written if e["event_type"] == "resume"]
    assert [m["step"] for m in marks] == [3] and marks[0]["data"]["checkpoint"].endswith(
        "checkpoint-2")
    assert [s for s, _ in steps(written)] == [0, 1, 2, 3, 4, 5, 3, 4, 5]  # nothing rewritten
    assert steps(current(written)) == steps(whole)


def test_max_grad_norm_zero_logs_the_norm_it_does_not_clip_to_cpu():
    """max_grad_norm 0 turns clipping off, not the reading: the step event's grad_norm is the
    gradient's norm, the same as a clip too high to bite reports."""
    import json as _json

    torch = _require_ml()

    def norms(max_grad_norm):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        with tempfile.TemporaryDirectory() as d:
            tr = _make_trainer(torch, model, d, max_steps=3)
            tr.config.max_grad_norm = max_grad_norm
            tr.train(_loader(torch, cfg.vocab_size), steps=3)
            evs = [_json.loads(x) for x in (Path(d) / "events.jsonl").read_text().splitlines()]
        return [e["data"]["grad_norm"] for e in evs if e["event_type"] == "step"]

    off, high = norms(0.0), norms(1e9)
    assert all(n > 0 for n in off) and off == high


def test_resuming_with_more_steps_trains_on_the_longer_schedule_cpu():
    """A run resumed with a larger step count trains its next update at the longer schedule's
    rate, as the recipe loop's Run does, not the rate the shorter schedule left in the
    optimizer's state."""
    torch = _require_ml()

    def rates(d, steps, **train_kw):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        tr = _make_trainer(torch, model, d, max_steps=steps)
        tr.config.save_steps = 5
        seen = []
        tr.optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: seen.append(opt.param_groups[0]["lr"]))
        tr.train(_loader(torch, cfg.vocab_size, n=24), steps=steps, **train_kw)
        return seen

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        whole = rates(a, 12)
        short = rates(b, 6)
        longer = rates(b, 12, resume_from=str(Path(b) / "checkpoint-5"))
    assert short[-1] != whole[5]  # the two schedules differ where the short one ends
    assert longer == whole[6:]


def test_the_warmup_cap_holds_however_the_length_is_given_cpu():
    """warmup_steps is capped at about a tenth of a run's steps whether its length comes from
    train(steps=) or from the config's max_steps: the same 20 steps train at the same rates
    either way."""
    import json as _json

    torch = _require_ml()

    def rates(steps_kw):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        with tempfile.TemporaryDirectory() as d:
            tr = _make_trainer(torch, model, d, max_steps=20)
            tr.config.warmup_steps = 100
            tr.train(_loader(torch, cfg.vocab_size, n=40), **steps_kw)
            evs = [_json.loads(x) for x in (Path(d) / "events.jsonl").read_text().splitlines()]
        return [e["data"]["lr"] for e in evs if e["event_type"] == "step"]

    configured, given = rates({}), rates({"steps": 20})
    assert len(configured) == 20 and configured == given
    assert max(configured[:3]) > 0.9 * max(configured)  # at its peak in 2 steps, not 100


def test_a_new_run_is_refused_where_another_left_checkpoints_and_a_resume_keeps_its_id_cpu():
    """A new run in an output_dir an earlier run left checkpoints in is refused before anything
    is written, naming them (its checkpoints would be pruned against the old run's, and
    `latest` would pick the old run); a resume continues the run the checkpoint was (its id on
    every checkpoint it writes)."""
    torch = _require_ml()
    from poros.core import checkpoints
    from poros.core.errors import PorosConfigError

    def trainer(d):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        tr = _make_trainer(torch, model, d, max_steps=6)
        tr.config.save_steps = 2
        return tr, cfg

    with tempfile.TemporaryDirectory() as d:
        tr, cfg = trainer(d)
        tr.train(_loader(torch, cfg.vocab_size), steps=6)
        first = {p: checkpoints.run_of(p) for _, p in checkpoints.list_checkpoints(Path(d))}
        assert len(set(first.values())) == 1 and None not in first.values()
        events = (Path(d) / "events.jsonl").read_text()
        again, cfg = trainer(d)
        with pytest.raises(PorosConfigError, match="checkpoint-2, checkpoint-4"):
            again.train(_loader(torch, cfg.vocab_size), steps=6)
        assert (Path(d) / "events.jsonl").read_text() == events
        again.train(_loader(torch, cfg.vocab_size), steps=6,
                    resume_from=str(Path(d) / "checkpoint-2"))
        assert checkpoints.runs_in(Path(d)) == set(first.values())


@needs_ml
def test_pruning_reads_this_runs_checkpoints_alone(tmp_path):
    from types import SimpleNamespace

    from poros.core import checkpoints
    from poros.train.trainer import PorosTrainer

    def made(step, run):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if run:
            (d / checkpoints.RUN_MARKER).write_text(run)

    for step in (1, 2, 3):
        made(step, "a" * 16)
    made(50, None)  # an older Poros's, unstamped
    made(60, "b" * 16)  # another run's
    t = PorosTrainer.__new__(PorosTrainer)
    t._run_id = "a" * 16
    t.config = SimpleNamespace(save_total_limit=1, output_dir=str(tmp_path))
    t._prune_checkpoints()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint-3", "checkpoint-50", "checkpoint-60"]


def test_the_cli_refuses_a_new_run_and_a_latest_that_would_mix_runs(tmp_path):
    import typer

    from poros.cli.commands.train import _resolve_resume_from
    from poros.core import checkpoints
    from poros.preflight.command import fresh_run_refusal

    for step, run in ((2, "a" * 16), (4, "b" * 16)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        (d / checkpoints.RUN_MARKER).write_text(run)
        (d / checkpoints.FORMAT_MARKER).write_text("1")
        (d / checkpoints.COMPLETE_MARKER).write_text(str(step))
    cfg = type("C", (), {"output_dir": str(tmp_path)})()
    refused = fresh_run_refusal(None, cfg)
    assert refused is not None and "checkpoint-2, checkpoint-4" in refused.reason
    assert fresh_run_refusal(str(tmp_path / "checkpoint-4"), cfg) is None
    assert fresh_run_refusal(None, type("C", (), {"output_dir": str(tmp_path / "new")})()) is None
    with pytest.raises((typer.Exit, SystemExit)):
        _resolve_resume_from("latest", str(tmp_path))


def test_an_epoch_reads_each_row_once_and_ends_on_a_short_update_cpu():
    """An epoch of 10 micro-batches at 4 an update is 2 full updates and a last one that takes
    the 2 left (as the recipes do), re-reading no row from the start. And an iterable with no
    length and no max_steps is refused, not a raw TypeError."""
    torch = _require_ml()
    from poros.core.errors import PorosConfigError

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    seen = []
    ids = torch.randint(0, cfg.vocab_size, (10, 16), generator=torch.Generator().manual_seed(0))

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, i):
            seen.append(i)
            return {"input_ids": ids[i], "labels": ids[i].clone()}

    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, grad_accum=4, max_steps=-1)
        tr.config.num_epochs = 1
        out = tr.train(torch.utils.data.DataLoader(Rows(), batch_size=1, shuffle=False))
    assert sorted(seen) == list(range(10)) and len(out["loss_history"]) == 3

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter([{"input_ids": ids[0], "labels": ids[0].clone()}])

    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=-1)
        with pytest.raises(PorosConfigError, match="no length"):
            tr.train(torch.utils.data.DataLoader(Stream(), batch_size=1))


def test_a_held_out_set_smaller_than_a_batch_scores_every_row_cpu():
    """evaluate() and train(eval_dataset=) keep every held-out row, in order: a set smaller than
    a batch is scored (no NaN), and the last rows of any other are too (no drop_last)."""
    import math as _math

    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    ids = torch.randint(0, cfg.vocab_size, (3, 16), generator=torch.Generator().manual_seed(1))
    seen = []

    class Held(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, i):
            seen.append(i)
            return {"input_ids": ids[i], "labels": ids[i].clone()}

    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=1)
        tr.config.per_device_batch_size = 2
        out = tr.evaluate(Held())
    assert _math.isfinite(out["eval_loss"]) and seen == [0, 1, 2]


def test_a_checkpoints_files_reach_the_disk_before_its_completion_marker_cpu(monkeypatch):
    """The completion marker vouches for every file, so no power cut may leave a marker over
    bytes still in the page cache: each file and the folder are flushed before the marker, the
    marker and folder after, and the promoting rename's folder after it."""
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("reads which file a descriptor is through /proc")
    torch = _require_ml()
    from poros.core import checkpoints

    flushed = []
    real = os.fsync

    def fsync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        marker = (path if path.is_dir() else path.parent) / checkpoints.COMPLETE_MARKER
        flushed.append((path, marker.exists()))
        return real(fd)

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=1)
        tr.config.save_steps = 1
        monkeypatch.setattr(os, "fsync", fsync)
        tr.train(_loader(torch, cfg.vocab_size), steps=2)
        monkeypatch.setattr(os, "fsync", real)
    names = {p.name: before for p, before in flushed}
    assert names["optimizer_state.pt"] is False and names["rng_state.pt"] is False
    assert names[checkpoints.COMPLETE_MARKER] is True
    assert any(p.name == Path(d).name for p, _ in flushed)  # the folder the rename lands in
