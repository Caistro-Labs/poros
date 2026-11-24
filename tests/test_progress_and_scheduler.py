"""CPU coverage for the trainer's live progress output and its LR schedule's length.

1. ``PorosTrainer.train()`` emits a live stdout progress display
(compact per-step line + a final summary) independent of any root-logger
configuration, gated by ``config.progress`` / the ``POROS_PROGRESS`` env var.

2. The LR scheduler is rebuilt in ``train()`` on the *effective* step
count (after the ``steps`` override), so a short run does not train inside a
warmup sized for the placeholder 1000-step run. warmup is clamped to the run;
a full-length configured run (no override + explicit max_steps) is untouched.

All tests skip cleanly when torch/transformers/peft are absent (mirrors
``tests/test_train_loop_cpu.py``); the manual-attach wiring builds a tiny
PEFT-wrapped Qwen2 so no bitsandbytes/CUDA/network is required.
"""

from __future__ import annotations

import tempfile

import pytest


def _require_ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def _tiny_peft_qwen2(torch):
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
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model, cfg


def _loader(torch, vocab, *, n=64, seq=16, seed=0):
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


def _make_trainer(
    torch,
    model,
    outdir,
    *,
    max_steps=5,
    warmup_steps=0,
    lr_scheduler="cosine",
    progress=True,
):
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
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler=lr_scheduler,
        save_steps=0,
        eval_steps=0,
        logging_steps=1,
        lora_dropout=0.0,
        progress=progress,
    )
    tr = PorosTrainer(cfg)
    tr.model = model
    tr.device = torch.device("cpu")
    tr._adapter_backend = "peft"
    tr._setup_blockwise_training()
    return tr


# ── progress display ───────────────────────────────────────────────────


def test_progress_display_prints_steps_and_summary(capsys):
    """With progress on (default), train() prints per-step lines AND the final
    summary to stdout even though nothing configures logging.basicConfig."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=3, progress=True)
        tr.train(_loader(torch, cfg.vocab_size), steps=3, seq_len=16)

    out = capsys.readouterr().out
    # Per-step compact lines (step N/total ... loss ... s/step ...).
    assert "s/step" in out
    assert "/3" in out
    assert out.count("loss") >= 3
    # The final summary always prints (steps + final loss are the headline).
    assert "Poros Training Complete" in out
    assert "Final loss:" in out
    assert "Total steps:" in out


def test_progress_false_is_silent_on_stdout(capsys):
    """With progress off and no logging configured, train() prints nothing to
    stdout -- exactly the (silent) logger-only behavior."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=3, progress=False)
        tr.train(_loader(torch, cfg.vocab_size), steps=3, seq_len=16)

    out = capsys.readouterr().out
    assert "s/step" not in out
    assert "Poros Training Complete" not in out


def test_poros_progress_env_forces_off(capsys, monkeypatch):
    """POROS_PROGRESS=0 overrides config.progress=True and silences stdout."""
    torch = _require_ml()
    monkeypatch.setenv("POROS_PROGRESS", "0")
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=3, progress=True)
        assert tr._progress_enabled() is False
        tr.train(_loader(torch, cfg.vocab_size), steps=3, seq_len=16)

    out = capsys.readouterr().out
    assert "Poros Training Complete" not in out


def test_poros_progress_env_forces_on():
    """POROS_PROGRESS=1 overrides config.progress=False."""
    torch = _require_ml()
    with tempfile.TemporaryDirectory() as d:
        model, _ = _tiny_peft_qwen2(torch)
        tr = _make_trainer(torch, model, d, progress=False)
        assert tr._progress_enabled() is False
        import os

        os.environ["POROS_PROGRESS"] = "1"
        try:
            assert tr._progress_enabled() is True
        finally:
            del os.environ["POROS_PROGRESS"]


# ── LR scheduler rebuilt on the effective step count ────────────────────


def test_scheduler_built_before_override_bakes_in_placeholder_warmup():
    """Sanity: at from_pretrained time the scheduler uses warmup=100 against
    the placeholder max_steps=1000 -- so a short run would train inside warmup
    at a near-zero LR unless train() rebuilds it."""
    torch = _require_ml()
    with tempfile.TemporaryDirectory() as d:
        model, _ = _tiny_peft_qwen2(torch)
        tr = _make_trainer(
            torch, model, d, max_steps=-1, warmup_steps=100, lr_scheduler="linear"
        )
        lam = tr._lr_scheduler.lr_lambdas[0]
        # step 4 of a nominal 8-step run: 4/100 -- effectively no learning.
        assert lam(4) == pytest.approx(4 / 100)


def test_rebuild_clamps_warmup_to_short_run():
    """_rebuild_lr_scheduler_for_run(8) clamps warmup to
    min(100, max(1, 8//10)) = 1, so the LR is at full scale right after step 1
    instead of stuck near zero."""
    torch = _require_ml()
    with tempfile.TemporaryDirectory() as d:
        model, _ = _tiny_peft_qwen2(torch)
        tr = _make_trainer(
            torch, model, d, max_steps=-1, warmup_steps=100, lr_scheduler="linear"
        )
        tr._rebuild_lr_scheduler_for_run(8)
        lam = tr._lr_scheduler.lr_lambdas[0]
        # warmup clamped to 1: step 1 is at full LR, step 4 well into decay
        # (linear: 1 - (4-1)/(8-1) ~= 0.571), NOT the old 0.04.
        assert lam(1) == pytest.approx(1.0)
        assert lam(4) > 0.5


def test_train_steps_override_rebuilds_scheduler():
    """The rebuild is wired into train(): a steps override fixes the live
    scheduler so a short run is not stuck inside warmup."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(
            torch, model, d, max_steps=-1, warmup_steps=100,
            lr_scheduler="linear", progress=False,
        )
        tr.train(_loader(torch, cfg.vocab_size), steps=8, seq_len=16)
        lam = tr._lr_scheduler.lr_lambdas[0]
        assert lam(1) == pytest.approx(1.0)
        assert lam(4) > 0.5


def test_explicit_fulllength_run_keeps_scheduler():
    """No steps override + explicit max_steps: the schedule is the one train(steps=) would build,
    warmup capped at about a tenth of the run (warmup_for), from from_pretrained on: a 6-step run
    does not keep the uncapped 100-step warmup."""
    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(
            torch, model, d, max_steps=6, warmup_steps=100,
            lr_scheduler="linear", progress=False,
        )
        before = tr._lr_scheduler.lr_lambdas[0]
        assert before(1) == pytest.approx(1.0)  # warmup_for(6) = 1 step, not 100
        tr.train(_loader(torch, cfg.vocab_size), seq_len=16)
        after = tr._lr_scheduler.lr_lambdas[0]
        assert [after(s) for s in range(6)] == [before(s) for s in range(6)]


def test_constant_scheduler_stays_none():
    """A constant schedule has no LambdaLR; the rebuild must not create one."""
    torch = _require_ml()
    with tempfile.TemporaryDirectory() as d:
        model, _ = _tiny_peft_qwen2(torch)
        tr = _make_trainer(
            torch, model, d, max_steps=-1, warmup_steps=100,
            lr_scheduler="constant",
        )
        assert tr._lr_scheduler is None
        tr._rebuild_lr_scheduler_for_run(8)
        assert tr._lr_scheduler is None


def test_progress_lines_count_the_steps_done_from_one(capsys):
    """The line said `step    0/3` for the first step, where the CLI's says `step 1/3`: both
    count the steps done. On disk nothing changes: the event's step is the update's index."""
    import json

    torch = _require_ml()
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=3, progress=True)
        tr.train(_loader(torch, cfg.vocab_size), steps=3, seq_len=16)
        events = [json.loads(x) for x in open(f"{d}/events.jsonl") if '"step"' in x]

    out = capsys.readouterr().out
    assert [x.split()[1] for x in out.splitlines() if x.startswith("step ")] == [
        "1/3", "2/3", "3/3"]
    assert [e["step"] for e in events if e["event_type"] == "step"] == [0, 1, 2]
