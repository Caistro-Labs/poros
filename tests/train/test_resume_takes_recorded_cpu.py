"""On the engine: a run of a config that names no LoRA shape, only the
adapter it starts from (rank and alpha not the defaults), stopped after a checkpoint and resumed
from the same config, trains on to the uninterrupted run's adapter and optimizer exactly.

The config form's steps before the load are the ones `poros train` takes (``start_refusal`` for
a new run; ``adopt_recorded`` and ``resume_refusal`` for a resume); the model is built at the
shape they resolved, as ``PorosTrainer.from_pretrained(config=...)`` builds it.
"""

from __future__ import annotations

import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

RANK, ALPHA, TARGETS = 4, 12, ["q_proj", "v_proj"]  # the adapter's: none of them the defaults


def _model(rank, alpha, targets):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    small = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=64)
    return get_peft_model(Qwen2ForCausalLM(small), LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=list(targets), bias="none",
        task_type="CAUSAL_LM")), small


def _engine(cfg):
    from poros.train.trainer import PorosTrainer
    from tests.test_train_loop_cpu import _loader

    model, small = _model(cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules)
    tr = PorosTrainer(cfg)
    tr.model, tr.device, tr._adapter_backend = model, torch.device("cpu"), "peft"
    tr._setup_blockwise_training()
    return tr, _loader(torch, small.vocab_size, n=32)


# what each run's config sets besides: the same in every run (a CPU loop, no schedule warmup)
SET = dict(model_name_or_path="dummy", block_size=2, deterministic=True,
           blockwise_arch_check="off", enable_prefetch=False, per_device_batch_size=1,
           gradient_accumulation_steps=1, warmup_steps=0, lr_scheduler="cosine", eval_steps=0,
           logging_steps=1, lora_dropout=0.0, learning_rate=1e-2)


def _config(tmp_path, out):
    """cut.yaml's fields: no lora_rank, lora_alpha or target modules."""
    from poros.core.config import PorosConfig

    return PorosConfig(**SET, dataset=str(tmp_path / "rows.jsonl"), init_adapter=str(
        tmp_path / "src"), output_dir=str(tmp_path / out), max_steps=6, save_steps=3)


def _state(tr):
    adapter = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    opt = tr.optimizer.state_dict()["state"]
    return adapter, {(k, name): v.clone() for k, s in opt.items() for name, v in s.items()
                     if torch.is_tensor(v)}


@pytest.fixture
def src(tmp_path):
    """A finished run's adapter in ./src: rank 4, alpha 12, two target modules."""
    from poros.core.config import PorosConfig

    (tmp_path / "rows.jsonl").write_text('{"text": "a row"}\n' * 8)
    cfg = PorosConfig(**SET, lora_rank=RANK, lora_alpha=ALPHA, lora_target_modules=TARGETS,
                      output_dir=str(tmp_path / "src"), max_steps=2, save_steps=0)
    tr, rows = _engine(cfg)
    tr.train(rows, steps=2)
    tr._save_checkpoint()
    return tmp_path


@pytest.mark.parametrize("source", ["kept", "deleted"])
def test_a_resume_of_the_same_config_is_the_uninterrupted_run(src, source):
    from poros.core.checkpoints import list_checkpoints
    from poros.core.errors import PorosCancelledError
    from poros.preflight import command
    from poros.preflight.resume import adopt_recorded

    def started(out):  # a new run: the start takes the adapter's shape
        cfg = _config(src, out)
        assert command.start_refusal(None, cfg) is None
        assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules) == (RANK, ALPHA, TARGETS)
        return _engine(cfg)

    full, rows = started("full")
    full.train(rows, steps=6)
    cut, rows = started("cut")
    with pytest.raises(PorosCancelledError):  # stopped once checkpoint-3 is saved
        cut.train(rows, steps=6, should_stop=lambda: cut.global_step >= 3)
    ckpt = str(list_checkpoints(src / "cut")[-1][1])
    saved = json.loads((src / "cut" / ckpt.rsplit("/", 1)[-1] / "poros_config.json").read_text())
    assert (saved["lora_rank"], saved["lora_alpha"]) == (RANK, ALPHA)
    if source == "deleted":  # a resume needs only its own checkpoint
        shutil.rmtree(src / "src")

    cfg = _config(src, "cut")  # the same config file, again
    assert command.resume_refusal(ckpt, cfg.model_copy()) is not None  # its defaults: refused
    assert adopt_recorded(ckpt, cfg) == {"lora_rank": RANK, "lora_alpha": ALPHA,
                                         "lora_target_modules": TARGETS}
    assert command.start_refusal(ckpt, cfg) is None and command.resume_refusal(ckpt, cfg) is None
    resumed, rows = _engine(cfg)
    resumed.start_from_adapter = lambda: pytest.fail("a resume loads no source adapter")
    resumed.train(rows, steps=6, resume_from=ckpt)

    (a_full, o_full), (a_res, o_res) = _state(full), _state(resumed)
    assert a_full.keys() == a_res.keys() and o_full.keys() == o_res.keys() and o_full
    assert all(torch.equal(a_full[n], a_res[n]) for n in a_full)
    assert all(torch.equal(o_full[k], o_res[k]) for k in o_full)
    assert resumed.global_step == full.global_step


def test_the_api_resumes_with_the_adapter_folder_gone(src, monkeypatch):
    """`PorosTrainer.from_pretrained(init_adapter=...)` then `train(resume_from=...)`: the
    adapter it began from had to be there, though a resume never reads it. Where the output
    folder holds the run's checkpoints, no new run can start there (train() refuses one), so its
    start is not checked; a new run's still is."""
    from poros.train.trainer import PorosTrainer

    class Built(Exception):  # the load is next: every check before it has passed
        pass

    def built(self, config):
        raise Built

    (src / "cut" / "checkpoint-3").mkdir(parents=True)  # the run's own checkpoint
    shutil.rmtree(src / "src")
    monkeypatch.setattr(PorosTrainer, "__init__", built)
    with pytest.raises(Built):
        PorosTrainer.from_pretrained("dummy", init_adapter=str(src / "src"), lora_rank=RANK,
                                     lora_alpha=ALPHA, output_dir=str(src / "cut"))
    with pytest.raises(FileNotFoundError, match="no folder at"):  # a new run: before the load
        PorosTrainer.from_pretrained("dummy", init_adapter=str(src / "src"), lora_rank=RANK,
                                     lora_alpha=ALPHA, output_dir=str(src / "fresh"))
