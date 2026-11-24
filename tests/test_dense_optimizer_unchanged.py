"""A dense run's optimizer is torch's AdamW, built as the engine built it before ExpertAdamW.

What this proves: on the code under test, the trainer's own optimizer for a dense run (all
trainable tensors fp32) is ``torch.optim.AdamW`` with exactly the old construction's arguments
and defaults, and a tiny PEFT Qwen2 trained through it gives the same losses, grad norms, adapter
and Adam state, bit for bit, as the same trainer with the earlier ``_initialize_optimizer``
construction (copied below as it was) monkeypatched in. So nothing between the trainer and
torch's AdamW changed for a dense run: not the choice, not the arguments, not the parameter
order.

What it does not prove: that the rest of the dense path computes as it did before ExpertAdamW (the
monkeypatch keeps everything else of HEAD in both runs), nor anything across torch versions.
The whole-tree comparison was made once, by hand: the same run on the earlier tree and on this
file's commit gave the digest da3ade45be5494984ef44cfe4733ea7a9ca3c32e1c30fc3890528599e42dfe12
(recorded in the message of the commit that added this file). A pinned digest is not a guard
worth keeping in the suite: CPU matmul and attention kernels differ across torch releases and
CPU instruction sets, so it would fail
on numerics this change never touched. The guard of the dense path's results is the hardware
certification of the dense recipes at the release commit.
"""

from __future__ import annotations

from tests.test_train_loop_cpu import _loader, _make_trainer, _require_ml, _tiny_peft_qwen2


def _as_before_expert_adamw(named, config):
    """``PorosTrainer._initialize_optimizer``'s optimizer before ExpertAdamW, verbatim."""
    import torch.optim as optim

    trainable = [param for _, param in named]
    return optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def _train(torch, tmp_path, monkeypatch=None, old=False):
    import poros.train.expert_adamw as ea

    if old:
        monkeypatch.setattr(ea, "build_optimizer", _as_before_expert_adamw)
    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    tr = _make_trainer(torch, model, str(tmp_path), grad_accum=2, max_steps=4)
    tr.train(_loader(torch, cfg.vocab_size), steps=4, seq_len=16)
    return tr


def test_a_dense_run_trains_as_it_did_before_expert_adamw(tmp_path, monkeypatch):
    torch = _require_ml()
    new = _train(torch, tmp_path / "new")
    assert type(new.optimizer) is torch.optim.AdamW
    cfg = new.config
    assert new.optimizer.defaults == torch.optim.AdamW(
        [torch.nn.Parameter(torch.zeros(1))], lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay).defaults  # foreach, fused, amsgrad... all torch's
    assert all(p.dtype == torch.float32 for p in new.trainable_parameters)
    old = _train(torch, tmp_path / "old", monkeypatch, old=True)
    assert new.optimizer.defaults == old.optimizer.defaults
    assert new.history.losses == old.history.losses
    assert new.history.grad_norms == old.history.grad_norms
    names = [n for n, p in new.model.named_parameters() if p.requires_grad]
    assert names == [n for n, p in old.model.named_parameters() if p.requires_grad]
    for (n, a), b in zip([(n, p) for n, p in new.model.named_parameters() if p.requires_grad],
                         old.trainable_parameters, strict=True):
        assert torch.equal(a, b), n
    sa, sb = new.optimizer.state_dict(), old.optimizer.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    for i, st in sa["state"].items():
        for k, v in st.items():
            assert torch.equal(v, sb["state"][i][k]), (i, k)
