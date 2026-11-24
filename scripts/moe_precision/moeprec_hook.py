"""Run ``poros`` with this kit's two additions, then exit as ``poros`` exits:

* ``MOEPREC_FP32=1``: the fp32 reference. Every expert LoRA factor is made fp32 (the store still
  decodes the base to bf16). The fused expert kernels refuse an fp32 adapter, so the engine runs
  its reference expert compute (config.resolved.json says so in ``compute_reason``); every
  trainable tensor is then fp32, so the optimizer is torch's AdamW in fp32: the precision the
  tool Poros is compared with trains at.
* ``MOEPREC_SAMPLE=<file.jsonl>``: after every optimizer step, one line about a fixed sample of
  1,048,576 elements spread over every trainable tensor (the same elements in every run of a
  model): the share whose stored weight did not move, the share whose bf16 parameter did not
  move, the median of v over an fp64 moving average of the same clipped grads (1.0: v tracks
  its grads; above 1 and rising: v is frozen), and the relative error of v and m against those
  averages.

* ``MOEPREC_OPTIMIZER=torch-bf16``: the control. The engine under test unchanged, except that
  ``build_optimizer`` returns ``torch.optim.AdamW`` over the same trainable tensors with the
  same settings, stepping them in their own dtype (a MoE model's bf16 expert adapters in bf16):
  the update rule before 1.0.0, on the frozen engine, so it differs from the new arm in the
  optimizer alone. The run's records say so: ran_compute's ``expert_optimizer`` names the
  substitution and the tensors' dtypes, compute_reason says it in words, and each checkpoint's
  optimizer.json carries what the engine writes for any torch AdamW (its rule) with
  ``trainable_dtype`` bfloat16, which is true. The kit writes ``.kit_arm`` beside ``.kit_commit``.

The patches go on ``poros.train.loop._Loop._initialize_optimizer`` and
``poros.moe.adapters.ExpertLoraAdapter``, which exist from before ExpertAdamW on, so the same file
drives the old engine, the new one and the reference. Python finds poros where PYTHONPATH says:

    PYTHONPATH=$NEW python moeprec_hook.py train job.yaml --output-dir runs/job --yes --json
"""

from __future__ import annotations

import json
import math
import os
import sys
import types

import torch

SAMPLE = 1 << 20


def _fp32_adapters() -> None:
    from poros.moe import adapters

    init = adapters.ExpertLoraAdapter.__init__

    def fp32_init(self, *args, **kwargs):
        kwargs["dtype"] = torch.float32
        init(self, *args, **kwargs)

    adapters.ExpertLoraAdapter.__init__ = fp32_init


class _Sampler:
    def __init__(self, optimizer, path: str) -> None:
        self.opt, self.path = optimizer, path
        params = [p for g in optimizer.param_groups for p in g["params"]]
        total = sum(p.numel() for p in params)
        gen = torch.Generator().manual_seed(0)
        self.idx = []
        for p in params:
            k = max(1, min(p.numel(), round(SAMPLE * p.numel() / total)))
            self.idx.append(torch.randperm(p.numel(), generator=gen)[:k].sort().values
                            .to(p.device))
        self.params = params
        b1, b2 = optimizer.param_groups[0]["betas"]
        self.b1, self.b2 = b1, b2
        # The fp64 averages and every sample they meet live on the CPU: MPS has no float64.
        self.sm = [torch.zeros(len(i), dtype=torch.float64) for i in self.idx]
        self.sv = [torch.zeros_like(x) for x in self.sm]
        self.update = 0

    def _held(self, p, idx):
        w = p.detach().reshape(-1)[idx].cpu().double()
        lo = self.opt.state[p].get("remainder") if p in self.opt.state else None
        return w if lo is None else w + lo.reshape(-1)[idx].cpu().double()

    def around(self, step):
        def wrapped(*args, **kwargs):
            pairs = [(p, i) for p, i in zip(self.params, self.idx)]
            before = [(p.detach().reshape(-1)[i].clone(), self._held(p, i)) for p, i in pairs]
            grads = [None if p.grad is None else p.grad.reshape(-1)[i].cpu().double()
                     for p, i in pairs]
            out = step(*args, **kwargs)
            self.update += 1
            same_p = same_w = n = 0
            ratios, dv, nv, dm, nm = [], 0.0, 0.0, 0.0, 0.0
            for k, (p, i) in enumerate(pairs):
                g = grads[k]
                if g is None:
                    continue
                self.sm[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
                self.sv[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                st = self.opt.state[p]
                v = st["exp_avg_sq"].reshape(-1)[i].cpu().double()
                m = st["exp_avg"].reshape(-1)[i].cpu().double()
                same_p += int((p.detach().reshape(-1)[i] == before[k][0]).sum())
                same_w += int((self._held(p, i) == before[k][1]).sum())
                n += len(i)
                ok = self.sv[k] > 0
                ratios.append((v[ok] / self.sv[k][ok]).float())
                dv += float((v - self.sv[k]).pow(2).sum())
                nv += float(self.sv[k].pow(2).sum())
                dm += float((m - self.sm[k]).pow(2).sum())
                nm += float(self.sm[k].pow(2).sum())
            r = torch.cat(ratios) if ratios else torch.zeros(0)
            line = {"update": self.update, "lr": self.opt.param_groups[0]["lr"],
                    "elements": n, "param_unchanged": same_p / max(n, 1),
                    "weight_unchanged": same_w / max(n, 1),
                    "v_over_exact_median": float(r.median()) if r.numel() else None,
                    "v_relerr": math.sqrt(dv / nv) if nv else None,
                    "m_relerr": math.sqrt(dm / nm) if nm else None,
                    "optimizer": type(self.opt).__name__,
                    "dtype": str(self.params[0].dtype)}
            with open(self.path, "a") as f:
                f.write(json.dumps(line) + "\n")
            return out
        return wrapped


def _sampling(path: str) -> None:
    from poros.train import loop

    init = loop._Loop._initialize_optimizer

    def with_sampler(self):
        init(self)
        sampler = _Sampler(self.optimizer, path)
        around = sampler.around(self.optimizer.step)
        # a bound method, as the step it replaces: torch's LR scheduler wraps optimizer.step
        # through its __func__, which a plain function has not (an AttributeError at the
        # scheduler's construction on torch 2.13 and 2.14)
        self.optimizer.step = types.MethodType(lambda _opt, *a, **kw: around(*a, **kw),
                                               self.optimizer)

    loop._Loop._initialize_optimizer = with_sampler


CONTROL = "torch-bf16"


def _torch_optimizer_control() -> None:
    """``build_optimizer`` returning torch's AdamW for every trainable tensor (as the engine before
    1.0.0 did for bf16 ones), with the run's record saying so."""
    from poros.train import expert_adamw, loop

    def torch_adamw(named_params, config):
        params = [p for _, p in list(named_params)]
        return torch.optim.AdamW(params, lr=config.learning_rate,
                                 betas=(config.adam_beta1, config.adam_beta2),
                                 eps=config.adam_epsilon, weight_decay=config.weight_decay)

    expert_adamw.build_optimizer = torch_adamw  # the loop imports it from the module at its call
    record = loop._Loop._record_optimizer

    def recorded(self):
        record(self)
        dtypes = sorted({str(p.dtype).removeprefix("torch.") for p in self.trainable_parameters})
        said = ("moeprec control: torch.optim.AdamW, substituted for the engine's optimizer by "
                f"scripts/moe_precision/moeprec_hook.py, steps the {'/'.join(dtypes)} trainable "
                "tensors in their own dtype")
        print(said, file=sys.stderr)
        compute = getattr(self, "compute", None)
        if compute is not None:
            compute.reasons.append(said)
            compute.note(expert_optimizer={"substituted": "torch.optim.AdamW", "by":
                                           "scripts/moe_precision/moeprec_hook.py",
                                           "dtypes": dtypes})

    loop._Loop._record_optimizer = recorded


def main() -> None:
    if os.environ.get("MOEPREC_FP32") == "1":
        _fp32_adapters()
    arm = os.environ.get("MOEPREC_OPTIMIZER", "")
    if arm == CONTROL:
        _torch_optimizer_control()
    elif arm:
        raise SystemExit(f"MOEPREC_OPTIMIZER={arm!r}: the kit knows {CONTROL!r} only")
    if os.environ.get("MOEPREC_SAMPLE"):
        _sampling(os.environ["MOEPREC_SAMPLE"])
    from poros.cli.main import cli

    sys.argv = ["poros", *sys.argv[1:]]
    cli()


if __name__ == "__main__":
    main()
