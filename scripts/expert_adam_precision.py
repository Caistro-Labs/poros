"""How much an expert LoRA adapter's AdamW loses in bf16, and what each fix gets back. CPU only.

One fused expert projection of Qwen3-30B-A3B (gate_up_proj: in 2048, out 1536) with the engine's
LoRA: A kaiming-uniform (a = sqrt 5, the ``poros.moe.adapters`` init, values bf16), B zeros,
scaling alpha / rank. It fits a rank-64 target delta (decaying spectrum, so rank 16 cannot fit
all of it) under label noise, one update a step from 8 micro-batches of 32 tokens, with the
engine's AdamW settings (betas 0.9 / 0.999, eps 1e-8, weight decay 0.01, clip at 1.0, constant
rate). Every arm sees the same data and the same initial bytes; only the arithmetic of the update
differs. The forward of every arm but the fp32 reference uses bf16 weights with fp32 sums, as
the fused expert kernels do (``poros.moe.fused``).

Arms (bytes per trainable parameter held on the device: weights, grads, Adam's m and v):

  fp32        torch AdamW, fp32 weights, grads and state. The reference. 16 B
  bf16        torch AdamW on bf16 weights, grads and state: the engine before 1.0.0. 8 B
  expert_adamw  poros.train.expert_adamw.ExpertAdamW, the engine's own code: the
              nearest bf16 weight, and the remainder, m and v each a keyed stochastic rounding
              of its fp32 value. 10 B
  v2          an earlier design, in this script's own code: bf16 weight plus a remainder rounded
              to nearest, m rounded to nearest, v keyed (kahan_rnm_srv). 10 B
  master      fp32 master weights, fp32 m and v, a bf16 copy for the forward. 18 B
  kahan       bf16 weight plus a bf16 remainder (hi + lo), fp32 m and v. 14 B
  sr          bf16 weight written with keyed stochastic rounding, fp32 m and v. 12 B
  sr_m        as sr, m bf16 also stochastically rounded. 10 B
  sr_all      as sr, m and v bf16, both stochastically rounded. 8 B
  kahan_srmv  hi + lo weights, m and v bf16 stochastically rounded. 10 B
  sr_f32grad  as sr, but the 8 micro-batches' grads summed in fp32. 12 B (+ the fp32 sum)
  kahan_rnm   hi + lo weights, m bf16 rounded to nearest, v fp32. 12 B, no randomness
  kahan_rnm_srv  hi + lo weights, m bf16 rounded to nearest, v bf16 stochastically rounded. 10 B

Every arm but fp32 and sr_f32grad adds its micro-batches' bf16 grads in bf16, one after
another, as ``poros.moe.fused.accumulate_adapter_grads`` does today.

Stochastic rounding is keyed, as ``poros.core.sequential_rng.KeyedRng`` keys dropout: the 16
random bits for element i of tensor t at update u are a hash of (seed, u, t, i), so they depend
on no order and no device. What it reports, per arm, rate and data seed:

  * held-out loss every 20 updates (the noise floor subtracted: what is left to learn)
  * per update, the fraction of weight elements an update left unchanged (A and B apart), and
    of v and m elements
  * v and m against an fp64 shadow EMA of the same run's own grads (median ratio; relative
    error): what the stored moments lost to rounding, apart from where the trajectory went
  * at the end, || dW - dW_fp32 || / || dW_fp32 || for the same seed, beside the same distance
    between two fp32 runs on different data (the noise floor)

A condition is ``rate:schedule`` and optionally ``:key=value,...``: ``steps`` (its own length),
``beta1``, ``wd`` (weight decay), ``noise`` (the targets' noise; lower is a steadier gradient),
``idle=A/I`` (a second expert, routed half of the tokens for A updates, then none for I updates,
in turn: its gradient is exactly zero while idle, as a MoE expert no token reaches). For an
idle condition the report adds each arm's shrink of the idle expert's weights over each idle
stretch, from 100 updates into it (its momentum spent: what is left is the weight decay) to its
end, 1 - |w_end| / |w_start|, beside fp32's.

    python scripts/expert_adam_precision.py --out precision.json            # 2,000 updates
    python scripts/expert_adam_precision.py --steps 200 --seeds 1 --out q.json  # a quick look
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's poros

H, OUT, RANK, ALPHA = 2048, 1536, 16, 32
TARGET_RANK, MICRO, TOKENS = 64, 8, 32
BETAS, EPS, WD, CLIP = (0.9, 0.999), 1e-8, 0.01, 1.0
TARGET_RMS, NOISE = 3e-3, 0.1
LOSS_SCALE = 1e-2  # puts the grad's RMS per element near a real 30B-A3B run's (about 1e-5..1e-4)

ARMS = {  # name: (weights, m, v, grads summed in fp32)
    "fp32": ("fp32", "fp32", "fp32", True),
    "bf16": ("torch_bf16", "bf16", "bf16", False),
    "expert_adamw": ("poros", "sr", "sr", False),
    "v2": ("kahan", "rn", "sr", False),
    "master": ("master", "fp32", "fp32", False),
    "kahan": ("kahan", "fp32", "fp32", False),
    "sr": ("sr", "fp32", "fp32", False),
    "sr_m": ("sr", "sr", "fp32", False),
    "sr_all": ("sr", "sr", "sr", False),
    "kahan_srmv": ("kahan", "sr", "sr", False),
    "kahan_rnm": ("kahan", "rn", "fp32", False),
    "kahan_rnm_srv": ("kahan", "rn", "sr", False),
    "sr_f32grad": ("sr", "fp32", "fp32", True),
}
BYTES = {"fp32": 16, "bf16": 8, "expert_adamw": 10, "v2": 10, "master": 18,
         "kahan": 14, "sr": 12, "sr_m": 10, "sr_all": 8,
         "kahan_srmv": 10, "sr_f32grad": 12, "kahan_rnm": 12, "kahan_rnm_srv": 10}

_M32 = 0xFFFFFFFF


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """``x * c`` mod 2**32 for x below 2**32, with no product reaching 2**63: the same integers on
    every device (no signed overflow for a backend to treat its own way)."""
    lo = x * (c & 0xFFFF)
    hi = ((x * (c >> 16)) & 0xFFFF) << 16
    return (lo + hi) & _M32


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash (lowbias32), on int64 lanes holding values below 2**32."""
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def keyed_bits(seed: int, update: int, tensor: int, n: int) -> torch.Tensor:
    """16 random bits per element, a pure function of (seed, update, tensor, element index)."""
    key = _mix32(torch.tensor([(seed * 0x9E3779B1 + update) & _M32], dtype=torch.int64))
    key = _mix32((key ^ ((tensor * 0x85EBCA77) & _M32)) & _M32)
    idx = torch.arange(n, dtype=torch.int64)
    return _mix32(_mix32(idx ^ key) ^ (key >> 7)) & 0xFFFF


def sr_bf16(x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """``x`` (fp32) to bf16, rounded up in magnitude with probability (its dropped bits) / 2**16:
    E[result] = x. Adds 16 random bits under the kept ones and truncates."""
    i = x.contiguous().view(torch.int32).to(torch.int64) & _M32
    out = ((i + bits.view(x.shape)) & 0xFFFF0000)
    out = torch.where(torch.isfinite(x), out, i)  # inf and nan pass as they are
    out = out - ((out >> 31) << 32)  # back to int32's signed range
    return out.to(torch.int32).view(torch.float32).to(torch.bfloat16)


@dataclasses.dataclass(frozen=True)
class Setting:
    """One condition: its rate and schedule, its length, Adam's betas, the decay, the targets'
    noise, and the second expert's routing (None: one expert, always routed)."""

    lr: float
    schedule: str = "constant"
    steps: int = 2000
    beta1: float = 0.9
    beta2: float = 0.999
    wd: float = WD
    noise: float = NOISE
    idle: tuple[int, int] | None = None  # (updates routed, updates idle), in turn

    @property
    def experts(self) -> int:
        return 1 if self.idle is None else 2

    def label(self) -> str:
        extra = [f"steps={self.steps}" if self.steps != _STEPS[0] else "",
                 f"beta1={self.beta1:g}" if self.beta1 != 0.9 else "",
                 f"wd={self.wd:g}" if self.wd != WD else "",
                 f"noise={self.noise:g}" if self.noise != NOISE else "",
                 f"idle={self.idle[0]}/{self.idle[1]}" if self.idle else ""]
        extra = [e for e in extra if e]
        return f"{self.lr:g} {self.schedule}" + (f" ({', '.join(extra)})" if extra else "")

    def routed(self, update: int) -> bool:
        """Whether the second expert is routed at this update (1-based)."""
        if self.idle is None:
            return False
        on, off = self.idle
        return (update - 1) % (on + off) < on


_STEPS = [2000]  # the default length: --steps


def parse(condition: str) -> Setting:
    lr, schedule, *rest = condition.split(":")
    kw: dict[str, Any] = {"lr": float(lr), "schedule": schedule, "steps": _STEPS[0]}
    for item in (rest[0].split(",") if rest else []):
        key, value = item.split("=")
        if key == "idle":
            on, off = value.split("/")
            kw["idle"] = (int(on), int(off))
        elif key == "steps":
            kw["steps"] = int(value)
        else:
            kw[{"beta1": "beta1", "wd": "wd", "noise": "noise"}[key]] = float(value)
    return Setting(**kw)


class Data:
    """Inputs and noisy targets for update ``u`` of data seed ``s``: the same for every arm."""

    def __init__(self, seed: int, noise: float):
        g = torch.Generator().manual_seed(1000)  # one target for every seed: only the data moves
        u = torch.randn(OUT, TARGET_RANK, generator=g)
        v = torch.randn(TARGET_RANK, H, generator=g)
        spectrum = 1.0 / (1.0 + torch.arange(TARGET_RANK, dtype=torch.float32) / 8.0)
        u = u * spectrum
        scale = TARGET_RMS / float((u @ v).pow(2).mean().sqrt())
        self.u, self.v = u * math.sqrt(scale), v * math.sqrt(scale)  # target delta = u @ v
        self.seed, self.noise = seed, noise
        eg = torch.Generator().manual_seed(77)  # one held-out batch for every seed and arm
        self.x_eval = torch.randn(1024, H, generator=eg).bfloat16().float()
        self.t_eval = (self.x_eval @ self.v.T) @ self.u.T

    def batch(self, update: int, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed((self.seed * 1_000_003 + update) * 16 + micro)
        x = torch.randn(TOKENS, H, generator=g).bfloat16().float()
        t = (x @ self.v.T) @ self.u.T + self.noise * torch.randn(TOKENS, OUT, generator=g)
        return x, t


def initial(experts: int) -> list[torch.Tensor]:
    """The engine's init, per expert: kaiming-uniform A on a bf16 tensor, zero B."""
    torch.manual_seed(0)  # one init for all: only the data seed changes between seeds
    out = []
    for _ in range(experts):
        a = torch.empty(RANK, H, dtype=torch.bfloat16)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        out += [a.float(), torch.zeros(OUT, RANK)]
    return out


class Arm:
    """One way of storing and updating the factors [A0, B0, A1, B1, ...]."""

    def __init__(self, name: str, init: list[torch.Tensor], setting: Setting, seed: int):
        self.name, self.lr, self.seed, self.set = name, setting.lr, seed, setting
        self.betas = (setting.beta1, setting.beta2)
        self.wkind, self.mkind, self.vkind, self.f32grad = ARMS[name]
        self.t = 0
        names = [f"{f}{k // 2}" for k, f in zip(range(len(init)), ("A", "B") * len(init))]
        if self.wkind == "poros":
            from poros.train.expert_adamw import ExpertAdamW

            self.params = [torch.nn.Parameter(x.bfloat16()) for x in init]
            self.names = names
            self.opt = ExpertAdamW(list(zip(names, self.params)), lr=self.lr, betas=self.betas,
                                   eps=EPS, weight_decay=setting.wd, seed=seed)
            return
        if self.wkind in ("fp32", "torch_bf16"):
            dt = torch.float32 if self.wkind == "fp32" else torch.bfloat16
            self.params = [torch.nn.Parameter(x.to(dt)) for x in init]
            self.opt = torch.optim.AdamW(self.params, lr=self.lr, betas=self.betas, eps=EPS,
                                         weight_decay=setting.wd, foreach=False)
            return
        self.hi = [x.bfloat16() for x in init]  # what the forward reads
        self.master = [x.clone() for x in init] if self.wkind == "master" else None
        self.lo = [torch.zeros_like(h) for h in self.hi] if self.wkind == "kahan" else None
        self.m = [torch.zeros_like(x) for x in init]
        self.v = [torch.zeros_like(x) for x in init]
        if self.mkind in ("sr", "rn"):
            self.m = [x.bfloat16() for x in self.m]
        if self.vkind == "sr":
            self.v = [x.bfloat16() for x in self.v]

    def weights(self) -> list[torch.Tensor]:
        """The fp32 values the forward multiplies by (bf16 values, except the reference's)."""
        if self.wkind in ("fp32", "torch_bf16", "poros"):
            return [p.detach().float() for p in self.params]
        return [h.float() for h in self.hi]

    def stored(self) -> list[torch.Tensor]:
        """The weight as stored (what the next update starts from), fp32."""
        if self.wkind == "poros":
            held = self.opt.weights()
            return [held[n] for n in self.names]
        if self.wkind in ("fp32", "torch_bf16"):
            return [p.detach().float().clone() for p in self.params]
        if self.wkind == "master":
            return [w.clone() for w in self.master]
        if self.wkind == "kahan":
            return [h.float() + lo.float() for h, lo in zip(self.hi, self.lo)]
        return [h.float() for h in self.hi]

    def moments(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.wkind in ("fp32", "torch_bf16", "poros"):
            st = [self.opt.state[p] or {"exp_avg": torch.zeros_like(p),
                                        "exp_avg_sq": torch.zeros_like(p)} for p in self.params]
            return ([s["exp_avg"].float().clone() for s in st],
                    [s["exp_avg_sq"].float().clone() for s in st])
        return [m.float().clone() for m in self.m], [v.float().clone() for v in self.v]

    def _bits(self, kind: int, tensor: int, n: int) -> torch.Tensor:
        return keyed_bits(self.seed, self.t, kind * 8 + tensor, n)

    def step(self, grads: list[torch.Tensor], lr: float) -> None:
        self.t += 1
        self.lr = lr
        if self.wkind in ("fp32", "torch_bf16", "poros"):
            for group in self.opt.param_groups:
                group["lr"] = lr
            for p, g in zip(self.params, grads):
                p.grad = g.to(p.dtype)
            self.opt.step()
            return
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, g in enumerate(grads):
            g = g.float()
            m = self.m[k].float().lerp(g, 1 - b1)  # torch's single-tensor AdamW, in fp32
            v = self.v[k].float().mul(b2).addcmul(g, g, value=1 - b2)
            self.m[k] = (m if self.mkind == "fp32" else m.bfloat16() if self.mkind == "rn"
                         else sr_bf16(m, self._bits(1, k, m.numel())))
            self.v[k] = v if self.vkind == "fp32" else sr_bf16(v, self._bits(2, k, v.numel()))
            w = self.stored()[k]
            w.mul_(1 - self.lr * self.set.wd)
            denom = (v.sqrt() / math.sqrt(bc2)).add_(EPS)
            w.addcdiv_(m, denom, value=-self.lr / bc1)
            if self.wkind == "master":
                self.master[k] = w
                self.hi[k] = w.bfloat16()
            elif self.wkind == "kahan":
                self.hi[k] = w.bfloat16()
                self.lo[k] = (w - self.hi[k].float()).bfloat16()
            else:
                self.hi[k] = sr_bf16(w, self._bits(0, k, w.numel()))


def grads_of(arm: Arm, data: Data, update: int) -> tuple[list[torch.Tensor], float]:
    """One update's grads, 8 micro-batches, each rounded and added as the arm does, then clipped
    at 1.0 by global norm in the grads' dtype (as clip_grad_norm_ does). With a second expert
    routed, it takes the micro-batch's second half of the tokens; idle, its grads are exactly
    zero."""
    ws = arm.weights()
    experts = len(ws) // 2
    second = experts == 2 and arm.set.routed(update)
    acc: list[torch.Tensor] | None = None
    exact = arm.name == "fp32"
    scaling = ALPHA / RANK
    total = 0.0
    for micro in range(MICRO):
        x, t = data.batch(update, micro)
        leaves = [w.clone().requires_grad_(True) for w in ws]
        split = TOKENS // 2 if second else TOKENS
        pred0 = scaling * (x[:split] @ leaves[0].T) @ leaves[1].T
        loss = (pred0 - t[:split]).pow(2).sum()
        if second:
            pred1 = scaling * (x[split:] @ leaves[2].T) @ leaves[3].T
            loss = loss + (pred1 - t[split:]).pow(2).sum()
        loss = LOSS_SCALE * loss / (TOKENS * OUT) / MICRO
        gs = list(torch.autograd.grad(loss, leaves, allow_unused=True))
        gs = [torch.zeros_like(w) if g is None else g for g, w in zip(gs, ws)]
        total += float(loss.detach())
        gs = gs if exact else [g.bfloat16() for g in gs]
        if arm.f32grad:
            gs = [g.float() for g in gs]
        acc = gs if acc is None else [s + g for s, g in zip(acc, gs)]
    assert acc is not None
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in acc]))
    coef = (CLIP / (norm + 1e-6)).clamp(max=1.0)
    return [g * coef.to(g.dtype) for g in acc], total


def eval_loss(arm: Arm, data: Data) -> float:
    """Held-out loss, averaged over the experts (each on the whole held-out batch)."""
    ws = arm.weights()
    losses = []
    for e in range(len(ws) // 2):
        pred = (ALPHA / RANK) * (data.x_eval @ ws[2 * e].T) @ ws[2 * e + 1].T
        losses.append(float((pred - data.t_eval).pow(2).mean()))
    return sum(losses) / len(losses)


def rate(peak: float, schedule: str, update: int, steps: int) -> float:
    """The engine's schedule (poros.train.loop): warmup to the peak, then constant or cosine."""
    if schedule == "constant":
        return peak
    warmup = min(100, steps // 10)
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / max(1, steps - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def run(name: str, setting: Setting, seed: int, every: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    data = Data(seed, setting.noise)
    init = initial(setting.experts)
    arm = Arm(name, init, setting, seed)
    b1, b2 = arm.betas
    shadow_m = [torch.zeros_like(x, dtype=torch.float64) for x in init]
    shadow_v = [torch.zeros_like(s) for s in shadow_m]
    keys = ("w_same", "v_same", "m_same", "v_ratio_med", "v_relerr", "m_relerr", "train")
    rec: dict[str, list] = {k: [] for k in ("step", "eval", *keys)}
    same = {"w_same": 0.0, "v_same": 0.0, "m_same": 0.0}
    n_acc = 0
    rec["step"].append(0)
    rec["eval"].append(eval_loss(arm, data))
    for k in keys:
        rec[k].append(None)
    shrink, idle_start = [], None
    on, off = setting.idle or (0, 0)
    settle = min(100, off // 2)  # the idle expert's momentum has died down by then
    for u in range(1, setting.steps + 1):
        pos = (u - 1) % (on + off) if setting.idle else -1
        if setting.idle and pos == on + settle and idle_start is None:
            idle_start = float(torch.cat([w.double().flatten() for w in arm.stored()[2:]]).norm())
        grads, train = grads_of(arm, data, u)
        w_before, (m_before, v_before) = arm.stored(), arm.moments()
        arm.step(grads, rate(setting.lr, setting.schedule, u - 1, setting.steps))
        w_after, (m_after, v_after) = arm.stored(), arm.moments()
        if idle_start is not None and (u == setting.steps or setting.routed(u + 1)):
            end = float(torch.cat([w.double().flatten() for w in w_after[2:]]).norm())
            shrink.append(1 - end / idle_start)
            idle_start = None
        for k, g in enumerate(grads):
            g64 = g.double()
            shadow_m[k].mul_(b1).add_(g64, alpha=1 - b1)
            shadow_v[k].mul_(b2).addcmul_(g64, g64, value=1 - b2)
        if u > 1:  # the first update always moves everything (m, v from zero)
            n_w = sum(x.numel() for x in w_after)
            same["w_same"] += float(sum(int((x == y).sum()) for x, y in zip(w_after, w_before))
                                    / n_w)
            same["v_same"] += float(sum(int((x == y).sum()) for x, y in zip(v_after, v_before))
                                    / n_w)
            same["m_same"] += float(sum(int((x == y).sum()) for x, y in zip(m_after, m_before))
                                    / n_w)
            n_acc += 1
        if u % every == 0 or u == setting.steps:
            vs = torch.cat([x.double().flatten() for x in v_after])
            sv = torch.cat([x.flatten() for x in shadow_v])
            ms = torch.cat([x.double().flatten() for x in m_after])
            sm = torch.cat([x.flatten() for x in shadow_m])
            ok = sv > 0
            rec["step"].append(u)
            rec["eval"].append(eval_loss(arm, data))
            rec["train"].append(train / LOSS_SCALE)
            for k in same:
                rec[k].append(same[k] / max(n_acc, 1))
                same[k] = 0.0
            n_acc = 0
            rec["v_ratio_med"].append(float((vs[ok] / sv[ok]).median()))
            rec["v_relerr"].append(float((vs - sv).norm() / sv.norm()))
            rec["m_relerr"].append(float((ms - sm).norm() / sm.norm()))
    w = arm.stored()
    delta = torch.cat([((ALPHA / RANK) * w[2 * e + 1] @ w[2 * e]).flatten()
                       for e in range(len(w) // 2)])
    return {"arm": name, "condition": setting.label(), "seed": seed, "rec": rec,
            "idle_shrink": shrink, "_delta": delta}


def _job(args: tuple) -> dict[str, Any]:
    name, setting, seed, every = args
    out = run(name, setting, seed, every)
    delta = out.pop("_delta")
    torch.save(delta, os.path.join(_job.tmp, _file(name, setting, seed)))
    return out


def _file(name: str, setting: Setting, seed: int) -> str:
    tag = "".join(c if c.isalnum() else "_" for c in setting.label())
    return f"{name}_{tag}_{seed}.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--conditions", nargs="+", default=["2e-5:constant", "2e-4:constant",
                                                         "2e-4:cosine"],
                    help="rate:schedule[:key=value,...] (see the module's docstring)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--tmp", default=None,
                    help="folder for the final deltas (default: next to --out)")
    ap.add_argument("--keep-every", type=int, default=100,
                    help="the curves the output keeps: every this many updates")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _STEPS[0] = args.steps
    tmp = args.tmp or (os.path.splitext(args.out)[0] + "_deltas")
    os.makedirs(tmp, exist_ok=True)
    settings = [parse(c) for c in args.conditions]
    jobs = [(arm, st, seed, args.every)
            for st in settings for seed in range(args.seeds) for arm in args.arms]
    jobs.sort(key=lambda j: -j[1].steps * j[1].experts)  # the longest first
    started = time.time()
    # each worker gets the run's state from its initializer, never from this process's module:
    # a worker started by forkserver or spawn (Python 3.14's default on Linux) imports the
    # module afresh
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(args.steps, tmp)) as pool:
        results = list(pool.map(_job, jobs))
    deltas = {(r["arm"], r["condition"], r["seed"]): torch.load(
        os.path.join(tmp, _file(r["arm"], parse_label(settings, r["condition"]), r["seed"])))
        for r in results}
    summary: dict[str, Any] = {}
    for st in settings:
        cond = st.label()
        rows = {}
        for arm in args.arms:
            runs = sorted((r for r in results if r["arm"] == arm and r["condition"] == cond),
                          key=lambda r: r["seed"])
            finals = [r["rec"]["eval"][-1] for r in runs]
            dist = []
            for r in runs:
                ref = deltas.get(("fp32", cond, r["seed"]))
                if ref is not None:
                    dist.append(float((deltas[(arm, cond, r["seed"])] - ref).norm() / ref.norm()))

            def late(key: str, runs=runs) -> float:
                vals = [x for r in runs for x in r["rec"][key][len(r["rec"][key]) // 2:]
                        if x is not None]
                return sum(vals) / len(vals) if vals else float("nan")

            rows[arm] = {
                "bytes_per_param": BYTES[arm],
                "eval_start": runs[0]["rec"]["eval"][0],
                "eval_final_each_seed": finals,
                "rel_dist_to_fp32_same_seed": dist,
                "w_unchanged_2nd_half": late("w_same"),
                "v_unchanged_2nd_half": late("v_same"),
                "m_unchanged_2nd_half": late("m_same"),
                "v_over_shadow_median_end": [r["rec"]["v_ratio_med"][-1] for r in runs],
                "idle_shrink_each_seed": [r["idle_shrink"] for r in runs] if st.idle else None,
            }
        ref = rows.get("fp32")
        for arm, row in rows.items():
            if ref:
                row["minus_fp32_each_seed"] = [a - b for a, b in zip(
                    row["eval_final_each_seed"], ref["eval_final_each_seed"])]
            if st.idle and ref:
                row["idle_shrink_over_fp32_each_seed"] = [
                    sum(a) / sum(b) for a, b in zip(row["idle_shrink_each_seed"],
                                                    ref["idle_shrink_each_seed"])]
        floor = []
        seeds = sorted({r["seed"] for r in results})
        for i in seeds:
            for j in seeds:
                if i < j and ("fp32", cond, i) in deltas and ("fp32", cond, j) in deltas:
                    di, dj = deltas[("fp32", cond, i)], deltas[("fp32", cond, j)]
                    floor.append(float((di - dj).norm() / dj.norm()))
        fp32_finals = ref["eval_final_each_seed"] if ref else []
        summary[cond] = {
            "setting": dataclasses.asdict(st), "arms": rows,
            "fp32_seed_sd": statistics.pstdev(fp32_finals) if len(fp32_finals) > 1 else None,
            "fp32_seed_to_seed_rel_dist": floor}
    report = {"what": __doc__.split("\n")[0], "steps": args.steps, "every": args.every,
              "shape": {"in": H, "out": OUT, "rank": RANK, "alpha": ALPHA,
                        "target_rank": TARGET_RANK, "micro_batches": MICRO, "tokens": TOKENS},
              "adamw": {"eps": EPS, "clip": CLIP},
              "seconds": round(time.time() - started, 1), "summary": summary,
              "curves_every": args.keep_every,
              "runs": [{**r, "rec": {k: [x for s_, x in zip(r["rec"]["step"], v)
                                         if s_ % args.keep_every == 0]
                                     for k, v in r["rec"].items()}} for r in results]}
    with open(args.out, "w") as f:
        json.dump(report, f, separators=(",", ":"))
    for cond, s in summary.items():
        sd = s["fp32_seed_sd"]
        print(f"{cond}  fp32 seed sd {sd:.1e}" if sd is not None else cond)
        for arm, row in s["arms"].items():
            diff = ["%+.1e" % d for d in row.get("minus_fp32_each_seed", [])]
            dist = [round(d, 4) for d in row["rel_dist_to_fp32_same_seed"]]
            idle = row.get("idle_shrink_over_fp32_each_seed")
            print(f"  {arm:12s} minus fp32 {diff}  dist {dist}  w-same "
                  f"{row['w_unchanged_2nd_half']:.3f}"
                  + (f"  idle shrink / fp32's {[round(x, 3) for x in idle]}" if idle else ""))


def _init_worker(steps: int, tmp: str) -> None:
    _STEPS[0] = steps
    _job.tmp = tmp


def parse_label(settings: list[Setting], label: str) -> Setting:
    return next(s for s in settings if s.label() == label)


if __name__ == "__main__":
    main()
