#!/usr/bin/env python3
"""anymodel_api.py MODEL MODEL_TYPE OUT.json -- the Python surfaces of "test any model" on one
family Poros has not validated, in one process, in this order (the registry is global, so the
unregistered checks come first):

  gate_error   PorosTrainer.from_pretrained(MODEL) at the default blockwise_arch_check ("error"):
               must raise PorosArchNotValidatedError (code E_ARCH_UNSUPPORTED) with a remediation;
               seconds taken
  gate_warn    blockwise_arch_check="warn": loads and trains 2 synthetic steps (finite losses)
  gate_off     blockwise_arch_check="off": the same
  register     poros.register_architecture(MODEL_TYPE): arch_status == user_asserted, and the
               validated set does not take it in
  trainer_registered  PorosTrainer at the default gate after registration: trains 2 steps

Every check is recorded with ok, seconds and what it saw; exit 0 iff all ok."""

from __future__ import annotations

import json
import math
import sys
import time
import traceback

import torch

model_id, mtype, out = sys.argv[1:4]
rec: dict = {"model": model_id, "model_type": mtype, "checks": {}}


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            got = fn()
            ok = bool(got.pop("ok"))
        except Exception as exc:  # recorded, never raised: the next check still runs
            got, ok = (
                {
                    "exception": f"{type(exc).__name__}: {exc}"[:600],
                    "trace": traceback.format_exc()[-1500:],
                },
                False,
            )
        got.update(ok=ok, seconds=round(time.time() - t0, 2))
        rec["checks"][name] = got
        print("CHECK", name, json.dumps(got)[:600], flush=True)
        return fn

    return wrap


KW = dict(
    quantization="nf4",
    lora_rank=8,
    lora_alpha=16,
    max_steps=2,
    progress=False,
    max_seq_length=128,
    per_device_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=0,
)


def trainer_steps(**kw):
    from poros.train.trainer import PorosTrainer

    tr = PorosTrainer.from_pretrained(model_id, output_dir=f"/tmp/fg_api_{mtype}", **KW, **kw)
    m = tr.train("synthetic", steps=2, seq_len=128)
    losses = m.get("loss_history") or []
    del tr
    torch.cuda.empty_cache()
    return {"ok": len(losses) == 2 and all(math.isfinite(x) for x in losses), "losses": losses}


@check("gate_error")
def _():
    from poros.core.errors import PorosArchNotValidatedError
    from poros.train.trainer import PorosTrainer

    try:
        PorosTrainer.from_pretrained(model_id, output_dir=f"/tmp/fg_api_{mtype}", **KW)
    except PorosArchNotValidatedError as exc:
        return {
            "ok": getattr(exc, "code", None) == "E_ARCH_UNSUPPORTED",
            "code": exc.code,
            "message": str(exc)[:400],
            "remediation": str(getattr(exc, "remediation", ""))[:300],
        }
    return {
        "ok": False,
        "message": "no PorosArchNotValidatedError: an unvalidated family loaded at the "
        "default gate",
    }


@check("gate_warn")
def _():
    return trainer_steps(blockwise_arch_check="warn")


@check("gate_off")
def _():
    return trainer_steps(blockwise_arch_check="off")


@check("register")
def _():
    import poros
    from poros.core.arch_registry import arch_status

    poros.register_architecture(mtype)
    st = arch_status(mtype)
    return {
        "ok": st == "user_asserted" and mtype not in poros.validated_arch_types(),
        "arch_status_registry": st,
    }


@check("trainer_registered")
def _():
    return trainer_steps()


rec["pass"] = all(c["ok"] for c in rec["checks"].values())
json.dump(rec, open(out, "w"), indent=1)
print(
    "ANYMODEL_API",
    json.dumps({"pass": rec["pass"], **{k: v["ok"] for k, v in rec["checks"].items()}}),
)
sys.exit(0 if rec["pass"] else 1)
