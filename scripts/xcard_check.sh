#!/usr/bin/env bash
# xcard_check.sh -- one command, on any rented NVIDIA card: does Poros run correctly HERE?
#
#     bash scripts/xcard_check.sh [OUT_DIR]          # from the tree's root (a checkout or tarball)
#
# Needs a Python with Poros and its [dev] extra -- POROS_PYTHON, default python3; from the
# tree: pip install -e ".[dev]". That is torch, bitsandbytes, transformers, peft, accelerate,
# datasets, safetensors, triton (for the fused compute) and pytest (the test lanes); the `deps`
# check FAILs naming what is missing, and without pytest the test lanes SKIP saying so. Installs
# nothing.
# Downloads Qwen/Qwen2.5-0.5B (~1 GB, public, no token) and, for the refcheck,
# OpenAssistant/oasst1 (Apache-2.0). A whole run on an RTX 4090: docs/validation/xcard/rtx4090/
# (RESULT.txt: its start, end and every check). XCARD_QUICK=1 skips the refcheck (its
# three 100-update runs). Every GPU step holds POROS_GPU_LOCK (default /tmp/poros-gpu.lock), the test
# lanes for up to XCARD_LANE_TIMEOUT s each (default 2400). On a shared card, XCARD_LANE_PLUGIN=NAME
# runs the lanes with the pytest plugin NAME (-p NAME, importable from PYTHONPATH) instead of one
# hold for the whole lane: a plugin that takes POROS_GPU_LOCK per test itself.
#
# Prints PASS / FAIL / SKIP per check and writes everything under OUT_DIR (default
# ./xcard-<gpu>-<date>): RESULT.txt, env.json, the pytest logs and junit files, each run's config,
# log, events and adapter digests. Copy that folder back as the receipt. Exit 0: every check
# passed or was skipped for a stated reason.
#
# Checks:
#   env        the card, driver and stack (env.json); Poros's verdict for this class
#   deps       the Python stack the checks import (a missing module is named; FAIL)
#   cpu-sim    the card-class choosers, every class simulated (tests/core/test_xcard.py)
#   gpu-lane   pytest -m gpu over tests/, one process
#   gpu-dirs   tests/gpu tests/moe tests/sss + tests/test_run_block_major_cuda.py, one process
#   ranks      LoRA rank 8..256 trains (fused where the kernels fit this card's shared memory, else
#              PEFT's modules with the reason recorded) -- scripts/xcard/rank_probe.py; ranks-fit:
#              the compute each rank ran is the one poros.core.fused_tiles' table (the plan's) says
#   refuse     a config this card cannot run is refused in seconds, before any download
#   dense-*    Qwen2.5-0.5B, lora_dropout 0.05, 8 updates: sequential == efficient == fast ==
#              fast fully resident, bit for bit (losses, grad norms, every adapter tensor), per
#              compute: fused (default), fused_compute false, fp16. (No mode emulates an older
#              class or a host without Triton on this card: faking the capability makes Triton
#              compile for the faked one, which the real device cannot load, and hiding Triton
#              breaks torch's own imports. Pre-Ampere cards are not supported; refuse-bf16-emu75
#              checks their refusal, exit 4 E_GPU_UNSUPPORTED; the CPU suite's
#              tests/core/test_xcard.py checks the choosers.)
#   dense-fused-r64 / -r128  the same four paths at LoRA rank 64 and 128 (their own tiles), fused
#   refcheck   fused vs PEFT's modules, 100 updates on oasst1 rows, docs/validation/fused's rule
#              (mean and max |dloss| <= the micro-batch-split noise's mean and max)
#   moe-build  that small random Qwen3-MoE, built here from the dense model's tokenizer
#   moe-*      on it (no download): fused experts (default, and at LoRA rank 128), reference;
#              sequential == efficient == fast per compute
set -uo pipefail

TREE="$(cd "$(dirname "$0")/.." && pwd)"
CALLER="$(pwd)"
PY="${POROS_PYTHON:-python3}"
LOCK="${POROS_GPU_LOCK:-/tmp/poros-gpu.lock}"
MODEL="${XCARD_MODEL:-Qwen/Qwen2.5-0.5B}"
FUSED="$TREE/docs/validation/fused/scripts"
X="$TREE/scripts/xcard"
touch "$LOCK"
cd "$TREE"  # python -c puts the cwd first on sys.path: it must be the tree under test

GPU_TAG=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0).replace('NVIDIA ','').replace(' ','_'))" 2>/dev/null || echo nogpu)
OUT="${1:-xcard-${GPU_TAG}-$(date -u +%Y%m%dT%H%MZ)}"
case "$OUT" in /*) ;; *) OUT="$CALLER/$OUT" ;; esac
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
mkdir -p "$OUT/cfg" "$OUT/logs" "$OUT/runs" "$OUT/cmp"
RESULT="$OUT/RESULT.txt"; : > "$RESULT"
export TMPDIR="${TMPDIR:-$OUT/tmp}"; mkdir -p "$TMPDIR"
export PYTHONPATH="$TREE${PYTHONPATH:+:$PYTHONPATH}"   # the tree under test
export POROS_PROGRESS=0
FAILS=0

record() {  # record STATUS NAME DETAIL
  printf '%-4s  %-24s %s\n' "$1" "$2" "$3" | tee -a "$RESULT"
  [ "$1" = FAIL ] && FAILS=$((FAILS + 1))
  return 0
}
gpu() { flock "$LOCK" "$@"; }

echo "xcard_check: tree $TREE -> $OUT" | tee -a "$RESULT"
echo "start $(date -u +%FT%TZ)" | tee -a "$RESULT"

# --- env --------------------------------------------------------------------------------------
if "$PY" "$X/env_report.py" "$OUT/env.json" > "$OUT/logs/env.txt" 2>&1; then
  CAP=$("$PY" -c "import json;c=json.load(open('$OUT/env.json'))['capability'];print(f'{c[0]}.{c[1]}')")
  record PASS env "$(tail -1 "$OUT/logs/env.txt" | cut -c1-300)"
else
  record FAIL env "no CUDA device, or the stack does not import: logs/env.txt"
  cat "$RESULT"; exit 1
fi
# --- deps: what the checks import ---------------------------------------------------------
MISSING=$("$PY" - <<'PY' 2>/dev/null
import importlib.util as u
need = ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "safetensors",
        "pytest"]
print(" ".join(m for m in need if u.find_spec(m) is None))
PY
)
PYTEST=1
case " $MISSING " in *" pytest "*) PYTEST=0 ;; esac
if [ -z "$MISSING" ]; then record PASS deps "the training and [dev] stack imports"
else record FAIL deps "missing: $MISSING (pip install -e \".[dev]\")"; fi
NATIVE_BF16=$("$PY" -c "print(int(tuple(map(int,'$CAP'.split('.'))) >= (8,0)))")
TRITON=$("$PY" -c "import importlib.util as u; print(int(u.find_spec('triton') is not None))")
DTYPE=bfloat16; [ "$NATIVE_BF16" = 1 ] || DTYPE=float16   # pre-Ampere: the remedy Poros states
echo "capability $CAP  native_bf16=$NATIVE_BF16  triton=$TRITON  run dtype=$DTYPE" | tee -a "$RESULT"

# --- cpu-sim ----------------------------------------------------------------------------------
if [ "$PYTEST" = 0 ]; then record SKIP cpu-sim "pytest is not installed (deps)"
elif "$PY" -m pytest "$TREE/tests/core/test_xcard.py" "$TREE/tests/core/test_fused_flash_usable.py" \
  -q -p no:cacheprovider --junitxml="$OUT/logs/cpu_sim.xml" > "$OUT/logs/cpu_sim.txt" 2>&1; then
  record PASS cpu-sim "$(tail -1 "$OUT/logs/cpu_sim.txt")"
else record FAIL cpu-sim "$(tail -1 "$OUT/logs/cpu_sim.txt")"; fi

# --- the GPU test lanes -----------------------------------------------------------------------
lane() {  # lane NAME TIMEOUT pytest-args...
  local n=$1 to=$2; shift 2
  if [ "$PYTEST" = 0 ]; then record SKIP "$n" "pytest is not installed (deps)"; return; fi
  local hold=gpu plugin=()
  if [ -n "${XCARD_LANE_PLUGIN:-}" ]; then hold=env; plugin=(-p "$XCARD_LANE_PLUGIN"); fi
  (cd "$TREE" && $hold timeout -k 30 "$to" "$PY" -m pytest "$@" -p no:cacheprovider -rfEs \
     ${plugin[@]+"${plugin[@]}"} --junitxml="$OUT/logs/$n.xml" > "$OUT/logs/$n.txt" 2>&1)
  local rc=$? summary
  summary=$(grep -E '^(=+ )?[0-9]+ (passed|failed)|(passed|failed|error).* in [0-9.]+s' "$OUT/logs/$n.txt" | tail -1)
  if [ $rc -eq 0 ]; then record PASS "$n" "$summary"
  elif [ $rc -eq 5 ]; then record SKIP "$n" "no tests collected"
  else record FAIL "$n" "rc=$rc ${summary:-see logs/$n.txt}"; fi
}
LANE_TO="${XCARD_LANE_TIMEOUT:-2400}"
lane gpu-lane "$LANE_TO" -m gpu tests
lane gpu-dirs "$LANE_TO" tests/gpu tests/moe tests/sss tests/test_run_block_major_cuda.py

# --- ranks: every LoRA rank trains here (fused where its kernels fit, else PEFT's modules, said) --
mkdir -p "$OUT/ranks"
gpu timeout -k 30 1800 "$PY" "$X/rank_probe.py" "$OUT/ranks" > "$OUT/logs/ranks.txt" 2> "$OUT/logs/ranks.err"
if [ "$(grep -c '^RANK .* OK' "$OUT/logs/ranks.txt")" = 6 ]; then
  record PASS ranks "$(grep '^RANK' "$OUT/logs/ranks.txt" | sed 's/ reason=.*//' | tr '\n' ' ' | cut -c1-260)"
else
  record FAIL ranks "$(grep '^RANK .* FAIL' "$OUT/logs/ranks.txt" | head -2 | cut -c1-200)"
fi
rm -rf "$OUT/ranks"/rank*/checkpoint-*
# each rank ran the compute the plan's fit table says for this card (poros.core.fused_tiles: the
# probe's model has head dim 128): fused up to 128 from sm_70 on (sm_75: 64), PEFT's modules above
if "$PY" - "$OUT/logs/ranks.txt" "$CAP" > "$OUT/logs/ranks_fit.txt" 2>&1 <<'PY'
import re, sys
from poros.core.card import fused_rank_reason
cap = tuple(int(x) for x in sys.argv[2].split("."))
bad, seen = [], 0
for line in open(sys.argv[1]):
    m = re.match(r"RANK (\d+) OK fused=(\w+)", line)
    if m:
        seen += 1
        r, fused = int(m.group(1)), m.group(2) == "True"
        want = fused_rank_reason(cap, r, head_dim=128) is None
        if fused != want:
            bad.append(f"rank {r}: fused={fused}, the table says {want}")
print("ranks match the fit table" if seen == 6 and not bad else f"{seen} ranks; {bad}")
sys.exit(0 if seen == 6 and not bad else 1)
PY
then record PASS ranks-fit "$(tail -1 "$OUT/logs/ranks_fit.txt")"
else record FAIL ranks-fit "$(tail -1 "$OUT/logs/ranks_fit.txt" | cut -c1-220)"; fi

# --- refuse: what this card cannot run, refused before any download --------------------------
refuse_check() {  # refuse_check NAME "ENV" EXPECT CODE EXIT key=val...
  local n=$1 e=$2 expect=$3 code=$4 want=$5; shift 5
  "$PY" "$FUSED/mkcfg.py" "$OUT/cfg/$n.yaml" "$MODEL" synthetic "$@" > /dev/null
  local t0=$(date +%s)
  # the emulation (POROS_XCARD_CAP) is imported by this Python alone, before Poros
  env $e timeout 300 "$PY" -c "import sys; sys.path.insert(0, '$X'); import xcard_emulate
from poros.cli.main import cli; cli()" train "$OUT/cfg/$n.yaml" \
    --output-dir "$OUT/runs/$n" --yes --json > "$OUT/logs/$n.txt" 2>&1
  local rc=$? dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq "$want" ] && grep -q "$expect" "$OUT/logs/$n.txt" \
      && grep -Eq "\"code\": ?\"$code\"" "$OUT/logs/$n.txt" && [ $dt -le 120 ]; then
    record PASS "$n" "refused in ${dt}s, exit $rc $code: $(grep -o "$expect[^\"]*" "$OUT/logs/$n.txt" | head -1 | cut -c1-120)"
  else
    record FAIL "$n" "rc=$rc after ${dt}s; expected exit $want $code naming '$expect' (logs/$n.txt)"
  fi
}
if [ "$NATIVE_BF16" = 1 ]; then  # emulate a Turing card's choosers: bf16 must be refused
  refuse_check refuse-bf16-emu75 "POROS_XCARD_CAP=7.5" "needs an NVIDIA GPU with bf16" \
    E_GPU_UNSUPPORTED 4 torch_dtype=bfloat16
else
  refuse_check refuse-bf16 "" "needs an NVIDIA GPU with bf16" E_GPU_UNSUPPORTED 4 \
    torch_dtype=bfloat16
fi

# --- runs -------------------------------------------------------------------------------------
run() {  # run NAME "ENV" MODEL DATA key=val...  -> $OUT/runs/NAME
  local n=$1 e=$2 m=$3 d=$4; shift 4
  "$PY" "$FUSED/mkcfg.py" "$OUT/cfg/$n.yaml" "$m" "$d" "$@" > /dev/null
  rm -rf "$OUT/runs/$n"
  echo "$(date -u +%FT%TZ) run $n env=[$e] $*" >> "$OUT/logs/runs.log"
  gpu env $e timeout -k 30 1800 "$PY" -c "from poros.cli.main import cli; cli()" train \
    "$OUT/cfg/$n.yaml" --output-dir "$OUT/runs/$n" --yes --json > "$OUT/logs/$n.txt" 2>&1
  local rc=$?
  rm -rf "$OUT/runs/$n"/checkpoint-* "$OUT/runs/$n"/optimizer_state.pt
  return $rc
}
ran() {  # ran NAME -> "fused_compute=.. moe=.. reason=.." (the run's setup event, else its config)
  "$PY" - "$OUT/runs/$1" <<'PY' 2>/dev/null || echo "no setup event or config.resolved.json"
import json, os, sys
d = sys.argv[1]
c = json.load(open(os.path.join(d, "config.resolved.json")))
r = dict(c.get("ran_compute") or {}, compute_reason=c.get("compute_reason"))
try:
    for line in open(os.path.join(d, "events.jsonl")):
        e = json.loads(line)
        if e.get("event_type") == "setup":
            r.update(e["data"])
            break
except FileNotFoundError:
    pass
print(f"fused_compute={r.get('fused_compute')} moe={r.get('moe_expert_compute')} "
      f"dtype={c.get('torch_dtype')} path={c.get('ran_path') or c.get('path')} "
      f"reason={r.get('compute_reason')}")
PY
}
compare() {  # compare REF CAND -> 0 on bitwise equal
  "$PY" "$FUSED/compare_runs.py" "$OUT/runs/$1" "$OUT/runs/$2" "$OUT/cmp/$1__$2.json" \
    > "$OUT/cmp/$1__$2.txt" 2>&1
  grep -q '"pass": true' "$OUT/cmp/$1__$2.txt"
}
digest() { "$PY" -c "import json;print(json.load(open('$OUT/cmp/$1')).get('ref_digest',''))" 2>/dev/null; }

# A mode: sequential, efficient, fast and fast with every block resident, each compared to
# sequential. MODE ENV EXPECT_FUSED(1/0/-) key=val...
mode() {
  local mo=$1 e=$2 want=$3; shift 3
  local base="lora_dropout=0.05 block_size=4 per_device_batch_size=2 gradient_accumulation_steps=4 max_steps=8 seed=42"
  local ok=1 why="" p
  for p in seq:path=sequential eff:path=efficient fast:path=fast "res:path=fast resident_blocks=999"; do
    local tag=${p%%:*} kv=${p#*:}
    run "d_${mo}_$tag" "$e" "$MODEL" synthetic $base $kv "$@" || { ok=0; why="$why $tag:run-failed(logs/d_${mo}_$tag.txt)"; continue; }
    if [ "$tag" != seq ]; then compare "d_${mo}_seq" "d_${mo}_$tag" || { ok=0; why="$why $tag!=seq"; }; fi
  done
  local r; r=$(ran "d_${mo}_seq")
  if [ "$want" = 1 ] && ! grep -q "fused_compute=True" <<< "$r"; then ok=0; why="$why expected-fused"; fi
  if [ "$want" = 0 ] && ! grep -q "fused_compute=False" <<< "$r"; then ok=0; why="$why expected-unfused"; fi
  local d; d=$(digest "d_${mo}_seq__d_${mo}_fast.json")
  if [ $ok = 1 ]; then record PASS "dense-$mo" "seq==eff==fast==resident digest ${d:0:16} | $r"
  else record FAIL "dense-$mo" "$why | $r"; fi
}

D="torch_dtype=$DTYPE"
if [ "$TRITON" = 1 ]; then mode fused "" 1 $D; else record SKIP dense-fused "triton is not installed here"; fi
mode nofused "" 0 $D fused_compute=false
# the larger ranks' own tiles (64 and 128: smaller blocks, rows in chunks), fused where they fit
for R in 64 128; do
  if [ "$TRITON" = 1 ] && [ "$("$PY" -c "from poros.core.card import fused_rank_reason as f; print(int(f(tuple(map(int,'$CAP'.split('.'))), $R, head_dim=64) is None))")" = 1 ]; then
    mode "fused-r$R" "" 1 $D lora_rank=$R lora_alpha=$((2 * R))
  fi
done
if [ "$NATIVE_BF16" = 1 ]; then
  mode fp16 "" "$TRITON" torch_dtype=float16
  mode fp16-nofused "" 0 torch_dtype=float16 fused_compute=false
fi

# --- refcheck (fused vs PEFT on oasst1) -------------------------------------------------------
if [ "${XCARD_QUICK:-0}" = 1 ] || [ "$TRITON" != 1 ]; then
  record SKIP refcheck "XCARD_QUICK=1 or no triton"
elif "$PY" "$FUSED/make_rows.py" "$OUT/oasst_long2400.jsonl" > "$OUT/logs/make_rows.txt" 2>&1; then
  C="lora_dropout=0.05 learning_rate=1e-4 max_steps=100 eval_split=0.0 path=fast block_size=4 seed=42 $D"
  run rc_peft_mb1 "" "$MODEL" "$OUT/oasst_long2400.jsonl" $C fused_compute=false
  run rc_fused_mb1 "" "$MODEL" "$OUT/oasst_long2400.jsonl" $C
  run rc_peft_mb2 "" "$MODEL" "$OUT/oasst_long2400.jsonl" $C fused_compute=false per_device_batch_size=2 gradient_accumulation_steps=4
  if "$PY" "$X/refcheck.py" "$OUT/runs/rc_peft_mb1" "$OUT/runs/rc_fused_mb1" "$OUT/runs/rc_peft_mb2" \
       "$OUT/cmp/refcheck.json" > "$OUT/logs/refcheck.txt" 2>&1 && grep -q '"pass": true' "$OUT/logs/refcheck.txt"; then
    record PASS refcheck "$(tail -1 "$OUT/logs/refcheck.txt" | cut -c1-220)"
  else
    record FAIL refcheck "$(tail -1 "$OUT/logs/refcheck.txt" | cut -c1-220)"
  fi
else
  record FAIL refcheck "could not build the oasst1 rows (logs/make_rows.txt)"
fi

# --- MoE --------------------------------------------------------------------------------------
MOE="${XCARD_MOE_MODEL:-$OUT/tiny_qwen3_moe}"
MOE_OK=1
if [ -n "${XCARD_MOE_MODEL:-}" ]; then
  record PASS moe-build "XCARD_MOE_MODEL=$MOE"
elif "$PY" "$X/mk_tiny_moe.py" "$MOE" "$MODEL" > "$OUT/logs/mk_moe.txt" 2>&1; then
  record PASS moe-build "$(tail -1 "$OUT/logs/mk_moe.txt" | cut -c1-200)"
else
  MOE_OK=0
  record FAIL moe-build "the tiny Qwen3-MoE was not built: $(tail -1 "$OUT/logs/mk_moe.txt" | cut -c1-160) (logs/mk_moe.txt)"
fi
moe_mode() {  # moe_mode NAME ENV EXPECT_MOE key=val...
  local mo=$1 e=$2 want=$3; shift 3
  local ok=1 why="" p
  for p in seq:path=sequential eff:path=efficient fast:path=fast; do
    local tag=${p%%:*} kv=${p#*:}
    run "m_${mo}_$tag" "$e" "$MOE" synthetic lora_dropout=0.05 per_device_batch_size=2 \
      gradient_accumulation_steps=4 max_steps=6 max_seq_length=256 seed=42 lora_target_modules=null $kv "$@" \
      || { ok=0; why="$why $tag:run-failed(logs/m_${mo}_$tag.txt)"; continue; }
    if [ "$tag" != seq ]; then compare "m_${mo}_seq" "m_${mo}_$tag" || { ok=0; why="$why $tag!=seq"; }; fi
  done
  local r; r=$(ran "m_${mo}_seq")
  grep -q "moe=$want" <<< "$r" || { ok=0; why="$why expected-moe=$want"; }
  local d; d=$(digest "m_${mo}_seq__m_${mo}_fast.json")
  if [ $ok = 1 ]; then record PASS "moe-$mo" "seq==eff==fast digest ${d:0:16} | $r"
  else record FAIL "moe-$mo" "$why | $r"; fi
}
if [ "$MOE_OK" = 0 ]; then
  record SKIP moe-modes "no tiny MoE model (moe-build)"
elif [ "$NATIVE_BF16" = 1 ]; then
  if [ "$TRITON" = 1 ]; then moe_mode fused "" fused; moe_mode fused-r128 "" fused lora_rank=128 lora_alpha=256; fi
  moe_mode reference "" reference moe_expert_compute=reference
else
  moe_mode fp16 "" reference torch_dtype=float16
fi

echo "end $(date -u +%FT%TZ)" | tee -a "$RESULT"
echo "FAILS=$FAILS" | tee -a "$RESULT"
[ $FAILS -eq 0 ]
