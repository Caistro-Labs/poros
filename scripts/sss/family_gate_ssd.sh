#!/usr/bin/env bash
# family_gate_ssd.sh -- every validated MoE family on the SSD expert store (store: ssd), on THIS card.
#
#     bash scripts/sss/family_gate_ssd.sh [OUT_DIR]
#
# The SSD store's side of scripts/family_gate.sh: the same models (families.tsv, its MoE rows), the
# same rows (oasst_long2400.jsonl), the same workload (LoRA dropout 0.05, mb2 x 4, 8 updates, seed 42)
# and its helpers (scripts/family_gate/: fetch, mkcfg, compare_runs, resume_cmp), read from beside
# this script, never from the tree under test.
#
# Env: as family_gate.sh -- FG_TREE, POROS_PYTHON, POROS_GPU_LOCK, FG_FAMILIES (comma list of MoE
# families.tsv keys; default: every MoE row), FG_MODELS, FG_RUN_TIMEOUT, FG_PRUNE=1 (delete each cut
# model and its .sss once its family is done) -- plus FG_STORE, the store directory (default
# OUT/store; put it on the SSD under test).
#
# Per family (each run its own lock hold):
#   ssd==ram   path fast on the RAM store, then on the SSD store (its first run builds the .sss):
#              every update's loss and grad norm (==) and every adapter tensor (torch.equal)
#   repeat     the SSD run again, reusing the .sss: bitwise to the building run
#   paths      on the SSD store, sequential == efficient == fast, bitwise
#   ref        the reference experts (moe_expert_compute: reference), RAM vs SSD, bitwise (where the
#              default is the fused compute; elsewhere the default runs them already)
#   gate       scripts/moe_absolute_gate.py --loader streamed --store ssd: experts streamed from the
#              .sss vs every expert resident in VRAM (read from the same .sss), bitwise
#   resume     SSD store, 8 updates saving at 4; resumed from checkpoint-4: steps 5-8 and adapters
#              bitwise
# PASS / FAIL per check in OUT/RESULT.txt, every run's config, log, events and comparison under OUT;
# the s/update, peaks (GiB) and host RAM of the ram and ssd runs in OUT/speed.txt. Exit 0 iff no FAIL.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
H="$(cd "$HERE/.." && pwd)/family_gate"
TREE="$(cd "${FG_TREE:-$HERE/../..}" && pwd)"
CALLER="$(pwd)"
PY="${POROS_PYTHON:-python3}"
LOCK="${POROS_GPU_LOCK:-/tmp/poros-gpu.lock}"
MODELS="${FG_MODELS:-$HOME/.cache/poros-family-gate}"
RTO="${FG_RUN_TIMEOUT:-1200}"
touch "$LOCK"; mkdir -p "$MODELS"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 POROS_PROGRESS=0 PYTHONUNBUFFERED=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONPATH="$TREE"
cd "$TREE"

OUT="${1:-family-gate-ssd-$(date -u +%Y%m%dT%H%MZ)}"
case "$OUT" in /*) ;; *) OUT="$CALLER/$OUT" ;; esac
mkdir -p "$OUT"/{cfg,logs,runs,cmp} && OUT="$(cd "$OUT" && pwd)"
STORE="${FG_STORE:-$OUT/store}"; mkdir -p "$STORE"
RESULT="$OUT/RESULT.txt"; touch "$RESULT"
export TMPDIR="${TMPDIR:-$OUT/tmp}"; mkdir -p "$TMPDIR"
FAILS=0

record() { printf '%-4s  %-30s %s\n' "$1" "$2" "$3" | tee -a "$RESULT"; [ "$1" = FAIL ] && FAILS=$((FAILS + 1)); return 0; }
gpu() { flock "$LOCK" "$@"; }
jget() { "$PY" -c "import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):
    d = d.get(k) if isinstance(d, dict) else None
print(d)" "$@" 2>/dev/null; }

SHA=$(git -C "$TREE" rev-parse HEAD 2>/dev/null || cut -d' ' -f1 "$TREE/TREE_SHA" 2>/dev/null || echo unknown)
echo "family_gate_ssd: tree $TREE ($SHA) -> $OUT, store $STORE" | tee -a "$RESULT"
echo "start $(date -u +%FT%TZ)" | tee -a "$RESULT"
"$PY" -c "import os, poros, torch; t=os.path.realpath('$TREE')+os.sep
assert os.path.realpath(poros.__file__).startswith(t), poros.__file__
print(torch.cuda.get_device_name(0), torch.__version__)" > "$OUT/logs/env.txt" 2>&1 \
  && record PASS env "sha ${SHA:0:12} $(tail -1 "$OUT/logs/env.txt")" \
  || { record FAIL env "poros does not import from the tree under test, or no CUDA (logs/env.txt)"; exit 1; }

DATA="$MODELS/oasst_long2400.jsonl"
[ -s "$DATA" ] || "$PY" "$H/make_rows.py" "$DATA" > "$OUT/logs/make_rows.txt" 2>&1
[ -s "$DATA" ] && record PASS data "$(sha256sum "$DATA" | cut -c1-16) $DATA" \
  || { record FAIL data "oasst1 rows not built (logs/make_rows.txt)"; exit 1; }

RUNEXTRA=()
run() {  # run NAME MODEL [key=val...]: one `poros train`, its own lock hold
  local n=$1 m=$2; shift 2
  "$PY" "$H/mkcfg.py" "$OUT/cfg/$n.yaml" "$m" "$DATA" "$@" > /dev/null
  rm -rf "$OUT/runs/$n"; mkdir -p "$OUT/runs/$n"
  # the launch time is taken once the lock is held (load and T(1) count no wait for the GPU)
  gpu bash -c 'date +%s.%N > "$1"; shift; exec "$@"' _ "$OUT/runs/$n/t0" \
    timeout -k 30 "$RTO" "$PY" -c "from poros.cli.main import cli; cli()" train "$OUT/cfg/$n.yaml" \
    --output-dir "$OUT/runs/$n" --yes --json ${RUNEXTRA[@]+"${RUNEXTRA[@]}"} > "$OUT/logs/$n.txt" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $n rc=$rc" >> "$OUT/logs/runs.log"
  [ "${KEEP_CKPT:-0}" = 1 ] || rm -rf "$OUT/runs/$n"/checkpoint-*
  rm -f "$OUT/runs/$n"/optimizer_state.pt "$OUT/runs/$n"/tokenizer.json
  return $rc
}
compare() {  # compare REF CAND -> 0 on bitwise equal
  "$PY" "$H/compare_runs.py" "$OUT/runs/$1" "$OUT/runs/$2" "$OUT/cmp/$1__$2.json" > "$OUT/cmp/$1__$2.txt" 2>&1
  grep -q '"pass": true' "$OUT/cmp/$1__$2.txt"
}
digest() { jget "$OUT/cmp/$1__$2.json" ref_digest | cut -c1-16; }
speed() {  # speed RUN: s/update (median of 2..N), load, peaks (GiB), host RAM (GiB), what ran
  "$PY" - "$OUT/runs/$1" "$1" <<'PY' >> "$OUT/speed.txt" 2>&1
import json, os, statistics, sys
d, n = sys.argv[1:3]
ev = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
st = [e for e in ev if e.get("event_type") == "step"]
setup = next((e for e in ev if e.get("event_type") == "setup"), None)
t0 = float(open(os.path.join(d, "t0")).read())
s = [e["data"]["s_per_step"] for e in st]
r = json.load(open(os.path.join(d, "receipt.json")))
c = json.load(open(os.path.join(d, "config.resolved.json")))
g = lambda k: None if r.get(k) is None else round(r[k] * 1e9 / 2**30, 2)
print("SPEED", n, json.dumps({"s_med": round(statistics.median(s[1:]), 3) if len(s) > 1 else None,
      "load_s": round(setup["timestamp"] - t0, 1) if setup else None, "T1_s": round(st[0]["timestamp"] - t0, 1),
      "alloc_GiB": g("peak_vram_allocated_gb"), "reserved_GiB": g("peak_vram_reserved_gb"),
      "host_GiB": g("peak_host_ram_gb"), "store": c.get("store"), "path": c.get("ran_path"),
      "ran": c.get("ran_compute")}))
PY
}

family() {  # family KEY REPO REV MODE LAYERS EXPECT
  local k=$1 repo=$2 rev=$3 mode=$4 L=$5 exp=$6
  local m="$MODELS/$(echo "$repo" | tr '/' '_')__${mode}${L}"
  local st="$STORE/$k"
  echo "--- $k ($repo @ ${rev:0:8} $mode $L) $(date -u +%FT%TZ)" | tee -a "$RESULT"
  local tf="$MODELS/$(echo "$repo" | tr '/' '_')__trunc${L}"
  if FG_TEXT_FROM="$([ "$mode" = text ] && [ -s "$tf/fetch.json" ] && echo "$tf")" \
      timeout -k 30 7200 "$PY" "$H/fetch.py" "$m" "$repo" "$rev" "$mode" "$L" > "$OUT/logs/fetch_$k.txt" 2>&1; then
    record PASS "$k:fetch" "$(tail -1 "$OUT/logs/fetch_$k.txt" | cut -c1-160)"
  else record FAIL "$k:fetch" "$(tail -1 "$OUT/logs/fetch_$k.txt" | cut -c1-160)"; return; fi
  rm -rf "$st"; mkdir -p "$st"
  local B="lora_dropout=0.05 per_device_batch_size=2 gradient_accumulation_steps=4 max_steps=8 seed=42 lora_target_modules=null"
  local S="store=ssd store_path=$st"
  run "${k}_ram" "$m" $B path=fast; local r0=$?
  run "${k}_ssd" "$m" $B path=fast $S; local r1=$?   # builds the .sss
  run "${k}_ssd2" "$m" $B path=fast $S; local r2=$?  # reuses it
  for n in "${k}_ram" "${k}_ssd" "${k}_ssd2"; do [ -s "$OUT/runs/$n/receipt.json" ] && speed "$n"; done
  if [ $r0 = 0 ] && [ $r1 = 0 ] && compare "${k}_ram" "${k}_ssd"; then
    record PASS "$k:ssd==ram" "path fast, digest $(digest "${k}_ram" "${k}_ssd") | $(jget "$OUT/runs/${k}_ssd/config.resolved.json" ran_compute | cut -c1-150)"
  else record FAIL "$k:ssd==ram" "rc ram=$r0 ssd=$r1 (cmp/${k}_ram__${k}_ssd.txt)"; fi
  if [ $r2 = 0 ] && compare "${k}_ssd" "${k}_ssd2"; then
    record PASS "$k:repeat" "built .sss, then reused: bitwise, digest $(digest "${k}_ssd" "${k}_ssd2")"
  else record FAIL "$k:repeat" "rc=$r2 (cmp/${k}_ssd__${k}_ssd2.txt)"; fi
  local ok=1 why=""
  for p in seq:sequential eff:efficient; do
    run "${k}_ssd_${p%%:*}" "$m" $B path=${p#*:} $S || { ok=0; why="$why ${p%%:*}:rc"; continue; }
    compare "${k}_ssd" "${k}_ssd_${p%%:*}" || { ok=0; why="$why ${p%%:*}!=fast"; }
  done
  [ $ok = 1 ] && record PASS "$k:paths" "on the SSD store, seq==eff==fast digest $(digest "${k}_ssd" "${k}_ssd_seq")" \
    || record FAIL "$k:paths" "$why"
  if grep -q "moe=fused" <<< "$exp"; then
    run "${k}_ram_ref" "$m" $B path=fast moe_expert_compute=reference; local a=$?
    run "${k}_ssd_ref" "$m" $B path=fast moe_expert_compute=reference $S; local b=$?
    if [ $a = 0 ] && [ $b = 0 ] && compare "${k}_ram_ref" "${k}_ssd_ref"; then
      record PASS "$k:ref" "reference experts, ssd==ram digest $(digest "${k}_ram_ref" "${k}_ssd_ref")"
    else record FAIL "$k:ref" "rc ram=$a ssd=$b (cmp/${k}_ram_ref__${k}_ssd_ref.txt)"; fi
  else record SKIP "$k:ref" "the default runs the reference experts here ($exp)"; fi
  gpu timeout -k 30 "$RTO" "$PY" "$TREE/scripts/moe_absolute_gate.py" --model "$m" --loader streamed \
    --store ssd --store-path "$st" --lora-dropout 0.05 --seq-len 64 --steps 4 --out "$OUT/cmp/${k}_gate.json" \
    > "$OUT/logs/${k}_gate.txt" 2>&1; local grc=$?
  if [ $grc = 0 ] && [ "$(jget "$OUT/cmp/${k}_gate.json" parity.passed)" = True ] \
      && [ "$(jget "$OUT/cmp/${k}_gate.json" store)" = ssd ]; then
    record PASS "$k:gate" "streamed from the .sss == resident (losses, grad norms, adapters, eval CE)"
  else record FAIL "$k:gate" "rc=$grc parity=$(jget "$OUT/cmp/${k}_gate.json" parity.passed) $(tail -1 "$OUT/logs/${k}_gate.txt" | cut -c1-140)"; fi
  local R="$B path=fast save_steps=4 $S"
  KEEP_CKPT=1 run "${k}_resA" "$m" $R; local ra=$?
  rm -rf "$OUT/runs/${k}_resB"; mkdir -p "$OUT/runs/${k}_resB"
  if [ $ra = 0 ] && [ -d "$OUT/runs/${k}_resA/checkpoint-4" ]; then
    RUNEXTRA=(--resume-from "$OUT/runs/${k}_resA/checkpoint-4")
    run "${k}_resB" "$m" $R; local rb=$?
    RUNEXTRA=()
    if [ $rb = 0 ] && "$PY" "$H/resume_cmp.py" "$OUT/runs/${k}_resA" "$OUT/runs/${k}_resB" 4 "$OUT/cmp/${k}_resume.json" \
         > "$OUT/logs/${k}_resume.txt" 2>&1 && grep -q '"pass": true' "$OUT/logs/${k}_resume.txt"; then
      record PASS "$k:resume" "$(grep RESUME "$OUT/logs/${k}_resume.txt" | cut -c8-180)"
    else record FAIL "$k:resume" "rc=$rb $(grep RESUME "$OUT/logs/${k}_resume.txt" 2>/dev/null | cut -c8-180)"; fi
  else record FAIL "$k:resume" "run A rc=$ra or no checkpoint-4 (logs/${k}_resA.txt)"; fi
  rm -rf "$OUT/runs/${k}_resA"/checkpoint-* "$OUT/runs/${k}_resB"/checkpoint-*
  find "$OUT/runs" -maxdepth 2 -name adapter_weights.pt -path "*/${k}_*" -delete 2>/dev/null
  if [ "${FG_PRUNE:-0}" = 1 ]; then
    rm -rf "$st"
    if [ "$mode" != snapshot ] && ! { [ "$mode" = trunc ] && grep -qP "^[^#][^\t]*\tmoe\t$repo\t$rev\ttext\t$L\t" "$H/families.tsv" \
        && [ ! -s "$MODELS/$(echo "$repo" | tr '/' '_')__text${L}/fetch.json" ]; }; then
      rm -rf "$m"; [ "$mode" = text ] && rm -rf "$tf"
    fi
  fi
}

while IFS=$'\t' read -r k entry repo rev mode L tier exp _; do
  [[ -z "$k" || "$k" == \#* || "$entry" != moe ]] && continue
  [ -n "${FG_FAMILIES:-}" ] && [[ ",$FG_FAMILIES," != *",$k,"* ]] && continue
  family "$k" "$repo" "$rev" "$mode" "$L" "$exp"
done < "$H/families.tsv"
echo "end $(date -u +%FT%TZ) FAILS=$FAILS" | tee -a "$RESULT"
exit $((FAILS > 0))
