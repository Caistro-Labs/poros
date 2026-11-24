#!/usr/bin/env bash
# family_gate.sh -- every validated model family, dense and MoE, on THIS card; and "test any model" end to end.
#
#     bash scripts/family_gate.sh [OUT_DIR]
#
# Re-runnable unchanged on any tree: FG_TREE=/path/to/tree (default: the tree this script sits in) is the
# Poros under test (put first on PYTHONPATH; the gate checks `import poros` resolves inside it). The gate's own
# helpers are read from beside this script (scripts/family_gate/), never from the tree under test.
#
# Env:
#   POROS_PYTHON     the Python with Poros's training stack (default python3)
#   POROS_GPU_LOCK   flock file every GPU step holds, one run per hold (default /tmp/poros-gpu.lock)
#   FG_TSV           another table of rows in families.tsv's columns (default: family_gate/families.tsv), e.g.
#                    family_gate/catalog.tsv: one row per catalog model, the same checks on that model itself;
#                    each row's header line then ends with its notes column (the model the row stands for);
#   FG_DRY=1         print the header line of every row that would run, and stop: no fetch, no GPU;
#   FG_PLAN=1        print, row by row, every stage and run the gate would do (`PLAN ...` lines), running none of
#                    them: no fetch, no GPU, no Python (each result reads as empty); the any-model ladder is left
#                    out. What the gate does for a row can be compared before and after a change to this script;
#                    FG_PLAN_VISITS (comma list of keys) plans those rows as looped decoders;
#   FG_FAMILIES      comma list of families.tsv keys (default: all); FG_ANYMODEL=0 skips the any-model ladder;
#   FG_ANY           comma list of anymodel.tsv keys (default: all); FG_RESUME (default "qwen2,olmoe")
#   FG_MODELS        where models are fetched/cut (default $HOME/.cache/poros-family-gate)
#   FG_UPDATES       refcheck updates per arm (default 100); FG_RUN_TIMEOUT seconds per run (default 1200)
#   FG_KEEP_WEIGHTS=1 keeps adapter weights in OUT (default: deleted once compared; digests stay)
#   FG_PRUNE=1       deletes each cut model once its family is done (logged in OUT/DELETED.log);
#   FG_PRUNE_SNAPSHOTS=1  with FG_PRUNE: a whole-model row's snapshot leaves the HF cache too, once its row is done;
#   FG_MIN_FREE_GB   a fetch that would leave less free disk than this is refused (default 15)
#   FG_RESIDENT_EVIDENCE  for a row whose model does not fit this GPU with every block resident
#                    (a whole large model): where its streamed==resident evidence is; a resident
#                    run (paths' res arm, moe-gate, parity) that runs out of memory is then a SKIP
#                    naming it, not a FAIL. Unset (default): such a run's OOM fails, as any.
#                    Not for a looped decoder (visits.py): its paths compare sequential with
#                    sequential only, so its row FAILS unless moe-gate (streamed==resident) PASSES
#                    on this hardware: a SKIP there, or moe-gate not run, is a FAIL.
#   FG_PLAN_GPU_RC, FG_PLAN_GPU_SAYS  with FG_PLAN: what each planned GPU step exits with and
#                    prints (default 0, nothing), to plan a stage's verdict on a failure.
# Downloads public Hub files only, with no token (HF_HUB_DISABLE_IMPLICIT_TOKEN=1, token=False).
#
# Per family (families.tsv), with LoRA dropout 0.05 and the default compute (fused where it applies):
#   check    `poros check <dir> --json`: the registry type and status the model resolves to
#   paths    sequential == efficient == fast == fast with every block resident: every update's loss and grad
#            norm (==) and every adapter tensor (torch.equal), 8 updates of mb2 x 4 on oasst1 rows.
#            A decoder that runs its layers as visits (a looped one; it declares it, visits.py) trains on
#            the sequential path alone: there, path efficient and path fast must each be refused before
#            it trains (exit 2, E_CONFIG, the reason in its words), and path left unset must train
#            sequential with that reason as path_reason; repeat is the sequential run twice, and resume
#            always runs (sequential). Streamed against resident is its moe-gate, as for every MoE row.
#   repeat   fast run twice: bitwise
#   ran      every run's resolved compute and reasons (expect.py): the path asked ran (no path_reason), no
#            offload_reason, fused_compute / moe_expert_compute and compute_reason as families.tsv expects,
#            MoE expert cache: one layer on efficient and sequential, sized to the GPU on fast
#   refcheck (tier full) fused vs reference, FG_UPDATES updates mb1 x 8, lr 1e-4, the pre-declared rule of
#            docs/validation/fused: mean and max |dloss| <= the reference's own mb2-vs-mb1 noise mean and max
#            (dense: fused_compute false is the reference; MoE: moe_expert_compute reference)
#   moe-ref  (MoE, fused default) the reference experts: sequential == fast bitwise, moe=reference
#   moe-gate (MoE, tier full) scripts/moe_absolute_gate.py --loader streamed: streamed vs resident bitwise
#   resume   (FG_RESUME) 8 updates saving at 4; resumed from checkpoint-4: steps 5-8 and adapters bitwise
#   parity   (dense) the validation contract's training gate as maintainers run it,
#            scripts/training_parity.py (resident vs streamed, 8 steps, torch.equal)
# FG_CHECKS=paths,moeref,refcheck,moegate,resume,parity picks checks (default: all).
# Any model (anymodel.tsv): the ladder of docs/models.md on unvalidated families, and the refusals.
#
# Prints PASS / FAIL / SKIP per check to OUT/RESULT.txt; every run's config, log, events, resolved config,
# receipt, comparison and expectation record under OUT. Exit 0 iff no FAIL.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
H="$HERE/family_gate"
TREE="$(cd "${FG_TREE:-$HERE/..}" && pwd)"
CALLER="$(pwd)"
PY="${POROS_PYTHON:-python3}"
LOCK="${POROS_GPU_LOCK:-/tmp/poros-gpu.lock}"
MODELS="${FG_MODELS:-$HOME/.cache/poros-family-gate}"
TSV="${FG_TSV:-$H/families.tsv}"; case "$TSV" in /*) ;; *) TSV="$CALLER/$TSV" ;; esac
UPDATES="${FG_UPDATES:-100}"
RTO="${FG_RUN_TIMEOUT:-1200}"
touch "$LOCK"; mkdir -p "$MODELS"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 POROS_PROGRESS=0 PYTHONUNBUFFERED=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONPATH="$TREE"   # the tree under test, alone
cd "$TREE"                  # python -c puts the cwd first on sys.path: it must be the tree under test

GPU_TAG=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0).replace('NVIDIA ','').replace(' ','_'))" 2>/dev/null || echo nogpu)
VRAM_GIB=$("$PY" -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))" 2>/dev/null || echo "?")
resident_oom() {  # resident_oom LOG -> 0 (and the reason on stdout) where FG_RESIDENT_EVIDENCE names
  # the evidence and LOG ran out of GPU memory or was refused as not fitting
  [ -n "${FG_RESIDENT_EVIDENCE:-}" ] || return 1
  local line; line=$(grep -m1 -oE "E_OOM_PREDICTED.{0,120}|OutOfMemoryError: .{0,130}|PorosOOMError: .{0,120}" "$1") || return 1
  line=${line%%Including*}
  echo "does not fit ${GPU_TAG//_/ } (${VRAM_GIB} GiB) with every block resident: ${line% }; streamed==resident evidence = $FG_RESIDENT_EVIDENCE"
}
OUT="${1:-family-gate-${GPU_TAG}-$(date -u +%Y%m%dT%H%MZ)}"
case "$OUT" in /*) ;; *) OUT="$CALLER/$OUT" ;; esac
mkdir -p "$OUT"/{cfg,logs,runs,cmp,ran,any} && OUT="$(cd "$OUT" && pwd)"
RESULT="$OUT/RESULT.txt"; touch "$RESULT"
export TMPDIR="${TMPDIR:-$OUT/tmp}"; mkdir -p "$TMPDIR"
FAILS=0

record() {  # record STATUS NAME DETAIL
  printf '%-4s  %-34s %s\n' "$1" "$2" "$3" | tee -a "$RESULT"
  [ "$1" = FAIL ] && FAILS=$((FAILS + 1))
  return 0
}
header() {  # header KEY REPO REV MODE LAYERS NOTES: a row's line in RESULT; with FG_TSV it names what the row stands for
  echo "--- $1 ($2 @ ${3:0:8} $4 $5)$([ -n "${FG_TSV:-}" ] && [ -n "$6" ] && echo " | $6")"
}
want() { [ -z "${FG_CHECKS:-}" ] || [[ ",$FG_CHECKS," == *",$1,"* ]]; }  # FG_CHECKS: paths,moeref,refcheck,moegate,resume,parity
if [ "${FG_DRY:-0}" = 1 ]; then  # the rows that would run, by their header lines: no fetch, no GPU
  while IFS=$'\t' read -r k entry repo rev mode L tier exp notes; do
    case "$k" in ''|'#'*) continue ;; esac
    if [ -n "${FG_FAMILIES:-}" ] && [[ ",$FG_FAMILIES," != *",$k,"* ]]; then continue; fi
    header "$k" "$repo" "$rev" "$mode" "$L" "$notes"
  done < "$TSV"
  rm -f "$RESULT"; rmdir "$OUT"/{cfg,logs,runs,cmp,ran,any,tmp} "$OUT" 2>/dev/null
  exit 0
fi
gpu() { flock "$LOCK" "$@"; }
poros() { "$PY" -c "from poros.cli.main import cli; cli()" "$@"; }
jget() { "$PY" -c "import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):
    d = d.get(k) if isinstance(d, dict) else None
print(d)" "$@" 2>/dev/null; }
lastjson() {  # lastjson LOG OUTJSON: the last line of LOG that is a JSON object
  "$PY" - "$1" "$2" <<'PY'
import json, sys
last = None
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        try:
            last = json.loads(line)
        except ValueError:
            pass
json.dump(last, open(sys.argv[2], "w"), indent=1)
PY
}

echo "family_gate: tree $TREE -> $OUT" | tee -a "$RESULT"
echo "start $(date -u +%FT%TZ)" | tee -a "$RESULT"

PLAN="${FG_PLAN:-0}"
if [ "$PLAN" = 1 ]; then PY=true; fi  # every Python step runs as `true`: succeeds, prints nothing
# --- env and tree -------------------------------------------------------------------------------
if [ "$PLAN" != 1 ]; then
SHA=$(git -C "$TREE" rev-parse HEAD 2>/dev/null || cat "$TREE/TREE_SHA" 2>/dev/null || echo unknown)
DIRTY=$(git -C "$TREE" status --porcelain --untracked-files=no 2>/dev/null | wc -l)
"$PY" - "$OUT/env.json" "$TREE" "$SHA" <<'PY' > "$OUT/logs/env.txt" 2>&1
import json, os, platform, sys
import torch, transformers, peft, bitsandbytes
try:
    import triton; tv = triton.__version__
except ImportError:
    tv = None
import poros
out, tree, sha = sys.argv[1:4]
p = torch.cuda.get_device_properties(0)
env = {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
       "vram_gib": round(p.total_memory / 2**30, 2), "driver": None, "torch": torch.__version__,
       "cuda": torch.version.cuda, "transformers": transformers.__version__, "peft": peft.__version__,
       "bitsandbytes": bitsandbytes.__version__, "triton": tv, "python": platform.python_version(),
       "poros": getattr(poros, "__version__", None), "poros_file": poros.__file__, "tree": tree, "tree_sha": sha,
       "poros_in_tree": os.path.realpath(poros.__file__).startswith(os.path.realpath(tree) + os.sep),
       "host_ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
       "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF")}
try:
    import subprocess
    env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.strip()
except OSError:
    pass
json.dump(env, open(out, "w"), indent=1)
print(json.dumps(env))
PY
if [ "$(jget "$OUT/env.json" poros_in_tree)" = True ]; then
  record PASS env "sha ${SHA:0:12} dirty=$DIRTY $(tail -1 "$OUT/logs/env.txt" | cut -c1-260)"
else
  record FAIL env "poros does not import from the tree under test, or no CUDA (logs/env.txt)"; cat "$RESULT"; exit 1
fi

# --- data ---------------------------------------------------------------------------------------
DATA="$MODELS/oasst_long2400.jsonl"
if [ ! -s "$DATA" ]; then "$PY" "$H/make_rows.py" "$DATA" > "$OUT/logs/make_rows.txt" 2>&1; fi
if [ -s "$DATA" ]; then record PASS data "$(sha256sum "$DATA" | cut -c1-16) $DATA"
else record FAIL data "oasst1 rows not built (logs/make_rows.txt)"; cat "$RESULT"; exit 1; fi
fi  # PLAN

# --- one run --------------------------------------------------------------------------------------
# run NAME MODEL [key=val...]: one `poros train` (its own lock hold), receipts kept under OUT/runs/NAME
RUNEXTRA=()
run() {
  local n=$1 m=$2; shift 2
  "$PY" "$H/mkcfg.py" "$OUT/cfg/$n.yaml" "$m" "$DATA" "$@" > /dev/null
  rm -rf "$OUT/runs/$n"
  echo "$(date -u +%FT%TZ) run $n $* ${RUNEXTRA[*]:-}" >> "$OUT/logs/runs.log"
  local t0; t0=$(date +%s)
  gpu timeout -k 30 "$RTO" "$PY" -c "from poros.cli.main import cli; cli()" train "$OUT/cfg/$n.yaml" \
    --output-dir "$OUT/runs/$n" --yes --json ${RUNEXTRA[@]+"${RUNEXTRA[@]}"} > "$OUT/logs/$n.txt" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $n rc=$rc $(( $(date +%s) - t0 ))s" >> "$OUT/logs/runs.log"
  [ "${KEEP_CKPT:-0}" = 1 ] || rm -rf "$OUT/runs/$n"/checkpoint-*
  rm -f "$OUT/runs/$n"/optimizer_state.pt "$OUT/runs/$n"/tokenizer.json  # the model's own tokenizer: not a receipt
  return $rc
}
compare() {  # compare REF CAND -> 0 on bitwise equal
  "$PY" "$H/compare_runs.py" "$OUT/runs/$1" "$OUT/runs/$2" "$OUT/cmp/$1__$2.json" > "$OUT/cmp/$1__$2.txt" 2>&1
  grep -q '"pass": true' "$OUT/cmp/$1__$2.txt"
}
digest() { jget "$OUT/cmp/$1" ref_digest; }
expect() {  # expect RUN SPEC... -> 0 when the run ran what SPEC says
  local r=$1; shift
  "$PY" "$H/expect.py" "$OUT/runs/$r" "$OUT/logs/$r.txt" "$OUT/ran/$r.json" "$@" > "$OUT/ran/$r.txt" 2>&1
}
fails_of() { "$PY" -c "import json;print('; '.join(json.load(open('$OUT/ran/$1.json'))['fails'])[:300])" 2>/dev/null; }
ranline() { "$PY" -c "import json;d=json.load(open('$OUT/ran/$1.json'))
print(' '.join(f'{k}={d.get(k)}' for k in ('ran_path','fused_compute','moe_expert_compute','moe_expert_cache_gib','compute_reason')))[:300]" 2>/dev/null; }
strip_weights() { [ "${FG_KEEP_WEIGHTS:-0}" = 1 ] || find "$OUT/runs" -maxdepth 3 \( -name 'adapter_model.safetensors' -o -name 'adapter_weights.pt' \) -path "*/$1*" -delete 2>/dev/null; }

# --- a family -------------------------------------------------------------------------------------
family() {  # family KEY ENTRY REPO REV MODE LAYERS TIER EXPECT NOTES
  local k=$1 entry=$2 repo=$3 rev=$4 mode=$5 L=$6 tier=$7 exp=$8 notes=${9:-}
  local slug; slug="$(echo "$repo" | tr '/' '_')__${mode}${L}"
  local m="$MODELS/$slug"
  echo "$(header "$k" "$repo" "$rev" "$mode" "$L" "$notes") $(date -u +%FT%TZ)" | tee -a "$RESULT"
  # fetch (no GPU): a text model is cut from its trunc sibling when that is here
  local tf="$MODELS/$(echo "$repo" | tr '/' '_')__trunc${L}"
  if FG_TEXT_FROM="$([ "$mode" = text ] && [ -s "$tf/fetch.json" ] && echo "$tf")" \
      timeout -k 30 7200 "$PY" "$H/fetch.py" "$m" "$repo" "$rev" "$mode" "$L" > "$OUT/logs/fetch_$k.txt" 2>&1; then
    record PASS "$k:fetch" "$(tail -1 "$OUT/logs/fetch_$k.txt" | cut -c1-200)"
  else
    record FAIL "$k:fetch" "$(tail -1 "$OUT/logs/fetch_$k.txt" | cut -c1-200)"; return
  fi
  # check: what the registry says of it (config only)
  poros check "$m" --json > "$OUT/logs/check_$k.txt" 2>&1; local crc=$?
  lastjson "$OUT/logs/check_$k.txt" "$OUT/any/check_$k.json"
  local ct cs; ct=$(jget "$OUT/any/check_$k.json" model_type); cs=$(jget "$OUT/any/check_$k.json" status)
  if [ $crc = 0 ] && [ "$cs" = validated ]; then record PASS "$k:check" "model_type=$ct status=$cs"
  else record FAIL "$k:check" "rc=$crc model_type=$ct status=$cs (logs/check_$k.txt)"; fi

  local BS="block_size=4"; [ "$entry" = moe ] && BS=""   # the MoE path refuses block_size (it streams per expert)
  local B="lora_dropout=0.05 $BS per_device_batch_size=2 gradient_accumulation_steps=4 max_steps=8 seed=42 lora_target_modules=null"
  local p ok=1 why="" tag kv rskip=""
  # A decoder that runs its layers as visits (a looped one) says so itself; visits.py prints the
  # words Poros gives for training it on the sequential path alone, else nothing.
  local vwhy; vwhy=$("$PY" "$H/visits.py" "$m" 2>"$OUT/logs/visits_$k.txt") || {
    record FAIL "$k:visits" "could not tell whether its decoder runs as visits (logs/visits_$k.txt)"; return; }
  if [ "$PLAN" = 1 ] && [[ ",${FG_PLAN_VISITS:-}," == *",$k,"* ]]; then vwhy="(its sequential reason)"; fi
  if want paths && [ -n "$vwhy" ]; then  # block-major is refused for it: that refusal is the check
    local refok=1
    for p in eff:path=efficient fast:path=fast; do
      tag=${p%%:*}; kv=${p#*:}
      run "${k}_$tag" "$m" $B $kv; local frc=$?
      if "$PY" "$H/refused.py" "$OUT/logs/${k}_$tag.txt" "$OUT/runs/${k}_$tag" "$frc" "$vwhy" \
           > "$OUT/ran/${k}_$tag.txt" 2>&1; then
        record PASS "$k:refused:$tag" "$kv refused before it trained: exit 2 E_CONFIG, the reason in words"
      else refok=0; record FAIL "$k:refused:$tag" "$(tail -1 "$OUT/ran/${k}_$tag.txt" | cut -c1-220)"; fi
    done
    for tag in seq rep; do  # path left unset: the run decides the sequential path, and says why
      run "${k}_$tag" "$m" $B; local src=$?
      if [ $src != 0 ]; then ok=0; why="$why $tag:rc$src(logs/${k}_$tag.txt)"; continue; fi
      if expect "${k}_$tag" ran=sequential "path_reason~$vwhy" offload=none $exp cache=sequential; then :
      else ok=0; record FAIL "$k:ran:$tag" "$(fails_of "${k}_$tag")"; fi
    done
    if [ $ok = 1 ] && [ $refok = 1 ]; then
      record PASS "$k:paths" "a looped decoder: efficient and fast refused in words, unset trains sequential | $(ranline "${k}_seq")"
    else record FAIL "$k:paths" "refused_ok=$refok$why | $(ranline "${k}_seq" || true)"; fi
    if [ -d "$OUT/runs/${k}_rep" ] && compare "${k}_seq" "${k}_rep"; then
      record PASS "$k:repeat" "sequential twice bitwise, digest $(digest "${k}_seq__${k}_rep.json" | cut -c1-16)"
    else record FAIL "$k:repeat" "sequential run-to-run differs or did not run (cmp/${k}_seq__${k}_rep.txt)"; fi
    [ $ok = 1 ] && record PASS "$k:ran" "every run: sequential, for its reason; no offload_reason, $exp, expert cache one layer"
  fi
  if want paths && [ -z "$vwhy" ]; then
  for p in seq:path=sequential eff:path=efficient fast:path=fast "res:path=fast resident_blocks=999" rep:path=fast; do
    tag=${p%%:*}; kv=${p#*:}
    run "${k}_$tag" "$m" $B $kv; local rrc=$?
    if [ $rrc != 0 ] && [ $tag = res ] && rskip=$(resident_oom "$OUT/logs/${k}_res.txt"); then
      record SKIP "$k:paths-resident" "$rskip"; continue
    fi
    if [ $rrc != 0 ]; then ok=0; why="$why $tag:rc$rrc(logs/${k}_$tag.txt)"; continue; fi
    local pa=${kv%% *}; pa=${pa#path=}
    local cache=""
    if [ "$entry" = moe ]; then
      case $tag in seq) cache=cache=sequential;; eff) cache=cache=efficient;; fast|rep) cache=cache=sized;; esac
    fi
    if expect "${k}_$tag" "path=$pa" offload=none $exp $cache; then :
    else ok=0; record FAIL "$k:ran:$tag" "$(fails_of "${k}_$tag")"; fi
  done
  local cmpok=1 cwhy=""
  for p in eff fast res; do
    [ $p = res ] && [ -n "${rskip:-}" ] && continue  # said above: a SKIP, with its evidence
    [ -d "$OUT/runs/${k}_$p" ] && [ -d "$OUT/runs/${k}_seq" ] || { cmpok=0; continue; }
    compare "${k}_seq" "${k}_$p" || { cmpok=0; cwhy="$cwhy $p!=seq"; }
  done
  local d; d=$(digest "${k}_seq__${k}_fast.json")
  if [ $cmpok = 1 ] && [ -n "$d" ]; then
    record PASS "$k:paths" "seq==eff==fast$([ -n "${rskip:-}" ] && echo " (resident: SKIP)" || echo "==resident") digest ${d:0:16} | $(ranline "${k}_fast")"
  else record FAIL "$k:paths" "$why$cwhy | $(ranline "${k}_fast" || true)"; fi
  if [ -d "$OUT/runs/${k}_rep" ] && compare "${k}_fast" "${k}_rep"; then
    record PASS "$k:repeat" "fast twice bitwise, digest ${d:0:16}"
  else record FAIL "$k:repeat" "fast run-to-run differs or did not run (cmp/${k}_fast__${k}_rep.txt)"; fi
  local cw=""; [ "$entry" = moe ] && cw=", expert cache one layer on seq+eff, sized on fast"
  [ $ok = 1 ] && record PASS "$k:ran" "every run: the path asked ran, no offload_reason, $exp$cw"
  fi

  if want moeref && [ "$entry" = moe ] && grep -q "moe=fused" <<< "$exp"; then  # the reference experts on the same model
    local rok=1
    run "${k}_mref_seq" "$m" $B path=sequential moe_expert_compute=reference || rok=0
    run "${k}_mref_fast" "$m" $B path=fast moe_expert_compute=reference || rok=0
    if [ $rok = 1 ] && compare "${k}_mref_seq" "${k}_mref_fast" && expect "${k}_mref_fast" path=fast moe=reference reason=none; then
      record PASS "$k:moe-ref" "reference experts seq==fast digest $(digest "${k}_mref_seq__${k}_mref_fast.json" | cut -c1-16)"
    else record FAIL "$k:moe-ref" "rc/compare/expect failed (cmp/${k}_mref_seq__${k}_mref_fast.txt, ran/${k}_mref_fast.json)"; fi
  fi

  if [ "$tier" = full ] && want refcheck; then
    local C="lora_dropout=0.05 learning_rate=1e-4 max_steps=$UPDATES eval_split=0.0 path=fast $BS seed=42 lora_target_modules=null"
    local REF="fused_compute=false"; [ "$entry" = moe ] && REF="moe_expert_compute=reference"
    if [ "$entry" = moe ] && ! grep -q "moe=fused" <<< "$exp"; then
      record SKIP "$k:refcheck" "the default is the reference experts here (fused cannot run them: $(jget "$OUT/ran/${k}_fast.json" compute_reason | cut -c1-120))"
    else
      local rcok=1
      run "${k}_rc_ref1" "$m" $C $REF || rcok=0
      run "${k}_rc_fused1" "$m" $C || rcok=0
      run "${k}_rc_ref2" "$m" $C $REF per_device_batch_size=2 gradient_accumulation_steps=4 || rcok=0
      if [ $rcok = 1 ] && "$PY" "$H/refcheck.py" "$OUT/runs/${k}_rc_ref1" "$OUT/runs/${k}_rc_fused1" "$OUT/runs/${k}_rc_ref2" \
           "$OUT/cmp/${k}_refcheck.json" > "$OUT/logs/${k}_refcheck.txt" 2>&1 && grep -q '"pass": true' "$OUT/logs/${k}_refcheck.txt"; then
        record PASS "$k:refcheck" "$(tail -1 "$OUT/logs/${k}_refcheck.txt" | cut -c10-230)"
      else record FAIL "$k:refcheck" "rc_ok=$rcok $(tail -1 "$OUT/logs/${k}_refcheck.txt" 2>/dev/null | cut -c10-230)"; fi
    fi
  fi
  if [ "$tier" = full ] && want moegate; then
    if [ "$entry" = moe ]; then  # streamed vs resident, poros.moe's own gate
      gpu timeout -k 30 "$RTO" "$PY" "$TREE/scripts/moe_absolute_gate.py" --model "$m" --loader streamed \
        --lora-dropout 0.05 --seq-len 64 --steps 4 --out "$OUT/cmp/${k}_moe_gate.json" > "$OUT/logs/${k}_moe_gate.txt" 2>&1
      local grc=$?
      local gskip
      if [ $grc != 0 ] && gskip=$(resident_oom "$OUT/logs/${k}_moe_gate.txt") && [ -n "$vwhy" ]; then
        # a looped decoder's only streamed==resident comparison on this hardware: no SKIP
        record FAIL "$k:moe-gate" "a looped decoder's row needs streamed==resident here (its paths compare sequential with sequential): $gskip"
      elif [ $grc != 0 ] && gskip=$(resident_oom "$OUT/logs/${k}_moe_gate.txt"); then
        record SKIP "$k:moe-gate" "$gskip"
      elif [ $grc = 0 ] && [ "$(jget "$OUT/cmp/${k}_moe_gate.json" parity.passed)" = True ]; then
        record PASS "$k:moe-gate" "streamed==resident (losses, grad norms, adapters, eval CE) $(grep -o 'streamed_over_resident[^,]*' "$OUT/cmp/${k}_moe_gate.json" | head -1)"
      else record FAIL "$k:moe-gate" "rc=$grc parity=$(jget "$OUT/cmp/${k}_moe_gate.json" parity.passed) $(tail -1 "$OUT/logs/${k}_moe_gate.txt" | cut -c1-160)"; fi
    fi
  elif [ -n "$vwhy" ]; then  # a looped decoder's row is not done without it
    record FAIL "$k:moe-gate" "not run (tier $tier, FG_CHECKS ${FG_CHECKS:-all}): a looped decoder's row needs streamed==resident on this hardware"
  fi

  if [ "$entry" = dense ] && want parity; then  # the validation contract's training gate, as maintainers run it
    gpu timeout -k 30 "$RTO" env POROS_USER_ASSERTED_ARCHS= "$PY" "$TREE/scripts/training_parity.py" \
      --model "$m" --block-size 4 --g 1 --steps 8 --seq-len 256 --lora-dropout 0.0 --out "$OUT/cmp/${k}_parity.jsonl" \
      > "$OUT/logs/${k}_parity.txt" 2>&1; local prc=$?
    local pj; pj=$("$PY" -c "import json,sys
try:
    d=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
except Exception as e:
    print('none'); sys.exit()
print(d.get('all_equal'), 'loss_max_abs_diff=%s grads_max_abs_diff=%s first_losses=%s/%s' % (d.get('loss_max_abs_diff'), d.get('grads_max_abs_diff'), (d.get('resident_losses') or [None])[0], (d.get('blockwise_losses') or [None])[0]))" "$OUT/cmp/${k}_parity.jsonl")
    local pskip
    if [ $prc != 0 ] && pskip=$(resident_oom "$OUT/logs/${k}_parity.txt"); then record SKIP "$k:parity" "$pskip"
    elif [ $prc = 0 ] && [ "${pj%% *}" = True ]; then record PASS "$k:parity" "training gate (scripts/training_parity.py, 8 steps): resident==streamed ${pj#* }"
    else record FAIL "$k:parity" "rc=$prc all_equal=${pj%% *} ${pj#* } (logs/${k}_parity.txt)"; fi
  fi
  # a looped decoder is always resumed (FG_RESUME or not), on the sequential path it trains on
  if want resume && { [[ ",${FG_RESUME:-qwen2,olmoe}," == *",$k,"* ]] || [ -n "$vwhy" ]; }; then
    local R="lora_dropout=0.05 $BS per_device_batch_size=2 gradient_accumulation_steps=4 max_steps=8 seed=42 lora_target_modules=null path=fast save_steps=4"
    [ -n "$vwhy" ] && R="${R/ path=fast/}"
    KEEP_CKPT=1 run "${k}_resA" "$m" $R; local ra=$?
    rm -rf "$OUT/runs/${k}_resB"; mkdir -p "$OUT/runs/${k}_resB"
    if [ $ra = 0 ] && [ -d "$OUT/runs/${k}_resA/checkpoint-4" ]; then
      RUNEXTRA=(--resume-from "$OUT/runs/${k}_resA/checkpoint-4")
      run "${k}_resB" "$m" $R; local rb=$?
      RUNEXTRA=()
      if [ $rb = 0 ] && "$PY" "$H/resume_cmp.py" "$OUT/runs/${k}_resA" "$OUT/runs/${k}_resB" 4 "$OUT/cmp/${k}_resume.json" \
           > "$OUT/logs/${k}_resume.txt" 2>&1 && grep -q '"pass": true' "$OUT/logs/${k}_resume.txt"; then
        record PASS "$k:resume" "$(grep RESUME "$OUT/logs/${k}_resume.txt" | cut -c8-220)"
      else record FAIL "$k:resume" "rc=$rb $(grep RESUME "$OUT/logs/${k}_resume.txt" 2>/dev/null | cut -c8-220)"; fi
    else record FAIL "$k:resume" "run A rc=$ra or no checkpoint-4 (logs/${k}_resA.txt)"; fi
    rm -rf "$OUT/runs/${k}_resA"/checkpoint-* "$OUT/runs/${k}_resB"/checkpoint-*
  fi
  strip_weights "$k"
  if [ "${FG_PRUNE:-0}" = 1 ] && [ "${FG_PRUNE_SNAPSHOTS:-0}" = 1 ] && [ "$mode" = snapshot ] && [ -L "$m" ]; then
    local snap repo_dir; snap=$(readlink "$m"); repo_dir=$(dirname "$(dirname "$snap")")  # .../models--Org--Name
    case "$(basename "$repo_dir")" in models--*)
      echo "$(date -u +%FT%TZ) deleted $repo_dir ($(du -sh "$repo_dir" 2>/dev/null | cut -f1)) after $k" >> "$OUT/DELETED.log"
      rm -rf "$repo_dir" "$m" "$m.fetch.json" ;;
    esac
  fi
  if [ "${FG_PRUNE:-0}" = 1 ] && [ "$mode" != snapshot ]; then  # this gate's own cut model, rebuilt on demand
    local keep_trunc=0  # a trunc whose text sibling is still to come stays until that is cut
    [ "$mode" = trunc ] && grep -qP "^[^#][^\t]*\t[^\t]*\t$repo\t$rev\ttext\t$L\t" "$TSV" \
      && [ ! -s "$MODELS/$(echo "$repo" | tr '/' '_')__text${L}/fetch.json" ] && keep_trunc=1
    if [ $keep_trunc = 0 ]; then
      echo "$(date -u +%FT%TZ) deleted $m ($(du -sh "$m" 2>/dev/null | cut -f1)) after $k" >> "$OUT/DELETED.log"
      rm -rf "$m"
      [ "$mode" = text ] && [ -d "$tf" ] && { echo "$(date -u +%FT%TZ) deleted $tf ($(du -sh "$tf" | cut -f1)) after $k" >> "$OUT/DELETED.log"; rm -rf "$tf"; }
    fi
  fi
}

if [ "$PLAN" = 1 ]; then  # say each step instead of taking it (FG_PLAN)
  run() { echo "PLAN run $1 ${*:3} ${RUNEXTRA[*]:-}"; return 0; }
  compare() { echo "PLAN compare $1 $2"; return 0; }
  expect() { echo "PLAN expect $*"; return 0; }
  gpu() { local a="$*"; a=${a//$OUT/OUT}; a=${a//$MODELS/MODELS}; echo "PLAN gpu ${a//$TREE/TREE}"
          [ -z "${FG_PLAN_GPU_SAYS:-}" ] || echo "$FG_PLAN_GPU_SAYS"; return "${FG_PLAN_GPU_RC:-0}"; }
fi

while IFS=$'\t' read -r k entry repo rev mode L tier exp notes; do
  case "$k" in ''|'#'*) continue ;; esac
  if [ -n "${FG_FAMILIES:-}" ] && [[ ",$FG_FAMILIES," != *",$k,"* ]]; then continue; fi
  applies=0  # a family none of the chosen checks apply to is not fetched
  for c in paths refcheck resume $([ "$entry" = dense ] && echo parity || echo moeref moegate); do want $c && applies=1; done
  [ $applies = 1 ] || continue
  family "$k" "$entry" "$repo" "$rev" "$mode" "$L" "$tier" "$exp" "$notes"
done < "$TSV"

# --- any model ------------------------------------------------------------------------------------
guarded() {  # guarded LOGBASE poros-args...: one poros command in a fresh HF cache, stopped at 200 MB of it (guard.py)
  local base=$1; shift
  rm -rf "$OUT/tmp/hf_guard"
  gpu "$PY" "$H/guard.py" "$OUT/tmp/hf_guard" 200 300 "$base.guard.json" -- \
    "$PY" -c "from poros.cli.main import cli; cli()" "$@" > "$base.txt" 2>&1
}
any_ladder() {  # any_ladder KEY REPO REV
  local k=$1 repo=$2 rev=$3 A="$OUT/any/$k"; mkdir -p "$A"
  echo "--- any:$k ($repo @ ${rev:0:8}) $(date -u +%FT%TZ)" | tee -a "$RESULT"
  # rung 0a: poros check -- not validated (not in the registry, or a dense roadmap target), exit 4,
  # says how to run it once
  poros check "$repo" --json > "$A/check.txt" 2>&1; local rc=$?; lastjson "$A/check.txt" "$A/check.json"
  local mt st; mt=$(jget "$A/check.json" model_type); st=$(jget "$A/check.json" status)
  if [ $rc = 4 ] && { [ "$st" = not-in-registry ] || [ "$st" = target ]; }; then
    record PASS "any:$k:check" "exit 4 model_type=$mt status=$st"
    grep -q -- "--unvalidated" "$A/check.txt" || record NOTE "any:$k:check-json" "--json names no way on (reason \"$(jget "$A/check.json" reason)\", no next): the formatted view says poros train $repo --unvalidated"
  else record FAIL "any:$k:check" "rc=$rc model_type=$mt status=$(jget "$A/check.json" status) (any/$k/check.txt)"; fi
  # rung 0: agents arch
  poros agents arch "$repo" > "$A/arch.txt" 2>&1; rc=$?; lastjson "$A/arch.txt" "$A/arch.json"
  if [ $rc = 0 ] && [ "$(jget "$A/arch.json" status)" != validated ] && grep -q "arch try" "$A/arch.json"; then
    record PASS "any:$k:arch" "status=$(jget "$A/arch.json" status) weights.usable=$(jget "$A/arch.json" weights.usable) next=$(jget "$A/arch.json" next | cut -c1-80)"
  else record FAIL "any:$k:arch" "rc=$rc status=$(jget "$A/arch.json" status) (any/$k/arch.txt)"; fi
  # rung 1: agents arch try (CPU)
  timeout -k 30 900 "$PY" -c "from poros.cli.main import cli; cli()" agents arch try "$repo" > "$A/try.txt" 2>&1; rc=$?; lastjson "$A/try.txt" "$A/try.json"
  if [ $rc = 0 ] && [ "$(jget "$A/try.json" tried.ok)" = True ]; then
    record PASS "any:$k:arch-try" "trainer.ok=$(jget "$A/try.json" tried.routes.trainer.ok) bitwise=$(jget "$A/try.json" tried.routes.trainer.bitwise) next=$(jget "$A/try.json" next | cut -c1-60)"
  else record FAIL "any:$k:arch-try" "rc=$rc ok=$(jget "$A/try.json" tried.ok) $(jget "$A/try.json" tried.first_difference | cut -c1-120) (any/$k/try.txt)"; fi
  # rung 2a: prove without the flag asks (exit 2, needs_input, --unvalidated)
  poros agents prove "$repo" > "$A/prove_ask.txt" 2>&1; rc=$?; lastjson "$A/prove_ask.txt" "$A/prove_ask.json"
  if [ $rc = 2 ] && grep -q "needs_input" "$A/prove_ask.txt" && grep -q -- "--unvalidated" "$A/prove_ask.txt"; then
    record PASS "any:$k:prove-asks" "exit 2 $(jget "$A/prove_ask.json" schema)"
  else record FAIL "any:$k:prove-asks" "rc=$rc (any/$k/prove_ask.txt)"; fi
  # rung 2: prove --unvalidated (both GPU gates)
  local t0; t0=$(date +%s)
  gpu timeout -k 30 "$RTO" "$PY" -c "from poros.cli.main import cli; cli()" agents prove "$repo" --unvalidated --steps 20 --download > "$A/prove.txt" 2>&1; rc=$?
  lastjson "$A/prove.txt" "$A/prove.json"
  local pl pt; pl=$(jget "$A/prove.json" load.identical); pt=$(jget "$A/prove.json" training.identical)
  if [ $rc = 0 ] && [ "$(jget "$A/prove.json" passed)" = True ] && [ "$(jget "$A/prove.json" support)" = user_asserted ]; then
    record PASS "any:$k:prove" "load.identical=$pl ($(jget "$A/prove.json" load.tensors_compared) tensors) training.identical=$pt support=user_asserted $(( $(date +%s) - t0 ))s"
  else record FAIL "any:$k:prove" "rc=$rc passed=$(jget "$A/prove.json" passed) load=$pl training=$pt support=$(jget "$A/prove.json" support) differs=$(jget "$A/prove.json" differs | cut -c1-120) (any/$k/prove.txt)"; fi
  # train: refused without the flag (exit 2 asks), then --unvalidated, then POROS_USER_ASSERTED_ARCHS
  local T="max_steps=4 max_seq_length=256 per_device_batch_size=1 gradient_accumulation_steps=2 lora_target_modules=null model_revision=$rev"
  "$PY" "$H/mkcfg.py" "$OUT/cfg/any_$k.yaml" "$repo" synthetic $T > /dev/null
  guarded "$A/train_noflag" train "$OUT/cfg/any_$k.yaml" --output-dir "$A/run_noflag" --yes --json; rc=$?
  local g="$A/train_noflag.guard.json"
  if [ $rc != 0 ] && [ "$(jget "$g" killed)" = None ] && [ "$(jget "$g" weight_files)" = 0 ] && grep -q -- "--unvalidated" "$A/train_noflag.txt"; then
    record PASS "any:$k:train-noflag" "refused, exit $rc in $(jget "$g" seconds)s, no weights fetched: $(grep -o '"code": *"[A-Z_]*"\|needs_input' "$A/train_noflag.txt" | head -1), asks --unvalidated"
  else record FAIL "any:$k:train-noflag" "rc=$rc $(cat "$g" 2>/dev/null) (any/$k/train_noflag.txt)"; fi
  gpu timeout -k 30 "$RTO" "$PY" -c "from poros.cli.main import cli; cli()" train "$OUT/cfg/any_$k.yaml" --output-dir "$A/run_unval" --yes --json --unvalidated > "$A/train_unval.txt" 2>&1; rc=$?
  lastjson "$A/train_unval.txt" "$A/train_unval.json"
  if [ $rc = 0 ] && [ "$(jget "$A/train_unval.json" support)" = user_asserted ]; then
    record PASS "any:$k:train-unvalidated" "support=user_asserted model_type=$(jget "$A/train_unval.json" model_type) loss=$(jget "$A/train_unval.json" metrics.final_loss)"
  else record FAIL "any:$k:train-unvalidated" "rc=$rc support=$(jget "$A/train_unval.json" support) (any/$k/train_unval.txt)"; fi
  gpu env POROS_USER_ASSERTED_ARCHS="$mt" timeout -k 30 "$RTO" "$PY" -c "from poros.cli.main import cli; cli()" train "$OUT/cfg/any_$k.yaml" --output-dir "$A/run_env" --yes --json > "$A/train_env.txt" 2>&1; rc=$?
  lastjson "$A/train_env.txt" "$A/train_env.json"
  local sup; sup=$(jget "$A/train_env.json" support)
  if [ $rc = 0 ] && [ "$sup" != None ] && [ "$sup" != validated ]; then
    record PASS "any:$k:train-env-asserted" "POROS_USER_ASSERTED_ARCHS=$mt support=$sup"
  else record FAIL "any:$k:train-env-asserted" "rc=$rc support=$sup (any/$k/train_env.txt)"; fi
  # the Python surfaces: register_architecture, arch_check error / warn / off
  gpu timeout -k 30 "$RTO" "$PY" "$H/anymodel_api.py" "$repo" "$mt" "$A/api.json" > "$A/api.txt" 2>&1; rc=$?
  if [ $rc = 0 ]; then record PASS "any:$k:api" "$(grep ANYMODEL_API "$A/api.txt" | cut -c14-220)"
  else record FAIL "any:$k:api" "rc=$rc $(grep ANYMODEL_API "$A/api.txt" | cut -c14-220) (any/$k/api.txt)"; fi
  rm -rf "$A"/run_*/checkpoint-* "$A"/run_*/adapter_model.safetensors "$A"/run_*/optimizer_state.pt
}

any_refuse() {  # any_refuse KEY REPO REV: refused before any weights are fetched, in seconds, with a remedy
  local k=$1 repo=$2 rev=$3 A="$OUT/any/$k"; mkdir -p "$A"
  echo "--- any:$k ($repo @ ${rev:0:8}, must refuse) $(date -u +%FT%TZ)" | tee -a "$RESULT"
  poros check "$repo" --json > "$A/check.txt" 2>&1; local rc=$?; lastjson "$A/check.txt" "$A/check.json"
  if [ $rc != 0 ]; then record PASS "any:$k:check" "exit $rc status=$(jget "$A/check.json" status) model_type=$(jget "$A/check.json" model_type)"
  else record FAIL "any:$k:check" "exit 0: check calls it usable (any/$k/check.txt)"; fi
  poros agents arch "$repo" > "$A/arch.txt" 2>&1; rc=$?; lastjson "$A/arch.txt" "$A/arch.json"
  # docs/models.md, rung 0: what no flag trains is refused by arch itself, E_ARCH_UNSUPPORTED (exit 4)
  local acode; acode=$(grep -o '"code": *"[A-Z_]*"' "$A/arch.txt" | head -1)
  if [ $rc = 4 ] && grep -q E_ARCH_UNSUPPORTED <<< "$acode"; then
    record PASS "any:$k:arch" "refused exit 4 $acode"
  else record FAIL "any:$k:arch" "exit $rc status=$(jget "$A/arch.json" status) code=[$acode] next=$(jget "$A/arch.json" next | cut -c1-80) cannot_place=$(jget "$A/arch.json" cannot_place | cut -c1-80): train then refuses it (no flag trains it)"; fi
  "$PY" "$H/mkcfg.py" "$OUT/cfg/any_$k.yaml" "$repo" synthetic max_steps=2 max_seq_length=128 lora_target_modules=null model_revision=$rev > /dev/null
  local f
  for f in "" --unvalidated; do
    guarded "$A/train$f" train "$OUT/cfg/any_$k.yaml" --output-dir "$A/run$f" --yes --json $f; rc=$?
    local g="$A/train$f.guard.json" code rem secs
    secs=$(jget "$g" seconds); secs=${secs%.*}
    code=$(grep -o '"code": *"[A-Z_]*"' "$A/train$f.txt" | head -1)
    [ -z "$f" ] && [ -z "$code" ] && grep -q "needs_input" "$A/train$f.txt" && code=needs_input  # asks first: fine
    rem=$(grep -c -i '"remediation": *"[^"]\|"hint": *"[^"]\|"next": *\[ *"\|"questions": *\[ *{' "$A/train$f.txt")
    if [ $rc != 0 ] && [ "$(jget "$g" killed)" = None ] && [ "$(jget "$g" weight_files)" = 0 ] \
        && [ "${secs:-999}" -le 60 ] 2>/dev/null && [ -n "$code" ] && [ "$rem" -gt 0 ]; then
      record PASS "any:$k:train${f:-noflag}" "refused exit $rc in $(jget "$g" seconds)s $code, no weights fetched ($(jget "$g" cache_bytes) B cached)"
    else record FAIL "any:$k:train${f:-noflag}" "rc=$rc $(cat "$g" 2>/dev/null) code=[$code] remedy_lines=$rem (any/$k/train$f.txt)"; fi
  done
  rm -rf "$OUT/tmp/hf_guard"
}

if [ "${FG_ANYMODEL:-1}" = 1 ] && [ "$PLAN" != 1 ]; then
  while IFS=$'\t' read -r k repo rev kind notes; do
    case "$k" in ''|'#'*) continue ;; esac
    if [ -n "${FG_ANY:-}" ] && [[ ",$FG_ANY," != *",$k,"* ]]; then continue; fi
    if [ "$kind" = ladder ]; then any_ladder "$k" "$repo" "$rev"; else any_refuse "$k" "$repo" "$rev"; fi
  done < "$H/anymodel.tsv"
fi

echo "end $(date -u +%FT%TZ)" | tee -a "$RESULT"
echo "FAILS=$FAILS" | tee -a "$RESULT"
[ $FAILS -eq 0 ]
