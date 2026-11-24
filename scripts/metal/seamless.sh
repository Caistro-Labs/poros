#!/bin/bash
# The `seamless` gate: the whole product path, in a fresh HOME with a plain PATH.
#
#   seamless.sh --root DIR --wheel poros.whl              # the whole gate, from nothing
#   seamless.sh --root DIR --python PY --hub DIR          # with a Python and weights already here
#
# Questions are answered from poros.needs_input.v1, by the command each choice names, and one the
# contract does not list fails its row. Everything made goes under --root. A row that cannot run
# says why and counts as skipped, and any skip makes this a probe, not the gate:
#   --may-copy        rows that fetch shards or build a pack, which takes real disk
#   --may-mount       rows that attach a disk image. It mounts in /Volumes while the row runs
#   --mountroot DIR   or here instead. Writes under DIR are refused while it runs, as under /Volumes,
#                     so a pulled drive does not become an empty folder. hdiutil may refuse DIR on a
#                     volume that ignores ownership: the row then fails, and says so
#   --owner-present   OWNER-PRESENT rows: sleep, a drive pulled by hand, memory pressure, no SSD. They
#                     need someone at the machine, and pressure needs the machine to itself.
#   --rows a,b        only these rows
# Each row's logs are in <root>/logs. The result is <root>/seamless.json.
set -u

ROWS="install fit tier download pack sleep disk-full drive memory prove stop-resume no-ssd words"
ROOT="" WHEEL="" PY="" HUB="" MAY_COPY="" MAY_MOUNT="" OWNER="" MOUNTROOT=/Volumes
while [ $# -gt 0 ]; do
    case $1 in
        --root) ROOT=$2; shift 2 ;;
        --wheel) WHEEL=$2; shift 2 ;;
        --python) PY=$2; shift 2 ;;
        --hub) HUB=$2; shift 2 ;;
        --rows) ROWS=$(echo "$2" | tr , ' '); shift 2 ;;
        --may-copy) MAY_COPY=1; shift ;;
        --may-mount) MAY_MOUNT=1; shift ;;
        --mountroot) MOUNTROOT=$2; shift 2 ;;
        --owner-present) OWNER=1; shift ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done
[ -n "$ROOT" ] && { [ -n "$WHEEL" ] || [ -n "$PY" ]; } || { sed -n 2,18p "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
LOGS=$ROOT/logs
mkdir -p "$LOGS" "$ROOT/home" "$ROOT/tmp"

export HOME=$ROOT/home TMPDIR=$ROOT/tmp PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset POROS_HOME HF_HOME HF_HUB_CACHE XDG_CACHE_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
[ -n "$HUB" ] && export HF_HOME=$HUB HF_HUB_CACHE=$HUB/hub
[ -z "$PY" ] && unset PYTHONPATH
BANNED='resident|fit line|window|witness|validated|covered|unproven'
SMALL=Qwen/Qwen2.5-0.5B
MAC=""
[ "$(uname)" = Darwin ] && MAC=1

poros() { "$PY" -c 'from poros.cli.main import cli; cli()' "$@"; }
start() { ( exec "$PY" -c 'from poros.cli.main import cli; cli()' "$@" ) & PID=$!; }

die() { STATUS=fail; WHY=$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-500); return 1; }
skip() { STATUS=skip; WHY=$1; return 0; }

field() { # the Python expression $2, over j: file $1 as JSON, or its last JSON line
    "$PY" - "$1" "$2" 2>/dev/null <<'PY'
import json, sys
text = open(sys.argv[1]).read().strip()
try:
    j = json.loads(text)
except ValueError:
    j = next(json.loads(x) for x in reversed(text.splitlines()) if x.startswith("{"))
p = j.get("plan") or {}
print(eval(sys.argv[2]))
PY
}

check() { # check <label> <expression>: a row fails unless the expression over its JSON is true
    [ "$(field "$LOGS/$1.json" "$2")" = True ] || die "$1: not ($2): $(tail -c 300 "$LOGS/$1.json")"
}

answer() { # the next command for this question, or nothing: it is off the contract's list
    "$PY" - "$1" <<'PY'
import json, sys
lines = open(sys.argv[1]).read().strip().splitlines()
question = json.loads(lines[-1])["questions"][0]
if question["id"] in ("model", "confirm", "store_path"):
    choices = question["choices"]
    default = next((c for c in choices if c["value"] == question.get("default")), choices[0])
    print(default["command"])
PY
}

ask() { # ask <label> <words...>: poros --json, each question answered. RC is its exit code.
    local label=$1 out="$LOGS/$1.json" asked=0 next
    shift
    poros "$@" --json > "$out" 2> "$LOGS/$label.err"
    RC=$?
    while [ "$RC" -eq 2 ] && [ "$(field "$out" 'j["schema"]')" = poros.needs_input.v1 ]; do
        asked=$((asked + 1))
        [ "$asked" -le 5 ] || { die "$label: still asking after 5 answers"; return 1; }
        next=$(answer "$out")
        [ -n "$next" ] || { die "$label: a question off the list: $(tail -c 300 "$out")"; return 1; }
        eval "$next" > "$out" 2>> "$LOGS/$label.err"
        RC=$?
    done
    return 0
}

human() { # human <label> <words...>: what a person sees, kept for the word scan
    local label=$1
    shift
    poros "$@" > "$LOGS/$label.human" 2>&1
    RC=$?
}

until_gone_or() { # until_gone_or <pid> <shell test>: 0 when the test came true while pid lived
    while kill -0 "$1" 2>/dev/null; do
        eval "$2" && return 0
        sleep 0.2
    done
    return 1
}

run_dir() { ls -dt "$HOME/.poros/runs/"* 2>/dev/null | head -1; }

steps_done() { [ "$(grep -c '"event_type": *"step"' "$(run_dir)/events.jsonl" 2>/dev/null)" -ge "$1" ]; }

image() { # image <name> <megabytes>: a sparse disk of its own, no sudo, at $MNT, gone with its row
    mkdir -p "$MOUNTROOT"
    rm -f "$ROOT/$1.sparseimage"
    hdiutil create -quiet -type SPARSE -size "${2}m" -fs APFS -volname "$1" "$ROOT/$1" \
        && attach "$1" || { die "no disk image in $MOUNTROOT: $HDIUTIL"; return 1; }
}

writable() { [ "$MOUNTROOT" = /Volumes ] || chmod u+w "$MOUNTROOT"; }

mounted() { # mounted <image or its folder/>: where hdiutil itself says those images mounted
    hdiutil info -plist 2>/dev/null | "$PY" -c '
import plistlib, sys
for image in plistlib.loads(sys.stdin.buffer.read())["images"]:
    if image["image-path"].startswith(sys.argv[1]):
        for entity in image["system-entities"]:
            if "mount-point" in entity:
                print(entity["mount-point"])
' "$1"
}

attach() { # $MNT is where it really mounted
    writable
    ATTACHED=1
    HDIUTIL=$(hdiutil attach -quiet -nobrowse -mountroot "$MOUNTROOT" "$ROOT/$1.sparseimage" 2>&1) || return 1
    MNT=$(mounted "$ROOT/$1.sparseimage")
    [ "$MOUNTROOT" = /Volumes ] || chmod a-w "$MOUNTROOT"
}

tidy() { # detach all this script attached, by where each mounted: after every row, and at its end
    [ -n "$ATTACHED" ] || return 0
    mounted "$ROOT/" | while IFS= read -r m; do hdiutil detach -quiet -force "$m"; done
    rm -f "$ROOT"/*.sparseimage
    writable
    ATTACHED=""
}
ATTACHED=""
trap tidy EXIT

free_mb() { df -m "$1" | awk 'NR==2 {print $4}'; }

fill_to() { # fill_to <mb free>: a ballast file leaves this much on the image
    mkfile "$(($(free_mb "$MNT") - $1))m" "$MNT/ballast"
}

notes() { # 500 Alpaca rows, as text in notes.jsonl and as chats in chats.jsonl
    [ -f "$ROOT/chats.jsonl" ] && return
    "$PY" - "$ROOT/notes.jsonl" "$ROOT/chats.jsonl" <<'PY'
import json, sys
from datasets import load_dataset
rows = load_dataset("tatsu-lab/alpaca", split="train[:500]")
with open(sys.argv[1], "w") as text, open(sys.argv[2], "w") as chats:
    for row in rows:
        text.write(json.dumps({"text": row["text"]}) + "\n")
        chats.write(json.dumps({"messages": [
            {"role": "user", "content": f"{row['instruction']}\n\n{row['input']}".strip()},
            {"role": "assistant", "content": row["output"]},
        ]}) + "\n")
PY
}

# -- the rows -------------------------------------------------------------------------------------

row_install() {
    [ -n "$WHEEL" ] || { skip "a Python was named, so nothing was installed"; return; }
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$ROOT/uv" UV_NO_MODIFY_PATH=1 sh > "$LOGS/install.log" 2>&1
    export UV_PYTHON_INSTALL_DIR=$ROOT/python
    "$ROOT/uv/uv" python install 3.12 >> "$LOGS/install.log" 2>&1
    "$ROOT/uv/uv" venv "$ROOT/venv" --python 3.12 >> "$LOGS/install.log" 2>&1
    "$ROOT/uv/uv" pip install --python "$ROOT/venv/bin/python" "$WHEEL" \
        >> "$LOGS/install.log" 2>&1 || { die "install failed: $(tail -3 "$LOGS/install.log")"; return 1; }
    PY=$ROOT/venv/bin/python
    human install-machine machine
    [ "$RC" -eq 0 ] || { die "poros machine exited $RC"; return 1; }
    script -q "$LOGS/install-terminal.human" "$ROOT/venv/bin/poros" machine > /dev/null
    notes
    ask install-train train qwen 0.5b --data "$ROOT/notes.jsonl" --steps 2 --yes || return 1
    [ "$RC" -eq 0 ] || { die "train exited $RC: $(tail -c 300 "$LOGS/install-train.json")"; return 1; }
    check install-train 'j["steps"] == 2'
}

row_fit() {
    ask fit-7b fit Qwen/Qwen2.5-7B || return 1
    [ "$RC" -eq 0 ] || { die "fit exited $RC"; return 1; }
    ask fit-72b fit 72b || return 1
    [ "$RC" -eq 0 ] || { die "fit 72b exited $RC"; return 1; }
    local told='all(f["value"] is not None or f["why"] for f in (p["s_per_step"], p["total_time"], p["download"]))'
    check fit-7b "$told and p['store'] == 'ssd' and j['action']['seq_len'] == 128" || return 1
    check fit-72b "$told and p['store'] == 'ssd' and j['action']['seq_len'] == 128"
}

row_tier() {
    [ -n "$HUB$MAY_COPY" ] || { skip "no --hub and no --may-copy: this would fetch 80 GB"; return; }
    local size tier
    for pair in 0.5B:ram 7B:ssd 32B:ssd; do
        size=${pair%:*} tier=${pair#*:}
        ask "tier-$size" train "Qwen/Qwen2.5-$size" --data synthetic --steps 1 --yes || return 1
        [ "$RC" -eq 0 ] || { die "$size exited $RC: $(tail -c 300 "$LOGS/tier-$size.json")"; return 1; }
        check "tier-$size" "json.load(open(j['output_dir'] + '/config.resolved.json'))['store'] == '$tier'" \
            || return 1
    done
}

row_download() {
    [ -n "$MAY_COPY" ] || { skip "needs --may-copy: it fetches 1 GB of shards"; return; }
    local hf=$ROOT/hf-fresh
    (
        export HF_HOME=$hf HF_HUB_CACHE=$hf/hub
        start train $SMALL --data synthetic --steps 1 --yes --download --json > "$LOGS/download-1.json" 2>&1
        until_gone_or "$PID" '[ -n "$(find "$hf" -name "*.incomplete" -size +1M 2>/dev/null | head -1)" ]' \
            || exit 3
        kill "$PID"
        wait "$PID"
        ask download-2 train $SMALL --data synthetic --steps 1 --yes || exit 4
        [ "$RC" -eq 0 ] && [ "$(grep -c needs_input "$LOGS/download-2.json")" -eq 0 ]
    )
    case $? in
        0) rm -rf "$hf" ;;
        3) die "the download ended before it could be killed" ;;
        *) die "did not resume: $(tail -c 300 "$LOGS/download-2.err")" ;;
    esac
}

row_pack() {
    [ -n "$MAY_COPY" ] || { skip "needs --may-copy: it builds a 0.46 GB pack, twice"; return; }
    local one=$ROOT/pack-1 two=$ROOT/pack-2 sss
    ask pack-whole train $SMALL --data synthetic --steps 1 --yes --store "$one" || return 1
    start train $SMALL --data synthetic --steps 1 --yes --store "$two" --json > "$LOGS/pack-killed.json" 2>&1
    until_gone_or "$PID" '[ -n "$(find "$two" -name model.sss -size +1k 2>/dev/null | head -1)" ]' \
        || { die "the pack ended before it could be killed"; return 1; }
    sleep "0.$((RANDOM % 9))"
    kill -9 "$PID"
    wait "$PID" 2>/dev/null
    sss=$(find "$two" -name model.sss | head -1)
    [ ! -f "$sss.idx" ] || { die "a killed pack left a store that looks complete"; return 1; }
    ask pack-again train $SMALL --data synthetic --steps 1 --yes --store "$two" || return 1
    [ "$RC" -eq 0 ] || { die "rerun exited $RC"; return 1; }
    [ "$(shasum -a 256 < "$sss")" = "$(shasum -a 256 < "$(find "$one" -name model.sss | head -1)")" ] \
        || { die "rebuilt with another digest"; return 1; }
    rm -rf "$one" "$two"
}

row_sleep() { # OWNER-PRESENT
    [ -n "$OWNER" ] || { skip "OWNER-PRESENT: someone must wake the machine"; return; }
    start train $SMALL --data synthetic --steps 60 --yes save_steps=5 --json > "$LOGS/sleep.json" 2>&1
    until_gone_or "$PID" 'steps_done 10' || { die "ended before 10 steps"; return 1; }
    pmset sleepnow
    wait "$PID"
    RC=$?
    [ "$RC" -eq 0 ] && return 0
    ask sleep-continue runs continue last || return 1
    [ "$RC" -eq 0 ] || die "runs continue exited $RC"
}

row_disk_full() {
    [ -n "$MAC" ] && [ -n "$MAY_MOUNT" ] || { skip "needs --may-mount, on a Mac"; return; }
    # Room for one 0.5B checkpoint, about 106 MB, and not for two.
    image PorosSeamlessFull 400 || return 1
    fill_to 170
    ask full-checkpoint train $SMALL --data synthetic --steps 8 --yes save_steps=2 --output "$MNT/run" || return 1
    check full-checkpoint 'j["error"]["code"] == "E_DISK"' || return 1
    rm -f "$MNT/ballast"
    ask full-continue runs continue last || return 1
    [ "$RC" -eq 0 ] || { die "after space returned, runs continue exited $RC"; return 1; }
    check full-continue 'j["runs"][0]["state"] == "succeeded" and j["runs"][0]["step"] == 8' || return 1
    tidy
    [ -n "$MAY_COPY" ] || { WHY="mid-checkpoint only: mid-pack needs --may-copy"; return 0; }
    image PorosSeamlessPack 2500 || return 1
    start train $SMALL --data synthetic --steps 1 --yes --store "$MNT/poros" --json > "$LOGS/full-pack.json" 2>&1
    # A pack writes model.sss.tmp, and model.sss only once it is whole. It holds still while the disk
    # fills, so this tests a pack and not a race; dd keeps what it wrote when the disk runs out, where
    # mkfile deletes its own file.
    until_gone_or "$PID" '[ -n "$(find "$MNT" -name model.sss.tmp 2>/dev/null | head -1)" ]' \
        || { die "the pack ended before it could be filled"; return 1; }
    kill -STOP "$PID"
    dd if=/dev/zero of="$MNT/ballast" bs=1048576 2>/dev/null
    kill -CONT "$PID"
    wait "$PID"
    check full-pack 'j["error"]["code"] == "E_DISK"' || return 1
    [ ! -f "$(find "$MNT" -name model.sss | head -1).idx" ] || { die "a torn store looks complete"; return 1; }
    rm -f "$MNT/ballast"
    ask full-pack-again train $SMALL --data synthetic --steps 1 --yes --store "$MNT/poros" || return 1
    [ "$RC" -eq 0 ] || { die "after space returned, the pack exited $RC"; return 1; }
}

row_drive() {
    [ -n "$MAC" ] && [ -n "$MAY_MOUNT" ] && [ -n "$MAY_COPY" ] \
        || { skip "needs --may-mount and --may-copy, on a Mac: its pack lives on the image"; return; }
    # A disk image stands in for the drive. Pulling a real one by hand is OWNER-PRESENT. A run keeps
    # its files beside a model off the internal disk, so its checkpoints go with the drive too.
    image PorosSeamlessDrive 2500 || return 1
    start train $SMALL --data synthetic --steps 60 --yes save_steps=3 --store "$MNT/poros" --json \
        > "$LOGS/drive.json" 2>&1
    until_gone_or "$PID" '[ -n "$(find "$MNT/poros/runs" -type d -name "checkpoint-[0-9]" 2>/dev/null)" ]' \
        || { die "ended before its checkpoint"; return 1; }
    hdiutil detach -force -quiet "$MNT"
    wait "$PID"
    check drive 'j["error"]["code"] == "E_DISK"' || return 1
    ask drive-gone runs continue last || return 1
    check drive-gone 'j["error"]["code"] == "E_DISK" and "PorosSeamlessDrive" in json.dumps(j)' || return 1
    attach PorosSeamlessDrive || { die "could not attach it again"; return 1; }
    ask drive-back runs continue last || return 1
    [ "$RC" -eq 0 ] || { die "with the drive back, runs continue exited $RC"; return 1; }
    check drive-back 'j["runs"][0]["state"] == "succeeded" and j["runs"][0]["step"] == 60'
    [ -n "$OWNER" ] || WHY="by detaching an image: pulling a drive by hand is OWNER-PRESENT"
}

row_memory() { # OWNER-PRESENT
    [ -n "$OWNER" ] || { skip "OWNER-PRESENT: it pushes the machine to critical pressure"; return; }
    local ballast
    start train $SMALL --data synthetic --steps 400 --yes save_steps=5 --json > "$LOGS/memory.json" 2>&1
    until_gone_or "$PID" 'steps_done 6' || { die "ended before 6 steps"; return 1; }
    "$PY" -c '
import mmap, subprocess, time
held = []
while subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                     capture_output=True, text=True).stdout.strip() != "4":
    block = mmap.mmap(-1, 1 << 28)
    block.write(b"\x01" * (1 << 28))
    held.append(block)
time.sleep(600)
' &
    ballast=$!
    wait "$PID"
    RC=$?
    kill "$ballast" 2>/dev/null
    [ "$RC" -eq 3 ] || { die "exit $RC, wanted E_OOM's 3"; return 1; }
    check memory 'j["error"]["code"] == "E_OOM" and "poros runs continue" in json.dumps(j)' || return 1
    ask memory-continue runs continue last || return 1
    [ "$RC" -eq 0 ] || die "runs continue exited $RC"
}

row_prove() {
    human prove prove
    [ "$RC" -eq 0 ] || { die "prove exited $RC: $(tail -5 "$LOGS/prove.human")"; return 1; }
    cp "$(ls -t "$HOME"/.poros/receipts/prove-*.json | head -1)" "$LOGS/prove-receipt.json"
    check prove-receipt 'j["identical"] and j["method"] == "ram-vs-ssd" and j["witness"]["checked"] > 0'
}

row_stop_resume() {
    notes
    for data in notes chats; do # the classic loop, then poros.Run and the SFT recipe
        "$PY" "$HERE/resume.py" --model $SMALL --data "$ROOT/$data.jsonl" --save-steps 10 \
            --out "$ROOT/resume-$data" > "$LOGS/stop-resume-$data.json" 2>&1 \
            || { die "$data, not byte-equal to uninterrupted: $(tail -c 400 "$LOGS/stop-resume-$data.json")"; return 1; }
    done
    start train $SMALL --data "$ROOT/notes.jsonl" --steps 12 --yes --json > "$LOGS/follow-train.json" 2>&1
    until_gone_or "$PID" 'steps_done 1' || { die "ended before its first step"; return 1; }
    human follow runs last --follow
    wait "$PID"
    [ "$RC" -eq 0 ] || { die "runs --follow exited $RC"; return 1; }
    human export-adapter export last --format adapter
    [ "$RC" -eq 0 ] || { die "export adapter exited $RC"; return 1; }
    [ -n "$MAY_COPY" ] || { WHY="adapter only: a merged export needs --may-copy"; return 0; }
    human export-merged export last --format merged
    [ "$RC" -eq 0 ] || die "export merged exited $RC"
}

row_no_ssd() { # OWNER-PRESENT
    [ -n "$OWNER" ] || { skip "OWNER-PRESENT: the SSD must be unmounted, and --root on the internal disk"; return; }
    ask no-ssd-7b train Qwen/Qwen2.5-7B --data synthetic --steps 1 --yes || return 1
    [ "$RC" -eq 0 ] || { die "7B exited $RC"; return 1; }
    ask no-ssd-72b fit Qwen/Qwen2.5-72B || return 1
    check no-ssd-72b '"GB" in " ".join(p["warnings"]) or j.get("error")'
}

row_words() { # last: it scans what the other rows printed, too
    human words-machine machine
    human words-models models
    human words-fit fit Qwen/Qwen2.5-7B
    human words-train train $SMALL --data synthetic --steps 2 --yes
    human words-runs runs
    human words-export export last --format adapter --overwrite
    local hits
    hits=$(grep -EHi "$BANNED" "$LOGS"/*.human | head -5)
    [ -z "$hits" ] || die "banned words: $hits"
}

# -- the gate ------------------------------------------------------------------------------------

SEEN="" FAILED=0 SKIPPED=0
for name in $ROWS; do
    STATUS=pass WHY=""
    "row_$(echo "$name" | tr - _)" || [ "$STATUS" = fail ] || STATUS=fail
    tidy
    [ "$STATUS" = fail ] && FAILED=$((FAILED + 1))
    [ "$STATUS" = skip ] && SKIPPED=$((SKIPPED + 1))
    printf '%-12s %-5s %s\n' "$name" "$STATUS" "$WHY"
    SEEN="$SEEN$name	$STATUS	$WHY
"
done
printf '%s' "$SEEN" | "$PY" -c '
import json, sys
rows = [dict(zip(("row", "status", "why"), line.rstrip("\n").split("\t"))) for line in sys.stdin]
json.dump({"gate": "seamless", "rows": rows}, open(sys.argv[1], "w"), indent=1)
' "$ROOT/seamless.json"
[ "$SKIPPED" -eq 0 ] || echo "$SKIPPED row(s) skipped: this is a probe, not the gate"
[ "$FAILED" -eq 0 ]
