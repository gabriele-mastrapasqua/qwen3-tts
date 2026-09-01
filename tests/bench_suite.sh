#!/usr/bin/env bash
set -u

MODEL=""; PROFILE="axion-16c-ttfa"; OUT="/tmp/bench_suite"; BIN="./qwen_tts"
SPK="ryan"; LANG_="English"; TOPO="2x8"
DEFAULT_MODEL="qwen3-tts-1.7b"
CORPUS=""                                  # optional: a duration-calibrated corpus
FAST="tests/load_texts_en.txt"             # inner loop: the short classes of the bank
REAL="tests/load_texts_en.txt"             # qualification: the whole bank
FAST_CLASSES="short"
REAL_CLASSES=""
SKIP_IDLE=0
ONLY=""                                    # comma list: run only these rungs
IDENTITY=0          # the identity gate needs a traced pass of its own; see below
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --bin) BIN="$2"; shift 2;;
    --speaker) SPK="$2"; shift 2;;
    --language) LANG_="$2"; shift 2;;
    --topo) TOPO="$2"; shift 2;;
    --corpus) CORPUS="$2"; shift 2;;
    --bank-fast) FAST="$2"; shift 2;;
    --bank-real) REAL="$2"; shift 2;;
    --fast-classes) FAST_CLASSES="$2"; shift 2;;
    --real-classes) REAL_CLASSES="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --skip-idle-gate) SKIP_IDLE=1; shift;;
    --identity-gate) IDENTITY=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] || MODEL="$DEFAULT_MODEL"

FAILED=0
gate () { echo "  $1"; }
die  () { echo "GATE FAILED: $1" >&2; exit 3; }

echo "########## PREFLIGHT — every one of these exits non-zero ##########"

[ -x "$BIN" ] || die "$BIN is not an executable"
BSHA=$("$BIN" --caps >/dev/null 2>&1 && shasum -a 256 "$BIN" 2>/dev/null | cut -d' ' -f1)
[ -n "$BSHA" ] || BSHA=$(sha256sum "$BIN" | cut -d' ' -f1)
BTAG=$("$BIN" --caps 2>&1 | sed -n 's/^  build: *//p' | awk '{print $1}')
gate "binary            = $BIN"
gate "binary_sha256     = $BSHA"
gate "binary_build_tag  = $BTAG"

[ -d "$MODEL" ] || die "model directory $MODEL does not exist"
gate "model             = $MODEL"

SENV=$(python3 tools/perf_profile.py server-env "$PROFILE") || die "profile $PROFILE did not resolve"
gate "profile           = $PROFILE"
gate "server_env        = $SENV"
for V in $(python3 tools/perf_profile.py forbidden-env "$PROFILE"); do
  eval "PRESENT=\${$V+set}"
  if [ "${PRESENT:-}" = "set" ]; then
    die "$V is set in this environment; the profile declares it must not be. Value: $(eval echo \$$V)"
  fi
done
gate "forbidden env     = none present"

STALE=$(pgrep -x qwen_tts 2>/dev/null | wc -l | tr -d ' ')
[ "${STALE:-0}" = "0" ] || die "$STALE qwen_tts processes already running"
gate "stale engines     = 0"
if [ "$SKIP_IDLE" = "0" ] && [ -r /proc/loadavg ]; then
  L1=$(cut -d' ' -f1 /proc/loadavg)
  awk -v l="$L1" 'BEGIN{exit !(l < 2.0)}' || die "loadavg $L1 >= 2.0; the box is not idle"
  gate "loadavg           = $L1"
fi

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)
DIRTY=$(git status --porcelain 2>/dev/null | head -1 | grep -q . && echo yes || echo no)
gate "source_commit     = $COMMIT   dirty= $DIRTY"
[ "$DIRTY" = "no" ] || echo "  WARNING: dirty tree. Final numbers require dirty=no (commit first)."

mkdir -p "$OUT"
MAN="$OUT/manifest.txt"
{
  echo "# bench_suite manifest"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "binary=$BIN"; echo "binary_sha256=$BSHA"; echo "binary_build_tag=$BTAG"
  echo "source_commit=$COMMIT"; echo "dirty=$DIRTY"
  echo "model=$MODEL"; echo "speaker=$SPK"; echo "language=$LANG_"; echo "topology=$TOPO"
  echo "profile=$PROFILE"; echo "server_env=$SENV"
  echo "host=$(hostname)"; echo "kernel=$(uname -sr)"; echo "vcpu=$(nproc 2>/dev/null || echo ?)"
} > "$MAN"

echo
echo "########## RUNGS ##########"
PORT=9400
rung () {   # name bank conc waves [extra...]
  NAME="$1"; BANK="$2"; CONC="$3"; W="$4"; shift 4
  PORT=$((PORT + 10))
  echo "=== $NAME  bank=$(basename "$BANK")  conc=$CONC  waves=$W ==="
  CMD="python3 tests/serve_parallel_wave.py --model $MODEL --bin $BIN --speaker $SPK
       --language $LANG_ --topo $TOPO --conc $CONC --waves $W --seed 42 --precision int8
       --profile $PROFILE --text-file $BANK --out $OUT/$NAME --port $PORT --label $NAME $*"
  echo "cmd: $(echo $CMD)" >> "$MAN"
  $CMD > "$OUT/$NAME.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    WHY=$(grep -m1 -E 'REFUSING TO RUN|CROSS-CHECK FAILED|Traceback|GATE FAILED|Error' \
          "$OUT/$NAME.log" 2>/dev/null | cut -c1-100)
    echo "  rc=$RC  RUNG FAILED${WHY:+ — $WHY}"
    echo "          full log: $OUT/$NAME.log"
    FAILED=1; return
  fi
  grep -E '^topo|^2x8|^4x4|^8x2|^1x16' "$OUT/$NAME.log" | head -20
  if [ "$IDENTITY" = "1" ] && [ -f tests/serve_identity_gate.py ]; then
    PORT=$((PORT + 1))
    QWEN_LIFE_TRACE=1 QWEN_REQ_TRACE=1 python3 tests/serve_parallel_wave.py \
        --model "$MODEL" --bin "$BIN" --speaker "$SPK" --language "$LANG_" --topo "$TOPO" \
        --conc 1,4 --waves 2 --seed 42 --precision int8 --profile "$PROFILE" \
        --text-file "$BANK" --out "$OUT/${NAME}_identity" --port $PORT \
        --label "${NAME}_identity" "$@" > "$OUT/${NAME}_identity.log" 2>&1
    if python3 tests/serve_identity_gate.py "$OUT/${NAME}_identity" >> "$OUT/${NAME}_identity.log" 2>&1
    then echo "  identity gate: PASS (traced pass, C=1 and C=4)"
    else echo "  IDENTITY GATE FAILED for $NAME"; FAILED=1; fi
  fi
}

cls_arg () { [ -n "$1" ] && printf -- "--classes %s" "$1"; }
want () { case ",$ONLY," in ,,) return 0;; *",$1,"*) return 0;; *) echo "=== $1: SKIPPED, not in --only $ONLY ==="; return 1;; esac; }

echo "rungs=${ONLY:-realistic,fast,short-diverse,long-diverse}" >> "$MAN"
want realistic && rung realistic "$REAL" 1,2,4,6,8 4 $(cls_arg "$REAL_CLASSES")
want fast      && rung fast      "$FAST" 1,4       5 $(cls_arg "$FAST_CLASSES")

if [ -n "$CORPUS" ] && [ -f "$CORPUS" ]; then
  want short-diverse && rung short-diverse "$CORPUS" 4,6   3 --classes short
  want long-diverse  && rung long-diverse  "$CORPUS" 1,4,6 3 --classes long
else
  echo "=== short-diverse / long-diverse: SKIPPED, no --corpus given ==="
  echo "    (pass --corpus FILE with a duration-calibrated bank; the classes must come from measured audio)"
  echo "skipped_rungs=short-diverse,long-diverse (no corpus)" >> "$MAN"
fi

echo
echo "########## AUDIO LENGTH PER CELL — comparability, not decoration ##########"
echo "# Two cells of the same bank can draw different texts when n differs, and then a"
echo "# TTFA difference is a corpus difference. The number that settles it is audio p50."
python3 - "$OUT" <<'PY'
import json, glob, os, sys
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*_requests.jsonl"))):
    r = [json.loads(l) for l in open(f) if l.strip()]
    if not r: continue
    a = sorted(x["audio_s"] for x in r)
    rows.append((os.path.relpath(f, root), len(r), a[len(a)//2]))
print("%-64s %5s %10s" % ("cell", "n", "audio_p50"))
for n, c, m in rows:
    print("%-64s %5d %9.2fs" % (n[-64:], c, m))
PY

echo
echo "########## RESULT ##########"
cat "$MAN"
if [ "$FAILED" != "0" ]; then
  echo; echo "SUITE FAILED — at least one rung or identity gate did not pass."; exit 1
fi
echo; echo "SUITE PASSED — artifacts in $OUT, manifest in $MAN"
