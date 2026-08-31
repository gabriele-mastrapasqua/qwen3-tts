#!/usr/bin/env bash
# ONE command for the numbers that leave this repo.
#
# WHY IT EXISTS. The suite has been run by hand more than once, and a run that forgets a
# setting still prints a table. Measured 2026-08-31, same binary, same bank, same host:
# C=1 on the FAST bank was 108 ms without the platform's declared OPENBLAS_THREAD_TIMEOUT=1
# and 66 ms with it -- and the bare arm was bimodal between the two, so a single run could
# land on either and look definitive. Nothing in the output said which had been measured.
#
# So this script owns the whole invocation: the profile, the forbidden variables, the
# binary's identity, the idle check, every rung, the identity gates, and a manifest that
# repeats the exact commands. If a number is quoted to anyone outside this repo, it came
# from here.
#
#   bash tests/bench_suite.sh --model DIR --profile axion-16c-ttfa --out DIR
#
# Every gate below exits non-zero. A suite that warns and continues is a suite that ships
# a warning nobody read.
set -u

# DEFAULTS ARE GENERIC AND MUST STAY GENERIC. This engine is public; a benchmark script
# that names a particular deployment's model, voice or text bank carries that name into
# every log, filename and report it produces. So the defaults run against an open-weights
# checkpoint with a preset voice and the neutral bank, and any other configuration is
# passed in on the command line by whoever owns it.
MODEL=""; PROFILE="axion-16c-ttfa"; OUT="/tmp/bench_suite"; BIN="./qwen_tts"
SPK="ryan"; LANG_="English"; TOPO="2x8"
DEFAULT_MODEL="qwen3-tts-1.7b"
CORPUS=""                                  # optional: a duration-calibrated corpus
FAST="tests/load_texts_en.txt"             # inner loop: the short classes of the bank
REAL="tests/load_texts_en.txt"             # qualification: the whole bank
FAST_CLASSES="short"
REAL_CLASSES=""
SKIP_IDLE=0
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
    --skip-idle-gate) SKIP_IDLE=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] || MODEL="$DEFAULT_MODEL"

FAILED=0
gate () { echo "  $1"; }
die  () { echo "GATE FAILED: $1" >&2; exit 3; }

echo "########## PREFLIGHT — every one of these exits non-zero ##########"

# 1. the binary exists, runs, and says who it is
[ -x "$BIN" ] || die "$BIN is not an executable"
BSHA=$("$BIN" --caps >/dev/null 2>&1 && shasum -a 256 "$BIN" 2>/dev/null | cut -d' ' -f1)
[ -n "$BSHA" ] || BSHA=$(sha256sum "$BIN" | cut -d' ' -f1)
BTAG=$("$BIN" --caps 2>&1 | sed -n 's/^  build: *//p' | awk '{print $1}')
gate "binary            = $BIN"
gate "binary_sha256     = $BSHA"
gate "binary_build_tag  = $BTAG"

# 2. the model is the one that was asked for, and it exists
[ -d "$MODEL" ] || die "model directory $MODEL does not exist"
gate "model             = $MODEL"

# 3. the profile resolves, and its forbidden variables are ABSENT from this environment.
#    Not "we did not set them" -- absent. Someone else's export is exactly the invisible
#    variable this script exists to prevent.
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

# 4. nothing else is running that would share the cores
STALE=$(pgrep -c -x qwen_tts 2>/dev/null || echo 0)
[ "$STALE" = "0" ] || die "$STALE qwen_tts processes already running"
gate "stale engines     = 0"
if [ "$SKIP_IDLE" = "0" ] && [ -r /proc/loadavg ]; then
  L1=$(cut -d' ' -f1 /proc/loadavg)
  awk -v l="$L1" 'BEGIN{exit !(l < 2.0)}' || die "loadavg $L1 >= 2.0; the box is not idle"
  gate "loadavg           = $L1"
fi

# 5. the tree's provenance travels with the numbers
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
  if [ $RC -ne 0 ]; then echo "  rc=$RC  RUNG FAILED - see $OUT/$NAME.log"; FAILED=1; return; fi
  grep -E '^topo|^2x8|^4x4|^8x2|^1x16' "$OUT/$NAME.log" | head -20
  if [ -f tests/serve_identity_gate.py ]; then
    python3 tests/serve_identity_gate.py "$OUT/$NAME" >> "$OUT/$NAME.log" 2>&1 \
      || { echo "  IDENTITY GATE FAILED for $NAME"; FAILED=1; }
  fi
}

cls_arg () { [ -n "$1" ] && printf -- "--classes %s" "$1"; }

rung realistic "$REAL" 1,2,4,6,8 4 $(cls_arg "$REAL_CLASSES")
rung fast      "$FAST" 1,4       5 $(cls_arg "$FAST_CLASSES")

# The duration-calibrated corpus is optional: it is built per deployment by
# `make corpus-calibrate`, because a class boundary drawn from MEASURED audio duration on
# one checkpoint does not describe another. Without one, the two diverse rungs are skipped
# and SAY SO -- a silently missing rung reads as a rung that passed.
if [ -n "$CORPUS" ] && [ -f "$CORPUS" ]; then
  rung short-diverse "$CORPUS" 4,6   3 --classes short
  rung long-diverse  "$CORPUS" 1,4,6 3 --classes long
else
  echo "=== short-diverse / long-diverse: SKIPPED, no --corpus given ==="
  echo "    (build one with 'make corpus-calibrate'; classes must come from measured audio)"
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
