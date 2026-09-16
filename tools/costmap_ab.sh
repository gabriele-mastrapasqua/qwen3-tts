#!/usr/bin/env bash
# tools/costmap_ab.sh — measure what the cost map costs, with INTERLEAVED replicas.
#
#   tools/costmap_ab.sh --model DIR --profile NAME [--conc 1] [--level 1] [--reps 2]
#
# One clean/instrumented pair back to back cannot separate the instrumentation from
# the drift of the box (the c8a moved 184 -> 302 ms on C=4 between identical runs on
# 2026-09-04).  So the arms are run A B B A: the two A arms bracket the two B arms,
# and a linear drift over the sequence cancels in the A-mean vs B-mean comparison.
# The SAME binary is used for both arms: only QWEN_COST_MAP differs.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=${BIN:-./qwen_tts}; MODEL=""; SPROF=""; TOPO=${TOPO:-2x8}; CONC=1; LEVEL=1
WAVES=${WAVES:-3}; REPS=2; PORT=${PORT:-9860}; OUT=""
BANK=${BANK:-tests/load_texts_en.txt}; CLASSES=${CLASSES:-short}
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2;;  --profile) SPROF=$2; shift 2;;
    --conc) CONC=$2; shift 2;;    --level) LEVEL=$2; shift 2;;
    --waves) WAVES=$2; shift 2;;  --reps) REPS=$2; shift 2;;
    --topo) TOPO=$2; shift 2;;    --port) PORT=$2; shift 2;;
    --out) OUT=$2; shift 2;;      *) echo "unknown arg $1"; exit 2;;
  esac
done
[ -n "$MODEL" ] && [ -n "$SPROF" ] || { echo "usage: --model DIR --profile NAME"; exit 2; }
OUT=${OUT:-profiles/costmap_ab_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"

# A refused or crashed arm must not leave a server behind: the next run would find the
# box busy and refuse, and the orphan would keep answering on the port (§3.1).
cleanup() { pkill -9 -f 'qwen_tts.*--serve' 2>/dev/null; pkill -9 -f serve_parallel_wave 2>/dev/null; }
trap cleanup EXIT INT TERM

surv() { pgrep -f '(^|/)qwen_tts( |$)|serve_parallel_wave|load_test\.py' 2>/dev/null | wc -l | tr -d ' '; }
[ "$(surv)" = "0" ] || { echo "surviving=$(surv): refusing to measure on a busy box"; exit 1; }

run_arm() {  # arm-label sequence-index extra-env
    local arm=$1 idx=$2 extra=$3 label="${1}_${2}"
    local args=(--model "$MODEL" --bin "$BIN" --topo "$TOPO" --conc "$CONC" --waves "$WAVES"
                --seed 42 --precision int8 --profile "$SPROF" --text-file "$BANK"
                --classes "$CLASSES" --out "$OUT/$label" --port "$PORT" --label "$label"
                --no-crosscheck)
    [ -n "$extra" ] && args+=(--server-env "$extra")
    echo "  arm $idx: $arm ${extra:+($extra)}"
    python3 tests/serve_parallel_wave.py "${args[@]}" > "$OUT/$label.log" 2>&1 || {
        echo "  arm $label FAILED, see $OUT/$label.log"; return 1; }
    PORT=$((PORT + 20))
}

echo "interleaved A/B/B/A  model=$MODEL profile=$SPROF conc=$CONC level=$LEVEL waves=$WAVES reps=$REPS"
SEQ=""
i=0
r=0
while [ $r -lt "$REPS" ]; do
    # A B B A per replica: the clean arms bracket the instrumented ones
    for arm in A B B A; do
        i=$((i + 1))
        if [ "$arm" = A ]; then run_arm A $i "" || exit 1
        else run_arm B $i "QWEN_COST_MAP=$LEVEL,QWEN_COSTMAP_JSON=$OUT/costmap-B$i-%d.json" || exit 1
        fi
        SEQ="$SEQ $arm:$i"
    done
    r=$((r + 1))
done

python3 - "$OUT" "$CONC" "$LEVEL" <<'PY' | tee "$OUT/overhead.txt"
import glob, json, os, statistics, sys
out, conc, level = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def cell(label):
    f = glob.glob(os.path.join(out, label, "parallel_*.json"))
    if not f: return None
    for r in json.load(open(f[0])):
        if r["conc"] == conc: return r
    return None
arms = {"A": [], "B": []}
for d in sorted(os.listdir(out)):
    if len(d) > 2 and d[0] in "AB" and d[1] == "_":
        c = cell(d)
        if c: arms[d[0]].append((d, c))
print(f"  interleaved A/B/B/A at C={conc}, cost-map level {level}")
print(f"  {'metric':<12}{'clean(A)':>11}{'costmap(B)':>12}{'delta':>9}{'A spread':>11}")
res, worst = {}, 0.0
for m, lab in (("ttfb_p50","TTFB p50"),("ttfa_p50","TTFA p50"),("stream_p50","STREAM p50"),
               ("rtf_p50","TOTAL p50"),("ttfa_p95","TTFA p95"),("stream_p95","STREAM p95")):
    va = [c[m] for _, c in arms["A"] if c.get(m) and c[m] == c[m]]
    vb = [c[m] for _, c in arms["B"] if c.get(m) and c[m] == c[m]]
    if not va or not vb: continue
    ma, mb = statistics.mean(va), statistics.mean(vb)
    d = 100.0 * (mb - ma) / ma if ma else 0.0
    spread = 100.0 * (max(va) - min(va)) / ma if ma and len(va) > 1 else 0.0
    if m in ("ttfa_p50", "stream_p50", "rtf_p50"): worst = max(worst, d)
    res[m] = {"clean": ma, "costmap": mb, "delta_pct": d, "clean_spread_pct": spread}
    print(f"  {lab:<12}{ma:>11.3f}{mb:>12.3f}{d:>+8.1f}%{spread:>10.1f}%")
verdict = "PASS(<3%)" if worst < 3 else ("ACCEPTABLE(<=5%)" if worst <= 5 else "FAIL(>5%)")
print(f"  worst p50 overhead (TTFA/STREAM/TOTAL): {worst:+.1f}%  -> {verdict}")
sa = max((r["clean_spread_pct"] for r in res.values()), default=0.0)
if sa > abs(worst):
    print(f"  NOTE: the clean arms themselves spread {sa:.1f}%, which is larger than the")
    print(f"        measured effect: at this C the box noise dominates, treat as observational.")
json.dump({"conc": conc, "level": level, "metrics": res, "worst_p50_delta_pct": worst,
           "verdict": verdict, "clean_spread_pct": sa},
          open(os.path.join(out, "overhead.json"), "w"), indent=1)
PY
echo "artifacts: $OUT   surviving=$(surv)"
