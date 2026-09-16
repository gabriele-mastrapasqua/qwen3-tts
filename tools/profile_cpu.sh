#!/usr/bin/env bash
# tools/profile_cpu.sh — `make profile-cpu`, V1 step 1: what the engine ACTUALLY executed.
#
#   make profile-cpu PROFILE_MODEL=qwen3-tts-1.7b-base PROFILE_PROFILE=aws-c8a-16c-vnni-ttfa \
#                    PROFILE_TOPO=2x8 PROFILE_CONC=1,4 PROFILE_WAVES=3
#
# 1. make cpu-check              provenance (fingerprint from the binary), hardware, resolved
#                                 dispatch map, expected-vs-observed, coverage of the registry
# 2. clean run                    serve_parallel_wave, the profile's env, NO diagnostics
# 3. census run                   the same, plus QWEN_SHAPE_CENSUS=1 and QWEN_CENSUS_JSON per
#                                 process (prefork workers included: the parent forwards SIGUSR1)
# 4. tools/census_report.py       call map, coverage, UNKNOWN/fallback, selected-vs-executed
# 5. overhead                     census run vs clean run on TTFA / STREAM_RTF / TOTAL_RTF p50
#
# Everything lands in profiles/LATEST/profile/ next to the cpu-check artifacts; summary.md
# is the human page, census_summary.json / overhead.json the machine ones.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=${BIN:-./qwen_tts}
PROFILES_DIR=${PROFILES_DIR:-profiles}
MODEL=${CPU_MODEL:?CPU_MODEL / PROFILE_MODEL is required}
SPROF=${CPU_PROFILE:?CPU_PROFILE / PROFILE_PROFILE is required}
TOPO=${PROFILE_TOPO:-2x8}; CONC=${PROFILE_CONC:-1,4}; WAVES=${PROFILE_WAVES:-3}
CLASSES=${PROFILE_CLASSES:-short}; BANK=${PROFILE_BANK:-tests/load_texts_en.txt}
PORT=${PROFILE_PORT:-9800}
OVERHEAD_MAX_PCT=${OVERHEAD_MAX_PCT:-5}

echo "PROFILE CPU (V1 step 1: executed paths)"
echo "======================================="
# 1. preflight, fresh directory
MEMBW_BIN=${MEMBW_BIN:-} PROFILES_DIR=$PROFILES_DIR CPU_PROFILE=$SPROF CPU_MODEL=$MODEL bash tools/cpu_check.sh > /tmp/profile_cpu_check.txt 2>&1
CRC=$?
D=$(readlink -f "$PROFILES_DIR/LATEST" 2>/dev/null || echo "$PROFILES_DIR/LATEST")
grep -E '^\[(PASS|FAIL|WARN|SKIP|INFO)\]|VALID' /tmp/profile_cpu_check.txt | sed 's/^/  /'
[ $CRC = 0 ] || { echo "cpu-check FAILED: not profiling a build that does not pass its own preflight"; exit 1; }
P=$D/profile; rm -rf "$P"; mkdir -p "$P/clean" "$P/census"

SURV=$(pgrep -f '(^|/)qwen_tts( |$)|serve_parallel_wave|load_test\.py' 2>/dev/null | wc -l | tr -d ' ')
[ "${SURV:-0}" = "0" ] || { echo "surviving=$SURV engine/harness processes: refusing to profile a busy box"; exit 1; }

run_wave() {   # label extra-server-env
    local label=$1 extra=$2
    local args=(--model "$MODEL" --bin "$BIN" --topo "$TOPO" --conc "$CONC" --waves "$WAVES" --seed 42
                --precision int8 --profile "$SPROF" --text-file "$BANK" --classes "$CLASSES"
                --out "$P/$label" --port "$PORT" --label "$label" --no-crosscheck)
    [ -n "$extra" ] && args+=(--server-env "$extra")
    echo; echo "=== run: $label  topo=$TOPO conc=$CONC waves=$WAVES ${extra:+env+=$extra} ==="
    python3 tests/serve_parallel_wave.py "${args[@]}" > "$P/$label.log" 2>&1
    local rc=$?
    grep -E 'flags verified|C=[0-9]|TOTAL_RTF|REFUS|Error|Traceback' "$P/$label.log" | sed 's/^/  /'
    PORT=$((PORT + 20))
    return $rc
}
run_wave clean "" || { echo "clean run failed (see $P/clean.log)"; exit 1; }
run_wave census "QWEN_SHAPE_CENSUS=1,QWEN_CENSUS_JSON=$P/census/census-%d.json" || { echo "census run failed (see $P/census.log)"; exit 1; }

echo; echo "=== census report ==="
NJ=$(ls "$P"/census/census-*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$NJ" = "0" ]; then
    echo "no census JSON produced: the server did not dump (QWEN_CENSUS_JSON not honoured?) — see $P/census.log"; CRC2=1
else
    python3 tools/census_report.py "$P"/census/census-*.json --dispatch "$D/dispatch.json" --out "$P/census_summary.json" > "$P/census_report.txt" 2>&1; CRC2=$?
    cat "$P/census_report.txt"
fi

echo; echo "=== overhead: census run vs clean run (same config, same bank, same waves) ==="
python3 - "$P" "$OVERHEAD_MAX_PCT" <<'PY' | tee "$P/overhead.txt"
import json, sys, glob, os
P, lim = sys.argv[1], float(sys.argv[2])
def rows(label):
    f = glob.glob(os.path.join(P, label, "parallel_*.json"))
    return {r["conc"]: r for r in json.load(open(f[0]))} if f else {}
c, s = rows("clean"), rows("census")
out = {"limit_pct": lim, "cells": []}
worst = 0.0
print(f"  {'C':>3}{'metric':>12}{'clean':>10}{'census':>10}{'delta':>9}")
for conc in sorted(set(c) & set(s)):
    for m in ("ttfb_p50", "ttfa_p50", "stream_p50", "rtf_p50", "stream_p95", "ttfa_p95"):
        a, b = c[conc].get(m), s[conc].get(m)
        if not a or not b or a != a or b != b: continue
        d = 100.0 * (b - a) / a
        worst = max(worst, d) if m in ("stream_p50", "ttfa_p50", "rtf_p50") else worst
        out["cells"].append({"conc": conc, "metric": m, "clean": a, "census": b, "delta_pct": d})
        print(f"  {conc:>3}{m:>12}{a:>10.3f}{b:>10.3f}{d:>+8.1f}%")
out["worst_p50_delta_pct"] = worst
verdict = "PASS" if worst <= lim else "WARN"
out["verdict"] = verdict
print(f"  measurement overhead (worst p50 delta on TTFA/STREAM/TOTAL): {worst:+.1f}%  limit {lim:.0f}%  -> {verdict}")
json.dump(out, open(os.path.join(P, "overhead.json"), "w"), indent=1)
PY

# summary.md
{
  echo "# profile-cpu — V1 step 1 (executed paths)"
  echo
  echo "artifact: \`$D\`"
  echo
  echo "## preflight (make cpu-check)"; echo '```'; grep -E '^\[(PASS|FAIL|WARN|SKIP|INFO)\]|VALID' /tmp/profile_cpu_check.txt; echo '```'
  echo "## workload"; echo "model=$MODEL profile=$SPROF topo=$TOPO conc=$CONC waves=$WAVES bank=$BANK classes=$CLASSES"
  echo; echo "## census (what executed)"; echo '```'; cat "$P/census_report.txt" 2>/dev/null; echo '```'
  echo "## overhead"; echo '```'; cat "$P/overhead.txt"; echo '```'
  echo "## dispatch gate"; echo '```'; cat "$D/dispatch_gate.txt" 2>/dev/null; echo '```'
} > "$P/summary.md"
{ echo; echo "profile-cpu: census gate rc=$CRC2  overhead=$(grep -o 'PASS\|WARN' "$P/overhead.txt" | tail -1)"; } >> "$D/gate.txt"
echo; echo "artifacts: $P  (summary: $P/summary.md)"
SURV=$(pgrep -f '(^|/)qwen_tts( |$)|serve_parallel_wave|load_test\.py' 2>/dev/null | wc -l | tr -d ' ')
echo "surviving=$SURV"
exit $CRC2
