#!/usr/bin/env bash
# tools/costmap_parity.sh — does turning the cost map on change WHAT the engine executes?
#
#   tools/costmap_parity.sh --model DIR --profile NAME [--conc 1] [--level 1]
#
# Runs the same workload twice with the SAME binary: arm A with the shape census only,
# arm B with the census plus QWEN_COST_MAP.  The census report of the two arms must
# agree on every executed path id, on coverage, on UNKNOWN and on fallback calls; if it
# does not, the instrumentation is changing dispatch and every number it produces is
# worthless.  Arm B also writes the cost map itself.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=${BIN:-./qwen_tts}; MODEL=""; SPROF=""; TOPO=${TOPO:-2x8}; CONC=1; LEVEL=1
WAVES=${WAVES:-3}; PORT=${PORT:-9880}; OUT=""
BANK=${BANK:-tests/load_texts_en.txt}; CLASSES=${CLASSES:-short}
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2;;  --profile) SPROF=$2; shift 2;;
    --conc) CONC=$2; shift 2;;    --level) LEVEL=$2; shift 2;;
    --waves) WAVES=$2; shift 2;;  --topo) TOPO=$2; shift 2;;
    --port) PORT=$2; shift 2;;    --out) OUT=$2; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
[ -n "$MODEL" ] && [ -n "$SPROF" ] || { echo "usage: --model DIR --profile NAME"; exit 2; }
OUT=${OUT:-profiles/costmap_parity_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT/A" "$OUT/B"

# A refused or crashed arm must not leave a server behind: the next run would find the
# box busy and refuse, and the orphan would keep answering on the port (§3.1).
cleanup() { pkill -9 -f 'qwen_tts.*--serve' 2>/dev/null; pkill -9 -f serve_parallel_wave 2>/dev/null; }
trap cleanup EXIT INT TERM

surv() { pgrep -f '(^|/)qwen_tts( |$)|serve_parallel_wave|load_test\.py' 2>/dev/null | wc -l | tr -d ' '; }
[ "$(surv)" = "0" ] || { echo "surviving=$(surv): refusing to measure on a busy box"; exit 1; }

arm() {  # label extra-env
    local label=$1 extra=$2
    python3 tests/serve_parallel_wave.py --model "$MODEL" --bin "$BIN" --topo "$TOPO" \
        --conc "$CONC" --waves "$WAVES" --seed 42 --precision int8 --profile "$SPROF" \
        --text-file "$BANK" --classes "$CLASSES" --out "$OUT/$label" --port "$PORT" \
        --label "$label" --no-crosscheck --server-env "$extra" > "$OUT/$label.log" 2>&1
    local rc=$?
    PORT=$((PORT + 20))
    [ $rc = 0 ] || { echo "arm $label FAILED (see $OUT/$label.log)"; return 1; }
}

echo "parity: census-only (A) vs census+cost-map level $LEVEL (B), C=$CONC"
arm A "QWEN_SHAPE_CENSUS=1,QWEN_CENSUS_JSON=$OUT/A/census-%d.json" || exit 1
arm B "QWEN_SHAPE_CENSUS=1,QWEN_CENSUS_JSON=$OUT/B/census-%d.json,QWEN_COST_MAP=$LEVEL,QWEN_COSTMAP_JSON=$OUT/B/costmap-%d.json" || exit 1

for a in A B; do
    python3 tools/census_report.py "$OUT"/$a/census-*.json --out "$OUT/$a/census_summary.json" \
        > "$OUT/$a/census_report.txt" 2>&1
done

python3 - "$OUT" <<'PY'
import json, glob, sys, os
out = sys.argv[1]

def load(arm):
    """What the run EXECUTED, at three levels of detail."""
    ids, leaves, shapes = set(), set(), set()
    for f in glob.glob(os.path.join(out, arm, "census-*.json")):
        d = json.load(open(f))
        for r in d.get("rows", []):
            ids.add((r["comp"], r["path_id"], r["path"]))
            shapes.add((r["comp"], r["path_id"], r["N"], r["K"], r["B"]))
            for l in r.get("leaves", []):
                leaves.add((r["comp"], l))
    cov = {}
    sfile = os.path.join(out, arm, "census_summary.json")
    if os.path.exists(sfile):
        cov = json.load(open(sfile))
    return ids, leaves, shapes, cov

ia, la, sa, ca = load("A")
ib, lb, sb, cb = load("B")
ok, notes = True, []

# 1. dispatch identity: which paths ran, and which kernel class ran inside them.
print("  1. executed path ids (component, path_id, path)")
print(f"     A {len(ia)}   B {len(ib)}   ", end="")
if ia == ib:
    print("IDENTICAL")
else:
    ok = False; print("DIFFER")
    for k in sorted(ia - ib): print(f"     ONLY IN A: {k}")
    for k in sorted(ib - ia): print(f"     ONLY IN B: {k}")

print("  2. executed kernel/leaf classes per component")
print(f"     A {sorted(la)}")
print(f"     B {sorted(lb)}")
if la != lb:
    ok = False; print("     DIFFER")
else:
    print("     IDENTICAL")

# 2. coverage: the PERCENTAGES and the absence of UNKNOWN/fallback, not the absolute
#    call volume (which legitimately moves between two runs, see note 4).
print("  3. coverage per component (optimized / blas / fallback / UNKNOWN)")
ka, kb = ca.get("coverage", {}), cb.get("coverage", {})
print(f"     {'component':<10}{'arm':<4}{'opt%':>8}{'blas%':>8}{'fallb%':>8}{'UNKNOWN':>9}{'calls':>9}")
for comp in sorted(set(ka) | set(kb)):
    for arm, k in (("A", ka), ("B", kb)):
        c = k.get(comp, {})
        print(f"     {comp:<10}{arm:<4}{c.get('optimized_pct', 0):>8.2f}"
              f"{c.get('blas_pct', 0):>8.2f}{c.get('fallback_pct', 0):>8.2f}"
              f"{c.get('unknown_calls', 0):>9}{c.get('calls', 0):>9}")
    x, y = ka.get(comp, {}), kb.get(comp, {})
    for f in ("optimized_pct", "blas_pct", "fallback_pct"):
        if abs(x.get(f, 0) - y.get(f, 0)) > 0.1:
            ok = False; print(f"     {comp}: {f} differs by more than 0.1 pt")
    if x.get("unknown_calls", 0) or y.get("unknown_calls", 0):
        ok = False; print(f"     {comp}: UNKNOWN calls are not zero")
if ca.get("unknown_calls", 0) or cb.get("unknown_calls", 0):
    ok = False; print("     total UNKNOWN calls are not zero")

# 3. shapes: reported, never gated.  The streaming decoder picks its chunk length from
#    how far it lags the generator, so a slower run visits a different set of chunk
#    lengths.  That moves (N,K,B) rows without moving a single dispatch decision, and
#    gating on it would make this check fail for a reason that is not about dispatch.
print("  4. shape tuples (component, path_id, N, K, B) — OBSERVATION, not a gate")
print(f"     A {len(sa)}   B {len(sb)}   symmetric difference {len(sa ^ sb)}")
onlyA = sorted({(c, p) for c, p, _, _, _ in sa - sb})
onlyB = sorted({(c, p) for c, p, _, _, _ in sb - sa})
if onlyA or onlyB:
    print(f"     paths whose shape set moved: only-A {onlyA}  only-B {onlyB}")
    print("     expected on the decoder: adaptive streaming chunk length is timing-dependent.")
    notes.append("shape tuples differ on %s" % (onlyA + onlyB))

print(f"  PARITY: {'PASS' if ok else 'FAIL'}"
      + ("  (path ids, kernel classes, coverage, UNKNOWN and fallback all match)" if ok else ""))
json.dump({"path_ids_equal": ia == ib, "leaves_equal": la == lb,
           "coverage_a": ka, "coverage_b": kb,
           "shape_symmetric_diff": len(sa ^ sb),
           "shape_paths_only_a": [list(k) for k in onlyA],
           "shape_paths_only_b": [list(k) for k in onlyB],
           "notes": notes, "verdict": "PASS" if ok else "FAIL"},
          open(os.path.join(out, "parity.json"), "w"), indent=1)
sys.exit(0 if ok else 1)
PY
RC=${PIPESTATUS[0]}
echo
echo "=== cost map (arm B) ==="
python3 tools/costmap_report.py "$OUT"/B/costmap-*.json --census "$OUT/B/census-*.json" \
    --out "$OUT/costmap_summary.json" | tee "$OUT/costmap_report.txt"
echo "artifacts: $OUT   surviving=$(surv)"
exit $RC
