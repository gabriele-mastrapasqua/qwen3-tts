#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${MODEL:-qwen3-tts-0.6b}"
STREAMS="${STREAMS:-3}"                 # stream concorrenti serviti (= S)
CORES="${CORES:-$(nproc 2>/dev/null || echo 8)}"
QUANT="${QUANT:---int8}"
REQS="${REQS:-8}"                       # requests PER STREAM
SPK="${SPK:-ryan}"; LANG_="${LANG_:-english}"
BANK="${BANK:-tests/load_texts_en.txt}"
CLASSES="${CLASSES:-short,medium}"
INTERVAL="${INTERVAL:-30}"              # s between arrivals of one client (uniform)
REQ_TIMEOUT="${REQ_TIMEOUT:-180}"
PORT0="${PORT0:-8970}"
OUT="${OUT:-/tmp/tts/shard_vs_batch}"
PY="${PY:-python3}"
BIN=./qwen_tts

case "$MODEL" in *models/*|*outputs/*) echo "refusing a model from a private tree: $MODEL" >&2; exit 1;; esac
[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
mkdir -p "$OUT"

JS=$(( CORES / STREAMS )); [ "$JS" -lt 1 ] && JS=1     # thread per shard
echo "=== un processo grosso o $STREAMS piccoli? ==="
echo "  modello   $MODEL $QUANT · $STREAMS stream · $CORES core totali"
echo "  BATCH     1 server -j$CORES --batch-size $STREAMS"
echo "  SHARD     $STREAMS server -j$JS (uno per stream)"
echo "  load      $REQS requests per stream · uniform arrivals every ${INTERVAL}s · classes $CLASSES"

kill_all() { pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 2; }
trap kill_all EXIT

pss_of() {  # somma il PSS di tutti i server, in MB
    local t=0 v
    for pid in $(pgrep -f "qwen_tts.*--serve" 2>/dev/null); do
        v=$(awk '/^Pss:/ {s+=$2} END {print s+0}' "/proc/$pid/smaps_rollup" 2>/dev/null)
        t=$((t + ${v:-0}))
    done
    echo $((t / 1024))
}
used_of() {  # memoria di sistema occupata (MB), esclusa la page cache
    free -m 2>/dev/null | awk '/^Mem:/ {print $3}'
}
WAIT_READY_S="${WAIT_READY_S:-600}"
wait_ready() {  # $1 = porta
    local i
    for i in $(seq 1 "$WAIT_READY_S"); do
        curl -s -m 2 "http://localhost:$1/v1/health" 2>/dev/null | grep -q '"ok"' && {
            [ "$i" -gt 30 ] && echo "     porta $1 pronta dopo ${i}s"
            return 0
        }
        case $i in 60|120|240|360|480) echo "     ...porta $1: ancora in caricamento a ${i}s" ;; esac
        sleep 1
    done
    echo "     port $1: no \"ok\" from /v1/health within ${WAIT_READY_S}s — last state:"
    curl -s -m 2 "http://localhost:$1/v1/health" 2>/dev/null | head -c 200 | sed 's/^/       /'
    echo
    return 1
}

run_arm() {  # $1 = batch | shard
    local arm="$1" ports="" p n
    kill_all
    if [ "$arm" = batch ]; then
        $BIN -d "$MODEL" $QUANT --serve "$PORT0" --batch-size "$STREAMS" -j "$CORES" \
            > "$OUT/${arm}_srv0.log" 2>&1 &
        wait_ready "$PORT0" || { echo "  🚨 server non partito"; return 1; }
        for n in $(seq 1 "$STREAMS"); do ports="$ports $PORT0"; done
    else
        for n in $(seq 0 $((STREAMS - 1))); do
            p=$((PORT0 + 1 + n))
            $BIN -d "$MODEL" $QUANT --serve "$p" --batch-size 2 -j "$JS" > "$OUT/${arm}_srv$n.log" 2>&1 &
            ports="$ports $p"
            sleep 2   # staggered: S processes opening the same file together contend
        done
        for p in $ports; do wait_ready "$p" || { echo "  🚨 server $p non partito"; return 1; }; done
    fi

    for p in $ports; do
        curl -s -m "$REQ_TIMEOUT" -o /dev/null -X POST "http://localhost:$p/v1/tts/stream" \
            -d "{\"text\":\"Warm up the engine before measuring.\",\"speaker\":\"$SPK\",\"language\":\"$LANG_\",\"seed\":7}" || true
    done
    local pss used; pss=$(pss_of); used=$(used_of)

    local n=0 pids=""
    for p in $ports; do
        QWEN_LT_ARRIVAL=uniform QWEN_LT_INTERVAL="$INTERVAL" \
        $PY tests/load_test.py --url "http://localhost:$p" --path /v1/tts/stream \
            --concurrency 1 --requests "$REQS" --text-file "$BANK" --classes "$CLASSES" \
            --speaker "$SPK" --language "$LANG_" --timeout "$REQ_TIMEOUT" --no-save-audio \
            --json "$OUT/${arm}_c$n.json" --csv "$OUT/${arm}_c$n.csv" > /dev/null 2>&1 &
        pids="$pids $!"; n=$((n + 1))
    done
    for pid in $pids; do wait "$pid"; done
    kill_all

    $PY - "$OUT" "$arm" "$STREAMS" "$pss" "$used" <<'PYEOF'
import csv, json, sys, glob, os
out, arm, streams = sys.argv[1], sys.argv[2], int(sys.argv[3])
pss, used = int(sys.argv[4]), int(sys.argv[5])

ttfa = []
for f in sorted(glob.glob(os.path.join(out, arm + "_c*.csv"))):
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = r.get("ttfa_ms", "")
            if v not in ("", None):
                try:
                    ttfa.append(float(v))
                except ValueError:
                    pass
audio = wall = 0.0
for f in sorted(glob.glob(os.path.join(out, arm + "_c*.json"))):
    d = json.load(open(f))
    for lvl in (d if isinstance(d, list) else [d]):
        if isinstance(lvl, dict):
            audio += lvl.get("audio_s", 0.0)
            wall = max(wall, lvl.get("wall_s", 0.0))
ttfa.sort()

def pct(v, q):
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]

qq = audio / wall if wall else float("nan")
print("  %-6s TTFA p50 %6.0f ms . p95 %6.0f ms . samples %2d . Q %.2f . PSS %d MB . system %d MB"
      % (arm, pct(ttfa, 0.5), pct(ttfa, 0.95), len(ttfa), qq, pss, used))
PYEOF
}

run_arm batch
run_arm shard
echo
echo "  Percentiles come from the PER-REQUEST records (--csv) of all clients pooled:"
echo "  the JSON spikes block holds only the requests over budget."
echo "  ⚠️  PSS, non somma di RSS: l'RSS conta le pagine condivise in ogni processo che le"
echo "      mappa. Atteso ~1x i pesi mmappati + Sx la copia quantizzata."
