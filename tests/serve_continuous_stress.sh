#!/usr/bin/env bash
set -u
MODEL="${1:-qwen3-tts-0.6b}"
PORT="${2:-8786}"
N="${3:-6}"
MB="${4:-2}"
BIN=./qwen_tts
CMP=tests/compare_audio.py
TMP=$(mktemp -d)
trap 'pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; rm -rf "$TMP"' EXIT

SPK=(ryan vivian serena ryan vivian serena ryan vivian)
LNG=(Italian English Italian Italian English Italian Italian English)

echo "=== continuous admission stress (model=$MODEL N=$N max_batch=$MB) ==="
pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1
QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --serve "$PORT" --batch-size "$MB" > "$TMP/srv.log" 2>&1 &
sleep 6

pids=""
for i in $(seq 0 $((N-1))); do
  s=$((500+i))
  spk=${SPK[$i]}; lng=${LNG[$i]}
  body="{\"text\":\"Continuous request number $i exercising the batching path.\",\"speaker\":\"$spk\",\"language\":\"$lng\",\"temperature\":0,\"seed\":$s}"
  timeout 150 curl -s "http://localhost:$PORT/v1/tts" -d "$body" -o "$TMP/c_$i.wav" &
  pids="$pids $!"
  sleep 0.5      # stagger arrivals so admission must interleave with generation
done
for p in $pids; do wait "$p"; done   # only curl pids, never the server

echo "--- responses ---"
ok=0
for i in $(seq 0 $((N-1))); do
  b=$(wc -c < "$TMP/c_$i.wav" 2>/dev/null || echo 0)
  if [ "${b:-0}" -gt 1000 ]; then echo "  c_$i: $b bytes OK"; ok=$((ok+1)); else echo "  c_$i: $b bytes FAIL"; fi
done
echo "--- continuous admission log (admitted should reach $N > $MB) ---"
grep "BATCH\] done" "$TMP/srv.log" | tail -n "$N"
maxadm=$(grep -o "admitted=[0-9]*" "$TMP/srv.log" | sed 's/admitted=//' | sort -n | tail -1)
echo "  peak admitted=$maxadm (batch width=$MB)"

QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --text "Richiesta continua numero 0 di prova del batching." -s ryan -l Italian -T 0 --seed 500 -j1 -o "$TMP/r0.wav" --silent 2>/dev/null
echo "--- correctness spot-check c_0 vs single-stream ---"
python3 "$CMP" "$TMP/r0.wav" "$TMP/c_0.wav" 2>/dev/null

pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1
echo "=== $ok/$N completed ==="
[ "$ok" -eq "$N" ]
