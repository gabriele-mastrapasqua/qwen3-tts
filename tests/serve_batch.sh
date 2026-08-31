#!/usr/bin/env bash
# serve_batch.sh — validate the vLLM-style request-batching server (--batch-size N).
#
# Fires concurrent requests with DIFFERENT text/speaker/language at temp 0 and,
# with QWEN_BATCH_FORCE_MATVEC=1 (batched compute bit-exact to single-stream),
# asserts each response mel-corr 1.0 vs its own single-stream reference + zero
# cross-talk. Also smoke-tests the production matmat path and the stream fallback.
#
# Usage: tests/serve_batch.sh [model_dir] [port]
# Requires: python3 + librosa (tests/compare_audio.py), a quiet machine.
#
# NOTE: always kill the server by name and use timeout on curls — never `wait` on
# the server pid (it never exits).
set -u
MODEL="${1:-qwen3-tts-0.6b}"
PORT="${2:-8779}"
BIN=./qwen_tts
CMP=tests/compare_audio.py
TMP=$(mktemp -d)
trap 'pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; rm -rf "$TMP"' EXIT

pass=0; fail=0
chk() { # label refwav gotwav  -> expect PASS
  local r; r=$(python3 "$CMP" "$2" "$3" 2>/dev/null)
  if echo "$r" | grep -q PASS; then echo "  PASS  $1 :: $r"; pass=$((pass+1));
  else echo "  FAIL  $1 :: $r"; fail=$((fail+1)); fi
}

echo "=== request-batching server test (model=$MODEL port=$PORT) ==="
pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1

# ── A) rigorous gate: force_matvec, 3 different concurrent requests, temp 0 ──
QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --serve "$PORT" --batch-size 4 > "$TMP/srv.log" 2>&1 &
sleep 6
TA='{"text":"Il primo messaggio di prova del server.","speaker":"ryan","language":"Italian","temperature":0,"seed":100}'
TB='{"text":"This is the second concurrent test message.","speaker":"vivian","language":"English","temperature":0,"seed":200}'
TC='{"text":"Terzo ed ultimo esempio di sintesi vocale batched.","speaker":"serena","language":"Italian","temperature":0,"seed":300}'
timeout 90 curl -s "http://localhost:$PORT/v1/tts" -d "$TA" -o "$TMP/sa.wav" & PA=$!
timeout 90 curl -s "http://localhost:$PORT/v1/tts" -d "$TB" -o "$TMP/sb.wav" & PB=$!
timeout 90 curl -s "http://localhost:$PORT/v1/tts" -d "$TC" -o "$TMP/sc.wav" & PC=$!
wait $PA $PB $PC            # only the curls — NEVER the server
echo "--- batch log ---"; grep BATCH "$TMP/srv.log" | tail -3
pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1

# single-stream references (force_matvec → bit-exact), same params
QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --text "Il primo messaggio di prova del server." -s ryan -l Italian -T 0 --seed 100 -j1 -o "$TMP/ra.wav" --silent 2>/dev/null
QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --text "This is the second concurrent test message." -s vivian -l English -T 0 --seed 200 -j1 -o "$TMP/rb.wav" --silent 2>/dev/null
QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --text "Terzo ed ultimo esempio di sintesi vocale batched." -s serena -l Italian -T 0 --seed 300 -j1 -o "$TMP/rc.wav" --silent 2>/dev/null
echo "--- per-request correctness (force_matvec, expect PASS) ---"
chk "A ryan/IT"   "$TMP/ra.wav" "$TMP/sa.wav"
chk "B vivian/EN" "$TMP/rb.wav" "$TMP/sb.wav"
chk "C serena/IT" "$TMP/rc.wav" "$TMP/sc.wav"
echo "--- cross-talk (expect FAIL = independent) ---"
xr=$(python3 "$CMP" "$TMP/rc.wav" "$TMP/sa.wav" 2>/dev/null)
echo "$xr" | grep -q FAIL && echo "  OK (independent) :: $xr" || { echo "  CROSS-TALK! :: $xr"; fail=$((fail+1)); }

# ── B) production smoke: matmat path batches + stream fallback works ──
echo "--- production smoke (matmat path) ---"
$BIN -d "$MODEL" --serve "$PORT" --batch-size 4 > "$TMP/srv2.log" 2>&1 &
sleep 6
timeout 90 curl -s "http://localhost:$PORT/v1/tts" -d '{"text":"Prova produzione uno.","speaker":"ryan","language":"Italian","seed":42}' -o "$TMP/pa.wav" & Q1=$!
timeout 90 curl -s "http://localhost:$PORT/v1/tts" -d '{"text":"Production test two.","speaker":"vivian","language":"English","seed":42}' -o "$TMP/pb.wav" & Q2=$!
timeout 90 curl -s "http://localhost:$PORT/v1/tts/stream" -d '{"text":"Stream fallback test.","speaker":"ryan","language":"English","seed":42}' -o "$TMP/ps.pcm" & Q3=$!
wait $Q1 $Q2 $Q3
grep -E "BATCH|Streamed" "$TMP/srv2.log" | tail -3
for f in pa.wav pb.wav ps.pcm; do
  if [ -s "$TMP/$f" ]; then echo "  OK  $f ($(wc -c <"$TMP/$f") bytes)"; else echo "  FAIL $f empty"; fail=$((fail+1)); fi
done
pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1

echo "=== serve_batch: $pass passed, $fail failed ==="
pgrep -fl qwen_tts >/dev/null && echo "WARNING: stray qwen_tts procs" || true
[ "$fail" -eq 0 ]
