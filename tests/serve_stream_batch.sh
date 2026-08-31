#!/usr/bin/env bash
set -u
MODEL="${1:-qwen3-tts-0.6b}"
PORT="${2:-8788}"
BIN=./qwen_tts
TMP=$(mktemp -d)
trap 'pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; rm -rf "$TMP"' EXIT

pcmcorr() { # ref.wav  stream.pcm
python3 - "$1" "$2" <<'PY'
import sys, numpy as np, wave
ref, pcm = sys.argv[1], sys.argv[2]
w=wave.open(ref,'rb'); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32); w.close()
b=np.fromfile(pcm,dtype=np.int16).astype(np.float32)
if len(b)==0: print("FAIL empty stream"); sys.exit(1)
m=min(len(a),len(b)); a2,b2=a[:m],b[:m]
c=np.corrcoef(a2,b2)[0,1] if m>10 else 0.0
dr=abs(len(a)-len(b))/max(len(a),1)
ok = c>=0.98 and dr<=0.05
print(f"{'PASS' if ok else 'FAIL'} corr={c:.5f} ref={len(a)} stream={len(b)} dur_rel={dr:.1%}")
sys.exit(0 if ok else 1)
PY
}

echo "=== streaming×batching test (model=$MODEL port=$PORT) ==="
pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1
QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --serve "$PORT" --batch-size 4 > "$TMP/srv.log" 2>&1 &
sleep 6

A='{"text":"Primo flusso in streaming dal server batched.","speaker":"ryan","language":"Italian","temperature":0,"seed":100}'
B='{"text":"Second streaming flow served concurrently.","speaker":"vivian","language":"English","temperature":0,"seed":200}'
timeout 90 curl -s "http://localhost:$PORT/v1/tts/stream" -d "$A" -o "$TMP/sa.pcm" & PA=$!
timeout 90 curl -s "http://localhost:$PORT/v1/tts/stream" -d "$B" -o "$TMP/sb.pcm" & PB=$!
wait $PA $PB
echo "--- batch log (should show streamed + batched admission) ---"; grep "BATCH\] done" "$TMP/srv.log" | tail -4
pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1

QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --text "Primo flusso in streaming dal server batched." -s ryan -l Italian -T 0 --seed 100 -j1 -o "$TMP/ra.wav" --silent 2>/dev/null
QWEN_BATCH_FORCE_MATVEC=1 $BIN -d "$MODEL" --text "Second streaming flow served concurrently." -s vivian -l English -T 0 --seed 200 -j1 -o "$TMP/rb.wav" --silent 2>/dev/null

fail=0
echo "--- streamed-vs-single-stream (expect PASS) ---"
echo -n "  A ryan/IT:   "; pcmcorr "$TMP/ra.wav" "$TMP/sa.pcm" || fail=1
echo -n "  B vivian/EN: "; pcmcorr "$TMP/rb.wav" "$TMP/sb.pcm" || fail=1

echo "=== streaming×batch: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ==="
[ $fail -eq 0 ]
