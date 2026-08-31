#!/usr/bin/env bash
set -u
MODEL="${1:-qwen3-tts-0.6b}"
PORT="${2:-8900}"
N="${3:-4}"          # --batch-size
M="${4:-4}"          # concurrent clients
TH="${5:-4}"         # -j threads
BIN=./qwen_tts
SPK=ryan; LNG=Italian; SEED=42
TXT="Quel ramo del lago di Como, che volge a mezzogiorno, viene a ristringersi. Don Abbondio tornava bel bello verso casa, recitando tranquillamente il suo ufficio."
TMP=$(mktemp -d)
trap 'pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; rm -rf "$TMP"' EXIT

[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
[ -d "$MODEL" ] || { echo "model '$MODEL' not found"; exit 1; }

now(){ python3 -c 'import time;print(time.time())'; }
audio_s(){ python3 -c "import wave,sys;print(round(wave.open(sys.argv[1]).getnframes()/24000,3))" "$1" 2>/dev/null || echo 0; }
body(){ printf '{"text":"%s","speaker":"%s","language":"%s","temperature":0,"seed":%s}' "$TXT" "$SPK" "$LNG" "$SEED"; }

run_prec(){ # $1=label  $2=extra qwen flag (e.g. --int8 or "")
  local label="$1" flag="$2"
  pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1
  $BIN -d "$MODEL" --serve "$PORT" --batch-size "$N" -j "$TH" $flag > "$TMP/srv.log" 2>&1 &
  sleep 6
  timeout 180 curl -s "http://localhost:$PORT/v1/tts" -d "$(body)" -o "$TMP/warm.wav" >/dev/null 2>&1
  local s0 s1 single_wall aud
  s0=$(now); timeout 180 curl -s "http://localhost:$PORT/v1/tts" -d "$(body)" -o "$TMP/base.wav" >/dev/null 2>&1; s1=$(now)
  single_wall=$(python3 -c "print(round($s1-$s0,2))")
  aud=$(audio_s "$TMP/base.wav")
  local single_rtf="n/a"; [ "$aud" != "0" ] && single_rtf=$(python3 -c "print(round($single_wall/$aud,2))")
  local b0 b1 burst_wall pids=""
  b0=$(now)
  for i in $(seq 1 "$M"); do
    timeout 240 curl -s "http://localhost:$PORT/v1/tts" -d "$(body)" -o "$TMP/c_$i.wav" >/dev/null 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p"; done
  b1=$(now); burst_wall=$(python3 -c "print(round($b1-$b0,2))")
  local total_aud=0 ok=0
  for i in $(seq 1 "$M"); do
    local a; a=$(audio_s "$TMP/c_$i.wav")
    [ "$a" != "0" ] && { total_aud=$(python3 -c "print(round($total_aud+$a,3))"); ok=$((ok+1)); }
  done
  local speedup="n/a" agg_rtf="n/a"
  if [ "$ok" -gt 0 ] && [ "$single_wall" != "0" ]; then
    speedup=$(python3 -c "print(round($M*$single_wall/$burst_wall,2))")
    agg_rtf=$(python3 -c "print(round($burst_wall/$total_aud,2))")
  fi
  printf "  %-8s single_RTF %-5s | %d clients: burst %5ss  aggRTF %-5s  speedup %-5s  (%d/%d ok)\n" \
         "$label" "$single_rtf" "$M" "$burst_wall" "$agg_rtf" "${speedup}x" "$ok" "$M"
  pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1
}

echo "── server request-batching THROUGHPUT (model=$MODEL  batch=$N  clients=$M  -j$TH) ──"
echo "  speedup = (M × single_wall) / burst_wall  ;  aggRTF = burst_wall / total_audio"
echo "  (speedup >1 = real throughput win from weight-stationary batching; ~$N is the ceiling)"
run_prec "bf16" ""
run_prec "int8" "--int8"
run_prec "int4" "--int4"
echo "  (continuous admission itself is gated by tests/serve_continuous_stress.sh)"
