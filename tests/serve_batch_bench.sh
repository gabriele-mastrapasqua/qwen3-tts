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
case "$MODEL" in *-base|*-base/) echo "REFUSING TO RUN: '$MODEL' is a Base (clone-only) model; this bench posts speaker=$SPK to /v1/tts, which a Base model rejects. Use a CustomVoice model dir (qwen3-tts-0.6b / qwen3-tts-1.7b)."; exit 2;; esac

now(){ python3 -c 'import time;print(time.time())'; }
audio_s(){ python3 -c "import wave,sys;print(round(wave.open(sys.argv[1]).getnframes()/24000,3))" "$1" 2>/dev/null || echo 0; }
body(){ printf '{"text":"%s","speaker":"%s","language":"%s","temperature":0,"seed":%s}' "$TXT" "$SPK" "$LNG" "$SEED"; }

run_prec(){ # $1=label  $2=extra qwen flag (e.g. --int8 or "")
  local label="$1" flag="$2"
  pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1
  $BIN -d "$MODEL" --serve "$PORT" --batch-size "$N" -j "$TH" $flag > "$TMP/srv.log" 2>&1 &
  # wait for the banner, not a fixed sleep: a slow load (int4 quantises at start-up, a big
  # model, a cold page cache) used to leave every curl refused and the row read 0/M ok
  local w; for w in $(seq 1 180); do grep -q "Server listening" "$TMP/srv.log" 2>/dev/null && break; sleep 1; done
  grep -q "Server listening" "$TMP/srv.log" 2>/dev/null || { echo "  $label: server did not come up in 180 s (see $TMP/srv.log)"; return; }
  timeout 180 curl -s -H "Content-Type: application/json" "http://localhost:$PORT/v1/tts" -d "$(body)" -o "$TMP/warm.wav" >/dev/null 2>&1
  local s0 s1 single_wall aud
  local cw; s0=$(now); cw=$(timeout 180 curl -s -H "Content-Type: application/json" -w '%{http_code} %{time_starttransfer}' "http://localhost:$PORT/v1/tts" -d "$(body)" -o "$TMP/base.wav" 2>/dev/null); s1=$(now)
  local single_code=${cw%% *} single_ttfb=${cw##* }
  if [ "$single_code" != "200" ]; then
    echo "  $label: FAIL — HTTP ${single_code:-none} from /v1/tts, no numbers for this row. Server said: $(head -c 160 "$TMP/base.wav" 2>/dev/null | tr '\n' ' ')"
    pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1; return
  fi
  single_wall=$(python3 -c "print(round($s1-$s0,2))")
  aud=$(audio_s "$TMP/base.wav")
  local single_rtf="n/a"; [ "$aud" != "0" ] && single_rtf=$(python3 -c "print(round($single_wall/$aud,2))")
  local b0 b1 burst_wall pids=""
  b0=$(now)
  for i in $(seq 1 "$M"); do
    timeout 240 curl -s -H "Content-Type: application/json" -w '%{time_starttransfer}\n' "http://localhost:$PORT/v1/tts" -d "$(body)" -o "$TMP/c_$i.wav" > "$TMP/ttfb_$i" 2>/dev/null &
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
  local burst_ttfb; burst_ttfb=$(cat "$TMP"/ttfb_* 2>/dev/null | python3 -c "import sys;v=sorted(float(x) for x in sys.stdin.read().split() if x);print(round(v[len(v)//2],2) if v else 'n/a')")
  printf "  %-8s single_RTF %-5s TTFB %-5ss | %d clients: burst %5ss  aggRTF %-5s  speedup %-5s  TTFB p50 %-5ss  (%d/%d ok)\n" \
         "$label" "$single_rtf" "${single_ttfb:-n/a}" "$M" "$burst_wall" "$agg_rtf" "${speedup}x" "$burst_ttfb" "$ok" "$M"
  pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1
}

echo "── server request-batching THROUGHPUT (model=$MODEL  batch=$N  clients=$M  -j$TH) ──"
echo "  speedup = (M × single_wall) / burst_wall  ;  aggRTF = burst_wall / total_audio"
echo "  TTFB = curl time_starttransfer on the NON-streaming /v1/tts: the WAV header goes out after the whole synthesis, so here TTFB ~= total"
echo "  (speedup >1 = real throughput win from weight-stationary batching; ~$N is the ceiling)"
run_prec "bf16" ""
run_prec "int8" "--int8"
run_prec "int4" "--int4"
echo "  (continuous admission itself is gated by tests/serve_continuous_stress.sh)"
