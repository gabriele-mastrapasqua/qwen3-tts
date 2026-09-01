#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
MODEL="qwen3-tts-0.6b"; FULL=0; SILICON=0
for a in "$@"; do
    case "$a" in
        --full)         FULL=1 ;;
        --silicon-only) SILICON=1 ;;
        -*)             echo "unknown option: $a"; exit 2 ;;
        *)              MODEL="$a" ;;
    esac
done
BIN=./qwen_tts
SEED=42; SPK=ryan; LANG=Italian
TXT="Quel ramo del lago di Como, che volge a mezzogiorno, viene a ristringersi. Don Abbondio tornava bel bello verso casa. Stava recitando tranquillamente il suo ufficio. Alzando gli occhi, vide due uomini fermi sul sentiero. Quel tipo di incontro non prometteva nulla di buono. Il povero curato si fermò di colpo, impietrito. Sentiva il cuore battergli forte nel petto."

[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
if [ "$SILICON" = "0" ]; then
    command -v python3 >/dev/null || { echo "python3 required for RTF"; exit 1; }
fi

hr(){ printf '%.0s─' {1..72}; echo; }
audio_s(){ python3 -c "import wave,sys; print(round(wave.open(sys.argv[1]).getnframes()/24000,2))" "$1" 2>/dev/null || echo 0; }
rtf_run(){ # $1=label  $2..=qwen args
  local label="$1"; shift
  local out=/tmp/bm_$$.wav
  local t0 t1 wall aud
  t0=$(python3 -c 'import time;print(time.time())')
  "$BIN" "$@" -o "$out" --silent >/dev/null 2>&1
  t1=$(python3 -c 'import time;print(time.time())')
  wall=$(python3 -c "print(round($t1-$t0,2))")
  aud=$(audio_s "$out")
  local rtf="n/a"
  [ "$aud" != "0" ] && rtf=$(python3 -c "print(round($wall/$aud,2))")
  printf "  %-22s wall %6ss  audio %6ss  RTF %s\n" "$label" "$wall" "$aud" "$rtf"
  rm -f "$out"
}

echo; hr
if [ "$SILICON" = "1" ]; then
    echo "  qwen-tts BOX REPORT (silicon only, no model)   $(date 2>/dev/null)"
else
    echo "  qwen-tts BENCH MATRIX   model=$MODEL   $(date 2>/dev/null)"
fi
hr

echo "### 0. Hardware inventory + memory bandwidth (tools/box_info.sh) ###"
HW_JSON="${HW_JSON:-/tmp/tts/box_info.json}"
if [ -r tools/box_info.sh ]; then
    mkdir -p "$(dirname "$HW_JSON")" 2>/dev/null
    bash tools/box_info.sh --out "$HW_JSON"
    [ -s "$HW_JSON" ] && echo "  (JSON confrontabile fra box: $HW_JSON)"
else
    echo "  tools/box_info.sh missing — hardware inventory skipped"
fi
echo

echo "### 1. SIMD capabilities (--caps) ###"
"$BIN" --caps 2>&1 | sed 's/^/  /'
echo

echo "### 2. Kernel correctness (--self-test) ###"
if "$BIN" --self-test >/tmp/bm_st.txt 2>&1; then echo "  native dispatch: PASS"; else echo "  native dispatch: FAIL (see /tmp/bm_st.txt)"; fi
if QWEN_NO_SDOT=1 QWEN_NO_VNNI=1 "$BIN" --self-test >/tmp/bm_st2.txt 2>&1; then echo "  scalar fallback: PASS"; else echo "  scalar fallback: FAIL (see /tmp/bm_st2.txt)"; fi
echo

echo "### 3. Batched matmat twins (--matmat-bench, B*matvec vs matmat) ###"
"$BIN" --matmat-bench 2>&1 | sed 's/^/  /'
echo "  --- single thread (compute-bound reference) ---"
"$BIN" --matmat-bench -j 1 2>&1 | sed 's/^/  /'
echo

if [ "$SILICON" = "1" ]; then
    hr
    echo "  This is the GROUND TRUTH ABOUT THE SILICON. Before any server number:"
    echo "    · --self-test red              -> the matrix below measures the wrong kernel"
    echo "    · --caps without the expected primitive -> it measures a FALLBACK, not this machine"
    echo "  Poi: ./download_model.sh --model small && make bench-matrix"
    hr
    exit 0
fi

echo "### 4. RTF matrix: single vs batched x precision (temp0, seed $SEED) ###"
for P in "bf16:" "int8:--int8" "int4:--int4"; do
  N=${P%%:*}; FLAG=${P#*:}
  echo "  --- $N ---"
  rtf_run "single"  "$BIN" -d "$MODEL" $FLAG -T 0 --seed "$SEED" -s "$SPK" -l "$LANG" --text "$TXT"
  rtf_run "batched" "$BIN" -d "$MODEL" $FLAG --batch --batch-words 14 -T 0 --seed "$SEED" -s "$SPK" -l "$LANG" --text "$TXT"
done
echo

if [ "$FULL" = "1" ]; then
  echo "### 5. Streaming (TTFA + RTF, int8) ###"
  "$BIN" -d "$MODEL" --int8 --stream -T 0 --seed "$SEED" -s "$SPK" -l "$LANG" --text "$TXT" -o /tmp/bm_stream.wav 2>&1 \
    | grep -iE "TTFA|RTF|Audio" | sed 's/^/  /'
  rm -f /tmp/bm_stream.wav
  echo

  echo "### 6. Server warm RTF (int8, 3 requests) ###"
  "$BIN" -d "$MODEL" --int8 --serve 8771 >/tmp/bm_srv.log 2>&1 &
  sleep 4
  for i in 1 2 3; do
    t0=$(python3 -c 'import time;print(time.time())')
    timeout 90 curl -s localhost:8771/v1/tts \
      -d "{\"text\":\"$TXT\",\"seed\":$SEED,\"speaker\":\"$SPK\",\"language\":\"$LANG\"}" -o /tmp/bm_sv.wav
    t1=$(python3 -c 'import time;print(time.time())')
    aud=$(audio_s /tmp/bm_sv.wav); wall=$(python3 -c "print(round($t1-$t0,2))")
    rtf="n/a"; [ "$aud" != "0" ] && rtf=$(python3 -c "print(round($wall/$aud,2))")
    printf "  req %d: wall %ss audio %ss RTF %s\n" "$i" "$wall" "$aud" "$rtf"
  done
  pkill -9 -f "qwen_tts.*--serve" 2>/dev/null
  rm -f /tmp/bm_sv.wav
  echo

  echo "### 7. Server request-batching THROUGHPUT (the x86 lever) ###"
  bash tests/serve_batch_bench.sh "$MODEL" 8901 4 4 4 2>&1 | sed 's/^/  /'
  echo
fi

hr
echo "  Paste this block + the --caps output into the per-box hardware notes."
echo "  Correctness gate (separate, run once): make test-serve-all"
hr
