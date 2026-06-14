#!/usr/bin/env bash
# RELEASE STEP-1 regression: every expressivity path × precision × voice still produces coherent
# audio (WAV valid + sane duration) + capture RTF. Single-mode first slice (the LoRA/instruct/clone
# paths that predated quant/batching were never re-tested). Prints a PASS/FAIL + RTF table.
# Usage: tests/regression_matrix.sh [model_dir] [out_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-qwen3-tts-1.7b}"
OUT="${2:-/tmp/regr}"; mkdir -p "$OUT"
EXPR="presets/expr/italian_bb027_ep5_r32.expr"
ICL="voices/galatea_icl.qvoice"
TEXT="Oggi è una giornata splendida e voglio raccontarti una storia che mi è rimasta nel cuore."
INSTR="Speak with warm, bright happiness, smiling through the words."
SEED=42; LANG=Italian
RES="$OUT/results.tsv"; : > "$RES"

run() {  # $1=label  $2..=extra qwen args
  local label="$1"; shift
  local wav="$OUT/${label}.wav"
  local log; log=$("./qwen_tts" -d "$MODEL" --text "$TEXT" --seed $SEED -l $LANG -T 0.9 -o "$wav" "$@" 2>&1)
  local rtf; rtf=$(echo "$log" | grep -oE "RTF [0-9.]+" | head -1 | awk '{print $2}')
  local dur; dur=$(python3 -c "import soundfile as sf;i=sf.info('$wav');print(round(i.frames/i.samplerate,2))" 2>/dev/null || echo "")
  local verdict="FAIL"
  if [ -n "$dur" ] && python3 -c "import sys; sys.exit(0 if float('$dur')>1.0 else 1)" 2>/dev/null; then verdict="PASS"; fi
  printf "%-34s\t%s\t%s\t%s\n" "$label" "${rtf:-NA}" "${dur:-NA}" "$verdict" | tee -a "$RES"
}

echo -e "CASE\tRTF\tDUR(s)\tVERDICT" | tee "$RES.head"
for PREC in bf16 int8 int4; do
  Q=""; [ "$PREC" = int8 ] && Q="--int8"; [ "$PREC" = int4 ] && Q="--int4"
  # preset ryan
  run "ryan_${PREC}_plain"        -s ryan $Q
  run "ryan_${PREC}_expr"         -s ryan $Q --expr "$EXPR" --expr-weight 0.6
  run "ryan_${PREC}_instruct"     -s ryan $Q -I "$INSTR"
  run "ryan_${PREC}_expr+instr"   -s ryan $Q --expr "$EXPR" --expr-weight 0.6 -I "$INSTR"
  # clone galatea (lite ICL)
  run "galicl_${PREC}_plain"      --load-voice "$ICL" $Q
  run "galicl_${PREC}_expr"       --load-voice "$ICL" $Q --expr "$EXPR" --expr-weight 0.6
  run "galicl_${PREC}_instruct"   --load-voice "$ICL" $Q -I "$INSTR"
  run "galicl_${PREC}_expr+instr" --load-voice "$ICL" $Q --expr "$EXPR" --expr-weight 0.6 -I "$INSTR"
done

echo "=== SUMMARY ==="
awk -F'\t' '{v[$4]++} END{for(k in v) printf "%s: %d\n", k, v[k]}' "$RES"
echo "PASS expected = 24 (3 prec × 8 cases). Full table: $RES"