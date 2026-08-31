#!/usr/bin/env bash
# seed_sweep.sh — hunt "seed magics" for an emotion.
#
# WHY: with temp>0 the seed fixes the ENTIRE sampling trajectory (one LCG, qwen_tts_sampling.c). At
# expressive temp+steering the conditional is wide, so different seeds realize VERY different valid
# renderings of the SAME emotion instruct — some "rage", some "easy", some glitch. A fixed seed can
# systematically land on a flat/broken mode. This sweeps N seeds at a FIXED (pack,weight,temp,instruct,
# text) and reports duration as a cheap STABILITY proxy (runaway/glitch renders are abnormally long or
# abnormally short vs the median). Ear-pick the winners; copy good seeds into a per-emotion pool.
#
# Usage:
#   tests/seed_sweep.sh -d qwen3-tts-1.7b -s ryan -l Spanish \
#       --expr presets/expr/italian_csp.expr --expr-weight 1.6 -T 1.3 \
#       -I "Speak with strong disgust, repulsed and contemptuous." \
#       --text "No puedo creer lo que ha pasado hoy." \
#       --seeds "7 42 123 777 999 2024 31337" --out /tmp/sweep_disgust
#
# Then:  afplay /tmp/sweep_disgust/seed_<N>.wav   # ear-judge; the report flags length outliers.
set -uo pipefail

MODEL="qwen3-tts-1.7b"; SPK="ryan"; LANG="Spanish"; TEMP="1.3"
EXPR=""; EXPRW="1.6"; INSTR=""; TEXT="No puedo creer lo que ha pasado hoy."
SEEDS="7 42 123 777 999 2024 31337"; OUTDIR="/tmp/seed_sweep"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    -d) MODEL="$2"; shift 2;;
    -s) SPK="$2"; shift 2;;
    -l) LANG="$2"; shift 2;;
    -T) TEMP="$2"; shift 2;;
    --expr) EXPR="$2"; shift 2;;
    --expr-weight) EXPRW="$2"; shift 2;;
    -I|--instruct) INSTR="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --out) OUTDIR="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

mkdir -p "$OUTDIR"
BIN="./qwen_tts"
[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
EXPRARGS=(); [ -n "$EXPR" ] && EXPRARGS=(--expr "$EXPR" --expr-weight "$EXPRW")
INSTRARGS=(); [ -n "$INSTR" ] && INSTRARGS=(-I "$INSTR")

echo "sweep: model=$MODEL spk=$SPK lang=$LANG T=$TEMP expr=${EXPR:-none} w=$EXPRW"
echo "instruct: ${INSTR:-<none>}"
echo "text: $TEXT"
echo "seeds: $SEEDS"
echo "----"
declare -a DUR
for s in $SEEDS; do
  f="$OUTDIR/seed_${s}.wav"
  "$BIN" -d "$MODEL" -s "$SPK" -l "$LANG" -T "$TEMP" --seed "$s" --silent \
      "${EXPRARGS[@]}" "${INSTRARGS[@]}" "${EXTRA[@]}" --text "$TEXT" -o "$f" >/dev/null 2>&1
  # duration via wav byte size (24kHz/16-bit mono => 48000 B/s, 44-byte header)
  bytes=$(wc -c < "$f" 2>/dev/null || echo 0)
  secs=$(awk "BEGIN{printf \"%.2f\", ($bytes-44)/48000.0}")
  printf "seed %-7s -> %6.2fs  %s\n" "$s" "$secs" "$f"
done | tee "$OUTDIR/_report.txt"
echo "----"
echo "median-outliers (too long = runaway/glitch, too short = broken) — ear-judge the rest:"
awk '{print $2, $4}' "$OUTDIR/_report.txt" 2>/dev/null | sort -k2 -n | awk '{print "  ", $0}'
echo "tip: copy a clean+expressive seed into your per-emotion recipe; this is a FREE expressivity lever."
