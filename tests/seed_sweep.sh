#!/usr/bin/env bash
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
  bytes=$(wc -c < "$f" 2>/dev/null || echo 0)
  secs=$(awk "BEGIN{printf \"%.2f\", ($bytes-44)/48000.0}")
  printf "seed %-7s -> %6.2fs  %s\n" "$s" "$secs" "$f"
done | tee "$OUTDIR/_report.txt"
echo "----"
echo "median-outliers (too long = runaway/glitch, too short = broken) — ear-judge the rest:"
awk '{print $2, $4}' "$OUTDIR/_report.txt" 2>/dev/null | sort -k2 -n | awk '{print "  ", $0}'
echo "tip: copy a clean+expressive seed into your per-emotion recipe; this is a FREE expressivity lever."
