#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model dir, e.g. qwen3-tts-0.6b}"
OUTDIR="${2:?output dir}"
shift 2
VOICES=("$@")

BIN=./qwen_tts
SEED=42
LANG=Italian
TEXT="Non ci posso credere, è davvero una notizia fantastica! Sono così felice per te."
EMO_DIR=presets/emotions/it
TONES=(neutral happy excited eager proud calm sad)

mkdir -p "$OUTDIR"

for vpath in "${VOICES[@]}"; do
  vname=$(basename "$vpath" .qvoice)
  echo "==== voice: $vname ($vpath) ===="
  for t in "${TONES[@]}"; do
    out="$OUTDIR/${vname}__${t}.wav"
    if [ "$t" = "neutral" ]; then
      "$BIN" -d "$MODEL" -j1 --seed $SEED -l "$LANG" \
        --load-voice "$vpath" --text "$TEXT" -o "$out" --silent
    else
      QWEN_EMOTION_DIR="$EMO_DIR" "$BIN" -d "$MODEL" -j1 --seed $SEED -l "$LANG" \
        --load-voice "$vpath" --emotion "$t" --text "$TEXT" -o "$out" --silent
    fi
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" 2>/dev/null || echo "?")
    printf "  %-10s -> %s  (%ss)\n" "$t" "$out" "$dur"
  done
done

echo
echo "Listen, grouped by tone (A/B across voices):"
echo "  for t in ${TONES[*]}; do for v in $(for p in "${VOICES[@]}"; do basename "$p" .qvoice; done | tr '\n' ' '); do echo \"\$v \$t\"; afplay $OUTDIR/\${v}__\${t}.wav; done; done"
