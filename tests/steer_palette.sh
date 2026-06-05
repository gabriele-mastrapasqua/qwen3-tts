#!/usr/bin/env bash
# Build the AUTHORITATIVE, calibrated emotion/delivery palette and a demo per tone.
#
# Each preset .vec = (mean cp_x[instruct] - mean cp_x[neutral]) * recommended_weight,
# captured on a multi-sentence text so content averages out and only the delivery
# remains. The recommended weight is BAKED IN, so the shipped vector is calibrated:
#   ./qwen_tts --emotion happy ...                 # uses the baked weight (= good)
#   ./qwen_tts --emotion happy --steer-weight 1.3  # push it further
#   ./qwen_tts --emotion happy:0.5,proud:0.5 ...   # blend two moods
#
# NOTE: tones are named by the RESULT we heard, not the literal instruct (the
# control-vector direction often lands on a nearby mood — that's the discovery).
#
# Usage: bash tests/steer_palette.sh [model_dir] [lang] [outdir]
set -euo pipefail

MODEL="${1:-qwen3-tts-1.7b}"     # instruct is 1.7B-only → capture needs 1.7B
LANG="${2:-English}"
OUTDIR="${3:-presets/emotions}"
SPK=ryan; SEED=42
BIN=./qwen_tts

if [ "$LANG" = "Italian" ]; then
  CAP_TEXT="La riunione è fissata per domani. Per favore controlla i documenti prima di allora. Ti manderò il rapporto finale via email stasera."
  DEMO_TEXT="Allora, ecco la notizia che tutti stavano aspettando."
  OUTDIR="${3:-presets/emotions/it}"
else
  CAP_TEXT="The meeting is scheduled for tomorrow. Please review the documents before then. I will send you the final report by email tonight."
  DEMO_TEXT="Well, here is the news everyone has been waiting for."
fi

mkdir -p "$OUTDIR"

# name | recommended weight | instruct (delivery prompt that produced this mood)
NAMES=(happy   excited eager   proud   sad     gloomy  news    dramatic calm)
WEIGHTS=(1.4   1.0     0.7     0.8     0.7     0.7     0.7     0.7      0.8)
INSTRUCTS=(
  "Speak in a happy, cheerful, upbeat tone, smiling"
  "Speak with high energy and excitement, enthusiastic and fast"
  "Speak in a manic, whimsical, unpredictable and zany way, full of energy and wild pitch swings"
  "Speak in a flamboyant, over-the-top, comedic cartoon character voice, exaggerated and lively"
  "Speak in a sad, sorrowful, downcast tone"
  "Speak in a dark, gloomy, somber tone"
  "Speak like a professional news anchor, clear and authoritative"
  "Speak in a dramatic, suspenseful, storytelling tone"
  "Speak in a calm, soft, soothing, relaxed tone"
)

echo "== [$LANG] neutral baseline =="
QWEN_STEER_CAPTURE="$OUTDIR/_neutral.vec" "$BIN" -d "$MODEL" --text "$CAP_TEXT" \
  --seed $SEED -s $SPK -l "$LANG" -o /tmp/_pal_neutral.wav --silent

for i in "${!NAMES[@]}"; do
  n="${NAMES[$i]}"; w="${WEIGHTS[$i]}"; ins="${INSTRUCTS[$i]}"
  echo "== [$LANG] capture: $n (baked weight $w) =="
  QWEN_STEER_CAPTURE="$OUTDIR/_$n.vec" "$BIN" -d "$MODEL" --text "$CAP_TEXT" \
    -I "$ins" --seed $SEED -s $SPK -l "$LANG" -o /tmp/_pal_$n.wav --silent
  python3 tests/steer_make.py "$OUTDIR/_$n.vec" "$OUTDIR/_neutral.vec" "$OUTDIR/$n.vec" --scale "$w"
  # demo applied at the baked calibration (global --steer-weight default 1.0);
  # force QWEN_EMOTION_DIR so the just-built (this-language) vec is the one used.
  QWEN_EMOTION_DIR="$OUTDIR" "$BIN" -d "$MODEL" --text "$DEMO_TEXT" --emotion "$n" \
    --seed $SEED -s $SPK -l "$LANG" -o "/tmp/demo_${LANG}_$n.wav" --silent
done

"$BIN" -d "$MODEL" --text "$DEMO_TEXT" --seed $SEED -s $SPK -l "$LANG" -o "/tmp/demo_${LANG}_neutral.wav" --silent
rm -f "$OUTDIR"/_*.vec
echo
echo "Calibrated palette -> $OUTDIR/   (weights baked; use: --emotion <name>)"
echo "Listen: afplay /tmp/demo_${LANG}_neutral.wav ; for n in ${NAMES[*]}; do echo \$n; afplay /tmp/demo_${LANG}_\$n.wav; done"
