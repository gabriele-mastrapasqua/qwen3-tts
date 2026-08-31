#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

LANG_NAME="${1:-Italian}"
EXPR="${2:-presets/expr/italian_l1626_r64.expr}"
MODEL="${3:-qwen3-tts-1.7b}"
OUT="${4:-samples/lora_${LANG_NAME,,}_matrix}"
SEED=42
TEXT="Allora, lascia che ti spieghi come stanno le cose."
mkdir -p "$OUT"
REPORT="$OUT/LISTEN.md"
EXPR_BASE="$(basename "$EXPR")"

EMOS=(
  "neutral|"
  "joy|Speak happily, bright and warm, smiling through the words."
  "anger|Speak with hot, furious anger, sharp and forceful."
  "sadness|Speak with a sad, sorrowful, downcast tone, voice low and heavy."
  "fear|Speak with fear, tense and trembling, your voice wary."
  "surprise|Speak with surprise, startled and taken aback, held through the whole sentence."
  "disgust|Speak with physical disgust, repulsed and recoiling."
)
VOICES=(
  "galatea-smallICL|CLONED (small ICL file)|small ICL clone (ref_codes — NOT the 3GB WDELTA); timbre from ICL, emotion from L16-26 LoRA + instruct|--load-voice voices/galatea_icl.qvoice"
  "galatea-heavyWDELTA|CLONED (heavy 3GB qvoice WDELTA)|full WDELTA weight-swap clone (the heavy path) + L16-26 LoRA + instruct — for comparison vs small ICL|--load-voice voices/galatea_graft.qvoice"
  "ryan-preset|PRESET (no clone)|preset voice ryan + L16-26 LoRA + instruct (the strong-emotion reference)|-s ryan"
)

sep="────────────────────────────────────────────────────────────────────────────"
: > "$REPORT"
{
  echo "# Emotion matrix — $LANG_NAME — LoRA \`$EXPR_BASE\` (L16-26 emotion band, alpha=2×r)"
  echo "model=$MODEL · seed=$SEED · text=\"$TEXT\" · generated $(date '+%Y-%m-%d %H:%M')"
  echo
} >> "$REPORT"

for V in "${VOICES[@]}"; do
  IFS='|' read -r vkey vkind vdesc vargs <<< "$V"
  {
    echo
    echo "$sep"
    echo "##  VOICE: $vkey  —  $vkind"
    echo "$sep"
  } >> "$REPORT"
  for T in 0.9 1.1; do
    for E in "${EMOS[@]}"; do
      emo="${E%%|*}"; instr="${E#*|}"
      wav="$OUT/${vkey}_${emo}_T${T}.wav"
      abs="$REPO/$wav"
      args=(-d "$MODEL" $vargs --expr "$EXPR" -l "$LANG_NAME" -T "$T" --seed "$SEED" -j1 --silent --text "$TEXT" -o "$wav")
      [ -n "$instr" ] && args+=(--instruct "$instr")
      cmd="./qwen_tts -d $MODEL $vargs --expr $EXPR -l $LANG_NAME -T $T --seed $SEED -j1"
      [ -n "$instr" ] && cmd="$cmd --instruct \"$instr\""
      cmd="$cmd --text \"$TEXT\" -o $wav"
      ./qwen_tts "${args[@]}" 2>/dev/null
      dur=$(python3 -c "import wave;w=wave.open('$wav');print(f'{w.getnframes()/w.getframerate():.2f}')" 2>/dev/null || echo ERR)
      {
        echo
        echo "### ${emo^^}  ·  T$T  ·  $vkind  ·  dur ${dur}s"
        echo "desc    : $vdesc"
        echo "instruct: ${instr:-(none — neutral anchor)}"
        echo
        echo "full command:"
        echo "    $cmd"
        echo
        echo "listen (copy-paste):"
        echo "    afplay $abs   # $vkey · ${emo} · T$T"
      } >> "$REPORT"
    done
  done
done

echo "### matrix done -> $REPORT"
cat "$REPORT"
