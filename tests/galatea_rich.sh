#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=samples/galatea_rich; mkdir -p "$OUT"
TXT="Allora, lascia che ti spieghi come stanno le cose."
R64=presets/expr/italian_lora_r64.expr; R128=presets/expr/italian_lora_r128.expr
GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
C="-l Italian -T 1.1 --seed 42 -j1 --silent"
declare -A INST=(
  [bitter]="Speak with bitter, resentful sarcasm, cold and wounded."
  [melancholic]="Speak with deep melancholy, wistful and aching, lingering on the words."
  [ecstatic]="Speak in pure ecstatic joy, overflowing, almost breathless with delight."
  [terrified]="Speak terrified, trembling, voice breaking with dread."
  [playful]="Speak playfully and lightly, a teasing grin in the voice."
  [desperate]="Speak with raw desperation, pleading and urgent, clutching at hope."
  [vengeful]="Speak with cold, vengeful menace, promising payback, slow and deliberate."
  [relieved]="Speak with deep relief, a long exhale, tension melting away."
  [ashamed]="Speak with shame, halting and embarrassed, wishing to disappear."
  [proud]="Speak with warm pride, chin up, savoring the accomplishment."
)
for emo in bitter melancholic ecstatic terrified playful desperate vengeful relieved ashamed proud; do
  ./qwen_tts -d qwen3-tts-1.7b $GAL --expr $R64  $C --instruct "${INST[$emo]}" --text "$TXT" -o "$OUT/${emo}_r64.wav"  2>/dev/null
  ./qwen_tts -d qwen3-tts-1.7b $GAL --expr $R128 $C --instruct "${INST[$emo]}" --text "$TXT" -o "$OUT/${emo}_r128.wav" 2>/dev/null
  d64=$(python3 -c "import wave;w=wave.open('$OUT/${emo}_r64.wav');print(f'{w.getnframes()/w.getframerate():.1f}')" 2>/dev/null)
  d128=$(python3 -c "import wave;w=wave.open('$OUT/${emo}_r128.wav');print(f'{w.getnframes()/w.getframerate():.1f}')" 2>/dev/null)
  echo "  $emo: r64 ${d64}s / r128 ${d128}s"
done
echo "DONE -> $OUT  (afplay ${OUT}/<emotion>_r64.wav vs _r128.wav)"
