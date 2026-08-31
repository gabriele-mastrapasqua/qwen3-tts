#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=samples/lora_r64ab; mkdir -p "$OUT"
TXT="Allora, lascia che ti spieghi come stanno le cose."
declare -A E=([r16]=presets/expr/italian_lora.expr [r32]=presets/expr/italian_lora_r32.expr [r64]=presets/expr/italian_lora_r64.expr)
C="-l Italian -T 1.1 --seed 42 -j1 --silent"
declare -A INST=(
  [sad]="Speak deeply sad and heartbroken, a slow broken voice on the verge of tears."
  [anger]="Speak with intense, heated, furious anger."
)
GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
for emo in sad anger; do
  for r in r16 r32 r64; do
    ./qwen_tts -d qwen3-tts-1.7b $GAL --expr "${E[$r]}" $C --instruct "${INST[$emo]}" --text "$TXT" -o "$OUT/gal_${emo}_${r}.wav" 2>/dev/null
  done
  for r in r32 r64; do
    ./qwen_tts -d qwen3-tts-1.7b -s vivian --expr "${E[$r]}" $C --instruct "${INST[$emo]}" --text "$TXT" -o "$OUT/viv_${emo}_${r}.wav" 2>/dev/null
  done
  echo -n "  gal $emo r32-vs-r64: "; python3 tests/compare_audio.py "$OUT/gal_${emo}_r32.wav" "$OUT/gal_${emo}_r64.wav" 2>&1 | tail -1
done
echo "DONE -> $OUT"
