#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=samples/clone_boost; mkdir -p "$OUT"
TXT="Allora, lascia che ti spieghi come stanno le cose."
R32=presets/expr/italian_lora_r32.expr
GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
C="-l Italian -T 1.1 --seed 42 -j1 --silent"
declare -A INST=(
  [sad]="Speak deeply sad and heartbroken, a slow broken voice on the verge of tears."
  [anger]="Speak with intense, heated, furious anger."
)
for emo in sad anger; do
  for w in 1.0 1.5 2.0 2.5; do
    out="$OUT/gal_${emo}_w${w}.wav"
    ./qwen_tts -d qwen3-tts-1.7b $GAL --expr $R32 --expr-weight $w $C --instruct "${INST[$emo]}" --text "$TXT" -o "$out" 2>/dev/null
    dur=$(python3 -c "import wave;w=wave.open('$out');print(f'{w.getnframes()/w.getframerate():.2f}')" 2>/dev/null || echo "ERR")
    echo "  gal $emo w=$w -> ${dur}s"
  done
done
echo "DONE -> $OUT  (collapse if dur << 2s)"
