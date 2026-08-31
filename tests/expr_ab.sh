#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
OUT=samples/expr_ab; mkdir -p "$OUT"
TXT="Allora, lascia che ti spieghi come stanno le cose."
BASE=qwen3-tts-1.7b
FT=qwen3-tts-1.7b-expr
EXPR=presets/expr/italian.expr
COMMON="-l Italian -T 1.1 --seed 42 -j1 --silent"

declare -A INST=(
  [neutral]=""
  [anger]="Speak with intense, heated, furious anger."
  [sad]="Speak deeply sad and heartbroken, a slow broken voice on the verge of tears."
)

run() { # $1=model-args  $2=out  $3=instruct  [extra...]
  local margs="$1" out="$2" inst="$3"; shift 3
  if [ -z "$inst" ]; then
    ./qwen_tts $margs $COMMON "$@" --text "$TXT" -o "$out" 2>/dev/null
  else
    ./qwen_tts $margs $COMMON "$@" --instruct "$inst" --text "$TXT" -o "$out" 2>/dev/null
  fi
}

echo "### VIVIAN (preset, Chinese-native voice speaking Italian)"
for e in neutral anger sad; do
  run "-d $BASE"          "$OUT/viv_base_$e.wav" "${INST[$e]}" -s vivian
  run "-d $BASE --expr $EXPR" "$OUT/viv_ft_$e.wav"   "${INST[$e]}" -s vivian
  echo -n "  viv $e: base-vs-ft(expr) "; python3 tests/compare_audio.py "$OUT/viv_base_$e.wav" "$OUT/viv_ft_$e.wav" 2>&1 | tail -1
done

echo "### PROOF: ft-via-expr == full 4GB checkpoint (vivian neutral)"
run "-d $FT" "$OUT/viv_ftckpt_neutral.wav" "" -s vivian
echo -n "  expr-vs-fullckpt: "; python3 tests/compare_audio.py "$OUT/viv_ft_neutral.wav" "$OUT/viv_ftckpt_neutral.wav" 2>&1 | tail -1

echo "### GALATEA (cloned Italian voice, --icl-only graft)"
GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
for e in neutral anger sad; do
  run "-d $BASE $GAL"          "$OUT/gal_base_$e.wav" "${INST[$e]}"
  run "-d $BASE $GAL --expr $EXPR" "$OUT/gal_ft_$e.wav"   "${INST[$e]}"
  echo -n "  gal $e: base-vs-ft(expr) "; python3 tests/compare_audio.py "$OUT/gal_base_$e.wav" "$OUT/gal_ft_$e.wav" 2>&1 | tail -1
done
echo "DONE -> $OUT"
