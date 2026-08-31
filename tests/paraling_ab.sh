#!/usr/bin/env bash
set -uo pipefail
EXPR="${1:?usage: paraling_ab.sh <expr-file> [speaker]}"
S="${2:-ryan}"
D=qwen3-tts-1.7b
SEED=42
T=1.1
OUT="${OUT:-/tmp/paraling_ab}"   # throwaway scratch -> /tmp (override with OUT=... ); not samples/
mkdir -p "$OUT"
Q() { echo "+ $*"; "$@"; }

MARK_EN="Oh that's so funny [laugh] I really can't believe it. [sigh] Anyway, let's keep going."
MARK_IT="Oh che ridere [laugh] non ci posso credere. [sigh] Vabbè, andiamo avanti."
PLAIN_EN="The weather today is quite nice, so I think we should go for a long walk in the park."
PLAIN_IT="Il tempo oggi è davvero bello, quindi penso che dovremmo fare una lunga passeggiata al parco."

echo "===== 1) MARKER RENDERING — English ====="
Q ./qwen_tts -d $D -s $S -l English --seed $SEED -T $T --text "$MARK_EN"                          -o "$OUT/1_marker_en_macro.wav"
Q ./qwen_tts -d $D -s $S -l English --seed $SEED -T $T --text "$MARK_EN" --no-compose             -o "$OUT/1_marker_en_base.wav"
Q ./qwen_tts -d $D -s $S -l English --seed $SEED -T $T --text "$MARK_EN" --no-compose --expr "$EXPR" -o "$OUT/1_marker_en_lora.wav"

echo "===== 1b) MARKER RENDERING — Italian (cross-lingual transfer) ====="
Q ./qwen_tts -d $D -s $S -l Italian --seed $SEED -T $T --text "$MARK_IT"                          -o "$OUT/1_marker_it_macro.wav"
Q ./qwen_tts -d $D -s $S -l Italian --seed $SEED -T $T --text "$MARK_IT" --no-compose             -o "$OUT/1_marker_it_base.wav"
Q ./qwen_tts -d $D -s $S -l Italian --seed $SEED -T $T --text "$MARK_IT" --no-compose --expr "$EXPR" -o "$OUT/1_marker_it_lora.wav"

echo "===== 2) PLAIN ENGLISH (no markers) — LoRA off vs on ====="
Q ./qwen_tts -d $D -s $S -l English --seed $SEED -T $T --text "$PLAIN_EN"                -o "$OUT/2_plain_en_off.wav"
Q ./qwen_tts -d $D -s $S -l English --seed $SEED -T $T --text "$PLAIN_EN" --expr "$EXPR" -o "$OUT/2_plain_en_on.wav"

echo "===== 3) PLAIN ITALIAN (no markers) — LoRA off vs on (anglicization check) ====="
Q ./qwen_tts -d $D -s $S -l Italian --seed $SEED -T $T --text "$PLAIN_IT"                -o "$OUT/3_plain_it_off.wav"
Q ./qwen_tts -d $D -s $S -l Italian --seed $SEED -T $T --text "$PLAIN_IT" --expr "$EXPR" -o "$OUT/3_plain_it_on.wav"

echo
echo "DONE -> $OUT/  (expr=$EXPR speaker=$S)"
echo "Listen, in order:"
echo "  1) marker:  1_marker_en_{macro,base,lora}.wav  +  1_marker_it_{macro,base,lora}.wav"
echo "  2) plainEN: 2_plain_en_{off,on}.wav"
echo "  3) plainIT: 3_plain_it_{off,on}.wav"