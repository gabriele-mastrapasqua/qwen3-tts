#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=samples/para_ft_test; rm -rf $O; mkdir -p $O
M=qwen3-tts-1.7b; EXPR=presets/expr/para_nonverbal_b027.expr; SEED=42
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; *) echo "-s $1";; esac; }
ITEMS=(
"laugh|Non ci posso credere, [laugh] e una notizia fantastica!"
"cough|Scusatemi un momento, [cough] fatemi continuare."
"sigh|Che giornata pesante, [sigh] sono davvero stanco."
"sniff|Che raffreddore terribile, [sniff] non mi passa."
"breath|Allora dunque, [breath] vorrei dire una cosa importante."
)
echo "===== PARA-FT EXP1 A/B  expr=$EXPR -> $O ====="
for sp in ryan galatea vivian; do
  echo "--- $sp ---"
  for it in "${ITEMS[@]}"; do
    IFS='|' read -r id txt <<< "$it"
    ./qwen_tts -d $M $(vf $sp) -l Italian --seed $SEED -T 1.0 --expr $EXPR --no-compose --text "$txt" -o $O/${sp}_${id}_FT.wav >/dev/null 2>&1
    echo "  ${sp}_${id}_FT -> $(dur $O/${sp}_${id}_FT.wav)"
  done
done
echo "--- ryan BASELINE (no expr) ---"
for it in "${ITEMS[@]}"; do
  IFS='|' read -r id txt <<< "$it"
  ./qwen_tts -d $M -s ryan -l Italian --seed $SEED -T 1.0 --no-compose --text "$txt" -o $O/ryan_${id}_BASE.wav >/dev/null 2>&1
  echo "  ryan_${id}_BASE -> $(dur $O/ryan_${id}_BASE.wav)"
done
echo "DONE -> $O"
