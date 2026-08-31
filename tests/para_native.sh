#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=samples/para_native; rm -rf $O; mkdir -p $O
M=qwen3-tts-1.7b; IT=presets/expr/italian_csp_topk6.expr; SEED=42
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; *) echo "-s $1";; esac; }
ew(){ [ "$1" = galatea ] && echo 1.0 || echo 1.2; }
gen(){ local sp="$1" id="$2" ins="$3" txt="$4"
  ./qwen_tts -d $M $(vf $sp) -l Italian --seed $SEED -T 1.1 --expr $IT --expr-weight $(ew $sp) -I "$ins" \
    --text "$txt" -o $O/${sp}_${id}.wav >/dev/null 2>&1
  echo "  ${sp}_${id} -> $(dur $O/${sp}_${id}.wav)"
}
ITEMS=(
"disgust_ugh|React with strong disgust, a revolted ugh.|Che cosa orribile, ugh, non riesco nemmeno a guardarla."
"throatclear_ahem|Clear your throat audibly before speaking.|Allora, ahem, vorrei dire una cosa molto importante."
"tsk_tsk|Click your tongue in sharp disapproval, tsk.|Tsk tsk, non avresti proprio dovuto farlo."
"tsk_tch|Click your tongue in sharp disapproval, tch.|Tch, non avresti proprio dovuto farlo."
"yawn_haaa|Yawn widely and sleepily, a long tired yawn.|Sono cosi stanco, haaaah, ho bisogno di dormire."
"cough_zh|Cough hard, a real chesty cough.|Scusatemi un momento, 咳咳, fatemi continuare."
"cough_khh|Cough hard, a real chesty cough.|Scusatemi un momento, khh khh, fatemi continuare."
"pleasure_mmm|Savor with deep pleasure and delight, a satisfied mmm.|Che meraviglia, mmm, e semplicemente delizioso."
"joy_ahah|Burst out laughing with bright joyful giggles.|Non ci posso credere, ahah, e fantastico!"
"excited_woah|Speak full of thrilled excitement and energy, woah.|Woah, e incredibile, non vedo l'ora di iniziare!"
)
echo "===== PARA NATIVE (IT, cross-voice) -> $O ====="
for sp in ryan galatea vivian; do
  echo "--- $sp ---"
  for it in "${ITEMS[@]}"; do
    IFS='|' read -r id ins txt <<< "$it"
    gen "$sp" "$id" "$ins" "$txt"
  done
done
echo "DONE -> $O"
