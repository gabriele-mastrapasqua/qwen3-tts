#!/usr/bin/env bash
set -uo pipefail
cd /Users/gabrielemastrapasqua/source/personal/qwen-tts
O=samples/para_splice; rm -rf $O; mkdir -p $O
C=samples/para_real_clips; M=qwen3-tts-1.7b; IT=presets/expr/italian_csp_topk6.expr; SEED=42
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; *) echo "-s $1";; esac; }
ew(){ [ "$1" = galatea ] && echo 1.0 || echo 1.2; }
synth(){ ./qwen_tts -d $M $(vf "$1") -l Italian --seed $SEED -T 1.0 --expr $IT --expr-weight $(ew "$1") --text "$2" -o "$3" >/dev/null 2>&1; }

EVENTS=(
"cough|cough_1|Scusatemi un momento,|devo schiarirmi la voce prima di continuare."
"sneeze|sneeze_1|Aspetta un attimo,|scusa, non riesco proprio a trattenermi."
"sniff|sniff_2|Che raffreddore terribile,|non mi passa piu da giorni."
"throatclear|throatclear_2|Allora, dunque,|vorrei dire una cosa molto importante."
)
echo "===== REAL-AUDIO SPLICE -> $O ====="
for sp in galatea ryan vivian; do
  for e in "${EVENTS[@]}"; do
    IFS='|' read -r name clip pre post <<< "$e"
    synth "$sp" "$pre" $O/_${sp}_${name}_pre.wav
    synth "$sp" "$post" $O/_${sp}_${name}_post.wav
    python3 tests/para_splice.py $O/${sp}_${name}.wav $O/_${sp}_${name}_pre.wav $C/${clip}.wav $O/_${sp}_${name}_post.wav
  done
done
rm -f $O/_*.wav
echo "DONE -> $O"
