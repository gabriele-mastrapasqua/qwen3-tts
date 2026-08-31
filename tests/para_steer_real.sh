#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=samples/para_steer_real; rm -rf $O; mkdir -p $O
K=samples/para_real_clips/codes; M=qwen3-tts-1.7b; IT=presets/expr/italian_csp_topk6.expr; SEED=42
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; *) echo "-s $1";; esac; }
ew(){ [ "$1" = galatea ] && echo 1.0 || echo 1.2; }

QWEN_TF_CODES=$K/neutral.codes QWEN_ACT_MAP=$K/neutral_real.qamp ./qwen_tts -d $M -s ryan -l English --seed $SEED -T 0.7 --text "Listen to this now." -o /tmp/_n.wav >/dev/null 2>&1

EVENTS=(
"cough|Scusatemi un momento, devo schiarirmi la voce prima di continuare."
"sneeze|Aspetta un attimo, scusa, non riesco proprio a trattenermi."
"sniff|Che raffreddore terribile, non mi passa piu da giorni."
"throatclear|Allora, dunque, vorrei dire una cosa molto importante."
)
echo "===== STEER-FROM-REAL -> $O ====="
for e in "${EVENTS[@]}"; do
  IFS='|' read -r name carrier <<< "$e"
  QWEN_TF_CODES=$K/${name}.codes QWEN_ACT_MAP=$K/${name}_real.qamp ./qwen_tts -d $M -s ryan -l English --seed $SEED -T 0.7 --text "Listen to this now." -o /tmp/_e.wav >/dev/null 2>&1
  python3 tests/act_map_steer.py $K/neutral_real.qamp $K/${name}_real.qamp $O/${name}.qlsteer --unit-per-layer 2>/dev/null | grep -E "L2[0-7]:|final" | tail -4
  echo "--- [$name] inject ---"
  for sp in galatea vivian ryan; do
    for w in 8 12; do
      ./qwen_tts -d $M $(vf $sp) -l Italian --seed $SEED -T 1.0 --expr $IT --expr-weight $(ew $sp) \
        --ml-steer $O/${name}.qlsteer --ml-weight $w --ml-range 21-25 --text "$carrier" -o $O/${sp}_${name}_w${w}.wav >/dev/null 2>&1
      echo "  ${sp}_${name}_w${w} -> $(dur $O/${sp}_${name}_w${w}.wav)"
    done
  done
done
echo "DONE -> $O"
