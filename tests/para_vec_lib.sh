#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=samples/para_vectors; mkdir -p $O
M=qwen3-tts-1.7b
IT=presets/expr/italian_csp_topk6.expr
SEED=42
EVTXT="Hahaha wow, oh no, mmm, ugh, ahh, grr, that is really something else entirely."
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; *) echo "-s $1";; esac; }
ew(){ [ "$1" = galatea ] && echo 1.0 || echo 1.2; }

EVENTS=(
"growl|Snarl and growl with furious aggression, voice harsh, low and gravelly, baring teeth in rage.|Speak in a soft, tender, gentle and soothing tone, calm and warm.|Come ti permetti di parlarmi cosi, grr, questo proprio non lo accetto.|6,8"
"disgust|React with strong revulsion and disgust, recoiling, going ugh and yuck, sneering with contempt.|Savor with delight and pleasure, going mmm, warm and satisfied and pleased.|Che cosa disgustosa, bleah, non riesco nemmeno a guardarla.|6,8"
"gasp|Gasp sharply in shock and surprise, a sudden sharp intake of breath, startled and amazed.|Exhale a long slow weary sigh, breath flowing out slowly, deflated and tired.|Oh, non ci posso credere, ah, non me lo aspettavo per niente!|6,8"
)
[ "${WAVE:-1}" = 2 ] && EVENTS=(
"moan|Groan and moan in pain, voice strained and aching, wincing and suffering.|Laugh brightly with joyful giggles, light and happy and amused.|Ahi, mi fa cosi male, ohh, non riesco a sopportarlo.|6,8"
"hmm|Hum thoughtfully and ponder, going hmm, pensive and considering, thinking it over.|Answer with flat certainty and finality, decisive and sure, no hesitation.|Aspetta, hmm, fammi pensare bene a come risolverlo.|6,8"
"gasp_relief|Sigh with deep relief, tension melting away, breathing out in calm relief.|Tense up sharply with sudden alarm and fright, breath catching.|Per fortuna e finita, ahh, che sollievo, temevo il peggio.|6,8"
)

build(){ local name="$1" ev="$2" op="$3"
  QWEN_ACT_MAP=$O/_${name}_event.qamp ./qwen_tts -d $M -s ryan -l English --seed $SEED -T 1.1 \
    -I "$ev" --text "$EVTXT" -o $O/_${name}_event.wav >/dev/null 2>&1
  QWEN_ACT_MAP=$O/_${name}_opp.qamp   ./qwen_tts -d $M -s ryan -l English --seed $SEED -T 1.1 \
    -I "$op" --text "$EVTXT" -o $O/_${name}_opp.wav   >/dev/null 2>&1
  python3 tests/act_map_steer.py $O/_${name}_opp.qamp $O/_${name}_event.qamp $O/${name}.qlsteer --unit-per-layer >/dev/null 2>&1
  echo "  built $O/${name}.qlsteer  (event $(dur $O/_${name}_event.wav) / opp $(dur $O/_${name}_opp.wav))"
}
testgrid(){ local name="$1" txt="$2" ws="$3" ins="$4"
  for sp in galatea vivian ryan; do
    IFS=',' read -ra WW <<< "$ws"
    for w in "${WW[@]}"; do
      ./qwen_tts -d $M $(vf $sp) -l Italian --seed $SEED -T 1.1 --expr $IT --expr-weight $(ew $sp) -I "$ins" \
        --ml-steer $O/${name}.qlsteer --ml-weight $w --ml-range 21-25 --text "$txt" \
        -o $O/${sp}_${name}_w${w}.wav >/dev/null 2>&1
      echo "  ${sp}_${name}_w${w} -> $(dur $O/${sp}_${name}_w${w}.wav)"
    done
  done
}

echo "===== PARA-VECTOR LIBRARY  wave=${WAVE:-1} -> $O ====="
for e in "${EVENTS[@]}"; do
  IFS='|' read -r name ev op txt ws <<< "$e"
  echo "--- [$name] capture+build ---"; build "$name" "$ev" "$op"
  echo "--- [$name] cross-voice grid ---"; testgrid "$name" "$txt" "$ws" "$ev"
done
echo "===== DONE -> $O ====="
