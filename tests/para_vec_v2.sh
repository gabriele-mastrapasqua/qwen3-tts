#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=samples/para_vectors; M=qwen3-tts-1.7b; IT=presets/expr/italian_csp_topk6.expr; SEED=42
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; *) echo "-s $1";; esac; }
ew(){ [ "$1" = galatea ] && echo 1.0 || echo 1.2; }
wt(){ [ "$1" = ryan ] && echo 6 || echo 8; }

EVENTS=(
"growl2|Snarl and growl with raw furious aggression, low gravelly chest growl, baring teeth, seething.|Grrr, how dare you, get away from me right now, grrr, I am so furious, grrr!|Speak in a soft tender gentle soothing tone, calm warm and reassuring.|It is all right, just calm down, everything is fine now, take a slow deep breath.|Come ti permetti di parlarmi cosi, grr, questo proprio non lo accetto.|Come ti permetti di parlarmi cosi, questo proprio non lo accetto."
"disgust2|React with intense revulsion and disgust, recoiling, sneering, going ugh and yuck and ewww.|Ugh, yuck, that is so gross and revolting, ewww, disgusting, it makes me sick, ugh!|Savor with warm delight and pleasure, going mmm, satisfied and pleased and content.|Mmm, this is so delicious and wonderful, mmm, so good, mmm, absolutely satisfying.|Che cosa disgustosa, bleah, non riesco nemmeno a guardarla.|Che cosa disgustosa, non riesco nemmeno a guardarla."
"gasp2|Gasp sharply in sudden shock, a quick sharp startled intake of breath, amazed and stunned.|Gasp! What?! Oh my, no way, huh, I really can't believe it, gasp!|Exhale a long slow weary sigh, breath flowing out slowly, deflated and tired.|Haaah, a long slow tired sigh, breathing out slowly, so weary and drained.|Oh, non ci posso credere, ah, non me lo aspettavo per niente!|Oh, non ci posso credere, non me lo aspettavo per niente!"
)
build(){ local n="$1" ev="$2" evt="$3" op="$4" opt="$5"
  QWEN_ACT_MAP=$O/_${n}_event.qamp ./qwen_tts -d $M -s ryan -l English --seed $SEED -T 1.1 -I "$ev" --text "$evt" -o $O/_${n}_event.wav >/dev/null 2>&1
  QWEN_ACT_MAP=$O/_${n}_opp.qamp   ./qwen_tts -d $M -s ryan -l English --seed $SEED -T 1.1 -I "$op" --text "$opt" -o $O/_${n}_opp.wav   >/dev/null 2>&1
  python3 tests/act_map_steer.py $O/_${n}_opp.qamp $O/_${n}_event.qamp $O/${n}.qlsteer --unit-per-layer >/dev/null 2>&1
  echo "  built ${n}.qlsteer  (event $(dur $O/_${n}_event.wav) / opp $(dur $O/_${n}_opp.wav))"
}
gen(){ local sp="$1" n="$2" txt="$3" tag="$4" ins="$5" w; w=$(wt $sp)
  ./qwen_tts -d $M $(vf $sp) -l Italian --seed $SEED -T 1.1 --expr $IT --expr-weight $(ew $sp) -I "$ins" \
    --ml-steer $O/${n}.qlsteer --ml-weight $w --ml-range 21-25 --text "$txt" -o $O/${sp}_${n}_${tag}_w${w}.wav >/dev/null 2>&1
  echo "  ${sp}_${n}_${tag}_w${w} -> $(dur $O/${sp}_${n}_${tag}_w${w}.wav)"
}
echo "===== PARA-VEC v2 retry -> $O ====="
for e in "${EVENTS[@]}"; do
  IFS='|' read -r n ev evt op opt ctag cnotag <<< "$e"
  echo "--- [$n] build (event-specific source) ---"; build "$n" "$ev" "$evt" "$op" "$opt"
  echo "--- [$n] grid: tag vs no-tag ---"
  for sp in galatea vivian ryan; do gen "$sp" "$n" "$ctag" tag "$ev"; gen "$sp" "$n" "$cnotag" notag "$ev"; done
done
echo "===== DONE -> $O ====="
