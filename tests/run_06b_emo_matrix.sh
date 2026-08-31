#!/bin/bash
set -u
cd "$(dirname "$0")/.."
D=samples/tests/2026-08-05_06b_full_matrix
mkdir -p "$D/donors" "$D/voices" "$D/out"
SMALL=qwen3-tts-0.6b
BIG=qwen3-tts-1.7b
BASE=qwen3-tts-0.6b-base
SEED=42
EVAL="Non è possibile che succeda sempre la stessa cosa, ogni singola volta."
LONG="Ma ti rendi conto di quello che è successo? Te l'avevo detto mille volte, mille volte, e alla fine è andata proprio così. Adesso non venirmi a dire che non lo sapevi, perché lo sapevi benissimo. È sempre la stessa storia, ogni volta la stessa identica storia, e io continuo a ripetere le cose a vuoto. Basta, davvero, non ne posso più di questa situazione."
EMOS="sad joy anger fear disgust surprise"

say() { echo; echo "=============== $* ==============="; }

say "PART 1: para su clone graft (16.8MB) nel piccolo"
for tag in sigh laugh wow yawn scoff; do
  ./qwen_tts -d $SMALL --load-voice voices/galatea_06b_graft.qvoice --icl-only \
    -l Italian --seed $SEED --text "[$tag] Non è possibile che succeda sempre la stessa cosa." \
    -o "$D/out/para_graft_$tag.wav" 2>&1 | grep -E "Paralinguistics|Audio:" | sed "s/^/  [$tag] /"
done

say "PART 2: donatori emotivi dal 1.7B (6 emozioni x 2 voci)"
for emo in $EMOS; do
  ./qwen_tts -d $BIG --load-voice voices/galatea_graft.qvoice --icl-only -l Italian --seed $SEED \
    --emotion "$emo" --text "$LONG" -o "$D/donors/galatea_$emo.wav" 2>&1 | grep -E "Audio:" | sed "s/^/  galatea-$emo /"
  ./qwen_tts -d $BIG -s ryan -l Italian --seed $SEED \
    --emotion "$emo" --text "$LONG" -o "$D/donors/ryan_$emo.wav" 2>&1 | grep -E "Audio:" | sed "s/^/  ryan-$emo /"
done
./qwen_tts -d $BIG --load-voice voices/galatea_graft.qvoice --icl-only -l Italian --seed $SEED \
  --text "$LONG" -o "$D/donors/galatea_neutral.wav" 2>&1 | grep -E "Audio:" | sed 's/^/  galatea-neutral /'
./qwen_tts -d $BIG -s ryan -l Italian --seed $SEED \
  --text "$LONG" -o "$D/donors/ryan_neutral.wav" 2>&1 | grep -E "Audio:" | sed 's/^/  ryan-neutral /'

say "PART 3: extract the 4 KB x-vector from each donor"
for v in galatea ryan; do
  for emo in $EMOS neutral; do
    ./qwen_tts -d $BASE --ref-audio "$D/donors/${v}_${emo}.wav" \
      --save-voice "$D/voices/${v}_${emo}.bin" --silent >/dev/null 2>&1
    [ -f "$D/voices/${v}_${emo}.bin" ] && echo "  ok ${v}_${emo}.bin ($(stat -f%z "$D/voices/${v}_${emo}.bin") B)" \
                                      || echo "  FAIL ${v}_${emo}"
  done
done

say "PART 4: matrice sul PICCOLO — 6 emozioni x 2 voci (asset 4KB)"
for v in galatea ryan; do
  for emo in $EMOS neutral; do
    ./qwen_tts -d $SMALL --load-voice "$D/voices/${v}_${emo}.bin" --xvector-only \
      -l Italian --seed $SEED --text "$EVAL" -o "$D/out/${v}_${emo}.wav" 2>&1 \
      | grep -E "Audio:" | sed "s/^/  ${v}-${emo} /"
  done
done

say "PART 5: graft emotivo (x-vector emotivo dentro il graft 16.8MB)"
for emo in anger sad joy; do
  python3 tests/graft_set_xvector.py voices/galatea_06b_graft.qvoice \
    "$D/voices/galatea_${emo}.bin" "$D/voices/galatea_${emo}_graft.qvoice" 2>&1 | sed 's/^/  /'
  ./qwen_tts -d $SMALL --load-voice "$D/voices/galatea_${emo}_graft.qvoice" --icl-only \
    -l Italian --seed $SEED --text "$EVAL" -o "$D/out/graft_galatea_${emo}.wav" 2>&1 \
    | grep -E "Voice:|Audio:" | sed "s/^/  graft-${emo} /"
done

say "PART 6: TUTTO INSIEME — graft emotivo + tag para, nel piccolo"
for emo in anger sad joy; do
  ./qwen_tts -d $SMALL --load-voice "$D/voices/galatea_${emo}_graft.qvoice" --icl-only \
    -l Italian --seed $SEED --text "[sigh] Non è possibile che succeda sempre la stessa cosa." \
    -o "$D/out/all_${emo}_sigh.wav" 2>&1 | grep -E "Paralinguistics|Audio:" | sed "s/^/  ${emo}+sigh /"
done

say "DONE — $(ls "$D/out" | wc -l | tr -d ' ') file in $D/out"
