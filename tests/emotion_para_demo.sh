#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42
OUT="${EMO_PARA_DEMO_DIR:-samples/tests/emotion_para_demo}"; mkdir -p "$OUT"   # samples/tests/ is gitignored
[ -d "$M" ] || { echo "SKIP: $M not present — get it with ./download_model.sh"; exit 0; }

gen(){
  local out="$1"; shift
  local vf=(); while [ "$1" != "--" ]; do vf+=("$1"); shift; done; shift
  local emo="$1" lang="$2" txt="$3"
  printf '  %-28s ' "$out"
  if $BIN -d "$M" --seed $SEED "${vf[@]}" -l "$lang" --emotion "$emo" --text "$txt" \
        -o "$OUT/$out" 2>"$OUT/${out%.wav}.log"; then
    { grep -hoE "mode=[A-Z]+" "$OUT/${out%.wav}.log" | head -1 | tr -d '\n'; printf ' + '; \
      grep -hoE "paraling steer: [^ /]+/[^ ]+|composed [0-9]+ spans" "$OUT/${out%.wav}.log" | tr '\n' ' '; echo; }
  else echo "FAIL"; tail -3 "$OUT/${out%.wav}.log"; fi
}

echo "== ryan (preset) — Italian, emotion + paralinguistic =="
gen ryan_it_joy_laugh.wav      -s ryan -- joy      Italian "Non ci posso credere, [laugh] è la notizia più bella della mia vita!"
gen ryan_it_sad_sigh.wav       -s ryan -- sad      Italian "Ho perso tutto quello che avevo, [sigh] e adesso non so più cosa fare."
gen ryan_it_anger_huff.wav     -s ryan -- anger    Italian "Come ti permetti di parlarmi così? [huff] Questo non lo accetto!"
gen ryan_it_disgust_ugh.wav    -s ryan -- disgust  Italian "Ma che roba è questa? [ugh] Fa davvero schifo, non riesco neanche a guardarla."

echo "== ryan (preset) — English, emotion + paralinguistic =="
gen ryan_en_joy_laugh.wav      -s ryan -- joy      English "I can't believe it, [laugh] this is the best news of my whole life!"
gen ryan_en_sad_sigh.wav       -s ryan -- sad      English "I've lost everything I had, [sigh] and now I don't know what to do."

echo "== multilingual highlights (native preset per language, [laugh]/[sigh] vector path) =="
gen de_vivian_anger_huff.wav   -s vivian   -- anger German   "Wie kannst du es wagen, so mit mir zu reden? [huff] Das akzeptiere ich nicht!"
gen fr_vivian_sad_sigh.wav     -s vivian   -- sad   French   "J'ai tout perdu, [sigh] et maintenant je ne sais plus quoi faire."
gen es_vivian_joy_laugh.wav    -s vivian   -- joy   Spanish  "No me lo puedo creer, [laugh] ¡es la mejor noticia de mi vida!"
gen ja_ono_anna_sad_sigh.wav   -s ono_anna -- sad   Japanese "私が持っていたものを全て失って、[sigh] もうどうすればいいのか分からない。"
gen ko_sohee_joy_laugh.wav     -s sohee    -- joy   Korean   "믿을 수가 없어, [laugh] 내 인생 최고의 소식이야!"

if [ -f voices/galatea_graft.qvoice ]; then
  echo "== galatea (25 MB cloned voice) — Italian, --emotion auto-COMBINE + [laugh] (clone emotes + laughs) =="
  GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
  gen galatea_it_joy_laugh.wav $GAL -- joy Italian "Non ci posso credere, [laugh] è la notizia più bella della mia vita!"
else
  echo "== galatea clone SKIPPED (voices/galatea_graft.qvoice not present — bash download_voices.sh) =="
fi

echo ""
echo "Done. Emotion+paralinguistic demo WAVs in:  $OUT/"
find "$OUT" -name '*.wav' | sort | sed 's/^/  /'
echo "Listen (macOS):  for f in $OUT/*.wav; do echo \$f; afplay \$f; done"
