#!/usr/bin/env bash
# ============================================================================================
# emotion_demo.sh — render the VALIDATED --emotion recipe for new users to listen to.
#
# Uses the WIN texts from the ear-validated scripts (recipe_final.sh = Italian; crosslang_emo.sh =
# RU/ZH/JA/KO; german/french/spanish_ab.sh). The engine's `--emotion` auto-router applies the
# ear-validated per-(voice×emotion) recipe itself (expr / steer / instruct / temperature) — this
# script only supplies voice + language + text, so the win can't drift.
#
#   ryan (preset)   — Italian, ALL 6 emotions (the primary showcase)
#   ryan (preset)   — multilingual highlights (emotion works in every Qwen language)
#   galatea (clone) — Italian, a few emotions (the 25 MB cloned voice emotes too)
#
# Output dir: $EMO_DEMO_DIR (default samples/emotion_demo). Needs the 1.7B model + the .expr packs
# (`bash download_assets.sh` — STEER-only cells still render without them). 1.7B CustomVoice only.
# ============================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42
OUT="${EMO_DEMO_DIR:-samples/tests/emotion_demo}"; mkdir -p "$OUT"   # samples/tests/ is gitignored — never write loose in samples/
[ -d "$M" ] || { echo "SKIP: $M not present — get it with ./download_model.sh"; exit 0; }
[ -f presets/expr/italian_csp_topk6.expr ] || \
  echo "NOTE: emotion .expr packs missing — run: bash download_assets.sh (EXPR/COMBINE cells need them; STEER cells still render)"

# gen <out.wav>  <voiceflags...>  --  <emotion> <language> <text>
gen(){
  local out="$1"; shift
  local vf=(); while [ "$1" != "--" ]; do vf+=("$1"); shift; done; shift
  local emo="$1" lang="$2" txt="$3"
  printf '  %-24s ' "$out"
  if $BIN -d "$M" --seed $SEED "${vf[@]}" -l "$lang" --emotion "$emo" --text "$txt" \
        -o "$OUT/$out" 2>"$OUT/${out%.wav}.log"; then
    grep -hoE "mode=[A-Z]+|expr-carried" "$OUT/${out%.wav}.log" | head -1
  else echo "FAIL"; tail -3 "$OUT/${out%.wav}.log"; fi
}

echo "== ryan (preset) — Italian, all 6 emotions (recipe_final win texts) =="
gen ryan_it_sad.wav      -s ryan -- sad      Italian "Ho perso tutto quello che avevo, e adesso non so più cosa fare."
gen ryan_it_joy.wav      -s ryan -- joy      Italian "Non ci posso credere, è la notizia più bella della mia vita, sono felicissimo!"
gen ryan_it_anger.wav    -s ryan -- anger    Italian "Come ti permetti di parlarmi così? Questo non lo accetto, è inaccettabile!"
gen ryan_it_fear.wav     -s ryan -- fear     Italian "C'è qualcuno in casa, ho sentito dei passi... ho paura, non so cosa fare."
gen ryan_it_disgust.wav  -s ryan -- disgust  Italian "Ma che roba è questa? Fa davvero schifo, non riesco nemmeno a guardarla."
gen ryan_it_surprise.wav -s ryan -- surprise Italian "Cosa?! Non me lo aspettavo per niente, è incredibile, sono sbalordito!"

echo "== multilingual highlights (emotion works in every Qwen language, on the NATIVE preset per language) =="
gen de_vivian_anger.wav   -s vivian   -- anger German   "Also, lass mich dir in Ruhe erklaeren, wie die Dinge wirklich stehen."
gen fr_vivian_sad.wav     -s vivian   -- sad   French   "Bon, laisse-moi t'expliquer calmement comment les choses se passent vraiment."
gen es_vivian_joy.wav     -s vivian   -- joy   Spanish  "Bueno, dejame explicarte con calma como estan realmente las cosas."
gen ru_vivian_anger.wav   -s vivian   -- anger Russian  "Как ты смеешь так со мной разговаривать? Это неприемлемо, я этого не потерплю!"
gen zh_vivian_joy.wav     -s vivian   -- joy   Chinese  "我简直不敢相信，这是我一生中最好的消息，我太高兴了！"
gen ja_ono_anna_sad.wav   -s ono_anna -- sad   Japanese "私が持っていたものを全て失って、もうどうすればいいのか分からない。"
gen ko_sohee_anger.wav    -s sohee    -- anger Korean   "네가 어떻게 나한테 그렇게 말할 수 있어? 이건 절대 받아들일 수 없어!"

if [ -f voices/galatea_graft.qvoice ]; then
  echo "== galatea (25 MB cloned voice) — Italian, the clone emotes too =="
  GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
  gen galatea_it_anger.wav $GAL -- anger Italian "Come ti permetti di parlarmi così? Questo non lo accetto, è inaccettabile!"
  gen galatea_it_sad.wav   $GAL -- sad   Italian "Ho perso tutto quello che avevo, e adesso non so più cosa fare."
  gen galatea_it_joy.wav   $GAL -- joy   Italian "Non ci posso credere, è la notizia più bella della mia vita, sono felicissimo!"
else
  echo "== galatea clone: voices/galatea_graft.qvoice not present — SKIP =="
fi

echo ""
echo "Done. Emotion demo WAVs in:  $OUT/"
ls -1 "$OUT"/*.wav 2>/dev/null
echo "Listen (macOS):  for f in $OUT/*.wav; do echo \$f; afplay \$f; done"
