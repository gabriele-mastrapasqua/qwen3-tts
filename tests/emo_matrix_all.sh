#!/usr/bin/env bash
# ============================================================================================
# emo_matrix_all.sh — FULL re-verification matrix: every language × every emotion, driven by the
# C engine's `--emotion` flag (THE recipe, docs/emotion-THE-recipe.md). Recommended GOLD voice per
# language (ryan = IT/EN/PT-Romance; vivian = DE/FR/ES/ZH; ono_anna = JA; sohee = KO). One neutral
# carrier per language → the emotion comes purely from the recipe (instruct/steer/expr the router applies).
# Output: samples/tests/2026-06-29_emo-matrix-all/  (gitignored). 1.7B CustomVoice, seed 42.
# ============================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42
OUT=samples/tests/2026-06-29_emo-matrix-all; mkdir -p "$OUT"
[ -d "$M" ] || { echo "SKIP: $M not present"; exit 0; }
EMOS="sad joy anger fear disgust surprise"

# tag | language | voiceflags | carrier text
LANGS=(
"it|Italian|-s ryan|Allora, lascia che ti spieghi come stanno le cose."
"en|English|-s ryan|So, let me explain to you how things really are."
"de|German|-s vivian|Also, lass mich dir in Ruhe erklaeren, wie die Dinge wirklich stehen."
"fr|French|-s vivian|Bon, laisse-moi t'expliquer calmement comment les choses se passent vraiment."
"es|Spanish|-s vivian|Bueno, dejame explicarte con calma como estan realmente las cosas."
"zh|Chinese|-s vivian|那么，让我来跟你解释一下事情到底是怎么回事。"
"ja|Japanese|-s ono_anna|では、物事が実際にどうなっているのか説明させてください。"
"ko|Korean|-s sohee|그럼, 상황이 실제로 어떤지 설명해 드릴게요."
"ru|Russian|-s vivian|Итак, позвольте мне объяснить, как обстоят дела на самом деле."
"pt|Portuguese|-s ryan|Bem, deixa-me explicar-te com calma como as coisas realmente são."
)

for row in "${LANGS[@]}"; do
  IFS='|' read -r tag lang vf txt <<< "$row"
  echo "== $lang ($vf) =="
  for e in $EMOS; do
    out="$OUT/${tag}_${e}.wav"
    if $BIN -d $M $vf -l "$lang" --emotion "$e" --seed $SEED --text "$txt" -o "$out" >"$OUT/${tag}_${e}.log" 2>&1; then
      m=$(grep -hoE "mode=[A-Z]+|expr-carried" "$OUT/${tag}_${e}.log" | head -1)
      printf '  %-12s %s\n' "${tag}_${e}" "$m"
    else echo "  FAIL ${tag}_${e}"; tail -2 "$OUT/${tag}_${e}.log"; fi
  done
done

# ---- A/B: DE/FR/ES are EXPR-only in the recipe (never tested WITH steer). Add a COMBINE (+steer) variant
#      so we can hear if steer improves them (user hypothesis: steer almost always helps). Manual flags. ----
echo "== +STEER A/B (DE/FR/ES COMBINE: native/IT expr + ryan_<emo> steer) =="
declare -A IN=(
 [sad]="Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears."
 [joy]="Speak with bright, radiant joy, light and warm, smiling through every word."
 [anger]="Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage."
 [fear]="Speak in a frightened, trembling, anxious tone, voice shaky and breathless with dread."
 [disgust]="Speak with deep disgust and revulsion, lip-curling contempt, as if something repels you."
 [surprise]="Speak with sudden astonishment and surprise, gasping and caught off guard.")
abtok(){ case "$1" in anger)echo ang;; *)echo "$1";; esac; }
ABLANGS=(
"de|German|-s vivian|presets/expr/german_csp_k6.expr|Also, lass mich dir in Ruhe erklaeren, wie die Dinge wirklich stehen."
"fr|French|-s vivian|presets/expr/french_csp_k6.expr|Bon, laisse-moi t'expliquer calmement comment les choses se passent vraiment."
"es|Spanish|-s vivian|presets/expr/italian_csp_topk6.expr|Bueno, dejame explicarte con calma como estan realmente las cosas."
)
for row in "${ABLANGS[@]}"; do
  IFS='|' read -r tag lang vf ex txt <<< "$row"
  for e in $EMOS; do
    w=8; case "$e" in fear|surprise) w=4;; esac
    out="$OUT/${tag}_${e}_steer.wav"
    if $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --expr "$ex" --expr-weight 1.2 \
         --ml-steer "presets/steer/emotion/ryan_$(abtok $e).qlsteer" --ml-weight $w --ml-range 21-25 \
         --instruct "${IN[$e]}" --text "$txt" -o "$out" >"$OUT/${tag}_${e}_steer.log" 2>&1; then
      printf '  %-16s COMBINE(+steer w%s)\n' "${tag}_${e}_steer" "$w"
    else echo "  FAIL ${tag}_${e}_steer"; fi
  done
done
echo ""; echo "Done -> $OUT  ($(ls "$OUT"/*.wav 2>/dev/null | wc -l | tr -d ' ') clips)"
echo "Listen: for f in $OUT/*.wav; do echo \$f; afplay \$f; done"
