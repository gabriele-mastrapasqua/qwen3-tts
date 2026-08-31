#!/usr/bin/env bash
set -u
VOICE="${1:-ryan}"
MODEL="${2:-qwen3-tts-0.6b}"
OUT="${3:-samples/sound_suite}"
BIN=./qwen_tts
SEED=42
mkdir -p "$OUT"

CANDIDATES=(
  "laugh_hehhh_en|English|-|0|0.87|0.67|Hehhh..."
  "laugh_hehhh_it|Italian|-|0|0.87|0.67|Hehhh..."
  "laugh_ahah_it|Italian|-|0|1.00|0.80|Ahah!"
  "laugh_ahahah_it|Italian|-|0|1.00|0.80|Ahahah!"
  "laugh_eheh_it|Italian|-|0|0.95|0.78|Eheh..."
  "laugh_hihi_it|Italian|-|0|0.95|0.78|Hihi!"
  "laugh_haha_en|English|-|0|1.00|0.80|Haha!"
  "sigh_hah_it|Italian|-|0|0.95|0.67|Hah..."
  "sigh_hooo_it|Italian|-|0|0.90|0.65|Hooo..."
  "sigh_hmpf_it|Italian|-|0|1.00|0.75|Hmpf..."
  "ahh_haaa_it|Italian|-|0|0.87|0.70|Haaa..."
  "gasp_ah_it|Italian|-|0|1.05|0.85|Ah!"
  "gasp_oh_it|Italian|-|0|1.05|0.85|Oh!"
  "wow_uao_it|Italian|-|0|1.00|0.85|Uao..."
  "hmm_hmmm_it|Italian|-|0|0.88|0.65|Hmmm..."
  "mah_it|Italian|-|0|0.95|0.78|Mah..."
  "boh_it|Italian|-|0|0.95|0.78|Boh..."
  "uhm_it|Italian|-|0|0.92|0.72|Uhm..."
  "huff_uff_it|Italian|-|0|0.95|0.82|Uff..."
  "ugh_it|Italian|-|0|0.92|0.80|Ugh..."
  "bleah_it|Italian|-|0|1.00|0.82|Bleah!"
  "tsk_it|Italian|-|0|1.00|0.80|Tsk tsk..."
  "ouch_ahi_it|Italian|-|0|1.00|0.85|Ahi!"
  "ouch_en|English|-|0|1.00|0.85|Ouch!"
  "cry_buaa_it|Italian|-|0|0.90|0.78|Buaa..."
  "sniff_it|Italian|-|0|0.95|0.70|Snif... snif..."
  "emoji_joytears_it|Italian|-|0|1.00|0.85|😂😂"   # -> faint sigh (validated)
  "emoji_lol_it|Italian|-|0|1.00|0.85|lol 😆"
  "emoji_cry_it|Italian|-|0|1.00|0.85|😢"
  "zh_haha|Italian|-|0|1.00|0.85|哈哈"            # -> laugh (validated TOP)
  "zh_heihei|Italian|-|0|1.00|0.85|嘿嘿"          # -> laugh
  "zh_hehe|Italian|-|0|1.00|0.85|呵呵"            # -> laugh
  "zh_ai_sigh|Italian|-|0|1.00|0.80|唉"           # -> weary sigh (validated)
  "zh_en_mmm|Italian|-|0|1.00|0.85|嗯"            # -> soft "mmm" (validated -> [mmm])
)

INDEX="$OUT/INDEX.txt"
: > "$INDEX"
echo "# sound_suite — id, params, and the exact replicable CLI command (vary --text to probe)." >> "$INDEX"
echo "# voice=$VOICE model=$MODEL seed=$SEED" >> "$INDEX"
echo >> "$INDEX"

n=0; ok=0
for row in "${CANDIDATES[@]}"; do
  IFS='|' read -r id lang emo steer rate vol text <<< "$row"
  n=$((n+1))
  args=( -d "$MODEL" -j1 -T 0 --seed "$SEED" -s "$VOICE" -l "$lang"
         --rate "$rate" --volume "$vol" --text "$text" -o "$OUT/$id.wav" )
  emo_str=""
  [ "$emo" != "-" ] && { args+=( --emotion "$emo" --steer-weight "$steer" ); emo_str=" --emotion $emo --steer-weight $steer"; }
  cmd="$BIN -d $MODEL -j1 -T 0 --seed $SEED -s $VOICE -l $lang --rate $rate --volume $vol$emo_str --text \"$text\" -o $OUT/$id.wav"
  if "$BIN" "${args[@]}" >/dev/null 2>&1; then
    ok=$((ok+1)); status="ok "
  else
    status="ERR"
  fi
  printf "[%s] %-22s %-8s rate=%s vol=%s  text=%s\n" "$status" "$id" "$lang" "$rate" "$vol" "\"$text\"" | tee -a "$INDEX"
  echo "      $cmd" >> "$INDEX"
  echo >> "$INDEX"
done

echo
echo "Generated $ok/$n sounds in $OUT/  (index + replicable CLI per sound: $INDEX)"
echo "Listen:  for f in $OUT/*.wav; do echo \"\$f\"; afplay \"\$f\"; done"
echo "Vary a prompt fast: copy the CLI line from $INDEX, change --text, re-run."
echo "Then tell me the winning IDs and I'll bake them as [tag] macros."
