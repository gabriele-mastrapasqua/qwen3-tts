#!/usr/bin/env bash
# sound_suite.sh — paralinguistic SOUND-DISCOVERY suite.
#
# Mass-generates a labeled grid of onomatopoeia/vowel/emoji candidates so we can map
# which strings -> which sounds (laughs, cries, gasps, sighs, ...) each voice/language
# can produce. Workflow: run it -> listen -> tell me the winning IDs -> I bake them as
# reusable [tag] macros.
#
# Usage:
#   tests/sound_suite.sh [voice] [model_dir] [outdir]
#   ./qwen_tts must be built (make blas).  Default voice=ryan, model=qwen3-tts-0.6b.
#
# Add candidates by appending "ID|LANG|EMOTION|STEER|RATE|VOL|TEXT" rows to CANDIDATES
# below. EMOTION="-" means no steering (pure soft prosody — best for fillers).
set -u
VOICE="${1:-ryan}"
MODEL="${2:-qwen3-tts-0.6b}"
OUT="${3:-samples/sound_suite}"
BIN=./qwen_tts
SEED=42
mkdir -p "$OUT"

# id            lang     emo  steer rate  vol   text
CANDIDATES=(
  # --- laughs (EN tends to actually laugh; IT often a breathy sigh) ---
  "laugh_hehhh_en|English|-|0|0.87|0.67|Hehhh..."
  "laugh_hehhh_it|Italian|-|0|0.87|0.67|Hehhh..."
  "laugh_ahah_it|Italian|-|0|1.00|0.80|Ahah!"
  "laugh_ahahah_it|Italian|-|0|1.00|0.80|Ahahah!"
  "laugh_eheh_it|Italian|-|0|0.95|0.78|Eheh..."
  "laugh_hihi_it|Italian|-|0|0.95|0.78|Hihi!"
  "laugh_haha_en|English|-|0|1.00|0.80|Haha!"
  # --- sighs / exhale ---
  "sigh_hah_it|Italian|-|0|0.95|0.67|Hah..."
  "sigh_hooo_it|Italian|-|0|0.90|0.65|Hooo..."
  "sigh_hmpf_it|Italian|-|0|1.00|0.75|Hmpf..."
  # --- relief / surprise / gasp ---
  "ahh_haaa_it|Italian|-|0|0.87|0.70|Haaa..."
  "gasp_ah_it|Italian|-|0|1.05|0.85|Ah!"
  "gasp_oh_it|Italian|-|0|1.05|0.85|Oh!"
  "wow_uao_it|Italian|-|0|1.00|0.85|Uao..."
  # --- thinking / hesitation ---
  "hmm_hmmm_it|Italian|-|0|0.88|0.65|Hmmm..."
  "mah_it|Italian|-|0|0.95|0.78|Mah..."
  "boh_it|Italian|-|0|0.95|0.78|Boh..."
  "uhm_it|Italian|-|0|0.92|0.72|Uhm..."
  # --- irritation / disgust ---
  "huff_uff_it|Italian|-|0|0.95|0.82|Uff..."
  "ugh_it|Italian|-|0|0.92|0.80|Ugh..."
  "bleah_it|Italian|-|0|1.00|0.82|Bleah!"
  "tsk_it|Italian|-|0|1.00|0.80|Tsk tsk..."
  # --- pain ---
  "ouch_ahi_it|Italian|-|0|1.00|0.85|Ahi!"
  "ouch_en|English|-|0|1.00|0.85|Ouch!"
  # --- cry ---
  "cry_buaa_it|Italian|-|0|0.90|0.78|Buaa..."
  "sniff_it|Italian|-|0|0.95|0.70|Snif... snif..."
  # --- emoji probes (does the tokenizer turn these into anything?) ---
  "emoji_joytears_it|Italian|-|0|1.00|0.85|😂😂"
  "emoji_lol_it|Italian|-|0|1.00|0.85|lol 😆"
  "emoji_cry_it|Italian|-|0|1.00|0.85|😢"
)

INDEX="$OUT/INDEX.txt"
: > "$INDEX"
printf "%-22s %-8s %-6s %-5s %-5s  %s\n" "ID" "LANG" "STEER" "RATE" "VOL" "TEXT" | tee "$INDEX"
printf '%.0s-' {1..78}; echo

n=0; ok=0
for row in "${CANDIDATES[@]}"; do
  IFS='|' read -r id lang emo steer rate vol text <<< "$row"
  n=$((n+1))
  args=( -d "$MODEL" -j1 -T 0 --seed "$SEED" -s "$VOICE" -l "$lang"
         --rate "$rate" --volume "$vol" --text "$text" -o "$OUT/$id.wav" )
  [ "$emo" != "-" ] && args+=( --emotion "$emo" --steer-weight "$steer" )
  if "$BIN" "${args[@]}" >/dev/null 2>&1; then
    ok=$((ok+1))
    line=$(printf "%-22s %-8s %-6s %-5s %-5s  %s" "$id" "$lang" "$steer" "$rate" "$vol" "$text")
    echo "$line" | tee -a "$INDEX"
  else
    echo "FAILED: $id" | tee -a "$INDEX"
  fi
done

echo
echo "Generated $ok/$n sounds in $OUT/  (index: $INDEX)"
echo "Listen:  for f in $OUT/*.wav; do echo \"\$f\"; afplay \"\$f\"; done"
echo "Then tell me the winning IDs and I'll bake them as [tag] macros."
