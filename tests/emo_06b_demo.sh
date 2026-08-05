#!/bin/bash
# The 0.6B showcase: emotion + paralinguistics + cloning, all on the SMALL model, sub-realtime.
#
# For a long time the 0.6B was "the fast neutral TTS" — --emotion did nothing on it. That changed
# 2026-08-05: on the small model the emotion rides on the VOICE (docs/emotion-06b-recipe.md), so
# everything the 1.7B does expressively is now available here too, at RTF < 1.
set -u
cd "$(dirname "$0")/.."
D="${OUT:-samples/tests/emo_06b_demo}"
M="${MODEL:-qwen3-tts-0.6b}"
V="${VOICE:-ryan}"
L="${TTS_LANG:-Italian}"
Q="${QUANT:---int8}"
mkdir -p "$D"

if [ ! -f "presets/emovoice/${V}_ang.bin" ] && [ ! -f "presets/emovoice/${V}_ang.qvoice" ]; then
  echo "No emotional voice assets for '$V' yet. Build them once (~3 min):"
  echo "    make emovoice VOICE=$V"
  exit 1
fi

say() { printf "\n\033[1m%s\033[0m\n" "$*"; }
run() { # run <label> <outfile> <args...>
  local lbl="$1" out="$2"; shift 2
  printf "  %-34s" "$lbl"
  local rtf
  rtf=$(./qwen_tts -d "$M" -l "$L" $Q "$@" -o "$D/$out" 2>&1 | grep -oE "RTF [0-9.]+" | tail -1)
  echo "${rtf:-?}   $D/$out"
}

echo "=== 0.6B expressivity demo — model=$M voice=$V lang=$L quant=${Q:-bf16} ==="

say "1. the six emotions (--emotion, resolved to a 4 KB voice asset)"
for e in sad joy anger fear disgust surprise; do
  run "--emotion $e" "01_${e}.wav" -s "$V" --seed 42 --emotion "$e" \
    --text "Non è possibile che succeda sempre la stessa cosa, ogni singola volta."
done
run "(neutral reference)" "01_neutral.wav" -s "$V" --seed 42 \
  --text "Non è possibile che succeda sempre la stessa cosa, ogni singola volta."

say "2. paralinguistics — inline [tag], 0.6B-tuned seeds"
for t in sigh laugh yawn wow scoff; do
  run "[$t]" "02_${t}.wav" -s "$V" --text "[$t] Non è possibile che succeda sempre la stessa cosa."
done

say "3. both at once — emotion + paralinguistic tag, one generation"
run "sad + [sigh]"   "03_sad_sigh.wav"   -s "$V" --emotion sad   --text "[sigh] Non è possibile che succeda sempre la stessa cosa."
run "joy + [laugh]"  "03_joy_laugh.wav"  -s "$V" --emotion joy   --text "[laugh] Non è possibile che succeda sempre la stessa cosa."

say "4. a cloned voice, if one is available"
if [ -f voices/galatea_06b_graft.qvoice ]; then
  run "clone (neutral)" "04_clone.wav" --load-voice voices/galatea_06b_graft.qvoice --icl-only \
    --seed 42 --text "Non è possibile che succeda sempre la stessa cosa, ogni singola volta."
  run "clone + [sigh]"  "04_clone_sigh.wav" --load-voice voices/galatea_06b_graft.qvoice --icl-only \
    --text "[sigh] Non è possibile che succeda sempre la stessa cosa."
else
  echo "  (skipped — no voices/galatea_06b_graft.qvoice; any 0.6B .qvoice/.bin works)"
fi

echo
echo "Done -> $D/   ($(ls "$D" | wc -l | tr -d ' ') files).  Listen:  afplay $D/01_anger.wav"
echo "Everything above ran on the 0.6B. The 1.7B is only used ONCE, offline, to build the voice assets."
