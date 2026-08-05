#!/bin/bash
# Build the 6 emotional VOICE assets that give the 0.6B its --emotion flag.
#
# On the small model emotion is not an inference-time lever, it is a property of the voice
# (docs/emotion-06b-recipe.md). This builds one asset per emotion for a given voice:
#   1.7B generates ~25 s of that voice, emotional  ->  0.6B Base extracts the 4 KB x-vector
#
# Usage:
#   make emovoice VOICE=ryan                                  # a preset
#   make emovoice VOICE=galatea LOAD=voices/galatea_graft.qvoice   # a cloned voice
#   make emovoice VOICE=ryan TTS_LANG=English GRAFT=voices/galatea_06b_graft.qvoice
#
# GRAFT=<0.6B graft .qvoice> additionally emits a 16.8 MB graft per emotion (x-vector swapped in).
# Recommended for high-arousal emotions: with a bare 4 KB x-vector, anger compresses the delivery
# and can swallow a short word — the graft's TPAD+WOVR hold the sentence together (ear 2026-08-05).
set -u
cd "$(dirname "$0")/.."

VOICE="${VOICE:-ryan}"
LANG_="${TTS_LANG:-Italian}"
LOAD="${LOAD:-}"
GRAFT="${GRAFT:-}"
BIG="${BIG:-qwen3-tts-1.7b}"
BASE="${BASE:-qwen3-tts-0.6b-base}"
OUT="${OUT:-presets/emovoice}"
SEED="${SEED:-42}"
DONORS="${DONORS:-samples/tests/emovoice_donors/$VOICE}"

# ~25 s of carrier text: enough for a stable ECAPA embedding.
TEXT="Ma ti rendi conto di quello che è successo? Te l'avevo detto mille volte, mille volte, e alla fine è andata proprio così. Adesso non venirmi a dire che non lo sapevi, perché lo sapevi benissimo. È sempre la stessa storia, ogni volta la stessa identica storia, e io continuo a ripetere le cose a vuoto."
[ "$LANG_" = "English" ] && TEXT="Do you realise what just happened? I told you a thousand times, a thousand times, and in the end it went exactly like that. Now do not tell me you did not know, because you knew perfectly well. It is always the same story, every single time the same story, and I keep repeating myself for nothing."

if [ ! -d "$BASE" ]; then
  echo "Error: need the 0.6B Base model at '$BASE' (it carries the ECAPA speaker encoder)."
  echo "  ./download_model.sh --model 0.6b-base"
  exit 1
fi
mkdir -p "$OUT" "$DONORS"

if [ -n "$LOAD" ]; then VOICE_ARGS=(--load-voice "$LOAD" --icl-only); else VOICE_ARGS=(-s "$VOICE"); fi

echo "Building emotional voice assets for '$VOICE' ($LANG_) -> $OUT/"
echo

for spec in sad:sad joy:joy anger:ang fear:fear disgust:disgust surprise:surprise; do
  emo="${spec%%:*}"; tok="${spec##*:}"
  donor="$DONORS/${VOICE}_${tok}.wav"

  if [ -f "$donor" ]; then
    echo "  [$tok] donor cached"
  else
    printf "  [%s] donor from the 1.7B ... " "$tok"
    ./qwen_tts -d "$BIG" "${VOICE_ARGS[@]}" -l "$LANG_" --seed "$SEED" --emotion "$emo" \
      --text "$TEXT" -o "$donor" --silent >/dev/null 2>&1
    [ -f "$donor" ] && echo "ok" || { echo "FAILED"; continue; }
  fi

  printf "  [%s] extracting the 4 KB voice ... " "$tok"
  ./qwen_tts -d "$BASE" --ref-audio "$donor" --save-voice "$OUT/${VOICE}_${tok}.bin" \
    --silent >/dev/null 2>&1
  if [ -f "$OUT/${VOICE}_${tok}.bin" ]; then echo "$OUT/${VOICE}_${tok}.bin"; else echo "FAILED"; continue; fi

  if [ -n "$GRAFT" ] && [ -f "$GRAFT" ]; then
    python3 tests/graft_set_xvector.py "$GRAFT" "$OUT/${VOICE}_${tok}.bin" \
      "$OUT/${VOICE}_${tok}.qvoice" >/dev/null 2>&1 \
      && echo "        + graft $OUT/${VOICE}_${tok}.qvoice (16.8 MB, preferred for anger)"
  fi
done

echo
echo "Done. Use them on the SMALL model — the 1.7B is no longer in the path:"
echo "  ./qwen_tts -d qwen3-tts-0.6b -s $VOICE -l $LANG_ --int8 --emotion anger --text \"...\""
echo "Donor audio kept in $DONORS/ (delete it to force a rebuild)."
