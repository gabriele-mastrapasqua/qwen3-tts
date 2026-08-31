#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
DEST=voices
BASE_URL="${BASE_URL:-https://huggingface.co/gabrione/qwen3-tts-voices/resolve/main}"
mkdir -p "$DEST"

read -r -d '' VOICES <<'EOF'
galatea_graft.qvoice  0ba4be1fab09b511b19fb2ff765f5737d2ead963964b9c8d9d605916ffd42994
quijote_graft.qvoice  5f15d4c8964b77eadc0b7232756bdc8bb97759a2b3655cd6bdc69ac780a714c9
ohenry_graft.qvoice   c39e14a1b33f822fc5481e89c0e7b51b80f5c6ff715fcf50a7bb8dd5343f2861
hugo_graft.qvoice     5d2f352c578e42823fa62b5f2043338f7b0c07921aca73b43c4928f7eb7000cc
EOF

VERIFY_ONLY=0; [ "${1:-}" = "--verify" ] && VERIFY_ONLY=1
sha(){ shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }
echo "$VOICES" | while read -r name want; do
  [ -z "$name" ] && continue
  f="$DEST/$name"
  if [ -f "$f" ]; then
    got=$(sha "$f")
    if [ "$got" = "$want" ]; then echo "OK    $name"; else echo "BAD   $name (sha mismatch)"; [ $VERIFY_ONLY = 1 ] || rm -f "$f"; fi
  fi
  [ $VERIFY_ONLY = 1 ] && continue
  if [ ! -f "$f" ]; then
    echo "GET   $name  <- $BASE_URL/$name"
    if curl -fL --retry 3 -o "$f.part" "$BASE_URL/$name"; then
      mv "$f.part" "$f"; got=$(sha "$f"); [ "$got" = "$want" ] && echo "  verified" || echo "  WARNING sha mismatch after download"
    else echo "  FAILED (set BASE_URL; see https://huggingface.co/gabrione/qwen3-tts-voices)"; rm -f "$f.part"; fi
  fi
done
echo "done. CC0/PD reference voices in voices/ — load with: --load-voice voices/<name>.qvoice --icl-only"
