#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
BIN=./qwen_tts
M=${M:-qwen3-tts-1.7b}
OUT=samples/tests/$(printf '%(%Y-%m-%d)T' -1 2>/dev/null || echo 2026-07-08)_para-demo
mkdir -p "$OUT"

say() { # $1=name $2=lang $3=text  [extra --emotion ...]
  local name="$1" lang="$2" txt="$3"; shift 3
  local out="$OUT/$name.wav"
  echo ""
  echo "# $name  ($lang)  — $txt"
  echo "$BIN -d $M -s ryan -l $lang --text \"$txt\" $* -o $out"
  $BIN -d "$M" -s ryan -l "$lang" --text "$txt" "$@" -o "$out" 2>/dev/null
  echo "  afplay $out"
}

echo "=== make para-demo — shipped inline [tag]s (ryan, 1.7B) ==="

say laugh_it Italian "Non ci posso credere, [laugh] è la cosa più assurda che abbia mai sentito."
say sigh_it  Italian "Che giornata... [sigh] non ce la faccio più."

say wow_en    English "I opened the box and [wow], it was exactly what I asked for."          --emotion surprise
say yawn_en   English "It's almost midnight and [yawn], I can barely stay awake."             --emotion sad
say scoff_en  English "He promised he'd pay me back, [scoff], as if that'll ever happen."     --emotion disgust
say scoff_it  Italian "Mi ha chiesto scusa, [scoff], figurati se ci credo."                   --emotion disgust

say giggle_en English "He tripped over the cat and [giggle], I couldn't stop myself."

echo ""
echo "=== done → $OUT ==="
ls -1 "$OUT"/*.wav
