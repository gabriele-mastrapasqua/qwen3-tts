#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen3-tts-0.6b}"
BIN="${BIN:-./qwen_tts}"
OUT_INPUT="${OUT:-}"
KEEP_OUT="${KEEP_OUT:-0}"
if [ -n "$OUT_INPUT" ]; then
    OUT="$OUT_INPUT"
    OWN_OUT=0
else
    OUT=$(mktemp -d /tmp/qwen-stream-layout.XXXXXX)
    OWN_OUT=1
fi

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model directory not present: $MODEL"
    exit 0
fi
[ -x "$BIN" ] || { echo "FAIL: build $BIN first"; exit 1; }

cleanup() {
    if [ "$OWN_OUT" -eq 1 ] && [ "$KEEP_OUT" != 1 ]; then rm -rf "$OUT"; fi
}
trap cleanup EXIT
mkdir -p "$OUT"

texts=(
    "Short boundary test."
    "This medium known text exercises several tokenizer positions before the first generated codec frame."
    "This longer known text exercises the dual track layout over a wider range of text positions so that the trailing text sequence is materially longer than the initial prefill."
)
labels=(short medium long)

echo "=== known-text streaming-layout smoke (model=$MODEL) ==="
for i in "${!texts[@]}"; do
    label="${labels[$i]}"
    control_log="$OUT/${label}.control.log"
    stream_log="$OUT/${label}.stream.log"
    control_wav="$OUT/${label}.control.wav"
    stream_wav="$OUT/${label}.stream.wav"

    env -u QWEN_TTS_STREAM_LAYOUT "$BIN" -d "$MODEL" \
        --speaker ryan --language English --temperature 0 --seed 4242 \
        --max-tokens 4 --text "${texts[$i]}" --output "$control_wav" \
        >"$control_log" 2>&1
    QWEN_TTS_STREAM_LAYOUT=1 "$BIN" -d "$MODEL" \
        --speaker ryan --language English --temperature 0 --seed 4242 \
        --max-tokens 4 --text "${texts[$i]}" --output "$stream_wav" \
        >"$stream_log" 2>&1

    grep -q 'text+eos=' "$control_log"
    grep -q 'stream_common=1' "$stream_log"
    trailing=$(sed -n 's/.*trailing_text=\([0-9][0-9]*\).*/\1/p' "$stream_log" | head -1)
    [ -n "$trailing" ] && [ "$trailing" -gt 0 ]
    test -s "$control_wav"
    test -s "$stream_wav"

    python3 - "$control_wav" "$stream_wav" <<'PY'
import sys
import wave

paths = sys.argv[1:]
with wave.open(paths[0], "rb") as a, wave.open(paths[1], "rb") as b:
    assert a.getframerate() == b.getframerate() == 24000
    assert a.getnchannels() == b.getnchannels() == 1
    assert a.getnframes() == b.getnframes() > 0
print("PASS: equal output frame count")
PY
    printf 'PASS: %-6s control/full-prefill vs stream/common+tail (trailing=%s)\n' "$label" "$trailing"
done

# Clone/ICL requires a caller-provided base model, reference WAV and transcript.
# Keep the default smoke cheap and deterministic; exercise this branch when the
# local checkout has the assets instead of silently pretending it was covered.
if [ -n "${STREAM_LAYOUT_ICL_MODEL:-}" ] && \
   [ -n "${STREAM_LAYOUT_REF_AUDIO:-}" ] && [ -n "${STREAM_LAYOUT_REF_TEXT:-}" ]; then
    icl_log="$OUT/icl.stream.log"
    icl_wav="$OUT/icl.stream.wav"
    QWEN_TTS_STREAM_LAYOUT=1 "$BIN" -d "$STREAM_LAYOUT_ICL_MODEL" \
        --ref-audio "$STREAM_LAYOUT_REF_AUDIO" --language English \
        --temperature 0 --seed 4242 --max-tokens 4 \
        --text "${texts[0]}" --output "$icl_wav" \
        >"$icl_log" 2>&1
    grep -q 'stream_common=' "$icl_log"
    grep -q 'trailing_text=' "$icl_log"
    test -s "$icl_wav"
    echo "PASS: optional ICL/clone layout"
else
    echo "SKIP: ICL/clone layout (set STREAM_LAYOUT_ICL_MODEL, STREAM_LAYOUT_REF_AUDIO, STREAM_LAYOUT_REF_TEXT)"
fi

echo "stream-layout smoke: PASS"
