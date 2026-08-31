#!/bin/bash
set -u
cd "$(dirname "$0")/.."

MODEL=${MODEL:-qwen3-tts-0.6b}
PORT=${PORT:-8091}
MIN_CORR=${MIN_CORR:-0.98}
TEXT="Parallel synthesis correctness test, one two three."
SEED=42
TMP=$(mktemp -d)
RC=0

sweep() { pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 1; }
trap 'sweep; rm -rf "$TMP"' EXIT

start_server() { # $1=workers  $2..=extra flags
    local workers=$1; shift
    sweep
    ./qwen_tts -d "$MODEL" --serve "$PORT" --workers "$workers" "$@" >"$TMP/srv.log" 2>&1 &
    for i in $(seq 1 50); do
        if timeout 3 curl -s "http://localhost:$PORT/v1/health" >/dev/null 2>&1; then return 0; fi
        sleep 0.5
    done
    echo "  !! server failed to start"; cat "$TMP/srv.log" | tail -5; return 1
}

req() { # $1=outfile  -> echoes http_code
    timeout 120 curl -s -o "$1" -w "%{http_code}" "http://localhost:$PORT/v1/tts" \
        -d "{\"text\":\"$TEXT\",\"speaker\":\"ryan\",\"language\":\"English\",\"seed\":$SEED,\"temperature\":0}"
}

valid_wav() { [ -s "$1" ] && [ "$(head -c4 "$1")" = "RIFF" ]; }

test_config() { # $1=label  $2..=server flags
    local label=$1; shift
    echo "── config: $label ─────────────────────────────"

    start_server 1 "$@" || { RC=1; return; }
    local rc_ref; rc_ref=$(req "$TMP/ref.wav")
    sweep
    if [ "$rc_ref" != "200" ] || ! valid_wav "$TMP/ref.wav"; then
        echo "  FAIL: reference request http=$rc_ref"; RC=1; return
    fi
    echo "  ref: http=200 bytes=$(wc -c <"$TMP/ref.wav")"

    start_server 2 "$@" || { RC=1; return; }
    local ca cb
    ( req "$TMP/a.wav" >"$TMP/ca" ) &
    local pa=$!
    ( req "$TMP/b.wav" >"$TMP/cb" ) &
    local pb=$!
    wait "$pa" "$pb"      # local curl pids only — server is killed by name regardless
    ca=$(cat "$TMP/ca"); cb=$(cat "$TMP/cb")
    sweep

    if [ "$ca" != "200" ] || [ "$cb" != "200" ]; then
        echo "  FAIL: concurrent http a=$ca b=$cb"; cat "$TMP/srv.log" | tail -8; RC=1; return
    fi
    if ! valid_wav "$TMP/a.wav" || ! valid_wav "$TMP/b.wav"; then
        echo "  FAIL: concurrent output not valid WAV"; RC=1; return
    fi
    echo "  concurrent: a=200 ($(wc -c <"$TMP/a.wav")B)  b=200 ($(wc -c <"$TMP/b.wav")B)"

    for f in a b; do
        if python3 tests/compare_audio.py "$TMP/ref.wav" "$TMP/$f.wav" --min-corr "$MIN_CORR" >"$TMP/cmp" 2>&1; then
            echo "  $f vs ref: PASS ($(grep -o 'corr=[0-9.]*' "$TMP/cmp" | head -1))"
        else
            echo "  $f vs ref: FAIL"; cat "$TMP/cmp" | tail -3; RC=1
        fi
    done
}

echo "═══ Concurrent worker-pool test (model=$MODEL, port=$PORT) ═══"
test_config "bf16 (no quant)"
test_config "int8"           --int8
test_config "int4"           --int4
if [ -f voices/silvio_06b.qvoice ]; then
    test_config "voice silvio_06b + int8" --load-voice voices/silvio_06b.qvoice --int8
else
    echo "── skip voice config (voices/silvio_06b.qvoice missing) ──"
fi

echo ""
echo "═══ procs after (must be empty) ═══"; pgrep -fl qwen_tts || echo "  OK none"
echo ""
if [ "$RC" = "0" ]; then echo "✅ PARALLEL TEST PASS"; else echo "❌ PARALLEL TEST FAIL"; fi
exit $RC
