#!/bin/bash
# bench.sh — RTF benchmark suite for qwen-tts
# Usage: ./bench.sh [--level basic|full] [--model-dir <path>] [--seed N]
#
# Levels:
#   basic  — short + long text, normal + streaming (default)
#   full   — basic + server, voice clone (.qvoice), instruct, INT8
#
# Automatically skips tests when models/files are not present.

set -e

BINARY=./qwen_tts
BENCH_DIR=/tmp/qwen_tts_bench
SEED=42
LEVEL=basic
SPEAKER=ryan

# Short text (~2-4s audio)
TEXT_SHORT="Hello, this is a short benchmark test."
TEXT_SHORT_IT="Ciao, questo è un test breve di benchmark."

# Long text (~10-20s audio)
TEXT_LONG="The quick brown fox jumps over the lazy dog on a beautiful sunny afternoon. \
The birds are singing in the trees, and a gentle breeze carries the scent of wildflowers \
across the meadow. It is truly a perfect day for a long walk through the countryside, \
enjoying nature at its finest."
TEXT_LONG_IT="La volpe marrone e veloce salta sopra il cane pigro in un bellissimo pomeriggio \
di sole. Gli uccelli cantano sugli alberi e una leggera brezza porta il profumo dei fiori \
di campo attraverso il prato. È davvero una giornata perfetta per una lunga passeggiata \
in campagna, godendosi la natura nella sua forma più bella."

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)  LEVEL="$2"; shift 2 ;;
        --seed)   SEED="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$BENCH_DIR"

# ── Helpers ──

model_exists() {
    [ -d "$1" ] && [ -f "$1/config.json" ]
}

qvoice_find() {
    # Find first .qvoice file for a given model size pattern (e.g., "06b", "17b")
    local pattern="$1"
    local found=""
    for f in *.qvoice voices/*.qvoice samples/*.qvoice; do
        [ -f "$f" ] || continue
        if echo "$f" | grep -qi "$pattern"; then
            found="$f"
            break
        fi
    done
    # If no pattern match, try any .qvoice
    if [ -z "$found" ]; then
        for f in *.qvoice voices/*.qvoice samples/*.qvoice; do
            [ -f "$f" ] && found="$f" && break
        done
    fi
    echo "$found"
}

extract_rtf() {
    # Extract RTF from log output
    grep -oE 'RTF [0-9]+\.[0-9]+' "$1" 2>/dev/null | head -1 | awk '{print $2}'
}

extract_audio_dur() {
    grep -oE '[0-9]+\.[0-9]+s generated' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+'
}

extract_wall() {
    grep -oE 'in [0-9]+\.[0-9]+s \(RTF' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1
}

PASS=0
SKIP=0
RESULTS=""

run_bench() {
    local label="$1"
    local logfile="$2"
    shift 2

    printf "  %-45s" "$label"

    # Run without --silent so RTF summary is printed to stderr
    if "$BINARY" "$@" 2>&1 | tee "$logfile" >/dev/null; then
        local rtf=$(extract_rtf "$logfile")
        local dur=$(extract_audio_dur "$logfile")
        local wall=$(extract_wall "$logfile")
        if [ -n "$rtf" ]; then
            printf "RTF %-6s  audio %-5ss  wall %-5ss\n" "$rtf" "$dur" "$wall"
            RESULTS="${RESULTS}${label}|${rtf}|${dur}|${wall}\n"
            PASS=$((PASS + 1))
        else
            echo "WARN: no RTF in output"
            PASS=$((PASS + 1))
        fi
    else
        echo "FAIL (exit $?)"
    fi
}

skip_bench() {
    local label="$1"
    local reason="$2"
    printf "  %-45s SKIP (%s)\n" "$label" "$reason"
    SKIP=$((SKIP + 1))
}

# ── Header ──

echo "============================================"
echo "  qwen-tts RTF Benchmark (level=$LEVEL)"
echo "  seed=$SEED  speaker=$SPEAKER"
echo "============================================"
echo ""

# ── 0.6B Model ──

MODEL_06B=qwen3-tts-0.6b
if model_exists "$MODEL_06B"; then
    echo "── 0.6B ($MODEL_06B) ──"

    # Basic: short + long, normal
    run_bench "0.6B short normal" "$BENCH_DIR/06b_short.log" \
        -d "$MODEL_06B" --text "$TEXT_SHORT" -s "$SPEAKER" -l English --seed "$SEED" -o "$BENCH_DIR/06b_short.wav"
    run_bench "0.6B long normal" "$BENCH_DIR/06b_long.log" \
        -d "$MODEL_06B" --text "$TEXT_LONG" -s "$SPEAKER" -l English --seed "$SEED" -o "$BENCH_DIR/06b_long.wav"
    # Basic: streaming
    run_bench "0.6B short stream" "$BENCH_DIR/06b_short_stream.log" \
        -d "$MODEL_06B" --text "$TEXT_SHORT" -s "$SPEAKER" -l English --seed "$SEED" --stream -o "$BENCH_DIR/06b_short_stream.wav"
    run_bench "0.6B long stream" "$BENCH_DIR/06b_long_stream.log" \
        -d "$MODEL_06B" --text "$TEXT_LONG" -s "$SPEAKER" -l English --seed "$SEED" --stream -o "$BENCH_DIR/06b_long_stream.wav"
    # Italian
    run_bench "0.6B short Italian" "$BENCH_DIR/06b_short_it.log" \
        -d "$MODEL_06B" --text "$TEXT_SHORT_IT" -s "$SPEAKER" -l Italian --seed "$SEED" -o "$BENCH_DIR/06b_short_it.wav"
    if [ "$LEVEL" = "full" ]; then
        # .qvoice
        QVOICE_06B=$(qvoice_find "06b")
        if [ -n "$QVOICE_06B" ]; then
            run_bench "0.6B short qvoice" "$BENCH_DIR/06b_qvoice_short.log" \
                -d "$MODEL_06B" --load-voice "$QVOICE_06B" --text "$TEXT_SHORT" --seed "$SEED" -o "$BENCH_DIR/06b_qvoice_short.wav"
            run_bench "0.6B long qvoice" "$BENCH_DIR/06b_qvoice_long.log" \
                -d "$MODEL_06B" --load-voice "$QVOICE_06B" --text "$TEXT_LONG" --seed "$SEED" -o "$BENCH_DIR/06b_qvoice_long.wav"        else
            skip_bench "0.6B short qvoice" "no .qvoice file"
            skip_bench "0.6B long qvoice" "no .qvoice file"
        fi

        # Server bench (cold + warm)
        echo ""
        echo "  ── 0.6B server bench ──"
        "$BINARY" -d "$MODEL_06B" --serve 8099 --silent &>/dev/null & SERVER_PID=$!
        sleep 3

        if curl -s http://localhost:8099/v1/health | grep -q '"ok"' 2>/dev/null; then
            # Cold
            printf "  %-45s" "0.6B server cold"
            T=$(curl -s -w "%{time_total}" -X POST http://localhost:8099/v1/tts \
                -H "Content-Type: application/json" \
                -d "{\"text\":\"$TEXT_SHORT\",\"speaker\":\"$SPEAKER\",\"language\":\"English\",\"seed\":$SEED}" \
                -o "$BENCH_DIR/06b_server_cold.wav")
            DUR=$(soxi -D "$BENCH_DIR/06b_server_cold.wav" 2>/dev/null || echo "?")
            printf "wall %-6ss  audio %-5ss\n" "$T" "$DUR"
            PASS=$((PASS + 1))

            # Warm
            printf "  %-45s" "0.6B server warm"
            T=$(curl -s -w "%{time_total}" -X POST http://localhost:8099/v1/tts \
                -H "Content-Type: application/json" \
                -d "{\"text\":\"$TEXT_SHORT\",\"speaker\":\"$SPEAKER\",\"language\":\"English\",\"seed\":$SEED}" \
                -o "$BENCH_DIR/06b_server_warm.wav")
            DUR=$(soxi -D "$BENCH_DIR/06b_server_warm.wav" 2>/dev/null || echo "?")
            printf "wall %-6ss  audio %-5ss\n" "$T" "$DUR"
            PASS=$((PASS + 1))

            # Long
            printf "  %-45s" "0.6B server long"
            T=$(curl -s -w "%{time_total}" -X POST http://localhost:8099/v1/tts \
                -H "Content-Type: application/json" \
                -d "{\"text\":\"$TEXT_LONG\",\"speaker\":\"$SPEAKER\",\"language\":\"English\",\"seed\":$SEED}" \
                -o "$BENCH_DIR/06b_server_long.wav")
            DUR=$(soxi -D "$BENCH_DIR/06b_server_long.wav" 2>/dev/null || echo "?")
            printf "wall %-6ss  audio %-5ss\n" "$T" "$DUR"
            PASS=$((PASS + 1))

            kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
        else
            kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
            skip_bench "0.6B server" "server failed to start"
        fi
    fi

    echo ""
else
    echo "── 0.6B: SKIP ($MODEL_06B not found) ──"
    echo ""
fi

# ── 1.7B Model ──

MODEL_17B=qwen3-tts-1.7b
if model_exists "$MODEL_17B"; then
    echo "── 1.7B ($MODEL_17B) ──"

    run_bench "1.7B short normal" "$BENCH_DIR/17b_short.log" \
        -d "$MODEL_17B" --text "$TEXT_SHORT" -s "$SPEAKER" -l English --seed "$SEED" -o "$BENCH_DIR/17b_short.wav"
    run_bench "1.7B long normal" "$BENCH_DIR/17b_long.log" \
        -d "$MODEL_17B" --text "$TEXT_LONG" -s "$SPEAKER" -l English --seed "$SEED" -o "$BENCH_DIR/17b_long.wav"
    run_bench "1.7B short stream" "$BENCH_DIR/17b_short_stream.log" \
        -d "$MODEL_17B" --text "$TEXT_SHORT" -s "$SPEAKER" -l English --seed "$SEED" --stream -o "$BENCH_DIR/17b_short_stream.wav"
    run_bench "1.7B long stream" "$BENCH_DIR/17b_long_stream.log" \
        -d "$MODEL_17B" --text "$TEXT_LONG" -s "$SPEAKER" -l English --seed "$SEED" --stream -o "$BENCH_DIR/17b_long_stream.wav"
    if [ "$LEVEL" = "full" ]; then
        # Instruct
        run_bench "1.7B short instruct (angry)" "$BENCH_DIR/17b_instruct.log" \
            -d "$MODEL_17B" --text "$TEXT_SHORT" -s "$SPEAKER" -l English --seed "$SEED" \
            --instruct "Speak in a very angry tone" -o "$BENCH_DIR/17b_instruct.wav"
        # INT8
        run_bench "1.7B short INT8" "$BENCH_DIR/17b_int8_short.log" \
            -d "$MODEL_17B" --text "$TEXT_SHORT" -s "$SPEAKER" -l English --seed "$SEED" --int8 -o "$BENCH_DIR/17b_int8_short.wav"
        run_bench "1.7B long INT8" "$BENCH_DIR/17b_int8_long.log" \
            -d "$MODEL_17B" --text "$TEXT_LONG" -s "$SPEAKER" -l English --seed "$SEED" --int8 -o "$BENCH_DIR/17b_int8_long.wav"
        # .qvoice
        QVOICE_17B=$(qvoice_find "17b")
        if [ -n "$QVOICE_17B" ]; then
            run_bench "1.7B short qvoice" "$BENCH_DIR/17b_qvoice_short.log" \
                -d "$MODEL_17B" --load-voice "$QVOICE_17B" --text "$TEXT_SHORT" --seed "$SEED" -o "$BENCH_DIR/17b_qvoice_short.wav"
            run_bench "1.7B long qvoice" "$BENCH_DIR/17b_qvoice_long.log" \
                -d "$MODEL_17B" --load-voice "$QVOICE_17B" --text "$TEXT_LONG" --seed "$SEED" -o "$BENCH_DIR/17b_qvoice_long.wav"        else
            skip_bench "1.7B short qvoice" "no .qvoice file"
            skip_bench "1.7B long qvoice" "no .qvoice file"
        fi
    fi

    echo ""
else
    echo "── 1.7B: SKIP ($MODEL_17B not found) ──"
    echo ""
fi

# ── Summary ──

echo "============================================"
echo "  Summary: $PASS passed, $SKIP skipped"
echo "============================================"

if [ -n "$RESULTS" ]; then
    echo ""
    printf "%-45s  %-6s  %-7s  %-7s\n" "Test" "RTF" "Audio" "Wall"
    printf "%-45s  %-6s  %-7s  %-7s\n" "----" "---" "-----" "----"
    echo -e "$RESULTS" | while IFS='|' read -r label rtf dur wall; do
        [ -z "$label" ] && continue
        printf "%-45s  %-6s  %-5ss  %-5ss\n" "$label" "$rtf" "$dur" "$wall"
    done
fi

echo ""
echo "Logs and WAVs in $BENCH_DIR/"
