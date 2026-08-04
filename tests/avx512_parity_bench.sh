#!/usr/bin/env bash
# ============================================================================
# avx512_parity_bench.sh — validation battery for the perf/avx512-parity branch
# on a real AVX-512 box (Scaleway STANDARD3-X4C = EPYC 9555P Zen5, VNNI+BF16).
#
# What this branch adds (all COMPILE-CHECKED ONLY on M1 — this script is the
# hardware gate):
#   - attention dots/accumulators + rms_norm + bf16<->f32 bulk conv: AVX-512
#     (compile-time; A/B = branch binary vs main binary, same SIMD=)
#   - C4  bf16 matvec via VDPBF16PS       (runtime A/B: QWEN_NO_BF16DOT=1)
#   - C7v4 q4-VNNI deferred-reduce matvec (ladder: QWEN_Q4_VNNI_V4=0 -> v3,
#                                          + QWEN_Q4_VNNI_V3=0 -> v2)
#   - fused-QKV q4 VNNI twin              (runtime A/B: QWEN_NO_VNNI_QKV=1)
#   - + the audit leftover:                QWEN_PREFILL_MATMAT=1/0 A/B on Linux
#
# Usage on the box (QUIET machine, read --caps FIRST — lesson 2026-07-11:
# the default build is portable AVX2; without SIMD= the whole battery is invalid):
#   git clone https://github.com/<repo> && cd qwen-tts && git checkout perf/avx512-parity
#   apt install -y build-essential libopenblas-dev python3-pip && pip3 install librosa
#   ./download_model.sh   # 0.6B (+ 1.7B if benching int4/quant-mixed properly)
#   bash tests/avx512_parity_bench.sh qwen3-tts-0.6b
# Optional: MAIN_BIN=/path/qwen_tts_main for branch-vs-main RTF + mel-corr
# (build it from a worktree: git worktree add /tmp/qmain main && cd /tmp/qmain &&
#  make blas SIMD=avx512bf16 && cp qwen_tts /tmp/qwen_tts_main).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MODEL="${1:-qwen3-tts-0.6b}"
TXT="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
OUT=/tmp/apar; mkdir -p "$OUT"

if [ ! -d "$MODEL" ]; then
    echo "Model dir '$MODEL' not found."; exit 1
fi

build() { # $1=SIMD $2=outname
    if [ -x "$2" ]; then echo ">> reuse $2"; return 0; fi
    echo ">> building $2 (SIMD=$1) ..."
    make clean >/dev/null 2>&1
    if make blas SIMD="$1" >"$OUT/build_$1.log" 2>&1; then
        cp -f qwen_tts "$2"; echo "   ok"
    else
        echo "   BUILD FAILED (SIMD=$1):"; tail -8 "$OUT/build_$1.log"; exit 1
    fi
}
build avx512bf16 qwen_tts_avx512bf16     # the full-parity build (VNNI+BF16)
BIN=./qwen_tts_avx512bf16

run() { # $1=label $2=outwav then env/cmd+flags
    local label="$1" wav="$2"; shift 2
    local out rtf
    out=$("$@" -d "$MODEL" --text "$TXT" --seed 42 --temperature 0 -s ryan -l English -o "$wav" 2>&1)
    rtf=$(printf '%s\n' "$out" | grep -oE 'RTF [0-9.]+' | head -1 | awk '{print $2}')
    printf "  %-42s RTF %-7s\n" "$label" "${rtf:-ERR}"
}
melcmp() { # $1=label $2=ref.wav $3=out.wav
    printf "  mel-corr %-32s " "$1"
    python3 tests/compare_audio.py "$2" "$3" --min-corr 0.99 2>&1 | tail -1
}

echo "================================================================"
echo " avx512-parity battery — $MODEL"
echo " CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
echo "================================================================"
echo "[0] caps — the build AND the CPU must both show bf16+vnni:"
$BIN --caps 2>&1 | grep -iE 'matvec|int8 dot|bf16 dot|rms|runtime cpu|lever|WARNING'
echo
echo "[1] self-test ladder (every rung must PASS):"
for cfg in \
    "default(v4+dpbf16)|" \
    "no-bf16dot|QWEN_NO_BF16DOT=1" \
    "q4-v3|QWEN_Q4_VNNI_V4=0" \
    "q4-v2|QWEN_Q4_VNNI_V4=0 QWEN_Q4_VNNI_V3=0" \
    "fallback(no vnni/sdot)|QWEN_NO_VNNI=1 QWEN_NO_SDOT=1"; do
  name="${cfg%%|*}"; envs="${cfg#*|}"
  printf "  %-24s " "$name"
  env $envs $BIN --self-test 2>&1 | grep -oE 'SELF-TEST (PASSED|FAILED).*'
done
echo
echo "[2] C4 bf16: VDPBF16PS vs widen+FMA (speed + mel-corr gate):"
run "bf16 dpbf16 ON  -j4" "$OUT/x.wav"        $BIN -j4
run "bf16 dpbf16 OFF -j4" "$OUT/x.wav"        env QWEN_NO_BF16DOT=1 $BIN -j4
run "bf16 dpbf16 ON  -j1" "$OUT/bf16_on.wav"  $BIN -j1
run "bf16 dpbf16 OFF -j1" "$OUT/bf16_off.wav" env QWEN_NO_BF16DOT=1 $BIN -j1
melcmp "dpbf16 ON vs OFF (-j1 temp0)" "$OUT/bf16_off.wav" "$OUT/bf16_on.wav"
echo
echo "[3] C7 int4 ladder (default = v3, Zen5-measured 2026-08-04; beat int8):"
run "int8 (reference)      -j1" "$OUT/i8.wav" $BIN --int8 -j1
run "int4 default(v3)      -j1" "$OUT/v4.wav" $BIN --int4 -j1
run "int4 v4 (opt-in)      -j1" "$OUT/x.wav"  env QWEN_Q4_VNNI_V4=1 $BIN --int4 -j1
run "int4 v2               -j1" "$OUT/x.wav"  env QWEN_Q4_VNNI_V4=0 QWEN_Q4_VNNI_V3=0 $BIN --int4 -j1
run "int4 v4 QKV-vnni OFF  -j1" "$OUT/x.wav"  env QWEN_NO_VNNI_QKV=1 $BIN --int4 -j1
run "int8                  -j4" "$OUT/x.wav"  $BIN --int8 -j4
run "int4 default(v3)      -j4" "$OUT/x.wav"  $BIN --int4 -j4
run "int4 v4 (opt-in)      -j4" "$OUT/x.wav"  env QWEN_Q4_VNNI_V4=1 $BIN --int4 -j4
melcmp "int4 default vs int8 (sanity)" "$OUT/i8.wav" "$OUT/v4.wav"
echo "  (int4-vs-int8 mel-corr is a sanity ear-proxy, not a bit gate — different quant)"
echo
echo "[4] prefill A/B (audit leftover, Linux BLAS vs threaded matmat):"
run "prefill BLAS   (=0) bf16 -j4" "$OUT/x.wav" env QWEN_PREFILL_MATMAT=0 $BIN -j4
run "prefill matmat (=1) bf16 -j4" "$OUT/x.wav" env QWEN_PREFILL_MATMAT=1 $BIN -j4
echo
echo "[5] batched matmat regression (int8 ~3x / q4 ~1.6x expected, 2026-07-09):"
$BIN --matmat-bench 2>&1 | grep -E 'bf16|int8|int4|SPEEDUP' | head -9
if [ -n "${MAIN_BIN:-}" ] && [ -x "$MAIN_BIN" ]; then
  echo
  echo "[6] branch vs main (same SIMD=avx512bf16 — isolates the compile-time wins:"
  echo "    attention AVX-512 + rms_norm + bulk conv; plus everything above):"
  run "main   bf16 -j4" "$OUT/x.wav"         "$MAIN_BIN" -j4
  run "branch bf16 -j4" "$OUT/x.wav"         $BIN -j4
  run "main   int4 -j4" "$OUT/x.wav"         "$MAIN_BIN" --int4 -j4
  run "branch int4 -j4" "$OUT/x.wav"         $BIN --int4 -j4
  run "main   bf16 -j1" "$OUT/main_bf16.wav" "$MAIN_BIN" -j1
  run "branch bf16 -j1" "$OUT/br_bf16.wav"   $BIN -j1
  run "main   int4 -j1" "$OUT/main_i4.wav"   "$MAIN_BIN" --int4 -j1
  run "branch int4 -j1" "$OUT/br_i4.wav"     $BIN --int4 -j1
  melcmp "bf16 branch vs main (-j1 temp0)" "$OUT/main_bf16.wav" "$OUT/br_bf16.wav"
  melcmp "int4 branch vs main (-j1 temp0)" "$OUT/main_i4.wav"   "$OUT/br_i4.wav"
fi
echo "================================================================"
echo "Verdicts to fill: [2] dpbf16 faster? mel-corr>=0.99? [3] does v4 close the"
echo "int4-vs-int8 gap (was +21% with v3)? QKV-OFF shows the QKV twin's share."
echo "[4] which prefill default for __AVX512VNNI__ boxes. Paste the whole output."
