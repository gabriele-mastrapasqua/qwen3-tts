#!/usr/bin/env bash
# ============================================================================
# vps_validate.sh — turnkey AVX-512/VNNI validation for a rented x86 VPS.
#
# Validates the UNVALIDATED AVX-512 kernels written 2026-06-04 (PLAN 21.3):
#   - VNNI native int8 dot   (_mm512_dpbusd_epi32, SIMD=avx512vnni, commit d67648a)
#   - __m512 16-wide bf16 matvec (commit b89f30e)
# The Ryzen 7 6800H is AVX2-only and CAN'T run these — needs a Zen4+/Intel box
# with AVX-512 (ideally a V-cache chip like a 9950X3D → likely sub-1.0 RTF).
#
# Usage (on the VPS, inside the repo):
#     bash tests/vps_validate.sh                 # correctness only (no model needed)
#     bash tests/vps_validate.sh qwen3-tts-0.6b  # + RTF bench on that model dir
#
# THE DECISIVE GATE is `make test-selftest` (kernel numeric self-test): it compares
# the VNNI/AVX-512 matvecs to an f32 reference, so it's IMMUNE to the greedy-decode
# trajectory fork that makes cross-ISA end-to-end audio mel-corr a false alarm.
# Paste the full output back.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MODEL="${1:-}"
fail=0

hr() { printf '\n======== %s ========\n' "$1"; }

hr "0. CPU capabilities"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -iE "model name|^Flags" | sed 's/  */ /g' | head -2 || true
fi
# What we need (NOTE: the Linux /proc/cpuinfo flags use underscores for VNNI/BF16:
# 'avx512_vnni', 'avx512_bf16' — but 'avx512f/bw/vl/dq/cd' have NO underscore. The gcc
# build flags are the opposite (-mavx512vnni, no underscore). Don't mix them up.)
for feat in avx2 avx512f avx512bw avx512vl avx512_vnni avx512_bf16; do
    if grep -qw "$feat" /proc/cpuinfo 2>/dev/null; then
        echo "  HAVE  $feat"
    else
        echo "  MISS  $feat"
    fi
done
if ! grep -qw avx512_vnni /proc/cpuinfo 2>/dev/null; then
    echo "  !! This CPU lacks AVX-512-VNNI. The avx512vnni build will fail the runtime ISA"
    echo "     guard. Use a Zen4+/Intel-with-VNNI box, or build SIMD=avx2 to test AVX2 only."
fi

hr "1. Deps (build-essential + OpenBLAS)"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential libopenblas-dev \
        || echo "  (apt failed — install build-essential + libopenblas-dev manually)"
fi

# Build a given SIMD level into a uniquely-named binary, report size + caps.
build_one() {
    local simd="$1" bin="qwen_tts_$1"
    hr "BUILD SIMD=$simd"
    make clean >/dev/null 2>&1
    if make blas SIMD="$simd" >/tmp/build_$simd.log 2>&1; then
        cp -f qwen_tts "$bin"
        echo "  built -> $bin ($(du -h "$bin" | cut -f1))"
        ./"$bin" --caps
    else
        echo "  BUILD FAILED (tail of /tmp/build_$simd.log):"; tail -15 /tmp/build_$simd.log
        return 1
    fi
}

# ---- AVX-512 VNNI build = the target under test ----
if grep -qw avx512_vnni /proc/cpuinfo 2>/dev/null; then
    build_one avx512vnni || fail=1
    if [ -x qwen_tts_avx512vnni ]; then
        cp -f qwen_tts_avx512vnni qwen_tts
        ./qwen_tts --caps | grep -qi "VNNI" \
            && echo "  OK: --caps reports VNNI int8 dot" \
            || { echo "  FAIL: --caps does NOT report VNNI (path not compiled in?)"; fail=1; }

        hr "2. KERNEL SELF-TEST — THE correctness gate (VNNI vs scalar reference)"
        echo "  Expect: int8 rel_L2 ~4e-3 (VNNI correct) AND ~1e-7 on the fallback."
        echo "  A blown-up int8 rel_L2 on the dispatched path = broken VNNI offset math."
        make test-selftest || { echo "  FAIL: kernel self-test"; fail=1; }
    fi
else
    echo "  (skipping avx512vnni build — no VNNI on this CPU)"
fi

# ---- AVX2 baseline for A/B (always buildable on any x86-64) ----
build_one avx2 || build_one scalar || fail=1

# ---- Optional RTF bench (needs a model dir) ----
if [ -n "$MODEL" ] && [ -d "$MODEL" ]; then
    hr "3. RTF bench on $MODEL (int8 + int4, -j1 and -j4; VNNI on vs off)"
    TXT="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
    run() { # label  binary  extra-args
        local label="$1" bin="$2"; shift 2
        echo "--- $label ---"
        ./"$bin" -d "$MODEL" --text "$TXT" --seed 42 -s ryan -l English \
            "$@" -o /tmp/vps_out.wav 2>&1 | grep -iE "RTF|ms/f|Talker|Code Predictor|frames" | head -8
    }
    if [ -x qwen_tts_avx512vnni ]; then
        run "VNNI int8 -j4"            qwen_tts_avx512vnni --int8 -j 4
        run "VNNI int4 -j4 (x86 lever)" qwen_tts_avx512vnni --int4 -j 4
        run "VNNI int8 -j1"            qwen_tts_avx512vnni --int8 -j 1
        echo "--- VNNI OFF (int8 widen->FMA, -j4) for A/B ---"
        QWEN_NO_VNNI=1 ./qwen_tts_avx512vnni -d "$MODEL" --text "$TXT" --seed 42 -s ryan -l English \
            --int8 -j 4 -o /tmp/vps_out.wav 2>&1 | grep -iE "RTF|ms/f" | head -4
    fi
else
    echo ""
    echo "(no model dir given/found -> skipped RTF bench. Re-run with a model dir for perf:"
    echo "   bash tests/vps_validate.sh qwen3-tts-0.6b )"
fi

hr "SUMMARY"
if [ "$fail" -eq 0 ]; then
    echo "  ALL CORRECTNESS CHECKS PASSED. Paste this whole log back."
    echo "  Key lines to confirm: '--caps reports VNNI' + 'SELF-TEST PASSED' on the dispatched path."
else
    echo "  SOME CHECKS FAILED (see above). Paste the whole log back."
fi
exit "$fail"
