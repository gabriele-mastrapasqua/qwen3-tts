#!/usr/bin/env bash
# tools/check_isa.sh — compile-check (syntax + intrinsics, not run) the kernel paths THIS
# machine cannot execute, so a change to an #ifdef'd ISA section is caught on the dev box
# instead of on the rented one.
#
#   make check-isa            # both directions where the toolchain allows it
#
# Arm passes: dotprod-only and -march=armv8.6-a+i8mm+bf16 (+dotprod), with and without KAI
# x86 passes: AVX2/FMA, AVX-512F/BW/VL without VNNI, then VNNI + BF16 + AMX
# On an Arm Mac both passes run (Apple clang targets x86_64-apple-macos with the same SDK).
# On Linux x86 only the x86 pass runs natively; the Arm pass needs a cross sysroot (SKIP).
set -u
cd "$(dirname "$0")/.." || exit 1
CC_=${CC:-cc}
ROOT=$PWD
KAI_DIR=third_party/kleidiai
INGOT=third_party/ingot
INC="-Ivendor -I. -I$INGOT/include -I$KAI_DIR"
DEFS="-DUSE_BLAS -DQWEN_GIT_REV=\"check-isa\" -DQWEN_SIMD_PROFILE=\"check-isa\""
ENGINE="main.c qwen_tts.c qwen_tts_gguf.c qwen_tts_talker.c qwen_tts_code_predictor.c \
        qwen_tts_speech_decoder.c qwen_tts_kernels.c qwen_tts_dispatch.c qwen_tts_thread.c \
        qwen_tts_kernels_generic.c qwen_tts_kernels_neon.c qwen_tts_kernels_avx.c qwen_tts_audio.c \
        qwen_tts_emotion.c qwen_tts_compose.c qwen_tts_sampling.c qwen_tts_tokenizer.c \
        qwen_tts_server.c qwen_tts_voice_clone.c qwen_tts_speech_encoder.c qwen_tts_kleidi.c \
        qwen_tts_kleidi_dotprod.c qwen_tts_q8repack.c qwen_tts_q4export.c"
KAI_SRCS=$(ls $KAI_DIR/kai/ukernels/matmul/pack/*.c $KAI_DIR/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/*.c 2>/dev/null)
KAI_DOTPROD_SRCS="$KAI_DIR/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c \
                 $KAI_DIR/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c \
                 $KAI_DIR/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c \
                 $KAI_DIR/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c \
                 $KAI_DIR/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c \
                 $KAI_DIR/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c"
UNAME_S=$(uname -s); UNAME_M=$(uname -m)
FAIL=0
pass() {  # name  flags...
    local name="$1"; shift
    local log=/tmp/qwen_check_isa_${name}.log
    echo "=== $name: $* ==="
    local rc=0
    for f in $ENGINE $EXTRA_SRCS; do
        $CC_ -fsyntax-only -Wall -Wextra -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter \
            $INC $DEFS "$@" "$f" >>"$log" 2>&1 || { echo "  FAIL $f"; rc=1; }
    done
    if [ $rc = 0 ]; then echo "  PASS ($(echo $ENGINE $EXTRA_SRCS | wc -w | tr -d ' ') files)"; else FAIL=1; echo "  see $log"; fi
}
if [ "$UNAME_S" = Darwin ]; then
    DEFS="$DEFS -DACCELERATE_NEW_LAPACK"
    rm -f /tmp/qwen_check_isa_*.log
    EXTRA_SRCS=""
    pass x86-avx2-fma -target x86_64-apple-macos -mavx2 -mfma
    pass x86-avx512f-no-vnni -target x86_64-apple-macos -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl
    EXTRA_SRCS="$KAI_DOTPROD_SRCS"
    pass arm-dotprod-only -target arm64-apple-macos -march=armv8.2-a+dotprod
    EXTRA_SRCS="$KAI_SRCS"
    pass arm-i8mm-bf16 -target arm64-apple-macos -march=armv8.6-a+i8mm+bf16+dotprod
    EXTRA_SRCS=""
    pass x86-avx512-vnni-bf16-amx -target x86_64-apple-macos -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512dq \
         -mavx512vnni -mavx512bf16 -mamx-tile -mamx-int8 -mamx-bf16
else
    DEFS="$DEFS -DUSE_OPENBLAS -I/usr/include/openblas"
    rm -f /tmp/qwen_check_isa_*.log
    case "$UNAME_M" in
        x86_64)  EXTRA_SRCS=""; pass x86-avx2-fma -mavx2 -mfma
                 pass x86-avx512f-no-vnni -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl
                 pass x86-avx512-vnni-bf16-amx -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512dq \
                     -mavx512vnni -mavx512bf16 -mamx-tile -mamx-int8 -mamx-bf16
                 echo "=== arm-dotprod-only/i8mm-bf16: SKIP (needs an aarch64 cross sysroot on this host) ===" ;;
        aarch64) EXTRA_SRCS="$KAI_DOTPROD_SRCS"; pass arm-dotprod-only -march=armv8.2-a+dotprod
                 EXTRA_SRCS="$KAI_SRCS"; pass arm-i8mm-bf16 -march=armv8.6-a+i8mm+bf16+dotprod
                 echo "=== x86: SKIP (needs an x86_64 cross sysroot on this host) ===" ;;
    esac
fi
[ $FAIL = 0 ] && echo "check-isa: PASS" || { echo "check-isa: FAIL"; exit 1; }
