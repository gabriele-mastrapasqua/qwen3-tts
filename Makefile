# Makefile for Qwen3-TTS Pure C Inference Engine

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
CC = gcc

# Architecture / SIMD baseline (PLAN 21.3).
#   - macOS:      -march=native (single-vendor host; Apple Silicon NEON or Intel)
#   - Linux x86:  PORTABLE -mavx2 -mfma by default (Haswell 2013+). We deliberately
#                 do NOT use -march=native off-Mac: it locks codegen to the build
#                 host and SIGILLs on any older CPU in the field (the reported
#                 "AVX-512 Ryzen ran our scalar/illegal build" bug). Override with:
#                   SIMD=scalar  -> no AVX2 (pre-2013 CPUs; uses portable C fallback)
#                   SIMD=avx512  -> add AVX-512 (validate with Intel SDE / real HW)
#   - Linux ARM:  -march=native (NEON; M1-class and up)
SIMD ?= auto

# ── SIMD=auto: scegli il profilo PIU' ALTO che questa macchina e questo compilatore
# reggono, invece di ricadere in silenzio sull'AVX2 portabile.
#
# PERCHE' ESISTE (successo il 2026-08-20, ed e' costata mezz'ora di caccia al colpevole
# sbagliato): su una Emerald Rapids un `make blas` liscio compilava il default portabile,
# quindi niente AMX, niente VNNI, niente AVX-512-BF16 — tutti gli speedup spenti. Non
# fallisce, non avverte, passa il self-test: l'unico sintomo e' "~1.5x piu' lento", che si
# scambia per la macchina, per il vicino rumoroso o per il proprio ultimo commit. Un
# default che perde silenziosamente il 40% e' un default sbagliato.
#
# DUE CONDIZIONI, entrambe necessarie: la CPU dichiara l'estensione in /proc/cpuinfo E il
# compilatore accetta il flag. Un gcc vecchio su una Sapphire non sa cosa sia -mamx-int8,
# e senza la seconda prova la build fallirebbe invece di scendere di un gradino.
#
# ⚠️ IL BINARIO RESTA LEGATO ALL'HOST. E' il motivo per cui il default era portabile: con
# -mavx512f il compilatore emette AVX-512 anche fuori dai nostri kernel, e quel binario va
# in SIGILL su una CPU piu' vecchia. I percorsi AMX/VNNI nostri sono anche gated a runtime,
# il codice auto-vettorizzato no. Per un binario da distribuire:  make blas SIMD=portable
ifeq ($(SIMD),auto)
    ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
        ifeq ($(UNAME_S),Linux)
            # ⚠️ niente `case` qui dentro: $(shell ...) di Make conta le parentesi, e il
            # `*)` di un ramo case chiude $( in anticipo e tronca il comando. Solo grep.
            SIMD := $(shell \
                F=$$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null); \
                cc_ok() { $(CC) $$1 -E -x c /dev/null >/dev/null 2>&1; }; \
                has() { echo "$$F" | grep -qw "$$1"; }; \
                if has amx_int8 && has avx512_bf16 && cc_ok "-mamx-int8 -mamx-bf16 -mamx-tile"; then echo amx; \
                elif has avx512_bf16 && cc_ok -mavx512bf16; then echo avx512bf16; \
                elif has avx512_vnni && cc_ok -mavx512vnni; then echo avx512vnni; \
                elif has avx512f && cc_ok -mavx512f; then echo avx512; \
                elif has avx2; then echo portable; \
                else echo scalar; fi)
            $(info [simd] auto -> $(SIMD)   (make blas SIMD=portable per un binario non legato a questo host))
        else
            SIMD := portable
        endif
    else
        # macOS e Linux ARM restano su -march=native: l'etichetta serve solo perche' il
        # binario sappia dire di che build e' fatto (finisce in --caps).
        SIMD := native
    endif
endif
ifeq ($(UNAME_S),Darwin)
    ARCH_FLAGS = -march=native
else ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
    ifeq ($(SIMD),scalar)
        ARCH_FLAGS =
    else ifeq ($(SIMD),amx)
        # AMX (Sapphire Rapids = GCP c3, Emerald Rapids = GCP c4) + everything below it.
        # AMX is the only true MATRIX-MATRIX integer unit on x86: a 16x64 int8 tile takes
        # B=16 concurrent requests in one operand, which is the lever for server batching
        # (PLAN 0.nonies S2) rather than another few percent of SIMD width.
        # The AMX code is #ifdef'd AND runtime-gated (CPUID + the XTILEDATA permission,
        # which is requested once and read back), so this binary still runs on a CPU
        # without AMX — it just never enters the tile path. Check `--caps` says
        # "AMX ACTIVE" on the box: if it says permission DENIED the kernel is < 5.16 and
        # every number after that is a VNNI number.
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 \
                     -mamx-tile -mamx-int8 -mamx-bf16 -mavx2 -mfma
    else ifeq ($(SIMD),avx512bf16)
        # AVX-512 + VNNI + BF16 (native int8 dot + native bf16 dot VDPBF16PS).
        # Zen4/Zen5 (EPYC 9xx4/9xx5, Ryzen 7xxx+), Cooper Lake, Sapphire Rapids.
        # Check `--caps` runtime line for avx512bf16 BEFORE using this build.
        # -mavx512dq: _mm512_insertf32x8 (C7v4 scale-vec) is a DQ intrinsic — gcc
        # enforces the always_inline target match (clang cross-compile does not;
        # caught on the EPYC 2026-08-04). Every VNNI-era CPU has DQ.
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mavx2 -mfma
    else ifeq ($(SIMD),avx512vnni)
        # AVX-512 + VNNI (native int8 dot). Cascade Lake+/Ice Lake/Zen4+ (e.g. Ryzen 9950X3D).
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx2 -mfma
    else ifeq ($(SIMD),avx512)
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx2 -mfma
    else
        ARCH_FLAGS = -mavx2 -mfma
    endif
else
    ARCH_FLAGS = -march=native
endif

# ── PROVENIENZA INCISA NEL BINARIO ────────────────────────────────────────────
# Un report deve dire QUALE binario ha prodotto i numeri, non quale checkout c'era
# nella cartella quando lo hai letto. Su un box affittato le due cose divergono
# continuamente: si copiano sorgenti a mano, si ricompila, si dimentica. Il 2026-08-20
# ci sono volute mezz'ora e un braccio di controllo per scoprire che il binario in prova
# era compilato senza AMX — e la domanda "che versione sta girando?" non aveva risposta.
# `-dirty` compare se l'albero ha modifiche non committate: un numero preso da un albero
# sporco non e' riproducibile e deve dirlo da solo.
# Si puo' SCAVALCARE dalla riga di comando, ed e' il modo giusto di costruire su un box
# sincronizzato con scp: li' il checkout git e' quello vecchio della macchina e i sorgenti
# arrivano da fuori, quindi lo sha locale descriverebbe il codice sbagliato. Da usare:
#     make blas GIT_REV=$(git -C <repo locale> rev-parse --short HEAD)
GIT_REV := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
CFLAGS_BASE = -Wall -Wextra -O3 $(ARCH_FLAGS) -ffast-math \
              -DQWEN_GIT_REV=\"$(GIT_REV)\" -DQWEN_SIMD_PROFILE=\"$(SIMD)\"
LDLIBS = -lm -lpthread

# LZ4 (embedded in vendor/ — no external dependency needed)
CFLAGS_BASE += -Ivendor

# BLAS (Accelerate on macOS, OpenBLAS on Linux)
ifeq ($(UNAME_S),Darwin)
    CFLAGS_BASE += -DUSE_BLAS -DACCELERATE_NEW_LAPACK
    LDLIBS += -framework Accelerate
else
    CFLAGS_BASE += -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
    LDLIBS += -lopenblas
endif

CFLAGS = $(CFLAGS_BASE) -I$(INGOT_DIR)/include $(KAI_INC) $(EXTRA_CFLAGS)

# Source files
# KleidiAI (Apache-2.0, vendored subset - third_party/kleidiai/NOTICE.md). Only built
# on aarch64: the micro-kernels are Arm intrinsics. On any other host the engine-side
# glue compiles to stubs, so nothing else has to know.
# Gate on what the COMPILER can actually emit, not on `uname -m`: an M1 is aarch64 but
# has no i8mm, and the micro-kernel refuses to compile there (`#error i8mm extension
# required`). So probe for __ARM_FEATURE_MATMUL_INT8 under the real ARCH_FLAGS. On a
# box without it the engine-side glue compiles to stubs and nothing else changes.
KAI_HAS_I8MM := $(shell $(CC) $(ARCH_FLAGS) -dM -E -x c /dev/null 2>/dev/null | grep -c __ARM_FEATURE_MATMUL_INT8)
ifeq ($(KAI_HAS_I8MM),1)
KAI_DIR  = third_party/kleidiai
KAI_SRCS = $(KAI_DIR)/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c \
           $(KAI_DIR)/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c \
           $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c \
           $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
# The two matmul micro-kernels keep their inner loop in hand-written assembly; the .c
# half is only the C entry point, so BOTH must be built or the link fails with an
# undefined `kai_kernel_...`.
KAI_ASM  = $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod_asm.S \
           $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm_asm.S

# ── The INT8 slice: qai8dxp (LHS, quantized+packed per call) x qsi8cxp (RHS, one
# scale per output channel = OUR int8 weight format, byte for byte). Four candidates
# because the census has to answer "which tile", not only "Kleidi or not":
# two GEMV (dotprod, kr=4 and kr=8) and two GEMM (dotprod 16x4, i8mm 16x4).
KAI_I8_DIR = $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp
KAI_SRCS += $(KAI_DIR)/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c \
            $(KAI_DIR)/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c
# Only two of the four ship an assembly half; the other two are pure intrinsics.
# Listing a .S that does not exist is a make error, listing one too few is a link
# error - so this list is per-kernel, not per-directory.
KAI_ASM  += $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_asm.S \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_asm.S

# ── The BF16 slice: only where the compiler can emit BFMMLA. Separate probe from
# i8mm on purpose - Neoverse V1/V2 have both, but the two features are independent
# and a box with one and not the other must still build.
# grep -c would count __ARM_FEATURE_BF16_SCALAR_ARITHMETIC and
# __ARM_FEATURE_BF16_VECTOR_ARITHMETIC too, return 3, and silently fail the
# ifeq - which is exactly how the first Axion build linked without the bf16
# kernels. Match the bare macro.
KAI_HAS_BF16 := $(shell $(CC) $(ARCH_FLAGS) -dM -E -x c /dev/null 2>/dev/null | grep -qE '^#define __ARM_FEATURE_BF16 ' && echo 1 || echo 0)
ifeq ($(KAI_HAS_BF16),1)
KAI_BF_DIR = $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p
KAI_SRCS += $(KAI_DIR)/kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p1x4_f32_neon.c \
            $(KAI_DIR)/kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.c \
            $(KAI_DIR)/kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.c \
            $(KAI_BF_DIR)/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.c \
            $(KAI_BF_DIR)/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.c
endif
KAI_AOBJ = $(KAI_ASM:.S=.o)
KAI_INC  = -I$(KAI_DIR)
endif

SRCS = main.c \
       qwen_tts_kleidi.c qwen_tts_q8repack.c qwen_tts_q4export.c $(KAI_SRCS) \
       qwen_tts.c \
       qwen_tts_gguf.c \
       qwen_tts_talker.c \
       qwen_tts_code_predictor.c \
       qwen_tts_speech_decoder.c \
       qwen_tts_kernels.c \
       qwen_tts_thread.c \
       qwen_tts_kernels_generic.c \
       qwen_tts_kernels_neon.c \
       qwen_tts_kernels_avx.c \
       qwen_tts_audio.c \
       qwen_tts_emotion.c \
       qwen_tts_compose.c \
       qwen_tts_sampling.c \
       qwen_tts_tokenizer.c \
       qwen_tts_server.c \
       qwen_tts_voice_clone.c \
       qwen_tts_speech_encoder.c \
       vendor/lz4.c

OBJS = $(SRCS:.c=.o) $(KAI_AOBJ)

# KleidiAI's assembly halves. Plain aarch64 asm, assembled with the same flags so the
# .S sees the same -march (the i8mm one needs it).
%_asm.o: %_asm.S
	$(CC) $(CFLAGS) -c -o $@ $<
TARGET = qwen_tts

# ── ingot: the GGUF/safetensors library, vendored as a git subtree. Built by
# its own Makefile so this one never learns how it is compiled. ──────────────
INGOT_DIR := third_party/ingot
INGOT_LIB := $(INGOT_DIR)/libingot.a
$(INGOT_LIB):
	$(MAKE) -C $(INGOT_DIR) lib

# Refresh the vendored subtree from upstream (clean working tree required).
update-ingot:
	git subtree pull --prefix $(INGOT_DIR) https://github.com/mynah-org/ingot.git main --squash

# The vendored library has its own clean.
clean-ingot:
	@$(MAKE) -C $(INGOT_DIR) clean

.PHONY: update-ingot clean-ingot

# ── Build ─────────────────────────────────────────────────────────────────────
$(TARGET): $(OBJS) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(INGOT_LIB) $(LDLIBS)

blas: $(TARGET)

MODEL_DIR = qwen3-tts-0.6b

# Default: show help
all: help

help:
	@echo "qwen_tts — Qwen3-TTS Pure C Inference - Build Targets"
	@echo ""
	@echo "Build:"
	@echo "  make blas      - Build with BLAS acceleration (Accelerate/OpenBLAS)"
	@echo "  make debug     - Debug build with AddressSanitizer"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make info      - Show build configuration"
	@echo ""
	@echo "Test (requires models downloaded via ./download_model.sh):"
	@echo "  make test-small      - Run all 0.6B tests (English + Italian)"
	@echo "  make test-large      - Run all 1.7B tests (config + English + Italian)"
	@echo "  make test-large-int8 - Run 1.7B INT8 tests (Italian + English, seed 42)"
	@echo "  make test-large-int4 - Run 1.7B INT4 tests (Italian + English, seed 42)"
	@echo "  make test-large-quant - Run all 1.7B quantization tests (INT8 + INT4)"
	@echo "  make emotion-demo    - Render ryan through ALL mapped emotions via --emotion (1.7B); prints the output folder"
	@echo "  make emotion-para-demo - Emotion + inline paralinguistic [tag] ([laugh]/[sigh]/...) across langs/speakers (1.7B)"
	@echo "  make para-demo       - Shipped inline [tag]s ([wow]/[yawn]/[scoff]/[giggle]/[laugh]/[sigh]) on natural sentences (1.7B)"
	@echo "  make test-emotion-ft - Emotion fine-tune (.expr graft) smoke: CSP Italian on 1.7B (preset+clone, seed 42)"
	@echo "  make test-lora-it    - Emotion×voice×temp listening matrix (L16-26 LoRA; afplay links + full cmds)"
	@echo "  make emotion-seeds   - Seed-finder palette → the design notes (recommended seeds/lang/voice/emo; SLOW)"
	@echo "  make test-clone      - Voice clone e2e (generate ref → clone → stream)"
	@echo "  make demo-clone      - Voice clone demo using sample WAV"
	@echo "  make test-regression - Cross-model regression checks"
	@echo "  make test-all        - Run everything (0.6B + 1.7B + regression)"
	@echo ""
	@echo "Benchmark:"
	@echo "  make bench           - RTF benchmark (short+long, normal+stream)"
	@echo "  make bench-full      - Full benchmark (+ server, qvoice, instruct, INT8)"
	@echo "  make cp-microbench   - Build qwen_tts_cpbench (per-op Code Predictor breakdown)"
	@echo ""
	@echo "Box appena affittato (in QUESTO ordine — vedi docs/hardware-testing.md):"
	@echo "  make server-hw-check       - LA VERITA' SUL SILICIO: hardware + banda di memoria"
	@echo "  make server-batch-microbench - la curva B=1->2->4 e BatchEfficiency (~4 min, solo OSS)"
	@echo "  make mini-bench-06b|-17b   - 1/2/4 richieste parallele su un modello OSS"
	@echo "  make kernel-tune           - misura le soglie del dispatcher invece di indovinarle"
	@echo "  make tune-archive BOX=<nome> - idem, e archivia il JSON the design notes per il confronto cross-ISA"
	@echo "                               (tools/box_info.sh + tests/membw.c) + --caps + --self-test"
	@echo "                               + --matmat-bench. Nessun modello richiesto. JSON in HW_JSON=."
	@echo "                               (alias storico: make box-report)"
	@echo "  make membw                 - solo la banda: Copy/Triad con sweep di thread + ginocchio"
	@echo "  make bench-matrix[-full]   - poi la matrice RTF (richiede un modello scaricato)"
	@echo "  make check-matmat-parity   - i gemelli batched fanno l'aritmetica che dichiarano? (ISA nativa)"
	@echo "  make check-matmat-parity-x86 - idem sui kernel AVX2 x86, eseguiti sotto Rosetta 2 dal Mac ARM"
	@echo "  make check-isa             - compile-check dei percorsi ISA che questa macchina non ha"
	@echo "  make test-decoder-tool - Build qwen_tts_decoder_tool (decode a QWEN_DUMP_CODES dump alone)"
	@echo ""
	@echo "  Variabili: SPK (def. $(AX_SPK)) · TEXT · SEED ($(AX_SEED)) · Q ($(AX_Q))"
	@echo ""
	@echo "Example: make blas && ./$(TARGET) -d $(MODEL_DIR) -t \"Hello world\" -o output.wav"


# ── benchmark workflow (the design notes ────────────────────────

debug: CFLAGS = $(CFLAGS_BASE) -I$(INGOT_DIR)/include $(KAI_INC) -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
debug: LDLIBS += -fsanitize=address -fsanitize=undefined
debug: clean $(TARGET)

# Info
info:
	@echo "Platform: $(UNAME_S)"
	@echo "CC:       $(CC)"
	@echo "CFLAGS:   $(CFLAGS)"
	@echo "LDLIBS:   $(LDLIBS)"
	@echo "SRCS:     $(SRCS)"
	@echo "TARGET:   $(TARGET)"

# ── Experimental GPU backends (opt-in; `make blas` is NEVER affected) ──────────
# These add the backend seam (qwen_tts_backend) + one GPU TU and rebuild a fresh
# qwen_tts with -DQWEN_HAVE_{METAL,CUDA}. Clean rebuild so GPU/CPU .o never mix.
# See plan_v4 §E4 / docs/gpu-accel-analysis.md. Metal is dev-testable on M1;
# CUDA is cuBLAS-first (no nvcc for v1) and RTF-measured on the DGX/5090.
GPU_OBJS = qwen_tts_backend.o qwen_tts_cuda.o

# CUDA toolkit location — AUTO-DETECTED, because distros disagree: the NVIDIA
# .run/.deb installers use /usr/local/cuda, Arch Linux's `cuda` package uses
# /opt/cuda, and module/conda environments use neither. Order: an nvcc already on
# PATH wins (covers modules/conda/custom prefixes), then the two common locations.
# Override explicitly when needed: `make cuda CUDA_HOME=/path/to/cuda`.
CUDA_HOME ?= $(shell \
	if command -v nvcc >/dev/null 2>&1; then dirname "$$(dirname "$$(command -v nvcc)")"; \
	elif [ -x /usr/local/cuda/bin/nvcc ]; then echo /usr/local/cuda; \
	elif [ -x /opt/cuda/bin/nvcc ]; then echo /opt/cuda; \
	else echo /usr/local/cuda; fi)
# lib64 on x86_64 distro packages; lib on aarch64/Tegra and some prefixes.
CUDA_LIBDIR ?= $(shell \
	if [ -d "$(CUDA_HOME)/lib64" ]; then echo "$(CUDA_HOME)/lib64"; \
	else echo "$(CUDA_HOME)/lib"; fi)

.PHONY: metal cuda metal_build cuda_build

# Metal (macOS): clang compiles the one ObjC TU; gcc the rest; +Metal/Foundation.
metal:
	$(MAKE) clean
	$(MAKE) metal_build
metal_build: EXTRA_CFLAGS += -DQWEN_HAVE_METAL
metal_build: $(OBJS) $(GPU_OBJS) qwen_tts_metal.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(GPU_OBJS) qwen_tts_metal.o $(INGOT_LIB) $(LDLIBS) \
		-framework Metal -framework Foundation
	@echo ""
	@echo "Built ./$(TARGET) with Metal backend. Try: ./$(TARGET) --gpu-selftest --backend metal"

# The ObjC TU is clang-only (gcc can't build ObjC); ARC manages the Metal objects.
qwen_tts_metal.o: qwen_tts_metal.m
	clang -fobjc-arc -O3 -Wall -Wextra -Ivendor -MMD -MP -c -o $@ $<

# CUDA (Linux/DGX): cuBLAS-first + custom .cu kernels (nvcc). Needs the CUDA
# toolkit on the build box. NVCC_ARCH overrides the gencode (default sm_80 = Ampere+,
# bf16 tensor cores). qwen_tts_cuda_kernels.cu is compiled by nvcc, the rest by gcc.
NVCC ?= $(CUDA_HOME)/bin/nvcc
# PORTABLE multi-arch default: native cubins for the common CUDA GPUs + a PTX
# fallback so it also runs (via JIT) on archs not listed — old AND new cards, not
# just the DGX. Ampere sm_80/86 (A100/3090/3060), Ada sm_89 (40xx), Blackwell
# sm_120 (5090/5070) + compute_120 PTX (JITs onto sm_121 GB10 and future archs).
# Override for a single-arch fast build, e.g. `make cuda NVCC_ARCH="-arch=sm_121"`.
NVCC_ARCH ?= -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_120,code=compute_120
cuda:
	@if [ ! -x "$(NVCC)" ]; then \
		echo "ERROR: nvcc not found at $(NVCC) (CUDA_HOME=$(CUDA_HOME))."; \
		echo "Install the CUDA toolkit, or point CUDA_HOME at it explicitly:"; \
		echo "  make cuda CUDA_HOME=/opt/cuda        # Arch Linux (cuda package)"; \
		echo "  make cuda CUDA_HOME=/usr/local/cuda  # NVIDIA .run/.deb installer"; \
		exit 1; \
	fi
	@echo "CUDA toolkit: $(CUDA_HOME) (libs: $(CUDA_LIBDIR))"
	$(MAKE) clean
	$(MAKE) cuda_build
cuda_build: EXTRA_CFLAGS += -DQWEN_HAVE_CUDA -I$(CUDA_HOME)/include
cuda_build: $(OBJS) $(GPU_OBJS) qwen_tts_cuda_kernels.o qwen_tts_cuda_talker.o qwen_tts_cuda_decoder.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(GPU_OBJS) qwen_tts_cuda_kernels.o qwen_tts_cuda_talker.o qwen_tts_cuda_decoder.o $(INGOT_LIB) $(LDLIBS) \
		-L$(CUDA_LIBDIR) -lcublas -lcudart -lstdc++
	@# -lstdc++: the nvcc-compiled .cu object pulls in C++ ABI (__cxa_guard*/libstdc++);
	@#          the final link is driven by gcc, which doesn't add it automatically.
	@echo ""
	@echo "Built ./$(TARGET) with CUDA backend. Try: ./$(TARGET) --gpu-selftest --backend cuda"

# The .cu compute kernels are nvcc-only (host code stays gcc/cuBLAS).
# NOTE: these three had NO header prerequisites at all, yet _talker.cu and
# _decoder.cu include qwen_tts.h and embed qwen_tts_ctx_t — so a struct change
# left them stale and silently corrupting (same class of bug as the .c list;
# see the -MMD -MP comment above). Listed explicitly for now; switch to nvcc's
# `-MMD -MP` + `-include *.d` on a box where the CUDA build can be tested.
qwen_tts_cuda_kernels.o: qwen_tts_cuda_kernels.cu
	$(NVCC) $(NVCC_ARCH) -O3 -c -o $@ $<

# GPU-resident fused Talker step + resident decoder. --default-stream per-thread so the
# main-thread (Talker/CP) and the background decoder thread use INDEPENDENT default streams
# → the GPU overlaps generation and decode instead of serializing them (RTF win).
qwen_tts_cuda_talker.o: qwen_tts_cuda_talker.cu qwen_tts.h qwen_tts_kernels.h
	$(NVCC) $(NVCC_ARCH) -O3 --default-stream per-thread -I. -I$(CUDA_HOME)/include -c -o $@ $<

# GPU-resident ConvNet speech decoder (M3).
qwen_tts_cuda_decoder.o: qwen_tts_cuda_decoder.cu qwen_tts.h qwen_tts_kernels.h
	$(NVCC) $(NVCC_ARCH) -O3 --default-stream per-thread -I. -I$(CUDA_HOME)/include -c -o $@ $<

# CP micro-benchmark: separate binary instrumented with -DCP_MICROBENCH.
# Partitions per-frame Code Predictor time among sub-ops (QKV/attn/FFN/norm/lm_head).
# Clean rebuild into qwen_tts_cpbench so instrumented and normal .o never mix.
cp-microbench:
	$(MAKE) clean
	$(MAKE) TARGET=qwen_tts_cpbench EXTRA_CFLAGS=-DCP_MICROBENCH qwen_tts_cpbench
	@echo ""
	@echo "Built ./qwen_tts_cpbench  (run a normal generation; CP breakdown prints in the summary)"

# Standalone speech-decoder tool: decode a QWEN_DUMP_CODES text dump (16 ints/line)
# through the decoder alone (fixed input, no sampling) — A/B the decoder across changes.
test-decoder-tool: $(filter-out main.o,$(OBJS)) test_decoder_standalone.o
	$(CC) $(CFLAGS) -o qwen_tts_decoder_tool $^ $(LDLIBS)
	@echo "Built ./qwen_tts_decoder_tool  (usage: ./qwen_tts_decoder_tool codes.txt [model_dir] [out.wav])"

# Compile C sources. -MMD -MP emits a .d per object listing every header it
# actually included, so a header edit rebuilds exactly what depends on it. The
# hand-written list below used to do this and drifted: qwen_tts_compose.o,
# _audio.o, _emotion.o and _speech_encoder.o included qwen_tts.h without a rule,
# so growing a struct in it left them compiled against the OLD layout — silent
# heap corruption at runtime, far from the edit. Never hand-maintain this again.
%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# The RVQ nearest-neighbor encode (speech encoder) is numerically sensitive: -ffast-math
# miscompiles it and SIGSEGVs on 1.7B (a bad codegen in the NN distance search; harmless on
# 0.6B due to heap layout). It's an offline one-time path (voice creation), NOT a hot loop, so
# compile this single TU without -ffast-math. See PLAN (icl-encode root-cause TODO).
qwen_tts_speech_encoder.o: qwen_tts_speech_encoder.c
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -MMD -MP -c -o $@ $<

# Header dependencies: auto-generated by -MMD -MP into <obj>.d, pulled in here.
# (`-` so a first, .d-less build does not warn.)
-include $(OBJS:.o=.d) qwen_tts_backend.d qwen_tts_cuda.d qwen_tts_metal.d

# Clean (also the opt-in GPU objects, which are NOT in $(OBJS) — else a stale
# stub .o from a `make metal` build would silently be reused by `make cuda`).
clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(TARGET) qwen_tts_backend.o qwen_tts_cuda.o qwen_tts_metal.o qwen_tts_cuda_kernels.o
	rm -f qwen_tts_backend.d qwen_tts_cuda.d qwen_tts_metal.d vendor/lz4.d
	rm -f test_decoder_standalone.o test_decoder_standalone.d qwen_tts_decoder_tool
	rm -f tests/decode_quantum_bench.o tests/decode_quantum_bench.d qwen_tts_decode_quantum

# Debug build
# ⚠️ $(KAI_INC) IS REQUIRED HERE. Without it the KleidiAI sources cannot find
# kai/kai_common.h and `make debug` fails on every box that HAS i8mm - i.e. exactly the
# hardware we deploy on. It never showed up on the M1 dev machine because there
# KAI_HAS_I8MM is 0, so KAI_SRCS is empty and the include is not needed. Found 2026-08-25
# on the Axion box: the sanitizer build, the one gate that would catch a use-after-free in
# the cancellation path, had never been buildable on ARM+KleidiAI.

bench-server: $(TARGET)
	@bash tests/serve_batch_bench.sh $(MODEL_SMALL)

check-isa:
	@echo "=== Compile-check newer-ISA paths (syntax only, not run) ==="

# Emotion + PARALINGUISTIC demo: like emotion-demo, but each clip embeds an inline para [tag]
# ([laugh]/[sigh]/[huff]/[ugh]) in a real emotional sentence, across languages/speakers/emotions.
# Same one-flag --emotion router as emotion-demo (preset→STEER, clone→COMBINE), which now composes
# with the inline [tag] (engine fix 2026-06-30: the routed emotion steer is preserved on the spoken
# spans). Logic in tests/emotion_para_demo.sh. Needs 1.7B (+ .expr packs only for the galatea
# COMBINE clip). Override: EMO_PARA_DEMO_DIR=...
emotion-para-demo: $(TARGET)
	@bash tests/emotion_para_demo.sh

# Real-kernel batched matmat throughput: qwen_matmat_{bf16,int8,q4_0} vs B*matvec,
# per precision, at the current thread count (override with QWEN_BATCH_B / -j via THREADS).
matmat-bench: $(TARGET)
	@echo "=== Batched matmat twins vs B*matvec (real kernels, 4 threads) ==="
	@./$(TARGET) --matmat-bench
	@echo "=== (single thread = compute-bound reference) ==="
	@./$(TARGET) --matmat-bench -j 1

# ── Kernel numeric self-test (matvec correctness vs f32 reference) ──
# Cross-ISA correctness gate for the SIMD matvecs (bf16/int8/argmax) that does NOT depend
# on a full-pipeline golden, so it's immune to the greedy-decode trajectory fork that makes
# end-to-end audio mel-corr a FALSE ALARM cross-ISA (the test-golden cross-ISA caveat).
# Runs the dispatched path AND the scalar/widen fallback (QWEN_NO_SDOT/QWEN_NO_VNNI). On the
# AVX-512 VPS this is THE proof the VNNI int8 dot + __m512 bf16 matvec are correct.
# Needs no model — fast, run anywhere.
test-selftest: $(TARGET)
	@echo "=== Kernel self-test (dispatched path) ==="
	@./$(TARGET) --self-test || { echo "FAIL: kernel self-test (dispatched)"; exit 1; }
	@echo "=== Kernel self-test (scalar/widen fallback: QWEN_NO_SDOT=1 QWEN_NO_VNNI=1) ==="
	@QWEN_NO_SDOT=1 QWEN_NO_VNNI=1 ./$(TARGET) --self-test || { echo "FAIL: kernel self-test (fallback)"; exit 1; }
	@echo "PASS: kernel self-test (both paths numerically correct)"
	@echo ""

# ── Test targets ──────────────────────────────────────────────────────────────
# Models must be downloaded first via ./download_model.sh
# Tests verify: model loading, config parsing, generation, WAV output, non-zero audio

MODEL_SMALL = qwen3-tts-0.6b
MODEL_LARGE = qwen3-tts-1.7b
MODEL_BASE_SMALL = qwen3-tts-0.6b-base
MODEL_VOICE_DESIGN = qwen3-tts-voice-design
TEST_DIR = /tmp/qwen_tts_tests

# Helper script for test validation
# Usage: validate_test <wav_file> <label>
define validate_wav
	@if [ ! -f $(1) ]; then echo "FAIL: $(1) not created"; exit 1; fi
	@WAV_SIZE=$$(stat -f%z $(1) 2>/dev/null || stat -c%s $(1) 2>/dev/null); \
	 if [ "$$WAV_SIZE" -le 44 ]; then echo "FAIL: $(1) is empty ($$WAV_SIZE bytes)"; exit 1; fi
	@if ! grep -q "Generated [1-9]" $(1).log; then echo "FAIL: no frames generated"; exit 1; fi
	@if grep -qi "nan" $(1).log; then echo "WARN: NaN detected in output"; fi
	@if grep -q "MISSING" $(1).log; then echo "FAIL: speech decoder weights MISSING"; exit 1; fi
	@echo "PASS: $(2)"
	@echo ""
endef

# ── Small model (0.6B) tests ──

test-small-en:
	@echo "--- 0.6B English ryan ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_SMALL) -s ryan -l English \
		--text "Hello, this is a test of the text to speech system." \
		-o $(TEST_DIR)/small_en.wav 2>&1 | tee $(TEST_DIR)/small_en.wav.log
	$(call validate_wav,$(TEST_DIR)/small_en.wav,0.6B English ryan)

test-small-it:
	@echo "--- 0.6B Italian ryan ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_SMALL) -s ryan -l Italian \
		--text "Ciao, questa è una prova del sistema di sintesi vocale." \
		-o $(TEST_DIR)/small_it.wav 2>&1 | tee $(TEST_DIR)/small_it.wav.log
	$(call validate_wav,$(TEST_DIR)/small_it.wav,0.6B Italian ryan)

test-small-vivian:
	@echo "--- 0.6B Italian vivian ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_SMALL) -s vivian -l Italian \
		--text "Buongiorno, come state oggi?" \
		-o $(TEST_DIR)/small_vivian.wav 2>&1 | tee $(TEST_DIR)/small_vivian.wav.log
	$(call validate_wav,$(TEST_DIR)/small_vivian.wav,0.6B Italian vivian)

test-small-stream:
	@echo "--- 0.6B Streaming WAV ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_SMALL) -s ryan -l English \
		--text "Hello, this is a streaming test of the system." \
		--stream -o $(TEST_DIR)/small_stream.wav 2>&1 | tee $(TEST_DIR)/small_stream.wav.log
	$(call validate_wav,$(TEST_DIR)/small_stream.wav,0.6B Streaming WAV)

test-small-stdout:
	@echo "--- 0.6B Raw PCM stdout ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_SMALL) -s ryan -l English \
		--text "Hello, this is a stdout test." \
		--stdout > $(TEST_DIR)/small_stdout.raw 2>$(TEST_DIR)/small_stdout.log
	@RAW_SIZE=$$(stat -f%z $(TEST_DIR)/small_stdout.raw 2>/dev/null || stat -c%s $(TEST_DIR)/small_stdout.raw 2>/dev/null); \
	 if [ "$$RAW_SIZE" -le 0 ]; then echo "FAIL: stdout produced no data"; exit 1; fi
	@echo "PASS: 0.6B Raw PCM stdout"
	@echo ""

test-small: test-small-en test-small-it test-small-vivian test-small-stream test-small-stdout
	@echo "=== All 0.6B tests passed ==="

# ── Large model (1.7B) tests ──

test-large-en:
	@echo "--- 1.7B English ryan ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_LARGE) -s ryan -l English \
		--text "Hello, this is a test of the text to speech system." \
		-o $(TEST_DIR)/large_en.wav 2>&1 | tee $(TEST_DIR)/large_en.wav.log
	$(call validate_wav,$(TEST_DIR)/large_en.wav,1.7B English ryan)

test-large-it:
	@echo "--- 1.7B Italian ryan ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_LARGE) -s ryan -l Italian \
		--text "Ciao, questa è una prova del sistema." \
		-o $(TEST_DIR)/large_it.wav 2>&1 | tee $(TEST_DIR)/large_it.wav.log
	$(call validate_wav,$(TEST_DIR)/large_it.wav,1.7B Italian ryan)

test-large-config:
	@echo "--- 1.7B config validation ---"
	@# Regression: config parser truncated nested objects, losing hidden_size=2048
	./$(TARGET) -d $(MODEL_LARGE) --text "Test." -o $(TEST_DIR)/large_cfg.wav 2>&1 | tee $(TEST_DIR)/large_cfg.log
	@if ! grep -q "hidden=2048" $(TEST_DIR)/large_cfg.log; then echo "FAIL: 1.7B hidden_size should be 2048"; exit 1; fi
	@if ! grep -q "inter=6144" $(TEST_DIR)/large_cfg.log; then echo "FAIL: 1.7B intermediate_size should be 6144"; exit 1; fi
	@if ! grep -q "MTP projection" $(TEST_DIR)/large_cfg.log; then echo "FAIL: 1.7B should have MTP projection"; exit 1; fi
	@if grep -q "MISSING" $(TEST_DIR)/large_cfg.log; then echo "FAIL: speech decoder weights MISSING"; exit 1; fi
	@echo "PASS: 1.7B config validation"
	@echo ""

test-large-instruct:
	@echo "--- 1.7B Instruct: angry ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_LARGE) -s ryan -l English \
		--text "I cannot believe you did that to me." \
		--instruct "Speak in a very angry and aggressive tone" \
		-o $(TEST_DIR)/large_angry.wav 2>&1 | tee $(TEST_DIR)/large_angry.wav.log
	$(call validate_wav,$(TEST_DIR)/large_angry.wav,1.7B Instruct angry)
	@echo "--- 1.7B Instruct: slow whisper ---"
	./$(TARGET) -d $(MODEL_LARGE) -s ryan -l English \
		--text "I cannot believe you did that to me." \
		--instruct "Speak very slowly and softly, in a sad whisper" \
		-o $(TEST_DIR)/large_whisper.wav 2>&1 | tee $(TEST_DIR)/large_whisper.wav.log
	$(call validate_wav,$(TEST_DIR)/large_whisper.wav,1.7B Instruct whisper)
	@echo "--- 1.7B Instruct: happy ---"
	./$(TARGET) -d $(MODEL_LARGE) -s ryan -l English \
		--text "I cannot believe you did that to me." \
		--instruct "Speak in a very happy, cheerful and excited tone" \
		-o $(TEST_DIR)/large_happy.wav 2>&1 | tee $(TEST_DIR)/large_happy.wav.log
	$(call validate_wav,$(TEST_DIR)/large_happy.wav,1.7B Instruct happy)

test-large-int8:
	@echo "--- 1.7B INT8 Italian ryan (seed 42) ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_LARGE) -s ryan -l Italian --seed 42 \
		--text "Ciao, come stai oggi? Spero tutto bene." \
		--int8 \
		-o $(TEST_DIR)/large_int8_it.wav 2>&1 | tee $(TEST_DIR)/large_int8_it.wav.log
	$(call validate_wav,$(TEST_DIR)/large_int8_it.wav,1.7B INT8 Italian ryan)
	@echo "--- 1.7B INT8 English ryan (seed 42) ---"
	./$(TARGET) -d $(MODEL_LARGE) -s ryan --seed 42 \
		--text "Hello, how are you doing today? I hope everything is going well." \
		--int8 \
		-o $(TEST_DIR)/large_int8_en.wav 2>&1 | tee $(TEST_DIR)/large_int8_en.wav.log
	$(call validate_wav,$(TEST_DIR)/large_int8_en.wav,1.7B INT8 English ryan)

test-large-int4:
	@echo "--- 1.7B INT4 Italian ryan (seed 42) ---"
	@mkdir -p $(TEST_DIR)
	./$(TARGET) -d $(MODEL_LARGE) -s ryan -l Italian --seed 42 \
		--text "Ciao, come stai oggi? Spero tutto bene." \
		--int4 \
		-o $(TEST_DIR)/large_int4_it.wav 2>&1 | tee $(TEST_DIR)/large_int4_it.wav.log
	$(call validate_wav,$(TEST_DIR)/large_int4_it.wav,1.7B INT4 Italian ryan)
	@echo "--- 1.7B INT4 English ryan (seed 42) ---"
	./$(TARGET) -d $(MODEL_LARGE) -s ryan --seed 42 \
		--text "Hello, how are you doing today? I hope everything is going well." \
		--int4 \
		-o $(TEST_DIR)/large_int4_en.wav 2>&1 | tee $(TEST_DIR)/large_int4_en.wav.log
	$(call validate_wav,$(TEST_DIR)/large_int4_en.wav,1.7B INT4 English ryan)

test-large-quant: test-large-int8 test-large-int4
	@echo "=== All 1.7B quantization tests passed ==="

test-large: test-large-config test-large-en test-large-it test-large-instruct
	@echo "=== All 1.7B tests passed ==="

# ── Cross-model regression tests ──

# Error-handling regression: bad invocations must FAIL cleanly (non-zero + clear message),
# never crash or silently succeed. No model needed -> fast + CI-friendly.
test-errors: $(TARGET)
	@echo "=== Error-handling test ==="
	@mkdir -p $(TEST_DIR)
	@if ./$(TARGET) -d $(MODEL_SMALL) >/dev/null 2>$(TEST_DIR)/err_notext.txt; then echo "FAIL: missing --text/--serve should error (exit 0)"; exit 1; fi
	@grep -qiE "text.*serve|--text" $(TEST_DIR)/err_notext.txt || { echo "FAIL: no clear message for missing --text"; cat $(TEST_DIR)/err_notext.txt; exit 1; }
	@echo "  PASS: missing --text/--serve errors cleanly"
	@if ./$(TARGET) -d /nonexistent_model_dir_xyz --text "x" -o /dev/null >/dev/null 2>$(TEST_DIR)/err_nomodel.txt; then echo "FAIL: nonexistent model dir should error (exit 0)"; exit 1; fi
	@echo "  PASS: nonexistent model dir errors cleanly"
	@if ./$(TARGET) --load-voice /nonexistent.qvoice -d $(MODEL_SMALL) --text "x" -o /dev/null >/dev/null 2>$(TEST_DIR)/err_novoice.txt; then echo "FAIL: missing .qvoice should error (exit 0)"; exit 1; fi
	@echo "  PASS: missing .qvoice errors cleanly"
	@echo "PASS: error-handling"
	@echo ""

test-emotion: $(TARGET)
	@echo "=== Expressivity / emotion (STEER) smoke test ==="
	@mkdir -p $(TEST_DIR)
	@# Emotion is a 1.7B qlsteer-STEER feature (the legacy .vec mood-palette manifest was
	@# retired 2026-07-08); the recipe weight (w12) is authoritative. 0.6B emotion is
	@# parked-neutral. Source of truth: docs/emotion-THE-recipe.md + resolve_emotion_recipe.
	@if [ -d $(MODEL_LARGE) ]; then \
	   ./$(TARGET) -d $(MODEL_LARGE) -j1 -T 0 --seed 42 -s ryan -l Italian --emotion joy \
	     --text "La riunione inizia domani mattina." -o $(TEST_DIR)/em_joy.wav 2>$(TEST_DIR)/em_joy.log; \
	   grep -qi "Emotion 'joy': mode=STEER" $(TEST_DIR)/em_joy.log || { echo "FAIL: --emotion joy did not resolve to STEER"; cat $(TEST_DIR)/em_joy.log; exit 1; }; \
	   grep -qi "ryan_joy.qlsteer" $(TEST_DIR)/em_joy.log || { echo "FAIL: joy steer vector not loaded"; cat $(TEST_DIR)/em_joy.log; exit 1; }; \
	   test -s $(TEST_DIR)/em_joy.wav || { echo "FAIL: joy produced no audio"; exit 1; }; \
	   echo "  PASS: --emotion joy -> STEER ryan_joy.qlsteer + audio"; \
	   ./$(TARGET) -d $(MODEL_LARGE) -j1 -T 0 --seed 42 -s ryan -l Italian --emotion sad \
	     --text "La riunione inizia domani mattina." -o $(TEST_DIR)/em_sad.wav 2>$(TEST_DIR)/em_sad.log; \
	   grep -qi "Emotion 'sad': mode=STEER" $(TEST_DIR)/em_sad.log || { echo "FAIL: --emotion sad did not resolve to STEER"; cat $(TEST_DIR)/em_sad.log; exit 1; }; \
	   grep -qi "ryan_sad.qlsteer" $(TEST_DIR)/em_sad.log || { echo "FAIL: sad steer vector not loaded"; cat $(TEST_DIR)/em_sad.log; exit 1; }; \
	   echo "  PASS: --emotion sad -> STEER ryan_sad.qlsteer"; \
	 else echo "  SKIP: 1.7B model absent (emotion is a 1.7B STEER feature)"; fi
	@# 0.6B emotion is parked-neutral: --emotion must not crash, just produces audio.
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian --emotion joy \
		--text "Ciao." -o $(TEST_DIR)/em_06b.wav 2>/dev/null; \
	 test -s $(TEST_DIR)/em_06b.wav || { echo "FAIL: 0.6B --emotion produced no audio"; exit 1; }
	@echo "  PASS: 0.6B --emotion parked-neutral (no crash, audio written)"
	@# Standalone --volume/--rate DSP (model-agnostic, no --emotion)
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian --volume 1.2 --rate 0.9 \
		--text "Ciao." -o $(TEST_DIR)/em_vr.wav 2>$(TEST_DIR)/em_vr.log
	@grep -qi "Volume: 1.20" $(TEST_DIR)/em_vr.log && grep -qi "Rate: 0.90" $(TEST_DIR)/em_vr.log || { echo "FAIL: standalone --volume/--rate not applied"; cat $(TEST_DIR)/em_vr.log; exit 1; }
	@echo "  PASS: standalone --volume/--rate"
	@echo "PASS: expressivity/emotion smoke"
	@echo ""

# Emotion fine-tune (.expr graft) SMOKE — the CSP Italian emotion FT this branch ships.
# The WOW recipe: 1.7B + EN instruct + T1.1 + the dense .expr applied on a preset and on an
# --icl-only clone graft. 2 short renders (seed 42 = stable; some seeds glitch). Asserts the
# pack loads (N tensors applied) + non-empty audio. The .expr packs are local-only (large),
# so SKIP cleanly when absent instead of failing a fresh checkout. Pack override: EXPR_FT=...
# The shipped WIN pack (was italian_csp.expr — stale name, the test silently SKIPped;
# caught 2026-07-11 auditing a "too fast" test-all).
EXPR_FT ?= presets/expr/italian_csp_topk6.expr
EMO_FT_INSTR = Speak with warm, bright happiness, smiling through the words.
EMO_FT_TEXT  = Che bella notizia, sono davvero felicissimo oggi!
test-emotion-ft: $(TARGET)
	@echo "=== Emotion fine-tune (.expr graft) smoke — CSP Italian on 1.7B ==="
	@mkdir -p $(TEST_DIR)
	@# One shell block so a SKIP truly skips the whole target: in make each @-line is a
	@# separate shell, so `exit 0` on its own line only ends that line — the render below
	@# would still run (missing --expr pack) and FAIL. Keep the guards + body together.
	@if [ ! -f $(EXPR_FT) ]; then echo "  SKIP: $(EXPR_FT) not present (local-only emotion FT pack)"; exit 0; fi; \
	 if [ ! -d $(MODEL_LARGE) ]; then echo "  SKIP: $(MODEL_LARGE) not present"; exit 0; fi; \
	 ./$(TARGET) -d $(MODEL_LARGE) -j1 -T 1.1 --seed 42 -s ryan -l Italian \
		--expr $(EXPR_FT) --instruct "$(EMO_FT_INSTR)" \
		--text "$(EMO_FT_TEXT)" -o $(TEST_DIR)/ft_ryan.wav 2>$(TEST_DIR)/ft_ryan.log; \
	 grep -qiE "Expressivity: applied [1-9][0-9]*/" $(TEST_DIR)/ft_ryan.log || { echo "FAIL: .expr pack not applied (preset)"; cat $(TEST_DIR)/ft_ryan.log; exit 1; }; \
	 test -s $(TEST_DIR)/ft_ryan.wav || { echo "FAIL: preset+FT produced no audio"; exit 1; }; \
	 echo "  PASS: emotion FT pack applied on preset ryan -> audio"; \
	 if [ -f voices/galatea_graft.qvoice ]; then \
		./$(TARGET) -d $(MODEL_LARGE) -j1 -T 1.1 --seed 42 -l Italian \
			--load-voice voices/galatea_graft.qvoice --icl-only \
			--expr $(EXPR_FT) --instruct "$(EMO_FT_INSTR)" \
			--text "$(EMO_FT_TEXT)" -o $(TEST_DIR)/ft_clone.wav 2>$(TEST_DIR)/ft_clone.log; \
		grep -qiE "Expressivity: applied [1-9][0-9]*/" $(TEST_DIR)/ft_clone.log || { echo "FAIL: .expr pack not applied (clone graft)"; cat $(TEST_DIR)/ft_clone.log; exit 1; }; \
		test -s $(TEST_DIR)/ft_clone.wav || { echo "FAIL: clone+FT produced no audio"; exit 1; }; \
		echo "  PASS: emotion FT pack applied on galatea --icl-only graft -> audio"; \
	 else echo "  SKIP: voices/galatea_graft.qvoice not present (run: bash download_voices.sh)"; fi; \
	 echo "PASS: emotion fine-tune (.expr) smoke"
	@echo ""

# Emotion DEMO for new users: render the validated `--emotion` recipe across emotions, LANGUAGES and a
# clone, using the WIN texts (recipe_final.sh / crosslang_emo.sh). The engine's --emotion auto-router
# picks expr/steer/instruct/temp itself, so the win can't get lost. Logic lives in tests/emotion_demo.sh
# (richer than a Makefile loop). Needs 1.7B + the .expr packs (`bash download_assets.sh`). Override: EMO_DEMO_DIR=...
emotion-demo: $(TARGET)
	@bash tests/emotion_demo.sh


# ── The SMALL model (0.6B) expressivity stack — emotion + para + clone, sub-realtime ──────────
# On the 0.6B `--emotion` used to be a no-op: the model has no steerable emotion subspace. Since
# 2026-08-05 the emotion rides on the VOICE instead (docs/emotion-06b-recipe.md) — you build one
# 4 KB voice asset per emotion, once, and the small model gets the full expressive stack at RTF<1.
#
# make emovoice VOICE=ryan                                     # build the 6 assets for a preset
# make emovoice VOICE=galatea LOAD=voices/galatea_graft.qvoice  # ... for a cloned voice
# make emovoice VOICE=ryan GRAFT=voices/galatea_06b_graft.qvoice # ... + 16.8MB grafts (best for anger)
#   (language: TTS_LANG=English — NOT LANG, which is the shell's own locale variable)
emovoice: $(TARGET)
	@bash tests/emovoice_build.sh

# make emo-06b-demo — the showcase: 6 emotions + 5 [tag]s + both together + a clone, all on the
# 0.6B under --int8, with the RTF printed for each. Needs `make emovoice VOICE=<v>` first.
emo-06b-demo: $(TARGET)
	@bash tests/emo_06b_demo.sh

# make para-demo — shipped inline paralinguistic [tag]s on natural sentences (post 2026-07-08 gate:
# wow/yawn/scoff + laugh/sigh, scoff s42, giggle standalone; phew parked). Prints afplay links.
para-demo: $(TARGET)
	@bash tests/para_demo.sh

# Ordered emotion TEST SUITE (tests/emo_suite.sh): per-LANGUAGE subfolders, emotion-MATCHED prompts, a steer
# weight sweep (w6/w8/w10) + a COMBINE variant per (lang×emotion), filenames that encode voice/mode/weight,
# plus a galatea-clone cross-language folder. For finding/confirming the per-language win. Scope it:
#   make emo-suite                      # all languages (long)
#   LANGS="de fr es" make emo-suite     # subset of languages
#   EMOS="anger sad" make emo-suite     # subset of emotions
emo-suite: $(TARGET)
	@bash tests/emo_suite.sh

# Emotion × voice × temp LISTENING matrix for the L16-26 emotion LoRA. Prints, per clip,
# a comment (voice/emotion/temp/.expr + instruct) + the FULL command + a `cd ... && afplay` link,
# so you can verify what produced each sound (e.g. SMALL ICL file, NOT the heavy qvoice WDELTA).
# All 7 EMOVO emotions (incl. disgust/fear/surprise) on { galatea SMALL-ICL clone, ryan preset }
# × { T0.9, T1.1 }. Override the pack with EXPR=... (e.g. the r32 instead of r64).
EXPR ?= presets/expr/italian_l1626_r64.expr
test-lora-it: $(TARGET)
	@bash tests/lora_matrix.sh Italian $(EXPR)

# Emotion seed-finder → recommended-seeds palette doc (the design notes. For each
# (language × voice × emotion) renders N seeds with --seed-audition --audition-keep, records the
# auto-pick (glitch+dur) + every take, and writes a usage doc with afplay links + full commands.
# Opt-in + SLOW (full IT+ES+clone matrix on 1.7B). Override scope via env: LANGS/VOICES_IT/EMOS_IT/N.
emotion-seeds: $(TARGET)
	@bash tests/emotion_seed_finder.sh $(if $(OUT_MD),$(OUT_MD),the design notes $(if $(N),$(N),5)

# Il cablaggio del batching sotto le due TRANSIZIONI che test-batch non esercita:
# N->1 (una richiesta resta sola) e 1->N (una viene ammessa accanto a una in corso).
# Entrambi i bug del 2026-08-20 vivevano li' e passavano test-batch senza un graffio.
# Aritmetica fissata (QWEN_BATCH_NOMATMUL=1) + greedy + seed fisso -> si pretende
# l'identita' bit a bit, quindi niente soglie da tarare.
test-batch-invariance: $(TARGET)
	@echo "=== 1/2 cablaggio: percorsi pinnati, identita' bit a bit dovuta ==="
	@MODEL=$(MODEL_SMALL) EP=/v1/tts PIN_ENV="env QWEN_BATCH_NO_SOLO=1 QWEN_BATCH_NOMATMUL=1" CRITERION=byte \
	  bash tests/serve_batch_invariance.sh
	@echo "=== 2/2 percorsi veri: si pretende la stessa DURATA, non gli stessi byte ==="
	@MODEL=$(MODEL_SMALL) CRITERION=frames bash tests/serve_batch_invariance.sh

test-batch: $(TARGET)
	@echo "=== Batched Talker step correctness (opt-in path vs single-stream) ==="
	@./$(TARGET) -d $(MODEL_SMALL) -j1 --batch-test 2>&1 | grep -E "probe|wiring|matmat path|batch-test"
	@./$(TARGET) -d $(MODEL_SMALL) -j1 --batch-test >/dev/null 2>&1 || { echo "FAIL: batched wiring not bit-exact vs single-stream"; exit 1; }
	@echo "  PASS: batched Talker step wiring is bit-exact; matmat path is a valid fp-order variant"
	@echo ""

batching-bench:
	@echo "=== Batching premise microbench (GEMV xB vs GEMM B) ==="
	$(CC) $(CFLAGS_BASE) -o /tmp/batching_bench tests/batching_bench.c -lm
	@/tmp/batching_bench

# ── LA VERITA' SUL SILICIO — si esegue PER PRIMA su un box appena affittato ──
# box-report = inventario hardware (tools/box_info.sh) + --caps + --self-test
# (nativo e fallback) + --matmat-bench (a pieni thread e a -j1).
#
# PERCHE' PRIMA DI TUTTO, e non "quando c'e' tempo". Un numero di server preso su
# una macchina di cui non sai (a) quanti core VERI ha, (b) se il governor scala la
# frequenza, (c) se sei su due nodi NUMA, (d) se il binario sta davvero eseguendo
# la primitiva che credi, non e' una misura: e' un aneddoto. Ed e' gia' successo —
# il q6 AVX-512 e' compilato da giugno e non l'abbiamo MAI visto girare: finche'
# --caps non lo dichiara attivo SU QUEL BOX, quel percorso non esiste.
#
# Se --self-test e' rosso, o --caps non nomina la primitiva attesa, FERMATI: la
# matrice che seguirebbe misura un fallback, e la si scoprirebbe dopo aver pagato.
#
# LA BANDA DI MEMORIA fa parte del report, e non e' un contorno: il Code Predictor
# rilegge i suoi pesi 16 VOLTE PER FRAME, quindi il carico e' memory-bound. Il numero
# che decide il batching non e' "quanti GFLOP" ma A QUANTI THREAD LA BANDA SATURA:
# se satura a 2, dare 8 thread a UNA richiesta e' sprecato e conviene dare 2 thread a
# QUATTRO richieste. tests/membw.c misura quel ginocchio (Copy+Triad, sweep di thread,
# array >= 4x L3 per non misurare la cache), e su piu' nodi NUMA aggiunge le celle
# local-vs-cross. Accanto resta la banda TEORICA, sempre etichettata come STIMA.
#
# Non richiede il modello scaricato: gira su un box vergine, prima di download_model.sh.
#
# HW_JSON=<path> cambia dove finisce l'artefatto JSON (default /tmp/tts/box_info.json).
# Quel JSON e' la fonte da INCORPORARE nel report del microbench di batching: chiavi
# stabili, autoesplicativo. Non riscrivere quella logica altrove — leggila da qui.
HW_JSON ?= /tmp/tts/box_info.json
MEMBW_BIN ?= /tmp/qwen_membw

$(MEMBW_BIN): tests/membw.c
	@$(CC) -Wall -Wextra -O2 -o $@ tests/membw.c -lpthread -lm



# ── Le due corse corte e mirate: 1/2/4 paralleli, quant di default, solo OSS ──
# Servono a rispondere subito a "2-4 richieste parallele sono decenti su questa
# macchina?" con la tabella per livello (TTFA p50/p95, RTF p50/p95, Q, BEff, RSS)
# e il verdetto contro le soglie dichiarate PRIMA (PLAN 0.nonies).
#
# Il 0.6B risponde in fretta ed e' il primo da lanciare; il 1.7B e' la taglia del
# prodotto (stessa architettura di un finetune reale, quindi gli stessi colli
# di bottiglia) ed e' quello il cui numero conta.
# I TRE QUANT, ognuno sul PROPRIO server pulito (lo script riavvia il server per
# braccio: page cache, allocatore e pool ripartono da zero, altrimenti il secondo
# quant erediterebbe lo stato del primo e il confronto non varrebbe).
#
#   --int8                     il gold: e' la qualita' di riferimento
#   --int4                     il piu' piccolo: meta' dei byte dell'int8
#   --quant-mixed-int6=q4n14   la mappa mista del 17/08: 14 layer int8 + 14 q4_0,
#                              -21,9% di traffico pesi contro l'all-int8, e sulla
#                              language identity aveva tenuto il pari con l'int8 gold
#
# La domanda a cui rispondono INSIEME: la mappa mista rende davvero, o e' solo un
# int4 con un nome piu' lungo? Serve vederli sulla stessa macchina, stessi livelli,
# stessi testi — e con il controllo che il quant chiesto sia DAVVERO quello caricato
# (lo script lo legge dal log del server e lo dichiara: un flag che non si applica
# non fallisce, degrada in silenzio).
MINI_ARMS ?= --int8 --int4 --quant-mixed-int6=q4n14



# ── Decoder batchato fra slot: parita' contro il percorso per-slot ────────────
# Stessi codici, stesso calendario ragged dei chunk (compresi i giri a 0 frame),
# decodificati in ENTRAMBI i modi e confrontati campione per campione.
# QWEN_PARITY_PAT=2 toglie i chunk da 1 frame, che isola l'ordine di riduzione
# GEMV-contro-GEMM di Accelerate da questo motore: con M=1 la BLAS prende un kernel
# diverso e l'ultimo bit si muove: e' una proprieta' della libreria, non del batching,
# e il percorso per-slot ce l'ha gia'.
tests/decoder_batch_parity.o: tests/decoder_batch_parity.c
	$(CC) $(CFLAGS) -I. -c -o $@ $<

# $(INGOT_LIB) e' esplicito: il caricatore safetensors e' vendorizzato in una libreria
# a parte, non negli OBJS, e senza di lei il link muore su _ingot_st_open_dir.
test-decoder-batch-parity: $(filter-out main.o,$(OBJS)) tests/decoder_batch_parity.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o qwen_tts_batch_parity $^ $(LDLIBS)
	@echo "--- calendario ragged CON chunk da 1 frame (percorso GEMV della BLAS in gioco) ---"
	@./qwen_tts_batch_parity $(MODEL_SMALL) 4 6
	@echo "--- stesso calendario, senza chunk da 1 frame (atteso: bit-identico) ---"
	@QWEN_PARITY_PAT=2 ./qwen_tts_batch_parity $(MODEL_SMALL) 4 6

# ── Costo di UNA decode call, contro (group x chunk) ────────────────────────
# Il trace di serving registra la durata della chiamata e quanti SLOT c'erano
# dentro, ma NON quanti frame: quindi la curva costo-per-frame non e' derivabile
# dagli artifact, e ogni domanda su "spezzare il quantum" dipende proprio da
# quella. Questo la misura sullo stesso entry point che usa il driver.
tests/decode_quantum_bench.o: tests/decode_quantum_bench.c
	$(CC) $(CFLAGS) -I. -c -o $@ $<

test-decode-quantum: $(filter-out main.o,$(OBJS)) tests/decode_quantum_bench.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o qwen_tts_decode_quantum $^ $(LDLIBS)
	@echo "Built ./qwen_tts_decode_quantum  (usage: ./qwen_tts_decode_quantum <model_dir> [threads] [reps])"

# ── Censimento kernel per shape: NOSTRI kernel vs KleidiAI, sulle shape VERE ──
# Due sole slice verticali, nessun catalogo: INT8 per-row (i nostri stessi pesi,
# stessa scala per riga) contro qsi8cxp, e il prefill BF16 contro bf16p. Il pack
# dell'RHS e' in colonna separata perche' al load si paga una volta; il pack
# dell'LHS e' dentro ns/call perche' in inferenza si paga a ogni chiamata.
kernel-census: $(filter-out main.o,$(OBJS)) tests/kernel_census_bench.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o qwen_kernel_census $^ $(LDLIBS)
	@./qwen_kernel_census --model $(or $(CMODEL),1.7b) $(CENSUS_ARGS)


# Le soglie misurate su QUESTA macchina, archiviate nel DB dei box (the design notes con lo
# stesso nome del suo box_info, cosi' che un confronto cross-arch possa mettere x86 e
# ARM nella stessa tabella. Le soglie NON si ereditano fra architetture, e nemmeno fra
# due binari con micro-kernel diversi: un tune vale per il binario che l'ha prodotto.
#   make tune-archive BOX=2026-08-21_gcp-c4a-standard-16_axion MODEL=qwen3-tts-1.7b-base
BOX ?= $(shell date +%Y-%m-%d)_$(shell uname -m)-$(shell hostname | tr -cd 'a-zA-Z0-9-')


server-hw-check: $(TARGET) $(MEMBW_BIN)
	@MEMBW_BIN=$(MEMBW_BIN) HW_JSON=$(HW_JSON) bash tests/bench_matrix.sh --silicon-only

# Alias storico: i doc e gli script che dicevano `box-report` continuano a funzionare.
box-report: server-hw-check

# La banda da sola, senza il resto del report (utile per un A/B rapido su un box).
membw: $(MEMBW_BIN)
	@$(MEMBW_BIN)

# Full per-box SIMD-check + RTF matrix (docs/hardware-testing.md). Copy onto any
# rented ARM/x86 box. `make bench-matrix` = box-info+caps+self-test+matmat-bench+
# single/batch RTF; `make bench-matrix-full` adds streaming + server. Quiet machine only.
bench-matrix: $(TARGET)
	@bash tests/bench_matrix.sh $(MODEL_SMALL)
bench-matrix-full: $(TARGET)
	@bash tests/bench_matrix.sh $(MODEL_SMALL) --full



# ── PARITA' DEI GEMELLI BATCHED — check-isa dice che COMPILA, questo che e' GIUSTO ──
# check-isa e' solo -fsyntax-only: garantisce che il percorso VNNI/SMMLA/AVX2 esista,
# non che faccia l'aritmetica che dichiara. tests/matmat_parity.c confronta i gemelli
# batched (int8, q4_0) contro un riferimento INTERO in C semplice che quantizza le
# attivazioni esattamente come il motore. Dove il dispatcher ha un GEMM intero vero,
# l'errore atteso e' ZERO — l'aritmetica intera non ha ordine di somma da discutere,
# quindi un rel != 0 e' un bug, non "rumore in virgola mobile".
PARITY_SRC = tests/matmat_parity.c qwen_tts_kernels.c qwen_tts_thread.c
PARITY_CF  = -Wall -Wextra -O2 -Ivendor -I.
check-matmat-parity:
	@echo "=== matmat parity — ISA nativa ==="
ifeq ($(UNAME_S),Darwin)
	@clang $(PARITY_CF) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -march=native \
	  $(PARITY_SRC) -framework Accelerate -lm -o /tmp/matmat_parity
else
	@$(CC) $(PARITY_CF) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas $(ARCH_FLAGS) \
	  $(PARITY_SRC) -lopenblas -lm -lpthread -o /tmp/matmat_parity
endif
	@/tmp/matmat_parity

# ⭐ ESEGUE DAVVERO I KERNEL x86 AVX2 DAL MAC ARM. Rosetta 2 supporta AVX2 (verificato
# su questa macchina il 2026-08-18), quindi il binario cross-compilato per x86-64-v3
# non si limita a compilare: gira, e i suoi numeri valgono. E' il modo di validare il
# kernel della classe di VPS piu' affittata (AVX2 senza VNNI) senza affittare nulla.
#
# ⚠️ DUE COSE DA SAPERE, o si legge male l'output:
#   1. Rosetta NON emula AVX-512. VNNI e AMX restano da validare SUL METALLO: qui
#      passano solo i percorsi AVX2. Un PASS qui non dice niente su c3d/c3.
#   2. Sotto Rosetta `--caps` riporta "runtime cpu: sse2" e stampa
#      "WARNING: built with AVX2 but this CPU lacks it -> will SIGILL". E' un FALSO
#      ALLARME dell'emulazione (CPUID non espone AVX2 mentre le istruzioni girano),
#      non un difetto del binario: il test infatti passa. Non "aggiustare" nulla.
check-matmat-parity-x86:
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
	@echo "=== matmat parity — x86-64-v3 (AVX2) sotto Rosetta 2 ==="
	@clang -target x86_64-apple-macos13 $(PARITY_CF) -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
	  -march=x86-64-v3 $(PARITY_SRC) -framework Accelerate -lm -o /tmp/matmat_parity_x86
	@/tmp/matmat_parity_x86
else
	@echo "check-matmat-parity-x86: solo su Mac ARM (serve Rosetta 2). Su x86 usa check-matmat-parity."
endif

test-compose: $(TARGET)
	@echo "=== Inline markup / --compose smoke test ==="
	@mkdir -p $(TEST_DIR)
	@# Inline markup auto-detected in --text -> multi-span compose
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian \
		--text "Che bella notizia! [pause:400ms] [sad] Devo andare... [sigh] [neutral] Ciao." \
		-o $(TEST_DIR)/mk_inline.wav 2>$(TEST_DIR)/mk_inline.log
	@grep -qi "Inline markup detected" $(TEST_DIR)/mk_inline.log || { echo "FAIL: inline markup not auto-detected in --text"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@# 3 spans: neutral / sad / neutral. [sigh] is an INLINE para tag -> folded as an
	@# onomatopoeia into the adjacent (sad) span's text (one generation, voice-native
	@# timbre), NOT a separate span. The old split/steering-span "splice" that made it a
	@# 4th span was rejected & deleted (CLAUDE.md para protocol) — so 3 is correct here.
	@grep -qi "composed 3 spans" $(TEST_DIR)/mk_inline.log || { echo "FAIL: expected 3 spans (neutral/sad+[sigh]/neutral)"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@grep -qi "inline \[tag\]->onomatopoeia" $(TEST_DIR)/mk_inline.log || { echo "FAIL: [sigh] not folded inline as onomatopoeia"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@grep -qi "pause 0.40s" $(TEST_DIR)/mk_inline.log || { echo "FAIL: [pause:400ms] not parsed"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@test -s $(TEST_DIR)/mk_inline.wav || { echo "FAIL: no audio"; exit 1; }
	@echo "  PASS: inline [tag] markup in --text (3 spans, pause, [sigh] folded inline)"
	@# Plain text (no tags) must NOT trigger compose
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian \
		--text "Frase normale senza tag." -o $(TEST_DIR)/mk_plain.wav 2>$(TEST_DIR)/mk_plain.log
	@if grep -qi "compose mode" $(TEST_DIR)/mk_plain.log; then echo "FAIL: plain text wrongly routed to compose"; cat $(TEST_DIR)/mk_plain.log; exit 1; fi
	@echo "  PASS: plain text stays on the normal path"
	@# Explicit --compose with English macros + pause
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l English \
		--compose "[excited] We won! | [pause:0.5] | [sad] But it is over. [sigh]" \
		-o $(TEST_DIR)/mk_compose.wav 2>$(TEST_DIR)/mk_compose.log
	@grep -qi "composed" $(TEST_DIR)/mk_compose.log || { echo "FAIL: --compose did not render"; cat $(TEST_DIR)/mk_compose.log; exit 1; }
	@echo "  PASS: explicit --compose"
	@echo "PASS: inline markup / compose smoke"
	@echo ""

test-regression:
	@echo "=== Regression tests ==="
	@echo ""
	@echo "--- Safetensors format (must load standard HF format, not custom .bin) ---"
	@# Both models must load from model.safetensors (no weights.bin)
	@if [ -f $(MODEL_SMALL)/weights.bin ]; then echo "WARN: weights.bin found in 0.6B dir (should use model.safetensors)"; fi
	@if [ -f $(MODEL_LARGE)/weights.bin ]; then echo "WARN: weights.bin found in 1.7B dir (should use model.safetensors)"; fi
	@if [ ! -f $(MODEL_SMALL)/model.safetensors ]; then echo "FAIL: 0.6B model.safetensors missing"; exit 1; fi
	@if [ ! -f $(MODEL_LARGE)/model.safetensors ]; then echo "FAIL: 1.7B model.safetensors missing"; exit 1; fi
	@if [ ! -f $(MODEL_SMALL)/speech_tokenizer/model.safetensors ]; then echo "FAIL: 0.6B speech_tokenizer missing"; exit 1; fi
	@if [ ! -f $(MODEL_LARGE)/speech_tokenizer/model.safetensors ]; then echo "FAIL: 1.7B speech_tokenizer missing"; exit 1; fi
	@echo "PASS: safetensors files present"
	@echo ""
	@echo "--- 0.6B vs 1.7B config divergence ---"
	./$(TARGET) -d $(MODEL_SMALL) --text "x" -o /dev/null 2>&1 | grep "^Config:" > $(TEST_DIR)/cfg_small.txt || true
	./$(TARGET) -d $(MODEL_LARGE) --text "x" -o /dev/null 2>&1 | grep "^Config:" > $(TEST_DIR)/cfg_large.txt || true
	@# 0.6B must have hidden=1024, 1.7B must have hidden=2048
	@if ! grep -q "hidden=1024" $(TEST_DIR)/cfg_small.txt; then echo "FAIL: 0.6B should have hidden=1024"; exit 1; fi
	@if ! grep -q "hidden=2048" $(TEST_DIR)/cfg_large.txt; then echo "FAIL: 1.7B should have hidden=2048"; exit 1; fi
	@# Both must have same head_dim=128 and same CP hidden=1024
	@if ! grep -q "head_dim=128" $(TEST_DIR)/cfg_small.txt; then echo "FAIL: 0.6B head_dim"; exit 1; fi
	@if ! grep -q "head_dim=128" $(TEST_DIR)/cfg_large.txt; then echo "FAIL: 1.7B head_dim"; exit 1; fi
	@echo "PASS: config divergence correct"
	@echo ""
	@echo "=== All regression tests passed ==="

# ── Combined ──

test-all: test-small test-large test-regression test-errors test-emotion test-emotion-ft test-compose test-caps test-selftest test-golden test-serve-repro
	@echo ""
	@echo "========================================="
	@echo "  All tests passed (0.6B + 1.7B)"
	@echo "========================================="

# ── Capability self-report regression (catches "we thought AVX existed") ──
# Asserts the binary's --caps report is internally consistent with the build arch, so a
# false "we have AVX2/threading" belief can't survive: the binary states the truth and this
# test enforces it. On ARM it MUST report NEON; on x86 it MUST report AVX2 (default build) or
# scalar (SIMD=scalar) for matvec — a regression to the old un-wired SCALAR fails loudly. The
# threads line must report an active pool (GCD/pthread/Win32), never SINGLE-THREAD (PLAN 21.2).
# Pure introspection, no model needed.

test-caps: $(TARGET)
	@echo "=== Capability report test ==="
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) --caps | tee $(TEST_DIR)/caps.txt
	@grep -q "matvec + attn:" $(TEST_DIR)/caps.txt || { echo "FAIL: --caps missing matvec line"; exit 1; }
	@grep -q "matvec threads:" $(TEST_DIR)/caps.txt || { echo "FAIL: --caps missing threads line"; exit 1; }
	@grep -q "int8 dot:" $(TEST_DIR)/caps.txt || { echo "FAIL: --caps missing int8 dot line"; exit 1; }
	@if grep -q "arch:.*arm64" $(TEST_DIR)/caps.txt; then \
	   grep -q "matvec + attn:    NEON" $(TEST_DIR)/caps.txt || { echo "FAIL: arm64 build must report NEON matvec"; exit 1; }; \
	 elif grep -q "arch:.*x86-64" $(TEST_DIR)/caps.txt; then \
	   grep -qE "matvec \+ attn:    (AVX2|scalar)" $(TEST_DIR)/caps.txt || { echo "FAIL: x86 must report AVX2 (default) or scalar (SIMD=scalar) matvec"; exit 1; }; \
	   if grep -q "WARNING: built with AVX2 but this CPU lacks it" $(TEST_DIR)/caps.txt; then echo "FAIL: AVX2 build on a non-AVX2 CPU"; exit 1; fi; \
	 fi
	@grep -q "matvec threads:" $(TEST_DIR)/caps.txt && ! grep -q "SINGLE-THREAD" $(TEST_DIR)/caps.txt || { echo "FAIL: threads must report an active pool (GCD/pthread/Win32), not SINGLE-THREAD"; exit 1; }
	@echo "PASS: --caps report consistent with build arch"
	@echo ""


# ── Golden-reference correctness (mel-correlation + duration) ──
# Regenerates output deterministically (-j1 --temperature 0 --seed 42) and compares to the
# committed golden WAVs in tests/golden/ via mel-spectrogram correlation (>=0.99) + duration
# (<=5%). Unlike validate_wav (which only checks "non-empty + frames"), this catches NUMERICAL
# regressions — a broken kernel that still emits audio fails here. mel-corr (not md5) is robust
# to benign +-1 LSB noise AND is the correct cross-ISA check for the future AVX2/x86 work
# (x86 won't be bit-identical to the ARM golden, but a correct build must still score ~0.99+).
# Requires python3 + librosa (numpy/scipy). RUN ON A QUIET MACHINE (heavy load can perturb
# the trajectory). Regenerate goldens after an INTENDED output change: make golden-update.
GOLDEN_EN = The quick brown fox jumps over the lazy dog on a sunny afternoon.
GOLDEN_IT = Buongiorno a tutti, questa è una dimostrazione del sistema di sintesi vocale.
GOLDEN_DET = -j1 --temperature 0 --seed 42

test-golden: $(TARGET)
	@echo "=== Golden-reference correctness test (mel-corr + duration) ==="
	@if ! python3 -c "import librosa" 2>/dev/null; then echo "SKIP: python3 librosa not installed (pip install librosa)"; exit 0; fi
	@mkdir -p $(TEST_DIR)
	@FAIL=0; \
	 ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o $(TEST_DIR)/gold_06b_en.wav >/dev/null 2>&1; \
	 python3 tests/compare_audio.py tests/golden/golden_06b_en.wav $(TEST_DIR)/gold_06b_en.wav --label "0.6B en" || FAIL=1; \
	 ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l Italian --text "$(GOLDEN_IT)" -o $(TEST_DIR)/gold_06b_it.wav >/dev/null 2>&1; \
	 python3 tests/compare_audio.py tests/golden/golden_06b_it.wav $(TEST_DIR)/gold_06b_it.wav --label "0.6B it" || FAIL=1; \
	 ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --int8 --text "$(GOLDEN_EN)" -o $(TEST_DIR)/gold_06b_en_int8.wav >/dev/null 2>&1; \
	 python3 tests/compare_audio.py tests/golden/golden_06b_en_int8.wav $(TEST_DIR)/gold_06b_en_int8.wav --label "0.6B en int8" || FAIL=1; \
	 if [ -d $(MODEL_LARGE) ]; then \
	   ./$(TARGET) -d $(MODEL_LARGE) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o $(TEST_DIR)/gold_17b_en.wav >/dev/null 2>&1; \
	   python3 tests/compare_audio.py tests/golden/golden_17b_en.wav $(TEST_DIR)/gold_17b_en.wav --label "1.7B en" || FAIL=1; \
	 else echo "SKIP: 1.7B (model absent)"; fi; \
	 if [ "$$FAIL" -ne 0 ]; then echo "FAIL: golden-reference mismatch (numerical regression?)"; exit 1; fi; \
	 echo "PASS: all golden references match"
	@echo ""

# Regenerate the committed golden WAVs (run after an INTENTIONAL, reviewed output change).
golden-update: $(TARGET)
	@echo "=== Regenerating golden references (review the diff before committing!) ==="
	@mkdir -p tests/golden
	./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o tests/golden/golden_06b_en.wav
	./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l Italian --text "$(GOLDEN_IT)" -o tests/golden/golden_06b_it.wav
	./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --int8 --text "$(GOLDEN_EN)" -o tests/golden/golden_06b_en_int8.wav
	@if [ -d $(MODEL_LARGE) ]; then ./$(TARGET) -d $(MODEL_LARGE) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o tests/golden/golden_17b_en.wav; fi
	@echo "Done. git diff tests/golden/ and commit if intended."

# ── Quant-ladder: per-codebook argmax-agreement across CP precisions ──
# The cheap "measure first" instrument (PLAN.md future-research C). The CP output
# (codebooks 1-15) FEEDS BACK into the Talker, so a free-running precision sweep
# forks the whole trajectory (different code0, different length) and measures
# nothing. So this TEACHER-FORCES: phase A runs bf16 to lay down reference "rails"
# (the 16-codes-per-frame stream); phase B replays those rails (QWEN_TF_CODES) at
# each CP precision (QWEN_CP_PREC, Talker stays bf16) so every precision sees
# bit-identical per-step inputs → the recorded argmax disagreement is PURE CP quant
# drift. quant_ladder.py then reports WHERE and HOW MUCH int4 drifts vs int8/bf16
# (per codebook index). Phase A also prints FFN activation sparsity %.
QL_DIR  = /tmp/qwen_qladder
QL_TEXT = The quick brown fox jumps over the lazy dog on a sunny afternoon, and then it ran across the wide green field without stopping.
quant-ladder: $(TARGET)
	@echo "=== Quant-ladder: teacher-forced CP precision sweep (Talker bf16) ==="
	@mkdir -p $(QL_DIR)
	@echo "  phase A: bf16 reference rails (+ FFN sparsity)"
	@QWEN_CP_PREC=bf16 QWEN_FFN_SPARSITY=1e-4 QWEN_DUMP_CODES=$(QL_DIR)/ref.codes ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(QL_TEXT)" -o $(QL_DIR)/ref.wav 2>&1 | grep -i "sparsity" || true
	@echo "  phase B: teacher-forced replay at each CP precision"
	@QWEN_TF_CODES=$(QL_DIR)/ref.codes QWEN_CP_PREC=bf16 QWEN_DUMP_CODES=$(QL_DIR)/bf16.codes ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(QL_TEXT)" -o $(QL_DIR)/bf16.wav --silent
	@QWEN_TF_CODES=$(QL_DIR)/ref.codes QWEN_CP_PREC=int8 QWEN_DUMP_CODES=$(QL_DIR)/int8.codes ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(QL_TEXT)" -o $(QL_DIR)/int8.wav --silent
	@QWEN_TF_CODES=$(QL_DIR)/ref.codes QWEN_CP_PREC=int4 QWEN_DUMP_CODES=$(QL_DIR)/int4.codes ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(QL_TEXT)" -o $(QL_DIR)/int4.wav --silent
	@QWEN_TF_CODES=$(QL_DIR)/ref.codes QWEN_CP_PREC=int4 QWEN_CP_Q2_FFN=down QWEN_DUMP_CODES=$(QL_DIR)/q2down.codes ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(QL_TEXT)" -o $(QL_DIR)/q2down.wav --silent
	@echo ""
	@python3 tests/quant_ladder.py ref:$(QL_DIR)/ref.codes bf16:$(QL_DIR)/bf16.codes int8:$(QL_DIR)/int8.codes int4:$(QL_DIR)/int4.codes q2:$(QL_DIR)/q2down.codes

# ── Mode matrix: quant × delivery (the combinations real usage hits) ──
# Each combination must RUN and produce coherent audio (non-empty + frames). Numeric
# correctness for the deterministic configs is covered by test-golden; here we assert the
# CROSS-PRODUCT works: int8/bf16 × normal/stream, plus SDOT on/off. One shell so a failure
# stops cleanly. Reloads the model each run (natural gap → reliable, unlike rapid-fire).
test-modes: $(TARGET)
	@echo "=== Mode matrix (quant × delivery) 0.6B ==="
	@mkdir -p $(TEST_DIR)
	@chk() { sz=$$(stat -f%z "$$1" 2>/dev/null || stat -c%s "$$1" 2>/dev/null || echo 0); \
	   if [ "$$sz" -le 44 ] || ! grep -q "Generated [1-9]" "$$1.log"; then echo "FAIL: $$2"; exit 1; fi; \
	   if grep -qi "nan" "$$1.log"; then echo "FAIL: $$2 (NaN)"; exit 1; fi; \
	   echo "  PASS: $$2 ($$sz B)"; }; \
	 ./$(TARGET) -d $(MODEL_SMALL) -j1 --seed 42 -s ryan -l English --text "$(GOLDEN_EN)" -o $(TEST_DIR)/m_bf.wav >$(TEST_DIR)/m_bf.wav.log 2>&1; chk $(TEST_DIR)/m_bf.wav "bf16 normal"; \
	 ./$(TARGET) -d $(MODEL_SMALL) -j1 --seed 42 -s ryan -l English --stream --text "$(GOLDEN_EN)" -o $(TEST_DIR)/m_bfs.wav >$(TEST_DIR)/m_bfs.wav.log 2>&1; chk $(TEST_DIR)/m_bfs.wav "bf16 stream"; \
	 ./$(TARGET) -d $(MODEL_SMALL) -j1 --seed 42 -s ryan -l English --int8 --text "$(GOLDEN_EN)" -o $(TEST_DIR)/m_i8.wav >$(TEST_DIR)/m_i8.wav.log 2>&1; chk $(TEST_DIR)/m_i8.wav "int8 normal (SDOT)"; \
	 ./$(TARGET) -d $(MODEL_SMALL) -j1 --seed 42 -s ryan -l English --int8 --stream --text "$(GOLDEN_EN)" -o $(TEST_DIR)/m_i8s.wav >$(TEST_DIR)/m_i8s.wav.log 2>&1; chk $(TEST_DIR)/m_i8s.wav "int8 stream"; \
	 QWEN_NO_SDOT=1 ./$(TARGET) -d $(MODEL_SMALL) -j1 --seed 42 -s ryan -l English --int8 --text "$(GOLDEN_EN)" -o $(TEST_DIR)/m_i8n.wav >$(TEST_DIR)/m_i8n.wav.log 2>&1; chk $(TEST_DIR)/m_i8n.wav "int8 normal (SDOT off)"; \
	 echo "PASS: mode matrix (5 combinations)"
	@echo ""

# ── Custom voice (.qvoice) — skip-if-absent (voices/ is gitignored / local-only) ──
test-qvoice: $(TARGET)
	@echo "=== Custom voice (.qvoice) test ==="
	@if [ ! -f voices/silvio_06b.qvoice ]; then echo "SKIP: voices/silvio_06b.qvoice not present (local-only)"; exit 0; fi; \
	 mkdir -p $(TEST_DIR); \
	 chk() { sz=$$(stat -f%z "$$1" 2>/dev/null || stat -c%s "$$1" 2>/dev/null || echo 0); \
	   if [ "$$sz" -le 44 ] || ! grep -q "Generated [1-9]" "$$1.log"; then echo "FAIL: $$2"; exit 1; fi; echo "  PASS: $$2 ($$sz B)"; }; \
	 ./$(TARGET) -d $(MODEL_SMALL) -j1 --seed 42 -l Italian --load-voice voices/silvio_06b.qvoice --text "Buongiorno, questo e un test della voce." -o $(TEST_DIR)/qv.wav >$(TEST_DIR)/qv.wav.log 2>&1; chk $(TEST_DIR)/qv.wav "qvoice bf16"; \
	 ./$(TARGET) -d $(MODEL_SMALL) -j1 --seed 42 --int8 -l Italian --load-voice voices/silvio_06b.qvoice --text "Buongiorno, questo e un test della voce." -o $(TEST_DIR)/qvi.wav >$(TEST_DIR)/qvi.wav.log 2>&1; chk $(TEST_DIR)/qvi.wav "qvoice int8"; \
	 echo "PASS: custom voice (bf16 + int8)"
	@echo ""

# ── E2E: ONE command that runs EVERYTHING available (skips missing models/voices) ──
# This is the comprehensive regression: small/large/regression/errors/caps/golden +
# quant (int8/int4) + mode matrix + custom voice + clone + voice-design + server suite.
e2e: $(TARGET)
	@echo "######################## E2E FULL REGRESSION ########################"
	@$(MAKE) --no-print-directory test-all
	@$(MAKE) --no-print-directory test-large-quant
	@$(MAKE) --no-print-directory test-modes
	@$(MAKE) --no-print-directory test-qvoice
	@$(MAKE) --no-print-directory test-clone
	@$(MAKE) --no-print-directory test-voice-design
	@$(MAKE) --no-print-directory test-serve-all
	@echo "######################## E2E COMPLETE — all green ########################"

# ── HTTP Server ──

serve: $(TARGET)
	./$(TARGET) -d $(MODEL_SMALL) --serve 8080

test-serve: $(TARGET)
	@echo "--- HTTP Server test ---"
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) -d $(MODEL_SMALL) --serve 8090 &>/dev/null & SERVER_PID=$$!; \
	 sleep 4; \
	 echo "  Testing /v1/health..."; \
	 HEALTH=$$(curl -s http://localhost:8090/v1/health); \
	 if ! echo "$$HEALTH" | grep -q '"ok"'; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: health check"; exit 1; fi; \
	 echo "  Testing /v1/speakers..."; \
	 SPEAKERS=$$(curl -s http://localhost:8090/v1/speakers); \
	 if ! echo "$$SPEAKERS" | grep -q '"ryan"'; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: speakers"; exit 1; fi; \
	 echo "  Testing /v1/tts..."; \
	 curl -s -X POST http://localhost:8090/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"Test.","speaker":"ryan"}' \
	   -o $(TEST_DIR)/serve_test.wav; \
	 if [ ! -f $(TEST_DIR)/serve_test.wav ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: no WAV"; exit 1; fi; \
	 WAV_SIZE=$$(stat -f%z $(TEST_DIR)/serve_test.wav 2>/dev/null || stat -c%s $(TEST_DIR)/serve_test.wav 2>/dev/null); \
	 if [ "$$WAV_SIZE" -le 44 ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: empty WAV"; exit 1; fi; \
	 kill $$SERVER_PID 2>/dev/null; \
	 echo "PASS: HTTP Server test"
	@echo ""

# ── Server benchmark: 2 sequential runs, same seed (bit-identical output) ──

test-serve-bench: $(TARGET)
	@echo "=== Server Benchmark (seed=42, 2 runs) ==="
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) -d $(MODEL_SMALL) --serve 8091 &>/dev/null & SERVER_PID=$$!; \
	 sleep 4; \
	 echo "--- Run 1 (cold) ---"; \
	 T1=$$(curl -s -w "%{time_total}" -X POST http://localhost:8091/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"The quick brown fox jumps over the lazy dog on a sunny afternoon.","speaker":"ryan","language":"English","seed":42}' \
	   -o $(TEST_DIR)/bench_run1.wav); \
	 S1=$$(stat -f%z $(TEST_DIR)/bench_run1.wav 2>/dev/null || stat -c%s $(TEST_DIR)/bench_run1.wav 2>/dev/null); \
	 echo "  $${T1}s, $$S1 bytes"; \
	 if [ "$$S1" -le 44 ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: empty WAV"; exit 1; fi; \
	 echo "--- Run 2 (warm) ---"; \
	 T2=$$(curl -s -w "%{time_total}" -X POST http://localhost:8091/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"The quick brown fox jumps over the lazy dog on a sunny afternoon.","speaker":"ryan","language":"English","seed":42}' \
	   -o $(TEST_DIR)/bench_run2.wav); \
	 S2=$$(stat -f%z $(TEST_DIR)/bench_run2.wav 2>/dev/null || stat -c%s $(TEST_DIR)/bench_run2.wav 2>/dev/null); \
	 echo "  $${T2}s, $$S2 bytes"; \
	 echo "--- Comparing outputs ---"; \
	 MD5_1=$$(md5sum $(TEST_DIR)/bench_run1.wav 2>/dev/null | cut -d' ' -f1 || md5 -q $(TEST_DIR)/bench_run1.wav 2>/dev/null); \
	 MD5_2=$$(md5sum $(TEST_DIR)/bench_run2.wav 2>/dev/null | cut -d' ' -f1 || md5 -q $(TEST_DIR)/bench_run2.wav 2>/dev/null); \
	 if [ "$$MD5_1" != "$$MD5_2" ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: outputs differ ($$MD5_1 vs $$MD5_2)"; exit 1; fi; \
	 kill $$SERVER_PID 2>/dev/null; \
	 echo "PASS: identical output ($$MD5_1)"
	@echo ""

# ── Server OpenAI-compatible API test ──

test-serve-openai: $(TARGET)
	@echo "=== Server OpenAI API test ==="
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) -d $(MODEL_SMALL) --serve 8092 &>/dev/null & SERVER_PID=$$!; \
	 sleep 4; \
	 echo "--- /v1/audio/speech (OpenAI-compatible) ---"; \
	 HTTP_CODE=$$(curl -s -w "%{http_code}" -X POST http://localhost:8092/v1/audio/speech \
	   -H "Content-Type: application/json" \
	   -d '{"input":"Hello, this is a test of the OpenAI compatible endpoint.","voice":"ryan","seed":42}' \
	   -o $(TEST_DIR)/openai_test.wav); \
	 if [ "$$HTTP_CODE" != "200" ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: HTTP $$HTTP_CODE"; exit 1; fi; \
	 WAV_SIZE=$$(stat -f%z $(TEST_DIR)/openai_test.wav 2>/dev/null || stat -c%s $(TEST_DIR)/openai_test.wav 2>/dev/null); \
	 if [ "$$WAV_SIZE" -le 44 ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: empty WAV ($$WAV_SIZE bytes)"; exit 1; fi; \
	 echo "  HTTP 200, $$WAV_SIZE bytes"; \
	 echo "--- Verify same seed produces same output via /v1/tts ---"; \
	 curl -s -X POST http://localhost:8092/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"Hello, this is a test of the OpenAI compatible endpoint.","speaker":"ryan","seed":42}' \
	   -o $(TEST_DIR)/openai_ref.wav; \
	 MD5_OAI=$$(md5sum $(TEST_DIR)/openai_test.wav 2>/dev/null | cut -d' ' -f1 || md5 -q $(TEST_DIR)/openai_test.wav 2>/dev/null); \
	 MD5_REF=$$(md5sum $(TEST_DIR)/openai_ref.wav 2>/dev/null | cut -d' ' -f1 || md5 -q $(TEST_DIR)/openai_ref.wav 2>/dev/null); \
	 if [ "$$MD5_OAI" != "$$MD5_REF" ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: OpenAI and TTS endpoints differ"; exit 1; fi; \
	 kill $$SERVER_PID 2>/dev/null; \
	 echo "PASS: OpenAI API (identical to /v1/tts)"
	@echo ""

# ── Server parallel requests test ──

test-serve-parallel: $(TARGET)
	@echo "=== Server Parallel Requests test ==="
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) -d $(MODEL_SMALL) --serve 8093 &>/dev/null & SERVER_PID=$$!; \
	 sleep 4; \
	 echo "--- Sending 2 concurrent requests ---"; \
	 curl -s -w "Req1: HTTP %{http_code} in %{time_total}s\n" -X POST http://localhost:8093/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"Hello, this is request number one.","speaker":"ryan","seed":100}' \
	   -o $(TEST_DIR)/parallel_1.wav & PID1=$$!; \
	 curl -s -w "Req2: HTTP %{http_code} in %{time_total}s\n" -X POST http://localhost:8093/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"And this is request number two.","speaker":"vivian","seed":200}' \
	   -o $(TEST_DIR)/parallel_2.wav & PID2=$$!; \
	 wait $$PID1; wait $$PID2; \
	 echo "--- Validating outputs ---"; \
	 FAIL=0; \
	 for f in $(TEST_DIR)/parallel_1.wav $(TEST_DIR)/parallel_2.wav; do \
	   if [ ! -f "$$f" ]; then echo "FAIL: $$f not created"; FAIL=1; continue; fi; \
	   SZ=$$(stat -f%z "$$f" 2>/dev/null || stat -c%s "$$f" 2>/dev/null); \
	   if [ "$$SZ" -le 44 ]; then echo "FAIL: $$f empty ($$SZ bytes)"; FAIL=1; else echo "  $$f: $$SZ bytes"; fi; \
	 done; \
	 kill $$SERVER_PID 2>/dev/null; \
	 if [ "$$FAIL" -ne 0 ]; then echo "FAIL: parallel test"; exit 1; fi; \
	 echo "PASS: 2 parallel requests served"
	@echo ""

# ── True concurrent worker-pool test (--workers 2) ──
# Unlike test-serve-parallel (single-worker server → serialized), this exercises
# the concurrent worker pool: it fires 2 simultaneous requests at a 2-worker
# server and verifies each output matches a single-worker reference via mel-corr
# (proving per-worker clones share weights correctly and don't corrupt state),
# across {bf16, int8, int4, voice+int8}. Kills the server BY NAME (runaway-safe).

test-serve-concurrent: $(TARGET)
	@MODEL=$(MODEL_SMALL) bash tests/test_parallel.sh
	@echo ""

# ── Server reproducibility regression (delta-prefill stale dec_x bug, fixed cbfa979) ──
# Two+ IDENTICAL consecutive requests MUST produce bit-identical output. Runs -j1
# --temperature 0 for full determinism (no threading FP noise / no sampling butterfly),
# so any difference is a real state-leak bug, not benign +-1 LSB.

test-serve-repro: $(TARGET)
	@echo "=== Server Reproducibility test (3 identical requests, -j1 temp0) ==="
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) -d $(MODEL_SMALL) -j1 --serve 8094 &>/dev/null & SERVER_PID=$$!; \
	 sleep 4; \
	 REQ='{"text":"The quick brown fox jumps over the lazy dog on a sunny afternoon.","speaker":"ryan","language":"English","seed":42,"temperature":0}'; \
	 for n in 1 2 3; do \
	   curl -s -X POST http://localhost:8094/v1/tts -H "Content-Type: application/json" \
	     -d "$$REQ" -o $(TEST_DIR)/repro_$$n.wav; \
	 done; \
	 kill $$SERVER_PID 2>/dev/null; \
	 S1=$$(stat -f%z $(TEST_DIR)/repro_1.wav 2>/dev/null || stat -c%s $(TEST_DIR)/repro_1.wav 2>/dev/null); \
	 if [ "$$S1" -le 44 ]; then echo "FAIL: empty WAV"; exit 1; fi; \
	 python3 tests/compare_repro.py $(TEST_DIR)/repro_1.wav $(TEST_DIR)/repro_2.wav $(TEST_DIR)/repro_3.wav
	@echo ""

# ── Combined server tests ──

# vLLM-style request-batching server: per-request correctness (force_matvec mel 1.0),
# zero cross-talk, real batching ([BATCH] N req), production matmat + stream fallback.
test-serve-batch: $(TARGET)
	@bash tests/serve_batch.sh $(MODEL_SMALL)

# Continuous admission: N=6 requests at max_batch=2 must all complete as the
# scheduler refills slots freed by EOS'd requests (peak admitted > batch width).
test-serve-continuous: $(TARGET)
	@bash tests/serve_continuous_stress.sh $(MODEL_SMALL) 8786 6 2

# Per-request streaming composed with batching (vLLM-style): concurrent /v1/tts/stream
# requests batched AND streamed; streamed PCM mel-corr ~1.0 vs single-stream.
test-serve-stream-batch: $(TARGET)
	@bash tests/serve_stream_batch.sh $(MODEL_SMALL)

## make server-soak — S16.5: il soak lungo. Trenta minuti di carico costante e le
## quattro derive che una corsa da tre minuti non puo' vedere: RSS (pendenza sulla
## seconda meta', non valore assoluto), thread/descrittori, latenza per finestra da
## 5 minuti, e una SONDA audio identica ogni 10 minuti — perche' la degenerazione di
## un decoder la sente l'orecchio, non una soglia. Solo modelli OSS.
##   make server-soak                      # 30 min, 0.6B int8, c=2
##   make server-soak MINUTES=60 LEVEL=4 MODEL=qwen3-tts-1.7b
MINUTES ?= 30
LEVEL ?= 2


test-serve-all: test-serve test-serve-bench test-serve-repro test-serve-openai test-serve-parallel test-serve-concurrent test-serve-batch test-serve-continuous test-serve-stream-batch test-stage-policy
	@echo "=== All server tests passed ==="

# ── RTF Benchmarks ──
# Quick RTF measurements across configs. Auto-skips missing models/voices.

bench: $(TARGET)
	@./bench.sh --level basic --seed 42

bench-full: $(TARGET)
	@./bench.sh --level full --seed 42

# ── Voice Clone e2e test ──
# Step 1: Generate reference audio with CustomVoice model
# Step 2: Use that audio as voice clone reference with Base model (different text)
# Step 3: Also test streaming + voice clone

test-clone: $(TARGET)
	@echo "=== Voice Clone e2e test ==="
	@if [ ! -d $(MODEL_SMALL) ]; then echo "SKIP: $(MODEL_SMALL) not found (run ./download_model.sh --model small)"; exit 0; fi
	@if [ ! -d $(MODEL_BASE_SMALL) ]; then echo "SKIP: $(MODEL_BASE_SMALL) not found (run ./download_model.sh --model base-small)"; exit 0; fi
	@mkdir -p $(TEST_DIR)
	@echo ""
	@echo "--- Step 1: Generate reference audio (CustomVoice) ---"
	./$(TARGET) -d $(MODEL_SMALL) -s ryan -l English \
		--text "The weather is beautiful today, perfect for a walk in the park." \
		--seed 42 \
		-o $(TEST_DIR)/clone_ref.wav 2>&1 | tee $(TEST_DIR)/clone_ref.wav.log
	$(call validate_wav,$(TEST_DIR)/clone_ref.wav,Voice Clone: reference generation)
	@echo "--- Step 2: Clone voice with different text ---"
	./$(TARGET) -d $(MODEL_BASE_SMALL) \
		--text "I love programming in C, it gives you complete control over the machine." \
		--ref-audio $(TEST_DIR)/clone_ref.wav \
		--xvector-only \
		-o $(TEST_DIR)/clone_output.wav 2>&1 | tee $(TEST_DIR)/clone_output.wav.log
	$(call validate_wav,$(TEST_DIR)/clone_output.wav,Voice Clone: cloned output)
	@if ! grep -q "Voice clone:" $(TEST_DIR)/clone_output.wav.log; then echo "FAIL: voice clone not active"; exit 1; fi
	@if ! grep -q "speaker embedding" $(TEST_DIR)/clone_output.wav.log; then echo "FAIL: no speaker embedding extracted"; exit 1; fi
	@echo "--- Step 3: Clone voice + streaming ---"
	./$(TARGET) -d $(MODEL_BASE_SMALL) \
		--text "Streaming also works perfectly with voice cloning mode." \
		--ref-audio $(TEST_DIR)/clone_ref.wav \
		--xvector-only \
		--stream \
		-o $(TEST_DIR)/clone_stream.wav 2>&1 | tee $(TEST_DIR)/clone_stream.wav.log
	$(call validate_wav,$(TEST_DIR)/clone_stream.wav,Voice Clone: streaming)
	@if ! grep -q "streamed" $(TEST_DIR)/clone_stream.wav.log; then echo "FAIL: not streamed"; exit 1; fi
	@echo "=== Voice Clone e2e test passed ==="
	@echo "Listen:"
	@echo "  Reference:  afplay $(TEST_DIR)/clone_ref.wav"
	@echo "  Cloned:     afplay $(TEST_DIR)/clone_output.wav"
	@echo "  Streamed:   afplay $(TEST_DIR)/clone_stream.wav"

# ── VoiceDesign test ──

# NOTE: the whole body runs in ONE shell (\ continuations) so the SKIP `exit 0`
# actually stops the recipe — a per-line `@if ...; exit 0; fi` only exits its own
# sub-shell and the following model-run lines would still execute (the old bug).
test-voice-design: $(TARGET)
	@echo "=== VoiceDesign test ==="
	@if [ ! -f $(MODEL_VOICE_DESIGN)/model.safetensors ]; then \
	   echo "SKIP: $(MODEL_VOICE_DESIGN) not found or incomplete (run ./download_model.sh --model voice-design)"; \
	   exit 0; \
	 fi; \
	 mkdir -p $(TEST_DIR); \
	 echo "--- VoiceDesign: British male ---"; \
	 ./$(TARGET) -d $(MODEL_VOICE_DESIGN) -l English --voice-design \
	   --instruct "A deep male voice with a British accent, speaking slowly and calmly" \
	   --text "Good evening, welcome to the broadcast." \
	   -o $(TEST_DIR)/vd_british.wav 2>&1 | tee $(TEST_DIR)/vd_british.wav.log; \
	 if [ ! -s $(TEST_DIR)/vd_british.wav ] || ! grep -q "Generated [1-9]" $(TEST_DIR)/vd_british.wav.log; then echo "FAIL: VoiceDesign British male"; exit 1; fi; \
	 echo "PASS: VoiceDesign British male"; \
	 echo "--- VoiceDesign: energetic female ---"; \
	 ./$(TARGET) -d $(MODEL_VOICE_DESIGN) -l English --voice-design \
	   --instruct "Young energetic female, cheerful and fast-paced" \
	   --text "Oh my gosh, this is so exciting!" \
	   -o $(TEST_DIR)/vd_cheerful.wav 2>&1 | tee $(TEST_DIR)/vd_cheerful.wav.log; \
	 if [ ! -s $(TEST_DIR)/vd_cheerful.wav ] || ! grep -q "Generated [1-9]" $(TEST_DIR)/vd_cheerful.wav.log; then echo "FAIL: VoiceDesign energetic female"; exit 1; fi; \
	 echo "PASS: VoiceDesign energetic female"; \
	 echo "=== VoiceDesign test passed ==="

# ── Voice Clone Demo ──
# Uses an existing sample WAV as reference to clone a voice with new text.
# Requires: Base model (download with ./download_model.sh --model base-small)

# Voice Clone Demo
# Usage:
#   make demo-clone                              (uses default sample)
#   make demo-clone REF=my_voice.wav             (use your own audio)
#   make demo-clone REF=my_voice.wav TEXT="Hi!"  (custom text too)
# Output saved to samples/ for easy listening.

REF ?= samples/voice_clone_english.wav
TEXT ?= I love programming in C, it gives you complete control over the machine.
TEXT_IT ?= Buongiorno, questa e una dimostrazione della clonazione vocale.

demo-clone: $(TARGET)
	@echo "=== Voice Clone Demo ==="
	@if [ ! -d $(MODEL_BASE_SMALL) ]; then \
		echo "Error: $(MODEL_BASE_SMALL) not found"; \
		echo "Download it with: ./download_model.sh --model base-small"; \
		exit 1; \
	fi
	@if [ ! -f "$(REF)" ]; then \
		echo "Error: $(REF) not found"; \
		echo "Usage: make demo-clone REF=your_audio.wav"; \
		exit 1; \
	fi
	@mkdir -p samples
	@echo ""
	@echo "Reference audio: $(REF)"
	@echo ""
	@echo "--- Cloning voice (English) ---"
	./$(TARGET) -d $(MODEL_BASE_SMALL) -l English \
		--text "$(TEXT)" \
		--ref-audio "$(REF)" \
		--xvector-only \
		-o samples/clone_output_en.wav
	@echo ""
	@echo "--- Cloning voice (Italian) ---"
	./$(TARGET) -d $(MODEL_BASE_SMALL) -l Italian \
		--text "$(TEXT_IT)" \
		--ref-audio "$(REF)" \
		--xvector-only \
		-o samples/clone_output_it.wav
	@echo ""
	@echo "=== Demo complete ==="
	@echo "Output saved to samples/"
	@echo ""
	@echo "Listen:"
	@echo "  Reference:  afplay $(REF)"
	@echo "  English:    afplay samples/clone_output_en.wav"
	@echo "  Italian:    afplay samples/clone_output_it.wav"

# Legacy aliases
test-en: test-small-en
test-it-ryan: test-small-it

.PHONY: server-hw-check box-report membw check-matmat-parity check-matmat-parity-x86 \
	server-batch-microbench server-batch-microbench-full mini-bench-06b mini-bench-17b \
	kernel-tune kernel-tune-quick test-decoder-batch-parity server-soak
.PHONY: all help blas clean debug info serve cp-microbench batching-bench test-batch test-batch-invariance test-errors test-emotion test-emotion-ft emotion-demo emo-suite emotion-seeds test-compose test-caps test-selftest test-golden golden-update emovoice emo-06b-demo quant-ladder test-modes test-qvoice e2e \
        emotion-para-demo para-demo \
        test-serve test-serve-bench test-serve-repro test-serve-openai test-serve-parallel test-serve-concurrent test-serve-batch test-serve-continuous test-serve-stream-batch test-stage-policy test-serve-all \
        test-clone test-voice-design \
        demo-clone \
        test-small test-small-en test-small-it test-small-vivian test-small-stream test-small-stdout \
        test-large test-large-en test-large-it test-large-config test-large-instruct \
        test-large-int8 test-large-int4 test-large-quant \
        test-regression test-all test-en test-it-ryan
