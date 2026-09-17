
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
CC = gcc

SIMD ?= auto

ifeq ($(SIMD),auto)
    ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
        ifeq ($(UNAME_S),Linux)
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
            $(info [simd] auto -> $(SIMD)   (make blas SIMD=portable for a binary not tied to this host))
        else
            SIMD := portable
        endif
    else
        SIMD := native
    endif
endif
ifeq ($(UNAME_S),Darwin)
    ARCH_FLAGS = -march=native
else ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
    ifeq ($(SIMD),scalar)
        ARCH_FLAGS =
    else ifeq ($(SIMD),amx)
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 \
                     -mamx-tile -mamx-int8 -mamx-bf16 -mavx2 -mfma
    else ifeq ($(SIMD),avx512bf16)
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mavx2 -mfma
    else ifeq ($(SIMD),avx512vnni)
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx2 -mfma
    else ifeq ($(SIMD),avx512)
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx2 -mfma
    else
        ARCH_FLAGS = -mavx2 -mfma
    endif
else
    ARCH_FLAGS = -march=native
endif

# Source identity, computed by tools/source_fingerprint.sh where git is and read from the
# shipped .source_fingerprint on a box without it.  It goes into a GENERATED header
# (qwen_build_id.h) that is rewritten only when its content changes, so the objects that
# embed it are rebuilt exactly when the identity changes — a -D flag is invisible to make
# and used to leave a stale build tag inside a fresh binary.
SRC_FP  := $(shell bash tools/source_fingerprint.sh 2>/dev/null || echo unknown)
GIT_REV ?= $(firstword $(subst :, ,$(SRC_FP)))
CFLAGS_BASE = -Wall -Wextra -O3 $(ARCH_FLAGS) -ffast-math -DQWEN_SIMD_PROFILE=\"$(SIMD)\"
LDLIBS = -lm -lpthread

CFLAGS_BASE += -Ivendor

ifeq ($(UNAME_S),Darwin)
    CFLAGS_BASE += -DUSE_BLAS -DACCELERATE_NEW_LAPACK
    LDLIBS += -framework Accelerate
else
    CFLAGS_BASE += -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
    LDLIBS += -lopenblas
endif

CFLAGS = $(CFLAGS_BASE) -I$(INGOT_DIR)/include $(KAI_INC) $(EXTRA_CFLAGS)

# Compile flags are invisible to make: `make blas` followed by
# `make blas ARCH_FLAGS="-march=armv8-a"` reused every object and produced a MIXED binary
# (portable kernels with a native dispatch, observed 2026-09-10).  Keep the effective flags
# in a stamp file rewritten ONLY when they change; every compiled object depends on it, so a
# flag change forces a full recompile and an unchanged build stays a no-op.  Changing SIMD
# alone is the same trap because it also changes -DQWEN_SIMD_PROFILE.
#
# The stamp is a real target so the 1 s settle runs only when the value changes: GNU make
# 3.81 (the one Apple ships) compares mtimes in whole seconds, and a stamp written in the
# same second as the last object would silently leave that object stale.
ARCH_STAMP_DIR   := .build_state
ARCH_STAMP       := $(ARCH_STAMP_DIR)/compile_flags
ARCH_STAMP_VALUE := SIMD=$(SIMD) ARCH_FLAGS=$(ARCH_FLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS)

$(ARCH_STAMP): FORCE
	@mkdir -p $(ARCH_STAMP_DIR)
	@printf '%s\n' '$(ARCH_STAMP_VALUE)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv $@.tmp $@; sleep 1; fi

KAI_HAS_I8MM := $(shell $(CC) $(ARCH_FLAGS) -dM -E -x c /dev/null 2>/dev/null | grep -c __ARM_FEATURE_MATMUL_INT8)
ifeq ($(KAI_HAS_I8MM),1)
KAI_DIR  = third_party/kleidiai
KAI_SRCS = $(KAI_DIR)/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c \
           $(KAI_DIR)/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c \
           $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c \
           $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
KAI_ASM  = $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod_asm.S \
           $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm_asm.S

KAI_I8_DIR = $(KAI_DIR)/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp
KAI_SRCS += $(KAI_DIR)/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c \
            $(KAI_DIR)/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c
KAI_ASM  += $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_asm.S \
            $(KAI_I8_DIR)/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_asm.S

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
       qwen_tts_dispatch.c \
       qwen_tts_costmap.c \
       qwen_tts_sd_gemm.c \
       qwen_tts_thread.c \
       qwen_tts_kernels_generic.c \
       qwen_tts_kernels_neon.c \
       qwen_tts_kernels_avx.c \
       qwen_tts_audio.c \
       qwen_tts_emotion.c \
       qwen_tts_compose.c \
       qwen_tts_sampling.c \
       qwen_tts_tokenizer.c \
       qwen_json.c \
       qwen_tts_server.c \
       qwen_tts_voice_clone.c \
       qwen_tts_speech_encoder.c \
       vendor/lz4.c

OBJS = $(SRCS:.c=.o) $(KAI_AOBJ)

%_asm.o: %_asm.S $(ARCH_STAMP)
	$(CC) $(CFLAGS) -c -o $@ $<
TARGET = qwen_tts

qwen_build_id.h: FORCE
	@printf '#define QWEN_BUILD_GIT_REV "%s"\n#define QWEN_BUILD_SOURCE_FP "%s"\n#define QWEN_BUILD_SIMD "%s"\n' \
	  "$(GIT_REV)" "$(SRC_FP)" "$(SIMD)" > qwen_build_id.h.tmp; \
	if cmp -s qwen_build_id.h.tmp qwen_build_id.h 2>/dev/null; then rm -f qwen_build_id.h.tmp; else mv qwen_build_id.h.tmp qwen_build_id.h; fi
FORCE:
.PHONY: FORCE
qwen_tts_kernels.o qwen_tts_dispatch.o: qwen_build_id.h

INGOT_DIR := third_party/ingot
INGOT_LIB := $(INGOT_DIR)/libingot.a
$(INGOT_LIB):
	$(MAKE) -C $(INGOT_DIR) lib

update-ingot:
	git subtree pull --prefix $(INGOT_DIR) https://github.com/mynah-org/ingot.git main --squash

clean-ingot:
	@$(MAKE) -C $(INGOT_DIR) clean

.PHONY: update-ingot clean-ingot

$(TARGET): $(OBJS) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(INGOT_LIB) $(LDLIBS)

blas: $(TARGET)

MODEL_DIR = qwen3-tts-0.6b

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
	@echo "  make emotion-seeds   - Seed-finder palette → samples/emotion_seeds.md (local artifact; OUT_MD=path to override; SLOW)"
	@echo "  make test-clone      - Voice clone e2e (generate ref → clone → stream)"
	@echo "  make demo-clone      - Voice clone demo using sample WAV"
	@echo "  make test-regression - Cross-model regression checks"
	@echo "  make test-stream-layout - Known-text dual-track layout smoke (default-off)"
	@echo "  make test-all        - Run everything (0.6B + 1.7B + regression)"
	@echo ""
	@echo "Benchmark:"
	@echo "  make bench           - RTF benchmark (short+long, normal+stream)"
	@echo "  make bench-full      - Full benchmark (+ server, qvoice, instruct, INT8)"
	@echo "  make cp-microbench   - Build qwen_tts_cpbench (per-op Code Predictor breakdown)"
	@echo ""
	@echo "CPU profiling gate (docs/cpu-profiling.md) — run BEFORE any CPU optimisation:"
	@echo "  make cpu-check             - 15 s preflight: provenance, hardware, RESOLVED dispatch map, self-test,"
	@echo "                               expected-vs-observed per ISA class -> profiles/<date>_<host>_<sha8>/"
	@echo "  make doctor                - first serving preflight: <1 min, no model; on 32-core Arm also"
	@echo "                               simultaneous 1x8/2x8/4x8 GEMV scaling, then identity + dispatch + draft profile"
	@echo "  make dispatch-map          - just the resolved dispatch table (./qwen_tts --dispatch-map)"
	@echo "  make profile-cpu-check     - is the last profile still valid for THIS binary/source/env/host?"
	@echo "  make cost-map              - coarse cost map: where the wall time goes AROUND the kernels."
	@echo "                               Runs census-only vs census+cost-map with the SAME binary and"
	@echo "                               proves the instrumentation moved no executed path."
	@echo "                               COSTMAP_LEVEL=1 macro, =2 adds per-layer CP + prefill layout."
	@echo "  make cost-map-overhead     - what the cost map costs, A/B/B/A interleaved (box drift)"
	@echo "  make roofs ROOF_MASKS=0-15,0-7,8-15"
	@echo "                             - bandwidth roof PER CPU MASK (read/copy/triad + saturation"
	@echo "                               curve, t90/t95/t99), cached per hardware fingerprint."
	@echo "                               A worker roof is measured, never divided from the host."
	@echo "  make topology-report       - execution domain + roofs + scope-checked comparison"
	@echo "  make envelope WAVE_JSON=…  - scaling envelope + cliffs from a canonical wave run"
	@echo ""
	@echo "A newly provisioned box (IN THIS ORDER — see docs/hardware-testing.md):"
	@echo "  make server-hw-check       - the truth about the silicon: hardware + memory bandwidth"
	@echo "                               (tools/box_info.sh + tests/membw.c) + --caps + --self-test"
	@echo "                               + --matmat-bench. No model needed. JSON in HW_JSON=."
	@echo "                               (historical alias: make box-report)"
	@echo "  make server-batch-microbench - server request-batching: M clients vs single-stream, TTFB+RTF (CustomVoice 0.6B; -full = 1.7B)"
	@echo "  make mini-bench-06b|-17b   - 1/2/4 simultaneous requests on an OSS model, int8, compiled defaults"
	@echo "  make kernel-tune[-quick]   - measure the dispatcher thresholds instead of guessing them (KT_ARGS=)"
	@echo "  make tune-archive BOX=<name> - the same, and archives the JSON in docs/boxes/ for tests/mm_tune_compare.py"
	@echo "  make membw                 - bandwidth only: Copy/Triad with a thread sweep, and the knee"
	@echo "  make bench-matrix[-full]   - then the RTF matrix (needs a downloaded model)"
	@echo "  make check-matmat-parity   - do the batched twins do the arithmetic they claim? (native ISA)"
	@echo "  make check-flag-registry   - every runtime flag the engine reads is one it declares in [FLAGS]"
	@echo "  make prefill-bench         - where prefill time goes: packing, per-call cost, weight stream"
	@echo "  make check-matmat-parity-x86 - the same on the x86 AVX2 kernels, run under Rosetta 2 from an Arm Mac"
	@echo "  make check-isa             - compile-check the ISA paths this machine does not have"
	@echo "  make test-decoder-tool - Build qwen_tts_decoder_tool (decode a QWEN_DUMP_CODES dump alone)"
	@echo ""
	@echo "Serving qualification (docs/serving/cpu-operations.md):"
	@echo "  make bench-fingerprint     - what this machine actually is: cpu, cores, SMT, cache, NUMA, measured bandwidth"
	@echo "  make bench-topo            - sweep prefork topologies to find W x K (BENCH_TOPO=1x16,2x8,4x4)"
	@echo "  make bench-suite           - the qualification suite: preflight gates, rungs, audio length, manifest"
	@echo "                               (BENCH_MODEL= BENCH_PROFILE= BENCH_TOPO= BENCH_RUNG=fast BENCH_OUT=)"
	@echo "  make bench-soak            - closed-loop soak with rolling/per-class KPI analysis"
	@echo "                               (SOAK_MODEL= SOAK_PROFILE= SOAK_CONCURRENCY= SOAK_MINUTES= SOAK_MIN_PER_CLASS_P95= SOAK_OUT=)"
	@echo "  make bench-suite-full      - qualification suite followed by the closed-loop soak"
	@echo ""
	@echo "Example: make blas && ./$(TARGET) -d $(MODEL_DIR) -t \"Hello world\" -o output.wav"

debug: CFLAGS = $(CFLAGS_BASE) -I$(INGOT_DIR)/include $(KAI_INC) -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
debug: LDLIBS += -fsanitize=address -fsanitize=undefined
debug: clean $(TARGET)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "CC:       $(CC)"
	@echo "CFLAGS:   $(CFLAGS)"
	@echo "LDLIBS:   $(LDLIBS)"
	@echo "SRCS:     $(SRCS)"
	@echo "TARGET:   $(TARGET)"

GPU_OBJS = qwen_tts_backend.o qwen_tts_cuda.o

CUDA_HOME ?= $(shell \
	if command -v nvcc >/dev/null 2>&1; then dirname "$$(dirname "$$(command -v nvcc)")"; \
	elif [ -x /usr/local/cuda/bin/nvcc ]; then echo /usr/local/cuda; \
	elif [ -x /opt/cuda/bin/nvcc ]; then echo /opt/cuda; \
	else echo /usr/local/cuda; fi)
CUDA_LIBDIR ?= $(shell \
	if [ -d "$(CUDA_HOME)/lib64" ]; then echo "$(CUDA_HOME)/lib64"; \
	else echo "$(CUDA_HOME)/lib"; fi)

.PHONY: metal cuda metal_build cuda_build

metal:
	$(MAKE) clean
	$(MAKE) metal_build
metal_build: EXTRA_CFLAGS += -DQWEN_HAVE_METAL
metal_build: $(OBJS) $(GPU_OBJS) qwen_tts_metal.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(GPU_OBJS) qwen_tts_metal.o $(INGOT_LIB) $(LDLIBS) \
		-framework Metal -framework Foundation
	@echo ""
	@echo "Built ./$(TARGET) with Metal backend. Try: ./$(TARGET) --gpu-selftest --backend metal"

qwen_tts_metal.o: qwen_tts_metal.m
	clang -fobjc-arc -O3 -Wall -Wextra -Ivendor -MMD -MP -c -o $@ $<

NVCC ?= $(CUDA_HOME)/bin/nvcc
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
	@echo ""
	@echo "Built ./$(TARGET) with CUDA backend. Try: ./$(TARGET) --gpu-selftest --backend cuda"

qwen_tts_cuda_kernels.o: qwen_tts_cuda_kernels.cu
	$(NVCC) $(NVCC_ARCH) -O3 -c -o $@ $<

qwen_tts_cuda_talker.o: qwen_tts_cuda_talker.cu qwen_tts.h qwen_tts_kernels.h
	$(NVCC) $(NVCC_ARCH) -O3 --default-stream per-thread -I. -I$(CUDA_HOME)/include -c -o $@ $<

qwen_tts_cuda_decoder.o: qwen_tts_cuda_decoder.cu qwen_tts.h qwen_tts_kernels.h
	$(NVCC) $(NVCC_ARCH) -O3 --default-stream per-thread -I. -I$(CUDA_HOME)/include -c -o $@ $<

cp-microbench:
	$(MAKE) clean
	$(MAKE) TARGET=qwen_tts_cpbench EXTRA_CFLAGS=-DCP_MICROBENCH qwen_tts_cpbench
	@echo ""
	@echo "Built ./qwen_tts_cpbench  (run a normal generation; CP breakdown prints in the summary)"

test-decoder-tool: $(filter-out main.o,$(OBJS)) test_decoder_standalone.o
	$(CC) $(CFLAGS) -o qwen_tts_decoder_tool $^ $(LDLIBS)
	@echo "Built ./qwen_tts_decoder_tool  (usage: ./qwen_tts_decoder_tool codes.txt [model_dir] [out.wav])"

%.o: %.c $(ARCH_STAMP)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

qwen_tts_speech_encoder.o: qwen_tts_speech_encoder.c $(ARCH_STAMP)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -MMD -MP -c -o $@ $<

-include $(OBJS:.o=.d) qwen_tts_backend.d qwen_tts_cuda.d qwen_tts_metal.d

clean:
	@rm -f qwen_build_id.h qwen_build_id.h.tmp
	rm -rf $(ARCH_STAMP_DIR)
	rm -f $(OBJS) $(OBJS:.o=.d) $(TARGET) qwen_tts_backend.o qwen_tts_cuda.o qwen_tts_metal.o qwen_tts_cuda_kernels.o
	rm -f qwen_tts_backend.d qwen_tts_cuda.d qwen_tts_metal.d vendor/lz4.d
	rm -f test_decoder_standalone.o test_decoder_standalone.d qwen_tts_decoder_tool
	rm -f tests/decode_quantum_bench.o tests/decode_quantum_bench.d qwen_tts_decode_quantum

bench-server: $(TARGET)
	@bash tests/serve_batch_bench.sh $(MODEL_SMALL)

check-isa:
	@bash tools/check_isa.sh

emotion-para-demo: $(TARGET)
	@bash tests/emotion_para_demo.sh

matmat-bench: $(TARGET)
	@echo "=== Batched matmat twins vs B*matvec (real kernels, 4 threads) ==="
	@./$(TARGET) --matmat-bench
	@echo "=== (single thread = compute-bound reference) ==="
	@./$(TARGET) --matmat-bench -j 1

test-selftest: $(TARGET)
	@echo "=== Kernel self-test (dispatched path) ==="
	@./$(TARGET) --self-test || { echo "FAIL: kernel self-test (dispatched)"; exit 1; }
	@echo "=== Kernel self-test (scalar/widen fallback: QWEN_NO_SDOT=1 QWEN_NO_VNNI=1) ==="
	@QWEN_NO_SDOT=1 QWEN_NO_VNNI=1 ./$(TARGET) --self-test || { echo "FAIL: kernel self-test (fallback)"; exit 1; }
	@echo "PASS: kernel self-test (both paths numerically correct)"
	@echo ""

test-json-parser:
	@$(CC) $(CFLAGS) -I. tests/json_string_parser_test.c qwen_json.c -o /tmp/qwen_json_string_parser_test
	@/tmp/qwen_json_string_parser_test

MODEL_SMALL = qwen3-tts-0.6b
MODEL_LARGE = qwen3-tts-1.7b
MODEL_BASE_SMALL = qwen3-tts-0.6b-base
MODEL_VOICE_DESIGN = qwen3-tts-voice-design
TEST_DIR = /tmp/qwen_tts_tests

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
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian --emotion joy \
		--text "Ciao." -o $(TEST_DIR)/em_06b.wav 2>/dev/null; \
	 test -s $(TEST_DIR)/em_06b.wav || { echo "FAIL: 0.6B --emotion produced no audio"; exit 1; }
	@echo "  PASS: 0.6B --emotion parked-neutral (no crash, audio written)"
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian --volume 1.2 --rate 0.9 \
		--text "Ciao." -o $(TEST_DIR)/em_vr.wav 2>$(TEST_DIR)/em_vr.log
	@grep -qi "Volume: 1.20" $(TEST_DIR)/em_vr.log && grep -qi "Rate: 0.90" $(TEST_DIR)/em_vr.log || { echo "FAIL: standalone --volume/--rate not applied"; cat $(TEST_DIR)/em_vr.log; exit 1; }
	@echo "  PASS: standalone --volume/--rate"
	@echo "PASS: expressivity/emotion smoke"
	@echo ""

EXPR_FT ?= presets/expr/italian_csp_topk6.expr
EMO_FT_INSTR = Speak with warm, bright happiness, smiling through the words.
EMO_FT_TEXT  = Che bella notizia, sono davvero felicissimo oggi!
test-emotion-ft: $(TARGET)
	@echo "=== Emotion fine-tune (.expr graft) smoke — CSP Italian on 1.7B ==="
	@mkdir -p $(TEST_DIR)
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

emotion-demo: $(TARGET)
	@bash tests/emotion_demo.sh

emovoice: $(TARGET)
	@bash tests/emovoice_build.sh

emo-06b-demo: $(TARGET)
	@bash tests/emo_06b_demo.sh

para-demo: $(TARGET)
	@bash tests/para_demo.sh

emo-suite: $(TARGET)
	@bash tests/emo_suite.sh

EXPR ?= presets/expr/italian_l1626_r64.expr
test-lora-it: $(TARGET)
	@bash tests/lora_matrix.sh Italian $(EXPR)

# The palette is a GENERATED artifact, and the doc it used to overwrite was archived as
# superseded (docs/archive/emotion-seeds.md: k4-era, contradicts the shipped k6/T1.1/steer
# recipe).  Writing the regenerated palette back into docs/ would resurrect a superseded
# document in the canonical tree, so it lands beside the audio palette it describes, under
# samples/ where the rest of that output is already gitignored.  OUT_MD overrides.
emotion-seeds: $(TARGET)
	@bash tests/emotion_seed_finder.sh $(if $(OUT_MD),$(OUT_MD),samples/emotion_seeds.md) $(if $(N),$(N),5)

test-batch-invariance: $(TARGET)
	@echo "=== 1/2 cablaggio: percorsi pinnati, identita' bit a bit dovuta ==="
	@MODEL=$(MODEL_SMALL) EP=/v1/tts PIN_ENV="env QWEN_BATCH_NO_SOLO=1 QWEN_BATCH_NOMATMUL=1" CRITERION=byte \
	  bash tests/serve_batch_invariance.sh
	@echo "=== 2/2 real paths: same DURATION is required, not the same bytes ==="
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

HW_JSON ?= /tmp/tts/box_info.json
MEMBW_BIN ?= /tmp/qwen_membw

$(MEMBW_BIN): tests/membw.c
	@$(CC) -Wall -Wextra -O2 -o $@ tests/membw.c -lpthread -lm

MINI_ARMS ?= --int8 --int4 --quant-mixed-int6=q4n14

tests/decoder_batch_parity.o: tests/decoder_batch_parity.c $(ARCH_STAMP)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

test-decoder-batch-parity: $(filter-out main.o,$(OBJS)) tests/decoder_batch_parity.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o qwen_tts_batch_parity $^ $(LDLIBS)
	@echo "--- ragged schedule WITH one-frame chunks (the BLAS GEMV path is in play) ---"
	@./qwen_tts_batch_parity $(MODEL_SMALL) 4 6
	@echo "--- same schedule, without one-frame chunks (expected: bit-identical) ---"
	@QWEN_PARITY_PAT=2 ./qwen_tts_batch_parity $(MODEL_SMALL) 4 6

tests/decode_quantum_bench.o: tests/decode_quantum_bench.c $(ARCH_STAMP)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

test-decode-quantum: $(filter-out main.o,$(OBJS)) tests/decode_quantum_bench.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o qwen_tts_decode_quantum $^ $(LDLIBS)
	@echo "Built ./qwen_tts_decode_quantum  (usage: ./qwen_tts_decode_quantum <model_dir> [threads] [reps])"

kernel-census: $(filter-out main.o,$(OBJS)) tests/kernel_census_bench.o $(INGOT_LIB)
	$(CC) $(CFLAGS) -o qwen_kernel_census $^ $(LDLIBS)
	@./qwen_kernel_census --model $(or $(CMODEL),1.7b) $(CENSUS_ARGS)

X86_QKV_BENCH_SRC = tests/x86_qkv_bench.c qwen_tts_kernels.c qwen_tts_thread.c \
                    qwen_tts_kleidi.c qwen_tts_q8repack.c $(KAI_SRCS) $(KAI_ASM)
x86-qkv-bench: $(INGOT_LIB)
ifeq ($(UNAME_S),Darwin)
	@clang $(PARITY_CF) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -march=native \
	  $(X86_QKV_BENCH_SRC) $(INGOT_LIB) -framework Accelerate -lm -o /tmp/x86_qkv_bench
else
	@$(CC) $(PARITY_CF) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas $(ARCH_FLAGS) \
	  $(X86_QKV_BENCH_SRC) $(INGOT_LIB) -lopenblas -lm -lpthread -o /tmp/x86_qkv_bench
endif
	@/tmp/x86_qkv_bench $(X86_QKV_THREADS) $(X86_QKV_REPS)

# Track A / B=1 GEMV A/B harness.  Built with the PRODUCTION optimisation flags
# (-O3 -ffast-math) so the numbers are comparable with the served engine.
X86_B1_BENCH_CF = -Wall -Wextra -O3 -ffast-math -Ivendor -I. -I$(INGOT_DIR)/include $(KAI_INC)
X86_B1_GEMV_BENCH_SRC = tests/x86_b1_gemv_bench.c qwen_tts_kernels.c qwen_tts_thread.c \
                        qwen_tts_kleidi.c qwen_tts_q8repack.c $(KAI_SRCS) $(KAI_ASM)
x86-b1-gemv-bench: $(INGOT_LIB)
	@$(CC) $(X86_B1_BENCH_CF) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas $(ARCH_FLAGS) \
	  $(X86_B1_GEMV_BENCH_SRC) $(INGOT_LIB) -lopenblas -lm -lpthread -o /tmp/x86_b1_gemv_bench
	@/tmp/x86_b1_gemv_bench $(X86_B1_THREADS) $(X86_B1_REPS)

X86_AMX_B32_BENCH_SRC = tests/x86_amx_b32_bench.c qwen_tts_kernels.c qwen_tts_thread.c \
                        qwen_tts_kleidi.c qwen_tts_q8repack.c $(KAI_SRCS) $(KAI_ASM)
x86-amx-b32-bench: $(INGOT_LIB)
ifeq ($(UNAME_S),Darwin)
	@clang $(PARITY_CF) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -march=native \
	  $(X86_AMX_B32_BENCH_SRC) $(INGOT_LIB) -framework Accelerate -lm -o /tmp/x86_amx_b32_bench
else
	@$(CC) $(PARITY_CF) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas $(ARCH_FLAGS) \
	  $(X86_AMX_B32_BENCH_SRC) $(INGOT_LIB) -lopenblas -lm -lpthread -o /tmp/x86_amx_b32_bench
endif
	@QWEN_AMX_B32=1 /tmp/x86_amx_b32_bench

BOX ?= $(shell date +%Y-%m-%d)_$(shell uname -m)-$(shell hostname | tr -cd 'a-zA-Z0-9-')

server-hw-check: $(TARGET) $(MEMBW_BIN)
	@MEMBW_BIN=$(MEMBW_BIN) HW_JSON=$(HW_JSON) bash tests/bench_matrix.sh --silicon-only

box-report: server-hw-check

membw: $(MEMBW_BIN)
	@$(MEMBW_BIN)

bench-matrix: $(TARGET)
	@bash tests/bench_matrix.sh $(MODEL_SMALL)
bench-matrix-full: $(TARGET)
	@bash tests/bench_matrix.sh $(MODEL_SMALL) --full

BENCH_MODEL   ?= $(MODEL_LARGE)
BENCH_PROFILE ?= recommended
BENCH_TOPO    ?= 2x8
BENCH_SPEAKER ?= ryan
BENCH_BANK    ?= tests/load_texts_en.txt
BENCH_CONC    ?= 1,4
BENCH_WAVES   ?= 3
BENCH_OUT     ?= /tmp/bench_suite
BENCH_RUNG    ?=
BENCH_ARGS    ?=

bench-fingerprint: $(MEMBW_BIN)
	@MEMBW_BIN=$(MEMBW_BIN) bash tools/box_info.sh

bench-topo: $(TARGET)
	@python3 tests/serve_parallel_wave.py --model $(BENCH_MODEL) --bin ./$(TARGET) \
	  --speaker $(BENCH_SPEAKER) --topo $(BENCH_TOPO) --conc $(BENCH_CONC) \
	  --waves $(BENCH_WAVES) --seed 42 --precision int8 --profile $(BENCH_PROFILE) \
	  --text-file $(BENCH_BANK) --classes short --out $(BENCH_OUT)/topo \
	  --port 9500 --label topo $(BENCH_ARGS)

bench-suite: $(TARGET)
	@bash tests/bench_suite.sh --model $(BENCH_MODEL) --profile $(BENCH_PROFILE) \
	  --topo $(BENCH_TOPO) --speaker $(BENCH_SPEAKER) --bank-fast $(BENCH_BANK) \
	  --bank-real $(BENCH_BANK) --out $(BENCH_OUT) \
	  $(if $(BENCH_RUNG),--only $(BENCH_RUNG),) $(BENCH_ARGS)

SOAK_MODEL        ?= $(BENCH_MODEL)
SOAK_PROFILE      ?= $(BENCH_PROFILE)
SOAK_NO_PROFILE   ?= exploratory soak with compiled defaults
SOAK_BANK        ?= $(BENCH_BANK)
SOAK_SPEAKER     ?= $(BENCH_SPEAKER)
SOAK_LANGUAGE    ?= English
SOAK_CONCURRENCY ?= 2
SOAK_MINUTES     ?= 10
SOAK_TEMP        ?= 0.9
SOAK_PRECISION   ?= int8
SOAK_MIN_PER_CLASS_P95 ?= 20
SOAK_OUT         ?= $(BENCH_OUT)/soak
SOAK_ARGS        ?=

bench-soak: $(TARGET)
	@python3 tests/serve_soak.py --model "$(SOAK_MODEL)" --bin "./$(TARGET)" \
	  --bank "$(SOAK_BANK)" --speaker "$(SOAK_SPEAKER)" --language "$(SOAK_LANGUAGE)" \
	  --concurrency "$(SOAK_CONCURRENCY)" --minutes "$(SOAK_MINUTES)" \
	  --min-per-class-p95 "$(SOAK_MIN_PER_CLASS_P95)" \
	  --temperature "$(SOAK_TEMP)" --precision "$(SOAK_PRECISION)" --out "$(SOAK_OUT)" \
	  $(if $(SOAK_PROFILE),--profile "$(SOAK_PROFILE)",--no-profile "$(SOAK_NO_PROFILE)") \
	  $(SOAK_ARGS)

server-soak: bench-soak

# SOAK-FAST: adaptive short SCREEN that finds the playback knee in minutes, so the
# 30-minute canonical soak is only ever spent on a point that already looks like a winner.
# A screen is never a qualification -- see the header of tests/soak_fast.py.
SOAKFAST_LADDER   ?= 2:8
SOAKFAST_OUT      ?= $(BENCH_OUT)/soak-fast
SOAKFAST_WARMUP_S ?= 30
SOAKFAST_WINDOW_S ?= 90
SOAKFAST_WINDOWS  ?= 2
SOAKFAST_ARGS     ?=

soak-fast: $(TARGET)
	@python3 tests/soak_fast.py --model "$(SOAK_MODEL)" --bin "./$(TARGET)" \
	  --bank "$(SOAK_BANK)" --speaker "$(SOAK_SPEAKER)" --language "$(SOAK_LANGUAGE)" \
	  --ladder "$(SOAKFAST_LADDER)" --precision "$(SOAK_PRECISION)" \
	  --warmup-s "$(SOAKFAST_WARMUP_S)" --window-s "$(SOAKFAST_WINDOW_S)" \
	  --min-windows "$(SOAKFAST_WINDOWS)" --out "$(SOAKFAST_OUT)" \
	  $(if $(SOAK_PROFILE),--profile "$(SOAK_PROFILE)",--no-profile "$(SOAK_NO_PROFILE)") \
	  $(SOAKFAST_ARGS)

bench-suite-full: bench-suite
	@$(MAKE) bench-soak

# qwen_tts_costmap.c is not optional: qwen_tts_thread.c calls qwen_region_begin_/end_, so
# leaving it out breaks the link, not the measurement.
PARITY_SRC = tests/matmat_parity.c qwen_tts_kernels.c qwen_tts_thread.c \
             qwen_tts_costmap.c qwen_tts_kleidi.c qwen_tts_q8repack.c $(KAI_SRCS) $(KAI_ASM)
PARITY_CF  = -Wall -Wextra -O2 -Ivendor -I. -I$(INGOT_DIR)/include $(KAI_INC)
check-matmat-parity: $(INGOT_LIB)
	@echo "=== matmat parity — native ISA ==="
ifeq ($(UNAME_S),Darwin)
	@clang $(PARITY_CF) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -march=native \
	  $(PARITY_SRC) $(INGOT_LIB) -framework Accelerate -lm -o /tmp/matmat_parity
else
	@$(CC) $(PARITY_CF) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas $(ARCH_FLAGS) \
	  $(PARITY_SRC) $(INGOT_LIB) -lopenblas -lm -lpthread -o /tmp/matmat_parity
endif
	@/tmp/matmat_parity

check-matmat-parity-x86:
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
	@echo "=== matmat parity — x86-64-v3 (AVX2) under Rosetta 2 ==="
	@clang -target x86_64-apple-macos13 $(PARITY_CF) -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
	  -march=x86-64-v3 $(PARITY_SRC) -framework Accelerate -lm -o /tmp/matmat_parity_x86
	@/tmp/matmat_parity_x86
else
	@echo "check-matmat-parity-x86: Arm Mac only (needs Rosetta 2). On x86 use check-matmat-parity."
endif

test-compose: $(TARGET)
	@echo "=== Inline markup / --compose smoke test ==="
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian \
		--text "Che bella notizia! [pause:400ms] [sad] Devo andare... [sigh] [neutral] Ciao." \
		-o $(TEST_DIR)/mk_inline.wav 2>$(TEST_DIR)/mk_inline.log
	@grep -qi "Inline markup detected" $(TEST_DIR)/mk_inline.log || { echo "FAIL: inline markup not auto-detected in --text"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@grep -qi "composed 3 spans" $(TEST_DIR)/mk_inline.log || { echo "FAIL: expected 3 spans (neutral/sad+[sigh]/neutral)"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@grep -qi "inline \[tag\]->onomatopoeia" $(TEST_DIR)/mk_inline.log || { echo "FAIL: [sigh] not folded inline as onomatopoeia"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@grep -qi "pause 0.40s" $(TEST_DIR)/mk_inline.log || { echo "FAIL: [pause:400ms] not parsed"; cat $(TEST_DIR)/mk_inline.log; exit 1; }
	@test -s $(TEST_DIR)/mk_inline.wav || { echo "FAIL: no audio"; exit 1; }
	@echo "  PASS: inline [tag] markup in --text (3 spans, pause, [sigh] folded inline)"
	@./$(TARGET) -d $(MODEL_SMALL) -j1 -T 0 --seed 42 -s ryan -l Italian \
		--text "Frase normale senza tag." -o $(TEST_DIR)/mk_plain.wav 2>$(TEST_DIR)/mk_plain.log
	@if grep -qi "compose mode" $(TEST_DIR)/mk_plain.log; then echo "FAIL: plain text wrongly routed to compose"; cat $(TEST_DIR)/mk_plain.log; exit 1; fi
	@echo "  PASS: plain text stays on the normal path"
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
	@if ! grep -q "hidden=1024" $(TEST_DIR)/cfg_small.txt; then echo "FAIL: 0.6B should have hidden=1024"; exit 1; fi
	@if ! grep -q "hidden=2048" $(TEST_DIR)/cfg_large.txt; then echo "FAIL: 1.7B should have hidden=2048"; exit 1; fi
	@if ! grep -q "head_dim=128" $(TEST_DIR)/cfg_small.txt; then echo "FAIL: 0.6B head_dim"; exit 1; fi
	@if ! grep -q "head_dim=128" $(TEST_DIR)/cfg_large.txt; then echo "FAIL: 1.7B head_dim"; exit 1; fi
	@echo "PASS: config divergence correct"
	@echo ""
	@echo "=== All regression tests passed ==="

check-flag-registry:
	@python3 tools/check_flag_registry.py

# ── profile-cpu, V1 step 1: what the engine ACTUALLY executed ─────────────────
# cpu-check (fresh dir) + one clean serving run + the same run with the shape census on
# (stable path ids, kernel + leaf attribution, JSON per process) -> call map, coverage,
# UNKNOWN/fallback count joined with the dispatch map, and the census overhead vs clean.
PROFILE_MODEL   ?= qwen3-tts-1.7b-base
PROFILE_PROFILE ?= recommended
PROFILE_TOPO    ?= 2x8
PROFILE_CONC    ?= 1,4
PROFILE_WAVES   ?= 3
PROFILE_CLASSES ?= short
PROFILE_BANK    ?= tests/load_texts_en.txt
profile-cpu: $(TARGET) $(MEMBW_BIN)
	@MEMBW_BIN=$(MEMBW_BIN) PROFILES_DIR=$(PROFILES_DIR) CPU_PROFILE=$(PROFILE_PROFILE) CPU_MODEL=$(PROFILE_MODEL) \
	  PROFILE_TOPO=$(PROFILE_TOPO) PROFILE_CONC=$(PROFILE_CONC) PROFILE_WAVES=$(PROFILE_WAVES) \
	  PROFILE_CLASSES=$(PROFILE_CLASSES) PROFILE_BANK=$(PROFILE_BANK) bash tools/profile_cpu.sh

# ── cost-map, V1 step 2: where the wall time goes AROUND the kernels ──────────
# Coarse semantic regions (inclusive, thread-local, merged at dump).  `cost-map` runs
# the workload twice with the same binary — census only, then census + cost map — so
# the report always ships with the proof that the instrumentation did not move the
# executed paths.  `cost-map-overhead` measures what it costs, A/B/B/A interleaved
# because one clean run followed by one instrumented run cannot outvote box drift.
COSTMAP_LEVEL ?= 1
COSTMAP_CONC  ?= 1
cost-map: $(TARGET)
	@bash tools/costmap_parity.sh --model $(PROFILE_MODEL) --profile $(PROFILE_PROFILE) \
	  --conc $(COSTMAP_CONC) --level $(COSTMAP_LEVEL) --waves $(PROFILE_WAVES) --topo $(PROFILE_TOPO)
cost-map-overhead: $(TARGET)
	@bash tools/costmap_ab.sh --model $(PROFILE_MODEL) --profile $(PROFILE_PROFILE) \
	  --conc $(COSTMAP_CONC) --level $(COSTMAP_LEVEL) --waves $(PROFILE_WAVES) --topo $(PROFILE_TOPO)

# ── topology-aware roofs (B1-B6) ──────────────────────────────────────────────
# A bandwidth roof belongs to ONE cpu mask.  On a multi-CCX part the host roof and a
# pinned worker's roof differ by ~2x, and a worker roof can never be derived by dividing
# a host roof -- on any architecture.  `roofs` measures each mask separately and caches
# them per hardware fingerprint (a roof does not depend on the model, so a model switch
# must not force a re-measurement).  ROOF_MASKS lists the execution domains to qualify.
ROOF_MASKS ?=
ROOF_STORE ?= $(PROFILES_DIR)/roofs
roofs: $(MEMBW_BIN)
	@python3 tools/roofs.py measure --membw $(MEMBW_BIN) --store $(ROOF_STORE) \
	  $(if $(ROOF_MASKS),--masks $(ROOF_MASKS),) $(if $(HW_JSON),--hardware $(HW_JSON),)
	@python3 tools/roofs.py show --store $(ROOF_STORE) $(if $(HW_JSON),--hardware $(HW_JSON),)
topology-report:
	@python3 tools/topology_report.py --store $(ROOF_STORE) \
	  $(if $(HW_JSON),--hardware $(HW_JSON),) $(if $(TOPO_JSON),--topology $(TOPO_JSON),) \
	  $(if $(COMPARE_JSON),--compare $(COMPARE_JSON),)
envelope:
	@python3 tools/envelope_report.py --wave $(WAVE_JSON) --store $(ROOF_STORE) \
	  $(if $(HW_JSON),--hardware $(HW_JSON),) $(if $(TOPO_JSON),--topology $(TOPO_JSON),) \
	  $(if $(CENSUS_JSON),--census $(CENSUS_JSON),) \
	  $(if $(COSTMAP_JSON),--costmap $(COSTMAP_JSON),) $(if $(ENV_OUT),--out $(ENV_OUT),)

# ── CPU profiling gate ────────────────────────────────────────────────────────────
# One artifact directory per run, the same gate language everywhere.  cpu-check is the
# preflight that every CPU optimisation session starts with; profile-cpu-check is the
# 1-second "is that preflight still valid for this binary/source/env/host?".
PROFILES_DIR ?= profiles
CPU_PROFILE  ?=
CPU_MODEL    ?=
cpu-check: $(TARGET) $(MEMBW_BIN)
	@MEMBW_BIN=$(MEMBW_BIN) PROFILES_DIR=$(PROFILES_DIR) CPU_PROFILE=$(CPU_PROFILE) \
	  CPU_MODEL=$(CPU_MODEL) bash tools/cpu_check.sh
# doctor: the under-a-minute, model-free FIRST look at a box for the streaming server.
# Identity + bandwidth (cached per hardware fingerprint in profiles/roofs) + caps + RESOLVED
# dispatch + model-free matmat shapes, then a PREDICTED W x K / batch cap / quantum / env set
# with a label on every number and a schema-valid draft profile. On a 32-core Arm Linux box it
# first runs the simultaneous 1x8/2x8/4x8 roof_matvec_int8 topology discriminator and archives
# arm_gemv_scaling.json. It never replaces cpu-check or a wave; it tells the next agent where to start.
#   make doctor                                 # ~10-40 s
#   make doctor DOCTOR_ARGS="--full"            # + --self-test and the quick --matmat-tune grid
#   make doctor DOCTOR_ARGS="--no-measure"      # cached roofs only, never runs membw
DOCTOR_ARGS ?=
# The int8 GEMV roof (tests/roof_matvec_int8.c): the Talker's own kernel streaming 28 distinct
# weight matrices per shape from DRAM, linked against the built engine objects so it runs the
# same code the server does.  membw's read loop under-reports what this kernel achieves
# (35 vs 58 GB/s at 4T on an 8-core Emerald Rapids), so the doctor uses THIS for the Talker
# term and `--layers 2` (a CP-sized working set) for the cache-resident one.
ROOF_MATVEC_BIN ?= /tmp/qwen_roof_matvec
$(ROOF_MATVEC_BIN): tests/roof_matvec_int8.c $(filter-out main.o,$(OBJS)) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
roof-matvec: $(ROOF_MATVEC_BIN)
	@$(ROOF_MATVEC_BIN) --threads $(or $(ROOF_THREADS),4) --reps $(or $(ROOF_REPS),3)
doctor: $(TARGET) $(MEMBW_BIN) $(ROOF_MATVEC_BIN)
	@python3 tools/doctor.py --bin ./$(TARGET) --membw $(MEMBW_BIN) --roof $(ROOF_MATVEC_BIN) \
	  --store $(PROFILES_DIR)/roofs $(DOCTOR_ARGS)
test-doctor:
	@python3 tests/test_doctor.py

# Fast legacy CPU candidate screen. This is model-free and does not replace cpu-check,
# doctor or qualification. Keep physical-core and SMT runs in separate manifests.
LEGACY_SCREEN_OUT ?= /tmp/qwen-legacy-cpu-screen-$(shell date +%Y%m%d-%H%M%S)
LEGACY_SCREEN_MODE ?= unspecified
LEGACY_SCREEN_THREADS ?= 1
LEGACY_SCREEN_CPUS ?=
LEGACY_SCREEN_ARGS ?=
legacy-cpu-screen: $(TARGET) $(MEMBW_BIN) $(ROOF_MATVEC_BIN)
	@python3 tools/legacy_cpu_screen.py --bin ./$(TARGET) --membw $(MEMBW_BIN) \
	  --roof $(ROOF_MATVEC_BIN) --out "$(LEGACY_SCREEN_OUT)" \
	  --mode "$(LEGACY_SCREEN_MODE)" --threads "$(LEGACY_SCREEN_THREADS)" \
	  $(if $(LEGACY_SCREEN_CPUS),--cpus "$(LEGACY_SCREEN_CPUS)",) $(LEGACY_SCREEN_ARGS)
# doctor-wave: run the grid the doctor recommended (profiles/doctor/LATEST/wave-plan.json)
# sequentially from ONE python file: no shell chain, no pgrep, one log per run, a summary
# table at the end.  WAVE_ARGS="--dry-run" prints the commands; --only <label,...> a subset.
WAVE_PLAN ?= $(PROFILES_DIR)/doctor/LATEST/wave-plan.json
WAVE_ARGS ?=
doctor-wave: $(TARGET)
	@python3 tools/doctor_wave.py $(WAVE_PLAN) --bin ./$(TARGET) $(WAVE_ARGS)
dispatch-map: $(TARGET)
	@./$(TARGET) --dispatch-map

# PARITY-2: what the engine is ACTUALLY doing, per declared flag.  The scope table it embeds is
# generated from the sources, so `check-flag-parity` regenerates and diffs it: a new backend
# guard around a flag's effect cannot drift away from what the engine reports.
effective-config: $(TARGET)
	@./$(TARGET) --effective-config
test-sd-pool-config: $(TARGET)
	@bash tests/test_sd_pool_config.sh ./$(TARGET)
check-flag-parity:
	@python3 tools/flag_parity.py --emit-c /tmp/qwen_flag_scope.gen.h >/dev/null
	@diff -u qwen_flag_scope.h /tmp/qwen_flag_scope.gen.h \
	  || { echo "FAIL: qwen_flag_scope.h is stale — run tools/flag_parity.py --emit-c qwen_flag_scope.h"; exit 1; }
	@python3 tools/flag_parity.py --check | tail -8
profile-cpu-check: $(TARGET)
	@python3 tools/profile_check.py --profiles $(PROFILES_DIR) --bin ./$(TARGET) $(if $(CPU_MODEL),--model $(CPU_MODEL),)

# ── dispatcher thresholds and the small serving curves ──────────────────────────
# These four were listed in `make help` and .PHONY without a recipe, so `make kernel-tune`
# printed "Nothing to be done" and exited 0.  Wired to the scripts that exist.
KT_ARGS ?=
MODEL_BASE_LARGE ?= qwen3-tts-1.7b-base
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
kernel-tune: $(TARGET)
	@bash tests/kernel_tune.sh $(KT_ARGS)
kernel-tune-quick: $(TARGET)
	@bash tests/kernel_tune.sh --quick $(KT_ARGS)
tune-archive: $(TARGET)
	@[ -n "$(BOX)" ] || { echo "usage: make tune-archive BOX=<name>   (e.g. BOX=aws-c8a-16c-zen5)"; exit 2; }
	@mkdir -p docs/boxes
	@bash tests/kernel_tune.sh -o /tmp/qwen_tune_$(BOX) $(KT_ARGS) && \
	  cp /tmp/qwen_tune_$(BOX)/tune.json docs/boxes/$(BOX)_tune.json && \
	  echo "archived docs/boxes/$(BOX)_tune.json  (compare: python3 tests/mm_tune_compare.py)"
# /v1/tts with a preset speaker: CustomVoice models only (a Base model has no presets)
server-batch-microbench: $(TARGET)
	@bash tests/serve_batch_bench.sh $(MODEL_SMALL)
server-batch-microbench-full: $(TARGET)
	@bash tests/serve_batch_bench.sh $(MODEL_LARGE)
mini-bench-06b: $(TARGET)
	@python3 tests/serve_parallel_wave.py --model $(MODEL_BASE_SMALL) --bin ./$(TARGET) \
	  --topo 1x$(NPROC) --conc 1,2,4 --waves 2 --seed 42 --precision int8 \
	  --no-profile "mini-bench: compiled defaults" --classes short --out /tmp/mini_bench_06b \
	  --port 9600 --label mini06b
mini-bench-17b: $(TARGET)
	@python3 tests/serve_parallel_wave.py --model $(MODEL_BASE_LARGE) --bin ./$(TARGET) \
	  --topo 1x$(NPROC) --conc 1,2,4 --waves 2 --seed 42 --precision int8 \
	  --no-profile "mini-bench: compiled defaults" --classes short --out /tmp/mini_bench_17b \
	  --port 9601 --label mini17b

PREFILL_BENCH_SRC = tests/prefill_bench.c qwen_tts_kernels.c qwen_tts_thread.c \
                    qwen_tts_costmap.c qwen_tts_kleidi.c qwen_tts_q8repack.c $(KAI_SRCS) $(KAI_ASM)
prefill-bench: $(INGOT_LIB)
	@echo "=== prefill cost: what the per-call fixed cost actually is ==="
ifeq ($(UNAME_S),Darwin)
	@clang $(PARITY_CF) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -march=native \
	  $(PREFILL_BENCH_SRC) $(INGOT_LIB) -framework Accelerate -lm -o /tmp/prefill_bench
else
	@$(CC) $(PARITY_CF) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas $(ARCH_FLAGS) \
	  $(PREFILL_BENCH_SRC) $(INGOT_LIB) -lopenblas -lm -lpthread -o /tmp/prefill_bench
endif
	@/tmp/prefill_bench $(PB_ROWS) $(PB_COLS) $(PB_THREADS)

PB_ROWS    ?= 2048
PB_COLS    ?= 2048
PB_THREADS ?= 1
# NOTE: this target compiles its own binary, so it uses SIMD as given on ITS command line.
# make prefill-bench SIMD=portable measures the portable kernels, not whatever SIMD=auto
# picks for this host. The harness prints its own dispatcher table so a mismatch is visible.

test-all: test-small test-large test-regression test-errors test-emotion test-emotion-ft test-compose test-caps check-flag-registry test-selftest test-golden test-serve-repro
	@echo ""
	@echo "========================================="
	@echo "  All tests passed (0.6B + 1.7B)"
	@echo "========================================="

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
	   grep -qE "matvec \+ attn:    (AVX-512|AVX2|scalar)" $(TEST_DIR)/caps.txt || { echo "FAIL: x86 must report AVX-512 (SIMD=avx512/avx512vnni/avx512bf16/amx), AVX2 (portable) or scalar"; exit 1; }; \
	   if grep -q "WARNING: built with AVX2 but this CPU lacks it" $(TEST_DIR)/caps.txt; then echo "FAIL: AVX2 build on a non-AVX2 CPU"; exit 1; fi; \
	 fi
	@grep -q "matvec threads:" $(TEST_DIR)/caps.txt && ! grep -q "SINGLE-THREAD" $(TEST_DIR)/caps.txt || { echo "FAIL: threads must report an active pool (GCD/pthread/Win32), not SINGLE-THREAD"; exit 1; }
	@echo "PASS: --caps report consistent with build arch"
	@echo ""

GOLDEN_EN = The quick brown fox jumps over the lazy dog on a sunny afternoon.
GOLDEN_IT = Buongiorno a tutti, questa è una dimostrazione del sistema di sintesi vocale.
GOLDEN_DET = -j1 --temperature 0 --seed 42

test-golden: $(TARGET)
	@echo "=== Golden-reference correctness test (mel-corr + duration) ==="
	@if ! python3 -c "import librosa" 2>/dev/null; then echo "SKIP: python3 librosa not installed (pip install librosa)"; exit 0; fi
	@mkdir -p $(TEST_DIR)
# MODEL_SMALL is guarded like MODEL_LARGE: a rented box usually carries ONE checkpoint, and a
# missing model is not a numerical regression.  RAN counts what actually ran, so "everything
# skipped" cannot be reported as PASS.
	@FAIL=0; RAN=0; \
	 if [ -d $(MODEL_SMALL) ]; then RAN=1; \
	   ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o $(TEST_DIR)/gold_06b_en.wav >/dev/null 2>&1; \
	   python3 tests/compare_audio.py tests/golden/golden_06b_en.wav $(TEST_DIR)/gold_06b_en.wav --label "0.6B en" || FAIL=1; \
	   ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l Italian --text "$(GOLDEN_IT)" -o $(TEST_DIR)/gold_06b_it.wav >/dev/null 2>&1; \
	   python3 tests/compare_audio.py tests/golden/golden_06b_it.wav $(TEST_DIR)/gold_06b_it.wav --label "0.6B it" || FAIL=1; \
	   ./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --int8 --text "$(GOLDEN_EN)" -o $(TEST_DIR)/gold_06b_en_int8.wav >/dev/null 2>&1; \
	   python3 tests/compare_audio.py tests/golden/golden_06b_en_int8.wav $(TEST_DIR)/gold_06b_en_int8.wav --label "0.6B en int8" || FAIL=1; \
	 else echo "SKIP: 0.6B (model absent)"; fi; \
	 if [ -d $(MODEL_LARGE) ]; then RAN=1; \
	   ./$(TARGET) -d $(MODEL_LARGE) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o $(TEST_DIR)/gold_17b_en.wav >/dev/null 2>&1; \
	   python3 tests/compare_audio.py tests/golden/golden_17b_en.wav $(TEST_DIR)/gold_17b_en.wav --label "1.7B en" || FAIL=1; \
	 else echo "SKIP: 1.7B (model absent)"; fi; \
	 if [ "$$RAN" -eq 0 ]; then echo "FAIL: no golden model present, nothing was checked"; exit 1; fi; \
	 if [ "$$FAIL" -ne 0 ]; then echo "FAIL: golden-reference mismatch (numerical regression?)"; exit 1; fi; \
	 echo "PASS: all golden references present matched"
	@echo ""

test-stream-layout: $(TARGET)
	@bash tests/stream_layout_smoke.sh

golden-update: $(TARGET)
	@echo "=== Regenerating golden references (review the diff before committing!) ==="
	@mkdir -p tests/golden
	./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o tests/golden/golden_06b_en.wav
	./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l Italian --text "$(GOLDEN_IT)" -o tests/golden/golden_06b_it.wav
	./$(TARGET) -d $(MODEL_SMALL) $(GOLDEN_DET) -s ryan -l English --int8 --text "$(GOLDEN_EN)" -o tests/golden/golden_06b_en_int8.wav
	@if [ -d $(MODEL_LARGE) ]; then ./$(TARGET) -d $(MODEL_LARGE) $(GOLDEN_DET) -s ryan -l English --text "$(GOLDEN_EN)" -o tests/golden/golden_17b_en.wav; fi
	@echo "Done. git diff tests/golden/ and commit if intended."

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

test-serve-bench: $(TARGET)
	@echo "=== Server Benchmark (seed=42, 2 runs) ==="
	@mkdir -p $(TEST_DIR)
	@./$(TARGET) -d $(MODEL_SMALL) --serve 8091 &>/dev/null & SERVER_PID=$$!; \
	 sleep 4; \
	 echo "--- Run 1 (cold) ---"; \
	 T1=$$(curl -s -w "total %{time_total}s  TTFB %{time_starttransfer}s" -X POST http://localhost:8091/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"The quick brown fox jumps over the lazy dog on a sunny afternoon.","speaker":"ryan","language":"English","seed":42,"temperature":0}' \
	   -o $(TEST_DIR)/bench_run1.wav); \
	 S1=$$(stat -f%z $(TEST_DIR)/bench_run1.wav 2>/dev/null || stat -c%s $(TEST_DIR)/bench_run1.wav 2>/dev/null); \
	 echo "  $${T1}, $$S1 bytes   (non-streaming /v1/tts: TTFB = whole synthesis, headers go with the WAV)"; \
	 if [ "$$S1" -le 44 ]; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: empty WAV"; exit 1; fi; \
	 echo "--- Run 2 (warm) ---"; \
	 T2=$$(curl -s -w "total %{time_total}s  TTFB %{time_starttransfer}s" -X POST http://localhost:8091/v1/tts \
	   -H "Content-Type: application/json" \
	   -d '{"text":"The quick brown fox jumps over the lazy dog on a sunny afternoon.","speaker":"ryan","language":"English","seed":42,"temperature":0}' \
	   -o $(TEST_DIR)/bench_run2.wav); \
	 S2=$$(stat -f%z $(TEST_DIR)/bench_run2.wav 2>/dev/null || stat -c%s $(TEST_DIR)/bench_run2.wav 2>/dev/null); \
	 echo "  $${T2}, $$S2 bytes"; \
	 echo "--- Comparing outputs (mel-corr + duration, not md5: multi-thread FP order flips a sample now and then) ---"; \
	 if ! python3 tests/compare_audio.py $(TEST_DIR)/bench_run1.wav $(TEST_DIR)/bench_run2.wav --label "cold vs warm"; then kill $$SERVER_PID 2>/dev/null; echo "FAIL: cold and warm outputs diverge"; exit 1; fi; \
	 kill $$SERVER_PID 2>/dev/null; \
	 echo "PASS: cold and warm outputs match (temperature 0, seed 42)"
	@echo ""

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

test-serve-concurrent: $(TARGET)
	@MODEL=$(MODEL_SMALL) bash tests/test_parallel.sh
	@echo ""

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

test-serve-batch: $(TARGET)
	@bash tests/serve_batch.sh $(MODEL_SMALL)

test-serve-continuous: $(TARGET)
	@bash tests/serve_continuous_stress.sh $(MODEL_SMALL) 8786 6 2

test-serve-stream-batch: $(TARGET)
	@bash tests/serve_stream_batch.sh $(MODEL_SMALL)

MINUTES ?= 30
LEVEL ?= 2

test-serve-all: test-serve test-serve-bench test-serve-repro test-serve-openai test-serve-parallel test-serve-concurrent test-serve-batch test-serve-continuous test-serve-stream-batch test-stage-policy
	@echo "=== All server tests passed ==="

bench: $(TARGET)
	@./bench.sh --level basic --seed 42

bench-full: $(TARGET)
	@./bench.sh --level full --seed 42

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

test-en: test-small-en
test-it-ryan: test-small-it

.PHONY: bench-fingerprint bench-topo bench-suite bench-soak bench-suite-full check-flag-registry prefill-bench \
	cpu-check doctor test-doctor legacy-cpu-screen roof-matvec dispatch-map profile-cpu-check profile-cpu tune-archive
.PHONY: server-hw-check box-report membw check-matmat-parity check-matmat-parity-x86 \
	server-batch-microbench server-batch-microbench-full mini-bench-06b mini-bench-17b \
	kernel-tune kernel-tune-quick test-decoder-batch-parity server-soak x86-qkv-bench x86-amx-b32-bench x86-b1-gemv-bench
.PHONY: all help blas clean debug info serve cp-microbench batching-bench test-batch test-batch-invariance test-errors test-emotion test-emotion-ft emotion-demo emo-suite emotion-seeds test-compose test-caps test-selftest test-golden test-stream-layout test-sd-pool-config golden-update emovoice emo-06b-demo quant-ladder test-modes test-qvoice e2e \
		test-json-parser \
        emotion-para-demo para-demo \
        test-serve test-serve-bench test-serve-repro test-serve-openai test-serve-parallel test-serve-concurrent test-serve-batch test-serve-continuous test-serve-stream-batch test-stage-policy test-serve-all \
        test-clone test-voice-design \
        demo-clone \
        test-small test-small-en test-small-it test-small-vivian test-small-stream test-small-stdout \
        test-large test-large-en test-large-it test-large-config test-large-instruct \
        test-large-int8 test-large-int4 test-large-quant \
        test-regression test-all test-en test-it-ryan
