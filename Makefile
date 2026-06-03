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
ifeq ($(UNAME_S),Darwin)
    ARCH_FLAGS = -march=native
else ifneq (,$(filter x86_64 amd64,$(UNAME_M)))
    ifeq ($(SIMD),scalar)
        ARCH_FLAGS =
    else ifeq ($(SIMD),avx512)
        ARCH_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx2 -mfma
    else
        ARCH_FLAGS = -mavx2 -mfma
    endif
else
    ARCH_FLAGS = -march=native
endif

CFLAGS_BASE = -Wall -Wextra -O3 $(ARCH_FLAGS) -ffast-math
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

CFLAGS = $(CFLAGS_BASE) $(EXTRA_CFLAGS)

# Source files
SRCS = main.c \
       qwen_tts.c \
       qwen_tts_talker.c \
       qwen_tts_code_predictor.c \
       qwen_tts_speech_decoder.c \
       qwen_tts_kernels.c \
       qwen_tts_thread.c \
       qwen_tts_kernels_generic.c \
       qwen_tts_kernels_neon.c \
       qwen_tts_kernels_avx.c \
       qwen_tts_audio.c \
       qwen_tts_sampling.c \
       qwen_tts_tokenizer.c \
       qwen_tts_safetensors.c \
       qwen_tts_server.c \
       qwen_tts_voice_clone.c \
       qwen_tts_speech_encoder.c \
       vendor/lz4.c

OBJS = $(SRCS:.c=.o)
TARGET = qwen_tts
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
	@echo "Example: make blas && ./$(TARGET) -d $(MODEL_DIR) -t \"Hello world\" -o output.wav"

# Build
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

blas: $(TARGET)

# CP micro-benchmark: separate binary instrumented with -DCP_MICROBENCH.
# Partitions per-frame Code Predictor time among sub-ops (QKV/attn/FFN/norm/lm_head).
# Clean rebuild into qwen_tts_cpbench so instrumented and normal .o never mix.
cp-microbench:
	$(MAKE) clean
	$(MAKE) TARGET=qwen_tts_cpbench EXTRA_CFLAGS=-DCP_MICROBENCH qwen_tts_cpbench
	@echo ""
	@echo "Built ./qwen_tts_cpbench  (run a normal generation; CP breakdown prints in the summary)"

# Compile C sources
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Header dependencies
main.o: main.c qwen_tts.h qwen_tts_audio.h qwen_tts_kernels.h qwen_tts_server.h
qwen_tts.o: qwen_tts.c qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h qwen_tts_tokenizer.h qwen_tts_audio.h
qwen_tts_talker.o: qwen_tts_talker.c qwen_tts.h qwen_tts_kernels.h
qwen_tts_code_predictor.o: qwen_tts_code_predictor.c qwen_tts.h qwen_tts_kernels.h
qwen_tts_speech_decoder.o: qwen_tts_speech_decoder.c qwen_tts.h qwen_tts_kernels.h
qwen_tts_kernels.o: qwen_tts_kernels.c qwen_tts_kernels.h qwen_tts_kernels_impl.h
qwen_tts_kernels_generic.o: qwen_tts_kernels_generic.c qwen_tts_kernels_impl.h
qwen_tts_kernels_neon.o: qwen_tts_kernels_neon.c qwen_tts_kernels_impl.h
qwen_tts_kernels_avx.o: qwen_tts_kernels_avx.c qwen_tts_kernels_impl.h
qwen_tts_audio.o: qwen_tts_audio.c qwen_tts_audio.h
qwen_tts_sampling.o: qwen_tts_sampling.c qwen_tts.h
qwen_tts_tokenizer.o: qwen_tts_tokenizer.c qwen_tts_tokenizer.h
qwen_tts_safetensors.o: qwen_tts_safetensors.c qwen_tts_safetensors.h
qwen_tts_server.o: qwen_tts_server.c qwen_tts_server.h qwen_tts.h
qwen_tts_voice_clone.o: qwen_tts_voice_clone.c qwen_tts_voice_clone.h qwen_tts.h qwen_tts_safetensors.h

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

# Debug build
debug: CFLAGS = $(CFLAGS_BASE) -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
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

test-all: test-small test-large test-regression test-errors test-caps test-golden
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
	 M1=$$(md5sum $(TEST_DIR)/repro_1.wav 2>/dev/null | cut -d' ' -f1 || md5 -q $(TEST_DIR)/repro_1.wav 2>/dev/null); \
	 M2=$$(md5sum $(TEST_DIR)/repro_2.wav 2>/dev/null | cut -d' ' -f1 || md5 -q $(TEST_DIR)/repro_2.wav 2>/dev/null); \
	 M3=$$(md5sum $(TEST_DIR)/repro_3.wav 2>/dev/null | cut -d' ' -f1 || md5 -q $(TEST_DIR)/repro_3.wav 2>/dev/null); \
	 S1=$$(stat -f%z $(TEST_DIR)/repro_1.wav 2>/dev/null || stat -c%s $(TEST_DIR)/repro_1.wav 2>/dev/null); \
	 echo "  run1=$$M1 ($$S1 B)  run2=$$M2  run3=$$M3"; \
	 if [ "$$S1" -le 44 ]; then echo "FAIL: empty WAV"; exit 1; fi; \
	 if [ "$$M1" != "$$M2" ] || [ "$$M2" != "$$M3" ]; then \
	   echo "FAIL: identical requests produced DIFFERENT output (state leaks between requests)"; exit 1; fi; \
	 echo "PASS: 3 identical requests are bit-identical"
	@echo ""

# ── Combined server tests ──

test-serve-all: test-serve test-serve-bench test-serve-repro test-serve-openai test-serve-parallel
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

.PHONY: all help blas clean debug info serve cp-microbench test-errors test-caps test-golden golden-update test-modes test-qvoice e2e \
        test-serve test-serve-bench test-serve-repro test-serve-openai test-serve-parallel test-serve-all \
        test-clone test-voice-design \
        demo-clone \
        test-small test-small-en test-small-it test-small-vivian test-small-stream test-small-stdout \
        test-large test-large-en test-large-it test-large-config test-large-instruct \
        test-large-int8 test-large-int4 test-large-quant \
        test-regression test-all test-en test-it-ryan
