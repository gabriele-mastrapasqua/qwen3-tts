# Makefile for Qwen3-TTS Pure C Inference Engine

UNAME_S := $(shell uname -s)
CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDLIBS = -lm -lpthread

# BLAS (Accelerate on macOS, OpenBLAS on Linux)
ifeq ($(UNAME_S),Darwin)
    CFLAGS_BASE += -DUSE_BLAS -DACCELERATE_NEW_LAPACK
    LDLIBS += -framework Accelerate
else
    CFLAGS_BASE += -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
    LDLIBS += -lopenblas
endif

CFLAGS = $(CFLAGS_BASE)

# Source files
SRCS = main.c \
       qwen_tts.c \
       qwen_tts_talker.c \
       qwen_tts_code_predictor.c \
       qwen_tts_speech_decoder.c \
       qwen_tts_kernels.c \
       qwen_tts_kernels_generic.c \
       qwen_tts_kernels_neon.c \
       qwen_tts_kernels_avx.c \
       qwen_tts_audio.c \
       qwen_tts_sampling.c \
       qwen_tts_tokenizer.c \
       qwen_tts_safetensors.c \
       qwen_tts_server.c \
       qwen_tts_voice_clone.c \
       qwen_tts_speech_encoder.c

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
	@echo "  make test-clone      - Voice clone e2e (generate ref → clone → stream)"
	@echo "  make demo-clone      - Voice clone demo using sample WAV"
	@echo "  make test-regression - Cross-model regression checks"
	@echo "  make test-all        - Run everything (0.6B + 1.7B + regression)"
	@echo ""
	@echo "Example: make blas && ./$(TARGET) -d $(MODEL_DIR) -t \"Hello world\" -o output.wav"

# Build
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

blas: $(TARGET)

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

test-large: test-large-config test-large-en test-large-it test-large-instruct
	@echo "=== All 1.7B tests passed ==="

# ── Cross-model regression tests ──

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

test-all: test-small test-large test-regression
	@echo ""
	@echo "========================================="
	@echo "  All tests passed (0.6B + 1.7B)"
	@echo "========================================="

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

# ── Combined server tests ──

test-serve-all: test-serve test-serve-bench test-serve-openai test-serve-parallel
	@echo "=== All server tests passed ==="

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

test-voice-design: $(TARGET)
	@echo "=== VoiceDesign test ==="
	@if [ ! -d $(MODEL_VOICE_DESIGN) ]; then echo "SKIP: $(MODEL_VOICE_DESIGN) not found (run ./download_model.sh --model voice-design)"; exit 0; fi
	@mkdir -p $(TEST_DIR)
	@echo ""
	@echo "--- VoiceDesign: British male ---"
	./$(TARGET) -d $(MODEL_VOICE_DESIGN) -l English \
		--voice-design \
		--instruct "A deep male voice with a British accent, speaking slowly and calmly" \
		--text "Good evening, welcome to the broadcast." \
		-o $(TEST_DIR)/vd_british.wav 2>&1 | tee $(TEST_DIR)/vd_british.wav.log
	$(call validate_wav,$(TEST_DIR)/vd_british.wav,VoiceDesign: British male)
	@echo "--- VoiceDesign: energetic female ---"
	./$(TARGET) -d $(MODEL_VOICE_DESIGN) -l English \
		--voice-design \
		--instruct "Young energetic female, cheerful and fast-paced" \
		--text "Oh my gosh, this is so exciting!" \
		-o $(TEST_DIR)/vd_cheerful.wav 2>&1 | tee $(TEST_DIR)/vd_cheerful.wav.log
	$(call validate_wav,$(TEST_DIR)/vd_cheerful.wav,VoiceDesign: energetic female)
	@echo "=== VoiceDesign test passed ==="
	@echo "Listen:"
	@echo "  British:   afplay $(TEST_DIR)/vd_british.wav"
	@echo "  Cheerful:  afplay $(TEST_DIR)/vd_cheerful.wav"

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

.PHONY: all help blas clean debug info serve \
        test-serve test-serve-bench test-serve-openai test-serve-parallel test-serve-all \
        test-clone test-voice-design \
        demo-clone \
        test-small test-small-en test-small-it test-small-vivian test-small-stream test-small-stdout \
        test-large test-large-en test-large-it test-large-config test-large-instruct \
        test-regression test-all test-en test-it-ryan
