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
       qwen_tts_safetensors.c

OBJS = $(SRCS:.c=.o)
TARGET = qwen_tts_bin
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
	@echo "Test (requires ./$(TARGET) and model in $(MODEL_DIR)/):"
	@echo "  make test-en        - English test (ryan)"
	@echo "  make test-it-ryan   - Italian test (ryan)"
	@echo "  make test-it-vivian - Italian test (vivian, female)"
	@echo "  make test-all       - Run all tests"
	@echo ""
	@echo "Example: make blas && ./$(TARGET) -d $(MODEL_DIR) -t \"Hello world\" -o output.wav"

# Build
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

blas: $(TARGET)

# Compile
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Header dependencies
main.o: main.c qwen_tts.h qwen_tts_audio.h qwen_tts_kernels.h
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

test-en:
	@echo "=== English test (ryan) ==="
	./$(TARGET) -d $(MODEL_DIR) -s ryan -l English \
		--text "Hello, this is a test of the text to speech system." -o test_en.wav
	@echo "Output: test_en.wav"

test-it-ryan:
	@echo "=== Italian test (ryan) ==="
	./$(TARGET) -d $(MODEL_DIR) -s ryan -l Italian \
		--text "Ciao, questa è una prova del sistema di sintesi vocale." -o test_it_ryan.wav
	@echo "Output: test_it_ryan.wav"

test-it-vivian:
	@echo "=== Italian test (vivian, female) ==="
	./$(TARGET) -d $(MODEL_DIR) -s vivian -l Italian \
		--text "Buongiorno, come state oggi? Spero tutto bene." -o test_it_vivian.wav
	@echo "Output: test_it_vivian.wav"

test-all: test-en test-it-ryan test-it-vivian
	@echo "=== All tests complete ==="

.PHONY: all help blas clean debug info test-en test-it-ryan test-it-vivian test-all
