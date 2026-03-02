# qwen_tts — Qwen3-TTS Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = qwen_tts.c qwen_tts_kernels.c qwen_tts_kernels_generic.c qwen_tts_kernels_neon.c qwen_tts_kernels_avx.c qwen_tts_talker.c qwen_tts_code_predictor.c qwen_tts_speech_decoder.c qwen_tts_audio.c qwen_tts_sampling.c qwen_tts_tokenizer.c qwen_tts_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = qwen_tts

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas

# Default: show available targets
all: help

help:
	@echo "qwen_tts — Qwen3-TTS Pure C Inference - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
	@echo ""
	@echo "Other targets:"
	@echo "  make debug    - Debug build with AddressSanitizer"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./qwen_tts -d model_dir --text 'Hello world' -o output.wav"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c qwen_tts.h qwen_tts_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Sample generation (all output to /tmp)
# =============================================================================
MODEL_DIR = qwen3-tts-0.6b
QWEN = ./$(TARGET)
SAMPLE_FLAGS = --seed 42

sample-ita: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "Ciao mondo, come stai oggi? Spero tutto bene!" \
		--speaker Serena --language Italian $(SAMPLE_FLAGS) -o /tmp/sample_ita.wav

sample-eng: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "Hello world, how are you doing today?" \
		--speaker Serena --language English $(SAMPLE_FLAGS) -o /tmp/sample_eng.wav

sample-fra: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "Bonjour le monde, comment allez-vous aujourd'hui?" \
		--speaker Serena --language French $(SAMPLE_FLAGS) -o /tmp/sample_fra.wav

sample-deu: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "Hallo Welt, wie geht es Ihnen heute?" \
		--speaker Serena --language German $(SAMPLE_FLAGS) -o /tmp/sample_deu.wav

sample-esp: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "Hola mundo, como estas hoy? Espero que bien!" \
		--speaker Serena --language Spanish $(SAMPLE_FLAGS) -o /tmp/sample_esp.wav

sample-jpn: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "こんにちは世界、今日はお元気ですか？" \
		--speaker Serena --language Japanese $(SAMPLE_FLAGS) -o /tmp/sample_jpn.wav

sample-kor: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "안녕하세요 세계, 오늘 어떻게 지내세요?" \
		--speaker Serena --language Korean $(SAMPLE_FLAGS) -o /tmp/sample_kor.wav

sample-zho: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "你好世界，今天你好吗？希望一切都好！" \
		--speaker Serena --language Chinese $(SAMPLE_FLAGS) -o /tmp/sample_zho.wav

sample-rus: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "Привет мир, как у вас дела сегодня?" \
		--speaker Serena --language Russian $(SAMPLE_FLAGS) -o /tmp/sample_rus.wav

sample-por: $(TARGET)
	$(QWEN) -d $(MODEL_DIR) --text "Olá mundo, como você está hoje? Espero que bem!" \
		--speaker Serena --language Portuguese $(SAMPLE_FLAGS) -o /tmp/sample_por.wav

sample-all: sample-ita sample-eng sample-fra sample-deu sample-esp sample-jpn sample-kor sample-zho sample-rus sample-por
	@echo ""
	@echo "All samples generated in /tmp/:"
	@ls -lh /tmp/sample_*.wav

.PHONY: sample-ita sample-eng sample-fra sample-deu sample-esp sample-jpn sample-kor sample-zho sample-rus sample-por sample-all

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) main.o $(TARGET)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backend: blas (Apple Accelerate)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

# =============================================================================
# Dependencies
# =============================================================================
qwen_tts.o: qwen_tts.c qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h qwen_tts_tokenizer.h qwen_tts_audio.h
qwen_tts_kernels.o: qwen_tts_kernels.c qwen_tts_kernels.h qwen_tts_kernels_impl.h
qwen_tts_kernels_generic.o: qwen_tts_kernels_generic.c qwen_tts_kernels_impl.h
qwen_tts_kernels_neon.o: qwen_tts_kernels_neon.c qwen_tts_kernels_impl.h
qwen_tts_kernels_avx.o: qwen_tts_kernels_avx.c qwen_tts_kernels_impl.h
qwen_tts_talker.o: qwen_tts_talker.c qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h
qwen_tts_code_predictor.o: qwen_tts_code_predictor.c qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h
qwen_tts_speech_decoder.o: qwen_tts_speech_decoder.c qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h
qwen_tts_audio.o: qwen_tts_audio.c qwen_tts_audio.h
qwen_tts_sampling.o: qwen_tts_sampling.c qwen_tts.h
qwen_tts_tokenizer.o: qwen_tts_tokenizer.c qwen_tts_tokenizer.h
qwen_tts_safetensors.o: qwen_tts_safetensors.c qwen_tts_safetensors.h
main.o: main.c qwen_tts.h qwen_tts_kernels.h
