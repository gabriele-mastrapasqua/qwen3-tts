# ingot — GGUF + safetensors reader in C11. No dependencies, no build system
# required: `make` is enough, and `cc -Iinclude src/*.c` works too.
#
# SPDX-License-Identifier: MIT

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2
WARN     = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wvla
INCLUDE  = -Iinclude
LDLIBS   = -lpthread -lm

# Container only: no dequant, no kernels. ~40 KB of object code.
CORE_SRC   = src/dtype.c src/gguf.c src/safetensors.c src/wfile.c src/write.c
# The optional half.
QUANT_SRC  = src/cpu.c src/dequant.c src/dequant_iq.c src/kernels.c src/generic.c src/quantize.c

SRC        = $(CORE_SRC) $(QUANT_SRC)
OBJ        = $(SRC:.c=.o)
CORE_OBJ   = $(CORE_SRC:.c=.o)

LIB      = libingot.a
TESTS    = build/test_gguf build/test_safetensors build/test_quant build/test_formats \
           build/test_oracle build/test_wfile build/test_write build/test_convert
TOOLS    = build/ingot-dump

.PHONY: all lib tools test clean core-only help
all: lib tools examples

## help: this list
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  make /' 

## lib: build libingot.a
lib: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

# -MMD: header dependency tracking. Without it, editing a public header
# rearchives only the objects whose .c changed, and the ABI mismatch between
# fresh and stale objects shows up as a segfault in the tests.
%.o: %.c
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) -MMD -MP -c $< -o $@
-include $(SRC:.c=.d)

# Changing CFLAGS must invalidate the objects: `make test-asan` followed by a
# plain `make` otherwise links the surviving sanitized .o files and dies with
# undefined __asan references — or worse, a plain .o survives into a sanitized
# run and that file silently goes unsanitized. The stamp name encodes the
# flags; a different set removes the old stamp and forces the rebuild.
FLAGS_ID  := $(firstword $(shell printf '%s' '$(WARN) $(CFLAGS)' | cksum))
FLAGSTAMP := build/.flags-$(FLAGS_ID)
$(FLAGSTAMP): | build
	@rm -f build/.flags-*
	@touch $@
$(OBJ): $(FLAGSTAMP)

# A build with the container readers alone, to prove the split is real: a
# consumer that only wants to open files must not have to link the SIMD.
## core-only: build the readers without the quantization half
core-only:
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) -DINGOT_NO_KERNELS -c $(CORE_SRC) src/dequant.c
	@echo "core-only build OK"

# This machine is aarch64, so the x86 paths would otherwise only ever meet a
# compiler on someone else's box. Compile-only: it catches #ifdef rot and
# intrinsic misuse, not runtime bugs — those still need real x86 hardware.
## check-x86: cross-compile everything for x86-64 at AVX2+F16C and AVX-512 F/BW/VL/VNNI/BF16
X86_TARGET ?= x86_64-apple-macos12
check-x86:
	@mkdir -p build/x86
	@for f in $(SRC); do \
	  $(CC) -target $(X86_TARGET) $(WARN) $(CFLAGS) $(INCLUDE) -mavx2 -mfma -mf16c \
	    -c $$f -o build/x86/$$(basename $$f .c).avx2.o || exit 1; \
	  $(CC) -target $(X86_TARGET) $(WARN) $(CFLAGS) $(INCLUDE) \
	    -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512bf16 -mf16c \
	    -c $$f -o build/x86/$$(basename $$f .c).avx512.o || exit 1; \
	done
	@echo "x86-64 cross-compile OK (avx2+f16c; avx512 f/bw/vl/vnni/bf16)"

build:
	@mkdir -p build

build/%: tests/%.c $(LIB) | build
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) $< $(LIB) $(LDLIBS) -o $@

build/ingot-dump: tools/ingot_dump.c $(LIB) | build
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) $< $(LIB) $(LDLIBS) -o $@

## tools: build ingot-dump
tools: $(TOOLS)

.PHONY: examples
## examples: build examples/minimal
examples: build/minimal
build/minimal: examples/minimal.c $(LIB) | build
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) $< $(LIB) $(LDLIBS) -o $@

## test: build and run the whole suite
# RUN= an optional launcher for the test binaries (an emulator, mostly:
# `make test RUN="sde64 -spr --"` exercises AVX-512/VNNI paths on any x86).
test: $(TESTS)
	@fail=0; for t in $(TESTS); do \
		echo "== $$t"; $(RUN) ./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "TESTS FAILED"; exit 1; fi

# Same suite under the sanitizers. The malformed-file fixtures exist precisely
# so this catches the reads a plain run would not notice.
.PHONY: test-asan
# ── fuzzing ────────────────────────────────────────────────────────────────
## fuzz: mutation-fuzz both readers (ROUNDS=n to change the count)
ROUNDS ?= 6000
fuzz: build/fuzz_mutate
	$(RUN) ./build/fuzz_mutate $(ROUNDS)

## fuzz-leaks: the same under the memory checker
fuzz-leaks: build/fuzz_mutate
	MallocStackLogging=1 leaks --atExit --quiet -- ./build/fuzz_mutate $(ROUNDS) \
		| grep -E "rounds:|leaks for"

build/fuzz_mutate: tests/fuzz_mutate.c $(LIB) | build
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) $< $(LIB) $(LDLIBS) -o $@

.PHONY: fuzz fuzz-leaks

# ── memory checking ────────────────────────────────────────────────────────
# On macOS `leaks` is the gate: ASan frequently hangs there, and a hung run
# reads as a slow test rather than a broken one. ASan stays for Linux/CI.
## test-leaks: the suite under macOS `leaks` (the memory gate on Mac)
test-leaks: $(TESTS)
	@fail=0; for t in $(TESTS); do \
		echo "== leaks $$t"; \
		MallocStackLogging=1 leaks --atExit --quiet -- ./$$t > build/leaks-$$(basename $$t).log 2>&1; \
		rc=$$?; \
		grep -E "leaks for [0-9]+ total leaked bytes|0 leaks for 0 total leaked bytes" \
			build/leaks-$$(basename $$t).log | tail -1; \
		if [ $$rc -ne 0 ]; then fail=1; \
			echo "   ^ see build/leaks-$$(basename $$t).log"; fi; \
	done; \
	if [ $$fail -eq 0 ]; then echo "NO LEAKS"; else echo "LEAKS FOUND"; exit 1; fi

.PHONY: test-leaks

## test-asan: the suite under ASan+UBSan (Linux; on macOS prefer test-leaks)
# SIMD= extra ISA flags so the sanitizers see the vector paths too
# (`make test-asan SIMD="-mavx2 -mfma"`); without it only the baseline ISA
# of the host is compiled, which is how an unaligned NEON store once hid.
test-asan:
	$(MAKE) clean
	$(MAKE) test CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(SIMD)"

## test-x86-rosetta: on an Apple Silicon Mac, build the suite as x86_64 with
## AVX2+FMA and run it under Rosetta (macOS 15+ translates AVX2). The cheapest
## way to execute the AVX2 lanes without an x86 machine. AVX-512/VNNI are NOT
## translated — those run in CI under Intel SDE instead.
test-x86-rosetta:
	$(MAKE) clean
	$(MAKE) test CC="$(CC) -arch x86_64" CFLAGS="-O2 -mavx2 -mfma"
	$(MAKE) clean

.PHONY: test-x86-rosetta

# ── regenerating what is generated ─────────────────────────────────────────
# Both need llama.cpp's `gguf` package. The outputs are committed, so a plain
# checkout builds and tests without Python; you only run these when the format
# gains a type or when you want to re-derive the tables from a newer llama.cpp.
PYTHON ?= python3

## gen-tables: re-extract the IQ codebooks from the gguf package
gen-tables:
	$(PYTHON) tools/gen_iq_tables.py

## gen-fixtures: re-generate the oracle fixtures from the gguf package
gen-fixtures:
	$(PYTHON) tools/gen_reference.py

.PHONY: gen-tables gen-fixtures check-real
## check-real: cross-check a real checkpoint against an independent parse
check-real: tools
	@test -n "$(MODEL)" || { echo "usage: make check-real MODEL=path/to/model.safetensors"; exit 2; }
	$(PYTHON) tools/check_against_python.py "$(MODEL)"

# ── the two-file drop-in build ─────────────────────────────────────────────
# amalgam/ is generated from src/. `amalgam-test` runs the ENTIRE suite against
# the generated pair instead of the library, which is the only way to keep the
# two builds from drifting: a source file added to src/ but not to the
# generator would pass `make test` and fail here.
.PHONY: amalgam amalgam-test
## amalgam: regenerate the two-file drop-in build
amalgam:
	$(PYTHON) tools/amalgamate.py

AMALGAM_SHIM = build/shim/ingot

## amalgam-test: run the whole suite against the generated pair
amalgam-test: amalgam
	@mkdir -p $(AMALGAM_SHIM) build/amalgam
	@for h in dtype gguf safetensors quant wfile write; do 		printf '#include <ingot.h>\n' > $(AMALGAM_SHIM)/$$h.h; 	done
	@fail=0; for t in tests/test_*.c; do 		name=$$(basename $$t .c); 		$(CC) $(WARN) $(CFLAGS) -Ibuild/shim -Iamalgam $$t amalgam/ingot.c 			$(LDLIBS) -o build/amalgam/$$name || fail=1; 		echo "== amalgam/$$name"; $(RUN) ./build/amalgam/$$name > /dev/null || fail=1; 	done; 	if [ $$fail -eq 0 ]; then echo "AMALGAM MATCHES THE LIBRARY"; 	else echo "AMALGAM BUILD FAILED"; exit 1; fi
	@$(CXX) -std=c++17 -Wall -Wextra -Iamalgam -fsyntax-only -x c++ amalgam/ingot.h \
		&& echo "the header also parses as C++"

## clean: remove build products
clean:
	rm -f $(OBJ) $(LIB) src/*.o src/*.d
	rm -rf build
