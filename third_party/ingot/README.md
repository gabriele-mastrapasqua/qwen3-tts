# ingot

A zero-dependency C11 library for reading and writing **GGUF** and
**safetensors** — the two weight containers used by llama.cpp and the
Hugging Face ecosystem. One static library or a two-file amalgam. MIT.

## Features

- **GGUF v2/v3**: typed metadata KV store (scalars, arrays, vocabularies),
  split files (`-00001-of-00003.gguf`), zero-copy `mmap` plus a `pread` path
  for platforms where mapping is not an option.
- **safetensors**: single file, sharded directory
  (`model.safetensors.index.json`, with glob fallback) or an explicit shard
  list; `__metadata__` exposed; `F32/F16/BF16/F8_E4M3/F8_E5M2/I8..I64/U8..U64/BOOL`.
- **All 33 ggml quantized block types decode** — the K-quants, the full
  `IQ1`–`IQ4` codebook family, ternary `TQ`, microscaling `MXFP4`/`NVFP4` —
  verified bit-for-bit against llama.cpp. See [docs/QUANTS.md](docs/QUANTS.md).
- **SIMD kernels**: matvec and batched matmat for
  `Q2_K Q3_K Q4_K Q5_K Q6_K Q8_0`, with NEON, AVX2, ARM SDOT/SMMLA and
  AVX-512 VNNI paths selected at runtime; dense **BF16/F16** matvec/matmat
  multiply straight through the stored bytes (widen in-register, no f32
  conversion pass). `ingot_matvec(type, …)` / `ingot_matmat(…)` use the
  specialized kernel when one exists and decode row-by-row otherwise, so
  callers never branch on the format.
- **Fast loads**: BF16/F16→F32 bulk conversion and the hot dequants
  (`Q4_K Q5_K Q6_K Q8_0`) are vectorized on NEON, AVX2 and AVX-512 —
  dequant-at-load engines stop walking a 20 GB checkpoint one element at a
  time.
- **Writers** for both containers, including quantize-on-write: hand in f32
  and a target type (`F16 BF16 F32 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q6_K`).
- **Page-cache control**: `prefault`, `dontneed`, `drop_cache` for large
  checkpoints on shared/unified memory.
- **O(1) lookup** over tensor names and metadata keys (FNV-1a index).
- **Predictable behavior**: errors come back as strings and nothing is ever
  written to stderr; quantized tensors are returned as stored — dequantization
  is an explicit call; no hidden allocations; malformed or unknown input fails
  with a precise message (`IQ4_XS` rather than `unknown type 23`), never a
  silent truncation.

## Quick start

```c
#include <ingot/gguf.h>

char err[256];
ingot_gguf *g;
if (ingot_gguf_open(&g, "model.gguf", err, sizeof err) != 0) {
    fprintf(stderr, "%s\n", err);   /* errors are returned, never printed */
    return 1;
}
const ingot_tensor *t = ingot_gguf_find(g, "blk.0.attn_q.weight");  /* O(1) */
const void *weights = ingot_gguf_data(g, t);                        /* mmap  */
ingot_q4_k_matvec(weights, rows, cols, x, y);                       /* no copy */
ingot_gguf_close(g);
```

```c
#include <ingot/safetensors.h>

ingot_st *st;
ingot_st_open_dir(&st, "./Qwen3-8B", err, sizeof err);   /* index.json, shards, or a single file */
const ingot_st_tensor *e = ingot_st_find(st, "model.embed_tokens.weight");
const uint16_t *bf16 = ingot_st_data(st, e);
```

`examples/minimal.c` shows the full flow in 80 lines: open either format,
find a tensor, multiply a vector through it regardless of its quantization.

## Using it in your project

Build the static library once and link it — no build system, no configure:

```sh
make -C ingot lib
cc -std=c11 -O2 -Iingot/include app.c ingot/libingot.a -lpthread -lm
```

Or copy the **two-file amalgam** in and compile the `.c` like any other
source — zero build integration:

```sh
cp ingot/amalgam/ingot.h ingot/amalgam/ingot.c yourproject/vendor/
cc -std=c11 -O2 yourproject/vendor/ingot.c ... -lpthread -lm
```

Good to know:

- Requirements are just C11, POSIX and `-lpthread -lm`; the header also
  parses as C++.
- `-DINGOT_NO_KERNELS` drops the quantization half (141 KB → 70 KB of object
  code) for tools that only read containers.
- Only the container half is mandatory (`dtype.c`, `gguf.c`,
  `safetensors.c`, `wfile.c`, `write.c`); `cpu.c`, `dequant.c`, `kernels.c`,
  `generic.c` and `quantize.c` are optional. `cc -Iinclude -c src/*.c` works
  if you want no Makefile at all.
- Vendoring the whole repo (e.g. as a `git subtree` under `third_party/`)
  works well too: have your Makefile run `make -C third_party/ingot lib` and
  link the result.

## Documentation

| | |
|---|---|
| [docs/QUANTS.md](docs/QUANTS.md) | every quantization format, support status, bits/weight |
| [include/ingot/gguf.h](include/ingot/gguf.h) | GGUF reader: tensors, metadata KVs, split files |
| [include/ingot/safetensors.h](include/ingot/safetensors.h) | safetensors reader: shards, `index.json`, page-cache control |
| [include/ingot/quant.h](include/ingot/quant.h) | decode, encode, kernels, precision contract, threads |
| [include/ingot/wfile.h](include/ingot/wfile.h) | one handle for either container |
| [include/ingot/write.h](include/ingot/write.h) | writing both formats |
| [examples/minimal.c](examples/minimal.c) | the whole library in 80 lines |

The headers are the reference: each one documents its API and its limits.

## Build & test

```sh
make                # libingot.a, ingot-dump, examples
make test           # full suite, no model files and no Python needed
make help           # every target, one line each
```

The Makefile adds no `-march` flags of its own: kernels compile for the
compiler's default target — on aarch64 that already includes NEON, but on
x86-64 the default is baseline SSE2, which leaves every AVX lane dark. To
build and test the AVX2/AVX-512 paths, name the target explicitly, e.g.
`make test CFLAGS="-O2 -march=x86-64-v4 -mavx512vnni"`, and use
`INGOT_CAPS=scalar|avx2|vnni` (with `neon`/`dotprod` as ARM aliases) to pin
the runtime dispatch below what the CPU offers.

`make core-only` builds the readers without the quantization half and proves
the split on every commit. `amalgam/` is generated from `src/` by
`make amalgam`, and `make amalgam-test` runs the entire suite against the
generated pair.

| target | |
|---|---|
| `make test-asan` | suite under AddressSanitizer + UBSan (Linux) |
| `make test-leaks` | suite under macOS `leaks` (the memory gate on a Mac, where ASan tends to hang) |
| `make fuzz` / `fuzz-leaks` | mutation-fuzz both readers (`ROUNDS=n`) |
| `make amalgam` / `amalgam-test` | regenerate the two-file build and test it |
| `make gen-tables` / `gen-fixtures` | re-derive IQ codebooks and oracle fixtures from llama.cpp |
| `make check-real MODEL=…` | cross-check a real checkpoint against an independent parse |

## Inspecting a file

```sh
$ ingot-dump model.gguf
GGUF v3  model.gguf
  architecture: llama
  alignment:    32
  shards:       1
  metadata:     26 keys
  tensors:      291

type census:
  F32         81 tensors     52.14 MiB
  Q4_K       189 tensors   3821.02 MiB
  Q6_K        21 tensors    412.66 MiB
  TOTAL      291 tensors   4285.82 MiB
```

`-v` lists every tensor and metadata key. Pointed at a directory, it resolves
the safetensors shards.

## Precision contract

From two tokens up, the batched `Q4_K`/`Q5_K` kernels quantize the
**activations** to int8 by default: 1.5–2.5× faster, at ~2.4e-3 relative error
instead of the last few ulp of a reordered sum. Everything else is an exact
reorder. The library reports which mode applies, so parity gates can pick the
right tolerance:

```c
const double budget = ingot_matmat_is_exact(tokens) ? 1e-5 : 5e-3;
```

`ingot_q4_k_matmat_exact()` / `ingot_q5_k_matmat_exact()` are always exact,
and `INGOT_SDOT=0` in the environment disables the int8 path globally.
`ingot_has_kernel(type)` reports whether a type has a specialized kernel.

## Threads

ingot does not own a thread pool — consumers already have one. Hand yours in;
the default runs inline:

```c
static void my_pool(size_t n, ingot_range_fn fn, void *user) { /* ... */ }
ingot_set_parallel_for(my_pool);
```

## Non-goals

ingot loads weights; it is not an inference framework. No compute graph, no
tensor arithmetic or broadcasting, no allocator or arena, no tokenizer (the
GGUF metadata that carries one is fully readable), and no imposed memory
layout for kernel data — a Metal simdgroup and a CUDA row-split want different
layouts, and that choice belongs to the engine.

## Testing

`make test` builds synthetic containers in a temp directory and checks both
directions: valid files parse, and malformed ones — bad magic, truncated
headers, payloads past EOF, byte counts that disagree with shape × dtype,
alignments that are not powers of two, duplicate tensor names across shards —
are rejected cleanly, with a message and without a crash.

Correctness is anchored outside the library, so a shared misreading of a spec
cannot self-validate:

- `tests/test_oracle.c` decodes fixtures whose expected values come from
  llama.cpp's own `gguf` package — bit-for-bit on all 23 decodable block
  types.
- `tools/check_against_python.py` reparses real safetensors checkpoints with
  nothing but `struct` + `json` and compares every tensor.
- All 34 block geometries are cross-checked against ggml's
  `GGML_QUANT_SIZES` as a standing test.

`make fuzz` mutation-fuzzes both readers (thousands of rounds; the mutants
that still parse are the valuable ones, since they exercise the accept path
with values no writer produces). `make test-asan` is the sanitizer gate on
Linux, `make test-leaks` the one on macOS.

The SIMD matrix is validated on real silicon at both ends, not only under
emulation in CI: Apple M-series (NEON, SDOT/SMMLA) and AMD Zen 5 (AVX2,
AVX-512 VNNI) — full suite, sanitizers, fuzz and `check-real` at every
`INGOT_CAPS` level. An alignment-invariance test places the same quantized
bytes at data offsets +0/32/64/96 and requires byte-identical
matvec/matmat/dequant output from every lane, closing off the class of bug
where results depend on where a tensor happens to land in the file.

## Portability and limits

- Both formats are little-endian by definition and parsing is
  host-independent; bulk conversion of 4- and 8-byte element types currently
  assumes a little-endian host (fine on x86-64 and aarch64).
- `Q1_0` opens and its bytes are accounted, but it has no decoder: llama.cpp's
  reference package has none either, so there is nothing to verify against.
  Details in [docs/QUANTS.md](docs/QUANTS.md).
- gcc 15.2 miscompiles the *combined* `-fsanitize=address,undefined` build
  when AVX-512 codegen is enabled: a spurious SEGV in the quantizer at `-O1`
  and above, with each sanitizer alone green and `-O0` green (reduced to a
  20-line repro, toolchain bug — not an ingot defect). When sanitizing with
  an AVX-512 `-march` under gcc 15, run ASan and UBSan as separate builds.

## License

MIT — see [LICENSE](LICENSE). The ggml block layouts are ggml's (MIT); the
decoders are written from the spec and checked against it.
