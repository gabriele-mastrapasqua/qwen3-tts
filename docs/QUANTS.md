# GGUF quantization formats in ingot

**Short version: every ggml block format has a geometry here, and all but one
have a decoder that matches llama.cpp bit for bit.** Run `ingot-dump` on a file
and it will tell you exactly what is inside.

---

## First: `Q4_K_M` is not a format

This trips everyone once, and it decides most of what you actually need to
know. A file called `model-Q4_K_M.gguf` does not contain a "Q4_K_M" type —
there is no such tensor type. The suffix names a **recipe**: which of the real
block types llama.cpp assigned to which tensors.

A typical `Q4_K_M` file holds `Q4_K` for most weights, `Q6_K` for
`attn_v`/`ffn_down`, and `F32` for norms and biases. `Q4_K_S` uses the same
types with a stingier recipe. Unsloth's `UD-Q4_K_XL`, bartowski's `-L` variants
and the rest are the same idea with different assignments.

So the question "does ingot read Q4_K_M?" is really "does ingot read every type
that recipe uses", and the table below answers it. In practice:

| file name you downloaded | types inside | ingot |
|---|---|---|
| `Q2_K`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K`, `Q8_0` | K-quants + `Q8_0` + `F32` | ✅ read, multiply, no dequantization needed |
| `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` (legacy) | the 32-value blocks | ✅ |
| `IQ1_S`, `IQ1_M`, `IQ2_XXS…IQ3_M`, `IQ4_XS`, `IQ4_NL` | the codebook families | ✅ read and decode |
| `UD-Q4_K_XL` and friends | mixes of the above | ✅ — it is a recipe, not a type |
| `TQ1_0`, `TQ2_0` (ternary / BitNet) | ternary blocks | ✅ |
| `MXFP4` (gpt-oss) | microscaling FP4 | ✅ |
| anything using `Q1_0` | `Q1_0` | ⚠️ opens, reports the type, cannot decode |

`ingot-dump model.gguf` prints a per-type census and flags anything it cannot
decode by name, so you never have to guess.

---

## The full type table

`bits/weight` is `bytes × 8 ÷ block`, the real on-disk cost including scales.

**decode** — turn blocks into f32.
**encode** — turn f32 into blocks (what a converter needs).
**kernel** — multiply straight off the quantized bytes, no dequantization.
Types without a kernel still work through `ingot_matvec(type, …)`, which
decodes a row at a time; you lose the SIMD inner loop, not the result.

### 32-value blocks (the legacy set)

| type | id | bytes | bits/w | decode | encode | kernel |
|---|---|---|---|---|---|---|
| `Q4_0` | 2 | 18 | 4.50 | ✅ | ✅ | generic |
| `Q4_1` | 3 | 20 | 5.00 | ✅ | ✅ | generic |
| `Q5_0` | 6 | 22 | 5.50 | ✅ | ✅ | generic |
| `Q5_1` | 7 | 24 | 6.00 | ✅ | ✅ | generic |
| `Q8_0` | 8 | 34 | 8.50 | ✅ | ✅ | ✅ SIMD |
| `Q8_1` | 9 | 40 | 10.00 | ✅ | — | generic |

`Q8_1` is an activation-side intermediate; no checkpoint stores it. Note the
40 bytes: ggml widened `d`/`s` from f16 to f32 at some point, and ingot had it
at 36 until the geometry cross-check caught it.

### 256-value super-blocks (the K-quants)

| type | id | bytes | bits/w | decode | encode | kernel |
|---|---|---|---|---|---|---|
| `Q2_K` | 10 | 84 | 2.63 | ✅ | — | ✅ SIMD |
| `Q3_K` | 11 | 110 | 3.44 | ✅ | — | ✅ SIMD |
| `Q4_K` | 12 | 144 | 4.50 | ✅ | ✅ | ✅ SIMD + int8 |
| `Q5_K` | 13 | 176 | 5.50 | ✅ | — | ✅ SIMD + int8 |
| `Q6_K` | 14 | 210 | 6.56 | ✅ | ✅ | ✅ SIMD |
| `Q8_K` | 15 | 292 | 9.13 | ✅ | — | generic |

These are what a modern `Q4_K_M` download is made of. `Q4_K` and `Q5_K` have an
int8 activation path for batched work — see [the precision
contract](../README.md#the-precision-contract) before you use one as a
reference for a GPU comparison.

### Codebook families (the IQ set)

These do not compute values from a formula: each indexes a hand-trained lookup
grid, plus a sign codebook. That is what buys them sub-2-bit rates.

| type | id | bytes | bits/w | grid | decode | encode | kernel |
|---|---|---|---|---|---|---|---|
| `IQ1_S` | 19 | 50 | 1.56 | 2048 × 8 | ✅ | — | generic |
| `IQ1_M` | 29 | 56 | 1.75 | 2048 × 8 | ✅ | — | generic |
| `IQ2_XXS` | 16 | 66 | 2.06 | 256 × 8 | ✅ | — | generic |
| `IQ2_XS` | 17 | 74 | 2.31 | 512 × 8 | ✅ | — | generic |
| `IQ2_S` | 22 | 82 | 2.56 | 1024 × 8 | ✅ | — | generic |
| `IQ3_XXS` | 18 | 98 | 3.06 | 256 × 4 | ✅ | — | generic |
| `IQ3_S` | 21 | 110 | 3.44 | 512 × 4 | ✅ | — | generic |
| `IQ4_XS` | 23 | 136 | 4.25 | 16 levels | ✅ | — | generic |
| `IQ4_NL` | 20 | 18 | 4.50 | 16 levels | ✅ | — | generic |

The grids are **not typed by hand**: `tools/gen_iq_tables.py` extracts them
from llama.cpp's own `gguf` package into `src/iq_tables.inc` (200 KiB of
generated source, ~33 KiB of `int8` at runtime). That script is why this family
took until now: a wrong entry in a lookup table produces plausible numbers, not
a crash, and no round-trip test catches a self-consistent mistake.

### Ternary and microscaling

| type | id | block | bytes | bits/w | decode | encode | kernel |
|---|---|---|---|---|---|---|---|
| `TQ1_0` | 34 | 256 | 54 | 1.69 | ✅ | — | generic |
| `TQ2_0` | 35 | 256 | 66 | 2.06 | ✅ | — | generic |
| `MXFP4` | 39 | 32 | 17 | 4.25 | ✅ | — | generic |
| `NVFP4` | 40 | 64 | 36 | 4.50 | ✅ | — | generic |
| `Q1_0` | 41 | 128 | 18 | 1.13 | ❌ | — | — |

`TQ1_0` packs five base-3 digits per byte — the trick is a multiply by a power
of three that wraps in eight bits, then a fixed-point divide. `MXFP4` is the
OCP microscaling format gpt-oss ships in.

**`Q1_0` is the one gap.** llama.cpp's own reference package carries no decoder
for it either, so there is nothing to check an implementation against, and
writing one would be guesswork. It still has a geometry: a file using it opens,
its bytes account exactly, and the error names the type instead of saying
"unknown type 41". When a reference appears, the work is an afternoon.

### Not block formats

`F32 F16 BF16 F64 I8 I16 I32 I64` decode (and `F32 F16 BF16` encode) through
the same calls. On the safetensors side you also get `F8_E4M3`, `F8_E5M2`,
`U8 U16 U32 U64` and `BOOL`.

---

## How any of this is trustworthy

Three independent checks, all in `make test`:

1. **Geometry against ggml itself.** `tests/fixtures/geometry.txt` is generated
   from `GGML_QUANT_SIZES`, and the test compares all 34 types. This is what
   caught `Q8_1` at 36 bytes when ggml says 40 — a silent mis-sizing of every
   tensor of that type.
2. **Values against llama.cpp's dequantizer.** `tests/test_oracle.c` decodes
   fixtures whose expected output came from the `gguf` package: a different
   implementation, in a different language, by the people who define the
   format. All 23 decodable block types currently match **bit for bit**
   (`worst rel 0.00e+00`), on pseudo-random blocks that walk the whole
   codebook rather than the corner a smooth signal would touch.
3. **The two internal decoders against each other.** `src/dequant.c` (scalar
   reference) and `src/kernels.c` (the fast paths) decode the six K-quants
   independently. This is what caught a `Q6_K` decoder reading `d` from the
   first two bytes when ggml puts it in the last two.

Regenerate the derived files with `make gen-tables` and `make gen-fixtures`
(both need the `gguf` package). Both outputs are committed, so a plain checkout
builds and tests without Python.

---

## Choosing a format when you are the one writing the file

ingot encodes `F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q6_K`. Measured
round-trip relative L2 on a Gaussian weight distribution (`make test` prints
these on every run, so drift is visible):

| type | rel L2 | bits/w | note |
|---|---|---|---|
| `Q8_0` | 0.005 | 8.50 | effectively lossless, halves an f16 file |
| `Q6_K` | 0.019 | 6.56 | the usual choice for output/embedding layers |
| `Q5_1` | 0.037 | 6.00 | best of the legacy 32-value set |
| `Q5_0` | 0.042 | 5.50 | |
| `Q4_K` | 0.073 | 4.50 | same bits as `Q4_0`, visibly better |
| `Q4_1` | 0.078 | 5.00 | |
| `Q4_0` | 0.083 | 4.50 | keep for compatibility, not for quality |

`Q4_K` and `Q6_K` also have hand-written SIMD kernels, so they are the two to
reach for if the file is going to be multiplied rather than just stored.
`ingot_has_kernel(type)` answers that at runtime.

---

## The API, in one place

```c
int ingot_type_geometry(int type, uint64_t *block_elems, uint64_t *block_bytes);
int ingot_type_nbytes(int type, uint64_t nelem, uint64_t *out);
int ingot_type_can_dequant(int type);      /* before you rely on a decode  */
int ingot_has_kernel(int type);            /* fast path or generic path    */
int ingot_can_quantize(int type);          /* before you pick a target     */
const char *ingot_type_name(int type);     /* for the error message        */

int ingot_dequant(int type, const void *src, size_t nelem, float *dst);
int ingot_quantize(int type, const float *values, size_t count, void *out);
int ingot_matvec(int type, const void *w, size_t rows, size_t cols,
                 const float *x, float *y);
```

Full signatures in [`include/ingot/quant.h`](../include/ingot/quant.h) and
[`include/ingot/dtype.h`](../include/ingot/dtype.h).
