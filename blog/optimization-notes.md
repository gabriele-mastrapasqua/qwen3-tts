# From RTF 3.5 to RTF 1.26: Optimizing a Pure C TTS Engine

*How cache alignment, NEON intrinsics, pipeline threading, and lessons from 1990s game programming nearly tripled our inference speed.*

## The Starting Point

We have a pure C inference engine for Qwen3-TTS, a text-to-speech model with
a 28-layer transformer (Talker), a 5-layer code predictor, and a convolutional
speech decoder. No Python, no PyTorch, no GPU — just C, Apple Accelerate BLAS,
and NEON intrinsics on an Apple M1 with 16 GB RAM.

After getting the pipeline correct and implementing the first round of NEON
kernels (fused 2-row bf16 matvec, unified QKV dispatch, fused gate+up SwiGLU),
we were at **RTF ~3.5** on short text and **RTF ~2.5** on longer text (the fixed
costs of prefill and speech decoding amortize over longer audio).

This post covers the optimizations that brought us to **RTF ~1.26** (server warm,
long text) — up to a 2.7x total speedup with zero algorithmic changes and zero
new dependencies.

> **RTF** = Real-Time Factor = processing_time / audio_duration. Lower is better.
> RTF < 1.0 means faster than real-time.

## The Abrash Instinct: Cache Alignment Still Matters

If you grew up reading Michael Abrash's *Graphics Programming Black Book*
(1997), you remember the chapters on data alignment. Abrash hammered on a
simple point: on the 386 and 486, unaligned memory accesses caused extra
wait-states that destroyed performance. Word-aligned on 386, dword-aligned
on 486 — he had the diagrams, the tables, the rules.

John Carmack talked about this too in his `.plan` files and QuakeCon talks,
in his typical informal way — "align your structs, pack your data, think about
cache lines." But the systematic treatment, the benchmarks, the rules of thumb?
That was Abrash. Chapter after chapter of the Black Book devoted to data
alignment, struct layout, and how the CPU bus punishes you for sloppy memory
access patterns.

Here's the thing: **those lessons still apply.** The penalty isn't wait-states
anymore — it's SIMD throughput. Modern CPUs like the Apple M1 have 128-bit
NEON units that operate on 16-byte-aligned data natively. When you feed
misaligned buffers to BLAS routines like `cblas_sgemm`, the library can't
use its fastest SIMD paths. Apple Accelerate checks alignment at runtime and
falls back to slower code when buffers aren't aligned.

### The Fix: 3 Lines of Code

```c
static inline void *aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
}
```

We replaced every `malloc()` and `calloc()` in the hot path with
`posix_memalign(64, ...)`. Not just the BLAS buffers — the KV caches, the
decode buffers, the prefill temporaries. Everything that touches a SIMD
instruction or a BLAS call got 64-byte alignment (one cache line on M1).

### The Result: 24% Total Speedup

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Prefill (BLAS sgemm) | 475ms | 260ms | **84%** |
| Speech Decoder (BLAS sgemm) | 2,580ms | 1,648ms | **36%** |
| Code Predictor (NEON matvec) | 66.4 ms/f | 60.8 ms/f | **9%** |
| **Total pipeline** | **10.4s** | **7.9s** | **24%** |

The prefill stage — which does batch matrix multiplication via `cblas_sgemm` —
nearly doubled in speed. The speech decoder, which also relies heavily on
sgemm for its convolutions, improved by 36%. Even the Code Predictor, which
uses our hand-written NEON bf16 matvec kernel, gained 9% from aligned KV
cache and decode buffers.

And the output is **bit-identical**. Same seed, same text, same bytes in the
WAV file. This is pure implementation overhead we were leaving on the table.

### Why 64 Bytes?

The M1's L1 cache line is 128 bytes on the P-cores, but the common denominator
across ARM and x86 is 64 bytes. The BLAS library needs at least 16-byte
alignment for NEON (32-byte for AVX), but 64 bytes guarantees that no buffer
straddles a cache line boundary unnecessarily. It's the sweet spot for
cross-platform code.

`posix_memalign` is POSIX standard — it works on Linux and macOS without any
platform-specific code. On Windows/WSL2, it's available too. Three lines of
code, zero dependencies, cross-platform.

## The Speech Decoder: Scalar Code Hiding in Plain Sight

After the alignment win, we profiled again. The speech decoder still took
~1,650ms for 62 frames. Digging into the code, we found something
embarrassing: **six scalar RMSNorm loops** that were never converted to NEON.

The Talker and Code Predictor used our NEON-optimized `qwen_rms_norm()`
function. But the speech decoder had its own hand-written scalar version:

```c
// Before: scalar, called 480 times per generation (60 frames x 8 layers)
for (int s = 0; s < n_frames; s++) {
    float sum_sq = 0;
    for (int i = 0; i < 512; i++) sum_sq += xs[i] * xs[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / 512 + eps);
    for (int i = 0; i < 512; i++) xn[i] = xs[i] * inv_rms * weight[i];
}

// After: one line
qwen_rms_norm(x_norm, hidden, l->attn_norm, n_frames, dec_hidden, eps);
```

The NEON version processes 8 floats per iteration with fused multiply-accumulate,
versus one float at a time in the scalar version.

Same story with RoPE (rotary position embeddings) — the speech decoder had a
scalar loop doing paired rotations at 32 elements per head. We replaced it
with NEON intrinsics that process 4 pairs at once, fusing Q and K rotation
in the same pass:

```c
// NEON: 4-wide fused Q+K rotation
float32x4_t c = vld1q_f32(cos_ptr + i);
float32x4_t si = vld1q_f32(sin_ptr + i);
float32x4_t q1 = vld1q_f32(qh + i), q2 = vld1q_f32(qh + i + half);
float32x4_t k1 = vld1q_f32(kh + i), k2 = vld1q_f32(kh + i + half);
vst1q_f32(qh + i,        vmlsq_f32(vmulq_f32(q1, c), q2, si));
vst1q_f32(qh + i + half, vmlaq_f32(vmulq_f32(q2, c), q1, si));
vst1q_f32(kh + i,        vmlsq_f32(vmulq_f32(k1, c), k2, si));
vst1q_f32(kh + i + half, vmlaq_f32(vmulq_f32(k2, c), k1, si));
```

We also replaced the scalar attention dot-product loop with our NEON-optimized
windowed causal attention kernel — online softmax with 16-element-wide dot
products and fused V accumulation.

And the VQ dequantization step, which did per-frame scalar matrix-vector
products for codebook projection, was batched into a single `cblas_sgemm`
call across all frames.

**Combined result: speech decoder 11% faster** (1,446ms to 1,288ms).

## Eliminating Per-Token Malloc

The generation loop was doing malloc/free for every token:

- `topk_filter()`: `malloc(vocab_size * sizeof(float))` + `free()` per sample
- `topp_filter()`: `malloc(vocab_size * sizeof(int))` + `free()` per sample
- `embed_one_text_token()`: two `malloc(text_hidden * sizeof(float))` + `free()` per text token
- `qwen_talker_prefill()`: 14 large buffers allocated and freed per generation

For a typical generation of 60 frames, that's ~120 malloc/free pairs just for
sampling, plus ~14 large buffer allocations for prefill.

We pre-allocated everything:
- Sampling buffers persist as module-level statics (allocated once on first call)
- Text embedding temps stored in the context struct
- Prefill buffers (including ~50MB of f32 weight conversion temps) persist across
  generations

The single-run impact is negligible (<1%), but in **server mode**, where the
model handles many sequential requests, the second request runs **38% faster**
because all buffers are warm in cache and no allocation overhead.

The generation loop now has **zero per-token malloc calls**.

## Text Embedding Cache: Avoiding Redundant Work

Each text token goes through a two-layer MLP projection (bf16 lookup → fc1 2048×2048
SiLU → fc2 1024×2048) — about 12 million FLOPs per token. For a 57-token long prompt,
that's ~29ms of pure compute. On a server handling the same or similar requests, this
is entirely redundant.

We added two levels of caching:

**Special token cache** (computed once at model load): `tts_pad`, `tts_bos`, and
`tts_eos` are used in *every* request. Pre-computing them at load time eliminates
3 matvec pairs per generation — trivial change, zero runtime cost.

**LRU hash map** for all text tokens: An open-addressing hash table maps `token_id →
float[hidden]` with 2048 slots. On a cache hit, a single 4KB memcpy replaces two bf16
matrix-vector multiplications. The table uses Knuth multiplicative hashing with linear
probing and LRU eviction when full.

Memory cost: 2048 × 1024 × 4 bytes = **8MB** — negligible compared to the ~1.2GB
model weights. Always active (both CLI and server) since the overhead is near-zero.

**Result: 14% faster on long-text server cold call** (RTF 1.55 → 1.33). On warm calls
the improvement is smaller (~2%) because subsequent requests already benefit from OS
page cache and buffer reuse.

## Decoder Thread: Pipeline Parallelism

The TTS pipeline has three stages: Talker generates a codec token, the Code Predictor
fills in 15 more codebook entries, then the speech decoder converts those codes to
audio. The original code ran these strictly sequentially — the speech decoder waited
until ALL frames were generated, then processed everything in one batch.

But the speech decoder is completely independent of the Talker and Code Predictor.
It reads completed codec frames and writes audio. No shared weights, no shared KV
cache. And it's already designed for incremental operation: the pre-transformer uses
sliding-window causal attention (window=72), and the ConvNet is fully causal.

The fix: a producer-consumer pipeline with two threads:

```
Main thread:    [Talker → CP → push frame] → [Talker → CP → push frame] → ...
Decoder thread: [wait] → [decode chunk] → [wait] → [decode chunk] → ...
```

The main thread pushes completed frames to a mutex-guarded queue. The decoder thread
wakes on a condition variable, pulls available frames, and decodes them incrementally
using the existing streaming decoder path. At the end, the main thread joins the
decoder thread and collects the accumulated audio.

~150 lines of pthreads code: mutex + condvar queue, producer push, consumer loop,
join + audio collection.

### The Result

| Mode | Before | After | Improvement |
|------|--------|-------|-------------|
| CLI short (~5s audio) | RTF 2.01 | RTF 1.74 | **14%** |
| Server short cold | RTF 1.85 | RTF 1.50 | **19%** |
| Server long warm | RTF 1.31 | **RTF 1.26** | **4%** |

The gain is largest on short text where the speech decoder is a bigger fraction of
total time. On long text, Talker+CP dominate and the decoder overlap has less to
hide. The "drain" at the end (waiting for the decoder to finish its last chunk) is
only ~500ms on short text.

One trade-off: the decoder thread competes with the main thread for CPU cores and
memory bandwidth. Talker+CP ms/frame increases slightly (~10%) due to contention,
but the net wall-time improvement from overlapping far exceeds this cost.

## What We Analyzed and Skipped

Not every optimization idea pans out. Here's what we investigated and rejected:

**Struct field reordering** (est. 3-7%, actual: 0%). The `qwen_tts_ctx_t`
struct is 7.6 KB spanning 119 cache lines. We built a layout analyzer and
found that the hot decode fields (KV cache pointers, decode buffers) already
sit on adjacent cache lines 112-118. More importantly, the struct is accessed
via pointer indirection — the CPU loads the pointer once and the struct stays
in L1. The bottleneck is the data these pointers *reference* (multi-MB weight
matrices), not the 8-byte pointer loads themselves.

**L1 cache blocking for matvec** (est. 3-5%, actual: not worth the complexity).
Our bf16 matvec kernel already processes 2 rows at a time with 8 NEON
accumulators, doing 32 elements per inner loop iteration. The input vector
(4 KB for hidden=1024) fits entirely in L1. The weight matrix access is
sequential, which the hardware prefetcher handles well. The bottleneck is
main memory bandwidth (~10 GB/s effective out of 68 GB/s peak), not cache
misses.

**Prefetch hints in CP loop** (est. 0.5-1%, actual: not possible). Each Code
Predictor layer has ~26 MB of weights. The M1's shared L2 is 12 MB. You
can't prefetch what doesn't fit. The hardware prefetcher handles sequential
access within each matvec just fine — it's the layer *transitions* that cause
cold misses, and those are unavoidable without smaller weights.

**INT4/INT8 quantization on 0.6B** (tested, slower or neutral). The hidden
dimension of 1024 produces matrices too small to be bandwidth-bound. The
dequantization overhead (3 NEON ops for INT8, 8 ops for INT4) exceeds the
bandwidth savings. BF16-to-f32 conversion is essentially free — a single
`vshll` instruction. Quantization might help on the 1.7B model where matrices
are 4x larger.

## The Numbers

| Metric | Baseline | After all optimizations |
|--------|----------|------------------------|
| Talker | 46.9 ms/f | ~22 ms/f |
| Code Predictor | 104.7 ms/f | ~60 ms/f |
| Speech Decoder | ~2,600ms (blocking) | overlapped (background thread) |
| Prefill | ~1,800ms | ~1,000–1,600ms |
| **RTF (CLI, short ~5s audio)** | **~3.5** | **1.74** |
| **RTF (CLI, long ~17s audio)** | **~2.5** | **~1.4** |
| Per-token malloc calls | ~120+ | **0** |
| **RTF (server warm, short)** | — | **1.39** |
| **RTF (server warm, long)** | — | **1.26** |

All on an Apple M1 8-core, 16 GB RAM, 4 threads. RTF improves with longer
audio because prefill is a fixed cost that amortizes over more frames. The
speech decoder runs in a background thread, overlapping most of its work with
generation. Server mode with embedding cache, warm buffers, and decoder thread
overlap delivers the best RTF at **1.26**.

## Lessons

1. **Alignment matters more than you think.** A 24% speedup from
   `posix_memalign` is absurd in 2026, but BLAS libraries really do check
   alignment and choose different code paths. Abrash was right in 1997 and
   he's right now.

2. **Profile before you optimize.** We nearly implemented L1 cache blocking
   for the matvec kernel — a complex change — before realizing the kernel was
   already bandwidth-bound and the complexity would gain nothing.

3. **Look for scalar code in SIMD codebases.** When different components are
   written at different times, it's easy for one file to miss the optimization
   that all others have. Six scalar RMSNorm loops hiding in the speech decoder.

4. **Zero-malloc decode loops matter for servers.** The single-run difference
   is negligible, but for a long-running server handling request after request,
   eliminating allocation churn in the hot loop adds up.

5. **Cache computed results, not just data.** The LRU text embedding cache
   avoids recomputing token projections (12M FLOPs each) across requests. At
   8MB for 2048 tokens, it's practically free. The lesson: when you spot a
   pure function called repeatedly with the same inputs, memoize it.

6. **Pipeline independent stages.** The speech decoder doesn't share any state
   with the Talker or Code Predictor. Once we recognized that, overlapping them
   with a simple producer-consumer thread was ~150 lines for a 14-19% speedup.
   Look for stages in your pipeline that only consume the output of previous
   stages — those are free parallelism.

7. **Read the old books.** Abrash's *Graphics Programming Black Book* and
   Carmack's `.plan` files are from another era, but the principles — cache
   friendliness, data alignment, knowing your memory hierarchy — are timeless.
   The specific rules change (64-byte cache lines instead of dword alignment),
   but the instinct to think about how data flows through the CPU is exactly
   the same.

---

*This is part of the [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts)
project — a pure C inference engine for Qwen3-TTS text-to-speech models.*
