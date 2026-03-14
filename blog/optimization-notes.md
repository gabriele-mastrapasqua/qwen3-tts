# From RTF 3.5 to RTF 1.26: Optimizing a Pure C TTS Engine

*How cache alignment, SIMD intrinsics (NEON/AVX), pipeline threading, algorithm fixes, and lessons from 1990s game programming nearly tripled our inference speed.*

## The Starting Point

We have a pure C inference engine for Qwen3-TTS, a text-to-speech model with
a 28-layer transformer (Talker), a 5-layer code predictor, and a convolutional
speech decoder. No Python, no PyTorch, no GPU — just C, Apple Accelerate BLAS,
and SIMD intrinsics (NEON on ARM, AVX on x86) on an Apple M1 with 16 GB RAM.

After getting the pipeline correct and implementing the first round of SIMD
kernels (NEON/AVX: fused 2-row bf16 matvec, unified QKV dispatch, fused gate+up SwiGLU),
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
anymore — it's SIMD throughput. Modern CPUs have SIMD units (128-bit NEON on
ARM, 256-bit AVX on x86) that operate on aligned data natively. When you feed
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
| Code Predictor (SIMD matvec) | 66.4 ms/f | 60.8 ms/f | **9%** |
| **Total pipeline** | **10.4s** | **7.9s** | **24%** |

The prefill stage — which does batch matrix multiplication via `cblas_sgemm` —
nearly doubled in speed. The speech decoder, which also relies heavily on
sgemm for its convolutions, improved by 36%. Even the Code Predictor, which
uses our hand-written SIMD bf16 matvec kernel (NEON/AVX), gained 9% from aligned KV
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
embarrassing: **six scalar RMSNorm loops** that were never converted to SIMD.

The Talker and Code Predictor used our SIMD-optimized `qwen_rms_norm()`
(NEON on ARM, AVX on x86)
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

The SIMD version (NEON on ARM, AVX on x86) processes 4-8 floats per iteration
with fused multiply-accumulate, versus one float at a time in the scalar version.

Same story with RoPE (rotary position embeddings) — the speech decoder had a
scalar loop doing paired rotations at 32 elements per head. We replaced it
with SIMD intrinsics that process 4 pairs at once, fusing Q and K rotation
in the same pass (shown here with NEON; AVX variant in `qwen_tts_kernels_avx.c`):

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

We also replaced the scalar attention dot-product loop with our SIMD-optimized
windowed causal attention kernel (NEON/AVX) — online softmax with wide dot
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

## Quickselect: When the Algorithm Is the Bug

After all the SIMD and threading work, we noticed the "Codec head+sampling"
line in the timing report: **93ms** for 101 frames. That's almost 1ms per frame
spent on... sampling? Something was off.

The top-k filter used **selection sort** to find the k-th largest logit:

```c
// O(k × n) — selection sort to find top-k threshold
for (int i = 0; i < k; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++)
        if (tmp[j] > tmp[max_idx]) max_idx = j;
    float t = tmp[i]; tmp[i] = tmp[max_idx]; tmp[max_idx] = t;
}
```

With `k=50` and `n=3072` (codec vocabulary), that's **153,600 comparisons per
frame** × 101 frames = 15.5M comparisons. It's technically O(kn), but the
constant is awful.

The fix: **quickselect** (Hoare's algorithm). It finds the k-th element in
O(n) average time using 3-way partitioning:

```c
static float quickselect_kth_largest(float *arr, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float pivot = arr[lo + (hi - lo) / 2];
        // 3-way partition: [>pivot] [==pivot] [<pivot]
        // ...
    }
    return arr[lo];
}
```

**Result: 93ms → 21ms (4.4× faster).** Output bit-identical — same threshold,
same filtering, same samples. The only thing that changed was how fast we find
the threshold value.

We also checked softmax (3 scalar passes over vocab) and top-p (O(n²) full
sort). Softmax turned out to be ~1.5ms total — with `-ffast-math` on macOS,
`expf` is already vectorized by the compiler via Accelerate. And top-p is
skipped entirely at the default `top_p=1.0`. So quickselect was the only
sampling fix that mattered.

## Streaming Pipeline: Closing the Last Gap

With streaming mode (`--stream`), the user hears audio as it generates — chunks
of ~0.8s arrive progressively. But streaming was **30% slower** than normal mode
(RTF 2.0 vs 1.4). Why?

Normal mode uses a **decoder thread**: the speech decoder runs in the background
while Talker+CP generate the next frame. The two stages overlap in time:

```
Main thread:    [Gen F1] [Gen F2] [Gen F3] ...
Decoder thread:          [Dec F1] [Dec F2] ...
```

But streaming mode ran the decoder **synchronously in the main thread**. Every
10 frames, the main thread stopped generating to decode audio and call the
callback. The main thread was blocked during decode:

```
Main thread:    [Gen F1-10] [DECODE+CALLBACK] [Gen F11-20] [DECODE+CALLBACK] ...
                             ^^^^ BLOCKED ^^^^               ^^^^ BLOCKED ^^^^
```

The fix: use the decoder thread for streaming too. Instead of accumulating audio
in a buffer, the decoder thread calls the audio callback directly. The main
thread never blocks on decode:

```c
if (dt->audio_cb) {
    int ret = dt->audio_cb(chunk_audio, chunk_samples, dt->audio_cb_userdata);
    if (ret != 0) dt->cb_aborted = 1;
} else {
    dt_append_audio(dt, chunk_audio, chunk_samples);
}
```

The callback (`fwrite` + `fflush` to a WAV file, or `send()` to an HTTP socket)
is called from the decoder thread. Both are thread-safe by default.

**Result: Streaming RTF 2.04 → 1.38** — identical to normal mode. The change
was `-80` lines, `+53` lines (net simpler!), because we deleted the entire
synchronous streaming code path and unified everything through the decoder
thread.

The output is **bit-identical** across all four modes: CLI normal, CLI streaming,
HTTP server normal, HTTP server streaming. Same seed, same speaker, same language
→ same bytes in the WAV file.

## Batch vvexpf: Transcendentals Are Expensive One at a Time

After the algorithmic wins, we went hunting for smaller gains. The SwiGLU
activation function in every transformer layer computes `x * sigmoid(x)`, and
sigmoid needs `expf()`. In a 28-layer Talker and a 5-layer Code Predictor
running 15 passes per frame, that's ~163,000 individual `expf()` calls per
audio frame.

Each `expf()` is a transcendental function — high latency, hard to pipeline.
But calling them one by one wastes the CPU's SIMD units. The fix: batch them.

On macOS, Apple's Accelerate framework provides `vvexpf()` — a vectorized
exponential that processes an entire array at once using optimized SIMD paths
internally. We wrote a `qwen_swiglu_inplace()` kernel that computes
`gate = vvexpf(-gate); gate = x / (1 + gate); gate *= up` over the full
intermediate dimension in one call:

```c
void qwen_swiglu_inplace(float *gate, const float *up, int n) {
#if defined(__APPLE__) && defined(USE_BLAS)
    int ni = n;
    // gate = -gate
    vDSP_vneg(gate, 1, gate, 1, ni);
    // gate = exp(-gate)  (batch)
    vvexpf(gate, gate, &ni);
    // gate = 1 + exp(-gate)
    float one = 1.0f;
    vDSP_vsadd(gate, 1, &one, gate, 1, ni);
    // gate = x / (1 + exp(-gate))  →  sigmoid(x) * x via up vector
    vDSP_vdiv(gate, 1, up, 1, gate, 1, ni);
#else
    // scalar fallback — compiler auto-vectorizes with -ffast-math
    for (int i = 0; i < n; i++)
        gate[i] = up[i] * gate[i] / (1.0f + expf(-gate[i]));
#endif
}
```

**Result: Code Predictor 8% faster** (76 ms/f → 70 ms/f). Those ~163K scalar
`expf` calls per frame collapsed into ~206 batched `vvexpf` calls. Not a
headline number, but it's free — the output is bit-identical and the code is
actually cleaner than the inline scalar loop it replaced.

The Abrash lesson applies here too: just as he taught us that unaligned memory
access wastes bus cycles, calling transcendentals one at a time wastes SIMD
lanes. The hardware *wants* to process 4-8 values at once — you just have to
feed it that way.

## SIMD BF16 Accumulation: One More Scalar Loop

The codec embedding lookup accumulates 15 codebook vectors per audio frame —
each a BF16-to-F32 conversion followed by a vector add. The original code did
this scalar:

```c
for (int i = 0; i < dim; i++) {
    uint32_t bits = (uint32_t)src_bf16[i] << 16;
    float val; memcpy(&val, &bits, sizeof(float));
    dst[i] += val;
}
```

We wrote `qwen_bf16_accum_f32()` with NEON and AVX2 paths. The NEON version
processes 8 BF16 values per iteration — load, shift-widen to F32, add:

```c
// NEON: 8-wide BF16→F32 accumulate
uint16x8_t bf = vld1q_u16(src_bf16 + i);
float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
vst1q_f32(dst + i,     vaddq_f32(vld1q_f32(dst + i), f0));
vst1q_f32(dst + i + 4, vaddq_f32(vld1q_f32(dst + i + 4), f1));
```

The AVX2 version does the same with 256-bit registers — `cvtepu16_epi32` to
zero-extend, `slli_epi32` to shift into F32 position, `add_ps` to accumulate.

The per-frame impact is small (~0.5-1ms), but it adds up over hundreds of
frames and eliminates yet another scalar loop hiding in a SIMD codebase —
exactly the kind of thing Abrash warned about: the fast path is only fast if
*all* the code on it is optimized.

## Delta Prefill: Reusing the KV Cache Across Requests

The Talker's prompt has a fixed structure: ChatML header, speaker token,
language token, codec control tokens, then the actual text. For a server
handling multiple requests with the same speaker and language, the prefix
is identical every time — but we were re-prefilling it from scratch on every
call.

Causal attention gives us a nice property: prefix tokens produce identical
KV cache entries regardless of what comes after. If the first 8 tokens of
the prompt match the previous request, their KV entries are already in the
cache. We just need to prefill the *new* tokens.

The implementation compares the current input embeddings against the previous
call's cached embeddings (stored in `prev_input_embeds`). If the first N
embeddings match, we skip to position N and only process the delta:

```
Request 1: [header][speaker][lang][codec][text_A]  →  full prefill (18 tokens)
Request 2: [header][speaker][lang][codec][text_B]  →  delta prefill (skip 8, process 10)
Request 3: [header][speaker][lang][codec][text_C]  →  delta prefill (skip 8, process 7)
```

When the speaker or language changes, the prefix differs and we fall back to
full prefill automatically — no special-casing needed.

**Result: ~50% prefill time savings on repeated speaker** in server mode. For
a chatbot or voice assistant scenario where you're generating many responses
in the same voice, this eliminates the biggest fixed cost in the pipeline.

## Quantization: What the 1.7B Model Taught Us

We'd already tried INT4 and INT8 quantization on the 0.6B model and found
them slower or neutral — the matrices are too small (hidden=1024) to be
bandwidth-bound, so dequantization overhead dominates. But the 1.7B model
has `hidden=2048` and `intermediate=6144` — 4× larger matrices. Time to
revisit.

**INT8 (`--int8`): 20% Talker speedup on 1.7B.** Per-row absmax quantization
at load time (scale = max(|row|) / 127), NEON int8 matvec for decode. The
Talker went from 79.3 ms/f to 67.4 ms/f. Audio quality is good — no
perceptible degradation in A/B tests.

**INT4 Q4_0 (`--int4`): no speedup, actually 4% slower.** We used the same
nibble-packed format as llama.cpp (32 weights per block, 16 bytes + 1 fp32
scale). The NEON unpack path needs AND, SHR, subtract-8, widen, convert —
about 8 ops per 32 weights versus 1 op for BF16 (`vshll`). Even at 2048-wide,
the compute overhead exceeds the bandwidth savings.

| Config | Talker ms/f | CP ms/f | RTF |
|--------|------------|---------|-----|
| 1.7B BF16 | 79.3 | 87.0 | 4.32 |
| 1.7B INT8 | 67.4 | 78.7 | 3.59 |
| 1.7B INT4 | 82.6 | 81.7 | 4.51 |
| 0.6B BF16 | 22.5 | 82.0 | 2.15 |

The takeaway: quantization is not a universal win. It depends on whether
you're compute-bound or bandwidth-bound at your specific matrix dimensions.
INT8 hits the sweet spot for 1.7B — enough bandwidth reduction to matter,
low enough unpack overhead (3 NEON ops vs BF16's 1) to not eat the gains.
INT4's nibble unpacking (8 ops) crosses the break-even point. And on 0.6B,
nothing helps because you're compute-bound anyway.

This echoes what Abrash wrote about optimization traps: "the fastest code
is the code you don't execute." INT4 adds *more* code per weight (unpack,
shift, subtract, widen, convert, scale, accumulate) than BF16 (shift,
accumulate). The memory savings are real, but speed is what matters for
realtime TTS.

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
Our bf16 matvec kernel already processes 2 rows at a time with 8 SIMD
accumulators (NEON/AVX), doing 32 elements per inner loop iteration. The input vector
(4 KB for hidden=1024) fits entirely in L1. The weight matrix access is
sequential, which the hardware prefetcher handles well. The bottleneck is
main memory bandwidth (~10 GB/s effective out of 68 GB/s peak), not cache
misses.

**Prefetch hints in CP loop** (est. 0.5-1%, actual: not possible). Each Code
Predictor layer has ~26 MB of weights. The M1's shared L2 is 12 MB. You
can't prefetch what doesn't fit. The hardware prefetcher handles sequential
access within each matvec just fine — it's the layer *transitions* that cause
cold misses, and those are unavoidable without smaller weights.

**INT4/INT8 quantization on 0.6B** (tested, slower or neutral). See the
Quantization section above — the 0.6B model's hidden=1024 matrices are
compute-bound, not bandwidth-bound. Quantization only helped on 1.7B (INT8:
20% Talker speedup), while INT4 was slower even there.

**Softmax SIMD vectorization** (est. 2-4×, actual: not worth it). After
quickselect reduced total sampling from 93ms to 21ms, softmax is only ~1.5ms
of the remaining 21ms. With `-ffast-math`, the compiler already vectorizes
`expf` via platform libraries (Accelerate on macOS, libm on Linux). No
headroom for custom NEON/AVX exp.

**Speech decoder depthwise conv / LayerNorm SIMD** (est. 1.5-3×, actual: not
worth it). The speech decoder runs in a background thread overlapped with
generation. It finishes *before* Talker+CP complete — it's not the bottleneck.
ConvNeXt depthwise conv does 1.4M FLOPs vs 838M FLOPs for the BLAS-accelerated
pointwise convolutions. Optimizing 0.2% of the decoder's compute is pointless.

**Separating INT8 fields from CP layer struct** (est. 2-3% cache, actual: not
worth it). Only 5 layers × 264 bytes = 1.3KB total. The bottleneck is the
weight data (26MB per layer), not the 8-byte pointer loads in the struct.

## The Numbers

### 0.6B Model (Primary Target)

| Metric | Baseline | After all optimizations |
|--------|----------|------------------------|
| Talker | 46.9 ms/f | ~22 ms/f |
| Code Predictor | 104.7 ms/f | ~60 ms/f (batch vvexpf: 70→60) |
| Speech Decoder | ~2,600ms (blocking) | overlapped (background thread) |
| Prefill | ~1,800ms | ~1,000–1,600ms (delta: ~500ms repeat) |
| Codec head+sampling | 93ms | 21ms |
| Per-token malloc calls | ~120+ | **0** |
| **RTF (CLI, short ~5s audio)** | **~3.5** | **~1.4–1.7** |
| **RTF (CLI, long ~17s audio)** | **~2.5** | **~1.3** |
| **RTF (CLI `--stream`)** | **~3.5** | **~1.4–1.7** (same as normal) |
| **RTF (server warm, short)** | — | **1.39** |
| **RTF (server warm, long)** | — | **1.26** |

### 1.7B Model (with INT8)

| Metric | BF16 | INT8 (`--int8`) |
|--------|------|-----------------|
| Talker | 79.3 ms/f | 67.4 ms/f (**20% faster**) |
| Code Predictor | 87.0 ms/f | 78.7 ms/f |
| **RTF** | **4.32** | **3.59** |

All on an Apple M1 8-core, 16 GB RAM, 4 threads. RTF improves with longer
audio because prefill is a fixed cost that amortizes over more frames. The
speech decoder runs in a background thread, overlapping most of its work with
generation — including streaming mode, where the decoder thread calls the audio
callback directly. Server mode with embedding cache, warm buffers, delta prefill,
and decoder thread overlap delivers the best RTF at **1.26**.

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

7. **Check your algorithms, not just your SIMD.** A 4× sampling speedup from
   replacing selection sort with quickselect — no intrinsics, no threading,
   just a better algorithm. Profile first, but when you find O(kn) in a hot
   loop, fix the algorithm before reaching for SIMD.

8. **Unify code paths.** Streaming was 30% slower because it had its own
   synchronous decode path. When we unified it with the decoder thread (the
   same path normal mode uses), the gap disappeared. Two code paths that do
   the same thing will always diverge in performance.

9. **Batch your transcendentals.** Calling `expf()` 163,000 times per frame
   is slower than calling `vvexpf()` 206 times — same math, same result,
   8% faster. SIMD units want batches. This is the Abrash data alignment
   lesson in a different guise: don't waste hardware lanes by feeding values
   one at a time.

10. **Exploit causal structure for caching.** Causal attention means prefix
    tokens produce identical KV entries regardless of suffix. Delta prefill
    cuts server prefill time in half for repeated speakers — zero accuracy
    cost, because the math guarantees identical outputs.

11. **Quantization is not free compression.** INT8 works on 1.7B (20% win)
    because the matrices are large enough to be bandwidth-bound. INT4 loses
    on every model size we tested — the nibble unpack overhead exceeds the
    bandwidth savings. Always measure before assuming "smaller weights = faster."

12. **Read the old books.** Abrash's *Graphics Programming Black Book* and
    Carmack's `.plan` files are from another era, but the principles — cache
    friendliness, data alignment, knowing your memory hierarchy — are timeless.
    The specific rules change (64-byte cache lines instead of dword alignment),
    but the instinct to think about how data flows through the CPU is exactly
    the same. Every optimization in this post — alignment, SIMD batching,
    pipeline parallelism, algorithmic complexity — traces back to ideas those
    two articulated thirty years ago. The hardware evolved; the thinking didn't.

---

*This is part of the [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts)
project — a pure C inference engine for Qwen3-TTS text-to-speech models.*
