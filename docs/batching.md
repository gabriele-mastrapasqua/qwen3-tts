# Text-chunk batching — premise test (feat/batching)

> **Design constraints (non-negotiable):**
> 1. **OPT-IN alternative path, never the default.** Like vLLM, you turn it on (`--batch` / a
>    server mode) when the workload fits. The single-stream path (today's code) stays the default,
>    **untouched** — golden tests bit-identical. The batched path is `if (--batch) { new flow }
>    else { exactly as today }`. Reuse existing functions where possible; write NEW code for the
>    batched parts — never zap/rewrite the working single-stream code.
> 2. **Multi-ISA always.** Every batched kernel ships NEON **and** AVX2 **and** AVX-512 paths plus
>    a scalar fallback (same dispatch discipline as the rest of the engine). `qwen_matmat_bf16`
>    already does (vectorizes over the B dimension: AVX-512 16-wide → AVX2 8 → NEON 4 → scalar).
> 3. **Newer-ISA leads (annotate now, exploit later)** — see the TODO in `bf16_matmat_slice`:
>    ARM bf16 BFDOT/BFMMLA + i8mm SMMLA (Apple **M2/M3/M4/M5**, Neoverse V1/V2, **NVIDIA Grace /
>    DGX Spark**), ARM **SVE/SVE2** (Grace/Spark, vector-length-agnostic), x86 **AVX-512-BF16**
>    (VDPBF16PS) and **AVX-512-VNNI** for the int8 batched twin. And add `qwen_matmat_int8/_int4`
>    twins — batching pays MOST at low precision (it amortizes the unpack).
>
> **Validation roadmap (in order):** (1) M1 Mac — build + correctness + RTF; (2) the AMD Ryzen
> mini-PC (Zen2/3) over RDP; (3) the EPYC **Turin** VPS (AVX-512/VNNI); (4) only *after* those, use
> the numbers to discuss with Leo. Don't block on hardware we can't drive directly.


**Question:** if we split a long text into B chunks and step them *together* (so each bf16
weight is read from DRAM once and reused across all B chunks — matrix-MATRIX instead of
matrix-VECTOR), do we get throughput? This is the only lever the earlier analysis left open
(worker-pool concurrency was a dead-end: +8%, M1 bandwidth saturated by ONE synthesis).

## Microbench (`make batching-bench`, M1, single-thread + 4-thread)

`16× GEMV` (weights re-read 16×, as today) vs `GEMM(16)` (weights read once, FMA'd into 16
register-resident accumulators) on representative Talker weight shapes:

| shape | size | 1T speedup | 4T speedup (realistic) |
|---|---|---|---|
| 0.6B gate_up | 5.5 MB | 1.34× | 1.44× |
| 1.7B gate_up | 22 MB | 1.75× | 1.90× |
| 1.7B down | 22 MB | 2.01× | 2.04× |
| big (64 MB) | 64 MB | 1.98× | 1.88× |

## Verdict: the idea HOLDS — but the ceiling is ~1.5–2×, not B×

- A single bf16 GEMV reaches only ~12–16 GB/s effective on one M1 core — **well under** the
  ~60–100 GB/s the core can pull. So single-stream is **compute-bound on the NEON bf16 path**,
  not purely memory-bound. Batching amortizes the weight-read (~40–50% of the work), giving
  ~2×; the other ~50% is FMA compute that B chunks still have to do.
- The 4-thread (realistic) column is ~the same as 1-thread → threading already scales well and
  doesn't push us hard into the memory-bound regime where batching would pay 4–16×.
- This still **beats the worker-pool** dead-end (+8%): batched-GEMM is the right mechanism,
  confirming the prior conclusion. Just don't expect more than ~2× on M1.

## If we build it (the real prototype)

A ~2× throughput win requires a real rewrite — scope before committing:
1. **B independent sequences** in flight, each with its own Talker + CP KV cache.
2. **GEMM step kernels** (replace the per-token GEMV in `qwen_tts_kernels.c` with a
   register-blocked [out × B] matmul) for Talker QKV/O/gate_up/down AND the Code Predictor
   (CP is 90% matvec and runs 15×/frame — batching it matters most).
3. **Batched sampling / EOS** — chunks finish at different lengths; need ragged-batch handling
   (drop a chunk when it hits EOS, compact the batch).
4. **Chunk scheduler** — split long text on sentence boundaries, keep the batch full.
5. Output: re-stitch chunk audio in order (the `--compose`/`render_spans` concat already does
   seamless joins; reuse it).

**Recommendation:** worth it only for a *throughput/serving* goal (many requests, or one very
long document where 2× wall-time matters). For single short utterances it does nothing. The
gain is ~2× on M1; a higher-bandwidth box (where single-stream is more memory-bound) could see
more. Decide based on whether 2× justifies the rewrite + the per-sequence complexity.

## ⚠️ REAL-MODEL result (2026-06-07): batched compute is 4–12× SLOWER on M1

After building the batched Talker step + batched CP (both correctness-verified) and timing the
**actual per-frame compute** end-to-end (`--batch-bench`, B=8 K=50 on 0.6B M1):

| | single-stream | batched | speedup |
|---|---|---|---|
| 4 threads | 13.1 frames/s | 3.2 frames/s | **0.24×** |
| 1 thread | 11.7 frames/s | 0.9 frames/s | **0.08×** |

**The batched path is a LOSS on M1.** Why the premise microbench mispredicted: it compared two of
*our own* naive kernels (a simple NEON gemv vs the matmat). The **production `qwen_matvec_bf16` is
far more optimized** (8-wide NEON, 2 rows at a time, multiple register-resident accumulators to hide
FMA latency), while our `bf16_matmat_slice` is naive — scalar bf16 decode + an `acc[64]` array that
lives in **L1, not registers** (load/store every k), plus gather/scatter transposes. So B× the
production matvec beats our matmat, and on M1 (compute-bound, high per-core bandwidth) batching loses.

**Decision: do NOT build the full `--batch` integration on M1.** The two batched compute kernels are
built + correctness-verified (reusable), but batching only becomes worth integrating IF: (1) the
batched matmat is rewritten to production quality (register-blocked B-tiles, 2-rows × B, NEON bf16
decode like `bf16_matvec_fused`) AND (2) validated on a **memory-bound x86 box** (Ryzen/Turin) where
single-stream is bandwidth-starved and read-once amortization actually pays. On M1, single-stream wins.
This confirms the standing finding: M1 is compute-bound; batching is an x86/throughput lever at best.

## ✅ CORRECTION (2026-06-08): register-blocked matmat → batching WINS on M1 (~2×)

The "4–12× slower / x86-only" conclusion above was an **artifact of the naive matmat**, exactly as
condition (1) predicted. Rewrote `bf16_matmat_slice` to production quality: **compile-time-B
specializations** (`bf16_matmat_b1..b8,b16`) where the BV accumulators are register-resident (the
unrolled inner `for(j<BV)` makes them named scalars, not the spilling `acc[64]`), rows blocked 2 at a
time, and the broadcast-FMA auto-vectorizes per ISA under `-march=native`. The naive intrinsic loop is
kept only as `bf16_matmat_generic` (fallback for un-specialized B). Self-test still PASS (matmat(B=8)
vs B×matvec L2_rel ~3e-7); golden untouched (opt-in path).

Re-ran `--batch-bench` (0.6B, M1, K=50):

| | single-stream (B× sequential) | batched | speedup |
|---|---|---|---|
| **4 threads** | 11.9 frames/s | 25.1 frames/s | **2.10× (was 0.24×)** |
| 1 thread | 11.2 frames/s | 9.9 frames/s | 0.88× (was 0.08×) |

**B-sweep at 4 threads** (the natural "split one paragraph into N parallel chunks" counts):

| B | 2 | 3 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|
| speedup | 1.87× | 2.01× | **2.23×** | 1.59× | 2.21× | 1.60× |

**The finding flips: batching is a WIN on M1 at the default 4 threads** — even a 2-way split gives
1.87×, the sweet spot (B≈4–8) ~2.2×. Why it now wins where it lost: single-stream re-reads the weights
B× (once per sequence), which saturates the shared memory controller across 4 cores → bandwidth-bound;
the batched matmat reads weights ONCE → ~B× less DRAM traffic. At 1 thread it's still ~break-even (0.88×)
because a single core isn't bandwidth-starved and the hand-tuned `bf16_matvec_fused` is a hair tighter
per-FMA than the auto-vectorized matmat — but nobody runs 1 thread. The plateau ~2× is the bf16 ceiling
the premise microbench predicted; **lower precision (int4/int8) should push past it** (next section), and
a bandwidth-bound x86 box should do as well or better. **This is the audiobook/long-text lever:** split a
long paragraph into 2–4 chunks, step them batched → ~2× wall-clock vs sequential, reusing the weights.

**Revised decision: building the `--batch` integration IS worth it on M1** (not x86-only). Remaining work:
per-stream sampling + ragged-EOS + a chunk scheduler (split text, keep the batch full, re-stitch audio via
`render_spans`), then the int8/int4 batched twins. Validate on the Ryzen/Turin boxes where it should pay more.

## ✅ int8 / int4 batched twins MEASURED with the REAL kernels (2026-06-08)

Added `qwen_matmat_int8` (W int8 + per-row scale, f32 activation) and `qwen_matmat_q4_0` (q4_0 blocks,
nibble unpack amortized over B) — same compile-time-B register-blocking as bf16. Correctness in
`--self-test` (matmat_int8 vs B×matvec_int8 L2_rel ~4e-3 = activation-quant noise; matmat_q4_0 vs
B×matvec_q4_0 L2_rel ~3e-7). New **`make matmat-bench`** (`./qwen_tts --matmat-bench`) times the REAL
library kernels (B×matvec [today] vs matmat [batched]) per precision — supersedes the naive premise
microbench above (which compared two of our own scalar kernels). 0.6B/CP shapes, M1, B=8:

| shape | bf16 4T | int8 4T | int4 4T | bf16 1T | int8 1T | int4 1T |
|---|---|---|---|---|---|---|
| 3072×1024 | 1.36× | 0.70× | 1.23× | 0.97× | 0.55× | 1.09× |
| 1024×3072 | 1.60× | 1.12× | 1.31× | 1.40× | 1.09× | 1.29× |
| 2048×1024 | 1.77× | 1.16× | 1.35× | 1.54× | 1.18× | 1.38× |

Reading these (kernel-level numbers UNDERSTATE the full pipeline — a single 6 MB matrix partly fits cache,
so less DRAM-bound than the 28-layer Talker + 16×-per-frame CP where bf16 batching hit 2.1×):

- **int4 + batching = the real M1 lever, confirmed.** It WINS at BOTH 1T and 4T (1.1–1.4×) — the only
  precision that pays even single-threaded, because q4_0 single-stream is **unpack-bound** and batching
  does the nibble unpack ONCE for all B. And int4 is otherwise the *slowest* single-stream path on M1, so
  this is exactly where batching earns its keep → **int4+batching could make int4 viable on M1.** PLAN
  prediction validated with real kernels (no longer the inflated scalar 6–8×).
- **bf16 + batching** wins at 4T (1.4–1.8× kernel; 2.1× full pipeline), break-even at 1T — the
  bandwidth-vs-compute story.
- **int8 + batching currently LOSES / break-even** — but for an instructive reason: the int8 *sequential*
  matvec uses **SDOT integer dot** (very fast — 0.22–0.45 ms), while the int8 *twin accumulates in f32*
  (throwing SDOT away), so it's only as fast as the bf16 matmat. **TODO: an integer-dot int8 batched twin
  (SDOT on ARM / VNNI on x86)** — quantize each column's activation, accumulate int32 — to beat SDOT
  sequential. Until then, pair batching with **int4** (where the f32 twin already wins) or bf16, not int8.

## (superseded) Batching × PRECISION — naive premise microbench (int8 / int4 / int2)

Re-ran the bench storing weights at their real byte size (`make batching-bench`, 4-thread).
Two opposing effects: lower precision shrinks the weight READ (less to amortize → batching
helps less) BUT makes the UNPACK costlier, and GEMV redoes that unpack per token while GEMM
does it once (→ batching amortizes unpack → helps more). Measured trend:

| precision | weight read | batching speedup (trend) |
|---|---|---|
| bf16 | 2 B/w | ~2× (clean NEON kernels) |
| int8 | 1 B/w | ~1–1.2× (cheap decode, small read → little to amortize) |
| **int4** | 0.5 B/w | **large — unpack-bound GEMV, GEMM amortizes the nibble unpack** |
| **int2** | 0.25 B/w | **large — same, bit unpack** |

**The unpack-amortization effect dominates at low precision: batching is WORTH MORE the lower
you quantize.** Caveat: our microbench decodes scalar, so the int4/int2 GEMV is un-vectorized
and the raw 6–8× is inflated; a production int4 kernel unpacks faster, so the real speedup is
smaller — but the *synergy is real and the direction is solid*. This matters because **int4 is
slow on M1 today precisely because nibble-unpack dominates** (per the quant notes — "int4 is the
x86 lever, not M1"); batching amortizes that unpack, so **int4 + batching could make int4 viable
on M1** and compound on x86. → If we build batching, pair it with int4/int8, not bf16.

## Prediction for other CPUs (AMD VPS / mini-PC / Leo's new Zen5)

Batching pays in proportion to how MEMORY/UNPACK-bound single-stream is. M1 is the *worst* case
(high per-core bandwidth + wide caches → compute-bound → only ~2× at bf16). Expectations:

- **AMD Ryzen mini-PC (Zen3+, 6800H):** more memory-bound per core than M1 (int4-MT was already
  THE lever there, RTF 2.81→2.02). Batching at bf16 likely **>2×**; **int4 + batching the sweet
  spot** (x86 + nibble-unpack amortization compound). Watch thread placement.
- **EPYC server VPS (Zen4/5, many cores/channels):** per-core bandwidth lower, designed for
  throughput; single-stream tops ~1.6–1.9 (quant notes). Batching is the *intended* use —
  expect ~2–4× per core **× linear core scaling** = the real throughput play. Pin threads to one
  CCD (a 4-vCPU VM scattering across CCDs already hurt single-stream).
- **Leo's new "super AMD" (Zen5, 9950X / Turin):** native VNNI int8 (validated ~1.85× at equal
  core) + AVX-512. A batched **int8-VNNI GEMM** is very efficient (matrix VNNI amortizes both
  read and unpack); strong AVX-512 compute means bf16 stays compute-bound (modest batching) but
  **int8/int4 + batching should be excellent**. Best target for the throughput prototype.

Net: M1 is the floor for batching benefit; every x86 box we've touched should do **as well or
better**, and the win grows when paired with int4/int8. Validate on the Ryzen box + Leo's Zen5
before/while building the full prototype.

Bench: `tests/batching_bench.c` (`make batching-bench`).
