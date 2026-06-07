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

## Batching × PRECISION (int8 / int4 / int2)

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
