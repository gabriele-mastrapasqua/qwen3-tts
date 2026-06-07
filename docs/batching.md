# Text-chunk batching — premise test (feat/batching)

> **Design constraint: batching is an OPT-IN alternative path, never the default.** Like vLLM,
> you turn it on (`--batch` / a server mode) when the workload fits — many requests, or one long
> document where throughput matters. The single-stream path (today's code) stays the default and
> is untouched. Default behavior + golden tests must remain bit-identical.
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
