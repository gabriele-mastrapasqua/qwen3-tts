# Scaleway 16-core VNNI reference

Measured on a 16-core x86 host without AMX, using the open 1.7B base checkpoint and the
`scaleway-16c-vnni-ttfa` profile. This page records the workload boundary: short synchronized
waves are useful for topology comparison, while the realistic bank and open-loop tests describe
what happens under mixed request lengths.

## Hardware and launch

| item | measured value |
|---|---|
| CPU | AMD EPYC 9555P, Zen 5 Turin |
| cores | 16 physical, one thread per core |
| NUMA / L3 | 1 NUMA node / 256 MiB reported L3 |
| ISA used | AVX-512 VNNI and AVX-512 BF16 |
| AMX | absent |
| memory probe | about 88 GB/s STREAM-like roof |
| server | 2 workers × 8 threads, `--batch-size 8` |
| model path | `qwen3-tts-1.7b-base`, `--int8` |

The profile pins `QWEN_POOL_SPIN=4096`, `QWEN_DECODER_BATCH=0`, `QWEN_PREFILL_MATMAT=1`,
`QWEN_CP_PREFILL2=1`, `QWEN_PREFIX_CACHE=1`, `QWEN_STREAM_DECODE_CHUNK=8`,
`QWEN_VNNI_GEMV_MR=2` and `OPENBLAS_THREAD_TIMEOUT=1`. `OPENBLAS_NUM_THREADS` is absent.

## Short synchronized wave

Five waves per cell, short diverse bank, two workers × eight threads. Values are TTFA p50/p95 in
milliseconds, followed by stream RTF p50.

| topology | C=1 | C=2 | C=4 | C=6 | C=8 |
|---|---:|---:|---:|---:|---:|
| 1×16 | 70/77 · 0.51 | 123/144 · 0.82 | 233/267 · 1.18 | 331/390 · 1.55 | 424/497 · 1.93 |
| **2×8** | **66/73 · 0.47** | **101/213 · 0.73** | **185/221 · 1.11** | **250/310 · 1.34** | **310/374 · 1.55** |
| 4×4 | 82/109 · 0.55 | 106/204 · 0.75 | 184/374 · 1.34 | 313/480 · 1.64 | 286/376 · 1.72 |
| 8×2 | 133/189 · 0.85 | 131/343 · 0.91 | 153/395 · 1.31 | 260/705 · 1.85 | 234/519 · 2.21 |

2×8 is the balanced profile. 1×16 is the single-stream latency alternative; 8×2 is not a
useful choice for this model. Short-bank C=4 is still RTF 1.11, so it does not support a
sub-1.0 claim.

## Realistic bank and open loop

The realistic wave uses the full mixed-length bank. It is deliberately not interchangeable with
the short topology sweep.

| C | TTFA p50/p95/max (ms) | stream RTF p50 | requests/s |
|---:|---:|---:|---:|
| 1 | 100/300/300 | 0.45 | 0.25 |
| 2 | 184/657/657 | 0.60 | 0.39 |
| 4 | 640/1017/1047 | 1.20 | 0.34 |
| 6 | 710/1763/1797 | 1.18 | 0.41 |
| 8 | 1167/1297/1338 | 1.62 | 0.47 |

The C=1 p95 has only three observations and is descriptive, not a production percentile.

Exact-profile Poisson checks at the nominal C=4 used 24 arrivals per rate, no audio capture and
the same 60-second request cap:

| arrival rate | TTFA p50/p95/max | over 500 ms | stream RTF p50/p95 | errors |
|---:|---:|---:|---:|---:|
| 1 req/s | 370/1058/1320 ms | 10/24 | 1.98/2.68 | 0 |
| 2 req/s | 945/3882/5195 ms | 14/24 | 2.42/3.96 | 0 |

Poisson is open loop: its mean in-flight load rose above four (6.88 and 9.08), so these rows
measure backlog pressure, not a fixed four-request closed-loop service point.

## Closed-loop soak

The 10-minute stratified soak at C=4 discarded the first 60 seconds as warm-up and completed 267
requests, of which 251 were KPI samples. There were 0 errors, 0 queue rejections, 0 queue
timeouts and 0 request timeouts. The pooled soak verdict was `PASS`:

| metric | first comparison window | last comparison window | drift |
|---|---:|---:|---:|
| TTFA p50 | 254.0 ms | 285.5 ms | +12.4% |
| TTFA p95 | 819.2 ms | 814.5 ms | −0.6% |
| stream RTF p50 | 1.0873 | 1.0802 | −0.7% |

PSS growth was −0.16%, RSS growth +0.24%, thread count stayed at 84 and open file descriptors
stayed within a range of three. Per-class p95 was `PARTIAL`, not failed: each class had only
4–6 samples in the comparison windows while the analyzer requires 20 for a class p95.

## Dispatch census and A/B decisions

At C=4, diagnostic counters attributed work as follows:

| path | share |
|---|---:|
| INT8 VNNI matrix-matrix | 32.9% |
| BF16 AVX-512 matrix-matrix | 12.7% |
| INT8 GEMV | 35.6% |
| solo / single-slot work | 18.6% |

The remaining ceiling is therefore mostly GEMV and single-slot work. The decoder is already
mostly batched VNNI; making that GEMM faster cannot move the whole request by the same percentage.

Kept in the profile: activation quantization, cached INT8 row sums, GEMV MR=2, pool spin 4096,
decoder batching off, and decode chunk 8. Rejected as global defaults: GEMV MR=4, a global VNNI
NCHUNK value, and decoder batching on. VNNI tiling remains enabled as the native path, but its
end-to-end A/B showed no material win. These decisions are host/workload measurements, not
universal x86 defaults.

The profile remains provisional. This host has useful AVX-512 VNNI throughput, but its memory
roof and the large GEMV share keep the 1.7B model above RTF 1.0 at C=4. A future improvement
needs to reduce weight traffic, increase effective reuse, or make the real GEMV path cheaper.

## VNNI prepack candidate

The branch also contains an opt-in parent-side packed RHS experiment. With
`QWEN_VNNI_PREPACK=cp`, the parent built 41 CP matrices (about 102 MB) before prefork and the
workers inherited them. A corrected five-wave A/B on the same 2×8 setup and the AVX-512 BF16 +
VNNI build measured:

| mode | C=1 TTFA p50/p95 | C=4 TTFA p50/p95 | C=4 total RTF | errors/rejections |
|---|---:|---:|---:|---:|
| prepack off | 66/73 ms | 167/318 ms | 1.10 | 0/0 |
| CP prepack | 66/73 ms | 178/319 ms | 1.13 | 0/0 |

The quality run produced identical WAV bytes for all four requests. The candidate is therefore
kept available for other CPUs and shapes, but it is not part of this profile: it does not
improve the server objective and only targets batched matmat, while the census says the larger
remaining share is GEMV/single-slot work.

The separate oneDNN oracle and its fairness caveat are recorded in
[`x86-oracle.md`](x86-oracle.md).
