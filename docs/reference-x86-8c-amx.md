# An 8-core Intel AMX serving box, measured end to end

The x86 counterpart to [`reference-arm-16c.md`](reference-arm-16c.md): one machine, one build,
one profile, so that [`serving-operations.md`](serving-operations.md) can describe a *procedure*
without carrying a table per cell, and so a second x86 box has something to be compared against.

**Nothing here transfers by itself.** The shapes are the transferable part — which stage AMX
pays for, where realtime is lost, why a lever measured on Arm is wrong here. The milliseconds
are not.

## The setup

| | |
|---|---|
| CPU | 8 cores, Intel Xeon Platinum 8581C (Emerald Rapids), **SMT disabled** (`Thread(s) per core: 1`), 1 NUMA node |
| cache / bandwidth | 260 MiB shared L3 · measured Triad **82 GB/s**, knee at 8 threads |
| build | `make blas SIMD=amx`, OpenBLAS pthread, `--caps` reports AMX INT8 and AMX BF16 both ACTIVE, `--self-test` PASSED |
| model | open weights, 1.7B, `--int8` |
| profile | [`x86-8c-amx-multiclient-ttfa`](../configs/perf/x86-8c-amx-multiclient-ttfa.json), verified in every server's `[FLAGS]` line |
| topology | `2x4` — two pre-forked workers, four threads each |
| bank | `tests/load_texts_en.txt`, class `short` |
| KPIs | TTFA = send → first audio chunk · stream RTF = per request, after the first chunk · errors |

For scale, the Arm reference box in the sibling page has **twice the cores and four times the
memory bandwidth** (336 GB/s). Read the two pages together and the differences stop being
surprising.

## 1. Choosing the topology

`make bench-topo BENCH_TOPO=1x8,2x4,4x2 BENCH_CONC=1,2,4,6,8` — three waves per level, zero
errors and zero rejections in all fifteen cells:

| topology | C | TTFA p50 | TTFA p95 | stream RTF | req/s | in-flight batch |
|---|---:|---:|---:|---:|---:|---:|
| `1x8` | 1 | **98 ms** | **99 ms** | **0.68** | 0.78 | — |
| `1x8` | 2 | 157 ms | 195 ms | 1.10 | 0.98 | — |
| `1x8` | 4 | 308 ms | 357 ms | 1.82 | 1.15 | — |
| `1x8` | 8 | 553 ms | 702 ms | 3.22 | 1.30 | — |
| `2x4` | 1 | 134 ms | 135 ms | 1.02 | 0.52 | 0.78 |
| `2x4` | 2 | 138 ms | 223 ms | 1.05 | 1.01 | 1.50 |
| `2x4` | 4 | 245 ms | **259 ms** | 1.49 | **1.39** | 3.10 |
| `2x4` | 6 | 318 ms | **374 ms** | 1.88 | **1.61** | 4.81 |
| `2x4` | 8 | 440 ms | **500 ms** | 2.34 | **1.74** | 6.52 |
| `4x2` | 1 | 201 ms | 204 ms | 1.69 | 0.31 | 0.85 |
| `4x2` | 4 | **236 ms** | 285 ms | 1.84 | 1.10 | 3.06 |
| `4x2` | 8 | 442 ms | 458 ms | 2.42 | 1.67 | 6.40 |

Everything on one request wins C=1 outright and collapses fastest; four two-thread workers are
too narrow for a 1.7B (RTF 1.69 even alone); `2x4` holds both ends and takes throughput at every
level from C=4 up. Use `1x8` when single-stream latency is the objective, `2x4` otherwise.

**Client concurrency is not batch width.** At C=4 on `2x4` the measured in-flight batch is 3.10
*in the system*, about 1.55 per worker. That single fact explains the decoder result in §3.

## 2. What AMX is worth, per stage

`QWEN_NO_AMX` removes two unrelated consumers at once, so it cannot attribute anything. The
per-family controls can:

| control | C=1 TTFA | C=4 p50 / p95 | C=4 stream RTF | req/s |
|---|---:|---:|---:|---:|
| AMX fully on | 135 ms | 251 / 324 ms | 1.50 | 1.38 |
| `QWEN_NO_AMX_BF16=1` | **188 ms** | **374 / 556 ms** | 1.54 | 1.34 |
| `QWEN_NO_AMX_INT8=1` | 134 ms | 258 / 337 ms | **1.64** | **1.24** |
| `QWEN_NO_AMX=1` | 196 ms | 370 / 601 ms | 1.72 | 1.20 |

**AMX BF16 is first audio; AMX INT8 is sustained throughput.** Removing BF16 costs 39% of C=1
TTFA and 72% of C=4 p95 while stream RTF hardly moves — it runs the prefill projections.
Removing INT8 leaves TTFA alone and costs 9% of RTF and 10% of throughput — it runs the batched
decode. Disabling both yields the TTFA of the first and the RTF of the second, so the two
compose. The kernel audit confirms the mechanism rather than inferring it: with AMX on, the
Talker's 39.5 GMAC sit on `bf16 AMX tiles`, and under `QWEN_NO_AMX=1` the identical GMAC appear
on `bf16 AVX-512 dpbf16`.

On an x86 host **without** AMX the same work lands on the AVX-512 paths, and the VNNI tiling is
what matters: with AMX off, additionally setting `QWEN_NO_VNNI_TILE=1` cost a further 10% of
throughput and 7% of stream RTF here.

## 3. Two levers that do not port from the Arm profile

| lever | Arm profile | here | why |
|---|---|---|---|
| `QWEN_POOL_SPIN` | 65536 | **4096** (the x86 compiled default) | 0 costs +13% stream RTF at C=1 and 31k vs 2k context switches; 1024 is in between; 65536 costs p95 358 vs 339 ms. A spinning worker needs a core to spin on, and on 8 cores it takes one from a worker that has work |
| `QWEN_DECODER_BATCH` | 1 | **0** | measured over three interleaved repeats, off is 12% better at C=4 TTFA p95 (290 vs 331 ms mean) with identical RTF and throughput |

The decoder result has a mechanism, not just a number. `QWEN_SERVE_PROFILE=1` reports:

```
C=1:  decoder batch: calls / mean   60   1.00     max slots 1
C=4:  decoder batch: calls / mean   42   1.40     max slots 2
      [serve-profile] mean 1.38 active slots
```

The gang never forms. With ~1.55 slots per worker there is nothing to amortise over, so the
batch pays the wait and collects the tail. **Expect this to flip on a box wide enough to give
one worker several slots** — which is exactly what the Arm host does.

Note that the server enables decoder batching *itself* with `setenv(..., overwrite=0)`. A
profile that stays silent gets it on; only an explicit `0` turns it off. An A/B that sets the
variable to `1` in one arm and omits it in the other compares a configuration against itself.

## 4. Where the time goes

`QWEN_SERVE_PROFILE=1`, C=4, `2x4` (attribution run, not a timing run):

| stage | share |
|---|---:|
| speech decode + embed | 36.8% |
| talker step (batched) | 34.9% |
| code predictor (batched) | 18.7% |
| admission + prefill | 4.2% |
| codec head (GEMM) | 0.3% |

Prefill is only 4% of the *loop*, and simultaneously the thing that decides TTFA, because it is
serial in front of the first audio chunk. Both statements are true and they answer different
questions.

## 5. Prefill is priced per tile group, not per token

Talker prefill on this host, `-j8`, 1.7B int8, varying the input length:

| positions | 16-position calls | prefill |
|---:|---:|---:|
| 14 | 1 | 55 ms |
| 21 | 2 | 94 ms |
| 29 | 2 | 112 ms |
| 37 | 3 | 149 ms |
| 45 | 3 | 162 ms |
| 53 | 4 | 201 ms |

Six points fit `prefill_ms ≈ 20 × ceil(positions/16) + 2.2 × positions` to within 8%. The fixed
term is architectural: the AMX tile is 16 rows × 64 bytes and the accumulator is configured
`colsb = B*4`, so `B ≤ 16` and a 53-position prefill is four calls whose cost only partly
depends on how full each one is. Going wider means multiple accumulator tiles, which is a kernel
change and is not in this branch.

An attempt to avoid the re-read by blocking output rows and moving the chunk loop inside was
measured and **rejected**: 57 positions cost 214 ms unblocked against 967 / 584 / 427 / 307 /
240 ms at row blocks of 256 / 512 / 1024 / 2048 / 4096. The weights are already resident in the
260 MiB L3 across the chunks of one layer, so there was nothing to amortise and the extra
dispatches and packing were pure cost.

## 6. What this class of machine is for

- **One realtime stream.** `1x8` at C=1 is RTF 0.68. Everything else on this box is above 1.
- **Good first audio under concurrency.** `2x4` holds TTFA p95 under 400 ms out to C=6 and under
  500 ms at C=8.
- **Not four realtime streams.** At C=4, stream RTF is 1.49. The Arm box with twice the cores
  and four times the bandwidth holds 0.72 at the same point.

AMX buys the compute-bound half of the problem and cannot buy memory bandwidth, and this
workload re-reads the Code Predictor weights 16 times per frame. Choose an AMX box for time to
first audio under concurrency; count cores and GB/s for the number of realtime streams.

## Reproducing this page

```bash
make blas SIMD=amx GIT_REV=$(git rev-parse --short HEAD)
./qwen_tts --caps                       # AMX INT8 and AMX BF16 must both say ACTIVE
./qwen_tts --self-test                  # and again with QWEN_NO_AMX=1
make check-flag-registry                # every flag the engine reads is one it declares
make bench-fingerprint                  # SMT off, cores, cache, measured bandwidth
make bench-topo  BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=x86-8c-amx-recommended \
                 BENCH_TOPO=1x8,2x4,4x2 BENCH_CONC=1,2,4,6,8
make bench-suite BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=x86-8c-amx-recommended \
                 BENCH_TOPO=2x4 BENCH_OUT=/tmp/bench_x86
```

Every table above comes from those commands, on an idle box, one fresh server process per cell.
