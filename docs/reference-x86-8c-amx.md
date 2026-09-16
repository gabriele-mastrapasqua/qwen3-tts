# An 8-core Intel AMX serving box, measured end to end

The x86 counterpart to [`reference-arm-16c.md`](reference-arm-16c.md): one machine, one build,
one profile, so that [`serving-operations.md`](serving-operations.md) can describe a *procedure*
without carrying a table per cell, and so a second x86 box has something to be compared against.

**Nothing here transfers by itself.** The shapes are the transferable part — which stage AMX
pays for, where realtime is lost, why a lever measured on Arm is wrong here. The milliseconds
are not.

**What this page is and is not.** It is a validation of the AMX and VNNI paths and an honest
account of what this class of machine does. It is *not* a report of a new AMX or VNNI kernel
optimisation: the batch-width fix described in §5 does not reach either of those dispatchers.

## The setup

| | |
|---|---|
| CPU | 8 cores, Intel Xeon Platinum 8581C (Emerald Rapids), **SMT disabled** (`Thread(s) per core: 1`), 1 NUMA node |
| cache / bandwidth | 260 MiB shared L3 · measured Triad **82 GB/s**, knee at 8 threads |
| build | `make blas SIMD=amx`, OpenBLAS pthread, `--caps` reports AMX INT8 and AMX BF16 both ACTIVE, `--self-test` PASSED with and without AMX |
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
| `1x8` | 1 | **93.4 ms** | **94.7 ms** | **0.65** | 0.82 | — |
| `1x8` | 2 | 137.6 ms | 178.6 ms | 1.07 | 1.02 | — |
| `1x8` | 4 | 304.0 ms | 353.5 ms | 1.75 | 1.19 | — |
| `1x8` | 6 | 391.2 ms | 495.4 ms | 2.36 | 1.30 | — |
| `1x8` | 8 | 539.9 ms | 655.8 ms | 3.07 | 1.35 | — |
| `2x4` | 1 | 128.0 ms | 128.7 ms | 0.98 | 0.54 | 0.79 |
| `2x4` | 2 | 132.9 ms | 210.6 ms | 1.03 | 1.04 | 1.50 |
| `2x4` | 4 | 239.2 ms | **252.0 ms** | 1.43 | **1.40** | 3.06 |
| `2x4` | 6 | 317.2 ms | **364.8 ms** | 1.87 | **1.66** | 4.83 |
| `2x4` | 8 | 421.7 ms | 483.6 ms | 2.28 | **1.77** | 6.47 |
| `4x2` | 1 | 196.8 ms | 197.6 ms | 1.65 | 0.32 | 0.85 |
| `4x2` | 4 | **224.5 ms** | 281.1 ms | 1.77 | 1.14 | 3.07 |
| `4x2` | 8 | 427.2 ms | **444.7 ms** | 2.32 | 1.73 | 6.37 |

Commit `266f706`, binary `f1105dcd…`, three waves per level, **zero errors and zero rejections in
all fifteen cells**.

Everything on one request wins C=1 outright and collapses fastest; four two-thread workers are
too narrow for a 1.7B (RTF 1.69 even alone); `2x4` holds both ends and takes throughput at every
level from C=4 up. Each topology has one scope and it is worth stating them separately:

- **`1x8` — single-stream latency.** 98 ms first audio and the only RTF below 1 on this box.
- **`2x4` — the general profile.** Best p95 at C=4 and C=6 and best throughput from C=4 up.
- **`4x2` — the C=8 tail, and nothing else.** At C=8 it has the better p95 (458 vs 500 ms) for
  slightly less throughput; at C=4 it takes only the median and loses p95, RTF and throughput;
  at C=1 it is twice the latency of `2x4`. It is a saturated-box profile.

**TTFA and sustained RTF are separate claims and this box separates them sharply.** First audio
p95 stays under 400 ms out to C=6 on `2x4`, while stream RTF passes 1 between C=1 and C=2. Four
or six concurrent requests get their first audio quickly; they are not realtime while they run.

**Read the percentiles unrounded.** The printed table gives `2x4` a p95 of `500` at C=8, which a
500 ms budget would call a pass. The unrounded value was 500.180 ms in one run and 495.415 ms in
another: it straddles the threshold, so the honest claim stops at C=6, and the topology that
actually holds C=8 under 500 ms is `4x2`, at 472.4 ms.

## 1b. The 0.6B — a different model, measured separately

Same host, same profile, same binary as every table above. **This section is about the 0.6B and
nothing else**: it is not evidence about the 1.7B's latency, quality or production readiness, and
the two should never be quoted as one result.

| topology | C | TTFA p50 / p95 | stream RTF | req/s |
|---|---:|---:|---:|---:|
| `1x8` | 1 | 72.8 / 76.0 ms | **0.49** | 1.01 |
| `1x8` | 2 | 94.2 / 165.0 ms | **0.88** | 1.12 |
| `1x8` | 4 | 211.8 / 265.9 ms | 1.50 | 1.28 |
| `2x4` | 1 | 93.2 / 93.9 ms | **0.69** | 0.72 |
| `2x4` | 2 | 94.3 / **117.1 ms** | **0.70** | 1.40 |
| `2x4` | 4 | 163.7 / 172.0 ms | 1.13 | 1.71 |
| `2x4` | 6 | 193.3 / 242.8 ms | 1.51 | 1.87 |
| `2x4` | 8 | 271.3 / 324.0 ms | 1.90 | 1.98 |

**Two concurrent realtime streams** on `2x4` — RTF 0.70 at C=2 with 117 ms p95 first audio — and
C=4 misses by 13% rather than by a lot. `1x8` also holds C=2 (0.88) but with a worse tail
(165 ms p95), so `2x4` is the better realtime pair here as well.

The 1.7B never reaches this on this hardware. If a deployment needs realtime under any
concurrency at all, that is an argument for running the 0.6B on this box — it is not an argument
about how the 1.7B sounds, which no timing table can settle.

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
| `QWEN_DECODER_BATCH` | 1 | **0** | three interleaved repeats at each end of the range: off is 12% better at C=4 TTFA p95 (290 vs 331 ms) and 11% better at C=8 TTFA p50 (443 vs 498 ms), with RTF and throughput inside noise |

The decoder result has a mechanism, and it is worth reporting precisely because the obvious
explanation is only half right. `QWEN_SERVE_PROFILE=1` reports:

```
C=1:  decoder batch: calls / mean   60   1.00     max slots 1
C=4:  decoder batch: calls / mean   42   1.40     max slots 2    (mean 1.38 active slots)
C=8:  decoder batch: calls / mean   42   2.24     max slots 4    (mean 2.16 and 3.56 per worker)
```

At C=1 and C=4 the gang barely forms: with ~1.55 slots per worker there is nothing to amortise,
so the batch pays the wait and collects the tail. **At C=8 it does form** — 2.24 slots deep, up
to 4 — and turning it off is *still* better: TTFA p50 443 vs 498 ms and p95 546 vs 580 ms over
three interleaved repeats, for about 1% of throughput, which is inside this box's noise. So the
finding is not "batching never happens"; it is that a gang this deep buys nothing measurable
here and costs first audio. The C=4 result did not license the C=8 conclusion and both were
measured.

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

### A fix that does NOT apply to this machine

The same microbenchmark found that the fallback matmat twins were missing fixed-width kernels —
bf16 for B=9..15, int8 for B=1, 5, 7 and 9..15 — and paid 5-10x for them. That is a real defect
and it is fixed, but **it changes nothing here**, and the dispatch counters say so rather than
leaving it to be assumed. On real 1.7B prefill shapes at C=4:

```
SIMD=amx           fallback twin dispatch: never reached
QWEN_NO_AMX=1      fallback twin dispatch: never reached
```

Every batched call on this host is taken by a wider matmat: `bf16 AMX tiles` and `int8 AMX
tiles`, or with AMX disabled `bf16 AVX-512 dpbf16` and `int8 VNNI vpdpbusd` — the same GMAC
moving between named kernels. The twin is only reached by builds without AVX-512, which on x86
means the AVX2 / `SIMD=portable` level, and on Arm by a build with no BF16 matrix unit, which is
where the 5-10x was measured.

It is not nothing on *every* x86 build. At `SIMD=portable` the dispatcher does reach the twin for
bf16 — `--caps` shows `bf16 -> fixed-B twin` at every batch width — and the same harness compiled
against both trees measures, on this host's AVX2 kernels:

| width | bf16 before → after | int8 before → after |
|---|---|---|
| B=9 | **11.87 → 2.25 ms** | unchanged |
| B=13 | **10.80 → 7.97 ms** | unchanged |
| B=1 | 1.44 → 1.30 ms | **10.03 → 1.32 ms** |
| B=5, 7, 16 | unchanged | unchanged |

int8 moves only at B=1 because this build sends B=2..16 to `int8 AVX2 maddubs`, which the caps
table says and the counters confirm. **A kernel fix is worth what the dispatcher lets it be
worth**: 5-10x at the portable level and on an Arm build with no BF16 unit, and zero on the AMX
or AVX-512 builds this page is otherwise about.

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
