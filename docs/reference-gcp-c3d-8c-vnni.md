# GCP `c3d-standard-16` — 8 physical cores, AVX-512 VNNI/BF16, thin DRAM slice

This is a fast curiosity screen of the GCP x86 box, not a deployment qualification. The VM
exposes 16 vCPUs from 8 physical AMD cores, and its measured memory bandwidth is only about a
third of the AWS c8a Zen 5 reference. That prediction is borne out by the first server run:
Base 1.7B int8 reaches **157 ms TTFA at C=1**, but its sustained stream RTF is already **1.12**;
at C=4 it produces **0.94 req/s**, with **462 ms TTFA p95** and **2.24 stream RTF**.

The source snapshot was copied from the dirty x86 branch to the VM. No source fix was made for
this screen.

## Setup

| | measured value |
|---|---|
| VM | GCP `c3d-standard-16`, `us-central1-f`, non-preemptible |
| CPU | AMD EPYC `9B14` (Genoa / Zen 4 family), 1 socket |
| cores | **8 physical**, 16 vCPU exposed; SMT disabled for the final measurements, CPUs `0-7` online |
| memory / NUMA | about 62.8 GiB, no swap, 1 NUMA node |
| cache | 32 MiB L3, one shared instance, 4 MiB per physical core |
| ISA | AVX2, AVX-512F/BW/VL/DQ/CD, `avx512_vnni`, `avx512_bf16`; **no AMX** |
| build | `make server-hw-check`, GCC 15, OpenBLAS pthread, `SIMD=avx512bf16` |
| binary | `1e897e7c64963947af05478c1b1f0ce5b188886a9af18175210e857de0380253` |
| source snapshot | `7bea5f3-dirty:2dc8984b8e92` |
| model | `qwen3-tts-1.7b-base`, open weights, `--int8` |

The snapshot was measured on 2026-09-04. `lscpu` continues to print `CPU(s): 16` after SMT is
disabled, but only `0-7` are online and `nproc` returns 8; the online set is the authoritative
post-SMT topology.

## 1. DRAM bandwidth and SMT

The comparable measurement is three 512 MiB `double` buffers, best of seven repetitions. The
final SMT-off sweep was:

| threads | Copy GB/s | Triad GB/s |
|---:|---:|---:|
| 1 | 24.51 | 24.78 |
| 2 | 24.70 | 29.59 |
| 4 | 25.45 | 30.12 |
| 8 | 26.53 | **32.45** |

The 32.45 GB/s peak is reached with all 8 physical cores. The smaller default `box_info.sh`
probe (128 MiB buffers) measured 30.56 GB/s and reported a knee at 4 threads; the honest
description is therefore **bandwidth starts flattening around 4 threads and is at its roof by
8**. The 460 GB/s figure reported for c3d is the full physical socket estimate, not the VM's
share, and must not be used for planning this slice.

The SMT A/B was effectively neutral:

| arm | peak Triad |
|---|---:|
| SMT on, physical candidates `0-7` | 32.16 GB/s |
| SMT on, all logical CPUs `0-15` | 32.41 GB/s, with 29.33 at the 16-thread point |
| SMT off, online CPUs `0-7` | **32.45 GB/s** |

Disabling SMT does **not** create DRAM bandwidth here: the clean A/B gain is about 0.1%.
It is still the right measurement mode because it removes sibling contention, makes the core
count truthful, and avoids scheduling work onto the second hardware thread. Use 8 online cores,
not 16 vCPUs, when choosing the serving topology.

### Against AWS c8a.4xlarge

The AWS reference uses the same 512 MiB protocol on a 16-core EPYC 9R45 Zen 5:

| threads | 1 | 2 | 4 | 8 | 16 | peak |
|---:|---:|---:|---:|---:|---:|---:|
| GCP c3d Triad | 24.78 | 29.59 | 30.12 | **32.45** | — | **32.45** |
| AWS c8a Triad | 44.1 | 84.9 | 100.6 | 103.0 | **103.6** | **103.6** |

The GCP slice has **31% of the AWS total roof**; AWS is about **3.2× faster** at peak and
about 3.3× faster at four threads. Normalised by physical core, the difference is smaller but
still material: 4.06 GB/s/core on GCP versus 6.48 GB/s/core on AWS, or about 1.6× for AWS.
Both expose roughly 4 MiB of L3 per core, so the decisive difference for the 1.7B is the memory
roof, not the per-core LLC budget.

## 2. Server-mode mini-benchmark

The CLI matrix was stopped as requested. The useful run used the server harness and a true
synchronized wave:

```bash
python3 tests/serve_parallel_wave.py \
  --model qwen3-tts-1.7b-base --precision int8 \
  --profile aws-c8a-16c-vnni-ttfa \
  --topo 2x4 --conc 1,2,4 --waves 1 --classes short
```

The AWS profile was used for its measured VNNI/TTFA environment, while `2x8` was adapted to
`2x4` because this VM has only 8 physical cores. The server ran two prefork workers, four
threads per worker, batch cap 8 per worker, `max-queue=1`, and zero queue timeout. The harness
used `/v1/tts/stream`; every cell had zero errors and zero rejections, and the C=1 harness
cross-check passed within 1%.

The following are the clean timing results, without costmap or serve-profiler overhead. `B` is
the measured in-flight batch, not the requested concurrency.

| topology | C | TTFA p50 / p95 | stream RTF | total RTF | req/s | B |
|---|---:|---:|---:|---:|---:|---:|
| `2x4` | 1 | 157 / 157 ms | 1.124 | 1.164 | 0.47 | 0.49 |
| `2x4` | 2 | 208 / 492 ms | 1.736 | 1.777 | 0.63 | 1.19 |
| `2x4` | **4** | **444 / 462 ms** | 2.238 | 2.369 | **0.94** | **2.54** |

The important distinction is TTFA versus sustainable playback. C=4 passes a 500 ms TTFA p95
threshold in this one-wave screen, but its stream RTF is 2.24 and it needs about 2.4 seconds of
prebuffer to avoid underrun. This is a batching-throughput result, **not realtime serving**.

### `1x8` topology A/B

The single-worker shape is a useful no-code experiment because `2x4` leaves one worker idle at
C=1 and splits C=4 into two smaller batches. With the same Base 1.7B int8 profile and one wave:

| topology | C | TTFA p50 / p95 | stream RTF | total RTF | req/s |
|---|---:|---:|---:|---:|---:|
| `1x8` | 1 | 142 / 142 ms | **1.054** | 1.088 | 0.55 |
| `1x8` | 2 | 228 / 268 ms | 1.448 | 1.524 | 0.78 |
| `1x8` | 4 | 494 / 532 ms | **1.971** | 2.194 | **1.05** |

Against `2x4`, `1x8` improves C=1 stream RTF by about 6% and C=4 throughput by about 12%,
but it moves C=4 TTFA p95 above 500 ms. The single worker is therefore worth keeping as a
low-concurrency/throughput arm, not as a way to make the 1.7B realtime on this memory slice.

### The 0.6B control

The smaller model shows that the limitation is not only the server scheduler. It was tested
without code changes, still int8 and with the same AWS-derived runtime profile:

| topology | C | TTFA p50 / p95 | stream RTF | total RTF | req/s |
|---|---:|---:|---:|---:|---:|
| `1x8`, SMT off | 1 | 89 / 89 ms | **0.679** | 0.695 | 0.64 |
| `1x8`, SMT off | 2 | 124 / 166 ms | **0.917** | 0.945 | 0.65 |
| `1x8`, SMT off | 4 | 272 / 310 ms | 1.500 | 1.557 | 0.74 |
| `2x4`, SMT off | 1 | 102 / 102 ms | **0.765** | 0.784 | 0.54 |
| `2x4`, SMT off | 2 | 131 / 184 ms | **0.956** | 0.989 | 0.60 |
| `2x4`, SMT off | 4 | 202 / 207 ms | 1.462 | 1.473 | 0.74 |
| `2x8`, SMT on | 1 | 88 / 88 ms | **0.678** | 0.693 | 0.61 |
| `2x8`, SMT on | 2 | 147 / 221 ms | 0.958 | 1.002 | 0.59 |
| `2x8`, SMT on | 4 | 203 / 205 ms | 1.491 | 1.502 | 0.73 |

The `1x8` versus `2x4` rows are the fair SMT-off comparison. `1x8` is slightly better in stream
RTF at C=1/2, while `2x4` gives lower TTFA at C=4 and the same C=4 throughput; neither shape
changes the basic conclusion. `2x8` was run with SMT temporarily enabled so that its 16 worker
threads had 16 online logical CPUs; it is not an 8-vs-16 physical-core comparison and does not
beat either physical-core topology meaningfully. On this GCP, the 0.6B int8 is the first
configuration that is comfortably realtime at C=1 and near the line at C=2, while the 1.7B
remains above it.

For orientation, the AWS c8a reference on 16 physical cores reports `2x8` values of
`99/102 ms`, `101/120 ms`, and `186/199 ms` TTFA p50/p95 at C=1/2/4, with stream RTF
`0.710/0.721/0.952` and throughput `0.83/1.54/2.25 req/s`. The GCP result is directionally
consistent with its much smaller core and DRAM budget, but this GCP run has one wave and is a
screen, not a same-day qualification campaign.

## 3. SIMD and batching engagement proof

The binary and the server log confirm the intended native paths:

- `--caps` resolved `SIMD=avx512bf16`, runtime `isa_class=x86_avx512bf16`, native
  `_mm512_dpbusd_epi32` VNNI int8 dot, AVX-512 BF16 `VDPBF16PS`, and AVX-512 BF16 matmat.
- Native and scalar/VNNI-disabled self-tests both passed; the dispatch gate passed.
- The server `[FLAGS]` line verified `QWEN_PREFILL_MATMAT=1`, `QWEN_CP_PREFILL2=1`, prefix
  cache, VNNI GEMV MR=2, decode chunk 8, and `QWEN_DECODER_BATCH=0`.
- The server banner reported continuous request-batching with `max_batch=8`, and at C=4 both
  workers were assigned two requests.

The diagnostic twin run enabled `QWEN_SERVE_PROFILE=1`, `QWEN_BATCH_STATS=1`, and the branch's
costmap without changing code. Its batch audit showed, cumulatively on a worker across the
short cells:

| observed kernel | calls | time |
|---|---:|---:|
| AVX-512 BF16 `dpbf16` matrix-matrix | 6,796 | 129.89 ms |
| int8 VNNI `vpdpbusd` matrix-matrix | 50,424 | 121.36 ms |
| int8 VNNI GEMV | 69,219 | 416.96 ms |
| fallback twin | never reached |

The log gates were `bf16 AVX-512 dpbf16: ON` and `int8 VNNI vpdpbusd: ON`; AMX was absent and
not compiled. This is the requested proof that the VNNI/BF16 512-bit routes are active in server
batching, rather than merely present in `/proc/cpuinfo`.

The same check was repeated inside a diagnostic `1x8` server run. The resolved profile was:

```text
QWEN_PREFILL_MATMAT=1  QWEN_PREFIX_CACHE=1  QWEN_CP_PREFILL2=1
QWEN_VNNI_GEMV_MR=2   QWEN_STREAM_DECODE_CHUNK=8
QWEN_STREAM_DECODE_CHUNK_BUSY=0  QWEN_DECODER_BATCH=0  QWEN_POOL_SPIN=4096
```

All `QWEN_NO_VNNI*` and `QWEN_NO_BF16*` opt-outs were absent, so the native paths were not
disabled. The dispatch map resolved BF16 AVX-512 `dpbf16` **ON**, int8 VNNI `vpdpbusd` **ON**,
and decoder int8 **ON**. `QWEN_DECODER_BATCH=0` is intentionally off from the AWS profile, and
`QWEN_VNNI_PREPACK` is intentionally absent/off because it was rejected as an end-to-end win on
the reference Zen 5. The `1x8` batch audit counted 100,560 int8 VNNI matrix-matrix calls and
12,272 BF16 `dpbf16` matrix-matrix calls; the fallback twin was never reached. This confirms
that `1x8` is actually reusing weights within batched matmat calls, although its single worker
cannot provide the same worker-assignment counters as a prefork topology.

One branch-level defect was observed in the measured remote snapshot and deliberately left
untouched: `make cpu-check` reports only the flag-registry failure because `qwen_tts_costmap.c`
reads `QWEN_COSTMAP_JSON` and `QWEN_COST_MAP`, while those names are not yet in
`g_qwen_reported_flags[]` in that snapshot. Hardware, compiler, ISA, native/fallback self-tests,
dispatch, SMT, governor, and cgroup gates passed. The registry issue makes `CPU_CHECK_VALID=NO`;
it does not indicate a failed AVX/VNNI dispatch.

## 4. Verdict and limits

This VM is useful as a cheap AVX-512/VNNI correctness and batching test target, but the measured
DRAM slice is poor for the 1.7B. The fast server screen confirms the expected outcome: batching
does form and reaches 1.05 req/s at C=4 with `1x8`, yet sustained RTF is still 1.97; `2x4` has
slightly better C=4 TTFA but lower throughput. There is no configuration-only evidence here for
1.7B realtime at C=4. The 0.6B int8, by contrast, reaches RTF 0.68 at C=1 and 0.92 at C=2 on
`1x8`, without sacrificing the main model's int8 quality target.

This page intentionally does not claim a full suite, a broad topology sweep, a realistic
mixed-length campaign, a soak, or production quality. Remote artifacts are under
`/tmp/gcp_c3d_serve_fast*`, `/tmp/gcp_c3d_serve_1x8`, `/tmp/gcp_c3d_serve_06b_*`, and
`/tmp/gcp_c3d_costmap_*.json` on the VM; the copied source snapshot and binary hash above are the
provenance for the reported runs.
