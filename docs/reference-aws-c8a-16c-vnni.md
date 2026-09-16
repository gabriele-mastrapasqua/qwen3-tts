# An AWS `c8a.4xlarge` — 16 real cores, VNNI only, and the best cost per stream measured

This is the x86 box for a project that wants **one dispatcher to optimise**. Sixteen physical
cores, no SMT to disable, one NUMA node, AVX-512 with VNNI and BF16, **no AMX** — so there is no
second matrix path to qualify, no `QWEN_AMX_*` lever to re-derive per host, and nothing about
the machine that has to be learned before it can be used.

It is also, on every cell measured, the fastest and the cheapest of the five AWS instances
screened on 2026-09-03.

**Headline: `2x8` holds four concurrent realtime streams** — C=4 stream RTF **0.952**, with p95
0.979 and even the worst single request at 0.999. That is 42% cheaper per realtime stream than
the AMX box, and it was not predictable: the arithmetic beforehand said ~1.04, just above the
line. It had to be measured.

Structurally this is the closest x86 to the Arm reference in
[`reference-arm-16c.md`](reference-arm-16c.md) — same core count, same absent SMT, same single
NUMA node, same `2x8` answer — which is why an Axion-shaped workflow ports here with no new
concepts. It is **not** its equal in bandwidth; see §6.

## The setup

| | measured value |
|---|---|
| instance | AWS **`c8a.4xlarge`**, `us-east-1b` |
| CPU | **AMD EPYC 9R45** (Zen 5), 1 socket |
| cores | **16 physical**, SMT **not supported** — nothing to disable, nothing to declare |
| NUMA / memory | 1 node, 30.8 GiB |
| ISA | AVX-512F/BW/VL/DQ/CD, `avx512_vnni`, `avx512_bf16` — **no AMX** |
| L1d / L2 | 48 KiB · 1.0 MiB per core |
| **L3** | **32 MiB × 2 CCDs = 64 MiB**, 4.0 MiB per core — a real figure, see §2 |
| measured Triad | **103.6 GB/s**, peak at 16 threads, **knee at 4** |
| build | `make blas`, `SIMD=auto` → **`avx512bf16`**; gcc 15, OpenBLAS |
| model | `qwen3-tts-1.7b-base`, open weights, `--int8` |
| profile | [`scaleway-16c-vnni-ttfa`](../configs/perf/scaleway-16c-vnni-ttfa.json), verified in every server's `[FLAGS]` line |
| bank | `tests/load_texts_en.txt`, class `short`, speaker `ryan`, seed 42 |
| gates | SMT **PASS** · no cgroup quota **PASS** · frequency stable **PASS** |

```text
source_commit=a0e38d3 (feature/x86-amx-vnni-oss)  dirty=yes
source_diff_sha256=f32f37fc08462c3e0f019d044a533aed007f156c8802395b9cf86a2a2f2768de
binary_sha256=b3c7bdea53631130f7e13f421d765799c3c9e9649121e55cf3683c27ce96dd06
serving=make bench-topo, TRUE_SIMULTANEOUS_WAVE, 3 waves per cell, one fresh server per cell
```

Dirty tree, three waves per cell, one bank, one class: **a screening campaign, not a
qualification.** No realistic mixed-length bank, no open-loop arrivals, no soak.

## 1. Bandwidth

`box_info.sh --membw`, three 512 MiB `double` buffers, best of five, idle box:

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Triad GB/s | **44.1** | 84.9 | 100.6 | 103.0 | **103.6** |

Peak 103.6 GB/s at 16 threads, **knee at 4**. Sustained series, four back-to-back runs:
102.0 / 103.1 / 103.2 / 101.2 at 8 threads and 102.6 / 102.9 / 103.2 / 102.6 at 16 — a 2.0% and
0.6% spread, the tightest measured in this set.

**Two numbers define this machine.** Per-core bandwidth is **44.1 GB/s**, 2.2× what the Intel
boxes deliver, which is what prices a single stream. And the knee is at **4 of 16 threads**:
going from 4 to 16 threads adds 3.0% of bandwidth, so twelve cores are free for compute that is
not memory-bound. The Intel 8-core boxes have to spend every core just to reach their roof.

An honest note on the roof. An earlier probe of this same instance type measured a peak of
106.5, and `r8a.4xlarge` — identical silicon — measured 106.3. This run measured 103.6 on a box
that had been up five hours with 8 GiB in page cache. **Read the AMD roof as 103–106.5 GB/s,
not as a single number**; the ~3% band is instance and state variation, and it changes nothing
below.

## 2. The L3 is a real number here

Every Intel box in this set reported 480 MiB of L3 in one instance — the whole socket, seen by a
guest owning eight of its cores — and `box_info.sh` turned that into "the 1.7B working set
FITS", which is not believable. AMD exposes L3 per CCD, so this guest sees what it owns:

```text
usable LLC   64 MiB  (L3, sum of 2 instances), 4.0 MiB per physical core
             32768K shared by cpu 0-7
             32768K shared by cpu 8-15
```

With a credible number the verdict flips and can be trusted: the 0.6B CP working set (~60 MB)
fits, the **1.7B (~120 MB) does not**. That is consistent with everything below — this box is
bandwidth-bound on the 1.7B, and its advantage is that it has bandwidth to spare per core.

The two L3 domains are exactly `cpu 0-7` and `cpu 8-15`. **A `2x8` topology maps one worker per
CCD** — which is also the topology that won §3. Whether the CCD alignment is *why* it won was
not tested; that needs an explicit affinity A/B.

## 3. The topology sweep

`make bench-topo BENCH_TOPO=1x16,2x8,4x4 BENCH_CONC=1,2,4,6,8`, three synchronized waves per
cell, one fresh server per cell. **Zero errors and zero rejections in all fifteen cells.** RTF is
`STREAM_RTF p50`.

| topology | C | TTFA p50 | TTFA p95 | stream RTF | req/s | batch in system |
|---|---:|---:|---:|---:|---:|---:|
| `1x16` | 1 | **72 ms** | **74 ms** | **0.523** | 1.19 | — |
| `1x16` | 2 | 133 ms | 170 ms | 0.731 | 1.51 | — |
| `1x16` | 4 | 231 ms | 274 ms | 1.100 | 1.96 | — |
| `1x16` | 6 | 327 ms | 389 ms | 1.473 | 2.12 | — |
| `1x16` | 8 | 435 ms | 536 ms | 1.882 | 2.22 | — |
| **`2x8`** | 1 | 99 ms | 102 ms | 0.710 | 0.83 | 0.72 |
| **`2x8`** | 2 | 101 ms | 120 ms | 0.721 | 1.54 | 1.39 |
| **`2x8`** | **4** | 186 ms | 199 ms | **0.952** | 2.25 | 2.93 |
| **`2x8`** | 6 | 249 ms | 276 ms | 1.128 | 2.77 | 4.54 |
| **`2x8`** | 8 | 328 ms | **364 ms** | 1.300 | **3.15** | 6.21 |
| `4x4` | 1 | 115 ms | 140 ms | 0.765 | 0.75 | 0.71 |
| `4x4` | 2 | 129 ms | 243 ms | 1.129 | 1.04 | 1.51 |
| `4x4` | 4 | 171 ms | 206 ms | 1.169 | 1.99 | 2.88 |
| `4x4` | 6 | 198 ms | 306 ms | 1.382 | 2.25 | 4.47 |
| `4x4` | 8 | 282 ms | 319 ms | 1.516 | 2.86 | 6.23 |

**`2x8` is the profile.** It takes throughput at every level from C=2 up, holds TTFA p95 under
400 ms all the way to C=8, and is the only topology whose C=4 stream RTF is below 1.0.

**`1x16` is the single-stream choice**: 72 ms to first audio, stream RTF 0.523, and a prebuffer
of 12 ms — playback can start **84 ms** after the request and never stall. That is the best
single-stream latency of any x86 box measured here.

`4x4` is dominated: it loses to `2x8` on RTF at every level and never wins throughput. Note its
C=2 anomaly (RTF 1.129, worse than its own C=4) — four narrow workers with two requests leave
half the machine idle while each request runs on four threads.

### The latency a listener actually feels

> Reading note (2026-09-07): the prebuffer below is the zero-buffer client diagnostic of the
> harness at the time; fixed-buffer stall rates and per-request `safe_play_start` were not
> measured, so "never stall"/"gapless" in this section is not a qualified continuity claim.
> Definitions: `docs/serving-operations.md` §5.

TTFA plus the prebuffer that makes the stream gapless:

| C | `1x16` | `2x8` | `4x4` |
|---:|---:|---:|---:|
| 1 | **84 ms** | 157 ms | 258 ms |
| 2 | 223 ms | **176 ms** | 761 ms |
| 4 | 871 ms | **625 ms** | 826 ms |
| 8 | 2271 ms | **1294 ms** | 1509 ms |

The crossover is at C=2, as on the Intel box: below the realtime line the prebuffer is tens of
milliseconds, above it the buffer has to absorb the whole deficit.

## 4. Against the AMX box, on the same binary

This is the comparison that matters, and it is unusually clean: **the same commit, the same
patch, the same binary source, the same harness, the same bank, measured hours apart on the same
day** — each box on its own best topology.

| C | `c8a` `2x8` TTFA p50 | `c8i` `2x4` TTFA p50 | `c8a` stream RTF | `c8i` stream RTF | `c8a` req/s | `c8i` req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **99 ms** | 121 ms | **0.710** | 0.790 | 0.83 | 0.65 |
| 2 | **101 ms** | 123 ms | **0.721** | 0.832 | 1.54 | 1.24 |
| 4 | **186 ms** | 235 ms | **0.952** | 1.304 | 2.25 | 1.51 |
| 6 | **249 ms** | 299 ms | **1.128** | 1.659 | 2.77 | 1.77 |
| 8 | **328 ms** | 393 ms | **1.300** | 2.013 | **3.15** | 1.92 |

**The VNNI-only box wins every cell**: 16–27% better TTFA, 10–35% better stream RTF, and **64%
more throughput at C=8**. Single stream it also wins — `1x16` at 72 ms and RTF 0.523 against
`1x8` at 90 ms and 0.571.

This does not mean AMX is worthless; it means **AMX did not compensate for having half the
cores** on this workload at these batch widths. The AMX attribution on that box says why: AMX
INT8 was worth nothing measurable there, because the batch per worker was about 1.5 and the
tiles are only reached at B≥2, so the only AMX term that paid was BF16 prefill. Sixteen cores
and 2.2× per-core bandwidth beat one matrix unit that the dispatcher barely reaches.

## 5. Cost

On-demand, Linux, shared tenancy, list price, captured 2026-09-03.

| | `us-east-1` | `eu-west-1` | realtime streams | **$ per stream·hr** | req/s at C=8 | $ per req/s |
|---|---:|---:|---:|---:|---:|---:|
| **`c8a.4xlarge`** | **$0.862** | $0.925 | **4** (C=4, RTF 0.952) | **$0.216** | **3.15** | **$0.274** |
| `c8i.4xlarge` | $0.7497 | ~$0.81–0.84 | 2 (C=2, RTF 0.832) | $0.375 | 1.92 | $0.390 |

**42% cheaper per realtime stream and 43% more throughput per dollar**, while costing 15% more
per hour. Per core it is cheaper still: $0.0539 against $0.0937, a 42% saving.

## 6. What it is not: an Axion

The structural resemblance to [`reference-arm-16c.md`](reference-arm-16c.md) is real — 16 cores,
no SMT, one NUMA node, `2x8`, and the same absence of an exotic unit to manage. The performance
resemblance is not:

| | Arm 16c reference | `c8a.4xlarge` |
|---|---:|---:|
| cores / SMT | 16 / absent | 16 / absent |
| LLC per core | 5.0 MiB | 4.0 MiB |
| **measured Triad** | **336 GB/s** | **103.6 GB/s** |
| **knee** | **16 threads** | **4 threads** |
| `2x8` C=1 TTFA · RTF | 54 ms · 0.43 | 99 ms · 0.710 |
| `2x8` C=4 TTFA · RTF | 124 ms · 0.72 | 186 ms · 0.952 |

Arm has **3.2× the bandwidth** and a knee at every core it owns; this box saturates DRAM with a
quarter of its cores. Both hold four realtime streams at `2x8`, but Arm holds them with margin
(0.72) and this one holds them at the edge (0.952, max 0.999).

So the right expectation is: **the workflow ports, the numbers do not.** Same profile shape, same
topology answer, same levers — and roughly 30% worse latency and RTF at the same concurrency.

## 7. Kernel cells, and the standing int4 defect

`--matmat-bench` at B=8, single thread — batched matmat against `B×matvec`:

| shape | bf16 | int8 | **int4** |
|---|---:|---:|---:|
| 3072×1024 | 2.79× | 2.18× | **0.97×** |
| 1024×3072 | 2.48× | 2.87× | **0.96×** |
| 2048×1024 | 2.68× | 2.94× | **0.96×** |

**Batched q4 is slower than sequential.** This reproduces exactly the open follow-up recorded in
[`hardware-testing.md`](hardware-testing.md) — *"batched q4 matmat is now 0.80× vs the faster seq
matvec"* — on a third machine, and it is the single clearest optimisation target on this box:
`q4_matmat_vnni_slice` gets no reuse from batching. The AMX box reaches 4.2× on the same cells,
which confirms the deficit is in the VNNI slice and not in the idea of batching q4.

`--caps` reports `SIMD=avx512bf16`, VNNI and `VDPBF16PS` native, and a dispatcher that stays on
`bf16 AVX-512 dpbf16` / `int8 VNNI vpdpbusd` at **every** batch width from B=1 to B=16 — one
path, which is the point of this box. `--self-test` PASSED, 0 failures, natively and with
`QWEN_NO_VNNI=1 QWEN_NO_SDOT=1`.

The `-j16` matmat cells are not reported: 0.01 ms resolution on 0.01–0.15 ms timings, so the
apparent 32× ratios are artifacts of the sequential arm, not signal.

## 8. What this page does not claim

Screening, not qualification: dirty tree, three waves, one bank, one class, no `bench-suite`, no
soak, no open-loop arrivals, no per-class percentiles. **The C=4 realtime result sits at 0.952
with a worst case of 0.999 — it is a threshold result with no margin, and a soak is exactly what
would tell you whether it survives.** That is the first thing to run if this box is chosen.

The comparison against the Scaleway 16-core table in
[`reference-scaleway-16c-vnni.md`](reference-scaleway-16c-vnni.md) was deliberately not made a
headline: that campaign ran on a different commit, and its `2x8` C=1 figure (0.47) is far better
than this box's (0.710) in a way bandwidth does not explain. Either the binary or the host
differs in something unrecorded. **Same-binary comparisons only** — which is what §4 is.

## Reproducing this page

```bash
gcc -O2 -pthread -o membw tests/membw.c
bash tools/box_info.sh --membw ./membw --out box.json     # no SMT to disable on this part
make blas && ./qwen_tts --caps && ./qwen_tts --self-test
QWEN_NO_VNNI=1 QWEN_NO_SDOT=1 ./qwen_tts --self-test
./qwen_tts --matmat-bench -j 1

bash download_model.sh --model base-large --dir qwen3-tts-1.7b-base
make bench-topo BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=scaleway-16c-vnni-ttfa \
                BENCH_TOPO=1x16,2x8,4x4 BENCH_CONC=1,2,4,6,8
```

Note that `scaleway-16c-vnni-ttfa.json` is untracked, so a `git archive` of the branch does not
carry it; copy it onto the box explicitly. The next step for this host is a `c8a`-specific
profile — the Scaleway one was borrowed to get comparable numbers, and its pins were measured on
a different machine.
