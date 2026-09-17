# An AWS `c8i.4xlarge` (Granite Rapids) — 8 cores, AMX, and 28% more bandwidth

This page exists to answer one question with a controlled experiment rather than a spec sheet:
**on an 8-core AMX box, what does more memory bandwidth buy?**

The comparison is unusually clean. [`reference-x86-8c-amx.md`](reference-x86-8c-amx.md) measured
an 8-core Intel Emerald Rapids host — same core count, same one NUMA node, same AMX INT8 and
BF16 — at **82 GB/s**. This instance is the same shape at **104.8 GB/s**. Same engine, same
profile, same bank, same harness, same commands. Everything that differs between the two tables
is bandwidth and microarchitecture, and the answer is specific: **it buys sustained RTF and
throughput, and it does not buy first audio.**

Read it together with [`x86-box-selection.md`](x86-box-selection.md), which is where this
instance is weighed against the AMD candidates.

**Status: closed as a screening, not qualified.** What was run: the silicon probe, the ISA and
correctness gates, the kernel cells, a fifteen-cell topology sweep and the AMX attribution A/B.
What was not: `make bench-suite` — the realistic mixed-length bank, the open-loop arrivals and
the soak. The instance was released after this page. Everything below is reproducible from
§Reproducing; nothing below should be quoted as a qualification.

## The setup

| | measured value |
|---|---|
| instance | AWS `c8i.4xlarge`, `us-east-1` |
| CPU | Intel Xeon 6975P-C (Granite Rapids, family 6 model 173 stepping 1), 1 socket |
| cores | **8 physical**, 16 vCPU as delivered — see §1 |
| NUMA | 1 node, 30.8 GiB |
| ISA | AVX-512F/BW/VL/DQ/CD, `avx512_vnni`, `avx512_bf16`, **`amx_tile` + `amx_int8` + `amx_bf16`**, plus `avx512_fp16`, `avx_vnni`, `movdir64b`, `serialize`, `cldemote` |
| L1d / L2 | 48 KiB · 2.0 MiB per core |
| L3 as reported | 480 MiB, one instance — **do not trust this number, see §8** |
| measured Triad | **104.8 GB/s**, knee at 8 threads (SMT off) |
| governor | not exposed; the hypervisor owns the frequency |
| build | `make blas` with `SIMD=auto`, which selects **`amx`**; gcc 15.2.0, OpenBLAS, kernel `7.0.0-1006-aws` |
| model | `qwen3-tts-1.7b-base`, open weights, `--int8` |
| profile | [`x86-8c-amx-recommended`](../configs/perf/x86-8c-amx-recommended.json) → `x86-8c-amx-multiclient-ttfa`, verified in every server's `[FLAGS]` line |
| bank | `tests/load_texts_en.txt`, class `short`, speaker `ryan`, seed 42 |

Provenance for every number below:

```text
source_commit=a0e38d3 (feature/x86-amx-vnni-oss)  dirty=yes
source_diff_sha256=f32f37fc08462c3e0f019d044a533aed007f156c8802395b9cf86a2a2f2768de
binary_sha256=335a0145bca07a1fd0f5e0bcc79c578a16106aec1681fd8f6dd7e80a3e9bb58c
probe=tools/box_info.sh --membw <tests/membw.c>   membw: 3 x 512 MiB double, best of 5
serving=make bench-topo, TRUE_SIMULTANEOUS_WAVE, 3 waves per cell, one fresh server per cell
```

**The tree was dirty and its source commit was not recorded by the harness.** This is a
screening campaign against an existing baseline, not a qualification: it cannot replace the
identified run that produced the profile it borrows.

## 1. It is delivered as 8 cores wearing 16 vCPUs

`c8i.4xlarge` arrives with SMT **on**: 16 vCPU are 8 physical cores. The sibling map is the
clean interleaved one, `cpu N` paired with `cpu N+8` over `core_id 0..7`, one package, one L3
domain. So the box does not have to be recreated — the hyperthreads can be taken offline:

```bash
echo off | sudo tee /sys/devices/system/cpu/smt/control    # cpu8-15 offline, nproc 16 -> 8
```

After that `tools/box_info.sh` reports **all three gates PASS**. The change is runtime-only and
does not survive a reboot; a campaign has to re-assert it or declare that it did not. Every
serving number on this page was taken with SMT off, which is also what the profile requires.

Comparing this instance to a 16-vCPU AMD instance on vCPU count compares 8 cores against 16.

## 2. Memory bandwidth

Triad, three 512 MiB `double` buffers, best of five, in both SMT states:

| threads | SMT on (16 vCPU / 8 cores) | **SMT off (8 cores)** |
|---:|---:|---:|
| 1 | 20.1 | **20.1** |
| 2 | 39.1 | **37.9** |
| 4 | 70.2 | **69.9** |
| 8 | 105.6 | **104.8** |
| 16 | 111.8 | — |
| peak / knee | 111.8 GB/s @16 | **104.8 GB/s @8, knee at 8** |

SMT adds 6.7% of peak on a pure streaming kernel — two threads hiding DRAM latency on one core,
not extra compute — and changes nothing per core. The right-hand column is the number to carry.

Two properties matter more than the peak. **Per-core bandwidth is 20.1 GB/s**, which is what
prices a single stream on a workload whose Code Predictor re-reads its weights sixteen times per
frame. And **the knee is at 8 threads, every core the instance has**: there is no configuration
where some cores saturate DRAM while others do compute.

## 3. The ISA fires, and the dispatcher agrees

`make blas` auto-selects `SIMD=amx`. From `--caps`:

```text
build:            unknown-dirty · SIMD=amx
int8 dot:         VNNI _mm512_dpbusd_epi32 (native)
bf16 dot:         VDPBF16PS _mm512_dpbf16_ps (native)
x86 amx int8:     AMX ACTIVE (tile 16x64 int8 GEMM for batched matmat)
x86 amx bf16:     AMX ACTIVE (tile 16x32 bf16 GEMM for batched matmat)
```

The kernel-selection table names the batch width at which the machine changes character:

| B | bf16 | int8 | q4_0 |
|---:|---|---|---|
| 1 | 2-row fused matvec | VNNI `vpdpbusd` | VNNI `vpdpbusd` |
| 2 | AVX-512 `dpbf16` | VNNI `vpdpbusd` | VNNI `vpdpbusd` |
| **4** | **bf16 AMX tiles** | **int8 AMX tiles** | **q4 AMX tiles** |
| 8, 16 | bf16 AMX tiles | int8 AMX tiles | q4 AMX tiles |

`--self-test` **PASSED, 0 cases failed**, natively and with `QWEN_NO_VNNI=1 QWEN_NO_SDOT=1`. The
batched twins are bit-exact against `B×matvec` on the AMX path (`matmat_int8(B=8)` and
`matmat_q4_0(B=8)` both `L2_rel=0.00e+00`) — the correctness gate this branch had not yet had on
real AMX silicon.

## 4. The topology sweep

`make bench-topo BENCH_TOPO=1x8,2x4,4x2 BENCH_CONC=1,2,4,6,8`, three synchronized waves per
cell, one fresh server per cell. **Zero errors and zero rejections in all fifteen cells.** RTF is
`STREAM_RTF p50` — the per-request figure after the first chunk, which is the column the sibling
page reports and the one the profile's `measured` block stores.

| topology | C | TTFA p50 | TTFA p95 | stream RTF | req/s | batch in system |
|---|---:|---:|---:|---:|---:|---:|
| `1x8` | 1 | **90 ms** | **92 ms** | **0.571** | 0.90 | — |
| `1x8` | 2 | 133 ms | 175 ms | 1.000 | 1.03 | — |
| `1x8` | 4 | 282 ms | 321 ms | 1.550 | 1.28 | — |
| `1x8` | 6 | 404 ms | 521 ms | 2.314 | 1.26 | — |
| `1x8` | 8 | 519 ms | 640 ms | 2.827 | 1.39 | — |
| `2x4` | 1 | 121 ms | 122 ms | 0.790 | 0.65 | 0.75 |
| `2x4` | 2 | 123 ms | 144 ms | **0.832** | 1.24 | 1.45 |
| `2x4` | 4 | 235 ms | 242 ms | 1.304 | **1.51** | 3.08 |
| `2x4` | 6 | 299 ms | 351 ms | 1.659 | **1.77** | 4.77 |
| `2x4` | 8 | 393 ms | 454 ms | 2.013 | **1.92** | 6.42 |
| `4x2` | 1 | 185 ms | 187 ms | 1.326 | 0.39 | 0.82 |
| `4x2` | 2 | 188 ms | 297 ms | 1.289 | 0.78 | 1.54 |
| `4x2` | 4 | 201 ms | 275 ms | 1.375 | 1.41 | 2.98 |
| `4x2` | 6 | 312 ms | 371 ms | 1.902 | 1.49 | 4.34 |
| `4x2` | 8 | 379 ms | **385 ms** | 2.013 | 1.90 | 6.34 |

Three things are worth stating separately.

- **`2x4` holds two realtime streams.** C=2 is stream RTF **0.832**, p95 0.844. On the Emerald
  Rapids box the same cell was 1.027 — above the line. This is the one difference between the
  two machines that is a change of kind rather than a percentage.
- **`2x4` now keeps TTFA p95 under 500 ms all the way to C=8** (454 ms). The sibling page had to
  stop its claim at C=6, because C=8 measured 483.6 / 495.4 / 500.2 ms across three runs — *on*
  the threshold rather than under it.
- **`4x2` is no longer the throughput sacrifice it is on the reference box.** At C=8 it takes
  1.90 req/s against `2x4`'s 1.92 while holding p95 at 385 ms, so on this host the four-worker
  profile buys a better tail for almost nothing. It remains unusable at low concurrency: C=1 is
  stream RTF 1.33.

`1x8` is still the single-stream choice, and a good one: 90 ms to first audio, stream RTF 0.571,
and an underrun requirement of only 24 ms — playback can start 114 ms after the request and
never stall.

## 4b. The latency a listener actually feels

> Reading note (2026-09-07): the prebuffer in this section is the zero-buffer client
> diagnostic of the harness at the time; fixed-buffer stall rates and per-request
> `safe_play_start` were not measured, so "never stall"/"gapless" here means "under the
> observed arrival timeline with exactly that delay", not a qualified continuity claim.
> Definitions: `docs/serving/cpu-operations.md` §5.

TTFA is when the first chunk arrives, not when playback can begin. A player that starts at TTFA
and then stalls is worse than one that waits. The harness reports the prebuffer that would make
a stream gapless, so the honest single number is **TTFA + prebuffer**:

| C | `1x8` | `2x4` | `4x2` |
|---:|---:|---:|---:|
| 1 | **114 ms** | 299 ms | 1204 ms |
| 2 | 615 ms | **356 ms** | 1122 ms |
| 4 | 1607 ms | **1161 ms** | 1249 ms |
| 6 | 2998 ms | **1757 ms** | 2139 ms |
| 8 | 3867 ms | 2394 ms | **2478 ms** |

The topology choice flips at C=2, and **TTFA alone does not show it**: at C=2 the two candidates
are 133/175 ms against 123/144 ms, close enough to look like a tie, while the playable latency is
615 ms against 356 ms — a 1.7× difference. The cause is in §4: `1x8` at C=2 is stream RTF 1.000,
so the stream barely keeps up with itself and the player has to buffer the whole deficit, while
`2x4` at 0.832 generates faster than realtime and needs almost nothing.

This is the practical form of the realtime threshold: below stream RTF 1.0 the prebuffer is a
couple of hundred milliseconds, above it the prebuffer grows with the length of the utterance.
Any capacity claim for this class of box should be read on this table, not on TTFA.

## 5. What the extra 28% of bandwidth actually bought

Same profile, same topology, same bank, same harness. Left column is the Emerald Rapids
qualification stored in
[`x86-8c-amx-multiclient-ttfa.json`](../configs/perf/x86-8c-amx-multiclient-ttfa.json); right is
this host.

| C | TTFA p50 | TTFA p95 | **stream RTF** | req/s |
|---:|---|---|---|---|
| 1 | 128 → 121 (−6%) | 129 → 122 (−5%) | 0.983 → **0.790 (−20%)** | 0.54 → 0.65 (+20%) |
| 2 | 133 → 123 (−7%) | 211 → 144 (−32%) | 1.027 → **0.832 (−19%)** | 1.04 → 1.24 (+20%) |
| 4 | 239 → 235 (−2%) | 252 → 242 (−4%) | 1.427 → **1.304 (−9%)** | 1.40 → 1.51 (+8%) |
| 6 | 317 → 299 (−6%) | 365 → 351 (−4%) | 1.866 → **1.659 (−11%)** | 1.66 → 1.77 (+7%) |
| 8 | 422 → 393 (−7%) | 484 → 454 (−6%) | 2.278 → **2.013 (−12%)** | 1.77 → 1.92 (+9%) |

**+28% of Triad bought 9–20% of stream RTF and 7–20% of throughput, and 2–7% of TTFA.** The
split is the point. First audio is prefill, prefill is the AMX BF16 path, and both machines have
it — so TTFA barely moves. Sustained RTF is the Code Predictor re-reading its weights, and that
is exactly what the memory roof prices.

The two boxes are not equally far apart everywhere: the effective batch is nearly identical at
every level (3.08 vs 3.06 in system at C=4), so this is not a scheduling difference wearing a
bandwidth costume.

**A caution on the C=2 p95.** The −32% is the largest number in the table and the least solid:
p95 at three waves is a rank statistic over six samples. Across three same-configuration runs of
the C=4 cell on this host, TTFA p95 measured 242, 262 and 262 ms — about ±10%. Treat any p95
delta smaller than that as noise. The stream-RTF column is far steadier: 1.304, 1.306 and 1.310
in the same three runs.

## 6. What AMX is worth here — and where this host disagrees with the reference

`QWEN_NO_AMX` removes two unrelated consumers at once, so the per-family controls are what
attribute anything. `2x4`, C=1 and C=4, one fresh server per arm:

| arm | C=1 TTFA p50/p95 | C=4 TTFA p50/p95 | C=4 stream RTF | C=4 req/s |
|---|---:|---:|---:|---:|
| AMX fully on (control) | 122 / 124 ms | 235 / 262 ms | 1.306 | 1.49 |
| `QWEN_NO_AMX_BF16=1` | **157 / 196 ms** | **299 / 440 ms** | 1.315 | 1.38 |
| `QWEN_NO_AMX_INT8=1` | 121 / 123 ms | 236 / 262 ms | 1.317 | 1.52 |
| `QWEN_NO_AMX=1` | 158 / 198 ms | 297 / 440 ms | 1.299 | 1.43 |

**AMX BF16 is first audio, confirmed and larger here than on the reference box**: removing it
costs 29% of C=1 TTFA p50, 58% of its p95, and 68% of C=4 p95 — far outside the ±10% band
established in §5.

**AMX INT8 is worth nothing measurable on this host.** Stream RTF moves 0.8%, throughput moves
the wrong way by 2%, TTFA is identical to the control down to the millisecond. That is a real
disagreement with the sibling page, where removing it cost 9% of RTF and 10% of throughput. And
the two controls confirm each other: `QWEN_NO_AMX` lands on `QWEN_NO_AMX_BF16`'s numbers
(158/198 against 157/196), which is only possible if the INT8 term is zero.

The mechanism is in the table in §4 rather than in a guess. At C=4 the batch **per worker** is
1.51 and 1.57. `QWEN_AMX_MIN_B=2` means the int8 tile path is only reached at B≥2, so most
decode calls on this host are B=1 GEMV and never arrive at a tile at all. A wider box, or a
topology with fewer and fatter workers, would move this; two workers of four cores does not.

The corollary is that the profile's `QWEN_AMX_MIN_B=2` pin does not port. It was measured on
Emerald Rapids as 1.47 against 1.63 stream RTF. Here, against the compiled default of 4:

| `QWEN_AMX_MIN_B` | C=1 TTFA p50/p95 | C=4 TTFA p50/p95 | C=4 stream RTF | C=4 req/s |
|---|---:|---:|---:|---:|
| 2 (the profile's pin) | 122 / 124 ms | 235 / 262 ms | 1.306 | 1.49 |
| 4 (compiled default) | 119 / 123 ms | 229 / 272 ms | 1.310 | 1.50 |

Indistinguishable, on every column, inside the noise band. **A lever is worth what the
dispatcher lets it be worth**, and on this host the dispatcher does not reach it. If this
instance is ever qualified, it needs its own profile rather than the Emerald Rapids one, and
that profile should re-derive `QWEN_AMX_MIN_B`, `QWEN_DECODER_BATCH` and `QWEN_POOL_SPIN` rather
than inherit three measurements taken on a different machine.

## 7. Kernel cells

`--matmat-bench` at B=8, AMX on against `QWEN_NO_AMX=1` on the same binary. Batched-kernel time
in milliseconds; single thread, the compute-bound and readable reference:

| shape | bf16 AMX / VNNI | int8 AMX / VNNI | int4 AMX / VNNI |
|---|---|---|---|
| 3072×1024 | 0.30 / 0.62 → **2.07×** | 0.17 / 0.27 → **1.59×** | 0.51 / 1.85 → **3.63×** |
| 1024×3072 | 0.25 / 0.50 → **2.00×** | 0.14 / 0.18 → 1.29× | 0.51 / 1.85 → **3.63×** |
| 2048×1024 | 0.20 / 0.40 → **2.00×** | 0.09 / 0.13 → 1.44× | 0.34 / 1.23 → **3.62×** |

At `-j8` the same A/B is directionally identical but its resolution is 0.01 ms on times of
0.02–0.26 ms, so its ratios are not quotable.

Note what §6 does to this table: the int8 column is a real 1.3–1.6× on the kernel and **zero on
the request**, because the request rarely presents a batched int8 call to that kernel. This is
the same lesson the sibling page records from the other direction, and it is the reason a kernel
cell is never promoted to a serving claim here.

### The int4 finding

int4 is the interesting column. On the Zen 5 hosts, batched q4 has been the standing defect —
[`hardware-testing.md`](hardware-testing.md) records the open follow-up that *"batched q4 matmat
is now 0.80× vs the faster seq matvec"*, and the EPYC VNNI validation measured int4 about 37%
slower than int8 at `-j1`. This host reproduces that on the VNNI path and removes it on the AMX
path:

| path | int4 batched vs `B×matvec`, `-j1` |
|---|---:|
| `QWEN_NO_AMX=1` (VNNI) | 1.14 – 1.16× — the known non-win |
| AMX tiles | **4.19 – 4.21×** |

So on a machine with a matrix unit the answer for batched q4 is the tile path, not a better VNNI
slice. That does not retire the VNNI work: it is the answer for this class of host only, and the
AMD candidates in the same rental set have no AMX at all. It says nothing about int4 quality,
and — per §6 — nothing yet about int4 in a request, which was not measured.

### The `B=32` accumulation prototype

The branch head (`a0e38d3`, *prototype AMX B32 accumulation*) had not run on AMX silicon.
`make x86-amx-b32-bench`, 2048×2048, 30 reps: cold 443.1 → 365.3 µs (1.21×), warm 428.0 → 360.6
µs (**1.19×**), `max_abs=0.000e+00` — bit-identical output. Correct and locally faster on real
AMX; nothing about what it is worth in a request.

### The fused QKV wrapper is component-dependent

`make x86-qkv-bench` (4 threads, 15 reps, shared activation), every cell bit-exact:

| component | B=1 | B=2 | B=4 | B=8 |
|---|---:|---:|---:|---:|
| Talker (in 2048, q 2048, kv 1024) | **0.90×** | 0.98× | 1.04× | 1.24× |
| Code Predictor (in 1024, q 2048, kv 1024) | 1.26× | 1.25× | 1.33× | **1.46×** |

The Code Predictor gains at every width. The Talker **loses 10% at B=1** and only turns positive
once the batch is wide enough to reach the tiles — the wrong side of the trade for a
single-stream profile, and a per-component decision on this host rather than a global one.

## 8. What this page does not claim

**The reported L3 is a socket figure, not your slice.** `lscpu` says 480 MiB shared by all
online CPUs, and `tools/box_info.sh` propagates that into "60 MiB per core → the 1.7B Code
Predictor working set FITS". On a virtualised slice of eight cores out of a full Granite Rapids
socket that is almost certainly false, and the same objection applies to the 256 MiB reported on
the Scaleway Zen 5 host and the 260 MiB on the Emerald Rapids box. Nothing here measures the LLC
actually available to the instance. The cheapest discriminator is an array-size sweep in
`tests/membw.c` — it currently takes the L3 size as an input rather than finding it — and until
that exists the `llc_fits_*` field should not drive a profile decision.

**This is a screening campaign, not a qualification.** Dirty tree, `source_commit=UNKNOWN` in the
result identity, three waves per cell, one bank, one class. No realistic mixed-length bank, no
Poisson open-loop, no soak, no per-class percentiles, no memory-growth check. The sibling page
has all of those; this one does not, and its `2x4` C=2 realtime result deserves a soak before it
is deployed on.

**The frequency is not observable.** No `cpufreq` in sysfs: the hypervisor decides. The box was
idle (load 0.8, `snapd` and `unattended-upgrades` stopped) and the three repeats agree, but no
governor was pinned because there is none to pin.

**Price is not measured.** The comparison against the AMD candidates in
[`x86-box-selection.md`](x86-box-selection.md) still has no cost-per-realtime-stream axis.

## 9. What it costs

On-demand, Linux, shared tenancy, captured **2026-09-03**. Prices move; re-check before
quoting, and note that these are list prices with no Savings Plan, Reserved or Spot discount.

| region | `c8i.4xlarge` |
|---|---:|
| `us-east-1` | **$0.7497 /hr** |
| `eu-west-1` | ~$0.805 /hr — *derived*, see below |
| `eu-central-1` | not captured |

The `eu-west-1` figure was not fetched directly. `c8a` (0.862→0.925) and `c8i-flex`
(0.712→0.764) each carry a 7.3% EU-over-US uplift, so $0.7497 × 1.073 is the working estimate.
**It is an inference, not a quote — and a weak one**: a third instance measured later,
`m8i-flex` (0.8044→0.8966), carries **11.5%**, so the uplift is not a family constant. Read the
row as "somewhere around $0.81–0.84" and fetch the real number before spending on it.

Because §4 established how many realtime streams this box holds, the useful number is derivable
for once:

| | value |
|---|---|
| realtime streams at `2x4` (C=2, stream RTF 0.832) | **2** |
| cost per realtime stream, `us-east-1` | **~$0.375 /hr** |
| cost per realtime stream, `eu-west-1` (derived price) | ~$0.40 /hr |

The comparison that would make this meaningful — the same figure for the Emerald Rapids
reference box, which holds one stream at the same topology — cannot be computed, because that
page does not record which instance type it was rented as. **Any future box page should record
the instance type and the on-demand price at measurement time**, since cost per realtime stream
is the only number that actually ranks candidates and it is unrecoverable afterwards.

Caveats that keep this from being a purchasing recommendation: it is one screening campaign on
one short bank, the price is list rather than committed, and "realtime stream" here means stream
RTF below 1.0 on a synchronized wave, not a soaked production SLO.

## Reproducing this page

```bash
echo off | sudo tee /sys/devices/system/cpu/smt/control
gcc -O2 -pthread -o membw tests/membw.c
bash tools/box_info.sh --membw ./membw --out box.json

make blas && ./qwen_tts --caps && ./qwen_tts --self-test
QWEN_NO_VNNI=1 QWEN_NO_SDOT=1 ./qwen_tts --self-test
./qwen_tts --matmat-bench -j 1 ; QWEN_NO_AMX=1 ./qwen_tts --matmat-bench -j 1
make x86-amx-b32-bench ; make x86-qkv-bench

bash download_model.sh --model base-large --dir qwen3-tts-1.7b-base
make bench-topo BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=x86-8c-amx-recommended \
                BENCH_TOPO=1x8,2x4,4x2 BENCH_CONC=1,2,4,6,8

# the AMX attribution of §6, one fresh server per arm
for e in "" QWEN_NO_AMX_BF16=1 QWEN_NO_AMX_INT8=1 QWEN_NO_AMX=1 QWEN_AMX_MIN_B=4; do
  make bench-topo BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=x86-8c-amx-recommended \
       BENCH_TOPO=2x4 BENCH_CONC=1,4 BENCH_ARGS="${e:+--server-env $e}"
done
```

The missing step, if this instance is rented again, is `make bench-suite BENCH_TOPO=2x4` — the
realistic bank, the open-loop arrivals and the soak — which is what would turn §4 from a
screening table into a qualification.
