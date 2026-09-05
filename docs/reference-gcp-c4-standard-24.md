# GCP `c4-standard-24` - VNNI-only reference, 12 physical Intel cores

> **Audit correction (2026-09-05):** this historical campaign used
> `SIMD=avx512vnni`, which deliberately did **not** compile `-mavx512bf16`. The
> server environment did contain `QWEN_PREFILL_MATMAT=1`, but that request could
> not select AVX-512 BF16 or AMX in this binary. The saved `dispatch.txt` was
> also generated before the server environment was applied, so its
> `f32_blas_fallback=ON` line was not a faithful description of the serving
> process. On the serving path the explicit flag entered the generic BF16
> matmat fallback. The model numbers below therefore remain valid observations
> of that exact dirty binary/environment, but they are **not** measurements of
> a native BF16-prefill VNNI build. The corrected starting profile is
> `configs/perf/gcp-c4-standard-24-vnni-ttfa.json`, built with
> `SIMD=avx512bf16`; the dispatch gate now rejects this contradiction before a
> suite can start.

This is the VNNI-only campaign for the new GCP C4 candidate. The host exposes AMX as well,
but the binary used here was rebuilt with `SIMD=avx512vnni`, so AMX is not compiled into this
report. The AMX build and its model campaign will be recorded separately.

The model measurements are reproducible observations from the current dirty local branch,
not a production qualification: the binary carries the matching dirty source fingerprint,
but the source tree contains uncommitted engine edits that predate this campaign.

The source used on the VM was copied directly with `rsync` from the local dirty branch
`feature/x86-amx-vnni-oss` at `620279c`, including the current uncommitted C changes. No
GitHub fetch, clone, or source cleanup was used. Private data, models, virtual environments,
and local build artifacts were excluded from the transfer.

## Result in one page

The VM is a serious VNNI candidate, but the first stable topology is not the c8a-shaped
2x8 preset:

- GCP `c4-standard-24`, `us-central1-a`, non-preemptible.
- Intel Xeon Platinum 8581C (Emerald Rapids), 1 socket, 12 physical cores and 24 logical CPUs.
- SMT is on by default; sibling pairs are `0-11` and `12-23`.
- `avx512_vnni`, `avx512_bf16`, `amx_tile`, `amx_int8`, and `amx_bf16` are exposed.
- One NUMA node, 88.38 GiB RAM, no swap, cgroup CPU and memory limits are `max`.
- Guest-visible cache is 48 KiB L1d, 32 KiB L1i, 2 MiB L2 per core, and 260 MiB shared L3.
- The measured Triad screen is about **106 GB/s**, with the 95% knee at **12 threads**.
- On the same physical mask, SMT changes Triad by **0.01%**. Adding the 12 sibling CPUs
  raises the full-mask Triad roof by only **4.48%**.
- The current VNNI binary is x86-native, `SIMD=avx512vnni`, and its `--caps` report says
  native VNNI GEMV/matmat plus BF16 widen-and-FMA; AMX is detected by the CPU but not compiled.
- The open `qwen3-tts-1.7b` model was downloaded on the VM with the dedicated model script.
- In the synchronized model sweep, **SMT-off 2x6** is the safest tail candidate. In the
  longer suite, **SMT-on 4x6** is nearly identical at C4 and uses all 24 logical CPUs, while
  SMT-on 3x8 wins median C4 but has a worse p95 tail.
- No realistic C4 cell in this dirty-branch campaign is a production claim below `STREAM_RTF`
  1.0 at p95. The best suite C4 p95 values are 1.018 (4x6 on) and 1.019 (2x6 off).

The important shape is therefore not `24 vCPU = 24 cores`. The physical execution domain is
12 cores. SMT does not create a second memory roof, and the measured host roof must not be
divided by 24 and then reused as a worker roof.

## Hardware identity

| field | measured value |
|---|---|
| VM | GCP `c4-standard-24` |
| zone | `us-central1-a` |
| scheduling | non-preemptible (`FALSE`) |
| CPU | Intel Xeon Platinum 8581C @ 2.30 GHz |
| family/model/stepping | `6 / 207 / 2` |
| sockets | 1 |
| physical cores | **12** |
| logical CPUs in inventory | 24 |
| threads per core | 2 with SMT on, 1 in the SMT-off run |
| SMT default | on |
| sibling map | `0-11` with `12-23` |
| NUMA | one node; CPUs `0-23` when SMT-on |
| RAM | 88.38 GiB total, 87.1 GiB available at collection |
| swap | 0 |
| cgroup | `cpu.max=max 100000`, `memory.max=max` |
| frequency control | no cpufreq/governor exposed; hypervisor controls frequency |
| compiler | GCC 15.2.0 |

The guest-visible ISA flags include AVX-512 VNNI/BF16 and AMX INT8/BF16/tile. The VNNI binary
used for this report passed its native and fallback self-tests, and its resolved dispatch map
is `x86_avx512vnni`. This proves the VNNI path was compiled and selected; it does not qualify
the separate AMX build.

## Cache and memory context

The guest exposes one shared L3 instance of 260 MiB (`266240K`, shared by the CPUs in the
online mask). The simple arithmetic is 21.67 MiB of visible LLC per physical core, but this
is not an independent per-core cache: it is a shared pool, and the effective cache available
to the model still needs a working-set sweep.

`box_info.sh` estimates the int8 Code Predictor working set at approximately 60 MiB for the
0.6B model and 120 MiB for the 1.7B model, so it reports both as fitting in the visible 260 MiB
LLC. Treat that as a hypothesis, not proof. In particular, the current `membw` cap is 512 MiB
per buffer. With a 260 MiB L3, the tool's formal per-buffer `>= 4x L3` test cannot pass, so the
raw JSON correctly records `residency=cache`, despite the three buffers totaling 1.5 GiB and
each buffer being larger than L3. This run is consequently a bandwidth screen, not a formally
DRAM-qualified roof measurement. An L3/working-set sweep belongs in the next phase.

## Measurement protocol

The benchmark was the branch's `tests/membw.c`, compiled on the VM as:

```text
gcc -Wall -Wextra -O3 -march=native -mtune=native -pthread tests/membw.c -o /tmp/qwen_membw_c4_dirty
```

Each invocation used three 512 MiB `double` buffers, five repetitions, and the best result per
thread count. It reports Copy (read/write), Triad (read/write), and Read (read-only), plus the
full saturation curve and the 90/95/99% thread points. The source SHA-256 was identical before
and after transfer:

```text
tests/membw.c       4e9ada389b7135c0cc88e89b3d6a9eec94c9119c9677e0775abd1e6e82bbaece
/tmp/qwen_membw...   3ae7c4da55a0af178a31ad398b11dac0d8d74a904e86018d7e7cc6058be08de2
```

The complete bandwidth JSON is archived under
`profiles/gcp_c4_20260905/{smt_on,smt_off}/`; model, profile, census, and cost-map artifacts
are under `profiles/gcp_c4_20260905/vnni/`. The remote copies remain under `/tmp/` on the VM.
The final remote state was restored and verified as SMT-on.

## SMT A/B and per-mask roofs

The `roofs.py` sweep measured each execution domain separately. These are the canonical peak
values from the per-mask store:

| state and CPU mask | CPUs allowed | physical cores | Copy peak | Triad peak | Read peak | Triad peak / physical core* |
|---|---:|---:|---:|---:|---:|---:|
| SMT-on host `0-23` | 24 | 12 | 93.06 GB/s | **105.99 GB/s** | 130.79 GB/s | 8.83 GB/s/core |
| SMT-on physical `0-11` | 12 | 12 | 91.78 GB/s | **101.45 GB/s** | 82.44 GB/s | 8.45 GB/s/core |
| SMT-off online `0-11` | 12 | 12 | 91.94 GB/s | **101.46 GB/s** | 83.25 GB/s | 8.46 GB/s/core |

\* The last column is a reporting normalization, not a claim that bandwidth is additive or
that a worker roof can be derived by division.

The exact physical-mask A/B is the clean answer to the SMT question:

| metric | SMT-on `0-11` | SMT-off `0-11` | change |
|---|---:|---:|---:|
| Copy peak | 91.78 | 91.94 | +0.17% |
| Triad peak | 101.45 | 101.46 | +0.01% |
| Read peak | 82.44 | 83.25 | +0.98% |

The full-mask comparison answers the second question, namely whether using both hardware
threads helps the shared memory roof:

- Copy: 91.78 -> 93.06 GB/s, +1.39%.
- Triad: 101.45 -> 105.99 GB/s, +4.48%.
- Read-only: 82.44 -> 130.79 GB/s, +58.65%.

Read-only traffic benefits from extra outstanding memory concurrency on this virtualized
host, but that is not the same as a second physical-core bandwidth roof. For the read/write
Triad signal that best represents the sustained weight-streaming pressure, the sibling gain is
small and the curve is already at 95% by 12 threads.

### Triad saturation curve

| threads | SMT-on `0-23` | SMT-on `0-11` | SMT-off `0-11` |
|---:|---:|---:|---:|
| 1 | 14.70 | 14.70 | 14.50 |
| 2 | 28.43 | 28.29 | 28.43 |
| 4 | 52.67 | 52.56 | 52.28 |
| 6 | - | 72.29 | 71.42 |
| 12 | 101.90 | 101.45 | 101.46 |
| 24 | 105.99 | - | - |

The independent `box_info.sh` host sweep was consistent: 106.07 GB/s peak Triad at 24 threads,
101.50 GB/s at 12 threads, and a knee at 12. The small difference from `roofs.py` is expected
because they are separate best-of-five sweeps.

## VNNI build and model provenance

This campaign deliberately used the dirty local checkout as transferred, without GitHub,
clone, pull, or commit. The source was copied with direct `rsync`; the engine C/H edits already
present in the checkout were not changed during measurement.

| item | value |
|---|---|
| local branch | `feature/x86-amx-vnni-oss` |
| source commit/fingerprint | `620279c-dirty:f878db3cef9e` |
| VNNI build | `make -B blas SIMD=avx512vnni -j12` |
| compiler/target | GCC 15.2.0, `x86_64-linux-gnu` |
| Ingot | clean-rebuilt on the VM with the x86 compiler |
| binary | `qwen_tts_vnni`, SHA-256 `8f921692809b0287821abfc9badaa6c7dd09e892580ff7a06312c3ebcf484163` |
| model | open `qwen3-tts-1.7b`, downloaded on the VM by `download_model.sh --model large` |
| model weights SHA-256 | `38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137` |

The matching dirty fingerprint means the binary is not stale relative to the copied tree. It
does not turn the uncommitted branch into a production build; every model result below remains
provisional until the exact source snapshot is intentionally frozen.

### Runtime profile and flags

The c8a VNNI profile was validated on the VM and reused for its environment contract. Its
hardware description is not being claimed for C4 qualification. The resolved server environment
was:

```text
OPENBLAS_THREAD_TIMEOUT=1
QWEN_CP_PREFILL2=1
QWEN_DECODER_BATCH=0
QWEN_POOL_SPIN=4096
QWEN_PREFILL_MATMAT=1
QWEN_PREFIX_CACHE=1
QWEN_STREAM_DECODE_CHUNK=8
QWEN_STREAM_DECODE_CHUNK_BUSY=0
QWEN_VNNI_GEMV_MR=2
```

`OPENBLAS_NUM_THREADS`, `QWEN_NO_VNNI_ACT_QUANT`, `QWEN_NO_VNNI_ROWSUM`,
`QWEN_NO_VNNI_TILE`, `QWEN_PREFILL_QUANT`, `QWEN_VNNI_NCHUNK`, and `QWEN_VNNI_PREPACK`
were explicitly absent from every VNNI serving run. The profile was passed to the serving
harness for the sweep and suite; the exact soak runs replayed the same nine values explicitly
because the profile's complete c8a server command resolves to 2x8, which is oversubscribed on
this 12-core VM.

## VNNI dispatch and isolated kernel checks

The VNNI `--caps` output recorded:

- native `_mm512_dpbusd_epi32` VNNI;
- VNNI tiled INT8 GEMM for the measured batch shapes;
- BF16 widen-to-FMA and no AVX-512-BF16/AMX code compiled into this binary;
- AMX detected in the CPU but explicitly inactive in the binary.

The native self-test and the forced fallback self-test both passed. The resolved dispatch map
was `isa_class=x86_avx512vnni`, with no suspicious or mismatched VNNI dispatch entries.

The isolated `--matmat-bench -j4` check was run once with SMT-off physical CPUs and once with
SMT-on logical CPUs. The BF16 rows are the VNNI build's non-AMX fixed-B twin; the INT8 rows are
the relevant VNNI batching check.

| shape | INT8 seq/batch off | speedup off | INT8 seq/batch on | speedup on |
|---|---:|---:|---:|---:|
| 3072x1024 | 0.15 / 0.07 ms | 2.04x | 0.26 / 0.13 ms | 1.97x |
| 1024x3072 | 0.15 / 0.05 ms | 2.78x | 0.15 / 0.05 ms | 2.93x |
| 2048x1024 | 0.10 / 0.04 ms | 2.74x | 0.10 / 0.04 ms | 2.67x |

The one-thread control was also archived (`matmat-bench-j1.log`); it is not mixed into the
four-thread comparison.

## Model topology sweep

This was the true simultaneous-wave sweep on the short bank, three waves per cell, INT8
1.7B serving, with the profile environment above. `STREAM_RTF` is reported as p50/p95 and
`TOTAL_RTF` as p50/p95. A cell with a nonzero error/reject count is not a candidate.

### SMT-off: 12 physical cores

| topology | C4 STREAM_RTF | C4 TOTAL_RTF | errors/rejects | reading |
|---|---:|---:|---:|---|
| 1x12 | 1.417 / 1.483 | 1.638 / 1.740 | 0 / 0 | best single worker, poor C4 sharing |
| **2x6** | **1.025 / 1.047** | **1.226 / 1.396** | **0 / 0** | best balanced physical-mask result |
| 3x4 | 1.311 / 1.376 | 1.636 / 1.782 | 0 / 0 | worse tail |
| 4x3 | 1.085 / 1.135 | 1.373 / 1.479 | 0 / 0 | below 2x6 only at low pressure |
| 6x2 | 1.465 / 1.493 | 1.850 / 2.031 | 0 / 0 | too little worker budget |

The `1x12,C=12` cell had nine errors and is retained only as a failed observation, not as a
qualification point. All other listed C4 cells completed with zero errors and rejects.

### SMT-on: 24 logical CPUs

| topology | C4 STREAM_RTF | C4 TOTAL_RTF | errors/rejects | reading |
|---|---:|---:|---:|---|
| 1x24 | 1.537 / 1.720 | 1.709 / 1.962 | 0 / 0 | no benefit from one oversized worker |
| 2x12 | 1.109 / 1.295 | 1.437 / 1.448 | 0 / 0 | stable but slower than 4x6 |
| **3x8** | **0.916 / 1.457** | **1.209 / 1.791** | **0 / 0** | best median, tail moves sharply |
| **4x6** | **1.029 / 1.090** | **1.335 / 1.409** | **0 / 0** | best SMT-on tail in the sweep |
| 6x4 | 1.315 / 1.374 | 1.520 / 1.755 | 0 / 0 | too fragmented |
| 2x8 | 1.061 / 1.108 | 1.385 / 1.661 | 0 / 0 | execution mask is 2x12, not 2x8 physical cores |

The `1x24` C12 and C16 cells produced errors under overload and are invalid observations.
SMT-on 3x8 wins only on median C4; 4x6 is the more defensible SMT-on candidate because its
p95 is much closer to 1.0.

## Serving suite on the 1.7B model

The branch's `bench_suite.sh --only realistic,fast` was run serially for the three relevant
topologies. The realistic bank used four synchronized waves at C=1,2,4,6,8; the fast bank used
five waves at C=1,4. Every listed cell had `err=0` and `rej=0`.

### Realistic bank

| topology/state | C4 TTFA p50/p95 | C4 STREAM_RTF p50/p95 | C4 TOTAL_RTF p50/p95 |
|---|---:|---:|---:|
| 2x6, SMT-off | 1686 / 1970 ms | **0.981 / 1.019** | 1.179 / 1.748 |
| 3x8, SMT-on | 1600 / 2778 ms | 0.812 / 1.356 | 0.973 / 1.745 |
| 4x6, SMT-on | **996 / 2835 ms** | **0.973 / 1.018** | **1.076 / 1.228** |

The p50/p95 split is the key result: 3x8 is faster in the middle but has a visibly worse
tail; 4x6 is the best SMT-on balance; 2x6 off remains the clean physical-core baseline.
The different TTFA tails are also why the result is not reduced to one pooled “RTF” number.

### Fast short bank

| topology/state | C4 TTFA p50/p95 | C4 STREAM_RTF p50/p95 | C4 TOTAL_RTF p50/p95 |
|---|---:|---:|---:|
| 2x6, SMT-off | 497 / 731 ms | 0.976 / 1.097 | 1.170 / 1.371 |
| 3x8, SMT-on | 610 / 935 ms | 0.934 / 1.701 | 1.196 / 2.111 |
| 4x6, SMT-on | 581 / 726 ms | 1.064 / 1.134 | 1.297 / 1.440 |

These suite rows are not interchangeable with the topology-sweep rows: they use a different
bank/wave count and are kept separate to avoid hiding corpus or arrival-pattern effects.

## Closed-loop soak

Two exact five-minute soaks were run after a 30-second warm-up, with five 60-second windows,
concurrency 4, batch 8, and the same replayed profile environment. The exact server commands
were 2x6 with SMT-off and 4x6 with SMT-on.

| topology/state | completed | errors | queue/server timeout | resource stability | sustained STREAM_RTF p50/p95 | sustained TTFA p50/p95 |
|---|---:|---:|---:|---|---:|---:|
| 2x6, SMT-off | 103 | 0 | 0 / 0 | PASS | 1.105 / 1.288 | 639 / 1651 ms |
| 4x6, SMT-on | 106 | 0 | 0 / 0 | PASS | 1.044 / 1.129 | 986 / 2745 ms |

Both automatic soak verdicts were `FAIL` only because the closed-loop class mix changed by
0.324/0.388 and the per-class p95 sample requirement was not met. The pooled latency KPI was
therefore `NOT_ASSESSED`; the raw per-window values are the honest comparison. Stream RTF p50
over the five windows was 1.097-1.113 off and 1.025-1.089 on. There were 16 audio probes in
each run, no request errors, no queue rejects, no server timeouts, and resource stability PASS.

An initial soak started directly with the c8a profile command and resolved to 2x8 on the
12-core mask; it produced service-cap timeouts and was interrupted. It is archived under
`soak_profile_smt_off_invalid/` and is explicitly excluded from the table above.

## Executed call map and cost map

The VNNI `profile-cpu` run used 2x6 SMT-off, C=1/4, three waves, and the short bank. Its
preflight passed hardware, dispatch, native/fallback self-test, profile, and model gates.
The census recorded 476 rows across three processes and zero dropped operations.

| component | calls | GMAC | optimized | BLAS | fallback | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| talker | 42,362 | 756.741 | 63.9% | 0.0% | 36.1% | 0 |
| CP | 120,404 | 468.539 | 100.0% | 0.0% | 0.0% | 0 |
| decoder | 10,423 | 1,519.292 | 83.9% | 16.0% | 0.1% | 0 |

The executed map contained 144,243 VNNI INT8 matvec calls and 14,353 VNNI gate calls. CP
was entirely on the VNNI path; decoder used VNNI for its INT8 work and BLAS for its FP32
work. The talker fallback percentage is the expected non-AMX BF16/f32 side of this VNNI-only
build, not an AMX dispatch result.

The level-2 cost map ran on C4 with 16 requests and three process roles. Its region gate passed:
zero nesting mismatches, stack overflow, unbalanced ends, and leaked regions.

| serving role/region | calls | total ms | ms/request | note |
|---|---:|---:|---:|---|
| serve `runtime.request.total` | 16 | 39,799.8 | 2,487.5 | derived wall interval |
| serve `runtime.admission` | 16 | 1,558.1 | 97.4 | derived scheduler/admission interval |
| serve `talker.prefill.total` | 16 | 3,825.3 | 239.1 | inclusive |
| serve `talker.decode.total` | 336 | 7,378.7 | 461.2 | inclusive |
| serve `cp.decode.total` | 336 | 4,059.3 | 253.7 | inclusive |
| serve `decoder.total` | 119 | 10,091.6 | 630.7 | inclusive |
| serve `decoder.conv_stack` | 119 | 9,662.3 | 603.9 | 95.7% of decoder total |
| serve `runtime.pool_dispatch` | 124,888 | 22,282.1 | 1,392.6 | inclusive pool interval |
| serve `runtime.pool_wait_completion` | 124,888 | 1,073.0 | 67.1 | 4.8% of dispatch |
| serve `runtime.pool_submit_wait` | 124,888 | 14.1 | 0.9 | 0.1% of dispatch |

The dominant Talker prefill children were gate/up 1,732.9 ms, down 970.3 ms, and QKV
710.7 ms. The roles run concurrently and the region numbers are inclusive, so they must not
be added across roles as if they were one serial timeline. The cost-map A/B/B/A overhead check
measured worst p50 overhead of **+1.5%**, while the clean arms themselves spread by **9.1%**;
the effect is below the observed VM noise.

The cost-map path-id and kernel-class parity was identical A/B and the cost-map integrity gate
passed. The outer parity script returned FAIL because Talker coverage moved by 1.6 percentage
points and 13 adaptive decoder shape tuples moved in this small pair; that is recorded in
`parity.json` rather than hidden. The interleaved overhead result is the stronger instrumentation
cost signal.

## What the numbers mean for the next comparison

The C4 and AWS C8a screens have similar total Triad peaks (about 106 GB/s here versus 103.6
GB/s in the C8a table), but their saturation shapes are different: C4 needs roughly 12 threads
to reach its roof, while C8a reached its knee at 4 threads. Equal peak bandwidth therefore does
not imply equal single-request or multi-request capacity.

The C4 remains interesting because it has 12 physical cores, a large shared LLC, and AMX in
addition to VNNI. The VNNI-only run shows that the extra SMT siblings do not materially raise
the Triad roof, while 4x6 SMT-on can nevertheless keep the C4 stream tail near the 2x6
SMT-off baseline. Whether AMX shortens the compute-heavy Talker/CP stages enough to change
that ranking is intentionally deferred to the separate AMX report.

For future server runs, the safe accounting rules are:

1. Use 12 as the physical-core denominator. `105.99 / 12 = 8.83 GB/s per physical core` is a
   descriptive normalization; `105.99 / 24 = 4.42 GB/s per vCPU` is not a physical-core roof.
2. Measure the exact worker mask. If a server worker is pinned to a subset, compare it only
   with a roof measured under that same mask and residency. Never divide the host roof to get
   a worker roof.
3. Keep SMT-off for the clean qualification baseline, or record explicitly that a run is the
   full `0-23` SMT-on topology. The A/B says SMT itself is not the bandwidth lever here.
4. Before interpreting model timings, run the binary's `--caps`, native/fallback self-tests,
   and the VNNI dispatch gate. This page proves the VNNI build only; it contains no AMX model
   timing or production claim.

## Reproduction

On the copied checkout:

```bash
cd ~/qwen-a1/src
taskset -c 0-23 env MEMBW_REPS=5 bash tools/box_info.sh \
  --membw /tmp/qwen_membw_c4_dirty --out /tmp/gcp_c4_box_info.json
python3 tools/roofs.py measure --membw /tmp/qwen_membw_c4_dirty \
  --store /tmp/gcp_c4_roofs_smt_on --hardware /tmp/gcp_c4_box_info.json \
  --masks 0-23,0-11 --reps 5 --l3-mb 260 --force
sudo sh -c 'echo off > /sys/devices/system/cpu/smt/control'
taskset -c 0-11 env MEMBW_REPS=5 bash tools/box_info.sh \
  --membw /tmp/qwen_membw_c4_dirty --out /tmp/gcp_c4_box_info_smt_off.json
python3 tools/roofs.py measure --membw /tmp/qwen_membw_c4_dirty \
  --store /tmp/gcp_c4_roofs_smt_off --hardware /tmp/gcp_c4_box_info_smt_off.json \
  --masks 0-11 --reps 5 --l3-mb 260 --force
sudo sh -c 'echo on > /sys/devices/system/cpu/smt/control'
```

The model-side sequence used the already transferred dirty checkout, then built the x86 Ingot
library and VNNI binary locally on the VM:

```bash
cd ~/qwen-a1/src
bash download_model.sh --model large --dir qwen3-tts-1.7b
make -C third_party/ingot clean
make clean
make -B blas SIMD=avx512vnni -j12
cp -f qwen_tts qwen_tts_vnni
./qwen_tts_vnni --caps
./qwen_tts_vnni --self-test
```

The serving sweep used `tests/serve_parallel_wave.py` through
`tests/bench_suite.sh`, always with the forbidden VNNI environment variables unset. The
executed-path map used `BIN=./qwen_tts_vnni` with `tools/profile_cpu.sh`; the coarse wall-time
map used the same binary through `tools/costmap_parity.sh`, followed by the interleaved
`tools/costmap_ab.sh` overhead check. All raw outputs from those commands are retained under
`profiles/gcp_c4_20260905/vnni/`.

The final remote state after this screen was verified as `smt=on`, `active=1`, `online=0-23`,
`nproc=24`.
