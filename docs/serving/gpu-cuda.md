# The CUDA streaming server (NVIDIA) — ⚠️ work in progress

_[Serving index](README.md) · [api](api.md) · [CPU server](cpu.md) · **CUDA server** · [boxes](boxes.md)_

> **Read this first.** This page describes the **GPU** streaming server. It is
> **implemented, runtime-verified and listened to under load — and it is NOT
> performance-qualified.** Every soak behind the numbers below is a two- or three-minute
> screen on one box. No 30-minute qualification has been run on any GPU.
>
> If you need a server you can put in front of users today, use the **CPU** one:
> [`cpu.md`](cpu.md). That is the path with a break-in procedure, deployment profiles,
> qualified hosts and 30-minute strict-KPI soaks behind it.

The HTTP API, the request bodies and the streaming contract are identical on both backends:
[`api.md`](api.md). Everything else on this page is CUDA-only.

**Nothing here transfers to the CPU server, and no number from one may be quoted for the
other.** The two are sized by different things — the CPU server by cores, cache and memory
channels, this one by GPU memory bandwidth — and they have different flags, different optimal
batch widths and different failure modes.

For the CUDA backend itself — build, single-stream RTF, how RTF scales with card bandwidth,
the A100 measurements and the kernel work — see
[`../cuda-performance.md`](../cuda-performance.md).

---

## Quick start

```bash
make cuda                                     # multi-arch: sm_80/86/89/120 + PTX

QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_DECODER=1 QWEN_CUDA_CONVDEC=1 \
QWEN_CUDA_BATCH=1 QWEN_CUDA_BATCH_COMPACT=1 \
  ./qwen_tts -d qwen3-tts-0.6b --backend cuda --serve 8000 \
  --batch-size 8 --prefork 1 --prefork-threads 8
```

Then verify the build before believing any number it produces:

```bash
./qwen_tts -d qwen3-tts-0.6b --backend cuda --gpu-selftest
./qwen_tts -d qwen3-tts-0.6b --backend cuda --gpu-batch-bench 8
```

**Do not add `--int8` to the server command**, and **do not pass `--prefork-threads 1`.** Each
of those silently removes most of the GPU server; both are explained in
[Two configuration traps](#two-configuration-traps-that-silently-cost-most-of-the-server).

---

## Maturity — what is actually established

**Maturity: implemented and runtime-verified, NOT performance-qualified. Opt-in, not a default.**

What that means precisely, so nobody has to infer it:

| claim | status |
| --- | --- |
| compiles and runs (`--backend cuda --serve`) | **runtime verified** — A100 2026-09-15, A6000 2026-09-16 |
| batched steps produce the same codes as single-stream | **verified** — `--gpu-batch-bench` reads `0.00e+00` on three checks |
| audio is correct under concurrent load | **ear-validated** at C4 |
| meets a serving KPI target | **NOT established.** Every soak run to date is a 3-minute screen and every one reports `SOAK RESULT: FAIL — per-class KPI drift`. No 30-minute qualification has been run on any GPU. |
| is a default | **no.** It requires `--backend cuda` plus the resident-path env flags below. |

Treat every concurrency table below as a measured screen on one box, not as a supported envelope.

**What is still missing, concretely**, so the gap is a list rather than a feeling:

- a 30-minute strict-KPI closed-loop soak on any GPU — the CPU lane's bar;
- a deployment profile. `configs/perf/*.json` describes a CPU deployment: topology, thread
  split, pinning, forbidden environment. There is no GPU equivalent, so a GPU run has no
  gate that can refuse to start with the wrong configuration — which is exactly why the two
  traps below are able to cost most of the server without erroring;
- more than one card per generation. Two boxes are a shape, not an envelope;
- an audio quality gate under sustained load, rather than the ear checks done so far.

Until those exist, the honest statement is: **the CUDA server works, it is fast, and it is not
qualified.** Both of those halves are load-bearing.

## Concurrency the streaming server holds — RTX A6000, 0.6B

RTX A6000 (768 GB/s) behind a 10-core EPYC 7402, 0.6B, `--precision default`, all resident paths
on, 3-minute closed-loop soaks:

| | C2 | C4 | C6 | C8 |
| --- | --- | --- | --- | --- |
| RTF p50 | 0.43 | **0.56** | 0.90 | 1.11 |
| stall rate @1000 ms | 0% | **4%** | 35% | 63% |
| safe_play_start p50/p95 | 0.69 / 1.6 s | 1.02 / 1.95 s | 1.65 / 4.0 s | 2.76 / 7.9 s |
| requests / 3 min | 57 | 91 | 93 | 101 |

**C4 comfortable, C6 usable.** Throughput keeps climbing to C8, so past C6 the limit is playback
quality rather than capacity. The knee scales with GPU memory bandwidth, not with core count: the
code predictor reads ~2.24 GB of weights per audio frame regardless of how many requests share
them, so the concurrency a card holds tracks its bandwidth almost linearly.

### RTX PRO 6000 Blackwell — 2026-09-16 (WIP screens, not a qualification)

**Status: WORK IN PROGRESS.** Eight two-minute screens on one box. Enough to locate the knee and
to show the shape of the degradation; not a qualification, and the per-class drift gate reports
FAIL on every rung for the reason at the end of this section.

RTX PRO 6000 Blackwell Server Edition (97 GB), 30 cores, CUDA 12.8, 0.6B, `--precision default`.
Steady-state window (65-110 s); p50 unless marked.

| | C2 | C4 | C8 | C9 | C10 | **C11** | C12 | C16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTF p50 | 0.19 | 0.25 | 0.36 | 0.41 | 0.41 | **0.47** | 0.69 | 0.86 |
| RTF p95 | 0.23 | 0.30 | 0.40 | 0.48 | 0.46 | **0.55** | 0.88 | 0.98 |
| TTFB p50 | 14 | 15 | 20 | 19 | 21 | **20** | 22 | 41 ms |
| TTFB p95 | 20 | 38 | 72 | 78 | 91 | **93** | 100 | 152 ms |
| TTFA p50 | 48 | 62 | 90 | 92 | 98 | **93** | 119 | 150 ms |
| TTFA p95 | 110 | 133 | 189 | 199 | 180 | **179** | 228 | 339 ms |
| safe_play_start p50 | 49 | 59 | 94 | 102 | 107 | **104** | 183 | 408 ms |
| safe_play_start p95 | 110 | 125 | 189 | 251 | 226 | **237** | 564 | 749 ms |
| max_gap p95 | 0.21 | 0.29 | 0.38 | 0.44 | 0.45 | **0.46** | 0.61 | 0.75 s |
| stall @100 ms | 0% | 0% | 0% | 2% | 2% | **2%** | 22% | 94% |
| stall @250 ms | 0% | 0% | 0% | 0% | 0% | **0%** | 12% | 41% |
| stall @500 ms | 0% | 0% | 0% | 0% | 0% | **0%** | 1% | 3% |
| stall @1000 ms | 0% | 0% | 0% | 0% | 0% | **0%** | 0% | 0% |
| requests / 2 min | 119 | 158 | 220 | 213 | 236 | **238** | 178 | 192 |
| audio-s / wall-s | 10.5 | 16.0 | 22.2 | 21.4 | **24.1** | 23.0 | 17.4 | 18.8 |

**C11 is the highest clean rung.** Everything from C8 to C11 sits on a plateau: stalls are 0% from
250 ms upward (2% at the tightest 100 ms threshold from C9), safe_play_start stays near
105 / 240 ms — an order of magnitude under the one-second line — TTFB is ~20 ms and TTFA ~95 ms,
and throughput keeps climbing to 238 requests.

**The break is between C11 and C12, and it is a cliff, not a slope.** One rung costs 12% of
stalls at 250 ms where there were none, doubles safe_play_start p95 from 237 to 564 ms, and drops
throughput from 238 requests to 178. Concurrency past that point buys queueing rather than work:
audio-seconds per wall-second peak at **24.1 around C10** and fall to 17.4 at C12.

The judgement is on the whole envelope, never on RTF. C12 would pass an RTF test at 0.69 and a
safe_play_start test at 183/564 ms while stalling 22% of the time at 100 ms. The listening
metrics fail first — the A6000 showed the same ordering at C6, RTF 0.90 with a 35% stall rate at
1000 ms.

A note on how this table was arrived at, because it matters for reading any ladder. The first
pass ran C2/C4/C8/C12/C16 and concluded that C8 was the last clean rung and that throughput
peaked there — which happened to coincide with the batched kernel's own per-stream optimum at
B=8, an elegant agreement. Filling in C9, C10 and C11 dissolved both claims: the plateau extends
to C11 and throughput peaks at C10-C11. "The highest rung measured clean" and "the highest rung
that is clean" are different statements, and a doubling ladder only ever establishes the first.

### The exact configuration behind that table

```bash
QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_DECODER=1 QWEN_CUDA_CONVDEC=1 \
QWEN_CUDA_BATCH=1 QWEN_CUDA_BATCH_COMPACT=1 \
  ./qwen_tts -d qwen3-tts-0.6b --backend cuda --serve PORT \
  --batch-size C --prefork 1 --prefork-threads 8          # batch-size capped at 16 (QB_MAX)
```

Closed-loop clients from `tests/soak_client.py`, two minutes per rung, `--speaker ryan
--language English --temperature 0.9 --schedule stratified --schedule-seed 42`, text bank
`tests/load_texts_en.txt`: 21 items over five classes from 4 words (short) to 59 (long), so the
load is length-varied rather than one sentence repeated. Analysed with `tests/soak_drift.py` at
`--warmup-s 20 --window-s 45`. `audio-s / wall-s` is C divided by RTF p50, the unit vLLM-Omni
reports, so the row is comparable with published figures for other engines.

`SOAK RESULT: FAIL — per-class KPI drift` on every rung, including those with zero stalls. That
gate measures drift BETWEEN text classes across 45-second windows, and with mixed lengths and a
few dozen samples per class it does not settle on a screen this short. It is not a serving defect
and not evidence of one, and it is not something to weaken in order to see green.

### Two configuration traps that silently cost most of the server

1. **`--precision default` is mandatory.** The backend seam is bf16-only, so `--int8` sends the
   *seam* back to the CPU. `tests/serve_soak.py` defaults `--precision` to int8, which produces a
   healthy-looking run with the GPU near 0%.
2. **Size the engine pool.** `--prefork-threads` defaults to `cpus/n` and that default is correct;
   passing `1` explicitly — as `tests/serve_soak.py` does — sizes the whole server pool to one
   thread. Measured at C4: RTF p50 0.68 → 0.58, stall@1000 11% → 4%, 82 → 91 requests, with the
   GPU-side cost unchanged. Four threads captures it, eight adds nothing.

Note that the server is deterministic for a **fixed** thread count and not across thread counts:
the reduction order follows the pool size. Hold it fixed for any A/B.

### What the batched path does on the GPU

`QWEN_CUDA_BATCH=1` steps every in-flight request together, so each weight row is read once for
all lanes. Since 2026-09-16 the batched bodies also:

- **replay from CUDA graphs**, cached per effective lane count. Only the single-stream bodies had
  graphs before; the server runs the batched ones, which were issuing ~1950 launches per frame.
- **compact idle lanes** (`QWEN_CUDA_BATCH_COMPACT=1`) so a server sized for 8 lanes serving 2
  does not pay for the empty ones. Worth 17% of code-predictor time at low occupancy. The KV stays
  addressed by slot, so a compacted lane still reads its own request's history.
- **run the code-predictor head on the GPU** — final norm, lm_head and argmax. The CPU fallback
  re-reads a 4 MB lm_head once *per lane*, fifteen times per frame; the GPU reads each weight row
  once for all lanes. 11.6 → 1.35 ms/frame, and codes are bit-identical.
- **size the per-lane accumulators to the actual lane count**, which is worth 34% at 4 lanes.

Measured effect of the 2026-09-16 work, `--gpu-batch-bench 8`:

| | before | after |
| --- | --- | --- |
| Talker step | 7.56 ms/frame | **4.75** (−37%) |
| Code predictor | 14.35 ms/frame | **10.44** (−27%) |
| aggregate GAIN | 4.73× | **6.82×** |

Every one of those is bit-identical: a fixed-seed temperature-0 streaming request returns the same
md5 before and after, and the three batched self-tests stay at `0.00e+00`.

### Batch size

`--batch-size 8` is the measured optimum and `QB_MAX` is 16. Wider batches are allowed but do not
pay: per-stream code-predictor cost is 2.05 ms at B=4, **1.34 at B=8**, 1.43 at B=12 and 1.47 at
B=16. Past eight lanes the kernel stops being weight-bound — the weights are read once regardless
— and becomes bound by per-lane registers.

### int8 on the GPU

int8 weights work on the resident path and are selected by `--int8` (the seam caveat above applies
only to the seam, not to the resident kernels). They help the **single-request** path, where the
kernel is weight-bound: Talker 5.23 → 4.64 ms/frame, code predictor 7.70 → 6.00. On the batched
path the gain is small — about 5% of RTF at C4 — because halving the bytes does not halve the time
of a kernel that is latency-bound rather than bandwidth-bound.

### The exact configuration behind the numbers above

```bash
QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_DECODER=1 QWEN_CUDA_CONVDEC=1 \
QWEN_CUDA_BATCH=1 QWEN_CUDA_BATCH_COMPACT=1 \
  ./qwen_tts -d qwen3-tts-0.6b --backend cuda --serve 8000 \
  --batch-size 8 --prefork 1 --prefork-threads 8
```

All five resident-path flags are on, `--precision` is left at its default (never `--int8` on the
seam), and the engine pool is sized explicitly. Omitting any of them does not reproduce the table.

### Verifying a GPU build

```bash
./qwen_tts -d qwen3-tts-0.6b --backend cuda --gpu-selftest        # seam vs CPU
./qwen_tts -d qwen3-tts-0.6b --backend cuda --gpu-batch-bench 8   # batched exactness + throughput
```

`--gpu-batch-bench` prints three exactness checks, all of which must read `0.00e+00`: batched
Talker against single-stream, batched code predictor against single-stream, and **partial
occupancy against independent per-lane references** — the last one is the only check that
exercises lane compaction, since the first two step every lane. With `--int4` the first two read
non-zero by design: the single-stream q4 path uses the dp4a kernel, which quantises activations to
int8, so the two sides are deliberately different arithmetic (`QWEN_CUDA_DP4A=0` makes them agree).


---

## The flags that matter for serving

The full CUDA flag reference — including the single-stream ones and the dp4a int4 path — is in
[`../cuda-performance.md`](../cuda-performance.md#how-to-run). These are the ones that decide
whether you have a GPU server or a CPU server with a GPU attached:

| flag | default | what it decides |
|---|---|---|
| `--backend cuda` | — | selects the backend at all |
| `--precision default` | default | **must stay default.** The backend seam is bf16-only; `--int8` sends the seam back to the host CPU |
| `--prefork-threads N` | `cpus/n` | the engine pool. The default is right; passing `1` sizes the whole server to one thread |
| `--batch-size N` | 1 | lanes stepped together. 8 is the measured optimum, 16 (`QB_MAX`) the ceiling |
| `QWEN_CUDA_FUSED_TALKER=1` | off | GPU-resident fused Talker + Code Predictor |
| `QWEN_CUDA_DECODER=1` | off | speech-decoder pointwise convolutions through cuBLAS on the device |
| `QWEN_CUDA_CONVDEC=1` | off | GPU-resident ConvNet speech decoder — the whole conv stack |
| `QWEN_CUDA_BATCH=1` | off | batched fused steps; without it each request is stepped alone |
| `QWEN_CUDA_BATCH_COMPACT=1` | off | idle lanes cost nothing; worth 17% of code-predictor time at low occupancy |
| `QWEN_CUDA_BATCH_GRAPH` | **on** | CUDA-graph replay of the batched bodies, cached per lane count |
| `QWEN_CUDA_CP_HEAD` | **on** | the code-predictor head (norm + lm_head + argmax) on the GPU |
| `QWEN_CUDA_CUBLAS`, `QWEN_CUDA_TC` | off | **both change what the model generates.** Not for serving |

The first five have to be passed together. Omit one and nothing errors — you measure a slower
machine, which is the failure mode this page exists to prevent, exactly as on the CPU side.

## See also

- [`cpu.md`](cpu.md) — the **qualified** CPU streaming server: this is what to deploy today
- [`api.md`](api.md) — the HTTP API, identical on both backends
- [`../cuda-performance.md`](../cuda-performance.md) — the CUDA backend: build, single-stream
  RTF, bandwidth scaling, the A100 measurements
- [`boxes.md`](boxes.md) — every box measured for serving, CPU and GPU, and what each one is
  worth as evidence
- [`../../PLAN.md`](../../PLAN.md) — the open CUDA work items (CUDA-13, CUDA-17 native GPU
  prefill) and what they would buy
