# Serving Qwen3-TTS

One HTTP API, two backends, and **two very different maturity levels**. This directory keeps them
apart deliberately: the CPU server is qualified and the GPU one is not, and the fastest way to
publish a wrong number is to let their tables sit in the same document.

```
docs/serving/
├── README.md            you are here — pick a lane
├── api.md               the HTTP API: endpoints, request bodies, streaming, errors  (both backends)
│
├── cpu.md               ▶ the CPU streaming server — PRODUCTION. Start here.
├── cpu-operations.md      the operations manual: break-in, benchmark suite, soak, profiles
├── cpu-batching.md        how continuous request-batching works inside a worker
│
├── gpu-cuda.md          ⚠ the CUDA streaming server — WORK IN PROGRESS, not qualified
│
└── boxes.md             every box measured for serving, its profile JSON, and what it holds
```

## Which one do you want

| | [CPU](cpu.md) | [GPU / CUDA](gpu-cuda.md) |
|---|---|---|
| maturity | **production** — qualified on named hosts | ⚠️ **WIP** — runtime-verified, never qualified |
| evidence behind the numbers | 30-minute strict-KPI closed-loop soaks | 2–3 minute screens on two cards |
| configuration gate | `configs/perf/*.json` — a profile that refuses to be optional | none exists for GPU |
| sized by | cores, cache, memory channels | GPU memory bandwidth |
| what it holds | 0.6B C12–C16 and 1.7B C10–C16 on 32 cores (30-minute soaks); 1.7B C11 on 32-core Zen5 | C11 on an RTX PRO 6000 Blackwell screen |
| deploy it today? | yes | only if you are prepared to qualify it yourself |

If you are unsure, the answer is the CPU server. The GPU one is genuinely fast and genuinely
unproven, and [`gpu-cuda.md`](gpu-cuda.md) opens with the list of what is missing.

### How to read a number in this directory

Every number here carries the host it was measured on, and the host tells you which **era** it
belongs to. The **v2 serving architecture** — decoder lane split, direct dilated residual
convolutions, admission and cohort policy, stream layout — landed between 2026-09-07 and
2026-09-15, and **the only hosts qualified on it are the four 32-core boxes**: a Graviton4, an
Axion, and a Zen5 Turin with its control arm. That was deliberate — brute-force the biggest CPUs
available and find out what the architecture can actually do — and it is also the entire extent of
the current capacity evidence.

Numbers attributed to a **16-core or 8-core host date from 2026-09-01 or earlier** and describe the
previous engine, usually measured with three-wave TTFA sweeps rather than 30-minute closed-loop
soaks. They are kept, and labelled, wherever they teach something that is still true — which
topology wins at which concurrency, what a flag costs on which silicon, why a spinning BLAS
matters. **None of them is a current capacity claim**, and they are listed apart in
[`boxes.md`](boxes.md).

## Before anything else, on any box

```bash
make doctor
```

Under a minute, no model, no download: what this machine is, whether SMT/governor/cgroup gates
pass, its measured bandwidth roof, which kernels the binary actually resolves, and a **predicted**
topology with the full environment and a `why` per line. Every number carries a label —
`[MEASURED]`, `[CACHED]`, `[TRANSFERRED]`, `[PREDICTED]`, `[UNKNOWN]` — and the labels are the
point. See [`cpu.md`](cpu.md) §1.

## What "it holds concurrency N" has to mean

Never RTF alone. A concurrency is acceptable on the whole envelope or not at all, and **the
listening metrics fail first**:

| metric | requirement |
|---|---|
| `stall_rate@B` | ideally 0%, at every buffer (100 / 250 / 500 / 1000 ms), not only the loosest |
| `safe_play_start` | **under 1 s. One second or more is not acceptable** |
| `TTFB` / `TTFA` | low / lowish |
| `max_gap` | small — it is what forces the prebuffer |
| `STREAM_RTF` | < 1: necessary, nowhere near sufficient. It is a mean rate, not a continuity proof |
| rejects / errors | 0 |

Measured case that fixes the ordering in mind: an A6000 at C6 had RTF 0.90 — comfortably
realtime — with a 35% stall rate at 1000 ms and `safe_play_start` p95 of 4.0 s. The metric that
looked best was the one that mattered least.

## The rest of the map

- [`../feature-flags.md`](../feature-flags.md) — every runtime flag, its default per ISA, the
  measurement that chose it, and how to prove the process actually read it
- [`../../configs/perf/README.md`](../../configs/perf/README.md) — the deployment profile format
- [`../cuda-performance.md`](../cuda-performance.md) — the CUDA backend itself: build,
  single-stream RTF, how it scales with card bandwidth
- [`../hardware-testing.md`](../hardware-testing.md) — benchmarking your own box, Apple Metal
- [`../ENGINEERING-METHOD.md`](../ENGINEERING-METHOD.md) — why the measurement rules are shaped
  the way they are
- [A CPU streaming server that never stalls](../../blog/cpu-streaming-server-that-never-stalls.md)
  — the narrative version: the v2 design and the measurements that chose each piece of it
