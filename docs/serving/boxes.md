# The boxes this server has been measured on

_[Serving index](README.md) · [api](api.md) · [CPU server](cpu.md) · [CUDA server](gpu-cuda.md) · **boxes**_

A serving configuration is discovered on the box, never chosen from a datasheet. This page is the
index of what has actually been measured: which machine, which profile JSON holds its
configuration, what status that profile carries, and what the host was observed to hold.

**Two things have to be read together: the status, and the era.**

*Status* is the strength of the claim. `qualified` means a configuration was measured end to end on
that hardware and the numbers that earned it are in the file. `provisional` means a campaign
measured it but a gate is still open. `unqualified` means the file is a starting point, a
host-scoped candidate or an ISA contract — several `unqualified` files carry excellent measured
evidence in `preferred_concurrency_evidence` without that making them qualifications.

*Era* is which engine the number describes. The **v2 serving architecture** — the decoder lane
split, the direct dilated residual convolutions, the admission and cohort policy, the stream layout
— landed between 2026-09-07 and 2026-09-15. **Only the 32-core hosts below were qualified on it.**
Everything measured before that describes a different engine, usually with a different arrival
model too (three-wave TTFA sweeps rather than 30-minute closed-loop soaks), and its capacity
numbers are **not** current and **not** comparable with the v2 ones. Those hosts are kept here,
labelled, because the topology and flag lessons they taught still hold — not because the
concurrency they reached still does.

Start a new box with [`cpu.md`](cpu.md) §1 (`make doctor`), not by copying a row from here.

---

## Qualified on v2 — the current capacity evidence

All four are 32-core hosts, and all four points come from **30-minute closed-loop soaks with
strict KPI checking** (the Turin point from multi-class waves), 2026-09-09 to 2026-09-15.

| profile | silicon | cores | topology | status | what it holds |
|---|---|---|---|---|---|
| [`aws-c8g-8xlarge-32c-arm-v2-all-on`](../../configs/perf/aws-c8g-8xlarge-32c-arm-v2-all-on.json) | AWS Graviton4 / Neoverse-V2, `c8g.8xlarge` | 32 | `4x8` cap 8 | unqualified (host-scoped) | **0.6B C12** and **1.7B C10**: zero errors/rejects/timeouts, STREAM p95 0.836 / 0.831, TTFA p95 198 / 207 ms, safe-play-start p95 356 / 364 ms, stall@250 and @500 both **0%** |
| [`axion-c4a-highcpu32-0p6b-all-on`](../../configs/perf/axion-c4a-highcpu32-0p6b-all-on.json) | GCP Axion / Neoverse-V2, `c4a-highcpu-32` | 32 | `4x8` cap 8 | unqualified (host-scoped) | **C16 on both models**: 1.7B STREAM p95 0.789, 0.6B 0.845, zero errors, zero 250/500 ms stalls. C20 is an edge case, not a claim |
| [`turin-c8a-32c-vnni-product`](../../configs/perf/turin-c8a-32c-vnni-product.json) | AMD EPYC 9R45 Zen5, AWS `c8a.8xlarge` | 32 | `4x8` cap 4 (one worker per CCX) | **provisional** | preferred **C11**. C12 waves at temperature 0: per-class STREAM p95 0.816–0.848, TTFA p95 269–276 ms, TTFB p95 102–105 ms, safe_play_start p95 466–468 ms, stall@250 = stall@500 = **0%** |
| [`turin-c8a-32c-vnni-control`](../../configs/perf/turin-c8a-32c-vnni-control.json) | same host | 32 | `4x8` cap 4 | provisional | not a capacity claim: the committed **A/B control arm** (`QWEN_SD_RES1_V2=0`, `QWEN_SD_MULTISLOT=0`). An A/B arm must be a preflighted profile, not a shell override |

All four points assume a **client prebuffer of at least 250 ms**: at these densities `stall@100` is
22–23%, and that belongs in the claim rather than in a footnote to it.

Read the ordering rather than the individual rows. These hosts have comparable core counts and land
at different concurrencies, and the ordering follows memory bandwidth. **Matrix units buy first
audio; bandwidth buys concurrency.**

## ISA contracts — not host results

These carry no topology and no capacity claim. They answer "best supported implementation per ISA"
(`*-product`) and "same serving architecture, different hardware" (`common-control`). **Never pool
the two kinds into one result.**

| profile | lane |
|---|---|
| [`arm-product`](../../configs/perf/arm-product.json) | Neoverse-V2 / KleidiAI: per-item INT8 DOTPROD decoder, KAI for Talker/CP/Q4 |
| [`amx-product`](../../configs/perf/amx-product.json) | Intel AMX: ragged Design-D INT8, fused residual |
| [`vnni-product`](../../configs/perf/vnni-product.json) · [`vnni-bf16-product`](../../configs/perf/vnni-bf16-product.json) | AVX-512 VNNI, and the same plus native bf16 prefill for Zen5 / SPR+ |
| [`common-control`](../../configs/perf/common-control.json) | the deliberately un-accelerated common lane, for cross-ISA comparison |
| [`generic-16c-starting-point`](../../configs/perf/generic-16c-starting-point.json) | a place to **start** on a 16-core Arm server. Nothing in it was measured on your machine, and it says so in its own notes |

---

## Measured before v2 — history, not current capacity

⚠️ **Do not quote the concurrency column as what this engine does today.** These were qualified on
the pre-v2 engine, most of them with three-wave TTFA sweeps rather than closed-loop soaks. What
survives is the *method* and the *flag* lessons — which `W x K` won and why, what `QWEN_POOL_SPIN`
cost on each silicon, what the prefill matmat was worth. Requalifying a 16-core and an 8-core host
on v2 is work that has not been done.

| profile | silicon | cores | topology | status | what it was observed to hold, **then** |
|---|---|---|---|---|---|
| [`axion-16c-ttfa`](../../configs/perf/axion-16c-ttfa.json) | Google Axion / Neoverse-V2 | 16 | `2x8` cap 8 | qualified (pre-v2) | C4 on 1.7B int8; at C6 the realtime factor measured ~1.1. The host behind the profile-environment A/B and the `QWEN_POOL_SPIN=65536` measurement |
| [`aws-c8a-16c-vnni-ttfa`](../../configs/perf/aws-c8a-16c-vnni-ttfa.json) | AMD EPYC 9R45, AWS `c8a.4xlarge` | 16 | `2x8` cap 8 | provisional (pre-v2) | C4. The host where the AVX-512-BF16 prefill default was measured: pinning `QWEN_PREFILL_MATMAT=1` moved TTFA p50 from 416 to 124 ms at C=1 |
| [`scaleway-16c-vnni-ttfa`](../../configs/perf/scaleway-16c-vnni-ttfa.json) | AMD EPYC 9555P Zen5 | 16 | `2x8` cap 8 | provisional (pre-v2) | C4. TTFA p50/p95 66/73 ms at C1 rising to 310/374 at C8; stream RTF 0.47 → 1.55 across the same range |
| [`x86-8c-amx-single-stream-ttfa`](../../configs/perf/x86-8c-amx-single-stream-ttfa.json) | Intel Xeon Platinum 8581C | 8 | `1x8` cap 16 | qualified (pre-v2, commit `266f706`) | C1 — the only configuration on that host then faster than realtime: TTFA 93 ms, stream RTF 0.65 |
| [`x86-8c-amx-multiclient-ttfa`](../../configs/perf/x86-8c-amx-multiclient-ttfa.json) (alias `x86-8c-amx-recommended`) | same | 8 | `2x4` cap 8 | qualified (pre-v2) | C2 balanced; best measured throughput from C4 up; C4 TTFA p95 252 ms at stream RTF 1.43 |
| [`x86-8c-amx-tail-latency-ttfa`](../../configs/perf/x86-8c-amx-tail-latency-ttfa.json) | same | 8 | `4x2` cap 4 | qualified (pre-v2) | the C8 **tail** only — TTFA p95 444.7 ms against 483.6 on the two-worker profile. Everywhere else it loses, badly at low concurrency |
| [`gcp-c4-standard-24-vnni-ttfa`](../../configs/perf/gcp-c4-standard-24-vnni-ttfa.json) | Intel Xeon 8581C Emerald Rapids | 12 phys / 24 log | `2x6` cap 8 | unqualified | the streaming reference host of the plan's C4 work: C2/C3 good, C4 marginal, C5 the first point not streamable. Requires `SIMD=avx512bf16` |
| [`gcp-milan-8c-avx2-v2-cross-screen`](../../configs/perf/gcp-milan-8c-avx2-v2-cross-screen.json) | AMD EPYC 7B13 Milan | 8 | `1x8` cap 1 | unqualified (diagnostic) | a 2-minute C1 A/B, not a capacity result: what the architecture does with no VNNI and no decoder INT8 |

Reference write-ups, same caveat: [`reference-arm-16c.md`](../reference-arm-16c.md) ·
[`reference-x86-8c-amx.md`](../reference-x86-8c-amx.md) ·
[`reference-aws-c8a-16c-vnni.md`](../reference-aws-c8a-16c-vnni.md) ·
[`reference-aws-c8i-8c-amx.md`](../reference-aws-c8i-8c-amx.md) (Granite Rapids, +28% bandwidth) ·
[`reference-aws-c8i-flex-8c.md`](../reference-aws-c8i-flex-8c.md) ·
[`reference-aws-m8i-flex-8c.md`](../reference-aws-m8i-flex-8c.md) ·
[`reference-aws-r8a-16c.md`](../reference-aws-r8a-16c.md) ·
[`reference-gcp-c3d-8c-vnni.md`](../reference-gcp-c3d-8c-vnni.md) ·
[`reference-gcp-c4-standard-24.md`](../reference-gcp-c4-standard-24.md) ·
[`reference-scaleway-16c-vnni.md`](../reference-scaleway-16c-vnni.md) ·
[`x86-box-selection.md`](../x86-box-selection.md) — how these were chosen and what they cost.

Before serving or soaking on a 32-core Arm box, run the
[Arm topology preflight](../arm-topology-preflight.md).

---

## GPU — ⚠️ no profiles, no qualifications

**There is no GPU equivalent of `configs/perf/*.json`.** The profile format describes a CPU
deployment — topology, thread split, pinning, forbidden environment — and a GPU run therefore has
no gate that can refuse to start with the wrong configuration. That is not a detail: it is why
[two settings](gpu-cuda.md#two-configuration-traps-that-silently-cost-most-of-the-server) can
remove most of the GPU server without anything erroring.

Everything below is a **screen**, two or three minutes long, on one card. None of it is a
qualification, and none of it may be quoted beside a CPU number.

| GPU | host | model | what was screened |
|---|---|---|---|
| RTX PRO 6000 Blackwell Server Edition, 97 GB | 30 cores, CUDA 12.8 | 0.6B | eight 2-minute rungs, C2 → C16. **C11 is the highest clean rung**; C8–C11 is a plateau; C12 is a cliff. Peak 24.1 audio-s/wall-s near C10 |
| RTX A6000, 768 GB/s | 10-core EPYC 7402 | 0.6B | four 3-minute rungs. C4 comfortable, C6 usable at RTF 0.90 but 35% stall@1000 ms — the ordering the whole envelope exists to catch |
| A100-SXM4-40GB (cloud) | — | 1.7B / 0.6B | single-stream RTF 0.39–0.50, and the measurement that found the **launch-latency floor**: 5–6× the bandwidth of the reference card did not buy 5× the RTF |
| RTX 4060-class, ~270 GB/s | — | 1.7B | the single-stream reference point: RTF 0.44 `--quant-mixed`; batch 3.35× at B=8 |

Full tables, the exact commands and the traps: [`gpu-cuda.md`](gpu-cuda.md) and
[`../cuda-performance.md`](../cuda-performance.md). Apple Metal numbers are in
[`../hardware-testing.md`](../hardware-testing.md).

---

## Adding a box

```bash
tools/perf_profile.py new <id> [--like aws-c8g-8xlarge-32c-arm-v2-all-on]   # skeleton, unqualified
$EDITOR configs/perf/<id>.json
tools/perf_profile.py validate
```

`new` starts from an existing profile's **structure** and blanks everything that was measured.
Copying a profile by hand is how a value from another machine becomes a claim about this one:
every field is filled in, every field looks deliberate, and nothing says which of them anybody
measured.

Then run the break-in sweep in [`cpu-operations.md`](cpu-operations.md) §2, put what it measured
into the file, and set `qualification.status` to `qualified` with the numbers that earned it.
Naming is `<platform>-<cores>c-<objective>`, because the same machine has a different answer for
first-audio latency than for throughput.

**A value in a profile without a `why` that cites a measurement on *that* hardware is a value
inherited from somewhere else** — and inheriting is how the Arm spin count nearly ended up on an
8-core Xeon. Format and rules: [`../../configs/perf/README.md`](../../configs/perf/README.md).
