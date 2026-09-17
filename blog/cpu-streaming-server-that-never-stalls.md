---
title: "RTF 0.90 and stalling 35% of the time: designing a CPU streaming TTS server around the metric that actually breaks"
published: false
description: "How the v2 CPU streaming server in our pure-C Qwen3-TTS engine got designed — and why almost every decision came from a metric that is not the realtime factor. Admission, decode quantum, pinned workers, and the configuration gate that exists because a benchmark was bimodal and nobody could see it."
tags: performance, c, tts, machinelearning
---

*Part of [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) — a pure C inference engine
for Qwen3-TTS, no Python, no PyTorch, and (for this post) no GPU. Follow-up to
[Making Qwen3-TTS fast on every CPU](making-qwen3-tts-fast-on-every-cpu.md).*

## TL;DR

We had a text-to-speech engine that ran comfortably faster than real time on a CPU. Turning it
into a **streaming server** — one that feeds N concurrent players continuously, not one that
finishes N files quickly — took an architecture rewrite, and almost none of the work was where the
realtime factor pointed.

- **RTF is a mean rate and it lies about continuity.** One measured case: RTF 0.90, and the same
  run stalled in 35% of streams with a one-second jitter buffer.
- The thing that actually breaks is **`safe_play_start`** — the earliest moment a player can start
  and still finish without an underrun. Our hard line is **under one second**.
- **Most of first audio, under load, was not inference.** At the first concurrency that failed,
  **over 97%** of the time from client to first PCM elapsed *before the request reached the
  engine*.
- The **decode quantum** sets the prebuffer almost by itself: q8 → ~0.7 s, q32 → ~2.5 s, while RTF
  barely moves. We rejected q32 as a streaming policy on that alone.
- A serving **configuration is part of the product**. A benchmark that forgot one environment
  variable was *bimodal* — 108 ms or 66 ms, and a single run looked equally definitive either way.
  Now a profile file is a gate that refuses to be optional.
- Today it holds **C12–C16 on 32 Arm cores** with zero stalls at a 250 ms buffer, from 30-minute
  soaks. On 8 cores it holds **one**. That range is the honest answer, and both ends are the point.

---

## 1. The metric that does not do what its name suggests

RTF — processing time ÷ audio duration — is the number every TTS project quotes, and it is the
right number for a batch job. Below 1.0 the machine produces audio faster than it is played, so it
keeps up.

On *average*.

A stream can average twice real time and still hiccup for two seconds, and it is the hiccup a
listener hears. So we stopped quoting one number and started quoting a row:

| metric | what it is | what it decides |
|---|---|---|
| `TTFB` | send → status line and headers | perceived responsiveness of the API |
| `TTFA` | send → first audio chunk | perceived responsiveness of the *voice* |
| `max_gap` | the largest inter-chunk gap | the mechanism: it is what forces a prebuffer |
| `required_prebuffer` | smallest delay after first audio at which a 1× player then never pauses | how much you must buffer |
| **`safe_play_start`** | earliest time after the request at which playback can begin *and finish* without an underrun | **the product metric** |
| `stall_rate@B` | share of requests stalling under a B ms jitter buffer (100/250/500/1000) | what fraction of listeners hear a gap |
| `STREAM_RTF` | the old friend | capacity — necessary, nowhere near sufficient |

The ordering matters more than the list: **these fail at different concurrencies, and the listening
ones fail first.** The case that fixed it in our heads was measured on an RTX A6000 at six
concurrent streams: RTF **0.90** — comfortably realtime, a number you would happily put in a README
— while the stall rate at a one-second buffer was already **35%** and `safe_play_start` p95 was
**4.0 s**.

Every single metric there is true. Quoting the flattering one would have been wrong in the only way
that matters.

One sharp edge worth stating, because it cost us a day of confusion: percentiles are taken **over
requests**, and `safe_play_start` is computed from each request's own timeline — never as "TTFA plus
a prebuffer percentile". Composing percentiles is how a "part" once came out larger than the
"whole" in a table nobody could explain.

## 2. Where first audio actually goes

With the right metric in hand, the profile of the problem changed completely.

The engine's shape is a 28-layer Talker, a 5-layer Code Predictor that re-reads its weights 16×
per 80 ms audio frame, and a convolutional speech decoder. Decode is DRAM-bound: we already knew
that. What we did not know was how little of a *loaded server's* first audio was decode at all.

At the first concurrency that failed the envelope on our reference host, we decomposed the wall
clock and found:

- three full-wave requests waited **3.6–4.9 seconds before the parent process even called
  `accept()`**;
- **over 97%** of client-to-first-PCM elapsed before the request was admitted to the engine;
- setting `--max-queue 0` converted exactly the same overload into three immediate `503`s.

So the failure at that point was **admission**, not inference. That distinction is the difference
between "buy a faster CPU" and "fix the queue", and no RTF number could have told them apart.

A second structural cost showed up next to it: **prefill runs inline**, and admitting a new request
stalled every established stream by 108–240 ms while its prompt was processed. A streaming server
has to protect the streams it already has from the ones arriving — that is a scheduling property,
not a kernel one.

## 3. The v2 architecture, and which measurement put each piece there

```
              accept()          conn queue        readers          job queue      continuous scheduler (owns ctx)
 client ──────────────────▶  [ cq ] ──────▶ parse HTTP ──────▶  [ jq ] ──────▶ ┌────────────────────────────────┐
                                             read-only on ctx                  │ persistent frame loop:          │
                                                                               │  • admit queued → free slots    │
                                                                               │  • batched ragged Talker + CP   │
                                                                               │  • per-slot sample / EOS        │
                                                                               │  • per-slot stream OR decode    │
                                                                               └────────────────────────────────┘
```

**Pre-forked, pinned workers.** `--prefork W --prefork-threads K` forks W workers after the weights
are loaded and pins each to a contiguous slice of cores; the weights are shared copy-on-write, so W
workers cost roughly one model in RAM rather than W. The parent does not synthesize — it accepts and
hands the file descriptor over, so a slow request occupies a worker and not the accept loop.

**One scheduler thread per worker owns the model context** and runs a persistent frame loop:
admit queued requests into free slots, step every active slot together through the batched Talker
and Code Predictor, sample per slot with that request's own parameters and RNG state, and on EOS
free the slot **and immediately refill it**. Continuous batching, not static: no waiting for the
slowest member of a group.

**Streaming composes with batching, and that is the whole point.** Because the loop steps one frame
at a time, after every batched step each active request has a new frame — so each streaming client
gets its own PCM chunk immediately while the matrix work stays shared. Weight-stationary throughput
and per-request streaming at the same time.

**Per-request independence is bit-exact.** Each slot carries its own text, voice, sampling
parameters, seed and RNG state, swapped in and out per slot per frame. A request in a batch of 8
reproduces its single-stream output bit-for-bit (validated at mel-correlation 1.0), and cross-talk
measures zero.

**The decoder can leave the critical path.** `QWEN_SD_LANE_SPLIT` gives the speech decoder its own
thread team with one mailbox per slot, and `QWEN_SD_LANE_ELASTIC` narrows the engine pool only
while a decode unit is actually in flight.

### The decode quantum decides the prebuffer

This one surprised us enough to change a default. The quantum is how many frames the decoder
converts per unit of work. Sweeping it:

| decode quantum | prebuffer p95 | RTF |
|---|---|---|
| q8 | ~0.7 s | — |
| q32 | ~2.5 s | barely different |

RTF is almost flat across a 3.5× change in the number a listener actually waits. **q32 is a
perfectly good batch policy and an unacceptable streaming one**, and only the playback metrics can
see the difference.

### One idea we killed with our own measurement

We built utilization-aware admission: hold a new request back when the box is busy, admit it when
there is headroom, thresholds at 40/60/80 ms. It is an appealing design.

It admitted every tested arrival at all three thresholds, and the streams that were already running
went to `STREAM_RTF` p95 0.985–1.004, **50%** stalls at a 250 ms buffer, and post-admission
`max_gap` p95 of 653–704 ms. The simple cap-plus-fail-fast policy beat it. The feature stayed in the
tree as a default-off diagnostic and the negative result is written down next to it, because the
next person to have that idea should get the measurement instead of the enthusiasm.

## 4. The configuration is part of the product

Here is the measurement that changed how we ship this thing more than any kernel did.

Same binary, same text bank, same host, arms interleaved, varying **only** whether the platform's
declared runtime environment was applied. First audio at concurrency 1:

| round | without the environment | with it |
|---|---:|---:|
| 1 | 108 ms | **66 ms** |
| 2 | 66 ms | **66 ms** |
| 3 | 99 ms | **66 ms** |

The bare arm is **bimodal**. It lands on either value, and a single run looks equally definitive
whichever it lands on. The mechanism was visible beside it — 42,500 context switches per second
against 12,000 — because the BLAS library idles by *spinning* and fights the engine's own thread
pool. Nothing in the output said which configuration had produced the number.

A server with ~40 relevant environment variables, several of which do not transfer between
machines, cannot be launched from memory. So the configuration lives in a JSON file and the engine
emits its own invocation from it:

```bash
eval "$(tools/perf_profile.py command <profile> --model MODEL_DIR --port 8080)"
tools/perf_profile.py check-flags <profile> --log server.log
```

and three things now refuse to run without it: the benchmark harness (no `--profile`, no start),
the forbidden-variable check (some values must be **absent from the environment**, not merely unset
by us — the engine backs off sizing BLAS entirely when `OPENBLAS_NUM_THREADS` already exists, so
somebody else's `export` silently replaces a qualified thread split), and the suite that produces
any number leaving the repository.

**A flag is on when the process says so, never when the invocation intended it.** The engine prints
one machine-readable line naming every registered variable it actually read, and `check-flags`
compares it against the profile.

Two values in that set are worth naming because they are the ones people copy:

- **`QWEN_POOL_SPIN`** — how long a pool worker re-reads before parking. On a 16-core Arm host,
  65536; the 4096 default tuned elsewhere cost **40% of the Code Predictor** there (16.0 → 9.6
  ms/frame, 491,320 → 35,132 context switches). On 8 x86 cores, 65536 measures *worse* and 4096 is
  right. Copying it across would have been a 13% regression wearing the costume of a best practice.
- **`QWEN_DECODER_BATCH`** — one decoder pass for all active slots. Pays when a worker genuinely
  holds several slots, and not otherwise. Pinned per profile, never as a portable default.

And a nuance we kept in the docs because it is more useful than the flattering version: re-measured
on a current build, that whole environment set is worth **nothing at concurrency 1** — most of it
became a compiled default in the meantime — and **18% of first audio at concurrency 4**. It earns
its keep where the machine is loaded. It stays pinned at C=1 anyway, so a change of default
elsewhere cannot move a qualified deployment quietly.

## 5. `make doctor`: under a minute, no model

Every one of these lessons was expensive, and none of them helps somebody who just cloned the repo
onto a rented box. So they are compiled into one command that runs **in under a minute and does not
need the model downloaded**:

```bash
make doctor
```

It reports, in eight blocks: what the machine is and whether the SMT / governor / cgroup **gates**
pass (a failed gate means every later number describes a different machine); the measured memory
bandwidth roof per CPU mask, swept by thread count, plus the engine's own GEMV roof; which kernels
this binary actually **resolves** on this host; whether batching pays at each width, with no model
loaded; a predicted frame time per topology; a recommended worker×thread shape with **the full
environment and a `why` per line**; the exact commands to verify it; and a ceiling — how many
streams the host can hold at most, and whether the wall is bandwidth or the decoder.

The design decision inside it that we would repeat anywhere: **every number carries a provenance
label** — `[MEASURED]`, `[CACHED]`, `[TRANSFERRED]`, `[PREDICTED]`, `[UNKNOWN]`. The doctor predicts
a topology; it does not qualify one, and its draft profile is written to disk marked
`unqualified` so that promoting it is a deliberate act.

## 6. What it holds

From 30-minute closed-loop soaks with strict KPI checking, stratified over a five-class text bank:

| host | model | concurrency | evidence |
|---|---|---|---|
| 32-core Graviton4 | 0.6B | **C12** | STREAM p95 0.836, TTFA p95 198 ms, safe-start p95 356 ms, stall@250 and @500 **0%**, zero errors |
| 32-core Graviton4 | 1.7B | **C10** | STREAM p95 0.831, TTFA p95 207 ms, safe-start p95 364 ms, stall@250/@500 **0%** |
| 32-core Axion | both | **C16** | STREAM p95 0.789 / 0.845, zero errors, zero 250/500 ms stalls |
| 32-core EPYC Zen5 | 1.7B | **C11** | per-class STREAM p95 0.816–0.848, safe-start p95 ~467 ms, stall@250 = stall@500 = 0% |
| 16-core Axion | 1.7B | **C4** | at C6 the realtime factor is ~1.1: first audio still good, sustained realtime not |
| 8-core Xeon AMX | 1.7B | **C1** | the only configuration on that host faster than realtime, and only with all eight cores on one request |

Those points assume a client prebuffer of at least 250 ms — at these densities `stall@100` is
22–23%, and saying so is part of the claim rather than a footnote to it.

The 8-core row is there deliberately. This is a memory-bandwidth problem: eight cores of Xeon with
AMX serve one realtime stream, and thirty-two Arm cores serve sixteen. Matrix units buy first
audio; **bandwidth buys concurrency**.

## 7. The last lesson, which is about ladders

We measured the concurrency ladder of a new box at C2, C4, C8, C12, C16 — a sensible doubling
sweep — and concluded that C8 was the last clean rung. Better still, C8 happened to coincide with
the batched kernel's own measured per-stream optimum at B=8. Two independent roads meeting: the
kind of agreement that feels like understanding.

Then we filled in C9, C10 and C11. All three were clean. The plateau ran to **C11**, throughput
peaked at C10–C11, and the elegant coincidence was an artifact of where the ladder happened to stop.

**"The highest rung measured clean" and "the highest rung that is clean" read almost the same and
are different statements, and a doubling ladder only ever establishes the first.** That one is
cheap to remember and, going by our own record, not cheap to learn.

---

*Everything here is in the repository: [`docs/serving/`](https://github.com/gabriele-mastrapasqua/qwen3-tts/tree/main/docs/serving)
has the CPU server guide, the operations manual and the measured-box index;
[`configs/perf/`](https://github.com/gabriele-mastrapasqua/qwen3-tts/tree/main/configs/perf) has the
profiles, including the negative results. The GPU server is documented separately, and labelled
work-in-progress, for exactly the reasons above.*
