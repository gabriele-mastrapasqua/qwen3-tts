---
title: "RTF 0.90 and stalling 35% of the time: designing a CPU streaming TTS server around the metric that actually breaks"
published: false
description: "How the v2 CPU streaming server in our pure-C Qwen3-TTS engine got designed — and why almost every decision came from a metric that is not the realtime factor. Admission, decode quantum, AMX Design D, the feature we shipped and then retired, and the 30-minute soaks that failed where the benchmarks passed."
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
- The **decode quantum** sets the prebuffer almost by itself: q8 → 726 ms p95, q32 → **2544 ms**,
  while RTF moves from 0.901 to 1.008. q32 even had *better* TTFA. We rejected it anyway.
- We built an AMX decoder path with persistent weight tiles, measured it honestly, and found **the
  kernel was never the problem** — AMX can touch at most 10–20% of a request's wall clock.
- We shipped a decoder feature, then **retired it on measurement**: STREAM p95 1.058 → 0.829,
  `safe_play_start` p95 1049 → 415 ms, stall@250 10% → 0% by turning it *off*.
- **Benchmarks passed where the soak failed.** At C16 on a 32-core Zen5, every wave class looked
  fine; the 30-minute closed-loop soak produced 596 rejects and 111 broken pipes.
- Today it holds **C10–C16 on 32-core hosts** with zero stalls at a 250 ms buffer, from 30-minute
  strict-KPI soaks.

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

## 2. Measuring, in Abrash's sense

This project owes a debt to Michael Abrash's *[Graphics Programming Black
Book](https://www.jagregory.com/abrash-black-book/)*, and not only for the cache-line alignment that
once bought us 24%. The debt is methodological, and it is the reason section 1 exists at all.

Abrash's recurring argument is that **you cannot reason your way to where the time goes.** He wrote
the Zen Timer — a small, precise instrument — not because timing was hard but because *intuition was
wrong*, reliably and expensively, even for people who knew the hardware intimately. Half the
chapters in that book are a confident prediction followed by a measurement that demolishes it, and
the demolition is the content. His other point, quieter and more useful, is that **the instrument
determines what you are able to notice.** A profiler that samples at 10 ms cannot see a 2 ms
stall; a metric that averages cannot see a hiccup.

Which is exactly the trap we were in. RTF is not a wrong measurement — it is a **coarse
instrument** pointed at a question it cannot resolve. Averaging a stream's rate destroys precisely
the information a listener cares about. So the first real engineering of this cycle was not a
kernel: it was building a better Zen Timer.

That turned into one shared metric core, `tests/playback_sim.py`, which every harness now uses.
It takes a request's chunk-arrival timeline and simulates a real 1× player against it: when could
playback have started so it never underruns, how large was the worst gap, and what fraction of
requests stall under a 100 / 250 / 500 / 1000 ms jitter buffer. Three deliberate design decisions
in it, all of them Abrash-shaped:

- **Marks are client-observed**, not server-emitted. What the server *believes* it sent is not
  what the listener got.
- **The instrument reports its own fidelity.** A client that is late can get several already-queued
  chunks back in one read; the harness counts those `coalesced reads`. When the share is small the
  cadence numbers are tight upper bounds on server lateness — when it is large the run is a
  diagnostic and must not be quoted. That column has saved us from citing at least two runs.
- **Every number carries its provenance.** Our preflight tool labels each value `[MEASURED]`,
  `[CACHED]`, `[TRANSFERRED]`, `[PREDICTED]` or `[UNKNOWN]`, so a model's guess can never be
  mistaken later for an observation.

The rest of this post is mostly a sequence of predictions the measurements killed. That is not a
confession — by Abrash's standard it is the job working correctly.

## 3. Where first audio actually goes

With the right metric in hand, the profile of the problem changed completely.

The engine's shape is a 28-layer Talker, a 5-layer Code Predictor that re-reads its weights 16×
per 80 ms audio frame, and a convolutional speech decoder. Decode is DRAM-bound: we already knew
that. What we did not know was how little of a *loaded server's* first audio was decode at all.

At the first concurrency that failed the envelope on the streaming reference host — a
12-physical-core GCP `c4-standard-24`, SMT off, 1.7B INT8 — we decomposed the wall clock and found:

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

## 4. The v2 architecture, and which measurement put each piece there

```
              accept()          conn queue        readers          job queue      continuous scheduler (owns ctx)
 client ──────────────────▶  [ cq ] ──────▶ parse HTTP ──────▶  [ jq ] ──────▶ ┌────────────────────────────────┐
                                             read-only on ctx                  │ persistent frame loop:          │
                                                                               │  • admit queued → free slots    │
                                                                               │  • batched ragged Talker + CP   │
                                                                               │  • per-slot sample / EOS        │
                                                                               │  • per-slot stream OR decode    │
                                                                               └────────────────────────────────┘
                                                                                          │
                                                                                          ▼
                                                                               decoder lane (private team,
                                                                               one mailbox per slot, elastic)
```

**Pre-forked, pinned workers.** `--prefork W --prefork-threads K` forks W workers after the weights
are loaded and pins each to a contiguous slice of cores; the weights are shared copy-on-write, so W
workers cost roughly one model in RAM rather than W. The parent does not synthesize — it accepts and
hands the file descriptor over, so a slow request occupies a worker and not the accept loop.

The CPUs are ordered **core-major** before slicing, each physical core's threads adjacent. That is
there because of a measured defect: Linux numbers every core's first thread before any sibling, so
on a 12-core SMT-2 host `--prefork 2` used to hand worker 0 cpus 0–11 and worker 1 cpus 12–23 — *the
same twelve physical cores*, one worker per hyperthread. The workers were not isolated at all, and
since the AMX tile unit is per physical core they serialised on it while the numbers looked like
clean isolation.

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

### Design D, and the lesson that the kernel was never the problem

The speech decoder is a convolutional stack, and on Intel hosts with AMX it looked like the obvious
place to spend effort. So we spent it. **Design D** quantises to INT8 and **prebuilds immutable AMX
`B` weight tiles at model load**, then feeds the quantised im2col panel straight in as AMX `A`,
covering the decoder's real M = 96 / 192 / 384 / 768 shapes and falling back cleanly when a shape is
unsupported. A 1.7B model loads 24 persistent Design-D packs, 19.3 MB, once. There is a BF16 twin
(`TDPBF16PS` with immutable BF16 packs) and a warm-strip path that evaluates only the newly produced
output columns of a streaming causal convolution instead of recomputing the left context.

It works, it is exact, and it is the default on AMX hosts. And here is the honest accounting, which
is the part worth publishing:

> The decoder's MACs already run on real AMX with wide N. **Its wall is glue** — im2col,
> quantization, roughly 41 thread rendezvous and 110 BLAS calls per call, the snake activation,
> the tails. **AMX can touch at most ~10–20% of a request's wall clock**, and splitting the work
> into more tile tasks made it *worse*.

We went looking for a faster matrix multiply and found that the matrix multiply was already fast.
Everything that mattered afterwards — admission, the quantum, the decoder lane, the cohort — was
scheduling and data movement. If there is one Abrash lesson this project relearned at full price,
it is that one.

What *did* pay on the same path was removing a fixed cost rather than speeding one up. The fused
residual epilogue eliminates a whole pass over the decoder's residual path. Here is the full
envelope, both arms, same binary, same bank, arms interleaved — the shape of table we now require
before anything is allowed to become a default:

| arm | C | TTFB p50/p95 | TTFA p50/p95 | STREAM p50/p95 | TOTAL p50/p95 | prebuffer p50/p95 | safe-start p50/p95 | max gap p95 | stall@250 / @500 | req/s | batch B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| fused **off** | 3 | 1 / 57 ms | 159 / 166 ms | 0.782 / 0.832 | 0.836 / 0.886 | 149 / 298 ms | 310 / 457 ms | 522 ms | 11% / 0% | 1.47 | 1.89 |
| fused **on** | 3 | 1 / 63 ms | 153 / 166 ms | 0.667 / 0.820 | 0.698 / 0.875 | 171 / 285 ms | 324 / 441 ms | 510 ms | 11% / 0% | 1.38 | 1.78 |
| fused **off** | 4 | 54 / 60 ms | 171 / 179 ms | 0.778 / 0.831 | 0.835 / 0.905 | 187 / **389** ms | 359 / **568** ms | 521 ms | **17%** / 0% | 1.71 | 2.35 |
| fused **on** | 4 | 53 / 60 ms | 165 / 170 ms | 0.764 / **0.788** | 0.826 / 0.862 | 160 / **266** ms | 328 / **426** ms | 510 ms | **0%** / 0% | 1.77 | 2.33 |

Read the C4 row pair. STREAM p95 improves 0.831 → 0.788, which is the kind of number that gets
quoted. But prebuffer p95 goes 389 → 266 ms, safe-start p95 568 → 426 ms, and **stall@250 goes 17%
→ 0%** — and that last column is the one a listener would have noticed. `B` is the *measured*
effective batch, not the client concurrency, which is why it is in the table: at C3 a worker was
averaging 1.89 slots, so "concurrency 3" was not exercising the batched kernel the way the name
suggests.

### The decode quantum decides the prebuffer

This one surprised us enough to change a default. The quantum is how many frames the decoder
converts per unit of work. Same host, same everything, one five-minute closed-loop soak per arm:

| quantum | STREAM p50/p95 | TTFA p50/p95 | prebuffer p50/p95 | safe-start p50/p95 | stall@100/250/500/1000 | max gap p50/p95 | coalesced reads |
|---|---:|---:|---:|---:|---|---:|---:|
| **q8** | 0.868 / 0.901 | 254 / 549 ms | 391 / **726 ms** | 702 / **1023 ms** | 100% / 79% / 8% / **0%** | 614 / 964 ms | 4.6% |
| **q32** | 0.897 / 1.008 | **199 / 513 ms** | 2129 / **2544 ms** | 2451 / **2844 ms** | 100% / 100% / 96% / **87%** | 2305 / 2469 ms | 33.0% |

```
required prebuffer p95      (lower is better · 1 block ≈ 70 ms)
  q8   ██████████                              726 ms
  q32  ████████████████████████████████████   2544 ms

TTFA p50                    (lower is better · 1 block = 20 ms)
  q8   █████████████                           254 ms
  q32  ██████████                              199 ms   ← q32 WINS this one

stall rate @ 1000 ms jitter buffer   (1 block = 2.5 points)
  q8                                             0%
  q32  ███████████████████████████████████      87%
```

Look at the TTFA column. **q32 is better on first audio** — 199 ms against 254 — and slightly
better at p95 too. If we had been optimising TTFA, which is the metric everyone reports, q32 would
have looked like a win, shipped, and made 87% of streams stall against a one-second buffer. The
realtime factor barely moves (0.901 → 1.008). The only columns that scream are the playback ones.

q32 is a perfectly good *batch* policy and an unacceptable *streaming* one. Its 33% coalesced-read
share is itself the evidence: delivery had become bursty, which is exactly what a large quantum
does. **Rejected as a production streaming quantum**, and written down so nobody re-proposes it.

### A feature we shipped, then retired on measurement

We built a decoder "cohort": pair two streams into one batched decoder call, on the reasonable
theory that sharing the pass amortises the fixed cost. It shipped. Then a C10 diagnostic on a
32-core Arm host localised a playback knee to workers holding three active streams, and we ran the
one-change A/B:

| arm | completions | STREAM p95 | TTFA p95 | safe-start p95 | stall@250 | stall@500 |
|---|---:|---:|---:|---:|---:|---:|
| paired cohort (as shipped) | 120 | 1.058 | 312 ms | **1049 ms** | **10.0%** | **3.6%** |
| singleton decoder | **126** | **0.829** | **159 ms** | **415 ms** | **0%** | **0%** |

Turning the feature **off** moved every column. The lifecycle trace said why:

| split | paired cohort p95 | singleton p95 |
|---|---:|---:|
| B=3 enqueue → lane start | 213.8 ms | **135.1 ms** |
| B=3 decoder compute | 269.5 ms | **83.5 ms** |
| B=2 enqueue → lane start | 55.6 ms | 65.0 ms |
| B=2 decoder compute | 144.9 ms | 86.2 ms |

Pairing did not amortise the pass; at three active streams it *serialised* it. The per-call penalty
came out at 1.63× on Graviton4 and 1.20× on Axion, and disabling it was numerically free —
`mel_corr 1.00000`. The cohort is now retired on Arm, kept on the x86 product profile **on weaker
evidence and labelled as such**, because it was promoted there from a combined smoke test rather
than an isolated arm. Writing "this one rests on worse evidence than that one" in the profile
itself is not self-flagellation; it is the only thing that stops the weak result being cited later
as if it were the strong one.

### One more idea we killed with our own measurement

Utilization-aware admission: hold a new request back when the box is busy, admit it when there is
headroom, thresholds at 40/60/80 ms. An appealing design.

It admitted every tested arrival at all three thresholds, and the streams already running went to
`STREAM_RTF` p95 0.985–1.004, **50%** stalls at a 250 ms buffer, and post-admission `max_gap` p95 of
653–704 ms. The simple cap-plus-fail-fast policy beat it. The feature stayed in the tree as a
default-off diagnostic and the negative result is written down next to it, because the next person
to have that idea should get the measurement instead of the enthusiasm.

## 5. The configuration is part of the product

Here is the measurement that changed how we ship this thing more than any kernel did. It is from
2026-09-01, on the 16-core Arm host we used before the current architecture existed — which is
worth saying, because the *number* has moved since and the *mechanism* has not.

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

This is the Abrash lesson one level up: an unstable measurement is not noise to be averaged away,
it is a **signal about the system**, and the right response is to find the bistability rather than
to run the benchmark again until it looks good.

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
on the build current that day, that whole environment set is worth **nothing at concurrency 1** —
most of it became a compiled default in the meantime — and **18% of first audio at concurrency 4**.
It earns its keep where the machine is loaded. It stays pinned at C=1 anyway, so a change of default
elsewhere cannot move a qualified deployment quietly.

## 6. `make doctor`: under a minute, no model

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
`unqualified` so that promoting it is a deliberate act. It is a Zen Timer with a conscience.

## 7. The soak, and what continuous realistic load found that benchmarks did not

Everything up to here was measured with **waves**: fire C requests simultaneously, wait for all of
them, fire the next wave. Waves are the harshest arrival model and the one comparable to firing N
streams at an accelerator, and we used them for every A/B in this post.

Waves are also not what a server experiences. So the qualification gate is a different test: a
**30-minute closed-loop soak**.

- **Closed loop.** C conversations stay open; each sends its next request only when the previous
  response finishes. Nothing is synchronised, so the workers drift out of phase exactly as they do
  in production — which is the regime where "effective batch" stops equalling "client concurrency".
- **Variable, realistic text.** A seeded, stratified schedule across five classes — `short`,
  `medium`, `long`, `conversational` and `italian` — so a length-heavy class cannot quietly take
  over the end of the run, and so first audio is tested against prompts of 4 to 59 words rather
  than one sentence repeated.
- **Three independent verdicts**, reported separately: pooled latency KPI, **per-class** KPI, and
  **resource stability** — memory, thread count and file descriptors across the whole process tree.
- **Probe WAVs every five minutes**, excluded from the latency statistics, so there is something to
  listen to afterwards. No script scores speech.

The per-class split earns its keep on its own. Pooled numbers can pass while one text class is
quietly broken, and percentiles over four or five observations are effectively the maximum — so
the analyzer carries separate evidence thresholds and will say `PARTIAL` rather than pass a tail it
cannot assess. A gate that reports "I could not tell" is worth more than one that guesses green.

**And then it did the thing that justified building it.** On a 32-core Zen5 at C16, every wave
class looked deployable:

| C16, waves | short | medium | long | mixed | across classes |
|---|---:|---:|---:|---:|---|
| STREAM p95 | 0.963 | 0.934 | 0.911 | 0.944 | all < 1.0 |
| required prebuffer p95 | — | — | — | — | 374–394 ms |
| safe_play_start p95 | — | — | — | — | 636–664 ms |
| stall@250 | — | — | — | — | ≤ 2% |

Sub-1.0 everywhere, safe-start comfortably inside the one-second line, stalls near zero. On wave
evidence alone, C16 ships.

The 30-minute soak at the same point: **STREAM p50/p95 0.948 / 1.004** (short class 1.045), **596
per-worker rejects** and **111 broken-pipe / connection-reset errors**. C16 is the hard capacity
edge, and nothing but sustained closed-loop load showed it.

C12 on the same host, same binary, same profile:

| C12, 30-minute closed-loop soak | value |
|---|---:|
| completed | 2205 |
| errors / rejects / timeouts | **0 / 0 / 0** |
| TTFA p95 | 170 ms |
| STREAM_RTF p50 / p95 | 0.838 / 0.912 |
| TOTAL_RTF p95 | 0.941 |
| required prebuffer p95 | 262 ms |
| safe_play_start p95 | 408 ms |
| stall@250 / @500 | ≤ 1% / **0%** |
| per-class STREAM p95 | long 0.880 · italian 0.888 · medium 0.885 · conversational 0.914 · **short 0.959** |
| per 5-minute window STREAM p95 | 0.898 – 0.917 |

Two things to notice. The per-5-minute windows are **flat**: 0.898 to 0.917 over half an hour, which
is the actual definition of "sustained" and cannot be established by any number of waves. And the
class that is worst is `short` at 0.959 — short texts have the least generation to amortise their
fixed admission cost, so they are the tail. That is invisible in a pooled number and it is why the
per-class row exists.

The profile stayed **provisional** anyway, because the preferred (not mandatory) STREAM p95 ≤ 0.90
holds in every wave class and in the medium/long/italian soak classes but **not** for short and
conversational under closed loop. Honest status beats a green badge.

## 8. What it holds

From 30-minute closed-loop soaks with strict KPI checking, 2026-09-15:

| host | model | C | completed | TTFA p95 | STREAM p95 | safe-start p95 | stall@250 / @500 | err/rej/timeout |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 32-core Graviton4 | 0.6B | **C12** | 3962 | 198 ms | 0.836 | 356 ms | **0% / 0%** | 0 / 0 / 0 |
| 32-core Graviton4 | 1.7B | **C10** | 3665 | 207 ms | 0.831 | 364 ms | **0% / 0%** | 0 / 0 / 0 |
| 32-core Axion | 1.7B | **C16** | 5207 | 175 ms | 0.789 | 332 ms | **0% / 0%** | 0 / 0 / 0 |
| 32-core Axion | 0.6B | **C16** | 5281 | 195 ms | 0.845 | 371 ms | **0% / 0%** | 0 / 0 / 0 |
| 32-core EPYC Zen5 | 1.7B | **C12** (profile prefers C11) | 2205 | 170 ms | 0.912 | 408 ms | ≤1% / **0%** | 0 / 0 / 0 |

```
safe_play_start p95 — the one-second line is the product limit, not a target

Graviton4 0.6B C12   356  ██████████████████                                │
Graviton4 1.7B C10   364  ██████████████████                                │
Axion     1.7B C16   332  █████████████████                                 │
Axion     0.6B C16   371  ███████████████████                               │
Zen5      1.7B C12   408  ████████████████████                              │
                          0           250          500          750        1000 ms
                                                                    HARD LINE ┘
```

Those points assume a client prebuffer of at least 250 ms — at these densities `stall@100` is
22–23%, and saying so is part of the claim rather than a footnote to it.

**Every row is a 32-core host, and that is not a marketing choice — it is the honest extent of the
evidence.** These were the machines we brute-forced the architecture on, because the question was
what it can do when the hardware is not the limit. Smaller hosts were qualified earlier, on the
architecture that preceded this one, with three-wave TTFA sweeps rather than 30-minute soaks; those
numbers are kept in the repository, labelled as history, and do not appear in this table.
Requalifying a 16- and an 8-core host on v2 is work that has not been done.

What the rows do say is that this is a **memory-bandwidth** problem rather than a core-count one.
Three 32-core hosts land at C10, C12 and C16, and the ordering follows their bandwidth. Matrix
units buy first audio; **bandwidth buys concurrency**.

One attribution caveat we insist on carrying, because it is the sort of thing that quietly becomes
folklore: the 1.7B Axion point moved C12 → C16, and it is tempting to credit the cohort retirement
for all of it. The paired screens say the retirement alone is worth about −17.5% safe-start and
+4.6% completions. **Most of the C12 → C16 move is simply that the ladder had never been climbed
that far before.** Which brings us to the last thing.

## 9. The last lesson, which is about ladders

We measured the concurrency ladder of a new box at C2, C4, C8, C12, C16 — a sensible doubling
sweep — and concluded that C8 was the last clean rung. Better still, C8 happened to coincide with
the batched kernel's own measured per-stream optimum at B=8. Two independent roads meeting: the
kind of agreement that feels like understanding.

Then we filled in C9, C10 and C11. All three were clean. The plateau ran to **C11**, throughput
peaked at C10–C11, and the elegant coincidence was an artifact of where the ladder happened to stop.

**"The highest rung measured clean" and "the highest rung that is clean" read almost the same and
are different statements, and a doubling ladder only ever establishes the first.** That one is
cheap to remember and, going by our own record, not cheap to learn.

Abrash would have recognised the failure immediately: we had let an elegant agreement between two
measurements substitute for checking whether one of them was under-sampled. The measurement was
not wrong. The question we asked it was.

---

*Everything here is in the repository: [`docs/serving/`](https://github.com/gabriele-mastrapasqua/qwen3-tts/tree/main/docs/serving)
has the CPU server guide, the operations manual and the measured-box index;
[`configs/perf/`](https://github.com/gabriele-mastrapasqua/qwen3-tts/tree/main/configs/perf) has the
profiles, including the negative results and the "this rests on weaker evidence" notes. The GPU
server is documented separately, and labelled work-in-progress, for exactly the reasons above.*
