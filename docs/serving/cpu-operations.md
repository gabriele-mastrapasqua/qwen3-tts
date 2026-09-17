# Running the CPU server in production, and finding the configuration first

_[Serving index](README.md) · [api](api.md) · [CPU server](cpu.md) · **operations** · [boxes](boxes.md)_


[`api.md`](api.md) is the API: endpoints, request bodies, streaming. **This document is
the other half** — how to run the process, how to find the configuration for a particular
machine before quoting any number from it, and how to measure it so the number means
something a week later.

The short version: **a serving configuration is discovered on the box, not chosen from a
datasheet.** The engine ships a break-in procedure and a benchmark suite for exactly that, and
a profile format so the answer survives the session that found it.

> **Scope: this document is about the CPU backend.** Its topology work, its `W x K` search, its
> pool and pinning discussion and every number in it are sized by cores, cache and memory
> channels. The CUDA server is sized by GPU memory bandwidth instead, has its own flags and its
> own traps, and is not performance-qualified — see
> [`gpu-cuda.md`](gpu-cuda.md). Nothing here transfers to
> it, and a number from one must never be quoted for the other.

---

## 1. Three ways to run it, and the one you probably want

| mode | command | when |
|---|---|---|
| single process, one shared pool | `./qwen_tts -d MODEL --serve 8080 -j 16 --batch-size 16` | development, a single stream, non-Linux |
| **pre-forked pinned workers** | `./qwen_tts -d MODEL --int8 --serve 8080 --batch-size 8 --prefork 2 --prefork-threads 8` | **production on Linux** |
| pre-fork with elastic cores | add `--prefork-elastic` | measured *worse* on the reference host; opt-in, and it says so |

`--prefork` forks *W* workers after the weights are loaded and pins each to a contiguous slice
of *K* cores. The weights are shared copy-on-write, so *W* workers cost roughly one model in
resident memory rather than *W*. It needs `sched_setaffinity` and `SCM_RIGHTS`, so it is Linux
only; elsewhere the engine prints `prefork: not supported on this platform` and runs a single
server rather than pretending.

**The parent does not synthesize.** It accepts connections and hands the file descriptor to a
worker, so a slow request occupies one worker and not the accept loop.

### What `--prefork` and `--batch-size` actually control

Three mechanics decide how many requests run at once, and each one has surprised somebody:

**`--batch-size` is also the per-worker in-flight cap.** The parent hands a worker at most
`--batch-size` connections at a time, and the default is **1**. With `--prefork 12` and no
`--batch-size`, twelve requests run and the rest wait — they are not rejected, they sit in the
listen backlog until a worker frees a slot. The startup line says which it is:

```
prefork: 12 workers x 2 threads, 24 cpus (2 per worker), cap 1 in flight each, port 8080
```

`--batch-size` also selects the scheduler inside each worker: at `1` a worker serves one
request at a time, from `2` up it runs the continuous-batching scheduler. So `W x cap` is the
number of requests in flight, and the backlog (16) is what queues behind it.

**Core slices come from `--prefork`, not from `--prefork-threads`.** The parent orders the CPUs
it is allowed to use **core-major** — each physical core's threads adjacent — and hands worker *w*
the contiguous range `[w * (ncpu / W), …]` of *that* order. `--prefork-threads` then sizes the pool
*inside* the slice. More threads than the slice is oversubscription, fewer leaves cores idle, and
when *W* does not divide `ncpu` the remainder is unused.

Core-major ordering is there because of a measured defect: Linux numbers every core's first thread
before any sibling, so on a 12-core SMT-2 host (siblings `N` and `N+12`) `--prefork 2` used to give
worker 0 cpus 0–11 and worker 1 cpus 12–23 — **the same twelve physical cores, one worker per
hyperthread.** The workers were not isolated at all, and since the AMX tile unit is per physical
core they serialised on it while the numbers looked like isolation. Each worker now prints the mask
it actually set, and says which ordering produced it:

```
prefork topology: 24 logical cpus = 12 physical cores x 2 SMT · 12 workers x 2 threads · 2 cpus per worker (core-major masks)
prefork: worker 0 pid 4711 cpus 0,12 threads 2 (core-major slice, siblings kept together)
```

Read that second line carefully on an SMT host: with `per = 2` and SMT 2, a worker's slice is **one
physical core**, not two. The engine warns about it at startup. It falls back to plain logical order
when sysfs topology is unavailable and says so, in which case `lscpu -e` is what tells you which id
is a sibling of which core and which socket each belongs to.

**On more than one socket, the weights are loaded before the fork** and shared copy-on-write,
so workers pinned to the second socket read them across the interconnect. That is measurable,
and it is worth one `numactl` arm before accepting a topology on a two-socket box.

`kill -USR1 <parent pid>` prints a per-worker line — assigned, completed, in flight, mean batch
— which is the cheapest way to see whether the work actually spread:

```
[prefork-stats] mean_inflight 3.940 dispatched 24 rejected 0 · w0[asg=6 done=6 act=2 B=1.97] ...
```

A single server (`--prefork 1`, or any non-Linux platform) has no parent holding those
per-worker counters. It handles the signal anyway and prints whatever counters the build
carries (the pool counters need a `QWEN_POOL_STATS` build, the kernel census needs
`QWEN_BATCH_STATS`) — the point being that `SIGUSR1` is safe to send to any server: a process
without a handler for it is *terminated* by default, which is how a stats signal once ended a
benchmark arm. `--prefork 1` also sizes the single server's pool from
`--prefork-threads`, so `1xK` is a real arm of a topology sweep rather than a default-threaded
one wearing its name.

---

## 2. Break-in: find `W x K` before you quote anything

A 16-core box can be `1x16`, `2x8`, `4x4`, `8x2`. These are not equivalent, and which one wins
depends on the model, the precision and the concurrency you actually expect. On the reference
16-core Arm host `2x8` won at the target concurrency and `4x4` won at much higher concurrency —
so "four workers is better" and "two workers is better" are both true, of different questions.

Three facts decide the shape, and all three are measurable:

**A worker only sees a batch when it has more than one request.** With *W* workers, client
concurrency *C* spreads across them: a worker reaches batch ≥ 2 only when **`C ≥ 2W`**. On
`2x8` a client concurrency of 2 gives one request per worker and stays on the matrix-vector
path entirely. This is why `client_concurrency` and `engine_batch` are different fields and why
a result should print both — the second one **measured**, from the engine's own counters, never
deduced from *C*.

**The kernel changes with the batch width.** Below batch 2 the engine runs a GEMV kernel; above
it, a GEMM. More workers means narrower batches per worker, which can keep every worker on the
slower path at exactly the concurrency you care about.

**Threads do not add up.** Each worker's pool is `K` threads, and the BLAS inside it has its own
pool. Oversubscribing turns latency into context switches — see §4, where that costs 40 ms.

### Turn SMT off before any of this

The engine pins workers to contiguous **logical** CPU ids, so with SMT on a two-thread slice is
one physical core wearing two hats, and a topology sweep compares configurations that are not
what their names say. `make bench-fingerprint` gates on it and prints `GATE SMT off … FAIL`
with the count when it is on.

Cloud instances differ: an Arm host such as Google Axion (c4a) reports `Thread(s) per core: 1`
and there is nothing to do, while the x86 families (c4, c3, and their AMD equivalents) ship SMT
enabled. On GCP it is an instance property — recreate the VM with `--threads-per-core=1` — and
on hardware you own it is a BIOS setting or `echo off > /sys/devices/system/cpu/smt/control`.
If you cannot turn it off, halve the thread budget deliberately (`-j` = physical cores) and say
so beside the numbers; what you must not do is let a `2x8` slice quietly mean four cores.

### The procedure

```bash
# 0. the preflight, and the artifact directory every later number refers to
make doctor                       # first command on a new box; <1 min, no model: identity,
                                  # bandwidth, and on 32-core Arm Linux a simultaneous 1x8/2x8/4x8
                                  # roof_matvec_int8 scaling preflight; then the engine's own GEMV
                                  # roof, resolved dispatch, shape probe -> a PREDICTED W x K /
                                  # cap / quantum / env set and a draft profile
make cpu-check                    # provenance + hardware + self-test + RESOLVED dispatch map vs
                                  # what this ISA class should select; fails on a silent fallback

# 1. what does this machine actually have?
make bench-fingerprint            # cpu, cores, SMT, cache, NUMA, measured memory bandwidth
./qwen_tts --caps                 # which kernels the binary would pick, per batch width
./qwen_tts --self-test            # cross-ISA correctness oracle

# 2. sweep the topologies at the concurrency you expect
make bench-topo BENCH_MODEL=<dir> BENCH_TOPO=1x16,2x8,4x4 BENCH_CONC=1,4

# 3. write the winner into a profile, and stop retyping it
$EDITOR configs/perf/<your-host>.json
```

Steps 1 and 2 are cheap. Step 3 is what makes the result last: see §3.

`make doctor` is the topology screen, not a serving qualification. Its Arm
multi-worker GEMV block is visible near the top and archives
`arm_gemv_scaling.json`; a poor 4×8 verdict means test `2x16`, then `1x32`,
before paying for a wave or soak. The verdict concerns the tested 4×8 shape,
not the whole instance. Details and interpretation are in
[Arm topology preflight](../arm-topology-preflight.md).

> ⚠️ The 16-core Axion and 8-core AMX numbers in this document were measured **before the v2
> serving work** (decoder lane split, direct dilated residual convs, admission and cohort
> policy; 2026-09-07 to 09-15) and mostly with three-wave TTFA sweeps rather than closed-loop
> soaks. They are kept because the *procedure* and the *shape of the trade* are what this
> document is teaching, and those still hold. **Do not quote their concurrencies as current
> capacity** — the v2-qualified points are the 32-core hosts in [`boxes.md`](boxes.md).

`bench-topo` starts one server per topology, fires true simultaneous waves at each concurrency
and prints one row per cell. Measured on the 16-core Axion reference host, 1.7B open weights at
int8, profile `axion-16c-ttfa`, short bank, three waves:

| topology | C | TTFA p50 | TTFA p95 | RTF p50 | measured batch | errors |
|---|---:|---:|---:|---:|---:|---:|
| `1x16` | 1 | **46 ms** | 54 ms | 0.35 | — | 0 |
| `1x16` | 4 | 141 ms | 142 ms | 0.91 | — | 0 |
| `2x8` | 1 | 54 ms | 72 ms | 0.43 | 0.64 | 0 |
| `2x8` | 4 | **124 ms** | 138 ms | 0.72 | 2.73 | 0 |
| `4x4` | 1 | 87 ms | 110 ms | 0.72 | 0.73 | 0 |
| `4x4` | 4 | 136 ms | 160 ms | 0.80 | 2.77 | 0 |

One worker with every core wins first audio at C=1 and then gives it back at C=4, where its
realtime factor climbs to 0.91; two workers is the shape that holds both. That trade is the
whole point of running the sweep instead of picking a shape. The measured-batch column is empty
for `1x16` because that number comes from the pre-fork parent's counters, and a single server
has no parent to keep them.

---

## 3. The deployment profile, and why it is a gate

`configs/perf/*.json` describes one deployment: hardware, build, topology, per-component
precision, and the runtime environment. Every value is meant to come from a qualification run
rather than from memory, and anything not measured says `"unspecified"` instead of guessing.

```bash
tools/perf_profile.py validate                     # schema + semantics, every profile
tools/perf_profile.py show      <name>
tools/perf_profile.py command   <name> --model DIR --port 8080   # the exact argv and env
tools/perf_profile.py server-env <name>                          # comma form for a harness
tools/perf_profile.py forbidden-env <name>                       # what must NOT be set
tools/perf_profile.py check-flags <name> --log server.log        # what the process actually read
```

That last one closes the loop: the engine prints one `[FLAGS]` line naming every registered
variable it actually read, and `check-flags` compares it against what the profile asked for. **A
flag is on when the process says so, never when the invocation intended it.**

### Why it refuses to be optional

The benchmark harness will not start without `--profile NAME` or an explicit
`--no-profile '<reason>'`. That is deliberate, and it comes from a measurement rather than from
taste. Same binary, same bank, same host, arms interleaved, varying only whether the platform's
declared runtime environment was applied:

| round | without the profile | with it |
|---|---:|---:|
| 1 | 108 ms | **66 ms** |
| 2 | 66 ms | **66 ms** |
| 3 | 99 ms | **66 ms** |

First audio at concurrency 1. The bare arm is **bimodal**: it lands on either value, and a
single run looks equally definitive whichever it lands on. The mechanism is visible beside it —
42 500 context switches per second against 12 000, and 7.9 cores busy against 7.2 — because the
BLAS library idles by spinning and contends with the engine's own pool. Nothing in the output
said which configuration had been measured.

Its scope was measured too, not assumed: at concurrency 4 on the same host the difference was
173 against 169 ms, i.e. nothing — when every worker is busy there is no idle time for a
spinning thread to waste.

**That scope has since moved, and the re-measurement says so.** Same host, current build,
`2x8`, four waves of the short bank, the two arms differing only in whether the profile
environment was applied:

| arm | C | TTFA p50 | TTFA p95 | stream RTF p50 | context switches/s |
|---|---:|---:|---:|---:|---:|
| profile `axion-16c-ttfa` | 1 | 57 ms | 69 ms | 0.44 | 11,910 |
| compiled defaults | 1 | 51 ms | 67 ms | 0.45 | 7,880 |
| profile `axion-16c-ttfa` | 4 | **109 ms** | 160 ms | **0.72** | 30,153 |
| compiled defaults | 4 | 133 ms | 157 ms | 0.81 | 68,984 |

At concurrency 1 the two arms are now within noise of each other, because most of that set
became the **compiled default** in the meantime: the native prefill matmat, the prompt-prefix
cache, the 65536-generation pool spin and the batched decoder are all on without anyone asking.
At concurrency 4 the profiled arm is ahead by 18% on first audio and 11% on realtime factor,
and the mechanism is in the last column — 69,000 context switches per second against 30,000 is
a spinning BLAS competing with busy workers.

So the current claim is the opposite of the earlier one, and it is the reason both are written
here rather than only the flattering one: **a profile earns its keep where the machine is
loaded, and the defaults have absorbed most of what it used to buy at C=1.** A single 4-wave
arm at C=1 is also exactly the shape of run that the bimodality above can flatter, which is why
nothing in this paragraph rests on those two rows.

### Variables that must be ABSENT, not merely unset

A profile can declare a variable `null`, meaning it must not be present in the environment at
all. The case that matters is the BLAS thread count: the engine binds it to the thread budget at
startup and backs off entirely when the environment already sets it, so an unrelated `export` in
someone's shell silently replaces the topology the profile qualified — and no table would show
it. `forbidden-env` lists them and the suite refuses to run when one is present.

---

## 4. Measuring: one command, and what each rung answers

```bash
make bench-suite                                   # open weights, preset voice, neutral bank
make bench-suite BENCH_MODEL=<dir> BENCH_PROFILE=<name> BENCH_TOPO=2x8 \
                 BENCH_SPEAKER=<voice> BENCH_BANK=<file> BENCH_OUT=<dir>
make bench-suite BENCH_RUNG=fast                   # one rung only, the engineering inner loop
make bench-suite BENCH_ARGS="--corpus <file>"      # adds the two duration-diverse rungs
```

With no variables it runs a generic configuration end to end, which is also the fastest way to
check that a new box works at all. Everything deployment-specific arrives on the command line
and lands in that run's manifest rather than in a tracked file.

A complete inner-loop run on the reference host, from a shell in the repository:

```bash
make blas GIT_REV=$(git rev-parse --short HEAD)
make bench-suite BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=axion-16c-ttfa \
                 BENCH_RUNG=fast BENCH_TOPO=2x8 BENCH_OUT=/tmp/bench_fast
```

which prints its preflight (binary sha256, model, profile, resolved server env, forbidden
variables absent, no stale engines, loadavg, source commit), then the rung, then the audio
length per cell, then the manifest. `SUITE PASSED` is the only line that means it ran:

```
topo    C  TTFA50  TTFA95  TTFAmax  RTF50  RTF95  ttc50  ttc95  req/s     B   ...  rej  err
2x8     1      51      71       71   0.43   0.44    0.8    0.8   1.31  0.73   ...    0    0
2x8     4     101     140      140   0.73   0.75    1.4    1.4   2.81  3.00   ...    0    0
SUITE PASSED — artifacts in /tmp/bench_fast, manifest in /tmp/bench_fast/manifest.txt
```

**The idle gate is not advisory.** The suite refuses to start at `loadavg >= 2.0`, which
includes the decay from the run you just finished — wait for the box to settle rather than
chaining two suites back to back.

The same rung on the same host, both models at int8, profile `axion-16c-ttfa`, topology `2x8`,
five waves of a short bank — the shape to expect when a box is set up correctly:

| model | C | TTFA p50 | TTFA p95 | stream RTF p50 | audio p50 | errors |
|---|---:|---:|---:|---:|---:|---:|
| 1.7B | 1 | 51 ms | 71 ms | 0.43 | 1.84 s | 0 |
| 1.7B | 4 | 101 ms | 140 ms | 0.73 | 1.84 s | 0 |
| 0.6B | 1 | 40 ms | 44 ms | 0.33 | 2.48 s | 0 |
| 0.6B | 4 | 69 ms | 89 ms | 0.62 | 2.48 s | 0 |

The audio-length column is why the two models are not compared row against row: they drew
different amounts of speech from the same bank, so only the within-model columns carry a
comparison.

The suite owns the whole invocation, and every preflight check exits non-zero:

- the binary exists, runs, and reports its own sha256 and build tag
- the profile resolves, and its forbidden variables are absent from the environment
- no stale engine processes, and the box is idle
- the source commit and dirty flag travel with the numbers
- each rung runs, then its identity gate
- **each cell's audio length is printed**, so comparability can be read instead of assumed
- a manifest repeats the exact commands

| rung | selected with | what it answers |
|---|---|---|
| `realistic` | `BENCH_RUNG=realistic` | the quotable curve: first audio and sustained realtime against concurrency |
| `fast` | `BENCH_RUNG=fast` | the engineering inner loop — short texts, fast iteration, never a production figure |
| `short-diverse` / `long-diverse` | `BENCH_ARGS="--corpus <file>"` | how input length moves first audio, which it does a lot |

**The default bank is bilingual, and the fast rung does not use all of it.**
`tests/load_texts_en.txt` carries five classes — `short`, `medium`, `long`,
`conversational` and `italian` — so the `realistic` rung exercises English *and* Italian, while
`fast` filters to `short` and is therefore English-only and short-prompt-only. That is
deliberate (an inner loop wants one variable), but it means a `fast` number says nothing about
what a different language or a longer prompt does to first audio. `BENCH_BANK=<file>` swaps the
bank; the same `<class>\t<text>` format is all it needs.

### Arrival models are three different questions

Never call any of them simply "concurrency N".

| harness | arrival | what it measures |
|---|---|---|
| `tests/serve_parallel_wave.py` | *C* requests at t=0, wait for all, next wave | **parallel capacity** |
| `tests/load_test.py --arrival poisson` | independent arrivals at a rate | **load** |
| `tests/load_test.py --arrival all-at-once` | semaphore, *C* in flight | **closed-loop saturation** |

A threshold from one does not transfer to another. The wave is the hardest and the one
comparable to firing N streams at an accelerator.

### Closed-loop soak: stability over time

The soak is a separate, deliberately longer test. It keeps `C` streaming conversations open
in a closed loop: each conversation sends its next request only after the previous response
finishes. The default schedule is seeded and stratified across the text-bank classes, so a
length-heavy class cannot silently take over the end of the run. Every five minutes it also
saves a fixed probe response for listening; probe requests are excluded from latency KPIs.

Run it after the qualification suite, with the same open model and deployment profile:

```bash
make bench-soak SOAK_MODEL=qwen3-tts-1.7b-base \
                 SOAK_PROFILE=<measured-profile> \
                 SOAK_CONCURRENCY=2 SOAK_MINUTES=30 \
                 SOAK_OUT=/tmp/qwen_tts_soak

# Run both the qualification suite and the soak as two identifiable artifacts.
# The Make target passes the BENCH_* model/profile/bank defaults to the soak and
# starts it only after the qualification suite completes.
make bench-suite-full BENCH_MODEL=qwen3-tts-1.7b-base \
                      BENCH_PROFILE=<measured-profile> \
                      SOAK_MINUTES=30
```

The runner refuses an unknown model name: public examples are limited to
`qwen3-tts-0.6b`, `qwen3-tts-0.6b-base`, `qwen3-tts-1.7b` and `qwen3-tts-1.7b-base`.
Use `--no-profile "reason"` only for an explicitly exploratory run. The output directory
contains `manifest.json`, `server.log`, `requests.csv`, `resources.csv`, `soak_summary.json`
and a small set of probe WAVs.

The analyzer reports three separate decisions:

| result | meaning |
|---|---|
| `LATENCY KPI: PASS` | comparable rolling windows stayed within the declared TTFA/RTF limits |
| `LATENCY KPI: NOT_ASSESSED` | the run completed, but there were too few windows or the completed-text mix changed too much; use the per-class rows, not a pooled drift claim |
| `RESOURCE STABILITY: PASS` / `FAIL` | the server process tree did or did not show memory, thread or descriptor growth |

`SOAK RESULT: PARTIAL` is intentional: it means the run is useful for stability, but latency
drift was not scientifically assessable. The default command does not fail for that case;
add `--strict-kpi` through `SOAK_ARGS` when a pipeline must reject an unassessed latency KPI.
Intentional fail-fast `503` responses are recorded separately as admission outcomes and do
not fail the run; queue timeouts and server-side request timeouts still do. The analyzer
fails only on request errors, resource growth, those timeout counters, or a measured KPI
regression.

Per-class p50 and p95 have separate evidence thresholds (`--min-per-class-p50`, default 5;
`--min-per-class-p95`, default 20) because with four or five observations the percentile is
effectively the maximum and one scheduling or text outlier can look like a regression. A short soak may therefore show
`PER-CLASS KPI: <class>=PARTIAL` while pooled latency and resource checks pass. That is not a
model failure. To assess per-class tails, use larger comparison windows and a longer run, for
example `--window-s 300 --min-per-class 5 --min-per-class-p95 15` for a 15–30 minute soak.
`--strict-kpi` makes an unassessed per-class result fail the command; normal mode keeps it
diagnostic while still failing on an assessed regression.
The full table remains in `soak_summary.json`, while the console output is short enough to
scan during a long run.

When the binary was transferred without its `.git` directory, pass the revision that produced
it explicitly so the manifest remains attributable:

```bash
QWEN_SOURCE_COMMIT=<revision-or-build-id> make bench-soak \
  SOAK_MODEL=qwen3-tts-1.7b-base SOAK_PROFILE=<measured-profile>
```

---

## 5. The three numbers, and the many that are not

| | what it is | why it is the one |
|---|---|---|
| **TTFB** | send → status line + headers parsed by the client | the number every TTS server benchmark quotes; printed by every harness (wave, poisson, soak) next to TTFA and stamped independently of it. The batched path now sends the `200` header at admission, before synthesis; the non-batched path already did so. The harness reports `header_to_audio_ms` for the remaining header-to-audio interval |
| **TTFA** | send → first audio chunk | what a caller hears as responsiveness |
| **STREAM_RTF** | `(t_done − t_first_chunk) / (audio after the first chunk)`, **per request** | the steady-state capacity metric: below 1.0 the server produces audio faster than it is played, on average over the stream. **Superseded reading (2026-09-07): it does NOT prove that a player starting at the first chunk never stalls** — it is a mean rate and hides delivery in large quanta; use the playback metrics below for continuity |
| **required_prebuffer** | per request, `max(0, max_i[(t_i − t_first) − audio held before chunk i])` | the smallest delay after first audio at which a 1x player that then never pauses finishes without underrun |
| **safe_play_start** | per request, `max_i (t_i − audio held before chunk i)`, then p50/p95 over requests | the earliest time after the request at which playback can begin and finish without underrun; computed from each request's own timeline, never as TTFA plus a prebuffer percentile |
| **stall_rate@B** | share of requests with at least one stall under a B ms audio jitter buffer (100/250/500/1000), with re-buffering after an underrun | the production question: what fraction of streams play continuously with a realistic buffer |
| **rejects / errors** | refused or failed requests | a fast server that drops requests is not fast |

`STREAM_RTF` is computed per request and then aggregated. Percentiles are taken over requests,
**never as a ratio of percentiles** — that is how a "part" once came out larger than the "whole"
in a table nobody could explain for a day.

All playback quantities are **client-observed**: a mark is the return of the client's chunked
read, which does not wait for the chunk's trailing CRLF but can return already-queued data
when the reader is late (several chunks then carry near-identical timestamps). The harness
counts those `coalesced reads`; when the share is small the cadence numbers are tight upper
bounds on server lateness, when it is large the run is a diagnostic. Detail and the transport
audit: `.work/professional-streaming-architecture.md` (MT-1). Total RTF, engine service time
and queue decomposition remain diagnostics that explain a KPI.

---

## 6. Input length is prefill, and it shows up in first audio

Prompt positions grow one-for-one with text tokens, and the prompt-prefix cache covers only the
request-independent head, so the first non-cached position is the first text token. Measured on
a 16-core Arm host at concurrency 1, the same three texts truncated to word prefixes:

| words | prompt positions | TTFA p50 |
|---:|---:|---:|
| 5 | 20 | 77 ms |
| 20 | 39 | 143 ms |
| 55 | 79 | 186 ms |

`STREAM_RTF` barely moves across the same range, because it measures what happens *after* the
first chunk. That is the expected shape and it is worth knowing before promising a latency
figure for a workload whose text length you have not seen.

### What the server refuses, and where that limit comes from

The same fact has a serving side: a request whose text cannot be finished inside the per-request
cap is refused at admission rather than started. The input limit is not a constant, it is derived
from two things the profile already fixes — a batch slot's prompt budget
(`QWEN_BATCH_MAX_PROMPT x 3.5` characters, 1792 at the default) and the generation cap
(`--max-request-seconds x 30` characters per second, 1800 at the default 60 s) — and the smaller
one wins, floored at 200. On a stock server that is 1792 characters, and the startup line says so:

```
[serve] per-request generation cap: 60 s -> text limit 1792 characters, frame cap 750 = 60.0 s of audio (from --max-request-seconds); a request that reaches the frame cap is TRUNCATED and logged (--max-request-seconds N / --max-text-chars N; 0 disables the text cap)
```

Two consequences worth having in mind before a deployment quotes anything. Lowering
`max_request_seconds` **tightens the accepted input length** with it, silently, because one is
computed from the other: a 10 s cap accepts 300 characters. And a profile that leaves
`max_text_chars` unspecified is not an unlimited server — it is a server that derives its limit,
reports it in `GET /v1/health` as `max_text_chars`, and refuses anything longer with a `400` that
names which of the two bounds it hit. Fix a number in the profile only to go *tighter* than the
derived one.

The rest of the envelope — `405`, `413`, `415`, unknown-field rejection, the `503` at queue full
or queue timeout, the parameter clamps — is one table in [`api.md`](api.md#limits-validation-and-errors),
and it is identical on all three POST endpoints and in both server modes.

---

## 7. Checking the audio, not only the clock

A configuration that is fast and broken is not a configuration.

```bash
python3 tools/wav_qc.py <dir>      # clipped runs, holes, step discontinuities, silent files
```

It answers "did the waveform break?" and says so in its own output: a clean row means the
waveform is intact, **not** that the speech is good. Its hole threshold is 1.2 s because natural
sentence pauses run 0.5–0.9 s, and a lower bar flags correct speech as broken.

The harness can save one WAV per request with `--save-audio DIR`. That perturbs timing on
purpose, so such a run is a quality gate and its percentiles must not be quoted.

**Listen at concurrency 1 and at concurrency ≥ 2W.** Above batch 2 the engine takes a different
kernel, so a defect there is invisible at concurrency 1. On the reference host the two paths
measured identical length and a maximum sample difference of 1 in 32768 — last-bit rounding
from a different arithmetic order, not a change of content.

---

## 8. Runtime flags

Every environment flag, its default per ISA, and the one-line incantation that restores the
previous numerics: [`feature-flags.md`](../feature-flags.md). The engine also declares what it
actually read, in one machine-readable line:

```
[FLAGS] v=1 pid=12345 QWEN_PREFIX_CACHE=1 ...
```

Two rules that have each cost a day:

- a flag that changes a default is written into the register **in the same change that
  introduces it**, because the symptom shows up days later and the first question is always
  "what is on by default now that was not on for the last good measurement?";
- a log that describes intentions rather than behaviour is how a bench measures one
  configuration believing it measured another. The server prints its **effective** state.

---

## 8b. Recommended decoder setup, split by ISA

The decoder is the one place where the right answer genuinely differs between x86 and Arm,
so keep the two lanes separate rather than carrying one "all-on" set across both.

### Arm (Neoverse-V2 / Graviton4 / Axion, KleidiAI)

| Knob | Setting | Why |
|---|---|---|
| decoder precision | `QWEN_SD_INT8=1` | per-item int8 DOTPROD is the qualified leaf; `--dispatch-map` must resolve `per-item-int8-dotprod` |
| residual convs | `QWEN_SD_RES1_V2=1` | DL-4 direct dilated int8; serves res1, res2 and the initial/pre convs |
| decoder lane | `QWEN_SD_LANE_SPLIT=4`, `QWEN_SD_LANE_ELASTIC=1` | private decoder team; engine narrows only while a unit is in flight. Linux-only, not ISA-specific |
| **stream cohort** | **`QWEN_SD_MULTISLOT=0`** | **retired on Arm.** Pairing two streams into one batched call costs 1.63x the sequential decode on Graviton4 and 1.20x on Axion. Numerically free to disable (`mel_corr 1.00000`) |
| Design D / fused residual | unsupported | x86 AMX only; the gates read `UNSUPPORTED`, which is correct, not a fallback |
| ConvT stack / ConvNeXt int8 | off unless separately qualified | `QWEN_SD_CONVT_STACK`, `QWEN_SD_CONVT_I8`, `QWEN_SD_CNEXT_I8` change numerics and need their own paired audio gate |

Reference profiles: `arm-product`, `aws-c8g-8xlarge-32c-arm-v2-all-on`,
`axion-c4a-highcpu32-0p6b-all-on`, `axion-16c-ttfa` &mdash; all four now ship
`QWEN_SD_MULTISLOT=0`.

### x86 (AVX-512 VNNI / AMX)

| Knob | Setting | Why |
|---|---|---|
| decoder precision | `QWEN_SD_INT8=1` (default on VNNI) | the VNNI leaf is the qualified default; AMX hosts add Design D |
| Design D | `QWEN_SD_AMX_D=1` on AMX hosts | persistent int8 packs, the AMX decoder path |
| residual convs | `QWEN_SD_RES1_V2=1` | same DL-4 leaf, VNNI twin |
| **stream cohort** | **`QWEN_SD_MULTISLOT=2` on the Turin product profile** | **kept, but on weaker evidence than the Arm retirement.** It was promoted from a combined lane+V2+cohort smoke, never from an isolated arm, and the VNNI multi kernel has the same runtime-indexed accumulator structure that spills on Arm. Treat as provisional until the three-cell microbench runs on x86 |
| control arm | `turin-c8a-32c-vnni-control` | the committed A/B arm: `QWEN_SD_RES1_V2=0`, `QWEN_SD_MULTISLOT=0` |

### The one-command check that the ISA lane is what you think

```bash
./qwen_tts --dispatch-map | grep -iE "decoder|multislot|res1"
python3 tools/serving_profile.py preflight <profile> --binary ./qwen_tts
```

The preflight records `multislot_active`, `feature_status.multislot` and
`resolved_decoder_mode`; a profile whose parity block lists `["ACTIVE", "VALID FALLBACK"]`
stays valid with the cohort either way, so the preflight passing is **not** on its own
evidence that the cohort is on or off. Read the recorded value.

---

## 9. A worked example: a 16-core Arm host

```bash
make blas GIT_REV=$(git rev-parse --short HEAD)
./qwen_tts --caps                      # confirm the matrix-unit paths are live
./qwen_tts --self-test                 # and that they compute the right thing
make bench-fingerprint                 # the machine describes itself, and gates on SMT

make bench-topo BENCH_MODEL=qwen3-tts-1.7b-base \
                BENCH_TOPO=1x16,2x8,4x4        # one row per cell; 1x16 wins C=1, 2x8 holds both
$EDITOR configs/perf/my-host.json      # topology 2x8, threads 8, batch 8, env from the sweep
python3 tools/perf_profile.py validate --engine ./qwen_tts

make bench-suite BENCH_PROFILE=my-host BENCH_MODEL=qwen3-tts-1.7b-base
```

and the quality gate is its own run, because saving the audio perturbs the timing:

```bash
python3 tests/serve_parallel_wave.py --model qwen3-tts-1.7b-base --bin ./qwen_tts \
    --speaker ryan --topo 2x8 --conc 2 --waves 1 --seed 42 --precision int8 \
    --profile my-host --classes short --out /tmp/qc --port 9600 --label qc \
    --save-audio /tmp/qc/audio
python3 tools/wav_qc.py /tmp/qc/audio      # then listen: no script scores speech
```

Then serve it with the same profile, so what runs is what was measured:

```bash
eval "$(tools/perf_profile.py command my-host --model MODEL_DIR --port 8080)"
```

---

## The same procedure on x86

Nothing above is Arm-specific: `bench-fingerprint`, `bench-topo` and `bench-suite` read the
machine and take the topology names from it. Three things differ in practice.

**SMT is usually on, and you have to turn it off** (see the section above) — the Arm instance
families report `Thread(s) per core: 1` on their own, the x86 ones do not.

**Build level is a decision.** `make blas` on Linux/x86 runs `SIMD=auto`, which reads
`/proc/cpuinfo`, probes the compiler and picks the highest level both support — `amx` on
Sapphire/Emerald Rapids, `avx512bf16` on Zen4/5, down to `portable`. It announces itself as
`[simd] auto -> …`, and the resulting binary is **not portable to an older CPU**. Pin the level
in the profile so the box that reproduces your numbers compiles the same kernels.

**The flags are not the same flags.** `QWEN_KAI_*` exists only where KleidiAI compiled in, so
it is inert on x86; the x86 side has the AMX and VNNI gates instead. Two values in particular
do not travel: `QWEN_POOL_SPIN`, where the Arm profile's 65536 is measurably wrong on 8 cores,
and `QWEN_DECODER_BATCH`, which pays only when a worker really holds several slots. Both are
covered in [`feature-flags.md`](../feature-flags.md) with the measurement on each side.

An x86 run of the same inner loop, with the profile that ships for an 8-core AMX host:

```bash
make blas SIMD=amx GIT_REV=$(git rev-parse --short HEAD)
make bench-topo  BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=x86-8c-amx-recommended \
                 BENCH_TOPO=1x8,2x4,4x2 BENCH_CONC=1,4
make bench-suite BENCH_MODEL=qwen3-tts-1.7b-base BENCH_PROFILE=x86-8c-amx-recommended \
                 BENCH_RUNG=fast BENCH_TOPO=2x4 BENCH_OUT=/tmp/bench_x86
```

Note the topology names: on an 8-core box the cells are `1x8`, `2x4` and `4x2`, not the 16-core
`2x8`/`4x4`. Pick them from `make bench-fingerprint`, never by copying another host's profile.

**What that box can and cannot do**, measured and written up in
[`reference-x86-8c-amx.md`](../reference-x86-8c-amx.md): on the 1.7B, first audio is competitive —
C=4 TTFA p95 252 ms — while **sustained stream RTF at C=4 is 1.43**, so it serves four concurrent
requests with a good time to first audio and keeps one of them realtime. That is a bandwidth
result, not a kernel one: 82 GB/s against the Arm host's 336. The 0.6B, measured separately on
the same box, holds two concurrent realtime streams.

---

## See also

- [`reference-arm-16c.md`](../reference-arm-16c.md) — every rung of this suite, measured on one
  16-core Arm box: topology sweep, qualification curve for both models, input-length effect,
  the three arrival models, the profile A/B
- [`reference-x86-8c-amx.md`](../reference-x86-8c-amx.md) — the same procedure on an 8-core Intel
  AMX box: what AMX buys per stage, and where this class of machine stops
- [`x86-optimization.md`](../x86-optimization.md) — the x86 kernel work behind those numbers
- [`api.md`](api.md) — the HTTP API
- [`feature-flags.md`](../feature-flags.md) — every runtime flag and its default
- [`configs/perf/README.md`](../../configs/perf/README.md) — the profile format
- [`ENGINEERING-METHOD.md`](../ENGINEERING-METHOD.md) — why the measurement rules above are shaped
  the way they are
