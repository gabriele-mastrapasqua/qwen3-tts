# Running the server in production, and finding the configuration first

[`server.md`](server.md) is the API: endpoints, request bodies, streaming. **This document is
the other half** — how to run the process, how to find the configuration for a particular
machine before quoting any number from it, and how to measure it so the number means
something a week later.

The short version: **a serving configuration is discovered on the box, not chosen from a
datasheet.** The engine ships a break-in procedure and a benchmark suite for exactly that, and
a profile format so the answer survives the session that found it.

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

**Core slices come from `--prefork`, not from `--prefork-threads`.** Worker *w* is pinned to
the contiguous logical CPUs `[w * (nproc / W), …]`; `--prefork-threads` then sizes the pool
*inside* that slice. More threads than the slice is oversubscription, fewer leaves cores idle,
and when *W* does not divide `nproc` the remainder is unused. There is no CLI way to exclude
SMT siblings: the slices are contiguous *logical* ids, so read `lscpu -e` to see which id is a
sibling of which core, and which socket each belongs to.

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
for `1x16` because it comes from the pre-fork parent's counters, and a single server has no
parent — it answers `SIGUSR1` with its pool counters instead.

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

---

## 5. The three numbers, and the many that are not

| | what it is | why it is the one |
|---|---|---|
| **TTFA** | send → first audio chunk | what a caller hears as responsiveness |
| **STREAM_RTF** | `(t_done − t_first_chunk) / (audio after the first chunk)`, **per request** | below 1.0 a player starting at the first chunk never stalls |
| **rejects / errors** | refused or failed requests | a fast server that drops requests is not fast |

`STREAM_RTF` is computed per request and then aggregated. Percentiles are taken over requests,
**never as a ratio of percentiles** — that is how a "part" once came out larger than the "whole"
in a table nobody could explain for a day.

Everything else the harnesses print — total RTF, engine service time, queue decomposition,
prebuffer and underrun simulation — is **diagnostic**. It explains a KPI; it does not become one.

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
previous numerics: [`feature-flags.md`](feature-flags.md). The engine also declares what it
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

## See also

- [`server.md`](server.md) — the HTTP API
- [`feature-flags.md`](feature-flags.md) — every runtime flag and its default
- [`configs/perf/README.md`](../configs/perf/README.md) — the profile format
- [`ENGINEERING-METHOD.md`](ENGINEERING-METHOD.md) — why the measurement rules above are shaped
  the way they are
