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

### The procedure

```bash
# 1. what does this machine actually have?
make bench-fingerprint            # cpu, cores, SMT, cache, kernel, BLAS, ISA extensions
./qwen_tts --caps                 # which kernels the binary would pick, per batch width
./qwen_tts --self-test            # cross-ISA correctness oracle

# 2. sweep the topologies at the concurrency you expect
make bench-fast   TOPO=1x16,2x8,4x4   # short bank, the fast inner loop
make bench-realistic                  # a mixed-length bank, the one that can be quoted

# 3. write the winner into a profile, and stop retyping it
$EDITOR configs/perf/<your-host>.json
```

Steps 1 and 2 are cheap. Step 3 is what makes the result last: see §3.

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

Its scope is measured too, not assumed: at concurrency 4 on the same host the difference is 173
against 169 ms, i.e. nothing. When every worker is busy there is no idle time for a spinning
thread to waste. **So the profile matters most exactly where first-audio latency is measured.**

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
make bench-suite MODEL=<dir> SPEAKER=<voice> \
                 BANK_FAST=<file> BANK_REAL=<file> CORPUS_ARG=<file>
```

With no variables it runs a generic configuration end to end, which is also the fastest way to
check that a new box works at all. Everything deployment-specific arrives on the command line
and lands in that run's manifest rather than in a tracked file.

The suite owns the whole invocation, and every preflight check exits non-zero:

- the binary exists, runs, and reports its own sha256 and build tag
- the profile resolves, and its forbidden variables are absent from the environment
- no stale engine processes, and the box is idle
- the source commit and dirty flag travel with the numbers
- each rung runs, then its identity gate
- **each cell's audio length is printed**, so comparability can be read instead of assumed
- a manifest repeats the exact commands

| rung | what it answers |
|---|---|
| `realistic` | the quotable curve: first audio and sustained realtime against concurrency |
| `fast` | the engineering inner loop — short texts, fast iteration, never a production figure |
| `short-diverse` / `long-diverse` | how input length moves first audio, which it does a lot |

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
make blas
./qwen_tts --caps                      # confirm the matrix-unit paths are live
make bench-fingerprint                 # the machine describes itself

make bench-fast TOPO=1x16,2x8,4x4      # → 2x8 wins at the target concurrency
$EDITOR configs/perf/my-host.json      # topology 2x8, threads 8, batch 8, env from the sweep
tools/perf_profile.py validate

make bench-suite PROFILE=my-host       # the numbers, with their provenance and a manifest
python3 tools/wav_qc.py /tmp/bench_suite/audio
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
