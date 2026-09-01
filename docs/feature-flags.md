# Runtime flags: what each one does, what it defaults to, and how to prove it is on

Every lever in this list is an environment variable read at runtime. They exist because a
default that is right on one machine is wrong on another, and because a measurement needs a
way to go back to the previous numerics without a rebuild.

**A flag is on when the process says so, never when the invocation intended it.** The engine
prints one machine-readable line at startup naming every registered variable it actually read:

```
[FLAGS] v=1 pid=12345 QWEN_PREFIX_CACHE=1 QWEN_POOL_SPIN=65536 ...
```

`./qwen_tts --caps` prints the same set in human form (`active flags:`), and
`tools/perf_profile.py check-flags <profile> --log server.log` compares that line against what
a deployment profile asked for. The register itself lives in `qwen_tts_kernels.c`
(`g_qwen_reported_flags[]`); `perf_profile.py validate --engine ./qwen_tts` reads it from the
source and refuses a profile that sets a variable this engine would never declare.

Two rules that have each cost a day:

- a flag that changes a default is added to the register **in the same change that introduces
  it**, because the symptom shows up days later and the first question is always "what is on
  by default now that was not on for the last good measurement?";
- the flags in the tables below are *declared*; the ones in
  [§8](#8-levers-outside-the-register) are not, so no log can prove their state. Prefer a
  declared lever when both exist.

---

## 0. The set that makes first audio fast, and why it is a set

Most of the first-audio work of the last cycle landed as **defaults plus a handful of
variables that have to travel together**. Run the server without them and nothing errors —
you simply measure a slower machine, which is the failure mode this page exists to prevent.
The reference 16-core Arm deployment (`configs/perf/axion-16c-ttfa.json`) pins exactly these:

| variable | value | what it buys, measured |
|---|---|---|
| `OPENBLAS_THREAD_TIMEOUT` | `1` | without it OpenBLAS idles by **spinning** and contends with the engine's own pool: first audio at C=1 measured 108 ms without against 66 ms with, and the bare arm was *bimodal* — 42,500 context switches per second against 12,000. That was measured when it was worth nothing at C=4; the re-measurement below found the scope reversed, so read both |
| `OPENBLAS_NUM_THREADS` | **absent** | the engine sizes BLAS per worker at startup and backs off entirely when this is already set, so a stray `export` silently replaces a qualified thread split |
| `QWEN_PREFIX_CACHE` | `1` | reuses the request-independent prompt head; on production prompts that is 9 of 13–79 prompt positions never computed again |
| `QWEN_PREFILL_MATMAT` | `1` | routes prefill projections through the native bf16 matmat instead of BLAS: −29% prefill at `-j1` and −46% at `-j16` on that host. It is already the default where a matrix unit exists; pinning it means a change of default elsewhere cannot move this deployment quietly |
| `QWEN_KAI_NCHUNK` | `384` | sub-tiles the GEMM's n dimension so the microkernel's second pass finds the packed RHS in cache: prefill p50 45.0 → 42.9 ms, first audio 72.2 → 70.2 ms, output bitwise identical. 192 and 96 both measured worse — smaller is not better |
| `QWEN_POOL_SPIN` | `65536` | generations a pool worker re-reads before parking. The 4096 default tuned elsewhere cost 40% of the Code Predictor here: 16.0 → 9.6 ms/frame, 491,320 → 35,132 context switches |
| `QWEN_DECODER_BATCH` | `1` | one pass over the speech-decoder weights for **all** active slots: +8.1% throughput with first audio p50 −8.9% at C=4. The server turns this on itself; the profile pins it so the run records what was on |
| `QWEN_STREAM_DECODE_CHUNK` | `8` | frames per streaming chunk — the trade between first-chunk latency and per-chunk overhead |

**What that set is worth today is not what it was worth when it was found.** Re-measured on the
same 16-core Arm host with the current build, `2x8`, four waves of a short bank, the only
difference between the arms being whether the profile environment was applied:

| arm | C | TTFA p50 | stream RTF p50 | context switches/s |
|---|---:|---:|---:|---:|
| with the profile | 1 | 57 ms | 0.44 | 11,910 |
| compiled defaults | 1 | 51 ms | 0.45 | 7,880 |
| with the profile | 4 | **109 ms** | **0.72** | 30,153 |
| compiled defaults | 4 | 133 ms | 0.81 | 68,984 |

At one request the two are within noise: `QWEN_PREFILL_MATMAT`, `QWEN_PREFIX_CACHE`,
`QWEN_POOL_SPIN` and `QWEN_DECODER_BATCH` have all **become defaults** since that table was
written, so setting them changes nothing on a current build of this platform — they are pinned
in the profile so that a change of default elsewhere cannot move a qualified deployment
quietly. At four requests the profile is 18% ahead on first audio, and the context-switch
column names the cause: the OpenBLAS thread timeout, the one lever in the set that is *not* a
compiled default and cannot be one, because it belongs to a library the engine only links.

Two things follow from all of that:

- **the levers are of different kinds and they compose**: one keeps a foreign thread pool from
  stealing cores (`OPENBLAS_*`), two change which kernel runs (`QWEN_PREFILL_MATMAT`,
  `QWEN_KAI_NCHUNK`), one changes how the engine's own pool waits (`QWEN_POOL_SPIN`), and two
  are about serving many requests at once (`QWEN_DECODER_BATCH`, `QWEN_STREAM_DECODE_CHUNK`).
  A batching win can be eaten by a spinning BLAS, so measuring one at a time on a box that has
  none of the others tells you very little;
- **the scope of each is measured too**: the OpenBLAS lever is worth 40 ms at concurrency 1 and
  nothing at concurrency 4. A flag with no effect on your workload is not a flag to cargo-cult;
  it is a flag to leave to the profile that measured it.

The practical form of all this is not a shell alias but a profile:

```bash
eval "$(tools/perf_profile.py command axion-16c-ttfa --model MODEL_DIR --port 8080)"
```

which emits the argv **and** the environment, and `check-flags` afterwards proves the process
read them. See [`serving-operations.md`](serving-operations.md) §3.

### One deployment, explained piece by piece

A single missing argument or variable does not fail — it changes what the engine does, and the
only visible trace is a number that is worse than it should be. So here is one complete,
qualified invocation with every part named. It is what `perf_profile.py command axion-16c-ttfa`
emits, and nothing in it is decorative:

```bash
OPENBLAS_THREAD_TIMEOUT=1 QWEN_DECODER_BATCH=1 QWEN_KAI_NCHUNK=384 QWEN_POOL_SPIN=65536 \
QWEN_PREFILL_MATMAT=1 QWEN_PREFIX_CACHE=1 QWEN_STREAM_DECODE_CHUNK=8 \
QWEN_STREAM_DECODE_CHUNK_BUSY=0 \
./qwen_tts -d MODEL_DIR --int8 --serve 8080 --batch-size 8 \
           --prefork 2 --prefork-threads 8 \
           --max-queue 1 --queue-timeout-ms 0 --max-request-seconds 60
```

| part | what it decides | leave it out and |
|---|---|---|
| `--int8` | Talker and Code Predictor weights quantized at load | the box reads bf16 weights: on a bandwidth-bound machine that is the single biggest regression available |
| `--prefork 2` | two worker processes, each pinned to 8 contiguous CPUs, weights shared copy-on-write | one process shares one pool across every request; measured worse for first audio here, and the sweep is what said so |
| `--prefork-threads 8` | pool size inside each slice | the pool is sized from the machine, not from the slice, and threads cross the pinning |
| `--batch-size 8` | per-worker in-flight cap **and** the continuous-batching scheduler | at `1` each worker serves one request at a time and never reaches the GEMM path, so concurrency turns into queueing |
| `--max-queue 1` | one request may wait beyond the slots | unbounded waiting: a caller sees latency instead of a refusal, which is the worse failure |
| `--queue-timeout-ms 0` | no deadline on that wait | — (0 is the deliberate choice here, recorded so a later change is visible) |
| `--max-request-seconds 60` | generation cap per request, from which a text-length limit is derived | one pathological text can hold a slot for minutes |
| `OPENBLAS_THREAD_TIMEOUT=1` | OpenBLAS parks instead of spinning | the two pools fight for the same cores: 108 ms against 66 ms for first audio at C=1, bimodally |
| `QWEN_PREFILL_MATMAT=1` | prefill projections on the native bf16 matmat | prefill goes back through BLAS: −29% / −46% (at `-j1` / `-j16`) given away |
| `QWEN_KAI_NCHUNK=384` | GEMM n-dimension sub-tiling for cache reuse | ~2 ms of prefill p50 and ~2 ms of first audio, with bit-identical output |
| `QWEN_PREFIX_CACHE=1` | the request-independent prompt head is computed once | every request recomputes the same leading positions |
| `QWEN_POOL_SPIN=65536` | how long a pool worker re-reads before parking | the Code Predictor pays 40% more on this host, in context switches |
| `QWEN_DECODER_BATCH=1` | one decoder pass for all active slots | +8.1% throughput and −8.9% first audio p50 at C=4, given away |
| `QWEN_STREAM_DECODE_CHUNK=8` | frames per streamed chunk | the first chunk arrives later or the stream pays more overhead, depending which way it moves |

Three of those are the prefill work specifically — `QWEN_PREFILL_MATMAT`, `QWEN_KAI_NCHUNK`
and `QWEN_PREFIX_CACHE` — and prefill is most of what first audio *is* at concurrency 1. That
is also why first audio moves with the length of the input text (measured 77 → 143 → 186 ms
for 5, 20 and 55 words on this host): the prompt-prefix cache covers the request-independent
head, so the first position it cannot reuse is the first text token.

**Then prove it rather than trusting the line above.** The engine prints the flags it read,
and `tools/perf_profile.py check-flags axion-16c-ttfa --log server.log` compares them:

```
ok: engine declares QWEN_DECODER_BATCH=1 QWEN_KAI_NCHUNK=384 QWEN_POOL_SPIN=65536
    QWEN_PREFILL_MATMAT=1 QWEN_PREFIX_CACHE=1 QWEN_STREAM_DECODE_CHUNK=8
    QWEN_STREAM_DECODE_CHUNK_BUSY=0 pid=7491
```

---

## 1. Restoring the previous numerics

The flags that can change the arithmetic (not just the speed) are the kernel gates, the
prefill route, the batched decoder and the q4 quantizer. One line puts an ARM box back on the
pre-matrix-unit, pre-batched-decoder path:

```bash
QWEN_PREFILL_MATMAT=0 QWEN_DECODER_BATCH=0 QWEN_NO_SDOT=1 QWEN_NO_BFMMLA=1 \
QWEN_NO_SMMLA=1 QWEN_NO_KLEIDI=1 QWEN_Q4_NAIVE=1 ./qwen_tts -d MODEL --text "..."
```

and the x86 equivalent swaps the ARM gates for `QWEN_NO_VNNI=1 QWEN_NO_AMX=1
QWEN_NO_BF16DOT=1 QWEN_SD_INT8=0`.

Everything else on this page changes scheduling, memory or diagnostics — not the samples.
When you need to know whether a lever moved the audio rather than the clock, compare with
`tests/compare_audio.py` (mel-correlation), never with a checksum: a different arithmetic
order is expected and benign, a different *result* is not.

---

## 2. Kernel dispatch — which GEMM runs

The default is always "use the widest primitive this build has, at the batch width where it
wins". These exist to take one away and measure what it was worth.

| flag | ISA | default | effect |
|---|---|---|---|
| `QWEN_NO_SDOT` | ARM | unset | `=1` drops the dotprod int8 path back to f32 accumulation |
| `QWEN_NO_SMMLA` / `QWEN_NO_BFMMLA` | ARM | unset | `=1` drops the i8mm / bf16 matmat kernels |
| `QWEN_ARM_BFDOT` | ARM | unset (off) | `=1` opts *into* BFDOT for the bf16 matvec |
| `QWEN_APPLE_MMLA` | Apple | unset (off) | MMLA is opt-in on Apple silicon; `=1` enables it |
| `QWEN_NO_VNNI` | x86 | unset | `=1` drops the VNNI int8 path (matvec and matmat) |
| `QWEN_NO_BF16DOT` | x86 | unset | `=1` drops the AVX-512 bf16 dot path |
| `QWEN_NO_AMX` | x86 | unset | `=1` disables every AMX matmat kernel at once |
| `QWEN_NO_AVX2MM` | x86 | unset | `=1` drops the AVX2 matmat |
| `QWEN_AMX_MIN_B` · `QWEN_VNNI_MIN_B` · `QWEN_AVX2MM_MIN_B` | x86 | 4 · 2 · 2 | smallest batch width that may take that matmat |
| `QWEN_BFMMLA_MIN_B` · `QWEN_SMMLA_MIN_B` · `QWEN_KLEIDI_MIN_B` | ARM | 2 · 2 · 1 | the same thresholds on the ARM kernels |

A gate for a kernel the build does not contain is simply inert, so an invocation can carry
both families — but only the ones for this ISA will appear in the `[FLAGS]` line, and only if
you set them.

`--caps` answers what the binary *would* pick, per batch width, and `--self-test` is the
cross-ISA correctness oracle for the kernel you just enabled or disabled. Both are cheap and
both belong before any number.

## 3. Prefill

| flag | default | effect |
|---|---|---|
| `QWEN_PREFILL_MATMAT` | on where the build has a bf16 matrix unit (AMX or ARM BF16), else BLAS | `=0` routes prefill projections back through BLAS, `=1` forces the native matmat |
| `QWEN_PREFILL_QUANT` | off | `=1` runs prefill on the quantized weights and frees the bf16 copy (~4 GB on the 1.7B). **It measurably costs the accent on a finetune** — measured language identification 96% → 38%. Base models only, and the server says so when you turn it on |
| `QWEN_KAI_NCHUNK` **(ARM only)** | 384 | sub-tiles the KleidiAI GEMM's n dimension so the second height pass finds the packed RHS in cache. `=0` restores one kernel call per slice |
| `QWEN_KAI_OPS` **(ARM only)** | all families on | comma list restricting which KleidiAI families may be used; empty means every one |
| `QWEN_KAI_REPEAT` **(ARM only)** | off | `=1` times a second identical call — a microbenchmark, not a serving flag |

## 4. Server, admission and first audio

| flag | default | effect |
|---|---|---|
| `QWEN_PREFIX_CACHE` | **on** | reuses the request-independent prompt head across requests; `=0` disables it |
| `QWEN_POOL_SPIN` | 65536 on Linux/arm64, 4096 elsewhere | generations a pool worker re-reads before parking on the condvar. On a 16-core Arm host 4096 cost 40% of the Code Predictor: 65536 measured CP 16.0 → 9.6 ms/frame and 491,320 → 35,132 context switches. `=0` parks immediately |
| `QWEN_SERVE_BLAS` | 0 (the engine's own thread budget) | BLAS threads while a single slot is busy |
| `QWEN_SERVE_BLAS_BUSY` | 0 (same) | BLAS threads from two busy slots up |
| `QWEN_TTFA_PRIORITY` | 0 (off) | N > 0 lets N prefilling requests take priority over decoding ones, clamped to 8 |
| `QWEN_ADMIT_M1` | off | admits a new request one step earlier in the scheduler; opt-in, measured per box |
| `QWEN_THP` | off | `=1` advises transparent huge pages over the mapped weights (Linux) |

Note that `OPENBLAS_NUM_THREADS` is not in this table because it must be **absent**: the engine
sizes OpenBLAS itself at startup and backs off entirely when that variable is already set, so
an unrelated `export` in the operating shell silently replaces a qualified thread split.
`tools/perf_profile.py forbidden-env <profile>` lists the variables a deployment declares must
not be present, and the benchmark suite refuses to run when one is.

## 5. Speech decoder and streaming

| flag | default | effect |
|---|---|---|
| `QWEN_DECODER_BATCH` | **on in the server** (`[serve]` says so), off in the CLI | one pass over the decoder weights for every active slot. `=0` opts out |
| `QWEN_SERVER_NO_DECODER_BATCH` | unset | present = the server does not turn the above on for you |
| `QWEN_DECODER_THREAD` | off | runs the decoder on its own thread beside the Talker |
| `QWEN_STREAM_DECODE_CHUNK` | 8 (max 32) | frames decoded per streaming chunk |
| `QWEN_STREAM_DECODE_CHUNK_BUSY` | 0 (off) | a different chunk size once more than one slot is busy |
| `QWEN_DECODER_GANG_LEAD` | 4 | slots from which the decoder gang gets a leader |
| `QWEN_DECODER_GANG_MIN` | 2 | smallest gang that is worth forming |
| `QWEN_SD_INT8` | on where the build has AVX-512 VNNI, off elsewhere | int8 speech-decoder convolutions; `=0` forces fp32 |

## 6. Precision and voice

| flag | default | effect |
|---|---|---|
| `QWEN_CP_PREC` | follows `--int8` / `--int4` | `int8` or `int4` for the Code Predictor alone — the lever behind the mixed-precision configurations |
| `QWEN_TALKER_PREC` | follows `--int8` / `--int4` | the same for the Talker |
| `QWEN_CP_Q2_FFN` | off | `gateup`, `down` or `both` push those Code Predictor projections to 2 bits. Quality gate first |
| `QWEN_ICL_FRAMES` | the context's own cap | caps the reference frames an in-context voice keeps (anchor dilution) |
| `QWEN_Q4_NAIVE` | unset | present = the legacy q4_0 quantizer instead of the LSQ scale |

## 7. Diagnostics — never in a run that produces a number

`QWEN_TTFA_TRACE`, `QWEN_SD_PHASE`, `QWEN_LIFE_TRACE`, `QWEN_REQ_TRACE`, `QWEN_BATCH_STATS`,
`QWEN_SERVE_PROFILE`, `QWEN_TF_CODES`, `QWEN_TF_PREFIX`.

They print phase tables, per-request lifecycles and kernel censuses, and every one of them
costs time inside the region being timed. A deployment profile declares them `null` for that
reason: counters and timing do not share a binary in a run that produces a published figure.

## 8. Levers outside the register

These change behaviour but are **not** printed in the `[FLAGS]` line, so `check-flags` cannot
verify them and a log will not show them. Use them for an experiment, not for a deployment:

`QWEN_NO_BFMMLA`, `QWEN_NO_SMMLA`, `QWEN_NO_KLEIDI`, `QWEN_NO_AVX2MM`, `QWEN_APPLE_MMLA`
(MMLA is opt-in on Apple silicon), `QWEN_PREFORK_ELASTIC` (set by `--prefork-elastic`),
`QWEN_MAX_REQUEST_S` and `QWEN_MAX_TEXT_CHARS` (also `--max-request-seconds` /
`--max-text-chars`), `QWEN_QUEUE_UNBOUNDED` (removes the queue bound — the old behaviour, kept
for A/B only), `QWEN_SERVER_STRICT`, `QWEN_CANCEL_ON_DISCONNECT`, `QWEN_TTFA_FREEZE_CAP`,
`QWEN_TTFA_PRIO_STRICT`, `QWEN_FREE_BF16`, `QWEN_PREFILL_HELPER`, `QWEN_POOL_NARROW`.

Where a CLI flag exists for the same thing, the CLI flag is the one to use: it lands in the
process arguments, which a `ps` can read months later.

---

## 9. What applies on which ISA, and what x86 does not have yet

A profile written on one architecture does not port by copying. Three groups:

**Everywhere** — `QWEN_PREFIX_CACHE`, `QWEN_POOL_SPIN`, `QWEN_DECODER_BATCH`,
`QWEN_DECODER_THREAD`, `QWEN_STREAM_DECODE_CHUNK*`, `QWEN_DECODER_GANG_*`, `QWEN_TTFA_*`,
`QWEN_ADMIT_M1`, `QWEN_SERVE_BLAS*`, `QWEN_CP_PREC`, `QWEN_TALKER_PREC`, `QWEN_PREFILL_QUANT`,
every diagnostic, and `OPENBLAS_THREAD_TIMEOUT` / `OPENBLAS_NUM_THREADS` wherever OpenBLAS is
the BLAS. `QWEN_THP` is Linux-only in effect, whatever the CPU. These are the ones a profile
carries across a port unchanged.

**ARM only** — every `QWEN_KAI_*` and `QWEN_NO_KLEIDI`, because KleidiAI is compiled in only
when the toolchain reports `__ARM_FEATURE_MATMUL_INT8` or `__ARM_FEATURE_BF16`; plus
`QWEN_NO_SDOT`, `QWEN_NO_SMMLA`, `QWEN_NO_BFMMLA`, `QWEN_ARM_BFDOT`, `QWEN_APPLE_MMLA` and
their `*_MIN_B` thresholds. `QWEN_PREFILL_MATMAT` exists on both, but what it selects differs:
the KleidiAI bf16 matmat on ARM, the AMX one on x86.

**x86 only** — `QWEN_NO_VNNI`, `QWEN_NO_AMX`, `QWEN_NO_AVX2MM`, `QWEN_NO_BF16DOT`,
`QWEN_SD_INT8` (on by default only where AVX-512 VNNI exists), and the AMX/VNNI/AVX2
thresholds.

**The gap worth naming:** there is no x86 counterpart to `QWEN_KAI_NCHUNK` today. On ARM that
lever exists because the GEMM's n dimension is sub-tiled so the microkernel's second pass finds
the packed right-hand side still in cache; on x86 the AMX and VNNI matmats are entered through
the gate table's batch/rows/cols thresholds and have no cache sub-tiling knob at all. So an
x86 deployment profile can pin *when* those kernels are used, but not how they tile — if a
future x86 host shows the same second-pass cache miss, the knob has to be written, not
configured. Say so in the profile's `qualification.notes` rather than silently copying the ARM
value into a file where it does nothing.

---

## See also

- [`serving-operations.md`](serving-operations.md) — how to run and measure the server
- [`reference-arm-16c.md`](reference-arm-16c.md) — one box with all of this applied, measured
- [`configs/perf/README.md`](../configs/perf/README.md) — the profile format that pins these values
- [`performance.md`](performance.md) — what the numbers mean once they are taken
