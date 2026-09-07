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
| `QWEN_NO_AMX` | x86 | unset | `=1` disables every AMX matmat kernel at once. Use it to answer "is AMX doing anything", never to attribute a result — it removes two unrelated consumers |
| `QWEN_NO_AMX_BF16` · `QWEN_NO_AMX_INT8` · `QWEN_NO_AMX_Q4` | x86 | unset | one AMX consumer each, which is what a measurement needs. On an 8-core Emerald Rapids the two do disjoint jobs: dropping **bf16** costs C=1 TTFA +39% and C=4 p95 +72% while stream RTF barely moves (it is the *prefill*), and dropping **int8** leaves TTFA alone while costing 9% of RTF and 10% of throughput (it is the *decode*) |
| `QWEN_NO_AVX2MM` | x86 | unset | `=1` drops the AVX2 matmat |
| `QWEN_NO_BF16_MATMUL` | x86 | unset | `=1` drops the AVX-512 bf16 matmat, leaving the per-row twin. Only reachable where AMX is absent or declined |
| `QWEN_NO_VNNI_TILE` | x86 | unset | `=1` drops the *tiled* VNNI matmat back to one row at a time. It does **not** disable VNNI — that is `QWEN_NO_VNNI` |
| `QWEN_VNNI_TILE_M4N2` | x86 | unset (off) | `=1` tries the fixed `M4xN2` VNNI tile for observed `B=2` calls. It is an opt-in candidate inspired by the ARM small-B cross-product path; qualify it on the complete server path before enabling it |
| `QWEN_VNNI_GEMV_MR` | x86 | 2 | output rows handled by the VNNI GEMV microkernel; `4` is an experimental alternative and must be qualified per workload |
| `QWEN_NO_VNNI_QKV` | x86 | unset | `=1` drops the fused Q/K/V **GEMV** (one activation quantisation shared by the three projections) back to three separate matvecs |
| `QWEN_NO_X86_QKV` | x86 | unset | `=1` drops the fused Q/K/V **matmat** (int8 and bf16, VNNI and AMX) back to three separate matmats. This is the gate the persistent regions ask about, not the one above |
| `QWEN_VNNI_TILE_N8` | x86 | unset (off) | `=1` tries the 8-column VNNI tile. Opt-in candidate; qualify on the server path |
| `QWEN_Q4_VNNI_V3` · `QWEN_Q4_VNNI_V4` | x86 | v3 on | which q4 VNNI microkernel variant runs; `QWEN_Q4_VNNI_V4=1` selects the v4 experiment |
| `QWEN_AMX_PREPACK` | x86 AMX | **off** | `=1` pre-tiles weights once into the AMX tile layout and caches them by source pointer. When off the kernel simply reads the source with stride `cols`: there is NO per-call re-tiling anywhere, and the earlier claim that `=0` re-tiles per call was wrong |
| `QWEN_AMX_PREPACK_KINDS` | x86 AMX | all | limits the prepack cache to some weight kinds. Recognised values are `int8`, `bf16`, `both`, `all` ONLY - `q4` is not one of them and silently disables ALL prepacking |
| `QWEN_AMX_PERSIST_CFG` | x86 AMX | on | keeps the AMX tile configuration loaded across calls instead of `ldtilecfg`/`tilerelease` per call |
| `QWEN_AMX_B32` | x86 AMX | unset (off) | prototype 32-wide AMX int8 matmat. No production caller; reachable only from `make x86-amx-b32-bench` |
| `QWEN_VNNI_PREPACK` | x86 | unset | `=1`/`all` prepack eligible INT8 matrices in the parent; `=cp` or `=talker` limits the parent prepack to one component. It changes the batched VNNI matmat layout, not GEMV. Keep unset unless the target host shows a stable end-to-end win |
| `QWEN_NO_VNNI_ROWSUM` | x86 | unset | `=1` disables the cached INT8 weight row sums used by VNNI GEMV; keep unset for the native path |
| `QWEN_NO_VNNI_ACT_QUANT` | x86 | unset | `=1` disables AVX-512 activation quantization used before VNNI; keep unset for the native path |
| `QWEN_AMX_MIN_B` · `QWEN_VNNI_MIN_B` · `QWEN_AVX2MM_MIN_B` | x86 | 4 · 2 · 2 | smallest batch width that may take that matmat |
| `QWEN_AMX_BF16_MIN_B` · `QWEN_AMX_INT8_MIN_B` | x86 | fall back to `QWEN_AMX_MIN_B` | split the AMX gate when one threshold does not suit both datatypes; each overrides the shared one for its type only |
| `QWEN_AMX_INT8_QKV_MIN_B` | x86 | inherits `QWEN_AMX_INT8_MIN_B` | additional lower bound for the fused INT8 QKV path only; other INT8 projections keep the normal AMX gate |
| `QWEN_AMX_INT8_MIN_ROWS_PER_THREAD` | x86 | 256 | AMX INT8 also needs enough output rows PER WORKER (`rows >= N * threads`; the fused QKV counts `q+2kv`). Measured: below ~256 the tile setup and activation pack are not amortised and VNNI wins, and the same projection flips sign with the thread count. 0 disables the rule |
| `QWEN_BFMMLA_MIN_B` · `QWEN_SMMLA_MIN_B` · `QWEN_KLEIDI_MIN_B` | ARM | 2 · 2 · 1 | the same thresholds on the ARM kernels |

The batch gates say *when* a kernel is allowed; these say *how it tiles the output rows* once it is:

| flag | ISA | default | effect |
|---|---|---|---|
| `QWEN_X86_NCHUNK` | x86 | 0 (off) | output-row chunk for every x86 matmat that has no family value set |
| `QWEN_AMX_NCHUNK` · `QWEN_VNNI_NCHUNK` · `QWEN_AVX512_NCHUNK` | x86 | 0 (off) | the same, per family: AMX int8/bf16, VNNI int8, AVX-512 bf16. A family value overrides `QWEN_X86_NCHUNK` |
| `QWEN_KAI_NCHUNK` | ARM | 384 | the KleidiAI equivalent, on by default because it was measured to win there |

Zero, unset or a value below one row tile means "one call per thread slice", which is the shape
the kernels had before the knob existed — so leaving these alone reproduces the old numbers
exactly. Values are rounded **down** to the kernel's row tile (16 for AMX, 4 or 2 for the
AVX-512 families depending on batch width), and a value that rounds to less than one tile is
ignored rather than honoured as "no chunking at all".

A gate for a kernel the build does not contain is simply inert, so an invocation can carry
both families — but only the ones for this ISA will appear in the `[FLAGS]` line, and only if
you set them.

`--caps` answers what the binary *would* pick, per batch width, and `--self-test` is the
cross-ISA correctness oracle for the kernel you just enabled or disabled. Both are cheap and
both belong before any number.

## 3. Prefill

| flag | default | effect |
|---|---|---|
| `QWEN_PREFILL_MATMAT` | on where the build has a bf16 matrix unit: AMX, ARM BF16 (not Apple), or **AVX-512 BF16**; else BLAS | `=0` routes prefill projections back through BLAS, `=1` requests the native matmat but cannot create a missing capability. If the request is `1` and no compiled/runtime-enabled native BF16 unit exists, the engine resolves to the auditable BLAS fallback and the profile/dispatch gate refuses the run. The AVX-512-BF16 arm was added 2026-09-03: before it, an AVX-512-BF16 host **without AMX** fell back to BLAS, which converts every weight matrix to f32 first — and that conversion (`bf16_to_f32_matrix`) is single-threaded, so it became a serial stage in front of a parallel GEMM. Measured on AWS c8a.4xlarge (EPYC 9R45, 16c, no AMX) in server mode, four sequential arms: TTFA p50 416 → 124 ms at C=1, p95 939 → 507 ms at C=4, TOTAL_RTF −13% at C=4, with the delta localised in the server-side `admission + prefill` stage (2844 → 1126 ms) while Talker and CP absolute times and `STREAM_RTF` were unchanged. **It changes the sampled trajectory** (mel-corr ~0.4–0.8 against the BLAS path on the same prompt), so it is a different generation, not a rounding difference |
| `QWEN_PREFILL_QUANT` | off | `=1` runs prefill on the quantized weights and frees the bf16 copy (~4 GB on the 1.7B). **It measurably degrades output quality on some models.** Base models only, and the server says so when you turn it on |
| `QWEN_KAI_NCHUNK` **(ARM only)** | 384 | sub-tiles the KleidiAI GEMM's n dimension so the second height pass finds the packed RHS in cache. `=0` restores one kernel call per slice |
| `QWEN_KAI_OPS` **(ARM only)** | all families on | comma list restricting which KleidiAI families may be used; empty means every one |
| `QWEN_KAI_REPEAT` **(ARM only)** | off | `=1` times a second identical call — a microbenchmark, not a serving flag |

## 4. Server, admission and first audio

| flag | default | effect |
|---|---|---|
| `QWEN_PREFIX_CACHE` | **on** | reuses the request-independent prompt head across requests; `=0` disables it |
| `QWEN_POOL_SPIN` | 65536 on Linux/arm64, 4096 elsewhere | generations a pool worker re-reads before parking on the condvar. On a 16-core Arm host 4096 cost 40% of the Code Predictor: 65536 measured CP 16.0 → 9.6 ms/frame and 491,320 → 35,132 context switches. `=0` parks immediately. **The two defaults are both right.** On an 8-core x86 host the Arm value is worse (C=4 p95 358 vs 339 ms) and so is 0 (+13% stream RTF at C=1, 31k vs 2k context switches): a spinning worker needs a core to spin on, and on 8 cores it is stealing from the worker that has work. Pin the measured value per box; do not port this one |
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
| `QWEN_DECODER_BATCH` | **on in the server** (`[serve]` says so), off in the CLI | one pass over the decoder weights for every active slot. `=0` opts out. It pays only where a worker actually holds several slots: on an 8-core host split `2x4`, C=4 gives each worker ~1.4 active slots, the gang never exceeds 2, and turning it **off** measured 12% better at C=4 TTFA p95 with identical RTF. Read `decoder batch: calls / mean` before believing either direction |
| `QWEN_SERVER_NO_DECODER_BATCH` | unset | present = the server does not turn the above on for you |
| `QWEN_DECODER_THREAD` | off | runs the decoder on its own thread beside the Talker |
| `QWEN_SD_POOL` | server: `engine` | `engine` runs decoder tiles on the engine pool (inline when already inside a region); `private` keeps the decoder's own worker team. Legacy aliases `qwen`, `q`, `1` (engine) and `0` (private) remain accepted; unknown values fail fast |
| `QWEN_BLAS_OWN` | server: `1` | `1` holds OpenBLAS at one thread and partitions the decoder SGEMMs across the engine pool (exact sub-problems, output bit-identical); `0` lets OpenBLAS run its own team |
| `QWEN_CP_REGION` | on (x86 VNNI **and AMX**, int8 CP) | runs each batched code-predictor transformer step as ONE persistent parallel region with spin barriers between phases instead of 20 pool dispatches; per-slot sections run one slot per thread; outputs bit-identical; `0` restores the dispatched path. The in-region runner follows the same gate table the dispatcher uses, so an AMX host runs AMX tiles in-region at `B>=4` and VNNI row blocks below that |
| `QWEN_TK_REGION` | on (x86 VNNI, int8 Talker) | same design for the batched Talker step: one pool entry per step (28 layers, projections as VNNI row blocks, per-slot sections one slot per thread) instead of 112 dispatches; outputs bit-identical; `0` restores the dispatched path. Off automatically on every other ISA, with int4/bf16 weights and on the GCD pool |
| `QWEN_CP_FRAME_REGION` | on (same build/shape conditions as `QWEN_CP_REGION`) | runs the WHOLE 16-step code-predictor frame — every MTP projection, every transformer step and every lm_head argmax — inside ONE pool entry instead of 47. The embedding row of each step is either `code0` or an argmax this frame produced, so the sequence is decidable before entering; kernels, quantiser and argmax order are unchanged and the codes are bit-identical. `0` restores the per-call path |
| `QWEN_CP_BATCH_HEAD` | on (x86 VNNI **and AMX**, int8 CP heads) | at concurrency >= 2 runs the MTP projection and each lm_head once for all active slots as one int8 matmat (same quantiser, exact int32 dots, codes bit-identical) instead of one GEMV per slot; `0` restores the per-slot path; int4/bf16 heads and other ISAs keep the per-slot path automatically |
| `QWEN_SD_SCRATCH_STATS` | off | diagnostic: when a stream ends, prints its decoder scratch arena (blocks, bytes, peak per chunk, spills) |
| `QWEN_SD_THREADS` | = `-j` | thread count of the decoder's tile jobs (int8 conv, int8 GEMM, snake), whichever pool runs them |
| `QWEN_PREFILL_LOW_MS` | 0 | with `QWEN_PREFILL_HELPER=1`: for this many ms each prefill submits to the pool at LOW priority, taking only the windows the frame loop leaves free (trade-off knob: STREAM −3%, TTFA +100/+250 ms on c8a C4) |
| `QWEN_POOL_HI_WINDOW_US` | 200 | a LOW submitter waits while an ordinary one dispatched within this window |
| `QWEN_SD_SGEMM_CENSUS` | off | diagnostic: prints every decoder SGEMM shape with its wall time (every 100 calls and at exit) |
| `QWEN_STREAM_DECODE_CHUNK` | 8 (max 32) | frames decoded per streaming chunk |
| `QWEN_STREAM_DECODE_CHUNK_BUSY` | 0 (off) | a different chunk size once more than one slot is busy |
| `QWEN_DECODER_GANG_LEAD` | 4 | slots from which the decoder gang gets a leader |
| `QWEN_DECODER_GANG_MIN` | 2 | smallest gang that is worth forming |
| `QWEN_SD_INT8` | on where the build has AVX-512 VNNI, off elsewhere | int8 speech-decoder convolutions; `=0` forces fp32. Kernels exist for VNNI and ARM dotprod only; on ARM it is opt-in (`=1`) until the first-frame cost is measured there |
| `QWEN_SD_AMX` | off | decoder INT8 AMX control path (V1); requires the AMX INT8 build/capability and `QWEN_SD_INT8=1`, otherwise the decoder falls back to its selected INT8/FP32 path |
| `QWEN_SD_AMX_D` | off | experimental decoder INT8 Design D; prebuilds immutable AMX B weight tiles at model load and loads the quantised im2col panel directly as AMX A. It supports the real M=96/192/384/768 decoder shapes, reports persistent pack bytes and falls back to V1/INT8 if a shape is unsupported. FAST GCP tests now prove the path under prefork and actual continuous ragged batching; full streaming qualification and production-default selection remain pending |
| `QWEN_SD_AMX_BF16` | off | experimental decoder AMX BF16 arm; uses real `TDPBF16PS` with activation-as-A and immutable BF16 B packs on the same decoder shapes. It can run with `QWEN_SD_INT8=0`, reports conversion/persistent-pack/tile counters, and is wired into the continuous ragged-batch path. FAST GCP evidence proves execution and parity, but its current C4 STREAM_RTF is behind Design D, so it is not a production default |
| `QWEN_SD_STREAM_STRIP` | off | experimental SQ-1 decoder slice; on warm streaming causal convolutions with Design D, evaluates only the newly produced output columns from the existing causal tail instead of computing and discarding the left-context columns. Requires `QWEN_SD_AMX_D=1`; falls back to the unchanged full-window path otherwise |
| `QWEN_SD_DIRECT_CONVT` | off | experimental SQ-2 decoder preparation path; streaming and ragged transposed convolutions accumulate each existing per-tap GEMM directly into the useful output range and request-local carry instead of materializing/copying a full overlap buffer. It preserves the control path and is independently disable-able; allocation failure falls back to the existing implementation |
| `QWEN_SD_DIRECT_DWCONV` | off | experimental SQ-2b ragged decoder path; keeps depthwise ConvNeXt input/output in the global ragged workset and removes the per-item temporary copy-in/copy-out. It preserves the existing arithmetic and tail update order, is independently disable-able, and falls back to the existing implementation if the direct helper rejects the shape |
| `QWEN_SD_DIRECT_INPUT` | off | experimental SQ-2c warm streaming path; prepares Design-D INT8 panels directly from the request-local causal tail plus new input instead of materializing a concatenated fp32 `[tail | input]` buffer. It preserves the existing range path/fallback and updates the tail with the established helper after successful execution |
| `QWEN_SD_DIRECT_QUANT` | off | experimental SQ-2e ragged INT8 path; gathers one convolution row into a worker-local buffer and quantises it immediately, avoiding the large temporary fp32 `[N][K]` panel on successful AMX execution. If AMX rejects, the panel is reconstructed for the established BLAS fallback |
| `QWEN_SD_FUSED_RESIDUAL` | off | experimental SQ-2d decoder path; for same-width 1x1 residual projections, adds the residual in the Design-D AMX epilogue while storing the projection result, avoiding the separate full-size residual-add pass in both per-slot and ragged worksets. The output buffer remains separate so unsupported paths retain the existing fallback |
| `QWEN_SD_INT8_BLK` | compiled default | block size of the int8 decoder convolution tiles |
| `QWEN_SD_CONV_NC` | all | auto | output columns per work item in the INT8 decoder conv. Auto sizes the panel from the layer length and the pool so a short layer still fills it; `=128` restores the old fixed panel (the A/B arm). Each column is im2col'd, quantised and scaled independently, so the panel size changes only WHO computes a column, never its value |
| `QWEN_SD_RAG_MIN_PANELS` | 8 | minimum ragged decoder panel count that submits to the engine pool; experimental A/B control only, with worker/kernel/fallback behavior unchanged |
| `QWEN_SD_WINDOWED` | off | windowed decoder evaluation; diagnostic for the streaming boundary |
| `QWEN_DEC_FIRSTCHUNK_GROUP` | 0 (off) | `=1` groups the first streaming chunk of several slots into one decoder pass |
| `QWEN_THREADS_TALKER` · `QWEN_THREADS_DECODER` | unset (both = `-j`) | split the thread budget between the Talker/CP phase and the decoder phase inside one worker |
| `QWEN_NO_SIN_POLY` | unset | `=1` drops the polynomial sine used by the snake activation back to `sinf`; the polynomial is only used where the argument is in range |

## 6. Precision and voice

| flag | default | effect |
|---|---|---|
| `QWEN_CP_PREC` | follows `--int8` / `--int4` | `int8` or `int4` for the Code Predictor alone — the lever behind the mixed-precision configurations |
| `QWEN_CP_LAYER_PREC` · `QWEN_CP_LMHEAD_PREC` | follow `QWEN_CP_PREC` | the same choice for the CP's layers and its lm_heads separately |
| `QWEN_CP_PREFILL2` | **on** for an AVX-512-VNNI build, off elsewhere | runs the CP's first pass two positions at a time so it reaches a batched kernel. `=0` opts out, and the kernel audit then shows the CP's batched rows disappearing entirely |
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

Used for **attribution** rather than timing, two of them answer questions the wall clock cannot:

- `QWEN_BATCH_STATS=1` prints `[batch-audit]`, which names the kernel that did each batched
  projection and splits it by Talker / Code Predictor / speech decoder. This is how you prove a
  flag did what it claims: with `QWEN_PREFILL_MATMAT=1` on an AMX host the Talker's 39.5 GMAC
  sit on `bf16 AMX tiles`, and adding `QWEN_NO_AMX=1` moves the *same* GMAC to
  `bf16 AVX-512 dpbf16`. A number that does not move under its own control was not measuring
  what you thought.
- `QWEN_SERVE_PROFILE=1` prints `[serve-profile]`: per-stage milliseconds, the mean number of
  active slots, and — the line to read before trusting anything about batching —
  `decoder batch: calls / mean` with `max slots`. A mean of 1.00 means the batch never formed,
  whatever the flag says.

### Which kernel actually ran, and whether a fix can reach it

`[batch-audit]` names the kernel per component, and one line under it answers a question the MAC
table cannot:

```
fallback twin dispatch: bf16 222 fixed-width / 0 generic  ·  int8 96 fixed-width / 0 generic
fallback twin dispatch: never reached (a wider matmat took every batched call on this build)
```

The twins are what run when no VNNI, AMX, SDOT, SMMLA or KleidiAI matmat takes the call, and
the counters record which arm of their fixed-width switch was used — including "never reached",
which is the honest answer on a build whose dispatcher sends everything to a matrix unit. It
exists because a kernel improvement is worth exactly what the dispatcher lets it be worth: the
fixed-width kernels added for bf16 B=9..15 and int8 B=1/5/7/9..15 are a 5-10x improvement on
builds that reach them, and **zero** on an AMX or AVX-512 host, where these counters stay at
zero on real prefill shapes, while at `SIMD=portable`, where `--caps` shows
`bf16 -> fixed-B twin`, the same widths move 11.87 -> 2.25 ms (B=9) and 10.03 -> 1.32 ms (int8
B=1). Measure the dispatch before claiming the speedup.

### What the flags RESOLVED to, not what was typed

`[FLAGS]` is the raw environment: a variable nobody set is absent, a compiled default is
invisible, and a predicate that decides a whole path can live outside the gate table. That is
how an AVX-512-BF16 host ran the Talker prefill on the f32/SGEMM fallback for weeks (~400 ms of
TTFA at C=1) while every `[FLAGS]` line looked right. `./qwen_tts --dispatch-map` prints the
other side, per logical feature: **compiled · supported on this CPU · env · resolved · reason**,
where *resolved* is obtained by calling the runtime predicate itself, and a second table with
every `g_mm_gate[]` row as `qwen_mm_use()` answers it now. When `tools/cpu_check.sh` receives a
profile, it runs this map under the profile's actual environment; an explicit
`QWEN_PREFILL_MATMAT=1` with no native BF16 unit is a hard dispatch-gate finding instead of a
warning that can survive into a long suite:

```
[DISPATCH] v=1 pid=… isa_class=x86_avx512bf16 build=1496938 simd=avx512bf16
  feature                      compiled supported env                       resolved reason
  talker.prefill.matmat_bf16   yes      yes       QWEN_PREFILL_MATMAT=unset ON       avx512_bf16_matmat_available (VDPBF16PS)
  talker.prefill.f32_blas_fallback yes  yes       -                         OFF      not taken: bf16 matmat selected
  prepack.vnni                 yes      yes       QWEN_VNNI_PREPACK=unset   OFF      opt-in; int8 only; REJECTED …
[DISPATCH-GATE] v=1 rows=13 …
  gate.int8.vnni   int8 VNNI vpdpbusd   yes  yes  ON   2(2) …  QWEN_NO_VNNI   default ON
```

The Arm rows are there too — `talker.prefill.matmat_bf16` via `arm_bf16_matmat_available`
(BFMMLA), `matvec.int8.sdot`, the opt-in `matvec.bf16.bfdot` (`QWEN_ARM_BFDOT`), `q8repack.neon`,
every `kleidi.*` knob (`QWEN_NO_KLEIDI`, `QWEN_NO_KAI_I8`, `QWEN_NO_KAI_BF16`, `QWEN_KAI_OPS`,
`QWEN_KAI_QKV_FUSED`, `QWEN_KAI_LHS`, `QWEN_KAI_NCHUNK`), the Apple `apple_off` gate rows with
`QWEN_APPLE_MMLA`, and `QWEN_POOL_SPIN` with its Linux/aarch64 default of 65536 — so the §9
"what does not port" list can be read off the machine instead of remembered.

`QWEN_DISPATCH_JSON=path` writes the same rows as JSON; `tools/dispatch_gate.py` compares them
with `tools/dispatch_expect.json` for the host's `isa_class` and prints `SUSPICIOUS` for a
feature that is expected ON, compiled, supported and still OFF — the automatic detector for
that class of bug. `make cpu-check` runs all of it (see [cpu-profiling.md](cpu-profiling.md)).
`QWEN_DISPATCH_MAP=1` (or any of `QWEN_SERVE_PROFILE` / `QWEN_SHAPE_CENSUS`) makes the server
print the table in its own banner, so the engagement proof sits inside the timed run's log.
Every SIGUSR1 counter dump is now bracketed by `[DUMP] v=1 pid=… seq=N … begin` / `end`, so a
harness that signals before and after a cell can separate the two.

### Is a flag even declarable?

`[FLAGS]` can only report what `g_qwen_reported_flags[]` lists, so a flag the engine reads but
never declares is one a deployment cannot audit — `check-flags` will happily pass while the
process runs a configuration nobody asked for. `make check-flag-registry` compares the two sets
in both directions and fails if they differ; it runs inside `make test-all`.

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

### VNNI parent prepack — candidate, not a default

`QWEN_VNNI_PREPACK=1` (also `all`) builds the eligible INT8 weight layout before prefork;
`cp` and `talker` restrict that work to one component. The packed layout is reused by the
batched VNNI matmat path for `B=2..8`; GEMV is unchanged. This is deliberately different from
the ARM `QWEN_KAI_NCHUNK` knob: it changes the weight representation, not the GEMM's n tile.

On the 16-core VNNI reference host, the corrected `cp` experiment prepacked 41 CP matrices
(about 102 MB) and produced byte-identical audio. It did not improve the measured server
objective: C=4 TTFA p95 was 318 ms without it and 319 ms with it, while total RTF moved from
1.10 to 1.13. The profile therefore leaves the variable null. Use the flag to qualify a new
host or workload, and record the parent prepack count before interpreting the result.

---

## 9. What applies on which ISA, and what does not port

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

**x86 only** — `QWEN_NO_VNNI`, `QWEN_NO_VNNI_TILE`, `QWEN_NO_AMX`, `QWEN_NO_AVX2MM`,
`QWEN_NO_BF16DOT`, `QWEN_NO_BF16_MATMUL`, `QWEN_SD_INT8` (on by default only where AVX-512 VNNI
exists), the AMX/VNNI/AVX2 batch thresholds, and the `*_NCHUNK` row-chunk family.

**The two sides are not symmetric, and the asymmetry is the point.** `QWEN_KAI_NCHUNK` is on by
default at 384 because sub-tiling the KleidiAI GEMM was measured to win on ARM. The x86
`*_NCHUNK` knobs exist but default to **off**: the kernels they tile are entered through the gate
table's batch/rows/cols thresholds, and no x86 host has yet shown the second-pass cache miss that
makes chunking pay. Treat them as instrumentation for a new box — measure, and only then pin a
value in that box's profile. A value copied across architectures means nothing: the ARM number is
an n-dimension tile inside a packed GEMM, the x86 ones are output-row chunks in a different
kernel, and neither reads the other's units.

---

## See also

- [`serving-operations.md`](serving-operations.md) — how to run and measure the server
- [`reference-arm-16c.md`](reference-arm-16c.md) — one box with all of this applied, measured
- [`configs/perf/README.md`](../configs/perf/README.md) — the profile format that pins these values
- [`performance.md`](performance.md) — what the numbers mean once they are taken
