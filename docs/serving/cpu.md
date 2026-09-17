# The CPU streaming server — the production path

_[Serving index](README.md) · [api](api.md) · **CPU server** · [operations](cpu-operations.md) · [batching](cpu-batching.md) · [boxes](boxes.md)_

This is the backend to deploy. It is the one with a break-in procedure, a profile format that
refuses to be optional, reference hosts measured end to end, and 30-minute strict-KPI soaks
behind its numbers. The [GPU server](gpu-cuda.md) is faster on the hardware it runs on and is
**not qualified**; the two are documented apart on purpose.

The HTTP API — endpoints, request bodies, the streaming contract, the error envelope — is
[`api.md`](api.md) and is identical on both backends. This page is about **how to run the
process so it performs**, which is a different question and the one that goes wrong quietly.

| | |
|---|---|
| **status** | production; qualified on named hosts, see [`boxes.md`](boxes.md) |
| **what it holds** | 0.6B **C12–C16** and 1.7B **C10–C16** on a 32-core Arm host; 1.7B **C11** on a 32-core Zen5 — measured per box on the v2 architecture, never extrapolated. Smaller hosts have **no current v2 capacity number**: see [`boxes.md`](boxes.md) |
| **the one hard rule** | the flags are not decoration. A server started without them does not error. It is simply slower, and nothing in the output says so |

---

## The short version

Four commands, in this order. Each one exists because skipping it has cost somebody a day.

```bash
make blas                                        # 1. build (Linux/x86 picks the ISA itself)
make doctor                                      # 2. what IS this box, and what should it run
eval "$(tools/perf_profile.py command recommended --model qwen3-tts-0.6b --port 8080)"
                                                 # 3. launch with the whole environment
make bench-suite BENCH_PROFILE=recommended       # 4. and prove it on YOUR box
```

Step 3 is the part people replace with a hand-typed command line, and it is the part that
decides most of the performance. Read on for why.

---

## 1. `make doctor` — always first, on every new box

**Under a minute, no model, no download.** It is the first command to run on a machine you have
never served from, and it is cheap enough that there is no reason not to.

```bash
make doctor                            # ~10-40 s, reuses cached roofs
make doctor DOCTOR_ARGS="--full"       # + --self-test and the quick --matmat-tune grid
```

What it answers, in eight numbered blocks:

| block | question |
|---|---|
| 1. MACHINE | cores, SMT, sockets, NUMA, LLC per core, ISA — **and the gates**: SMT off, governor, cgroup quota. A gate that says `FAIL` means every number after it describes a different machine |
| 2. BANDWIDTH | the read roof per CPU mask, swept by thread count, plus the **engine's own GEMV roof** — the 1.7B Talker frame streamed from DRAM. This is the number that bounds Talker and Code Predictor at batch 1 |
| 3. BINARY + DISPATCH | what this binary compiled, what the host supports, and which kernel actually resolves per batch width. A silent fallback shows up here and nowhere else |
| 4. SHAPES | `--matmat-bench` without a model: batched matmat against B × GEMV. A speedup below 1 means batching does not pay at that width on this silicon |
| 5. COST MODEL | predicted frame time and `rho` per topology and batch |
| 6. **RECOMMENDATION** | a predicted `W x K`, a batch cap, **the argv and the full environment with a `why` per line**, what must stay off, what to A/B — and a draft profile on disk |
| 7. VERIFY NEXT | the exact commands that replace each `[PREDICTED]` with a `[MEASURED]` |
| 8. CEILING | how many streams this host can hold at most, and whether the wall is bandwidth or the decoder |

**Every number carries a label** — `[MEASURED]`, `[CACHED]`, `[TRANSFERRED]`, `[PREDICTED]`,
`[UNKNOWN]` — and the labels are the point. The doctor predicts a topology; it does not qualify
one. Its draft profile lands at `profiles/doctor/<utc>_<host>/` marked `unqualified`, and copying
it into `configs/perf/` before running the wave is exactly the mistake the label exists to stop.

Run the grid it recommends without retyping anything:

```bash
make doctor-wave                       # runs profiles/doctor/LATEST/wave-plan.json in order
```

On a 32-core Arm host the doctor also runs a simultaneous `1x8 / 2x8 / 4x8` GEMV scaling
preflight near the top of its report. A poor `4x8` verdict means test `2x16`, then `1x32`,
*before* paying for a wave or a soak — see [Arm topology preflight](../arm-topology-preflight.md).

### Turn SMT off first

The engine orders the CPUs **core-major** (each core's threads adjacent) and slices that order, so
siblings stay inside one worker and two workers never share a physical core. That is correct, and
it is also why a slice does not mean what its width suggests: with SMT on and `per = 2`, a worker's
slice is **one physical core with both its threads**, not two cores. A topology sweep then compares
configurations that are not what their names say. The Arm instance families report
`Thread(s) per core: 1` on their own; the x86 ones usually do not.

```bash
lscpu -e                                       # which id is a sibling of which core
echo off | sudo tee /sys/devices/system/cpu/smt/control    # where you own the box
# on GCP: recreate the VM with --threads-per-core=1
```

If you genuinely cannot turn it off, halve the thread budget deliberately (`-j` = physical cores)
and say so beside every number.

---

## 2. Build for the ISA you actually have

`make blas` on Linux/x86 runs `SIMD=auto`: it reads `/proc/cpuinfo`, probes the compiler and picks
the highest level both support. It announces itself (`[simd] auto -> …`), and **the resulting
binary is not portable to an older CPU.**

| host | build | notes |
|---|---|---|
| Arm (Graviton3/4, Axion, Ampere) | `make blas` | KleidiAI compiles in automatically when the compiler defines `__ARM_FEATURE_MATMUL_INT8` |
| x86 with AMX (Sapphire/Emerald/Granite Rapids) | `make blas SIMD=amx` | or let `auto` find it |
| x86 Zen4/Zen5 | `make blas SIMD=avx512bf16` | VNNI **plus** native bf16 prefill; `avx512vnni` cannot compile the bf16 prefill and costs ~600 ms per admission on those CPUs |
| x86 AVX-512 without VNNI | `make blas SIMD=avx512` | no native integer GEMV, no decoder INT8 — see the portability note below |
| x86 AVX2 only | `make blas SIMD=portable` | despite the name this is the AVX2 baseline, not scalar |
| Apple Silicon | `make blas` | Accelerate; development and correctness, not a serving target |

Then prove the binary does what its name says:

```bash
make cpu-check                # provenance + hardware + self-test + RESOLVED dispatch map
./qwen_tts --caps             # what compiled, what the host supports, per batch width
./qwen_tts --self-test        # the cross-ISA correctness oracle
./qwen_tts --dispatch-map     # which kernel each stage actually selects
```

**The portability cliffs are specific, not "everything falls to scalar".** AVX2 and AVX-512
without VNNI have no native integer GEMV and no decoder INT8; dotprod-only Arm has a strong GEMV
but no default SDOT matmat. Those are serving-level differences, not micro-optimisations —
detail in [`.work/legacy-cpu-v2-audit-20260916.md`](../../.work/legacy-cpu-v2-audit-20260916.md).

---

## 3. Launch — one command, and what it expands to

**The flags are the deployment.** There are roughly forty of them on a product profile, they
differ per ISA, several do not port between machines, and a wrong or missing one does not error —
it just gives back performance silently. Nobody is going to remember them, which is why they are
not meant to be remembered: they live in `configs/perf/*.json`, and the engine emits its own
invocation from the profile.

```bash
# the correct way to start a server, on every box
eval "$(tools/perf_profile.py command <profile> --model MODEL_DIR --port 8080)"
```

and afterwards, the loop is closed by the process itself rather than by intent:

```bash
tools/perf_profile.py check-flags <profile> --log server.log
```

The engine prints one machine-readable `[FLAGS]` line naming every registered variable it
actually read; `check-flags` compares that against what the profile asked for. **A flag is on when
the process says so, never when the invocation intended it.**

### What that expands to on Arm

`tools/perf_profile.py command arm-product` — a Neoverse-V2 / KleidiAI host:

```bash
OPENBLAS_THREAD_TIMEOUT=1 QWEN_ARM_BFDOT=0 QWEN_BFMMLA_MIN_B=2 QWEN_BLAS_OWN=1 \
QWEN_CP_PREC=int8 QWEN_CP_PREFILL2=1 QWEN_DECODER_BATCH=1 QWEN_INT8_SDOT_MIN_B=2 \
QWEN_INT8_SDOT_MM=0 QWEN_KAI_LHS=asym QWEN_KAI_NCHUNK=384 QWEN_KAI_QKV_FUSED=1 \
QWEN_KLEIDI_MIN_B=1 QWEN_NO_BFMMLA=0 QWEN_NO_KAI_BF16=0 QWEN_NO_KAI_I8=0 \
QWEN_NO_KLEIDI=0 QWEN_NO_Q8REPACK=0 QWEN_NO_SDOT=0 QWEN_NO_SMMLA=0 \
QWEN_POOL_SPIN=65536 QWEN_PREFILL_MATMAT=1 QWEN_PREFIX_CACHE=1 QWEN_Q4_NAIVE=0 \
QWEN_SD_INT8=1 QWEN_SD_INT8_BLK=64 QWEN_SD_RES1_V2=1 QWEN_SD_MULTISLOT=0 \
QWEN_SD_POOL=engine QWEN_SD_RAG_MIN_PANELS=2 QWEN_SD_STREAM_STRIP=0 QWEN_SD_WINDOWED=0 \
QWEN_SERVER_ASYNC_OUTPUT=0 QWEN_SMMLA_MIN_B=2 QWEN_STREAM_DECODE_CHUNK=4 \
QWEN_STREAM_DECODE_CHUNK_BUSY=0 QWEN_TTS_STREAM_LAYOUT=1 \
  ./qwen_tts -d MODEL_DIR --int8 --serve 8080 --batch-size 2 \
             --max-queue 0 --queue-timeout-ms 0
```

A 32-core Graviton4 or Axion deployment (`aws-c8g-8xlarge-32c-arm-v2-all-on`,
`axion-c4a-highcpu32-0p6b-all-on`) adds the elastic decoder lane and a wider shape:

```bash
… QWEN_SD_BF16_PREUP=1 QWEN_SD_LANE_SPLIT=4 QWEN_SD_LANE_ELASTIC=1 … \
  ./qwen_tts -d MODEL_DIR --int8 --serve 8080 --batch-size 8 \
             --prefork 4 --prefork-threads 8 \
             --max-queue 0 --queue-timeout-ms 0 --max-request-seconds 60
```

### What that expands to on x86

`tools/perf_profile.py command turin-c8a-32c-vnni-product` — a 32-core Zen5, VNNI + AVX-512 BF16,
one worker per CCX:

```bash
OPENBLAS_THREAD_TIMEOUT=1 QWEN_BLAS_OWN=1 QWEN_CP_PREC=int8 QWEN_CP_PREFILL2=1 \
QWEN_DECODER_BATCH=0 QWEN_NO_AMX=1 QWEN_NO_AMX_BF16=1 QWEN_NO_AMX_INT8=1 \
QWEN_NO_AMX_Q4=1 QWEN_NO_BF16DOT=0 QWEN_NO_BF16_MATMUL=0 QWEN_NO_KLEIDI=1 \
QWEN_NO_VNNI=0 QWEN_NO_VNNI_ACT_QUANT=0 QWEN_NO_VNNI_QKV=0 QWEN_NO_VNNI_ROWSUM=0 \
QWEN_NO_VNNI_TILE=0 QWEN_NO_X86_QKV=0 QWEN_POOL_SPIN=65536 QWEN_PREFILL_MATMAT=1 \
QWEN_PREFIX_CACHE=1 QWEN_Q4_NAIVE=0 QWEN_SD_INT8=1 QWEN_SD_INT8_BLK=256 \
QWEN_SD_LANE_SPLIT=4 QWEN_SD_LANE_ELASTIC=1 QWEN_SD_RES1_V2=1 QWEN_SD_MULTISLOT=2 \
QWEN_SD_POOL=engine QWEN_SD_RAG_MIN_PANELS=2 QWEN_SD_STREAM_STRIP=0 \
QWEN_SD_WINDOWED=0 QWEN_SERVER_ASYNC_OUTPUT=0 QWEN_STREAM_DECODE_CHUNK=4 \
QWEN_STREAM_DECODE_CHUNK_BUSY=0 QWEN_TTS_STREAM_LAYOUT=1 QWEN_VNNI_GEMV_MR=2 \
QWEN_VNNI_MIN_B=2 \
  ./qwen_tts -d MODEL_DIR --int8 --serve 8080 --batch-size 4 \
             --prefork 4 --prefork-threads 8 \
             --max-queue 0 --queue-timeout-ms 0 --max-request-seconds 60
```

On an 8-core Intel AMX host (`x86-8c-amx-recommended`) the shape is `2x4`, `QWEN_POOL_SPIN` is
**4096** rather than 65536, `QWEN_SD_AMX_D=1` turns on the Design-D decoder and
`QWEN_SD_FUSED_RESIDUAL=1` the fused residual.

**Do not copy either block to a different machine.** Two values in particular are measured
properties of the host and travel badly:

- **`QWEN_POOL_SPIN`** — 65536 on the 16-core Arm host, where the 4096 default cost 40% of the
  Code Predictor (16.0 → 9.6 ms/frame, 491,320 → 35,132 context switches). On 8 x86 cores 65536
  measured *worse* and 4096 is right. "Explicitly disable it" would have cost 13% of stream RTF.
- **`QWEN_DECODER_BATCH`** — pays only when a worker really holds several slots. The Turin
  product profile pins `0`; the Arm profiles pin `1`.

`QWEN_KAI_*` exists only where KleidiAI compiled in and is inert on x86; the x86 side has the AMX
and VNNI gates instead.

### The parts of the command line, and what each one decides

| part | decides | leave it out and |
|---|---|---|
| `--int8` | Talker + CP weights quantized at load | the box reads bf16 weights — the single biggest regression available on a bandwidth-bound machine |
| `--prefork W` | W worker processes, each pinned to a contiguous slice of cores, weights shared copy-on-write | one process shares one pool across every request |
| `--prefork-threads K` | pool size **inside** that slice | the pool is sized from the machine rather than the slice, and threads cross the pinning |
| `--batch-size N` | per-worker **in-flight cap** *and* the scheduler: at `1` a worker serves one request at a time, from `2` up it runs the continuous-batching scheduler | **the default is 1** — concurrency turns into queueing and the engine never reaches the GEMM path |
| `--max-queue` | who is refused instead of held open. With prefork, `0` keeps the parent accepting and returns an immediate `503` | unbounded waiting: the caller sees latency instead of a refusal, which is the worse failure |
| `--max-request-seconds` | the generation cap per request, **and the input text limit derived from it** | one pathological text holds a slot for minutes |
| `OPENBLAS_THREAD_TIMEOUT=1` | OpenBLAS parks instead of spinning | two pools fight for the same cores: first audio 108 ms against 66 ms at C=1, *bimodally* — and 42,500 context switches per second against 12,000 |
| `OPENBLAS_NUM_THREADS` | must be **absent**, not merely unset | the engine backs off sizing BLAS entirely when it is already set, so a stray `export` in somebody's shell silently replaces the qualified thread split |

That last row is why a profile can declare a variable `null`, and why
`tools/perf_profile.py forbidden-env <profile>` exists and the benchmark suite refuses to run when
one of them is present.

### Why the environment is worth applying — and where it is not

Measured on a 16-core Arm host, current build, `2x8`, four waves, the only difference between the
arms being whether the profile environment was applied:

| arm | C | TTFA p50 | stream RTF p50 | context switches/s |
|---|---:|---:|---:|---:|
| with the profile | 1 | 57 ms | 0.44 | 11,910 |
| compiled defaults | 1 | 51 ms | 0.45 | 7,880 |
| with the profile | 4 | **109 ms** | **0.72** | 30,153 |
| compiled defaults | 4 | 133 ms | 0.81 | 68,984 |

At one request the two are within noise, because most of that set has **become the compiled
default** since it was found — the native prefill matmat, the prompt-prefix cache, the pool spin
and the batched decoder are all on without anyone asking. At four requests the profiled arm is
18% ahead on first audio and 11% on realtime factor, and the last column names the cause.

So: **a profile earns its keep where the machine is loaded.** It is pinned anyway at C=1, so that
a change of default elsewhere cannot move a qualified deployment quietly.

---

## 4. Qualify the box — and judge the whole envelope, not RTF

```bash
make bench-fingerprint                                   # the machine describes itself; gates on SMT
make bench-topo  BENCH_MODEL=<dir> BENCH_TOPO=1x16,2x8,4x4 BENCH_CONC=1,4
make bench-suite BENCH_MODEL=<dir> BENCH_PROFILE=<name> BENCH_TOPO=2x8
make bench-soak  SOAK_MODEL=<dir> SOAK_PROFILE=<name> SOAK_CONCURRENCY=8 SOAK_MINUTES=30
```

`SUITE PASSED` is the only line that means the suite ran. The full procedure — what each rung
answers, the three different arrival models, the idle gate, the manifest — is
[`cpu-operations.md`](cpu-operations.md) §2 and §4.

**A concurrency is acceptable on the whole envelope or not at all.** RTF is necessary and nowhere
near sufficient:

| metric | requirement | why it is in the list |
|---|---|---|
| `stall_rate@B` | ideally 0%, at **every** buffer, not only @1000 ms | the production question: what share of streams play continuously |
| `safe_play_start` | **under 1 s — a hard line** | the earliest moment playback can begin and still finish without an underrun |
| `TTFB` | low | send → status line; what every TTS benchmark quotes |
| `TTFA` | lowish | send → first audio chunk; what a caller hears as responsiveness |
| `max_gap` | small | the mechanism behind the other two: it is what forces the prebuffer |
| `STREAM_RTF` | < 1 | capacity. It is a **mean rate** and does not prove a player never stalls |
| rejects / errors | 0 | a fast server that drops requests is not fast |

**These fail at different concurrencies, and the listening ones fail first.** On an A6000 at C6 the
RTF was 0.90 — comfortably realtime — while the stall rate at 1000 ms was already 35% and
`safe_play_start` p95 was 4.0 s. Quoting "RTF 0.90, fine" there would have been wrong in the way
that matters to a listener.

Then check the audio, which no clock does:

```bash
python3 tools/wav_qc.py <dir>     # clipped runs, holes, step discontinuities, silent files
```

and **listen at C=1 and at C ≥ 2W**, because above batch 2 the engine takes a different kernel and
a defect there is invisible at concurrency 1.

---

## 5. Three mistakes that look like a slow engine

**`--batch-size` defaults to 1, and it is also the per-worker in-flight cap.** With `--prefork 12`
and no `--batch-size`, twelve requests run and the rest wait in the listen backlog — not rejected,
just invisible. The startup line says which it is:

```
prefork: 12 workers x 2 threads, 24 cpus (2 per worker), cap 1 in flight each, port 8080
```

So requests in flight is `W × cap`, and a worker only reaches the GEMM path at batch ≥ 2, which
needs client concurrency `C ≥ 2W`. More workers means narrower batches per worker, which can keep
every worker on the slower kernel at exactly the concurrency you care about.

**More workers is not more throughput past the bandwidth roof.** Talker and Code Predictor are
DRAM-bound weight streams; W workers read the weights W times. On a 16-core Arm host `1x16` won
first audio at C=1 and gave it back at C=4, while `2x8` held both — and on a two-socket box the
weights are loaded *before* the fork and shared copy-on-write, so workers pinned to the second
socket read them across the interconnect. That is worth one `numactl` arm before accepting a
topology.

**Threads do not add up.** Each worker's pool is K threads and the BLAS inside it has its own.
Oversubscription turns latency into context switches — the 40 ms in the table above.

A worked example of all three at once, from a dual-socket 12-physical-core host with SMT on
(24 logical CPUs), serving the 1.7B to 24 concurrent callers:

```bash
./qwen_tts -d qwen3-tts-1.7b --load-voice v.qvoice --icl-only --serve 8000            --prefork 12 --prefork-threads 2
```

Core-major slicing gives each of the twelve workers `per = 24/12 = 2` cpus, which on an SMT-2 host
is **one physical core with both its threads**. So the whole machine is in use — but as twelve
one-core servers. Then `--batch-size` is absent, so the per-worker in-flight cap is 1 and only
twelve of the twenty-four requests are running; the rest sit in the backlog. And no worker ever
sees a batch, because a batch needs two requests in one worker. Three separate ways of leaving
performance on the table, none of which prints an error. The fix is fewer, wider workers with a
real cap — `--prefork 2 --prefork-threads 6 --batch-size 8` on the twelve physical cores — plus
`--int8`, which on a DDR3-class machine is the single biggest lever available.

Ask the parent what actually happened rather than inferring it:

```bash
kill -USR1 <parent pid>
# [prefork-stats] mean_inflight 3.940 dispatched 24 rejected 0 · w0[asg=6 done=6 act=2 B=1.97] ...
```

`mean_inflight` and the per-worker `B` are the measured batch. `client_concurrency` and
`engine_batch` are different fields for a reason, and the second is never deduced from the first.

---

## See also

- [`api.md`](api.md) — the HTTP API: endpoints, request bodies, streaming, the error envelope
- [`cpu-operations.md`](cpu-operations.md) — the operations manual: the full break-in procedure,
  the benchmark suite rung by rung, arrival models, soak analysis, input-length effects
- [`cpu-batching.md`](cpu-batching.md) — how continuous request-batching works inside a worker:
  the scheduler, per-request independence, why streaming composes with batching
- [`boxes.md`](boxes.md) — every box measured for serving, its profile JSON and what it holds
- [`../feature-flags.md`](../feature-flags.md) — every runtime flag, its default per ISA, and the
  measurement that chose it
- [`../../configs/perf/README.md`](../../configs/perf/README.md) — the profile format, and why it
  is a gate rather than a document
- [`gpu-cuda.md`](gpu-cuda.md) — the CUDA server, and why its numbers must not be quoted here
