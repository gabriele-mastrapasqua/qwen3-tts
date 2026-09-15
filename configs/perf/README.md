# Server performance profiles

**One question: given this hardware and this objective, how should the engine be run?**

```
configs/perf/schema.json          the format, architecture-neutral
configs/perf/axion-16c-ttfa.json  recommended low-TTFA config for a 16-core Google Axion host
configs/perf/recommended.json     stable entry point; POINTS at the current recommendation
configs/perf/x86-8c-amx-recommended.json  x86-specific alias for the measured AMX profile
configs/perf/scaleway-16c-vnni-ttfa.json provisional VNNI profile for a 16-core Scaleway host
configs/perf/aws-c8a-16c-vnni-ttfa.json  provisional VNNI profile for an AWS c8a.4xlarge (EPYC 9R45, 16c, no AMX)
configs/perf/gcp-c4-standard-24-vnni-ttfa.json  unqualified VNNI/BF16-prefill starting profile for GCP c4-standard-24 (12 physical / 24 logical Intel cores)
```

The measured Scaleway campaign is summarized in
[`../../docs/reference-scaleway-16c-vnni.md`](../../docs/reference-scaleway-16c-vnni.md).

```bash
tools/perf_profile.py validate                                     # all profiles
tools/perf_profile.py command recommended --model DIR --port 8000  # the exact invocation
tools/perf_profile.py server-env recommended                       # comma form for the harness
tools/perf_profile.py check-flags recommended --log server.log     # what the engine declared
python3 tests/test_perf_profile.py
```

## Operational ISA parity lanes (2026-09-08)

The four `*-product.json`/`common-control.json` profiles are strict serving contracts,
not qualification claims:

| profile | question | decoder contract |
|---|---|---|
| `amx-product` | best current AMX implementation | ragged Design-D INT8; AMX/fused requested |
| `vnni-product` | best current VNNI implementation | per-item INT8 VNNI; Design-D is an explicit valid fallback |
| `vnni-bf16-product` | VNNI lane for CPUs with native AVX-512 BF16 (Zen5, SPR+): same as `vnni-product` plus native bf16 prefill | `vnni-product` pins f32 prefill, which on a BF16 CPU costs ~600 ms per admission (Turin 2026-09-09); QWEN_POOL_SPIN=65536 measured on c8a.8xlarge |
| `turin-c8a-32c-vnni-product` | the measured Turin product point: `vnni-bf16-product` contract + 4 workers x 8 threads (one per CCX), cap 4, decode quantum 4, elastic decoder lane (`QWEN_SD_LANE_SPLIT=4`, `QWEN_SD_LANE_ELASTIC=1`), direct dilated conv (`QWEN_SD_RES1_V2=1`) and verified 2-slot DL-4 cohort (`QWEN_SD_MULTISLOT=2`), fail-fast admission | c8a.8xlarge 2026-09-09/10: the inline engine tops out near C8; with lane + V2 + cohort, the 2026-09-11 three-wave short/mixed FAST smoke improved C8-C16 sustained metrics with 0 errors/rejects. The preflight requires `decoder.lane`, `decoder.res1_v2` and `decoder.multislot` ON; paired audio quality remains required |
| `aws-c8g-8xlarge-32c-arm-v2-all-on` | AWS c8g.8xlarge / Graviton4 32-vCPU Arm lane: KAI INT8/BF16, BF16 pre-up, elastic lane split 4, RES1_V2, **`QWEN_SD_MULTISLOT=0`** (cohort retired 2026-09-15), 4x8 / cap 8 | qualified 2026-09-15 by two 30-minute strict-KPI soaks: 0.6B C12 (STREAM p95 0.836) and 1.7B C10 (0.831), both zero errors and zero 250/500 ms stalls, per-class and resource KPI PASS. `preferred_concurrency` carries both points and the >= 250 ms client prebuffer assumption |
| `axion-c4a-highcpu32-0p6b-all-on` | GCP c4a-highcpu-32 / Axion Neoverse-V2 Arm lane, same shape, **`QWEN_SD_MULTISLOT=0`** (cohort retired 2026-09-15) | qualified 2026-09-15: 1.7B C16 (STREAM p95 0.789) and 0.6B C16 (0.845), both zero errors and zero 250/500 ms stalls. The cohort was retired on a measured 1.20x per-call penalty plus two paired serving screens, not by analogy with the AWS host |
| `turin-c8a-32c-vnni-control` | the same file with `QWEN_SD_RES1_V2=0` and `QWEN_SD_MULTISLOT=0`: the reference arm of the paired RES1_V2/multi-slot quality bank and lane A/Bs | an A/B arm must be a committed, preflighted profile; the strict preflight refuses a shell override that changes a parity value |
| `arm-product` | best current KleidiAI implementation | per-item INT8 DOTPROD; KAI covers generic Talker/CP/Q4 |
| `common-control` | same serving/decoder shape across ISAs | `QWEN_DECODER_BATCH=0`, per-item INT8 |

The product lanes answer “best supported implementation per ISA”; the common lane answers
“same serving architecture, different hardware”. They must never be pooled into one result.
The product profiles explicitly pin decoder, prefill/CP, pool, quantum, backend gates and
relevant threshold/absence settings. Host topology remains `unspecified` until that host is
preflighted and requalified.

For a new box, run `make doctor` after the final target build and before choosing a
campaign topology. It is useful here as a fast, model-free starting point: its measured
identity/roofs and labelled predictions can suggest `W x K`, cap and candidate flags.
Treat the generated draft as a hypothesis only; reconcile it with one of these parity
profiles, then run `tools/serving_profile.py preflight` and the quality gate. Doctor does
not authorize a fallback or turn a prediction into a qualification.

Before a parity-lane run, apply the profile environment and run the in-process gate:

```bash
tools/serving_profile.py preflight amx-product --binary ./qwen_tts \
  --out RUN/profile-preflight.json
```

The gate records the ISA class, source/binary identity, requested decoder batch, resolved
decoder leaf, precision, Talker/CP/prefill/Q4 families, Design-D/fused/KAI status and all
profile flags. It exits non-zero for an invalid requested path or missing capability. The
parallel-wave and SOAK harnesses run this gate automatically for these parity profiles and
embed the resulting JSON in their artifacts. A profile's `VALID FALLBACK` is intentional and
must remain visible; it is not an AMX-equivalent result.

## The separation that makes this worth having

| | contains | example |
|---|---|---|
| **performance profile** | how the server should be configured | `2 workers x 8 threads, --int8, OPENBLAS_THREAD_TIMEOUT=1` |
| **run manifest** | what exactly was measured that once | commit, binary sha, corpus sha, seed, timestamp, host |

A profile carries **no** measurement provenance, so it survives new binaries and new
campaigns. A benchmark result names the profile it used:

```json
{ "performance_profile": "axion-16c-ttfa" }
```

Later this becomes `axion-16c-throughput`, `x86-32c-ttfa`, … under the same schema, each with
its own qualification. Nobody will have to remember *why* Axion was 2x8.

## Rules the loader enforces

- a profile is **complete** or an **alias**, never half of each — an alias that repeats a
  value is a copy that will drift;
- `workers x threads` may not exceed the declared physical cores;
- an environment value may not contain whitespace: `--server-env` splits on **commas**, and a
  space folds every later variable into the previous one's *value*. That defect produced an
  A/B whose two arms both ran with the flag unset;
- a `QWEN_*` variable the engine does not declare is refused: it would be set and never
  verifiable;
- a setting fixed both on the command line and in the profile body is refused.

`objective` states the regime, because `2x8` is a measured property of this machine and not a
universal one. `precision` is per component, because "int8" names completely different paths
in the talker, the code predictor and the speech decoder.

**These are deployment recommendations, not engine defaults.** Nothing here is compiled in;
`recommended.json` does not claim 2x8 is right on every CPU.

### The limits a profile records, and the one that is derived

Four keys in the `server` block are admission and validation rather than throughput, and they are
in the profile for the same reason the topology is: a deployment that has to remember them will
forget them. `max_queue` and `queue_timeout_ms` decide who is answered `503` instead of held open,
`max_request_seconds` caps one request's generation, and `max_text_chars` caps the input.

`max_text_chars` is normally `"unspecified"`, and that does **not** mean unlimited. The server
then derives the limit from `max_request_seconds` and the batch prompt budget, floors it at 200,
caps it at the compiled 8192, and reports what it settled on in `GET /v1/health`. It is written
here anyway, unset, so the shape of the safety envelope is visible in the profile rather than
discoverable only by reading the engine. Give it a number only when a deployment needs a limit
**tighter** than the derived one — a larger one is not honoured beyond the ceiling.

## The profile is a gate, not a document (2026-08-31)

A profile that has to be remembered is a profile that will be forgotten, and the run that
forgets it still prints a table. Measured on the qualification host, same binary, same bank,
same uptime, interleaved arms:

| round | without the profile | with it |
|---|---:|---:|
| 1 | 108 ms | **66 ms** |
| 2 | 66 ms | **66 ms** |
| 3 | 99 ms | **66 ms** |

C=1 on the FAST bank. The bare arm is bimodal — it lands on either value and looks equally
definitive — and the mechanism is visible beside it: 42 500 context switches per second
against 12 000, and 7.9 cores busy against 7.2, because OpenBLAS idles by spinning and
contends with the engine's own pool. Nothing in the output said which configuration had
produced the number.

So three things now enforce it, and each exits non-zero:

- **`tests/serve_parallel_wave.py` refuses to start** without `--profile <name>` or an
  explicit `--no-profile '<reason>'`. Values passed with `--server-env` are merged on top and
  every override is printed as `profile_override=`.
- **`tools/perf_profile.py forbidden-env`** lists the variables whose profile value is `null`.
  They must be **absent from the environment**, not merely unset by us: the engine sizes
  OpenBLAS to the thread budget at startup with `openblas_set_num_threads()`, and
  `qwen_blas_set_threads()` returns immediately when
  `OPENBLAS_NUM_THREADS` is present, so somebody else's export silently replaces the
  qualified topology.
- **`make bench-suite`** is the single entry point for numbers that leave this repo. It owns
  the profile, checks the environment, stamps the binary, runs every rung, runs the identity
  gates and writes a manifest containing the exact commands.

Individual rungs remain available for investigation. The suite is the gate for a report.

## Which profiles ship, and why there are not more

| profile | status | what it is |
|---|---|---|
| `axion-16c-ttfa` | **qualified** | a 16-core Arm host, measured end to end: topology, thread split, batch width, runtime environment and the concurrency band the claim covers |
| `generic-16c-starting-point` | **unqualified** | a place to START on a 16-core Arm server. Nothing in it was measured on your machine, and it says so in its own `qualification.notes` |
| `x86-8c-amx-single-stream-ttfa` | see the file | an 8-core Intel AMX host, one worker with eight physical-core threads; C=1-2 latency |
| `x86-8c-amx-multiclient-ttfa` | see the file | the same host, two workers x four physical-core threads; balanced C=2-8 operation |
| `x86-8c-amx-tail-latency-ttfa` | see the file | four workers x two threads: the best measured TTFA p95 at C=8 on a saturated box, and worse than the two-worker profile everywhere else |
| `aws-c8a-16c-vnni-ttfa` | **provisional** | an AWS c8a.4xlarge (EPYC 9R45, 16 cores, no AMX). The host on which the AVX-512-BF16 prefill default was measured: pinning `QWEN_PREFILL_MATMAT=1` there moved TTFA p50 from 416 to 124 ms at C=1 and p95 from 939 to 507 ms at C=4 |
| `gcp-c4-standard-24-vnni-ttfa` | **unqualified** | a GCP c4-standard-24 starting profile. It deliberately requires `SIMD=avx512bf16` (VNNI + AVX-512 BF16, no AMX); the previous C4 run used `SIMD=avx512vnni`, which could not compile the native BF16 prefill |
| `x86-8c-amx-recommended` | alias | resolves to the balanced x86 AMX profile; carries no values of its own |
| `recommended` | alias | resolves to the qualified one; carries no values of its own |

### An environment block is an argument, not a list

Every entry in `runtime.environment` carries a `why`, and on the x86 profiles that `why` is the
measurement that chose the value — including the ones set to `null`, which say what was
deliberately left out and what it cost to leave it in. Three of them are worth reading before
copying any profile to a new box:

- **`QWEN_POOL_SPIN` does not port.** The Arm profile pins 65536 because it was measured there;
  the x86 profiles pin 4096, which is the x86 compiled default, because 0, 1024 and 65536 all
  measured worse on 8 cores. "Explicitly disable it" would have cost 13% of stream RTF.
- **`QWEN_DECODER_BATCH` is explicit per profile**, not a portable default: the current
  `amx-product` and `vnni-product` lanes request `1` and the preflight records whether that
  resolves to AMX ragged or the intentional per-item VNNI leaf; `common-control` pins `0`.
  Older qualified x86 profiles may pin `0` for their own measured topology, but must not be
  copied into the Turin campaign without requalification.
- **The `*_NCHUNK` family is `null` with the numbers that say why.** They are experimental, and
  a profile is not the place to park an unproven lever.

A value that appears in a profile without a `why` that cites a measurement on *that* hardware is
a value inherited from somewhere else, and inheriting is how the Arm spin count nearly ended up
on an 8-core Xeon.

There is deliberately no profile per machine type we have ever touched. A profile claims
that a configuration was **measured** on that hardware, and a file that looks like a
qualification while carrying guesses is worse than no file: it is a wrong default that
nobody re-measures, which is the same failure the runtime-environment gate above exists to
prevent.

The way to add one is the break-in sweep in
[`../../docs/serving-operations.md`](../../docs/serving-operations.md) §2, then a profile
with `status: qualified` and the numbers that earned it.

### The one rule that generalises, and the one that does not

**Generalises:** a single request saturates at about eight threads and regresses at sixteen.
That is a property of the work, not of a particular chip.

**Does not:** which `W x K` topology wins. It depends on the concurrency you expect, because
a worker only sees a batch at `C >= 2W` and the engine takes a different kernel above batch
2 — so four workers can win at high concurrency and lose at the concurrency you actually
care about. That one is measured per box, every time.

## Adding a machine

More of these are expected: a profile per deployment that has actually been measured, not per
machine type anyone has touched.

```bash
tools/perf_profile.py new <id> [--like axion-16c-ttfa]   # a skeleton, marked unqualified
$EDITOR configs/perf/<id>.json
tools/perf_profile.py validate
```

`new` starts from an existing profile's STRUCTURE and blanks everything that was measured:
the cpu family, the hardware notes and the whole qualification block. Copying a profile by
hand is how a value from another machine becomes a claim about this one — every field is
filled in, every field looks deliberate, and nothing says which of them anybody measured.
A skeleton starts at `unqualified` and makes filling a field a decision.

Then run the break-in sweep in
[`../../docs/serving-operations.md`](../../docs/serving-operations.md) §2, put what it
measured into the file, and set `qualification.status` to `qualified` with the numbers that
earned it.

### Naming

`<platform>-<cores>c-<objective>` — `axion-16c-ttfa`, `graviton4-32c-throughput`. The
platform is what a reader would recognise, the core count is the shape the topology was
qualified at, and the objective is what the profile optimises, because the same machine has
a different answer for first-audio latency than for throughput.

### The `recommended` alias

`recommended.json` carries a pointer and **no values of its own** — a second copy is a copy
that drifts. It resolves to whichever profile is currently the default recommendation, and
changing that is editing one field:

```json
{ "profile": { "id": "recommended", "alias_of": "<the qualified profile>" } }
```

It must point at a profile whose status is `qualified`. Pointing it at a starting point would
make an unmeasured configuration the default answer to "what should I run", which is the
failure this whole directory exists to prevent.
