# AMD Turin/VNNI capacity campaign — pre-registered plan (2026-09-08)

Task · Prepare the first paid 16-physical-core AVX-512/VNNI campaign without changing
runtime architecture or spending machine time deciding what to measure.

Status · PRE-REGISTERED / NO HOST RUN. This document is a runbook and prediction
record, not a qualification result. Product and common-control lanes are separate.

## Current software contract

The source/profile checkpoint must be clean and fingerprinted before a host run. The
current known-text server contract is:

| lane | requested decoder | resolved expectation | product meaning |
|---|---|---|---|
| `vnni-product` | `QWEN_DECODER_BATCH=1`, INT8 VNNI | `per-item-int8-vnni`, `VALID FALLBACK` | best current VNNI implementation; not AMX ragged parity |
| `common-control` | `QWEN_DECODER_BATCH=0`, INT8 | per-item INT8 leaf on the target ISA | same server/decoder shape across ISAs; no AMX/KAI-only path |

Both lanes explicitly use known-text `QWEN_TTS_STREAM_LAYOUT=1`, q4, prefix cache,
engine-owned pool, synchronous output, `--max-queue 0`, threshold 2 and the same
playback metric definitions. The VNNI product lane disables AMX/Design-D/fused-only
features and pins FP32/BLAS prefill; the common lane disables AMX/KAI-only execution.
The product lane and common-control lane must never be pooled into one result.

The profile preflight is authoritative. A caps string or a shared function name is not
evidence that the selected decoder leaf ran.

## Phase 0 — host identity and gate

Run on the real host, after any reboot, before loading a model:

```bash
git rev-parse HEAD
git status --short
sha256sum ./qwen_tts
make cpu-check
make bench-fingerprint
make doctor DOCTOR_ARGS="--full"
./qwen_tts --caps
./qwen_tts --self-test
./qwen_tts --dispatch-map
```

Record the full outputs and stop if any of these are false:

* 16 physical cores are online and the SMT state is explicit (prefer SMT off for the
  reference comparison);
* CPU model, ISA, CCD/L3 domains, NUMA and governor are known;
* no cgroup CPU quota or survivor server/benchmark exists;
* source and binary are clean/fingerprinted;
* self-test and dispatch gate pass.

Use these additional Linux checks where available:

```bash
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
cat /sys/devices/system/cpu/online
cat /sys/devices/system/cpu/smt/control
numactl --hardware
cpupower frequency-info
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
```

Do not infer CCDs from the instance name. Group CPUs from `lscpu` and the shared
`index3/shared_cpu_list` cache files. The campaign masks are derived from that result.

## Phase 1 — per-mask roofs

Measure actual masks with the repository roof tool, not a divided full-host number:

```bash
make roofs ROOF_MASKS=<all-16>,<ccd0-8>,<ccd1-8>,<representative-4>
```

`<all-16>` is the real online mask, `<ccd0-8>` and `<ccd1-8>` are the observed LLC/
CCD domains, and `<representative-4>` is a contiguous or domain-local four-core mask
chosen from the fingerprint. Record read/copy/triad curves, saturation knee, and any
asymmetry. `make doctor` is a PREDICTION ONLY; its recommended topology is not scored.

## Phase 2 — build and strict dispatch

Build the actual VNNI target cleanly:

```bash
make clean
make SIMD=avx512vnni blas
sha256sum ./qwen_tts
tools/serving_profile.py preflight vnni-product --binary ./qwen_tts \
  --out RUN/profile-vnni-product.json
tools/serving_profile.py preflight common-control --binary ./qwen_tts \
  --out RUN/profile-common-control.json
```

If the host/compiler requires the AVX-512 BF16 build for the chosen VNNI prefill
policy, build and fingerprint that binary separately; do not silently substitute it
for `SIMD=avx512vnni`. A requested invalid fallback is a STOP, not a usable data point.

The preflight JSON must be embedded in every WAVE/SOAK result. Record the exact
`resolved_decoder_mode`, decoder precision, Talker/CP/Q4/prefill family, and fallback
status. On the VNNI product lane, `VALID FALLBACK` means per-item VNNI, not ragged
Design-D.

## Phase 3 — cheap chain/primitive pre-screen

Before paid serving waves, use existing tools and, if practical, one real-model short
probe to estimate but not qualify:

```bash
make roof-matvec ROOF_THREADS=4
make roof-matvec ROOF_THREADS=8
make roof-matvec ROOF_THREADS=16
./qwen_tts --matmat-bench
```

The existing roof and matmat commands are shape/capability probes. They are not a
Doctor-v2 replacement. If a real-model microprobe is available without invasive fake
model machinery, collect `T_step(B,K)` and `D(B,K)` for `B=1,2,3,4` and `K=4,8,16`
with the actual VNNI decoder leaf, including preparation and epilogue. Store:

```text
T_step(B,K) = Talker + CP region and its existing rendezvous, without decoder
D(B,K)       = decoder preparation + kernel + epilogue for the real q4 shape
prediction   = q * T_step(B,K) + D(B,K)
```

Label every result MEASURED or PREDICTED. Do not turn the equation into a capacity
claim. The only pre-registered gate for a 0.6B `4x4` arm is:

```text
real 0.6B B=1, K=4 chain wall <= 50 ms, with no unresolved fallback or quota issue
```

If it fails, do not rent time for 0.6B `4x4`.

## Phase 4 — topology and concurrency screen

Use `tests/serve_parallel_wave.py` with the strict profile, identical bank/seed/voice/
language/settings, three short waves per cell, and no profiler/census. Explicitly
record any `--batch-cap` override. Start with:

### 1.7B

* `2x8` and `1x16`;
* C1, C2, C3 and C4;
* do not continue to C5/C6 unless C4 is genuinely GOOD.

### 0.6B

* `2x8` at C1, C3, C4, then C5 and C6 while the previous point remains GOOD;
* test `4x4` only if the pre-registered 4-thread chain gate passes;
* test C8 only if C6 is comfortably GOOD and has no cadence warning.

The exact command family is:

```bash
python3 tests/serve_parallel_wave.py \
  --model <MODEL_DIR> --bin ./qwen_tts --profile vnni-product \
  --topo 2x8,1x16 --conc 1,2,3,4 --waves 3 \
  --text-file <CANONICAL_BANK> --classes short,medium,long,mixed \
  --speaker ryan --language English --seed 42 --precision int8 \
  --out RUN/vnni-17b-screen --port 9500
```

Use `--profile common-control` as a separate arm on the same host only when the
question is semantic/common-shape comparison. Do not use a common-control result as
the VNNI product capacity result.

Every cell reports at least TTFB/TTFA, STREAM/TOTAL RTF, required prebuffer,
safe-play-start, max gap, fixed-buffer stalls at 100/250/500/1000 ms, req/s,
effective B, errors/rejects/timeouts and receive coalescing. CPU/core-equivalent,
context switches and iteration/decoder B walls are supporting diagnostics, not a
replacement for playback gates.

## Promotion and stop rules

Use the existing full-envelope semantics:

* `STREAM_RTF` p95 < 1 is mandatory; <= .90 preferred;
* required prebuffer p95 <=500 ms acceptable, <=300 ms strong;
* stall@500 must be zero or effectively zero for a serious point;
* stall@250 should be zero at a preferred point;
* safe-play-start preferably <=800–1000 ms;
* no hidden admission backlog, unexpected fallback, internal error or quality failure;
* receive coalescing must be small; a large share makes cadence evidence non-citation-
  grade.

Stop a topology at the first clear failure rather than running a SOAK on it. For 1.7B,
the first not-good C4 means no C5/C6. For 0.6B, continue only while the previous
concurrency has useful margin. A screen-only point is never an economic denominator.

Predictions recorded before Turin measurement:

* Fable/judge hypothesis: 1.7B `2x8` can reach full-envelope C3; C4 is an important
  screen point.
* Fable/judge hypothesis: 0.6B `2x8` can plausibly reach C5/C6; `4x4` is worth one
  screen only under the 4-thread chain gate.
* Independent review: these are plausible but sensitive to per-CCD L3/bandwidth and
  the VNNI per-item decoder path; no probability or acceptance is assigned.

## Phase 5 — selected-candidate service tests

Select at most one topology/concurrency per model for paid long tests. Then run:

1. short, medium, long and mixed class waves;
2. long-arrival interference (`2+1` for a C2 candidate, `3+1` for C3, etc.);
3. N+1 fail-fast overload with `--max-queue 0`;
4. one five-minute closed-loop SOAK;
5. short Poisson/open-arrival transition with accepted and intentional-503 populations;
6. transport/coalescing and current-generation structural/audio gate.

Do not run two-hour qualification on every screen point. Do not enable diagnostic
stage tracing in KPI arms.

## Phase 6 — first-failure diagnostic

At the first meaningful failure only, run a separate non-qualifying diagnostic arm
with `QWEN_STAGE_TRACE=1 QWEN_TTFA_TRACE=1` and preserve server stderr plus request
JSONL. The stage trace is default-off and has monotonic absolute iteration bounds.
Summarize it with:

```bash
python3 tools/stage_pressure.py <SERVER_STDERR> \
  --client-jsonl <REQUESTS_JSONL> --gap-ms 250 --json > RUN/stage-pressure.json
```

This reports overlap between client-observed gaps and coarse admission/prefill/head/
sample/CP/decode/Talker/output/serial intervals, decoder group geometry and per-item
versus ragged use. It is diagnostic overlap, not automatic causality. If DRAM pressure
is needed, use a separate low-rate `perf stat` arm and label it NON-QUALIFYING.

Interpretation is pre-registered:

* Talker/CP near the measured per-mask roof with pool busy → PHYSICAL pressure;
* decoder/serial windows with material idle/pool-wait share and memory below roof →
  ENGINE pressure;
* both → MIXED;
* no phase claim without timestamps that actually overlap the receive gap.

Do not implement the partitioned step/decoder team from this evidence. It is a future
falsifier only if the arithmetic fits and the second host reproduces the ENGINE/MIXED
signature.

## Economic result schema

For each host/model record:

| host | CPU/ISA | physical cores | topology | memory roof | profile/decoder leaf | screen GOOD | full-envelope GOOD | first NOT GOOD | TTFA p95 | STREAM p95 | prebuffer p95 | safe-start p95 | max-gap p95 | stall250 | stall500 | req/s | hourly cost | cost / GOOD stream-hour |
|---|---|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GCP 8c AMX | Emerald Rapids / AMX | 8 | 1x8 | ~110 GB/s | amx-product / Design-D | C3 | C2 (1.7B), C3 (0.6B) | C3 / C4 | transferred | transferred | transferred | transferred | transferred | transferred | transferred | transferred | UNKNOWN | UNKNOWN |
| AMD Turin | EPYC / VNNI | 16 | measured | measured | vnni-product / per-item VNNI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | capture-dated | TBD |

Cost per good stream-hour is `hourly_cost / highest_full_envelope_GOOD`; never use a
screen-only concurrency or raw req/s as the denominator. Historical AWS prices in older
documents are transferred context until a dated price is attached to the actual host.

## Paid-machine budget estimate

These are planning estimates, not measured durations:

* preflight/build/roofs/strict dispatch: 10–20 minutes;
* primitive or real-model chain probe: 10–20 minutes;
* two-topology screen for one model: 25–45 minutes;
* both-model screens plus optional 4x4: 45–75 minutes;
* one SOAK per selected model and short open-arrival probes: 30–60 minutes.

First useful topology/capacity verdict: roughly 60–90 minutes after a healthy SSH host
is available. Full selected-point campaign if the screen earns GO: roughly 2–3 hours.
Stop early on dispatch mismatch, dirty identity, quota/SMT failure, coalesced KPI
artifacts, or an obviously bad first concurrency.

## Final local gate before requesting AWS

The repository must have:

* clean source and explicit profile semantics;
* `make doctor` still labelled prediction-only;
* strict VNNI and common-control preflight paths;
* stage-pressure trace/helper default-off and tested;
* no profile that silently reintroduces `QWEN_TTS_STREAM_LAYOUT=0`;
* PLAN pointing here, with no claim that QL-2 hardware comparison is closed.

No AWS instance is requested by this plan. The next action after this checkpoint is
to provide a 16-physical-core AMD Turin/VNNI host and run Phase 0 exactly once.
