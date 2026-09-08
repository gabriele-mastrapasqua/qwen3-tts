# Task · Question

Task: make the cross-ISA serving comparison operational before renting hardware.

Question: can AMX, x86 VNNI and Arm/KleidiAI runs be rejected or accepted from the
measured process's resolved dispatch, without confusing common server semantics with
an ISA-specific decoder implementation?

## Known facts

- The source branch is `feature/x86-amx-vnni-oss`; the audit started from the clean
  parity-audit descendant `8c0ac04` (the earlier audit checkpoint is `851fef6`).
- No cloud instance was started and no hardware performance benchmark was run for this
  checkpoint.
- Current AMX serving reference remains cap2/q4/fused-residual/2x6, Design-D INT8,
  ragged threshold 2, engine-owned decoder pool, synchronous output, prefix cache and
  `--max-queue 0` fail-fast.
- Current VNNI and Arm builds do not contain an equivalent Design-D ragged decoder.
  Their product lanes therefore make the per-item INT8 decoder fallback explicit.

## Unknowns

- Target-host AMX, AVX-512/VNNI and Neoverse-V2 HWCAP dispatch have not been executed
  by this checkpoint.
- Per-ISA audio/quality gates, runtime decoder census, throughput, playback envelope and
  topology remain unmeasured.
- A future portable ragged decoder may or may not be worthwhile; this task does not
  implement one.

## Files/functions inspected

- `configs/perf/schema.json`, the four parity profiles, `tools/perf_profile.py` and
  `tools/serving_profile.py`.
- `qwen_tts_dispatch.c`, `qwen_tts_kernels.h`, `qwen_tts_speech_decoder.c`.
- `tests/serve_parallel_wave.py`, `tests/serve_soak.py` and their existing manifest
  conventions.
- `ENGINEERING.md`, `PLAN.md`, `docs/BENCHMARKING.md` and the preceding cross-ISA
  audit `.work/cross-isa-serving-parity-audit-20260908.md`.

## Evidence

### 1. Four explicit lanes

| lane | decoder request and expected resolved leaf | generic backend contract | optional-path policy |
|---|---|---|---|
| `amx-product` | `QWEN_DECODER_BATCH=1` → `ragged-design-d-int8` | AMX Talker/CP/Q4 and AMX prefill where the resolved gate selects them | Design-D and fused residual `ACTIVE`; KAI `UNSUPPORTED` |
| `vnni-product` | `QWEN_DECODER_BATCH=1` → `per-item-int8-vnni` | VNNI Talker/CP/Q4, FP32 prefill control | per-item decoder is an explicit `VALID FALLBACK`; AMX/KAI unsupported |
| `arm-product` | `QWEN_DECODER_BATCH=1` → `per-item-int8-dotprod` | KleidiAI Talker/CP/Q4/BF16 policy | per-item decoder is an explicit `VALID FALLBACK`; KAI active; AMX unsupported |
| `common-control` | `QWEN_DECODER_BATCH=0` → per-item INT8 | same server semantics and per-item decoder shape, native backend per ISA | AMX/KAI-only paths are disabled or allowed only as `UNSUPPORTED`/valid fallback |

The product lanes answer “best currently supported implementation per ISA”. The
common-control lane answers “same server/decoder semantics on different hardware”.
They must not be pooled into one performance result.

### 2. Strict resolved-dispatch gate

`tools/serving_profile.py preflight` runs only `--dispatch-map` and `--caps` under the
merged profile environment; it never starts a server. It emits a compact JSON summary
containing:

- profile/lane/ISA, source and binary identity;
- requested decoder batch, resolved decoder mode and precision;
- actual Talker/CP, prefill and Q4 families;
- `ACTIVE`, `VALID FALLBACK` or `UNSUPPORTED` status for decoder, Design-D, fused
  residual and KAI;
- requested profile flags, forbidden/null flags and quality-gate contract.

An explicit profile mismatch exits non-zero. A requested batch that resolves to the
known per-item VNNI/DOTPROD decoder is accepted only in the product lanes where the
profile declares `VALID FALLBACK`; the AMX product requires the ragged Design-D leaf.
The preflight reports the effective prefill leaf as FP32 when the prefill predicate is
off rather than mistaking a generic BF16 capability probe for the operation that ran.

Both canonical server harnesses run this gate automatically for profiles carrying the
parity contract and embed the resulting JSON in their result/manifest artifacts.
Historical profiles without the contract retain their existing behavior until migrated.

### 3. Explicit controls

The profiles pin the relevant serving and backend controls instead of inheriting them:

- cap/batch/fail-fast semantics, engine decoder pool, synchronous output and prefix
  cache;
- q4 steady-state quantum and startup ramp;
- decoder batch request, decoder INT8/block policy and decoder Design-D/fused/strip
  switches;
- CP precision/prefill policy and prefill matmat policy;
- AMX/VNNI/KAI enable/disable gates and important min-B/N-chunk settings;
- OpenBLAS ownership/timeout, with `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS`
  explicitly required absent where the profile owns the budget;
- KAI-specific LHS/QKV/NCHUNK settings only in the Arm product lane.

Host topology is intentionally `unspecified` in these profiles. Worker count, threads,
CPU masks, NUMA and SMT are measured host facts and must be supplied/requalified by the
campaign rather than copied from the AMX reference.

`make doctor` is a useful first-pass companion after the final build: it measures the
new box's identity/roofs and predicts a starting `W x K`, cap and candidate environment
without a model. Its labels and draft profile are hypotheses, not dispatch or quality
authorization. The campaign must reconcile the draft with a parity profile and still run
the strict resolved-dispatch and quality gates.

### 4. Harness integration

`serve_parallel_wave.py` records `profile_name` and `profile_preflight`; `serve_soak.py`
records the same preflight in `manifest.json`. The profile gate runs after the profile
environment is resolved and before a server process is launched. The result metadata
therefore separates requested settings from the process's resolved dispatch.

### 5. Local validation

Passed on the available Apple Arm development host:

- JSON parse and Python compilation for the changed tools/harnesses;
- all 14 performance profiles through `tools/perf_profile.py validate`;
- `tests/test_perf_profile.py`;
- `tests/test_serving_profile.py`, including invalid AMX/VNNI fallback cases;
- `tools/flag_parity.py --check`;
- native `./qwen_tts --self-test`.

`make blas SIMD=avx512vnni`, `make blas SIMD=amx` and
`make blas SIMD=arm-i8mm-bf16` all completed as compile checks. On Darwin the Makefile
uses `-march=native` for the host, so these are not executable VNNI/AMX/Neoverse proof;
the binary was restored to the native build afterward. Product preflight correctly
rejects the native Apple binary because its resolved ISA/leaf is outside each target
lane. No performance number is claimed.

## Fallback and quality policy

Every benchmark artifact must include the profile name, source/binary identity, host
topology, all profile flags, resolved dispatch JSON and exact decoder mode. The existing
audio/quality gate is mandatory per product lane using the same model, voice, text bank,
seeds, temperature and output format. Bit identity across ISA is not required, but a
wrong decoder precision, unintended generic/f32 fallback, invalid requested path or
quality regression rejects the performance result.

The common-control lane must keep the same `QWEN_DECODER_BATCH=0` logical decoder shape
on all hosts. Product lanes may use their supported accelerated implementation, but their
fallback status must remain visible and their results must not be described as AMX/VNNI/
Arm decoder parity.

## Exact benchmark contract

Hold constant across hosts:

- model size/identity (run 0.6B and 1.7B as separate campaigns), quantization, voice,
  text bank, seeds, temperature and output format;
- server semantics: prefork/cap policy, `--max-queue 0` fail-fast, synchronous output,
  prefix-cache policy, request bank and admission workload;
- q4 quantum policy unless a separately re-qualified ISA profile changes it;
- playback definitions and client receive-fidelity/coalescing reporting;
- C1/C2/C4 waves, the 4+1 admission probe, and fixed-concurrency SOAK/Poisson rules.

Allow per host/ISA only when recorded and requalified:

- worker topology, thread split, CPU masks, SMT and NUMA placement;
- decoder quantum only as an explicit policy experiment;
- kernel/backend target, packing layout, AMX/VNNI/KAI thresholds and NCHUNK values;
- host-specific pool spin and other scheduling knobs.

The campaign sequence remains preflight → bandwidth/topology probe → self-test/quality
gate → C1/C4 screening → playback-aware SOAK. No result is accepted from a dirty or
unresolved binary, and no vendor bandwidth claim substitutes for a measured roof.

## Remaining unknowns and GO/NO-GO

| lane | status before target-host preflight | expected decoder | expected Talker/CP | mandatory gate | blocker remaining |
|---|---|---|---|---|---|
| AMX product | GO to preflight/campaign; not performance-qualified | ragged Design-D INT8 | AMX-resolved | AMX dispatch + audio/quality | target AMX execution and quality/perf still unmeasured |
| VNNI product | GO to preflight/campaign; not performance-qualified | per-item INT8 VNNI | VNNI-resolved | VNNI dispatch + audio/quality | target VNNI execution and quality/perf still unmeasured |
| Arm product | GO to preflight/campaign; not performance-qualified | per-item INT8 DOTPROD | KAI-resolved | HWCAP/KAI dispatch + audio/quality | Neoverse-V2 runtime and quality/perf still unmeasured |
| common control | GO to preflight/campaign; not performance-qualified | per-item INT8 | native per ISA | same shape/quality gate | target builds and quality/perf still unmeasured |

## Conclusion

Operational parity is **GO for the campaign contract**, not a claim that the three
accelerated decoder implementations are equivalent. The current server semantics are
portable; the AMX Design-D/fused ragged decoder remains an AMX product advantage and is
isolated from the common-control result. No cloud benchmark or further AMX optimization
is justified before these gates are used on the candidate hosts.

## Next action

On each candidate host, build the explicit ISA target, run the matching product and
common-control preflights, pass the per-ISA quality gate, then start the smallest C1/C4
screen. Do not start a hardware campaign from a profile name or requested flag alone.
