# Task · Question

Task: cross-ISA serving parity audit before the hardware campaign.

Question: can the current streaming serving contract be compared across x86 AMX,
x86 AVX-512/VNNI-only, and Arm Neoverse-V2/KleidiAI without confusing common
server semantics with an ISA-specific decoder implementation?

## Known facts

- Source checkpoint: `ee0c417cd9870a27d86ad1df0cefbf9a1067eac2`.
- Branch: `feature/x86-amx-vnni-oss`.
- The checkout was clean at the start of this audit.
- Current AMX serving reference: 1.7B INT8, 2x6 prefork on 12 physical cores,
  cap2, `--max-queue 0`, q4 steady-state decoder quantum, ragged threshold 2,
  engine-owned decoder pool, synchronous output, warm strip, Design-D INT8,
  fused residual enabled for the AMX arm, prefix cache on.
- F3 rejected natural cross-worker global batching; permanent cap3 and the
  utilization-aware temporary B3 policy were rejected. They are not reopened here.

## Unknowns

- No new binary was built and no new host was started or benchmarked in this audit.
- Neoverse-V2 runtime dispatch, numerical/audio quality, and serving performance
  remain unmeasured at this checkpoint.
- A VNNI/Arm implementation equivalent to the AMX Design-D ragged decoder path does
  not exist at this HEAD; its performance and whether it is worth implementing are
  separate questions.

## Files/functions inspected

- `PLAN.md`, `ENGINEERING.md`, `docs/BENCHMARKING.md`,
  `docs/cross-backend-audit-2026-09-05.md`, `docs/reference-gcp-c4-standard-24.md`,
  `docs/runtime-map-c8a-c4.md`, `docs/feature-flags.md`.
- Server: `qwen_tts_server.c` (`server_default_decoder_batch`, `sink_next_job`,
  async output queue, prefork admission), `qwen_tts.c` (continuous loop, q4
  quantum, engine-owned budget, prefix/stream layout).
- Decoder: `qwen_tts_speech_decoder.c` (`sd_int8_enabled`, Design-D/fused gates,
  `sd_stream_batch_body`, `sd_batch_fallback`, persistent AMX packs).
- Dispatch/kernels: `qwen_tts_dispatch.c`, `qwen_tts_kernels.c` (capability gates,
  AMX/VNNI/Arm dispatch, region runner and quantization contracts).
- Arm backend: `qwen_tts_kleidi.c/.h` (runtime HWCAP, Q4/I8/BF16 dispatch,
  persistent RHS and per-call LHS packing).

## Evidence

This is a source audit. Existing `.work/` results are used only where they describe
the current serving contract; no old benchmark number is promoted as a new result.
The central code fact is:

```c
sd_batch_amx = sd_amx_d_enabled() || sd_amx_bf16_enabled();
if (nb == 1 || !sd_exact_stream_enabled() ||
    (sd_int8_enabled() && !sd_batch_amx))
    return sd_batch_fallback(...);
```

Therefore server-level decoder batching is a common scheduling request, but the
full ragged decoder workset is not a common backend implementation.

## Conclusion

The common server semantics are portable enough for capability-normalized baseline
runs. A direct comparison of the current AMX accelerated reference against VNNI or
Arm is not yet apples-to-apples: Design-D, fused residual, and the corresponding
ragged decoder execution are AMX-specific. The campaign must use an explicit common
control lane plus a separately labelled best-per-ISA lane, with requested/resolved
dispatch recorded from the measured process.

## Next action

Freeze the contract below. Before renting hardware, make the benchmark profiles
reject an explicit AMX-only request that resolves to a fallback and require an
ISA-specific quality/dispatch preflight. Do not add another AMX optimization to the
current box solely to improve the cross-ISA headline.

# 1. Repository truth and architecture-freeze status

| item | current value |
|---|---|
| HEAD | `ee0c417cd9870a27d86ad1df0cefbf9a1067eac2` |
| branch | `feature/x86-amx-vnni-oss` |
| worktree | clean at audit start |
| PLAN | P0/P1/P2 closed; P3/P4/P5 falsifiers recorded; P6 QL-1 qualification and QL-2 backend comparison remain open |
| current product reference | cap2/q4/fused residual/2x6, with cap2 fail-fast and no global batching |
| current AMX lane | Design-D INT8 with persistent B packs, warm strip, engine pool, ragged threshold 2 |
| output reference | synchronous PCM output; async output is implemented but default-off |

The architecture is sufficiently frozen for a cross-host **baseline and parity
campaign**. It is not correct to describe it as one identical low-level backend on
all three ISAs. QL-2 needs a capability-normalized contract before its results can
be called a fair AMX/VNNI/Arm comparison.

Current reference environment, written as requested settings rather than inferred
defaults, is:

```text
--prefork 2 --prefork-threads 6 --batch-size 2 --max-queue 0
QWEN_SD_INT8=1
QWEN_SD_AMX_D=1                 # AMX lane only
QWEN_SD_AMX_BF16=0
QWEN_SD_STREAM_STRIP=1          # warm strip; AMX lane
QWEN_SD_FUSED_RESIDUAL=1        # effective only with AMX Design-D
QWEN_SD_RAG_MIN_PANELS=2
QWEN_SD_POOL=engine
QWEN_BLAS_OWN=1
QWEN_DECODER_BATCH=1
QWEN_STREAM_DECODE_CHUNK=4      # startup still uses the existing 1/2/4 ramp
QWEN_STREAM_DECODE_CHUNK_BUSY=0
QWEN_SERVER_ASYNC_OUTPUT=0
QWEN_PREFIX_CACHE=1
QWEN_PREFILL_MATMAT=1
QWEN_CP_PREFILL2=1
QWEN_POOL_SPIN=4096             # reference value, not a portable default
```

`q4` here is the streaming decoder quantum, not Q4_0 model-weight quantization.
The AMX flags must not be copied to a non-AMX profile: an explicit request that
resolves to a fallback is a benchmark abort under `ENGINEERING.md`.

# 2. Parity matrix

Classification used below:

- **A/common** — server semantics and state machine are shared.
- **B/common design, backend implementation** — same operation contract, different
  kernel/packing/threshold implementation.
- **C/ISA-specific** — the current code intentionally exists only on one ISA family.

| feature | class / current state | x86 AMX | x86 AVX-512/VNNI-only | Arm Neoverse-V2/KleidiAI | parity decision |
|---|---|---|---|---|---|
| Prefork topology and cap semantics | A; implemented; Linux server behavior | 2x6/cap2 reference | same semantics; topology must match physical cores, not vCPU labels | same Linux semantics | **SAFE / TUNE PER ISA** |
| `--max-queue 0` fail-fast | A; implemented | parent returns immediate overload response at full cap | same | same on Linux | **SAFE unchanged**; required for comparable overload behavior |
| Header timing / TTFB boundary | A; implemented; synchronous reference sends header at admission | same | same | same | **SAFE unchanged** |
| Async bounded output queue | A; implemented, default-off | not reference | not reference | not reference | **SAFE unchanged**, qualify only as a separate output arm |
| q4 decoder quantum | A; implemented in common continuous loop | same 1/2/4 startup ramp and q4 steady state | same logical calls | same logical calls | **SAFE semantics**; wall/cadence must be requalified per ISA |
| Ragged decoder batching request | A at server API, B in decoder | full ragged workset when Design-D/BF16 path is enabled | `QWEN_DECODER_BATCH=1` resolves to per-item decoder fallback for INT8 | same when decoder INT8 is enabled; KAI is not the decoder ragged runner | **BLOCKER for direct accelerated parity** |
| Design-D INT8 decoder | C-AMX; opt-in | persistent AMX B packs + TDPBSSD | no Design-D path; regular VNNI decoder path | no Design-D path; dotprod decoder path | **AMX-only; separate lane** |
| Fused residual | C-AMX; default-off | effective only when AMX-D and its pack are available | flag is a no-op/falls back to ordinary projection + residual | same | **AMX-only; never claim VNNI/Arm support from the flag** |
| Generic persistent weight prepack | B/C; backend-specific opt-in/registry | AMX tile layouts and Design-D packs | selected VNNI packed RHS/row-sum caches | KAI persistent RHS registry for Q4/I8/BF16 | **SAFE only with resolved leaf; tune per ISA** |
| Decoder INT8 | B; VNNI default-on, Arm opt-in | Design-D if explicitly enabled | VNNI decoder kernels | DOTPROD decoder kernels, default-off pending Arm qualification | **SAFE baseline with explicit resolved policy** |
| Engine-owned decoder pool | A/B; server enables engine budget | pthread pool, engine-owned decoder work | same pthread design | same expected on Linux pthread; spin/threshold defaults differ | **SAFE semantics / TUNE PER ISA** |
| Ragged threshold 2 | A policy knob, default/profile setting | measured AMX reference | same semantics, different kernel cost | same semantics, different KAI/fallback cost | **SAFE semantics / requalify value** |
| Prefix cache | A; default-on | common FP32 K/V cache and keying | same | same | **SAFE unchanged**; memory/pre-fill wall still measured |
| CP prefill2 | C-ish backend gate; VNNI default-on, Arm opt-in | may use VNNI-compatible x86 path in AMX build | available when CP layers satisfy int8/int4 requirements | no default; explicit opt-in requires dispatch/quality gate | **FIX BEFORE BENCH**: pin requested/resolved policy |
| Talker prefill matmat | B | AMX BF16 if compiled/selected, otherwise fallback | VNNI-only build may use f32/SGEMM; AVX512BF16 build is a different lane | BFMMLA/KAI BF16 when compiled, otherwise fallback | **FIX BEFORE BENCH**: never compare unresolved precision paths |
| SL-1 known-text streaming layout | A; implemented default-off | common prompt/step code | same | same source semantics | **SAFE functionally; quality/ICL clone remains UNKNOWN** |
| `[TTFA]`, `[F2]`, `[ITER]` traces | A diagnostics; default-off | same clock/meaning | same | same on Linux | **SAFE diagnostics**, never KPI qualification evidence |
| Playback metric harness | A; Python/client-observed | same definitions | same | same | **SAFE unchanged**; report coalesced-read share |
| Realistic WAVE/SOAK harness | A; canonical tools in `docs/BENCHMARKING.md` | same | same | same | **SAFE unchanged** after identity/dispatch gate |
| 4+1 admission probe | A workload shape, C implementation detail | LS-4 is Linux-prefork diagnostic and rejected | same diagnostic code if Linux | same Linux workload semantics | **Use as workload only; do not call LS-4 promoted** |
| `QWEN_POOL_SPIN`, narrow/priority | B pthread runtime | 4096 reference | 4096 is not portable by proof | Arm64 default is 65536; priority/implementation differs by OS | **TUNE PER ISA**, never copy values |
| AMX/VNNI/KAI min-B, N-chunk, row gates | C/B backend tuning | AMX B/row/column gates and tile chunks | VNNI B>=2 and VNNI pack/tile gates | KAI Q4/I8/BF16 gates and i8mm/dotprod | **TUNE PER ISA; dispatch must be captured** |

# 3. x86 AVX-512/VNNI audit

## Server semantics

The VNNI binary follows the same continuous server loop, prefork parent, cap,
fail-fast, q4 quantum, prefix cache, engine-budget request, synchronous output and
playback metric definitions. `qwen_exec_budget_engine_owned("serve")` is called by
the batched and prefork server entry points. The server does not change its causal
state machine based on AMX availability.

The q4 setting is therefore safe to carry as a logical policy. It does not make
VNNI execute AMX-shaped decoder work; it only selects the same complete decoder-call
boundaries. q1/q2/q4/q8 are complete calls, not intra-call preemption.

## Actual kernel differences

- Generic Talker/CP `qwen_matmat_int8` dispatches AMX first on an AMX build and
  VNNI on a VNNI build. AMX INT8 defaults to B>=3 plus row/column gates; VNNI
  defaults to B>=2. B=1 remains a matvec/small-M path.
- Generic Q4_0 matmat dispatches AMX at its AMX gate or VNNI at its VNNI gate.
  The source Q4 representation is shared, but VNNI expands/quantizes activations
  differently from the AMX path. This is not the same as the q4 streaming quantum.
- VNNI has persistent/general packed-RHS support for selected projections, but it
  is not the decoder's AMX Design-D B-pack representation.
- CP/Talker region runners are x86 VNNI-family code. They may be present in an AMX
  build when their shape gate selects them, but Arm does not enter those regions.
- The decoder batch entry explicitly falls back to `sd_batch_fallback()` on an
  INT8 non-AMX build. That function loops over items and calls the complete
  per-stream decoder. Consequently `QWEN_DECODER_BATCH=1` in the VNNI log is not
  proof of one ragged decoder pass.
- `QWEN_SD_FUSED_RESIDUAL=1` does not activate a VNNI fused kernel: the actual
  helper additionally requires `sd_amx_d_enabled()` and an AMX-D pack, so VNNI
  executes the ordinary projection and residual add.

## VNNI verdict

**Safe to benchmark as a VNNI baseline, conditionally.** Build a clean explicit
VNNI binary, disable AMX-only requests, and verify `[FLAGS]`, `--dispatch-map`, and
the actual decoder leaf before using a result. It is **not safe to call it a direct
parity run against the current AMX Design-D/fused decoder reference**. The report
must say whether the decoder is per-item fallback and must not assign AMX decoder
coverage to it.

The old AWS C8a/older VNNI notes are not a current control: they contain historical
2x8/SMT and q8/batch assumptions, and some were corrected after a dispatch was
generated before the serving environment was applied. They can supply hypotheses
only. Do not port `QWEN_POOL_SPIN`, topology, batch cap, q4 policy or prefill
precision from those notes without a fresh profile and dispatch proof.

# 4. Arm Neoverse-V2 / KleidiAI audit

## Actual backend

`qwen_tts_kleidi.c` is built only when AArch64 i8mm and dot-product compile
features are present. On Linux, `qwen_kleidi_supported()` additionally checks
ASIMDDP and I8MM HWCAP bits. Neoverse-V2 is therefore a valid target only after
the preflight proves both compile and runtime support.

- KAI Q4 uses a dot-product GEMV kernel at B=1 and an i8mm GEMM kernel at B>1;
  Q4 RHS is persistently registered and LHS is packed/quantized per call.
- KAI INT8 uses its GEMV/GEMM family based on B and also keeps persistent RHS
  registrations. The normal dispatcher tries KAI before the in-house Arm paths.
- KAI BF16 is a separate family and has its own operation/packing gates.
- The current in-region CP/Talker `qwen_region_i8_*` path is guarded by the x86
  VNNI compilation condition; `qwen_region_i8_usable()` is not an Arm decoder
  batching implementation. Arm may use ordinary KAI dispatcher calls but not the
  current x86 held-team region fast path.
- The speech decoder's int8 kernels are Arm DOTPROD code, not KAI decoder Design-D.
  `QWEN_SD_INT8` is off by default on Arm because its first-frame cost was not
  qualified there. Turning it on is an explicit policy choice, not an inferred
  parity fact.
- The common server decoder-batch flag therefore still falls back per item for
  Arm INT8, just as it does for VNNI non-AMX. The server remains functionally
  correct, but its decoder geometry and weight reuse are different.

## Numerical and policy caveats

The x86 and Arm SIMD activation quantizers have different tie rounding contracts
(x86 ties-away, Arm ties-to-even). KAI also has its own LHS pack/quantization mode
(`QWEN_KAI_LHS`, asymmetric by default). Existing code explicitly documents that
audio was qualified under the native per-platform contracts, not bit identity.
`QWEN_SD_INT8_BLK` also defaults to 256 in an AVX512VNNI build and 64 on the other
builds unless pinned. These differences do not imply incorrectness, but they make
quality/audio gates mandatory before treating RTF differences as hardware results.

## Arm verdict

**Safe to benchmark as an Arm semantic/kernel baseline, conditionally.** Require a
clean Neoverse-V2 KAI build, HWCAP dispatch proof, explicit decoder precision choice,
and audio/quality gates. It is **not safe to claim current-AMX decoder parity**:
Design-D, fused residual, x86 regions and the AMX ragged decoder pass have no Arm
equivalent at this HEAD. KAI's persistent RHS is a useful Arm analogue for generic
Talker/CP/Q4 work, but not evidence that the decoder path is equivalent.

# 5. Blocking findings

## BLOCKER — direct accelerated-lane comparison is invalid as written

The AMX reference's most visible decoder features are not shared:

1. Design-D INT8 uses persistent AMX B packs and TDPBSSD.
2. Fused residual is entered only through Design-D AMX.
3. The ragged decoder batch path is retained only when AMX-D or decoder BF16 AMX
   is enabled; VNNI/Arm INT8 calls are per-item fallback.

Thus “AMX vs VNNI vs Arm with the same `QWEN_DECODER_BATCH=1`” would compare an
AMX fused ragged decoder against two different algorithms. It can still be a valid
**best-per-ISA product comparison**, but it is not a kernel/ISA parity comparison.
The campaign must publish both a common-control lane and an accelerated lane, or
implement and qualify a portable ragged decoder path before claiming direct parity.

## FIX BEFORE BENCH — explicit resolved profiles

The following must be pinned in the campaign profile and checked after environment
application:

- `QWEN_SD_INT8` (VNNI default-on, Arm default-off);
- `QWEN_SD_AMX_D`, `QWEN_SD_FUSED_RESIDUAL`, `QWEN_SD_STREAM_STRIP`;
- `QWEN_DECODER_BATCH` versus actual decoder leaf;
- `QWEN_CP_PREFILL2`;
- `QWEN_PREFILL_MATMAT` and the selected native/f32 prefill path;
- `QWEN_SD_INT8_BLK`, `QWEN_POOL_SPIN`, and backend min-B/N-chunk gates.

An explicit flag resolving to a fallback is an abort, not a datapoint for the
requested feature. An `auto` path may be reported only with its resolved leaf.

## FIX BEFORE BENCH — quality contract

Cross-ISA outputs need not be bit-identical because quantization rounding and KAI
LHS packing differ. The same bank, seed, model and request settings must still pass
the existing audio/quality gate per ISA before throughput is compared. A performance
number from a lower-quality or FP32-fallback path is not a valid backend win.

## SAFE / TUNE PER ISA

Prefork, cap/fail-fast, transport timing, q4 call boundaries, prefix cache, engine
pool ownership, playback definitions and the WAVE/SOAK lifecycle are safe common
semantics on Linux. Topology, worker width, pool spin, backend gates, packing and
possibly q-min may be tuned per ISA only in explicitly requalified profiles.

## UNKNOWN

- KAI's real C4 decoder/serving margin on Neoverse-V2.
- Whether a portable ragged decoder implementation would be worth its cost.
- Whether a VNNI non-AMX decoder can reach the current AMX reference envelope.
- Whether Arm's KAI LHS mode and block choice should be aligned for quality or left
  native; source inspection cannot choose that policy.

# 6. PLAN audit

The current PLAN correctly keeps QL-1 and QL-2 open and does not reopen rejected
cap3/global-batching/LS-4 work. The missing item is an explicit gate that prevents
QL-2 from presenting the AMX Design-D reference as if it were common backend work.

The audit therefore adds a small completed parity checkpoint (`QL-2a`) and changes
QL-2 to consume this contract. No runtime behavior or optimization task is added.
The existing P3/P4 structural decoder item remains open as architecture work, but it
is not a reason to keep optimizing the AMX box before the first controlled campaign.

# 7. Frozen ISA-neutral benchmark contract

## Must remain identical

- Model revision, model size, stored quantization and voice assets.
- Text/request bank, language/speaker/instruct settings, seeds and request order.
- C1/C2/C4 true-wave workload and the deterministic 4+1 admission workload.
- Output sample format, audio callback semantics, synchronous output for the first
  contract, and `--max-queue 0` fail-fast semantics.
- Prefix-cache policy, decoder quantum policy (`q4` anchor), ragged policy, and
  server batch/cap semantics in the common-control lane.
- Client metric definitions: TTFB, TTFA, STREAM_RTF, TOTAL_RTF, required_prebuffer,
  safe_play_start, max_gap, fixed-buffer stalls, total stall time and coalesced-read
  share. Client marks remain upper bounds on lateness when reads coalesce.
- Wave/soak lifecycle, warmup, duration, seed discipline, error/reject/timeout
  accounting, survivor cleanup and artifact identity.

## Two required lanes

**Common-control lane.** Disable AMX-only decoder requests and use the same server
policy on all hosts. For a strict algorithm control, either force the common
per-item decoder path (`QWEN_DECODER_BATCH=0`) or prove that every backend resolves
to the same ragged implementation; leaving `QWEN_DECODER_BATCH=1` set while AMX
alone takes the ragged path is not a control. Record the resolved decoder
implementation in either case. This lane answers semantic serving and cost-envelope
questions.

**Best-per-ISA lane.** Permit AMX Design-D/fused residual, VNNI/KAI native generic
matmat, and Arm-specific decoder policy only when each dispatch is proven. Report
AMX Design-D as an AMX feature result, not as an ISA-neutral improvement. The two
lanes must not be collapsed into one table without the lane label.

## Preflight and qualification rules

Every host runs, in order, `make cpu-check`, topology/NUMA/SMT verification,
cache and measured bandwidth curves, an explicit clean SIMD build, `--caps`,
`--self-test`, `--dispatch-map` under the final environment, and a single-request
sanity check. The process's `[FLAGS]` plus a diagnostic census/path counter prove
the actual leaf. No intrusive census/profiler is enabled in KPI arms.

Use WAVE for C1/C2/C4 screening and only then the canonical five-minute SOAK for
the chosen operating point. Report the complete playback envelope, not bare RTF.
If coalesced-read share is unexpectedly high, label cadence evidence non-citation-
grade. No old C8a/Arm/AMX profile is a control unless its source, binary, env and
dispatch identity exactly matches the stated lane.

# 8. Per-ISA tuning allowances

Allowed after the common control is established:

- physical-core topology, worker count/threads, CPU masks, SMT and NUMA policy;
- pool spin/narrow/priority knobs and OpenBLAS ownership where the platform
  supports the same semantics;
- AMX/VNNI/KAI minimum-B, N-chunk, tile and rows-per-thread gates;
- persistent packing layout and backend-specific cache preparation;
- `QWEN_SD_INT8`/block policy and `QWEN_CP_PREFILL2`, provided the resolved path and
  quality gate are recorded;
- q-min/decoder quantum only after a fresh playback-aware requalification.

Not allowed to vary silently: model/quantization, request bank/seeds, admission
semantics, metric definitions, output mode, or a feature requested as enabled but
resolved to fallback.

# 9. Hardware campaign order (no instances started by this audit)

Recommended first three new probes, chosen to cover one lower-cost same-family
AMX point and the two non-AMX ISA families:

1. **GCP C4 highcpu-16 / 8 physical AMX cores** — same Emerald Rapids family and
   software contract; tests cost/scaling before attributing a result to a new ISA.
2. **One AMD Turin VNNI slot** — AWS C8a.4xlarge or GCP C4D equivalent, selected
   by availability; establishes the x86 non-AMX control with explicit VNNI/decoder
   fallback labels.
3. **GCP Axion 16-core Neoverse-V2** — verifies KAI i8mm/Q4 and Arm decoder policy
   under the same semantic lane.

Run AWS C8i.4xlarge and C8id.4xlarge Xeon-6 AMX after the control lanes are clean;
the latter is specifically useful for separating memory-bandwidth effects from
instruction-set effects. Add Graviton4 only after Axion KAI parity is established.
Vendor bandwidth claims are hypotheses, not model capacity numbers.

Every slot first receives the preflight and bandwidth/matvec/matmat screen; no TTS
serving run starts on a host whose dispatch or quality contract is unresolved.

# 10. STOP / GO verdict

- **VNNI:** GO for a labelled semantic/common-control baseline after clean-build
  and dispatch/quality preflight; NO-GO for an unqualified direct comparison to
  AMX Design-D/fused ragged decoder.
- **Arm/KleidiAI:** GO for a labelled semantic/KAI baseline after Neoverse-V2
  HWCAP/build/quality preflight; NO-GO for claiming current-AMX decoder parity.
- **Blockers:** resolve the benchmark-lane distinction and explicit dispatch/quality
  gates before spending on hardware. No large runtime fix is required merely to run
  the baseline lanes; a portable ragged decoder is required only if the product
  question demands identical accelerated decoder algorithms.
- **Current AMX box:** do not give it another optimization cycle before the campaign.
  Freeze cap2/q4/fused/2x6 as the AMX product reference, keep the common-control
  lane available, and use the measured cross-ISA audit to prevent false parity.
